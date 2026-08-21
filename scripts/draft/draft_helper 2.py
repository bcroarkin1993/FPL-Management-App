import time

import numpy as np
import pandas as pd
import streamlit as st

import config
from scripts.common.error_helpers import show_api_error
from scripts.common.styled_tables import render_styled_table
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_current_gameweek,
    get_fixture_difficulty_grid,
    get_ffp_projections_data,
    get_league_entries,
    get_rotowire_player_projections,
    get_rotowire_season_rankings,
    get_starting_team_composition,
    position_converter,
)
from scripts.common.analytics import compute_early_season_scores, season_progress_weight

# Standard Draft squad composition (confirmed for this league): 15 total.
_STANDARD_SQUAD_SLOTS = {"G": 2, "D": 5, "M": 5, "F": 3}
# GK's small pool (~66 players) makes positional percentile gaps coarser than
# outfield positions, and GK's real-world scoring ceiling/variance is far
# lower (saves + clean sheets vs. goals/assists) — raw VORP overstates GK's
# draft-day scarcity relative to real ADP wisdom. Dampen it; other positions
# keep their full scarcity signal.
_VORP_DAMPENING = {"G": 0.7, "D": 1.0, "M": 1.0, "F": 1.0}
_POS_LABELS = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}
_POS_ORDER = ["G", "D", "M", "F"]
_DEFAULT_NUM_TEAMS = 10
_REFRESH_INTERVAL_SECONDS = 25

_SCORE_COL_FORMATS = {
    "Season": "{:.2f}", "Week1": "{:.2f}", "Fixture": "{:.2f}",
    "VORP": "{:+.2f}", "Score": "{:.2f}", "GW1 Proj": "{:.1f}",
}


def _ensure_session():
    if "draft_taken_keys" not in st.session_state:
        st.session_state.draft_taken_keys = set()
    if "draft_mine_keys" not in st.session_state:
        st.session_state.draft_mine_keys = set()
    if "draft_last_sync_ts" not in st.session_state:
        st.session_state.draft_last_sync_ts = 0.0


def _player_key(row: pd.Series) -> str:
    """Unique key for a player row to persist selection reliably."""
    return f"{row.get('Player','')}|{row.get('Team','')}|{row.get('Position','')}"


def _build_full_player_pool(bootstrap: dict) -> pd.DataFrame:
    """Build the full candidate pool (Player/Team/Position + availability context).

    Names are built the same way (first_name + second_name) as the FPL Draft
    API's own player_mapping (see get_starting_team_composition), so live
    draft picks can be matched against this pool by exact name — both derive
    from the same FPL bootstrap-static element data.
    """
    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}

    rows = []
    for p in elements:
        team_id = p.get("team")
        team_info = teams.get(team_id, {})
        team_short = team_info.get("short_name", "???")
        position = position_converter(p.get("element_type"))
        full_name = f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()

        rows.append({
            "Player_ID": p.get("id"),
            "Player": full_name or p.get("web_name", "Unknown"),
            "Team": team_short,
            "Position": position,
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
            "news": p.get("news", "") or "",
            "total_points": p.get("total_points", 0),
            "form": float(p.get("form", 0) or 0),
        })

    return pd.DataFrame(rows)


def _compute_vorp(scored: pd.DataFrame, num_teams: int) -> pd.DataFrame:
    """Add Replacement Score / VORP / Draft Player Score columns.

    Draft's exclusive, no-share rosters mean cross-position scarcity is real
    (only num_teams x slots_at_position exist leaguewide) in a way Classic
    never has to model. Replacement Score is each position's Player Score at
    the leaguewide replacement-rank cutoff; VORP is the margin above that;
    Draft Player Score is VORP percentile-ranked *across all positions*,
    which is what actually surfaces scarcity — it stops comparing GK-vs-GK
    and starts asking "how much better is this than what's left once the
    league fills up," naturally steeper for thin positions (FWD, then GK)
    than deep ones (MID/DEF). GK's VORP is dampened (see _VORP_DAMPENING)
    since its small pool and low real-world scoring ceiling otherwise
    overstate its scarcity relative to typical draft ADP.
    """
    result = scored.copy()
    result["Replacement Score"] = 0.0
    result["VORP"] = 0.0

    for pos, slots in _STANDARD_SQUAD_SLOTS.items():
        pos_mask = result["Position"] == pos
        pos_df = result.loc[pos_mask].sort_values("Player Score", ascending=False)
        if pos_df.empty:
            continue
        replacement_rank = num_teams * slots
        cutoff_idx = min(replacement_rank, len(pos_df)) - 1
        replacement_score = float(pos_df.iloc[cutoff_idx]["Player Score"])
        result.loc[pos_mask, "Replacement Score"] = replacement_score
        dampener = _VORP_DAMPENING.get(pos, 1.0)
        result.loc[pos_mask, "VORP"] = (result.loc[pos_mask, "Player Score"] - replacement_score) * dampener

    if len(result) > 1:
        result["Draft Player Score"] = result["VORP"].rank(pct=True, method="average")
    else:
        result["Draft Player Score"] = 0.5

    return result


def _names_to_keys(names, pool: pd.DataFrame) -> set:
    """Map live-draft player names (from the FPL Draft API) to board keys.

    Matches by exact Player name against the bootstrap-derived pool — both
    the pool and the Draft API's choices payload build names the same way
    (first_name + second_name from FPL bootstrap-static), so no fuzzy
    matching is needed.
    """
    if not names:
        return set()
    name_set = set(names)
    matched = pool[pool["Player"].isin(name_set)]
    return set((matched["Player"] + "|" + matched["Team"] + "|" + matched["Position"]).tolist())


def _gw1_url_selector() -> str:
    """Manual GW1 Rotowire URL input, matching Initial Squad Optimizer's pattern."""
    url = st.text_input(
        "GW1 Rotowire URL",
        value=config.ROTOWIRE_GW1_URL or "",
        placeholder="https://www.rotowire.com/soccer/article/...",
        help="Rotowire's GW1 'best picks' article isn't auto-discoverable — "
             "paste/update the URL here if it needs to change.",
    )
    return url.strip()


def _positional_needs_section(rankings: pd.DataFrame, mine_keys: set, taken_keys: set, num_teams: int):
    """Personal need vs. leaguewide scarcity, per position."""
    st.markdown("#### Positional Needs & Scarcity")
    st.caption(
        "Your drafted count vs. standard slots, and how much of the leaguewide "
        "pool at each position is already gone. FWD and GK are the scarcest — "
        "no player sharing means only a fixed number of slots exist leaguewide."
    )
    mine_df = rankings[rankings["key"].isin(mine_keys)]
    taken_df = rankings[rankings["key"].isin(taken_keys)]

    cols = st.columns(4)
    for i, pos in enumerate(_POS_ORDER):
        slots = _STANDARD_SQUAD_SLOTS[pos]
        my_count = int((mine_df["Position"] == pos).sum())
        league_slots = num_teams * slots
        league_gone = int((taken_df["Position"] == pos).sum())
        scarcity_pct = min(100, round(100 * league_gone / max(league_slots, 1)))
        accent = "#f87171" if scarcity_pct >= 80 else ("#f0c419" if scarcity_pct >= 50 else "#00ff87")
        with cols[i]:
            st.markdown(
                f'<div style="border:1px solid #333;border-radius:10px;padding:12px;'
                f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
                f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:6px;">{_POS_LABELS[pos]}</div>'
                f'<div style="color:#e0e0e0;font-size:18px;font-weight:700;">{my_count}/{slots} mine</div>'
                f'<div style="color:{accent};font-size:13px;margin-top:4px;">'
                f'{league_gone}/{league_slots} leaguewide gone ({scarcity_pct}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _suggested_picks_tab(rankings: pd.DataFrame, mine_keys: set, taken_keys: set, num_teams: int):
    _positional_needs_section(rankings, mine_keys, taken_keys, num_teams)
    st.markdown("---")

    available = rankings[~rankings["key"].isin(taken_keys)].copy()
    if available.empty:
        st.info("No available players left to suggest.")
        return

    display_cols = {
        "Player": "Player", "Team": "Team", "Position": "Pos",
        "Season Score": "Season", "Week1 Score": "Week1", "Fixture Score": "Fixture",
        "VORP": "VORP", "Draft Player Score": "Score",
    }

    def _fmt(df: pd.DataFrame) -> pd.DataFrame:
        d = df[list(display_cols.keys())].rename(columns=display_cols)
        return d

    st.markdown("#### 🎯 Top 5 Overall Available")
    top5 = available.sort_values("Draft Player Score", ascending=False).head(5)
    render_styled_table(_fmt(top5), col_formats=_SCORE_COL_FORMATS, positive_color_cols=["Score"])

    st.markdown("#### Top 3 Available Per Position")
    pos_cols = st.columns(4)
    for i, pos in enumerate(_POS_ORDER):
        with pos_cols[i]:
            st.caption(_POS_LABELS[pos])
            top3 = (
                available[available["Position"] == pos]
                .sort_values("Draft Player Score", ascending=False)
                .head(3)
            )
            render_styled_table(_fmt(top3), col_formats=_SCORE_COL_FORMATS, positive_color_cols=["Score"])


def show_draft_helper_page():
    """
    Draft Helper — Blended Rankings, Live Draft Sync, and Suggested Picks.

    - Blends preseason Rotowire Season Rankings + GW1 projections + opening
      FDR into a Player Score (weights shift toward Week1/Fixture as the
      draft progresses), then a VORP scarcity adjustment (Draft Player Score)
      that accounts for this league's exclusive, no-player-sharing rosters.
    - Optionally syncs Taken/Mine live from the FPL Draft API, unioned with
      manual session-state overrides.
    - Suggested Picks tab surfaces top available players + positional needs.
    """
    st.markdown("""
    <style>
    .draft-title {
        background: linear-gradient(135deg, #37003c, #5a0060);
        padding: 16px 20px; border-radius: 10px; margin-bottom: 0.5rem;
    }
    .draft-title h2 { color: #00ff87; margin: 0; font-size: 1.5rem; }
    .draft-title p { color: rgba(255,255,255,0.8); margin: 4px 0 0 0; font-size: 0.9rem; }
    .draft-summary {
        background: #1a1a2e; border: 1px solid #333; border-radius: 8px;
        padding: 12px 16px; color: #e0e0e0; margin-top: 0.5rem;
    }
    .draft-summary .num { color: #00ff87; font-weight: 700; }
    </style>
    <div class="draft-title">
        <h2>🧠 Draft Helper</h2>
        <p>Blended Season/Week1/Fixture rankings, scarcity-adjusted for this league's exclusive rosters, with live draft sync.</p>
    </div>
    """, unsafe_allow_html=True)

    if not getattr(config, "ROTOWIRE_SEASON_RANKINGS_URL", None):
        st.error("Missing `config.ROTOWIRE_SEASON_RANKINGS_URL`. Please set it to your Rotowire season rankings page.")
        return

    _ensure_session()

    league_id = getattr(config, "FPL_DRAFT_LEAGUE_ID", None) or None
    sync_available = bool(league_id)

    default_num_teams = _DEFAULT_NUM_TEAMS
    if sync_available:
        try:
            entries = get_league_entries(league_id)
            if entries:
                default_num_teams = len(entries)
        except Exception:
            pass

    with st.expander("⚙️ Settings", expanded=False):
        gw1_url = _gw1_url_selector()
        horizon = st.slider(
            "Opening Fixture Horizon (GWs)", min_value=1, max_value=5, value=3,
            help="How many opening gameweeks count toward the Fixture Score",
        )
        num_teams = st.number_input(
            "League Size (teams)", min_value=2, max_value=20, value=default_num_teams, step=1,
            help="Used for the positional scarcity (VORP) calculation — leaguewide slots at each "
                 "position = teams x standard squad slots.",
        )
        st.markdown("---")
        manual_weights = st.checkbox(
            "Pin fixed weights (override auto draft-progress weighting)", value=False,
            help="By default, weights shift from Season-heavy (early picks) to Week1/Fixture-heavy "
                 "(late picks) as the draft progresses. Check this to pin fixed weights instead.",
        )
        if manual_weights:
            wcol1, wcol2, wcol3 = st.columns(3)
            with wcol1:
                w_season_pct = st.slider("Season-Long Weight (%)", 0, 100, 55)
            with wcol2:
                w_week1_pct = st.slider("Week 1 Weight (%)", 0, 100, 30)
            with wcol3:
                w_fixture_pct = st.slider("Fixture Ease Weight (%)", 0, 100, 15)
            w_total = max(w_season_pct + w_week1_pct + w_fixture_pct, 1)
            manual_weight_tuple = (w_season_pct / w_total, w_week1_pct / w_total, w_fixture_pct / w_total)
        else:
            manual_weight_tuple = None

    st.markdown("#### Live Draft Sync")
    sync_col1, sync_col2, sync_col3 = st.columns([1.3, 1, 1])
    with sync_col1:
        live_sync = st.checkbox(
            "🔴 Sync Taken/Mine from live draft", value=sync_available, disabled=not sync_available,
            help="Auto-pulls who's been drafted from the FPL Draft API. Requires FPL_DRAFT_LEAGUE_ID "
                 "to be configured." if sync_available else
                 "Configure FPL_DRAFT_LEAGUE_ID (League Setup page) to enable live sync.",
        )
    with sync_col2:
        auto_refresh = st.checkbox(
            f"Auto-refresh (~{_REFRESH_INTERVAL_SECONDS}s)", value=True,
            disabled=not (sync_available and live_sync),
        )
    with sync_col3:
        manual_refresh = st.button("🔄 Refresh Now", disabled=not (sync_available and live_sync))

    # --- Load base player pool ---
    try:
        bootstrap = get_classic_bootstrap_static()
    except Exception as e:
        show_api_error("loading player data for Draft Helper", exception=e)
        return
    if not bootstrap:
        show_api_error("loading player data for Draft Helper")
        return

    pool = _build_full_player_pool(bootstrap)
    if pool.empty:
        st.error("No players available.")
        return
    pool["key"] = pool.apply(_player_key, axis=1)

    # --- Live draft sync fetch ---
    api_taken_keys, api_mine_keys = set(), set()
    if sync_available and live_sync:
        do_refresh = manual_refresh or (
            auto_refresh and (time.time() - st.session_state.draft_last_sync_ts) >= _REFRESH_INTERVAL_SECONDS
        )
        if do_refresh:
            get_starting_team_composition.clear()
            st.session_state.draft_last_sync_ts = time.time()
        try:
            draft_picks = get_starting_team_composition(league_id)
        except Exception as e:
            draft_picks = {}
            st.warning(f"Live draft sync failed: {e}")

        all_taken_names = [name for team in draft_picks.values() for name in team.get("players", [])]
        my_names = draft_picks.get(config.FPL_DRAFT_TEAM_ID, {}).get("players", [])
        api_taken_keys = _names_to_keys(all_taken_names, pool)
        api_mine_keys = _names_to_keys(my_names, pool)

    taken_keys = st.session_state.draft_taken_keys | api_taken_keys
    mine_keys = st.session_state.draft_mine_keys | api_mine_keys

    # --- Draft-progress weighting ---
    total_draft_picks = int(num_teams) * sum(_STANDARD_SQUAD_SLOTS.values())
    picks_made = len(taken_keys)
    if manual_weight_tuple is not None:
        w_season, w_week1, w_fixture = manual_weight_tuple
    else:
        p = season_progress_weight(picks_made, total_gws=total_draft_picks)
        w_season = 0.60 - 0.35 * p
        w_week1 = 0.25 + 0.35 * p
        w_fixture = 0.15

    # --- Load scoring inputs ---
    try:
        season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
    except Exception as e:
        st.warning(f"Could not load Rotowire season rankings: {e}")
        season_rankings_df = pd.DataFrame()

    gw1_projections_df = pd.DataFrame()
    if gw1_url:
        try:
            gw1_projections_df = get_rotowire_player_projections(gw1_url)
        except Exception as e:
            st.warning(f"Could not load GW1 Rotowire projections: {e}")

    try:
        ffp_df = get_ffp_projections_data()
    except Exception:
        ffp_df = None

    try:
        _, _, fdr_avg = get_fixture_difficulty_grid(weeks=horizon)
    except Exception as e:
        st.warning(f"Could not load fixture difficulty data: {e}")
        fdr_avg = pd.Series(dtype=float)

    current_gw = get_current_gameweek() or 1

    scored = compute_early_season_scores(
        pool, gw1_projections_df, season_rankings_df, ffp_df, fdr_avg,
        current_gw, w_season, w_week1, w_fixture, format_context="draft",
    )
    scored = _compute_vorp(scored, int(num_teams))

    scored["Taken"] = scored["key"].isin(taken_keys)
    scored["Mine"] = scored["key"].isin(mine_keys)
    rankings = scored.sort_values("Draft Player Score", ascending=False).reset_index(drop=True)
    rankings["Rank"] = rankings.index + 1

    st.caption(
        f"Draft progress: pick ~{picks_made + 1} of {total_draft_picks} | "
        f"Weights — Season {w_season:.0%} / Week1 {w_week1:.0%} / Fixture {w_fixture:.0%}"
        + (" (pinned)" if manual_weight_tuple is not None else " (auto)")
    )

    st.markdown("---")

    tab_board, tab_suggested = st.tabs(["📋 Board", "🎯 Suggested Picks"])

    with tab_board:
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        with c1:
            search = st.text_input("Search player", "")
        with c2:
            pos_options = sorted([p for p in rankings["Position"].dropna().unique().tolist() if p != ""])
            pos_filter = st.multiselect("Filter positions", pos_options, default=[])
        with c3:
            show_only_available = st.checkbox("Show only available", value=True)
        with c4:
            if st.button("Reset Board", type="secondary"):
                st.session_state.draft_taken_keys = set()
                st.session_state.draft_mine_keys = set()
                st.rerun()

        df = rankings.copy()
        if search.strip():
            needle = search.lower()
            df = df[df["Player"].str.lower().str.contains(needle, na=False)]
        if pos_filter:
            df = df[df["Position"].isin(pos_filter)]
        if show_only_available:
            df = df[~df["Taken"]]

        board_display = df[[
            "Rank", "Player", "Team", "Position", "Season Score", "Week1 Score",
            "Fixture Score", "VORP", "Draft Player Score", "Taken", "Mine",
        ]].copy()
        for c in ["Season Score", "Week1 Score", "Fixture Score", "VORP", "Draft Player Score"]:
            board_display[c] = board_display[c].round(2)

        st.markdown(
            '<div style="background:linear-gradient(135deg,#37003c,#5a0060);padding:10px 16px;'
            'border-radius:8px;margin:0.5rem 0;color:#00ff87;font-weight:700;font-size:1.1rem;">'
            '📋 Rankings</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Sorted by Draft Player Score (scarcity-adjusted). Taken/Mine auto-sync from the live "
            "draft when enabled above — check/uncheck manually to override or fill gaps."
        )

        edited = st.data_editor(
            board_display,
            key="draft_editor",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", disabled=True),
                "Player": st.column_config.TextColumn("Player", disabled=True),
                "Team": st.column_config.TextColumn("Team", disabled=True),
                "Position": st.column_config.TextColumn("Position", disabled=True),
                "Season Score": st.column_config.NumberColumn("Season", disabled=True),
                "Week1 Score": st.column_config.NumberColumn("Week1", disabled=True),
                "Fixture Score": st.column_config.NumberColumn("Fixture", disabled=True),
                "VORP": st.column_config.NumberColumn("VORP", disabled=True),
                "Draft Player Score": st.column_config.NumberColumn("Score", disabled=True),
                "Taken": st.column_config.CheckboxColumn("Taken"),
                "Mine": st.column_config.CheckboxColumn("Mine"),
            },
        )

        save_col, mine_col = st.columns([1, 2])
        with save_col:
            if st.button("💾 Save Changes", type="primary"):
                edited_keys = (
                    edited[["Player", "Team", "Position"]]
                    .assign(key=lambda x: x.apply(_player_key, axis=1))
                    .merge(edited[["Taken", "Mine"]], left_index=True, right_index=True)
                )
                for _, row in edited_keys.iterrows():
                    k = row["key"]
                    if bool(row["Taken"]):
                        st.session_state.draft_taken_keys.add(k)
                    else:
                        st.session_state.draft_taken_keys.discard(k)
                    if bool(row["Mine"]):
                        st.session_state.draft_mine_keys.add(k)
                    else:
                        st.session_state.draft_mine_keys.discard(k)
                st.success("Board updated.")
                st.rerun()

        with mine_col:
            with st.expander("📋 My Picks (selected as Mine)"):
                mine_df = rankings[rankings["key"].isin(mine_keys)]
                mine_df = mine_df.sort_values("Rank")[["Rank", "Player", "Team", "Position"]]
                render_styled_table(mine_df)

        total_players = len(rankings)
        taken_n = len(taken_keys)
        available_n = total_players - taken_n
        st.markdown(
            f'<div class="draft-summary">'
            f'<span class="num">{available_n}</span> Available &nbsp;&bull;&nbsp; '
            f'<span class="num">{taken_n}</span> Taken &nbsp;&bull;&nbsp; '
            f'<span class="num">{total_players}</span> Total'
            f'</div>',
            unsafe_allow_html=True,
        )

    with tab_suggested:
        _suggested_picks_tab(rankings, mine_keys, taken_keys, int(num_teams))

    if sync_available and live_sync and auto_refresh:
        time.sleep(1.0)
        st.rerun()
