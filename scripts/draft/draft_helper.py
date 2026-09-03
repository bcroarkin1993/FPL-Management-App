import time

import streamlit as st
import pandas as pd
import config
from scripts.common.utils import get_rotowire_season_rankings
from scripts.common.styled_tables import render_styled_table
from scripts.common.scraping import (
    club_news_cache_status,
    get_club_transfer_news,
    get_transfer_news,
    start_club_news_prefetch,
    start_transfer_news_prefetch,
    transfer_news_cache_status,
)
from scripts.common.transfer_risk_app import (
    build_inbound_competition,
    build_transfer_risk,
    get_pl_team_names,
)
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.draft.draft_helper")

def _ensure_session():
    if "draft_taken_keys" not in st.session_state:
        st.session_state.draft_taken_keys = set()
    if "draft_mine_keys" not in st.session_state:
        st.session_state.draft_mine_keys = set()

def _player_key(row: pd.Series) -> str:
    """Unique key for a player row to persist selection reliably."""
    return f"{row.get('Player','')}|{row.get('Team','')}|{row.get('Position','')}"

#: How deep to scan for transfer news by default. One request per player, so
#: this is a real cost -- and beyond a couple of hundred names a draft board is
#: not going to reach anybody anyway.
_DEFAULT_SCAN_DEPTH = 150

#: Measured throughput of the parallel fetcher (150 players in ~8s), rounded up
#: a little so the estimate errs toward overpromising time rather than under.
_SECONDS_PER_PLAYER = 0.07

#: Where the computed risk frame is parked between reruns.
_RISK_STATE_KEY = "draft_transfer_risk_df"
#: Same, for the inbound (minutes competition) frame and the arrivals watchlist.
_MINUTES_STATE_KEY = "draft_minutes_competition_df"
_ARRIVALS_STATE_KEY = "draft_arrivals_df"


def _risk_columns():
    return ["Transfer_Risk", "Transfer_Mult", "Transfer_Destination",
            "Transfer_Outlets", "Transfer_Note"]


def _minutes_columns():
    return ["Minutes_Mult", "Competition"]


def _merge_risk(rankings: pd.DataFrame, scored: pd.DataFrame,
                minutes: pd.DataFrame = None, use_risk: bool = True,
                use_minutes: bool = True) -> pd.DataFrame:
    """Left-join risk (and minutes competition) onto the full board.

    The unscanned tail fills neutrally — a player nobody checked is not a player
    with no news, and must not be ranked as though he were penalised.

    ``Adj Points`` shows exactly the adjustments that are switched on, so the
    column and the ordering can never disagree about what was applied.
    """
    merged = rankings.merge(
        scored[["Player", "Team", "Position"] + _risk_columns()],
        on=["Player", "Team", "Position"], how="left",
    )
    merged["Transfer_Risk"] = merged["Transfer_Risk"].fillna(0.0)
    merged["Transfer_Mult"] = merged["Transfer_Mult"].fillna(1.0)
    for col in ("Transfer_Destination", "Transfer_Note"):
        merged[col] = merged[col].fillna("")
    merged["Transfer_Outlets"] = merged["Transfer_Outlets"].fillna(0).astype(int)

    if minutes is not None and not getattr(minutes, "empty", True):
        merged = merged.merge(
            minutes[["Player", "Team", "Position"] + _minutes_columns()],
            on=["Player", "Team", "Position"], how="left",
        )
    if "Minutes_Mult" not in merged.columns:
        merged["Minutes_Mult"] = 1.0
        merged["Competition"] = ""
    merged["Minutes_Mult"] = merged["Minutes_Mult"].fillna(1.0)
    merged["Competition"] = merged["Competition"].fillna("")

    factor = pd.Series(1.0, index=merged.index)
    if use_risk:
        factor = factor * merged["Transfer_Mult"]
    if use_minutes:
        factor = factor * merged["Minutes_Mult"]
    merged["Adj Points"] = (
        pd.to_numeric(merged["Points"], errors="coerce") * factor
    ).round(1)
    return merged


def _apply_transfer_risk(rankings: pd.DataFrame):
    """Discount season projections by the risk a player leaves the Premier League.

    The board previously ranked purely on projected output, which is how Ollie
    Watkins came to be a top-32 pick weeks before moving to Al-Hilal: a player who
    cannot score a point all season was priced as if he would play 38 games.

    Results are held in session state, so filtering or searching the board does
    not discard them -- every Streamlit interaction reruns this function, and
    recomputing from an empty cache would blank the risk columns mid-draft.
    """
    st.markdown(
        '<div style="background:linear-gradient(135deg,#37003c,#5a0060);padding:10px 16px;'
        'border-radius:8px;margin:0.5rem 0;color:#00ff87;font-weight:700;font-size:1.1rem;">'
        '\U0001F6A8 Transfer Risk</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.1, 1, 1.4])
    with c1:
        adjust = st.checkbox(
            "Adjust for transfer risk", value=True,
            help="Discount each player's season projection by how likely he is to "
                 "leave the Premier League, and how much of the season that would cost.",
        )
        adjust_minutes = st.checkbox(
            "Adjust for incoming signings", value=True,
            help="Discount players whose club is signing someone in their position. "
                 "Capped at 25%, so it separates similar players rather than "
                 "overriding quality.",
        )
    with c2:
        depth = int(st.number_input(
            "Scan depth", min_value=25, max_value=400, value=_DEFAULT_SCAN_DEPTH, step=25,
            help="How far down the board to check for transfer news.",
        ))

    pool = rankings.head(depth)
    pairs = tuple((str(r["Player"]), str(r["Team"])) for _, r in pool.iterrows())

    try:
        cached_n, missing_n = transfer_news_cache_status(pairs)
    except Exception:
        cached_n, missing_n = 0, len(pairs)

    with c3:
        scan = st.button("\U0001F50D Scan transfer news", type="secondary")
        if missing_n:
            st.caption("%d of %d players cached \u00b7 scanning the rest takes about %ds."
                       % (cached_n, len(pairs), max(1, round(missing_n * _SECONDS_PER_PLAYER))))
        else:
            st.caption("All %d players cached. Re-scan before a draft or on deadline day."
                       % len(pairs))

    # Start warming the cache the moment the page opens, so the work is usually
    # already done by the time anyone presses the button.
    if missing_n and not scan:
        start_transfer_news_prefetch(pairs, label="draft_helper_%d" % depth)

    availability, pl_teams = _load_reference_data()

    # Club signings are 20 requests against the player scan's 150, so they ride
    # along with the same button rather than getting one of their own.
    try:
        clubs_cached, clubs_missing = club_news_cache_status(pl_teams)
    except Exception:
        clubs_cached, clubs_missing = 0, len(pl_teams or [])
    if clubs_missing and not scan:
        start_club_news_prefetch(pl_teams, label="draft_helper_clubs")

    news = None
    try:
        if scan:
            progress = st.progress(0.0, text="Checking transfer news...")
            started = time.time()

            def _on_progress(done, total, _player):
                frac = done / float(total) if total else 1.0
                if done:
                    remaining = (time.time() - started) / done * (total - done)
                    text = ("Checking transfer news... %d/%d \u00b7 about %ds left"
                            % (done, total, max(1, round(remaining))))
                else:
                    text = "Checking transfer news... 0/%d" % total
                progress.progress(min(1.0, frac), text=text)

            news = get_transfer_news(pairs, force_refresh=True, progress=_on_progress)
            progress.empty()
        else:
            news = get_transfer_news(pairs, cached_only=True)
    except Exception as e:
        st.warning("Transfer news unavailable (%s). Rankings shown undiscounted." % e)

    if news is not None and not news.empty:
        scored = build_transfer_risk(pool, availability, news, pl_teams)
        st.session_state[_RISK_STATE_KEY] = scored[
            ["Player", "Team", "Position"] + _risk_columns()
        ]
    else:
        scored = st.session_state.get(_RISK_STATE_KEY)

    # --- Inbound: who is arriving, and whose minutes it costs -----------------
    club_news = None
    try:
        club_news = get_club_transfer_news(
            pl_teams, force_refresh=bool(scan), cached_only=not scan)
    except Exception as e:
        _logger.warning("Club signings news unavailable: %s", e)

    if club_news is not None and not club_news.empty:
        # Run this over the *scored* pool when there is one: a player who is
        # himself leaving must be exempt from competition for his own place.
        base = scored if scored is not None and not getattr(scored, "empty", True) else pool
        if "Transfer_Status" not in getattr(base, "columns", []):
            base = pool
        arrivals, discounted = build_inbound_competition(base, club_news, pl_teams)
        st.session_state[_ARRIVALS_STATE_KEY] = arrivals
        if not discounted.empty and "Minutes_Mult" in discounted.columns:
            st.session_state[_MINUTES_STATE_KEY] = discounted[
                ["Player", "Team", "Position"] + _minutes_columns()
            ]

    arrivals = st.session_state.get(_ARRIVALS_STATE_KEY)
    minutes = st.session_state.get(_MINUTES_STATE_KEY)
    _render_arrivals(arrivals)

    if scored is None or getattr(scored, "empty", True):
        if missing_n:
            st.info(
                "Checking transfer news in the background \u2014 press **Scan transfer "
                "news** to fetch now, or reload in a moment. Until then the board is "
                "Rotowire's raw ranking."
            )
        else:
            st.warning("No transfer news found. Rankings shown undiscounted.")
        return rankings, False

    merged = _merge_risk(rankings, scored, minutes,
                         use_risk=adjust, use_minutes=adjust_minutes)

    flagged = merged[merged["Transfer_Risk"] > 0.05]
    if len(flagged):
        biggest = flagged.nlargest(1, "Transfer_Risk").iloc[0]
        st.warning(
            "**%d player%s flagged.** Biggest faller: **%s** (%s) \u2014 %s, "
            "%.0f%% risk, %.0f \u2192 %.0f projected points."
            % (len(flagged), "" if len(flagged) == 1 else "s",
               biggest["Player"], biggest["Team"],
               biggest["Transfer_Destination"] or "destination unclear",
               100 * biggest["Transfer_Risk"],
               float(biggest["Points"]), float(biggest["Adj Points"]))
        )
    else:
        st.success("No meaningful transfer risk found.")

    if adjust or adjust_minutes:
        merged = merged.sort_values("Adj Points", ascending=False, na_position="last")
        merged["RW Rank"] = merged["Rank"]
        merged["Rank"] = range(1, len(merged) + 1)
    else:
        merged["RW Rank"] = merged["Rank"]

    return merged, True


def _render_arrivals(arrivals) -> None:
    """List the signings the model believes are happening.

    Shown whether or not the discount is switched on: knowing a club has just
    bought a striker is useful in its own right, and it is also the only way to
    see *why* an incumbent slipped down the board.
    """
    if arrivals is None or getattr(arrivals, "empty", True):
        return

    with st.expander("\U0001F6EC Incoming signings (%d)" % len(arrivals), expanded=False):
        st.caption(
            "Reported arrivals, corroborated by at least two outlets. A signing "
            "discounts players at the same club and position by up to 25% — the "
            "established starter least of all."
        )
        show = arrivals.copy()
        show["Fee"] = show["Fee"].map(
            lambda v: "" if pd.isna(v) else ("£%.0fm" % v if v >= 10 else "£%.1fm" % v)
        )
        show["Confidence"] = (pd.to_numeric(show["Confidence"], errors="coerce")
                              .fillna(0.0) * 100).round(0).astype(int).astype(str) + "%"
        show["Position"] = show["Position"].fillna("?")
        render_styled_table(
            show[["Player", "Club", "Position", "Fee", "Outlets", "Confidence", "Headline"]],
            max_height=400,
        )


@st.cache_data(ttl=600)
def _load_reference_data():
    """FPL availability (ground truth on completed moves) and the PL club list."""
    from scripts.common.fpl_classic_api import get_classic_bootstrap_static
    from scripts.fpl.injuries import get_fpl_availability_df

    try:
        availability = get_fpl_availability_df()
    except Exception:
        availability = pd.DataFrame()
    try:
        pl_teams = get_pl_team_names(get_classic_bootstrap_static())
    except Exception:
        pl_teams = []
    return availability, pl_teams


def show_draft_helper_page():
    """
    Draft Helper — Season-long Top 400 board with in-place selection.

    - Loads rankings via get_rotowire_player_projections(config.ROTOWIRE_SEASON_RANKINGS_URL, limit=400)
    - Lets you mark players as Taken (any team) and Mine (your picks)
    - Persists selections with st.session_state
    - Filters to show only available (default) or all, with search and position filter
    """
    # Dark theme CSS for Draft Helper
    st.markdown("""
    <style>
    /* Draft Helper dark theme */
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
        <h2>🧠 Draft Helper — Season Rankings</h2>
        <p>Mark draftees as <strong>Taken</strong> or <strong>Mine</strong> to keep your live board clean during the draft.</p>
    </div>
    """, unsafe_allow_html=True)

    # Guard: require configured URL
    if not getattr(config, "ROTOWIRE_SEASON_RANKINGS_URL", None):
        st.error("Missing `config.ROTOWIRE_SEASON_RANKINGS_URL`. Please set it to your Rotowire season rankings page.")
        return

    _ensure_session()

    # --- Load Rankings (Top 400) ---
    try:
        rankings = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL, limit=400).copy()
        # Format numeric columns for dataframe sorting
        numeric_cols = [
            "Overall Rank", "FW Rank", "MID Rank", "DEF Rank", "GK Rank",
            "Price", "TSB %", "Points", "PP/90", "Pos Rank", "Value"
        ]
        for c in numeric_cols:
            if c in rankings.columns:
                rankings[c] = pd.to_numeric(rankings[c], errors="coerce")
    except Exception as e:
        st.error(f"Failed to load season rankings: {e}")
        return

    # Normalize expected columns & types
    # Rename "Overall Rank" -> "Rank" if present
    if "Overall Rank" in rankings.columns:
        rankings = rankings.rename(columns={"Overall Rank": "Rank"})
    # Ensure columns exist
    for col in ["Rank", "Player", "Team", "Position", "Points", "PP/90", "Pos Rank"]:
        if col not in rankings.columns:
            rankings[col] = pd.NA

    # Coerce numerics
    rankings["Rank"] = pd.to_numeric(rankings["Rank"], errors="coerce")
    rankings["Points"] = pd.to_numeric(rankings["Points"], errors="coerce")
    rankings["PP/90"] = pd.to_numeric(rankings["PP/90"], errors="coerce")
    rankings["Pos Rank"] = pd.to_numeric(rankings["Pos Rank"], errors="coerce")

    # Drop exact duplicates (keep best-ranked)
    rankings = rankings.sort_values(["Rank", "Player"], na_position="last").drop_duplicates(
        subset=["Player", "Team", "Position"], keep="first"
    )

    # Discount the board by transfer risk before anything is ranked or displayed.
    rankings, has_risk = _apply_transfer_risk(rankings)

    # Session-state flags mapped to each row via a stable key
    rankings["key"] = rankings.apply(_player_key, axis=1)
    rankings["Taken"] = rankings["key"].isin(st.session_state.draft_taken_keys)
    rankings["Mine"] = rankings["key"].isin(st.session_state.draft_mine_keys)

    # --- Controls ---
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

    # Filter
    df = rankings.copy()
    if search.strip():
        needle = search.lower()
        df = df[df["Player"].str.lower().str.contains(needle, na=False)]
    if pos_filter:
        df = df[df["Position"].isin(pos_filter)]
    if show_only_available:
        df = df[~df["Taken"]]

    # Display columns (keep it compact & useful).  The risk columns only appear
    # once a scan has actually produced data, so the board stays clean before then.
    if has_risk:
        # Risk sits immediately after Position: it is the reason the board is
        # ordered the way it is, so it must be readable without scrolling past
        # three numeric columns to find it.
        display_cols = ["Rank", "RW Rank", "Player", "Team", "Position",
                        "Risk", "Transfer Note", "Points", "Adj Points",
                        "PP/90", "Pos Rank", "Taken", "Mine"]
        df = df.rename(columns={"Transfer_Note": "Transfer Note"})
        df["Risk"] = (pd.to_numeric(df["Transfer_Risk"], errors="coerce")
                      .fillna(0.0) * 100).round(0)
    else:
        display_cols = ["Rank", "Player", "Team", "Position", "Points", "PP/90", "Pos Rank", "Taken", "Mine"]
    display_cols = [c for c in display_cols if c in df.columns]

    st.markdown(
        '<div style="background:linear-gradient(135deg,#37003c,#5a0060);padding:10px 16px;'
        'border-radius:8px;margin:0.5rem 0;color:#00ff87;font-weight:700;font-size:1.1rem;">'
        '📋 Rankings</div>',
        unsafe_allow_html=True,
    )
    st.caption("Tip: Uncheck **Show only available** if you need to mark players as taken.")

    edited = st.data_editor(
        df[display_cols],
        key="draft_editor",
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", help="Overall Rank", disabled=True),
            "Player": st.column_config.TextColumn("Player", disabled=True),
            "Team": st.column_config.TextColumn("Team", disabled=True),
            "Position": st.column_config.TextColumn("Position", disabled=True),
            "Points": st.column_config.NumberColumn(
                "Points", help="Rotowire projected season points, undiscounted", disabled=True),
            "RW Rank": st.column_config.NumberColumn(
                "RW Rank", help="Rotowire's original overall rank", disabled=True),
            "Adj Points": st.column_config.NumberColumn(
                "Adj Points", help="Season points after the transfer-risk discount", disabled=True),
            "Risk": st.column_config.NumberColumn(
                "Risk", format="%d%%",
                help="Chance he leaves the Premier League, weighted by destination",
                disabled=True),
            "Transfer Note": st.column_config.TextColumn(
                "Transfer Note", help="Destination and how many outlets report it", disabled=True),
            "PP/90": st.column_config.NumberColumn("PP/90", disabled=True),
            "Pos Rank": st.column_config.NumberColumn("Pos Rank", disabled=True),
            "Taken": st.column_config.CheckboxColumn("Taken"),
            "Mine": st.column_config.CheckboxColumn("Mine"),
        },
    )

    # Save changes (only touches rows currently visible/edited)
    save_col, mine_col = st.columns([1, 2])
    with save_col:
        if st.button("💾 Save Changes", type="primary"):
            # Map edited rows back to session-state sets using their keys
            # Rebuild keys for the edited subset (since edited has no 'key' col)
            edited_keys = (
                edited[["Player", "Team", "Position"]]
                .assign(key=lambda x: x.apply(_player_key, axis=1))
                .merge(
                    edited[["Taken", "Mine"]],
                    left_index=True, right_index=True
                )
            )
            # Update session sets
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
            mine_df = rankings[rankings["key"].isin(st.session_state.draft_mine_keys)]
            mine_df = mine_df.sort_values("Rank", na_position="last")[["Rank", "Player", "Team", "Position"]]
            render_styled_table(mine_df)

    # Summary footer
    total_players = len(rankings)
    taken = len(st.session_state.draft_taken_keys)
    available = total_players - taken
    st.markdown(
        f'<div class="draft-summary">'
        f'<span class="num">{available}</span> Available &nbsp;&bull;&nbsp; '
        f'<span class="num">{taken}</span> Taken &nbsp;&bull;&nbsp; '
        f'<span class="num">{total_players}</span> Total'
        f'</div>',
        unsafe_allow_html=True,
    )
