"""
Classic FPL - Initial Squad Optimizer

Builds a season-opening 15-man squad (before GW1) using PuLP ILP, blending:
- Season value:  literal preseason Rotowire Season Rankings, as points per GW
- Week 1 value:  GW1 Rotowire + FFP blended projection x start likelihood,
                 reusing the app's exact 1GW methodology via compute_player_scores()
- Fixtures:      a small multiplier on the opening slate, not a headline term

The optimizer's objective is denominated in **expected points per gameweek**,
not percentiles. That distinction decides the whole squad, because Classic has a
budget: a premium out-projects a mid-price player by ~20% in points, but as a
positional percentile can rank *below* one (percentile has no headroom above
1.0). Optimizing percentiles makes every premium bad value, leaves money in the
bank, and parks real cash on the bench. Percentiles are still shown — they read
far better than raw points — but nothing optimizes on them.

Unlike Wildcard/Free Hit, this intentionally does NOT use ROS/Transfer/Keep
Score — those are diluted by live-season signals (form, starts, multi-GW
FFP data) that don't exist yet before GW1. See CLAUDE.md's "Transfer Scoring
Model" section for why ROS is a poor fit here.

Captaincy is baked into the objective (not just assigned post-hoc): a
season-pedigree-heavy CapPts is rewarded for whichever starter the solver
designates captain, giving it a real incentive to *acquire* one standout option
rather than several merely-good players. In points currency that bonus is worth
what the armband is actually worth — a doubled score — rather than a rounding
error on a percentile.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional
import config

from scripts.common.error_helpers import get_logger, show_api_error
from scripts.common.optimization import solve_squad_ilp
from scripts.common.styled_tables import render_styled_table
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_current_gameweek,
    get_rotowire_player_projections,
    get_fixture_difficulty_grid,
    position_converter,
)
from scripts.common.fixture_helpers import style_fixture_difficulty
from scripts.common.text_helpers import format_last_updated, to_display_name
from scripts.common.analytics import (
    compute_player_scores,
    merge_ffp_single_gw_data,
    merge_season_projections,
    positional_percentile,
)
from scripts.common.data_validation import check_initial_squad, format_issues
from scripts.common.scraping import (
    get_ffp_projections_data,
    get_rotowire_article_updated,
    get_rotowire_season_rankings,
)

_logger = get_logger("fpl_app.initial_squad")

# --- Scoring constants ---
# Defaults are fixed rather than dialled in: a season-opening squad is a
# hold-for-many-weeks decision, so season-long pedigree should dominate. The
# sliders in the advanced expander exist for experimentation, not routine use.
DEFAULT_W_SEASON = 0.70
DEFAULT_W_OPENING = 0.30
# Bench slots are worth something (rotation cover, an early Bench Boost) but not
# much: every pound spent on the bench is a pound not in the XI, and at 0.2 the
# solver was buying real players to sit them.
DEFAULT_BENCH_WEIGHT = 0.10
GWS_PER_SEASON = 38
NEUTRAL_FDR = 3.0
# 10% per FDR point away from neutral, applied to the opening component only.
# Across the league's actual FDR spread that is worth ~0.25 expected points,
# against a ~1.9-point gap between the best and 10th-best midfielder -- so
# fixtures separate similar players without ever outranking quality.
FIXTURE_TILT = 0.10
# What an unranked player is worth, as a fraction of the *weakest ranked* player
# at their position. Absence from a 400-deep list is evidence of being worse than
# everyone on it, so the floor sits below the ranked minimum rather than inside
# the ranked distribution -- see _compute_scores.
UNRANKED_FLOOR_FRACTION = 0.5
CAPTAIN_SEASON_WEIGHT = 0.85

# The armband doubles a player's score. That is the rule, not a tunable
# preference, so it is a constant rather than a slider.
CAPTAIN_MULTIPLIER = 2.0
# Triple Captain is available twice a season (once per half), and each use turns
# one gameweek into 3x instead of 2x. Amortised across the season that lifts the
# captain's effective multiplier slightly above 2x -- and it accrues to whoever
# your standout captain is, which is a reason to pay up for one at build time.
TRIPLE_CAPTAIN_USES = 2
CAPTAIN_EFFECTIVE_MULTIPLIER = CAPTAIN_MULTIPLIER + TRIPLE_CAPTAIN_USES / GWS_PER_SEASON
# solve_squad_ilp() adds this on top of the player's own score, so the bonus is
# the multiplier minus the one copy already counted in the starting XI.
CAPTAIN_BONUS_WEIGHT = CAPTAIN_EFFECTIVE_MULTIPLIER - 1.0


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def _format_money(value: float) -> str:
    return f"£{value:.1f}m"


def _score_card(label: str, value: str, accent: str = "#00ff87") -> str:
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'</div>'
    )


# Streamlit widgets own their value via session_state, so binding two sliders
# together means writing the partner's key in an on_change callback -- which
# runs before the rerun that redraws it.
_W_SEASON_KEY = "isq_w_season_pct"
_W_OPENING_KEY = "isq_w_opening_pct"


def _init_weight_state() -> None:
    """Seed the weight sliders once per session."""
    if _W_SEASON_KEY not in st.session_state:
        st.session_state[_W_SEASON_KEY] = int(round(DEFAULT_W_SEASON * 100))
    if _W_OPENING_KEY not in st.session_state:
        st.session_state[_W_OPENING_KEY] = int(round(DEFAULT_W_OPENING * 100))


def _sync_weight_from_season() -> None:
    st.session_state[_W_OPENING_KEY] = 100 - st.session_state[_W_SEASON_KEY]


def _sync_weight_from_opening() -> None:
    st.session_state[_W_SEASON_KEY] = 100 - st.session_state[_W_OPENING_KEY]


def _data_source_urls():
    """Editable URLs for both Rotowire feeds.

    Neither slug is auto-discoverable the way weekly rankings articles are, so
    both are pinned in config and re-pinned each preseason. The season URL used
    to be uneditable, which meant a stale slug could only be fixed by editing
    config.py — and because a failed fetch degrades to an empty frame, the page
    looked identical either way.
    """
    with st.expander("Rotowire URLs", expanded=False):
        st.caption(
            "This squad is built from **two** Rotowire tables: the season-long "
            "Top 400 and the GW1 rankings. Neither article slug is "
            "auto-discoverable, so update them here each preseason."
        )
        season_url = st.text_input(
            "Season rankings article URL",
            value=config.ROTOWIRE_SEASON_RANKINGS_URL or "",
            placeholder="https://www.rotowire.com/soccer/article/...",
            help="Rotowire's 'Top 400 for the season' article — drives season-long value.",
        )
        gw1_url = st.text_input(
            "GW1 rankings article URL",
            value=config.ROTOWIRE_GW1_URL or "",
            placeholder="https://www.rotowire.com/soccer/article/...",
            help="Rotowire's GW1 'best picks' article — drives the Week 1 projection.",
        )
    return season_url.strip(), gw1_url.strip()


def _source_status_row(name: str, feeds: str, rows: int, matched: Optional[int],
                       ok: bool, note: str = "", updated=None,
                       show_updated: bool = True) -> dict:
    """One row of the Data Sources status table."""
    if not ok or not rows:
        status = "🔴 Failed"
    elif note:
        status = f"🟡 {note}"
    else:
        status = "🟢 OK"
    if matched is None:
        match_txt = "—"
    elif rows:
        match_txt = f"{matched} ({matched / rows:.0%})"
    else:
        match_txt = "0"
    return {"Source": name, "Feeds": feeds, "Status": status,
            "Rows": rows if rows else 0, "Matched": match_txt,
            # When the rankings were published. A table written before the last
            # team-news cycle is worth less than one written after it.
            "Updated": format_last_updated(updated) if show_updated else "—"}


def _render_source_status(rows: list) -> None:
    """Render the Data Sources status table.

    Match rate is the headline number here. A name-based merge that silently
    misses is the quietest bug in this app — unmatched players fall back to a
    neutral default, so the #2 asset in the game can score as exactly average
    with nothing on screen to suggest anything went wrong.
    """
    if not rows:
        return
    st.markdown("##### Data Sources")
    render_styled_table(pd.DataFrame(rows))
    degraded = [r for r in rows if r["Status"].startswith("🔴")]
    if degraded:
        st.warning(
            "⚠️ %s did not load. Scores fall back to neutral defaults, which makes "
            "the squad below unreliable — check the URLs under **Data Sources**."
            % ", ".join(r["Source"] for r in degraded)
        )


def _build_full_player_pool(bootstrap: dict) -> pd.DataFrame:
    """Build the full, unfiltered candidate pool (price/team/position/status).

    Deliberately unfiltered so downstream percentile scoring is computed
    against the true full player universe — injury/availability filtering
    happens afterward, narrowing the *candidate set* without skewing the
    *reference distribution* percentiles are computed against.
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
            # `Player` stays the full legal name because the projection merges
            # match on it. `Display_Name` is what the UI shows -- see
            # to_display_name() for why neither raw field is presentable.
            "Player": full_name or p.get("web_name", "Unknown"),
            "Display_Name": to_display_name(
                p.get("first_name"), p.get("second_name"), p.get("web_name")),
            "Web_Name": p.get("web_name", "Unknown"),
            "Team": team_short,
            "Position": position,
            "Price": p.get("now_cost", 0) / 10.0,
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
            "news": p.get("news", "") or "",
            "total_points": p.get("total_points", 0),
            "form": float(p.get("form", 0) or 0),
            "AvgFDR": 3.0,  # placeholder — ROS is discarded but compute_player_scores needs the column present
        })

    return pd.DataFrame(rows)


# Below this many candidates at a position, dropping dead weight risks making
# the squad unsolvable, so the rule yields rather than the optimizer failing.
_MIN_CANDIDATES_PER_POSITION = {"G": 4, "D": 10, "M": 10, "F": 6}


def _drop_dead_weight(pool: pd.DataFrame) -> pd.DataFrame:
    """Remove players with no points appeal at all.

    A player in neither the season rankings nor the GW1 table has no evidence
    he will ever score. That is fine for a slot you never use, but Bench Boost
    turns all four bench slots live, and the alternative costs almost nothing:
    a ranked replacement is the same price at DEF/MID/FWD and £0.5m more at GK.

    This is deliberately not the same lever as bench weight. Bench weight asks
    how much bench *points* count; this asks whether a slot can score at all.

    Positions that would be left too thin keep their dead weight — an
    unsolvable squad is worse than a weak bench slot.
    """
    if "No_Appeal" not in pool.columns:
        return pool

    keep = ~pool["No_Appeal"].fillna(False)
    for position, minimum in _MIN_CANDIDATES_PER_POSITION.items():
        at_pos = pool["Position"] == position
        if int((keep & at_pos).sum()) < minimum:
            _logger.warning(
                "Only %d %s candidates survive the dead-weight filter (need %d); "
                "keeping them to stay solvable.",
                int((keep & at_pos).sum()), position, minimum,
            )
            keep = keep | at_pos
    return pool[keep].reset_index(drop=True)


def _apply_eligibility_filters(
    pool: pd.DataFrame,
    exclude_injured: bool = True,
    min_chance_of_playing: int = 75,
    exclude_dead_weight: bool = True,
) -> pd.DataFrame:
    """Narrow the scored pool down to squad-selection candidates."""
    if exclude_dead_weight:
        pool = _drop_dead_weight(pool)
    if not exclude_injured:
        return pool

    is_cheap_fodder = pool["Price"] <= 4.5
    chance = pool["chance_of_playing_next_round"]
    news_flag = pool["news"].str.lower().apply(
        lambda n: any(word in n for word in ["injured", "illness", "suspended", "unavailable", "out"])
    )
    low_chance = chance.notna() & (chance < min_chance_of_playing)

    exclude_mask = (~is_cheap_fodder) & (low_chance | news_flag)
    return pool[~exclude_mask].reset_index(drop=True)


def _compute_scores(
    pool: pd.DataFrame,
    gw1_projections_df: pd.DataFrame,
    season_rankings_df: pd.DataFrame,
    ffp_df: Optional[pd.DataFrame],
    fdr_avg: pd.Series,
    current_gw: int,
    w_season: float,
    w_opening: float,
    fixture_tilt: float = FIXTURE_TILT,
    stats: Optional[dict] = None,
) -> pd.DataFrame:
    """Merge Season/Week1/Fixture signals into an expected-points objective.

    The optimizer maximizes ``ExpPts`` -- expected points per gameweek -- not a
    percentile. Under a budget constraint that distinction decides the whole
    squad: Haaland out-projects a mid-price midfielder by ~20% in points, but as
    a positional percentile he can rank *below* one (0.974 vs 0.977), because
    percentile has no headroom above 1.0. Optimizing percentiles therefore makes
    every premium bad value, leaves money unspent, and puts real cash on the
    bench. Percentile columns are still computed -- they are far more readable
    than raw points -- but only for display.
    """
    result = pool.copy()

    season_stats, gw1_stats = {}, {}

    # Season-long value: literal preseason Rotowire Season Rankings, not the
    # ongoing (and pre-season-unreliable) ROS blend.
    result = merge_season_projections(
        result, season_rankings_df, output_col="SeasonProjection", stats=season_stats)

    # GW1 projection — reuses merge_season_projections since Rotowire's weekly
    # rankings table has the same Player/Team/Points shape as season rankings.
    result = merge_season_projections(
        result, gw1_projections_df, output_col="Points", stats=gw1_stats)
    result["Points"] = result["Points"].fillna(0)

    # FFP single-GW data (Predicted, Start, LongStart) for the 1GW blend.
    result = merge_ffp_single_gw_data(result, ffp_df)

    # Week1 Score: reuse the app's exact 1GW methodology (0.6 Rotowire + 0.4
    # FFP Predicted, x start likelihood). Discard ROS/Transfer/Keep — those
    # depend on live-season signals (form, starts, multi-GW data) that don't
    # exist pre-season.
    scored = compute_player_scores(result, result, current_gw=current_gw, format_context="classic")
    result["Week1 Score"] = scored["1GW"]
    result["GW1 Proj Pts"] = pd.to_numeric(scored["_effective_proj"], errors="coerce").fillna(0.0)

    # Season Score: positional percentile of the literal season-long ranking.
    result["Season Score"] = positional_percentile(
        result, result, value_col="SeasonProjection", position_col="Position"
    )

    # Fixture Score: opening-slate FDR, inverted (easier = better) and percentile-ranked.
    result["Team_AvgFDR"] = result["Team"].map(fdr_avg).fillna(3.0)
    result["_fixture_ease_raw"] = -result["Team_AvgFDR"]
    result["Fixture Score"] = positional_percentile(
        result, result, value_col="_fixture_ease_raw", position_col="Position"
    )

    # --- Expected points per gameweek (the optimizer's actual currency) ---

    # Season total -> per-GW rate, so both components are in the same unit.
    result["SeasonPG"] = pd.to_numeric(result["SeasonProjection"], errors="coerce") / GWS_PER_SEASON

    # A player absent from a 400-deep season ranking is not average — absence is
    # itself the signal, the same way absence from Rotowire's weekly table means
    # "not starting".
    #
    # The floor must be a fraction of the position's *weakest ranked* player, not
    # a quantile inside the ranked distribution. Coverage is not uniform across
    # positions: Rotowire lists roughly one goalkeeper per club, so only ~21 of
    # 67 GKs are ranked and they are all starters. A 10th-percentile-of-ranked
    # floor therefore paid every backup keeper a starter's rate — 2.65 against a
    # backup defender's 0.40 — and bought them onto the bench.
    result["Unranked"] = result["SeasonPG"].isna()
    n_unmatched = int(result["Unranked"].sum())
    for pos, group in result.groupby("Position"):
        ranked = group["SeasonPG"].dropna()
        floor = float(ranked.min()) * UNRANKED_FLOOR_FRACTION if not ranked.empty else 0.0
        result.loc[(result["Position"] == pos) & result["SeasonPG"].isna(), "SeasonPG"] = floor
    result["SeasonPG"] = result["SeasonPG"].fillna(0.0)

    # "Dead weight": invisible to both sources, so there is no evidence they will
    # ever score. Distinct from simply being cheap — a cheap ranked player still
    # returns something on a Bench Boost, these return nothing.
    result["No_Appeal"] = result["Unranked"] & ~(pd.to_numeric(
        result["Points"], errors="coerce").fillna(0) > 0)

    # Opening fixtures modify the *opening* term only. Applying them to the
    # whole blend double-counts: a season-long projection already prices in all
    # 38 fixtures, so multiplying it by opening-slate ease inflates a player's
    # full-season value for a soft first month he will have long since
    # transferred around.
    result["OpeningPG"] = result["GW1 Proj Pts"] * (
        1.0 + fixture_tilt * (NEUTRAL_FDR - result["Team_AvgFDR"])
    )

    # The trade-off is now a single question: how much does a fast start matter
    # against season-long quality?
    result["ExpPts"] = w_season * result["SeasonPG"] + w_opening * result["OpeningPG"]

    # The armband goes on a week-in, week-out producer, so it leans harder on
    # season pedigree than the squad score does.
    result["CapPts"] = (CAPTAIN_SEASON_WEIGHT * result["SeasonPG"]
                        + (1.0 - CAPTAIN_SEASON_WEIGHT) * result["GW1 Proj Pts"])

    result["Player Score"] = (
        w_season * result["Season Score"]
        + w_opening * result["Week1 Score"]
    )
    result["Captain Score"] = (CAPTAIN_SEASON_WEIGHT * result["Season Score"]
                               + (1.0 - CAPTAIN_SEASON_WEIGHT) * result["Week1 Score"])

    if n_unmatched:
        _logger.info(
            "Initial squad: %d/%d players had no season ranking — floored at "
            "%.0f%% of the weakest ranked player in their position.",
            n_unmatched, len(result), UNRANKED_FLOOR_FRACTION * 100,
        )
    if stats is not None:
        stats["season"] = season_stats
        stats["gw1"] = gw1_stats
        stats["unmatched_season"] = n_unmatched

    return result


def _positional_color_ratios(rows: pd.DataFrame, pool: pd.DataFrame,
                             col: str = "ExpPts"):
    """Grade each row's `col` against others in the *same position*, as 0-1.

    Positions are not comparable on raw expected points: the best goalkeeper in
    the game projects about what a mid-table midfielder does. Grading everyone
    on one scale therefore paints every keeper red for being a keeper. Scaled
    within position, the best available GK reads 1.00 — which is the useful
    statement.

    Returns None if the inputs can't support it, so callers fall back to the
    renderer's own column-wide range.
    """
    if pool is None or rows is None or rows.empty:
        return None
    if col not in pool.columns or col not in rows.columns:
        return None
    if "Position" not in pool.columns or "Position" not in rows.columns:
        return None

    bounds = {}
    pool_vals = pd.to_numeric(pool[col], errors="coerce")
    for position, group in pool.groupby("Position"):
        vals = pool_vals.loc[group.index]
        vals = vals[np.isfinite(vals)]
        if not vals.empty and vals.min() != vals.max():
            bounds[position] = (float(vals.min()), float(vals.max()))
    if not bounds:
        return None

    ratios = []
    for _, row in rows.iterrows():
        value = pd.to_numeric(row.get(col), errors="coerce")
        span = bounds.get(row.get("Position"))
        if span is None or not np.isfinite(value):
            ratios.append(np.nan)
        else:
            lo, hi = span
            ratios.append(min(1.0, max(0.0, (value - lo) / (hi - lo))))
    return ratios


def _display_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        is_cap = bool(row.get("Is_Captain", False))
        display = row.get("Display_Name") or row["Player"]
        name = f"{display} (C)" if is_cap else display
        rows.append({
            "Player": name,
            "Team": row["Team"],
            "Pos": row["Position"],
            "Price": f"£{row['Price']:.1f}m",
            "Season": row["Season Score"],
            "Week1": row["Week1 Score"],
            "Fixture": row["Fixture Score"],
            "Score": row["Player Score"],
            "Exp Pts/GW": row["ExpPts"],
            "GW1 Proj": row["GW1 Proj Pts"],
        })
    return pd.DataFrame(rows)


_SCORE_COL_FORMATS = {
    "Season": "{:.2f}", "Week1": "{:.2f}", "Fixture": "{:.2f}",
    "Score": "{:.2f}", "Exp Pts/GW": "{:.2f}", "GW1 Proj": "{:.1f}",
}


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_initial_squad_optimizer_page():
    st.title("🆕 Initial Squad Optimizer")
    st.caption(
        f"Build your season-opening 15-man squad by trading **season-long value "
        f"({DEFAULT_W_SEASON:.0%})** against a **fast start ({DEFAULT_W_OPENING:.0%})** "
        "— the latter combining the GW1 projection with opening-fixture ease. "
        "Players are ranked by expected points per gameweek, so a premium has to "
        "out-score its price tag, and captaincy is part of that maths rather than "
        "assigned after the fact."
    )

    current_gw = get_current_gameweek() or 1

    st.markdown("### Settings")
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.number_input(
            "Total Budget (£m)", min_value=80.0, max_value=120.0,
            value=100.0, step=0.1, format="%.1f",
        )
    with col2:
        horizon = st.slider(
            "Opening Fixture Horizon (GWs)", min_value=1, max_value=5, value=3,
            help="How many opening gameweeks count toward the Fixture Score",
        )
    with col3:
        formation = st.selectbox(
            "Formation",
            ["auto", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
            index=0,
            help="Auto lets the optimizer choose the best formation",
        )

    with st.expander("Advanced — Scoring Weights & Filters", expanded=False):
        st.caption(
            f"One trade-off: season-long quality versus a fast start. Defaults to "
            f"{DEFAULT_W_SEASON:.0%}/{DEFAULT_W_OPENING:.0%} — a season-opening squad is a "
            "hold-for-many-weeks decision, so pedigree should lead. You shouldn't "
            "need to touch this."
        )
        # The two weights are a single split, so they are bound to each other:
        # moving one drives the other down to keep the total at 100%. The
        # alternative -- two free sliders normalized after the fact -- silently
        # rescales whatever you type, so 70/70 becomes 50/50 and the numbers on
        # screen stop meaning what they say.
        _init_weight_state()
        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.slider(
                "Season-Long Weight (%)", 0, 100, step=5,
                key=_W_SEASON_KEY, on_change=_sync_weight_from_season,
                help="Rotowire's season-long Top 400 — value over all 38 "
                     "gameweeks. Leads by default.",
            )
        with wcol2:
            st.slider(
                "Fast Start Weight (%)", 0, 100, step=5,
                key=_W_OPENING_KEY, on_change=_sync_weight_from_opening,
                help="Rotowire's GW1 projection, scaled by how kind the opening "
                     "fixtures are over your chosen horizon. Raise it to "
                     "prioritise points now over points across the season.",
            )
        w_season_pct = st.session_state[_W_SEASON_KEY]
        w_opening_pct = st.session_state[_W_OPENING_KEY]
        st.caption(
            f"Split: **{w_season_pct}% season-long / {w_opening_pct}% fast start** "
            "— the two always total 100%. Fast start = GW1 projection adjusted "
            f"by opening fixtures (±{FIXTURE_TILT:.0%} per FDR point away from average)."
        )
        w_season = w_season_pct / 100.0
        w_opening = w_opening_pct / 100.0


        st.markdown("---")
        bench_weight = st.slider(
            "Bench Weight", min_value=0.0, max_value=0.5,
            value=DEFAULT_BENCH_WEIGHT, step=0.05,
            help="How much bench players' score counts toward the objective. "
                 "Well below Wildcard's default — this is a starting point, not a "
                 "final squad, and budget spent on the bench is budget missing "
                 "from the XI. Raise it if you want a Bench-Boost-ready squad.",
        )
        st.caption(
            f"**Captaincy is fixed, not adjustable.** Your captain scores "
            f"{CAPTAIN_MULTIPLIER:.0f}x, so the optimizer counts your best player "
            f"twice over when choosing the squad — that is the rule, not a "
            f"preference. It values him at **{CAPTAIN_EFFECTIVE_MULTIPLIER:.2f}x** "
            f"rather than exactly {CAPTAIN_MULTIPLIER:.0f}x because Triple Captain "
            f"can be played twice a season, turning two of his gameweeks into 3x."
        )

        st.markdown("#### Player Filters")
        exclude_dead_weight = st.checkbox(
            "Avoid players with no points appeal", value=True,
            help="Skip players who appear in neither the season rankings nor the "
                 "GW1 table — there is no evidence they will ever score. Bench "
                 "Boost makes all four bench slots live, and a ranked "
                 "replacement costs the same at DEF/MID/FWD and £0.5m more at GK. "
                 "Separate from Bench Weight, which asks how much bench points "
                 "count rather than whether a slot can score at all.",
        )
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            exclude_injured = st.checkbox("Exclude injured/doubtful players", value=True)
        with fcol2:
            min_chance = st.slider(
                "Min chance of playing (%)", min_value=0, max_value=100, value=75, step=25,
                disabled=not exclude_injured,
            )

    season_url, gw1_url = _data_source_urls()

    st.markdown("---")

    if st.button("Build Optimal Squad", type="primary"):
        with st.spinner("Loading player data, projections, and fixture data..."):
            bootstrap = get_classic_bootstrap_static()
            if not bootstrap:
                show_api_error("loading player data for the Initial Squad Optimizer")
                return

            gw1_projections_df = pd.DataFrame()
            gw1_error = None
            if gw1_url:
                try:
                    gw1_projections_df = get_rotowire_player_projections(gw1_url)
                except Exception as e:
                    gw1_error = str(e)
            else:
                gw1_error = "no URL configured"

            season_error = None
            try:
                season_rankings_df = get_rotowire_season_rankings(season_url) \
                    if season_url else pd.DataFrame()
                if not season_url:
                    season_error = "no URL configured"
            except Exception as e:
                season_error = str(e)
                season_rankings_df = pd.DataFrame()

            try:
                ffp_df = get_ffp_projections_data()
            except Exception:
                ffp_df = None

            try:
                _, _, fdr_avg = get_fixture_difficulty_grid(weeks=horizon)
            except Exception as e:
                st.warning(f"Could not load fixture difficulty data: {e}")
                fdr_avg = pd.Series(dtype=float)

            full_pool = _build_full_player_pool(bootstrap)
            if full_pool.empty:
                st.error("No players available.")
                return

            merge_stats = {}
            scored_pool = _compute_scores(
                full_pool, gw1_projections_df, season_rankings_df, ffp_df, fdr_avg,
                current_gw, w_season, w_opening, stats=merge_stats,
            )
            candidate_pool = _apply_eligibility_filters(
                scored_pool, exclude_injured=exclude_injured,
                min_chance_of_playing=min_chance,
                exclude_dead_weight=exclude_dead_weight,
            )

            if candidate_pool.empty:
                st.error("No players available after filtering.")
                return

        season_stats = merge_stats.get("season", {})
        gw1_stats = merge_stats.get("gw1", {})
        ffp_rows = 0 if ffp_df is None else len(ffp_df)
        ffp_note = ""
        if ffp_rows and "FFP_Predicted" in scored_pool.columns \
                and not scored_pool["FFP_Predicted"].notna().any():
            # Expected preseason: FFP publishes Predicted only once GW1 is close,
            # so the Week 1 projection is Rotowire-only until then.
            ffp_note = "Predicted not published yet"

        _render_source_status([
            _source_status_row(
                "Rotowire Season Rankings", f"Season value — {w_season:.0%} of score",
                len(season_rankings_df), season_stats.get("matched"),
                ok=season_error is None and not season_rankings_df.empty,
                updated=get_rotowire_article_updated(season_url)),
            _source_status_row(
                "Rotowire GW1 Rankings", f"Fast start — {w_opening:.0%} of score",
                len(gw1_projections_df), gw1_stats.get("matched"),
                ok=gw1_error is None and not gw1_projections_df.empty,
                updated=get_rotowire_article_updated(gw1_url)),
            # FFP is a live Google Sheet with no published revision time.
            _source_status_row(
                "FFP Points Predictor", "Start likelihood",
                ffp_rows, None, ok=ffp_rows > 0, note=ffp_note, show_updated=False),
        ])
        for label, err in (("season rankings", season_error), ("GW1 rankings", gw1_error)):
            if err:
                st.warning(f"Could not load Rotowire {label}: {err}")
        if ffp_note:
            st.caption(
                "ℹ️ FFP hasn't published GW1 point predictions yet, so the Week 1 "
                "component is Rotowire-only. FFP start percentages are still applied."
            )
        st.caption(f"Candidate pool: {len(candidate_pool)} players")

        with st.spinner("Running optimization..."):
            squad_df, totals = solve_squad_ilp(
                candidate_pool,
                budget,
                score_col="ExpPts",
                formation=formation,
                bench_weight=bench_weight,
                captain_score_col="CapPts",
                captain_bonus_weight=CAPTAIN_BONUS_WEIGHT,
                problem_name="FPL_Initial_Squad_Optimizer",
            )

            if squad_df is None:
                st.error(
                    "Optimization failed. Try increasing your budget or changing the formation. "
                    "The constraints may be too restrictive."
                )
                return

        st.success("Squad built!")

        starters = squad_df[squad_df["Is_Starter"]].copy()
        bench = squad_df[~squad_df["Is_Starter"]].copy()
        bench_gk = bench[bench["Position"] == "G"]
        bench_outfield = bench[bench["Position"] != "G"].sort_values("ExpPts", ascending=False)
        bench_ordered = pd.concat([bench_gk, bench_outfield])
        bench_ordered["Bench_Order"] = range(1, len(bench_ordered) + 1)

        if "Is_Captain" in squad_df.columns and squad_df["Is_Captain"].any():
            cap_name = squad_df.loc[squad_df["Is_Captain"], "Display_Name"].iloc[0]
        else:
            cap_name = starters.loc[starters["CapPts"].idxmax(), "Display_Name"]

        total_cost = squad_df["Price"].sum()
        remaining = budget - total_cost
        xi_exp_pts = starters["ExpPts"].sum()

        issues = check_initial_squad(squad_df, budget)
        if issues:
            _logger.warning(format_issues(issues))
            for issue in issues:
                if issue.severity == "error":
                    st.error(f"⚠️ {issue.message}. {issue.hint}")

        st.markdown("### Squad Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                _score_card("Starting XI", f"{xi_exp_pts:.1f} pts/GW"), unsafe_allow_html=True)
        with col2:
            st.markdown(_score_card("Captain", cap_name, accent="#f0c419"), unsafe_allow_html=True)
        with col3:
            st.markdown(_score_card("Total Cost", _format_money(total_cost), accent="#e0e0e0"), unsafe_allow_html=True)
        with col4:
            budget_color = "#00ff87" if remaining >= 0 else "#f87171"
            st.markdown(_score_card("Remaining Budget", _format_money(remaining), accent=budget_color), unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### Starting XI")
        formation_str = (
            f"{starters[starters['Position'] == 'D'].shape[0]}-"
            f"{starters[starters['Position'] == 'M'].shape[0]}-"
            f"{starters[starters['Position'] == 'F'].shape[0]}"
        )
        st.caption(f"Formation: {formation_str} | Captain: {cap_name}")

        # Colour both tables by where each player sits among others in their
        # own position across the whole pool — not among the 15 shown. Scaled
        # within the squad, the weakest of an optimal XI renders pure red while
        # sitting in the top 5% of the league; scaled across all positions at
        # once, every goalkeeper looks bad for being a goalkeeper.
        starter_ratios = _positional_color_ratios(starters, scored_pool)
        bench_ratios = _positional_color_ratios(bench_ordered, scored_pool)

        render_styled_table(
            _display_rows(starters),
            col_formats=_SCORE_COL_FORMATS,
            positive_color_cols=["Exp Pts/GW"],
            color_values={"Exp Pts/GW": starter_ratios} if starter_ratios else None,
            color_range_overrides={"Exp Pts/GW": (0.0, 1.0)} if starter_ratios else None,
        )

        st.markdown("---")
        st.markdown("### Bench")
        bench_display = _display_rows(bench_ordered)
        bench_display.insert(0, "Order", bench_ordered["Bench_Order"].values)
        render_styled_table(
            bench_display,
            col_formats=_SCORE_COL_FORMATS,
            positive_color_cols=["Exp Pts/GW"],
            color_values={"Exp Pts/GW": bench_ratios} if bench_ratios else None,
            color_range_overrides={"Exp Pts/GW": (0.0, 1.0)} if bench_ratios else None,
        )

        st.markdown("---")
        st.markdown("### Opening Fixture Difficulty (Squad Teams)")
        st.caption(
            "Difficulty of each squad team's opening fixtures, easiest first. "
            "1 = easiest, 5 = hardest."
        )
        try:
            _, fdr_diffs, _ = get_fixture_difficulty_grid(weeks=horizon)
            squad_teams = squad_df["Team"].unique().tolist()
            gw_cols = [f"GW{current_gw + i}" for i in range(horizon)]
            available_cols = [c for c in gw_cols if c in fdr_diffs.columns]
            if available_cols:
                fdr_filtered = fdr_diffs.loc[fdr_diffs.index.isin(squad_teams), available_cols].copy()
                avg_fdr = fdr_filtered.fillna(3).mean(axis=1)
                fdr_filtered = fdr_filtered.loc[avg_fdr.sort_values().index]

                # A single gameweek's FDR is one of five discrete grades, so it
                # renders as an integer. Only the average across gameweeks is a
                # real fraction and keeps a decimal.
                disp = pd.DataFrame(index=fdr_filtered.index)
                disp.insert(0, "Team", fdr_filtered.index)
                for col in available_cols:
                    disp[col] = [
                        "—" if pd.isna(v) else str(int(round(float(v))))
                        for v in fdr_filtered[col]
                    ]
                disp["Avg FDR"] = avg_fdr.loc[fdr_filtered.index].round(1)

                # Shared renderer, so this grid matches the FDR tables elsewhere
                # in the app rather than inventing its own palette.
                st.markdown(
                    style_fixture_difficulty(disp, fdr_filtered),
                    unsafe_allow_html=True,
                )
            else:
                st.info("FDR data not available for the selected horizon.")
        except Exception as e:
            _logger.warning("Squad FDR grid unavailable: %s", e)
            st.info("FDR data not available.")

        st.markdown("---")
        with st.expander("Team Breakdown"):
            team_counts = squad_df.groupby("Team").agg(
                {"Player": "count", "Price": "sum", "Player Score": "sum"}
            ).rename(columns={"Player": "Players", "Price": "Total Cost", "Player Score": "Total Score"})
            team_counts["Total Cost"] = team_counts["Total Cost"].apply(lambda x: f"£{x:.1f}m")
            team_counts["Total Score"] = team_counts["Total Score"].round(2)
            render_styled_table(team_counts.reset_index())

        with st.expander("Position Breakdown"):
            pos_breakdown = squad_df.groupby("Position").agg(
                {"Player": "count", "Price": "sum", "Player Score": "sum"}
            ).rename(columns={"Player": "Count", "Price": "Total Cost", "Player Score": "Total Score"})
            pos_breakdown["Total Cost"] = pos_breakdown["Total Cost"].apply(lambda x: f"£{x:.1f}m")
            pos_breakdown["Total Score"] = pos_breakdown["Total Score"].round(2)
            render_styled_table(pos_breakdown.reset_index())

        st.markdown("---")
        st.info(
            "**Tips:**\n"
            "- Season/Week1/Fixture scores are positional percentiles (0-1) — 0.85 means top 15% at that position\n"
            "- The captain is chosen by the optimizer itself, not picked afterward — a higher Captain Bonus Weight "
            "rewards owning one standout season-long performer over several merely-good players\n"
            "- Raise the Week 1 weight if you want the squad to lean harder into a strong opening fixture run\n"
            "- Transfers can reshape this squad as the season develops — treat this as a strong starting point, not a final answer"
        )
