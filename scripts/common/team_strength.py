"""
Draft team power rankings — roster strength scoring.

Answers "how good is each roster in my league, right now?" with a Team Score in
0-100 plus four positional scores, for every team in a Draft league.

Design (see the plan and CLAUDE.md's "Transfer Scoring Model" for the philosophy):

* Every input is a **positional percentile against the full FPL player pool**, so a
  score of 85 means "top 15% at this position" regardless of position, and a league
  of strong teams can legitimately all score 70+.
* Long-term **quality** dominates.  It blends preseason pedigree (Rotowire Top-400)
  with in-season actuals, shifting toward actuals via ``season_progress_weight()``.
* Actuals use **points per start**, not total points — so a star returning from six
  weeks out is not punished for the games he missed.  Below a minimum start count the
  sample is too small to trust, and pedigree is used alone.
* An **injury discount** scales by the fraction of the *remaining* season a player
  will miss, so the same absence costs more in April than in August.

Draft enforces a 2 GK / 5 DEF / 5 MID / 3 FWD squad, so every roster has the same
shape and a flat mean across all 15 is directly comparable between teams.

The two ``compute_*``/``aggregate_*`` functions are pure (no Streamlit, no network)
and unit-testable offline.  Only ``build_league_strength()`` touches the wire.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from scripts.common.analytics import (
    blend_fixture_projections,
    merge_season_projections,
    positional_percentile,
    season_progress_weight,
)
from scripts.common.error_helpers import get_logger
from scripts.common.injury_helpers import estimate_games_to_miss, injury_multiplier
from scripts.common.text_helpers import _map_position_to_rw

_logger = get_logger("fpl_app.team_strength")

# --- Model constants (all tuning lives here) ---------------------------------

#: Contribution of each signal to a player's pre-injury strength.  Must sum to 1.0.
_STRENGTH_WEIGHTS = {
    "quality": 0.60,        # pedigree <-> actuals blend (the dominant signal)
    "gw_proj": 0.25,        # this gameweek's blended projection
    "start_security": 0.10, # nailed-on starter vs. rotation risk
    "fixture_ease": 0.05,   # upcoming fixture difficulty (supplementary)
}

#: Below this many starts, points-per-start is too noisy to trust — use pedigree alone.
_MIN_STARTS_FOR_ACTUALS = 4

#: Split of the actuals signal between points-per-start and recent form.
_ACTUALS_FORM_WEIGHT = 0.30

#: Reference-pool minutes filter, so never-played deadwood doesn't inflate percentiles.
#: Disabled very early in the season when nobody has accumulated minutes yet.
_REF_MIN_MINUTES = 90
_MIN_MINUTES_FROM_GW = 4

#: Fixture-difficulty horizon, in gameweeks.
_FDR_HORIZON = 5

_POS_ORDER = ["G", "D", "M", "F"]
_POS_LABELS = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}
_SQUAD_SIZE = 15


# =============================================================================
# PURE SCORING
# =============================================================================

def compute_player_strength(
    roster_df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    current_gw: int,
) -> pd.DataFrame:
    """Score each rostered player's strength in [0, 1].

    ``roster_df`` and ``reference_df`` must share a schema — ``reference_df`` is the
    full FPL pool that percentiles are taken against, and ``roster_df`` is normally a
    subset of it.  Both are produced by :func:`_build_reference_pool`.

    Adds: ``Pedigree``, ``PPS``, ``Actuals``, ``Quality``, ``GW_Proj_Pctile``,
    ``Start_Security``, ``Fixture_Ease``, ``Raw_Strength``, ``GWs_Missed``,
    ``Injury_Mult``, ``Player_Strength``.
    """
    result = roster_df.copy()
    if result.empty:
        for col in ("Pedigree", "PPS", "Actuals", "Quality", "GW_Proj_Pctile",
                    "Start_Security", "Fixture_Ease", "Raw_Strength",
                    "GWs_Missed", "Injury_Mult", "Player_Strength"):
            result[col] = pd.Series(dtype=float)
        return result

    ref = reference_df if reference_df is not None else result
    min_minutes = _REF_MIN_MINUTES if int(current_gw) >= _MIN_MINUTES_FROM_GW else 0

    # --- Preseason pedigree ---------------------------------------------------
    # If the Rotowire season rankings never merged, fall back to season points so the
    # page degrades rather than reporting a flat 0.5 for everybody.
    pedigree_col = "SeasonProjection"
    if pedigree_col not in ref.columns or not pd.to_numeric(
        ref[pedigree_col], errors="coerce"
    ).notna().any():
        _logger.warning("No season projections available; falling back to total_points for pedigree")
        pedigree_col = "total_points"
    result["Pedigree"] = positional_percentile(result, ref, pedigree_col)

    # --- Actuals: points per START, plus recent form ---------------------------
    result["PPS"] = _points_per_start(result)
    pps_pctile = positional_percentile(result, ref, "PPS", min_minutes=min_minutes)

    # Preseason, every player's form is 0.0, so its percentile is 0.0 for everyone —
    # blending that in would drag all actuals down by a flat 30% rather than
    # discriminating between players.  Fall back to points-per-start alone.
    if _has_signal(ref, "form"):
        form_pctile = positional_percentile(result, ref, "form", min_minutes=min_minutes)
        actuals = ((1.0 - _ACTUALS_FORM_WEIGHT) * pps_pctile
                   + _ACTUALS_FORM_WEIGHT * form_pctile)
    else:
        _logger.info("Form is flat across the pool (preseason); actuals use points-per-start alone")
        actuals = pps_pctile

    # Minimum-starts floor: too small a sample, so lean entirely on pedigree.
    starts = pd.to_numeric(result.get("starts"), errors="coerce").fillna(0)
    thin_sample = starts < _MIN_STARTS_FOR_ACTUALS
    actuals = actuals.where(~thin_sample, result["Pedigree"])
    result["Actuals"] = actuals

    # --- Blend pedigree -> actuals as the season progresses --------------------
    p = season_progress_weight(int(current_gw))
    result["Quality"] = p * result["Actuals"] + (1.0 - p) * result["Pedigree"]

    # --- Supporting signals ----------------------------------------------------
    result["GW_Proj_Pctile"] = positional_percentile(result, ref, "Proj_Blended")
    result["Start_Security"] = positional_percentile(
        result, ref, "Start_Security_Raw", min_minutes=min_minutes
    )
    result["Fixture_Ease"] = positional_percentile(result, ref, "Fixture_Ease_Raw")

    # --- Combine ---------------------------------------------------------------
    w = _STRENGTH_WEIGHTS
    result["Raw_Strength"] = (
        w["quality"] * result["Quality"]
        + w["gw_proj"] * result["GW_Proj_Pctile"]
        + w["start_security"] * result["Start_Security"]
        + w["fixture_ease"] * result["Fixture_Ease"]
    ).clip(0.0, 1.0)

    # --- Injury discount (season-aware) ----------------------------------------
    result["GWs_Missed"] = [
        estimate_games_to_miss(
            row.get("news"), row.get("chance_of_playing_next_round"), row.get("status")
        )
        for _, row in result.iterrows()
    ]
    result["Injury_Mult"] = [
        injury_multiplier(g, current_gw) for g in result["GWs_Missed"]
    ]
    result["Player_Strength"] = result["Raw_Strength"] * result["Injury_Mult"]

    return result


def attach_reference_pps(pool: pd.DataFrame) -> pd.DataFrame:
    """Add a ``PPS`` column suitable for use as a percentile *reference*.

    Two things matter here and both have bitten:

    1. The column must exist on the reference pool at all.  ``positional_percentile``
       silently falls back to min-max over the input frame when the reference lacks
       the column, which turns "percentile against ~700 FPL players" into "min-max
       across these 15", with no error.
    2. Players below the start threshold are set to NaN, not 0, so they drop out of
       the denominator (``positional_percentile`` dropna()s the reference).  A player
       with 35 points from 2 starts scores 17.5/start and would otherwise outrank
       Haaland, dragging every legitimate star's percentile down.
    """
    result = pool.copy()
    result["PPS"] = _points_per_start(result)
    starts = pd.to_numeric(result.get("starts"), errors="coerce").fillna(0)
    result.loc[starts < _MIN_STARTS_FOR_ACTUALS, "PPS"] = np.nan
    return result


def _has_signal(df: Optional[pd.DataFrame], col: str) -> bool:
    """True if *col* varies at all — a flat column carries no ranking information."""
    if df is None or col not in df.columns:
        return False
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return not vals.empty and float(vals.max()) > float(vals.min())


def _points_per_start(df: pd.DataFrame) -> pd.Series:
    """Total points per start.  Zero starts yields 0.0 rather than an infinity."""
    pts = pd.to_numeric(df.get("total_points"), errors="coerce").fillna(0.0)
    starts = pd.to_numeric(df.get("starts"), errors="coerce").fillna(0.0)
    pps = pd.Series(0.0, index=df.index, dtype=float)
    played = starts > 0
    pps[played] = pts[played] / starts[played]
    return pps.replace([np.inf, -np.inf], 0.0)


def aggregate_team_strength(player_df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-player strength up into one row per team.

    Team Score is a flat mean across the whole 15-player squad; each positional
    score is a flat mean across that position's players.  ``Injury_Cost`` is the
    gap between the injury-free and injury-adjusted scores.
    """
    cols = (["Team_ID", "Team_Name", "Score", "Healthy_Score", "Injury_Cost", "Players"]
            + [_POS_LABELS[p] for p in _POS_ORDER]
            + [f"{_POS_LABELS[p]}_Rank" for p in _POS_ORDER]
            + ["Rank"])
    if player_df.empty or "Team_ID" not in player_df.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    for team_id, grp in player_df.groupby("Team_ID"):
        row = {
            "Team_ID": team_id,
            "Team_Name": grp["Team_Name"].iloc[0],
            "Score": 100.0 * grp["Player_Strength"].mean(),
            "Healthy_Score": 100.0 * grp["Raw_Strength"].mean(),
            "Players": int(len(grp)),
        }
        row["Injury_Cost"] = row["Healthy_Score"] - row["Score"]
        for pos in _POS_ORDER:
            at_pos = grp[grp["Position"] == pos]
            label = _POS_LABELS[pos]
            row[label] = 100.0 * at_pos["Player_Strength"].mean() if not at_pos.empty else np.nan
        rows.append(row)

    team_df = pd.DataFrame(rows)

    # League ranks: 1 = strongest.  Ties share the lower (better) rank.
    for pos in _POS_ORDER:
        label = _POS_LABELS[pos]
        team_df[f"{label}_Rank"] = team_df[label].rank(ascending=False, method="min").astype("Int64")
    team_df["Rank"] = team_df["Score"].rank(ascending=False, method="min").astype("Int64")

    team_df = team_df.sort_values("Score", ascending=False).reset_index(drop=True)
    return team_df[cols]


# =============================================================================
# DATA ASSEMBLY (impure)
# =============================================================================

def _build_reference_pool(current_gw: int) -> pd.DataFrame:
    """Build the full ~700-player FPL pool with every scoring input attached."""
    from scripts.common.fixture_helpers import get_fixture_difficulty_grid
    from scripts.common.fpl_draft_api import pull_fpl_player_stats
    from scripts.common.scraping import (
        get_ffp_projections_data,
        get_rotowire_player_projections,
        get_rotowire_season_rankings,
    )
    import config

    stats = pull_fpl_player_stats()
    if stats is None or stats.empty:
        _logger.warning("pull_fpl_player_stats returned no rows")
        return pd.DataFrame()

    pool = pd.DataFrame({
        "Player_ID": pd.to_numeric(stats["id"], errors="coerce"),
        "Player": stats["player"],
        "Web_Name": stats["web_name"],
        "Team": stats["team_name_abbrv"],
        # CRITICAL: bootstrap gives GKP/DEF/MID/FWD but every analytics function
        # groups on G/D/M/F.  Skip this and positional grouping silently returns 0.5.
        "Position": stats["position_abbrv"].apply(_map_position_to_rw),
        "total_points": pd.to_numeric(stats["total_points"], errors="coerce").fillna(0),
        "starts": pd.to_numeric(stats["starts"], errors="coerce").fillna(0),
        "minutes": pd.to_numeric(stats["minutes"], errors="coerce").fillna(0),
        "form": pd.to_numeric(stats["form"], errors="coerce").fillna(0),
        "status": stats["status"],
        "news": stats["news"],
        "chance_of_playing_next_round": stats["chance_of_playing_next_round"],
    })

    # --- Preseason pedigree (Rotowire Top-400).  This scraper RAISES on failure. ---
    try:
        rankings = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
    except Exception as e:
        _logger.warning("Rotowire season rankings unavailable: %s", e)
        rankings = pd.DataFrame()
    pool = merge_season_projections(pool, rankings, output_col="SeasonProjection")

    # --- This gameweek's projection (Rotowire weekly, blended 60/40 with FFP) ---
    try:
        weekly = get_rotowire_player_projections(config.ROTOWIRE_URL)
    except Exception as e:
        _logger.warning("Rotowire weekly projections unavailable: %s", e)
        weekly = pd.DataFrame()
    pool = merge_season_projections(pool, weekly, output_col="Points")
    pool["Points"] = pd.to_numeric(pool["Points"], errors="coerce").fillna(0)

    try:
        ffp = get_ffp_projections_data()
    except Exception as e:
        _logger.warning("FFP projections unavailable: %s", e)
        ffp = None
    # Also attaches FFP_Predicted / FFP_Start / FFP_LongStart.
    pool = blend_fixture_projections(pool, ffp, rotowire_col="Points")

    # --- Points per start, for the actuals percentile -------------------------
    # Computed on the pool (not just on rosters) so percentiles are genuinely taken
    # against the full FPL player pool.  Without this column here,
    # positional_percentile() silently falls back to min-max over the roster alone.
    #
    # Sub-threshold players are set to NaN rather than 0 so they drop out of the
    # reference denominator entirely: positional_percentile() dropna()s the pool.
    # Otherwise a 2-start fluke (35 points from 2 starts = 17.5/start) outranks
    # Haaland and deflates every legitimate star's percentile.
    pool = attach_reference_pps(pool)

    # --- Start security: FFP LongStart%, falling back to starts per GW played ---
    gws_played = max(1, int(current_gw) - 1)
    long_start = pd.to_numeric(pool.get("FFP_LongStart"), errors="coerce")
    fallback = (pool["starts"] / gws_played).clip(0, 1) * 100.0
    pool["Start_Security_Raw"] = long_start.fillna(fallback)

    # --- Fixture ease: negated FDR so higher is better -------------------------
    try:
        _, _, fdr_avg = get_fixture_difficulty_grid(weeks=_FDR_HORIZON)
    except Exception as e:
        _logger.warning("Fixture difficulty grid unavailable: %s", e)
        fdr_avg = pd.Series(dtype=float)
    pool["Team_AvgFDR"] = pool["Team"].map(fdr_avg).fillna(3.0)
    pool["Fixture_Ease_Raw"] = -pd.to_numeric(pool["Team_AvgFDR"], errors="coerce").fillna(3.0)

    return pool


@st.cache_data(ttl=600)
def build_league_strength(league_id, current_gw: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute power rankings for a Draft league.

    Returns ``(team_df, player_df)``.  Both are empty if the league has not drafted
    yet or upstream data is unavailable — callers should show an info message.
    """
    from scripts.common.fpl_draft_api import get_league_rosters_with_ids

    rosters = get_league_rosters_with_ids(league_id)
    if not rosters:
        _logger.info("No rosters for league %s (pre-draft or API unavailable)", league_id)
        return pd.DataFrame(), pd.DataFrame()

    pool = _build_reference_pool(current_gw)
    if pool.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Exact join on integer element ID — no name matching anywhere in this path.
    by_id = pool.set_index("Player_ID", drop=False)
    frames = []
    for team_id, blob in rosters.items():
        ids = [pid for pid in blob["player_ids"] if pid in by_id.index]
        missing = len(blob["player_ids"]) - len(ids)
        if missing:
            _logger.warning("Team %s: %d rostered element(s) absent from bootstrap", team_id, missing)
        if not ids:
            continue
        squad = by_id.loc[ids].copy()
        squad["Team_ID"] = team_id
        squad["Team_Name"] = blob["team_name"]
        frames.append(squad)

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    roster_df = pd.concat(frames, ignore_index=True)
    player_df = compute_player_strength(roster_df, pool, current_gw)
    team_df = aggregate_team_strength(player_df)
    return team_df, player_df


def describe_blend(current_gw: int) -> str:
    """Human-readable statement of where the pedigree/actuals blend currently sits."""
    p = season_progress_weight(int(current_gw))
    return f"GW{int(current_gw)} — {p * 100:.0f}% actuals / {(1 - p) * 100:.0f}% preseason pedigree"
