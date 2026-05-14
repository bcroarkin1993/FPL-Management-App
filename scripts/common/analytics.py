"""
Player Analytics Functions.

FDR/form enrichment, availability penalties, multi-GW projections,
positional depth analysis, positional percentile normalization,
and related analytics.
"""

import requests
from dataclasses import dataclass
from typing import Optional, Callable, Dict

import numpy as np
import pandas as pd
import streamlit as st

import config
from scripts.common.error_helpers import get_logger
from scripts.common.player_matching import canonical_normalize
from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

_logger = get_logger("fpl_app.analytics")

# Position code → short label for display
_POS_LABELS = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}


# =============================================================================
# SEASON PROGRESS — TIME-DECAY BLEND
# =============================================================================
# Early season: trust Rotowire season projections more (small sample of actuals).
# Late season: trust actual season points more (projections are stale).
# Uses a concave curve (power < 1) so actuals are trusted faster mid-season.

def season_progress_weight(current_gw: int, total_gws: int = 38) -> float:
    """Return weight for *actual* season points vs season projections.

    Uses a concave curve so actuals are trusted faster than linear:
      - GW  1 → 0.10  (mostly projection)
      - GW 10 → 0.40  (projections waning)
      - GW 19 → 0.58  (actuals lead)
      - GW 30 → 0.82  (actuals dominate, projection ~18%)
      - GW 38 → 0.95  (projection nearly irrelevant)

    The floor/ceiling (0.10/0.95) ensures neither signal is fully ignored.
    """
    raw = current_gw / max(total_gws, 1)
    curved = raw ** 0.7  # concave: faster rise early/mid-season
    return max(0.10, min(0.95, curved))


# =============================================================================
# FORM DAMPENING
# =============================================================================

def dampen_form_by_starts(form_norm: pd.Series, starts: pd.Series,
                          min_starts: int = 5, floor: float = 0.2) -> pd.Series:
    """Blend form_norm toward 0.5 (neutral) for players with few starts.

    Players with a small sample size of starts have unreliable form.
    This function dampens their form toward neutral (0.5) proportionally
    to how far below min_starts they are.

    confidence = clip(starts / min_starts, floor, 1.0)
    dampened = confidence * form_norm + (1 - confidence) * 0.5

    Args:
        form_norm: Already-normalized form values in [0, 1].
        starts:    Number of starts for each player (NaN treated as 0).
        min_starts: Starts needed for full confidence (default 5).
        floor:     Minimum confidence even with 0 starts (default 0.2).

    Returns:
        pd.Series of dampened form values in [0, 1].
    """
    s = pd.to_numeric(starts, errors="coerce").fillna(0)
    f = pd.to_numeric(form_norm, errors="coerce").fillna(0.5)
    confidence = (s / max(min_starts, 1)).clip(lower=floor, upper=1.0)
    return confidence * f + (1 - confidence) * 0.5


# =============================================================================
# POSITIONAL PERCENTILE NORMALIZATION
# =============================================================================

def _min_max_norm_series(series: pd.Series) -> pd.Series:
    """Min-max normalization to [0,1]. Safe for constants/NaN."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)


def enrich_reference_with_projections(
    fpl_stats: pd.DataFrame,
    projections_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge Rotowire projection values onto the reference pool for 1GW scoring.

    ``compute_player_scores()`` compares each player's single-GW projection
    against the reference pool.  Without projections in the reference, the
    comparison is projections-vs-ppg (scale mismatch).  Merging projections
    into the reference pool enables apples-to-apples comparison.

    Classic's ``all_players`` already has ``Projected_Points`` from its own
    merge step, so this helper is primarily needed for the Draft pipeline
    where ``fpl_stats`` lacks Rotowire projections.
    """
    if projections_df is None or projections_df.empty:
        return fpl_stats
    # Don't overwrite if already present
    if "Projected_Points" in fpl_stats.columns and fpl_stats["Projected_Points"].notna().any():
        return fpl_stats

    result = fpl_stats.copy()
    proj_slim = projections_df[["Player", "Team", "Position", "Points"]].copy()
    proj_slim.rename(columns={"Points": "Projected_Points"}, inplace=True)
    proj_slim["_cn"] = proj_slim["Player"].apply(canonical_normalize)
    result["_cn"] = result["Player"].apply(canonical_normalize)

    result = result.merge(
        proj_slim[["_cn", "Team", "Position", "Projected_Points"]],
        on=["_cn", "Team", "Position"],
        how="left",
    )
    result.drop(columns=["_cn"], inplace=True)
    return result


def positional_percentile(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    value_col: str,
    position_col: str = "Position",
    ref_value_col: Optional[str] = None,
    min_minutes: int = 0,
) -> pd.Series:
    """
    Compute within-position percentile for each player in *df* using *reference_df*
    as the full player pool.

    For each player, the percentile is the fraction of same-position players in the
    reference pool with a **strictly lower** value.  Result is in [0, 1] where
    1.0 = best at the position.

    Args:
        df:            Squad / candidate DataFrame containing value_col and position_col.
        reference_df:  Full FPL player pool (~700 players) with position_col and the
                       value column. If None or missing the column, falls back to
                       global min-max normalization on *df* alone.
        value_col:     Column name in *df* to percentile-normalize.
        position_col:  Column containing position codes (G/D/M/F).
        ref_value_col: Column name in *reference_df* if it differs from value_col.
                       E.g. roster has "Season_Points", reference has "total_points".
        min_minutes:   Filter reference players with fewer than this many minutes
                       (avoids 0-minute players distorting percentiles).  Ignored
                       if reference_df lacks a "minutes" column.

    Returns:
        pd.Series aligned to df.index with values in [0, 1].
    """
    ref_col = ref_value_col or value_col

    # Fallback: no usable reference
    if reference_df is None or ref_col not in reference_df.columns:
        return _min_max_norm_series(df[value_col])

    if position_col not in reference_df.columns or position_col not in df.columns:
        return _min_max_norm_series(df[value_col])

    # Prepare reference pool
    ref = reference_df.copy()
    ref[ref_col] = pd.to_numeric(ref[ref_col], errors="coerce")
    if min_minutes > 0 and "minutes" in ref.columns:
        ref = ref[pd.to_numeric(ref["minutes"], errors="coerce").fillna(0) >= min_minutes]

    # Build per-position value arrays
    pos_values: Dict[str, np.ndarray] = {}
    for pos, grp in ref.groupby(position_col):
        vals = grp[ref_col].dropna().values
        if len(vals) > 0:
            pos_values[str(pos)] = vals

    # Compute percentile for each player
    player_vals = pd.to_numeric(df[value_col], errors="coerce")
    result = pd.Series(0.5, index=df.index, dtype=float)

    for idx in df.index:
        pos = str(df.at[idx, position_col])
        val = player_vals.at[idx]
        if pd.isna(val) or pos not in pos_values:
            continue
        pool = pos_values[pos]
        n_below = np.sum(pool < val)
        result.at[idx] = float(n_below) / len(pool)

    return result


def compute_dynamic_alpha(
    position: str,
    ros_score: float,
    format_context: str = "draft",
    depth_map: Optional[Dict[str, "PositionalDepth"]] = None,
    current_gw: int = 0,
) -> float:
    """Compute the dynamic 1GW/ROS blend weight (alpha) for Transfer/Keep Score.

    Score = alpha * 1GW + (1 - alpha) * ROS

    Alpha is adjusted by:
    1. Format baseline: Draft=0.35 (ROS lean), Classic=0.55 (slight 1GW lean)
    2. Position: GK -0.10, FWD -0.05 (more ROS-oriented)
    3. Rank tier (player's ROS): Elite(>0.80) -0.10, Above avg(>0.60) -0.05, Below(<0.40) +0.05
    4. Squad depth: Critical +0.15, Low +0.10 (urgency → favor 1GW)
    5. Late season (GW35+): progressive +0.05/GW boost so 1GW dominates when ROS horizon shrinks
    6. Clamped to [0.15, 0.75]

    Args:
        position: Position code (G/D/M/F).
        ros_score: Player's ROS score (0-1).
        format_context: "draft" or "classic".
        depth_map: Positional depth map from compute_positional_depth().
        current_gw: Current gameweek (0 = unknown, no late-season adjustment).

    Returns:
        Alpha value in [0.15, 0.75].
    """
    # 1. Format baseline
    alpha = 0.35 if format_context == "draft" else 0.55

    # 2. Position adjustment
    if position == "G":
        alpha -= 0.10
    elif position == "F":
        alpha -= 0.05

    # 3. Rank tier adjustment (elite players → protect with more ROS weight)
    if ros_score > 0.80:
        alpha -= 0.10
    elif ros_score > 0.60:
        alpha -= 0.05
    elif ros_score < 0.40:
        alpha += 0.05

    # 4. Squad depth adjustment (low depth → urgency → favor 1GW)
    if depth_map is not None:
        depth = depth_map.get(position)
        if depth is not None:
            if depth.depth_level == "Critical":
                alpha += 0.15
            elif depth.depth_level == "Low":
                alpha += 0.10

    # 5. Late-season boost: as GWs remaining shrink, ROS becomes less informative.
    # At GW35: +0.10, GW36: +0.20, GW37: +0.30, GW38: +0.30 (capped).
    if current_gw >= 35:
        alpha += min(0.30, (current_gw - 34) * 0.10)

    # 6. Clamp
    return max(0.15, min(0.75, alpha))


def compute_player_scores(
    df: pd.DataFrame,
    all_players_df: Optional[pd.DataFrame],
    current_gw: int,
    format_context: str = "draft",
    depth_map: Optional[Dict[str, "PositionalDepth"]] = None,
) -> pd.DataFrame:
    """Compute player scores: 1GW, ROS, Transfer Score, and Keep Score.

    Uses positional percentile scoring against the full FPL player pool.
    A score of 0.85 means "top 15% at this position" — immediately interpretable.

    1GW (pure expected value):
        blended_projection = avg(Rotowire, FFP_Predicted) — whichever available
        effective_proj = blended_projection * start_likelihood
        1GW = positional_percentile(effective_proj)

    ROS (multi-GW dominant, dynamic weights):
        season_quality = p * season_pts_pctile + (1-p) * season_proj_pctile
        w_mgw=0.40-0.10p, w_sq=0.30+0.15p, w_form=0.15-0.05p, w_start=0.10, w_fdr=0.05
        ROS = w_mgw*multigw + w_sq*season_quality + w_form*form + w_start*start_consistency + w_fdr*fdr

    Transfer/Keep Score (dynamic alpha blend):
        Score = alpha * 1GW + (1-alpha) * ROS
        Alpha adapts to format, position, player quality, and squad depth.

    Args:
        df: Player DataFrame with columns like Projected_Points/Points, Form,
            Season_Points/total_points, AvgFDRNextN/AvgFDR, MultiGW_Proj,
            SeasonProjection, FFP_Predicted, FFP_Start, FFP_LongStart, starts, Position.
        all_players_df: Full FPL player pool (~700 players) for percentile reference.
        current_gw: Current gameweek number.
        format_context: "draft" or "classic" — affects alpha baseline.
        depth_map: Positional depth map for squad depth adjustment.

    Returns:
        df with '1GW', 'ROS', 'Transfer Score', and 'Keep Score' columns added.
    """
    result = df.copy()

    # --- Resolve column names (Draft vs Classic use different naming) ---
    proj_col = "Projected_Points" if "Projected_Points" in result.columns else "Points"
    season_col = "Season_Points" if "Season_Points" in result.columns else "total_points"
    form_col = "HealthyForm" if "HealthyForm" in result.columns else "Form" if "Form" in result.columns else "form"
    fdr_col = "AvgFDRNextN" if "AvgFDRNextN" in result.columns else "AvgFDR"

    # Coerce numerics
    result[proj_col] = pd.to_numeric(result.get(proj_col), errors="coerce").fillna(0)
    result[season_col] = pd.to_numeric(result.get(season_col), errors="coerce").fillna(0)
    result[form_col] = pd.to_numeric(result.get(form_col), errors="coerce").fillna(0)
    result[fdr_col] = pd.to_numeric(result.get(fdr_col), errors="coerce").fillna(3)

    # Resolve reference column names in all_players_df
    ref_form_col = "form" if (all_players_df is not None and "form" in all_players_df.columns) else form_col
    ref_season_col = "total_points" if (all_players_df is not None and "total_points" in all_players_df.columns) else season_col

    # --- 1GW Score (pure expected value) ---
    # Blend Rotowire projection with FFP Predicted (average if both, whichever if one)
    rotowire_proj = pd.to_numeric(result.get(proj_col), errors="coerce").fillna(0)
    ffp_pred = pd.to_numeric(result.get("FFP_Predicted"), errors="coerce") if "FFP_Predicted" in result.columns else pd.Series(np.nan, index=result.index)

    blended_proj = rotowire_proj.copy()
    # Treat FFP_Predicted <= 0 as missing (FFP publishes 0 when predictions aren't available)
    both_mask = rotowire_proj.gt(0) & ffp_pred.gt(0)
    ffp_only_mask = rotowire_proj.eq(0) & ffp_pred.gt(0)
    blended_proj[both_mask] = (rotowire_proj[both_mask] + ffp_pred[both_mask]) / 2
    blended_proj[ffp_only_mask] = ffp_pred[ffp_only_mask]

    # Start likelihood: FFP_Start (primary), FPL chance_of_playing (fallback), default 100%
    ffp_start = pd.to_numeric(result.get("FFP_Start"), errors="coerce") if "FFP_Start" in result.columns else pd.Series(np.nan, index=result.index)
    fpl_chance = pd.to_numeric(result.get("chance_of_playing_next_round"), errors="coerce") if "chance_of_playing_next_round" in result.columns else pd.Series(np.nan, index=result.index)

    start_likelihood = pd.Series(1.0, index=result.index)
    # Use FFP_Start where available (already a percentage)
    ffp_mask = ffp_start.notna()
    start_likelihood[ffp_mask] = ffp_start[ffp_mask] / 100.0
    # Fallback to FPL chance_of_playing where FFP is missing
    fpl_mask = ~ffp_mask & fpl_chance.notna()
    start_likelihood[fpl_mask] = fpl_chance[fpl_mask] / 100.0
    start_likelihood = start_likelihood.clip(lower=0, upper=1)

    # When Rotowire explicitly projects a player as a starter, apply a position-specific
    # floor to start_likelihood. Rotowire only projects expected starters, so their
    # presence is itself a confidence signal — prevents FFP uncertainty from fully
    # overriding an expert lineup projection. DEF floor is highest: defenders who start
    # play 90 mins; unlike FWD/MID there's no "subbed on late for 2 pts" scenario.
    _ROTOWIRE_START_FLOORS = {"G": 0.80, "D": 0.75, "M": 0.68, "F": 0.65}
    _pos_col = result["Position"] if "Position" in result.columns else pd.Series("M", index=result.index)
    _has_rotowire = rotowire_proj.gt(0)
    for _pos_code, _floor_val in _ROTOWIRE_START_FLOORS.items():
        _pos_mask = _has_rotowire & (_pos_col == _pos_code)
        start_likelihood[_pos_mask] = start_likelihood[_pos_mask].clip(lower=_floor_val)

    # Effective projected points
    result["_effective_proj"] = blended_proj * start_likelihood

    # Detect blank-GW players: 0 projection from ALL sources but not clearly injured/suspended.
    # These are players whose team has no fixture this GW — they deserve neutral 1GW (0.5),
    # not near-0 which would trigger false drop recommendations for elite players.
    # Use a blacklist approach: only exclude players who are clearly injured/suspended/unavailable.
    _status_col = result["status"] if "status" in result.columns else pd.Series(np.nan, index=result.index)
    _chance_col = (
        pd.to_numeric(result["chance_of_playing_next_round"], errors="coerce")
        if "chance_of_playing_next_round" in result.columns
        else pd.Series(np.nan, index=result.index)
    )
    _clearly_unavailable = (
        _status_col.isin(["i", "s", "u"])  # Injured, suspended, unavailable
        | (_chance_col.notna() & _chance_col.lt(50))  # < 50% chance → genuinely doubtful
    )
    blank_gw_mask = (
        (result["_effective_proj"] == 0)
        & (blended_proj == 0)           # No projection from any source
        & ~_clearly_unavailable         # Not injured/suspended/unavailable
    )
    result.loc[blank_gw_mask, "_effective_proj"] = np.nan

    # 1GW = two-tier positional percentile of effective projected points.
    #
    # Tier 1 (projected to play, effective_proj > 0):
    #   Rank among all projected players at same position → [0.50, 1.00].
    #   Uses Projected_Points from reference pool (Rotowire projections merged onto
    #   fpl_stats via enrich_reference_with_projections).  This avoids the scale
    #   mismatch of comparing single-GW projections against season-average ppg.
    #   Percentile: n_at_or_below / total → guarantees top player = 1.00.
    #
    # Tier 2 (not projected, effective_proj ≤ 0):
    #   Season points percentile vs full pool → [0.00, 0.50].
    #   Non-starters deserve lower 1GW scores, differentiated by underlying quality.
    #
    # Blank GW (effective_proj = NaN): neutral 0.50 (no information).
    result["1GW"] = 0.50  # default

    # Detect if reference pool has projection values for 1GW comparison
    _ref_proj_col = None
    if all_players_df is not None:
        for _cand in ("Projected_Points", "Points"):
            if _cand in all_players_df.columns and all_players_df[_cand].notna().any():
                _ref_proj_col = _cand
                break

    for pos in result["Position"].dropna().unique():
        pos_mask = result["Position"] == pos
        projected = pos_mask & result["_effective_proj"].gt(0)
        not_projected = pos_mask & result["_effective_proj"].eq(0)

        # Tier 1: projected players → [0.50, 1.00]
        if projected.sum() > 0:
            if _ref_proj_col is not None:
                # Compare effective_proj against reference projections (apples-to-apples)
                ref_pos = all_players_df[all_players_df["Position"] == pos]
                ref_vals = pd.to_numeric(ref_pos[_ref_proj_col], errors="coerce").dropna()
                ref_vals = ref_vals[ref_vals.gt(0)].values
            else:
                ref_vals = np.array([])

            if len(ref_vals) > 0:
                for idx in result.loc[projected].index:
                    val = result.at[idx, "_effective_proj"]
                    n_at_or_below = int(np.sum(ref_vals <= val))
                    pctile = n_at_or_below / len(ref_vals)
                    result.at[idx, "1GW"] = 0.50 + 0.50 * min(pctile, 1.0)
            else:
                # Fallback: self-rank among projected players in scoring set
                n_proj = projected.sum()
                if n_proj == 1:
                    result.loc[projected, "1GW"] = 0.75
                else:
                    rank_pctile = result.loc[projected, "_effective_proj"].rank(pct=True)
                    result.loc[projected, "1GW"] = 0.50 + 0.50 * rank_pctile

        # Tier 2: non-projected → season points percentile vs full pool → [0.00, 0.50]
        if not_projected.sum() > 0 and all_players_df is not None:
            ppg_pctile = positional_percentile(
                result.loc[not_projected], all_players_df, season_col,
                ref_value_col=ref_season_col, min_minutes=90,
            ).fillna(0.25)
            result.loc[not_projected, "1GW"] = ppg_pctile * 0.50

    # --- ROS Score (multi-GW dominant, dynamic weights) ---
    p = season_progress_weight(current_gw)

    # Season quality: blend actual season points with Rotowire season projection
    season_pts_pctile = positional_percentile(
        result, all_players_df, season_col, ref_value_col=ref_season_col, min_minutes=90
    ).fillna(0.5)

    if ("SeasonProjection" in result.columns and result["SeasonProjection"].notna().any()
            and all_players_df is not None and "SeasonProjection" in all_players_df.columns):
        season_proj_pctile = positional_percentile(
            result, all_players_df, "SeasonProjection",
            ref_value_col="SeasonProjection", min_minutes=90
        ).fillna(0.5)
    else:
        # SeasonProjection not in reference pool → use actual season points as proxy
        season_proj_pctile = season_pts_pctile

    season_quality = p * season_pts_pctile + (1 - p) * season_proj_pctile

    # Multi-GW projection percentile
    if "MultiGW_Proj" not in result.columns:
        result["MultiGW_Proj"] = 0
    result["MultiGW_Proj"] = pd.to_numeric(result["MultiGW_Proj"], errors="coerce").fillna(0)
    if all_players_df is not None and "MultiGW_Proj" not in all_players_df.columns and "points_per_game" in all_players_df.columns:
        ref_mgw = all_players_df.copy()
        ref_mgw["_mgw_proxy"] = pd.to_numeric(ref_mgw["points_per_game"], errors="coerce").fillna(0) * 3
        multigw_pctile = positional_percentile(
            result, ref_mgw, "MultiGW_Proj", ref_value_col="_mgw_proxy", min_minutes=90
        ).fillna(0.5)
    else:
        multigw_pctile = positional_percentile(
            result, all_players_df, "MultiGW_Proj", ref_value_col="MultiGW_Proj", min_minutes=90
        ).fillna(0.5)

    # Form dampened by starts
    form_pctile = positional_percentile(
        result, all_players_df, form_col, ref_value_col=ref_form_col, min_minutes=90
    ).fillna(0.5)
    starts_col = result["starts"] if "starts" in result.columns else pd.Series(0, index=result.index)
    starts = pd.to_numeric(starts_col, errors="coerce").fillna(0)
    form_dampened_pctile = dampen_form_by_starts(form_pctile, starts)

    # Start consistency percentile (FFP LongStart — rewards nailed-on starters)
    # Prefer FFP_LongStart in reference (same scale as candidate). Fall back to "starts".
    if (all_players_df is not None
            and "FFP_LongStart" in all_players_df.columns
            and all_players_df["FFP_LongStart"].notna().any()):
        ref_starts_col = "FFP_LongStart"
    elif all_players_df is not None and "starts" in all_players_df.columns:
        ref_starts_col = "starts"
    else:
        ref_starts_col = None

    if "FFP_LongStart" in result.columns and result["FFP_LongStart"].notna().any():
        result["_start_consistency"] = pd.to_numeric(result["FFP_LongStart"], errors="coerce").fillna(50)
    else:
        result["_start_consistency"] = starts

    if ref_starts_col:
        start_consistency_pctile = positional_percentile(
            result, all_players_df, "_start_consistency", ref_value_col=ref_starts_col, min_minutes=90
        ).fillna(0.5)
    else:
        start_consistency_pctile = _min_max_norm_series(result["_start_consistency"]).fillna(0.5)

    # FDR ease percentile (6 - AvgFDR → higher = easier fixtures)
    result["_FDREase"] = 6 - result[fdr_col]
    if all_players_df is not None and fdr_col in all_players_df.columns:
        ref_fdr = all_players_df.copy()
        ref_fdr["_FDREase"] = 6 - pd.to_numeric(ref_fdr[fdr_col], errors="coerce").fillna(3)
        fdr_ease_pctile = positional_percentile(
            result, ref_fdr, "_FDREase", ref_value_col="_FDREase", min_minutes=90
        ).fillna(0.5)
    else:
        fdr_ease_pctile = _min_max_norm_series(result["_FDREase"]).fillna(0.5)

    # Dynamic ROS weights (sum = 1.0 at all gameweeks)
    # FDR fades with season progress: overlaps with multi-GW (which already prices in fixtures),
    # and at late season the FDR window covers phantom games that won't be played.
    # Its freed weight shifts to season quality, which becomes the most reliable late-season signal.
    w_mgw = 0.40 - 0.10 * p          # 40% → 30% (multi-GW projections dominant)
    w_sq = 0.30 + 0.20 * p           # 30% → 49% (season quality grows; absorbs fading FDR)
    w_form = 0.15 - 0.05 * p         # 15% → 10% (trajectory indicator)
    w_start = 0.10                   # 10% constant (start consistency)
    w_fdr = 0.05 * (1 - p)           #  5% → ~0% (fades as remaining fixtures shrink)

    result["ROS"] = (
        w_mgw * multigw_pctile +
        w_sq * season_quality +
        w_form * form_dampened_pctile +
        w_start * start_consistency_pctile +
        w_fdr * fdr_ease_pctile
    )

    # --- Transfer Score / Keep Score (dynamic alpha blend) ---
    # Compute per-player alpha based on position, ROS quality, format, and depth
    transfer_score = pd.Series(0.0, index=result.index)
    keep_score = pd.Series(0.0, index=result.index)

    for idx in result.index:
        pos = str(result.at[idx, "Position"]) if "Position" in result.columns else "M"
        ros_val = float(result.at[idx, "ROS"])
        alpha = compute_dynamic_alpha(pos, ros_val, format_context, depth_map, current_gw)
        blended = alpha * float(result.at[idx, "1GW"]) + (1 - alpha) * ros_val
        transfer_score.at[idx] = blended
        keep_score.at[idx] = blended

    result["Transfer Score"] = transfer_score
    result["Keep Score"] = keep_score

    # Cleanup temp columns
    result.drop(columns=["_FDREase", "_start_consistency"], inplace=True, errors="ignore")

    return result


def positional_rank(
    df: pd.DataFrame,
    reference_df: Optional[pd.DataFrame],
    value_col: str,
    position_col: str = "Position",
    ref_value_col: Optional[str] = None,
) -> pd.Series:
    """
    Return ordinal rank strings like "#2 GK", "#15 MID" based on *value_col*
    within each position group from *reference_df*.

    Rank 1 = highest value at that position.  Tied values share the same rank.

    Args:
        df:            Squad DataFrame.
        reference_df:  Full FPL player pool.  If None, returns "N/A".
        value_col:     Column in df to rank by.
        position_col:  Column with position codes.
        ref_value_col: Column in reference_df if different from value_col.

    Returns:
        pd.Series of rank strings aligned to df.index.
    """
    ref_col = ref_value_col or value_col

    if (
        reference_df is None
        or ref_col not in reference_df.columns
        or position_col not in reference_df.columns
        or position_col not in df.columns
    ):
        return pd.Series("N/A", index=df.index)

    ref = reference_df.copy()
    ref[ref_col] = pd.to_numeric(ref[ref_col], errors="coerce")

    # Build sorted descending arrays per position
    pos_sorted: Dict[str, np.ndarray] = {}
    for pos, grp in ref.groupby(position_col):
        vals = grp[ref_col].dropna().sort_values(ascending=False).values
        if len(vals) > 0:
            pos_sorted[str(pos)] = vals

    player_vals = pd.to_numeric(df[value_col], errors="coerce")
    result = pd.Series("N/A", index=df.index)

    for idx in df.index:
        pos = str(df.at[idx, position_col])
        val = player_vals.at[idx]
        if pd.isna(val) or pos not in pos_sorted:
            continue
        pool = pos_sorted[pos]
        # Rank = number of players with strictly higher value + 1
        rank = int(np.sum(pool > val)) + 1
        label = _POS_LABELS.get(pos, pos)
        result.at[idx] = f"#{rank} {label}"

    return result


# =============================================================================
# MULTI-GW TRANSFER PLANNER — Shared Functions
# =============================================================================

def compute_healthy_form(
    player_id: int,
    last_n: int = 5,
    element_history_fn: Optional[Callable] = None,
) -> Optional[float]:
    """
    Compute form based only on GWs where a player actually played (minutes > 0).

    Unlike raw FPL form which includes 0-point GWs from injury/bench,
    this looks back further to find `last_n` GWs with actual minutes.

    Args:
        player_id: FPL element ID.
        last_n: Number of played GWs to average.
        element_history_fn: Injectable function(player_id) -> DataFrame.
            If None, uses _fetch_element_history.

    Returns:
        Average points across the last `last_n` played GWs, or None.
    """
    fetcher = element_history_fn or _fetch_element_history
    try:
        hist = fetcher(int(player_id))
    except Exception:
        return None

    if hist is None or hist.empty:
        return None

    gw_col = "round" if "round" in hist.columns else "event"
    pts_col = "total_points" if "total_points" in hist.columns else "points"
    min_col = "minutes"

    if pts_col not in hist.columns or min_col not in hist.columns:
        return None

    try:
        hist = hist.copy()
        hist[min_col] = pd.to_numeric(hist[min_col], errors="coerce").fillna(0)
        hist[pts_col] = pd.to_numeric(hist[pts_col], errors="coerce")

        # Filter to GWs where player actually played
        played = hist[hist[min_col] > 0].copy()
        if played.empty:
            return None

        # Sort by GW descending and take the last_n played GWs
        played = played.sort_values(gw_col, ascending=False)
        recent = played.head(last_n)
        vals = recent[pts_col].dropna()
        return float(vals.mean()) if not vals.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_element_history(player_id: int) -> Optional[pd.DataFrame]:
    """Standalone cached element-summary fetcher for Classic mode."""
    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        js = requests.get(url, timeout=30).json()
        hist = pd.DataFrame(js.get("history", []))
        return hist if not hist.empty else None
    except Exception:
        return None


@dataclass
class PositionalDepth:
    """Depth info for a single position."""
    position: str
    total: int
    healthy: int
    doubtful: int
    injured: int
    depth_level: str  # "Critical", "Low", "Adequate"


# Position-aware depth thresholds: healthy >= green → Adequate, >= yellow → Low, else Critical
_DEPTH_THRESHOLDS = {
    "G": {"green": 2, "yellow": 1},   # 2 slots: 2=green, 1=yellow, 0=red
    "D": {"green": 4, "yellow": 3},   # 5 slots: >=4=green, 3=yellow, <=2=red
    "M": {"green": 4, "yellow": 3},   # 5 slots: >=4=green, 3=yellow, <=2=red
    "F": {"green": 2, "yellow": 1},   # 3 slots: >=2=green, 1=yellow, 0=red
}


def compute_positional_depth(roster_df: pd.DataFrame) -> Dict[str, PositionalDepth]:
    """
    Compute squad depth at each position based on injury/availability data.

    Args:
        roster_df: Squad DataFrame with Position, chance_of_playing_next_round, status columns.

    Returns:
        Dict mapping position code (G/D/M/F) to PositionalDepth.
    """
    result = {}
    for pos in ["G", "D", "M", "F"]:
        pos_players = roster_df[roster_df["Position"] == pos]
        total = len(pos_players)
        if total == 0:
            result[pos] = PositionalDepth(pos, 0, 0, 0, 0, "Critical")
            continue

        healthy = 0
        doubtful = 0
        injured = 0
        for _, row in pos_players.iterrows():
            status = row.get("status")
            chance = row.get("chance_of_playing_next_round")

            # Classify each player into one of three states
            if pd.isna(status) and pd.isna(chance):
                healthy += 1  # Both missing → assume healthy
            elif not pd.isna(status) and str(status).lower() == "a":
                healthy += 1
            elif not pd.isna(status) and str(status).lower() == "d":
                doubtful += 1
            elif not pd.isna(status) and str(status).lower() in ("i", "s", "u"):
                injured += 1
            elif not pd.isna(chance):
                try:
                    c = float(chance)
                    if c >= 75:
                        healthy += 1
                    elif c >= 25:
                        doubtful += 1
                    else:
                        injured += 1
                except (ValueError, TypeError):
                    healthy += 1  # Can't parse, assume healthy
            else:
                injured += 1  # Unknown status, not available

        # Position-aware depth levels (doubtful counts as half-available)
        effective = healthy + doubtful * 0.5
        thresholds = _DEPTH_THRESHOLDS.get(pos, {"green": 2, "yellow": 1})
        if effective >= thresholds["green"]:
            level = "Adequate"
        elif effective >= thresholds["yellow"]:
            level = "Low"
        else:
            level = "Critical"

        result[pos] = PositionalDepth(pos, total, healthy, doubtful, injured, level)

    return result


def compute_transfer_urgency(position: str, depth_map: Dict[str, PositionalDepth]) -> str:
    """
    Return urgency label for a position based on squad depth.

    Returns:
        "URGENT" if Critical, "LOW DEPTH" if Low, "" if Adequate.
    """
    depth = depth_map.get(position)
    if depth is None:
        return ""
    if depth.depth_level == "Critical":
        return "URGENT"
    if depth.depth_level == "Low":
        return "LOW DEPTH"
    return ""


def blend_multi_gw_projections(
    player_df: pd.DataFrame,
    ffp_df: Optional[pd.DataFrame],
    single_gw_col: str = "Points",
    output_col: str = "MultiGW_Proj",
    remaining_gws: int = 3,
) -> pd.DataFrame:
    """
    Blend FFP multi-GW (Next3GWs) projections into a player DataFrame.

    Matches FFP players by normalized name + team short code.
    Unmatched players or missing FFP data falls back to single_gw_col * min(3, remaining_gws).

    Args:
        player_df: DataFrame with player names and a single-GW projection column.
        ffp_df: FFP projections DataFrame with Name, Team, Next3GWs columns (or None).
        single_gw_col: Column name for single-GW projection.
        output_col: Column name for the blended multi-GW value.
        remaining_gws: GWs left in season (caps fallback multiplier to avoid inflated projections
            near season end).

    Returns:
        player_df with output_col added.
    """
    result = player_df.copy()

    # Determine the player name column
    name_col = "Player" if "Player" in result.columns else "Name" if "Name" in result.columns else None

    # Fallback: single_gw * min(3, remaining_gws).
    # Cap at remaining_gws so we don't project 3× a single-GW value when only 1 GW is left.
    fallback_mult = min(3, max(1, remaining_gws))
    single_vals = pd.to_numeric(result.get(single_gw_col, 0), errors="coerce").fillna(0)
    result[output_col] = single_vals * fallback_mult
    # Second fallback: players with 0 single-GW proj (e.g. name merge failed) but valid ppg
    if "points_per_game" in result.columns:
        ppg = pd.to_numeric(result["points_per_game"], errors="coerce").fillna(0)
        zero_mask = result[output_col].eq(0) & ppg.gt(0)
        result.loc[zero_mask, output_col] = ppg[zero_mask] * fallback_mult

    if ffp_df is None or ffp_df.empty or name_col is None:
        return result

    if "Next3GWs" not in ffp_df.columns or "Name" not in ffp_df.columns:
        return result

    # Check if FFP multi-GW predictions are available (all-zero = not published yet)
    ffp_next3 = pd.to_numeric(ffp_df["Next3GWs"], errors="coerce")
    if not ffp_next3.gt(0).any():
        _logger.info("FFP Next3GWs not ready (all zero) — using fallback projections")
        return result

    # Build lookup from FFP data: (normalized_name, team_short) -> Next3GWs
    ffp = ffp_df[["Name", "Team", "Next3GWs"]].dropna(subset=["Next3GWs"]).copy()
    ffp["__norm"] = ffp["Name"].apply(canonical_normalize)
    ffp["__team_short"] = ffp["Team"].replace(TEAM_FULL_TO_SHORT)
    ffp["Next3GWs"] = pd.to_numeric(ffp["Next3GWs"], errors="coerce")

    # Build lookup dicts (individual zeros still skipped as defense-in-depth)
    lookup = {}          # (norm_name, team_short) -> Next3GWs  [primary]
    lookup_short = {}    # (last_word_of_norm, team_short)       [secondary]
    lookup_name = {}     # norm_name only                        [tertiary — team-agnostic]
    lookup_lastword = {} # last_word only                        [quaternary — last resort]
    for _, row in ffp.iterrows():
        key = (row["__norm"], str(row["__team_short"]))
        val = row["Next3GWs"]
        if pd.notna(val) and val > 0:
            lookup[key] = val
            last_word = row["__norm"].split()[-1] if row["__norm"] else ""
            if last_word:
                short_key = (last_word, str(row["__team_short"]))
                if short_key not in lookup_short:
                    lookup_short[short_key] = val
            if row["__norm"] and row["__norm"] not in lookup_name:
                lookup_name[row["__norm"]] = val
            # Also store reversed token order to catch "family given" → "given family" variants
            # (e.g. FFP "Tanaka Ao" ↔ FPL "Ao Tanaka" for Japanese players).
            norm_tokens = row["__norm"].split()
            if len(norm_tokens) > 1:
                norm_reversed = " ".join(reversed(norm_tokens))
                if norm_reversed not in lookup_name:
                    lookup_name[norm_reversed] = val
            if last_word and last_word not in lookup_lastword:
                lookup_lastword[last_word] = val

    if not lookup:
        return result

    # Match players
    result["__norm"] = result[name_col].apply(canonical_normalize)
    team_col = "Team" if "Team" in result.columns else None

    for idx in result.index:
        norm_name = result.at[idx, "__norm"]
        team_short = str(result.at[idx, team_col]) if team_col else ""
        team_short = TEAM_FULL_TO_SHORT.get(team_short, team_short)
        last_word = norm_name.split()[-1] if norm_name else ""

        key = (norm_name, team_short)
        if key in lookup:
            result.at[idx, output_col] = lookup[key]
        elif last_word and (last_word, team_short) in lookup_short:
            result.at[idx, output_col] = lookup_short[(last_word, team_short)]
        elif norm_name and norm_name in lookup_name:
            result.at[idx, output_col] = lookup_name[norm_name]
        elif last_word and last_word in lookup_lastword:
            result.at[idx, output_col] = lookup_lastword[last_word]

    result.drop(columns=["__norm"], inplace=True, errors="ignore")
    return result


def merge_season_projections(
    player_df: pd.DataFrame,
    season_rankings_df: Optional[pd.DataFrame],
    output_col: str = "SeasonProjection",
) -> pd.DataFrame:
    """Merge Rotowire season-long projected points onto a player DataFrame.

    Matches by normalized name + team short code.
    Unmatched players get NaN (callers should handle fallback).

    Args:
        player_df: DataFrame with player names and Team column.
        season_rankings_df: Rotowire season rankings with Player, Team, Points.
        output_col: Column name for the merged season projection.

    Returns:
        player_df with output_col added.
    """
    result = player_df.copy()
    result[output_col] = np.nan

    if season_rankings_df is None or season_rankings_df.empty:
        return result

    if "Points" not in season_rankings_df.columns or "Player" not in season_rankings_df.columns:
        return result

    name_col = "Player" if "Player" in result.columns else "Name" if "Name" in result.columns else None
    if name_col is None:
        return result

    # Build lookup: (normalized_name, team_short) -> Points
    sr = season_rankings_df[["Player", "Team", "Points"]].dropna(subset=["Points"]).copy()
    sr["__norm"] = sr["Player"].apply(canonical_normalize)
    sr["__team_short"] = sr["Team"].replace(TEAM_FULL_TO_SHORT)
    sr["Points"] = pd.to_numeric(sr["Points"], errors="coerce")

    lookup = {}
    for _, row in sr.iterrows():
        key = (row["__norm"], str(row["__team_short"]))
        val = row["Points"]
        if pd.notna(val):
            lookup[key] = val

    if not lookup:
        return result

    result["__norm"] = result[name_col].apply(canonical_normalize)
    team_col = "Team" if "Team" in result.columns else None

    for idx in result.index:
        norm_name = result.at[idx, "__norm"]
        team_short = str(result.at[idx, team_col]) if team_col else ""
        team_short = TEAM_FULL_TO_SHORT.get(team_short, team_short)
        key = (norm_name, team_short)
        if key in lookup:
            result.at[idx, output_col] = lookup[key]

    result.drop(columns=["__norm"], inplace=True, errors="ignore")
    return result


def merge_ffp_single_gw_data(
    player_df: pd.DataFrame,
    ffp_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge FFP single-GW data (Predicted, Start, LongStart) onto a player DataFrame.

    Matches by normalized name + team short code, following the same pattern as
    blend_multi_gw_projections().

    Args:
        player_df: DataFrame with player names and Team column.
        ffp_df: FFP projections DataFrame with Name, Team, Predicted, Start, LongStart.

    Returns:
        player_df with FFP_Predicted, FFP_Start, FFP_LongStart columns added (NaN if unmatched).
    """
    result = player_df.copy()
    result["FFP_Predicted"] = np.nan
    result["FFP_Start"] = np.nan
    result["FFP_LongStart"] = np.nan

    name_col = "Player" if "Player" in result.columns else "Name" if "Name" in result.columns else None
    if ffp_df is None or ffp_df.empty or name_col is None:
        return result

    # Check that at least one useful column exists
    ffp_cols = [c for c in ("Predicted", "Start", "LongStart") if c in ffp_df.columns]
    if not ffp_cols or "Name" not in ffp_df.columns:
        return result

    # Skip prediction columns that aren't ready (all-zero = FFP hasn't published yet).
    # Start/LongStart can legitimately be 0 for individual players, so only check Predicted.
    if "Predicted" in ffp_cols:
        pred_vals = pd.to_numeric(ffp_df["Predicted"], errors="coerce")
        if not pred_vals.gt(0).any():
            _logger.info("FFP Predicted not ready (all zero) — skipping single-GW predictions")
            ffp_cols = [c for c in ffp_cols if c != "Predicted"]
            if not ffp_cols:
                return result

    # Build lookup: (normalized_name, team_short) -> {Predicted, Start, LongStart}
    ffp = ffp_df[["Name", "Team"] + ffp_cols].copy()
    ffp["__norm"] = ffp["Name"].apply(canonical_normalize)
    ffp["__team_short"] = ffp["Team"].replace(TEAM_FULL_TO_SHORT)
    for col in ffp_cols:
        ffp[col] = pd.to_numeric(ffp[col], errors="coerce")

    lookup = {}          # (norm_name, team_short) -> data dict  [primary]
    lookup_short = {}    # (last_word_of_norm, team_short)        [secondary]
    lookup_name = {}     # norm_name only                         [tertiary — team-agnostic]
    lookup_lastword = {} # last_word only                         [quaternary — last resort]
    for _, row in ffp.iterrows():
        key = (row["__norm"], str(row["__team_short"]))
        data = {col: row[col] for col in ffp_cols if pd.notna(row[col])}
        lookup[key] = data
        last_word = row["__norm"].split()[-1] if row["__norm"] else ""
        if last_word:
            short_key = (last_word, str(row["__team_short"]))
            if short_key not in lookup_short:
                lookup_short[short_key] = data
        if row["__norm"] and row["__norm"] not in lookup_name:
            lookup_name[row["__norm"]] = data
        if last_word and last_word not in lookup_lastword:
            lookup_lastword[last_word] = data

    if not lookup:
        return result

    # Match players
    result["__norm"] = result[name_col].apply(canonical_normalize)
    team_col = "Team" if "Team" in result.columns else None

    col_map = {"Predicted": "FFP_Predicted", "Start": "FFP_Start", "LongStart": "FFP_LongStart"}

    for idx in result.index:
        norm_name = result.at[idx, "__norm"]
        team_short = str(result.at[idx, team_col]) if team_col else ""
        team_short = TEAM_FULL_TO_SHORT.get(team_short, team_short)
        last_word = norm_name.split()[-1] if norm_name else ""

        key = (norm_name, team_short)
        data = lookup.get(key)
        if data is None and last_word:
            data = lookup_short.get((last_word, team_short))
        if data is None and norm_name:
            data = lookup_name.get(norm_name)
        if data is None and last_word:
            data = lookup_lastword.get(last_word)
        if data:
            for src_col, dst_col in col_map.items():
                if src_col in data:
                    result.at[idx, dst_col] = data[src_col]

    result.drop(columns=["__norm"], inplace=True, errors="ignore")
    return result


def _add_fdr_and_form(
    df: pd.DataFrame,
    fpl_player_statistics_df: pd.DataFrame,
    current_gw: int,
    weeks: int
) -> pd.DataFrame:
    """
    Join AvgFDRNextN and Form onto df.

    - Requires df to have Player/Team/Position (will create if missing).
    - Merges in Player_ID, then computes Form via element-summary.
    - Fallback chain for Form: element-summary -> FPL 'form' -> 'points_per_game' -> 0.

    NOTE: This function references _avg_fdr_for_team and _avg_form_last_n which
    are not yet implemented. It will fail at runtime if called until those are added.
    """
    base = df.copy()
    # Ensure join keys exist
    for col in ("Player", "Team", "Position"):
        if col not in base.columns:
            base[col] = np.nan

    # Safely select merge cols from stats
    stats = fpl_player_statistics_df.copy()
    for col in ("Player", "Team", "Position", "Player_ID", "form", "points_per_game"):
        if col not in stats.columns:
            stats[col] = np.nan

    # Merge in Player_ID + FPL fallback stats
    base = base.merge(
        stats[["Player", "Team", "Position", "Player_ID", "form", "points_per_game"]],
        on=["Player", "Team", "Position"],
        how="left"
    )

    # Ensure Player_ID exists (for downstream)
    if "Player_ID" not in base.columns:
        base["Player_ID"] = np.nan

    # Compute Avg FDR next N GWs
    # NOTE: _avg_fdr_for_team is not implemented - this will fail at runtime
    base["AvgFDRNextN"] = base["Team"].apply(lambda t: _avg_fdr_for_team(str(t), current_gw, weeks))
    base["AvgFDRNextN"] = pd.to_numeric(base["AvgFDRNextN"], errors="coerce")

    # Robust Form calculation with fallbacks
    def _safe_form(pid, fallback_form, fallback_ppg):
        # element-summary average of last N
        val = None
        if pd.notna(pid):
            try:
                # NOTE: _avg_form_last_n is not implemented - this will fail at runtime
                val = _avg_form_last_n(int(pid), config.FORM_LOOKBACK_WEEKS)
            except Exception:
                val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_form, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_ppg, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        return float(val)

    base["Form"] = base.apply(
        lambda r: _safe_form(r.get("Player_ID"), r.get("form"), r.get("points_per_game")),
        axis=1
    )
    base["Form"] = pd.to_numeric(base["Form"], errors="coerce").fillna(0.0)

    return base


def apply_availability_penalty(df: pd.DataFrame, score_col: str, out_col: str) -> pd.DataFrame:
    """
    Multiply a score column by (PlayPct/100) so low availability downweights adds/drops.
    """
    out = df.copy()
    out[out_col] = pd.to_numeric(out[score_col], errors="coerce") * (pd.to_numeric(out["PlayPct"], errors="coerce")/100.0)
    return out


def attach_availability(df: pd.DataFrame, avail_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-merge PlayPct/StatusBucket/News onto df using Player+Team first,
    then try Player_ID if present.
    """
    base = df.copy()
    cols_keep = ["PlayPct","StatusBucket","News"]
    # Try exact Player+Team first (your normalizers map Team to short codes)
    base = base.merge(
        avail_df[["Player","Team","PlayPct","StatusBucket","News"]],
        on=["Player","Team"], how="left", suffixes=("","")
    )
    # If still missing and we have IDs, try by Player_ID
    if "Player_ID" in base.columns and "Player_ID" in avail_df.columns:
        mask = base["PlayPct"].isna()
        if mask.any():
            left = base.loc[mask, ["Player_ID"]].copy()
            right = avail_df[["Player_ID"] + cols_keep].copy()
            joined = left.merge(right, on="Player_ID", how="left")
            base.loc[mask, cols_keep] = joined[cols_keep].values

    # Fill PlayPct neutral defaults
    base["PlayPct"] = pd.to_numeric(base["PlayPct"], errors="coerce").fillna(50.0)
    base["StatusBucket"] = base["StatusBucket"].fillna("Questionable")
    base["News"] = base["News"].fillna("")
    return base


def simulate_auto_subs(squad_df, live_stats, element_to_team, finished_team_ids):
    """
    Simulate FPL auto-substitutions for a 15-player squad.

    A starter is subbed out if:
    - They have 0 minutes in live_stats
    - Their team's match is finished

    Bench players are subbed in following FPL rules:
    - GK bench slot (squad_position 12) only subs for GK starter
    - Outfield bench (squad_position 13→14→15) in order for outfield starters
    - Must maintain at least 3 DEF after sub

    Parameters:
    - squad_df: Full 15-player DataFrame with squad_position, element_id, Position columns.
    - live_stats: {element_id: {has_played, minutes, points, ...}} from get_live_gameweek_stats().
    - element_to_team: {element_id: team_id} mapping.
    - finished_team_ids: set of team_ids whose match is finished.

    Returns: (updated_squad_df, sub_list)
    - updated_squad_df: DataFrame with squad_positions swapped for auto-subs.
    - sub_list: list of (out_name, in_name) tuples for display.
    """
    df = squad_df.copy()
    sub_list = []

    if df.empty or not finished_team_ids:
        return df, sub_list

    # Identify starters needing a sub: 0 minutes AND their team's match is finished
    starters_out = []
    for idx, row in df[df["squad_position"].between(1, 11)].iterrows():
        eid = row.get("element_id")
        if eid is None:
            continue
        team_id = element_to_team.get(eid)
        if team_id is None or team_id not in finished_team_ids:
            continue
        stats = live_stats.get(eid, {})
        # Player must have 0 minutes (not played at all) to be auto-subbed
        if stats.get("minutes", 0) == 0 and not stats.get("has_played", False):
            starters_out.append(idx)

    if not starters_out:
        return df, sub_list

    # Get bench players ordered by squad_position (12, 13, 14, 15)
    bench = df[df["squad_position"].between(12, 15)].sort_values("squad_position")
    used_bench = set()

    for starter_idx in starters_out:
        starter_row = df.loc[starter_idx]
        starter_pos = starter_row["Position"]
        starter_squad_pos = starter_row["squad_position"]

        # Count current DEF starters (excluding this player being subbed out)
        current_def_count = len(
            df[(df["squad_position"].between(1, 11)) & (df["Position"] == "D") & (df.index != starter_idx)]
        )
        # Also subtract any DEF starters already subbed out in this loop
        for prev_idx in starters_out:
            if prev_idx != starter_idx and prev_idx in [s for s, _ in sub_list if isinstance(s, int)]:
                if df.loc[prev_idx, "Position"] == "D":
                    current_def_count -= 1

        if starter_pos == "G":
            # GK can only be replaced by bench GK (squad_position 12)
            for bench_idx, bench_row in bench.iterrows():
                if bench_idx in used_bench:
                    continue
                if bench_row["Position"] == "G":
                    # Swap squad positions
                    df.at[bench_idx, "squad_position"] = starter_squad_pos
                    df.at[starter_idx, "squad_position"] = bench_row["squad_position"]
                    used_bench.add(bench_idx)
                    starter_name = starter_row.get("Player", "Unknown")
                    bench_name = bench_row.get("Player", "Unknown")
                    sub_list.append((starter_name, bench_name))
                    break
        else:
            # Outfield: try bench positions 13, 14, 15 in order
            for bench_idx, bench_row in bench.iterrows():
                if bench_idx in used_bench:
                    continue
                if bench_row["Position"] == "G":
                    continue  # Skip GK bench slot for outfield subs

                # Check DEF minimum constraint: if removing this starter drops DEF below 3,
                # only a DEF can replace them
                if starter_pos == "D" and current_def_count < 3:
                    if bench_row["Position"] != "D":
                        continue

                # Valid sub found
                df.at[bench_idx, "squad_position"] = starter_squad_pos
                df.at[starter_idx, "squad_position"] = bench_row["squad_position"]
                used_bench.add(bench_idx)
                starter_name = starter_row.get("Player", "Unknown")
                bench_name = bench_row.get("Player", "Unknown")
                sub_list.append((starter_name, bench_name))
                break

    return df, sub_list


def team_optimizer(player_rankings):
    """
    Optimize team lineup based on player rankings.

    NOTE: This function references config.TEAM_LIST which may not exist.
    """
    # Loop over each team in the team_list
    for team_name, team_data in config.TEAM_LIST.items():
        # Convert the JSON to a DataFrame
        team_player_list = []
        for position, players in team_data.items():
            for player in players:
                team_player_list.append({'player_name': player, 'position': position})

        team_df = pd.DataFrame(team_player_list)

        # Perform a left join, filling missing Pts with 0
        merged_df = team_df.merge(player_rankings, left_on='player_name', right_on='Player', how='left')

        # Fill NaN values in the Pts column with 0
        merged_df['Pts'] = merged_df['Pts'].fillna(0)

        # Select relevant columns for the final output and rename them
        final_df = merged_df[['player_name', 'position', 'Pts']]
        final_df.columns = ['Player', 'Position', 'Projected_Points']

        # Ensure the 'Projected_Points' column is numeric
        final_df.loc[:, 'Projected_Points'] = pd.to_numeric(final_df['Projected_Points'], errors='coerce')

        # Return the DataFrame
        return(final_df)
