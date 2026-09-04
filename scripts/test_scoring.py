"""
Standalone scoring test script for fast iteration on analytics.py weights.

Usage:
    python -m scripts.test_scoring            # First run: fetch from APIs, cache, score
    python -m scripts.test_scoring            # Subsequent: load from cache (instant), score
    python -m scripts.test_scoring --refresh  # Re-fetch from APIs and update cache
"""

# ── Streamlit mock (must come before any project imports) ───────────────
import types
import sys

_st = types.ModuleType("streamlit")
_st.cache_data = lambda **kw: (lambda f: f)
_st.cache_resource = lambda **kw: (lambda f: f)
_st.error = lambda *a, **k: None
_st.warning = lambda *a, **k: None
_st.info = lambda *a, **k: None
_st.caption = lambda *a, **k: None
_st.stop = lambda *a, **k: None
sys.modules["streamlit"] = _st

# ── Standard imports ────────────────────────────────────────────────────
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ── Project imports ─────────────────────────────────────────────────────
import config
from scripts.common.player_matching import canonical_normalize
from scripts.common.text_helpers import TEAM_FULL_TO_SHORT
from scripts.common.utils import (
    get_rotowire_player_projections,
    pull_fpl_player_stats,
    normalize_fpl_players_to_rotowire_schema,
    normalize_rotowire_players,
    _bootstrap_teams_df,
    _backfill_player_ids,
    get_league_player_ownership,
    merge_fpl_players_and_projections,
    get_current_gameweek,
)
from scripts.common.scraping import get_ffp_feed, get_rotowire_season_rankings
from scripts.common.analytics import (
    compute_player_scores,
    compute_positional_depth,
    blend_multi_gw_projections,
    merge_season_projections,
    merge_ffp_single_gw_data,
    enrich_reference_with_projections,
    positional_percentile,
    positional_rank,
    season_progress_weight,
    compute_dynamic_alpha,
)
from scripts.draft.waiver_wire import (
    _enforce_rw_schema_fpl,
    _enforce_rw_schema_proj,
    _available_from_projections,
    _add_fdr_and_form,
    _add_injury_data,
    _ensure_team_col,
    _align_roster_player_names_to_projections,
    _flatten_owned_names,
    _prepare_proj_for_merge,
)

# ── Cache config ────────────────────────────────────────────────────────
CACHE_DIR = Path(".scoring_test_cache")
CACHE_FILE = CACHE_DIR / "scoring_data.pkl"


def _fetch_and_prepare_data() -> dict:
    """Fetch all data from APIs and prepare the scoring inputs.

    Returns a dict with keys:
        my_roster, avail_all, fpl_stats, current_gw, depth_map, my_team_name
    """
    print("Fetching data from APIs...")

    # 1. Raw data
    print("  [1/7] Rotowire projections...")
    projections_raw = get_rotowire_player_projections(config.ROTOWIRE_URL)

    print("  [2/7] FPL player stats...")
    fpl_stats_raw = pull_fpl_player_stats()

    print("  [3/7] FFP projections...")
    try:
        ffp_df = get_ffp_feed().df
    except Exception:
        ffp_df = None

    print("  [4/7] Season rankings...")
    try:
        season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
    except Exception:
        season_rankings_df = None

    print("  [5/7] League ownership...")
    ownership = get_league_player_ownership(config.FPL_DRAFT_LEAGUE_ID)

    print("  [6/7] Current gameweek...")
    try:
        current_gw = int(get_current_gameweek() or 1)
    except Exception:
        current_gw = 1

    # 2. Normalize
    print("  [7/7] Normalizing schemas...")
    teams_df = _bootstrap_teams_df()
    try:
        fpl_stats_norm = normalize_fpl_players_to_rotowire_schema(fpl_stats_raw, teams_df=teams_df)
    except Exception:
        fpl_stats_norm = fpl_stats_raw
    try:
        projections_norm = normalize_rotowire_players(projections_raw)
    except Exception:
        projections_norm = projections_raw

    fpl_stats = _enforce_rw_schema_fpl(fpl_stats_norm, teams_df)
    proj = _enforce_rw_schema_proj(projections_norm, teams_df)

    if "Season_Points" not in fpl_stats.columns:
        fpl_stats["Season_Points"] = 0

    # 3. Available players
    print("  Building available players pool...")
    avail_all = _available_from_projections(proj, fpl_stats, ownership)

    lookahead = config.UPCOMING_WEEKS_DEFAULT
    form_weeks = config.FORM_LOOKBACK_WEEKS

    avail_all = _add_fdr_and_form(avail_all, fpl_stats, current_gw, lookahead, form_weeks=form_weeks)
    avail_all = _add_injury_data(avail_all, fpl_stats)

    # Add Season_Points and starts
    if "Player_ID" in avail_all.columns and "Season_Points" not in avail_all.columns:
        sp_cols = ["Player_ID", "Season_Points"]
        if "starts" in fpl_stats.columns:
            sp_cols.append("starts")
        sp_data = fpl_stats[sp_cols].drop_duplicates(subset=["Player_ID"])
        avail_all = avail_all.merge(sp_data, on="Player_ID", how="left")
    elif "Player_ID" in avail_all.columns and "starts" not in avail_all.columns and "starts" in fpl_stats.columns:
        starts_data = fpl_stats[["Player_ID", "starts"]].drop_duplicates(subset=["Player_ID"])
        avail_all = avail_all.merge(starts_data, on="Player_ID", how="left")
    avail_all["Season_Points"] = pd.to_numeric(avail_all.get("Season_Points"), errors="coerce").fillna(0)
    avail_all["starts"] = pd.to_numeric(avail_all.get("starts", 0), errors="coerce").fillna(0)

    # 4. My roster
    print("  Building roster...")
    my_team_id = int(config.FPL_DRAFT_TEAM_ID)
    my_team = ownership.get(my_team_id, {})
    my_team_name = my_team.get("team_name", "(unknown)")

    my_players = []
    for pos, names in my_team.get("players", {}).items():
        for nm in names:
            my_players.append({"Player": nm, "Position": pos})
    my_roster = pd.DataFrame(my_players)

    if not my_roster.empty:
        my_roster["__norm_name"] = my_roster["Player"].apply(canonical_normalize)
        roster_merge_cols = ["Player", "Team", "Position", "Player_ID", "Season_Points"]
        if "starts" in fpl_stats.columns:
            roster_merge_cols.append("starts")
        fpl_stats_for_roster = fpl_stats[roster_merge_cols].copy()
        fpl_stats_for_roster["__norm_name"] = fpl_stats_for_roster["Player"].apply(canonical_normalize)

        merge_right_cols = ["__norm_name", "Team", "Position", "Player_ID", "Season_Points"]
        if "starts" in fpl_stats_for_roster.columns:
            merge_right_cols.append("starts")
        my_roster = my_roster.merge(
            fpl_stats_for_roster[merge_right_cols],
            on=["__norm_name", "Position"], how="left"
        )
        my_roster.drop(columns=["__norm_name"], inplace=True, errors="ignore")

        my_roster = _ensure_team_col(my_roster, fpl_stats)
        my_roster = _align_roster_player_names_to_projections(my_roster, proj)
        my_roster["Team"] = my_roster["Team"].replace(TEAM_FULL_TO_SHORT)

        # Add projected points
        proj_points_df = proj[["Player", "Team", "Position", "Points"]].rename(
            columns={"Points": "Projected_Points"}
        ).copy()
        proj_points_df["__norm_name"] = proj_points_df["Player"].apply(canonical_normalize)
        my_roster["__norm_name"] = my_roster["Player"].apply(canonical_normalize)
        my_roster = my_roster.merge(
            proj_points_df[["__norm_name", "Team", "Position", "Projected_Points"]],
            on=["__norm_name", "Team", "Position"], how="left"
        )
        my_roster.drop(columns=["__norm_name"], inplace=True, errors="ignore")

        my_roster = _backfill_player_ids(my_roster, fpl_stats)
        my_roster = _add_fdr_and_form(my_roster, fpl_stats, current_gw, lookahead)
        my_roster = _add_injury_data(my_roster, fpl_stats)

    # 5. Multi-GW, Season, FFP enrichment
    print("  Enriching with multi-GW / season / FFP data...")
    if not avail_all.empty:
        avail_all = blend_multi_gw_projections(avail_all, ffp_df, single_gw_col="Points")
    if not my_roster.empty:
        pts_col = "Projected_Points" if "Projected_Points" in my_roster.columns else "Points"
        my_roster = blend_multi_gw_projections(my_roster, ffp_df, single_gw_col=pts_col)

    if not avail_all.empty:
        avail_all = merge_season_projections(avail_all, season_rankings_df)
    if not my_roster.empty:
        my_roster = merge_season_projections(my_roster, season_rankings_df)

    if not avail_all.empty:
        avail_all = merge_ffp_single_gw_data(avail_all, ffp_df)
    if not my_roster.empty:
        my_roster = merge_ffp_single_gw_data(my_roster, ffp_df)

    # 6. Enrich reference pool
    fpl_stats = blend_multi_gw_projections(fpl_stats, ffp_df, single_gw_col="points_per_game")
    fpl_stats = merge_season_projections(fpl_stats, season_rankings_df)
    fpl_stats = merge_ffp_single_gw_data(fpl_stats, ffp_df)
    fpl_stats = enrich_reference_with_projections(fpl_stats, proj)

    # 7. Depth map
    depth_map = {}
    if not my_roster.empty:
        depth_map = compute_positional_depth(my_roster)

    return {
        "my_roster": my_roster,
        "avail_all": avail_all,
        "fpl_stats": fpl_stats,
        "current_gw": current_gw,
        "depth_map": depth_map,
        "my_team_name": my_team_name,
    }


def _load_cached() -> dict:
    """Load cached data from pickle."""
    with open(CACHE_FILE, "rb") as f:
        return pickle.load(f)


def _save_cache(data: dict):
    """Save data to pickle cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(data, f)
    print(f"  Cache saved to {CACHE_FILE}")


def _print_ros_weights(current_gw: int):
    """Print the ROS weight breakdown for the current gameweek."""
    p = season_progress_weight(current_gw)
    w_mgw = 0.40 - 0.10 * p
    w_sq = 0.30 + 0.15 * p
    w_form = 0.15 - 0.05 * p
    w_start = 0.10
    w_fdr = 0.05

    print(f"\nGW {current_gw}: p={p:.3f} | "
          f"w_mgw={w_mgw:.3f} w_sq={w_sq:.3f} w_form={w_form:.3f} "
          f"w_start={w_start:.2f} w_fdr={w_fdr:.2f}")


def _print_table(df: pd.DataFrame, title: str, score_label: str, top_n: int = None):
    """Print a formatted scoring table.

    Args:
        df: Scored DataFrame with 1GW, ROS, Transfer Score / Keep Score columns.
        title: Section title to print.
        score_label: Short display label ("Keep" or "Xfer").
        top_n: Limit to top N rows (None = all).
    """
    if df.empty:
        print(f"\n{title}\n  (empty)")
        return

    # Map short label to actual DataFrame column name
    score_col_map = {"Keep": "Keep Score", "Xfer": "Transfer Score"}
    actual_col = score_col_map.get(score_label, score_label)

    show = df.copy()
    if top_n:
        show = show.head(top_n)

    # Determine projection column
    proj_col = "Projected_Points" if "Projected_Points" in show.columns else "Points"
    proj_vals = pd.to_numeric(show.get(proj_col, 0), errors="coerce").fillna(0)

    # Build display DataFrame
    display = pd.DataFrame()
    display["Player"] = show["Player"].values
    display["Pos"] = show["Position"].values
    display["Team"] = show["Team"].values
    display["Proj"] = proj_vals.values

    # 1GW and effective proj
    display["1GW"] = pd.to_numeric(show.get("1GW", 0), errors="coerce").fillna(0).values

    # ROS
    display["ROS"] = pd.to_numeric(show.get("ROS", 0), errors="coerce").fillna(0).values

    # Alpha and final score
    alphas = []
    for _, row in show.iterrows():
        pos = str(row.get("Position", "M"))
        ros = float(row.get("ROS", 0) or 0)
        a = compute_dynamic_alpha(pos, ros, format_context="draft")
        alphas.append(a)
    display["Alpha"] = alphas

    if actual_col in show.columns:
        display[score_label] = pd.to_numeric(show[actual_col], errors="coerce").fillna(0).values
    else:
        display[score_label] = 0.0

    # Format
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    # Header
    header = (f"{'Player':<24s} {'Pos':<4s} {'Team':<5s} {'Proj':>5s} "
              f"{'1GW':>5s} {'ROS':>5s} {'Alpha':>6s} {score_label:>6s}")
    print(header)
    print("-" * len(header))

    for _, row in display.iterrows():
        line = (f"{str(row['Player'])[:23]:<24s} {str(row['Pos']):<4s} {str(row['Team']):<5s} "
                f"{float(row['Proj']):5.1f} "
                f"{float(row['1GW']):5.2f} {float(row['ROS']):5.2f} "
                f"{float(row['Alpha']):6.2f} {float(row[score_label]):6.2f}")
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Scoring model test script")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch from APIs")
    args = parser.parse_args()

    # Load or fetch data
    if CACHE_FILE.exists() and not args.refresh:
        print(f"Loading cached data from {CACHE_FILE}...")
        data = _load_cached()
        print("  Loaded.")
    else:
        data = _fetch_and_prepare_data()
        _save_cache(data)

    my_roster = data["my_roster"]
    avail_all = data["avail_all"]
    fpl_stats = data["fpl_stats"]
    current_gw = data["current_gw"]
    depth_map = data["depth_map"]
    my_team_name = data["my_team_name"]

    # ── Always run scoring fresh (picks up analytics.py weight changes) ──
    print(f"\nScoring with GW={current_gw}, format=draft...")

    if not my_roster.empty:
        my_roster = compute_player_scores(
            my_roster, fpl_stats, current_gw,
            format_context="draft", depth_map=depth_map
        )
        my_roster = my_roster.sort_values("Keep Score", ascending=False).reset_index(drop=True)

    if not avail_all.empty:
        avail_all = compute_player_scores(
            avail_all, fpl_stats, current_gw,
            format_context="draft"
        )
        avail_all = avail_all.sort_values("Transfer Score", ascending=False).reset_index(drop=True)

    # ── Print results ───────────────────────────────────────────────────
    _print_ros_weights(current_gw)

    # Reference pool stats
    ref_pos_counts = fpl_stats.groupby("Position").size()
    print(f"\nReference pool: {len(fpl_stats)} players")
    for pos in ["G", "D", "M", "F"]:
        count = ref_pos_counts.get(pos, 0)
        ffp_count = fpl_stats[fpl_stats["Position"] == pos]["FFP_Predicted"].notna().sum() if "FFP_Predicted" in fpl_stats.columns else 0
        print(f"  {pos}: {count} players ({ffp_count} with FFP_Predicted)")

    _print_table(my_roster, f"MY ROSTER — {my_team_name} (sorted by Keep Score)", "Keep")
    _print_table(avail_all, "AVAILABLE PLAYERS (top 30 by Transfer Score)", "Xfer", top_n=30)


if __name__ == "__main__":
    main()
