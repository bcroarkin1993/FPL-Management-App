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
    """The scoring inputs, from the one claim engine the app runs.

    Returns a dict with keys:
        my_roster, avail_all, fpl_stats, current_gw, depth_map, my_team_name

    **This used to hand-roll the Waiver Wire's pipeline**, and had drifted six
    ways from it -- which matters because this script exists to tune ROS
    weights, so every weight was being judged against inputs the app does not
    produce:

      - `element_states` was never passed to `_available_from_projections`,
        so every *locked* player scored as a claimable target;
      - `_attach_minutes_dependency` was absent, so no deputy's horizon was
        capped;
      - the reference pool had no `AvgFDRNextN`, so the fixture term fell back
        to a min-max of each frame against itself -- two populations compared
        as one, which is the exact defect CLAUDE.md records as repaired;
      - `remaining_gws` was omitted from all three multi-gameweek blends;
      - the roster's `Projected_Points` used only the primary name merge,
        without the two fallbacks the page carries;
      - `_attach_display_names` was absent.

    It now calls `build_waiver_board`, so there is one pipeline and tuning
    happens against what the app actually scores.
    """
    print("Fetching data from APIs...")
    from scripts.draft.waiver_core import build_waiver_board

    board = build_waiver_board(
        config.FPL_DRAFT_LEAGUE_ID, int(config.FPL_DRAFT_TEAM_ID),
        entry_id=getattr(config, "FPL_DRAFT_TEAM_ID", None))

    for severity, message in board.notes:
        print("  [%s] %s" % (severity, message))
    if not board.ok:
        raise SystemExit("Could not build the waiver board: %s" % board.status)

    print("  available=%d  roster=%d  reference=%d  gw=%d"
          % (len(board.available), len(board.roster), len(board.reference),
             board.current_gw))
    return {
        "my_roster": board.roster,
        "avail_all": board.available,
        "fpl_stats": board.reference,
        "current_gw": board.current_gw,
        "depth_map": board.depth_map,
        "my_team_name": board.team_name,
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
