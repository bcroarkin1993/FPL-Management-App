"""
Player Analytics Functions.

FDR/form enrichment, availability penalties, and related analytics.
"""

import numpy as np
import pandas as pd

import config
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.analytics")


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
