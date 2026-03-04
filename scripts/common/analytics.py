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
