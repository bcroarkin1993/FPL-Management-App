"""
Player Analytics Functions.

FDR/form enrichment, availability penalties, multi-GW projections,
positional depth analysis, and related analytics.
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
    depth_level: str  # "Critical", "Low", "Adequate"


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
            result[pos] = PositionalDepth(pos, 0, 0, "Critical")
            continue

        healthy = 0
        for _, row in pos_players.iterrows():
            status = row.get("status")
            chance = row.get("chance_of_playing_next_round")

            # Default: if both missing, assume healthy
            if pd.isna(status) and pd.isna(chance):
                healthy += 1
            elif not pd.isna(status) and str(status).lower() == "a":
                healthy += 1
            elif not pd.isna(chance):
                try:
                    if float(chance) >= 75:
                        healthy += 1
                except (ValueError, TypeError):
                    healthy += 1  # Can't parse, assume healthy
            # else: injured/doubtful, not counted

        # Determine depth level
        if healthy <= 1:
            level = "Critical"
        elif healthy < total:
            level = "Low"
        else:
            level = "Adequate"

        result[pos] = PositionalDepth(pos, total, healthy, level)

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
) -> pd.DataFrame:
    """
    Blend FFP multi-GW (Next3GWs) projections into a player DataFrame.

    Matches FFP players by normalized name + team short code.
    Unmatched players or missing FFP data falls back to single_gw_col * 3.

    Args:
        player_df: DataFrame with player names and a single-GW projection column.
        ffp_df: FFP projections DataFrame with Name, Team, Next3GWs columns (or None).
        single_gw_col: Column name for single-GW projection.
        output_col: Column name for the blended multi-GW value.

    Returns:
        player_df with output_col added.
    """
    result = player_df.copy()

    # Determine the player name column
    name_col = "Player" if "Player" in result.columns else "Name" if "Name" in result.columns else None

    # Fallback: single_gw * 3
    single_vals = pd.to_numeric(result.get(single_gw_col, 0), errors="coerce").fillna(0)
    result[output_col] = single_vals * 3

    if ffp_df is None or ffp_df.empty or name_col is None:
        return result

    if "Next3GWs" not in ffp_df.columns or "Name" not in ffp_df.columns:
        return result

    # Build lookup from FFP data: (normalized_name, team_short) -> Next3GWs
    ffp = ffp_df[["Name", "Team", "Next3GWs"]].dropna(subset=["Next3GWs"]).copy()
    ffp["__norm"] = ffp["Name"].apply(canonical_normalize)
    ffp["__team_short"] = ffp["Team"].replace(TEAM_FULL_TO_SHORT)
    ffp["Next3GWs"] = pd.to_numeric(ffp["Next3GWs"], errors="coerce")

    # Build lookup dict
    lookup = {}
    for _, row in ffp.iterrows():
        key = (row["__norm"], str(row["__team_short"]))
        val = row["Next3GWs"]
        if pd.notna(val):
            lookup[key] = val

    if not lookup:
        return result

    # Match players
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
