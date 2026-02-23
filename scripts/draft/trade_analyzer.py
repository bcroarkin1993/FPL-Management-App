# scripts/draft/trade_analyzer.py
"""
Trade Analyzer for Draft FPL leagues.

Identifies mutually beneficial trades by combining:
- Trade Value model (season pts, regression, form, FDR, minutes)
- Positional needs analysis (league percentile per position)
- Acceptance likelihood (value fairness + opponent positional benefit)
- Roster constraint handling (drop suggestions for uneven trades)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from scripts.common.styled_tables import render_styled_table
from scripts.common.player_matching import canonical_normalize
from scripts.common.error_helpers import get_logger
from scripts.common.utils import (
    get_league_entries,
    get_league_player_ownership,
    pull_fpl_player_stats,
)
from scripts.common.fpl_draft_api import (
    get_draft_points_by_position,
    get_draft_team_players_with_points,
    get_fpl_player_mapping,
)
from scripts.draft.waiver_wire import (
    _min_max_norm,
    _avg_fdr_for_team,
    _availability_multiplier,
    _format_availability,
    _load_bootstrap,
)
from scripts.fpl.player_statistics import prepare_advanced_stats_df

_logger = get_logger("fpl_app.draft.trade_analyzer")

# Position label mappings
_POS_LABELS = {"GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
_POS_SHORT_TO_DISPLAY = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}
_POS_DISPLAY_TO_SHORT = {"GK": "G", "DEF": "D", "MID": "M", "FWD": "F"}

# Standard Draft squad size per position
_SQUAD_LIMITS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Default Trade Value weights
_DEFAULT_WEIGHTS = {
    "w_season": 0.30,
    "w_regr": 0.25,
    "w_form": 0.20,
    "w_fdr": 0.15,
    "w_minutes": 0.10,
}

# Dark chart layout (consistent with app style)
_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=14),
    title=dict(font=dict(size=20, color="#ffffff"), x=0.5, xanchor="center"),
    xaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=12)),
    yaxis=dict(gridcolor="#444", zerolinecolor="#444", tickfont=dict(color="#ffffff", size=12)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=12)),
)


# ============================================================================
# DATA LOADING
# ============================================================================

def _load_all_rosters(league_id: int) -> Dict[int, Dict]:
    """
    Load all team rosters with player details.

    Returns:
        {team_id: {"team_name": str, "players": [{"name": str, "position": str,
         "team": str, "player_id": int, "total_points": int}, ...]}}
    """
    ownership = get_league_player_ownership(league_id)
    player_map = get_fpl_player_mapping()

    rosters = {}
    for team_id, blob in ownership.items():
        team_name = blob.get("team_name", f"Team {team_id}")
        players = []
        for pos_short, names in blob.get("players", {}).items():
            pos_display = _POS_SHORT_TO_DISPLAY.get(pos_short, pos_short)
            for name in names:
                # Find player_id from player_map
                pid = None
                norm_name = canonical_normalize(name)
                for p_id, p_info in player_map.items():
                    if canonical_normalize(p_info.get("Player", "")) == norm_name:
                        pid = p_id
                        break
                # Use FPL web_name (e.g., "Beto") for display when available
                p_info = player_map.get(pid, {}) if pid else {}
                web_name = p_info.get("Web_Name")  # None when same as full name
                players.append({
                    "name": name,
                    "display_name": web_name or name,
                    "position": pos_display,
                    "pos_short": pos_short,
                    "team": p_info.get("Team", "???"),
                    "player_id": pid,
                    "total_points": 0,  # filled later
                })
        rosters[int(team_id)] = {"team_name": team_name, "players": players}
    return rosters


def _enrich_with_stats(rosters: Dict, stats_df: pd.DataFrame, current_gw: int,
                       fdr_weeks: int, weights: Dict) -> Dict:
    """
    Enrich roster players with stats, regression, form, FDR, and compute TradeValue.
    """
    if stats_df.empty:
        return rosters

    # Build quick lookup by player_id
    stats_df = stats_df.copy()
    stats_df["id"] = pd.to_numeric(stats_df.get("id"), errors="coerce")
    stats_by_id = {}
    for _, row in stats_df.iterrows():
        pid = row.get("id")
        if pd.notna(pid):
            stats_by_id[int(pid)] = row

    # Prepare advanced stats for regression metrics
    # min_minutes=1 to avoid ZeroDivisionError in calculate_per_90 for
    # players with 0 minutes (0 < 0 is False, so the guard doesn't catch it)
    adv_df = prepare_advanced_stats_df(stats_df, min_minutes=1)
    adv_by_id = {}
    for _, row in adv_df.iterrows():
        pid = row.get("id")
        if pd.notna(pid):
            adv_by_id[int(pid)] = row

    for team_id, team_data in rosters.items():
        for player in team_data["players"]:
            pid = player.get("player_id")
            if pid is None:
                continue

            stats = stats_by_id.get(pid, {})
            adv = adv_by_id.get(pid, {})

            # Basic stats
            player["total_points"] = int(float(stats.get("total_points", 0) or 0))
            player["form"] = float(stats.get("form", 0) or 0)
            player["minutes"] = int(float(stats.get("minutes", 0) or 0))
            player["starts"] = int(float(stats.get("starts", 0) or 0))
            player["goals"] = int(float(stats.get("goals_scored", 0) or 0))
            player["assists"] = int(float(stats.get("assists", 0) or 0))

            # Regression metrics
            player["gi_minus_xgi"] = float(adv.get("gi_minus_xgi", 0) or 0)
            player["g_minus_xg"] = float(adv.get("g_minus_xg", 0) or 0)
            player["a_minus_xa"] = float(adv.get("a_minus_xa", 0) or 0)

            # Start percentage
            player["start_pct"] = float(adv.get("start_pct", 0) or 0)

            # Injury
            player["chance_of_playing"] = stats.get("chance_of_playing_next_round")
            player["status"] = stats.get("status")
            player["news"] = stats.get("news")

            # FDR
            team_short = player.get("team", "???")
            fdr = _avg_fdr_for_team(team_short, current_gw, fdr_weeks)
            player["avg_fdr"] = fdr if fdr is not None else 3.0

            # Availability
            player["availability"] = _availability_multiplier(
                player["chance_of_playing"], player["status"]
            )
            player["availability_str"] = _format_availability(
                player["chance_of_playing"], player["status"], player["news"]
            )

    # Compute TradeValue for all players
    _compute_trade_values(rosters, weights)
    return rosters


def _compute_trade_values(rosters: Dict, weights: Dict):
    """Compute normalized TradeValue for every player across the league."""
    # Collect all players into a flat list for joint normalization
    all_players = []
    for team_data in rosters.values():
        all_players.extend(team_data["players"])

    if not all_players:
        return

    # Extract arrays for normalization
    total_pts = pd.Series([p.get("total_points", 0) for p in all_players], dtype=float)
    # Regression: invert gi_minus_xgi so underperformers get higher value
    regression = pd.Series([-p.get("gi_minus_xgi", 0) for p in all_players], dtype=float)
    form = pd.Series([p.get("form", 0) for p in all_players], dtype=float)
    # FDR ease: lower FDR = easier = higher value
    fdr_ease = pd.Series([6.0 - p.get("avg_fdr", 3.0) for p in all_players], dtype=float)
    start_pct = pd.Series([p.get("start_pct", 0) for p in all_players], dtype=float)

    # Normalize
    pts_norm = _min_max_norm(total_pts).fillna(0.5)
    regr_norm = _min_max_norm(regression).fillna(0.5)
    form_norm = _min_max_norm(form).fillna(0.5)
    fdr_norm = _min_max_norm(fdr_ease).fillna(0.5)
    mins_norm = _min_max_norm(start_pct).fillna(0.5)

    w = weights
    denom = max(w["w_season"] + w["w_regr"] + w["w_form"] + w["w_fdr"] + w["w_minutes"], 1e-9)

    for i, player in enumerate(all_players):
        base_value = (
            w["w_season"] * pts_norm.iloc[i] +
            w["w_regr"] * regr_norm.iloc[i] +
            w["w_form"] * form_norm.iloc[i] +
            w["w_fdr"] * fdr_norm.iloc[i] +
            w["w_minutes"] * mins_norm.iloc[i]
        ) / denom

        player["trade_value"] = round(float(base_value * player.get("availability", 1.0)), 3)
        player["trade_value_raw"] = round(float(base_value), 3)


# ============================================================================
# POSITIONAL NEEDS
# ============================================================================

def _build_pos_pts_from_api(league_id: int, rosters: Dict) -> Dict[int, Dict[str, int]]:
    """
    Build {team_id: {"GK": pts, "DEF": pts, ...}} using the accurate
    GW-by-GW position data from get_draft_points_by_position() — the same
    source as the League Analysis page. This correctly attributes points
    to the team that owned each player at the time they scored.

    Falls back to summing current roster total_points if the API call fails.
    """
    try:
        pos_df = get_draft_points_by_position(league_id)
        if pos_df is not None and not pos_df.empty:
            # Map team_name -> team_id via rosters
            name_to_id = {data["team_name"]: tid for tid, data in rosters.items()}
            team_pos_pts = {}
            for _, row in pos_df.iterrows():
                tid = name_to_id.get(row["Team"])
                if tid is not None:
                    team_pos_pts[tid] = {
                        "GK": int(row.get("GK", 0)),
                        "DEF": int(row.get("DEF", 0)),
                        "MID": int(row.get("MID", 0)),
                        "FWD": int(row.get("FWD", 0)),
                    }
            # Fill any teams missing from the API response
            for tid in rosters:
                if tid not in team_pos_pts:
                    team_pos_pts[tid] = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
            return team_pos_pts
    except Exception:
        _logger.warning("Failed to load position data from API, falling back to roster stats", exc_info=True)

    # Fallback: sum from current roster (less accurate for traded players)
    team_pos_pts = {}
    for team_id, team_data in rosters.items():
        pos_pts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in team_data["players"]:
            pos = p.get("position", "")
            if pos in pos_pts:
                pos_pts[pos] += p.get("total_points", 0)
        team_pos_pts[team_id] = pos_pts
    return team_pos_pts


def _compute_positional_needs(
    team_pos_pts: Dict[int, Dict[str, int]],
) -> Dict[int, Dict[str, float]]:
    """
    For each team+position, compute a need score (0-1).
    High need = weak position relative to league.

    Returns: {team_id: {"GK": 0.8, "DEF": 0.3, ...}}
    """
    needs = {}
    for team_id in team_pos_pts:
        needs[team_id] = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            my_pts = team_pos_pts[team_id][pos]
            all_pts = [team_pos_pts[tid][pos] for tid in team_pos_pts]
            if not all_pts or max(all_pts) == min(all_pts):
                needs[team_id][pos] = 0.5
            else:
                below = sum(1 for p in all_pts if p < my_pts)
                percentile = below / len(all_pts)
                needs[team_id][pos] = round(1.0 - percentile, 2)
    return needs


def _get_positional_rank(
    team_pos_pts: Dict[int, Dict[str, int]],
) -> Dict[int, Dict[str, Tuple[int, int]]]:
    """
    For each team+position, compute (points, rank).
    Returns: {team_id: {"GK": (pts, rank), "DEF": (pts, rank), ...}}
    """
    result = {}
    for team_id in team_pos_pts:
        result[team_id] = {}
        for pos in ["GK", "DEF", "MID", "FWD"]:
            my_pts = team_pos_pts[team_id][pos]
            all_pts_sorted = sorted(
                [team_pos_pts[tid][pos] for tid in team_pos_pts], reverse=True
            )
            rank = all_pts_sorted.index(my_pts) + 1
            result[team_id][pos] = (my_pts, rank)
    return result


# ============================================================================
# TRADE DISCOVERY ENGINE
# ============================================================================

def _find_1_for_1_trades(
    my_team_id: int,
    rosters: Dict,
    needs: Dict,
    num_teams: int,
) -> List[Dict]:
    """Find 1-for-1 trade proposals."""
    proposals = []
    my_roster = rosters[my_team_id]["players"]
    my_needs = needs[my_team_id]

    for opp_id, opp_data in rosters.items():
        if opp_id == my_team_id:
            continue
        opp_roster = opp_data["players"]
        opp_needs = needs[opp_id]

        # For each position I need improvement at
        for pos in ["GK", "DEF", "MID", "FWD"]:
            my_need = my_needs[pos]
            opp_need_for_pos = opp_needs[pos]

            # I want to receive at this position — find what I can send
            # Look for complementary positions where I'm strong and they're weak
            for send_pos in ["GK", "DEF", "MID", "FWD"]:
                my_send_need = my_needs[send_pos]
                opp_receive_need = opp_needs[send_pos]

                # Skip if I need this position too (I shouldn't weaken myself)
                if my_send_need > 0.6:
                    continue

                # My players at send_pos sorted by trade_value (send worst)
                my_at_send = sorted(
                    [p for p in my_roster if p["position"] == send_pos],
                    key=lambda x: x.get("trade_value", 0),
                )
                # Their players at receive_pos sorted by trade_value desc (get best)
                their_at_pos = sorted(
                    [p for p in opp_roster if p["position"] == pos],
                    key=lambda x: x.get("trade_value", 0),
                    reverse=True,
                )

                if not my_at_send or not their_at_pos:
                    continue

                # Consider bottom 2 of mine, top 3 of theirs
                for send_p in my_at_send[:2]:
                    for recv_p in their_at_pos[:3]:
                        # Skip if I'd be sending a better player for a worse one
                        if send_p["trade_value"] > recv_p["trade_value"] * 1.3:
                            continue

                        proposal = _score_proposal(
                            my_team_id, opp_id, [send_p], [recv_p],
                            rosters, needs, num_teams,
                        )
                        if proposal and proposal["trade_score"] > 0.05:
                            proposals.append(proposal)

    return proposals


def _find_2_for_2_trades(
    my_team_id: int,
    rosters: Dict,
    needs: Dict,
    num_teams: int,
) -> List[Dict]:
    """
    Find position-mirrored 2-for-2 trades that avoid drops.

    Structure: swap within the SAME two positions so roster counts stay balanced.
    - upgrade_pos: position where I'm weak, opponent is strong → I upgrade here
    - sweetener_pos: position where I'm strong, opponent is weak → I give here
    - I send:    my worst at upgrade_pos  + my best at sweetener_pos
    - I receive: their best at upgrade_pos + their worst at sweetener_pos
    """
    proposals = []
    my_roster = rosters[my_team_id]["players"]
    my_needs = needs[my_team_id]
    _MIN_CALIBER = 0.10  # skip replacement-level players
    # GK excluded from sweetener role — only 20 starting GKs makes them
    # too scarce to use as trade filler
    _SWEETENER_POSITIONS = ["DEF", "MID", "FWD"]

    for opp_id, opp_data in rosters.items():
        if opp_id == my_team_id:
            continue
        opp_roster = opp_data["players"]
        opp_needs = needs[opp_id]

        # For each pair of positions (upgrade_pos, sweetener_pos)
        for upgrade_pos in ["GK", "DEF", "MID", "FWD"]:
            for sweetener_pos in _SWEETENER_POSITIONS:
                if upgrade_pos == sweetener_pos:
                    continue

                # I want to upgrade at upgrade_pos (some need)
                # I can afford to give at sweetener_pos (low need)
                # Opponent wants upgrade at sweetener_pos (some need)
                # Opponent can afford to give at upgrade_pos
                if my_needs[sweetener_pos] > 0.6:
                    continue  # I need sweetener_pos too much
                if opp_needs[sweetener_pos] < 0.2:
                    continue  # Opponent doesn't want my sweetener_pos

                # My players at each position
                my_at_upgrade = sorted(
                    [p for p in my_roster if p["position"] == upgrade_pos],
                    key=lambda x: x.get("trade_value", 0),  # worst first (I send worst)
                )
                my_at_sweetener = sorted(
                    [p for p in my_roster if p["position"] == sweetener_pos],
                    key=lambda x: x.get("trade_value", 0), reverse=True,  # best first (I send best)
                )
                # Their players at each position
                their_at_upgrade = sorted(
                    [p for p in opp_roster if p["position"] == upgrade_pos],
                    key=lambda x: x.get("trade_value", 0), reverse=True,  # best first (I get best)
                )
                their_at_sweetener = sorted(
                    [p for p in opp_roster if p["position"] == sweetener_pos],
                    key=lambda x: x.get("trade_value", 0),  # worst first (I get worst)
                )

                if not (my_at_upgrade and my_at_sweetener and their_at_upgrade and their_at_sweetener):
                    continue

                # Try top 2 candidates per slot
                for s_upgrade in my_at_upgrade[:2]:
                    for s_sweet in my_at_sweetener[:2]:
                        if s_upgrade["name"] == s_sweet["name"]:
                            continue
                        for r_upgrade in their_at_upgrade[:2]:
                            for r_sweet in their_at_sweetener[:2]:
                                if r_upgrade["name"] == r_sweet["name"]:
                                    continue

                                send_players = [s_upgrade, s_sweet]
                                recv_players = [r_upgrade, r_sweet]

                                # No overlap
                                if {p["name"] for p in send_players} & {p["name"] for p in recv_players}:
                                    continue

                                # Caliber floor — skip if any player is replacement-level
                                all_tv = [p.get("trade_value", 0) for p in send_players + recv_players]
                                if min(all_tv) < _MIN_CALIBER:
                                    continue

                                # The upgrade should actually be an upgrade for me
                                if r_upgrade.get("trade_value", 0) <= s_upgrade.get("trade_value", 0):
                                    continue

                                proposal = _score_proposal(
                                    my_team_id, opp_id, send_players, recv_players,
                                    rosters, needs, num_teams,
                                )
                                if proposal and proposal["trade_score"] > 0.03:
                                    proposals.append(proposal)

    return proposals


def _find_2_for_1_trades(
    my_team_id: int,
    rosters: Dict,
    needs: Dict,
    num_teams: int,
) -> List[Dict]:
    """
    Find 2-for-1 trades where the "1" player's position is always
    represented on the "2" side (position anchor).

    Direction 1 (I send 2, receive 1 star):
      I receive their strong FWD, I send my weaker FWD + a sweetener at
      a position they need. They keep their FWD slot filled.

    Direction 2 (I send 1 star, receive 2):
      I send my strong FWD, I receive their decent FWD (replacement) +
      a bonus player at a position I need. I keep my FWD slot filled.

    Both players on the "2" side must be above replacement level.
    """
    proposals = []
    my_roster = rosters[my_team_id]["players"]
    my_needs = needs[my_team_id]
    _MIN_CALIBER = 0.12  # both players on "2" side must be above this
    # GK excluded from filler roles — positional scarcity (only 20 starters)
    _FILLER_POSITIONS = ["DEF", "MID", "FWD"]

    for opp_id, opp_data in rosters.items():
        if opp_id == my_team_id:
            continue
        opp_roster = opp_data["players"]
        opp_needs = needs[opp_id]

        # Direction 1: I send 2, receive 1 star player
        # anchor_pos = position of the star player (same pos on both sides)
        for anchor_pos in ["GK", "DEF", "MID", "FWD"]:
            # Their best at anchor_pos — the star I want
            their_stars = sorted(
                [p for p in opp_roster if p["position"] == anchor_pos],
                key=lambda x: x.get("trade_value", 0), reverse=True,
            )
            # My worst at anchor_pos — position replacement I send back
            my_at_anchor = sorted(
                [p for p in my_roster if p["position"] == anchor_pos],
                key=lambda x: x.get("trade_value", 0),  # worst first
            )
            if not their_stars or not my_at_anchor:
                continue

            for target in their_stars[:2]:
                for my_anchor_send in my_at_anchor[:2]:
                    # Must be an upgrade for me
                    if target.get("trade_value", 0) <= my_anchor_send.get("trade_value", 0):
                        continue

                    # Find a sweetener from another position they need
                    for sweet_pos in _FILLER_POSITIONS:
                        if sweet_pos == anchor_pos:
                            continue
                        if opp_needs.get(sweet_pos, 0) < 0.3:
                            continue  # they don't need this position

                        my_sweeteners = sorted(
                            [p for p in my_roster if p["position"] == sweet_pos
                             and p["name"] != my_anchor_send["name"]],
                            key=lambda x: x.get("trade_value", 0), reverse=True,
                        )
                        for sweetener in my_sweeteners[:2]:
                            # Caliber floor
                            if my_anchor_send.get("trade_value", 0) < _MIN_CALIBER:
                                continue
                            if sweetener.get("trade_value", 0) < _MIN_CALIBER:
                                continue

                            proposal = _score_proposal(
                                my_team_id, opp_id,
                                [my_anchor_send, sweetener], [target],
                                rosters, needs, num_teams,
                            )
                            if proposal and proposal["trade_score"] > 0.03:
                                proposals.append(proposal)

        # Direction 2: I send 1 star, receive 2 (replacement + bonus)
        for anchor_pos in ["GK", "DEF", "MID", "FWD"]:
            if my_needs[anchor_pos] > 0.5:
                continue  # Can't weaken a needed position

            # My best at anchor_pos — the star I send
            my_stars = sorted(
                [p for p in my_roster if p["position"] == anchor_pos],
                key=lambda x: x.get("trade_value", 0), reverse=True,
            )
            # Their best at anchor_pos — replacement I get back
            their_at_anchor = sorted(
                [p for p in opp_roster if p["position"] == anchor_pos],
                key=lambda x: x.get("trade_value", 0), reverse=True,
            )
            if not my_stars or not their_at_anchor:
                continue

            for to_send in my_stars[:1]:
                for replacement in their_at_anchor[:2]:
                    # Replacement should be decent but less than what I send
                    if replacement.get("trade_value", 0) >= to_send.get("trade_value", 0):
                        continue

                    # Find a bonus player at a position I need
                    for bonus_pos in _FILLER_POSITIONS:
                        if bonus_pos == anchor_pos:
                            continue
                        if my_needs.get(bonus_pos, 0) < 0.3:
                            continue  # I don't need this position

                        their_bonus = sorted(
                            [p for p in opp_roster if p["position"] == bonus_pos
                             and p["name"] != replacement["name"]],
                            key=lambda x: x.get("trade_value", 0), reverse=True,
                        )
                        for bonus in their_bonus[:2]:
                            # Caliber floor — both received must be solid
                            if replacement.get("trade_value", 0) < _MIN_CALIBER:
                                continue
                            if bonus.get("trade_value", 0) < _MIN_CALIBER:
                                continue

                            proposal = _score_proposal(
                                my_team_id, opp_id,
                                [to_send], [replacement, bonus],
                                rosters, needs, num_teams,
                            )
                            if proposal and proposal["trade_score"] > 0.03:
                                proposals.append(proposal)

    return proposals


# ============================================================================
# TRADE SCORING
# ============================================================================

def _score_proposal(
    my_team_id: int,
    opp_id: int,
    send_players: List[Dict],
    recv_players: List[Dict],
    rosters: Dict,
    needs: Dict,
    num_teams: int,
) -> Optional[Dict]:
    """Score a trade proposal and return structured proposal dict (or None if invalid)."""
    my_needs = needs[my_team_id]
    opp_needs = needs[opp_id]

    # Total trade values
    send_value = sum(p.get("trade_value", 0) for p in send_players)
    recv_value = sum(p.get("trade_value", 0) for p in recv_players)

    # My positional improvement
    my_pos_gain = 0.0
    for p in recv_players:
        pos = p.get("position", "")
        if pos in my_needs:
            my_pos_gain += my_needs[pos] * p.get("trade_value", 0)
    for p in send_players:
        pos = p.get("position", "")
        if pos in my_needs:
            my_pos_gain -= (1.0 - my_needs[pos]) * p.get("trade_value", 0)

    # Net value gain for me
    net_value = recv_value - send_value

    # Fairness: how close are the total values (within ~15%)
    total_involved = max(send_value + recv_value, 1e-9)
    value_diff_pct = abs(send_value - recv_value) / (total_involved / 2)
    fairness = max(0.0, 1.0 - value_diff_pct / 0.3)

    # Acceptance likelihood
    opp_pos_benefit = 0.0
    for p in send_players:
        pos = p.get("position", "")
        if pos in opp_needs:
            opp_pos_benefit += opp_needs[pos] * p.get("trade_value", 0)
    for p in recv_players:
        pos = p.get("position", "")
        if pos in opp_needs:
            opp_pos_benefit -= (1.0 - opp_needs[pos]) * p.get("trade_value", 0)

    # Perception boost (form of players they receive)
    perception = np.mean([p.get("form", 0) for p in send_players]) if send_players else 0
    perception_norm = min(perception / 10.0, 1.0)

    value_fairness = fairness
    acceptance = (
        0.40 * value_fairness +
        0.40 * min(max(opp_pos_benefit, 0), 1) +
        0.20 * perception_norm
    )
    acceptance = min(max(acceptance, 0), 1)

    # Drop suggestion for uneven trades or cross-position
    # (computed before score so we can penalize trades requiring a drop)
    drop_suggestion = _get_drop_suggestion(
        my_team_id, opp_id, send_players, recv_players, rosters
    )

    # Player caliber premium — trades involving top players are more
    # impactful than swapping waiver-wire-level players
    all_involved = send_players + recv_players
    avg_caliber = np.mean([p.get("trade_value", 0) for p in all_involved]) if all_involved else 0
    caliber_norm = min(avg_caliber / 0.3, 1.0)  # 0.3 TV ≈ solid starter

    # Overall trade score
    # Normalize components to [0,1]
    pos_gain_norm = min(max(my_pos_gain, 0), 1)
    net_value_norm = min(max((net_value + 0.5) / 1.0, 0), 1)

    trade_score = (
        0.20 * pos_gain_norm +
        0.10 * net_value_norm +
        0.15 * acceptance +
        0.40 * caliber_norm +
        0.15 * fairness
    )

    # Near-zero multiplier for trades requiring a drop — these are
    # essentially non-starters in practice
    if drop_suggestion:
        trade_score *= 0.05

    # Determine trade type
    n_send = len(send_players)
    n_recv = len(recv_players)
    if n_send == 1 and n_recv == 1:
        trade_type = "1-for-1"
    elif n_send == 2 and n_recv == 2:
        trade_type = "2-for-2"
    else:
        trade_type = f"{n_send}-for-{n_recv}"

    # Acceptance label
    if acceptance >= 0.6:
        accept_label = "High"
    elif acceptance >= 0.35:
        accept_label = "Medium"
    else:
        accept_label = "Low"

    # Generate human-readable description
    description = _generate_trade_description(
        send_players, recv_players, my_needs, opp_needs,
        rosters[opp_id]["team_name"],
    )

    return {
        "trade_type": trade_type,
        "opp_id": opp_id,
        "opp_name": rosters[opp_id]["team_name"],
        "send": send_players,
        "receive": recv_players,
        "send_value": round(send_value, 3),
        "recv_value": round(recv_value, 3),
        "trade_score": round(trade_score, 3),
        "acceptance": round(acceptance, 2),
        "accept_label": accept_label,
        "fairness": round(fairness, 2),
        "my_pos_gain": round(my_pos_gain, 3),
        "net_value": round(net_value, 3),
        "drop_suggestion": drop_suggestion,
        "description": description,
    }


def _generate_trade_description(
    send_players: List[Dict],
    recv_players: List[Dict],
    my_needs: Dict[str, float],
    opp_needs: Dict[str, float],
    opp_name: str,
) -> str:
    """Generate a human-readable description of why this trade is beneficial."""
    parts = []

    # Identify positions being strengthened
    recv_positions = set(p.get("position", "") for p in recv_players)
    send_positions = set(p.get("position", "") for p in send_players)

    # Which received positions are weak spots for me?
    weak_recv = [pos for pos in recv_positions if my_needs.get(pos, 0) >= 0.5]
    strong_send = [pos for pos in send_positions if my_needs.get(pos, 0) <= 0.4]

    if weak_recv and strong_send:
        weak_str = " + ".join(weak_recv)
        strong_str = " + ".join(strong_send)
        parts.append(f"Boosts your weak {weak_str} by trading from your strong {strong_str}.")
    elif weak_recv:
        weak_str = " + ".join(weak_recv)
        parts.append(f"Strengthens your weak {weak_str} position.")
    elif strong_send:
        strong_str = " + ".join(strong_send)
        parts.append(f"Moves surplus {strong_str} depth to {opp_name}.")

    # Check for regression buy-low opportunities
    buy_low_names = []
    for p in recv_players:
        if p.get("gi_minus_xgi", 0) < -0.5:
            buy_low_names.append(p.get("display_name", p["name"]))
    if buy_low_names:
        names_str = " and ".join(buy_low_names)
        parts.append(f"Buy-low: {names_str} underperforming xGI, due for positive regression.")

    # Check for sell-high opportunities
    sell_high_names = []
    for p in send_players:
        if p.get("gi_minus_xgi", 0) > 0.5:
            sell_high_names.append(p.get("display_name", p["name"]))
    if sell_high_names:
        names_str = " and ".join(sell_high_names)
        parts.append(f"Sell-high: {names_str} overperforming, may regress.")

    # Check if opponent also benefits (makes acceptance likely)
    opp_weak_recv = [pos for pos in send_positions if opp_needs.get(pos, 0) >= 0.5]
    if opp_weak_recv:
        opp_str = " + ".join(opp_weak_recv)
        parts.append(f"{opp_name} also needs {opp_str}, making acceptance more likely.")

    if not parts:
        parts.append("Balanced swap that improves overall roster composition.")

    return " ".join(parts)


def _check_team_drop(roster: List[Dict], outgoing: List[Dict],
                     incoming: List[Dict], team_label: str) -> Optional[str]:
    """Check if a team needs to drop a player after a trade."""
    n_out = len(outgoing)
    n_in = len(incoming)

    # Position counts after trade
    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for p in roster:
        pos = p.get("position", "")
        if pos in pos_counts:
            pos_counts[pos] += 1
    for p in outgoing:
        pos = p.get("position", "")
        if pos in pos_counts:
            pos_counts[pos] -= 1
    for p in incoming:
        pos = p.get("position", "")
        if pos in pos_counts:
            pos_counts[pos] += 1

    # Roster size overflow (receives more than sends)
    if n_in > n_out:
        remaining = [p for p in roster if p["name"] not in {s["name"] for s in outgoing}]
        remaining.extend(incoming)
        if remaining:
            worst = min(remaining, key=lambda x: x.get("trade_value", 0))
            return (f"{team_label} drops {worst.get('display_name', worst['name'])} "
                    f"({worst.get('position', '?')}, Trade Value: {worst.get('trade_value', 0):.2f})")

    # Position overflow
    for pos, limit in _SQUAD_LIMITS.items():
        if pos_counts.get(pos, 0) > limit:
            remaining_at_pos = [
                p for p in roster
                if p["position"] == pos and p["name"] not in {s["name"] for s in outgoing}
            ]
            recv_at_pos = [p for p in incoming if p["position"] == pos]
            all_at_pos = remaining_at_pos + recv_at_pos
            if all_at_pos:
                worst = min(all_at_pos, key=lambda x: x.get("trade_value", 0))
                return (f"{team_label} drops {worst.get('display_name', worst['name'])} "
                        f"({pos}, Trade Value: {worst.get('trade_value', 0):.2f}) — exceeds {pos} limit")

    return None


def _get_drop_suggestion(
    my_team_id: int,
    opp_id: int,
    send_players: List[Dict],
    recv_players: List[Dict],
    rosters: Dict,
) -> Optional[str]:
    """
    If the trade creates a roster imbalance for EITHER side
    (cross-position or uneven), suggest who to drop.
    """
    my_roster = rosters[my_team_id]["players"]
    opp_roster = rosters[opp_id]["players"]

    drops = []

    # Check my side: I send outgoing, receive incoming
    my_drop = _check_team_drop(my_roster, send_players, recv_players, "You")
    if my_drop:
        drops.append(my_drop)

    # Check opponent side: they send recv_players, receive send_players
    opp_name = rosters[opp_id]["team_name"]
    opp_drop = _check_team_drop(opp_roster, recv_players, send_players, opp_name)
    if opp_drop:
        drops.append(opp_drop)

    return " | ".join(drops) if drops else None


# ============================================================================
# UI RENDERING
# ============================================================================

def _render_positional_profile(team_id: int, rosters: Dict, pos_ranks: Dict,
                               needs: Dict, team_pos_pts: Dict):
    """Render positional strength cards and bar chart."""
    my_ranks = pos_ranks.get(team_id, {})
    my_needs_data = needs.get(team_id, {})
    num_teams = len(rosters)

    # Compute league averages from API-sourced position points
    league_avg = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for tid, pos_pts in team_pos_pts.items():
        for pos in league_avg:
            league_avg[pos] += pos_pts.get(pos, 0)
    for pos in league_avg:
        league_avg[pos] = league_avg[pos] / max(len(team_pos_pts), 1)

    # Position cards
    pos_colors = {"GK": "#f0c040", "DEF": "#4caf50", "MID": "#2196f3", "FWD": "#e91e63"}
    cards_html = '<div style="display:flex;gap:12px;margin-bottom:1rem;flex-wrap:wrap;">'

    for pos in ["GK", "DEF", "MID", "FWD"]:
        pts, rank = my_ranks.get(pos, (0, 0))
        need = my_needs_data.get(pos, 0.5)
        color = pos_colors[pos]

        if need >= 0.6:
            need_icon = "&#x26A0;&#xFE0F;"  # warning sign
            border_color = "#e74c3c"
        elif need <= 0.3:
            need_icon = "&#x2705;"  # check mark
            border_color = "#4ecca3"
        else:
            need_icon = ""
            border_color = "#444"

        cards_html += (
            f'<div style="flex:1;min-width:140px;background:linear-gradient(135deg,#1a1a2e,#16213e);'
            f'border:2px solid {border_color};border-radius:10px;padding:14px;text-align:center;color:#e0e0e0;">'
            f'<div style="font-size:0.8em;color:{color};font-weight:700;margin-bottom:4px;">{pos}</div>'
            f'<div style="font-size:1.5em;font-weight:800;color:#fff;">{pts}</div>'
            f'<div style="font-size:0.8em;color:#999;">pts</div>'
            f'<div style="font-size:0.85em;margin-top:6px;">'
            f'#{rank}/{num_teams} {need_icon}</div>'
            f'</div>'
        )
    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)

    # Bar chart: My pts vs League Avg
    positions = ["GK", "DEF", "MID", "FWD"]
    my_pts = [my_ranks.get(p, (0, 0))[0] for p in positions]
    avg_pts = [round(league_avg.get(p, 0)) for p in positions]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Your Team", x=positions, y=my_pts,
        marker_color="#00ff87", text=my_pts, textposition="outside",
        textfont=dict(color="#ffffff"),
    ))
    fig.add_trace(go.Bar(
        name="League Avg", x=positions, y=avg_pts,
        marker_color="#5a0060", text=avg_pts, textposition="outside",
        textfont=dict(color="#ffffff"),
    ))
    fig.update_layout(
        barmode="group",
        height=400,
        margin=dict(t=80, b=40, l=70, r=20),
        **_DARK_CHART_LAYOUT,
    )
    fig.update_layout(title=dict(
        text="Points by Position: You vs League Average",
        font=dict(size=20, color="#ffffff"),
        x=0.5,
        xanchor="center",
        y=0.95,
    ))
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, use_container_width=True)


def _render_trade_card(proposal: Dict, idx: int):
    """Render a single trade proposal card."""
    accept_colors = {"High": "#4ecca3", "Medium": "#ffa726", "Low": "#e74c3c"}
    accept_bg = {"High": "#1a472a", "Medium": "#4a3728", "Low": "#5f2121"}

    accept_label = proposal["accept_label"]
    accept_color = accept_colors.get(accept_label, "#999")
    accept_bg_color = accept_bg.get(accept_label, "#333")

    # Header
    card_html = (
        f'<div style="border:1px solid #444;border-radius:10px;padding:16px;margin-bottom:12px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#e0e0e0;">'
        # Top row: trade type, partner, acceptance
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="background:#0f3460;color:#e0e0e0;padding:3px 10px;border-radius:12px;'
        f'font-size:0.8em;font-weight:bold;">{proposal["trade_type"]}</span>'
        f'<span style="font-weight:700;color:#fff;">Trade with {proposal["opp_name"]}</span>'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="background:{accept_bg_color};color:{accept_color};padding:3px 10px;'
        f'border-radius:12px;font-size:0.8em;font-weight:bold;">{accept_label} Acceptance</span>'
        f'</div></div>'
    )

    # Trade description
    description = proposal.get("description", "")
    if description:
        card_html += (
            f'<div style="color:#b0b0b0;font-size:0.85em;font-style:italic;'
            f'margin-bottom:12px;padding:6px 8px;border-left:3px solid #4ecca3;'
            f'background:rgba(78,204,163,0.05);">'
            f'{description}</div>'
        )

    # Trade details: SEND -> RECEIVE
    card_html += '<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">'

    # SEND column
    card_html += '<div style="flex:1;">'
    card_html += '<div style="color:#e74c3c;font-weight:bold;font-size:0.8em;margin-bottom:6px;">YOU SEND</div>'
    for p in proposal["send"]:
        gi = p.get("gi_minus_xgi", 0)
        gi_color = "#e74c3c" if gi > 0.5 else "#4ecca3" if gi < -0.5 else "#999"
        card_html += (
            f'<div style="background:#0d1117;border-radius:8px;padding:8px 10px;margin-bottom:4px;">'
            f'<div style="font-weight:700;color:#e0e0e0;">{p.get("display_name", p["name"])}</div>'
            f'<div style="font-size:0.8em;color:#999;">'
            f'{p.get("position", "?")} &bull; {p.get("team", "?")} &bull; '
            f'Trade Value: {p.get("trade_value", 0):.2f} &bull; Form: {p.get("form", 0):.1f} &bull; '
            f'<span style="color:{gi_color}">GI-xGI: {gi:+.1f}</span>'
            f'</div></div>'
        )
    card_html += '</div>'

    # Arrow
    card_html += '<div style="color:#888;font-size:2em;padding-top:24px;align-self:center;">&harr;</div>'

    # RECEIVE column
    card_html += '<div style="flex:1;text-align:right;">'
    card_html += '<div style="color:#4ecca3;font-weight:bold;font-size:0.8em;margin-bottom:6px;">YOU RECEIVE</div>'
    for p in proposal["receive"]:
        gi = p.get("gi_minus_xgi", 0)
        gi_color = "#e74c3c" if gi > 0.5 else "#4ecca3" if gi < -0.5 else "#999"
        card_html += (
            f'<div style="background:#0d1117;border-radius:8px;padding:8px 10px;margin-bottom:4px;text-align:right;">'
            f'<div style="font-weight:700;color:#e0e0e0;">{p.get("display_name", p["name"])}</div>'
            f'<div style="font-size:0.8em;color:#999;">'
            f'{p.get("position", "?")} &bull; {p.get("team", "?")} &bull; '
            f'Trade Value: {p.get("trade_value", 0):.2f} &bull; Form: {p.get("form", 0):.1f} &bull; '
            f'<span style="color:{gi_color}">GI-xGI: {gi:+.1f}</span>'
            f'</div></div>'
        )
    card_html += '</div></div>'

    # Footer: fairness + drop suggestion (removed confusing "net value" metric)
    footer_parts = [
        f'Fairness: {proposal["fairness"]:.0%}',
    ]
    if proposal.get("drop_suggestion"):
        footer_parts.append(
            f'<span style="color:#ffa726;">Drop needed: {proposal["drop_suggestion"]}</span>'
        )

    card_html += (
        f'<div style="color:#aaa;font-size:0.82em;border-top:1px solid #333;padding-top:8px;margin-top:8px;">'
        f'{" &bull; ".join(footer_parts)}'
        f'</div></div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)


def _render_explore_tab(my_team_id: int, rosters: Dict, needs: Dict):
    """Render the Explore Teams tab with side-by-side roster comparison."""
    opp_options = {
        tid: data["team_name"]
        for tid, data in rosters.items()
        if tid != my_team_id
    }
    if not opp_options:
        st.info("No opponent teams found.")
        return

    selected_name = st.selectbox(
        "Select team to compare",
        options=list(opp_options.values()),
        key="explore_team_select",
    )
    selected_id = next(tid for tid, name in opp_options.items() if name == selected_name)

    my_data = rosters[my_team_id]
    opp_data = rosters[selected_id]
    my_needs_data = needs[my_team_id]
    opp_needs_data = needs[selected_id]

    col1, col2 = st.columns(2)

    for col, team_data, team_needs, label in [
        (col1, my_data, my_needs_data, f"Your Team: {my_data['team_name']}"),
        (col2, opp_data, opp_needs_data, f"Opponent: {opp_data['team_name']}"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#37003c,#5a0060);color:#00ff87;'
                f'padding:8px 12px;border-radius:8px;font-weight:700;margin-bottom:8px;">{label}</div>',
                unsafe_allow_html=True,
            )

            for pos in ["GK", "DEF", "MID", "FWD"]:
                need = team_needs.get(pos, 0.5)
                need_color = "#e74c3c" if need >= 0.6 else "#4ecca3" if need <= 0.3 else "#ffa726"
                pos_players = sorted(
                    [p for p in team_data["players"] if p["position"] == pos],
                    key=lambda x: x.get("trade_value", 0), reverse=True,
                )

                st.markdown(
                    f'<div style="color:{need_color};font-weight:700;font-size:0.9em;'
                    f'margin:8px 0 4px;">{"&#x26A0;" if need >= 0.6 else ""} {pos} '
                    f'(Need: {need:.0%})</div>',
                    unsafe_allow_html=True,
                )

                if pos_players:
                    rows = []
                    for p in pos_players:
                        rows.append({
                            "Player": p.get("display_name", p["name"]),
                            "Club": p.get("team", "?"),
                            "Pts": p.get("total_points", 0),
                            "Trade Value": p.get("trade_value", 0),
                            "Form": p.get("form", 0),
                            "GI-xGI": p.get("gi_minus_xgi", 0),
                        })
                    df = pd.DataFrame(rows)
                    render_styled_table(
                        df,
                        col_formats={"Trade Value": "{:.2f}", "Form": "{:.1f}", "GI-xGI": "{:+.1f}"},
                        positive_color_cols=["Trade Value", "Form"],
                    )


def _render_regression_tab(my_team_id: int, rosters: Dict):
    """Render the Regression Watch tab."""
    my_names = {p["name"] for p in rosters[my_team_id]["players"]}

    # Collect all league players
    all_players = []
    for tid, data in rosters.items():
        for p in data["players"]:
            all_players.append({**p, "owner": data["team_name"], "is_mine": tid == my_team_id})

    if not all_players:
        st.info("No player data available.")
        return

    # Buy Low Targets: others' players with negative GI-xGI (underperforming)
    st.subheader("Buy Low Targets")
    st.caption("Players owned by opponents who are underperforming their expected stats — likely to improve.")
    buy_low = sorted(
        [p for p in all_players if not p["is_mine"] and p.get("gi_minus_xgi", 0) < -0.5],
        key=lambda x: x.get("gi_minus_xgi", 0),
    )[:15]

    if buy_low:
        rows = [{
            "Player": p.get("display_name", p["name"]),
            "Owner": p["owner"],
            "Pos": p.get("position", "?"),
            "Club": p.get("team", "?"),
            "Pts": p.get("total_points", 0),
            "GI-xGI": p.get("gi_minus_xgi", 0),
            "G-xG": p.get("g_minus_xg", 0),
            "A-xA": p.get("a_minus_xa", 0),
            "Form": p.get("form", 0),
            "Trade Value": p.get("trade_value", 0),
        } for p in buy_low]
        df = pd.DataFrame(rows)
        render_styled_table(
            df,
            col_formats={
                "GI-xGI": "{:+.1f}", "G-xG": "{:+.1f}", "A-xA": "{:+.1f}",
                "Form": "{:.1f}", "Trade Value": "{:.2f}",
            },
            positive_color_cols=["Trade Value"],
        )
    else:
        st.info("No significant underperformers found among opponents' players.")

    # Sell High Candidates: my players with positive GI-xGI (overperforming)
    st.subheader("Sell High Candidates")
    st.caption("Your players who are overperforming their expected stats — may regress downward.")
    sell_high = sorted(
        [p for p in all_players if p["is_mine"] and p.get("gi_minus_xgi", 0) > 0.5],
        key=lambda x: x.get("gi_minus_xgi", 0),
        reverse=True,
    )[:10]

    if sell_high:
        rows = [{
            "Player": p.get("display_name", p["name"]),
            "Pos": p.get("position", "?"),
            "Club": p.get("team", "?"),
            "Pts": p.get("total_points", 0),
            "GI-xGI": p.get("gi_minus_xgi", 0),
            "G-xG": p.get("g_minus_xg", 0),
            "A-xA": p.get("a_minus_xa", 0),
            "Form": p.get("form", 0),
            "Trade Value": p.get("trade_value", 0),
        } for p in sell_high]
        df = pd.DataFrame(rows)
        render_styled_table(
            df,
            col_formats={
                "GI-xGI": "{:+.1f}", "G-xG": "{:+.1f}", "A-xA": "{:+.1f}",
                "Form": "{:.1f}", "Trade Value": "{:.2f}",
            },
            positive_color_cols=["Trade Value"],
        )
    else:
        st.info("None of your players are significantly overperforming.")


# ============================================================================
# MAIN PAGE
# ============================================================================

def show_trade_analyzer_page():
    """Main entry point for the Trade Analyzer page."""
    st.header("Trade Analyzer")
    st.caption(
        "Find trades that strengthen your weak positions by targeting "
        "undervalued players due for positive regression."
    )

    league_id = config.FPL_DRAFT_LEAGUE_ID
    if not league_id:
        st.error("No Draft league configured. Set FPL_DRAFT_LEAGUE_ID in .env")
        return

    # Load rosters
    try:
        rosters = _load_all_rosters(league_id)
    except Exception as e:
        st.error(f"Failed to load league rosters: {e}")
        _logger.warning("Failed to load rosters", exc_info=True)
        return

    if not rosters:
        st.warning("No roster data found. The season may not have started yet.")
        return

    # Team selector
    team_options = sorted(
        [(tid, data["team_name"]) for tid, data in rosters.items()],
        key=lambda x: x[1].lower(),
    )
    default_tid = config.FPL_DRAFT_TEAM_ID
    default_idx = 0
    for i, (tid, _) in enumerate(team_options):
        if str(tid) == str(default_tid):
            default_idx = i
            break

    selected_label = st.selectbox(
        "Your Team",
        options=[f"{name} ({tid})" for tid, name in team_options],
        index=default_idx,
    )
    my_team_id = team_options[[f"{name} ({tid})" for tid, name in team_options].index(selected_label)][0]

    # Load player stats
    try:
        stats_df = pull_fpl_player_stats()
    except Exception as e:
        st.error(f"Failed to load player stats: {e}")
        _logger.warning("Failed to load stats", exc_info=True)
        stats_df = pd.DataFrame()

    # Current GW
    try:
        current_gw = int(config.CURRENT_GAMEWEEK)
    except Exception:
        current_gw = 1

    # Filters & Weights
    with st.expander("Filters & Weights", expanded=False):
        col_types, col_fdr = st.columns(2)
        with col_types:
            trade_types = st.multiselect(
                "Trade Types",
                ["1-for-1", "2-for-2", "2-for-1"],
                default=["1-for-1", "2-for-2", "2-for-1"],
                key="ta_trade_types",
            )
        with col_fdr:
            fdr_weeks = int(st.number_input(
                "FDR Lookahead (weeks)", min_value=1, max_value=8,
                value=config.UPCOMING_WEEKS_DEFAULT, key="ta_fdr_weeks",
            ))

        st.markdown("**Trade Value Weights:**")
        wc1, wc2, wc3, wc4, wc5 = st.columns(5)
        w_season = float(wc1.slider("Season Pts", 0.0, 1.0, 0.30, 0.05, key="ta_w_season"))
        w_regr = float(wc2.slider("Regression", 0.0, 1.0, 0.25, 0.05, key="ta_w_regr"))
        w_form = float(wc3.slider("Form", 0.0, 1.0, 0.20, 0.05, key="ta_w_form"))
        w_fdr = float(wc4.slider("FDR Ease", 0.0, 1.0, 0.15, 0.05, key="ta_w_fdr"))
        w_minutes = float(wc5.slider("Minutes", 0.0, 1.0, 0.10, 0.05, key="ta_w_minutes"))

    weights = {
        "w_season": w_season, "w_regr": w_regr, "w_form": w_form,
        "w_fdr": w_fdr, "w_minutes": w_minutes,
    }

    # Enrich rosters with stats and compute trade values
    rosters = _enrich_with_stats(rosters, stats_df, current_gw, fdr_weeks, weights)

    # Compute positional needs (using accurate GW-by-GW API data)
    team_pos_pts = _build_pos_pts_from_api(league_id, rosters)
    needs = _compute_positional_needs(team_pos_pts)
    pos_ranks = _get_positional_rank(team_pos_pts)

    # Positional profile
    st.subheader("Your Positional Profile")
    _render_positional_profile(my_team_id, rosters, pos_ranks, needs, team_pos_pts)

    # Tabs
    tab_recommended, tab_explore, tab_regression = st.tabs([
        "Recommended Trades", "Explore Teams", "Regression Watch"
    ])

    with tab_recommended:
        num_teams = len(rosters)
        all_proposals = []

        # Collect proposals by type
        proposals_by_type = {}
        if "1-for-1" in trade_types:
            proposals_by_type["1-for-1"] = _find_1_for_1_trades(my_team_id, rosters, needs, num_teams)
        if "2-for-2" in trade_types:
            proposals_by_type["2-for-2"] = _find_2_for_2_trades(my_team_id, rosters, needs, num_teams)
        if "2-for-1" in trade_types:
            proposals_by_type["2-for-1"] = _find_2_for_1_trades(my_team_id, rosters, needs, num_teams)

        # Deduplicate within each type
        for ttype in proposals_by_type:
            seen = set()
            unique = []
            for p in proposals_by_type[ttype]:
                key = (
                    p["opp_id"],
                    frozenset(pl["name"] for pl in p["send"]),
                    frozenset(pl["name"] for pl in p["receive"]),
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            unique.sort(key=lambda x: x["trade_score"], reverse=True)
            proposals_by_type[ttype] = unique

        # Reserve 5 slots per enabled trade type, then fill remaining
        # slots with the best remaining proposals across all types
        slots_per_type = 5
        reserved = []
        leftover = []
        for ttype, proposals in proposals_by_type.items():
            reserved.extend(proposals[:slots_per_type])
            leftover.extend(proposals[slots_per_type:])

        # Sort reserved by score, then append best leftovers up to 15 total
        reserved.sort(key=lambda x: x["trade_score"], reverse=True)
        leftover.sort(key=lambda x: x["trade_score"], reverse=True)
        top_proposals = reserved
        for p in leftover:
            if len(top_proposals) >= 15:
                break
            top_proposals.append(p)

        total_found = sum(len(v) for v in proposals_by_type.values())

        if top_proposals:
            st.caption(f"Showing top {len(top_proposals)} of {total_found} proposals found.")
            for i, proposal in enumerate(top_proposals):
                _render_trade_card(proposal, i)
        else:
            st.info(
                "No beneficial trade proposals found. "
                "Try adjusting the weights or enabling more trade types."
            )

    with tab_explore:
        _render_explore_tab(my_team_id, rosters, needs)

    with tab_regression:
        _render_regression_tab(my_team_id, rosters)
