"""Unit tests for Trade Analyzer logic.

Tests core functions with realistic data to catch issues like
ZeroDivisionError that smoke tests (which mock everything) miss.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _make_stats_df():
    """Minimal FPL stats DataFrame with players including one with 0 minutes."""
    return pd.DataFrame([
        {
            "id": 1, "player": "Aaron Ramsdale", "first_name": "Aaron", "second_name": "Ramsdale",
            "web_name": "Ramsdale", "team": 1, "team_id": 1, "team_name": "Arsenal",
            "team_name_abbrv": "ARS", "element_type": 1, "position_id": 1,
            "position_name": "Goalkeeper", "position_abbrv": "GKP",
            "total_points": 80, "goals_scored": 0, "assists": 1, "minutes": 1800,
            "starts": 20, "form": "4.0", "points_per_game": "3.5",
            "expected_goals": "0.1", "expected_assists": "0.5",
            "expected_goal_involvements": "0.6", "expected_goals_conceded": "25.0",
            "goals_conceded": 28, "saves": 60, "clean_sheets": 6,
            "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
            "bonus": 8, "bps": 400, "creativity": "10.0", "influence": "200.0",
            "threat": "5.0", "ict_index": "20.0", "red_cards": 0, "yellow_cards": 1,
            "now_cost": 50, "selected_by_percent": "5.0",
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
            "status": "a", "news": "", "news_added": None,
            "corners_and_indirect_freekicks_order": None,
            "corners_and_indirect_freekicks_text": "",
            "direct_freekicks_order": None, "direct_freekicks_text": "",
            "penalties_order": None, "penalties_text": "",
            "actual_goal_involvements": 1,
        },
        {
            "id": 99, "player": "Bench Warmer", "first_name": "Bench", "second_name": "Warmer",
            "web_name": "Warmer", "team": 1, "team_id": 1, "team_name": "Arsenal",
            "team_name_abbrv": "ARS", "element_type": 2, "position_id": 2,
            "position_name": "Defender", "position_abbrv": "DEF",
            "total_points": 0, "goals_scored": 0, "assists": 0, "minutes": 0,
            "starts": 0, "form": "0.0", "points_per_game": "0.0",
            "expected_goals": "0.0", "expected_assists": "0.0",
            "expected_goal_involvements": "0.0", "expected_goals_conceded": "0.0",
            "goals_conceded": 0, "saves": 0, "clean_sheets": 0,
            "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
            "bonus": 0, "bps": 0, "creativity": "0.0", "influence": "0.0",
            "threat": "0.0", "ict_index": "0.0", "red_cards": 0, "yellow_cards": 0,
            "now_cost": 40, "selected_by_percent": "0.1",
            "chance_of_playing_this_round": None, "chance_of_playing_next_round": None,
            "status": "a", "news": "", "news_added": None,
            "corners_and_indirect_freekicks_order": None,
            "corners_and_indirect_freekicks_text": "",
            "direct_freekicks_order": None, "direct_freekicks_text": "",
            "penalties_order": None, "penalties_text": "",
            "actual_goal_involvements": 0,
        },
    ])


def _make_rosters():
    """Minimal rosters dict with two teams."""
    return {
        1: {
            "team_name": "Team Alpha",
            "players": [
                {"name": "Aaron Ramsdale", "position": "GK", "pos_short": "G",
                 "team": "ARS", "player_id": 1, "total_points": 0},
                {"name": "Bench Warmer", "position": "DEF", "pos_short": "D",
                 "team": "ARS", "player_id": 99, "total_points": 0},
            ],
        },
        2: {
            "team_name": "Team Beta",
            "players": [
                {"name": "Some Keeper", "position": "GK", "pos_short": "G",
                 "team": "LIV", "player_id": None, "total_points": 0},
            ],
        },
    }


class TestEnrichWithStats:
    """Tests for _enrich_with_stats which calls prepare_advanced_stats_df."""

    def test_no_division_by_zero_with_zero_minutes_player(self):
        """Regression test: player with 0 minutes must not cause ZeroDivisionError."""
        from scripts.draft.trade_analyzer import _enrich_with_stats

        rosters = _make_rosters()
        stats_df = _make_stats_df()
        weights = {"w_season": 0.3, "w_regr": 0.25, "w_form": 0.2, "w_fdr": 0.15, "w_minutes": 0.1}

        # Patch FDR lookup to avoid network calls
        with patch("scripts.draft.trade_analyzer._avg_fdr_for_team", return_value=3.0), \
             patch("scripts.fpl.player_statistics.get_fixture_difficulty_grid",
                   return_value=(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))):
            # This should NOT raise ZeroDivisionError
            result = _enrich_with_stats(rosters, stats_df, current_gw=25, fdr_weeks=3, weights=weights)

        # Verify enrichment happened
        alpha = result[1]["players"]
        ramsdale = next(p for p in alpha if p["name"] == "Aaron Ramsdale")
        assert ramsdale["total_points"] == 80
        assert ramsdale["trade_value"] > 0

        warmer = next(p for p in alpha if p["name"] == "Bench Warmer")
        assert warmer["minutes"] == 0
        assert "trade_value" in warmer


class TestComputePositionalNeeds:
    def test_basic_needs(self):
        """Teams with different point totals get different need scores."""
        from scripts.draft.trade_analyzer import _compute_positional_needs

        team_pos_pts = {
            1: {"GK": 0, "DEF": 50, "MID": 200, "FWD": 0},
            2: {"GK": 0, "DEF": 200, "MID": 50, "FWD": 0},
        }
        needs = _compute_positional_needs(team_pos_pts)
        # Team 1 is strong at MID (low need) and weak at DEF (high need)
        assert needs[1]["MID"] < needs[1]["DEF"]
        # Team 2 is the opposite
        assert needs[2]["DEF"] < needs[2]["MID"]


class TestComputeTradeValues:
    def test_all_zero_weights(self):
        """All-zero weights should not crash (denom protected by 1e-9)."""
        from scripts.draft.trade_analyzer import _compute_trade_values

        rosters = {
            1: {"team_name": "T", "players": [
                {"total_points": 100, "gi_minus_xgi": -1.0, "form": 5.0,
                 "avg_fdr": 2.5, "start_pct": 80.0, "availability": 1.0},
            ]},
        }
        weights = {"w_season": 0, "w_regr": 0, "w_form": 0, "w_fdr": 0, "w_minutes": 0}
        _compute_trade_values(rosters, weights)
        assert "trade_value" in rosters[1]["players"][0]


def _make_full_rosters():
    """Two legal Draft squads (2 GK / 5 DEF / 5 MID / 3 FWD each) with trade values.

    Team Beta is deliberately stronger everywhere, so every finder has something
    to propose — otherwise a "no illegal trades" assertion passes vacuously.
    """
    shape = [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]
    rosters = {}
    for team_id, (name, base) in enumerate([("Team Alpha", 0.2), ("Team Beta", 0.6)], start=1):
        players = []
        for pos, count in shape:
            for i in range(count):
                players.append({
                    "name": f"{name} {pos}{i}",
                    "position": pos,
                    "pos_short": pos[0],
                    "team": "ARS",
                    "player_id": team_id * 100 + len(players),
                    "total_points": 50,
                    "trade_value": base + 0.05 * i,
                })
        rosters[team_id] = {"team_name": name, "players": players}
    return rosters


def _needs_for(rosters):
    return {tid: {"GK": 0.5, "DEF": 0.5, "MID": 0.5, "FWD": 0.5} for tid in rosters}


def _is_balanced(proposal):
    """Both sides move the same number of players in the same positions."""
    from collections import Counter
    send = Counter(p["position"] for p in proposal["send"])
    recv = Counter(p["position"] for p in proposal["receive"])
    return send == recv


class TestTradeLegality:
    """FPL requires identical position composition and count on both sides.

    The bug this guards: _find_1_for_1_trades() used to search *cross-position*
    swaps (send a MID, receive a FWD) and a _find_2_for_1_trades() finder
    proposed unequal counts. Both rendered trades that cannot be submitted.
    """

    def test_rejects_unequal_counts(self):
        from scripts.draft.trade_analyzer import _is_legal_trade

        send = [{"position": "MID"}]
        recv = [{"position": "MID"}, {"position": "FWD"}]
        assert not _is_legal_trade(send, recv)

    def test_rejects_cross_position_one_for_one(self):
        from scripts.draft.trade_analyzer import _is_legal_trade

        assert not _is_legal_trade([{"position": "MID"}], [{"position": "FWD"}])

    def test_rejects_same_count_different_composition(self):
        """FPL's own example: 1 MID + 2 FWD may not become 2 MID + 1 FWD."""
        from scripts.draft.trade_analyzer import _is_legal_trade

        send = [{"position": "MID"}, {"position": "FWD"}, {"position": "FWD"}]
        recv = [{"position": "MID"}, {"position": "MID"}, {"position": "FWD"}]
        assert not _is_legal_trade(send, recv)

    def test_accepts_matching_composition(self):
        from scripts.draft.trade_analyzer import _is_legal_trade

        assert _is_legal_trade([{"position": "MID"}], [{"position": "MID"}])
        send = [{"position": "MID"}, {"position": "FWD"}]
        recv = [{"position": "FWD"}, {"position": "MID"}]  # order must not matter
        assert _is_legal_trade(send, recv)

    def test_score_proposal_refuses_an_illegal_shape(self):
        """The guard sits in _score_proposal so no finder can bypass it."""
        from scripts.draft.trade_analyzer import _score_proposal

        rosters = _make_full_rosters()
        needs = _needs_for(rosters)
        send = [p for p in rosters[1]["players"] if p["position"] == "MID"][:1]
        recv = [p for p in rosters[2]["players"] if p["position"] == "FWD"][:1]

        assert _score_proposal(1, 2, send, recv, rosters, needs, 2) is None

    def test_every_discovered_trade_is_legal(self):
        """The real assertion: whatever the finders produce must be proposable."""
        from scripts.draft.trade_analyzer import (
            _find_1_for_1_trades,
            _find_2_for_2_trades,
        )

        rosters = _make_full_rosters()
        needs = _needs_for(rosters)

        one_for_one = _find_1_for_1_trades(1, rosters, needs, num_teams=2)
        two_for_two = _find_2_for_2_trades(1, rosters, needs, num_teams=2)

        assert one_for_one, "no 1-for-1 proposals generated — assertion would be vacuous"
        for proposal in one_for_one + two_for_two:
            assert _is_balanced(proposal), proposal["trade_type"]

    def test_two_for_one_discovery_is_gone(self):
        """2-for-1 is an unequal shape and cannot be proposed in FPL at all."""
        import scripts.draft.trade_analyzer as ta

        assert not hasattr(ta, "_find_2_for_1_trades")
