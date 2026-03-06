"""
Unit tests for scripts/common/bench_analysis.py.

Tests the optimal lineup algorithm and bench data computation functions.
"""

import pytest
from unittest.mock import patch, MagicMock

from scripts.common.bench_analysis import (
    find_optimal_gw_lineup,
    compute_classic_bench_data,
    compute_draft_bench_data,
    compute_draft_league_bench_data,
    compute_classic_league_bench_data,
    render_league_bench_analysis,
    _summarize_bench_data,
    _normalize_league_bench_results,
)


# =============================================================================
# TestFindOptimalGwLineup
# =============================================================================

class TestFindOptimalGwLineup:

    def _make_player(self, eid, pos, pts):
        return {"element_id": eid, "position": pos, "points": pts}

    def test_empty_input(self):
        assert find_optimal_gw_lineup([]) == []

    def test_picks_highest_scorers(self):
        """Best 11 should be the highest-scoring valid formation."""
        players = [
            # 2 GKs
            self._make_player(1, "GK", 6),
            self._make_player(2, "GK", 2),
            # 5 DEFs
            self._make_player(3, "DEF", 10),
            self._make_player(4, "DEF", 8),
            self._make_player(5, "DEF", 7),
            self._make_player(6, "DEF", 5),
            self._make_player(7, "DEF", 3),
            # 5 MIDs
            self._make_player(8, "MID", 12),
            self._make_player(9, "MID", 9),
            self._make_player(10, "MID", 6),
            self._make_player(11, "MID", 4),
            self._make_player(12, "MID", 1),
            # 3 FWDs
            self._make_player(13, "FWD", 11),
            self._make_player(14, "FWD", 8),
            self._make_player(15, "FWD", 2),
        ]

        result = find_optimal_gw_lineup(players)
        assert len(result) == 11

        total = sum(p["points"] for p in result)
        # Verify the selection is optimal
        selected_ids = {p["element_id"] for p in result}
        # GK: 1 (6), DEF: 3,4,5 (10,8,7), MID: 8,9,10 (12,9,6), FWD: 13,14 (11,8)
        # Remaining 3 flex slots filled by best available: DEF 6 (5), MID 11 (4), FWD 15 (2)
        # Actually: DEF 6 (5), MID 11 (4), ... need to check formation
        # Best would be: 1 GK(6), 3 DEF(10+8+7), 3 MID(12+9+6), 1 FWD(11) = 8 picked (69)
        # Remaining 3: DEF5(5), MID4(4), FWD8(8) -> pick FWD14(8), DEF6(5), MID11(4) = 17
        # Total = 86
        assert total == 86

    def test_respects_formation_rules(self):
        """Must have exactly 1 GK, 3-5 DEF, 3-5 MID, 1-3 FWD."""
        players = [
            self._make_player(1, "GK", 5),
            self._make_player(2, "GK", 3),
            self._make_player(3, "DEF", 10),
            self._make_player(4, "DEF", 9),
            self._make_player(5, "DEF", 8),
            self._make_player(6, "DEF", 7),
            self._make_player(7, "DEF", 6),
            self._make_player(8, "MID", 4),
            self._make_player(9, "MID", 3),
            self._make_player(10, "MID", 2),
            self._make_player(11, "MID", 1),
            self._make_player(12, "MID", 0),
            self._make_player(13, "FWD", 5),
            self._make_player(14, "FWD", 4),
            self._make_player(15, "FWD", 3),
        ]

        result = find_optimal_gw_lineup(players)
        assert len(result) == 11

        pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        for p in result:
            pos_counts[p["position"]] += 1

        assert pos_counts["GK"] == 1
        assert 3 <= pos_counts["DEF"] <= 5
        assert 3 <= pos_counts["MID"] <= 5
        assert 1 <= pos_counts["FWD"] <= 3

    def test_handles_ties(self):
        """When players have equal points, selection is deterministic by element_id."""
        players = [
            self._make_player(1, "GK", 5),
            self._make_player(2, "GK", 5),
            self._make_player(3, "DEF", 5),
            self._make_player(4, "DEF", 5),
            self._make_player(5, "DEF", 5),
            self._make_player(6, "DEF", 5),
            self._make_player(7, "DEF", 5),
            self._make_player(8, "MID", 5),
            self._make_player(9, "MID", 5),
            self._make_player(10, "MID", 5),
            self._make_player(11, "MID", 5),
            self._make_player(12, "MID", 5),
            self._make_player(13, "FWD", 5),
            self._make_player(14, "FWD", 5),
            self._make_player(15, "FWD", 5),
        ]

        result1 = find_optimal_gw_lineup(players)
        result2 = find_optimal_gw_lineup(players)

        ids1 = [p["element_id"] for p in result1]
        ids2 = [p["element_id"] for p in result2]
        assert ids1 == ids2  # Deterministic

    def test_all_fwds_high_scoring(self):
        """When FWDs outscore everyone, should max out at 3 FWDs."""
        players = [
            self._make_player(1, "GK", 2),
            self._make_player(2, "GK", 1),
            self._make_player(3, "DEF", 3),
            self._make_player(4, "DEF", 2),
            self._make_player(5, "DEF", 1),
            self._make_player(6, "DEF", 0),
            self._make_player(7, "DEF", 0),
            self._make_player(8, "MID", 3),
            self._make_player(9, "MID", 2),
            self._make_player(10, "MID", 1),
            self._make_player(11, "MID", 0),
            self._make_player(12, "MID", 0),
            self._make_player(13, "FWD", 15),
            self._make_player(14, "FWD", 14),
            self._make_player(15, "FWD", 13),
        ]

        result = find_optimal_gw_lineup(players)
        fwd_count = sum(1 for p in result if p["position"] == "FWD")
        assert fwd_count == 3  # Max FWDs


# =============================================================================
# TestComputeClassicBenchData
# =============================================================================

class TestComputeClassicBenchData:

    @patch("scripts.common.bench_analysis._get_classic_gw_live_points")
    @patch("scripts.common.bench_analysis.get_classic_team_picks")
    @patch("scripts.common.bench_analysis.get_classic_bootstrap_static")
    def test_basic_bench_points(self, mock_bootstrap, mock_picks, mock_live):
        """Verify bench vs starter split and bench points calculation."""
        mock_bootstrap.return_value = {
            "elements": [
                {"id": i, "web_name": f"P{i}", "element_type": et}
                for i, et in [
                    (1, 1),   # GK
                    (2, 1),   # GK
                    (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),  # DEF
                    (8, 3), (9, 3), (10, 3), (11, 3), (12, 3),  # MID
                    (13, 4), (14, 4), (15, 4),  # FWD
                ]
            ],
        }

        # Starters: positions 1-11, bench: 12-15
        picks = [{"element": i, "position": i, "multiplier": 2 if i == 1 else 1,
                  "is_captain": i == 1, "is_vice_captain": False}
                 for i in range(1, 16)]

        mock_picks.return_value = {"picks": picks, "active_chip": None}

        # All players score 5 points
        mock_live.return_value = {i: 5 for i in range(1, 16)}

        result = compute_classic_bench_data(1, 1)

        assert result is not None
        assert len(result["per_gw"]) == 1
        gw = result["per_gw"][0]
        assert gw["bench_pts"] == 20  # 4 bench players * 5 pts
        # Actual: 10 starters * 5 + captain * 5 (double) = 55 + 5 = 60
        assert gw["actual"] == 60

    @patch("scripts.common.bench_analysis._get_classic_gw_live_points")
    @patch("scripts.common.bench_analysis.get_classic_team_picks")
    @patch("scripts.common.bench_analysis.get_classic_bootstrap_static")
    def test_captain_multiplier(self, mock_bootstrap, mock_picks, mock_live):
        """Captain gets 2x in actual score."""
        mock_bootstrap.return_value = {
            "elements": [
                {"id": i, "web_name": f"P{i}", "element_type": et}
                for i, et in [
                    (1, 1), (2, 1),
                    (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                    (8, 3), (9, 3), (10, 3), (11, 3), (12, 3),
                    (13, 4), (14, 4), (15, 4),
                ]
            ],
        }

        picks = [{"element": i, "position": i,
                  "multiplier": 2 if i == 8 else 1,
                  "is_captain": i == 8, "is_vice_captain": False}
                 for i in range(1, 16)]

        mock_picks.return_value = {"picks": picks, "active_chip": None}

        # Player 8 (captain) scores 10, everyone else scores 3
        live = {i: 3 for i in range(1, 16)}
        live[8] = 10
        mock_live.return_value = live

        result = compute_classic_bench_data(1, 1)
        gw = result["per_gw"][0]

        # Actual: 10 non-captain starters * 3 + captain 10 * 2 = 30 + 20 = 50
        assert gw["actual"] == 50

    @patch("scripts.common.bench_analysis._get_classic_gw_live_points")
    @patch("scripts.common.bench_analysis.get_classic_team_picks")
    @patch("scripts.common.bench_analysis.get_classic_bootstrap_static")
    def test_captain_not_reoptimized(self, mock_bootstrap, mock_picks, mock_live):
        """Optimal score keeps actual captain — doesn't reassign to best scorer."""
        mock_bootstrap.return_value = {
            "elements": [
                {"id": i, "web_name": f"P{i}", "element_type": et}
                for i, et in [
                    (1, 1), (2, 1),
                    (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                    (8, 3), (9, 3), (10, 3), (11, 3), (12, 3),
                    (13, 4), (14, 4), (15, 4),
                ]
            ],
        }

        # Captain is player 8 (MID), but bench player 14 (FWD) scores highest
        picks = [{"element": i, "position": i,
                  "multiplier": 2 if i == 8 else 1,
                  "is_captain": i == 8, "is_vice_captain": False}
                 for i in range(1, 16)]

        mock_picks.return_value = {"picks": picks, "active_chip": None}

        live = {i: 3 for i in range(1, 16)}
        live[8] = 5   # captain scores 5
        live[14] = 15  # bench FWD scores 15
        mock_live.return_value = live

        result = compute_classic_bench_data(1, 1)
        gw = result["per_gw"][0]

        # Actual: 10 non-captain starters * 3 + captain 5 * 2 = 30 + 10 = 40
        assert gw["actual"] == 40
        # Optimal should keep captain (player 8) with 2x, not reassign to player 14
        # Optimal 11: swap weakest starter for bench player 14 (15 pts)
        # Captain 8 still gets 2x = 10, player 14 gets 1x = 15
        # Points lost should reflect only the bench swap, not captain change
        assert gw["points_lost"] == gw["optimal"] - gw["actual"]
        # If captain were re-optimized to player 14: optimal = best11_sum + 15*(2-1) = much higher
        # With captain kept: optimal = best11_sum + 5*(2-1) = more modest
        # Key assertion: points lost should be <= bench player's score (not inflated by captain)
        assert gw["points_lost"] <= 15  # can't lose more than the best bench player contributes

    @patch("scripts.common.bench_analysis._get_classic_gw_live_points")
    @patch("scripts.common.bench_analysis.get_classic_team_picks")
    @patch("scripts.common.bench_analysis.get_classic_bootstrap_static")
    def test_bench_boost_excluded_from_points_lost(self, mock_bootstrap, mock_picks, mock_live):
        """Bench Boost GWs should not count toward total points lost."""
        mock_bootstrap.return_value = {
            "elements": [
                {"id": i, "web_name": f"P{i}", "element_type": et}
                for i, et in [
                    (1, 1), (2, 1),
                    (3, 2), (4, 2), (5, 2), (6, 2), (7, 2),
                    (8, 3), (9, 3), (10, 3), (11, 3), (12, 3),
                    (13, 4), (14, 4), (15, 4),
                ]
            ],
        }

        # GW 1: normal, GW 2: bench boost
        def mock_picks_fn(team_id, gw):
            picks = [{"element": i, "position": i,
                      "multiplier": 2 if i == 1 else 1,
                      "is_captain": i == 1, "is_vice_captain": False}
                     for i in range(1, 16)]
            chip = "bboost" if gw == 2 else None
            return {"picks": picks, "active_chip": chip}

        mock_picks.side_effect = mock_picks_fn
        mock_live.return_value = {i: 5 for i in range(1, 16)}

        result = compute_classic_bench_data(1, 2)

        # BB GW should have bench_pts = 0 (all playing)
        bb_gw = next(g for g in result["per_gw"] if g["active_chip"] == "bboost")
        assert bb_gw["bench_pts"] == 0

        # Total points lost should only count GW 1
        normal_gw = next(g for g in result["per_gw"] if g["active_chip"] is None)
        assert result["total_points_lost"] == normal_gw["points_lost"]


# =============================================================================
# TestComputeDraftBenchData
# =============================================================================

class TestComputeDraftBenchData:

    @patch("scripts.common.bench_analysis.requests.get")
    @patch("scripts.common.bench_analysis._get_draft_gw_live_points")
    @patch("scripts.common.bench_analysis.get_fpl_player_mapping")
    def test_no_captain_multiplier(self, mock_player_map, mock_live, mock_get):
        """Draft has no captaincy — all players score 1x."""
        mock_player_map.return_value = {
            i: {"Player": f"Player {i}", "Web_Name": f"P{i}", "Position": pos}
            for i, pos in [
                (1, "G"), (2, "G"),
                (3, "D"), (4, "D"), (5, "D"), (6, "D"), (7, "D"),
                (8, "M"), (9, "M"), (10, "M"), (11, "M"), (12, "M"),
                (13, "F"), (14, "F"), (15, "F"),
            ]
        }

        picks = [{"element": i, "position": i} for i in range(1, 16)]
        mock_response = MagicMock()
        mock_response.json.return_value = {"picks": picks}
        mock_get.return_value = mock_response

        # All players score 5 pts
        mock_live.return_value = {i: 5 for i in range(1, 16)}

        result = compute_draft_bench_data(1, 1)

        assert result is not None
        gw = result["per_gw"][0]
        # No captain: 11 starters * 5 = 55
        assert gw["actual"] == 55
        # Bench: 4 * 5 = 20
        assert gw["bench_pts"] == 20
        # Optimal is also 55 (all equal points, no captain bonus)
        assert gw["optimal"] == 55
        assert gw["points_lost"] == 0


# =============================================================================
# TestSummarizeBenchData
# =============================================================================

class TestSummarizeBenchData:

    def test_basic_summary(self):
        bench_data = {
            "per_gw": [
                {"gw": 1, "actual": 50, "bench_pts": 15, "optimal": 55, "points_lost": 5, "active_chip": None},
                {"gw": 2, "actual": 60, "bench_pts": 10, "optimal": 65, "points_lost": 5, "active_chip": None},
            ],
            "total_bench_pts": 25,
            "total_actual": 110,
            "total_optimal": 120,
            "total_points_lost": 10,
        }

        result = _summarize_bench_data("Test Team", bench_data, 2)

        assert result is not None
        assert result["Team"] == "Test Team"
        assert result["Total Bench Pts"] == 25
        assert result["Total Pts Lost"] == 10
        assert result["Avg Bench/GW"] == 12.5
        assert result["Avg Lost/GW"] == 5.0
        # Selection %: 110/120 * 100 = 91.7
        assert result["Selection %"] == 91.7
        # Bench Strength and Bench Mgmt Score default to 0 (populated after normalization)
        assert "Bench Strength" in result
        assert "Bench Mgmt Score" in result
        assert "GW" in result["Worst GW"]

    def test_none_bench_data(self):
        assert _summarize_bench_data("Team", None, 5) is None

    def test_empty_per_gw(self):
        assert _summarize_bench_data("Team", {"per_gw": []}, 5) is None

    def test_bb_excluded_from_gw_count(self):
        bench_data = {
            "per_gw": [
                {"gw": 1, "actual": 50, "bench_pts": 15, "optimal": 55, "points_lost": 5, "active_chip": None},
                {"gw": 2, "actual": 80, "bench_pts": 0, "optimal": 80, "points_lost": 0, "active_chip": "bboost"},
            ],
            "total_bench_pts": 15,
            "total_actual": 130,
            "total_optimal": 135,
            "total_points_lost": 5,
        }

        result = _summarize_bench_data("Test", bench_data, 2)
        # Only 1 eligible GW (BB excluded)
        assert result["Avg Bench/GW"] == 15.0
        assert result["Avg Lost/GW"] == 5.0


# =============================================================================
# TestComputeDraftLeagueBenchData
# =============================================================================

class TestComputeDraftLeagueBenchData:

    @patch("scripts.common.bench_analysis.compute_draft_bench_data")
    @patch("scripts.common.bench_analysis.get_draft_league_details")
    def test_basic_league_data(self, mock_league, mock_bench):
        mock_league.return_value = {
            "league_entries": [
                {"entry_id": 100, "entry_name": "Team A"},
                {"entry_id": 200, "entry_name": "Team B"},
            ]
        }

        mock_bench.side_effect = [
            {
                "per_gw": [{"gw": 1, "actual": 50, "bench_pts": 10, "optimal": 55, "points_lost": 5, "active_chip": None}],
                "total_bench_pts": 10,
                "total_actual": 50,
                "total_optimal": 55,
                "total_points_lost": 5,
            },
            {
                "per_gw": [{"gw": 1, "actual": 45, "bench_pts": 20, "optimal": 55, "points_lost": 10, "active_chip": None}],
                "total_bench_pts": 20,
                "total_actual": 45,
                "total_optimal": 55,
                "total_points_lost": 10,
            },
        ]

        result = compute_draft_league_bench_data(1, 1)

        assert len(result) == 2
        # Both teams have Bench Strength and Bench Mgmt Score populated
        for row in result:
            assert "Bench Strength" in row
            assert "Bench Mgmt Score" in row
            assert "Selection %" in row
            assert row["Bench Mgmt Score"] > 0
        # Sorted by Bench Mgmt Score descending
        assert result[0]["Bench Mgmt Score"] >= result[1]["Bench Mgmt Score"]
        # Team A has better Selection % (90.9 vs 81.8) and lower bench (10 vs 20)
        # Team A: Selection % higher → sel_norm=100, Bench Strength=0 → score=60
        # Team B: Selection % lower → sel_norm=0, Bench Strength=100 → score=40
        assert result[0]["Team"] == "Team A"
        assert result[1]["Team"] == "Team B"

    @patch("scripts.common.bench_analysis.get_draft_league_details")
    def test_no_league_data(self, mock_league):
        mock_league.return_value = None
        result = compute_draft_league_bench_data(1, 1)
        assert result == []


# =============================================================================
# TestComputeClassicLeagueBenchData
# =============================================================================

class TestComputeClassicLeagueBenchData:

    @patch("scripts.common.bench_analysis.compute_classic_bench_data")
    def test_basic_league_data(self, mock_bench):
        mock_bench.side_effect = [
            {
                "per_gw": [{"gw": 1, "actual": 60, "bench_pts": 8, "optimal": 62, "points_lost": 2, "active_chip": None}],
                "total_bench_pts": 8,
                "total_actual": 60,
                "total_optimal": 62,
                "total_points_lost": 2,
            },
            {
                "per_gw": [{"gw": 1, "actual": 55, "bench_pts": 15, "optimal": 65, "points_lost": 10, "active_chip": None}],
                "total_bench_pts": 15,
                "total_actual": 55,
                "total_optimal": 65,
                "total_points_lost": 10,
            },
        ]

        import json
        team_names = json.dumps({"1": "Manager X", "2": "Manager Y"})
        result = compute_classic_league_bench_data((1, 2), team_names, 1)

        assert len(result) == 2
        # Both teams have normalized scores populated
        for row in result:
            assert "Bench Strength" in row
            assert "Bench Mgmt Score" in row
            assert "Selection %" in row
            assert row["Bench Mgmt Score"] > 0
        # Sorted by Bench Mgmt Score descending
        assert result[0]["Bench Mgmt Score"] >= result[1]["Bench Mgmt Score"]
        # Manager X: higher Selection % (96.8 vs 84.6), lower bench (8 vs 15)
        # Manager X: sel_norm=100, Bench Strength=0 → score=60
        # Manager Y: sel_norm=0, Bench Strength=100 → score=40
        assert result[0]["Team"] == "Manager X"
        assert result[1]["Team"] == "Manager Y"


# =============================================================================
# TestRenderLeagueBenchAnalysis
# =============================================================================

class TestRenderLeagueBenchAnalysis:

    @patch("scripts.common.bench_analysis.st")
    @patch("scripts.common.bench_analysis.render_styled_table")
    def test_renders_without_error(self, mock_table, mock_st):
        """Verify render function doesn't raise with valid input."""
        # st.columns(3) must return 3 context managers
        mock_col = MagicMock()
        mock_col.__enter__ = MagicMock(return_value=None)
        mock_col.__exit__ = MagicMock(return_value=False)
        mock_st.columns.return_value = [mock_col, mock_col, mock_col]

        league_data = [
            {
                "Team": "Team A",
                "Total Bench Pts": 100,
                "Avg Bench/GW": 10.0,
                "Total Pts Lost": 30,
                "Avg Lost/GW": 3.0,
                "Selection %": 95.0,
                "Bench Strength": 0.0,
                "Bench Mgmt Score": 60.0,
                "Worst GW": "GW5: 8 pts",
            },
            {
                "Team": "Team B",
                "Total Bench Pts": 150,
                "Avg Bench/GW": 15.0,
                "Total Pts Lost": 50,
                "Avg Lost/GW": 5.0,
                "Selection %": 90.0,
                "Bench Strength": 100.0,
                "Bench Mgmt Score": 40.0,
                "Worst GW": "GW3: 12 pts",
            },
        ]

        # Should not raise
        render_league_bench_analysis(league_data, is_classic=True)

        # Verify table was rendered
        mock_table.assert_called_once()

    @patch("scripts.common.bench_analysis.st")
    def test_empty_data(self, mock_st):
        render_league_bench_analysis([], is_classic=True)
        mock_st.info.assert_called_once()
