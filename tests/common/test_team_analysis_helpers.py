"""Tests for scripts/common/team_analysis_helpers.py."""

import pytest
from unittest.mock import patch

from scripts.common.team_analysis_helpers import (
    get_best_clubs,
    get_season_best_11,
    get_team_mvp,
)


@pytest.fixture
def sample_player_data():
    """Minimal player data covering all positions."""
    return [
        {"player": "Ramsdale", "position": "GK", "total_points": 80, "team": "Arsenal"},
        {"player": "Saliba", "position": "DEF", "total_points": 120, "team": "Arsenal"},
        {"player": "Gabriel", "position": "DEF", "total_points": 110, "team": "Arsenal"},
        {"player": "Van Dijk", "position": "DEF", "total_points": 130, "team": "Liverpool"},
        {"player": "Robertson", "position": "DEF", "total_points": 90, "team": "Liverpool"},
        {"player": "Salah", "position": "MID", "total_points": 180, "team": "Liverpool"},
        {"player": "De Bruyne", "position": "MID", "total_points": 160, "team": "Man City"},
        {"player": "Foden", "position": "MID", "total_points": 140, "team": "Man City"},
        {"player": "Saka", "position": "MID", "total_points": 150, "team": "Arsenal"},
        {"player": "Haaland", "position": "FWD", "total_points": 200, "team": "Man City"},
        {"player": "Watkins", "position": "FWD", "total_points": 100, "team": "Aston Villa"},
    ]


class TestGetBestClubs:
    def test_top_3_clubs(self, sample_player_data):
        result = get_best_clubs(sample_player_data, top_n=3)
        assert len(result) == 3
        assert "Rank" in result.columns
        assert "Club" in result.columns
        assert "Points" in result.columns
        assert "Players" in result.columns

    def test_ordering_by_points(self, sample_player_data):
        result = get_best_clubs(sample_player_data, top_n=3)
        # Man City: 160+140+200 = 500, Arsenal: 80+120+110+150 = 460, Liverpool: 130+90+180 = 400
        assert result.iloc[0]["Club"] == "Man City"
        assert result.iloc[0]["Points"] == 500

    def test_empty_input(self):
        result = get_best_clubs([], top_n=3)
        assert result.empty

    def test_top_n_limits(self, sample_player_data):
        result = get_best_clubs(sample_player_data, top_n=1)
        assert len(result) == 1


class TestGetSeasonBest11:
    def test_valid_formation(self, sample_player_data):
        result = get_season_best_11(sample_player_data)
        assert result["formation"] != "N/A"
        assert len(result["players"]) == 11
        assert result["total_points"] > 0

    def test_formation_format(self, sample_player_data):
        result = get_season_best_11(sample_player_data)
        # Formation should be like "X-Y-Z"
        parts = result["formation"].split("-")
        assert len(parts) == 3
        nums = [int(p) for p in parts]
        assert sum(nums) == 10  # 11 - 1 GK = 10

    def test_empty_input(self):
        result = get_season_best_11([])
        assert result["formation"] == "N/A"
        assert result["players"] == []
        assert result["total_points"] == 0

    def test_optimal_picks_highest_scorers(self, sample_player_data):
        result = get_season_best_11(sample_player_data)
        selected_names = [p["player"] for p in result["players"]]
        # Haaland (200) and Salah (180) should definitely be selected
        assert "Haaland" in selected_names
        assert "Salah" in selected_names

    def test_insufficient_position(self):
        """With no GK available, should fall back to best available."""
        data = [
            {"player": f"Player {i}", "position": "MID", "total_points": 50 + i, "team": "Team"}
            for i in range(15)
        ]
        result = get_season_best_11(data)
        assert result["formation"] == "Best Available"


class TestGetTeamMvp:
    def test_returns_highest_scorer(self, sample_player_data):
        mvp = get_team_mvp(sample_player_data)
        assert mvp is not None
        assert mvp["player"] == "Haaland"
        assert mvp["total_points"] == 200

    def test_enriches_with_bootstrap(self, sample_player_data, mock_bootstrap_data):
        mvp = get_team_mvp(sample_player_data, bootstrap_data=mock_bootstrap_data)
        assert mvp is not None
        assert mvp["player"] == "Haaland"
        assert mvp["goals"] == 25
        assert mvp["assists"] == 5
        assert mvp["starts"] == 25

    def test_empty_input(self):
        assert get_team_mvp([]) is None

    def test_no_bootstrap(self, sample_player_data):
        mvp = get_team_mvp(sample_player_data)
        assert mvp is not None
        assert mvp["goals"] == 0  # No bootstrap data to enrich with
