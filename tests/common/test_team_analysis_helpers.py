"""Tests for scripts/common/team_analysis_helpers.py."""

import pytest
from unittest.mock import patch

from scripts.common.team_analysis_helpers import (
    get_best_clubs,
    get_season_best_11,
    get_team_mvp,
    build_classic_season_history_df,
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


class TestBuildClassicSeasonHistoryDf:
    """build_classic_season_history_df() — joins live FPL season stats
    (Season/Points/Rank) with manually-entered % Finish and league placements."""

    @pytest.fixture
    def past_seasons(self):
        return [
            {"season_name": "2025/26", "total_points": 2187, "rank": 1037838},
            {"season_name": "2024/25", "total_points": 2125, "rank": 3932974},
        ]

    def test_points_and_rank_always_from_live_data(self, past_seasons):
        df = build_classic_season_history_df(past_seasons, {}, [])
        assert list(df["Season"]) == ["2025/26", "2024/25"]
        assert list(df["Points"]) == ["2,187", "2,125"]
        assert list(df["Rank"]) == ["1,037,838", "3,932,974"]

    def test_missing_pct_finish_and_placements_show_em_dash(self, past_seasons):
        df = build_classic_season_history_df(past_seasons, {}, [])
        assert (df["% Finish"] == "—").all()
        assert (df["League Placements"] == "—").all()

    def test_pct_finish_applied_per_season(self, past_seasons):
        season_notes = {"2025/26": {"pct_finish": 8.0}}
        df = build_classic_season_history_df(past_seasons, season_notes, [])
        row = df[df["Season"] == "2025/26"].iloc[0]
        assert row["% Finish"] == "8%"
        other = df[df["Season"] == "2024/25"].iloc[0]
        assert other["% Finish"] == "—"

    def test_multiple_league_placements_joined(self, past_seasons):
        league_history_records = [
            {"season": "2025/26", "league_id": 1161877, "league_name": "Super League DMV Starboys",
             "manual_stats": {"rank": 4, "total_points": None}},
            {"season": "2025/26", "league_id": 1555691, "league_name": "FAFO FPL",
             "manual_stats": {"rank": 1, "total_points": None}},
        ]
        df = build_classic_season_history_df(past_seasons, {}, league_history_records)
        row = df[df["Season"] == "2025/26"].iloc[0]
        assert "FAFO FPL (1st)" in row["League Placements"]
        assert "Super League DMV Starboys (4th)" in row["League Placements"]

    def test_placement_without_rank_is_skipped(self, past_seasons):
        """A league_history record with no manual rank (e.g. an auto-archived
        stale-ID entry with no stats yet) shouldn't produce a placement line."""
        league_history_records = [
            {"season": "2025/26", "league_id": 1161877, "league_name": "Starboys", "manual_stats": None},
        ]
        df = build_classic_season_history_df(past_seasons, {}, league_history_records)
        row = df[df["Season"] == "2025/26"].iloc[0]
        assert row["League Placements"] == "—"

    def test_live_rank_percentage_used_when_present(self):
        """FPL's entry-history endpoint actually returns rank_percentage per
        season — this must be used directly rather than requiring manual
        entry, which is only a fallback for the rare season missing it."""
        past_seasons = [{"season_name": "2025/26", "total_points": 2187, "rank": 1037838, "rank_percentage": "8"}]
        df = build_classic_season_history_df(past_seasons, {}, [])
        assert df.iloc[0]["% Finish"] == "8%"

    def test_live_rank_percentage_takes_precedence_over_manual(self):
        past_seasons = [{"season_name": "2025/26", "total_points": 2187, "rank": 1037838, "rank_percentage": "8"}]
        season_notes = {"2025/26": {"pct_finish": 99.0}}
        df = build_classic_season_history_df(past_seasons, season_notes, [])
        assert df.iloc[0]["% Finish"] == "8%"

    def test_manual_fallback_used_when_live_field_absent(self, past_seasons):
        """past_seasons fixture has no rank_percentage key at all — the
        season_notes fallback must still work (e.g. for a data source that
        predates this field, or an edge case in FPL's response)."""
        season_notes = {"2025/26": {"pct_finish": 8.0}}
        df = build_classic_season_history_df(past_seasons, season_notes, [])
        assert df[df["Season"] == "2025/26"].iloc[0]["% Finish"] == "8%"
