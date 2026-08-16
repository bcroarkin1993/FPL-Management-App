"""Tests for scripts/draft/season_wrapped.py's per-team auto-archive logic
and archived-season rendering."""

from unittest.mock import patch

import pandas as pd

from scripts.draft.season_wrapped import (
    _compute_mvp_summary,
    _team_history_to_list,
)

_MINIMAL_TEAM_ARCHIVE_DATA = {
    "overview": {
        "team": "Alpha", "num_teams": 2, "final_rank": 1, "total_pts": 1600,
        "avg_pts": 80.0, "wdl": "20W-0D-18L", "seasons_played": 1,
        "team_history": [{"gw": 1, "gw_points": 70, "total_points": 70, "league_position": 1}],
        "best_gw": {"gw": 1, "points": 70}, "worst_gw": {"gw": 1, "points": 70},
        "best_rank": 1, "worst_rank": 1, "total_transfers": 2, "perfect_gws": 1,
        "pts_lost_total": 5, "worst_lineup_gw": None,
    },
    "max_gw": 1,
    "team_players": [{"player": "Saka", "position": "M", "total_points": 80, "team": "ARS"}],
    "mvp_summary": {"player": "Saka", "team": "ARS", "position": "M", "total_points": 80,
                     "goals": 5, "assists": 3, "starts": 10, "bonus": 4, "clean_sheets": 0, "saves": 0},
    "formation_counts": {"4-4-2": 1},
    "bench_data": {
        "per_gw": [{"gw": 1, "actual": 70, "bench_pts": 0, "optimal": 70, "points_lost": 0,
                    "top_bench": "-", "active_chip": None}],
        "total_bench_pts": 0, "total_actual": 70, "total_optimal": 70, "total_points_lost": 0,
    },
    "transfers": [], "most_in": {}, "most_out": {},
    "draft_data": None,
    "superlatives": {},
}


class TestTeamHistoryToList:
    def test_converts_dataframe_to_plain_dicts(self):
        df = pd.DataFrame({
            "Team": ["Alpha", "Alpha"], "Gameweek": [2, 1],
            "GW_Points": [60, 70], "Total_Points": [130, 70], "League_Position": [2, 1],
        })
        result = _team_history_to_list(df)
        assert result == [
            {"gw": 1, "gw_points": 70, "total_points": 70, "league_position": 1},
            {"gw": 2, "gw_points": 60, "total_points": 130, "league_position": 2},
        ]

    def test_empty_or_none_returns_empty_list(self):
        assert _team_history_to_list(pd.DataFrame()) == []
        assert _team_history_to_list(None) == []


class TestComputeMvpSummary:
    def test_enriches_with_bootstrap_stats(self):
        team_players = [{"player": "Saka", "position": "M", "total_points": 80, "team": "ARS"}]
        bootstrap = {"elements": [{"web_name": "Saka", "goals_scored": 5, "assists": 3,
                                    "starts": 10, "bonus": 4, "clean_sheets": 0, "saves": 0}]}
        mvp = _compute_mvp_summary(team_players, bootstrap)
        assert mvp["player"] == "Saka"
        assert mvp["goals"] == 5
        assert mvp["bonus"] == 4

    def test_no_players_returns_none(self):
        assert _compute_mvp_summary([], {}) is None


class TestRenderArchivedSeasonWrapped:
    def test_all_sections_no_raise(self, mock_streamlit):
        from scripts.draft.season_wrapped import _render_archived_season_wrapped
        _render_archived_season_wrapped("2025/26", _MINIMAL_TEAM_ARCHIVE_DATA)  # should not raise


class TestShowSeasonWrappedPageAutoArchive:
    def test_live_page_view_saves_team_archive_snapshot(self, mock_all_utils):
        """Visiting Season Wrapped for a team once real gameweek data exists
        should auto-save a per-team snapshot via save_archived_team_season,
        the same protection League Wrapped's auto-archive gives at the
        league level — so a manager's own Season Wrapped survives next
        season's Draft league-ID rollover too."""
        league_data = {
            "league": {"name": "FPL Friends", "drafts": [{"draft_dt": "2026-08-09T16:00:00Z"}]},
            "league_entries": [
                {"id": 1, "entry_id": 56086, "entry_name": "Alpha"},
                {"id": 2, "entry_id": 56087, "entry_name": "Beta"},
            ],
            "matches": [],
            "standings": [
                {"league_entry": 1, "rank": 1, "matches_won": 2, "matches_drawn": 0, "matches_lost": 0,
                 "points_for": 200, "points_against": 100},
                {"league_entry": 2, "rank": 2, "matches_won": 0, "matches_drawn": 0, "matches_lost": 2,
                 "points_for": 100, "points_against": 200},
            ],
        }
        history_df = pd.DataFrame({
            "Team": ["Alpha", "Beta"], "Gameweek": [1, 1],
            "GW_Points": [70, 30], "Total_Points": [70, 30], "League_Position": [1, 2],
        })
        bench_data = {
            "per_gw": [{"gw": 1, "actual": 70, "bench_pts": 0, "optimal": 70, "points_lost": 0,
                        "top_bench": "-", "active_chip": None}],
            "total_bench_pts": 0, "total_actual": 70, "total_optimal": 70, "total_points_lost": 0,
        }

        with patch("scripts.draft.season_wrapped.get_league_teams", return_value={1: "Alpha", 2: "Beta"}), \
             patch("scripts.draft.season_wrapped.get_team_id_by_name", return_value=1), \
             patch("scripts.draft.season_wrapped.get_current_gameweek", return_value=1), \
             patch("scripts.draft.season_wrapped.build_draft_history_df", return_value=history_df), \
             patch("scripts.draft.season_wrapped.get_draft_league_details", return_value=league_data), \
             patch("scripts.draft.season_wrapped.get_draft_team_players_with_points", return_value={"Alpha": []}), \
             patch("scripts.draft.season_wrapped.get_classic_bootstrap_static", return_value={"elements": []}), \
             patch("scripts.draft.season_wrapped.compute_draft_bench_data", return_value=bench_data), \
             patch("scripts.draft.season_wrapped._compute_formation_stats", return_value={}), \
             patch("scripts.draft.season_wrapped._compute_transfer_stats", return_value=([], {}, {})), \
             patch("scripts.draft.season_wrapped._compute_draft_analysis", return_value=None), \
             patch("scripts.draft.season_wrapped._compute_league_superlatives", return_value={}), \
             patch("scripts.draft.season_wrapped.get_waiver_transactions_up_to_gameweek", return_value=[]), \
             patch("scripts.draft.season_wrapped._load_draft_season_history", return_value=[]), \
             patch("scripts.draft.season_wrapped.list_archived_team_seasons", return_value=[]), \
             patch("scripts.draft.season_wrapped.save_archived_team_season") as mock_save, \
             patch("config.get_draft_league_history_records", return_value=[]):
            from scripts.draft.season_wrapped import show_season_wrapped_page
            show_season_wrapped_page()

        assert mock_save.called
        season_arg, team_arg, data_arg = mock_save.call_args[0]
        assert season_arg == "2026/27"
        assert team_arg == "Alpha"
        assert data_arg["overview"]["team"] == "Alpha"
        assert data_arg["overview"]["team_history"] == [
            {"gw": 1, "gw_points": 70, "total_points": 70, "league_position": 1}
        ]
