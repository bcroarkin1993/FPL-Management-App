"""Tests for scripts/draft/league_wrapped.py's preseason-guard logic."""

from unittest.mock import patch

import pandas as pd

from scripts.draft.league_wrapped import _any_gameweek_played, _season_label_from_league_data


class TestAnyGameweekPlayed:
    def test_empty_history_is_false(self):
        assert _any_gameweek_played(pd.DataFrame()) is False

    def test_none_is_false(self):
        assert _any_gameweek_played(None) is False

    def test_all_zero_points_is_false(self):
        df = pd.DataFrame({"Team": ["A", "B"], "GW_Points": [0, 0]})
        assert _any_gameweek_played(df) is False

    def test_missing_gw_points_column_is_false(self):
        assert _any_gameweek_played(pd.DataFrame({"Team": ["A"]})) is False

    def test_real_score_is_true(self):
        df = pd.DataFrame({"Team": ["A", "B"], "GW_Points": [0, 65]})
        assert _any_gameweek_played(df) is True


class TestSeasonLabelFromLeagueData:
    def test_derives_from_draft_date_in_august(self):
        league_data = {"league": {"drafts": [{"draft_dt": "2026-08-09T16:00:00Z"}]}}
        assert _season_label_from_league_data(league_data) == "2026/27"

    def test_derives_from_draft_date_in_spring(self):
        """A draft in, say, March belongs to the season that started the
        previous August."""
        league_data = {"league": {"drafts": [{"draft_dt": "2027-03-01T16:00:00Z"}]}}
        assert _season_label_from_league_data(league_data) == "2026/27"

    def test_falls_back_when_no_draft_data(self):
        with patch("scripts.draft.league_wrapped.config.display_pl_season_label", return_value="2099/00"):
            assert _season_label_from_league_data({"league": {}}) == "2099/00"

    def test_falls_back_on_malformed_draft_dt(self):
        league_data = {"league": {"drafts": [{"draft_dt": "not-a-date"}]}}
        with patch("scripts.draft.league_wrapped.config.display_pl_season_label", return_value="2099/00"):
            assert _season_label_from_league_data(league_data) == "2099/00"


class TestShowLeagueWrappedPagePreseasonGuard:
    def test_no_games_played_shows_notice_not_crash(self, mock_all_utils):
        """The guard must short-circuit BEFORE any of the seven downstream
        data-computation calls, which otherwise cascade into real,
        unmocked network calls — this test relies on that early return to
        stay fast; it isn't itself a test of the downstream sections."""
        league_data = {
            "league": {"name": "FPL Friends", "drafts": [{"draft_dt": "2026-08-09T16:00:00Z"}]},
            "league_entries": [{"id": 1, "entry_id": 56086, "entry_name": "Stoned Squirrels"}],
            "matches": [],
            "standings": [],
        }
        with patch("scripts.draft.league_wrapped.get_draft_league_details", return_value=league_data), \
             patch("scripts.draft.league_wrapped.get_current_gameweek", return_value=1), \
             patch("scripts.draft.league_wrapped.build_draft_history_df", return_value=pd.DataFrame()):
            from scripts.draft.league_wrapped import show_league_wrapped_page
            show_league_wrapped_page()  # should not raise
