"""Tests for scripts/draft/league_wrapped.py's preseason-guard logic and
archived-season rendering."""

from unittest.mock import patch

import pandas as pd

from scripts.draft.league_wrapped import _any_gameweek_played, _season_label_from_league_data

_MINIMAL_ARCHIVE_DATA = {
    "standings": [
        {"rank": 1, "team": "Alpha", "w": 20, "d": 0, "l": 18, "league_pts": 60,
         "pts_for": 1600, "pts_against": 1500, "pts_diff": 100},
        {"rank": 2, "team": "Beta", "w": 18, "d": 0, "l": 20, "league_pts": 54,
         "pts_for": 1500, "pts_against": 1600, "pts_diff": -100},
    ],
    "top_players": {"awards": {}, "by_position": {"GK": [], "DEF": [], "MID": [], "FWD": []}},
    "league_awards": [{"icon": "🏆", "title": "Points Champion", "team": "Alpha", "detail": "1,600 pts", "accent": "#FFD700"}],
    "gw_highlights": {"highest_gw": {"team": "Alpha", "gw": 10, "score": 90}},
    "h2h_matrix": {"teams": ["Alpha", "Beta"], "cells": {"Alpha": {"Beta": "2-0-1"}, "Beta": {"Alpha": "1-0-2"}}},
    "draft_board": {"steals": [], "busts": [], "team_grades": {}},
    "transfer_window": {"per_team": {"Alpha": 5}, "best_transfer": {}, "worst_transfer": {}, "most_in": [], "most_out": []},
    "lineup_management": [{"Team": "Alpha", "Pts Scored": 1600, "Pts Possible": 1700, "Total Pts Lost": 100}],
}


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


class TestShowLeagueWrappedPageArchivedSeason:
    """Note: the season-selector widget itself (list_archived_seasons() ->
    st.selectbox -> load_archived_season() -> _render_archived_league_wrapped())
    is exercised end-to-end against real transcribed 2025/26 data via
    Streamlit's AppTest harness during manual verification of this feature —
    that's a more reliable way to test actual widget-selection behavior than
    this project's mock_streamlit fixture, whose selectbox mock doesn't
    faithfully reproduce real widget state. This test instead directly
    exercises the archived-render orchestration function, which is what
    actually needs regression coverage here."""

    def test_render_archived_league_wrapped_all_sections_no_raise(self, mock_streamlit):
        """Directly exercises the archived-render orchestration (bypassing
        the season-selector widget) against a minimal but complete data
        shape covering every section."""
        from scripts.draft.league_wrapped import _render_archived_league_wrapped
        _render_archived_league_wrapped("2025/26", _MINIMAL_ARCHIVE_DATA)  # should not raise
