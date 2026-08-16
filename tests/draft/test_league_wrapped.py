"""Tests for scripts/draft/league_wrapped.py's preseason-guard logic and
archived-season rendering."""

from unittest.mock import patch

import pandas as pd

from scripts.draft.league_wrapped import (
    _compute_league_awards_list,
    _h2h_df_to_dict,
    _season_label_from_league_data,
    _standings_list_from_league_data,
)

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


class TestShowLeagueWrappedPageSeasonNotConcludedGuard:
    def test_season_not_complete_shows_notice_not_crash(self, mock_all_utils):
        """League Wrapped is an end-of-season recap, not a live tracker — it
        must stay hidden while the season is still in progress, and the
        guard must short-circuit BEFORE any of the seven downstream
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
             patch("scripts.draft.league_wrapped.is_season_complete", return_value=False), \
             patch("scripts.draft.league_wrapped.build_draft_history_df") as mock_history:
            from scripts.draft.league_wrapped import show_league_wrapped_page
            show_league_wrapped_page()  # should not raise
        mock_history.assert_not_called()


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


class TestStandingsListFromLeagueData:
    def test_builds_flat_rows_matching_archive_shape(self):
        league_data = {
            "league_entries": [{"id": 1, "entry_name": "Alpha"}, {"id": 2, "entry_name": "Beta"}],
            "standings": [
                {"league_entry": 1, "rank": 1, "matches_won": 20, "matches_drawn": 0, "matches_lost": 18,
                 "points_for": 1600, "points_against": 1500},
                {"league_entry": 2, "rank": 2, "matches_won": 18, "matches_drawn": 0, "matches_lost": 20,
                 "points_for": 1500, "points_against": 1600},
            ],
        }
        rows = _standings_list_from_league_data(league_data)
        assert rows[0] == {"rank": 1, "team": "Alpha", "w": 20, "d": 0, "l": 18,
                            "league_pts": 60, "pts_for": 1600, "pts_against": 1500, "pts_diff": 100}
        assert rows[1]["team"] == "Beta"

    def test_missing_standings_or_entries_returns_empty(self):
        assert _standings_list_from_league_data({"league_entries": [], "standings": []}) == []


class TestH2hDfToDict:
    def test_round_trips_through_dataframe(self):
        df = pd.DataFrame({"Alpha": ["-", "2-0-1"], "Beta": ["1-0-2", "-"]}, index=["Alpha", "Beta"])
        result = _h2h_df_to_dict(df)
        assert result["teams"] == ["Alpha", "Beta"]
        assert result["cells"]["Alpha"]["Beta"] == "1-0-2"

    def test_empty_df_returns_empty_shape(self):
        assert _h2h_df_to_dict(pd.DataFrame()) == {"teams": [], "cells": {}}


class TestComputeLeagueAwardsList:
    def test_returns_eight_flat_award_dicts(self):
        history_df = pd.DataFrame({
            "Team": ["Alpha", "Beta"], "Gameweek": [1, 1],
            "GW_Points": [70, 30], "Total_Points": [70, 30],
        })
        awards = _compute_league_awards_list({}, history_df)
        assert len(awards) == 8
        assert all({"icon", "title", "team", "detail", "accent"} <= set(a) for a in awards)
        pts_champ = next(a for a in awards if a["title"] == "Points Champion")
        assert pts_champ["team"] == "Alpha"


class TestShowLeagueWrappedPageAutoArchive:
    def test_live_page_view_saves_archive_snapshot(self, mock_all_utils):
        """Visiting League Wrapped once the season has concluded should
        auto-save a season snapshot via save_archived_season, so H2H/awards/
        standings/etc. survive next season's Draft league-ID rollover
        instead of being lost the way 2025/26's data was."""
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
        with patch("scripts.draft.league_wrapped.get_draft_league_details", return_value=league_data), \
             patch("scripts.draft.league_wrapped.is_season_complete", return_value=True), \
             patch("scripts.draft.league_wrapped.get_current_gameweek", return_value=1), \
             patch("scripts.draft.league_wrapped.build_draft_history_df", return_value=history_df), \
             patch("scripts.draft.league_wrapped.compute_draft_league_bench_data", return_value=[]), \
             patch("scripts.draft.league_wrapped._compute_league_superlatives", return_value={}), \
             patch("scripts.draft.league_wrapped._compute_gw_highlights", return_value={}), \
             patch("scripts.draft.league_wrapped._compute_league_draft_board", return_value={}), \
             patch("scripts.draft.league_wrapped._compute_league_transfer_stats", return_value={}), \
             patch("scripts.draft.league_wrapped._compute_top_players", return_value={}), \
             patch("scripts.draft.league_wrapped.get_team_names", return_value={}), \
             patch("scripts.draft.league_wrapped.get_matches_df", return_value=pd.DataFrame()), \
             patch("scripts.draft.league_wrapped.list_archived_seasons", return_value=[]), \
             patch("scripts.draft.league_wrapped.save_archived_season") as mock_save, \
             patch("scripts.draft.league_wrapped.generate_league_wrapped_pdf", return_value=b""):
            from scripts.draft.league_wrapped import show_league_wrapped_page
            show_league_wrapped_page()

        assert mock_save.called
        season_arg, data_arg = mock_save.call_args[0]
        assert season_arg == "2026/27"
        assert data_arg["standings"][0]["team"] == "Alpha"
        assert data_arg["standings"][0]["league_pts"] == 6
        assert data_arg["h2h_matrix"] == {"teams": [], "cells": {}}
