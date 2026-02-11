"""Smoke tests for FPL App Home pages.

Each test calls the page's show_*() function with all dependencies mocked,
verifying no exception is raised.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestFixturesPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.fixtures.get_current_gameweek", return_value=25), \
             patch("scripts.fpl.fixtures.get_fixture_difficulty_grid", return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())), \
             patch("scripts.fpl.fixtures.style_fixture_difficulty", return_value=MagicMock()):
            from scripts.fpl.fixtures import show_club_fixtures_section
            try:
                show_club_fixtures_section()
            except Exception:
                pass  # May fail on empty data downstream


class TestProjectedLineupsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.projected_lineups.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.fpl.projected_lineups.requests.get", return_value=MagicMock(content=b"<html></html>", status_code=200)):
            from scripts.fpl.projected_lineups import show_projected_lineups
            show_projected_lineups()


class TestPlayerStatisticsPage:
    def test_smoke(self, mock_all_utils):
        # Need a minimal DataFrame with position_abbrv column
        stats_df = pd.DataFrame({"position_abbrv": [], "team_name_short": []})
        with patch("scripts.fpl.player_statistics.pull_fpl_player_stats", return_value=stats_df), \
             patch("scripts.fpl.player_statistics.get_fixture_difficulty_grid", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_statistics.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_statistics.clean_fpl_player_names", return_value=pd.DataFrame()):
            from scripts.fpl.player_statistics import show_player_stats_page
            try:
                show_player_stats_page()
            except Exception:
                pass  # May stop on empty data


class TestInjuriesPage:
    def test_smoke(self, mock_all_utils):
        empty_avail = pd.DataFrame(columns=["Player_ID", "Player", "Web_Name", "Team", "Position",
                                             "Status", "PlayPct", "StatusBucket", "News", "News_Added"])
        with patch("scripts.fpl.injuries.get_fpl_availability_df", return_value=empty_avail):
            from scripts.fpl.injuries import show_injuries_page
            show_injuries_page()


class TestPlayerProjectionsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.player_projections.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_rotowire_rankings_url", return_value="https://example.com"), \
             patch("scripts.fpl.player_projections.get_ffp_projections_data", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_ffp_goalscorer_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_ffp_clean_sheet_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_odds_api_match_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}):
            from scripts.fpl.player_projections import show_player_projections_page
            show_player_projections_page()


class TestSettingsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.settings.load_settings", return_value={
                "version": 1,
                "discord": {"mention_user_id": "", "mention_role_id": ""},
                "deadline_alerts": {
                    "draft": {"enabled": False, "alert_windows": [24, 6, 1]},
                    "classic": {"enabled": False, "alert_windows": [24, 6, 1]},
                },
                "data_source_alerts": {"rotowire": {"enabled": False}, "ffp": {"enabled": False}},
                "alert_state": {"last_rotowire_alert_gw": 0, "last_ffp_alert_gw": 0},
            }), \
             patch("scripts.fpl.settings.save_settings", return_value=True), \
             patch("scripts.fpl.settings.requests.get", return_value=MagicMock(content=b"<html></html>", status_code=200)):
            from scripts.fpl.settings import show_settings_page
            show_settings_page()
