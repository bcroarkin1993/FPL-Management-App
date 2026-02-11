"""Smoke tests for Classic pages.

Each test calls the page's show_*() function with all dependencies mocked,
verifying no exception is raised.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestClassicHomePage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.home.get_league_standings", return_value=None), \
             patch("scripts.classic.home.get_classic_team_history", return_value=None), \
             patch("scripts.classic.home.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.home.get_current_gameweek", return_value=25), \
             patch("scripts.classic.home.get_all_h2h_league_matches", return_value=[]), \
             patch("scripts.classic.home.extract_classic_h2h_gw_scores", return_value=pd.DataFrame()), \
             patch("scripts.classic.home.calculate_all_play_standings", return_value=pd.DataFrame()), \
             patch("scripts.classic.home.render_luck_adjusted_table"), \
             patch("scripts.classic.home.render_standings_table"):
            from scripts.classic.home import show_classic_home_page
            show_classic_home_page()


class TestClassicFixtureProjectionsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.fixture_projections.get_current_gameweek", return_value=25), \
             patch("scripts.classic.fixture_projections.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.fixture_projections.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.fixture_projections.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.fixture_projections.find_optimal_lineup", return_value=pd.DataFrame()), \
             patch("scripts.classic.fixture_projections.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.fixture_projections.get_league_standings", return_value=None), \
             patch("scripts.classic.fixture_projections.get_h2h_league_matches", return_value=[]), \
             patch("scripts.classic.fixture_projections.get_classic_h2h_record", return_value={"wins": 0, "draws": 0, "losses": 0, "record_str": "0-0-0", "matches": []}), \
             patch("scripts.classic.fixture_projections.get_classic_transfers", return_value=[]), \
             patch("scripts.classic.fixture_projections.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.fixture_projections.show_api_error"):
            from scripts.classic.fixture_projections import show_classic_fixture_projections_page
            show_classic_fixture_projections_page()


class TestClassicTransfersPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.transfers.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.transfers.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.transfers.get_classic_team_history", return_value=None), \
             patch("scripts.classic.transfers.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.transfers.get_current_gameweek", return_value=25), \
             patch("scripts.classic.transfers.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.transfers.get_classic_transfers", return_value=[]), \
             patch("scripts.classic.transfers.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.transfers.show_api_error"):
            from scripts.classic.transfers import show_classic_transfers_page
            show_classic_transfers_page()


class TestFreeHitPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.free_hit.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.free_hit.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.free_hit.get_current_gameweek", return_value=25), \
             patch("scripts.classic.free_hit.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.free_hit.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.free_hit.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.free_hit.show_api_error"):
            from scripts.classic.free_hit import show_free_hit_page
            show_free_hit_page()


class TestWildcardPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.wildcard.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.wildcard.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.wildcard.get_current_gameweek", return_value=25), \
             patch("scripts.classic.wildcard.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.wildcard.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.wildcard.get_fixture_difficulty_grid", return_value=pd.DataFrame()), \
             patch("scripts.classic.wildcard.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.wildcard.show_api_error"):
            from scripts.classic.wildcard import show_wildcard_page
            show_wildcard_page()


class TestClassicTeamAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.team_analysis.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.team_analysis.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.team_analysis.get_classic_team_history", return_value=None), \
             patch("scripts.classic.team_analysis.get_classic_team_position_data", return_value={}), \
             patch("scripts.classic.team_analysis.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.team_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.classic.team_analysis.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.team_analysis.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.team_analysis.render_season_highlights"):
            from scripts.classic.team_analysis import show_classic_team_analysis_page
            show_classic_team_analysis_page()


class TestClassicLeagueAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.league_analysis.get_league_standings", return_value=None), \
             patch("scripts.classic.league_analysis.get_classic_team_history", return_value=None), \
             patch("scripts.classic.league_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.classic.league_analysis.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.league_analysis.get_classic_team_position_data", return_value={}), \
             patch("scripts.classic.league_analysis.show_api_error"):
            from scripts.classic.league_analysis import show_classic_league_analysis_page
            show_classic_league_analysis_page()
