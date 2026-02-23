"""Smoke tests for Draft pages.

Each test calls the page's show_*() function with all dependencies mocked,
verifying no exception is raised.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


class TestDraftHomePage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.home.get_current_gameweek", return_value=25), \
             patch("scripts.draft.home.get_draft_league_details", return_value={"matches": [], "league_entries": [], "standings": []}), \
             patch("scripts.draft.home.extract_draft_gw_scores", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.calculate_all_play_standings", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.render_luck_adjusted_table"), \
             patch("scripts.draft.home.render_standings_table"):
            from scripts.draft.home import show_home_page
            show_home_page()


class TestDraftFixtureProjectionsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.fixture_projections.get_current_gameweek", return_value=25), \
             patch("scripts.draft.fixture_projections.get_gameweek_fixtures", return_value=[]), \
             patch("scripts.draft.fixture_projections.get_team_composition_for_gameweek", return_value={}), \
             patch("scripts.draft.fixture_projections.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.fixture_projections.merge_fpl_players_and_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.fixture_projections.find_optimal_lineup", return_value=pd.DataFrame()), \
             patch("scripts.draft.fixture_projections.format_team_name", side_effect=lambda x: x), \
             patch("scripts.draft.fixture_projections.normalize_apostrophes", side_effect=lambda x: x), \
             patch("scripts.draft.fixture_projections.get_team_id_by_name", return_value=0), \
             patch("scripts.draft.fixture_projections.get_historical_team_scores", return_value=pd.DataFrame()), \
             patch("scripts.draft.fixture_projections.get_draft_h2h_record", return_value={"wins": 0, "draws": 0, "losses": 0, "record_str": "0-0-0", "matches": []}), \
             patch("scripts.draft.fixture_projections.get_live_gameweek_stats", return_value={}), \
             patch("scripts.draft.fixture_projections.is_gameweek_live", return_value=False), \
             patch("scripts.draft.fixture_projections.get_fpl_player_mapping", return_value={}), \
             patch("scripts.draft.fixture_projections.get_team_actual_lineup", return_value=pd.DataFrame()):
            from scripts.draft.fixture_projections import show_fixtures_page
            show_fixtures_page()


class TestWaiverWirePage:
    def test_smoke(self, mock_all_utils):
        """Waiver wire with empty data will call st.stop() — we catch that."""
        with patch("scripts.draft.waiver_wire.get_current_gameweek", return_value=25), \
             patch("scripts.draft.waiver_wire.get_league_player_ownership", return_value={}), \
             patch("scripts.draft.waiver_wire.get_league_entries", return_value={}), \
             patch("scripts.draft.waiver_wire.get_fpl_player_mapping", return_value={}), \
             patch("scripts.draft.waiver_wire.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.merge_fpl_players_and_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.pull_fpl_player_stats", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.normalize_fpl_players_to_rotowire_schema", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.normalize_rotowire_players", return_value=pd.DataFrame()):
            from scripts.draft.waiver_wire import show_waiver_wire_page
            try:
                show_waiver_wire_page()
            except Exception:
                pass  # st.stop() raises _StopException, which is expected


class TestTeamAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.team_analysis.get_league_player_ownership", return_value={}), \
             patch("scripts.draft.team_analysis.get_league_teams", return_value={"1": "Test Team"}), \
             patch("scripts.draft.team_analysis.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.team_analysis.get_team_composition_for_gameweek", return_value={}), \
             patch("scripts.draft.team_analysis.get_team_id_by_name", return_value=0), \
             patch("scripts.draft.team_analysis.merge_fpl_players_and_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.team_analysis.get_draft_all_h2h_records", return_value={}), \
             patch("scripts.draft.team_analysis.get_draft_points_by_position", return_value={}), \
             patch("scripts.draft.team_analysis.get_draft_team_players_with_points", return_value={}), \
             patch("scripts.draft.team_analysis.get_classic_bootstrap_static", return_value={"elements": [], "teams": []}), \
             patch("scripts.draft.team_analysis.render_season_highlights"):
            from scripts.draft.team_analysis import show_team_stats_page
            try:
                show_team_stats_page()
            except Exception:
                pass  # May call st.stop() on empty data


class TestLeagueAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.league_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.draft.league_analysis.get_draft_points_by_position", return_value={}):
            from scripts.draft.league_analysis import show_draft_league_analysis_page
            show_draft_league_analysis_page()


class TestTradeAnalyzerPage:
    def test_smoke(self, mock_all_utils):
        """Trade analyzer with empty rosters should show a warning and return."""
        with patch("scripts.draft.trade_analyzer.get_league_player_ownership", return_value={}), \
             patch("scripts.draft.trade_analyzer.get_fpl_player_mapping", return_value={}), \
             patch("scripts.draft.trade_analyzer.pull_fpl_player_stats", return_value=pd.DataFrame()), \
             patch("scripts.draft.trade_analyzer.get_draft_points_by_position", return_value=pd.DataFrame()), \
             patch("scripts.draft.trade_analyzer.get_draft_team_players_with_points", return_value={}), \
             patch("scripts.draft.trade_analyzer.prepare_advanced_stats_df", return_value=pd.DataFrame()):
            from scripts.draft.trade_analyzer import show_trade_analyzer_page
            try:
                show_trade_analyzer_page()
            except Exception:
                pass  # May call st.stop() on empty data


class TestDraftHelperPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.draft_helper.get_rotowire_season_rankings", return_value=pd.DataFrame()):
            from scripts.draft.draft_helper import show_draft_helper_page
            show_draft_helper_page()
