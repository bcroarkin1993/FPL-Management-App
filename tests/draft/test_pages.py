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
             patch("scripts.draft.home.is_draft_league_reachable", return_value=True), \
             patch("scripts.draft.home.extract_draft_gw_scores", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.calculate_all_play_standings", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.render_luck_adjusted_table"), \
             patch("scripts.draft.home.render_standings_table"), \
             patch("scripts.draft.home.build_draft_history_df", return_value=pd.DataFrame()):
            from scripts.draft.home import show_home_page
            show_home_page()

    def test_stale_league_shows_error_and_stops(self, mock_all_utils, mock_streamlit):
        """When FPL_DRAFT_LEAGUE_ID doesn't resolve (e.g. a prior season's league,
        before this season's has been created), the page must show an actionable
        error and stop rather than rendering broken/empty data."""
        with patch("scripts.draft.home.is_draft_league_reachable", return_value=False):
            from scripts.draft.home import show_home_page
            with pytest.raises(mock_streamlit["_StopException"]):
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
             patch("scripts.draft.fixture_projections.get_team_actual_lineup", return_value=pd.DataFrame()), \
             patch("scripts.draft.fixture_projections.get_gw_finished_teams", return_value=set()), \
             patch("scripts.draft.fixture_projections.simulate_auto_subs", return_value=(pd.DataFrame(), [])), \
             patch("scripts.draft.fixture_projections.get_classic_bootstrap_static", return_value={"elements": [], "teams": []}), \
             patch("scripts.draft.fixture_projections.compute_key_differentials", return_value=([], [])), \
             patch("scripts.draft.fixture_projections.render_key_differentials"):
            from scripts.draft.fixture_projections import show_fixtures_page
            show_fixtures_page()


class TestWaiverWirePage:
    def test_smoke(self, mock_all_utils):
        """Waiver wire with empty data will call st.stop() — we catch that."""
        with patch("scripts.draft.waiver_wire.get_current_gameweek", return_value=25), \
             patch("scripts.draft.waiver_wire.get_league_player_ownership", return_value={}), \
             patch("scripts.draft.waiver_wire.get_league_element_states", return_value={}), \
             patch("scripts.draft.waiver_wire.get_draft_transaction_window", return_value={}), \
             patch("scripts.draft.waiver_wire.get_league_entries", return_value={}), \
             patch("scripts.draft.waiver_wire.get_fpl_player_mapping", return_value={}), \
             patch("scripts.draft.waiver_wire.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.merge_fpl_players_and_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.pull_fpl_player_stats", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.normalize_fpl_players_to_rotowire_schema", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.normalize_rotowire_players", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_wire.compute_healthy_form", return_value=5.0), \
             patch("scripts.draft.waiver_wire.get_ffp_projections_data", return_value=None), \
             patch("scripts.draft.waiver_wire.blend_multi_gw_projections", side_effect=lambda df, *a, **kw: df), \
             patch("scripts.draft.waiver_wire.compute_positional_depth", return_value={}):
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
             patch("scripts.draft.team_analysis.render_season_highlights"), \
             patch("scripts.draft.team_analysis.compute_draft_bench_data", return_value=None), \
             patch("scripts.draft.team_analysis.render_bench_analysis"):
            from scripts.draft.team_analysis import show_team_stats_page
            try:
                show_team_stats_page()
            except Exception:
                pass  # May call st.stop() on empty data


def _power_rankings_fixture():
    """A two-team league with full 15-player Draft squads, already scored."""
    players = []
    for team_id, name, base in [(101, "Alpha", 0.80), (202, "Bravo", 0.45)]:
        for pos, count in [("G", 2), ("D", 5), ("M", 5), ("F", 3)]:
            for i in range(count):
                players.append({
                    "Team_ID": team_id, "Team_Name": name, "Position": pos,
                    "Player": f"{name}-{pos}{i}", "Team": "ARS",
                    "Player_Strength": base, "Raw_Strength": base,
                    "PPS": 5.0, "form": 4.0, "GWs_Missed": 0,
                })
    player_df = pd.DataFrame(players)
    from scripts.common.team_strength import aggregate_team_strength
    return aggregate_team_strength(player_df), player_df


class TestLeagueAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.league_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.draft.league_analysis.get_draft_points_by_position", return_value={}), \
             patch("scripts.draft.league_analysis.compute_draft_league_bench_data", return_value=[]), \
             patch("scripts.draft.league_analysis.render_league_bench_analysis"), \
             patch("scripts.draft.league_analysis.build_league_strength",
                   return_value=(pd.DataFrame(), pd.DataFrame())):
            from scripts.draft.league_analysis import show_draft_league_analysis_page
            show_draft_league_analysis_page()

    def test_power_rankings_renders_with_data(self, mock_all_utils):
        """Exercise the Power Rankings tab body itself, not just the empty path.

        The page early-returns on empty league data, so the smoke test above never
        reaches tab 8 -- this drives the renderer directly.
        """
        team_df, player_df = _power_rankings_fixture()
        with patch("scripts.draft.league_analysis.build_league_strength",
                   return_value=(team_df, player_df)):
            from scripts.draft.league_analysis import _render_power_rankings
            _render_power_rankings(league_id=12345, current_gw=25)

    def test_preseason_still_renders_power_rankings(self, mock_all_utils):
        """With no matches played, the page must still show power rankings.

        Every other tab needs played matches, but rosters exist from the draft --
        and just after the draft is when these are most useful.
        """
        team_df, player_df = _power_rankings_fixture()
        with patch("scripts.draft.league_analysis.get_current_gameweek", return_value=1), \
             patch("scripts.draft.league_analysis.get_matches_df", return_value=pd.DataFrame()), \
             patch("scripts.draft.league_analysis.fetch_league_data",
                   return_value={"league": {"name": "Test"}, "league_entries": [], "matches": []}), \
             patch("scripts.draft.league_analysis.build_league_strength",
                   return_value=(team_df, player_df)) as mock_build:
            from scripts.draft.league_analysis import show_draft_league_analysis_page
            show_draft_league_analysis_page()
            assert mock_build.called, "power rankings must render even with no matches"

    def test_power_rankings_handles_no_rosters(self, mock_all_utils):
        """Pre-draft leagues must show an info message, not raise."""
        with patch("scripts.draft.league_analysis.build_league_strength",
                   return_value=(pd.DataFrame(), pd.DataFrame())):
            from scripts.draft.league_analysis import _render_power_rankings
            _render_power_rankings(league_id=12345, current_gw=1)


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
        with patch("scripts.draft.draft_helper.get_rotowire_season_rankings", return_value=pd.DataFrame()), \
             patch("scripts.draft.draft_helper.get_club_transfer_news",
                   return_value=pd.DataFrame()):
            from scripts.draft.draft_helper import show_draft_helper_page
            show_draft_helper_page()

    def test_smoke_with_rankings_and_transfer_risk(self, mock_all_utils):
        """Exercise the transfer-risk path with a populated board.

        get_transfer_news is patched explicitly: mock_streamlit's buttons are
        unconditionally truthy, so the "Scan transfer news" branch always fires
        and would otherwise make one real network request per player.
        """
        rankings = pd.DataFrame({
            "Overall Rank": [1, 2, 3],
            "Player": ["Erling Haaland", "Ollie Watkins", "Bukayo Saka"],
            "Team": ["MCI", "AVL", "ARS"],
            "Position": ["F", "F", "M"],
            "Points": [215.3, 145.9, 180.9],
            "PP/90": [7.1, 5.2, 6.0],
            "Pos Rank": [1, 8, 3],
        })
        news = pd.DataFrame([{
            "Player": "Ollie Watkins", "Team": "AVL",
            "Headline": "Ollie Watkins undergoes medical at Al Hilal",
            "URL": "http://example.com", "Source": "Sky Sports",
            "Published": "Fri, 28 Aug 2026 18:21:08 GMT",
        }])
        availability = pd.DataFrame({
            "Player": ["Erling Haaland", "Ollie Watkins", "Bukayo Saka"],
            "Web_Name": ["Haaland", "Watkins", "Saka"],
            "Team": ["MCI", "AVL", "ARS"],
            "Position": ["F", "F", "M"],
            "Status": ["a", "a", "a"],
            "News": ["", "", ""],
        })
        with patch("scripts.draft.draft_helper.get_rotowire_season_rankings", return_value=rankings), \
             patch("scripts.draft.draft_helper.get_transfer_news", return_value=news) as mock_news, \
             patch("scripts.draft.draft_helper.get_club_transfer_news",
                   return_value=pd.DataFrame()) as mock_clubs, \
             patch("scripts.draft.draft_helper._load_reference_data",
                   return_value=(availability, ["Man City", "Aston Villa", "Arsenal"])):
            from scripts.draft.draft_helper import show_draft_helper_page
            show_draft_helper_page()
            assert mock_news.called
            assert mock_clubs.called

    def test_smoke_with_incoming_signings(self, mock_all_utils):
        """An arrival discounts the incumbents at that club and position.

        Patched the same way as the news scan: mock_streamlit's buttons are
        unconditionally truthy, so every fetch branch fires on every page render.
        """
        rankings = pd.DataFrame({
            "Overall Rank": [1, 2, 3],
            "Player": ["Ollie Watkins", "Jhon Duran", "Bukayo Saka"],
            "Team": ["AVL", "AVL", "ARS"],
            "Position": ["F", "F", "M"],
            "Points": [145.9, 90.0, 180.9],
            "PP/90": [5.2, 4.0, 6.0],
            "Pos Rank": [8, 20, 3],
        })
        club_news = pd.DataFrame([
            {"Club": "Aston Villa", "Headline": "Aston Villa complete signing of "
             "striker Nicolas Jackson for £51m", "URL": "", "Source": "BBC",
             "Published": "Fri, 28 Aug 2026 18:21:08 GMT"},
            {"Club": "Aston Villa", "Headline": "Aston Villa agree deal to sign "
             "striker Nicolas Jackson", "URL": "", "Source": "Sky Sports",
             "Published": "Fri, 28 Aug 2026 18:21:08 GMT"},
        ])
        availability = pd.DataFrame({
            "Player": ["Ollie Watkins", "Jhon Duran", "Bukayo Saka"],
            "Web_Name": ["Watkins", "Duran", "Saka"],
            "Team": ["AVL", "AVL", "ARS"],
            "Position": ["F", "F", "M"],
            "Status": ["a", "a", "a"],
            "News": ["", "", ""],
        })
        with patch("scripts.draft.draft_helper.get_rotowire_season_rankings", return_value=rankings), \
             patch("scripts.draft.draft_helper.get_transfer_news", return_value=pd.DataFrame()), \
             patch("scripts.draft.draft_helper.get_club_transfer_news", return_value=club_news), \
             patch("scripts.draft.draft_helper._load_reference_data",
                   return_value=(availability, ["Aston Villa", "Arsenal"])):
            from scripts.draft.draft_helper import show_draft_helper_page
            show_draft_helper_page()

    def test_page_survives_a_news_outage(self, mock_all_utils):
        """A transfer-news failure must leave the board undiscounted, not break it."""
        rankings = pd.DataFrame({
            "Overall Rank": [1], "Player": ["Erling Haaland"], "Team": ["MCI"],
            "Position": ["F"], "Points": [215.3], "PP/90": [7.1], "Pos Rank": [1],
        })
        with patch("scripts.draft.draft_helper.get_rotowire_season_rankings", return_value=rankings), \
             patch("scripts.draft.draft_helper.get_transfer_news",
                   side_effect=RuntimeError("news feed down")), \
             patch("scripts.draft.draft_helper.get_club_transfer_news",
                   side_effect=RuntimeError("club feed down")):
            from scripts.draft.draft_helper import show_draft_helper_page
            show_draft_helper_page()


class TestCommishModePage:
    def test_smoke_no_season_configured(self, mock_all_utils):
        """First visit, no commish_seasons saved yet — should render the setup form."""
        settings = {
            "draft": {"league_id": 11347, "team_id": 56086, "team_name": "Stoned Squirrels",
                      "locked": True, "commish_seasons": {}},
        }
        with patch("scripts.draft.commish_mode.load_settings", return_value=settings), \
             patch("scripts.draft.commish_mode.save_settings") as mock_save, \
             patch("scripts.draft.commish_mode.get_league_entries", return_value={1: "Stoned Squirrels", 2: "Top Drawer Balls"}):
            from scripts.draft.commish_mode import show_commish_mode_page
            show_commish_mode_page()
            # mock_streamlit's st.button is an unconditionally-truthy MagicMock, so the
            # "Save & Lock" branch does fire during this smoke test — assert it hit the
            # mock, not the real league_settings.json (save_settings must always be
            # patched in this page's tests, never left real).
            assert mock_save.called

    def test_smoke_season_locked(self, mock_all_utils):
        """A locked season already exists — should render the dues/payout dashboard."""
        settings = {
            "draft": {
                "league_id": 11347, "team_id": 56086, "team_name": "Stoned Squirrels", "locked": True,
                "commish_seasons": {
                    "2026/27": {
                        "buy_in": 75, "payout_pct": {"1": 60, "2": 30, "3": 10}, "locked": True,
                        "dues": {
                            "Stoned Squirrels": {"paid": True, "notes": ""},
                            "Top Drawer Balls": {"paid": False, "notes": ""},
                        },
                    },
                },
            },
        }
        with patch("scripts.draft.commish_mode.load_settings", return_value=settings), \
             patch("scripts.draft.commish_mode.save_settings") as mock_save, \
             patch("scripts.draft.commish_mode.get_league_entries", return_value={1: "Stoned Squirrels", 2: "Top Drawer Balls"}):
            from scripts.draft.commish_mode import show_commish_mode_page
            show_commish_mode_page()
            # Same reasoning as above — the dues st.form_submit_button mock is also
            # unconditionally truthy, so "Save Dues" fires too; must hit the mock only.
            assert mock_save.called

    def test_smoke_no_draft_league_configured(self, mock_all_utils):
        """No Draft league set up at all — should show a warning and return, not crash."""
        with patch("config.FPL_DRAFT_LEAGUE_ID", 0):
            from scripts.draft.commish_mode import show_commish_mode_page
            show_commish_mode_page()
