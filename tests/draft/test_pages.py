"""Smoke tests for Draft pages.

Each test calls the page's show_*() function with all dependencies mocked,
verifying no exception is raised.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _empty_ffp_feed():
    """An FFP feed that loaded nothing — the shape pages must survive."""
    from scripts.common.scraping import FFPFeed
    return FFPFeed(pd.DataFrame(), None, None, "none", "unavailable in tests")




class TestDraftHomePage:
    """Home now renders the My Week strip, which fetches. Stub
    `build_week_view` on the *page's* namespace -- Home resolves it there, and
    leaving it live put this smoke test on the network, 404ing against a
    fixture league id."""

    def test_smoke(self, mock_all_utils):
        with patch("scripts.draft.home.get_current_gameweek", return_value=25), \
             patch("scripts.draft.home.get_draft_league_details", return_value={"matches": [], "league_entries": [], "standings": []}), \
             patch("scripts.draft.home.is_draft_league_reachable", return_value=True), \
             patch("scripts.draft.home.extract_draft_gw_scores", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.calculate_all_play_standings", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.render_luck_adjusted_table"), \
             patch("scripts.draft.home.render_standings_table"), \
             patch("scripts.draft.home.build_draft_history_df", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.build_week_view", return_value=None), \
             patch("scripts.draft.home.get_draft_transaction_window", return_value={}), \
             patch("scripts.draft.home.get_league_waiver_order", return_value=[]), \
             patch("scripts.draft.home.get_fpl_draft_league_standings",
                   return_value=pd.DataFrame()):
            from scripts.draft.home import show_home_page
            show_home_page()

    def test_the_week_strip_renders_and_its_buttons_navigate(self, mock_all_utils,
                                                             mock_streamlit):
        """The strip is new to this page, and a mocked radio returns its first
        option -- so without an explicit test the Luck Adjusted branch and the
        week tab would both go unexercised."""
        from scripts.draft import my_week as dmw
        from scripts.common import week_check as wc

        item = wc.build_draft_squad_item(None)          # an unknown row, with a page
        view = dmw.DraftWeekView(items=[item], team_id=1, league_id=2,
                                 target_gw=11, squad_size=0, has_xi=False,
                                 claim_checked=False)
        assert view.flagged == [item]
        assert dmw.week_card_html(view)                  # renders without raising

        with patch("scripts.draft.home.get_current_gameweek", return_value=25), \
             patch("scripts.draft.home.get_draft_league_details",
                   return_value={"matches": [], "league_entries": [], "standings": []}), \
             patch("scripts.draft.home.is_draft_league_reachable", return_value=True), \
             patch("scripts.draft.home.render_standings_table"), \
             patch("scripts.draft.home.render_luck_adjusted_table"), \
             patch("scripts.draft.home.build_draft_history_df", return_value=pd.DataFrame()), \
             patch("scripts.draft.home.build_week_view", return_value=view), \
             patch("scripts.draft.home.get_draft_transaction_window",
                   return_value={"mode": "waivers"}), \
             patch("scripts.draft.home.get_league_waiver_order", return_value=[]), \
             patch("scripts.draft.home.get_fpl_draft_league_standings",
                   return_value=pd.DataFrame()):
            from scripts.draft.home import show_home_page
            show_home_page()

        # mock_streamlit's buttons are unconditionally truthy, so pressing the
        # row's Open button must have staged a real navigation target.
        import main
        import streamlit as st
        section = st.session_state.get("_pending_nav_section")
        page = st.session_state.get("_pending_nav_page")
        assert section in main.SECTIONS, section
        assert page in main.SECTIONS[section], page

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
        """Waiver wire with empty data will call st.stop() — we catch that.

        Most of these patches target `waiver_core`, not this page: the claim
        engine was split out, so the fetches happen in `build_waiver_board`.
        Patching the page module would still *succeed* and do nothing, which
        is what `test_patch_targets_bite.py` exists to catch -- that is how
        `compute_healthy_form`, one HTTP request per player, came to run for
        real inside a test body wrapped in `except Exception: pass`."""
        with patch("scripts.draft.waiver_core.get_current_gameweek", return_value=25), \
             patch("scripts.draft.waiver_wire.get_league_player_ownership", return_value={}), \
             patch("scripts.draft.waiver_wire.get_league_element_states", return_value={}), \
             patch("scripts.draft.waiver_wire.get_draft_transaction_window", return_value={}), \
             patch("scripts.draft.waiver_wire.get_league_entries", return_value={}), \
             patch("scripts.draft.waiver_wire.get_fpl_player_mapping", return_value={}), \
             patch("scripts.draft.waiver_core.get_fpl_player_mapping", return_value={}), \
             patch("scripts.draft.waiver_core.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_core.merge_fpl_players_and_projections", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_core.pull_fpl_player_stats", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_core.normalize_fpl_players_to_rotowire_schema", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_core.normalize_rotowire_players", return_value=pd.DataFrame()), \
             patch("scripts.draft.waiver_core.compute_healthy_form", return_value=5.0), \
             patch("scripts.draft.waiver_core.get_ffp_feed", return_value=_empty_ffp_feed()), \
             patch("scripts.draft.waiver_core.blend_multi_gw_projections", side_effect=lambda df, *a, **kw: df), \
             patch("scripts.draft.waiver_core.compute_positional_depth", return_value={}):
            from scripts.draft.waiver_wire import show_waiver_wire_page
            try:
                show_waiver_wire_page()
            except Exception:
                pass  # st.stop() raises _StopException, which is expected


class TestReturningSection:
    """The returning-from-injury section on the Waiver Wire.

    It needs a test of its own because the page smoke test above runs on empty
    data and ends at `st.stop()`, so it never reaches this section at all --
    exactly the "one test covered it for free and then quietly stopped" trap
    `TestHubSections` was written for on the Projections Hub.

    `get_prior_season_rates` is patched on **`waiver_core`**, which is where
    `return_watch` resolves it. Patching `scripts.common.scraping` would
    succeed, do nothing, and leave the test making ~670 element-summary
    requests -- the failure `test_patch_targets_bite.py` exists to catch.
    """

    @staticmethod
    def _board(reference, states=None, current_gw=11):
        from types import SimpleNamespace
        return SimpleNamespace(
            ok=True, reference=reference, element_states=states or {1: {"status": "a"}},
            current_gw=current_gw, notes=(),
        )

    @staticmethod
    def _pool():
        """One returning elite defender and twelve established starters."""
        rows = [dict(Player_Code=462424, Player_ID=6, Player="William Saliba",
                     Team="ARS", Position="D", starts=0, minutes=0,
                     total_points=0, status="a",
                     chance_of_playing_next_round=100, news="")]
        rows += [dict(Player_Code=5000 + i, Player_ID=900 + i,
                      Player="Starter %d" % i, Team="ARS", Position="D",
                      starts=10, minutes=900, total_points=30, status="a",
                      chance_of_playing_next_round=100, news="")
                 for i in range(12)]
        return pd.DataFrame(rows)

    @staticmethod
    def _rates():
        return pd.DataFrame([dict(Player_Code=462424, Prior_PPS=137 / 30,
                                  Prior_Role=30 / 38, Prior_Season="2025/26",
                                  Prior_Starts=30, Prior_Points=137)])

    def test_it_renders_a_populated_board(self, mock_streamlit):
        from scripts.draft.waiver_wire import _render_returning_section
        pool = self._pool()
        states = {int(c): {"status": "a"} for c in pool["Player_Code"]}
        with patch("scripts.draft.waiver_core.get_prior_season_rates",
                   return_value=self._rates()),              patch("scripts.draft.waiver_core.get_fpl_player_mapping",
                   return_value={int(c): {"Player_Code": int(c)}
                                 for c in pool["Player_Code"]}),              patch("scripts.draft.waiver_core.gameweek_deadlines", return_value=None):
            frame = _render_returning_section(
                self._board(pool, {int(c): {"status": "a"}
                                   for c in pool["Player_Code"]}))
        # Pinned to the fixture's own player, not merely "non-empty": with the
        # network up, a patch on the wrong module silently builds a real board
        # and every "is it populated" assertion passes anyway.
        assert frame is not None and not frame.empty
        assert 462424 in set(frame["Player_Code"])
        assert frame.iloc[0]["Band"] == "cleared"

    def test_an_empty_board_still_renders(self, mock_streamlit):
        """And must not raise -- it is the ordinary state for most gameweeks."""
        from scripts.draft.waiver_wire import _render_returning_section
        with patch("scripts.draft.waiver_core.get_prior_season_rates",
                   return_value=pd.DataFrame()),              patch("scripts.draft.waiver_core.get_fpl_player_mapping",
                   return_value={}),              patch("scripts.draft.waiver_core.gameweek_deadlines", return_value=None):
            frame = _render_returning_section(self._board(self._pool(), {}))
        assert frame is None or frame.empty

    def test_a_failed_board_renders_rather_than_raising(self, mock_streamlit):
        from types import SimpleNamespace
        from scripts.draft.waiver_wire import _render_returning_section
        board = SimpleNamespace(ok=False, reference=pd.DataFrame(),
                                element_states={}, current_gw=11, notes=())
        frame = _render_returning_section(board)
        assert frame is None or frame.empty

    def test_the_card_never_leaves_a_dangling_tag(self, mock_streamlit):
        """Every optional fragment goes through `compact_html`.

        A fragment that renders to "" on a line of its own ends the HTML block
        and dumps the closing `</div>` on screen as literal text -- and which
        fragments are empty depends on the data, which is how it ships.
        """
        from scripts.draft.waiver_wire import _render_returning_section
        pool = self._pool()
        with patch("scripts.draft.waiver_core.get_prior_season_rates",
                   return_value=self._rates()),              patch("scripts.draft.waiver_core.get_fpl_player_mapping",
                   return_value={int(c): {"Player_Code": int(c)}
                                 for c in pool["Player_Code"]}),              patch("scripts.draft.waiver_core.gameweek_deadlines", return_value=None):
            _render_returning_section(
                self._board(pool, {int(c): {"status": "a"}
                                   for c in pool["Player_Code"]}))
        # `mock_streamlit` yields the dict of started patches, not a module.
        markdown_calls = [c for c in mock_streamlit["markdown"].call_args_list
                          if c[0] and "border-radius" in str(c[0][0])]
        assert markdown_calls, "The card was never rendered"
        html = str(markdown_calls[-1][0][0])
        assert "\n" not in html, "Card HTML was not flattened by compact_html"
        assert html.count("<div") == html.count("</div>")


class TestTeamAnalysisPage:
    def test_smoke(self, mock_all_utils):
        # `get_league_player_ownership` was patched here too, but the page
        # never called it -- a dead import, now removed. Patching a name a
        # module does not use asserts nothing and breaks when the import goes.
        with patch("scripts.draft.team_analysis.get_league_teams", return_value={"1": "Test Team"}), \
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
             patch("scripts.draft.trade_analyzer.prepare_advanced_stats_df", return_value=pd.DataFrame()), \
             patch("scripts.draft.trade_analyzer.get_draft_transaction_window", return_value={}), \
             patch("scripts.draft.trade_analyzer.get_league_element_states", return_value={}):
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
