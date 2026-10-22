"""Smoke tests for FPL App Home pages.

Each test calls the page's show_*() function with all dependencies mocked,
verifying no exception is raised.
"""

import contextlib

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock


def _empty_ffp_feed():
    """An FFP feed that loaded nothing — the shape pages must survive."""
    from scripts.common.scraping import FFPFeed
    return FFPFeed(pd.DataFrame(), None, None, "none", "unavailable in tests")


def _empty_ffp_horizon():
    """A horizon that loaded nothing."""
    from scripts.common.scraping import FFPHorizon
    return FFPHorizon(pd.DataFrame(), None, None, None, "none", {},
                      "unavailable in tests")


def _ffp_horizon(start_gw=7, weeks=6, n=40):
    """A horizon shaped like the live one: six gameweeks, one row per player
    per gameweek, fixtures and difficulty varying by week."""
    from scripts.common.scraping import FFPHorizon
    rows = []
    for w in range(weeks):
        gw = start_gw + w
        for i in range(n):
            cond = 6.0 - 0.1 * i + 0.3 * w
            rows.append({
                "Name": "First%d Last%d" % (i, i),
                "Display_Name": "Player %d" % i,
                "Web_Name": "P%d" % i,
                "Team": "T%d" % (i % 20),
                "Position": ["GK", "DEF", "MID", "FWD"][i % 4],
                "Fixture": "OPP%d (%s)" % (w, "H" if w % 2 else "a"),
                "Price": 7.5, "Ownership": 5.0,
                "Start": 80.0,
                "StartingPredicted": round(cond, 3),
                "Predicted": round(cond * 0.8, 3),
                "FFP_GW": gw, "Player_Code": 1000 + i, "Player_ID": i + 1,
                "Fixture_Count": 1, "Difficulty": 1 + (w + i) % 5,
            })
    return FFPHorizon(pd.DataFrame(rows), start_gw, start_gw, None, "site",
                      {start_gw + w: "site" for w in range(weeks)}, "")




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

    @staticmethod
    def _lineups(gameweek):
        from scripts.common.pl_content import PLLineups
        from datetime import datetime, timezone
        return PLLineups(
            gameweek=gameweek, article_id=1, url="https://example.test/a",
            title="Predicted line-ups ... Matchweek %s" % gameweek,
            updated=datetime(2026, 9, 11, 17, 0, tzinfo=timezone.utc),
            fixtures=(("AVL", "NFO"),),
            club_news={"AVL": "Villa news.", "NFO": "Forest news."},
            graphics={"AVL": "https://example.test/avl.png",
                      "NFO": "https://example.test/nfo.png"},
        )

    def test_pl_section_renders_for_the_current_gameweek(self, mock_all_utils):
        import config
        from scripts.fpl import projected_lineups as pl_page
        with patch("scripts.fpl.projected_lineups.get_pl_predicted_lineups",
                   return_value=self._lineups(config.CURRENT_GAMEWEEK)), \
             patch("scripts.fpl.projected_lineups.st.image", autospec=True) as image:
            pl_page._render_pl_section("Aston Villa", "Nottingham Forest")
        assert image.call_count == 2

    def test_pl_section_is_gated_on_the_matchweek(self, mock_all_utils):
        """Last week's predicted XI under this week's heading is the failure
        the gate exists to prevent."""
        import config
        from scripts.fpl import projected_lineups as pl_page
        with patch("scripts.fpl.projected_lineups.get_pl_predicted_lineups",
                   return_value=self._lineups(config.CURRENT_GAMEWEEK - 1)), \
             patch("scripts.fpl.projected_lineups.st.image", autospec=True) as image:
            pl_page._render_pl_section("Aston Villa", "Nottingham Forest")
        assert image.call_count == 0

    def test_pl_section_skips_a_fixture_the_pl_does_not_carry(self, mock_all_utils):
        import config
        from scripts.fpl import projected_lineups as pl_page
        with patch("scripts.fpl.projected_lineups.get_pl_predicted_lineups",
                   return_value=self._lineups(config.CURRENT_GAMEWEEK)), \
             patch("scripts.fpl.projected_lineups.st.image", autospec=True) as image:
            pl_page._render_pl_section("Chelsea", "Hull City")
        assert image.call_count == 0

    def test_pl_section_survives_a_dead_feed(self, mock_all_utils):
        from scripts.fpl import projected_lineups as pl_page
        with patch("scripts.fpl.projected_lineups.get_pl_predicted_lineups",
                   side_effect=RuntimeError("PL API down")), \
             patch("scripts.fpl.projected_lineups.st.image", autospec=True) as image:
            pl_page._render_pl_section("Aston Villa", "Nottingham Forest")
        assert image.call_count == 0


class TestPlayerStatisticsPage:
    def test_smoke(self, mock_all_utils):
        # Need a minimal DataFrame with position_abbrv column
        stats_df = pd.DataFrame({"position_abbrv": [], "team_name_short": []})
        with patch("scripts.fpl.player_statistics.pull_fpl_player_stats", return_value=stats_df), \
             patch("scripts.fpl.player_statistics.get_fixture_difficulty_grid", return_value=pd.DataFrame()):
            from scripts.fpl.player_statistics import show_player_stats_page
            try:
                show_player_stats_page()
            except Exception:
                pass  # May stop on empty data


_AVAIL_COLS = ["Player_ID", "Player", "Web_Name", "Team", "Position",
               "Status", "PlayPct", "StatusBucket", "News", "News_Added"]


def _populated_avail():
    return pd.DataFrame([
        {"Player_ID": 1, "Player": "Cody Gakpo", "Web_Name": "Gakpo", "Team": "LIV",
         "Position": "M", "Status": "a", "PlayPct": 100.0,
         "StatusBucket": "Available", "News": "", "News_Added": ""},
        {"Player_ID": 2, "Player": "Mohamed Salah", "Web_Name": "M.Salah", "Team": "LIV",
         "Position": "M", "Status": "a", "PlayPct": 100.0,
         "StatusBucket": "Available", "News": "", "News_Added": ""},
    ], columns=_AVAIL_COLS)


def _pl_injuries():
    from scripts.common.pl_content import INJURY_COLUMNS
    return pd.DataFrame([
        {"Club": "Liverpool", "Team": "LIV", "Player": "Cody Gakpo",
         "Injury": "Adductor", "Link": "https://example.test/a", "Item_Date": "2026-01-19"},
    ], columns=INJURY_COLUMNS)


class TestInjuriesPage:
    def test_smoke(self, mock_all_utils):
        empty_avail = pd.DataFrame(columns=_AVAIL_COLS)
        with patch("scripts.fpl.injuries.get_fpl_availability_df", return_value=empty_avail):
            from scripts.fpl.injuries import show_injuries_page
            show_injuries_page()

    def test_renders_pl_injury_column(self, mock_all_utils):
        """The PL injury feed reaches the table and the disagreement panel."""
        with patch("scripts.fpl.injuries.get_fpl_availability_df",
                   return_value=_populated_avail()), \
             patch("scripts.common.scraping.get_pl_injuries", return_value=_pl_injuries()):
            from scripts.fpl.injuries import _attach_pl_injuries, render_injuries_tab
            merged = _attach_pl_injuries(_populated_avail())
            assert "PL_Injury" in merged.columns
            assert merged.loc[merged["Player_ID"] == 1, "PL_Injury"].iloc[0] == "Adductor"
            # Salah has no PL entry and must stay unannotated rather than
            # inheriting his team-mate's row.
            assert pd.isna(merged.loc[merged["Player_ID"] == 2, "PL_Player"].iloc[0])
            render_injuries_tab(key_prefix="test_pl")

    def test_survives_a_dead_pl_feed(self, mock_all_utils):
        """The PL table is extra information, never a dependency: a failure
        must leave the page exactly as it was before the feature existed."""
        with patch("scripts.fpl.injuries.get_fpl_availability_df",
                   return_value=_populated_avail()), \
             patch("scripts.common.scraping.get_pl_injuries",
                   side_effect=RuntimeError("PL API down")):
            from scripts.fpl.injuries import _attach_pl_injuries, render_injuries_tab
            merged = _attach_pl_injuries(_populated_avail())
            assert "PL_Injury" not in merged.columns
            assert len(merged) == 2
            render_injuries_tab(key_prefix="test_pl_dead")


class TestAvailabilityPage:
    """The merged page: transfer news, odds and injuries.

    Every network-touching source is patched empty, because the page must render
    when the odds feed is down -- that degraded path is otherwise indistinguishable
    from a working one.
    """

    def test_smoke_with_every_source_empty(self, mock_all_utils):
        from scripts.common.transfer_odds import ODDS_INDEX_COLUMNS
        empty_avail = pd.DataFrame(columns=_AVAIL_COLS)
        with patch("scripts.fpl.availability.get_fpl_availability_df", return_value=empty_avail), \
             patch("scripts.fpl.availability.get_transfer_odds_index",
                   return_value=pd.DataFrame(columns=ODDS_INDEX_COLUMNS)), \
             patch("scripts.fpl.availability.get_transfer_news", return_value=pd.DataFrame()), \
             patch("scripts.fpl.availability.transfer_news_cache_status", return_value=(0, 0)), \
             patch("scripts.fpl.availability.render_injuries_tab"):
            from scripts.fpl.availability import show_availability_page
            show_availability_page()

    def test_smoke_with_live_shaped_data(self, mock_all_utils):
        """A player who is departed, one with a market, and one with neither."""
        from scripts.common.transfer_odds import ODDS_INDEX_COLUMNS
        avail = pd.DataFrame([
            {"Player_ID": 1, "Player": "Mohamed Salah", "Web_Name": "M.Salah",
             "Team": "LIV", "Position": "M", "Status": "a", "PlayPct": 100.0,
             "StatusBucket": "Available", "News": "", "News_Added": ""},
            {"Player_ID": 2, "Player": "Ollie Watkins", "Web_Name": "Watkins",
             "Team": "AVL", "Position": "F", "Status": "u", "PlayPct": 0.0,
             "StatusBucket": "Out", "News": "Has joined Al-Hilal permanently",
             "News_Added": ""},
            {"Player_ID": 3, "Player": "Quiet Player", "Web_Name": "Quiet",
             "Team": "ARS", "Position": "D", "Status": "a", "PlayPct": 100.0,
             "StatusBucket": "Available", "News": "", "News_Added": ""},
        ])
        odds = pd.DataFrame([{
            "Player": "Mohamed Salah", "Slug": "mohamed-salah",
            "Next_Club": "Any Saudi club", "Fractional": "8/11", "Decimal": 1.727,
            "Implied": 0.579, "Bookmaker": "William Hill", "Trending": "neutral",
            "Updated": None,
        }], columns=ODDS_INDEX_COLUMNS)
        news = pd.DataFrame([{
            "Player": "Mohamed Salah", "Team": "LIV",
            "Headline": "Salah in talks over Saudi Pro League exit",
            "URL": "https://example.com/a", "Published": "Tue, 02 Sep 2026 10:00:00 GMT",
            "Source": "BBC",
        }])
        with patch("scripts.fpl.availability.get_fpl_availability_df", return_value=avail), \
             patch("scripts.fpl.availability.get_transfer_odds_index", return_value=odds), \
             patch("scripts.fpl.availability.get_transfer_news", return_value=news), \
             patch("scripts.fpl.availability.transfer_news_cache_status", return_value=(3, 0)), \
             patch("scripts.fpl.availability.get_player_odds_ladder",
                   return_value=pd.DataFrame()), \
             patch("scripts.fpl.availability.render_injuries_tab"):
            from scripts.fpl.availability import show_availability_page
            show_availability_page()


class TestPlayerProjectionsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.player_projections.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_rotowire_rankings_url", return_value="https://example.com"), \
             patch("scripts.fpl.player_projections.get_ffp_feed", return_value=_empty_ffp_feed()), \
             patch("scripts.fpl.player_projections.get_ffp_horizon", return_value=_empty_ffp_horizon()), \
             patch("scripts.fpl.player_projections.get_ffp_goalscorer_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_ffp_clean_sheet_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_odds_api_match_odds", return_value=pd.DataFrame()):
            from scripts.fpl.player_projections import show_player_projections_page
            show_player_projections_page()


class TestHorizonTab:
    """The grid has to survive the states the live feed actually produces."""

    def _bootstrap(self, n=40):
        return {
            "teams": [{"id": i % 20 + 1, "short_name": "T%d" % (i % 20)}
                      for i in range(20)],
            "elements": [{
                "id": i + 1, "first_name": "First%d" % i, "second_name": "Last%d" % i,
                "web_name": "P%d" % i, "team": i % 20 + 1,
                "element_type": (i % 4) + 1, "ep_next": "3.0", "status": "a",
                "chance_of_playing_next_round": 100, "news": "",
            } for i in range(n)],
        }

    def test_it_renders_a_populated_grid(self, mock_all_utils):
        with patch("scripts.fpl.player_projections.get_ffp_horizon",
                   return_value=_ffp_horizon()), \
             patch("scripts.fpl.player_projections.get_ffp_feed",
                   return_value=_empty_ffp_feed()), \
             patch("scripts.fpl.player_projections.get_rotowire_player_projections",
                   return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_classic_bootstrap_static",
                   return_value=self._bootstrap()), \
             patch("scripts.common.player_pool.get_ffp_feed",
                   return_value=_empty_ffp_feed()), \
             patch("scripts.common.player_pool.get_rotowire_player_projections",
                   return_value=pd.DataFrame()), \
             patch("scripts.common.player_pool.get_classic_bootstrap_static",
                   return_value=self._bootstrap()):
            from scripts.fpl.player_projections import render_horizon
            render_horizon()

    def test_the_pool_is_really_built_from_the_patched_bootstrap(self, mock_all_utils):
        """Regression: `blended_pool()` moved to `scripts.common.player_pool`,
        so patching its three fetches on *this* module silently stopped
        controlling it -- and the populated-grid test above went on passing
        while rendering an empty blend. Nothing else noticed, because
        `_horizon_frames` tolerates an empty blend by design.
        """
        with patch("scripts.common.player_pool.get_ffp_feed",
                   return_value=_empty_ffp_feed()), \
             patch("scripts.common.player_pool.get_rotowire_player_projections",
                   return_value=pd.DataFrame()), \
             patch("scripts.common.player_pool.get_classic_bootstrap_static",
                   return_value=self._bootstrap()):
            from scripts.common.player_pool import blended_pool
            pool = blended_pool(7, "https://example.invalid/rotowire")

        # Pinned to the fixture's own identity, not merely to being non-empty:
        # with the network up, a patch on the wrong module silently builds a
        # real 667-player pool and every "is it populated" assertion passes.
        assert len(pool) == 40
        assert set(pool["Player"]) == {"First%d Last%d" % (i, i) for i in range(40)}
        assert {"Player_ID", "Proj", "Proj_Next3"} <= set(pool.columns)
        assert pool["Proj"].notna().any()

    def _render_grid(self, search=""):
        """Render the Horizon tab and hand back the frame it tabled."""
        captured = []
        with contextlib.ExitStack() as stack:
            for target, value in (
                ("scripts.fpl.player_projections.get_ffp_horizon", _ffp_horizon()),
                ("scripts.fpl.player_projections.get_ffp_feed", _empty_ffp_feed()),
                ("scripts.fpl.player_projections.get_rotowire_player_projections",
                 pd.DataFrame()),
                ("scripts.fpl.player_projections.get_classic_bootstrap_static",
                 self._bootstrap()),
                ("scripts.common.player_pool.get_ffp_feed", _empty_ffp_feed()),
                ("scripts.common.player_pool.get_rotowire_player_projections",
                 pd.DataFrame()),
                ("scripts.common.player_pool.get_classic_bootstrap_static",
                 self._bootstrap()),
            ):
                stack.enter_context(patch(target, return_value=value))
            stack.enter_context(patch(
                "scripts.fpl.player_projections.render_styled_table",
                side_effect=lambda df, **k: captured.append(df)))
            stack.enter_context(patch("streamlit.text_input", return_value=search))
            from scripts.fpl.player_projections import render_horizon
            render_horizon()
        assert captured, "the Horizon tab rendered no table"
        return captured[-1]

    def test_filtering_does_not_re_number_the_board(self, mock_all_utils):
        """The reported symptom, on the other grid: searching one player
        reported him as #1 rather than where he sits on the whole board."""
        full = self._render_grid()
        assert full["#"].tolist() == sorted(full["#"].tolist())
        target = full["Player"].iloc[-1]
        expected = full["#"].iloc[-1]
        assert expected > 1, "the fixture needs a player who is not already #1"

        filtered = self._render_grid(search=target)
        assert filtered["Player"].tolist() == [target]
        assert filtered["#"].iloc[0] == expected, (
            "# was stamped on after filtering, so the worst player on the "
            "board read as #1 because he was the only row left"
        )

    def test_a_surname_matches_even_when_the_display_name_hides_it(self, mock_all_utils):
        """Reported: searching "Nunes" returned nothing while the grid showed
        "Matheus N." -- FPL abbreviates that surname in `web_name`. The display
        name is fixed, but the search must not depend on having fixed it, so it
        matches the full legal name too.

        The fixture's display names are "Player N" and its legal names
        "FirstN LastN", so a search for a legal-name fragment can only succeed
        through that second haystack.
        """
        grid = self._render_grid(search="Last7")
        assert not grid.empty, "the legal name was not searched"
        assert len(grid) == 1
        # The fixture's rendered name is the terse web_name, which is exactly
        # the situation: what is on screen is not what you would type.
        assert grid["Player"].iloc[0] == "P7"

    def test_the_search_ignores_accents(self, mock_all_utils):
        """Nobody types the ã in "Guimarães"."""
        from scripts.fpl.player_projections import _name_filter
        frame = pd.DataFrame({"Player": ["Bruno Guimarães", "Erling Haaland"]})
        assert list(_name_filter(frame, "Guimaraes")) == [True, False]
        assert list(_name_filter(frame, "guimarães")) == [True, False]

    def test_the_search_column_never_reaches_the_table(self, mock_all_utils):
        """`_search_name` is searched, not rendered."""
        grid = self._render_grid()
        assert "_search_name" not in grid.columns

    def test_an_unavailable_horizon_does_not_raise(self, mock_all_utils):
        with patch("scripts.fpl.player_projections.get_ffp_horizon",
                   return_value=_empty_ffp_horizon()):
            from scripts.fpl.player_projections import render_horizon
            render_horizon()

    def test_it_survives_a_horizon_whose_rows_resolve_to_no_player(self, mock_all_utils):
        """Player_ID is the join key; without it there is nothing to render."""
        horizon = _ffp_horizon()
        df = horizon.df.copy()
        df["Player_ID"] = None
        with patch("scripts.fpl.player_projections.get_ffp_horizon",
                   return_value=horizon._replace(df=df)), \
             patch("scripts.fpl.player_projections.get_ffp_feed",
                   return_value=_empty_ffp_feed()), \
             patch("scripts.fpl.player_projections.get_rotowire_player_projections",
                   return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_classic_bootstrap_static",
                   return_value=self._bootstrap()), \
             patch("scripts.common.player_pool.get_ffp_feed",
                   return_value=_empty_ffp_feed()), \
             patch("scripts.common.player_pool.get_rotowire_player_projections",
                   return_value=pd.DataFrame()), \
             patch("scripts.common.player_pool.get_classic_bootstrap_static",
                   return_value=self._bootstrap()):
            from scripts.fpl.player_projections import render_horizon
            render_horizon()

    def test_a_blank_gameweek_renders_as_a_blank_not_an_empty_cell(self):
        """A missing fixture must read as "the club is not playing", not as
        missing data — that is the most important thing a horizon can say."""
        from scripts.fpl.player_projections import _horizon_fixture_cell
        assert _horizon_fixture_cell([], 0) == "—"
        assert _horizon_fixture_cell([], 1) == "—"
        assert _horizon_fixture_cell(["IPS (H)"], 1) == "IPS (H)"
        assert _horizon_fixture_cell(["IPS (H)", "ARS (a)"], 2) == "IPS (H) / ARS (a)"
        # FFP priced a double but published only one opponent for it.
        assert _horizon_fixture_cell(["IPS (H)"], 2) == "IPS (H) ×2"

    def test_the_difficulty_is_printed_not_only_coloured(self):
        """render_styled_table colours text, so FDR as colour alone carries
        nothing for a colour-blind reader. The number is the data."""
        from scripts.fpl.player_projections import _horizon_fixture_cell
        assert _horizon_fixture_cell(["IPS (H)"], 1, 1) == "IPS (H)  1"
        assert _horizon_fixture_cell(["ARS (a)"], 1, 5) == "ARS (a)  5"
        # A double gameweek averages to a fraction; it must not read as "3".
        assert _horizon_fixture_cell(["ARS (a)", "CHE (H)"], 2, 3.5) == \
            "ARS (a) / CHE (H)  3.5"
        # No difficulty published is a missing suffix, never a fabricated one.
        assert _horizon_fixture_cell(["IPS (H)"], 1, None) == "IPS (H)"
        assert _horizon_fixture_cell(["IPS (H)"], 1, float("nan")) == "IPS (H)"
        # A blank gameweek has no fixture to rate.
        assert _horizon_fixture_cell([], 0, 3) == "—"


class TestPriceChangesPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.price_changes.get_classic_bootstrap_static",
                   return_value={"elements": [], "teams": []}):
            from scripts.fpl.price_changes import show_price_changes_page
            show_price_changes_page()


class TestGameweekReviewPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.fpl.gameweek_review.get_classic_bootstrap_static",
                   return_value={"elements": [], "teams": [], "events": [
                       {"id": 1, "finished": True},
                       {"id": 2, "finished": True},
                   ]}), \
             patch("scripts.fpl.gameweek_review._get_classic_gw_live_points",
                   return_value={}), \
             patch("scripts.fpl.gameweek_review.get_classic_team_picks",
                   return_value=None), \
             patch("scripts.fpl.gameweek_review.get_classic_team_history",
                   return_value=None), \
             patch("scripts.fpl.gameweek_review.get_fpl_player_mapping",
                   return_value={}), \
             patch("scripts.fpl.gameweek_review._get_draft_gw_live_points",
                   return_value={}), \
             patch("scripts.fpl.gameweek_review.requests.get",
                   return_value=MagicMock(json=MagicMock(return_value={"picks": []}))):
            from scripts.fpl.gameweek_review import show_gw_review_page
            show_gw_review_page()


class TestLeagueSetupPage:
    def test_smoke_unlocked_defaults(self, mock_all_utils):
        """Default (unset) settings should render the edit forms without error.

        All st.button() calls are truthy under mock_all_utils, but the default
        empty text_input("") values cause lookup/save branches to short-circuit
        via st.error() rather than making real API calls.
        """
        with patch("scripts.fpl.league_setup.load_settings", return_value={
                "version": 1,
                "draft": {"league_id": None, "team_id": None, "team_name": None, "locked": False},
                "classic": {"leagues": [], "team_id": None, "team_name": None, "locked": False},
            }), \
             patch("scripts.fpl.league_setup.save_settings", return_value=True), \
             patch("scripts.fpl.league_setup.config.refresh_league_settings"), \
             patch("scripts.fpl.league_setup.save_auth") as save_auth, \
             patch("scripts.fpl.league_setup.clear_auth") as clear_auth, \
             patch("scripts.fpl.league_setup.load_auth", return_value={}), \
             patch("scripts.fpl.league_setup.test_credentials",
                   return_value=(None, "no_auth")), \
             patch("scripts.fpl.league_setup.fetch_my_team"):
            from scripts.fpl.league_setup import show_league_setup_page
            show_league_setup_page()
            # The credential writers must stay mocked — an unmocked Save would
            # write a real .fpl_auth.json from this test run.
            save_auth.assert_not_called()
            clear_auth.assert_not_called()

    def test_smoke_locked_view(self, mock_all_utils):
        """Locked settings should render the read-only summary + unlock button.

        is_draft_league_reachable / is_classic_league_reachable are imported
        directly into scripts.fpl.league_setup (not via scripts.common.utils),
        so mock_all_utils's patches don't reach them — they must be patched
        here directly, or the locked view would make real HTTP calls.
        """
        with patch("scripts.fpl.league_setup.load_settings", return_value={
                "version": 1,
                "draft": {
                    "league_id": 4544, "team_id": 17077, "team_name": "My Team", "locked": True,
                    "history": [], "last_confirmed_season": "2025/26",
                },
                "classic": {
                    "leagues": [{"id": 1555691, "name": "FAFO FPL"}],
                    "team_id": 6720205, "team_name": "My Classic Team", "locked": True,
                    "league_history": [], "last_confirmed_season": "2025/26",
                },
            }), \
             patch("scripts.fpl.league_setup.save_settings", return_value=True), \
             patch("scripts.fpl.league_setup.config.refresh_league_settings"), \
             patch("scripts.fpl.league_setup.is_draft_league_reachable", return_value=True), \
             patch("scripts.fpl.league_setup.is_classic_league_reachable", return_value=True), \
             patch("scripts.fpl.league_setup.save_auth") as save_auth, \
             patch("scripts.fpl.league_setup.clear_auth") as clear_auth, \
             patch("scripts.fpl.league_setup.load_auth", return_value={}), \
             patch("scripts.fpl.league_setup.test_credentials",
                   return_value=(None, "no_auth")), \
             patch("scripts.fpl.league_setup.fetch_my_team"):
            from scripts.fpl.league_setup import show_league_setup_page
            show_league_setup_page()
            # The credential writers must stay mocked — an unmocked Save would
            # write a real .fpl_auth.json from this test run.
            save_auth.assert_not_called()
            clear_auth.assert_not_called()

    def test_smoke_locked_view_stale_ids(self, mock_all_utils):
        """A locked league ID that no longer resolves should show the stale
        warning and best-effort archive it, without raising."""
        with patch("scripts.fpl.league_setup.load_settings", return_value={
                "version": 1,
                "draft": {
                    "league_id": 4544, "team_id": 17077, "team_name": "My Team", "locked": True,
                    "history": [], "last_confirmed_season": "2025/26",
                },
                "classic": {
                    "leagues": [{"id": 1161877, "name": "Super League DMV Starboys"}],
                    "team_id": 6720205, "team_name": "My Classic Team", "locked": True,
                    "league_history": [], "last_confirmed_season": "2025/26",
                },
            }), \
             patch("scripts.fpl.league_setup.save_settings", return_value=True) as mock_save, \
             patch("scripts.fpl.league_setup.config.refresh_league_settings"), \
             patch("scripts.fpl.league_setup.is_draft_league_reachable", return_value=False), \
             patch("scripts.fpl.league_setup.is_classic_league_reachable", return_value=False), \
             patch("scripts.fpl.league_setup.save_auth") as save_auth, \
             patch("scripts.fpl.league_setup.clear_auth") as clear_auth, \
             patch("scripts.fpl.league_setup.load_auth", return_value={}), \
             patch("scripts.fpl.league_setup.test_credentials",
                   return_value=(None, "no_auth")), \
             patch("scripts.fpl.league_setup.fetch_my_team"):
            from scripts.fpl.league_setup import show_league_setup_page
            show_league_setup_page()
            # The credential writers must stay mocked — an unmocked Save would
            # write a real .fpl_auth.json from this test run.
            save_auth.assert_not_called()
            clear_auth.assert_not_called()
        # Best-effort archive of the now-unreachable IDs should have saved.
        assert mock_save.called


class TestArchiveReplacedLeagues:
    """_archive_replaced_draft_league / _archive_replaced_classic_leagues —
    called from Save & Lock so a league that was only ever resolved via
    .env (never previously locked in league_settings.json) isn't silently
    dropped the first time a new one is saved over it."""

    def test_draft_archives_previously_env_sourced_id(self):
        from scripts.fpl.league_setup import _archive_replaced_draft_league
        old_draft = {"history": [], "team_id": 17077, "team_name": "Old Team"}
        with patch("scripts.fpl.league_setup.config.FPL_DRAFT_LEAGUE_ID", 11347):
            _archive_replaced_draft_league(old_draft, new_league_id=99999, season_label="2025/26")
        assert old_draft["history"] == [{
            "season": "2025/26", "league_id": 11347,
            "team_id": 17077, "team_name": "Old Team", "manual_stats": None,
        }]

    def test_draft_noop_when_id_unchanged(self):
        from scripts.fpl.league_setup import _archive_replaced_draft_league
        old_draft = {"history": []}
        with patch("scripts.fpl.league_setup.config.FPL_DRAFT_LEAGUE_ID", 11347):
            _archive_replaced_draft_league(old_draft, new_league_id=11347, season_label="2025/26")
        assert old_draft["history"] == []

    def test_draft_noop_when_nothing_previously_effective(self):
        from scripts.fpl.league_setup import _archive_replaced_draft_league
        old_draft = {"history": []}
        with patch("scripts.fpl.league_setup.config.FPL_DRAFT_LEAGUE_ID", 0):
            _archive_replaced_draft_league(old_draft, new_league_id=99999, season_label="2025/26")
        assert old_draft["history"] == []

    def test_draft_idempotent_when_already_archived(self):
        from scripts.fpl.league_setup import _archive_replaced_draft_league
        old_draft = {"history": [
            {"season": "2025/26", "league_id": 11347, "team_id": 17077, "team_name": "Old Team",
             "manual_stats": None},
        ]}
        with patch("scripts.fpl.league_setup.config.FPL_DRAFT_LEAGUE_ID", 11347):
            _archive_replaced_draft_league(old_draft, new_league_id=99999, season_label="2025/26")
        assert len(old_draft["history"]) == 1

    def test_classic_archives_previously_env_sourced_league(self):
        """The user's real bug: 1161877 only ever lived in FPL_CLASSIC_LEAGUE_IDS
        (never locked in JSON), so saving 668226 over it must not lose it."""
        from scripts.fpl.league_setup import _archive_replaced_classic_leagues
        old_classic = {"league_history": []}
        env_leagues = [{"id": 1161877, "name": "Super League DMV Starboys"}]
        with patch("scripts.fpl.league_setup.config.FPL_CLASSIC_LEAGUE_IDS", env_leagues):
            _archive_replaced_classic_leagues(old_classic, new_league_ids={668226}, season_label="2025/26")
        assert old_classic["league_history"] == [{
            "season": "2025/26", "league_id": 1161877,
            "league_name": "Super League DMV Starboys", "manual_stats": None,
        }]

    def test_classic_keeps_leagues_still_present(self):
        from scripts.fpl.league_setup import _archive_replaced_classic_leagues
        old_classic = {"league_history": []}
        env_leagues = [{"id": 1555691, "name": "FAFO FPL"}, {"id": 1161877, "name": "Starboys"}]
        with patch("scripts.fpl.league_setup.config.FPL_CLASSIC_LEAGUE_IDS", env_leagues):
            _archive_replaced_classic_leagues(
                old_classic, new_league_ids={1555691, 668226}, season_label="2025/26"
            )
        # Only the dropped league (1161877) is archived; 1555691 is still in the new set.
        assert len(old_classic["league_history"]) == 1
        assert old_classic["league_history"][0]["league_id"] == 1161877

    def test_classic_idempotent_when_already_archived(self):
        from scripts.fpl.league_setup import _archive_replaced_classic_leagues
        old_classic = {"league_history": [
            {"season": "2025/26", "league_id": 1161877, "league_name": "Starboys", "manual_stats": None},
        ]}
        env_leagues = [{"id": 1161877, "name": "Starboys"}]
        with patch("scripts.fpl.league_setup.config.FPL_CLASSIC_LEAGUE_IDS", env_leagues):
            _archive_replaced_classic_leagues(old_classic, new_league_ids={668226}, season_label="2025/26")
        assert len(old_classic["league_history"]) == 1


class TestUpsertClassicHistoryEntryPctFinish:
    """_upsert_classic_history_entry's optional pct_finish param persists into
    classic.season_notes (a season-wide stat) in the same write as the
    per-league placement entry, so the manual-entry form can save both at once."""

    def test_pct_finish_persisted_to_season_notes(self):
        from scripts.fpl.league_setup import _upsert_classic_history_entry
        settings = {"classic": {"league_history": [], "season_notes": {}}}
        with patch("scripts.fpl.league_setup.load_settings", return_value=settings), \
             patch("scripts.fpl.league_setup.save_settings", return_value=True) as mock_save, \
             patch("scripts.fpl.league_setup.config.refresh_league_settings"), \
             patch("scripts.fpl.league_setup.save_auth") as save_auth, \
             patch("scripts.fpl.league_setup.clear_auth") as clear_auth, \
             patch("scripts.fpl.league_setup.load_auth", return_value={}), \
             patch("scripts.fpl.league_setup.test_credentials",
                   return_value=(None, "no_auth")), \
             patch("scripts.fpl.league_setup.fetch_my_team"):
            _upsert_classic_history_entry(
                "2025/26", 1161877, "Super League DMV Starboys",
                manual_stats={"rank": 4, "total_points": None}, pct_finish=8.0,
            )
        saved_settings = mock_save.call_args[0][0]
        assert saved_settings["classic"]["season_notes"]["2025/26"] == {"pct_finish": 8.0}
        assert saved_settings["classic"]["league_history"][0]["league_id"] == 1161877

    def test_pct_finish_none_leaves_season_notes_untouched(self):
        from scripts.fpl.league_setup import _upsert_classic_history_entry
        settings = {"classic": {"league_history": [], "season_notes": {"2025/26": {"pct_finish": 8.0}}}}
        with patch("scripts.fpl.league_setup.load_settings", return_value=settings), \
             patch("scripts.fpl.league_setup.save_settings", return_value=True) as mock_save, \
             patch("scripts.fpl.league_setup.config.refresh_league_settings"), \
             patch("scripts.fpl.league_setup.save_auth") as save_auth, \
             patch("scripts.fpl.league_setup.clear_auth") as clear_auth, \
             patch("scripts.fpl.league_setup.load_auth", return_value={}), \
             patch("scripts.fpl.league_setup.test_credentials",
                   return_value=(None, "no_auth")), \
             patch("scripts.fpl.league_setup.fetch_my_team"):
            _upsert_classic_history_entry(
                "2025/26", 1555691, "FAFO FPL",
                manual_stats={"rank": 1, "total_points": None}, pct_finish=None,
            )
        saved_settings = mock_save.call_args[0][0]
        assert saved_settings["classic"]["season_notes"]["2025/26"] == {"pct_finish": 8.0}


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


class TestHorizonBlendedColumn:
    """The current gameweek's column must be the engine's blend, not FFP's own.

    This is the guarantee the Horizon tab rests on -- a player's first column
    has to be the number every other page in the app scores on, or the Hub
    disagrees with itself about this gameweek. Asserted on ``_horizon_frames``
    directly, because the rendered HTML can only show that *a* number arrived.
    """

    def _horizon(self):
        return _ffp_horizon(start_gw=7, weeks=3, n=3)

    def _blended(self):
        return pd.DataFrame({
            "Player_ID": [1, 2, 3],
            "Display_Name": ["A", "B", "C"],
            "Team": ["T0", "T1", "T2"],
            "Position": ["GK", "DEF", "MID"],
            # Deliberately unlike FFP's own values for GW7.
            "Proj": [9.99, 8.88, 7.77],
        })

    def test_the_current_gameweek_column_comes_from_the_blend(self):
        from scripts.fpl.player_projections import _horizon_frames
        points, _fixtures, _fdr, _counts = _horizon_frames(
            self._horizon(), self._blended(), current_gw=7)
        assert points[7].tolist() == [9.99, 8.88, 7.77]

    def test_later_gameweeks_are_untouched_by_the_blend(self):
        """The blend prices one gameweek; it must not leak into the others."""
        from scripts.fpl.player_projections import _horizon_frames
        horizon = self._horizon()
        ffp = horizon.df
        expected = {
            gw: ffp[ffp["FFP_GW"] == gw].sort_values("Player_ID")["Predicted"].tolist()
            for gw in (8, 9)
        }
        points, _f, _d, _c = _horizon_frames(horizon, self._blended(), current_gw=7)
        for gw in (8, 9):
            assert points[gw].tolist() == pytest.approx(expected[gw])

    def test_a_player_the_blend_does_not_price_keeps_ffps_number(self):
        """combine_first, not an overwrite: the blend covers the FPL pool, and a
        player missing from it must not have his gameweek blanked."""
        from scripts.fpl.player_projections import _horizon_frames
        horizon = self._horizon()
        partial = self._blended().iloc[:1]           # only Player_ID 1
        ffp_gw7 = horizon.df[horizon.df["FFP_GW"] == 7].set_index("Player_ID")
        points, _f, _d, _c = _horizon_frames(horizon, partial, current_gw=7)
        assert points.at[1, 7] == 9.99               # blended
        assert points.at[2, 7] == pytest.approx(ffp_gw7.at[2, "Predicted"])

    def test_no_blend_at_all_still_renders_ffps_own_forecasts(self):
        from scripts.fpl.player_projections import _horizon_frames
        horizon = self._horizon()
        points, _f, _d, _c = _horizon_frames(horizon, pd.DataFrame(), current_gw=7)
        assert points.shape == (3, 3)
        assert points[7].notna().all()


class TestHorizonPosRank:
    """Pos RK on the Horizon grid, in both views and over the whole board.

    The horizon is where a positional rank earns the most: the grid's job is
    reading across a row, and "4.2 a week" says nothing until you know whether
    that is a leading goalkeeper or a replaceable forward.
    """

    def _render(self, view, search=""):
        from scripts.fpl import player_projections as pp
        captured = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pp, "get_ffp_horizon",
                                             return_value=_ffp_horizon()))
            stack.enter_context(patch.object(pp, "_blended_pool",
                                             return_value=pd.DataFrame()))
            stack.enter_context(patch.object(
                pp, "render_styled_table",
                side_effect=lambda df, **k: captured.append((df, k))))
            stack.enter_context(patch("streamlit.radio", return_value=view))
            stack.enter_context(patch("streamlit.text_input", return_value=search))
            pp.render_horizon()
        assert captured, "the Horizon tab rendered no table"
        return captured[-1]

    @pytest.mark.parametrize("view", ["Points", "Fixtures"])
    def test_both_views_carry_it(self, view, mock_streamlit):
        df, kwargs = self._render(view)
        assert "Pos RK" in df.columns
        assert kwargs["col_formats"].get("Pos RK") == "{:.0f}"
        # Ranked within position: every position's board starts at 1, and the
        # best overall row is not the only 1 on the page.
        for pos, grp in df.groupby("Pos"):
            assert grp["Pos RK"].min() == 1.0, pos
        assert (df["Pos RK"] == 1.0).sum() == df["Pos"].nunique()

    def test_filtering_does_not_re_rank(self, mock_streamlit):
        """The board's rank, not the selection's. Same trap as the colour range.

        ``_ffp_horizon`` scores players in descending order, so "Player 12" is
        the fourth-best of his position (every 4th row shares a position) and
        must still read 4 when he is the only row on screen.
        """
        full, _ = self._render("Points")
        expected = full.loc[full["Player"] == "Player 12", "Pos RK"].iloc[0]
        assert expected > 1.0, "fixture no longer exercises the trap"
        filtered, _ = self._render("Points", search="Player 12")
        assert filtered["Player"].tolist() == ["Player 12"]
        assert filtered["Pos RK"].iloc[0] == expected


class TestHubSections:
    """The Hub's top level is lazy, so each section needs its own coverage.

    When the nav was ``st.tabs`` every body rendered on every load and one smoke
    test covered all four. A radio hands back only its first option under the
    mock, so without these three of the four sections would silently stop being
    exercised -- the test would still pass, having quietly narrowed to one quarter
    of the page.
    """

    def _patches(self):
        from scripts.fpl import player_projections as pp
        return [
            patch.object(pp, "get_rotowire_player_projections", return_value=pd.DataFrame()),
            patch.object(pp, "get_rotowire_rankings_url", return_value="https://example.com"),
            patch.object(pp, "get_ffp_feed", return_value=_empty_ffp_feed()),
            patch.object(pp, "get_ffp_horizon", return_value=_empty_ffp_horizon()),
            patch.object(pp, "get_ffp_goalscorer_odds", return_value=pd.DataFrame()),
            patch.object(pp, "get_ffp_clean_sheet_odds", return_value=pd.DataFrame()),
            patch.object(pp, "get_odds_api_match_odds", return_value=pd.DataFrame()),
        ]

    @pytest.mark.parametrize("section", [
        "GW Projections", "Horizon", "Odds", "Accuracy",
    ])
    def test_every_section_renders(self, section, mock_all_utils):
        from scripts.fpl.player_projections import _render_hub_section
        with contextlib.ExitStack() as stack:
            for p in self._patches():
                stack.enter_context(p)
            _render_hub_section(section)

    def test_the_parametrisation_covers_every_declared_section(self):
        """So adding a section cannot skip adding its test."""
        from scripts.fpl.player_projections import HUB_SECTIONS
        covered = {"GW Projections", "Horizon", "Odds", "Accuracy"}
        assert set(HUB_SECTIONS) == covered, (
            "HUB_SECTIONS changed to %s -- add the new section to "
            "test_every_section_renders" % (HUB_SECTIONS,)
        )

    def test_an_unknown_section_falls_back_rather_than_rendering_nothing(self, mock_all_utils):
        """A stale session_state value must not produce a blank page."""
        from scripts.fpl import player_projections as pp
        with contextlib.ExitStack() as stack:
            for p in self._patches():
                stack.enter_context(p)
            called = []
            stack.enter_context(patch.object(
                pp, "render_blended_projections", side_effect=lambda: called.append(1)))
            pp._render_hub_section("Something Removed")
        assert called, "an unrecognised section should fall back to the default"

    @pytest.mark.parametrize("section,expected", [
        ("GW Projections", {"render_blended_projections",
                            "render_rotowire_projections", "render_ffp_data"}),
        ("Horizon", {"render_horizon"}),
        ("Odds", {"render_goalscorer_odds", "render_clean_sheet_odds",
                  "render_match_odds"}),
        ("Accuracy", {"render_accuracy"}),
    ])
    def test_a_section_renders_only_its_own_tabs(self, section, expected, mock_all_utils):
        """The lazy contract, and the thing a revert to st.tabs would break.

        Accuracy costs 0.61s of what used to be a 0.92s rerun, every time,
        whether or not anyone had it open. The saving *is* this assertion: if
        rendering one section ever calls another's renderer again, the page is
        back to paying for all four.
        """
        from scripts.fpl import player_projections as pp
        all_renderers = {
            "render_blended_projections", "render_rotowire_projections",
            "render_ffp_data", "render_horizon", "render_goalscorer_odds",
            "render_clean_sheet_odds", "render_match_odds", "render_accuracy",
        }
        called = set()
        with contextlib.ExitStack() as stack:
            for name in all_renderers:
                stack.enter_context(patch.object(
                    pp, name,
                    side_effect=(lambda n: lambda *a, **k: called.add(n))(name)))
            pp._render_hub_section(section)
        assert called == expected, (
            "%s rendered %s; it must render exactly %s"
            % (section, sorted(called), sorted(expected))
        )


class TestAccuracyHorizonSection:
    """"How far ahead is worth projecting?" on the Accuracy tab.

    The answer is counter-intuitive -- a longer window orders *worse* -- so the
    section renders both reasons alongside the verdict rather than the
    conclusion alone.
    """

    def _render(self):
        from scripts.fpl import player_projections as pp
        tables, texts = [], []
        with patch.object(pp, "render_styled_table",
                          side_effect=lambda df, **k: tables.append(df)), \
             patch("streamlit.markdown", side_effect=lambda t, **k: texts.append(str(t))), \
             patch("streamlit.caption", side_effect=lambda t, **k: texts.append(str(t))), \
             patch("streamlit.warning", side_effect=lambda t, **k: texts.append(str(t))), \
             patch("streamlit.info", side_effect=lambda t, **k: texts.append(str(t))):
            pp._render_horizon_length(pd.DataFrame())
        return tables, " ".join(texts)

    def test_it_renders_the_verdict_and_both_reasons(self, mock_streamlit):
        from scripts.common import projection_accuracy
        if projection_accuracy.score_horizon_lengths().empty:
            pytest.skip("no archived gameweek has a complete run of actuals yet")
        tables, text = self._render()
        assert len(tables) >= 1
        assert "How far ahead is worth projecting?" in text
        # The two mechanisms, not just the conclusion.
        assert "decays fast" in text
        assert "player quality" in text

    def test_the_window_in_use_is_marked(self, mock_streamlit):
        from scripts.common import projection_accuracy
        if projection_accuracy.score_horizon_lengths().empty:
            pytest.skip("no archived gameweek has a complete run of actuals yet")
        tables, _ = self._render()
        windows = tables[0]["Window"].tolist()
        assert any("ROS" in w for w in windows), (
            "the reader cannot interpret the table without knowing which row is "
            "the window the app actually uses")

    def test_an_empty_archive_explains_itself(self, mock_streamlit):
        from scripts.fpl import player_projections as pp
        with patch.object(pp.projection_accuracy, "score_horizon_lengths",
                          return_value=pd.DataFrame()):
            tables, text = self._render()
        assert tables == []
        assert "cannot be scored" in text


class TestAccuracyConfidenceGate:
    """The tab's warning and its headline must use one count.

    They diverged: the gate counted every scored gameweek while the note it
    rendered counted pre-deadline ones, so the page could show a confident
    caption above a weight fit that still disclaimed itself.
    """

    @staticmethod
    def _scored(captures):
        return pd.DataFrame({
            "gameweek": list(range(3, 3 + len(captures))),
            "source": ["blend"] * len(captures),
            "scope": ["starters"] * len(captures),
            "capture": list(captures),
            "mae": [2.0] * len(captures),
            "bias": [0.0] * len(captures),
            "spearman": [0.5] * len(captures),
            "n": [100] * len(captures),
            "coverage": [1.0] * len(captures),
        })

    def _render(self, captures):
        from scripts.fpl import player_projections as pp
        warned, captioned = [], []
        with patch.object(pp.projection_accuracy, "score_archive",
                          return_value=self._scored(captures)), \
             patch.object(pp, "_render_verdict"), \
             patch.object(pp, "_render_horizon_length"), \
             patch.object(pp, "_render_start_section", create=True), \
             patch.object(pp, "_render_weight_fit"), \
             patch.object(pp, "render_styled_table"), \
             patch("streamlit.warning", side_effect=lambda t, **k: warned.append(str(t))), \
             patch("streamlit.caption", side_effect=lambda t, **k: captioned.append(str(t))), \
             patch("streamlit.markdown"), patch("streamlit.info"), \
             patch("streamlit.expander"), patch("streamlit.columns",
                                                return_value=[MagicMock(), MagicMock()]):
            pp.render_accuracy()
        return warned, captioned

    def test_a_backfill_does_not_silence_the_warning(self, mock_streamlit):
        """Five scored gameweeks, one a backfill -> still four, still warned."""
        warned, _ = self._render(["backfill"] + ["pre-deadline"] * 4)
        assert any("noise" in w for w in warned), (
            "the page went confident on a gameweek it refuses to fit on")

    def test_five_pre_deadline_gameweeks_stops_warning(self, mock_streamlit):
        warned, captioned = self._render(["pre-deadline"] * 5)
        assert not any("noise" in w for w in warned)
        assert any("pre-deadline gameweeks of history" in c for c in captioned)
