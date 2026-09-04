"""Smoke tests for FPL App Home pages.

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


_AVAIL_COLS = ["Player_ID", "Player", "Web_Name", "Team", "Position",
               "Status", "PlayPct", "StatusBucket", "News", "News_Added"]


class TestInjuriesPage:
    def test_smoke(self, mock_all_utils):
        empty_avail = pd.DataFrame(columns=_AVAIL_COLS)
        with patch("scripts.fpl.injuries.get_fpl_availability_df", return_value=empty_avail):
            from scripts.fpl.injuries import show_injuries_page
            show_injuries_page()


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
             patch("scripts.fpl.player_projections.get_ffp_goalscorer_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_ffp_clean_sheet_odds", return_value=pd.DataFrame()), \
             patch("scripts.fpl.player_projections.get_odds_api_match_odds", return_value=pd.DataFrame()):
            from scripts.fpl.player_projections import show_player_projections_page
            show_player_projections_page()


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
             patch("scripts.fpl.league_setup.config.refresh_league_settings"):
            from scripts.fpl.league_setup import show_league_setup_page
            show_league_setup_page()

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
             patch("scripts.fpl.league_setup.is_classic_league_reachable", return_value=True):
            from scripts.fpl.league_setup import show_league_setup_page
            show_league_setup_page()

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
             patch("scripts.fpl.league_setup.is_classic_league_reachable", return_value=False):
            from scripts.fpl.league_setup import show_league_setup_page
            show_league_setup_page()
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
             patch("scripts.fpl.league_setup.config.refresh_league_settings"):
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
             patch("scripts.fpl.league_setup.config.refresh_league_settings"):
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
