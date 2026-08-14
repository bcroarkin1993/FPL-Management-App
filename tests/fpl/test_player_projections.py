"""Tests for scripts/fpl/player_projections.py's pure helper functions."""

from scripts.fpl.player_projections import is_rotowire_url_stale


class TestIsRotowireUrlStale:
    def test_preview_range_covers_current_gw(self):
        url = "https://www.rotowire.com/soccer/article/best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238"
        assert is_rotowire_url_stale(url, 1) is False
        assert is_rotowire_url_stale(url, 5) is False

    def test_preview_range_does_not_cover_current_gw(self):
        url = "https://www.rotowire.com/soccer/article/best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238"
        assert is_rotowire_url_stale(url, 6) is True

    def test_single_gw_article_matches(self):
        url = "https://www.rotowire.com/soccer/article/fantasy-premier-league-player-rankings-gameweek-26-fpl-gw26-arsenal-104779"
        assert is_rotowire_url_stale(url, 26) is False

    def test_single_gw_article_mismatch(self):
        url = "https://www.rotowire.com/soccer/article/fantasy-premier-league-player-rankings-gameweek-26-fpl-gw26-arsenal-104779"
        assert is_rotowire_url_stale(url, 27) is True

    def test_season_rankings_url_has_no_gw_marker(self):
        """A season-long rankings article (no gameweek in the URL) can't be
        judged stale by this check — that's handled elsewhere."""
        url = "https://www.rotowire.com/soccer/article/fantasy-premier-league-fpl-rankings-top-400-for-2026-27-season-124261"
        assert is_rotowire_url_stale(url, 1) is False
