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


class TestGwList:
    """The Accuracy tab tells the reader which gameweeks it holds, so the empty
    state is informative rather than just blank."""

    def test_reads_as_a_sentence(self):
        from scripts.fpl.player_projections import _gw_list
        assert _gw_list([]) == "none"
        assert _gw_list([3]) == "GW3"
        assert _gw_list([1, 2]) == "GW1 and GW2"
        assert _gw_list([1, 2, 4]) == "GW1, GW2 and GW4"


class TestRenderAccuracyEmptyState:
    """With no scoreable gameweek the tab must explain *why*, not render blank.

    This is the state every user sees until a gameweek completes after snapshots
    began, so it is the most-seen state the page has.
    """

    def test_explains_which_halves_are_missing(self, monkeypatch):
        import pandas as pd
        from unittest.mock import MagicMock, patch
        import scripts.fpl.player_projections as pp

        monkeypatch.setattr(pp.projection_accuracy, "score_archive",
                            lambda **k: pd.DataFrame())
        monkeypatch.setattr(pp.projection_archive, "list_pre", lambda: [4])
        monkeypatch.setattr(pp.projection_archive, "list_actuals", lambda: [1, 2])

        said = []
        with patch.object(pp.st, "info", lambda t, *a, **k: said.append(t)), \
             patch.object(pp.st, "caption", lambda t, *a, **k: said.append(t)):
            pp.render_accuracy()

        blob = " ".join(said)
        assert "No gameweek yet" in blob
        assert "GW4" in blob and "GW1 and GW2" in blob
