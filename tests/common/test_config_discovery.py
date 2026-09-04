"""Tests for config._discover_rotowire_article().

Regression coverage for the GW1 bug: discovery selected Rotowire's
"Best FPL Picks for Gameweeks 1-5" article, whose Points column is a cumulative
5-gameweek total, because the correct single-GW article's slug shape
(fpl-gameweek-1-...-rankings-gw1-<id>) matched none of the discovery patterns.
Every ROTOWIRE_URL consumer was ~5x inflated as a result.
"""

from unittest.mock import MagicMock, patch

import pytest

import config

# The four real anchors present on Rotowire's FPL rankings index in 2026-08.
LIVE_INDEX_HTML = """
<html><body>
<a href="/soccer/article/fantrax-sleeper-premier-league-player-rankings-gameweek-1-gw1-127529">Fantrax GW1</a>
<a href="/soccer/article/fpl-gameweek-1-best-players-captain-picks-2026-27-rankings-gw1-127487">FPL GW1</a>
<a href="/soccer/article/best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238">GW1-5 picks</a>
<a href="/soccer/article/fantasy-premier-league-fpl-rankings-top-400-for-2026-27-season-124261">Season top 400</a>
<a href="/soccer/article/fpl-gw38-fantasy-premier-league-player-rankings-gameweek-38-115088">FPL GW38 (last season)</a>
</body></html>
"""

RANGE_ONLY_INDEX_HTML = """
<html><body>
<a href="/soccer/article/best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238">GW1-5 picks</a>
</body></html>
"""


def _mock_response(html):
    resp = MagicMock()
    resp.content = html.encode("utf-8")
    resp.raise_for_status = MagicMock()
    return resp


@pytest.fixture(autouse=True)
def _unpin_rotowire_url(monkeypatch):
    """conftest pins ROTOWIRE_URL in the env, which short-circuits discovery."""
    monkeypatch.delenv("ROTOWIRE_URL", raising=False)
    # The season-staleness filter compares against the current PL season string.
    monkeypatch.setattr(config, "current_pl_season_str", lambda: "2026-27")


def _discover(html, gw):
    with patch("requests.get", return_value=_mock_response(html)):
        return config._discover_rotowire_article(gw)


class TestRotowireArticleDiscovery:
    def test_gw1_prefers_single_gw_article_over_range_article(self):
        """The bug: the GW1-5 cumulative article was chosen for GW1."""
        url = _discover(LIVE_INDEX_HTML, 1)
        assert url.endswith("fpl-gameweek-1-best-players-captain-picks-2026-27-rankings-gw1-127487")
        assert "best-fpl-picks-for-gameweeks" not in url

    def test_fantrax_article_is_never_selected(self):
        """Fantrax is a different scoring system -- its GW1 article must not match."""
        assert "fantrax" not in _discover(LIVE_INDEX_HTML, 1)

    def test_range_article_is_never_selected_even_as_a_last_resort(self):
        """"Best picks for gameweeks X-Y" is not a projection source at any
        priority: its column is an adjusted value total over the range, not
        points for a gameweek. No projections beats invented ones -- the app
        already surfaces a clear "projections unavailable" warning."""
        assert _discover(RANGE_ONLY_INDEX_HTML, 3) == ""

    def test_range_article_still_sets_the_current_season_floor(self):
        """It stays parsed for its article id, which is how stale prior-season
        articles get filtered out before the season's own articles exist."""
        html = RANGE_ONLY_INDEX_HTML.replace(
            "</body>",
            '<a href="/soccer/article/fpl-gw38-fantasy-premier-league-player-rankings-gameweek-38-115088">old</a></body>',
        )
        # The GW38 article predates the 2026-27 floor (126238), so it is stale.
        assert _discover(html, 38) == ""

    def test_prior_season_article_is_not_used_as_a_nearest_gw_match(self):
        """GW38 from last season must not win GW2 on 'closest gameweek'."""
        url = _discover(LIVE_INDEX_HTML, 2)
        assert "115088" not in url

    def test_no_matching_articles_returns_empty_string(self):
        assert _discover("<html><body><a href='/soccer/news/whatever'>x</a></body></html>", 1) == ""


class TestLeagueIdEnvFallback:
    """The env fallback used when league_settings.json is absent or unlocked.

    `.env.example` ships FPL_DRAFT_LEAGUE_ID= and friends with empty values, and
    the documented setup is `cp .env.example .env`. So "key present, value
    empty" is the state a brand-new install is actually in — it must resolve to
    0, not raise.
    """

    @pytest.fixture(autouse=True)
    def _unlocked(self, monkeypatch):
        monkeypatch.setattr(config, "_get_league_settings",
                            lambda: {"draft": {}, "classic": {}})

    RESOLVERS = [
        ("FPL_DRAFT_LEAGUE_ID", "_resolve_draft_league_id"),
        ("FPL_DRAFT_TEAM_ID", "_resolve_draft_team_id"),
        ("FPL_CLASSIC_TEAM_ID", "_resolve_classic_team_id"),
    ]

    @pytest.mark.parametrize("env_name,resolver", RESOLVERS)
    def test_blank_value_resolves_to_zero(self, monkeypatch, env_name, resolver):
        """int(os.getenv(NAME, "0")) looks safe and is not: the default only
        fires when the key is absent, so a bare `NAME=` reaches int("")."""
        monkeypatch.setenv(env_name, "")
        assert getattr(config, resolver)() == 0

    @pytest.mark.parametrize("env_name,resolver", RESOLVERS)
    def test_absent_value_resolves_to_zero(self, monkeypatch, env_name, resolver):
        monkeypatch.delenv(env_name, raising=False)
        assert getattr(config, resolver)() == 0

    @pytest.mark.parametrize("env_name,resolver", RESOLVERS)
    def test_set_value_is_used(self, monkeypatch, env_name, resolver):
        monkeypatch.setenv(env_name, "4544")
        assert getattr(config, resolver)() == 4544

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_classic_league_ids_resolve_to_empty_list(self, monkeypatch, value):
        monkeypatch.setenv("FPL_CLASSIC_LEAGUE_IDS", value)
        assert config._resolve_classic_league_ids() == []

    def test_locked_settings_win_over_env(self, monkeypatch):
        """The reason the stale IDs in .env went unnoticed for a season."""
        monkeypatch.setattr(config, "_get_league_settings", lambda: {
            "draft": {}, "classic": {"locked": True, "team_id": 4474334}})
        monkeypatch.setenv("FPL_CLASSIC_TEAM_ID", "6720205")
        assert config._resolve_classic_team_id() == 4474334
