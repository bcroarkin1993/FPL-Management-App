"""Caching and concurrency behaviour for transfer news.

One request per player makes fetching the slowest thing the draft board does, so
these pin the three properties that keep it usable: scans are incremental, a
player with no news is not refetched forever, and a cached read never goes to the
network.
"""

import threading
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.common.transfer_feeds import NEWS_COLUMNS, build_query_url, fetch_transfer_news_batch


def _news_row(player, headline="Player linked with Al Hilal move"):
    return {"Player": player, "Team": "AVL", "Headline": headline,
            "URL": "http://example.com", "Published": "Fri, 28 Aug 2026 10:00:00 GMT",
            "Source": "Sky Sports"}


@pytest.fixture
def sqlite_cache(tmp_path):
    """Point the SQLite cache at a temp DB so tests never touch the real one."""
    import scripts.common.cache as cache_mod

    conn = cache_mod._init_connection(str(tmp_path / "test_cache.db"))
    original = cache_mod._connection
    cache_mod._connection = conn
    yield conn
    # start_transfer_news_prefetch() starts a real daemon thread that writes to
    # this very connection. Closing it while that thread is mid-write is a
    # use-after-close inside SQLite's C layer: the suite died with "Fatal Python
    # error: Bus error" / "Segmentation fault" roughly one run in three, in
    # whichever test happened to be running when the thread came back. Join it
    # first -- and here rather than in the one test that starts it, so any future
    # test that triggers a prefetch is covered too.
    _join_prefetch_threads()
    cache_mod._connection = original
    conn.close()


#: Every prefetch worker `scraping` starts is named "<something>-prefetch", and
#: the join below matches on that suffix rather than on one thread's name.
#: Matching "transfer-news-prefetch" exactly missed "club-news-prefetch", which
#: writes to the same connection through the same `_read_cached_club_news` --
#: so the club worker outlived the fixture, read a connection the teardown had
#: closed, and segfaulted inside SQLite's C layer in whichever test was running
#: when it came back. On Python 3.9 that lost the race often enough to look
#: clean; on 3.12 the suite died about three runs in five.
_PREFETCH_THREAD_SUFFIX = "-prefetch"


def _join_prefetch_threads(timeout: float = 10.0):
    for thread in threading.enumerate():
        if thread.name.endswith(_PREFETCH_THREAD_SUFFIX):
            thread.join(timeout=timeout)
            assert not thread.is_alive(), (
                "%s thread outlived its cache connection" % thread.name)


def test_every_prefetch_worker_is_named_so_the_join_finds_it():
    """The fixture can only join threads whose names it can recognise.

    A worker named off-pattern is not a failing test: it is a use-after-close
    on a connection the teardown has already shut, surfacing as a segfault
    somewhere else entirely. So the naming convention is asserted at the
    source rather than trusted.
    """
    import ast
    import pathlib as _pathlib

    src = _pathlib.Path("scripts/common/scraping.py").read_text()
    names = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        attr = getattr(func, "attr", None)
        if attr != "Thread":
            continue
        kw = {k.arg: k.value for k in node.keywords}
        name = kw.get("name")
        assert isinstance(name, ast.Constant), (
            "a background thread in scraping.py is started without a literal "
            "name, so the test fixture cannot join it")
        names.append(name.value)

    assert names, "no background threads found -- has scraping.py moved?"
    off_pattern = [n for n in names if not n.endswith(_PREFETCH_THREAD_SUFFIX)]
    assert not off_pattern, (
        "thread(s) %s do not end in %r, so _join_prefetch_threads will not "
        "join them" % (off_pattern, _PREFETCH_THREAD_SUFFIX))


class TestBatchFetching:
    def test_fetches_every_player(self):
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            out = fetch_transfer_news_batch([("A", "X"), ("B", "Y"), ("C", "Z")])
        assert set(out["Player"]) == {"A", "B", "C"}

    def test_deduplicates_players(self):
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            fetch_transfer_news_batch([("A", "X"), ("A", "X"), ("B", "Y")])
        assert m.call_count == 2

    def test_progress_is_reported_to_completion(self):
        seen = []
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            fetch_transfer_news_batch(
                [("A", "X"), ("B", "Y")], progress=lambda d, t, p: seen.append((d, t))
            )
        assert seen[0][0] == 0 and seen[-1] == (2, 2)

    def test_on_result_fires_per_player(self):
        """Results are persisted as they land, so an interrupted scan keeps its work."""
        got = {}
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            fetch_transfer_news_batch(
                [("A", "X"), ("B", "Y")], on_result=lambda p, r: got.__setitem__(p, r)
            )
        assert set(got) == {"A", "B"}

    def test_one_failure_does_not_sink_the_batch(self):
        def flaky(name, team, timeout=15):
            if name == "B":
                raise RuntimeError("boom")
            return pd.DataFrame([_news_row(name)])

        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news", side_effect=flaky):
            out = fetch_transfer_news_batch([("A", "X"), ("B", "Y"), ("C", "Z")])
        assert set(out["Player"]) == {"A", "C"}

    def test_batch_is_capped(self):
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame(columns=NEWS_COLUMNS)
            fetch_transfer_news_batch([(str(i), "X") for i in range(50)], max_players=10)
        assert m.call_count == 10

    def test_empty_input_makes_no_requests(self):
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            out = fetch_transfer_news_batch([])
        assert out.empty and m.call_count == 0


class TestQueryBuilding:
    def test_multi_token_name_is_quoted_alone(self):
        url = build_query_url("Ollie Watkins", "Aston Villa")
        assert "%22Ollie+Watkins%22" in url
        assert "Aston" not in url

    def test_mononym_gets_the_club_for_disambiguation(self):
        """Quoting a single word is not a search — 'Gabriel' matches every Gabriel."""
        url = build_query_url("Gabriel", "Arsenal")
        assert "Arsenal" in url


class TestPerPlayerCache:
    def test_second_call_makes_no_requests(self, sqlite_cache):
        from scripts.common.scraping import get_transfer_news

        pairs = (("A", "X"), ("B", "Y"))
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            first = get_transfer_news(pairs)
            assert m.call_count == 2
            second = get_transfer_news(pairs)
            assert m.call_count == 2, "cached players must not be refetched"
        assert len(second) == len(first)

    def test_widening_the_scan_only_fetches_the_new_players(self, sqlite_cache):
        """Raising scan depth used to invalidate a whole-batch key and refetch
        everything; the cache is keyed per player so it is incremental."""
        from scripts.common.scraping import get_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            get_transfer_news((("A", "X"), ("B", "Y")))
            assert m.call_count == 2
            get_transfer_news((("A", "X"), ("B", "Y"), ("C", "Z")))
            assert m.call_count == 3

    def test_player_with_no_news_is_not_refetched(self, sqlite_cache):
        """An empty result is a cache *hit*, not a miss — otherwise the quiet
        majority of the board is refetched on every single scan."""
        from scripts.common.scraping import get_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame(columns=NEWS_COLUMNS)
            get_transfer_news((("A", "X"),))
            get_transfer_news((("A", "X"),))
            assert m.call_count == 1

    def test_cached_only_never_hits_the_network(self, sqlite_cache):
        from scripts.common.scraping import get_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            out = get_transfer_news((("A", "X"),), cached_only=True)
            assert m.call_count == 0
        assert out.empty

    def test_cached_only_returns_previously_scanned_results(self, sqlite_cache):
        """The regression that blanked the board: a cached_only read after a scan
        must return the scan's results, not an empty frame."""
        from scripts.common.scraping import get_transfer_news

        pairs = (("A", "X"),)
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            get_transfer_news(pairs, force_refresh=True)
            later = get_transfer_news(pairs, cached_only=True)
            assert m.call_count == 1
        assert not later.empty, "risk data must survive a rerun (search, filter, toggle)"

    def test_force_refresh_ignores_the_cache(self, sqlite_cache):
        from scripts.common.scraping import get_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            get_transfer_news((("A", "X"),))
            get_transfer_news((("A", "X"),), force_refresh=True)
            assert m.call_count == 2

    def test_cache_status_counts_hits_and_misses(self, sqlite_cache):
        from scripts.common.scraping import get_transfer_news, transfer_news_cache_status

        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            get_transfer_news((("A", "X"),))
        assert transfer_news_cache_status((("A", "X"), ("B", "Y"))) == (1, 1)


class TestPrefetch:
    def test_prefetch_runs_once_per_label(self, sqlite_cache):
        import scripts.common.scraping as scraping

        scraping._PREFETCH_STARTED.discard("unit-test")
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as m:
            m.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            assert scraping.start_transfer_news_prefetch((("A", "X"),), label="unit-test")
            assert not scraping.start_transfer_news_prefetch((("A", "X"),), label="unit-test")
            # Inside the patch, deliberately: the worker runs asynchronously, so
            # leaving the block first lets it reach the real Google News feed --
            # the test then measures the network instead of the mock, and the
            # fixture's backstop join waits on an actual HTTP round trip.
            _join_prefetch_threads()
        scraping._PREFETCH_STARTED.discard("unit-test")

    def test_prefetch_with_no_players_does_nothing(self):
        from scripts.common.scraping import start_transfer_news_prefetch

        assert not start_transfer_news_prefetch([], label="unit-test-empty")


def _club_row(club, headline="Aston Villa complete signing of striker"):
    return {"Club": club, "Headline": headline, "URL": "http://example.com",
            "Published": "Fri, 28 Aug 2026 10:00:00 GMT", "Source": "BBC"}


class TestPerClubCache:
    """Signings are queried per club, not per player: the interesting arrivals
    are not in the FPL pool yet, so there is no name to query with."""

    def test_second_call_makes_no_requests(self, sqlite_cache):
        from scripts.common.scraping import get_club_transfer_news

        clubs = ["Aston Villa", "Arsenal"]
        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            m.side_effect = lambda c, timeout=15: pd.DataFrame([_club_row(c)])
            first = get_club_transfer_news(clubs)
            assert m.call_count == 2
            second = get_club_transfer_news(clubs)
            assert m.call_count == 2, "cached clubs must not be refetched"
        assert len(second) == len(first)

    def test_quiet_club_is_not_refetched(self, sqlite_cache):
        """Same empty-list-is-a-hit rule as the player cache."""
        from scripts.common.scraping import get_club_transfer_news
        from scripts.common.transfer_feeds import CLUB_NEWS_COLUMNS

        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            m.side_effect = lambda c, timeout=15: pd.DataFrame(columns=CLUB_NEWS_COLUMNS)
            get_club_transfer_news(["Arsenal"])
            get_club_transfer_news(["Arsenal"])
            assert m.call_count == 1

    def test_cached_only_never_hits_the_network(self, sqlite_cache):
        from scripts.common.scraping import get_club_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            out = get_club_transfer_news(["Arsenal"], cached_only=True)
            assert m.call_count == 0
        assert out.empty

    def test_force_refresh_ignores_the_cache(self, sqlite_cache):
        from scripts.common.scraping import get_club_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            m.side_effect = lambda c, timeout=15: pd.DataFrame([_club_row(c)])
            get_club_transfer_news(["Arsenal"])
            get_club_transfer_news(["Arsenal"], force_refresh=True)
            assert m.call_count == 2

    def test_cache_status_counts_hits_and_misses(self, sqlite_cache):
        from scripts.common.scraping import club_news_cache_status, get_club_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            m.side_effect = lambda c, timeout=15: pd.DataFrame([_club_row(c)])
            get_club_transfer_news(["Arsenal"])
        cached, missing = club_news_cache_status(["Arsenal", "Chelsea"])
        assert (cached, missing) == (1, 1)

    def test_empty_input_makes_no_requests(self):
        from scripts.common.scraping import club_news_cache_status, get_club_transfer_news

        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            assert get_club_transfer_news([]).empty
            assert m.call_count == 0
        assert club_news_cache_status([]) == (0, 0)

    def test_prefetch_runs_once_per_label(self, sqlite_cache):
        import scripts.common.scraping as scraping

        scraping._PREFETCH_STARTED.discard("clubs:unit-test")
        with patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as m:
            m.side_effect = lambda c, timeout=15: pd.DataFrame([_club_row(c)])
            assert scraping.start_club_news_prefetch(["Arsenal"], label="unit-test")
            assert not scraping.start_club_news_prefetch(["Arsenal"], label="unit-test")
            _join_prefetch_threads()
        scraping._PREFETCH_STARTED.discard("clubs:unit-test")

    def test_club_and_player_prefetch_labels_do_not_collide(self, sqlite_cache):
        """Both default to label="default"; a shared key would silently cancel one."""
        import scripts.common.scraping as scraping

        scraping._PREFETCH_STARTED.discard("collide")
        scraping._PREFETCH_STARTED.discard("clubs:collide")
        with patch("scripts.common.transfer_feeds.fetch_player_transfer_news") as mp, \
             patch("scripts.common.transfer_feeds.fetch_club_transfer_news") as mc:
            mp.side_effect = lambda n, t, timeout=15: pd.DataFrame([_news_row(n)])
            mc.side_effect = lambda c, timeout=15: pd.DataFrame([_club_row(c)])
            assert scraping.start_transfer_news_prefetch((("A", "X"),), label="collide")
            assert scraping.start_club_news_prefetch(["Arsenal"], label="collide")
            _join_prefetch_threads()
        scraping._PREFETCH_STARTED.discard("collide")
        scraping._PREFETCH_STARTED.discard("clubs:collide")
