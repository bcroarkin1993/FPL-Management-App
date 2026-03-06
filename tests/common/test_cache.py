"""Unit tests for scripts/common/cache.py — SQLite persistent cache."""

import time

import pytest

from scripts.common.cache import (
    cache_get,
    cache_set,
    cached_api_call,
    clear_cache,
    get_cache_db,
)


@pytest.fixture
def cache_conn(tmp_path):
    """Create an isolated SQLite cache in a temp directory."""
    db_path = str(tmp_path / "test_cache.db")
    conn = get_cache_db(db_path=db_path)
    yield conn
    conn.close()


class TestCacheSetAndGet:
    def test_round_trip_dict(self, cache_conn):
        cache_set("key1", {"a": 1, "b": [2, 3]}, conn=cache_conn)
        result = cache_get("key1", conn=cache_conn)
        assert result == {"a": 1, "b": [2, 3]}

    def test_round_trip_list(self, cache_conn):
        cache_set("key2", [1, 2, 3], conn=cache_conn)
        assert cache_get("key2", conn=cache_conn) == [1, 2, 3]

    def test_round_trip_string(self, cache_conn):
        cache_set("key3", "hello", conn=cache_conn)
        assert cache_get("key3", conn=cache_conn) == "hello"

    def test_round_trip_number(self, cache_conn):
        cache_set("key4", 42, conn=cache_conn)
        assert cache_get("key4", conn=cache_conn) == 42


class TestCacheMiss:
    def test_missing_key_returns_none(self, cache_conn):
        assert cache_get("nonexistent", conn=cache_conn) is None


class TestCacheExpiry:
    def test_expired_entry_returns_none(self, cache_conn):
        # Set with 1-second TTL
        cache_set("expiring", "value", ttl=1, conn=cache_conn)
        assert cache_get("expiring", conn=cache_conn) == "value"

        # Wait for expiry
        time.sleep(1.1)
        assert cache_get("expiring", conn=cache_conn) is None

    def test_permanent_entry_never_expires(self, cache_conn):
        cache_set("permanent", "forever", ttl=None, conn=cache_conn)
        # Should still be there (no time.sleep needed — ttl=None means no expiry)
        assert cache_get("permanent", conn=cache_conn) == "forever"


class TestCacheOverwrite:
    def test_overwrite_existing_key(self, cache_conn):
        cache_set("key", "old_value", conn=cache_conn)
        assert cache_get("key", conn=cache_conn) == "old_value"

        cache_set("key", "new_value", conn=cache_conn)
        assert cache_get("key", conn=cache_conn) == "new_value"


class TestCachedApiCall:
    def test_calls_fetch_fn_on_miss(self, cache_conn):
        call_count = {"n": 0}

        def fetch():
            call_count["n"] += 1
            return {"data": "from_api"}

        result = cached_api_call("api_key", fetch, conn=cache_conn)
        assert result == {"data": "from_api"}
        assert call_count["n"] == 1

    def test_returns_cached_on_hit(self, cache_conn):
        call_count = {"n": 0}

        def fetch():
            call_count["n"] += 1
            return {"data": "from_api"}

        # First call populates cache
        cached_api_call("api_key2", fetch, conn=cache_conn)
        assert call_count["n"] == 1

        # Second call should use cache, not call fetch
        result = cached_api_call("api_key2", fetch, conn=cache_conn)
        assert result == {"data": "from_api"}
        assert call_count["n"] == 1  # Still 1 — not called again

    def test_respects_ttl(self, cache_conn):
        call_count = {"n": 0}

        def fetch():
            call_count["n"] += 1
            return {"data": f"call_{call_count['n']}"}

        # First call with short TTL
        result1 = cached_api_call("ttl_key", fetch, ttl=1, conn=cache_conn)
        assert result1 == {"data": "call_1"}

        # Wait for expiry
        time.sleep(1.1)

        # Should call fetch again
        result2 = cached_api_call("ttl_key", fetch, ttl=1, conn=cache_conn)
        assert result2 == {"data": "call_2"}
        assert call_count["n"] == 2

    def test_does_not_cache_none(self, cache_conn):
        call_count = {"n": 0}

        def fetch():
            call_count["n"] += 1
            return None

        result = cached_api_call("none_key", fetch, conn=cache_conn)
        assert result is None

        # Should call fetch again since None wasn't cached
        result2 = cached_api_call("none_key", fetch, conn=cache_conn)
        assert result2 is None
        assert call_count["n"] == 2


class TestClearCache:
    def test_clears_all_entries(self, cache_conn):
        cache_set("a", 1, conn=cache_conn)
        cache_set("b", 2, conn=cache_conn)
        cache_set("c", 3, conn=cache_conn)

        deleted = clear_cache(conn=cache_conn)
        assert deleted == 3

        assert cache_get("a", conn=cache_conn) is None
        assert cache_get("b", conn=cache_conn) is None
        assert cache_get("c", conn=cache_conn) is None

    def test_clear_empty_cache(self, cache_conn):
        deleted = clear_cache(conn=cache_conn)
        assert deleted == 0
