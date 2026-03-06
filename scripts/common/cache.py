"""
SQLite Persistent Cache — key-value store for immutable FPL historical data.

Provides a transparent caching layer underneath Streamlit's in-memory cache.
Historical gameweek data (live points, picks, fixtures) never changes once
a GW finishes, so we cache it permanently in SQLite to survive app restarts.

Zero external dependencies — uses Python stdlib sqlite3.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Optional

from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.cache")

# Singleton connection — lazy-initialised by get_cache_db()
_connection: Optional[sqlite3.Connection] = None

_CREATE_TABLE_SQL = """CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at REAL
)"""


def _find_project_root() -> Path:
    """Walk up from this file (scripts/common/) to find the project root."""
    here = Path(__file__).resolve().parent
    # scripts/common/ -> scripts/ -> project root
    return here.parent.parent


def _default_db_path() -> Path:
    return _find_project_root() / ".fpl_cache.db"


def _init_connection(path: str) -> sqlite3.Connection:
    """Open a SQLite connection and ensure the cache table exists."""
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def _reset_default_connection() -> sqlite3.Connection:
    """Delete the corrupt DB file and create a fresh connection."""
    global _connection

    db_path = _default_db_path()
    _logger.warning("Resetting corrupt cache DB: %s", db_path)

    # Close existing connection if any
    if _connection is not None:
        try:
            _connection.close()
        except Exception:
            pass
        _connection = None

    # Delete the corrupt file (and WAL/SHM sidecars)
    for suffix in ("", "-wal", "-shm"):
        p = Path(str(db_path) + suffix)
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    # Create fresh
    _connection = _init_connection(str(db_path))
    return _connection


def get_cache_db(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Lazy-init singleton connection to .fpl_cache.db at project root.

    Parameters:
    - db_path: Override path (used by tests). If None, uses project root.

    Returns:
    - sqlite3.Connection with WAL mode and check_same_thread=False.
    """
    global _connection

    if db_path is not None:
        # Custom path (tests) — always create a new connection
        return _init_connection(db_path)

    if _connection is not None:
        return _connection

    path = _default_db_path()
    try:
        _connection = _init_connection(str(path))
    except sqlite3.DatabaseError:
        # Corrupt DB on first open — reset it
        _connection = _reset_default_connection()
    return _connection


def cache_get(key: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Any]:
    """
    Get value by key. Returns None if missing or expired.
    Deletes expired entries opportunistically.
    Auto-recovers from corrupt DB by deleting and recreating it.

    Parameters:
    - key: Cache key string.
    - conn: Optional connection override (for tests).
    """
    db = conn or get_cache_db()
    try:
        row = db.execute(
            "SELECT value, expires_at FROM cache WHERE key = ?", (key,)
        ).fetchone()
    except sqlite3.DatabaseError:
        _logger.warning("Cache read error (corrupt DB) for key %s", key, exc_info=True)
        if conn is None:
            _reset_default_connection()
        return None
    except Exception:
        _logger.warning("Cache read error for key %s", key, exc_info=True)
        return None

    if row is None:
        return None

    value_json, expires_at = row

    # Check expiry
    if expires_at is not None and time.time() > expires_at:
        # Expired — clean up and return None
        try:
            db.execute("DELETE FROM cache WHERE key = ?", (key,))
            db.commit()
        except Exception:
            pass
        return None

    try:
        return json.loads(value_json)
    except (json.JSONDecodeError, TypeError):
        _logger.warning("Cache deserialization error for key %s", key)
        return None


def cache_set(
    key: str,
    value: Any,
    ttl: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> None:
    """
    Store value as JSON. ttl=None means never expires (permanent).
    Auto-recovers from corrupt DB by deleting and recreating it.

    Parameters:
    - key: Cache key string.
    - value: Any JSON-serializable value.
    - ttl: Time-to-live in seconds. None = permanent.
    - conn: Optional connection override (for tests).
    """
    db = conn or get_cache_db()
    expires_at = (time.time() + ttl) if ttl is not None else None

    try:
        value_json = json.dumps(value)
        db.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (key, value_json, expires_at),
        )
        db.commit()
    except sqlite3.DatabaseError:
        _logger.warning("Cache write error (corrupt DB) for key %s — resetting", key, exc_info=True)
        if conn is None:
            new_db = _reset_default_connection()
            # Retry the write once on the fresh DB
            try:
                value_json = json.dumps(value)
                new_db.execute(
                    "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                    (key, value_json, expires_at),
                )
                new_db.commit()
            except Exception:
                _logger.warning("Cache write retry failed for key %s", key, exc_info=True)
    except Exception:
        _logger.warning("Cache write error for key %s", key, exc_info=True)


def cached_api_call(
    key: str,
    fetch_fn: Callable[[], Any],
    ttl: Optional[int] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> Any:
    """
    Check cache first, call fetch_fn on miss, store result.

    Parameters:
    - key: Cache key string.
    - fetch_fn: Zero-arg callable that returns the data on cache miss.
    - ttl: Time-to-live in seconds. None = permanent.
    - conn: Optional connection override (for tests).

    Returns:
    - Cached or freshly fetched data.
    """
    cached = cache_get(key, conn=conn)
    if cached is not None:
        return cached

    result = fetch_fn()
    if result is not None:
        cache_set(key, result, ttl=ttl, conn=conn)
    return result


def clear_cache(conn: Optional[sqlite3.Connection] = None) -> int:
    """
    Delete all entries from the cache. Returns number of rows deleted.
    Useful for development/debugging.
    """
    db = conn or get_cache_db()
    try:
        cursor = db.execute("DELETE FROM cache")
        db.commit()
        return cursor.rowcount
    except sqlite3.DatabaseError:
        _logger.warning("Cache clear error (corrupt DB) — resetting", exc_info=True)
        if conn is None:
            _reset_default_connection()
        return 0
    except Exception:
        _logger.warning("Cache clear error", exc_info=True)
        return 0
