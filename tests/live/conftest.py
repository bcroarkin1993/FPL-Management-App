"""Fixtures for the live plausibility suite.

These tests hit real Rotowire / FFP / FPL endpoints on purpose. The rest of the
suite mocks everything, which is why it stayed green through two bugs that were
entirely about upstream data changing shape.

Contract for everything in this directory:
  * unreachable upstream  -> SKIP (a Rotowire outage must not block a push)
  * reachable but implausible -> FAIL (that is the whole point)
  * a bug in our own code -> FAIL, never SKIP

That last line is not decoration. ``skip_if_unreachable`` once caught bare
``Exception``, so an ``AttributeError`` inside ``compute_player_scores`` was
reported as "Draft league strength unreachable" and four Power Rankings checks
stood down for days while the page was broken. Only transport failures skip.

Set FPL_SKIP_LIVE_TESTS=1 to opt out entirely (e.g. on a plane).
"""

import json
import os
import pathlib
import xml.etree.ElementTree as ET

import pytest
import requests

import config

# The root conftest pins these so config.py never hits the network. Live tests
# need the real values, so they are cleared for this directory only.
_PINNED_FOR_OFFLINE_TESTS = ("FPL_CURRENT_GAMEWEEK", "ROTOWIRE_URL")


def pytest_collection_modifyitems(items):
    """Tag everything in *this directory* with the `live` marker.

    The hook is handed every collected item in the run, not just this package's,
    so it must filter by path. Without that it marked the whole suite live and
    `-m "not live"` deselected all 844 tests instead of skipping the ~35 that
    touch the network.
    """
    here = str(pathlib.Path(__file__).parent.resolve())
    for item in items:
        if str(pathlib.Path(str(item.fspath)).resolve()).startswith(here):
            item.add_marker(pytest.mark.live)


@pytest.fixture(scope="session", autouse=True)
def _unpin_offline_env():
    """Restore real network-backed config resolution for this directory."""
    if os.environ.get("FPL_SKIP_LIVE_TESTS"):
        pytest.skip("FPL_SKIP_LIVE_TESTS is set", allow_module_level=True)

    saved = {k: os.environ.pop(k, None) for k in _PINNED_FOR_OFFLINE_TESTS}
    config._GW_CACHE = None
    config._RW_URL_CACHE = None
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value
        config._GW_CACHE = None
        config._RW_URL_CACHE = None


#: Failures that mean "the network or the far end let us down". Everything else
#: is our own code and must fail loudly. requests.RequestException covers
#: connection/timeout/HTTP-status; OSError covers socket-level failures; the JSON
#: and XML decoders signal a truncated or non-JSON response.
_TRANSPORT_ERRORS = (
    requests.RequestException,
    OSError,            # includes socket.timeout, ConnectionError
    json.JSONDecodeError,
    ET.ParseError,
)


def _is_transport_failure(exc) -> bool:
    """True if this exception, or anything that caused it, is a transport failure.

    The chain matters: app code routinely catches a request error and re-raises
    something of its own, and that is still an outage, not a bug.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, _TRANSPORT_ERRORS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def skip_if_unreachable(fn, what):
    """Call fn(); skip only if the upstream source could not be reached.

    A bug in our own code is *not* an outage and must not be filed as one — see
    the contract at the top of this file.
    """
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - re-raised below unless transport
        if _is_transport_failure(exc):
            pytest.skip("%s unreachable: %s" % (what, exc))
        raise AssertionError(
            "%s failed with %s: %s\n"
            "This is a bug in our code, not an upstream outage -- the live suite "
            "fails rather than skipping so it cannot hide."
            % (what, type(exc).__name__, exc)
        ) from exc
    if result is None:
        pytest.skip("%s returned nothing" % what)
    return result


@pytest.fixture(scope="session")
def current_gw():
    return skip_if_unreachable(lambda: config.CURRENT_GAMEWEEK, "FPL gameweek endpoint")


@pytest.fixture(scope="session")
def rotowire_url():
    url = skip_if_unreachable(lambda: config.ROTOWIRE_URL, "Rotowire article index")
    if not url:
        pytest.skip("Rotowire article discovery returned an empty URL")
    return url


@pytest.fixture(scope="session")
def rotowire_projections(rotowire_url):
    from scripts.common.scraping import get_rotowire_player_projections
    df = skip_if_unreachable(lambda: get_rotowire_player_projections(rotowire_url), "Rotowire article")
    if df.empty:
        pytest.skip("Rotowire article parsed to zero rows")
    return df


@pytest.fixture(scope="session")
def ffp_feed():
    from scripts.common.scraping import get_ffp_feed
    feed = skip_if_unreachable(lambda: get_ffp_feed(), "FFP")
    if not feed.ok:
        pytest.skip("FFP returned no rows: %s" % (feed.note or "unknown"))
    return feed


@pytest.fixture(scope="session")
def ffp_projections(ffp_feed):
    return ffp_feed.df


@pytest.fixture(scope="session")
def draft_league_id():
    league_id = config.FPL_DRAFT_LEAGUE_ID
    if not league_id:
        pytest.skip("No Draft league configured")
    return league_id


@pytest.fixture(scope="session")
def classic_player_pool():
    """The full FPL player pool as the Initial Squad Optimizer builds it."""
    from scripts.classic.initial_squad import _build_full_player_pool
    from scripts.common.utils import get_classic_bootstrap_static

    bootstrap = skip_if_unreachable(get_classic_bootstrap_static, "FPL bootstrap")
    pool = _build_full_player_pool(bootstrap or {})
    if pool.empty:
        pytest.skip("FPL bootstrap returned no players")
    return pool


@pytest.fixture(scope="session")
def rotowire_season_rankings():
    from scripts.common.scraping import get_rotowire_season_rankings

    df = skip_if_unreachable(
        lambda: get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL),
        "Rotowire season rankings",
    )
    if df is None or df.empty:
        pytest.skip("Rotowire season rankings parsed to zero rows")
    return df
