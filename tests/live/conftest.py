"""Fixtures for the live plausibility suite.

These tests hit real Rotowire / FFP / FPL endpoints on purpose. The rest of the
suite mocks everything, which is why it stayed green through two bugs that were
entirely about upstream data changing shape.

Contract for everything in this directory:
  * unreachable upstream  -> SKIP (a Rotowire outage must not block a push)
  * reachable but implausible -> FAIL (that is the whole point)

Set FPL_SKIP_LIVE_TESTS=1 to opt out entirely (e.g. on a plane).
"""

import os

import pytest

import config

# The root conftest pins these so config.py never hits the network. Live tests
# need the real values, so they are cleared for this directory only.
_PINNED_FOR_OFFLINE_TESTS = ("FPL_CURRENT_GAMEWEEK", "ROTOWIRE_URL")


def pytest_collection_modifyitems(items):
    """Tag everything here with the `live` marker so `-m "not live"` works."""
    for item in items:
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


def skip_if_unreachable(fn, what):
    """Call fn(); skip (never fail) if the upstream source can't be reached."""
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - any transport failure means "skip"
        pytest.skip("%s unreachable: %s" % (what, exc))
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
def ffp_projections():
    from scripts.common.scraping import get_ffp_projections_data
    return skip_if_unreachable(lambda: get_ffp_projections_data(), "FFP sheet")


@pytest.fixture(scope="session")
def draft_league_id():
    league_id = config.FPL_DRAFT_LEAGUE_ID
    if not league_id:
        pytest.skip("No Draft league configured")
    return league_id
