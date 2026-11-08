"""FPL's published chip rules, against what the app assumes.

Every chip window this app relies on is published in
``bootstrap-static.chips``, and for a season it was compiled in instead -- which
is how Free Hit and Triple Captain came to be modelled as single-use after FPL
made them double. A manager who had played one was told it was gone when a
second opens at GW20.

A rule change is otherwise **invisible**: the app keeps enforcing last season's
rules and every number it renders stays plausible. So this is the tripwire, and
it follows the ``tests/live/`` contract -- unreachable skips, reachable but
implausible fails.
"""

import pytest

from scripts.common.chips import (
    DEFAULT_CHIP_WINDOWS,
    KNOWN_CHIPS,
    chip_states,
    parse_chip_definitions,
)
from scripts.common.data_validation import check_chip_definitions, raise_on_error
from scripts.common.fpl_classic_api import get_classic_bootstrap_static
from tests.live.conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def bootstrap():
    return skip_if_unreachable(get_classic_bootstrap_static,
                               "FPL bootstrap-static")


def test_fpl_publishes_a_chips_list(bootstrap):
    """If this ever goes away the app silently falls back to compiled-in rules."""
    assert isinstance(bootstrap.get("chips"), list)
    assert bootstrap["chips"], "chips list is present but empty"


def test_the_published_rules_are_plausible(bootstrap):
    raise_on_error(check_chip_definitions(bootstrap, known_chips=KNOWN_CHIPS),
                   context="bootstrap-static.chips")


def test_every_published_chip_parses(bootstrap):
    """A window that fails to parse is a chip the expiry clock cannot time."""
    published = [c for c in bootstrap["chips"]
                 if c.get("start_event") is not None]
    assert len(parse_chip_definitions(bootstrap)) == len(published)


def test_the_compiled_fallback_still_matches_what_fpl_publishes(bootstrap):
    """`DEFAULT_CHIP_WINDOWS` is what a feed outage falls back to.

    Like `projection_engine.DEFAULT_START_FLOORS` tracking `config.py`, it has
    to be kept honest -- a stale fallback means the app and its own backup
    disagree about the rules, and only during an outage, which is the worst
    time to find out.
    """
    live = sorted((c["name"], int(c["start_event"]), int(c["stop_event"]))
                  for c in bootstrap["chips"] if c.get("start_event") is not None)
    fallback = sorted((w.name, w.start_event, w.stop_event)
                      for w in DEFAULT_CHIP_WINDOWS)
    assert live == fallback, (
        "chips.DEFAULT_CHIP_WINDOWS no longer matches the live rules; update it")


def test_all_four_chips_are_double_use_this_season(bootstrap):
    """Measured 2026-11-06: eight chips, two windows each.

    This is the specific fact the app got wrong. A failure here is not
    necessarily a bug -- FPL may have changed the rules again -- but it means
    `chips.KNOWN_CHIPS`, the fallback and the page's captions all need re-reading.
    """
    names = [c.get("name") for c in bootstrap["chips"]]
    for chip in KNOWN_CHIPS:
        assert names.count(chip) == 2, f"{chip} has {names.count(chip)} windows"


def test_the_two_halves_partition_the_season(bootstrap):
    """The expiry clock says "they expire after GW N", so the windows must
    actually tile the season rather than overlap or leave a hole."""
    defs = parse_chip_definitions(bootstrap)
    for chip in KNOWN_CHIPS:
        windows = sorted(((w.start_event, w.stop_event) for w in defs
                          if w.name == chip))
        assert windows, chip
        for (a_start, a_stop), (b_start, b_stop) in zip(windows, windows[1:]):
            assert b_start == a_stop + 1, f"{chip}: gap or overlap at GW{a_stop}"
        assert windows[-1][1] == 38, f"{chip} does not run to GW38"


def test_a_spent_chip_reports_its_next_window_rather_than_reading_as_gone(bootstrap):
    """The user-visible bug, pinned against the real rules."""
    defs = parse_chip_definitions(bootstrap)
    states = chip_states(defs, [{"name": "freehit", "event": 7}], current_gw=10)
    assert states["freehit"].playable_now is False
    assert states["freehit"].next_window_opens == 20
