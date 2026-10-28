"""Sportsgambler's projected XIs, against the live page.

The engine's **second** start opinion, and the reason it has one: a benching is
a claim about everyone an XI leaves out, and with a single source there was
nothing to corroborate or contradict it. Measured 2026-10-22 at GW8, three
published XIs agreed on only eight of Manchester City's eleven.

**Every failure mode here is invisible from the app.** Each source fails to "no
opinion" independently and the others carry on, so a page that has stopped
parsing looks exactly like one that simply has not published yet -- which is
why the floors below are asserted against the live fixture list rather than
against a fixed count.
"""

import pytest

import config
from scripts.common import rotowire_lineups, sportsgambler_lineups
from scripts.common.data_validation import (
    check_lineup_consensus,
    check_lineup_xi,
    raise_on_error,
)

from .conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def index_rows():
    return skip_if_unreachable(
        lambda: sportsgambler_lineups.parse_index(
            sportsgambler_lineups._get(config.SPORTSGAMBLER_LINEUPS_URL).text),
        "Sportsgambler lineups index",
    )


@pytest.fixture(scope="module")
def lineup_xi(current_gw):
    return skip_if_unreachable(
        lambda: sportsgambler_lineups.fetch_lineup_xi(gameweek=current_gw),
        "Sportsgambler lineups page",
    )


def test_the_index_still_lists_fixtures_with_ids(index_rows):
    """The ids are what the AJAX endpoint takes, so losing them loses everything."""
    assert len(index_rows) >= 10, (
        "only %d fixtures parsed from the lineups index -- the "
        "`a.lineups-toggle-button` anchors or the `.lineup-row` wrapper have "
        "probably changed" % len(index_rows))
    assert all(fid and home and away for fid, home, away in index_rows)


def test_the_index_carries_more_than_one_gameweek(index_rows, current_gw):
    """Why the filter is a per-fixture lookup and not a page-wide vote.

    Every gameweek on this page is a complete set of ten, so a vote over all of
    them scores each equally. If this ever stops being true the lookup is still
    correct -- but the reasoning in the module docstring would no longer be.
    """
    if len(index_rows) <= 12:
        pytest.skip("the page is listing roughly one gameweek at the moment")
    pairs = rotowire_lineups._gameweek_fixture_pairs(current_gw)
    if not pairs:
        pytest.skip("no fixture list for the current gameweek")
    kept = sportsgambler_lineups._fixtures_in_gameweek(index_rows, pairs)
    assert len(kept) < len(index_rows), (
        "the gameweek filter kept every fixture on a page carrying %d of them, "
        "so it is not filtering at all" % len(index_rows))


def test_every_club_label_resolves(index_rows):
    """An unmapped label **fails closed**, so the club gets no XI opinion at all.

    "Man United" and "Nott'm Forest" with a curly apostrophe were both missing
    from TEAM_FULL_TO_SHORT when this source was added, costing 2 of 20 clubs.
    """
    from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

    unmapped = sorted({
        label for _fid, home, away in index_rows for label in (home, away)
        if not TEAM_FULL_TO_SHORT.get(str(label).strip())
    })
    assert not unmapped, (
        "club labels this app cannot resolve: %r. Each one silently removes a "
        "club's XI opinion entirely -- add them to TEAM_FULL_TO_SHORT."
        % (unmapped,))


def test_the_xis_are_plausible(lineup_xi, current_gw):
    """One assertion for the shape of the whole signal."""
    raise_on_error(check_lineup_xi(lineup_xi, current_gw,
                                   source="Sportsgambler"))


def test_the_xis_belong_to_the_gameweek_being_scored(lineup_xi, current_gw):
    if not lineup_xi.ok:
        pytest.skip("no XI resolved yet this gameweek")
    assert lineup_xi.gameweek == current_gw


def test_the_two_sources_are_describing_the_same_gameweek(current_gw):
    """The failure neither source's own checks can see.

    A source that has drifted onto another gameweek is perfectly well-formed --
    eleven names a club, every label resolved -- and the engine would bench
    players who are not in this week's squad at all.
    """
    rw = skip_if_unreachable(
        lambda: rotowire_lineups.fetch_lineup_xi(gameweek=current_gw),
        "Rotowire lineups page")
    sg = skip_if_unreachable(
        lambda: sportsgambler_lineups.fetch_lineup_xi(gameweek=current_gw),
        "Sportsgambler lineups page")
    if not (rw.ok and sg.ok):
        pytest.skip("only one source has published so far this gameweek")
    raise_on_error(check_lineup_consensus({"rotowire": rw, "sportsgambler": sg}))


def test_names_resolve_at_a_high_rate(lineup_xi, index_rows, current_gw):
    """Measured 196 of 198 (99.0%) at GW8 through the shared resolver.

    A collapse here means a transliteration the matcher cannot bridge, and it
    costs the *downward* half of the signal first: a club one name short can
    promote its starters but bench nobody.
    """
    if not lineup_xi.ok:
        pytest.skip("no XI resolved yet this gameweek")
    pairs = rotowire_lineups._gameweek_fixture_pairs(current_gw)
    if not pairs:
        pytest.skip("no fixture list for the current gameweek")
    wanted = sportsgambler_lineups._fixtures_in_gameweek(index_rows, pairs)
    expected = len(wanted) * 2 * rotowire_lineups.XI_SIZE
    if not expected:
        pytest.skip("no fixtures for this gameweek on the page yet")
    rate = len(lineup_xi.player_ids) / float(expected)
    assert rate >= 0.90, (
        "only %d of ~%d starters resolved (%.0f%%) -- %s"
        % (len(lineup_xi.player_ids), expected, rate * 100, lineup_xi.note))
