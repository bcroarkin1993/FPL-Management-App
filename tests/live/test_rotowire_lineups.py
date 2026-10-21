"""Rotowire's projected XIs, against the live page.

This is the projection engine's start signal: it decides, for every player in
the game, whether Rotowire expects him to start -- the difference between the
positional floor (0.86 for a midfielder) and the omission prior (0.02). So the
whole projection turns on it.

**Every failure mode here is invisible from the app**, because the engine is
deliberately fail-open: an unreachable page, a changed selector, or an
unresolvable club name yields "no opinion", the engine falls back to Rotowire's
weekly rankings article, and the page renders exactly what it rendered before
this source existed. Which is the stale number this exists to replace.
"""

import pytest

import config
from scripts.common import rotowire_lineups
from scripts.common.data_validation import check_rotowire_lineup_xi, raise_on_error

from .conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def lineup_xi(current_gw):
    return skip_if_unreachable(
        lambda: rotowire_lineups.fetch_lineup_xi(gameweek=current_gw),
        "Rotowire lineups page",
    )


def test_the_xis_are_plausible(lineup_xi, current_gw):
    """One assertion for the shape of the whole signal."""
    raise_on_error(check_rotowire_lineup_xi(lineup_xi, current_gw))


def test_most_clubs_resolve_completely(lineup_xi):
    """Only a club whose eleven *all* resolve may bench anybody.

    One unmatched starter is indistinguishable from a benched one, so a club
    that falls short keeps promoting its resolved starters and benches nobody.
    Measured 2026-10-21 at GW8: 19 of 20 complete, Brighton short on two names
    FPL and Rotowire genuinely spell differently.
    """
    assert len(lineup_xi.clubs) >= 17, (
        "only %d clubs' XIs resolved completely (%s). The downward half of the "
        "start signal is off for the rest, silently."
        % (len(lineup_xi.clubs), lineup_xi.note))


def test_the_xis_belong_to_the_gameweek_being_scored(current_gw):
    """Rotowire's page carries whatever matches it has lineups for, which runs
    past the current gameweek. Importing next week's XI as this week's start
    signal would bench the players who are actually playing."""
    scrape = skip_if_unreachable(
        lambda: rotowire_lineups.scrape_lineups(
            config.ROTOWIRE_LINEUPS_URL, current_gw),
        "Rotowire lineups page",
    )
    pairs = skip_if_unreachable(
        lambda: rotowire_lineups._gameweek_fixture_pairs(current_gw),
        "FPL fixture list",
    )
    if not pairs:
        pytest.skip("no fixture list for GW%s, so nothing to check against" % current_gw)

    from scripts.common.text_helpers import TEAM_FULL_TO_SHORT
    unknown = []
    for home, away, _ in scrape.matchups:
        h = TEAM_FULL_TO_SHORT.get(str(home).strip())
        a = TEAM_FULL_TO_SHORT.get(str(away).strip())
        if not h or not a:
            unknown.append((home, away))
            continue
        assert (h, a) in pairs, (
            "%s v %s is not a GW%s fixture but survived the filter"
            % (home, away, current_gw))

    # An unmapped club label keeps its matchup (fail-open), so it cannot be an
    # error -- but it does disable the filter for that match, which is worth
    # knowing about before it lets a stray fixture through.
    assert not unknown, (
        "club labels missing from TEAM_FULL_TO_SHORT, so the gameweek filter is "
        "off for these matchups: %s" % unknown)


def test_the_two_rotowire_publications_mostly_agree(lineup_xi, current_gw):
    """A sanity bound on the disagreement, not a demand that they agree.

    The article and the lineups page are different publications of different
    vintages, and the gap between them is exactly what this source exists to
    capture: measured 2026-10-21, 3 of 220 players. A gap of *most* of the
    table is something else -- a matching collapse, or two different gameweeks
    -- and would silently reprice a third of the league.
    """
    from scripts.common.scraping import get_rotowire_player_projections
    from scripts.common.player_pool import blended_pool

    article = skip_if_unreachable(
        lambda: get_rotowire_player_projections(config.ROTOWIRE_URL),
        "Rotowire rankings article",
    )
    if article is None or article.empty:
        pytest.skip("Rotowire has published no rankings article for GW%s" % current_gw)

    pool = skip_if_unreachable(
        lambda: blended_pool(current_gw, config.ROTOWIRE_URL), "FPL player pool")
    if pool is None or pool.empty or "Start_Pct__rotowire" not in pool.columns:
        pytest.skip("no blended pool to compare against")

    # Players the engine recorded a Rotowire start decision for, at a club whose
    # XI resolved completely -- the only cohort where the two can be compared.
    scoped = pool[pool["Team"].isin(lineup_xi.clubs)]
    if scoped.empty:
        pytest.skip("no club resolved completely")
    priced = scoped["Points"].notna() if "Points" in scoped.columns else None
    if priced is None:
        pytest.skip("the pool carries no Rotowire column")

    in_xi = scoped["Player_ID"].isin(lineup_xi.player_ids)
    disagree = int((priced.fillna(False) != in_xi).sum())
    assert disagree <= 0.25 * len(scoped), (
        "the rankings article and the lineups page disagree about %d of %d "
        "players at fully-resolved clubs. That is far past a stale article and "
        "looks like a matching failure or two different gameweeks."
        % (disagree, len(scoped)))
