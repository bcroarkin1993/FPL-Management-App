"""Live plausibility checks for the Premier League content API.

This is the layer that catches the PL's CMS changing shape. The offline tests
pin a payload captured on 2026-09-11; only these can notice that the live one
stopped looking like it.

Contract (see conftest): unreachable -> SKIP, reachable but implausible -> FAIL.
"""

import pytest

import config
from scripts.common import pl_content
from scripts.common.data_validation import check_pl_content, raise_on_error
from scripts.fpl.injuries import get_fpl_availability_df

from .conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def lineups():
    return skip_if_unreachable(
        lambda: pl_content.get_predicted_lineups(config.CURRENT_GAMEWEEK) or None,
        "PL predicted-lineups article",
    )


@pytest.fixture(scope="module")
def injuries():
    df = skip_if_unreachable(pl_content.fetch_pl_injuries, "PL injury table")
    if df.empty:
        pytest.skip("PL injury table returned no rows")
    return df


# =============================================================================
# Discovery
# =============================================================================

def test_articles_are_discoverable_by_tag():
    """The article is found by tag rather than pinned per gameweek. If the tag
    id changes the whole feature silently stops -- with no error, because an
    empty listing is indistinguishable from 'not published yet'."""
    listing = skip_if_unreachable(
        lambda: pl_content.list_predicted_lineup_articles() or None,
        "PL predicted-lineups tag listing",
    )
    assert len(listing) >= 3, "tag %s returned %d articles" % (
        config.PL_PREDICTED_LINEUPS_TAG, len(listing))
    weeks = [a["gameweek"] for a in listing if a["gameweek"] is not None]
    assert weeks, "no edition stated a matchweek in its title: %s" % (
        [a["title"] for a in listing[:3]],)


# =============================================================================
# The article
# =============================================================================

def test_article_is_for_the_current_gameweek(lineups):
    """Either the PL has published this matchweek and it parses as this
    matchweek, or it has not published yet and we say so. What must never
    happen is last week's XI rendering under this week's heading."""
    if not lineups.ok:
        pytest.skip("PL has not published GW%s yet: %s"
                    % (config.CURRENT_GAMEWEEK, lineups.note))
    assert lineups.gameweek == config.CURRENT_GAMEWEEK, lineups.note


def test_article_shape_is_plausible(lineups):
    if not lineups.ok:
        pytest.skip("PL has not published GW%s yet" % config.CURRENT_GAMEWEEK)
    raise_on_error(check_pl_content(
        lineups=lineups, expected_gw=config.CURRENT_GAMEWEEK))


def test_every_club_the_pl_publishes_resolves(lineups):
    """An unmapped club label is the FFP 'Notts Forest' failure: matching and
    gameweek voting are both scoped by club, so it drops that club silently."""
    if not lineups.ok:
        pytest.skip("PL has not published GW%s yet" % config.CURRENT_GAMEWEEK)
    assert lineups.unresolved_labels == (), (
        "add these to TEAM_FULL_TO_SHORT: %s" % (lineups.unresolved_labels,))


def test_fixtures_agree_with_the_real_fixture_list(lineups):
    """The article must be able to prove its own gameweek independently of its
    title -- the title is prose and prose gets edited."""
    if not lineups.ok:
        pytest.skip("PL has not published GW%s yet" % config.CURRENT_GAMEWEEK)
    voted = pl_content.resolve_gameweek_from_fixtures(lineups.fixtures)
    if voted is None:
        pytest.skip("FPL fixture list unavailable for the cross-check")
    assert voted == config.CURRENT_GAMEWEEK


def test_xi_graphics_are_published_for_both_sides(lineups):
    if not lineups.ok:
        pytest.skip("PL has not published GW%s yet" % config.CURRENT_GAMEWEEK)
    expected = {code for pair in lineups.fixtures for code in pair}
    missing = expected - set(lineups.graphics)
    assert len(missing) <= 2, (
        "no XI graphic for %s -- document order or the photo-title guard has "
        "probably broken" % sorted(missing))


# =============================================================================
# The injury table
# =============================================================================

def test_injury_table_covers_the_league(injuries):
    raise_on_error(check_pl_content(injuries=injuries))
    assert injuries["Club"].nunique() >= 18


def test_injury_clubs_all_resolve(injuries):
    unmapped = sorted(injuries.loc[injuries["Team"].isna(), "Club"].unique())
    assert not unmapped, "add to TEAM_FULL_TO_SHORT: %s" % unmapped


def test_injury_rows_match_the_fpl_pool(injuries):
    """Measured 97.6% on 2026-09-11. A collapse means the club-scoped key or
    the token-subset fallback broke -- a silently empty column otherwise."""
    pool = skip_if_unreachable(get_fpl_availability_df, "FPL bootstrap")
    matched = pl_content.attach_pl_injuries(
        pl_content.add_display_names(pool), injuries)
    raise_on_error(check_pl_content(injuries=injuries, matched_rows=len(matched)))


def test_injury_types_are_short_labels_not_prose(injuries):
    """``description`` is a one- or two-word injury type ("ACL", "Hamstring").
    If the PL started writing sentences there, the table column would become
    unreadable and the field would mean something different."""
    longest = injuries["Injury"].fillna("").str.len().max()
    assert longest <= 40, "PL injury descriptions are now prose (%d chars)" % longest


def test_cms_item_date_is_still_unreliable(injuries):
    """Pins the observation that justifies not rendering it: on 2026-09-11, 46
    of 83 rows were over 60 days old and 20 shared one date. If this ever stops
    being true the field becomes worth surfacing -- so the test failing is the
    signal to revisit, not a breakage."""
    dates = injuries["Item_Date"].dropna()
    dates = dates[dates.str.len() == 10]
    if dates.empty:
        pytest.skip("no item dates published")
    top_share = dates.value_counts().iloc[0] / len(dates)
    assert top_share > 0.1, (
        "PL item dates now look genuinely per-injury (top value is only %.0f%% "
        "of rows) -- reconsider surfacing Item_Date" % (top_share * 100))
