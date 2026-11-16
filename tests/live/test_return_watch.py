"""Live plausibility checks for the returning-elite watch.

This board is built from two things nothing else in the app reads: the
``history_past`` array on ``element-summary``, and the ``PP/90`` column Rotowire
publishes beside its season totals. Both are upstream shapes, and the offline
tests pin fixtures captured on 2026-11-15 -- only these can notice that the live
ones stopped looking like them.

What makes that worth a live test is the failure mode. Every row on this board
is a player the rest of the app rates at zero, so there is no projection to
contradict a wrong answer and nothing downstream will object. If
``history_past`` stops carrying ``starts``, the whole prior-season rung silently
produces nothing and the board quietly shrinks to the players Rotowire happened
to price -- which looks exactly like a quiet week.

Contract (see conftest): unreachable -> SKIP, reachable but implausible -> FAIL.
"""

import pandas as pd
import pytest

import config
from scripts.common.analytics import merge_season_projections
from scripts.common.data_validation import check_return_watch, raise_on_error
from scripts.common.fpl_draft_api import pull_fpl_player_stats
from scripts.common.player_history import (
    SEASON_GWS,
    fetch_history_past,
    prior_season_rate,
)
from scripts.common.return_watch import BANDS, build_return_watch
from scripts.common.scraping import get_rotowire_season_rankings

from .conftest import skip_if_unreachable

# Enough of the pool to be representative without 670 requests per run.
_SAMPLE = 60


@pytest.fixture(scope="module")
def pool():
    df = skip_if_unreachable(pull_fpl_player_stats, "FPL bootstrap player stats")
    if df.empty:
        pytest.skip("FPL player stats returned no rows")
    return df


@pytest.fixture(scope="module")
def history(pool):
    """Prior-season rows for the most prominent slice of the pool."""
    top = pool.sort_values("total_points", ascending=False).head(_SAMPLE)
    ids = dict(zip(pd.to_numeric(top["id"], errors="coerce").astype(int),
                   pd.to_numeric(top["code"], errors="coerce").astype(int)))
    df = skip_if_unreachable(lambda: fetch_history_past(ids),
                             "element-summary history_past")
    if df.empty:
        pytest.skip("No prior-season history published for the sampled players")
    return df


@pytest.fixture(scope="module")
def season_rankings(rotowire_url):
    return skip_if_unreachable(
        lambda: get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL),
        "Rotowire season rankings",
    )


# =============================================================================
# history_past -- the prior-season rung
# =============================================================================

def test_history_past_still_carries_the_fields_the_rate_needs(history):
    for col in ("Player_Code", "Season_Name", "total_points", "starts"):
        assert col in history.columns, "history_past lost %r" % col
    assert history["Player_Code"].notna().all()
    assert history["Season_Name"].notna().all()


def test_starts_are_published_for_recent_seasons(history):
    """FPL did not track ``starts`` before 2022/23.

    So a zero there is legitimate for an old season and a *silent killer* for a
    recent one: the prior-season rung filters on ``starts``, and an all-zero
    column would remove the rung entirely while every remaining number stayed
    plausible.
    """
    recent = history[history["Season_Name"] >= "2022/23"]
    if recent.empty:
        pytest.skip("No seasons from 2022/23 onward in the sample")
    played = recent[pd.to_numeric(recent["minutes"], errors="coerce").fillna(0) > 900]
    assert len(played) > 0, "No sampled player logged a season of real minutes"
    assert (pd.to_numeric(played["starts"], errors="coerce").fillna(0) > 0).mean() > 0.9, \
        ("`starts` is missing for seasons where the player clearly played. The "
         "prior-season rung filters on it, so this empties the board silently.")


def test_a_rate_is_plausible_for_a_footballer(history):
    rates = prior_season_rate(history, min_starts=10)
    assert not rates.empty, "No sampled player has a usable prior season"
    assert rates["Prior_PPS"].between(0, 15).all(), \
        "Points per start outside [0, 15]: %s" % rates["Prior_PPS"].describe()
    assert rates["Prior_Role"].between(0, 1).all()
    assert rates["Prior_Starts"].le(SEASON_GWS + 4).all(), \
        "More starts than a season has gameweeks -- the payload changed shape."
    # These are the game's best players, so most should have been regulars.
    assert rates["Prior_Role"].median() > 0.4


def test_the_rate_is_one_row_per_player(history):
    rates = prior_season_rate(history, min_starts=10)
    assert rates["Player_Code"].is_unique


# =============================================================================
# PP/90 -- the preseason rung, and the column the app used to discard
# =============================================================================

def test_rotowire_still_publishes_a_rate_beside_its_total(season_rankings):
    assert "PP/90" in season_rankings.columns, \
        ("Rotowire stopped publishing PP/90. The season *total* bakes in a "
         "minutes assumption and correlates with real pedigree at 0.101 against "
         "the rate's 0.602, so the preseason rung has no honest substitute.")
    rate = pd.to_numeric(season_rankings["PP/90"], errors="coerce")
    assert rate.notna().mean() > 0.9
    assert rate.between(0, 15).all(), "PP/90 outside [0, 15]: %s" % rate.describe()


def test_the_merge_carries_the_rate_onto_the_pool(pool, season_rankings):
    merged = merge_season_projections(pool.rename(columns={"player": "Player",
                                                           "team_name_abbrv": "Team"}),
                                      season_rankings)
    assert "Season_PP90" in merged.columns
    got = pd.to_numeric(merged["Season_PP90"], errors="coerce")
    assert got.notna().sum() > 100, \
        "Only %d players got a PP/90 -- the name merge is failing." % got.notna().sum()


def test_the_total_and_the_rate_imply_a_sane_number_of_nineties(season_rankings):
    """The preseason rung's role signal is ``Points / PP90 / 38``.

    Rotowire builds the total that way, so the quotient is the minutes it
    expected. If the two columns ever stopped sharing that relationship the
    implied role would be nonsense while both columns still looked fine.
    """
    total = pd.to_numeric(season_rankings["Points"], errors="coerce")
    rate = pd.to_numeric(season_rankings["PP/90"], errors="coerce")
    ok = total.notna() & rate.notna() & (rate > 0)
    nineties = (total[ok] / rate[ok])
    assert nineties.between(0, SEASON_GWS + 2).mean() > 0.95, \
        ("Implied nineties outside [0, 40] for %.0f%% of rows -- Points and "
         "PP/90 no longer describe the same forecast."
         % (100 * (1 - nineties.between(0, SEASON_GWS + 2).mean())))


# =============================================================================
# The board itself
# =============================================================================

@pytest.fixture(scope="module")
def board(pool, season_rankings, current_gw):
    """The real detector over the real pool, with real league ownership."""
    league_id = config.FPL_DRAFT_LEAGUE_ID
    if not league_id:
        pytest.skip("No Draft league configured")
    from scripts.draft.waiver_core import _code_states
    from scripts.common.fpl_draft_api import get_league_element_states

    states = skip_if_unreachable(
        lambda: _code_states(get_league_element_states(league_id)) or None,
        "Draft element states",
    )
    enriched = merge_season_projections(
        pool.rename(columns={"player": "Player", "team_name_abbrv": "Team",
                             "code": "Player_Code", "id": "Player_ID"}),
        season_rankings)
    ids = dict(zip(pd.to_numeric(enriched["Player_ID"], errors="coerce").astype(int),
                   pd.to_numeric(enriched["Player_Code"], errors="coerce").astype(int)))
    rates = prior_season_rate(fetch_history_past(ids),
                              min_starts=config.RETURN_WATCH_PRIOR_MIN_STARTS)
    notes = []
    frame = build_return_watch(enriched, states, current_gw=current_gw,
                               prior_rates=rates, notes=notes)
    return frame, notes


def test_the_live_board_is_plausible(board, current_gw):
    frame, _notes = board
    raise_on_error(check_return_watch(frame, current_gw=current_gw),
                   "live returning-elite board")


def test_no_departed_player_reaches_the_live_board(board):
    """The likeliest false positive, and the one with no other symptom.

    ``status`` is 'u' for a loan, a permanent exit and an ordinary absence
    alike, so a departure can only be told from the prose. Measured on the live
    pool, 7 of 18 raw candidate rows had left the league -- every one reading as
    a perfectly ordinary injury row.
    """
    frame, _notes = board
    if frame.empty:
        pytest.skip("Empty board -- a legitimate state")
    gone = frame[frame["News"].astype("string").fillna("").str.contains(
        "has joined|has left|not included in squad|loan", case=False, na=False)]
    assert gone.empty, "Departed players on the board: %s" % \
        gone["Display_Name"].tolist()


def test_every_row_is_claimable_and_banded(board):
    frame, _notes = board
    if frame.empty:
        pytest.skip("Empty board -- a legitimate state")
    assert set(frame["Draft_State"].astype(str)) <= {"a", "l"}
    assert set(frame["Band"].astype(str)) <= set(BANDS)
    assert frame["Display_Name"].notna().all()
    # A dotted abbreviation means the display-name resolution was bypassed.
    assert not frame["Display_Name"].astype(str).str.contains(r"\.").any()


def test_the_board_is_not_the_whole_pool(board, pool):
    """A gate that has silently stopped gating looks like a working board.

    If the not-playing or role gate ever failed open this would be hundreds of
    rows of entirely plausible-looking percentages.
    """
    frame, _notes = board
    assert len(frame) <= config.RETURN_WATCH_LIMIT
    assert len(frame) < 0.2 * len(pool)


def test_an_empty_board_always_states_why(board):
    """"Nothing to do" and "we could not check" are opposite claims."""
    frame, notes = board
    if frame.empty:
        assert notes, ("The board is empty and recorded no reason, so a reader "
                       "cannot tell a quiet week from a broken feed.")
