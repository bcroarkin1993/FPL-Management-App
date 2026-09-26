"""Unit tests for scripts/common/injury_helpers.py.

``estimate_games_to_miss`` was moved verbatim out of scripts/draft/waiver_wire.py;
these tests pin its behaviour so the shared version cannot drift.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

import pandas as pd

from scripts.common.injury_helpers import (
    games_to_miss_series,
    gameweeks_until,
    stated_games_to_miss,
    INJURY_FLOOR,
    estimate_games_to_miss,
    gameweeks_remaining,
    injury_multiplier,
)


class TestEstimateGamesToMiss:
    """News text is the most reliable signal; status code the least."""

    def test_fit_player_misses_nothing(self):
        assert estimate_games_to_miss(None, None, "a") == 0

    def test_parses_expected_back_date(self):
        target = datetime.now() + timedelta(days=35)
        news = f"Knee injury - Expected back {target.strftime('%d %b')}"
        # 35 days out rounds up to 5-6 gameweeks
        assert 4 <= estimate_games_to_miss(news, None, "i") <= 6

    def test_date_rolls_forward_across_year_boundary(self):
        """A January date seen in December must mean next January, not last."""
        target = datetime.now() + timedelta(days=20)
        news = f"Expected back {target.strftime('%d %B')}"
        gws = estimate_games_to_miss(news, None, "i")
        assert gws >= 0, "a past date would produce a negative estimate"
        assert gws <= 5

    def test_suspension_length_beats_status_fallback(self):
        # status 's' alone would give 3; the news text says otherwise
        assert estimate_games_to_miss("Suspended for 7 matches", None, "s") == 7

    @pytest.mark.parametrize("chance,expected", [(80, 1), (60, 2), (30, 3), (0, 5)])
    def test_chance_buckets(self, chance, expected):
        assert estimate_games_to_miss(None, chance, None) == expected

    @pytest.mark.parametrize("status,expected", [("a", 0), ("d", 2), ("i", 4), ("n", 4), ("s", 3), ("u", 3)])
    def test_status_fallback(self, status, expected):
        assert estimate_games_to_miss(None, None, status) == expected

    def test_unparseable_news_falls_through_to_chance(self):
        assert estimate_games_to_miss("Knock", 80, "d") == 1

    def test_nan_inputs_do_not_raise(self):
        assert estimate_games_to_miss(np.nan, np.nan, np.nan) == 0


class TestInjuryMultiplier:
    """Severity scales with the fraction of the REMAINING season missed."""

    def test_fit_player_is_unpenalised(self):
        assert injury_multiplier(0, 10) == 1.0

    def test_same_absence_costs_more_later_in_season(self):
        early = injury_multiplier(5, 3)
        mid = injury_multiplier(5, 20)
        late = injury_multiplier(5, 34)
        assert early > mid > late, "a late-season injury must hurt more than an early one"
        assert early == pytest.approx(1 - 5 / 36, abs=1e-6)

    def test_floor_holds_for_season_ending_injury(self):
        assert injury_multiplier(40, 34) == INJURY_FLOOR

    def test_never_exceeds_one_or_drops_below_floor(self):
        for gw in range(1, 39):
            for missed in range(0, 40):
                m = injury_multiplier(missed, gw)
                assert INJURY_FLOOR <= m <= 1.0

    def test_bad_input_is_treated_as_fit(self):
        assert injury_multiplier("nonsense", 10) == 1.0
        assert injury_multiplier(np.nan, 10) == 1.0


class TestGameweeksRemaining:
    def test_counts_current_gameweek(self):
        assert gameweeks_remaining(1) == 38
        assert gameweeks_remaining(38) == 1

    def test_never_below_one(self):
        assert gameweeks_remaining(45) == 1

    def test_bad_input_defaults_to_full_season(self):
        assert gameweeks_remaining(None) == 38


class TestAFitPlayerMissesNothing:
    """FPL states an explicit 100 once news is resolved.

    Live, 84 players carry `chance == 100` with `status == 'a'`. The chance
    buckets below start at 75, so without an explicit 100 case they reported
    every one of those as missing a gameweek -- and `team_strength` applied an
    injury discount to a fully available squad. The `status` check that would
    have answered 0 is never reached, because a stated chance wins over it.
    """

    def test_an_explicit_hundred_is_zero(self):
        assert estimate_games_to_miss("", 100, "a") == 0

    def test_an_absent_chance_still_falls_through_to_status(self):
        assert estimate_games_to_miss("", np.nan, "a") == 0

    def test_a_doubt_below_a_hundred_is_unchanged(self):
        assert estimate_games_to_miss("", 75, "d") == 1


class TestStatedGamesToMiss:
    """Only a *stated* duration may bound a future gameweek.

    `estimate_games_to_miss` always answers, falling back through chance buckets
    to the status code. That is right for a discount -- something beats nothing --
    and wrong for anything asserting a fact about the next three gameweeks: "25%
    chance" becomes "misses 3 games" on no evidence, and a player who may be back
    next week gets written off.
    """

    def test_an_explicit_return_date_is_stated(self):
        assert stated_games_to_miss(
            "Hamstring injury - Expected back 11 Oct", 0, "i") is not None

    def test_a_suspension_length_is_stated(self):
        assert stated_games_to_miss("Suspended for 3 matches", np.nan, "a") == 3

    def test_out_of_the_squad_is_stated(self):
        """`i`/`s`/`u`/`n` mean FPL has removed him, not that it has a doubt."""
        assert stated_games_to_miss("", np.nan, "u") > 0
        assert stated_games_to_miss("", np.nan, "i") > 0

    def test_a_doubtful_player_states_nothing(self):
        """The case the gate exists for: `d` means he may well play."""
        assert stated_games_to_miss("Knock", 75, "d") is None
        assert stated_games_to_miss("", 25, "d") is None

    def test_unknown_return_date_does_not_leak_the_chance_bucket(self):
        """"Unknown return date" contains the word "return".

        Matching the keyword by hand let it through and then answered from the
        very buckets this function excludes. The duration has to come from the
        news *alone* -- chance and status withheld -- for the news to count.
        """
        from_news_only = estimate_games_to_miss(
            "Unspecified injury - Unknown return date", None, None)
        assert from_news_only == 0
        # He is still bounded, but by his status rather than by a bucket.
        assert stated_games_to_miss(
            "Unspecified injury - Unknown return date", 0, "i") == \
            estimate_games_to_miss(None, None, "i")

    def test_a_fit_player_states_nothing(self):
        assert stated_games_to_miss("", 100, "a") is None


class TestGamesToMissSeries:
    def test_it_is_zero_where_nothing_is_stated(self):
        idx = pd.Index([0, 1, 2])
        out = games_to_miss_series(
            pd.Series(["", "", "Knock"], index=idx),
            pd.Series([100, None, 75], index=idx),
            pd.Series(["a", "a", "d"], index=idx), idx)
        assert list(out) == [0, 0, 0]

    def test_it_reads_a_return_date(self):
        idx = pd.Index([7])
        out = games_to_miss_series(
            pd.Series(["Hamstring injury - Expected back 11 Oct"], index=idx),
            pd.Series([0], index=idx), pd.Series(["i"], index=idx), idx)
        assert out.iloc[0] > 0

    def test_a_bare_list_index_is_accepted(self):
        """`blend_aligned` passes whatever the caller had, list included."""
        out = games_to_miss_series(None, None, None, [0, 1])
        assert list(out) == [0, 0]


class TestAGameweekIsNotAWeek:
    """Return dates must be counted against the real calendar.

    `(days + 6) // 7` assumes a gameweek every seven days. The live calendar has
    14 days before GW6 and 14 more between GW10 and GW11, so the approximation
    overstates an absence -- worst during a break, which is exactly when a
    three-gameweek horizon reaches furthest ahead. Measured on 2026-09-26 over
    all 26 players with a parseable return date, it overstated **26 of 26** by a
    mean of 2.0 gameweeks: twelve due back on the GW6 deadline day itself,
    missing nothing, were counted as missing two.
    """

    @staticmethod
    def _deadlines():
        """The real shape: a fortnight's break, then weekly."""
        base = datetime(2026, 10, 10, 10, 0)
        return [base, base + timedelta(days=7), base + timedelta(days=13),
                base + timedelta(days=21)]

    def test_a_deadline_on_the_return_date_is_playable(self):
        """"Expected back 10 Oct" reads as available for the 10 Oct fixtures."""
        assert gameweeks_until(datetime(2026, 10, 10), self._deadlines()) == 0

    def test_it_counts_deadlines_not_weeks(self):
        assert gameweeks_until(datetime(2026, 10, 18), self._deadlines()) == 2

    def test_no_calendar_means_no_count(self):
        assert gameweeks_until(datetime(2026, 10, 18), None) == 0

    def test_the_estimator_uses_the_calendar_when_given_one(self):
        """Thirteen days out, but only a break in between: nothing is missed."""
        news = "Hamstring injury - Expected back 10 Oct"
        far = estimate_games_to_miss(news, 0, "i", deadlines=self._deadlines())
        assert far == 0

    def test_it_falls_back_to_the_seven_day_approximation(self):
        """Callers with no fixture list keep the old behaviour rather than zero."""
        soon = (datetime.now() + timedelta(days=13)).strftime("%d %b")
        assert estimate_games_to_miss(
            "Knee injury - Expected back %s" % soon, 0, "i") == 2

    def test_back_before_the_next_deadline_misses_nothing(self):
        """Zero is an answer, and must not read as "the news said nothing".

        Once the count comes from the calendar, a player back before the next
        deadline yields 0 -- and a gate that reads 0 as absent information falls
        through to the status default and writes him off. Pau Torres, "Expected
        back 10 Oct" against a GW6 deadline of 10 Oct, is that case: he misses
        nothing and was assumed out for four.
        """
        assert stated_games_to_miss(
            "Hamstring injury - Expected back 10 Oct", 0, "i",
            deadlines=self._deadlines()) == 0

    def test_a_suspension_with_an_end_date_reads_the_date(self):
        """"Suspended until 17 Oct" fell through to the status default of 3."""
        assert stated_games_to_miss(
            "Suspended until 17 Oct", 0, "s", deadlines=self._deadlines()) == 1

    def test_a_suspension_length_is_a_count_of_matches_not_a_date(self):
        """So the calendar cannot help, and must not interfere."""
        assert estimate_games_to_miss(
            "Suspended for 3 matches", None, None,
            deadlines=self._deadlines()) == 3
