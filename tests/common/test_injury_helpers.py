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
