"""Unit tests for scripts/common/injury_helpers.py.

``estimate_games_to_miss`` was moved verbatim out of scripts/draft/waiver_wire.py;
these tests pin its behaviour so the shared version cannot drift.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from scripts.common.injury_helpers import (
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
