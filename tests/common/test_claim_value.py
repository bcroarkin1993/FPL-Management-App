"""Rent-vs-keep claim valuation.

Every number in these fixtures is one the app actually produced on league 11347
at GW10, which is the gameweek the defect was reported on. The values matter:
the model's job is to rank a player with a big this-week projection and a thin
horizon *above* a mid/mid player, and that only holds at realistic magnitudes.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common import claim_value as cv


# Live GW10 values. Madueke was the highest-projected available player in the
# game and was suggested nowhere; Bobb was the one the shipped score preferred.
MADUEKE = {"Display_Name": "Noni Madueke", "Position": "M", "Proj": 5.371,
           "Proj_Next3": 11.50}
BOBB = {"Display_Name": "Oscar Bobb", "Position": "M", "Proj": 4.610,
        "Proj_Next3": 12.74}
# Roster midfielders, with their real release costs against a 3.67 MID floor.
GOMES = {"Display_Name": "Joao Gomes", "Position": "M", "Proj": 1.840,
         "Proj_Next3": 5.84}        # forward rate 2.00 -> free to release
IWOBI = {"Display_Name": "Alex Iwobi", "Position": "M", "Proj": 4.348,
         "Proj_Next3": 11.62}       # forward rate 3.64 -> free to release
TEL = {"Display_Name": "Mathys Tel", "Position": "M", "Proj": 4.005,
       "Proj_Next3": 13.12}         # forward rate 4.56 -> expensive

FLOORS = {"M": 3.67, "D": 2.78}


def _frame(*rows):
    return pd.DataFrame(list(rows))


class TestForwardRate:
    def test_it_excludes_the_current_gameweek(self):
        """`Proj_Next3` includes the current gameweek, so the forward rate is
        (Next3 - Proj) / 2 and never Next3 / 3.

        This decides the ordering the whole feature turns on: the naive division
        gives Madueke 3.83 and Bobb 4.25, the correct one 3.06 and 4.07.
        """
        got = cv.forward_rate(_frame(MADUEKE, BOBB))
        assert got.iloc[0] == pytest.approx((11.50 - 5.371) / 2, abs=1e-9)
        assert got.iloc[1] == pytest.approx((12.74 - 4.610) / 2, abs=1e-9)
        # And is emphatically not the naive form.
        assert got.iloc[0] != pytest.approx(11.50 / 3, abs=0.01)

    def test_an_unknown_horizon_stays_unknown(self):
        """NaN must not become zero: a player nobody has priced over the window
        is *unknown*, and calling that 'scores nothing from here' is exactly what
        makes releasing him look free."""
        got = cv.forward_rate(_frame({**MADUEKE, "Proj_Next3": np.nan}))
        assert pd.isna(got.iloc[0])

    def test_it_never_goes_negative(self):
        got = cv.forward_rate(_frame({**MADUEKE, "Proj_Next3": 1.0}))
        assert got.iloc[0] == 0.0


class TestWaiverFloors:
    def _pool(self, n, pos="M", base=5.0):
        return _frame(*[{"Display_Name": f"p{i}", "Position": pos, "Proj": 1.0,
                         "Proj_Next3": 1.0 + 2 * (base - i * 0.5)}
                        for i in range(n)])

    def test_it_takes_the_player_after_those_expected_to_go(self):
        """E midfielders go before your pick, so the (floor(E)+1)-th best is the
        one you can still expect to claim."""
        pool = self._pool(10)
        rates = {"M": 0.5}          # 5 ahead x 0.5 = 2.38-ish; use exactly 2
        floors = cv.waiver_floors(pool, n_ahead=4, claim_rates=rates)
        ranked = cv.forward_rate(pool).sort_values(ascending=False).tolist()
        assert floors["M"] == pytest.approx(ranked[2])   # floor(4*0.5) == 2

    def test_claiming_first_needs_no_history(self):
        floors = cv.waiver_floors(self._pool(10), n_ahead=0, claim_rates={})
        ranked = cv.forward_rate(self._pool(10)).max()
        assert floors["M"] == pytest.approx(ranked)

    # --- the three fail-closed gates ------------------------------------
    def test_a_thin_pool_has_no_floor(self):
        """Live there was exactly **one** available goalkeeper. A 'floor' there
        is that player's own projection presented as a market rate."""
        thin = self._pool(cv.MIN_FLOOR_POOL - 1, pos="G")
        assert "G" not in cv.waiver_floors(thin, 4, {"G": 0.5})

    def test_no_claim_history_means_no_floor(self):
        assert cv.waiver_floors(self._pool(10), n_ahead=4, claim_rates={}) == {}
        assert cv.waiver_floors(self._pool(10), n_ahead=4, claim_rates=None) == {}

    def test_an_absent_position_has_no_floor(self):
        """All 220 Rotowire forwards were owned, so F was genuinely unmeasurable."""
        assert "F" not in cv.waiver_floors(self._pool(10), 4, {"M": 0.5, "F": 0.5})

    def test_a_floor_of_zero_is_never_invented(self):
        """The whole point of the gates: a zero floor does not degrade the lane,
        it inverts it -- every release reads as free."""
        for args in ((None, 4, {"M": 0.5}), (self._pool(0), 4, {"M": 0.5})):
            assert cv.waiver_floors(*args) == {}


class TestReleaseCost:
    def test_a_player_the_wire_can_replace_is_free_to_release(self):
        """Iwobi's forward rate (3.64) is below the 3.67 MID floor, so the wire
        offers his equal -- yet his Keep Score of 0.80 made him undroppable."""
        got = cv.release_cost(_frame(IWOBI, GOMES), FLOORS)
        assert got.iloc[0] == 0.0
        assert got.iloc[1] == 0.0

    def test_a_better_player_costs_what_he_is_worth_above_the_floor(self):
        got = cv.release_cost(_frame(TEL), FLOORS)
        assert got.iloc[0] == pytest.approx((13.12 - 4.005) / 2 - 3.67, abs=1e-6)

    def test_no_floor_means_unknown_not_free(self):
        got = cv.release_cost(_frame({**TEL, "Position": "F"}), FLOORS)
        assert pd.isna(got.iloc[0])


class TestRentalGain:
    def test_the_reported_case(self):
        """The move the app would not suggest. Madueke over a replaceable drop is
        worth +3.53 points, and he was the best this-week play on the board."""
        got = cv.rental_gain(GOMES, MADUEKE, FLOORS)
        assert got == pytest.approx(5.371 - 1.840, abs=1e-6)
        assert got > 3.5

    def test_it_refuses_an_expensive_drop(self):
        """Tel costs 0.88 pts/GW to lose, so renting over him loses points even
        though the add outscores him this week. Refusing these is what makes the
        lane trustworthy."""
        assert MADUEKE["Proj"] > TEL["Proj"]
        assert cv.rental_gain(TEL, MADUEKE, FLOORS) < 0

    def test_a_marginal_drop_is_marginal(self):
        got = cv.rental_gain(IWOBI, MADUEKE, FLOORS)
        assert 0.9 < got < 1.1

    def test_the_adds_own_horizon_never_enters(self):
        """He goes back on the wire next week, so what follows cancels. Two adds
        with the same `Proj` and wildly different futures rent identically."""
        a = cv.rental_gain(GOMES, MADUEKE, FLOORS)
        b = cv.rental_gain(GOMES, {**MADUEKE, "Proj_Next3": 30.0}, FLOORS)
        assert a == pytest.approx(b)

    def test_no_floor_refuses_rather_than_assuming(self):
        assert cv.rental_gain({**GOMES, "Position": "F"}, MADUEKE, FLOORS) is None

    def test_an_unpriced_player_refuses(self):
        assert cv.rental_gain({**GOMES, "Proj_Next3": np.nan}, MADUEKE, FLOORS) is None
        assert cv.rental_gain(GOMES, {**MADUEKE, "Proj": np.nan}, FLOORS) is None


class TestKeeperValue:
    def test_the_floor_is_an_option_not_a_cap(self):
        """No player's value is bounded below by his own thin horizon, because
        you can always drop him and re-claim."""
        thin = {"Display_Name": "x", "Position": "M", "Proj": 5.0, "Proj_Next3": 5.0}
        got = cv.keeper_value(_frame(thin), FLOORS).iloc[0]
        assert got == pytest.approx(5.0 + 2 * 3.67)

    def test_it_reproduces_the_three_gameweek_ordering(self):
        """At H=3 the shipped ordering is recovered -- Bobb a hair above Madueke
        -- which is the evidence that the horizon, not the inputs, was the lever."""
        got = cv.keeper_value(_frame(MADUEKE, BOBB), FLOORS)
        assert got.iloc[1] > got.iloc[0]
        assert got.iloc[1] - got.iloc[0] < 0.05


class TestRecommendedLane:
    def test_a_thin_horizon_is_a_rental(self):
        assert cv.recommended_lane(MADUEKE, FLOORS) == cv.LANE_RENT

    def test_a_real_horizon_is_a_hold(self):
        assert cv.recommended_lane(BOBB, FLOORS) == cv.LANE_HOLD

    def test_it_falls_back_to_hold_when_it_cannot_tell(self):
        """A hold is advice about the player; a rental depends on being able to
        claim again, so 'cannot tell' must not become 'rent him'."""
        assert cv.recommended_lane({**MADUEKE, "Position": "F"}, FLOORS) == cv.LANE_HOLD
        assert cv.recommended_lane({**MADUEKE, "Proj_Next3": np.nan},
                                   FLOORS) == cv.LANE_HOLD


def test_the_module_is_streamlit_free():
    """The Actions collector and the offline suite must be able to import this
    without Streamlit, the same constraint `waiver_priority` carries."""
    import pathlib
    src = pathlib.Path(cv.__file__).read_text()
    assert "import streamlit" not in src
    assert "from scripts.common.analytics" not in src, (
        "analytics imports Streamlit; `_column` duplicates numeric_col's guard "
        "deliberately, per projection_sources._column")
