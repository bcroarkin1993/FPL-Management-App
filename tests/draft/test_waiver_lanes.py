"""The rent/hold lanes on the Draft Waiver Wire.

The reported defect: Noni Madueke was the highest-projected available player in
the game at GW10 (`Proj` 5.371, top of the board) and was suggested nowhere. He
ranked 4th at midfield on `Transfer Score` -- which is ~60% ROS in Draft, and he
had missed seven gameweeks injured -- and his one threshold-clearing pairing was
then taken off him by a rival add under the one-add-one-drop assignment.

These fixtures reproduce that shape with the live numbers.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common import claim_value as cv
from scripts.draft.waiver_wire import _compute_transfer_suggestions


FLOOR_CTX = {"n_ahead": 5, "claim_rates": {"M": 0.475, "D": 0.25}}


def _avail():
    """Nine available midfielders: the rental target plus a padded pool, since a
    floor needs MIN_FLOOR_POOL players to be measurable at all."""
    rows = [
        # The reported player: best this week, thin horizon.
        {"Player": "Noni Madueke", "Display_Name": "Noni Madueke", "Team": "ARS",
         "Position": "M", "Proj": 5.371, "Proj_Next3": 11.50, "Points": 5.56,
         "Transfer Score": 0.713, "Season_Points": 14, "status": "a",
         "chance_of_playing_next_round": 100, "Draft_State": "a"},
        # The player the shipped score preferred: mid on both.
        {"Player": "Oscar Bobb", "Display_Name": "Oscar Bobb", "Team": "FUL",
         "Position": "M", "Proj": 4.610, "Proj_Next3": 12.74, "Points": 4.26,
         "Transfer Score": 0.779, "Season_Points": 32, "status": "a",
         "chance_of_playing_next_round": 100, "Draft_State": "a"},
    ]
    for i in range(7):
        rows.append({"Player": f"filler{i}", "Display_Name": f"filler{i}",
                     "Team": "BHA", "Position": "M", "Proj": 2.0,
                     "Proj_Next3": 6.0 + i * 0.2, "Points": 2.0,
                     "Transfer Score": 0.30, "Season_Points": 10, "status": "a",
                     "chance_of_playing_next_round": 100, "Draft_State": "a"})
    return pd.DataFrame(rows)


def _roster():
    """Two midfielders: one the wire can replace, one it cannot."""
    return pd.DataFrame([
        # Forward rate 2.00 -- below any plausible floor, so free to release.
        {"Player": "Joao Gomes", "Display_Name": "Joao Gomes", "Team": "AVL",
         "Position": "M", "Proj": 1.840, "Proj_Next3": 5.84,
         "Projected_Points": 1.5, "Keep Score": 0.454, "Season_Points": 12,
         "status": "a", "chance_of_playing_next_round": 100},
        # Forward rate 4.56 -- clearly above it, so expensive to lose.
        {"Player": "Mathys Tel", "Display_Name": "Mathys Tel", "Team": "TOT",
         "Position": "M", "Proj": 4.005, "Proj_Next3": 13.12,
         "Projected_Points": 3.85, "Keep Score": 0.687, "Season_Points": 23,
         "status": "a", "chance_of_playing_next_round": 100},
    ])


def _run(**kw):
    kw.setdefault("floor_context", FLOOR_CTX)
    kw.setdefault("positions", ["M"])
    kw.setdefault("roster_candidates", None)
    kw.setdefault("avail_candidates", None)
    kw.setdefault("one_per_position", False)
    kw.setdefault("top_n", None)
    kw.setdefault("waiver_context", {})
    sg, dbg = _compute_transfer_suggestions(_avail(), _roster(), **kw)
    return sg


class TestTheReportedCase:
    def test_the_rental_lane_surfaces_the_best_this_week_play(self):
        """He is 4th on `Transfer Score` and 1st on `Proj`, which is why the hold
        lane never reaches him and the rental lane leads with him."""
        rent = [s for s in _run() if s["lane"] == cv.LANE_RENT]
        assert rent, "the rental lane found nothing"
        assert rent[0]["add_player"] == "Noni Madueke"
        assert rent[0]["points_gain"] > 3.0

    def test_the_hold_lane_does_not_reach_him(self):
        """Not a bug -- the hold lane is ranked on durable value, and a player
        with three starts behind him genuinely has little. It is why the rental
        lane has to exist rather than the hold lane being retuned."""
        hold = [s for s in _run(lanes=(cv.LANE_HOLD,)) if s["lane"] == cv.LANE_HOLD]
        assert all(s["add_player"] != "Noni Madueke" for s in hold)

    def test_the_rental_lane_ranks_on_proj_not_transfer_score(self):
        """The whole reason it is a separate pass. `avail_iter` is the top few by
        `Transfer Score`; the player worth renting is by definition the one with a
        big this-week number and a thin future."""
        rent = [s for s in _run(avail_candidates=1) if s["lane"] == cv.LANE_RENT]
        assert rent and rent[0]["add_player"] == "Noni Madueke", (
            "with a one-player window the rental lane must still pick the "
            "highest-Proj add, not the highest-Transfer-Score one")


class TestItRefusesExpensiveDrops:
    def test_an_expensive_drop_is_never_rented_over(self):
        """Tel costs ~0.9 pts/GW to lose. Refusing these is what makes the lane
        trustworthy -- without it a rental would propose dropping your best
        player for a one-week punt."""
        rent = [s for s in _run() if s["lane"] == cv.LANE_RENT]
        assert all(s["drop_player"] != "Mathys Tel" for s in rent)

    def test_every_rental_gain_is_positive(self):
        for s in _run():
            if s["lane"] == cv.LANE_RENT:
                assert s["points_gain"] > 0


class TestFailClosed:
    def test_no_claim_history_disables_the_rental_lane(self):
        """No floor means every release would read as free, which does not
        degrade the lane but inverts it."""
        sg = _run(floor_context={})
        assert not [s for s in sg if s["lane"] == cv.LANE_RENT]

    def test_a_thin_pool_disables_it_at_that_position(self):
        thin = _avail().head(cv.MIN_FLOOR_POOL - 1)
        sg, _ = _compute_transfer_suggestions(
            thin, _roster(), top_n=None, positions=["M"], roster_candidates=None,
            avail_candidates=None, one_per_position=False, waiver_context={},
            floor_context=FLOOR_CTX)
        assert not [s for s in sg if s["lane"] == cv.LANE_RENT]

    def test_every_rental_card_carries_its_floor(self):
        """A rental priced without a floor is the silent-gate failure this app
        has paid for four times."""
        for s in _run():
            if s["lane"] == cv.LANE_RENT:
                assert s["position_floor"] is not None
                assert s["release_cost"] is not None


class TestLaneSelection:
    def test_each_lane_can_be_requested_alone(self):
        assert {s["lane"] for s in _run(lanes=(cv.LANE_RENT,))} <= {cv.LANE_RENT}
        assert {s["lane"] for s in _run(lanes=(cv.LANE_HOLD,))} <= {cv.LANE_HOLD}

    def test_both_lanes_by_default(self):
        assert {s["lane"] for s in _run()} == {cv.LANE_HOLD, cv.LANE_RENT}

    def test_every_suggestion_is_labelled(self):
        """The renderer and the claim plan both split on `lane`, so an unlabelled
        card would silently land in the hold lane."""
        for s in _run():
            assert s["lane"] in (cv.LANE_HOLD, cv.LANE_RENT)

    def test_hold_cards_sort_before_rental_cards(self):
        lanes = [s["lane"] for s in _run()]
        assert lanes == sorted(lanes, key=lambda l: l != cv.LANE_HOLD)

    def test_top_n_is_applied_per_lane(self):
        """A flat slice lets the hold lane fill the whole quota and the rental
        lane vanish -- the exact failure this feature fixes, re-created by the
        display limit."""
        sg = _run(top_n=1)
        assert len([s for s in sg if s["lane"] == cv.LANE_RENT]) == 1
        assert len([s for s in sg if s["lane"] == cv.LANE_HOLD]) <= 1
