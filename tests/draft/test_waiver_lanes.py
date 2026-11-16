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
import scripts.draft.waiver_wire as ww
from scripts.draft.waiver_wire import _compute_transfer_suggestions
from scripts.draft.waiver_core import (
    LANE_HOLD,
    LANE_RENT,
    _dedupe_across_lanes,
    _hold_lane_improves_ros,
    _rent_lane_keeps_the_better_asset,
    _rentable_drops,
)


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


class TestTheCardNamesWhoseAbsenceItDependsOn:
    """The reported gap: `add_injury` reports the *add's* own fitness, and a
    deputy is typically perfectly fit. Chelsea's deputy goalkeeper was offered
    as a rest-of-season upgrade worth +0.26 pts/GW; the same move is worth
    -6.16 over the window if the first choice returns next week.
    """

    def _pool(self):
        """A full-pool frame of the kind `_attach_minutes_dependency` needs: the
        incumbent is injured, so he is absent from Rotowire's XI and therefore
        from the available pool by construction."""
        return pd.DataFrame([
            {"Player": "Emiliano Martínez Romero", "Display_Name": "Emiliano Martínez",
             "Team": "CHE", "Position": "G", "status": "i", "starts": 6,
             "chance_of_playing_next_round": 0,
             "news": "Head injury - Unknown return date"},
            {"Player": "Mike Penders", "Display_Name": "Mike Penders", "Team": "CHE",
             "Position": "G", "status": "a", "starts": 2,
             "chance_of_playing_next_round": None, "news": ""},
            {"Player": "Robin Roefs", "Display_Name": "Robin Roefs", "Team": "SUN",
             "Position": "G", "status": "a", "starts": 9,
             "chance_of_playing_next_round": None, "news": ""},
        ])

    def _avail(self):
        return pd.DataFrame([
            {"Player": "Mike Penders", "Display_Name": "Mike Penders", "Team": "CHE",
             "Position": "G", "starts": 2, "Proj": 3.63, "Proj_Next3": 10.57},
        ])

    def test_it_names_the_absent_incumbent(self):
        got = ww._attach_minutes_dependency(self._avail(), self._pool(), current_gw=10)
        assert got.loc[0, "Dep_Blocker"] == "Emiliano Martínez"
        assert got.loc[0, "Dep_Weeks"] > 0

    def test_it_reports_that_the_duration_was_not_stated(self):
        """The distinction the card turns on: "~5 GW" from a bucket is not the
        same claim as "2 GW, back 22 Nov"."""
        got = ww._attach_minutes_dependency(self._avail(), self._pool(), current_gw=10)
        assert got.loc[0, "Dep_Source"] != "news"

    def test_a_stated_return_is_reported_as_stated(self):
        pool = self._pool()
        pool.loc[0, "news"] = "Head injury - Expected back 22 Nov"
        got = ww._attach_minutes_dependency(self._avail(), pool, current_gw=10)
        assert got.loc[0, "Dep_Source"] == "news"

    def test_a_departure_is_not_an_absence(self):
        """`status` 'u' covers a loan and a permanent exit as well as an injury,
        and a transferred player is not a man his deputy is keeping warm for."""
        pool = self._pool()
        pool.loc[0, "status"] = "u"
        pool.loc[0, "news"] = "Has joined Al Hilal permanently"
        got = ww._attach_minutes_dependency(self._avail(), pool, current_gw=10)
        assert got.loc[0, "Dep_Blocker"] == ""

    def test_a_regular_starter_depends_on_nobody(self):
        avail = self._avail()
        avail.loc[0, "starts"] = 8          # 8/9 -- a regular, not a deputy
        got = ww._attach_minutes_dependency(avail, self._pool(), current_gw=10)
        assert got.loc[0, "Dep_Blocker"] == ""

    def test_an_unrelated_club_is_not_a_blocker(self):
        avail = self._avail()
        avail.loc[0, "Team"] = "SUN"
        got = ww._attach_minutes_dependency(avail, self._pool(), current_gw=10)
        assert got.loc[0, "Dep_Blocker"] == ""

    def test_it_degrades_rather_than_raising_on_a_bare_frame(self):
        got = ww._attach_minutes_dependency(self._avail(), pd.DataFrame(), current_gw=10)
        assert (got["Dep_Blocker"] == "").all()


class TestTheThinWireWarning:
    def test_a_position_with_no_floor_is_flagged(self):
        """The same fact `waiver_floors` refused to quote a number for, from the
        manager's side: live there was one available goalkeeper, so a GK claim
        that goes wrong cannot be undone. The hold lane has no floor gate of its
        own -- it recommended the deputy keeper anyway."""
        avail = pd.DataFrame([
            {"Player": "Mike Penders", "Display_Name": "Mike Penders", "Team": "CHE",
             "Position": "G", "Proj": 3.63, "Proj_Next3": 10.57, "Points": 3.6,
             "Transfer Score": 0.70, "Season_Points": 8, "status": "a",
             "chance_of_playing_next_round": 100, "Draft_State": "a"},
        ])
        roster = pd.DataFrame([
            {"Player": "Robin Roefs", "Display_Name": "Robin Roefs", "Team": "SUN",
             "Position": "G", "Proj": 3.10, "Proj_Next3": 9.79,
             "Projected_Points": 3.0, "Keep Score": 0.40, "Season_Points": 30,
             "status": "a", "chance_of_playing_next_round": 100},
            {"Player": "Caoimhín Kelleher", "Display_Name": "Caoimhín Kelleher",
             "Team": "BRE", "Position": "G", "Proj": 3.01, "Proj_Next3": 10.06,
             "Projected_Points": 3.2, "Keep Score": 0.64, "Season_Points": 28,
             "status": "a", "chance_of_playing_next_round": 100},
        ])
        sg, _ = _compute_transfer_suggestions(
            avail, roster, top_n=None, positions=["G"], roster_candidates=None,
            avail_candidates=None, one_per_position=False, waiver_context={},
            floor_context=FLOOR_CTX)
        assert sg, "a GK upgrade should still be offered"
        assert all(s["thin_wire"] for s in sg), (
            "one available GK must be flagged as hard to undo")


class TestItSurvivesTheFramesThePageActuallyBuilds:
    """Regression: this took the Waiver Wire down with
    ``AttributeError: 'numpy.float64' object has no attribute 'fillna'``.

    Two faults compounding. ``pd.to_numeric(df.get("starts"), ...).fillna(0)``
    is the idiom CLAUDE.md documents as a latent crash -- ``DataFrame.get``
    returns **None** for a missing column, so the ``.fillna`` is called on a
    scalar. And the callsite sat four lines *before* the page merges ``starts``
    onto the available pool, so the column really was absent.

    The tests above all passed because they fed the function an already-enriched
    frame. **A frame the page never produces at that point is not a fixture** --
    these assert on the shapes it does.
    """

    def _pool(self):
        return pd.DataFrame([
            {"Player": "Emiliano Martínez Romero", "Display_Name": "Emiliano Martínez",
             "Team": "CHE", "Position": "G", "status": "i", "starts": 6,
             "chance_of_playing_next_round": 0,
             "news": "Head injury - Expected back 22 Nov"},
            {"Player": "Mike Penders", "Display_Name": "Mike Penders", "Team": "CHE",
             "Position": "G", "status": "a", "starts": 2,
             "chance_of_playing_next_round": None, "news": ""},
        ])

    def test_an_available_frame_with_no_starts_column(self):
        """Exactly the pre-merge shape, which is what crashed."""
        avail = pd.DataFrame([
            {"Player": "Mike Penders", "Display_Name": "Mike Penders",
             "Team": "CHE", "Position": "G", "Proj": 3.63, "Proj_Next3": 10.57},
        ])
        assert "starts" not in avail.columns
        got = ww._attach_minutes_dependency(avail, self._pool(), current_gw=10)
        assert len(got) == 1                      # no raise is the assertion

    def test_a_reference_pool_with_no_starts_column(self):
        pool = self._pool().drop(columns=["starts"])
        avail = pd.DataFrame([
            {"Player": "Mike Penders", "Display_Name": "Mike Penders", "Team": "CHE",
             "Position": "G", "starts": 2, "Proj": 3.63, "Proj_Next3": 10.57},
        ])
        got = ww._attach_minutes_dependency(avail, pool, current_gw=10)
        assert (got["Dep_Blocker"] == "").all(), "no depth chart, no opinion"

    def test_no_status_or_news_column(self):
        avail = pd.DataFrame([
            {"Player": "x", "Display_Name": "x", "Team": "CHE", "Position": "G",
             "starts": 2, "Proj": 1.0, "Proj_Next3": 3.0},
        ])
        got = ww._attach_minutes_dependency(
            avail, self._pool().drop(columns=["news"]), current_gw=10)
        assert len(got) == 1

    def test_the_builder_attaches_it_after_the_starts_merge(self):
        """The ordering fault, pinned in the source. The depth rule reads season
        starts to tell a deputy from a regular, so attaching before the merge
        treats every available player as having none.

        Pinned on `build_waiver_board`, which is where the pipeline lives since
        the claim engine was split out of the page. The builder also asserts it
        at runtime, but an assertion only fires on a frame that reaches it --
        this catches the lines being reordered at all."""
        import inspect
        import scripts.draft.waiver_core as wc
        src = inspect.getsource(wc.build_waiver_board)
        i_starts = src.index("avail_all['starts'] = numeric_col")
        i_dep = src.index("_attach_minutes_dependency(avail_all")
        assert i_starts < i_dep, (
            "_attach_minutes_dependency must run after the starts merge")


class TestLaneVetoes:
    """Four bad suggestions reported from the live board, each pinned.

    They share one root: the engine models "what do I gain" well and "what do I
    lose" poorly. Every number on every one of these cards was individually
    plausible.
    """

    @staticmethod
    def _row(**kw):
        base = dict(Player="X", Position="D", ROS=0.5, Proj=3.0, Proj_Next3=9.0,
                    Season_Points=20, **{"Keep Score": 0.5, "Transfer Score": 0.5})
        base.update(kw)
        return pd.Series(base)

    def test_hold_refuses_an_add_that_is_worse_rest_of_season(self):
        """Igor Thiago -> Callum Wilson.

        Thiago: elite forward, suspended until 5 Dec, so the engine correctly
        wrote both his projections to zero. Wilson: 1.3 points across the next
        three gameweeks and ROS 0.19. Every observable the sanity veto compares
        favoured Wilson; only the rest-of-season view could see it.
        """
        thiago = self._row(Player="Igor Thiago", ROS=0.38, Proj=0.0,
                           Proj_Next3=0.0, Season_Points=30)
        wilson = self._row(Player="Callum Wilson", ROS=0.19, Proj=3.23,
                           Proj_Next3=1.27, Season_Points=6)
        assert not _hold_lane_improves_ros(thiago, wilson)

    def test_hold_allows_a_genuine_rest_of_season_upgrade(self):
        hume = self._row(Player="Trai Hume", ROS=0.64, Proj=2.37, Proj_Next3=7.24)
        collins = self._row(Player="Nathan Collins", ROS=0.70, Proj=3.60,
                            Proj_Next3=10.80)
        assert _hold_lane_improves_ros(hume, collins)

    def test_hold_passes_when_either_side_has_no_ros(self):
        """No comparable data is not a finding -- the veto must not fire on a
        degraded feed, which would suppress every suggestion at once."""
        a = self._row(ROS=float("nan"))
        b = self._row(ROS=0.9)
        assert _hold_lane_improves_ros(a, b)
        assert _hold_lane_improves_ros(b, a)

    def test_rent_refuses_to_give_up_the_better_window(self):
        """James Tarkowski -> Abdukodir Khusanov.

        A rental lasts a week and the drop is permanent, so trading 7.1 points
        over three gameweeks for 6.1 to gain 0.8 this week is a loss however
        cheap `release_cost` claims the release is. The percentile gap on this
        pair was already negative.
        """
        tark = self._row(Player="James Tarkowski", Proj=2.62, Proj_Next3=7.09)
        khusanov = self._row(Player="Abdukodir Khusanov", Proj=3.42,
                             Proj_Next3=6.08)
        assert not _rent_lane_keeps_the_better_asset(tark, khusanov)

    def test_rent_allows_a_rental_that_upgrades_the_window_too(self):
        muharemovic = self._row(Proj=2.52, Proj_Next3=7.86)
        pinnock = self._row(Proj=3.40, Proj_Next3=8.44)
        assert _rent_lane_keeps_the_better_asset(muharemovic, pinnock)

    def test_rent_passes_when_the_horizon_is_unknown(self):
        a = self._row(Proj_Next3=float("nan"))
        b = self._row(Proj_Next3=9.0)
        assert _rent_lane_keeps_the_better_asset(a, b)

    def test_one_card_per_pair_and_hold_wins(self):
        """Hume -> Collins was rendering in both lanes.

        Hold is the stronger claim: "worth keeping" subsumes "worth renting",
        and the rent card would otherwise invite a lasting upgrade followed by
        handing the player back.
        """
        hold = {"lane": LANE_HOLD, "drop_player": "Trai Hume",
                "add_player": "Nathan Collins"}
        rent = {"lane": LANE_RENT, "drop_player": "Trai Hume",
                "add_player": "Nathan Collins"}
        other = {"lane": LANE_RENT, "drop_player": "Alex Iwobi",
                 "add_player": "Ethan Pinnock"}
        kept = _dedupe_across_lanes([hold, rent, other])
        assert [k["lane"] for k in kept] == [LANE_HOLD, LANE_RENT]
        assert kept[0]["drop_player"] == "Trai Hume"
        assert kept[1]["drop_player"] == "Alex Iwobi"

    def test_a_different_add_for_the_same_drop_is_not_a_duplicate(self):
        a = {"lane": LANE_HOLD, "drop_player": "Trai Hume", "add_player": "A"}
        b = {"lane": LANE_RENT, "drop_player": "Trai Hume", "add_player": "B"}
        assert len(_dedupe_across_lanes([a, b])) == 2


class TestRentLaneDropSelection:
    """Which player the rent lane offers up, not which one it brings in.

    Reported from the app: "Enzo Le Fee is not the person that should be dropped
    for Rayan Vitor. I have better options that could be dropped instead." He
    was the *fourth* weakest of five midfielders and still won the pairing.

    Two causes, both pinned here. The live numbers for adding Vitor, by drop:

        Tel        keep 0.623  ROS 0.70  gain -1.71  (rejected)
        Madueke    keep 0.668  ROS 0.58  gain +0.58
        Iwobi      keep 0.735  ROS 0.81  gain +1.24
        Le Fee     keep 0.773  ROS 0.82  gain +1.46  <- chosen
        Dewsbury-Hall keep 0.820 ROS 0.86 gain +0.48

    The gain rises with how much you want to keep the player.
    """

    @staticmethod
    def _roster():
        """Five midfielders with the live Keep Score / projection shape."""
        return pd.DataFrame([
            dict(Player="Mathys Tel", Position="M", Proj=2.16, Proj_Next3=13.77,
                 ROS=0.70, Season_Points=24,
                 **{"Keep Score": 0.623, "Transfer Score": 0.60}),
            dict(Player="Noni Madueke", Position="M", Proj=4.44, Proj_Next3=10.62,
                 ROS=0.58, Season_Points=22,
                 **{"Keep Score": 0.668, "Transfer Score": 0.65}),
            dict(Player="Alex Iwobi", Position="M", Proj=3.08, Proj_Next3=10.82,
                 ROS=0.81, Season_Points=40,
                 **{"Keep Score": 0.735, "Transfer Score": 0.72}),
            dict(Player="Enzo Le Fee", Position="M", Proj=3.56, Proj_Next3=10.40,
                 ROS=0.82, Season_Points=46,
                 **{"Keep Score": 0.773, "Transfer Score": 0.76}),
            dict(Player="Kiernan Dewsbury-Hall", Position="M", Proj=3.72,
                 Proj_Next3=11.58, ROS=0.86, Season_Points=50,
                 **{"Keep Score": 0.820, "Transfer Score": 0.80}),
        ])

    @staticmethod
    def _avail():
        rows = [dict(Player="Rayan Vitor", Position="M", Proj=5.02,
                     Proj_Next3=14.34, ROS=0.88, Season_Points=30,
                     **{"Keep Score": 0.5, "Transfer Score": 0.88})]
        # A floor needs MIN_FLOOR_POOL players to be quoted at all.
        rows += [dict(Player="Filler %d" % i, Position="M", Proj=3.4 - i * 0.1,
                      Proj_Next3=10.5 - i * 0.2, ROS=0.5, Season_Points=15,
                      **{"Keep Score": 0.4, "Transfer Score": 0.5})
                 for i in range(8)]
        return pd.DataFrame(rows)

    def _run(self, **kw):
        opts = dict(roster_candidates=2, avail_candidates=5,
                    one_per_position=False, lanes=(LANE_RENT,),
                    floor_context=FLOOR_CTX, top_n=None)
        opts.update(kw)
        sugg, _dbg = _compute_transfer_suggestions(
            self._avail(), self._roster(), **opts)
        return [s for s in sugg if s["lane"] == LANE_RENT]

    def test_it_offers_up_a_player_you_would_actually_drop(self):
        """The rent lane used to iterate the *whole* position.

        The hold lane has always windowed its drop candidates to the weakest N;
        the rent lane did not, so a player who was never a drop candidate in
        Hold at all could be offered here.
        """
        rent = self._run()
        assert rent, "the lane should still find a rental"
        dropped = {s["drop_player"] for s in rent}
        assert "Enzo Le Fee" not in dropped
        assert "Kiernan Dewsbury-Hall" not in dropped

    def test_among_competing_drops_it_releases_the_one_you_value_least(self):
        """`release_cost` floors at zero, and then the ranking inverts.

        Every roster player whose forward rate sits below the wire's floor looks
        *equally* free to release. Once it saturates, `rental_gain` is driven by
        `-Proj(drop)` -- whoever contributes least **this week** wins the
        pairing, which is a different question from who you would rather keep.
        """
        rent = self._run(roster_candidates=None)
        vitor = [s for s in rent if s["add_player"] == "Rayan Vitor"]
        assert vitor, "Vitor should still be claimed by somebody"
        assert vitor[0]["drop_player"] == "Noni Madueke", (
            "expected the weakest viable midfielder, got %s"
            % vitor[0]["drop_player"])

    def test_the_biggest_gain_no_longer_wins_outright(self):
        """Le Fee's pairing yields more points and is still not the one taken.

        Every candidate has already cleared `RENTAL_MIN_GAIN`, so each is worth
        doing on its own terms; the only question left is which player to give
        up, and a bigger number against a better player is not a better answer.
        """
        rent = self._run(roster_candidates=None)
        by_drop = {s["drop_player"]: s["points_gain"] for s in rent}
        if "Enzo Le Fee" in by_drop and "Noni Madueke" in by_drop:
            assert by_drop["Noni Madueke"] <= by_drop["Enzo Le Fee"]
        vitor = [s for s in rent if s["add_player"] == "Rayan Vitor"][0]
        assert vitor["drop_player"] == "Noni Madueke"


class TestARentalIsFundedFromTheWeakerHalf:
    """Reported from the app, twice: "Tarkowski is an ELITE def. I am NOT
    dropping him, even if another player has a better projection for a couple
    GWs."

    He kept reappearing because he is genuinely mid-table *by Keep Score* --
    6th of 15 -- so every rank-based drop window still included him, and the
    engine's own tiers put his ROS of 0.73 below the 0.75 "strong" band, into
    the churn zone where short-term value is meant to dominate.

    A rental buys one gameweek and costs a roster spot permanently, so the man
    you give up has to be one you were close to moving on anyway.
    """

    #: The live roster, ROS only. Median 0.706.
    ROSTER = [("Igor Thiago", "F", 0.375), ("Taiwo Awoniyi", "F", 0.457),
              ("Noni Madueke", "M", 0.585), ("Robin Roefs", "G", 0.585),
              ("Trai Hume", "D", 0.635), ("Caoimhin Kelleher", "G", 0.687),
              ("Mathys Tel", "M", 0.700), ("Ian Maatsen", "D", 0.706),
              ("James Tarkowski", "D", 0.732), ("Tarik Muharemovic", "D", 0.754),
              ("Oli McBurnie", "F", 0.793), ("Alex Iwobi", "M", 0.806),
              ("Antonio Joao Silva", "D", 0.818), ("Enzo Le Fee", "M", 0.824),
              ("Kiernan Dewsbury-Hall", "M", 0.862)]

    def _roster(self):
        return pd.DataFrame([dict(Player=n, Position=p, ROS=r)
                             for n, p, r in self.ROSTER])

    def test_the_bar_is_the_squads_own_median(self):
        roster = self._roster()
        eligible = set(_rentable_drops(roster, roster)["Player"])
        assert "James Tarkowski" not in eligible
        assert "Enzo Le Fee" not in eligible
        assert "Tarik Muharemovic" not in eligible
        assert "Noni Madueke" in eligible
        assert "Trai Hume" in eligible

    def test_it_is_relative_rather_than_an_absolute_band(self):
        """"Elite" on a strong roster is not "elite" on a weak one.

        The same player is protected in one squad and available in another,
        which is right: the question is only ever which of *your* players to
        give up. An absolute threshold would have to be tuned until one named
        player fell the right side of it.
        """
        weak = pd.DataFrame([dict(Player="A", Position="D", ROS=0.20),
                             dict(Player="B", Position="D", ROS=0.30),
                             dict(Player="Tarkowski", Position="D", ROS=0.73)])
        assert "Tarkowski" not in set(_rentable_drops(weak, weak)["Player"])
        strong = pd.DataFrame([dict(Player="A", Position="D", ROS=0.90),
                               dict(Player="B", Position="D", ROS=0.95),
                               dict(Player="Tarkowski", Position="D", ROS=0.73)])
        assert "Tarkowski" in set(_rentable_drops(strong, strong)["Player"])

    def test_a_position_entirely_above_the_median_funds_no_rental(self):
        """Which is a real answer, not a gap to be filled."""
        roster = self._roster()
        defs = roster[roster["Position"] == "D"]
        elite_only = defs[defs["ROS"] > 0.70]
        assert _rentable_drops(roster, elite_only).empty

    def test_it_falls_open_without_ros(self):
        """A lane that silently stops working is worse than one occasionally
        too generous."""
        roster = pd.DataFrame([dict(Player="A", Position="D"),
                               dict(Player="B", Position="D")])
        assert len(_rentable_drops(roster, roster)) == 2
        blank = self._roster().assign(ROS=float("nan"))
        assert len(_rentable_drops(blank, blank)) == len(self.ROSTER)
