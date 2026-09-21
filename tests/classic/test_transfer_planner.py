"""The Classic transfer planner.

What it replaces could not propose the move that makes multiple free transfers
worth having: sell a premium and a mid-price player, buy a premium somewhere
else and a cheaper replacement. Three things stopped it, and
`TestObjectiveIsPoints` pins all three.

Its drops came from `nsmallest(6, "Keep Score")`, so a premium was never a
candidate to sell. Its objective summed positional *percentiles*, which
saturate near the top -- Haaland's 213.7 season points against a mid-price
midfielder's 178.3 is 0.974 against 0.977, so the premium ranks *lower* -- and
invert across positions. And it was fixed at two legs.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.classic.transfers import (
    _annotate_suggestions_against_plan,
    _build_multi_transfer_plan,
    _plan_leg_rationale,
    build_plan_pool,
    build_plan_scores,
    build_transfer_plan,
    plan_horizon,
)


def _player(pid, pos, team, price, proj, next3=None, ffp=True, **kw):
    row = {
        "Player_ID": pid, "Player": "P%d" % pid, "Position": pos, "Team": team,
        "now_cost": int(round(price * 10)), "selling_price": int(round(price * 10)),
        "Proj": proj, "Proj_Next3": next3 if next3 is not None else proj * 3,
        "FFP_Starting_Predicted": proj if ffp else np.nan,
        "status": "a", "minutes": 900, "chance_of_playing_next_round": None,
    }
    row.update(kw)
    return row


def _squad_and_pool(squad_rows, extra_rows):
    squad = pd.DataFrame(squad_rows)
    pool = pd.DataFrame(squad_rows + extra_rows)
    return squad, pool


def _base_squad(proj=4.0, price=5.0):
    """A legal 2/5/5/3, one club each so the 3-per-club rule never bites."""
    rows, pid = [], 0
    for pos, n in (("G", 2), ("D", 5), ("M", 5), ("F", 3)):
        for _ in range(n):
            rows.append(_player(pid, pos, "T%d" % pid, price, proj))
            pid += 1
    return rows


def _filler(start_pid, proj=3.0, price=4.5):
    """Bench-grade free agents at every position, so a legal squad always exists."""
    rows, pid = [], start_pid
    for pos in "GDMF":
        for _ in range(4):
            rows.append(_player(pid, pos, "X%d" % pid, price, proj))
            pid += 1
    return rows


def _plan(squad_rows, extra_rows, bank_m=0.0, ft=1, hits=0, w_now=0.4, w_next3=0.6):
    squad, pool_src = _squad_and_pool(squad_rows, extra_rows)
    pool = build_plan_pool(pool_src, squad, w_now, w_next3)
    return build_transfer_plan(pool, squad, int(round(bank_m * 10)),
                               free_transfers=ft, max_extra_hits=hits,
                               w_now=w_now, w_next3=w_next3)


class TestHorizon:
    def test_pure_this_gameweek_is_one(self):
        assert plan_horizon(1.0, 0.0) == pytest.approx(1.0)

    def test_pure_next_three_is_three(self):
        assert plan_horizon(0.0, 1.0) == pytest.approx(3.0)

    def test_the_default_split_sits_between(self):
        assert 1.0 < plan_horizon(0.4, 0.6) < 3.0


class TestPlanScores:
    def test_next3_is_normalised_to_a_per_gameweek_rate(self):
        """A player whose 3-gameweek total is exactly 3x his gameweek must
        score the same at every split, or the horizon term is a hidden 3x."""
        df = pd.DataFrame([_player(1, "M", "T1", 7.0, 5.0, next3=15.0)])
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert build_plan_scores(df, w, 1 - w).iloc[0] == pytest.approx(5.0)

    def test_a_fixture_run_moves_the_score(self):
        df = pd.DataFrame([_player(1, "M", "T1", 7.0, 5.0, next3=21.0)])
        assert build_plan_scores(df, 1.0, 0.0).iloc[0] == pytest.approx(5.0)
        assert build_plan_scores(df, 0.0, 1.0).iloc[0] == pytest.approx(7.0)

    def test_the_horizon_term_is_taken_as_given(self):
        """Correcting `Proj_Next3`'s basis is the engine's job, not this
        function's.

        This used to trust the horizon term only where FFP had matched and fall
        back to `Proj` otherwise -- a workaround for the fallbacks reaching
        `Proj_Next3` on a conditional basis. That is fixed upstream now (see
        `per_source_next3_basis`), so a player Rotowire priced but FFP did not
        keeps his fixture information instead of collapsing to a flat rate.
        Re-introducing the workaround here would double-count the correction.
        """
        df = pd.DataFrame([_player(1, "M", "T1", 7.0, 2.0, next3=18.0, ffp=False)])
        assert build_plan_scores(df, 0.0, 1.0).iloc[0] == pytest.approx(6.0)

    def test_a_missing_horizon_value_falls_back_to_the_gameweek_projection(self):
        """No multi-gameweek number at all is different from a low one: it
        means nothing published a window for him, not that the window is bad."""
        df = pd.DataFrame([_player(1, "M", "T1", 7.0, 2.0)])
        df["Proj_Next3"] = np.nan
        assert build_plan_scores(df, 0.0, 1.0).iloc[0] == pytest.approx(2.0)

    def test_a_missing_projection_scores_zero_rather_than_nan(self):
        df = pd.DataFrame([{"Player_ID": 1, "Position": "M"}])
        assert np.isfinite(build_plan_scores(df, 0.4, 0.6)).all()


class TestChangeCount:
    @pytest.mark.parametrize("ft", [1, 2, 3])
    def test_never_more_legs_than_transfers_allowed(self, ft):
        squad = _base_squad()
        better = [_player(90 + i, "M", "Y%d" % i, 5.0, 9.0) for i in range(4)]
        plan = _plan(squad, better + _filler(200), ft=ft)
        assert len(plan["legs"]) <= ft

    def test_an_optimal_squad_is_reported_as_a_hold(self):
        """A valuable answer, not a blank."""
        squad = _base_squad(proj=9.0)
        plan = _plan(squad, _filler(200), ft=2)
        assert plan.get("hold") is True

    def test_an_owned_player_with_no_projection_is_still_sellable(self):
        """The ILP pre-filter drops score <= 0; an owner dropped from the pool
        makes his sale free and uncounted against K."""
        squad = _base_squad()
        squad[7]["Proj"] = 0.0
        squad[7]["Proj_Next3"] = 0.0
        squad[7]["now_cost"] = squad[7]["selling_price"] = 90  # above the 4.5 escape hatch
        better = [_player(90, "M", "Y0", 9.0, 9.0)]
        plan = _plan(squad, better + _filler(200), bank_m=5.0, ft=1)
        assert not plan.get("error")
        assert len(plan.get("legs", [])) <= 1


class TestObjectiveIsPoints:
    def test_a_premium_can_be_sold_to_fund_a_premium_elsewhere(self):
        """The motivating case, and the one the percentile objective could not
        reach: the forward sale is a downgrade on its own leg and the pair is
        a clear gain."""
        squad = _base_squad()
        # Make one forward a £14.0m premium worth 8.0, and one midfielder a
        # £6.0m 4.0. Neither is weak, so neither would be a "worst by Keep
        # Score" drop candidate.
        fwd = next(r for r in squad if r["Position"] == "F")
        fwd.update(_player(fwd["Player_ID"], "F", fwd["Team"], 14.0, 8.0))
        mid = next(r for r in squad if r["Position"] == "M")
        mid.update(_player(mid["Player_ID"], "M", mid["Team"], 6.0, 4.0))

        # A £14.0m midfielder worth 12.0 and a £6.0m forward worth 6.5. Selling
        # the premium forward and the mid funds both, for +6.5 a gameweek.
        extra = [_player(90, "M", "Y0", 14.0, 12.0),
                 _player(91, "F", "Y1", 6.0, 6.5)]
        plan = _plan(squad, extra + _filler(200), bank_m=0.0, ft=2)

        assert not plan.get("hold"), "planner declined a clearly winning move"
        sold = {leg["out_id"] for leg in plan["legs"]}
        bought = {leg["in_id"] for leg in plan["legs"]}
        assert fwd["Player_ID"] in sold, "a premium was never offered as a sale"
        assert 90 in bought, "the premium midfielder was not bought"
        assert plan["gain_net"] > 0

    def test_the_whole_squad_is_a_drop_candidate(self):
        """Not `nsmallest(6, "Keep Score")` -- the best player you own has to
        be sellable, or the reallocation above is unreachable."""
        squad = _base_squad(proj=4.0)
        squad[0]["Proj"] = 9.0  # the best player in the squad, a goalkeeper
        squad[0]["Proj_Next3"] = 27.0
        squad[0]["FFP_Starting_Predicted"] = 9.0
        squad[0]["now_cost"] = squad[0]["selling_price"] = 60
        # A far better keeper, affordable only by selling him.
        extra = [_player(90, "G", "Y0", 6.0, 14.0)]
        plan = _plan(squad, extra + _filler(200), bank_m=0.0, ft=1)
        assert not plan.get("hold")
        assert squad[0]["Player_ID"] in {leg["out_id"] for leg in plan["legs"]}


class TestHits:
    def _tempting(self, gain):
        squad = _base_squad()
        extra = []
        for i, pos in enumerate(("D", "M", "F")):
            extra.append(_player(90 + i, pos, "Y%d" % i, 5.0, 4.0 + gain))
        return squad, extra

    def test_no_hit_when_the_extra_transfer_does_not_pay(self):
        squad, extra = self._tempting(gain=0.5)
        plan = _plan(squad, extra + _filler(200), ft=1, hits=2)
        assert plan.get("hold") or plan["hits"] == 0

    def test_a_hit_is_taken_when_it_clears_four_over_the_horizon(self):
        squad, extra = self._tempting(gain=8.0)
        plan = _plan(squad, extra + _filler(200), ft=1, hits=2)
        assert plan["hits"] >= 1
        assert plan["gain_net"] > 0

    def test_the_horizon_decides_whether_a_hit_pays(self):
        """A per-gameweek gain has more gameweeks to repay a one-off -4 at a
        3-week horizon than at a 1-week one."""
        squad, extra = self._tempting(gain=2.0)
        pool = extra + _filler(200)
        short = _plan(squad, pool, ft=1, hits=1, w_now=1.0, w_next3=0.0)
        long_ = _plan(squad, pool, ft=1, hits=1, w_now=0.0, w_next3=1.0)
        short_hits = 0 if short.get("hold") else short["hits"]
        long_hits = 0 if long_.get("hold") else long_["hits"]
        assert long_hits >= short_hits

    def test_zero_extra_hits_never_spends_points(self):
        squad, extra = self._tempting(gain=20.0)
        plan = _plan(squad, extra + _filler(200), ft=1, hits=0)
        assert plan.get("hold") or plan["hits"] == 0


class TestBudget:
    def test_the_plan_is_affordable_as_a_set(self):
        """Two incoming players come out of one pot -- each affordable alone
        and the pair impossible is the bug the brute-force version shipped."""
        squad = _base_squad(price=5.0)
        extra = [_player(90, "M", "Y0", 8.0, 30.0), _player(91, "M", "Y1", 7.6, 30.0)]
        plan = _plan(squad, extra + _filler(200), bank_m=0.7, ft=2)
        if not plan.get("hold"):
            assert plan["bank_after"] >= -1e-6

    def test_the_bank_is_reported_consistently_with_the_legs(self):
        squad = _base_squad()
        extra = [_player(90, "M", "Y0", 6.0, 9.0)]
        plan = _plan(squad, extra + _filler(200), bank_m=2.0, ft=1)
        if not plan.get("hold"):
            implied = plan["bank_before"] + plan["released"] - plan["spent"]
            assert plan["bank_after"] == pytest.approx(implied, abs=0.01)


class TestDegradesQuietly:
    def test_an_empty_squad_returns_nothing(self):
        assert build_transfer_plan(pd.DataFrame(), pd.DataFrame(), 0, 1) is None

    def test_an_empty_pool_returns_nothing(self):
        squad = pd.DataFrame(_base_squad())
        assert build_transfer_plan(pd.DataFrame(), squad, 0, 1) is None

    def test_an_illegal_squad_is_reported_rather_than_planned_around(self):
        """Four players from one club: K=0 is infeasible, which is the canary."""
        squad = _base_squad()
        for r in squad[:4]:
            r["Team"] = "SAME"
        plan = _plan(squad, _filler(200), ft=1)
        assert plan.get("error") == "baseline"

    def test_missing_projection_columns_do_not_crash(self):
        squad = [{"Player_ID": i, "Player": "P%d" % i, "Position": p, "Team": "T%d" % i,
                  "now_cost": 50, "selling_price": 50, "status": "a", "minutes": 900,
                  "chance_of_playing_next_round": None}
                 for i, p in enumerate(["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3)]
        plan = _plan(squad, _filler(200), ft=1)
        assert plan is None or plan.get("hold") or "legs" in plan


class TestLegRationale:
    def test_a_money_freeing_leg_says_so(self):
        leg = {"net_cost": -7.0, "delta": -1.0}
        assert "Frees £7.0m" in _plan_leg_rationale(leg, released=21.0)

    def test_an_upgrade_names_the_pot_it_spends_from(self):
        leg = {"net_cost": 7.0, "delta": 3.0}
        text = _plan_leg_rationale(leg, released=21.0)
        assert "Upgrade" in text and "£21.0m released" in text

    def test_a_like_for_like_swap_is_neither(self):
        leg = {"net_cost": 0.0, "delta": 1.0}
        assert "Straight swap" in _plan_leg_rationale(leg, released=0.0)


class TestPlanAnnotation:
    """The cards and the planner optimise different things. A silent
    contradiction between them is worse than either answer."""

    def test_an_agreeing_card_is_marked_in_plan(self):
        suggestions = [{"drop_id": 1, "add_id": 2}]
        plan = {"legs": [{"out_id": 1, "in_id": 2, "in_player": "B"}]}
        _annotate_suggestions_against_plan(suggestions, plan)
        assert suggestions[0]["plan_status"][0] == "IN PLAN"

    def test_a_disagreeing_card_names_the_planners_pick(self):
        suggestions = [{"drop_id": 1, "add_id": 9}]
        plan = {"legs": [{"out_id": 1, "in_id": 2, "in_player": "Salah"}]}
        _annotate_suggestions_against_plan(suggestions, plan)
        assert suggestions[0]["plan_status"] == ("PLAN DIFFERS", "Salah")

    def test_a_card_the_plan_does_not_touch_is_unmarked(self):
        suggestions = [{"drop_id": 5, "add_id": 6}]
        plan = {"legs": [{"out_id": 1, "in_id": 2, "in_player": "B"}]}
        _annotate_suggestions_against_plan(suggestions, plan)
        assert suggestions[0]["plan_status"] is None

    def test_no_plan_leaves_every_card_unmarked(self):
        suggestions = [{"drop_id": 1, "add_id": 2}]
        _annotate_suggestions_against_plan(suggestions, None)
        assert suggestions[0]["plan_status"] is None


class TestWhatTheOldPlannerCouldNotDo:
    """The contrast, stated as a test rather than only in prose.

    `_build_multi_transfer_plan` is kept as the fallback when the solver is
    unavailable, so it stays exercised -- but it must not be mistaken for an
    equivalent answer.
    """

    @staticmethod
    def _frames():
        def row(pid, name, pos, team, price, keep, transfer, proj):
            return {"Player_ID": pid, "Player": name, "Full Name": name,
                    "Position": pos, "Team": team,
                    "now_cost": int(price * 10), "selling_price": int(price * 10),
                    "Keep Score": keep, "Transfer Score": transfer,
                    "Projected_Points": proj, "_effective_proj": proj,
                    "total_points": 50, "MultiGW_Proj": proj * 3,
                    "ep_next": proj, "form": 5.0}

        squad = [row(1, "Premium FWD", "F", "A", 14.0, 0.97, 0.97, 8.0),
                 row(2, "Mid MID", "M", "B", 6.0, 0.55, 0.55, 4.0)]
        squad += [row(10 + i, "Filler%d" % i, "D", "C%d" % i, 4.5,
                      0.30 + i * 0.01, 0.30, 2.0) for i in range(13)]
        avail = [row(90, "Premium MID", "M", "Y", 14.0, 0.99, 0.985, 12.0),
                 row(91, "Cheap FWD", "F", "Z", 6.0, 0.80, 0.80, 6.5)]
        return pd.DataFrame(squad), pd.DataFrame(avail)

    def test_the_premium_is_not_even_in_its_drop_pool(self):
        """`nsmallest(6, "Keep Score")` is six fillers. The two players the
        winning reallocation needs to sell are both outside it."""
        squad, _ = self._frames()
        pool = set(squad.nsmallest(6, "Keep Score")["Player"])
        assert "Premium FWD" not in pool
        assert "Mid MID" not in pool

    def test_it_cannot_propose_the_reallocation(self):
        squad, avail = self._frames()
        plan = _build_multi_transfer_plan(squad, avail, bank=0)
        sold = {leg.get("drop_player") for leg in (plan or [])}
        assert "Premium FWD" not in sold

    def test_the_ilp_proposes_it_from_the_same_position(self):
        """Same shape, expressed the planner's way: it sells the premium."""
        squad_rows = _base_squad()
        fwd = next(r for r in squad_rows if r["Position"] == "F")
        fwd.update(_player(fwd["Player_ID"], "F", fwd["Team"], 14.0, 8.0))
        mid = next(r for r in squad_rows if r["Position"] == "M")
        mid.update(_player(mid["Player_ID"], "M", mid["Team"], 6.0, 4.0))
        extra = [_player(90, "M", "Y0", 14.0, 12.0), _player(91, "F", "Y1", 6.0, 6.5)]
        plan = _plan(squad_rows, extra + _filler(200), bank_m=0.0, ft=2)
        assert fwd["Player_ID"] in {leg["out_id"] for leg in plan["legs"]}
