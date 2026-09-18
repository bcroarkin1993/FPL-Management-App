"""The shared squad-building ILP, and transfer mode on top of it.

`solve_squad_ilp` had no test file at all. It now also answers "the best squad
reachable in at most K transfers", which is three new ways to be quietly wrong:
a change constraint that miscounts, a budget that prices a kept player at the
market rate, and a hit penalty that does not match FPL's arithmetic.

The three existing callers -- Free Hit, Wildcard, Initial Squad -- pass none of
the new arguments and must be unaffected, which `TestLegacyCallersUnaffected`
pins.
"""

import pandas as pd
import pytest

from scripts.common.optimization import (
    HIT_COST,
    SQUAD_SIZE,
    diff_squads,
    pair_transfer_legs,
    solve_squad_ilp,
)


def _pool(n_per_pos=(6, 12, 12, 8), price=5.0, score=4.0):
    """A pool wide enough to build several legal squads from."""
    rows = []
    pid = 0
    for pos, n in zip("GDMF", n_per_pos):
        for k in range(n):
            rows.append({
                "Player_ID": pid, "Player": "%s%d" % (pos, k), "Position": pos,
                # One club per player: the 3-per-club rule never bites by accident.
                "Team": "T%d" % pid,
                "Price": price, "Sell_Price": price, "Score": score,
            })
            pid += 1
    return pd.DataFrame(rows)


def _owned(pool):
    """A legal 2/5/5/3 taken off the front of each position."""
    ids = []
    for pos, n in (("G", 2), ("D", 5), ("M", 5), ("F", 3)):
        ids += pool[pool["Position"] == pos]["Player_ID"].head(n).tolist()
    return ids


def _solve(pool, owned, K, budget=200.0, free_transfers=None, **kw):
    return solve_squad_ilp(
        pool, budget, score_col="Score", owned_ids=owned, max_changes=K,
        sell_price_col="Sell_Price", free_transfers=free_transfers, **kw)


def _n_changes(pool, owned, squad):
    return SQUAD_SIZE - len(set(squad["Player_ID"]) & set(owned))


class TestChangeConstraint:
    @pytest.mark.parametrize("K", [0, 1, 2, 3, 5])
    def test_at_most_k_players_are_swapped(self, K):
        pool = _pool()
        owned = _owned(pool)
        # Make a handful of unowned players clearly better, so the solver wants
        # more changes than it is allowed.
        pool.loc[~pool["Player_ID"].isin(owned), "Score"] = 9.0
        squad, totals = _solve(pool, owned, K)
        assert squad is not None
        assert _n_changes(pool, owned, squad) <= K
        assert totals["n_changes"] <= K

    def test_k_zero_returns_the_squad_you_own(self):
        pool = _pool()
        owned = _owned(pool)
        pool.loc[~pool["Player_ID"].isin(owned), "Score"] = 9.0
        squad, totals = _solve(pool, owned, 0)
        assert set(squad["Player_ID"]) == set(owned)
        assert totals["n_changes"] == 0

    def test_an_owned_player_with_no_projection_stays_in_the_pool(self):
        """The pre-filter drops score <= 0. Dropping an owner makes his sale
        free and uncounted against K -- silently."""
        pool = _pool()
        owned = _owned(pool)
        pool.loc[pool["Player_ID"] == owned[7], "Score"] = 0.0  # an injured £5m MID
        pool.loc[~pool["Player_ID"].isin(owned), "Score"] = 9.0
        squad, totals = _solve(pool, owned, 1)
        assert _n_changes(pool, owned, squad) <= 1

    def test_several_zero_scored_owners_do_not_make_it_infeasible(self):
        """Two dropped owners turn K=1 into "keep 14 of 13" -- unsolvable."""
        pool = _pool()
        owned = _owned(pool)
        pool.loc[pool["Player_ID"].isin(owned[:4]), "Score"] = 0.0
        squad, _ = _solve(pool, owned, 1)
        assert squad is not None

    def test_an_owned_player_missing_from_the_pool_raises(self):
        """Loud, not silent: the constraint counts kept players out of 15, so a
        missing owner buys a free transfer."""
        pool = _pool()
        owned = _owned(pool)
        trimmed = pool[pool["Player_ID"] != owned[0]]
        with pytest.raises(ValueError, match="absent from the candidate pool"):
            _solve(trimmed, owned, 1)


class TestPricing:
    def test_keeping_a_risen_player_stays_affordable(self):
        """Priced at the market rate, keeping your own squad comes out
        infeasible the moment prices rise above what you paid."""
        pool = _pool(price=6.0)
        owned = _owned(pool)
        pool.loc[pool["Player_ID"].isin(owned), "Sell_Price"] = 6.0
        pool.loc[pool["Player_ID"].isin(owned), "Price"] = 7.0  # market has risen
        budget = 0.5 + 15 * 6.0  # bank + selling value
        squad, _ = _solve(pool, owned, 0, budget=budget)
        assert squad is not None

    def test_an_incoming_player_is_charged_the_market_price(self):
        pool = _pool(price=5.0)
        owned = _owned(pool)
        pool.loc[~pool["Player_ID"].isin(owned), ["Price", "Score"]] = [9.0, 9.0]
        budget = 15 * 5.0  # no bank: the upgrade is unaffordable
        squad, _ = _solve(pool, owned, 1, budget=budget)
        assert set(squad["Player_ID"]) == set(owned)

    def test_a_missing_selling_price_falls_back_to_the_market_one(self):
        pool = _pool()
        owned = _owned(pool)
        pool.loc[pool["Player_ID"] == owned[0], "Sell_Price"] = None
        squad, _ = _solve(pool, owned, 1)
        assert squad is not None

    def test_the_plan_is_affordable_as_a_set_not_one_leg_at_a_time(self):
        """Two incoming players come out of one pot. Each affordable alone and
        the pair impossible is the bug the brute-force version shipped."""
        pool = _pool(price=5.0)
        owned = _owned(pool)
        # Two tempting £8.0m midfielders against £5.4m of headroom.
        upgrades = pool[(pool["Position"] == "M") & (~pool["Player_ID"].isin(owned))]
        pool.loc[upgrades.index[:2], ["Price", "Score"]] = [8.0, 20.0]
        budget = 0.4 + 15 * 5.0
        squad, totals = _solve(pool, owned, 2, budget=budget)
        assert totals["squad_cost"] <= budget + 1e-6


class TestHitPenalty:
    def _upgrade_pool(self, gain):
        """Upgrades spread across outfield positions.

        Taking the first four unowned rows instead picks four goalkeepers, only
        one of whom can start -- so the second transfer gains nothing and no
        hit is ever worth taking. The fixture has to let the solver actually
        bank the gain.
        """
        pool = _pool()
        owned = _owned(pool)
        free = pool[~pool["Player_ID"].isin(owned)]
        for pos in ("D", "M", "F"):
            idx = free[free["Position"] == pos].index[:2]
            pool.loc[idx, "Score"] = 4.0 + gain
        return pool, owned

    def test_no_hit_when_the_gain_does_not_clear_the_cost(self):
        pool, owned = self._upgrade_pool(gain=1.0)  # +1.0 per extra transfer
        squad, totals = _solve(pool, owned, 3, free_transfers=1)
        assert totals["hits"] == 0
        assert totals["n_changes"] <= 1

    def test_a_hit_is_taken_when_the_gain_clears_the_cost(self):
        pool, owned = self._upgrade_pool(gain=HIT_COST + 3.0)
        squad, totals = _solve(pool, owned, 3, free_transfers=1)
        assert totals["hits"] >= 1
        assert totals["n_changes"] > 1

    def test_hits_are_changes_beyond_the_allowance(self):
        pool, owned = self._upgrade_pool(gain=20.0)
        squad, totals = _solve(pool, owned, 4, free_transfers=2)
        assert totals["hits"] == max(0, totals["n_changes"] - 2)

    def test_changes_within_the_allowance_cost_nothing(self):
        pool, owned = self._upgrade_pool(gain=20.0)
        squad, totals = _solve(pool, owned, 2, free_transfers=4)
        assert totals["hits"] == 0

    def test_free_transfers_price_but_max_changes_bounds(self):
        """Conflating the two is the bug waiting to happen: K caps the search,
        free_transfers only decides what it costs."""
        pool, owned = self._upgrade_pool(gain=20.0)
        _, totals = _solve(pool, owned, 1, free_transfers=5)
        assert totals["n_changes"] <= 1


class TestLegacyCallersUnaffected:
    """Free Hit, Wildcard and Initial Squad pass none of the new arguments."""

    def _legacy(self, **kw):
        pool = _pool()
        pool["Score"] = [4.0 + (i % 7) * 0.5 for i in range(len(pool))]
        return solve_squad_ilp(pool, 100.0, score_col="Score", **kw)

    def test_free_hit_argument_set(self):
        squad, totals = self._legacy(formation="auto", bench_weight=0.0,
                                     problem_name="FPL_Free_Hit_Optimizer")
        assert squad is not None and len(squad) == SQUAD_SIZE
        assert "n_changes" not in totals  # transfer mode stayed off

    def test_wildcard_and_initial_squad_argument_sets(self):
        squad, totals = self._legacy(formation="3-4-3", bench_weight=0.1,
                                     captain_score_col="Score",
                                     captain_bonus_weight=1.0)
        assert squad is not None and squad["Is_Starter"].sum() == 11
        assert squad["Is_Captain"].sum() == 1
        assert "hits" not in totals

    def test_a_zero_scored_player_is_still_filtered_out_without_owned_ids(self):
        """The exemption is scoped to transfer mode; loosening it for everyone
        would change what these three build."""
        pool = _pool()
        pool["Score"] = 4.0
        pool.loc[pool["Position"] == "F", "Score"] = 0.0
        pool["Price"] = 9.0  # above the cheap-bench escape hatch
        squad, _ = solve_squad_ilp(pool, 200.0, score_col="Score")
        assert squad is None  # no legal squad without forwards


class TestDiffAndPairing:
    def test_position_multisets_always_match(self):
        pool = _pool()
        owned = _owned(pool)
        pool.loc[~pool["Player_ID"].isin(owned), "Score"] = 9.0
        squad, _ = _solve(pool, owned, 3)
        before = pool[pool["Player_ID"].isin(owned)]
        outs, ins = diff_squads(before, squad)
        assert sorted(outs["Position"]) == sorted(ins["Position"])

    def test_every_leg_is_within_position(self):
        outs = pd.DataFrame([
            {"Player_ID": 1, "Player": "Out F", "Position": "F", "Sell_Price": 14.5, "Score": 7.0},
            {"Player_ID": 2, "Player": "Out M", "Position": "M", "Sell_Price": 7.0, "Score": 4.0},
        ])
        ins = pd.DataFrame([
            {"Player_ID": 3, "Player": "In M", "Position": "M", "Price": 14.0, "Score": 8.0},
            {"Player_ID": 4, "Player": "In F", "Position": "F", "Price": 7.5, "Score": 5.0},
        ])
        legs = pair_transfer_legs(outs, ins, score_col="Score")
        assert all(leg["position"] == leg["out_row"]["Position"] for leg in legs)
        by_out = {leg["out_player"]: leg["in_player"] for leg in legs}
        assert by_out == {"Out F": "In F", "Out M": "In M"}

    def test_pairing_is_by_price_within_position(self):
        """The dearest sale funds the dearest purchase, so the released money
        sits next to the money spent."""
        outs = pd.DataFrame([
            {"Player_ID": 1, "Player": "Dear", "Position": "M", "Sell_Price": 12.0, "Score": 6.0},
            {"Player_ID": 2, "Player": "Cheap", "Position": "M", "Sell_Price": 5.0, "Score": 4.0},
        ])
        ins = pd.DataFrame([
            {"Player_ID": 3, "Player": "Premium", "Position": "M", "Price": 13.0, "Score": 8.0},
            {"Player_ID": 4, "Player": "Budget", "Position": "M", "Price": 4.5, "Score": 5.0},
        ])
        legs = pair_transfer_legs(outs, ins, score_col="Score")
        pairs = {leg["out_player"]: leg["in_player"] for leg in legs}
        assert pairs == {"Dear": "Premium", "Cheap": "Budget"}

    def test_legs_are_ordered_so_the_bank_never_goes_negative(self):
        """Money-freeing legs first, so staging them in FPL's UI is feasible."""
        outs = pd.DataFrame([
            {"Player_ID": 1, "Player": "Haaland", "Position": "F", "Sell_Price": 14.5, "Score": 7.0},
            {"Player_ID": 2, "Player": "Mid", "Position": "M", "Sell_Price": 7.0, "Score": 4.0},
        ])
        ins = pd.DataFrame([
            {"Player_ID": 3, "Player": "Salah", "Position": "M", "Price": 14.0, "Score": 8.5},
            {"Player_ID": 4, "Player": "Wissa", "Position": "F", "Price": 7.5, "Score": 5.0},
        ])
        legs = pair_transfer_legs(outs, ins, score_col="Score")
        bank = 0.2
        for leg in legs:
            bank += leg["out_price"] - leg["in_price"]
            assert bank >= -1e-9, "bank went negative at %s" % leg["out_player"]

    def test_pairing_is_deterministic_under_equal_prices(self):
        outs = pd.DataFrame([
            {"Player_ID": 1, "Player": "Worse", "Position": "D", "Sell_Price": 5.0, "Score": 2.0},
            {"Player_ID": 2, "Player": "Better", "Position": "D", "Sell_Price": 5.0, "Score": 4.0},
        ])
        ins = pd.DataFrame([
            {"Player_ID": 3, "Player": "Best", "Position": "D", "Price": 5.0, "Score": 9.0},
            {"Player_ID": 4, "Player": "Good", "Position": "D", "Price": 5.0, "Score": 6.0},
        ])
        legs = pair_transfer_legs(outs, ins, score_col="Score")
        pairs = {leg["out_player"]: leg["in_player"] for leg in legs}
        assert pairs == {"Worse": "Best", "Better": "Good"}

    def test_empty_input_is_no_legs_rather_than_a_crash(self):
        assert pair_transfer_legs(pd.DataFrame(), pd.DataFrame()) == []
