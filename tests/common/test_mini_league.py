"""Mini-league ownership arithmetic, against hand-counted squads.

Every expected number here is computed by hand in the test name or a comment,
because the whole hazard this module guards against is a plausible percentage
over the wrong denominator.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common.mini_league import (
    DEFAULT_DIFFERENTIAL_MAX_OWN,
    DEFAULT_TEMPLATE_MIN_EO,
    OWNERSHIP_COLUMNS,
    TOTALS_COLUMNS,
    compute_league_ownership,
    differentials,
    projected_team_totals,
    template_gaps,
    template_score,
)


def _pick(element, position, multiplier=None, is_captain=False):
    pick = {"element": element, "position": position, "is_captain": is_captain}
    if multiplier is not None:
        pick["multiplier"] = multiplier
    return pick


def _squad(elements, captain_index=0, bench_from=12, triple=False):
    """A 15-pick squad with FPL's own multiplier semantics."""
    picks = []
    for slot, element in enumerate(elements, start=1):
        if slot >= bench_from:
            mult = 0
        elif slot == captain_index + 1:
            mult = 3 if triple else 2
        else:
            mult = 1
        picks.append(_pick(element, slot, mult, is_captain=(slot == captain_index + 1)))
    return picks


def _pool(rows):
    return pd.DataFrame([
        {"Player_ID": pid, "Display_Name": name, "Position": pos,
         "Team": team, "Proj": proj, "Proj_Next3": proj * 3}
        for pid, name, pos, team, proj in rows
    ])


FIFTEEN_A = list(range(1, 16))          # players 1..15
FIFTEEN_B = [1] + list(range(16, 30))   # shares only player 1


class TestEffectiveOwnership:
    def test_a_player_everyone_starts_is_100_percent(self):
        # Player 2 starts for both squads, captained by neither: 2/2 = 100%.
        squads = {10: _squad(FIFTEEN_A, captain_index=0),
                  11: _squad(FIFTEEN_A, captain_index=0)}
        own = compute_league_ownership(squads)
        row = own[own["Player_ID"] == 2].iloc[0]
        assert row["Own_Pct"] == 100.0
        assert row["EO_Pct"] == 100.0
        assert row["Captains"] == 0

    def test_captaincy_can_push_effective_ownership_past_100(self):
        # Player 1 is captained by both: (2 + 2) / 2 = 200%. The ceiling is 300,
        # not 100, and a check that bounded it at 100 would fire on this.
        squads = {10: _squad(FIFTEEN_A, captain_index=0),
                  11: _squad(FIFTEEN_A, captain_index=0)}
        own = compute_league_ownership(squads)
        row = own[own["Player_ID"] == 1].iloc[0]
        assert row["Own_Pct"] == 100.0
        assert row["EO_Pct"] == 200.0
        assert row["Captains"] == 2

    def test_triple_captain_counts_three(self):
        squads = {10: _squad(FIFTEEN_A, captain_index=0, triple=True)}
        own = compute_league_ownership(squads)
        assert own[own["Player_ID"] == 1].iloc[0]["EO_Pct"] == 300.0

    def test_a_benched_player_is_owned_but_carries_no_effective_ownership(self):
        # Slot 13 is on the bench: Own 100%, EO 0%. Conflating the two would
        # price a benched player as a starter.
        squads = {10: _squad(FIFTEEN_A, captain_index=0)}
        own = compute_league_ownership(squads)
        row = own[own["Player_ID"] == 13].iloc[0]
        assert row["Own_Pct"] == 100.0
        assert row["EO_Pct"] == 0.0

    def test_bench_boost_pays_the_bench(self):
        picks = _squad(FIFTEEN_A, captain_index=0, bench_from=99)  # nobody benched
        own = compute_league_ownership({10: picks})
        assert own[own["Player_ID"] == 13].iloc[0]["EO_Pct"] == 100.0

    def test_partial_ownership_uses_the_number_of_squads_as_denominator(self):
        # Player 1 in both squads, player 2 in one of two: 100% and 50%.
        squads = {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_B)}
        own = compute_league_ownership(squads)
        assert own[own["Player_ID"] == 1].iloc[0]["Own_Pct"] == 100.0
        assert own[own["Player_ID"] == 2].iloc[0]["Own_Pct"] == 50.0

    def test_owner_counts_sum_to_fifteen_per_squad(self):
        # The exact invariant check_mini_league_ownership() relies on.
        squads = {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_B), 12: _squad(FIFTEEN_A)}
        own = compute_league_ownership(squads)
        assert int(own["Owners"].sum()) == 15 * 3

    def test_an_absent_multiplier_is_derived_from_the_slot(self):
        # The un-normalised my-team payload publishes no multiplier at all.
        # Deriving it from the slot is the safe reading; a *flat* one is left
        # for the validator to catch.
        picks = [_pick(e, slot) for slot, e in enumerate(FIFTEEN_A, start=1)]
        own = compute_league_ownership({10: picks})
        assert own[own["Player_ID"] == 1].iloc[0]["EO_Pct"] == 100.0   # slot 1
        assert own[own["Player_ID"] == 13].iloc[0]["EO_Pct"] == 0.0    # slot 13

    def test_no_squads_returns_an_empty_frame_with_the_contract_columns(self):
        own = compute_league_ownership({})
        assert own.empty
        assert list(own.columns) == OWNERSHIP_COLUMNS

    def test_columns_are_the_declared_contract(self):
        own = compute_league_ownership({10: _squad(FIFTEEN_A)})
        assert list(own.columns) == OWNERSHIP_COLUMNS


class TestPoolJoin:
    def test_projection_and_names_join_on_the_element_id(self):
        pool = _pool([(1, "Erling Haaland", "F", "MCI", 7.1),
                      (2, "Cole Palmer", "M", "CHE", 4.6)])
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, pool)
        row = own[own["Player_ID"] == 1].iloc[0]
        assert row["Display_Name"] == "Erling Haaland"
        assert row["Position"] == "F"
        assert row["Team"] == "MCI"
        assert row["Proj"] == pytest.approx(7.1)
        assert row["Proj_Next3"] == pytest.approx(21.3)

    def test_a_player_the_pool_does_not_carry_keeps_his_row(self):
        # Dropping him would break sum(Owners) == 15 x n_squads, which is the
        # only exact invariant available -- so the bootstrap lagging a new
        # signing must not silently shrink the table.
        pool = _pool([(1, "Erling Haaland", "F", "MCI", 7.1)])
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, pool)
        assert int(own["Owners"].sum()) == 15
        orphan = own[own["Player_ID"] == 9].iloc[0]
        assert orphan["Display_Name"] == "Player 9"
        assert pd.isna(orphan["Proj"])

    def test_no_pool_still_produces_ownership(self):
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, None)
        assert len(own) == 15
        assert own["Proj"].isna().all()

    def test_an_empty_pool_does_not_raise(self):
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, pd.DataFrame())
        assert len(own) == 15


class TestMine:
    def test_mine_marks_only_the_configured_team(self):
        squads = {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_B)}
        own = compute_league_ownership(squads, None, my_team_id=11)
        assert own[own["Player_ID"] == 16].iloc[0]["Mine"]       # only in squad 11
        assert not own[own["Player_ID"] == 2].iloc[0]["Mine"]    # only in squad 10
        assert own[own["Player_ID"] == 1].iloc[0]["Mine"]        # in both

    def test_no_configured_team_leaves_mine_false(self):
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, None, my_team_id=None)
        assert not own["Mine"].any()

    def test_a_team_id_outside_the_league_marks_nothing(self):
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, None, my_team_id=999)
        assert not own["Mine"].any()


class TestDifferentialsAndGaps:
    def _three_team_league(self):
        # Player 1 in all three, player 2 in two, players 16+ in team 12 only.
        return {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_A), 12: _squad(FIFTEEN_B)}

    def test_differentials_are_mine_and_thinly_owned(self):
        pool = _pool([(16, "Mine Alone", "M", "BRE", 5.0),
                      (1, "Template", "F", "MCI", 7.0)])
        own = compute_league_ownership(self._three_team_league(), pool, my_team_id=12)
        diffs = differentials(own)
        assert 16 in set(diffs["Player_ID"])
        # Player 1 is mine but owned by all three -- not a differential.
        assert 1 not in set(diffs["Player_ID"])

    def test_differentials_are_ordered_by_projection(self):
        pool = _pool([(16, "Low", "M", "BRE", 1.0), (17, "High", "M", "BRE", 9.0)])
        own = compute_league_ownership(self._three_team_league(), pool, my_team_id=12)
        diffs = differentials(own)
        assert diffs.iloc[0]["Display_Name"] == "High"

    def test_a_player_i_do_not_own_is_never_a_differential(self):
        own = compute_league_ownership(self._three_team_league(), None, my_team_id=12)
        assert differentials(own)["Mine"].all()

    def test_template_gaps_exclude_what_i_own(self):
        own = compute_league_ownership(self._three_team_league(), None, my_team_id=12)
        gaps = template_gaps(own)
        assert not gaps["Mine"].any()
        # Player 2 is in two of three squads as a starter: EO 66.7% >= 50.
        assert 2 in set(gaps["Player_ID"])

    def test_template_gaps_are_ordered_by_effective_ownership(self):
        own = compute_league_ownership(self._three_team_league(), None, my_team_id=12)
        gaps = template_gaps(own)
        assert gaps["EO_Pct"].is_monotonic_decreasing

    def test_thresholds_are_honoured(self):
        own = compute_league_ownership(self._three_team_league(), None, my_team_id=12)
        assert (differentials(own, max_own_pct=0.0)).empty
        assert (template_gaps(own, min_eo_pct=1000.0)).empty

    def test_empty_ownership_is_handled(self):
        empty = pd.DataFrame(columns=OWNERSHIP_COLUMNS)
        assert differentials(empty).empty
        assert template_gaps(empty).empty
        assert template_score(empty) is None

    def test_defaults_are_the_documented_halves(self):
        assert DEFAULT_DIFFERENTIAL_MAX_OWN == 50.0
        assert DEFAULT_TEMPLATE_MIN_EO == 50.0


class TestTemplateScore:
    def test_it_is_the_mean_effective_ownership_of_my_squad(self):
        # My squad: player 1 at 100% EO (both squads start him), the other 14
        # at 50% or 0% -- so just assert it lands between them rather than
        # restating the arithmetic.
        squads = {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_A)}
        own = compute_league_ownership(squads, None, my_team_id=10)
        score = template_score(own)
        assert 0.0 < score <= 200.0
        assert score == pytest.approx(own[own["Mine"]]["EO_Pct"].mean())

    def test_an_unknown_team_scores_none_not_zero(self):
        # "We do not know who you are" and "you own nothing the league owns"
        # are different statements.
        own = compute_league_ownership({10: _squad(FIFTEEN_A)}, None, my_team_id=None)
        assert template_score(own) is None


class TestProjectedTeamTotals:
    def test_the_multiplier_applies_the_lineup(self):
        # 11 starters at 1.0 each, captain doubled, 4 bench at zero:
        # 10 x 1.0 + 1 x 2.0 = 12.0.
        pool = _pool([(e, f"P{e}", "M", "BRE", 1.0) for e in FIFTEEN_A])
        totals = projected_team_totals({10: _squad(FIFTEEN_A)}, pool)
        assert totals.iloc[0]["Proj_Total"] == pytest.approx(12.0)

    def test_bench_boost_pays_all_fifteen(self):
        pool = _pool([(e, f"P{e}", "M", "BRE", 1.0) for e in FIFTEEN_A])
        picks = _squad(FIFTEEN_A, bench_from=99)
        totals = projected_team_totals({10: picks}, pool)
        assert totals.iloc[0]["Proj_Total"] == pytest.approx(16.0)

    def test_gap_is_measured_against_my_own_total(self):
        pool = _pool([(e, f"P{e}", "M", "BRE", 1.0) for e in range(1, 30)])
        squads = {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_B, triple=True)}
        totals = projected_team_totals(squads, pool, my_team_id=10)
        mine = totals[totals["Mine"]].iloc[0]
        assert mine["Gap_To_You"] == pytest.approx(0.0)
        other = totals[~totals["Mine"]].iloc[0]
        assert other["Gap_To_You"] == pytest.approx(other["Proj_Total"] - mine["Proj_Total"])

    def test_no_own_team_leaves_the_gap_unknown(self):
        pool = _pool([(e, f"P{e}", "M", "BRE", 1.0) for e in FIFTEEN_A])
        totals = projected_team_totals({10: _squad(FIFTEEN_A)}, pool, my_team_id=None)
        assert totals["Gap_To_You"].isna().all()

    def test_totals_are_ordered_best_first(self):
        pool = _pool([(e, f"P{e}", "M", "BRE", 1.0) for e in range(1, 30)])
        squads = {10: _squad(FIFTEEN_A), 11: _squad(FIFTEEN_B, triple=True)}
        totals = projected_team_totals(squads, pool)
        assert totals["Proj_Total"].is_monotonic_decreasing

    def test_manager_names_are_used_when_supplied(self):
        pool = _pool([(e, f"P{e}", "M", "BRE", 1.0) for e in FIFTEEN_A])
        totals = projected_team_totals({10: _squad(FIFTEEN_A)}, pool,
                                       team_names={10: "Stoned Squirrels"})
        assert totals.iloc[0]["Manager"] == "Stoned Squirrels"

    def test_an_unnamed_team_falls_back_to_its_id(self):
        totals = projected_team_totals({10: _squad(FIFTEEN_A)}, None, team_names={})
        assert totals.iloc[0]["Manager"] == "Team 10"

    def test_an_unpriced_player_contributes_nothing_rather_than_nan(self):
        # One NaN must not wipe out a whole squad's total.
        pool = _pool([(1, "Priced", "F", "MCI", 7.0)])
        totals = projected_team_totals({10: _squad(FIFTEEN_A)}, pool)
        assert totals.iloc[0]["Proj_Total"] == pytest.approx(14.0)  # captain

    def test_no_squads_returns_the_contract_columns(self):
        totals = projected_team_totals({}, None)
        assert totals.empty
        assert list(totals.columns) == TOTALS_COLUMNS


class TestPurity:
    def test_the_module_does_not_import_streamlit(self):
        # Keeps the arithmetic testable offline, the way ffp_feed and
        # transfer_risk are.
        import ast
        from pathlib import Path
        src = Path("scripts/common/mini_league.py").read_text()
        names = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert "streamlit" not in names
        assert "requests" not in names
