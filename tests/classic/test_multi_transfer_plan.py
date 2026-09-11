"""Tests for the Classic 2-Transfer Plan.

The plan proposes two transfers made together, so it has to be legal and
buyable *as a pair*. Each check here is a way it was not: two adds priced
against the same pot independently, the same player recommended twice, and a
club taken to four players.
"""

import pandas as pd
import pytest

from scripts.classic.transfers import _build_multi_transfer_plan


def _outlay(plan):
    """What the plan actually spends, read back off the rendered prices."""
    return sum(round(float(leg["add_price"].strip("£m")) * 10) for leg in plan)


def _squad_row(pid, pos, team, now_cost, keep_score, selling_price=None):
    return {
        "Player_ID": pid,
        "Player": f"Squad{pid}",
        "Full Name": f"Squad Player {pid}",
        "Team": team,
        "Team_ID": pid,
        "Position": pos,
        "squad_position": pid,
        "now_cost": now_cost,
        "selling_price": now_cost if selling_price is None else selling_price,
        "form": 5.0,
        "total_points": 50,
        "news": "",
        "chance_of_playing_next_round": None,
        "Keep Score": keep_score,
        "Projected_Points": 3.0,
    }


def _avail_row(pid, pos, team, now_cost, transfer_score, ownership=10.0):
    return {
        "Player_ID": pid,
        "Player": f"Target{pid}",
        "Full Name": f"Target Player {pid}",
        "Team": team,
        "Team_ID": pid,
        "Position": pos,
        "now_cost": now_cost,
        "form": 6.0,
        "total_points": 80,
        "news": "",
        "chance_of_playing_next_round": None,
        "Transfer Score": transfer_score,
        "Projected_Points": 6.0,
        "selected_by_percent": ownership,
    }


def _base_squad():
    """Fifteen players, one per club, so the club rule never bites by accident."""
    rows = []
    pid = 1
    for pos, count, cost in [("G", 2, 45), ("D", 5, 45), ("M", 5, 55), ("F", 3, 60)]:
        for _ in range(count):
            # Keep Score ascending with id, so the first players listed are the
            # ones nsmallest() picks as drop candidates.
            rows.append(_squad_row(pid, pos, f"CLB{pid}", cost, 0.10 + pid * 0.01))
            pid += 1
    return pd.DataFrame(rows)


class TestJointAffordability:
    def test_pair_must_fit_one_pot(self):
        """The reported bug: two premiums, each affordable alone, bought together.

        Bank £0.5m plus a £4.1m and a £6.1m sale is a £10.7m pot. An £8.0m
        defender and a £7.6m midfielder each clear that on their own; together
        they cost £15.6m and cannot be bought.
        """
        squad = _base_squad()
        squad.loc[squad["Player_ID"] == 3, ["now_cost", "selling_price", "Keep Score"]] = [41, 41, 0.10]
        squad.loc[squad["Player_ID"] == 8, ["now_cost", "selling_price", "Keep Score"]] = [61, 61, 0.11]

        available = pd.DataFrame([
            _avail_row(101, "D", "ARS", 80, 0.95),   # premium DEF
            _avail_row(102, "M", "CHE", 76, 0.94),   # premium MID
            _avail_row(103, "D", "BOU", 45, 0.70),   # affordable DEF
            _avail_row(104, "M", "EVE", 55, 0.68),   # affordable MID
        ])

        plan = _build_multi_transfer_plan(squad, available, bank=5)

        assert len(plan) == 2
        funds = 5 + 41 + 61  # bank + both selling prices, in tenths of a million
        outlay = _outlay(plan)
        assert outlay <= funds, (
            f"plan spends £{outlay/10:.1f}m of £{funds/10:.1f}m — the two adds were "
            "each priced against the whole pot instead of against one another"
        )
        # The caption quotes these, so they must describe the same move.
        assert plan[0]["plan_funds"] == funds
        assert plan[0]["plan_outlay"] == outlay
        # Exactly one premium is affordable alongside a cheap partner.
        names = {p["add_player"] for p in plan}
        assert not {"Target101", "Target102"} <= names

    def test_returns_nothing_when_no_pair_is_affordable(self):
        squad = _base_squad()
        available = pd.DataFrame([
            _avail_row(101, "D", "ARS", 130, 0.95),
            _avail_row(102, "M", "CHE", 130, 0.94),
        ])
        assert _build_multi_transfer_plan(squad, available, bank=0) == []

    def test_selling_price_funds_the_move_not_current_price(self):
        """A player bought before a price rise sells for less than he now costs."""
        squad = _base_squad()
        squad.loc[squad["Player_ID"] == 3, ["now_cost", "selling_price", "Keep Score"]] = [60, 50, 0.10]
        squad.loc[squad["Player_ID"] == 8, ["now_cost", "selling_price", "Keep Score"]] = [60, 50, 0.11]

        available = pd.DataFrame([
            _avail_row(101, "D", "ARS", 55, 0.95),
            _avail_row(102, "M", "CHE", 55, 0.94),
            _avail_row(103, "D", "BOU", 40, 0.70),
            _avail_row(104, "M", "EVE", 40, 0.68),
        ])

        plan = _build_multi_transfer_plan(squad, available, bank=0)

        assert plan, "a £10.0m pot should still fund a pair"
        assert _outlay(plan) <= 100, (
            "the pot is two £5.0m selling prices, not two £6.0m current prices"
        )
        assert plan[0]["plan_funds"] == 100


class TestPairIsLegal:
    def test_two_drops_in_one_position_get_two_different_players(self):
        """Identical position, identical candidate list — and once, one player twice."""
        squad = _base_squad()
        # Make the two weakest players both defenders.
        squad.loc[squad["Player_ID"] == 3, "Keep Score"] = 0.01
        squad.loc[squad["Player_ID"] == 4, "Keep Score"] = 0.02

        available = pd.DataFrame([
            _avail_row(101, "D", "ARS", 45, 0.95),
            _avail_row(102, "D", "BOU", 45, 0.90),
            _avail_row(103, "M", "CHE", 45, 0.40),
        ])

        plan = _build_multi_transfer_plan(squad, available, bank=0)

        assert len(plan) == 2
        assert plan[0]["add_player"] != plan[1]["add_player"], "same player added twice"

    def test_two_adds_cannot_take_a_club_to_four(self):
        squad = _base_squad()
        # Two existing Arsenal players, plus two cheap drops to make room.
        squad.loc[squad["Player_ID"] == 6, "Team"] = "ARS"
        squad.loc[squad["Player_ID"] == 7, "Team"] = "ARS"
        squad.loc[squad["Player_ID"] == 3, "Keep Score"] = 0.01
        squad.loc[squad["Player_ID"] == 8, "Keep Score"] = 0.02

        available = pd.DataFrame([
            _avail_row(101, "D", "ARS", 45, 0.95),
            _avail_row(102, "M", "ARS", 55, 0.94),
            _avail_row(103, "M", "EVE", 55, 0.60),
        ])

        plan = _build_multi_transfer_plan(squad, available, bank=100)

        assert len(plan) == 2
        add_teams = [p["add_team"] for p in plan]
        assert add_teams.count("ARS") <= 1, "plan puts a fourth Arsenal player in the squad"

    def test_a_drop_is_never_paired_with_itself(self):
        squad = _base_squad()
        available = pd.DataFrame([
            _avail_row(101, "D", "ARS", 45, 0.95),
            _avail_row(102, "D", "CHE", 45, 0.94),
            _avail_row(103, "M", "EVE", 55, 0.90),
        ])
        plan = _build_multi_transfer_plan(squad, available, bank=100)
        assert len(plan) == 2
        assert plan[0]["drop_player"] != plan[1]["drop_player"]


class TestDegradesQuietly:
    @pytest.mark.parametrize("squad,available", [
        (pd.DataFrame(), pd.DataFrame([{"Player_ID": 1}])),
        (pd.DataFrame([{"Player_ID": 1}]), pd.DataFrame()),
    ])
    def test_empty_frames(self, squad, available):
        assert _build_multi_transfer_plan(squad, available, bank=10) == []

    def test_no_keep_score_column(self):
        squad = _base_squad().drop(columns=["Keep Score"])
        available = pd.DataFrame([_avail_row(101, "D", "ARS", 45, 0.95)])
        assert _build_multi_transfer_plan(squad, available, bank=10) == []
