"""Unit tests for simulate_auto_subs() in analytics.py."""

import pandas as pd
import pytest

from scripts.common.analytics import simulate_auto_subs


def _make_squad():
    """Build a standard 15-player squad DataFrame for testing.

    Returns a DataFrame with:
    - 1 GK starter (squad_position 1)
    - 4 DEF starters (squad_position 2-5)
    - 4 MID starters (squad_position 6-9)
    - 2 FWD starters (squad_position 10-11)
    - 1 GK bench (squad_position 12)
    - 3 outfield bench (squad_position 13-15)
    """
    rows = [
        # Starters
        {"element_id": 101, "Player": "GK Starter", "Position": "G", "squad_position": 1},
        {"element_id": 201, "Player": "DEF 1", "Position": "D", "squad_position": 2},
        {"element_id": 202, "Player": "DEF 2", "Position": "D", "squad_position": 3},
        {"element_id": 203, "Player": "DEF 3", "Position": "D", "squad_position": 4},
        {"element_id": 204, "Player": "DEF 4", "Position": "D", "squad_position": 5},
        {"element_id": 301, "Player": "MID 1", "Position": "M", "squad_position": 6},
        {"element_id": 302, "Player": "MID 2", "Position": "M", "squad_position": 7},
        {"element_id": 303, "Player": "MID 3", "Position": "M", "squad_position": 8},
        {"element_id": 304, "Player": "MID 4", "Position": "M", "squad_position": 9},
        {"element_id": 401, "Player": "FWD 1", "Position": "F", "squad_position": 10},
        {"element_id": 402, "Player": "FWD 2", "Position": "F", "squad_position": 11},
        # Bench
        {"element_id": 102, "Player": "GK Bench", "Position": "G", "squad_position": 12},
        {"element_id": 205, "Player": "DEF Bench", "Position": "D", "squad_position": 13},
        {"element_id": 305, "Player": "MID Bench", "Position": "M", "squad_position": 14},
        {"element_id": 403, "Player": "FWD Bench", "Position": "F", "squad_position": 15},
    ]
    return pd.DataFrame(rows)


def _element_to_team():
    """All players on team 1 by default."""
    return {eid: 1 for eid in range(100, 500)}


class TestSimulateAutoSubsBasic:
    def test_no_subs_when_all_played(self):
        """No subs should happen when all starters have minutes."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert subs == []
        # Squad positions unchanged
        assert sorted(result[result["squad_position"] <= 11]["element_id"].tolist()) == sorted(
            squad[squad["squad_position"] <= 11]["element_id"].tolist()
        )

    def test_no_subs_when_match_not_finished(self):
        """No subs when the player's team match is not finished yet."""
        squad = _make_squad()
        # MID 1 has 0 minutes but team match not finished
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[301] = {"minutes": 0, "has_played": False, "points": 0}
        finished = set()  # No matches finished

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert subs == []

    def test_no_subs_when_player_has_minutes(self):
        """No sub for a player who played even 1 minute (even if 0 points)."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[301] = {"minutes": 1, "has_played": True, "points": 0}
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert subs == []

    def test_empty_squad(self):
        """Empty squad returns empty result."""
        squad = pd.DataFrame(columns=["element_id", "Player", "Position", "squad_position"])
        result, subs = simulate_auto_subs(squad, {}, {}, set())
        assert subs == []
        assert result.empty

    def test_no_finished_teams(self):
        """No subs when finished_team_ids is empty."""
        squad = _make_squad()
        live_stats = {301: {"minutes": 0, "has_played": False, "points": 0}}
        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), set())
        assert subs == []


class TestSimulateAutoSubsOutfield:
    def test_basic_outfield_sub(self):
        """A MID starter with 0 minutes and finished match gets subbed by first bench outfield."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[301] = {"minutes": 0, "has_played": False, "points": 0}  # MID 1 didn't play
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 1
        assert subs[0] == ("MID 1", "DEF Bench")

        # DEF Bench should now be in starting XI
        new_starters = result[result["squad_position"] <= 11]
        assert 205 in new_starters["element_id"].values
        assert 301 not in new_starters["element_id"].values

    def test_bench_order_respected(self):
        """Bench positions 13→14→15 should be tried in order."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        # Two outfield starters didn't play
        live_stats[301] = {"minutes": 0, "has_played": False, "points": 0}  # MID 1
        live_stats[302] = {"minutes": 0, "has_played": False, "points": 0}  # MID 2
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 2
        # First sub uses bench pos 13 (DEF Bench), second uses bench pos 14 (MID Bench)
        assert subs[0] == ("MID 1", "DEF Bench")
        assert subs[1] == ("MID 2", "MID Bench")

    def test_multiple_outfield_subs(self):
        """Three starters out: all three bench outfield players come in."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[301] = {"minutes": 0, "has_played": False, "points": 0}
        live_stats[302] = {"minutes": 0, "has_played": False, "points": 0}
        live_stats[303] = {"minutes": 0, "has_played": False, "points": 0}
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 3


class TestSimulateAutoSubsGK:
    def test_gk_sub(self):
        """GK starter with 0 min gets replaced by bench GK (pos 12)."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[101] = {"minutes": 0, "has_played": False, "points": 0}  # GK didn't play
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 1
        assert subs[0] == ("GK Starter", "GK Bench")

        new_starters = result[result["squad_position"] <= 11]
        assert 102 in new_starters["element_id"].values
        assert 101 not in new_starters["element_id"].values

    def test_gk_not_replaced_by_outfield(self):
        """GK can only be replaced by GK bench, not outfield bench.

        If bench GK also didn't play, no sub happens for GK.
        """
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[101] = {"minutes": 0, "has_played": False, "points": 0}

        # Remove the bench GK entirely by setting their position to outfield
        squad.loc[squad["element_id"] == 102, "Position"] = "D"
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        # No sub should happen because there's no bench GK
        assert len(subs) == 0


class TestSimulateAutoSubsDEFConstraint:
    def test_def_minimum_respected(self):
        """If removing a DEF drops below 3 DEF, only a DEF bench player can sub in."""
        squad = _make_squad()
        # Only have 3 DEF starting (remove DEF 4 from starters, replace with FWD)
        squad.loc[squad["element_id"] == 204, "Position"] = "F"

        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        # DEF 1 didn't play — this would drop DEF to 2 if replaced by non-DEF
        live_stats[201] = {"minutes": 0, "has_played": False, "points": 0}
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 1
        # Should sub in DEF Bench (pos 13), not MID Bench (pos 14) even though MID is next
        assert subs[0] == ("DEF 1", "DEF Bench")

    def test_def_sub_when_enough_def(self):
        """With 4 DEF, subbing one out still leaves 3, so any outfield can replace."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[201] = {"minutes": 0, "has_played": False, "points": 0}  # DEF 1 didn't play
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 1
        # With 4 DEF, removing 1 leaves 3 — any bench player can sub in
        # First outfield bench is DEF Bench (pos 13)
        assert subs[0] == ("DEF 1", "DEF Bench")


class TestSimulateAutoSubsPartialFinished:
    def test_only_finished_match_players_subbed(self):
        """Only players whose team match is finished get auto-subbed."""
        squad = _make_squad()
        # Put MID 1 on team 1 (finished), MID 2 on team 2 (not finished)
        e2t = _element_to_team()
        e2t[302] = 2  # MID 2 on team 2

        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[301] = {"minutes": 0, "has_played": False, "points": 0}  # MID 1, team 1
        live_stats[302] = {"minutes": 0, "has_played": False, "points": 0}  # MID 2, team 2
        finished = {1}  # Only team 1 finished

        result, subs = simulate_auto_subs(squad, live_stats, e2t, finished)
        assert len(subs) == 1
        assert subs[0][0] == "MID 1"  # Only MID 1 subbed (team 1 finished)

    def test_bench_player_subbed_in_even_with_0_min(self):
        """A bench player can be subbed in even if they also got 0 minutes."""
        squad = _make_squad()
        live_stats = {eid: {"minutes": 90, "has_played": True, "points": 5} for eid in range(100, 500)}
        live_stats[301] = {"minutes": 0, "has_played": False, "points": 0}  # MID 1 didn't play
        live_stats[205] = {"minutes": 0, "has_played": False, "points": 0}  # DEF Bench also didn't play
        finished = {1}

        result, subs = simulate_auto_subs(squad, live_stats, _element_to_team(), finished)
        assert len(subs) == 1
        # DEF Bench comes in even though they have 0 minutes (they just score 0)
        assert subs[0] == ("MID 1", "DEF Bench")
