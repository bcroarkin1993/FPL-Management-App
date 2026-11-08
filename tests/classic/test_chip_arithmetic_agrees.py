"""The chip arithmetic exists twice, so it is pinned to agree.

``fixture_projections._calculate_projected_score()`` has applied Bench Boost
and Triple Captain for a year -- all fifteen count under ``bboost``, and
``captain_mult = 3 if active_chip == "3xc" else 2``. It is gameweek-agnostic
(it reads whichever single-gameweek points column is on the frame) and it lives
in a Streamlit page module, so ``scripts/common/chips.py`` could not import it
and has its own implementation.

Two implementations of one piece of arithmetic is the twin-drift failure this
codebase has paid for three times -- the FFP double discount,
``compute_player_scores`` against ``blend_fixture_projections``, and the
``Proj_Next3`` assembly. In a *chip* valuation a drift would name the wrong
gameweek to spend an irreplaceable resource on, while every number on screen
stayed plausible. Hence this file.
"""

import pandas as pd
import pytest

from scripts.classic.fixture_projections import _calculate_projected_score
from scripts.common import chips
from scripts.common.optimization import best_xi_index

_POINTS = [6, 1, 7, 6, 5, 2, 1, 9, 8, 7, 3, 2, 10, 4, 1]
_POSITIONS = ["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3


def _aligned_squad():
    """A squad whose stored lineup *is* the optimal XI.

    The two implementations answer slightly different questions -- the page
    scores the lineup as picked, `chips` scores the best legal one -- so they
    can only be compared where those coincide. That is the right comparison:
    given one lineup, the chip arithmetic must be identical.
    """
    squad = pd.DataFrame({
        "element_id": range(1, 16),
        "Player_ID": range(1, 16),
        "Position": _POSITIONS,
        "Points": _POINTS,
    })
    chosen = best_xi_index(squad.assign(_p=squad["Points"]), "_p")
    squad["squad_position"] = [
        (chosen.index(i) + 1) if i in chosen else 12 + sorted(
            set(squad.index) - set(chosen)).index(i)
        for i in squad.index
    ]
    best = squad.loc[chosen, "Points"].idxmax()
    squad["is_captain"] = squad.index == best
    return squad


class TestBenchBoostAgrees:
    def test_the_two_implementations_match(self):
        squad = _aligned_squad()
        page = (_calculate_projected_score(squad, active_chip="bboost")
                - _calculate_projected_score(squad, active_chip=None))
        matrix = chips.squad_points_matrix(
            squad, pd.DataFrame({10: _POINTS}, index=range(1, 16)), [10])
        assert chips.bench_boost_value(squad, matrix)[10] == pytest.approx(page)


class TestTripleCaptainAgrees:
    def test_the_two_implementations_match(self):
        squad = _aligned_squad()
        page = (_calculate_projected_score(squad, active_chip="3xc")
                - _calculate_projected_score(squad, active_chip=None))
        matrix = chips.squad_points_matrix(
            squad, pd.DataFrame({10: _POINTS}, index=range(1, 16)), [10])
        assert chips.triple_captain_value(squad, matrix)[10] == pytest.approx(page)

    def test_it_really_is_one_extra_copy_and_not_two(self):
        """The armband already doubles. Charging 2x more is the double-count
        this comparison exists to catch."""
        squad = _aligned_squad()
        captain_points = float(squad.loc[squad["is_captain"], "Points"].iloc[0])
        page = (_calculate_projected_score(squad, active_chip="3xc")
                - _calculate_projected_score(squad, active_chip=None))
        assert page == pytest.approx(captain_points)
