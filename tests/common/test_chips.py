"""Chip valuations, and the rules they rest on.

Every number here is hand-computed rather than captured from the
implementation, because the failure mode this guards against is a valuation
that is plausible and wrong -- a chip grid cannot look suspicious, it can only
name the wrong gameweek to spend an irreplaceable resource on.
"""

import math

import pandas as pd
import pytest

from scripts.common import chips


def _squad(points, positions=None):
    """A legal 15: 2 GK, 5 DEF, 5 MID, 3 FWD, with the given projections."""
    positions = positions or (["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3)
    return pd.DataFrame({
        "element_id": range(1, 16),
        "Position": positions,
        "Proj": points,
    })


#: A squad whose best legal XI is hand-solvable. Minimums take GK 6, DEF
#: 7/6/5, MID 9/8, FWD 10 (= 51 over 7 players); the four free slots then go to
#: the best remaining that maxima admit -- MID 7, FWD 4, MID 3, DEF 2 (= 16).
#: So the XI is 67, the whole squad 72, and the bench 5.
_POINTS = [6, 1, 7, 6, 5, 2, 1, 9, 8, 7, 3, 2, 10, 4, 1]
_SQUAD_TOTAL = 72
_XI_TOTAL = 67


def _matrix(squad, points_by_gw, gameweeks):
    return chips.squad_points_matrix(squad, points_by_gw, gameweeks)


class TestBenchBoost:
    def test_it_is_the_squad_total_minus_the_best_xi(self):
        squad = _squad(_POINTS)
        grid = pd.DataFrame({10: _POINTS}, index=range(1, 16))
        value = chips.bench_boost_value(squad, _matrix(squad, grid, [10]))
        assert value[10] == pytest.approx(_SQUAD_TOTAL - _XI_TOTAL)

    def test_a_squad_of_equal_players_benches_four_of_them(self):
        """Eleven of fifteen start, so four identical players are the bench."""
        squad = _squad([2.0] * 15)
        grid = pd.DataFrame({10: [2.0] * 15}, index=range(1, 16))
        value = chips.bench_boost_value(squad, _matrix(squad, grid, [10]))
        assert value[10] == pytest.approx(8.0)

    def test_it_is_never_negative(self):
        squad = _squad([0.0] * 15)
        grid = pd.DataFrame({10: [0.0] * 15}, index=range(1, 16))
        assert chips.bench_boost_value(squad, _matrix(squad, grid, [10]))[10] == 0.0


class TestTripleCaptain:
    def test_it_is_one_extra_copy_of_the_best_starter(self):
        """The armband already doubles, so tripling adds one more copy."""
        squad = _squad(_POINTS)
        grid = pd.DataFrame({10: _POINTS}, index=range(1, 16))
        value = chips.triple_captain_value(squad, _matrix(squad, grid, [10]))
        assert value[10] == pytest.approx(max(_POINTS))

    def test_the_captain_is_drawn_from_the_xi_not_the_squad(self):
        """Only a starter can wear the armband.

        For a legal fifteen these happen to coincide -- the better keeper always
        starts and the best outfielder always fills a minimum -- so this cannot
        be shown with a counterexample, only by asserting the captain is one of
        the eleven. Worth pinning anyway: a valuation that read the whole squad
        would be right today and wrong the moment a formation rule changed.
        """
        from scripts.common.optimization import best_xi_index
        points = list(_POINTS)
        points[1] = 99.0                      # a keeper who is now first choice
        squad = _squad(points)
        grid = pd.DataFrame({10: points}, index=range(1, 16))
        matrix = _matrix(squad, grid, [10])
        value = chips.triple_captain_value(squad, matrix)[10]

        frame = pd.DataFrame({"Position": squad["Position"].map(
            lambda p: {"GKP": "G", "DEF": "D", "MID": "M", "FWD": "F"}[p]),
            "_p": matrix[10]})
        xi = matrix[10].loc[best_xi_index(frame, "_p")]
        assert value == pytest.approx(xi.max())
        assert value in set(xi.values)

    def test_it_never_exceeds_the_squad_total(self):
        squad = _squad(_POINTS)
        grid = pd.DataFrame({10: _POINTS}, index=range(1, 16))
        value = chips.triple_captain_value(squad, _matrix(squad, grid, [10]))
        assert value[10] <= _SQUAD_TOTAL


class TestPositionCodesCannotSilentlyDisableIt:
    """Mismatched position codes do not raise -- they match no group, so the
    formation gate silently does nothing. That is the failure that once had
    every Power Rankings team scoring exactly 50."""

    @pytest.mark.parametrize("codes", [
        ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
        ["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3,
        ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3,
        [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4],
    ])
    def test_every_spelling_gives_the_same_answer(self, codes):
        squad = _squad(_POINTS, positions=codes)
        grid = pd.DataFrame({10: _POINTS}, index=range(1, 16))
        matrix = _matrix(squad, grid, [10])
        assert chips.bench_boost_value(squad, matrix)[10] == pytest.approx(
            _SQUAD_TOTAL - _XI_TOTAL)


class TestPointsMatrix:
    def test_a_missing_player_scores_zero_not_nan(self):
        """A blank gameweek and an unpriced player both score nothing.

        Left NaN, a chip's value would depend on how many of the fifteen happen
        to be unpriced, and a *smaller* bench would read as a better Bench
        Boost week.
        """
        squad = _squad(_POINTS)
        grid = pd.DataFrame({10: [5.0]}, index=[1])      # only one of fifteen
        matrix = _matrix(squad, grid, [10])
        assert matrix[10].notna().all()
        assert matrix[10].sum() == pytest.approx(5.0)

    def test_it_reads_the_element_id_under_any_spelling(self):
        grid = pd.DataFrame({10: _POINTS}, index=range(1, 16))
        for col in ("Player_ID", "element_id", "element", "player_id"):
            squad = _squad(_POINTS).rename(columns={"element_id": col})
            matrix = _matrix(squad, grid, [10])
            assert matrix[10].sum() == pytest.approx(_SQUAD_TOTAL), col

    def test_a_frame_with_no_id_is_loud(self):
        squad = _squad(_POINTS).drop(columns=["element_id"])
        with pytest.raises(KeyError):
            _matrix(squad, pd.DataFrame({10: [1.0]}, index=[1]), [10])

    def test_an_empty_grid_gives_zeroes_rather_than_raising(self):
        squad = _squad(_POINTS)
        matrix = _matrix(squad, pd.DataFrame(), [10, 11])
        assert list(matrix.columns) == [10, 11]
        assert (matrix == 0).all().all()


class TestDeadWeight:
    def test_it_counts_the_unavailable_and_the_unprojected(self):
        squad = _squad(_POINTS)
        squad.loc[0, "status"] = "i"          # injured
        squad.loc[14, "Proj"] = 0.0           # nobody expects him to start
        assert chips.dead_weight_count(squad) == 2

    def test_a_fit_projected_squad_has_none(self):
        squad = _squad([3.0] * 15)
        squad["status"] = "a"
        assert chips.dead_weight_count(squad) == 0

    def test_one_player_is_not_counted_twice(self):
        squad = _squad(_POINTS)
        squad.loc[0, "status"] = "i"
        squad.loc[0, "Proj"] = 0.0
        assert chips.dead_weight_count(squad) == 1


def _windows(*specs):
    return {"chips": [{"name": n, "start_event": a, "stop_event": b,
                       "chip_type": "team"} for n, a, b in specs]}


class TestChipDefinitions:
    def test_it_reads_the_published_windows(self):
        defs = chips.parse_chip_definitions(
            _windows(("bboost", 1, 19), ("bboost", 20, 38)))
        assert [(d.name, d.start_event, d.stop_event) for d in defs] == [
            ("bboost", 1, 19), ("bboost", 20, 38)]

    @pytest.mark.parametrize("payload", [None, {}, {"chips": []},
                                         {"chips": "nonsense"}])
    def test_an_unreadable_payload_falls_back_rather_than_reporting_none(self, payload):
        """Unknown must never render as "you have no chips"."""
        defs = chips.parse_chip_definitions(payload)
        assert len(defs) == len(chips.DEFAULT_CHIP_WINDOWS)
        assert {d.name for d in defs} == set(chips.KNOWN_CHIPS)

    def test_the_fallback_matches_what_fpl_publishes(self):
        """All four chips, twice. Measured live 2026-11-06.

        If this ever disagrees with the live payload, `check_chip_definitions`
        says so against the real bootstrap -- this only pins the fallback.
        """
        names = [d.name for d in chips.DEFAULT_CHIP_WINDOWS]
        assert len(chips.DEFAULT_CHIP_WINDOWS) == 8
        assert all(names.count(c) == 2 for c in chips.KNOWN_CHIPS)

    def test_a_malformed_entry_is_skipped_not_fatal(self):
        defs = chips.parse_chip_definitions({"chips": [
            {"name": "bboost", "start_event": 1, "stop_event": 19},
            {"name": "", "start_event": 1, "stop_event": 19},
            {"name": "3xc", "start_event": None, "stop_event": 19},
            {"name": "freehit", "start_event": 20, "stop_event": 2},
        ]})
        assert [(d.name, d.start_event) for d in defs] == [("bboost", 1)]


class TestChipStates:
    def _states(self, used, current_gw, windows=None):
        defs = chips.parse_chip_definitions(windows) if windows else list(
            chips.DEFAULT_CHIP_WINDOWS)
        history = [{"name": n, "event": e} for n, e in used]
        return chips.chip_states(defs, history, current_gw)

    def test_an_untouched_chip_is_playable(self):
        state = self._states([], 10)["bboost"]
        assert state.playable_now
        assert state.used_in == ()

    def test_a_spent_window_is_not_playable_but_reopens(self):
        state = self._states([("freehit", 7)], 10)["freehit"]
        assert not state.playable_now
        assert state.used_in == (7,)
        assert state.next_window_opens == 20

    def test_the_second_window_is_playable_once_it_opens(self):
        state = self._states([("freehit", 7)], 20)["freehit"]
        assert state.playable_now
        assert state.next_window_opens is None

    def test_both_windows_spent_is_gone(self):
        state = self._states([("3xc", 5), ("3xc", 25)], 30)["3xc"]
        assert not state.playable_now
        assert state.next_window_opens is None

    def test_an_unplayed_window_is_forfeited_once_it_closes(self):
        """FPL does not carry a first-half chip into the second half."""
        state = self._states([], 25)["bboost"]
        assert state.playable_now           # the *second* one
        assert state.window == (20, 38)

    def test_gws_left_counts_the_gameweek_you_can_still_act_on(self):
        """GW10 through GW19 inclusive is ten gameweeks, not nine."""
        state = self._states([], 10)["bboost"]
        assert state.expires_after == 19
        assert state.gws_left == 10

    def test_a_chip_is_not_playable_before_its_window_opens(self):
        """Wildcard and Free Hit are published as opening at GW2."""
        state = self._states([], 1)["wildcard"]
        assert not state.playable_now
        assert state.next_window_opens == 2

    def test_a_published_window_beats_any_hardcoded_split(self):
        state = self._states([("bboost", 5)], 10,
                             windows=_windows(("bboost", 1, 9), ("bboost", 10, 38)))["bboost"]
        assert state.playable_now           # second window opened at GW10

    def test_every_use_is_recorded(self):
        """The advisor rendered `events[0]`, so a chip spent twice showed one."""
        state = self._states([("wildcard", 3), ("wildcard", 22)], 25)["wildcard"]
        assert state.used_in == (3, 22)


class TestH2HValue:
    """The game-theory half: a chip is worth what it changes, not what it scores."""

    SIGMA = 17.3        # measured from a live Classic H2H league

    def test_the_optimum_is_trailing_by_half_the_gain(self):
        """`d(dP)/dd = 0` where `|d + g| = |d|`, i.e. at `d = -g/2`.

        Checked against a numeric sweep of the real curve rather than against
        the formula, so the two have to agree.
        """
        gain = 12.0
        best = max(
            ((chips.h2h_chip_value(gain, 50 + d / 10.0, 50, self.SIGMA), d / 10.0)
             for d in range(-600, 601)),
            key=lambda pair: pair[0],
        )
        assert best[1] == pytest.approx(chips.ideal_h2h_margin(gain), abs=0.15)
        assert chips.ideal_h2h_margin(gain) == -6.0

    def test_a_weak_opponent_wastes_the_chip(self):
        """The instinct that prompted this: don't burn it on a win you have."""
        level = chips.h2h_chip_value(12, 50, 50, self.SIGMA)
        ahead = chips.h2h_chip_value(12, 95, 50, self.SIGMA)
        assert ahead < level / 5          # measured: +0.023 against +0.188

    def test_but_so_does_a_much_stronger_one(self):
        """The half the usual framing gets wrong -- you lose anyway."""
        level = chips.h2h_chip_value(12, 50, 50, self.SIGMA)
        behind = chips.h2h_chip_value(12, 5, 50, self.SIGMA)
        assert behind < level / 3         # measured: +0.056 against +0.188

    def test_a_close_match_beats_both(self):
        close = chips.h2h_chip_value(12, 44, 50, self.SIGMA)
        for margin in (-45, -25, 10, 25, 45):
            assert close >= chips.h2h_chip_value(12, 50 + margin, 50, self.SIGMA)

    def test_it_is_a_probability_change_so_it_stays_in_range(self):
        for margin in (-60, -10, 0, 10, 60):
            delta = chips.h2h_chip_value(20, 50 + margin, 50, self.SIGMA)
            assert 0.0 <= delta <= 1.0

    def test_a_chip_worth_no_points_changes_nothing(self):
        assert chips.h2h_chip_value(0, 50, 50, self.SIGMA) == pytest.approx(0.0)

    def test_a_degenerate_sigma_is_declined_rather_than_guessed(self):
        """A zero spread turns a one-point edge into a certainty."""
        assert chips.h2h_chip_value(12, 50, 50, 0) is None

    def test_missing_inputs_give_none(self):
        assert chips.h2h_chip_value(12, float("nan"), 50, self.SIGMA) is None
        assert chips.h2h_chip_value(None, 50, 50, self.SIGMA) is None

    def test_win_probability_is_a_half_at_level(self):
        assert chips.win_probability(0, self.SIGMA) == pytest.approx(0.5)


class TestBestWeek:
    def test_it_picks_the_largest(self):
        assert chips.best_week({10: 1.0, 11: 5.0, 12: 3.0}) == (11, 5.0)

    def test_a_tie_prefers_the_nearer_gameweek(self):
        """A forecast loses about a third of its ordering by the third week out
        (`projection_accuracy.score_horizon_lengths`), so two weeks projected
        equal are not equally trustworthy."""
        assert chips.best_week({10: 5.0, 14: 5.0}) == (10, 5.0)

    def test_a_minimum_can_rule_everything_out(self):
        assert chips.best_week({10: 0.4, 11: 0.2}, minimum=1.0) is None

    def test_nothing_to_choose_from_is_none(self):
        assert chips.best_week({}) is None
        assert chips.best_week({10: float("nan")}) is None
