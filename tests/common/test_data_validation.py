"""Tests for scripts/common/data_validation.py.

Every "bad data" case below is a real number this app actually displayed, not a
hypothetical. If a check here stops firing, the corresponding bug can ship again.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common.data_validation import (
    Issue,
    check_projected_team_total,
    check_score_std,
    check_single_gw_projections,
    check_source_scale_agreement,
    check_element_states,
    check_initial_squad,
    check_merge_match_rate,
    check_team_strength,
    check_win_probability,
    format_issues,
    raise_on_error,
)


def _errors(issues):
    return [i for i in issues if i.severity == "error"]


def _healthy_projection_table(n=220, seed=0):
    """A table shaped like Rotowire's real weekly rankings: 20 teams x 11
    projected starters, points clustered around 4.5."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "Player": ["Player %d" % i for i in range(n)],
        "Team": ["T%02d" % (i % 20) for i in range(n)],
        "Points": np.clip(rng.normal(4.5, 1.0, n), 0.5, 9.0),
    })


class TestSingleGwProjections:
    def test_healthy_table_passes(self):
        assert check_single_gw_projections(_healthy_projection_table()) == []

    def test_cumulative_multi_gw_table_is_rejected(self):
        """The actual bug: Rotowire's "gameweeks 1-5" article, whose Points
        column was a 5-week cumulative total (mean 22.5, median 21.7, min 18.1,
        100 rows). Rotowire headed the column "Adj Total", not "Pts".
        Kelleher showed 19.6 for a single gameweek."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "Player": ["Player %d" % i for i in range(100)],
            "Points": np.clip(rng.normal(22.5, 3.6, 100), 18.1, 37.7),
        })
        errors = _errors(check_single_gw_projections(df, source="rotowire GW1-5"))
        assert errors, "a median of 21.7 points in one gameweek must be rejected"
        assert any("multi-gameweek" in e.hint for e in errors)

    def test_season_long_table_is_rejected(self):
        """Rotowire's top-400 season rankings: right shape, wrong magnitude."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({"Points": np.clip(rng.normal(120, 30, 400), 20, 260)})
        assert _errors(check_single_gw_projections(df, source="season rankings"))

    def test_all_zero_table_is_rejected(self):
        """What a source that hasn't published yet looks like -- or total
        name-matching failure."""
        df = pd.DataFrame({"Points": [0.0] * 220})
        assert _errors(check_single_gw_projections(df, source="unpublished"))

    def test_truncated_table_is_rejected(self):
        df = _healthy_projection_table(n=12)
        errors = _errors(check_single_gw_projections(df))
        assert any("rows" in e.message for e in errors)

    def test_single_absurd_player_is_rejected(self):
        df = _healthy_projection_table()
        df.loc[0, "Points"] = 45.0
        errors = _errors(check_single_gw_projections(df))
        assert any("top projection" in e.message for e in errors)

    def test_empty_and_missing_column_are_rejected(self):
        assert _errors(check_single_gw_projections(None))
        assert _errors(check_single_gw_projections(pd.DataFrame()))
        assert _errors(check_single_gw_projections(pd.DataFrame({"Player": ["x"] * 200})))

    def test_negative_projection_warns(self):
        df = _healthy_projection_table()
        df.loc[0, "Points"] = -3.0
        issues = check_single_gw_projections(df)
        assert any(i.severity == "warning" and "negative" in i.message for i in issues)


class TestScoreStd:
    def test_realistic_sigma_passes(self):
        assert check_score_std(14.2, n_samples=300) == []

    def test_zero_sigma_is_rejected(self):
        """The actual bug: preseason, every historical score is 0, so the std
        was 0.0 and every fixture resolved to a near-certainty."""
        errors = _errors(check_score_std(0.0, n_samples=380))
        assert errors
        assert "step function" in errors[0].message

    def test_none_and_nan_are_rejected(self):
        assert _errors(check_score_std(None))
        assert _errors(check_score_std(float("nan")))
        assert _errors(check_score_std(float("inf")))

    def test_negative_sigma_is_rejected(self):
        assert _errors(check_score_std(-5.0))

    def test_cumulative_total_sigma_is_rejected(self):
        """Taking the std over cumulative season totals instead of per-gameweek
        scores lands far too high, flattening every fixture to 50/50."""
        assert _errors(check_score_std(650.0, n_samples=380))

    def test_non_numeric_is_rejected(self):
        assert _errors(check_score_std("fifteen"))


class TestWinProbability:
    def test_near_tie_reads_as_a_coin_flip(self):
        assert check_win_probability(0.52, 55.8, 54.7) == []

    def test_reported_bug_is_rejected(self):
        """55.8 vs 54.7 displayed as 85%/15%."""
        errors = _errors(check_win_probability(0.85, 55.8, 54.7))
        assert errors
        assert any("near a coin flip" in e.hint for e in errors)

    def test_lopsided_fixture_may_be_lopsided(self):
        assert check_win_probability(0.83, 39.7, 19.6) == []

    def test_extreme_call_on_a_small_gap_is_rejected(self):
        assert _errors(check_win_probability(0.97, 50.0, 46.0))

    def test_probability_must_follow_the_scoreline(self):
        errors = _errors(check_win_probability(0.20, 60.0, 40.0))
        assert any("favours the lower-projected" in e.message for e in errors)

    def test_percentage_instead_of_probability_is_rejected(self):
        assert _errors(check_win_probability(85.0, 55.8, 54.7))

    def test_non_finite_is_rejected(self):
        assert _errors(check_win_probability(None, 50.0, 50.0))
        assert _errors(check_win_probability(float("nan"), 50.0, 50.0))


class TestProjectedTeamTotal:
    def test_normal_xi_passes(self):
        assert check_projected_team_total(49.6, 11, label="Stoned Squirrels") == []

    def test_inflated_total_is_rejected(self):
        """What the 5-gameweek article produced: an XI summing to ~200."""
        assert _errors(check_projected_team_total(203.4, 11))

    def test_weak_squad_warns_but_does_not_fail(self):
        """19.6 is legitimate -- six of the XI are not expected to start, and
        absence from the projected-starter list is exactly that signal."""
        issues = check_projected_team_total(19.6, 11, label="Chappy's Goats")
        assert _errors(issues) == []

    def test_wrong_lineup_size_is_rejected(self):
        assert _errors(check_projected_team_total(45.0, 9))

    def test_negative_and_non_finite_are_rejected(self):
        assert _errors(check_projected_team_total(-1.0, 11))
        assert _errors(check_projected_team_total(float("nan"), 11))


class TestSourceScaleAgreement:
    def test_same_scale_passes(self):
        rng = np.random.default_rng(3)
        a = rng.normal(4.5, 1.0, 200)
        b = rng.normal(4.2, 1.2, 200)
        assert check_source_scale_agreement(a, b, "rotowire", "ffp") == []

    def test_multi_gw_versus_single_gw_is_rejected(self):
        """The signature of the original bug, catchable without knowing which
        source is wrong: one is ~5x the other."""
        rng = np.random.default_rng(4)
        a = rng.normal(22.5, 3.6, 200)   # 5-gameweek cumulative
        b = rng.normal(4.5, 1.0, 200)    # single gameweek
        errors = _errors(check_source_scale_agreement(a, b, "rotowire", "ffp"))
        assert errors
        assert "not in the same units" in errors[0].hint

    def test_unpublished_source_warns_rather_than_failing(self):
        """FFP's Predicted column is all zeros until they publish the gameweek."""
        issues = check_source_scale_agreement([4.5] * 200, [0.0] * 200, "rotowire", "ffp")
        assert _errors(issues) == []
        assert any(i.severity == "warning" for i in issues)


class TestReporting:
    def test_raise_on_error_raises_only_for_errors(self):
        raise_on_error([Issue("c", "warning", "just odd")])
        with pytest.raises(AssertionError, match="Implausible data in the overview"):
            raise_on_error([Issue("c", "error", "impossible")], context="the overview")

    def test_format_issues_is_readable(self):
        text = format_issues([Issue("c", "error", "impossible", "do this instead")])
        assert "ERROR" in text and "impossible" in text and "do this instead" in text
        assert format_issues([]) == "no issues"


class TestCheckTeamStrength:
    """Draft power-ranking scores.

    Every "bad" fixture below is a number this model could actually emit.
    """

    @staticmethod
    def _good(n=4):
        import numpy as np
        return pd.DataFrame({
            "Team_Name": [f"T{i}" for i in range(n)],
            "Score": np.linspace(72, 41, n),
            "Healthy_Score": np.linspace(75, 43, n),
            "Injury_Cost": np.linspace(3, 2, n),
            "GK": np.linspace(70, 40, n),
            "DEF": np.linspace(80, 44, n),
            "MID": np.linspace(76, 39, n),
            "FWD": np.linspace(65, 35, n),
            "Players": [15] * n,
        })

    def test_plausible_table_passes(self):
        assert check_team_strength(self._good()) == []

    def test_empty_table_is_an_error(self):
        issues = check_team_strength(pd.DataFrame())
        assert any(i.severity == "error" for i in issues)

    def test_none_is_an_error(self):
        assert any(i.severity == "error" for i in check_team_strength(None))

    def test_all_teams_identical_is_an_error(self):
        """The position-code bug: every player defaults to 0.5, every team to 50.0."""
        df = self._good()
        for col in ("Score", "Healthy_Score", "GK", "DEF", "MID", "FWD"):
            df[col] = 50.0
        df["Injury_Cost"] = 0.0
        issues = check_team_strength(df)
        assert any(i.severity == "error" for i in issues)
        assert any("percentile join" in i.hint for i in issues)

    def test_score_above_one_hundred_is_an_error(self):
        df = self._good()
        df.loc[0, "Score"] = 340.0
        assert any(i.severity == "error" and "0-100" in i.message
                   for i in check_team_strength(df))

    def test_negative_score_is_an_error(self):
        df = self._good()
        df.loc[0, "DEF"] = -12.0
        assert any(i.severity == "error" for i in check_team_strength(df))

    def test_short_squad_is_an_error(self):
        df = self._good()
        df.loc[0, "Players"] = 13
        assert any(i.severity == "error" and "players" in i.message
                   for i in check_team_strength(df))

    def test_wrong_team_count_is_an_error(self):
        assert any(i.severity == "error"
                   for i in check_team_strength(self._good(n=4), expected_teams=10))

    def test_negative_injury_cost_is_an_error(self):
        df = self._good()
        df.loc[0, "Injury_Cost"] = -5.0
        assert any(i.severity == "error" for i in check_team_strength(df))

    def test_absurd_injury_cost_is_an_error(self):
        df = self._good()
        df.loc[0, "Injury_Cost"] = 85.0
        assert any(i.severity == "error" for i in check_team_strength(df))

    def test_all_zero_scores_is_an_error(self):
        df = self._good()
        for col in ("Score", "Healthy_Score", "GK", "DEF", "MID", "FWD", "Injury_Cost"):
            df[col] = 0.0
        assert any(i.severity == "error" for i in check_team_strength(df))


class TestCheckMergeMatchRate:
    """The tripwire for a name merge that quietly stops matching.

    The real numbers: Rotowire's season rankings hold 425 rows, and the old
    strict (name, team) key matched 356 of them. Nothing raised. The 69 misses
    -- Bruno Fernandes, Gabriel, Alisson, David Raya, Ruben Dias among them --
    each fell back to a neutral 0.5 percentile, so the #2 asset in the game
    rendered as an exactly average player.
    """

    def test_full_match_is_silent(self):
        assert check_merge_match_rate(425, 425, "season rankings") == []

    def test_healthy_match_rate_is_silent(self):
        assert check_merge_match_rate(410, 425, "season rankings") == []

    def test_the_real_regression_is_flagged(self):
        issues = check_merge_match_rate(356, 425, "season rankings")
        assert issues and "356/425" in issues[0].message

    def test_severe_miss_rate_is_an_error(self):
        assert _errors(check_merge_match_rate(100, 425, "season rankings"))

    def test_total_collapse_is_an_error(self):
        assert _errors(check_merge_match_rate(0, 425, "season rankings"))

    def test_empty_reference_warns_rather_than_dividing_by_zero(self):
        issues = check_merge_match_rate(0, 0, "season rankings")
        assert issues and issues[0].severity == "warning"


class TestCheckInitialSquad:
    """A legal, sensibly-priced 15-man Classic squad."""

    @staticmethod
    def _good():
        positions = ["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3
        return pd.DataFrame({
            "Player": ["P%d" % i for i in range(15)],
            "Position": positions,
            "Team": ["T%d" % (i % 8) for i in range(15)],
            "Price": [6.5] * 15,
            "ExpPts": [4.7] * 11 + [3.0] * 4,
            "Is_Starter": [True] * 11 + [False] * 4,
        })

    def test_valid_squad_is_silent(self):
        assert check_initial_squad(self._good(), 100.0) == []

    def test_empty_squad_is_an_error(self):
        assert _errors(check_initial_squad(pd.DataFrame(), 100.0))
        assert _errors(check_initial_squad(None, 100.0))

    def test_wrong_squad_size_is_an_error(self):
        assert _errors(check_initial_squad(self._good().head(14), 100.0))

    def test_wrong_starter_count_is_an_error(self):
        df = self._good()
        df.loc[11, "Is_Starter"] = True
        assert _errors(check_initial_squad(df, 100.0))

    def test_position_quota_violation_is_an_error(self):
        df = self._good()
        df.loc[0, "Position"] = "D"
        assert _errors(check_initial_squad(df, 100.0))

    def test_gkp_style_position_codes_are_caught(self):
        """analytics.py groups on G/D/M/F; the bootstrap supplies GKP/DEF/MID/FWD.

        Feeding the long codes through matches no quota at all, which is the
        same class of silent failure check_team_strength() guards against.
        """
        df = self._good()
        df["Position"] = df["Position"].map(
            {"G": "GKP", "D": "DEF", "M": "MID", "F": "FWD"})
        assert _errors(check_initial_squad(df, 100.0))

    def test_more_than_three_from_one_club_is_an_error(self):
        df = self._good()
        df.loc[:3, "Team"] = "MCI"
        assert _errors(check_initial_squad(df, 100.0))

    def test_over_budget_is_an_error(self):
        df = self._good()
        df["Price"] = 8.0
        assert _errors(check_initial_squad(df, 100.0))

    def test_underspend_warns(self):
        """The visible symptom of a scale-free objective.

        When the ILP maximizes percentiles, a premium can never repay its price,
        so the solver buys a flat mid-price squad and banks the change.
        """
        df = self._good()
        df["Price"] = 5.0  # 75.0 total against a 100.0 budget
        issues = check_initial_squad(df, 100.0)
        assert issues and all(i.severity == "warning" for i in issues)

    def test_percentile_scale_objective_is_an_error(self):
        """An XI summing to ~10 means the objective is still in percentiles."""
        df = self._good()
        df["ExpPts"] = 0.9
        assert _errors(check_initial_squad(df, 100.0))

    def test_season_totals_not_divided_down_is_an_error(self):
        """An XI summing to ~2000 means season totals never became a per-GW rate."""
        df = self._good()
        df["ExpPts"] = 180.0
        assert _errors(check_initial_squad(df, 100.0))

    def test_missing_projection_is_an_error(self):
        """pandas .sum() skips NaN, so an unprojected starter would otherwise
        contribute 0 to the objective and leave a plausible-looking total."""
        df = self._good()
        df.loc[0, "ExpPts"] = np.nan
        issues = _errors(check_initial_squad(df, 100.0))
        assert issues and "no ExpPts value" in issues[0].message

    def test_missing_optional_columns_are_tolerated(self):
        df = self._good().drop(columns=["ExpPts", "Team"])
        assert check_initial_squad(df, 100.0) == []


class TestCheckElementStates:
    """The Draft element-status endpoint, shaped like the real one.

    The bug: the Waiver Wire suggested Oliver McBurnie, who had just been dropped
    by another manager and was therefore *locked* — on nobody's roster, but not
    claimable either. Ownership data alone cannot tell the two apart.
    """

    @staticmethod
    def _states(n_teams=10, n_locked=20, n_available=446):
        """616 elements split the way league 11347 really was: 150 owned by 10
        teams of 15, 20 locked, the rest available."""
        states = {}
        element = 1
        for team in range(n_teams):
            for _ in range(15):
                states[element] = {"status": "o", "owner": 56000 + team,
                                   "in_accepted_trade": False}
                element += 1
        for _ in range(n_locked):
            states[element] = {"status": "l", "owner": None, "in_accepted_trade": False}
            element += 1
        for _ in range(n_available):
            states[element] = {"status": "a", "owner": None, "in_accepted_trade": False}
            element += 1
        return states

    def test_the_real_payload_is_clean(self):
        assert check_element_states(self._states(), expected_teams=10) == []

    def test_empty_map_is_an_error(self):
        """An empty map makes the page fall back to 'everyone is available',
        which is exactly the state that produced the McBurnie suggestion."""
        assert _errors(check_element_states({}))
        assert _errors(check_element_states(None))

    def test_unknown_status_code_is_an_error(self):
        states = self._states()
        states[1]["status"] = "x"
        issues = _errors(check_element_states(states))
        assert issues and "unrecognised status code" in issues[0].message

    def test_owned_player_without_an_owner_is_an_error(self):
        states = self._states()
        states[1]["owner"] = None
        assert _errors(check_element_states(states))

    def test_unowned_player_with_an_owner_is_an_error(self):
        states = self._states()
        locked_id = next(k for k, v in states.items() if v["status"] == "l")
        states[locked_id]["owner"] = 56000
        assert _errors(check_element_states(states))

    def test_owned_count_must_match_squad_arithmetic(self):
        """Draft squads are a fixed 15, so 10 teams own exactly 150 players."""
        states = self._states(n_teams=9)
        assert _errors(check_element_states(states, expected_teams=10))

    def test_everything_locked_is_a_warning(self):
        """If 'l' were ever read as something broader, the waiver wire would
        empty out. Warn rather than fail — the boundary is judgement, not law."""
        states = self._states(n_locked=400, n_available=66)
        issues = check_element_states(states)
        assert issues and all(i.severity == "warning" for i in issues)

    def test_no_available_players_is_an_error(self):
        states = self._states(n_available=0)
        assert _errors(check_element_states(states))
