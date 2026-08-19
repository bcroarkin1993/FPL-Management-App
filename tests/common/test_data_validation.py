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
