"""Tests for scripts/common/data_validation.py.

Every "bad data" case below is a real number this app actually displayed, not a
hypothetical. If a check here stops firing, the corresponding bug can ship again.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common.data_validation import (
    check_ffp_feed,
    check_transfer_risk,
    check_transfer_windows,
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
    check_resolved_squad,
    check_blended_projections,
    check_free_transfers,
    check_transfer_plan,
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

    def test_a_frame_smaller_than_the_reference_is_not_judged(self):
        """The Waiver Wire merges ~105 available players against a 424-row
        reference, so it can never claim more than a quarter of them. It logged
        an ERROR at "24.8%" on every page load while matching 100% of its input,
        and a check that cries wolf is a check nobody reads."""
        assert check_merge_match_rate(105, 424, "season rankings", input_rows=105) == []
        assert check_merge_match_rate(60, 424, "season rankings", input_rows=105) == []

    def test_full_pool_regression_still_fires_with_input_rows(self):
        """622 pool rows against 425 reference rows: the reference is the binding
        constraint, so the original 356/425 regression must still be caught."""
        issues = check_merge_match_rate(356, 425, "season rankings", input_rows=622)
        assert issues and "356/425" in issues[0].message

    def test_empty_input_frame_is_not_a_failure(self):
        assert check_merge_match_rate(0, 425, "season rankings", input_rows=0) == []

    def test_a_full_pool_caller_is_still_judged(self):
        """622 pool rows against 425 reference rows: the caller could have
        claimed every reference row, so the floor still applies."""
        assert check_merge_match_rate(420, 425, "season rankings", input_rows=622) == []
        assert _errors(check_merge_match_rate(100, 425, "season rankings", input_rows=622))


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


class TestCheckTransferRisk:
    """Every failure mode here is silent: a broken feed renders a page identical
    to a working one, just with nobody discounted — which is the state that let
    Watkins be drafted at rank 32 in the first place."""

    @staticmethod
    def _frame(n_clean=60, **overrides):
        data = {
            "Transfer_Risk": [0.0] * n_clean,
            "Transfer_Mult": [1.0] * n_clean,
            "Transfer_Outlets": [0] * n_clean,
            "Transfer_Note": [""] * n_clean,
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_healthy_frame_passes(self):
        df = self._frame()
        df.loc[0, ["Transfer_Risk", "Transfer_Mult", "Transfer_Outlets"]] = [0.79, 0.34, 5]
        assert not _errors(check_transfer_risk(df))

    def test_empty_frame_is_an_error(self):
        assert _errors(check_transfer_risk(pd.DataFrame()))
        assert _errors(check_transfer_risk(None))

    def test_missing_columns_is_an_error(self):
        assert _errors(check_transfer_risk(pd.DataFrame({"Player": ["x"]})))

    def test_risk_outside_probability_range_is_an_error(self):
        assert _errors(check_transfer_risk(
            pd.DataFrame({"Transfer_Risk": [1.4], "Transfer_Mult": [0.5]})))

    def test_multiplier_above_one_is_an_error(self):
        """Above 1.0 would *inflate* a player's season projection."""
        assert _errors(check_transfer_risk(
            pd.DataFrame({"Transfer_Risk": [0.3], "Transfer_Mult": [1.6]})))

    def test_multiplier_below_the_floor_is_an_error(self):
        assert _errors(check_transfer_risk(
            pd.DataFrame({"Transfer_Risk": [0.3], "Transfer_Mult": [0.01]})))

    def test_whole_league_at_risk_is_an_error(self):
        """A window moves a handful of players. A third of the league means the
        matcher broke and attached one player's news to everybody."""
        df = self._frame(n_clean=0, **{
            "Transfer_Risk": [0.8] * 30 + [0.0] * 70,
            "Transfer_Mult": [0.3] * 30 + [1.0] * 70,
            "Transfer_Outlets": [5] * 100,
            "Transfer_Note": ["x"] * 100,
        })
        assert _errors(check_transfer_risk(df))

    def test_fraction_check_ignores_tiny_frames(self):
        """One genuinely at-risk player is 25% of a four-row frame. A check that
        cries wolf gets muted."""
        df = pd.DataFrame({"Transfer_Risk": [0.8, 0.0, 0.0, 0.0],
                           "Transfer_Mult": [0.3, 1.0, 1.0, 1.0]})
        assert not _errors(check_transfer_risk(df))

    def test_single_outlet_high_risk_warns(self):
        df = self._frame()
        df.loc[0, ["Transfer_Risk", "Transfer_Mult", "Transfer_Outlets"]] = [0.9, 0.3, 1]
        issues = check_transfer_risk(df)
        assert issues and all(i.severity == "warning" for i in issues)

    def test_resolved_departure_does_not_trip_the_outlet_warning(self):
        """The bootstrap said so; it needs no corroborating newspapers."""
        df = self._frame()
        df.loc[0, ["Transfer_Risk", "Transfer_Mult", "Transfer_Outlets"]] = [1.0, 0.1, 0]
        df.loc[0, "Transfer_Note"] = "Departed — Al Qadsiah"
        assert not check_transfer_risk(df)


class TestCheckTransferWindows:
    """The window calendar is hardcoded and cannot be discovered. Once it lapses,
    exposure is permanently 0 and the whole feature is a silent no-op."""

    def test_current_windows_pass(self):
        from datetime import date
        from scripts.common.transfer_risk import TRANSFER_WINDOWS
        assert not check_transfer_windows(TRANSFER_WINDOWS, today=date(2026, 8, 29))

    def test_lapsed_calendar_warns(self):
        from datetime import date
        from scripts.common.transfer_risk import TRANSFER_WINDOWS
        issues = check_transfer_windows(TRANSFER_WINDOWS, today=date(2029, 1, 1))
        assert issues and all(i.severity == "warning" for i in issues)

    def test_empty_calendar_is_an_error(self):
        assert _errors(check_transfer_windows({}))


class TestTransferRiskGroundTruthExclusion:
    """A frame of confirmed departures is not a broken matcher.

    The Availability tracker deliberately lists departed players, so 80%+ of its
    rows legitimately score 1.0. Counting those against the at-risk fraction made
    the check fire on a page that was working perfectly — and a check that cries
    wolf gets muted.
    """

    @staticmethod
    def _rows(n, **kw):
        base = {"Transfer_Risk": 0.0, "Transfer_Mult": 1.0,
                "Transfer_Status": "", "Transfer_Note": ""}
        base.update(kw)
        return [dict(base) for _ in range(n)]

    def test_confirmed_departures_do_not_trip_the_fraction_check(self):
        df = pd.DataFrame(
            self._rows(60, Transfer_Risk=1.0, Transfer_Mult=0.1,
                       Transfer_Status="Departed", Transfer_Note="Departed — Al Hilal")
            + self._rows(40))
        assert not [i for i in check_transfer_risk(df) if i.severity == "error"]

    def test_a_broken_matcher_still_errors(self):
        """The signature the check exists for must survive the exclusion."""
        df = pd.DataFrame(self._rows(
            60, Transfer_Risk=0.8, Transfer_Mult=0.3, Transfer_Status="At risk",
            Transfer_Note="Real Madrid (2 outlets)"))
        assert [i for i in check_transfer_risk(df) if i.severity == "error"]

    def test_odds_weight_must_be_a_decay_factor(self):
        df = pd.DataFrame(self._rows(3, Odds_Weight=1.4))
        assert any("odds weight" in i.message for i in check_transfer_risk(df)
                   if i.severity == "error")

    def test_odds_risk_must_be_a_probability(self):
        df = pd.DataFrame(self._rows(3, Odds_Risk=1.8))
        assert any("Odds_Risk" in i.message for i in check_transfer_risk(df)
                   if i.severity == "error")


class TestCheckFfpFeed:
    """Every fixture here is a shape the live feed actually produced."""

    @staticmethod
    def _feed(n=300, start=90.0, cond=5.0, invert=False):
        conditional = [cond] * n
        unconditional = [round(c * start / 100.0, 3) for c in conditional]
        if invert:
            conditional, unconditional = unconditional, conditional
        return pd.DataFrame({
            "Name": ["Player %d" % i for i in range(n)],
            "Team": ["Arsenal"] * n,
            "Position": ["MID"] * n,
            "Start": [start] * n,
            "StartingPredicted": conditional,
            "Predicted": unconditional,
        })

    def test_a_current_feed_is_clean(self):
        assert not check_ffp_feed(self._feed(), gameweek=3, expected_gw=3, age_days=0.2)

    def test_an_empty_feed_is_an_error(self):
        issues = check_ffp_feed(pd.DataFrame(), gameweek=3, expected_gw=3)
        assert [i for i in issues if i.severity == "error"]

    def test_the_wrong_gameweek_is_an_error(self):
        """The live bug: 561 plausible rows, one gameweek behind."""
        issues = check_ffp_feed(self._feed(n=561), gameweek=2, expected_gw=3)
        errors = [i for i in issues if i.severity == "error"]
        assert errors and "GW2" in errors[0].message and "GW3" in errors[0].message

    def test_an_unknown_gameweek_warns_rather_than_errors(self):
        issues = check_ffp_feed(self._feed(), gameweek=None, expected_gw=3)
        assert issues and all(i.severity == "warning" for i in issues)

    def test_the_two_prediction_bases_being_swapped_is_an_error(self):
        """FFP's site names these the opposite way round from its spreadsheet.

        Mapped across by name instead of by basis, `Predicted` comes out larger
        than `StartingPredicted` -- and the start discount is then applied to
        the wrong one.
        """
        issues = check_ffp_feed(self._feed(invert=True), gameweek=3, expected_gw=3)
        assert any("Predicted exceeds StartingPredicted" in i.message
                   for i in issues if i.severity == "error")

    def test_start_percentages_must_be_percentages(self):
        df = self._feed()
        df["Start"] = 0.9              # a fraction where a percentage belongs
        issues = check_ffp_feed(df, gameweek=3, expected_gw=3)
        assert not [i for i in issues if "Start%" in i.message]   # 0.9 is in range
        df["Start"] = 900.0
        assert [i for i in check_ffp_feed(df, gameweek=3, expected_gw=3)
                if i.severity == "error" and "Start%" in i.message]

    def test_a_truncated_table_warns(self):
        issues = check_ffp_feed(self._feed(n=12), gameweek=3, expected_gw=3)
        assert any("only 12 FFP rows" in i.message for i in issues)

    def test_a_feed_that_stopped_moving_warns(self):
        issues = check_ffp_feed(self._feed(), gameweek=3, expected_gw=3, age_days=40)
        assert any(i.severity == "warning" and "40 days ago" in i.message for i in issues)

    def test_a_future_stamp_is_an_error(self):
        issues = check_ffp_feed(self._feed(), gameweek=3, expected_gw=3, age_days=-2)
        assert [i for i in issues if i.severity == "error" and "future" in i.message]


class TestCheckResolvedSquad:
    """Every 'bad data' fixture here is a shape the resolver can really produce."""

    def _good(self, **over):
        base = {
            "picks": [{"element": i} for i in range(1, 16)],
            "entry_history": {"bank": 15, "value": 1006},
            "source": "my_team",
            "source_gw": 3,
            "target_gw": 3,
            "is_stale": False,
            "auth_status": "ok",
        }
        base.update(over)
        return base

    def _bootstrap(self, team_of=None):
        team_of = team_of or {}
        return {"elements": [{"id": i, "team": team_of.get(i, (i % 20) + 1)}
                             for i in range(1, 40)]}

    def test_healthy_squad_is_silent(self):
        assert check_resolved_squad(self._good(), self._bootstrap()) == []

    def test_a_chip_from_another_gameweek_is_an_error(self):
        """A wildcard played in GW3 rendered as "Active Chip" throughout GW4."""
        issues = check_resolved_squad(
            self._good(source_gw=3, target_gw=4, is_stale=True,
                       active_chip="wildcard"),
            self._bootstrap())
        chip = [i for i in issues if "wildcard" in i.message]
        assert len(chip) == 1 and chip[0].severity == "error"

    def test_a_chip_on_the_target_gameweek_is_fine(self):
        assert check_resolved_squad(
            self._good(active_chip="bboost"), self._bootstrap()) == []

    def test_a_stale_squad_with_no_chip_only_warns(self):
        issues = check_resolved_squad(
            self._good(source_gw=3, target_gw=4, is_stale=True), self._bootstrap())
        assert [i.severity for i in issues] == ["warning"]

    def test_missing_resolution_is_an_error(self):
        issues = check_resolved_squad(None)
        assert [i.severity for i in issues] == ["error"]

    def test_empty_squad_is_an_error(self):
        issues = check_resolved_squad(self._good(picks=[]))
        assert issues and issues[0].severity == "error"

    def test_wrong_squad_size_is_an_error(self):
        issues = check_resolved_squad(self._good(picks=[{"element": 1}] * 14))
        assert any("14 players" in i.message for i in issues)

    def test_duplicate_players_are_an_error(self):
        picks = [{"element": i} for i in range(1, 15)] + [{"element": 1}]
        issues = check_resolved_squad(self._good(picks=picks))
        assert any("duplicate" in i.message for i in issues)

    def test_negative_bank_is_an_error(self):
        """The signature of a transfer replayed onto a squad that already
        contains it: the pick swap no-ops, the bank adjustment does not."""
        issues = check_resolved_squad(
            self._good(entry_history={"bank": -25, "value": 1006}))
        assert any("negative bank" in i.message for i in issues)

    def test_squad_value_in_the_wrong_units_is_an_error(self):
        issues = check_resolved_squad(
            self._good(entry_history={"bank": 0, "value": 100}))
        assert any("outside the plausible range" in i.message for i in issues)

    def test_too_many_players_from_one_club_is_an_error(self):
        every_player_at_club_1 = {i: 1 for i in range(1, 16)}
        issues = check_resolved_squad(
            self._good(), self._bootstrap(every_player_at_club_1))
        assert any("more than 3 players from one club" in i.message for i in issues)

    def test_players_absent_from_bootstrap_are_an_error(self):
        issues = check_resolved_squad(
            self._good(picks=[{"element": i} for i in range(900, 915)]),
            self._bootstrap())
        assert any("absent from the bootstrap" in i.message for i in issues)

    def test_stale_squad_is_a_warning_not_an_error(self):
        """The reported bug. Between gameweeks without a credential this is also
        the only available answer, so it must warn rather than fail — what it
        must never do is pass unnoticed."""
        issues = check_resolved_squad(
            self._good(source="picks", source_gw=2, target_gw=3, is_stale=True),
            self._bootstrap())
        assert issues
        assert all(i.severity == "warning" for i in issues)
        assert any("GW2" in i.message and "GW3" in i.message for i in issues)

    def test_expired_credentials_warn(self):
        issues = check_resolved_squad(
            self._good(auth_status="expired"), self._bootstrap())
        assert any("expired" in i.message for i in issues)


class TestCheckBlendedProjections:
    """The blend is the number the whole app renders, and nothing validated it.

    Every fixture below is a shape the app either shipped or would have shipped
    without the projection engine's basis contract.
    """

    def _good(self):
        return pd.DataFrame({
            "Proj_Start": [10.0, 6.0, 4.0],
            "Start_Pct": [0.9, 0.5, 1.0],
            "Proj": [9.0, 3.0, 4.0],
            "Proj_Start__rotowire": [11.0, 6.0, np.nan],
            "Proj_Start__ffp": [8.5, np.nan, 4.0],
        })

    def test_consistent_frame_is_clean(self):
        assert check_blended_projections(self._good()) == []

    def test_empty_frame_is_an_error(self):
        issues = check_blended_projections(pd.DataFrame())
        assert any(i.severity == "error" for i in issues)

    def test_expected_value_above_conditional_is_an_error(self):
        """Proj > Proj_Start means the start multiplier ran backwards."""
        df = self._good()
        df.loc[0, "Proj"] = 12.0
        issues = check_blended_projections(df)
        assert any("above Proj_Start" in i.message for i in issues)

    def test_start_pct_stored_as_a_percentage_is_caught(self):
        """0-100 instead of 0-1 inflates every projection a hundredfold."""
        df = self._good()
        df["Start_Pct"] = [90.0, 50.0, 100.0]
        issues = check_blended_projections(df)
        assert any("outside [0, 1]" in i.message for i in issues)

    def test_identity_drift_is_caught(self):
        """Proj, Proj_Start and Start_Pct are one identity. If a page hand-writes
        one of them they stop agreeing -- which is exactly how the app came to
        carry two blends that differed."""
        df = self._good()
        df.loc[1, "Proj"] = 5.5      # not 6.0 * 0.5
        issues = check_blended_projections(df)
        assert any("Proj_Start x Start_Pct" in i.message for i in issues)

    def test_blend_outside_its_own_sources_is_caught(self):
        """A weighted mean cannot escape its inputs. If it has, a source was
        converted to the wrong basis."""
        df = self._good()
        df.loc[0, "Proj_Start"] = 25.0
        df.loc[0, "Proj"] = 22.5
        issues = check_blended_projections(df)
        assert any("outside the range of its own sources" in i.message for i in issues)

    def test_missing_contract_columns_is_an_error(self):
        issues = check_blended_projections(pd.DataFrame({"Points": [5.0]}))
        assert any("missing Proj/Proj_Start" in i.message for i in issues)


def _warnings(issues):
    return [i for i in issues if i.severity == "warning"]


class TestCheckFreeTransfers:
    """The count gates every hit verdict on the Transfers page.

    Reconstructed without a credential, it was one too high all season: the
    replay seeded the bank at 1 and then credited GW1 as well, which grants
    nothing. The live symptom was 4 free transfers reported at GW5 against
    FPL's 3.
    """

    def test_a_plausible_count_is_silent(self):
        assert check_free_transfers(3, gameweek=5, chip_gws=[3]) == []

    def test_no_count_at_all_is_an_error(self):
        assert _errors(check_free_transfers(None))

    def test_a_non_numeric_count_is_an_error(self):
        assert _errors(check_free_transfers("two"))

    def test_negative_is_an_error(self):
        """Overspending is a points hit, not a negative balance."""
        assert _errors(check_free_transfers(-1, gameweek=5))

    def test_above_the_cap_is_an_error(self):
        assert _errors(check_free_transfers(6, gameweek=20))

    def test_gameweek_one_cannot_have_any(self):
        """Changes before the first deadline are unlimited, not banked."""
        assert _errors(check_free_transfers(1, gameweek=1))

    def test_more_than_have_been_awarded_is_an_error(self):
        """A quiet season reported one high: GW2-GW5 award four, not five."""
        assert _errors(check_free_transfers(5, gameweek=5))
        assert check_free_transfers(4, gameweek=5) == []

    def test_a_chip_gameweek_awards_nothing(self):
        """The live case. A wildcard in GW3 makes 4 at GW5 impossible."""
        assert _errors(check_free_transfers(4, gameweek=5, chip_gws=[3]))

    def test_the_ceiling_stays_loose_without_the_chip_list(self):
        """A plausibility check must never cry wolf, so an unknown chip
        history gets the looser bound rather than a guess."""
        assert check_free_transfers(4, gameweek=5) == []

    def test_the_ceiling_never_exceeds_the_cap(self):
        assert check_free_transfers(5, gameweek=30) == []

    def test_a_stated_limit_is_an_allowance_not_a_remainder(self):
        """limit 1 / made 2 leaves 0; reporting 1 is the bug this catches."""
        assert _errors(check_free_transfers(1, limit=1, made=2))
        assert check_free_transfers(0, limit=1, made=2) == []

    def test_fpl_saying_the_next_transfer_costs_contradicts_a_free_one(self):
        assert _warnings(check_free_transfers(2, gameweek=5, status="cost"))

    def test_fpl_saying_free_contradicts_a_count_of_zero(self):
        assert _warnings(check_free_transfers(0, gameweek=5, status="free"))

    def test_agreeing_status_is_silent(self):
        assert check_free_transfers(2, gameweek=5, status="free") == []

    def test_logging_past_the_allowance_warns_rather_than_errors(self):
        """A legitimate state -- the excess costs 4 points each -- but the
        page has to say so rather than clamp to zero silently."""
        issues = check_free_transfers(2, gameweek=5, logged=3)
        assert _warnings(issues) and not _errors(issues)


class TestCheckTransferPlan:
    """A plan is a set of moves made together, so every rule is joint.

    The brute-force planner this validates the replacement for shipped three
    ways of being individually plausible and collectively impossible: two adds
    each affordable alone against one pot, two legs naming the same incoming
    player, and two adds from one club taking it to four.
    """

    @staticmethod
    def _good():
        return {
            "legs": [
                {"out_id": 1, "in_id": 101, "out_player": "Haaland",
                 "in_player": "Wissa", "position": "F",
                 "out_price": 14.5, "in_price": 7.5, "delta": -1.0},
                {"out_id": 2, "in_id": 102, "out_player": "Gray",
                 "in_player": "Salah", "position": "M",
                 "out_price": 7.0, "in_price": 14.0, "delta": 3.0},
            ],
            "max_changes": 2, "free_transfers": 2, "hits": 0,
            "bank_before": 0.8, "bank_after": 0.8,
            "gain_net": 2.0, "horizon_gws": 2.2,
        }

    def test_a_workable_plan_is_silent(self):
        assert check_transfer_plan(self._good()) == []

    def test_no_plan_is_an_error(self):
        assert _errors(check_transfer_plan(None))
        assert _errors(check_transfer_plan({}))

    def test_more_legs_than_allowed_is_an_error(self):
        """The silent one: an owned player filtered out of the pool grants a
        transfer the change constraint never counts."""
        plan = dict(self._good(), max_changes=1)
        assert _errors(check_transfer_plan(plan))

    def test_the_hit_count_must_match_the_legs(self):
        plan = dict(self._good(), free_transfers=1)  # 2 legs, 1 free, 0 hits claimed
        assert _errors(check_transfer_plan(plan))

    def test_selling_the_same_player_twice_is_an_error(self):
        plan = self._good()
        plan["legs"][1]["out_id"] = plan["legs"][0]["out_id"]
        assert _errors(check_transfer_plan(plan))

    def test_buying_the_same_player_twice_is_an_error(self):
        plan = self._good()
        plan["legs"][1]["in_id"] = plan["legs"][0]["in_id"]
        assert _errors(check_transfer_plan(plan))

    def test_selling_and_buying_one_player_is_an_error(self):
        plan = self._good()
        plan["legs"][1]["in_id"] = plan["legs"][0]["out_id"]
        assert _errors(check_transfer_plan(plan))

    def test_a_negative_bank_is_an_error(self):
        """Both incoming players come out of one pot."""
        plan = dict(self._good(), bank_after=-0.3)
        assert _errors(check_transfer_plan(plan))

    def test_bank_arithmetic_must_agree_with_the_legs(self):
        plan = dict(self._good(), bank_after=5.0)
        assert _errors(check_transfer_plan(plan))

    def test_a_sequence_that_dips_negative_warns(self):
        """Affordable as a set, not in that order -- the manager can reorder."""
        plan = self._good()
        plan["legs"].reverse()  # buy Salah before selling Haaland
        issues = check_transfer_plan(plan)
        assert _warnings(issues) and not _errors(issues)

    def test_a_plan_that_loses_points_is_an_error(self):
        """Why propose it? The percentile objective could rank a plan well
        while it lost points, and nothing downstream could tell."""
        plan = dict(self._good(), gain_net=-0.5)
        assert _errors(check_transfer_plan(plan))

    def test_an_implausibly_large_gain_warns(self):
        """The signature of a 3-gameweek total used as a per-gameweek rate."""
        plan = dict(self._good(), gain_net=60.0)
        assert _warnings(check_transfer_plan(plan))

    def test_a_horizon_outside_one_to_three_is_an_error(self):
        assert _errors(check_transfer_plan(dict(self._good(), horizon_gws=4.5)))
        assert _errors(check_transfer_plan(dict(self._good(), horizon_gws=0.5)))

    def test_prices_in_tenths_are_an_error(self):
        plan = self._good()
        plan["legs"][0]["in_price"] = 75.0
        assert _errors(check_transfer_plan(plan))

    def test_a_non_finite_delta_is_an_error(self):
        plan = self._good()
        plan["legs"][0]["delta"] = float("nan")
        assert _errors(check_transfer_plan(plan))

    def test_a_leg_swapping_across_positions_is_an_error(self):
        """FPL's squad is fixed at 2/5/5/3, so a forward cannot become a
        midfielder. Reading one `position` field for both sides made this
        check compare a list with itself."""
        plan = self._good()
        plan["legs"][0]["out_position"] = "F"
        plan["legs"][0]["in_position"] = "M"
        plan["legs"][1]["out_position"] = "M"
        plan["legs"][1]["in_position"] = "M"
        assert _errors(check_transfer_plan(plan))

    def test_legs_that_stay_within_position_pass(self):
        plan = self._good()
        for leg in plan["legs"]:
            leg["out_position"] = leg["in_position"] = leg["position"]
        assert check_transfer_plan(plan) == []

    def test_the_resulting_squad_must_be_legal(self):
        positions = ["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3
        squad = pd.DataFrame({
            "Player_ID": list(range(15)),
            "Position": positions,
            "Team": ["T1"] * 5 + ["T%d" % i for i in range(10)],  # 5 from one club
        })
        assert _errors(check_transfer_plan(self._good(), squad_after=squad))

    def test_a_legal_resulting_squad_passes(self):
        positions = ["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3
        squad = pd.DataFrame({
            "Player_ID": list(range(15)),
            "Position": positions,
            "Team": ["T%d" % i for i in range(15)],
        })
        assert check_transfer_plan(self._good(), squad_after=squad) == []

    def test_a_duplicated_pick_in_the_result_is_an_error(self):
        positions = ["G"] * 2 + ["D"] * 5 + ["M"] * 5 + ["F"] * 3
        squad = pd.DataFrame({
            "Player_ID": [0] + list(range(14)),
            "Position": positions,
            "Team": ["T%d" % i for i in range(15)],
        })
        assert _errors(check_transfer_plan(self._good(), squad_after=squad))
