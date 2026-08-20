"""Unit tests for scripts/common/team_strength.py.

All tests are offline and use hand-built frames.  The reference pool and the roster
share a schema, matching what ``_build_reference_pool`` produces.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common.team_strength import (
    _MIN_STARTS_FOR_ACTUALS,
    _POS_LABELS,
    _STRENGTH_WEIGHTS,
    aggregate_team_strength,
    compute_player_strength,
    describe_blend,
)

_SCHEMA_DEFAULTS = {
    "SeasonProjection": 100.0,
    "total_points": 50.0,
    "starts": 10.0,
    "minutes": 900.0,
    "form": 4.0,
    "Proj_Blended": 5.0,
    "Start_Security_Raw": 80.0,
    "Fixture_Ease_Raw": -3.0,
    "status": "a",
    "news": "",
    "chance_of_playing_next_round": np.nan,
}


def _players(*specs, position="M"):
    """Build a frame from partial dicts, filling the rest of the schema."""
    rows = []
    for i, spec in enumerate(specs):
        row = dict(_SCHEMA_DEFAULTS)
        row["Position"] = position
        row["Player"] = f"P{i}"
        row["Player_ID"] = 1000 + i
        row.update(spec)
        rows.append(row)
    return pd.DataFrame(rows)


def _pool(n=40, position="M"):
    """A reference pool spanning a wide range of every value."""
    return pd.DataFrame({
        "Player": [f"Ref{i}" for i in range(n)],
        "Player_ID": list(range(n)),
        "Position": [position] * n,
        "SeasonProjection": np.linspace(20, 200, n),
        "total_points": np.linspace(0, 200, n),
        "starts": np.linspace(0, 30, n),
        "minutes": np.linspace(0, 2700, n),
        "form": np.linspace(0, 9, n),
        "Proj_Blended": np.linspace(0, 10, n),
        "Start_Security_Raw": np.linspace(0, 100, n),
        "Fixture_Ease_Raw": np.linspace(-5, -1, n),
        "PPS": np.linspace(0, 10, n),
        "status": ["a"] * n,
        "news": [""] * n,
        "chance_of_playing_next_round": [np.nan] * n,
    })


class TestPointsPerStart:
    """Points per START, so games missed through injury don't dilute quality."""

    def test_returning_star_outrates_durable_journeyman(self):
        """The case that motivated pts/start over total points.

        A returning star has fewer total points than a always-available squad player,
        but is plainly the better asset.  Total points would rank them backwards.
        """
        roster = _players(
            {"Player": "Star", "total_points": 72, "starts": 8, "form": 7.5},
            {"Player": "Journeyman", "total_points": 90, "starts": 25, "form": 3.0},
        )
        scored = compute_player_strength(roster, _pool(), current_gw=26)

        assert scored.loc[0, "PPS"] == pytest.approx(9.0)
        assert scored.loc[1, "PPS"] == pytest.approx(3.6)
        assert scored.loc[0, "Actuals"] > scored.loc[1, "Actuals"], (
            "the returning star must rate above the journeyman despite fewer total points"
        )

    def test_zero_starts_yields_zero_not_infinity(self):
        roster = _players({"total_points": 10, "starts": 0})
        scored = compute_player_strength(roster, _pool(), current_gw=20)
        assert scored.loc[0, "PPS"] == 0.0
        assert np.isfinite(scored.loc[0, "Player_Strength"])


class TestMinimumStartsFloor:
    """Below the start threshold the sample is too thin — use pedigree alone."""

    def test_thin_sample_falls_back_to_pedigree_exactly(self):
        roster = _players({"total_points": 36, "starts": _MIN_STARTS_FOR_ACTUALS - 1})
        scored = compute_player_strength(roster, _pool(), current_gw=20)
        assert scored.loc[0, "Actuals"] == pytest.approx(scored.loc[0, "Pedigree"])

    def test_at_threshold_actuals_are_used(self):
        """One start either side of the threshold must behave differently."""
        thin = _players({"total_points": 40, "starts": _MIN_STARTS_FOR_ACTUALS - 1, "form": 9.0})
        thick = _players({"total_points": 40, "starts": _MIN_STARTS_FOR_ACTUALS, "form": 9.0})
        pool = _pool()
        a = compute_player_strength(thin, pool, current_gw=20).loc[0, "Actuals"]
        b = compute_player_strength(thick, pool, current_gw=20).loc[0, "Actuals"]
        assert a != pytest.approx(b)


class TestSeasonProgressBlend:
    """Early season leans on pedigree; late season leans on actuals."""

    def test_blend_shifts_toward_actuals(self):
        # A player whose actuals and pedigree disagree sharply.
        roster = _players({"SeasonProjection": 30.0, "total_points": 190, "starts": 20, "form": 9.0})
        pool = _pool()
        early = compute_player_strength(roster, pool, current_gw=2).loc[0]
        late = compute_player_strength(roster, pool, current_gw=36).loc[0]

        assert early["Pedigree"] == pytest.approx(late["Pedigree"])
        assert late["Quality"] > early["Quality"], (
            "a high-actuals / low-pedigree player should rate higher late in the season"
        )

    def test_quality_is_bounded_by_its_two_inputs(self):
        roster = _players({"SeasonProjection": 190.0, "total_points": 5, "starts": 10, "form": 0.2})
        s = compute_player_strength(roster, _pool(), current_gw=19).loc[0]
        lo, hi = sorted([s["Pedigree"], s["Actuals"]])
        assert lo - 1e-9 <= s["Quality"] <= hi + 1e-9


class TestInjuryDiscount:
    def test_injured_player_is_discounted(self):
        roster = _players(
            {"Player": "Fit", "status": "a", "news": ""},
            {"Player": "Hurt", "status": "i", "news": "Hamstring injury - Expected back 20 May"},
        )
        scored = compute_player_strength(roster, _pool(), current_gw=20)
        assert scored.loc[0, "Injury_Mult"] == 1.0
        assert scored.loc[1, "Injury_Mult"] < 1.0
        assert scored.loc[1, "GWs_Missed"] > 0
        assert scored.loc[1, "Player_Strength"] < scored.loc[1, "Raw_Strength"]

    def test_healthy_player_strength_equals_raw(self):
        roster = _players({"status": "a", "news": ""})
        s = compute_player_strength(roster, _pool(), current_gw=15).loc[0]
        assert s["Player_Strength"] == pytest.approx(s["Raw_Strength"])


class TestRawStrengthComposition:
    def test_weights_sum_to_one(self):
        assert sum(_STRENGTH_WEIGHTS.values()) == pytest.approx(1.0)

    def test_raw_strength_matches_weighted_sum(self):
        roster = _players({})
        s = compute_player_strength(roster, _pool(), current_gw=20).loc[0]
        expected = (
            _STRENGTH_WEIGHTS["quality"] * s["Quality"]
            + _STRENGTH_WEIGHTS["gw_proj"] * s["GW_Proj_Pctile"]
            + _STRENGTH_WEIGHTS["start_security"] * s["Start_Security"]
            + _STRENGTH_WEIGHTS["fixture_ease"] * s["Fixture_Ease"]
        )
        assert s["Raw_Strength"] == pytest.approx(expected)

    def test_all_outputs_are_within_unit_range(self):
        roster = _players({}, {"total_points": 0, "starts": 0, "form": 0}, {"total_points": 300, "starts": 30})
        scored = compute_player_strength(roster, _pool(), current_gw=20)
        for col in ("Pedigree", "Actuals", "Quality", "Raw_Strength", "Player_Strength"):
            assert scored[col].between(0.0, 1.0).all(), f"{col} escaped [0,1]"

    def test_empty_roster_returns_empty_with_schema(self):
        out = compute_player_strength(pd.DataFrame(), _pool(), current_gw=10)
        assert out.empty
        assert "Player_Strength" in out.columns


class TestPedigreeFallback:
    def test_missing_season_projections_falls_back_to_points(self):
        """Rotowire being down must degrade, not flatten every score to 0.5."""
        pool = _pool()
        pool["SeasonProjection"] = np.nan
        roster = _players(
            {"SeasonProjection": np.nan, "total_points": 5},
            {"SeasonProjection": np.nan, "total_points": 190},
        )
        scored = compute_player_strength(roster, pool, current_gw=20)
        assert scored.loc[1, "Pedigree"] > scored.loc[0, "Pedigree"]
        assert scored["Pedigree"].nunique() > 1, "scores must not collapse to a constant"


def _team_frame():
    """Two teams, full 2/5/5/3 Draft squads, team A stronger than team B."""
    rows = []
    for team_id, name, base in [(1, "Alpha", 0.80), (2, "Bravo", 0.40)]:
        for pos, count in [("G", 2), ("D", 5), ("M", 5), ("F", 3)]:
            for i in range(count):
                rows.append({
                    "Team_ID": team_id, "Team_Name": name, "Position": pos,
                    "Player": f"{name}-{pos}{i}",
                    "Raw_Strength": base, "Player_Strength": base,
                })
    return pd.DataFrame(rows)


class TestAggregateTeamStrength:
    def test_scores_are_flat_mean_of_fifteen(self):
        team_df = aggregate_team_strength(_team_frame())
        assert list(team_df["Players"]) == [15, 15]
        assert team_df.loc[0, "Score"] == pytest.approx(80.0)
        assert team_df.loc[1, "Score"] == pytest.approx(40.0)

    def test_positional_score_uses_only_that_position(self):
        players = _team_frame()
        # Make Alpha's forwards elite and everything else unchanged.
        mask = (players["Team_ID"] == 1) & (players["Position"] == "F")
        players.loc[mask, "Player_Strength"] = 0.95
        team_df = aggregate_team_strength(players).set_index("Team_Name")
        assert team_df.loc["Alpha", "FWD"] == pytest.approx(95.0)
        assert team_df.loc["Alpha", "DEF"] == pytest.approx(80.0), "DEF must be unaffected"

    def test_ranks_are_assigned_best_first(self):
        team_df = aggregate_team_strength(_team_frame())
        assert team_df.loc[0, "Rank"] == 1
        assert team_df.loc[1, "Rank"] == 2
        for label in _POS_LABELS.values():
            assert set(team_df[f"{label}_Rank"]) == {1, 2}

    def test_injury_cost_is_gap_between_healthy_and_actual(self):
        players = _team_frame()
        players.loc[0, "Player_Strength"] = 0.20  # one injured Alpha keeper
        team_df = aggregate_team_strength(players).set_index("Team_Name")
        assert team_df.loc["Alpha", "Injury_Cost"] == pytest.approx(
            team_df.loc["Alpha", "Healthy_Score"] - team_df.loc["Alpha", "Score"]
        )
        assert team_df.loc["Alpha", "Injury_Cost"] > 0
        assert team_df.loc["Bravo", "Injury_Cost"] == pytest.approx(0.0)

    def test_sorted_by_score_descending(self):
        team_df = aggregate_team_strength(_team_frame())
        assert team_df["Score"].is_monotonic_decreasing

    def test_empty_input_returns_empty_with_schema(self):
        out = aggregate_team_strength(pd.DataFrame())
        assert out.empty
        for col in ("Score", "GK", "DEF", "MID", "FWD", "Injury_Cost", "Rank"):
            assert col in out.columns


class TestDescribeBlend:
    def test_states_both_halves_of_the_blend(self):
        assert "GW12" in describe_blend(12)
        assert "actuals" in describe_blend(12)
        assert "pedigree" in describe_blend(12)

    def test_shifts_toward_actuals_over_the_season(self):
        assert describe_blend(2) != describe_blend(36)


class TestReferencePoolPPS:
    """The reference pool must carry PPS, and must exclude tiny samples from it."""

    def test_thin_sample_players_are_excluded_from_the_reference(self):
        from scripts.common.team_strength import attach_reference_pps

        pool = pd.DataFrame({
            "Player": ["Fluke", "Regular", "Star"],
            "Position": ["F", "F", "F"],
            "total_points": [35, 40, 120],
            "starts": [2, 10, 20],
        })
        out = attach_reference_pps(pool)
        assert np.isnan(out.loc[0, "PPS"]), "a 2-start player must not set the scale"
        assert out.loc[1, "PPS"] == pytest.approx(4.0)
        assert out.loc[2, "PPS"] == pytest.approx(6.0)

    def test_star_is_not_deflated_by_small_sample_flukes(self):
        """Regression: 11 tiny-sample forwards pushed Haaland to the 83rd percentile.

        With flukes excluded from the denominator he lands near the top, where a
        7-points-per-start forward belongs.
        """
        from scripts.common.team_strength import attach_reference_pps

        # 30 forwards: 15 genuine (2-6 pts/start), 15 two-start flukes at 15+/start.
        pool = pd.DataFrame({
            "Player": [f"F{i}" for i in range(30)],
            "Position": ["F"] * 30,
            "total_points": list(np.linspace(20, 60, 15)) + [32] * 15,
            "starts": [10] * 15 + [2] * 15,
            "minutes": [900] * 30,
            "form": np.linspace(1, 9, 30),
            "SeasonProjection": np.linspace(50, 150, 30),
            "Proj_Blended": np.linspace(1, 8, 30),
            "Start_Security_Raw": np.linspace(10, 95, 30),
            "Fixture_Ease_Raw": [-3.0] * 30,
            "status": ["a"] * 30, "news": [""] * 30,
            "chance_of_playing_next_round": [np.nan] * 30,
        })
        ref = attach_reference_pps(pool)

        star = _players({"total_points": 140, "starts": 20, "form": 8.0}, position="F")
        scored = compute_player_strength(star, ref, current_gw=25)
        assert scored.loc[0, "PPS"] == pytest.approx(7.0)
        assert scored.loc[0, "Actuals"] > 0.85, (
            "a 7-points-per-start forward must rate near the top of his position"
        )


class TestFlatFormFallback:
    """Preseason form is 0.0 for everyone; blending it in would deflate all actuals."""

    def test_flat_form_falls_back_to_points_per_start_alone(self):
        from scripts.common.team_strength import attach_reference_pps

        pool = attach_reference_pps(_pool())
        pool["form"] = 0.0
        roster = _players({"total_points": 100, "starts": 10, "form": 0.0})
        scored = compute_player_strength(roster, pool, current_gw=25)

        from scripts.common.analytics import positional_percentile
        expected = positional_percentile(scored, pool, "PPS", min_minutes=90).iloc[0]
        assert scored.loc[0, "Actuals"] == pytest.approx(expected), (
            "with no form signal, actuals should be points-per-start alone"
        )

    def test_varying_form_is_blended_in(self):
        from scripts.common.team_strength import attach_reference_pps

        pool = attach_reference_pps(_pool())
        hot = _players({"total_points": 100, "starts": 10, "form": 9.0})
        cold = _players({"total_points": 100, "starts": 10, "form": 0.1})
        a = compute_player_strength(hot, pool, current_gw=25).loc[0, "Actuals"]
        b = compute_player_strength(cold, pool, current_gw=25).loc[0, "Actuals"]
        assert a > b, "identical points-per-start but hotter form must rate higher"


class TestHasSignal:
    def test_flat_column_has_no_signal(self):
        from scripts.common.team_strength import _has_signal
        assert not _has_signal(pd.DataFrame({"x": [0.0, 0.0, 0.0]}), "x")

    def test_varying_column_has_signal(self):
        from scripts.common.team_strength import _has_signal
        assert _has_signal(pd.DataFrame({"x": [0.0, 1.0]}), "x")

    def test_missing_column_has_no_signal(self):
        from scripts.common.team_strength import _has_signal
        assert not _has_signal(pd.DataFrame({"y": [1.0]}), "x")
        assert not _has_signal(None, "x")
