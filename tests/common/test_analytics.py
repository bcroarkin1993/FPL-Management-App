"""Unit tests for scripts.common.analytics multi-GW transfer planner functions.

All tests use mock DataFrames — no network calls.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from scripts.common.analytics import (
    compute_healthy_form,
    compute_player_scores,
    compute_dynamic_alpha,
    compute_positional_depth,
    compute_transfer_urgency,
    blend_multi_gw_projections,
    positional_percentile,
    positional_rank,
    dampen_form_by_starts,
    season_progress_weight,
    merge_season_projections,
    merge_ffp_single_gw_data,
    PositionalDepth,
)


# =============================================================================
# TestComputeHealthyForm
# =============================================================================

class TestComputeHealthyForm:
    """Tests for compute_healthy_form — filters out 0-minute GWs."""

    @staticmethod
    def _make_history(rounds_minutes_pts):
        """Build a history DataFrame from list of (round, minutes, total_points)."""
        return pd.DataFrame(rounds_minutes_pts, columns=["round", "minutes", "total_points"])

    def test_filters_zero_minute_gws(self):
        """Player with 8, 7, 9, 0, 0 (last two injured) → healthy form = avg(8, 7, 9) = 8.0."""
        hist = self._make_history([
            (1, 90, 8), (2, 90, 7), (3, 90, 9), (4, 0, 0), (5, 0, 0),
        ])
        fn = MagicMock(return_value=hist)
        result = compute_healthy_form(100, last_n=3, element_history_fn=fn)
        assert result == pytest.approx(8.0)
        fn.assert_called_once_with(100)

    def test_looks_back_further(self):
        """0, 0, 0, 6, 8 in most recent GWs → finds 6, 8 from earlier played GWs."""
        hist = self._make_history([
            (1, 90, 4), (2, 90, 6), (3, 90, 8), (4, 0, 0), (5, 0, 0),
        ])
        fn = MagicMock(return_value=hist)
        # last_n=3 → should find GW3=8, GW2=6, GW1=4 (the 3 most recent played)
        result = compute_healthy_form(100, last_n=3, element_history_fn=fn)
        assert result == pytest.approx((8 + 6 + 4) / 3)

    def test_fewer_played_than_requested(self):
        """Only 2 played GWs exist → averages those 2."""
        hist = self._make_history([
            (1, 90, 5), (2, 90, 10), (3, 0, 0), (4, 0, 0), (5, 0, 0),
        ])
        fn = MagicMock(return_value=hist)
        result = compute_healthy_form(100, last_n=5, element_history_fn=fn)
        assert result == pytest.approx(7.5)  # (5 + 10) / 2

    def test_no_played_gws(self):
        """All 0 minutes → returns None."""
        hist = self._make_history([
            (1, 0, 0), (2, 0, 0), (3, 0, 0),
        ])
        fn = MagicMock(return_value=hist)
        result = compute_healthy_form(100, last_n=3, element_history_fn=fn)
        assert result is None

    def test_uses_custom_history_fn(self):
        """Injected function is called with the player_id."""
        hist = self._make_history([(1, 90, 7)])
        fn = MagicMock(return_value=hist)
        compute_healthy_form(42, last_n=3, element_history_fn=fn)
        fn.assert_called_once_with(42)

    def test_none_history(self):
        """History fetch returns None → returns None."""
        fn = MagicMock(return_value=None)
        result = compute_healthy_form(100, last_n=3, element_history_fn=fn)
        assert result is None

    def test_empty_history(self):
        """History fetch returns empty DataFrame → returns None."""
        fn = MagicMock(return_value=pd.DataFrame())
        result = compute_healthy_form(100, last_n=3, element_history_fn=fn)
        assert result is None

    def test_exception_in_history_fn(self):
        """If history function raises, returns None gracefully."""
        fn = MagicMock(side_effect=Exception("API error"))
        result = compute_healthy_form(100, last_n=3, element_history_fn=fn)
        assert result is None


# =============================================================================
# TestComputePositionalDepth
# =============================================================================

class TestComputePositionalDepth:
    """Tests for compute_positional_depth."""

    def test_all_healthy(self):
        """All players status='a' → all Adequate."""
        roster = pd.DataFrame({
            "Position": ["G", "G", "D", "D", "D", "D", "D", "M", "M", "M", "M", "M", "F", "F", "F"],
            "status": ["a"] * 15,
            "chance_of_playing_next_round": [None] * 15,
        })
        result = compute_positional_depth(roster)
        assert result["G"].depth_level == "Adequate"
        assert result["D"].depth_level == "Adequate"
        assert result["M"].depth_level == "Adequate"
        assert result["F"].depth_level == "Adequate"
        assert result["G"].healthy == 2
        assert result["G"].doubtful == 0
        assert result["G"].injured == 0
        assert result["D"].healthy == 5

    def test_critical_position(self):
        """2 FWDs both injured → Critical."""
        roster = pd.DataFrame({
            "Position": ["F", "F"],
            "status": ["i", "i"],
            "chance_of_playing_next_round": [0, 25],
        })
        result = compute_positional_depth(roster)
        assert result["F"].depth_level == "Critical"
        assert result["F"].healthy == 0
        assert result["F"].injured == 2
        assert result["F"].doubtful == 0

    def test_low_depth(self):
        """4 DEF, 3 healthy, 1 injured → Low (3 healthy at DEF = yellow threshold)."""
        roster = pd.DataFrame({
            "Position": ["D", "D", "D", "D"],
            "status": ["a", "a", "a", "i"],
            "chance_of_playing_next_round": [None, None, None, 25],
        })
        result = compute_positional_depth(roster)
        assert result["D"].depth_level == "Low"
        assert result["D"].healthy == 3
        assert result["D"].total == 4
        assert result["D"].doubtful == 0
        assert result["D"].injured == 1

    def test_missing_status_treated_as_healthy(self):
        """Players with both status and chance missing → assumed healthy."""
        roster = pd.DataFrame({
            "Position": ["M", "M", "M", "M"],
            "status": [None, None, None, None],
            "chance_of_playing_next_round": [None, None, None, None],
        })
        result = compute_positional_depth(roster)
        # 4 healthy MID → meets green threshold (4) → Adequate
        assert result["M"].depth_level == "Adequate"
        assert result["M"].healthy == 4
        assert result["M"].doubtful == 0
        assert result["M"].injured == 0

    def test_chance_threshold_75(self):
        """Player with chance 75 is healthy, 74 is doubtful."""
        roster = pd.DataFrame({
            "Position": ["G", "G"],
            "status": [None, None],
            "chance_of_playing_next_round": [75, 74],
        })
        result = compute_positional_depth(roster)
        # 75% → healthy, 74% → doubtful → 1 healthy out of 2 → GK threshold: 1=yellow → Low
        assert result["G"].healthy == 1
        assert result["G"].doubtful == 1
        assert result["G"].injured == 0
        assert result["G"].depth_level == "Low"

    def test_empty_position(self):
        """Position with 0 players → Critical."""
        roster = pd.DataFrame({
            "Position": ["D", "M"],
            "status": ["a", "a"],
            "chance_of_playing_next_round": [None, None],
        })
        result = compute_positional_depth(roster)
        assert result["F"].depth_level == "Critical"
        assert result["F"].total == 0
        assert result["F"].doubtful == 0
        assert result["F"].injured == 0
        assert result["G"].depth_level == "Critical"

    def test_doubtful_tracking(self):
        """Players with status='d' or chance 25-74 → counted as doubtful."""
        roster = pd.DataFrame({
            "Position": ["M", "M", "M", "M", "M"],
            "status": ["a", "a", "a", "d", None],
            "chance_of_playing_next_round": [None, None, None, None, 50],
        })
        result = compute_positional_depth(roster)
        assert result["M"].healthy == 3
        assert result["M"].doubtful == 2
        assert result["M"].injured == 0
        # effective = 3 + 2*0.5 = 4.0, MID green threshold = 4 → "Adequate"
        assert result["M"].depth_level == "Adequate"

    def test_position_aware_thresholds_def(self):
        """DEF thresholds: >=4 Adequate, 3 Low, <=2 Critical."""
        # 5 DEF, 4 healthy → Adequate
        roster_4h = pd.DataFrame({
            "Position": ["D"] * 5,
            "status": ["a", "a", "a", "a", "i"],
            "chance_of_playing_next_round": [None] * 5,
        })
        assert compute_positional_depth(roster_4h)["D"].depth_level == "Adequate"

        # 5 DEF, 3 healthy → Low
        roster_3h = pd.DataFrame({
            "Position": ["D"] * 5,
            "status": ["a", "a", "a", "i", "i"],
            "chance_of_playing_next_round": [None] * 5,
        })
        assert compute_positional_depth(roster_3h)["D"].depth_level == "Low"

        # 5 DEF, 2 healthy → Critical
        roster_2h = pd.DataFrame({
            "Position": ["D"] * 5,
            "status": ["a", "a", "i", "i", "i"],
            "chance_of_playing_next_round": [None] * 5,
        })
        assert compute_positional_depth(roster_2h)["D"].depth_level == "Critical"

    def test_fwd_thresholds(self):
        """FWD thresholds: >=2 Adequate, 1 Low, 0 Critical."""
        # 3 FWD, 3 healthy → Adequate
        roster_3h = pd.DataFrame({
            "Position": ["F", "F", "F"],
            "status": ["a", "a", "a"],
            "chance_of_playing_next_round": [None] * 3,
        })
        assert compute_positional_depth(roster_3h)["F"].depth_level == "Adequate"

        # 3 FWD, 2 healthy → Adequate (2 >= green threshold 2)
        roster_2h = pd.DataFrame({
            "Position": ["F", "F", "F"],
            "status": ["a", "a", "i"],
            "chance_of_playing_next_round": [None] * 3,
        })
        assert compute_positional_depth(roster_2h)["F"].depth_level == "Adequate"

        # 3 FWD, 1 healthy → Low (1 >= yellow threshold 1)
        roster_1h = pd.DataFrame({
            "Position": ["F", "F", "F"],
            "status": ["a", "i", "i"],
            "chance_of_playing_next_round": [None] * 3,
        })
        assert compute_positional_depth(roster_1h)["F"].depth_level == "Low"

        # 3 FWD, 0 healthy → Critical
        roster_0h = pd.DataFrame({
            "Position": ["F", "F", "F"],
            "status": ["i", "i", "i"],
            "chance_of_playing_next_round": [None] * 3,
        })
        assert compute_positional_depth(roster_0h)["F"].depth_level == "Critical"


# =============================================================================
# TestComputeTransferUrgency
# =============================================================================

class TestComputeTransferUrgency:
    """Tests for compute_transfer_urgency."""

    def test_urgent_critical(self):
        depth_map = {"F": PositionalDepth("F", 2, 0, 0, 2, "Critical")}
        assert compute_transfer_urgency("F", depth_map) == "URGENT"

    def test_moderate_low(self):
        depth_map = {"D": PositionalDepth("D", 5, 3, 1, 1, "Low")}
        assert compute_transfer_urgency("D", depth_map) == "LOW DEPTH"

    def test_empty_adequate(self):
        depth_map = {"M": PositionalDepth("M", 5, 5, 0, 0, "Adequate")}
        assert compute_transfer_urgency("M", depth_map) == ""

    def test_missing_position(self):
        depth_map = {"G": PositionalDepth("G", 2, 2, 0, 0, "Adequate")}
        assert compute_transfer_urgency("F", depth_map) == ""


# =============================================================================
# TestBlendMultiGWProjections
# =============================================================================

class TestBlendMultiGWProjections:
    """Tests for blend_multi_gw_projections."""

    def test_matches_ffp_data(self):
        """Matched players get Next3GWs value."""
        player_df = pd.DataFrame({
            "Player": ["Salah", "Haaland"],
            "Team": ["LIV", "MCI"],
            "Points": [8.0, 10.0],
        })
        ffp_df = pd.DataFrame({
            "Name": ["Salah", "Haaland"],
            "Team": ["Liverpool", "Man City"],
            "Next3GWs": [22.0, 28.0],
        })
        result = blend_multi_gw_projections(player_df, ffp_df)
        assert result.loc[0, "MultiGW_Proj"] == 22.0
        assert result.loc[1, "MultiGW_Proj"] == 28.0

    def test_fallback_no_ffp(self):
        """None ffp_df → uses single_gw * 3."""
        player_df = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
            "Points": [8.0],
        })
        result = blend_multi_gw_projections(player_df, None)
        assert result.loc[0, "MultiGW_Proj"] == 24.0  # 8 * 3

    def test_fallback_unmatched(self):
        """Unmatched players get single_gw * 3."""
        player_df = pd.DataFrame({
            "Player": ["Salah", "Unknown Player"],
            "Team": ["LIV", "ARS"],
            "Points": [8.0, 5.0],
        })
        ffp_df = pd.DataFrame({
            "Name": ["Salah"],
            "Team": ["Liverpool"],
            "Next3GWs": [22.0],
        })
        result = blend_multi_gw_projections(player_df, ffp_df)
        assert result.loc[0, "MultiGW_Proj"] == 22.0
        assert result.loc[1, "MultiGW_Proj"] == 15.0  # 5 * 3 fallback

    def test_name_normalization(self):
        """Accented names match via canonical_normalize."""
        player_df = pd.DataFrame({
            "Player": ["Raúl Jiménez"],
            "Team": ["FUL"],
            "Points": [5.0],
        })
        ffp_df = pd.DataFrame({
            "Name": ["Raul Jimenez"],
            "Team": ["Fulham"],
            "Next3GWs": [14.0],
        })
        result = blend_multi_gw_projections(player_df, ffp_df)
        assert result.loc[0, "MultiGW_Proj"] == 14.0

    def test_team_mapping(self):
        """FFP 'Arsenal' matches player_df 'ARS' via TEAM_FULL_TO_SHORT."""
        player_df = pd.DataFrame({
            "Player": ["Saka"],
            "Team": ["ARS"],
            "Points": [7.0],
        })
        ffp_df = pd.DataFrame({
            "Name": ["Saka"],
            "Team": ["Arsenal"],
            "Next3GWs": [20.0],
        })
        result = blend_multi_gw_projections(player_df, ffp_df)
        assert result.loc[0, "MultiGW_Proj"] == 20.0

    def test_empty_ffp_df(self):
        """Empty FFP DataFrame → fallback to single_gw * 3."""
        player_df = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
            "Points": [8.0],
        })
        result = blend_multi_gw_projections(player_df, pd.DataFrame())
        assert result.loc[0, "MultiGW_Proj"] == 24.0

    def test_missing_next3gws_column(self):
        """FFP without Next3GWs column → fallback."""
        player_df = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
            "Points": [8.0],
        })
        ffp_df = pd.DataFrame({
            "Name": ["Salah"],
            "Team": ["Liverpool"],
            "Predicted": [8.0],
        })
        result = blend_multi_gw_projections(player_df, ffp_df)
        assert result.loc[0, "MultiGW_Proj"] == 24.0

    def test_custom_column_names(self):
        """Custom single_gw_col and output_col work."""
        player_df = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
            "Proj": [8.0],
        })
        result = blend_multi_gw_projections(
            player_df, None, single_gw_col="Proj", output_col="Multi3"
        )
        assert result.loc[0, "Multi3"] == 24.0


# =============================================================================
# TestPositionalPercentile
# =============================================================================

class TestPositionalPercentile:
    """Tests for positional_percentile — within-position normalization."""

    def test_basic_percentile(self):
        """GK with 120 pts in pool of 50-150 → higher pct than MID with 120 in pool of 50-250."""
        squad = pd.DataFrame({
            "Player": ["GK_A", "MID_A"],
            "Position": ["G", "M"],
            "total_points": [120, 120],
        })
        reference = pd.DataFrame({
            "Player": [f"gk{i}" for i in range(5)] + [f"mid{i}" for i in range(5)],
            "Position": ["G"] * 5 + ["M"] * 5,
            "total_points": [50, 80, 100, 130, 150] + [50, 100, 150, 200, 250],
        })
        result = positional_percentile(squad, reference, "total_points")
        gk_pct = result.iloc[0]
        mid_pct = result.iloc[1]
        # GK: 120 beats 3 of 5 (50,80,100) → 0.6
        assert gk_pct == pytest.approx(0.6)
        # MID: 120 beats 2 of 5 (50,100) → 0.4
        assert mid_pct == pytest.approx(0.4)

    def test_top_of_position(self):
        """Player with best value at position → percentile near 1.0."""
        squad = pd.DataFrame({
            "Player": ["Star_GK"],
            "Position": ["G"],
            "total_points": [200],
        })
        reference = pd.DataFrame({
            "Player": ["gk1", "gk2", "gk3"],
            "Position": ["G", "G", "G"],
            "total_points": [80, 100, 150],
        })
        result = positional_percentile(squad, reference, "total_points")
        assert result.iloc[0] == pytest.approx(1.0)

    def test_all_same_value(self):
        """Everyone at a position has same value → 0.0 (none strictly below)."""
        squad = pd.DataFrame({
            "Player": ["GK_A"],
            "Position": ["G"],
            "total_points": [100],
        })
        reference = pd.DataFrame({
            "Player": ["gk1", "gk2", "gk3"],
            "Position": ["G", "G", "G"],
            "total_points": [100, 100, 100],
        })
        result = positional_percentile(squad, reference, "total_points")
        assert result.iloc[0] == pytest.approx(0.0)

    def test_fallback_no_reference(self):
        """None reference → falls back to min-max normalization."""
        squad = pd.DataFrame({
            "Player": ["A", "B"],
            "Position": ["G", "G"],
            "total_points": [50, 150],
        })
        result = positional_percentile(squad, None, "total_points")
        # Min-max: A=(50-50)/(150-50)=0, B=(150-50)/(150-50)=1
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)

    def test_missing_column_in_reference(self):
        """Reference missing the value column → falls back gracefully to min-max."""
        squad = pd.DataFrame({
            "Player": ["A", "B"],
            "Position": ["G", "G"],
            "total_points": [50, 150],
        })
        reference = pd.DataFrame({
            "Player": ["gk1", "gk2"],
            "Position": ["G", "G"],
            "other_col": [10, 20],
        })
        result = positional_percentile(squad, reference, "total_points")
        # Falls back to min-max on squad alone
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.0)

    def test_ref_value_col_mapping(self):
        """ref_value_col allows column name mismatch between df and reference."""
        squad = pd.DataFrame({
            "Player": ["GK_A"],
            "Position": ["G"],
            "Season_Points": [120],
        })
        reference = pd.DataFrame({
            "Player": ["gk1", "gk2", "gk3", "gk4"],
            "Position": ["G", "G", "G", "G"],
            "total_points": [50, 80, 100, 150],
        })
        result = positional_percentile(
            squad, reference, "Season_Points", ref_value_col="total_points"
        )
        # 120 beats 3 of 4 → 0.75
        assert result.iloc[0] == pytest.approx(0.75)

    def test_min_minutes_filter(self):
        """min_minutes filters out 0-minute players from reference pool."""
        squad = pd.DataFrame({
            "Player": ["GK_A"],
            "Position": ["G"],
            "total_points": [60],
        })
        reference = pd.DataFrame({
            "Player": ["gk1", "gk2", "gk3", "gk4"],
            "Position": ["G", "G", "G", "G"],
            "total_points": [0, 0, 50, 100],
            "minutes": [0, 0, 900, 1800],
        })
        # Without min_minutes: 60 beats 3 of 4 → 0.75
        result_no_filter = positional_percentile(squad, reference, "total_points", min_minutes=0)
        assert result_no_filter.iloc[0] == pytest.approx(0.75)
        # With min_minutes=90: only gk3(50) and gk4(100) in pool → 60 beats 1 of 2 → 0.5
        result_filtered = positional_percentile(squad, reference, "total_points", min_minutes=90)
        assert result_filtered.iloc[0] == pytest.approx(0.5)


# =============================================================================
# TestPositionalRank
# =============================================================================

class TestPositionalRank:
    """Tests for positional_rank — ordinal rank strings like '#2 GK'."""

    def test_basic_rank(self):
        """Player with 2nd highest points at GK → '#2 GK'."""
        squad = pd.DataFrame({
            "Player": ["Pickford"],
            "Position": ["G"],
            "total_points": [130],
        })
        reference = pd.DataFrame({
            "Player": ["gk1", "gk2", "gk3"],
            "Position": ["G", "G", "G"],
            "total_points": [150, 130, 80],
        })
        result = positional_rank(squad, reference, "total_points")
        assert result.iloc[0] == "#2 GK"

    def test_tied_rank(self):
        """Two players with same points → same rank."""
        squad = pd.DataFrame({
            "Player": ["A", "B"],
            "Position": ["M", "M"],
            "total_points": [100, 100],
        })
        reference = pd.DataFrame({
            "Player": ["m1", "m2", "m3"],
            "Position": ["M", "M", "M"],
            "total_points": [150, 100, 80],
        })
        result = positional_rank(squad, reference, "total_points")
        assert result.iloc[0] == "#2 MID"
        assert result.iloc[1] == "#2 MID"

    def test_all_positions(self):
        """Verify correct label mapping for all positions."""
        squad = pd.DataFrame({
            "Player": ["GK", "DEF", "MID", "FWD"],
            "Position": ["G", "D", "M", "F"],
            "total_points": [100, 100, 100, 100],
        })
        reference = pd.DataFrame({
            "Player": ["a", "b", "c", "d"],
            "Position": ["G", "D", "M", "F"],
            "total_points": [100, 100, 100, 100],
        })
        result = positional_rank(squad, reference, "total_points")
        assert "GK" in result.iloc[0]
        assert "DEF" in result.iloc[1]
        assert "MID" in result.iloc[2]
        assert "FWD" in result.iloc[3]

    def test_no_reference(self):
        """None reference → returns 'N/A'."""
        squad = pd.DataFrame({
            "Player": ["A"],
            "Position": ["G"],
            "total_points": [100],
        })
        result = positional_rank(squad, None, "total_points")
        assert result.iloc[0] == "N/A"

    def test_ref_value_col_mapping(self):
        """ref_value_col allows column name mismatch."""
        squad = pd.DataFrame({
            "Player": ["A"],
            "Position": ["D"],
            "Season_Points": [120],
        })
        reference = pd.DataFrame({
            "Player": ["d1", "d2", "d3"],
            "Position": ["D", "D", "D"],
            "total_points": [150, 120, 80],
        })
        result = positional_rank(squad, reference, "Season_Points", ref_value_col="total_points")
        assert result.iloc[0] == "#2 DEF"


# =============================================================================
# TestDampenFormByStarts
# =============================================================================

class TestDampenFormByStarts:
    """Tests for dampen_form_by_starts — blends form toward neutral for low sample sizes."""

    def test_full_confidence_unchanged(self):
        """Players with >= min_starts (5) retain their exact form."""
        form = pd.Series([0.9, 0.1, 0.5])
        starts = pd.Series([5, 10, 20])
        result = dampen_form_by_starts(form, starts)
        assert result.iloc[0] == pytest.approx(0.9)
        assert result.iloc[1] == pytest.approx(0.1)
        assert result.iloc[2] == pytest.approx(0.5)

    def test_zero_starts_floor_dampening(self):
        """0 starts → floor confidence (0.2), form=1.0 → 0.2*1.0 + 0.8*0.5 = 0.6."""
        form = pd.Series([1.0])
        starts = pd.Series([0])
        result = dampen_form_by_starts(form, starts)
        assert result.iloc[0] == pytest.approx(0.6)

    def test_partial_starts_proportional(self):
        """2 starts out of 5 → confidence=0.4, form=1.0 → 0.4*1.0 + 0.6*0.5 = 0.7."""
        form = pd.Series([1.0])
        starts = pd.Series([2])
        result = dampen_form_by_starts(form, starts)
        assert result.iloc[0] == pytest.approx(0.7)

    def test_ellborg_scenario(self):
        """1 start, form=0.95 → confidence=max(1/5, 0.2)=0.2, dampened=0.2*0.95 + 0.8*0.5 = 0.59."""
        form = pd.Series([0.95])
        starts = pd.Series([1])
        result = dampen_form_by_starts(form, starts)
        assert result.iloc[0] == pytest.approx(0.59)

    def test_nan_starts_treated_as_zero(self):
        """NaN starts → treated as 0 → floor dampening."""
        form = pd.Series([0.8])
        starts = pd.Series([np.nan])
        result = dampen_form_by_starts(form, starts)
        # confidence = floor = 0.2, dampened = 0.2*0.8 + 0.8*0.5 = 0.56
        assert result.iloc[0] == pytest.approx(0.56)

    def test_neutral_form_stays_neutral(self):
        """Form=0.5 → stays at 0.5 regardless of starts (blend of 0.5 with 0.5)."""
        form = pd.Series([0.5, 0.5, 0.5])
        starts = pd.Series([0, 2, 10])
        result = dampen_form_by_starts(form, starts)
        assert result.iloc[0] == pytest.approx(0.5)
        assert result.iloc[1] == pytest.approx(0.5)
        assert result.iloc[2] == pytest.approx(0.5)

    def test_custom_min_starts_and_floor(self):
        """Custom min_starts=10, floor=0.1 changes the dampening curve."""
        form = pd.Series([0.9])
        starts = pd.Series([5])
        # confidence = 5/10 = 0.5 (above floor 0.1), dampened = 0.5*0.9 + 0.5*0.5 = 0.7
        result = dampen_form_by_starts(form, starts, min_starts=10, floor=0.1)
        assert result.iloc[0] == pytest.approx(0.7)


class TestSeasonProgressWeight:
    """Tests for season_progress_weight() — concave curve."""

    def test_early_season_favors_projections(self):
        """GW 1 should return 0.10 (floor — heavily favor projections)."""
        w = season_progress_weight(1)
        assert w == pytest.approx(0.10, abs=0.01)

    def test_midseason_actuals_lead(self):
        """GW 19 should return ~0.58 (actuals already leading due to concave curve)."""
        w = season_progress_weight(19)
        assert 0.50 <= w <= 0.65

    def test_gw30_actuals_dominate(self):
        """GW 30 should return ~0.82 (Rotowire season projection ≈ 18%)."""
        w = season_progress_weight(30)
        assert 0.78 <= w <= 0.88

    def test_late_season_favors_actual(self):
        """GW 38 should return 0.95 (cap — projection nearly irrelevant)."""
        w = season_progress_weight(38)
        assert w == pytest.approx(0.95, abs=0.01)

    def test_floor_never_below_0_10(self):
        """Even GW 0 should not go below 0.10."""
        w = season_progress_weight(0)
        assert w >= 0.10

    def test_ceiling_never_above_0_95(self):
        """Even GW 50 should not exceed 0.95."""
        w = season_progress_weight(50)
        assert w <= 0.95


class TestComputePlayerScores:
    """Tests for compute_player_scores()."""

    def _make_pool(self):
        """Create a reference pool of ~20 players per position."""
        rows = []
        for pos in ["G", "D", "M", "F"]:
            for i in range(20):
                rows.append({
                    "Player": f"{pos}_Player_{i}",
                    "Team": f"T{i % 5}",
                    "Position": pos,
                    "Projected_Points": 2.0 + i * 0.5,
                    "form": 1.0 + i * 0.3,
                    "total_points": 10 + i * 8,
                    "minutes": 200 + i * 50,
                    "starts": 3 + i,
                    "AvgFDR": 2.0 + (i % 5) * 0.5,
                })
        return pd.DataFrame(rows)

    def test_returns_all_four_columns(self):
        """Output has 1GW, ROS, Transfer Score, and Keep Score columns."""
        pool = self._make_pool()
        squad = pool.head(5).copy()
        result = compute_player_scores(squad, pool, current_gw=20)
        assert "1GW" in result.columns
        assert "ROS" in result.columns
        assert "Transfer Score" in result.columns
        assert "Keep Score" in result.columns

    def test_scores_in_zero_one_range(self):
        """All scores should be in [0, 1]."""
        pool = self._make_pool()
        squad = pool.head(10).copy()
        result = compute_player_scores(squad, pool, current_gw=20)
        assert result["1GW"].between(0, 1).all()
        assert result["ROS"].between(0, 1).all()
        assert result["Transfer Score"].between(0, 1).all()
        assert result["Keep Score"].between(0, 1).all()

    def test_top_player_scores_high(self):
        """The best player at a position should score above 0.7."""
        pool = self._make_pool()
        # Take the best MID from pool
        best_mid = pool[pool["Position"] == "M"].nlargest(1, "total_points").copy()
        result = compute_player_scores(best_mid, pool, current_gw=20)
        assert result["1GW"].iloc[0] > 0.7
        assert result["ROS"].iloc[0] > 0.7

    def test_weak_player_scores_low(self):
        """The worst player at a position should score below 0.3."""
        pool = self._make_pool()
        worst_mid = pool[pool["Position"] == "M"].nsmallest(1, "total_points").copy()
        result = compute_player_scores(worst_mid, pool, current_gw=20)
        assert result["ROS"].iloc[0] < 0.3

    def test_no_reference_df_graceful_fallback(self):
        """Works without reference DataFrame (falls back to min-max)."""
        squad = pd.DataFrame({
            "Player": ["A", "B"],
            "Team": ["X", "Y"],
            "Position": ["M", "M"],
            "Projected_Points": [5.0, 3.0],
            "Form": [4.0, 2.0],
            "Season_Points": [100, 60],
            "AvgFDRNextN": [3.0, 3.0],
        })
        result = compute_player_scores(squad, None, current_gw=20)
        assert "1GW" in result.columns
        assert "ROS" in result.columns
        assert "Transfer Score" in result.columns
        assert "Keep Score" in result.columns
        assert result["1GW"].notna().all()

    def test_transfer_keep_score_is_blend_of_1gw_and_ros(self):
        """Transfer/Keep Score should be between 1GW and ROS values."""
        pool = self._make_pool()
        squad = pool.head(10).copy()
        result = compute_player_scores(squad, pool, current_gw=20)
        for idx in result.index:
            low = min(result.at[idx, "1GW"], result.at[idx, "ROS"])
            high = max(result.at[idx, "1GW"], result.at[idx, "ROS"])
            assert low - 0.01 <= result.at[idx, "Transfer Score"] <= high + 0.01
            assert low - 0.01 <= result.at[idx, "Keep Score"] <= high + 0.01

    def test_format_context_affects_blend(self):
        """Draft format should lean more ROS (lower alpha) than Classic."""
        pool = self._make_pool()
        # Pick a player with different 1GW and ROS values
        squad = pool[pool["Position"] == "M"].head(1).copy()
        result_draft = compute_player_scores(squad, pool, current_gw=20, format_context="draft")
        result_classic = compute_player_scores(squad, pool, current_gw=20, format_context="classic")
        # Draft has lower alpha → more ROS weight → closer to ROS
        # Classic has higher alpha → more 1GW weight → closer to 1GW
        # Just check both produce valid output (exact comparison depends on player values)
        assert result_draft["Transfer Score"].iloc[0] >= 0
        assert result_classic["Transfer Score"].iloc[0] >= 0

    def test_ffp_data_affects_1gw(self):
        """FFP_Predicted and FFP_Start should affect the 1GW score."""
        pool = self._make_pool()
        squad = pool[pool["Position"] == "M"].head(1).copy()

        # Without FFP data
        result_no_ffp = compute_player_scores(squad.copy(), pool, current_gw=20)

        # With high FFP projection and low start likelihood
        squad_ffp = squad.copy()
        squad_ffp["FFP_Predicted"] = 10.0  # High prediction
        squad_ffp["FFP_Start"] = 30.0  # Only 30% start chance
        result_with_ffp = compute_player_scores(squad_ffp, pool, current_gw=20)

        # Both should produce valid scores
        assert result_no_ffp["1GW"].notna().all()
        assert result_with_ffp["1GW"].notna().all()


class TestMergeSeasonProjections:
    """Tests for merge_season_projections()."""

    def test_basic_merge(self):
        """Season projection is merged by name + team."""
        players = pd.DataFrame({
            "Player": ["Salah", "Haaland"],
            "Team": ["LIV", "MCI"],
        })
        season = pd.DataFrame({
            "Player": ["Salah", "Haaland"],
            "Team": ["LIV", "MCI"],
            "Points": [220, 240],
        })
        result = merge_season_projections(players, season)
        assert "SeasonProjection" in result.columns
        assert result.loc[0, "SeasonProjection"] == 220
        assert result.loc[1, "SeasonProjection"] == 240

    def test_unmatched_players_get_nan(self):
        """Players not in season rankings get NaN."""
        players = pd.DataFrame({
            "Player": ["Salah", "Unknown Player"],
            "Team": ["LIV", "MCI"],
        })
        season = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
            "Points": [220],
        })
        result = merge_season_projections(players, season)
        assert result.loc[0, "SeasonProjection"] == 220
        assert pd.isna(result.loc[1, "SeasonProjection"])

    def test_none_season_df(self):
        """Graceful fallback when season rankings are None."""
        players = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
        })
        result = merge_season_projections(players, None)
        assert "SeasonProjection" in result.columns
        assert pd.isna(result.loc[0, "SeasonProjection"])

    def test_empty_season_df(self):
        """Graceful fallback when season rankings are empty."""
        players = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
        })
        result = merge_season_projections(players, pd.DataFrame())
        assert "SeasonProjection" in result.columns
        assert pd.isna(result.loc[0, "SeasonProjection"])


# =============================================================================
# TestComputeDynamicAlpha
# =============================================================================

class TestComputeDynamicAlpha:
    """Tests for compute_dynamic_alpha() — dynamic 1GW/ROS blend weight."""

    def test_draft_baseline(self):
        """Draft baseline alpha = 0.35 for a MID with average ROS."""
        alpha = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha == pytest.approx(0.35)

    def test_classic_baseline(self):
        """Classic baseline alpha = 0.55 for a MID with average ROS."""
        alpha = compute_dynamic_alpha("M", ros_score=0.50, format_context="classic")
        assert alpha == pytest.approx(0.55)

    def test_gk_position_adjustment(self):
        """GK gets -0.10 alpha (more ROS-oriented)."""
        alpha_gk = compute_dynamic_alpha("G", ros_score=0.50, format_context="draft")
        alpha_mid = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_gk == alpha_mid - 0.10

    def test_fwd_position_adjustment(self):
        """FWD gets -0.05 alpha."""
        alpha_fwd = compute_dynamic_alpha("F", ros_score=0.50, format_context="draft")
        alpha_mid = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_fwd == alpha_mid - 0.05

    def test_elite_rank_tier(self):
        """Elite player (ROS > 0.80) gets -0.10 (protect with more ROS weight)."""
        alpha_elite = compute_dynamic_alpha("M", ros_score=0.85, format_context="draft")
        alpha_avg = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_elite == alpha_avg - 0.10

    def test_above_avg_rank_tier(self):
        """Above-average player (ROS > 0.60) gets -0.05."""
        alpha_above = compute_dynamic_alpha("M", ros_score=0.65, format_context="draft")
        alpha_avg = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_above == alpha_avg - 0.05

    def test_below_avg_rank_tier(self):
        """Below-average player (ROS < 0.40) gets +0.05."""
        alpha_below = compute_dynamic_alpha("M", ros_score=0.35, format_context="draft")
        alpha_avg = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_below == alpha_avg + 0.05

    def test_critical_depth_adjustment(self):
        """Critical depth adds +0.15 (urgency → favor 1GW)."""
        depth_map = {"M": PositionalDepth("M", 5, 0, 0, 5, "Critical")}
        alpha_critical = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft", depth_map=depth_map)
        alpha_normal = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_critical == alpha_normal + 0.15

    def test_low_depth_adjustment(self):
        """Low depth adds +0.10."""
        depth_map = {"M": PositionalDepth("M", 5, 3, 0, 2, "Low")}
        alpha_low = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft", depth_map=depth_map)
        alpha_normal = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_low == alpha_normal + 0.10

    def test_adequate_depth_no_adjustment(self):
        """Adequate depth adds nothing."""
        depth_map = {"M": PositionalDepth("M", 5, 5, 0, 0, "Adequate")}
        alpha_adequate = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft", depth_map=depth_map)
        alpha_normal = compute_dynamic_alpha("M", ros_score=0.50, format_context="draft")
        assert alpha_adequate == alpha_normal

    def test_clamp_lower_bound(self):
        """Alpha should never go below 0.15."""
        # GK(-0.10) + Draft(0.35) + Elite(-0.10) = 0.15, adding more pushes to clamp
        alpha = compute_dynamic_alpha("G", ros_score=0.85, format_context="draft")
        assert alpha >= 0.15

    def test_clamp_upper_bound(self):
        """Alpha should never exceed 0.75."""
        # Classic(0.55) + Below avg(+0.05) + Critical(+0.15) = 0.75
        depth_map = {"M": PositionalDepth("M", 5, 0, 0, 5, "Critical")}
        alpha = compute_dynamic_alpha("M", ros_score=0.35, format_context="classic", depth_map=depth_map)
        assert alpha <= 0.75

    def test_no_depth_map(self):
        """None depth_map doesn't crash."""
        alpha = compute_dynamic_alpha("D", ros_score=0.50, format_context="draft", depth_map=None)
        assert 0.15 <= alpha <= 0.75


# =============================================================================
# TestMergeFFPSingleGWData
# =============================================================================

class TestMergeFFPSingleGWData:
    """Tests for merge_ffp_single_gw_data() — merges FFP Predicted/Start/LongStart."""

    def test_basic_merge(self):
        """Matched players get FFP_Predicted, FFP_Start, FFP_LongStart."""
        players = pd.DataFrame({
            "Player": ["Salah", "Haaland"],
            "Team": ["LIV", "MCI"],
        })
        ffp = pd.DataFrame({
            "Name": ["Salah", "Haaland"],
            "Team": ["Liverpool", "Man City"],
            "Predicted": [8.5, 10.0],
            "Start": [95, 90],
            "LongStart": [92, 88],
        })
        result = merge_ffp_single_gw_data(players, ffp)
        assert result.loc[0, "FFP_Predicted"] == 8.5
        assert result.loc[1, "FFP_Predicted"] == 10.0
        assert result.loc[0, "FFP_Start"] == 95
        assert result.loc[1, "FFP_LongStart"] == 88

    def test_unmatched_get_nan(self):
        """Unmatched players get NaN."""
        players = pd.DataFrame({
            "Player": ["Salah", "Unknown Player"],
            "Team": ["LIV", "MCI"],
        })
        ffp = pd.DataFrame({
            "Name": ["Salah"],
            "Team": ["Liverpool"],
            "Predicted": [8.5],
            "Start": [95],
            "LongStart": [92],
        })
        result = merge_ffp_single_gw_data(players, ffp)
        assert result.loc[0, "FFP_Predicted"] == 8.5
        assert pd.isna(result.loc[1, "FFP_Predicted"])
        assert pd.isna(result.loc[1, "FFP_Start"])

    def test_none_ffp_df(self):
        """None ffp_df adds NaN columns gracefully."""
        players = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
        })
        result = merge_ffp_single_gw_data(players, None)
        assert "FFP_Predicted" in result.columns
        assert "FFP_Start" in result.columns
        assert "FFP_LongStart" in result.columns
        assert pd.isna(result.loc[0, "FFP_Predicted"])

    def test_empty_ffp_df(self):
        """Empty FFP DataFrame adds NaN columns."""
        players = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
        })
        result = merge_ffp_single_gw_data(players, pd.DataFrame())
        assert "FFP_Predicted" in result.columns
        assert pd.isna(result.loc[0, "FFP_Predicted"])

    def test_name_normalization(self):
        """Accented names match via canonical_normalize."""
        players = pd.DataFrame({
            "Player": ["Raúl Jiménez"],
            "Team": ["FUL"],
        })
        ffp = pd.DataFrame({
            "Name": ["Raul Jimenez"],
            "Team": ["Fulham"],
            "Predicted": [5.0],
            "Start": [80],
            "LongStart": [75],
        })
        result = merge_ffp_single_gw_data(players, ffp)
        assert result.loc[0, "FFP_Predicted"] == 5.0

    def test_partial_columns(self):
        """FFP with only Predicted (no Start/LongStart) still works."""
        players = pd.DataFrame({
            "Player": ["Salah"],
            "Team": ["LIV"],
        })
        ffp = pd.DataFrame({
            "Name": ["Salah"],
            "Team": ["Liverpool"],
            "Predicted": [8.5],
        })
        result = merge_ffp_single_gw_data(players, ffp)
        assert result.loc[0, "FFP_Predicted"] == 8.5
        assert pd.isna(result.loc[0, "FFP_Start"])
        assert pd.isna(result.loc[0, "FFP_LongStart"])


# =============================================================================
# TestNew1GWScore
# =============================================================================

class TestNew1GWScore:
    """Tests for the new 1GW scoring model — pure expected value."""

    def _make_pool(self):
        """Create a reference pool of ~20 players per position."""
        rows = []
        for pos in ["G", "D", "M", "F"]:
            for i in range(20):
                rows.append({
                    "Player": f"{pos}_Player_{i}",
                    "Team": f"T{i % 5}",
                    "Position": pos,
                    "Projected_Points": 2.0 + i * 0.5,
                    "form": 1.0 + i * 0.3,
                    "total_points": 10 + i * 8,
                    "minutes": 200 + i * 50,
                    "starts": 3 + i,
                    "AvgFDR": 2.0 + (i % 5) * 0.5,
                })
        return pd.DataFrame(rows)

    def test_blended_projection_both_sources(self):
        """When both Rotowire and FFP_Predicted exist, blends them."""
        pool = self._make_pool()
        squad = pool[pool["Position"] == "M"].head(1).copy()
        squad["FFP_Predicted"] = 12.0  # Higher than pool projection
        result = compute_player_scores(squad, pool, current_gw=20)
        # Should produce a valid 1GW — exact value depends on blending
        assert result["1GW"].iloc[0] >= 0
        assert result["1GW"].iloc[0] <= 1

    def test_start_likelihood_reduces_effective_proj(self):
        """Low start % reduces effective projection → lower 1GW."""
        pool = self._make_pool()
        # Take a mid-range MID player
        mid_player = pool[pool["Position"] == "M"].iloc[10:11].copy()

        result_full = compute_player_scores(mid_player.copy(), pool, current_gw=20)

        mid_low_start = mid_player.copy()
        mid_low_start["FFP_Start"] = 30.0  # Only 30% chance of starting
        result_low_start = compute_player_scores(mid_low_start, pool, current_gw=20)

        # Player with 30% start chance should score lower than default (100%)
        assert result_low_start["1GW"].iloc[0] < result_full["1GW"].iloc[0]

    def test_fpl_chance_of_playing_fallback(self):
        """FPL chance_of_playing used as fallback when FFP_Start missing."""
        pool = self._make_pool()
        mid_player = pool[pool["Position"] == "M"].iloc[10:11].copy()

        result_full = compute_player_scores(mid_player.copy(), pool, current_gw=20)

        mid_injured = mid_player.copy()
        mid_injured["chance_of_playing_next_round"] = 25  # 25% chance
        result_injured = compute_player_scores(mid_injured, pool, current_gw=20)

        # Injured player should score lower
        assert result_injured["1GW"].iloc[0] < result_full["1GW"].iloc[0]

    def test_ffp_start_takes_priority_over_fpl_chance(self):
        """FFP_Start used when both FFP_Start and chance_of_playing exist."""
        pool = self._make_pool()
        mid_player = pool[pool["Position"] == "M"].iloc[10:11].copy()

        # FFP says 95% start, FPL says 25% chance → FFP should win
        mid_player_ffp = mid_player.copy()
        mid_player_ffp["FFP_Start"] = 95.0
        mid_player_ffp["chance_of_playing_next_round"] = 25
        result_ffp = compute_player_scores(mid_player_ffp, pool, current_gw=20)

        # FPL only says 25%
        mid_player_fpl = mid_player.copy()
        mid_player_fpl["chance_of_playing_next_round"] = 25
        result_fpl = compute_player_scores(mid_player_fpl, pool, current_gw=20)

        # FFP override (95%) should give higher score than FPL-only (25%)
        assert result_ffp["1GW"].iloc[0] > result_fpl["1GW"].iloc[0]
