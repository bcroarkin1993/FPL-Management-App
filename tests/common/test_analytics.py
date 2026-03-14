"""Unit tests for scripts.common.analytics multi-GW transfer planner functions.

All tests use mock DataFrames — no network calls.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import MagicMock

from scripts.common.analytics import (
    compute_healthy_form,
    compute_positional_depth,
    compute_transfer_urgency,
    blend_multi_gw_projections,
    positional_percentile,
    positional_rank,
    dampen_form_by_starts,
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
