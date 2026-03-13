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

    def test_low_depth(self):
        """4 DEF, 3 healthy → Low (one injury from critical)."""
        roster = pd.DataFrame({
            "Position": ["D", "D", "D", "D"],
            "status": ["a", "a", "a", "i"],
            "chance_of_playing_next_round": [None, None, None, 25],
        })
        result = compute_positional_depth(roster)
        assert result["D"].depth_level == "Low"
        assert result["D"].healthy == 3
        assert result["D"].total == 4

    def test_missing_status_treated_as_healthy(self):
        """Players with both status and chance missing → assumed healthy."""
        roster = pd.DataFrame({
            "Position": ["M", "M", "M"],
            "status": [None, None, None],
            "chance_of_playing_next_round": [None, None, None],
        })
        result = compute_positional_depth(roster)
        assert result["M"].depth_level == "Adequate"
        assert result["M"].healthy == 3

    def test_chance_threshold_75(self):
        """Player with chance 75 is healthy, 74 is not."""
        roster = pd.DataFrame({
            "Position": ["G", "G"],
            "status": [None, None],
            "chance_of_playing_next_round": [75, 74],
        })
        result = compute_positional_depth(roster)
        # 75% → healthy, 74% → not healthy → 1 healthy out of 2 → Critical
        assert result["G"].healthy == 1
        assert result["G"].depth_level == "Critical"

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
        assert result["G"].depth_level == "Critical"


# =============================================================================
# TestComputeTransferUrgency
# =============================================================================

class TestComputeTransferUrgency:
    """Tests for compute_transfer_urgency."""

    def test_urgent_critical(self):
        depth_map = {"F": PositionalDepth("F", 2, 0, "Critical")}
        assert compute_transfer_urgency("F", depth_map) == "URGENT"

    def test_moderate_low(self):
        depth_map = {"D": PositionalDepth("D", 5, 3, "Low")}
        assert compute_transfer_urgency("D", depth_map) == "LOW DEPTH"

    def test_empty_adequate(self):
        depth_map = {"M": PositionalDepth("M", 5, 5, "Adequate")}
        assert compute_transfer_urgency("M", depth_map) == ""

    def test_missing_position(self):
        depth_map = {"G": PositionalDepth("G", 2, 2, "Adequate")}
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
