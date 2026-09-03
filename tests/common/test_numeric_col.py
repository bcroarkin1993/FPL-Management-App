"""Regression tests for missing-column handling in the scoring pipeline.

`DataFrame.get(col)` returns **None** for a missing column and
`DataFrame.get(col, 0)` returns the scalar `0` — neither is a Series. The
`pd.to_numeric(...).fillna(...)` idiom these replace therefore raised
`AttributeError: 'numpy.float64' object has no attribute 'fillna'` exactly when a
frame arrived without the column, which is the case the defaults existed for.

It surfaced as Draft Power Rankings failing, and the live plausibility suite
recorded it as *"Draft league strength unreachable"* — an app bug wearing an
outage's clothes.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common.analytics import compute_player_scores, numeric_col


class TestNumericCol:
    def test_reads_an_existing_column(self):
        df = pd.DataFrame({"A": ["1.5", "2", None]})
        out = numeric_col(df, "A", 0)
        assert list(out) == [1.5, 2.0, 0.0]

    def test_missing_column_becomes_the_default(self):
        df = pd.DataFrame({"A": [1.0, 2.0]})
        out = numeric_col(df, "Nope", 3.0)
        assert isinstance(out, pd.Series)
        assert list(out) == [3.0, 3.0]

    def test_result_aligns_with_the_frames_index(self):
        """A misaligned Series silently produces NaN on assignment."""
        df = pd.DataFrame({"A": [1.0, 2.0]}, index=[7, 9])
        assert list(numeric_col(df, "Nope", 1.0).index) == [7, 9]

    def test_empty_frame_is_safe(self):
        out = numeric_col(pd.DataFrame(), "Nope", 0)
        assert isinstance(out, pd.Series)
        assert out.empty

    def test_non_numeric_values_fall_back_to_the_default(self):
        df = pd.DataFrame({"A": ["not a number", "4"]})
        assert list(numeric_col(df, "A", -1)) == [-1.0, 4.0]

    def test_assigning_it_back_does_not_raise(self):
        """The shape the old idiom broke on."""
        df = pd.DataFrame({"A": [1.0, 2.0]})
        df["Missing"] = numeric_col(df, "Missing", 0)
        assert list(df["Missing"]) == [0.0, 0.0]


class TestComputePlayerScoresMissingColumns:
    def _minimal(self):
        """The columns the caller genuinely always supplies — nothing more."""
        return pd.DataFrame({
            "Player": ["A. Player", "B. Player", "C. Player"],
            "Team": ["MCI", "ARS", "LIV"],
            "Position": ["F", "M", "D"],
            "Points": [8.0, 6.0, 4.0],
        })

    def _reference(self):
        """A full-pool reference frame, as the callers pass."""
        return pd.DataFrame({
            "Player": ["A. Player", "B. Player", "C. Player", "D. Player"],
            "Team": ["MCI", "ARS", "LIV", "CHE"],
            "Position": ["F", "M", "D", "M"],
            "Points": [8.0, 6.0, 4.0, 5.0],
        })

    def test_survives_without_season_points_or_fdr(self):
        """Season_Points and AvgFDRNextN are absent preseason and on any frame
        built from a degraded source."""
        out = compute_player_scores(self._minimal(), self._reference(), current_gw=3)
        for col in ("1GW", "ROS", "Transfer Score", "Keep Score"):
            assert col in out.columns
            assert out[col].notna().all()

    def test_scores_stay_in_range(self):
        out = compute_player_scores(self._minimal(), self._reference(), current_gw=3)
        for col in ("1GW", "ROS", "Transfer Score", "Keep Score"):
            assert out[col].between(0.0, 1.0).all(), col

    def test_missing_projection_column_is_not_fatal(self):
        df = self._minimal().drop(columns=["Points"])
        out = compute_player_scores(df, self._reference(), current_gw=3)
        assert len(out) == 3

    def test_empty_frame_is_safe(self):
        out = compute_player_scores(
            pd.DataFrame(columns=["Player", "Team", "Position", "Points"]),
            self._reference(), current_gw=3)
        assert out.empty or "1GW" in out.columns
