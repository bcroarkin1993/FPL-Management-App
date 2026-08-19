"""Tests for scripts/common/styled_tables.py's colour scaling.

Regression coverage for a Projections Hub crash. Rotowire listed a player with
Price 0.0; the page computed Value = Points / Price without a guard, producing
inf. render_styled_table's colour range then ran from 0.28 to inf, the ratio for
that cell came out inf/inf = NaN, and int(NaN) raised ValueError -- taking down
the entire page from inside the render loop.

`pd.isna(val) or col_max == col_min` does not catch this: NaN == NaN is False,
so an all-NaN range slips through too.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from scripts.common.styled_tables import _color_scale, _is_finite, render_styled_table


class TestIsFinite:
    @pytest.mark.parametrize("value", [0, 1, -1, 3.5, np.float64(2.0)])
    def test_real_numbers_are_finite(self, value):
        assert _is_finite(value) is True

    @pytest.mark.parametrize(
        "value", [float("inf"), float("-inf"), float("nan"), np.nan, None, "abc", "", [1]]
    )
    def test_everything_else_is_not(self, value):
        assert _is_finite(value) is False


class TestColorScale:
    def test_normal_value_is_coloured(self):
        assert "color: rgb(" in _color_scale(1.0, 0.0, 2.0)

    def test_infinite_value_returns_no_style(self):
        """The actual crash: points/price where price is 0."""
        assert _color_scale(float("inf"), 0.28, float("inf")) == ""

    def test_infinite_bound_returns_no_style(self):
        assert _color_scale(0.5, 0.28, float("inf")) == ""

    def test_all_nan_range_returns_no_style(self):
        """NaN == NaN is False, so the col_max == col_min guard never fired."""
        assert _color_scale(5.0, float("nan"), float("nan")) == ""

    def test_nan_value_returns_no_style(self):
        assert _color_scale(float("nan"), 0.0, 2.0) == ""

    def test_flat_column_returns_no_style(self):
        assert _color_scale(1.0, 1.0, 1.0) == ""

    def test_direction_inverts_the_gradient(self):
        low_positive = _color_scale(0.0, 0.0, 10.0, "positive")
        low_negative = _color_scale(0.0, 0.0, 10.0, "negative")
        assert low_positive != low_negative

    @pytest.mark.parametrize("ratio_value", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_every_channel_stays_in_range(self, ratio_value):
        style = _color_scale(ratio_value, 0.0, 1.0)
        channels = [int(c) for c in style.split("rgb(")[1].split(")")[0].split(",")]
        assert all(0 <= c <= 255 for c in channels), style


class TestRenderStyledTable:
    def test_infinite_cell_does_not_raise(self, mock_streamlit):
        """The end-to-end failure: one bad cell used to kill the whole page."""
        df = pd.DataFrame({
            "Player": ["A", "B", "C"],
            "Value": [1.0, 2.0, float("inf")],
        })
        render_styled_table(df, positive_color_cols=["Value"])
        rendered = " ".join(str(c) for c in mock_streamlit["markdown"].call_args_list)
        assert "A" in rendered and "C" in rendered

    def test_all_nan_column_does_not_raise(self, mock_streamlit):
        df = pd.DataFrame({"Player": ["A", "B"], "Value": [np.nan, np.nan]})
        render_styled_table(df, positive_color_cols=["Value"])

    def test_non_numeric_column_does_not_raise(self, mock_streamlit):
        df = pd.DataFrame({"Player": ["A", "B"], "Value": ["n/a", "-"]})
        render_styled_table(df, positive_color_cols=["Value"])

    def test_non_finite_values_are_flagged_in_the_log(self, mock_streamlit, caplog):
        """An infinity in a numeric column is an unguarded division upstream,
        not real data -- say so rather than silently rendering it plain."""
        df = pd.DataFrame({"Player": ["A", "B"], "Value": [1.0, float("inf")]})
        with caplog.at_level(logging.WARNING, logger="fpl_app.styled_tables"):
            render_styled_table(df, positive_color_cols=["Value"])
        assert any("non-finite" in r.message for r in caplog.records)

    def test_finite_values_still_get_coloured_alongside_an_infinity(self, mock_streamlit):
        """The infinity is excluded from the range rather than disabling colour
        for the whole column."""
        df = pd.DataFrame({"Player": ["A", "B", "C"], "Value": [1.0, 5.0, float("inf")]})
        render_styled_table(df, positive_color_cols=["Value"])
        rendered = " ".join(str(c) for c in mock_streamlit["markdown"].call_args_list)
        assert "color: rgb(" in rendered
