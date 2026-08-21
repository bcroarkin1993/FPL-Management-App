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
import re
from unittest.mock import patch

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


class TestColorRangeOverrides:
    """Colouring a selection against the population it was drawn from.

    An optimizer's starting XI is the best 11 of ~600 players. Scaled against
    only its own rows, the weakest of them renders pure red even though it sits
    in the top 5% of the league -- the colour says the opposite of the truth.
    """

    DF = pd.DataFrame({"Player": ["Best", "Weakest starter", "Bench GK"],
                       "Exp Pts/GW": [6.06, 4.01, 1.90]})

    def _colors(self, **kwargs):
        out = {}
        with patch("scripts.common.styled_tables.st") as mock_st:
            mock_st.markdown.side_effect = lambda html, *a, **k: out.setdefault("html", html)
            render_styled_table(self.DF, positive_color_cols=["Exp Pts/GW"], **kwargs)
        return re.findall(r"color: (rgb\([^)]*\)); font-weight: 600", out["html"])

    @staticmethod
    def _redness(color):
        r, g, _ = (int(v) for v in re.findall(r"\d+", color))
        return r - g

    def test_without_override_the_weakest_row_is_pure_red(self):
        """Documents the behaviour the override exists to correct."""
        assert self._colors()[-1] == "rgb(220,60,60)"

    def test_override_lifts_a_row_that_is_good_in_the_wider_pool(self):
        pool = self._colors(color_range_overrides={"Exp Pts/GW": (0.23, 6.06)})
        squad = self._colors()
        assert self._redness(pool[-1]) < self._redness(squad[-1])
        assert self._redness(pool[1]) < self._redness(squad[1])

    def test_top_value_stays_green_either_way(self):
        assert self._colors()[0] == self._colors(
            color_range_overrides={"Exp Pts/GW": (0.23, 6.06)})[0]

    def test_degenerate_range_falls_back_to_the_table(self):
        """A zero-width or non-finite range must not blank out the column."""
        for bad in ((5.0, 5.0), (float("nan"), 6.0), (0.0, float("inf"))):
            assert self._colors(color_range_overrides={"Exp Pts/GW": bad}) == self._colors()

    def test_override_for_an_uncoloured_column_is_ignored(self):
        assert self._colors(color_range_overrides={"Player": (0.0, 1.0)}) == self._colors()


class TestColorValues:
    """Colouring a column by a parallel series instead of its own numbers.

    Needed when the right comparison differs per row. Positions are not
    comparable on raw expected points -- the best goalkeeper in the game
    projects roughly what a mid-table midfielder does -- so one column-wide
    range paints every keeper red for being a keeper.
    """

    DF = pd.DataFrame({
        "Player": ["Best GK", "Best MID", "Backup GK"],
        "Pos": ["G", "M", "G"],
        "Exp Pts/GW": [4.31, 6.06, 1.90],
    })

    def _colors(self, **kwargs):
        out = {}
        with patch("scripts.common.styled_tables.st") as mock_st:
            mock_st.markdown.side_effect = lambda html, *a, **k: out.setdefault("html", html)
            render_styled_table(self.DF, positive_color_cols=["Exp Pts/GW"], **kwargs)
        return re.findall(r"color: (rgb\([^)]*\)); font-weight: 600", out["html"])

    def _positional(self):
        # Best-in-position both grade 1.0 despite very different raw values.
        return self._colors(
            color_values={"Exp Pts/GW": [1.0, 1.0, 0.34]},
            color_range_overrides={"Exp Pts/GW": (0.0, 1.0)},
        )

    def test_best_in_position_grade_equally(self):
        colors = self._positional()
        assert colors[0] == colors[1], (
            "the best goalkeeper and the best midfielder should read the same, "
            "even though the keeper's raw projection is far lower"
        )

    def test_without_it_the_best_gk_is_penalised_for_being_a_gk(self):
        """Documents the behaviour color_values exists to correct."""
        assert self._colors()[0] != self._colors()[1]

    def test_displayed_values_are_untouched(self):
        out = {}
        with patch("scripts.common.styled_tables.st") as mock_st:
            mock_st.markdown.side_effect = lambda html, *a, **k: out.setdefault("html", html)
            render_styled_table(
                self.DF, positive_color_cols=["Exp Pts/GW"],
                col_formats={"Exp Pts/GW": "{:.2f}"},
                color_values={"Exp Pts/GW": [1.0, 1.0, 0.34]},
                color_range_overrides={"Exp Pts/GW": (0.0, 1.0)},
            )
        assert "4.31" in out["html"] and "1.90" in out["html"]

    def test_length_mismatch_falls_back_rather_than_misaligning(self):
        """Colouring row 0 with row 1's grade would be worse than not colouring."""
        assert self._colors(color_values={"Exp Pts/GW": [1.0, 0.5]}) == self._colors()

    def test_nan_grade_leaves_the_cell_uncoloured(self):
        colors = self._colors(
            color_values={"Exp Pts/GW": [1.0, float("nan"), 0.34]},
            color_range_overrides={"Exp Pts/GW": (0.0, 1.0)},
        )
        assert len(colors) == 2
