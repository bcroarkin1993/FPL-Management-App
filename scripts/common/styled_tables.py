# scripts/common/styled_tables.py
"""
Dark-themed HTML table renderer for Streamlit.

Provides `render_styled_table()` which converts a DataFrame to an HTML table
matching the app's existing dark aesthetic (#1a1a2e backgrounds, #00ff87 accents,
FPL purple gradient headers).
"""

import math

import numpy as np
import pandas as pd
import streamlit as st
from typing import Callable, Dict, List, Optional, Sequence

from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.styled_tables")


# ---------------------------------------------------------------------------
# Inline styles (applied directly to each table element — no CSS class deps)
# ---------------------------------------------------------------------------
_WRAP_STYLE = "border:1px solid #333;border-radius:10px;overflow:hidden;margin-bottom:1rem;"
_WRAP_SCROLL = "overflow-y:auto;"
_TITLE_STYLE = "background:linear-gradient(135deg,#37003c 0%,#5a0060 100%);color:#00ff87;font-weight:700;font-size:1.05rem;padding:10px 16px;margin:0;"
_TABLE_STYLE = "width:100%;border-collapse:collapse;font-size:14px;background:#1a1a2e;"
_TH_STYLE = "background:linear-gradient(135deg,#37003c,#5a0060);color:#00ff87;font-weight:600;font-size:13px;padding:10px 12px;border-bottom:2px solid #00ff87;position:sticky;top:0;z-index:1;"
_TD_STYLE = "padding:8px 12px;color:#e0e0e0;border-bottom:1px solid #333;"
_TR_EVEN_BG = "background:rgba(255,255,255,0.03);"
_TR_ODD_BG = "background:#1a1a2e;"
_TR_HIGHLIGHT = "border-left:3px solid #00ff87;"


# ---------------------------------------------------------------------------
# Color-scale helpers
# ---------------------------------------------------------------------------
def _is_finite(val) -> bool:
    """True only for a real, finite number. Guards against NaN, +/-inf, and
    anything non-numeric that happens to be truthy."""
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _color_scale(val, col_min, col_max, direction="positive"):
    """
    Return an inline CSS color string for a numeric value.

    direction='positive': low=red, high=green
    direction='negative': low=green, high=red
    """
    # Non-finite input has to be rejected explicitly. NaN == NaN is False, so an
    # all-NaN column slips past a `col_max == col_min` check, and an infinite
    # value (e.g. points/price where price is 0) makes the ratio inf/inf = NaN.
    # Either way int(NaN) raises and, because this runs inside the render loop,
    # a single bad cell used to take down the entire page.
    if not all(_is_finite(v) for v in (val, col_min, col_max)) or col_max == col_min:
        return ""
    ratio = (val - col_min) / (col_max - col_min)
    if direction == "negative":
        ratio = 1 - ratio
    # red (0) -> yellow (0.5) -> green (1)
    if ratio <= 0.5:
        t = ratio / 0.5
        r, g, b = int(220 - 60 * t), int(60 + 140 * t), 60
    else:
        t = (ratio - 0.5) / 0.5
        r, g, b = int(160 - 120 * t), int(200 + 20 * t), int(60 + 40 * t)
    return f"color: rgb({r},{g},{b}); font-weight: 600;"


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------
def render_styled_table(
    df: pd.DataFrame,
    title: str = None,
    col_formats: Dict[str, str] = None,
    text_align: Dict[str, str] = None,
    highlight_row: Callable = None,
    positive_color_cols: List[str] = None,
    negative_color_cols: List[str] = None,
    color_range_overrides: Dict[str, tuple] = None,
    color_values: Dict[str, Sequence[float]] = None,
    max_height: int = None,
    font_size: int = 14,
    title_font_size: int = None,
    cell_padding: str = None,
):
    """
    Render a DataFrame as a dark-themed HTML table via st.markdown.

    Parameters
    ----------
    df : DataFrame to display.
    title : Optional header rendered above the table inside the wrapper.
    col_formats : {col: format_spec} for value formatting.
    text_align : {col: "left"|"center"|"right"}.
    highlight_row : fn(row) -> bool. Matching rows get an accent left-border.
    positive_color_cols : Columns where higher values are greener.
    color_range_overrides : {col: (min, max)} to colour against an external
        reference instead of the rows being rendered. Use this whenever the
        table is a *selection* from a larger population: the weakest of 11
        hand-picked players is not a bad value, but scaled against only its
        peers it renders as pure red. Colour it against the full pool and it
        reads as what it is.
    color_values : {col: sequence aligned to df rows} to drive the colour from
        something other than the displayed number. Needed when the right
        comparison differs per row -- e.g. grading a goalkeeper against other
        goalkeepers and a midfielder against midfielders, where one column-wide
        range cannot express both. Pair it with color_range_overrides, since
        otherwise the range is taken from these values within this table and
        the outside reference is lost again.
    negative_color_cols : Columns where higher values are redder.
    max_height : Optional max-height in px (enables vertical scroll).
    font_size : Data and column-header font size in px (default 14).
    title_font_size : Title bar font size in px; defaults to slightly above font_size.
    cell_padding : Explicit CSS padding for data cells e.g. "14px 16px".
                   Overrides the formula-based default.
    """
    if df is None or df.empty:
        st.info("No data to display.")
        return

    col_formats = col_formats or {}
    text_align = text_align or {}
    positive_color_cols = positive_color_cols or []
    negative_color_cols = negative_color_cols or []
    color_range_overrides = color_range_overrides or {}
    color_values = color_values or {}
    # Align supplied colour values positionally to the rendered rows.
    color_series = {}
    for col, values in color_values.items():
        series = pd.Series(list(values))
        if len(series) != len(df):
            _logger.warning(
                "color_values for %r has %d entries but the table has %d rows; "
                "ignoring and colouring by the displayed value.",
                col, len(series), len(df))
            continue
        color_series[col] = pd.to_numeric(series, errors="coerce").to_numpy()

    # Pre-compute min/max for color-scaled columns
    color_ranges = {}
    for col in positive_color_cols + negative_color_cols:
        if col in color_range_overrides:
            lo, hi = color_range_overrides[col]
            if _is_finite(lo) and _is_finite(hi) and hi != lo:
                color_ranges[col] = (lo, hi)
                continue
            _logger.warning(
                "Ignoring degenerate colour range %r for column %r; falling "
                "back to the table's own values.", color_range_overrides[col], col)
        if col in color_series:
            numeric_vals = pd.Series(color_series[col])
        elif col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            finite_vals = numeric_vals[np.isfinite(numeric_vals)]
            n_nonfinite = int(numeric_vals.notna().sum() - len(finite_vals))
            if n_nonfinite:
                # Worth surfacing: an infinity in a numeric column is almost
                # always an unguarded division upstream, not real data.
                _logger.warning(
                    "Styled table column %r contains %d non-finite value(s) "
                    "(inf/-inf); they are excluded from the colour range and "
                    "rendered uncoloured. This usually means a division by zero "
                    "upstream.", col, n_nonfinite,
                )
            if not finite_vals.empty:
                color_ranges[col] = (finite_vals.min(), finite_vals.max())

    # Determine default alignment per column
    def _align(col):
        if col in text_align:
            return text_align[col]
        if df[col].dtype.kind in ("i", "f", "u"):  # numeric
            return "right"
        return "left"

    # Build HTML with fully inline styles (no CSS class dependencies)
    parts = []

    # Wrapper
    wrap_style = _WRAP_STYLE
    if max_height:
        wrap_style += _WRAP_SCROLL + f"max-height:{max_height}px;"
    parts.append(f'<div style="{wrap_style}">')

    # Title
    if title:
        t_size = f"font-size:{title_font_size}px;" if title_font_size else ""
        title_style = _TITLE_STYLE.replace("font-size:1.05rem;", f"font-size:1.05rem;{t_size}")
        parts.append(f'<div style="{title_style}">{title}</div>')

    th_size = f"font-size:{font_size + 1}px;" if font_size != 14 else ""
    if cell_padding:
        td_pad = f"padding:{cell_padding};"
    elif font_size != 14:
        td_pad = f"padding:{max(6, font_size - 5)}px {max(10, font_size - 2)}px;"
    else:
        td_pad = ""
    table_style = _TABLE_STYLE.replace("font-size:14px;", f"font-size:{font_size}px;")
    parts.append(f'<table style="{table_style}">')

    # Header
    parts.append("<thead><tr>")
    for col in df.columns:
        align = _align(col)
        parts.append(f'<th style="{_TH_STYLE}{th_size}text-align:{align};">{col}</th>')
    parts.append("</tr></thead>")

    # Body
    parts.append("<tbody>")
    for row_idx, (_, row) in enumerate(df.iterrows()):
        row_bg = _TR_EVEN_BG if row_idx % 2 == 1 else _TR_ODD_BG
        row_extra = ""
        if highlight_row and highlight_row(row):
            row_extra = _TR_HIGHLIGHT
        parts.append(f'<tr style="{row_bg}{row_extra}">')

        for col in df.columns:
            val = row[col]
            align = _align(col)

            # Format value
            if col in col_formats and pd.notna(val):
                try:
                    display_val = col_formats[col].format(val)
                except (ValueError, TypeError):
                    display_val = str(val) if pd.notna(val) else ""
            else:
                if pd.isna(val):
                    display_val = ""
                elif isinstance(val, float):
                    display_val = f"{val:g}"
                else:
                    display_val = str(val)

            # Color scaling
            extra_style = ""
            # Colour from the supplied series when given, so the shade can
            # reflect a comparison the displayed number doesn't encode.
            color_val = color_series[col][row_idx] if col in color_series else val
            if col in positive_color_cols and col in color_ranges and pd.notna(color_val):
                cmin, cmax = color_ranges[col]
                extra_style = _color_scale(float(color_val), cmin, cmax, "positive")
            elif col in negative_color_cols and col in color_ranges and pd.notna(color_val):
                cmin, cmax = color_ranges[col]
                extra_style = _color_scale(float(color_val), cmin, cmax, "negative")

            parts.append(
                f'<td style="{_TD_STYLE}{td_pad}text-align:{align};{extra_style}">{display_val}</td>'
            )

        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    st.markdown("".join(parts), unsafe_allow_html=True)
