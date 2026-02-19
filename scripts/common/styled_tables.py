# scripts/common/styled_tables.py
"""
Dark-themed HTML table renderer for Streamlit.

Provides `render_styled_table()` which converts a DataFrame to an HTML table
matching the app's existing dark aesthetic (#1a1a2e backgrounds, #00ff87 accents,
FPL purple gradient headers).
"""

import pandas as pd
import streamlit as st
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# CSS (injected once per session)
# ---------------------------------------------------------------------------
_CSS_KEY = "_styled_tables_css_injected"

_TABLE_CSS = """
<style>
.styled-tbl-wrap {
    border: 1px solid #333;
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.styled-tbl-wrap.scroll {
    overflow-y: auto;
}
.styled-tbl-title {
    background: linear-gradient(135deg, #37003c 0%, #5a0060 100%);
    color: #00ff87;
    font-weight: 700;
    font-size: 1.05rem;
    padding: 10px 16px;
    margin: 0;
}
table.styled-tbl {
    width: 100%;
    border-collapse: collapse;
    font-size: 14px;
    background: #1a1a2e;
}
table.styled-tbl thead th {
    background: linear-gradient(135deg, #37003c, #5a0060);
    color: #00ff87;
    font-weight: 600;
    font-size: 13px;
    padding: 10px 12px;
    border-bottom: 2px solid #00ff87;
    position: sticky;
    top: 0;
    z-index: 1;
}
table.styled-tbl tbody td {
    padding: 8px 12px;
    color: #e0e0e0;
    border-bottom: 1px solid #333;
}
table.styled-tbl tbody tr {
    background: #1a1a2e;
}
table.styled-tbl tbody tr:nth-child(even) {
    background: rgba(255,255,255,0.03);
}
table.styled-tbl tbody tr:hover {
    background: rgba(0,255,135,0.05);
}
table.styled-tbl tbody tr.highlight-row {
    border-left: 3px solid #00ff87;
}
</style>
"""


def _inject_css():
    """Inject table CSS once per Streamlit session."""
    if not st.session_state.get(_CSS_KEY):
        st.markdown(_TABLE_CSS, unsafe_allow_html=True)
        st.session_state[_CSS_KEY] = True


# ---------------------------------------------------------------------------
# Color-scale helpers
# ---------------------------------------------------------------------------
def _color_scale(val, col_min, col_max, direction="positive"):
    """
    Return an inline CSS color string for a numeric value.

    direction='positive': low=red, high=green
    direction='negative': low=green, high=red
    """
    if pd.isna(val) or col_max == col_min:
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
    max_height: int = None,
):
    """
    Render a DataFrame as a dark-themed HTML table via st.markdown.

    Parameters
    ----------
    df : DataFrame to display.
    title : Optional header rendered above the table inside the wrapper.
    col_formats : {col: format_spec} for value formatting.
        Examples: {"Points": "{:,.0f}", "Price": "£{:.1f}m"}
    text_align : {col: "left"|"center"|"right"}.
        Defaults: "left" for object/string cols, "right" for numeric cols.
    highlight_row : fn(row) -> bool. Matching rows get an accent left-border.
    positive_color_cols : Columns where higher values are greener.
    negative_color_cols : Columns where higher values are redder.
    max_height : Optional max-height in px (enables vertical scroll).
    """
    _inject_css()

    if df is None or df.empty:
        st.info("No data to display.")
        return

    col_formats = col_formats or {}
    text_align = text_align or {}
    positive_color_cols = positive_color_cols or []
    negative_color_cols = negative_color_cols or []

    # Pre-compute min/max for color-scaled columns
    color_ranges = {}
    for col in positive_color_cols + negative_color_cols:
        if col in df.columns:
            numeric_vals = pd.to_numeric(df[col], errors="coerce")
            color_ranges[col] = (numeric_vals.min(), numeric_vals.max())

    # Determine default alignment per column
    def _align(col):
        if col in text_align:
            return text_align[col]
        if df[col].dtype.kind in ("i", "f", "u"):  # numeric
            return "right"
        return "left"

    # Build HTML
    parts = []

    # Wrapper open
    scroll_class = " scroll" if max_height else ""
    max_h_style = f"max-height:{max_height}px;" if max_height else ""
    parts.append(f'<div class="styled-tbl-wrap{scroll_class}" style="{max_h_style}">')

    # Title
    if title:
        parts.append(f'<div class="styled-tbl-title">{title}</div>')

    parts.append('<table class="styled-tbl">')

    # Header
    parts.append("<thead><tr>")
    for col in df.columns:
        align = _align(col)
        parts.append(f'<th style="text-align:{align};">{col}</th>')
    parts.append("</tr></thead>")

    # Body
    parts.append("<tbody>")
    for _, row in df.iterrows():
        row_class = ""
        if highlight_row and highlight_row(row):
            row_class = ' class="highlight-row"'
        parts.append(f"<tr{row_class}>")

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
            if col in positive_color_cols and col in color_ranges and pd.notna(val):
                cmin, cmax = color_ranges[col]
                extra_style = _color_scale(float(val), cmin, cmax, "positive")
            elif col in negative_color_cols and col in color_ranges and pd.notna(val):
                cmin, cmax = color_ranges[col]
                extra_style = _color_scale(float(val), cmin, cmax, "negative")

            parts.append(
                f'<td style="text-align:{align};{extra_style}">{display_val}</td>'
            )

        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    st.markdown("".join(parts), unsafe_allow_html=True)
