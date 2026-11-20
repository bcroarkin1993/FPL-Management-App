"""The dark plotly theme, in one place.

Nine page-local copies of ``_DARK_CHART_LAYOUT`` had grown up across the app in
**five** distinct variants, so "the house chart theme" was not one thing, and a
tenth copy was the likely outcome of any new charting work. That is the
twin-drift failure this codebase keeps paying for, arriving in the styling
layer -- see "Projection Engine -- one blend, one contract" in CLAUDE.md for
the expensive version of it.

Three of those variants were *exactly* duplicated across two or more pages, so
collapsing each onto a shared constant is provably a no-op. **Each constant
below keeps the notation its originals used**, nested ``title=dict(...)`` or
flat ``title_font``/``title_x``, rather than being normalised onto one form:
plotly treats the two as equivalent, but a migration that is byte-identical
needs no argument about whether it is.

Two copies have genuinely diverged and are deliberately left where they are
rather than silently restyled:

* ``draft/league_analysis.py`` -- ``#333`` gridlines and ``hovermode="x unified"``
* ``draft/trade_analyzer.py`` -- 12px tick fonts

``tests/common/test_chart_theme.py`` is the guard against a tenth copy.

**Never pass ``xaxis=`` or ``yaxis=`` alongside ``**DARK_LAYOUT``.** Both are
keys of these dicts, so ``update_layout(**DARK_LAYOUT, xaxis=dict(...))``
raises ``TypeError: got multiple values for keyword argument 'xaxis'`` -- a
recurring papercut recorded in the project's own notes. Use
``fig.update_xaxes()`` / ``fig.update_yaxes()``, or ``dark_layout(xaxis=...)``
below, which merges one level deep instead of colliding.

Streamlit-free on purpose: this is a dict of colours, and importing it must not
drag the UI framework into a module that only wants to draw.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

#: The palette these layouts are built from, shared with ``cards.py``.
PANEL = "#1a1a2e"
GRID = "#444"
TEXT = "#ffffff"
LEGEND_BG = "rgba(0,0,0,0)"

#: The canonical layout: styled axes, a 20px centred title.
#: Originals: ``classic/team_analysis.py``, ``classic/league_analysis.py``,
#: ``common/bench_analysis.py``.
DARK_LAYOUT: Dict[str, Any] = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=14),
    title_font=dict(size=20, color=TEXT),
    title_x=0.5,
    title_xanchor="center",
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color=TEXT, size=13)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color=TEXT, size=13)),
    legend=dict(bgcolor=LEGEND_BG, font=dict(color=TEXT, size=13)),
)

#: As above with a 22px title, expressed in the nested form.
#: Originals: ``draft/home.py``, ``classic/home.py``.
DARK_LAYOUT_BIG_TITLE: Dict[str, Any] = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=14),
    title=dict(font=dict(size=22, color=TEXT), x=0.5, xanchor="center"),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color=TEXT, size=13)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID,
               tickfont=dict(color=TEXT, size=13)),
    legend=dict(bgcolor=LEGEND_BG, font=dict(color=TEXT, size=13)),
)

#: No axis styling at all -- for the Wrapped pages, whose charts set their own.
#: Originals: ``draft/season_wrapped.py``, ``draft/league_wrapped.py``.
DARK_LAYOUT_BARE: Dict[str, Any] = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=13),
    legend=dict(bgcolor=LEGEND_BG, font=dict(color=TEXT, size=12)),
)


#: A chart that shares a row, re-proportioned. See `compact_layout`.
DARK_LAYOUT_COMPACT: Dict[str, Any] = dict(
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    font=dict(color=TEXT, size=11),
    showlegend=False,
    # Margins are the half that actually breaks. Plotly does **not** grow a
    # margin to fit an axis title or a tick label -- it draws them into
    # whatever room it was given and lets them overlap or clip. So these are
    # reserved deliberately: left for the y tick labels plus a rotated title,
    # bottom for the x labels plus its title, top for a little air.
    margin=dict(l=62, r=16, t=16, b=52),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, zeroline=False,
               tickfont=dict(color=TEXT, size=10),
               title=dict(font=dict(color="#9b96ad", size=11), standoff=10)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, zeroline=False,
               tickfont=dict(color=TEXT, size=10),
               title=dict(font=dict(color="#9b96ad", size=11), standoff=10)),
    legend=dict(bgcolor=LEGEND_BG, font=dict(color=TEXT, size=10)),
    hoverlabel=dict(bgcolor="#16213e", bordercolor=GRID,
                    font=dict(color=TEXT, size=11)),
)


def compact_layout(**overrides: Any) -> Dict[str, Any]:
    """`DARK_LAYOUT` re-proportioned for a chart that shares a row.

    `DARK_LAYOUT` was tuned for one full-width chart with a 20px centred
    title: 14px body text, 13px ticks, and **no opinion about margins at
    all**. Two charts side by side in `st.columns` are about 340px wide, where
    those sizes crowd the plot area and an axis title has nowhere to sit.

    The margins are the part that actually goes wrong, and it is worth being
    explicit about why: plotly does not grow a margin to fit axis furniture, so
    a 10px left margin does not shrink the y-axis title -- it draws the title
    and the tick labels on top of each other, which is exactly how the
    Projections Hub's first pair of charts came out.

    Legends are off by default: a horizontal legend above a 330px chart costs a
    tenth of its height, and a per-chart heading and caption carry the same
    information in the page's own voice. Turn it back on with
    `showlegend=True` where a chart genuinely needs one.
    """
    return dark_layout(DARK_LAYOUT_COMPACT, **overrides)


def dark_layout(base: Optional[Dict[str, Any]] = None,
                **overrides: Any) -> Dict[str, Any]:
    """``DARK_LAYOUT`` with ``overrides`` merged in, one level deep.

    The point of merging rather than replacing is the ``xaxis`` collision in
    the module docstring: ``dark_layout(xaxis=dict(title="Points"))`` keeps the
    theme's gridcolour and tick font and adds the title, where
    ``{**DARK_LAYOUT, "xaxis": {"title": "Points"}}`` would throw both away.

    Only one level deep, deliberately -- a recursive merge would make
    ``tickfont`` silently inheritable too, and the shallow rule is the one a
    caller can predict without reading this function. Pass a complete nested
    dict where you want to replace one outright.

    Returns a fresh deep copy every call, so a caller mutating the result
    cannot reach the module constants -- the same reasoning as
    ``blended_pool`` using ``cache_data`` rather than ``cache_resource``.
    """
    merged = deepcopy(base if base is not None else DARK_LAYOUT)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            current.update(value)
        else:
            merged[key] = value
    return merged
