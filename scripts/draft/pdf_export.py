"""
Server-side PDF generation for League Wrapped using Playwright.

Completely independent of the Streamlit UI — builds a standalone HTML document,
renders it with a headless Chromium browser, and returns PDF bytes.
The Streamlit page is never touched; all layout decisions live here.
"""

import html as _html_lib
from typing import Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from scripts.draft.league_analysis import build_h2h_matrix, get_matches_df, get_team_names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PURPLE = "#7B2FBE"
_GOLD = "#FFD700"
_RED = "#FF4B4B"
_GREEN = "#00ff87"

_TEAM_COLORS = [
    "#7B2FBE", "#00b4d8", "#f72585", "#4cc9f0", "#43aa8b",
    "#f8961e", "#90be6d", "#f94144", "#9d4edd", "#3a86ff",
    "#06d6a0", "#ef476f",
]

# ---------------------------------------------------------------------------
# Global CSS injected once into <head>
# ---------------------------------------------------------------------------
_CSS = """
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
@page { size: A4; }
body {
    background: #ffffff;
    color: #1a1a2e;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 14px;
    margin: 0;
    padding: 0 1.2cm;
    box-sizing: border-box;
}
h2 {
    color: """ + _PURPLE + """;
    border-bottom: 2px solid """ + _PURPLE + """;
    padding-bottom: 8px;
    margin: 0 0 6px 0;
    font-size: 1.3em;
}
h2 .subtitle {
    display: block;
    color: #6b7280;
    font-size: 0.65em;
    font-weight: 400;
    margin-top: 3px;
}
.page-break { page-break-before: always; break-before: always; }
.no-break    { page-break-inside: avoid; break-inside:  avoid; }
.section     { margin-bottom: 12px; }
.two-col     { display: flex; gap: 12px; }
.two-col > * { flex: 1; }
.grid-4      { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; }
.grid-2      { display: grid; grid-template-columns: repeat(2,1fr); gap: 8px; }

/* ── Light card ── */
.card {
    border-radius: 10px;
    padding: 14px 16px;
    background: #f8f9fa;
    border: 1px solid #e2e8f0;
}
.card-gold  { border-color: """ + _GOLD  + """; background: linear-gradient(135deg,#fffbeb,#fef3c7); }
.card-red   { border-color: """ + _RED   + """; background: linear-gradient(135deg,#fff5f5,#fee2e2); }
.card-green { border-color: """ + _GREEN + """; }
.card-purple{ border-color: """ + _PURPLE + """; }

/* ── Light table ── */
table { width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 8px; overflow: hidden; }
thead tr th {
    background: linear-gradient(135deg, #f3e8ff, #ede9fe);
    color: """ + _PURPLE + """;
    font-weight: 600;
    font-size: 13px;
    padding: 8px 10px;
    border-bottom: 2px solid """ + _PURPLE + """;
    text-align: left;
}
th.center, td.center { text-align: center; }
th.right,  td.right  { text-align: right;  }
tbody tr td { padding: 7px 10px; color: #1a1a2e; border-bottom: 1px solid #e5e7eb; font-size: 13px; }
tbody tr:nth-child(even) td { background: #f9fafb; }
tbody tr.champion td {
    background: rgba(0, 204, 102, 0.12) !important;
    border-top: 1px solid rgba(0, 204, 102, 0.40);
    border-bottom: 1px solid rgba(0, 204, 102, 0.40);
}
tbody tr.champion td:first-child { border-left: 4px solid """ + _GREEN + """; }
tbody tr.runner-up td {
    background: rgba(157, 78, 221, 0.10) !important;
    border-top: 1px solid rgba(157, 78, 221, 0.35);
    border-bottom: 1px solid rgba(157, 78, 221, 0.35);
}
tbody tr.runner-up td:first-child { border-left: 4px solid #9d4edd; }
tbody tr.third-place td {
    background: rgba(245, 158, 11, 0.10) !important;
    border-top: 1px solid rgba(245, 158, 11, 0.35);
    border-bottom: 1px solid rgba(245, 158, 11, 0.35);
}
tbody tr.third-place td:first-child { border-left: 4px solid #f59e0b; }
tbody tr.relegated td {
    background: rgba(255, 75, 75, 0.10) !important;
    border-top: 1px solid rgba(255, 75, 75, 0.35);
    border-bottom: 1px solid rgba(255, 75, 75, 0.35);
}
tbody tr.relegated td:first-child { border-left: 4px solid """ + _RED + """; }
.pos-col { color: #059669; font-weight: 600; }
.neg-col { color: """ + _RED + """; font-weight: 600; }

/* ── Pick cards (draft) ── */
.pick-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 8px;
    padding: 10px 12px;
    background: #f8f9fa;
    margin-bottom: 8px;
}
.pick-card .name { font-weight: 700; font-size: 14px; }
.pick-card .meta { color: #6b7280; font-size: 11px; }
.pick-card .grade { font-size: 16px; font-weight: 700; text-align: right; }
.pick-card .pts   { color: #6b7280; font-size: 11px; text-align: right; }
"""

# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _esc(s) -> str:
    return _html_lib.escape(str(s))


def _section(icon: str, title: str, subtitle: str = "", page_break: bool = False,
             extra_style: str = "") -> str:
    pb = ' page-break' if page_break else ''
    sub = f'<span class="subtitle">{_esc(subtitle)}</span>' if subtitle else ''
    style_attr = f' style="{extra_style}"' if extra_style else ''
    return f'<div class="section{pb}"{style_attr}><h2>{icon} {_esc(title)}{sub}</h2>'


def _chart(fig: go.Figure, include_js: bool = False, height: int = 360,
           bottom_margin: int = 50, left_margin: int = 70, right_margin: int = 20) -> str:
    fig.update_layout(height=height, width=680,
                      margin=dict(t=50, b=bottom_margin, l=left_margin, r=right_margin))
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="cdn" if include_js else False,
        config={"displayModeBar": False, "staticPlot": True},
    )


def _color_scale_value(val: float, col_vals: List[float],
                       direction: str = "positive") -> str:
    """Return an inline CSS color string scaled relative to the column's range."""
    try:
        v = float(str(val).replace(",", "").replace("+", ""))
    except (ValueError, TypeError):
        return ""
    valid = [x for x in col_vals if x is not None]
    if not valid:
        return ""
    col_min, col_max = min(valid), max(valid)
    if direction == "diverging":
        # 0 → neutral (#6b7280), positive → green, negative → red
        abs_max = max(abs(col_min), abs(col_max), 1)
        ratio = 0.5 + 0.5 * max(-1.0, min(1.0, v / abs_max))
    else:
        if col_max == col_min:
            return ""
        ratio = (v - col_min) / (col_max - col_min)
        if direction == "negative":
            ratio = 1 - ratio
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.5:
        t = ratio / 0.5
        r, g, b = int(220 - 60 * t), int(60 + 140 * t), 60
    else:
        t = (ratio - 0.5) / 0.5
        r, g, b = int(160 - 120 * t), int(200 + 20 * t), int(60 + 40 * t)
    return f"color:rgb({r},{g},{b});font-weight:600;"


def _table(headers: List[str], rows: List[List], alignments: List[str] = None,
           highlight_row: int = None, highlight_rows: Dict[int, str] = None,
           pos_cols: List[int] = None, neg_cols: List[int] = None,
           inline_styles: List[List[str]] = None) -> str:
    """
    inline_styles: 2-D list [row_idx][col_idx] of extra inline CSS strings.
    When provided for a cell, overrides pos_cols/neg_cols coloring for that cell.
    """
    pos_cols = pos_cols or []
    neg_cols = neg_cols or []
    alignments = alignments or ["left"] + ["right"] * (len(headers) - 1)

    # Merge legacy highlight_row into highlight_rows dict
    row_classes: Dict[int, str] = dict(highlight_rows or {})
    if highlight_row is not None:
        row_classes.setdefault(highlight_row, "champion")

    def _th(h, a):
        cls = " center" if a == "center" else (" right" if a == "right" else "")
        return f'<th class="{cls}">{_esc(h)}</th>'

    def _td(val, row_i, col_i, a):
        align_cls = "center" if a == "center" else ("right" if a == "right" else "")
        # Inline style override (color scaling) takes priority over pos/neg CSS classes
        cell_style = (inline_styles[row_i][col_i]
                      if inline_styles and row_i < len(inline_styles)
                      and col_i < len(inline_styles[row_i])
                      else "")
        if cell_style:
            cls = f' class="{align_cls}"' if align_cls else ""
            return f'<td{cls} style="{cell_style}">{_esc(val)}</td>'
        cls_parts = [align_cls] if align_cls else []
        if col_i in pos_cols:
            cls_parts.append("pos-col")
        if col_i in neg_cols:
            cls_parts.append("neg-col")
        cls = f' class="{" ".join(cls_parts)}"' if cls_parts else ""
        return f'<td{cls}>{_esc(val)}</td>'

    head = "".join(_th(h, a) for h, a in zip(headers, alignments))
    body_rows = []
    for i, row in enumerate(rows):
        css_class = row_classes.get(i, "")
        tr_cls = f' class="{css_class}"' if css_class else ""
        cells = "".join(_td(v, i, j, alignments[j]) for j, v in enumerate(row))
        body_rows.append(f'<tr{tr_cls}>{cells}</tr>')

    return (
        f'<table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
    )


def _award_card(icon: str, title: str, team: str, detail: str, color: str) -> str:
    return (
        f'<div class="card no-break" style="border-color:{color};text-align:center;padding:8px 5px;">'
        f'<div style="font-size:1.4em;margin-bottom:3px;">{icon}</div>'
        f'<div style="color:#9ca3af;font-size:9px;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:3px;">{_esc(title)}</div>'
        f'<div style="color:{color};font-size:13px;font-weight:700;margin-bottom:2px;">{_esc(team)}</div>'
        f'<div style="color:#888;font-size:10px;">{_esc(detail)}</div>'
        f'</div>'
    )


def _mini_card(label: str, value: str, detail: str, color: str) -> str:
    return (
        f'<div class="card no-break" style="border-color:{color};padding:10px 12px;">'
        f'<div style="color:#9ca3af;font-size:10px;text-transform:uppercase;'
        f'letter-spacing:1px;margin-bottom:4px;">{_esc(label)}</div>'
        f'<div style="color:{color};font-size:17px;font-weight:800;margin-bottom:3px;">{_esc(value)}</div>'
        f'<div style="color:#888;font-size:11px;">{_esc(detail)}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Section 0 — Cover
# ---------------------------------------------------------------------------

def _build_cover(league_name: str) -> str:
    return f"""
<div class="no-break" style="
    text-align:center;
    background:linear-gradient(135deg,#0d0d1a 0%,#1e0e3f 45%,#2d1b69 75%,#1a1a2e 100%);
    border:2px solid {_PURPLE};border-radius:14px;
    padding:40px 30px 36px;margin-bottom:28px;">
  <div style="font-size:1em;color:#9ca3af;letter-spacing:4px;
              text-transform:uppercase;margin-bottom:14px;">
    Fantasy Premier League - Draft &nbsp;·&nbsp; 2025/26
  </div>
  <div style="font-size:3em;font-weight:900;color:{_GOLD};
              letter-spacing:1px;line-height:1.1;margin-bottom:12px;">
    {_esc(league_name)}
  </div>
  <div style="font-size:2em;color:#ffffff;font-weight:700;
              letter-spacing:4px;text-transform:uppercase;margin-bottom:20px;">
    Season Wrapped
  </div>
  <div style="width:72px;height:2px;
              background:linear-gradient(90deg,transparent,{_PURPLE},transparent);
              margin:0 auto 18px;"></div>
  <div style="color:#b0b8cc;font-size:1.05em;letter-spacing:0.5px;">
    End-of-season review &nbsp;·&nbsp; Stats &nbsp;·&nbsp; Stories &nbsp;·&nbsp; Awards
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Section 1 — League Champion
# ---------------------------------------------------------------------------

def _build_champion(league_data: dict) -> str:
    standings = league_data.get("standings", [])
    entries   = league_data.get("league_entries", [])
    if not standings or not entries:
        return "<p>No standings data.</p>"

    entry_names = {e["id"]: e["entry_name"] for e in entries}
    sorted_s    = sorted(standings, key=lambda s: s.get("rank", 999))

    winner = sorted_s[0]
    wname  = entry_names.get(winner.get("league_entry"), "?")
    ww, wd, wl = winner.get("matches_won",0), winner.get("matches_drawn",0), winner.get("matches_lost",0)
    wpts   = winner.get("points_for", 0)
    wlp    = ww * 3 + wd

    parts = []

    # Champion hero banner — compact to keep standings on same page
    parts.append(f"""
<div class="no-break" style="
    background:linear-gradient(135deg,#1a1a2e 0%,#2d1b69 50%,#1a1a2e 100%);
    border:2px solid {_GOLD};border-radius:14px;padding:18px 24px;
    text-align:center;margin-bottom:10px;">
  <div style="font-size:2.2em;margin-bottom:4px;">🏆</div>
  <div style="color:{_GOLD};font-size:2em;font-weight:800;margin-bottom:3px;">{_esc(wname)}</div>
  <div style="color:#e0e0e0;font-size:1em;margin-bottom:8px;">2025/26 League Champion</div>
  <div style="color:#9ca3af;font-size:0.9em;">
    {ww}W – {wd}D – {wl}L &nbsp;|&nbsp; {wlp} league pts &nbsp;|&nbsp; {wpts:,} FPL pts
  </div>
</div>
""")

    # Runner-up | 3rd place | Relegated — three-column row
    n_teams = len(sorted_s)
    if n_teams >= 2:
        ru = sorted_s[1]
        ru_name = entry_names.get(ru.get("league_entry"), "?")
        ru_w, ru_d, ru_l = ru.get("matches_won",0), ru.get("matches_drawn",0), ru.get("matches_lost",0)
        ru_lp = ru_w * 3 + ru_d

        ru_html = f"""
<div class="card no-break" style="flex:1;border-color:#9d4edd;text-align:center;padding:12px 10px;">
  <div style="font-size:1.5em;margin-bottom:4px;">🥈</div>
  <div style="color:#9ca3af;font-size:11px;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:4px;">2nd Place</div>
  <div style="color:#9d4edd;font-size:15px;font-weight:700;margin-bottom:3px;">{_esc(ru_name)}</div>
  <div style="color:#888;font-size:11px;">{ru_w}W–{ru_d}D–{ru_l}L &nbsp;·&nbsp; {ru_lp} pts</div>
</div>"""

        third_html = ""
        if n_teams >= 3:
            th = sorted_s[2]
            th_name = entry_names.get(th.get("league_entry"), "?")
            th_w, th_d, th_l = th.get("matches_won",0), th.get("matches_drawn",0), th.get("matches_lost",0)
            th_lp = th_w * 3 + th_d
            third_html = f"""
<div class="card no-break" style="flex:1;border-color:#f59e0b;text-align:center;padding:12px 10px;
    background:linear-gradient(135deg,#fffbeb,#fef3c7);">
  <div style="font-size:1.5em;margin-bottom:4px;">🥉</div>
  <div style="color:#92400e;font-size:11px;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:4px;">3rd Place</div>
  <div style="color:#b45309;font-size:15px;font-weight:700;margin-bottom:3px;">{_esc(th_name)}</div>
  <div style="color:#78350f;font-size:11px;">{th_w}W–{th_d}D–{th_l}L &nbsp;·&nbsp; {th_lp} pts</div>
</div>"""

        rel_html = ""
        if n_teams >= 2:
            rel = sorted_s[-1]
            rel_name = entry_names.get(rel.get("league_entry"), "?")
            rw, rd, rl = rel.get("matches_won",0), rel.get("matches_drawn",0), rel.get("matches_lost",0)
            rlp = rw * 3 + rd
            rel_html = f"""
<div class="card-red card no-break" style="flex:1;text-align:center;padding:12px 10px;">
  <div style="font-size:1.5em;margin-bottom:4px;">❌</div>
  <div style="color:{_RED};font-size:11px;font-weight:900;
              text-transform:uppercase;letter-spacing:2px;margin-bottom:4px;">Relegated</div>
  <div style="color:#dc2626;font-size:15px;font-weight:700;margin-bottom:3px;">{_esc(rel_name)}</div>
  <div style="color:#991b1b;font-size:11px;">
    {rw}W–{rd}D–{rl}L &nbsp;·&nbsp; {rlp} pts &nbsp;·&nbsp; <em>Removed 2026/27</em>
  </div>
</div>"""

        parts.append(
            f'<div style="display:flex;gap:10px;margin-bottom:10px;">'
            f'{ru_html}{third_html}{rel_html}'
            f'</div>'
        )

    # Standings table — identify rows to highlight + collect numeric values for color scaling
    rows = []
    highlight_rows: Dict[int, str] = {}
    lp_vals, pf_vals, pa_vals, diff_vals = [], [], [], []
    for i, row in enumerate(sorted_s):
        eid  = row.get("league_entry")
        name = entry_names.get(eid, "?")
        rw   = row.get("matches_won", 0)
        rd   = row.get("matches_drawn", 0)
        rl   = row.get("matches_lost", 0)
        pf   = row.get("points_for", 0)
        pa   = row.get("points_against", 0)
        lp   = rw * 3 + rd
        diff = pf - pa
        rank = row.get("rank", i + 1)
        if rank == 1:
            highlight_rows[i] = "champion"
        elif rank == 2:
            highlight_rows[i] = "runner-up"
        elif rank == 3:
            highlight_rows[i] = "third-place"
        elif i == len(sorted_s) - 1:
            highlight_rows[i] = "relegated"
        lp_vals.append(lp); pf_vals.append(pf); pa_vals.append(pa); diff_vals.append(diff)
        rows.append([str(rank), name, str(rw), str(rd), str(rl),
                     str(lp), f"{pf:,}", f"{pa:,}", f"{diff:+,}"])

    # Build inline color styles: cols 5=LeaguePts, 6=PtsFor, 7=PtsAgainst, 8=PtsDiff
    n_cols = 9
    inline_styles = [[""] * n_cols for _ in range(len(rows))]
    for i in range(len(rows)):
        inline_styles[i][5] = _color_scale_value(lp_vals[i],   lp_vals,   "positive")
        inline_styles[i][6] = _color_scale_value(pf_vals[i],   pf_vals,   "positive")
        inline_styles[i][7] = _color_scale_value(pa_vals[i],   pa_vals,   "negative")
        inline_styles[i][8] = _color_scale_value(diff_vals[i], diff_vals, "diverging")

    table_html = _table(
        ["Rank", "Team", "W", "D", "L", "League Pts", "Pts For", "Pts Against", "Pts Diff"],
        rows,
        alignments=["center", "left", "center", "center", "center",
                    "right", "right", "right", "right"],
        highlight_rows=highlight_rows,
        inline_styles=inline_styles,
    )
    parts.append(f"""
<div class="no-break" style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;margin-top:2px;">
  <div style="background:linear-gradient(135deg,#37003c,#5a0060);color:{_GREEN};
              font-weight:700;font-size:1em;padding:8px 14px;">Final League Standings</div>
  {table_html}
</div>
""")

    return '<div class="section">' + "".join(parts) + "</div>"


# ---------------------------------------------------------------------------
# Section 2 — Season Journey
# ---------------------------------------------------------------------------

def _build_season_journey(history_df: pd.DataFrame) -> str:
    if history_df is None or history_df.empty:
        return _section("📈", "Season Journey", page_break=True) + "<p>No data.</p></div>"

    teams    = sorted(history_df["Team"].unique().tolist())
    last_gw  = history_df["Gameweek"].max()
    final    = history_df[history_df["Gameweek"] == last_gw][["Team", "League_Position"]]
    champion = final.loc[final["League_Position"].idxmin(), "Team"] if not final.empty else None

    color_map = {t: _TEAM_COLORS[i % len(_TEAM_COLORS)] for i, t in enumerate(teams)}
    if champion:
        color_map[champion] = _GOLD

    layout_base = dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#f9fafb",
        font=dict(color="#1a1a2e", size=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1a1a2e", size=11)),
    )

    # Cumulative points
    fig1 = go.Figure()
    for team in teams:
        tdf      = history_df[history_df["Team"] == team].sort_values("Gameweek")
        is_champ = team == champion
        fig1.add_trace(go.Scatter(
            x=tdf["Gameweek"], y=tdf["Total_Points"], name=team, mode="lines",
            line=dict(color=color_map[team], width=3 if is_champ else 1.5),
        ))
    fig1.update_layout(
        **layout_base,
        title=dict(text="📈 Cumulative Points Race", font=dict(size=16, color="#1a1a2e"),
                   x=0.5, xanchor="center"),
    )
    fig1.update_xaxes(title="Gameweek", dtick=2, gridcolor="#e5e7eb")
    fig1.update_yaxes(title="Total FPL Points", gridcolor="#e5e7eb")

    # League position
    fig2 = go.Figure()
    n = len(teams)
    for team in teams:
        tdf      = history_df[history_df["Team"] == team].sort_values("Gameweek")
        is_champ = team == champion
        fig2.add_trace(go.Scatter(
            x=tdf["Gameweek"], y=tdf["League_Position"], name=team, mode="lines",
            line=dict(color=color_map[team], width=3 if is_champ else 1.5),
        ))
    fig2.update_layout(
        **layout_base,
        title=dict(text="🏅 League Position Timeline", font=dict(size=16, color="#1a1a2e"),
                   x=0.5, xanchor="center"),
    )
    fig2.update_xaxes(title="Gameweek", dtick=2, gridcolor="#e5e7eb")
    fig2.update_yaxes(
        title="League Position", autorange="reversed",
        dtick=1, range=[n + 0.5, 0.5], gridcolor="#e5e7eb",
    )

    return (
        _section("📈", "Season Journey",
                 "Cumulative points race and league position over the season",
                 page_break=True)
        + f'<div class="no-break">{_chart(fig1, include_js=True)}</div>'
        + f'<div class="no-break" style="margin-top:12px;">{_chart(fig2)}</div>'
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 3 — League Awards
# ---------------------------------------------------------------------------

def _build_awards(superlatives: Dict, history_df: pd.DataFrame) -> str:
    most_active  = superlatives.get("most_active",  {})
    best_mgr     = superlatives.get("best_mgr",     {})
    luckiest     = superlatives.get("luckiest",     {})
    unluckiest   = superlatives.get("unluckiest",   {})
    best_drafter = superlatives.get("best_drafter", {})

    pts_champ = high_gw = consistent = {"team": "?", "value": ""}
    played = history_df[history_df["GW_Points"] > 0] if history_df is not None and not history_df.empty else pd.DataFrame()
    if not played.empty:
        last_gw = played["Gameweek"].max()
        final   = played[played["Gameweek"] == last_gw][["Team", "Total_Points"]]
        if not final.empty:
            idx       = final["Total_Points"].idxmax()
            pts_champ = {"team": final.loc[idx, "Team"],
                         "value": f'{int(final.loc[idx, "Total_Points"]):,} total FPL pts'}
        h_idx  = played["GW_Points"].idxmax()
        high_gw = {
            "team":  played.loc[h_idx, "Team"],
            "value": f'{int(played.loc[h_idx, "GW_Points"])} pts in GW{int(played.loc[h_idx, "Gameweek"])}',
        }
        std_df = played.groupby("Team")["GW_Points"].agg(["std", "count"])
        std_df = std_df[std_df["count"] >= 5]
        if not std_df.empty:
            c_team    = std_df["std"].idxmin()
            consistent = {"team": c_team, "value": f'σ = {round(std_df.loc[c_team,"std"],1)} pts/GW'}

    awards = [
        ("🏆", "Points Champion",    pts_champ.get("team","?"),    pts_champ.get("value",""),   _GOLD),
        ("📊", "Highest Single GW",  high_gw.get("team","?"),      high_gw.get("value",""),     "#00b4d8"),
        ("🎯", "Most Consistent",    consistent.get("team","?"),   consistent.get("value",""),  _GREEN),
        ("🔀", "Most Active Mgr",    most_active.get("team","?"),
         f'{most_active.get("value",0)} transactions',             "#f8961e"),
        ("🧠", "Best Lineup Mgr",    best_mgr.get("team","?"),     best_mgr.get("value",""),    _PURPLE),
        ("✏️", "Best Drafter",       best_drafter.get("team","?"),best_drafter.get("value",""),"#9d4edd"),
        ("🍀", "Luckiest Manager",   luckiest.get("team","?"),     luckiest.get("value",""),    "#43aa8b"),
        ("😤", "Most Unlucky",       unluckiest.get("team","?"),   unluckiest.get("value",""),  _RED),
    ]

    cards_html = "".join(_award_card(ic, ti, te, de, co) for ic, ti, te, de, co in awards)
    return (
        _section("🎖️", "League Awards",
                 "Eight superlatives celebrating the best (and worst) of the season",
                 page_break=True)
        + f'<div class="grid-4">{cards_html}</div>'
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 4 — GW Highlights
# ---------------------------------------------------------------------------

def _build_highlights(highlights: Dict) -> str:
    high   = highlights.get("highest_gw",      {})
    low    = highlights.get("lowest_gw",       {})
    blowout= highlights.get("biggest_blowout", {})
    swing  = highlights.get("biggest_rank_swing", {})

    cards = []
    if high:
        cards.append(_mini_card(
            "🔥 Highest GW Score",
            f'{high.get("score","?")} pts',
            f'{high.get("team","?")} — GW{high.get("gw","?")}', _GOLD))
    if low:
        cards.append(_mini_card(
            "🥶 Lowest GW Score",
            f'{low.get("score","?")} pts',
            f'{low.get("team","?")} — GW{low.get("gw","?")}', _RED))
    if blowout:
        cards.append(_mini_card(
            "💥 Biggest Blowout",
            f'{blowout.get("score1","?")} – {blowout.get("score2","?")}',
            f'{blowout.get("winner","?")} def. {blowout.get("loser","?")} '
            f'by {blowout.get("margin","?")} pts (GW{blowout.get("gw","?")})',
            "#00b4d8"))
    if swing:
        cards.append(_mini_card(
            f'{swing.get("direction","📊")} Biggest Rank Swing',
            f'{swing.get("swing","?")} places',
            f'{swing.get("team","?")} — #{swing.get("from_rank","?")} → '
            f'#{swing.get("to_rank","?")} in GW{swing.get("gw","?")}',
            "#9d4edd"))

    if not cards:
        return ""

    return (
        _section("⚡", "Gameweek Highlights", "The most memorable moments of the season",
                 extra_style="margin-top:20px;")
        + f'<div class="grid-2">{"".join(cards)}</div>'
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 5 — H2H Records
# ---------------------------------------------------------------------------

def _build_h2h(league_data: dict) -> str:
    try:
        team_names = get_team_names(league_data)
        matches_df = get_matches_df(league_data, team_names)
        if matches_df.empty:
            return ""
        h2h = build_h2h_matrix(matches_df, team_names)
        if h2h.empty:
            return ""
    except Exception:
        return ""

    teams = h2h.index.tolist()
    n_teams = len(teams)

    # Fixed-layout: name col 88px, data cols distribute the rest equally
    NAME_COL_W = 88

    def _abbr(name: str, max_len: int = 9) -> str:
        return name if len(name) <= max_len else name[:max_len - 1] + "…"

    def _parse(cell):
        try:
            parts = str(cell).split("-")
            return int(parts[0]), int(parts[1]), int(parts[2])
        except Exception:
            return 0, 0, 0

    def _cell_style(cell):
        if cell == "-":
            return "background:#f3f4f6;color:#9ca3af;"
        w, d, l = _parse(cell)
        if w > l:
            return "background:rgba(0,168,85,0.15);color:#059669;font-weight:700;"
        if w < l:
            return f"background:rgba(255,75,75,0.12);color:{_RED};font-weight:600;"
        return "background:rgba(255,215,0,0.15);color:#b45309;"

    _TH_BASE = ("background:linear-gradient(135deg,#37003c,#5a0060);color:#00ff87;"
                "font-weight:600;font-size:10px;padding:6px 4px;"
                "border-bottom:2px solid #5a0060;text-align:center;"
                "overflow:hidden;white-space:nowrap;text-overflow:ellipsis;")

    # Header row: name column (fixed width) + one col per team (auto-distributed)
    header_row = f'<th style="{_TH_BASE}text-align:left;width:{NAME_COL_W}px;"></th>'
    for t in teams:
        header_row += f'<th style="{_TH_BASE}">{_esc(_abbr(t))}</th>'

    rows_html = ""
    for team in teams:
        cells = ""
        for opp in teams:
            val   = str(h2h.loc[team, opp]) if (team in h2h.index and opp in h2h.columns) else "-"
            style = _cell_style(val)
            cells += (f'<td style="padding:4px 3px;font-size:10px;text-align:center;'
                      f'border:1px solid #f0f0f0;{style}">{_esc(val)}</td>')
        rows_html += (
            f'<tr><td style="padding:4px 8px;font-size:10px;font-weight:600;'
            f'color:#1a1a2e;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'
            f'width:{NAME_COL_W}px;max-width:{NAME_COL_W}px;background:#f8f9fa;">'
            f'{_esc(team)}</td>{cells}</tr>'
        )

    matrix_html = (
        f'<div style="border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;">'
        f'<table style="width:100%;table-layout:fixed;border-collapse:collapse;background:#ffffff;">'
        f'<thead><tr>{header_row}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table></div>'
    )

    return (
        _section("⚔️", "Head-to-Head Records",
                 "Full W-D-L matrix across all league matchups",
                 extra_style="margin-top:28px;")
        + matrix_html
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 6 — Draft Board
# ---------------------------------------------------------------------------

def _build_draft_board(draft_data: Dict) -> str:
    steals = draft_data.get("steals", [])
    busts  = draft_data.get("busts",  [])

    def _pick_card(p, color):
        return (
            f'<div class="pick-card no-break" style="border:1px solid {color};">'
            f'<div><div class="name" style="color:{color};">{_esc(p.get("player","?"))}</div>'
            f'<div class="meta">{_esc(p.get("team","?"))} · Rd {p.get("round","?")} Pick {p.get("pick","?")}</div></div>'
            f'<div><div class="grade" style="color:{color};">{_esc(p.get("grade","?"))}</div>'
            f'<div class="pts">{p.get("pts","?")} pts · Δ{p.get("delta",0):+.0f}</div></div>'
            f'</div>'
        )

    steals_html = "".join(_pick_card(p, _GREEN) for p in steals) or '<p style="color:#888;">None found.</p>'
    busts_html  = "".join(_pick_card(p, _RED)   for p in busts)  or '<p style="color:#888;">None found.</p>'

    two_col = (
        f'<div class="two-col">'
        f'<div><div style="color:{_GREEN};font-weight:700;margin-bottom:8px;">🔥 Top Steals</div>{steals_html}</div>'
        f'<div><div style="color:{_RED};font-weight:700;margin-bottom:8px;">💀 Top Busts</div>{busts_html}</div>'
        f'</div>'
    )

    # Grade summary table
    team_grades = draft_data.get("team_grades", {})
    grade_table = ""
    if team_grades:
        grade_rows = sorted(
            [{"team": t, **g} for t, g in team_grades.items()],
            key=lambda r: r.get("Steal 🔥", 0), reverse=True,
        )
        grade_table = (
            '<div style="margin-top:16px;">'
            + _table(
                ["Team", "Steals 🔥", "Value ✅", "Fair ⚖️", "Miss 📉", "Busts 💀"],
                [[r["team"], r.get("Steal 🔥",0), r.get("Value ✅",0),
                  r.get("Fair ⚖️",0), r.get("Miss 📉",0), r.get("Bust 💀",0)]
                 for r in grade_rows],
                alignments=["left","right","right","right","right","right"],
                pos_cols=[1,2], neg_cols=[4,5],
            )
            + '</div>'
        )

    return (
        _section("🃏", "Draft Board Retrospective",
                 "League-wide draft steals and busts", page_break=True)
        + two_col + grade_table
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 7 — Transfer Window
# ---------------------------------------------------------------------------

def _build_transfer_window(transfer_data: Dict) -> str:
    per_team = transfer_data.get("per_team", {})
    best     = transfer_data.get("best_transfer",  {})
    worst    = transfer_data.get("worst_transfer", {})
    most_in  = transfer_data.get("most_in",  [])
    most_out = transfer_data.get("most_out", [])

    parts = []

    # Activity bar chart
    if per_team:
        sorted_teams = sorted(per_team.items(), key=lambda x: x[1], reverse=True)
        fig = go.Figure(go.Bar(
            x=[t for t, _ in sorted_teams],
            y=[c for _, c in sorted_teams],
            marker_color=_PURPLE,
        ))
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#f9fafb",
            font=dict(color="#1a1a2e", size=12),
            title=dict(text="Transfer Activity by Team", font=dict(size=15, color="#1a1a2e"),
                       x=0.5, xanchor="center"),
            yaxis=dict(gridcolor="#e5e7eb", range=[0, max(c for _, c in sorted_teams) * 1.2]),
            xaxis=dict(tickangle=45, gridcolor="#e5e7eb"),
        )
        # left_margin=40: no y-axis title, so 70 wastes space and shifts bars right
        # right_margin=55: angled last label extends right of the final bar
        # bottom_margin=90: room for the diagonal x-axis team name labels
        parts.append(f'<div class="no-break">{_chart(fig, height=320, bottom_margin=90, left_margin=40, right_margin=55)}</div>')

    # Best / worst transfer callouts
    if best or worst:
        best_html = worst_html = ""
        if best:
            best_html = (
                f'<div class="card no-break" style="border-color:{_GREEN};">'
                f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:6px;">Best Transfer In</div>'
                f'<div style="color:{_GREEN};font-size:16px;font-weight:700;margin-bottom:3px;">'
                f'{_esc(best.get("player_in","?"))}</div>'
                f'<div style="color:#888;font-size:12px;">picked up for {_esc(best.get("player_out","?"))}</div>'
                f'<div style="color:#9ca3af;font-size:12px;margin-top:4px;">'
                f'{_esc(best.get("team","?"))} — +{best.get("net","?")} net pts</div>'
                f'</div>'
            )
        if worst:
            worst_html = (
                f'<div class="card no-break" style="border-color:{_RED};">'
                f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:1px;margin-bottom:6px;">Worst Transfer Out</div>'
                f'<div style="color:{_RED};font-size:16px;font-weight:700;margin-bottom:3px;">'
                f'{_esc(worst.get("player_out","?"))}</div>'
                f'<div style="color:#888;font-size:12px;">dropped for {_esc(worst.get("player_in","?"))}</div>'
                f'<div style="color:#9ca3af;font-size:12px;margin-top:4px;">'
                f'{_esc(worst.get("team","?"))} — scored {worst.get("pts_out_after","?")} pts after being dropped</div>'
                f'</div>'
            )
        parts.append(f'<div class="two-col" style="margin-top:12px;">{best_html}{worst_html}</div>')

    # Most transferred in / out (items are (name, count) tuples)
    if most_in or most_out:
        def _player_rows(players):
            return [[str(p[0]), str(p[1])] for p in players[:5]]

        in_table  = _table(["Most Transferred In",  "Times"], _player_rows(most_in),
                           alignments=["left","right"]) if most_in else ""
        out_table = _table(["Most Transferred Out", "Times"], _player_rows(most_out),
                           alignments=["left","right"]) if most_out else ""
        parts.append(
            f'<div class="two-col no-break" style="margin-top:12px;">'
            f'<div>{in_table}</div><div>{out_table}</div></div>'
        )

    return (
        _section("🔀", "Transfer Window",
                 "Who was most active and who won (or lost) the transfer game",
                 page_break=True)
        + "".join(parts)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Section 8 — Lineup Management
# ---------------------------------------------------------------------------

def _build_lineup_management(bench_data_list: List[Dict]) -> str:
    if not bench_data_list:
        return ""

    df = pd.DataFrame(bench_data_list)

    parts = []

    totals_cols = [c for c in ["Team","Pts Scored","Pts Possible","Total Pts Lost"] if c in df.columns]
    if len(totals_cols) >= 3:
        df_t = df[totals_cols].copy()
        if "Pts Scored" in df_t.columns and "Pts Possible" in df_t.columns:
            df_t["Efficiency %"] = (df_t["Pts Scored"] / df_t["Pts Possible"] * 100).round(1).astype(str) + "%"
        rows = [list(row) for _, row in df_t.iterrows()]
        parts.append(
            '<div class="no-break">'
            + _table(list(df_t.columns), rows,
                     alignments=["left"] + ["right"] * (len(df_t.columns) - 1),
                     pos_cols=[1, len(df_t.columns) - 1],
                     neg_cols=[3] if len(df_t.columns) > 3 else [])
            + '</div>'
        )

    detail_cols = [c for c in ["Team","Avg Lost/GW","Selection %","Bench Mgmt Score",
                                "Avg Bench/GW","Worst GW"] if c in df.columns]
    if detail_cols:
        df_d = df[detail_cols].copy()
        for col in ["Avg Lost/GW","Avg Bench/GW","Bench Mgmt Score"]:
            if col in df_d.columns:
                df_d[col] = df_d[col].round(1)
        if "Selection %" in df_d.columns:
            df_d["Selection %"] = df_d["Selection %"].round(1).astype(str) + "%"
        rows = [list(row) for _, row in df_d.iterrows()]
        parts.append(
            '<div class="no-break" style="margin-top:14px;">'
            + _table(list(df_d.columns), rows,
                     alignments=["left"] + ["right"] * (len(df_d.columns) - 1))
            + '</div>'
        )

    return (
        _section("🧩", "Lineup Management",
                 "Bench points missed — who managed their squad best?", page_break=True)
        + "".join(parts)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Assemble full HTML document
# ---------------------------------------------------------------------------

def _build_html(league_data: dict, history_df: pd.DataFrame,
                bench_data_list: List[Dict], superlatives: Dict,
                highlights: Dict, draft_data: Dict,
                transfer_data: Dict) -> str:

    league_name = league_data.get("league", {}).get("name", "FPL Draft League")

    sections = [
        _build_cover(league_name),
        _build_champion(league_data),
        _build_season_journey(history_df),
        _build_awards(superlatives, history_df),
        _build_highlights(highlights),
        _build_h2h(league_data),
        _build_draft_board(draft_data),
        _build_transfer_window(transfer_data),
        _build_lineup_management(bench_data_list),
    ]

    body = "\n".join(s for s in sections if s)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_esc(league_name)} — Season Wrapped 2025/26</title>
  <style>{_CSS}</style>
</head>
<body>{body}</body>
</html>"""


# ---------------------------------------------------------------------------
# Playwright renderer
# ---------------------------------------------------------------------------

def _render_with_playwright(html: str) -> bytes:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page    = browser.new_page()
        page.set_viewport_size({"width": 794, "height": 1123})  # A4 at 96 dpi
        page.set_content(html, wait_until="networkidle", timeout=60_000)
        pdf = page.pdf(
            format="A4",
            print_background=True,
            margin={"top": "1.5cm", "bottom": "1.5cm",
                    "left": "1.2cm",  "right": "1.2cm"},
        )
        browser.close()

    return pdf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_league_wrapped_pdf(
    league_data:     dict,
    history_df:      pd.DataFrame,
    bench_data_list: List[Dict],
    superlatives:    Dict,
    highlights:      Dict,
    draft_data:      Dict,
    transfer_data:   Dict,
) -> bytes:
    """Build and return a League Wrapped PDF as bytes."""
    html = _build_html(
        league_data, history_df, bench_data_list,
        superlatives, highlights, draft_data, transfer_data,
    )
    return _render_with_playwright(html)
