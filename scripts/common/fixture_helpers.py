"""
Fixture & Scheduling Utilities.

Fixture difficulty grid, kickoff times, deadline calculations, and FDR styling.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

import config
from scripts.common.error_helpers import get_logger
from scripts.common.text_helpers import TZ_ET

_logger = get_logger("fpl_app.fixture_helpers")


def _bootstrap_teams_df() -> pd.DataFrame:
    """Return FPL bootstrap teams as a 2-col DF: id, short_name."""
    try:
        resp = requests.get("https://draft.premierleague.com/api/bootstrap-static", timeout=20)
        resp.raise_for_status()
        teams = resp.json().get("teams", [])
        return pd.DataFrame(teams)[["id", "short_name"]]
    except Exception:
        _logger.warning("Failed to fetch bootstrap teams data", exc_info=True)
        # Fallback: empty DF (the normalizer will still attempt heuristics)
        return pd.DataFrame(columns=["id", "short_name"])


def get_earliest_kickoff_et(gw: int) -> datetime:
    """
    Pull fixtures for the given GW from the classic FPL endpoint and
    return the earliest kickoff in ET.
    """
    r = requests.get(config.FPL_FIXTURES_BY_EVENT.format(gw=gw), timeout=20)
    r.raise_for_status()
    fixtures = r.json()
    # Filter fixtures that actually have a kickoff_time
    times = []
    for fx in fixtures:
        k = fx.get("kickoff_time")
        if not k:
            continue
        # k is ISO string in UTC, e.g., "2024-12-03T19:30:00Z"
        # Normalize 'Z' to '+00:00' for fromisoformat
        k2 = k.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(k2)
        times.append(dt_utc.astimezone(TZ_ET))
    if not times:
        raise RuntimeError(f"No kickoff times found for GW {gw}")
    return min(times)


def get_fixture_difficulty_grid(weeks: int = 6):
    """
    Returns:
      disp  : display DF with first col 'Team', then GW columns (strings like 'WHU (H)')
      diffs : numeric DF (index=team short, cols=GW) with difficulty (1..5, avg if DGW)
      avg   : Series of per-team average difficulty across horizon (NaN->3 neutral)
    """
    from scripts.common.fpl_draft_api import get_current_gameweek

    current_gw = int(get_current_gameweek())

    teams = _bootstrap_teams_df()  # id, short_name
    id2short = {int(r.id): str(r.short_name) for _, r in teams.iterrows()}

    cols = [f"GW{gw}" for gw in range(current_gw, current_gw + weeks)]
    idx = [id2short[i] for i in sorted(id2short)]
    disp_core = pd.DataFrame("—", index=idx, columns=cols)
    diffs = pd.DataFrame(np.nan, index=idx, columns=cols)

    def _fixtures_for_event(gw: int):
        """
        Return fixtures for a single gameweek `gw` from the canonical FPL endpoint.
        Uses explicit query params so the server actually filters by GW.
        """
        url = "https://fantasy.premierleague.com/api/fixtures/"
        headers = {"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
        try:
            r = requests.get(url, params={"event": int(gw)}, headers=headers, timeout=20)
            r.raise_for_status()
            js = r.json()
            # This endpoint returns only the requested GW as a list
            return js if isinstance(js, list) else []
        except Exception:
            return []

    # Fill each team's own schedule per GW (no carry-over across columns)
    for gw, col in zip(range(current_gw, current_gw + weeks), cols):
        fx = _fixtures_for_event(gw)
        for f in fx:
            h, a = f.get("team_h"), f.get("team_a")
            dh, da = f.get("team_h_difficulty"), f.get("team_a_difficulty")
            if h is None or a is None:
                continue
            hs, as_ = id2short.get(int(h), str(h)), id2short.get(int(a), str(a))

            # Home team cell
            prev_h = disp_core.at[hs, col]
            disp_core.at[hs, col] = f"{as_} (H)" if prev_h == "—" else f"{prev_h} / {as_} (H)"
            diffs.at[hs, col] = np.nanmean([diffs.at[hs, col], float(dh) if dh is not None else np.nan])

            # Away team cell
            prev_a = disp_core.at[as_, col]
            disp_core.at[as_, col] = f"{hs} (A)" if prev_a == "—" else f"{prev_a} / {hs} (A)"
            diffs.at[as_, col] = np.nanmean([diffs.at[as_, col], float(da) if da is not None else np.nan])

    # Sort rows by easiest average run (NaN -> neutral 3)
    avg = diffs.fillna(3).mean(axis=1)
    order = avg.sort_values().index
    disp_core = disp_core.loc[order]
    diffs = diffs.loc[order]
    avg = avg.loc[order]

    # Add sticky Team column for Y-axis labels
    disp = disp_core.copy()
    disp.insert(0, "Team", disp.index)
    disp["Avg FDR"] = avg.round(1)
    return disp, diffs, avg


def get_next_transaction_deadline(offset_hours: int, gw: int):
    """
    Returns (deadline_et, gameweek) where deadline = earliest kickoff - offset_hours.
    Uses ET, leveraging your get_earliest_kickoff_et(gw).
    """
    from scripts.common.fpl_draft_api import get_current_gameweek

    if offset_hours is None:
        offset_hours = getattr(config, "TRANSACTION_DEADLINE_HOURS_BEFORE_KICKOFF", 24)
    if gw is None:
        gw = int(get_current_gameweek())
    kickoff_et = get_earliest_kickoff_et(gw)
    return kickoff_et - timedelta(hours=offset_hours), gw


def style_fixture_difficulty(disp: pd.DataFrame, diffs: pd.DataFrame) -> str:
    """
    Build a dark-themed HTML table for the FDR grid.

    Returns an HTML string (not a Styler) to render via st.markdown.
    'Team' column acts as row labels; GW cells are colored by difficulty.
    """
    # FDR palette: 1=easy(green) -> 5=hard(red), high-contrast dark-theme
    # Key: backgrounds must be clearly distinguishable between adjacent levels
    PALETTE = {
        1: ("#047857", "#ecfdf5"),  # bright emerald bg, near-white text
        2: ("#1a3a2e", "#86efac"),  # dark muted green bg, green text
        3: ("#2a2a3e", "#9ca3af"),  # neutral dark bg, grey text
        4: ("#5f2121", "#fecaca"),  # muted dark red bg, pink text
        5: ("#b91c1c", "#fef2f2"),  # bright red bg, near-white text
    }

    def _fdr_text_color(val: float) -> str:
        """Continuous color interpolation for Avg FDR text (1.0=green → 5.0=red).

        Uses a power curve to concentrate color change around 3.0 so that
        values in the typical 2.5-3.3 range show visible differentiation
        instead of all mapping to yellow.
        """
        import math
        v = max(1.0, min(5.0, val))
        # Center at 3.0, normalize to [-1, 1]
        norm = (v - 3.0) / 2.0
        # Power curve (exp < 1) stretches values away from center
        sign = 1 if norm >= 0 else -1
        expanded = sign * (abs(norm) ** 0.55)
        # Map back to [0, 1] for color interpolation
        t = max(0.0, min(1.0, (expanded + 1) / 2))
        if t <= 0.5:
            # Green (#4ade80) to Yellow (#facc15)
            s = t / 0.5
            r = int(74 + (250 - 74) * s)
            g = int(222 + (204 - 222) * s)
            b = int(128 + (21 - 128) * s)
        else:
            # Yellow (#facc15) to Red (#ef4444)
            s = (t - 0.5) / 0.5
            r = int(250 + (239 - 250) * s)
            g = int(204 + (68 - 204) * s)
            b = int(21 + (68 - 21) * s)
        return f"rgb({r},{g},{b})"
    gw_cols = [c for c in disp.columns if c not in ("Team", "Avg FDR")]

    # Header style
    th_style = (
        "background:linear-gradient(135deg,#37003c,#5a0060);color:#00ff87;"
        "font-weight:600;font-size:13px;padding:10px 12px;border-bottom:2px solid #00ff87;"
        "text-align:center;position:sticky;top:0;z-index:1;"
    )
    team_th_style = th_style.replace("text-align:center", "text-align:left")

    parts = [
        '<div style="border:1px solid #333;border-radius:10px;overflow:hidden;margin-bottom:1rem;">',
        '<table style="width:100%;border-collapse:collapse;font-size:14px;background:#1a1a2e;">',
        "<thead><tr>",
    ]

    for col in disp.columns:
        s = team_th_style if col == "Team" else th_style
        parts.append(f'<th style="{s}">{col}</th>')
    parts.append("</tr></thead><tbody>")

    for row_idx, (idx, row) in enumerate(disp.iterrows()):
        row_bg = "background:rgba(255,255,255,0.03);" if row_idx % 2 == 1 else "background:#1a1a2e;"
        parts.append(f'<tr style="{row_bg}">')
        for col in disp.columns:
            val = row[col]
            if col == "Team":
                parts.append(
                    f'<td style="padding:8px 12px;color:#00ff87;font-weight:700;'
                    f'border-bottom:1px solid #333;white-space:nowrap;">{val}</td>'
                )
            elif col == "Avg FDR":
                # Text-only coloring with continuous interpolation (green→yellow→red)
                fdr_val = float(val) if pd.notna(val) else 3.0
                txt = _fdr_text_color(fdr_val)
                parts.append(
                    f'<td style="padding:8px 12px;color:{txt};'
                    f'font-weight:800;font-size:16px;text-align:center;border-bottom:1px solid #333;">'
                    f'{fdr_val:.1f}</td>'
                )
            else:
                d = diffs.at[idx, col] if col in diffs.columns else np.nan
                k = 3 if pd.isna(d) else max(1, min(5, int(round(float(d)))))
                bg, txt = PALETTE[k]
                parts.append(
                    f'<td style="padding:8px 12px;background:{bg};color:{txt};'
                    f'font-weight:600;text-align:center;border-bottom:1px solid #333;'
                    f'white-space:nowrap;">{val}</td>'
                )
        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return "".join(parts)


def compute_key_differentials(
    team1_players: pd.DataFrame,
    team2_players: pd.DataFrame,
    team1_name: str,
    team2_name: str,
    points_col: str = "Points",
) -> tuple:
    """Find players unique to one squad but not the other.

    Works with both Draft format (Player as index, cols: Team, Position,
    Matchup, Points) and Classic format (Player as column, cols: Team,
    Position, Matchup, Points, squad_position).

    Returns (team1_diffs, team2_diffs) — each a list of dicts sorted by
    projected points descending:
        {"player": str, "epl_team": str, "position": str,
         "points": float, "matchup": str}
    """
    from scripts.common.player_matching import canonical_normalize

    if team1_players.empty or team2_players.empty:
        return [], []

    def _extract_names(df):
        if "Player" in df.columns:
            return set(canonical_normalize(n) for n in df["Player"].tolist())
        return set(canonical_normalize(n) for n in df.index.tolist())

    def _get_player_col(df, col, player_name):
        """Get a column value for a player, handling both index and column formats."""
        if "Player" in df.columns:
            mask = df["Player"].apply(canonical_normalize) == canonical_normalize(player_name)
            rows = df[mask]
        else:
            norm = canonical_normalize(player_name)
            rows = df[df.index.map(canonical_normalize) == norm]
        if rows.empty:
            return None
        return rows.iloc[0].get(col, "")

    def _iter_players(df):
        """Yield (player_name, row_series) for each player."""
        if "Player" in df.columns:
            for _, row in df.iterrows():
                yield row["Player"], row
        else:
            for name, row in df.iterrows():
                yield name, row

    names1 = _extract_names(team1_players)
    names2 = _extract_names(team2_players)

    shared = names1 & names2
    pcol = points_col if points_col in team1_players.columns else "Points"

    def _build_diffs(df, shared_names, pcol_name):
        diffs = []
        for player_name, row in _iter_players(df):
            if canonical_normalize(player_name) in shared_names:
                continue
            pts = float(row.get(pcol_name, 0) or 0)
            diffs.append({
                "player": player_name,
                "epl_team": row.get("Team", ""),
                "position": row.get("Position", ""),
                "points": pts,
                "matchup": row.get("Matchup", ""),
            })
        diffs.sort(key=lambda d: d["points"], reverse=True)
        return diffs

    # For Classic, use the correct points column from each DF
    pcol2 = points_col if points_col in team2_players.columns else "Points"
    team1_diffs = _build_diffs(team1_players, shared, pcol)
    team2_diffs = _build_diffs(team2_players, shared, pcol2)

    return team1_diffs, team2_diffs


def render_key_differentials(
    team1_diffs: list,
    team2_diffs: list,
    team1_name: str,
    team2_name: str,
    captain_info: dict = None,
    is_draft: bool = False,
):
    """Render the Key Differentials section.

    Parameters
    ----------
    captain_info : dict, optional (Classic only)
        Keys: team1_captain, team1_captain_pts, team1_multiplier,
              team2_captain, team2_captain_pts, team2_multiplier,
              is_predicted (bool)
    is_draft : bool
        If True, cap at top 5 per team with explanatory caption.
    """
    st.subheader("Key Differentials")

    # Captain comparison bar (Classic only)
    if captain_info:
        t1_cap = captain_info.get("team1_captain", "Unknown")
        t1_pts = captain_info.get("team1_captain_pts", 0)
        t1_mult = captain_info.get("team1_multiplier", 2)
        t2_cap = captain_info.get("team2_captain", "Unknown")
        t2_pts = captain_info.get("team2_captain_pts", 0)
        t2_mult = captain_info.get("team2_multiplier", 2)
        pred_suffix = " (Predicted)" if captain_info.get("is_predicted") else ""
        t1_eff = t1_pts * t1_mult
        t2_eff = t2_pts * t2_mult

        cap_parts = [
            '<div style="background:linear-gradient(135deg,#37003c 0%,#5a0060 100%);'
            'padding:16px;border-radius:10px;margin-bottom:16px;color:#e0e0e0;">',
            f'<div style="text-align:center;font-size:13px;font-weight:600;'
            f'color:#00ff87;margin-bottom:12px;">Captain Comparison{pred_suffix}</div>',
            '<div style="display:flex;justify-content:space-between;align-items:center;">',
        ]
        for name, cap, eff, pts, mult in [
            (team1_name, t1_cap, t1_eff, t1_pts, t1_mult),
            (team2_name, t2_cap, t2_eff, t2_pts, t2_mult),
        ]:
            if name == team2_name:
                cap_parts.append(
                    '<div style="font-size:14px;color:rgba(255,255,255,0.4);padding:0 12px;">vs</div>'
                )
            cap_parts.append(
                f'<div style="text-align:center;flex:1;color:#e0e0e0;">'
                f'<div style="font-size:13px;color:rgba(255,255,255,0.7);">{name}</div>'
                f'<div style="font-size:16px;font-weight:700;color:white;">{cap}</div>'
                f'<div style="font-size:22px;font-weight:700;color:#00ff87;">{eff:.1f}</div>'
                f'<div style="font-size:11px;color:rgba(255,255,255,0.5);">'
                f'{pts:.1f} &times; {mult}x</div>'
                f'</div>'
            )
        cap_parts.append('</div></div>')
        st.markdown("".join(cap_parts), unsafe_allow_html=True)

    # Cap display at top 5 for Draft (all players are differentials)
    max_show = 5 if is_draft else None
    show1 = team1_diffs[:max_show] if max_show else team1_diffs
    show2 = team2_diffs[:max_show] if max_show else team2_diffs

    POS_COLORS = {"G": "#f59e0b", "D": "#3b82f6", "M": "#10b981", "F": "#ef4444"}

    def _build_card(diffs, team_label):
        parts = [
            '<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            'padding:16px;border-radius:10px;color:#e0e0e0;">',
            f'<div style="font-size:14px;font-weight:600;color:#00ff87;'
            f'margin-bottom:12px;">{team_label}</div>',
        ]
        if not diffs:
            parts.append(
                '<div style="color:rgba(255,255,255,0.5);font-size:13px;">No unique players</div>'
            )
            parts.append('</div>')
            return "".join(parts)

        total = sum(d["points"] for d in diffs)
        for d in diffs:
            pos_color = POS_COLORS.get(d["position"], "#9ca3af")
            matchup_html = (
                f' <span style="color:rgba(255,255,255,0.4);font-size:11px;">({d["matchup"]})</span>'
                if d["matchup"] else ""
            )
            parts.append(
                f'<div style="display:flex;align-items:center;padding:8px 0;'
                f'border-bottom:1px solid rgba(255,255,255,0.08);color:#e0e0e0;">'
                f'<span style="background:{pos_color};color:white;font-size:10px;'
                f'font-weight:700;padding:2px 6px;border-radius:4px;'
                f'margin-right:8px;min-width:20px;text-align:center;">{d["position"]}</span>'
                f'<span style="flex:1;font-size:13px;color:#e0e0e0;">'
                f'{d["player"]}'
                f' <span style="color:rgba(255,255,255,0.5);font-size:11px;">{d["epl_team"]}</span>'
                f'{matchup_html}</span>'
                f'<span style="font-weight:700;color:#00ff87;font-size:14px;">{d["points"]:.1f}</span>'
                f'</div>'
            )
        parts.append(
            f'<div style="display:flex;justify-content:space-between;margin-top:12px;'
            f'padding-top:8px;border-top:1px solid rgba(255,255,255,0.15);color:#e0e0e0;">'
            f'<span style="font-size:13px;font-weight:600;color:rgba(255,255,255,0.7);">Differential Total</span>'
            f'<span style="font-size:15px;font-weight:700;color:#00ff87;">{total:.1f}</span>'
            f'</div>'
        )
        parts.append('</div>')
        return "".join(parts)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_build_card(show1, team1_name), unsafe_allow_html=True)
    with col2:
        st.markdown(_build_card(show2, team2_name), unsafe_allow_html=True)

    if is_draft:
        st.caption(
            "In Draft leagues all players are unique to one team. "
            "Showing top 5 differentials per side by projected points."
        )
