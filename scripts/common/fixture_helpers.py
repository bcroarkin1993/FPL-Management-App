"""
Fixture & Scheduling Utilities.

Fixture difficulty grid, kickoff times, deadline calculations, and FDR styling.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

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
    # FDR palette: 1=easy(green) -> 5=hard(red), dark-theme friendly
    PALETTE = {
        1: ("#166534", "#86efac"),  # (bg, text) dark green bg, light green text
        2: ("#14532d", "#bbf7d0"),  # slightly lighter green
        3: ("#1a1a2e", "#e0e0e0"),  # neutral dark bg
        4: ("#7f1d1d", "#fecaca"),  # dark red bg, light red text
        5: ("#991b1b", "#fca5a5"),  # darker red bg
    }
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
                # Color Avg FDR by value (lower = greener)
                fdr_val = float(val) if pd.notna(val) else 3.0
                k = max(1, min(5, int(round(fdr_val))))
                bg, txt = PALETTE[k]
                parts.append(
                    f'<td style="padding:8px 12px;background:{bg};color:{txt};'
                    f'font-weight:700;text-align:right;border-bottom:1px solid #333;">'
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
