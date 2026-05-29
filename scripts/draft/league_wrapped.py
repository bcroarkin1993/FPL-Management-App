"""
FPL Draft League Wrapped — league-wide end-of-season summary page.

8-part narrative:
  1. League Champion     (hero banner + final standings)
  2. Season Journey      (cumulative points race + league position timeline)
  3. League Awards       (8 superlatives)
  4. Gameweek Highlights (highest/lowest GW, closest H2H, biggest rank swing)
  5. Head-to-Head Records (full W-D-L matrix + notable rivalries)
  6. Draft Board         (league-wide steals & busts)
  7. Transfer Window     (activity leaderboard + best/worst moves)
  8. Lineup Management   (bench points missed leaderboard)
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from scripts.common.bench_analysis import compute_draft_league_bench_data
from scripts.common.error_helpers import get_logger
from scripts.common.styled_tables import render_styled_table
from scripts.common.fpl_draft_api import (
    _get_draft_gw_live_points,
    get_current_gameweek,
    get_draft_league_details,
    get_fpl_player_mapping,
    get_waiver_transactions_up_to_gameweek,
    pull_fpl_player_stats,
)
from scripts.common.luck_analysis import calculate_all_play_standings, extract_draft_gw_scores
from scripts.common.player_matching import canonical_normalize
from scripts.draft.home import build_draft_history_df
from scripts.draft.league_analysis import build_h2h_matrix, get_matches_df, get_team_names
from scripts.draft.pdf_export import generate_league_wrapped_pdf
from scripts.draft.season_wrapped import (
    _compute_league_superlatives,
    show_season_wrapped_page,
)

_logger = get_logger("fpl_app.draft.league_wrapped")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PURPLE = "#7B2FBE"
_GOLD = "#FFD700"
_RED = "#FF4B4B"
_GREEN = "#00ff87"

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=13),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=12)),
)

_TEAM_COLORS = [
    "#7B2FBE", "#00b4d8", "#f72585", "#4cc9f0", "#43aa8b",
    "#f8961e", "#90be6d", "#f94144", "#9d4edd", "#3a86ff",
    "#06d6a0", "#ef476f",
]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _section_header(icon: str, title: str, subtitle: str = "") -> None:
    sub_html = f'<p style="color:#9ca3af;margin-top:4px;font-size:14px;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<h2 style="color:{_PURPLE};border-bottom:2px solid {_PURPLE};padding-bottom:8px;">'
        f'{icon} {title}</h2>{sub_html}',
        unsafe_allow_html=True,
    )


def _award_card(icon: str, title: str, team: str, detail: str, accent: str = _PURPLE) -> str:
    return (
        f'<div style="border:1px solid {accent};border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;'
        f'color:#e0e0e0;height:100%;">'
        f'<div style="font-size:2em;margin-bottom:8px;">{icon}</div>'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{title}</div>'
        f'<div style="color:{accent};font-size:16px;font-weight:700;margin-bottom:4px;">{team}</div>'
        f'<div style="color:#888;font-size:12px;">{detail}</div>'
        f'</div>'
    )


def _mini_card(label: str, value: str, detail: str = "", accent: str = _PURPLE) -> str:
    det = f'<div style="color:#888;font-size:11px;margin-top:3px;">{detail}</div>' if detail else ""
    return (
        f'<div style="border:1px solid #333;border-radius:8px;padding:12px 14px;'
        f'background:#16213e;text-align:center;color:#e0e0e0;">'
        f'<div style="color:#9ca3af;font-size:10px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{accent};font-size:18px;font-weight:700;">{value}</div>'
        f'{det}</div>'
    )


# ---------------------------------------------------------------------------
# Data: GW highlights
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_gw_highlights(league_id: int, max_gw: int) -> Dict:
    """Compute notable GW moments across the entire league."""
    league_data = get_draft_league_details(league_id)
    if not league_data:
        return {}

    history_df = build_draft_history_df(league_id)
    if history_df is None or history_df.empty:
        return {}

    matches = league_data.get("matches", [])
    entries = league_data.get("league_entries", [])
    entry_names = {e["id"]: e["entry_name"] for e in entries}

    played = history_df[history_df["GW_Points"] > 0].copy()
    if played.empty:
        return {}

    # Highest / lowest scoring GW
    high_idx = played["GW_Points"].idxmax()
    low_idx = played["GW_Points"].idxmin()
    highest_gw = {
        "team": played.loc[high_idx, "Team"],
        "gw": int(played.loc[high_idx, "Gameweek"]),
        "score": int(played.loc[high_idx, "GW_Points"]),
    }
    lowest_gw = {
        "team": played.loc[low_idx, "Team"],
        "gw": int(played.loc[low_idx, "Gameweek"]),
        "score": int(played.loc[low_idx, "GW_Points"]),
    }

    # Biggest blowout (largest margin of victory in a single match)
    biggest_blowout = {}
    max_margin = -1
    for match in matches:
        pts1 = match.get("league_entry_1_points", 0) or 0
        pts2 = match.get("league_entry_2_points", 0) or 0
        if pts1 == 0 and pts2 == 0:
            continue
        margin = abs(pts1 - pts2)
        if margin > max_margin:
            max_margin = margin
            t1 = entry_names.get(match.get("league_entry_1"), "?")
            t2 = entry_names.get(match.get("league_entry_2"), "?")
            if pts1 >= pts2:
                biggest_blowout = {"winner": t1, "loser": t2, "score1": pts1, "score2": pts2,
                                   "gw": match.get("event", 0), "margin": margin}
            else:
                biggest_blowout = {"winner": t2, "loser": t1, "score1": pts2, "score2": pts1,
                                   "gw": match.get("event", 0), "margin": margin}

    # Biggest single-GW league-position swing
    biggest_swing: Dict = {}
    if "League_Position" in history_df.columns:
        swing_df = history_df.sort_values(["Team", "Gameweek"]).copy()
        swing_df["Pos_Delta"] = swing_df.groupby("Team")["League_Position"].diff().abs()
        swing_df = swing_df.dropna(subset=["Pos_Delta"])
        if not swing_df.empty:
            sw_idx = swing_df["Pos_Delta"].idxmax()
            sw_team = swing_df.loc[sw_idx, "Team"]
            sw_gw = int(swing_df.loc[sw_idx, "Gameweek"])
            prev = history_df[(history_df["Team"] == sw_team) & (history_df["Gameweek"] == sw_gw - 1)]
            curr = history_df[(history_df["Team"] == sw_team) & (history_df["Gameweek"] == sw_gw)]
            from_rank = int(prev["League_Position"].values[0]) if len(prev) > 0 else "?"
            to_rank = int(curr["League_Position"].values[0]) if len(curr) > 0 else "?"
            direction = "📈" if isinstance(to_rank, int) and isinstance(from_rank, int) and to_rank < from_rank else "📉"
            biggest_swing = {
                "team": sw_team,
                "gw": sw_gw,
                "swing": int(swing_df.loc[sw_idx, "Pos_Delta"]),
                "from_rank": from_rank,
                "to_rank": to_rank,
                "direction": direction,
            }

    return {
        "highest_gw": highest_gw,
        "lowest_gw": lowest_gw,
        "biggest_blowout": biggest_blowout,
        "biggest_rank_swing": biggest_swing,
    }


# ---------------------------------------------------------------------------
# Data: League draft board (all teams)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_league_draft_board(league_id: int) -> Dict:
    """
    League-wide draft analysis: top 5 steals, top 5 busts across all teams,
    plus per-team pick summary.
    """
    league_data = get_draft_league_details(league_id)
    if not league_data:
        return {}

    entries = league_data.get("league_entries", [])
    num_teams = len(entries)
    if num_teams == 0:
        return {}

    try:
        draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
        choices_raw = requests.get(draft_url, timeout=30).json().get("choices", [])
    except Exception:
        _logger.warning("Failed to fetch draft choices for league %s", league_id, exc_info=True)
        return {}

    if not choices_raw:
        return {}

    try:
        fpl_stats = pull_fpl_player_stats()
    except Exception:
        fpl_stats = pd.DataFrame()

    player_map = get_fpl_player_mapping()

    pts_lookup: Dict[str, int] = {}
    if not fpl_stats.empty:
        for _, row in fpl_stats.iterrows():
            pts_lookup[canonical_normalize(str(row.get("player", "")))] = int(row.get("total_points", 0))

    def _get_pts(player_id: int) -> int:
        info = player_map.get(player_id, {})
        name = info.get("Player", "")
        if not name:
            return 0
        norm = canonical_normalize(name)
        pts = pts_lookup.get(norm, 0)
        if pts == 0 and norm:
            last = norm.split()[-1]
            for k, v in pts_lookup.items():
                if k.split()[-1] == last:
                    return v
        return pts

    # Parse all choices
    all_choices = []
    for idx, c in enumerate(choices_raw):
        pid = c.get("element")
        ename = c.get("entry_name", "")
        if not pid:
            continue
        r = idx // num_teams + 1
        pick_in_round = idx % num_teams + 1
        all_choices.append({
            "overall": idx + 1,
            "round": r,
            "pick": pick_in_round,
            "player_id": int(pid),
            "entry_name": ename,
        })

    if not all_choices:
        return {}

    # Round averages
    round_pts: Dict[int, List[int]] = {}
    for c in all_choices:
        pts = _get_pts(c["player_id"])
        round_pts.setdefault(c["round"], []).append(pts)
    round_avg = {r: sum(v) / len(v) for r, v in round_pts.items()}

    # Grade each pick
    all_picks = []
    for c in all_choices:
        pts = _get_pts(c["player_id"])
        avg = round_avg.get(c["round"], pts)
        delta = pts - avg
        if delta >= 30:
            grade = "Steal 🔥"
        elif delta >= 10:
            grade = "Value ✅"
        elif delta >= -10:
            grade = "Fair ⚖️"
        elif delta >= -30:
            grade = "Miss 📉"
        else:
            grade = "Bust 💀"
        info = player_map.get(c["player_id"], {})
        all_picks.append({
            "team": c["entry_name"],
            "round": c["round"],
            "pick": c["pick"],
            "player": info.get("Player", f"ID {c['player_id']}"),
            "position": info.get("Position", ""),
            "pts": pts,
            "avg": round(avg, 1),
            "delta": round(delta, 1),
            "grade": grade,
        })

    steals = sorted([p for p in all_picks if p["grade"] == "Steal 🔥"], key=lambda x: -x["delta"])[:5]
    busts = sorted([p for p in all_picks if p["grade"] == "Bust 💀"], key=lambda x: x["delta"])[:5]

    # Per-team draft stats: count of each grade
    team_grades: Dict[str, Dict[str, int]] = {}
    for p in all_picks:
        team_grades.setdefault(p["team"], {"Steal 🔥": 0, "Value ✅": 0, "Fair ⚖️": 0, "Miss 📉": 0, "Bust 💀": 0})
        team_grades[p["team"]][p["grade"]] = team_grades[p["team"]].get(p["grade"], 0) + 1

    return {"steals": steals, "busts": busts, "team_grades": team_grades}


# ---------------------------------------------------------------------------
# Data: League transfer stats
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_league_transfer_stats(league_id: int, max_gw: int) -> Dict:
    """
    League-wide transfer analysis: activity leaderboard, best/worst moves, most in/out.
    """
    league_data = get_draft_league_details(league_id)
    if not league_data:
        return {}

    entries = league_data.get("league_entries", [])
    raw_to_league = {e.get("entry_id"): e["id"] for e in entries if e.get("entry_id")}
    league_to_name = {e["id"]: e["entry_name"] for e in entries}

    player_map = get_fpl_player_mapping()
    all_tx = get_waiver_transactions_up_to_gameweek(league_id, max_gw)

    # Pre-compute GW live points for all GWs (SQLite-cached for finished GWs)
    gw_cache: Dict[int, Dict[int, int]] = {}
    for gw in range(1, max_gw + 1):
        gw_cache[gw] = _get_draft_gw_live_points(gw)

    def _pts_after(el_id: int, from_gw: int, window: int = 0) -> int:
        end = (from_gw + window) if window > 0 else max_gw
        return sum(gw_cache.get(g, {}).get(el_id, 0) for g in range(from_gw + 1, min(end, max_gw) + 1))

    def _player_name(el_id: int) -> str:
        info = player_map.get(el_id, {})
        return info.get("Player") or info.get("Web_Name") or f"Player {el_id}"

    per_team: Dict[str, int] = {}
    most_in: Dict[str, int] = {}
    most_out: Dict[str, int] = {}

    best_net = float("-inf")
    best_transfer: Dict = {}
    worst_kept = float("-inf")
    worst_transfer: Dict = {}

    for tx in all_tx:
        if tx.get("result") != "a":
            continue
        el_in = tx.get("element_in")
        el_out = tx.get("element_out")
        trade_gw = tx.get("event", 0)
        raw_eid = tx.get("entry")
        if not el_in or not el_out:
            continue

        league_eid = raw_to_league.get(raw_eid, raw_eid)
        team_name = league_to_name.get(league_eid, f"Team {raw_eid}")

        per_team[team_name] = per_team.get(team_name, 0) + 1
        most_in[_player_name(el_in)] = most_in.get(_player_name(el_in), 0) + 1
        most_out[_player_name(el_out)] = most_out.get(_player_name(el_out), 0) + 1

        pts_in = _pts_after(el_in, trade_gw)
        pts_out = _pts_after(el_out, trade_gw)
        net = pts_in - pts_out

        if net > best_net:
            best_net = net
            best_transfer = {
                "team": team_name,
                "player_in": _player_name(el_in),
                "player_out": _player_name(el_out),
                "gw": trade_gw,
                "pts_in": pts_in,
                "pts_out": pts_out,
                "net": net,
            }

        # Worst drop: player dropped who then scored the most
        pts_out_season = _pts_after(el_out, trade_gw)
        if pts_out_season > worst_kept:
            worst_kept = pts_out_season
            worst_transfer = {
                "team": team_name,
                "player_out": _player_name(el_out),
                "player_in": _player_name(el_in),
                "gw": trade_gw,
                "pts_out_after": pts_out_season,
            }

    return {
        "per_team": dict(sorted(per_team.items(), key=lambda x: -x[1])),
        "best_transfer": best_transfer,
        "worst_transfer": worst_transfer,
        "most_in": sorted(most_in.items(), key=lambda x: -x[1])[:5],
        "most_out": sorted(most_out.items(), key=lambda x: -x[1])[:5],
    }


# ---------------------------------------------------------------------------
# Part 1: League Champion
# ---------------------------------------------------------------------------

def _render_champion_banner(league_data: dict) -> None:
    standings = league_data.get("standings", [])
    entries = league_data.get("league_entries", [])
    if not standings or not entries:
        st.info("No standings data yet.")
        return

    entry_names = {e["id"]: e["entry_name"] for e in entries}
    sorted_standings = sorted(standings, key=lambda s: s.get("rank", 999))

    winner = sorted_standings[0]
    winner_name = entry_names.get(winner.get("league_entry"), "?")
    w = winner.get("matches_won", 0)
    d = winner.get("matches_drawn", 0)
    l = winner.get("matches_lost", 0)
    pts_for = winner.get("points_for", 0)
    league_pts = w * 3 + d

    st.markdown(
        f'<div style="background:linear-gradient(135deg,#1a1a2e 0%,#2d1b69 50%,#1a1a2e 100%);'
        f'border:2px solid {_GOLD};border-radius:16px;padding:32px;text-align:center;margin-bottom:20px;">'
        f'<div style="font-size:3em;margin-bottom:8px;">🏆</div>'
        f'<div style="color:{_GOLD};font-size:2.2em;font-weight:800;margin-bottom:4px;">{winner_name}</div>'
        f'<div style="color:#e0e0e0;font-size:1.1em;margin-bottom:12px;">2025/26 League Champion</div>'
        f'<div style="color:#9ca3af;font-size:1em;">'
        f'{w}W – {d}D – {l}L &nbsp;|&nbsp; {league_pts} league pts &nbsp;|&nbsp; {pts_for:,} FPL pts'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Runner-up + Relegated side by side
    if len(sorted_standings) >= 2:
        ru = sorted_standings[1]
        ru_name = entry_names.get(ru.get("league_entry"), "?")
        ru_w = ru.get("matches_won", 0)
        ru_d = ru.get("matches_drawn", 0)
        ru_l = ru.get("matches_lost", 0)
        ru_lp = ru_w * 3 + ru_d

        ru_html = (
            f'<div style="flex:1;border:1px solid #9d4edd;border-radius:10px;padding:16px 14px;'
            f'background:#16213e;text-align:center;color:#e0e0e0;">'
            f'<div style="font-size:1.8em;margin-bottom:6px;">🥈</div>'
            f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;'
            f'letter-spacing:2px;margin-bottom:6px;">Runner-Up</div>'
            f'<div style="color:#9d4edd;font-size:17px;font-weight:700;margin-bottom:5px;">{ru_name}</div>'
            f'<div style="color:#888;font-size:12px;">'
            f'{ru_w}W – {ru_d}D – {ru_l}L &nbsp;·&nbsp; {ru_lp} league pts</div>'
            f'</div>'
        )

        rel_html = ""
        if len(sorted_standings) >= 3:
            rel = sorted_standings[-1]
            rel_name = entry_names.get(rel.get("league_entry"), "?")
            rel_w = rel.get("matches_won", 0)
            rel_d = rel.get("matches_drawn", 0)
            rel_l = rel.get("matches_lost", 0)
            rel_lp = rel_w * 3 + rel_d
            rel_html = (
                f'<div style="flex:1;background:linear-gradient(135deg,#2d0000 0%,#1a0000 100%);'
                f'border:2px solid {_RED};border-radius:10px;padding:16px 14px;text-align:center;">'
                f'<div style="font-size:1.8em;margin-bottom:6px;">❌</div>'
                f'<div style="color:{_RED};font-size:12px;font-weight:900;'
                f'text-transform:uppercase;letter-spacing:3px;margin-bottom:6px;">Relegated</div>'
                f'<div style="color:#ff8080;font-size:17px;font-weight:700;margin-bottom:5px;">{rel_name}</div>'
                f'<div style="color:#cc4444;font-size:12px;">'
                f'{rel_w}W – {rel_d}D – {rel_l}L &nbsp;·&nbsp; {rel_lp} league pts'
                f'&nbsp;·&nbsp; <em>Removed for 2026/27</em></div>'
                f'</div>'
            )

        st.markdown(
            f'<div style="display:flex;gap:12px;margin-bottom:14px;">'
            f'{ru_html}{rel_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Final standings table
    rows = []
    for row in sorted_standings:
        eid = row.get("league_entry")
        name = entry_names.get(eid, "?")
        rw = row.get("matches_won", 0)
        rd = row.get("matches_drawn", 0)
        rl = row.get("matches_lost", 0)
        pf = row.get("points_for", 0)
        pa = row.get("points_against", 0)
        lp = rw * 3 + rd
        rank = row.get("rank", "?")
        rows.append({"Rank": rank, "Team": name, "W": rw, "D": rd, "L": rl,
                     "League Pts": lp, "Pts For": pf, "Pts Against": pa,
                     "Pts Diff": pf - pa})
    if rows:
        df_standings = pd.DataFrame(rows)
        render_styled_table(
            df_standings,
            title="Final League Standings",
            col_formats={"Pts For": "{:,}", "Pts Against": "{:,}", "Pts Diff": "{:+,}"},
            text_align={"Rank": "center", "Team": "left",
                        "W": "center", "D": "center", "L": "center"},
            positive_color_cols=["League Pts", "Pts For", "Pts Diff"],
            negative_color_cols=["Pts Against"],
            highlight_row=lambda row: row.get("Rank") == 1,
        )


# ---------------------------------------------------------------------------
# Part 2: Season Journey
# ---------------------------------------------------------------------------

def _render_season_journey(history_df: pd.DataFrame) -> None:
    if history_df is None or history_df.empty:
        st.info("No season history data available.")
        return

    teams = sorted(history_df["Team"].unique().tolist())

    # Identify champion (lowest final League_Position)
    last_gw = history_df["Gameweek"].max()
    final = history_df[history_df["Gameweek"] == last_gw][["Team", "League_Position"]]
    champion = final.loc[final["League_Position"].idxmin(), "Team"] if not final.empty else None

    color_map = {team: _TEAM_COLORS[i % len(_TEAM_COLORS)] for i, team in enumerate(teams)}
    if champion:
        color_map[champion] = _GOLD

    # Cumulative points race
    fig1 = go.Figure()
    for team in teams:
        tdf = history_df[history_df["Team"] == team].sort_values("Gameweek")
        is_champ = team == champion
        fig1.add_trace(go.Scatter(
            x=tdf["Gameweek"],
            y=tdf["Total_Points"],
            name=team,
            mode="lines",
            line=dict(color=color_map[team], width=3 if is_champ else 1.5),
        ))
    fig1.update_layout(
        **_DARK_CHART_LAYOUT,
        title=dict(text="📈 Cumulative Points Race", font=dict(size=18, color="#ffffff"),
                   x=0.5, xanchor="center"),
        height=420,
    )
    fig1.update_xaxes(title="Gameweek", dtick=2, gridcolor="#2a2a3e")
    fig1.update_yaxes(title="Total FPL Points", gridcolor="#2a2a3e")
    st.plotly_chart(fig1, use_container_width=True)

    # League position timeline
    fig2 = go.Figure()
    num_teams = len(teams)
    for team in teams:
        tdf = history_df[history_df["Team"] == team].sort_values("Gameweek")
        is_champ = team == champion
        fig2.add_trace(go.Scatter(
            x=tdf["Gameweek"],
            y=tdf["League_Position"],
            name=team,
            mode="lines",
            line=dict(color=color_map[team], width=3 if is_champ else 1.5),
        ))
    fig2.update_layout(
        **_DARK_CHART_LAYOUT,
        title=dict(text="🏅 League Position Timeline", font=dict(size=18, color="#ffffff"),
                   x=0.5, xanchor="center"),
        height=420,
    )
    fig2.update_xaxes(title="Gameweek", dtick=2, gridcolor="#2a2a3e")
    fig2.update_yaxes(
        title="League Position",
        autorange="reversed",
        dtick=1,
        range=[num_teams + 0.5, 0.5],
        gridcolor="#2a2a3e",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Part 3: League Awards
# ---------------------------------------------------------------------------

def _render_league_awards(superlatives: Dict, history_df: pd.DataFrame) -> None:
    # 5 existing superlatives
    most_active = superlatives.get("most_active", {})
    best_mgr = superlatives.get("best_mgr", {})
    luckiest = superlatives.get("luckiest", {})
    unluckiest = superlatives.get("unluckiest", {})
    best_drafter = superlatives.get("best_drafter", {})

    # 3 new awards from history_df
    pts_champ: Dict = {"team": "?", "value": ""}
    high_gw: Dict = {"team": "?", "value": ""}
    consistent: Dict = {"team": "?", "value": ""}

    played = history_df[history_df["GW_Points"] > 0] if history_df is not None and not history_df.empty else pd.DataFrame()
    if not played.empty:
        last_gw = played["Gameweek"].max()
        final = played[played["Gameweek"] == last_gw][["Team", "Total_Points"]]
        if not final.empty:
            idx = final["Total_Points"].idxmax()
            pts_champ = {
                "team": final.loc[idx, "Team"],
                "value": f'{int(final.loc[idx, "Total_Points"]):,} total FPL pts',
            }

        h_idx = played["GW_Points"].idxmax()
        high_gw = {
            "team": played.loc[h_idx, "Team"],
            "value": f'{int(played.loc[h_idx, "GW_Points"])} pts in GW{int(played.loc[h_idx, "Gameweek"])}',
        }

        std_df = played.groupby("Team")["GW_Points"].agg(["std", "count"])
        std_df = std_df[std_df["count"] >= 5]
        if not std_df.empty:
            c_team = std_df["std"].idxmin()
            c_val = round(std_df.loc[c_team, "std"], 1)
            consistent = {"team": c_team, "value": f"σ = {c_val} pts/GW (lowest variance)"}

    awards = [
        ("🏆", "Points Champion", pts_champ.get("team", "?"), pts_champ.get("value", ""), _GOLD),
        ("📊", "Highest Single GW", high_gw.get("team", "?"), high_gw.get("value", ""), "#00b4d8"),
        ("🎯", "Most Consistent", consistent.get("team", "?"), consistent.get("value", ""), _GREEN),
        ("🔀", "Most Active Manager", most_active.get("team", "?"),
         f'{most_active.get("value", 0)} approved transactions', "#f8961e"),
        ("🧠", "Best Lineup Manager", best_mgr.get("team", "?"),
         best_mgr.get("value", ""), _PURPLE),
        ("✏️", "Best Drafter", best_drafter.get("team", "?"),
         best_drafter.get("value", ""), "#9d4edd"),
        ("🍀", "Luckiest Manager", luckiest.get("team", "?"),
         luckiest.get("value", ""), "#43aa8b"),
        ("😤", "Most Unlucky", unluckiest.get("team", "?"),
         unluckiest.get("value", ""), _RED),
    ]

    # 4 cols × 2 rows
    for row_start in range(0, len(awards), 4):
        cols = st.columns(4)
        for col, (icon, title, team, detail, accent) in zip(cols, awards[row_start:row_start + 4]):
            col.markdown(_award_card(icon, title, team, detail, accent), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Part 4: Gameweek Highlights
# ---------------------------------------------------------------------------

def _render_gw_highlights(highlights: Dict) -> None:
    if not highlights:
        st.info("Not enough data for GW highlights.")
        return

    high = highlights.get("highest_gw", {})
    low = highlights.get("lowest_gw", {})
    blowout = highlights.get("biggest_blowout", {})
    swing = highlights.get("biggest_rank_swing", {})

    cards = []

    if high:
        cards.append(_mini_card(
            "🔥 Highest GW Score",
            f'{high.get("score", "?")} pts',
            f'{high.get("team", "?")} — GW{high.get("gw", "?")}',
            _GOLD,
        ))
    if low:
        cards.append(_mini_card(
            "🥶 Lowest GW Score",
            f'{low.get("score", "?")} pts',
            f'{low.get("team", "?")} — GW{low.get("gw", "?")}',
            _RED,
        ))
    if blowout:
        cards.append(_mini_card(
            "💥 Biggest Blowout",
            f'{blowout.get("score1", "?")} – {blowout.get("score2", "?")}',
            f'{blowout.get("winner", "?")} def. {blowout.get("loser", "?")} by {blowout.get("margin", "?")} pts (GW{blowout.get("gw", "?")})',
            "#00b4d8",
        ))
    if swing:
        cards.append(_mini_card(
            f'{swing.get("direction", "📊")} Biggest Rank Swing',
            f'{swing.get("swing", "?")} places',
            f'{swing.get("team", "?")} — #{swing.get("from_rank", "?")} → #{swing.get("to_rank", "?")} in GW{swing.get("gw", "?")}',
            "#9d4edd",
        ))

    cols = st.columns(len(cards)) if cards else st.columns(1)
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Part 5: Head-to-Head Records
# ---------------------------------------------------------------------------

def _render_h2h_records(league_data: dict) -> None:
    team_names = get_team_names(league_data)
    matches_df = get_matches_df(league_data, team_names)
    if matches_df.empty:
        st.info("No match data available.")
        return

    h2h_matrix = build_h2h_matrix(matches_df, team_names)
    if h2h_matrix.empty:
        st.info("No H2H data available.")
        return

    teams = h2h_matrix.index.tolist()

    def _parse_wdl(cell: str) -> Tuple[int, int, int]:
        if cell == "-" or not cell:
            return (0, 0, 0)
        parts = cell.split("-")
        if len(parts) == 3:
            try:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                return (0, 0, 0)
        return (0, 0, 0)

    _TH = (
        "background:linear-gradient(135deg,#37003c,#5a0060);color:#00ff87;"
        "font-weight:600;font-size:12px;padding:10px 12px;"
        "border-bottom:2px solid #5a0060;white-space:nowrap;text-align:center;"
    )
    _TD_TEAM = (
        "padding:10px 12px;background:#1e1e2e;color:#e0e0e0;font-weight:700;"
        "white-space:nowrap;border:1px solid #2a2a3a;font-size:13px;"
    )

    def _cell_bg(cell: str) -> str:
        if cell == "-":
            return "background:#1a1a2e;color:#444;"
        w, d, l = _parse_wdl(cell)
        if w > l:
            return f"background:rgba(0,168,85,0.18);color:#00a855;font-weight:700;"
        if w < l:
            return f"background:rgba(255,75,75,0.15);color:{_RED};font-weight:600;"
        return f"background:rgba(255,215,0,0.10);color:{_GOLD};"

    # Build HTML table (Season Wrapped–style: purple gradient header)
    header_cells = "".join(
        f'<th style="{_TH}max-width:90px;overflow:hidden;text-overflow:ellipsis;">'
        f'{t[:12]}</th>'
        for t in teams
    )
    html = (
        f'<div style="border:1px solid #333;border-radius:10px;overflow:hidden;'
        f'overflow-x:auto;margin-bottom:1rem;">'
        f'<table style="width:100%;border-collapse:collapse;background:#1a1a2e;font-size:13px;">'
        f'<thead><tr>'
        f'<th style="{_TH}text-align:left;">Team</th>'
        f'{header_cells}'
        f'</tr></thead><tbody>'
    )
    for i, team in enumerate(teams):
        row_bg = "rgba(255,255,255,0.03)" if i % 2 == 1 else "#1a1a2e"
        row_cells = "".join(
            f'<td style="padding:10px 12px;text-align:center;border:1px solid #2a2a3a;'
            f'{_cell_bg(h2h_matrix.loc[team, opp])}">'
            f'{h2h_matrix.loc[team, opp]}</td>'
            for opp in teams
        )
        html += (
            f'<tr style="background:{row_bg};">'
            f'<td style="{_TD_TEAM}">{team}</td>'
            f'{row_cells}</tr>'
        )
    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)
    st.caption("Green = winning record · Red = losing record · Gold = .500")

    # Notable rivalries
    _render_notable_rivalries(h2h_matrix, _parse_wdl)


def _render_notable_rivalries(h2h_matrix: pd.DataFrame, parse_fn) -> None:
    """Find and display the most one-sided and most evenly matched H2H pairs."""
    teams = h2h_matrix.index.tolist()
    seen: set = set()
    most_lopsided = {"margin": -1, "team1": "", "team2": "", "record": ""}
    most_even = {"margin": float("inf"), "team1": "", "team2": "", "record": ""}

    for i, t1 in enumerate(teams):
        for t2 in teams[i + 1:]:
            pair_key = tuple(sorted([t1, t2]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            w1, d1, l1 = parse_fn(h2h_matrix.loc[t1, t2])
            total = w1 + d1 + l1
            if total == 0:
                continue
            lopsided_margin = abs(w1 - l1)
            if lopsided_margin > most_lopsided["margin"]:
                most_lopsided = {
                    "margin": lopsided_margin,
                    "team1": t1 if w1 >= l1 else t2,
                    "team2": t2 if w1 >= l1 else t1,
                    "record": f"{max(w1, l1)}-{d1}-{min(w1, l1)}",
                }
            even_margin = abs(w1 - l1)
            if even_margin < most_even["margin"]:
                most_even = {
                    "margin": even_margin,
                    "team1": t1,
                    "team2": t2,
                    "record": f"{w1}-{d1}-{l1}",
                }

    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(2)
    if most_lopsided["team1"]:
        with cols[0]:
            st.markdown(
                _award_card("⚔️", "Most One-Sided Rivalry",
                            f'{most_lopsided["team1"]} vs {most_lopsided["team2"]}',
                            most_lopsided["record"], _RED),
                unsafe_allow_html=True,
            )
    if most_even["team1"]:
        with cols[1]:
            st.markdown(
                _award_card("🤝", "Most Evenly Matched",
                            f'{most_even["team1"]} vs {most_even["team2"]}',
                            most_even["record"], "#00b4d8"),
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Part 6: Draft Board
# ---------------------------------------------------------------------------

def _render_draft_board(draft_data: Dict) -> None:
    if not draft_data:
        st.info("No draft data available.")
        return

    steals = draft_data.get("steals", [])
    busts = draft_data.get("busts", [])

    def _pick_card(pick: Dict, accent: str) -> str:
        return (
            f'<div style="border:1px solid {accent};border-radius:8px;padding:12px;'
            f'background:#16213e;margin-bottom:8px;color:#e0e0e0;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;">'
            f'<div><div style="color:{accent};font-weight:700;font-size:14px;">{pick["player"]}</div>'
            f'<div style="color:#888;font-size:11px;">{pick["team"]} · Rd {pick["round"]} Pick {pick["pick"]}</div>'
            f'</div>'
            f'<div style="text-align:right;">'
            f'<div style="color:{accent};font-size:16px;font-weight:700;">{pick["grade"]}</div>'
            f'<div style="color:#9ca3af;font-size:11px;">{pick["pts"]} pts · Δ{pick["delta"]:+.0f}</div>'
            f'</div></div></div>'
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div style="color:{_GREEN};font-size:14px;font-weight:700;margin-bottom:8px;">🔥 Top 5 Steals</div>', unsafe_allow_html=True)
        if steals:
            for p in steals:
                st.markdown(_pick_card(p, _GREEN), unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#888;font-size:13px;">No steals found.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div style="color:{_RED};font-size:14px;font-weight:700;margin-bottom:8px;">💀 Top 5 Busts</div>', unsafe_allow_html=True)
        if busts:
            for p in busts:
                st.markdown(_pick_card(p, _RED), unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#888;font-size:13px;">No busts found.</div>', unsafe_allow_html=True)

    # Per-team grade summary table
    team_grades = draft_data.get("team_grades", {})
    if team_grades:
        st.markdown("<br>", unsafe_allow_html=True)
        rows = []
        for team, grades in team_grades.items():
            rows.append({
                "Team": team,
                "Steals 🔥": grades.get("Steal 🔥", 0),
                "Value ✅": grades.get("Value ✅", 0),
                "Fair ⚖️": grades.get("Fair ⚖️", 0),
                "Miss 📉": grades.get("Miss 📉", 0),
                "Busts 💀": grades.get("Bust 💀", 0),
            })
        df_grades = pd.DataFrame(rows).sort_values("Steals 🔥", ascending=False)
        render_styled_table(
            df_grades,
            title="Draft Grade Summary by Team",
            text_align={"Team": "left"},
            positive_color_cols=["Steals 🔥", "Value ✅"],
            negative_color_cols=["Busts 💀", "Miss 📉"],
        )


# ---------------------------------------------------------------------------
# Part 7: Transfer Window
# ---------------------------------------------------------------------------

def _render_transfer_window(transfer_data: Dict) -> None:
    if not transfer_data:
        st.info("No transfer data available.")
        return

    per_team = transfer_data.get("per_team", {})
    best = transfer_data.get("best_transfer", {})
    worst = transfer_data.get("worst_transfer", {})
    most_in = transfer_data.get("most_in", [])
    most_out = transfer_data.get("most_out", [])

    # Activity leaderboard
    if per_team:
        df = pd.DataFrame([{"Team": t, "Approved Moves": c} for t, c in per_team.items()])
        fig = go.Figure(go.Bar(
            x=df["Team"],
            y=df["Approved Moves"],
            marker_color=[_GOLD if i == 0 else _PURPLE for i in range(len(df))],
            text=df["Approved Moves"],
            textposition="outside",
            textfont=dict(color="#e0e0e0", size=13),
        ))
        fig.update_layout(
            **_DARK_CHART_LAYOUT,
            title=dict(text="Transfer Activity by Team", font=dict(size=15, color="#ffffff"),
                       x=0.5, xanchor="center", y=0.97),
            height=360,
            showlegend=False,
            margin=dict(t=55, l=65, r=20, b=90),
        )
        fig.update_xaxes(tickangle=-35, gridcolor="#2a2a3e")
        fig.update_yaxes(gridcolor="#2a2a3e", range=[0, df["Approved Moves"].max() * 1.2])
        st.plotly_chart(fig, use_container_width=True)

    # Best and worst transfer cards
    if best or worst:
        col1, col2 = st.columns(2)
        if best:
            with col1:
                st.markdown(
                    f'<div style="border:1px solid {_GREEN};border-radius:10px;padding:16px;'
                    f'background:#16213e;color:#e0e0e0;">'
                    f'<div style="color:{_GREEN};font-weight:700;margin-bottom:6px;">✅ Best Transfer In (League)</div>'
                    f'<div style="font-size:15px;font-weight:700;color:#e0e0e0;">{best.get("player_in", "?")} picked up for {best.get("player_out", "?")}</div>'
                    f'<div style="color:#888;font-size:12px;margin-top:4px;">'
                    f'{best.get("team", "?")} · GW{best.get("gw", "?")} · '
                    f'+{best.get("net", 0)} net pts ({best.get("pts_in", 0)} in vs {best.get("pts_out", 0)} out)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        if worst:
            with col2:
                st.markdown(
                    f'<div style="border:1px solid {_RED};border-radius:10px;padding:16px;'
                    f'background:#16213e;color:#e0e0e0;">'
                    f'<div style="color:{_RED};font-weight:700;margin-bottom:6px;">❌ Biggest Mistake Drop (League)</div>'
                    f'<div style="font-size:15px;font-weight:700;color:#e0e0e0;">{worst.get("player_out", "?")} dropped for {worst.get("player_in", "?")}</div>'
                    f'<div style="color:#888;font-size:12px;margin-top:4px;">'
                    f'{worst.get("team", "?")} · GW{worst.get("gw", "?")} · '
                    f'{worst.get("player_out", "?")} went on to score {worst.get("pts_out_after", 0)} pts</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Most in/out tables
    if most_in or most_out:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        def _ranked_table(items, header_color, count_label):
            if not items:
                return
            rows_html = "".join(
                f'<tr style="border-bottom:1px solid #2a2a3a;background:{"rgba(255,255,255,0.03)" if i % 2 else "#1a1a2e"};">'
                f'<td style="padding:10px 14px;color:#e0e0e0;font-size:13px;">{i + 1}. {name}</td>'
                f'<td style="padding:10px 14px;text-align:right;">'
                f'<span style="background:{header_color};color:#000;border-radius:12px;'
                f'padding:3px 12px;font-size:13px;font-weight:700;">{cnt}</span>'
                f'</td></tr>'
                for i, (name, cnt) in enumerate(items)
            )
            st.markdown(
                f'<div style="border:1px solid #333;border-radius:10px;overflow:hidden;">'
                f'<table style="width:100%;border-collapse:collapse;background:#1a1a2e;">'
                f'<thead><tr style="background:linear-gradient(135deg,#37003c,#5a0060);">'
                f'<th style="padding:10px 14px;color:#00ff87;text-align:left;font-size:13px;'
                f'font-weight:600;text-transform:uppercase;">Player</th>'
                f'<th style="padding:10px 14px;color:#00ff87;text-align:right;font-size:13px;'
                f'font-weight:600;text-transform:uppercase;">{count_label}</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
                unsafe_allow_html=True,
            )

        with col1:
            st.markdown("##### 📥 Most Transferred In")
            _ranked_table(most_in, _GREEN, "Times In")
        with col2:
            st.markdown("##### 📤 Most Transferred Out")
            _ranked_table(most_out, _RED, "Times Out")


# ---------------------------------------------------------------------------
# Part 8: Lineup Management
# ---------------------------------------------------------------------------

def _render_lineup_management(bench_data_list: List[Dict]) -> None:
    if not bench_data_list:
        st.info("No bench data available.")
        return

    df = pd.DataFrame(bench_data_list)

    # Show scored vs possible totals as a summary table at top
    totals_cols = [c for c in ["Team", "Pts Scored", "Pts Possible", "Total Pts Lost"] if c in df.columns]
    if len(totals_cols) >= 3:
        df_totals = df[totals_cols].copy()
        if "Pts Scored" in df_totals.columns and "Pts Possible" in df_totals.columns:
            df_totals["Efficiency %"] = (
                df_totals["Pts Scored"] / df_totals["Pts Possible"] * 100
            ).round(1)
        render_styled_table(
            df_totals,
            title="Points Scored vs. Optimal (Season Totals)",
            text_align={"Team": "left"},
            positive_color_cols=["Pts Scored", "Efficiency %"],
            negative_color_cols=["Total Pts Lost"],
            col_formats={"Efficiency %": "{:.1f}%"},
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # Lineup management detail table
    detail_cols = [c for c in ["Team", "Avg Lost/GW", "Selection %", "Bench Mgmt Score",
                                "Avg Bench/GW", "Worst GW"] if c in df.columns]
    if detail_cols:
        df_detail = df[detail_cols].copy()
        render_styled_table(
            df_detail,
            title="Lineup Management Detail",
            text_align={"Team": "left", "Worst GW": "left"},
            col_formats={
                "Avg Lost/GW": "{:.1f}",
                "Selection %": "{:.1f}%",
                "Bench Mgmt Score": "{:.1f}",
                "Avg Bench/GW": "{:.1f}",
            },
            positive_color_cols=["Bench Mgmt Score", "Selection %"],
            negative_color_cols=["Avg Lost/GW"],
        )
        st.caption("Sorted by Bench Mgmt Score (best → worst)")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def show_league_wrapped_page():
    st.title("League Wrapped 🏆")
    st.write("The complete 2025/26 FPL Draft season — league-wide story.")

    league_id = config.FPL_DRAFT_LEAGUE_ID
    if not league_id:
        st.error("FPL_DRAFT_LEAGUE_ID is not configured. Check your .env file.")
        return

    # ---------------------------------------------------------------------------
    # Load all data under a single spinner
    # ---------------------------------------------------------------------------
    with st.spinner("Building League Wrapped..."):
        try:
            league_data = get_draft_league_details(league_id)
        except Exception as exc:
            st.error(f"Could not fetch league data: {exc}")
            return

        if not league_data:
            st.error("No league data returned. Check your FPL_DRAFT_LEAGUE_ID setting.")
            return

        max_gw = min(get_current_gameweek(), 38)

        try:
            history_df = build_draft_history_df(league_id)
        except Exception:
            _logger.warning("Failed to build history_df", exc_info=True)
            history_df = pd.DataFrame()

        try:
            bench_data_list = compute_draft_league_bench_data(league_id, max_gw)
        except Exception:
            _logger.warning("Failed to compute bench data", exc_info=True)
            bench_data_list = []

        try:
            superlatives = _compute_league_superlatives(league_id, max_gw)
        except Exception:
            _logger.warning("Failed to compute superlatives", exc_info=True)
            superlatives = {}

        try:
            highlights = _compute_gw_highlights(league_id, max_gw)
        except Exception:
            _logger.warning("Failed to compute GW highlights", exc_info=True)
            highlights = {}

        try:
            draft_data = _compute_league_draft_board(league_id)
        except Exception:
            _logger.warning("Failed to compute draft board", exc_info=True)
            draft_data = {}

        try:
            transfer_data = _compute_league_transfer_stats(league_id, max_gw)
        except Exception:
            _logger.warning("Failed to compute transfer stats", exc_info=True)
            transfer_data = {}

    # ---------------------------------------------------------------------------
    # Render sections
    # ---------------------------------------------------------------------------

    # Part 1: League Champion
    _section_header("🏆", "League Champion", "The 2025/26 Draft League final standings")
    try:
        _render_champion_banner(league_data)
    except Exception as exc:
        _logger.warning("champion banner failed", exc_info=True)
        st.warning(f"Could not render champion banner: {exc}")
    st.markdown("---")

    # Part 2: Season Journey
    _section_header("📈", "Season Journey", "Cumulative points race and league position over the season")
    try:
        _render_season_journey(history_df)
    except Exception as exc:
        _logger.warning("season journey failed", exc_info=True)
        st.warning(f"Could not render season journey: {exc}")
    st.markdown("---")

    # Part 3: League Awards
    _section_header("🎖️", "League Awards", "Eight superlatives celebrating the best (and worst) of the season")
    try:
        _render_league_awards(superlatives, history_df)
    except Exception as exc:
        _logger.warning("league awards failed", exc_info=True)
        st.warning(f"Could not render league awards: {exc}")
    st.markdown("---")

    # Part 4: Gameweek Highlights
    _section_header("⚡", "Gameweek Highlights", "The most memorable moments of the season")
    try:
        _render_gw_highlights(highlights)
    except Exception as exc:
        _logger.warning("gw highlights failed", exc_info=True)
        st.warning(f"Could not render GW highlights: {exc}")
    st.markdown("---")

    # Part 5: Head-to-Head Records
    _section_header("⚔️", "Head-to-Head Records", "Full W-D-L matrix across all league matchups")
    try:
        _render_h2h_records(league_data)
    except Exception as exc:
        _logger.warning("h2h records failed", exc_info=True)
        st.warning(f"Could not render H2H records: {exc}")
    st.markdown("---")

    # Part 6: Draft Board Retrospective
    _section_header("🃏", "Draft Board Retrospective", "League-wide draft steals and busts")
    try:
        _render_draft_board(draft_data)
    except Exception as exc:
        _logger.warning("draft board failed", exc_info=True)
        st.warning(f"Could not render draft board: {exc}")
    st.markdown("---")

    # Part 7: Transfer Window
    _section_header("🔀", "Transfer Window", "Who was most active and who won (or lost) the transfer game")
    try:
        _render_transfer_window(transfer_data)
    except Exception as exc:
        _logger.warning("transfer window failed", exc_info=True)
        st.warning(f"Could not render transfer window: {exc}")
    st.markdown("---")

    # Part 8: Lineup Management
    _section_header("🧩", "Lineup Management", "Bench points missed — who managed their squad best?")
    try:
        _render_lineup_management(bench_data_list)
    except Exception as exc:
        _logger.warning("lineup management failed", exc_info=True)
        st.warning(f"Could not render lineup management: {exc}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("📄 Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF... (this may take ~30 seconds)"):
            try:
                pdf_bytes = generate_league_wrapped_pdf(
                    league_data=league_data,
                    history_df=history_df,
                    bench_data_list=bench_data_list,
                    superlatives=superlatives,
                    highlights=highlights,
                    draft_data=draft_data,
                    transfer_data=transfer_data,
                )
                st.download_button(
                    label="⬇️ Download League Wrapped PDF",
                    data=pdf_bytes,
                    file_name="league_wrapped_2025_26.pdf",
                    mime="application/pdf",
                )
            except Exception as exc:
                _logger.error("PDF generation failed", exc_info=True)
                st.error(f"PDF generation failed: {exc}")


# ---------------------------------------------------------------------------
# Combined entry point (tabs: League + My Team)
# ---------------------------------------------------------------------------

def show_wrapped_page():
    """Single entry point with tabs for League Wrapped and My Team Wrapped."""
    tab_league, tab_team = st.tabs(["🏆  League Wrapped", "🎬  My Team Wrapped"])
    with tab_league:
        show_league_wrapped_page()
    with tab_team:
        show_season_wrapped_page()
