"""
FPL Draft Season Wrapped — end-of-season summary page.

Structured as a 7-part narrative:
  1. Season Overview  (hero + journey chart + rank chart + stat cards)
  2. Your Squad       (Captain's Armband MVP + Season Best XI)
  3. Your Decisions   (formation stats + lineup management)
  4. Transfer Window  (best/worst moves + league-wide transfer leaders)
  5. Draft Retrospective (pick grades, steals/busts, retention)
  6. Looking Back     (cross-season Draft history if configured)
  7. League Awards    (5 superlatives across all teams)
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import config
from scripts.common.bench_analysis import (
    compute_draft_bench_data,
    compute_draft_league_bench_data,
    render_bench_analysis,
)
from scripts.common.fpl_classic_api import get_classic_bootstrap_static
from scripts.common.fpl_draft_api import (
    _get_draft_entry_full_picks_for_gw,
    _get_draft_gw_live_points,
    get_current_gameweek,
    get_draft_league_details,
    get_draft_team_players_with_points,
    get_fpl_player_mapping,
    get_league_teams,
    get_team_composition_for_gameweek,
    get_team_id_by_name,
    get_waiver_transactions_up_to_gameweek,
    pull_fpl_player_stats,
)
from scripts.common.luck_analysis import (
    calculate_all_play_standings,
    extract_draft_gw_scores,
)
from scripts.common.player_matching import canonical_normalize
from scripts.common.team_analysis_helpers import (
    render_season_best_11,
    render_team_mvp,
)
from scripts.draft.home import build_draft_history_df

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PURPLE = "#7B2FBE"
_PURPLE_FADED = "rgba(103,58,183,0.4)"
_GOLD = "#FFD700"
_RED = "#FF4B4B"
_GREEN = "#00ff87"

_DARK_CHART_LAYOUT = dict(
    paper_bgcolor="#1a1a2e",
    plot_bgcolor="#1a1a2e",
    font=dict(color="#ffffff", size=13),
    title=dict(font=dict(size=20, color="#ffffff"), x=0.5, xanchor="center"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=12)),
)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _stat_card(label: str, value: str, accent: str = _PURPLE, sub: str = "") -> str:
    sub_html = f'<div style="color:#888;font-size:11px;margin-top:4px;">{sub}</div>' if sub else ""
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;color:#e0e0e0;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'{sub_html}</div>'
    )


def _award_card(icon: str, title: str, team: str, detail: str, accent: str = _PURPLE) -> str:
    return (
        f'<div style="border:1px solid {accent};border-radius:10px;padding:16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;color:#e0e0e0;height:100%;">'
        f'<div style="font-size:2em;margin-bottom:8px;">{icon}</div>'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{title}</div>'
        f'<div style="color:{accent};font-size:16px;font-weight:700;margin-bottom:4px;">{team}</div>'
        f'<div style="color:#888;font-size:12px;">{detail}</div>'
        f'</div>'
    )


def _section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<p style="color:#9ca3af;margin-top:4px;font-size:14px;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<h2 style="color:{_PURPLE};border-bottom:2px solid {_PURPLE};padding-bottom:8px;">'
        f'{icon} {title}</h2>{sub_html}',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data: Formation stats
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_formation_stats(entry_id: int, max_gw: int) -> Dict[str, int]:
    """Return {formation_str: gws_used} for all GWs 1..max_gw."""
    player_map = get_fpl_player_mapping()
    counts: Dict[str, int] = {}

    for gw in range(1, max_gw + 1):
        picks = _get_draft_entry_full_picks_for_gw(entry_id, gw)
        if not picks:
            continue
        starters = [p for p in picks if p.get("position", 99) <= 11]
        pos_counts: Dict[str, int] = Counter(
            player_map.get(p["element"], {}).get("Position", "M")
            for p in starters
            if player_map.get(p["element"], {}).get("Position") not in ("G", None)
        )
        d = pos_counts.get("D", 0)
        m = pos_counts.get("M", 0)
        f = pos_counts.get("F", 0)
        if d + m + f > 0:
            key = f"{d}-{m}-{f}"
            counts[key] = counts.get(key, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Data: Transfer stats
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_transfer_stats(
    entry_id: int, league_id: int, max_gw: int
) -> Tuple[List[Dict], Dict[str, int], Dict[str, int]]:
    """
    Returns:
      - transfers: list of dicts per transaction for this team
      - most_in: {player_name: count} league-wide transfers in
      - most_out: {player_name: count} league-wide transfers out
    """
    player_map = get_fpl_player_mapping()
    all_transactions = get_waiver_transactions_up_to_gameweek(league_id, max_gw)

    # Pre-compute GW live points for all GWs (SQLite-cached for finished GWs)
    gw_points_cache: Dict[int, Dict[int, int]] = {}
    for gw in range(1, max_gw + 1):
        gw_points_cache[gw] = _get_draft_gw_live_points(gw)

    def _pts_after_trade(element_id: int, trade_gw: int) -> int:
        total = 0
        for gw in range(trade_gw + 1, max_gw + 1):
            total += gw_points_cache.get(gw, {}).get(element_id, 0)
        return total

    def _player_name(element_id: int) -> str:
        info = player_map.get(element_id, {})
        return info.get("Player") or info.get("Web_Name") or f"Player {element_id}"

    transfers = []
    for tx in all_transactions:
        if tx.get("result") != "a":
            continue
        trade_gw = tx.get("event", 0)
        el_in = tx.get("element_in")
        el_out = tx.get("element_out")
        if not el_in or not el_out:
            continue

        if tx.get("entry") == entry_id:
            pts_in = _pts_after_trade(el_in, trade_gw)
            pts_out = _pts_after_trade(el_out, trade_gw)
            transfers.append({
                "gw": trade_gw,
                "player_in": _player_name(el_in),
                "player_out": _player_name(el_out),
                "pts_in": pts_in,
                "pts_out": pts_out,
                "net": pts_in - pts_out,
            })

    transfers.sort(key=lambda x: -x["net"])

    # League-wide most transferred in/out
    most_in: Dict[str, int] = {}
    most_out: Dict[str, int] = {}
    for tx in all_transactions:
        if tx.get("result") != "a":
            continue
        el_in = tx.get("element_in")
        el_out = tx.get("element_out")
        if el_in:
            name = _player_name(el_in)
            most_in[name] = most_in.get(name, 0) + 1
        if el_out:
            name = _player_name(el_out)
            most_out[name] = most_out.get(name, 0) + 1

    return transfers, most_in, most_out


# ---------------------------------------------------------------------------
# Data: Draft analysis
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_draft_analysis(
    entry_id: int, league_id: int, num_teams: int
) -> Optional[Dict]:
    """
    Returns:
      - my_picks: list of {round, pick, player, pts, avg_round_pts, delta, grade}
      - retention_count: int (how many original picks still on roster)
      - retention_total: int (total original picks = 15)
    """
    try:
        draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
        choices_raw = requests.get(draft_url, timeout=30).json().get("choices", [])
    except Exception:
        return None

    if not choices_raw:
        return None

    # Build all picks: {overall_pick, player_id, team_id}
    all_choices = []
    for c in choices_raw:
        pick_num = c.get("pick")
        player_id = c.get("element")
        team_id = c.get("entry")
        if pick_num is None or not player_id or not team_id:
            continue
        all_choices.append({"pick": int(pick_num), "player_id": int(player_id), "team_id": int(team_id)})

    if not all_choices:
        return None

    # Get season points for all players from Classic FPL stats
    try:
        fpl_stats = pull_fpl_player_stats()
    except Exception:
        fpl_stats = pd.DataFrame()

    def _get_season_pts(player_id: int) -> int:
        player_map = get_fpl_player_mapping()
        info = player_map.get(player_id, {})
        full_name = info.get("Player", "")
        if fpl_stats.empty or not full_name:
            return 0
        norm_name = canonical_normalize(full_name)
        mask = fpl_stats["player"].apply(lambda n: canonical_normalize(str(n))) == norm_name
        matched = fpl_stats[mask]
        if not matched.empty:
            return int(matched.iloc[0].get("total_points", 0))
        # Fallback: last-word match
        last_word = norm_name.split()[-1] if norm_name else ""
        if last_word:
            mask2 = fpl_stats["player"].apply(
                lambda n: canonical_normalize(str(n)).split()[-1] if n else ""
            ) == last_word
            matched2 = fpl_stats[mask2]
            if not matched2.empty:
                return int(matched2.iloc[0].get("total_points", 0))
        return 0

    # Compute round average pts by round number
    round_pts: Dict[int, List[int]] = {}
    for c in all_choices:
        r = (c["pick"] - 1) // num_teams + 1
        pts = _get_season_pts(c["player_id"])
        round_pts.setdefault(r, []).append(pts)
    round_avg = {r: sum(v) / len(v) for r, v in round_pts.items()}

    # My picks
    player_map = get_fpl_player_mapping()
    my_choices = [c for c in all_choices if c["team_id"] == entry_id]
    my_picks = []
    for c in sorted(my_choices, key=lambda x: x["pick"]):
        r = (c["pick"] - 1) // num_teams + 1
        pts = _get_season_pts(c["player_id"])
        avg = round_avg.get(r, pts)
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
        my_picks.append({
            "round": r,
            "pick": c["pick"],
            "player": info.get("Player", f"ID {c['player_id']}"),
            "pts": pts,
            "avg_round_pts": round(avg, 1),
            "delta": round(delta, 1),
            "grade": grade,
        })

    # Retention: how many original picks are still on roster
    max_gw = min(get_current_gameweek(), 38)
    try:
        current_roster_df = get_team_composition_for_gameweek(league_id, entry_id, max_gw)
        current_names = set(
            canonical_normalize(n) for n in current_roster_df["Player"].tolist()
        ) if not current_roster_df.empty else set()
    except Exception:
        current_names = set()

    original_names = set(
        canonical_normalize(player_map.get(c["player_id"], {}).get("Player", ""))
        for c in my_choices
    )
    retention = sum(1 for n in original_names if n and n in current_names)

    return {
        "my_picks": my_picks,
        "retention_count": retention,
        "retention_total": len(my_choices),
    }


# ---------------------------------------------------------------------------
# Data: Draft season history
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _load_draft_season_history(team_name: str, current_league_id: int) -> List[Dict]:
    """
    Load cross-season Draft history for a team.

    Uses FPL_DRAFT_LEAGUE_HISTORY config for past seasons + current league for current season.
    Matches team by name (since entry_id may differ year to year).
    """
    history = []

    def _extract_team_standing(league_response: dict, name: str) -> Optional[Dict]:
        entries = league_response.get("league_entries", [])
        standings = league_response.get("standings", [])
        entry_map = {e["id"]: e["entry_name"] for e in entries}
        # Find entry_id for this team name (case-insensitive)
        target_id = None
        for eid, ename in entry_map.items():
            if ename.strip().lower() == name.strip().lower():
                target_id = eid
                break
        if target_id is None:
            return None
        for row in standings:
            if row.get("league_entry") == target_id:
                return {
                    "total_points": row.get("points_for", 0),
                    "rank": row.get("rank", "?"),
                    "wins": row.get("matches_won", 0),
                    "draws": row.get("matches_drawn", 0),
                    "losses": row.get("matches_lost", 0),
                }
        return None

    # Past configured seasons
    for season_label, hist_league_id in config.FPL_DRAFT_LEAGUE_HISTORY:
        try:
            league_data = get_draft_league_details(hist_league_id)
            if not league_data:
                continue
            standing = _extract_team_standing(league_data, team_name)
            if standing:
                standing["season"] = season_label
                history.append(standing)
        except Exception:
            continue

    # Current season
    try:
        current_data = get_draft_league_details(current_league_id)
        if current_data:
            standing = _extract_team_standing(current_data, team_name)
            if standing:
                standing["season"] = "2025/26"
                history.append(standing)
    except Exception:
        pass

    return sorted(history, key=lambda x: x.get("season", ""))


# ---------------------------------------------------------------------------
# Data: League superlatives
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _compute_league_superlatives(league_id: int, max_gw: int) -> Dict:
    """Compute 5 league-wide awards."""
    league_data = get_draft_league_details(league_id)
    if not league_data:
        return {}

    entries = league_data.get("league_entries", [])
    team_names = {e["id"]: e["entry_name"] for e in entries}

    # 1. Most Active: count approved transactions per team
    all_tx = get_waiver_transactions_up_to_gameweek(league_id, max_gw)
    tx_counts: Dict[int, int] = {}
    for tx in all_tx:
        if tx.get("result") == "a":
            eid = tx.get("entry")
            if eid:
                tx_counts[eid] = tx_counts.get(eid, 0) + 1
    # Map entry_id to league_entry_id using entries
    entry_id_map = {e.get("entry_id"): e["id"] for e in entries if e.get("entry_id")}
    tx_by_league_id = {entry_id_map.get(eid, eid): cnt for eid, cnt in tx_counts.items()}
    most_active_id = max(tx_by_league_id, key=lambda k: tx_by_league_id[k]) if tx_by_league_id else None
    most_active = {
        "team": team_names.get(most_active_id, "?"),
        "value": tx_by_league_id.get(most_active_id, 0),
    }

    # 2. Best Lineup Manager: Selection % from bench data
    bench_league = compute_draft_league_bench_data(league_id, max_gw)
    best_mgr_row = max(bench_league, key=lambda r: r.get("Selection %", 0)) if bench_league else None
    best_mgr = {
        "team": best_mgr_row["Team"] if best_mgr_row else "?",
        "value": f'{best_mgr_row["Selection %"]:.1f}%' if best_mgr_row else "?",
    }

    # 3. Luck: All-Play standings
    gw_scores = extract_draft_gw_scores(league_data)
    standings_df = calculate_all_play_standings(gw_scores)
    luckiest = {"team": "?", "value": ""}
    unluckiest = {"team": "?", "value": ""}
    if not standings_df.empty and "Luck +/-" in standings_df.columns:
        luck_col = standings_df["Luck +/-"].apply(
            lambda x: int(str(x).replace("+", "")) if pd.notna(x) else 0
        )
        standings_df = standings_df.copy()
        standings_df["_luck_int"] = luck_col
        luckiest_row = standings_df.loc[standings_df["_luck_int"].idxmin()]
        unluckiest_row = standings_df.loc[standings_df["_luck_int"].idxmax()]
        luck_val = luckiest_row["_luck_int"]
        luckiest = {
            "team": luckiest_row["Team"],
            "value": f"{abs(luck_val)} rank places lucky" if luck_val < 0 else "0 rank delta",
        }
        unluck_val = unluckiest_row["_luck_int"]
        unluckiest = {
            "team": unluckiest_row["Team"],
            "value": f"{unluck_val} rank places unlucky",
        }

    # 4. Best Drafter: sum season points of ORIGINAL draft picks only
    try:
        fpl_stats = pull_fpl_player_stats()
        player_map = get_fpl_player_mapping()
        draft_url = f"https://draft.premierleague.com/api/draft/{league_id}/choices"
        choices = requests.get(draft_url, timeout=30).json().get("choices", [])

        def _norm(name: str) -> str:
            return canonical_normalize(name)

        pts_lookup: Dict[str, int] = {}
        if not fpl_stats.empty:
            for _, row in fpl_stats.iterrows():
                pts_lookup[_norm(str(row.get("player", "")))] = int(row.get("total_points", 0))

        draft_pts_by_entry: Dict[int, int] = {}
        for c in choices:
            eid_raw = c.get("entry")
            pid = c.get("element")
            if not eid_raw or not pid:
                continue
            pname = player_map.get(pid, {}).get("Player", "")
            pts = pts_lookup.get(_norm(pname), 0)
            draft_pts_by_entry[eid_raw] = draft_pts_by_entry.get(eid_raw, 0) + pts

        # Map entry_id → league_entry_id
        entry_id_to_league = {e.get("entry_id"): e["id"] for e in entries if e.get("entry_id")}
        draft_pts_league_id = {
            entry_id_to_league.get(eid, eid): pts
            for eid, pts in draft_pts_by_entry.items()
        }
        best_drafter_id = max(draft_pts_league_id, key=lambda k: draft_pts_league_id[k]) if draft_pts_league_id else None
        best_drafter = {
            "team": team_names.get(best_drafter_id, "?"),
            "value": f"{draft_pts_league_id.get(best_drafter_id, 0):,} pts from draft picks",
        }
    except Exception:
        best_drafter = {"team": "?", "value": ""}

    return {
        "most_active": most_active,
        "best_mgr": best_mgr,
        "luckiest": luckiest,
        "unluckiest": unluckiest,
        "best_drafter": best_drafter,
    }


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

def show_season_wrapped_page():
    st.title("Season Wrapped 🎬")
    st.write("Your complete 2025/26 FPL Draft season in review.")

    # Team selector
    try:
        team_dict = get_league_teams(config.FPL_DRAFT_LEAGUE_ID)  # {entry_id: team_name}
    except Exception:
        st.error("Could not load league teams. Check your FPL_DRAFT_LEAGUE_ID config.")
        return

    if not team_dict:
        st.warning("No teams found in your Draft league.")
        return

    team_list = list(team_dict.values())
    default_team = team_dict.get(config.FPL_DRAFT_TEAM_ID, team_list[0])
    try:
        default_idx = team_list.index(default_team)
    except ValueError:
        default_idx = 0

    selected_team = st.selectbox("View Season Wrapped for:", team_list, index=default_idx)
    entry_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, selected_team)
    num_teams = len(team_dict)

    _raw_gw = get_current_gameweek()
    max_gw = min(int(_raw_gw), 38) if _raw_gw is not None else 38
    if max_gw < 1:
        st.info("No gameweek data available yet.")
        return

    # Load core data with spinner
    with st.spinner("Building your Season Wrapped..."):
        history_df = build_draft_history_df(config.FPL_DRAFT_LEAGUE_ID)
        league_data = get_draft_league_details(config.FPL_DRAFT_LEAGUE_ID)
        player_data_dict = get_draft_team_players_with_points(config.FPL_DRAFT_LEAGUE_ID)
        bootstrap = get_classic_bootstrap_static()
        bench_data = compute_draft_bench_data(entry_id, max_gw)

    team_history = pd.DataFrame()
    if not history_df.empty:
        team_history = history_df[history_df["Team"] == selected_team].sort_values("Gameweek")

    team_players = player_data_dict.get(selected_team, [])

    # Standings for this team
    standings = league_data.get("standings", []) if league_data else []
    entries = league_data.get("league_entries", []) if league_data else []
    entry_id_map = {e["entry_name"]: e["id"] for e in entries}
    league_entry_id = entry_id_map.get(selected_team)
    final_rank = "?"
    total_pts = 0
    wdl = "?"
    for row in standings:
        if row.get("league_entry") == league_entry_id:
            final_rank = row.get("rank", "?")
            total_pts = row.get("points_for", 0)
            w = row.get("matches_won", 0)
            d = row.get("matches_drawn", 0)
            lv = row.get("matches_lost", 0)
            wdl = f"{w}W-{d}D-{lv}L"
            break

    avg_pts = round(total_pts / max_gw, 1) if max_gw > 0 else 0

    # Seasons of Draft played
    draft_history = _load_draft_season_history(selected_team, config.FPL_DRAFT_LEAGUE_ID)
    seasons_played = len(draft_history) if draft_history else 1

    st.divider()

    # =========================================================================
    # PART 1: THE SEASON OVERVIEW
    # =========================================================================
    _section_header("📊", "The Season Overview", "How did the season play out?")

    # Hero banner
    hero_html = (
        f'<div style="background:linear-gradient(135deg,#1a1a2e 0%,#2d1b69 50%,#16213e 100%);'
        f'border:2px solid {_PURPLE};border-radius:14px;padding:30px;text-align:center;color:#e0e0e0;margin-bottom:20px;">'
        f'<div style="font-size:2.5em;font-weight:800;color:#ffffff;margin-bottom:6px;">{selected_team}</div>'
        f'<div style="color:#9ca3af;font-size:14px;letter-spacing:1px;text-transform:uppercase;margin-bottom:20px;">'
        f'2025/26 FPL Draft Season</div>'
        f'<div style="display:flex;justify-content:center;gap:40px;flex-wrap:wrap;">'
        f'<div><div style="color:{_GOLD};font-size:2em;font-weight:700;">{final_rank}<span style="color:#888;font-size:0.6em;">/{num_teams}</span></div>'
        f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;">Final Rank</div></div>'
        f'<div><div style="color:{_GREEN};font-size:2em;font-weight:700;">{total_pts:,}</div>'
        f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;">Total Points</div></div>'
        f'<div><div style="color:{_PURPLE};font-size:2em;font-weight:700;">{avg_pts}</div>'
        f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;">Avg Pts/GW</div></div>'
        f'<div><div style="color:#04f5ff;font-size:2em;font-weight:700;">{seasons_played}</div>'
        f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;">Draft Seasons</div></div>'
        f'<div><div style="color:#e0e0e0;font-size:1.6em;font-weight:600;">{wdl}</div>'
        f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;">Record</div></div>'
        f'</div></div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    # Season Journey chart
    if not team_history.empty:
        gws = team_history["Gameweek"].tolist()
        pts_list = team_history["GW_Points"].tolist()
        avg_gw = sum(pts_list) / len(pts_list) if pts_list else 0
        best_gw_idx = pts_list.index(max(pts_list))
        worst_gw_idx = pts_list.index(min(pts_list))

        bar_colors = []
        for i, pt in enumerate(pts_list):
            if i == best_gw_idx:
                bar_colors.append(_GOLD)
            elif i == worst_gw_idx:
                bar_colors.append(_RED)
            elif pt < avg_gw:
                bar_colors.append(_PURPLE_FADED)
            else:
                bar_colors.append(_PURPLE)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=gws,
            y=pts_list,
            marker_color=bar_colors,
            hovertemplate="GW %{x}: %{y} pts<extra></extra>",
        ))
        fig.add_hline(
            y=avg_gw,
            line_dash="dash",
            line_color="rgba(255,255,255,0.6)",
            annotation_text=f"Avg: {avg_gw:.1f}",
            annotation_position="right",
            annotation_font_color="#ffffff",
        )
        fig.update_layout(
            **_DARK_CHART_LAYOUT,
            title="Your Season Journey",
            height=350,
            showlegend=False,
        )
        fig.update_xaxes(title="Gameweek", dtick=2)
        fig.update_yaxes(title="Points")
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # League position chart
        rank_list = team_history["League_Position"].tolist()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=gws,
            y=rank_list,
            mode="lines+markers",
            line=dict(color=_PURPLE, width=3),
            marker=dict(size=6, color=_PURPLE),
            hovertemplate="GW %{x}: Rank %{y}<extra></extra>",
        ))
        fig2.update_layout(
            **_DARK_CHART_LAYOUT,
            title="Your League Position Over the Season",
            height=300,
            showlegend=False,
        )
        fig2.update_xaxes(title="Gameweek", dtick=2)
        fig2.update_yaxes(title="League Position", autorange="reversed", dtick=1)
        st.plotly_chart(fig2, use_container_width=True, theme=None)

    # Stat cards — 2 rows of 4
    if not team_history.empty:
        best_gw_row = team_history.loc[team_history["GW_Points"].idxmax()]
        worst_gw_row = team_history.loc[team_history["GW_Points"].idxmin()]
        best_rank = int(team_history["League_Position"].min())
        worst_rank = int(team_history["League_Position"].max())
    else:
        best_gw_row = worst_gw_row = None
        best_rank = worst_rank = 0

    total_transfers = 0
    try:
        all_tx = get_waiver_transactions_up_to_gameweek(config.FPL_DRAFT_LEAGUE_ID, max_gw)
        # Need to map entry_id to league_entry_id
        entry_id_to_lid = {e.get("entry_id"): e["id"] for e in entries if e.get("entry_id")}
        my_league_entries = {e_id for e_id, lid in entry_id_to_lid.items() if lid == league_entry_id}
        my_league_entries.add(entry_id)  # include both forms
        total_transfers = sum(
            1 for tx in all_tx
            if tx.get("result") == "a" and tx.get("entry") in my_league_entries
        )
    except Exception:
        pass

    perfect_gws = 0
    pts_lost_total = 0
    worst_lineup_gw = None
    if bench_data:
        perfect_gws = sum(1 for g in bench_data["per_gw"] if g["points_lost"] == 0)
        pts_lost_total = bench_data["total_points_lost"]
        eligible = [g for g in bench_data["per_gw"] if g.get("active_chip") not in ("bboost", "freehit")]
        if eligible:
            worst_lineup_gw = max(eligible, key=lambda g: g["points_lost"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        label = f"GW{int(best_gw_row['Gameweek'])}: {int(best_gw_row['GW_Points'])} pts" if best_gw_row is not None else "—"
        st.markdown(_stat_card("Best GW", label, _GOLD), unsafe_allow_html=True)
    with col2:
        label = f"GW{int(worst_gw_row['Gameweek'])}: {int(worst_gw_row['GW_Points'])} pts" if worst_gw_row is not None else "—"
        st.markdown(_stat_card("Worst GW", label, _RED), unsafe_allow_html=True)
    with col3:
        st.markdown(_stat_card("Best League Rank", f"#{best_rank}", _GREEN), unsafe_allow_html=True)
    with col4:
        st.markdown(_stat_card("Worst League Rank", f"#{worst_rank}", "#9ca3af"), unsafe_allow_html=True)

    st.markdown("")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(_stat_card("Total Transfers", str(total_transfers), _PURPLE), unsafe_allow_html=True)
    with col6:
        st.markdown(_stat_card("Perfect GWs", str(perfect_gws), _GREEN, "Optimal lineup set"), unsafe_allow_html=True)
    with col7:
        st.markdown(_stat_card("Points Lost to Bench", str(pts_lost_total), _RED), unsafe_allow_html=True)
    with col8:
        wlgw_label = f"GW{worst_lineup_gw['gw']}: {worst_lineup_gw['points_lost']} pts" if worst_lineup_gw else "—"
        st.markdown(_stat_card("Worst Lineup GW", wlgw_label, _RED), unsafe_allow_html=True)

    st.divider()

    # =========================================================================
    # PART 2: YOUR SQUAD
    # =========================================================================
    _section_header("⚽", "Your Squad", "The players that defined your season")

    if team_players:
        col_mvp, col_xi = st.columns(2)
        with col_mvp:
            st.markdown("#### 🏆 Captain's Armband")
            render_team_mvp(team_players, bootstrap_data=bootstrap, team_id=None, is_classic=False)
        with col_xi:
            st.markdown("#### ⭐ Season Best XI")
            render_season_best_11(team_players)
    else:
        st.info("No player data available for this team yet.")

    st.divider()

    # =========================================================================
    # PART 3: YOUR DECISIONS
    # =========================================================================
    _section_header("🧠", "Your Decisions", "How well did you manage the team each week?")

    # Formation stats
    st.markdown("#### 📐 Most Used Formation")
    try:
        player_map = get_fpl_player_mapping()
        formation_counts = _compute_formation_stats(entry_id, max_gw)
        if formation_counts:
            top_formation = next(iter(formation_counts))
            top_count = formation_counts[top_formation]

            top_html = (
                f'<div style="border:2px solid {_PURPLE};border-radius:12px;padding:24px;'
                f'background:linear-gradient(135deg,#1a1a2e 0%,#2d1b69 100%);text-align:center;margin-bottom:16px;">'
                f'<div style="color:#9ca3af;font-size:12px;text-transform:uppercase;letter-spacing:1px;">Most Used Formation</div>'
                f'<div style="color:{_PURPLE};font-size:3em;font-weight:800;margin:8px 0;">{top_formation}</div>'
                f'<div style="color:#e0e0e0;font-size:14px;">{top_count} of {max_gw} gameweeks</div>'
                f'</div>'
            )
            st.markdown(top_html, unsafe_allow_html=True)

            # Formation breakdown table
            rows = [{"Formation": k, "Gameweeks": v, "% of Season": f"{v/max_gw*100:.0f}%"}
                    for k, v in formation_counts.items()]
            df_form = pd.DataFrame(rows)
            st.dataframe(df_form, hide_index=True, use_container_width=False, height=38 + len(df_form) * 35)
        else:
            st.info("Formation data not available.")
    except Exception as e:
        st.info(f"Formation data could not be computed: {e}")

    # Lineup management
    st.markdown("#### 📋 Lineup Management")
    if bench_data:
        render_bench_analysis(bench_data, is_classic=False)
    else:
        st.info("Lineup data not available.")

    st.divider()

    # =========================================================================
    # PART 4: YOUR TRANSFER WINDOW
    # =========================================================================
    _section_header("🔄", "Transfer Window", "How did your waiver wire moves shape the season?")

    try:
        transfers, most_in, most_out = _compute_transfer_stats(entry_id, config.FPL_DRAFT_LEAGUE_ID, max_gw)

        if transfers:
            best_tx = transfers[0]
            worst_tx = max(transfers, key=lambda x: -(x["net"]))

            col_b, col_w = st.columns(2)
            with col_b:
                st.markdown(
                    _stat_card(
                        "Best Transfer In",
                        best_tx["player_in"],
                        _GREEN,
                        f"+{best_tx['pts_in']} pts post-trade (net {best_tx['net']:+d})",
                    ),
                    unsafe_allow_html=True,
                )
            with col_w:
                worst_out = max(transfers, key=lambda x: x["pts_out"])
                st.markdown(
                    _stat_card(
                        "Worst Transfer Out",
                        worst_out["player_out"],
                        _RED,
                        f"{worst_out['pts_out']} pts scored after you dropped them",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<p style="color:#9ca3af;font-size:12px;font-style:italic;margin-top:8px;">'
                "Points In = pts scored by acquired player after the trade GW. "
                "Points Out = pts scored by dropped player after the trade GW. "
                "Net = In minus Out.</p>",
                unsafe_allow_html=True,
            )

            # Full transfer table
            with st.expander("All Transfers This Season", expanded=False):
                df_tx = pd.DataFrame(transfers).rename(columns={
                    "gw": "GW", "player_in": "Player In", "player_out": "Player Out",
                    "pts_in": "Pts In", "pts_out": "Pts Out", "net": "Net",
                })
                st.dataframe(df_tx, hide_index=True, use_container_width=True,
                             height=38 + len(df_tx) * 35)
        else:
            st.info("No transfers made this season.")

        # League-wide most transferred
        if most_in or most_out:
            col_mi, col_mo = st.columns(2)
            with col_mi:
                st.markdown("##### 📈 Most Transferred In (League)")
                top_in = sorted(most_in.items(), key=lambda x: -x[1])[:5]
                if top_in:
                    df_in = pd.DataFrame(top_in, columns=["Player", "Times Acquired"])
                    st.dataframe(df_in, hide_index=True, use_container_width=True,
                                 height=38 + len(df_in) * 35)
            with col_mo:
                st.markdown("##### 📉 Most Transferred Out (League)")
                top_out = sorted(most_out.items(), key=lambda x: -x[1])[:5]
                if top_out:
                    df_out = pd.DataFrame(top_out, columns=["Player", "Times Dropped"])
                    st.dataframe(df_out, hide_index=True, use_container_width=True,
                                 height=38 + len(df_out) * 35)
    except Exception as e:
        st.info(f"Transfer data could not be computed: {e}")

    st.divider()

    # =========================================================================
    # PART 5: DRAFT RETROSPECTIVE
    # =========================================================================
    _section_header("🧬", "Draft Retrospective", "How did your pre-season draft hold up?")

    try:
        draft_data = _compute_draft_analysis(entry_id, config.FPL_DRAFT_LEAGUE_ID, num_teams)
        if draft_data and draft_data["my_picks"]:
            picks = draft_data["my_picks"]
            retention = draft_data["retention_count"]
            total_orig = draft_data["retention_total"]

            avg_delta = sum(p["delta"] for p in picks) / len(picks)
            steals = sorted([p for p in picks if "Steal" in p["grade"] or "Value" in p["grade"]],
                            key=lambda x: -x["delta"])[:3]
            busts = sorted([p for p in picks if "Bust" in p["grade"] or "Miss" in p["grade"]],
                           key=lambda x: x["delta"])[:3]

            col_r, col_d = st.columns(2)
            with col_r:
                st.markdown(
                    _stat_card("Retention Rate", f"{retention}/{total_orig}", _PURPLE,
                               "Original draft picks still on roster"),
                    unsafe_allow_html=True,
                )
            with col_d:
                sign = "+" if avg_delta >= 0 else ""
                color = _GREEN if avg_delta >= 0 else _RED
                st.markdown(
                    _stat_card("Avg Pick Delta", f"{sign}{avg_delta:.1f} pts", color,
                               "vs. league avg for that draft round"),
                    unsafe_allow_html=True,
                )

            st.markdown("")
            if steals:
                st.markdown("##### 🔥 Your Steals")
                steal_cols = st.columns(min(3, len(steals)))
                for i, p in enumerate(steals):
                    with steal_cols[i]:
                        st.markdown(
                            _stat_card(f"Rd {p['round']} Pick {p['pick']}", p["player"],
                                       _GREEN, f"{p['pts']} pts (+{p['delta']:.0f} vs avg)"),
                            unsafe_allow_html=True,
                        )
            if busts:
                st.markdown("##### 💀 Your Busts")
                bust_cols = st.columns(min(3, len(busts)))
                for i, p in enumerate(busts):
                    with bust_cols[i]:
                        st.markdown(
                            _stat_card(f"Rd {p['round']} Pick {p['pick']}", p["player"],
                                       _RED, f"{p['pts']} pts ({p['delta']:.0f} vs avg)"),
                            unsafe_allow_html=True,
                        )

            with st.expander("Full Draft Board", expanded=False):
                df_picks = pd.DataFrame(picks).rename(columns={
                    "round": "Round", "pick": "Overall Pick", "player": "Player",
                    "pts": "Season Pts", "avg_round_pts": "Avg for Round",
                    "delta": "Delta", "grade": "Grade",
                })
                st.dataframe(df_picks, hide_index=True, use_container_width=True,
                             height=38 + len(df_picks) * 35)
        else:
            st.info("Draft analysis data not available.")
    except Exception as e:
        st.info(f"Draft analysis could not be computed: {e}")

    st.divider()

    # =========================================================================
    # PART 6: LOOKING BACK
    # =========================================================================
    _section_header("📅", "Looking Back", "How does this season compare to previous years?")

    if not config.FPL_DRAFT_LEAGUE_HISTORY:
        st.info(
            "Configure `FPL_DRAFT_LEAGUE_HISTORY` in your `.env` to see cross-season Draft history. "
            "Format: `2023/24:league_id,2024/25:league_id` — add each past season's league ID."
        )
    else:
        try:
            season_history = _load_draft_season_history(selected_team, config.FPL_DRAFT_LEAGUE_ID)
            if season_history:
                df_hist = pd.DataFrame(season_history).rename(columns={
                    "season": "Season", "total_points": "Total Points",
                    "rank": "Final Rank", "wins": "W", "draws": "D", "losses": "L",
                })
                df_hist["Record"] = df_hist["W"].astype(str) + "W-" + df_hist["D"].astype(str) + "D-" + df_hist["L"].astype(str) + "L"
                df_hist = df_hist[["Season", "Final Rank", "Total Points", "Record"]]
                st.dataframe(df_hist, hide_index=True, use_container_width=False,
                             height=38 + len(df_hist) * 35)

                # Rank progression chart if multiple seasons
                if len(season_history) > 1:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=[h["season"] for h in season_history],
                        y=[h["rank"] for h in season_history],
                        mode="lines+markers+text",
                        text=[f"#{h['rank']}" for h in season_history],
                        textposition="top center",
                        line=dict(color=_PURPLE, width=3),
                        marker=dict(size=8, color=_PURPLE),
                    ))
                    fig_hist.update_layout(
                        **_DARK_CHART_LAYOUT,
                        title="League Rank by Season",
                        height=300,
                        showlegend=False,
                    )
                    fig_hist.update_yaxes(autorange="reversed", title="Final Rank", dtick=1)
                    st.plotly_chart(fig_hist, use_container_width=True, theme=None)
            else:
                st.info("No historical data found for this team name in the configured seasons.")
        except Exception as e:
            st.info(f"Historical data could not be loaded: {e}")

    st.divider()

    # =========================================================================
    # PART 7: LEAGUE AWARDS
    # =========================================================================
    _section_header("🏆", "League Awards", "Who stood out across the whole league this season?")

    try:
        superlatives = _compute_league_superlatives(config.FPL_DRAFT_LEAGUE_ID, max_gw)
        if superlatives:
            col1, col2, col3 = st.columns(3)
            col4, col5 = st.columns(2)

            with col1:
                st.markdown(
                    _award_card("🔁", "Most Active Manager",
                                superlatives["most_active"]["team"],
                                f"{superlatives['most_active']['value']} transfers", _PURPLE),
                    unsafe_allow_html=True,
                )
            with col2:
                st.markdown(
                    _award_card("🎯", "Best Lineup Manager",
                                superlatives["best_mgr"]["team"],
                                f"{superlatives['best_mgr']['value']} selection accuracy", _GREEN),
                    unsafe_allow_html=True,
                )
            with col3:
                st.markdown(
                    _award_card("🧠", "Best Drafter",
                                superlatives["best_drafter"]["team"],
                                superlatives["best_drafter"]["value"], _GOLD),
                    unsafe_allow_html=True,
                )
            with col4:
                st.markdown(
                    _award_card("🍀", "Luckiest Manager",
                                superlatives["luckiest"]["team"],
                                superlatives["luckiest"]["value"], "#4ade80"),
                    unsafe_allow_html=True,
                )
            with col5:
                st.markdown(
                    _award_card("😢", "Most Unlucky Manager",
                                superlatives["unluckiest"]["team"],
                                superlatives["unluckiest"]["value"], _RED),
                    unsafe_allow_html=True,
                )
        else:
            st.info("League superlatives could not be computed.")
    except Exception as e:
        st.info(f"League awards could not be computed: {e}")
