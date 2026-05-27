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
    get_team_mvp,
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
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff", size=12)),
)

def _chart_title(text: str) -> dict:
    return dict(text=text, font=dict(size=20, color="#ffffff"), x=0.5, xanchor="center")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _stat_card(label: str, value: str, accent: str = _PURPLE, sub: str = "") -> str:
    # Always render sub line (invisible placeholder when empty) so all cards share the same height.
    sub_content = sub if sub else "&nbsp;"
    sub_color = "#888" if sub else "transparent"
    return (
        f'<div style="border:1px solid #333;border-radius:10px;padding:14px 16px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;color:#e0e0e0;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
        f'<div style="color:{sub_color};font-size:11px;margin-top:4px;">{sub_content}</div>'
        f'</div>'
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


def _render_wrapped_mvp(team_players: list, bootstrap_data: Optional[dict]) -> None:
    """Render Captain's Armband MVP card with bonus, clean sheets, and saves stats."""
    mvp = get_team_mvp(team_players, bootstrap_data)
    if not mvp:
        st.info("No MVP data available.")
        return

    bonus = clean_sheets = saves = 0
    if bootstrap_data:
        for elem in bootstrap_data.get("elements", []):
            if elem.get("web_name") == mvp["player"]:
                bonus = elem.get("bonus", 0) or 0
                clean_sheets = elem.get("clean_sheets", 0) or 0
                saves = elem.get("saves", 0) or 0
                break

    pos = mvp.get("position", "")

    def _stat(val, label, color):
        return (
            f'<div style="text-align:center;padding:8px 10px;">'
            f'<div style="color:{color};font-size:1.5em;font-weight:bold;">{val}</div>'
            f'<div style="color:#888;font-size:0.8em;">{label}</div></div>'
        )

    stats = (
        _stat(mvp["starts"], "Starts*", "#9b59b6")
        + _stat(mvp["goals"], "Goals", "#e74c3c")
        + _stat(mvp["assists"], "Assists", "#3498db")
        + _stat(clean_sheets, "Clean Sheets", "#2ecc71")
        + _stat(bonus, "Bonus Pts", "#f39c12")
    )
    if pos == "G":
        stats += _stat(saves, "Saves", "#1abc9c")

    html = (
        f'<div style="border:2px solid #ffd700;border-radius:12px;padding:20px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
        f'<div style="font-size:2em;margin-bottom:6px;">👑</div>'
        f'<div style="color:#ffd700;font-size:1.4em;font-weight:bold;">{mvp["player"]}</div>'
        f'<div style="color:#aaa;font-size:0.95em;margin-bottom:14px;">{mvp["team"]} • {pos}</div>'
        f'<div style="display:flex;justify-content:space-around;flex-wrap:wrap;">{stats}</div>'
        f'<div style="color:#4ecca3;font-size:1.6em;font-weight:bold;margin-top:12px;">'
        f'{mvp["total_points"]} pts</div>'
        f'<div style="color:#666;font-size:0.75em;margin-top:6px;font-style:italic;">'
        f'*Starts = games with 60+ minutes</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _section_header(icon: str, title: str, subtitle: str = ""):
    sub_html = f'<p style="color:#9ca3af;margin-top:4px;font-size:14px;">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<h2 style="color:{_PURPLE};border-bottom:2px solid {_PURPLE};padding-bottom:8px;">'
        f'{icon} {title}</h2>{sub_html}',
        unsafe_allow_html=True,
    )


def _inject_print_styles():
    """Inject CSS that makes window.print() produce a clean dark-themed PDF."""
    st.markdown(
        """
<style>
@media print {
    /* Hide Streamlit chrome */
    [data-testid="stSidebar"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stBottom"],
    [data-testid="stMainMenu"],
    .stDeployButton,
    footer,
    .no-print {
        display: none !important;
    }
    /* Preserve dark background colors exactly */
    * {
        -webkit-print-color-adjust: exact !important;
        print-color-adjust: exact !important;
        color-adjust: exact !important;
    }
    /* A4 page with comfortable margins */
    @page { size: A4; margin: 1.5cm 1cm; }
    /* Remove Streamlit container padding */
    [data-testid="stMainBlockContainer"],
    [data-testid="block-container"] {
        padding: 0 !important;
        max-width: 100% !important;
    }
    /* Page break after each section divider */
    hr { page-break-after: always; border: none; }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _render_export_button():
    """Render a 'Save as PDF' button that triggers the browser print dialog."""
    st.markdown(
        """
<div class="no-print" style="display:flex;justify-content:flex-end;margin-bottom:8px;">
  <a href="javascript:window.print()"
     style="display:inline-block;padding:8px 18px;
            background:linear-gradient(135deg,#7B2FBE,#9d4de0);
            color:#ffffff;font-size:14px;font-weight:600;
            border-radius:8px;text-decoration:none;
            border:1px solid #9d4de0;cursor:pointer;">
    📄 Save as PDF
  </a>
</div>
""",
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

    def _pts_after_trade(element_id: int, trade_gw: int, window: int = 0) -> int:
        """Sum points for element_id from trade_gw+1 onward. window=0 means season-long."""
        end_gw = (trade_gw + window) if window > 0 else max_gw
        total = 0
        for gw in range(trade_gw + 1, min(end_gw, max_gw) + 1):
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
            pts_in_5gw = _pts_after_trade(el_in, trade_gw, window=5)
            pts_out_5gw = _pts_after_trade(el_out, trade_gw, window=5)
            transfers.append({
                "gw": trade_gw,
                "player_in": _player_name(el_in),
                "player_out": _player_name(el_out),
                "pts_in": pts_in,
                "pts_out": pts_out,
                "pts_in_5gw": pts_in_5gw,
                "pts_out_5gw": pts_out_5gw,
                "net": pts_in - pts_out,
                "net_5gw": pts_in_5gw - pts_out_5gw,
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
    entry_id: int, league_id: int, num_teams: int, team_name: str = ""
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

    # Build all picks. The `pick` field is the within-round pick position (1..num_teams),
    # NOT the overall sequential number. The choices API returns entries in true draft order,
    # so we use the list index to derive the overall pick, round, and pick-in-round.
    all_choices = []
    for idx, c in enumerate(choices_raw):
        player_id = c.get("element")
        team_id = c.get("entry")
        c_team_name = c.get("entry_name", "")
        if not player_id or not team_id:
            continue
        overall = idx + 1
        r = (idx) // num_teams + 1
        pick_in_round = (idx) % num_teams + 1
        all_choices.append({
            "overall_pick": overall,
            "pick": pick_in_round,
            "round": r,
            "player_id": int(player_id),
            "team_id": int(team_id),
            "team_name": c_team_name,
        })

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
        pts = _get_season_pts(c["player_id"])
        round_pts.setdefault(c["round"], []).append(pts)
    round_avg = {r: sum(v) / len(v) for r, v in round_pts.items()}

    # My picks — prefer entry_name matching (most robust across API ID variations),
    # fall back to entry_id comparison.
    player_map = get_fpl_player_mapping()
    norm_team = team_name.strip().lower() if team_name else ""
    if norm_team:
        my_choices = [c for c in all_choices if c["team_name"].strip().lower() == norm_team]
    else:
        my_choices = [c for c in all_choices if c["team_id"] == entry_id]
    my_picks = []
    for c in sorted(my_choices, key=lambda x: x["overall_pick"]):
        r = c["round"]
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
            "overall_pick": c["overall_pick"],
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

    # 2. Best Lineup Manager: Bench Mgmt Score (composite of Selection % + Bench Strength).
    # Using composite rather than raw Selection % prevents inactive managers (all 0-point
    # players) from dominating simply because any lineup is "optimal" when everyone scores 0.
    bench_league = compute_draft_league_bench_data(league_id, max_gw)
    # Filter to managers with meaningful activity (avg bench pts > 0 OR total actual > 0)
    active_bench = [r for r in bench_league if r.get("Total Bench Pts", 0) > 0 or r.get("Avg Bench/GW", 0) > 0]
    candidates = active_bench if active_bench else bench_league
    best_mgr_row = max(candidates, key=lambda r: r.get("Bench Mgmt Score", 0)) if candidates else None
    best_mgr = {
        "team": best_mgr_row["Team"] if best_mgr_row else "?",
        "value": f'{best_mgr_row["Selection %"]:.1f}% selection accuracy' if best_mgr_row else "?",
    }

    # 3. Luck: All-Play standings — must pass actual standings so Luck +/- is computed
    gw_scores = extract_draft_gw_scores(league_data)
    actual_rows = []
    for row in league_data.get("standings", []):
        eid = row.get("league_entry")
        t_name = team_names.get(eid, "")
        if t_name:
            actual_rows.append({
                "team": t_name,
                "actual_rank": row.get("rank", 0),
                "actual_pts": row.get("points_for", 0),
            })
    actual_standings_df = pd.DataFrame(actual_rows) if actual_rows else None
    standings_df = calculate_all_play_standings(gw_scores, actual_standings_df)
    luckiest = {"team": "?", "value": ""}
    unluckiest = {"team": "?", "value": ""}
    if not standings_df.empty and "Luck +/-" in standings_df.columns:
        sdf = standings_df.reset_index()  # Fair Rank was the index
        sdf["_luck"] = pd.to_numeric(sdf["Luck +/-"], errors="coerce").fillna(0).astype(int)
        lucky_row = sdf.loc[sdf["_luck"].idxmin()]   # most negative = luckiest
        unlucky_row = sdf.loc[sdf["_luck"].idxmax()]  # most positive = most unlucky
        lv = int(lucky_row["_luck"])
        uv = int(unlucky_row["_luck"])
        luckiest = {
            "team": lucky_row["Team"],
            # Negative Luck +/- means actual rank better than fair rank = lucky
            "value": f"{abs(lv)} win{'s' if abs(lv) != 1 else ''} over expectation" if lv < 0 else "On par with expectation",
        }
        unluckiest = {
            "team": unlucky_row["Team"],
            # Positive Luck +/- means actual rank worse than fair rank = unlucky
            "value": f"{uv} win{'s' if uv != 1 else ''} under expectation" if uv > 0 else "On par with expectation",
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
    _inject_print_styles()
    _render_export_button()

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
            title=_chart_title("Your Season Journey"),
            height=350,
            margin=dict(t=50, b=40, l=50, r=60),
            showlegend=False,
        )
        fig.update_xaxes(title="Gameweek", dtick=2)
        fig.update_yaxes(title="Points")
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # League position chart — color-coded by rank (gold=1st, pink=high, purple=low)
        rank_list = team_history["League_Position"].tolist()
        worst_rank_val = max(rank_list) if rank_list else num_teams
        best_rank_val = min(rank_list) if rank_list else 1

        def _rank_color(r):
            if r == 1:
                return _GOLD
            # Interpolate: rank=2 → bright pink, rank=worst → faded purple
            span = max(worst_rank_val - 2, 1)
            t = min(max((r - 2) / span, 0), 1)  # 0 at rank 2, 1 at worst rank
            # pink (#FF69B4) → purple (#7B2FBE)
            pr = int(255 + t * (123 - 255))
            pg = int(105 + t * (47 - 105))
            pb = int(180 + t * (190 - 180))
            return f"rgb({pr},{pg},{pb})"

        marker_colors = [_rank_color(r) for r in rank_list]
        marker_sizes = [12 if r == 1 else 7 for r in rank_list]
        marker_symbols = ["star" if r == 1 else "circle" for r in rank_list]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=gws,
            y=rank_list,
            mode="lines+markers",
            line=dict(color="rgba(123,47,190,0.5)", width=2),
            marker=dict(size=marker_sizes, color=marker_colors, symbol=marker_symbols,
                        line=dict(width=1, color="rgba(255,255,255,0.3)")),
            hovertemplate="GW %{x}: Rank #%{y}<extra></extra>",
        ))
        fig2.update_layout(
            **_DARK_CHART_LAYOUT,
            title=_chart_title("Your League Position Over the Season"),
            height=300,
            margin=dict(t=50, b=40, l=50, r=30),
            showlegend=False,
        )
        fig2.update_xaxes(title="Gameweek", dtick=2)
        fig2.update_yaxes(title="League Position", autorange="reversed", dtick=1)
        st.plotly_chart(fig2, use_container_width=True, theme=None)
        # Legend explaining the colour coding
        has_first = any(r == 1 for r in rank_list)
        legend_parts = [
            f'<span style="color:{_GOLD};font-weight:700;">★</span> <span style="color:#ccc;">1st place</span>',
            f'<span style="color:#FF69B4;font-weight:700;">●</span> <span style="color:#ccc;">Top half</span>',
            f'<span style="color:{_PURPLE};font-weight:700;">●</span> <span style="color:#ccc;">Bottom half</span>',
        ] if has_first else [
            f'<span style="color:#FF69B4;font-weight:700;">●</span> <span style="color:#ccc;">Top half</span>',
            f'<span style="color:{_PURPLE};font-weight:700;">●</span> <span style="color:#ccc;">Bottom half</span>',
        ]
        st.markdown(
            '<div style="font-size:12px;text-align:center;margin-top:-8px;margin-bottom:8px;">'
            + " &nbsp;·&nbsp; ".join(legend_parts) + "</div>",
            unsafe_allow_html=True,
        )

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
        col_mvp, col_xi = st.columns([5, 6])
        with col_mvp:
            st.markdown("#### 🏆 Captain's Armband")
            _render_wrapped_mvp(team_players, bootstrap)
        with col_xi:
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
        formation_counts = _compute_formation_stats(entry_id, max_gw)
        if formation_counts:
            top_formation = next(iter(formation_counts))
            top_count = formation_counts[top_formation]

            num_rows = len(formation_counts)
            row_h = 44  # approx px per data row
            card_h = max(180, num_rows * row_h + 56)  # header row + padding
            col_feat, col_tbl = st.columns([2, 3])
            with col_feat:
                top_html = (
                    f'<div style="border:2px solid {_PURPLE};border-radius:12px;padding:32px 24px;'
                    f'background:linear-gradient(135deg,#1a1a2e 0%,#2d1b69 100%);text-align:center;'
                    f'min-height:{card_h}px;display:flex;flex-direction:column;justify-content:center;">'
                    f'<div style="color:#e0e0e0;font-size:12px;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Most Used Formation</div>'
                    f'<div style="color:#c87bff;font-size:3.2em;font-weight:800;margin:12px 0;letter-spacing:4px;">{top_formation}</div>'
                    f'<div style="color:#ffffff;font-size:15px;font-weight:600;">{top_count} GWs</div>'
                    f'<div style="color:#e0e0e0;font-size:12px;margin-top:4px;">out of {max_gw} played</div>'
                    f'</div>'
                )
                st.markdown(top_html, unsafe_allow_html=True)
            with col_tbl:
                rows_html = ""
                for k, v in formation_counts.items():
                    pct = v / max_gw * 100
                    bar_w = int(pct)
                    is_top = k == top_formation
                    row_bg = "rgba(123,47,190,0.25)" if is_top else "#1e1e35"
                    badge_color = _PURPLE if is_top else "#4a4a6a"
                    rows_html += (
                        f'<tr style="border-bottom:1px solid #2a2a4a;background:{row_bg};">'
                        f'<td style="padding:11px 14px;color:#ffffff;'
                        f'font-weight:{"700" if is_top else "500"};font-size:17px;">{k}</td>'
                        f'<td style="padding:11px 14px;color:#ffffff;text-align:center;font-size:14px;">{v}</td>'
                        f'<td style="padding:11px 14px;">'
                        f'<div style="display:flex;align-items:center;gap:8px;">'
                        f'<div style="background:{badge_color};border-radius:4px;height:8px;width:{bar_w}%;min-width:4px;"></div>'
                        f'<span style="color:#e0e0e0;font-size:13px;white-space:nowrap;">{pct:.0f}%</span>'
                        f'</div></td></tr>'
                    )
                tbl_html = (
                    f'<div style="border:1px solid #2a2a4a;border-radius:10px;overflow:hidden;background:#1a1a2e;">'
                    f'<table style="width:100%;border-collapse:collapse;background:#1a1a2e;">'
                    f'<thead><tr style="background:rgba(123,47,190,0.4);">'
                    f'<th style="padding:12px 14px;color:#ffffff;text-align:left;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Formation</th>'
                    f'<th style="padding:12px 14px;color:#ffffff;text-align:center;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">GWs</th>'
                    f'<th style="padding:12px 14px;color:#ffffff;text-align:left;font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Share</th>'
                    f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
                )
                st.markdown(tbl_html, unsafe_allow_html=True)
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
            best_tx = max(transfers, key=lambda x: x["net"])
            worst_out_5gw = max(transfers, key=lambda x: x["pts_out_5gw"])

            col_b, col_w = st.columns(2)
            with col_b:
                st.markdown(
                    _stat_card(
                        "Best Transfer In",
                        best_tx["player_in"],
                        _GREEN,
                        f"+{best_tx['pts_in']} pts rest of season (net {best_tx['net']:+d})",
                    ),
                    unsafe_allow_html=True,
                )
            with col_w:
                st.markdown(
                    _stat_card(
                        "Worst Transfer Out (5-GW)",
                        worst_out_5gw["player_out"],
                        _RED,
                        f"{worst_out_5gw['pts_out_5gw']} pts in 5 GWs after you dropped them",
                    ),
                    unsafe_allow_html=True,
                )

            st.markdown(
                '<p style="color:#9ca3af;font-size:12px;font-style:italic;margin-top:8px;">'
                "Best In = highest net (Pts In minus Pts Out, rest of season). "
                "Worst Out = most pts scored in the 5 GWs immediately after being dropped "
                "— capped at 5 GWs to make early-season drops comparable to late-season ones.</p>",
                unsafe_allow_html=True,
            )

            # Full transfer table
            with st.expander("All Transfers This Season", expanded=False):
                df_tx = pd.DataFrame(transfers)[
                    ["gw", "player_in", "player_out", "pts_in_5gw", "pts_out_5gw", "net_5gw", "pts_in", "pts_out", "net"]
                ].rename(columns={
                    "gw": "GW", "player_in": "Player In", "player_out": "Player Out",
                    "pts_in_5gw": "In (5 GW)", "pts_out_5gw": "Out (5 GW)", "net_5gw": "Net (5 GW)",
                    "pts_in": "In (Season)", "pts_out": "Out (Season)", "net": "Net (Season)",
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
                    max_in = top_in[0][1]
                    rows_in = "".join(
                        f'<tr style="border-bottom:1px solid #2a2a4a;background:#1e1e35;">'
                        f'<td style="padding:10px 14px;color:#ffffff;font-size:14px;">{i+1}. {name}</td>'
                        f'<td style="padding:10px 14px;text-align:right;">'
                        f'<span style="background:{_GREEN};color:#000;border-radius:12px;padding:3px 12px;font-size:13px;font-weight:700;">{cnt}</span>'
                        f'</td></tr>'
                        for i, (name, cnt) in enumerate(top_in)
                    )
                    st.markdown(
                        f'<div style="border:1px solid #2a2a4a;border-radius:10px;overflow:hidden;background:#1a1a2e;">'
                        f'<table style="width:100%;border-collapse:collapse;background:#1a1a2e;">'
                        f'<thead><tr style="background:rgba(0,255,135,0.2);">'
                        f'<th style="padding:10px 14px;color:#ffffff;text-align:left;font-size:13px;font-weight:700;text-transform:uppercase;">Player</th>'
                        f'<th style="padding:10px 14px;color:#ffffff;text-align:right;font-size:13px;font-weight:700;text-transform:uppercase;">Times In</th>'
                        f'</tr></thead><tbody>{rows_in}</tbody></table></div>',
                        unsafe_allow_html=True,
                    )
            with col_mo:
                st.markdown("##### 📉 Most Transferred Out (League)")
                top_out = sorted(most_out.items(), key=lambda x: -x[1])[:5]
                if top_out:
                    rows_out = "".join(
                        f'<tr style="border-bottom:1px solid #2a2a4a;background:#1e1e35;">'
                        f'<td style="padding:10px 14px;color:#ffffff;font-size:14px;">{i+1}. {name}</td>'
                        f'<td style="padding:10px 14px;text-align:right;">'
                        f'<span style="background:{_RED};color:#fff;border-radius:12px;padding:3px 12px;font-size:13px;font-weight:700;">{cnt}</span>'
                        f'</td></tr>'
                        for i, (name, cnt) in enumerate(top_out)
                    )
                    st.markdown(
                        f'<div style="border:1px solid #2a2a4a;border-radius:10px;overflow:hidden;background:#1a1a2e;">'
                        f'<table style="width:100%;border-collapse:collapse;background:#1a1a2e;">'
                        f'<thead><tr style="background:rgba(255,75,75,0.2);">'
                        f'<th style="padding:10px 14px;color:#ffffff;text-align:left;font-size:13px;font-weight:700;text-transform:uppercase;">Player</th>'
                        f'<th style="padding:10px 14px;color:#ffffff;text-align:right;font-size:13px;font-weight:700;text-transform:uppercase;">Times Out</th>'
                        f'</tr></thead><tbody>{rows_out}</tbody></table></div>',
                        unsafe_allow_html=True,
                    )
    except Exception as e:
        st.info(f"Transfer data could not be computed: {e}")

    st.divider()

    # =========================================================================
    # PART 5: DRAFT RETROSPECTIVE
    # =========================================================================
    _section_header("🧬", "Draft Retrospective", "How did your pre-season draft hold up?")

    try:
        draft_data = _compute_draft_analysis(entry_id, config.FPL_DRAFT_LEAGUE_ID, num_teams, team_name=selected_team)
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
                df_picks = pd.DataFrame(picks)[
                    ["round", "pick", "overall_pick", "player", "pts", "avg_round_pts", "delta", "grade"]
                ].rename(columns={
                    "round": "Round", "pick": "Pick in Round", "overall_pick": "Overall Pick",
                    "player": "Player", "pts": "Season Pts", "avg_round_pts": "Avg for Round",
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
                        title=_chart_title("League Rank by Season"),
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
                                superlatives["best_mgr"]["value"], _GREEN),
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

    st.divider()

    # =========================================================================
    # APPENDIX: Metric Definitions
    # =========================================================================
    with st.expander("📖 How are these metrics calculated?", expanded=False):
        st.markdown("""
**Season Journey** — Your FPL points each gameweek. Gold bar = best GW, red = worst. Dashed line = your season average.

**League Position** — Your H2H league rank after each gameweek. Gold star = you were in 1st place that week.

**Perfect GWs** — Gameweeks where your actual starting XI was also the optimal lineup (zero points lost to incorrect bench ordering).

**Points Lost to Bench** — Total points scored by bench players who would have improved your score if fielded. Does not include bench-boost or free-hit gameweeks.

**Captain's Armband** — Your highest-scoring player across the season. Stats shown: Starts (games with 60+ minutes), Goals, Assists, Clean Sheets, Bonus Points, and Saves (GKs only).

**Season Best XI** — The optimal 11-man starting lineup from your squad based on total season points scored, using a valid formation (1 GK, 3–5 DEF, 3–5 MID, 1–3 FWD).

**Most Used Formation** — The DEF-MID-FWD formation you used most often across all 38 gameweeks, based on your actual weekly picks.

**Lineup Management / Bench Analysis** — Compares your actual weekly score vs the optimal score you could have achieved with perfect lineup selection. Selection % = actual ÷ optimal.

**Transfer Window** — Best Transfer In = the acquisition with the highest net points swing (pts in minus pts out, rest of season). Worst Transfer Out = the player who scored the most FPL points in the 5 gameweeks immediately after you dropped them (capped at 5 GWs so early-season drops are comparable to late-season ones).

**Draft Analysis** — Grades each of your 15 draft picks relative to the league average for that round. A "Steal 🔥" means you picked a player who outscored the average player taken in that round by 30+ points.

**Retention Rate** — How many of your original 15 draft picks are still on your roster at the end of the season.

**Cross-Season Draft History** — Pulls final standings from each season's league configured in `FPL_DRAFT_LEAGUE_HISTORY`. Ranks and points are the final end-of-season values.

**Most Active Manager** — Total approved waiver transactions made during the season.

**Best Lineup Manager** — Highest Bench Mgmt Score (composite of Selection % and Bench Strength). Bench Strength reflects the quality of available bench players, so teams with weak or inactive rosters cannot game this metric through 100% selection on players who all scored 0.

**Luckiest / Most Unlucky Manager** — Based on All-Play Record: every team is simulated against every other team each week. A team's "Fair Rank" is where they'd finish if everyone played everyone each GW. Luck +/- = Actual Rank minus Fair Rank. A team that finishes 3rd but would have ranked 1st in all-play had a lucky schedule.
""")

