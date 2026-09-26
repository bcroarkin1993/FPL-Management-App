"""
Classic FPL - Transfer Suggestions Page

Displays transfer targets ranked by projected points, form, FDR, and price.
Shows squad analysis with suggested transfers and upcoming fixtures.
"""

import config
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from scripts.common.error_helpers import show_api_error
from scripts.common.utils import (
    get_classic_bootstrap_static,
    get_classic_team_picks,
    get_classic_team_history,
    get_entry_details,
    get_current_gameweek,
    get_rotowire_player_projections,
    get_classic_transfers,
    position_converter,
)
from scripts.common.data_validation import (
    check_free_transfers,
    check_transfer_plan,
    format_issues,
)
from scripts.common.error_helpers import get_logger
from scripts.common.optimization import (
    HIT_COST,
    diff_squads,
    pair_transfer_legs,
    solve_squad_ilp,
)
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import compact_html
from scripts.common.transfer_sanity import (horizon_column, horizon_points,
                                             sanity_check_suggestion)
from scripts.common.analytics import (
    _claim_reference_rows,
    compute_player_scores,
    numeric_col,
    compute_healthy_form,
    _fetch_element_history,
    compute_positional_depth,
    compute_transfer_urgency,
    blend_multi_gw_projections,
    positional_rank,
    merge_season_projections,
    merge_ffp_single_gw_data,
    blend_projections_onto,
)
from scripts.common.scraping import get_ffp_feed, get_rotowire_season_rankings, render_ffp_status
from scripts.common.cache import purge_cache_prefix
from scripts.common.fpl_auth import fetch_my_team
from scripts.common.classic_squad import (
    PENDING_FILE as _PENDING_FILE,
    apply_pending_transfers as _apply_pending_transfers,
    load_pending_file as _load_pending_file,
    resolve_classic_squad,
    save_pending_file as _save_pending_file,
)

_logger = get_logger("fpl_app.classic_transfers")


# ---------------------------
# LOCAL PENDING TRANSFERS
# Dual-persisted: JSON file (survives hot-reloads and browser refreshes)
# + st.session_state (fast within-session access).
# The file is the source of truth; session_state is a cache.
# ---------------------------

_PENDING_KEY = "fpl_classic_pending_transfers"

# _PENDING_FILE / _load_pending_file / _save_pending_file / _apply_pending_transfers
# are imported from scripts.common.classic_squad — this page owns the session-state
# layer and the add/remove UI, the shared resolver owns the file and the replay so
# read-only consumers (Fixture Projections, the optimizers) see the same squad.


def _init_pending_state(_state=None) -> None:
    """Populate session_state from file on first page load (or after hot-reload)."""
    state = st.session_state if _state is None else _state
    if _PENDING_KEY not in state:
        state[_PENDING_KEY] = _load_pending_file()


def _get_pending_local(team_id, _state=None) -> list:
    """Return locally-logged pending transfers for this team.
    _state is injectable for testing; production uses st.session_state.
    """
    state = st.session_state if _state is None else _state
    return [t for t in state.get(_PENDING_KEY, [])
            if t.get("team_id") == team_id]


def _add_pending_local(team_id, event: int, element_out: int, element_in: int,
                        out_cost: int, in_cost: int, _state=None) -> None:
    state = st.session_state if _state is None else _state
    # _init_pending_state must be called before this so session_state has file contents.
    existing = list(state.get(_PENDING_KEY, []))
    # Replace any existing transfer with the same element_out for this team
    existing = [t for t in existing
                if not (t.get("team_id") == team_id and t.get("element_out") == element_out)]
    existing.append({
        "team_id": team_id,
        "event": int(event),
        "element_out": int(element_out),
        "element_in": int(element_in),
        "element_out_cost": int(out_cost),
        "element_in_cost": int(in_cost),
        "time": datetime.now(timezone.utc).isoformat(),
        "local": True,
    })
    state[_PENDING_KEY] = existing
    if _state is None:
        _save_pending_file(existing)


def _remove_pending_local(team_id, element_out: int, _state=None) -> None:
    state = st.session_state if _state is None else _state
    updated = [
        t for t in state.get(_PENDING_KEY, [])
        if not (t.get("team_id") == team_id and t.get("element_out") == element_out)
    ]
    state[_PENDING_KEY] = updated
    if _state is None:
        _save_pending_file(updated)


def _sync_pending_local(team_id, api_transfers: list, _state=None) -> list:
    """Drop confirmed transfers; return remaining. _state injectable for tests."""
    state = st.session_state if _state is None else _state
    # Merge file into session state (in case of hot-reload or refresh)
    if _state is None and _PENDING_KEY not in state:
        state[_PENDING_KEY] = _load_pending_file()
    # Match on (element_out, element_in, event) triplet — NOT just the player pair.
    # Using only (out, in) would incorrectly clear a new pending transfer if the
    # same player pair was ever swapped in a previous GW (historical season transfers
    # are all returned by the FPL API, not just the current GW).
    confirmed = {(t["element_out"], t["element_in"], t.get("event", 0)) for t in api_transfers}
    updated = [
        t for t in state.get(_PENDING_KEY, [])
        if not (t.get("team_id") == team_id
                and (t.get("element_out"), t.get("element_in"), t.get("event", 0)) in confirmed)
    ]
    # Only write when something actually cleared. This runs on every page load,
    # and rewriting the file unconditionally meant every rerun touched disk to
    # store what was already there.
    changed = len(updated) != len(state.get(_PENDING_KEY, []))
    state[_PENDING_KEY] = updated
    if _state is None and changed:
        _save_pending_file(updated)
    return _get_pending_local(team_id, state)


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def _format_money(value: int) -> str:
    """Format FPL money value (stored as tenths) to display format."""
    if value is None:
        return "N/A"
    return f"£{value / 10:.1f}m"


def _format_price_change(change: int) -> str:
    """Format price change with indicator."""
    if change is None or change == 0:
        return ""
    if change > 0:
        return f"↑{change/10:.1f}"
    return f"↓{abs(change)/10:.1f}"


@st.cache_data(ttl=300)
def _load_future_fixtures() -> pd.DataFrame:
    """
    Returns future fixtures with difficulties.
    Columns: event, team_h, team_a, team_h_difficulty, team_a_difficulty
    """
    import requests
    url = "https://fantasy.premierleague.com/api/fixtures/?future=1"
    try:
        fx = requests.get(url, timeout=30).json()
        df = pd.DataFrame(fx)
        keep = ["event", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
        df = df[[c for c in keep if c in df.columns]].copy()
        return df
    except Exception:
        return pd.DataFrame()


def _get_team_fixtures(team_id: int, n_weeks: int, current_gw: int) -> List[Dict]:
    """Get next n fixtures for a team with FDR."""
    fixtures = _load_future_fixtures()
    if fixtures.empty:
        return []

    fixtures = fixtures.dropna(subset=["event"])
    fixtures["event"] = fixtures["event"].astype(int)

    upcoming = fixtures[
        (fixtures["event"] >= current_gw) &
        (fixtures["event"] < current_gw + n_weeks)
    ].copy()

    result = []
    for _, row in upcoming.iterrows():
        if row.get("team_h") == team_id:
            result.append({
                "gw": int(row["event"]),
                "opponent": int(row["team_a"]),
                "home": True,
                "fdr": row.get("team_h_difficulty", 3)
            })
        elif row.get("team_a") == team_id:
            result.append({
                "gw": int(row["event"]),
                "opponent": int(row["team_h"]),
                "home": False,
                "fdr": row.get("team_a_difficulty", 3)
            })

    return sorted(result, key=lambda x: x["gw"])


def _avg_fdr_for_team(team_id: int, current_gw: int, n_weeks: int) -> Optional[float]:
    """Average FDR over next n_weeks for a team."""
    fixtures = _get_team_fixtures(team_id, n_weeks, current_gw)
    if not fixtures:
        return None
    fdr_values = [f["fdr"] for f in fixtures if f.get("fdr")]
    return float(np.mean(fdr_values)) if fdr_values else None


def _avg_fdr_by_team(current_gw: int, n_weeks: int) -> Dict[int, Optional[float]]:
    """`_avg_fdr_for_team` for every club at once.

    FDR is a property of the club, and there are twenty of them. Mapping the
    per-player call over a ~700-row frame ran `_get_team_fixtures` seven
    hundred times -- each doing a `dropna`, an `astype`, a boolean filter, a
    `.copy()` and an `iterrows()` over the whole fixture table -- to produce
    twenty distinct answers.
    """
    fixtures = _load_future_fixtures()
    if fixtures.empty:
        return {}
    team_ids = set(pd.to_numeric(fixtures.get("team_h"), errors="coerce").dropna().astype(int))
    team_ids |= set(pd.to_numeric(fixtures.get("team_a"), errors="coerce").dropna().astype(int))
    return {t: _avg_fdr_for_team(t, current_gw, n_weeks) for t in team_ids}


def _get_fdr_color(fdr: float) -> str:
    """Get background color for FDR value."""
    if fdr is None:
        return "#808080"
    if fdr <= 2:
        return "#00c853"  # Green - easy
    elif fdr <= 2.5:
        return "#7cb342"  # Light green
    elif fdr <= 3:
        return "#ffc107"  # Yellow - medium
    elif fdr <= 3.5:
        return "#ff9800"  # Orange
    else:
        return "#dc3545"  # Red - hard


def _add_projections(df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """Attach Rotowire's weekly `Points` and `Pos Rank` to a player frame.

    Goes through `_claim_reference_rows()` -- the shared tiered matcher -- for
    the reasons CLAUDE.md's "Player Matching" section gives, and this callsite
    was the last one in the app that did not.

    What it replaces was a hand-rolled ladder that scanned the whole
    projections frame per player, scored every row with `fuzz.ratio` and
    accepted any hit at **60**, with team and position contributing a +15 nudge
    rather than scoping the search. Two consequences, measured live against a
    659-player pool and Rotowire's 220 rows:

    * **It matched more players than there were rows to match.** 323 players
      were given a projection from a 220-row table: 172 of them were wearing
      another player's numbers, because nothing stopped one reference row being
      claimed over and over. David Raya (ARS, GK) shared a row with Rayan,
      Allan and Gray; Ødegaard with Merino, Martinelli and Nørgaard; eight
      goalkeepers shared one row between them. Every value on screen was
      plausible.
    * **It cost ~6 seconds per page load**, twice -- once for the squad and
      once for the pool -- which is most of why logging a transfer froze: with
      no `st.form`, every dropdown change paid it again.

    `_claim_reference_rows` scopes every tier below the first two by position
    and team, and gives one reference row to at most one player, resolving
    contention in favour of the stronger tier.
    """
    if projections_df is None or projections_df.empty:
        df["Projected_Points"] = None
        df["Pos_Rank"] = None
        return df

    # The matcher expects the full legal name in `Player` and the short form in
    # `Web_Name`; this page's frames carry them the other way round, under
    # `Full Name` and `Player`. Probe with the names it expects rather than
    # renaming the real frame, which every merge on this page keys on.
    probe = pd.DataFrame({
        "Player": df["Full Name"] if "Full Name" in df.columns else df["Player"],
        "Web_Name": df["Player"],
        "Team": df.get("Team"),
        "Position": df.get("Position"),
    }, index=df.index)

    stats: Dict = {}
    mapping = _claim_reference_rows(
        probe, projections_df,
        name_col="Player", ref_name_col="Player",
        ref_team_col="Team" if "Team" in projections_df.columns else None,
        source_name="Rotowire weekly (classic transfers)",
        stats=stats,
    )
    if stats:
        _logger.info("Rotowire weekly: matched %s of %s rows (%.0f%%)",
                     stats.get("matched"), stats.get("total"),
                     100 * stats.get("rate", 0.0))

    points = pd.Series(index=df.index, dtype="float64")
    ranks = pd.Series(index=df.index, dtype="object")
    for player_idx, ref_idx in mapping.items():
        points.at[player_idx] = pd.to_numeric(
            projections_df.at[ref_idx, "Points"], errors="coerce")
        if "Pos Rank" in projections_df.columns:
            ranks.at[player_idx] = projections_df.at[ref_idx, "Pos Rank"]

    df["Projected_Points"] = points
    df["Pos_Rank"] = ranks
    return df


def _build_all_players_df(bootstrap: dict, current_gw: int, n_weeks: int) -> pd.DataFrame:
    """Build a DataFrame of all players with relevant stats."""
    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}

    rows = []
    for p in elements:
        team_id = p.get("team")
        team_info = teams.get(team_id, {})

        rows.append({
            "Player_ID": p.get("id"),
            "Player": p.get("web_name"),
            "Full Name": f"{p.get('first_name', '')} {p.get('second_name', '')}".strip(),
            "Team": team_info.get("short_name", "???"),
            "Team_ID": team_id,
            "Position": position_converter(p.get("element_type")),
            "element_type": p.get("element_type"),
            "now_cost": p.get("now_cost", 0),
            "selling_price": p.get("selling_price", p.get("now_cost", 0)),
            "form": float(p.get("form", 0) or 0),
            "points_per_game": float(p.get("points_per_game", 0) or 0),
            "total_points": p.get("total_points", 0),
            "selected_by_percent": float(p.get("selected_by_percent", 0) or 0),
            "transfers_in_event": p.get("transfers_in_event", 0),
            "transfers_out_event": p.get("transfers_out_event", 0),
            "cost_change_event": p.get("cost_change_event", 0),
            "ep_next": float(p.get("ep_next", 0) or 0),  # Expected points next GW
            "minutes": p.get("minutes", 0),
            "starts": p.get("starts", 0),
            "news": p.get("news", ""),
            "status": p.get("status", "a"),
            "chance_of_playing_next_round": p.get("chance_of_playing_next_round"),
        })

    df = pd.DataFrame(rows)

    # Add average FDR for next n weeks. Twenty clubs, not seven hundred players.
    df["AvgFDR"] = df["Team_ID"].map(_avg_fdr_by_team(current_gw, n_weeks))

    return df


def _build_squad_df(picks: list, bootstrap: dict, entry_history: dict) -> pd.DataFrame:
    """Build squad DataFrame from picks."""
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    rows = []
    for pick in picks:
        element_id = pick["element"]
        player = elements.get(element_id, {})
        team_id = player.get("team")

        rows.append({
            "Player_ID": element_id,
            "Player": player.get("web_name", "Unknown"),
            "Full Name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "Team": teams.get(team_id, "???"),
            "Team_ID": team_id,
            "Position": position_converter(player.get("element_type")),
            "element_type": player.get("element_type"),
            "squad_position": pick["position"],
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "multiplier": pick.get("multiplier", 1),
            "now_cost": player.get("now_cost", 0),
            "selling_price": pick.get("selling_price", player.get("now_cost", 0)),
            "form": float(player.get("form", 0) or 0),
            "points_per_game": float(player.get("points_per_game", 0) or 0),
            "total_points": player.get("total_points", 0),
            "ep_next": float(player.get("ep_next", 0) or 0),
            "minutes": player.get("minutes", 0),
            "starts": player.get("starts", 0),
            "news": player.get("news", ""),
            "chance_of_playing_next_round": player.get("chance_of_playing_next_round"),
        })

    return pd.DataFrame(rows)


def _compute_transfer_score(df: pd.DataFrame,
                            all_players_df: Optional[pd.DataFrame] = None,
                            current_gw: int = 19) -> pd.DataFrame:
    """Compute transfer target scores using positional percentile scoring.

    Delegates to shared compute_player_scores() with format_context="classic"
    and adds a small price-efficiency adjustment to Transfer Score only.
    """
    tmp = compute_player_scores(df, all_players_df, current_gw, format_context="classic")

    # Classic-specific: small price-efficiency adjustment for Transfer Score only
    # Cheaper players are more valuable as transfer targets (budget flexibility)
    if "now_cost" in tmp.columns and "Transfer Score" in tmp.columns:
        max_cost = pd.to_numeric(tmp["now_cost"], errors="coerce").max()
        if pd.notna(max_cost) and max_cost > 0:
            price_bonus = (max_cost - pd.to_numeric(tmp["now_cost"], errors="coerce")) / max_cost * 0.05
            tmp["Transfer Score"] = (tmp["Transfer Score"] + price_bonus).clip(upper=1.0)

    return tmp


def _compute_keep_score(df: pd.DataFrame,
                        all_players_df: Optional[pd.DataFrame] = None,
                        current_gw: int = 19,
                        depth_map: Optional[Dict] = None) -> pd.DataFrame:
    """Compute keep scores using positional percentile scoring.

    Delegates to shared compute_player_scores() with format_context="classic"
    and depth_map for squad depth awareness.
    """
    return compute_player_scores(df, all_players_df, current_gw,
                                 format_context="classic", depth_map=depth_map)


def _format_fixtures_html(fixtures: List[Dict], teams: Dict[int, str], n_show: int = 5) -> str:
    """Format fixtures as HTML with colored FDR badges."""
    if not fixtures:
        return "<span style='color: #888;'>No fixtures</span>"

    html_parts = []
    for f in fixtures[:n_show]:
        opp = teams.get(f["opponent"], "???")
        venue = "H" if f["home"] else "A"
        fdr = f.get("fdr", 3)
        color = _get_fdr_color(fdr)
        html_parts.append(
            f"<span style='background-color:{color}; color:white; padding:2px 6px; "
            f"border-radius:4px; margin-right:4px; font-size:0.85em;'>{opp}({venue})</span>"
        )

    return "".join(html_parts)


def _get_availability_indicator(chance: Optional[int], news: str) -> str:
    """Get availability indicator based on chance of playing."""
    if chance is None:
        if news:
            return f"⚠️ {news[:30]}..."
        return "✓"
    if chance == 0:
        return f"❌ {news[:25]}..." if news else "❌ Out"
    elif chance <= 25:
        return f"🔴 {chance}%"
    elif chance <= 50:
        return f"🟠 {chance}%"
    elif chance <= 75:
        return f"🟡 {chance}%"
    else:
        return "✓"


# First gameweek in which a double-use chip's second slot may be played.
CHIP_SLOT2_FIRST_GW = 20


def _parse_chip_status(history: dict, current_gw: int) -> dict:
    """Parse chip usage and availability from team history.

    Both Wildcard and Bench Boost are double-use chips:
      - Slot 1: used before GW20 (event < 20)
      - Slot 2: used in GW20+ (event >= 20)
    Free Hit and Triple Captain remain single-use.

    **"Available" means playable this gameweek, not owned somewhere in the
    season.** The two are different for a double-use chip and conflating them
    listed "Wildcard" under Chips Available at GW4 for a manager who had played
    it in GW3 -- the second slot exists, but not until GW20. The Chip Strategy
    advisor reads the same flag, so it was also capable of recommending a
    wildcard rebuild that could not be actioned for sixteen weeks.

    Chips whose next slot opens later are reported separately in
    ``available_later``, since "you have no wildcard" would be just as wrong.
    """
    chips_used = history.get("chips", []) if history else []

    # Wildcard and Bench Boost are tracked as lists (double-use)
    used: Dict[str, Any] = {"wildcard": [], "bboost": [], "freehit": None, "3xc": None}
    for chip in chips_used:
        name = chip.get("name", "")
        event = chip.get("event", 0)
        if name in ("wildcard", "bboost"):
            used[name].append(event)
        elif name in used:
            used[name] = event

    def _playable_now(events: list) -> bool:
        """Is this double-use chip's *current* slot still unused?"""
        if current_gw < CHIP_SLOT2_FIRST_GW:
            return not any(e < CHIP_SLOT2_FIRST_GW for e in events)
        return not any(e >= CHIP_SLOT2_FIRST_GW for e in events)

    def _playable_later(events: list) -> bool:
        """Second slot still to come, but not before GW20."""
        return (current_gw < CHIP_SLOT2_FIRST_GW
                and not any(e >= CHIP_SLOT2_FIRST_GW for e in events))

    wildcard_1_used = any(e < CHIP_SLOT2_FIRST_GW for e in used["wildcard"])
    wildcard_2_used = any(e >= CHIP_SLOT2_FIRST_GW for e in used["wildcard"])
    wildcard_available = _playable_now(used["wildcard"])

    bboost_1_used = any(e < CHIP_SLOT2_FIRST_GW for e in used["bboost"])
    bboost_2_used = any(e >= CHIP_SLOT2_FIRST_GW for e in used["bboost"])
    bboost_available = _playable_now(used["bboost"])

    available = []
    if wildcard_available:
        available.append("wildcard")
    if bboost_available:
        available.append("bboost")
    if used["freehit"] is None:
        available.append("freehit")
    if used["3xc"] is None:
        available.append("3xc")

    # Owned, but not yet playable: the second slot of a double-use chip whose
    # first slot is spent. Listing it as available is wrong; dropping it
    # silently reads as "that chip is gone".
    available_later = [
        chip for chip, events in (("wildcard", used["wildcard"]), ("bboost", used["bboost"]))
        if chip not in available and _playable_later(events)
    ]

    return {
        "used": used,
        "available": available,
        "available_later": available_later,
        "slot2_first_gw": CHIP_SLOT2_FIRST_GW,
        "wildcard_1_used": wildcard_1_used,
        "wildcard_2_used": wildcard_2_used,
        "wildcard_available": wildcard_available,
        "bboost_1_used": bboost_1_used,
        "bboost_2_used": bboost_2_used,
        "bboost_available": bboost_available,
    }


# FPL caps accumulated free transfers. A rules constant: update it if FPL does.
# The bootstrap publishes the same number as `game_settings.max_extra_free_transfers`
# (4 *extra* on top of the base 1); this is that plus one.
MAX_BANKED_FREE_TRANSFERS = 5


def _compute_free_transfers(history: dict, entry_history: dict, current_gw: int,
                             fh_gws: Optional[set] = None) -> int:
    """Free transfers available this gameweek.

    **Prefer the number FPL states.** The authenticated `my-team` payload
    carries `transfers.limit` -- the gameweek's *allowance* -- and
    `transfers.made`. `normalise_my_team()` forwards them as
    `event_transfers_limit` and `event_transfers`, and the remaining count is
    the difference: the payload `{"limit": 1, "made": 2, "cost": 4}` is one
    free transfer, two made, hence a four-point hit. Returning `limit` alone
    reported 1 there when the answer was 0. `limit` is absent while a chip
    grants unlimited transfers, so a missing key means "reconstruct", never
    "zero".

    Otherwise replay the season. FPL's rule: one free transfer per gameweek
    **from GW2** -- there is no transfer allowance before the first deadline,
    when the squad is being picked -- with unused ones accumulating to
    `MAX_BANKED_FREE_TRANSFERS`.

    `available` entering each iteration is the limit *for that gameweek*, so
    the seed is the limit entering the first gameweek in the history, which is
    zero: the accrual line then gives GW2 exactly one. Seeding it at 1 charged
    GW1 as a banking gameweek and reported one too many for the rest of the
    season -- 4 free transfers at GW5 against FPL's 3. Seeding zero rather than
    special-casing `gw == 1` also handles a manager who joined mid-season,
    whose first deadline is likewise unlimited: `max(0, 0 - made) + 1 == 1`
    however many transfers that week registered.

    **A wildcard or free hit week neither spends nor earns a transfer.** The
    bank is retained across the chip, but "you don't get an extra free transfer
    in the week you Wildcard" -- so `limit(chip_gw + 1) == limit(chip_gw)`,
    which is what skipping the gameweek entirely produces. A wildcard also
    registers a dozen transfers in `event_transfers` that were never charged;
    subtracting them wiped the bank. Free Hit was excluded here before Wildcard
    was.

    Three further faults this replaces:

    - It **stopped at the first gameweek back**, so the answer could never
      exceed 2. A manager who sat out three gameweeks was told they had 2.
    - The "already took a hit this week" guard tested
      `event_transfers_cost < 0`. FPL publishes that cost as a **positive**
      number -- this page renders it as `f"-{transfer_cost} pts"` -- so the
      guard never once fired.
    - Transfers logged in-app but not yet confirmed by FPL were never counted.
      That is handled by the caller, which knows whether the squad came from
      the authenticated payload (where FPL has already counted them).

    The replay still understates when a gameweek spent *some* of a larger bank,
    because the history records transfers made but never the limit they were
    made against. Understating is the safe direction -- it warns of a hit that
    turns out to be free -- and the authenticated path has no such gap. Neither
    path can see an ad-hoc grant such as the GW16 2025/26 AFCON top-up, which
    is why the page also offers a manual override.
    """
    limit = (entry_history or {}).get("event_transfers_limit")
    if limit is not None:
        made = int((entry_history or {}).get("event_transfers", 0) or 0)
        return max(0, int(limit) - made)

    if not history:
        return 1

    unlimited_gws = set(fh_gws or ())
    for chip in history.get("chips", []) or []:
        if chip.get("name") in ("freehit", "wildcard") and chip.get("event") is not None:
            unlimited_gws.add(chip["event"])

    entries = sorted(
        (e for e in (history.get("current") or []) if e.get("event") is not None),
        key=lambda e: e["event"],
    )

    available = 0
    made_this_gw = 0
    for entry in entries:
        gw = entry["event"]
        if gw > current_gw:
            break
        if gw in unlimited_gws:
            continue
        made = int(entry.get("event_transfers", 0) or 0)
        if gw == current_gw:
            # Read the current gameweek from history, not from entry_history:
            # between deadlines the latter belongs to the *last* gameweek's
            # picks, and subtracting its transfers would charge them twice.
            made_this_gw = made
            break
        available = min(MAX_BANKED_FREE_TRANSFERS, max(0, available - made) + 1)

    return max(0, available - made_this_gw)


def _ft_override_key(team_id, gameweek: int) -> str:
    """Session key for a manually-stated free-transfer count.

    Scoped to the gameweek as well as the team: an override is an answer to
    "how many do I have *now*", and carrying last week's answer forward would
    be a stale number presented as a stated one.
    """
    return "ft_override_%s_%s" % (team_id, gameweek)


def resolve_free_transfers(history: dict, entry_history: dict, current_gw: int,
                            fh_gws: Optional[set] = None,
                            team_id=None, logged_pending: int = 0,
                            squad_source: str = "", _state=None) -> dict:
    """How many free transfers are available, and where that number came from.

    Three sources, in precedence order, because each is more trustworthy than
    the one below it:

    1. **A manual override.** FPL grants transfers this app cannot reconstruct
       -- the GW16 2025/26 top-up handed everyone extra to absorb AFCON
       departures -- so a manager who can read the real number on FPL's own
       site must be able to say so.
    2. **`limit - made`** from the authenticated `my-team` payload, which is
       the answer outright.
    3. **The replay**, which is a reconstruction and the only option without a
       credential.

    `logged_pending` is subtracted only off the replay. The authenticated
    payload already counts a logged transfer in `transfers.made`, and
    `_reconcile_pending_log()` retires the entry -- subtracting again would
    charge the same move twice. Without that subtraction the panel read
    "N FTs Banked / All free this gameweek" with transfers already spent.

    Returns a dict with `count`, `source`, `computed` (before any override),
    and `hits`, the number of logged transfers that exceeded the allowance --
    which must be reported rather than swallowed by the clamp at zero.
    """
    state = st.session_state if _state is None else _state

    computed = _compute_free_transfers(history, entry_history, current_gw,
                                       fh_gws=fh_gws)
    stated = (entry_history or {}).get("event_transfers_limit") is not None
    source = "my_team" if stated else "replay"

    hits = 0
    if not stated and logged_pending and squad_source != "my_team":
        hits = max(0, int(logged_pending) - computed)
        computed = max(0, computed - int(logged_pending))

    count = computed
    override = state.get(_ft_override_key(team_id, current_gw))
    if override is not None:
        count = max(0, int(override))
        source = "manual"

    return {"count": count, "source": source, "computed": computed,
            "hits": hits, "logged": int(logged_pending or 0)}


def _blended_proj(row) -> float:
    """Expected points for one row, preferring the engine's blend.

    Suggestion cards printed `Projected_Points` -- Rotowire's raw "if he starts"
    number -- under the same "Proj" label the tables use for the blend, so the
    same player read two different ways a few hundred pixels apart. `Proj` is
    the blend across every source that priced him, with start likelihood already
    applied; Rotowire alone is the fallback for a frame built before the blend.
    """
    for col in ("Proj", "Projected_Points"):
        value = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(value):
            return float(value)
    return float("nan")


def _ownership_badge(pct: float) -> str:
    """Return HTML ownership badge for template (>20%) or differential (<5%) players."""
    pct = float(pct or 0)
    if pct >= 20:
        return ('<span style="background:#7c3aed;color:#fff;padding:2px 8px;border-radius:10px;'
                'font-size:0.75em;font-weight:bold;">Template</span>')
    elif pct <= 5:
        return ('<span style="background:#0891b2;color:#fff;padding:2px 8px;border-radius:10px;'
                'font-size:0.75em;font-weight:bold;">Differential</span>')
    return ""


def _format_price_trend(cost_change_event: int, transfers_in_event: int,
                         transfers_out_event: int) -> dict:
    """Format price trend badge from GW transfer activity."""
    cost_change = int(cost_change_event or 0)
    t_in = int(transfers_in_event or 0)
    t_out = int(transfers_out_event or 0)

    if cost_change > 0:
        indicator = "Rising"
        badge_html = (f'<span style="color:#4ecca3;font-size:0.78em;font-weight:bold;">'
                      f'&#8593; Rising +£{cost_change/10:.1f}m</span>')
    elif cost_change < 0:
        indicator = "Falling"
        badge_html = (f'<span style="color:#f87171;font-size:0.78em;font-weight:bold;">'
                      f'&#8595; Falling £{cost_change/10:.1f}m</span>')
    elif t_in > t_out * 1.5 and t_in > 50_000:
        indicator = "Likely Rising"
        badge_html = '<span style="color:#86efac;font-size:0.78em;">&#8593; Likely Rising</span>'
    else:
        indicator = "Stable"
        badge_html = ""

    rush_html = ""
    if t_in > 200_000:
        rush_html = ('<span style="background:#d97706;color:#fff;padding:1px 6px;border-radius:8px;'
                     'font-size:0.72em;font-weight:bold;margin-left:4px;">Transfer Rush</span>')

    return {"indicator": indicator, "badge_html": badge_html, "rush_html": rush_html}


def _compute_hit_verdict(ep_delta: float, is_hit: bool) -> dict:
    """Determine whether a transfer is worth a -4 hit based on FPL expected points delta.

    ``display_str`` answers only the question the verdict cannot: what the move
    is worth *after* the hit. A free transfer has no hit to net off, so it gets
    no number here -- the card already prints the delta, and pairing a green
    "FREE" with a bare "-3.0 pts net (free)" read as though making the transfer
    cost three points rather than as the projection difference it is.
    """
    net_gain = ep_delta - 4.0 if is_hit else ep_delta

    if not is_hit:
        verdict = "FREE"
        verdict_color = "#4ecca3"
    elif net_gain >= 2.0:
        verdict = "YES"
        verdict_color = "#4ecca3"
    elif net_gain >= 0:
        verdict = "MARGINAL"
        verdict_color = "#fbbf24"
    else:
        verdict = "NO"
        verdict_color = "#f87171"

    if is_hit:
        sign = "+" if net_gain >= 0 else "&minus;"
        display_str = f"{sign}{abs(net_gain):.1f} xPts after the &minus;4 hit"
    else:
        display_str = ""

    return {
        "net_gain": net_gain,
        "verdict": verdict,
        "verdict_color": verdict_color,
        "display_str": display_str,
    }


def _render_transfer_status_panel(bank: int, squad_value: int, free_transfers: int,
                                   chip_status: dict, active_chip: Optional[str],
                                   ft_info: Optional[dict] = None,
                                   team_id=None, current_gw: Optional[int] = None):
    """Render a top-of-page status panel: free transfers, bank, chips.

    `ft_info` is `resolve_free_transfers()`'s result. The panel says where the
    free-transfer number came from, because the three sources are not equally
    trustworthy -- the replay is a reconstruction that cannot see an ad-hoc
    grant, and presenting it identically to FPL's own stated number is what let
    a wrong count go unquestioned.
    """
    chip_names = {"wildcard": "Wildcard", "bboost": "Bench Boost",
                  "freehit": "Free Hit", "3xc": "Triple Captain"}
    chip_colors = {"wildcard": "#7c3aed", "bboost": "#166534",
                   "freehit": "#0891b2", "3xc": "#d97706"}

    def stat_card_html(label: str, value: str, accent: str = "#00ff87", subtitle: str = "") -> str:
        sub = (f'<div style="color:#aaa;font-size:11px;margin-top:4px;">{subtitle}</div>'
               if subtitle else "")
        return (
            f'<div style="border:1px solid #333;border-radius:10px;padding:14px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            f'text-align:center;color:#e0e0e0;height:100%;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
            f'<div style="color:{accent};font-size:20px;font-weight:700;">{value}</div>'
            f'{sub}</div>'
        )

    # Free transfer card
    info = ft_info or {}
    if free_transfers >= 2:
        ft_val, ft_color, ft_sub = (f"{free_transfers} FTs Banked", "#4ecca3",
                                    "All free this gameweek")
    elif free_transfers == 1:
        ft_val, ft_color, ft_sub = "1 Free Transfer", "#00ff87", "Next costs &minus;4 pts"
    else:
        ft_val, ft_color, ft_sub = "0 FTs (Hit GW)", "#f87171", "&minus;4 pts per transfer"

    # Logged transfers beyond the allowance cost 4 points each. That is a real
    # state, and clamping the count to zero without saying so hides the hit.
    if info.get("hits"):
        ft_sub = (f"{info['logged']} logged, {info['logged'] - info['hits']} free "
                  f"&mdash; &minus;{4 * info['hits']} pts")
        ft_color = "#f87171"
    elif info.get("logged"):
        ft_sub = f"{info['logged']} logged this week &mdash; {ft_sub.lower()}"

    _FT_SOURCE_LABEL = {
        "manual": "you stated this",
        "my_team": "from your FPL account",
        # Plain text: this labels an st.expander, which renders markdown, not
        # HTML -- an entity here reaches the user as literal "&mdash;".
        "replay": "reconstructed — not stated by FPL",
    }
    ft_source = info.get("source", "replay")

    # Active chip card
    if active_chip:
        active_val = chip_names.get(active_chip, active_chip)
        active_color = chip_colors.get(active_chip, "#fbbf24")
        active_sub = "Currently active!"
    else:
        active_val, active_color, active_sub = "None Active", "#6b7280", ""

    # Available chips card (with pill badges). A chip whose next slot does not
    # open until GW20 is shown muted and dated rather than as available now.
    avail = chip_status.get("available", [])
    later = chip_status.get("available_later", [])
    slot2_gw = chip_status.get("slot2_first_gw", 20)
    if avail or later:
        badges = "".join(
            f'<span style="background:{chip_colors.get(c, "#444")};color:#fff;'
            f'padding:2px 8px;border-radius:10px;font-size:0.72em;font-weight:bold;margin:2px;">'
            f'{chip_names.get(c, c)}</span>'
            for c in avail
        )
        badges += "".join(
            f'<span style="background:#2d2d2d;color:#9ca3af;border:1px dashed #4b5563;'
            f'padding:2px 8px;border-radius:10px;font-size:0.72em;margin:2px;">'
            f'{chip_names.get(c, c)} &middot; GW{slot2_gw}</span>'
            for c in later
        )
        chips_card = (
            f'<div style="border:1px solid #333;border-radius:10px;padding:14px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            f'text-align:center;color:#e0e0e0;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:8px;">Chips Available</div>'
            f'<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:4px;">{badges}</div>'
            f'</div>'
        )
    else:
        chips_card = stat_card_html("Chips Available", "All Used", "#6b7280")

    cols = st.columns(5)
    with cols[0]:
        st.markdown(stat_card_html("Free Transfers", ft_val, ft_color, ft_sub),
                    unsafe_allow_html=True)
    with cols[1]:
        st.markdown(stat_card_html("In the Bank", _format_money(bank)), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(stat_card_html("Squad Value", _format_money(squad_value)),
                    unsafe_allow_html=True)
    with cols[3]:
        st.markdown(stat_card_html("Active Chip", active_val, active_color, active_sub),
                    unsafe_allow_html=True)
    with cols[4]:
        st.markdown(chips_card, unsafe_allow_html=True)

    # Where the number came from, and a way to correct it. The replay cannot
    # see an ad-hoc grant -- FPL handed every manager extra free transfers
    # before GW16 of 2025/26 to absorb AFCON departures -- so a manager reading
    # the real number on FPL's own site has to be able to say so.
    if ft_info is not None and team_id is not None and current_gw is not None:
        key = _ft_override_key(team_id, current_gw)
        label = _FT_SOURCE_LABEL.get(ft_source, ft_source)
        with st.expander(f"Free transfers: {label}", expanded=False):
            if ft_source == "replay":
                st.caption(
                    "Without an FPL credential this count is replayed from your "
                    "transfer history: one per gameweek from GW2, banking to "
                    f"{MAX_BANKED_FREE_TRANSFERS}, with none granted in a "
                    "wildcard or free hit week. It cannot see a one-off grant, "
                    "and it understates when a gameweek spent part of a larger "
                    "bank. Set your credentials on the League Setup page for "
                    "FPL's own number."
                )
            elif ft_source == "my_team":
                st.caption(
                    "Taken from your authenticated FPL account: this "
                    "gameweek's allowance less the transfers you have already "
                    "made. No reconstruction involved."
                )
            else:
                st.caption(
                    f"Overriding the computed value of {info.get('computed')}. "
                    "Clear the box to go back to it."
                )

            current = st.session_state.get(key)
            stated = st.number_input(
                "FPL says I have",
                min_value=0, max_value=MAX_BANKED_FREE_TRANSFERS,
                value=int(current) if current is not None else int(free_transfers),
                step=1, key=f"{key}_input",
                help="Read the number off FPL's own Transfers page and enter it "
                     "here if it disagrees.",
            )
            c_set, c_clear = st.columns(2)
            with c_set:
                if st.button("Use this number", key=f"{key}_set"):
                    st.session_state[key] = int(stated)
                    st.rerun()
            with c_clear:
                if current is not None and st.button("Clear override", key=f"{key}_clear"):
                    del st.session_state[key]
                    st.rerun()

    st.markdown("")  # spacing


def _validate_pending_transfer(out_el: dict, in_el: dict, picks: list,
                                elements_by_id: dict, bank: int) -> Optional[str]:
    """Why this swap could not have been made, or None if it could.

    `_add_pending_local()` stored whatever it was handed, so a mis-click was
    persisted as fact and the resolved squad went on to fail
    `check_resolved_squad()` -- a validation error about an illegal squad,
    several steps removed from the typo that caused it. Checking at the point
    of entry says what is actually wrong.
    """
    if not out_el or not in_el:
        return "Could not identify one of those players."
    if int(in_el.get("id", -1)) == int(out_el.get("id", -2)):
        return "That is the same player on both sides."

    if in_el.get("element_type") != out_el.get("element_type"):
        pos_map = {1: "goalkeeper", 2: "defender", 3: "midfielder", 4: "forward"}
        return ("A transfer replaces like with like: %s is a %s, %s is a %s."
                % (out_el.get("web_name", "?"),
                   pos_map.get(out_el.get("element_type"), "?"),
                   in_el.get("web_name", "?"),
                   pos_map.get(in_el.get("element_type"), "?")))

    squad_ids = {p["element"] for p in picks}
    if int(in_el.get("id", -1)) in squad_ids:
        return "%s is already in your squad." % in_el.get("web_name", "That player")

    # Three per club, counted after the outgoing player has left.
    in_team = in_el.get("team")
    same_club = sum(1 for p in picks
                    if p["element"] != out_el.get("id")
                    and elements_by_id.get(p["element"], {}).get("team") == in_team)
    if same_club >= 3:
        return ("That would be a fourth player from the same club, which FPL "
                "does not allow.")

    funds = int(bank or 0) + int(out_el.get("now_cost", 0) or 0)
    cost = int(in_el.get("now_cost", 0) or 0)
    if cost > funds:
        return ("%s costs £%.1fm and you would have £%.1fm to spend."
                % (in_el.get("web_name", "That player"), cost / 10, funds / 10))
    return None


def _render_log_transfer_ui(team_id: int, current_gw: int,
                              picks: list, elements_by_id: dict,
                              local_pending: list, bank: int = 0) -> None:
    """Expander UI to log a pending transfer before FPL's API confirms it.

    FPL's /api/entry/{id}/transfers/ only includes confirmed (post-deadline)
    transfers. Transfers made before the deadline won't appear until the GW
    kicks off. This UI lets the user manually log such transfers so the app
    can immediately reflect the correct squad.

    **The selectboxes live in an `st.form`.** Outside one, every dropdown
    change reruns the whole page -- and this page rebuilds a ~700-player pool
    and re-runs its projection merges on every rerun, so picking two players
    paid that cost three or four times over and the UI locked up between
    clicks. Inside a form, nothing runs until submit.

    That costs the ability to narrow the incoming list to the outgoing
    player's position, since a form cannot react to its own widgets. So the
    list carries every position with the position in the label, and the match
    is checked on submit by `_validate_pending_transfer()` -- one clear error
    after submit beats a full page rebuild per click.
    """
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # Show active pending transfers above the expander with remove buttons
    if local_pending:
        for t in local_pending:
            out_el   = elements_by_id.get(t["element_out"], {})
            in_el    = elements_by_id.get(t["element_in"],  {})
            out_name = out_el.get("web_name", str(t["element_out"]))
            in_name  = in_el.get("web_name",  str(t["element_in"]))
            col_msg, col_btn = st.columns([5, 1])
            with col_msg:
                st.info(
                    f"⏳ Pending transfer applied: **{out_name} → {in_name}** (GW{t['event']})  "
                    f"— auto-clears once FPL confirms it after the deadline.",
                )
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Remove", key=f"rm_pending_{t['element_out']}"):
                    _remove_pending_local(team_id, t["element_out"])
                    st.rerun()

    with st.expander("Log a Pending Transfer", expanded=not local_pending):
        st.caption(
            "FPL only exposes confirmed transfers via its API (after the GW deadline). "
            "Log a transfer you've already made so the squad and suggestions update immediately."
        )

        squad_ids = {p["element"] for p in picks}

        # ── Player Out: sorted by ep_next ascending (worst expected points first) ──
        squad_options = []
        for pick in picks:
            el      = elements_by_id.get(pick["element"], {})
            ep_next = float(el.get("ep_next") or 0)
            name    = el.get("web_name", str(pick["element"]))
            pos     = pos_map.get(el.get("element_type"), "?")
            cost    = el.get("now_cost", 0)
            squad_options.append({
                "label":        f"{name} ({pos}, £{cost/10:.1f}m, xPts: {ep_next:.1f})",
                "id":           int(pick["element"]),
                "element_type": el.get("element_type", 3),
                "now_cost":     cost,
                "ep_next":      ep_next,
            })
        # Worst expected points at top → most likely to want to transfer out
        squad_options.sort(key=lambda o: o["ep_next"])

        if not squad_options:
            st.warning("Squad not loaded — reload the page.")
            return

        # Every position, because a form cannot narrow this list in response to
        # the Player Out choice. The position is in the label and the match is
        # checked on submit.
        in_candidates = [
            e for e in elements_by_id.values()
            if int(e["id"]) not in squad_ids
        ]
        in_candidates.sort(
            key=lambda e: (e.get("element_type", 9), -int(e.get("now_cost", 0) or 0)))

        if not in_candidates:
            st.warning("No candidates found.")
            return

        in_labels = [
            f"{e['web_name']} ({pos_map.get(e.get('element_type'), '?')}, "
            f"£{e.get('now_cost',0)/10:.1f}m)"
            for e in in_candidates
        ]

        with st.form("log_pending_transfer"):
            out_idx = st.selectbox(
                "Player Out (sorted worst → best xPts)",
                range(len(squad_options)),
                format_func=lambda i: squad_options[i]["label"],
                key="pending_out",
            )
            in_idx = st.selectbox(
                "Player In (type to search — must match the position above)",
                range(len(in_labels)),
                format_func=lambda i: in_labels[i],
                key="pending_in",
            )
            submitted = st.form_submit_button("✅ Confirm & Apply Transfer",
                                               type="primary")

        if submitted:
            out_option   = squad_options[out_idx]
            in_candidate = in_candidates[in_idx]
            out_el = elements_by_id.get(out_option["id"], {})

            problem = _validate_pending_transfer(
                out_el, in_candidate, picks, elements_by_id, bank)
            if problem:
                st.error(problem)
                return

            out_name = out_el.get("web_name", str(out_option["id"]))
            in_name  = in_candidate.get("web_name", "?")
            _add_pending_local(
                team_id=team_id,
                event=current_gw,
                element_out=out_option["id"],
                element_in=int(in_candidate["id"]),
                out_cost=out_option["now_cost"],
                in_cost=in_candidate.get("now_cost", 0),
            )
            st.session_state["_pending_transfer_success"] = (
                f"Transfer logged: **{out_name} → {in_name}**. Squad updated below."
            )
            st.rerun()


def _render_chip_advisor(chip_status: dict, squad_df: pd.DataFrame, current_gw: int):
    """Render chip strategy advisor with rule-based advice."""
    chip_names = {"wildcard": "Wildcard", "bboost": "Bench Boost",
                  "freehit": "Free Hit", "3xc": "Triple Captain"}
    chip_colors = {"wildcard": "#7c3aed", "bboost": "#166534",
                   "freehit": "#0891b2", "3xc": "#d97706"}
    chip_used_events = chip_status.get("used", {})

    with st.expander("Chip Strategy", expanded=False):
        # Chip status display
        avail = chip_status.get("available", [])
        all_chips = ["wildcard", "bboost", "freehit", "3xc"]

        avail_html = ""
        for c in all_chips:
            if c in avail:
                avail_html += (
                    f'<span style="background:{chip_colors.get(c, "#444")};color:#fff;'
                    f'padding:3px 10px;border-radius:12px;font-size:0.8em;font-weight:bold;margin:3px;">'
                    f'{chip_names.get(c, c)}</span>'
                )
            else:
                events = chip_used_events.get(c)
                if isinstance(events, list):
                    used_label = f'GW{events[0]}' if events else '?'
                else:
                    used_label = f'GW{events}' if events else 'Used'
                avail_html += (
                    f'<span style="background:#2d2d2d;color:#666;'
                    f'padding:3px 10px;border-radius:12px;font-size:0.8em;margin:3px;">'
                    f'{chip_names.get(c, c)} ({used_label})</span>'
                )

        st.markdown(
            f'<div style="margin-bottom:10px;color:#e0e0e0;">{avail_html}</div>',
            unsafe_allow_html=True
        )

        # Rule-based advice
        if "Keep Score" in squad_df.columns and not squad_df.empty:
            avg_keep = float(squad_df["Keep Score"].mean())
            weak_count = int((squad_df["Keep Score"] < 0.50).sum())

            if chip_status["wildcard_available"] and current_gw >= 25 and avg_keep < 0.45:
                st.warning(
                    f"**Wildcard Alert:** Your squad's avg Keep Score is {avg_keep:.2f} (below 0.45). "
                    "Consider using your Wildcard to rebuild with a stronger set of players."
                )
            elif chip_status["wildcard_available"] and weak_count >= 6:
                st.warning(
                    f"**Wildcard Alert:** {weak_count} players have Keep Score below 0.50. "
                    "A Wildcard rebuild could significantly improve your squad quality."
                )

            if not chip_status["wildcard_2_used"] and current_gw >= 30:
                st.info(
                    f"**WC2 available** — your second Wildcard has not been used yet. "
                    "Best deployed in GW30–35 for the run-in."
                )

        if not avail:
            st.success("All chips used — focus on optimizing weekly transfers.")


# =============================================================================
# TRANSFER PLANNER — the best squad reachable in K transfers
# =============================================================================
#
# What this replaces, `_build_multi_transfer_plan()`, could not propose the move
# that makes multiple free transfers worth having: sell a premium and a
# mid-price player, buy a premium somewhere else and a cheaper replacement.
# Three things stopped it.
#
# Its drops came from `squad_df.nsmallest(6, "Keep Score")`, so a premium was
# never a candidate to sell. Its objective was a sum of positional
# *percentiles*, which saturate near the top -- Haaland's 213.7 season points
# against a mid-price midfielder's 178.3 is 0.974 against 0.977, so the premium
# ranks *lower* -- and invert across positions, since the best of 32 forwards
# scores 0.969 where the best of 48 midfielders scores 0.979. A points-positive
# reallocation cannot win a percentile contest. And it was fixed at two legs,
# gated on having two free transfers, so 3, 4 and 5 banked all got the same
# answer.
#
# The ILP answers the question directly, in expected points, over the whole
# squad, for every K at once. Measured against a live 659-player pool: ~0.06s a
# solve, so the whole frontier costs less than half a second.

_W_NOW_KEY = "xfer_plan_w_now_pct"
_W_NEXT3_KEY = "xfer_plan_w_next3_pct"
_EXTRA_HITS_KEY = "xfer_plan_extra_hits"

#: Bench players count for something, but only a little. At exactly 0 the solver
#: is indifferent between bench compositions with the same XI, so it will spend
#: a transfer rearranging the bench for no modelled gain; `initial_squad.py`
#: records that at 0.2 it starts buying real players to sit them.
_PLAN_BENCH_WEIGHT = 0.1

#: How many candidates per position reach the solver. Three overlapping slices
#: (best, best per pound, cheapest) plus every owned player.
_PLAN_POOL_TOP = 40
_PLAN_POOL_VALUE = 15
_PLAN_POOL_CHEAP = 6


def _init_plan_weight_state(_state=None) -> None:
    state = st.session_state if _state is None else _state
    state.setdefault(_W_NOW_KEY, 40)
    state.setdefault(_W_NEXT3_KEY, 60)


def _sync_plan_weight_from_now():
    st.session_state[_W_NEXT3_KEY] = 100 - st.session_state[_W_NOW_KEY]


def _sync_plan_weight_from_next3():
    st.session_state[_W_NOW_KEY] = 100 - st.session_state[_W_NEXT3_KEY]


def plan_horizon(w_now: float, w_next3: float) -> float:
    """Gameweeks the objective is denominated over.

    `Plan_Rate` is points **per gameweek**; a -4 hit is points **once**.
    Subtracting 4 from a per-gameweek objective asks "does this gain 4 points
    every week?", which at a 3-gameweek horizon is a 3x too strict test and
    would essentially never recommend a hit. Multiplying the squad term by this
    restores the comparison.

    It is a positive constant, so it does not change *which* squad is optimal
    at a fixed hit count -- only whether the hit is worth paying. The two jobs
    are separable on purpose: the weights choose the squad, this chooses
    whether to pay for it.
    """
    return w_now * 1.0 + w_next3 * 3.0


def build_plan_scores(df: pd.DataFrame, w_now: float, w_next3: float) -> pd.Series:
    """Expected points per gameweek, blended across the chosen horizon.

    **`Proj_Next3` is a 3-gameweek total that includes the current gameweek**
    (CLAUDE.md pins this live: `Next2GWsStart == StartingPredicted + GW2` at MAE
    0.03 against 0.45 for `GW2 + GW3`). So it is divided by 3 to get a rate, and
    the two weights are not a partition of disjoint windows -- they weigh "only
    this week matters" against "the next three matter equally".

    Dividing it by 3 and setting the result beside `Proj` is only meaningful
    because the two now share a basis. They did not: the fallbacks behind
    `MultiGW_Proj` are conditional and reached `Proj_Next3` undiscounted, so an
    unmatched player's horizon rate was roughly his `Proj_Start` while his
    `Proj` was the expected value -- a median 10.6x inflation that weighting
    the horizon up would have turned into a systematic bias toward rotation
    risks. The conversion now happens in the engine, where every other basis
    change does; see "Projection Engine" in CLAUDE.md.

    This function carried a workaround for that -- trusting the horizon term
    only where FFP matched, and falling back to `Proj` otherwise. It is gone,
    so a player Rotowire priced but FFP did not keeps his fixture information
    instead of collapsing to a flat rate.
    """
    proj = numeric_col(df, "Proj", 0.0).fillna(0.0)
    next3 = (numeric_col(df, "Proj_Next3", np.nan) / 3.0).fillna(proj)
    return (w_now * proj + w_next3 * next3).fillna(0.0)


def build_plan_pool(all_players: pd.DataFrame, squad_df: pd.DataFrame,
                    w_now: float, w_next3: float) -> pd.DataFrame:
    """Candidates for the solver, in millions, with the owned 15 guaranteed in.

    Built from `all_players`, never from `available`. That frame excludes the
    owned 15 by construction and is narrowed by the position multiselect and
    the max-price slider -- display filters. Constraining the optimizer by a
    display filter gives a plan the user cannot explain, and worse, deselecting
    a position removes owned players from the pool, which turns the change
    constraint into "keep 12 of 12": a free rebuild of that position.

    Buy-eligibility is applied to non-owned rows only. An injured player you
    own must stay sellable.
    """
    if all_players is None or all_players.empty or squad_df is None or squad_df.empty:
        return pd.DataFrame()

    pool = all_players.copy()
    squad_ids = set(squad_df["Player_ID"])
    pool["Is_Owned"] = pool["Player_ID"].isin(squad_ids)

    # The bootstrap publishes no selling price, so `all_players` carries
    # now_cost in that column for everyone. Overwrite it for the 15 we own with
    # what FPL will actually pay.
    sell = squad_df.set_index("Player_ID").apply(_selling_price, axis=1)
    pool["_sell"] = pool["Player_ID"].map(sell)
    pool["Price"] = pd.to_numeric(pool["now_cost"], errors="coerce") / 10.0
    pool["Sell_Price"] = np.where(pool["_sell"].notna(),
                                  pool["_sell"], pool["now_cost"]) / 10.0

    pool["Plan_Score"] = build_plan_scores(pool, w_now, w_next3)

    eligible = (
        pool["status"].isin(["a", "d"])
        & (pd.to_numeric(pool["minutes"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(pool["chance_of_playing_next_round"],
                         errors="coerce").fillna(100) >= 50)
    )
    pool = pool[eligible | pool["Is_Owned"]].copy()

    # Trim per position. The value-per-pound and cheapest slices are not
    # decoration: the optimal plan is often "downgrade the bench keeper to fund
    # the premium", which a best-N-only pool cannot express.
    keep_idx = set(pool.index[pool["Is_Owned"]])
    per_pound = pool["Plan_Score"] / pool["Price"].replace(0, np.nan)
    for pos in pool["Position"].dropna().unique():
        at_pos = pool[pool["Position"] == pos]
        keep_idx |= set(at_pos.nlargest(_PLAN_POOL_TOP, "Plan_Score").index)
        keep_idx |= set(per_pound.loc[at_pos.index].nlargest(_PLAN_POOL_VALUE).index)
        keep_idx |= set(at_pos.nsmallest(_PLAN_POOL_CHEAP, "Price").index)

    return pool.loc[sorted(keep_idx)].reset_index(drop=True)


def build_transfer_plan(pool: pd.DataFrame, squad_df: pd.DataFrame, bank: int,
                        free_transfers: int, max_extra_hits: int = 0,
                        w_now: float = 0.4, w_next3: float = 0.6) -> Optional[Dict]:
    """Solve every K from 0 upward and return the frontier plus the best plan.

    **K=0 is the baseline**, solved with the identical objective rather than
    read off the current starting XI. `find_optimal_lineup()` is greedy, has no
    captain term and no bench term, so comparing against it manufactures a
    fraction of a point of "gain" for a plan that changes nothing.

    It is also the cheapest feasibility canary: pricing kept players at their
    selling price makes K=0 feasible by construction, so if it fails the squad
    is structurally broken -- more than three from a club, a duplicated pick,
    prices in the wrong units -- and that is worth saying rather than rendering
    a plan on top of it.

    Solving the whole frontier rather than one K is nearly free and is the more
    useful answer: on a live squad it read 1 transfer +4.7, 2 +8.9, 3 +11.9,
    and K=4 and K=5 returned the *same* 3-transfer plan. "Three is the sweet
    spot, a fourth adds nothing" beats any single number.
    """
    if pool is None or pool.empty or squad_df is None or squad_df.empty:
        return None

    owned = [int(p) for p in squad_df["Player_ID"]]
    budget = (float(bank or 0) / 10.0) + float(
        pool.loc[pool["Player_ID"].isin(owned), "Sell_Price"].sum())
    horizon = plan_horizon(w_now, w_next3)
    k_max = max(0, int(free_transfers) + int(max_extra_hits))

    def _solve(k):
        return solve_squad_ilp(
            pool, budget, score_col="Plan_Score", price_col="Price",
            formation="auto", bench_weight=_PLAN_BENCH_WEIGHT,
            # The armband doubles a starter's score, and without it the model
            # under-prices losing your captain by exactly one more copy of it --
            # which is the whole question when the plan proposes selling him.
            captain_score_col="Plan_Score", captain_bonus_weight=1.0,
            problem_name="FPL_Transfer_Planner",
            owned_ids=owned, id_col="Player_ID", max_changes=k,
            sell_price_col="Sell_Price", free_transfers=int(free_transfers),
            time_limit=20,
        )

    def _value(totals):
        """The objective, in points over the horizon, net of any hit."""
        gross = (totals["starter_score"]
                 + _PLAN_BENCH_WEIGHT * totals["bench_score"]
                 + totals.get("captain_score", 0.0))
        return horizon * gross - HIT_COST * totals.get("hits", 0)

    base_squad, base_totals = _solve(0)
    if base_squad is None:
        return {"error": "baseline"}

    base_value = _value(base_totals)
    frontier, best = [], None
    for k in range(1, k_max + 1):
        squad, totals = _solve(k)
        if squad is None:
            continue
        gain = _value(totals) - base_value
        entry = {"k": k, "squad": squad, "totals": totals, "gain": gain,
                 "n_changes": totals["n_changes"], "hits": totals.get("hits", 0)}
        frontier.append(entry)
        if best is None or gain > best["gain"] + 1e-9:
            best = entry

    if best is None or best["n_changes"] == 0 or best["gain"] <= 1e-9:
        return {"hold": True, "frontier": frontier, "horizon": horizon,
                "baseline": base_value}

    outs, ins = diff_squads(squad_df, best["squad"])
    # `squad_df` carries selling prices in tenths; the solver worked in millions.
    outs = outs.copy()
    outs["Sell_Price"] = outs.apply(_selling_price, axis=1) / 10.0
    outs["Plan_Score"] = build_plan_scores(outs, w_now, w_next3)

    legs = pair_transfer_legs(outs, ins, score_col="Plan_Score")
    released = sum(leg["out_price"] for leg in legs)
    spent = sum(leg["in_price"] for leg in legs)
    bank_before = float(bank or 0) / 10.0

    return {
        "legs": legs,
        "squad_after": best["squad"],
        "max_changes": best["k"],
        "free_transfers": int(free_transfers),
        "hits": int(best["hits"]),
        "n_changes": int(best["n_changes"]),
        "bank_before": bank_before,
        "bank_after": bank_before + released - spent,
        "released": released,
        "spent": spent,
        "gain_net": best["gain"],
        "horizon_gws": horizon,
        "frontier": frontier,
        "baseline": base_value,
    }


def _plan_stat_card(label: str, value: str, accent: str = "#00ff87",
                    subtitle: str = "") -> str:
    sub = (f'<div style="color:#aaa;font-size:11px;margin-top:4px;">{subtitle}</div>'
           if subtitle else "")
    return compact_html(
        f'<div style="border:1px solid #333;border-radius:10px;padding:14px;'
        f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
        f'text-align:center;color:#e0e0e0;height:100%;">'
        f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
        f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
        f'<div style="color:{accent};font-size:20px;font-weight:700;">{value}</div>'
        f'{sub}</div>'
    )


def _plan_leg_rationale(leg: Dict, released: float) -> str:
    """One line saying what this leg is *for*.

    For a reallocation the plan's value is invisible in any single leg: "sell
    the premium forward to fund a premium midfielder" renders as one card that
    looks like a downgrade and one that looks unaffordable, and neither can
    explain the other. This is the sentence that connects them.
    """
    net = leg["net_cost"]
    delta = leg.get("delta")
    delta_txt = f"{delta:+.1f} pts/GW" if delta is not None else ""
    if net < -0.05:
        return (f"Frees £{-net:.1f}m for the rest of the plan"
                + (f" at a cost of {delta_txt}" if delta is not None and delta < 0
                   else (f", and gains {delta_txt}" if delta_txt else "")))
    if net > 0.05:
        return f"Upgrade — spends £{net:.1f}m of the £{released:.1f}m released"
    return f"Straight swap{', ' + delta_txt if delta_txt else ''}"


def _render_transfer_plan(plan: Optional[Dict], free_transfers: int,
                          w_now: float, w_next3: float) -> None:
    """The planner's output: headline, funding line, legs, frontier."""
    if not plan:
        return

    if plan.get("error") == "baseline":
        st.warning(
            "Could not price your current squad, so there is nothing to compare "
            "a plan against. That usually means the squad is illegal as loaded "
            "— more than three players from one club, or a duplicated pick. "
            "Check the squad source above, or use Refresh."
        )
        return

    horizon = plan.get("horizon_gws", plan_horizon(w_now, w_next3))

    if plan.get("hold"):
        st.success(
            f"**Hold your transfer{'s' if free_transfers != 1 else ''}.** No "
            f"move improves this squad over the next {horizon:.1f} gameweeks at "
            f"the current split — banking gives you a wider choice next week."
        )
        return

    issues = check_transfer_plan(plan, plan.get("squad_after"))
    blocking = [i for i in issues if i.severity == "error"]
    if blocking:
        _logger.error("Transfer plan failed validation: %s", format_issues(issues))
        st.warning(
            "A transfer plan was found but did not pass its own sanity checks, "
            "so it is not being shown. The single-transfer suggestions below "
            "are unaffected."
        )
        return
    if issues:
        _logger.warning("Transfer plan: %s", format_issues(issues))

    legs = plan["legs"]
    hits = plan["hits"]
    n = len(legs)

    tf_sub = (f"{n - hits} free + {hits} hit" if hits
              else ("all free" if n else ""))
    gain = plan["gain_net"]
    cols = st.columns(4)
    with cols[0]:
        st.markdown(_plan_stat_card(
            "Transfers", str(n), "#4ecca3", tf_sub), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(_plan_stat_card(
            "Net Gain", f"{gain:+.1f} pts",
            "#00ff87" if gain > 0 else "#f87171",
            f"over {horizon:.1f} GWs, after any hit"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(_plan_stat_card(
            "Bank After", f"£{plan['bank_after']:.1f}m", "#e0e0e0",
            f"from £{plan['bank_before']:.1f}m"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(_plan_stat_card(
            "Points Hit", f"&minus;{4 * hits} pts" if hits else "None",
            "#f87171" if hits else "#6b7280",
            "already netted off above" if hits else "within your free transfers"),
            unsafe_allow_html=True)

    # The funding line, above the legs: for a reallocation this is the only
    # place the plan makes sense as a whole.
    outs = ", ".join(f"{l['out_player']} £{l['out_price']:.1f}m" for l in legs)
    ins = ", ".join(f"{l['in_player']} £{l['in_price']:.1f}m" for l in legs)
    st.caption(
        f"Releases **£{plan['released']:.1f}m** ({outs}) · spends "
        f"**£{plan['spent']:.1f}m** ({ins}) · bank £{plan['bank_before']:.1f}m → "
        f"£{plan['bank_after']:.1f}m"
    )

    for i, leg in enumerate(legs, 1):
        delta = leg.get("delta")
        delta_txt = f"{delta:+.1f} pts/GW" if delta is not None else "—"
        delta_color = "#00ff87" if (delta or 0) >= 0 else "#f87171"
        # A hit is a property of the plan, not of a leg. Marking the last one is
        # presentational, so it says so rather than implying this move caused it.
        hit_badge = ("" if not hits or i != len(legs) else
                     '<span style="background:#7f1d1d;color:#fecaca;padding:2px 8px;'
                     'border-radius:10px;font-size:0.72em;font-weight:bold;'
                     f'margin-left:8px;">HIT &minus;{4 * hits}</span>')
        st.markdown(compact_html(
            f'<div style="border:1px solid #333;border-radius:10px;padding:12px 16px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            f'color:#e0e0e0;margin-bottom:8px;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px;">'
            f'Leg {i} of {len(legs)} &middot; {leg["position"]}{hit_badge}</div>'
            f'<div style="font-size:16px;font-weight:600;">'
            f'<span style="color:#f87171;">{leg["out_player"]}</span> '
            f'<span style="color:#9ca3af;">£{leg["out_price"]:.1f}m</span>'
            f' &nbsp;→&nbsp; '
            f'<span style="color:#00ff87;">{leg["in_player"]}</span> '
            f'<span style="color:#9ca3af;">£{leg["in_price"]:.1f}m</span>'
            f'<span style="float:right;color:{delta_color};font-weight:700;">{delta_txt}</span>'
            f'</div>'
            f'<div style="color:#aaa;font-size:12px;margin-top:6px;">'
            f'{_plan_leg_rationale(leg, plan["released"])}</div>'
            f'</div>'
        ), unsafe_allow_html=True)

    frontier = plan.get("frontier") or []
    if len(frontier) > 1:
        parts = []
        seen = set()
        for e in frontier:
            if e["n_changes"] in seen:
                continue
            seen.add(e["n_changes"])
            label = f"{e['n_changes']} tf"
            if e["hits"]:
                label += f" (−{4 * e['hits']})"
            parts.append(f"{label} **{e['gain']:+.1f}**")
        extra = ""
        if max(e["n_changes"] for e in frontier) < max(e["k"] for e in frontier):
            extra = " — more transfers than that gain nothing"
        st.caption("Worth per number of transfers: " + " · ".join(parts) + extra)


def _selling_price(row) -> float:
    """What a squad player releases into the bank when sold."""
    price = row.get("selling_price")
    if price is None or (isinstance(price, float) and pd.isna(price)):
        price = row.get("now_cost", 0)
    return float(price or 0)


def _plan_card(drop_row, add_row, pos_labels: Dict, depth_map: Optional[Dict],
               funds: float, outlay: float) -> Dict:
    """One leg of a multi-transfer plan, in the shape the card renderer wants."""
    pos = drop_row["Position"]
    add_form_col = "HealthyForm" if "HealthyForm" in add_row.index else "form"
    drop_form_col = "HealthyForm" if "HealthyForm" in drop_row.index else "form"
    proj = _blended_proj(add_row)
    return {
        "position": pos_labels.get(pos, pos),
        "score_diff": float(add_row.get("Transfer Score", 0)) - float(drop_row.get("Keep Score", 0)),
        "drop_player": drop_row["Player"],
        "drop_full_name": drop_row.get("Full Name") or drop_row["Player"],
        "drop_team": drop_row["Team"],
        "drop_price": f"£{_selling_price(drop_row)/10:.1f}m",
        "drop_form": f"{float(drop_row.get(drop_form_col, 0) or 0):.1f}",
        "drop_season_pts": drop_row.get("total_points", 0),
        "drop_injury": _get_availability_indicator(
            drop_row.get("chance_of_playing_next_round"), drop_row.get("news", "")),
        "add_player": add_row["Player"],
        "add_full_name": add_row.get("Full Name") or add_row["Player"],
        "add_team": add_row["Team"],
        "add_price": f"£{add_row['now_cost']/10:.1f}m",
        "add_form": f"{float(add_row.get(add_form_col, 0) or 0):.1f}",
        "add_proj_pts": f"{proj:.1f}" if pd.notna(proj) else "N/A",
        "add_injury": _get_availability_indicator(
            add_row.get("chance_of_playing_next_round"), add_row.get("news", "")),
        "rationale": "Part of optimal 2-transfer plan",
        "urgency": compute_transfer_urgency(pos, depth_map) if depth_map else "",
        "ep_delta": None,
        "price_trend": None,
        "add_ownership_badge": _ownership_badge(add_row.get("selected_by_percent", 0)),
        "hit_verdict": None,
        "plan_label": "2-Transfer Plan (Both Free)",
        # Budget the whole plan was solved against, carried on each leg so the
        # renderer can show what the pair actually costs.
        "plan_funds": funds,
        "plan_outlay": outlay,
    }


def _build_multi_transfer_plan(squad_df: pd.DataFrame, available_df: pd.DataFrame,
                                bank: int, depth_map: Optional[Dict] = None,
                                candidates_per_position: int = 25) -> List[Dict]:
    """Find the best *affordable* pair of transfers when both are free.

    Affordability is a **joint** constraint. Both incoming players are bought
    out of one pot -- the bank plus both selling prices -- so pricing each add
    against the whole pot independently is how this came to propose two
    premiums that could not be bought together: Gabriel (£8.0m) and Rogers
    (£7.6m) against a pot of £10.7m, each of which cleared the test on its own.

    The same "legal as a pair, not one transfer at a time" rule applies twice
    more, and both were broken for the same reason: the two legs could name the
    **same player** (identical position, identical candidate list, identical
    winner), and two adds from one club could take that club to four.
    """
    if squad_df.empty or available_df.empty or "Keep Score" not in squad_df.columns:
        return []

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}

    # Best-first within a position, so a truncated candidate list keeps the
    # players worth having. The caller sorts this way already; do not rely on it.
    pool = available_df
    if "Transfer Score" in pool.columns:
        pool = pool.sort_values("Transfer Score", ascending=False)

    drop_list = [row for _, row in squad_df.nsmallest(6, "Keep Score").iterrows()]

    best_score = -999.0
    best: Optional[tuple] = None

    for i, drop1 in enumerate(drop_list):
        for drop2 in drop_list[i + 1:]:
            if drop1["Player_ID"] == drop2["Player_ID"]:
                continue

            funds = bank + _selling_price(drop1) + _selling_price(drop2)
            squad_without = squad_df[
                ~squad_df["Player_ID"].isin([drop1["Player_ID"], drop2["Player_ID"]])
            ]
            team_counts = squad_without["Team"].value_counts().to_dict()

            by_position = {}
            for pos in {drop1["Position"], drop2["Position"]}:
                affordable = pool[(pool["Position"] == pos) & (pool["now_cost"] <= funds)]
                by_position[pos] = [r for _, r in affordable.head(candidates_per_position).iterrows()]

            for add1 in by_position[drop1["Position"]]:
                for add2 in by_position[drop2["Position"]]:
                    if add1["Player_ID"] == add2["Player_ID"]:
                        continue
                    if float(add1["now_cost"]) + float(add2["now_cost"]) > funds:
                        continue
                    counts = dict(team_counts)
                    legal = True
                    for add in (add1, add2):
                        team = add.get("Team")
                        counts[team] = counts.get(team, 0) + 1
                        if counts[team] > 3:
                            legal = False
                            break
                    if not legal:
                        continue

                    combined = (
                        float(add1.get("Transfer Score", 0)) - float(drop1.get("Keep Score", 0))
                        + float(add2.get("Transfer Score", 0)) - float(drop2.get("Keep Score", 0))
                    )
                    if combined > best_score:
                        best_score = combined
                        best = (drop1, add1, drop2, add2, funds)

    if best is None:
        return []

    drop1, add1, drop2, add2, funds = best
    outlay = float(add1["now_cost"]) + float(add2["now_cost"])
    return [
        _plan_card(drop1, add1, pos_labels, depth_map, funds, outlay),
        _plan_card(drop2, add2, pos_labels, depth_map, funds, outlay),
    ]


def _render_multi_transfer_plan(plan: List[Dict], free_transfers: int = 2):
    """Render the optimal 2-transfer plan side-by-side."""
    if not plan:
        return

    st.subheader("2-Transfer Plan (Both Free)")
    # The section only renders when the transfers are banked, so stating the
    # condition reads as a hypothetical. State the fact instead.
    banked = "2 free transfers" if free_transfers == 2 else f"{free_transfers} free transfers"
    st.caption(f"You have {banked} banked — this is the best pair to spend two of them on, "
               "within your budget.")

    cols = st.columns(2)
    for col, s in zip(cols, plan):
        with col:
            card_html = f"""
            <div style="border: 1px solid #444; border-radius: 10px; padding: 16px; margin-bottom: 12px;
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e0e0e0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span style="background: #0f3460; color: #e0e0e0; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">{s['position']}</span>
                    <span style="background: #1a472a; color: #4ecca3; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">+{s['score_diff']:.3f}</span>
                </div>
                <div style="color: #e74c3c; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">DROP</div>
                <div style="color: #e0e0e0; font-weight: bold;">{s['drop_full_name']} ({s['drop_team']})</div>
                <div style="color: #999; font-size: 0.82em; margin-bottom: 8px;">
                    {s['drop_price']} &bull; Form: {s['drop_form']} &bull; {s['drop_injury']}
                </div>
                <div style="color: #4ecca3; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">ADD</div>
                <div style="color: #e0e0e0; font-weight: bold;">{s['add_full_name']} ({s['add_team']})</div>
                <div style="color: #999; font-size: 0.82em;">
                    {s['add_price']} &bull; Proj: {s['add_proj_pts']} &bull; {s['add_injury']}
                    {s.get('add_ownership_badge', '')}
                </div>
            </div>
            """
            st.markdown(compact_html(card_html), unsafe_allow_html=True)

    funds = plan[0].get("plan_funds")
    outlay = plan[0].get("plan_outlay")
    if funds is not None and outlay is not None:
        st.caption(
            f"Both transfers are free this gameweek — optimal pair based on Transfer/Keep Scores. "
            f"Cost £{outlay/10:.1f}m of the £{funds/10:.1f}m you'd have available "
            f"(bank plus both sales), leaving £{(funds - outlay)/10:.1f}m."
        )
    else:
        st.caption("Both transfers are free this gameweek — optimal pair based on Transfer/Keep Scores.")


def _get_blanking_team_ids(current_gw: int, bootstrap: dict) -> set:
    """Return team IDs with no fixture in the current GW (blank GW teams)."""
    all_team_ids = {t["id"] for t in bootstrap.get("teams", [])}
    fixtures = _load_future_fixtures()
    if fixtures.empty:
        return set()
    gw_fixtures = fixtures[fixtures["event"] == current_gw]
    teams_with_fixture: set = set()
    for _, row in gw_fixtures.iterrows():
        if pd.notna(row.get("team_h")):
            teams_with_fixture.add(int(row["team_h"]))
        if pd.notna(row.get("team_a")):
            teams_with_fixture.add(int(row["team_a"]))
    return all_team_ids - teams_with_fixture


def _render_blank_gw_alert(squad_df: pd.DataFrame, blanking_team_ids: set, current_gw: int):
    """Warn when starting XI players have blank GWs and surface hit-value context."""
    if not blanking_team_ids or squad_df.empty or "squad_position" not in squad_df.columns:
        return
    blanking_starters = squad_df[
        squad_df["Team_ID"].isin(blanking_team_ids) &
        (squad_df["squad_position"] <= 11)
    ]
    if blanking_starters.empty:
        return

    names = ", ".join(
        f"**{r['Player']}** ({r['Team']})" for _, r in blanking_starters.iterrows()
    )
    ep_vals = [float(r.get("ep_next", 0) or 0) for _, r in blanking_starters.iterrows()]
    max_ep = max(ep_vals) if ep_vals else 0

    msg = (
        f"**GW{current_gw} Blank Alert — {len(blanking_starters)} starter(s) have no fixture:** "
        f"{names}. "
    )
    if max_ep >= 4.0:
        msg += (
            f"Best replacement EP: ~{max_ep:.1f} pts — worth a **−4 hit** "
            f"(net +{max_ep - 4:.1f} pts)."
        )
    elif max_ep > 0:
        msg += (
            f"Best replacement EP: ~{max_ep:.1f} pts — marginally worth a hit "
            f"(net {max_ep - 4:+.1f} pts)."
        )
    st.warning(msg)


def _build_transfer_suggestions(squad_df: pd.DataFrame, available_df: pd.DataFrame,
                                 bank: int, top_n: int = 3, depth_map: Optional[Dict] = None,
                                 free_transfers: int = 1,
                                 blanking_team_ids: Optional[set] = None) -> List[Dict]:
    """Build transfer suggestions pairing lowest-keep-score squad players with best replacements."""
    if squad_df.empty or available_df.empty:
        return []

    if blanking_team_ids is None:
        blanking_team_ids = set()

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}
    suggestions = []
    remaining_ft = free_transfers

    # Phase 1: Priority candidates — blank GW starters with no positional cover.
    # These are selected first regardless of Keep Score, since they will score 0
    # if played and there's no available sub to cover them.
    priority_seen: set = set()
    priority_candidates = []
    if blanking_team_ids:
        for _, row in squad_df.iterrows():
            pos = row["Position"]
            is_starter_p = int(row.get("squad_position", 12) or 12) <= 11
            if not is_starter_p or row.get("Team_ID") not in blanking_team_ids:
                continue
            squad_same = squad_df[
                (squad_df["Position"] == pos) & (squad_df["Player_ID"] != row["Player_ID"])
            ]
            has_cover = any(
                r["Team_ID"] not in blanking_team_ids for _, r in squad_same.iterrows()
            )
            if has_cover:
                continue  # position is covered by a non-blanking squadmate — not urgent
            if pos not in priority_seen:
                priority_seen.add(pos)
                priority_candidates.append(row)
            if len(priority_candidates) == top_n:
                break

    # Phase 2: Fill remaining slots from lowest Keep Score (one per position max).
    # Prevents e.g. two GK suggestions both recommending the same replacement.
    seen_positions: set = set(r["Position"] for r in priority_candidates)
    drop_candidates = list(priority_candidates)
    for _, row in squad_df.nsmallest(top_n * 4, "Keep Score").iterrows():
        pos = row["Position"]
        if pos not in seen_positions:
            seen_positions.add(pos)
            drop_candidates.append(row)
        if len(drop_candidates) == top_n:
            break

    # A player can only be bought once, so a target claimed by an earlier
    # suggestion has to leave the next drop its runner-up. Without this the same
    # standout replacement was offered against every weak squad player, which
    # reads as a plan but is one transfer written several times.
    #
    # Drops are deliberately *not* re-sorted by gain the way the Draft page does
    # it: drop_candidates is already ordered by urgency -- unavailable players
    # and uncovered blanks first -- and that ordering is more useful here than
    # raw score improvement, so first-come-first-served on it gives the best
    # target to the player who most needs replacing.
    used_add_ids: set = set()

    # A veto that can see none of its inputs passes everything while looking
    # exactly like protection. Counted so a frame missing _effective_proj /
    # total_points / MultiGW_Proj surfaces as a warning rather than as silence.
    sanity_evaluated = 0
    sanity_blind = 0

    for drop_row in drop_candidates:
        pos = drop_row["Position"]
        drop_id = drop_row["Player_ID"]
        selling_price = drop_row.get("selling_price", drop_row.get("now_cost", 0))
        budget = bank + selling_price

        # Find best replacement at same position within budget
        candidates = available_df[
            (available_df["Position"] == pos) &
            (available_df["now_cost"] <= budget)
        ].copy()

        if candidates.empty:
            continue

        # Club rule: max 3 players from same club — skip candidates that would breach it
        squad_without_drop = squad_df[squad_df["Player_ID"] != drop_id]
        add_row = None
        for _, candidate in candidates.head(10).iterrows():
            if candidate.get("Player_ID") in used_add_ids:
                continue  # already suggested for another drop — take the next best
            cand_team = candidate.get("Team")
            if (squad_without_drop["Team"] == cand_team).sum() >= 3:
                continue  # would violate 3-per-club rule
            add_row = candidate
            break

        if add_row is None:
            # Fallback if every top-10 candidate breaches the club rule or is
            # already spoken for. Still must not re-suggest a claimed player.
            _free = candidates[~candidates["Player_ID"].isin(used_add_ids)]
            if _free.empty:
                continue
            add_row = _free.iloc[0]

        # Calculate score improvement using Transfer Score vs Keep Score
        score_diff = add_row.get("Transfer Score", 0) - drop_row.get("Keep Score", 0)

        # --- Threshold logic ---
        keep_score = float(drop_row.get("Keep Score", 0.5) or 0.5)
        drop_chance = drop_row.get("chance_of_playing_next_round")
        drop_status = drop_row.get("status", "a")

        # Season-ending / 0% injury: bypass elite protection entirely
        is_unavailable = (
            (drop_chance is not None and float(drop_chance) == 0) and
            drop_status in ("i", "u", "s")
        )

        # Blank GW for a starting XI player
        is_starter = int(drop_row.get("squad_position", 12) or 12) <= 11
        has_blank = drop_row.get("Team_ID") in blanking_team_ids and is_starter

        # Check if same position has a non-blanking backup (position is covered)
        squad_same_pos = squad_df[
            (squad_df["Position"] == pos) & (squad_df["Player_ID"] != drop_id)
        ]
        has_non_blanking_backup = any(
            r["Team_ID"] not in blanking_team_ids
            for _, r in squad_same_pos.iterrows()
        )

        if is_unavailable:
            min_threshold = 0.01   # Always suggest — they're not playing at all
        elif has_blank and not has_non_blanking_backup:
            min_threshold = 0.01   # No cover for this blank — treat like unavailable
        elif has_blank and has_non_blanking_backup:
            min_threshold = 0.20   # Covered by a teammate — very low priority
        elif keep_score > 0.7:
            min_threshold = 0.15
        elif keep_score > 0.5:
            min_threshold = 0.08
        else:
            min_threshold = 0.02

        # GKs are near-guaranteed starters; a backup can cover one bad GW.
        # Classic transfers are precious — require a massive improvement for GK swaps.
        if pos == "G" and not is_unavailable:
            min_threshold = max(min_threshold, 0.25)

        if score_diff < min_threshold:
            continue

        # Clearing the score threshold is not enough. Transfer Score is a blend
        # of percentiles, and it can rank the incoming player higher while every
        # raw number a manager would look at says the opposite. Draft has vetoed
        # that since the waiver engine was written; Classic did not, which is
        # the worse way round -- a waiver claim is free and this can cost a -4.
        sanity_ok, sanity_reason = sanity_check_suggestion(drop_row, add_row)
        sanity_evaluated += 1
        if sanity_reason == "no data":
            sanity_blind += 1
        if not sanity_ok:
            _logger.info("Transfers: vetoed %s -> %s (%s)",
                         drop_row.get("Player"), add_row.get("Player"), sanity_reason)
            continue

        # Availability info
        drop_news = drop_row.get("news", "")
        drop_injury = _get_availability_indicator(drop_chance, drop_news)

        add_chance = add_row.get("chance_of_playing_next_round")
        add_news = add_row.get("news", "")
        add_injury = _get_availability_indicator(add_chance, add_news)

        # Build rationale
        reasons = []
        add_form_col = "HealthyForm" if "HealthyForm" in add_row.index else "form"
        drop_form_col = "HealthyForm" if "HealthyForm" in drop_row.index else "form"

        if is_unavailable:
            reasons.append("out for the season / unavailable — empty squad spot")
        elif has_blank and not has_non_blanking_backup:
            reasons.append(f"blank GW{drop_row.get('Team', '')} — no cover in this position")
        elif has_blank and has_non_blanking_backup:
            backup = squad_same_pos[~squad_same_pos["Team_ID"].isin(blanking_team_ids)].iloc[0]
            reasons.append(f"covered by {backup['Player']} this week — low priority")

        form_diff = float(add_row.get(add_form_col, 0) or 0) - float(drop_row.get(drop_form_col, 0) or 0)
        if form_diff > 0:
            reasons.append(f"+{form_diff:.1f} form improvement")
        proj_add = _blended_proj(add_row)
        proj_drop = _blended_proj(drop_row)
        if pd.notna(proj_add) and pd.notna(proj_drop) and proj_add > proj_drop:
            reasons.append(f"+{proj_add - proj_drop:.1f} projected points")
        # The engine's converted horizon, and only where both sides resolve to
        # the same column -- `Proj_Next3` is expected points while
        # `MultiGW_Proj` can be a conditional "if he starts" total, so mixing
        # them charges a rotation risk to one player only.
        if horizon_column(add_row) is not None and \
                horizon_column(add_row) == horizon_column(drop_row):
            add_multi = horizon_points(add_row)
            drop_multi = horizon_points(drop_row)
            if add_multi > drop_multi and add_multi > 0:
                reasons.append(f"3GW outlook: {add_multi:.1f} vs {drop_multi:.1f} pts")
        add_fdr = add_row.get("AvgFDR")
        drop_fdr = drop_row.get("AvgFDR")
        if pd.notna(add_fdr) and pd.notna(drop_fdr) and add_fdr < drop_fdr:
            reasons.append("easier upcoming fixtures")
        if drop_news and not is_unavailable:
            reasons.append(f"current player: {drop_news[:40]}")

        rationale = " • ".join(reasons) if reasons else "Better overall transfer score"

        if is_unavailable:
            urgency = "URGENT"
        elif has_blank and not has_non_blanking_backup:
            urgency = "BLANK GW"
        elif has_blank and has_non_blanking_backup:
            urgency = "LOW PRIORITY"
        elif pos == "G":
            urgency = "GK CAUTION"
        else:
            urgency = compute_transfer_urgency(pos, depth_map) if depth_map else ""

        add_form_val = float(add_row.get(add_form_col, 0) or 0)
        drop_form_val = float(drop_row.get(drop_form_col, 0) or 0)

        # ep_next delta — use 0 for the drop player if they're blanking or unavailable
        ep_next_add = float(add_row.get("ep_next", 0) or 0)
        if is_unavailable or (has_blank and is_starter):
            ep_next_drop = 0.0  # Blank/unavailable players score nothing this week
        else:
            ep_next_drop = float(drop_row.get("ep_next", 0) or 0)
        ep_delta = ep_next_add - ep_next_drop

        # Price trend and ownership intelligence
        price_trend = _format_price_trend(
            add_row.get("cost_change_event", 0),
            add_row.get("transfers_in_event", 0),
            add_row.get("transfers_out_event", 0),
        )
        add_ownership_badge = _ownership_badge(add_row.get("selected_by_percent", 0))
        add_ownership_pct = float(add_row.get("selected_by_percent", 0) or 0)

        # Hit verdict
        is_hit = remaining_ft <= 0
        hit_verdict = _compute_hit_verdict(ep_delta, is_hit)
        remaining_ft = max(0, remaining_ft - 1)

        # Claim the target only now that the pairing has actually cleared its
        # threshold -- marking it earlier would burn a good replacement on a
        # suggestion that was never made.
        used_add_ids.add(add_row.get("Player_ID"))

        suggestions.append({
            "position": pos_labels.get(pos, pos),
            "score_diff": score_diff,
            # Element ids so the planner's legs can be matched against these
            # cards without going through names.
            "drop_id": drop_row.get("Player_ID"),
            "add_id": add_row.get("Player_ID"),
            "drop_player": drop_row["Player"],
            "drop_full_name": drop_row.get("Full Name") or drop_row["Player"],
            "drop_team": drop_row["Team"],
            "drop_price": f"£{drop_row['now_cost']/10:.1f}m",
            "drop_form": f"{drop_form_val:.1f}",
            "drop_season_pts": drop_row.get("total_points", 0),
            "drop_injury": drop_injury,
            "add_player": add_row["Player"],
            "add_full_name": add_row.get("Full Name") or add_row["Player"],
            "add_team": add_row["Team"],
            "add_price": f"£{add_row['now_cost']/10:.1f}m",
            "add_form": f"{add_form_val:.1f}",
            "add_proj_pts": f"{proj_add:.1f}" if pd.notna(proj_add) else "N/A",
            "add_injury": add_injury,
            "rationale": rationale,
            "urgency": urgency,
            "ep_delta": ep_delta,
            "ep_next_add": ep_next_add,
            "ep_next_drop": ep_next_drop,
            "price_trend": price_trend,
            "add_ownership_badge": add_ownership_badge,
            "add_ownership_pct": add_ownership_pct,
            "hit_verdict": hit_verdict,
        })

    if sanity_evaluated and sanity_blind == sanity_evaluated:
        _logger.warning(
            "Transfers: the sanity veto saw no comparable metrics on any of %d "
            "candidate swaps -- it is passing everything. Check the squad frame "
            "still carries _effective_proj / total_points / MultiGW_Proj.",
            sanity_evaluated)

    return suggestions


def _render_depth_card(depth_map: Dict):
    """Render a compact horizontal positional depth summary."""
    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}
    level_colors = {'Critical': '#dc3545', 'Low': '#ff9800', 'Adequate': '#4ecca3'}

    items_html = []
    for pos_code in ['G', 'D', 'M', 'F']:
        depth = depth_map.get(pos_code)
        if depth is None or depth.total == 0:
            continue
        label = pos_labels.get(pos_code, pos_code)
        color = level_colors.get(depth.depth_level, '#888')
        # Three-state dots, uniform color from depth level
        # Use inline CSS circles for consistent sizing
        dot_full = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                    f'background:{color};margin:0 1px;vertical-align:middle;"></span>')
        dot_half = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                    f'background:linear-gradient(90deg,{color} 50%,transparent 50%);'
                    f'border:1.5px solid {color};box-sizing:border-box;'
                    f'margin:0 1px;vertical-align:middle;"></span>')
        dot_empty = (f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                     f'border:1.5px solid {color};box-sizing:border-box;'
                     f'margin:0 1px;vertical-align:middle;"></span>')
        dots_html = (dot_full * depth.healthy) + (dot_half * depth.doubtful) + (dot_empty * depth.injured)
        # Count doubtful as 0.5
        effective = depth.healthy + depth.doubtful * 0.5
        count_str = f"{effective:g}/{depth.total}"
        level_text = depth.depth_level if depth.depth_level != "Adequate" else ""
        level_span = (
            f'<span style="color:{color};font-weight:bold;font-size:0.8em;margin-left:4px;">'
            f'{level_text}</span>' if level_text else ""
        )
        items_html.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="font-weight:bold;color:#e0e0e0;">{label}</span>'
            f'<span style="color:#aaa;font-size:0.85em;">{count_str}</span>'
            f'<span style="display:inline-flex;align-items:center;">{dots_html}</span>'
            f'{level_span}'
            f'</div>'
        )

    if items_html:
        card = (
            '<div style="border:1px solid #444;border-radius:8px;padding:10px 16px;margin-bottom:10px;'
            'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);color:#e0e0e0;'
            'display:flex;align-items:center;gap:24px;flex-wrap:wrap;">'
            '<span style="font-weight:bold;font-size:0.9em;color:#aaa;">Squad Depth</span>'
            + ''.join(items_html)
            + '</div>'
        )
        st.markdown(card, unsafe_allow_html=True)


def _build_hit_verdict_row(s: dict) -> str:
    """Build the hit verdict HTML row for a suggestion card."""
    verdict_data = s.get("hit_verdict")
    if not verdict_data:
        return ""
    ep_add = s.get("ep_next_add", 0)
    ep_drop = s.get("ep_next_drop", 0)
    ep_delta = s.get("ep_delta", 0)
    color = verdict_data["verdict_color"]
    verdict = verdict_data["verdict"]
    display = verdict_data["display_str"]

    # Spell the comparison out and colour it by direction. "in" minus "out" as
    # a bare signed number, sitting beside a green FREE badge, was read as the
    # cost of making the transfer rather than as the gap between two players.
    if ep_delta >= 0:
        delta_text = f"<b style=\"color:#4ecca3;\">+{ep_delta:.1f} xPts</b>"
    else:
        delta_text = f"<b style=\"color:#f87171;\">&minus;{abs(ep_delta):.1f} xPts</b>"
    breakdown = (
        f'{s.get("add_player", "in")} {ep_add:.1f} vs '
        f'{s.get("drop_player", "out")} {ep_drop:.1f} &rarr; {delta_text} this GW'
    )

    badge_text = f"{verdict} &nbsp; {display}" if display else verdict
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-top:6px;padding-top:6px;border-top:1px solid #2d2d2d;">'
        f'<span style="color:#9ca3af;font-size:0.78em;">FPL xPts: {breakdown}</span>'
        f'<span style="background:{color};color:#0d1117;padding:2px 10px;border-radius:10px;'
        f'font-size:0.78em;font-weight:bold;">{badge_text}</span>'
        f'</div>'
    )


def _build_trend_ownership_row(s: dict) -> str:
    """Build the price trend + ownership HTML row for a suggestion card."""
    pt = s.get("price_trend")
    badge = s.get("add_ownership_badge", "")
    pct = s.get("add_ownership_pct", 0)

    badges = []
    if pt:
        if pt.get("badge_html"):
            badges.append(pt["badge_html"])
        if pt.get("rush_html"):
            badges.append(pt["rush_html"])
    if pct:
        badges.append(
            f'<span style="color:#9ca3af;font-size:0.78em;">Own: {pct:.1f}%</span>'
        )
    if badge:
        badges.append(badge)

    if not badges:
        return ""
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:4px;">'
        + "".join(badges)
        + "</div>"
    )


def _annotate_suggestions_against_plan(suggestions: List[Dict],
                                        plan: Optional[Dict]) -> None:
    """Mark where the cards and the planner agree, and where they do not.

    The two sections answer different questions and optimise different things:
    the cards rank individual swaps on positional percentiles with a sanity
    veto, the planner maximises expected points over a set of moves. Both are
    worth having -- the cards carry urgency, price trends, blank-gameweek
    reasoning and the veto, none of which the planner models.

    But the same swap appearing twice with a percentile `+0.03` and a points
    `+2.4` is the app visibly disagreeing with itself, which is the failure
    `_blended_proj` was written for: the same player read two different ways a
    few hundred pixels apart. So say which it is.
    """
    if not suggestions:
        return
    if not plan or not plan.get("legs"):
        for s in suggestions:
            s["plan_status"] = None
        return

    by_out = {leg["out_id"]: leg for leg in plan["legs"]}
    for s in suggestions:
        leg = by_out.get(s.get("drop_id"))
        if leg is None:
            s["plan_status"] = None
        elif leg["in_id"] == s.get("add_id"):
            s["plan_status"] = ("IN PLAN", None)
        else:
            s["plan_status"] = ("PLAN DIFFERS", leg["in_player"])


def _render_transfer_suggestions(suggestions: List[Dict], free_transfers: int = 1):
    """Render transfer suggestion cards using styled HTML."""
    if not suggestions:
        st.info("No beneficial transfers found. Your squad looks strong at all positions.")
        return

    st.subheader("Transfer Suggestions")
    # A move can be recommended while projecting fewer points this week, and
    # without this line that reads as the page contradicting itself.
    st.caption(
        "FPL xPts compares the two players over this gameweek alone, using FPL's own "
        "expected points. Suggestions weigh rest-of-season value too, so a move can be "
        "worth making even when it projects fewer points this week."
    )

    for s in suggestions:
        # Urgency badge
        urgency = s.get('urgency', '')
        urgency_html = ""
        if urgency == "URGENT":
            urgency_html = ('<span style="background:#dc3545;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">URGENT</span>')
        elif urgency == "BLANK GW":
            urgency_html = ('<span style="background:#7c3aed;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">BLANK GW</span>')
        elif urgency == "LOW PRIORITY":
            urgency_html = ('<span style="background:#374151;color:#9ca3af;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">LOW PRIORITY</span>')
        elif urgency == "GK CAUTION":
            urgency_html = ('<span style="background:#b45309;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">GK CAUTION</span>')
        elif urgency == "LOW DEPTH":
            urgency_html = ('<span style="background:#ff9800;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">LOW DEPTH</span>')

        # Where this card and the planner above are talking about the same
        # drop, say whether they agree -- a silent contradiction is worse than
        # either answer.
        plan_status = s.get("plan_status")
        if plan_status:
            label, other = plan_status
            if other:
                urgency_html += (
                    '<span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;'
                    'border-radius:12px;font-size:0.8em;font-weight:bold;margin-left:8px;">'
                    f'{label}: {other}</span>')
            else:
                urgency_html += (
                    '<span style="background:#1a472a;color:#4ecca3;padding:3px 10px;'
                    'border-radius:12px;font-size:0.8em;font-weight:bold;margin-left:8px;">'
                    f'{label}</span>')

        card_html = f"""
        <div style="border: 1px solid #444; border-radius: 10px; padding: 16px; margin-bottom: 12px;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div>
                    <span style="background: #0f3460; color: #e0e0e0; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">{s['position']}</span>{urgency_html}
                </div>
                <span style="background: #1a472a; color: #4ecca3; padding: 3px 12px; border-radius: 12px;
                             font-size: 0.85em; font-weight: bold;">+{s['score_diff']:.3f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="flex: 1;">
                    <div style="color: #e74c3c; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">DROP</div>
                    <div style="color: #e0e0e0; font-weight: bold;">{s['drop_full_name']} ({s['drop_team']})</div>
                    <div style="color: #999; font-size: 0.85em;">
                        Price: {s['drop_price']} &bull; Form: {s['drop_form']} (healthy) &bull;
                        Season: {s['drop_season_pts']} &bull; {s['drop_injury']}
                    </div>
                </div>
                <div style="color: #888; font-size: 1.5em; padding: 0 16px;">&rarr;</div>
                <div style="flex: 1; text-align: right;">
                    <div style="color: #4ecca3; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">ADD</div>
                    <div style="color: #e0e0e0; font-weight: bold;">{s['add_full_name']} ({s['add_team']})</div>
                    <div style="color: #999; font-size: 0.85em;">
                        Price: {s['add_price']} &bull; Proj: {s['add_proj_pts']} &bull;
                        Form: {s['add_form']} (healthy) &bull; {s['add_injury']}
                    </div>
                </div>
            </div>
            <div style="color: #aaa; font-size: 0.82em; font-style: italic; border-top: 1px solid #333;
                        padding-top: 6px;">{s['rationale']}</div>
            {_build_hit_verdict_row(s)}
            {_build_trend_ownership_row(s)}
        </div>
        """
        st.markdown(compact_html(card_html), unsafe_allow_html=True)


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_classic_transfers_page():
    """Display the Classic FPL Transfers page."""

    # Seed session_state from file on first load / after hot-reload.
    # Must happen before any read of _PENDING_KEY.
    _init_pending_state()

    col_title, col_refresh = st.columns([5, 1])
    with col_title:
        st.title("Transfer Suggestions")
        st.caption("Find the best transfer targets based on projections, form, fixtures, and price.")

    # Show post-rerun success message (set before st.rerun() call in the form)
    if "_pending_transfer_success" in st.session_state:
        st.success(st.session_state.pop("_pending_transfer_success"))
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Refresh Data", help="Clear cached data and reload from FPL API"):
            # Clear Streamlit in-memory caches
            get_classic_transfers.clear()
            get_classic_team_picks.clear()
            get_classic_team_history.clear()
            fetch_my_team.clear()
            # Also clear SQLite picks cache — permanent cache can store stale/corrupt data
            # (e.g. FH squad data stored under GW32 key during a FH period)
            _team_id_for_clear = config.FPL_CLASSIC_TEAM_ID
            if _team_id_for_clear:
                purge_cache_prefix(f"classic_picks:{_team_id_for_clear}:")
            st.rerun()

    # Check configuration
    team_id = config.FPL_CLASSIC_TEAM_ID
    if not team_id:
        st.warning("No Classic FPL team configured.")
        st.info(
            "Add your team ID to your `.env` file:\n\n"
            "```\nFPL_CLASSIC_TEAM_ID=123456\n```\n\n"
            "You can find your team ID in the URL when viewing your team on the FPL website."
        )
        return

    # Load data
    with st.spinner("Loading data..."):
        bootstrap = get_classic_bootstrap_static()
        entry = get_entry_details(team_id)
        current_gw = get_current_gameweek() or 1
        history = get_classic_team_history(team_id)

    if not bootstrap:
        show_api_error("loading player data for transfer analysis")
        return

    if not entry:
        show_api_error(f"loading team details for team ID {team_id}", hint_key="team_id")
        return

    # Team name (used in squad header below)
    team_name = entry.get("name", "Unknown Team")

    # Skip any GW where Free Hit was active — squad reverts after FH so those
    # picks represent a temporary squad, not the real registered 15.
    chips_list = history.get("chips", []) if history else []
    fh_gws = {c["event"] for c in chips_list if c.get("name") == "freehit"}

    # Fetch confirmed transfers from FPL API.
    # Note: FPL only returns confirmed (post-deadline) transfers here.
    # Pre-deadline transfers won't appear until after the GW kicks off.
    all_transfers = get_classic_transfers(team_id) or []

    # Build element lookup once — used by pending-transfer UI and squad reconstruction.
    elements_by_id = {e["id"]: e for e in bootstrap.get("elements", [])}

    # Sync session-state pending transfers: drop any that FPL has now confirmed.
    local_pending = _sync_pending_local(team_id, all_transfers)

    # Merge confirmed API transfers + still-pending local ones.
    # Use (out, in, event) triplet so a historical transfer with the same players
    # doesn't incorrectly mask a new pending local transfer.
    confirmed_keys = {(t["element_out"], t["element_in"], t.get("event", 0)) for t in all_transfers}
    extra_local = [t for t in local_pending
                   if (t["element_out"], t["element_in"], t.get("event", 0)) not in confirmed_keys]
    effective_transfers = all_transfers + extra_local

    # Resolve the squad through the shared resolver so this page, Fixture
    # Projections and the two optimizers always agree on the same 15. The
    # session-state-merged transfer list is passed in explicitly — this page
    # owns that layer, the resolver only reads the file.
    resolution = resolve_classic_squad(
        team_id, bootstrap, current_gw,
        history=history, extra_transfers=effective_transfers,
    )

    if not resolution.ok:
        show_api_error("loading your current squad")
        return

    picks_data = resolution.as_picks_data()
    picks_source_gw = resolution.source_gw
    applied_transfers = resolution.applied_transfers

    # Surface where the squad came from.
    if extra_local:
        local_names = []
        for t in extra_local:
            out_name = elements_by_id.get(t["element_out"], {}).get("web_name", str(t["element_out"]))
            in_name  = elements_by_id.get(t["element_in"],  {}).get("web_name", str(t["element_in"]))
            local_names.append(f"{out_name} → {in_name}")
        st.info(
            f"⏳ Locally-logged transfer(s) applied: **{', '.join(local_names)}**  "
            f"— these will auto-clear once FPL confirms them after the deadline."
        )
    elif resolution.source == "my_team":
        st.caption(f"✅ {resolution.provenance}")
    elif picks_source_gw and picks_source_gw < current_gw - 1:
        fh_gw = picks_source_gw + 1
        msg = f"Base squad loaded from GW{picks_source_gw} (GW{fh_gw} was a Free Hit)."
        if applied_transfers:
            msg += f" Applied pending transfer(s): {', '.join(applied_transfers)}."
        st.info(msg)
    elif applied_transfers:
        st.info(f"Applied confirmed transfer(s): {', '.join(applied_transfers)}.")
    elif resolution.is_stale:
        st.warning(f"⚠️ {resolution.provenance}")

    picks = picks_data.get("picks", [])
    active_chip = picks_data.get("active_chip")  # e.g., "wildcard", "freehit", "bboost", "3xc"
    entry_history = picks_data.get("entry_history", {})

    bank = entry_history.get("bank", 0)
    squad_value = entry_history.get("value", 0)
    transfers_made = entry_history.get("event_transfers", 0)
    transfer_cost = entry_history.get("event_transfers_cost", 0)

    # Compute chip status and free transfers (pass fh_gws so FH GWs are
    # excluded from the FT-banking calculation)
    chip_status = _parse_chip_status(history, current_gw)
    ft_info = resolve_free_transfers(
        history, entry_history, current_gw, fh_gws=fh_gws,
        team_id=team_id, logged_pending=len(extra_local),
        squad_source=resolution.source,
    )
    free_transfers = ft_info["count"]

    # A count that cannot be right gates every hit verdict on the page, so
    # check it rather than trusting the reconstruction. Logged, not raised:
    # a page that renders a suspect number with a warning beside it is more
    # use than one that refuses to render.
    chip_gws = {c["event"] for c in chips_list
                if c.get("name") in ("wildcard", "freehit")
                and c.get("event") is not None}
    ft_issues = check_free_transfers(
        free_transfers,
        gameweek=current_gw,
        limit=entry_history.get("event_transfers_limit"),
        made=(entry_history.get("event_transfers")
              if entry_history.get("event_transfers_limit") is not None else None),
        status=entry_history.get("event_transfers_status"),
        chip_gws=chip_gws,
        logged=ft_info["logged"],
    )
    if ft_issues and ft_info["source"] != "manual":
        _logger.warning("Free-transfer count looks wrong: %s",
                        format_issues(ft_issues))

    # Status panel — shown before filters so users see FT count immediately
    _render_transfer_status_panel(bank, squad_value, free_transfers, chip_status,
                                  active_chip, ft_info=ft_info, team_id=team_id,
                                  current_gw=current_gw)

    # Controls
    with st.expander("Filters", expanded=True):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            pos_filter = st.multiselect(
                "Position Filter",
                ["GK", "DEF", "MID", "FWD"],
                default=["GK", "DEF", "MID", "FWD"]
            )

        with col_b:
            # Default lookahead shrinks near end of season (no point looking past GW38)
            default_fdr = min(5, max(2, 38 - current_gw))
            fdr_weeks = st.slider("FDR Lookahead (weeks)", 1, 8, default_fdr)

        with col_c:
            max_price = st.slider(
                "Max Price",
                5.0, 15.0, 15.0, 0.5,
                format="£%.1fm"
            )

        st.caption("Scores use positional percentiles (0-1) against the full FPL pool. "
                   "A score of 0.85 = top 15% at this position. Weights auto-adjust by gameweek.")

    # Build DataFrames
    ffp_feed_result = None
    with st.spinner("Analyzing players..."):
        # Build all players DataFrame
        all_players = _build_all_players_df(bootstrap, current_gw, fdr_weeks)

        # Build squad DataFrame
        squad_df = _build_squad_df(picks, bootstrap, entry_history)

        # Warn loudly if a logged pending transfer hasn't taken effect
        if extra_local:
            current_ids = {p["element"] for p in picks}
            for t in extra_local:
                out_el  = elements_by_id.get(t["element_out"], {})
                in_el   = elements_by_id.get(t["element_in"],  {})
                out_name = out_el.get("web_name", str(t["element_out"]))
                in_name  = in_el.get("web_name",  str(t["element_in"]))
                if t["element_out"] in current_ids:
                    # The diagnostic detail (element ids, source gameweek, the
                    # whole squad) goes to the log, not the page -- it is for
                    # whoever debugs this, and it told the user nothing they
                    # could act on.
                    _logger.error(
                        "Pending transfer not applied: element_out=%s element_in=%s "
                        "event=%s picks_source_gw=%s fh_gws=%s squad=%s",
                        t["element_out"], t["element_in"], t["event"],
                        picks_source_gw, fh_gws, sorted(current_ids))
                    st.warning(
                        f"⚠️ **{out_name}** is still in your squad even though you "
                        f"logged **{out_name} → {in_name}**. Remove and re-log the "
                        f"transfer, or use Refresh to reload the squad from FPL."
                    )
                elif t["element_in"] in current_ids:
                    st.success(f"✅ {out_name} → {in_name} applied successfully.")
        squad_df["AvgFDR"] = squad_df["Team_ID"].map(
            _avg_fdr_by_team(current_gw, fdr_weeks))

        # Load projections
        projections_df = None
        try:
            rotowire_url = config.ROTOWIRE_URL
            if rotowire_url:
                projections_df = get_rotowire_player_projections(rotowire_url)
            else:
                st.warning(
                    "⚠️ Rotowire player projections are unavailable — the app could not discover the current "
                    "rankings article URL. Rotowire may have changed their URL format. "
                    "To fix immediately, add `ROTOWIRE_URL=<article URL>` to your `.env` file and restart the app."
                )
        except Exception as e:
            st.warning(f"Could not load projections: {e}")

        # Add projections
        all_players = _add_projections(all_players, projections_df)
        squad_df = _add_projections(squad_df, projections_df)

        # Blank GW detection — needed here to prevent ep_next fallback from
        # showing non-zero projections for teams with no fixture this GW.
        blanking_team_ids = _get_blanking_team_ids(current_gw, bootstrap)

        # FPL's ep_next used to be written straight into Projected_Points -- the
        # Rotowire slot -- when Rotowire had not published. That gave it 60% of
        # the blend while being labelled Rotowire everywhere downstream, and
        # _proj_source (which would have disclosed it) is not written on this
        # path. It is now a declared fallback source inside the engine, so it
        # fills the same gap and says so in Proj_Src. All that remains here is
        # the blank-gameweek rule, which no projection source models.
        for _df in [all_players, squad_df]:
            if blanking_team_ids and "Team_ID" in _df.columns:
                _blank = _df["Team_ID"].isin(blanking_team_ids)
                if "Projected_Points" in _df.columns:
                    _df.loc[_blank, "Projected_Points"] = 0
                if "ep_next" in _df.columns:
                    _df.loc[_blank, "ep_next"] = 0

        # Compute healthy form for squad players (15 players — fast)
        for idx, row in squad_df.iterrows():
            pid = row.get("Player_ID")
            if pd.notna(pid):
                hf = compute_healthy_form(int(pid), element_history_fn=_fetch_element_history)
                squad_df.at[idx, "HealthyForm"] = hf if hf is not None else row.get("form", 0)
            else:
                squad_df.at[idx, "HealthyForm"] = row.get("form", 0)

        # Load FFP multi-GW projections
        try:
            ffp_feed_result = get_ffp_feed()
            ffp_df = ffp_feed_result.df
        except Exception:
            ffp_feed_result, ffp_df = None, None

        # Add multi-GW projections (cap fallback multiplier to remaining GWs)
        _remaining_gws = max(1, 38 - current_gw)
        squad_df = blend_multi_gw_projections(
            squad_df, ffp_df, single_gw_col="Projected_Points", remaining_gws=_remaining_gws
        )
        all_players = blend_multi_gw_projections(
            all_players, ffp_df, single_gw_col="Projected_Points", remaining_gws=_remaining_gws
        )

        # Rotowire Season Projections
        try:
            season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
        except Exception:
            season_rankings_df = None
        squad_df = merge_season_projections(squad_df, season_rankings_df)
        all_players = merge_season_projections(all_players, season_rankings_df)

        # FFP Single-GW Data (Predicted, Start, LongStart)
        squad_df = merge_ffp_single_gw_data(squad_df, ffp_df)
        all_players = merge_ffp_single_gw_data(all_players, ffp_df)

        # The canonical blend. "Proj Pts" on this page used to be raw Rotowire
        # while the Fixture Projections page showed the blended number for the
        # same player, under a near-identical heading.
        squad_df = blend_projections_onto(squad_df, ffp_df, rotowire_col="Projected_Points")
        all_players = blend_projections_onto(all_players, ffp_df, rotowire_col="Projected_Points")

    render_ffp_status(ffp_feed_result, current_gw)

    # Positional depth
    depth_map = {}
    if not squad_df.empty:
        # Map status from bootstrap for depth calculation
        elements_status = {p["id"]: p for p in bootstrap.get("elements", [])}
        for idx, row in squad_df.iterrows():
            pid = row.get("Player_ID")
            el = elements_status.get(pid, {})
            if "status" not in squad_df.columns:
                squad_df["status"] = None
            squad_df.at[idx, "status"] = el.get("status", "a")
        depth_map = compute_positional_depth(squad_df)

    # Compute keep scores (after depth is known)
    if not squad_df.empty:
        squad_df = _compute_keep_score(squad_df, all_players_df=all_players,
                                       current_gw=current_gw, depth_map=depth_map)

    # Chip strategy advisor (needs Keep Scores to be computed first)
    _render_chip_advisor(chip_status, squad_df, current_gw)

    # Get squad player IDs for filtering
    squad_ids = set(squad_df["Player_ID"].tolist())
    teams_map = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    # Position mapping for filter
    pos_map = {"GK": "G", "DEF": "D", "MID": "M", "FWD": "F"}
    filter_positions = [pos_map.get(p, p) for p in pos_filter]

    # Filter available players — must be active in FPL (status a=available, d=doubtful)
    # and have played at some point this season. This excludes players who are
    # injured, suspended, unavailable, or not in squad (status i/s/u/n).
    available = all_players[
        (~all_players["Player_ID"].isin(squad_ids)) &
        (all_players["Position"].isin(filter_positions)) &
        (all_players["now_cost"] <= max_price * 10) &
        (all_players["minutes"] > 0) &
        (all_players["status"].isin(["a", "d"]))
    ].copy()

    # Compute healthy form for top transfer candidates (selective — not all 600+ players)
    # Which 50 players are worth an element-summary fetch each. Ranked on the
    # blend rather than Rotowire alone: when Rotowire has not published, that
    # column is entirely NaN and this fell back to bootstrap order, spending the
    # whole budget of fetches on whoever happened to be first.
    _proj_rank_col = next(
        (c for c in ("Proj", "Projected_Points")
         if c in available.columns and available[c].notna().any()),
        None,
    )
    top_candidates = (available.nlargest(50, _proj_rank_col, keep="all")
                      if _proj_rank_col else available.head(50))
    for idx in top_candidates.index:
        pid = available.at[idx, "Player_ID"]
        if pd.notna(pid):
            hf = compute_healthy_form(int(pid), element_history_fn=_fetch_element_history)
            available.at[idx, "HealthyForm"] = hf if hf is not None else available.at[idx, "form"]
        else:
            available.at[idx, "HealthyForm"] = available.at[idx, "form"]
    # Fill remaining candidates with FPL form
    if "HealthyForm" not in available.columns:
        available["HealthyForm"] = available["form"]
    available["HealthyForm"] = available["HealthyForm"].fillna(available["form"])

    # Compute transfer score
    available = _compute_transfer_score(available, all_players_df=all_players, current_gw=current_gw)
    available = available.sort_values("Transfer Score", ascending=False)

    # ---------------------------
    # TRANSFER PLANNER
    # ---------------------------
    _render_blank_gw_alert(squad_df, blanking_team_ids, current_gw)

    # Above the suggestion cards: this decides how many transfers you are
    # spending, and it is the stronger recommendation -- it is denominated in
    # expected points where the cards below rank on positional percentiles.
    st.subheader("Transfer Planner")
    st.caption(
        "The best squad reachable from yours in a given number of transfers, "
        "maximising **expected points** — so it can propose selling a premium "
        "to fund a better one elsewhere, which ranking by percentile cannot."
    )

    _init_plan_weight_state()
    c_now, c_next, c_hits = st.columns([2, 2, 2])
    with c_now:
        st.slider("This Gameweek %", 0, 100, key=_W_NOW_KEY, step=5,
                  on_change=_sync_plan_weight_from_now)
    with c_next:
        st.slider("Next 3 Gameweeks % (incl. this one)", 0, 100,
                  key=_W_NEXT3_KEY, step=5, on_change=_sync_plan_weight_from_next3)
    with c_hits:
        max_extra_hits = st.slider(
            "Allow extra transfers (−4 each)", 0, 2, 0, key=_EXTRA_HITS_KEY,
            help="A hit is only ever proposed when it wins on points after the "
                 "4 is subtracted.")

    _w_now = st.session_state[_W_NOW_KEY] / 100.0
    _w_next3 = st.session_state[_W_NEXT3_KEY] / 100.0
    _horizon = plan_horizon(_w_now, _w_next3)
    st.caption(
        f"Split: **{st.session_state[_W_NOW_KEY]}% this gameweek / "
        f"{st.session_state[_W_NEXT3_KEY]}% next 3 gameweeks** — the two always "
        f"total 100%, and \"next 3\" includes this one. The split also sets how "
        f"long a −4 hit has to pay itself back: at this setting, "
        f"**{_horizon:.1f} gameweeks**."
    )

    transfer_plan = None
    if free_transfers + max_extra_hits >= 1:
        try:
            with st.spinner("Solving…"):
                _plan_pool = build_plan_pool(all_players, squad_df, _w_now, _w_next3)
                transfer_plan = build_transfer_plan(
                    _plan_pool, squad_df, bank,
                    free_transfers=free_transfers,
                    max_extra_hits=max_extra_hits,
                    w_now=_w_now, w_next3=_w_next3,
                )
        except Exception as exc:
            # A solver failure must not take the page down. Fall back to the
            # brute-force pair, captioned so a degraded answer is never mistaken
            # for the real one.
            _logger.exception("Transfer planner failed: %s", exc)
            transfer_plan = None
            if free_transfers >= 2:
                multi_plan = _build_multi_transfer_plan(
                    squad_df, available, bank, depth_map=depth_map)
                if multi_plan:
                    st.caption(
                        "⚠️ The optimizer was unavailable, so this is the older "
                        "percentile-based pair rather than the points-optimal plan."
                    )
                    _render_multi_transfer_plan(multi_plan, free_transfers=free_transfers)
    else:
        st.info(
            "You have no free transfers this gameweek. Raise the slider above "
            "to see what a −4 hit would buy."
        )

    _render_transfer_plan(transfer_plan, free_transfers, _w_now, _w_next3)

    st.markdown("---")

    # ---------------------------
    # TRANSFER SUGGESTION CARDS
    # ---------------------------
    st.caption(
        "Individual swaps, ranked by positional percentile and filtered by the "
        "sanity veto. They answer \"what single move is worth making\"; the "
        "planner above answers \"what set of moves\"."
    )
    suggestions = _build_transfer_suggestions(
        squad_df, available, bank, top_n=3, depth_map=depth_map,
        free_transfers=free_transfers, blanking_team_ids=blanking_team_ids,
    )
    _annotate_suggestions_against_plan(suggestions, transfer_plan)
    _render_transfer_suggestions(suggestions, free_transfers=free_transfers)

    st.markdown("---")

    # Pending transfer logger — lets user manually log pre-deadline transfers.
    # Sits after the suggestions and before the squad: you read the advice, log
    # the move you made on it, and the squad below is what the log produces.
    _render_log_transfer_ui(team_id, current_gw, picks, elements_by_id, extra_local,
                            bank=bank)

    # ---------------------------
    # SQUAD ANALYSIS SECTION (with depth card)
    # ---------------------------
    st.header(f"Your Squad — {team_name}")

    # Stat cards for bank, value, transfers
    def _stat_card(label: str, value: str, accent: str = "#00ff87") -> str:
        return (
            f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
            f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
            f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
            f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
            f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
            f'</div>'
        )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(_stat_card("Bank", _format_money(bank)), unsafe_allow_html=True)
    with col2:
        st.markdown(_stat_card("Squad Value", _format_money(squad_value)), unsafe_allow_html=True)
    with col3:
        st.markdown(_stat_card("Transfers Made", str(transfers_made)), unsafe_allow_html=True)
    with col4:
        cost_text = f"-{transfer_cost} pts" if transfer_cost else "0 pts"
        cost_color = "#f87171" if transfer_cost else "#00ff87"
        st.markdown(_stat_card("Transfer Cost", cost_text, accent=cost_color), unsafe_allow_html=True)

    st.markdown("")  # spacing between stat cards and depth card

    if depth_map:
        _render_depth_card(depth_map)

    # Split into starting XI and bench
    starting_xi = squad_df[squad_df["squad_position"] <= 11].copy()
    bench = squad_df[squad_df["squad_position"] > 11].copy()

    # Show squad with keep scores (sort by Keep Score for holistic view)
    squad_display = squad_df.sort_values("Keep Score", ascending=True).copy()

    # Add fixtures for each player
    fixture_html_list = []
    for _, row in squad_display.iterrows():
        fixtures = _get_team_fixtures(row["Team_ID"], fdr_weeks, current_gw)
        fixture_html_list.append(_format_fixtures_html(fixtures, teams_map, 5))
    squad_display["Fixtures"] = fixture_html_list

    # Add availability
    squad_display["Status"] = squad_display.apply(
        lambda r: _get_availability_indicator(r["chance_of_playing_next_round"], r["news"]),
        axis=1
    )

    # Add positional rank
    squad_display["Pos_Rank"] = positional_rank(
        squad_display, all_players, "total_points", ref_value_col="total_points"
    )

    # Format display columns — use HealthyForm if available
    form_display_col = "HealthyForm" if "HealthyForm" in squad_display.columns else "form"
    display_cols = ["Player", "Team", "Position", "Pos_Rank", "now_cost", form_display_col, "total_points",
                    "Proj", "Proj_Src", "AvgFDR", "1GW", "ROS", "Keep Score", "Status"]
    display_cols = [c for c in display_cols if c in squad_display.columns]
    squad_show = squad_display[display_cols].copy()
    squad_show["Price"] = squad_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
    squad_show["AvgFDR"] = squad_show["AvgFDR"].round(2)
    for sc in ["1GW", "ROS", "Keep Score"]:
        if sc in squad_show.columns:
            squad_show[sc] = squad_show[sc].round(3)
    # "Proj Pts" is the blend: expected points, start likelihood priced in.
    # "Src" says which sources produced it, so a number resting on FFP alone --
    # or on FPL's own expected points because Rotowire has not published -- is
    # visible rather than indistinguishable from a fully-corroborated one.
    if "Proj" in squad_show.columns:
        squad_show["Proj"] = squad_show["Proj"].fillna("-")

    # Rename for display
    squad_show = squad_show.rename(columns={
        form_display_col: "Form",
        "total_points": "Season Pts",
        "Proj": "Proj Pts",
        "Proj_Src": "Src",
        "Pos_Rank": "Pos Rank",
        "AvgFDR": "Avg FDR",
    })

    squad_final_cols = ["Player", "Team", "Position", "Pos Rank", "Price", "Form", "Season Pts",
                        "Proj Pts", "Src", "Avg FDR", "1GW", "ROS", "Keep Score", "Status"]
    squad_final_cols = [c for c in squad_final_cols if c in squad_show.columns]
    render_styled_table(
        squad_show[squad_final_cols],
        col_formats={"Proj Pts": "{:.1f}", "Form": "{:.1f}", "Avg FDR": "{:.2f}",
                     "1GW": "{:.3f}", "ROS": "{:.3f}", "Keep Score": "{:.3f}"},
        positive_color_cols=["1GW", "ROS", "Keep Score"],
    )

    # ---------------------------
    # TRANSFER TARGETS SECTION
    # ---------------------------
    st.header("Transfer Targets")

    # Show top targets
    st.subheader("Top Transfer Targets (All Positions)")

    # Prepare display DataFrame
    top_targets = available.head(20).copy()

    # Add fixtures
    target_fixture_list = []
    for _, row in top_targets.iterrows():
        fixtures = _get_team_fixtures(row["Team_ID"], fdr_weeks, current_gw)
        target_fixture_list.append(_format_fixtures_html(fixtures, teams_map, 5))
    top_targets["Fixtures"] = target_fixture_list

    # Add availability
    top_targets["Status"] = top_targets.apply(
        lambda r: _get_availability_indicator(r["chance_of_playing_next_round"], r["news"]),
        axis=1
    )

    # Add price change indicator
    top_targets["Price_Change"] = top_targets["cost_change_event"].apply(_format_price_change)

    # Add transfer rush indicator
    top_targets["Rush"] = top_targets["transfers_in_event"].apply(
        lambda x: "🔥" if int(x or 0) > 200_000 else ""
    )

    # Format for display — use HealthyForm if available
    target_form_col = "HealthyForm" if "HealthyForm" in top_targets.columns else "form"
    target_display_cols = [
        "Player", "Team", "Position", "now_cost", "Price_Change", "Rush", target_form_col,
        "total_points", "Proj", "Proj_Src", "selected_by_percent",
        "AvgFDR", "1GW", "ROS", "Transfer Score", "Status"
    ]
    target_display_cols = [c for c in target_display_cols if c in top_targets.columns]
    targets_show = top_targets[target_display_cols].copy()

    targets_show["Price"] = targets_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
    targets_show["AvgFDR"] = targets_show["AvgFDR"].round(2)
    for sc in ["1GW", "ROS", "Transfer Score"]:
        if sc in targets_show.columns:
            targets_show[sc] = targets_show[sc].round(3)
    if "Proj" in targets_show.columns:
        targets_show["Proj"] = targets_show["Proj"].fillna("-")
    targets_show["Ownership"] = targets_show["selected_by_percent"].apply(lambda x: f"{x:.1f}%")

    # Rename for display
    targets_show = targets_show.rename(columns={
        target_form_col: "Form",
        "total_points": "Season Pts",
        "Proj": "Proj Pts",
        "Proj_Src": "Src",
        "AvgFDR": "Avg FDR",
        "Price_Change": "Δ",
    })

    display_cols_final = ["Player", "Team", "Position", "Price", "Δ", "Rush", "Form",
                          "Season Pts", "Proj Pts", "Src", "Ownership", "Avg FDR",
                          "1GW", "ROS", "Transfer Score", "Status"]
    display_cols_final = [c for c in display_cols_final if c in targets_show.columns]
    render_styled_table(
        targets_show[display_cols_final],
        col_formats={"Proj Pts": "{:.1f}", "Form": "{:.1f}", "Avg FDR": "{:.2f}",
                     "1GW": "{:.3f}", "ROS": "{:.3f}", "Transfer Score": "{:.3f}"},
        positive_color_cols=["1GW", "ROS", "Transfer Score"],
        max_height=500,
    )

    st.markdown("---")

    # ---------------------------
    # POSITION-SPECIFIC TARGETS
    # ---------------------------
    st.subheader("Position-Specific Targets")

    tabs = st.tabs(["Goalkeepers", "Defenders", "Midfielders", "Forwards"])

    position_codes = {"Goalkeepers": "G", "Defenders": "D", "Midfielders": "M", "Forwards": "F"}

    for tab, (pos_name, pos_code) in zip(tabs, position_codes.items()):
        with tab:
            pos_targets = available[available["Position"] == pos_code].head(10).copy()

            if pos_targets.empty:
                st.info(f"No {pos_name.lower()} match your filter criteria.")
                continue

            # Add fixtures
            pos_fixture_list = []
            for _, row in pos_targets.iterrows():
                fixtures = _get_team_fixtures(row["Team_ID"], fdr_weeks, current_gw)
                pos_fixture_list.append(_format_fixtures_html(fixtures, teams_map, 5))
            pos_targets["Fixtures"] = pos_fixture_list

            # Format for display
            pos_cols = ["Player", "Team", "now_cost", "form", "total_points",
                        "Proj", "Proj_Src", "selected_by_percent", "AvgFDR",
                        "1GW", "ROS", "Transfer Score"]
            pos_cols = [c for c in pos_cols if c in pos_targets.columns]
            pos_show = pos_targets[pos_cols].copy()

            pos_show["Price"] = pos_show["now_cost"].apply(lambda x: f"£{x/10:.1f}m")
            pos_show["AvgFDR"] = pos_show["AvgFDR"].round(2)
            for sc in ["1GW", "ROS", "Transfer Score"]:
                if sc in pos_show.columns:
                    pos_show[sc] = pos_show[sc].round(3)
            if "Proj" in pos_show.columns:
                pos_show["Proj"] = pos_show["Proj"].fillna("-")
            pos_show["Own%"] = pos_show["selected_by_percent"].apply(lambda x: f"{x:.1f}%")

            pos_display_cols = ["Player", "Team", "Price", "form", "total_points",
                          "Proj", "Proj_Src", "Own%", "AvgFDR", "1GW", "ROS", "Transfer Score"]
            pos_display_cols = [c for c in pos_display_cols if c in pos_show.columns]
            pos_display = pos_show[pos_display_cols].copy()
            pos_display = pos_display.rename(columns={
                "form": "Form", "total_points": "Season Pts",
                "Proj": "Proj Pts", "Proj_Src": "Src", "AvgFDR": "Avg FDR",
            })
            render_styled_table(
                pos_display,
                col_formats={"Proj Pts": "{:.1f}", "Form": "{:.1f}", "Avg FDR": "{:.2f}",
                             "1GW": "{:.3f}", "ROS": "{:.3f}", "Transfer Score": "{:.3f}"},
                positive_color_cols=["1GW", "ROS", "Transfer Score"],
            )

    st.markdown("---")

    # ---------------------------
    # TRANSFER COMPARISON TOOL
    # ---------------------------
    st.header("Transfer Comparison")
    st.caption("Compare a player from your squad with potential replacements.")

    col_out, col_in = st.columns(2)

    with col_out:
        st.subheader("Transfer Out")
        squad_options = squad_df["Player"].tolist()
        selected_out = st.selectbox("Select player to transfer out", squad_options)

        if selected_out:
            out_player = squad_df[squad_df["Player"] == selected_out].iloc[0]
            st.markdown(f"**{out_player['Player']}** ({out_player['Team']})")
            st.caption(f"Position: {out_player['Position']} | Price: £{out_player['now_cost']/10:.1f}m")
            st.caption(f"Form: {out_player['form']:.1f} | Season Pts: {out_player['total_points']}")

            fixtures = _get_team_fixtures(out_player["Team_ID"], fdr_weeks, current_gw)
            st.markdown("**Upcoming fixtures:**")
            st.markdown(_format_fixtures_html(fixtures, teams_map, 6), unsafe_allow_html=True)

    with col_in:
        st.subheader("Transfer In")

        if selected_out:
            out_player = squad_df[squad_df["Player"] == selected_out].iloc[0]
            out_pos = out_player["Position"]
            selling_price = out_player["selling_price"]

            # Calculate budget
            budget = bank + selling_price

            # Filter replacements
            replacements = available[
                (available["Position"] == out_pos) &
                (available["now_cost"] <= budget)
            ].head(20)

            if replacements.empty:
                st.warning("No affordable replacements found.")
            else:
                in_options = replacements["Player"].tolist()
                selected_in = st.selectbox("Select replacement", in_options)

                if selected_in:
                    in_player = replacements[replacements["Player"] == selected_in].iloc[0]
                    st.markdown(f"**{in_player['Player']}** ({in_player['Team']})")
                    st.caption(f"Position: {in_player['Position']} | Price: £{in_player['now_cost']/10:.1f}m")
                    st.caption(f"Form: {in_player['form']:.1f} | Season Pts: {in_player['total_points']}")
                    st.caption(f"Ownership: {in_player['selected_by_percent']:.1f}%")

                    fixtures = _get_team_fixtures(in_player["Team_ID"], fdr_weeks, current_gw)
                    st.markdown("**Upcoming fixtures:**")
                    st.markdown(_format_fixtures_html(fixtures, teams_map, 6), unsafe_allow_html=True)

                    # Show comparison summary
                    st.markdown("---")
                    st.markdown("**Transfer Summary:**")
                    cost_diff = in_player["now_cost"] - selling_price
                    if cost_diff > 0:
                        st.caption(f"Cost: +£{cost_diff/10:.1f}m (Budget: £{budget/10:.1f}m)")
                    else:
                        st.caption(f"Cost: -£{abs(cost_diff)/10:.1f}m (saves money)")

                    form_diff = in_player["form"] - out_player["form"]
                    st.caption(f"Form change: {'+' if form_diff >= 0 else ''}{form_diff:.1f}")

                    proj_in = _blended_proj(in_player)
                    proj_out = _blended_proj(out_player)
                    if pd.notna(proj_in) and pd.notna(proj_out):
                        proj_diff = proj_in - proj_out
                        st.caption(f"Projected points change: {'+' if proj_diff >= 0 else ''}{proj_diff:.1f}")

    st.markdown("---")

    # ---------------------------
    # RECENT TRANSFERS SECTION
    # ---------------------------
    st.header("Your Recent Transfers")

    transfers = get_classic_transfers(team_id)
    if transfers:
        elements = {p["id"]: p for p in bootstrap.get("elements", [])}

        recent = transfers[:10]  # Last 10 transfers

        transfer_rows = []
        for t in recent:
            in_id = t.get("element_in")
            out_id = t.get("element_out")
            in_player = elements.get(in_id, {})
            out_player = elements.get(out_id, {})

            transfer_rows.append({
                "GW": t.get("event", "?"),
                "Out": out_player.get("web_name", "Unknown"),
                "In": in_player.get("web_name", "Unknown"),
                "In Cost": f"£{t.get('element_in_cost', 0)/10:.1f}m",
                "Out Cost": f"£{t.get('element_out_cost', 0)/10:.1f}m",
            })

        transfers_df = pd.DataFrame(transfer_rows)
        render_styled_table(transfers_df, text_align={"GW": "center"})
    else:
        st.info("No transfers found for this season.")
