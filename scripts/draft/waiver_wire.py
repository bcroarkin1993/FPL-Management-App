# waiver_wire.py
import os
import re
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

import config
from openai import OpenAI
from scripts.common.injury_helpers import (
    ABSENCE_FROM_NEWS,
    OUT_OF_SQUAD_STATUSES,
    absence_estimate,
    estimate_games_to_miss,
)
from scripts.common.projection_engine import DEPUTY_MAX_START_RATE
from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.draft.waiver_wire")

from scripts.common.utils import (
    get_current_gameweek,
    merge_fpl_players_and_projections,
    get_league_player_ownership,
    get_league_entries,
    get_fpl_player_mapping,
    get_rotowire_player_projections,
    pull_fpl_player_stats,
    normalize_fpl_players_to_rotowire_schema,
    normalize_rotowire_players,
    _bootstrap_teams_df,
    _norm_text,
    TEAM_FULL_TO_SHORT
)
from scripts.common.fpl_draft_api import (
    get_league_element_states,
    get_draft_transaction_window,
    get_league_waiver_order,
    ELEMENT_STATE_LOCKED,
    ELEMENT_STATE_AVAILABLE,
    ELEMENT_STATE_OWNED,
    TRANSACTION_MODE_FREE_AGENCY,
)
from scripts.common.claim_value import (
    HORIZON_GWS,
    MIN_FLOOR_POOL,
    LANE_HOLD,
    LANE_RENT,
    recommended_lane,
    release_cost as _release_cost,
    rental_gain,
    waiver_floors,
)
from scripts.common.waiver_priority import (
    BAND_CONTESTED,
    BAND_LIKELY,
    BAND_LONG_SHOT,
    claim_outlook,
    expected_gone_before,
    expected_taken_at_position,
    managers_ahead,
    my_waiver_pick,
    rank_claim_plan,
)
from scripts.common.display_names import (
    attach_display_names,
    display_of,
    display_series,
)
from scripts.common.player_matching import canonical_normalize
from scripts.common.text_helpers import (compact_html, format_deadline_stamp,
                                         format_time_until)
from scripts.common.styled_tables import render_styled_table
from scripts.common.transfer_sanity import (horizon_column, horizon_points,
                                             sanity_check_suggestion)
from scripts.common.analytics import (
    gameweek_deadlines,
    compute_player_scores,
    compute_dynamic_alpha,
    compute_healthy_form,
    compute_positional_depth,
    compute_transfer_urgency,
    blend_multi_gw_projections,
    positional_rank,
    dampen_form_by_starts,
    merge_season_projections,
    merge_ffp_single_gw_data,
    enrich_reference_with_projections,
    numeric_col,
    attach_reference_ids,
)
from scripts.common.scraping import get_ffp_feed, get_rotowire_season_rankings, render_ffp_status

# ---------------------------
# SUGGESTION VIEW
# ---------------------------


# ---------------------------------------------------------------
# The claim engine lives in `waiver_core`; this module renders it.
# Re-exported here because `trade_analyzer` and `classic/transfers`
# import several of these by name, and the tests patch others.
# ---------------------------------------------------------------
from scripts.draft.waiver_core import (
    WaiverDataUnavailable,
    POSITION_CODE_TO_LABEL,
    POSITION_LABEL_TO_CODE,
    RENTAL_MIN_GAIN,
    _HORIZON_GWS,
    _IDENTITY_COLS,
    _POS_NUM_TO_LETTER,
    _POS_WORD_TO_LETTER,
    _add_fdr_and_form,
    _add_injury_data,
    _align_roster_player_names_to_projections,
    _attach_display_names,
    _attach_minutes_dependency,
    _availability_multiplier,
    _available_from_projections,
    _avg_fdr_by_short_name,
    _avg_fdr_for_team,
    _build_floor_context,
    _build_rationale,
    _build_suggestion,
    _build_waiver_context,
    _claim_points_rate,
    _codes_in_state,
    _compute_keep_score,
    _compute_transfer_suggestions,
    _compute_waiver_score,
    _display_names,
    _display_of,
    _element_history_df,
    _enforce_rw_schema_fpl,
    _enforce_rw_schema_proj,
    _ensure_team_col,
    _estimate_games_to_miss,
    _fetch_all_transactions,
    _flatten_owned_names,
    _format_availability,
    _load_bootstrap,
    _load_future_fixtures,
    _locked_rotowire_names,
    _min_max_norm,
    _names_for_codes,
    _next_waiver_deadline,
    _prepare_proj_for_merge,
    _roster_injury_factor,
    _sanity_check_suggestion,
    _series_to_short_team,
    _team_id_by_short,
    _teams_map_short,
    _to_position_letter,
    _waiver_claim_stats,
)



def _filter_by_position(df, labels):
    """Rows at the selected positions. Empty selection means empty board.

    Frames on this page carry either the short codes analytics groups on
    (``G/D/M/F``) or the long ones the Draft API publishes (``GK/DEF/MID/FWD``),
    depending on which merge produced them, so both are accepted. Matching only
    one silently drops every row -- the same mismatch that once had every team
    in the Power Rankings scoring exactly 50.
    """
    if df is None or df.empty or "Position" not in df.columns:
        return df
    wanted = {POSITION_LABEL_TO_CODE[l] for l in (labels or []) if l in POSITION_LABEL_TO_CODE}
    if not wanted:
        return df.iloc[0:0]
    if len(wanted) == len(POSITION_LABEL_TO_CODE):
        return df
    pos = df["Position"].astype(str).str.upper().str.strip()
    pos = pos.map(lambda p: POSITION_LABEL_TO_CODE.get(p, p))
    return df[pos.isin(wanted)]

# "Best per position" is the compact view the page shipped with: the single
# strongest move at each position. The rest scan the whole roster against the
# whole available pool.
#
# **It is no longer the default, because it hides moves worth making.** It tries
# only the 2 weakest roster players against the 5 best available at each
# position and then emits *one* move per position, so a second good move at a
# position someone has already claimed simply disappears -- and disappears
# silently, since a player who was never evaluated produces no debug row and so
# reads exactly like one that was evaluated and rejected. Measured on the live
# GW9 pool it showed 2 moves where the full scan found 5, dropping ones worth
# +0.92, +0.79 and +0.17 points per gameweek.
#
# The default is "Top 10": the full roster against the full pool, ordered by
# expected points gained, capped so the list stays readable. The cap is a
# display limit, not a search limit -- `total_found` reports what the scan
# actually found, so a truncated list says so.
SUGGESTION_VIEW_BEST_PER_POS = "Best per position"
SUGGESTION_VIEW_DEFAULT = "Top 10"
SUGGESTION_VIEW_OPTIONS = [
    SUGGESTION_VIEW_BEST_PER_POS,
    "Top 5",
    SUGGESTION_VIEW_DEFAULT,
    "All improvements",
]
SUGGESTION_VIEW_LIMITS = {"Top 5": 5, "Top 10": 10}

# How long you intend to hold the player you claim. **This is the control the
# page never had, and its absence was the bug.** `Transfer Score` is
# `alpha * 1GW + (1 - alpha) * ROS`, which is a fixed, hidden choice of horizon
# applied to every move -- and therefore optimal for neither of the two
# transactions a manager actually makes.
#
# Measured live at GW10, ranking the same available midfielders on points over
# an explicit horizon: 1GW gives Madueke, Estevao, Guimaraes; 6GW gives Ayari,
# Bobb, Madueke. `Transfer Score` behaved like the 6GW end, which is why the
# highest-projected available player in the game was suggested nowhere.
HOLD_RENT = "Just this week"
HOLD_BOTH = "Both"
HOLD_KEEP = "Rest of season"
HOLD_OPTIONS = [HOLD_BOTH, HOLD_RENT, HOLD_KEEP]
HOLD_DEFAULT = HOLD_BOTH


# ---------------------------
# FPL API READS
# ---------------------------








# ---------------------------
# HARDENING HELPERS
# ---------------------------






# ---------------------------
# FEATURE ENGINEERING
# ---------------------------






# ---------------------------
# OWNERSHIP / AVAILABLES
# ---------------------------













# ---------------------------
# DRAFT CAPITAL (OPTIONAL)
# ---------------------------


# ---------------------------
# SCORING
# ---------------------------




# ---------------------------
# INJURY HELPERS
# ---------------------------









# ---------------------------
# INJURY DATA PIPELINE
# ---------------------------



# ---------------------------
# TRANSFER SUGGESTIONS
# ---------------------------















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


#: How a claim's reachability reads on a card. The wording is deliberately about
#: the *queue*, not about the player: "Long shot" is a statement that managers
#: ahead of you want this position, never that the player is bad.
_OUTLOOK_STYLE = {
    BAND_LIKELY: ("Likely yours", "#1a472a", "#4ecca3"),
    BAND_CONTESTED: ("Contested", "#4a3a12", "#f0c040"),
    BAND_LONG_SHOT: ("Long shot", "#4a1f1f", "#ff8080"),
}


def _outlook_pill(band: str) -> str:
    """Badge for a claim's reachability, or "" when the outlook is unknown.

    An unknown outlook renders nothing rather than a neutral badge: the waiver
    order or the power rankings being unavailable is not a finding about this
    player, and a grey "Unknown" pill on every card is noise.
    """
    style = _OUTLOOK_STYLE.get(band)
    if not style:
        return ""
    label, bg, fg = style
    return (f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:12px;'
            f'font-size:0.8em;font-weight:bold;margin-left:8px;">{label}</span>')


def _render_waiver_queue(order: List[Dict], my_entry_id, stats: Optional[Dict] = None,
                         free_agency: bool = False):
    """The league's waiver order, with this manager's slot marked.

    Rendered because the number on the banner card ("pick 7 of 10") is only half
    the story — which *teams* pick ahead of you is what makes a contested target
    predictable.

    ``waiver_pick`` is always the order for the **next** round, so in free agency
    it is a forward-looking fact rather than a live one. It stays on the page —
    planning the next round is exactly what you do during free agency — but the
    title and caption say which round it describes, since the same number read as
    current would imply a queue that is not running.
    """
    if not order:
        return

    pick = my_waiver_pick(order, my_entry_id)
    n_ahead = len(managers_ahead(order, my_entry_id))
    _when = "next waiver round" if free_agency else "waiver order"
    title = (f"{_when.capitalize()} — you pick {pick} of {len(order)}"
             if pick else f"{_when.capitalize()} ({len(order)} teams)")

    with st.expander(title, expanded=False):
        rows = []
        for row in order:
            mine = row.get("entry_id") == my_entry_id
            colour = "#4ecca3" if mine else "#e0e0e0"
            marker = " &larr; you" if mine else ""
            rows.append(
                f'<div style="padding:4px 10px;color:{colour};">'
                f'<b>{row["pick"]}.</b> {row["team_name"]}{marker}</div>'
            )
        st.markdown(compact_html(
            '<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            'border:1px solid #333;border-radius:10px;padding:10px;color:#e0e0e0;">'
            + "".join(rows) + '</div>'
        ), unsafe_allow_html=True)

        st.caption(
            ("Free agency is open, so this applies to the **next** waiver round "
             "rather than to anything you do today. " if free_agency else "")
            + "FPL re-derives this order from the standings each gameweek — the "
            "bottom team claims first — and a **successful** claim sends that "
            "manager to the back of the queue for the rest of the round. A "
            "**failed** claim costs nothing, so rank your claims by what you "
            "actually want rather than by what you think you can get."
        )

        _rate = (stats or {}).get("participation_rate_recent")
        if _rate is None:
            _rate = (stats or {}).get("participation_rate")
        if stats and _rate is not None and n_ahead:
            gone = expected_gone_before(pick, _rate)
            # The recent window, not the whole season. Measured live on this
            # league the two were 63% and 71%: the rounds just after a draft are
            # churn, and quoting them reports the league as more active than it is.
            _recent = stats.get("recent_events") or []
            _scope = (f"the last {len(_recent)} waiver round"
                      f"{'s' if len(_recent) != 1 else ''}"
                      if _recent else
                      f"{stats['events_observed']} waiver round"
                      f"{'s' if stats['events_observed'] != 1 else ''}")
            line = (
                f"Across {_scope} in this league, "
                f"{_rate * 100:.0f}% of managers claimed in a "
                f"given round — so expect roughly **{gone:.0f}** of the {n_ahead} "
                f"managers ahead of you to take a player before your first claim "
                f"is looked at."
            )
            _rates = stats.get("position_claim_rates") or {}
            if _rates and n_ahead:
                _per_pos = ", ".join(
                    f"{POSITION_CODE_TO_LABEL.get(p, p)} {expected_taken_at_position(p, n_ahead, _rates):.1f}"
                    for p in ("G", "D", "M", "F") if _rates.get(p) is not None
                )
                if _per_pos:
                    line += (f" By position, the players they typically take are: "
                             f"{_per_pos}.")
                    # The badges are suppressed in free agency, so pointing at
                    # them there would describe something not on the page.
                    line += (" That is what the reachability badge on each "
                             "suggestion is calibrated on, rather than how badly "
                             "a rival is ranked somewhere."
                             if not free_agency else
                             " That is what the reachability badges will be "
                             "calibrated on once the next waiver round opens.")
            if stats.get("contested_share") is not None:
                line += (f" {stats['contested_share'] * 100:.0f}% of successful "
                         f"claims were for a player someone else also wanted")
                peak = stats.get("max_contention") or 0
                if peak > 1:
                    line += f" (peak: {peak} managers on one player)"
                line += "."
            st.caption(line)


def _render_claim_plan(suggestions: List[Dict], free_agency: bool = False,
                       lane: str = LANE_HOLD, lane_label: str = ""):
    """The priority-ordered list of claims to submit to FPL.

    FPL asks you to rank your waiver requests, and the instinct is to lead with
    something safe. That is backwards: processing walks down your list until one
    claim succeeds, and only a *success* costs your slot — so a long shot at the
    top is free. ``rank_claim_plan()`` carries that rule and its evidence.

    **In free agency none of that applies**, so the framing changes rather than
    the list. There is no queue to be processed in, no priority to spend and
    nothing to be beaten to: each add simply happens. The ordering still earns
    its place, because every add costs a drop and the pool shrinks as the rest of
    the league acts, but presenting it as a ranked *request* list would describe
    a mechanic that is not running.
    """
    # **One plan per lane.** A rental and a rest-of-season claim are not
    # comparable in a single priority order -- they are ranked in different
    # currencies and answer different questions -- so merging them would stamp a
    # submission order on a list that has no single meaning. The caller renders
    # one expander per lane and names which it is.
    plan = rank_claim_plan([x for x in suggestions
                            if x.get('lane', LANE_HOLD) == lane])
    if len(plan) < 2:
        return

    _suffix = f" — {lane_label}" if lane_label else ""
    title = (f"Suggested pickup order ({len(plan)} moves){_suffix}" if free_agency
             else f"Suggested claim order ({len(plan)} requests){_suffix}")
    with st.expander(title, expanded=False):
        st.caption(
            "Free agency is open, so each of these processes the moment you make "
            "it — there is no waiver priority to spend and nobody to beat. Work "
            "down the list: every add costs you a drop, and the best targets go "
            "first as the rest of the league picks over the same pool."
            if free_agency else
            "Submit these in this order. A failed claim does not cost your waiver "
            "priority, so the long shots belong at the top — only a claim that "
            "**succeeds** moves you to the back of the queue."
        )
        rows = []
        for row in plan:
            band = row.get("outlook_band", "")
            label = _OUTLOOK_STYLE.get(band, ("", "", "#888"))[0]
            tag = f' <span style="color:#888;">({label})</span>' if label else ""
            rows.append(
                f'<div style="padding:5px 10px;color:#e0e0e0;">'
                f'<b>{row["claim_priority"]}.</b> Drop {row["drop_player"]} '
                f'&rarr; claim <b>{row["add_player"]}</b> ({row["add_team"]}){tag}</div>'
            )
        st.markdown(compact_html(
            '<div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            'border:1px solid #333;border-radius:10px;padding:10px;">'
            + "".join(rows) + '</div>'
        ), unsafe_allow_html=True)






def _render_transfer_suggestions(
    suggestions: List[Dict],
    current_gw: int = 0,
    total_found: Optional[int] = None,
    free_agency: bool = False,
    waiver_gw: Optional[int] = None,
):
    """Render suggestion cards using styled HTML.

    The subheader and view controls are rendered by the caller (they have to
    exist before the suggestions are computed), so this draws cards only.

    Args:
        total_found: how many moves the search turned up before the view limit
            was applied — surfaced so a truncated list says so.
        free_agency: whether adds process immediately. These cards carry no
            reachability badge in that window (see `_build_waiver_context`), so
            the caption has to say why they are all simply available — and name
            the one exception, which is that a player dropped this week is locked
            until the next waiver round and cannot be picked up as a free agent.
        waiver_gw: the gameweek whose waiver round a locked player enters.
    """
    if not suggestions:
        st.info("No beneficial transfers found. Your roster looks strong at all positions.")
        return

    shown = len(suggestions)
    if total_found is not None and total_found > shown:
        note = (f"Showing the top {shown} of {total_found} improving moves — "
                f"switch **Show** to *All improvements* to see the rest.")
    else:
        note = f"{shown} improving move{'s' if shown != 1 else ''} found."
    # Every add and every drop appears at most once, so this is a set of moves
    # you can actually make rather than a list of alternatives competing for the
    # same target. Worth saying: the cards used to repeat one standout player
    # against each weak roster spot, which read as a plan but was one move.
    if len(suggestions) > 1:
        note += (" Each player appears once, so these can all be made — the best "
                 "target goes to the player you most want to replace, and the "
                 "next drop takes the next-best target. They are ordered by the "
                 "**expected points** the swap adds per gameweek, which is the "
                 "only way a goalkeeper move and a forward move can be compared "
                 "— so a big upgrade over a weak player outranks a smaller one "
                 "over a player who is already decent, even where the incoming "
                 "player is the better footballer. **Every swap is "
                 "position-locked**, which is FPL's rule rather than ours: a "
                 "claim replaces a squad player with an unselected one at the "
                 "*same* position, so a defender can never be swapped for a "
                 "midfielder however much that would improve the squad.")
    if free_agency:
        # No reachability badges in this window, so say why: there is no queue to
        # lose a player in. The exception is worth naming, because it is the one
        # way an "available" player still cannot be had today.
        _wv = f"GW{waiver_gw} waiver round" if waiver_gw else "next waiver round"
        note += (" Free agency is open, so these are direct swaps that process "
                 "immediately — no waiver priority and nobody to beat. The one "
                 f"exception: a player dropped this week is locked until the {_wv} "
                 "and cannot be picked up as a free agent.")
    st.caption(note)

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}
    mgw_window = min(3, max(1, 38 - current_gw)) if current_gw > 0 else 3
    mgw_label = "Next GW" if mgw_window == 1 else f"Next {mgw_window} GW"

    # **Two lanes, drawn as two sections.** They answer different questions and
    # their gain figures are different quantities -- the hold lane's is points
    # per gameweek, the rental lane's is total points over the window -- so a
    # single merged list would be the cross-position percentile mistake one level
    # up: two numbers that look alike and are not comparable.
    _hold = [x for x in suggestions if x.get('lane', LANE_HOLD) == LANE_HOLD]
    _rent = [x for x in suggestions if x.get('lane') == LANE_RENT]

    for _lane_items, _lane in ((_hold, LANE_HOLD), (_rent, LANE_RENT)):
        if not _lane_items:
            continue
        if _hold and _rent:
            if _lane == LANE_HOLD:
                st.markdown("##### 📌 Hold — rest-of-season upgrades")
                st.caption(
                    "Ranked on durable value, so a player with a thin season "
                    "behind him ranks low here even when he is this week's best "
                    "play. That is what the rental lane below is for.")
            else:
                st.markdown("##### ⚡ Rent — just this week")
                st.caption(
                    "Ranked on **this gameweek's** projection, and charged only "
                    "for the player you release: the one you claim goes back on "
                    "the wire next week, so his own future never enters. "
                    "**Release cost** is what dropping your player actually costs "
                    "once the wire's replacement is counted — at or near zero "
                    "means someone of his standard is claimable at that position, "
                    "so the rental is close to free. The two lanes can want the "
                    "same roster slot, so treat them as alternatives rather than "
                    "a combined plan.")
        _render_suggestion_cards(_lane_items, pos_labels, mgw_label)


def _render_suggestion_cards(suggestions, pos_labels, mgw_label):
    """The cards for one lane. Split out of `_render_transfer_suggestions` so the
    two lanes can carry their own heading and caption without duplicating 120
    lines of card HTML."""
    for s in suggestions:
        pos_label = pos_labels.get(s['drop_position'], s['drop_position'])
        score = s['transaction_score']

        # The badge shows what the list is *ordered* by. It used to show the
        # percentile gap, which is not a comparable quantity between positions --
        # 0.10 was worth 0.091 expected points at GK and 0.405 at FWD on the live
        # GW6 pool -- so two cards showing "+0.10" were not offering the same
        # thing. The percentile gap is still what qualifies a swap, and stays on
        # the card as a hover.
        points_gain = float(s.get('points_gain', 0) or 0)
        _is_rent = s.get('lane') == LANE_RENT
        if points_gain > 0 and _is_rent:
            # Total points over the window, not a per-gameweek rate: the add is
            # held for one gameweek and the drop's cost runs for the rest of it.
            badge = f"+{points_gain:.2f} pts"
            _rc = s.get('release_cost')
            badge_title = ("This gameweek's gain, less what releasing your player "
                           "costs over the window"
                           + (f" ({_rc:.2f} pts/GW)" if _rc is not None else ""))
        elif points_gain > 0:
            badge = f"+{points_gain:.2f} pts/GW"
            badge_title = f"Score gap +{score:.3f} (positional percentile)"
        else:
            badge = f"+{score:.3f}"
            badge_title = "Positional percentile gap — no projection to price the move"

        # Urgency badge
        urgency = s.get('urgency', '')
        urgency_html = ""
        if urgency == "URGENT":
            urgency_html = ('<span style="background:#dc3545;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">URGENT</span>')
        elif urgency == "LOW DEPTH":
            urgency_html = ('<span style="background:#ff9800;color:#fff;padding:3px 10px;border-radius:12px;'
                            'font-size:0.8em;font-weight:bold;margin-left:8px;">LOW DEPTH</span>')

        # Can you actually win this claim at your waiver priority? Renders to ""
        # for an unknown outlook, which is why the card goes through compact_html:
        # an empty fragment on its own line ends the HTML block and dumps the
        # closing </div> on screen as literal text.
        outlook_html = _outlook_pill(s.get('outlook_band', ''))
        outlook_note = (
            f'<div style="color:#888;font-size:0.78em;margin-top:4px;">'
            f'{s["outlook_reason"]}</div>'
            if s.get('outlook_reason') else ""
        )

        # Build GW projection display strings with data-quality indicator
        drop_proj_str = (
            f"{s['drop_proj_pts']:.1f} pts" if s.get('drop_has_data')
            else '<span style="color:#e67e22;">No proj</span>'
        )
        add_proj_str = (
            f"{s['add_proj_pts']:.1f} pts" if s.get('add_has_data')
            else '<span style="color:#e67e22;">No proj</span>'
        )
        # Why a rental is cheap, on the card rather than only inside the ranking.
        # Release cost is measured over a 3-gameweek window -- the only horizon
        # the app has -- which *understates* the cost of releasing a genuinely
        # good player, so the number is shown rather than buried.
        #
        # Whose absence this add's minutes rest on, and how well we know it.
        #
        # This and the two fragments below render to "" most of the time, which
        # is exactly the optional-fragment case `compact_html` exists for: a
        # blank line ends the HTML block and Markdown then prints the closing
        # </div> on screen as literal text.
        blocker_html = ""
        if s.get('add_blocker'):
            _w = s.get('add_blocker_weeks')
            _src = s.get('add_blocker_source')
            if _src == ABSENCE_FROM_NEWS and _w is not None:
                _conf = "back in ~%d GW" % int(round(_w))
                _tone = "#8ab4f8"
                blocker_html = (
                    f'<div style="color:{_tone};font-size:0.78em;margin-top:4px;">'
                    f'Starts while {s["add_blocker"]} is out &mdash; {_conf}, '
                    f'so his minutes are priced to end then.</div>')
            else:
                # The estimate came from a status code or a chance bucket, not
                # from the text, so **nothing is capped** and the note must not
                # assert a date. It is also the looser identification: a player
                # with few starts who is suddenly in the XI may be deputising, or
                # may simply be back from his own injury (live, Bruno Guimarães
                # was flagged behind Bukayo Saka on exactly that signature). So
                # this is a prompt to check, not a claim.
                blocker_html = (
                    f'<div style="color:#e0a33e;font-size:0.78em;margin-top:4px;">'
                    f'May be starting while {s["add_blocker"]} is out '
                    f'&mdash; no stated return date, so how long this lasts is '
                    f'unknown and nothing here assumes it.</div>')

        thin_html = ""
        if s.get('thin_wire'):
            _pl = pos_labels.get(s['drop_position'], s['drop_position'])
            thin_html = (
                f'<div style="color:#e0a33e;font-size:0.78em;margin-top:4px;">'
                f'&#9888; Thin wire at {_pl} &mdash; too few available to price a '
                f'replacement, so this is hard to undo.</div>')

        release_html = ""
        if _is_rent and s.get('release_cost') is not None:
            _rc = float(s['release_cost'])
            _fl = s.get('position_floor')
            _verdict = ("the wire can replace him" if _rc <= 0.05
                        else f"costs {_rc:.2f} pts/GW to lose")
            _fl_txt = (f" &bull; {pos_labels.get(s['drop_position'], s['drop_position'])} "
                       f"wire floor {_fl:.2f} pts/GW" if _fl is not None else "")
            release_html = (
                f'<div style="color:#8ab4f8;font-size:0.78em;margin-top:4px;">'
                f'Release cost {_rc:.2f} pts/GW &mdash; {_verdict}{_fl_txt}</div>')

        drop_mgw_str = f" &bull; {mgw_label}: {s['drop_3gw_proj']:.1f}" if s.get('drop_3gw_proj', 0) > 0 else ""
        add_mgw_str = f" &bull; {mgw_label}: {s['add_3gw_proj']:.1f}" if s.get('add_3gw_proj', 0) > 0 else ""

        card_html = f"""
        <div style="border: 1px solid #444; border-radius: 10px; padding: 16px; margin-bottom: 12px;
                    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <div>
                    <span style="background: #0f3460; color: #e0e0e0; padding: 3px 12px; border-radius: 12px;
                                 font-size: 0.85em; font-weight: bold;">{pos_label}</span>{urgency_html}{outlook_html}
                </div>
                <span title="{badge_title}" style="background: #1a472a; color: #4ecca3; padding: 3px 12px;
                             border-radius: 12px; font-size: 0.85em; font-weight: bold;">{badge}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="flex: 1;">
                    <div style="color: #e74c3c; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">DROP</div>
                    <div style="color: #e0e0e0; font-weight: bold;">{s['drop_player']} ({s['drop_team']})</div>
                    <div style="color: #999; font-size: 0.85em;">
                        Keep: {s['drop_value']} &bull; GW: {drop_proj_str}{drop_mgw_str} &bull;
                        Season: {s['drop_season_pts']} pts &bull; {s['drop_injury']}
                    </div>
                </div>
                <div style="color: #888; font-size: 1.5em; padding: 0 16px;">&rarr;</div>
                <div style="flex: 1; text-align: right;">
                    <div style="color: #4ecca3; font-weight: bold; font-size: 0.8em; margin-bottom: 2px;">ADD</div>
                    <div style="color: #e0e0e0; font-weight: bold;">{s['add_player']} ({s['add_team']})</div>
                    <div style="color: #999; font-size: 0.85em;">
                        Score: {s['add_value']} &bull; GW: {add_proj_str}{add_mgw_str} &bull;
                        Season: {s.get('add_season_pts', 0)} pts &bull; Form: {s['add_form']} &bull; {s['add_injury']}
                    </div>
                </div>
            </div>
            <div style="color: #aaa; font-size: 0.82em; font-style: italic; border-top: 1px solid #333;
                        padding-top: 6px;">{s['rationale']}</div>{blocker_html}{thin_html}{release_html}{outlook_note}
        </div>
        """
        st.markdown(compact_html(card_html), unsafe_allow_html=True)


# ---------------------------
# SCORING
# ---------------------------



def _render_transfer_debug(debug_rows: List[Dict]):
    """Render a collapsed Transfer Analysis expander showing why each position did/didn't fire."""
    if not debug_rows:
        return

    pos_labels = {'G': 'GK', 'D': 'DEF', 'M': 'MID', 'F': 'FWD'}

    with st.expander("Transfer Analysis", expanded=False):
        st.caption(
            "Shows how each position was evaluated. "
            "Gap = best available adj. score − worst roster adj. score. "
            "Adj. score = raw score × injury multiplier. "
            "🚫 = passed score threshold but vetoed by sanity check (ADD worse on raw metrics)."
        )

        _locked_excluded = max(
            (int(r.get('locked_excluded', 0) or 0) for r in debug_rows), default=0
        )
        if _locked_excluded:
            st.caption(
                f"🔒 {_locked_excluded} locked player(s) were excluded from consideration — "
                "they were recently dropped or added and only become claimable at the "
                "next waiver deadline."
            )

        # Collected since this gate was written and never shown, which is half
        # of why "why isn't X suggested?" was unanswerable from this panel: a
        # player removed here produces no pair row at all, so he reads exactly
        # like one that was evaluated and rejected.
        _unavailable = max(
            (int(r.get('unavailable_excluded', 0) or 0) for r in debug_rows), default=0
        )
        if _unavailable:
            st.caption(
                f"🚑 {_unavailable} player(s) were excluded as unable to play — "
                "injured, suspended or unavailable, so they cannot be scored "
                "against anyone this gameweek."
            )

        st.caption(
            "A player missing from the lists below was **never evaluated**, "
            "which is not the same as rejected. The *Best per position* view "
            "tries only your 2 weakest players at each position against the 5 "
            "best available and shows one move per position; every other view "
            "searches the whole roster against the whole pool."
        )

        for row in debug_rows:
            pos = row['pos']
            label = pos_labels.get(pos, pos)
            if row.get('skipped'):
                st.markdown(f"**{label}** — skipped ({row.get('reason', '')})")
                continue

            pairs = row.get('pairs', [])
            if not pairs:
                st.markdown(f"**{label}** — no comparisons evaluated")
                continue

            lines = []
            for p in pairs:
                sanity = p.get('sanity', 'n/a')
                gap_passed = p['gap'] > p['threshold']
                if p['passed']:
                    icon = "✅"
                elif gap_passed and sanity not in ('n/a', 'ok', 'no data', 'injury override'):
                    icon = "🚫"  # threshold passed but sanity vetoed
                else:
                    icon = "❌"
                chance = p.get('add_chance')
                chance_str = f" ({int(chance)}% chance)" if chance is not None and not pd.isna(chance) else ""
                sanity_str = f" | Vetoed: {sanity}" if icon == "🚫" else ""
                lines.append(
                    f"{icon} **Drop** {p['drop']} (Keep {p['drop_keep']}, adj {p['drop_adj']}) → "
                    f"**Add** {p['add']}{chance_str} (Transfer {p['add_transfer']}, adj {p['add_adj']}) "
                    f"| Gap {p['gap']:+.3f} vs threshold {p['threshold']:.2f}{sanity_str}"
                )
            st.markdown(f"**{label}**\n" + "\n\n".join(lines))


# ---------------------------
# AZURE OPENAI (OPTIONAL)
# ---------------------------


# ---------------------------
# TRANSFER ACTIVITY
# ---------------------------





def _build_transfer_activity_summary(
    transactions_df: pd.DataFrame,
    team_names: Dict[int, str],
    player_map: Dict[int, Dict]
) -> pd.DataFrame:
    """
    Build a summary of transfer activity by team.

    Returns DataFrame with columns:
        Team, Free Transfers, Accepted Waivers, Failed Waivers, Total
    """
    if transactions_df.empty:
        return pd.DataFrame(columns=['Team', 'Free Transfers', 'Accepted Waivers', 'Failed Waivers', 'Total'])

    # Initialize counters for all teams
    summary = {tid: {
        'Team': tname,
        'Free Transfers': 0,
        'Accepted Waivers': 0,
        'Failed Waivers': 0,
    } for tid, tname in team_names.items()}

    for _, tx in transactions_df.iterrows():
        team_id = tx.get('entry')
        kind = tx.get('kind')
        result = tx.get('result')

        if team_id not in summary:
            continue

        if kind == 'f' and result == 'a':
            summary[team_id]['Free Transfers'] += 1
        elif kind == 'w':
            if result == 'a':
                summary[team_id]['Accepted Waivers'] += 1
            else:
                summary[team_id]['Failed Waivers'] += 1

    # Convert to DataFrame
    rows = list(summary.values())
    df = pd.DataFrame(rows)
    df['Total'] = df['Free Transfers'] + df['Accepted Waivers']

    # Sort by total transfers descending
    df = df.sort_values('Total', ascending=False).reset_index(drop=True)

    return df


def _build_transfer_history_table(
    transactions_df: pd.DataFrame,
    team_names: Dict[int, str],
    player_map: Dict[int, Dict]
) -> pd.DataFrame:
    """
    Build a detailed history of all transfers.

    Returns DataFrame with columns:
        GW, Team, Type, Result, Player In, Player Out, Date
    """
    if transactions_df.empty:
        return pd.DataFrame(columns=['GW', 'Team', 'Type', 'Result', 'Player In', 'Player Out', 'Date'])

    rows = []
    for _, tx in transactions_df.iterrows():
        team_id = tx.get('entry')
        team_name = team_names.get(team_id, f"Team {team_id}")

        kind = tx.get('kind')
        kind_label = 'Waiver' if kind == 'w' else 'Free Transfer' if kind == 'f' else kind

        result = tx.get('result')
        result_map = {'a': 'Accepted', 'di': 'Denied', 'do': 'Dropped'}
        result_label = result_map.get(result, result)

        element_in = tx.get('element_in')
        element_out = tx.get('element_out')

        player_in_info = player_map.get(element_in, {})
        player_out_info = player_map.get(element_out, {})

        player_in = player_in_info.get('Player', f"Unknown ({element_in})")
        player_in_team = player_in_info.get('Team', '')
        player_out = player_out_info.get('Player', f"Unknown ({element_out})")
        player_out_team = player_out_info.get('Team', '')

        # Format player names with their team
        player_in_display = f"{player_in} ({player_in_team})" if player_in_team else player_in
        player_out_display = f"{player_out} ({player_out_team})" if player_out_team else player_out

        # Parse date
        added = tx.get('added', '')
        if added:
            try:
                dt = datetime.fromisoformat(added.replace('Z', '+00:00'))
                date_str = dt.strftime('%Y-%m-%d')
            except Exception:
                date_str = added[:10] if len(added) >= 10 else added
        else:
            date_str = ''

        rows.append({
            'GW': tx.get('event', ''),
            'Team': team_name,
            'Type': kind_label,
            'Result': result_label,
            'Player In': player_in_display,
            'Player Out': player_out_display,
            'Date': date_str,
        })

    df = pd.DataFrame(rows)

    # Sort by GW descending, then date descending
    df = df.sort_values(['GW', 'Date'], ascending=[False, False]).reset_index(drop=True)

    return df


def _render_transfer_activity_chart(summary_df: pd.DataFrame):
    """Render a stacked bar chart of transfer activity by team."""
    import plotly.graph_objects as go

    if summary_df.empty:
        st.info("No transfer activity data available.")
        return

    fig = go.Figure()

    # Add Free Transfers bar
    fig.add_trace(go.Bar(
        name='Free Transfers',
        x=summary_df['Team'],
        y=summary_df['Free Transfers'],
        marker_color='#4ecca3',
        text=summary_df['Free Transfers'],
        textposition='inside',
    ))

    # Add Accepted Waivers bar
    fig.add_trace(go.Bar(
        name='Accepted Waivers',
        x=summary_df['Team'],
        y=summary_df['Accepted Waivers'],
        marker_color='#0f3460',
        text=summary_df['Accepted Waivers'],
        textposition='inside',
    ))

    fig.update_layout(
        barmode='stack',
        title='Transfer Activity by Team',
        xaxis_title='Team',
        yaxis_title='Number of Transfers',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=400,
        margin=dict(t=80, b=80),
    )

    # Rotate x-axis labels if many teams
    if len(summary_df) > 6:
        fig.update_xaxes(tickangle=45)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# TRANSACTION WINDOW BANNER
# ---------------------------



def _render_transaction_window(window: Dict[str, Any], locked_count: int,
                               waiver_order: Optional[List[Dict]] = None,
                               my_entry_id=None):
    """Render the current waiver / free-agency state as a compact card row.

    Free agency and waivers are different windows with different rules: during free
    agency an available player can be taken immediately, whereas locked players are
    only claimable once the next waiver round processes. Which window is open
    decides whether a suggestion is actionable today, so it belongs on the page.
    """
    if not window or not window.get("mode"):
        return

    mode = window.get("mode")
    next_gw = window.get("next_event") or window.get("current_event")

    free_agency = mode == TRANSACTION_MODE_FREE_AGENCY
    if free_agency:
        window_icon, window_label = "🟢", "Free Agency open"
        # The exception matters: an "available" player who was just dropped is
        # locked, and that is the only way a free-agency add can still fail.
        window_note = "Adds process immediately — except players just dropped"
    else:
        window_icon, window_label = "⏳", "Waivers pending"
        window_note = "Requests process at the waiver deadline"

    deadline = _next_waiver_deadline(next_gw)
    if deadline is not None:
        # **The date, not just the weekday.** This read "Fri 6:00 AM ET", which a
        # manager takes to mean *this* Friday -- and during an international break
        # it does not. Measured 2026-09-26 the GW6 waiver deadline was Friday 9
        # October, thirteen days out, and GW6 and GW7 rendered identically.
        deadline_value = format_deadline_stamp(deadline)

        _notes = [f"GW{next_gw} waiver deadline" if next_gw else "Next waiver deadline"]
        _until = format_time_until(deadline)
        if _until:
            _notes.append(_until)
        deadline_note = " · ".join(_notes)
    else:
        deadline_value = "—"
        deadline_note = "Deadline unavailable"

    locked_note = (
        f"Enter the GW{next_gw} waiver round" if next_gw else "Claimable at the next waiver"
    )

    # Where you sit in the queue decides whether a suggested claim is a plan or a
    # wish. FPL publishes it on every league entry; nothing here read it until now.
    pick = my_waiver_pick(waiver_order or [], my_entry_id)
    if pick is not None:
        n_ahead = len(managers_ahead(waiver_order, my_entry_id))
        pick_value = f"{pick} of {len(waiver_order)}"
        pick_note = (
            "You claim first" if n_ahead == 0
            else f"{n_ahead} manager{'s' if n_ahead != 1 else ''} claim before you"
        )
        # `waiver_pick` is always the *next* round's order, so during free agency
        # it is a plan rather than a live constraint. Unlabelled it reads as a
        # queue you are sitting in today, which is the thing that is not running.
        if free_agency:
            pick_note = f"Next round · {pick_note}"
    else:
        pick_value = "—"
        pick_note = "Waiver order unavailable"

    def _card(icon: str, value: str, label: str, note: str, color: str) -> str:
        return (
            '<div style="flex:1;text-align:center;padding:14px 12px;color:#e0e0e0;'
            'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
            'border:1px solid #333;border-radius:10px;margin:0 6px;">'
            f'<div style="font-size:1.3em;margin-bottom:4px;color:#e0e0e0;">{icon}</div>'
            f'<div style="color:{color};font-size:1.05em;font-weight:bold;">{value}</div>'
            f'<div style="color:#e0e0e0;font-size:0.85em;margin-top:4px;">{label}</div>'
            f'<div style="color:#888;font-size:0.75em;margin-top:2px;">{note}</div>'
            '</div>'
        )

    st.markdown(
        '<div style="display:flex;margin-bottom:1rem;">'
        + _card(window_icon, window_label, "Transaction window", window_note, "#4ecca3")
        + _card("⏰", deadline_value, "Next deadline", deadline_note, "#3498db")
        + _card("🔒", str(locked_count), "Locked players", locked_note, "#ff9800")
        + _card("🎟️", pick_value, "Your waiver pick", pick_note, "#9b7fd4")
        + '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------
# PAGE
# ---------------------------

def show_waiver_wire_page():
    st.header("🔁 Waiver Wire Assistant")
    st.caption("Blends weekly projections, recent form, fixture difficulty, and injury status to suggest transfers.")

    # Placeholder for the transaction banner and suggestion cards (filled later)
    window_container = st.container()
    suggestion_container = st.container()

    # --- Team picker (moved up so suggestions can use it) ---
    # Load ownership first for team picker
    try:
        ownership = get_league_player_ownership(config.FPL_DRAFT_LEAGUE_ID)
    except Exception as e:
        st.error(f"Unable to load league ownership: {e}")
        ownership = {}

    # Per-player transaction state (owned / available / locked). Both readers fail
    # open: an empty result means the page behaves as it did before states existed.
    try:
        element_states = get_league_element_states(config.FPL_DRAFT_LEAGUE_ID)
    except Exception:
        _logger.warning("Unable to load Draft element states", exc_info=True)
        element_states = {}
    try:
        transaction_window = get_draft_transaction_window(config.FPL_DRAFT_LEAGUE_ID)
    except Exception:
        _logger.warning("Unable to load Draft transaction window", exc_info=True)
        transaction_window = {}

    try:
        waiver_order = get_league_waiver_order(config.FPL_DRAFT_LEAGUE_ID)
    except Exception:
        _logger.warning("Unable to load Draft waiver order", exc_info=True)
        waiver_order = []

    locked_count = sum(
        1 for s in element_states.values() if s.get("status") == ELEMENT_STATE_LOCKED
    )

    my_entry_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)

    # Which window is open decides whether reachability is a question at all: in
    # free agency an available player is simply added, so there is no queue to be
    # beaten in. An unreadable mode is *not* treated as free agency — see
    # `_build_waiver_context`.
    _mode = transaction_window.get("mode")
    _free_agency = _mode == TRANSACTION_MODE_FREE_AGENCY
    _waiver_gw = (transaction_window.get("next_event")
                  or transaction_window.get("current_event"))

    with window_container:
        _render_transaction_window(transaction_window, locked_count,
                                   waiver_order, my_entry_id)
        _render_waiver_queue(waiver_order, my_entry_id, _waiver_claim_stats(
            config.FPL_DRAFT_LEAGUE_ID, len(waiver_order)
        ), free_agency=_free_agency)

    # Suggestion view controls. They render inside the suggestion container at the
    # top of the page, but have to be *created* here — before the search runs —
    # because they are what the search is parameterised on.
    with suggestion_container:
        st.subheader("Transfer Suggestions")
        _ctl_view, _ctl_hold, _ctl_pos = st.columns([1, 1, 2])
        with _ctl_hold:
            sugg_hold = st.selectbox(
                "Hold for",
                options=HOLD_OPTIONS,
                index=HOLD_OPTIONS.index(HOLD_DEFAULT),
                help="A waiver claim is two different decisions and they rank "
                     "players differently. 'Rest of season' is the durable "
                     "upgrade. 'Just this week' is a rental: it ranks on this "
                     "gameweek's projection and charges you only for the player "
                     "you release, since the one you claim goes back on the wire "
                     "next week. A rental is cheap when the player you drop is "
                     "one the wire can replace.",
            )
        with _ctl_view:
            sugg_view = st.selectbox(
                "Show",
                options=SUGGESTION_VIEW_OPTIONS,
                index=SUGGESTION_VIEW_OPTIONS.index(SUGGESTION_VIEW_DEFAULT),
                help="These views search your whole roster against the entire "
                     "available pool, ordered by expected points gained. No "
                     "player is added or dropped twice, so the list is a set of "
                     "moves you can all make. 'Best per position' is the "
                     "narrow view: it tries only your 2 weakest players at each "
                     "position against the 5 best available and shows one move "
                     "per position, so it does hide moves worth making.",
            )
        with _ctl_pos:
            sugg_pos_labels = st.multiselect(
                "Suggest for positions",
                options=list(POSITION_LABEL_TO_CODE.keys()),
                default=list(POSITION_LABEL_TO_CODE.keys()),
                help="Limit suggestions to the positions you actually want to change.",
            )
    sugg_positions = [
        POSITION_LABEL_TO_CODE[lbl] for lbl in (sugg_pos_labels or [])
        if lbl in POSITION_LABEL_TO_CODE
    ]
    sugg_lanes = {
        HOLD_RENT: (LANE_RENT,),
        HOLD_KEEP: (LANE_HOLD,),
    }.get(sugg_hold, (LANE_HOLD, LANE_RENT))

    team_options = []
    for tid, blob in ownership.items():
        tname = blob.get("team_name", f"Team {tid}")
        team_options.append((int(tid), f"{tname} ({tid})"))
    team_options = sorted(team_options, key=lambda x: x[1].lower())

    default_tid = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    default_idx = 0
    if default_tid is not None:
        for i, (tid, label) in enumerate(team_options):
            if str(tid) == str(default_tid):
                default_idx = i
                break

    if team_options:
        choice_label = st.selectbox(
            "Your Team",
            options=[label for _, label in team_options],
            index=default_idx
        )
        my_team_id = next(tid for tid, label in team_options if label == choice_label)
        my_team = ownership.get(my_team_id, {})
    else:
        st.info("No teams found in league ownership data.")
        my_team = {}
        my_team_id = None

    # Controls
    with st.expander("Filters", expanded=False):
        colA, colB, colC = st.columns(3)
        # Distinct from the "Suggest for positions" control by the suggestion cards:
        # this one filters the available-players table further down the page.
        pos_filter = colA.multiselect("Positions (available players table)",
                                      ["G", "D", "M", "F"], default=["G", "D", "M", "F"])
        lookahead = int(colB.number_input("Upcoming GWs to average FDR", min_value=1, max_value=8, value=config.UPCOMING_WEEKS_DEFAULT))
        form_weeks = int(colC.number_input("Form lookback GWs", min_value=1, max_value=5, value=config.FORM_LOOKBACK_WEEKS))

        hide_locked = st.checkbox(
            "Hide locked players",
            value=False,
            help="Locked players were recently dropped or added and cannot be picked up "
                 "until the next waiver deadline. They are never suggested as transfers, "
                 "but are shown by default so you can plan your waiver claims.",
        )

        st.caption("Scores use positional percentiles (0-1) against the full FPL pool. "
                   "A score of 0.85 = top 15% at this position. Weights auto-adjust by gameweek.")

    # Quick visibility
    owned_sample = _flatten_owned_names(ownership)
    st.caption(f"Ownership snapshot: {len(owned_sample)} owned player names found.")
    if len(owned_sample) == 0:
        st.info(
            "No owned players detected. If the season hasn't started or the API returns no element owners, this is expected.")

    # Projections & FPL stats
    rotowire_url = config.ROTOWIRE_URL
    if not rotowire_url:
        st.warning(
            "⚠️ Rotowire player projections are unavailable — the app could not discover the current "
            "rankings article URL. Rotowire may have changed their URL format. "
            "To fix immediately, add `ROTOWIRE_URL=<article URL>` to your `.env` file and restart the app."
        )
    projections_raw = get_rotowire_player_projections(rotowire_url)
    fpl_stats_raw = pull_fpl_player_stats()

    # Normalize both to a unified schema (and enforce minimal columns)
    teams_df = _bootstrap_teams_df()
    try:
        fpl_stats_norm = normalize_fpl_players_to_rotowire_schema(fpl_stats_raw, teams_df=teams_df)
    except Exception as e:
        _logger.warning("FPL stats normalization failed, using raw data: %s", e)
        fpl_stats_norm = fpl_stats_raw
    try:
        projections_norm = normalize_rotowire_players(projections_raw)
    except Exception as e:
        _logger.warning("Projections normalization failed, using raw data: %s", e)
        projections_norm = projections_raw

    fpl_stats = _enforce_rw_schema_fpl(fpl_stats_norm, teams_df)
    proj = _enforce_rw_schema_proj(projections_norm, teams_df)

    # Guard-rails
    req = {"Player", "Team", "Position"}
    miss_stats = req - set(fpl_stats.columns)
    if miss_stats:
        st.error(f"Failed to compute available players: fpl_player_statistics_df missing {sorted(miss_stats)} after normalization.")
        st.stop()

    miss_proj = req - set(proj.columns)
    if miss_proj:
        st.error(f"Projections table missing {sorted(miss_proj)} after normalization.")
        st.stop()

    if "Season_Points" not in fpl_stats.columns:
        fpl_stats["Season_Points"] = 0

    # Current GW
    try:
        current_gw = int(get_current_gameweek() or 1)
    except Exception:
        current_gw = 1

    # AVAILABLE PLAYERS — compute on ALL positions (no filter yet) for suggestions
    #
    # `engine_notes` is how `waiver_core` tells the page something the reader
    # should see. It cannot say so itself: a module that renders can also halt,
    # and `_available_from_projections` used to end this page from inside a
    # function whose job is to filter a DataFrame.
    engine_notes = []
    try:
        avail_all = _available_from_projections(proj, fpl_stats, ownership,
                                                element_states, notes=engine_notes)
    except WaiverDataUnavailable as unavailable:
        # The engine's terminal status. Rendered exactly as it was when the
        # engine rendered it itself -- an honest message, then stop; *not* the
        # "failed to compute" debug dump below, which is for a real fault.
        st.info(unavailable.message)
        st.stop()
    except Exception as e:
        st.error(f"Failed to compute available players: {e}")
        try:
            st.caption("**Projections columns:** " + ", ".join(list(proj.columns)))
            st.dataframe(proj.head(5), use_container_width=True)
        except Exception:
            pass
        try:
            owned_names = _flatten_owned_names(ownership)
            st.caption(f"**Sample owned (FPL) names (first 10 of {len(owned_names)}):** {owned_names[:10]}")
            st.caption("**FPL stats columns:** " + ", ".join(list(fpl_stats.columns)))
        except Exception:
            pass
        st.stop()

    for _severity, _message in engine_notes:
        (st.info if _severity == "info" else st.warning)(_message)

    # Add FDR, Form, Injury data, and Season_Points to available players
    avail_all = _add_fdr_and_form(avail_all, fpl_stats, current_gw, lookahead, form_weeks=form_weeks)
    avail_all = _add_injury_data(avail_all, fpl_stats)

    # Add Season_Points and starts to available players via Player_ID
    if 'Player_ID' in avail_all.columns and 'Season_Points' not in avail_all.columns:
        sp_cols = ['Player_ID', 'Season_Points']
        if 'starts' in fpl_stats.columns:
            sp_cols.append('starts')
        sp_data = fpl_stats[sp_cols].drop_duplicates(subset=['Player_ID'])
        avail_all = avail_all.merge(sp_data, on='Player_ID', how='left')
    elif 'Player_ID' in avail_all.columns and 'starts' not in avail_all.columns and 'starts' in fpl_stats.columns:
        starts_data = fpl_stats[['Player_ID', 'starts']].drop_duplicates(subset=['Player_ID'])
        avail_all = avail_all.merge(starts_data, on='Player_ID', how='left')
    avail_all['Season_Points'] = numeric_col(avail_all, 'Season_Points', 0)
    avail_all['starts'] = numeric_col(avail_all, 'starts', 0)

    # Whose absence is each available player starting because of. Needs the
    # **whole** pool, because the man he is deputising for is injured and so is
    # absent from Rotowire's XI -- and therefore from `avail_all` -- by
    # construction.
    #
    # **After the `starts` merge above, not before it.** The depth rule reads
    # season starts to tell a deputy from a regular, and this ran four lines
    # earlier where the column does not exist yet: it silently treated every
    # available player as having zero starts, and took the page down on the
    # unsafe accessor into the bargain.
    avail_all = _attach_minutes_dependency(avail_all, fpl_stats, current_gw)

    # Resolve the name to render once, here, so the cards and the tables below all
    # show the same thing. Player_ID is set by _add_fdr_and_form() above.
    avail_all = _attach_display_names(avail_all)

    # --- Build MY ROSTER ---
    my_players = []
    for pos, names in my_team.get("players", {}).items():
        for nm in names:
            my_players.append({"Player": nm, "Position": pos})
    my_roster = pd.DataFrame(my_players)

    if not my_roster.empty:
        # Attach Team/Player_ID/Season_Points/starts from FPL master using normalized names
        my_roster["__norm_name"] = my_roster["Player"].apply(canonical_normalize)
        roster_merge_cols = ["Player", "Team", "Position", "Player_ID", "Season_Points"]
        if "starts" in fpl_stats.columns:
            roster_merge_cols.append("starts")
        fpl_stats_for_roster = fpl_stats[roster_merge_cols].copy()
        fpl_stats_for_roster["__norm_name"] = fpl_stats_for_roster["Player"].apply(canonical_normalize)

        merge_right_cols = ["__norm_name", "Team", "Position", "Player_ID", "Season_Points"]
        if "starts" in fpl_stats_for_roster.columns:
            merge_right_cols.append("starts")
        my_roster = my_roster.merge(
            fpl_stats_for_roster[merge_right_cols],
            on=["__norm_name", "Position"], how="left"
        )
        my_roster.drop(columns=["__norm_name"], inplace=True, errors="ignore")

        # Ensure Team present and align names
        my_roster = _ensure_team_col(my_roster, fpl_stats)
        my_roster = _align_roster_player_names_to_projections(
            my_roster, proj, notes=engine_notes)
        my_roster["Team"] = my_roster["Team"].replace(TEAM_FULL_TO_SHORT)

        # After name alignment the Player column now holds Rotowire canonical names.
        # Recover Team from proj for any player whose Team is still NaN/missing —
        # this happens when the FPL short web-name ("Eze") didn't match the full FPL
        # stats name ("Eberechi Eze") on the earlier normalized-name merge, leaving
        # Team blank even though the name was correctly aligned to Rotowire.
        _nan_team = my_roster["Team"].isna() | my_roster["Team"].isin(["nan", "None", ""])
        if _nan_team.any():
            _proj_team_lookup = (
                proj[["Player", "Team"]]
                .drop_duplicates(subset=["Player"], keep="first")
                .set_index("Player")["Team"]
            )
            my_roster.loc[_nan_team, "Team"] = (
                my_roster.loc[_nan_team, "Player"].map(_proj_team_lookup)
            )

        # Add next-GW projected points
        proj_points_df = proj[["Player", "Team", "Position", "Points"]].rename(
            columns={"Points": "Projected_Points"}
        ).copy()
        proj_points_df["__norm_name"] = proj_points_df["Player"].apply(canonical_normalize)
        my_roster["__norm_name"] = my_roster["Player"].apply(canonical_normalize)

        # Primary merge: name + team + position (exact)
        my_roster = my_roster.merge(
            proj_points_df[["__norm_name", "Team", "Position", "Projected_Points"]],
            on=["__norm_name", "Team", "Position"],
            how="left"
        )
        # Fallback 1: name + position only (catches team-code mismatches).
        _still_missing = my_roster["Projected_Points"].isna()
        if _still_missing.any():
            _proj_name_pos = (
                proj_points_df[["__norm_name", "Position", "Projected_Points"]]
                .drop_duplicates(subset=["__norm_name", "Position"], keep="first")
            )
            _fallback = my_roster.loc[_still_missing, ["__norm_name", "Position"]].merge(
                _proj_name_pos, on=["__norm_name", "Position"], how="left"
            )
            my_roster.loc[_still_missing, "Projected_Points"] = _fallback["Projected_Points"].values

        # Fallback 2: name only (catches FPL vs Rotowire position classification differences,
        # e.g. FPL MID vs Rotowire FWD for attacking players like Eze).
        _still_missing2 = my_roster["Projected_Points"].isna()
        if _still_missing2.any():
            _proj_name_only = (
                proj_points_df[["__norm_name", "Projected_Points"]]
                .drop_duplicates(subset=["__norm_name"], keep="first")
            )
            _fallback2 = my_roster.loc[_still_missing2, ["__norm_name"]].merge(
                _proj_name_only, on="__norm_name", how="left"
            )
            my_roster.loc[_still_missing2, "Projected_Points"] = _fallback2["Projected_Points"].values

        my_roster.drop(columns=["__norm_name"], inplace=True, errors="ignore")

        # `_attach_display_names` resolves by id first, so the ids must already
        # be on the frame -- they come from the normalized-name merge above,
        # which resolves 15 of 15 on the live roster. The `fuzz.WRatio`
        # backfill that used to sit here filled 0 and is deleted; see the
        # tombstone in `player_matching`.
        my_roster = _attach_display_names(my_roster)
        # `form_weeks` as well as `lookahead`: the available pool got the widget
        # value and the roster silently kept `config.FORM_LOOKBACK_WEEKS`, so moving
        # the slider compared a 2-week pool form against a 4-week roster form -- in
        # `_build_rationale`, which sets them side by side, and in the ROS form term.
        # They agree at the default, which is why it never showed.
        my_roster = _add_fdr_and_form(my_roster, fpl_stats, current_gw, lookahead,
                                      form_weeks=form_weeks)
        my_roster = _add_injury_data(my_roster, fpl_stats)

    # --- Multi-GW Projections (FFP) ---
    try:
        ffp_feed_result = get_ffp_feed()
        ffp_df = ffp_feed_result.df
    except Exception:
        ffp_feed_result, ffp_df = None, None

    _remaining_gws = max(1, 38 - current_gw)
    if not avail_all.empty:
        avail_all = blend_multi_gw_projections(avail_all, ffp_df, single_gw_col="Points", remaining_gws=_remaining_gws)
    if not my_roster.empty:
        # Roster uses Projected_Points as single-GW col
        pts_col = "Projected_Points" if "Projected_Points" in my_roster.columns else "Points"
        my_roster = blend_multi_gw_projections(my_roster, ffp_df, single_gw_col=pts_col, remaining_gws=_remaining_gws)

    # --- Rotowire Season Projections ---
    try:
        season_rankings_df = get_rotowire_season_rankings(config.ROTOWIRE_SEASON_RANKINGS_URL)
    except Exception:
        season_rankings_df = None

    if not avail_all.empty:
        avail_all = merge_season_projections(avail_all, season_rankings_df)
    if not my_roster.empty:
        my_roster = merge_season_projections(my_roster, season_rankings_df)

    # --- FFP Single-GW Data (Predicted, Start, LongStart) ---
    if not avail_all.empty:
        avail_all = merge_ffp_single_gw_data(avail_all, ffp_df)
    if not my_roster.empty:
        my_roster = merge_ffp_single_gw_data(my_roster, ffp_df)

    # --- Enrich reference pool with same data sources ---
    # fpl_stats is used as the reference pool for positional_percentile().
    # It must have the same enrichment columns (MultiGW_Proj, SeasonProjection, FFP_*)
    # as the scoring datasets for percentile comparisons to be meaningful.
    fpl_stats = blend_multi_gw_projections(fpl_stats, ffp_df, single_gw_col="points_per_game", remaining_gws=_remaining_gws)
    fpl_stats = merge_season_projections(fpl_stats, season_rankings_df)
    fpl_stats = merge_ffp_single_gw_data(fpl_stats, ffp_df)
    fpl_stats = enrich_reference_with_projections(fpl_stats, proj)
    # **The fixture term's reference column, which never existed.** `AvgFDRNextN`
    # was written onto the scored frames (available pool and roster) and not onto
    # the reference pool, so `compute_player_scores` took its min-max fallback:
    # the 15-row roster normalised against itself, the 98-row pool against
    # itself, and the suggestion gap then subtracts one from the other. Two
    # populations, min-max rather than percentile, and no visible symptom.
    #
    # `fpl_stats["Team"]` is a short code after `_enforce_rw_schema_fpl`, so the
    # key space matches the club-indexed Series exactly.
    fpl_stats["AvgFDRNextN"] = pd.to_numeric(
        fpl_stats["Team"].astype(str).map(_avg_fdr_by_short_name(current_gw, lookahead)),
        errors="coerce")

    render_ffp_status(ffp_feed_result, current_gw)

    # --- Positional Depth ---
    depth_map = {}
    if not my_roster.empty:
        depth_map = compute_positional_depth(my_roster)

    # --- Compute transfer suggestions (before rendering) ---
    # Pre-compute scores on both pools so _compute_transfer_suggestions can use them
    suggestions = []
    debug_rows: List[Dict] = []
    total_found = 0
    waiver_context = _build_waiver_context(waiver_order, my_entry_id, current_gw,
                                           transaction_mode=_mode)
    if not my_roster.empty and not avail_all.empty and sugg_positions:
        try:
            avail_all = _compute_waiver_score(avail_all, fpl_stats, current_gw=current_gw)
            my_roster = _compute_keep_score(my_roster, fpl_stats, current_gw=current_gw, depth_map=depth_map)
            # "Best per position" keeps the narrow 2x5 candidate window; every other
            # view scans the full roster against the full pool so the list really is
            # every move that would improve the team.
            _compact = sugg_view == SUGGESTION_VIEW_BEST_PER_POS
            suggestions, debug_rows = _compute_transfer_suggestions(
                avail_all, my_roster,
                top_n=None, depth_map=depth_map,
                positions=sugg_positions,
                roster_candidates=2 if _compact else None,
                avail_candidates=5 if _compact else None,
                one_per_position=_compact,
                waiver_context=waiver_context,
                floor_context=_build_floor_context(
                    waiver_order, my_entry_id, config.FPL_DRAFT_LEAGUE_ID),
                lanes=sugg_lanes,
            )
            total_found = len(suggestions)
            _limit = SUGGESTION_VIEW_LIMITS.get(sugg_view)
            if _limit is not None:
                # **Per lane, not across both.** A flat slice lets the hold lane
                # fill the whole quota and the rental lane vanish -- which is the
                # exact failure this feature exists to fix, re-created by the
                # display limit. `total_found` still reports the real count.
                _kept = []
                for _lane in (LANE_HOLD, LANE_RENT):
                    _kept.extend([x for x in suggestions if x['lane'] == _lane][:_limit])
                suggestions = _kept
        except Exception as e:
            st.warning(f"Could not compute transfer suggestions: {e}")

    # --- RENDER: Suggestion cards at top ---
    with suggestion_container:
        if not sugg_positions:
            st.info("Select at least one position above to see transfer suggestions.")
        else:
            _render_transfer_suggestions(
                suggestions, current_gw=current_gw, total_found=total_found,
                free_agency=_free_agency, waiver_gw=_waiver_gw,
            )
            _render_claim_plan(suggestions, free_agency=_free_agency,
                               lane=LANE_HOLD, lane_label="rest of season")
            _render_claim_plan(suggestions, free_agency=_free_agency,
                               lane=LANE_RENT, lane_label="just this week")
        _render_transfer_debug(debug_rows)

    st.markdown("---")

    # ---------------------------
    # MY ROSTER SECTION (with depth card)
    # ---------------------------
    if not my_roster.empty:
        # Scores already computed above; compute if not yet done (e.g. empty avail_all path)
        if "Keep Score" not in my_roster.columns:
            my_roster = _compute_keep_score(my_roster, fpl_stats, current_gw=current_gw, depth_map=depth_map)
        my_roster = my_roster.sort_values("Keep Score", ascending=False).reset_index(drop=True)

        st.subheader(f"Your Squad — {my_team.get('team_name', '(unknown)')}")

        # Compact depth card next to roster header
        if depth_map:
            _render_depth_card(depth_map)

        _display_roster = my_roster.copy()
        # Render the common name, not the roster feed's legal one -- see
        # _attach_display_names(). ("Daniel Munoz Mejia" -> "Daniel Munoz")
        _display_roster["Player"] = _display_names(_display_roster)
        _display_roster["Pos_Rank"] = positional_rank(
            _display_roster, fpl_stats, "Season_Points", ref_value_col="Season_Points"
        )
        for col in _display_roster.select_dtypes(include=[np.number]).columns:
            _display_roster[col] = _display_roster[col].round(2)

        # "Proj Pts" is the blend that the 1GW score is computed from, not raw
        # Rotowire. These tables used to print the raw number beside scores
        # derived from a different one, so a player could look mediocre in the
        # column and strong in the score with nothing to explain the gap.
        display_cols_roster = ["Player", "Team", "Position", "Pos_Rank", "Season_Points", "Proj", "Proj_Src", "Form", "AvgFDRNextN", "1GW", "ROS", "Keep Score"]
        display_cols_roster = [c for c in display_cols_roster if c in _display_roster.columns]
        roster_show = _display_roster[display_cols_roster].rename(
            columns={"Pos_Rank": "Pos Rank", "Proj": "Proj Pts", "Proj_Src": "Src"})
        render_styled_table(
            roster_show,
            col_formats={"Proj Pts": "{:.1f}", "Form": "{:.1f}", "AvgFDRNextN": "{:.1f}",
                         "1GW": "{:.2f}", "ROS": "{:.2f}", "Keep Score": "{:.2f}"},
            positive_color_cols=["1GW", "ROS", "Keep Score"],
        )

    else:
        st.subheader(f"Your Squad — {my_team.get('team_name', '(unknown)')}")
        st.info("No roster data available for this team.")

    st.markdown("---")

    # ---------------------------
    # AVAILABLE PLAYERS SECTION
    # ---------------------------
    # Waiver Score for available players (apply position filter for display)
    avail_display = avail_all.copy()
    if pos_filter:
        avail_display = avail_display[avail_display["Position"].isin(pos_filter)]
    if hide_locked and "Draft_State" in avail_display.columns:
        avail_display = avail_display[avail_display["Draft_State"] != ELEMENT_STATE_LOCKED]
    # Scores already computed above; compute if not yet done
    if "Transfer Score" not in avail_display.columns:
        avail_display = _compute_waiver_score(avail_display, fpl_stats, current_gw=current_gw)
    avail_display = avail_display.sort_values("Transfer Score", ascending=False).reset_index(drop=True)

    _display_avail = avail_display.copy()
    # Render the common name ("Savio"), not the source's legal one
    # ("Savio Moreira de Oliveira") -- see _attach_display_names().
    _display_avail["Player"] = _display_names(_display_avail)
    for col in _display_avail.select_dtypes(include=[np.number]).columns:
        _display_avail[col] = _display_avail[col].round(2)

    # Availability badge. Locked players stay in the table on purpose — they are
    # the next waiver round's targets — but they are never suggested as transfers.
    if "Draft_State" in _display_avail.columns:
        _locked_label = f"🔒 Locked (GW{_waiver_gw} waiver)" if _waiver_gw else "🔒 Locked"
        _display_avail["Status"] = np.where(
            _display_avail["Draft_State"] == ELEMENT_STATE_LOCKED,
            _locked_label,
            "✅ Free agent",
        )

    st.subheader("Available Players (ranked)")
    _avail_pos_labels = st.multiselect(
        "Positions",
        options=list(POSITION_LABEL_TO_CODE.keys()),
        default=list(POSITION_LABEL_TO_CODE.keys()),
        key="waiver_avail_positions",
        help="Narrow the board to the positions you are actually shopping for. "
             "Ranking is unaffected — these are the same scores, filtered.",
    )
    _total_avail = len(_display_avail)
    _display_avail = _filter_by_position(_display_avail, _avail_pos_labels)

    if _display_avail.empty:
        st.info(
            "No available players at "
            f"{', '.join(_avail_pos_labels) if _avail_pos_labels else 'the selected positions'}."
            if _avail_pos_labels else "Select at least one position to see the board."
        )
    else:
        # Counted after filtering: a caption reporting the locked players in the
        # whole pool while the table shows one position is a different number
        # from the one on screen.
        _n_locked_shown = int(
            _display_avail.get("Status", pd.Series(dtype=str)).str.startswith("🔒").sum()
        ) if "Status" in _display_avail.columns else 0
        _shown = (f"Showing {len(_display_avail)} of {_total_avail} available players."
                  if len(_display_avail) != _total_avail else
                  f"{_total_avail} available players.")
        if _n_locked_shown:
            _shown += (
                f" 🔒 {_n_locked_shown} are locked — recently dropped or added, so they "
                "cannot be picked up until the next waiver deadline. They are ranked here "
                "for waiver planning but are never proposed as transfers."
            )
        st.caption(_shown)

        display_cols_avail = ["Player", "Team", "Position", "Status", "Proj", "Proj_Src", "Form", "AvgFDRNextN", "Season_Points", "1GW", "ROS", "Transfer Score"]
        display_cols_avail = [c for c in display_cols_avail if c in _display_avail.columns]
        render_styled_table(
            _display_avail[display_cols_avail].reset_index(drop=True).rename(
                columns={"Proj": "Proj Pts", "Proj_Src": "Src"}),
            col_formats={"Proj Pts": "{:.1f}", "Form": "{:.1f}", "AvgFDRNextN": "{:.1f}",
                         "1GW": "{:.2f}", "ROS": "{:.2f}", "Transfer Score": "{:.2f}"},
            positive_color_cols=["1GW", "ROS", "Transfer Score"],
            max_height=500,
        )

    # ---------------------------
    # TRANSFER ACTIVITY SECTION
    # ---------------------------
    st.divider()
    st.header("📊 League Transfer Activity")

    # Fetch transaction data
    transactions_df = _fetch_all_transactions(config.FPL_DRAFT_LEAGUE_ID)

    if transactions_df.empty:
        st.info("No transaction data available for this league.")
    else:
        # Get team names and player mapping
        team_names = get_league_entries(config.FPL_DRAFT_LEAGUE_ID)
        player_map = get_fpl_player_mapping()

        # Build summary for chart
        summary_df = _build_transfer_activity_summary(transactions_df, team_names, player_map)

        # Summary stats as styled cards (above chart)
        total_free = summary_df['Free Transfers'].sum()
        total_waivers = summary_df['Accepted Waivers'].sum()
        total_failed = summary_df['Failed Waivers'].sum()

        def _stat_card(label, value, color, icon):
            return (
                f'<div style="flex:1;text-align:center;padding:16px 12px;background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);'
                f'border:1px solid #333;border-radius:10px;margin:0 6px;">'
                f'<div style="font-size:1.4em;margin-bottom:4px;">{icon}</div>'
                f'<div style="color:{color};font-size:1.8em;font-weight:bold;">{value}</div>'
                f'<div style="color:#888;font-size:0.85em;margin-top:4px;">{label}</div>'
                f'</div>'
            )

        cards_html = (
            '<div style="display:flex;margin-bottom:1rem;">'
            + _stat_card("Free Transfers", total_free, "#4ecca3", "🔄")
            + _stat_card("Accepted Waivers", total_waivers, "#3498db", "✅")
            + _stat_card("Failed Waivers", total_failed, "#e74c3c", "❌")
            + '</div>'
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        # Render the stacked bar chart
        _render_transfer_activity_chart(summary_df)

        # Build and display transfer history table
        st.subheader("Transfer History")

        history_df = _build_transfer_history_table(transactions_df, team_names, player_map)

        # Filters for the history table
        with st.expander("Filter History", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)

            # Team filter
            all_teams = ['All'] + sorted(history_df['Team'].unique().tolist())
            selected_team = filter_col1.selectbox("Team", all_teams, key="history_team_filter")

            # Type filter
            all_types = ['All'] + sorted(history_df['Type'].unique().tolist())
            selected_type = filter_col2.selectbox("Type", all_types, key="history_type_filter")

            # Result filter
            all_results = ['All'] + sorted(history_df['Result'].unique().tolist())
            selected_result = filter_col3.selectbox("Result", all_results, key="history_result_filter")

        # Apply filters
        filtered_history = history_df.copy()
        if selected_team != 'All':
            filtered_history = filtered_history[filtered_history['Team'] == selected_team]
        if selected_type != 'All':
            filtered_history = filtered_history[filtered_history['Type'] == selected_type]
        if selected_result != 'All':
            filtered_history = filtered_history[filtered_history['Result'] == selected_result]

        render_styled_table(filtered_history.reset_index(drop=True), max_height=400)
