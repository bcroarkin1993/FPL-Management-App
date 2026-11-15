"""My Week, for Draft: the pre-deadline round as one list.

The Draft manager's week is spread over four pages -- the Waiver Wire (which
window is open, when it shuts, where you sit in the queue, what claim is worth
making), Fixture Projections (who you play, is the XI right), Availability
(fitness) and the Trade Analyzer. Draft Home said nothing about any of it.

This module is the fetching and rendering half; the verdicts live in
`scripts/common/week_check.py` with Classic's, and the card, rows and buttons
come from `scripts/common/week_render.py`. It computes no projection of its
own: every number here is one the engine already produced, so a value that
looks wrong on this strip is wrong on the page that owns it too.

**Two things differ structurally from Classic**, and both make the Draft strip
simpler rather than harder:

- **The XI is readable, but not the pending one.** `/api/entry/{id}/event/{gw}`
  publishes Draft picks without credentials -- no `my-team` equivalent is
  needed -- but it **404s for any gameweek whose deadline has not passed**,
  exactly as Classic's `picks` does. Measured on 2026-11-15 with GW11 five
  days out: GW9 and GW10 return fifteen picks, GW11 and GW12 return 404. So
  the lineup compared here is *last* deadline's, the row says so, and it can
  be confirmed away the way Classic's can -- the alert is a claim the app
  cannot retract on its own, because the manager may have fixed it in FPL's
  own UI a minute ago.
- **The claim row is gated.** Building the claim engine's board costs ~15s on
  a cold cache (110 element-summary requests plus the league-strength pass),
  which is more than a Home page should spend unasked. Until the button is
  pressed that row is `unknown` -- never `ok` -- because it has derived
  nothing, and "nothing to do" and "not looked yet" are opposite claims.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

import config
from scripts.common import cards, lineup_confirm, week_check, week_render
from scripts.common.analytics import gameweek_deadlines
from scripts.common.data_validation import check_week_plan
from scripts.common.display_names import attach_display_names, display_of
from scripts.common.error_helpers import get_logger
from scripts.common.fixture_helpers import club_fixture_counts
from scripts.common.fpl_draft_api import (
    ELEMENT_STATE_LOCKED,
    get_current_gameweek,
    get_draft_transaction_window,
    get_gameweek_fixtures,
    get_league_element_states,
    get_league_player_ownership,
    get_league_waiver_order,
    get_team_actual_lineup,
    get_team_composition_for_gameweek,
    requires_trade_approval,
)
from scripts.common.optimization import best_xi_index
from scripts.common.player_pool import blended_pool
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import _map_position_to_rw
from scripts.common.waiver_priority import managers_ahead, my_waiver_pick
from scripts.fpl.injuries import get_fpl_availability_df

_logger = get_logger("fpl_app.draft.my_week")

#: How far ahead the blanks-and-doubles row looks.
FIXTURE_HORIZON_WEEKS = 6

#: Matches `classic.my_week.CACHE_TTL`, and sits under the 300s on the fetches
#: underneath, so it can never serve data older than those would have.
CACHE_TTL = 120

#: Below this the bench is in a defensible order; auto-subs are a conditional
#: loss and a tenth of a point of it is not worth a row.
BENCH_MIN_GAIN = 1.0

#: Session key the claim gate writes. Kept out of `_assemble`'s cache key by
#: being passed in, for the reason `confirmed_signature` is in Classic's.
CLAIM_STATE_KEY = "_draft_my_week_claim"


# ------------------------------------------------------------------
# Inputs. Each resolves one question and degrades to None, never to a default:
# the builders turn a None into an honest "unknown" row.
# ------------------------------------------------------------------
def _deadlines():
    try:
        return gameweek_deadlines()
    except Exception as exc:
        _logger.warning("Could not read the gameweek calendar: %s", exc)
        return None


def _roster(league_id, team_id, gw):
    """The fifteen, with the blend's projection attached.

    Joined on `Player_Code` first -- the only id both FPL APIs agree on -- and
    on the Classic `Player_ID` second. `get_team_composition_for_gameweek`
    publishes both, already translated out of the Draft id space.
    """
    try:
        roster = get_team_composition_for_gameweek(league_id, team_id, gw)
    except Exception as exc:
        _logger.warning("Could not resolve the roster: %s", exc)
        return None
    if roster is None or roster.empty:
        return None
    return _attach_projection(roster, gw)


def _attach_projection(frame, gw):
    """`Proj` from the shared blend, which the Hub and Chip Strategy also read."""
    if frame is None or frame.empty:
        return frame
    try:
        pool = blended_pool(int(gw), config.ROTOWIRE_URL)
    except Exception as exc:
        _logger.warning("Could not read the blended pool: %s", exc)
        return frame
    if pool is None or pool.empty:
        return frame

    out = frame.copy()
    # Resolve the rendered name here, once, rather than letting `display_of`
    # fall back to whatever `Player` holds. These frames come from the Draft
    # bootstrap, so `Player` is the full legal name -- "Igor Thiago Nascimento
    # Rodrigues" where the card wants "Igor Thiago". Match on `Player`,
    # display `Display_Name`, per "Player Display Names".
    try:
        out = attach_display_names(out)
    except Exception as exc:
        _logger.warning("Could not resolve display names: %s", exc)

    # `Team_ID` as well as `Proj`: the roster frame carries a short club code
    # and nothing numeric, and every fixture join in the app is keyed on the
    # club id. Without it the blanks-and-doubles row is handed an empty set of
    # clubs and reports `unknown` on a perfectly readable fixture list --
    # which is the "a gate keyed on a column some frames do not carry" failure
    # this codebase has paid for four times.
    wanted = [c for c in ("Proj", "Team_ID") if c in pool.columns]
    for key in ("Player_Code", "Player_ID"):
        if key not in out.columns or key not in pool.columns:
            continue
        ref = (pool[[key] + wanted].dropna(subset=[key])
               .drop_duplicates(subset=[key]).set_index(key))
        ids = pd.to_numeric(out[key], errors="coerce")
        for col in wanted:
            mapped = ids.map(ref[col])
            out[col] = (mapped if col not in out.columns
                        else out[col].where(out[col].notna(), mapped))
        if "Proj" in out.columns and out["Proj"].notna().all():
            break
    return out


def _availability_flags(roster, deadlines):
    try:
        return week_check.squad_availability(
            roster, get_fpl_availability_df(), deadlines=deadlines)
    except Exception as exc:
        _logger.warning("Could not read availability: %s", exc)
        return None


def _fixture_counts(target_gw):
    """Per-club fixture counts, or an empty frame.

    An unreadable fixture list returns empty, which the builder reports as
    unknown -- never as twenty blanking clubs, which is what an outage looks
    like and would zero every projection on the page.
    """
    gameweeks = [gw for gw in range(int(target_gw),
                                    int(target_gw) + FIXTURE_HORIZON_WEEKS)
                 if gw <= 38]
    try:
        return club_fixture_counts(gameweeks), gameweeks
    except Exception as exc:
        _logger.warning("Could not read the fixture list: %s", exc)
        return pd.DataFrame(), gameweeks


def _optimal_total(roster):
    """What the fifteen are worth arranged correctly. No captain: Draft
    disables them, so this is the eleven and nothing doubled."""
    if roster is None or roster.empty or "Proj" not in roster.columns:
        return None
    points = pd.to_numeric(roster["Proj"], errors="coerce").fillna(0.0)
    frame = pd.DataFrame({"Position": roster["Position"].map(_map_position_to_rw),
                          "_p": points})
    try:
        chosen = best_xi_index(frame, "_p")
    except KeyError as exc:
        _logger.warning("Could not pick an XI: %s", exc)
        return None
    return float(points.loc[chosen].sum()) if chosen else None


def _set_lineup(team_id, gw):
    """``(total, swap, bench_issue, source_gw)`` for the XI FPL has stored.

    Draft publishes picks without credentials, but **not for a gameweek whose
    deadline has not passed** -- that 404s, exactly as Classic's `picks` does.
    So this reads the most recent gameweek it can and reports which, and the
    row discloses that rather than implying a live view.

    `squad_position` 1-11 is the XI, 12 is the reserve keeper and 13-15 the
    outfield auto-sub order.

    ``swap`` is the single most valuable legal change, found by re-running
    `best_xi_index` over the proposed eleven rather than by encoding the
    formation quotas a second time.
    """
    lineup, source_gw = None, None
    # The current gameweek's picks 404 until its deadline passes, so fall back
    # to the most recent one FPL will publish. Two steps rather than one: a
    # gameweek can be skipped entirely around a blank.
    for candidate in (int(gw), int(gw) - 1, int(gw) - 2):
        if candidate < 1:
            break
        try:
            attempt = get_team_actual_lineup(team_id, candidate)
        except Exception as exc:
            _logger.warning("Could not read the set lineup: %s", exc)
            continue
        if attempt is not None and not attempt.empty:
            lineup, source_gw = attempt, candidate
            break
    if lineup is None or "squad_position" not in lineup.columns:
        return None, None, None, None

    lineup = _attach_projection(lineup, gw)
    if "Proj" not in lineup.columns:
        return None, None, None, None

    slots = pd.to_numeric(lineup["squad_position"], errors="coerce")
    points = pd.to_numeric(lineup["Proj"], errors="coerce").fillna(0.0)
    starters = lineup.index[slots <= 11]
    bench = lineup.index[slots > 11]
    if len(starters) != 11:
        return None, None, None, None

    total = float(points.loc[starters].sum())
    positions = lineup["Position"].map(_map_position_to_rw)

    best = None
    for out_idx in starters:
        for in_idx in bench:
            delta = float(points.loc[in_idx] - points.loc[out_idx])
            if delta <= 0 or (best is not None and delta <= best["delta"]):
                continue
            proposed = [i for i in starters if i != out_idx] + [in_idx]
            frame = pd.DataFrame({"Position": positions.loc[proposed],
                                  "_p": points.loc[proposed]})
            try:
                if len(best_xi_index(frame, "_p")) != 11:
                    continue           # the swap would leave an illegal formation
            except KeyError:
                continue
            best = {"delta": delta,
                    "out_name": display_of(lineup.loc[out_idx]),
                    "in_name": display_of(lineup.loc[in_idx])}

    return (total, best, _bench_issue(lineup, slots, points, positions, bench),
            source_gw)


def _bench_issue(lineup, slots, points, positions, bench):
    """A later outfield sub projecting above an earlier one.

    Auto-subs come on in slot order, so a stronger player at 15 than at 13
    is points left on the bench -- but only if a starter does not play, which
    is why the builder makes this a warning rather than an action.

    Slot 12 is excluded: it is the reserve goalkeeper, who can only replace
    the keeper, so comparing him with an outfielder says nothing.
    """
    outfield = [i for i in bench
                if int(slots.loc[i]) > 12 and positions.loc[i] != "G"]
    if len(outfield) < 2:
        return None
    ordered = sorted(outfield, key=lambda i: int(slots.loc[i]))
    for earlier in range(len(ordered) - 1):
        for later in range(earlier + 1, len(ordered)):
            a, b = ordered[earlier], ordered[later]
            gain = float(points.loc[b] - points.loc[a])
            if gain >= BENCH_MIN_GAIN:
                return {"better_name": display_of(lineup.loc[b]),
                        "better_proj": float(points.loc[b]),
                        "ahead_name": display_of(lineup.loc[a]),
                        "ahead_proj": float(points.loc[a])}
    return None


def _opponent(league_id, team_id, gw, roster_total):
    """``(kind, data)`` -- "ok" / "no_fixture" / "unknown".

    Three absences that are not the same absence, so a kind rather than an
    optional tuple: Draft is always H2H, so there is no "this league has no
    opponent" case, but a bye week and an unreadable fixture list are
    different claims.
    """
    try:
        fixtures = get_gameweek_fixtures(league_id, gw)
        ownership = get_league_player_ownership(league_id)
    except Exception as exc:
        _logger.warning("Could not read this gameweek's fixture: %s", exc)
        return "unknown", {}
    if not fixtures:
        return "no_fixture", {}

    mine = (ownership.get(int(team_id), {}) or {}).get("team_name", "")
    opponent_name = None
    for line in fixtures:
        text = str(line)
        if mine and mine in text:
            halves = [h.strip() for h in text.replace(" vs ", "|").split("|")]
            for half in halves:
                if mine not in half:
                    opponent_name = half
                    break
            break
    if not opponent_name:
        return "unknown", {}

    opp_id = next((tid for tid, blob in ownership.items()
                   if (blob or {}).get("team_name", "").strip()
                   and (blob or {}).get("team_name", "").strip() in opponent_name), None)
    theirs = None
    if opp_id is not None:
        theirs = _optimal_total(_roster(league_id, opp_id, gw))
    if roster_total is None or theirs is None:
        return "unknown", {"opponent": opponent_name}
    return "ok", {"opponent": opponent_name, "mine": roster_total,
                  "theirs": theirs}


def _league_state(league_id):
    """The window, the queue and the locked count, each failing open."""
    try:
        window = get_draft_transaction_window(league_id)
    except Exception:
        _logger.warning("Could not read the transaction window", exc_info=True)
        window = None
    try:
        order = get_league_waiver_order(league_id)
    except Exception:
        _logger.warning("Could not read the waiver order", exc_info=True)
        order = []
    try:
        states = get_league_element_states(league_id)
    except Exception:
        _logger.warning("Could not read element states", exc_info=True)
        states = {}
    return window, order, states


def _waiver_deadline(gw):
    try:
        from scripts.common.waiver_alerts import get_next_transaction_deadline
        deadline, _kickoff, _gw = get_next_transaction_deadline(gw=gw)
        return deadline
    except Exception as exc:
        _logger.warning("Could not resolve the waiver deadline: %s", exc)
        return None


def _trade_deadline(waiver_deadline, trades_code):
    """24h earlier where approval applies. An unreadable setting resolves to
    "approval required" -- the earlier deadline -- because the two errors are
    not symmetric: a reminder a day early is an annoyance, one after the
    window has shut costs the manager the gameweek."""
    if waiver_deadline is None:
        return None, False
    try:
        from scripts.draft.trade_analyzer import trade_deadline_from
        approval = requires_trade_approval(trades_code)
        assumed = trades_code is None or approval is None
        return trade_deadline_from(waiver_deadline, True if assumed else approval), assumed
    except Exception as exc:
        _logger.warning("Could not resolve the trade deadline: %s", exc)
        return None, True


def _my_pending_trades(states, roster):
    """How many of *your* players sit inside an accepted trade.

    `in_accepted_trade` on element-status is the only pending-trade signal FPL
    publishes; there is no offers inbox wired into this app.
    """
    if not states or roster is None or roster.empty:
        return 0
    try:
        from scripts.draft.waiver_core import _codes_in_state  # noqa: F401
    except Exception:
        pass
    pending = {int(eid) for eid, blob in states.items()
               if (blob or {}).get("in_accepted_trade")}
    if not pending or "Player_ID" not in roster.columns:
        return 0
    try:
        from scripts.common.element_ids import code_maps
        maps = code_maps() or {}
        draft_to_code = maps.get("draft_id_to_code") or {}
        mine_codes = set(pd.to_numeric(roster.get("Player_Code"),
                                       errors="coerce").dropna().astype(int))
        pending_codes = {draft_to_code.get(int(e)) for e in pending}
        return len(mine_codes & {c for c in pending_codes if c is not None})
    except Exception as exc:
        _logger.warning("Could not resolve pending trades: %s", exc)
        return 0


def _confirm_key(team_id):
    """Namespace the Draft entry id.

    `lineup_confirm` keys its file on `"<team_id>:<gw>"`, and a Draft entry id
    and a Classic entry id are both bare integers -- so without a prefix one
    format could silence the other's alert.
    """
    return "draft-%s" % team_id


def _squad_signature(league_id, team_id, gw) -> str:
    """A signature of the fifteen, so a transfer re-arms the lineup alert.

    Membership, not arrangement: re-ordering a bench should not re-arm a
    confirmation, but claiming a player should.
    """
    roster = None
    try:
        roster = get_team_composition_for_gameweek(league_id, team_id, gw)
    except Exception:
        return ""
    if roster is None or roster.empty:
        return ""
    ids = pd.to_numeric(roster.get("Player_Code"), errors="coerce").dropna()
    if ids.empty:
        ids = pd.to_numeric(roster.get("Player_ID"), errors="coerce").dropna()
    return lineup_confirm.squad_signature([int(x) for x in ids])


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------
@dataclass(frozen=True)
class DraftWeekView:
    """One assembly of the week, so Home can place its three parts.

    The card, the buttons and the detail land in different containers -- a CSS
    grid, a controls row and a tab -- and all three need the same items, so
    the assembly happens once and is passed around rather than re-run.
    """

    items: list
    team_id: Any
    league_id: Any
    target_gw: int
    squad_size: int
    has_xi: bool
    claim_checked: bool
    claim_plan: tuple = ()
    signature: str = ""

    @property
    def flagged(self) -> list:
        return (week_check.attention_items(self.items)
                + [i for i in self.items
                   if i.status == week_check.STATUS_UNKNOWN])


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _assemble(league_id, team_id, current_gw, claim_state, confirmed=False,
              signature=""):
    """Every row but the claim search, keyed on primitives only.

    `claim_state` and `confirmed` are both passed *in* rather than read
    inside, or pressing either button would be swallowed by the cached entry
    it is meant to invalidate -- the same reason Classic's
    `confirmed_signature` is in its key.
    """
    deadlines = _deadlines()
    roster = _roster(league_id, team_id, current_gw)
    window, order, states = _league_state(league_id)
    locked = sum(1 for s in (states or {}).values()
                 if (s or {}).get("status") == ELEMENT_STATE_LOCKED)

    team_name = ""
    try:
        team_name = (get_league_player_ownership(league_id).get(int(team_id), {})
                     or {}).get("team_name", "")
    except Exception:
        pass

    optimal = _optimal_total(roster)
    set_total, swap, bench, source_gw = _set_lineup(team_id, current_gw)
    counts, gameweeks = _fixture_counts(current_gw)

    squad_team_ids = set()
    if roster is not None and not roster.empty and "Team_ID" in roster.columns:
        squad_team_ids = set(pd.to_numeric(roster["Team_ID"],
                                           errors="coerce").dropna().astype(int))

    kind, opp = _opponent(league_id, team_id, current_gw, optimal)
    opponent_item = None
    if kind == "ok":
        opponent_item = week_check.build_opponent_item(
            opp.get("opponent"), opp.get("mine"), opp.get("theirs"),
            section=week_check.SECTION_DRAFT)
    elif kind == "unknown":
        opponent_item = week_check.build_opponent_item(
            section=week_check.SECTION_DRAFT)
    # "no_fixture" contributes nothing: a bye week is not a question we failed
    # to answer.

    waiver_deadline = _waiver_deadline(current_gw)
    trades_code = (window or {}).get("trades")
    trade_deadline, assumed = _trade_deadline(waiver_deadline, trades_code)

    claim_state = claim_state or {}
    items = week_check.build_draft_week_checklist(
        deadline=week_check.build_deadline_item(deadlines, current_gw),
        opponent=opponent_item,
        squad=week_check.build_draft_squad_item(roster, team_name, locked),
        availability=week_check.build_availability_item(
            _availability_flags(roster, deadlines)),
        fixtures=week_check.build_fixtures_item(counts, squad_team_ids,
                                                gameweeks, current_gw),
        lineup=week_check.build_draft_lineup_item(
            optimal, set_total, swap, bench, source_gw=source_gw,
            target_gw=current_gw, confirmed=bool(confirmed)),
        waivers=week_check.build_waivers_item(
            window, waiver_deadline, my_waiver_pick(order, team_id),
            len(managers_ahead(order, team_id) or []), locked),
        claim=week_check.build_claim_item(
            claim=claim_state.get("claim"),
            checked=bool(claim_state.get("checked")),
            pool_size=claim_state.get("pool"),
            reason=claim_state.get("reason", "")),
        trades=week_check.build_trades_item(
            window, trade_deadline, _my_pending_trades(states, roster), assumed),
    )
    size = 0 if roster is None else len(roster)
    return (items, size, set_total is not None,
            tuple(claim_state.get("plan") or ()), signature)


def run_claim_search(league_id, team_id, current_gw) -> dict:
    """The gated half: build the board and ask it for the best claim.

    Separated from `_assemble` because it is the expensive one -- ~15s on a
    cold cache -- and because a failure here must read as `unknown` rather
    than as "no claim worth making".
    """
    from scripts.draft import waiver_core

    try:
        board = waiver_core.build_waiver_board(
            league_id, team_id, current_gw, entry_id=team_id)
    except Exception as exc:
        _logger.warning("Could not build the waiver board: %s", exc, exc_info=True)
        return {"checked": True, "reason": "The claim search could not run."}

    if not board.ok:
        reason = next((m for sev, m in board.notes if sev in ("error", "info")),
                      "The claim search could not run.")
        return {"checked": True, "reason": reason}

    best = waiver_core.best_claim(board)
    suggestions, _, _ = waiver_core.search_board(
        board, positions=["G", "D", "M", "F"], compact=False)
    plan = []
    try:
        from scripts.common.waiver_priority import rank_claim_plan
        for row in (rank_claim_plan(suggestions) or [])[:5]:
            plan.append({
                "Priority": row.get("claim_priority"),
                "Add": row.get("add_player"),
                "Drop": row.get("drop_player"),
                "Gain": round(float(row.get("points_gain") or 0.0), 2),
            })
    except Exception as exc:
        _logger.warning("Could not rank the claim plan: %s", exc)

    return {"checked": True, "claim": best, "pool": int(len(board.available)),
            "plan": tuple(tuple(sorted(p.items())) for p in plan)}


def build_week_view(league_id=None, current_gw=None) -> Optional[DraftWeekView]:
    """The whole week, or ``None`` when there is nothing to build it from.

    Never raises: Draft Home renders the standings below this strip, and a
    fault here must cost the week rather than the page.
    """
    team_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    league_id = league_id or getattr(config, "FPL_DRAFT_LEAGUE_ID", None)
    if not team_id or not league_id:
        return None
    if current_gw is None:
        try:
            current_gw = int(get_current_gameweek() or 1)
        except Exception:
            current_gw = 1

    claim_state = st.session_state.get(CLAIM_STATE_KEY) or {}
    if claim_state.get("gw") != current_gw:
        claim_state = {}          # a new gameweek re-arms the question

    signature = _squad_signature(league_id, team_id, int(current_gw))
    confirmed = bool(signature) and lineup_confirm.is_confirmed(
        _confirm_key(team_id), int(current_gw), signature)
    try:
        items, size, has_xi, plan, signature = _assemble(
            league_id, team_id, int(current_gw),
            {k: v for k, v in claim_state.items() if k != "gw"},
            confirmed=confirmed, signature=signature)
    except Exception:
        _logger.warning("Could not assemble the Draft week", exc_info=True)
        return None

    view = DraftWeekView(items=items, team_id=team_id, league_id=league_id,
                         target_gw=int(current_gw), squad_size=size,
                         has_xi=has_xi,
                         claim_checked=bool(claim_state.get("checked")),
                         claim_plan=plan, signature=signature)

    for issue in check_week_plan(items, squad_size=size,
                                 xi_size=11 if has_xi else 0) or []:
        _logger.log(40 if issue.severity == "error" else 20,
                    "Draft week plan: %s", issue)
    return view


# ------------------------------------------------------------------
# Rendering. The card, rows and buttons are shared with Classic.
# ------------------------------------------------------------------
def week_card_html(view: DraftWeekView) -> str:
    """The strip as a card, for Home's top band."""
    if view is None:
        return ""
    return week_render.week_card_html(view, all_clear="Nothing needs doing "
                                                      "before the deadline.")


def _run_claim_search(view: DraftWeekView) -> None:
    state = run_claim_search(view.league_id, view.team_id, view.target_gw)
    state["gw"] = view.target_gw
    st.session_state[CLAIM_STATE_KEY] = state


def render_week_actions(view: DraftWeekView) -> None:
    """One button per flagged row, plus the claim gate.

    The gate leads rather than trails: it is the row a Draft manager most
    wants answered, and the only one that is not already answered.
    """
    if view is None:
        return
    leading = []
    lineup_row = next((i for i in view.items if i.key == "lineup"), None)
    if (lineup_row is not None and lineup_row.status == week_check.STATUS_ACT
            and view.signature):
        leading.append(week_render.ActionButton(
            label="✓ I've set my XI", key="confirm_xi",
            help=("Silences the lineup alert for this gameweek. FPL publishes "
                  "no pending XI before the deadline, so the comparison is "
                  "against your last one; this comes back if you transfer."),
            on_click=lambda: lineup_confirm.confirm_lineup(
                _confirm_key(view.team_id), view.target_gw, view.signature)))
    if not view.claim_checked:
        leading.append(week_render.ActionButton(
            label="🔎 Check for claims", key="claim_search",
            help=("Searches the whole available pool against your roster. It "
                  "is not run automatically because a cold search costs "
                  "around fifteen seconds."),
            on_click=lambda: _run_claim_search(view)))
    week_render.render_actions(view, key_prefix="draft_my_week", leading=leading)


def render_week_detail(view: DraftWeekView) -> None:
    """Every row, and the claim plan behind it."""
    if view is None:
        st.caption("Set your Draft league and team on League Setup to see your "
                   "week here.")
        if st.button("🆔 Go to League Setup", key="draft_my_week_setup"):
            st.session_state["_pending_nav_section"] = week_check.SECTION_FPL
            st.session_state["_pending_nav_page"] = "\U0001f194  League Setup"
            st.rerun()
        return

    checks_tab, plan_tab = st.tabs(["All checks", "Claim plan"])
    with checks_tab:
        week_render.render_checks(view.items)
    with plan_tab:
        _render_claim_plan(view)


def _render_claim_plan(view: DraftWeekView) -> None:
    """The order to submit claims in.

    A failed claim costs nothing -- only a successful one rotates you to the
    back of the queue -- so the honest order puts long shots first, which is
    what `rank_claim_plan` builds and why it is not a plain sort.
    """
    if not view.claim_checked:
        st.caption("Press **Check for claims** above to search the available "
                   "pool. It is not automatic because a cold search costs "
                   "around fifteen seconds.")
        return
    if not view.claim_plan:
        st.caption("No claim worth making was found.")
        return
    rows = [dict(entry) for entry in view.claim_plan]
    render_styled_table(pd.DataFrame(rows), title="Suggested claim order",
                        col_formats={"Gain": "{:+.1f}"},
                        positive_color_cols=["Gain"])
    st.caption("A failed claim costs nothing — only a successful one sends you "
               "to the back of the queue — so put the long shots first.")
