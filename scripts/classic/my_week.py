"""My Week — the pre-deadline alert strip on Classic Home.

The app answered every pre-deadline question and answered each one somewhere
else: the deadline and squad on Home, fitness on Availability, the XI and
captain on Fixture Projections, free transfers on Transfer Suggestions, the chip
call on Chip Strategy, prices on Price Changes. Six pages, and the only way to
find out that five of them had nothing to say was to open all six.

This renders that round as a strip at the top of Classic Home: one pill per
thing that needs attention, a two-tab detail block under it, and an **Open**
button per row that jumps straight to the page which fixes it. It computes no
projection of its own -- every figure is the one the engine already produced --
so a number that looks wrong here is wrong on the page it came from too.

Three properties it is built around:

* **Nothing is behind a button.** A flag you have to ask for is not a flag: the
  thing that makes the transfers row an *alert* is that there is a +11.5 point
  move available, which is exactly the half that used to need prompting. The ILP
  frontier is 0.27s measured (CLAUDE.md's own figure is "under half a second"),
  and the whole assembled checklist is cached, so a rerun pays nothing.
* **It never guesses.** A question whose inputs are missing renders *unknown*,
  in grey, worded as a failure to check -- never as "nothing to do". Those two
  read identically on a card and are opposite claims, which is why the item
  model carries ``evidence`` and ``check_week_plan`` errors on an all-clear that
  recorded none. An outage must not be able to hide under a row of green ticks.
* **It is your week, not the league's.** Home is league-scoped and has a league
  selector; eight of the nine rows are about your own squad. The exception is
  the head-to-head row, which is scoped to the league Home has selected.

The checklist logic is pure and lives in ``scripts/common/week_check.py``; this
module is fetching and rendering only.
"""

import pandas as pd
import streamlit as st

import config
from scripts.common import chips, h2h, lineup_confirm, week_check
from scripts.common.analytics import gameweek_deadlines
from scripts.common.classic_squad import SOURCE_MY_TEAM, resolve_classic_squad
from scripts.common.data_validation import check_week_plan
from scripts.common.display_names import display_of
from scripts.common.error_helpers import get_logger
from scripts.common.fixture_helpers import club_fixture_counts
from scripts.common.fpl_auth import has_auth
from scripts.common.fpl_classic_api import (
    get_classic_bootstrap_static,
    get_classic_team_history,
)
from scripts.common.optimization import best_xi_index
from scripts.common.player_pool import blended_pool, squad_frame
from scripts.common.price_signals import build_price_data, squad_price_pressure
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import (
    _map_position_to_rw,
    compact_html,
    format_deadline_stamp,
    format_time_until,
)
from scripts.classic.fixture_projections import _is_gameweek_started
from scripts.fpl.injuries import get_fpl_availability_df

_logger = get_logger("fpl_app.classic.my_week")

#: How far ahead the fixture check looks for blanks and doubles. Six weeks is
#: the horizon FFP publishes and the one Chip Strategy uses, so the two surfaces
#: cannot disagree about what is "coming up".
FIXTURE_HORIZON_WEEKS = 6

#: Short enough to notice a transfer you have just made, and it sits under
#: `blended_pool`'s own 300s -- so it can never serve a blend older than that
#: cache would have served anyway. The same two-layer arrangement as
#: `ffp_feed._memoised`.
CACHE_TTL = 120

#: The strip's height, in pixels, computed rather than left to the browser --
#: the same approach as the Draft lineup cards, and for the same reason: the
#: league card beside it has to be the same height, and only Python knows how
#: many alert rows there are. Getting these wrong costs alignment and nothing
#: else, so they are tuned by eye rather than derived.
STRIP_CHROME_PX = 58       #: border, padding and the "Your week" header
STRIP_ROW_PX = 31          #: one alert row
STRIP_CAVEAT_PX = 24       #: the grey staleness line
STRIP_BUTTON_PX = 46       #: the button row under the list

#: Status -> (icon, colour). Unknown is grey rather than green: a question we
#: could not answer must not read as one answered "fine".
_STATUS_STYLE = {
    week_check.STATUS_OK: ("✅", "#00ff87"),
    week_check.STATUS_UNKNOWN: ("❔", "#9ca3af"),
    week_check.STATUS_WARN: ("⚠️", "#ffa502"),
    week_check.STATUS_ACT: ("❗", "#ff4757"),
}

#: What each pill says. Deliberately shorter than the row's own verdict: the bar
#: is scanned, the rows are read.
_PILL_TEXT = {
    "squad": "Squad not current",
    "availability": "Fitness",
    "fixtures": "Fixtures",
    "lineup": "Your XI",
    "transfers": "Transfers",
    "chips": "Chips",
    "prices": "Prices",
    "opponent": "This week",
    "deadline": "Deadline",
}


# ------------------------------------------------------------------
# Inputs. Each resolves one question and degrades to None, never to a default:
# the builders turn a None into an honest "unknown" row.
# ------------------------------------------------------------------
def _availability_flags(squad_df, deadlines):
    try:
        return week_check.squad_availability(
            squad_df, get_fpl_availability_df(), deadlines=deadlines)
    except Exception as exc:
        _logger.warning("Could not read availability: %s", exc)
        return None


def _fixture_counts(target_gw):
    """Per-club fixture counts over the window, or an empty frame.

    An unreadable fixture list returns empty, which the builder reports as
    unknown -- never as twenty blanking clubs, which is what an outage looks
    like and would zero every projection on the page.
    """
    gameweeks = [gw for gw in range(int(target_gw),
                                    int(target_gw) + FIXTURE_HORIZON_WEEKS)
                 if gw <= chips.LAST_GAMEWEEK]
    try:
        return club_fixture_counts(gameweeks), gameweeks
    except Exception as exc:
        _logger.warning("Could not read the fixture list: %s", exc)
        return pd.DataFrame(), gameweeks


def _free_transfers(history, resolution, current_gw, status, team_id):
    """The free-transfer count, through the page that owns the rules.

    ``fh_gws`` matters to the replay lane: a Free Hit week neither spends a
    transfer nor earns one, and a replay that counts it reports the wrong number
    for the rest of the season.
    """
    try:
        from scripts.classic.transfers import resolve_free_transfers
        used = (status or {}).get("used") or {}
        return resolve_free_transfers(
            history, resolution.entry_history or {}, current_gw,
            fh_gws=set(used.get(chips.CHIP_FREEHIT) or []),
            team_id=team_id, squad_source=resolution.source,
        )
    except Exception as exc:
        _logger.warning("Could not resolve free transfers: %s", exc)
        return None


def _gw_points(pool, target_gw):
    """This gameweek's expected points per player, indexed by element id."""
    if pool is None or pool.empty or "Player_ID" not in pool.columns:
        return None
    return (pool.set_index("Player_ID")[["Proj"]]
            .rename(columns={"Proj": int(target_gw)}))


def _lineup(squad_df):
    """``(optimal total, captain, his projection)`` for the best legal XI.

    The *optimal* arrangement of the fifteen -- what the squad is worth if it is
    set correctly, which is the number the set lineup is measured against.
    """
    if squad_df is None or squad_df.empty or "Proj" not in squad_df.columns:
        return None, None, None
    points = pd.to_numeric(squad_df["Proj"], errors="coerce").fillna(0.0)
    frame = pd.DataFrame({"Position": squad_df["Position"].map(_map_position_to_rw),
                          "_p": points})
    try:
        chosen = best_xi_index(frame, "_p")
    except KeyError as exc:
        _logger.warning("Could not pick an XI: %s", exc)
        return None, None, None
    if not chosen:
        return None, None, None
    xi = points.loc[chosen]
    skipper = xi.idxmax()
    return (float(xi.sum() + xi.max()), display_of(squad_df.loc[skipper]),
            float(xi.max()))


def _set_lineup(squad_df):
    """``(total, captain, best available swap)`` for the XI FPL has stored.

    The lineup a manager has actually *set*, read from ``pick_position`` (1-11
    is the XI). Scored the way the gameweek will score it: the eleven plus
    another copy of whoever wears the armband -- not the optimal captain, since
    the point is to measure the arrangement as it stands.

    ``swap`` is the single most valuable legal change, which is what makes the
    alert actionable rather than merely true. It is the best bench player
    against the worst starter **whose positions leave the formation legal** --
    checked by re-running `best_xi_index` over the proposed eleven rather than
    by encoding the quotas a second time.
    """
    if (squad_df is None or squad_df.empty or "pick_position" not in squad_df.columns
            or "Proj" not in squad_df.columns):
        return None, None, None
    picks = pd.to_numeric(squad_df["pick_position"], errors="coerce")
    if not picks.notna().any():
        return None, None, None

    points = pd.to_numeric(squad_df["Proj"], errors="coerce").fillna(0.0)
    starters = squad_df.index[picks <= 11]
    bench = squad_df.index[picks > 11]
    if len(starters) != 11:
        return None, None, None

    captain_idx = None
    if "is_captain" in squad_df.columns:
        marked = squad_df.index[squad_df["is_captain"].astype(bool)]
        captain_idx = marked[0] if len(marked) else None
    if captain_idx is None or captain_idx not in starters:
        captain_idx = points.loc[starters].idxmax()

    total = float(points.loc[starters].sum() + points.loc[captain_idx])
    captain = display_of(squad_df.loc[captain_idx])

    positions = squad_df["Position"].map(_map_position_to_rw)
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
                    continue       # the swap would leave an illegal formation
            except KeyError:
                continue
            best = {"delta": delta, "out_name": display_of(squad_df.loc[out_idx]),
                    "in_name": display_of(squad_df.loc[in_idx])}
    return total, captain, best


def _transfers_made(team_id, target_gw, resolution):
    """How many transfers are already in for this gameweek, or ``None``.

    FPL's own transfer log is the source and it **does** publish per gameweek --
    verified live, 18 rows across five gameweeks for one entry. Whether it
    withholds an in-week move until the deadline is still unobserved (CLAUDE.md
    records the same open question for the replay), so "no rows for this week"
    is reported as *unknown* rather than as zero unless another source can
    confirm it: an authenticated squad states the count outright, and a locally
    logged transfer is proof of at least one.
    """
    logged = 0
    try:
        # A flat list of {team_id, event, element_out, element_in, ...}, which
        # is what `transfers._add_pending_local` writes.
        from scripts.common.classic_squad import load_pending_file
        logged = sum(1 for e in load_pending_file() or []
                     if int(e.get("team_id") or 0) == int(team_id)
                     and int(e.get("event") or 0) == int(target_gw))
    except Exception as exc:
        _logger.warning("Could not read the pending transfer log: %s", exc)

    # An authenticated squad knows the real count for the current gameweek.
    if getattr(resolution, "source", "") == SOURCE_MY_TEAM:
        made = (resolution.entry_history or {}).get("event_transfers")
        if made is not None:
            return int(made) + logged

    try:
        from scripts.common.fpl_classic_api import get_classic_transfers
        rows = get_classic_transfers(team_id) or []
        from_log = sum(1 for t in rows
                       if int(t.get("event") or 0) == int(target_gw))
    except Exception as exc:
        _logger.warning("Could not read the transfer log: %s", exc)
        return logged or None

    if from_log or logged:
        return from_log + logged
    # Nothing recorded anywhere, and we cannot prove the absence.
    return None


def _chip_values(squad_df, pool, target_gw):
    """What Bench Boost and Triple Captain are worth *this* week.

    Both are solver-free -- the bench four, and one more copy of the captain --
    so they are cheap enough to render unprompted. Free Hit and Wildcard are
    ILPs and belong to Chip Strategy.
    """
    points = _gw_points(pool, target_gw)
    if points is None:
        return {}
    try:
        matrix = chips.squad_points_matrix(squad_df, points, [int(target_gw)])
        bb = chips.bench_boost_value(squad_df, matrix).get(int(target_gw))
        tc = chips.triple_captain_value(squad_df, matrix).get(int(target_gw))
    except (KeyError, ValueError) as exc:
        _logger.warning("Could not value this week's chips: %s", exc)
        return {}
    return {k: float(v) for k, v in
            ((chips.CHIP_BBOOST, bb), (chips.CHIP_3XC, tc)) if v is not None}


def _price_pressure(bootstrap, squad_df):
    try:
        codes = (squad_df["Player_Code"].dropna().tolist()
                 if squad_df is not None and "Player_Code" in squad_df.columns else [])
        return squad_price_pressure(build_price_data(bootstrap), codes)
    except Exception as exc:
        _logger.warning("Could not read price pressure: %s", exc)
        return None


def _opponent(league_id, team_id, bootstrap, current_gw, target_gw, pool, squad_df):
    """``(kind, data)`` for this gameweek's head-to-head fixture.

    ``kind`` is ``"ok"`` with ``(name, mine, theirs, win probability)``,
    ``"no_fixture"`` where the schedule simply has no pairing for this gameweek
    (an H2H league can have a blank week, and there is then genuinely no
    opponent to report), or ``"unknown"`` where something we needed could not be
    read. The last two must not be conflated: a question that does not exist is
    an absent row, and one we failed to answer is a grey one.

    Every input is already paid for: Home fetches the H2H match payload for its
    streaks and season odds, so the schedule, the opponent's name and the
    league's sigma are cache hits. Only the rival's squad is a new resolve.

    Their lineup is the *optimal* XI rather than their stored multipliers, which
    for an unplayed gameweek are last deadline's choices -- reading those would
    project every rival as benching their best player.
    """
    try:
        schedule = h2h.h2h_schedule(league_id, team_id, [int(target_gw)])
        opponent_id = schedule.get(int(target_gw))
        if not opponent_id:
            return "no_fixture", None
        points = _gw_points(pool, target_gw)
        if points is None:
            return "unknown", None
        mine = h2h.lineup_projection(
            squad_df, chips.squad_points_matrix(squad_df, points, [int(target_gw)]),
            int(target_gw))
        opp_squad = h2h.opponent_squad(opponent_id, bootstrap, current_gw, pool)
        theirs = None
        if opp_squad is not None and not opp_squad.empty:
            theirs = h2h.lineup_projection(
                opp_squad,
                chips.squad_points_matrix(opp_squad, points, [int(target_gw)]),
                int(target_gw))
        name = h2h.h2h_entry_names(league_id).get(int(opponent_id), "your opponent")
        prob = None
        if mine is not None and theirs is not None:
            sigma, _ = h2h.h2h_sigma(league_id)
            prob = chips.win_probability(mine - theirs, sigma)
        if mine is None or theirs is None:
            return "unknown", None
        return "ok", (name, mine, theirs, prob)
    except Exception as exc:
        _logger.warning("Could not resolve the head-to-head fixture: %s", exc)
        return "unknown", None


def _solve_plan(pool, squad_df, resolution, free_transfers):
    """The best transfer plan reachable with the transfers in hand.

    Built from the **whole** pool. `solve_squad_ilp` raises when an owned player
    is missing from it, because the change constraint counts kept players out of
    fifteen -- so a narrowed pool silently grants a free transfer.
    """
    try:
        from scripts.classic.transfers import build_plan_pool, build_transfer_plan
        bank = int((resolution.entry_history or {}).get("bank") or 0)
        plan_pool = build_plan_pool(pool, squad_df, 0.4, 0.6)
        if plan_pool is None or plan_pool.empty:
            return None
        return build_transfer_plan(plan_pool, squad_df, bank,
                                   free_transfers=int(free_transfers or 0))
    except Exception as exc:
        _logger.warning("Could not solve a transfer plan: %s", exc)
        return None


def _leg_name(leg, side: str) -> str:
    """The common name for one side of a transfer leg.

    ``<side>_player`` is the full legal name the solver matched on, so the row
    is preferred -- and it is a Series, which must be tested against None rather
    than for truth: `Series or {}` raises.
    """
    fallback = str(leg.get("%s_player" % side) or "")
    row = leg.get("%s_row" % side)
    return display_of(row, fallback) if row is not None else fallback


def _plan_summary(plan) -> dict:
    """The plan reduced to primitives, so no frame enters the cache.

    `build_transfer_plan` returns squad frames and per-leg Series; caching those
    would put a few hundred rows behind every key, and the strip renders none of
    them. Shapes are its own: ``frontier`` is a **list** of entries, and leg
    prices are already in millions.
    """
    if not plan:
        return {}
    rows, seen = [], set()
    for entry in plan.get("frontier") or []:
        changes = int(entry.get("n_changes") or 0)
        if changes in seen:
            continue
        seen.add(changes)
        rows.append({"Transfers": changes,
                     "Hit": (-4 * int(entry.get("hits") or 0)) or 0,
                     "Gain (pts)": round(float(entry.get("gain") or 0.0), 2)})
    return {
        "error": plan.get("error"),
        "hold": bool(plan.get("hold")),
        "gain_net": (None if plan.get("gain_net") is None
                     else float(plan["gain_net"])),
        "n_changes": plan.get("n_changes"),
        "horizon": plan.get("horizon_gws") or plan.get("horizon"),
        "frontier": rows,
        "legs": [{"out": _leg_name(leg, "out"), "in": _leg_name(leg, "in"),
                  "net": float(leg.get("net_cost") or 0.0)}
                 for leg in plan.get("legs") or []],
    }


# ------------------------------------------------------------------
# Assembly, cached as one object so a Home rerun costs nothing
# ------------------------------------------------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def _assemble(team_id: int, current_gw: int, league_id, is_h2h: bool,
              confirmed_signature: str = ""):
    """``(items, plan_summary, squad_size, xi_present, signature)``.

    Everything the strip renders, resolved once. Keyed on primitives only -- the
    heavy input is `blended_pool`, which owns its own 300s cache and is shared
    with the Projections Hub, Chip Strategy and the Rival Tracker, so the cold
    cost is paid once across the app rather than once per page.

    ``confirmed_signature`` is in the key rather than read inside, so pressing
    "I have set my XI" invalidates this entry instead of being swallowed by it.
    """
    bootstrap = get_classic_bootstrap_static() or {}
    history = get_classic_team_history(team_id) or {}

    resolution = resolve_classic_squad(team_id, bootstrap, current_gw)
    target_gw = int(getattr(resolution, "target_gw", None) or current_gw)
    started = _is_gameweek_started(target_gw, bootstrap)

    pool = blended_pool(current_gw, config.ROTOWIRE_URL)
    squad_df = squad_frame(getattr(resolution, "picks", []) or [], pool)

    status = chips.chip_status_from_bootstrap(bootstrap, history, current_gw)
    deadlines = gameweek_deadlines()
    counts, window = _fixture_counts(target_gw)
    team_names = {int(t["id"]): t.get("short_name", str(t["id"]))
                  for t in bootstrap.get("teams", []) if t.get("id") is not None}

    ft_info = _free_transfers(history, resolution, current_gw, status, team_id)
    bank = (resolution.entry_history or {}).get("bank")
    bank = (float(bank) / 10.0) if bank is not None else None

    # Eager, because the gain *is* the alert. 0.27s measured over K=0..3.
    plan = (None if squad_df.empty else
            _solve_plan(pool, squad_df, resolution, (ft_info or {}).get("count")))

    signature = ("" if squad_df.empty else
                 lineup_confirm.squad_signature(squad_df["element_id"].tolist()))
    confirmed = bool(signature) and signature == confirmed_signature

    optimal_total, captain, _ = _lineup(squad_df)
    set_total, set_captain, swap = _set_lineup(squad_df)
    stale = bool(getattr(resolution, "is_stale", False))

    opponent_item = None
    if is_h2h and league_id and not squad_df.empty:
        kind, data = _opponent(league_id, team_id, bootstrap, current_gw,
                               target_gw, pool, squad_df)
        if kind == "ok":
            opponent_item = week_check.build_opponent_item(*data, stale=True)
        elif kind == "unknown":
            # The question exists and we failed to answer it, so it gets a grey
            # row. "no_fixture" is the other case and gets none.
            opponent_item = week_check.build_opponent_item()

    items = week_check.build_week_checklist(
        deadline=week_check.build_deadline_item(deadlines, target_gw, started),
        opponent=opponent_item,
        squad=week_check.build_squad_item(resolution),
        availability=week_check.build_availability_item(
            _availability_flags(squad_df, deadlines)),
        fixtures=week_check.build_fixtures_item(
            counts,
            (squad_df["Team_ID"].dropna().tolist()
             if "Team_ID" in squad_df.columns else []),
            window, target_gw, team_names),
        lineup=week_check.build_lineup_item(
            optimal_total, set_total, swap, captain, set_captain,
            confirmed=confirmed, stale=stale),
        transfers=week_check.build_transfers_item(
            ft_info, bank, plan,
            made_this_week=_transfers_made(team_id, target_gw, resolution)),
        chips_item=week_check.build_chips_item(
            status, _chip_values(squad_df, pool, target_gw), chips.CHIP_LABELS),
        prices=week_check.build_prices_item(_price_pressure(bootstrap, squad_df)),
    )
    return (items, _plan_summary(plan), len(squad_df),
            optimal_total is not None, signature)


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def strip_height(n_rows: int, caveat: bool, buttons: bool) -> int:
    """How tall the strip will render, so the card beside it can match."""
    return (STRIP_CHROME_PX + STRIP_ROW_PX * max(1, int(n_rows))
            + (STRIP_CAVEAT_PX if caveat else 0)
            + (STRIP_BUTTON_PX if buttons else 0))


def _alert_row(item) -> str:
    """One line of the alert list: status, what it is, and the number.

    This replaces the pill strip *and* the "Needs action" tab, which said the
    same thing twice -- the feedback that prompted the rewrite. One list, each
    row carrying its own verdict, so nothing is repeated and nothing needs
    opening to be read.
    """
    icon, colour = _STATUS_STYLE.get(item.status, ("•", "#9ca3af"))
    label = _PILL_TEXT.get(item.key, item.title)
    return compact_html(f'''
        <div style="display:flex;align-items:baseline;gap:8px;padding:5px 0;
                    border-top:1px solid #2a2a3e;">
          <span style="font-size:0.9em;">{icon}</span>
          <span style="color:{colour};font-size:0.72em;font-weight:700;
                       text-transform:uppercase;letter-spacing:0.5px;
                       min-width:78px;">{label}</span>
          <span style="color:#e0e0e0;font-size:0.9em;line-height:1.35;">
            {item.verdict}</span>
        </div>
    ''')


def _render_strip(items, signature, team_id, target_gw, authed):
    """The strip: a header, the alert list, and the buttons that act on it.

    Returns the height it occupies, which Home uses to size the league card
    beside it -- the two sitting at different heights is the thing this returns
    for.
    """
    deadline = next((i for i in items if i.key == "deadline"), None)
    stamp = ""
    when = (deadline.evidence or {}).get("deadline") if deadline else None
    if when:
        try:
            moment = pd.Timestamp(when).to_pydatetime()
            stamp = "GW%s &middot; %s &middot; %s" % (
                (deadline.evidence or {}).get("gameweek", ""),
                format_deadline_stamp(moment), format_time_until(moment))
        except (ValueError, TypeError):
            stamp = ""

    flagged = week_check.attention_items(items) + [
        i for i in items if i.status == week_check.STATUS_UNKNOWN]
    squad_row = next((i for i in items if i.key == "squad"), None)
    caveat = ""
    if squad_row is not None and (squad_row.evidence or {}).get("stale"):
        # The staleness travels with the numbers, in one grey line, rather than
        # as a row that is permanently yellow and cannot be acted on.
        caveat = ("Reading your GW%s squad — FPL publishes no pending squad "
                  "without a connected account."
                  % (squad_row.evidence or {}).get("source_gw"))

    rows = "".join(_alert_row(i) for i in flagged) or compact_html('''
        <div style="padding:8px 0 2px;border-top:1px solid #2a2a3e;
                    color:#00ff87;font-size:0.92em;font-weight:600;">
          ✅ Nothing needs doing before the deadline.
        </div>
    ''')

    buttons = [i for i in flagged if i.page]
    lineup_row = next((i for i in items if i.key == "lineup"), None)
    has_buttons = bool(buttons) or not authed or (
        lineup_row is not None and lineup_row.status == week_check.STATUS_ACT
        and bool(signature))
    height = strip_height(len(flagged), bool(caveat), has_buttons)

    st.markdown(compact_html(f'''
        <div style="border:1px solid #333;border-radius:12px;padding:14px 16px;
                    box-sizing:border-box;min-height:{height - (STRIP_BUTTON_PX if has_buttons else 0)}px;
                    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                    color:#e0e0e0;">
          <div style="display:flex;justify-content:space-between;
                      align-items:baseline;margin-bottom:4px;">
            <span style="color:#00ff87;font-size:0.8em;font-weight:700;
                         text-transform:uppercase;letter-spacing:1px;">
              Your week</span>
            <span style="color:#9ca3af;font-size:0.78em;">{stamp}</span>
          </div>
          {rows}
          {f'<div style="color:#6b7280;font-size:0.74em;margin-top:9px;">{caveat}</div>' if caveat else ""}
        </div>
    '''), unsafe_allow_html=True)

    _render_actions(flagged, items, signature, team_id, target_gw, authed)
    return height


def _render_actions(flagged, items, signature, team_id, target_gw, authed):
    """One button per flagged row, plus the two that only exist here.

    `main.py` applies `_pending_nav_section` / `_pending_nav_page` before its
    sidebar radios are instantiated, so a button here lands on the page that
    owns the row -- Home's own "Go to League Setup" link is the precedent.
    """
    buttons = [i for i in flagged if i.page]
    lineup_row = next((i for i in items if i.key == "lineup"), None)
    can_confirm = (lineup_row is not None
                   and lineup_row.status == week_check.STATUS_ACT
                   and signature)

    if not buttons and not can_confirm and authed:
        return

    cols = st.columns(max(1, len(buttons) + (1 if can_confirm else 0)
                          + (0 if authed else 1)))
    slot = 0
    if can_confirm:
        with cols[slot]:
            if st.button("✓ I've set my XI", key="my_week_confirm_xi",
                         use_container_width=True,
                         help="Silences the lineup alert for this gameweek. It "
                              "comes back if you transfer, since the eleven "
                              "would need setting again."):
                lineup_confirm.confirm_lineup(team_id, target_gw, signature)
                st.rerun()
        slot += 1
    for item in buttons:
        with cols[slot]:
            if st.button(item.page.strip(), key="my_week_open_%s" % item.key,
                         use_container_width=True,
                         help="Go to %s" % item.page.strip()):
                st.session_state["_pending_nav_section"] = item.section
                st.session_state["_pending_nav_page"] = item.page
                st.rerun()
        slot += 1
    if not authed:
        with cols[slot]:
            if st.button("Connect FPL account", key="my_week_connect",
                         use_container_width=True,
                         help="Without it, FPL publishes no pre-deadline squad, "
                              "so every row above reads last deadline's team."):
                st.session_state["_pending_nav_section"] = week_check.SECTION_FPL
                st.session_state["_pending_nav_page"] = "\U0001f194  League Setup"
                st.rerun()


def _item_card(item) -> str:
    """One checklist row, for the full-checklist tab.

    Through `compact_html` because `detail` is optional on every row: a blank
    line ends an HTML block in Markdown, so an empty fragment on its own line
    puts the closing tag on screen as literal text.
    """
    icon, colour = _STATUS_STYLE.get(item.status, ("•", "#9ca3af"))
    detail = (f'<div style="color:#9ca3af;font-size:0.84em;margin-top:5px;">'
              f'{item.detail}</div>') if item.detail else ""
    return compact_html(f'''
        <div style="border:1px solid #333;border-left:4px solid {colour};
                    border-radius:8px;padding:10px 13px;margin-bottom:7px;
                    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                    color:#e0e0e0;">
          <div style="display:flex;align-items:baseline;gap:9px;">
            <span>{icon}</span>
            <span style="color:#9ca3af;font-size:0.74em;text-transform:uppercase;
                         letter-spacing:0.6px;min-width:104px;">{item.title}</span>
            <span style="font-size:0.95em;">{item.verdict}</span>
          </div>
          {detail}
        </div>
    ''')


def _render_plan(summary):
    """What the transfers in hand are worth, and the legs that realise it."""
    if not summary:
        st.caption("No transfer plan could be solved — the squad or the player "
                   "pool is incomplete. Transfer Suggestions has the full picture.")
        return
    if summary.get("error"):
        st.caption("Your squad could not be reproduced by the optimizer, so no "
                   "plan is offered. This usually means a selling price is "
                   "missing — connect your account on League Setup.")
        return
    if summary.get("hold"):
        st.success("Holding is the best move: no combination of your transfers "
                   "gains enough to be worth making.")

    if summary.get("frontier"):
        render_styled_table(pd.DataFrame(summary["frontier"]),
                            title="What each number of transfers buys",
                            col_formats={"Gain (pts)": "{:+.1f}", "Hit": "{:d}"},
                            positive_color_cols=["Gain (pts)"])
    for leg in summary.get("legs") or []:
        money = ("frees £%.1fm" % -leg["net"]) if leg["net"] < 0 \
            else ("costs £%.1fm" % leg["net"])
        st.markdown("- **%s** → **%s** (%s)" % (leg["out"], leg["in"], money))
    if summary.get("horizon"):
        st.caption("Scored over %.1f gameweeks, which is how a −4 hit is priced. "
                   "Transfer Suggestions renders the legs with the sanity veto, "
                   "price trends and ownership." % float(summary["horizon"]))


def render_my_week_section(league_id=None, scoring_type: str = "",
                           current_gw=None):
    """Render the strip, and return the height it took.

    Called by Classic Home, in the column beside the league selector and card;
    the height lets that card match it rather than ending short. ``None`` means
    nothing was rendered, and the card should size itself.

    ``scoring_type`` is Home's own label for the selected league; the
    head-to-head row renders only for ``"H2H"``, since a total-points league has
    no opponent and an *unknown* row there would claim we failed to answer a
    question that does not exist.
    """
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        st.caption("Set your own Classic team on the 🆔 League Setup page to see "
                   "your week checked here — fitness, transfers, chips and prices.")
        if st.button("🆔 Go to League Setup", key="my_week_setup_link"):
            st.session_state["_pending_nav_section"] = week_check.SECTION_FPL
            st.session_state["_pending_nav_page"] = "🆔  League Setup"
            st.rerun()
        return None

    gw = int(current_gw or config.CURRENT_GAMEWEEK or 1)
    is_h2h = str(scoring_type).upper() == "H2H"
    team_id = int(team_id)
    try:
        authed = has_auth()
    except Exception:
        authed = False

    # Read the confirmation *before* assembling, and pass it into the cache key:
    # pressing the button must invalidate the entry rather than be swallowed by
    # it. The signature it is checked against comes back from the assembly, so
    # the first pass looks the stored one up by gameweek alone.
    stored = _confirmed_signature(team_id, gw)
    try:
        with st.spinner("Checking your week…"):
            items, plan, squad_size, has_xi, signature = _assemble(
                team_id, gw, league_id, is_h2h, stored)
    except Exception as exc:
        # The strip must never take Home down: the standings below it are the
        # reason most visits happen.
        _logger.warning("Could not assemble the week checklist: %s", exc,
                        exc_info=True)
        st.caption("Could not check your week just now. The league below is "
                   "unaffected.")
        return None

    target_gw = (items[0].evidence or {}).get("gameweek") if items else gw
    height = _render_strip(items, signature, team_id, int(target_gw or gw),
                           authed)

    with st.expander("Your week in full", expanded=False):
        tab_all, tab_plan = st.tabs(["All checks", "Transfer plan"])
        with tab_all:
            for item in items:
                st.markdown(_item_card(item), unsafe_allow_html=True)
        with tab_plan:
            _render_plan(plan)

    # Logged rather than raised: a suspect row with a warning in the log beats a
    # Home page that will not render. The check that matters is an all-clear with
    # no evidence behind it -- a checklist's one unique way of being wrong.
    for issue in check_week_plan(items, squad_size=squad_size,
                                 xi_size=(11 if has_xi else 0)):
        _logger.log(40 if issue.severity == "error" else 20, "my week: %s", issue)

    return height


def _confirmed_signature(team_id, gameweek) -> str:
    """The signature this manager has confirmed for this gameweek, if any.

    Read by gameweek rather than compared to a known signature, because the
    squad's own signature is not known until the assembly has run -- and the
    assembly needs this value in its cache key. `is_confirmed` does the real
    comparison inside the assembly.
    """
    try:
        entry = lineup_confirm._load().get(
            lineup_confirm._key(team_id, gameweek)) or {}
        return str(entry.get("signature") or "")
    except Exception:
        return ""
