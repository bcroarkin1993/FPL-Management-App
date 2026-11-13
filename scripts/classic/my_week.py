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
from scripts.common import chips, h2h, week_check
from scripts.common.analytics import gameweek_deadlines
from scripts.common.classic_squad import SOURCE_MY_TEAM, resolve_classic_squad
from scripts.common.data_validation import check_week_plan
from scripts.common.display_names import display_of
from scripts.common.error_helpers import get_logger
from scripts.common.fixture_helpers import club_fixture_counts
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
    "lineup": "Captain",
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
    """``(captain-doubled XI total, captain name, his projection)``.

    The *optimal* XI and the optimal armband, not the stored multipliers: for a
    gameweek whose deadline has not passed those are last deadline's choices, so
    reading them would project a manager as benching his best player.
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


def _stored_captain(squad_df):
    """Whose armband FPL currently records, if the frame says."""
    if squad_df is None or squad_df.empty or "is_captain" not in squad_df.columns:
        return None
    marked = squad_df[squad_df["is_captain"].astype(bool)]
    if marked.empty:
        return None
    return display_of(marked.iloc[0]) or None


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
def _assemble(team_id: int, current_gw: int, league_id, is_h2h: bool):
    """``(items, plan_summary, squad_size, xi_present)``.

    Everything the strip renders, resolved once. Keyed on primitives only -- the
    heavy input is `blended_pool`, which owns its own 300s cache and is shared
    with the Projections Hub, Chip Strategy and the Rival Tracker, so the 3.7s
    cold cost is paid once across the app rather than once per page.
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

    xi_total, captain, captain_pts = _lineup(squad_df)

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
            xi_total, captain, captain_pts, _stored_captain(squad_df),
            comparable=(getattr(resolution, "source", "") == SOURCE_MY_TEAM
                        and not started)),
        transfers=week_check.build_transfers_item(ft_info, bank, plan),
        chips_item=week_check.build_chips_item(
            status, _chip_values(squad_df, pool, target_gw), chips.CHIP_LABELS),
        prices=week_check.build_prices_item(_price_pressure(bootstrap, squad_df)),
    )
    return items, _plan_summary(plan), len(squad_df), xi_total is not None


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def _pill(item) -> str:
    icon, colour = _STATUS_STYLE.get(item.status, ("•", "#9ca3af"))
    label = _PILL_TEXT.get(item.key, item.title)
    if item.status == week_check.STATUS_UNKNOWN:
        label += " not checked"
    return (f'<span style="background:{colour}22;color:{colour};'
            f'border:1px solid {colour}66;padding:4px 11px;border-radius:12px;'
            f'font-size:0.82em;font-weight:600;margin:3px 6px 3px 0;'
            f'display:inline-block;">{icon} {label}</span>')


def _render_alert_bar(items):
    """The strip itself: the deadline, then one pill per thing to look at.

    Flagged rows first, then anything we could not check -- which is a pill in
    its own right, because a bar that silently shortens on an outage is the
    defaulted all-clear this whole section exists to avoid.
    """
    deadline = next((i for i in items if i.key == "deadline"), None)
    when = (deadline.evidence or {}).get("deadline") if deadline else None
    head = "Your week"
    if when:
        try:
            moment = pd.Timestamp(when).to_pydatetime()
            head = "Your week &middot; GW%s &middot; %s (%s)" % (
                (deadline.evidence or {}).get("gameweek", ""),
                format_deadline_stamp(moment), format_time_until(moment))
        except (ValueError, TypeError):
            head = "Your week"

    flagged = week_check.attention_items(items)
    unknown = [i for i in items if i.status == week_check.STATUS_UNKNOWN]
    pills = "".join(_pill(i) for i in flagged + unknown)

    if not pills:
        body = ('<span style="color:#00ff87;font-weight:600;">✅ Nothing needs '
                'doing</span> <span style="color:#9ca3af;">— all %d checks '
                'passed.</span>' % len(items))
    else:
        body = pills

    st.markdown(compact_html(f'''
        <div style="border:1px solid #333;border-radius:10px;padding:12px 14px;
                    margin-bottom:10px;color:#e0e0e0;
                    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);">
          <div style="color:#9ca3af;font-size:0.74em;text-transform:uppercase;
                      letter-spacing:0.8px;margin-bottom:8px;">{head}</div>
          <div>{body}</div>
        </div>
    '''), unsafe_allow_html=True)


def _item_card(item) -> str:
    """One checklist row.

    Through `compact_html` because `detail` is optional on every row: a blank
    line ends an HTML block in Markdown, so an empty fragment on its own line
    puts the closing tag on screen as literal text.
    """
    icon, colour = _STATUS_STYLE.get(item.status, ("•", "#9ca3af"))
    detail = (f'<div style="color:#9ca3af;font-size:0.84em;margin-top:5px;">'
              f'{item.detail}</div>') if item.detail else ""
    return compact_html(f'''
        <div style="border:1px solid #333;border-left:4px solid {colour};
                    border-radius:8px;padding:10px 13px;
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


def _render_rows(items, key_prefix: str):
    """The rows, each with a one-click jump to the page that owns it.

    The button drives `main.py`'s `_pending_nav_section` / `_pending_nav_page`,
    which are applied before the sidebar radios are instantiated -- the same
    mechanism as Home's "Go to League Setup" link. The deadline row carries no
    page, since the strip is already on Home.
    """
    for item in items:
        card, nav = st.columns([5, 1])
        with card:
            st.markdown(_item_card(item), unsafe_allow_html=True)
        with nav:
            if not item.page:
                continue
            if st.button("Open →", key="%s_open_%s" % (key_prefix, item.key),
                         help="Go to %s" % item.page.strip()):
                st.session_state["_pending_nav_section"] = item.section
                st.session_state["_pending_nav_page"] = item.page
                st.rerun()


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
    """Render the alert strip. Called by Classic Home under the league header.

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
        return

    gw = int(current_gw or config.CURRENT_GAMEWEEK or 1)
    is_h2h = str(scoring_type).upper() == "H2H"
    try:
        with st.spinner("Checking your week…"):
            items, plan, squad_size, has_xi = _assemble(
                int(team_id), gw, league_id, is_h2h)
    except Exception as exc:
        # The strip must never take Home down: the standings below it are the
        # reason most visits happen.
        _logger.warning("Could not assemble the week checklist: %s", exc,
                        exc_info=True)
        st.caption("Could not check your week just now. The league below is "
                   "unaffected.")
        return

    _render_alert_bar(items)

    flagged = week_check.attention_items(items) + [
        i for i in items if i.status == week_check.STATUS_UNKNOWN]
    tab_action, tab_all, tab_plan = st.tabs([
        "Needs action (%d)" % len(flagged), "Full checklist",
        "Transfer plan"])

    with tab_action:
        if flagged:
            _render_rows(flagged, "my_week_action")
        else:
            st.caption("Nothing to action this week. The full checklist is in "
                       "the next tab.")
    with tab_all:
        _render_rows(items, "my_week_all")
    with tab_plan:
        _render_plan(plan)

    # Logged rather than raised: a suspect row with a warning in the log beats a
    # Home page that will not render. The check that matters is an all-clear with
    # no evidence behind it -- a checklist's one unique way of being wrong.
    for issue in check_week_plan(items, squad_size=squad_size,
                                 xi_size=(11 if has_xi else 0)):
        _logger.log(40 if issue.severity == "error" else 20, "my week: %s", issue)
