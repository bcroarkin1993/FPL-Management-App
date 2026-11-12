"""My Week — everything that needs deciding before the deadline, on one page.

The app answered every pre-deadline question and answered each one somewhere
else: the deadline and squad on Home, fitness on Availability, the XI and
captain on Fixture Projections, free transfers on Transfer Suggestions, the chip
call on Chip Strategy, prices on Price Changes. Six pages, and no way to find out
that five of them had nothing to say except by opening all six.

This page runs that round and reports only the verdict: one row per question,
with the number that settles it and a pointer to the page that owns the detail.
It computes no projection of its own -- every figure here is the one the engine
already produced, so a number that looks wrong on this page is wrong on the page
it came from too.

Two things it deliberately does not do:

* **Guess.** A question whose inputs are missing renders as *unknown*, never as
  "nothing to do". Those two read identically on a card and are opposite claims,
  which is the whole reason the item model carries ``evidence`` and
  ``check_week_plan`` fires on an all-clear that recorded none.
* **Solve anything twice.** The transfer plan is an ILP frontier and sits behind
  a button; Free Hit and Wildcard are eight more solves and are left to Chip
  Strategy entirely. Everything else is one cached pool, one bootstrap and one
  squad resolve.

The checklist logic is pure and lives in ``scripts/common/week_check.py``; this
module is fetching and rendering only.
"""

import pandas as pd
import streamlit as st

import config
from scripts.common import chips, week_check
from scripts.common.analytics import gameweek_deadlines
from scripts.common.classic_squad import (
    SOURCE_MY_TEAM,
    resolve_classic_squad,
)
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
#: the horizon FFP publishes and the one Chip Strategy uses, so the two pages
#: cannot disagree about what is "coming up".
FIXTURE_HORIZON_WEEKS = 6

#: Status -> (icon, colour). Unknown is grey rather than green: a question we
#: could not answer must not read as one answered "fine".
_STATUS_STYLE = {
    week_check.STATUS_OK: ("✅", "#00ff87"),
    week_check.STATUS_UNKNOWN: ("❔", "#9ca3af"),
    week_check.STATUS_WARN: ("⚠️", "#ffa502"),
    week_check.STATUS_ACT: ("❗", "#ff4757"),
}


# ------------------------------------------------------------------
# Rendering
# ------------------------------------------------------------------
def _item_card(item: week_check.WeekItem) -> str:
    """One checklist row as a card.

    Through `compact_html` because `detail` is optional on every row: a blank
    line ends an HTML block in Markdown, so an empty fragment on its own line
    puts the closing tag on screen as literal text.
    """
    icon, colour = _STATUS_STYLE.get(item.status, ("•", "#9ca3af"))
    detail = (f'<div style="color:#9ca3af;font-size:0.86em;margin-top:5px;">'
              f'{item.detail}</div>') if item.detail else ""
    page = (f'<div style="color:#6b7280;font-size:0.78em;margin-top:6px;">'
            f'More on <strong>{item.page}</strong></div>') if item.page else ""
    return compact_html(f'''
        <div style="border:1px solid #333;border-left:4px solid {colour};
                    border-radius:8px;padding:12px 14px;margin-bottom:9px;
                    background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);
                    color:#e0e0e0;">
          <div style="display:flex;align-items:baseline;gap:9px;">
            <span style="font-size:1.0em;">{icon}</span>
            <span style="color:#9ca3af;font-size:0.76em;text-transform:uppercase;
                         letter-spacing:0.6px;min-width:112px;">{item.title}</span>
            <span style="color:#e0e0e0;font-size:0.97em;">{item.verdict}</span>
          </div>
          {detail}
          {page}
        </div>
    ''')


def _render_headline(items, deadline_item):
    """One line saying whether anything needs doing, and by when."""
    flagged = week_check.attention_items(items)
    unknown = [i for i in items if i.status == week_check.STATUS_UNKNOWN]
    when = (deadline_item.evidence or {}).get("deadline")
    stamp = ""
    if when:
        try:
            moment = pd.Timestamp(when).to_pydatetime()
            stamp = "%s (%s)" % (format_deadline_stamp(moment),
                                 format_time_until(moment))
        except (ValueError, TypeError):
            stamp = ""

    if flagged:
        titles = ", ".join(i.title.lower() for i in flagged)
        text = "**%d thing%s to look at** before the deadline: %s." % (
            len(flagged), "" if len(flagged) == 1 else "s", titles)
        (st.error if any(i.status == week_check.STATUS_ACT for i in flagged)
         else st.warning)(text + (f" Deadline {stamp}." if stamp else ""))
    elif unknown:
        st.info("Nothing needs attention, but %d check%s could not be made — "
                "see the grey rows." % (len(unknown),
                                        "" if len(unknown) == 1 else "s"))
    else:
        st.success("Nothing needs doing this week." +
                   (f" Deadline {stamp}." if stamp else ""))


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
    if pool is None or pool.empty or "Player_ID" not in pool.columns:
        return {}
    try:
        points = (pool.set_index("Player_ID")[["Proj"]]
                  .rename(columns={"Proj": int(target_gw)}))
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
    is preferred -- and it is a Series, which must be tested against None
    rather than for truth: `Series or {}` raises.
    """
    fallback = str(leg.get("%s_player" % side) or "")
    row = leg.get("%s_row" % side)
    return display_of(row, fallback) if row is not None else fallback


def _render_plan(plan):
    """The frontier, as the one line that answers "is it worth it".

    Shapes come from `build_transfer_plan`: ``frontier`` is a **list** of
    ``{k, gain, n_changes, hits}`` entries (several k can reach the same number
    of changes, so they are de-duplicated the way Transfer Suggestions does it),
    and leg prices are already in millions.
    """
    if not plan:
        st.caption("No plan could be solved — the squad or the player pool is "
                   "incomplete. Transfer Suggestions has the full picture.")
        return
    if plan.get("error"):
        st.caption("Your squad could not be reproduced by the optimizer, so no "
                   "plan is offered. This usually means a selling price is "
                   "missing — connect your account on League Setup.")
        return
    if plan.get("hold"):
        st.success("Holding is the best move: no combination of your transfers "
                   "gains enough to be worth making.")

    rows, seen = [], set()
    for entry in plan.get("frontier") or []:
        changes = int(entry.get("n_changes") or 0)
        if changes in seen:
            continue
        seen.add(changes)
        rows.append({"Transfers": changes,
                     "Hit": (-4 * int(entry.get("hits") or 0)) or 0,
                     "Gain (pts)": round(float(entry.get("gain") or 0.0), 2)})
    if rows:
        render_styled_table(pd.DataFrame(rows),
                            title="What each number of transfers buys",
                            col_formats={"Gain (pts)": "{:+.1f}", "Hit": "{:d}"},
                            positive_color_cols=["Gain (pts)"])

    for leg in plan.get("legs") or []:
        net = float(leg.get("net_cost") or 0.0)
        out_name, in_name = _leg_name(leg, "out"), _leg_name(leg, "in")
        money = ("frees £%.1fm" % -net) if net < 0 else ("costs £%.1fm" % net)
        st.markdown("- **%s** → **%s** (%s)" % (out_name, in_name, money))

    horizon = plan.get("horizon_gws") or plan.get("horizon")
    if horizon:
        st.caption("Scored over %.1f gameweeks, which is how a −4 hit is priced. "
                   "Transfer Suggestions renders the legs with the sanity veto, "
                   "price trends and ownership." % float(horizon))


# ------------------------------------------------------------------
# The page
# ------------------------------------------------------------------
def show_my_week_page():
    """Render the My Week checklist."""
    st.title("🗓️ My Week")
    st.caption(
        "One pass over everything that needs deciding before the deadline. "
        "Every number here comes from the page that owns it — this is the round "
        "trip, not a second opinion."
    )

    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        st.warning(
            "No Classic team is configured. Set one on the **🆔 League Setup** "
            "page (under FPL App Home) to see your own week checked."
        )
        return

    current_gw = int(config.CURRENT_GAMEWEEK or 1)
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

    # The ILP lane is latched in session state and solved *before* the cards, so
    # the Transfers row and the plan below it cannot disagree about whether a
    # plan exists. The Rival Tracker's load button is the precedent.
    plan_key = "my_week_plan_%s_%s" % (team_id, target_gw)
    plan = None
    if st.session_state.get(plan_key) and not squad_df.empty:
        with st.spinner("Solving your transfer options…"):
            plan = _solve_plan(pool, squad_df, resolution,
                               (ft_info or {}).get("count"))

    xi_total, captain, captain_pts = _lineup(squad_df)
    items = week_check.build_week_checklist(
        deadline=week_check.build_deadline_item(deadlines, target_gw, started),
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

    _render_headline(items, items[0])
    for item in items:
        st.markdown(_item_card(item), unsafe_allow_html=True)

    st.markdown("### Is a transfer worth making?")
    st.caption(
        "Solving every number of transfers against the whole player pool is a "
        "handful of ILP solves, so it is computed on request."
    )
    if st.button("Value my transfer options", key="my_week_plan",
                 type="secondary" if plan else "primary"):
        st.session_state[plan_key] = True
        st.rerun()
    if st.session_state.get(plan_key):
        _render_plan(plan)

    st.caption(
        "Free Hit and Wildcard are eight more squad solves and live on "
        "**🎯 Chip Strategy**, which also prices every week in the window "
        "rather than only this one."
    )

    # Logged rather than raised: a suspect row with a warning in the log beats a
    # page that will not render. The check that matters is an all-clear with no
    # evidence behind it -- a checklist's one unique way of being wrong.
    for issue in check_week_plan(items, squad_size=len(squad_df),
                                 xi_size=(11 if xi_total is not None else 0)):
        _logger.log(40 if issue.severity == "error" else 20, "my week: %s", issue)
