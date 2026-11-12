# scripts/fpl/price_changes.py
"""Price Changes page — who moves tonight, your squad's exposure, and why.

**FPL publishes its own price-change prediction and this page ignored it.** The
bootstrap carries five fields nothing in the app read:

===============================  ===========================================
``price_change_percent``         progress to the next change, 100 = threshold
``price_change_projections``     FPL's forward view, per day offset, with a
                                 1-5 ``likelihood``
``price_change_hourly_rate``     net transfers an hour, so a change has an ETA
``price_change_locked_until``    a player may change only once a day
``price_change_calibrating``     the model is still warming up
===============================  ===========================================

All five are populated for all 670 players live. That matters because the
alternative was to model price pressure from net transfers over ownership --
inventing a precision we could not validate -- when the thing a manager wants
("does Gabriel rise tonight?") is published outright. Measured on 2026-11-06:
Gabriel sat at 105.5% with no change yet, Wissa at -139.5%.

Two states have to be rendered rather than averaged away, because both make a
bare percentage misleading:

* **Locked.** A player changes at most once a day, so one who has already moved
  sits above 100% and is *not* about to move again. 39 of 670 were in that state
  live, and 20 carried an explicit ``locked_until``.
* **Calibrating.** FPL says when its own figure is not yet trustworthy, and a
  progress bar that does not say so is a confident claim the source disclaims.
"""

import pandas as pd
import streamlit as st

import logging

import plotly.graph_objects as go

import config
from scripts.common.classic_squad import resolve_classic_squad
from scripts.common.display_names import display_of
from scripts.common.fpl_classic_api import get_classic_team_history
from scripts.common.price_signals import (
    POS_LETTER as _POS_LETTER,
    build_price_data,
    change_state as _change_state,
)
from scripts.common.text_helpers import to_display_name
from scripts.common.styled_tables import render_styled_table
from scripts.common.utils import get_classic_bootstrap_static

_logger = logging.getLogger("fpl_app.price_changes")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _format_price(tenths: int) -> str:
    """Convert FPL price (tenths) to display string, e.g. 100 -> '£10.0m'."""
    return f"£{tenths / 10:.1f}m"


def _format_change(tenths: int) -> str:
    """Signed price change, e.g. 5 -> '+£0.5m', -3 -> '-£0.3m'.

    The sign goes **outside** the currency symbol. The previous form put it
    inside and rendered "£-0.3m", contradicting this docstring's own example.
    """
    try:
        val = float(tenths)
    except (TypeError, ValueError):
        return ""
    sign = "+" if val > 0 else ("-" if val < 0 else "")
    return "%s£%.1fm" % (sign, abs(val) / 10)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def _build_price_data() -> pd.DataFrame:
    """Fetch the bootstrap and hand it to the shared builder.

    The frame itself is built in `scripts/common/price_signals.py` so the
    Classic My Week checklist can read the same states without importing a
    page's private helper. `_change_state` is re-exported above for the same
    reason, and because this module's tests import it by that name.
    """
    return build_price_data(get_classic_bootstrap_static())


def _format_momentum(rate) -> str:
    """FPL's published net transfers an hour, as a signed short figure.

    **Deliberately not converted into an ETA.** Turning a rate into "changes in
    ~3h" needs the threshold expressed in transfers, and FPL publishes the
    progress as a percentage without saying what 100% is worth -- so an ETA
    would rest on an assumption this page cannot check, which is the fabricated
    precision the whole design avoids. The rate on its own answers the question
    that is actually answerable: is he accelerating towards the change or
    stalling short of it.
    """
    val = pd.to_numeric(rate, errors="coerce")
    if pd.isna(val) or val == 0:
        return ""
    sign = "+" if val > 0 else "-"
    mag = abs(float(val))
    if mag >= 1000:
        return "%s%.1fk/h" % (sign, mag / 1000)
    return "%s%d/h" % (sign, int(mag))


def _progress_width(pct) -> float:
    """Bar width 0-100 for a percentage that legitimately exceeds +/-100.

    The real figure is printed beside the bar, so clamping the *bar* loses
    nothing and keeps a -139% from drawing outside its track.
    """
    val = pd.to_numeric(pct, errors="coerce")
    if pd.isna(val):
        return 0.0
    return float(min(100.0, abs(float(val))))


# ------------------------------------------------------------------
# CSS for price cards
# ------------------------------------------------------------------
def _price_card_css() -> str:
    return """
    <style>
    .price-cards-container { display: flex; flex-direction: column; gap: 4px; }
    .price-row {
        display: flex; align-items: center; gap: 8px;
        padding: 6px 10px; border-radius: 8px;
        color: #e0e0e0; font-size: 0.88rem;
    }
    .price-row-rise { background: #16213e; border-left: 3px solid #00ff87; }
    .price-row-fall { background: #16213e; border-left: 3px solid #ff4757; }
    .price-row-name { font-weight: 700; flex: 1; color: #e0e0e0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .price-row-meta { color: #9ca3af; font-size: 0.78rem; min-width: 48px; }
    .price-row-price { font-weight: 600; color: #e0e0e0; min-width: 58px; text-align: right; }
    .price-row-change {
        font-weight: 800; font-size: 0.8rem;
        padding: 2px 8px; border-radius: 10px; min-width: 60px; text-align: center;
    }
    .change-rise { background: rgba(0,255,135,0.2); color: #00ff87; }
    .change-fall { background: rgba(255,71,87,0.2); color: #ff4757; }
    /* Watchlist rows: a progress track plus the published figure. */
    .pc-track { flex: 1; height: 7px; border-radius: 4px; background: #0f1628; overflow: hidden; min-width: 70px; }
    .pc-fill { height: 100%; border-radius: 4px; }
    .pc-fill-rise { background: linear-gradient(90deg,#0b6b45,#00ff87); }
    .pc-fill-fall { background: linear-gradient(90deg,#7a1f28,#ff4757); }
    .pc-pct { font-weight: 800; font-size: 0.78rem; min-width: 56px; text-align: right; }
    .pc-mom { color: #9ca3af; font-size: 0.72rem; min-width: 64px; text-align: right; }
    .pc-flag { font-size: 0.68rem; padding: 1px 6px; border-radius: 8px;
               background: rgba(224,163,62,0.18); color: #e0a33e; font-weight: 700; }
    .pc-summary { display: flex; gap: 10px; margin-bottom: 10px; }
    .pc-stat { flex: 1; text-align: center; padding: 14px 10px; border-radius: 10px;
               background: linear-gradient(135deg,#1a1a2e 0%,#16213e 100%); border: 1px solid #333; }
    .pc-stat-val { font-size: 1.7em; font-weight: 800; }
    .pc-stat-lbl { color: #888; font-size: 0.8em; margin-top: 3px; }
    .no-changes-msg {
        text-align: center; padding: 20px; color: #9ca3af;
        background: #1a1a2e; border-radius: 8px; border: 1px solid #333;
    }
    </style>
    """


# ------------------------------------------------------------------
# Section 0: your squad, and what tonight costs you
# ------------------------------------------------------------------
def _my_squad_codes() -> set:
    """The configured Classic squad as player **codes**, or an empty set.

    Codes, not element ids: `code` is the only identifier both FPL APIs agree
    on, and this frame is built from the Classic bootstrap while the squad may
    be resolved from either. See CLAUDE.md, "Player identity".

    Goes through `resolve_classic_squad` -- the one resolver -- rather than
    reaching for `picks` directly, so this page sees the same fifteen as every
    other Classic page, including a pre-deadline wildcard.
    """
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        return set()
    try:
        bootstrap = get_classic_bootstrap_static()
        res = resolve_classic_squad(int(team_id), bootstrap, config.CURRENT_GAMEWEEK)
        if not res.ok:
            return set()
        codes = {p.get("code") for p in bootstrap.get("elements", [])
                 if p.get("id") in {q.get("element") for q in res.picks}}
        return {c for c in codes if c is not None}
    except Exception:
        _logger.warning("Could not resolve the Classic squad for price exposure",
                        exc_info=True)
        return set()


def _render_my_exposure(df: pd.DataFrame, squad_codes: set):
    """What tonight's changes do to *your* squad.

    The league-wide board answers "who is moving"; this answers "does it cost me
    anything", which is the only reason to look before a deadline. A faller you
    own is money off your team value; a riser you do not own is money it will
    cost you to buy him.
    """
    if not squad_codes:
        return
    mine = df[df["Player_Code"].isin(squad_codes)].copy()
    if mine.empty:
        return
    mine["_state"] = mine.apply(_change_state, axis=1)

    rising = mine[mine["_state"] == "rise"]
    falling = mine[mine["_state"] == "fall"]

    st.subheader("Your Squad Tonight")
    st.markdown(
        '<div class="pc-summary">'
        f'<div class="pc-stat"><div class="pc-stat-val" style="color:#00ff87;">{len(rising)}</div>'
        '<div class="pc-stat-lbl">of yours rising</div></div>'
        f'<div class="pc-stat"><div class="pc-stat-val" style="color:#ff4757;">{len(falling)}</div>'
        '<div class="pc-stat-lbl">of yours falling</div></div>'
        f'<div class="pc-stat"><div class="pc-stat-val" style="color:#e0e0e0;">{len(mine)}</div>'
        '<div class="pc-stat-lbl">squad players tracked</div></div>'
        '</div>', unsafe_allow_html=True)

    # **Ranked by proximity, not by whether the threshold is already crossed.**
    # Filtering to rise/fall only printed "none of your squad is close to a price
    # change" while a player sat at -95% and accelerating at -27k/h, which is
    # exactly the player a manager needs to see. The counts above still mean
    # "past the threshold"; this list means "closest".
    near = mine.dropna(subset=["Change_Pct"]).copy()
    if near.empty:
        st.markdown('<div class="no-changes-msg">FPL is not publishing '
                    'price-change progress for your squad.</div>',
                    unsafe_allow_html=True)
    else:
        near["_prox"] = near["Change_Pct"].abs()
        near = near.nlargest(min(6, len(near)), "_prox")
        # Fallers first within that: a fall costs you money you already have,
        # where a rise on a player you own is free.
        st.markdown(_watchlist_html(near.sort_values("Change_Pct")),
                    unsafe_allow_html=True)
    st.caption(
        "Your six closest to a change, nearest first — **not** only those past "
        "the threshold, which is what the two counts above mean. A **fall** takes "
        "0.1m off your squad value; a rise on a player you already own is free. "
        "Selling prices are not shown because FPL publishes them only to an "
        "authenticated request — see League Setup."
    )


# ------------------------------------------------------------------
# Section 0b: team value over the season
# ------------------------------------------------------------------
def _render_value_tracker():
    """Squad value and money in the bank, per gameweek.

    Both come straight from `entry/{id}/history`. They are rendered as FPL's own
    two fields rather than summed into a single "team value", because the two
    conventions in circulation disagree about whether `value` already includes
    the bank and this page has no way to check without an authenticated
    `my-team` read.
    """
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        return
    try:
        history = get_classic_team_history(int(team_id)) or {}
    except Exception:
        _logger.warning("Could not load Classic history for the value tracker",
                        exc_info=True)
        return
    rows = history.get("current") or []
    if len(rows) < 2:
        return

    hist = pd.DataFrame(rows)
    hist["Squad value"] = pd.to_numeric(hist["value"], errors="coerce") / 10.0
    hist["In the bank"] = pd.to_numeric(hist["bank"], errors="coerce") / 10.0
    first, last = hist.iloc[0], hist.iloc[-1]
    delta = float(last["Squad value"]) - float(first["Squad value"])

    st.subheader("Your Team Value")
    st.markdown(
        '<div class="pc-summary">'
        f'<div class="pc-stat"><div class="pc-stat-val" style="color:#e0e0e0;">'
        f'£{last["Squad value"]:.1f}m</div><div class="pc-stat-lbl">squad value</div></div>'
        f'<div class="pc-stat"><div class="pc-stat-val" style="color:#e0e0e0;">'
        f'£{last["In the bank"]:.1f}m</div><div class="pc-stat-lbl">in the bank</div></div>'
        f'<div class="pc-stat"><div class="pc-stat-val" '
        f'style="color:{"#00ff87" if delta >= 0 else "#ff4757"};">'
        f'{"+" if delta >= 0 else ""}£{delta:.1f}m</div>'
        '<div class="pc-stat-lbl">since GW%d</div></div>'
        '</div>' % int(first["event"]), unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["event"], y=hist["Squad value"], name="Squad value",
        mode="lines+markers", line=dict(color="#00ff87", width=2),
        hovertemplate="GW%{x}<br>Squad value £%{y:.1f}m<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=hist["event"], y=hist["In the bank"], name="In the bank",
        mode="lines+markers", line=dict(color="#6CABDD", width=2, dash="dot"),
        hovertemplate="GW%{x}<br>Bank £%{y:.1f}m<extra></extra>"))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#e0e0e0"), legend=dict(orientation="h", y=1.15, x=0),
        hovermode="x unified",
    )
    # One tick per gameweek: the x-axis is a small integer range, and Plotly's
    # default picks half-gameweeks on a short season.
    fig.update_xaxes(title="Gameweek", dtick=1, showgrid=False)
    fig.update_yaxes(title="£m", showgrid=True, gridcolor="#222")
    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------------------------------
# Section 0c: the league-wide watchlist
# ------------------------------------------------------------------
def _watchlist_html(rows: pd.DataFrame) -> str:
    """Rows of ``name | team/pos | price | progress bar | % | momentum``."""
    html = '<div class="price-cards-container">'
    for _, row in rows.iterrows():
        state = _change_state(row)
        pct = pd.to_numeric(row.get("Change_Pct"), errors="coerce")
        rising = bool(pd.notna(pct) and pct >= 0)
        flag = ""
        if state == "locked":
            flag = ('<span class="pc-flag" title="FPL allows one change per '
                    'player per day, so this one cannot move again yet.">LOCKED</span>')
        elif state == "calibrating":
            flag = ('<span class="pc-flag" title="FPL reports its own figure as '
                    'still calibrating for this player.">CALIBRATING</span>')
        else:
            # A player can change more than once in a gameweek, on separate
            # days, so a high percentage after a change is not a contradiction
            # -- but it is context. Evanilson read +160% having already risen
            # 0.1 this gameweek, and the row said nothing about it.
            moved = pd.to_numeric(row.get("cost_change_event"), errors="coerce")
            if pd.notna(moved) and moved != 0:
                flag = ('<span class="pc-flag" title="Already moved %s this '
                        'gameweek; FPL permits a further change on a later day.'
                        '">%s TODAY</span>' % (_format_change(int(moved)),
                                               _format_change(int(moved))))
        html += (
            '<div class="price-row %s">'
            '<span class="price-row-name">%s</span>'
            '<span class="price-row-meta">%s &middot; %s</span>'
            '<span class="price-row-price">%s</span>'
            '<span class="pc-track"><span class="pc-fill %s" style="width:%.0f%%;"></span></span>'
            '<span class="pc-pct %s">%s</span>'
            '<span class="pc-mom">%s</span>%s'
            '</div>' % (
                "price-row-rise" if rising else "price-row-fall",
                display_of(row), row.get("Team", ""), row.get("Pos", ""),
                _format_price(row.get("now_cost", 0)),
                "pc-fill-rise" if rising else "pc-fill-fall",
                _progress_width(pct),
                "change-rise" if rising else "change-fall",
                "n/a" if pd.isna(pct) else "%+.0f%%" % pct,
                _format_momentum(row.get("Hourly")), flag,
            )
        )
    return html + "</div>"


def _render_watchlist(df: pd.DataFrame):
    """Who FPL's own model has closest to a change.

    This is the section the page existed to have and did not: everything below
    reports changes that have **already happened**, which is history by the time
    a manager reads it.
    """
    st.subheader("Closest to a Change")
    work = df.dropna(subset=["Change_Pct"]).copy()
    if work.empty:
        st.markdown('<div class="no-changes-msg">FPL is not publishing '
                    'price-change progress right now.</div>', unsafe_allow_html=True)
        return
    work["_state"] = work.apply(_change_state, axis=1)
    n_cal = int((work["_state"] == "calibrating").sum())
    n_lock = int((work["_state"] == "locked").sum())

    show_n = st.slider("How many to show", 5, 25, 10, key="pc_watch_n")
    only_mine = False
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Heading for a rise**")
        risers = work[work["Change_Pct"] > 0].nlargest(show_n, "Change_Pct")
        st.markdown(_watchlist_html(risers) if not risers.empty
                    else '<div class="no-changes-msg">Nobody rising.</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown("**Heading for a fall**")
        fallers = work[work["Change_Pct"] < 0].nsmallest(show_n, "Change_Pct")
        st.markdown(_watchlist_html(fallers) if not fallers.empty
                    else '<div class="no-changes-msg">Nobody falling.</div>',
                    unsafe_allow_html=True)

    note = ("**100% is the threshold** — FPL's own published progress, not our "
            "estimate. The rate beside it is net transfers an hour, which says "
            "whether a player is accelerating or stalling; it is deliberately "
            "not turned into a countdown, because FPL does not publish what "
            "100% is worth in transfers.")
    if n_lock:
        note += (" %d player%s **locked** — already changed today, so cannot move "
                 "again until FPL's next window." % (n_lock, "s are" if n_lock != 1 else " is"))
    if n_cal:
        note += (" %d **calibrating**, where FPL says its own figure is not yet "
                 "settled." % n_cal)
    st.caption(note)


# ------------------------------------------------------------------
# Section 1: GW Price Changes
# ------------------------------------------------------------------
def _render_gw_price_changes(df: pd.DataFrame):
    st.subheader("This Gameweek's Price Changes")
    st.markdown(_price_card_css(), unsafe_allow_html=True)

    risers = df[df["cost_change_event"] > 0].sort_values("cost_change_event", ascending=False)
    fallers = df[df["cost_change_event"] < 0].sort_values("cost_change_event", ascending=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Risers**")
        if risers.empty:
            st.markdown('<div class="no-changes-msg">No risers this gameweek yet</div>',
                        unsafe_allow_html=True)
        else:
            html = '<div class="price-cards-container">'
            for _, row in risers.iterrows():
                html += (
                    f'<div class="price-row price-row-rise">'
                    f'<span class="price-row-name">{display_of(row)}</span>'
                    f'<span class="price-row-meta">{row["Team"]} · {row["Pos"]}</span>'
                    f'<span class="price-row-price">{_format_price(row["now_cost"])}</span>'
                    f'<span class="price-row-change change-rise">{_format_change(row["cost_change_event"])}</span>'
                    f'</div>'
                )
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)

    with col2:
        st.markdown("**Fallers**")
        if fallers.empty:
            st.markdown('<div class="no-changes-msg">No fallers this gameweek yet</div>',
                        unsafe_allow_html=True)
        else:
            html = '<div class="price-cards-container">'
            for _, row in fallers.iterrows():
                html += (
                    f'<div class="price-row price-row-fall">'
                    f'<span class="price-row-name">{display_of(row)}</span>'
                    f'<span class="price-row-meta">{row["Team"]} · {row["Pos"]}</span>'
                    f'<span class="price-row-price">{_format_price(row["now_cost"])}</span>'
                    f'<span class="price-row-change change-fall">{_format_change(row["cost_change_event"])}</span>'
                    f'</div>'
                )
            html += '</div>'
            st.markdown(html, unsafe_allow_html=True)


# ------------------------------------------------------------------
# Section 2: Season Price Movers
# ------------------------------------------------------------------
def _render_season_movers(df: pd.DataFrame):
    st.subheader("Season Price Movers")

    df = df.copy()
    df["start_price"] = df["now_cost"] - df["cost_change_start"]

    top_risers = df[df["cost_change_start"] > 0].nlargest(20, "cost_change_start")
    top_fallers = df[df["cost_change_start"] < 0].nsmallest(20, "cost_change_start")

    col1, col2 = st.columns(2)

    with col1:
        if top_risers.empty:
            st.info("No season risers yet.")
        else:
            display = top_risers[["Player", "Team", "Pos"]].copy()
            display["Start"] = top_risers["start_price"].apply(_format_price)
            display["Current"] = top_risers["now_cost"].apply(_format_price)
            display["Change"] = top_risers["cost_change_start"].apply(_format_change)
            render_styled_table(display, title="Biggest Risers")

    with col2:
        if top_fallers.empty:
            st.info("No season fallers yet.")
        else:
            display = top_fallers[["Player", "Team", "Pos"]].copy()
            display["Start"] = top_fallers["start_price"].apply(_format_price)
            display["Current"] = top_fallers["now_cost"].apply(_format_price)
            display["Change"] = top_fallers["cost_change_start"].apply(_format_change)
            render_styled_table(display, title="Biggest Fallers")


# ------------------------------------------------------------------
# Section 3: Transfer Activity
# ------------------------------------------------------------------
def _render_transfer_activity(df: pd.DataFrame):
    st.subheader("Transfer Activity (Price Pressure)")
    st.caption("Players with the highest net transfers are most likely to see price changes.")

    df = df.copy()
    df["net_transfers"] = df["transfers_in_event"] - df["transfers_out_event"]

    most_in = df.nlargest(20, "net_transfers")
    most_out = df.nsmallest(20, "net_transfers")

    col1, col2 = st.columns(2)

    with col1:
        if most_in.empty:
            st.info("No transfer data available.")
        else:
            display = most_in[["Player", "Team"]].copy()
            display["Price"] = most_in["now_cost"].apply(_format_price)
            display["Net"] = most_in["net_transfers"].apply(lambda x: f"{x:+,}")
            display["Own%"] = most_in["selected_by_percent"].astype(str) + "%"
            render_styled_table(display, title="Most Transferred In")

    with col2:
        if most_out.empty:
            st.info("No transfer data available.")
        else:
            display = most_out[["Player", "Team"]].copy()
            display["Price"] = most_out["now_cost"].apply(_format_price)
            display["Net"] = most_out["net_transfers"].apply(lambda x: f"{x:+,}")
            display["Own%"] = most_out["selected_by_percent"].astype(str) + "%"
            render_styled_table(display, title="Most Transferred Out")


# ------------------------------------------------------------------
# Main page function
# ------------------------------------------------------------------
def show_price_changes_page():
    st.header("💰 Price Changes")
    df = _build_price_data()
    if df.empty:
        st.warning("Could not load player data. Try again in a bit.")
        return
    st.markdown(_price_card_css(), unsafe_allow_html=True)

    # **Ordered by what a manager can still act on.** Everything below the
    # watchlist reports changes that have already happened, which is history by
    # the time it is read -- and that was the whole page.
    _render_my_exposure(df, _my_squad_codes())
    _render_value_tracker()
    st.divider()
    _render_watchlist(df)
    st.divider()
    _render_gw_price_changes(df)
    st.divider()
    _render_season_movers(df)
    st.divider()
    _render_transfer_activity(df)
