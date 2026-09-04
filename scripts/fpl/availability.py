"""Availability: transfer news, market odds and injuries on one page.

These are the same question asked in the two halves of the calendar — *will this
player be on the pitch to score me points* — and which half you are in decides
which answer matters.  While a window is open a squad can lose a player outright;
once every window has shut, the only thing that can take him away is his body.

So the tabs reorder themselves: Tracker leads while any window is open, Injuries
leads when none is.  The banner says which, because nothing in the app used to.
``transfer_exposure`` quietly returns 0 once the calendar is spent, and a page
with the transfer feature switched off looked exactly like a page where nobody
was at risk.

Note the two windows do not close together.  On 2026-09-04 the English window has
been shut for three days and the Saudi one runs for another five weeks — the gap
that let Ollie Watkins be sold out of a squad five gameweeks into a season whose
window had "closed".
"""

import datetime as _dt
from email.utils import parsedate_to_datetime

import pandas as pd
import streamlit as st

from scripts.common.error_helpers import get_logger
from scripts.common.scraping import (get_player_odds_ladder, get_transfer_news,
                                     get_transfer_odds_index,
                                     transfer_news_cache_status)
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import to_display_name
from scripts.common.transfer_odds import (age_band, group_ladder,
                                          ladder_overround, odds_age_days)
from scripts.common.transfer_risk import (TIER_A, TIER_B, TIER_C,
                                          attach_transfer_risk,
                                          classify_headline, window_status)
from scripts.common.transfer_risk_app import attach_odds, get_pl_team_names
from scripts.common.utils import get_classic_bootstrap_static
from scripts.fpl.injuries import get_fpl_availability_df, render_injuries_tab

_logger = get_logger("fpl_app.availability")

#: How many players to scan news for. Matches the Draft Helper's default so the
#: two pages share cache entries rather than each warming their own.
_SCAN_DEPTH = 150
_SECONDS_PER_PLAYER = 0.07

_TIER_LABELS = [
    (TIER_A, "Confirmed", "#00ff87"),
    (TIER_B, "In talks", "#ffd166"),
    (TIER_C, "Linked", "#8ecae6"),
]

_BAND_STYLE = {
    "live": ("🟢", "live"),
    "aging": ("🟡", "aging"),
    "stale": ("🟠", "stale"),
    "archival": ("🔴", "archival"),
}


def _tier_label(headline):
    """Headline -> (label, colour). ``None`` when it carries no transfer signal."""
    tier = classify_headline(headline)
    for threshold, label, colour in _TIER_LABELS:
        if tier >= threshold:
            return label, colour
    return None


def _parse_published(value):
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError, IndexError):
        return None
    if parsed is None:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)


def _age_text(when):
    if when is None:
        return "—"
    delta = _dt.datetime.now(_dt.timezone.utc) - when
    hours = delta.total_seconds() / 3600.0
    if hours < 1:
        return "%dm ago" % max(1, int(delta.total_seconds() / 60))
    if hours < 48:
        return "%dh ago" % int(hours)
    return "%dd ago" % int(hours / 24)


@st.cache_data(ttl=600)
def _load_pool():
    """FPL availability plus the current PL club names.

    Cross-format on purpose: this page sits under FPL App Home and must work for
    a Classic-only user with no Draft league configured.
    """
    try:
        pool = get_fpl_availability_df()
    except Exception:
        _logger.warning("Availability pool unavailable", exc_info=True)
        pool = pd.DataFrame()

    pl_teams = []
    try:
        bootstrap = get_classic_bootstrap_static()
        pl_teams = get_pl_team_names(bootstrap)
        # Rank by prominence so a capped news scan spends its ~150 requests on
        # players anyone would notice leaving. The availability frame arrives in
        # element-id order, which is arbitrary.
        points = {e["id"]: e.get("total_points") or 0
                  for e in (bootstrap or {}).get("elements", [])}
        cost = {e["id"]: e.get("now_cost") or 0
                for e in (bootstrap or {}).get("elements", [])}
        if points and not pool.empty and "Player_ID" in pool.columns:
            pool = pool.copy()
            pool["_prominence"] = pool["Player_ID"].map(
                lambda i: (points.get(i, 0), cost.get(i, 0)))
            pool = pool.sort_values("_prominence", ascending=False).drop(
                columns=["_prominence"])
    except Exception:
        _logger.warning("Could not rank availability pool by prominence", exc_info=True)
    return pool, pl_teams


# --- Window banner ----------------------------------------------------------

def _render_window_banner():
    """State of every window, and how long is left. Returns True if any is open."""
    status = window_status()
    open_regions = [s for s in status.values() if s["open"]]

    if open_regions:
        soonest = min(open_regions, key=lambda s: s["days_remaining"])
        bg, border = "#052e16", "#00ff87"
        headline = "Transfer window open — %s closes in %d days (%s)" % (
            soonest["label"], soonest["days_remaining"],
            soonest["closes"].strftime("%d %b"))
    else:
        bg, border = "#2a1215", "#ff6b6b"
        upcoming = [s for s in status.values() if s["days_until"] is not None]
        if upcoming:
            nxt = min(upcoming, key=lambda s: s["days_until"])
            headline = "All windows closed — %s reopens in %d days (%s)" % (
                nxt["label"], nxt["days_until"], nxt["opens"].strftime("%d %b"))
        else:
            headline = "All windows closed for the season — no move can complete"

    parts = []
    for region in sorted(status, key=lambda r: (not status[r]["open"], r)):
        s = status[region]
        if s["open"]:
            parts.append("🟢 %s open until %s" % (s["label"], s["closes"].strftime("%d %b")))
        elif s["days_until"] is not None:
            parts.append("🔴 %s shut, reopens %s" % (s["label"], s["opens"].strftime("%d %b")))
        else:
            parts.append("🔴 %s done for the season" % s["label"])

    st.markdown(
        '<div style="background:%s;border-left:4px solid %s;padding:12px 16px;'
        'border-radius:4px;margin-bottom:16px;color:#e0e0e0;">'
        '<strong style="color:#ffffff;">%s</strong>'
        '<br><small style="opacity:0.85;">%s</small></div>'
        % (bg, border, headline, " &nbsp;·&nbsp; ".join(parts)),
        unsafe_allow_html=True,
    )
    return bool(open_regions)


# --- Shared data ------------------------------------------------------------

def _scan_pairs(pool, odds_index):
    """Which players to fetch news for.

    Everyone with a live betting market, then down the pool to the scan depth.
    Markets come first because a quoted player is by definition one the market
    thinks is moving, and there are only ~57 of them.
    """
    if pool is None or pool.empty:
        return []
    priority, seen, pairs = set(), set(), []
    if odds_index is not None and not odds_index.empty:
        matched = attach_odds(pool, odds_index)
        priority = set(matched["Player"].astype(str)) if not matched.empty else set()

    ordered = pd.concat([
        pool[pool["Player"].astype(str).isin(priority)],
        pool[~pool["Player"].astype(str).isin(priority)],
    ]) if priority else pool

    for _, row in ordered.head(_SCAN_DEPTH).iterrows():
        name = str(row.get("Player") or "")
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        pairs.append((name, str(row.get("Team") or "")))
    return pairs


def _build_board(pool, news_df, odds_index, pl_teams):
    """Score the pool, keeping only players the sources actually say something about.

    Returns ``(board, matched_odds)``; the caller needs the matched frame for the
    site's own URL slugs.
    """
    if pool is None or pool.empty:
        return pd.DataFrame(), None

    named = set()
    if news_df is not None and not news_df.empty:
        named |= {str(p).lower() for p in news_df["Player"].dropna()}

    matched_odds = attach_odds(pool, odds_index) if odds_index is not None else None
    if matched_odds is not None and not matched_odds.empty:
        named |= {str(p).lower() for p in matched_odds["Player"]}

    # Departed players matter even when nobody is writing about them any more.
    subset = pool[
        pool["Player"].astype(str).str.lower().isin(named)
        | pool["Status"].astype(str).str.lower().eq("u")
    ].copy()
    if subset.empty:
        return pd.DataFrame(), matched_odds

    subset["status"] = subset["Status"]
    subset["news"] = subset["News"]
    try:
        return attach_transfer_risk(subset, news_df, pl_teams, odds_df=matched_odds), matched_odds
    except Exception:
        _logger.warning("Transfer board scoring failed", exc_info=True)
        return pd.DataFrame(), matched_odds


def _display_name(row):
    parts = str(row.get("Player") or "").split()
    first = parts[0] if parts else ""
    second = " ".join(parts[1:]) if len(parts) > 1 else ""
    try:
        return to_display_name(first, second, row.get("Web_Name"))
    except Exception:
        return str(row.get("Player") or "")


# --- Tabs -------------------------------------------------------------------

def render_headlines_tab(news_df):
    """Reverse-chronological feed of everything the news scan has cached.

    The article ``URL`` is fetched and cached today but has never been rendered
    anywhere; this is the first page to use it.
    """
    st.caption(
        "Every transfer headline in the cache, newest first. Tiers come from the "
        "same classifier that drives the risk score, so what you read here is "
        "what the model read."
    )
    if news_df is None or news_df.empty:
        st.info("No headlines cached yet. Use **Scan transfer news** on the Tracker tab.")
        return

    rows = []
    for _, item in news_df.iterrows():
        headline = str(item.get("Headline") or "")
        tier = _tier_label(headline)
        if tier is None:
            continue
        when = _parse_published(item.get("Published"))
        rows.append({
            "Tier": tier[0], "Player": str(item.get("Player") or ""),
            "Club": str(item.get("Team") or ""), "Headline": headline,
            "Source": str(item.get("Source") or ""), "Age": _age_text(when),
            "URL": str(item.get("URL") or ""), "_sort": when,
        })
    if not rows:
        st.info("Nothing in the cache carries a transfer signal right now.")
        return

    feed = pd.DataFrame(rows)
    feed = feed.sort_values("_sort", ascending=False, na_position="last")

    c1, c2, c3 = st.columns([1, 1, 2])
    tiers = [t[1] for t in _TIER_LABELS]
    tier_sel = c1.multiselect("Tier", tiers, default=tiers, key="hl_tier")
    clubs = sorted(c for c in feed["Club"].unique() if c)
    club_sel = c2.multiselect("Club", clubs, default=None, key="hl_club")
    query = c3.text_input("Search headlines", key="hl_q").strip().lower()

    if tier_sel:
        feed = feed[feed["Tier"].isin(tier_sel)]
    if club_sel:
        feed = feed[feed["Club"].isin(club_sel)]
    if query:
        feed = feed[feed["Headline"].str.lower().str.contains(query, regex=False)
                    | feed["Player"].str.lower().str.contains(query, regex=False)]

    st.caption("%d headlines" % len(feed))
    for _, item in feed.head(120).iterrows():
        colour = next((c for _t, lab, c in _TIER_LABELS if lab == item["Tier"]), "#8ecae6")
        link = item["Headline"]
        if item["URL"]:
            link = '<a href="%s" target="_blank" style="color:#e0e0e0;text-decoration:none;">%s</a>' % (
                item["URL"], item["Headline"])
        st.markdown(
            '<div style="background:#1a1a2e;border-left:3px solid %s;padding:8px 12px;'
            'margin-bottom:6px;border-radius:4px;color:#e0e0e0;">'
            '<span style="background:%s;color:#0b0b16;font-weight:700;font-size:11px;'
            'padding:1px 6px;border-radius:3px;">%s</span> '
            '<strong style="color:#ffffff;">%s</strong> '
            '<span style="color:#9aa0b4;">%s</span><br>%s'
            '<br><small style="color:#9aa0b4;">%s · %s</small></div>'
            % (colour, colour, item["Tier"], item["Player"], item["Club"],
               link, item["Source"], item["Age"]),
            unsafe_allow_html=True,
        )


def _render_ladder(player, slug):
    """Destination ladder for one player, fetched only when asked for.

    Streamlit executes an expander's body whether or not it is open, so putting a
    network call in here unguarded fetches every ladder on every page load — the
    opposite of on-demand.  The cache is therefore read first and the network is
    touched only behind a button, which reruns the script and lands on the cached
    path the second time.
    """
    if not slug:
        st.caption("No market published for this player.")
        return

    ladder = get_player_odds_ladder(player, slug=slug, cached_only=True)
    if ladder is None or ladder.empty:
        if st.button("Load destination odds", key="ladder_%s" % slug):
            with st.spinner("Fetching destination odds…"):
                get_player_odds_ladder(player, slug=slug)
            st.rerun()
        return

    rows = ladder.to_dict("records")
    updated = rows[0].get("Updated")
    age = odds_age_days(updated)
    icon, band = _BAND_STYLE.get(age_band(age), ("⚪", "unknown"))
    st.caption(
        "%s Quote is **%s** (%d days old) · book overround %.0f%% · "
        "shares are *given that he moves*, not odds of moving."
        % (icon, band, round(age), 100 * ladder_overround(rows))
    )

    for entry in group_ladder(rows):
        bar = "█" * max(1, int(round(entry["Probability"] * 20)))
        st.markdown(
            '<div style="color:#e0e0e0;font-family:monospace;">'
            '%-22s <span style="color:#00ff87;">%s</span> %.0f%%</div>'
            % (entry["Destination"][:22], bar, 100 * entry["Probability"]),
            unsafe_allow_html=True,
        )
        for member in entry.get("Members", []):
            st.markdown(
                '<div style="color:#9aa0b4;font-family:monospace;padding-left:22px;">'
                '└ %s — %s (%.0f%% raw)</div>'
                % (member["Destination"], member["Fractional"] or "—",
                   100 * member["Implied"]),
                unsafe_allow_html=True,
            )


def render_tracker_tab(pool, odds_index, pl_teams):
    """The board: who might leave, how likely, and where to."""
    pairs = _scan_pairs(pool, odds_index)
    cached_n, missing_n = transfer_news_cache_status(pairs)

    c1, c2 = st.columns([1, 3])
    scan = c1.button("🔍 Scan transfer news", key="trk_scan")
    c2.caption(
        "%d of %d players cached. A scan fetches the missing %d (~%ds)."
        % (cached_n, len(pairs), missing_n, max(1, int(missing_n * _SECONDS_PER_PLAYER)))
    )

    if scan and pairs:
        bar = st.progress(0.0, text="Fetching transfer news…")

        def _on_progress(done, total, player):
            bar.progress(min(1.0, done / float(total or 1)),
                         text="Fetching transfer news… %s (%d/%d)" % (player, done, total))

        news_df = get_transfer_news(tuple(pairs), force_refresh=True, progress=_on_progress)
        bar.empty()
    else:
        news_df = get_transfer_news(tuple(pairs), cached_only=True)

    st.session_state["availability_news_df"] = news_df

    board, matched_odds = _build_board(pool, news_df, odds_index, pl_teams)
    if board is None or board.empty:
        if missing_n:
            st.info("Nothing cached yet — press **Scan transfer news** to fill the board.")
        else:
            st.success("No transfer activity found for any player right now.")
        return

    board = board[(board["Transfer_Risk"] > 0) | (board["Odds_Risk"] > 0)
                  | (board["Transfer_Status"] != "")].copy()
    if board.empty:
        st.success("No transfer activity found for any player right now.")
        return

    board["Display"] = board.apply(_display_name, axis=1)
    board = board.sort_values("Transfer_Risk", ascending=False)

    show = pd.DataFrame({
        "Player": board["Display"],
        "Club": board["Team"],
        "Pos": board["Position"],
        "Exit %": (board["Transfer_Risk"] * 100).round(0).astype("Int64"),
        "Status": board["Transfer_Status"].replace("", "—"),
        "Destination": board["Transfer_Destination"].replace("", "—"),
        "Odds": board["Odds_Fractional"].replace("", "—"),
        "Mkt %": (board["Odds_Risk"] * 100).round(0).astype("Int64"),
        "Age": board["Odds_Age_Days"].apply(
            lambda d: "—" if pd.isna(d) else "%dd" % round(d)),
        "Outlets": board["Transfer_Outlets"],
    })
    render_styled_table(show, positive_color_cols=["Exit %"], max_height=520)
    st.caption(
        "**Exit %** blends the news model with the market, the market aged down "
        "by how old its quote is. **Mkt %** is the raw bookmaker price before "
        "that discount. A completed move in the FPL data overrides both."
    )

    slug_by_player = {}
    if matched_odds is not None and not matched_odds.empty and "Odds_Slug" in matched_odds.columns:
        slug_by_player = dict(zip(matched_odds["Player"].astype(str),
                                  matched_odds["Odds_Slug"].astype(str)))

    # A completed departure scores 1.0 and sorts to the top, so ranking the
    # evidence list by risk fills it with settled facts that have nothing left to
    # explain. The interesting rows are the unresolved ones — someone with a live
    # market, or a story still being written.
    unresolved = board[board["Transfer_Status"] != "Departed"]
    with_market = unresolved[unresolved["Odds_Risk"] > 0]
    speculative = unresolved[(unresolved["Odds_Risk"] <= 0)
                             & (unresolved["Transfer_Risk"] > 0)]
    evidence = pd.concat([with_market, speculative]).head(12)

    if evidence.empty:
        st.caption("Nothing unresolved to explain — every move on the board is "
                   "already confirmed in the FPL data.")
        return

    st.markdown("#### Evidence")
    st.caption("Players still in play. Confirmed departures are settled and sit "
               "in the table above.")
    for _, row in evidence.iterrows():
        label = "%s · %s — %d%%" % (row["Display"], row["Team"],
                                    round(100 * row["Transfer_Risk"]))
        with st.expander(label, expanded=False):
            note = row.get("Transfer_Note") or "No summary available."
            st.markdown("**%s**" % note)
            if row.get("Odds_Fractional"):
                st.caption("Market: %s to %s (%s)" % (
                    row["Odds_Fractional"], row.get("Odds_Destination") or "—",
                    row.get("Odds_Bookmaker") or "unknown book"))
            _render_ladder(row["Player"], slug_by_player.get(str(row["Player"]), ""))


def show_availability_page():
    st.title("🚦 Availability")
    st.caption("Transfer news, market odds and injuries — everything that decides "
               "whether a player is yours to pick next week.")

    window_open = _render_window_banner()
    pool, pl_teams = _load_pool()
    odds_index = get_transfer_odds_index()

    headlines = "📰 Headlines"
    tracker = "📈 Tracker"
    injuries = "🏥 Injuries"

    # Whichever question the calendar makes urgent goes first.
    order = ([tracker, headlines, injuries] if window_open
             else [injuries, tracker, headlines])
    tabs = dict(zip(order, st.tabs(order)))

    with tabs[tracker]:
        render_tracker_tab(pool, odds_index, pl_teams)
    with tabs[headlines]:
        render_headlines_tab(st.session_state.get("availability_news_df"))
    with tabs[injuries]:
        render_injuries_tab(key_prefix="avail_inj")
