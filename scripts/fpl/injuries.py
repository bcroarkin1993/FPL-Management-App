# injuries_official.py
import re

import pandas as pd
import streamlit as st
from scripts.common.error_helpers import get_logger
from scripts.common.display_names import attach_display_names
from scripts.common.utils import get_classic_bootstrap_static
from scripts.common.styled_tables import render_styled_table
from scripts.common.analytics import gameweek_deadlines
from scripts.common.injury_helpers import (
    ABSENCE_FROM_NEWS,
    ABSENCE_NONE,
    absence_estimate,
)

_logger = get_logger("fpl_app.injuries")

# FPL element_type -> position letter
#: News text meaning the player has gone, not that he is hurt. Matched on the
#: prose because `status` cannot tell a departure from an injury: both are 'u'.
_DEPARTED_RE = re.compile(r"has joined|has left|on loan", re.I)

_POS_LETTER = {1:"G", 2:"D", 3:"M", 4:"F"}

_AVAILABILITY_COLUMNS = [
    "Player_ID", "Player_Code", "Player", "Web_Name", "Team", "Position",
    "Status", "PlayPct", "StatusBucket", "News", "News_Added",
]

def _bucket_from_playpct(pct: float) -> str:
    if pct is None: return "Questionable"
    if pct <= 0:    return "Out"
    if pct <= 33:   return "Doubtful"
    if pct <= 66:   return "Questionable"
    if pct < 100:   return "Likely"
    return "Available"

def _fallback_pct_from_status(status: str) -> int:
    # FPL status codes: a=available, d=doubtful, i=injured, n=not available, s=suspended, u=unavailable
    s = (status or "").lower()
    if s == "a": return 100
    if s == "d": return 75   # no % provided? assume likely-ish but not 100
    if s in {"i","n","s","u"}: return 0
    return 50

def get_fpl_availability_df() -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
      ['Player_ID','Player','Web_Name','Team','Position','Status','PlayPct','StatusBucket','News','News_Added']
    using only the public FPL bootstrap-static endpoint.

    **The Classic bootstrap, because `Player_ID` leaves this function.** Every
    field read below is published identically by both APIs -- names, club,
    position, status, chances, news -- so for a year this read the Draft one and
    nothing looked wrong. Only `id` differs, and it differs for 64 of 669
    players (see `scripts/common/element_ids.py`). Every consumer of this frame
    joins it to Classic-keyed data: `availability.py` maps it against the
    Classic bootstrap, `projected_lineups.py` joins it to Classic elements, and
    `_attach_pl_injuries` below merges on `Player_ID`. A Draft id resolves to
    *somebody* on each of those joins, just the wrong somebody -- measured
    live, 10 of the affected players carry news, so this page was attributing
    ten injuries to the wrong player.
    """
    js = get_classic_bootstrap_static() or {}
    teams = {t["id"]: t["short_name"] for t in js.get("teams", [])}

    rows = []
    for p in js.get("elements", []):
        pid = p["id"]
        full = f'{p["first_name"]} {p["second_name"]}'.strip()
        web  = p.get("web_name") or full
        team_short = teams.get(p["team"])
        pos = _POS_LETTER.get(p["element_type"])
        status = p.get("status")  # 'a','d','i','n','s','u'
        # prefer official chances if present (this round, else next round)
        c_this = p.get("chance_of_playing_this_round")
        c_next = p.get("chance_of_playing_next_round")
        play_pct = c_this if c_this is not None else c_next
        if play_pct is None:
            play_pct = _fallback_pct_from_status(status)
        bucket = _bucket_from_playpct(float(play_pct) if play_pct is not None else None)
        news = p.get("news") or ""
        news_added = p.get("news_added") or ""

        rows.append({
            "Player_ID": pid,
            "Player_Code": p.get("code"),
            "Player": full,
            "Web_Name": web,
            "Team": team_short,
            "Position": pos,
            "Status": status,
            "PlayPct": float(play_pct) if play_pct is not None else None,
            "StatusBucket": bucket,
            "News": news,
            "News_Added": news_added,
        })
    # Columns are declared even when there are no rows. `get_classic_bootstrap_static`
    # returns {} on failure where the old raw `requests.get` raised, so an empty
    # bootstrap now reaches this line instead of propagating -- and a frame with
    # no columns raises KeyError below, on a page that has no try/except.
    df = pd.DataFrame(rows, columns=_AVAILABILITY_COLUMNS)
    # basic clean
    df["Team"] = df["Team"].astype("string")
    df["Position"] = df["Position"].astype("string")
    return df

def _attach_pl_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """Join premierleague.com's official injury table onto the FPL frame.

    Returns the frame unchanged on any failure -- the PL feed is extra
    information, never a dependency of this page.
    """
    # Imported lazily so an import-time failure in the PL feed cannot take the
    # Availability page with it.
    try:
        from scripts.common import pl_content
        from scripts.common.scraping import get_pl_injuries

        injuries = get_pl_injuries()
        if injuries is None or injuries.empty:
            return df
        matched = pl_content.attach_pl_injuries(pl_content.add_display_names(df), injuries)
        if matched.empty:
            return df
        return df.merge(matched, on="Player_ID", how="left")
    except Exception as exc:                # never take the page down
        _logger.warning("PL injury feed unavailable: %s", exc)
        return df


def _render_pl_disagreements(df: pd.DataFrame):
    """Players the PL reports carrying a knock that FPL still rates available.

    This is the reason the PL feed is worth having: the two desks update
    independently, so each catches the other lagging. It is deliberately framed
    as a disagreement rather than a correction -- **neither source overrides the
    other**. Measured on 2026-09-11 the PL table still listed Nico O'Reilly with
    a back problem on the same day the PL's own article quoted his manager
    saying he was "100 per cent available", so the PL side lags too.
    """
    if "PL_Injury" not in df.columns:
        return

    flagged = df[
        df["PL_Player"].notna()
        & (df["Status"] == "a")
        & (df["PlayPct"].fillna(100) >= 100)
    ]
    if flagged.empty:
        return

    with st.expander(
        "⚠️ Premier League reports a knock, FPL rates them available (%d)"
        % len(flagged),
        expanded=False,
    ):
        st.caption(
            "premierleague.com's injury desk and FPL's bootstrap update "
            "independently, so a player can appear on one and not the other. "
            "This is a watchlist, not a correction — the PL table can lag FPL "
            "just as easily as lead it."
        )
        name_col = "Display_Name" if "Display_Name" in flagged.columns else "Player"
        show = flagged[[name_col, "Team", "Position", "PL_Injury"]].copy()
        show["PL_Injury"] = show["PL_Injury"].replace("", "Unspecified")
        show = show.rename(columns={"PL_Injury": "PL Injury",
                                    name_col: "Player"})
        render_styled_table(show, max_height=340)


def render_injuries_tab(key_prefix: str = "inj"):
    """The availability table: filters plus one styled table.

    Split out of ``show_injuries_page`` so the Availability page can render it as
    a tab beside transfer news. ``key_prefix`` namespaces the widget keys —
    without it a second set of filters on the same page collides on Streamlit's
    auto-generated keys.
    """
    df = get_fpl_availability_df()
    if df.empty:
        st.warning("No data from FPL. Try again in a bit.")
        return

    # Resolved here rather than inside `_attach_pl_injuries`, which returns the
    # frame untouched when the PL feed is down -- the rendered name must not
    # depend on an optional source being reachable.
    df = attach_display_names(df)
    df = _attach_pl_injuries(df)
    _render_pl_disagreements(df)

    # Filters on page (not sidebar)
    c1, c2, c3 = st.columns(3)
    teams = sorted(df["Team"].dropna().unique().tolist())
    poss  = ["G","D","M","F"]
    team_sel = c1.multiselect("Teams", teams, default=None, key="%s_teams" % key_prefix)
    pos_sel  = c2.multiselect("Positions", poss, default=poss, key="%s_pos" % key_prefix)
    min_play = c3.slider("Min Play %", 0, 100, 0, 5, key="%s_minplay" % key_prefix)

    show = df.copy()
    if team_sel:
        show = show[show["Team"].isin(team_sel)]
    if pos_sel:
        show = show[show["Position"].isin(pos_sel)]
    show = show[show["PlayPct"].fillna(0) >= min_play]

    # Nice view. "PL Injury" is the Premier League's own injury-type taxonomy
    # (ACL, Achilles, Hamstring...), which FPL's free-text News does not carry
    # reliably. Absent when the PL feed is unavailable, so it is opt-in here.
    # `Display_Name` instead of `Player` *and* `Web_Name`: this table showed
    # both "Bruno Borges Fernandes" and "B.Fernandes", which are precisely the
    # two fields CLAUDE.md says are not presentable. `Player` stays on the
    # frame for matching.
    # **How long is he out, and how well do we know?** The parsed estimate drives
    # the Power Rankings injury discount, the engine's horizon cap and the
    # Waiver Wire's drop protection, and until now it was rendered nowhere -- so
    # the one page about injuries showed FPL's prose and nothing else.
    #
    # Reported from the app: a Waiver Wire card recommended Chelsea's deputy
    # goalkeeper, a move whose whole value depends on how long the first choice
    # is out. The news said "Head injury - Unknown return date" and the app's own
    # 4-5 gameweek figure came from a status/chance bucket, not from the text.
    #
    # **So the provenance is rendered beside the number, not just the number.**
    # "2 GW (back 22 Nov)" and "~5 GW (no stated date)" are the same column and
    # very different claims, and only the first says anything about *which*
    # gameweeks -- which is exactly what a manager deciding on a deputy needs.
    _deadlines = gameweek_deadlines()
    _est, _src = [], []
    for _, r in show.iterrows():
        weeks, source = absence_estimate(r.get("News"), r.get("PlayPct"),
                                         r.get("Status"), deadlines=_deadlines)
        _news = str(r.get("News") or "")
        # **A departure is not an absence.** `status` 'u' covers both a loan or
        # permanent exit and an unavailable player, and the chance bucket then
        # reports a transferred player as "~5 GW" -- which reads as a man due
        # back next month. This is the same distinction `transfer_risk` draws:
        # the line is Premier League membership, not a return date.
        if _DEPARTED_RE.search(_news):
            _est.append("season"); _src.append("left the club")
            continue
        if source == ABSENCE_NONE or not weeks:
            _est.append(""); _src.append("")
            continue
        if source == ABSENCE_FROM_NEWS:
            _est.append("%d GW" % weeks); _src.append("stated")
        else:
            # Hedged on purpose: a bucket default is a guess about *duration*
            # and says nothing about which gameweeks. Printing it flat is how a
            # guess comes to read as a read.
            _est.append("~%d GW" % weeks)
            _src.append("no stated date")
    show = show.copy()
    show["Out For"] = _est
    show["Basis"] = _src

    cols = ["Display_Name","Team","Position","PlayPct","StatusBucket",
            "Out For","Basis","News","News_Added"]
    if "PL_Injury" in show.columns:
        show = show.rename(columns={"PL_Injury": "PL Injury"})
        show["PL Injury"] = show["PL Injury"].fillna("")
        cols.insert(8, "PL Injury")
    show = show[[c for c in cols if c in show.columns]].copy()
    show = show.rename(columns={"Display_Name": "Player"})
    show["PlayPct"] = show["PlayPct"].round(0).astype("Int64")

    render_styled_table(
        show,
        positive_color_cols=["PlayPct"],
        max_height=500,
    )


def show_injuries_page():
    st.header("\U0001FA79 Player Availability (Official FPL)")
    render_injuries_tab()
