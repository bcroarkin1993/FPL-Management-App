# injuries_official.py
import requests
import pandas as pd
import streamlit as st
from scripts.common.error_helpers import get_logger
from scripts.common.styled_tables import render_styled_table

_logger = get_logger("fpl_app.injuries")

# FPL element_type -> position letter
_POS_LETTER = {1:"G", 2:"D", 3:"M", 4:"F"}

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
    """
    js = requests.get("https://draft.premierleague.com/api/bootstrap-static", timeout=30).json()
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
    df = pd.DataFrame(rows)
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
        show = flagged[["Player", "Team", "Position", "PL_Injury"]].copy()
        show["PL_Injury"] = show["PL_Injury"].replace("", "Unspecified")
        show = show.rename(columns={"PL_Injury": "PL Injury"})
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
    cols = ["Player","Web_Name","Team","Position","PlayPct","StatusBucket","News","News_Added"]
    if "PL_Injury" in show.columns:
        show = show.rename(columns={"PL_Injury": "PL Injury"})
        show["PL Injury"] = show["PL Injury"].fillna("")
        cols.insert(6, "PL Injury")
    show = show[cols].copy()
    show["PlayPct"] = show["PlayPct"].round(0).astype("Int64")

    render_styled_table(
        show,
        positive_color_cols=["PlayPct"],
        max_height=500,
    )


def show_injuries_page():
    st.header("\U0001FA79 Player Availability (Official FPL)")
    render_injuries_tab()
