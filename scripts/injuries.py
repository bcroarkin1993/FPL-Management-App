# injuries_official.py
import requests
import pandas as pd
import streamlit as st

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

def show_injuries_page():
    st.header("🩹 Player Availability (Official FPL)")
    df = get_fpl_availability_df()
    if df.empty:
        st.warning("No data from FPL. Try again in a bit.")
        return

    # Filters on page (not sidebar)
    c1, c2, c3 = st.columns(3)
    teams = sorted(df["Team"].dropna().unique().tolist())
    poss  = ["G","D","M","F"]
    team_sel = c1.multiselect("Teams", teams, default=None)
    pos_sel  = c2.multiselect("Positions", poss, default=poss)
    min_play = c3.slider("Min Play %", 0, 100, 0, 5)

    show = df.copy()
    if team_sel:
        show = show[show["Team"].isin(team_sel)]
    if pos_sel:
        show = show[show["Position"].isin(pos_sel)]
    show = show[show["PlayPct"].fillna(0) >= min_play]

    # Nice view
    show = show[["Player","Web_Name","Team","Position","PlayPct","StatusBucket","News","News_Added"]].copy()
    show["PlayPct"] = show["PlayPct"].round(0).astype("Int64")

    st.dataframe(show, use_container_width=True)

    # Tip: color by PlayPct using st.data_editor (optional)
    st.caption("Tip: Use st.data_editor with a progress column if you want in-cell color bars.")
