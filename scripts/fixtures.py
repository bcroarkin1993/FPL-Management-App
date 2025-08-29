from datetime import datetime, timedelta
import pandas as pd
import random
import requests
import streamlit as st
from typing import Tuple
from scripts.utils import get_current_gameweek
from scripts.discord_alerts import send_transactions_reminder

# ---------- optional local tz (EST) ----------
try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/New_York")
except Exception:
    _TZ = None

# ---------- tiny CSS to reduce whitespace & hide indexes in styled tables ----------
_COMPACT_CSS = """
<style>
/* tighten vertical gaps */
h2, h3, h4 { margin: 0.4rem 0 0.6rem 0; }
section.main > div { padding-top: 0.4rem; }
.block-container { padding-top: 0.8rem; }
/* styled-table: hide row index & top-left blank cell */
table thead th.blank, table th.row_heading { display:none !important; }
/* smaller table font for dense look */
table { font-size: 0.95rem; }
</style>
"""

# ---------- FPL helpers ----------
def _get_teams_reference() -> Tuple[pd.DataFrame, dict, dict]:
    data = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/").json()
    teams_df = pd.DataFrame(data.get("teams", []))[["id", "name", "short_name"]].rename(
        columns={"name": "Team", "short_name": "Short"}
    )
    id_to_team = dict(zip(teams_df["id"], teams_df["Team"]))
    id_to_short = dict(zip(teams_df["id"], teams_df["Short"]))
    return teams_df, id_to_team, id_to_short

def _to_est_dt(utc_iso: str):
    if not utc_iso:
        return None
    try:
        dt_utc = datetime.fromisoformat(utc_iso.replace("Z", "+00:00"))
        return dt_utc.astimezone(_TZ) if _TZ else dt_utc
    except Exception:
        return None

def _fetch_fixtures_range(start_gw: int, end_gw: int) -> pd.DataFrame:
    all_rows = []
    for gw in range(start_gw, end_gw + 1):
        url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
        rows = requests.get(url).json()
        for r in rows:
            r["event"] = r.get("event", gw)
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if df.empty:
        return df
    df["KickoffDT"] = df.get("kickoff_time", pd.Series([None]*len(df))).apply(_to_est_dt)
    return df

def _parse_kickoff_utc(kickoff_str: str) -> datetime:
    """
    Parse FPL kickoff_time strings like '2024-12-03T19:30:00Z' to aware UTC datetime.
    """
    if not kickoff_str:
        raise ValueError("Missing kickoff_time")
    # Handle trailing 'Z'
    return datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))

def _next_upcoming_kickoff_utc(fixtures_raw: dict) -> datetime:
    """
    Given the raw fixtures dict (event -> list[match]), return the earliest upcoming UTC kickoff.
    Expects each fixture dict to have 'kickoff_time' and 'finished' keys.
    """
    upcoming = []
    for _, matches in fixtures_raw.items():
        for m in matches:
            # Skip finished or missing times
            if m.get("finished") is True:
                continue
            kt = m.get("kickoff_time")
            if kt:
                try:
                    upcoming.append(_parse_kickoff_utc(kt))
                except Exception:
                    continue
    if not upcoming:
        return None
    return min(upcoming)

def _compute_deadline_et(kickoff_utc: datetime) -> datetime:
    """
    Your rule: deadline is 25h30m before the first match kickoff.
    Returned in America/New_York (ET).
    """
    et = ZoneInfo("America/New_York")
    deadline_utc = kickoff_utc - timedelta(hours=25, minutes=30)
    return deadline_utc.astimezone(et)

def _compute_primary_alert_et(deadline_et: datetime, kickoff_et: datetime) -> datetime:
    """
    Heuristic to match your examples:
      - If the first match is on Friday (ET), primary reminder at Thu 10:00 ET.
      - If the first match is early Saturday (<=09:00 ET), primary reminder at Thu 18:00 ET.
      - Otherwise: 6 hours before the deadline (rounded to nearest 30m).
    """
    # Friday kickoff -> Thursday 10:00 ET
    if kickoff_et.weekday() == 4:  # Fri
        return deadline_et.replace(hour=10, minute=0, second=0, microsecond=0)

    # Early Saturday kickoff -> Thursday 18:00 ET
    if kickoff_et.weekday() == 5 and kickoff_et.hour <= 9:  # Sat <= 09:00
        # The deadline is Friday morning; "primary" is Thursday evening
        thursday = (deadline_et - timedelta(days=1)).date()
        return datetime.combine(thursday, datetime.min.time(), tzinfo=deadline_et.tzinfo).replace(hour=18)

    # Default: 6 hours before deadline (rounded to nearest 30 minutes)
    cand = deadline_et - timedelta(hours=6)
    minute = 0 if cand.minute < 30 else 30
    return cand.replace(minute=minute, second=0, microsecond=0)

def _fmt_et(dt: datetime) -> str:
    """Nicely format ET timestamps for Discord/UI."""
    return dt.strftime("%a %b %d, %I:%M %p ET")

def _styler_hide_index(styler):
    """
    Hide the index on a pandas Styler across pandas versions & Streamlit.

    Order of attempts:
      1) pandas >=1.4: styler.hide(axis="index")
      2) older pandas: styler.hide_index()
      3) CSS fallback: hide header + data index cells
    """
    # 1) New API
    try:
        return styler.hide(axis="index")
    except Exception:
        pass

    # 2) Older API
    if hasattr(styler, "hide_index"):
        try:
            return styler.hide_index()
        except Exception:
            pass

    # 3) CSS fallback that works with Streamlit's HTML rendering
    try:
        styler = styler.set_table_styles(
            [
                {"selector": "th.row_heading",           "props": [("display", "none")]},
                {"selector": "th.blank",                 "props": [("display", "none")]},
                {"selector": "th.blank.level0",          "props": [("display", "none")]},
                {"selector": "td.row_heading",           "props": [("display", "none")]},
                {"selector": "tbody tr th:first-child",  "props": [("display", "none")]},
            ],
            overwrite=False,
        )
    except Exception:
        # If anything goes sideways, just return the original styler
        return styler

    return styler

# ---------- centered FDR color (1=green, 3=white center, 5=red) ----------
def _hex_to_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def _rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)
def _blend(c1, c2, t):
    r1,g1,b1=_hex_to_rgb(c1); r2,g2,b2=_hex_to_rgb(c2)
    r=int(round(r1+(r2-r1)*t)); g=int(round(g1+(g2-g1)*t)); b=int(round(b1+(b2-b1)*t))
    return _rgb_to_hex((r,g,b))

def _fdr_color_centered(val):
    if pd.isna(val): return "#ffffff"
    try: x=float(val)
    except: return "#ffffff"
    x=min(5.0,max(1.0,x))
    if x<=3.0:  # green -> white
        t=(x-1.0)/2.0
        return _blend("#2ecc71", "#ffffff", t)
    else:       # white -> red
        t=(x-3.0)/2.0
        return _blend("#ffffff", "#e74c3c", t)

def _fdr_bg(val):
    if pd.isna(val): return ""
    return f"background-color: {_fdr_color_centered(val)}; color: #000;"

# ---------- tables ----------
def _make_match_table(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    _, id_to_team, _ = _get_teams_reference()
    if fixtures_df.empty:
        return pd.DataFrame(columns=["Gameweek","Date (EST)","Time (EST)","Home Team","Away Team","Home FDR","Away FDR"])
    df = fixtures_df.copy().dropna(subset=["KickoffDT"]).sort_values("KickoffDT")

    def fmt_date(dt): return dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else ""
    def fmt_time(dt): return dt.strftime("%I:%M %p") if isinstance(dt, datetime) else ""

    out = []
    for _, r in df.iterrows():
        dt = r["KickoffDT"]
        out.append({
            "Gameweek": int(r.get("event")),
            "Date (EST)": fmt_date(dt),
            "Time (EST)": fmt_time(dt),
            "Home Team": id_to_team.get(r.get("team_h"), f"Team {r.get('team_h')}"),
            "Away Team": id_to_team.get(r.get("team_a"), f"Team {r.get('team_a')}"),
            "Home FDR": pd.to_numeric(r.get("team_h_difficulty"), errors="coerce"),
            "Away FDR": pd.to_numeric(r.get("team_a_difficulty"), errors="coerce")
        })
    return pd.DataFrame(out)

def _make_club_fixtures_long(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    _, id_to_team, _ = _get_teams_reference()
    if fixtures_df.empty:
        return pd.DataFrame(columns=["Gameweek","Club","Opponent","Venue","Date (EST)","Time (EST)","FDR","KickoffDT"])

    rows=[]
    for _, r in fixtures_df.iterrows():
        dt=r.get("KickoffDT")
        date_str = dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else ""
        time_str = dt.strftime("%I:%M %p") if isinstance(dt, datetime) else ""
        rows += [
            {"Gameweek": int(r["event"]), "Club": id_to_team.get(r["team_h"]), "Opponent": id_to_team.get(r["team_a"]),
             "Venue":"H", "Date (EST)":date_str, "Time (EST)":time_str,
             "FDR": pd.to_numeric(r.get("team_h_difficulty"), errors="coerce"), "KickoffDT": dt},
            {"Gameweek": int(r["event"]), "Club": id_to_team.get(r["team_a"]), "Opponent": id_to_team.get(r["team_h"]),
             "Venue":"A", "Date (EST)":date_str, "Time (EST)":time_str,
             "FDR": pd.to_numeric(r.get("team_a_difficulty"), errors="coerce"), "KickoffDT": dt},
        ]
    df = pd.DataFrame(rows)
    return df.sort_values(["KickoffDT","Club"]).reset_index(drop=True)

# ---------- text view (bottom): bold GW, medium date, small fixtures, two days per row ----------
def _render_text_schedule_two_cols(fixtures_df: pd.DataFrame):
    _, id_to_team, _ = _get_teams_reference()
    df = fixtures_df.dropna(subset=["KickoffDT"]).copy()
    if df.empty:
        st.info("No fixtures to show for the selected range.")
        return

    df = df.sort_values(["event","KickoffDT"])
    for gw, chunk in df.groupby("event", sort=False):
        st.markdown(f"<div style='font-size:1.5rem;font-weight:800;margin:0.4rem 0 0.6rem'>Gameweek {int(gw)}</div>", unsafe_allow_html=True)
        chunk["DayKey"] = chunk["KickoffDT"].dt.date
        day_keys = list(dict.fromkeys(chunk.sort_values("KickoffDT")["DayKey"].tolist()))
        for i in range(0, len(day_keys), 2):
            c1, c2 = st.columns(2)
            for idx, day in enumerate(day_keys[i:i+2]):
                col = c1 if idx == 0 else c2
                day_block = chunk[chunk["DayKey"] == day].sort_values("KickoffDT")
                if day_block.empty:
                    continue
                date_hdr = pd.Timestamp(day).strftime("%A %d %B %Y")
                col.markdown(f"<div style='font-size:1.05rem;font-weight:700;margin:0.3rem 0'>{date_hdr}</div>", unsafe_allow_html=True)
                for _, r in day_block.iterrows():
                    t = r["KickoffDT"].strftime("%I:%M %p")
                    home = id_to_team.get(r["team_h"], f"Team {r['team_h']}")
                    away = id_to_team.get(r["team_a"], f"Team {r['team_a']}")
                    if pd.notna(r.get("team_h_score")) and pd.notna(r.get("team_a_score")):
                        line = f"<div style='font-size:0.95rem'> <b>{home}</b> <b>{int(r['team_h_score'])} - {int(r['team_a_score'])}</b> <b>{away}</b> · {t}</div>"
                    else:
                        line = f"<div style='font-size:0.95rem'> <b>{home}</b> vs <b>{away}</b> · {t}</div>"
                    col.markdown(line, unsafe_allow_html=True)
        st.markdown("---")

# ============== MAIN SECTION (filters at top, text fixtures at bottom) ==============
def show_club_fixtures_section():
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)
    st.header("📅 Club Fixtures & Difficulty (FDR)")

    # ---- Filters (TOP) ----
    try:
        current_gw = get_current_gameweek()  # use your helper if available
    except Exception:
        current_gw = None
    current_gw = current_gw or 1

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        start_gw = st.number_input("Start Gameweek", min_value=1, max_value=38, value=int(current_gw), step=1)
    with f2:
        weeks = st.slider("How many weeks?", min_value=1, max_value=10, value=5, step=1)
    with f3:
        # global team filter (moved out of Average FDR)
        teams_df, id_to_team, _ = _get_teams_reference()
        all_clubs = sorted(teams_df["Team"].tolist())
        team_filter = st.multiselect("Filter clubs (optional)", options=all_clubs, default=[])

    fixtures_raw = _fetch_fixtures_range(start_gw, start_gw + weeks - 1)
    club_long = _make_club_fixtures_long(fixtures_raw)

    # --- Discord Alerts panel ---
    st.markdown("### 🔔 Discord Alerts")

    # 1) Find next kickoff
    next_ko_utc = _next_upcoming_kickoff_utc(fixtures_raw)
    if not next_ko_utc:
        st.info("No upcoming fixtures found — alerts will show once new fixtures are listed.")
    else:
        et = ZoneInfo("America/New_York")
        kickoff_et = next_ko_utc.astimezone(et)

        # 2) Compute deadline + suggested alert times
        deadline_et = _compute_deadline_et(next_ko_utc)
        primary_et = _compute_primary_alert_et(deadline_et, kickoff_et)
        last30_et = deadline_et - timedelta(minutes=30)

        # 3) Show info
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Next Kickoff (ET)", _fmt_et(kickoff_et))
        with c2:
            st.metric("Transaction Deadline (ET)", _fmt_et(deadline_et))
        with c3:
            st.metric("Last 30-min Reminder", _fmt_et(last30_et))

        st.caption("Deadline uses kickoff − 25h30m per your rule of thumb.")

        # 4) Buttons to send reminders now
        col_a, col_b = st.columns(2)
        if col_a.button("Send Primary Reminder Now"):
            ok = send_transactions_reminder(
                league_name="FPL Draft",
                deadline_str_local=_fmt_et(deadline_et),
                notes=f"Primary reminder (suggested time: {_fmt_et(primary_et)}).",
                # webhook_url=...  # optional override; otherwise discord_alerts pulls from config/.env
            )
            st.success("Primary reminder sent!") if ok else st.error("Discord rejected the message.")

        if col_b.button("Send 30-Minute Reminder Now"):
            ok = send_transactions_reminder(
                league_name="FPL Draft",
                deadline_str_local=_fmt_et(deadline_et),
                notes="30 minutes to go!",
            )
            st.success("30-minute reminder sent!") if ok else st.error("Discord rejected the message.")

    # ---- Upper content (tables) ----
    st.subheader("All Clubs — Sorted Fixture List")

    match_tbl = _make_match_table(fixtures_raw)

    # Optional filter
    if team_filter:
        match_tbl = match_tbl[
            match_tbl["Home Team"].isin(team_filter) | match_tbl["Away Team"].isin(team_filter)
            ]

    # Ensure numeric
    match_tbl["Home FDR"] = pd.to_numeric(match_tbl["Home FDR"], errors="coerce")
    match_tbl["Away FDR"] = pd.to_numeric(match_tbl["Away FDR"], errors="coerce")

    st.dataframe(
        match_tbl.reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Home FDR": st.column_config.ProgressColumn(
                "Home FDR", min_value=1.0, max_value=5.0, format="%.1f"
            ),
            "Away FDR": st.column_config.ProgressColumn(
                "Away FDR", min_value=1.0, max_value=5.0, format="%.1f"
            ),
        },
    )

    # ---- Text Fixtures Table (MIDDLE) ----
    left, right = st.columns([1.75, 1])

    # All matches (one row per match), with FDR color for both sides, ascending by kickoff
    with left:
        st.subheader("🔎 Team Fixture Table")
        clubs_available = sorted(club_long["Club"].dropna().unique().tolist())
        if clubs_available:
            default_team = random.choice(clubs_available)
            default_idx = clubs_available.index(default_team)
            team_pick = st.selectbox("Select a team", options=clubs_available, index=default_idx)
            team_tbl = (club_long[club_long["Club"] == team_pick]
                        [["Gameweek", "Date (EST)", "Time (EST)", "Opponent", "Venue", "FDR", "KickoffDT"]]
                        .sort_values("KickoffDT")
                        .drop(columns=["KickoffDT"])
                        .reset_index(drop=True))
            styled_team_tbl = _styler_hide_index(team_tbl.style.map(_fdr_bg, subset=["FDR"]))
            st.write(styled_team_tbl, unsafe_allow_html=True)
        else:
            st.info("No team fixtures in the selected window.")

    with right:
        # Average Upcoming FDR (sorted ascending)
        st.subheader("Average Upcoming FDR")
        if team_filter:
            club_long = club_long[club_long["Club"].isin(team_filter)]

        fdr_summary = (
            club_long.dropna(subset=["FDR"])
                .groupby("Club", as_index=False)
                .agg(Avg_FDR=("FDR", "mean"), Matches=("FDR", "count"))
        )

        # ensure numeric + round values
        fdr_summary["Avg_FDR"] = pd.to_numeric(fdr_summary["Avg_FDR"], errors="coerce").round(1)

        # sort by the numeric column
        fdr_summary = fdr_summary.sort_values("Avg_FDR", ascending=True).reset_index(drop=True)

        # style: apply background & enforce 1-decimal display
        styled_summary = (
            fdr_summary.style
                .format({"Avg_FDR": "{:.1f}"})  # <-- enforce display rounding
                .applymap(_fdr_bg, subset=["Avg_FDR"])  # keep your color gradient
        )
        styled_summary = _styler_hide_index(styled_summary)

        st.write(styled_summary, unsafe_allow_html=True)

    # ---- Text Fixtures (BOTTOM) ----
    st.subheader("🗓️ Fixtures (Text View)")
    # (No FDR shown here; two-day columns; GW title big > date medium > fixtures small)
    if fixtures_raw.empty:
        st.info("No fixtures to show for the selected range.")
    else:
        # apply team filter if any
        if team_filter:
            filtered = fixtures_raw[
                fixtures_raw["team_h"].isin(teams_df[teams_df["Team"].isin(team_filter)]["id"]) |
                fixtures_raw["team_a"].isin(teams_df[teams_df["Team"].isin(team_filter)]["id"])
            ]
        else:
            filtered = fixtures_raw
        _render_text_schedule_two_cols(filtered)
