from datetime import datetime, timedelta, timezone
import pandas as pd
import requests
import streamlit as st
from typing import Tuple
from scripts.common.api import get_current_gameweek

# ---------- optional local tz (EST) ----------
try:
    from zoneinfo import ZoneInfo

    _TZ = ZoneInfo("America/New_York")
except Exception:
    _TZ = None

# ---------- CSS to fix table width and alignment ----------
_COMPACT_CSS = """
<style>
/* Force tables to take full width */
table { 
    width: 100% !important; 
    font-size: 0.95rem;
    border-collapse: collapse !important;
}

/* Fix header alignment */
th {
    text-align: center !important;
    background-color: #f0f2f6;
    padding: 8px !important;
}

/* Center cells */
td {
    text-align: center !important;
    padding: 8px !important;
}

/* Tighten vertical gaps */
h2, h3, h4 { margin: 0.4rem 0 0.6rem 0; }
section.main > div { padding-top: 0.4rem; }
.block-container { padding-top: 0.8rem; }
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
    df["KickoffDT"] = df.get("kickoff_time", pd.Series([None] * len(df))).apply(_to_est_dt)
    return df


# ---------- FDR / Color Helpers ----------
def _hex_to_rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)


def _blend(c1, c2, t):
    r1, g1, b1 = _hex_to_rgb(c1);
    r2, g2, b2 = _hex_to_rgb(c2)
    r = int(round(r1 + (r2 - r1) * t));
    g = int(round(g1 + (g2 - g1) * t));
    b = int(round(b1 + (b2 - b1) * t))
    return _rgb_to_hex((r, g, b))


def _fdr_color_centered(val):
    if pd.isna(val): return "#ffffff"
    try:
        x = float(val)
    except:
        return "#ffffff"
    x = min(5.0, max(1.0, x))
    if x <= 3.0:  # green -> white
        t = (x - 1.0) / 2.0
        return _blend("#2ecc71", "#ffffff", t)
    else:  # white -> red
        t = (x - 3.0) / 2.0
        return _blend("#ffffff", "#e74c3c", t)


def _fdr_bg(val):
    if pd.isna(val): return ""
    return f"background-color: {_fdr_color_centered(val)}; color: #000;"


# ---------- LOGIC: Fixture Grid ----------

def get_fixture_difficulty_grid(weeks=5):
    """
    Generates the grid needed for the overview expander.
    Returns: (display_df, difficulty_df, avg_series)
    """
    # 1. Determine Range
    start_gw = get_current_gameweek()
    end_gw = start_gw + weeks - 1

    # 2. Fetch Data
    fixtures = _fetch_fixtures_range(start_gw, end_gw)
    _, _, id_to_short = _get_teams_reference()

    if fixtures.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.Series()

    # 3. Build Grid
    grid_data = {}  # {Team: {GW: (Text, Diff)}}

    # Initialize all teams
    for team_id in id_to_short.keys():
        team_name = id_to_short[team_id]
        grid_data[team_name] = {}

    for _, row in fixtures.iterrows():
        gw = row['event']
        h_id = row['team_h']
        a_id = row['team_a']
        h_short = id_to_short.get(h_id, str(h_id))
        a_short = id_to_short.get(a_id, str(a_id))
        h_diff = row['team_h_difficulty']
        a_diff = row['team_a_difficulty']

        # Home Entry
        grid_data[h_short][gw] = (f"{a_short} (H)", h_diff)
        # Away Entry
        grid_data[a_short][gw] = (f"{h_short} (A)", a_diff)

    # 4. Convert to DataFrames
    disp_rows = []
    diff_rows = []

    # Ensure we cover all requested gameweeks
    gw_cols = list(range(start_gw, end_gw + 1))

    for team, data in grid_data.items():
        disp_row = {"Team": team}
        diff_row = {"Team": team}

        for gw in gw_cols:
            if gw in data:
                text, diff = data[gw]
                disp_row[gw] = text
                diff_row[gw] = diff
            else:
                disp_row[gw] = "-"
                diff_row[gw] = 0  # Neutral

        disp_rows.append(disp_row)
        diff_rows.append(diff_row)

    disp_df = pd.DataFrame(disp_rows)
    diff_df = pd.DataFrame(diff_rows)

    # 5. Sorting Logic
    # Calculate Average Difficulty (excluding blanks)
    diff_numeric = diff_df.set_index("Team").replace(0, pd.NA)
    avg_series = diff_numeric.mean(axis=1).sort_values()

    # Sort the main DFs by this average
    sorted_teams = avg_series.index.tolist()

    # Set 'Team' as index first to reindex, but then reset it
    # so 'Team' becomes a regular column for clean display
    disp_df = disp_df.set_index("Team").reindex(sorted_teams).reset_index()
    diff_df = diff_df.set_index("Team").reindex(sorted_teams).reset_index()

    return disp_df, diff_df, avg_series


def style_fixture_difficulty(disp_df, diff_df):
    """
    Applies background colors to disp_df based on values in diff_df.
    """
    if disp_df.empty:
        return disp_df.style

    # Create a DataFrame of CSS strings matching the shape of disp_df
    css_df = pd.DataFrame("", index=disp_df.index, columns=disp_df.columns)

    # We want to style columns that are Gameweeks (integers), skip 'Team'
    gw_cols = [c for c in disp_df.columns if isinstance(c, int)]

    for idx in disp_df.index:
        for col in gw_cols:
            # diff_df has the same shape/index
            val = diff_df.at[idx, col]
            # If value is 0 (blank week), no color
            if val != 0:
                css_df.at[idx, col] = _fdr_color_centered(val)

    # Function to apply the styles
    def apply_style(x):
        return css_df.applymap(lambda v: f"background-color: {v}; color: black;" if v else "")

    # Apply style and hide index
    styler = disp_df.style.apply(
        lambda x: css_df.applymap(lambda v: f"background-color: {v}; color: black;" if v else ""), axis=None)

    # Streamlit specific: We hide the numerical index to make 'Team' look like the primary key
    styler.hide(axis="index")

    return styler


# ---------- tables ----------
def _make_match_table(fixtures_df: pd.DataFrame) -> pd.DataFrame:
    _, id_to_team, _ = _get_teams_reference()
    if fixtures_df.empty:
        return pd.DataFrame(
            columns=["Gameweek", "Date (EST)", "Time (EST)", "Home Team", "Away Team", "Home FDR", "Away FDR"])
    df = fixtures_df.copy().dropna(subset=["KickoffDT"]).sort_values("KickoffDT")

    def fmt_date(dt):
        return dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else ""

    def fmt_time(dt):
        return dt.strftime("%I:%M %p") if isinstance(dt, datetime) else ""

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
        return pd.DataFrame(
            columns=["Gameweek", "Club", "Opponent", "Venue", "Date (EST)", "Time (EST)", "FDR", "KickoffDT"])

    rows = []
    for _, r in fixtures_df.iterrows():
        dt = r.get("KickoffDT")
        date_str = dt.strftime("%Y-%m-%d") if isinstance(dt, datetime) else ""
        time_str = dt.strftime("%I:%M %p") if isinstance(dt, datetime) else ""
        rows += [
            {"Gameweek": int(r["event"]), "Club": id_to_team.get(r["team_h"]), "Opponent": id_to_team.get(r["team_a"]),
             "Venue": "H", "Date (EST)": date_str, "Time (EST)": time_str,
             "FDR": pd.to_numeric(r.get("team_h_difficulty"), errors="coerce"), "KickoffDT": dt},
            {"Gameweek": int(r["event"]), "Club": id_to_team.get(r["team_a"]), "Opponent": id_to_team.get(r["team_h"]),
             "Venue": "A", "Date (EST)": date_str, "Time (EST)": time_str,
             "FDR": pd.to_numeric(r.get("team_a_difficulty"), errors="coerce"), "KickoffDT": dt},
        ]
    df = pd.DataFrame(rows)
    return df.sort_values(["KickoffDT", "Club"]).reset_index(drop=True)


# ---------- text view (bottom) ----------
def _render_text_schedule_two_cols(fixtures_df: pd.DataFrame):
    _, id_to_team, _ = _get_teams_reference()
    df = fixtures_df.dropna(subset=["KickoffDT"]).copy()
    if df.empty:
        st.info("No fixtures to show for the selected range.")
        return

    df = df.sort_values(["event", "KickoffDT"])
    for gw, chunk in df.groupby("event", sort=False):
        st.markdown(f"<div style='font-size:1.5rem;font-weight:800;margin:0.4rem 0 0.6rem'>Gameweek {int(gw)}</div>",
                    unsafe_allow_html=True)
        chunk["DayKey"] = chunk["KickoffDT"].dt.date
        day_keys = list(dict.fromkeys(chunk.sort_values("KickoffDT")["DayKey"].tolist()))
        for i in range(0, len(day_keys), 2):
            c1, c2 = st.columns(2)
            for idx, day in enumerate(day_keys[i:i + 2]):
                col = c1 if idx == 0 else c2
                day_block = chunk[chunk["DayKey"] == day].sort_values("KickoffDT")
                if day_block.empty:
                    continue
                date_hdr = pd.Timestamp(day).strftime("%A %d %B %Y")
                col.markdown(f"<div style='font-size:1.05rem;font-weight:700;margin:0.3rem 0'>{date_hdr}</div>",
                             unsafe_allow_html=True)
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


# ============== MAIN SECTION ==============
def show_club_fixtures_section():
    # Inject CSS
    st.markdown(_COMPACT_CSS, unsafe_allow_html=True)

    st.header("📅 Club Fixtures & Difficulty (FDR)")

    # --- Fixture Difficulty Grid ---
    with st.expander("Fixture Difficulty Grid (overview)", expanded=True):
        weeks = st.slider("Horizon (GWs)", 1, 10, 6, 1, key="fdr_horizon")
        disp, diffs, avg = get_fixture_difficulty_grid(weeks=weeks)

        if not disp.empty:
            # Display styled table
            st.write(style_fixture_difficulty(disp, diffs).to_html(), unsafe_allow_html=True)
            st.caption(
                "Rows = Team; cells = upcoming opponents (H/A). Colors = FPL Difficulty (Green=Easy, Red=Hard). Sorted by easiest schedule.")
        else:
            st.warning("No fixture data available.")
    # ------------------------------------

    # ---- Filters ----
    current_gw = get_current_gameweek()

    # ---- Upper content (tables) ----
    st.subheader("All Clubs — Sorted Fixture List")

    f1, f2, f3 = st.columns([1, 1, 1])
    with f1:
        start_gw = st.number_input("Start Gameweek", min_value=1, max_value=38, value=int(current_gw), step=1)
    with f2:
        weeks_list = st.slider("How many weeks?", min_value=1, max_value=10, value=5, step=1)
    with f3:
        # global team filter
        teams_df, id_to_team, _ = _get_teams_reference()
        all_clubs = sorted(teams_df["Team"].tolist())
        team_filter = st.multiselect("Filter clubs (optional)", options=all_clubs, default=[])

    fixtures_raw = _fetch_fixtures_range(start_gw, start_gw + weeks_list - 1)

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

    # ---- Text Fixtures (BOTTOM) ----
    st.subheader("🗓️ Fixtures (Text View)")
    if fixtures_raw.empty:
        st.info("No fixtures to show for the selected range.")
    else:
        if team_filter:
            filtered = fixtures_raw[
                fixtures_raw["team_h"].isin(teams_df[teams_df["Team"].isin(team_filter)]["id"]) |
                fixtures_raw["team_a"].isin(teams_df[teams_df["Team"].isin(team_filter)]["id"])
                ]
        else:
            filtered = fixtures_raw
        _render_text_schedule_two_cols(filtered)