import os
import config
import math
import pandas as pd
import streamlit as st

# Imports
from scripts.common.api import (
    get_draft_league_details,
    get_rotowire_player_projections,
    get_current_gameweek,
    get_draft_league_teams,
    get_historical_team_scores,
    get_league_player_ownership
)
from scripts.common.utils import (
    merge_fpl_players_and_projections,
    select_optimal_lineup as find_optimal_lineup,
    format_team_name,
    normalize_apostrophes,
    clean_fpl_player_names,
    get_team_id_by_name
)


# ==============================================================================
# HELPERS
# ==============================================================================

def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _estimate_score_std(league_id: int) -> tuple[float, int]:
    try:
        hist = get_historical_team_scores(league_id)
    except Exception:
        hist = None
    if isinstance(hist, pd.DataFrame) and not hist.empty:
        col = 'total_points' if 'total_points' in hist.columns else ('score' if 'score' in hist.columns else None)
        if col:
            s = pd.to_numeric(hist[col], errors='coerce').dropna()
            if len(s) >= 2:
                return float(s.std(ddof=1)), int(len(s))

    # Try CSV from config or default path
    try:
        csv_path = getattr(config, 'HISTORICAL_SCORES_CSV', 'data/historical_team_scores.csv')
        df = pd.read_csv(csv_path)
        col = 'total_points' if 'total_points' in df.columns else ('score' if 'score' in df.columns else None)
        if col:
            s = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(s) >= 2:
                return float(s.std(ddof=1)), int(len(s))
    except Exception:
        pass
    return 15.0, 0


def _render_winprob_bar(team1_name: str, team2_name: str, p_team1: float):
    p1 = max(0.0, min(100.0, round(p_team1 * 100, 1)))
    p2 = round(100.0 - p1, 1)
    html = f"""
    <style>
      .wpb-wrap {{ margin-top: 0.25rem; margin-bottom: 0.5rem; }}
      .wpb-labels, .wpb-bar {{ display: grid; grid-template-columns: {p1}% {p2}%; gap: 0; width: 100%; }}
      .wpb-labels div {{ text-align: center; font-weight: 600; font-size: 0.95rem; line-height: 1.2; white-space: nowrap; }}
      .wpb-bar {{ height: 36px; border-radius: 9999px; overflow: hidden; box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08); }}
      .wpb-left {{ background: #2563eb; }} .wpb-right {{ background: #dc2626; }} .wpb-subtle {{ color: rgba(0,0,0,0.65); }}
    </style>
    <div class="wpb-wrap">
      <div class="wpb-labels"><div class="wpb-subtle">{team1_name} {p1}%</div><div class="wpb-subtle">{p2}% {team2_name}</div></div>
      <div class="wpb-bar" role="img"><div class="wpb-left"></div><div class="wpb-right"></div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def get_gameweek_fixtures(league_id, gameweek):
    details = get_draft_league_details(league_id)
    matches = details.get("matches", [])
    entries = details.get("league_entries", [])
    id_to_name = {e['id']: e['entry_name'] for e in entries}
    fixtures = []
    for m in matches:
        if m.get("event") == gameweek:
            n1 = id_to_name.get(m.get("league_entry_1"), "Average")
            n2 = id_to_name.get(m.get("league_entry_2"), "Average")
            fixtures.append(f"{n1} vs {n2}")
    return fixtures


def analyze_fixture_projections(fixture, league_id, projections_df, debug=False):
    if debug: print(f"\n--- DEBUG: Analyzing fixture: {fixture} ---")

    fixture = normalize_apostrophes(fixture)
    team1_name = fixture.split(' vs ')[0].split(' (')[0].strip()
    team2_name = fixture.split(' vs ')[1].split(' (')[0].strip()

    team1_id = get_team_id_by_name(league_id, team1_name)
    team2_id = get_team_id_by_name(league_id, team2_name)
    if debug: print(f"DEBUG: IDs found -> Team1: {team1_id}, Team2: {team2_id}")

    # Load Ownership
    ownership = get_league_player_ownership(league_id)
    if debug:
        if not ownership:
            print("DEBUG: Ownership dictionary is EMPTY.")
        else:
            print(f"DEBUG: Ownership keys (Global IDs): {list(ownership.keys())}")

    def get_roster_from_ownership(tid):
        if not tid or tid not in ownership:
            if debug: print(f"DEBUG: Team ID {tid} not in ownership keys.")
            return pd.DataFrame(columns=['Player', 'Position'])

        data = ownership[tid]
        rows = []
        for pos, players in data.get("players", {}).items():
            for p in players:
                rows.append({"Player": p, "Position": pos})

        df = pd.DataFrame(rows)
        if debug: print(f"DEBUG: Roster for {tid} created. Shape: {df.shape}. Columns: {df.columns.tolist()}")
        return df

    team1_composition = get_roster_from_ownership(team1_id)
    team2_composition = get_roster_from_ownership(team2_id)

    # Prepare Projections
    if not projections_df.empty:
        projections_df['Player'] = projections_df['Player'].apply(clean_fpl_player_names)

    # CRITICAL: Exclude 'Position' from projections to avoid collision
    proj_subset = projections_df[['Player', 'Points', 'Matchup', 'Pos Rank']].copy()
    if debug: print(f"DEBUG: Projection subset columns: {proj_subset.columns.tolist()}")

    # Merge
    if debug: print("DEBUG: Merging Team 1...")
    team1_df = merge_fpl_players_and_projections(team1_composition, proj_subset)
    if debug: print(f"DEBUG: Team 1 Merge Result Columns: {team1_df.columns.tolist()}")

    if debug: print("DEBUG: Merging Team 2...")
    team2_df = merge_fpl_players_and_projections(team2_composition, proj_subset)
    if debug: print(f"DEBUG: Team 2 Merge Result Columns: {team2_df.columns.tolist()}")

    # Handle missing Position column (Safety Check)
    for i, df in enumerate([team1_df, team2_df]):
        team_num = i + 1
        if 'Position' not in df.columns:
            if debug: print(f"ERROR: 'Position' column MISSING in Team {team_num} DF.")
            if 'Position_x' in df.columns:
                if debug: print(f"DEBUG: Restoring Position from Position_x for Team {team_num}...")
                df.rename(columns={'Position_x': 'Position'}, inplace=True)
            elif 'Position_y' in df.columns:
                if debug: print(f"DEBUG: Restoring Position from Position_y for Team {team_num}...")
                df.rename(columns={'Position_y': 'Position'}, inplace=True)
            else:
                if debug: print(f"CRITICAL: Creating empty DF for Team {team_num} to prevent crash.")
                if team_num == 1:
                    team1_df = pd.DataFrame(columns=['Player', 'Position', 'Points', 'Is_Starter'])
                else:
                    team2_df = pd.DataFrame(columns=['Player', 'Position', 'Points', 'Is_Starter'])

    # Fill NaNs
    if 'Points' in team1_df.columns: team1_df['Points'] = team1_df['Points'].fillna(0.0)
    if 'Points' in team2_df.columns: team2_df['Points'] = team2_df['Points'].fillna(0.0)

    # Optimize
    if debug: print("DEBUG: Running find_optimal_lineup...")
    if not team1_df.empty and 'Position' in team1_df.columns:
        team1_df = find_optimal_lineup(team1_df)
    else:
        team1_df['Is_Starter'] = False

    if not team2_df.empty and 'Position' in team2_df.columns:
        team2_df = find_optimal_lineup(team2_df)
    else:
        team2_df['Is_Starter'] = False

    # Filter to starters
    if 'Is_Starter' in team1_df.columns:
        team1_df = team1_df[team1_df['Is_Starter']].copy()
    if 'Is_Starter' in team2_df.columns:
        team2_df = team2_df[team2_df['Is_Starter']].copy()

    # Sort
    pos_order = ['G', 'D', 'M', 'F']
    if not team1_df.empty:
        team1_df['Position'] = pd.Categorical(team1_df['Position'], categories=pos_order, ordered=True)
        team1_df = team1_df.sort_values(by=['Position', 'Points'], ascending=[True, False])

    if not team2_df.empty:
        team2_df['Position'] = pd.Categorical(team2_df['Position'], categories=pos_order, ordered=True)
        team2_df = team2_df.sort_values(by=['Position', 'Points'], ascending=[True, False])

    return team1_df, team2_df, team1_name, team2_name


# ==============================================================================
# MAIN PAGE
# ==============================================================================

def show_fixture_projections_page():
    st.title("Upcoming Fixtures & Projections")

    # READ FROM .ENV VIA OS
    # Assuming RUN_TYPE is loaded into env variables
    default_debug = os.getenv("RUN_TYPE", "PRODUCTION") == "DEBUG"
    debug_mode = st.checkbox("Enable Debug Mode (Console Logs)", value=default_debug)

    gw = get_current_gameweek()
    league_id = config.FPL_DRAFT_LEAGUE_ID
    fixtures = get_gameweek_fixtures(league_id, gw)

    if fixtures:
        st.subheader(f"Gameweek {gw} Fixtures")
        selected_fixture = st.selectbox("Select a fixture to analyze:", fixtures)
    else:
        st.warning(f"No fixtures found for Gameweek {gw}")
        return

    st.subheader("Match Projections")
    with st.spinner("Loading projections..."):
        projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    if selected_fixture:
        res = analyze_fixture_projections(selected_fixture, league_id, projections, debug=debug_mode)

        if res:
            t1_df, t2_df, t1_name, t2_name = res

            s1 = t1_df['Points'].sum() if not t1_df.empty else 0.0
            s2 = t2_df['Points'].sum() if not t2_df.empty else 0.0

            sigma, n_hist = _estimate_score_std(league_id)
            denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0
            z = (s1 - s2) / denom
            p_team1 = _normal_cdf(z)

            st.subheader("Win Probability")
            _render_winprob_bar(format_team_name(t1_name), format_team_name(t2_name), p_team1)
            hist_note = f"σ≈{sigma:.2f} from {n_hist} games" if n_hist > 0 else f"σ≈{sigma:.2f} (default)"
            st.caption(f"Model: P(A>B) = Φ((μA−μB)/√(σ²+σ²)); assumes independent totals. {hist_note}.")

            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**{format_team_name(t1_name)}**")
                st.markdown(f"Projected: **{s1:.2f}**")
                if not t1_df.empty:
                    st.dataframe(t1_df[['Player', 'Position', 'Points', 'Matchup']].style.format({'Points': '{:.2f}'}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No active players.")
            with c2:
                st.write(f"**{format_team_name(t2_name)}**")
                st.markdown(f"Projected: **{s2:.2f}**")
                if not t2_df.empty:
                    st.dataframe(t2_df[['Player', 'Position', 'Points', 'Matchup']].style.format({'Points': '{:.2f}'}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No active players.")