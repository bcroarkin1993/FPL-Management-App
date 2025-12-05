import config
import math
import pandas as pd
import streamlit as st
from scripts.common.utils import find_optimal_lineup, format_team_name, get_current_gameweek, get_gameweek_fixtures, \
    get_team_id_by_name, get_rotowire_player_projections, get_team_composition_for_gameweek, \
    merge_fpl_players_and_projections, normalize_apostrophes, get_historical_team_scores

def _normal_cdf(x: float) -> float:  # <<< ADD
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _estimate_score_std(league_id: int) -> tuple[float, int]:  # <<< ADD
    """
    Returns (std, n) for historical single-team weekly scores if available.
    Tries: scripts.utils.get_historical_team_scores(league_id) -> DataFrame with 'total_points' or 'score'.
    Fallback: CSV path in config.HISTORICAL_SCORES_CSV (or 'data/historical_team_scores.csv').
    Final fallback: (15.0, 0) — a reasonable league-wide prior.
    """
    # Try utils function if it exists
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
    return 15.0, 0  # conservative default if nothing available

# --- Win % bar (two-color) ---
def _render_winprob_bar(team1_name: str, team2_name: str, p_team1: float):
    p1 = max(0.0, min(100.0, round(p_team1 * 100, 1)))
    p2 = round(100.0 - p1, 1)
    html = f"""
    <style>
      .wpb-wrap {{
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
      }}
      .wpb-labels, .wpb-bar {{
        display: grid;
        grid-template-columns: {p1}% {p2}%;
        gap: 0;
        width: 100%;
      }}
      .wpb-labels div {{
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
        line-height: 1.2;
        white-space: nowrap;
      }}
      .wpb-bar {{
        height: 36px;                  /* thicker bar */
        border-radius: 9999px;
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
      }}
      .wpb-left  {{ background: #2563eb; }}  /* blue  */
      .wpb-right {{ background: #dc2626; }}  /* red   */
      .wpb-subtle {{ color: rgba(0,0,0,0.65); }}
    </style>
    <div class="wpb-wrap">
      <div class="wpb-labels">
        <div class="wpb-subtle">{team1_name} {p1}%</div>
        <div class="wpb-subtle">{p2}% {team2_name}</div>
      </div>
      <div class="wpb-bar" role="img" aria-label="Win probability: {team1_name} {p1} percent, {team2_name} {p2} percent.">
        <div class="wpb-left"></div>
        <div class="wpb-right"></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def analyze_fixture_projections(fixture, league_id, projections_df):
    """
    Returns two DataFrames representing optimal projected lineups and points for each team in a fixture,
    sorted by position (GK, DEF, MID, FWD) and then by descending projected points within each position.

    Parameters:
    - fixture (str): The selected fixture, formatted as "Team1 (Player1) vs Team2 (Player2)".
    - league_id (int): The ID of the FPL Draft league.
    - projections_df (DataFrame): DataFrame containing player projections from Rotowire.

    Returns:
    - Tuple of two DataFrames: (team1_df, team2_df, team1_name, team2_name)
    """
    # Normalize the apostrophes in the fixture string
    fixture = normalize_apostrophes(fixture)

    # Extract the team names only (ignore player names inside parentheses)
    team1_name = fixture.split(' vs ')[0].split(' (')[0].strip()
    team2_name = fixture.split(' vs ')[1].split(' (')[0].strip()

    # Get the team ids based on the team names
    team1_id = get_team_id_by_name(league_id, team1_name)
    team2_id = get_team_id_by_name(league_id, team2_name)

    # Get the current gameweek
    gameweek = get_current_gameweek()

    # Retrieve team compositions for the current gameweek and convert to dataframes
    team1_composition = get_team_composition_for_gameweek(league_id, team1_id, gameweek)
    team2_composition = get_team_composition_for_gameweek(league_id, team2_id, gameweek)

    # Merge FPL players with projections for both teams
    team1_df = merge_fpl_players_and_projections(
        team1_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    )
    team2_df = merge_fpl_players_and_projections(
        team2_composition, projections_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    )

    # Debugging: Check if 'Points' column exists
    if 'Points' not in team1_df or 'Points' not in team2_df:
        print("Error: 'Points' column not found in one or both dataframes.")
        print("Team 1 DataFrame:\n", team1_df.head())
        print("Team 2 DataFrame:\n", team2_df.head())
        return None  # Exit the function early if the column is missing

    # Fill NaN values in 'Points' column with 0.0
    team1_df['Points'] = pd.to_numeric(team1_df['Points'], errors='coerce').fillna(0.0)
    team2_df['Points'] = pd.to_numeric(team2_df['Points'], errors='coerce').fillna(0.0)

    # Find the optimal lineup (top 11 players) for each team
    team1_df = find_optimal_lineup(team1_df)
    team2_df = find_optimal_lineup(team2_df)

    # Define the position order for sorting
    position_order = ['G', 'D', 'M', 'F']
    for df in [team1_df, team2_df]:
        df['Position'] = pd.Categorical(df['Position'], categories=position_order, ordered=True)
        df.sort_values(by=['Position', 'Points'], ascending=[True, False], inplace=True)

    # Select the final columns to use
    team1_df = team1_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]
    team2_df = team2_df[['Player', 'Team', 'Position', 'Matchup', 'Points', 'Pos Rank']]

    # Format team DataFrames to use player names as the index
    team1_df.set_index('Player', inplace=True)
    team2_df.set_index('Player', inplace=True)

    # Return the final DataFrames and team names
    return team1_df, team2_df, team1_name, team2_name

def show_fixtures_page():
    st.title("Upcoming Fixtures & Projections")

    # Find the fixtures for the current gameweek
    gameweek_fixtures = get_gameweek_fixtures(config.FPL_DRAFT_LEAGUE_ID, config.CURRENT_GAMEWEEK)

    # Display each of the current gameweek fixtures
    if gameweek_fixtures:
        st.subheader(f"Gameweek {config.CURRENT_GAMEWEEK} Fixtures")
        for fixture in gameweek_fixtures:
            st.text(fixture)

    # Subheader for match projections
    st.subheader("Match Projections")

    # Pull FPL player projections from Rotowire
    fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    # Create a dropdown to choose a fixture
    fixture_selection = st.selectbox("Select a fixture to analyze deeper:", gameweek_fixtures)

    # Create the Streamlit visuals
    if fixture_selection:
        # Analyze fixture projections
        team1_df, team2_df, team1_name, team2_name = analyze_fixture_projections(fixture_selection,
                                                                                 config.FPL_DRAFT_LEAGUE_ID,
                                                                                 fpl_player_projections)

        # Extract team scores from df
        team1_score = team1_df['Points'].sum()
        team2_score = team2_df['Points'].sum()

        # --- Win Probability (Normal model) ---
        sigma, n_hist = _estimate_score_std(config.FPL_DRAFT_LEAGUE_ID)
        denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0
        z = (team1_score - team2_score) / denom
        p_team1 = _normal_cdf(z)
        p_team2 = 1.0 - p_team1

        st.subheader("Win Probability")
        _render_winprob_bar(format_team_name(team1_name), format_team_name(team2_name), p_team1)
        hist_note = f"σ≈{sigma:.2f} from {n_hist} historical team scores" if n_hist > 0 else f"σ≈{sigma:.2f} (default)"
        st.caption(f"Model: P(A>B) = Φ((μA−μB)/√(σ²+σ²)); assumes independent team totals. {hist_note}.")

        # Create columns for side-by-side detailed display
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"{format_team_name(team1_name)} Projections")
            st.dataframe(team1_df,
                         use_container_width=True,
                         height=422  # Adjust the height to ensure the entire table shows
                         )
            st.markdown(f"**Projected Score: {team1_score:.2f}**")

        with col2:
            st.write(f"{format_team_name(team2_name)} Projections")
            st.dataframe(team2_df,
                         use_container_width=True,
                         height=422  # Adjust the height to ensure the entire table shows
                         )
            st.markdown(f"**Projected Score: {team2_score:.2f}**")