import config
import math
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from scripts.common.utils import find_optimal_lineup, format_team_name, get_current_gameweek, get_gameweek_fixtures, \
    get_team_id_by_name, get_rotowire_player_projections, get_team_composition_for_gameweek, \
    merge_fpl_players_and_projections, normalize_apostrophes, get_historical_team_scores, get_draft_h2h_record

def _normal_cdf(x: float) -> float:  # <<< ADD
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _estimate_score_std(league_id: int) -> tuple[float, int]:  # <<< ADD
    """
    Returns (std, n) for historical single-team weekly scores if available.
    Tries: scripts.common.utils.get_historical_team_scores(league_id) -> DataFrame with 'total_points' or 'score'.
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

def _get_win_pct_color(pct: float) -> str:
    """
    Returns a color on a red-to-green gradient based on win percentage.
    Uses a compressed scale so colors diverge more quickly from 50%.

    0-35% = strong red
    35-45% = red to yellow
    45-55% = yellow (narrow band)
    55-65% = yellow to green
    65-100% = strong green
    """
    if pct <= 35:
        # Strong red
        return "rgb(220, 53, 69)"  # Bootstrap danger red
    elif pct <= 45:
        # Red to Yellow (35-45%)
        ratio = (pct - 35) / 10
        r = 220 + int((255 - 220) * ratio)  # 220 to 255
        g = 53 + int((193 - 53) * ratio)    # 53 to 193
        b = 69 - int((69 - 7) * ratio)      # 69 to 7
        return f"rgb({r}, {g}, {b})"
    elif pct <= 55:
        # Yellow zone (45-55%) - narrow band
        ratio = (pct - 45) / 10
        r = 255 - int((255 - 200) * ratio)  # 255 to 200
        g = 193 + int((200 - 193) * ratio)  # 193 to 200
        b = 7 + int((80 - 7) * ratio)       # 7 to 80
        return f"rgb({r}, {g}, {b})"
    elif pct <= 65:
        # Yellow to Green (55-65%)
        ratio = (pct - 55) / 10
        r = 200 - int((200 - 40) * ratio)   # 200 to 40
        g = 200 - int((200 - 167) * ratio)  # 200 to 167
        b = 80 - int((80 - 69) * ratio)     # 80 to 69
        return f"rgb({r}, {g}, {b})"
    else:
        # Strong green (65%+)
        return "rgb(40, 167, 69)"  # Bootstrap success green


def _render_fixtures_overview(fixtures: list, league_id: int, projections_df: pd.DataFrame, sigma: float):
    """
    Render an overview table showing all fixtures with projected scores and win probabilities.
    """
    if not fixtures:
        return

    overview_data = []
    denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0

    with st.spinner("Calculating projections for all fixtures..."):
        for fixture in fixtures:
            try:
                result = analyze_fixture_projections(fixture, league_id, projections_df)
                if result is None:
                    continue

                team1_df, team2_df, team1_name, team2_name = result

                team1_score = team1_df['Points'].sum()
                team2_score = team2_df['Points'].sum()

                # Calculate win probability
                z = (team1_score - team2_score) / denom
                p_team1 = _normal_cdf(z)
                p_team2 = 1.0 - p_team1

                overview_data.append({
                    "team1": format_team_name(team1_name),
                    "proj1": team1_score,
                    "pct1": p_team1 * 100,
                    "pct2": p_team2 * 100,
                    "proj2": team2_score,
                    "team2": format_team_name(team2_name),
                })
            except Exception:
                continue

    if not overview_data:
        st.warning("Could not calculate projections for fixtures.")
        return

    # Build HTML table with fancy styling
    html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: transparent;
        }
        .fixtures-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 10px 0;
        }
        .fixtures-table th {
            background: linear-gradient(135deg, #37003c 0%, #5a0050 100%);
            color: white;
            padding: 14px 12px;
            text-align: center;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .fixtures-table th:first-child {
            border-radius: 10px 0 0 0;
        }
        .fixtures-table th:last-child {
            border-radius: 0 10px 0 0;
        }
        .fixtures-table td {
            padding: 14px 12px;
            text-align: center;
            border-bottom: 1px solid #e0e0e0;
            font-size: 14px;
        }
        .fixtures-table tr:last-child td:first-child {
            border-radius: 0 0 0 10px;
        }
        .fixtures-table tr:last-child td:last-child {
            border-radius: 0 0 10px 0;
        }
        .fixtures-table tr:hover td {
            background-color: #f8f4f9;
        }
        .team-name {
            font-weight: 600;
            color: #1a1a2e;
            min-width: 140px;
        }
        .team-left {
            text-align: right !important;
            padding-right: 20px !important;
        }
        .team-right {
            text-align: left !important;
            padding-left: 20px !important;
        }
        .proj-score {
            font-weight: 500;
            color: #444;
            min-width: 55px;
        }
        .win-pct {
            font-weight: 700;
            font-size: 15px;
            min-width: 65px;
            padding: 8px 12px !important;
            border-radius: 6px;
        }
        .vs-cell {
            color: #888;
            font-weight: 500;
            font-size: 12px;
            min-width: 40px;
        }
        .prob-bar-container {
            width: 100%;
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin-top: 12px;
            overflow: hidden;
            display: flex;
        }
        .prob-bar-left {
            height: 100%;
            transition: width 0.3s ease;
        }
        .prob-bar-right {
            height: 100%;
            transition: width 0.3s ease;
        }
    </style>
    </head>
    <body>
    <table class="fixtures-table">
    <thead>
        <tr>
            <th>Team</th>
            <th>Proj</th>
            <th>Win %</th>
            <th></th>
            <th>Win %</th>
            <th>Proj</th>
            <th>Team</th>
        </tr>
    </thead>
    <tbody>
    """

    for row in overview_data:
        color1 = _get_win_pct_color(row["pct1"])
        color2 = _get_win_pct_color(row["pct2"])

        html += f"""
        <tr>
            <td class="team-name team-left">{row["team1"]}</td>
            <td class="proj-score">{row["proj1"]:.1f}</td>
            <td class="win-pct" style="background: {color1}; color: white;">{row["pct1"]:.0f}%</td>
            <td class="vs-cell">vs</td>
            <td class="win-pct" style="background: {color2}; color: white;">{row["pct2"]:.0f}%</td>
            <td class="proj-score">{row["proj2"]:.1f}</td>
            <td class="team-name team-right">{row["team2"]}</td>
        </tr>
        """

    html += """
    </tbody>
    </table>
    </body>
    </html>
    """

    # Calculate height based on number of fixtures
    table_height = 60 + (len(overview_data) * 52)
    components.html(html, height=table_height, scrolling=False)


def show_fixtures_page():
    st.title("Upcoming Fixtures & Projections")

    # Find the fixtures for the current gameweek
    gameweek_fixtures = get_gameweek_fixtures(config.FPL_DRAFT_LEAGUE_ID, config.CURRENT_GAMEWEEK)

    if not gameweek_fixtures:
        st.warning("No fixtures found for the current gameweek.")
        return

    st.subheader(f"Gameweek {config.CURRENT_GAMEWEEK} Fixtures Overview")

    # Pull FPL player projections from Rotowire
    fpl_player_projections = get_rotowire_player_projections(config.ROTOWIRE_URL)

    if fpl_player_projections is None or fpl_player_projections.empty:
        st.warning("Rotowire projections unavailable.")
        # Still show fixtures list
        for fixture in gameweek_fixtures:
            st.text(fixture)
        return

    # Get sigma for win probability calculations
    sigma, n_hist = _estimate_score_std(config.FPL_DRAFT_LEAGUE_ID)

    # Render the fixtures overview table
    _render_fixtures_overview(gameweek_fixtures, config.FPL_DRAFT_LEAGUE_ID, fpl_player_projections, sigma)

    hist_note = f"σ≈{sigma:.2f} from {n_hist} historical scores" if n_hist > 0 else f"σ≈{sigma:.2f} (default)"
    st.caption(f"Win probability model: P(A>B) = Φ((μA−μB)/√(2σ²)). {hist_note}")

    # Divider before detailed view
    st.divider()

    # Detailed view section
    st.subheader("Detailed Match Analysis")

    # Create a dropdown to choose a fixture
    fixture_selection = st.selectbox("Select a fixture to analyze:", gameweek_fixtures)

    # Create the Streamlit visuals
    if fixture_selection:
        # Analyze fixture projections
        result = analyze_fixture_projections(fixture_selection, config.FPL_DRAFT_LEAGUE_ID, fpl_player_projections)

        if result is None:
            st.error(
                "**Could not analyze this fixture.** Player projections may be unavailable "
                "or team rosters could not be resolved. Try selecting a different fixture."
            )
            return

        team1_df, team2_df, team1_name, team2_name = result

        # Extract team scores from df
        team1_score = team1_df['Points'].sum()
        team2_score = team2_df['Points'].sum()

        # --- Win Probability (Normal model) ---
        denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0
        z = (team1_score - team2_score) / denom
        p_team1 = _normal_cdf(z)

        st.subheader("Win Probability")
        _render_winprob_bar(format_team_name(team1_name), format_team_name(team2_name), p_team1)

        # --- Head-to-Head History ---
        team1_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, team1_name)
        team2_id = get_team_id_by_name(config.FPL_DRAFT_LEAGUE_ID, team2_name)

        if team1_id and team2_id:
            h2h = get_draft_h2h_record(config.FPL_DRAFT_LEAGUE_ID, team1_id, team2_id)

            if h2h["wins"] + h2h["draws"] + h2h["losses"] > 0:
                st.subheader("Head-to-Head History")

                h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
                with h2h_col1:
                    st.metric(
                        label=f"{format_team_name(team1_name)} Wins",
                        value=h2h["wins"]
                    )
                with h2h_col2:
                    st.metric(
                        label="Draws",
                        value=h2h["draws"]
                    )
                with h2h_col3:
                    st.metric(
                        label=f"{format_team_name(team2_name)} Wins",
                        value=h2h["losses"]
                    )

                # Show recent matchups if available
                if h2h["matches"]:
                    with st.expander("View Past Matchups"):
                        match_data = []
                        for m in reversed(h2h["matches"]):  # Most recent first
                            match_data.append({
                                "Gameweek": f"GW{m['gameweek']}",
                                format_team_name(team1_name): m["my_pts"],
                                format_team_name(team2_name): m["opp_pts"],
                                "Result": m["outcome"]
                            })
                        st.dataframe(
                            pd.DataFrame(match_data),
                            use_container_width=True,
                            hide_index=True
                        )

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