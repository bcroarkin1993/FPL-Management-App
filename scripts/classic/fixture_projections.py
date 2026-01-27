"""
Classic FPL - Fixture Projections Page

Handles both league formats:
- H2H leagues: Show matchups between teams with win probability
- Classic scoring leagues: Show projected leaderboard with standings movement
"""

import config
import math
import pandas as pd
import random
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime, timezone
from fuzzywuzzy import fuzz
from scripts.common.utils import (
    find_optimal_lineup,
    get_classic_bootstrap_static,
    get_classic_h2h_record,
    get_classic_team_picks,
    get_classic_transfers,
    get_current_gameweek,
    get_entry_details,
    get_h2h_league_matches,
    get_league_standings,
    get_rotowire_player_projections,
    position_converter,
)


# =============================================================================
# Helper Functions
# =============================================================================

def _is_gameweek_started(gw: int, bootstrap: dict) -> bool:
    """
    Check if a gameweek's deadline has passed (lineups are locked).

    Parameters:
    - gw: The gameweek number to check.
    - bootstrap: Bootstrap-static data containing events.

    Returns:
    - True if the deadline has passed, False otherwise.
    """
    events = bootstrap.get("events", [])
    event = next((e for e in events if e.get("id") == gw), None)

    if not event:
        return False

    # If the event is finished, it has definitely started
    if event.get("finished"):
        return True

    # Check deadline time
    deadline_str = event.get("deadline_time")
    if not deadline_str:
        return False

    try:
        deadline = datetime.fromisoformat(deadline_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return now >= deadline
    except Exception:
        return False


def _get_team_current_squad(team_id: int, target_gw: int, bootstrap: dict) -> list:
    """
    Get a team's current squad by looking at previous GW picks and applying transfers.

    This is used when the target GW's picks are not yet available (pre-deadline).

    Parameters:
    - team_id: The FPL team ID.
    - target_gw: The gameweek we're projecting for.
    - bootstrap: Bootstrap-static data.

    Returns:
    - List of element IDs representing the current squad.
    """
    # Try to get picks from the most recent completed gameweek
    prev_gw = target_gw - 1

    # Find the most recent GW with available picks
    squad_elements = None
    source_gw = None

    for gw in range(prev_gw, 0, -1):
        picks_data = get_classic_team_picks(team_id, gw)
        if picks_data and picks_data.get("picks"):
            squad_elements = [p["element"] for p in picks_data["picks"]]
            source_gw = gw
            break

    if not squad_elements:
        return []

    # Apply any transfers made after the source GW
    transfers = get_classic_transfers(team_id)
    if transfers:
        # Sort transfers by event (gameweek)
        transfers = sorted(transfers, key=lambda t: t.get("event", 0))

        for transfer in transfers:
            transfer_gw = transfer.get("event", 0)
            # Only apply transfers that happened after our source GW and before/during target GW
            if source_gw < transfer_gw <= target_gw:
                element_out = transfer.get("element_out")
                element_in = transfer.get("element_in")

                if element_out in squad_elements:
                    squad_elements.remove(element_out)
                if element_in and element_in not in squad_elements:
                    squad_elements.append(element_in)

    return squad_elements


def _build_squad_from_elements(element_ids: list, bootstrap: dict) -> pd.DataFrame:
    """
    Build a squad dataframe from a list of element IDs.

    Unlike _build_squad_dataframe which uses picks with positions,
    this creates a squad without set positions (for optimal lineup calculation).

    Parameters:
    - element_ids: List of player element IDs.
    - bootstrap: Bootstrap-static data.

    Returns:
    - DataFrame with player info for optimal lineup calculation.
    """
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    rows = []
    for element_id in element_ids:
        player = elements.get(element_id, {})
        if not player:
            continue

        rows.append({
            "element_id": element_id,
            "Player": player.get("web_name", "Unknown"),
            "Full Name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "Team": teams.get(player.get("team"), "???"),
            "Position": position_converter(player.get("element_type")),
            "squad_position": 0,  # Will be set by optimal lineup
            "is_captain": False,
            "is_vice_captain": False,
            "multiplier": 1,
        })

    return pd.DataFrame(rows)


def _assign_optimal_captain(squad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign captain to the highest projected scorer.

    Parameters:
    - squad_df: DataFrame with 'Points' column.

    Returns:
    - DataFrame with is_captain set for highest scorer.
    """
    if squad_df.empty or "Points" not in squad_df.columns:
        return squad_df

    squad_df = squad_df.copy()
    squad_df["Points"] = pd.to_numeric(squad_df["Points"], errors="coerce").fillna(0.0)

    # Find highest scorer
    max_idx = squad_df["Points"].idxmax()
    squad_df["is_captain"] = False
    squad_df.loc[max_idx, "is_captain"] = True
    squad_df.loc[max_idx, "multiplier"] = 2

    return squad_df


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _render_winprob_bar(team1_name: str, team2_name: str, p_team1: float):
    """Render a two-color win probability bar."""
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
        height: 36px;
        border-radius: 9999px;
        overflow: hidden;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.08);
      }}
      .wpb-left  {{ background: #2563eb; }}
      .wpb-right {{ background: #dc2626; }}
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


def _lookup_projection(player_name: str, team: str, position: str, projections_df: pd.DataFrame) -> dict:
    """Look up projection for a player using fuzzy matching."""
    if projections_df is None or projections_df.empty:
        return {"Points": None, "Pos Rank": None}

    best_match = None
    best_score = 0

    for _, row in projections_df.iterrows():
        proj_name = str(row.get("Player", ""))
        proj_team = str(row.get("Team", ""))
        proj_pos = str(row.get("Position", ""))

        # Calculate name similarity
        score = fuzz.ratio(player_name.lower(), proj_name.lower())

        # Boost score if team and position match
        if proj_team == team and proj_pos == position:
            score += 15

        if score > best_score and score >= 60:
            best_score = score
            best_match = row

    if best_match is not None:
        return {
            "Points": best_match.get("Points"),
            "Pos Rank": best_match.get("Pos Rank", "N/A"),
        }

    return {"Points": None, "Pos Rank": None}


def _build_squad_dataframe(picks: list, bootstrap: dict) -> pd.DataFrame:
    """Map element IDs from picks to player info from bootstrap data."""
    elements = {p["id"]: p for p in bootstrap.get("elements", [])}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}

    rows = []
    for pick in picks:
        element_id = pick["element"]
        player = elements.get(element_id, {})

        rows.append({
            "element_id": element_id,
            "Player": player.get("web_name", "Unknown"),
            "Full Name": f"{player.get('first_name', '')} {player.get('second_name', '')}".strip(),
            "Team": teams.get(player.get("team"), "???"),
            "Position": position_converter(player.get("element_type")),
            "squad_position": pick["position"],
            "is_captain": pick.get("is_captain", False),
            "is_vice_captain": pick.get("is_vice_captain", False),
            "multiplier": pick.get("multiplier", 1),
        })

    return pd.DataFrame(rows)


def _add_projections_to_squad(squad_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """Add Rotowire projections to squad dataframe."""
    points_list = []
    rank_list = []

    for _, row in squad_df.iterrows():
        proj = _lookup_projection(
            row["Player"],
            row["Team"],
            row["Position"],
            projections_df
        )
        points_list.append(proj["Points"])
        rank_list.append(proj["Pos Rank"])

    squad_df["Points"] = points_list
    squad_df["Pos Rank"] = rank_list
    return squad_df


def _calculate_projected_score(squad_df: pd.DataFrame, active_chip: str = None) -> float:
    """
    Calculate total projected score for a squad.

    - Starting XI (positions 1-11) count normally
    - Captain gets 2x (or 3x for Triple Captain)
    - Bench Boost: all 15 players count
    """
    if squad_df.empty:
        return 0.0

    # Determine which players count
    if active_chip == "bboost":
        # Bench Boost: all players count
        active_players = squad_df.copy()
    else:
        # Normal: only starting XI (positions 1-11)
        active_players = squad_df[squad_df["squad_position"] <= 11].copy()

    # Calculate base points
    active_players["Points"] = pd.to_numeric(active_players["Points"], errors="coerce").fillna(0.0)

    # Apply captain multiplier
    captain_mult = 3 if active_chip == "3xc" else 2

    total = 0.0
    for _, row in active_players.iterrows():
        pts = row["Points"]
        if row.get("is_captain", False):
            pts *= captain_mult
        total += pts

    return total


def _format_movement(current_rank: int, projected_rank: int) -> str:
    """Format rank movement with arrow indicator."""
    if current_rank is None or projected_rank is None:
        return "-"

    diff = current_rank - projected_rank  # Positive = moving up (lower rank is better)

    if diff > 0:
        return f"↑ {diff}"
    elif diff < 0:
        return f"↓ {abs(diff)}"
    return "→ 0"


def _style_movement(val):
    """Style movement values with colors."""
    if isinstance(val, str):
        if val.startswith("↑"):
            return "color: #00c853; font-weight: bold;"
        elif val.startswith("↓"):
            return "color: #ff5252; font-weight: bold;"
    return "color: #757575;"


def _get_league_display_options() -> list:
    """Build list of league options for the dropdown."""
    leagues = config.FPL_CLASSIC_LEAGUE_IDS or []
    options = []

    for league in leagues:
        league_id = league["id"]
        if league["name"]:
            name = league["name"]
        else:
            data = get_league_standings(league_id)
            if data and "league" in data:
                name = data["league"].get("name", f"League {league_id}")
            else:
                name = f"League {league_id}"

        options.append({
            "id": league_id,
            "name": name,
            "display": f"{name} ({league_id})"
        })

    return options


# =============================================================================
# H2H Fixture Projections
# =============================================================================

def _get_team_squad_and_lineup(
    team_id: int,
    current_gw: int,
    bootstrap: dict,
    projections_df: pd.DataFrame,
    gw_started: bool
) -> tuple:
    """
    Get a team's squad and calculate projections.

    Returns:
    - (squad_df, active_chip, is_predicted) tuple
      - squad_df: DataFrame with squad and projections
      - active_chip: Active chip for the GW (or None)
      - is_predicted: True if lineup was predicted (pre-deadline)
    """
    picks_data = None
    active_chip = None
    is_predicted = False

    # Try to get actual picks if GW has started
    if gw_started:
        picks_data = get_classic_team_picks(team_id, current_gw)

    if picks_data and picks_data.get("picks"):
        # Use actual picks
        squad_df = _build_squad_dataframe(picks_data.get("picks", []), bootstrap)
        active_chip = picks_data.get("active_chip")
    else:
        # Fallback: Get current squad and calculate optimal lineup
        is_predicted = True
        squad_elements = _get_team_current_squad(team_id, current_gw, bootstrap)

        if not squad_elements:
            return pd.DataFrame(), None, True

        # Build squad from elements
        squad_df = _build_squad_from_elements(squad_elements, bootstrap)

    # Add projections
    if projections_df is not None and not projections_df.empty:
        squad_df = _add_projections_to_squad(squad_df, projections_df)
    else:
        squad_df["Points"] = 0.0
        squad_df["Pos Rank"] = "N/A"

    # If predicted lineup, calculate optimal and assign captain
    if is_predicted and not squad_df.empty:
        # Use find_optimal_lineup to get best 11
        squad_df["Points"] = pd.to_numeric(squad_df["Points"], errors="coerce").fillna(0.0)
        optimal_xi = find_optimal_lineup(squad_df.copy())

        # Mark starting XI
        optimal_players = optimal_xi["Player"].tolist()
        squad_df["squad_position"] = squad_df["Player"].apply(
            lambda x: optimal_players.index(x) + 1 if x in optimal_players else 12
        )

        # Assign captain to highest scorer in starting XI
        starting_xi = squad_df[squad_df["squad_position"] <= 11].copy()
        if not starting_xi.empty:
            max_idx = starting_xi["Points"].idxmax()
            squad_df["is_captain"] = False
            squad_df["multiplier"] = 1
            squad_df.loc[max_idx, "is_captain"] = True
            squad_df.loc[max_idx, "multiplier"] = 2

    return squad_df, active_chip, is_predicted


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


def _render_h2h_fixtures_overview(
    fixture_options: list,
    current_gw: int,
    bootstrap: dict,
    projections_df: pd.DataFrame,
    gw_started: bool,
    sigma: float = 15.0
):
    """
    Render an overview table showing all H2H fixtures with projected scores and win probabilities.
    """
    if not fixture_options:
        return

    overview_data = []
    denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0

    progress_bar = st.progress(0, text="Calculating projections for all fixtures...")

    for i, fixture in enumerate(fixture_options):
        try:
            # Get squad and lineup for both teams
            squad_1, chip_1, _ = _get_team_squad_and_lineup(
                fixture["team1_id"], current_gw, bootstrap, projections_df, gw_started
            )
            squad_2, chip_2, _ = _get_team_squad_and_lineup(
                fixture["team2_id"], current_gw, bootstrap, projections_df, gw_started
            )

            if squad_1.empty or squad_2.empty:
                continue

            score_1 = _calculate_projected_score(squad_1, chip_1)
            score_2 = _calculate_projected_score(squad_2, chip_2)

            # Calculate win probability
            z = (score_1 - score_2) / denom
            p_team1 = _normal_cdf(z)
            p_team2 = 1.0 - p_team1

            overview_data.append({
                "team1": fixture["team1_name"],
                "proj1": score_1,
                "pct1": p_team1 * 100,
                "pct2": p_team2 * 100,
                "proj2": score_2,
                "team2": fixture["team2_name"],
            })
        except Exception:
            continue

        progress_bar.progress((i + 1) / len(fixture_options))

    progress_bar.empty()

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


def _show_h2h_fixture_projections(league_id: int, league_name: str, current_gw: int):
    """Display H2H fixture projections with matchups and win probability."""

    st.subheader(f"Gameweek {current_gw} H2H Fixtures Overview")

    # Load bootstrap data first (needed for deadline check)
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        st.error("Failed to load player data.")
        return

    # Check if GW has started
    gw_started = _is_gameweek_started(current_gw, bootstrap)

    if not gw_started:
        st.info(
            f"**Pre-deadline mode:** Gameweek {current_gw} hasn't started yet. "
            "Showing predicted optimal lineups based on current squads."
        )

    # Fetch H2H matches for current gameweek
    matches_data = get_h2h_league_matches(league_id, event=current_gw)

    if not matches_data or not matches_data.get("results"):
        st.warning("No H2H fixtures found for this gameweek.")
        return

    matches = matches_data["results"]

    # Filter to only current gameweek matches
    gw_matches = [m for m in matches if m.get("event") == current_gw]

    if not gw_matches:
        st.info(f"No fixtures scheduled for Gameweek {current_gw}.")
        return

    # Build fixture list for dropdown
    fixture_options = []
    for match in gw_matches:
        entry_1 = match.get("entry_1_entry")
        entry_2 = match.get("entry_2_entry")
        name_1 = match.get("entry_1_name", "Unknown")
        name_2 = match.get("entry_2_name", "Unknown")

        if entry_1 and entry_2:
            fixture_options.append({
                "display": f"{name_1} vs {name_2}",
                "team1_id": entry_1,
                "team2_id": entry_2,
                "team1_name": name_1,
                "team2_name": name_2,
            })

    if not fixture_options:
        st.info("No valid H2H matchups found.")
        return

    # Load projections once for the overview
    projections_df = None
    try:
        rotowire_url = config.ROTOWIRE_URL
        if rotowire_url:
            projections_df = get_rotowire_player_projections(rotowire_url)
    except Exception:
        pass

    if projections_df is None or projections_df.empty:
        st.warning("Rotowire projections unavailable. Scores will show 0.")

    # Render fixtures overview table
    sigma = 15.0
    _render_h2h_fixtures_overview(fixture_options, current_gw, bootstrap, projections_df, gw_started, sigma)

    st.caption(f"Win probability model: P(A>B) = Φ((μA−μB)/√(2σ²)); σ≈{sigma:.2f} (default)")

    # Divider before detailed view
    st.divider()

    # Detailed view section
    st.subheader("Detailed Match Analysis")

    # Fixture selector
    selected_fixture = st.selectbox(
        "Select a fixture to analyze:",
        options=[opt["display"] for opt in fixture_options],
        key="h2h_fixture_selector"
    )

    # Find the selected fixture
    selected = next((f for f in fixture_options if f["display"] == selected_fixture), None)

    if not selected:
        return

    # Fetch data for both teams
    with st.spinner("Loading detailed team data..."):
        # Get squad and lineup for both teams
        squad_1, chip_1, predicted_1 = _get_team_squad_and_lineup(
            selected["team1_id"], current_gw, bootstrap, projections_df, gw_started
        )
        squad_2, chip_2, predicted_2 = _get_team_squad_and_lineup(
            selected["team2_id"], current_gw, bootstrap, projections_df, gw_started
        )

        if squad_1.empty or squad_2.empty:
            st.error("Failed to load team data. Please try again later.")
            return

        # Calculate projected scores
        score_1 = _calculate_projected_score(squad_1, chip_1)
        score_2 = _calculate_projected_score(squad_2, chip_2)

    # Win probability calculation
    denom = math.sqrt(2.0 * (sigma ** 2)) if sigma > 0 else 1.0
    z = (score_1 - score_2) / denom
    p_team1 = _normal_cdf(z)

    st.subheader("Win Probability")
    _render_winprob_bar(selected["team1_name"], selected["team2_name"], p_team1)

    # --- Head-to-Head History ---
    h2h = get_classic_h2h_record(league_id, selected["team1_id"], selected["team2_id"])

    if h2h["wins"] + h2h["draws"] + h2h["losses"] > 0:
        st.subheader("Head-to-Head History")

        h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
        with h2h_col1:
            st.metric(
                label=f"{selected['team1_name']} Wins",
                value=h2h["wins"]
            )
        with h2h_col2:
            st.metric(
                label="Draws",
                value=h2h["draws"]
            )
        with h2h_col3:
            st.metric(
                label=f"{selected['team2_name']} Wins",
                value=h2h["losses"]
            )

        # Show recent matchups if available
        if h2h["matches"]:
            with st.expander("View Past Matchups"):
                match_data = []
                for m in reversed(h2h["matches"]):  # Most recent first
                    match_data.append({
                        "Gameweek": f"GW{m['gameweek']}",
                        selected["team1_name"]: m["my_pts"],
                        selected["team2_name"]: m["opp_pts"],
                        "Result": m["outcome"]
                    })
                st.dataframe(
                    pd.DataFrame(match_data),
                    use_container_width=True,
                    hide_index=True
                )

    # Side-by-side team displays
    col1, col2 = st.columns(2)

    # Prepare display DataFrames (starting XI only for display)
    def prepare_display_df(squad_df, chip, is_predicted):
        if chip == "bboost":
            df = squad_df.copy()
        else:
            df = squad_df[squad_df["squad_position"] <= 11].copy()

        # Add captain indicator
        def format_name(row):
            name = row["Player"]
            if row.get("is_captain"):
                suffix = " (C*)" if is_predicted else " (C)"
                return f"{name}{suffix}"
            elif row.get("is_vice_captain"):
                return f"{name} (V)"
            return name

        df["Player"] = df.apply(format_name, axis=1)

        # Sort by position
        position_order = ['G', 'D', 'M', 'F']
        df['Position'] = pd.Categorical(df['Position'], categories=position_order, ordered=True)
        df = df.sort_values(by=['Position', 'Points'], ascending=[True, False])

        return df[["Player", "Team", "Position", "Points", "Pos Rank"]]

    display_1 = prepare_display_df(squad_1, chip_1, predicted_1)
    display_2 = prepare_display_df(squad_2, chip_2, predicted_2)

    with col1:
        chip_text = f" ({chip_1.upper()})" if chip_1 else ""
        predicted_text = " [Predicted]" if predicted_1 else ""
        st.markdown(f"**{selected['team1_name']}{chip_text}{predicted_text}**")
        st.dataframe(
            display_1.set_index("Player"),
            use_container_width=True,
            height=422
        )
        st.markdown(f"**Projected Score: {score_1:.2f}**")

    with col2:
        chip_text = f" ({chip_2.upper()})" if chip_2 else ""
        predicted_text = " [Predicted]" if predicted_2 else ""
        st.markdown(f"**{selected['team2_name']}{chip_text}{predicted_text}**")
        st.dataframe(
            display_2.set_index("Player"),
            use_container_width=True,
            height=422
        )
        st.markdown(f"**Projected Score: {score_2:.2f}**")

    # Show prediction disclaimer if applicable
    if predicted_1 or predicted_2:
        st.caption(
            "**Note:** [Predicted] lineups show optimal starting XI based on current squad and projections. "
            "Actual lineups may differ. Captain marked with (C*) is predicted based on highest projected scorer."
        )


# =============================================================================
# Classic Leaderboard Projections
# =============================================================================

def _show_classic_leaderboard_projections(league_id: int, league_name: str, current_gw: int, standings_data: dict):
    """Display projected leaderboard with standings movement for Classic scoring leagues."""

    st.subheader(f"Gameweek {current_gw} Projected Standings")

    standings = standings_data.get("standings", {})
    results = standings.get("results", [])

    if not results:
        st.warning("No standings data available.")
        return

    # Team limit slider
    max_teams = len(results)
    default_teams = min(20, max_teams)

    num_teams = st.slider(
        "Number of teams to analyze:",
        min_value=1,
        max_value=max_teams,
        value=default_teams,
        help="Limit the number of teams to reduce API calls and loading time."
    )

    # Get user's team ID to highlight
    my_team_id = config.FPL_CLASSIC_TEAM_ID
    my_team_name = None
    if my_team_id:
        entry = get_entry_details(my_team_id)
        if entry:
            my_team_name = entry.get("name")

    # Load bootstrap and projections
    with st.spinner("Loading player data and projections..."):
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap:
            st.error("Failed to load player data.")
            return

        # Check if GW has started
        gw_started = _is_gameweek_started(current_gw, bootstrap)

        if not gw_started:
            st.info(
                f"**Pre-deadline mode:** Gameweek {current_gw} hasn't started yet. "
                "Showing predicted optimal lineups based on current squads."
            )

        projections_df = None
        projections_available = False
        try:
            rotowire_url = config.ROTOWIRE_URL
            if rotowire_url:
                projections_df = get_rotowire_player_projections(rotowire_url)
                if projections_df is not None and not projections_df.empty:
                    projections_available = True
        except Exception:
            pass

        if not projections_available:
            st.warning("Rotowire projections unavailable. Projected GW points will show 0.")

    # Process each team
    projection_data = []
    any_predicted = False

    progress_bar = st.progress(0, text="Calculating projections...")

    for i, entry in enumerate(results[:num_teams]):
        team_id = entry.get("entry")
        team_name = entry.get("entry_name", "Unknown")
        manager_name = entry.get("player_name", "Unknown")
        current_rank = entry.get("rank", i + 1)
        current_total = entry.get("total", 0)

        # Get squad and lineup using unified function
        squad_df, active_chip, is_predicted = _get_team_squad_and_lineup(
            team_id, current_gw, bootstrap, projections_df, gw_started
        )

        if is_predicted:
            any_predicted = True

        projected_gw = 0.0
        if not squad_df.empty:
            projected_gw = _calculate_projected_score(squad_df, active_chip)

        projected_total = current_total + projected_gw

        projection_data.append({
            "team_id": team_id,
            "Team": team_name,
            "Manager": manager_name,
            "Current Rank": current_rank,
            "Current Total": current_total,
            "Proj GW": projected_gw,
            "Proj Total": projected_total,
            "Chip": active_chip.upper() if active_chip else "",
            "Predicted": is_predicted,
        })

        progress_bar.progress((i + 1) / num_teams, text=f"Processing {team_name}...")

    progress_bar.empty()

    # Create DataFrame and calculate projected ranks
    df = pd.DataFrame(projection_data)

    # Sort by projected total (descending) and assign projected rank
    df = df.sort_values("Proj Total", ascending=False).reset_index(drop=True)
    df["Proj Rank"] = df.index + 1

    # Calculate movement
    df["Movement"] = df.apply(
        lambda row: _format_movement(row["Current Rank"], row["Proj Rank"]),
        axis=1
    )

    # Sort back by current rank for display
    df = df.sort_values("Current Rank").reset_index(drop=True)

    # Prepare display DataFrame
    display_df = df[[
        "Current Rank", "Team", "Manager", "Current Total",
        "Proj GW", "Proj Total", "Proj Rank", "Movement", "Chip"
    ]].copy()

    display_df.columns = ["Rank", "Team", "Manager", "Current", "Proj GW", "Proj Total", "Proj Rank", "Movement", "Chip"]

    # Format numbers
    display_df["Proj GW"] = display_df["Proj GW"].apply(lambda x: f"{x:.1f}")
    display_df["Proj Total"] = display_df["Proj Total"].apply(lambda x: f"{x:.1f}")

    st.markdown("---")

    # Display the table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Team": st.column_config.TextColumn("Team", width="medium"),
            "Manager": st.column_config.TextColumn("Manager", width="medium"),
            "Current": st.column_config.NumberColumn("Current", width="small"),
            "Proj GW": st.column_config.TextColumn("Proj GW", width="small"),
            "Proj Total": st.column_config.TextColumn("Proj Total", width="small"),
            "Proj Rank": st.column_config.NumberColumn("Proj Rank", width="small"),
            "Movement": st.column_config.TextColumn("Movement", width="small"),
            "Chip": st.column_config.TextColumn("Chip", width="small"),
        }
    )

    # Legend
    st.markdown("""
    **Legend:**
    - ↑ N (green) = Projected to move up N positions
    - ↓ N (red) = Projected to move down N positions
    - → 0 = No change in position
    - **Chip**: Active chip for this gameweek (BBOOST, 3XC, etc.)
    """)

    # Show prediction disclaimer if applicable
    if any_predicted:
        st.caption(
            "**Note:** Lineups are predicted based on optimal starting XI using current squads and projections. "
            "Actual lineups may differ. Captain is predicted based on highest projected scorer."
        )

    # Highlight user's team if configured
    if my_team_name:
        my_row = df[df["Team"] == my_team_name]
        if not my_row.empty:
            row = my_row.iloc[0]
            st.info(
                f"**Your Team ({my_team_name}):** "
                f"Current Rank: {row['Current Rank']} | "
                f"Projected GW: {row['Proj GW']:.1f} | "
                f"Projected Total: {row['Proj Total']:.1f} | "
                f"Projected Rank: {row['Proj Rank']} ({row['Movement']})"
            )


# =============================================================================
# Main Page Function
# =============================================================================

def show_classic_fixture_projections_page():
    """Display Classic FPL fixture projections page with league selector."""

    st.title("Classic League Fixture Projections")

    # Get configured leagues
    league_options = _get_league_display_options()

    if not league_options:
        st.warning("No Classic leagues configured. Add leagues to FPL_CLASSIC_LEAGUE_IDS in your .env file.")
        st.code("FPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League")
        return

    # Initialize random league selection on first load
    if "classic_proj_league_index" not in st.session_state:
        st.session_state.classic_proj_league_index = random.randint(0, len(league_options) - 1)

    # League selector
    display_options = [opt["display"] for opt in league_options]
    selected_display = st.selectbox(
        "Select League",
        options=display_options,
        index=st.session_state.classic_proj_league_index,
        key="classic_proj_league_selector"
    )

    # Update session state when user changes selection
    new_index = display_options.index(selected_display)
    if new_index != st.session_state.classic_proj_league_index:
        st.session_state.classic_proj_league_index = new_index

    # Get selected league
    selected_league = league_options[st.session_state.classic_proj_league_index]
    league_id = selected_league["id"]
    league_name = selected_league["name"]

    # Fetch league data to determine type
    with st.spinner("Loading league data..."):
        data = get_league_standings(league_id)

    if not data:
        st.error(f"Failed to load league data for league {league_id}. Please check the league ID.")
        return

    league_info = data.get("league", {})

    # Determine scoring type
    league_type = league_info.get("league_type", "x")  # "s" = H2H, "x" = Classic
    scoring = league_info.get("scoring", "c")          # "h" = H2H, "c" = Classic

    is_h2h = league_type == "s" or scoring == "h"

    # Display league info
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.markdown(f"### {league_info.get('name', league_name)}")
    with col2:
        st.metric("Format", "H2H" if is_h2h else "Classic")
    with col3:
        current_gw = get_current_gameweek()
        st.metric("Gameweek", current_gw)

    st.divider()

    # Branch based on league type
    if is_h2h:
        _show_h2h_fixture_projections(league_id, league_name, current_gw)
    else:
        _show_classic_leaderboard_projections(league_id, league_name, current_gw, data)
