"""
Classic FPL - Free Hit Optimizer

Uses linear programming (PuLP) to find the mathematically optimal squad
for a Free Hit chip, maximizing projected points for the starting XI
while using cheap bench fillers.
"""

import pulp
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, List, Tuple
import config

from scripts.common.error_helpers import show_api_error
from scripts.common.utils import (
    get_rotowire_player_projections,
    get_classic_bootstrap_static,
    get_current_gameweek,
    get_entry_details,
    get_classic_team_picks,
    position_converter,
)
from fuzzywuzzy import fuzz
from scripts.common.styled_tables import render_styled_table


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------

def _format_money(value: float) -> str:
    """Format money value to display format."""
    return f"£{value:.1f}m"


def _get_user_budget(team_id: int) -> Tuple[float, str]:
    """
    Fetch the user's total budget (Squad Value + Bank).
    Returns (budget, source_description) tuple.
    """
    if not team_id:
        return 100.0, "default (no team configured)"

    try:
        # Try to get from current GW picks (most accurate)
        current_gw = get_current_gameweek()
        if current_gw:
            picks_data = get_classic_team_picks(team_id, current_gw)
            if picks_data:
                entry_history = picks_data.get("entry_history", {})
                value = entry_history.get("value", 0)  # Squad value in 0.1M
                bank = entry_history.get("bank", 0)    # Bank in 0.1M
                if value > 0:
                    total = (value + bank) / 10.0
                    return total, f"detected (value: £{value/10:.1f}m + bank: £{bank/10:.1f}m)"

        # Fallback to entry details
        details = get_entry_details(team_id)
        if details:
            raw_value = details.get('last_deadline_value', 0)
            if raw_value > 0:
                return raw_value / 10.0, "from last deadline value"
    except Exception:
        pass

    return 100.0, "default (could not fetch)"


@st.cache_data(ttl=300)
def _load_fixtures_for_gw(gw: int) -> pd.DataFrame:
    """Load fixtures for a specific gameweek."""
    import requests
    try:
        url = f"https://fantasy.premierleague.com/api/fixtures/?event={gw}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())
    except Exception:
        return pd.DataFrame()


def _get_team_fixture_info(team_id: int, gw: int, teams_map: Dict[int, str]) -> str:
    """Get fixture info for a team in a specific GW."""
    fixtures = _load_fixtures_for_gw(gw)
    if fixtures.empty:
        return ""

    for _, row in fixtures.iterrows():
        if row.get("team_h") == team_id:
            opp = teams_map.get(row.get("team_a"), "???")
            fdr = row.get("team_h_difficulty", 3)
            return f"vs {opp} (H) FDR:{fdr}"
        elif row.get("team_a") == team_id:
            opp = teams_map.get(row.get("team_h"), "???")
            fdr = row.get("team_a_difficulty", 3)
            return f"vs {opp} (A) FDR:{fdr}"

    return "No fixture"


def _get_fdr_color(fdr: int) -> str:
    """Get color for FDR value."""
    if fdr <= 2:
        return "#00c853"
    elif fdr <= 3:
        return "#ffc107"
    elif fdr <= 4:
        return "#ff9800"
    else:
        return "#dc3545"


def _lookup_projection(player_name: str, team: str, position: str, projections_df: pd.DataFrame) -> Optional[float]:
    """Look up projection for a player using fuzzy matching.

    Requires team to match to avoid false positives (e.g., matching 'Ortega' to wrong player).
    """
    if projections_df is None or projections_df.empty:
        return None

    best_match = None
    best_score = 0

    for _, row in projections_df.iterrows():
        proj_name = str(row.get("Player", ""))
        proj_team = str(row.get("Team", ""))
        proj_pos = str(row.get("Position", ""))

        # REQUIRE team to match - this prevents matching players to wrong team's projections
        if proj_team != team:
            continue

        score = fuzz.ratio(player_name.lower(), proj_name.lower())

        # Boost if position matches
        if proj_pos == position:
            score += 10

        # Higher threshold since we're already requiring team match
        if score > best_score and score >= 65:
            best_score = score
            best_match = row

    if best_match is not None:
        return best_match.get("Points")

    return None


def _build_player_pool(
    bootstrap: dict,
    projections_df: pd.DataFrame,
    exclude_injured: bool = True,
    min_chance_of_playing: int = 75
) -> pd.DataFrame:
    """
    Build a DataFrame of all players with prices and projections.

    Args:
        bootstrap: FPL bootstrap data
        projections_df: Rotowire projections
        exclude_injured: If True, exclude players with injury news or low chance of playing
        min_chance_of_playing: Minimum chance of playing % to include (if exclude_injured=True)
    """
    elements = bootstrap.get("elements", [])
    teams = {t["id"]: t for t in bootstrap.get("teams", [])}

    rows = []
    for p in elements:
        team_id = p.get("team")
        team_info = teams.get(team_id, {})
        team_short = team_info.get("short_name", "???")
        position = position_converter(p.get("element_type"))

        chance_of_playing = p.get("chance_of_playing_next_round")
        news = p.get("news", "") or ""
        price = p.get("now_cost", 0) / 10.0

        # Filter out injured/doubtful players for starters
        # But allow cheap players through for bench fodder
        is_cheap_fodder = price <= 4.5

        if exclude_injured and not is_cheap_fodder:
            # Skip if chance of playing is set and below threshold
            if chance_of_playing is not None and chance_of_playing < min_chance_of_playing:
                continue
            # Skip if there's injury/suspension news
            if news and any(word in news.lower() for word in ['injured', 'illness', 'suspended', 'unavailable', 'out']):
                continue

        # Get projection
        proj_points = _lookup_projection(
            p.get("web_name", ""),
            team_short,
            position,
            projections_df
        )

        # Skip players without projections (they likely won't play)
        # But keep very cheap players as potential bench fodder
        if proj_points is None and not is_cheap_fodder:
            continue

        rows.append({
            "Player_ID": p.get("id"),
            "Player": p.get("web_name"),
            "Team": team_short,
            "Team_ID": team_id,
            "Position": position,
            "Price": price,
            "Points": proj_points if proj_points is not None else 0.0,
            "form": float(p.get("form", 0) or 0),
            "total_points": p.get("total_points", 0),
            "chance_of_playing": chance_of_playing,
            "news": news,
        })

    return pd.DataFrame(rows)


def solve_optimal_squad(
    df: pd.DataFrame,
    budget: float,
    formation: str = "auto"
) -> Tuple[Optional[pd.DataFrame], Optional[float]]:
    """
    Uses PuLP to solve the FPL Free Hit optimization problem.

    Constraints:
    - Budget <= User Input
    - Squad Size = 15 (2 GK, 5 DEF, 5 MID, 3 FWD)
    - Starting XI = 11 with valid formation
    - Max 3 players per team

    Objective:
    - Maximize total projected points of the STARTING XI
    - Minimize bench cost (to leave room for better starters)

    Returns:
    - DataFrame with selected squad and Is_Starter flag
    - Total projected points for starting XI
    """

    # Filter players with 0 or negative points unless very cheap (bench fodder)
    pool = df[
        (df['Points'] > 0) | (df['Price'] <= 4.5)
    ].reset_index(drop=True)

    if pool.empty:
        return None, None

    ids = pool.index.tolist()
    points = pool['Points'].to_dict()
    prices = pool['Price'].to_dict()
    teams = pool['Team'].to_dict()
    positions = pool['Position'].to_dict()

    # Define variables
    select = pulp.LpVariable.dicts("Select", ids, cat=pulp.LpBinary)
    start = pulp.LpVariable.dicts("Start", ids, cat=pulp.LpBinary)

    # Define problem (Maximize)
    prob = pulp.LpProblem("FPL_Free_Hit_Optimizer", pulp.LpMaximize)

    # Objective: Maximize starting XI points
    # Add tiny negative weight to bench cost to prefer cheaper bench
    prob += (
        pulp.lpSum([start[i] * points[i] for i in ids]) -
        0.001 * pulp.lpSum([(select[i] - start[i]) * prices[i] for i in ids])
    )

    # Constraints

    # Budget constraint
    prob += pulp.lpSum([select[i] * prices[i] for i in ids]) <= budget

    # Squad size = 15
    prob += pulp.lpSum([select[i] for i in ids]) == 15

    # Starting XI = 11
    prob += pulp.lpSum([start[i] for i in ids]) == 11

    # Can only start if selected
    for i in ids:
        prob += start[i] <= select[i]

    # Position constraints (full squad of 15)
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'G']) == 2
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'D']) == 5
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'M']) == 5
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'F']) == 3

    # Formation constraints (starting XI)
    # 1 GK always
    prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'G']) == 1

    # Parse formation if specified
    if formation != "auto":
        parts = formation.split("-")
        if len(parts) == 3:
            n_def, n_mid, n_fwd = int(parts[0]), int(parts[1]), int(parts[2])
            prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'D']) == n_def
            prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'M']) == n_mid
            prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'F']) == n_fwd
    else:
        # Auto formation - just enforce FPL minimum rules
        prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'D']) >= 3
        prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'D']) <= 5
        prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'M']) >= 2
        prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'M']) <= 5
        prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'F']) >= 1
        prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'F']) <= 3

    # Max 3 players per team
    unique_teams = pool['Team'].unique()
    for t in unique_teams:
        prob += pulp.lpSum([select[i] for i in ids if teams[i] == t]) <= 3

    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None

    # Extract results
    selected_indices = [i for i in ids if pulp.value(select[i]) == 1]
    starting_indices = [i for i in ids if pulp.value(start[i]) == 1]

    squad_df = pool.loc[selected_indices].copy()
    squad_df['Is_Starter'] = squad_df.index.isin(starting_indices)

    # Sort: Starters first (by position G-D-M-F), then Bench
    pos_order = {'G': 1, 'D': 2, 'M': 3, 'F': 4}
    squad_df['Pos_Order'] = squad_df['Position'].map(pos_order)
    squad_df = squad_df.sort_values(
        by=['Is_Starter', 'Pos_Order', 'Points'],
        ascending=[False, True, False]
    )

    total_points = sum(points[i] for i in starting_indices)

    return squad_df, total_points


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_free_hit_page():
    """Display the Free Hit Optimizer page."""

    st.title("Free Hit Optimizer")
    st.caption(
        "Find the mathematically optimal squad for your Free Hit chip. "
        "Maximizes projected points for your starting XI while using cheap bench fillers."
    )

    # Get user's budget
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    default_budget, budget_source = _get_user_budget(team_id)
    current_gw = get_current_gameweek() or 1

    # Controls
    st.markdown("### Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        budget = st.number_input(
            "Total Budget (£m)",
            min_value=80.0,
            max_value=120.0,
            value=default_budget,
            step=0.1,
            format="%.1f"
        )
        st.caption(f"Budget {budget_source}")

    with col2:
        target_gw = st.number_input(
            "Target Gameweek",
            min_value=1,
            max_value=38,
            value=current_gw
        )

    with col3:
        formation = st.selectbox(
            "Formation",
            ["auto", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
            index=0,
            help="Auto lets the optimizer choose the best formation"
        )

    # Advanced filters
    with st.expander("Player Filters", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            exclude_injured = st.checkbox(
                "Exclude injured/doubtful players",
                value=True,
                help="Filter out players with injury news or low chance of playing"
            )
        with col_b:
            min_chance = st.slider(
                "Min chance of playing (%)",
                min_value=0,
                max_value=100,
                value=75,
                step=25,
                disabled=not exclude_injured,
                help="Only include players with at least this chance of playing"
            )

    st.markdown("---")

    # Run optimization
    if st.button("Calculate Optimal Squad", type="primary"):
        with st.spinner("Loading player data and projections..."):
            # Load data
            bootstrap = get_classic_bootstrap_static()
            if not bootstrap:
                show_api_error("loading player data for Free Hit optimizer")
                return

            # Load projections
            projections_df = None
            try:
                rotowire_url = config.ROTOWIRE_URL
                if rotowire_url:
                    projections_df = get_rotowire_player_projections(rotowire_url)
            except Exception as e:
                st.warning(f"Could not load projections: {e}")

            if projections_df is None or projections_df.empty:
                st.error(
                    "**No projections available.** Cannot optimize without projected points.\n\n"
                    "Check if Rotowire has published projections for the upcoming gameweek."
                )
                return

            # Build player pool
            player_pool = _build_player_pool(
                bootstrap,
                projections_df,
                exclude_injured=exclude_injured,
                min_chance_of_playing=min_chance
            )

            if player_pool.empty:
                st.error("No players with projections found.")
                return

            # Show how many players are in the pool
            st.info(f"Player pool: {len(player_pool)} players available after filtering")

        with st.spinner("Running optimization..."):
            # Solve
            squad_df, total_points = solve_optimal_squad(player_pool, budget, formation)

            if squad_df is None:
                st.error(
                    "Optimization failed. Try increasing your budget or changing the formation. "
                    "The constraints may be too restrictive."
                )
                return

        # Display results
        st.success(f"Optimization complete!")

        # Get team maps for fixture display
        teams_map = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
        teams_id_map = {t["short_name"]: t["id"] for t in bootstrap.get("teams", [])}

        # Split into starters and bench
        starters = squad_df[squad_df['Is_Starter']].copy()
        bench = squad_df[~squad_df['Is_Starter']].copy()

        # Find captain (highest projected points)
        if starters.empty or starters['Points'].isna().all():
            cap_name = "N/A"
            cap_points = 0.0
        else:
            cap_idx = starters['Points'].idxmax()
            cap_name = starters.loc[cap_idx, 'Player']
            cap_points = starters.loc[cap_idx, 'Points']

        # Calculate totals
        starter_points = starters['Points'].sum()
        total_with_captain = starter_points + cap_points  # Captain gets double
        starter_cost = starters['Price'].sum()
        bench_cost = bench['Price'].sum()
        total_cost = starter_cost + bench_cost

        # Summary metrics
        def _score_card(label: str, value: str, accent: str = "#00ff87") -> str:
            return (
                f'<div style="border:1px solid #333;border-radius:10px;padding:16px;'
                f'background:linear-gradient(135deg,#1a1a2e 0%,#16213e 100%);text-align:center;">'
                f'<div style="color:#9ca3af;font-size:11px;text-transform:uppercase;'
                f'letter-spacing:0.5px;margin-bottom:6px;">{label}</div>'
                f'<div style="color:{accent};font-size:22px;font-weight:700;">{value}</div>'
                f'</div>'
            )

        st.markdown("### Projected Score")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(_score_card("Starting XI Points", f"{starter_points:.1f}"), unsafe_allow_html=True)
        with col2:
            st.markdown(_score_card("With Captain Bonus", f"{total_with_captain:.1f}"), unsafe_allow_html=True)
        with col3:
            st.markdown(_score_card("Total Cost", _format_money(total_cost), accent="#e0e0e0"), unsafe_allow_html=True)
        with col4:
            remaining = budget - total_cost
            budget_color = "#00ff87" if remaining >= 0 else "#f87171"
            st.markdown(_score_card("Remaining Budget", _format_money(remaining), accent=budget_color), unsafe_allow_html=True)

        st.markdown("---")

        # Starting XI
        st.markdown("### Starting XI")
        st.caption(f"Formation: {starters[starters['Position'] == 'D'].shape[0]}-"
                   f"{starters[starters['Position'] == 'M'].shape[0]}-"
                   f"{starters[starters['Position'] == 'F'].shape[0]}")

        # Add fixture info and captain marker
        starter_rows = []
        for _, row in starters.iterrows():
            team_id = teams_id_map.get(row['Team'])
            fixture = _get_team_fixture_info(team_id, target_gw, teams_map) if team_id else ""

            is_captain = row.name == cap_idx
            player_display = f"⭐ {row['Player']} (C)" if is_captain else row['Player']

            starter_rows.append({
                "Player": player_display,
                "Team": row['Team'],
                "Pos": row['Position'],
                "Price": f"£{row['Price']:.1f}m",
                "Proj Pts": row['Points'],
                "Fixture": fixture,
            })

        starters_display = pd.DataFrame(starter_rows)

        render_styled_table(
            starters_display,
            col_formats={"Proj Pts": "{:.1f}"},
        )

        st.markdown("---")

        # Bench
        st.markdown("### Bench (Cheap Fillers)")
        st.caption(f"Bench cost: {_format_money(bench_cost)}")

        bench_rows = []
        for _, row in bench.iterrows():
            bench_rows.append({
                "Player": row['Player'],
                "Team": row['Team'],
                "Pos": row['Position'],
                "Price": f"£{row['Price']:.1f}m",
                "Proj Pts": row['Points'],
            })

        bench_display = pd.DataFrame(bench_rows)

        render_styled_table(
            bench_display,
            col_formats={"Proj Pts": "{:.1f}"},
        )

        st.markdown("---")

        # Team breakdown
        with st.expander("Team Breakdown"):
            team_counts = squad_df.groupby('Team').agg({
                'Player': 'count',
                'Price': 'sum',
                'Points': 'sum'
            }).rename(columns={
                'Player': 'Players',
                'Price': 'Total Cost',
                'Points': 'Total Proj Pts'
            })
            team_counts['Total Cost'] = team_counts['Total Cost'].apply(lambda x: f"£{x:.1f}m")
            team_counts['Total Proj Pts'] = team_counts['Total Proj Pts'].round(1)
            render_styled_table(team_counts.reset_index())

        # Position breakdown
        with st.expander("Position Breakdown"):
            pos_breakdown = squad_df.groupby('Position').agg({
                'Player': 'count',
                'Price': 'sum',
                'Points': 'sum'
            }).rename(columns={
                'Player': 'Count',
                'Price': 'Total Cost',
                'Points': 'Total Proj Pts'
            })
            pos_breakdown['Total Cost'] = pos_breakdown['Total Cost'].apply(lambda x: f"£{x:.1f}m")
            pos_breakdown['Total Proj Pts'] = pos_breakdown['Total Proj Pts'].round(1)
            render_styled_table(pos_breakdown.reset_index())

        # Tips
        st.markdown("---")
        st.info(
            "**Tips:**\n"
            "- The captain (⭐) is automatically assigned to the highest projected scorer\n"
            "- Bench players are chosen to minimize cost while meeting squad requirements\n"
            "- Try different formations to see which gives the best projected score\n"
            "- Consider fixtures and FDR when making your final decision"
        )
