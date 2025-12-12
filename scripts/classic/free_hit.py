import pulp
import pandas as pd
import streamlit as st
import config
from scripts.common.api import (
    get_rotowire_player_projections,
    pull_fpl_player_stats,
    get_current_gameweek,
    get_entry_details
)
from scripts.common.utils import (
    normalize_rotowire_players,
    normalize_fpl_players_to_rotowire_schema,
    _bootstrap_teams_df,
    _enforce_rw_schema_fpl,
    merge_fpl_players_and_projections
)


def _get_user_budget(team_id: int) -> float:
    """
    Attempts to fetch the user's total budget (Squad Value + Bank).
    Returns a default of 100.0 if unable to fetch.
    """
    if not team_id:
        return 100.0

    try:
        details = get_entry_details(team_id)
        if details:
            # value and bank are in units of 0.1M (e.g., 1025 = 102.5)
            # 'last_deadline_value' includes the bank
            raw_value = details.get('last_deadline_value', 1000)
            return raw_value / 10.0
    except Exception as e:
        st.warning(f"Could not fetch live budget: {e}")

    return 100.0


def solve_optimal_lineup(df: pd.DataFrame, budget: float):
    """
    Uses PuLP to solve the FPL knapsack problem.
    Constraints:
    - Budget <= User Input
    - Squad Size = 15 (2 GK, 5 DEF, 5 MID, 3 FWD)
    - Starting XI = 11 (1 GK, min 3 DEF, min 2 MID, min 1 FWD)
    - Max 3 players per team

    Objective:
    - Maximize total points of the STARTING XI.
    """

    # 1. Setup Data
    # Filter out unavailable or players with 0 points to speed up solver
    # We keep cheap bench fodder (price <= 4.0) even if 0 points
    pool = df[
        (df['Points'] > 0) | (df['Price'] <= 4.0)
        ].reset_index(drop=True)

    ids = pool.index.tolist()
    points = pool['Points'].to_dict()
    prices = pool['Price'].to_dict()
    teams = pool['Team'].to_dict()
    positions = pool['Position'].to_dict()
    names = pool['Player'].to_dict()

    # 2. Define Variables
    # select_i: 1 if player i is in the squad (15 man), 0 otherwise
    select = pulp.LpVariable.dicts("Select", ids, cat=pulp.LpBinary)
    # start_i: 1 if player i is in the starting XI, 0 otherwise
    start = pulp.LpVariable.dicts("Start", ids, cat=pulp.LpBinary)
    # captain_i: 1 if player i is captain (double points), 0 otherwise
    # (Optional refinement: Solver will always cap the highest scorer in XI,
    # but strictly speaking "optimal lineup" implies optimal captaincy too)

    # 3. Define Problem (Maximize Points)
    prob = pulp.LpProblem("FPL_Optimal_Free_Hit", pulp.LpMaximize)

    # Objective: Maximize sum of points for starters
    # (We usually don't care about bench points for Free Hit unless autosub,
    # but we can add a tiny weight 0.01 for bench to pick 'best' bench fodder)
    prob += pulp.lpSum([start[i] * points[i] for i in ids]) + \
            0.01 * pulp.lpSum([(select[i] - start[i]) * points[i] for i in ids])

    # 4. Constraints

    # A. Budget constraint
    prob += pulp.lpSum([select[i] * prices[i] for i in ids]) <= budget

    # B. Squad Size Constraints (Must pick exactly 15)
    prob += pulp.lpSum([select[i] for i in ids]) == 15

    # C. Starting XI Size (Must pick exactly 11)
    prob += pulp.lpSum([start[i] for i in ids]) == 11

    # D. Start <= Select (Cannot start if not selected)
    for i in ids:
        prob += start[i] <= select[i]

    # E. Position Constraints (Squad of 15)
    # 2 GK, 5 DEF, 5 MID, 3 FWD
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'G']) == 2
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'D']) == 5
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'M']) == 5
    prob += pulp.lpSum([select[i] for i in ids if positions[i] == 'F']) == 3

    # F. Formation Constraints (Starting XI)
    # 1 GK
    prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'G']) == 1
    # Min 3 DEF
    prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'D']) >= 3
    # Min 2 MID (FPL rules: at least 1 goalkeeper, at least 3 defenders, at least 1 forward... wait.
    # Valid formations: 3-5-2, 3-4-3, 4-4-2, 4-3-3, 5-3-2, 5-4-1, etc.
    # The strict rules are: 1 GK. Min 3 DEF. Min 1 FWD. (MIDs just fill the rest)
    prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'F']) >= 1

    # G. Team Constraint (Max 3 per team)
    unique_teams = pool['Team'].unique()
    for t in unique_teams:
        prob += pulp.lpSum([select[i] for i in ids if teams[i] == t]) <= 3

    # 5. Solve
    # standard solver (CBC) usually bundled with PuLP
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None

    # 6. Extract Results
    selected_indices = [i for i in ids if pulp.value(select[i]) == 1]
    starting_indices = [i for i in ids if pulp.value(start[i]) == 1]

    squad_df = pool.loc[selected_indices].copy()
    squad_df['Is_Starter'] = squad_df.index.isin(starting_indices)

    # Sort: Starters first (by pos G-D-M-F), then Bench
    pos_map = {'G': 1, 'D': 2, 'M': 3, 'F': 4}
    squad_df['Pos_Ord'] = squad_df['Position'].map(pos_map)
    squad_df = squad_df.sort_values(by=['Is_Starter', 'Pos_Ord', 'Points'], ascending=[False, True, False])

    return squad_df, prob.objective.value()


def show_free_hit_page():
    st.header("🚀 Free Hit Optimizer")
    st.caption("Calculates the mathematically optimal lineup based on Rotowire projections and your budget.")

    # 1. Inputs
    col1, col2 = st.columns(2)

    # Attempt to fetch default budget
    user_team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    default_budget = _get_user_budget(user_team_id)

    with col1:
        budget = st.number_input("Total Budget (£M)", min_value=80.0, max_value=120.0, value=default_budget, step=0.1,
                                 format="%.1f")

    with col2:
        # Check current gameweek for projections
        curr_gw = get_current_gameweek()
        target_gw = st.number_input("Target Gameweek", min_value=1, max_value=38, value=curr_gw)

    if st.button("🔮 Calculate Optimal Lineup"):
        with st.spinner("Fetching data and optimizing..."):
            # A. Fetch Data
            try:
                # 1. Projections
                projections_raw = get_rotowire_player_projections(config.ROTOWIRE_URL)
                # 2. FPL Data (for Prices)
                fpl_stats_raw = pull_fpl_player_stats()
                teams_df = _bootstrap_teams_df()

                # B. Normalize & Merge
                proj_norm = normalize_rotowire_players(projections_raw)
                fpl_norm = normalize_fpl_players_to_rotowire_schema(fpl_stats_raw, teams_df)

                # We need PRICE from FPL data attached to Projections
                # Prepare FPL side for merge
                fpl_clean = _enforce_rw_schema_fpl(fpl_norm, teams_df)

                # We need to grab 'now_cost' from the raw fpl_stats before it was dropped/renamed
                # Easier way: normalize_fpl... keeps common cols. Let's inspect fpl_stats_raw directly
                # fpl_stats_raw usually has 'now_cost' (units of 0.1M, e.g. 55 = 5.5)
                prices_df = fpl_stats_raw[['id', 'now_cost']].copy()
                prices_df['Price'] = prices_df['now_cost'] / 10.0
                prices_df = prices_df.rename(columns={'id': 'Player_ID'})

                # Join Prices to Normalized FPL (which has Player_ID)
                fpl_with_price = fpl_clean.merge(prices_df, on="Player_ID", how="left")

                # Now fuzzy merge Projections with FPL (to get Price onto Projections)
                # We merge FPL (left) to Projections (right) ? No, we want Projections as base.
                merged = merge_fpl_players_and_projections(fpl_with_price, proj_norm)

                # 'merged' now has Player, Team, Position, Points (from proj), Price (from fpl)
                # Drop players with missing price (means we couldn't link FPL data)
                df_model = merged.dropna(subset=['Price', 'Points']).copy()

            except Exception as e:
                st.error(f"Data processing error: {e}")
                st.stop()

            # C. Optimize
            squad, total_score = solve_optimal_lineup(df_model, budget)

            if squad is None:
                st.error("Optimization failed. Constraints could not be met (check Budget).")
                return

            # D. Display
            starters = squad[squad['Is_Starter']].copy()
            bench = squad[~squad['Is_Starter']].copy()

            # Captain Calculation (Naive: Highest projected points)
            cap_idx = starters['Points'].idxmax()
            cap_name = starters.loc[cap_idx, 'Player']
            cap_pts = starters.loc[cap_idx, 'Points']

            # Total includes captain bonus
            final_projected = starters['Points'].sum() + cap_pts

            st.success(f"Optimal Projected Score: **{final_projected:.2f}**")

            st.subheader("Starting XI")
            st.dataframe(
                starters[['Player', 'Team', 'Position', 'Price', 'Points']].style.format(
                    {"Price": "{:.1f}", "Points": "{:.2f}"}),
                use_container_width=True
            )

            st.subheader("Bench")
            st.dataframe(
                bench[['Player', 'Team', 'Position', 'Price', 'Points']].style.format(
                    {"Price": "{:.1f}", "Points": "{:.2f}"}),
                use_container_width=True
            )

            st.info(f"Total Cost: £{squad['Price'].sum():.1f}m / £{budget}m")
            st.caption(f"Captain: {cap_name} ({cap_pts:.2f} pts)")