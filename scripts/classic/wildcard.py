"""
Classic FPL - Wildcard Optimizer

Uses linear programming (PuLP) to find the mathematically optimal 15-player squad
for the Wildcard chip. Unlike the Free Hit Optimizer (single GW, cheap bench),
this optimizes across **multiple gameweeks** with a **strong bench** and **captain bonus**.
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
    get_fixture_difficulty_grid,
    position_converter,
)
from fuzzywuzzy import fuzz


# ---------------------------
# FDR MULTIPLIERS
# ---------------------------
# Adjust points per game based on fixture difficulty
FDR_MULTIPLIERS = {
    1: 1.30,  # Very easy fixture
    2: 1.15,  # Easy
    3: 1.00,  # Medium (no adjustment)
    4: 0.85,  # Hard
    5: 0.70,  # Very hard
}


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


def _get_fdr_color(fdr: float) -> str:
    """Get color for FDR value."""
    fdr_int = int(round(fdr)) if not pd.isna(fdr) else 3
    if fdr_int <= 1:
        return "#00c853"
    elif fdr_int <= 2:
        return "#86efac"
    elif fdr_int <= 3:
        return "#ffc107"
    elif fdr_int <= 4:
        return "#ff9800"
    else:
        return "#dc3545"


def _lookup_projection(player_name: str, team: str, position: str, projections_df: pd.DataFrame) -> Optional[float]:
    """Look up projection for a player using fuzzy matching.

    Requires team to match to avoid false positives.
    """
    if projections_df is None or projections_df.empty:
        return None

    best_match = None
    best_score = 0

    for _, row in projections_df.iterrows():
        proj_name = str(row.get("Player", ""))
        proj_team = str(row.get("Team", ""))
        proj_pos = str(row.get("Position", ""))

        # REQUIRE team to match
        if proj_team != team:
            continue

        score = fuzz.ratio(player_name.lower(), proj_name.lower())

        # Boost if position matches
        if proj_pos == position:
            score += 10

        if score > best_score and score >= 65:
            best_score = score
            best_match = row

    if best_match is not None:
        return best_match.get("Points")

    return None


def _get_player_fdr_for_horizon(
    team_short: str,
    fdr_diffs: pd.DataFrame,
    current_gw: int,
    horizon: int
) -> List[float]:
    """
    Get FDR values for a team over the specified horizon.
    Returns a list of FDR values (one per GW), using 3.0 as neutral for missing data.
    """
    fdr_values = []
    for gw_offset in range(horizon):
        gw = current_gw + gw_offset
        col = f"GW{gw}"
        if col in fdr_diffs.columns and team_short in fdr_diffs.index:
            fdr = fdr_diffs.at[team_short, col]
            fdr_values.append(fdr if not pd.isna(fdr) else 3.0)
        else:
            fdr_values.append(3.0)  # Neutral if data missing
    return fdr_values


def _calculate_multi_gw_projection(
    current_proj: Optional[float],
    ppg: float,
    form: float,
    fdr_values: List[float]
) -> Tuple[float, List[float]]:
    """
    Calculate multi-GW projection for a player.

    Args:
        current_proj: Rotowire projection for current GW (None if unavailable)
        ppg: Points per game from FPL data
        form: Recent form (points per game over last few GWs) - 0 means not playing recently
        fdr_values: List of FDR values for each GW in horizon

    Returns:
        (total_projection, per_gw_projections)
    """
    per_gw = []

    # If player has no recent form, they're not playing - don't project future points
    has_recent_form = form > 0

    for i, fdr in enumerate(fdr_values):
        if i == 0:
            # Current GW: Use Rotowire projection if available, else 0
            # No Rotowire = not expected to play this week
            if current_proj is not None:
                per_gw.append(current_proj)
            else:
                per_gw.append(0.0)
        else:
            # Future GWs: Use PPG adjusted by FDR, but only if player has recent form
            if has_recent_form:
                fdr_int = max(1, min(5, int(round(fdr))))
                multiplier = FDR_MULTIPLIERS.get(fdr_int, 1.0)
                adjusted_pts = ppg * multiplier
                per_gw.append(adjusted_pts)
            else:
                # No recent form = not playing, project 0
                per_gw.append(0.0)

    return sum(per_gw), per_gw


def _build_player_pool(
    bootstrap: dict,
    projections_df: pd.DataFrame,
    fdr_diffs: pd.DataFrame,
    current_gw: int,
    horizon: int,
    exclude_injured: bool = True,
    min_chance_of_playing: int = 75
) -> pd.DataFrame:
    """
    Build a DataFrame of all players with prices and multi-GW projections.
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
        ppg = float(p.get("points_per_game", 0) or 0)
        minutes = p.get("minutes", 0) or 0
        form = float(p.get("form", 0) or 0)  # Recent form (PPG over last few GWs)

        # Filter out injured/doubtful players
        is_cheap_fodder = price <= 4.5

        if exclude_injured and not is_cheap_fodder:
            if chance_of_playing is not None and chance_of_playing < min_chance_of_playing:
                continue
            if news and any(word in news.lower() for word in ['injured', 'illness', 'suspended', 'unavailable', 'out']):
                continue

        # Get Rotowire projection for current GW
        rotowire_proj = _lookup_projection(
            p.get("web_name", ""),
            team_short,
            position,
            projections_df
        )

        # Get FDR values for this team over horizon
        fdr_values = _get_player_fdr_for_horizon(team_short, fdr_diffs, current_gw, horizon)

        # Calculate multi-GW projection
        # Exclude backup players who have no realistic chance of playing.
        # A player without a Rotowire projection must have recent form to be included.
        # No Rotowire + no recent form = backup who isn't playing, skip them.
        if rotowire_proj is None and form <= 0:
            continue

        total_proj, per_gw_proj = _calculate_multi_gw_projection(rotowire_proj, ppg, form, fdr_values)

        # Skip players with no projection potential
        if total_proj <= 0:
            continue

        rows.append({
            "Player_ID": p.get("id"),
            "Player": p.get("web_name"),
            "Team": team_short,
            "Team_ID": team_id,
            "Position": position,
            "Price": price,
            "Rotowire_Proj": rotowire_proj if rotowire_proj is not None else 0.0,
            "PPG": ppg,
            "Total_Points": total_proj,
            "Per_GW_Points": per_gw_proj,
            "FDR_Values": fdr_values,
            "form": float(p.get("form", 0) or 0),
            "total_season_points": p.get("total_points", 0),
            "chance_of_playing": chance_of_playing,
            "news": news,
        })

    return pd.DataFrame(rows)


def solve_wildcard_squad(
    df: pd.DataFrame,
    budget: float,
    formation: str = "auto",
    bench_weight: float = 0.4
) -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float]]:
    """
    Uses PuLP to solve the FPL Wildcard optimization problem.

    Objective:
    - Maximize starting XI points + bench_weight * bench points

    Constraints:
    - Budget <= User Input
    - Squad Size = 15 (2 GK, 5 DEF, 5 MID, 3 FWD)
    - Starting XI = 11 with valid formation
    - Max 3 players per team

    Returns:
    - DataFrame with selected squad and Is_Starter flag
    - Total projected points for starting XI
    - Bench projected points
    """

    # Filter players with 0 or negative points unless very cheap
    pool = df[
        (df['Total_Points'] > 0) | (df['Price'] <= 4.5)
    ].reset_index(drop=True)

    if pool.empty:
        return None, None, None

    ids = pool.index.tolist()
    points = pool['Total_Points'].to_dict()
    prices = pool['Price'].to_dict()
    teams = pool['Team'].to_dict()
    positions = pool['Position'].to_dict()

    # Define variables
    select = pulp.LpVariable.dicts("Select", ids, cat=pulp.LpBinary)
    start = pulp.LpVariable.dicts("Start", ids, cat=pulp.LpBinary)

    # Define problem (Maximize)
    prob = pulp.LpProblem("FPL_Wildcard_Optimizer", pulp.LpMaximize)

    # Objective: Maximize starting XI points + bench_weight * bench points
    prob += (
        pulp.lpSum([start[i] * points[i] for i in ids]) +
        bench_weight * pulp.lpSum([(select[i] - start[i]) * points[i] for i in ids])
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
    prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'G']) == 1

    if formation != "auto":
        parts = formation.split("-")
        if len(parts) == 3:
            n_def, n_mid, n_fwd = int(parts[0]), int(parts[1]), int(parts[2])
            prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'D']) == n_def
            prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'M']) == n_mid
            prob += pulp.lpSum([start[i] for i in ids if positions[i] == 'F']) == n_fwd
    else:
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
        return None, None, None

    # Extract results
    selected_indices = [i for i in ids if pulp.value(select[i]) == 1]
    starting_indices = [i for i in ids if pulp.value(start[i]) == 1]

    squad_df = pool.loc[selected_indices].copy()
    squad_df['Is_Starter'] = squad_df.index.isin(starting_indices)

    # Sort: Starters first (by position G-D-M-F), then Bench
    pos_order = {'G': 1, 'D': 2, 'M': 3, 'F': 4}
    squad_df['Pos_Order'] = squad_df['Position'].map(pos_order)
    squad_df = squad_df.sort_values(
        by=['Is_Starter', 'Pos_Order', 'Total_Points'],
        ascending=[False, True, False]
    )

    starter_points = sum(points[i] for i in starting_indices)
    bench_points = sum(points[i] for i in selected_indices if i not in starting_indices)

    return squad_df, starter_points, bench_points


def _assign_bench_order(squad_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign bench order based on projected points (higher = lower bench order).
    GK bench is always position 1 (only sub GK if starter GK doesn't play).
    """
    bench = squad_df[~squad_df['Is_Starter']].copy()

    # GK first
    bench_gk = bench[bench['Position'] == 'G']
    bench_outfield = bench[bench['Position'] != 'G'].sort_values('Total_Points', ascending=False)

    bench_ordered = pd.concat([bench_gk, bench_outfield])
    bench_ordered['Bench_Order'] = range(1, len(bench_ordered) + 1)

    return bench_ordered


def _get_captain_info(squad_df: pd.DataFrame) -> Tuple[str, float]:
    """
    Get captain info (highest projected starter).
    Returns (player_name, captain_bonus_points).
    """
    starters = squad_df[squad_df['Is_Starter']]
    cap_idx = starters['Total_Points'].idxmax()
    cap_name = starters.loc[cap_idx, 'Player']
    cap_points = starters.loc[cap_idx, 'Total_Points']
    return cap_name, cap_points


def _render_fdr_heatmap(
    squad_df: pd.DataFrame,
    current_gw: int,
    horizon: int,
    fdr_diffs: pd.DataFrame
) -> None:
    """Render FDR heatmap for teams in the selected squad."""

    # Get unique teams from squad
    squad_teams = squad_df['Team'].unique().tolist()

    # Filter FDR diffs to only show squad teams
    gw_cols = [f"GW{current_gw + i}" for i in range(horizon)]
    available_cols = [c for c in gw_cols if c in fdr_diffs.columns]

    if not available_cols:
        st.warning("No FDR data available for the selected horizon.")
        return

    # Create filtered dataframe
    fdr_filtered = fdr_diffs.loc[
        fdr_diffs.index.isin(squad_teams),
        available_cols
    ].copy()

    # Sort by average FDR (easiest first)
    avg_fdr = fdr_filtered.fillna(3).mean(axis=1)
    fdr_filtered = fdr_filtered.loc[avg_fdr.sort_values().index]

    # Create styled display
    PALETTE = {1: "#00c853", 2: "#86efac", 3: "#ffc107", 4: "#ff9800", 5: "#dc3545"}

    def style_cell(val):
        if pd.isna(val):
            return "background-color: #e5e7eb; color: #111; text-align: center;"
        fdr_int = max(1, min(5, int(round(val))))
        color = PALETTE[fdr_int]
        txt = "#ffffff" if fdr_int >= 5 else "#111111"
        return f"background-color: {color}; color: {txt}; text-align: center; font-weight: 600;"

    styled = fdr_filtered.style.applymap(style_cell)
    st.dataframe(styled, use_container_width=True)


# ---------------------------
# MAIN PAGE
# ---------------------------

def show_wildcard_page():
    """Display the Wildcard Optimizer page."""

    st.title("Wildcard Optimizer")
    st.caption(
        "Find the mathematically optimal 15-player squad for your Wildcard chip. "
        "Optimizes across multiple gameweeks with a strong bench and captain bonus."
    )

    # Get user's budget
    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    default_budget, budget_source = _get_user_budget(team_id)
    current_gw = get_current_gameweek() or 1

    # Settings
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
        horizon = st.slider(
            "Optimization Horizon (GWs)",
            min_value=3,
            max_value=6,
            value=4,
            help="Number of gameweeks to optimize for"
        )

    with col3:
        formation = st.selectbox(
            "Formation",
            ["auto", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
            index=0,
            help="Auto lets the optimizer choose the best formation"
        )

    # Advanced settings
    with st.expander("Advanced Settings", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            bench_weight = st.slider(
                "Bench Weight",
                min_value=0.2,
                max_value=0.6,
                value=0.4,
                step=0.1,
                help="How much to value bench players (0.4 = bench contributes 40% of their projected points to objective)"
            )
        with col_b:
            pass  # Reserved for future settings

        st.markdown("#### Player Filters")
        col_c, col_d = st.columns(2)
        with col_c:
            exclude_injured = st.checkbox(
                "Exclude injured/doubtful players",
                value=True,
                help="Filter out players with injury news or low chance of playing"
            )
        with col_d:
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
        with st.spinner("Loading player data, projections, and fixture data..."):
            # Load bootstrap data
            bootstrap = get_classic_bootstrap_static()
            if not bootstrap:
                show_api_error("loading player data for Wildcard optimizer")
                return

            # Load Rotowire projections
            projections_df = None
            try:
                rotowire_url = config.ROTOWIRE_URL
                if rotowire_url:
                    projections_df = get_rotowire_player_projections(rotowire_url)
            except Exception as e:
                st.warning(f"Could not load Rotowire projections: {e}. Using PPG-only projections.")

            if projections_df is None:
                projections_df = pd.DataFrame()

            # Load FDR data
            try:
                _, fdr_diffs, _ = get_fixture_difficulty_grid(weeks=horizon)
            except Exception as e:
                st.warning(f"Could not load fixture difficulty data: {e}")
                fdr_diffs = pd.DataFrame()

            # Build player pool
            player_pool = _build_player_pool(
                bootstrap,
                projections_df,
                fdr_diffs,
                current_gw,
                horizon,
                exclude_injured=exclude_injured,
                min_chance_of_playing=min_chance
            )

            if player_pool.empty:
                st.error("No players available after filtering.")
                return

            # Show pool info
            has_rotowire = not projections_df.empty
            data_sources = "Rotowire + FDR-adjusted PPG" if has_rotowire else "FDR-adjusted PPG only"
            st.info(f"Player pool: {len(player_pool)} players | Data: {data_sources} | Horizon: GW{current_gw}-GW{current_gw + horizon - 1}")

        with st.spinner("Running optimization..."):
            squad_df, starter_points, bench_points = solve_wildcard_squad(
                player_pool, budget, formation, bench_weight
            )

            if squad_df is None:
                st.error(
                    "Optimization failed. Try increasing your budget or changing the formation. "
                    "The constraints may be too restrictive."
                )
                return

        # Display results
        st.success("Optimization complete!")

        # Split into starters and bench
        starters = squad_df[squad_df['Is_Starter']].copy()
        bench_ordered = _assign_bench_order(squad_df)

        # Get captain info
        cap_name, cap_bonus = _get_captain_info(squad_df)

        # Calculate totals
        total_with_captain = starter_points + cap_bonus
        starter_cost = starters['Price'].sum()
        bench_cost = bench_ordered['Price'].sum()
        total_cost = starter_cost + bench_cost

        # Summary metrics
        st.markdown("### Projected Score")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Starting XI Points", f"{starter_points:.1f}")
        with col2:
            st.metric("Captain Bonus", f"+{cap_bonus:.1f}", help=f"Captain: {cap_name}")
        with col3:
            st.metric("Bench Value", f"{bench_points:.1f}")
        with col4:
            st.metric("Total (with Cap)", f"{total_with_captain:.1f}")

        st.markdown("")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Total Cost", _format_money(total_cost))
        with col6:
            st.metric("Remaining Budget", _format_money(budget - total_cost))
        with col7:
            pass
        with col8:
            pass

        st.markdown("---")

        # Starting XI
        st.markdown("### Starting XI")
        formation_str = (
            f"{starters[starters['Position'] == 'D'].shape[0]}-"
            f"{starters[starters['Position'] == 'M'].shape[0]}-"
            f"{starters[starters['Position'] == 'F'].shape[0]}"
        )
        st.caption(f"Formation: {formation_str} | Captain: {cap_name}")

        # Build GW columns
        gw_cols = [f"GW{current_gw + i}" for i in range(horizon)]

        # Prepare starter display data
        starter_rows = []
        for _, row in starters.iterrows():
            is_captain = row['Player'] == cap_name
            player_display = f"{row['Player']} (C)" if is_captain else row['Player']

            row_data = {
                "Player": player_display,
                "Team": row['Team'],
                "Pos": row['Position'],
                "Price": f"£{row['Price']:.1f}m",
            }

            # Add per-GW projections
            per_gw = row.get('Per_GW_Points', [])
            for i, col in enumerate(gw_cols):
                if i < len(per_gw):
                    row_data[col] = per_gw[i]
                else:
                    row_data[col] = 0.0

            row_data["Total"] = row['Total_Points']
            starter_rows.append(row_data)

        starters_display = pd.DataFrame(starter_rows)

        # Column config for display
        col_config = {
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Pos": st.column_config.TextColumn("Pos", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Total": st.column_config.NumberColumn("Total", format="%.1f"),
        }
        for col in gw_cols:
            col_config[col] = st.column_config.NumberColumn(col, format="%.1f")

        st.dataframe(
            starters_display,
            use_container_width=True,
            hide_index=True,
            column_config=col_config
        )

        st.markdown("---")

        # Bench
        st.markdown("### Bench")
        st.caption(f"Bench cost: {_format_money(bench_cost)} | Weighted value: {bench_points * bench_weight:.1f} pts")

        bench_rows = []
        for _, row in bench_ordered.iterrows():
            row_data = {
                "Order": int(row['Bench_Order']),
                "Player": row['Player'],
                "Team": row['Team'],
                "Pos": row['Position'],
                "Price": f"£{row['Price']:.1f}m",
            }

            per_gw = row.get('Per_GW_Points', [])
            for i, col in enumerate(gw_cols):
                if i < len(per_gw):
                    row_data[col] = per_gw[i]
                else:
                    row_data[col] = 0.0

            row_data["Total"] = row['Total_Points']
            bench_rows.append(row_data)

        bench_display = pd.DataFrame(bench_rows)

        bench_col_config = {
            "Order": st.column_config.NumberColumn("Order", width="small"),
            "Player": st.column_config.TextColumn("Player", width="medium"),
            "Team": st.column_config.TextColumn("Team", width="small"),
            "Pos": st.column_config.TextColumn("Pos", width="small"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Total": st.column_config.NumberColumn("Total", format="%.1f"),
        }
        for col in gw_cols:
            bench_col_config[col] = st.column_config.NumberColumn(col, format="%.1f")

        st.dataframe(
            bench_display,
            use_container_width=True,
            hide_index=True,
            column_config=bench_col_config
        )

        st.markdown("---")

        # FDR Heatmap
        st.markdown("### Fixture Difficulty (Squad Teams)")
        if not fdr_diffs.empty:
            _render_fdr_heatmap(squad_df, current_gw, horizon, fdr_diffs)
        else:
            st.info("FDR data not available.")

        st.markdown("---")

        # Team breakdown
        with st.expander("Team Breakdown"):
            team_counts = squad_df.groupby('Team').agg({
                'Player': 'count',
                'Price': 'sum',
                'Total_Points': 'sum'
            }).rename(columns={
                'Player': 'Players',
                'Price': 'Total Cost',
                'Total_Points': 'Total Proj Pts'
            })
            team_counts['Total Cost'] = team_counts['Total Cost'].apply(lambda x: f"£{x:.1f}m")
            team_counts['Total Proj Pts'] = team_counts['Total Proj Pts'].round(1)
            st.dataframe(team_counts, use_container_width=True)

        # Position breakdown
        with st.expander("Position Breakdown"):
            pos_breakdown = squad_df.groupby('Position').agg({
                'Player': 'count',
                'Price': 'sum',
                'Total_Points': 'sum'
            }).rename(columns={
                'Player': 'Count',
                'Price': 'Total Cost',
                'Total_Points': 'Total Proj Pts'
            })
            pos_breakdown['Total Cost'] = pos_breakdown['Total Cost'].apply(lambda x: f"£{x:.1f}m")
            pos_breakdown['Total Proj Pts'] = pos_breakdown['Total Proj Pts'].round(1)
            st.dataframe(pos_breakdown, use_container_width=True)

        # Tips
        st.markdown("---")
        st.info(
            "**Tips:**\n"
            "- The captain (C) is the highest projected scorer and adds their points again as bonus\n"
            "- Bench players are strong rotation options, ordered by projected points\n"
            "- FDR multipliers adjust future GW projections: easier fixtures boost PPG, harder ones reduce it\n"
            "- Try different horizons to see how fixture swings affect optimal picks\n"
            "- A higher bench weight prioritizes rotation options over maximizing starting XI points"
        )
