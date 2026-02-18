"""
Lineup Optimization Functions.

FPL lineup validation and optimal lineup selection.
"""

import pandas as pd


def check_valid_lineup(df):
    """
    Given a dataframe with a lineup, check to see if it is a valid lineup.
    Requirements:
    - 11 total players
    - 1 GK
    - Min of 3 DEF
    - Max of 5 DEF
    - Min of 3 MID
    - Max of 5 MID
    - Min of 1 FWD
    - Max of 3 FWD
    """
    # Check the total players count
    players = len(df)

    # Count occurrences of each value in the 'Position' column
    position_counts = df['position'].value_counts()

    # Perform the checks
    player_check = players == 11
    gk_check = position_counts['G'] == 1
    def_check = position_counts['D'] >= 3 and position_counts['D'] <= 5
    mid_check = position_counts['M'] >= 3 and position_counts['M'] <= 5
    fwd_check = position_counts['F'] >= 1 and position_counts['F'] <= 3

    # Lineup is valid is all checks are true
    return (player_check & gk_check & def_check & mid_check & fwd_check)


def find_optimal_lineup(df):
    """
    Function to find a team's optimal lineup given their player_projections_df.

    Enforces valid FPL formation:
    - Exactly 1 GK
    - 3-5 DEF
    - 2-5 MID
    - 1-3 FWD
    - Total of 11 players

    :param df: a dataframe of the team's player projections
    :return: optimal 11-player lineup DataFrame
    """
    # 1. Find the top scoring GK (exactly 1)
    top_gk = df[df['Position'] == 'G'].sort_values(by='Points', ascending=False).head(1)

    # 2. Find the top 3 scoring DEF (minimum required)
    all_def = df[df['Position'] == 'D'].sort_values(by='Points', ascending=False)
    top_def = all_def.head(3)

    # 3. Find the top 2 scoring MID (minimum required, need at least 2)
    all_mid = df[df['Position'] == 'M'].sort_values(by='Points', ascending=False)
    top_mid = all_mid.head(2)

    # 4. Find the top scoring FWD (minimum 1)
    all_fwd = df[df['Position'] == 'F'].sort_values(by='Points', ascending=False)
    top_fwd = all_fwd.head(1)

    # 5. Combine the base selected players (1 GK + 3 DEF + 2 MID + 1 FWD = 7 players)
    selected_players = pd.concat([top_gk, top_def, top_mid, top_fwd])
    selected_names = set(selected_players['Player'].tolist())

    # 6. Need to add 4 more players from remaining (excluding GKs)
    # Remaining pool: DEF (up to 2 more), MID (up to 3 more), FWD (up to 2 more)
    remaining_def = all_def[~all_def['Player'].isin(selected_names)].head(2)  # Can add up to 2 more DEF
    remaining_mid = all_mid[~all_mid['Player'].isin(selected_names)].head(3)  # Can add up to 3 more MID
    remaining_fwd = all_fwd[~all_fwd['Player'].isin(selected_names)].head(2)  # Can add up to 2 more FWD

    # Combine remaining candidates (no GKs allowed)
    remaining_pool = pd.concat([remaining_def, remaining_mid, remaining_fwd])
    remaining_pool = remaining_pool.sort_values(by='Points', ascending=False)

    # Track position counts as we add players
    pos_counts = {'G': 1, 'D': 3, 'M': 2, 'F': 1}
    max_counts = {'G': 1, 'D': 5, 'M': 5, 'F': 3}

    players_to_add = []
    for _, player in remaining_pool.iterrows():
        if len(players_to_add) >= 4:
            break
        pos = player['Position']
        if pos_counts.get(pos, 0) < max_counts.get(pos, 0):
            players_to_add.append(player)
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

    # 7. Combine all selected players
    if players_to_add:
        additional = pd.DataFrame(players_to_add)
        final_selection = pd.concat([selected_players, additional])
    else:
        final_selection = selected_players

    # 8. Organize the final selection by Position and descending Projected_Points
    final_selection = final_selection.sort_values(
        by=['Position', 'Points'],
        key=lambda x: x.map({'G': 0, 'D': 1, 'M': 2, 'F': 3}),
        ascending=[True, False]
    ).reset_index(drop=True)

    return final_selection
