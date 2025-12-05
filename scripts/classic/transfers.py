import pandas as pd

def get_transfer_recommendations(
        current_squad_df: pd.DataFrame,
        all_players_df: pd.DataFrame,
        bank_budget: float,
        sort_metric: str = 'Projected_Points'
) -> pd.DataFrame:
    """
    Suggests transfers for Classic mode.

    Logic:
    1. Identify weaknesses in current squad (e.g., lowest projected player).
    2. Filter 'all_players' for affordable upgrades.
    """

    recommendations = []

    # 1. Calculate sellable value of current players
    # (In FPL API, selling_price is often different from now_cost,
    # ensure you are using the correct field from 'picks' endpoint if available)

    for idx, player in current_squad_df.iterrows():
        current_pos = player['Position']
        sell_price = player['cost']  # or 'now_cost' / 10 depending on data source

        # Max budget if we sell this player
        available_funds = bank_budget + sell_price

        # 2. Find replacements
        # Filter: Same Position, Affordable, Not already in team
        candidates = all_players_df[
            (all_players_df['Position'] == current_pos) &
            (all_players_df['now_cost'] <= available_funds) &
            (~all_players_df['Player_ID'].isin(current_squad_df['Player_ID']))
            ].copy()

        # 3. Calculate Gain (Projected Points Diff)
        candidates['Point_Diff'] = candidates[sort_metric] - player[sort_metric]

        # Filter for positive gain
        better_options = candidates[candidates['Point_Diff'] > 0].sort_values(
            by='Point_Diff', ascending=False
        ).head(3)  # Top 3 replacements per player

        for _, option in better_options.iterrows():
            recommendations.append({
                'Sell': player['Player'],
                'Buy': option['Player'],
                'Cost_Diff': option['now_cost'] - sell_price,
                'Projected_Gain': option['Point_Diff'],
                'New_Player_FDR': option.get('AvgFDRNextN', 0)
            })

    return pd.DataFrame(recommendations).sort_values(by='Projected_Gain', ascending=False)