"""
Lineup Optimization Functions.

FPL lineup validation, optimal lineup selection, and shared squad-building ILP.
"""

from typing import Collection, Dict, List, Optional, Tuple

import pandas as pd
import pulp

#: FPL's squad size. The change constraint counts kept players out of this, so
#: it must be the rulebook constant and never ``len(owned_in_pool)`` -- an owned
#: player missing from the pool would otherwise buy a free transfer.
SQUAD_SIZE = 15

#: Points charged for a transfer beyond the free allowance.
HIT_COST = 4.0


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


def find_optimal_lineup(df, points_col='Points'):
    """
    Function to find a team's optimal lineup given their player_projections_df.

    Enforces valid FPL formation:
    - Exactly 1 GK
    - 3-5 DEF
    - 2-5 MID
    - 1-3 FWD
    - Total of 11 players

    :param df: a dataframe of the team's player projections
    :param points_col: column to rank players by. Callers that have already blended
        projections (e.g. 'Proj_Blended') must pass that column, otherwise the XI is
        selected on one metric while the team total is reported on another -- an
        FFP-only player (Points == 0, Proj_Blended > 0) could never be picked.
    :return: optimal 11-player lineup DataFrame
    """
    if points_col not in df.columns:
        # Deliberately loud. This used to fall back to 'Points' silently, which
        # during the migration to the projection engine would let a page ask for
        # 'Proj' and quietly get raw un-blended Rotowire instead -- the XI would
        # look fine and be selected on the wrong number. A caller that genuinely
        # wants Rotowire can say so.
        raise KeyError(
            f"find_optimal_lineup: {points_col!r} is not on the frame "
            f"(columns: {sorted(df.columns)}). Pass the column you actually "
            f"want the XI ranked by -- silently substituting 'Points' selects "
            f"the lineup on a different metric than the caller reports."
        )
    # 1. Find the top scoring GK (exactly 1)
    top_gk = df[df['Position'] == 'G'].sort_values(by=points_col, ascending=False).head(1)

    # 2. Find the top 3 scoring DEF (minimum required)
    all_def = df[df['Position'] == 'D'].sort_values(by=points_col, ascending=False)
    top_def = all_def.head(3)

    # 3. Find the top 2 scoring MID (minimum required, need at least 2)
    all_mid = df[df['Position'] == 'M'].sort_values(by=points_col, ascending=False)
    top_mid = all_mid.head(2)

    # 4. Find the top scoring FWD (minimum 1)
    all_fwd = df[df['Position'] == 'F'].sort_values(by=points_col, ascending=False)
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
    remaining_pool = remaining_pool.sort_values(by=points_col, ascending=False)

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

    # 8. Organize the final selection by Position, then descending projected points.
    # The position key must only be applied to the Position column -- a shared `key=`
    # maps the points column to all-NaN and silently disables the secondary sort.
    final_selection = final_selection.assign(
        __pos_order=final_selection['Position'].map({'G': 0, 'D': 1, 'M': 2, 'F': 3})
    ).sort_values(
        by=['__pos_order', points_col],
        ascending=[True, False]
    ).drop(columns='__pos_order').reset_index(drop=True)

    return final_selection


# =============================================================================
# SHARED SQUAD-BUILDING ILP
# =============================================================================
# The mechanical "pick 15 players under FPL's rules" constraint set is
# identical across Free Hit, Wildcard, and Initial Squad optimizers — only
# the *scoring* of a player (which column drives the objective) differs per
# page, since each makes different assumptions about what data is trustworthy
# (short-horizon form vs. preseason season-long value, etc). This function
# owns only the shared constraint/objective mechanics; callers own scoring.

def solve_squad_ilp(
    df: pd.DataFrame,
    budget: float,
    score_col: str,
    price_col: str = "Price",
    team_col: str = "Team",
    position_col: str = "Position",
    formation: str = "auto",
    bench_weight: float = 0.0,
    captain_score_col: Optional[str] = None,
    captain_bonus_weight: float = 0.0,
    problem_name: str = "FPL_Squad_Optimizer",
    owned_ids: Optional[Collection] = None,
    id_col: str = "Player_ID",
    max_changes: Optional[int] = None,
    sell_price_col: Optional[str] = None,
    free_transfers: Optional[int] = None,
    hit_cost: float = HIT_COST,
    time_limit: Optional[int] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
    """
    Generic PuLP ILP squad-builder shared by the Classic squad optimizers.

    Constraints:
    - Budget <= budget
    - Squad size = 15 (2 GK, 5 DEF, 5 MID, 3 FWD)
    - Starting XI = 11 (1 GK, formation window or exact "D-M-F")
    - Max 3 players per club
    - Can only start if selected

    Objective (maximize):
        sum(start[i] * score[i])
        + bench_weight * sum((select[i] - start[i]) * score[i])
        + captain_bonus_weight * sum(is_captain[i] * captain_score[i])   # only if captain args given

    The captain mechanic (only active when both `captain_score_col` and a
    positive `captain_bonus_weight` are supplied) adds a binary "is this
    player the captain" variable constrained to a chosen starter, exactly
    one per squad. Rewarding that pick in the objective gives the solver a
    real incentive to *acquire* a standout captain-caliber player, not just
    label one after the fact.

    Args:
        df: Candidate player pool with score_col, price_col, team_col, position_col.
        budget: Total squad budget.
        score_col: Column driving the primary objective (points-per-player value).
        formation: "auto" (FPL min/max windows) or an exact "D-M-F" string, e.g. "3-4-3".
        bench_weight: Weight applied to bench players' score in the objective (0 = ignore bench).
        captain_score_col: Column used for the captain bonus term, if captaincy should
            influence selection (may differ from score_col — e.g. a season-pedigree-heavy
            score rather than the general blended score).
        captain_bonus_weight: Weight applied to the captain bonus term (0 = no captain mechanic).
        problem_name: PuLP problem name (cosmetic, shows in solver logs only).
        owned_ids: values of `id_col` for the squad currently owned. Supplying
            this together with `max_changes` switches on **transfer mode**,
            where the solver answers "the best squad reachable in at most K
            changes" rather than "the best squad". Every existing caller passes
            neither and is unaffected.
        id_col: column identifying a player across the two frames. A *value*
            column, not the index: the pool is filtered and reindexed below.
        max_changes: K, the most players that may be swapped out. Bounds the
            *search*; `free_transfers` sets the *price*.
        sell_price_col: what an owned player costs to keep. FPL credits a
            selling price that lags the market, so pricing a kept player at
            `price_col` charges you the market rate to re-buy someone you
            already have -- with prices risen, keeping your own squad comes out
            infeasible. Falls back to `price_col` where it is missing.
        free_transfers: how many changes are free. Each one beyond costs
            `hit_cost`, subtracted in the objective, so a hit is proposed only
            when it wins on points.
        hit_cost: points charged per transfer beyond the allowance.
        time_limit: seconds before CBC returns its best incumbent. None = no
            limit, which is every existing caller's behaviour.

    Returns:
        (squad_df, totals) where squad_df has an added 'Is_Starter' bool column
        (plus 'Is_Captain' if the captain mechanic was used), and totals is a
        dict with 'starter_score', 'bench_score', and 'captain_score' (if used).
        Returns (None, None) if no optimal solution is found.
    """
    use_transfer_mode = owned_ids is not None and max_changes is not None
    owned_set = set(owned_ids) if owned_ids is not None else set()

    # Filter players with 0 or negative score unless very cheap (bench fodder).
    #
    # In transfer mode an owned player is exempt, because the change constraint
    # counts kept players out of SQUAD_SIZE: drop one of the fifteen and his
    # sale becomes free and uncounted against K, drop two and K=1 is infeasible
    # with no explanation. Neither is hypothetical -- the projection engine
    # writes Proj = 0.0 for an unpriced player, and a blank-gameweek club is
    # zeroed outright, so a £9m injured midfielder disappears from his own
    # squad. The exemption is scoped to transfer mode: loosening the filter for
    # everyone would change what Free Hit, Wildcard and Initial Squad build.
    keep = (df[score_col] > 0) | (df[price_col] <= 4.5)
    if use_transfer_mode:
        keep = keep | df[id_col].isin(owned_set)
    pool = df[keep].reset_index(drop=True)

    if pool.empty:
        return None, None

    if use_transfer_mode:
        missing = owned_set - set(pool[id_col])
        if missing:
            raise ValueError(
                "solve_squad_ilp: %d owned player(s) absent from the candidate "
                "pool: %s. The change constraint counts kept players out of %d, "
                "so a missing owner silently grants a free transfer. Pass the "
                "unfiltered player pool, not one narrowed by display filters."
                % (len(missing), sorted(missing), SQUAD_SIZE))

    ids = pool.index.tolist()
    scores = pool[score_col].to_dict()
    prices = pool[price_col].to_dict()

    # What each player costs against the budget. An owned player costs his
    # selling price -- keeping him is not a purchase at the market rate.
    costs = pool[price_col].astype(float)
    if use_transfer_mode and sell_price_col and sell_price_col in pool.columns:
        sell = pd.to_numeric(pool[sell_price_col], errors="coerce")
        is_owned = pool[id_col].isin(owned_set)
        # A missing selling price falls back to the market price, which
        # understates funds -- the direction that keeps the plan buyable.
        costs = costs.where(~is_owned | sell.isna(), sell)
    costs = costs.to_dict()
    teams = pool[team_col].to_dict()
    positions = pool[position_col].to_dict()

    use_captain = captain_score_col is not None and captain_bonus_weight > 0
    captain_scores = pool[captain_score_col].to_dict() if use_captain else {}

    # Define variables
    select = pulp.LpVariable.dicts("Select", ids, cat=pulp.LpBinary)
    start = pulp.LpVariable.dicts("Start", ids, cat=pulp.LpBinary)
    is_captain = pulp.LpVariable.dicts("Captain", ids, cat=pulp.LpBinary) if use_captain else None

    owned_in_pool = [i for i in ids if pool.at[i, id_col] in owned_set] if use_transfer_mode else []
    use_hits = use_transfer_mode and free_transfers is not None
    hits = None
    if use_hits:
        hits = pulp.LpVariable("Hits", lowBound=0,
                               upBound=max(0, int(max_changes) - int(free_transfers)),
                               cat=pulp.LpInteger)

    # Define problem (Maximize)
    prob = pulp.LpProblem(problem_name, pulp.LpMaximize)

    # Objective
    objective = (
        pulp.lpSum([start[i] * scores[i] for i in ids]) +
        bench_weight * pulp.lpSum([(select[i] - start[i]) * scores[i] for i in ids])
    )
    if use_captain:
        objective += captain_bonus_weight * pulp.lpSum([is_captain[i] * captain_scores[i] for i in ids])
    if use_hits:
        # `hits` carries a single negative objective coefficient and a single
        # binding lower bound, so a maximiser drives it to exactly
        # max(0, changes - free_transfers) -- FPL's rule, with no need to
        # constrain it from above for correctness.
        objective -= hit_cost * hits
    prob += objective

    # Constraints

    # Budget constraint
    prob += pulp.lpSum([select[i] * costs[i] for i in ids]) <= budget

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
    unique_teams = pool[team_col].unique()
    for t in unique_teams:
        prob += pulp.lpSum([select[i] for i in ids if teams[i] == t]) <= 3

    # At most `max_changes` players leave the current squad.
    if use_transfer_mode:
        n_kept = pulp.lpSum([select[i] for i in owned_in_pool])
        prob += n_kept >= SQUAD_SIZE - int(max_changes)
        if use_hits:
            prob += hits >= (SQUAD_SIZE - n_kept) - int(free_transfers)

    # Captain: must be a starter, exactly one per squad
    if use_captain:
        for i in ids:
            prob += is_captain[i] <= start[i]
        prob += pulp.lpSum([is_captain[i] for i in ids]) == 1

    # Solve
    solver = (pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
              if time_limit else pulp.PULP_CBC_CMD(msg=False))
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != 'Optimal':
        return None, None

    # Extract results
    selected_indices = [i for i in ids if pulp.value(select[i]) == 1]
    starting_indices = [i for i in ids if pulp.value(start[i]) == 1]

    squad_df = pool.loc[selected_indices].copy()
    squad_df['Is_Starter'] = squad_df.index.isin(starting_indices)

    # Sort: Starters first (by position G-D-M-F), then Bench
    pos_order = {'G': 1, 'D': 2, 'M': 3, 'F': 4}
    squad_df['Pos_Order'] = squad_df[position_col].map(pos_order)
    squad_df = squad_df.sort_values(
        by=['Is_Starter', 'Pos_Order', score_col],
        ascending=[False, True, False]
    )

    totals = {
        'starter_score': sum(scores[i] for i in starting_indices),
        'bench_score': sum(scores[i] for i in selected_indices if i not in starting_indices),
        'squad_cost': sum(costs[i] for i in selected_indices),
    }

    if use_transfer_mode:
        kept = sum(1 for i in selected_indices if pool.at[i, id_col] in owned_set)
        totals['owned_kept'] = kept
        totals['n_changes'] = SQUAD_SIZE - kept
        totals['hits'] = (max(0, totals['n_changes'] - int(free_transfers))
                          if free_transfers is not None else 0)

    if use_captain:
        captain_indices = [i for i in ids if pulp.value(is_captain[i]) == 1]
        squad_df['Is_Captain'] = squad_df.index.isin(captain_indices)
        if captain_indices:
            totals['captain_score'] = captain_scores[captain_indices[0]]

    return squad_df, totals


# =============================================================================
# TURNING A SOLVED SQUAD BACK INTO A LIST OF TRANSFERS
# =============================================================================

def diff_squads(before: pd.DataFrame, after: pd.DataFrame,
                id_col: str = "Player_ID") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(outs, ins) — who left the squad and who arrived."""
    before_ids = set(before[id_col])
    after_ids = set(after[id_col])
    outs = before[~before[id_col].isin(after_ids)]
    ins = after[~after[id_col].isin(before_ids)]
    return outs, ins


def pair_transfer_legs(outs: pd.DataFrame, ins: pd.DataFrame,
                       position_col: str = "Position",
                       out_price_col: str = "Sell_Price",
                       in_price_col: str = "Price",
                       score_col: Optional[str] = None,
                       id_col: str = "Player_ID",
                       name_col: str = "Player") -> List[Dict]:
    """Pair each outgoing player with an incoming one, and order the result.

    **A legal pairing always exists.** Both squads satisfy FPL's 2/5/5/3
    quotas, so for each position the number leaving equals the number arriving,
    and any within-position bijection is legal. The choice is presentational --
    and it is the only thing standing between the user and a card that reads
    "Haaland -> Wissa" with nothing to explain it.

    **Paired by price, descending, within position.** For a plan whose value is
    a reallocation -- sell the premium forward and a mid-price midfielder, buy
    a premium midfielder and a cheaper forward -- price pairing puts the
    released money next to the money spent: `Haaland £14.5m -> Wissa £7.5m`
    beside `mid-MID £7.0m -> Salah £14.0m` reads as one move. Pairing by
    points-delta instead manufactures one huge-gain leg and one huge-loss leg
    whenever two players share a position, misrepresenting a plan whose value
    is in the aggregate. Price is also stable: it is an exact integer in
    tenths, where score deltas reorder the cards on every slider nudge.

    **Then ordered by net cost ascending**, money-freeing legs first, so the
    running bank never goes negative as the user stages the transfers in FPL's
    own UI. Whenever the plan is affordable as a whole, this order is
    executable leg by leg.
    """
    legs: List[Dict] = []
    if outs is None or ins is None or outs.empty or ins.empty:
        return legs

    for pos in sorted(set(outs[position_col]) | set(ins[position_col])):
        pos_outs = outs[outs[position_col] == pos]
        pos_ins = ins[ins[position_col] == pos]

        # Ties broken on score so the ordering is deterministic: the weaker
        # player leaving is paired with the stronger arriving.
        def _order(frame, price_col, ascending_score):
            cols, asc = [price_col], [False]
            if score_col and score_col in frame.columns:
                cols.append(score_col)
                asc.append(ascending_score)
            return frame.sort_values(by=cols, ascending=asc)

        pos_outs = _order(pos_outs, out_price_col, True)
        pos_ins = _order(pos_ins, in_price_col, False)

        for (_, o), (_, i) in zip(pos_outs.iterrows(), pos_ins.iterrows()):
            out_price = float(o.get(out_price_col, o.get(in_price_col, 0)) or 0)
            in_price = float(i.get(in_price_col, 0) or 0)
            legs.append({
                "position": pos,
                "out_id": o.get(id_col), "out_player": o.get(name_col),
                "out_price": out_price,
                "in_id": i.get(id_col), "in_player": i.get(name_col),
                "in_price": in_price,
                "net_cost": in_price - out_price,
                "delta": ((float(i.get(score_col, 0) or 0) - float(o.get(score_col, 0) or 0))
                          if score_col and score_col in ins.columns else None),
                "out_row": o, "in_row": i,
            })

    legs.sort(key=lambda leg: leg["net_cost"])
    return legs
