"""Ownership *within a mini-league* -- the question no other surface answers.

Every projection in this app answers "how many points will he score". None
answered "how many points will that score me relative to the twelve people I am
actually playing against", which is the question the table turns on: a haul from
a player eleven of your rivals also own moves you nowhere.

The app had no notion of league ownership at all. `selected_by_percent` is
*global* ownership -- a statement about eleven million managers, not about your
twelve -- and it is the only ownership figure anything here rendered.

**Two ownership numbers, because they fail differently.**

    Own%(p) = 100 x (squads holding him in the 15) / n_squads
    EO%(p)  = 100 x sum_over_squads(multiplier) / n_squads

`Own%` is robust: it needs only the fifteen names, which every resolution path
supplies. `EO%` is the quantity that actually decides a mini-league, since a
captained player pays double, but it rests on the `multiplier` -- and for every
rival that is a *last gameweek* captain choice, because FPL publishes no picks
for a gameweek whose deadline has not passed. The page reports both and says
which squads are stale.

`multiplier` is 0 on the bench, 1 for a starter, 2 for the captain, 3 under
Triple Captain, and 1 for the bench under Bench Boost. So **EO can legitimately
exceed 100%** -- twelve of twelve owning and captaining is 300% -- and anything
bounding it at 100 is wrong.

**A player the pool does not carry is kept, not dropped.** The bootstrap can lag
a new signing by a day. Dropping him would silently break the one exact
invariant this module offers -- `sum(Owners) == 15 x n_squads`, which is what
`check_mini_league_ownership()` uses to catch a squad counted twice or quietly
lost -- so he is kept with a NaN projection and an id-derived name.

Streamlit-free and network-free: it computes over frames somebody else fetched,
which is what makes the arithmetic testable offline against hand-counted squads.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# FPL's Classic squad size. Not read from `settings` on purpose: a feed outage
# must not read as a rules change -- the same reasoning as the Draft quota
# constants, with `check_draft_game_settings()` as the tripwire for a real one.
SQUAD_SIZE = 15

# A player held by at most this share of the league is a differential. Half is
# the natural cut -- above it he is helping more rivals than he is helping you.
DEFAULT_DIFFERENTIAL_MAX_OWN = 50.0

# Effective ownership at which a player you do *not* own is an exposure rather
# than a mere omission.
DEFAULT_TEMPLATE_MIN_EO = 50.0

OWNERSHIP_COLUMNS = [
    "Player_ID", "Display_Name", "Position", "Team",
    "Owners", "Own_Pct", "EO_Pct", "Captains", "Mine",
    "Proj", "Proj_Next3",
]

TOTALS_COLUMNS = ["Team_ID", "Manager", "Proj_Total", "Gap_To_You", "Mine"]


def _pick_multiplier(pick: dict) -> float:
    """The pick's multiplier, derived from its slot only when absent.

    An *absent* multiplier is the un-normalised `my-team` payload, which
    publishes none (see `fpl_auth.normalise_my_team`). Deriving it from the slot
    is the safe reading. A multiplier that is *present* but flat at 1 across a
    whole squad is a different fault and deliberately left alone here, so
    `check_mini_league_ownership()` can see it.
    """
    mult = pick.get("multiplier")
    if mult is None:
        return 0.0 if int(pick.get("position", 0) or 0) > 11 else 1.0
    try:
        return float(mult)
    except (TypeError, ValueError):
        return 0.0


def compute_league_ownership(squads: Dict[int, List[dict]],
                             pool: Optional[pd.DataFrame] = None,
                             my_team_id: Optional[int] = None) -> pd.DataFrame:
    """One row per player owned by anyone in the league.

    Parameters
    ----------
    squads : team id -> that team's picks, as `get_classic_team_picks()` shapes
        them (`element`, `position`, `multiplier`, `is_captain`).
    pool : the blended projection frame (`scripts.common.player_pool`), joined
        on the integer `Player_ID`. This is the one cross-source join in the
        feature and it needs no name matching at all.
    my_team_id : the user's entry, for the `Mine` column. None is a legitimate
        state -- no Classic team configured, or the user outside the analysed
        subset -- and leaves `Mine` uniformly False.
    """
    if not squads:
        return pd.DataFrame(columns=OWNERSHIP_COLUMNS)

    n_squads = len(squads)
    owners: Dict[int, int] = {}
    weight: Dict[int, float] = {}
    captains: Dict[int, int] = {}
    mine: set = set()

    for team_id, picks in squads.items():
        for pick in picks or []:
            element = pick.get("element")
            if element is None:
                continue
            element = int(element)
            mult = _pick_multiplier(pick)
            owners[element] = owners.get(element, 0) + 1
            weight[element] = weight.get(element, 0.0) + mult
            if mult >= 2:
                captains[element] = captains.get(element, 0) + 1
            if my_team_id is not None and int(team_id) == int(my_team_id):
                mine.add(element)

    rows = [{
        "Player_ID": element,
        "Owners": count,
        "Own_Pct": 100.0 * count / n_squads,
        "EO_Pct": 100.0 * weight.get(element, 0.0) / n_squads,
        "Captains": captains.get(element, 0),
        "Mine": element in mine,
    } for element, count in owners.items()]

    df = pd.DataFrame(rows)
    df = _attach_pool_columns(df, pool)
    df = df.sort_values(["EO_Pct", "Own_Pct"], ascending=False).reset_index(drop=True)
    return df[OWNERSHIP_COLUMNS]


def _attach_pool_columns(df: pd.DataFrame,
                         pool: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Join name, position, club and projection on the element id.

    Every column is filled even when the pool is missing or short, so a caller
    never has to test for a column's existence -- `DataFrame.get()` is not a
    safe accessor in this codebase (see CLAUDE.md) and this is where that is
    paid for once.
    """
    wanted = ["Player_ID", "Display_Name", "Position", "Team", "Proj", "Proj_Next3"]
    if pool is not None and not pool.empty and "Player_ID" in pool.columns:
        cols = [c for c in wanted if c in pool.columns]
        slim = pool[cols].drop_duplicates(subset=["Player_ID"]).copy()
        slim["Player_ID"] = pd.to_numeric(slim["Player_ID"], errors="coerce").astype("Int64")
        df["Player_ID"] = pd.to_numeric(df["Player_ID"], errors="coerce").astype("Int64")
        df = df.merge(slim, on="Player_ID", how="left")

    for col in ("Display_Name", "Position", "Team"):
        if col not in df.columns:
            df[col] = pd.NA
    for col in ("Proj", "Proj_Next3"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # A player the pool does not carry keeps his row -- see the module docstring.
    unnamed = df["Display_Name"].isna() | (df["Display_Name"].astype(str).str.strip() == "")
    df.loc[unnamed, "Display_Name"] = "Player " + df.loc[unnamed, "Player_ID"].astype(str)
    df["Position"] = df["Position"].fillna("")
    df["Team"] = df["Team"].fillna("")
    return df


def differentials(ownership: pd.DataFrame,
                  max_own_pct: float = DEFAULT_DIFFERENTIAL_MAX_OWN) -> pd.DataFrame:
    """Players you own that the league largely does not, best projection first."""
    if ownership is None or ownership.empty:
        return pd.DataFrame(columns=OWNERSHIP_COLUMNS)
    picked = ownership[ownership["Mine"] & (ownership["Own_Pct"] <= max_own_pct)]
    return picked.sort_values("Proj", ascending=False, na_position="last").reset_index(drop=True)


def template_gaps(ownership: pd.DataFrame,
                  min_eo_pct: float = DEFAULT_TEMPLATE_MIN_EO) -> pd.DataFrame:
    """Players the league owns heavily and you do not -- the exposure half.

    Sorted by effective ownership rather than projection: the cost of missing a
    player is what the rest of the league gets from him, which is exactly what
    EO measures.
    """
    if ownership is None or ownership.empty:
        return pd.DataFrame(columns=OWNERSHIP_COLUMNS)
    picked = ownership[(~ownership["Mine"]) & (ownership["EO_Pct"] >= min_eo_pct)]
    return picked.sort_values("EO_Pct", ascending=False).reset_index(drop=True)


def template_score(ownership: pd.DataFrame) -> Optional[float]:
    """Mean effective ownership of your own squad -- "how template am I".

    None when the user's squad is not in the frame, which must not read as 0:
    "we do not know who you are" and "you own nothing the league owns" are
    different statements.
    """
    if ownership is None or ownership.empty or "Mine" not in ownership.columns:
        return None
    ours = ownership[ownership["Mine"]]
    if ours.empty:
        return None
    return float(ours["EO_Pct"].mean())


def projected_team_totals(squads: Dict[int, List[dict]],
                          pool: Optional[pd.DataFrame] = None,
                          team_names: Optional[Dict[int, str]] = None,
                          my_team_id: Optional[int] = None) -> pd.DataFrame:
    """Each team's projected gameweek total, and the gap to yours.

    `multiplier x Proj` summed over the fifteen is a projected score by
    construction: the bench carries a 0 and the captain a 2, so the arithmetic
    applies the lineup without the caller having to know it.

    `Proj` is the engine's *expected* points, with start probability already
    inside it -- never `Proj_Start`, which is "points if he starts" and would
    price every rotation risk as a nailed-on starter. (`team_strength.py`
    percentiles `Proj_Start` for the opposite and equally deliberate reason.)
    """
    if not squads:
        return pd.DataFrame(columns=TOTALS_COLUMNS)

    proj_by_id: Dict[int, float] = {}
    if pool is not None and not pool.empty and {"Player_ID", "Proj"} <= set(pool.columns):
        ids = pd.to_numeric(pool["Player_ID"], errors="coerce")
        vals = pd.to_numeric(pool["Proj"], errors="coerce")
        proj_by_id = {int(i): float(v) for i, v in zip(ids, vals)
                      if pd.notna(i) and pd.notna(v)}

    names = {int(k): v for k, v in (team_names or {}).items()}
    rows = []
    for team_id, picks in squads.items():
        total = 0.0
        for pick in picks or []:
            element = pick.get("element")
            if element is None:
                continue
            total += _pick_multiplier(pick) * proj_by_id.get(int(element), 0.0)
        rows.append({
            "Team_ID": int(team_id),
            "Manager": names.get(int(team_id), f"Team {team_id}"),
            "Proj_Total": total,
            "Mine": my_team_id is not None and int(team_id) == int(my_team_id),
        })

    df = pd.DataFrame(rows).sort_values("Proj_Total", ascending=False).reset_index(drop=True)

    ours = df[df["Mine"]]
    if ours.empty:
        df["Gap_To_You"] = np.nan
    else:
        df["Gap_To_You"] = df["Proj_Total"] - float(ours["Proj_Total"].iloc[0])

    return df[TOTALS_COLUMNS]
