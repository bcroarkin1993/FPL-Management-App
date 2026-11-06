"""
Renting a player versus keeping him — the two kinds of waiver claim.

The Waiver Wire ranked every claim with one number, ``Transfer Score`` =
``alpha * 1GW + (1 - alpha) * ROS``. That is a *fixed, hidden choice of holding
horizon* applied to every move, and it is therefore optimal for neither of the
two transactions a manager actually makes: the player you want for the rest of
the season, and the player you want for one gameweek and will drop again.

Reported from the app at GW10: Noni Madueke was the highest projected scorer
available at any position (``Proj`` 5.371, ``1GW`` 0.964 — top of the board) and
was suggested nowhere. He ranked 4th at midfield on ``Transfer Score`` because
Draft's alpha is ~0.40, so ROS carried 60% and his ROS was 0.546 — he had missed
seven gameweeks injured. His one threshold-clearing pairing was then taken off
him by a rival add under the one-add-one-drop assignment.

Measured over the live pool, ranking the same players on points over an explicit
horizon reproduces both transaction types exactly:

====== ==========================================
     H  top three available midfielders
====== ==========================================
     1  **Madueke**, Estevao, Guimaraes
     2  **Madueke**, Bobb, Estevao
     3  Bobb, **Madueke**, Ayari  (12.74 v 12.72)
     6  **Ayari**, Bobb, Madueke
====== ==========================================

``Transfer Score`` behaves like H~6. Retuning alpha cannot fix this: passing Bobb
needs alpha > 0.605, which would make Draft *more* single-gameweek-led than
Classic (0.55) and contradict the sound reason Draft is ROS-led in the first
place. The horizon is the lever, not the weight.

Two ideas make a short horizon expressible, and neither existed:

**A rental's future comes from the wire, not from the player.** You are going to
drop him, so his own thin horizon should never bind. What binds instead is what
you can expect to *re-claim*, which is estimable from the league's own
transaction log: ``waiver_priority.expected_taken_at_position()`` says how many
players go at a position before your pick, so the next-best available player
after those is your realistic floor. Live at pick 6 of 10, 2.38 midfielders go
first and the floor is 3.67 points a gameweek.

**Replacement cost belongs on the drop side.** ``Keep Score`` is a positional
percentile against ~700 players and has no idea what is *available*. Measured on
one live roster, Alex Iwobi's forward rate (3.64/GW) is **below** the midfield
floor — the wire offers his equal, so releasing him costs nothing — yet his Keep
Score of 0.80 sits behind the 0.12 "elite" drop threshold and made him
effectively undroppable.

So: ``rental_gain`` charges the *drop* for permanence and the *add* for one week
only. That asymmetry is the whole model. A keeper claim is the same arithmetic
with the add's own horizon counted, which is what ``keeper_value`` does.

**This amends a documented decision rather than overturning it.** CLAUDE.md rules
VORP out of the waiver search because a replacement-level term cancels in a
same-position swap: ``VORP(add) - VORP(drop) == add - drop``. That is correct
*when both players are held*. It does not cover a rental, whose future is the
wire's rather than the player's, so the cancellation does not apply and the
floor has real work to do.

Pure and Streamlit-free, for the same reason ``waiver_priority`` and
``transfer_risk`` are: offline unit tests, and nothing here should need a page
to exercise. ``_column`` duplicates ``analytics.numeric_col``'s guard for the
same stated reason ``projection_sources._column`` does — importing analytics
would pull in Streamlit.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from scripts.common.waiver_priority import expected_taken_at_position

_logger = logging.getLogger("fpl_app.claim_value")

#: The window every forward rate is measured over. ``Proj_Next3`` is the only
#: multi-gameweek projection the app carries, and the accuracy harness found
#: longer windows order *worse* (``projection_accuracy.score_horizon_lengths``:
#: rho 0.461 at 2GW, 0.459 at 3GW, 0.450 at 6GW, monotone after that). So three
#: is the evidenced window rather than a compromise.
#:
#: One consequence to state plainly: release cost is therefore priced over three
#: gameweeks, which **understates** the cost of releasing a genuinely good
#: player. That is the unsafe direction, which is why the rental card shows the
#: cost rather than only burying it inside the ranking.
HORIZON_GWS = 3

#: Below this many available players a position has no measurable floor. Live
#: there was exactly **one** available goalkeeper, so a "floor" there would be
#: one player's projection presented as a market rate.
MIN_FLOOR_POOL = 5

#: Horizons the page offers. ``None`` means "the whole window we have data for".
HORIZON_RENT = 1
HORIZON_HOLD = HORIZON_GWS

LANE_RENT = "rent"
LANE_HOLD = "hold"

_POS_CODES = ("G", "D", "M", "F")


def _column(df: pd.DataFrame, col: str) -> pd.Series:
    """A numeric Series for ``col``, all-NaN when the column is absent.

    The same guard as ``analytics.numeric_col``, repeated here because this
    module must not import from analytics (Streamlit) — the precedent is
    ``projection_sources._column``.
    """
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _scalar(row: Any, col: str) -> float:
    """One row's numeric value, or NaN."""
    try:
        val = row.get(col) if hasattr(row, "get") else row[col]
    except (KeyError, IndexError, TypeError):
        return float("nan")
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return float("nan") if math.isnan(out) else out


def forward_rate(df: pd.DataFrame, horizon: int = HORIZON_GWS) -> pd.Series:
    """Points per gameweek **after** this one, floored at zero.

    ``Proj_Next3`` *includes* the current gameweek (see CLAUDE.md, "Fantasy
    Football Pundit feed"), so the forward rate is
    ``(Proj_Next3 - Proj) / (horizon - 1)`` and **not** ``Proj_Next3 / 3``. That
    distinction decides the ordering here: on the live pool it is 3.06/GW for
    Madueke against the 3.83 the naive division gives, and 4.07 against 4.25 for
    Bobb — enough to swap them.

    NaN propagates rather than becoming zero. A player with no horizon is
    *unknown*, and treating unknown as "scores nothing from here" is what makes
    releasing him look free.
    """
    if horizon <= 1:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return ((_column(df, "Proj_Next3") - _column(df, "Proj"))
            / float(horizon - 1)).clip(lower=0.0)


def waiver_floors(
    available_df: Optional[pd.DataFrame],
    n_ahead: Optional[int],
    claim_rates: Optional[Dict[str, float]],
    horizon: int = HORIZON_GWS,
) -> Dict[str, float]:
    """What you can realistically re-claim at each position next round, pts/GW.

    ``expected_taken_at_position()`` gives ``E``, the number of players this
    league's log says go at a position before your pick. The ``floor(E) + 1``-th
    best available player is therefore the one you can expect to still be there,
    and his forward rate is the floor.

    **A position with no measurable floor is absent from the result, never zero.**
    A floor of zero makes every release look free, which does not degrade the
    rental lane — it *inverts* it. Three ways a floor is unmeasurable, all of
    which fail closed:

    * ``E`` is None — the league has no claim history, or free agency is open and
      there is no queue to be behind. ``_build_waiver_context()`` already returns
      an empty context in free agency, so an empty ``claim_rates`` arrives here
      and the lane switches itself off.
    * fewer than ``MIN_FLOOR_POOL`` players available at that position. Live,
      goalkeeper had one.
    * no player at that position has a readable forward rate.

    ``n_ahead == 0`` is the one case that needs no history: you claim first, so
    nothing can be taken ahead of you and ``E`` is exactly zero.
    """
    floors: Dict[str, float] = {}
    if available_df is None or available_df.empty or "Position" not in available_df.columns:
        return floors

    rates = claim_rates or {}
    rate_df = available_df.assign(_fwd=forward_rate(available_df, horizon))

    for pos in _POS_CODES:
        pool = rate_df[rate_df["Position"] == pos]
        if len(pool) < MIN_FLOOR_POOL:
            continue

        if n_ahead == 0:
            expected = 0.0
        else:
            expected = expected_taken_at_position(pos, n_ahead, rates)
            if expected is None:
                continue

        ranked = pool["_fwd"].dropna().sort_values(ascending=False)
        if ranked.empty:
            continue
        idx = min(int(math.floor(expected)), len(ranked) - 1)
        floors[pos] = float(ranked.iloc[idx])

    return floors


def release_cost(
    df: pd.DataFrame,
    floors: Dict[str, float],
    horizon: int = HORIZON_GWS,
) -> pd.Series:
    """Points per gameweek permanently given up by releasing each player.

    ``max(0, forward_rate - floor)`` — net of what the wire would hand back. A
    player at or below his position's floor costs **nothing** to release, which
    is the fact ``Keep Score`` cannot express: it ranks him against the whole
    FPL pool rather than against the players you could actually claim instead.

    NaN where the position has no floor or the player has no forward rate.
    Unknown is not zero here: a zero would say "free to release" about a player
    nobody has priced.
    """
    fwd = forward_rate(df, horizon)
    if "Position" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    floor = df["Position"].map(floors).astype("float64")
    return (fwd - floor).clip(lower=0.0)


def rental_gain(
    drop_row: Any,
    add_row: Any,
    floors: Dict[str, float],
    horizon: int = HORIZON_GWS,
) -> Optional[float]:
    """Points gained by claiming ``add`` for one gameweek and releasing ``drop``.

    ``(Proj(add) - Proj(drop)) - (horizon - 1) * release_cost(drop)``

    The add is temporary and the drop is permanent, and that asymmetry is the
    model. The add's own horizon never appears: you are going to drop him, so
    what follows this gameweek is the wire's floor for both players and cancels.
    What does not cancel is the drop — he is gone, and if he was worth more than
    the floor you have given up the difference for the rest of the window.

    Verified against the live roster: it refuses to rent over Mathys Tel
    (release cost 0.88/GW, gain -0.41) and is near-indifferent on Dewsbury-Hall
    (0.13/GW, +0.17), while Madueke over Joao Gomes is +3.53. Refusing the
    expensive drops is what makes the lane trustworthy.

    None when either projection or the drop's release cost is unknown — see
    ``release_cost``.
    """
    add_proj = _scalar(add_row, "Proj")
    drop_proj = _scalar(drop_row, "Proj")
    if math.isnan(add_proj) or math.isnan(drop_proj):
        return None

    pos = drop_row.get("Position") if hasattr(drop_row, "get") else None
    floor = floors.get(pos) if pos is not None else None
    if floor is None:
        return None

    drop_fwd = _scalar(drop_row, "Proj_Next3")
    if math.isnan(drop_fwd):
        return None
    fwd = max(0.0, (drop_fwd - drop_proj) / float(max(1, horizon - 1)))
    cost = max(0.0, fwd - float(floor))

    return (add_proj - drop_proj) - float(horizon - 1) * cost


def keeper_value(
    df: pd.DataFrame,
    floors: Dict[str, float],
    horizon: int = HORIZON_GWS,
) -> pd.Series:
    """Points over ``horizon`` gameweeks from claiming and holding each player.

    ``Proj + (horizon - 1) * max(forward_rate, floor)``

    The ``max`` is the option the single-score model ignores: you can always drop
    him and re-claim, so no player's value is bounded below by his own thin
    horizon. It is also what makes "rent or keep" fall out rather than needing a
    separate rule — the max is attained at the floor exactly when renting beats
    holding, which is what ``recommended_lane`` reads.

    Where a position has no floor this degrades to ``Proj + (horizon - 1) *
    forward_rate``, i.e. the honest no-option answer, rather than refusing.
    """
    fwd = forward_rate(df, horizon)
    if "Position" in df.columns:
        floor = df["Position"].map(floors).astype("float64")
        best = pd.concat([fwd, floor], axis=1).max(axis=1, skipna=True)
    else:
        best = fwd
    return _column(df, "Proj") + float(horizon - 1) * best


def recommended_lane(
    row: Any,
    floors: Dict[str, float],
    horizon: int = HORIZON_GWS,
) -> str:
    """``LANE_RENT`` where the wire beats this player's own horizon, else hold.

    Falls back to hold wherever the comparison cannot be made, which is the
    conservative direction: a hold is advice about the player, a rental is
    advice that depends on being able to claim again.
    """
    pos = row.get("Position") if hasattr(row, "get") else None
    floor = floors.get(pos) if pos is not None else None
    if floor is None:
        return LANE_HOLD
    proj = _scalar(row, "Proj")
    n3 = _scalar(row, "Proj_Next3")
    if math.isnan(proj) or math.isnan(n3):
        return LANE_HOLD
    fwd = max(0.0, (n3 - proj) / float(max(1, horizon - 1)))
    return LANE_RENT if fwd < float(floor) else LANE_HOLD
