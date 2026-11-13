"""Head-to-head helpers: who you play, how noisy the league is, what they score.

Pure and Streamlit-free. Lifted out of ``scripts/classic/chip_strategy.py`` when
the Classic Home alert strip became a second reader -- the third time one of that
page's privates has been needed elsewhere, after ``price_signals`` and
``player_pool.squad_frame``. That page re-exports every name below under the
private spelling it used, so it is unchanged.

Two properties are load-bearing wherever these are used:

* **Only *scheduled* gameweeks appear** in a schedule. A win probability is
  undefined where there is no fixture, and inventing an average opponent would
  make a lane look informative precisely where it knows nothing.
* **A rival's squad is last deadline's**, because `my-team` is the user's own
  credential, so anything projected from it assumes they make no transfers --
  and their *lineup* must be the optimal XI rather than their stored
  multipliers, which are last deadline's choices and would project every rival
  as benching their best player.
"""

import pandas as pd

from scripts.common.error_helpers import get_logger
from scripts.common.text_helpers import _map_position_to_rw

_logger = get_logger("fpl_app.common.h2h")


def is_h2h_league(league_id) -> bool:
    """``True`` if this Classic league is scored head-to-head."""
    from scripts.common.fpl_classic_api import get_classic_or_h2h_league_standings
    data = get_classic_or_h2h_league_standings(league_id)
    info = (data or {}).get("league", {}) or {}
    return info.get("league_type") == "s" or info.get("scoring") == "h"


def h2h_schedule(league_id, team_id, gameweeks) -> dict:
    """``{gameweek: opponent entry id}`` for the window, from the real schedule."""
    from scripts.common.fpl_classic_api import get_all_h2h_league_matches
    wanted = {int(g) for g in gameweeks}
    out = {}
    for match in get_all_h2h_league_matches(league_id) or []:
        event = match.get("event")
        if event is None or int(event) not in wanted:
            continue
        a, b = match.get("entry_1_entry"), match.get("entry_2_entry")
        if a is None or b is None:
            continue
        if int(a) == int(team_id):
            out[int(event)] = int(b)
        elif int(b) == int(team_id):
            out[int(event)] = int(a)
    return out


def h2h_sigma(league_id):
    """``(sigma, n)`` -- the league's own weekly scoring spread, or the prior.

    Measured from the H2H match payload, which carries every manager's score for
    every played fixture, so this is a cache hit rather than N requests. Pooled
    *within* manager by `season_sim.weekly_score_sigma`: the spread between
    managers is squad quality, which is the thing being projected, and folding it
    in would flatten every fixture towards a coin toss.
    """
    from scripts.common.fpl_classic_api import get_all_h2h_league_matches
    from scripts.common.season_sim import DEFAULT_SIGMA, weekly_score_sigma
    rows = []
    for match in get_all_h2h_league_matches(league_id) or []:
        for side in ("1", "2"):
            entry, pts = match.get(f"entry_{side}_entry"), match.get(f"entry_{side}_points")
            if entry is None or not pts:
                continue
            rows.append({"Team": int(entry), "GW_Points": float(pts)})
    if not rows:
        return DEFAULT_SIGMA, 0
    sigma, n = weekly_score_sigma(pd.DataFrame(rows))
    return (sigma if sigma else DEFAULT_SIGMA), n


def h2h_entry_names(league_id) -> dict:
    """``{entry id: team name}`` for an H2H league's standings.

    An opponent's *name* is the only part of a head-to-head row a reader
    recognises, and the match payload carries it alongside the ids.
    """
    from scripts.common.fpl_classic_api import get_all_h2h_league_matches
    out = {}
    for match in get_all_h2h_league_matches(league_id) or []:
        for side in ("1", "2"):
            entry = match.get(f"entry_{side}_entry")
            name = match.get(f"entry_{side}_name")
            if entry is not None and name:
                out[int(entry)] = str(name)
    return out


def opponent_squad(entry_id, bootstrap, current_gw, pool):
    """A rival's fifteen, resolved through the one shared resolver.

    ``allow_auth`` is left at its default, which resolves False for anybody but
    the configured user -- so this can never reach for a credential on someone
    else's behalf.
    """
    from scripts.common.classic_squad import resolve_classic_squad
    from scripts.common.player_pool import squad_frame
    try:
        resolution = resolve_classic_squad(entry_id, bootstrap, current_gw)
    except Exception as exc:
        _logger.warning("Could not resolve rival %s: %s", entry_id, exc)
        return None
    if not resolution.ok:
        return None
    return squad_frame(resolution.picks, pool)


def lineup_projection(squad_df, matrix, gw):
    """A squad's projected total for ``gw``: best legal XI, best captain doubled.

    Deliberately the *optimal* lineup rather than the stored multipliers -- see
    the module docstring.
    """
    from scripts.common.optimization import best_xi_index
    if squad_df is None or squad_df.empty or matrix is None or gw not in matrix.columns:
        return None
    points = pd.to_numeric(matrix[gw], errors="coerce").fillna(0.0)
    frame = pd.DataFrame({"Position": squad_df["Position"].map(_map_position_to_rw),
                          "_p": points})
    chosen = best_xi_index(frame, "_p")
    if not chosen:
        return None
    xi = points.loc[chosen]
    return float(xi.sum() + xi.max())      # the captain scores twice
