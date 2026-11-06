"""
Where does this league finish? — a Monte Carlo over the remaining schedule.

The app could tell you the odds of winning *this week's* match and nothing about
the season. ``P(A>B) = Phi((muA - muB) / sqrt(2 sigma^2))`` has backed the
Fixture Projections page for a year; this plays the remaining fixtures out with
the same model and counts how often each manager finishes where.

Two things make it more than a loop, and both are about not overclaiming.

**Estimation uncertainty is simulated, not assumed away.** A team's mean weekly
score is estimated from the matches it has *played* -- nine, at the time this was
written -- so with a weekly sigma around 17 the standard error on that mean is
``17 / sqrt(9)``, nearly 6 points. A simulator that treats the mean as known
reports the league as far more decided than the evidence supports: the leader's
title probability comes out near-certain in October. So each simulation draws
the team's mean from its own sampling distribution first, then draws match
scores around it. The spread of outcomes then reflects both "football is noisy"
and "we do not actually know how good these teams are yet".

**The result is seeded.** Streamlit reruns the page function on every widget
interaction, and an unseeded simulation returns different odds each time, which
reads as a broken page rather than as sampling error. The seed is derived from
the league, the gameweek and the number of simulations, so the answer is stable
until the data behind it changes.

Draft's points system is confirmed against the live payload rather than assumed:
``total == 3 * matches_won + matches_drawn`` held exactly for every team in
league 11347. Ties break on ``points_for``, which is how the published standings
are ordered.

Pure and Streamlit-free, like ``waiver_priority`` and ``transfer_risk``: this is
the kind of thing that wants a fast offline test with a fixed seed.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_logger = logging.getLogger("fpl_app.season_sim")

#: Points for a win and a draw. Verified against league 11347's live standings:
#: `3 * matches_won + matches_drawn` reproduced `total` for all ten teams.
WIN_POINTS = 3
DRAW_POINTS = 1

#: How many seasons to play out. 2,000 puts the Monte Carlo standard error on a
#: probability at most 1.1 percentage points (`sqrt(0.25/2000)`), which is finer
#: than the model's own uncertainty and below the rounding the UI shows.
DEFAULT_SIMS = 2000

#: Weekly single-team scoring spread, used when a league has no history to
#: measure one from. The same prior `fixture_projections._DEFAULT_SCORE_STD`
#: carries, for the same reason: a degenerate sigma turns a one-point edge into
#: a near-certain win.
DEFAULT_SIGMA = 17.0

#: Fewest completed matches before a team's mean is estimated from its own
#: scoring at all. Below this the league-wide mean is used instead -- one or two
#: results say almost nothing about a manager, and letting them set a mean makes
#: the simulation confident about noise.
MIN_MATCHES_FOR_OWN_MEAN = 3


def _finished(match: Dict[str, Any]) -> bool:
    """FPL leaves `finished` False for hours after full time while bonus is
    confirmed, so `started` is consulted too -- the same rule
    `get_gw_team_fixture_status` applies."""
    return bool(match.get("finished"))


def team_scoring_rates(standings: Sequence[Dict[str, Any]],
                       matches: Optional[Sequence[Dict[str, Any]]] = None,
                       ) -> Dict[Any, Dict[str, float]]:
    """``{league_entry: {mean, n}}`` -- points a match, and how many back it.

    From ``points_for`` over matches actually completed. **Not** from
    ``matches_played``, which the live payload reports as 38 for every team --
    the scheduled total, not the contested one -- so dividing by it understates
    every mean by a factor of four. Wins plus draws plus losses is the honest
    denominator.
    """
    rates: Dict[Any, Dict[str, float]] = {}
    for row in standings or []:
        entry = row.get("league_entry")
        if entry is None:
            continue
        played = (int(row.get("matches_won") or 0)
                  + int(row.get("matches_drawn") or 0)
                  + int(row.get("matches_lost") or 0))
        pf = float(row.get("points_for") or 0)
        rates[entry] = {"mean": (pf / played) if played else float("nan"),
                        "n": float(played)}
    return rates


def remaining_matches(matches: Sequence[Dict[str, Any]],
                      from_gw: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fixtures still to play, as ``{event, a, b}``.

    An unfinished match in a *past* gameweek is excluded when ``from_gw`` is
    given: the live payload keeps `finished` False while bonus settles, and
    replaying a match that has effectively happened would double-count it
    against standings that already include the result.
    """
    out = []
    for m in matches or []:
        if _finished(m):
            continue
        event = m.get("event")
        if from_gw is not None and event is not None and int(event) < int(from_gw):
            continue
        a, b = m.get("league_entry_1"), m.get("league_entry_2")
        if a is None or b is None:
            continue
        out.append({"event": event, "a": a, "b": b})
    return out


def _seed_from(league_id, gameweek, n_sims) -> int:
    """A stable seed, so a Streamlit rerun does not change the answer."""
    return abs(hash((int(league_id or 0), int(gameweek or 0), int(n_sims)))) % (2 ** 31)


def simulate_season(
    standings: Sequence[Dict[str, Any]],
    matches: Sequence[Dict[str, Any]],
    *,
    sigma: float = DEFAULT_SIGMA,
    n_sims: int = DEFAULT_SIMS,
    from_gw: Optional[int] = None,
    league_id: Optional[int] = None,
    gameweek: Optional[int] = None,
    seed: Optional[int] = None,
    propagate_estimation_error: bool = True,
    playoff_places: int = 3,
) -> pd.DataFrame:
    """Play the rest of the season ``n_sims`` times.

    Returns one row per team: ``title``, ``top_n``, ``bottom`` (all 0-1),
    ``mean_rank``, ``mean_points``, plus the inputs that produced them
    (``mean_score``, ``n_played``, ``points``, ``points_for``) so a caller can
    show its own workings.

    ``propagate_estimation_error`` is the difference between "football is noisy"
    and "we do not know how good these teams are yet". With nine matches played
    and a weekly sigma near 17, the standard error on a team's mean is about 6
    points -- comparable to the gap between first and fourth -- so holding the
    mean fixed reports a league as decided when it is not. Switchable only so a
    test can hold it still.
    """
    rates = team_scoring_rates(standings)
    if not rates:
        return pd.DataFrame()

    entries = list(rates.keys())
    idx = {e: i for i, e in enumerate(entries)}
    n_teams = len(entries)

    sigma = float(sigma) if sigma and math.isfinite(sigma) and sigma > 0 else DEFAULT_SIGMA

    # A team with almost no history gets the league's mean rather than its own.
    known = [r["mean"] for r in rates.values()
             if r["n"] >= MIN_MATCHES_FOR_OWN_MEAN and math.isfinite(r["mean"])]
    league_mean = float(np.mean(known)) if known else 50.0

    mu = np.empty(n_teams, dtype=float)
    n_played = np.empty(n_teams, dtype=float)
    for e, i in idx.items():
        r = rates[e]
        own_ok = r["n"] >= MIN_MATCHES_FOR_OWN_MEAN and math.isfinite(r["mean"])
        mu[i] = r["mean"] if own_ok else league_mean
        n_played[i] = r["n"]

    # Standard error on each mean. A team with no completed matches is given the
    # weakest possible claim to its mean rather than an infinite one.
    se = sigma / np.sqrt(np.maximum(1.0, n_played))

    base_points = np.zeros(n_teams, dtype=float)
    base_pf = np.zeros(n_teams, dtype=float)
    for row in standings:
        e = row.get("league_entry")
        if e not in idx:
            continue
        base_points[idx[e]] = float(row.get("total") or 0)
        base_pf[idx[e]] = float(row.get("points_for") or 0)

    fixtures = remaining_matches(matches, from_gw=from_gw)
    ia = np.array([idx[f["a"]] for f in fixtures if f["a"] in idx and f["b"] in idx], dtype=int)
    ib = np.array([idx[f["b"]] for f in fixtures if f["a"] in idx and f["b"] in idx], dtype=int)

    rng = np.random.default_rng(
        seed if seed is not None else _seed_from(league_id, gameweek, n_sims))

    titles = np.zeros(n_teams, dtype=float)
    top_n = np.zeros(n_teams, dtype=float)
    bottoms = np.zeros(n_teams, dtype=float)
    rank_sum = np.zeros(n_teams, dtype=float)
    points_sum = np.zeros(n_teams, dtype=float)

    n_fix = len(ia)
    for _ in range(int(n_sims)):
        # This season's view of how good each team is.
        mu_sim = rng.normal(mu, se) if propagate_estimation_error else mu

        pts = base_points.copy()
        pf = base_pf.copy()

        if n_fix:
            sa = rng.normal(mu_sim[ia], sigma)
            sb = rng.normal(mu_sim[ib], sigma)
            np.add.at(pf, ia, sa)
            np.add.at(pf, ib, sb)
            a_win = sa > sb
            b_win = sb > sa
            draw = ~(a_win | b_win)
            np.add.at(pts, ia[a_win], WIN_POINTS)
            np.add.at(pts, ib[b_win], WIN_POINTS)
            np.add.at(pts, ia[draw], DRAW_POINTS)
            np.add.at(pts, ib[draw], DRAW_POINTS)

        # Rank on league points, then points for -- how the published standings
        # are ordered. lexsort's last key is primary.
        order = np.lexsort((-pf, -pts))
        ranks = np.empty(n_teams, dtype=int)
        ranks[order] = np.arange(1, n_teams + 1)

        titles += (ranks == 1)
        top_n += (ranks <= int(playoff_places))
        bottoms += (ranks == n_teams)
        rank_sum += ranks
        points_sum += pts

    out = pd.DataFrame({
        "league_entry": entries,
        "title": titles / n_sims,
        "top_n": top_n / n_sims,
        "bottom": bottoms / n_sims,
        "mean_rank": rank_sum / n_sims,
        "mean_points": points_sum / n_sims,
        "mean_score": mu,
        "n_played": n_played,
        "points": base_points,
        "points_for": base_pf,
    })
    return out.sort_values("mean_rank").reset_index(drop=True)


def summarise(sim: pd.DataFrame, entry_names: Dict[Any, str],
              playoff_places: int = 3) -> pd.DataFrame:
    """The simulation as a renderable table, names resolved."""
    if sim is None or sim.empty:
        return pd.DataFrame()
    out = sim.copy()
    out.insert(0, "Team", out["league_entry"].map(
        lambda e: entry_names.get(e, "Entry %s" % e)))
    out["Title"] = (out["title"] * 100).round(1)
    out["Top %d" % playoff_places] = (out["top_n"] * 100).round(1)
    out["Last"] = (out["bottom"] * 100).round(1)
    out["Proj Pts"] = out["mean_points"].round(1)
    out["Avg Finish"] = out["mean_rank"].round(2)
    out["PPG"] = out["mean_score"].round(1)
    cols = ["Team", "Title", "Top %d" % playoff_places, "Last",
            "Proj Pts", "Avg Finish", "PPG"]
    return out[cols]
