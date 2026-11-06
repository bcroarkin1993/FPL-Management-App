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
    """Has this match been played?

    **`finished` cannot be trusted on its own.** Draft sets it to a real bool,
    but Classic's H2H payload leaves it ``None`` on every match -- played or
    not -- so a truth test alone marks the entire season unplayed. Recorded
    points are the reliable signal: a fixture nobody has scored in has not
    happened, and one with a score has.

    Without this the only thing standing between a Classic league and replaying
    its played fixtures is the caller remembering ``from_gw`` -- and the damage
    would be invisible, since double-counting inflates every total without
    disturbing the rank permutation `check_season_odds` tests.
    """
    if match.get("finished"):
        return True
    for key in ("league_entry_1_points", "league_entry_2_points",
                "entry_1_points", "entry_2_points"):
        try:
            if float(match.get(key) or 0) != 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


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


# =============================================================================
# TOTAL-POINTS LEAGUES
# =============================================================================
#
# A Classic league with no head-to-head fixtures is a different simulation, not
# the same one with an empty schedule: nobody plays anybody, and the table is
# ordered on accumulated points. So the question is how each manager's *total*
# finishes, and the fixtures drop out entirely.

#: Weekly points spread for a single Classic manager, when a league has no
#: history to measure one from. Deliberately wider than the Draft prior: a
#: Classic squad takes captaincy and chips, so its week-to-week range is larger.
DEFAULT_CLASSIC_SIGMA = 20.0


def weekly_score_sigma(weekly: pd.DataFrame, score_col: str = "GW_Points",
                       team_col: str = "Team") -> tuple:
    """``(sigma, n)`` -- the **within-manager** weekly spread, pooled.

    Pooled within-team rather than the spread of all scores together, because
    those are two different quantities and only one of them is noise. The spread
    *between* managers is squad quality -- exactly the thing the simulation is
    trying to carry forward -- so folding it into sigma would make every
    manager look equally likely to score anything and flatten the table towards
    a coin toss.

    Returns ``(None, 0)`` when there is too little history to measure, so a
    caller can fall back to the documented prior and say that it did.
    """
    if weekly is None or weekly.empty or score_col not in weekly.columns:
        return None, 0
    if team_col not in weekly.columns:
        return None, 0
    vals = weekly[[team_col, score_col]].copy()
    vals[score_col] = pd.to_numeric(vals[score_col], errors="coerce")
    vals = vals.dropna()
    # Each team's deviations from its own mean; two observations is the minimum
    # that says anything about a spread.
    devs, n = [], 0
    for _, grp in vals.groupby(team_col):
        if len(grp) < 2:
            continue
        devs.append(grp[score_col] - grp[score_col].mean())
        n += len(grp)
    if not devs:
        return None, 0
    pooled = pd.concat(devs)
    # The pooled estimate loses one degree of freedom per team, not one overall.
    k = int(vals[team_col].nunique())
    denom = max(1, len(pooled) - k)
    sigma = float(np.sqrt(float((pooled ** 2).sum()) / denom))
    if not math.isfinite(sigma) or sigma <= 0:
        return None, 0
    return sigma, n


def simulate_points_league(
    entries: Sequence[Dict[str, Any]],
    *,
    gameweeks_left: int,
    sigma: float = DEFAULT_CLASSIC_SIGMA,
    n_sims: int = DEFAULT_SIMS,
    league_id: Optional[int] = None,
    gameweek: Optional[int] = None,
    seed: Optional[int] = None,
    propagate_estimation_error: bool = True,
    playoff_places: int = 3,
) -> pd.DataFrame:
    """Where a **total-points** league finishes.

    ``entries`` is ``[{entry, total, mean_score, n_played}]``. Each remaining
    gameweek draws a score per manager and adds it to the running total; the
    table is then ordered on that total alone, because that is how a Classic
    league is ranked.

    Same two commitments as the head-to-head version. The manager's underlying
    mean is re-drawn from its own sampling distribution each simulation, since it
    rests on the gameweeks played so far -- and the result is seeded, so a
    Streamlit rerun cannot change the odds.

    With no gameweeks left this reports the table as it stands, which is correct
    rather than a degenerate case.
    """
    rows = [e for e in (entries or []) if e.get("entry") is not None]
    if not rows:
        return pd.DataFrame()

    sigma = float(sigma) if sigma and math.isfinite(sigma) and sigma > 0 else DEFAULT_CLASSIC_SIGMA
    gws = max(0, int(gameweeks_left or 0))

    ids = [e["entry"] for e in rows]
    totals = np.array([float(e.get("total") or 0) for e in rows], dtype=float)
    played = np.array([float(e.get("n_played") or 0) for e in rows], dtype=float)
    means = np.array([float(e.get("mean_score") or float("nan")) for e in rows], dtype=float)

    known = means[np.isfinite(means) & (played >= MIN_MATCHES_FOR_OWN_MEAN)]
    league_mean = float(np.mean(known)) if known.size else 50.0
    thin = ~(np.isfinite(means) & (played >= MIN_MATCHES_FOR_OWN_MEAN))
    means = np.where(thin, league_mean, means)

    se = sigma / np.sqrt(np.maximum(1.0, played))
    n_teams = len(ids)

    rng = np.random.default_rng(
        seed if seed is not None else _seed_from(league_id, gameweek, n_sims))

    titles = np.zeros(n_teams)
    top_n = np.zeros(n_teams)
    bottoms = np.zeros(n_teams)
    rank_sum = np.zeros(n_teams)
    points_sum = np.zeros(n_teams)

    for _ in range(int(n_sims)):
        mu_sim = rng.normal(means, se) if propagate_estimation_error else means
        final = totals.copy()
        if gws:
            # One draw per manager per remaining gameweek, summed.
            final = final + rng.normal(
                mu_sim[:, None] * np.ones((1, gws)), sigma).sum(axis=1)
        order = np.argsort(-final, kind="stable")
        ranks = np.empty(n_teams, dtype=int)
        ranks[order] = np.arange(1, n_teams + 1)

        titles += (ranks == 1)
        top_n += (ranks <= int(playoff_places))
        bottoms += (ranks == n_teams)
        rank_sum += ranks
        points_sum += final

    return pd.DataFrame({
        "league_entry": ids,
        "title": titles / n_sims,
        "top_n": top_n / n_sims,
        "bottom": bottoms / n_sims,
        "mean_rank": rank_sum / n_sims,
        "mean_points": points_sum / n_sims,
        "mean_score": means,
        "n_played": played,
        "points": totals,
        "points_for": totals,
    }).sort_values("mean_rank").reset_index(drop=True)
