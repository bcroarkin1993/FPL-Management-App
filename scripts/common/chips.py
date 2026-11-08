"""What is a chip worth, and which gameweek should spend it?

The app could tell you which chips you still owned and nothing about when to
play them. ``_render_chip_advisor`` spoke only about the Wildcard, from three
thresholds on the *mean of a positional percentile*, and Bench Boost, Triple
Captain and Free Hit had no recommendation logic anywhere in the codebase.

Two things are wrong with answering this in percentiles, and both are recorded
in CLAUDE.md under "Initial Squad Model" and "Multi-Transfer Planner". A
percentile saturates near the top -- 213.7 season points against 178.3 is 0.974
against 0.977, so the premium ranks *lower* -- and it inverts across positions,
since the best of 32 forwards scores 31/32 while the best of 48 midfielders
scores 47/48. Neither can answer "is spending this resource worth it". A chip is
worth a number of points, so that is the currency here.

**The rules are read, not derived.** ``bootstrap-static.chips`` publishes all
eight chips with an explicit ``start_event``/``stop_event``, and nothing in this
app had ever read it. Measured live on 2026-11-06, all four chips are
double-use -- 2-19 and 20-38 for Wildcard and Free Hit, 1-19 and 20-38 for
Bench Boost and Triple Captain -- while ``_parse_chip_status`` modelled Free Hit
and Triple Captain as single-use, so a manager who had played one was told it
was gone when a second opens at GW20. Same situation as the Draft deadlines
before "Deadlines are published too -- stop deriving them".

**Who you play changes the answer, and not the way it is usually put.** In an
H2H league a chip's value is the change in win probability it buys, not its
points, and differentiating ``Phi((d + g) / (sqrt(2) sigma)) - Phi(d / ...)``
puts the maximum at ``d = -g/2``: trailing by half the chip's gain. At the
measured Classic H2H sigma of 17.3, a 12-point Bench Boost is worth +0.194 of
win probability when 6 points behind, +0.188 when level, and only +0.023 when 45
ahead -- so a weak opponent does waste it, eight times over. But a *much
stronger* one wastes it too (+0.056 at 45 behind), because you lose anyway. The
rule the arithmetic supports is "spend it where it is just big enough to turn a
narrow projected loss into a narrow win", which is a close opponent rather than
a strong one.

Pure and Streamlit-free, like ``season_sim`` / ``waiver_priority`` /
``claim_value``: every function here is offline-testable with no network, and
the formation arithmetic is borrowed from ``optimization.best_xi_index`` rather
than re-implemented, because a second copy of a formation solver would mislabel
which gameweek to spend a chip on while every number in the grid stayed
plausible.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import pandas as pd

from scripts.common.optimization import SQUAD_SIZE, best_xi_index
from scripts.common.projection_engine import player_id_column
from scripts.common.text_helpers import _map_position_to_rw

_logger = logging.getLogger("fpl_app.chips")

# --------------------------------------------------------------------------
# The chips themselves
# --------------------------------------------------------------------------

CHIP_WILDCARD = "wildcard"
CHIP_FREEHIT = "freehit"
CHIP_BBOOST = "bboost"
CHIP_3XC = "3xc"

#: The chips this module knows how to value. ``manager`` (Assistant Manager)
#: is deliberately absent: it is not in this season's published list, and a
#: label for a chip nobody has is worse than no label.
KNOWN_CHIPS = (CHIP_WILDCARD, CHIP_BBOOST, CHIP_3XC, CHIP_FREEHIT)

CHIP_LABELS = {
    CHIP_WILDCARD: "Wildcard",
    CHIP_BBOOST: "Bench Boost",
    CHIP_3XC: "Triple Captain",
    CHIP_FREEHIT: "Free Hit",
}

CHIP_COLORS = {
    CHIP_WILDCARD: "#7c3aed",
    CHIP_BBOOST: "#166534",
    CHIP_3XC: "#d97706",
    CHIP_FREEHIT: "#0891b2",
}

#: Chips that can be withdrawn before the deadline. Wildcard and Free Hit
#: cannot be cancelled once confirmed, which is a real asymmetry in how much a
#: recommendation costs if it is wrong.
CANCELLABLE_CHIPS = (CHIP_BBOOST, CHIP_3XC)

LAST_GAMEWEEK = 38


class ChipWindow(NamedTuple):
    """One playable chip and the gameweeks it may be played in."""

    name: str
    start_event: int
    stop_event: int
    chip_type: str = ""


#: The fallback when ``bootstrap-static.chips`` cannot be read, matching the
#: live payload measured on 2026-11-06. It exists so a feed outage reads as
#: "we could not check" rather than "you have no chips" -- and like
#: ``projection_engine.DEFAULT_START_FLOORS`` tracking ``config.py``, it has to
#: be kept honest, which is what ``check_chip_definitions()`` is for.
DEFAULT_CHIP_WINDOWS: Tuple[ChipWindow, ...] = (
    ChipWindow(CHIP_WILDCARD, 2, 19, "transfer"),
    ChipWindow(CHIP_WILDCARD, 20, 38, "transfer"),
    ChipWindow(CHIP_FREEHIT, 2, 19, "transfer"),
    ChipWindow(CHIP_FREEHIT, 20, 38, "transfer"),
    ChipWindow(CHIP_BBOOST, 1, 19, "team"),
    ChipWindow(CHIP_BBOOST, 20, 38, "team"),
    ChipWindow(CHIP_3XC, 1, 19, "team"),
    ChipWindow(CHIP_3XC, 20, 38, "team"),
)


def parse_chip_definitions(bootstrap: Optional[dict]) -> List[ChipWindow]:
    """The chip rulebook, from ``bootstrap-static.chips``.

    **Fails open to** :data:`DEFAULT_CHIP_WINDOWS`. An unreachable or reshaped
    bootstrap means "unknown", and reporting unknown as "no chips available"
    would silently switch the whole page off -- the same reasoning that makes an
    unreachable bootstrap a *warning* rather than an error in
    ``check_draft_game_settings``.

    An unrecognised chip name is carried rather than dropped, so a chip FPL adds
    mid-season still shows its window instead of vanishing; it simply gets no
    valuation, and ``check_chip_definitions`` reports it.
    """
    raw = (bootstrap or {}).get("chips")
    if not isinstance(raw, list) or not raw:
        return list(DEFAULT_CHIP_WINDOWS)

    out: List[ChipWindow] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        start, stop = entry.get("start_event"), entry.get("stop_event")
        if not name or start is None or stop is None:
            continue
        try:
            start, stop = int(start), int(stop)
        except (TypeError, ValueError):
            continue
        if stop < start:
            continue
        out.append(ChipWindow(name, start, stop, str(entry.get("chip_type") or "")))

    if not out:
        _logger.warning("bootstrap chips present but none parsed; using defaults")
        return list(DEFAULT_CHIP_WINDOWS)
    return out


class ChipState(NamedTuple):
    """Everything the UI needs to say about one chip, now.

    ``playable_now`` is "you can play this in ``current_gw``", which is a
    different question from "you own one somewhere this season". Conflating them
    listed Wildcard as available at GW4 for a manager who had played it in GW3,
    because the second slot exists -- it does not open until GW20.
    """

    name: str
    playable_now: bool
    window: Optional[Tuple[int, int]]      # the window `current_gw` falls in
    used_in: Tuple[int, ...]               # every gameweek this chip was played
    next_window_opens: Optional[int]       # a later window still unspent
    expires_after: Optional[int]           # last gameweek of the current window
    gws_left: Optional[int]                # inclusive, counting current_gw


def _used_by_name(history_chips: Optional[Sequence[dict]]) -> Dict[str, List[int]]:
    """``{chip name: [gameweeks played]}`` from an entry's history.

    Every gameweek is kept, not the first. The advisor rendered ``events[0]``,
    so a manager who had played both Wildcards saw the first one's gameweek
    against a chip that was actually spent twice.
    """
    used: Dict[str, List[int]] = {}
    for chip in history_chips or []:
        if not isinstance(chip, dict):
            continue
        name = str(chip.get("name") or "").strip()
        event = chip.get("event")
        if not name or event is None:
            continue
        try:
            used.setdefault(name, []).append(int(event))
        except (TypeError, ValueError):
            continue
    for name in used:
        used[name].sort()
    return used


def chip_states(definitions: Sequence[ChipWindow],
                history_chips: Optional[Sequence[dict]],
                current_gw: int) -> Dict[str, ChipState]:
    """Per-chip availability, derived from the published windows.

    A chip's windows are matched against the gameweeks it was played in, in
    order: a use inside a window spends that window. So "playable now" is
    simply "the window containing ``current_gw`` has an unspent slot", which
    generalises the old hardcoded single-vs-double-use split and is correct
    whatever FPL publishes next season.

    An unplayed window that has already closed is **forfeited**, not carried:
    FPL does not roll a first-half chip into the second half.
    """
    current_gw = int(current_gw or 0)
    used = _used_by_name(history_chips)
    states: Dict[str, ChipState] = {}

    by_name: Dict[str, List[ChipWindow]] = {}
    for win in definitions or ():
        by_name.setdefault(win.name, []).append(win)

    for name, windows in by_name.items():
        windows = sorted(windows, key=lambda w: (w.start_event, w.stop_event))
        plays = list(used.get(name, []))

        # Spend each play against the window it falls in. A play outside every
        # window should not be possible, and is ignored rather than trusted.
        spent = set()
        for gw in plays:
            for i, win in enumerate(windows):
                if i in spent:
                    continue
                if win.start_event <= gw <= win.stop_event:
                    spent.add(i)
                    break

        here = next((i for i, w in enumerate(windows)
                     if w.start_event <= current_gw <= w.stop_event), None)
        playable = here is not None and here not in spent
        window = ((windows[here].start_event, windows[here].stop_event)
                  if here is not None else None)
        expires = windows[here].stop_event if here is not None else None
        left = (max(0, expires - current_gw + 1) if expires is not None else None)

        later = next((w.start_event for i, w in enumerate(windows)
                      if i not in spent and w.start_event > current_gw), None)

        states[name] = ChipState(
            name=name,
            playable_now=playable,
            window=window,
            used_in=tuple(plays),
            next_window_opens=later,
            expires_after=expires,
            gws_left=left,
        )
    return states


def legacy_status_dict(states: Dict[str, ChipState], current_gw: int) -> dict:
    """The dict shape ``scripts/classic/transfers.py`` has always consumed.

    Kept as an adapter rather than changing five callsites and their tests at
    once. ``used`` now holds a list for *every* chip instead of a list for the
    two that used to be double-use and a bare int for the two that did not --
    the renderer already branched on ``isinstance(..., list)``, so both shapes
    were live and only one was right.
    """
    avail = [c for c in KNOWN_CHIPS
             if c in states and states[c].playable_now]
    later = [c for c in KNOWN_CHIPS
             if c in states and not states[c].playable_now
             and states[c].next_window_opens is not None]

    slot2 = min((s.next_window_opens for s in states.values()
                 if s.next_window_opens is not None), default=None)
    if slot2 is None:
        # Nothing re-opens, so nothing will be captioned with it; the historical
        # constant is the least surprising thing to report.
        slot2 = 20

    def _used(name: str) -> List[int]:
        return list(states[name].used_in) if name in states else []

    def _in_window(name: str, which: int) -> bool:
        """Was this chip played in its first (0) or second (1) window?"""
        plays = _used(name)
        return any(p >= slot2 for p in plays) if which else any(p < slot2 for p in plays)

    out = {
        "used": {c: _used(c) for c in KNOWN_CHIPS},
        "available": avail,
        "available_later": later,
        "slot2_first_gw": slot2,
        "states": states,
    }
    for name in (CHIP_WILDCARD, CHIP_BBOOST):
        out["%s_1_used" % name] = _in_window(name, 0)
        out["%s_2_used" % name] = _in_window(name, 1)
    for name in KNOWN_CHIPS:
        out["%s_available" % name] = name in avail
    return out


def chip_status_from_bootstrap(bootstrap: Optional[dict],
                               history: Optional[dict],
                               current_gw: int) -> dict:
    """The one call a page needs: published rules plus this entry's history."""
    definitions = parse_chip_definitions(bootstrap)
    states = chip_states(definitions, (history or {}).get("chips"), current_gw)
    return legacy_status_dict(states, current_gw)


# --------------------------------------------------------------------------
# What a chip is worth, in expected points
# --------------------------------------------------------------------------

def _positions(squad_df: pd.DataFrame) -> pd.Series:
    """The squad's positions as ``G/D/M/F``.

    Normalised here rather than at the callsites because the frames reaching
    this module disagree: ``blended_pool`` emits ``G/D/M/F``, the Classic squad
    builders emit ``GKP/DEF/MID/FWD``, and a raw ``element_type`` integer turns
    up too. Mismatched codes do not raise -- they match no position group, so
    every gate silently does nothing, which is the failure that once had every
    Power Rankings team scoring exactly 50.
    """
    if "Position" not in squad_df.columns:
        raise KeyError("chips: squad frame has no 'Position' column")
    return squad_df["Position"].map(_map_position_to_rw)


def squad_points_matrix(squad_df: pd.DataFrame,
                        points_by_gw: pd.DataFrame,
                        gameweeks: Sequence[int]) -> pd.DataFrame:
    """The squad's expected points per gameweek, aligned to ``squad_df``.

    ``points_by_gw`` is the wide player x gameweek grid the Horizon tab already
    builds (``player_projections._horizon_frames``), indexed on the element id,
    whose first column is the engine's own blend rather than a re-derivation of
    it.

    **A missing value becomes 0, and that is the honest answer rather than a
    convenience.** A blank gameweek means the club is not playing and a player
    no source priced is a declared non-starter -- both score nothing. Leaving
    them NaN would make a chip's value depend on how many of your fifteen
    happen to be unpriced, and summing with ``min_count`` would report a
    *smaller* bench as a *better* Bench Boost week.
    """
    gameweeks = [int(g) for g in gameweeks]
    ids = player_id_column(squad_df)
    if ids is None:
        raise KeyError(
            "chips: squad frame carries no element id under any of %s"
            % ", ".join(("Player_ID", "element_id", "element", "player_id")))

    out = pd.DataFrame(0.0, index=squad_df.index, columns=gameweeks)
    if points_by_gw is None or points_by_gw.empty:
        return out

    grid = points_by_gw.copy()
    grid.index = pd.to_numeric(grid.index, errors="coerce")
    for gw in gameweeks:
        if gw not in grid.columns:
            continue
        col = pd.to_numeric(grid[gw], errors="coerce")
        out[gw] = ids.map(col).astype(float).fillna(0.0).values
    return out


def _xi_total(positions: pd.Series, points: pd.Series) -> Tuple[float, List]:
    """``(total of the best legal XI, its index labels)``."""
    frame = pd.DataFrame({"Position": positions, "_p": points})
    chosen = best_xi_index(frame, "_p")
    return float(points.loc[chosen].sum()), chosen


def bench_boost_value(squad_df: pd.DataFrame,
                      matrix: pd.DataFrame) -> Dict[int, float]:
    """Expected points Bench Boost adds, per gameweek.

    Bench Boost pays the four players you would otherwise have benched, so its
    value is ``sum(all 15) - best legal XI`` -- the bench is whatever the XI
    leaves over, which means this needs no view on who you would actually have
    started.
    """
    positions = _positions(squad_df)
    out: Dict[int, float] = {}
    for gw in matrix.columns:
        points = pd.to_numeric(matrix[gw], errors="coerce").fillna(0.0)
        xi, _ = _xi_total(positions, points)
        out[int(gw)] = max(0.0, float(points.sum()) - xi)
    return out


def triple_captain_value(squad_df: pd.DataFrame,
                         matrix: pd.DataFrame) -> Dict[int, float]:
    """Expected points Triple Captain adds, per gameweek.

    The armband already doubles, so tripling adds exactly *one* further copy of
    the captain's score -- the gain is the best starter's expected points, not
    twice it. A double gameweek needs no special case: the horizon grid sums a
    club's two fixtures into one cell.
    """
    positions = _positions(squad_df)
    out: Dict[int, float] = {}
    for gw in matrix.columns:
        points = pd.to_numeric(matrix[gw], errors="coerce").fillna(0.0)
        _, chosen = _xi_total(positions, points)
        out[int(gw)] = float(points.loc[chosen].max()) if chosen else 0.0
    return out


#: Default bench weight when solving a squad whose bench will not score. Free
#: Hit lasts one gameweek, so the bench is genuinely worthless -- but at exactly
#: 0 the solver is indifferent between benches and will shuffle one for no
#: modelled gain, which is the reason the transfer planner uses 0.1 rather
#: than 0.
_BENCH_WEIGHT = 0.1

#: ``solve_squad_ilp`` prunes non-positive scores, and the engine writes
#: ``Proj = 0`` for a declared non-starter -- so an owned player can be dropped
#: from his own squad, which silently makes selling him free. Owned rows are
#: exempted the way the transfer planner exempts them; for the *pool* side a
#: tiny floor keeps a legitimately zero-projection player selectable as bench
#: fodder without making him attractive.
_POOL_SCORE_FLOOR = 1e-6


def _solve_best_xi(pool: pd.DataFrame, score_col: str, budget: float,
                   solver: Optional[Callable] = None,
                   **kwargs) -> Optional[float]:
    """The starting XI total of the best squad buildable from ``pool``."""
    from scripts.common.optimization import solve_squad_ilp
    solve = solver or solve_squad_ilp
    try:
        _squad, totals = solve(pool, budget, score_col=score_col,
                               bench_weight=_BENCH_WEIGHT, **kwargs)
    except Exception as exc:                       # pragma: no cover - solver
        _logger.warning("chip squad solve failed: %s", exc)
        return None
    if not totals:
        return None
    return float(totals.get("starter_score", 0.0))


def free_hit_value(squad_df: pd.DataFrame,
                   matrix: pd.DataFrame,
                   pool: pd.DataFrame,
                   pool_points_by_gw: pd.DataFrame,
                   budget: float,
                   solver: Optional[Callable] = None) -> Dict[int, float]:
    """Expected points Free Hit adds, per gameweek.

    A Free Hit is one gameweek of any squad you can afford, so its value is
    ``best XI from the whole pool - best XI from your own fifteen`` for that
    week. It peaks exactly where the published advice says it should: a week
    your own assets have hard fixtures, or a blank where several of them are
    not playing at all.

    **The floor is a real invariant.** Your own fifteen are in the pool, so the
    best XI drawn from the pool can never be worse than the best XI drawn from
    your squad -- a negative value means the pool was over-filtered or the
    budget is wrong, which is why ``check_chip_timing`` errors on it rather
    than letting a nonsense number render as advice.
    """
    positions = _positions(squad_df)
    out: Dict[int, float] = {}

    grid = None
    if pool_points_by_gw is not None and not pool_points_by_gw.empty:
        grid = pool_points_by_gw.copy()
        grid.index = pd.to_numeric(grid.index, errors="coerce")

    pool_ids = player_id_column(pool) if pool is not None and not pool.empty else None

    for gw in matrix.columns:
        gw = int(gw)
        mine = pd.to_numeric(matrix[gw], errors="coerce").fillna(0.0)
        own_xi, _ = _xi_total(positions, mine)

        if grid is None or pool_ids is None or gw not in grid.columns:
            continue
        scored = pool.copy()
        col = pd.to_numeric(grid[gw], errors="coerce")
        scored["_chip_score"] = (pool_ids.map(col).astype(float)
                                 .fillna(0.0).clip(lower=_POOL_SCORE_FLOOR).values)
        best = _solve_best_xi(scored, "_chip_score", budget, solver=solver)
        if best is None:
            continue
        out[gw] = max(0.0, best - own_xi)
    return out


def wildcard_value(squad_df: pd.DataFrame,
                   pool: pd.DataFrame,
                   score_col: str,
                   budget: float,
                   free_transfers: int,
                   sell_price_col: Optional[str] = None,
                   solver: Optional[Callable] = None) -> Optional[float]:
    """Expected points per gameweek a Wildcard buys **beyond your transfers**.

    A Wildcard is not worth "the best squad minus my squad": you could reach
    part of that with the free transfers you already have, and charging the
    chip for value you can get without it is how a tool talks somebody into
    burning one. So this is the gain at fifteen free changes minus the gain at
    ``free_transfers`` of them, which is the marginal thing the chip actually
    provides.

    Scored on whatever ``score_col`` the caller has blended over the window --
    a per-gameweek rate, so the answer reads in the same units as every other
    number on the page.

    **Cannot be negative, and that is an invariant rather than a clamp.**
    Fifteen free changes include every subset of ``free_transfers`` of them, so
    the wider search cannot do worse; a negative result means the two solves
    saw different pools and ``check_chip_timing`` errors on it.
    """
    if pool is None or pool.empty:
        return None
    ids = player_id_column(squad_df)
    if ids is None:
        return None
    owned = [int(v) for v in ids.dropna().tolist()]
    if not owned:
        return None

    kwargs = dict(owned_ids=owned, id_col="Player_ID")
    if sell_price_col and sell_price_col in pool.columns:
        kwargs["sell_price_col"] = sell_price_col

    free_transfers = max(0, int(free_transfers or 0))
    with_chip = _solve_best_xi(pool, score_col, budget, solver=solver,
                               max_changes=SQUAD_SIZE,
                               free_transfers=SQUAD_SIZE, **kwargs)
    without = _solve_best_xi(pool, score_col, budget, solver=solver,
                             max_changes=free_transfers,
                             free_transfers=free_transfers, **kwargs)
    if with_chip is None or without is None:
        return None
    return with_chip - without


#: FPL statuses that mean a player cannot be picked: injured, suspended,
#: unavailable (which also covers a departure), and not in the squad.
_UNAVAILABLE_STATUSES = frozenset({"i", "s", "u", "n"})


def dead_weight_count(squad_df: pd.DataFrame,
                      proj_col: str = "Proj") -> int:
    """How many of the fifteen cannot contribute at all.

    The article's "your squad needs an overhaul" made countable: a player FPL
    rates unavailable, or one the engine has priced at zero because no source
    expects him to start. This is the Wildcard signal that does not depend on a
    solver, so it still says something when the pool cannot be built.
    """
    if squad_df is None or squad_df.empty:
        return 0
    dead = pd.Series(False, index=squad_df.index)
    if "status" in squad_df.columns:
        dead |= squad_df["status"].astype(str).str.strip().isin(_UNAVAILABLE_STATUSES)
    if proj_col in squad_df.columns:
        proj = pd.to_numeric(squad_df[proj_col], errors="coerce")
        dead |= proj.notna() & (proj <= 0)
    return int(dead.sum())


# --------------------------------------------------------------------------
# Who you play: the H2H lane
# --------------------------------------------------------------------------
#
# The points a chip adds and the *good* it does are different quantities, and in
# an H2H league only the second one decides anything. Raised from the app: "I
# debated using a bench boost this week, but the team I am playing did not have
# a strong projection and is lower in the standings, so I figured I could save
# it for a better week against a stronger opponent."
#
# Right in direction, and the usual framing of it is wrong. Writing `d` for your
# projected margin before the chip and `g` for the chip's points:
#
#     dP = Phi((d + g) / (sqrt(2) sigma)) - Phi(d / (sqrt(2) sigma))
#
# d(dP)/dd = 0 where phi((d+g)/s) = phi(d/s), i.e. |d+g| = |d|, i.e. **d = -g/2**:
# the chip is worth most when you are trailing by half its gain. At the measured
# Classic H2H sigma of 17.3, for a 12-point Bench Boost:
#
#     margin   +45     +10    level    -6      -25     -45
#     dP      +0.023  +0.157  +0.188  +0.194  +0.144  +0.056
#
# So a weak opponent does waste it -- eight times over at +45. But a much
# stronger one wastes it too, because you lose anyway. The rule is "spend it
# where it is just big enough to turn a narrow projected loss into a narrow
# win", and that is a *close* opponent rather than a strong one.

#: Weekly single-manager scoring spread, used when a league has no history to
#: measure one from. Shared with ``season_sim.DEFAULT_SIGMA`` for the same
#: reason: a degenerate sigma turns a one-point edge into a near-certainty.
DEFAULT_SIGMA = 17.0


def _phi(x: float) -> float:
    """Standard normal CDF. Hand-rolled because scipy is not a dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def win_probability(margin: float, sigma: float) -> float:
    """``P(you outscore them)`` given a projected margin.

    The same ``Phi((muA - muB) / sqrt(2 sigma^2))`` that has backed Fixture
    Projections and ``season_sim`` for a year -- deliberately not a second
    model, so the chip page and the match projection cannot disagree about how
    one fixture resolves.
    """
    sigma = float(sigma or 0.0)
    if sigma <= 0 or not math.isfinite(sigma):
        return float("nan")
    return _phi(float(margin) / (math.sqrt(2.0) * sigma))


def h2h_chip_value(points_gain: float, my_proj: float, opp_proj: float,
                   sigma: float = DEFAULT_SIGMA) -> Optional[float]:
    """Win probability a chip buys in one H2H fixture, or None.

    Returns the *change* in ``P(win)``, which is the quantity that decides an
    H2H league -- never the points, which is what makes a chip look equally
    good against an opponent you were beating by 40 anyway.
    """
    for val in (points_gain, my_proj, opp_proj):
        if val is None or not math.isfinite(float(val)):
            return None
    margin = float(my_proj) - float(opp_proj)
    before = win_probability(margin, sigma)
    after = win_probability(margin + float(points_gain), sigma)
    if not (math.isfinite(before) and math.isfinite(after)):
        return None
    return after - before


def ideal_h2h_margin(points_gain: float) -> float:
    """The projected margin at which a chip of this size does the most good.

    ``-g/2`` -- trailing by half the gain. Exposed so the page can say what it
    is looking for rather than only scoring what it found, and so the result is
    pinned by a test against a numeric search over the real curve.
    """
    return -float(points_gain) / 2.0


def best_week(values: Dict[int, float],
              minimum: float = 0.0) -> Optional[Tuple[int, float]]:
    """``(gameweek, value)`` of the best week, preferring the nearest on a tie.

    Ties break toward the **earlier** gameweek, which is not arbitrary:
    ``projection_accuracy.score_horizon_lengths()`` measured that a forecast
    loses about a third of its ordering by the third week out, so two weeks
    projected equal are not equally trustworthy and the nearer claim is the
    stronger one.
    """
    if not values:
        return None
    best = None
    for gw in sorted(values):
        val = values[gw]
        if val is None or not math.isfinite(float(val)) or float(val) < minimum:
            continue
        if best is None or float(val) > best[1] + 1e-9:
            best = (int(gw), float(val))
    return best
