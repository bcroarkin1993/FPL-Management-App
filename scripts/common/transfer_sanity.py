"""Veto a suggested transfer where the incoming player is plainly worse.

The scoring model is a blend of percentiles, and a blend can rank a player above
another while every raw number a manager would actually look at says the
opposite -- form, fixtures and start consistency pulling hard enough to outvote
the points. When that happens the suggestion is indefensible on its face, and a
manager who acts on it is worse off.

So this is a second opinion in a different currency: not "is the composite score
higher" but "is the incoming player at least roughly comparable on the things
you can see". It only ever *removes* suggestions.

**Which signals exist depends on the format, and getting that wrong makes the
gate a no-op that looks like protection.** Draft frames carry ``Season_Points``;
Classic frames carry ``total_points`` and never define ``Season_Points`` at all.
``compute_player_scores`` already resolves exactly this pair the same way, and
the fallback here is deliberately identical to it -- parameterising the column
per caller would let a future page pass nothing and silently disable the veto.

This lived inside ``scripts/draft/waiver_wire.py`` and guarded only Draft.
Classic Transfers had no equivalent, which is the worse place to lack one: a
Draft waiver claim is free, and a Classic transfer can cost a -4 hit.
"""

from typing import List, Optional, Tuple

import pandas as pd

__all__ = [
    "SANITY_TOLERANCE",
    "sanity_check_signals",
    "sanity_check_suggestion",
    "is_seriously_injured",
]

#: How much worse the incoming player may be on a signal and still pass it.
#: Loose on purpose: the composite score legitimately knows things the raw
#: numbers do not (fixture run, start security, form trajectory), so this is a
#: "that cannot be right" boundary, not a "that looks unusual" one. A veto that
#: cries wolf gets switched off.
SANITY_TOLERANCE = 0.80

#: Season-points column, most specific first. Draft frames define the first,
#: Classic frames the second; this mirrors ``compute_player_scores``.
_SEASON_COLS = ("Season_Points", "total_points")

#: Expected-points column, most specific first. ``compute_player_scores`` sets
#: ``_effective_proj`` from the engine's ``Proj`` on every frame it touches;
#: ``Projected_Points`` is Rotowire's raw "if he starts" number, which some
#: Classic frames still carry alone, and is the last resort.
#:
#: **The two sides must resolve to the same column.** These are different bases
#: -- ``_effective_proj`` has start likelihood applied and ``Projected_Points``
#: does not -- so comparing one against the other would charge a rotation risk
#: to exactly one player. That is the basis confusion the projection engine was
#: built to end; it is not being reintroduced in the veto.
_PROJ_COLS = ("_effective_proj", "Proj", "Projected_Points")


def _number(value, default: float = 0.0) -> float:
    """Numeric value, or ``default`` for anything unusable."""
    number = pd.to_numeric(value, errors="coerce")
    try:
        if pd.isna(number):
            return default
    except (TypeError, ValueError):
        return default
    return float(number)


def _first_present(row, columns) -> Tuple[Optional[str], Optional[float]]:
    """``(column, value)`` for the first of ``columns`` this row carries.

    ``(None, None)`` means *no information* -- the column is absent, or present
    and NaN -- which is different from 0.0, and the difference decides whether a
    signal is skipped or counted against the incoming player.
    """
    for column in columns:
        if column not in _row_keys(row):
            continue
        number = pd.to_numeric(row.get(column), errors="coerce")
        if pd.notna(number):
            return column, float(number)
    return None, None


def _same_basis(drop_row, add_row, columns) -> Tuple[Optional[float], Optional[float]]:
    """Both sides' values, but only when they came from the *same* column."""
    drop_col, drop_val = _first_present(drop_row, columns)
    add_col, add_val = _first_present(add_row, columns)
    if drop_col is None or add_col is None or drop_col != add_col:
        return None, None
    return drop_val, add_val


def _row_keys(row):
    index = getattr(row, "index", None)
    if index is not None and not callable(index):
        return index
    return row.keys() if hasattr(row, "keys") else ()


def is_seriously_injured(row) -> bool:
    """Is this player unable to play, rather than merely doubtful?

    Replacing someone who cannot play is always defensible, so this lifts the
    veto entirely. Without it the gate blocks exactly the transfers a manager
    most needs -- the incoming player is often *statistically* worse than an
    injured star, and that is beside the point.
    """
    status = str(row.get("status", "") or "")
    if status in ("i", "s", "u"):
        return True
    chance = pd.to_numeric(row.get("chance_of_playing_next_round"), errors="coerce")
    return bool(pd.notna(chance) and float(chance) < 50)


def sanity_check_signals(drop_row, add_row,
                         tolerance: float = SANITY_TOLERANCE
                         ) -> List[Tuple[str, bool]]:
    """The individual raw-metric comparisons, as ``[(name, add_is_ok), ...]``.

    An empty list means nothing comparable was available -- which callers must
    treat as "cannot judge", never as "passed". Exposed separately from
    :func:`sanity_check_suggestion` so tests and the debug panel can see *which*
    signals fired rather than only the verdict.
    """
    checks: List[Tuple[str, bool]] = []

    # 1. Expected points this gameweek (projection x start likelihood).
    #
    # None (absent/NaN) and 0.0 mean different things and must not be merged: a
    # blank gameweek or a data gap carries no information, while a real 0 means
    # "not expected to start", which is a genuine signal in both directions.
    drop_proj, add_proj = _same_basis(drop_row, add_row, _PROJ_COLS)
    if drop_proj is not None and add_proj is not None:
        if drop_proj > 0 and add_proj > 0:
            checks.append(("proj_pts", add_proj >= drop_proj * tolerance))
        elif drop_proj == 0 and add_proj > 0:
            checks.append(("proj_pts", True))    # drop is not starting, add is
        elif drop_proj > 0 and add_proj == 0:
            checks.append(("proj_pts", False))   # add is not starting, drop is
        # both zero: neither is expected to start, so this says nothing.

    # 2. Points actually accumulated this season.
    _drop_season, _add_season = _same_basis(drop_row, add_row, _SEASON_COLS)
    drop_season, add_season = _number(_drop_season), _number(_add_season)
    if drop_season > 0 and add_season > 0:
        checks.append(("season_pts", add_season >= drop_season * tolerance))

    # 3. The three-gameweek window, which is where a fixture run shows up.
    drop_multi = _number(drop_row.get("MultiGW_Proj"))
    add_multi = _number(add_row.get("MultiGW_Proj"))
    if drop_multi > 0 and add_multi > 0:
        checks.append(("3gw_proj", add_multi >= drop_multi * tolerance))

    return checks


def sanity_check_suggestion(drop_row, add_row,
                            tolerance: float = SANITY_TOLERANCE
                            ) -> Tuple[bool, str]:
    """``(passes, reason)`` for one proposed ``drop -> add`` swap.

    Passing on "no data" is deliberate: a veto that fires because a column was
    missing would suppress every suggestion on a degraded feed, which is a far
    worse failure than letting a questionable one through. Callers that want to
    know the gate is actually working should count the empty-signal cases --
    see :func:`sanity_check_signals`.
    """
    if is_seriously_injured(drop_row):
        return True, "injury override"

    checks = sanity_check_signals(drop_row, add_row, tolerance)
    if not checks:
        return True, "no data"

    passed = sum(1 for _, ok in checks if ok)
    if passed >= (len(checks) + 1) // 2:      # a majority of what we can see
        return True, "ok"
    failed = [name for name, ok in checks if not ok]
    return False, "ADD worse on: %s" % ", ".join(failed)
