"""
Score archived projections against what actually happened.

The 60/40 Rotowire/FFP split has always been an assumption. This module is what
turns it into a measurement: it reads the pre/actual pairs written by
:mod:`scripts.common.projection_archive` and reports, per source, how wrong each
one was.

Pure and Streamlit-free, and dependency-free beyond pandas -- Spearman is
computed from ranks rather than via scipy, which is not installed and which the
Actions workflow would then have to install.

**Two scorings, because they answer different questions.**

``all``       ``proj`` (expected points) against actual points, over every
              player the source priced. The honest end-to-end number: it
              includes players who did not play and scored zero, which is a
              projection error like any other.
``starters``  ``proj_start`` (points if he starts) against actual points,
              restricted to players who *actually started*. This isolates the
              points model from the minutes model. A source can be excellent at
              predicting what a player scores when he plays and bad at
              predicting whether he plays, and one number cannot tell you which
              you are looking at.

**Coverage is not comparable, so a fair comparison needs a common subset.**
Measured on GW3: Rotowire priced 220 players, FFP 543, FPL's ep_this 368.
Rotowire lists only expected starters -- higher-scoring and higher-variance
players -- so scoring each source over its own coverage ranks Rotowire worst for
reasons that have nothing to do with accuracy. ``common_subset=True`` restricts
every source to the players all of them priced, which is the only way the MAEs
mean the same thing. Both views are reported, because coverage is itself a real
property of a source and hiding it would be its own distortion.

**A backfill is never fitted on.** ``captured_before_deadline`` marks a snapshot
taken after the fact, which can see team news -- in the limit, lineups -- that no
manager had. Those are shown, clearly labelled, but excluded from any weight
fitting, because a flattered source would move the weights the app runs on.
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.common import projection_archive

_logger = logging.getLogger("fpl_app.projections")

#: The blended projection, and each source that fed it.
SOURCE_COLUMNS = {
    "blend": ("proj", "proj_start"),
    "rotowire": (None, "proj_start__rotowire"),
    "ffp": (None, "proj_start__ffp"),
    "fpl_ep": (None, "proj_start__fpl_ep"),
}

SCOPE_ALL = "all"
SCOPE_STARTERS = "starters"

#: Below this many scored gameweeks, differences between sources are noise.
#: Reported anyway, but never presented as a conclusion.
MIN_GAMEWEEKS_FOR_CONFIDENCE = 5


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Rank correlation, without scipy.

    Spearman is Pearson on the ranks. ``Series.corr(method="spearman")`` imports
    scipy, which is not a dependency of this project and which the scheduled
    workflow would then have to install on every run.
    """
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() < 3:
        return float("nan")
    ra, rb = a[mask].rank(), b[mask].rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return float("nan")
    return float(ra.corr(rb))


def _metrics(pred: pd.Series, actual: pd.Series) -> dict:
    """MAE, RMSE, bias and rank correlation for one prediction column."""
    pred = pd.to_numeric(pred, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")
    mask = pred.notna() & actual.notna()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
                "spearman": np.nan, "mean_proj": np.nan, "mean_actual": np.nan}
    err = pred[mask] - actual[mask]
    return {
        "n": n,
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        # Signed, on purpose: a source that is consistently 0.4 points high is a
        # different problem from one that is noisy, and MAE cannot tell them apart.
        "bias": float(err.mean()),
        "spearman": spearman(pred[mask], actual[mask]),
        "mean_proj": float(pred[mask].mean()),
        "mean_actual": float(actual[mask].mean()),
    }


def _pair(gameweek: int):
    """The joined pre/actual frame for one gameweek, plus its meta."""
    pre, pmeta = projection_archive.load_pre(gameweek)
    act, _ = projection_archive.load_actuals(gameweek)
    if pre is None or act is None:
        return None, {}
    joined = pre.merge(act, on="player_id", how="inner", suffixes=("", "_actual"))
    return joined, pmeta


def score_gameweek(gameweek: int, common_subset: bool = False,
                   sources: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Score every source for one gameweek. Empty frame if the pair is missing.

    Args:
        gameweek: which gameweek to score.
        common_subset: restrict every source to the players all of them priced,
            so the numbers are comparable. Without it, a source that prices only
            likely starters is scored on a harder population than one that
            prices everybody.
        sources: which sources to score. Defaults to all of them.
    """
    joined, meta = _pair(gameweek)
    if joined is None or joined.empty:
        return pd.DataFrame()

    names = list(sources) if sources else [
        s for s in SOURCE_COLUMNS if _column_for(s, SCOPE_STARTERS) in joined.columns
    ]

    subset = joined
    if common_subset:
        mask = pd.Series(True, index=joined.index)
        for name in names:
            col = _column_for(name, SCOPE_STARTERS)
            if col in joined.columns:
                mask &= pd.to_numeric(joined[col], errors="coerce").notna()
        subset = joined[mask]

    pool = len(joined)
    rows = []
    for name in names:
        for scope in (SCOPE_ALL, SCOPE_STARTERS):
            col = _column_for(name, scope)
            if col is None or col not in subset.columns:
                continue
            if scope == SCOPE_STARTERS:
                # Scoring an if-he-starts projection against a player who never
                # started measures the minutes model, not the points model.
                frame = subset[pd.to_numeric(subset["started"], errors="coerce") > 0]
            else:
                # The `all` scope is deliberately *not* restricted to the common
                # subset. Its whole point is the full population, including the
                # players who never played and scored zero -- restricting it to
                # players every source priced would quietly drop exactly those
                # and turn the honest end-to-end number into a starters-only one.
                # Only the blend has an expected-value column anyway, so there is
                # no cross-source comparison here to make fair.
                frame = joined
            m = _metrics(frame[col], frame["points"])
            m.update({
                "gameweek": int(gameweek),
                "source": name,
                "scope": scope,
                "coverage": (pd.to_numeric(joined[col], errors="coerce").notna().sum() / pool
                             if pool else np.nan),
                "capture": ("pre-deadline" if meta.get("captured_before_deadline")
                            else "backfill"),
            })
            rows.append(m)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    lead = ["gameweek", "source", "scope", "capture", "n", "coverage",
            "mae", "rmse", "bias", "spearman", "mean_proj", "mean_actual"]
    return out[[c for c in lead if c in out.columns]]


def _column_for(source: str, scope: str) -> Optional[str]:
    """Which archived column holds ``source``'s projection on ``scope``'s basis.

    Only the blend stores an expected-value column: the per-source columns are
    all on the conditional basis, because that is what the engine blends. So a
    source other than the blend has nothing to report on the ``all`` scope, and
    reporting its conditional value there would compare an if-he-starts number
    against players who did not start.
    """
    ev_col, cond_col = SOURCE_COLUMNS.get(source, (None, None))
    return ev_col if scope == SCOPE_ALL else cond_col


def score_archive(gameweeks: Optional[Sequence[int]] = None,
                  common_subset: bool = False,
                  include_backfills: bool = True) -> pd.DataFrame:
    """Score every gameweek holding both halves of the pair.

    Backfills are included by default but carry ``capture == "backfill"`` so a
    reader can see them for what they are. They are excluded from weight fitting
    elsewhere, which is where a flattered number would actually change the app.
    """
    gws = list(gameweeks) if gameweeks else projection_archive.scoreable_gameweeks()
    frames = []
    for gw in gws:
        one = score_gameweek(gw, common_subset=common_subset)
        if one.empty:
            continue
        if not include_backfills and (one["capture"] == "backfill").all():
            continue
        frames.append(one)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarise(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-gameweek scores into one row per source and scope.

    MAE and RMSE are weighted by ``n``: a gameweek where a source priced 40
    players should not count as much as one where it priced 400.
    """
    if scored is None or scored.empty:
        return pd.DataFrame()

    rows = []
    for (source, scope), grp in scored.groupby(["source", "scope"], sort=False):
        n = grp["n"].sum()
        if not n:
            continue
        w = grp["n"]
        rows.append({
            "source": source,
            "scope": scope,
            "gameweeks": int(grp["gameweek"].nunique()),
            "n": int(n),
            "mae": float((grp["mae"] * w).sum() / n),
            "rmse": float((grp["rmse"] * w).sum() / n),
            "bias": float((grp["bias"] * w).sum() / n),
            "spearman": float(grp["spearman"].mean(skipna=True)),
            "coverage": float(grp["coverage"].mean(skipna=True)),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "mae"]).reset_index(drop=True) if not out.empty else out


def fit_blend_weights(gameweeks: Optional[Sequence[int]] = None,
                      sources: Sequence[str] = ("rotowire", "ffp", "fpl_ep"),
                      step: float = 0.05) -> dict:
    """Grid-search the source weights that would have minimised MAE.

    Fitted on the **common subset** and on **pre-deadline captures only**: a
    backfill can see team news no manager had, and letting one move the weights
    the app runs on is exactly the way a measurement harness makes things worse.

    Returns the best weights, the MAE they achieve, the MAE of the current
    configured weights for comparison, and the sample size. The caller decides
    whether the sample justifies acting on it -- ``MIN_GAMEWEEKS_FOR_CONFIDENCE``
    is the threshold below which the difference between two sources is noise.
    """
    gws = list(gameweeks) if gameweeks else projection_archive.scoreable_gameweeks()
    frames = []
    for gw in gws:
        joined, meta = _pair(gw)
        if joined is None or joined.empty:
            continue
        if not meta.get("captured_before_deadline"):
            _logger.info("GW%s excluded from weight fitting: backfill", gw)
            continue
        cols = {s: _column_for(s, SCOPE_STARTERS) for s in sources}
        if any(c not in joined.columns for c in cols.values()):
            continue
        keep = joined[["player_id", "points", "started"] + list(cols.values())].copy()
        for c in cols.values():
            keep[c] = pd.to_numeric(keep[c], errors="coerce")
        keep = keep.dropna(subset=list(cols.values()))
        keep = keep[pd.to_numeric(keep["started"], errors="coerce") > 0]
        if not keep.empty:
            keep["gameweek"] = gw
            frames.append(keep)

    if not frames:
        return {"fitted": None, "mae": np.nan, "current_mae": np.nan,
                "n": 0, "gameweeks": 0,
                "note": "No pre-deadline gameweek has both projections and actuals yet."}

    data = pd.concat(frames, ignore_index=True)
    actual = data["points"]
    preds = {s: data[_column_for(s, SCOPE_STARTERS)] for s in sources}

    best, best_mae = None, np.inf
    for weights in _simplex(len(sources), step):
        blended = sum(w * preds[s] for w, s in zip(weights, sources))
        mae = float((blended - actual).abs().mean())
        if mae < best_mae:
            best, best_mae = weights, mae

    current = _current_weights(sources)
    cur_blend = sum(w * preds[s] for w, s in zip(current, sources))
    current_mae = float((cur_blend - actual).abs().mean())

    return {
        "fitted": {s: round(w, 3) for s, w in zip(sources, best)},
        "mae": best_mae,
        "current": {s: round(w, 3) for s, w in zip(sources, current)},
        "current_mae": current_mae,
        "n": int(len(data)),
        "gameweeks": int(data["gameweek"].nunique()),
        "note": "",
    }


def _simplex(k: int, step: float) -> List[tuple]:
    """Every weight vector of length ``k`` on a ``step`` grid summing to 1."""
    steps = int(round(1.0 / step))

    def _rec(remaining, slots):
        if slots == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in _rec(remaining - i, slots - 1):
                yield (i,) + rest

    return [tuple(i / steps for i in combo) for combo in _rec(steps, k)]


def _current_weights(sources: Sequence[str]) -> List[float]:
    """The configured weights, renormalised over ``sources``.

    Renormalised because a source weighted 0 (``fpl_ep`` today) contributes
    nothing, and comparing a fitted vector against an unnormalised one would
    make the current configuration look worse than it is.
    """
    try:
        import config
        configured = getattr(config, "PROJECTION_SOURCE_WEIGHTS", {}) or {}
    except Exception:                       # pragma: no cover - config optional
        configured = {}
    raw = [float(configured.get(s, 0.0)) for s in sources]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(sources)] * len(sources)
    return [w / total for w in raw]


def confidence_note(scored: pd.DataFrame) -> str:
    """A plain sentence about whether the sample supports any conclusion."""
    if scored is None or scored.empty:
        return ("No gameweek yet has both a projection snapshot and actual "
                "points. Accuracy can be measured from the first completed "
                "gameweek after snapshots began.")
    gws = int(scored["gameweek"].nunique())
    if gws < MIN_GAMEWEEKS_FOR_CONFIDENCE:
        return ("%d gameweek%s of history. Differences between sources this "
                "small are noise -- treat the table as a sanity check, not a "
                "verdict. At least %d gameweeks are needed before the ordering "
                "means anything."
                % (gws, "" if gws == 1 else "s", MIN_GAMEWEEKS_FOR_CONFIDENCE))
    return "%d gameweeks of history." % gws
