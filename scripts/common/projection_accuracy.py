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

from scripts.common import projection_archive, projection_engine

_logger = logging.getLogger("fpl_app.projections")

#: The blended projection, and each source that fed it.
SOURCE_COLUMNS = {
    "blend": ("proj", "proj_start"),
    "rotowire": (None, "proj_start__rotowire"),
    "ffp": (None, "proj_start__ffp"),
    "fpl_ep": (None, "proj_start__fpl_ep"),
}

#: Where each source's start probability is archived. ``start_pct__rotowire``
#: records a *decision* rather than a published number -- the implied value where
#: Rotowire omitted the player, the positional floor where it listed him -- and it
#: has only been written since the engine started recording it, so it is absent
#: from the GW3 and GW4 snapshots. Coverage is reported per source, which is where
#: that shows up.
START_COLUMNS = {
    "blend": "start_pct",
    "rotowire": "start_pct__rotowire",
    "ffp": "start_pct__ffp",
    "fpl_ep": "start_pct__fpl_ep",
}

SCOPE_ALL = "all"
SCOPE_STARTERS = "starters"
SCOPE_START = "start"

#: Below this many scored gameweeks, differences between sources are noise.
#: Reported anyway, but never presented as a conclusion.
MIN_GAMEWEEKS_FOR_CONFIDENCE = 5

#: Fractional MAE band inside which two weightings are not distinguishable on
#: this sample. Mirrors `FIT_TIE_BAND` for the start constants and exists for
#: the same reason -- the live GW4-8 surface is monotone with its minimum on the
#: grid edge, and 0.00 through 0.25 of Rotowire all sit within 0.5% of the best.
WEIGHT_FIT_TIE_BAND = 0.005

#: Below this many rows, one *cohort* of the start model is not worth retuning --
#: which is a different question from how many gameweeks exist. The cohorts differ
#: in size by an order of magnitude within a single gameweek (measured over GW4 and
#: GW5: 397 omitted midfielders against 39 listed forwards), so a gameweek count
#: cannot say whether any particular constant is supported. A binary outcome over
#: several hundred rows is worth acting on; the same two gameweeks say nothing at
#: all about goalkeepers.
MIN_ROWS_PER_START_COHORT = 150

#: Shared rows two sources need before a head-to-head between them decides
#: anything. Coverage varies enormously -- Rotowire prices a third of the pool --
#: so some pairs barely overlap, and a narrow interval over 40 rows of an odd
#: population is not a verdict.
MIN_HEAD_TO_HEAD_ROWS = 100

#: Buckets for the calibration table. Deliberately uneven: the ends are where the
#: app's decisions are made and where the failures have been (208 players rendered
#: at exactly 100% is a top-bucket failure), so the ends are narrow.
START_BINS = (-0.001, 0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95, 1.001)


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


def auc(pred: pd.Series, actual: pd.Series) -> float:
    """Area under the ROC curve, without scipy.

    The rank form of the Mann-Whitney statistic: the probability that a randomly
    chosen starter was rated above a randomly chosen non-starter. Reported beside
    Brier because the two measure different failures -- a source can order players
    perfectly and still state every probability far too high, and a source can be
    beautifully calibrated in aggregate while ordering at random. Ties are handled
    by average ranks, so a source that says 1.0 for everybody scores 0.5 rather
    than looking good.
    """
    pred = pd.to_numeric(pred, errors="coerce")
    actual = pd.to_numeric(actual, errors="coerce")
    mask = pred.notna() & actual.notna()
    p, y = pred[mask], (actual[mask] > 0).astype(int)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = p.rank()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _as_probability(values: pd.Series) -> pd.Series:
    """Start probabilities as 0-1, whichever scale they were stored on.

    The engine works in 0-1 and the archive stores that, but FFP publishes 0-100
    and more than one bug in this app has been a percentage used as a fraction.
    Cheap to be indifferent.
    """
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().any() and float(v.max()) > 1.5:
        v = v / 100.0
    return v.clip(0.0, 1.0)


def _binary_metrics(pred: pd.Series, actual: pd.Series) -> dict:
    """Brier, log loss, AUC and signed bias for one start-probability column.

    Separate from :func:`_metrics` because the outcome is binary: MAE on a 0/1
    outcome is a worse Brier score and rank correlation against a two-valued
    series is not informative. Bias keeps its meaning exactly -- a model that
    says 40% where players start 20% of the time reads +0.20 -- and it is the
    number that actually moves the engine's constants, because a probability
    can be well-ordered and still systematically too high.
    """
    pred = _as_probability(pred)
    actual = pd.to_numeric(actual, errors="coerce")
    mask = pred.notna() & actual.notna()
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "brier": np.nan, "logloss": np.nan, "auc": np.nan,
                "bias": np.nan, "mean_proj": np.nan, "mean_actual": np.nan}
    p, y = pred[mask], (actual[mask] > 0).astype(float)
    # Clipped because a confident miss is otherwise infinite, and one row would
    # then decide the whole comparison.
    clipped = p.clip(1e-6, 1 - 1e-6)
    return {
        "n": n,
        "brier": float(((p - y) ** 2).mean()),
        "logloss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
        "auc": auc(p, y),
        "bias": float((p - y).mean()),
        "mean_proj": float(p.mean()),
        "mean_actual": float(y.mean()),
    }


def _started(frame: pd.DataFrame) -> pd.Series:
    """The binary outcome: did he start."""
    return (pd.to_numeric(frame["started"], errors="coerce") > 0).astype(float)


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
        s for s in SOURCE_COLUMNS
        if (_column_for(s, SCOPE_STARTERS) in joined.columns
            or _column_for(s, SCOPE_START) in joined.columns)
    ]

    # Two subsets, because the two questions have different coverage. Rotowire
    # prices 220 players but the engine records a start decision for all of them,
    # so a single mask over the points columns would throw away most of the start
    # evidence -- and folding the start columns into the points mask would quietly
    # change the points numbers this tab has been reporting.
    subset = _common_subset(joined, names, SCOPE_STARTERS) if common_subset else joined
    start_subset = _common_subset(joined, names, SCOPE_START) if common_subset else joined

    pool = len(joined)
    rows = []
    for name in names:
        for scope in (SCOPE_ALL, SCOPE_STARTERS, SCOPE_START):
            col = _column_for(name, scope)
            if col is None or col not in joined.columns:
                continue
            if scope == SCOPE_STARTERS:
                # Scoring an if-he-starts projection against a player who never
                # started measures the minutes model, not the points model.
                frame = subset[pd.to_numeric(subset["started"], errors="coerce") > 0]
                m = _metrics(frame[col], frame["points"])
            elif scope == SCOPE_START:
                # The minutes model on its own, over every player: the whole
                # question is who plays, so restricting to players who did would
                # be scoring it on the answer.
                frame = start_subset
                m = _binary_metrics(frame[col], _started(frame))
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
            "mae", "rmse", "bias", "spearman", "brier", "logloss", "auc",
            "mean_proj", "mean_actual"]
    return out[[c for c in lead if c in out.columns]]


def _common_subset(joined: pd.DataFrame, names: Sequence[str], scope: str) -> pd.DataFrame:
    """The rows every named source priced, on ``scope``'s column."""
    mask = pd.Series(True, index=joined.index)
    for name in names:
        col = _column_for(name, scope)
        if col and col in joined.columns:
            mask &= pd.to_numeric(joined[col], errors="coerce").notna()
    return joined[mask]


def _column_for(source: str, scope: str) -> Optional[str]:
    """Which archived column holds ``source``'s number for ``scope``.

    The ``start`` scope reads a start probability rather than a projection, so it
    has its own map. Only the blend stores an expected-value column: the per-source columns are
    all on the conditional basis, because that is what the engine blends. So a
    source other than the blend has nothing to report on the ``all`` scope, and
    reporting its conditional value there would compare an if-he-starts number
    against players who did not start.
    """
    if scope == SCOPE_START:
        return START_COLUMNS.get(source)
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
        row = {
            "source": source,
            "scope": scope,
            "gameweeks": int(grp["gameweek"].nunique()),
            "n": int(n),
            "coverage": float(grp["coverage"].mean(skipna=True)),
        }
        # NaN-skipping, because a metric belongs to one scope: the points scopes
        # carry no Brier and the start scope carries no MAE, and they share a
        # frame so that the tab, the backfill labelling and the per-gameweek
        # expander all work for both without a second pipeline.
        #
        # Row-wise metrics are `n`-weighted, which makes the aggregate exactly
        # the metric over the pooled rows. Rank statistics are not: Spearman and
        # AUC are computed over *pairs*, so there is no pooled quantity for a
        # weighted mean to approximate, and they keep the plain average over
        # gameweeks they have always had.
        for metric in ("mae", "rmse", "bias", "brier", "logloss"):
            row[metric] = _wmean(grp.get(metric), grp["n"])
        for metric in ("spearman", "auc"):
            col = grp.get(metric)
            row[metric] = (float(pd.to_numeric(col, errors="coerce").mean(skipna=True))
                           if col is not None else float("nan"))
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Sorted on whichever error metric the scope actually has.
    out["_key"] = out["mae"].fillna(out["brier"])
    return (out.sort_values(["scope", "_key"])
               .drop(columns="_key")
               .reset_index(drop=True))


def _wmean(values, weights) -> float:
    """Weighted mean over the rows where the metric exists. NaN if none do.

    Weighted by ``n`` for the same reason the unweighted version was wrong: a
    gameweek where a source priced 40 players should not count as much as one
    where it priced 400.
    """
    if values is None:
        return float("nan")
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def start_frame(gameweeks: Optional[Sequence[int]] = None,
                include_backfills: bool = True) -> pd.DataFrame:
    """Every archived player-gameweek carrying a start prediction and an outcome.

    One long frame rather than a per-gameweek loop at each callsite, because the
    calibration table, the cohort table and the constant fitter all want the same
    pooled rows and must not disagree about which ones they are.
    """
    gws = list(gameweeks) if gameweeks else projection_archive.scoreable_gameweeks()
    frames = []
    for gw in gws:
        joined, meta = _pair(gw)
        if joined is None or joined.empty or "started" not in joined.columns:
            continue
        before = bool(meta.get("captured_before_deadline"))
        if not include_backfills and not before:
            continue
        one = joined.copy()
        one["gameweek"] = int(gw)
        one["capture"] = "pre-deadline" if before else "backfill"
        # Which engine wrote it. A snapshot predating the current start-model
        # revision cannot be replayed, and saying *that* is far more use than
        # reporting the deviation it produces.
        one["start_model_revision"] = meta.get("start_model_revision")
        one["y"] = _started(one)
        # The engine's own test for "this source priced him", so the cohorts here
        # are the cohorts the constants are applied to and not an approximation
        # of them.
        one["rotowire_listed"] = (
            pd.to_numeric(one.get("proj_start__rotowire"), errors="coerce")
            .gt(0).fillna(False) if "proj_start__rotowire" in one.columns
            else pd.Series(False, index=one.index))
        frames.append(one)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def start_calibration(source: str = "blend",
                      gameweeks: Optional[Sequence[int]] = None,
                      include_backfills: bool = True,
                      by: Optional[str] = None) -> pd.DataFrame:
    """Predicted start probability against the observed rate, in buckets.

    The table a single Brier score cannot show you. A model can be well ordered
    and still state 100% for a cohort that starts a quarter of the time -- which
    is exactly what happened in GW3, where 208 players were rendered at 100%
    because a fallback, not a source, had supplied the number. Every individual
    value was plausible; only the bucket was absurd.
    """
    df = start_frame(gameweeks, include_backfills=include_backfills)
    col = START_COLUMNS.get(source)
    if df.empty or not col or col not in df.columns:
        return pd.DataFrame()
    df = df[pd.to_numeric(df[col], errors="coerce").notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["pred"] = _as_probability(df[col])
    df["bucket"] = pd.cut(df["pred"], list(START_BINS))

    keys = ["bucket"] + ([by] if by and by in df.columns else [])
    out = (df.groupby(keys, observed=True)
             .agg(n=("pred", "size"), predicted=("pred", "mean"), actual=("y", "mean"))
             .reset_index())
    out["n"] = out["n"].astype(int)
    out["gap"] = out["predicted"] - out["actual"]
    return out


def start_cohort_rates(gameweeks: Optional[Sequence[int]] = None,
                       include_backfills: bool = False) -> pd.DataFrame:
    """Predicted against observed start rate, per position and Rotowire cohort.

    These eight cells are the ones the engine's two tunable constants act on:
    ``ROTOWIRE_START_FLOORS`` clips the listed cohort up, ``ROTOWIRE_OMITTED_START``
    pulls the omitted cohort down. Reported here with the configured value beside
    the measurement, because the point of the table is to be able to see whether
    the constant is defensible -- and with ``n``, because the cells differ in size
    by an order of magnitude and two of them are far too thin to act on.

    Backfills are excluded by default, unlike everywhere else on this page: a
    snapshot taken after kickoff can have seen the actual team sheet, and a start
    model scored against news it already had is not measuring anything.
    """
    df = start_frame(gameweeks, include_backfills=include_backfills)
    if df.empty or "position" not in df.columns:
        return pd.DataFrame()
    df = df[pd.to_numeric(df[START_COLUMNS["blend"]], errors="coerce").notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["pred"] = _as_probability(df[START_COLUMNS["blend"]])

    floors, omitted = _configured_start_constants()
    rows = []
    for (pos, listed), grp in df.groupby(["position", "rotowire_listed"], observed=True):
        rows.append({
            "position": pos,
            "cohort": "listed" if listed else "omitted",
            "n": int(len(grp)),
            "predicted": float(grp["pred"].mean()),
            "actual": float(grp["y"].mean()),
            "gap": float(grp["pred"].mean() - grp["y"].mean()),
            "constant": (floors if listed else omitted).get(pos, float("nan")),
            "enough_rows": bool(len(grp) >= MIN_ROWS_PER_START_COHORT),
        })
    out = pd.DataFrame(rows)
    return (out.sort_values(["cohort", "position"]).reset_index(drop=True)
            if not out.empty else out)


def _configured_start_constants():
    """``(ROTOWIRE_START_FLOORS, ROTOWIRE_OMITTED_START)`` as configured."""
    try:
        import config
        return (dict(getattr(config, "ROTOWIRE_START_FLOORS", {}) or {}),
                dict(getattr(config, "ROTOWIRE_OMITTED_START", {}) or {}))
    except Exception:                       # pragma: no cover - config optional
        return {}, {}


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

    scored = []
    for weights in _simplex(len(sources), step):
        blended = sum(w * preds[s] for w, s in zip(weights, sources))
        scored.append((float((blended - actual).abs().mean()), tuple(weights)))
    scored.sort(key=lambda row: row[0])
    best_mae, best = scored[0]

    current = _current_weights(sources)
    cur_blend = sum(w * preds[s] for w, s in zip(current, sources))
    current_mae = float((cur_blend - actual).abs().mean())

    # --- The two guards, because an argmin always names something ------------
    #
    # Exactly the pair `fit_start_constants` carries, and for the same reason:
    # this search had neither, so it reported a single winner off a surface it
    # could not actually distinguish. On the live GW4-8 fit the MAE is monotone
    # in Rotowire's weight with no interior optimum, the minimum sits on the
    # grid *edge* at 0.00, and every weight from 0.00 to 0.25 is inside the
    # band -- so "the best fit is 0.00" was true and almost meaningless.
    cutoff = best_mae * (1.0 + WEIGHT_FIT_TIE_BAND)
    band = [w for mae, w in scored if mae <= cutoff]
    configured_in_band = tuple(round(w, 10) for w in current) in {
        tuple(round(x, 10) for x in w) for w in band
    }
    at_grid_edge = any(w <= 0.0 or w >= 1.0 for w in best)

    recommended = current if configured_in_band else _nearest_in_band(band, current)

    return {
        "fitted": {s: round(w, 3) for s, w in zip(sources, best)},
        "mae": best_mae,
        "current": {s: round(w, 3) for s, w in zip(sources, current)},
        "current_mae": current_mae,
        "recommended": {s: round(w, 3) for s, w in zip(sources, recommended)},
        "tie_band": [{s: round(w, 3) for s, w in zip(sources, cand)} for cand in band],
        "configured_in_band": bool(configured_in_band),
        "at_grid_edge": bool(at_grid_edge),
        "n": int(len(data)),
        "gameweeks": int(data["gameweek"].nunique()),
        "note": "",
    }


def _nearest_in_band(band, current):
    """The tie-band member that changes the configured weights least.

    **Which value to adopt is not the argmin.** Everything in the band is
    indistinguishable on this sample, so the honest move is the smallest one
    that leaves the band -- which also avoids adopting a grid-edge value, the
    thing `fit_start_constants` refuses outright for the positional floors.

    Ties on distance prefer the candidate that introduces no new source: giving
    weight to a source the config holds at zero is a bigger claim than
    reweighting two that already carry some, and on the live fit the two nearest
    members differ by exactly that (`fpl_ep` 0.00 against 0.05, which buys
    0.08% of MAE and is itself inside the band).
    """
    def _key(cand):
        l1 = sum(abs(c - w) for c, w in zip(cand, current))
        new_sources = sum(1 for c, w in zip(cand, current) if w <= 0.0 and c > 0.0)
        return (round(l1, 10), new_sources, tuple(-c for c in cand))

    return min(band, key=_key)


# --- Which source is actually best -------------------------------------------
#
# The tables above report each source's error. They do not answer the question a
# reader has, which is "so which one should I believe", and four rows of numbers
# to three decimal places is not an answer -- especially when the sources are not
# scored on the same players.
#
# **Sources are compared pairwise, on the players both of them priced.** Coverage
# runs from 33% (Rotowire, which lists only expected starters) to 100%, so a
# single common subset across all four collapses to whatever the sparsest source
# published: for the start question that is 143 rows of exactly the doubtful
# players FPL bothered to rate, which is no basis for a verdict about anybody.
# Pairwise keeps every comparison honest and each one as large as it can be.
#
# **The comparison is paired, which is why it can conclude anything at all from
# two gameweeks.** Comparing two aggregate MAEs throws away the fact that both
# sources were scored on the same players: a gameweek where everybody blanked
# moves both numbers together. Differencing per player removes that entirely, and
# the standard error of the *difference* is several times smaller than the
# standard error of either mean.
#
# What it cannot remove is week-to-week variation -- the interval is across
# players, not across gameweeks, so a source that happened to read these
# particular weeks well looks exactly like a better source. That is why the
# verdict also reports how many of the scored gameweeks the leader actually led,
# and why it says "on N gameweeks" rather than stating a fact about the season.

#: Confidence level for the head-to-head interval, as a z-multiplier.
_Z95 = 1.959964


def _paired_frame(scope: str, pre_only: bool = True) -> pd.DataFrame:
    """Every source's prediction and the outcome, one row per player-gameweek."""
    rows = []
    for gw in projection_archive.scoreable_gameweeks():
        joined, meta = _pair(gw)
        if joined is None or joined.empty:
            continue
        if pre_only and not meta.get("captured_before_deadline"):
            continue
        if scope == SCOPE_START:
            keep, outcome = joined, _started(joined)
            col_scope = SCOPE_START
        else:
            # An if-he-starts projection is only scoreable against a player who
            # did start; anywhere else it measures the minutes model instead.
            keep = joined[pd.to_numeric(joined["started"], errors="coerce") > 0]
            outcome = pd.to_numeric(keep["points"], errors="coerce")
            col_scope = SCOPE_STARTERS
        if keep.empty:
            continue
        one = pd.DataFrame(index=range(len(keep)))
        for source in SOURCE_COLUMNS:
            col = _column_for(source, col_scope)
            if col and col in keep.columns:
                values = pd.to_numeric(keep[col], errors="coerce")
                one[source] = (_as_probability(values) if scope == SCOPE_START
                               else values).values
        one["_outcome"] = outcome.values
        one["_gameweek"] = gw
        rows.append(one)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _loss(frame: pd.DataFrame, source: str, scope: str) -> pd.Series:
    """Per-row error, in whichever metric this scope is judged on."""
    err = frame[source] - frame["_outcome"]
    # Squared for a probability (Brier, a proper scoring rule), absolute for
    # points (MAE) -- the two headline metrics the tables already report.
    return err ** 2 if scope == SCOPE_START else err.abs()


#: Plain-language gloss per metric: what it measures, which way is good, and
#: what a reader should compare the number against. The page renders these
#: rather than assuming the reader knows what a Brier score is -- reported as
#: "not sure what Brier 0.076 means", which is a fair thing not to know.
METRIC_GLOSSARY = {
    "brier": (
        "Brier score",
        "lower",
        "The average squared miss on a probability. Predict 100% for a player "
        "who starts and you score 0; predict 100% for one who does not and you "
        "score 1. It rewards being right *and* being appropriately unsure.",
    ),
    "auc": (
        "AUC",
        "higher",
        "Ordering alone, ignoring whether the probabilities are the right size: "
        "the chance a player who started was rated above one who did not. 0.5 is "
        "a coin flip, 1.0 is perfect. A source that says 100% for everybody "
        "scores 0.5.",
    ),
    "mae": (
        "MAE",
        "lower",
        "The average miss, in points. 2.5 means the typical projection is two "
        "and a half points out.",
    ),
    "bias": (
        "Bias",
        "toward zero",
        "The average miss *with its sign*, so consistently-too-high and merely "
        "noisy are told apart. MAE cannot distinguish them.",
    ),
    "spearman": (
        "Rank correlation",
        "higher",
        "Whether the order is right, ignoring the sizes entirely. 0 is random, "
        "1 is a perfect ordering. This is the half that matters for picking "
        "between players, and the half MAE is blind to.",
    ),
}


def no_skill_reference(scope: str = SCOPE_STARTERS,
                       pre_only: bool = True) -> dict:
    """What a model carrying **no information** scores, so a metric has a scale.

    A raw error number is uninterpretable on its own: 0.076 is excellent for a
    Brier score and 2.5 is unremarkable for an MAE, and nothing on the page said
    so. Both become readable against the dumbest forecast that still respects
    the data -- one constant for everybody.

    * **start** -- everyone gets the observed start rate. That is the
      "climatological" forecast, and the honest baseline for a probability:
      Brier ``p(1-p)``.
    * **starters** / **all** -- everyone gets the same points. Reported against
      the **median**, because the median is what minimises MAE; using the mean
      would flatter every source by comparing it to a baseline that is not even
      the best constant.

    Returns ``{}`` when nothing is scoreable. ``skill`` on the result is the
    fraction of the way from that baseline to perfect, so 0 means "no better
    than one number for everybody" and 1 means exact.
    """
    frames = []
    for gw in projection_archive.scoreable_gameweeks():
        joined, meta = _pair(gw)
        if joined is None or joined.empty:
            continue
        if pre_only and not meta.get("captured_before_deadline"):
            continue
        frames.append(joined)
    if not frames:
        return {}

    data = pd.concat(frames, ignore_index=True)
    started = pd.to_numeric(data.get("started"), errors="coerce").fillna(0) > 0

    if scope == SCOPE_START:
        outcome = started.astype(float)
        if outcome.empty:
            return {}
        rate = float(outcome.mean())
        return {
            "scope": scope,
            "metric": "brier",
            "value": float(((rate - outcome) ** 2).mean()),
            "n": int(len(outcome)),
            "label": "everyone rated %.0f%%" % (rate * 100),
            "detail": "the observed start rate",
        }

    rows = data[started] if scope == SCOPE_STARTERS else data
    actual = pd.to_numeric(rows.get("points"), errors="coerce").dropna()
    if actual.empty:
        return {}
    middle = float(actual.median())
    return {
        "scope": scope,
        "metric": "mae",
        "value": float((actual - middle).abs().mean()),
        "n": int(len(actual)),
        "label": "everyone projected %.1f" % middle,
        "detail": "the median actual score, which is what minimises MAE",
    }


def skill_against_no_skill(value: float, reference: dict) -> Optional[float]:
    """How far from "no information" to perfect a score sits, as a fraction.

    0.0 is no better than one constant for everybody and 1.0 is exact.
    **Negative is a real answer**, not an error: a source can be worse than the
    best constant, and on points every source currently is -- which is the most
    useful thing the Accuracy tab can tell a reader about MAE.
    """
    if not reference or value is None or value != value:
        return None
    base = reference.get("value")
    if not base:
        return None
    return float(1.0 - (value / base))


def head_to_head(scope: str = SCOPE_STARTERS, pre_only: bool = True) -> pd.DataFrame:
    """Every pair of sources, compared on the players both of them priced.

    Returns one row per pair with the mean paired difference, its 95% interval,
    the winner where the interval excludes zero, and the sample size. A pair with
    fewer than ``MIN_HEAD_TO_HEAD_ROWS`` shared rows is reported with no verdict
    rather than dropped -- that they barely overlap is itself worth seeing.
    """
    frame = _paired_frame(scope, pre_only=pre_only)
    if frame.empty:
        return pd.DataFrame()
    sources = [c for c in frame.columns if not c.startswith("_")]

    rows = []
    for i, a in enumerate(sources):
        for b in sources[i + 1:]:
            both = frame.dropna(subset=[a, b, "_outcome"])
            n = len(both)
            if n < 2:
                continue
            diff = _loss(both, a, scope) - _loss(both, b, scope)
            se = float(diff.std(ddof=1)) / np.sqrt(n) if n > 1 else np.nan
            mean = float(diff.mean())
            lo, hi = mean - _Z95 * se, mean + _Z95 * se
            decided = n >= MIN_HEAD_TO_HEAD_ROWS and (lo > 0 or hi < 0)
            rows.append({
                "a": a, "b": b, "n": n,
                "difference": mean, "ci_low": lo, "ci_high": hi,
                "winner": (b if mean > 0 else a) if decided else None,
                "verdict": "clear" if decided else "too close to call",
            })
    return pd.DataFrame(rows)


def marginal_source_value(source: str, scope: str = SCOPE_STARTERS,
                          pre_only: bool = True,
                          weight: float = 0.05) -> dict:
    """Does **adding** this source to the configured blend help?

    `head_to_head` asks which source is better than which, and `fit_blend_weights`
    asks which weighting minimises the metric. Neither answers the question a
    weight decision for a *currently unused* source actually turns on, which is
    whether it adds anything **on top of what the blend already has**. A source
    can be the worst of the four on its own and still carry information none of
    the others do -- and the grid search cannot tell that apart from noise,
    because its argmin always names something and a one-step improvement is
    exactly what noise looks like.

    So this applies the two tests the rest of the tab already trusts: the
    **paired** per-player difference with its interval -- "level with" means the
    interval contains zero -- and **consistency across gameweeks**, since the
    interval is across players and a source that happened to read one week well
    looks identical to a source that helps.

    ``weight`` is the share to try giving it, taken from the rest
    proportionally so the blend still sums to one.

    Returns ``{}`` when the source is unscoreable here. ``supported`` is the
    answer: True only when the interval excludes zero *and* it helps in more
    than half the gameweeks.
    """
    frame = _paired_frame(scope, pre_only=pre_only)
    if frame.empty or source not in frame.columns:
        return {}

    configured = _weights_map()
    others = {k: v for k, v in configured.items()
              if k != source and v > 0 and k in frame.columns}
    if not others:
        return {}

    needed = list(others) + [source, "_outcome", "_gameweek"]
    both = frame.dropna(subset=[c for c in needed if c in frame.columns])
    if len(both) < 2:
        return {}

    total = sum(others.values())
    base = sum((w / total) * both[k] for k, w in others.items())
    candidate = (1.0 - weight) * base + weight * both[source]

    scored = both.assign(_base=base, _cand=candidate)
    loss_base = _loss(scored, "_base", scope)
    loss_cand = _loss(scored, "_cand", scope)
    diff = loss_base - loss_cand                 # positive => adding it helps

    n = len(diff)
    se = float(diff.std(ddof=1)) / np.sqrt(n) if n > 1 else np.nan
    mean = float(diff.mean())
    lo, hi = mean - _Z95 * se, mean + _Z95 * se

    helped = 0
    gameweeks = 0
    for _gw, grp in scored.groupby("_gameweek"):
        gameweeks += 1
        if _loss(grp, "_cand", scope).mean() < _loss(grp, "_base", scope).mean():
            helped += 1

    decided = n >= MIN_HEAD_TO_HEAD_ROWS and (lo > 0 or hi < 0)
    return {
        "source": source,
        "weight": float(weight),
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "n": int(n),
        "gameweeks_helped": int(helped),
        "gameweeks": int(gameweeks),
        "decided": bool(decided),
        "supported": bool(decided and mean > 0 and helped * 2 > gameweeks),
    }


def _weights_map() -> Dict[str, float]:
    """The configured weights as a dict, however config spells them."""
    try:
        import config
        w = getattr(config, "PROJECTION_SOURCE_WEIGHTS", None)
        if isinstance(w, dict) and w:
            return {k: float(v) for k, v in w.items()}
    except Exception:                               # pragma: no cover
        pass
    return {}


def verdict(scope: str = SCOPE_STARTERS, pre_only: bool = True) -> dict:
    """Which source is best at ``scope``, and whether that is settled.

    ``contenders`` are the sources the leader is *not* distinguishable from. A
    verdict naming one winner while a rival sits inside its interval would be
    the overclaiming this whole tab exists to avoid.
    """
    frame = _paired_frame(scope, pre_only=pre_only)
    if frame.empty:
        return {"scope": scope, "best": None, "gameweeks": 0,
                "note": "No pre-deadline gameweek has both projections and actuals yet."}

    sources = [c for c in frame.columns if not c.startswith("_")]
    # Ranked on each source's own coverage, which is the only number that exists
    # for all of them; the pairwise tests below are what actually decide it.
    scores = {}
    for source in sources:
        both = frame.dropna(subset=[source, "_outcome"])
        if len(both):
            scores[source] = float(_loss(both, source, scope).mean())
    if not scores:
        return {"scope": scope, "best": None, "gameweeks": 0,
                "note": "No source published anything scoreable."}

    best = min(scores, key=scores.get)
    h2h = head_to_head(scope, pre_only=pre_only)
    contenders = []
    for _, row in h2h.iterrows():
        if best not in (row["a"], row["b"]):
            continue
        if row["winner"] != best:
            contenders.append(row["b"] if row["a"] == best else row["a"])

    # How many of the scored gameweeks the leader actually led. The interval is
    # across players, so it says nothing about whether these weeks were typical.
    leads, weeks = 0, sorted(frame["_gameweek"].unique())
    for gw in weeks:
        one = frame[frame["_gameweek"] == gw]
        per_gw = {s: float(_loss(one.dropna(subset=[s, "_outcome"]), s, scope).mean())
                  for s in sources if one[s].notna().any()}
        if per_gw and min(per_gw, key=per_gw.get) == best:
            leads += 1

    return {
        "scope": scope,
        "best": best,
        "metric": "brier" if scope == SCOPE_START else "mae",
        "value": scores[best],
        "scores": scores,
        "contenders": contenders,
        "leads_in": leads,
        "gameweeks": len(weeks),
        "n": int(frame[best].notna().sum()),
        "note": "",
    }


# --- The start model's two constants -----------------------------------------
#
# `ROTOWIRE_START_FLOORS` and `ROTOWIRE_OMITTED_START` are fitted by **replaying
# the engine** over archived snapshots with candidate values, never by fitting a
# separate model of what the engine does. This app has twice paid for two
# implementations of one piece of arithmetic drifting apart, and a fitter is the
# worst place for it: the constants it produced would be optimal for a model
# nothing runs.
#
# `blend_aligned` takes both as optional arguments for exactly this, alongside
# `weights`, so nothing here mutates config.

#: How far a replay may sit from the archived value before the fit is refused.
REPLAY_TOLERANCE = 0.01

#: Relative band within which two Brier scores are the same answer. Where the
#: configured value falls inside it, the constant is **left alone** -- an argmin
#: will always name some value, and moving a constant across a flat minimum is
#: fitting the noise. Measured on GW4-GW5, the defensive floor's minimum is a
#: plateau from 0.74 to 0.76 with the configured 0.75 sitting exactly on it; a
#: bare argmin proposed 0.74, an improvement of zero at six decimal places.
FIT_TIE_BAND = 0.005

#: The implied start probability for a player Rotowire omits is never fitted to
#: zero, however much the Brier wants it. Rotowire's silence about a player in a
#: club it covers is usually a benching, but it is sometimes a name this app
#: failed to match -- the two are indistinguishable from here, and a zero states
#: that a player certainly will not start, which for a matching failure is
#: certainly wrong. It is also the value at which the term stops discriminating:
#: a player no other source prices gets the implied value as his whole start
#: probability, so a zero puts `Proj` at exactly 0 for him.
MIN_IMPLIED_START = 0.02


def _start_replay_inputs(gameweeks: Optional[Sequence[int]] = None) -> List[dict]:
    """Per-gameweek arguments for replaying the engine's start model.

    Pre-deadline captures only, per :func:`fit_blend_weights`'s reasoning: a
    backfill can have seen the team news -- in the limit the team sheet -- that
    the model is being asked to predict.

    ``start_pct__rotowire`` is deliberately *not* fed back in. It is the engine's
    own decision about a player rather than anything Rotowire published, and
    supplying it as a source would let the replay reproduce itself perfectly
    while telling us nothing.
    """
    df = start_frame(gameweeks, include_backfills=False)
    if df.empty:
        return []

    out = []
    for gw, one in df.groupby("gameweek"):
        one = one.drop_duplicates(subset="player_id").set_index("player_id")
        if "proj_start__rotowire" not in one.columns or "start_pct" not in one.columns:
            continue
        raw, startpct = {}, {}
        for name in ("rotowire", "ffp", "fpl_ep"):
            col = f"proj_start__{name}"
            if col in one.columns:
                raw[name] = pd.to_numeric(one[col], errors="coerce")
            col = f"start_pct__{name}"
            if name != "rotowire" and col in one.columns:
                startpct[name] = pd.to_numeric(one[col], errors="coerce")
        if "rotowire" not in raw:
            continue
        # The engine's own test, so the replay's cohorts are the real cohorts.
        priced = raw["rotowire"].gt(0).fillna(False)
        coverage = one.loc[priced, "team"].astype(str).value_counts().to_dict() \
            if "team" in one.columns else {}
        as_built = _snapshot_constants(one, priced)
        revisions = pd.to_numeric(
            one.get("start_model_revision"), errors="coerce").dropna().unique() \
            if "start_model_revision" in one.columns else []
        out.append({
            "gameweek": int(gw),
            "revision": int(revisions[0]) if len(revisions) == 1 else None,
            "index": one.index,
            "per_source_raw": raw,
            "per_source_startpct": startpct,
            "positions": one["position"] if "position" in one.columns else None,
            "teams": one["team"] if "team" in one.columns else None,
            "source_club_coverage": {"rotowire": {k: int(v) for k, v in coverage.items()}},
            "archived": _as_probability(one["start_pct"]),
            "y": one["y"].astype(float),
            "listed": priced,
            # The constants this snapshot was *built* with, which are not
            # necessarily the ones configured today -- see `_snapshot_constants`.
            "as_built_floors": as_built[0],
            "as_built_omitted": as_built[1],
        })
    return out


#: Share of a cohort its modal value must hold for that value to be read as the
#: constant which built the snapshot. From revision 3 a cohort legitimately
#: carries a few of the other constant -- `Start_Pct__rotowire` follows the
#: lineups page while `priced` follows the article -- and the observed worst case
#: is 0.91. Below this the cohort is genuinely split and recovers nothing.
MIN_COHORT_DOMINANCE = 0.75


def _snapshot_constants(one: pd.DataFrame, priced: pd.Series):
    """The floors and implied values this snapshot was built with, recovered.

    Fidelity has to be judged against the constants that were in force when the
    snapshot was written, not against today's. Checked against today's, every
    archived gameweek fails the moment the constants are retuned -- and the
    fitter then refuses to fit for ever, having been broken by its own last
    answer.

    Nothing extra needs storing, because the engine already records the decision
    per player: ``Start_Pct__rotowire`` holds the positional floor where Rotowire
    listed a player and the implied value where it did not. Both are constant
    within a position, so the dominant value per cohort *is* the constant.
    Returns ``(None, None)`` for a snapshot predating that column, which then has
    no way to prove what built it.

    **The cohort holds one constant but not only one value, and reading it as
    "exactly one" broke on the first revision-3 snapshot.** ``priced`` is who
    Rotowire's weekly *article* priced; from revision 3, ``Start_Pct__rotowire``
    follows its *projected-lineups page* instead, and the two disagree for a
    handful of players a week -- 6 of 668 at GW8, three swaps across Crystal
    Palace, Leeds and Sunderland. That put the floor *and* the prior inside one
    cohort, so recovery returned nothing and the start-constant fitter refused
    every revision-3+ snapshot permanently, which is exactly the "broken by its
    own last answer" failure the paragraph above exists to avoid.

    So the constant is the cohort's **modal** value, and the cohort is rejected
    only when no value dominates it (``MIN_COHORT_DOMINANCE``). Labelling by
    ``priced`` rather than by taking the larger value keeps this correct if a
    future retune ever ordered a floor below a prior.
    """
    col = one.get("start_pct__rotowire")
    if col is None or not pd.to_numeric(col, errors="coerce").notna().any():
        return None, None
    values = pd.to_numeric(col, errors="coerce")
    positions = one.get("position")
    if positions is None:
        return None, None

    recovered = {}
    for cohort, mask in (("listed", priced), ("omitted", ~priced)):
        found = {}
        for pos, grp in values[mask].groupby(positions[mask]):
            vals = grp.dropna()
            if vals.empty:
                continue
            counts = vals.value_counts()
            # A cohort with no dominant value is not the constant it is being
            # read as, so recover nothing rather than a plausible guess. Observed
            # worst case is 0.91 (21 of 23 forwards at GW8), so the margin here
            # is for small-cohort noise and not for drift.
            if counts.iloc[0] < MIN_COHORT_DOMINANCE * len(vals):
                return None, None
            found[str(pos)] = float(counts.index[0])
        recovered[cohort] = found or None
    return recovered.get("listed"), recovered.get("omitted")


def _replay_start_pct(job: dict, floors=None, omitted=None) -> pd.Series:
    """The engine's resolved ``Start_Pct`` for one archived gameweek."""
    from scripts.common import projection_engine
    from scripts.common.projection_sources import BASIS_CONDITIONAL

    out = projection_engine.blend_aligned(
        index=job["index"],
        per_source_raw=job["per_source_raw"],
        # The archived per-source columns are all `Proj_Start` -- already on the
        # if-he-starts basis, whatever basis the source published on.
        per_source_basis={n: BASIS_CONDITIONAL for n in job["per_source_raw"]},
        per_source_startpct=job["per_source_startpct"],
        starters_only={"rotowire"},
        positions=job["positions"],
        teams=job["teams"],
        source_club_coverage=job["source_club_coverage"],
        start_floors=floors,
        omitted_starts=omitted,
        gameweek=job["gameweek"],
    )
    return pd.to_numeric(out["Start_Pct"], errors="coerce").reindex(job["index"])


def _replay_fidelity(jobs: List[dict]) -> dict:
    """Does replaying with the *current* constants reproduce the archive?

    The load-bearing guard. A fit is only meaningful if the thing being fitted is
    the thing that runs, and the archive spans engine versions -- GW3's snapshot
    was written before Rotowire's omissions implied anything and before FFP's
    start percentage was recovered from its two point columns, so replaying it
    today produces 0.02 where it recorded 1.00. That gameweek must be refused,
    not quietly fitted on.

    Two known gaps are why this is a tolerance rather than equality:
    ``chance_of_playing`` is not archived, so FPL's availability ceiling cannot be
    reapplied (it moved 9 rows in 1,315), and ``priced`` reads a genuine zero
    projection as unpriced.
    """
    per_gw, usable, refused = {}, [], {}
    for job in jobs:
        gw = job["gameweek"]
        # A *constant* retune is recoverable from the snapshot (that is what
        # `_snapshot_constants` is for). A change to how the model resolves a
        # probability at all is not, so it is refused by name rather than left
        # to show up as a deviation -- which reads identically to the archive
        # and the engine having silently parted company.
        if job.get("revision") != projection_engine.START_MODEL_REVISION:
            per_gw[gw] = float("nan")
            refused[gw] = (
                "written by start-model revision %s; the engine is on %d"
                % (job.get("revision") if job.get("revision") is not None
                   else "an unrecorded revision",
                   projection_engine.START_MODEL_REVISION))
            continue
        floors, omitted = job.get("as_built_floors"), job.get("as_built_omitted")
        if floors is None or omitted is None:
            # Nothing in the snapshot says what produced it, so nothing can
            # establish that the engine still would.
            per_gw[gw] = float("nan")
            refused[gw] = "the snapshot does not record the constants that built it"
            continue
        diff = (_replay_start_pct(job, floors=floors, omitted=omitted)
                - job["archived"]).abs()
        worst = float(diff.max()) if len(diff) else float("nan")
        per_gw[gw] = worst
        if worst <= REPLAY_TOLERANCE:
            usable.append(job)
        else:
            refused[gw] = ("replays %.3f away from what was stored, against a "
                           "tolerance of %.2f" % (worst, REPLAY_TOLERANCE))
    return {"per_gameweek": per_gw, "usable": usable, "refused": refused}


def fit_start_constants(gameweeks: Optional[Sequence[int]] = None,
                        floor_grid: Optional[Sequence[float]] = None,
                        omitted_grid: Optional[Sequence[float]] = None) -> dict:
    """Fit the Rotowire start floors and implied values from archived outcomes.

    Scored on **Brier**, which is a proper scoring rule: unlike MAE on a binary
    outcome it cannot be gamed by always answering zero, which matters because
    the omitted cohort mostly does not start. ``config.py`` records that MAE alone
    drove this constant to 0 when it was last fitted by hand.

    **The search is exact rather than greedy, because the constants are
    separable.** Both are applied as ``positions.map(...)``, and a player belongs
    to exactly one position and one cohort, so a goalkeeper's floor cannot touch a
    midfielder's row. That also means one replay per candidate *value* scores that
    value for all four positions at once -- 56 replays rather than 224.

    Each cell is gated on its own ``n``, not on a gameweek count: see
    :data:`MIN_ROWS_PER_START_COHORT`. ``fitted_floors`` and ``fitted_omitted``
    carry the current value wherever the evidence is too thin, so a caller may
    apply them wholesale.
    """
    floors_now, omitted_now = _configured_start_constants()
    jobs = _start_replay_inputs(gameweeks)
    if not jobs:
        return {"fitted_floors": floors_now, "fitted_omitted": omitted_now,
                "current_floors": floors_now, "current_omitted": omitted_now,
                "cohorts": pd.DataFrame(), "gameweeks": [], "n": 0, "fidelity": {},
                "note": "No pre-deadline gameweek has both projections and actuals yet."}

    fid = _replay_fidelity(jobs)
    jobs = fid["usable"]
    if not jobs:
        return {"fitted_floors": floors_now, "fitted_omitted": omitted_now,
                "current_floors": floors_now, "current_omitted": omitted_now,
                "cohorts": pd.DataFrame(), "gameweeks": [], "n": 0,
                "fidelity": fid["per_gameweek"],
                "refused": fid["refused"],
                "note": ("No archived gameweek is replayable by the running "
                         "engine, so there is nothing here it would reproduce. "
                         "Refusing to fit. "
                         + "; ".join("GW%s %s" % (gw, why)
                                     for gw, why in sorted(fid["refused"].items())))}

    floor_grid = list(floor_grid) if floor_grid else [round(0.50 + 0.02 * i, 2) for i in range(25)]
    omitted_grid = list(omitted_grid) if omitted_grid else [round(0.01 * i, 2) for i in range(31)]
    positions = sorted(set(floors_now) | set(omitted_now)) or ["G", "D", "M", "F"]

    # value -> {(position, cohort): (brier, n)}
    scores: Dict[str, Dict[float, Dict[tuple, tuple]]] = {"listed": {}, "omitted": {}}
    for cohort, grid in (("listed", floor_grid), ("omitted", omitted_grid)):
        for value in grid:
            uniform = {p: value for p in positions}
            preds, ys, groups = [], [], []
            for job in jobs:
                got = _replay_start_pct(
                    job,
                    floors=uniform if cohort == "listed" else floors_now,
                    omitted=uniform if cohort == "omitted" else omitted_now)
                mask = job["listed"] if cohort == "listed" else ~job["listed"]
                preds.append(got[mask])
                ys.append(job["y"][mask])
                groups.append(job["positions"][mask])
            pred = pd.concat(preds)
            y = pd.concat(ys)
            pos = pd.concat(groups)
            scores[cohort][value] = {
                (p, cohort): (float(((pred[pos == p] - y[pos == p]) ** 2).mean()),
                              int((pos == p).sum()))
                for p in positions if (pos == p).any()
            }

    rows, fitted = [], {"listed": dict(floors_now), "omitted": dict(omitted_now)}
    current = {"listed": floors_now, "omitted": omitted_now}
    for cohort in ("listed", "omitted"):
        for pos in positions:
            per_value = {v: s[(pos, cohort)] for v, s in scores[cohort].items()
                         if (pos, cohort) in s}
            if not per_value:
                continue
            n = next(iter(per_value.values()))[1]
            grid = sorted(per_value)
            best = min(grid, key=lambda v: per_value[v][0])
            best_brier = per_value[best][0]
            now = current[cohort].get(pos)
            now_brier = (per_value[now][0] if now in per_value
                         else _brier_at(scores[cohort], pos, cohort, now))

            # The data wants to leave the space the constant lives in, so the
            # argmin is a statement about the grid rather than about football.
            at_boundary = best in (grid[0], grid[-1])
            # Two guards before anything moves, in this order.
            tied = (now_brier == now_brier) and now_brier <= best_brier * (1 + FIT_TIE_BAND)
            proposed = now if tied else best
            if cohort == "omitted" and proposed is not None:
                # Clamped rather than refused: the boundary here is zero, and the
                # clamp is the statement about what zero would mean.
                proposed = max(proposed, MIN_IMPLIED_START)
            elif at_boundary and not tied:
                # A floor pinned to the end of the grid is the search running out
                # of room, not a measurement. Hold and report it.
                proposed = now
            enough = n >= MIN_ROWS_PER_START_COHORT
            if enough and proposed is not None:
                fitted[cohort][pos] = proposed
            rows.append({
                "position": pos,
                "cohort": cohort,
                "constant": "floor" if cohort == "listed" else "implied",
                "n": n,
                "current": now,
                "argmin": best,
                "proposed": proposed,
                "current_brier": now_brier,
                "best_brier": best_brier,
                "at_boundary": at_boundary,
                "tied_with_current": bool(tied),
                "enough_rows": enough,
                "moves": bool(enough and proposed is not None and proposed != now),
            })

    return {
        "fitted_floors": fitted["listed"],
        "fitted_omitted": fitted["omitted"],
        "current_floors": floors_now,
        "current_omitted": omitted_now,
        "cohorts": pd.DataFrame(rows),
        "gameweeks": [j["gameweek"] for j in jobs],
        "n": int(sum(len(j["index"]) for j in jobs)),
        "fidelity": fid["per_gameweek"],
        "note": "",
    }


def _brier_at(by_value: dict, position: str, cohort: str, value) -> float:
    """The Brier at ``value``, or at the nearest grid point to it."""
    if value is None:
        return float("nan")
    candidates = [v for v, s in by_value.items() if (position, cohort) in s]
    if not candidates:
        return float("nan")
    nearest = min(candidates, key=lambda v: abs(v - value))
    return by_value[nearest][(position, cohort)][0]


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


def confident_gameweeks(scored: pd.DataFrame) -> int:
    """Gameweeks that may support a conclusion: **pre-deadline captures only**.

    Not the row count. A backfill is a snapshot taken after the deadline, so it
    can see team news -- in the limit, lineups -- that no manager had when they
    picked, and it flatters every source in it. ``fit_blend_weights()`` and
    ``verdict(pre_only=True)`` have always excluded them for that reason; the
    confidence gate did not, and counted them.

    That mattered the moment GW7 landed. Five gameweeks were scoreable, one of
    them the GW3 backfill, so the tab's headline stopped disclaiming itself --
    "5 gameweeks of history", no warning -- while the weight fit directly below
    still said four and still warned. The page was presenting itself as
    confident on the strength of a gameweek it refuses to fit on.

    Falls back to the plain count where the frame carries no ``capture``
    column, which is the looser answer and the safe direction for a caller that
    predates it.
    """
    if scored is None or scored.empty:
        return 0
    if "capture" not in scored.columns:
        return int(scored["gameweek"].nunique())
    pre = scored[scored["capture"] != "backfill"]
    return int(pre["gameweek"].nunique())


def confidence_note(scored: pd.DataFrame) -> str:
    """A plain sentence about whether the sample supports any conclusion."""
    if scored is None or scored.empty:
        return ("No gameweek yet has both a projection snapshot and actual "
                "points. Accuracy can be measured from the first completed "
                "gameweek after snapshots began.")
    gws = confident_gameweeks(scored)
    total = int(scored["gameweek"].nunique())
    # "pre-deadline" is claimed only where the frame actually says so. Without
    # a `capture` column the count is the plain one, and asserting a capture
    # time we do not know would be worse than the looser wording.
    kind = "pre-deadline gameweek" if "capture" in scored.columns else "gameweek"
    # Name the gap rather than quietly reporting a smaller number than the
    # table plainly shows rows for.
    held = ("" if total == gws else
            " (%d scored, but %d taken after the deadline and so not counted "
            "towards this)" % (total, total - gws))
    if gws < MIN_GAMEWEEKS_FOR_CONFIDENCE:
        return ("%d %s%s of history%s. Differences between sources this small "
                "are noise -- treat the table as a sanity check, not a "
                "verdict. At least %d are needed before the ordering means "
                "anything."
                % (gws, kind, "" if gws == 1 else "s", held,
                   MIN_GAMEWEEKS_FOR_CONFIDENCE))
    return "%d %ss of history%s." % (gws, kind, held)


# =============================================================================
# How far ahead is worth projecting?
# =============================================================================

#: FFP's cumulative horizon columns, shortest first. Each **includes** the
#: anchor gameweek -- ``Next2GWs == Predicted + the following week`` -- so the
#: window length is the number in the name, not a count of weeks beyond it.
#: See CLAUDE.md's "Fantasy Football Pundit feed".
HORIZON_COLUMNS = {
    1: "Predicted",
    2: "Next2GWs",
    3: "Next3GWs",
    4: "Next4GWs",
    5: "Next5GWs",
    6: "Next6GWs",
}

#: The column ROS actually percentiles, as `Proj_Next3`. Named here so the
#: report can mark it rather than the page hard-coding a 3.
ROS_HORIZON_WEEKS = 3

#: Rows a single anchor needs before its ordering means anything. Lower than
#: `MIN_HEAD_TO_HEAD_ROWS` because this is one population scored against itself
#: rather than two sources' overlap.
MIN_HORIZON_ROWS = 50


def _ffp_archive_frame(gameweek: int):
    """One archived FFP table, keyed on ``Player_ID``, plus its meta.

    ``(None, {})`` when the gameweek is absent *or* carries no ids. The GW3
    archive is the spreadsheet recovery, which has no ``Player_ID`` at all --
    the sheet publishes names only -- and this report joins on the integer id
    rather than name-matching, so that gameweek is simply not scoreable here.
    """
    from scripts.common import ffp_archive

    df, meta = ffp_archive.load_gameweek(gameweek)
    if df is None or df.empty or "Player_ID" not in df.columns:
        return None, {}
    df = df[df["Player_ID"].notna()].copy()
    if df.empty:
        return None, {}
    df["Player_ID"] = pd.to_numeric(df["Player_ID"], errors="coerce")
    df = df.dropna(subset=["Player_ID"])
    df["Player_ID"] = df["Player_ID"].astype("int64")
    return df.drop_duplicates("Player_ID").set_index("Player_ID"), meta


def _actual_points(gameweek: int):
    """Actual points for one gameweek, keyed on player id, or None."""
    df, _ = projection_archive.load_actuals(gameweek)
    if df is None or df.empty:
        return None
    for id_col in ("player_id", "Player_ID"):
        if id_col in df.columns:
            break
    else:
        return None
    if "points" not in df.columns:
        return None
    out = df[[id_col, "points"]].dropna(subset=[id_col]).copy()
    out[id_col] = pd.to_numeric(out[id_col], errors="coerce")
    out = out.dropna(subset=[id_col])
    out[id_col] = out[id_col].astype("int64")
    return (out.drop_duplicates(id_col).set_index(id_col)["points"]
            .pipe(pd.to_numeric, errors="coerce"))


def _actual_window(start_gw: int, weeks: int):
    """Total actual points over ``weeks`` gameweeks from ``start_gw``, or None.

    All or nothing: a partial window would score a 6-week forecast against 3
    weeks of outcome and flatter the shorter horizons, which is the comparison
    this whole report is making.
    """
    total = None
    for gw in range(start_gw, start_gw + weeks):
        points = _actual_points(gw)
        if points is None:
            return None
        total = points if total is None else total.add(points, fill_value=0)
    return total


def score_horizon_lengths() -> pd.DataFrame:
    """How well each horizon length orders what actually happened.

    The question behind the 40% multi-gameweek weight in ROS: a waiver claim or
    a transfer is a lasting decision, so is a longer forecast window worth more?
    FFP publishes six, the app scores three, and nothing had ever measured
    whether the extra weeks help.

    One row per (anchor, horizon length). Each anchor is scored over **the
    longest complete run of actuals available from it**, so a longer-lasting
    decision is judged over a longer outcome -- otherwise the test is rigged
    towards the short horizons. ``rho`` is Spearman against that total, because
    ROS percentiles this term: the ordering is what matters, not the scale.

    Measured GW4-GW6 the answer was a clean no -- rho fell monotonically past
    two gameweeks, on both available anchors, and `Next3GWs` and `Next6GWs`
    correlate 0.988 with each other so the choice barely moves a ranking at
    all. Two structural reasons, both visible in the other two reports below.
    """
    from scripts.common import ffp_archive

    actual_gws = set(projection_archive.list_actuals())
    rows = []
    for anchor in ffp_archive.list_archived():
        frame, meta = _ffp_archive_frame(anchor)
        if frame is None:
            continue
        # The longest complete run of actuals starting at this anchor.
        weeks_scored = 0
        while anchor + weeks_scored in actual_gws:
            weeks_scored += 1
        if weeks_scored < 1:
            continue
        outcome = _actual_window(anchor, weeks_scored)
        if outcome is None:
            continue
        common = frame.index.intersection(outcome.index)
        if len(common) < MIN_HORIZON_ROWS:
            continue
        y = outcome.reindex(common)
        for weeks, col in sorted(HORIZON_COLUMNS.items()):
            if col not in frame.columns:
                continue
            x = pd.to_numeric(frame[col].reindex(common), errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < MIN_HORIZON_ROWS:
                continue
            rows.append({
                "anchor_gw": anchor,
                "horizon_weeks": weeks,
                "column": col,
                "weeks_scored": weeks_scored,
                "rho": spearman(x[mask], y[mask]),
                "n": int(mask.sum()),
                "window_gw": meta.get("window_gw"),
                "is_ros_term": weeks == ROS_HORIZON_WEEKS,
            })
    return pd.DataFrame(rows)


def score_horizon_decay() -> pd.DataFrame:
    """How much worse a forecast gets the further ahead it is made.

    The *reason* behind :func:`score_horizon_lengths`, and the more durable
    measurement of the two: it compares one FFP publication against itself at
    different lead times, so the model and its vintage are held constant and
    only the distance to the target week varies.

    ``lead_weeks`` comes from the archive's own ``window_gw`` -- the gameweek
    FFP's publication window opened on -- so a table archived for GW6 from a
    GW4 window is a two-week-early forecast. Measured on the 11 September
    publication: rho +0.508 same-week, +0.466 one week out, +0.352 two weeks
    out, so roughly a third of the ordering is gone by the third week of a
    window. Each later week of a six-week horizon is worth less than the last,
    which is why lengthening the window does not pay.

    Scored on ``Predicted`` -- the week's own number -- not a cumulative total,
    since the point is to isolate one target week.
    """
    from scripts.common import ffp_archive

    rows = []
    for target in ffp_archive.list_archived():
        frame, meta = _ffp_archive_frame(target)
        if frame is None or "Predicted" not in frame.columns:
            continue
        window_gw = meta.get("window_gw")
        if window_gw is None:
            continue
        outcome = _actual_points(target)
        if outcome is None:
            continue
        common = frame.index.intersection(outcome.index)
        x = pd.to_numeric(frame["Predicted"].reindex(common), errors="coerce")
        y = outcome.reindex(common)
        mask = x.notna() & y.notna()
        if mask.sum() < MIN_HORIZON_ROWS:
            continue
        rows.append({
            "target_gw": target,
            "window_gw": int(window_gw),
            "lead_weeks": int(target) - int(window_gw),
            "rho": spearman(x[mask], y[mask]),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows).sort_values("lead_weeks") if rows else pd.DataFrame()


def score_horizon_overlap() -> pd.DataFrame:
    """What each horizon length is *made of*: fixtures, or just player quality.

    The second structural reason a longer window does not pay, and the one that
    argues against giving the far weeks their own separate term.

    FFP publishes **one** start probability per player across all six forecast
    weeks (verified in the live payload: 404 of 404 players carry exactly one
    distinct value, and the five forward weeks are identical). So the only thing
    varying across a horizon is fixture difficulty -- and over six weeks
    fixtures average out. The longer the window, the more it collapses into "is
    he a good player", which ROS already weights at 30-45% as ``season_quality``
    and measures from *actual* points rather than a forecast.

    ``rho_vs_season`` is each horizon's rank correlation with points scored
    **before** the anchor, so it is a statement about what the forecast
    duplicates, not about accuracy. Measured at GW4 it climbs +0.721 (1 week)
    -> +0.770 (3) -> +0.775 (6). ``rho_vs_ros`` is the correlation with the
    window ROS actually uses: at 0.988 for six weeks against three, the two
    orderings are nearly the same variable.
    """
    from scripts.common import ffp_archive

    actual_gws = set(projection_archive.list_actuals())
    rows = []
    for anchor in ffp_archive.list_archived():
        frame, _ = _ffp_archive_frame(anchor)
        if frame is None:
            continue
        prior = [gw for gw in sorted(actual_gws) if gw < anchor]
        if not prior:
            continue
        season = None
        for gw in prior:
            points = _actual_points(gw)
            if points is None:
                continue
            season = points if season is None else season.add(points, fill_value=0)
        if season is None:
            continue
        ros_col = HORIZON_COLUMNS[ROS_HORIZON_WEEKS]
        if ros_col not in frame.columns:
            continue
        ros = pd.to_numeric(frame[ros_col], errors="coerce")
        for weeks, col in sorted(HORIZON_COLUMNS.items()):
            if col not in frame.columns:
                continue
            x = pd.to_numeric(frame[col], errors="coerce")
            common = frame.index.intersection(season.index)
            xs, ys = x.reindex(common), season.reindex(common)
            mask = xs.notna() & ys.notna()
            if mask.sum() < MIN_HORIZON_ROWS:
                continue
            both = x.notna() & ros.notna()
            rows.append({
                "anchor_gw": anchor,
                "horizon_weeks": weeks,
                "column": col,
                "rho_vs_season": spearman(xs[mask], ys[mask]),
                "rho_vs_ros": (spearman(x[both], ros[both])
                               if both.sum() >= MIN_HORIZON_ROWS else float("nan")),
                "prior_gws": len(prior),
                "n": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def horizon_verdict() -> dict:
    """Should the multi-gameweek term look further ahead than it does?

    Returns ``{"best_weeks", "ros_weeks", "margin", "anchors", "agreed",
    "note"}``, or ``{"note": ...}`` alone when nothing is scoreable.

    ``agreed`` is how many anchors put ``best_weeks`` top, and it carries the
    weight here: each anchor is one FFP publication scored over a handful of
    weeks, so a horizon that wins on one anchor and loses on another has told
    us nothing. The note refuses to recommend a change on a single anchor, the
    same way ``verdict()`` reports "led in N of M gameweeks" rather than a bare
    margin.
    """
    scored = score_horizon_lengths()
    if scored.empty:
        return {"note": "No archived FFP gameweek has a complete run of actual "
                        "points after it yet, so horizon length cannot be scored."}

    per_anchor = scored.pivot_table(index="anchor_gw", columns="horizon_weeks",
                                   values="rho")
    winners = per_anchor.idxmax(axis=1).dropna()
    mean_rho = scored.groupby("horizon_weeks")["rho"].mean()
    best_weeks = int(mean_rho.idxmax())
    ros_rho = float(mean_rho.get(ROS_HORIZON_WEEKS, float("nan")))
    margin = float(mean_rho.max() - ros_rho) if ros_rho == ros_rho else float("nan")
    agreed = int((winners == best_weeks).sum())
    anchors = int(len(per_anchor))

    if best_weeks == ROS_HORIZON_WEEKS:
        note = ("%d-gameweek is the best-ordering window of the six on %d of %d "
                "anchors, which is what ROS already uses."
                % (best_weeks, agreed, anchors))
    elif anchors < 2 or agreed < anchors:
        note = ("%d-gameweek orders best on average (+%.3f rho over the "
                "%d-gameweek window ROS uses) but leads on only %d of %d "
                "anchors. One anchor is one FFP publication -- not grounds to "
                "move the term."
                % (best_weeks, margin, ROS_HORIZON_WEEKS, agreed, anchors))
    else:
        note = ("%d-gameweek orders best on every one of %d anchors, +%.3f rho "
                "over the %d-gameweek window ROS uses. Worth considering once "
                "there are %d scored gameweeks."
                % (best_weeks, anchors, margin, ROS_HORIZON_WEEKS,
                   MIN_GAMEWEEKS_FOR_CONFIDENCE))

    longer = mean_rho[mean_rho.index > ROS_HORIZON_WEEKS]
    if not longer.empty and longer.max() < ros_rho:
        note += (" Every window longer than %d gameweeks orders *worse*, so "
                 "looking further ahead is not the improvement it sounds like."
                 % ROS_HORIZON_WEEKS)

    return {"best_weeks": best_weeks, "ros_weeks": ROS_HORIZON_WEEKS,
            "margin": margin, "anchors": anchors, "agreed": agreed,
            "mean_rho": mean_rho.to_dict(), "note": note}
