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
        out.append({
            "gameweek": int(gw),
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
    within a position, so the unique value per cohort *is* the constant. Returns
    ``(None, None)`` for a snapshot predating that column, which then has no way
    to prove what built it.
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
            uniq = grp.dropna().unique()
            # More than one value in a cohort means this is not the constant it
            # is being read as, so recover nothing rather than a plausible guess.
            if len(uniq) != 1:
                return None, None
            found[str(pos)] = float(uniq[0])
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
    per_gw, usable = {}, []
    for job in jobs:
        floors, omitted = job.get("as_built_floors"), job.get("as_built_omitted")
        if floors is None or omitted is None:
            # Nothing in the snapshot says what produced it, so nothing can
            # establish that the engine still would.
            per_gw[job["gameweek"]] = float("nan")
            continue
        diff = (_replay_start_pct(job, floors=floors, omitted=omitted)
                - job["archived"]).abs()
        worst = float(diff.max()) if len(diff) else float("nan")
        per_gw[job["gameweek"]] = worst
        if worst <= REPLAY_TOLERANCE:
            usable.append(job)
    return {"per_gameweek": per_gw, "usable": usable}


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
                "note": ("No archived gameweek replays to within %.2f of what was "
                         "stored, so there is nothing here the running engine "
                         "would reproduce. Refusing to fit."% REPLAY_TOLERANCE)}

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
