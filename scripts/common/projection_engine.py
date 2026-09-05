"""
The projection engine — one blend, one contract, one place.

Before this module the app had two implementations of "the 60/40 blend"
(``compute_player_scores`` and ``blend_fixture_projections``), hand-copied from
each other and quietly divergent: only one fell back to the FPL
``chance_of_playing`` when FFP had not published a start percentage. The same
player therefore carried different "blended" projections depending on which
page you were looking at, and nothing said so. Three quarters of the app
sidestepped both and rendered raw Rotowire.

Everything here is pure: no Streamlit, no network. Sources are fetched by
:mod:`scripts.common.projection_sources` and passed in, so the app (cached) and
the GitHub Actions snapshot collector (uncached, possibly without Streamlit
installed) blend identically.

**The design idea is that sources declare their basis and the engine converts.**
A ``conditional`` source says what a player scores *if he starts*; an
``unconditional`` one has already priced in the chance he plays. Conversion --
``Proj = Proj_Start x Start_Pct`` -- happens exactly once, here. No caller ever
has to remember which kind of number it is holding, which is what caused the
double-discount bug three separate times.

Output contract (see ``CANONICAL_COLUMNS``):

===============  =========================================================
``Proj_Start``   points if he starts (conditional)
``Start_Pct``    P(starts), 0-1
``Proj``         expected points = ``Proj_Start * Start_Pct``
``Proj_Next3``   3-gameweek expected points
``Proj_Src``     which sources contributed, e.g. ``"RW+FFP"``
``Proj_Spread``  max-min of ``Proj_Start`` across sources -- disagreement
``Proj_GW``      the gameweek these numbers describe
===============  =========================================================
"""

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from scripts.common.name_matching import ReferenceMatcher
from scripts.common.projection_sources import (
    BASIS_CONDITIONAL,
    BASIS_UNCONDITIONAL,
    COVERS_STARTERS,
    SourceResult,
)

_logger = logging.getLogger("fpl_app.projections")

CANONICAL_COLUMNS = [
    "Proj", "Proj_Start", "Start_Pct", "Proj_Next3",
    "Proj_Src", "Proj_Spread", "Proj_GW",
]

#: Short labels for ``Proj_Src``. Kept terse because this renders in a table cell.
SOURCE_LABELS = {"rotowire": "RW", "ffp": "FFP", "fpl_ep": "xP", "odds": "ODDS"}

#: Default blend weights, matching the app's long-standing 60/40 Rotowire/FFP
#: split so this refactor is behaviour-preserving where both sources are present.
#: ``fpl_ep`` is wired but weighted 0 until the accuracy harness has an opinion:
#: it is carried through for display and for the snapshot either way.
#: Overridden by ``config.PROJECTION_SOURCE_WEIGHTS``.
DEFAULT_WEIGHTS = {"rotowire": 0.6, "ffp": 0.4, "fpl_ep": 0.0, "odds": 0.0}

#: A start-probability floor applied when *Rotowire* prices a player. Rotowire
#: only lists expected starters, so its presence is itself a confidence signal
#: and stops FFP's uncertainty from fully overriding an expert lineup call. The
#: DEF floor is highest because a defender who starts plays 90 minutes -- there
#: is no "came on late for two points" outcome the way there is for MID/FWD.
DEFAULT_START_FLOORS = {"G": 0.80, "D": 0.75, "M": 0.68, "F": 0.65}

#: Divisor floor when recovering a conditional value from an unconditional one.
START_RECOVERY_FLOOR = 0.05


def _weights() -> Dict[str, float]:
    try:
        import config
        w = getattr(config, "PROJECTION_SOURCE_WEIGHTS", None)
        if isinstance(w, dict) and w:
            return dict(w)
    except Exception:                       # pragma: no cover - config is optional here
        pass
    return dict(DEFAULT_WEIGHTS)


def _start_floors() -> Dict[str, float]:
    try:
        import config
        f = getattr(config, "ROTOWIRE_START_FLOORS", None)
        if isinstance(f, dict) and f:
            return dict(f)
    except Exception:                       # pragma: no cover
        pass
    return dict(DEFAULT_START_FLOORS)


def _resolve_ids(source: SourceResult, pool: pd.DataFrame) -> pd.Series:
    """Map each of ``source.df``'s rows to a pool ``Player_ID``.

    An integer id join is always preferred and is the only join FFP and the FPL
    bootstrap need -- FFP resolves 368/368 through the bootstrap ``code``. Only
    Rotowire publishes names alone, and it goes through ``ReferenceMatcher``
    rather than any bespoke matching: cross-source name matching is the single
    most frequent source of silent bugs in this app, and there is exactly one
    implementation of it on purpose.
    """
    df = source.df
    if "Player_ID" in df.columns:
        ids = pd.to_numeric(df["Player_ID"], errors="coerce")
        if ids.notna().any():
            return ids

    if "Player" not in df.columns:
        return pd.Series(np.nan, index=df.index)

    # Normalise positions on BOTH sides before matching. Every ReferenceMatcher
    # tier below the first two is scoped by position, so a G/D/M/F pool and a
    # GK/DEF/MID/FWD source share no group and every name that is not an exact
    # (name, team) hit falls straight through. FFP publishes GK/DEF/MID/FWD on
    # both its paths; the site payload is saved by its integer Player_ID, so this
    # only bites where there is no id to join on -- which is exactly the archived
    # and spreadsheet tables. Measured live on the GW3 archive: it costs roughly
    # a third of the matches, and the misses are silent.
    matcher_pool = pool.copy()
    if "Position" in matcher_pool.columns:
        matcher_pool["Position"] = _normalise_positions(matcher_pool["Position"])

    matcher = ReferenceMatcher(
        matcher_pool,
        name_col="Player",
        web_name_col="Web_Name" if "Web_Name" in matcher_pool.columns else None,
        team_col="Team" if "Team" in matcher_pool.columns else None,
        position_col="Position" if "Position" in matcher_pool.columns else None,
    )
    pool_ids = (matcher_pool["Player_ID"] if "Player_ID" in matcher_pool.columns
                else pd.Series(matcher_pool.index, index=matcher_pool.index))

    positions = (_normalise_positions(df["Position"]) if "Position" in df.columns
                 else pd.Series(None, index=df.index))

    out = []
    for idx_label, row in df.iterrows():
        # Query on the full name first, then the short one. A source publishes
        # common names ("Bruno Fernandes") while the bootstrap publishes legal
        # ones ("Bruno Borges Fernandes"), and either can be the one that hits.
        hit = matcher.match(row.get("Player"), row.get("Team"), positions.loc[idx_label])
        if hit is None and row.get("Web_Name"):
            hit = matcher.match(row.get("Web_Name"), row.get("Team"),
                                positions.loc[idx_label])
        out.append(pool_ids.loc[hit] if hit is not None else np.nan)
    return pd.Series(out, index=df.index, dtype="float64")


def _normalise_positions(values: pd.Series) -> pd.Series:
    """Positions as G/D/M/F, whatever spelling arrived."""
    from scripts.common.text_helpers import POS_MAP_TO_RW
    return values.astype(str).str.strip().map(POS_MAP_TO_RW).fillna(values)


def build_projections(
    sources: Sequence[SourceResult],
    *,
    gameweek: Optional[int],
    pool: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Blend every usable source into one canonical projection frame.

    Args:
        sources: ``SourceResult`` objects. Unusable ones (unreachable, empty, or
            published for a different gameweek) are dropped with a note rather
            than blended -- a wrong gameweek is worse than a missing source.
        gameweek: the gameweek being projected. A source whose own gameweek
            disagrees is excluded. An *unknown* source gameweek is not a wrong
            one and does not gate.
        pool: the canonical player universe. Needs ``Player_ID``; uses ``Player``,
            ``Team``, ``Position``, ``Web_Name``, ``status`` and
            ``chance_of_playing_next_round`` when present.
        weights: per-source blend weights. Defaults to
            ``config.PROJECTION_SOURCE_WEIGHTS``.

    Returns:
        A frame indexed by ``Player_ID`` carrying ``CANONICAL_COLUMNS`` plus one
        ``Proj_Start__<source>`` column per source -- the per-source values kept
        so the Projections Hub can show what went into the blend, and so the
        snapshot can record it for later accuracy scoring.
    """
    weights = dict(weights) if weights is not None else _weights()
    floors = _start_floors()

    if pool is None or pool.empty or "Player_ID" not in pool.columns:
        _logger.warning("build_projections: no usable player pool")
        return pd.DataFrame(columns=CANONICAL_COLUMNS)

    ids = pd.to_numeric(pool["Player_ID"], errors="coerce")
    out = pd.DataFrame(index=pd.Index(ids.dropna().astype("int64").unique(), name="Player_ID"))

    usable, dropped = [], []
    for s in sources:
        if not s.ok:
            dropped.append((s.name, s.note or "no rows"))
            continue
        if s.is_stale(gameweek):
            dropped.append((s.name, f"published for GW{s.gameweek}, not GW{gameweek}"))
            continue
        usable.append(s)
    for name, why in dropped:
        _logger.info("Projection source %r excluded: %s", name, why)

    # --- Per-source raw values ---------------------------------------------
    # Collected as published, *before* any basis conversion. Conversion needs
    # the resolved start probability, and that in turn needs to know which
    # sources priced each player (Rotowire's presence sets a floor), so the
    # order is: collect, resolve start, then convert.
    per_source_raw: Dict[str, pd.Series] = {}
    per_source_basis: Dict[str, str] = {}
    per_source_startpct: Dict[str, pd.Series] = {}
    per_source_next3: Dict[str, pd.Series] = {}

    for s in usable:
        resolved = _resolve_ids(s, pool)
        df = s.df.assign(_pid=resolved).dropna(subset=["_pid"])
        if df.empty:
            _logger.warning("Projection source %r resolved no players against the pool", s.name)
            continue
        df["_pid"] = df["_pid"].astype("int64")
        # One row per player. A source that lists a player twice (a per-fixture
        # table in a double gameweek) is summed, not silently halved.
        grouped = df.groupby("_pid", sort=False)

        if "Start_Pct" in df.columns:
            sp = grouped["Start_Pct"].max().reindex(out.index)
            per_source_startpct[s.name] = sp
            out[f"Start_Pct__{s.name}"] = sp

        if "Proj_Start" in df.columns:
            v = grouped["Proj_Start"].sum(min_count=1).reindex(out.index)
        elif "Proj" in df.columns:
            v = grouped["Proj"].sum(min_count=1).reindex(out.index)
        else:
            v = None

        if v is not None:
            per_source_raw[s.name] = v
            per_source_basis[s.name] = s.basis

        if "Proj_Next3" in df.columns:
            per_source_next3[s.name] = grouped["Proj_Next3"].sum(min_count=1).reindex(out.index)

    return blend_aligned(
        index=out.index,
        per_source_raw=per_source_raw,
        per_source_basis=per_source_basis,
        per_source_startpct=per_source_startpct,
        per_source_next3=per_source_next3,
        starters_only={s.name for s in usable if s.covers == COVERS_STARTERS},
        positions=_pool_col(pool, "Position", out.index, default="M"),
        chance_of_playing=_pool_col(pool, "chance_of_playing_next_round", out.index),
        status=_pool_col(pool, "status", out.index),
        weights=weights,
        gameweek=gameweek,
        extra=out,
    )


def _pool_col(pool, col, index, default=None):
    """One column of the pool, reindexed onto Player_ID. ``default`` when absent."""
    if col not in pool.columns:
        return pd.Series(default, index=index)
    by_id = pool.dropna(subset=["Player_ID"]).copy()
    by_id["Player_ID"] = pd.to_numeric(by_id["Player_ID"], errors="coerce").astype("int64")
    by_id = by_id.drop_duplicates(subset=["Player_ID"]).set_index("Player_ID")
    return by_id[col].reindex(index)


def blend_aligned(
    *,
    index,
    per_source_raw: Dict[str, pd.Series],
    per_source_basis: Dict[str, str],
    per_source_startpct: Optional[Dict[str, pd.Series]] = None,
    per_source_next3: Optional[Dict[str, pd.Series]] = None,
    starters_only: Optional[set] = None,
    positions: Optional[pd.Series] = None,
    chance_of_playing: Optional[pd.Series] = None,
    status: Optional[pd.Series] = None,
    weights: Optional[Dict[str, float]] = None,
    fallback_names: Optional[Sequence[str]] = None,
    gameweek: Optional[int] = None,
    extra: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """The blend itself, over Series that are already aligned to one index.

    This is the single implementation of the arithmetic. ``build_projections``
    calls it after resolving every source to a ``Player_ID``; the page-facing
    helpers in ``analytics.py`` call it directly, because there the sources are
    already columns on one frame (Rotowire values arrive merged, and FFP is
    joined by ``merge_ffp_single_gw_data``'s tiered matcher). Both paths must
    produce the same number for the same player, which is only guaranteed if
    there is one copy of this code -- there used to be two, and they disagreed.

    ``per_source_raw`` holds values *as published*; ``per_source_basis`` says
    what each one means. Conversion to a common basis happens here, once.
    """
    weights = dict(weights) if weights is not None else _weights()
    floors = _start_floors()
    per_source_startpct = dict(per_source_startpct or {})
    per_source_next3 = dict(per_source_next3 or {})
    starters_only = set(starters_only or ())

    out = extra if extra is not None else pd.DataFrame(index=index)
    if positions is None:
        positions = pd.Series("M", index=index)
    positions = positions.reindex(index)

    # --- Start probability --------------------------------------------------
    # FFP first (continuous 0-100, the only real start model any source gives
    # us), then FPL's own chance_of_playing, then "no news means he plays".
    # The chance_of_playing step is the fallback that existed in
    # compute_player_scores and was missing from blend_fixture_projections;
    # unifying on the version that has it is the point of this module.
    start_pct = pd.Series(np.nan, index=index, dtype="float64")
    for name in ("ffp", "fpl_ep", "odds"):
        if name in per_source_startpct:
            start_pct = start_pct.fillna(per_source_startpct[name].reindex(index))
    if chance_of_playing is not None:
        chance = pd.to_numeric(chance_of_playing.reindex(index), errors="coerce") / 100.0
        start_pct = start_pct.fillna(chance)
    start_pct = start_pct.fillna(1.0).clip(0, 1)

    # Rotowire's presence as a confidence signal: it lists only expected
    # starters, so a player it prices gets a positional floor on start
    # probability. This is why `covers` is part of the source contract.
    for name in starters_only:
        if name not in per_source_raw:
            continue
        priced = per_source_raw[name].reindex(index).gt(0)
        for pos_code, floor_val in floors.items():
            mask = priced & (positions == pos_code)
            start_pct[mask] = start_pct[mask].clip(lower=floor_val)
    out["Start_Pct"] = start_pct

    # --- Basis conversion ---------------------------------------------------
    # Every source is put on the conditional basis before blending. An
    # expected-value source averaged straight against if-he-starts sources
    # drags the blend down by exactly the start probability -- the same shape as
    # the double discount that ran the FFP term ~44% low. A source carrying its
    # own start probability is un-discounted by that; one that does not uses the
    # resolved value, never 1.0.
    per_source_start: Dict[str, pd.Series] = {}
    for name, v in per_source_raw.items():
        v = v.reindex(index)
        if per_source_basis.get(name) == BASIS_UNCONDITIONAL:
            sp = per_source_startpct.get(name)
            sp = start_pct if sp is None else sp.reindex(index).fillna(start_pct)
            v = v / sp.clip(lower=START_RECOVERY_FLOOR)
        per_source_start[name] = v
        out[f"Proj_Start__{name}"] = v

    # --- The blend ----------------------------------------------------------
    # Weights are renormalised over the sources that actually priced *this*
    # player, so a missing source is one rule instead of the ad-hoc mask
    # substitution each old callsite carried. A zero-weight source is carried
    # for display and snapshotting but never contributes.
    blend_names = [n for n in per_source_start if weights.get(n, 0) > 0]

    numer = pd.Series(0.0, index=index)
    denom = pd.Series(0.0, index=index)
    for name in blend_names:
        v = per_source_start[name]
        w = float(weights.get(name, 0.0))
        present = v.notna() & v.gt(0)
        numer = numer.add((v * w).where(present, 0.0), fill_value=0.0)
        denom = denom.add(pd.Series(w, index=index).where(present, 0.0), fill_value=0.0)

    proj_start = (numer / denom).where(denom.gt(0))

    # --- Fallback sources ---------------------------------------------------
    # A fallback fills only where no weighted source priced the player at all.
    # This is what FPL's `ep_next` was already doing on Classic Transfers, except
    # it was written *into the Rotowire column*, so it silently took Rotowire's
    # 60% weight and read as Rotowire everywhere downstream. Same behaviour,
    # declared: it appears in Proj_Src under its own name, and it can never
    # displace a source that actually priced the player.
    used_fallback = pd.Series(False, index=index)
    for _fb in (fallback_names or []):
        if _fb not in per_source_start:
            continue
        v = per_source_start[_fb]
        fills = proj_start.isna() & v.notna() & v.gt(0)
        proj_start = proj_start.fillna(v.where(fills))
        used_fallback |= fills

    out["Proj_Start"] = proj_start.round(3)
    out["Proj"] = (proj_start * start_pct).round(3)

    # A unit mismatch between two sources is the single most expensive failure
    # this app has had: Rotowire once published a five-gameweek cumulative table
    # under a weekly heading and every projection in the app was 5x too big,
    # with nothing raising. `check_source_scale_agreement` was written for
    # exactly that and, until now, ran only inside tests -- so the blend itself
    # was unguarded at runtime. Logged, never raised: a page must degrade, not die.
    if len(blend_names) >= 2:
        _warn_on_scale_disagreement(per_source_start, blend_names)

    # --- Provenance and disagreement ---------------------------------------
    if blend_names:
        stacked = pd.concat([per_source_start[n].rename(n) for n in blend_names], axis=1)
        priced = stacked.gt(0) & stacked.notna()
        out["Proj_Spread"] = (stacked.where(priced).max(axis=1)
                              - stacked.where(priced).min(axis=1)).round(3)
        labels = pd.Series("", index=index)
        for n in blend_names:
            tag = SOURCE_LABELS.get(n, n)
            labels = labels.where(~priced[n], labels.where(labels.eq(""), labels + "+") + tag)
        out["Proj_Src"] = labels.replace("", "None")
    else:
        out["Proj_Spread"] = np.nan
        out["Proj_Src"] = pd.Series("None", index=index)

    for _fb in (fallback_names or []):
        if _fb not in per_source_start:
            continue
        v = per_source_start[_fb]
        fills = used_fallback & v.notna() & v.gt(0)
        out["Proj_Src"] = out["Proj_Src"].where(~fills, SOURCE_LABELS.get(_fb, _fb))

    # --- Blank gameweeks are unknown, not zero ------------------------------
    # A player nobody priced who is not injured or suspended has almost always
    # had his fixture postponed. Scoring that as 0 reads as "drop him", which is
    # how an elite asset came to be recommended for the waiver wire in a blank.
    # NaN is the honest value and every consumer already treats it as neutral.
    unavailable = pd.Series(False, index=index)
    if status is not None:
        unavailable |= status.reindex(index).isin(["i", "s", "u"]).fillna(False)
    if chance_of_playing is not None:
        c = pd.to_numeric(chance_of_playing.reindex(index), errors="coerce")
        unavailable |= (c.notna() & c.lt(50))
    unpriced = out["Proj_Start"].isna()
    out.loc[unpriced & unavailable, ["Proj", "Proj_Start"]] = 0.0

    # --- Multi-gameweek -----------------------------------------------------
    next3 = pd.Series(np.nan, index=index, dtype="float64")
    for name in ("ffp", "rotowire", "fpl_ep"):
        if name in per_source_next3:
            next3 = next3.fillna(per_source_next3[name].reindex(index))
    out["Proj_Next3"] = next3

    out["Proj_GW"] = gameweek
    return out


def _warn_on_scale_disagreement(per_source_start, blend_names) -> None:
    """Log if two sources look denominated in different units.

    Independent projections disagree about individual players constantly; they
    should still agree within a factor of two on the *typical* player. A
    systematic multiple is a unit mismatch, not a difference of opinion.
    """
    try:
        from scripts.common.data_validation import check_source_scale_agreement
    except Exception:                       # pragma: no cover - defensive
        return
    for i, a in enumerate(blend_names):
        for b in blend_names[i + 1:]:
            for issue in check_source_scale_agreement(
                per_source_start[a].dropna(), per_source_start[b].dropna(),
                label_a=a, label_b=b,
            ):
                log = _logger.error if issue.severity == "error" else _logger.warning
                log("Projection blend: %s", issue)


def attach_projections(
    df: pd.DataFrame,
    projections: pd.DataFrame,
    *,
    on: str = "Player_ID",
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Join engine output onto a page's frame. The only supported way in.

    Joins on the FPL element id. Name joins are how "Igor Thiago" spent a full
    90 minutes rendering as not-yet-played, and the engine has already done the
    one name match the app genuinely needs (Rotowire) behind a single matcher.

    Columns already present on ``df`` are overwritten, so calling this twice is
    safe and the caller always ends up with engine values rather than a stale
    hand-computed blend.
    """
    if df is None or df.empty:
        return df
    if projections is None or projections.empty:
        return df
    cols = columns or [c for c in CANONICAL_COLUMNS if c in projections.columns]
    if on not in df.columns:
        _logger.warning("attach_projections: %r missing from frame; nothing joined", on)
        return df

    out = df.copy()
    key = pd.to_numeric(out[on], errors="coerce")
    src = projections[cols]
    for col in cols:
        out[col] = key.map(src[col])
    return out
