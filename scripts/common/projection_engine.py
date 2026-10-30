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

from scripts.common.injury_helpers import games_to_miss_series
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

#: Gameweeks `Proj_Next3` covers, including the current one -- pinned live under
#: "Fantasy Football Pundit feed". The window a return date is measured against.
HORIZON_GWS = 3

#: Short labels for ``Proj_Src``. Kept terse because this renders in a table cell.
SOURCE_LABELS = {"rotowire": "RW", "ffp": "FFP", "fpl_ep": "xP", "odds": "ODDS"}

#: Default blend weights -- the fallback when ``config`` is unreadable, so it
#: must track ``config.PROJECTION_SOURCE_WEIGHTS``, where the measurement behind
#: these numbers lives. (They were 60/40 Rotowire/FFP from the app's beginning
#: until GW8, when five pre-deadline gameweeks made the split fittable.)
#: ``fpl_ep`` is wired but weighted 0 until the accuracy harness has an opinion:
#: it is carried through for display and for the snapshot either way.
#: Overridden by ``config.PROJECTION_SOURCE_WEIGHTS``.
DEFAULT_WEIGHTS = {"rotowire": 0.2, "ffp": 0.8, "fpl_ep": 0.0, "odds": 0.0}

#: A start-probability floor applied when *Rotowire* prices a player. Rotowire
#: only lists expected starters, so its presence is itself a confidence signal
#: and stops FFP's uncertainty from fully overriding an expert lineup call.
#:
#: These two dicts are the fallback used when ``config`` is unreadable, so they
#: must track it. The measurement behind each value lives at their definitions in
#: ``config.py``; both were fitted by ``projection_accuracy.fit_start_constants()``
#: replaying this module over the archive.
DEFAULT_START_FLOORS = {"G": 0.80, "D": 0.75, "M": 0.86, "F": 0.65}

#: The other half of the same signal: what a starters-only source's *silence*
#: about a player implies, when it covered his club. Never zero -- a player no
#: other source prices takes this as his whole start probability, and Rotowire's
#: silence is sometimes a name this app failed to match rather than a benching.
DEFAULT_OMITTED_STARTS = {"G": 0.02, "D": 0.07, "M": 0.02, "F": 0.05}

#: Players a starters-only source must price at a club before its silence about
#: one of them is evidence of anything.
DEFAULT_MIN_CLUB_COVERAGE = 5

#: Divisor floor when recovering a conditional value from an unconditional one.
START_RECOVERY_FLOOR = 0.05

#: Smallest start probability the basis conversion will divide by, i.e. the most
#: an expected-value source may be inflated when converted to "if he starts".
#:
#: The conversion assumes the source discounted its number by exactly this start
#: probability, and that assumption weakens as the probability falls. FPL's
#: ``ep`` is a model output, not ``chance_of_playing`` times something, so a
#: player rated 25% with ep 6.1 would recover to 24.4 points -- more than any
#: gameweek produces. Capping the divisor at 0.5 bounds the inflation at 2x,
#: which covers the range where the assumption is sound (50-100%) and stops
#: guessing below it. A player who is genuinely unlikely to start still gets a
#: low ``Proj``, because ``Start_Pct`` is applied separately and is uncapped.
BASIS_RECOVERY_FLOOR = 0.5


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


def _omitted_starts() -> Dict[str, float]:
    try:
        import config
        f = getattr(config, "ROTOWIRE_OMITTED_START", None)
        if isinstance(f, dict) and f:
            return dict(f)
    except Exception:                       # pragma: no cover
        pass
    return dict(DEFAULT_OMITTED_STARTS)


#: Fallback for ``config.OMITTED_START_HISTORY_WEIGHT`` -- see there for the
#: calibration table behind it. Must track config, for the reason the floors and
#: priors above must: the app and its own fallback disagreeing is worse than
#: either value.
DEFAULT_OMITTED_START_HISTORY_WEIGHT = 0.25


def _history_weight() -> float:
    try:
        import config
        w = getattr(config, "OMITTED_START_HISTORY_WEIGHT", None)
        if w is not None:
            return float(w)
    except Exception:                       # pragma: no cover
        pass
    return DEFAULT_OMITTED_START_HISTORY_WEIGHT


#: Fallback for ``config.ROTOWIRE_OMISSION_SHARE`` -- see there. Must track
#: config, like every other constant in this block.
DEFAULT_OMISSION_SHARE = 0.6


def _omission_share() -> float:
    try:
        import config
        v = getattr(config, "ROTOWIRE_OMISSION_SHARE", None)
        if v is not None:
            return float(v)
    except Exception:                       # pragma: no cover
        pass
    return DEFAULT_OMISSION_SHARE


def _min_club_coverage() -> int:
    try:
        import config
        n = getattr(config, "ROTOWIRE_MIN_CLUB_COVERAGE", None)
        if n:
            return int(n)
    except Exception:                       # pragma: no cover
        pass
    return DEFAULT_MIN_CLUB_COVERAGE


def _resolve_ids(source: SourceResult, pool: pd.DataFrame) -> pd.Series:
    """Map each of ``source.df``'s rows to a pool ``Player_ID``."""
    return resolve_player_ids(source.df, pool)


def resolve_player_ids(df: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    """Map each of ``df``'s rows to a pool ``Player_ID``.

    An integer id join is always preferred and is the only join FFP and the FPL
    bootstrap need -- FFP resolves 368/368 through the bootstrap ``code``. Only
    Rotowire publishes names alone, and it goes through ``ReferenceMatcher``
    rather than any bespoke matching: cross-source name matching is the single
    most frequent source of silent bugs in this app, and there is exactly one
    implementation of it on purpose.
    """
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
    deadlines: Optional[Sequence] = None,
    lineup_xi: Optional[Dict[str, object]] = None,
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
        lineup_xi: ``{source name: LineupXI}`` -- a fresher lineup opinion for a
            ``starters_only`` source, which then decides who it expects to start
            in place of who it priced. See ``_listed_mask``.

    Returns:
        A frame indexed by ``Player_ID`` carrying ``CANONICAL_COLUMNS`` plus one
        ``Proj_Start__<source>`` column per source -- the per-source values kept
        so the Projections Hub can show what went into the blend, and so the
        snapshot can record it for later accuracy scoring.
    """
    weights = dict(weights) if weights is not None else _weights()

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
    per_source_next3_basis: Dict[str, str] = {}

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
            # A multi-gameweek total is on whatever basis the source publishes
            # in, exactly as its single-gameweek number is. Declaring it here
            # keeps this entry point and `blend_projections_onto` converting
            # the same way -- they are asserted to agree.
            per_source_next3_basis[s.name] = s.basis

    return blend_aligned(
        index=out.index,
        per_source_raw=per_source_raw,
        per_source_basis=per_source_basis,
        per_source_startpct=per_source_startpct,
        per_source_next3=per_source_next3,
        per_source_next3_basis=per_source_next3_basis,
        starters_only={s.name for s in usable if s.covers == COVERS_STARTERS},
        source_club_coverage={
            s.name: {str(k): int(v) for k, v in s.df["Team"].value_counts().items()}
            for s in usable
            if s.covers == COVERS_STARTERS and "Team" in s.df.columns and not s.df.empty
        },
        positions=_pool_col(pool, "Position", out.index, default="M"),
        teams=_pool_col(pool, "Team", out.index),
        chance_of_playing=_pool_col(pool, "chance_of_playing_next_round", out.index),
        news=_pool_col(pool, "news", out.index),
        deadlines=deadlines,
        status=_pool_col(pool, "status", out.index),
        # Floors the omitted prior on the player's own start rate -- see
        # `config.OMITTED_START_HISTORY_WEIGHT`. A pool without the column
        # leaves that a no-op.
        starts=_pool_col(pool, "starts", out.index),
        weights=weights,
        gameweek=gameweek,
        starter_overrides={
            name: ov for name, ov in (
                (name, starter_override(
                    xi,
                    player_ids=pd.Series(out.index, index=out.index),
                    teams=_pool_col(pool, "Team", out.index),
                    index=out.index))
                for name, xi in (lineup_xi or {}).items()
            ) if ov is not None
        },
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


#: How the start model resolves a probability. **Bump this whenever a change to
#: that resolution would make an older snapshot unreplayable** -- not for a
#: constant retune, which `projection_accuracy._snapshot_constants` already
#: recovers from the snapshot itself.
#:
#: The accuracy harness refuses to fit constants on a snapshot it cannot
#: reproduce, which is right, and until this existed it could only report the
#: refusal as a numeric deviation -- indistinguishable from the archive and the
#: engine having silently parted company. `projection_accuracy` names the
#: revision instead, and the snapshot collector stamps it.
#:
#: 1. The original model: FFP, then FPL's `chance_of_playing`, then 1.0, with
#:    Rotowire's positional floor clipping upward and its omission blending down.
#: 2. A bare `chance_of_playing = 100` stops counting as an opinion -- neither
#:    resolving `Start_Pct` nor standing as the "other opinion" the omission
#:    penalty blends against -- and a Rotowire-listed player no start model
#:    priced takes the positional floor as his *value* rather than as a lower
#:    bound on a fabricated 1.0.
#: 3. "Rotowire expects him to start" is read from Rotowire's projected-lineups
#:    page rather than from its weekly rankings article, per club and only where
#:    the whole XI resolves. The article keeps the points; it is no longer the
#:    lineup call. See `_listed_mask` and `rotowire_lineups.lineup_starters`.
#:
#:    **This one cannot be replayed from the archive at all**, which the two
#:    before it could: the snapshots record `Start_Pct__rotowire`, so a constant
#:    retune is recoverable from them, but they do not record *which* players the
#:    lineups page had in the XI. So revision 2 snapshots are refused as such
#:    rather than silently replayed against a different signal, and the start
#:    constants cannot be refitted until snapshots accrue under this revision.
#: 4. Two things, both about *benching*, which is the direction that inverts a
#:    projection rather than degrading it:
#:
#:    * More than one publication's XI is read, and they are combined by
#:      `combine_starter_overrides`: any source naming a player promotes him,
#:      and he is benched only where every source with an opinion leaves him
#:      out. Disagreement can no longer lower anybody.
#:    * The omission prior can no longer ignore a player's own start rate --
#:      see `config.OMITTED_START_HISTORY_WEIGHT`.
#:
#:    Unreplayable for the same reason as revision 3, and now doubly so: which
#:    players *each* page had in its XI is recorded only from this revision on,
#:    as the `Start_XI__<source>` columns.
START_MODEL_REVISION = 4

#: Sources that may resolve a player's start probability, in precedence order.
START_MODEL_SOURCES = ("ffp", "fpl_ep", "odds")

#: Of those, the ones whose "start percentage" is FPL's own availability figure
#: rather than a selection model. **A 1.0 from one of these is the absence of a
#: doubt, not a statement that he starts**, so it is not treated as an opinion
#: -- see the start-probability block in `blend_aligned`.
AVAILABILITY_SOURCES = ("fpl_ep",)


def _start_opinion(name: str, sp: pd.Series, index) -> pd.Series:
    """A source's start percentage, with non-opinions dropped.

    FPL publishes `chance_of_playing = 100` for every fit player, bench
    included, so that value carries no information about selection. Masking it
    here is what keeps it from outranking a real opinion while leaving every
    *stated doubt* exactly where it was -- which is the half the basis
    recovery, the omission blend and the ceiling all depend on.
    """
    sp = pd.to_numeric(sp.reindex(index), errors="coerce")
    return sp.where(sp.lt(1.0)) if name in AVAILABILITY_SOURCES else sp


def _priced_mask(raw: pd.Series, index) -> pd.Series:
    """True where this source published a usable value for the player.

    One definition, because it is read twice -- once to fill the start
    probability from a starters-only source's positional floor, once to clip
    it -- and those two must agree about who the source listed.
    """
    return raw.reindex(index).gt(0).fillna(False)


def starter_override(xi, *, player_ids: pd.Series, teams: Optional[pd.Series],
                     index) -> Optional[pd.Series]:
    """A tri-state "does this source expect him to start" Series from an XI.

    ``True`` in the XI, ``False`` at a club whose XI resolved completely but
    without him, and ``NaN`` -- no opinion -- everywhere else, which
    ``_listed_mask`` falls back from. The three states are the whole point: a
    club absent from the lineups page must not read as eleven benchings, so "no
    opinion" has to be representable and has to be the default.

    **The True case needs no club coverage and the False case does**, which is
    why the two are assigned separately rather than masked from one comparison.
    Promoting a resolved starter is safe wherever he was found; benching the
    players an XI leaves out is only safe once the whole XI resolved, because an
    unmatched starter is indistinguishable from a benched one. See
    ``rotowire_lineups.lineup_starters``.

    ``xi`` is duck-typed on ``.player_ids`` and ``.clubs`` so the engine does
    not import the scraper; ``rotowire_lineups.LineupXI`` is what the app
    passes. ``player_ids`` is aligned to ``index`` rather than read off it,
    because only ``build_projections`` indexes by element id -- the page-facing
    entry point blends frames with an index of their own.
    """
    if xi is None or not getattr(xi, "player_ids", None):
        return None
    ids = pd.to_numeric(pd.Series(player_ids).reindex(index), errors="coerce")
    in_xi = ids.isin(set(xi.player_ids))

    benched = pd.Series(False, index=index)
    if teams is not None and getattr(xi, "clubs", None):
        club = teams.reindex(index)
        if not club.isna().all():
            benched = club.astype(str).isin(set(xi.clubs)) & ~in_xi

    override = pd.Series(np.nan, index=index, dtype="float64")
    override[benched] = 0.0
    override[in_xi] = 1.0
    if override.isna().all():
        return None
    return override


def combine_starter_overrides(
    overrides: Optional[Dict[str, pd.Series]],
    index=None,
) -> Optional[pd.Series]:
    """Several publications' XIs as one tri-state opinion. **Unanimous to bench.**

    Any source naming a player promotes him; he is benched only where every
    source that has an opinion about him leaves him out, and at least one does.
    Everything else is NaN -- no opinion -- which falls back to who the points
    table priced.

    **So disagreement can never lower anybody.** That is the existing asymmetry
    of ``starter_override`` extended across sources, for the same reason:
    promoting a player is a claim about him alone and can only lift him towards
    a positional floor, while benching him is a claim about everyone an XI
    leaves out and *inverts* his projection. Reported from the app at GW8 --
    three published XIs agreed on only eight of Manchester City's eleven, and
    Phil Foden, named by two of the three, read 0.21 because the third was
    silent.

    **Combining here rather than applying each source in turn is deliberate.**
    The omission blend below is sequential: it lowers ``start_pct`` once per
    starters-only source, and a second source's blend is then computed against
    the already-lowered value, so two benchings would compound. Worse, it
    iterates a ``set``, so *which* compounded first would vary by hash order.
    Resolving one combined opinion keeps exactly one omission pass, which also
    sidesteps the omission share being drawn from the **points** weights (a
    start-only source would have weight 0 and so no influence at all) and
    ``has_other`` not counting other lineup sources.

    Returns None when nothing is stated anywhere, so a caller that resolved no
    XI at all is indistinguishable from one that never asked.
    """
    series = [s for s in (overrides or {}).values() if s is not None]
    if not series:
        return None
    if index is None:
        index = series[0].index

    stated = [pd.to_numeric(s.reindex(index), errors="coerce") for s in series]
    starts = pd.Series(False, index=index)
    benches = pd.Series(False, index=index)
    for s in stated:
        starts |= s.gt(0).fillna(False)
        benches |= s.le(0).fillna(False)

    combined = pd.Series(np.nan, index=index, dtype="float64")
    # Benched first, then promoted, so "any source starts him" wins outright --
    # which makes the result independent of the order the sources arrived in.
    combined[benches] = 0.0
    combined[starts] = 1.0
    if combined.isna().all():
        return None
    return combined


def _listed_mask(name: str, raw: pd.Series, index,
                 override: Optional[pd.Series]) -> pd.Series:
    """Who a starters-only source **expects to start**, which is not always who
    it priced.

    For a year these were the same question, because Rotowire's weekly rankings
    *article* was the only lineup opinion the engine had: a priced player was a
    projected starter, and the floor and the omission penalty both read
    ``_priced_mask``.

    Rotowire publishes a second table, though -- the projected-lineups page,
    whose own description says it is "updated as team news breaks" -- and where
    the two disagree the article is always the stale one. Measured 2026-10-21
    for GW8, with the article stamped the previous morning: 3 of 220 differed,
    and one was Phil Foden, benched on the lineups page, still in the article's
    XI, rated **20%** to start by FFP, and rendered by this engine at **86%**
    because presence in the article clips *up* to the positional floor. The
    app's own Projected Lineups page had him on the bench at the same moment.

    So an override states, per player, what the fresher table says: True he
    starts, False he does not, and **NaN no opinion** -- which falls back to
    ``_priced_mask`` and the behaviour that shipped for a year. The caller
    supplies it as a Series because only it can resolve lineup names to element
    ids; see ``rotowire_lineups.lineup_starters`` for the club-by-club gate that
    decides where an opinion exists at all.

    ``override`` is **one** Series, already combined across every publication by
    ``combine_starter_overrides``, and is applied to every starters-only source
    rather than looked up per source. It is a statement about selection, not
    about who this particular points table priced, so there is nothing for it to
    be keyed on -- and keying it per source made the omission penalty compound
    in set-iteration order once there was more than one XI.
    """
    priced = _priced_mask(raw, index)
    if override is None:
        return priced
    stated = pd.to_numeric(override.reindex(index), errors="coerce")
    known = stated.notna()
    if not known.any():
        return priced
    return (priced & ~known) | (known & stated.fillna(0).gt(0))


def blend_aligned(
    *,
    index,
    per_source_raw: Dict[str, pd.Series],
    per_source_basis: Dict[str, str],
    per_source_startpct: Optional[Dict[str, pd.Series]] = None,
    per_source_next3: Optional[Dict[str, pd.Series]] = None,
    per_source_next3_basis: Optional[Dict[str, str]] = None,
    starters_only: Optional[set] = None,
    positions: Optional[pd.Series] = None,
    teams: Optional[pd.Series] = None,
    chance_of_playing: Optional[pd.Series] = None,
    status: Optional[pd.Series] = None,
    news: Optional[pd.Series] = None,
    starts: Optional[pd.Series] = None,
    deadlines: Optional[Sequence] = None,
    weights: Optional[Dict[str, float]] = None,
    start_floors: Optional[Dict[str, float]] = None,
    omitted_starts: Optional[Dict[str, float]] = None,
    fallback_names: Optional[Sequence[str]] = None,
    gameweek: Optional[int] = None,
    source_club_coverage: Optional[Dict[str, Dict[str, int]]] = None,
    starter_overrides: Optional[Dict[str, pd.Series]] = None,
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
    # Overridable for the same reason `weights` is: the accuracy harness fits
    # these constants by replaying this function over archived snapshots with
    # candidate values. Fitting a separate model *of* the engine rather than the
    # engine itself is how two implementations of one blend come to disagree.
    floors = dict(start_floors) if start_floors is not None else _start_floors()
    source_club_coverage = dict(source_club_coverage or {})
    starter_overrides = dict(starter_overrides or {})
    # One combined opinion, resolved once -- see `combine_starter_overrides`
    # for why this is not applied per source.
    combined_override = combine_starter_overrides(starter_overrides, index)
    per_source_startpct = dict(per_source_startpct or {})
    per_source_next3 = dict(per_source_next3 or {})
    per_source_next3_basis = dict(per_source_next3_basis or {})
    starters_only = set(starters_only or ())

    out = extra if extra is not None else pd.DataFrame(index=index)
    if positions is None:
        positions = pd.Series("M", index=index)
    positions = positions.reindex(index)

    # --- Start probability --------------------------------------------------
    # The sources that publish a real start model first -- FFP above all, being
    # the only continuous 0-100 one -- then a starters-only source's listing,
    # then "no news means he plays".
    #
    # **FPL's `chance_of_playing = 100` is not an opinion about starting.** The
    # ceiling block further down sets out the measurement: as a *predictor* of
    # starting FPL's chance is dreadful (Brier 0.486, bias +0.49 where it
    # disagrees with FFP) precisely because it reads 100 for every fit bench
    # player -- it measures availability, not selection. Yet a bare 100 was
    # filling this chain and outranking every real signal, which is the
    # reported bug: **Phil Foden at 100% to start**, from FFP not pricing him
    # and FPL having no news about him.
    #
    # `_start_opinion` drops those, and *only* those. Every stated doubt is
    # still consumed here, which matters in three places that would otherwise
    # change meaning: an unconditional source's basis recovery divides by the
    # resolved value, the omission blend needs a real opinion to blend against,
    # and the round trip `Proj == Proj_Start x Start_Pct` has to return a
    # single-source number unchanged.
    #
    # Note `fpl_ep`'s declared start percentage *is* `chance_of_playing` -- it
    # must declare one, being unconditional (rule 1 of the Joao Pedro fix) --
    # so the same mask has to apply there or availability walks back in through
    # the front door. That is why this is keyed on the source, not on which
    # argument the number arrived in.
    start_pct = pd.Series(np.nan, index=index, dtype="float64")
    for name in START_MODEL_SOURCES:
        if name in per_source_startpct:
            start_pct = start_pct.fillna(
                _start_opinion(name, per_source_startpct[name], index))
    if chance_of_playing is not None:
        chance = pd.to_numeric(chance_of_playing.reindex(index), errors="coerce") / 100.0
        start_pct = start_pct.fillna(_start_opinion("fpl_ep", chance, index))

    # Where no start model priced the player at all, a starters-only source
    # *listing* him is the only opinion there is, and the positional floor is
    # that opinion's fitted value -- so it fills the gap rather than merely
    # bounding the 1.0 below.
    #
    # The clip further down cannot do this job. `clip(lower=floor)` on a value
    # already filled to 1.0 is a no-op, so the engine was recording its own
    # calibrated decision in `Start_Pct__rotowire` and then rendering a number
    # that contradicted it. Reported from the app: **Phil Foden at 100%**, on
    # Rotowire alone, with `Start_Pct__rotowire` reading 0.86.
    #
    # Measured live at GW7, that was the whole cohort above 0.95: 12 of 667
    # players, every one Rotowire-listed and priced by nothing else, against a
    # 75th percentile of 0.90 among the 219 listed players. FFP's model never
    # rates anybody above 95%, so each of those 100%s was this fallback rather
    # than any source's confidence -- the same silent degradation recorded
    # under "FFP stops publishing `start_pct` for the gameweek it is on", where
    # 208 of 220 players read 100% for the same structural reason.
    #
    # It is a fill, never an override: a player FFP or FPL *did* price keeps
    # that value and is then clipped by the floor exactly as before, which is
    # the calibration the bias measurements rest on.
    #
    # **And it is gated on a start model existing at all** -- the same shape of
    # gate as the omission penalty's club-coverage test, for the same reason. A
    # floor is calibrated as a *lower bound* on a blend, not as a point
    # estimate, and the four positions' floors differ by up to 1.32x. Where
    # some model published and simply missed this player, filling the gap with
    # his floor is the best available answer. Where *nothing* published for
    # anybody -- a dead feed, or the preseason pool the Initial Squad Optimizer
    # builds, which has a Rotowire article and no FFP at all -- it would
    # instead reprice every position against every other out of numbers that
    # were never meant to carry that, and that page's whole objective is
    # expected points per pound. So the gate fails open to the old 1.0.
    has_start_model = any(
        name not in AVAILABILITY_SOURCES
        and _start_opinion(name, sp, index).notna().any()
        for name, sp in per_source_startpct.items())
    if has_start_model:
        listed_floor = pd.Series(np.nan, index=index, dtype="float64")
        for name in starters_only:
            if name not in per_source_raw:
                continue
            listed = _listed_mask(name, per_source_raw[name], index,
                                  combined_override)
            listed_floor = listed_floor.fillna(
                positions.map(floors).astype("float64").where(listed))
        start_pct = start_pct.fillna(listed_floor)

    # "No news means he plays" -- the last resort, and not a start model. It now
    # only reaches players no source listed or priced, whose projection the
    # engine zeroes anyway.
    start_pct = start_pct.fillna(1.0).clip(0, 1)

    # Rotowire's presence as a confidence signal: it lists only expected
    # starters, so a player it prices gets a positional floor on start
    # probability. This is why `covers` is part of the source contract.
    #
    # ...and its *absence* is the same call in the other direction. Rotowire
    # publishes ~11 players per club, so a player missing from a club it covered
    # is being predicted not to start; the blend simply renormalised its weight
    # away and let FFP's start probability stand alone. Measured on the GW3
    # snapshot: listed players started 90.5% of the time, omitted ones 4.2% --
    # and of the omitted players this engine gave >=80% start probability, 0 of
    # 16 started (Joe Gomez, Dan Burn, three backup keepers).
    #
    # Two asymmetries here are deliberate and measured, not oversights:
    #
    #   * Presence CLIPS, absence BLENDS. Re-deriving the listed side as a blend
    #     too was tried and is worse -- bias on the 220 listed players moves from
    #     +0.058 to -0.314, because the floors are already well calibrated
    #     against a 90.5% observed start rate. They are left untouched here. Absence blends because a cap
    #     flattens an FFP-90% player and an FFP-20% player onto one number,
    #     discarding the only opinion left about which of them might play.
    #   * The blend can only ever LOWER. An omission is never evidence that a
    #     player *will* start, so a player FFP already rates below the implied
    #     value keeps FFP's number.
    omitted_starts = (dict(omitted_starts) if omitted_starts is not None
                      else _omitted_starts())
    # The start probability as the *sources* see it, before any omission penalty
    # is folded in. An unconditional source with no stated basis of its own is
    # recovered against this, not against the penalised value -- see the basis
    # conversion below for why the difference is load-bearing.
    start_pct_stated = start_pct.copy()

    min_coverage = _min_club_coverage()
    for name in starters_only:
        if name not in per_source_raw:
            continue
        listed = _listed_mask(name, per_source_raw[name], index,
                              combined_override)
        for pos_code, floor_val in floors.items():
            mask = listed & (positions == pos_code)
            start_pct[mask] = start_pct[mask].clip(lower=floor_val)

        # Club coverage, not the fixture list: a club this source listed nobody
        # at is blank, unpublished, or a wholesale matching failure, and none of
        # those is a lineup call. Without team labels there is no way to tell,
        # so the penalty is simply not applied -- fail open, like the Projected
        # Lineups gameweek filter.
        if teams is None:
            continue
        club = teams.reindex(index)
        # `_pool_col` returns an all-None Series rather than None when the pool
        # has no Team column, so "is not None" is not the test. Grouping those
        # on str(None) would put every player in one 20-club bucket, which is
        # trivially "covered" -- the penalty would then fire on exactly the
        # frames that carry no evidence for it.
        if club.isna().all():
            continue
        known_club = club.notna()
        # Coverage is a property of the SOURCE's table, not of the frame being
        # blended. Counting within the frame works only when the frame is the
        # whole player pool: a 15-player Classic squad holds two or three
        # players per club, so the threshold can never be met and the penalty
        # silently does nothing on every per-squad page -- which is exactly how
        # this shipped looking correct, having been measured against a 652-row
        # pool. `source_club_coverage` carries the real counts; the in-frame
        # count remains the fallback, and being unreachable it fails open.
        published = source_club_coverage.get(name)
        if published:
            club_count = club.astype(str).map(published).fillna(0)
        else:
            club_count = listed.groupby(club.astype(str)).transform("sum")
        covered = club_count.ge(min_coverage) & known_club
        implied = positions.map(omitted_starts).astype("float64")

        # **The prior is right for the cohort it was fitted on and wrong for
        # everybody else.** It was calibrated on players who started 0 of 16
        # times. Measured over every pre-deadline snapshot in the archive --
        # 1,114 rows sitting on it, bucketed by each player's own start rate
        # before that gameweek -- the engine said 0.044 against an observed
        # 0.031 for the never-started, and then under-predicted by +0.079,
        # +0.071 and +0.111 for the 25-50%, 50-75% and 75-100% buckets.
        # Monotone. Reported from the app as Rayan Cherki reading 21% to start
        # off five starts in eight gameweeks.
        #
        # A player every XI leaves out who has nonetheless been starting is not
        # a never-starter, so his own rate floors the prior. It can only ever
        # *raise* `implied`, and the `np.minimum` below still keeps the penalty
        # from raising `start_pct` -- so a benched player can only be pulled
        # down, just not as far. No `starts` supplied leaves this a no-op, which
        # is what keeps every existing caller byte-identical.
        if starts is not None and gameweek:
            played = max(1, int(gameweek) - 1)
            rate = (pd.to_numeric(starts.reindex(index), errors="coerce")
                    / float(played)).clip(0.0, 1.0)
            # A mid-season signing is understated by this denominator (three
            # starts in three games reads 0.43, not 1.0). That is the safe
            # direction, since the floor can only raise.
            implied = np.maximum(implied, _history_weight() * rate.fillna(0.0))

        target = covered & ~listed & implied.notna()
        if not target.any():
            continue

        # How much of the implied value survives the blend. **Its own constant,
        # not this source's points weight.**
        #
        # It was `w_self / (w_self + w_rest)`, which made the strength of a
        # *start* inference a function of how much the *points* blend trusts the
        # same source -- so retuning the points weights silently retuned the
        # start model. Measured on a player every published XI left out, with
        # FFP rating him 50%: moving Rotowire's points weight 0.60 -> 0.25 ->
        # 0.00 moved his resolved Start_Pct 0.212 -> 0.380 -> 0.500. At zero the
        # omission signal disappears and a benched player reads as though
        # nothing had been said about him -- which is how a points fit would
        # have undone the prior that was fitted against this very term.
        #
        # One constant rather than a per-source mapping because there is one
        # starters-only source. A second would need one.
        share = _omission_share()

        # "Another opinion" is judged by the same rule as the resolution above:
        # a bare `chance_of_playing = 100` is not one. Counting it left the
        # blend averaging the fitted implied value against a 1.0 that nobody
        # had asserted -- live that moved seven Rotowire-omitted players *up*,
        # including Joe Gomez and Dan Burn, two of the players this constant
        # was fitted on (0 of 16 such players started).
        has_other = pd.Series(False, index=index)
        for other, sp in per_source_startpct.items():
            if other != name:
                has_other |= _start_opinion(other, sp, index).notna()
        if chance_of_playing is not None:
            has_other |= _start_opinion(
                "fpl_ep",
                pd.to_numeric(chance_of_playing.reindex(index),
                              errors="coerce") / 100.0,
                index).notna()
        eff_share = pd.Series(share, index=index).where(has_other, 1.0)

        blended = eff_share * implied + (1.0 - eff_share) * start_pct
        start_pct = start_pct.where(~target, np.minimum(start_pct, blended))

        # Record what this source implied, so the accuracy harness can score the
        # decision rather than only its effect.
        out[f"Start_Pct__{name}"] = implied.where(target).fillna(
            positions.map(floors).astype("float64").where(listed))

    # Record each publication's *raw* XI opinion, separately from the combined
    # one that was applied. `Start_Pct__<points source>` above records the
    # engine's decision; these record who said what, which is the only thing
    # that makes a two-source signal scoreable after the fact -- and the gap
    # revision 3 left, where which players a lineups page had in its XI was
    # recorded nowhere and so no snapshot could be replayed against it.
    for _src, _ov in starter_overrides.items():
        if _ov is not None:
            out[f"Start_XI__{_src}"] = pd.to_numeric(
                _ov.reindex(index), errors="coerce")

    # --- FPL's availability is a ceiling, never a source ---------------------
    #
    # **A source listing a player cannot outvote FPL saying he is injured.**
    # The positional floor above clips a listed player's start probability *up*,
    # and it did so unconditionally -- so a player FPL rates 0% to play, with a
    # stated return date, came out at the floor and projected like a starter.
    # Live at GW6: Nikola Milenkovic 0.75 / 3.14 points ("Hamstring injury -
    # Expected back 11 Oct") and Dean Henderson 0.80 / 3.06 ("Foot injury -
    # Expected back 11 Oct"), both from the floor; Amar Dedic 0.90 / 3.31 from
    # FFP publishing a stale 90% start.
    #
    # The engine already knew: `unavailable` is computed below and zeroes the
    # *unpriced*. So it trusted FPL exactly where the projection was small and
    # ignored it where the projection was large -- the wrong way round, since a
    # priced player is the one who reaches the top of a board.
    #
    # **A ceiling, and never a source, because that is what the numbers say it
    # is.** Scored against actuals over GW4-GW5: as a *predictor* of starting,
    # FPL's chance is dreadful -- Brier 0.486 and bias +0.49 on the rows where
    # it disagrees with FFP -- because it reads 100 for every fit bench player.
    # It is measuring availability, not selection. As a ceiling it is flawless:
    # of the 350 rows where it said 0%, **0 started**; of the 9 where it merely
    # sat below the engine's resolved value, **0 started**. It only ever fires
    # where FPL has published news, which is why it touched 9 of 1315 rows.
    #
    # The aggregate effect is therefore noise-sized (Brier 0.0678 -> 0.0673) and
    # that is not the reason for it. The reason is that the alternative states,
    # confidently and in the app's most-read column, that a man with a hamstring
    # tear is a 3.1-point starter.
    #
    # Applied to the resolved value only, never to `start_pct_stated`: that is
    # the divisor for recovering an unconditional source's conditional basis,
    # and dividing one source's number by another's pessimism is the bug
    # recorded above under Joao Pedro.
    # The ceiling below is a statement about *this* gameweek. The multi-gameweek
    # horizon needs his ordinary start probability -- applying a one-week absence
    # uniformly across three weeks is the same error as FFP publishing a single
    # `start_pct` for its whole window. His absence reaches the horizon through
    # the stated-duration cap further down, which is the part that knows how many
    # of those weeks he actually misses.
    start_pct_window = start_pct.copy()

    if status is not None:
        out_of_squad = status.reindex(index).isin(["i", "s", "u"]).fillna(False)
        start_pct = start_pct.where(~out_of_squad, 0.0)
    if chance_of_playing is not None:
        ceiling = pd.to_numeric(
            chance_of_playing.reindex(index), errors="coerce") / 100.0
        start_pct = start_pct.where(ceiling.isna(),
                                    np.minimum(start_pct, ceiling.clip(0, 1)))

    out["Start_Pct"] = start_pct

    # --- Basis conversion ---------------------------------------------------
    # Every source is put on the conditional basis before blending. An
    # expected-value source averaged straight against if-he-starts sources
    # drags the blend down by exactly the start probability -- the same shape as
    # the double discount that ran the FFP term ~44% low.
    #
    # A source carrying its own start probability is un-discounted by that. One
    # that does not falls back to ``start_pct_stated`` -- the resolved value
    # *before* the omission penalty, never after it.
    #
    # That distinction is the whole of this block. The resolved value is the
    # right basis when it reflects what the sources actually said about the
    # player: where an expected-value source and an if-he-starts source describe
    # the same man, dividing by their shared start probability is exactly what
    # makes them commensurable, and using 1.0 instead understates the blend.
    #
    # The omission penalty is a different animal. It is an inference drawn from
    # a source's *silence*, and folding it in here divides one source's number
    # by another source's pessimism. Live on 2026-09-17: Joao Pedro, whom FPL
    # rated 75% to play and Rotowire omitted, resolved to 33%, so FPL's 6.1
    # expected points became 6.1 / 0.33 = 18.5 "if he starts" -- more than any
    # single gameweek can produce. Yann Gboho, with no stated doubt at all,
    # reached 16.7 the same way.
    #
    # Nothing caught it because the division is undone by the multiplication
    # that follows: Proj = Proj_Start x Start_Pct held exactly, both halves
    # wrong together, so every internal invariant passed. Only Proj_Start was
    # visibly absurd -- and team_strength percentiles Proj_Start, not Proj.
    per_source_start: Dict[str, pd.Series] = {}
    for name, v in per_source_raw.items():
        v = v.reindex(index)
        if per_source_basis.get(name) == BASIS_UNCONDITIONAL:
            sp = per_source_startpct.get(name)
            sp = (start_pct_stated if sp is None
                  else sp.reindex(index).fillna(start_pct_stated))
            v = v / sp.clip(lower=BASIS_RECOVERY_FLOOR)
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

    # --- Unpriced: not expected to start, or genuinely unknown? -------------
    # These need different answers and used to get the same one.
    #
    # A player no source priced is almost always a squad player nobody expects
    # to start -- Rotowire lists 20 clubs x 11, so absence from it *is* the
    # "not starting" signal. Scoring that as unknown hands him a neutral 0.50
    # on the 1GW percentile, which ranks him above players who are projected to
    # play but carry a doubt. Measured on GW4: 120 players sat on exactly 0.50,
    # and every one of the 20 clubs had a fixture, so not one of them was blank.
    #
    # The case the old rule was written for is real but rarer: a genuinely blank
    # gameweek, where scoring an elite asset as 0 reads as "drop him". The two
    # are told apart by whether the player's *club* was priced at all. A club
    # with a fixture has 20+ priced players in a healthy feed, so an unpriced
    # player there is a non-starter. A club with none is either blank or missing
    # from the feeds, and "unknown" is then the honest answer.
    #
    # Keying on club coverage rather than the fixture list also makes this
    # degrade correctly when a source is down: if the feeds carry nothing for
    # anyone, nobody is zeroed on the strength of a feed that isn't there.
    unpriced = out["Proj_Start"].isna()
    if teams is not None:
        club = teams.reindex(index).astype(str)
        club_priced = (~unpriced).groupby(club).transform("sum")
        club_known = club_priced.gt(0)
    else:
        # Without club information, fall back to whether anything was priced at
        # all -- coarse, but it still separates "a source is down" from
        # "this player is not in the lineup".
        club_known = pd.Series(bool((~unpriced).any()), index=index)

    unavailable = pd.Series(False, index=index)
    if status is not None:
        unavailable |= status.reindex(index).isin(["i", "s", "u"]).fillna(False)
    if chance_of_playing is not None:
        c = pd.to_numeric(chance_of_playing.reindex(index), errors="coerce")
        unavailable |= (c.notna() & c.lt(50))

    out.loc[unpriced & (club_known | unavailable), ["Proj", "Proj_Start"]] = 0.0

    # --- Multi-gameweek -----------------------------------------------------
    #
    # A multi-gameweek total has a basis exactly as a single-gameweek one does,
    # and it must match `Proj`'s -- the two are read side by side, and any
    # consumer dividing `Proj_Next3` by 3 to get a rate is comparing it against
    # `Proj` directly.
    #
    # FFP's `Next3GWs` is already start-adjusted, so it is unconditional and
    # needs nothing. The fallbacks are not: `blend_multi_gw_projections` fills
    # unmatched players with Rotowire's `Projected_Points x 3`, which is
    # "points if he starts", or `points_per_game x 3`, which averages only the
    # matches he actually featured in. Passed through undiscounted those read
    # as expected value, so a player projected 0.36 points this week carried a
    # 6.00/gameweek horizon rate -- a 16.7x inflation, and systematically
    # biased toward exactly the fringe players who should rank lowest.
    #
    # Converting *down* is multiplication, so unlike the conditional recovery
    # above it cannot explode and needs no floor.
    next3 = pd.Series(np.nan, index=index, dtype="float64")
    supplied = False
    for name in ("ffp", "rotowire", "fpl_ep"):
        if name not in per_source_next3:
            continue
        supplied = True
        values = per_source_next3[name].reindex(index)
        if per_source_next3_basis.get(name, BASIS_UNCONDITIONAL) == BASIS_CONDITIONAL:
            values = values * start_pct_window
        next3 = next3.fillna(values)

    # A player this blend has judged a non-starter has a horizon of zero, not
    # an unknown one -- the same call `Proj` and `Proj_Start` just made, for the
    # same reason. Left NaN he takes the neutral 0.50 that every percentile
    # fills with, which ranks him at the median of his position: measured live
    # at GW6, 289 of the 295 players with no horizon were ones the engine had
    # already scored 0. NaN survives only where the club itself is unpriced,
    # which is the honest "we cannot tell".
    next3 = next3.mask(unpriced & (club_known | unavailable), 0.0)

    # Only write the column when this call actually had something to say. A
    # caller who blended a horizon earlier and then asks for a single-gameweek
    # blend must not have it erased: `compute_player_scores` did exactly that to
    # the Classic planner's squad frame, so the legs it proposed selling were
    # priced over one gameweek while the legs it proposed buying were priced
    # over three.
    if supplied or "Proj_Next3" not in out.columns:
        out["Proj_Next3"] = next3

    # --- A return date is information about the window, not just this week ----
    #
    # The availability ceiling above zeroes `Proj`, which is right: he is not
    # playing this gameweek. It says nothing about the other two weeks in the
    # horizon, and the three paths into `Proj_Next3` then disagreed about him.
    # FFP publishes **one** `start_pct` per player across all six forecast weeks
    # (see "Fantasy Football Pundit feed"), so its horizon cannot model a return
    # either -- it is uniformly optimistic or uniformly pessimistic. The
    # conditional fallbacks are multiplied by the ceilinged `start_pct` and so
    # collapse to zero for the whole window, which writes off a player due back
    # next week. And a player no source priced carries NaN, which every
    # percentile fills with a neutral 0.50 -- ranking a man with a hamstring tear
    # at the median of his position.
    #
    # FPL's news carries the answer for the cases that matter: "Expected back 11
    # Oct" is three gameweeks at GW6. `estimate_games_to_miss` already parses it
    # -- for the Draft waiver hold logic and the Power Rankings injury discount --
    # and it falls back through suspension lengths, chance buckets and the status
    # code, so a player with no date still gets an estimate rather than nothing.
    #
    # **A cap, for the same reason the availability signal is a cap.** The bound
    # is `Proj_Start x weeks he is available`, which assumes he starts every week
    # he is fit -- generous on purpose, so it binds only where a source is
    # claiming more than the absence allows. For a fit player it is
    # `3 x Proj_Start`, above any honest horizon, and it is skipped outright
    # where nothing is flagged so a double gameweek is never clipped.
    #
    # Where the horizon is missing entirely the cap *becomes* the value, since
    # "out for the window" is a real number and NaN is not.
    if HORIZON_GWS > 0:
        gws_out = games_to_miss_series(news, chance_of_playing, status, index,
                                       deadlines=deadlines)
        flagged = gws_out.gt(0)
        if flagged.any():
            weeks_available = (HORIZON_GWS - gws_out).clip(lower=0, upper=HORIZON_GWS)
            cap = pd.to_numeric(out["Proj_Start"], errors="coerce") * weeks_available
            current = pd.to_numeric(out["Proj_Next3"], errors="coerce")
            capped = np.minimum(current.fillna(cap), cap)
            out["Proj_Next3"] = current.where(~flagged, capped)

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
