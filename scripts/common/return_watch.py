"""
Returning-elite watch — the player who is about to come back, before a rival takes him.

**The app could not see these players at all.** Measured live at GW10: William
Saliba, out all season, read 0 minutes, 0 starts, 0 points, ``ep_next`` 0.0, no
form and no projection, and he was *absent from Rotowire's weekly table* -- so
he was absent from the Waiver Wire's available pool too, which is built from it.
He was not ranked low; there was nothing to rank. Meanwhile FPL had already
cleared him (``status`` 'a', chance 100, ``news`` ""), so no injury row existed
to flag and no change-detector would ever have fired: his news had been empty
since July. A rival claimed him at waiver priority 1.

In Draft that is terminal -- a claim has to be made before the player plays. In
Classic you can wait for certainty and buy him then, which is why this is
Draft-only.

What the detector asks is the question none of the app's scores can express:
**who is not playing now, but was a productive regular in the last season that
described him, and can still be claimed?**

Pedigree is a **rate**; whether he walks back into the XI is a separate **gate**.
Multiplying the two was tried and is wrong -- this season's start share is over
elapsed gameweeks and a prior season's is over all 38, and mixing those
denominators buried the prior rung so thoroughly it took the live board to zero
rows. Gating instead prunes exactly the noise a bare rate admits: Tomiyasu's
7.00 points per start off 10 starts outranked every genuine elite defender until
the role gate removed him. After it, the top of the defender list reads Gabriel,
Guehi, Van Dijk, Senesi, Tarkowski.

This module computes and never renders: no ``st.*``, no ``st.stop()``, no
network. It is **not** Streamlit-free, because it reuses
``analytics._referenced_percentile`` and ``display_names.attach_display_names``
rather than growing private copies of them -- and a second implementation of a
percentile or a display-name resolver is the twin-drift failure this codebase
keeps paying for. The hard Streamlit-free constraint covers
``projection_engine``, ``projection_sources`` and ``name_matching``, which the
Actions collector imports; this module is not in that set.
"""

import logging

import numpy as np
import pandas as pd

import config
from scripts.common.analytics import _referenced_percentile
from scripts.common.display_names import attach_display_names
from scripts.common.injury_helpers import (
    ABSENCE_FROM_NEWS,
    OUT_OF_SQUAD_STATUSES,
    absence_estimate,
    is_departed,
    news_stated_games_to_miss,
)
from scripts.common.player_history import SEASON_GWS
from scripts.common.text_helpers import POS_MAP_TO_RW

_logger = logging.getLogger("fpl_app.return_watch")

#: Which rung produced a player's pedigree rate. Ordered best-first: a rate from
#: the season in progress beats one from last season, which beats a preseason
#: forecast. All three are **points per start**, which is what makes one shared
#: reference pool legitimate -- unlike ``analytics._start_consistency``, whose
#: rungs are in different units and so needs a per-rung reference.
RUNG_THIS = "this_season"
RUNG_PRIOR = "prior_season"
RUNG_PRESEASON = "preseason_rate"
RUNGS = (RUNG_THIS, RUNG_PRIOR, RUNG_PRESEASON)

#: How imminent the return is. Three *different claims*, not three scores.
#:
#: ``cleared`` is the band nothing else in the app can see, and the most
#: actionable: FPL has already said he is available and no projection source has
#: caught up. ``imminent`` rests on a date the news actually states.
BAND_CLEARED = "cleared"
BAND_IMMINENT = "imminent"
BAND_WATCH = "watch"
BANDS = (BAND_CLEARED, BAND_IMMINENT, BAND_WATCH)

_BAND_ORDER = {BAND_CLEARED: 0, BAND_IMMINENT: 1, BAND_WATCH: 2}

#: Starts needed this season for ``total_points / starts`` to be worth reading.
#: Mirrors ``team_strength._MIN_STARTS_FOR_ACTUALS``, which is 4 for the same
#: reason; not imported because that module is about a different model and the
#: coupling would be accidental rather than meaningful.
THIS_SEASON_MIN_STARTS = 4

#: Element states a player must be in to be worth showing. ``'l'`` is kept --
#: locked means claimable at the *next* waiver, which is exactly the round this
#: board is for -- and flagged, the way ``_available_from_projections`` treats it.
CLAIMABLE_STATES = ("a", "l")

WATCH_COLUMNS = [
    "Player_Code", "Player_ID", "Player", "Display_Name", "Team", "Position",
    "Pedigree", "Pedigree_Rate", "Pedigree_Rung", "Role", "Role_Src",
    "Band", "Return_GW", "Out_For", "Basis",
    "Draft_State", "starts", "minutes", "Appearances", "News",
    "Prior_Season", "Prior_Starts", "Evidence",
]


def _note(notes, message, severity="warning"):
    """Record something the reader should see, for the caller to render.

    An explicit sink, matching ``waiver_core._note``: this frame is cached one
    layer up, and module-level mutable state behind a cache shows a warning on
    the cold call and drops it on every warm one.
    """
    _logger.warning(message)
    if notes is not None:
        notes.append((severity, message))


def _empty(notes=None, reason=""):
    if reason:
        _note(notes, reason, "info")
    out = pd.DataFrame({c: pd.Series(dtype="object") for c in WATCH_COLUMNS})
    return out


def _col(df, *names, numeric=True):
    """First present column among ``names``, as a Series aligned to ``df``.

    The same shape as ``analytics._player_id_col`` and ``_chance_of_playing_col``,
    and for the same reason: these frames spell the id, the club and the position
    differently depending on which builder made them, and a gate keyed on one
    spelling silently switches itself off on every frame carrying another. That
    failure has now cost this codebase five separate surfaces.

    Note ``DataFrame.get`` is deliberately not used -- it returns ``None`` or a
    bare scalar for a missing column, not a Series.
    """
    for name in names:
        if name in df.columns:
            s = df[name]
            return pd.to_numeric(s, errors="coerce") if numeric else s
    if numeric:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.Series(pd.NA, index=df.index, dtype="object")


def build_return_watch(
    pool,
    code_states,
    *,
    current_gw,
    prior_rates=None,
    deadlines=None,
    limit=None,
    notes=None,
    now=None,
    min_role=None,
    max_start_rate=None,
    min_pctile=None,
    horizon=None,
    min_gws=None,
    min_reference=None,
    ref_min_start_rate=None,
    evidence_minutes=None,
    evidence_apps=None,
):
    """Claimable players who are not playing but were productive regulars.

    Args:
        pool: the full ~670-player frame. Needs a club, a position, ``starts``
            and ``total_points``; uses ``news``, ``status`` and
            ``chance_of_playing_next_round`` for the bands, ``Season_PP90`` and
            ``SeasonProjection`` for rung 3. Column names are resolved loosely.
        code_states: ``{player_code: 'a' | 'l' | 'o'}`` -- the league's own
            ownership truth. Keyed on the **permanent player code**, never an
            element id: `element-status` is Draft-keyed and this pool is
            Classic-keyed, and 64 of 669 ids disagree, so an id join here
            resolves to a real but wrong player. ``waiver_core`` builds it.
        prior_rates: ``player_history.prior_season_rate`` output (rung 2).
        deadlines: real gameweek deadlines, so a stated return date is counted
            against the calendar rather than in seven-day blocks.
        limit: how many rows to return, best pedigree first.

    Returns:
        A frame of ``WATCH_COLUMNS``, ranked. **Empty with a stated reason**
        rather than silently, whenever it cannot answer -- the distinction
        between "nothing to do" and "could not tell" is the whole point of
        ``week_check.STATUS_UNKNOWN``, and it applies just as much here.
    """
    min_role = config.RETURN_WATCH_MIN_ROLE if min_role is None else float(min_role)
    max_start_rate = (config.RETURN_WATCH_MAX_START_RATE
                      if max_start_rate is None else float(max_start_rate))
    min_pctile = config.RETURN_WATCH_MIN_PCTILE if min_pctile is None else float(min_pctile)
    horizon = config.RETURN_WATCH_HORIZON if horizon is None else int(horizon)
    min_gws = config.RETURN_WATCH_MIN_GWS if min_gws is None else int(min_gws)
    min_reference = (config.RETURN_WATCH_MIN_REFERENCE
                     if min_reference is None else int(min_reference))
    ref_min_start_rate = (config.RETURN_WATCH_REF_MIN_START_RATE
                          if ref_min_start_rate is None else float(ref_min_start_rate))
    evidence_minutes = (config.RETURN_WATCH_EVIDENCE_MINUTES
                        if evidence_minutes is None else float(evidence_minutes))
    evidence_apps = (config.RETURN_WATCH_EVIDENCE_APPS
                     if evidence_apps is None else float(evidence_apps))
    limit = config.RETURN_WATCH_LIMIT if limit is None else limit

    try:
        current_gw = int(current_gw)
    except (TypeError, ValueError):
        return _empty(notes, "No gameweek, so there is nothing to compare against.")

    # Abstain early in the season. At GW1 every player has nothing played, so
    # "he is not playing" is not a statement about anybody -- and the reference
    # pool of established starters does not exist yet either.
    if current_gw < min_gws:
        return _empty(notes, "Too early to tell who is missing: the watch needs "
                             "GW%d before a player's start rate means anything."
                             % min_gws)

    if pool is None or len(pool) == 0:
        return _empty(notes, "No player pool, so no returning players can be found.")

    elapsed = max(1, current_gw - 1)
    df = pool.copy()

    code = _col(df, "Player_Code", "code")
    if not code.notna().any():
        return _empty(notes, "The player pool carries no player code, so league "
                             "ownership cannot be resolved without name matching.")
    df["Player_Code"] = code
    df["Player_ID"] = _col(df, "Player_ID", "id", "element_id", "element")
    df["Team"] = _col(df, "Team", "team_name_abbrv", "team_short", numeric=False)
    # Normalised to G/D/M/F, case-insensitively, via the one shared mapping.
    # Frames reaching here spell it both ways -- `pull_fpl_player_stats` emits
    # GKP/DEF/MID/FWD -- and every percentile here is grouped by it. CLAUDE.md
    # records what wrong position codes cost once already: they matched no group
    # and every Power Rankings team silently scored exactly 50.
    df["Position"] = (
        _col(df, "Position", "position_abbrv", numeric=False)
        .astype(str).str.strip().str.upper()
        .map({k.upper(): v for k, v in POS_MAP_TO_RW.items()})
        .fillna("")
    )
    if "Player" not in df.columns:
        df["Player"] = _col(df, "player", "web_name", "Display_Name", numeric=False)

    starts = _col(df, "starts").fillna(0.0)
    minutes = _col(df, "minutes").fillna(0.0)
    points = _col(df, "total_points", "Season_Points").fillna(0.0)
    news = _col(df, "news", "News", numeric=False).astype("string").fillna("")
    status = _col(df, "status", "Status", numeric=False).astype("string").fillna("")
    chance = _col(df, "chance_of_playing_next_round", "chance_of_playing", "PlayPct")
    df["starts"], df["minutes"] = starts, minutes
    # Carried through so the card can show it and `check_return_watch` can see a
    # departure that slipped the gate -- the likeliest false positive here.
    df["News"] = news

    # Whether FPL has him out of the squad right now. Computed here because the
    # role resolution below needs it as well as the not-playing gate further
    # down: it is the one fact that says whether this season's start share is
    # evidence about selection or about a treatment table.
    currently_out = status.str.lower().isin(list(OUT_OF_SQUAD_STATUSES))

    # Has this season produced enough of him to read? An appearance is the
    # signal that matters: a fit substitute and an absentee both record 0
    # minutes in a given week, and only the appearance count tells them apart.
    appearances = pd.Series(np.nan, index=df.index, dtype="float64")
    if prior_rates is not None and len(prior_rates) > 0 \
            and {"Player_Code", "Appearances"}.issubset(set(prior_rates.columns)):
        app_map = (prior_rates.drop_duplicates(subset=["Player_Code"])
                   .set_index("Player_Code")["Appearances"])
        appearances = pd.to_numeric(df["Player_Code"].map(app_map), errors="coerce")
    df["Appearances"] = appearances
    has_current_evidence = (
        (minutes >= float(evidence_minutes))
        | (appearances.fillna(0.0) >= float(evidence_apps))
    )

    # ---- the pedigree ladder: rate, and the role that goes with it ----------
    # The **rate** comes from the best rung -- the most recent adequate sample.
    # The **role** is the most favourable rung that has an opinion, recorded in
    # `Role_Src`, and the two deliberately need not agree.
    #
    # They have different denominators and so answer different questions. A
    # player injured in GW3 has started 3 of 10 elapsed gameweeks, which reads
    # as a fringe player when he is nothing of the kind -- his this-season share
    # is depressed by the very absence that put him on this board. The gate asks
    # "was he an established regular when fit", whose only failure mode is
    # losing a real candidate, so the most favourable reading is the right one.
    rate = pd.Series(np.nan, index=df.index, dtype="float64")
    rung = pd.Series(pd.NA, index=df.index, dtype="object")
    roles = {}

    # Rung 3 first, so better rungs overwrite it.
    pp90 = _col(df, "Season_PP90")
    season_total = _col(df, "SeasonProjection")
    # Rotowire builds its season total as `PP/90 x expected 90s`, so the two
    # columns together *imply* the minutes it expected -- which is the only role
    # signal a preseason table carries. Measured, those implied nineties run 4.4
    # to 38.1 across the table, so this is a real discriminator and not a
    # constant. Saliba's own row implies 17.8, i.e. Rotowire already knew he
    # would miss half the season.
    implied_role = (season_total / pp90.replace(0.0, np.nan)) / float(SEASON_GWS)
    m3 = pp90.notna() & (pp90 > 0)
    rate = rate.mask(m3, pp90)
    rung = rung.mask(m3, RUNG_PRESEASON)
    roles[RUNG_PRESEASON] = implied_role.where(m3).clip(upper=1.0)

    # Rung 2: last season that actually described him.
    if prior_rates is not None and len(prior_rates) > 0 \
            and {"Player_Code", "Prior_PPS", "Prior_Role"}.issubset(set(prior_rates.columns)):
        pr = prior_rates.drop_duplicates(subset=["Player_Code"]).set_index("Player_Code")
        keyed = df["Player_Code"]
        prior_pps = keyed.map(pr["Prior_PPS"])
        prior_role = keyed.map(pr["Prior_Role"])
        prior_season = keyed.map(pr["Prior_Season"])
        prior_starts = keyed.map(pr["Prior_Starts"])
        m2 = prior_pps.notna()
        rate = rate.mask(m2, prior_pps)
        rung = rung.mask(m2, RUNG_PRIOR)
        roles[RUNG_PRIOR] = prior_role.where(m2)
        df["Prior_Season"] = prior_season
        df["Prior_Starts"] = prior_starts
    else:
        df["Prior_Season"] = pd.NA
        df["Prior_Starts"] = np.nan
        if prior_rates is None:
            _note(notes, "No prior-season history, so a player who has not "
                         "featured at all this season cannot be rated.", "info")

    # Rung 1: this season, where he has started enough for a rate to mean
    # something. Wins outright -- it is the only rung describing him *now*.
    m1 = starts >= THIS_SEASON_MIN_STARTS
    rate = rate.mask(m1, points / starts.replace(0.0, np.nan))
    rung = rung.mask(m1, RUNG_THIS)
    # The *role* is recorded for everyone, not only where the rate is usable:
    # a player with two starts has no readable rate and a perfectly readable
    # share, and those are separate questions.
    this_role = (starts / float(elapsed)).clip(upper=1.0)
    roles[RUNG_THIS] = this_role

    role_frame = pd.DataFrame(roles) if roles else pd.DataFrame(index=df.index)
    df["Pedigree_Rate"] = rate
    df["Pedigree_Rung"] = rung
    if len(role_frame.columns):
        best = role_frame.max(axis=1)
        # `idxmax` over an all-NA row is deprecated and due to raise, and a row
        # with no role signal at all is ordinary here -- a player no rung could
        # rate. Ask only of the rows that have one.
        has_role = role_frame.notna().any(axis=1)
        best_src = pd.Series(pd.NA, index=df.index, dtype="object")
        if has_role.any():
            best_src.loc[has_role] = role_frame.loc[has_role].idxmax(axis=1)

        # **Where this season has already answered the question, it wins.**
        #
        # The most favourable rung is right for a player whose *absence*
        # depressed his share, and wrong for one whose *selection* did. Reported
        # from the app: Senesi topped the board on 97% of starts last season
        # while having started 2 of 10 for his new club, fit throughout.
        #
        # `currently_out` is the discriminator, and it is the same test as the
        # second leg of the not-playing gate below -- an injured man's share is
        # evidence about his injury, not about whether his manager rates him.
        # A player with too little of this season to read falls back as before,
        # which is what keeps a genuine absentee (Saliba, no appearances at all)
        # and a man just returning (Minteh, 8 minutes) on the board.
        current_decides = (~currently_out) & has_current_evidence
        df["Role"] = best.mask(current_decides, this_role)
        df["Role_Src"] = best_src.mask(current_decides, RUNG_THIS)
        df["Role_Decided_By_Season"] = current_decides
    else:
        df["Role"] = np.nan
        df["Role_Src"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df["Role_Decided_By_Season"] = False

    # ---- the reference pool: established starters only ---------------------
    # `positional_percentile` compares against whatever the reference holds, and
    # both obvious choices are wrong. Scoping it to the same rung makes the
    # reference *selected on not playing* -- "elite among benchwarmers". Pooling
    # everyone with four starts lets a small-sample tail set the top (Guimaraes
    # at 7.00 off 4 starts), and a genuine 4.57 elite defender then lands at the
    # 77th percentile. So the reference is players whose rate is stable: regular
    # starters, this season. The same asymmetry `team_strength.attach_reference_pps`
    # applies, and for the same reason -- noise in the *denominator* reprices
    # everyone.
    ref = df.copy()
    established = m1 & (starts >= ref_min_start_rate * elapsed)
    ref["Ref_Rate"] = df["Pedigree_Rate"].where(established)
    thin = {
        pos: int(grp["Ref_Rate"].notna().sum())
        for pos, grp in ref.groupby("Position", dropna=False)
    }
    for pos, n in sorted(thin.items()):
        if 0 < n < min_reference:
            _note(notes, "Only %d established starter(s) at %s, so a pedigree "
                         "percentile there compares against almost nobody."
                         % (n, pos))

    df["Pedigree"] = _referenced_percentile(
        df, ref, "Pedigree_Rate", ref_value_col="Ref_Rate",
        term="return_watch_pedigree",
    )

    # ---- gates -------------------------------------------------------------
    # "Not playing" is two shapes, and keying on the season-long rate alone
    # caught only one of them. A season-long absentee has a near-zero start
    # rate (Saliba, 0 of 10). A regular who got hurt in GW9 has a *high* one --
    # and at a 0.25 rate gate he could never qualify at all, since rung 1 needs
    # four starts and 0.25 x 10 elapsed is two. The two conditions were
    # mutually exclusive until GW17, so rung 1 contributed nothing: measured,
    # 15 of 15 live rows came from the prior season.
    #
    # `OUT_OF_SQUAD_STATUSES` is the second leg, and it excludes `d` for the
    # reason it was written: a *doubtful* player who has started every gameweek
    # is fully visible to every score the app already has, and admitting him
    # would turn this board into a list of this week's knocks. Measured, the
    # `chance < 100` version admitted 17 players of whom 13 were ordinary
    # weekly doubts.
    not_playing = ((starts / float(elapsed)) <= max_start_rate) | currently_out
    was_regular = df["Role"].fillna(0.0) >= min_role
    rated = df["Pedigree_Rate"].notna()
    # A departure is not an absence. `status` is 'u' for a loan, a permanent exit
    # and an ordinary unavailability alike, so this is drawn on the prose. It is
    # load-bearing rather than tidy: 7 of 18 raw candidate rows were departures,
    # and every one reads as a perfectly plausible injury row.
    gone = news.map(is_departed).astype(bool)
    claimable = df["Player_Code"].map(
        lambda c: (code_states or {}).get(int(c)) if pd.notna(c) else None)
    df["Draft_State"] = claimable
    in_state = claimable.isin(CLAIMABLE_STATES)

    if not code_states:
        return _empty(notes, "League ownership could not be read, so there is no "
                             "way to tell which returning players are claimable.")

    cand = df[rated & not_playing & was_regular & ~gone & in_state].copy()
    if cand.empty:
        return _empty(notes, "No claimable player is returning from a spell out "
                             "with the pedigree to be worth a claim.")

    # ---- bands -------------------------------------------------------------
    bands, return_gws, out_for, basis = [], [], [], []
    for idx, row in cand.iterrows():
        row_news = str(news.get(idx) or "")
        row_status = str(status.get(idx) or "").lower()
        row_chance = chance.get(idx)
        weeks, source = absence_estimate(row_news, row_chance, row_status,
                                         deadlines=deadlines, now=now)
        # The strict gate: only a duration the *text* states may be turned into
        # a gameweek. `estimate_games_to_miss`' chance and status rungs infer one
        # from a code, so "Unknown return date" would become a confident date.
        stated = news_stated_games_to_miss(row_news, deadlines, now=now)

        cleared = (row_status == "a"
                   and (pd.isna(row_chance) or float(row_chance) >= 100)
                   and not row_news.strip())
        if cleared:
            bands.append(BAND_CLEARED)
            return_gws.append(current_gw)
        elif stated is not None and 0 <= int(stated) <= horizon:
            bands.append(BAND_IMMINENT)
            return_gws.append(current_gw + int(stated))
        else:
            bands.append(BAND_WATCH)
            return_gws.append(np.nan)
        out_for.append(int(weeks) if weeks else 0)
        basis.append("stated" if source == ABSENCE_FROM_NEWS
                     else ("" if not weeks else "no stated date"))

    cand["Band"] = bands
    cand["Return_GW"] = return_gws
    cand["Out_For"] = out_for
    cand["Basis"] = basis

    # ---- rank, floor, truncate --------------------------------------------
    # A *rank*, not a threshold. An absolute top-decile bar empties the board:
    # the percentile is against established starters, where a top-decile
    # defender is Gabriel at 6.97 a start, so a returning player's realistic
    # ceiling is ~0.78. A hard 0.70 was rejected on measurement too -- it would
    # have excluded Saliba at 0.748, missing the reported case by a rounding
    # margin. `min_pctile` only drops the tail.
    before = len(cand)
    cand = cand[cand["Pedigree"] >= min_pctile]
    if cand.empty:
        return _empty(notes, "%d returning player(s) found, but none above the "
                             "positional median starter." % before)

    cand = attach_display_names(cand)
    cand["_band_order"] = cand["Band"].map(_BAND_ORDER).fillna(len(BANDS))
    cand = cand.sort_values(["Pedigree", "Pedigree_Rate"], ascending=False,
                            kind="mergesort")

    cand["Evidence"] = [_evidence(r, current_gw, elapsed) for _, r in cand.iterrows()]

    for col in WATCH_COLUMNS:
        if col not in cand.columns:
            cand[col] = pd.NA
    out = cand[WATCH_COLUMNS].reset_index(drop=True)
    if limit:
        out = out.head(int(limit))
    return out


def _evidence(row, current_gw, elapsed):
    """What this row's verdict rests on, so a reader can disagree with it.

    A row that cannot explain itself has no business being on the page, and this
    is also what ``check_return_watch`` reads to tell a derived finding from a
    defaulted one.
    """
    rung = row.get("Pedigree_Rung")
    ev = {
        "rung": rung,
        "rate": _round(row.get("Pedigree_Rate"), 2),
        "pedigree": _round(row.get("Pedigree"), 3),
        "role": _round(row.get("Role"), 2),
        "role_src": row.get("Role_Src"),
        "role_from_this_season": bool(row.get("Role_Decided_By_Season")),
        "appearances": _round(row.get("Appearances"), 0),
        "band": row.get("Band"),
        "state": row.get("Draft_State"),
        "starts": _round(row.get("starts"), 0),
        "elapsed": elapsed,
    }
    if rung == RUNG_PRIOR:
        ev["season"] = row.get("Prior_Season")
        ev["season_starts"] = _round(row.get("Prior_Starts"), 0)
    if pd.notna(row.get("Return_GW")) and row.get("Band") == BAND_IMMINENT:
        ev["return_gw"] = int(row.get("Return_GW"))
    return ev


def _round(value, places):
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return int(value) if places == 0 else round(float(value), places)


def role_phrase(row):
    """How regularly he started, in words, for a card.

    Reads the evidence rather than recomputing, so the sentence and the gate can
    never disagree about which season is being described.
    """
    rung = row.get("Role_Src") or row.get("Pedigree_Rung")
    role = row.get("Role")
    if role is None or pd.isna(role):
        return ""
    if rung == RUNG_PRIOR:
        starts = row.get("Prior_Starts")
        season = row.get("Prior_Season") or "last season"
        if starts is not None and pd.notna(starts):
            return "started %d of %d in %s" % (int(starts), SEASON_GWS, season)
        return "started %.0f%% of %s" % (100 * float(role), season)
    if rung == RUNG_THIS:
        return "started %.0f%% of this season's gameweeks" % (100 * float(role))
    return "Rotowire expected him to start %.0f%% of the season" % (100 * float(role))
