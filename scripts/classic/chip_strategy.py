"""Chip Strategy — what each chip is worth, gameweek by gameweek.

The app knew which chips you still owned and had almost nothing to say about
when to play them: one collapsed expander on the Transfer Suggestions page,
three hardcoded rules, all three about the Wildcard, all three keyed on the mean
of a positional percentile. Bench Boost, Triple Captain and Free Hit had no
advice anywhere in the codebase.

This page answers the question in expected points, for every gameweek the
projections reach, and adds the two things published chip guidance actually
turns on:

* **An expiry clock read from FPL's own rules.** ``bootstrap-static.chips``
  publishes all eight chips with a ``start_event``/``stop_event``, and nothing
  in this app had ever read it -- which is why a spent Free Hit reported as
  gone when a second one opens at GW20.
* **Who you play.** In an H2H league a chip's value is the win probability it
  buys, which peaks when you are trailing by *half* the chip's gain -- so a
  weak opponent wastes it and so, nearly as much, does a very strong one. See
  ``scripts/common/chips.py`` for the arithmetic.

The valuation itself is pure and lives in ``scripts/common/chips.py``; this
module is fetching and rendering only.
"""

import pandas as pd
import streamlit as st

import config
from scripts.common import chips, h2h
from scripts.common.classic_squad import get_squad_budget, resolve_classic_squad
from scripts.common.data_validation import (
    check_chip_definitions,
    check_chip_timing,
)
from scripts.common.error_helpers import get_logger
from scripts.common.fixture_helpers import (
    blank_and_double_clubs,
    club_fixture_counts,
)
from scripts.common.fpl_classic_api import (
    get_classic_bootstrap_static,
    get_classic_team_history,
)
from scripts.common.player_pool import blended_pool, squad_frame
from scripts.common.scraping import get_ffp_horizon
from scripts.common.styled_tables import render_styled_table
from scripts.common.text_helpers import _map_position_to_rw
from scripts.fpl.player_projections import _horizon_frames

_logger = get_logger("fpl_app.classic.chip_strategy")

#: The picks-to-frame adapter now lives beside the pool it joins
#: against, so the My Week checklist reads the same fifteen.
_squad_frame = squad_frame

#: The head-to-head helpers now live in `scripts/common/h2h.py`, the Classic
#: Home alert strip having become a second reader. Re-exported under the names
#: this module has always used, so nothing below changed.
_h2h_scoring_league = h2h.is_h2h_league
_h2h_schedule = h2h.h2h_schedule
_h2h_sigma = h2h.h2h_sigma
_opponent_squad = h2h.opponent_squad
_lineup_projection = h2h.lineup_projection

#: Rows of the grid, in the order a manager decides them: the two "free" chips
#: that only need a good week first, then the two that rebuild a squad.
_GRID_ORDER = (chips.CHIP_BBOOST, chips.CHIP_3XC, chips.CHIP_FREEHIT)

#: Below this many expected points a chip recommendation is noise rather than
#: advice. The projections underneath carry an MAE around 2.5 points per
#: player, so this is a display floor and not a significance claim -- the same
#: reasoning as the Waiver Wire's `RENTAL_MIN_GAIN`.
MIN_INTERESTING_GAIN = 1.0


def _pos(value):
    """Normalise one position code to ``G/D/M/F``.

    Needed because the frames here disagree -- the pool emits ``G/D/M/F`` and a
    squad built from the bootstrap can carry ``GKP/DEF/MID/FWD`` or a raw
    ``element_type`` integer. A mismatched code does not raise: it matches no
    position group, so the formation gate silently does nothing, which is the
    failure that once had every Power Rankings team scoring exactly 50.
    """
    return _map_position_to_rw(value)


def _render_clock(status, current_gw):
    """The expiry clock, plus a pill per chip saying what it can do."""
    states = status.get("states") or {}
    avail = status.get("available", [])
    later = status.get("available_later", [])

    # The window is inclusive of the gameweek you can still act on, and it is
    # derived from the published `stop_event` rather than typed -- the whole
    # point of reading the rulebook instead of compiling it in.
    open_windows = sorted(
        {(s.expires_after, s.gws_left) for s in states.values()
         if s.playable_now and s.expires_after is not None},
        key=lambda pair: pair[0],
    )
    if open_windows:
        expires, left = open_windows[0]
        reopen = min((s.next_window_opens for s in states.values()
                      if s.next_window_opens is not None), default=None)
        tail = f" The next set opens GW{reopen}." if reopen else ""
        head = (f"**{left} gameweek{'s' if left != 1 else ''} left** for your "
                f"current chips — they expire after GW{expires} "
                f"(GW{current_gw}–GW{expires}).{tail}")
        (st.warning if left is not None and left <= 3 else st.info)(head)
    elif later:
        reopen = min((states[c].next_window_opens for c in later), default=None)
        st.info(f"No chip is playable this gameweek. Your next one opens "
                f"**GW{reopen}**.")
    else:
        st.success("Every chip has been played — nothing left to time.")

    pills = ""
    for chip in chips.KNOWN_CHIPS:
        label = chips.CHIP_LABELS[chip]
        state = states.get(chip)
        if chip in avail:
            pills += (f'<span style="background:{chips.CHIP_COLORS[chip]};color:#fff;'
                      f'padding:4px 11px;border-radius:12px;font-size:0.82em;'
                      f'font-weight:bold;margin:3px;">{label}</span>')
        elif chip in later:
            opens = getattr(state, "next_window_opens", None)
            pills += (f'<span style="background:#2d2d2d;color:#9ca3af;'
                      f'border:1px dashed #4b5563;padding:4px 11px;'
                      f'border-radius:12px;font-size:0.82em;margin:3px;">'
                      f'{label} &middot; GW{opens}</span>')
        else:
            played = ", ".join(f"GW{e}" for e in (getattr(state, "used_in", ()) or ()))
            pills += (f'<span style="background:#2d2d2d;color:#666;padding:4px 11px;'
                      f'border-radius:12px;font-size:0.82em;margin:3px;">'
                      f'{label} ({played or "used"})</span>')
    st.markdown(f'<div style="margin:6px 0 14px;color:#e0e0e0;">{pills}</div>',
                unsafe_allow_html=True)

    # Wildcard and Free Hit cannot be withdrawn once confirmed, which changes
    # how much a wrong recommendation costs and is worth saying beside one.
    st.caption(
        "Bench Boost and Triple Captain can be cancelled up to the deadline. "
        "Wildcard and Free Hit cannot be undone once confirmed."
    )


def _render_grid(values, gameweeks, status, horizon_end):
    """The chip x gameweek grid, in expected points."""
    rows = []
    for chip in _GRID_ORDER:
        per_gw = values.get(chip) or {}
        if not per_gw:
            continue
        best = chips.best_week(per_gw)
        row = {"Chip": chips.CHIP_LABELS[chip]}
        for gw in gameweeks:
            val = per_gw.get(gw)
            row[f"GW{gw}"] = float(val) if val is not None else float("nan")
        row["Best"] = (f"GW{best[0]}" if best else "—")
        rows.append(row)

    if not rows:
        st.info("No chip could be valued for this window — see the data notes below.")
        return

    grid = pd.DataFrame(rows)
    gw_cols = [f"GW{gw}" for gw in gameweeks]

    # One colour range across every gameweek column, taken over the whole grid.
    # Per-column ranges would make each week self-normalising, so a 16-point
    # Bench Boost week and a 9-point one would render the same green and the
    # only thing the grid is for -- reading across a row -- would be destroyed.
    finite = pd.concat([pd.to_numeric(grid[c], errors="coerce") for c in gw_cols])
    finite = finite[finite.notna()]
    span = (float(finite.min()), float(finite.max())) if not finite.empty else None

    render_styled_table(
        grid,
        title="Expected points each chip would add",
        col_formats={c: "{:.1f}" for c in gw_cols},
        text_align={**{c: "center" for c in gw_cols}, "Chip": "left", "Best": "center"},
        positive_color_cols=gw_cols,
        color_range_overrides=({c: span for c in gw_cols} if span else None),
    )
    st.caption(
        f"Expected points, so start probability is already inside each number. "
        f"The first column is the projection engine's own blend; later columns "
        f"are Fantasy Football Pundit's forecasts. Coloured on one scale across "
        f"all weeks so a row reads left to right."
    )


def _render_verdicts(values, status, gameweeks, horizon_end, dead, h2h):
    """One honest line per chip: the number, and what it does not know."""
    states = status.get("states") or {}
    avail = status.get("available", [])

    for chip in _GRID_ORDER:
        per_gw = values.get(chip) or {}
        if not per_gw:
            continue
        label = chips.CHIP_LABELS[chip]
        best = chips.best_week(per_gw, minimum=MIN_INTERESTING_GAIN)
        state = states.get(chip)
        expires = getattr(state, "expires_after", None)

        if best is None:
            st.markdown(f"**{label}** — nothing in this window is worth playing it for "
                        f"(every week under {MIN_INTERESTING_GAIN:.0f} points).")
            continue

        gw, val = best
        bits = [f"best week in view is **GW{gw}** at **{val:.1f} pts**"]

        # An H2H chip is worth what it changes, not what it scores.
        delta = (h2h or {}).get(chip, {}).get(gw)
        if delta is not None:
            bits.append(f"worth **{delta * 100:+.0f} pts of win probability** "
                        f"in that fixture")

        if chip not in avail:
            opens = getattr(state, "next_window_opens", None)
            bits.append(f"but you cannot play it until **GW{opens}**"
                        if opens else "but you have none left")
        elif expires is not None and horizon_end is not None and expires > horizon_end:
            bits.append(f"and your window runs to GW{expires}, "
                        f"{expires - horizon_end} week"
                        f"{'s' if expires - horizon_end != 1 else ''} past what "
                        f"these projections can see")

        st.markdown(f"**{label}** — " + "; ".join(bits) + ".")

    wc = values.get(chips.CHIP_WILDCARD)
    if wc is not None or dead:
        bits = []
        if wc is not None:
            bits.append(f"a rebuild buys about **{wc:+.1f} pts per gameweek** "
                        f"beyond what your free transfers already can")
        if dead:
            bits.append(f"**{dead}** of your fifteen cannot contribute at all "
                        f"(injured, suspended, or not expected to start)")
        if bits:
            st.markdown(f"**{chips.CHIP_LABELS[chips.CHIP_WILDCARD]}** — "
                        + "; ".join(bits) + ".")


def _render_schedule(gameweeks, squad_df, bootstrap):
    """Blanks and doubles in the window, and how many of your fifteen they hit."""
    counts = club_fixture_counts(gameweeks)
    if counts is None or counts.empty:
        st.caption("Could not read the fixture list, so blank and double "
                   "gameweeks are unknown for this window.")
        return

    found = blank_and_double_clubs(counts, pd.DataFrame(bootstrap.get("teams", [])))
    if not found:
        st.caption(
            f"No blank or double gameweek is scheduled in GW{min(gameweeks)}–"
            f"GW{max(gameweeks)}. That is the normal state for the first half of "
            f"a season — they appear once cup and European ties are rescheduled, "
            f"usually from December — so the values above rest on fixture "
            f"difficulty and form rather than on a doubled week."
        )
        return

    teams = {int(t["id"]): t["short_name"] for t in bootstrap.get("teams", [])}
    mine = set(squad_df["Team"].dropna().astype(str)) if "Team" in squad_df else set()
    for gw in sorted(found):
        parts = []
        for kind, word in (("blank", "blank"), ("double", "double")):
            clubs = found[gw].get(kind) or []
            if not clubs:
                continue
            yours = [c for c in clubs if c in mine]
            tag = f" (**{len(yours)}** of yours: {', '.join(yours)})" if yours else ""
            parts.append(f"{word}: {', '.join(clubs)}{tag}")
        if parts:
            st.markdown(f"- **GW{gw}** — " + "; ".join(parts))



# ---------------------------------------------------------------------------
# Who you play: the H2H lane
# ---------------------------------------------------------------------------
#
# Raised from the app: "I debated using a bench boost this week, but the team I
# am playing did not have a strong projection and is lower in the standings, so
# I figured I could save it for a better week against a stronger opponent."
#
# Correct in direction, and the usual framing of it is not. A chip's H2H value
# is the win probability it buys, and that peaks when you are trailing by *half*
# the chip's gain -- so a weak opponent wastes it (eight times over at a 45-point
# margin) and so, nearly as much, does a much stronger one, because you lose
# anyway. `chips.h2h_chip_value()` carries the arithmetic and the table.


def _h2h_deltas(values, squad_df, matrix, points, league_id, team_id,
                bootstrap, current_gw, pool, gameweeks):
    """``{chip: {gameweek: change in P(win)}}`` plus the context to caption it."""
    schedule = _h2h_schedule(league_id, team_id, gameweeks)
    if not schedule:
        return {}, {}
    sigma, n_obs = _h2h_sigma(league_id)

    context, deltas = {}, {}
    for gw, opponent in sorted(schedule.items()):
        mine = _lineup_projection(squad_df, matrix, gw)
        opp_squad = _opponent_squad(opponent, bootstrap, current_gw, pool)
        if mine is None or opp_squad is None or opp_squad.empty:
            continue
        opp_matrix = chips.squad_points_matrix(opp_squad, points, [gw])
        theirs = _lineup_projection(opp_squad, opp_matrix, gw)
        if theirs is None:
            continue
        context[gw] = {"opponent": opponent, "mine": mine, "theirs": theirs,
                       "sigma": sigma, "n": n_obs}
        for chip, per_gw in values.items():
            if not isinstance(per_gw, dict) or gw not in per_gw:
                continue
            delta = chips.h2h_chip_value(per_gw[gw], mine, theirs, sigma)
            if delta is not None:
                deltas.setdefault(chip, {})[gw] = delta
    return deltas, context


def _render_h2h(deltas, context, status, values):
    """The H2H lane: what each chip changes, not what it scores."""
    if not context:
        st.caption("No head-to-head fixture was found in this window, so there "
                   "is no opponent to time a chip against.")
        return

    any_sigma = next(iter(context.values()))
    rows = []
    for gw in sorted(context):
        ctx = context[gw]
        margin = ctx["mine"] - ctx["theirs"]
        row = {
            "GW": f"GW{gw}",
            "You": round(ctx["mine"], 1),
            "Them": round(ctx["theirs"], 1),
            "Margin": round(margin, 1),
            "P(win)": round(chips.win_probability(margin, ctx["sigma"]), 3),
        }
        for chip in _GRID_ORDER:
            val = (deltas.get(chip) or {}).get(gw)
            row[chips.CHIP_LABELS[chip]] = (round(val, 3) if val is not None
                                            else float("nan"))
        rows.append(row)

    chip_cols = [chips.CHIP_LABELS[c] for c in _GRID_ORDER]
    render_styled_table(
        pd.DataFrame(rows),
        title="Win probability each chip would buy",
        col_formats={**{c: "{:+.1%}" for c in chip_cols},
                     "P(win)": "{:.0%}", "You": "{:.1f}",
                     "Them": "{:.1f}", "Margin": "{:+.1f}"},
        text_align={c: "center" for c in ["GW", "You", "Them", "Margin",
                                          "P(win)"] + chip_cols},
        positive_color_cols=chip_cols,
    )

    # The counter-intuitive half, stated rather than left to be inferred.
    best = None
    for chip in _GRID_ORDER:
        hit = chips.best_week(deltas.get(chip) or {}, minimum=0.0)
        if hit and (best is None or hit[1] > best[2]):
            best = (chip, hit[0], hit[1])
    if best:
        chip, gw, val = best
        ctx = context[gw]
        margin = ctx["mine"] - ctx["theirs"]
        st.markdown(
            f"**Best use of a chip against who you actually play:** "
            f"{chips.CHIP_LABELS[chip]} in **GW{gw}**, worth "
            f"**{val * 100:+.0f} percentage points** of win probability "
            f"(projected margin {margin:+.1f})."
        )

        # Where the two lanes disagree, say so outright: this is the whole
        # reason the lane exists. The biggest points week is routinely *not*
        # the most useful one, because it can be a week you were winning
        # anyway -- and a manager reading only the points grid cannot see that.
        points_best = chips.best_week(values.get(chip) or {})
        if points_best and points_best[0] != gw:
            other_gw, other_pts = points_best
            other_delta = (deltas.get(chip) or {}).get(other_gw)
            other_ctx = context.get(other_gw)
            if other_delta is not None and other_ctx is not None:
                other_margin = other_ctx["mine"] - other_ctx["theirs"]
                st.info(
                    f"**The two readings disagree, and that is the useful part.** "
                    f"{chips.CHIP_LABELS[chip]} scores most in **GW{other_gw}** "
                    f"({other_pts:.1f} pts), but you are already "
                    f"{other_margin:+.1f} ahead there, so it only buys "
                    f"{other_delta * 100:+.0f} points of win probability. "
                    f"**GW{gw}** scores less and is worth "
                    f"{val * 100:+.0f} — the match is close enough for the "
                    f"chip to decide it."
                )

    st.caption(
        f"A chip is worth most where it is *just* big enough to turn a narrow "
        f"projected loss into a narrow win — the maximum is at a margin of "
        f"minus half the chip's gain, so a weak opponent wastes it and so, "
        f"nearly as much, does a much stronger one. Sigma "
        f"{any_sigma['sigma']:.1f} pts, measured from "
        f"{any_sigma['n']} league scores. Rival squads are last deadline's, so "
        f"a later gameweek assumes they make no transfers."
    )


def _render_solvers(squad_df, matrix, pool, points, gameweeks, status,
                    team_id, bootstrap, current_gw, free_transfers):
    """Free Hit and Wildcard, behind a button because each is an ILP solve.

    `st.tabs` renders every body on every rerun and this page has no tabs, but
    the same reasoning applies to a page section: eight squad solves is about a
    second of work, and somebody who opened this page for the expiry clock
    should not pay for it. Same gate as the Rival Tracker's "Scan".
    """
    st.markdown("### Free Hit and Wildcard")
    st.caption(
        "Both need a squad to be re-solved against the whole player pool, so "
        "they are computed on demand."
    )
    if not st.button("Value Free Hit and Wildcard", key="chip_solve"):
        return {}

    budget, budget_label = get_squad_budget(team_id, bootstrap, current_gw)
    out = {}
    with st.spinner("Solving squads…"):
        fh = chips.free_hit_value(squad_df, matrix, pool, points, budget)
        if fh:
            out[chips.CHIP_FREEHIT] = fh
        wc = chips.wildcard_value(
            squad_df, pool, score_col="Proj", budget=budget,
            free_transfers=free_transfers,
        )
        if wc is not None:
            out[chips.CHIP_WILDCARD] = wc

    if not out:
        st.info("Neither could be solved — the player pool or the budget could "
                "not be resolved. The chips above are unaffected.")
        return {}

    st.caption(f"Budget £{budget:.1f}m — {budget_label}.")
    if chips.CHIP_FREEHIT in out:
        best = chips.best_week(out[chips.CHIP_FREEHIT], minimum=MIN_INTERESTING_GAIN)
        if best:
            st.markdown(
                f"**Free Hit** — best week in view is **GW{best[0]}**, worth "
                f"**{best[1]:.1f} pts** over your own best XI that week.")
        else:
            st.markdown("**Free Hit** — your own squad is close enough to the "
                        "best available XI in every week here to not be worth it.")
    if chips.CHIP_WILDCARD in out:
        st.markdown(
            f"**Wildcard** — a full rebuild is worth about "
            f"**{out[chips.CHIP_WILDCARD]:+.1f} pts per gameweek** more than "
            f"your {free_transfers} free transfer"
            f"{'s' if free_transfers != 1 else ''} can already buy."
        )
    return out


def _h2h_league():
    """The configured Classic league that is scored head-to-head, if any.

    Returns the first one found. A manager in several leagues is usually only
    racing in one of them head-to-head, and asking which would be a question
    where a default is fine.
    """
    for entry in (getattr(config, "FPL_CLASSIC_LEAGUE_IDS", None) or []):
        league_id = entry.get("id") if isinstance(entry, dict) else entry
        if not league_id:
            continue
        try:
            if _h2h_scoring_league(league_id):
                return int(league_id)
        except Exception as exc:
            _logger.warning("Could not read league %s type: %s", league_id, exc)
    return None


def _free_transfer_count(history, resolution, current_gw):
    """How many free transfers are in hand, for the Wildcard's baseline.

    A Wildcard is only worth what it buys *beyond* the transfers already
    available, so this number is the baseline the chip is charged against.
    Resolved through the page that owns the question rather than re-counted
    here -- the rules (one a week from GW2, banked to five, chip weeks neither
    spend nor earn) are genuinely fiddly and have been got wrong before.
    """
    try:
        from scripts.classic.transfers import resolve_free_transfers
        info = resolve_free_transfers(
            history, resolution.entry_history or {}, current_gw,
            squad_source=resolution.source,
        )
        return int(info.get("count", 1) or 0)
    except Exception as exc:
        _logger.warning("Could not resolve free transfers: %s", exc)
        return 1

def show_chip_strategy_page():
    """Render the Chip Strategy page."""
    st.title("🃏 Chip Strategy")
    st.caption(
        "What each chip is worth in expected points, for every gameweek the "
        "projections reach — plus how long you have left to play it."
    )

    team_id = getattr(config, "FPL_CLASSIC_TEAM_ID", None)
    if not team_id:
        st.warning(
            "No Classic team is configured. Set one on the **🆔 League Setup** "
            "page (under FPL App Home) to see your chips valued against your "
            "own squad."
        )
        return

    current_gw = int(config.CURRENT_GAMEWEEK or 1)
    bootstrap = get_classic_bootstrap_static() or {}
    history = get_classic_team_history(team_id) or {}

    status = chips.chip_status_from_bootstrap(bootstrap, history, current_gw)
    _render_clock(status, current_gw)

    resolution = resolve_classic_squad(team_id, bootstrap, current_gw)
    if not resolution.ok:
        st.error("Could not resolve your squad, so there is nothing to value a "
                 "chip against. Check the team ID on the League Setup page.")
        return
    if resolution.is_stale:
        st.caption(f"⚠️ Squad as of GW{resolution.source_gw} "
                   f"({resolution.provenance or resolution.source}) — transfers "
                   f"made since are not reflected.")

    pool = blended_pool(current_gw, config.ROTOWIRE_URL)
    squad_df = _squad_frame(resolution.picks, pool)
    if squad_df.empty:
        st.error("Your squad resolved to no players.")
        return

    horizon = get_ffp_horizon()
    gameweeks, points = [], None
    if horizon.ok:
        built = _horizon_frames(horizon, pool, current_gw)
        points = built[0]
        if points is not None:
            gameweeks = [int(g) for g in horizon.gameweeks() if g in points.columns]
    if not gameweeks:
        st.warning(
            "Fantasy Football Pundit's multi-gameweek forecasts are "
            f"unavailable{': ' + horizon.note if horizon.note else ''}, so only "
            "this gameweek can be valued."
        )
        gameweeks = [current_gw]
        points = (pool.set_index("Player_ID")[["Proj"]].rename(columns={"Proj": current_gw})
                  if not pool.empty else None)

    matrix = chips.squad_points_matrix(squad_df, points, gameweeks)
    values = {
        chips.CHIP_BBOOST: chips.bench_boost_value(squad_df, matrix),
        chips.CHIP_3XC: chips.triple_captain_value(squad_df, matrix),
    }
    dead = chips.dead_weight_count(squad_df)
    horizon_end = max(gameweeks) if gameweeks else None

    _render_grid(values, gameweeks, status, horizon_end)

    # Free Hit and Wildcard are ILP solves, so they are opt-in; whatever they
    # return joins the same `values` mapping and so is validated with the rest.
    free_transfers = _free_transfer_count(history, resolution, current_gw)
    values.update(_render_solvers(squad_df, matrix, pool, points, gameweeks,
                                  status, team_id, bootstrap, current_gw,
                                  free_transfers))

    # Who you play. H2H only: a total-points league has no opponent, so the
    # change in win probability is undefined and only the points lane applies.
    deltas, context = {}, {}
    league = _h2h_league()
    if league:
        st.markdown("### Against who you actually play")
        try:
            deltas, context = _h2h_deltas(values, squad_df, matrix, points,
                                          league, team_id, bootstrap,
                                          current_gw, pool, gameweeks)
            _render_h2h(deltas, context, status, values)
        except Exception as exc:
            _logger.warning("H2H chip lane unavailable: %s", exc)
            st.caption("Could not resolve the head-to-head schedule, so the "
                       "win-probability lane is unavailable. The points above "
                       "are unaffected.")

    st.markdown("### What to do")
    _render_verdicts(values, status, gameweeks, horizon_end, dead, deltas)

    st.markdown("### Blank and double gameweeks")
    _render_schedule(gameweeks, squad_df, bootstrap)

    # Logged rather than raised: a page that renders a suspect number with a
    # warning in the log beats a page that will not render at all. The two
    # checks that matter here are negative values, which are impossible by
    # construction rather than merely unusual.
    for issue in check_chip_timing(values, squad_df, matrix):
        if issue.severity == "error":
            _logger.error("chip timing: %s", issue)
        else:
            _logger.info("chip timing: %s", issue)
    for issue in check_chip_definitions(bootstrap, known_chips=chips.KNOWN_CHIPS):
        _logger.log(40 if issue.severity == "error" else 20,
                    "chip definitions: %s", issue)
