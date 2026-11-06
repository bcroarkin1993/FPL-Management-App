"""
Season odds, rendered beside the league table in both formats.

The pure Monte Carlo lives in ``season_sim``. This is the Streamlit side, shared
by Draft Home and Classic Home for the reason ``luck_analysis`` is shared: two
copies of a standings renderer drift, and these two differ only in where the
schedule comes from.

**It sits under the league table rather than in a tab.** It was first built as a
ninth tab on Draft League Analysis, where Streamlit's tab strip overflows and
pushed it off-screen entirely -- and the odds belong next to the standings they
re-order anyway, behind a toggle like "Show Luck Adjusted Standings".

Three leagues, two simulations:

* **Draft** and **Classic H2H** play fixtures, so ``simulate_season`` plays the
  remaining ones out and awards 3 and 1.
* **Classic total-points** has no fixtures at all. It is not the same simulation
  with an empty schedule -- nobody plays anybody and the table is ordered on
  accumulated points -- so ``simulate_points_league`` draws a score per manager
  per remaining gameweek and ranks on the total.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import streamlit as st

from scripts.common import season_sim
from scripts.common.data_validation import check_season_odds
from scripts.common.styled_tables import render_styled_table

_logger = logging.getLogger("fpl_app.season_odds")

#: The last gameweek of a Premier League season.
FINAL_GW = 38


def _render(sim: pd.DataFrame, entry_names: Dict[Any, str], *,
            playoff_places: int, sigma: float, sigma_n: int,
            remaining: int, remaining_noun: str, played: int) -> None:
    """The table plus the two things that keep it honest."""
    issues = check_season_odds(sim, playoff_places=playoff_places,
                               remaining_fixtures=remaining)
    for issue in issues:
        if issue.severity == "error":
            _logger.error("season odds: %s %s", issue.message, issue.hint)
            st.warning("The simulation did not come out consistent, so it is "
                       "not shown: %s" % issue.message)
            return
        _logger.warning("season odds: %s", issue.message)

    table = season_sim.summarise(sim, entry_names, playoff_places=playoff_places)
    top_col = "Top %d" % playoff_places
    render_styled_table(
        table,
        col_formats={"Title": "{:.1f}%", top_col: "{:.1f}%", "Last": "{:.1f}%",
                     "Proj Pts": "{:.1f}", "Avg Finish": "{:.2f}", "PPG": "{:.1f}"},
        positive_color_cols=["Title", top_col],
        text_align={c: "center" for c in table.columns if c != "Team"},
    )

    sigma_note = ("σ≈%.1f measured from %d past scores" % (sigma, sigma_n)
                  if sigma_n else "σ≈%.1f (default — no history to measure)" % sigma)
    # The caption carries the standard error because it is the reason these
    # numbers are not more confident, and a reader is entitled to check it.
    se = sigma / max(1.0, float(played) ** 0.5) if played else sigma
    st.caption(
        "%d %s left, %s. Each run also re-draws every manager's underlying "
        "scoring rate from its own sampling distribution, because that rate "
        "rests on **%d %s** — the standard error on it is about %.1f points, so "
        "treating it as known would report this league as far more decided than "
        "it is. Seeded on the league and gameweek, so these do not move when you "
        "toggle something."
        % (remaining, remaining_noun, sigma_note, played,
           "gameweeks" if played != 1 else "gameweek", se)
    )


def render_draft_season_odds(league_id: int, current_gw: int,
                             playoff_places: int = 3,
                             n_sims: int = season_sim.DEFAULT_SIMS) -> None:
    """Draft: head-to-head fixtures, 3 for a win and 1 for a draw."""
    from scripts.common.fpl_draft_api import get_draft_league_details
    from scripts.draft.fixture_projections import _estimate_score_std

    try:
        details = get_draft_league_details(league_id)
    except Exception as exc:
        st.warning("Could not load the league schedule: %s" % exc)
        return
    standings = (details or {}).get("standings") or []
    matches = (details or {}).get("matches") or []
    if not standings or not matches:
        st.info("Season odds need the league's schedule, which is not available yet.")
        return

    remaining = season_sim.remaining_matches(matches, from_gw=current_gw)
    if not remaining:
        st.success("Every fixture has been played — the table is final.")
        return

    sigma, sigma_n = _estimate_score_std(league_id)
    sim = season_sim.simulate_season(
        standings, matches, sigma=sigma, n_sims=n_sims, from_gw=current_gw,
        league_id=league_id, gameweek=current_gw, playoff_places=playoff_places)
    if sim.empty:
        st.info("Not enough league history to simulate from yet.")
        return

    names = {e["id"]: e.get("entry_name", "Entry %s" % e["id"])
             for e in (details.get("league_entries") or [])}
    _render(sim, names, playoff_places=playoff_places, sigma=sigma,
            sigma_n=sigma_n, remaining=len(remaining), remaining_noun="fixtures",
            played=int(sim["n_played"].max()))


def render_classic_h2h_season_odds(league_id: int, current_gw: int,
                                   standings_results: Sequence[Dict[str, Any]],
                                   matches: Sequence[Dict[str, Any]],
                                   playoff_places: int = 3,
                                   n_sims: int = season_sim.DEFAULT_SIMS) -> None:
    """Classic H2H: the same simulation, from Classic's payload shape.

    Classic names its sides ``entry_1_entry`` / ``entry_2_entry`` and its
    standings key is ``entry``, where Draft uses ``league_entry_1`` and
    ``league_entry``. Translating here rather than in ``season_sim`` keeps the
    pure module with one vocabulary -- and these are **two id spaces** of the
    kind this codebase has been bitten by, so the mapping is explicit.
    """
    if not standings_results or not matches:
        st.info("Season odds need the league's schedule, which is not available yet.")
        return

    standings = [{
        "league_entry": r.get("entry"),
        "total": r.get("total") or 0,
        "points_for": r.get("points_for") or 0,
        "matches_won": r.get("matches_won") or 0,
        "matches_drawn": r.get("matches_drawn") or 0,
        "matches_lost": r.get("matches_lost") or 0,
    } for r in standings_results if r.get("entry") is not None]

    # Carry the points through, not just a `finished` flag: Classic leaves
    # `finished` None on every match, so recorded scores are what tell a played
    # fixture from a future one. See `season_sim._finished`.
    fixtures = [{
        "event": m.get("event"),
        "league_entry_1": m.get("entry_1_entry"),
        "league_entry_2": m.get("entry_2_entry"),
        "entry_1_points": m.get("entry_1_points"),
        "entry_2_points": m.get("entry_2_points"),
        "finished": m.get("finished"),
    } for m in matches]

    remaining = season_sim.remaining_matches(fixtures, from_gw=current_gw)
    if not remaining:
        st.success("Every fixture has been played — the table is final.")
        return

    # **Sigma is measurable here, and was being defaulted.** Classic's H2H
    # payload carries each manager's score for every played fixture, which is
    # exactly the per-manager weekly history `weekly_score_sigma` wants -- and
    # pooled within-manager, so the spread *between* managers (squad quality,
    # the thing being projected) stays out of the noise term.
    played_rows = []
    for m in matches:
        if not season_sim._finished(m):
            continue
        for side in (1, 2):
            eid = m.get("entry_%d_entry" % side)
            pts = m.get("entry_%d_points" % side)
            if eid is not None and pts is not None:
                played_rows.append({"Team": eid, "GW_Points": pts})
    sigma, sigma_n = season_sim.weekly_score_sigma(pd.DataFrame(played_rows)) \
        if played_rows else (None, 0)
    if not sigma:
        sigma, sigma_n = season_sim.DEFAULT_SIGMA, 0

    sim = season_sim.simulate_season(
        standings, fixtures, sigma=sigma, n_sims=n_sims, from_gw=current_gw,
        league_id=league_id, gameweek=current_gw, playoff_places=playoff_places)
    if sim.empty:
        st.info("Not enough league history to simulate from yet.")
        return

    names = {r.get("entry"): r.get("entry_name", "Entry %s" % r.get("entry"))
             for r in standings_results}
    _render(sim, names, playoff_places=playoff_places,
            sigma=sigma, sigma_n=sigma_n,
            remaining=len(remaining), remaining_noun="fixtures",
            played=int(sim["n_played"].max()))


def render_classic_points_season_odds(league_id: int, current_gw: int,
                                      standings_df: pd.DataFrame,
                                      sigma_sample: Optional[pd.DataFrame] = None,
                                      playoff_places: int = 3,
                                      n_sims: int = season_sim.DEFAULT_SIMS) -> None:
    """Classic total-points: no fixtures, so the table is ordered on the total.

    **Means come from the totals and sigma from a sample, and the split is
    deliberate.** A manager's mean gameweek score is ``Total Pts / gameweeks
    played``, which the standings already carry for everyone -- no extra
    requests, and the same basis for every manager in the table. Sigma is a
    league-level noise parameter, so it is measured from whatever per-manager
    history the page already fetched for its charts (``sigma_sample``) and
    applied league-wide.

    Mixing the two -- measured means for the ten managers with history and
    derived means for the rest -- would rank two populations against each other
    on different bases, which is the error the projection engine exists to
    prevent, one model over.

    The odds are over the managers in ``standings_df``, which for a large league
    is the first page rather than the whole thing. That is said on the page, the
    same disclosure the Rival Tracker carries for its truncated EO.
    """
    if standings_df is None or standings_df.empty:
        st.info("Season odds need the league standings, which are not available yet.")
        return
    gws_left = max(0, FINAL_GW - int(current_gw or 0))
    if gws_left == 0:
        st.success("The season is complete — the table is final.")
        return

    sigma, sigma_n = (season_sim.weekly_score_sigma(sigma_sample)
                      if sigma_sample is not None else (None, 0))
    if not sigma:
        sigma, sigma_n = season_sim.DEFAULT_CLASSIC_SIGMA, 0

    played = max(1, int(current_gw or 1) - 1)
    entries = []
    for _, row in standings_df.iterrows():
        eid = row.get("entry_id")
        if eid is None:
            continue
        total = float(row.get("Total Pts") or 0)
        entries.append({"entry": eid, "total": total,
                        "mean_score": total / played, "n_played": played})
    if not entries:
        st.info("Season odds need the league standings, which are not available yet.")
        return

    sim = season_sim.simulate_points_league(
        entries, gameweeks_left=gws_left, sigma=sigma, n_sims=n_sims,
        league_id=league_id, gameweek=current_gw, playoff_places=playoff_places)
    names = dict(zip(standings_df["entry_id"], standings_df["Team"]))
    _render(sim, names, playoff_places=playoff_places, sigma=sigma,
            sigma_n=sigma_n, remaining=gws_left, remaining_noun="gameweeks",
            played=played)
    if len(standings_df) and bool(standings_df.attrs.get("truncated")):
        st.caption("These odds are over the managers shown above. A large "
                   "league returns one page of standings, so this is a finish "
                   "within that group rather than within the whole league.")
