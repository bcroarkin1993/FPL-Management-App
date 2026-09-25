"""
Draft waiver priority — whose claim gets looked at first, and can yours win.

The Waiver Wire ranks players by how much they would improve your squad. That is
only half the question in Draft: a claim is a *request*, processed in league
order, and a target every manager ahead of you also wants is not a plan.

FPL publishes the order outright. ``league_entries[].waiver_pick`` on
``/api/league/{id}/details`` is the queue for the **next** waiver round, 1 first.
Nothing in this app read it until now.

Three rules from the platform shape everything here (see CLAUDE.md, "Draft
Transaction Rules"):

* The order is re-derived from the standings each gameweek — lowest-ranked
  manager picks first — and rotates *within* a round, a successful claimant going
  to the back. Measured live on league 11347 (2026-09-24): ``waiver_pick`` was the
  exact inverse of the standings.
* **A failed claim costs nothing.** Only a *successful* one sends you to the back.
  Recovered from the league's own transaction log by ``index`` ordering: one
  manager burned priorities 2, 3 and 4 in GW4 before an acceptance at priority 5,
  and kept their slot throughout. So the right claim list is your honest
  preference order, long shots included — which is what ``rank_claim_plan()``
  builds, and the opposite of how most managers hedge.
* Rivals' *pending* claims are never visible. Everything forward-looking here is
  therefore an estimate, and is labelled as one. What is not estimated is the
  queue itself, which is published.

**Two id spaces.** ``league_entries[].id`` keys ``standings.league_entry``;
``league_entries[].entry_id`` keys ``element_status.owner`` and
``transactions.entry``. ``waiver_pick`` sits on the row carrying both, and
``team_strength``'s ``Team_ID`` is the **entry_id** space. Crossing them yields a
silently empty join rather than an error, so both are carried explicitly and
``check_waiver_order()`` asserts your own entry resolves.

Pure and Streamlit-free, for the same reason ``transfer_risk.py`` is: offline
unit tests, and the Actions notifier can import it.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

_logger = logging.getLogger("fpl_app.waiver_priority")

# --- Transaction vocabulary --------------------------------------------------

#: `kind` on /api/draft/league/{id}/transactions.
TXN_KIND_WAIVER = "w"
TXN_KIND_FREE_AGENCY = "f"

#: `result`. Verified against 160 live rows on 2026-09-24: every one of the 58
#: `di` rows had that same `element_in` accepted by another manager in the same
#: event, so `di` is "beaten to this player by a higher priority claim". `do` is
#: some other decline (15 rows, only 9 of which had the player taken) — most
#: plausibly the manager's own earlier claim having already moved the outgoing
#: player. Only `a` is acted on here; the distinction matters for contention
#: counting, which must not read a `do` as evidence of competition.
TXN_RESULT_ACCEPTED = "a"
TXN_RESULT_DECLINED_TAKEN = "di"
TXN_RESULT_DECLINED_OTHER = "do"

# --- Outlook bands -----------------------------------------------------------

BAND_LIKELY = "likely"
BAND_CONTESTED = "contested"
BAND_LONG_SHOT = "long_shot"
BAND_UNKNOWN = "unknown"

#: A team is judged to *need* a position when its positional rank is at or worse
#: than this fraction of the league size — rank >= 6.67, so 7th-10th, in a
#: ten-team league. Deliberately coarse: this is a proxy for another manager's
#: intent, and a tighter threshold would dress a guess up as knowledge. The
#: user-facing wording says "among the weakest" rather than naming a fraction,
#: because rounding makes the true cut 4 of 10 rather than a literal third.
NEED_RANK_FRACTION = 2.0 / 3.0

_POS_CODES = ("G", "D", "M", "F")
#: team_strength.aggregate_team_strength() labels its columns GK/DEF/MID/FWD.
_POS_TO_STRENGTH_LABEL = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}


# =============================================================================
# THE PUBLISHED QUEUE
# =============================================================================

def parse_waiver_order(league_details: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The league's waiver queue, best pick first.

    Takes the whole ``/api/league/{id}/details`` payload and returns rows of
    ``{pick, entry_id, league_entry_id, team_name, manager}``.

    Both id spaces are carried deliberately — see the module docstring. Rows with
    no ``waiver_pick`` are dropped rather than defaulted: an absent pick is not
    pick 0, and inventing one would put a manager at the front of the queue.

    Returns ``[]`` for a payload that cannot be read, which callers must treat as
    "the order is unknown" and not as "there is no order".
    """
    if not isinstance(league_details, dict):
        return []
    entries = league_details.get("league_entries")
    if not isinstance(entries, list):
        return []

    rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        pick = entry.get("waiver_pick")
        if pick is None:
            continue
        try:
            pick = int(pick)
        except (TypeError, ValueError):
            _logger.warning("Non-integer waiver_pick %r on entry %r", pick, entry.get("id"))
            continue
        first = (entry.get("player_first_name") or "").strip()
        last = (entry.get("player_last_name") or "").strip()
        rows.append({
            "pick": pick,
            "entry_id": entry.get("entry_id"),
            "league_entry_id": entry.get("id"),
            "team_name": entry.get("entry_name") or "Unknown team",
            "manager": " ".join(p for p in (first, last) if p),
        })

    rows.sort(key=lambda r: r["pick"])
    return rows


def my_waiver_pick(order: Sequence[Dict[str, Any]], entry_id: Optional[int]) -> Optional[int]:
    """This manager's pick number, or None if they are not in the order.

    Matches on ``entry_id`` — the space ``element_status.owner`` and
    ``transactions.entry`` use, and the one ``config.FPL_DRAFT_TEAM_ID`` holds.
    """
    if entry_id is None:
        return None
    for row in order:
        if row.get("entry_id") == entry_id:
            return row.get("pick")
    return None


def managers_ahead(order: Sequence[Dict[str, Any]], entry_id: Optional[int]) -> List[Dict[str, Any]]:
    """Every manager whose claims are processed before this one's."""
    pick = my_waiver_pick(order, entry_id)
    if pick is None:
        return []
    return [r for r in order if r.get("pick") is not None and r["pick"] < pick]


# =============================================================================
# WHAT THE LEAGUE'S OWN HISTORY SAYS
# =============================================================================

def _waiver_rows(transactions: Optional[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not transactions:
        return []
    return [
        t for t in transactions
        if isinstance(t, dict) and t.get("kind") == TXN_KIND_WAIVER and t.get("event") is not None
    ]


def reconstruct_round_order(
    transactions: Optional[Iterable[Dict[str, Any]]],
    event: int,
) -> List[int]:
    """Entry ids in the order their claims were processed, for a past gameweek.

    Historical ``waiver_pick`` is not published — only today's is — so a past
    round's order has to be recovered. Each manager's *first* claim by ``index``
    is their turn in round one, so ordering managers by that recovers the queue
    as it stood.

    Only managers who actually claimed appear, which is why this is used for
    participation measurement rather than as a slot lookup.
    """
    rows = [t for t in _waiver_rows(transactions) if t.get("event") == event]
    first_seen: Dict[int, int] = {}
    for t in rows:
        entry = t.get("entry")
        idx = t.get("index")
        if entry is None or idx is None:
            continue
        if entry not in first_seen or idx < first_seen[entry]:
            first_seen[entry] = idx
    return [entry for entry, _ in sorted(first_seen.items(), key=lambda kv: kv[1])]


def claim_history_stats(
    transactions: Optional[Iterable[Dict[str, Any]]],
    n_managers: Optional[int] = None,
) -> Dict[str, Any]:
    """What this league's past waiver rounds actually looked like.

    The calibration that matters is ``participation_rate`` — the share of managers
    who bother to claim in a given round. The number of players gone before your
    turn is not your pick number; it is the number of managers ahead of you who
    *claim at all*, and that is a property of the league, measurable from its own
    log and from nothing else.

    ``contested_share`` and ``max_contention`` are reported for context: on league
    11347 one GW3 target drew eight competing claims, which is the situation the
    outlook bands exist to warn about.

    Returns neutral zeros for an unreadable log — never a fabricated rate.
    """
    rows = _waiver_rows(transactions)
    blank = {
        "events_observed": 0,
        "claimants_per_event": {},
        "accepted_per_event": {},
        "participation_rate": None,
        "median_accepted": None,
        "contested_share": None,
        "max_contention": 0,
    }
    if not rows:
        return blank

    events = sorted({t["event"] for t in rows})
    claimants: Dict[int, int] = {}
    accepted: Dict[int, int] = {}
    contention: Dict[tuple, int] = {}

    for ev in events:
        ev_rows = [t for t in rows if t["event"] == ev]
        claimants[ev] = len({t.get("entry") for t in ev_rows if t.get("entry") is not None})
        accepted[ev] = sum(1 for t in ev_rows if t.get("result") == TXN_RESULT_ACCEPTED)
        for t in ev_rows:
            el = t.get("element_in")
            if el is not None:
                contention[(ev, el)] = contention.get((ev, el), 0) + 1

    # A contested win is an accepted claim on a player more than one manager asked
    # for. Counting every `di` as contention would double-count: several declines
    # can trail one acceptance.
    accepted_keys = [
        (t["event"], t.get("element_in"))
        for t in rows
        if t.get("result") == TXN_RESULT_ACCEPTED and t.get("element_in") is not None
    ]
    contested_wins = sum(1 for k in accepted_keys if contention.get(k, 0) > 1)

    participation = None
    if n_managers:
        try:
            participation = statistics.mean(claimants.values()) / float(n_managers)
            participation = max(0.0, min(1.0, participation))
        except (statistics.StatisticsError, ZeroDivisionError):
            participation = None

    return {
        "events_observed": len(events),
        "claimants_per_event": claimants,
        "accepted_per_event": accepted,
        "participation_rate": participation,
        "median_accepted": statistics.median(accepted.values()) if accepted else None,
        "contested_share": (contested_wins / len(accepted_keys)) if accepted_keys else None,
        "max_contention": max(contention.values()) if contention else 0,
    }


def expected_gone_before(pick: Optional[int], participation_rate: Optional[float]) -> Optional[float]:
    """How many players to expect gone before your first claim is processed.

    ``(pick - 1) x participation_rate``: the managers ahead of you, discounted by
    how many of them historically claim at all. With no history the rate is
    unknown and this returns None rather than assuming everyone claims — an
    unmeasured worst case presented as a figure is how a guess becomes a fact.
    """
    if pick is None or participation_rate is None:
        return None
    return max(0.0, (int(pick) - 1) * float(participation_rate))


# =============================================================================
# WHO AHEAD OF YOU WANTS WHAT
# =============================================================================

def rival_needs(
    team_df,
    order: Sequence[Dict[str, Any]],
    entry_id: Optional[int],
) -> Dict[str, List[Any]]:
    """Positions the managers ahead of you are weak at, keyed ``G``/``D``/``M``/``F``.

    Reuses the Power Rankings model rather than inventing a second notion of squad
    need: ``team_strength.aggregate_team_strength()`` already scores every roster
    by position against the full FPL pool, and its ``Team_ID`` is the ``entry_id``
    space this module's order is keyed on, so the join is exact.

    A manager needs a position when their rank there is at or worse than
    ``NEED_RANK_FRACTION`` of the league size. Returns ``{}`` when power rankings
    are unavailable — the outlook then degrades to unknown rather than asserting
    nobody wants anything.
    """
    ahead = managers_ahead(order, entry_id)
    if not ahead or team_df is None:
        return {}
    try:
        if team_df.empty or "Team_ID" not in team_df.columns:
            return {}
    except AttributeError:
        return {}

    n_teams = len(team_df)
    if n_teams < 2:
        return {}
    threshold = NEED_RANK_FRACTION * n_teams
    ahead_ids = {r.get("entry_id") for r in ahead}

    needs: Dict[str, List[Any]] = {p: [] for p in _POS_CODES}
    for _, row in team_df.iterrows():
        if row.get("Team_ID") not in ahead_ids:
            continue
        for pos in _POS_CODES:
            rank = row.get(f"{_POS_TO_STRENGTH_LABEL[pos]}_Rank")
            try:
                rank = float(rank)
            except (TypeError, ValueError):
                continue
            if rank != rank:  # NaN
                continue
            if rank >= threshold:
                needs[pos].append(row.get("Team_ID"))
    return needs


# =============================================================================
# THE OUTLOOK
# =============================================================================

def claim_outlook(
    position: Optional[str],
    pos_rank: Optional[int],
    rivals_needing: Optional[int],
    n_ahead: Optional[int],
    expected_gone: Optional[float] = None,
    overall_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """How likely this claim is to survive the managers picking ahead of you.

    ``pos_rank`` is the target's rank among *available* players at his position,
    1 being the best. The reasoning is simply that each rival who needs that
    position takes one player off the top of it before you are reached, so a
    target sitting within that count is one you are unlikely to get.

    ``overall_rank`` widens it to competition from managers with no particular
    need at the position, using the league's measured ``expected_gone``.

    Returns ``{"band", "reason"}``. An unresolvable input gives ``BAND_UNKNOWN``
    and an empty reason, so a caller can render nothing rather than a guess.
    """
    blank = {"band": BAND_UNKNOWN, "reason": ""}
    if pos_rank is None or n_ahead is None:
        return blank
    try:
        pos_rank = int(pos_rank)
        n_ahead = int(n_ahead)
    except (TypeError, ValueError):
        return blank

    if n_ahead <= 0:
        return {"band": BAND_LIKELY, "reason": "You have the first waiver pick."}

    needing = int(rivals_needing) if rivals_needing is not None else 0
    pos_label = _POS_TO_STRENGTH_LABEL.get(position or "", position or "this position")

    if needing > 0 and pos_rank <= needing:
        return {
            "band": BAND_LONG_SHOT,
            "reason": (
                f"{needing} of the {n_ahead} managers ahead of you are among the "
                f"league's weakest at {pos_label}, and he is #{pos_rank} available "
                f"there."
            ),
        }

    if needing > 0 and pos_rank <= needing + 1:
        return {
            "band": BAND_CONTESTED,
            "reason": (
                f"{needing} of the {n_ahead} managers ahead of you are among the "
                f"league's weakest at {pos_label}; he is #{pos_rank} available "
                f"there, just past them."
            ),
        }

    if expected_gone is not None and overall_rank is not None:
        try:
            if int(overall_rank) <= expected_gone:
                return {
                    "band": BAND_CONTESTED,
                    "reason": (
                        f"No one ahead of you is short at {pos_label}, but he is "
                        f"#{int(overall_rank)} on the board and about "
                        f"{expected_gone:.0f} players typically go before your pick."
                    ),
                }
        except (TypeError, ValueError):
            pass

    if needing == 0:
        return {
            "band": BAND_LIKELY,
            "reason": (f"None of the {n_ahead} managers ahead of you is short at "
                       f"{pos_label}."),
        }
    return {
        "band": BAND_LIKELY,
        "reason": (
            f"He is #{pos_rank} available at {pos_label}, behind the {needing} "
            f"manager(s) ahead of you who need one."
        ),
    }


def rank_claim_plan(suggestions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Order suggestions into the priority list to submit to FPL.

    FPL asks you to rank your claims, and the instinct is to lead with something
    safe. That is wrong here, and the league's own log is why: processing walks
    down a manager's list until one claim succeeds, and **only a success costs
    the slot** — one manager was declined at priorities 2, 3 and 4 in GW4, won at
    5, and was never penalised for the misses.

    So the plan is the honest preference order, gain descending, long shots
    included and *not* demoted. This function exists rather than a ``sort`` at the
    callsite so that rule lives in one place with its evidence attached.

    Duplicate ``(drop, add)`` pairs are collapsed — the same swap cannot be
    claimed twice — and each row is stamped with its ``claim_priority``.

    **Gain is measured in expected points, not in the percentile gap.** This
    list is ranked across positions — a goalkeeper claim against a forward one —
    and a percentile difference is not the same quantity at two positions: on
    the live GW6 pool a 0.10 gain was worth 0.091 expected points at goalkeeper
    and 0.405 at forward, and this function ranked them equal. Since processing
    stops at your first success, mis-ordering the list costs you the claim you
    most wanted, which is the whole thing it exists to get right.

    ``transaction_score`` remains the tie-break, so a gameweek with no published
    projections degrades to the previous ordering rather than to an arbitrary
    one.
    """
    seen = set()
    rows = []
    for s in suggestions or []:
        if not isinstance(s, dict):
            continue
        key = (s.get("drop_player"), s.get("add_player"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(s))

    def _number(row, key):
        try:
            return float(row.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _gain(row):
        return (_number(row, "points_gain"), _number(row, "transaction_score"))

    rows.sort(key=_gain, reverse=True)
    for i, row in enumerate(rows, start=1):
        row["claim_priority"] = i
    return rows
