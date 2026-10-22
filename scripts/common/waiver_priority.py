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

**An estimate about rivals has to be measured, not reasoned about.** The
reachability bands were originally inferred from Power Rankings: a manager ahead
of you ranked badly at a position was assumed to spend a claim there, on the best
player available. Three assumptions at full weight, and the league's own log
refutes the middle one — of 87 accepted claims on league 11347, goalkeepers were
**4**, while the model flagged the top two goalkeepers as unreachable every
round. ``position_claim_rates()`` reads what the league actually does per
position instead, over ``RECENT_ROUNDS`` rather than the whole season, since the
rounds just after a draft are churn and flatter a league's current activity.
``rival_needs()`` survives as explanatory prose on a card and sets nothing.

The bands describe the *queue*, so they are meaningless once free agency opens
and an available player can simply be added. Suppressing them there is the
caller's job — ``waiver_wire._build_waiver_context()`` returns no context at all
in that window, which disables the outlook end to end.

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
#:
#: **This no longer sets a band.** It is explanatory colour on a card whose
#: reachability comes from ``position_claim_rates()`` — see ``claim_outlook()``.
NEED_RANK_FRACTION = 2.0 / 3.0

#: How many waiver rounds to calibrate on. Early-season rounds are not
#: representative of a league's settled behaviour: measured on league 11347
#: (2026-10-22) claimants per round ran 4, 10, 10 for GW1-3 — the post-draft
#: churn, when every manager is still reshaping a squad — and then 8, 6, 6, 7, 6
#: for GW4-8. Averaging all eight reports 71% participation where the last four
#: say 63%, and that figure feeds both the expected-gone line and the bands.
RECENT_ROUNDS = 4

#: Below an even chance that a *single* player goes at a position, there is
#: nothing to warn about. This is what keeps goalkeeper quiet: league 11347 has
#: accepted four GK claims all season (4.6% of 87), one of them in the last four
#: rounds, so the expected number taken ahead of a mid-queue pick is ~0.13.
CONTESTED_MIN_EXPECTED = 0.5

_POS_CODES = ("G", "D", "M", "F")
#: team_strength.aggregate_team_strength() labels its columns GK/DEF/MID/FWD.
_POS_TO_STRENGTH_LABEL = {"G": "GK", "D": "DEF", "M": "MID", "F": "FWD"}
#: Plural, because the figure these describe is a count and is rarely 1.
_POS_TO_PLURAL_NOUN = {
    "G": "goalkeepers", "D": "defenders", "M": "midfielders", "F": "forwards",
}
#: settings.squad.select_* — the most players a manager could hold, and so a
#: ceiling on accepted claims at one position in one round. Used only by
#: ``check_position_claim_rates()`` as an impossible-value bound.
POSITION_SQUAD_QUOTA = {"G": 2, "D": 5, "M": 5, "F": 3}


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
    recent_rounds: int = RECENT_ROUNDS,
) -> Dict[str, Any]:
    """What this league's past waiver rounds actually looked like.

    The calibration that matters is participation — the share of managers who
    bother to claim in a given round. The number of players gone before your turn
    is not your pick number; it is the number of managers ahead of you who *claim
    at all*, and that is a property of the league, measurable from its own log and
    from nothing else.

    **Two participation figures, and callers should prefer the recent one.**
    ``participation_rate`` is the whole log and ``participation_rate_recent`` the
    last ``recent_rounds``. They disagree, and systematically: the rounds just
    after a draft are churn, with every manager still reshaping a squad, so an
    all-season mean reports a league as more active than it now is. See
    ``RECENT_ROUNDS``.

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
        "participation_rate_recent": None,
        "recent_events": [],
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

    recent_events = events[-max(1, int(recent_rounds)):]

    def _participation(evs) -> Optional[float]:
        if not n_managers:
            return None
        try:
            rate = statistics.mean(claimants[ev] for ev in evs) / float(n_managers)
        except (statistics.StatisticsError, ZeroDivisionError, KeyError):
            return None
        return max(0.0, min(1.0, rate))

    return {
        "events_observed": len(events),
        "claimants_per_event": claimants,
        "accepted_per_event": accepted,
        "participation_rate": _participation(events),
        "participation_rate_recent": _participation(recent_events),
        "recent_events": recent_events,
        "median_accepted": statistics.median(accepted.values()) if accepted else None,
        "contested_share": (contested_wins / len(accepted_keys)) if accepted_keys else None,
        "max_contention": max(contention.values()) if contention else 0,
    }


def position_claim_rates(
    transactions: Optional[Iterable[Dict[str, Any]]],
    element_positions: Optional[Dict[Any, str]],
    n_managers: Optional[int] = None,
    recent_rounds: int = RECENT_ROUNDS,
) -> Dict[str, float]:
    """Accepted claims per manager per round, by position, over recent rounds.

    This is what replaced "a rival ranked badly at this position will take the
    best player there". That reasoning stacked three assumptions at full weight —
    that the rival claims at all, that he claims at *this* position, and that he
    takes the top player — and the league's own log contradicts the middle one
    hardest. Measured on league 11347 over eight rounds, the 87 accepted claims
    split MID 36, DEF 29, FWD 18, **GK 4**, and an active manager claimed at only
    1.77 of the four positions. So a manager short at goalkeeper almost never
    spends a claim there, and the old model flagged two goalkeepers a round.

    **Accepted claims, not submitted ones.** A declined claim did not take a
    player off the board, and `di` rows trail an acceptance several deep — the
    same double-count ``contested_share`` avoids.

    ``element_positions`` maps element id to ``G``/``D``/``M``/``F``. Rows whose
    player cannot be resolved are skipped rather than bucketed, because an
    unresolved element would otherwise deflate every rate at once.

    Returns ``{}`` when nothing can be measured — no log, no position map, no
    manager count. A caller must read that as "unknown", never as "nobody claims":
    a fabricated rate here would silently re-create the problem this replaced.
    """
    rows = _waiver_rows(transactions)
    if not rows or not element_positions or not n_managers:
        return {}

    events = sorted({t["event"] for t in rows})
    if not events:
        return {}
    recent = set(events[-max(1, int(recent_rounds)):])

    counts: Dict[str, int] = {p: 0 for p in _POS_CODES}
    for t in rows:
        if t["event"] not in recent or t.get("result") != TXN_RESULT_ACCEPTED:
            continue
        pos = element_positions.get(t.get("element_in"))
        if pos in counts:
            counts[pos] += 1

    denominator = float(len(recent)) * float(n_managers)
    if denominator <= 0:
        return {}
    return {p: counts[p] / denominator for p in _POS_CODES}


def expected_taken_at_position(
    position: Optional[str],
    n_ahead: Optional[int],
    claim_rates: Optional[Dict[str, float]],
) -> Optional[float]:
    """How many players to expect gone *at this position* before your claim.

    ``n_ahead x`` the league's measured per-manager rate at that position. Live on
    league 11347 at pick 6 of 10 that is MID 2.38, DEF 1.25, FWD 0.50, GK 0.13 —
    against a model that called the top two at every position a long shot and the
    top *four* at forward, the position its log claims least.

    None when the rate was never measured, which ``claim_outlook()`` reads as
    "no opinion" rather than as zero.
    """
    if not claim_rates or position is None or n_ahead is None:
        return None
    rate = claim_rates.get(position)
    if rate is None:
        return None
    try:
        return max(0.0, int(n_ahead) * float(rate))
    except (TypeError, ValueError):
        return None


def expected_gone_before(pick: Optional[int], participation_rate: Optional[float]) -> Optional[float]:
    """How many players to expect gone before your first claim is processed.

    ``(pick - 1) x participation_rate``: the managers ahead of you, discounted by
    how many of them historically claim at all. With no history the rate is
    unknown and this returns None rather than assuming everyone claims — an
    unmeasured worst case presented as a figure is how a guess becomes a fact.

    Pass ``claim_history_stats()``'s ``participation_rate_recent``, not its
    all-season ``participation_rate``: the two differed by 8 points live (63%
    against 71%) because the post-draft rounds are churn.
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
    n_ahead: Optional[int],
    expected_at_position: Optional[float] = None,
    rivals_needing: Optional[int] = None,
    expected_gone: Optional[float] = None,
    overall_rank: Optional[int] = None,
) -> Dict[str, Any]:
    """How likely this claim is to survive the managers picking ahead of you.

    ``pos_rank`` is the target's rank among *available* players at his position,
    1 being the best, and ``expected_at_position`` is how many players this
    league's own log says go at that position before your pick — the figure
    ``expected_taken_at_position()`` measures. So the top ``floor(E)`` are
    expected gone, and the next rank is the borderline.

    **The band is measured, not inferred from rival weakness.** It used to be
    driven by ``rivals_needing``: a count of managers ahead of you ranked badly at
    the position, with ``pos_rank <= needing`` a long shot outright. That read a
    positional *rank* as an intention to claim, and the league's log says it does
    not follow — on league 11347 goalkeepers were 4 of 87 accepted claims while
    the model flagged the top two every round, and forwards were 18 of 87 while it
    flagged the top *four*. ``rivals_needing`` survives as a trailing clause only,
    and only when it is non-zero, so it can add colour but can never contradict
    the band.

    ``overall_rank`` keeps the board-wide path: a standout nobody ahead of you is
    specifically short at still goes, and ``expected_gone`` is what prices that.

    Returns ``{"band", "reason"}``. An unresolvable input — or a league with no
    waiver history to calibrate on — gives ``BAND_UNKNOWN`` and an empty reason,
    so a caller renders nothing rather than a guess. That is deliberate: the
    previous model would happily band a league it had never observed.
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

    expected = None
    if expected_at_position is not None:
        try:
            expected = max(0.0, float(expected_at_position))
        except (TypeError, ValueError):
            expected = None

    if expected is None and expected_gone is None:
        # Nothing about this league has been measured. Saying anything here is
        # how the old model came to assert contention it had no evidence for.
        return blank

    pos_label = _POS_TO_STRENGTH_LABEL.get(position or "", position or "this position")
    noun = _POS_TO_PLURAL_NOUN.get(position or "", "players")

    # Rival weakness is corroborating prose, never the band. Appended only when
    # it agrees with there being competition, so a card cannot read "none of them
    # is short here" beside a Contested pill.
    try:
        needing = int(rivals_needing) if rivals_needing is not None else 0
    except (TypeError, ValueError):
        needing = 0
    colour = (
        f" {needing} of them {'is' if needing == 1 else 'are'} among the league's "
        f"weakest at {pos_label}."
        if needing > 0 else ""
    )

    if expected is not None:
        swept = int(expected)   # floor: the ranks expected to be gone
        if pos_rank <= swept:
            return {
                "band": BAND_LONG_SHOT,
                "reason": (
                    f"About {expected:.1f} {noun} go to the {n_ahead} managers "
                    f"ahead of you in a typical round, and he is #{pos_rank} "
                    f"available there.{colour}"
                ),
            }
        if expected >= CONTESTED_MIN_EXPECTED and pos_rank <= swept + 1:
            return {
                "band": BAND_CONTESTED,
                "reason": (
                    f"He is #{pos_rank} available at {pos_label}, right at the "
                    f"edge of the roughly {expected:.1f} {noun} that go to the "
                    f"{n_ahead} managers ahead of you.{colour}"
                ),
            }

    if expected_gone is not None and overall_rank is not None:
        try:
            if int(overall_rank) <= expected_gone:
                return {
                    "band": BAND_CONTESTED,
                    "reason": (
                        f"{pos_label} is not what the managers ahead of you "
                        f"usually claim, but he is #{int(overall_rank)} on the "
                        f"board and about {expected_gone:.0f} players typically "
                        f"go before your pick."
                    ),
                }
        except (TypeError, ValueError):
            pass

    if expected is not None:
        return {
            "band": BAND_LIKELY,
            "reason": (
                f"Only about {expected:.1f} {noun} go to the {n_ahead} managers "
                f"ahead of you in a typical round, and he is #{pos_rank} "
                f"available there."
            ),
        }
    return {
        "band": BAND_LIKELY,
        "reason": (
            f"He is #{pos_rank} available at {pos_label} and well outside the "
            f"{expected_gone:.0f} players that typically go before your pick."
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
