# scripts/common/classic_squad.py
#
# One resolver for "what 15 does this Classic manager currently have?", with
# explicit provenance.
#
# WHY THIS EXISTS
# ---------------
# Four Classic pages used to answer this question three different ways and could
# disagree about the same squad on the same day:
#
#   transfers.py          tried current_gw+1 / bootstrap is_next first, skipped
#                         Free Hit GWs, replayed confirmed + locally-logged
#                         transfers, and said where the squad came from
#   fixture_projections   walked back from target_gw-1, did NOT skip Free Hit
#                         GWs, replayed confirmed transfers only, and said
#                         nothing at all
#   free_hit / wildcard   two copies of _get_user_budget() reading value/bank
#                         straight off possibly-stale picks
#
# The silence was the bug that prompted this: a manager who had just played a
# wildcard saw the pre-wildcard squad rendered as if it were current. FPL
# certainly publishes no *picks* for an upcoming gameweek (404, pinned by
# tests/live/); whether it also withholds transfers and the active chip until
# the deadline is inferred rather than observed — see the note in fpl_auth.py.
# Whichever it is, a squad the page cannot verify must not be presented as
# current, which is what `is_stale` and `provenance` are for.
#
# PRECEDENCE
#   1. my_team        authenticated live squad — the only pre-deadline truth
#   2. local_pending  transfers the user logged in-app, replayed on the base
#   3. picks_replay   last-deadline picks + transfers FPL has confirmed
#   4. picks          last-deadline picks, nothing to replay
#
# Steps 2-4 are the unauthenticated path and are the default: auth is attempted
# only for the user's own configured team, so H2H opponents and the league
# leaderboard loop can never accidentally reach for it.

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from scripts.common.fpl_auth import (
    STATUS_NO_AUTH,
    fetch_my_team,
    has_auth,
)
from scripts.common.fpl_classic_api import (
    get_classic_team_history,
    get_classic_team_picks,
    get_classic_transfers,
    get_entry_details,
)

_logger = logging.getLogger(__name__)

__all__ = [
    "SquadResolution",
    "resolve_classic_squad",
    "get_squad_budget",
    "apply_pending_transfers",
    "load_pending_file",
    "save_pending_file",
    "PENDING_FILE",
]

PENDING_FILE = Path(".fpl_pending_transfers.json")

SOURCE_MY_TEAM = "my_team"
SOURCE_LOCAL_PENDING = "local_pending"
SOURCE_PICKS_REPLAY = "picks_replay"
SOURCE_PICKS = "picks"


@dataclass
class SquadResolution:
    """A resolved squad plus where it came from.

    `is_stale` is the load-bearing field: True means the squad predates
    transfers the manager may already have made, and the page must say so.
    """

    picks: List[dict] = field(default_factory=list)
    active_chip: Optional[str] = None
    entry_history: Dict[str, Any] = field(default_factory=dict)
    source: str = SOURCE_PICKS
    source_gw: Optional[int] = None
    target_gw: Optional[int] = None
    auth_status: Optional[str] = None
    is_stale: bool = False
    applied_transfers: List[str] = field(default_factory=list)
    # Locally-logged transfers the authenticated squad contradicts: logged in the
    # app but element_out is still in the real squad, so the move never went
    # through. Never replayed — inventing a squad the manager does not own is
    # the same class of silent-wrong-answer this module exists to prevent.
    local_stale: List[dict] = field(default_factory=list)
    provenance: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.picks)

    def as_picks_data(self) -> dict:
        """The shape get_classic_team_picks() returns, for existing consumers."""
        return {
            "picks": self.picks,
            "active_chip": self.active_chip,
            "entry_history": self.entry_history,
        }


# ---------------------------------------------------------------------------
# Locally-logged pending transfers (file layer only)
# ---------------------------------------------------------------------------
# The session_state layer and the add/remove UI stay in classic/transfers.py,
# which owns them. Read-only consumers (Fixture Projections, the optimizers)
# only need the file, so they get it here without dragging session state along.

def load_pending_file() -> list:
    """Read all locally-logged pending transfers. [] when absent or unreadable."""
    try:
        if PENDING_FILE.exists():
            data = json.loads(PENDING_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
    except (json.JSONDecodeError, OSError):
        _logger.warning("Could not read %s", PENDING_FILE.name, exc_info=True)
    return []


def save_pending_file(transfers: list) -> None:
    """Write all locally-logged pending transfers."""
    PENDING_FILE.write_text(json.dumps(transfers, indent=2), encoding="utf-8")


def apply_pending_transfers(picks_data: dict, picks_source_gw: int,
                            all_transfers: list, fh_gws: set,
                            bootstrap: dict) -> tuple:
    """Apply transfers registered after picks_source_gw (excluding FH GWs).

    Returns (updated_picks_data, list_of_applied_descriptions).
    This is needed because the FPL picks endpoint returns 404 for upcoming
    GWs, so between GWs we load the last completed GW's picks and replay
    any transfers on top to reconstruct the current registered squad.
    """
    # Apply permanent transfers registered after the base squad snapshot.
    # Two categories:
    #   - API transfers: exclude FH GW events (those are temporary FH squad
    #     changes stored with the FH event number; replaying them rebuilds
    #     the wrong squad).
    #   - Local transfers (local=True): always apply — these are user-confirmed
    #     permanent transfers. get_current_gameweek() may return the FH GW
    #     number between gameweeks, causing local transfers to be saved with
    #     event == fh_gw. We must NOT filter those out.
    pending = [
        t for t in (all_transfers or [])
        if (
            # Local transfers: allow event == picks_source_gw since the base picks
            # represent the squad *before* the user's intended transfer. Using strict >
            # would drop the transfer when FPL returns current-GW picks as the base.
            t.get("local") and t.get("event", 0) >= picks_source_gw
        ) or (
            not t.get("local")
            and t.get("event", 0) > picks_source_gw
            and t.get("event", 0) not in fh_gws
        )
    ]
    if not pending:
        return picks_data, []

    picks_data = copy.deepcopy(picks_data)
    picks = picks_data.get("picks", [])
    elements_lookup = {p["id"]: p for p in bootstrap.get("elements", [])}

    applied = []
    for transfer in sorted(pending, key=lambda t: (t.get("event", 0), t.get("time", ""))):
        out_id = transfer["element_out"]
        in_id = transfer["element_in"]
        for pick in picks:
            if pick["element"] == out_id:
                pick["element"] = in_id
                # selling_price = what was paid (can only sell for this or less if risen)
                pick["selling_price"] = transfer.get(
                    "element_in_cost",
                    elements_lookup.get(in_id, {}).get("now_cost", 0)
                )
                out_name = elements_lookup.get(out_id, {}).get("web_name", str(out_id))
                in_name = elements_lookup.get(in_id, {}).get("web_name", str(in_id))
                applied.append(f"{out_name} → {in_name} (GW{transfer['event']})")
                break

        # Update bank: selling price received minus purchase price
        out_cost = transfer.get("element_out_cost", 0)
        in_cost = transfer.get("element_in_cost", 0)
        if "entry_history" in picks_data and out_cost and in_cost:
            picks_data["entry_history"]["bank"] = (
                picks_data["entry_history"].get("bank", 0) + out_cost - in_cost
            )

    return picks_data, applied


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def _free_hit_gws(history: Optional[dict]) -> set:
    chips = (history or {}).get("chips") or []
    return {c["event"] for c in chips if c.get("name") == "freehit" and c.get("event")}


def _candidate_gws(bootstrap: dict, current_gw: int) -> List[int]:
    """Base-squad gameweeks to try, most recent first.

    current_gw + 1 and the bootstrap's is_next both appear because the two
    gameweek notions disagree between gameweeks: config/get_current_gameweek()
    roll forward to the upcoming GW the moment the previous one finishes, while
    bootstrap events are authoritative about which GW is actually next.
    """
    events = (bootstrap or {}).get("events") or []
    next_gw = next((e["id"] for e in events if e.get("is_next")), None)
    candidates = {current_gw + 1, next_gw, current_gw,
                  current_gw - 1, current_gw - 2, current_gw - 3}
    candidates.discard(None)
    return sorted((gw for gw in candidates if gw >= 1), reverse=True)


def _next_deadline_gw(bootstrap: dict, current_gw: int) -> int:
    events = (bootstrap or {}).get("events") or []
    next_gw = next((e["id"] for e in events if e.get("is_next")), None)
    return next_gw or current_gw


def _is_user_team(team_id) -> bool:
    try:
        return int(team_id) == int(config.FPL_CLASSIC_TEAM_ID or 0)
    except (TypeError, ValueError):
        return False


def _reconcile_pending_log(team_id, live_picks: List[dict]) -> List[dict]:
    """Retire locally-logged transfers the authenticated squad has settled.

    An entry whose `element_out` is gone from the live squad has gone through —
    drop it from the log, exactly as _sync_pending_local drops entries FPL has
    confirmed. An entry whose `element_out` is still there did NOT go through;
    it is returned so the page can say so, and is never applied.
    """
    pending = load_pending_file()
    mine = [t for t in pending if t.get("team_id") == team_id]
    if not mine:
        return []

    live_elements = {p.get("element") for p in live_picks}
    stale = [t for t in mine if t.get("element_out") in live_elements]
    settled = [t for t in mine if t.get("element_out") not in live_elements]

    if settled:
        keep = [t for t in pending if t not in settled]
        try:
            save_pending_file(keep)
        except OSError:
            _logger.warning("Could not prune the pending-transfer log", exc_info=True)
    return stale


def resolve_classic_squad(
    team_id: int,
    bootstrap: dict,
    current_gw: int,
    history: Optional[dict] = None,
    allow_auth: Optional[bool] = None,
    extra_transfers: Optional[list] = None,
) -> SquadResolution:
    """Resolve a Classic manager's current registered squad.

    Parameters
    ----------
    team_id : the Classic entry id.
    bootstrap : bootstrap-static (for `events` and `elements`).
    current_gw : the app's notion of the current gameweek.
    history : entry history, for Free Hit gameweeks. Fetched if omitted.
    allow_auth : attempt the authenticated my-team read. Defaults to
        "this is the user's own configured team" — so the unauthenticated
        path is the default and opponents are never attempted.
    extra_transfers : transfers to replay instead of the ones read from the
        pending file. classic/transfers.py passes its session-state-merged
        list here so its behaviour is unchanged.
    """
    if not team_id:
        return SquadResolution(provenance="No Classic team configured.")

    if allow_auth is None:
        allow_auth = _is_user_team(team_id)

    target_gw = _next_deadline_gw(bootstrap, current_gw)

    # ---- 1. Authenticated live squad -------------------------------------
    auth_status = None
    if allow_auth and has_auth():
        payload, auth_status = fetch_my_team(team_id)
        if payload and payload.get("picks"):
            # The authenticated squad is ground truth, so nothing is replayed on
            # top of it. Replaying a locally-logged transfer that the real squad
            # already contains would double-count its bank adjustment — the
            # pick-swap loop no-ops when element_out is gone, but the bank line
            # in apply_pending_transfers() fires regardless.
            stale = _reconcile_pending_log(team_id, payload["picks"])
            return SquadResolution(
                picks=payload["picks"],
                active_chip=payload.get("active_chip"),
                entry_history=payload.get("entry_history", {}),
                source=SOURCE_MY_TEAM,
                source_gw=target_gw,
                target_gw=target_gw,
                auth_status=auth_status,
                is_stale=False,
                local_stale=stale,
                provenance="Live squad from your FPL account.",
            )
    elif allow_auth:
        auth_status = STATUS_NO_AUTH

    # ---- 2-4. Public reconstruction --------------------------------------
    if history is None:
        history = get_classic_team_history(team_id) or {}
    fh_gws = _free_hit_gws(history)

    picks_data = None
    source_gw = None
    for gw in _candidate_gws(bootstrap, current_gw):
        # Free Hit GWs hold a temporary squad that reverts afterwards — using
        # one as the base rebuilds a squad the manager does not own.
        if gw in fh_gws:
            continue
        candidate = get_classic_team_picks(team_id, gw)
        if candidate and candidate.get("picks"):
            picks_data = candidate
            source_gw = gw
            break

    if not picks_data:
        return SquadResolution(
            target_gw=target_gw,
            auth_status=auth_status,
            provenance="Could not load a squad for this team.",
        )

    if extra_transfers is None:
        api_transfers = get_classic_transfers(team_id) or []
        local_pending = [t for t in load_pending_file() if t.get("team_id") == team_id]
        # Drop locally-logged transfers FPL has since confirmed. Match on the
        # (out, in, event) triplet — a bare player pair would clear a new
        # pending transfer whenever the same swap happened in an earlier GW.
        confirmed = {(t["element_out"], t["element_in"], t.get("event", 0))
                     for t in api_transfers}
        local_pending = [
            t for t in local_pending
            if (t.get("element_out"), t.get("element_in"), t.get("event", 0)) not in confirmed
        ]
        all_transfers = api_transfers + local_pending
    else:
        all_transfers = extra_transfers
        local_pending = [t for t in all_transfers if t.get("local")]

    picks_data, applied = apply_pending_transfers(
        picks_data, source_gw, all_transfers, fh_gws, bootstrap
    )

    has_local = bool(local_pending)
    source = SOURCE_LOCAL_PENDING if has_local else (
        SOURCE_PICKS_REPLAY if applied else SOURCE_PICKS
    )

    # Stale means: the base predates the upcoming deadline and nothing has been
    # replayed on top of it that covers that gap. FPL publishes neither pending
    # transfers nor an active chip until a deadline passes, so this is the
    # normal state between gameweeks — and the state the page must disclose.
    is_stale = (not has_local) and source_gw < target_gw

    if has_local:
        provenance = (
            f"Squad as of GW{source_gw}, plus {len(local_pending)} "
            f"locally-logged transfer(s)."
        )
    elif is_stale:
        provenance = (
            f"Squad as of the GW{source_gw} deadline. FPL does not publish pending "
            f"transfers or an active chip until the GW{target_gw} deadline passes, "
            f"so any moves you have made since are not shown."
        )
    else:
        provenance = f"Squad as registered for GW{source_gw}."

    return SquadResolution(
        picks=picks_data.get("picks", []),
        active_chip=picks_data.get("active_chip"),
        entry_history=picks_data.get("entry_history", {}),
        source=source,
        source_gw=source_gw,
        target_gw=target_gw,
        auth_status=auth_status,
        is_stale=is_stale,
        applied_transfers=applied,
        provenance=provenance,
    )


def get_squad_budget(team_id: int, bootstrap: Optional[dict] = None,
                     current_gw: Optional[int] = None) -> tuple:
    """Total budget (squad value + bank) in £m, plus a human-readable source.

    Shared by the Free Hit and Wildcard optimizers, which each carried a copy.
    bootstrap/current_gw are optional so the optimizers can keep calling this
    with a bare team id.
    """
    if not team_id:
        return 100.0, "default (no team configured)"

    try:
        if bootstrap is None:
            from scripts.common.fpl_classic_api import get_classic_bootstrap_static
            bootstrap = get_classic_bootstrap_static() or {}
        if current_gw is None:
            current_gw = int(config.CURRENT_GAMEWEEK)
        resolution = resolve_classic_squad(team_id, bootstrap, current_gw)
        entry_history = resolution.entry_history or {}
        value = entry_history.get("value", 0)
        bank = entry_history.get("bank", 0)
        if value > 0:
            total = (value + bank) / 10.0
            label = f"detected (value: £{value/10:.1f}m + bank: £{bank/10:.1f}m)"
            if resolution.source == SOURCE_MY_TEAM:
                label += " — live"
            elif resolution.is_stale:
                label += f" — as of GW{resolution.source_gw}"
            return total, label

        details = get_entry_details(team_id)
        if details:
            # last_deadline_value is the squad's selling value with the bank
            # EXCLUDED — reading it alone under-reports the budget by whatever
            # is banked, which is money the optimizer is then never allowed to spend.
            raw_value = details.get("last_deadline_value", 0)
            raw_bank = details.get("last_deadline_bank", 0) or 0
            if raw_value > 0:
                return (raw_value + raw_bank) / 10.0, "from last deadline value"
    except Exception:
        _logger.warning("Could not resolve budget for team %s", team_id, exc_info=True)

    return 100.0, "default (could not fetch)"
