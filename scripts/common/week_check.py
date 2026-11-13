"""My Week — the pre-deadline checklist, as data.

The app answers every pre-deadline question and answers each one on a different
page: the deadline and squad on Home, fitness on Availability, the XI and
captain on Fixture Projections, free transfers on Transfer Suggestions, the chip
call on Chip Strategy, prices on Price Changes. Nothing said *which of those
needs attention this week*, so the only way to learn there was nothing to do was
to visit all six.

This module turns that round into a list of :class:`WeekItem`. It is pure and
Streamlit-free, and it **fetches nothing** -- every builder takes data the caller
has already resolved, which is what makes the whole checklist testable against
frames rather than against a live account.

**An all-clear must be derived, never defaulted.** "Nothing to do" and "we could
not check" render identically on a card and are opposite claims, so
:data:`STATUS_UNKNOWN` is a first-class status, every builder returns it when its
inputs are missing, and each item carries the ``evidence`` its verdict was drawn
from. :func:`scripts.common.data_validation.check_week_plan` fires on an ``ok``
with no evidence behind it, which is the one failure unique to a checklist.

Nothing here computes a projection: every number is one the engine already
produced. If a value looks wrong on this page it is wrong on the page that owns
it too.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from scripts.common.display_names import display_of
from scripts.common.injury_helpers import ABSENCE_FROM_NEWS, absence_estimate

#: Statuses, weakest first. ``unknown`` sorts above ``ok`` deliberately: a
#: question we could not answer deserves more of a reader's attention than one
#: answered "fine", and sorting it below would let an outage hide under a row of
#: green ticks.
STATUS_OK = "ok"
STATUS_UNKNOWN = "unknown"
STATUS_WARN = "warn"
STATUS_ACT = "act"

STATUS_ORDER = (STATUS_OK, STATUS_UNKNOWN, STATUS_WARN, STATUS_ACT)

#: Statuses that mean "this needs you", for the headline count.
ATTENTION = (STATUS_WARN, STATUS_ACT)

#: Below this many expected points a transfer plan is noise rather than advice --
#: the projections underneath carry an MAE around 2.5 points a player. The same
#: display floor as `chip_strategy.MIN_INTERESTING_GAIN` and the Waiver Wire's
#: `RENTAL_MIN_GAIN`, and for the same reason.
MIN_INTERESTING_GAIN = 2.0

#: Below this, a lineup change is not worth telling anybody about: the
#: projections underneath carry an MAE around 2.5 points a player, so a
#: half-point bench swap is noise. Same reasoning as `MIN_INTERESTING_GAIN`.
LINEUP_MIN_GAIN = 1.0

#: A deadline inside this many hours is today's problem, not next week's.
DEADLINE_SOON_HOURS = 24.0

#: Free transfers accumulate to five; at the cap the next gameweek's accrual is
#: forfeited, which is a quiet cost worth naming.
MAX_BANKED_FREE_TRANSFERS = 5

#: FPL status codes that mean the player is not in the squad at all, as opposed
#: to carrying a doubt.
_OUT_STATUSES = {"i", "s", "u", "n"}

#: Navigation targets. A page label alone is ambiguous -- Availability lives
#: under FPL App Home while Transfer Suggestions is Classic -- and the strip's
#: "Open" buttons drive `main.py`'s `_pending_nav_section` / `_pending_nav_page`,
#: which need both. The labels must match `main.SECTIONS` exactly, two spaces
#: after the emoji included; `test_every_item_points_at_a_real_page` asserts it,
#: because a renamed page would otherwise turn every button into a no-op.
SECTION_FPL = "\u26bd  FPL App Home"
SECTION_CLASSIC = "\U0001f3c6  Classic"


@dataclass(frozen=True)
class WeekItem:
    """One row of the checklist.

    ``evidence`` is not decoration: it is what distinguishes a derived all-clear
    from a defaulted one, and the validation check reads it.
    """

    key: str
    title: str
    status: str
    verdict: str
    detail: str = ""
    page: str = ""
    section: str = SECTION_CLASSIC
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def needs_attention(self) -> bool:
        return self.status in ATTENTION


def _unknown(key: str, title: str, why: str, page: str = "",
             section: str = SECTION_CLASSIC) -> WeekItem:
    """The honest answer when an input is missing.

    Carries no evidence by construction -- there was none -- which is exactly
    what stops it being mistaken for an ``ok``.
    """
    return WeekItem(key=key, title=title, status=STATUS_UNKNOWN,
                    verdict=why, page=page, section=section)


def worst_status(items: Sequence[WeekItem]) -> str:
    """The headline status: the most urgent one present."""
    worst = STATUS_OK
    for item in items or []:
        if STATUS_ORDER.index(item.status) > STATUS_ORDER.index(worst):
            worst = item.status
    return worst


def attention_items(items: Sequence[WeekItem]) -> List[WeekItem]:
    """The rows a manager has to do something about, most urgent first."""
    flagged = [i for i in items or [] if i.needs_attention]
    return sorted(flagged, key=lambda i: -STATUS_ORDER.index(i.status))


# --------------------------------------------------------------------------
# The builders. Each takes resolved data and returns one item; none fetches.
# --------------------------------------------------------------------------
def build_deadline_item(deadlines: Optional[Sequence[datetime]],
                        target_gw: Optional[int],
                        gameweek_started: bool = False,
                        now: Optional[datetime] = None) -> WeekItem:
    """When the next deadline is, and whether it is today.

    ``deadlines`` is :func:`analytics.gameweek_deadlines`'s list of upcoming
    deadlines -- real calendar dates rather than a weekly offset, because the
    season has international breaks and "Fri 6:00 AM" has already been read as
    *this* Friday when it meant thirteen days out.
    """
    key, title, page = "deadline", "Deadline", ""
    if not deadlines:
        return _unknown(key, title,
                        "Could not read the gameweek calendar, so the deadline "
                        "is unknown.", page)

    now = now or datetime.now(timezone.utc)
    upcoming = sorted(d for d in deadlines if d > now)
    if not upcoming:
        return _unknown(key, title,
                        "No deadline left in the calendar — the season looks "
                        "finished.", page)

    when = upcoming[0]
    hours = (when - now).total_seconds() / 3600.0
    evidence = {"deadline": when.isoformat(), "hours_left": round(hours, 2),
                "gameweek": target_gw, "gameweek_started": bool(gameweek_started)}

    if gameweek_started:
        # Mid-gameweek there is nothing to change, and saying so is the useful
        # answer rather than counting down to a deadline that is not the one
        # the squad on screen belongs to.
        return WeekItem(key, title, STATUS_OK,
                        "GW%s is under way — the next deadline is in %s."
                        % (target_gw, _hours_phrase(hours)),
                        detail="Nothing can be changed until it passes.",
                        page=page, evidence=evidence)

    status = STATUS_WARN if hours <= DEADLINE_SOON_HOURS else STATUS_OK
    return WeekItem(key, title, status,
                    "GW%s deadline in %s." % (target_gw, _hours_phrase(hours)),
                    page=page, evidence=evidence)


def _hours_phrase(hours: float) -> str:
    if hours < 1:
        return "under an hour"
    if hours < 24:
        return "%d hour%s" % (int(hours), "" if int(hours) == 1 else "s")
    days = int(hours // 24)
    return "%d day%s" % (days, "" if days == 1 else "s")


def build_squad_item(resolution: Any) -> WeekItem:
    """Which fifteen every row below is talking about.

    **Informational, never a warning.** Between deadlines without credentials
    the squad is *always* last deadline's -- FPL publishes no pending squad --
    so flagging that made one row permanently yellow while saying nothing a
    manager could act on. The caveat still travels with the numbers (it is in
    this row's detail, and the strip repeats it in its header); what is
    actionable lives in the lineup and transfer rows instead.
    """
    key, title, page = "squad", "Your squad", "\U0001f504  Transfer Suggestions"
    if resolution is None or not getattr(resolution, "ok", False):
        return _unknown(key, title,
                        "Your squad could not be resolved, so nothing below it "
                        "could be checked.", page)

    picks = list(getattr(resolution, "picks", []) or [])
    source = getattr(resolution, "source", "")
    chip = getattr(resolution, "active_chip", None)
    stale = bool(getattr(resolution, "is_stale", False))
    source_gw = getattr(resolution, "source_gw", None)
    evidence = {"source": source, "players": len(picks), "stale": stale,
                "source_gw": source_gw,
                "target_gw": getattr(resolution, "target_gw", None),
                "active_chip": chip}

    bits = []
    if chip:
        bits.append("**%s** is active." % chip)
    if stale:
        bits.append("As at the GW%s deadline — FPL does not publish a pending "
                    "squad without a connected account." % source_gw)
    verdict = ("%d players, as at the GW%s deadline." % (len(picks), source_gw)
               if stale else "%d players, live from your account." % len(picks))
    return WeekItem(key, title, STATUS_OK, verdict, detail=" ".join(bits),
                    page=page, evidence=evidence)


def squad_availability(squad_df: Optional[pd.DataFrame],
                       availability_df: Optional[pd.DataFrame],
                       deadlines=None, now=None) -> Optional[List[dict]]:
    """One row per squad player carrying a doubt, or ``None`` if unknowable.

    Joined on ``Player_Code`` where both frames publish it and on the element id
    otherwise -- never on a name. ``code`` is the only identifier both FPL APIs
    agree on, and a tenth of the pool is numbered differently between them.

    Each row reports **how long and on whose authority**: `absence_estimate`
    distinguishes a duration the news states from one inferred from a chance
    bucket, and only the first says anything about which gameweeks. Live, of
    roughly 200 players carrying an estimate only ~20 are stated, so the hedge is
    the common case and must be rendered as one.
    """
    if (squad_df is None or squad_df.empty
            or availability_df is None or availability_df.empty):
        return None

    join = None
    for col in ("Player_Code", "Player_ID"):
        if col in squad_df.columns and col in availability_df.columns:
            join = col
            break
    if join is None:
        return None

    keys = pd.to_numeric(squad_df[join], errors="coerce").dropna().astype(int)
    avail_keys = pd.to_numeric(availability_df[join], errors="coerce")
    mine = availability_df[avail_keys.isin(set(keys))]
    if mine.empty:
        return None

    names = {}
    if "Display_Name" in squad_df.columns:
        names = {int(k): n for k, n in zip(
            pd.to_numeric(squad_df[join], errors="coerce").fillna(-1).astype(int),
            squad_df["Display_Name"].astype(str)) if k is not None}

    out = []
    for _, row in mine.iterrows():
        status = str(row.get("Status") or "a")
        play_pct = pd.to_numeric(row.get("PlayPct"), errors="coerce")
        if status == "a" and (pd.isna(play_pct) or play_pct >= 100):
            continue
        gws, basis = absence_estimate(row.get("News"), play_pct, status,
                                      deadlines=deadlines, now=now)
        ident = int(pd.to_numeric(row.get(join), errors="coerce"))
        out.append({
            # The squad frame's name wins because it came from the pool, which
            # owns display names; `display_of` resolves the availability row's
            # own columns otherwise, NaN handling included.
            "name": names.get(ident) or display_of(row),
            "status": status,
            "play_pct": (None if pd.isna(play_pct) else float(play_pct)),
            "out_for": gws,
            "stated": basis == ABSENCE_FROM_NEWS,
            "news": str(row.get("News") or ""),
            "unavailable": status in _OUT_STATUSES or (
                not pd.isna(play_pct) and float(play_pct) <= 0),
        })
    return out


def build_availability_item(flags: Optional[Sequence[dict]]) -> WeekItem:
    """Who in the fifteen is hurt, and for how long."""
    key, title, page = "availability", "Fitness", "🚦  Availability"
    section = SECTION_FPL
    if flags is None:
        return _unknown(key, title,
                        "Could not read FPL's availability data, so injuries "
                        "were not checked.", page, section)

    out = [f for f in flags if f.get("unavailable")]
    doubt = [f for f in flags if not f.get("unavailable")]
    evidence = {"flagged": len(flags), "out": len(out), "doubtful": len(doubt),
                "players": [f.get("name") for f in flags]}

    if not flags:
        return WeekItem(key, title, STATUS_OK,
                        "Every player is fully available.",
                        page=page, section=section, evidence={"flagged": 0, "checked": True})

    bits = []
    if out:
        bits.append("%d out" % len(out))
    if doubt:
        bits.append("%d doubtful" % len(doubt))
    worst = sorted(flags, key=lambda f: (not f.get("unavailable"),
                                         f.get("play_pct") if f.get("play_pct") is not None else 101))
    lead = worst[0]
    detail = "%s — %s" % (lead.get("name"), _absence_phrase(lead))
    return WeekItem(key, title, STATUS_ACT if out else STATUS_WARN,
                    " · ".join(bits) + " in your fifteen.",
                    detail=detail, page=page, section=section, evidence=evidence)


def _absence_phrase(flag: dict) -> str:
    """How long he is out, never asserting more than the source said."""
    gws = flag.get("out_for")
    if not gws:
        pct = flag.get("play_pct")
        return ("%d%% chance to play" % pct) if pct is not None else "a doubt"
    if flag.get("stated"):
        return "out ~%d GW (return date stated)" % gws
    return "out ~%d GW (inferred, no stated return)" % gws


def build_fixtures_item(counts: Optional[pd.DataFrame],
                        squad_team_ids: Optional[Iterable[int]],
                        gameweeks: Optional[Sequence[int]],
                        target_gw: Optional[int] = None,
                        team_names: Optional[Dict[int, str]] = None) -> WeekItem:
    """Blanks and doubles among the clubs you actually own.

    ``club_fixture_counts`` returns an empty frame when the fixture list could
    not be read, and that must never render as twenty blanking clubs -- so an
    empty frame is *unknown*, not "no blanks". Equally, finding nothing is the
    normal state for the first half of a season (blanks and doubles appear once
    cup ties are rescheduled), so the all-clear says through which gameweek it
    looked rather than implying the season is flat.
    """
    key, title, page = "fixtures", "Fixtures", "📅  Gameweek Fixtures"
    section = SECTION_FPL
    ids = {int(t) for t in (squad_team_ids or []) if pd.notna(t)}
    if counts is None or counts.empty or not ids or not gameweeks:
        return _unknown(key, title,
                        "Could not read the fixture list, so blanks and doubles "
                        "were not checked.", page, section)

    names = team_names or {}
    blanks, doubles = {}, {}
    for gw in gameweeks:
        if gw not in counts.columns:
            continue
        col = counts[gw]
        for tid in ids:
            if tid not in col.index:
                continue
            n = int(col.loc[tid])
            label = names.get(tid, str(tid))
            if n == 0:
                blanks.setdefault(int(gw), []).append(label)
            elif n >= 2:
                doubles.setdefault(int(gw), []).append(label)

    through = max(int(g) for g in gameweeks)
    evidence = {"blanks": blanks, "doubles": doubles, "through": through,
                "clubs": len(ids)}

    if not blanks and not doubles:
        return WeekItem(key, title, STATUS_OK,
                        "No blank or double for your clubs through GW%d." % through,
                        detail="Normal for this stage of the season — they appear "
                               "once cup and European ties are rescheduled.",
                        page=page, section=section, evidence=evidence)

    parts = []
    if blanks:
        first = min(blanks)
        parts.append("%d blank%s from GW%d (%s)"
                     % (sum(len(v) for v in blanks.values()),
                        "" if sum(len(v) for v in blanks.values()) == 1 else "s",
                        first, ", ".join(sorted(blanks[first]))))
    if doubles:
        first = min(doubles)
        parts.append("%d double%s from GW%d (%s)"
                     % (sum(len(v) for v in doubles.values()),
                        "" if sum(len(v) for v in doubles.values()) == 1 else "s",
                        first, ", ".join(sorted(doubles[first]))))

    # A blank in the gameweek you are picking is this week's problem; one four
    # weeks out is information.
    imminent = target_gw is not None and int(target_gw) in blanks
    return WeekItem(key, title, STATUS_ACT if imminent else STATUS_WARN,
                    " · ".join(parts) + ".", page=page, section=section, evidence=evidence)


def build_transfers_item(free_transfers: Optional[dict],
                         bank: Optional[float] = None,
                         plan: Optional[dict] = None,
                         made_this_week: Optional[int] = None) -> WeekItem:
    """What is in hand, what the best move is worth, and whether you made it.

    **The alert is "a move is available and you have not made one"**, not "a
    move is available" -- otherwise the row keeps flagging after you have acted,
    which is the complaint that got the squad row demoted one line up.

    ``made_this_week`` comes from FPL's own transfer log, which does publish per
    gameweek (verified live: 18 rows across five gameweeks for one entry). A
    ``None`` means we could not tell, which is a weaker claim and renders as a
    warning rather than an action.

    ``free_transfers`` is `transfers.resolve_free_transfers`'s dict, which
    reports *which* of its three lanes answered -- a number reconstructed by
    replay is not the same claim as one FPL stated, and the row says so.
    """
    key, title, page = "transfers", "Transfers", "\U0001f504  Transfer Suggestions"
    if not free_transfers:
        return _unknown(key, title,
                        "Could not work out how many free transfers you have.",
                        page)

    count = int(free_transfers.get("count") or 0)
    source = free_transfers.get("source") or ""
    gain = None if not plan else plan.get("gain_net")
    gain = None if gain is None else float(gain)
    evidence = {"free_transfers": count, "source": source, "bank": bank,
                "made_this_week": made_this_week,
                "plan_gain": gain,
                "plan_changes": (None if not plan else plan.get("n_changes"))}

    held = "%d free transfer%s" % (count, "" if count == 1 else "s")
    if bank is not None:
        held += " · £%.1fm in the bank" % bank

    if made_this_week:
        return WeekItem(key, title, STATUS_OK,
                        "%d transfer%s made this gameweek · %s."
                        % (made_this_week, "" if made_this_week == 1 else "s", held),
                        detail=("A further move is worth +%.1f pts." % gain
                                if gain and gain >= MIN_INTERESTING_GAIN else ""),
                        page=page, evidence=evidence)

    if gain is not None and gain >= MIN_INTERESTING_GAIN:
        changes = int((plan or {}).get("n_changes") or 0)
        unknown_whether = made_this_week is None
        notes = ["%s." % held]
        if unknown_whether:
            notes.append("FPL has published no transfer for this gameweek; it "
                         "may not until the deadline.")
        elif source == "replay":
            notes.append("Free transfers reconstructed from your history.")
        return WeekItem(
            key, title,
            STATUS_WARN if unknown_whether else STATUS_ACT,
            "No transfer made yet — the best use of %d is worth +%.1f pts."
            % (changes, gain),
            detail=" ".join(notes), page=page, evidence=evidence)

    status = STATUS_OK
    detail = ("Reconstructed from your transfer history — FPL does not publish "
              "the count without a connected account."
              if source == "replay" else "")
    verdict = held + "."
    if count >= MAX_BANKED_FREE_TRANSFERS:
        status = STATUS_WARN
        detail = ("You are at the %d-transfer cap, so this week's accrual is "
                  "lost if you make none." % MAX_BANKED_FREE_TRANSFERS)
    elif gain is not None:
        verdict = held + " · no transfer gains more than %.1f pts." % MIN_INTERESTING_GAIN
    return WeekItem(key, title, status, verdict, detail=detail, page=page,
                    evidence=evidence)


def build_lineup_item(optimal_total: Optional[float] = None,
                      set_total: Optional[float] = None,
                      swap: Optional[dict] = None,
                      captain: Optional[str] = None,
                      set_captain: Optional[str] = None,
                      confirmed: bool = False,
                      stale: bool = True) -> WeekItem:
    """Whether the eleven you have **set** is the eleven you should have.

    This used to report the best XI's total and compare captains, which was a
    statement about the squad rather than about anything the manager had done.
    The question that matters before a deadline is *"is my lineup right"* -- a
    bench player projected above a starter, or the armband on the wrong man --
    and that needs the lineup FPL has stored, which `pick_position` carries.

    ``confirmed`` is the manager saying they have been in and set it. The app
    cannot see a pre-deadline lineup without credentials, so it must be able to
    be told; the confirmation is tied to the squad's membership, so a transfer
    re-arms this row automatically. See `scripts/common/lineup_confirm.py`.
    """
    key, title, page = "lineup", "Your XI", "\U0001f4c5  Fixture Projections"
    if optimal_total is None:
        return _unknown(key, title,
                        "No projection was available for your squad, so the XI "
                        "could not be checked.", page)

    gain = (None if set_total is None
            else max(0.0, float(optimal_total) - float(set_total)))
    captain_wrong = bool(set_captain and captain and set_captain != captain)
    evidence = {"optimal_total": round(float(optimal_total), 2),
                "set_total": (None if set_total is None else round(float(set_total), 2)),
                "gain": (None if gain is None else round(gain, 2)),
                "captain": captain, "set_captain": set_captain,
                "captain_wrong": captain_wrong, "confirmed": bool(confirmed),
                "stale": bool(stale)}

    if confirmed:
        return WeekItem(key, title, STATUS_OK,
                        "You have set this gameweek's XI.",
                        detail=("Projected %.1f pts. This row re-arms itself if "
                                "you transfer." % float(optimal_total)),
                        page=page, section=SECTION_CLASSIC, evidence=evidence)

    # Nothing to compare against: report the best XI and say so, rather than
    # implying the stored one has been checked.
    if set_total is None:
        return WeekItem(key, title, STATUS_OK,
                        "Best XI projects %.1f pts · captain %s."
                        % (float(optimal_total), captain or "—"),
                        detail="Your saved lineup could not be read, so it is "
                               "not compared.",
                        page=page, evidence=evidence)

    reasons = []
    if gain and gain >= LINEUP_MIN_GAIN and swap:
        reasons.append("start **%s** over **%s** (+%.1f)"
                       % (swap.get("in_name"), swap.get("out_name"), gain))
    elif gain and gain >= LINEUP_MIN_GAIN:
        reasons.append("your bench projects %.1f pts more than your XI" % gain)
    if captain_wrong:
        reasons.append("the armband is on **%s**, the model prefers **%s**"
                       % (set_captain, captain))

    if not reasons:
        return WeekItem(key, title, STATUS_OK,
                        "Your XI is the best of your fifteen (%.1f pts)."
                        % float(optimal_total),
                        detail=("Captain %s." % captain) if captain else "",
                        page=page, evidence=evidence)

    detail = ("Based on your GW-deadline lineup, which is the latest FPL "
              "publishes without a connected account." if stale else "")
    return WeekItem(key, title, STATUS_ACT,
                    "Your XI is not optimal — " + "; ".join(reasons) + ".",
                    detail=detail, page=page, evidence=evidence)


def build_chips_item(status: Optional[dict],
                     values: Optional[Dict[str, float]] = None,
                     labels: Optional[Dict[str, str]] = None) -> WeekItem:
    """What is playable now, what it is worth this week, and when it expires."""
    key, title, page = "chips", "Chips", "🎯  Chip Strategy"
    if not status:
        return _unknown(key, title,
                        "Could not read FPL's chip rules, so nothing is known "
                        "about what you can play.", page)

    labels = labels or {}
    avail = list(status.get("available") or [])
    later = list(status.get("available_later") or [])
    states = status.get("states") or {}
    gws_left = [s.gws_left for name, s in states.items()
                if name in avail and getattr(s, "gws_left", None) is not None]
    left = min(gws_left) if gws_left else None
    evidence = {"available": avail, "available_later": later, "gws_left": left,
                "values": {k: round(float(v), 2) for k, v in (values or {}).items()}}

    if not avail:
        return WeekItem(key, title, STATUS_OK,
                        ("No chip is playable this gameweek."
                         if later else "Every chip has been played."),
                        detail=("Your next one opens later in the season."
                                if later else ""),
                        page=page, evidence=evidence)

    named = ", ".join(labels.get(c, c) for c in avail)
    verdict = "%s playable" % named
    if left is not None:
        verdict += " · %d gameweek%s before they expire" % (left, "" if left == 1 else "s")

    best = None
    if values:
        best = max(values.items(), key=lambda kv: kv[1])
    detail = ""
    if best and best[1] >= MIN_INTERESTING_GAIN:
        detail = ("%s is worth about %.1f pts this week — see Chip Strategy for "
                  "whether a later week is worth more."
                  % (labels.get(best[0], best[0]), best[1]))

    status_code = STATUS_WARN if (left is not None and left <= 3) else STATUS_OK
    return WeekItem(key, title, status_code, verdict + ".", detail=detail,
                    page=page, evidence=evidence)


def build_prices_item(pressure: Optional[dict]) -> WeekItem:
    """What tonight's price moves do to your squad value."""
    key, title, page = "prices", "Prices", "💰  Price Changes"
    section = SECTION_FPL
    if not pressure:
        return _unknown(key, title,
                        "Could not read FPL's price-change data for your squad.",
                        page, section)

    falling, rising = int(pressure.get("falling", 0)), int(pressure.get("rising", 0))
    evidence = dict(pressure)
    nearest = pressure.get("nearest") or []

    if falling:
        lead = next((n for n in nearest if n.get("state") == "fall"), None)
        detail = ("%s at %+.0f%%" % (lead["name"], lead["pct"])) if lead else ""
        return WeekItem(key, title, STATUS_WARN,
                        "%d of your players %s due to fall tonight."
                        % (falling, "is" if falling == 1 else "are"),
                        detail=detail, page=page, section=section, evidence=evidence)

    detail = ""
    if nearest:
        lead = nearest[0]
        detail = "Closest: %s at %+.0f%%." % (lead["name"], lead["pct"])
    verdict = "No player in your squad is past a price threshold."
    if rising:
        verdict = "%d of your players %s due to rise — free money." % (
            rising, "is" if rising == 1 else "are")
    return WeekItem(key, title, STATUS_OK, verdict, detail=detail,
                    page=page, section=section, evidence=evidence)


def build_opponent_item(opponent_name: Optional[str] = None,
                        mine: Optional[float] = None,
                        theirs: Optional[float] = None,
                        win_prob: Optional[float] = None,
                        stale: bool = True) -> Optional[WeekItem]:
    """Who you play this gameweek, and how the two squads project.

    Returns **None** in a total-points league -- the caller simply does not call
    it -- because "there is no opponent" is a property of the league rather than
    a question we failed to answer, and an *unknown* row would claim otherwise.
    Where the league is H2H but the fixture or a projection cannot be resolved,
    that **is** an unknown and says so.

    Status is always ``ok``: this is context for the rows above it, not a task.
    Their squad is last deadline's -- `my-team` is the user's own credential --
    so the row says so rather than implying a live view of a rival.
    """
    key, title = "opponent", "This week"
    page, section = "\U0001f4c5  Fixture Projections", SECTION_CLASSIC
    if not opponent_name or mine is None or theirs is None:
        return _unknown(key, title,
                        "Could not resolve this gameweek's head-to-head fixture.",
                        page, section)

    evidence = {"opponent": opponent_name, "mine": round(float(mine), 2),
                "theirs": round(float(theirs), 2),
                "win_prob": (None if win_prob is None else round(float(win_prob), 4)),
                "stale": bool(stale)}
    verdict = "You play %s — %.1f projected against %.1f" % (
        opponent_name, float(mine), float(theirs))
    if win_prob is not None:
        verdict += ", %.0f%% to win" % (100.0 * float(win_prob))
    detail = ("Their squad is last deadline's, so this assumes they make no "
              "transfers." if stale else "")
    return WeekItem(key, title, STATUS_OK, verdict + ".", detail=detail,
                    page=page, section=section, evidence=evidence)


#: The checklist's order, declared once so the page and its tests agree.
ITEM_ORDER = ("deadline", "opponent", "squad", "availability", "fixtures",
              "lineup", "transfers", "chips", "prices")


def build_week_checklist(*, deadline=None, opponent=None, squad=None,
                         availability=None, fixtures=None, lineup=None,
                         transfers=None, chips_item=None,
                         prices=None) -> List[WeekItem]:
    """Assemble the items in :data:`ITEM_ORDER`, dropping none.

    A builder that could not answer still contributes its ``unknown`` row --
    silently omitting it is how a checklist comes to read as complete while a
    question went unasked.
    """
    by_key = {}
    for item in (deadline, opponent, squad, availability, fixtures, lineup,
                 transfers, chips_item, prices):
        if item is not None:
            by_key[item.key] = item
    return [by_key[k] for k in ITEM_ORDER if k in by_key]
