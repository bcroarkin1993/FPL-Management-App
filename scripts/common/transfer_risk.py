"""
Transfer risk — how likely is this player to stop being an FPL asset, and how
much of the remaining season would that cost?

The model this exists to fix ranked Ollie Watkins as a top draft pick on the
assumption he would play the season for Aston Villa.  He moved to Al-Hilal.
Projected points were never discounted by the probability the player is still in
the Premier League when those points are scored.

The multiplier mirrors ``injury_helpers.injury_multiplier``:

    transfer_multiplier = max(FLOOR, 1 - risk * exposure)

``risk`` is how credible the move is, weighted by where he is going.
``exposure`` is how much of the rest of the season a completed move would cost,
which falls to zero once no window remains — so the discount switches itself off
after the January deadline without anybody toggling anything.

This module is pure: no Streamlit, no network, plain ``logging``.  It is
imported by GitHub Actions (``waiver_alerts.py``), so it must not reach for
``error_helpers`` (which imports Streamlit) or ``player_matching`` (likewise).
"""

import logging
import re
from datetime import date, datetime
from typing import NamedTuple

import pandas as pd

from scripts.common.text_helpers import TEAM_FULL_TO_SHORT, canonical_normalize
from scripts.common.transfer_odds import (ODDS_COLUMNS, blend_odds_risk,
                                          odds_age_days, odds_age_weight)

_logger = logging.getLogger(__name__)

# Floor on the multiplier, matching injury_helpers.INJURY_FLOOR.  Never 0: the
# Initial Squad ILP filters candidates on ``score_col > 0`` (optimization.py:189),
# so a zero would evict the player rather than merely price him down.
TRANSFER_FLOOR = 0.10

# --- Destination weights ------------------------------------------------------
# The dividing line is Premier League membership, not geography.  A season-long
# loan to a Championship club scores you exactly as many points as a move to
# Saudi Arabia: zero.  Live bootstrap examples of both, same status code:
#   'u' Watson | Has joined Leicester City on loan for the rest of the season
#   'u' Reijnders | Has joined Al Qadsiah permanently
WEIGHT_LEAVES_PL = 1.00
# Still in the game: in Draft you simply keep him, so this costs you nothing.
# It is deliberately *neutral* rather than a small discount, because the sign is
# genuinely ambiguous — Cody Gakpo leaving a Liverpool front line for a starting
# role at Spurs is plausibly an upgrade, and a 20% discount asserted a direction
# the evidence does not support.  The move is still surfaced as a flag so it can
# be judged by eye.
WEIGHT_INTRA_PL = 0.0
# Destination named but unparsed.  Most rumoured exits leave the league, so a
# parse failure must not read as safety.
WEIGHT_UNKNOWN = 0.60

# --- News scoring -------------------------------------------------------------
# Tiers are ordered by how much the language actually commits to.  "Undergoes a
# medical" is a different claim from "is being linked with".
TIER_A = 0.85
TIER_B = 0.50
TIER_C = 0.25

_TIER_A_PATTERNS = [
    r"\bmedical\b", r"\bhere we go\b", r"\bagree(?:d|s)?\s+(?:a\s+)?deal\b",
    r"\bdeal\s+(?:is\s+)?(?:agreed|done)\b", r"\bpersonal terms\s+agreed\b",
    r"\bagree(?:d|s)?\s+personal terms\b", r"\bbid\s+accepted\b",
    r"\baccept(?:ed|s)?\s+(?:a\s+)?(?:£|\$|€)?[\d.]*\s*m?\s*(?:bid|offer)\b",
    r"\bset\s+to\s+(?:sign|join|complete|move)\b", r"\bcompletes?\s+(?:a\s+)?move\b",
    r"\bunveiled\b", r"\bsigns?\s+for\b", r"\bjoins?\b", r"\bsold\s+to\b",
    r"\bagree(?:d|s)?\s+to\s+(?:a\s+)?sale\b", r"\bexit\s+confirmed\b",
    # "Al Hilal agree £51million Ollie Watkins transfer" — a fee agreed is a
    # Tier A commitment even though the word "deal" never appears.
    r"\bagree(?:d|s)?\b.{0,30}\b(?:transfer|move|fee|switch)\b",
]
# Tier B is reserved for *player-side* commitment — the player moving, not
# somebody wanting him. A bid or an offer is the selling club's problem, is
# reported constantly, and mostly comes to nothing: pricing it at Tier B
# discounted Bruno Fernandes by a third off a Galatasaray offer that United
# publicly rejected while extending his contract.
_TIER_B_PATTERNS = [
    r"\bin talks\b", r"\bhold(?:ing)?\s+talks\b", r"\bclose to\b",
    r"\bnegotiat", r"\btransfer request\b", r"\bpush(?:ing)?\s+to\s+sign\b",
    r"\bwants?\s+to\s+(?:leave|join)\b", r"\bopen\s+to\s+(?:a\s+)?(?:move|exit)\b",
    r"\bset\s+to\s+(?:leave|depart)\b", r"\bbound\s+for\b",
]
_TIER_C_PATTERNS = [
    # Club-side interest: real reporting, weak evidence he actually goes.
    r"\bbid\b", r"\boffer\b", r"\bapproach(?:ed)?\b", r"\bswoop\b",
    r"\bexit\b", r"\bdeparture\b", r"\bfarewell\b",
    r"\blink(?:ed|s)?\b", r"\binterest(?:ed)?\b", r"\bmonitor(?:ing)?\b",
    r"\beye(?:ing|d)?\b", r"\btarget(?:ing)?\b", r"\brumou?r\b", r"\bgossip\b",
    r"\bconsider(?:ing)?\b", r"\bcould\s+(?:leave|join|move)\b",
    r"\bwould\s+(?:leave|join)\b", r"\bspeculation\b",
]
# A denial caps the score at Tier C no matter what else the headline says.
# "Villa rule out Watkins sale" must not score as Tier A on the word "sale".
# A denial caps the headline at Tier C, and enough of them cap the player's whole
# score (see _STAY_SIGNAL_RISK_CAP).  "Contract talks" belongs here: the club
# tying a player down is the opposite of him leaving, and it was being read as
# Tier B "talks".
_NEGATION_PATTERNS = [
    r"\bnot for sale\b", r"\brule(?:d|s)?\s+out\b", r"\bden(?:y|ies|ied)\b",
    r"\breject(?:ed|s)?\b", r"\bturn(?:ed|s)?\s+down\b", r"\bknock(?:ed|s)?\s+back\b",
    r"\brebuff(?:ed|s)?\b", r"\bsnub(?:bed|s)?\b",
    r"\bsigns?\s+new\s+(?:deal|contract)\b", r"\bnew contract\b", r"\bnew deal\b",
    r"\bcontract talks\b", r"\bcontract extension\b", r"\bextend(?:s|ed)?\s+(?:his\s+)?(?:deal|contract|stay)\b",
    r"\bstay(?:s|ing)?\s+(?:at|put)\b", r"\bwill\s+not\s+leave\b",
    r"\bwon.?t\s+(?:sell|leave)\b", r"\bwill\s+(?:not|never)\s+sell\b",
    r"\bno\s+(?:plans|intention)\s+to\s+(?:sell|leave)\b", r"\bnot\s+entertain",
    r"\bcollapse[ds]?\b", r"\boff\b(?=.*\btransfer\b)", r"\bcall(?:ed|s)?\s+off\b",
    r"\bfails?\b",
    # A deal that fell through reads exactly like a deal that happened unless the
    # withdrawal is spelled out: "Monaco pull out of selling midfielder to
    # Chelsea in £47m deal" scored Tier A and discounted four Chelsea midfielders.
    r"\bpull(?:s|ed)?\s+out\b", r"\bwithdraw(?:s|n|ing)?\b",
    r"\bdeal\s+off\b", r"\boff\s+the\s+table\b",
    r"\bcool(?:s|ed|ing)?\s+(?:their\s+)?interest\b",
    r"\bend(?:s|ed)?\s+(?:their\s+)?(?:interest|pursuit)\b",
    r"\bmiss(?:es|ed)\s+out\b", r"\bpriced\s+out\b",
]

_TIERS = ((TIER_A, _TIER_A_PATTERNS), (TIER_B, _TIER_B_PATTERNS), (TIER_C, _TIER_C_PATTERNS))

# The arrival side of the same three tiers.  Written separately because the
# vocabulary genuinely differs: "completes the signing of" is the strongest
# possible inbound evidence and does not appear in the exit list at all, while
# "exit", "departure" and "sold to" are meaningless here.
_SIGNING_TIER_A_PATTERNS = [
    r"\bcomplete[sd]?\s+(?:the\s+)?(?:signing|move|deal|transfer)\b",
    r"\b(?:has|have)\s+signed\b", r"\bofficially\s+sign(?:ed|s)?\b",
    r"\bannounce[sd]?\s+(?:the\s+)?(?:signing|capture|arrival)\b",
    r"\bconfirm(?:s|ed)?\s+(?:the\s+)?(?:signing|capture|arrival)\b",
    r"\bunveil(?:ed|s)?\b", r"\bmedical\b", r"\bhere we go\b",
    r"\bsigns?\s+for\b", r"\bjoins?\b", r"\barriv(?:es|ed|al)\b",
    r"\bseal(?:s|ed)?\s+(?:a\s+)?(?:move|deal|transfer|switch|signing)\b",
    r"\bagree(?:d|s)?\s+(?:a\s+)?deal\b", r"\bdeal\s+(?:is\s+)?(?:agreed|done)\b",
    r"\bagree(?:d|s)?\s+(?:to\s+)?(?:personal\s+)?terms\b",
    r"\bfee\s+agreed\b", r"\bbid\s+accepted\b",
    r"\bset\s+to\s+(?:sign|join|complete)\b",
    r"\bagree(?:d|s)?\b.{0,30}\b(?:transfer|move|fee|switch)\b",
]
_SIGNING_TIER_B_PATTERNS = [
    r"\bin\s+talks\b", r"\bhold(?:ing)?\s+talks\b", r"\badvanced\s+talks\b",
    r"\bclose\s+to\b", r"\bnegotiat", r"\bfinali[sz](?:e|es|ing)\b",
    r"\bpush(?:ing)?\s+to\s+sign\b", r"\bconfident\s+of\s+(?:signing|landing)\b",
    r"\bsubmit(?:s|ted)?\s+(?:a\s+)?(?:bid|offer)\b", r"\bbid\s+(?:of|worth)\b",
    r"\bpersonal\s+terms\b",
]
_SIGNING_TIER_C_PATTERNS = [
    r"\blink(?:ed|s)?\b", r"\binterest(?:ed)?\b", r"\bmonitor(?:ing)?\b",
    r"\btarget(?:ing)?\b", r"\beye(?:ing|d)?\b", r"\bswoop\b", r"\bchase\b",
    r"\bbid\b", r"\boffer\b", r"\bapproach(?:ed)?\b", r"\brumou?r\b",
    r"\bgossip\b", r"\bconsider(?:ing)?\b", r"\bwant(?:s|ed)?\s+to\s+sign\b",
    r"\bmove\s+for\b", r"\bspeculation\b", r"\bshortlist(?:ed)?\b",
]
_SIGNING_TIERS = (
    (TIER_A, _SIGNING_TIER_A_PATTERNS),
    (TIER_B, _SIGNING_TIER_B_PATTERNS),
    (TIER_C, _SIGNING_TIER_C_PATTERNS),
)

# Headline evidence older than this is ignored outright.
NEWS_WINDOW_DAYS = 30
# Half-life of a headline's weight, in days.  Transfer stories go stale fast.
NEWS_HALF_LIFE_DAYS = 10.0
# Outlets needed for full confidence.  One outlet saying "medical" is a scoop or
# a mistake; four saying it is a fact.
FULL_CORROBORATION_OUTLETS = 4
# Share of the score a wholly uncorroborated story keeps.  Deliberately low:
# one outlet reporting a medical is a scoop or a mistake, and the cost of
# wrongly torching a good player's ranking is higher than the cost of a late
# discount on a real move.
CORROBORATION_FLOOR = 0.30
# When several outlets report the club refusing to sell, or extending him, that is
# real counter-evidence and not merely absence of evidence.  It cannot override a
# Tier A fact — a medical happens whatever the club said last week — but it caps
# everything weaker.
_MIN_STAY_SIGNALS_TO_CAP = 2
_STAY_SIGNAL_RISK_CAP = 0.10

# --- Transfer windows ---------------------------------------------------------
# Hardcoded and season-specific: THESE MUST BE UPDATED EACH SEASON.
# check_transfer_risk() warns when the latest close date is in the past.
#
# The Saudi row is the whole reason this is keyed by region.  The English window
# shut on 1 Sep 2026 but the Saudi one ran to 12 Oct, so a player could be sold
# out of a squad five gameweeks into a season whose window had "closed".
TRANSFER_WINDOWS = {
    "premier_league": [(date(2026, 6, 15), date(2026, 9, 1)),
                       (date(2027, 1, 1), date(2027, 2, 1))],
    "saudi":          [(date(2026, 7, 22), date(2026, 10, 12)),
                       (date(2027, 1, 1), date(2027, 2, 1))],
    "default":        [(date(2026, 6, 15), date(2026, 9, 1)),
                       (date(2027, 1, 1), date(2027, 2, 1))],
}
SEASON_END = date(2027, 5, 23)

# Saudi clubs are near-uniformly "Al-something", which also matches Al-Ahly
# (Egypt) and similar — acceptable, since every one of them is outside the PL.
# Matched against _norm()'d text, which has already stripped hyphens: "Al-Hilal"
# arrives as "alhilal", so the separator must be optional.
_SAUDI_PATTERNS = [
    r"\bsaudi\b",
    r"\bal ?(?:hilal|nassr|ittihad|ahli|qadsiah|ettifaq|shabab|taawoun|fateh|riyadh|khaleej|wehda)\b",
    r"\bpro league\b",
]

# Foreign clubs common in EPL exit stories.  Not exhaustive and does not need to
# be: an unmatched destination falls through to WEIGHT_UNKNOWN, which is already
# weighted toward "this is an exit".
_FOREIGN_CLUBS = [
    "real madrid", "barcelona", "atletico madrid", "atletico", "sevilla", "valencia",
    "real sociedad", "villarreal", "real betis", "getafe", "girona", "athletic bilbao",
    "bayern munich", "bayern", "borussia dortmund", "dortmund", "rb leipzig", "leipzig",
    "bayer leverkusen", "leverkusen", "eintracht frankfurt", "stuttgart", "wolfsburg",
    "hamburg", "juventus", "inter milan", "internazionale", "inter", "ac milan", "milan",
    "napoli", "roma", "as roma", "lazio", "atalanta", "fiorentina", "como", "torino",
    "paris saint-germain", "paris saint germain", "psg", "monaco", "marseille", "lyon",
    "lille", "nice", "rennes", "strasbourg", "ajax", "psv", "feyenoord", "benfica",
    "porto", "sporting", "sporting cp", "galatasaray", "fenerbahce", "besiktas",
    "trabzonspor", "konyaspor", "celtic", "rangers", "zenit", "shakhtar",
    "inter miami", "la galaxy", "lafc", "new england revolution", "toronto fc",
    "atlanta united", "seattle sounders", "orlando city", "mls",
]

# Non-PL English clubs get no special handling by name — they simply fail the
# "is this a Premier League club" test, which is the correct answer.
# The captured club name is bounded on both sides: length-capped, and stopped at
# the connectives a headline uses to keep going ("joined Tottenham *amid* Arsenal
# interest").  Unbounded, it swallowed the rest of the sentence and rendered a
# destination of "Tottenham amid Arsenal ...".
_JOINED_RE = re.compile(
    r"(?:has\s+)?(?:joined|returned to|departed(?:\s+the\s+club)?|left)\s+(?:to\s+)?"
    r"(?!(?:as|amid|after|with|from|in|to|for|on|and|the)\b)(.{2,40}?)"
    r"(?:\s+(?:permanently|on\s+loan|for\s+the\s+rest|as\s+a\s+free\s+agent|amid|after|"
    r"as|despite|with|from|in|following|but|while|and|on|to|for)\b|[.,;:!?]|$)",
    re.IGNORECASE,
)


_ALIAS_TO_CODE = {canonical_normalize(k): v for k, v in TEAM_FULL_TO_SHORT.items()}


def team_code(value):
    """Short club code from a full name *or* an already-short code.

    Frames disagree on which they carry: Rotowire and the availability table use
    codes ("MCI"), the bootstrap ``teams`` list uses names ("Man City"). Resolving
    both is what stops a player's own club being reported as his destination.
    """
    v = "" if value is None else str(value).strip()
    if not v:
        return None
    if v in TEAM_FULL_TO_SHORT:
        return TEAM_FULL_TO_SHORT[v]
    code = _ALIAS_TO_CODE.get(canonical_normalize(v))
    if code:
        return code
    if 2 <= len(v) <= 4 and v.isupper():
        return v
    return None


#: A fee at or above this (GBP millions) means the buying club intends to play
#: him. Useful in both directions: it is the strongest available evidence that an
#: incoming signing takes minutes off an incumbent.
BIG_FEE_GBP_M = 40.0

_FEE_RE = re.compile(
    r"(?:£|\$|€|eur|gbp|usd)\s?(\d+(?:\.\d+)?)\s*(m|million|bn|billion|k)?",
    re.IGNORECASE,
)


#: How near a player's name a fee must sit to be read as *his* fee, in characters.
_FEE_ATTRIBUTION_WINDOW = 45


def _fee_match_value(match):
    """Fee in millions from a regex match, or None if it is not a transfer fee."""
    try:
        value = float(match.group(1))
    except (TypeError, ValueError):
        return None
    unit = (match.group(2) or "").lower()
    if unit in ("bn", "billion"):
        value *= 1000.0
    elif unit not in ("m", "million"):
        # Transfer fees are always written with a magnitude ("£51m"). A bare
        # amount is something else — "£10 boots", a ticket price, a wage.
        return None
    if value <= 0 or value > 1000:
        return None
    return value


def parse_fee(text):
    """Largest transfer fee named in the text, in millions, or ``None``.

    Currencies are not converted — at this precision the distinction between a
    £50m and a €50m signing does not change any decision the model makes.
    """
    raw = "" if text is None else str(text)
    best = None
    for match in _FEE_RE.finditer(raw):
        value = _fee_match_value(match)
        if value is not None and (best is None or value > best):
            best = value
    return best


def parse_fee_for_player(text, player_name, other_surnames=None):
    """Fee attributable to *this* player, in millions, or ``None``.

    A transfer headline routinely prices somebody else: "Liverpool agree £123m
    Barcola deal as Gakpo decides to join Man City" names two players and one
    fee, and the fee is not Gakpo's. Attribution therefore requires the fee to be
    nearer this player's surname than to any other player in the pool.

    Same failure mode as reading a player's own club as his destination, and it
    matters more here — fee is the evidence for how big a role a signing takes.
    """
    raw = "" if text is None else str(text)
    if not raw:
        return None

    lowered = raw.lower()
    tokens = [t for t in _norm(player_name).split() if len(t) > 1]
    if not tokens:
        return None
    surname = tokens[-1]
    name_at = lowered.find(surname)
    if name_at < 0:
        return None

    rival_positions = []
    for other in other_surnames or ():
        if not other or other == surname:
            continue
        idx = lowered.find(other)
        if idx >= 0:
            rival_positions.append(idx)

    best = None
    for match in _FEE_RE.finditer(raw):
        value = _fee_match_value(match)
        if value is None:
            continue
        fee_at = match.start()
        own_distance = abs(fee_at - name_at)
        if own_distance > _FEE_ATTRIBUTION_WINDOW:
            continue
        if any(abs(fee_at - other_at) < own_distance for other_at in rival_positions):
            continue  # somebody else is closer to this number
        if best is None or value > best:
            best = value
    return best


def _norm(text) -> str:
    """Lowercase, accent-stripped, punctuation-free form for substring tests."""
    return canonical_normalize(text)


def _coerce_date(value):
    """Best-effort date from a datetime, date, pandas Timestamp or RFC-822 string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except (ValueError, TypeError):
        return None
    if parsed is None or pd.isna(parsed):
        return None
    try:
        return parsed.date()
    except AttributeError:
        return None


def build_ambiguous_tokens(names):
    """Name tokens that cannot identify a player on their own, within this pool.

    Derived from the pool itself rather than a hardcoded list, so it adapts to
    whoever is actually in the game.  Without it, a query for the mononym
    "Gabriel" collects news about every Gabriel in the league — which discounted
    Arsenal's Gabriel by 71% off thirteen outlets reporting on other players.
    """
    counts = {}
    for name in names or []:
        tokens = {t for t in _norm(name).split() if len(t) > 1}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
    return {t for t, c in counts.items() if c > 1}


def headline_mentions_player(headline, player_name, team=None, ambiguous=None) -> bool:
    """Is this headline actually about this player?

    Precision comes mostly from the query itself — the feed is fetched per player
    with the full name in quotes — so this is a guard against the loosely related
    items Google mixes in ("Exeter City transfer latest" comes back on a Watkins
    query), not the primary filter.

    Matching on the surname alone is deliberate: headlines overwhelmingly drop the
    first name ("Mateta set to sign for Nottingham Forest"), and requiring it
    counted a four-outlet done deal as a single-outlet rumour.

    But a surname shared with another player in the pool — or a mononym like
    "Gabriel" — proves nothing on its own, so those additionally require the full
    name or the club to appear.  This is the same rule the player matcher learned
    from attaching Cole Palmer's stats to Alex Palmer: a weak key needs a
    corroborating signal.
    """
    h = _norm(headline)
    p = _norm(player_name)
    if not h or not p:
        return False
    words = set(h.split())
    tokens = [t for t in p.split() if len(t) > 1]
    if not tokens:
        return False
    # Only a multi-token name is self-identifying. For a mononym the "full name"
    # *is* the bare surname, so this shortcut would skip the ambiguity guard below.
    if len(tokens) > 1 and p in h:
        return True

    surname = tokens[-1]
    if surname not in words:
        return False

    ambiguous = ambiguous or set()
    team_words = {t for t in _norm(team).split() if len(t) > 2} if team else set()
    team_mentioned = bool(team_words & words)

    # A shared or short surname needs the club (or the first name) to corroborate.
    if surname in ambiguous or len(surname) < 5 or len(tokens) == 1:
        if team_mentioned:
            return True
        return any(t in words for t in tokens[:-1])

    return True


def _recency_weight(published, today) -> float:
    """Exponential decay on headline age.  Undated headlines are treated as old."""
    pub = _coerce_date(published)
    if pub is None:
        return 0.5
    age_days = (today - pub).days
    if age_days < 0:
        age_days = 0
    if age_days > NEWS_WINDOW_DAYS:
        return 0.0
    return 0.5 ** (age_days / NEWS_HALF_LIFE_DAYS)


def _is_stay_signal(headline) -> bool:
    """Does this headline report the club refusing to sell, or tying him down?"""
    h = " " + _norm(headline) + " "
    return bool(h.strip()) and any(re.search(p, h) for p in _NEGATION_PATTERNS)


def _classify(headline, tiers) -> float:
    """Strongest tier in ``tiers`` the headline's language supports.

    A denial caps the result at Tier C: the story is real, the move is not.
    """
    h = " " + _norm(headline) + " "
    if not h.strip():
        return 0.0
    negated = any(re.search(p, h) for p in _NEGATION_PATTERNS)
    score = 0.0
    for tier_score, patterns in tiers:
        if any(re.search(p, h) for p in patterns):
            score = tier_score
            break
    if negated:
        score = min(score, TIER_C)
    return score


def classify_headline(headline) -> float:
    """Strongest tier score an *exit* headline's language supports."""
    return _classify(headline, _TIERS)


def classify_signing(headline) -> float:
    """Strongest tier score an *arrival* headline's language supports.

    A separate vocabulary from ``classify_headline()``, which is written for the
    question "is he leaving": its Tier A is about a player departing, and the
    single strongest inbound sentence there is -- "Villa complete signing of
    Nicolas Jackson" -- matches nothing in it and scored 0.0, so a confirmed
    arrival was dropped from the watchlist while a rumour survived.

    Calibrated to the same three tiers so the two sides stay comparable: A is
    language that commits, B is an active pursuit, C is interest.
    """
    return _classify(headline, _SIGNING_TIERS)


def pl_aliases(pl_teams):
    """Every spelling of a current Premier League club, plus their short codes.

    The bootstrap says "Nott'm Forest", "Man Utd" and "Spurs"; headlines say
    "Nottingham Forest", "Manchester United" and "Tottenham".  Comparing raw
    strings would read an intra-PL move as an exit — the expensive direction of
    error.  ``TEAM_FULL_TO_SHORT`` already maps every variant onto one code, so
    resolve through that and fall back to the bootstrap spelling for a promoted
    club the table has not caught up with yet.
    """
    codes = set()
    for team in pl_teams or []:
        code = TEAM_FULL_TO_SHORT.get(str(team).strip())
        if code:
            codes.add(code)
    names = {_norm(t) for t in (pl_teams or []) if _norm(t)}
    for alias, code in TEAM_FULL_TO_SHORT.items():
        if code in codes and _norm(alias):
            names.add(_norm(alias))
    return names, codes


def is_premier_league_club(name, pl_teams) -> bool:
    """Is this club in the current FPL bootstrap ``teams`` list?

    Membership is read from live data rather than a hardcoded league table, so it
    survives promotion and relegation with no maintenance — which is the whole
    point, because a Championship move costs exactly as much as a Saudi one.
    """
    n = _norm(name)
    if not n:
        return False
    names, codes = pl_aliases(pl_teams)
    if n in names:
        return True
    code = team_code(name)
    if code and code in codes:
        return True
    # Substring fallback for "joined Arsenal FC" style phrasing.
    return any(alias and (alias in n or n in alias) for alias in names if len(alias) > 4)


def parse_destination(text, pl_teams, exclude_team=None):
    """Extract the destination club from a headline and weight it.

    Returns ``(club_or_None, weight)``.

    ``exclude_team`` must be the player's *current* club, or the parse inverts on
    the commonest headline shape there is:
    "Al-Hilal agree deal to sign Aston Villa striker Ollie Watkins" names two
    clubs, and only one of them is where he is going.
    """
    raw = "" if text is None else str(text)
    n = _norm(raw)
    if not n:
        return None, WEIGHT_UNKNOWN

    # A free agent has no destination club yet, but he has still left the league.
    if re.search(r"free\s+agent", raw, re.IGNORECASE):
        return "Free agent", WEIGHT_LEAVES_PL

    # Explicit "has joined X" (the bootstrap's own phrasing) is the most reliable.
    joined = _JOINED_RE.search(raw)
    if joined:
        club = joined.group(1).strip()
        if club and _norm(club):
            weight = WEIGHT_INTRA_PL if is_premier_league_club(club, pl_teams) else WEIGHT_LEAVES_PL
            return club, weight

    if any(re.search(p, n) for p in _SAUDI_PATTERNS):
        return "Saudi Pro League", WEIGHT_LEAVES_PL

    for club in _FOREIGN_CLUBS:
        # Normalise the gazetteer too — "Paris Saint-Germain" is "paris saintgermain"
        # once punctuation is stripped, and would never match otherwise.
        needle = _norm(club)
        if needle and re.search(r"\b" + re.escape(needle) + r"\b", n):
            return club.title(), WEIGHT_LEAVES_PL

    names, _codes = pl_aliases(pl_teams)
    excluded_code = team_code(exclude_team)
    excluded = _norm(exclude_team)
    for alias in sorted(names, key=len, reverse=True):
        if not alias:
            continue
        alias_code = _ALIAS_TO_CODE.get(alias)
        if excluded_code and alias_code == excluded_code:
            continue
        if not excluded_code and excluded and (alias == excluded or alias in excluded or excluded in alias):
            continue
        if re.search(r"\b" + re.escape(alias) + r"\b", n):
            return alias.title(), WEIGHT_INTRA_PL

    return None, WEIGHT_UNKNOWN


class HeadlineScore(NamedTuple):
    """What the news says about one player.

    Seven positional values is too many to track by eye, hence the names.
    ``destinations`` is the full decayed vote per club, not just the winner:
    rendering it beside the bookmaker's ladder is the point of having two
    sources, and taking only the argmax threw that away.
    """
    risk: float
    destination: object
    weight: float
    outlets: int
    evidence: list
    fee: object
    destinations: list


def score_headlines(headlines, player_name, team=None, pl_teams=None, today=None,
                    ambiguous=None, other_surnames=None):
    """Score a player's news into a risk in ``[0, 1]``.

    ``headlines`` is an iterable of mappings with ``Headline``, ``Source`` and
    ``Published`` keys.

    Confidence comes from breadth of agreement rather than any single headline:

        base    = strongest tier seen, decayed by age
        outlets = distinct sources carrying a Tier-A or Tier-B headline
        risk    = base * (0.5 + 0.5 * min(1, outlets / 4))

    So one "linked with" lands near 0.16 while Tier-A language across six outlets
    lands near 0.85.  That corroboration gate is the main defence against a single
    speculative story wrecking a good player's ranking.

    Returns a :class:`HeadlineScore`. ``risk`` is zero for a move that keeps the
    player in the Premier League — the destination and fee are still reported so
    the caller can flag it.
    """
    today = today or date.today()
    pl_teams = pl_teams or []

    best = 0.0
    best_tier = 0.0
    best_fee = None
    stay_signals = 0
    outlets = set()
    evidence = []
    dest_votes = {}

    for item in headlines or []:
        try:
            title = item.get("Headline") or item.get("title") or ""
            source = (item.get("Source") or item.get("source") or "").strip()
            published = item.get("Published") or item.get("published")
        except AttributeError:
            continue

        if not headline_mentions_player(title, player_name, team, ambiguous):
            continue

        recency = _recency_weight(published, today)
        if recency > 0 and _is_stay_signal(title):
            stay_signals += 1

        tier = classify_headline(title)
        if tier <= 0:
            continue
        decayed = tier * recency
        if decayed <= 0:
            continue

        best = max(best, decayed)
        best_tier = max(best_tier, tier)
        if tier >= TIER_B and source:
            outlets.add(_norm(source))

        club, weight = parse_destination(title, pl_teams, exclude_team=team)
        if club:
            key = (club, weight)
            dest_votes[key] = dest_votes.get(key, 0.0) + decayed

        fee = parse_fee_for_player(title, player_name, other_surnames)
        if fee is not None and (best_fee is None or fee > best_fee):
            best_fee = fee

        evidence.append({
            "Headline": title, "Source": source, "Published": published,
            "Tier": tier, "Weight": round(decayed, 3), "Fee": fee,
        })

    if best <= 0:
        return HeadlineScore(0.0, None, WEIGHT_UNKNOWN, 0, [], None, [])

    n_outlets = len(outlets)
    corroboration = (CORROBORATION_FLOOR + (1.0 - CORROBORATION_FLOOR)
                     * min(1.0, n_outlets / float(FULL_CORROBORATION_OUTLETS)))
    raw_risk = best * corroboration

    if dest_votes:
        (destination, weight), _ = max(dest_votes.items(), key=lambda kv: kv[1])
    else:
        destination, weight = None, WEIGHT_UNKNOWN

    risk = max(0.0, min(1.0, raw_risk * weight))

    if best_tier < TIER_A and stay_signals >= _MIN_STAY_SIGNALS_TO_CAP:
        risk = min(risk, _STAY_SIGNAL_RISK_CAP)

    evidence.sort(key=lambda e: e["Weight"], reverse=True)

    total_votes = sum(dest_votes.values()) or 1.0
    destinations = [
        {"Destination": club,
         "Votes": round(votes, 3),
         "Share": round(votes / total_votes, 4),
         "Intra_PL": dest_weight == WEIGHT_INTRA_PL}
        for (club, dest_weight), votes in
        sorted(dest_votes.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return HeadlineScore(risk, destination, weight, n_outlets, evidence[:10],
                         best_fee, destinations)


def resolve_from_bootstrap(status, news, pl_teams):
    """Ground truth: has the move already happened?

    The FPL bootstrap marks departed players ``status='u'`` with news naming the
    destination ("Has joined Al Qadsiah permanently").  That resolves the question
    outright and must override any amount of speculation — it is the fix for a
    rumour source going stale, which is how a betting feed came to price Watkins
    at 6% months after the story moved on.

    Returns ``(risk, destination)`` or ``None`` when the player has not left.
    """
    s = "" if status is None or (isinstance(status, float) and pd.isna(status)) else str(status).strip().lower()
    text = "" if news is None or (isinstance(news, float) and pd.isna(news)) else str(news)
    if s != "u":
        return None
    if not re.search(r"\b(joined|returned to|departed|left)\b", text, re.IGNORECASE):
        # Unavailable for some other reason; not evidence of a transfer.
        return None
    club, weight = parse_destination(text, pl_teams)
    return weight, (club or "Unknown")


def window_for_destination(destination) -> str:
    """Which window calendar governs a move to this destination."""
    n = _norm(destination)
    if n and any(re.search(p, n) for p in _SAUDI_PATTERNS):
        return "saudi"
    return "default"


#: Human labels for the window calendar's region keys.
WINDOW_LABELS = {
    "premier_league": "English",
    "default": "European",
    "saudi": "Saudi Pro League",
}


def window_status(today=None):
    """State of every transfer window, for display and for page emphasis.

    Transfers matter to a squad only while some window is open, and injuries
    matter most when none is.  Nothing in the app could previously tell the
    difference: ``transfer_exposure`` silently returns 0 once the calendar is
    spent, which renders identically to "nobody is at risk".

    Returns ``{region: {label, open, opens, closes, days_remaining, days_until}}``.
    ``days_remaining`` counts down an open window; ``days_until`` counts up to
    the next one.  Both are ``None`` when there is no window left this season.
    """
    today = today or date.today()
    status = {}
    for region, spans in (TRANSFER_WINDOWS or {}).items():
        current = next((s for s in spans if s[0] <= today <= s[1]), None)
        upcoming = sorted([s for s in spans if s[0] > today and s[0] < SEASON_END])
        if current:
            status[region] = {
                "label": WINDOW_LABELS.get(region, region.replace("_", " ").title()),
                "open": True, "opens": current[0], "closes": current[1],
                "days_remaining": (current[1] - today).days, "days_until": None,
            }
        elif upcoming:
            nxt = upcoming[0]
            status[region] = {
                "label": WINDOW_LABELS.get(region, region.replace("_", " ").title()),
                "open": False, "opens": nxt[0], "closes": nxt[1],
                "days_remaining": None, "days_until": (nxt[0] - today).days,
            }
        else:
            status[region] = {
                "label": WINDOW_LABELS.get(region, region.replace("_", " ").title()),
                "open": False, "opens": None, "closes": None,
                "days_remaining": None, "days_until": None,
            }
    return status


def any_window_open(today=None) -> bool:
    """True while a move can still complete somewhere.

    Drives which tab leads on the Availability page.  Note this stays true after
    the English deadline while the Saudi window runs — the exact gap that let
    Ollie Watkins leave five gameweeks into a season whose window had "closed".
    """
    return any(s["open"] for s in window_status(today).values())


def next_window_close(destination, today=None):
    """Close date of the next window a move to ``destination`` could use.

    Returns ``None`` once no window remains before the end of the season.
    """
    today = today or date.today()
    windows = TRANSFER_WINDOWS.get(window_for_destination(destination),
                                   TRANSFER_WINDOWS["default"])
    upcoming = [close for _, close in windows if close > today and close < SEASON_END]
    return min(upcoming) if upcoming else None


def transfer_exposure(destination=None, today=None, season_end=SEASON_END) -> float:
    """Fraction of the rest of the season a completed move would cost.

    Measured in days to the end of the season rather than gameweeks, because
    windows are dates and the mapping to gameweeks is an approximation we do not
    need.  A move can only complete while a window is open, so the worst case is
    that it lands on the closing day:

        exposure = days_left_after_window_close / days_left_now

    Pre-draft that is ~0.92 for a European move and ~0.84 for a Saudi one, so the
    discount bites hardest exactly where it matters.  After the January deadline
    no window remains, exposure is 0, and the multiplier returns to exactly 1.0
    for the rest of the season.
    """
    today = today or date.today()
    if today >= season_end:
        return 0.0
    close = next_window_close(destination, today)
    if close is None:
        return 0.0
    days_now = (season_end - today).days
    days_after = (season_end - close).days
    if days_now <= 0:
        return 0.0
    return max(0.0, min(1.0, days_after / float(days_now)))


def transfer_multiplier(risk, exposure, floor: float = TRANSFER_FLOOR) -> float:
    """Season-value multiplier in ``[floor, 1.0]``.

    A player with no transfer risk always returns exactly 1.0, so this is safe to
    apply unconditionally.  Bad input degrades to 1.0 rather than raising, matching
    ``injury_helpers.injury_multiplier``.
    """
    try:
        r = float(risk)
        e = float(exposure)
    except (ValueError, TypeError):
        return 1.0
    if pd.isna(r) or pd.isna(e) or r <= 0 or e <= 0:
        return 1.0
    return max(floor, 1.0 - max(0.0, min(1.0, r)) * max(0.0, min(1.0, e)))


# Columns ``attach_transfer_risk`` guarantees on its output.
RISK_COLUMNS = [
    "Transfer_Risk", "Transfer_Exposure", "Transfer_Mult",
    "Transfer_Destination", "Transfer_Outlets", "Transfer_Note",
    "Transfer_Status", "Transfer_Fee",
]

#: Transfer_Status values. An intra-PL move is "Moving" rather than "At risk":
#: it carries no discount, but it is still something to know before drafting.
STATUS_NONE = ""
STATUS_AT_RISK = "At risk"
STATUS_MOVING_PL = "Moving"
STATUS_DEPARTED = "Departed"


def _format_fee(fee) -> str:
    if fee is None or (isinstance(fee, float) and pd.isna(fee)):
        return ""
    try:
        value = float(fee)
    except (TypeError, ValueError):
        return ""
    return "£%.0fm" % value if value >= 10 else "£%.1fm" % value


def _format_note(risk, destination, outlets, resolved, status, fee=None) -> str:
    where = destination or "destination unclear"
    fee_text = _format_fee(fee)
    suffix = ", %s" % fee_text if fee_text else ""

    if status == STATUS_DEPARTED:
        return "Departed — %s%s" % (where, suffix)
    if status == STATUS_MOVING_PL:
        # No discount attaches to this, so the note is the whole signal.
        return "→ %s%s (stays in EPL)" % (where, suffix)
    if status == STATUS_AT_RISK:
        if not outlets:
            return "%s%s (unconfirmed)" % (where, suffix)
        return "%s%s (%d outlet%s)" % (where, suffix, outlets,
                                       "" if outlets == 1 else "s")
    return ""


def _read_odds_row(odds_row, today=None):
    """``(odds_risk, age_days, age_weight)`` from a matched odds row.

    Returns ``(0.0, nan, 0.0)`` when there is no usable quote, which makes
    ``blend_odds_risk`` a no-op — so the caller never has to branch.
    """
    if odds_row is None:
        return 0.0, float("nan"), 0.0
    try:
        exit_p = odds_row.get("Odds_Exit")
        exit_p = float(exit_p) if exit_p is not None else 0.0
    except (TypeError, ValueError):
        exit_p = 0.0
    if not (exit_p > 0) or pd.isna(exit_p):
        return 0.0, float("nan"), 0.0
    updated = odds_row.get("Odds_Updated")
    age = odds_age_days(updated, today)
    return max(0.0, min(1.0, exit_p)), age, odds_age_weight(updated, today)


def attach_transfer_risk(player_df, news_df=None, pl_teams=None, today=None,
                         name_col: str = "Player", team_col: str = "Team",
                         odds_df=None):
    """Attach transfer-risk columns to a player frame.

    ``player_df`` needs a name column and, ideally, a team column.  If it also
    carries FPL bootstrap ``status`` and ``news`` columns, a completed departure
    found there overrides the news scoring entirely — speculation must never
    outrank a deal that has already happened.

    ``news_df`` is the output of ``transfer_feeds.fetch_transfer_news_batch``.

    Name matching is not needed here: the news is fetched *per player*, so each
    headline set already belongs to a known player.  That is why this module can
    stay free of ``player_matching`` (which imports Streamlit).

    ``odds_df`` is optional bookmaker next-club odds, already matched to this
    frame's ``name_col`` by ``transfer_risk_app.attach_odds`` (name matching needs
    ``ReferenceMatcher``, which reaches Streamlit, so it cannot happen here).  It
    needs ``Player`` plus any of ``Odds_Exit``, ``Odds_Destination``,
    ``Odds_Fractional``, ``Odds_Bookmaker``, ``Odds_Updated``.

    Odds are consulted **only** where the bootstrap has not already resolved the
    move.  A completed deal is ground truth and a price is speculation; letting a
    stale quote reopen a settled question is precisely how a betting feed came to
    price Ollie Watkins at 6% months after he had gone.

    A player with no news gets risk 0.0 and multiplier exactly 1.0, so callers can
    apply the multiplier unconditionally.
    """
    result = player_df.copy()
    if result.empty:
        for col in RISK_COLUMNS:
            result[col] = pd.Series(dtype="object" if "Note" in col or "Destination" in col else "float")
        for col in ODDS_COLUMNS:
            result[col] = pd.Series(dtype="object" if col in
                                    ("Odds_Destination", "Odds_Fractional",
                                     "Odds_Bookmaker", "Odds_Updated") else "float")
        return result

    today = today or date.today()
    pl_teams = list(pl_teams or [])

    by_player = {}
    if news_df is not None and not getattr(news_df, "empty", True):
        for player, group in news_df.groupby(name_col):
            by_player[_norm(player)] = group.to_dict("records")

    ambiguous = build_ambiguous_tokens(result[name_col].tolist())
    all_surnames = set()
    for other in result[name_col].tolist():
        parts = [t for t in _norm(other).split() if len(t) > 1]
        if parts:
            all_surnames.add(parts[-1])

    has_status = "status" in result.columns
    has_news = "news" in result.columns

    by_odds = {}
    if odds_df is not None and not getattr(odds_df, "empty", True):
        for _, orow in odds_df.iterrows():
            key = _norm(orow.get("Player"))
            if key:
                by_odds[key] = orow

    risks, exposures, mults, dests = [], [], [], []
    outlets_col, notes, statuses, fees = [], [], [], []
    odds_risks, odds_dests, odds_fracs = [], [], []
    odds_books, odds_updates, odds_ages, odds_weights = [], [], [], []

    for _, row in result.iterrows():
        name = row.get(name_col)
        team = row.get(team_col) if team_col in result.columns else None

        resolved = None
        if has_status:
            resolved = resolve_from_bootstrap(
                row.get("status"), row.get("news") if has_news else None, pl_teams
            )

        # Reported for every row so the page can show what the market said even
        # where it was not allowed to move the score.
        odds_row = by_odds.get(_norm(name))
        odds_risk, odds_age, odds_weight = _read_odds_row(odds_row, today)

        fee = None
        if resolved is not None:
            risk, destination = resolved
            n_outlets = 0
            news_text = row.get("news") if has_news else ""
            club, _ = parse_destination(news_text, pl_teams)
            destination = club or destination
            fee = parse_fee(news_text)
            # A completed intra-PL move is still a move, not a loss.
            status = STATUS_DEPARTED if risk > 0 else STATUS_MOVING_PL
        else:
            headlines = by_player.get(_norm(name), [])
            scored = score_headlines(
                headlines, name, team, pl_teams, today=today, ambiguous=ambiguous,
                other_surnames=all_surnames,
            )
            risk, destination, weight = scored.risk, scored.destination, scored.weight
            n_outlets, fee = scored.outlets, scored.fee

            if odds_risk > 0:
                risk = blend_odds_risk(risk, odds_risk, odds_weight)
                if not destination and odds_row is not None and odds_row.get("Odds_Destination"):
                    destination = str(odds_row.get("Odds_Destination"))
                    weight = WEIGHT_UNKNOWN

            if risk > 0:
                status = STATUS_AT_RISK
            elif destination and weight == WEIGHT_INTRA_PL:
                status = STATUS_MOVING_PL
            else:
                status = STATUS_NONE

        if resolved is not None:
            # He has already gone.  There is no window to wait on — the loss
            # applies to every remaining gameweek.
            exposure = 1.0
        elif risk > 0:
            exposure = transfer_exposure(destination, today=today)
        else:
            exposure = 0.0
        mult = transfer_multiplier(risk, exposure)

        risks.append(round(float(risk), 4))
        exposures.append(round(float(exposure), 4))
        mults.append(round(float(mult), 4))
        dests.append(destination or "")
        outlets_col.append(int(n_outlets))
        statuses.append(status)
        fees.append(float(fee) if fee is not None else float("nan"))
        notes.append(_format_note(risk, destination, n_outlets,
                                  resolved is not None, status, fee))

        odds_risks.append(round(float(odds_risk), 4))
        odds_ages.append(round(float(odds_age), 1) if odds_age is not None else float("nan"))
        odds_weights.append(round(float(odds_weight), 4))
        odds_dests.append(str(odds_row.get("Odds_Destination") or "") if odds_row is not None else "")
        odds_fracs.append(str(odds_row.get("Odds_Fractional") or "") if odds_row is not None else "")
        odds_books.append(str(odds_row.get("Odds_Bookmaker") or "") if odds_row is not None else "")
        odds_updates.append(str(odds_row.get("Odds_Updated") or "") if odds_row is not None else "")

    result["Transfer_Risk"] = risks
    result["Transfer_Exposure"] = exposures
    result["Transfer_Mult"] = mults
    result["Transfer_Destination"] = dests
    result["Transfer_Outlets"] = outlets_col
    result["Transfer_Note"] = notes
    result["Transfer_Status"] = statuses
    result["Transfer_Fee"] = fees

    result["Odds_Risk"] = odds_risks
    result["Odds_Destination"] = odds_dests
    result["Odds_Fractional"] = odds_fracs
    result["Odds_Bookmaker"] = odds_books
    result["Odds_Updated"] = odds_updates
    result["Odds_Age_Days"] = odds_ages
    result["Odds_Weight"] = odds_weights
    return result


# --- Inbound transfers: who is arriving, and whose minutes it costs ----------
#
# The outbound model asks "will he still be here?". This asks the two questions
# an arrival raises: is the incoming player worth targeting once he is added to
# the game, and whose minutes does he eat when he lands.
#
# This path is inherently noisier than the outbound one. A per-player query is
# already about that player; a per-club query is about the club, and the arriving
# player's name has to be pulled out of prose. Everything below is therefore
# deliberately conservative: corroboration is required, the discount is small and
# capped, and an unparsed name is dropped rather than guessed at.

#: FPL position implied by how reporters describe a player.  "Winger" maps to
#: midfield because that is how FPL classifies the overwhelming majority of them.
_POSITION_KEYWORDS = (
    ("G", ("goalkeeper", "keeper", "shot stopper", "shotstopper")),
    ("D", ("defender", "centre back", "center back", "centreback", "full back",
           "fullback", "left back", "right back", "wing back", "wingback")),
    ("F", ("striker", "centre forward", "center forward", "forward", "attacker",
           "number 9", "no 9", "marksman", "frontman")),
    ("M", ("midfielder", "midfield", "playmaker", "winger", "wide man",
           "attacking midfielder", "defensive midfielder")),
)

# The buying club is whichever club precedes the signing verb — never the club
# whose feed the headline arrived on. "Nottingham Forest attempt double Crystal
# Palace raid" surfaces under a Crystal Palace query and describes Forest buying;
# trusting the query would discount the selling club's squad, which is backwards.
_SIGNING_VERBS = (
    r"sign(?:s|ed|ing)?(?:\s+of)?", r"complete[sd]?\s+(?:the\s+)?(?:signing\s+of|move\s+for|deal\s+for)",
    r"agree(?:d|s)?\s+(?:a\s+)?deal\s+(?:to\s+sign|for)", r"agree(?:d|s)?\s+(?:to\s+sign|terms\s+with)",
    r"swoop\s+for", r"capture\s+of", r"move\s+for", r"land(?:s|ed)?", r"snap(?:s|ped)?\s+up",
    r"in\s+talks\s+to\s+sign", r"finali[sz]ing\s+deal\s+for", r"confident\s+of\s+signing",
)
_VERB_RE = re.compile(r"\b(?:%s)\b" % "|".join(_SIGNING_VERBS), re.IGNORECASE)

_NAME_RUN = r"(?:[A-Z][\w'\u00C0-\u024F\-]+)(?:\s+[A-Z][\w'\u00C0-\u024F\-]+){0,2}"
#: Google headlines very often lead with the subject: "Anan Khalaili transfer
#: news: Crystal Palace sign winger from Union Saint-Gilloise". That prefix is
#: the most reliable name in the sentence.
_PREFIX_NAME_RE = re.compile(r"^(%s)\s*(?::|\s+transfer\s+news\b)" % _NAME_RUN)
_AFTER_VERB_RE = re.compile(r"(%s)" % _NAME_RUN)
#: "Crystal Palace transfer news: Axel Disasi signs on loan from Chelsea" — the
#: player sits between the club's own headline prefix and the verb.
_BEFORE_VERB_RE = re.compile(r"(%s)\s*$" % _NAME_RUN)
_JOINS_RE = re.compile(
    r"(%s)\s+(?:joins?|set\s+to\s+join|agrees?\s+(?:to\s+join|move\s+to)|"
    r"completes?\s+move\s+to|signs?\s+for)\s+(%s)" % (_NAME_RUN, _NAME_RUN)
)

#: Words that look like a name but are not one. Without this the extractor
#: happily reports signing "Transfer News", "Premier League" and "England".
_NAME_STOPWORDS = frozenset("""
transfer transfers news latest live update updates deal deals move moves target
targets star striker forward winger midfielder defender goalkeeper keeper player
premier league championship efl summer window january deadline day report reports
paper talk gossip rumour rumours rumors boss manager captain ace kid teen wonderkid
new signing signings source exclusive done here we go official confirmed
england scotland wales ireland france spain germany italy portugal brazil
argentina netherlands belgium usmnt prospect loan contract medical
""".split())

#: How much of a player's value a same-position arrival can take, at most. A new
#: signing rarely removes a starter outright, and the depth chart is unknown, so
#: this stays small: it is a tiebreak between similar players, not a verdict.
MAX_MINUTES_IMPACT = 0.25
MINUTES_FLOOR = 1.0 - MAX_MINUTES_IMPACT
#: The established first choice at a position is markedly less threatened than
#: the players behind him.
INCUMBENT_TOP_SHARE = 0.4


def position_from_text(text):
    """FPL position code implied by a headline's description, or ``None``.

    Checked longest-keyword-first so "attacking midfielder" is not read as a
    forward on the word "attack".
    """
    n = _norm(text)
    if not n:
        return None
    best = None
    best_len = 0
    for code, keywords in _POSITION_KEYWORDS:
        for kw in keywords:
            if kw in n and len(kw) > best_len:
                best, best_len = code, len(kw)
    return best


def _looks_like_name(candidate, pl_teams) -> bool:
    """Reject club names, stopwords and other non-names from the extractor."""
    words = [w for w in _norm(candidate).split() if w]
    if not words or len(words) > 3:
        return False
    if any(w in _NAME_STOPWORDS for w in words):
        return False
    if is_premier_league_club(candidate, pl_teams):
        return False
    if any(_norm(c) == _norm(candidate) for c in _FOREIGN_CLUBS):
        return False
    if any(re.search(p, _norm(candidate)) for p in _SAUDI_PATTERNS):
        return False
    return all(len(w) > 1 for w in words)


def _club_before(text, index, pl_teams):
    """Nearest Premier League club named before ``index`` — the buying club."""
    head = _norm(text[:index])
    if not head:
        return None
    names, _codes = pl_aliases(pl_teams)
    best, best_at = None, -1
    for alias in names:
        if not alias:
            continue
        at = head.rfind(alias)
        if at > best_at:
            best, best_at = alias, at
    return best.title() if best else None


def extract_signing(headline, pl_teams, queried_club=None):
    """Who is arriving where, from a signing headline.

    Returns ``(player, buying_club, position, fee)`` or ``None``.

    The buying club is derived from the sentence, never from whose feed the
    headline arrived on — see the note on ``_SIGNING_VERBS``.
    """
    raw = "" if headline is None else str(headline)
    if not raw or _is_stay_signal(raw):
        return None

    position = position_from_text(raw)
    fee = parse_fee(raw)

    # "<Player> joins/signs for <Club>" states both ends explicitly.
    joins = _JOINS_RE.search(raw)
    if joins:
        player, club = joins.group(1).strip(), joins.group(2).strip()
        if _looks_like_name(player, pl_teams) and is_premier_league_club(club, pl_teams):
            return player, club, position, fee

    verb = _VERB_RE.search(raw)
    if not verb:
        return None

    buyer = _club_before(raw, verb.start(), pl_teams)
    if not buyer:
        return None

    # Prefer the headline's leading subject; fall back to the name after the verb.
    player = None
    prefix = _PREFIX_NAME_RE.match(raw)
    if prefix and _looks_like_name(prefix.group(1), pl_teams):
        player = prefix.group(1).strip()

    if not player:
        before = _BEFORE_VERB_RE.search(raw[:verb.start()])
        if before and _looks_like_name(before.group(1), pl_teams):
            player = before.group(1).strip()

    if not player:
        tail = raw[verb.end():]
        for match in _AFTER_VERB_RE.finditer(tail):
            candidate = match.group(1).strip()
            if _looks_like_name(candidate, pl_teams):
                player = candidate
                break

    if not player:
        return None
    return player, buyer, position, fee


def build_inbound_watchlist(club_news, pl_teams, today=None, min_outlets: int = 2,
                            known_players=None):
    """Players reportedly arriving at Premier League clubs.

    Two uses: they are waiver targets the moment they are added to the game, and
    they are the reason an incumbent at the same club and position is about to
    lose minutes.

    Corroboration is required (``min_outlets``) for the same reason it is on the
    outbound side — one paper's speculation is not a signing.

    Returns a frame of ``Player, Club, Position, Fee, Outlets, Confidence,
    Headline``.
    """
    today = today or date.today()
    columns = ["Player", "Club", "Position", "Fee", "Outlets", "Confidence", "Headline"]
    if club_news is None or getattr(club_news, "empty", True):
        return pd.DataFrame(columns=columns)

    known_positions = {}
    if known_players is not None and not getattr(known_players, "empty", True):
        if "Player" in known_players.columns and "Position" in known_players.columns:
            for rec in known_players.to_dict("records"):
                known_positions[_norm(rec.get("Player"))] = rec.get("Position")

    found = {}
    for record in club_news.to_dict("records"):
        club = record.get("Club") or ""
        headline = record.get("Headline") or ""
        source = (record.get("Source") or "").strip()
        recency = _recency_weight(record.get("Published"), today)
        if recency <= 0:
            continue

        parsed = extract_signing(headline, pl_teams, queried_club=club)
        if not parsed:
            continue
        name, club, position, fee = parsed
        if position is None:
            # An intra-PL mover is already in the game, so his position is known
            # exactly — far better than inferring it from prose.
            position = known_positions.get(_norm(name))

        tier = classify_signing(headline)
        if tier <= 0:
            continue

        key = (_norm(name), _norm(club))
        entry = found.setdefault(key, {
            "Player": name, "Club": club, "Position": position, "Fee": fee,
            "_outlets": set(), "_best": 0.0, "Headline": headline,
        })
        if source:
            entry["_outlets"].add(_norm(source))
        score = tier * recency
        if score > entry["_best"]:
            entry["_best"] = score
            entry["Headline"] = headline
        if position and not entry["Position"]:
            entry["Position"] = position
        if fee is not None and (entry["Fee"] is None or fee > entry["Fee"]):
            entry["Fee"] = fee

    rows = []
    for entry in found.values():
        n_outlets = len(entry["_outlets"])
        if n_outlets < min_outlets:
            continue
        corroboration = (CORROBORATION_FLOOR + (1.0 - CORROBORATION_FLOOR)
                         * min(1.0, n_outlets / float(FULL_CORROBORATION_OUTLETS)))
        rows.append({
            "Player": entry["Player"], "Club": entry["Club"],
            "Position": entry["Position"], "Fee": entry["Fee"],
            "Outlets": n_outlets,
            "Confidence": round(min(1.0, entry["_best"] * corroboration), 4),
            "Headline": entry["Headline"],
        })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "Confidence", ascending=False).reset_index(drop=True)


def _same_player(a, b) -> bool:
    """Whether two published names refer to the same person.

    Deliberately narrow. Over-matching here only costs a missed discount, but the
    names come from different sources and differ in accents ("Emiliano Martínez"
    vs "Emiliano Martinez") and in completeness — a headline routinely prints the
    surname alone ("Jackson signs for Aston Villa").
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    ta, tb = na.split(), nb.split()
    if len(ta) == 1 or len(tb) == 1:
        return ta[-1] == tb[-1]
    return False


def _arrival_threat(row) -> float:
    """How much of a squad place an arrival is likely to take, in ``[0, 1]``.

    Fee is the sharpest available evidence: a club that pays a large fee intends
    to play the player. Absent a fee, confidence in the reporting carries it.
    """
    confidence = float(row.get("Confidence") or 0.0)
    fee = row.get("Fee")
    if fee is None or (isinstance(fee, float) and pd.isna(fee)):
        fee_factor = 0.6
    else:
        fee_factor = 0.5 + 0.5 * min(1.0, float(fee) / BIG_FEE_GBP_M)
    return max(0.0, min(1.0, confidence * fee_factor))


def apply_minutes_competition(player_df, arrivals, team_col: str = "Team",
                              position_col: str = "Position",
                              value_col: str = "Points",
                              status_col: str = "Transfer_Status"):
    """Discount incumbents whose minutes an incoming signing threatens.

    Adds ``Minutes_Mult``, ``Minutes_Note`` and ``Competition`` — kept separate
    from ``Transfer_Mult`` because they answer different questions: one is "will
    he be here", this is "will he still play".

    The established first choice at a club and position is much safer than the
    players behind him, so he absorbs only ``INCUMBENT_TOP_SHARE`` of the threat.
    Ranking uses ``value_col`` (season projection), which is the best available
    proxy for who currently starts.

    A player who is himself leaving is exempt. The two effects are usually the
    same event seen from both ends — Nicolas Jackson arrives at Villa *because*
    Ollie Watkins is going to Al-Hilal — and charging Watkins for his own
    replacement double-counts a single move.
    """
    result = player_df.copy()
    result["Minutes_Mult"] = 1.0
    result["Minutes_Note"] = ""
    result["Competition"] = ""

    if arrivals is None or getattr(arrivals, "empty", True) or result.empty:
        return result
    if team_col not in result.columns or position_col not in result.columns:
        return result

    name_col = "Player" if "Player" in result.columns else None

    for arrival in arrivals.to_dict("records"):
        position = arrival.get("Position")
        club = arrival.get("Club")
        if not position or not club:
            continue  # cannot attribute competition without both

        club_code = team_code(club)
        mask = result[position_col].astype(str).eq(str(position))
        if club_code:
            mask &= result[team_col].map(lambda t: team_code(t) == club_code)
        else:
            mask &= result[team_col].astype(str).str.lower().eq(str(club).lower())

        # A signing already in the pool would otherwise compete with himself:
        # "James Trafford completes £40m move to Leeds" discounted Leeds' new
        # goalkeeper for arriving. He is still competition for everyone else at
        # the club, just not for his own place.
        if name_col:
            mask &= ~result[name_col].map(
                lambda n, a=arrival.get("Player"): _same_player(n, a))

        if status_col in result.columns:
            leaving = result[status_col].astype(str).isin(
                (STATUS_AT_RISK, STATUS_DEPARTED, STATUS_MOVING_PL))
            mask &= ~leaving

        incumbents = result[mask]
        if incumbents.empty:
            continue

        threat = _arrival_threat(arrival)
        if threat <= 0:
            continue

        if value_col in incumbents.columns:
            order = pd.to_numeric(incumbents[value_col], errors="coerce").fillna(0)
            first_choice = order.idxmax()
        else:
            first_choice = None

        fee_text = _format_fee(arrival.get("Fee"))
        label = "%s%s" % (arrival.get("Player") or "new signing",
                          " (%s)" % fee_text if fee_text else "")

        for idx in incumbents.index:
            share = INCUMBENT_TOP_SHARE if idx == first_choice else 1.0
            impact = threat * share * MAX_MINUTES_IMPACT
            new_mult = max(MINUTES_FLOOR, 1.0 - impact)
            if new_mult < result.at[idx, "Minutes_Mult"]:
                result.at[idx, "Minutes_Mult"] = round(new_mult, 4)
                result.at[idx, "Competition"] = label
                result.at[idx, "Minutes_Note"] = "%s arriving at %s" % (label, club)

    return result
