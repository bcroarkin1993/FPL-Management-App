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

import pandas as pd

from scripts.common.text_helpers import TEAM_FULL_TO_SHORT, canonical_normalize

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
# Still in the game.  In Draft you simply keep him.  The residual cost is
# settling in, a new fixture run and rotation risk.
WEIGHT_INTRA_PL = 0.20
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
_TIER_B_PATTERNS = [
    r"\bbid\b", r"\boffer\b", r"\bin talks\b", r"\bhold(?:ing)?\s+talks\b",
    r"\bclose to\b", r"\bnegotiat", r"\btransfer request\b", r"\bapproach(?:ed)?\b",
    r"\bpush(?:ing)?\s+to\s+sign\b", r"\bwants?\s+to\s+(?:leave|join)\b",
    r"\bopen\s+to\s+(?:a\s+)?(?:move|exit)\b", r"\bswoop\b",
    # Reporting a player's "exit" presupposes the exit.
    r"\bexit\b", r"\bdeparture\b", r"\bfarewell\b",
]
_TIER_C_PATTERNS = [
    r"\blink(?:ed|s)?\b", r"\binterest(?:ed)?\b", r"\bmonitor(?:ing)?\b",
    r"\beye(?:ing|d)?\b", r"\btarget(?:ing)?\b", r"\brumou?r\b", r"\bgossip\b",
    r"\bconsider(?:ing)?\b", r"\bcould\s+(?:leave|join|move)\b",
    r"\bwould\s+(?:leave|join)\b", r"\bspeculation\b",
]
# A denial caps the score at Tier C no matter what else the headline says.
# "Villa rule out Watkins sale" must not score as Tier A on the word "sale".
_NEGATION_PATTERNS = [
    r"\bnot for sale\b", r"\brule(?:d|s)?\s+out\b", r"\bden(?:y|ies|ied)\b",
    r"\breject(?:ed|s)?\b", r"\bturn(?:ed|s)?\s+down\b", r"\bsigns?\s+new\s+(?:deal|contract)\b",
    r"\bnew contract\b", r"\bstay(?:s|ing)?\s+(?:at|put)\b", r"\bwill\s+not\s+leave\b",
    r"\bno\s+(?:plans|intention)\s+to\s+(?:sell|leave)\b", r"\bcollapse[ds]?\b",
    r"\boff\b(?=.*\btransfer\b)", r"\bcall(?:ed|s)?\s+off\b", r"\bfails?\b",
]

_TIERS = ((TIER_A, _TIER_A_PATTERNS), (TIER_B, _TIER_B_PATTERNS), (TIER_C, _TIER_C_PATTERNS))

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
_JOINED_RE = re.compile(
    r"(?:has\s+)?(?:joined|returned to|departed(?:\s+the\s+club)?|left)\s+(?:to\s+)?(.+?)"
    r"(?:\s+(?:permanently|on\s+loan|for\s+the\s+rest|as\s+a\s+free\s+agent)|[.,]|$)",
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


def classify_headline(headline) -> float:
    """Strongest tier score the headline's language supports.

    A denial caps the result at Tier C: the story is real, the move is not.
    """
    h = " " + _norm(headline) + " "
    if not h.strip():
        return 0.0
    negated = any(re.search(p, h) for p in _NEGATION_PATTERNS)
    score = 0.0
    for tier_score, patterns in _TIERS:
        if any(re.search(p, h) for p in patterns):
            score = tier_score
            break
    if negated:
        score = min(score, TIER_C)
    return score


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


def score_headlines(headlines, player_name, team=None, pl_teams=None, today=None,
                    ambiguous=None):
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

    Returns ``(risk, destination, weight, n_outlets, evidence)``.
    """
    today = today or date.today()
    pl_teams = pl_teams or []

    best = 0.0
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

        tier = classify_headline(title)
        if tier <= 0:
            continue
        decayed = tier * _recency_weight(published, today)
        if decayed <= 0:
            continue

        best = max(best, decayed)
        if tier >= TIER_B and source:
            outlets.add(_norm(source))

        club, weight = parse_destination(title, pl_teams, exclude_team=team)
        if club:
            key = (club, weight)
            dest_votes[key] = dest_votes.get(key, 0.0) + decayed

        evidence.append({
            "Headline": title, "Source": source, "Published": published,
            "Tier": tier, "Weight": round(decayed, 3),
        })

    if best <= 0:
        return 0.0, None, WEIGHT_UNKNOWN, 0, []

    n_outlets = len(outlets)
    corroboration = (CORROBORATION_FLOOR + (1.0 - CORROBORATION_FLOOR)
                     * min(1.0, n_outlets / float(FULL_CORROBORATION_OUTLETS)))
    raw_risk = best * corroboration

    if dest_votes:
        (destination, weight), _ = max(dest_votes.items(), key=lambda kv: kv[1])
    else:
        destination, weight = None, WEIGHT_UNKNOWN

    risk = max(0.0, min(1.0, raw_risk * weight))
    evidence.sort(key=lambda e: e["Weight"], reverse=True)
    return risk, destination, weight, n_outlets, evidence[:10]


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
]


def _format_note(risk, destination, outlets, resolved) -> str:
    if risk <= 0:
        return ""
    where = destination or "destination unclear"
    if resolved:
        return "Departed — %s" % where
    if not outlets:
        # Only Tier-C language matched, so no outlet was counted. Say so rather
        # than printing "(0 outlets)", which reads as a bug.
        return "%s (unconfirmed)" % where
    return "%s (%d outlet%s)" % (where, outlets, "" if outlets == 1 else "s")


def attach_transfer_risk(player_df, news_df=None, pl_teams=None, today=None,
                         name_col: str = "Player", team_col: str = "Team"):
    """Attach transfer-risk columns to a player frame.

    ``player_df`` needs a name column and, ideally, a team column.  If it also
    carries FPL bootstrap ``status`` and ``news`` columns, a completed departure
    found there overrides the news scoring entirely — speculation must never
    outrank a deal that has already happened.

    ``news_df`` is the output of ``transfer_feeds.fetch_transfer_news_batch``.

    Name matching is not needed here: the news is fetched *per player*, so each
    headline set already belongs to a known player.  That is why this module can
    stay free of ``player_matching`` (which imports Streamlit).

    A player with no news gets risk 0.0 and multiplier exactly 1.0, so callers can
    apply the multiplier unconditionally.
    """
    result = player_df.copy()
    if result.empty:
        for col in RISK_COLUMNS:
            result[col] = pd.Series(dtype="object" if "Note" in col or "Destination" in col else "float")
        return result

    today = today or date.today()
    pl_teams = list(pl_teams or [])

    by_player = {}
    if news_df is not None and not getattr(news_df, "empty", True):
        for player, group in news_df.groupby(name_col):
            by_player[_norm(player)] = group.to_dict("records")

    ambiguous = build_ambiguous_tokens(result[name_col].tolist())

    has_status = "status" in result.columns
    has_news = "news" in result.columns

    risks, exposures, mults, dests, outlets_col, notes = [], [], [], [], [], []

    for _, row in result.iterrows():
        name = row.get(name_col)
        team = row.get(team_col) if team_col in result.columns else None

        resolved = None
        if has_status:
            resolved = resolve_from_bootstrap(
                row.get("status"), row.get("news") if has_news else None, pl_teams
            )

        if resolved is not None:
            risk, destination = resolved
            n_outlets = 0
            club, _ = parse_destination(
                row.get("news") if has_news else "", pl_teams
            )
            destination = club or destination
        else:
            headlines = by_player.get(_norm(name), [])
            risk, destination, _w, n_outlets, _ev = score_headlines(
                headlines, name, team, pl_teams, today=today, ambiguous=ambiguous
            )

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
        notes.append(_format_note(risk, destination, n_outlets, resolved is not None))

    result["Transfer_Risk"] = risks
    result["Transfer_Exposure"] = exposures
    result["Transfer_Mult"] = mults
    result["Transfer_Destination"] = dests
    result["Transfer_Outlets"] = outlets_col
    result["Transfer_Note"] = notes
    return result
