"""Bookmaker next-club odds as a transfer-risk signal.

Pure and network-free, with the same import constraints as ``transfer_risk``:
plain ``logging`` only, no ``error_helpers``/``cache``/``player_matching`` (all
three reach Streamlit), so GitHub Actions can import this.  Name matching against
the FPL pool therefore lives in ``transfer_risk_app``.

Why odds at all
---------------
News tells you a story is being written; a price tells you what someone is
willing to be wrong about.  The two disagree often enough to be worth seeing
side by side, and a market can exist for a player no headline has named yet.

The catch is that this feed goes stale silently -- the live Salah quote carries
``semanticOddsUpdatedAt = 2026-03-25``, five months old -- which is exactly how a
betting source came to price Ollie Watkins at 6% months after the story moved on.
So every quote is discounted by age (``odds_age_weight``) and a completed deal in
the FPL bootstrap still overrides everything.  Stale odds are a weak signal, not
a wrong one; they must never look like a fresh one.

A ladder is not a probability distribution
------------------------------------------
The single most important thing in this module.  A live ladder looks like::

    Any Saudi club   8/11   57.9%     <- contains Al Ittihad and Al Hilal
    Al Ittihad        7/4   36.4%
    Any MLS Team      5/2   28.6%
    Al Hilal          7/1   12.5%
    Any French club   8/1   11.1%
    Any Italian club  8/1   11.1%
                            ------
                            157.6%

That 157.6% is mostly *overlap*, not bookmaker margin.  Normalising it whole
would report Saudi at 37% when the market prices it at 58% -- understating every
row by counting the same outcome three times.  ``disjoint_ladder`` keeps each
outcome once (preferring the aggregate, which is the broader and more liquid
market) before anything is normalised.

What the numbers mean
---------------------
The feed quotes only *destinations*; no bookmaker prices "stays at Liverpool".
Without a stay price, normalising cannot yield P(leaves) -- it would force it to
1.0 by construction.  So the two outputs are deliberately different questions:

``normalise_ladder``  -- *given that he moves*, where to.  A conditional
                         distribution over the disjoint set; sums to 1.0.
``exit_probability``  -- the market's shortest quoted price on a departure.
                         Margin is not removed, so it slightly overstates that
                         one destination while remaining a fair floor on leaving
                         at all.

Both are model estimates from a comparison site, never a settled market price.
"""

import logging
import math
import re
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)


# --- Tunables ---------------------------------------------------------------

#: Half-life of a quote's influence, in days.  Much longer than the 10-day news
#: half-life: prices move slower than headlines, and a month-old price on a slow
#: saga is still informative where a month-old rumour usually is not.
ODDS_HALF_LIFE_DAYS = 45.0

#: Age assumed when a quote carries no timestamp.  Not zero -- a parse failure
#: must not read as freshness, the same reasoning as ``WEIGHT_UNKNOWN`` in
#: ``transfer_risk``.  Costs a quote ~63% of its weight.
ODDS_ASSUMED_AGE_DAYS = 30.0

#: Relative weights when blending the news model with the market.
W_NEWS = 1.0
W_ODDS = 0.8

#: Display bands, in days.
AGE_BAND_LIVE = 7
AGE_BAND_AGING = 45
AGE_BAND_STALE = 120

ODDS_LADDER_COLUMNS = ["Player", "Destination", "Fractional", "Decimal",
                       "Implied", "Kind", "Updated"]
ODDS_INDEX_COLUMNS = ["Player", "Slug", "Next_Club", "Fractional", "Decimal",
                      "Implied", "Bookmaker", "Trending", "Updated"]

#: Columns ``attach_transfer_risk`` adds when odds are supplied.
ODDS_COLUMNS = ["Odds_Risk", "Odds_Destination", "Odds_Fractional",
                "Odds_Bookmaker", "Odds_Updated", "Odds_Age_Days", "Odds_Weight"]


# --- Fractional odds --------------------------------------------------------

_EVENS = {"evens", "evs", "even", "1/1"}
_FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?))?\s*$")


def parse_fractional(text):
    """Fractional odds -> decimal odds.  ``None`` when unparseable.

    ``"8/11"`` -> 1.727, ``"1/2"`` -> 1.5, ``"evens"`` -> 2.0.  A bare numerator
    is the bookmaker shorthand for "to one": the live feed publishes ``"2"``
    alongside ``decimal: 3``, so ``"2"`` means 2/1, not decimal 2.
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    if s in _EVENS:
        return 2.0
    m = _FRACTION_RE.match(s)
    if not m:
        return None
    try:
        num = float(m.group(1))
        den = float(m.group(2)) if m.group(2) is not None else 1.0
    except (TypeError, ValueError):
        return None
    if den <= 0 or num < 0:
        return None
    return num / den + 1.0


def implied_probability(decimal_odds):
    """Decimal odds -> implied probability in ``(0, 1]``.  ``None`` if invalid.

    This is the raw book price and still carries the bookmaker's margin; it is
    not a true probability and a set of them does not sum to 1.
    """
    try:
        d = float(decimal_odds)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(d) or d < 1.0:
        return None
    return 1.0 / d


# --- Aggregate vs specific club ---------------------------------------------

#: An aggregate market covers a whole region or league ("Any Saudi club").
_AGGREGATE_RE = re.compile(
    r"\b(any\b|a\s+saudi|elsewhere|another\s+club|abroad|unattached|retire)", re.I)

#: Regions we can resolve, so a specific club can be recognised as sitting
#: inside a quoted aggregate.  Deliberately small: an unresolved club stays in
#: the disjoint set rather than being dropped, so the failure mode is a visible
#: overround (which ``check_transfer_odds`` errors on) instead of silent loss.
_REGION_CLUBS = {
    "saudi": [r"al[\s-]?(ittihad|hilal|nassr|ahli|qadsiah|shabab|ettifaq|fateh|riyadh|taawoun)"],
    "mls": [r"\b(inter\s+miami|la\s+galaxy|lafc|new\s+york\s+(city|red\s+bulls)|toronto\s+fc|seattle\s+sounders|atlanta\s+united)\b"],
    "french": [r"\b(psg|paris\s+saint[\s-]?germain|marseille|lyon|monaco|lille|nice|rennes|lens)\b"],
    "italian": [r"\b(juventus|inter\s+milan|ac\s+milan|napoli|roma|lazio|atalanta|fiorentina)\b"],
    "spanish": [r"\b(barcelona|real\s+madrid|atletico\s+madrid|sevilla|villarreal|real\s+sociedad|valencia|betis)\b"],
    "german": [r"\b(bayern\s+munich|borussia\s+dortmund|rb\s+leipzig|bayer\s+leverkusen|stuttgart|frankfurt)\b"],
    "turkish": [r"\b(galatasaray|fenerbahce|besiktas|trabzonspor)\b"],
}

#: Which region an aggregate label refers to.
_AGGREGATE_REGIONS = {
    "saudi": [r"saudi"],
    "mls": [r"\bmls\b", r"major\s+league\s+soccer", r"\busa\b", r"american"],
    "french": [r"french", r"ligue\s*1"],
    "italian": [r"italian", r"serie\s*a"],
    "spanish": [r"spanish", r"la\s*liga"],
    "german": [r"german", r"bundesliga"],
    "turkish": [r"turkish", r"super\s*lig"],
    "english": [r"english", r"premier\s+league", r"championship", r"\bepl\b"],
}


def classify_market(label):
    """``"aggregate"`` for a region/league market, ``"club"`` for a named club."""
    if not label:
        return "club"
    return "aggregate" if _AGGREGATE_RE.search(str(label)) else "club"


def aggregate_region(label):
    """Region an aggregate label covers, or ``None``."""
    if not label or classify_market(label) != "aggregate":
        return None
    text = str(label).lower()
    for region, patterns in _AGGREGATE_REGIONS.items():
        if any(re.search(p, text) for p in patterns):
            return region
    return None


def club_region(label):
    """Region a specific club sits in, or ``None`` when unresolved.

    ``None`` deliberately means "keep it": we would rather double-count and trip
    the overround check than silently drop a real destination.
    """
    if not label:
        return None
    text = str(label).lower()
    for region, patterns in _REGION_CLUBS.items():
        if any(re.search(p, text) for p in patterns):
            return region
    return None


# --- The ladder -------------------------------------------------------------

def _row_label(row):
    for key in ("Destination", "club", "Club", "destination", "label"):
        if isinstance(row, dict) and row.get(key):
            return str(row[key])
    return ""


def _row_decimal(row):
    if not isinstance(row, dict):
        return None
    for key in ("Decimal", "decimal"):
        if row.get(key) is not None:
            try:
                d = float(row[key])
                if math.isfinite(d) and d >= 1.0:
                    return d
            except (TypeError, ValueError):
                pass
    for key in ("Fractional", "odds", "Odds", "fractional", "bestOdds"):
        if row.get(key):
            d = parse_fractional(row[key])
            if d is not None:
                return d
    return None


def disjoint_ladder(rows):
    """Drop rows whose outcome is already covered by another row.

    Where an aggregate is quoted, it wins and its member clubs are dropped: the
    aggregate is the broader market and the specific clubs are a strict subset of
    it.  For Salah that turns six overlapping rows summing to 157.6% into four
    disjoint ones summing to ~108.7% -- a believable book margin.

    A club whose region we cannot resolve is kept, so unknown vocabulary costs a
    visible overround rather than a lost destination.
    """
    kept = []
    covered = set()
    for row in rows or []:
        label = _row_label(row)
        if classify_market(label) == "aggregate":
            region = aggregate_region(label)
            if region:
                covered.add(region)
    for row in rows or []:
        label = _row_label(row)
        if classify_market(label) == "club":
            region = club_region(label)
            if region and region in covered:
                continue
        kept.append(row)
    return kept


def normalise_ladder(rows):
    """Conditional destination distribution: *given that he moves*, where to.

    Returns a list of dicts with ``Destination``, ``Decimal``, ``Implied``,
    ``Kind`` and ``Probability``, sorted by probability descending.
    ``Probability`` sums to 1.0 across the returned rows.

    Not P(leaves) -- see the module docstring.  The feed quotes no stay price, so
    normalising says nothing about whether he moves at all, only about where.
    """
    disjoint = disjoint_ladder(rows)
    priced = []
    for row in disjoint:
        decimal = _row_decimal(row)
        implied = implied_probability(decimal)
        if implied is None:
            continue
        priced.append({
            "Destination": _row_label(row),
            "Decimal": decimal,
            "Implied": implied,
            "Kind": classify_market(_row_label(row)),
        })
    total = sum(p["Implied"] for p in priced)
    if total <= 0:
        return []
    for p in priced:
        p["Probability"] = p["Implied"] / total
    priced.sort(key=lambda p: p["Probability"], reverse=True)
    return priced


def group_ladder(rows):
    """Normalised ladder with each aggregate's member clubs attached beneath it.

    ``disjoint_ladder`` drops "Al Ittihad" because "Any Saudi club" already
    covers it -- correct for the arithmetic, but it throws away the specific-club
    detail that is the most interesting part of a ladder.  This keeps both: the
    aggregate carries the normalised probability, and its members ride along
    under ``Members`` with their own raw prices, never summed into the total.

    Each returned dict is a ``normalise_ladder`` row plus ``Members``, a list of
    ``{Destination, Decimal, Implied, Fractional}`` sorted by implied descending.
    """
    normalised = normalise_ladder(rows)
    by_region = {}
    for row in rows or []:
        label = _row_label(row)
        if classify_market(label) != "club":
            continue
        region = club_region(label)
        if not region:
            continue
        decimal = _row_decimal(row)
        implied = implied_probability(decimal)
        if implied is None:
            continue
        by_region.setdefault(region, []).append({
            "Destination": label,
            "Decimal": decimal,
            "Implied": implied,
            "Fractional": (row.get("Fractional") or row.get("odds")
                           if isinstance(row, dict) else None),
        })

    kept_labels = {p["Destination"] for p in normalised}
    for entry in normalised:
        region = aggregate_region(entry["Destination"])
        members = by_region.get(region, []) if region else []
        # A club kept in the disjoint set is its own row, never also a member.
        members = [m for m in members if m["Destination"] not in kept_labels]
        entry["Members"] = sorted(members, key=lambda m: m["Implied"], reverse=True)
    return normalised


def ladder_overround(rows):
    """Sum of implied probabilities over the disjoint set.

    Above 1.0 for any real book.  Below 1.0 means a parse failure or a missing
    row -- a bookmaker does not offer an arbitrage.

    Do not read the excess as bookmaker margin, and do not use it to back out a
    P(leaves).  These are quoted as *independent* binary bets ("Barcola to
    Liverpool"), each carrying its own margin, rather than as one coupled book
    over a partition of outcomes: the live Mateta ladder totals 1.75 across six
    clubs that do not overlap at all.  So the total is a sanity bound only, and
    ``check_transfer_odds`` warns rather than errors when it runs high -- the
    precise overlap detector is a comparison against this same function.
    """
    total = 0.0
    for row in disjoint_ladder(rows):
        implied = implied_probability(_row_decimal(row))
        if implied is not None:
            total += implied
    return total


def exit_probability(ladder):
    """The market's shortest quoted price on this player leaving, in ``[0, 1]``.

    Every row is a departure, so the best-priced one is the strongest single
    statement the market makes about him going.  Margin is not removed, which
    overstates that destination somewhat while keeping this a fair floor on
    leaving at all.

    Returns ``0.0`` for an empty or unpriceable ladder.
    """
    best = 0.0
    for row in disjoint_ladder(ladder):
        implied = implied_probability(_row_decimal(row))
        if implied is not None and implied > best:
            best = implied
    return max(0.0, min(1.0, best))


# --- Freshness --------------------------------------------------------------

def _coerce_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            _logger.debug("Unparseable odds timestamp: %r", value)
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def odds_age_days(updated, today=None):
    """Age of a quote in days.  ``ODDS_ASSUMED_AGE_DAYS`` when unknown."""
    when = _coerce_datetime(updated)
    if when is None:
        return ODDS_ASSUMED_AGE_DAYS
    ref = _coerce_datetime(today) or datetime.now(timezone.utc)
    age = (ref - when).total_seconds() / 86400.0
    return max(0.0, age)


def odds_age_weight(updated, today=None):
    """Exponential age decay in ``(0, 1]``.

    The live Salah quote is 163 days old and lands near 0.08 -- present on the
    page, all but absent from the score.
    """
    age = odds_age_days(updated, today)
    return max(0.0, min(1.0, 0.5 ** (age / ODDS_HALF_LIFE_DAYS)))


def age_band(age_days):
    """``"live"`` / ``"aging"`` / ``"stale"`` / ``"archival"`` for display."""
    try:
        age = float(age_days)
    except (TypeError, ValueError):
        return "archival"
    if age < AGE_BAND_LIVE:
        return "live"
    if age < AGE_BAND_AGING:
        return "aging"
    if age < AGE_BAND_STALE:
        return "stale"
    return "archival"


# --- Blending ---------------------------------------------------------------

def blend_odds_risk(news_risk, odds_risk, odds_weight):
    """Weighted mean of the news model and the aged-down market price.

    Returns ``news_risk`` unchanged when there is no usable odds signal, so this
    is safe to apply unconditionally.

    Note what a zero ``news_risk`` means here: the news model looked and found
    nothing, which is an observation, not a gap.  So a player with a live 58%
    market and no headlines blends to ~0.25 rather than to 0.58 -- the market
    pulls him off the floor without the app asserting a story no reporter has
    written.  Damping in that direction is deliberate; the opposite error prices
    a matching failure as a transfer saga.
    """
    try:
        news = float(news_risk)
    except (TypeError, ValueError):
        news = 0.0
    news = max(0.0, min(1.0, news if math.isfinite(news) else 0.0))

    try:
        odds = float(odds_risk)
        weight = float(odds_weight)
    except (TypeError, ValueError):
        return news
    if not (math.isfinite(odds) and math.isfinite(weight)) or odds <= 0 or weight <= 0:
        return news

    odds = max(0.0, min(1.0, odds))
    weight = max(0.0, min(1.0, weight))
    w_odds = W_ODDS * weight
    denominator = W_NEWS + w_odds
    if denominator <= 0:
        return news
    return max(0.0, min(1.0, (news * W_NEWS + odds * w_odds) / denominator))
