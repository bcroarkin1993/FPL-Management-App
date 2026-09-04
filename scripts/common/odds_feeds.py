"""Fetch next-club transfer odds from footballtransfers.co.uk.

No caching and no Streamlit here, matching ``transfer_feeds``: the SQLite cache
wrappers live in ``scraping.py`` so this module stays importable from GitHub
Actions.  Every failure path returns an empty frame with the right columns and
logs a warning -- a dead odds feed must never take a page down.

Why this source
---------------
It was rejected once, on the evidence of a per-player page that showed the
player's club as "Unknown" and a five-month-old stamp.  Both observations were
true and remain true; what was missed is that the numbers are still there and
still parseable, and that a stale price is usable once it is *labelled* stale.
``transfer_odds.odds_age_weight`` does the labelling, so this module's only job
is to report faithfully what the page says, including how old it says it is.

The alternatives were re-checked on 2026-09-04: oddschecker.com returns 403 to
any automated fetch, and bettingodds.com publishes prices inside prose with an
FAQ still claiming the window shuts in February 2024.  The Odds API carries no
transfer markets at all.

Page shapes
-----------
Both pages are server-rendered, so nothing here needs a browser:

* ``/odds`` embeds a React Server Component payload with one JSON record per
  player: ``player, slug, rumoredClub, bestOdds, bookmaker, decimal, trending,
  likelihood``.  ``likelihood`` is *not* a probability -- the live feed pairs
  ``decimal: 1.5`` (66.7% implied) with ``likelihood: 90`` -- so it is read but
  never used as one.  A fallback parser reads the visible ticker anchors when
  the payload shape changes.
* ``/odds/<slug>`` carries the ladder as a plain HTML table (club, fractional
  odds, implied probability) and the quote's real timestamp as
  ``semanticOddsUpdatedAt``.  The "Loading club odds..." spinners on that page
  belong to two *other* sections we do not read.
"""

import logging
import re
import threading
import unicodedata

import pandas as pd
import requests

from scripts.common.transfer_odds import (ODDS_INDEX_COLUMNS,
                                          ODDS_LADDER_COLUMNS,
                                          classify_market, implied_probability,
                                          parse_fractional)

_logger = logging.getLogger(__name__)

BASE_URL = "https://www.footballtransfers.co.uk"
ODDS_INDEX_URL = BASE_URL + "/odds"
ODDS_SITEMAP_URL = BASE_URL + "/odds-sitemap.xml"

#: robots.txt allows /odds (only /api/, /_next/ and /admin/ are disallowed).
#: Volume is tiny -- one index request plus a ladder per player expanded.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-GB,en;q=0.9",
}

_SESSION = None
_SESSION_LOCK = threading.Lock()


def _session():
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                s = requests.Session()
                s.headers.update(_HEADERS)
                s.mount("https://", requests.adapters.HTTPAdapter(
                    pool_connections=4, pool_maxsize=8))
                _SESSION = s
    return _SESSION


def _get(url, timeout=15):
    resp = _session().get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def player_slug(name):
    """Player name -> the site's URL slug ("Mohamed Salah" -> "mohamed-salah").

    Accents are stripped, matching the site's own slugs ("Darwin Nunez").
    """
    if not name:
        return ""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


# --- Index ------------------------------------------------------------------

_RECORD_RE = re.compile(
    r'"player":"(?P<player>[^"]+)",'
    r'"slug":"(?P<slug>[^"]+)",'
    r'"rumoredClub":"(?P<club>[^"]*)",'
    r'"bestOdds":"(?P<odds>[^"]*)",'
    r'"bookmaker":"(?P<book>[^"]*)",'
    r'"decimal":(?P<decimal>[0-9.]+)'
    r'(?:,"trending":"(?P<trending>[^"]*)")?'
)

#: Visible ticker anchors, used when the payload shape changes underneath us.
_TICKER_RE = re.compile(
    r'href="/odds/(?P<slug>[a-z0-9-]+)"[^>]*>'
    r'<span[^>]*>(?P<player>[^<]+)</span>'
    r'<span[^>]*>[^<]*</span>'
    r'<span[^>]*>(?P<club>[^<]+)</span>'
    r'<span[^>]*>(?P<odds>[0-9]+(?:/[0-9]+)?)</span>',
    re.I)


def _empty(columns):
    return pd.DataFrame(columns=columns)


def _parse_index(html_text):
    rows, seen = [], set()
    unescaped = html_text.replace('\\"', '"')

    for m in _RECORD_RE.finditer(unescaped):
        slug = m.group("slug")
        if slug in seen:
            continue
        seen.add(slug)
        decimal = None
        try:
            decimal = float(m.group("decimal"))
        except (TypeError, ValueError):
            pass
        if decimal is None or decimal < 1.0:
            decimal = parse_fractional(m.group("odds"))
        rows.append({
            "Player": m.group("player"),
            "Slug": slug,
            "Next_Club": m.group("club"),
            "Fractional": m.group("odds"),
            "Decimal": decimal,
            "Implied": implied_probability(decimal),
            "Bookmaker": m.group("book"),
            "Trending": m.group("trending") or "",
            "Updated": None,
        })

    if not rows:
        _logger.warning("Odds index: JSON payload did not match; using ticker fallback")
        for m in _TICKER_RE.finditer(html_text):
            slug = m.group("slug")
            if slug in seen:
                continue
            seen.add(slug)
            decimal = parse_fractional(m.group("odds"))
            rows.append({
                "Player": m.group("player").strip(),
                "Slug": slug,
                "Next_Club": m.group("club").strip(),
                "Fractional": m.group("odds"),
                "Decimal": decimal,
                "Implied": implied_probability(decimal),
                "Bookmaker": "",
                "Trending": "",
                "Updated": None,
            })
    return rows


def fetch_odds_index(timeout=15):
    """Every player with a live next-club market, one request.

    Columns: ``ODDS_INDEX_COLUMNS``.  ``Updated`` is ``None`` -- the index
    publishes no timestamp, so consumers fall back to
    ``ODDS_ASSUMED_AGE_DAYS`` rather than treating it as fresh.
    """
    try:
        html_text = _get(ODDS_INDEX_URL, timeout=timeout)
    except Exception as exc:
        _logger.warning("Odds index fetch failed: %s", exc)
        return _empty(ODDS_INDEX_COLUMNS)
    try:
        rows = _parse_index(html_text)
    except Exception as exc:
        _logger.warning("Odds index parse failed: %s", exc)
        return _empty(ODDS_INDEX_COLUMNS)
    if not rows:
        _logger.warning("Odds index returned no rows -- page shape may have changed")
        return _empty(ODDS_INDEX_COLUMNS)
    return pd.DataFrame(rows, columns=ODDS_INDEX_COLUMNS)


def fetch_odds_slugs(timeout=15):
    """Slugs of every player with an odds page, from the sitemap.

    Cheaper and broader than the index (101 vs 57 at the time of writing), and
    it lets a caller check whether a market exists before requesting a ladder
    instead of guessing a slug and taking a 404.
    """
    try:
        xml = _get(ODDS_SITEMAP_URL, timeout=timeout)
    except Exception as exc:
        _logger.warning("Odds sitemap fetch failed: %s", exc)
        return []
    return re.findall(r"<loc>[^<]*/odds/([a-z0-9-]+)</loc>", xml)


# --- Per-player ladder ------------------------------------------------------

_UPDATED_RE = re.compile(r'semanticOddsUpdatedAt\\?":\\?"([0-9T:.+\-]+)')
_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")


def _cell_text(fragment):
    return _TAG_RE.sub("", fragment).replace("&amp;", "&").strip()


def _parse_ladder(html_text, player):
    """Pull the club/odds table out of a player page.

    Located by its header text rather than by CSS class, so a restyle does not
    silently return an empty ladder.
    """
    updated = None
    m = _UPDATED_RE.search(html_text)
    if m:
        updated = m.group(1)

    rows = []
    for table in _TABLE_RE.findall(html_text):
        cells_by_row = [[_cell_text(c) for c in _CELL_RE.findall(tr)]
                        for tr in _ROW_RE.findall(table)]
        header = " ".join(cells_by_row[0]).lower() if cells_by_row else ""
        if "odds" not in header:
            continue
        for cells in cells_by_row[1:]:
            if len(cells) < 2:
                continue
            destination, fractional = cells[0], cells[1]
            decimal = parse_fractional(fractional)
            if not destination or decimal is None:
                continue
            rows.append({
                "Player": player,
                "Destination": destination,
                "Fractional": fractional,
                "Decimal": decimal,
                "Implied": implied_probability(decimal),
                "Kind": classify_market(destination),
                "Updated": updated,
            })
        if rows:
            break
    return rows


def fetch_player_odds_ladder(slug, player=None, timeout=15):
    """Full destination ladder for one player.  Columns: ``ODDS_LADDER_COLUMNS``.

    The ladder quotes departures only -- no bookmaker prices "stays at his
    current club" -- so it supports a conditional destination distribution and a
    floor on leaving, not a P(leaves) by normalisation.  See ``transfer_odds``.
    """
    if not slug:
        return _empty(ODDS_LADDER_COLUMNS)
    url = "%s/odds/%s" % (BASE_URL, slug)
    try:
        html_text = _get(url, timeout=timeout)
    except Exception as exc:
        _logger.warning("Odds ladder fetch failed for %s: %s", slug, exc)
        return _empty(ODDS_LADDER_COLUMNS)
    try:
        rows = _parse_ladder(html_text, player or slug)
    except Exception as exc:
        _logger.warning("Odds ladder parse failed for %s: %s", slug, exc)
        return _empty(ODDS_LADDER_COLUMNS)
    if not rows:
        _logger.info("No odds ladder found for %s", slug)
        return _empty(ODDS_LADDER_COLUMNS)
    return pd.DataFrame(rows, columns=ODDS_LADDER_COLUMNS)
