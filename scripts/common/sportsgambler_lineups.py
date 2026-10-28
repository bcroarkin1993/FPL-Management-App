"""
Sportsgambler's projected-lineups page -- pure, Streamlit-free.

**Why a second projected XI at all.** The engine took "does he start" from one
publication, and a benching is the direction that *inverts* a projection rather
than degrading it. Reported from the app at GW8: Rayan Cherki read **21%** to
start against Aston Villa off a 5.71 conditional projection, because Rotowire's
rankings article did not price him and its lineups page left him out of a
complete City XI. Rotowire's XI instead started Ayyoub Bouaddi, whom FFP rated
**20%** with one start all season, over Cherki at **50%** with five starts in
eight. Nothing in the app could corroborate or contradict either call.

Three XIs disagreed about City that week -- Rotowire, this source, and the
reader's own -- and they agreed on only eight of eleven. Phil Foden was the
clearest case: absent from Rotowire, named by both of the others, and crushed to
0.21 by one source's silence.

**What this page gives us.** The index lists every fixture with an id, and
``/lineups/lineups-load2.php?id=<id>`` returns server-rendered HTML per fixture:
``.lineups-home`` / ``.lineups-away``, eleven ``.player-name`` nodes each. The
names resolve at **99.0%** (196 of 198, GW8) through
``rotowire_lineups.resolve_lineup_name`` with no new matching code, so this
module is a scraper and nothing more: the resolver, the per-club benching gate
and the ``LineupXI`` type are all imported rather than re-implemented. A second
copy of a lineup gate is the twin-drift failure this codebase keeps paying for.

Two traps specific to this source:

* **A ``confirmed`` CSS class sits on every fixture button while the heading
  says "Predicted Lineup".** It is styling. Reading it as a reliability flag
  would promote a prediction to a team sheet.
* **The page carries about five gameweeks**, not one plus a stray. So unlike
  ``rotowire_lineups._matchup_is_in_gameweek``, which fails *open* because an
  extra matchup only costs a card, the gameweek filter here fails **closed**:
  without the fixture list, five gameweeks of XIs stacked together is not a
  lineup call at all, and the union of five would over-promote half the league.
  Filtering before fetching also drops 50 requests to 10.

Streamlit-free is a hard constraint, for the reason ``projection_engine`` and
``rotowire_lineups`` carry it: the Actions snapshot collector imports them.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup
import pandas as pd
import requests

import config
from scripts.common.rotowire_lineups import (
    LINEUP_COLUMNS,
    LineupScrape,
    LineupXI,
    _bootstrap_pool,
    _gameweek_fixture_pairs,
    lineup_starters,
)
from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

# Plain logging, not `error_helpers.get_logger`: that reaches Streamlit.
_logger = logging.getLogger("fpl_app.sportsgambler_lineups")

SOURCE_LABEL = "Sportsgambler"

#: The per-fixture XI endpoint, relative to the site root.
_AJAX_PATH = "/lineups/lineups-load2.php"

#: A module constant, never a parameter. Streamlit hashes every argument of a
#: cached function into its key, and the Rotowire dedupe silently did nothing
#: for exactly this reason -- two callers passing different timeouts produced
#: two cache entries and two downloads.
_TIMEOUT = 20
_ATTEMPTS = 3

#: Sportsgambler serves nothing useful to a bare client.
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-GB,en;q=0.9",
}

#: Ten fixtures a gameweek, so a small pool is plenty and stays polite.
_MAX_WORKERS = 5

#: Successes only, and shorter than the Streamlit TTL above it, so it can never
#: serve a payload older than that cache would have. Same rules as
#: ``ffp_feed._memoised`` and ``rotowire_lineups._memoised``.
_MEMO_TTL_SECONDS = 60
_memo: Dict[str, tuple] = {}


def _memoised(key: str, fn, ttl: int = _MEMO_TTL_SECONDS):
    now = time.time()
    hit = _memo.get(key)
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    if value is not None and getattr(value, "ok", True):
        _memo[key] = (now, value)
    return value


def clear_memo() -> None:
    """Drop the in-process memo. For tests and the Refresh buttons."""
    _memo.clear()


def _get(url: str, session: Optional[requests.Session] = None,
         **kwargs) -> Optional[requests.Response]:
    """GET with a short backoff. Returns None rather than raising.

    The house pattern (``ffp_feed._get``): a transient timeout must not be
    mistaken for a source that has changed shape, and must never be cached as
    one.
    """
    getter = session.get if session is not None else requests.get
    delay = 0.5
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            resp = getter(url, headers=_HEADERS, timeout=_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == _ATTEMPTS:
                _logger.warning("Sportsgambler: giving up on %s after %d attempts: %s",
                                url, _ATTEMPTS, exc)
                return None
            _logger.info("Sportsgambler: retrying %s (%d/%d): %s",
                         url, attempt, _ATTEMPTS, exc)
            time.sleep(delay)
            delay *= 2
    return None


def _site_root(url: str) -> str:
    """``https://host`` from any page URL on the site."""
    parts = str(url or "").split("/")
    return "/".join(parts[:3]) if len(parts) >= 3 else ""


def parse_index(html: str) -> List[Tuple[str, str, str]]:
    """``[(fixture_id, home_label, away_label)]`` in the page's own order.

    The id is the anchor's ``id`` and is what the AJAX endpoint takes; the club
    labels come from the enclosing row, home first.
    """
    rows: List[Tuple[str, str, str]] = []
    if not html:
        return rows
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.select("a.lineups-toggle-button"):
        fixture_id = (anchor.get("id") or "").strip()
        if not fixture_id:
            continue
        row = anchor.find_parent("div", class_="lineup-row")
        if row is None:
            continue
        teams = [e.get_text(" ", strip=True) for e in row.select(".fxs-team")]
        if len(teams) != 2 or not all(teams):
            continue
        rows.append((fixture_id, teams[0], teams[1]))
    return rows


def _fixtures_in_gameweek(
    index: Sequence[Tuple[str, str, str]],
    fixture_pairs,
) -> List[Tuple[str, str, str]]:
    """The index rows belonging to one gameweek, by ordered ``(home, away)`` pair.

    **Per fixture, never a page-wide vote.** The page carries about five
    gameweeks and every one of them is a complete set of ten, so a
    ``resolve_ffp_gameweek``-style vote over all fifty pairs scores GW8, GW9 and
    GW10 at ten each and cannot choose between them. An ordered pair occurs at
    most once in a season, so a direct lookup is exact.

    **Fails closed** (see the module docstring): an empty ``fixture_pairs``
    yields nothing rather than everything.
    """
    if not fixture_pairs:
        _logger.warning(
            "Sportsgambler: no fixture list for this gameweek, so no XI is "
            "resolved -- this page carries several gameweeks and the union of "
            "them is not a lineup call.")
        return []
    kept = []
    for fixture_id, home, away in index:
        h = TEAM_FULL_TO_SHORT.get(str(home).strip())
        a = TEAM_FULL_TO_SHORT.get(str(away).strip())
        if not h or not a:
            _logger.info("Sportsgambler: unmapped club label in %r vs %r", home, away)
            continue
        if (h, a) in fixture_pairs:
            kept.append((fixture_id, home, away))
    return kept


def parse_fixture_xi(html: str) -> Tuple[List[str], List[str]]:
    """``(home_names, away_names)`` from one fixture's AJAX fragment.

    Eleven ``.player-name`` nodes per side. The shirt number lives in a sibling
    ``.player-profile`` and is deliberately ignored: it is not a position, and
    the resolver does better with the name alone than with a number it would
    have to guess the meaning of.
    """
    if not html:
        return [], []
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for side in ("lineups-home", "lineups-away"):
        box = soup.select_one("." + side)
        names = ([e.get_text(" ", strip=True) for e in box.select(".player-name")]
                 if box is not None else [])
        out.append([n for n in names if n])
    return out[0], out[1]


def scrape_lineups(url: str, gameweek: Optional[int] = None) -> LineupScrape:
    """One pass over the index plus one request per fixture in ``gameweek``.

    Returns the same ``(players, matchups)`` shape ``rotowire_lineups`` does, in
    a single ``MatchupIndex`` space, so ``lineup_starters`` consumes it
    unchanged.

    ``Position`` is emitted empty. Sportsgambler publishes none, and
    ``resolve_lineup_name`` already retries across all four positions and
    resolves by match tier -- which is what the 99% resolution rate above was
    measured with. Deriving a position from the ``.players-line`` index is
    formation-dependent guesswork for no gain.
    """
    empty = LineupScrape(pd.DataFrame(columns=LINEUP_COLUMNS), [])
    root = _site_root(url)
    if not root:
        return empty

    index_resp = _get(url)
    if index_resp is None:
        return empty
    index = parse_index(index_resp.text)
    if not index:
        _logger.warning("Sportsgambler: no fixtures parsed from the lineups index.")
        return empty

    wanted = _fixtures_in_gameweek(index, _gameweek_fixture_pairs(gameweek))
    if not wanted:
        return empty

    ajax = root + _AJAX_PATH
    with requests.Session() as session:
        session.headers.update({"X-Requested-With": "XMLHttpRequest", "Referer": url})

        def _one(entry):
            fixture_id = entry[0]
            resp = _get(ajax, session=session, params={"id": fixture_id})
            return entry, (resp.text if resp is not None else "")

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            fetched = list(pool.map(_one, wanted))

    rows = []
    matchups = []
    for (fixture_id, home, away), html in fetched:
        home_names, away_names = parse_fixture_xi(html)
        if not home_names and not away_names:
            _logger.info("Sportsgambler: no XI parsed for %s vs %s", home, away)
            continue
        matchup_index = len(matchups)
        for label, names in ((home, home_names), (away, away_names)):
            for name in names:
                rows.append((label, "", name, matchup_index))
        # Assigned after filtering, so indices stay contiguous -- the renderer
        # pairs home and away by this index and a gap shows one side only.
        matchups.append((home, away))

    return LineupScrape(pd.DataFrame(rows, columns=LINEUP_COLUMNS), matchups)


def fetch_lineup_xi(url=None, gameweek=None) -> LineupXI:
    """Sportsgambler's projected starters for ``gameweek``, as FPL element ids.

    The whole pipeline: scrape, resolve against the bootstrap, gate per club.
    Never raises -- a failure is an empty ``LineupXI``, which the projection
    engine reads as "no opinion", leaving the other XI source in sole charge
    exactly as it was before this existed.
    """
    if url is None:
        url = getattr(config, "SPORTSGAMBLER_LINEUPS_URL", None)
    if not url:
        return LineupXI(set(), set(), gameweek,
                        "No Sportsgambler lineups URL configured.")
    if gameweek is None:
        gameweek = config.CURRENT_GAMEWEEK

    def _build():
        try:
            players = scrape_lineups(url, gameweek).players
            return lineup_starters(players, _bootstrap_pool(), gameweek,
                                   source=SOURCE_LABEL)
        except Exception as e:              # pragma: no cover - never break a blend
            _logger.warning("Sportsgambler lineups: could not resolve XIs: %s", e)
            return LineupXI(set(), set(), gameweek, str(e))

    return _memoised("lineup_xi:%s:%s" % (url, gameweek), _build)
