"""
Premier League content API — official predicted-XI graphics, team news, injuries.

premierleague.com publishes a weekly *"Predicted line-ups for every Premier
League team in Matchweek N"* article. **Its predicted XIs are PNG graphics, not
text** — each fixture carries two ``photo-gallery`` widgets whose media resolve
to images titled "Aston Villa Matchweek 4 line up". The article's *words* are
editorial team news. So there is no XI to parse here without OCR, and this
module does not pretend otherwise: it surfaces the graphics as images and the
prose as prose.

What makes it worth having anyway is that ``api.premierleague.com/content/`` is
public and keyless, and carries three things the app did not have:

1. **The article is discoverable per matchweek.** Tag 14349
   (``franchise:predicted-line-ups``) lists every edition, and the title states
   "Matchweek N" — so nothing has to be pinned per gameweek the way
   ``ROTOWIRE_GW1_URL`` is.
2. **The team news is the freshest available.** It is written after the Friday
   press conferences, later than Rotowire's weekly article.
3. **A structured official injury table.** 20 club playlists of
   ``{player, injury type, club article}``. Measured live 2026-09-11: 83 players
   across all 20 clubs, of which 81 matched the FPL pool (97.6%), and **11 that
   the PL reported injured while FPL's bootstrap still rated them fully
   available**.

**The injury table is a watchlist, not ground truth, and can lag FPL.** On
2026-09-11 the PL article prose said Nico O'Reilly was "100 per cent available"
while the PL injury table still listed him with a back problem. Consumers must
show it *beside* the FPL view, never as an override — which is also why nothing
in this module touches ``Start_Pct``, ``Proj`` or the projection engine.

No Streamlit here: plain ``logging``, and the caching wrappers live in
``scraping.py``, the same split as ``ffp_feed.py``.
"""

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

from scripts.common.text_helpers import (
    TEAM_FULL_TO_SHORT,
    canonical_normalize,
    to_display_name,
)

_logger = logging.getLogger("fpl_app.pl_content")

_HEADERS = {"User-Agent": "Mozilla/5.0"}

DEFAULT_TIMEOUT = 20
DEFAULT_ATTEMPTS = 3

#: How many recent editions of the predicted-lineups article to list when
#: hunting for one gameweek. The article is weekly, so ten covers two months.
DISCOVERY_PAGE_SIZE = 10

#: Confidence the fixture-pair vote must reach before it may override or stand in
#: for the matchweek stated in the article title. Same threshold as
#: ``ffp_feed.resolve_ffp_gameweek``.
_GW_VOTE_THRESHOLD = 0.60

#: A club label longer than this is prose that happened to be bold, not a label.
_MAX_CLUB_LABEL = 30

_MATCHWEEK_RE = re.compile(r"Matchweek\s+(\d+)", re.I)
_FIXTURE_HEADING_RE = re.compile(r"^(.+?)\s+v\s+(.+?)\s+predicted line-ups\s*$", re.I)

_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"


class PLLineups(NamedTuple):
    """One edition of the predicted-lineups article, parsed.

    ``fixtures`` are ordered ``(home_short, away_short)`` club codes.
    ``club_news`` and ``graphics`` are keyed on the short code, so a consumer
    holding Rotowire's long club names maps through ``TEAM_FULL_TO_SHORT`` once.
    """

    gameweek: Optional[int] = None
    article_id: Optional[int] = None
    url: str = ""
    title: str = ""
    updated: Optional[datetime] = None
    fixtures: Tuple[Tuple[str, str], ...] = ()
    # Read-only so the shared NamedTuple default cannot be mutated by a consumer.
    club_news: Mapping[str, str] = MappingProxyType({})
    graphics: Mapping[str, str] = MappingProxyType({})
    unresolved_labels: Tuple[str, ...] = ()
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.fixtures and self.club_news)


EMPTY_LINEUPS = PLLineups()

#: Column contract of :func:`fetch_pl_injuries`.
#:
#: ``Item_Date`` is the CMS item's own date and is **not** when the injury was
#: reported — measured live, 46 of 83 rows were over 60 days old and 20 shared
#: the single date 2026-01-19, which is a bulk-authoring artifact. It is carried
#: because it is what the API states, and deliberately not rendered: a stale
#: date under an "as of" label is a confident claim the data does not support.
INJURY_COLUMNS = ["Club", "Team", "Player", "Injury", "Link", "Item_Date"]


# =============================================================================
# Transport
# =============================================================================

def _api_base() -> str:
    try:
        import config
        return str(getattr(config, "PL_CONTENT_API_BASE", "")).rstrip("/") \
            or "https://api.premierleague.com"
    except Exception:
        return "https://api.premierleague.com"


def _get(url: str, timeout: int = DEFAULT_TIMEOUT, attempts: int = DEFAULT_ATTEMPTS,
         session: Optional[requests.Session] = None, **kwargs) -> Optional[requests.Response]:
    """GET with a short backoff. Returns None rather than raising.

    Same contract as ``ffp_feed._get`` and for the same reason: a transient
    timeout that a caller caches as a failure pins "temporarily unavailable" for
    the whole TTL while the site works fine.
    """
    getter = session.get if session is not None else requests.get
    delay = 0.5
    for attempt in range(1, attempts + 1):
        try:
            resp = getter(url, headers=_HEADERS, timeout=timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == attempts:
                _logger.warning("PL: giving up on %s after %d attempts: %s",
                                url, attempts, exc)
                return None
            _logger.info("PL: retrying %s (attempt %d/%d): %s", url, attempt, attempts, exc)
            time.sleep(delay)
            delay *= 2
    return None


def _get_json(url: str, **kwargs) -> Optional[dict]:
    resp = _get(url, **kwargs)
    if resp is None:
        return None
    try:
        return resp.json()
    except Exception as exc:
        _logger.warning("PL: %s did not return JSON: %s", url, exc)
        return None


# =============================================================================
# Gameweek resolution
# =============================================================================

def _fpl_fixture_pairs(timeout: int = DEFAULT_TIMEOUT) -> Dict[int, set]:
    """``{gameweek: {(home_short, away_short), ...}}`` from the FPL fixture list."""
    boot = _get_json(_BOOTSTRAP_URL, timeout=timeout)
    fixtures = _get_json(_FIXTURES_URL, timeout=timeout)
    if not boot or not fixtures:
        return {}
    try:
        short = {t["id"]: t["short_name"] for t in boot.get("teams", [])}
        out: Dict[int, set] = {}
        for fixture in fixtures:
            event = fixture.get("event")
            if not event:
                continue
            home, away = short.get(fixture.get("team_h")), short.get(fixture.get("team_a"))
            if home and away:
                out.setdefault(int(event), set()).add((home, away))
        return out
    except Exception as exc:
        _logger.warning("PL: could not build FPL fixture pairs: %s", exc)
        return {}


def resolve_gameweek_from_fixtures(
    fixtures: Optional[Tuple[Tuple[str, str], ...]],
    fixture_pairs: Optional[Dict[int, set]] = None,
) -> Optional[int]:
    """Vote the article's ordered ``(home, away)`` pairs against the real fixtures.

    A *set of club names* cannot identify a gameweek — all 20 clubs play every
    week — but ordered pairs can. Same technique as
    ``ffp_feed.resolve_ffp_gameweek``; here it is the cross-check on the
    matchweek the title states, and the fallback if that regex ever stops
    matching. Returns None when no week wins clearly, because "unknown" must
    never be reported as a gameweek.
    """
    pairs = {p for p in (fixtures or ()) if p and p[0] and p[1]}
    if not pairs:
        return None
    if fixture_pairs is None:
        fixture_pairs = _fpl_fixture_pairs()
    if not fixture_pairs:
        return None

    scored = sorted(
        ((len(pairs & known) / len(pairs), gw) for gw, known in fixture_pairs.items()),
        reverse=True,
    )
    if not scored or scored[0][0] < _GW_VOTE_THRESHOLD:
        return None
    if len(scored) > 1 and scored[1][0] == scored[0][0]:
        return None                     # ambiguous resolves to no answer
    return int(scored[0][1])


def _matchweek_from_title(title: Optional[str]) -> Optional[int]:
    match = _MATCHWEEK_RE.search(str(title or ""))
    return int(match.group(1)) if match else None


def _epoch_ms_to_dt(value) -> Optional[datetime]:
    try:
        ms = int(value)
    except (TypeError, ValueError):
        return None
    if ms <= 0:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


# =============================================================================
# Predicted-lineups article
# =============================================================================

def list_predicted_lineup_articles(
    limit: int = DISCOVERY_PAGE_SIZE, timeout: int = DEFAULT_TIMEOUT
) -> List[dict]:
    """Recent editions, newest first: ``[{id, title, gameweek, date}, ...]``."""
    try:
        import config
        tag = int(getattr(config, "PL_PREDICTED_LINEUPS_TAG", 14349))
    except Exception:
        tag = 14349

    url = "%s/content/premierleague/text/EN?tagIds=%d&pageSize=%d" % (
        _api_base(), tag, max(1, int(limit)))
    payload = _get_json(url, timeout=timeout)
    if not payload:
        return []

    out = []
    for item in (payload.get("content") or []):
        out.append({
            "id": item.get("id"),
            "title": item.get("title") or "",
            "gameweek": _matchweek_from_title(item.get("title")),
            "date": item.get("date") or "",
        })
    return out


def fetch_article(article_id: int, timeout: int = DEFAULT_TIMEOUT) -> Optional[dict]:
    url = "%s/content/premierleague/text/EN/%s?detail=DETAILED" % (_api_base(), article_id)
    return _get_json(url, timeout=timeout)


def fetch_photo(photo_id, timeout: int = DEFAULT_TIMEOUT,
                session: Optional[requests.Session] = None) -> Optional[dict]:
    url = "%s/content/premierleague/photo/EN/%s?detail=DETAILED" % (_api_base(), photo_id)
    return _get_json(url, timeout=timeout, session=session)


def _club_label_and_body(paragraph) -> Tuple[Optional[str], str]:
    """Split ``<p><strong>Club: </strong>prose</p>`` into label and prose.

    **The label is the text before the first colon *inside* the ``<strong>``,
    not the whole ``<strong>``.** For 16 of 20 clubs the markup is the tidy form
    above; for Bournemouth, Liverpool, Coventry and Leeds the ``<strong>`` wraps
    the *entire paragraph*. Taking ``strong.get_text()`` whole yields 16 clubs
    and silently drops 4 — verified against the live MW4 payload, which is
    exactly the plausible-but-incomplete failure nothing downstream can catch.
    """
    strong = paragraph.find("strong")
    if strong is None:
        return None, ""
    head, sep, _ = strong.get_text(" ", strip=True).partition(":")
    label = head.strip()
    if not sep or not label or len(label) > _MAX_CLUB_LABEL:
        return None, ""
    if label.lower().startswith("see"):
        return None, ""                 # "See: <club> team news" link paragraphs
    body = paragraph.get_text(" ", strip=True).partition(":")[2].strip()
    return label, body


def parse_predicted_lineups(article: Optional[dict],
                            fixture_pairs: Optional[Dict[int, set]] = None,
                            resolve_graphics: bool = True,
                            timeout: int = DEFAULT_TIMEOUT) -> PLLineups:
    """Parse one article payload into :class:`PLLineups`. Never raises."""
    if not article:
        return EMPTY_LINEUPS._replace(note="no article payload")

    title = str(article.get("title") or "")
    article_id = article.get("id")
    slug = article.get("titleUrlSegment") or ""
    url = "https://www.premierleague.com/en/news/%s/%s" % (article_id, slug) \
        if article_id else ""

    try:
        soup = BeautifulSoup(article.get("body") or "", "html.parser")
    except Exception as exc:
        return EMPTY_LINEUPS._replace(note="body would not parse: %s" % exc)

    fixtures: List[Tuple[str, str]] = []
    club_news: Dict[str, str] = {}
    unresolved: List[str] = []
    media_by_fixture: List[List[str]] = []

    for element in soup.find_all(["h5", "p", "div"]):
        if element.name == "h5":
            heading = _FIXTURE_HEADING_RE.match(element.get_text(" ", strip=True))
            if not heading:
                continue
            home_label, away_label = heading.group(1).strip(), heading.group(2).strip()
            home, away = (TEAM_FULL_TO_SHORT.get(home_label),
                          TEAM_FULL_TO_SHORT.get(away_label))
            for label, code in ((home_label, home), (away_label, away)):
                if code is None:
                    unresolved.append(label)
            fixtures.append((home or "", away or ""))
            media_by_fixture.append([])
            continue

        if element.name == "div":
            media_id = element.get("data-media-id")
            if media_id and media_by_fixture:
                media_by_fixture[-1].append(str(media_id))
            continue

        label, body = _club_label_and_body(element)
        if not label or not body:
            continue
        code = TEAM_FULL_TO_SHORT.get(label)
        if code is None:
            unresolved.append(label)
            continue
        club_news[code] = body

    fixtures_t = tuple(fixtures)
    stated = _matchweek_from_title(title)
    voted = resolve_gameweek_from_fixtures(fixtures_t, fixture_pairs)

    note = ""
    gameweek = stated
    if stated is None:
        gameweek = voted
        note = "matchweek not stated in title; taken from fixture vote"
    elif voted is not None and voted != stated:
        # Trust neither: the article says one week and its own fixtures say
        # another, which is the state "never report unknown as a gameweek" and
        # "a source must be able to prove its own gameweek" both exist for.
        gameweek = None
        note = "title says MW%d but its fixtures vote MW%d" % (stated, voted)

    graphics = {}
    if resolve_graphics:
        graphics = _resolve_graphics(fixtures_t, media_by_fixture, gameweek, timeout)

    return PLLineups(
        gameweek=gameweek,
        article_id=article_id,
        url=url,
        title=title,
        updated=_epoch_ms_to_dt(article.get("lastModified"))
        or _epoch_ms_to_dt(article.get("publishFrom")),
        fixtures=fixtures_t,
        club_news=club_news,
        graphics=graphics,
        unresolved_labels=tuple(dict.fromkeys(unresolved)),
        note=note,
    )


def _resolve_graphics(fixtures, media_by_fixture, gameweek, timeout) -> Dict[str, str]:
    """Map each XI graphic to its club.

    Media ids appear in document order, two per fixture, home then away — but
    position alone would render one club's XI under another club's name the
    first time the PL adds a third image to a section. The photo's own title
    ("Aston Villa Matchweek 4 line up") names the club, so position proposes and
    the title confirms; a pair that disagrees is dropped rather than shown.
    """
    jobs = []
    for fixture, media in zip(fixtures, media_by_fixture):
        if len(media) != 2:
            _logger.info("PL: %s had %d graphics, expected 2", fixture, len(media))
            continue
        for code, media_id in zip(fixture, media):
            if code:
                jobs.append((code, media_id))
    if not jobs:
        return {}

    session = requests.Session()
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            photos = list(pool.map(
                lambda job: fetch_photo(job[1], timeout=timeout, session=session), jobs))
    except Exception as exc:
        _logger.warning("PL: graphic lookup failed: %s", exc)
        return {}
    finally:
        session.close()

    out = {}
    for (code, media_id), photo in zip(jobs, photos):
        if not photo:
            continue
        image = photo.get("onDemandUrl") or photo.get("imageUrl")
        if not image:
            continue
        claimed = _club_from_photo_title(photo.get("title"))
        if claimed is not None and claimed != code:
            _logger.warning("PL: graphic %s is titled %r but sits in %s's slot — dropped",
                            media_id, photo.get("title"), code)
            continue
        stated_gw = _matchweek_from_title(photo.get("title"))
        if gameweek is not None and stated_gw is not None and stated_gw != gameweek:
            _logger.warning("PL: graphic %s says MW%d, article says MW%s — dropped",
                            media_id, stated_gw, gameweek)
            continue
        out[code] = image
    return out


def _club_from_photo_title(title: Optional[str]) -> Optional[str]:
    """Club code from "Aston Villa Matchweek 4 line up", or None if unrecognised."""
    text = str(title or "")
    head = _MATCHWEEK_RE.split(text)[0].strip()
    return TEAM_FULL_TO_SHORT.get(head)


def get_predicted_lineups(gameweek: Optional[int] = None,
                          timeout: int = DEFAULT_TIMEOUT) -> PLLineups:
    """Discover and parse the edition covering ``gameweek``. Never raises."""
    listing = list_predicted_lineup_articles(timeout=timeout)
    if not listing:
        return EMPTY_LINEUPS._replace(note="could not list PL predicted-lineup articles")

    if gameweek is None:
        chosen = listing[0]
    else:
        chosen = next((a for a in listing if a.get("gameweek") == int(gameweek)), None)
        if chosen is None:
            latest = listing[0].get("gameweek")
            return EMPTY_LINEUPS._replace(
                note="PL has not published Matchweek %s yet (latest is %s)"
                     % (gameweek, latest if latest is not None else "unknown"))

    article = fetch_article(chosen["id"], timeout=timeout)
    if not article:
        return EMPTY_LINEUPS._replace(
            note="could not fetch PL article %s" % chosen.get("id"))
    return parse_predicted_lineups(article, timeout=timeout)


# =============================================================================
# Official injury table
# =============================================================================

def _playlist_url(playlist_id) -> str:
    return ("%s/content/premierleague/playlist/EN/%s"
            "?detail=DETAILED&fullObjectResponse=true" % (_api_base(), playlist_id))


def fetch_pl_injuries(timeout: int = DEFAULT_TIMEOUT, workers: int = 8) -> pd.DataFrame:
    """The official per-club injury table. Empty frame on any failure.

    One hub request plus one per club: a ``detail``/``depth``/``expandNested``
    parameter sweep confirmed the hub returns only child ids, so the N+1 is the
    API's shape rather than a missed optimisation. Parallelised over a shared
    session, the 21 requests take about a second.
    """
    try:
        import config
        hub_id = getattr(config, "PL_INJURY_PLAYLIST_ID", 4509826)
    except Exception:
        hub_id = 4509826

    hub = _get_json(_playlist_url(hub_id), timeout=timeout)
    if not hub:
        return pd.DataFrame(columns=INJURY_COLUMNS)

    clubs = []
    for item in (hub.get("items") or []):
        response = item.get("response") or {}
        club_id, title = response.get("id"), response.get("title") or ""
        if club_id is not None:
            clubs.append((club_id, title.replace("Injury News - ", "").strip()))
    if not clubs:
        _logger.warning("PL: injury hub listed no clubs")
        return pd.DataFrame(columns=INJURY_COLUMNS)

    session = requests.Session()
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            payloads = list(pool.map(
                lambda c: _get_json(_playlist_url(c[0]), timeout=timeout, session=session),
                clubs))
    except Exception as exc:
        _logger.warning("PL: injury club fetch failed: %s", exc)
        return pd.DataFrame(columns=INJURY_COLUMNS)
    finally:
        session.close()

    rows = []
    for (_, club), payload in zip(clubs, payloads):
        if not payload:
            _logger.info("PL: no injury payload for %s", club)
            continue
        for item in (payload.get("items") or []):
            record = item.get("response") or {}
            name = (record.get("title") or "").strip()
            if not name:
                continue
            links = record.get("links") or []
            injury = (record.get("description") or "").strip()
            rows.append({
                "Club": club,
                "Team": TEAM_FULL_TO_SHORT.get(club),
                "Player": name,
                # The PL writes "-" where it has no detail; an empty string reads
                # as "no information" rather than as a claim.
                "Injury": "" if injury in {"-", "—"} else injury,
                "Link": (links[0] or {}).get("promoUrl") if links else None,
                "Item_Date": (record.get("date") or "")[:10],
            })

    if not rows:
        return pd.DataFrame(columns=INJURY_COLUMNS)
    return pd.DataFrame(rows, columns=INJURY_COLUMNS)


# =============================================================================
# Matching onto the FPL pool
# =============================================================================

def attach_pl_injuries(pool_df: pd.DataFrame,
                       injuries_df: pd.DataFrame) -> pd.DataFrame:
    """Match the PL injury table onto an FPL player pool, scoped by club.

    Returns ``Player_ID, PL_Player, PL_Injury, PL_Injury_Link`` for the players
    that matched; the rest are simply absent. The CMS ``Item_Date`` is
    deliberately not carried through — see :data:`INJURY_COLUMNS`.

    Why not ``ReferenceMatcher``: the PL publishes a name and a club and no
    position, and every tier of the shared matcher below the first two is scoped
    by position — with ``position=None`` they are all skipped, leaving an exact
    ``(name, team)`` key that misses the ~16% of players FPL files under a full
    legal name. This is the situation ``transfer_risk_app.attach_odds`` is
    documented as the exception for, so it uses the same two rules:

    1. **A key is kept only when it resolves to exactly one player.**
    2. **The fallback is a token subset either direction, never a bare
       surname** — a shared surname is how Alex Palmer once acquired Cole
       Palmer's stats.

    Scoping every lookup to the club the PL filed the player under makes both
    rules much safer than they are league-wide: the candidate set is ~25 players.
    Measured live 2026-09-11: 81 of 83 matched (97.6%), the two misses being
    players absent from the FPL pool entirely rather than matcher failures.
    """
    columns = ["Player_ID", "PL_Player", "PL_Injury", "PL_Injury_Link"]
    if (pool_df is None or getattr(pool_df, "empty", True)
            or injuries_df is None or getattr(injuries_df, "empty", True)):
        return pd.DataFrame(columns=columns)
    if "Player_ID" not in pool_df.columns or "Team" not in pool_df.columns:
        return pd.DataFrame(columns=columns)

    by_club: Dict[str, List[tuple]] = {}
    for row in pool_df.itertuples(index=False):
        team = getattr(row, "Team", None)
        if not team or pd.isna(team):
            continue
        keys = set()
        for value in (getattr(row, "Player", None),
                      getattr(row, "Display_Name", None),
                      getattr(row, "Web_Name", None)):
            key = canonical_normalize(str(value)) if value and not pd.isna(value) else ""
            if key:
                keys.add(key)
        if not keys:
            continue
        full = canonical_normalize(str(getattr(row, "Player", "") or ""))
        tokens = {t for t in full.split() if len(t) > 1}
        by_club.setdefault(str(team), []).append(
            (getattr(row, "Player_ID"), keys, tokens))

    rows, unmatched = [], []
    for record in injuries_df.itertuples(index=False):
        team = getattr(record, "Team", None)
        name = str(getattr(record, "Player", "") or "")
        key = canonical_normalize(name)
        candidates = by_club.get(str(team)) if team and not pd.isna(team) else None
        if not key or not candidates:
            unmatched.append(name)
            continue

        hits = [pid for pid, keys, _ in candidates if key in keys]
        if len(hits) != 1:
            query = {t for t in key.split() if len(t) > 1}
            hits = [pid for pid, _, tokens in candidates
                    if query and tokens and (query <= tokens or tokens <= query)] \
                if query else []
        if len(hits) != 1:
            unmatched.append(name)
            continue

        rows.append({
            "Player_ID": hits[0],
            "PL_Player": name,
            "PL_Injury": getattr(record, "Injury", "") or "",
            "PL_Injury_Link": getattr(record, "Link", None),
        })

    if unmatched:
        _logger.info("PL: %d of %d injury rows unmatched (%s)",
                     len(unmatched), len(injuries_df), ", ".join(unmatched[:8]))
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).drop_duplicates(subset=["Player_ID"])


def add_display_names(pool_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a ``Display_Name`` column, since the PL publishes common names.

    "Cody Gakpo" is what the PL writes and what ``to_display_name`` produces;
    the bootstrap's ``Player`` is "Cody Gakpo" only by luck and is usually the
    full legal name. Having both as exact keys is most of the 97.6%.
    """
    if pool_df is None or getattr(pool_df, "empty", True):
        return pool_df
    if "Display_Name" in pool_df.columns:
        return pool_df
    out = pool_df.copy()
    if "Player" not in out.columns:
        return out

    def _display(row):
        player = str(row.get("Player") or "")
        first, _, rest = player.partition(" ")
        return to_display_name(first, rest, row.get("Web_Name"))

    out["Display_Name"] = out.apply(_display, axis=1)
    return out
