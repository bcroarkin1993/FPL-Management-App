"""
Transfer news fetching — Google News RSS, one targeted query per player.

Why this source: it is free, needs no API key, is valid RSS parseable with the
standard library, and — unlike a betting feed — it is *timely*.  The Watkins move
was reported by Sky, Transfermarkt, the Telegraph, the Guardian and the Sun for
days before it completed, while the bookmaker page that also covered him was
stale to March and still priced a Saudi move at 6%.

Querying per player rather than filtering a firehose means precision comes from
the quoted query, and the returned set is already about the player.

No Streamlit and no caching here: this module is imported by GitHub Actions
(``waiver_alerts.py``), which cannot pull in ``error_helpers`` or ``cache`` —
both reach Streamlit.  Caching wrappers live in ``scraping.py`` on the app side.
"""

import logging
import re
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote_plus

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

_logger = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

# Google appends " - Publisher" to every headline; strip it for display.
_TITLE_SUFFIX_RE = re.compile(r"\s+-\s+[^-]{2,40}$")

# One request per player, so a 150-player board is 150 round trips. Serialised
# with a sleep between each that was ~2 minutes; the work is entirely network
# latency, so it parallelises almost perfectly. Concurrency is what limits the
# request rate now, instead of a sleep.
DEFAULT_WORKERS = 8
# Hard ceiling on a single batch run, so a misconfigured pool cannot fire
# thousands of requests.
MAX_BATCH_PLAYERS = 400

_SESSION = None
_SESSION_LOCK = threading.Lock()


def _session() -> requests.Session:
    """Shared connection pool.

    Every request goes to the same host, so reusing connections skips a TLS
    handshake per player — a large share of the per-request cost at this size.
    """
    global _SESSION
    if _SESSION is None:
        with _SESSION_LOCK:
            if _SESSION is None:
                sess = requests.Session()
                adapter = HTTPAdapter(pool_connections=DEFAULT_WORKERS,
                                      pool_maxsize=DEFAULT_WORKERS * 2)
                sess.mount("https://", adapter)
                sess.mount("http://", adapter)
                _SESSION = sess
    return _SESSION

NEWS_COLUMNS = ["Player", "Team", "Headline", "URL", "Published", "Source"]


def build_query_url(player: str, team=None) -> str:
    """Quoted per-player transfer query, UK edition.

    A mononym ("Gabriel", "Rayan") is not a search term — quoting it still returns
    every Gabriel in the league — so the club is added to the query for those.
    Multi-token names are left alone, since adding terms narrows recall and their
    quoted name is already specific.
    """
    name = (player or "").strip()
    terms = '"%s" transfer' % name
    if team and len(name.split()) == 1:
        terms = '"%s" %s transfer' % (name, str(team).strip())
    return "%s?q=%s&hl=en-GB&gl=GB&ceid=GB:en" % (GOOGLE_NEWS_RSS, quote_plus(terms))


def _clean_title(title: str):
    """Drop Google's trailing ' - Publisher' suffix."""
    if not title:
        return "", ""
    match = _TITLE_SUFFIX_RE.search(title)
    if match:
        return title[: match.start()].strip(), match.group(0).lstrip(" -").strip()
    return title.strip(), ""


def _item_source(item, fallback: str) -> str:
    """Publisher name — from the <source> element, else the title suffix."""
    for path in ("{*}source", "source"):
        try:
            node = item.find(path)
        except (SyntaxError, KeyError):
            node = None
        if node is not None and (node.text or "").strip():
            return node.text.strip()
    return fallback


def fetch_player_transfer_news(player: str, team=None, timeout: int = 15) -> pd.DataFrame:
    """Recent transfer headlines for one player.

    Returns an empty frame with the expected columns on any failure — a news
    outage must never take a page down, matching the contract on
    ``get_rotowire_article_updated``.
    """
    empty = pd.DataFrame(columns=NEWS_COLUMNS)
    if not player or not str(player).strip():
        return empty

    url = build_query_url(player, team)
    try:
        resp = _session().get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError, ValueError) as e:
        _logger.warning("Transfer news fetch failed for %s: %s", player, e)
        return empty

    rows = []
    for item in root.findall(".//item"):
        raw_title = (item.findtext("title") or "").strip()
        if not raw_title:
            continue
        title, suffix = _clean_title(raw_title)
        rows.append({
            "Player": player,
            "Team": team or "",
            "Headline": title,
            "URL": (item.findtext("link") or "").strip(),
            "Published": (item.findtext("pubDate") or "").strip(),
            "Source": _item_source(item, suffix),
        })

    if not rows:
        _logger.debug("No transfer news items for %s", player)
        return empty
    return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def fetch_transfer_news_batch(players, workers: int = DEFAULT_WORKERS,
                              max_players: int = MAX_BATCH_PLAYERS,
                              timeout: int = 15, progress=None,
                              on_result=None) -> pd.DataFrame:
    """Fetch news for many players concurrently.

    ``players`` is an iterable of ``(player, team)`` pairs.

    ``progress`` is an optional ``callable(done, total, player)``, invoked on the
    *calling* thread as results land — safe to drive a Streamlit progress bar from,
    which a worker thread would not be.

    ``on_result`` is an optional ``callable(player, records)`` fired per player as
    each completes, so a caller can persist results incrementally rather than
    losing everything if the run is interrupted.

    Individual failures are swallowed and logged; the batch always returns
    whatever it managed to collect.
    """
    pairs = []
    seen = set()
    for entry in players or []:
        if isinstance(entry, (tuple, list)):
            name, team = (list(entry) + [None])[:2]
        else:
            name, team = entry, None
        key = str(name).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        pairs.append((name, team))

    if len(pairs) > max_players:
        _logger.warning("Transfer news batch capped at %d of %d players",
                        max_players, len(pairs))
        pairs = pairs[:max_players]

    total = len(pairs)
    if not total:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    if progress is not None:
        try:
            progress(0, total, "")
        except Exception:
            pass

    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(fetch_player_transfer_news, name, team, timeout): (name, team)
            for name, team in pairs
        }
        for future in as_completed(futures):
            name, _team = futures[future]
            try:
                df = future.result()
            except Exception as e:  # fetch_player_transfer_news already guards, belt and braces
                _logger.warning("Transfer news task failed for %s: %s", name, e)
                df = pd.DataFrame(columns=NEWS_COLUMNS)

            if not df.empty:
                frames.append(df)
            if on_result is not None:
                try:
                    on_result(name, df.to_dict("records"))
                except Exception:
                    _logger.warning("Transfer news on_result failed for %s", name, exc_info=True)

            done += 1
            if progress is not None:
                try:
                    progress(done, total, name)
                except Exception:
                    pass

    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    return pd.concat(frames, ignore_index=True)
