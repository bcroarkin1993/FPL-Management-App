"""
Rotowire's projected-lineups page -- pure, Streamlit-free.

This is the scraper behind the Projected Lineups page, lifted out of it so the
projection engine can read the same XIs. **Streamlit-free is a hard
constraint**, for the reason ``projection_sources`` and ``projection_engine``
carry it: the GitHub Actions snapshot collector imports them, and that workflow
installs requirements best-effort.

**Why the engine wants this page at all.** Rotowire publishes its weekly numbers
twice -- a rankings *article* (20 clubs x 11 players, with projected points) and
this *lineups* page (20 clubs x 11 players, no points, and whose own meta
description says it is "updated as team news breaks"). The engine took "Rotowire
expects him to start" from the article, because that is where the points are.

The two disagree, and the direction is always the same: the article is written
once and the lineups page keeps moving. Measured 2026-10-21 for GW8, with the
article stamped the previous morning, 3 of 220 differed -- and one of them was
Phil Foden, whom FFP rated **20%** to start and the article still had in
Manchester City's XI. Presence in the article clips start probability *up* to
the positional floor, so the app rendered him at **86%** and 4.24 expected
points while its own Projected Lineups page had him on the bench. Every number
on both screens was plausible; they just came from different vintages of one
source.

So the article stays the points source -- it is the only Rotowire number -- and
this page becomes the start signal. See ``lineup_starters`` for the fail-open
rules, which matter more than the parsing.
"""

import logging
import re
import time
from typing import Dict, NamedTuple, Optional, Set

from bs4 import BeautifulSoup
import pandas as pd
import requests

import config
from scripts.common.name_matching import ReferenceMatcher
from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

# Plain logging, not `error_helpers.get_logger`: that reaches Streamlit.
_logger = logging.getLogger("fpl_app.rotowire_lineups")

_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

LINEUP_COLUMNS = ['Team', 'Position', 'Player', 'MatchupIndex']

#: A club's XI is eleven players. Anything else is a half-published section or
#: a parse failure, and neither is a lineup call -- see ``lineup_starters``.
XI_SIZE = 11

#: Rotowire publishes a tactical *role* on the lineups page; FPL registers a
#: position. They legitimately disagree -- a wing-back is "DMC" to Rotowire and
#: a DEF to FPL -- which is why position is a hint here and never a filter.
ROTOWIRE_TACTICAL_TO_POSITION = {
    'GK': 'G',
    'DL': 'D', 'DC': 'D', 'DR': 'D',
    'DML': 'M', 'DMC': 'M', 'DMR': 'M',
    'ML': 'M', 'MC': 'M', 'MR': 'M',
    'AML': 'M', 'AMC': 'M', 'AMR': 'M',
    'FL': 'F', 'FWL': 'F', 'FC': 'F', 'FW': 'F', 'FWR': 'F', 'FR': 'F',
}

_POS_LETTERS = ('G', 'D', 'M', 'F')

#: "J.Palhinha" -> "J. Palhinha". Rotowire abbreviates a first name with no
#: space after the dot, and canonical_normalize *deletes* the dot rather than
#: splitting on it, so the whole thing collapses to the single token
#: "jpalhinha" and every token-based tier misses. With the space restored the
#: last-word tier sees "palhinha" and resolves it inside the club.
_ABBREV_INITIAL_RE = re.compile(r'^([A-Za-z])\.(?=\S)')


def _expand_abbreviated_initial(name):
    return _ABBREV_INITIAL_RE.sub(r'\1. ', str(name or ''))


def extract_players(section, team_type, team_name, matchup_index):
    """
    Extracts valid players from the given section, excluding those listed in the Injuries section.

    Parameters:
    - section (BeautifulSoup): The section containing the team's lineup.
    - team_type (str): 'home' or 'visit' to determine the team type.
    - team_name (str): The name of the team.
    - matchup_index (int): Index of the matchup to track which game this lineup belongs to.

    Returns:
    - list: A list of tuples containing (Team, Position, Player, MatchupIndex) for valid players.
    """
    player_list = []
    injuries_section_reached = False  # Track if the Injuries section has been reached

    # Find the correct lineup section for the team
    players_section = section.find('ul', class_=f'lineup__list is-{team_type}')

    if players_section:
        for item in players_section.find_all('li', class_='lineup__player'):
            # Check if we reached the Injuries section
            if item.find_previous_sibling('li', class_='lineup__title is-middle'):
                injuries_section_reached = True

            # Skip players if we are in the Injuries section
            if not injuries_section_reached:
                try:
                    position = item.find('div', class_='lineup__pos').text.strip()
                    player_name = item.find('a').text.strip()
                    player_list.append((team_name, position, player_name, matchup_index))
                except AttributeError:
                    continue  # Skip if position or player name is missing

    return player_list


_ELEMENT_TYPE_TO_POSITION = {1: 'G', 2: 'D', 3: 'M', 4: 'F'}

#: Successes only, and short enough that it can never serve a payload older
#: than the Streamlit cache above it would have. Same rules as
#: ``ffp_feed._memoised``, and for the same reason: memoising a *failure* is how
#: a transient timeout once pinned "temporarily unavailable" for five minutes.
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


def _bootstrap_pool() -> pd.DataFrame:
    """The FPL player pool, as little of it as the matcher needs.

    Resolved here rather than taken from the caller so an XI is a *global* fact
    -- "these element ids are Rotowire's projected starters" -- independent of
    whatever frame is being blended. The alternative was the bug CLAUDE.md
    records for the omission penalty: a 15-player Classic squad holds two or
    three players per club, so a per-club gate judged against the frame can
    never be met and the whole signal silently does nothing on every per-squad
    page.
    """
    try:
        resp = requests.get(_BOOTSTRAP_URL, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as e:
        _logger.warning("Could not read the FPL player pool: %s", e)
        return pd.DataFrame()
    shorts = {int(t["id"]): str(t["short_name"]) for t in payload.get("teams", [])}
    rows = []
    for e in payload.get("elements", []):
        rows.append({
            "Player_ID": int(e["id"]),
            "Player_Code": e.get("code"),
            "Player": ("%s %s" % (e.get("first_name", ""),
                                  e.get("second_name", ""))).strip(),
            "Web_Name": e.get("web_name", ""),
            "Team": shorts.get(int(e.get("team", 0))),
            "Position": _ELEMENT_TYPE_TO_POSITION.get(int(e.get("element_type", 0))),
        })
    return pd.DataFrame(rows)


def fetch_lineup_xi(url=None, gameweek=None) -> "LineupXI":
    """Rotowire's projected starters for ``gameweek``, as FPL element ids.

    The whole pipeline: scrape, resolve against the bootstrap, gate per club.
    Never raises -- a failure is an empty ``LineupXI``, which the projection
    engine reads as "no opinion" and falls back to the rankings article for.
    """
    if url is None:
        url = getattr(config, "ROTOWIRE_LINEUPS_URL", None)
    if not url:
        return LineupXI(set(), set(), gameweek,
                        "No Rotowire lineups URL configured.")
    if gameweek is None:
        gameweek = config.CURRENT_GAMEWEEK

    def _build():
        try:
            players = scrape_lineups(url, gameweek).players
            return lineup_starters(players, _bootstrap_pool(), gameweek)
        except Exception as e:              # pragma: no cover - never break a blend
            _logger.warning("Rotowire lineups: could not resolve XIs: %s", e)
            return LineupXI(set(), set(), gameweek, str(e))

    return _memoised("lineup_xi:%s:%s" % (url, gameweek), _build)


def _bootstrap_team_shorts() -> Dict[int, str]:
    """``{team id: short_name}``, fetched directly.

    ``fixture_helpers._bootstrap_teams_df`` is the app's reader for this and is
    the better one -- it is cached. It also imports Streamlit, which this module
    must not, so the fetch is repeated here rather than the constraint broken.
    """
    try:
        resp = requests.get(_BOOTSTRAP_URL, timeout=15)
        resp.raise_for_status()
        return {int(t["id"]): str(t["short_name"])
                for t in resp.json().get("teams", [])}
    except Exception as e:
        _logger.warning("Could not read the FPL team list: %s", e)
        return {}


def _gameweek_fixture_pairs(gameweek):
    """``{(home_short, away_short)}`` for one gameweek, from the FPL fixture list.

    Returns an empty set when the list cannot be read, which callers treat as
    "do not filter" -- see :func:`_matchup_is_in_gameweek`.
    """
    try:
        short = _bootstrap_team_shorts()
        resp = requests.get(
            config.FPL_FIXTURES_BY_EVENT.format(gw=int(gameweek)), timeout=15
        )
        resp.raise_for_status()
        return {
            (short.get(f.get("team_h")), short.get(f.get("team_a")))
            for f in resp.json()
            if short.get(f.get("team_h")) and short.get(f.get("team_a"))
        }
    except Exception as e:
        _logger.warning("Could not read the GW%s fixture list: %s", gameweek, e)
        return set()


def _matchup_is_in_gameweek(home_team, away_team, fixture_pairs) -> bool:
    """Does this Rotowire matchup belong to the gameweek being displayed?

    Rotowire's lineups page lists whatever matches it has lineups for, which
    runs past the current gameweek -- observed 2026-09-10 with 11 matchups, ten
    from GW4 and one ("Brentford vs Chelsea", 18 September) from GW5. Rendered
    under a GW4 heading that is simply the wrong fixture.

    **Fails open.** An unresolvable club label, or an unreadable fixture list,
    keeps the matchup. Showing one extra match is a mild annoyance; silently
    dropping a real one is a functional loss, and club labels are exactly the
    thing that goes stale when a source changes its spelling.

    Note the asymmetry with :func:`lineup_starters`, which fails *closed* on the
    same uncertainty. Rendering an extra match costs a card; telling the
    projection engine that a player is benched costs him his projection.
    """
    if not fixture_pairs:
        return True
    home = TEAM_FULL_TO_SHORT.get(str(home_team).strip())
    away = TEAM_FULL_TO_SHORT.get(str(away_team).strip())
    if not home or not away:
        _logger.info(
            "Projected Lineups: unmapped club label in %r vs %r -- keeping the "
            "matchup rather than dropping it", home_team, away_team
        )
        return True
    return (home, away) in fixture_pairs


class LineupScrape(NamedTuple):
    """One pass over Rotowire's lineups page.

    ``players`` and ``matchups`` share a single ``MatchupIndex`` space by
    construction, which is the whole point of returning them together.
    """

    players: pd.DataFrame
    matchups: list


def scrape_lineups(url, gameweek=None) -> LineupScrape:
    """Scrape Rotowire's lineups page once: the players and the matchup list.

    **These must come from the same filtered pass.** Rotowire's page carries
    whatever matches it has lineups for, which runs past the gameweek the app is
    showing -- so both are filtered against the real fixture list, and
    ``MatchupIndex`` is assigned only to matchups that survive, keeping it
    contiguous. The renderer pairs home and away by that index, so a gap would
    leave a matchup showing one side.

    They used to be two functions doing two separate fetches, and only one of
    them filtered. That put a next-gameweek fixture in the dropdown (observed:
    "Brentford v Chelsea" under a GW4 heading) and, worse, gave the two sides
    different index spaces: the players were numbered 0..9 after filtering while
    the dropdown was numbered 0..10 before it. The stray fixture happened to
    sort last both times, so the indices coincided and nothing looked wrong --
    but Rotowire orders by kickoff, and a next-week fixture appearing anywhere
    earlier would have rendered every subsequent matchup's players under the
    wrong clubs. Every value on screen would still have been individually
    plausible.

    Parameters:
    - url (str): The URL of the Rotowire lineups page.
    - gameweek (int, optional): restrict to matchups belonging to this gameweek.
      Defaults to ``config.CURRENT_GAMEWEEK``; pass ``0`` to disable.
    """
    try:
        page = requests.get(url, timeout=30)
    except Exception as e:
        _logger.warning("Failed to fetch Rotowire lineups from %s: %s", url, e)
        return LineupScrape(pd.DataFrame(columns=LINEUP_COLUMNS), [])
    soup = BeautifulSoup(page.content, 'html.parser')

    if gameweek is None:
        gameweek = config.CURRENT_GAMEWEEK
    fixture_pairs = _gameweek_fixture_pairs(gameweek) if gameweek else set()

    # Matchups are read off the same `lineup__main` sections the players come
    # from, via find_previous, rather than from the parallel `lineup__matchup`
    # divs. They are 1:1 today, but deriving both from one iteration means they
    # cannot drift apart -- and a matchup with no lineup section could otherwise
    # reach the dropdown and render an empty card.
    lineup_sections = soup.find_all('div', class_='lineup__main')

    all_players = []
    matchups = []
    matchup_index = 0
    skipped_other_gw = 0
    for section in lineup_sections:
        try:
            home_team = section.find_previous('div', class_='lineup__mteam is-home').text.strip()
            away_team = section.find_previous('div', class_='lineup__mteam is-visit').text.strip()

            if not _matchup_is_in_gameweek(home_team, away_team, fixture_pairs):
                skipped_other_gw += 1
                continue

            # Extract players while excluding those listed in the Injuries section
            home_players = extract_players(section, 'home', home_team, matchup_index)
            away_players = extract_players(section, 'visit', away_team, matchup_index)

            all_players.extend(home_players + away_players)
            matchups.append((home_team, away_team, matchup_index))
            matchup_index += 1

        except AttributeError as e:
            _logger.warning("Error parsing lineup section (HTML structure may have changed): %s", e)

    if skipped_other_gw:
        _logger.info("Projected Lineups: skipped %d matchup(s) outside GW%s",
                     skipped_other_gw, gameweek)
    if not matchups and lineup_sections:
        _logger.warning("Rotowire: found %d lineup sections but parsed no matchups",
                        len(lineup_sections))

    return LineupScrape(pd.DataFrame(all_players, columns=LINEUP_COLUMNS), matchups)


def scrape_rotowire_lineups(url, gameweek=None):
    """The players half of :func:`scrape_lineups`. See it for the filtering rules."""
    return scrape_lineups(url, gameweek).players


def scrape_matchups(url, gameweek=None):
    """The matchup half of :func:`scrape_lineups`: ``(home, away, index)`` tuples.

    Note the ``gameweek`` parameter, which this deliberately did not have: an
    unfiltered matchup list is what put a next-gameweek fixture in the dropdown.
    """
    return scrape_lineups(url, gameweek).matchups


def resolve_lineup_name(matcher: ReferenceMatcher, name, team_code, tactical):
    """One Rotowire lineup entry -> a reference-pool index label, or None.

    Shared by the Projected Lineups cards (which want the player's stats) and
    the projection engine (which wants his element id), so the two cannot
    resolve the same name to different players.

    **Every lookup is scoped to the club.** The hand-rolled ladder this
    replaced was team- *and* position-agnostic at every stage, over a dict that
    additionally keyed players by bare surname and by ``web_name``. Both are
    ambiguous league-wide -- measured live, 24 surnames and 17 web_names
    collided, 51 and 36 players -- and a plain dict silently keeps whichever the
    bootstrap listed last. "Palmer" was one of them: Cole Palmer (CHE, elite
    MID) and Alex Palmer (IPS, GK), the exact pairing that has caused this bug
    elsewhere in the app.

    Position is a **hint, not a filter**: Rotowire's role and FPL's registered
    position disagree often enough to matter -- measured on one gameweek, 5 of
    66 starters (Cunha listed as a forward and registered as a midfielder,
    wing-backs listed in midfield). Retrying across all four keeps them, and
    stays safe because a name that resolves to two different players still
    resolves to none.

    **The retry is resolved by match tier, not by counting answers.** Two
    positions can both produce a hit for one query, and they are not equally
    good: live at GW8 "B. Thomas-Asante" matched Brandon Thomas-Asante (COV,
    FWD) on a complete surname at tier 3 and his team-mate *Bobby Thomas* on a
    looser tier 4, so treating that as ambiguous threw away a match an exact
    token had already earned and left Coventry's XI one short. Preferring the
    strongest tier is the rule ``ReferenceMatcher.match_with_tier`` exists for
    and is used for in ``_claim_reference_rows``; a genuine tie at the same tier
    still resolves to nothing.

    Residual risk, stated rather than papered over: two players at one club
    sharing a surname that FPL registers in different positions, whose names
    resolve at the *same* tier, are separated only by the position scoping -- so
    a Rotowire role that disagrees with FPL's registration could pick the other
    one. That needs all of those things at once, and the alternative -- ignoring
    position -- makes every such pair unresolvable and loses the wing-backs too.
    """
    if matcher is None or not team_code:
        return None
    expanded = _expand_abbreviated_initial(name)
    hit = matcher.match(expanded, team_code, tactical)
    if hit is not None:
        return hit

    by_tier: Dict[int, set] = {}
    for pos in _POS_LETTERS:
        found = matcher.match_with_tier(expanded, team_code, pos)
        if found is None:
            continue
        idx, tier = found
        if idx is None or tier is None:
            continue
        by_tier.setdefault(int(tier), set()).add(idx)
    if not by_tier:
        return None
    best = min(by_tier)
    winners = by_tier[best]
    return next(iter(winners)) if len(winners) == 1 else None


class LineupXI(NamedTuple):
    """Rotowire's projected starters, resolved to FPL element ids.

    Two fields, because the two directions of this signal need different
    evidence -- see :func:`lineup_starters`.

    ``player_ids`` is every starter that resolved, at any club. Saying "this
    man starts" needs only his own name to have resolved.

    ``clubs`` is the clubs whose XI resolved *completely*. Saying "this man does
    **not** start" is a claim about everyone the XI leaves out, so it needs the
    whole XI: one unresolved starter is indistinguishable from a benching.
    """

    player_ids: Set[int]
    clubs: Set[str]
    gameweek: Optional[int] = None
    note: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.player_ids)


def lineup_starters(players: pd.DataFrame, pool: pd.DataFrame,
                    gameweek: Optional[int] = None,
                    source: str = "Rotowire") -> LineupXI:
    """Resolve a scraped lineups frame against the FPL pool, club by club.

    ``source`` names the publication in the log lines only. There is one gate
    for every XI source by design: a second copy of this asymmetry is the
    twin-drift failure this codebase keeps paying for, and a benching gate is
    the worst place for it -- a drift there would not degrade a projection but
    invert it.

    **The two directions of the signal are gated differently, and that
    asymmetry is the whole safety design.**

    Promoting a player -- "Rotowire expects him to start" -- needs only his own
    name to have resolved, and can only raise his start probability towards the
    positional floor. A name this app fails to match simply does not get
    promoted, which is the behaviour that shipped for a year.

    Benching one -- "Rotowire does **not** expect him to start" -- is a claim
    about every player the XI leaves out, and it *lowers* him to the omitted
    prior. A starter whose name could not be resolved is indistinguishable from
    one who was left out, so a matching failure there would not degrade a number
    but invert it. Hence the club-level gate: only a club whose eleven all
    resolved can bench anybody, and a club that falls short still promotes the
    starters it did resolve.

    Live at GW8 that is 18 clubs resolving fully; Brighton falls short on two
    names FPL and Rotowire genuinely spell differently (``Ferdi Kadıoğlu``
    against "F. Kadioglu" -- the dotless Turkish ı does not fold to "i" -- and
    "Chema" for José María Andrés Baixauli, a nickname that appears nowhere in
    the FPL name). Nine Brighton starters are still promoted; none is benched.

    Returns clubs as FPL short codes, so the engine's ``Team`` column matches
    without further mapping.
    """
    empty = LineupXI(set(), set(), gameweek)
    if players is None or players.empty or pool is None or pool.empty:
        return empty._replace(note="No %s lineups were available." % source)
    if "Player_ID" not in pool.columns:
        return empty._replace(note="The player pool carries no element ids.")

    matcher_pool = pool.copy()
    matcher = ReferenceMatcher(
        matcher_pool,
        name_col="Player",
        web_name_col="Web_Name" if "Web_Name" in matcher_pool.columns else None,
        team_col="Team" if "Team" in matcher_pool.columns else None,
        position_col="Position" if "Position" in matcher_pool.columns else None,
    )
    ids_by_label = matcher_pool["Player_ID"]

    resolved: Set[int] = set()
    clubs: Set[str] = set()
    partial = []
    for club_label, rows in players.groupby(players["Team"].astype(str), sort=False):
        code = TEAM_FULL_TO_SHORT.get(club_label.strip())
        if not code:
            # Fail closed, unlike the matchup filter: an unmapped club here
            # would silently bench every player at it.
            _logger.info("%s lineups: unmapped club label %r", source, club_label)
            continue
        found = set()
        for _, row in rows.iterrows():
            tactical = ROTOWIRE_TACTICAL_TO_POSITION.get(
                str(row.get("Position") or "").strip().upper())
            hit = resolve_lineup_name(matcher, row.get("Player"), code, tactical)
            if hit is not None:
                try:
                    found.add(int(ids_by_label.loc[hit]))
                except (KeyError, TypeError, ValueError):   # pragma: no cover
                    continue
        # Every resolved starter is promotable, whatever his club's total.
        resolved |= found
        if len(found) == XI_SIZE and len(rows) == XI_SIZE:
            clubs.add(code)
        else:
            partial.append("%s %d/%d" % (code, len(found), len(rows)))

    if partial:
        _logger.info(
            "%s lineups: %d club(s) not fully resolved, so nobody there "
            "is benched on this signal: %s", source, len(partial), ", ".join(partial))

    note = "%d starter%s, %d club%s complete" % (
        len(resolved), "" if len(resolved) == 1 else "s",
        len(clubs), "" if len(clubs) == 1 else "s")
    if partial:
        note += " (%d partial: %s)" % (len(partial), ", ".join(partial))
    return LineupXI(resolved, clubs, gameweek, note)
