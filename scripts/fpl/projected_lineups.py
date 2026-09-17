from bs4 import BeautifulSoup
from collections import defaultdict
from typing import NamedTuple
import re
import config
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scripts.common.error_helpers import get_logger
from scripts.fpl.injuries import get_fpl_availability_df
from scripts.common.utils import get_classic_bootstrap_static
from scripts.common.text_helpers import (
    TEAM_FULL_TO_SHORT,
    compact_html,
    format_last_updated,
    to_display_name,
)
from scripts.common.fixture_helpers import _bootstrap_teams_df
from scripts.common.name_matching import ReferenceMatcher
from scripts.common.scraping import get_pl_predicted_lineups

_logger = get_logger("fpl_app.projected_lineups")

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

def _gameweek_fixture_pairs(gameweek):
    """``{(home_short, away_short)}`` for one gameweek, from the FPL fixture list.

    Returns an empty set when the list cannot be read, which callers treat as
    "do not filter" -- see :func:`_matchup_is_in_gameweek`.
    """
    try:
        teams = _bootstrap_teams_df()
        short = dict(zip(teams["id"], teams["short_name"]))
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


LINEUP_COLUMNS = ['Team', 'Position', 'Player', 'MatchupIndex']


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

#: Coarse fallback start probabilities, used only when the projection
#: engine has no view of a player.
_STATUS_BUCKET_START_PCT = {'Out': 0, 'Doubtful': 25,
                            'Questionable': 50, 'Likely': 75}
_ELEMENT_TYPE_TO_POSITION = {1: 'G', 2: 'D', 3: 'M', 4: 'F'}

#: "J.Palhinha" -> "J. Palhinha". Rotowire abbreviates a first name with no
#: space after the dot, and canonical_normalize *deletes* the dot rather than
#: splitting on it, so the whole thing collapses to the single token
#: "jpalhinha" and every token-based tier misses. With the space restored the
#: last-word tier sees "palhinha" and resolves it inside the club.
_ABBREV_INITIAL_RE = re.compile(r'^([A-Za-z])\.(?=\S)')


def _expand_abbreviated_initial(name):
    return _ABBREV_INITIAL_RE.sub(r'\1. ', str(name or ''))


class PlayerIndex(NamedTuple):
    """FPL player stats, resolvable from a Rotowire lineup name.

    ``stats`` is keyed on the reference pool's index label, so a resolved match
    is a row lookup rather than a second name-keyed dict.
    """

    matcher: object = None
    pool: object = None
    stats: dict = {}

    def lookup(self, player_name, team, position=None) -> dict:
        """Stats for one Rotowire lineup entry, or ``{}`` when unresolved.

        ``team`` is Rotowire's long club label and ``position`` its tactical
        code; both are resolved here so no caller has to.

        **Every lookup is scoped to the club.** The hand-rolled matcher this
        replaces was a six-stage ladder that was team- *and* position-agnostic
        at every stage, over a dict that additionally keyed players by bare
        surname and by ``web_name``. Both are ambiguous league-wide -- measured
        live, 24 surnames and 17 web_names collided, 51 and 36 players -- and a
        plain dict silently keeps whichever the bootstrap listed last. "Palmer"
        was one of them: Cole Palmer (CHE, elite MID) and Alex Palmer (IPS, GK),
        the exact pairing that already caused this bug once elsewhere in the
        app. The card would show one player's form, injury and news under the
        other's name, and every value on it would look perfectly ordinary.
        """
        if self.matcher is None or not len(self.stats):
            return {}

        team_code = TEAM_FULL_TO_SHORT.get(str(team or '').strip())
        if not team_code:
            # Without a club the only safe tiers are exact whole-name ones, and
            # a wrong card is worse than a bare one. Fail closed.
            _logger.info("Projected Lineups: unmapped club label %r", team)
            return {}

        name = _expand_abbreviated_initial(player_name)
        tactical = ROTOWIRE_TACTICAL_TO_POSITION.get(str(position or '').strip().upper())

        hit = self.matcher.match(name, team_code, tactical)
        if hit is None:
            # Rotowire's role and FPL's registered position disagree often
            # enough to matter -- measured on one gameweek, 5 of 66 starters
            # (Cunha listed as a forward and registered as a midfielder,
            # wing-backs listed in midfield). Retrying across all four keeps
            # them, and stays safe because a name that resolves to two
            # different players still resolves to none.
            hits = {self.matcher.match(name, team_code, p) for p in _POS_LETTERS}
            hits.discard(None)
            if len(hits) != 1:
                return {}
            hit = hits.pop()

        # Residual risk, stated rather than papered over: two players at one
        # club sharing a surname that FPL registers in different positions are
        # separated by the position scoping, so a Rotowire role that disagrees
        # with FPL's registration could pick the other one. That needs all four
        # of those things at once, and the alternative -- ignoring position --
        # makes every such pair unresolvable and loses the wing-backs too.

        return self.stats.get(hit, {})


EMPTY_PLAYER_INDEX = PlayerIndex()


def _attach_engine_start_pct(pool: pd.DataFrame, stats: dict) -> None:
    """Add the projection engine's ``Start_Pct`` to each player's stats, in place.

    This page used to compute its own start likelihood from a handful of status
    buckets and ``starts / 22`` -- a hard-coded season length. At GW5 that put
    every healthy player on the ``max(80, ...)`` floor, and it drifted upward
    through the season for no reason but the constant. Goalkeepers showed it
    worst: all 20 rendered at exactly 80% against the engine's 95%, so the most
    nailed-on position in the game fell in the "likely" band rather than "very
    likely". Page and engine agreed on 25 of 206 players.

    The engine already knows the answer -- FFP's start percentage, the Rotowire
    presence floors, FPL's chance of playing, the omission penalty -- and it is
    the number every other surface in the app renders. There is no reason for
    this page to hold a second opinion.

    Failure is silent by design: the heuristic remains as the fallback in
    :func:`plot_soccer_field`, so a missing projection feed costs colour
    accuracy rather than the page.
    """
    try:
        from scripts.common.analytics import (blend_projections_onto,
                                              merge_season_projections)
        from scripts.common.scraping import (get_ffp_feed,
                                             get_rotowire_player_projections)

        frame = pool.copy()
        frame['chance_of_playing_next_round'] = [
            stats.get(i, {}).get('chance_of_playing') for i in range(len(frame))]
        frame['status'] = [stats.get(i, {}).get('status', 'a') for i in range(len(frame))]

        rotowire_df = pd.DataFrame()
        if config.ROTOWIRE_URL:
            rotowire_df = get_rotowire_player_projections(config.ROTOWIRE_URL)
        if rotowire_df is not None and not rotowire_df.empty:
            frame = merge_season_projections(frame, rotowire_df, output_col='Points')

        feed = get_ffp_feed()
        blended = blend_projections_onto(frame, feed.df if feed.ok else None,
                                         expected_gw=config.CURRENT_GAMEWEEK)
        start_pct = pd.to_numeric(blended.get('Start_Pct'), errors='coerce')
        if start_pct is None:
            return
        for i, value in enumerate(start_pct):
            if pd.notna(value) and i in stats:
                stats[i]['start_pct_engine'] = float(value) * 100.0
    except Exception as e:
        _logger.warning("Could not attach engine start probabilities: %s", e)


def build_player_index() -> PlayerIndex:
    """Build the FPL reference pool and its matcher for the lineups page.

    Returns :data:`EMPTY_PLAYER_INDEX` on any failure -- the stats are an
    enhancement to the lineup cards, not a prerequisite for drawing them.
    """
    try:
        bootstrap = get_classic_bootstrap_static()
        if not bootstrap:
            return EMPTY_PLAYER_INDEX

        elements = bootstrap.get('elements', [])
        teams = {t['id']: t['short_name'] for t in bootstrap.get('teams', [])}

        # Availability is keyed on the element id, not the name: this frame and
        # the bootstrap are the same source, so there is nothing to match.
        avail_lookup = {}
        try:
            avail_df = get_fpl_availability_df()
            if not avail_df.empty:
                for row in avail_df.itertuples(index=False):
                    avail_lookup[getattr(row, 'Player_ID', None)] = {
                        'play_pct': getattr(row, 'PlayPct', 100),
                        'status_bucket': getattr(row, 'StatusBucket', 'Available'),
                        'news': getattr(row, 'News', ''),
                    }
        except Exception as e:
            _logger.warning("Availability data unavailable for lineups: %s", e)

        rows, stats = [], {}
        for idx, elem in enumerate(elements):
            web_name = elem.get('web_name', '')
            if not web_name:
                continue
            first_name = elem.get('first_name', '')
            second_name = elem.get('second_name', '')

            rows.append({
                'Player_ID': elem.get('id'),
                'Player': f"{first_name} {second_name}".strip(),
                'Web_Name': web_name,
                'Display_Name': to_display_name(first_name, second_name, web_name),
                'Team': teams.get(elem.get('team'), ''),
                'Position': _ELEMENT_TYPE_TO_POSITION.get(elem.get('element_type'), ''),
            })

            pdata = {
                'form': float(elem.get('form', 0) or 0),
                'points_per_game': float(elem.get('points_per_game', 0) or 0),
                'total_points': elem.get('total_points', 0),
                'minutes': elem.get('minutes', 0) or 0,
                'starts': elem.get('starts', 0) or 0,
                'goals_scored': elem.get('goals_scored', 0),
                'assists': elem.get('assists', 0),
                'clean_sheets': elem.get('clean_sheets', 0),
                'chance_of_playing': elem.get('chance_of_playing_this_round'),
                'status': elem.get('status', 'a'),
                'news': elem.get('news', ''),
                'team': teams.get(elem.get('team'), ''),
                'play_pct': 100,
                'status_bucket': 'Available',
            }
            pdata.update(avail_lookup.get(elem.get('id'), {}))
            stats[len(rows) - 1] = pdata

        if not rows:
            return EMPTY_PLAYER_INDEX

        pool = pd.DataFrame(rows)
        _attach_engine_start_pct(pool, stats)
        return PlayerIndex(matcher=ReferenceMatcher(pool), pool=pool, stats=stats)

    except Exception as e:
        _logger.warning("Failed to build player index for lineup enhancement: %s", e)
        return EMPTY_PLAYER_INDEX



def start_likelihood_pct(pdata: dict) -> float:
    """How likely is this player to start, as 0-100?

    **The projection engine's ``Start_Pct`` is the answer whenever it is
    available.** It already combines FFP's start percentage, the Rotowire
    presence floors, FPL's chance of playing and the omission penalty, and it is
    the number every other surface in the app renders; a second opinion here
    only creates two numbers for one question.

    The buckets below are the fallback for a page load where the projection
    feeds are down. They are deliberately coarse, and one detail matters: the
    historical rate divides by the gameweeks actually played. It used to divide
    by a hard-coded 22, which pinned every healthy player to the ``max(80, ...)``
    floor early in the season and let the figure drift upward as the constant
    was approached -- movement with no evidence behind it.

    That floor is why goalkeepers were the visible symptom: measured at GW5, all
    20 starting keepers rendered at exactly 80% against the engine's 95%, so the
    most nailed-on position in the game sat in the "likely" band rather than
    "very likely".
    """
    engine_start = pdata.get('start_pct_engine')
    if engine_start is not None:
        return float(engine_start)

    chance_of_playing = pdata.get('chance_of_playing')
    if chance_of_playing is not None:
        return float(chance_of_playing)

    bucket = pdata.get('status_bucket', 'Available')
    if bucket in _STATUS_BUCKET_START_PCT:
        return float(_STATUS_BUCKET_START_PCT[bucket])

    starts = pdata.get('starts', 0) or 0
    if starts > 0:
        played = max(1, int(config.CURRENT_GAMEWEEK) - 1)
        return max(80.0, min(100.0, (starts / played) * 100.0))
    return 80.0   # in Rotowire's XI but no starts yet -- a new signing


def get_availability_color(status_bucket, play_pct=None):
    """Returns color based on player availability status."""
    if status_bucket == 'Out':
        return '#e74c3c'  # Red
    elif status_bucket == 'Doubtful':
        return '#e67e22'  # Orange
    elif status_bucket == 'Questionable':
        return '#f1c40f'  # Yellow
    elif status_bucket == 'Likely':
        return '#2ecc71'  # Light green
    else:  # Available
        return '#27ae60'  # Green


def get_form_color(form):
    """Returns color based on player form rating."""
    if form >= 6:
        return '#27ae60'  # Green - excellent
    elif form >= 4:
        return '#2ecc71'  # Light green - good
    elif form >= 2:
        return '#f1c40f'  # Yellow - average
    else:
        return '#e74c3c'  # Red - poor

def plot_soccer_field(player_df, team_name, player_index=None):
    """
    Plots players on a soccer field based on their positions for a specific team.
    Enhanced with player form and availability indicators.

    Parameters:
    - player_df (pd.DataFrame): A DataFrame containing 'Position' and 'Player' columns.
    - team_name (str): The name of the team, displayed as the title above the field.
    - player_index (PlayerIndex): Optional FPL stats for form/availability enhancement.
    """
    if player_index is None:
        player_index = EMPTY_PLAYER_INDEX

    # Map Rotowire team names to TEAM_COLORS keys
    team_name_map = {
        'Manchester United': 'Man Utd',
        'Manchester City': 'Man City',
        'Tottenham Hotspur': 'Spurs',
        'Nottingham Forest': "Nott'm Forest",
        'Wolverhampton': 'Wolves',
        'West Ham United': 'West Ham',
        'Leicester City': 'Leicester',
        'Ipswich Town': 'Ipswich',
        'AFC Bournemouth': 'Bournemouth',
        'Brighton and Hove Albion': 'Brighton',
        'Newcastle United': 'Newcastle',
    }
    mapped_team_name = team_name_map.get(team_name, team_name)

    # Define positions with (x, y) coordinates
    position_mapping = {
        'GK': (5, 0.3),  # Goalkeeper
        'DL': (1.5, 1.5), 'DC': (5, 1.2), 'DR': (8.5, 1.5),  # Defenders
        'DML': (2, 2.1), 'DMC': (5, 2.1), 'DMR': (8, 2.1),  # Defensive Midfielders
        'ML': (1.5, 2.9), 'MC': (5, 2.7), 'MR': (8.5, 2.9),  # Midfielders
        'AML': (1.4, 3.5), 'AMC': (5, 3.5), 'AMR': (8.6, 3.5),  # Attacking Midfielders
        'FL': (2, 4.4), 'FWL': (2, 4.4), 'FC': (5, 4.6), 'FW': (5, 4.4), 'FWR': (8, 4.4), 'FR': (8, 4.4)  # Forwards
    }

    # Get the team's primary and secondary colors (fallback to visible colors, not background color)
    colors = config.TEAM_COLORS.get(mapped_team_name, {"primary": "#3498db", "secondary": "#FFFFFF"})
    primary_color = colors["primary"]
    secondary_color = colors["secondary"]

    # Group players by position
    grouped_players = defaultdict(list)
    for _, row in player_df.iterrows():
        grouped_players[row['Position']].append(row['Player'])

    # Create a Plotly figure
    fig = go.Figure()

    # Draw simple field lines (original style)
    # Field boundary
    fig.add_shape(type="rect", x0=0, y0=-0.3, x1=10, y1=5.2,
                  line=dict(color="white", width=2))

    # Center line
    fig.add_shape(type="line", x0=0, y0=2.5, x1=10, y1=2.5,
                  line=dict(color="white", dash="dash"))

    # Add players to the field
    for position, players in grouped_players.items():
        for i, player_name in enumerate(players):
            x, y = position_mapping.get(position, (5, 2.75))  # Default to center if not found

            # Adjust for multiple players in the same central position
            if len(players) == 2 and position in ['DC', 'MC', 'FC', 'FW', 'DMC', 'AMC']:
                x += -1.5 if i == 0 else 1.5
            elif len(players) == 3 and position in ['DC', 'MC', 'FC', 'DMC', 'AMC']:
                if i == 0:
                    x -= 2.0
                elif i == 2:
                    x += 2.0

            # Get player data for enhanced display. team_name is this frame's
            # club and `position` the tactical slot being drawn, so the lookup
            # is scoped to the club it belongs to.
            pdata = player_index.lookup(player_name, team_name, position)
            form = pdata.get('form', 0)
            status_bucket = pdata.get('status_bucket', 'Available')
            play_pct = pdata.get('play_pct', 100)
            chance_of_playing = pdata.get('chance_of_playing')  # Can be None
            news = pdata.get('news', '')
            total_points = pdata.get('total_points', 0)
            goals = pdata.get('goals_scored', 0)
            assists = pdata.get('assists', 0)
            starts = pdata.get('starts', 0)
            minutes = pdata.get('minutes', 0)

            start_likelihood = start_likelihood_pct(pdata)

            # Determine marker appearance based on start likelihood
            # Use opacity to show likelihood (more opaque = more likely to start)
            marker_opacity = max(0.4, start_likelihood / 100)

            # Border color based on likelihood
            if start_likelihood >= 90:
                border_color = '#27ae60'  # Green - very likely
            elif start_likelihood >= 70:
                border_color = '#2ecc71'  # Light green - likely
            elif start_likelihood >= 50:
                border_color = '#f1c40f'  # Yellow - questionable
            elif start_likelihood >= 25:
                border_color = '#e67e22'  # Orange - doubtful
            else:
                border_color = '#e74c3c'  # Red - unlikely

            # Build hover text with player details
            hover_lines = [
                f"<b>{player_name}</b>",
                f"Position: {position}",
                f"Start Likelihood: {start_likelihood:.0f}%",
                f"Form: {form:.1f}" if form else "Form: N/A",
                f"Total Points: {total_points}",
            ]
            if goals or assists:
                hover_lines.append(f"G: {goals} | A: {assists}")
            if starts > 0:
                hover_lines.append(f"Starts: {starts} | Mins: {minutes}")
            if status_bucket != 'Available':
                hover_lines.append(f"Status: {status_bucket}")
            if news:
                news_short = news[:50] + "..." if len(news) > 50 else news
                hover_lines.append(f"News: {news_short}")

            hover_text = "<br>".join(hover_lines)

            # Add player marker with start likelihood indicator
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(
                    size=22,
                    color=primary_color,
                    opacity=marker_opacity,
                    line=dict(color=border_color, width=3)
                ),
                text=player_name,
                textposition="top center",
                textfont=dict(color="#FFFFFF", size=14),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False
            ))

    fig.update_layout(
        width=500, height=600,
        xaxis=dict(range=[0, 10], visible=False),
        yaxis=dict(range=[-0.5, 5.5], visible=False),
        plot_bgcolor="#228B22",
        paper_bgcolor="#1a1a2e",
        showlegend=False,
        margin=dict(l=5, r=5, t=5, b=5),
    )

    return fig

def render_player_cards_html(player_df, player_index):
    """Renders all player cards as a single HTML block."""
    cards = []

    for _, row in player_df.iterrows():
        player_name = row['Player']
        position = row['Position']
        pdata = player_index.lookup(player_name, row.get('Team'), position)

        form = pdata.get('form', 0)
        status_bucket = pdata.get('status_bucket', 'Available')
        total_points = pdata.get('total_points', 0)
        goals = pdata.get('goals_scored', 0)
        assists = pdata.get('assists', 0)
        news = pdata.get('news', '')

        # Colors
        status_colors = {'Out': '#e74c3c', 'Doubtful': '#e67e22', 'Questionable': '#f1c40f', 'Likely': '#2ecc71', 'Available': '#27ae60'}
        status_color = status_colors.get(status_bucket, '#27ae60')
        form_color = get_form_color(form) if form > 0 else '#888'

        # Status badge
        status_badge = ''
        if status_bucket != 'Available':
            status_badge = f'<span style="background:{status_color};color:#fff;padding:2px 6px;border-radius:3px;font-size:0.75em;margin-left:8px;">{status_bucket}</span>'

        # News line
        news_line = ''
        if news:
            news_short = (news[:50] + '...') if len(news) > 50 else news
            news_line = f'<div style="color:#aaa;font-size:0.8em;margin-top:2px;font-style:italic;">{news_short}</div>'

        # Stats display - show N/A if no data found
        if pdata:
            stats_html = f'<span style="color:{form_color};font-weight:600;">Form: {form:.1f}</span><span style="color:#2ecc71;margin-left:15px;">Pts: {total_points}</span><span style="color:#fff;margin-left:15px;">G:{goals} A:{assists}</span>'
        else:
            stats_html = '<span style="color:#999;">Stats unavailable</span>'

        # Build card with solid background for better readability
        card = f'<div style="display:flex;align-items:center;padding:12px 14px;margin-bottom:8px;background:#2c3e50;border-radius:8px;border-left:4px solid {form_color};box-shadow:0 2px 4px rgba(0,0,0,0.2);"><div style="min-width:40px;text-align:center;background:#34495e;padding:6px 10px;border-radius:4px;margin-right:12px;"><span style="color:#fff;font-weight:bold;">{position}</span></div><div style="flex:1;"><div style="color:#fff;font-weight:bold;font-size:1.05em;">{player_name}{status_badge}</div><div style="margin-top:6px;font-size:0.9em;">{stats_html}</div>{news_line}</div></div>'
        cards.append(card)

    return ''.join(cards)


_POS_GROUP = {
    'GK': 'GK',
    'DL': 'DEF', 'DC': 'DEF', 'DR': 'DEF', 'DML': 'DEF', 'DMR': 'DEF',
    'DMC': 'MID', 'ML': 'MID', 'MC': 'MID', 'MR': 'MID',
    'AML': 'MID', 'AMC': 'MID', 'AMR': 'MID',
    'FL': 'FWD', 'FC': 'FWD', 'FR': 'FWD', 'FW': 'FWD', 'FWL': 'FWD', 'FWR': 'FWD',
}

_POS_ORDER = {'GK': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}

_GROUP_COLOR = {'GK': '#f1c40f', 'DEF': '#3498db', 'MID': '#2ecc71', 'FWD': '#e74c3c'}


def _build_lineup_card_html(home_team, away_team, home_players, away_players):
    """Build a compact HTML card showing both teams' lineups side-by-side, one player per line."""

    def _render_side(players_df):
        if players_df.empty:
            return '<div style="color:#999;font-size:0.85em;">Lineup not available</div>'
        # Add group column and sort by position order then original row order
        rows = []
        for _, row in players_df.iterrows():
            grp = _POS_GROUP.get(row['Position'], 'MID')
            rows.append({'Player': row['Player'], 'Position': row['Position'], 'Group': grp, 'Order': _POS_ORDER.get(grp, 9)})
        rows.sort(key=lambda r: r['Order'])

        lines = []
        current_group = None
        for r in rows:
            grp = r['Group']
            pos = r['Position']
            color = _GROUP_COLOR.get(grp, '#ccc')
            # Group separator
            if grp != current_group:
                if current_group is not None:
                    lines.append('<div style="height:4px;"></div>')
                current_group = grp
            group_badge = f'<span style="display:inline-block;background:{color};color:#fff;font-size:0.68em;font-weight:700;padding:1px 4px;border-radius:3px;min-width:28px;text-align:center;">{grp}</span>'
            pos_badge = f'<span style="display:inline-block;color:{color};font-size:0.78em;font-weight:600;min-width:26px;text-align:center;">{pos}</span>'
            lines.append(
                f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px;">'
                f'{group_badge}{pos_badge}'
                f'<span style="color:#ddd;font-size:0.88em;">{r["Player"]}</span>'
                f'</div>'
            )
        return "".join(lines)

    home_html = _render_side(home_players)
    away_html = _render_side(away_players)

    return f"""
    <div style="background:#1e2a3a;border-radius:10px;padding:14px 16px;margin-bottom:12px;border:1px solid #2c3e50;">
        <div style="text-align:center;font-weight:700;font-size:1.05em;color:#fff;margin-bottom:10px;">
            {home_team} <span style="color:#888;font-size:0.9em;">vs</span> {away_team}
        </div>
        <div style="display:flex;gap:16px;">
            <div style="flex:1;border-right:1px solid #2c3e50;padding-right:12px;">
                <div style="color:#aaa;font-size:0.75em;font-weight:600;margin-bottom:5px;text-transform:uppercase;">{home_team}</div>
                {home_html}
            </div>
            <div style="flex:1;padding-left:4px;">
                <div style="color:#aaa;font-size:0.75em;font-weight:600;margin-bottom:5px;text-transform:uppercase;">{away_team}</div>
                {away_html}
            </div>
        </div>
    </div>
    """


def _render_pl_section(home_team, away_team):
    """The PL's own predicted XI graphics and team news for one fixture.

    A second opinion beside Rotowire's XI, and the freshest team news available
    -- the PL writes it after the Friday press conferences, later than
    Rotowire's weekly article.

    The XIs are PNG graphics rather than text, so they are rendered as images
    and nothing here feeds a projection. Everything fails open: the PL section
    simply does not appear if the feed is unavailable or is for another week.
    """
    try:
        pl = get_pl_predicted_lineups(config.CURRENT_GAMEWEEK)
    except Exception as exc:                # never take the page down
        _logger.warning("PL predicted lineups unavailable: %s", exc)
        return

    if not pl.ok:
        if pl.note:
            st.caption("Premier League predicted line-ups: %s" % pl.note)
        return

    # The matchweek gate. The article proves its own week twice -- the title
    # states it and the fixtures vote on it -- and pl.gameweek is None when
    # those disagree. Showing last week's XI under this week's heading is the
    # whole failure this prevents.
    if pl.gameweek != config.CURRENT_GAMEWEEK:
        st.caption(
            "Premier League has not published Matchweek %s predicted line-ups yet."
            % config.CURRENT_GAMEWEEK
        )
        return

    home_code = TEAM_FULL_TO_SHORT.get(home_team)
    away_code = TEAM_FULL_TO_SHORT.get(away_team)
    if not home_code or not away_code:
        return
    if (home_code, away_code) not in pl.fixtures:
        return                            # PL does not carry this fixture

    st.markdown("---")
    header = "##### Premier League — official predicted XI"
    if pl.url:
        header += "  ·  [article](%s)" % pl.url
    st.markdown(header)
    if pl.updated:
        st.caption("Updated %s" % format_last_updated(pl.updated))

    graphic_cols = st.columns(2)
    for column, team, code in ((graphic_cols[0], home_team, home_code),
                               (graphic_cols[1], away_team, away_code)):
        image = pl.graphics.get(code)
        with column:
            if image:
                # use_column_width, not use_container_width: st.image only
                # gained the latter in Streamlit 1.40 and this app pins 1.38.
                st.image(image, use_column_width=True, caption=team)
            else:
                st.caption("%s — no graphic published" % team)

    news = [(team, pl.club_news.get(code))
            for team, code in ((home_team, home_code), (away_team, away_code))]
    if any(text for _, text in news):
        with st.expander("Premier League team news", expanded=False):
            for team, text in news:
                if text:
                    st.markdown("**%s** — %s" % (team, text))


def show_projected_lineups():
    st.title(f"Projected Lineups — GW {config.CURRENT_GAMEWEEK}")
    st.write("View projected starting lineups with player form and availability status.")

    # One fetch for both, so the matchup list and the player frame cannot
    # disagree about which fixture an index refers to.
    lineups_df, matchups = scrape_lineups(config.ROTOWIRE_LINEUPS_URL)

    if not matchups:
        st.warning("No matchups available. Rotowire may not have published lineups yet.")
        return

    # -- Overview cards: all matchups at a glance --
    if not lineups_df.empty:
        with st.expander("Lineup Overview (all matches)", expanded=True):
            cols = st.columns(2)
            for i, (home_team, away_team, idx) in enumerate(matchups):
                home_df = lineups_df[(lineups_df['Team'] == home_team) & (lineups_df['MatchupIndex'] == idx)]
                away_df = lineups_df[(lineups_df['Team'] == away_team) & (lineups_df['MatchupIndex'] == idx)]
                card_html = _build_lineup_card_html(home_team, away_team, home_df, away_df)
                cols[i % 2].markdown(compact_html(card_html), unsafe_allow_html=True)

    # -- Drill-down: full soccer field + squad detail cards --
    st.markdown("---")
    st.markdown("##### Match Drill-Down")
    # Create a drop-down to choose the matchup to view
    selected_matchup = st.selectbox(
        "Select a Matchup",
        matchups,
        format_func=lambda x: f"{x[0]} vs {x[1]}"
    )

    if selected_matchup:
        home_team, away_team, matchup_index = selected_matchup

        # Filter by BOTH team name AND matchup index to fix duplicate team bug
        home_team_df = lineups_df[
            (lineups_df['Team'] == home_team) &
            (lineups_df['MatchupIndex'] == matchup_index)
        ]
        away_team_df = lineups_df[
            (lineups_df['Team'] == away_team) &
            (lineups_df['MatchupIndex'] == matchup_index)
        ]

        # Fetch player data for enhancements
        with st.spinner("Loading player data..."):
            player_index = build_player_index()

        # Add legend for start likelihood colors (shown as border color on field)
        st.markdown("""
        <div style="display:flex;gap:15px;margin-bottom:15px;flex-wrap:wrap;">
            <span style="font-size:0.85em;color:#888;">Start Likelihood:</span>
            <span style="font-size:0.85em;"><span style="color:#27ae60;">●</span> 90%+</span>
            <span style="font-size:0.85em;"><span style="color:#2ecc71;">●</span> 70-89%</span>
            <span style="font-size:0.85em;"><span style="color:#f1c40f;">●</span> 50-69%</span>
            <span style="font-size:0.85em;"><span style="color:#e67e22;">●</span> 25-49%</span>
            <span style="font-size:0.85em;"><span style="color:#e74c3c;">●</span> &lt;25%</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{home_team}")
            home_fig = plot_soccer_field(home_team_df, home_team, player_index)
            st.plotly_chart(home_fig, use_container_width=True, key=f"home_{matchup_index}")

            # Enhanced player list
            st.markdown("##### Squad Details")
            if not home_team_df.empty:
                cards_html = render_player_cards_html(home_team_df, player_index)
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.info("No lineup data available for this team.")

        with col2:
            st.subheader(f"{away_team}")
            away_fig = plot_soccer_field(away_team_df, away_team, player_index)
            st.plotly_chart(away_fig, use_container_width=True, key=f"away_{matchup_index}")

            # Enhanced player list
            st.markdown("##### Squad Details")
            if not away_team_df.empty:
                cards_html = render_player_cards_html(away_team_df, player_index)
                st.markdown(cards_html, unsafe_allow_html=True)
            else:
                st.info("No lineup data available for this team.")

        _render_pl_section(home_team, away_team)


