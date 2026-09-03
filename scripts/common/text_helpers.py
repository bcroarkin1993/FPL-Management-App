"""
Text & String Normalization, Constants, and Position/Team Mapping.

Shared text-processing utilities, team name mappings, and position converters
used across the FPL Management App.
"""

import logging
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

from datetime import datetime
from zoneinfo import ZoneInfo

_logger = logging.getLogger("fpl_app.text_helpers")

#: Team names already warned about, so an unmapped club logs once per process
#: rather than once per row. See _to_short_team_code().
_UNKNOWN_TEAMS_WARNED = set()

# Timezone
TZ_ET = ZoneInfo("America/New_York")

# Team name mappings (RotoWire full names -> FPL short codes).
# Intentionally append-only across seasons: teams relegated out of the Premier
# League stay listed (rather than being deleted) since they may be promoted
# back in a future season, and keeping them costs nothing — they simply won't
# appear in current-season fixtures/rosters.
#
# Promoted clubs must still be *added* each season. Missing one is not loud: a
# newly-promoted Leeds fell through to _to_short_team_code()'s naive 3-letter
# guess, which happened to be right ("LEE") but logged a warning on every row of
# every player table. tests/live/ now fails when a current club is absent.
TEAM_FULL_TO_SHORT = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Coventry": "COV", "Coventry City": "COV",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Hull": "HUL", "Hull City": "HUL",
    "Ipswich": "IPS", "Ipswich Town": "IPS",
    "Leeds": "LEE", "Leeds United": "LEE", "Leeds Utd": "LEE",
    "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man Utd": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Southampton": "SOU", "Spurs": "TOT",
    "Sunderland": "SUN",
    "West Ham": "WHU", "Wolves": "WOL",
    # Common variations
    "Manchester City": "MCI", "Manchester United": "MUN",
    "Manchester Utd": "MUN", "Nottingham Forest": "NFO",
    "Tottenham": "TOT", "Tottenham Hotspur": "TOT",
    # Fantasy Football Pundit's spelling. Missing it sent all 28 Forest rows
    # past the (name, team) tiers and into the loose fallbacks.
    "Notts Forest": "NFO",
}

# Position mappings (various formats -> G/D/M/F)
POS_MAP_TO_RW = {
    "GK": "G", "GKP": "G", "G": "G", "Goalkeeper": "G",
    "DEF": "D", "D": "D", "Defender": "D",
    "MID": "M", "M": "M", "Midfielder": "M",
    "FWD": "F", "FW": "F", "F": "F", "Forward": "F",
}

# Player name aliases for difficult matches (FPL full name -> Rotowire display name)
# Brazilian players often have long full names but short display names
# Add new mappings here when you encounter incorrectly matched players
PLAYER_ALIASES = {
    "Carlos Henrique Casimiro": "Casemiro",
    "João Pedro Junqueira de Jesus": "Joao Pedro",
    "Pedro Porro Sauceda": "Pedro Porro",
}


# =============================================================================
# TEXT & STRING NORMALIZATION
# =============================================================================

def _clean_player_name(s: str) -> str:
    """Lowercase, remove accents and non-alphanumerics for robust matching keys."""
    s = _strip_accents(s).lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_text(x: str) -> str:
    """Lowercase, strip accents, collapse spaces for fuzzy matching."""
    s = str(x).strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = " ".join(s.lower().split())
    return s


def _strip_accents(s: str) -> str:
    """Remove diacritics/accents and normalize whitespace."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = s.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s).strip()


def clean_text(s: Any) -> str:
    """Clean and normalize text by collapsing whitespace."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_apostrophes(text):
    """
    Normalizes text by converting different apostrophe types to a standard straight apostrophe.

    Parameters:
    - text (str): The text to normalize.

    Returns:
    - str: The normalized text.
    """
    if text is None:
        return None
    # Normalize Unicode and replace curly apostrophes with straight apostrophes
    return unicodedata.normalize('NFKC', text).replace("\u2019", "'").strip().lower()


def normalize_name(name: str) -> str:
    """Remove diacritics, normalize spacing/case for matching."""
    if name is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = "".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub(r"\s+", " ", ascii_str).strip()


def remove_duplicate_words(name):
    """Function to remove duplicate consecutive words."""
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', name)


# =============================================================================
# POSITION & TEAM MAPPING
# =============================================================================

def _map_position_to_rw(pos_val):
    """Map any reasonable position variant to {'G','D','M','F'}."""
    if pd.isna(pos_val):
        return ""
    p = str(pos_val).strip()
    # direct mapping
    if p in POS_MAP_TO_RW:
        return POS_MAP_TO_RW[p]

    # If it's numeric (FPL element_type: 1..4)
    if p.isdigit():
        return {"1": "G", "2": "D", "3": "M", "4": "F"}.get(p, "")

    # Heuristics
    p_up = p.upper()
    for key, val in POS_MAP_TO_RW.items():
        if p_up.startswith(key):
            return val
    return p_up[:1]  # fallback: first letter


def _to_short_team_code(team_val, teams_df=None):
    """
    Convert a team value to a 3-letter short code.
    - If `teams_df` (FPL bootstrap teams) is provided, it should contain id + short_name.
    - If `team_val` already looks like a 3-letter code, keep it.
    - Else try mapping via TEAM_FULL_TO_SHORT.
    """
    if pd.isna(team_val):
        return ""
    s = str(team_val).strip()

    # Already like 'MCI'
    if re.fullmatch(r"[A-Z]{3}", s):
        return s

    # Try dictionary mapping (RotoWire-style team strings)
    if s in TEAM_FULL_TO_SHORT:
        return TEAM_FULL_TO_SHORT[s]

    # If it's a number and we have teams_df (FPL team id path)
    if teams_df is not None:
        try:
            tid = int(s)
            row = teams_df.loc[teams_df["id"] == tid]
            if not row.empty and "short_name" in row.columns:
                return str(row.iloc[0]["short_name"])
        except Exception:
            pass

    # Best effort: return uppercase 3-letter heuristic. This is a naive guess
    # (e.g. "Sheffield Utd" -> "SHE", not the real "SHU") — log it so a newly
    # promoted/renamed team that's missing from TEAM_FULL_TO_SHORT is visible
    # in logs rather than silently producing a wrong team code downstream.
    #
    # Warn once per unknown name per process. This runs per *row* of every player
    # table, so a single missing club used to emit the same line dozens of times
    # per page load and drown the log it was meant to draw attention to.
    guess = re.sub(r"[^A-Za-z]", "", s).upper()[:3]
    if s not in _UNKNOWN_TEAMS_WARNED:
        _UNKNOWN_TEAMS_WARNED.add(s)
        _logger.warning(
            "_to_short_team_code: %r not found in TEAM_FULL_TO_SHORT, guessing %r — "
            "add this team to TEAM_FULL_TO_SHORT if the guess is wrong.",
            s, guess,
        )
    return guess if len(guess) == 3 else s


def format_team_name(name):
    """
    Formats a team name by normalizing apostrophes and capitalizing each word.

    Parameters:
    - name (str): The team name to format.

    Returns:
    - str: The formatted team name.
    """
    if name is None:
        return None
    # Normalize Unicode and replace curly apostrophes with straight apostrophes
    normalized_name = unicodedata.normalize('NFKC', name).replace("\u2019", "'").strip()
    # Capitalize the first letter of each word
    return ' '.join(word.capitalize() for word in normalized_name.split())


def position_converter(element_type):
    """Converts element type to position name."""
    return {1: 'G', 2: 'D', 3: 'M', 4: 'F'}.get(element_type, 'Unknown')


def ordinal(n: int) -> str:
    """Formats an integer as an ordinal string, e.g. 1 -> '1st', 4 -> '4th'."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def to_display_name(first_name, second_name=None, web_name=None) -> str:
    """Build the app's standard display name for a player.

    **This is the preferred format for player names anywhere in the UI.**

    The FPL bootstrap stores a full legal name that nobody uses in conversation
    ("Bruno Borges Fernandes", "Rúben dos Santos Gato Alves Dias") and a
    `web_name` that is often too terse or abbreviated to stand alone
    ("B.Fernandes", "A.Becker", "Raya"). Neither is what a manager expects to
    read. This returns the common name instead: "Bruno Fernandes", "Alisson
    Becker", "David Raya".

    Rules, in order:
      1. Abbreviated web_name ("B.Fernandes") -> expand the initial from
         first_name: "Bruno Fernandes".
      2. web_name is already the whole name people use (it equals first_name,
         or first_name already contains it) -> use the longer of the two, so
         "Gabriel" stays "Gabriel" and "Thiago" becomes "Igor Thiago".
      3. web_name is a surname drawn from second_name -> "first_name web_name".
      4. Anything unexpected -> web_name, else the full name.

    Args:
        first_name: FPL `first_name`, or a full name if the others are omitted.
        second_name: FPL `second_name`.
        web_name: FPL `web_name`.

    Returns:
        The display name, or "" when there is nothing usable.
    """
    first = clean_text(first_name)
    second = clean_text(second_name)
    web = clean_text(web_name)

    if not web:
        return " ".join(p for p in (first, second) if p).strip()
    if not first:
        return web

    # 1. "B.Fernandes" / "A. Becker" -> take the surname, keep the real first name.
    if "." in web:
        surname = web.split(".")[-1].strip()
        if surname:
            return ("%s %s" % (first, surname)).strip()

    first_norm = _strip_accents(first).lower()
    web_norm = _strip_accents(web).lower()

    # 2. web_name adds nothing to first_name (mononyms, and players whose
    #    surname already sits in the first_name field).
    if web_norm == first_norm or web_norm in first_norm.split():
        return first if len(first) >= len(web) else web

    # 3. The usual case: web_name is the surname.
    if not second or web_norm in _strip_accents(second).lower():
        return ("%s %s" % (first, web)).strip()

    # 4. Unrecognised shape -- web_name is still the safer of the two to show.
    return web


def format_last_updated(when, include_age: bool = True) -> str:
    """Render a source's publish time as "Aug 20, 2026 10:54 AM ET (3h ago)".

    The age is the point of this: a weekly projection table published before the
    last team-news cycle is worth materially less than one published after it,
    and the raw timestamp alone doesn't make that obvious at a glance.

    Args:
        when: Timezone-aware datetime, or None.
        include_age: Append the relative age in parentheses.

    Returns:
        Formatted string, or "Unknown" when `when` is None.
    """
    if when is None:
        return "Unknown"

    try:
        local = when.astimezone(TZ_ET)
    except (ValueError, TypeError):
        return "Unknown"

    stamp = local.strftime("%b %-d, %Y %-I:%M %p ET")
    if not include_age:
        return stamp

    delta = datetime.now(TZ_ET) - local
    minutes = delta.total_seconds() / 60.0
    if minutes < 0:
        age = "just now"          # clock skew between us and the source
    elif minutes < 60:
        age = "%dm ago" % int(minutes)
    elif minutes < 60 * 24:
        age = "%dh ago" % int(minutes // 60)
    else:
        days = int(minutes // (60 * 24))
        age = "1 day ago" if days == 1 else "%d days ago" % days
    return "%s (%s)" % (stamp, age)


def canonical_normalize(name: str) -> str:
    """
    Single source of truth for name normalization.

    Converts player names to a canonical form for matching:
    - Manual substitution for special characters (ø, æ, ð, etc.)
    - NFKD unicode normalize (decomposes accented characters)
    - ASCII encode (strips accents)
    - Lowercase
    - Remove non-alphanumeric characters
    - Collapse whitespace

    Examples:
        "Raúl Jiménez" -> "raul jimenez"
        "Bruno Fernandes" -> "bruno fernandes"
        "Heung-Min Son" -> "heungmin son"
        "N'Golo Kanté" -> "ngolo kante"
        "Rasmus Højlund" -> "rasmus hojlund"

    Args:
        name: Player name to normalize

    Returns:
        Canonical normalized name string
    """
    if pd.isna(name) or name is None:
        return ""

    s = str(name).strip()

    # Manual substitution for special characters that don't decompose cleanly
    special_chars = {
        'ø': 'o', 'Ø': 'O',
        'æ': 'ae', 'Æ': 'AE',
        'œ': 'oe', 'Œ': 'OE',
        'ð': 'd', 'Ð': 'D',
        'þ': 'th', 'Þ': 'Th',
        'ł': 'l', 'Ł': 'L',
        'đ': 'd', 'Đ': 'D',
        'ß': 'ss',
    }
    for char, replacement in special_chars.items():
        s = s.replace(char, replacement)

    # NFKD decomposition separates accents from base characters
    s = unicodedata.normalize("NFKD", s)

    # Encode to ASCII (drops accent characters) then decode back
    s = s.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    s = s.lower()

    # Remove non-alphanumeric except spaces
    s = re.sub(r"[^a-z0-9 ]", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s
