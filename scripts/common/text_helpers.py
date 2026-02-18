"""
Text & String Normalization, Constants, and Position/Team Mapping.

Shared text-processing utilities, team name mappings, and position converters
used across the FPL Management App.
"""

import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

from zoneinfo import ZoneInfo

# Timezone
TZ_ET = ZoneInfo("America/New_York")

# Team name mappings (RotoWire full names -> FPL short codes)
TEAM_FULL_TO_SHORT = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU",
    "Brentford": "BRE", "Brighton": "BHA", "Chelsea": "CHE",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL",
    "Ipswich": "IPS", "Leicester": "LEI", "Liverpool": "LIV",
    "Man City": "MCI", "Man Utd": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Southampton": "SOU", "Spurs": "TOT",
    "West Ham": "WHU", "Wolves": "WOL",
    # Common variations
    "Manchester City": "MCI", "Manchester United": "MUN",
    "Manchester Utd": "MUN", "Nottingham Forest": "NFO",
    "Tottenham": "TOT", "Tottenham Hotspur": "TOT",
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

    # Best effort: return uppercase 3-letter heuristic
    guess = re.sub(r"[^A-Za-z]", "", s).upper()[:3]
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
