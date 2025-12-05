# scripts/common/utils.py

import re
import unicodedata
from typing import Any, Optional


# ==============================================================================
# STRING MANIPULATION & FORMATTING
# ==============================================================================

def clean_text(s: Any) -> str:
    """
    Standardizes input to a clean string.
    Handles None, NaN (float), and integers by converting to string
    and stripping extra whitespace.
    """
    if s is None:
        return ""
    # Check for NaN (float('nan')) without importing numpy/pandas
    if isinstance(s, float) and s != s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    # Collapse multiple spaces into one and strip ends
    return re.sub(r"\s+", " ", s).strip()


def format_team_name(name: Optional[str]) -> Optional[str]:
    """
    Formats a team name by normalizing apostrophes and capitalizing each word.
    Example: "nott'm forest" -> "Nott'm Forest"
    """
    if name is None:
        return None

    # Normalize Unicode and replace curly apostrophes with straight ones
    normalized_name = unicodedata.normalize('NFKC', name).replace('’', "'").strip()

    # Capitalize the first letter of each word (Title Case)
    return ' '.join(word.capitalize() for word in normalized_name.split())


def normalize_text(x: Any) -> str:
    """
    Aggressive normalization for fuzzy matching.
    1. Converts to string.
    2. Decomposes unicode chars (NFKD) to strip accents.
    3. Lowercases.
    4. Collapses all whitespace.
    """
    s = str(x).strip() if x is not None else ""

    # Strip accents (e.g. "Fernándes" -> "Fernandes")
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

    # Lowercase and collapse spaces
    return " ".join(s.lower().split())


def remove_duplicate_words(name: str) -> str:
    """
    Removes duplicate consecutive words.
    Useful for cleaning dirty data sources (e.g., "Salah Salah" -> "Salah").
    """
    if not isinstance(name, str):
        return ""
    # Regex matches a word boundary, a word, whitespace, then the same word again
    return re.sub(r'\b(\w+)\s+\1\b', r'\1', name)