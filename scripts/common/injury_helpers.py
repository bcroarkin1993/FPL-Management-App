"""
Injury duration helpers — shared across Draft and Classic.

The FPL bootstrap exposes three availability signals of decreasing reliability:
a free-text ``news`` string (often carrying an explicit return date), a coarse
``chance_of_playing_next_round`` percentage, and a single-letter ``status`` code.
``estimate_games_to_miss`` walks them in that order.

This module is pure (no Streamlit, no network) so it is safe to import from
tests and from GitHub Actions.
"""

import re
from datetime import datetime

import pandas as pd

from scripts.common.error_helpers import get_logger

_logger = get_logger("fpl_app.injury_helpers")

# Total gameweeks in a Premier League season.
TOTAL_GWS = 38

# Floor on the injury multiplier.  A long-term-injured elite asset is still a
# rostered asset who eventually returns — he is never worth literally zero.
INJURY_FLOOR = 0.10


def estimate_games_to_miss(news, chance, status) -> int:
    """Estimate how many gameweeks a player will miss.

    Resolution order, most to least reliable:
      1. An explicit return date in ``news`` ("Expected back 15 Nov").
      2. A suspension length in ``news`` ("Suspended for 3 matches").
      3. ``chance_of_playing`` buckets.
      4. The ``status`` code.

    Returns 0 for a fully available player.

    Moved verbatim from ``scripts/draft/waiver_wire.py`` so Draft waiver logic and
    team-strength scoring share one implementation.
    """
    news_str = "" if pd.isna(news) else str(news).strip()

    if news_str:
        # 1. Try "Expected back DD Mon" or similar date patterns
        back_match = re.search(
            r'(?:expected\s+back|return[s]?\s+)\s*(\d{1,2}\s+\w+(?:\s+\d{4})?)',
            news_str, re.IGNORECASE
        )
        if back_match:
            date_str = back_match.group(1)
            for fmt in ('%d %b %Y', '%d %B %Y', '%d %b', '%d %B'):
                try:
                    parsed = datetime.strptime(date_str, fmt)
                    if parsed.year == 1900:  # no year in format
                        now = datetime.now()
                        parsed = parsed.replace(year=now.year)
                        if parsed < now:
                            parsed = parsed.replace(year=now.year + 1)
                    days_until = (parsed - datetime.now()).days
                    return max(0, (days_until + 6) // 7)  # round up to GWs
                except ValueError:
                    continue

        # 2. Try "Suspended for X" matches
        susp_match = re.search(r'suspended\s+(?:for\s+)?(\d+)', news_str, re.IGNORECASE)
        if susp_match:
            return int(susp_match.group(1))

    # 3. Fallback from chance_of_playing
    if not pd.isna(chance):
        try:
            c = float(chance)
            if c >= 75:
                return 1
            if c >= 50:
                return 2
            if c >= 25:
                return 3
            return 5
        except (ValueError, TypeError):
            pass

    # 4. Fallback from status
    if not pd.isna(status):
        s = str(status).lower()
        if s == 'a':
            return 0
        if s == 'd':
            return 2
        if s in ('i', 'n'):
            return 4
        if s in ('s', 'u'):
            return 3

    return 0


def gameweeks_remaining(current_gw, total_gws: int = TOTAL_GWS) -> int:
    """Gameweeks left in the season *including* the current one.  Never below 1."""
    try:
        gw = int(current_gw)
    except (ValueError, TypeError):
        gw = 1
    return max(1, total_gws - gw + 1)


def injury_multiplier(gws_missed, current_gw, total_gws: int = TOTAL_GWS,
                      floor: float = INJURY_FLOOR) -> float:
    """Season-aware availability multiplier in ``[floor, 1.0]``.

    Scales by the *fraction of the remaining season* a player will miss, so the
    same absence costs more the later it lands:

        GW3,  5 GWs missed of 36 remaining -> 1 - 0.139 = 0.861
        GW20, 5 GWs missed of 19 remaining -> 1 - 0.263 = 0.737
        GW34, 5 GWs missed of  5 remaining -> floor

    A fully fit player (``gws_missed`` 0) always returns exactly 1.0.
    """
    try:
        missed = float(gws_missed)
    except (ValueError, TypeError):
        return 1.0

    if pd.isna(missed) or missed <= 0:
        return 1.0

    remaining = gameweeks_remaining(current_gw, total_gws)
    frac = min(1.0, missed / remaining)
    return max(floor, 1.0 - frac)
