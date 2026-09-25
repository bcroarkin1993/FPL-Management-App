"""
Injury duration helpers — shared across Draft and Classic.

The FPL bootstrap exposes three availability signals of decreasing reliability:
a free-text ``news`` string (often carrying an explicit return date), a coarse
``chance_of_playing_next_round`` percentage, and a single-letter ``status`` code.
``estimate_games_to_miss`` walks them in that order.

This module is pure (no Streamlit, no network) so it is safe to import from
tests and from GitHub Actions.
"""

import logging
import re
from datetime import datetime

import pandas as pd

# Plain `logging`, not `error_helpers.get_logger`: that module imports Streamlit,
# and this one's docstring has always claimed purity -- `projection_engine` now
# imports it, and a test asserts the engine loads with Streamlit absent. The
# logger it pulled in was never used.
_logger = logging.getLogger("fpl_app.injury_helpers")

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
            # **100 means fit, and must return 0.** FPL states an explicit 100
            # once news is resolved -- live, 84 players carry `chance == 100`
            # with `status == 'a'` -- so without this the bucket below reported
            # them as missing a gameweek, and `team_strength` applied an injury
            # discount to a fully available squad. The `status` check that would
            # have returned 0 is never reached, because a stated chance wins
            # over it by design.
            if c >= 100:
                return 0
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


#: Status codes meaning FPL has taken the player out of the squad, rather than
#: merely flagged a doubt. ``d`` is deliberately absent: a doubtful player may
#: well play, which is what makes him doubtful.
OUT_OF_SQUAD_STATUSES = ("i", "s", "u", "n")


def stated_games_to_miss(news, chance, status):
    """Games missed where the *source states a duration*, else ``None``.

    :func:`estimate_games_to_miss` always answers, falling back through
    ``chance`` buckets to the ``status`` code, which is right for a discount --
    something is better than nothing. It is wrong for anything that treats the
    answer as a fact about *future* gameweeks, because the buckets are a guess:
    "25% chance" becomes "misses 3 games" on no evidence at all, and applied to
    a three-gameweek horizon that writes off a player who may be back next week.

    So this returns a number only where the duration is stated:

      * an explicit return date in ``news`` ("Expected back 11 Oct"),
      * a suspension length ("Suspended for 3 matches"),
      * a status that means *out of the squad* rather than doubtful.

    A doubtful player with no date gets ``None``: whether he plays *this* week
    is already priced by his start probability, and nothing is known about the
    two weeks after it.
    """
    # Ask for the news-derived duration *alone*: with chance and status withheld,
    # the bucket fallbacks cannot fire, so anything above zero came from a real
    # date or a suspension length. Matching the keywords by hand instead lets
    # "Unspecified injury - Unknown return date" through on the word "return",
    # and then answers from the very buckets this function exists to exclude.
    from_news = estimate_games_to_miss(news, None, None)
    if from_news > 0:
        return from_news

    if not pd.isna(status) and str(status).lower() in OUT_OF_SQUAD_STATUSES:
        return estimate_games_to_miss(None, None, status)

    return None


def games_to_miss_series(news, chance, status, index) -> pd.Series:
    """:func:`stated_games_to_miss` over aligned Series, 0 where nothing is stated.

    The scalar version runs regexes over free text, so it is only called for
    players carrying *some* availability signal -- a status that is not ``a`` or
    a stated chance below 100. Live that is 205 of 667 rows, and the rest are
    zero by definition rather than by computation.
    """
    # Callers pass whatever they have -- `blend_aligned` takes a bare list as
    # often as an Index -- so normalise before any positional indexing.
    index = pd.Index(index)
    news = _as_series(news, index)
    chance = pd.to_numeric(_as_series(chance, index), errors="coerce")
    status = _as_series(status, index).astype("object")

    flagged = (status.notna() & ~status.isin(["a", ""])) | (chance.notna() & chance.lt(100))
    flagged = flagged.reindex(index).fillna(False).astype(bool)

    out = pd.Series(0, index=index, dtype="int64")
    for idx in index[flagged.to_numpy()]:
        stated = stated_games_to_miss(news.get(idx), chance.get(idx), status.get(idx))
        if stated is not None:
            out.at[idx] = stated
    return out


def _as_series(value, index) -> pd.Series:
    """``value`` as a Series on ``index``; an all-NaN one when it is absent."""
    if value is None:
        return pd.Series([float("nan")] * len(index), index=index, dtype="object")
    if isinstance(value, pd.Series):
        return value.reindex(index)
    return pd.Series([value] * len(index), index=index)


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
