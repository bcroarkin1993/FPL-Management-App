"""
App-side bridge between the pure transfer-risk model and the app's data sources.

``transfer_risk.py`` and ``transfer_feeds.py`` stay free of Streamlit and of
``player_matching`` so GitHub Actions can import them.  The cross-source name
matching that Draft and Classic pages need lives here instead, because
``ReferenceMatcher`` pulls in Streamlit.
"""

import pandas as pd

from scripts.common.error_helpers import get_logger
from scripts.common.player_matching import ReferenceMatcher
from scripts.common.transfer_risk import RISK_COLUMNS, attach_transfer_risk

_logger = get_logger("fpl_app.transfer_risk_app")


def get_pl_team_names(bootstrap) -> list:
    """Full club names of the current Premier League, from the live bootstrap.

    Used to decide whether a destination keeps a player in the game.  Read from
    live data, never hardcoded, so promotion and relegation need no maintenance —
    which matters because a Championship move costs a manager exactly as much as
    a Saudi one.
    """
    try:
        teams = (bootstrap or {}).get("teams") or []
        return [t.get("name") for t in teams if t.get("name")]
    except AttributeError:
        _logger.warning("Bootstrap had no usable teams list for transfer risk")
        return []


def attach_bootstrap_availability(rankings_df: pd.DataFrame,
                                  availability_df: pd.DataFrame) -> pd.DataFrame:
    """Merge FPL ``status`` / ``news`` onto a Rotowire-shaped frame.

    Rotowire publishes common names ("Bruno Fernandes"); the bootstrap publishes
    full legal names ("Bruno Borges Fernandes").  A strict key misses a sixth of
    the table *silently*, so this goes through the shared tiered matcher rather
    than a merge.

    Players with no match keep empty status/news, which reads as "no completed
    transfer" — the safe default, since news scoring still applies.
    """
    result = rankings_df.copy()
    result["status"] = ""
    result["news"] = ""

    if availability_df is None or availability_df.empty:
        _logger.warning("No FPL availability data — transfer ground truth unavailable")
        return result

    matcher = ReferenceMatcher(
        availability_df, name_col="Player", web_name_col="Web_Name",
        team_col="Team", position_col="Position",
    )

    statuses, newses = [], []
    matched = 0
    for _, row in result.iterrows():
        idx = matcher.match(row.get("Player"), row.get("Team"), row.get("Position"))
        if idx is None:
            statuses.append("")
            newses.append("")
            continue
        matched += 1
        ref = availability_df.loc[idx]
        statuses.append(ref.get("Status") or "")
        newses.append(ref.get("News") or "")

    result["status"] = statuses
    result["news"] = newses
    _logger.info("Transfer risk: matched %d/%d players to FPL availability",
                 matched, len(result))
    return result


def build_transfer_risk(rankings_df: pd.DataFrame,
                        availability_df: pd.DataFrame,
                        news_df: pd.DataFrame,
                        pl_teams,
                        today=None) -> pd.DataFrame:
    """Full pipeline: ground truth + news -> risk columns on the rankings frame.

    Never raises. Any failure degrades to a neutral multiplier of 1.0 for every
    player, which is the pre-existing behaviour — a broken news feed must not
    take the draft board down.
    """
    try:
        enriched = attach_bootstrap_availability(rankings_df, availability_df)
        return attach_transfer_risk(enriched, news_df, pl_teams, today=today)
    except Exception as e:
        _logger.warning("Transfer risk pipeline failed: %s", e, exc_info=True)
        fallback = rankings_df.copy()
        fallback["Transfer_Risk"] = 0.0
        fallback["Transfer_Exposure"] = 0.0
        fallback["Transfer_Mult"] = 1.0
        fallback["Transfer_Destination"] = ""
        fallback["Transfer_Outlets"] = 0
        fallback["Transfer_Note"] = ""
        return fallback
