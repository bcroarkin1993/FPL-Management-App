"""
Lightweight error-handling helpers for the FPL Management App.

* ``get_logger(name)`` — returns a stdlib logger that writes to stderr
  (visible in the terminal / container logs).
* ``show_api_error(context, …)`` — renders a user-facing ``st.error``
  with an actionable hint and optionally logs the exception.

NOTE: ``@st.cache_data`` functions must NOT call ``st.error`` / ``st.warning``
(Streamlit caches return values, not side effects).  Those functions should
use ``get_logger().warning(…)`` only.  Page-level (non-cached) functions
may use ``show_api_error(…)`` for user-facing messages.
"""

import logging
import streamlit as st


def get_logger(name: str = "fpl_app") -> logging.Logger:
    """App-wide logger with a StreamHandler (visible in terminal / container logs)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# Standard remediation hints shown to the user.
_HINTS = {
    "api_down":  "The FPL API may be temporarily unavailable. Try refreshing in a few minutes.",
    "league_id": "Please verify your league ID in `.env` (`FPL_DRAFT_LEAGUE_ID` or `FPL_CLASSIC_LEAGUE_IDS`).",
    "team_id":   "Please verify your team ID in `.env` (`FPL_DRAFT_TEAM_ID` or `FPL_CLASSIC_TEAM_ID`).",
    "rotowire":  "Rotowire may have changed their page layout, or projections aren't published yet for this gameweek.",
    "preseason": "This data becomes available once the season starts and games are played.",
    "network":   "Check your internet connection and try again.",
}


def show_api_error(
    context: str,
    *,
    hint_key: str = "api_down",
    exception: Exception = None,
    stop: bool = False,
) -> None:
    """Display a user-friendly ``st.error`` with an actionable hint.

    Parameters
    ----------
    context : str
        A short phrase describing what was happening, e.g.
        ``"loading player data for transfer analysis"``.
    hint_key : str
        Key into ``_HINTS`` for the remediation message.
    exception : Exception, optional
        If provided, the exception is logged at WARNING level.
    stop : bool
        If ``True``, ``st.stop()`` is called after displaying the error.
    """
    hint = _HINTS.get(hint_key, _HINTS["api_down"])
    st.error(f"**Could not load data** while {context}.\n\n{hint}")
    if exception:
        get_logger().warning("Error while %s: %s", context, exception)
    if stop:
        st.stop()
