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

import contextlib
import logging
import traceback
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
    "league_id": (
        "Please verify your league ID on the **🆔 League Setup** page "
        "(or `FPL_DRAFT_LEAGUE_ID` / `FPL_CLASSIC_LEAGUE_IDS` in `.env` if not using that page)."
    ),
    "draft_league_stale": (
        "FPL Draft leagues don't carry over between seasons — this is likely still last "
        "season's league ID. Create your new league at "
        "[draft.premierleague.com](https://draft.premierleague.com), then update it on the "
        "**🆔 League Setup** page (or `FPL_DRAFT_LEAGUE_ID` in `.env`)."
    ),
    "classic_league_stale": (
        "This Classic/H2H league ID no longer resolves — private mini-leagues are "
        "sometimes recreated with a new ID each season. Find the current ID on "
        "[fantasy.premierleague.com](https://fantasy.premierleague.com), then update it on "
        "the **🆔 League Setup** page (or `FPL_CLASSIC_LEAGUE_IDS` in `.env`)."
    ),
    "team_id": (
        "Please verify your team ID on the **🆔 League Setup** page "
        "(or `FPL_DRAFT_TEAM_ID` / `FPL_CLASSIC_TEAM_ID` in `.env` if not using that page)."
    ),
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


@contextlib.contextmanager
def page_error_boundary(page_name: str):
    """Catch anything a page raises and render it as a legible, actionable error.

    Without this, an unhandled exception anywhere in a page hands the reader
    Streamlit's raw traceback -- which is both alarming and useless to them --
    while the log line that would actually help scrolls past in a terminal
    nobody is watching. The opposite failure is just as bad: a bare `except`
    that swallows the error and leaves a blank space where a section should be.

    This does neither. The reader gets a plain statement of which page failed
    and what to try, the traceback stays available behind a disclosure, and the
    full exception is logged at ERROR with a stack trace.

    Usage::

        with page_error_boundary("Projections Hub"):
            show_player_projections_page()
    """
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all at the page boundary
        get_logger().error("Unhandled error rendering page %r", page_name, exc_info=True)
        st.error(
            f"**Something went wrong on {page_name}.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            "The rest of the app still works — pick another page from the sidebar, "
            "or use the Refresh button to re-fetch data. If it keeps happening, the "
            "details below identify where it broke."
        )
        with st.expander("Technical details"):
            st.code(traceback.format_exc(), language="text")
