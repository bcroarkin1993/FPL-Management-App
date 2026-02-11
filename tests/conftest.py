"""
Shared test fixtures for FPL Management App.

This module:
1. Neutralizes Streamlit caching decorators before any app code imports
2. Sets environment variables so config.py doesn't hit the network
3. Provides mock data fixtures for bootstrap, league, and H2H data
4. Provides a mock_streamlit fixture that patches all st.* display calls
5. Provides a mock_all_utils fixture for smoke tests
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# =====================================================================
# 1. Neutralize Streamlit caching BEFORE any app modules are imported
# =====================================================================
# Replace st.cache_data and st.cache_resource with pass-through decorators
# so that cached functions behave normally during testing.

import streamlit as st


def _passthrough_decorator(*args, **kwargs):
    """Decorator that does nothing — returns the function unchanged."""
    if args and callable(args[0]):
        return args[0]
    def wrapper(fn):
        return fn
    return wrapper


st.cache_data = _passthrough_decorator
st.cache_resource = _passthrough_decorator

# =====================================================================
# 2. Set environment variables so config.py doesn't hit the network
# =====================================================================
os.environ["FPL_CURRENT_GAMEWEEK"] = "25"
os.environ["ROTOWIRE_URL"] = "https://example.com/rotowire-test"
os.environ["FPL_DRAFT_LEAGUE_ID"] = "12345"
os.environ["FPL_DRAFT_TEAM_ID"] = "67890"
os.environ["FPL_CLASSIC_TEAM_ID"] = "11111"
os.environ["FPL_CLASSIC_LEAGUE_IDS"] = "99999:Test League"


# =====================================================================
# 3. Mock data fixtures
# =====================================================================

@pytest.fixture
def mock_bootstrap_data():
    """Minimal bootstrap-static data with 6 players across positions and 3 teams."""
    return {
        "teams": [
            {"id": 1, "short_name": "ARS", "name": "Arsenal"},
            {"id": 2, "short_name": "MCI", "name": "Man City"},
            {"id": 3, "short_name": "LIV", "name": "Liverpool"},
        ],
        "elements": [
            {
                "id": 1, "first_name": "Aaron", "second_name": "Ramsdale",
                "web_name": "Ramsdale", "team": 1, "element_type": 1,
                "total_points": 80, "goals_scored": 0, "assists": 1, "starts": 20,
            },
            {
                "id": 2, "first_name": "William", "second_name": "Saliba",
                "web_name": "Saliba", "team": 1, "element_type": 2,
                "total_points": 120, "goals_scored": 3, "assists": 2, "starts": 24,
            },
            {
                "id": 3, "first_name": "Kevin", "second_name": "De Bruyne",
                "web_name": "De Bruyne", "team": 2, "element_type": 3,
                "total_points": 160, "goals_scored": 10, "assists": 15, "starts": 22,
            },
            {
                "id": 4, "first_name": "Erling", "second_name": "Haaland",
                "web_name": "Haaland", "team": 2, "element_type": 4,
                "total_points": 200, "goals_scored": 25, "assists": 5, "starts": 25,
            },
            {
                "id": 5, "first_name": "Mohamed", "second_name": "Salah",
                "web_name": "Salah", "team": 3, "element_type": 3,
                "total_points": 180, "goals_scored": 18, "assists": 10, "starts": 24,
            },
            {
                "id": 6, "first_name": "Virgil", "second_name": "van Dijk",
                "web_name": "Van Dijk", "team": 3, "element_type": 2,
                "total_points": 130, "goals_scored": 4, "assists": 1, "starts": 25,
            },
        ],
        "events": [
            {"id": 1, "finished": True},
            {"id": 2, "finished": True},
            {"id": 25, "finished": False, "is_current": True},
        ],
    }


@pytest.fixture
def mock_league_response():
    """Minimal Draft league details with matches, entries, standings."""
    return {
        "league_entries": [
            {"id": 1, "entry_id": 101, "entry_name": "Team Alpha", "player_first_name": "Alice", "player_last_name": "Smith"},
            {"id": 2, "entry_id": 102, "entry_name": "Team Beta", "player_first_name": "Bob", "player_last_name": "Jones"},
            {"id": 3, "entry_id": 103, "entry_name": "Team Gamma", "player_first_name": "Carol", "player_last_name": "Lee"},
        ],
        "matches": [
            # GW 1
            {"event": 1, "league_entry_1": 1, "league_entry_2": 2, "league_entry_1_points": 60, "league_entry_2_points": 55},
            {"event": 1, "league_entry_1": 3, "league_entry_2": 1, "league_entry_1_points": 0, "league_entry_2_points": 0},  # bye/unplayed
            # GW 2
            {"event": 2, "league_entry_1": 2, "league_entry_2": 3, "league_entry_1_points": 70, "league_entry_2_points": 80},
            {"event": 2, "league_entry_1": 1, "league_entry_2": 3, "league_entry_1_points": 45, "league_entry_2_points": 0},  # valid: Team Gamma scored 0
        ],
        "standings": [
            {"league_entry": 1, "rank": 1, "total": 105, "event_total": 45},
            {"league_entry": 2, "rank": 2, "total": 125, "event_total": 70},
            {"league_entry": 3, "rank": 3, "total": 80, "event_total": 80},
        ],
    }


@pytest.fixture
def mock_h2h_matches():
    """Minimal Classic H2H match data."""
    return [
        {"event": 1, "entry_1_name": "Team A", "entry_1_entry": 201, "entry_1_points": 50,
         "entry_2_name": "Team B", "entry_2_entry": 202, "entry_2_points": 45, "finished": True},
        {"event": 1, "entry_1_name": "Team C", "entry_1_entry": 203, "entry_1_points": 60,
         "entry_2_name": "Team A", "entry_2_entry": 201, "entry_2_points": 0, "finished": True},  # valid 0
        {"event": 2, "entry_1_name": "Team B", "entry_1_entry": 202, "entry_1_points": 0,
         "entry_2_name": "Team C", "entry_2_entry": 203, "entry_2_points": 0, "finished": False},  # unplayed
    ]


# =====================================================================
# 4. Streamlit no-op fixture
# =====================================================================

class _SessionState(dict):
    """Dict subclass that supports attribute access (like Streamlit's session_state)."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name)


class _MockColumn:
    """Mock for st.columns() return values that support context manager."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __getattr__(self, name):
        return MagicMock()


@pytest.fixture
def mock_streamlit():
    """
    Patches all st.* display functions as no-ops.
    st.columns() and st.tabs() return lists of context-manager mocks.
    st.sidebar is a MagicMock with context-manager support.
    """
    patches = {}
    display_funcs = [
        "title", "header", "subheader", "markdown", "write", "text",
        "dataframe", "table", "json", "metric", "caption",
        "info", "warning", "error", "success", "exception",
        "image", "plotly_chart", "pyplot", "altair_chart",
        "button", "download_button",
        "form", "form_submit_button",
        "spinner", "progress", "balloons", "snow", "toast",
        "set_page_config",
        "divider", "toggle", "empty",
    ]

    active_patches = []

    for func_name in display_funcs:
        p = patch(f"streamlit.{func_name}", new_callable=MagicMock)
        active_patches.append(p)
        patches[func_name] = p.start()

    # Widget mocks that return sensible defaults
    def mock_selectbox(label="", options=None, *args, **kwargs):
        opts = options or kwargs.get("options", [])
        idx = kwargs.get("index", 0)
        if opts and 0 <= idx < len(opts):
            return opts[idx]
        return opts[0] if opts else None

    def mock_radio(label="", options=None, *args, **kwargs):
        opts = options or kwargs.get("options", [])
        idx = kwargs.get("index", 0)
        if opts and 0 <= idx < len(opts):
            return opts[idx]
        return opts[0] if opts else None

    def mock_multiselect(label="", options=None, *args, **kwargs):
        return kwargs.get("default", [])

    for name, side_eff in [
        ("selectbox", mock_selectbox),
        ("radio", mock_radio),
        ("multiselect", mock_multiselect),
    ]:
        p = patch(f"streamlit.{name}", side_effect=side_eff)
        active_patches.append(p)
        patches[name] = p.start()

    # Simple widget defaults
    for name, ret in [
        ("checkbox", False),
        ("text_input", ""),
        ("text_area", ""),
        ("number_input", 0),
        ("slider", 0),
        ("date_input", None),
        ("time_input", None),
        ("color_picker", "#000000"),
        ("file_uploader", None),
    ]:
        p = patch(f"streamlit.{name}", return_value=ret)
        active_patches.append(p)
        patches[name] = p.start()

    # st.stop raises SystemExit to halt page execution (matching real Streamlit behavior)
    class _StopException(Exception):
        pass

    def _mock_stop():
        raise _StopException("st.stop()")

    p = patch("streamlit.stop", side_effect=_mock_stop)
    active_patches.append(p)
    patches["stop"] = p.start()
    patches["_StopException"] = _StopException  # expose for tests to catch

    # st.rerun as no-op (pages may call this)
    p = patch("streamlit.rerun", new_callable=MagicMock)
    active_patches.append(p)
    patches["rerun"] = p.start()

    # st.columns returns list of context-manager mocks
    def mock_columns(spec=None, *args, **kwargs):
        n = spec if isinstance(spec, int) else len(spec) if isinstance(spec, (list, tuple)) else 2
        return [_MockColumn() for _ in range(n)]

    patches["columns"] = patch("streamlit.columns", side_effect=mock_columns)
    active_patches.append(patches["columns"])
    patches["columns"].start()

    # st.tabs returns list of context-manager mocks
    def mock_tabs(labels):
        return [_MockColumn() for _ in labels]

    patches["tabs"] = patch("streamlit.tabs", side_effect=mock_tabs)
    active_patches.append(patches["tabs"])
    patches["tabs"].start()

    # st.expander returns a context-manager mock
    def mock_expander(*args, **kwargs):
        return _MockColumn()

    patches["expander"] = patch("streamlit.expander", side_effect=mock_expander)
    active_patches.append(patches["expander"])
    patches["expander"].start()

    # st.container returns a context-manager mock
    patches["container"] = patch("streamlit.container", side_effect=lambda *a, **kw: _MockColumn())
    active_patches.append(patches["container"])
    patches["container"].start()

    # st.sidebar
    sidebar_mock = MagicMock()
    sidebar_mock.__enter__ = MagicMock(return_value=sidebar_mock)
    sidebar_mock.__exit__ = MagicMock(return_value=False)
    patches["sidebar"] = patch("streamlit.sidebar", sidebar_mock)
    active_patches.append(patches["sidebar"])
    patches["sidebar"].start()

    # st.session_state as an AttrDict (supports both dict and attribute access)
    patches["session_state"] = patch("streamlit.session_state", _SessionState())
    active_patches.append(patches["session_state"])
    patches["session_state"].start()

    yield patches

    for p in active_patches:
        p.stop()


# =====================================================================
# 5. mock_all_utils fixture for smoke tests
# =====================================================================

import pandas as pd


@pytest.fixture
def mock_all_utils(mock_streamlit):
    """
    Patches commonly-imported get_* functions from scripts.common.utils
    to return safe defaults (empty dicts/DataFrames/lists).
    Individual tests can override specific mocks as needed.
    """
    util_patches = {
        # Draft API
        "get_current_gameweek": 25,
        "get_fpl_player_mapping": {},
        "get_league_entries": {},
        "get_league_player_ownership": {},
        "get_draft_league_details": {"matches": [], "league_entries": [], "standings": []},
        "get_draft_picks": {},
        "get_gameweek_fixtures": [],
        "get_historical_team_scores": pd.DataFrame(),
        "get_draft_h2h_record": {"wins": 0, "draws": 0, "losses": 0, "points_for": 0, "points_against": 0, "record_str": "0-0-0", "matches": []},
        "get_draft_all_h2h_records": {},
        "get_draft_points_by_position": {},
        "get_draft_team_players_with_points": [],
        "get_starting_team_composition": {},
        "get_team_composition_for_gameweek": {},
        "get_team_id_by_name": 0,
        "get_league_teams": {},
        "get_live_gameweek_stats": {},
        "is_gameweek_live": False,
        "get_team_actual_lineup": pd.DataFrame(),
        # Classic API
        "get_classic_bootstrap_static": {"elements": [], "teams": [], "events": []},
        "get_classic_league_standings": None,
        "get_classic_team_history": None,
        "get_classic_team_picks": None,
        "get_classic_transfers": [],
        "get_classic_team_position_data": {},
        "get_entry_details": {"name": "Test Team", "id": 11111},
        "get_league_standings": None,
        "get_all_h2h_league_matches": [],
        "get_h2h_league_matches": [],
        "get_h2h_league_standings": None,
        "get_classic_h2h_record": {"wins": 0, "draws": 0, "losses": 0, "record_str": "0-0-0", "matches": []},
        # Rotowire
        "get_rotowire_player_projections": pd.DataFrame(),
        "get_rotowire_season_rankings": pd.DataFrame(),
        "get_rotowire_rankings_url": "https://example.com",
        # Player data
        "merge_fpl_players_and_projections": pd.DataFrame(),
        "pull_fpl_player_stats": pd.DataFrame(),
        "normalize_fpl_players_to_rotowire_schema": pd.DataFrame(),
        "normalize_rotowire_players": pd.DataFrame(),
        "clean_fpl_player_names": pd.DataFrame(),
        "find_optimal_lineup": pd.DataFrame(),
        "position_converter": "M",
        # Fixtures
        "get_fixture_difficulty_grid": pd.DataFrame(),
        "style_fixture_difficulty": MagicMock(),
        # Projections
        "get_ffp_projections_data": pd.DataFrame(),
        "get_ffp_goalscorer_odds": pd.DataFrame(),
        "get_ffp_clean_sheet_odds": pd.DataFrame(),
        "get_odds_api_match_odds": pd.DataFrame(),
    }

    active = []
    mocks = {}
    for func_name, return_val in util_patches.items():
        p = patch(f"scripts.common.utils.{func_name}", return_value=return_val)
        active.append(p)
        mocks[func_name] = p.start()

    # Also patch config attributes that pages read
    config_patches = [
        patch("config.FPL_DRAFT_LEAGUE_ID", 12345),
        patch("config.FPL_DRAFT_TEAM_ID", 67890),
        patch("config.FPL_CLASSIC_TEAM_ID", 11111),
        patch("config.CURRENT_GAMEWEEK", 25),
    ]
    for cp in config_patches:
        active.append(cp)
        cp.start()

    yield mocks

    for p in active:
        p.stop()
