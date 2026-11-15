"""The board's contract: cacheable, non-mutating, and never silently empty.

`build_waiver_board` is what stops the Waiver Wire page and the Draft My Week
strip computing two different answers to "is there a claim worth making".
Three properties make that safe, and each one has cost this codebase before.
"""

import ast
import inspect
from unittest.mock import patch

import pandas as pd
import pytest

import scripts.draft.waiver_core as wc


def _board(**over):
    """A minimal board. Built directly rather than through the builder: these
    are properties of the board's own contract, not of the fetch pipeline."""
    avail = pd.DataFrame([
        {"Player": "A", "Team": "ARS", "Position": "M", "Transfer Score": 0.9},
        {"Player": "B", "Team": "CHE", "Position": "D", "Transfer Score": 0.4},
    ])
    roster = pd.DataFrame([
        {"Player": "C", "Team": "LIV", "Position": "M", "Keep Score": 0.2},
    ])
    kw = dict(
        available=avail, roster=roster, reference=pd.DataFrame(),
        projections=pd.DataFrame(), ownership={}, element_states={},
        transaction_window={}, waiver_order=[], waiver_context={},
        floor_context={}, depth_map={}, locked_count=0, league_id=1,
        team_id=2, entry_id=2, current_gw=10, lookahead=3, form_weeks=4,
    )
    kw.update(over)
    return wc.WaiverBoard(**kw)


class TestSearchingNeverMutatesTheBoard:
    """The page mutates `avail_all` as it renders, and the strip may hold the
    same cached object inside one rerun. A search that handed out its own
    frames would let one caller corrupt the other's."""

    def test_the_frames_handed_to_the_engine_are_copies(self):
        board = _board()
        seen = {}

        def _capture(avail, roster, **kw):
            seen["avail"], seen["roster"] = avail, roster
            avail["Transfer Score"] = -1.0          # a caller scribbling
            roster.drop(roster.index, inplace=True)
            return [], []

        with patch.object(wc, "_compute_transfer_suggestions", side_effect=_capture):
            wc.search_board(board, positions=["M", "D"])

        assert seen, "the engine was never reached"
        assert seen["avail"] is not board.available
        assert seen["roster"] is not board.roster
        assert list(board.available["Transfer Score"]) == [0.9, 0.4]
        assert len(board.roster) == 1


class TestABoardThatCouldNotBeBuilt:
    def test_a_failed_board_yields_no_suggestions(self):
        board = _board(status=wc.BOARD_BAD_SCHEMA)
        assert wc.search_board(board, positions=["M"]) == ([], [], 0)
        assert wc.best_claim(board) is None

    def test_a_failed_board_always_carries_its_reason(self):
        """`check_week_plan`'s rule one layer down: "we could not check" and
        "nothing to do" are opposite claims and must not render alike."""
        for status in (wc.BOARD_BAD_SCHEMA, wc.BOARD_POOL_FAILED,
                       wc.BOARD_NO_PROJECTIONS):
            assert status != wc.BOARD_OK
        board = _board(status=wc.BOARD_POOL_FAILED, notes=(("error", "why"),))
        assert not board.ok and board.notes

    def test_no_positions_is_not_a_search(self):
        assert wc.search_board(_board(), positions=[]) == ([], [], 0)

    def test_none_is_tolerated(self):
        assert wc.search_board(None, positions=["M"]) == ([], [], 0)


class TestBestClaimCannotBypassTheGuards:
    """The locked-player filter and the cannot-play exclusion live *inside*
    `_compute_transfer_suggestions` so that no caller can forget them. The
    obvious shortcut for a checklist -- take the top row of `available` by
    Transfer Score -- skips both, which is how a player another manager
    dropped an hour ago gets recommended as a claim."""

    def test_it_routes_through_search_board(self):
        src = inspect.getsource(wc.best_claim)
        assert "search_board(" in src, (
            "best_claim must go through search_board; reading board.available "
            "directly bypasses the locked and cannot-play filters")
        assert "nlargest" not in src and "sort_values" not in src

    def test_it_returns_the_engine_s_first_suggestion(self):
        board = _board()
        picks = [{"lane": wc.LANE_HOLD, "points_gain": 3.0, "add_player": "A"},
                 {"lane": wc.LANE_HOLD, "points_gain": 1.0, "add_player": "B"}]
        with patch.object(wc, "_compute_transfer_suggestions",
                          return_value=(picks, [])):
            assert wc.best_claim(board)["add_player"] == "A"

    def test_no_suggestion_is_none_not_an_exception(self):
        with patch.object(wc, "_compute_transfer_suggestions", return_value=([], [])):
            assert wc.best_claim(_board()) is None


class TestTheCacheKeyIsPrimitivesOnly:
    """No frame may enter a cache key -- the `my_week._assemble` rule."""

    def test_the_builder_is_cached(self):
        """Read from the source, not the object: `tests/conftest.py` replaces
        `st.cache_data` with a pass-through before any app module is imported,
        so nothing in this suite is actually cached and the decorated function
        has no `.clear`."""
        import pathlib
        tree = ast.parse((pathlib.Path(wc.__file__)).read_text())
        fn = next(n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "build_waiver_board")
        decs = [ast.unparse(d) for d in fn.decorator_list]
        assert any("cache_data" in d for d in decs), (
            "build_waiver_board must be wrapped by st.cache_data; found %s" % decs)
        assert any("ttl=" in d for d in decs), (
            "give it an explicit TTL -- an unbounded cache pins a stale board, "
            "and this one is read the hour before a waiver deadline")

    def test_every_parameter_is_a_primitive(self):
        fn = getattr(wc.build_waiver_board, "__wrapped__", wc.build_waiver_board)
        for name, p in inspect.signature(fn).parameters.items():
            assert not isinstance(p.default, (pd.DataFrame, pd.Series, dict, list)), (
                "%s defaults to a container; the cache would hash it" % name)

    def test_the_search_controls_are_not_board_fields(self):
        """They belong to `search_board`. In the key they would rebuild the
        whole board on every filter change, which is the cost this split
        exists to remove."""
        fields = set(wc.WaiverBoard.__dataclass_fields__)
        assert not fields & {"positions", "lanes", "compact", "limit",
                             "sugg_view", "sugg_hold"}


class TestTheLaneLimitIsPerLane:
    """A flat slice lets the hold lane fill the quota and the rental lane
    vanish -- the exact failure the two lanes exist to fix."""

    def test_both_lanes_survive_a_limit(self):
        picks = ([{"lane": wc.LANE_HOLD, "points_gain": i} for i in range(5)]
                 + [{"lane": wc.LANE_RENT, "points_gain": i} for i in range(5)])
        with patch.object(wc, "_compute_transfer_suggestions", return_value=(picks, [])):
            got, _, total = wc.search_board(_board(), positions=["M"], limit=2)
        assert total == 10
        assert sum(1 for x in got if x["lane"] == wc.LANE_HOLD) == 2
        assert sum(1 for x in got if x["lane"] == wc.LANE_RENT) == 2


class TestTheBuilderActuallyRuns:
    """The tests above construct a `WaiverBoard` directly and mock the engine,
    which says nothing about whether `build_waiver_board` itself executes.

    It did not. The spliced pipeline referred to the page's name for the notes
    sink and raised `NameError` on the first line that reported anything --
    and every test here passed, because none of them ran the function. It was
    a timing script that found it. So one test drives the real body with the
    fetch layer stubbed.
    """

    @staticmethod
    def _stub(monkeypatch, **over):
        import numpy as np
        stats = pd.DataFrame({
            "Player": ["Alpha One", "Beta Two", "Gamma Three"],
            "Web_Name": ["Alpha", "Beta", "Gamma"],
            "Team": ["ARS", "CHE", "LIV"],
            "Position": ["M", "D", "F"],
            "Player_ID": [1, 2, 3], "Player_Code": [101, 102, 103],
            "code": [101, 102, 103],
            "Season_Points": [10, 20, 30], "starts": [3, 4, 5],
            "points_per_game": [2.0, 3.0, 4.0], "form": [1.0, 2.0, 3.0],
            "status": ["a", "a", "a"], "news": ["", "", ""],
            "chance_of_playing_next_round": [100, 100, 100],
            "minutes": [270, 360, 450],
        })
        proj = pd.DataFrame({
            "Player": ["Alpha One", "Beta Two", "Gamma Three"],
            "Team": ["ARS", "CHE", "LIV"],
            "Position": ["M", "D", "F"],
            "Points": [5.0, 4.0, 6.0], "Pos Rank": [1, 1, 1],
            "Player_ID": [1, 2, 3], "Player_Code": [101, 102, 103],
        })
        ownership = {7: {"team_name": "Mine", "players": {"M": ["Alpha One"]}}}
        stubs = {
            "get_league_player_ownership": lambda *a, **k: ownership,
            "get_league_element_states": lambda *a, **k: {},
            "get_draft_transaction_window": lambda *a, **k: {"mode": "waivers"},
            "get_league_waiver_order": lambda *a, **k: [],
            "get_rotowire_player_projections": lambda *a, **k: proj.copy(),
            "pull_fpl_player_stats": lambda *a, **k: stats.copy(),
            "normalize_fpl_players_to_rotowire_schema": lambda df, **k: df,
            "normalize_rotowire_players": lambda df, **k: df,
            "_bootstrap_teams_df": lambda *a, **k: pd.DataFrame(
                {"short_name": ["ARS", "CHE", "LIV"], "name": ["Arsenal", "Chelsea", "Liverpool"], "id": [1, 2, 3]}),
            "get_ffp_feed": lambda *a, **k: None,
            "get_rotowire_season_rankings": lambda *a, **k: None,
            "compute_healthy_form": lambda *a, **k: 2.0,
            "_element_history_df": lambda *a, **k: None,
            "_avg_fdr_by_short_name": lambda *a, **k: {"ARS": 3.0, "CHE": 3.0, "LIV": 3.0},
            "_build_waiver_context": lambda *a, **k: {},
            "_build_floor_context": lambda *a, **k: {},
            "_fetch_all_transactions": lambda *a, **k: [],
            "_waiver_claim_stats": lambda *a, **k: {},
        }
        stubs.update(over)
        for name, fn in stubs.items():
            if hasattr(wc, name):
                monkeypatch.setattr(wc, name, fn)

    def test_it_builds_a_board_without_touching_the_network(self, monkeypatch):
        self._stub(monkeypatch)
        board = wc.build_waiver_board(1, 7, 10, entry_id=7)
        assert isinstance(board, wc.WaiverBoard)
        assert board.current_gw == 10
        assert board.team_id == 7 and board.entry_id == 7
        # it ran far enough to produce the pools and the contexts
        assert not board.reference.empty
        assert board.status in (wc.BOARD_OK, wc.BOARD_NO_PROJECTIONS,
                                wc.BOARD_POOL_FAILED, wc.BOARD_BAD_SCHEMA)

    def test_defaults_come_from_config_not_from_nothing(self, monkeypatch):
        self._stub(monkeypatch)
        import config
        board = wc.build_waiver_board(1, 7, 10, entry_id=7)
        assert board.lookahead == int(config.UPCOMING_WEEKS_DEFAULT)
        assert board.form_weeks == int(config.FORM_LOOKBACK_WEEKS)

    def test_no_projections_is_a_status_and_not_an_exception(self, monkeypatch):
        """The engine must not end a page it knows nothing about."""
        self._stub(monkeypatch,
                   get_rotowire_player_projections=lambda *a, **k: pd.DataFrame())
        board = wc.build_waiver_board(1, 7, 10, entry_id=7)
        assert board.status == wc.BOARD_NO_PROJECTIONS
        assert board.notes, "a terminal board must carry its reason"
        assert not board.ok

    def test_a_broken_schema_is_reported_not_raised(self, monkeypatch):
        self._stub(monkeypatch,
                   _enforce_rw_schema_fpl=lambda df, teams: pd.DataFrame({"x": [1]}))
        board = wc.build_waiver_board(1, 7, 10, entry_id=7)
        assert board.status == wc.BOARD_BAD_SCHEMA
        assert any(sev == "error" for sev, _ in board.notes)
