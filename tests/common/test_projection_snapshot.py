"""Tests for the scheduled snapshot collector's decisions.

The collector's job is almost entirely *when* to write, not what. Each test
here pins one of those decisions, and each maps to a way the archive would
quietly become useless: a projection captured after the deadline is not a
projection, a provisional score is not an actual, and a snapshot taken for the
wrong gameweek is worse than no snapshot at all.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.common import projection_snapshot as snap


def _events(**overrides):
    base = [
        {"id": 1, "deadline_time": "2026-08-14T17:30:00Z", "finished": True,
         "data_checked": True, "is_current": False, "is_next": False},
        {"id": 2, "deadline_time": "2026-08-28T17:30:00Z", "finished": True,
         "data_checked": True, "is_current": False, "is_next": False},
        {"id": 3, "deadline_time": "2026-09-04T17:30:00Z", "finished": True,
         "data_checked": False, "is_current": True, "is_next": False},
        {"id": 4, "deadline_time": "2026-09-12T12:30:00Z", "finished": False,
         "data_checked": False, "is_current": False, "is_next": True},
    ]
    for ev in base:
        ev.update(overrides.get(ev["id"], {}))
    return {"events": base}


DEADLINE = "2026-09-12T12:30:00Z"


class TestPreDeadlineWindow:
    def test_inside_the_window(self):
        now = datetime(2026, 9, 12, 8, 0, tzinfo=timezone.utc)
        assert snap.within_pre_deadline_window(DEADLINE, now=now)

    def test_before_the_window_opens(self):
        now = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
        assert not snap.within_pre_deadline_window(DEADLINE, now=now)

    def test_the_deadline_itself_closes_the_window(self):
        """This is what freezes the file at the last state a manager could have
        acted on. One second later must not overwrite it."""
        assert not snap.within_pre_deadline_window(
            DEADLINE, now=datetime(2026, 9, 12, 12, 30, tzinfo=timezone.utc))
        assert not snap.within_pre_deadline_window(
            DEADLINE, now=datetime(2026, 9, 12, 18, 0, tzinfo=timezone.utc))

    def test_a_missing_or_unparseable_deadline_is_not_a_window(self):
        assert not snap.within_pre_deadline_window(None)
        assert not snap.within_pre_deadline_window("")
        assert not snap.within_pre_deadline_window("whenever")


class TestActualsGate:
    def test_finished_but_unchecked_gameweek_is_not_ready(self, monkeypatch):
        """finished goes true before bonus is confirmed. Storing a provisional
        score as an actual bakes a wrong number into the record permanently."""
        monkeypatch.setattr(snap.projection_archive, "list_actuals", lambda: [])
        assert 3 not in snap.pending_actuals(_events())

    def test_finished_and_checked_gameweeks_are_pending(self, monkeypatch):
        monkeypatch.setattr(snap.projection_archive, "list_actuals", lambda: [])
        assert snap.pending_actuals(_events()) == [1, 2]

    def test_already_stored_gameweeks_are_skipped(self, monkeypatch):
        monkeypatch.setattr(snap.projection_archive, "list_actuals", lambda: [1])
        assert snap.pending_actuals(_events()) == [2]

    def test_unfinished_gameweek_is_never_pending(self, monkeypatch):
        monkeypatch.setattr(snap.projection_archive, "list_actuals", lambda: [])
        assert 4 not in snap.pending_actuals(_events())


class TestGameweekSelection:
    def test_next_deadline_uses_the_is_next_flag(self):
        assert snap.next_deadline(_events()) == (4, DEADLINE)

    def test_preseason_falls_back_to_the_first_unfinished_gameweek(self):
        """Before the season starts nothing is flagged is_next."""
        evs = {"events": [
            {"id": 1, "deadline_time": "2026-08-14T17:30:00Z", "finished": False,
             "is_current": False, "is_next": False},
        ]}
        assert snap.next_deadline(evs) == (1, "2026-08-14T17:30:00Z")

    def test_deadline_for_returns_that_gameweeks_own_deadline(self):
        """A forced snapshot must record its own gameweek's deadline. Storing
        GW4's on a GW3 file is metadata nothing downstream can detect."""
        assert snap.deadline_for(_events(), 3) == "2026-09-04T17:30:00Z"
        assert snap.deadline_for(_events(), 4) == DEADLINE
        assert snap.deadline_for(_events(), 99) is None


class TestPool:
    def test_pool_carries_both_name_forms(self):
        bootstrap = {
            "teams": [{"id": 1, "short_name": "MUN"}],
            "elements": [{
                "id": 7, "first_name": "Bruno", "second_name": "Borges Fernandes",
                "web_name": "B.Fernandes", "team": 1, "element_type": 3,
                "ep_next": "5.2", "status": "a", "chance_of_playing_next_round": None,
            }],
        }
        pool = snap.build_pool(bootstrap)
        row = pool.iloc[0]
        # Match on the legal name, display the common one -- swapping them
        # silently degrades match rates.
        assert row["Player"] == "Bruno Borges Fernandes"
        assert row["Display_Name"] == "Bruno Fernandes"
        assert row["Position"] == "M"
        assert row["Team"] == "MUN"
        assert row["ep_next"] == pytest.approx(5.2)


class TestTheSnapshotRecordsWhatAReplayNeeds:
    """A snapshot is only evidence if the engine can reproduce it.

    Three of the start model's inputs are neither projections nor recoverable
    from the rest of the file, and before they were archived the accuracy
    harness refused every snapshot this engine wrote: the constants (revision 4
    made `start_pct__rotowire` per-player, so they can no longer be read back
    out of it), the player start counts that per-player floor reads, and FPL's
    availability ceiling.
    """

    @staticmethod
    def _bootstrap():
        return {
            "teams": [{"id": 1, "short_name": "ARS"}],
            "elements": [{
                "id": 1, "code": 111, "first_name": "Bukayo", "second_name": "Saka",
                "web_name": "Saka", "team": 1, "element_type": 3, "ep_next": "6.1",
                "status": "d", "chance_of_playing_next_round": 75, "starts": 7,
                "news": "Knock - 75% chance of playing",
            }],
            "events": [],
        }

    @pytest.fixture
    def _captured(self, monkeypatch):
        """`collect_pre` with every fetch stubbed, returning what it tried to save."""
        import pandas as pd
        from scripts.common import projection_archive

        saved = {}

        def _save(gameweek, frame, meta):
            saved["gameweek"], saved["frame"], saved["meta"] = gameweek, frame, meta
            return True

        monkeypatch.setattr(projection_archive, "save_pre", _save)
        monkeypatch.setattr(snap, "_rotowire_url", lambda gw: None)
        blank = type("S", (), {"name": "rw", "ok": False, "df": pd.DataFrame(),
                               "basis": "conditional", "gameweek": 4,
                               "updated": None, "note": "stub"})()
        monkeypatch.setattr(snap, "rotowire_source", lambda *a, **k: blank)
        monkeypatch.setattr(snap, "fetch_ffp_table",
                            lambda **k: (pd.DataFrame(), 4, None, "none", "stub"))
        monkeypatch.setattr(snap, "ffp_source", lambda *a, **k: blank)
        monkeypatch.setattr(snap, "fpl_ep_source", lambda **k: blank)
        monkeypatch.setattr(snap.rotowire_lineups, "fetch_lineup_xi", lambda **k: None)
        monkeypatch.setattr(snap.sportsgambler_lineups, "fetch_lineup_xi",
                            lambda **k: None)
        monkeypatch.setattr(snap, "build_projections", lambda *a, **k: pd.DataFrame(
            {"Proj": [4.0], "Proj_Start": [5.0], "Start_Pct": [0.8]},
            index=pd.Index([1], name="Player_ID")))
        assert snap.collect_pre(4, self._bootstrap(), deadline=None,
                                before_deadline=True)
        return saved

    def test_the_engine_inputs_are_archived_beside_the_projections(self, _captured):
        row = _captured["frame"].iloc[0]
        assert row["starts"] == 7
        assert row["status"] == "d"
        assert row["chance_of_playing"] == 75

    def test_the_constants_that_built_it_are_recorded(self, _captured):
        from scripts.common import projection_engine

        stated = _captured["meta"]["start_model_constants"]
        assert stated == projection_engine.start_model_constants()
        # The revision alone is not enough: a *constant* retune leaves the
        # revision untouched, and a replay judged against today's values would
        # then read as the archive and the engine having parted company.
        assert stated["revision"] == projection_engine.START_MODEL_REVISION
        assert set(stated) >= {"start_floors", "omitted_starts",
                               "omitted_history_weight", "omission_share"}


class TestRunSkipsWorkItDoesNotNeedToDo:
    def test_outside_the_window_no_pre_snapshot_is_taken(self, monkeypatch):
        calls = []
        monkeypatch.setattr(snap, "_get_json", lambda url, timeout=30: _events())
        monkeypatch.setattr(snap, "collect_pre",
                            lambda *a, **k: calls.append("pre") or True)
        monkeypatch.setattr(snap, "within_pre_deadline_window", lambda *a, **k: False)
        monkeypatch.setattr(snap.projection_archive, "list_actuals", lambda: [1, 2])
        monkeypatch.setattr(snap.projection_archive, "list_pre", lambda: [])
        monkeypatch.setattr(snap.projection_archive, "scoreable_gameweeks", lambda: [])
        assert snap.run() == 0
        assert calls == []

    def test_an_unreachable_bootstrap_writes_nothing(self, monkeypatch):
        def _boom(url, timeout=30):
            raise RuntimeError("network down")
        monkeypatch.setattr(snap, "_get_json", _boom)
        assert snap.run() == 0


class TestFfpArchiveIsKeptWarm:
    """FFP publishes a six-gameweek window and rolls it forward as soon as a
    gameweek kicks off, so a week only exists on their site for a limited time.
    Refreshing on every run -- not just inside the deadline window -- means one
    failed pre-deadline run cannot lose a gameweek the way GW3 was lost."""

    def test_refreshed_even_outside_the_deadline_window(self, monkeypatch):
        calls = []
        monkeypatch.setattr(snap, "_get_json", lambda url, timeout=30: _events())
        monkeypatch.setattr(snap, "within_pre_deadline_window", lambda *a, **k: False)
        monkeypatch.setattr(snap, "refresh_ffp_archive",
                            lambda gw=None: calls.append(gw))
        monkeypatch.setattr(snap.projection_archive, "list_actuals", lambda: [1, 2])
        monkeypatch.setattr(snap.projection_archive, "list_pre", lambda: [])
        monkeypatch.setattr(snap.projection_archive, "scoreable_gameweeks", lambda: [])
        snap.run()
        assert calls == [4]

    def test_a_failing_refresh_never_fails_the_run(self, monkeypatch):
        """The run's real job may be the actuals; keeping the archive warm is
        opportunistic and must not take it down."""
        def _boom(gameweek=None, **kw):
            raise RuntimeError("FFP unreachable")
        monkeypatch.setattr(snap, "fetch_ffp_table", _boom)
        snap.refresh_ffp_archive(4)      # must not raise
