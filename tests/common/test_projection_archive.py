"""Tests for the projection snapshot archive.

Nothing in this app had ever been scored after the fact: the 60/40 blend was
assumed, not measured, because no projection was ever written down. These files
are the record that makes measurement possible, and they are committed by a
scheduled job -- so the tests care as much about *not* writing as about writing.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.common import projection_archive


@pytest.fixture(autouse=True)
def _isolated_archive(tmp_path, monkeypatch):
    """Never touch the real archive/projections/ -- it is irreplaceable."""
    d = tmp_path / "projections"
    d.mkdir()
    monkeypatch.setattr(projection_archive, "_archive_dir", lambda: d)
    return d


def _pre(n=3, proj=5.0):
    return pd.DataFrame({
        "player_id": list(range(1, n + 1)),
        "display_name": [f"Player {i}" for i in range(1, n + 1)],
        "team": ["MCI"] * n,
        "position": ["M"] * n,
        "proj": [proj] * n,
        "proj_start": [proj + 1] * n,
        "start_pct": [0.9] * n,
    })


def _actual(n=3, points=6):
    return pd.DataFrame({
        "player_id": list(range(1, n + 1)),
        "points": [points] * n,
        "minutes": [90] * n,
        "started": [1] * n,
    })


class TestRoundTrip:
    def test_pre_snapshot_round_trip(self):
        assert projection_archive.save_pre(3, _pre(), {"deadline": "2026-09-04T17:30:00Z"})
        df, meta = projection_archive.load_pre(3)
        assert len(df) == 3
        assert meta["gameweek"] == 3
        assert meta["deadline"] == "2026-09-04T17:30:00Z"

    def test_actuals_round_trip(self):
        assert projection_archive.save_actuals(2, _actual())
        df, meta = projection_archive.load_actuals(2)
        assert len(df) == 3
        assert meta["kind"] == "actual"

    def test_absent_gameweek_is_not_an_error(self):
        assert projection_archive.load_pre(99) == (None, {})
        assert projection_archive.load_actuals(99) == (None, {})

    def test_damaged_file_degrades_to_nothing(self, _isolated_archive):
        (_isolated_archive / "GW05_pre.json").write_text("{not json")
        assert projection_archive.load_pre(5) == (None, {})

    def test_empty_frame_is_not_written(self):
        assert not projection_archive.save_pre(3, pd.DataFrame())
        assert projection_archive.list_pre() == []


class TestUnchangedRunsWriteNothing:
    """These files are committed by an hourly job. A rewrite that changes
    nothing would produce an hourly commit and bury the real changes."""

    def test_identical_rows_are_not_rewritten(self, _isolated_archive):
        projection_archive.save_pre(3, _pre())
        before = (_isolated_archive / "GW03_pre.json").read_bytes()
        assert not projection_archive.save_pre(3, _pre())
        assert (_isolated_archive / "GW03_pre.json").read_bytes() == before

    def test_changed_rows_are_rewritten(self):
        projection_archive.save_pre(3, _pre(proj=5.0))
        assert projection_archive.save_pre(3, _pre(proj=6.0))
        df, _ = projection_archive.load_pre(3)
        assert df["proj"].iloc[0] == 6.0

    def test_capture_time_survives_a_rewrite(self):
        """captured_at marks when this gameweek was first seen. Resetting it on
        every update would make the record of *when* we projected meaningless."""
        projection_archive.save_pre(3, _pre(proj=5.0))
        _, first = projection_archive.load_pre(3)
        projection_archive.save_pre(3, _pre(proj=6.0))
        _, second = projection_archive.load_pre(3)
        assert second["captured_at"] == first["captured_at"]
        assert second["updated_at"] >= first["updated_at"]

    def test_serialisation_is_deterministic(self, _isolated_archive):
        """Row order must not depend on frame order, or every run looks changed."""
        projection_archive.save_pre(3, _pre())
        before = (_isolated_archive / "GW03_pre.json").read_bytes()
        shuffled = _pre().iloc[::-1].reset_index(drop=True)
        assert not projection_archive.save_pre(3, shuffled)
        assert (_isolated_archive / "GW03_pre.json").read_bytes() == before


class TestActualsAreWrittenOnce:
    def test_actuals_are_never_revised(self):
        """The caller waits for FPL's data_checked, so points are final. A second
        write would be identical or wrong."""
        projection_archive.save_actuals(2, _actual(points=6))
        assert not projection_archive.save_actuals(2, _actual(points=99))
        df, _ = projection_archive.load_actuals(2)
        assert df["points"].iloc[0] == 6


class TestScoreablePairs:
    def test_scoreable_needs_both_halves(self):
        projection_archive.save_pre(3, _pre())
        projection_archive.save_pre(4, _pre())
        projection_archive.save_actuals(3, _actual())
        assert projection_archive.list_pre() == [3, 4]
        assert projection_archive.list_actuals() == [3]
        assert projection_archive.scoreable_gameweeks() == [3]

    def test_nothing_scoreable_when_only_actuals_exist(self):
        projection_archive.save_actuals(1, _actual())
        assert projection_archive.scoreable_gameweeks() == []
