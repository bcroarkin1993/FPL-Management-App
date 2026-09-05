"""Tests for the FFP archive.

FFP rolls its publication window forward and the old gameweek is gone. Measured
2026-09-05: the payload covered GW4-GW9 while the app was still scoring GW3, and
because FFP was persisted nowhere its GW3 numbers had ceased to exist. Nothing
raised -- FFP just silently stopped contributing partway through the gameweek.
Every test here is written against that failure.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.common import ffp_archive


@pytest.fixture(autouse=True)
def _isolated_archive(tmp_path, monkeypatch):
    """Never touch the real archive/ffp/ -- it holds irreplaceable data."""
    d = tmp_path / "ffp"
    d.mkdir()
    monkeypatch.setattr(ffp_archive, "_archive_dir", lambda: d)
    return d


def _table(n=3, points=5.0):
    return pd.DataFrame({
        "Name": [f"Player {i}" for i in range(n)],
        "Team": ["Arsenal"] * n,
        "Position": ["MID"] * n,
        "Start": [80] * n,
        "StartingPredicted": [points] * n,
        "Predicted": [points * 0.8] * n,
    })


NOW = datetime(2026, 9, 5, 3, 17, tzinfo=timezone.utc)


class TestRoundTrip:
    def test_save_then_load(self):
        assert ffp_archive.save_gameweek(_table(), 4, updated=NOW, window_gw=4)
        df, meta = ffp_archive.load_gameweek(4)
        assert len(df) == 3
        assert meta["gameweek"] == 4
        assert meta["window_gw"] == 4
        assert meta["provenance"] == ffp_archive.PROV_SITE

    def test_absent_gameweek_is_not_an_error(self):
        df, meta = ffp_archive.load_gameweek(99)
        assert df is None and meta == {}

    def test_damaged_file_degrades_to_no_archive(self, _isolated_archive):
        (_isolated_archive / "GW07.json").write_text("{not json")
        df, meta = ffp_archive.load_gameweek(7)
        assert df is None and meta == {}

    def test_empty_table_is_not_written(self):
        assert not ffp_archive.save_gameweek(pd.DataFrame(), 4, updated=NOW)
        assert ffp_archive.list_archived() == []

    def test_list_archived_is_sorted(self):
        for gw in (7, 4, 9):
            ffp_archive.save_gameweek(_table(), gw, updated=NOW)
        assert ffp_archive.list_archived() == [4, 7, 9]


class TestNewestPublicationWins:
    def test_later_publication_replaces_earlier(self):
        ffp_archive.save_gameweek(_table(points=4.0), 6, updated=NOW)
        ffp_archive.save_gameweek(_table(points=9.0), 6, updated=NOW + timedelta(days=1))
        df, _ = ffp_archive.load_gameweek(6)
        assert df["StartingPredicted"].iloc[0] == 9.0

    def test_earlier_publication_is_declined(self):
        """A GW6 forecast from a GW4 window is five weeks of team news less
        informed than one from a GW6 window. It must not overwrite it."""
        ffp_archive.save_gameweek(_table(points=9.0), 6, updated=NOW)
        assert not ffp_archive.save_gameweek(
            _table(points=4.0), 6, updated=NOW - timedelta(days=7))
        df, _ = ffp_archive.load_gameweek(6)
        assert df["StartingPredicted"].iloc[0] == 9.0

    def test_undated_entry_never_displaces_a_dated_one(self):
        """The spreadsheet publishes no revision time, so "no timestamp" means
        "vintage unknown" -- which must lose to a real publication."""
        ffp_archive.save_gameweek(_table(points=9.0), 3, updated=NOW)
        assert not ffp_archive.save_gameweek(
            _table(points=1.0), 3, updated=None,
            provenance=ffp_archive.PROV_SHEET_OFFSET)
        df, _ = ffp_archive.load_gameweek(3)
        assert df["StartingPredicted"].iloc[0] == 9.0

    def test_dated_entry_replaces_an_undated_one(self):
        ffp_archive.save_gameweek(_table(points=1.0), 3, updated=None,
                                  provenance=ffp_archive.PROV_SHEET_OFFSET)
        assert ffp_archive.save_gameweek(_table(points=9.0), 3, updated=NOW)
        df, meta = ffp_archive.load_gameweek(3)
        assert df["StartingPredicted"].iloc[0] == 9.0
        assert meta["provenance"] == ffp_archive.PROV_SITE

    def test_force_overrides_the_freshness_rule(self):
        ffp_archive.save_gameweek(_table(points=9.0), 3, updated=NOW)
        assert ffp_archive.save_gameweek(_table(points=1.0), 3, updated=None, force=True)


class TestArchiveWholeWindow:
    def test_every_gameweek_in_the_payload_is_stored(self):
        """Storing only the current slice is how GW3 was lost: the payload
        carries six gameweeks and the app renders one."""
        rows = []
        for gw in range(4, 10):
            for code in (1, 2, 3):
                rows.append({
                    "gw": gw, "player_code": code, "web_name": f"P{code}",
                    "first_name": "First", "second_name": f"Last{code}",
                    "element_type": 3, "team_name": "Arsenal", "is_home": True,
                    "opponent_abbr": "AVL", "predicted_points": 4.0,
                    "predicted_points_start": 3.2, "start_pct": 80.0,
                    "price": "6.0", "selected_by_percent": "10.0",
                    "fixture_count": 1,
                })
        written = ffp_archive.archive_payload(rows, window_gw=4, updated=NOW,
                                              code_to_id={1: 11, 2: 22, 3: 33})
        assert written == [4, 5, 6, 7, 8, 9]
        assert ffp_archive.list_archived() == [4, 5, 6, 7, 8, 9]

    def test_empty_payload_writes_nothing(self):
        assert ffp_archive.archive_payload([], window_gw=4, updated=NOW) == []


class TestSheetOffsetRecovery:
    """The spreadsheet's GW2..GW6 are the 2nd..6th gameweek of its window, so a
    GW2 sheet holds a GW3 forecast. Verified live: Next2GWsStart ==
    StartingPredicted + GW2 at MAE 0.029, against 0.45 for GW2 + GW3."""

    def _sheet(self):
        return pd.DataFrame({
            "Name": ["Bukayo Saka"],
            "Team": ["Arsenal"],
            "Position": ["MID"],
            "Start": [80],
            "StartingPredicted": [5.8],   # the window's own gameweek, GW2
            "Predicted": [4.6],
            "GW2": [5.6],                 # -> GW3
            "GW2s": [2.8],                # GW3, start-discounted
            "GW3": [5.7],                 # -> GW4
            "Next3GWs": [16.0],
        })

    def test_offset_one_reads_the_gw2_column(self):
        out = ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 3)
        assert out["StartingPredicted"].iloc[0] == pytest.approx(5.6)
        assert out["FFP_GW"].iloc[0] == 3

    def test_start_is_recovered_from_the_discounted_pair(self):
        """GWNs is that week already discounted, so the ratio gives that week's
        start rate -- not the window week's, which would be the wrong vintage."""
        out = ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 3)
        assert out["Start"].iloc[0] == pytest.approx(50.0, abs=1.0)

    def test_raw_offset_columns_are_dropped_not_shifted(self):
        """They describe the wrong weeks now. A silently mis-aligned multi-GW
        column is exactly the plausible-but-wrong value nothing downstream can
        catch."""
        out = ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 3)
        assert "GW2" not in out.columns
        assert "GW3" not in out.columns

    def test_multi_gw_totals_are_rebuilt_for_the_new_window(self):
        """Next3GWs is 40% of the ROS score. Dropping it would put every
        archived player on the `single_gw * 3` fallback silently."""
        out = ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 3)
        # NextN includes the gameweek itself: GW3 + GW4 == 5.6 + 5.7.
        assert out["Next2GWsStart"].iloc[0] == pytest.approx(5.6 + 5.7)

    def test_fixtures_are_repointed_at_the_recovered_gameweek(self, monkeypatch):
        """A table claiming GW3 while carrying GW2's fixtures cannot prove its
        own gameweek -- and resolve_ffp_gameweek() votes exactly that."""
        monkeypatch.setattr(ffp_archive, "_repoint_fixtures",
                            lambda df, gw: df.assign(Fixture="Chelsea (h)"))
        out = ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 3)
        assert out["Fixture"].iloc[0] == "Chelsea (h)"

    def test_gameweek_outside_the_window_returns_none(self):
        assert ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 20) is None

    def test_the_window_gameweek_itself_is_not_an_offset(self):
        assert ffp_archive.recover_from_sheet_offsets(self._sheet(), 2, 2) is None

    def test_empty_sheet_returns_none(self):
        assert ffp_archive.recover_from_sheet_offsets(pd.DataFrame(), 2, 3) is None
