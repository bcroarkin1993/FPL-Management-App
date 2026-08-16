"""Tests for scripts/common/wrapped_archive.py."""

from unittest.mock import patch

from scripts.common.wrapped_archive import (
    list_archived_seasons,
    load_archived_season,
    save_archived_season,
    _season_filename,
)


class TestSeasonFilename:
    def test_slash_replaced(self):
        assert _season_filename("2025/26") == "2025_26.json"

    def test_no_slash(self):
        assert _season_filename("2025-26") == "2025_26.json"


class TestArchiveRoundTrip:
    def test_save_and_load(self, tmp_path):
        with patch("scripts.common.wrapped_archive._archive_dir", return_value=tmp_path):
            assert save_archived_season("2025/26", {"standings": [{"team": "A"}]}) is True
            loaded = load_archived_season("2025/26")
        assert loaded["standings"] == [{"team": "A"}]
        assert loaded["_season"] == "2025/26"

    def test_load_missing_season_returns_none(self, tmp_path):
        with patch("scripts.common.wrapped_archive._archive_dir", return_value=tmp_path):
            assert load_archived_season("1999/00") is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        (tmp_path / "2025_26.json").write_text("{not valid json")
        with patch("scripts.common.wrapped_archive._archive_dir", return_value=tmp_path):
            assert load_archived_season("2025/26") is None

    def test_list_archived_seasons(self, tmp_path):
        with patch("scripts.common.wrapped_archive._archive_dir", return_value=tmp_path):
            save_archived_season("2024/25", {})
            save_archived_season("2025/26", {})
            seasons = list_archived_seasons()
        assert seasons == ["2024/25", "2025/26"]

    def test_list_ignores_corrupt_files(self, tmp_path):
        (tmp_path / "bad.json").write_text("{not valid")
        with patch("scripts.common.wrapped_archive._archive_dir", return_value=tmp_path):
            save_archived_season("2025/26", {})
            seasons = list_archived_seasons()
        assert seasons == ["2025/26"]
