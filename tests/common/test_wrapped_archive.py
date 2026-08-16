"""Tests for scripts/common/wrapped_archive.py."""

from unittest.mock import patch

from scripts.common.wrapped_archive import (
    list_archived_seasons,
    load_archived_season,
    save_archived_season,
    list_archived_team_wrapped_seasons,
    list_archived_teams_for_season,
    load_archived_team_season,
    save_archived_team_season,
    _season_filename,
    _team_filename,
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


class TestTeamFilename:
    def test_spaces_and_apostrophes_replaced(self):
        assert _team_filename("Stoned Squirrels") == "Stoned_Squirrels.json"
        assert _team_filename("RIP Gary’s Boys") == "RIP_Gary_s_Boys.json"


class TestTeamArchiveRoundTrip:
    def test_save_and_load(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path):
            assert save_archived_team_season("2025/26", "Alpha", {"overview": {"team": "Alpha"}}) is True
            loaded = load_archived_team_season("2025/26", "Alpha")
        assert loaded["overview"]["team"] == "Alpha"
        assert loaded["_season"] == "2025/26"
        assert loaded["_team"] == "Alpha"

    def test_load_missing_returns_none(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path):
            assert load_archived_team_season("2025/26", "NoSuchTeam") is None

    def test_load_corrupt_file_returns_none(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path):
            season_dir = tmp_path / "2025_26"
            season_dir.mkdir(parents=True)
            (season_dir / "Alpha.json").write_text("{not valid json")
            assert load_archived_team_season("2025/26", "Alpha") is None

    def test_list_archived_team_wrapped_seasons(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path):
            save_archived_team_season("2024/25", "Alpha", {})
            save_archived_team_season("2025/26", "Alpha", {})
            save_archived_team_season("2025/26", "Beta", {})
            assert list_archived_team_wrapped_seasons() == ["2024/25", "2025/26"]

    def test_list_archived_team_wrapped_seasons_no_archive_dir(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path / "missing"):
            assert list_archived_team_wrapped_seasons() == []

    def test_list_archived_teams_for_season(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path):
            save_archived_team_season("2025/26", "Alpha", {})
            save_archived_team_season("2025/26", "Beta", {})
            save_archived_team_season("2024/25", "Gamma", {})
            assert list_archived_teams_for_season("2025/26") == ["Alpha", "Beta"]
            assert list_archived_teams_for_season("2024/25") == ["Gamma"]

    def test_list_archived_teams_for_season_missing(self, tmp_path):
        with patch("scripts.common.wrapped_archive._team_archive_root", return_value=tmp_path):
            assert list_archived_teams_for_season("1999/00") == []
