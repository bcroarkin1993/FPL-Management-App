"""Tests for scripts/common/league_config.py."""

import json
from unittest.mock import patch

import config
from scripts.common.league_config import (
    load_settings,
    save_settings,
    DEFAULT_SETTINGS,
)


class TestLoadSettings:
    def test_missing_file_returns_defaults(self, tmp_path):
        """When config file doesn't exist, return defaults."""
        with patch("scripts.common.league_config._find_config_path", return_value=tmp_path / "nonexistent.json"):
            settings = load_settings()
        assert settings["version"] == DEFAULT_SETTINGS["version"]
        assert "draft" in settings
        assert "classic" in settings
        assert settings["draft"]["history"] == []
        assert settings["draft"]["commish_seasons"] == {}

    def test_legacy_file_without_new_keys_gets_defaults(self, tmp_path):
        """A settings file saved before the history/Commish Mode features
        existed should still deep-merge in their empty defaults, not KeyError."""
        config_path = tmp_path / "league_settings.json"
        config_path.write_text(json.dumps({
            "version": 1,
            "draft": {"league_id": 4544, "team_id": 17077, "team_name": "My Team", "locked": True},
        }))
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = load_settings()
        assert settings["draft"]["history"] == []
        assert settings["draft"]["commish_seasons"] == {}

    def test_valid_file(self, tmp_path):
        """When config file exists with partial data, merge with defaults."""
        config_path = tmp_path / "league_settings.json"
        config_path.write_text(json.dumps({
            "version": 1,
            "draft": {"league_id": 4544, "team_id": 17077, "team_name": "My Team", "locked": True},
        }))
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = load_settings()
        assert settings["draft"]["league_id"] == 4544
        assert settings["draft"]["locked"] is True
        # Default keys should still be present
        assert "classic" in settings
        assert settings["classic"]["leagues"] == []

    def test_corrupt_json_returns_defaults(self, tmp_path):
        """When config file has invalid JSON, return defaults."""
        config_path = tmp_path / "league_settings.json"
        config_path.write_text("{invalid json")
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = load_settings()
        assert settings == DEFAULT_SETTINGS


class TestSaveSettings:
    def test_round_trip(self, tmp_path):
        """Save and load should produce identical results."""
        config_path = tmp_path / "league_settings.json"
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = dict(DEFAULT_SETTINGS)
            settings["draft"] = {
                "league_id": 12345, "team_id": 999, "team_name": "Test Team", "locked": True,
            }
            assert save_settings(settings) is True

            loaded = load_settings()
            assert loaded["draft"]["league_id"] == 12345
            assert loaded["draft"]["locked"] is True

    def test_classic_leagues_round_trip(self, tmp_path):
        """Classic leagues list should persist unchanged through save/load."""
        config_path = tmp_path / "league_settings.json"
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = dict(DEFAULT_SETTINGS)
            settings["classic"] = {
                "leagues": [{"id": 1555691, "name": "FAFO FPL"}, {"id": 1161877, "name": "Starboys"}],
                "team_id": 6720205,
                "team_name": "My Classic Team",
                "locked": True,
            }
            save_settings(settings)

            loaded = load_settings()
            assert len(loaded["classic"]["leagues"]) == 2
            assert loaded["classic"]["leagues"][0]["name"] == "FAFO FPL"
            assert loaded["classic"]["locked"] is True

    def test_commish_seasons_round_trip(self, tmp_path):
        """Commish Mode dues/payout data should persist unchanged through save/load."""
        config_path = tmp_path / "league_settings.json"
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = dict(DEFAULT_SETTINGS)
            settings["draft"] = {
                "league_id": 11347, "team_id": 56086, "team_name": "Stoned Squirrels", "locked": True,
                "commish_seasons": {
                    "2026/27": {
                        "buy_in": 75,
                        "payout_pct": {"1": 60, "2": 30, "3": 10},
                        "locked": True,
                        "dues": {
                            "Stoned Squirrels": {"amount_paid": 75, "notes": ""},
                            "Top Drawer Balls": {"amount_paid": 0, "notes": "will pay at draft"},
                        },
                    },
                },
            }
            save_settings(settings)

            loaded = load_settings()
            season = loaded["draft"]["commish_seasons"]["2026/27"]
            assert season["buy_in"] == 75
            assert season["payout_pct"] == {"1": 60, "2": 30, "3": 10}
            assert season["dues"]["Stoned Squirrels"]["amount_paid"] == 75
            assert season["dues"]["Top Drawer Balls"]["notes"] == "will pay at draft"

    def test_draft_history_round_trip(self, tmp_path):
        """Draft history list should persist unchanged through save/load."""
        config_path = tmp_path / "league_settings.json"
        with patch("scripts.common.league_config._find_config_path", return_value=config_path):
            settings = dict(DEFAULT_SETTINGS)
            settings["draft"] = {
                "league_id": 4544, "team_id": 17077, "team_name": "Current Team", "locked": True,
                "history": [
                    {"season": "2024/25", "league_id": 111, "team_id": 1, "team_name": "Old Team"},
                    {"season": "2023/24", "league_id": 222, "team_id": 2, "team_name": "Older Team"},
                ],
            }
            save_settings(settings)

            loaded = load_settings()
            assert len(loaded["draft"]["history"]) == 2
            assert loaded["draft"]["history"][0]["season"] == "2024/25"


class TestResolveDraftLeagueHistory:
    """config._resolve_draft_league_history() merge logic (config.py, not league_config.py)."""

    def setup_method(self):
        config.refresh_league_settings()

    def teardown_method(self):
        config.refresh_league_settings()

    def test_env_only(self, monkeypatch):
        monkeypatch.setenv("FPL_DRAFT_LEAGUE_HISTORY", "2023/24:111,2024/25:222")
        with patch("config._get_league_settings", return_value={"draft": {"history": []}}):
            assert config.FPL_DRAFT_LEAGUE_HISTORY == [("2023/24", 111), ("2024/25", 222)]

    def test_json_only(self, monkeypatch):
        monkeypatch.delenv("FPL_DRAFT_LEAGUE_HISTORY", raising=False)
        settings = {"draft": {"history": [{"season": "2025/26", "league_id": 11347}]}}
        with patch("config._get_league_settings", return_value=settings):
            assert config.FPL_DRAFT_LEAGUE_HISTORY == [("2025/26", 11347)]

    def test_json_wins_on_season_collision(self, monkeypatch):
        monkeypatch.setenv("FPL_DRAFT_LEAGUE_HISTORY", "2024/25:999")
        settings = {"draft": {"history": [{"season": "2024/25", "league_id": 222}]}}
        with patch("config._get_league_settings", return_value=settings):
            assert config.FPL_DRAFT_LEAGUE_HISTORY == [("2024/25", 222)]

    def test_env_fills_seasons_missing_from_json(self, monkeypatch):
        monkeypatch.setenv("FPL_DRAFT_LEAGUE_HISTORY", "2023/24:111")
        settings = {"draft": {"history": [{"season": "2024/25", "league_id": 222}]}}
        with patch("config._get_league_settings", return_value=settings):
            assert config.FPL_DRAFT_LEAGUE_HISTORY == [("2023/24", 111), ("2024/25", 222)]

    def test_nothing_configured(self, monkeypatch):
        monkeypatch.delenv("FPL_DRAFT_LEAGUE_HISTORY", raising=False)
        with patch("config._get_league_settings", return_value={"draft": {"history": []}}):
            assert config.FPL_DRAFT_LEAGUE_HISTORY == []

    def test_manual_only_entry_excluded_from_tuple_form(self, monkeypatch):
        """A manual-stats entry with no league_id shouldn't appear in the
        lightweight tuple view (it has no league_id to expose)."""
        monkeypatch.delenv("FPL_DRAFT_LEAGUE_HISTORY", raising=False)
        settings = {"draft": {"history": [
            {"season": "2025/26", "league_id": None, "team_name": "Stoned Squirrels",
             "manual_stats": {"rank": 3, "total_points": 1500, "wins": 10, "draws": 2, "losses": 6}},
        ]}}
        with patch("config._get_league_settings", return_value=settings):
            assert config.FPL_DRAFT_LEAGUE_HISTORY == []


class TestGetDraftLeagueHistoryRecords:
    """config.get_draft_league_history_records() — full records, preserves manual_stats."""

    def setup_method(self):
        config.refresh_league_settings()

    def teardown_method(self):
        config.refresh_league_settings()

    def test_manual_stats_preserved(self, monkeypatch):
        monkeypatch.delenv("FPL_DRAFT_LEAGUE_HISTORY", raising=False)
        manual_stats = {"rank": 3, "total_points": 1500, "wins": 10, "draws": 2, "losses": 6}
        settings = {"draft": {"history": [
            {"season": "2025/26", "league_id": None, "team_id": None,
             "team_name": "Stoned Squirrels", "manual_stats": manual_stats},
        ]}}
        with patch("config._get_league_settings", return_value=settings):
            records = config.get_draft_league_history_records()
        assert len(records) == 1
        assert records[0]["manual_stats"] == manual_stats
        assert records[0]["league_id"] is None

    def test_env_derived_record_has_no_manual_stats(self, monkeypatch):
        monkeypatch.setenv("FPL_DRAFT_LEAGUE_HISTORY", "2023/24:111")
        with patch("config._get_league_settings", return_value={"draft": {"history": []}}):
            records = config.get_draft_league_history_records()
        assert records == [{
            "season": "2023/24", "league_id": 111, "team_id": None,
            "team_name": None, "manual_stats": None,
        }]

    def test_json_wins_on_season_collision(self, monkeypatch):
        monkeypatch.setenv("FPL_DRAFT_LEAGUE_HISTORY", "2024/25:999")
        settings = {"draft": {"history": [
            {"season": "2024/25", "league_id": None, "team_name": "My Team",
             "manual_stats": {"rank": 1, "total_points": 2000, "wins": 15, "draws": 0, "losses": 3}},
        ]}}
        with patch("config._get_league_settings", return_value=settings):
            records = config.get_draft_league_history_records()
        assert len(records) == 1
        assert records[0]["manual_stats"]["rank"] == 1

    def test_sorted_by_season(self, monkeypatch):
        monkeypatch.delenv("FPL_DRAFT_LEAGUE_HISTORY", raising=False)
        settings = {"draft": {"history": [
            {"season": "2024/25", "league_id": 222, "manual_stats": None},
            {"season": "2022/23", "league_id": 111, "manual_stats": None},
        ]}}
        with patch("config._get_league_settings", return_value=settings):
            records = config.get_draft_league_history_records()
        assert [r["season"] for r in records] == ["2022/23", "2024/25"]
