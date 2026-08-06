"""Tests for scripts/common/league_config.py."""

import json
from unittest.mock import patch

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
