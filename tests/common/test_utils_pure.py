"""Tests for pure functions in scripts/common/utils.py."""

import pandas as pd
import pytest

from scripts.common.utils import (
    normalize_name,
    normalize_apostrophes,
    _map_position_to_rw,
    format_team_name,
    check_valid_lineup,
    find_optimal_lineup,
    apply_availability_penalty,
    clean_text,
    _clean_player_name,
    _norm_text,
    _strip_accents,
    position_converter,
)


class TestNormalizeName:
    def test_basic(self):
        assert normalize_name("Bruno Fernandes") == "Bruno Fernandes"

    def test_accents(self):
        result = normalize_name("Raúl Jiménez")
        assert "ú" not in result
        assert "é" not in result

    def test_none_returns_empty(self):
        assert normalize_name(None) == ""

    def test_whitespace(self):
        result = normalize_name("  Bruno   Fernandes  ")
        assert "  " not in result


class TestNormalizeApostrophes:
    def test_curly_apostrophe(self):
        result = normalize_apostrophes("N\u2019Golo")
        assert result == "n'golo"

    def test_none_returns_none(self):
        assert normalize_apostrophes(None) is None

    def test_lowercase(self):
        result = normalize_apostrophes("HELLO World")
        assert result == "hello world"


class TestMapPositionToRw:
    def test_standard_mappings(self):
        assert _map_position_to_rw("GK") == "G"
        assert _map_position_to_rw("DEF") == "D"
        assert _map_position_to_rw("MID") == "M"
        assert _map_position_to_rw("FWD") == "F"

    def test_numeric_mappings(self):
        assert _map_position_to_rw("1") == "G"
        assert _map_position_to_rw("2") == "D"
        assert _map_position_to_rw("3") == "M"
        assert _map_position_to_rw("4") == "F"

    def test_already_short(self):
        assert _map_position_to_rw("G") == "G"
        assert _map_position_to_rw("D") == "D"
        assert _map_position_to_rw("M") == "M"
        assert _map_position_to_rw("F") == "F"

    def test_nan_returns_empty(self):
        assert _map_position_to_rw(float("nan")) == ""


class TestFormatTeamName:
    def test_basic(self):
        assert format_team_name("manchester united") == "Manchester United"

    def test_none_returns_none(self):
        assert format_team_name(None) is None

    def test_curly_apostrophe(self):
        result = format_team_name("nott\u2019m forest")
        assert "'" in result  # Should normalize to straight apostrophe


class TestCleanText:
    def test_basic(self):
        assert clean_text("  hello   world  ") == "hello world"

    def test_none(self):
        assert clean_text(None) == ""


class TestStripAccents:
    def test_basic(self):
        assert _strip_accents("Café") == "Cafe"

    def test_nan(self):
        assert _strip_accents(float("nan")) == ""


class TestPositionConverter:
    def test_all_positions(self):
        assert position_converter(1) == "G"
        assert position_converter(2) == "D"
        assert position_converter(3) == "M"
        assert position_converter(4) == "F"

    def test_unknown(self):
        assert position_converter(99) == "Unknown"


class TestCheckValidLineup:
    def test_valid_442(self):
        data = (
            [{"position": "G"}] * 1 +
            [{"position": "D"}] * 4 +
            [{"position": "M"}] * 4 +
            [{"position": "F"}] * 2
        )
        df = pd.DataFrame(data)
        assert check_valid_lineup(df) == True

    def test_valid_352(self):
        data = (
            [{"position": "G"}] * 1 +
            [{"position": "D"}] * 3 +
            [{"position": "M"}] * 5 +
            [{"position": "F"}] * 2
        )
        df = pd.DataFrame(data)
        assert check_valid_lineup(df) == True

    def test_invalid_too_few_players(self):
        """Only 10 players (need 11)."""
        data = (
            [{"position": "G"}] * 1 +
            [{"position": "D"}] * 3 +
            [{"position": "M"}] * 4 +
            [{"position": "F"}] * 2
        )
        df = pd.DataFrame(data)  # 10 players
        assert check_valid_lineup(df) == False

    def test_invalid_too_many_fwd(self):
        data = (
            [{"position": "G"}] * 1 +
            [{"position": "D"}] * 3 +
            [{"position": "M"}] * 3 +
            [{"position": "F"}] * 4  # Max is 3
        )
        df = pd.DataFrame(data)
        assert check_valid_lineup(df) == False


class TestFindOptimalLineup:
    def test_returns_11_players(self):
        players = []
        for i, (pos, pts) in enumerate([
            ("G", 5), ("G", 3),
            ("D", 8), ("D", 7), ("D", 6), ("D", 5), ("D", 4),
            ("M", 9), ("M", 8), ("M", 7), ("M", 6), ("M", 5),
            ("F", 10), ("F", 7), ("F", 3),
        ]):
            players.append({"Player": f"Player_{i}", "Position": pos, "Points": pts})
        df = pd.DataFrame(players)
        result = find_optimal_lineup(df)
        assert len(result) == 11

    def test_includes_best_players(self):
        players = []
        for i, (pos, pts) in enumerate([
            ("G", 5), ("G", 3),
            ("D", 8), ("D", 7), ("D", 6), ("D", 5), ("D", 4),
            ("M", 9), ("M", 8), ("M", 7), ("M", 6), ("M", 5),
            ("F", 10), ("F", 7), ("F", 3),
        ]):
            players.append({"Player": f"Player_{i}", "Position": pos, "Points": pts})
        df = pd.DataFrame(players)
        result = find_optimal_lineup(df)
        selected = result["Player"].tolist()
        # Best FWD (10 pts) should be included
        assert "Player_12" in selected
        # Best MID (9 pts) should be included
        assert "Player_7" in selected


class TestApplyAvailabilityPenalty:
    def test_basic_penalty(self):
        df = pd.DataFrame({
            "Player": ["A", "B"],
            "Score": [10.0, 8.0],
            "PlayPct": [100.0, 50.0],
        })
        result = apply_availability_penalty(df, "Score", "AdjScore")
        assert result.loc[0, "AdjScore"] == 10.0  # 100% play
        assert result.loc[1, "AdjScore"] == 4.0    # 50% play

    def test_zero_availability(self):
        df = pd.DataFrame({
            "Player": ["A"],
            "Score": [10.0],
            "PlayPct": [0.0],
        })
        result = apply_availability_penalty(df, "Score", "AdjScore")
        assert result.loc[0, "AdjScore"] == 0.0
