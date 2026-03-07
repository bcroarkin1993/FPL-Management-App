"""Unit tests for compute_key_differentials()."""

import pandas as pd
import pytest
from scripts.common.fixture_helpers import compute_key_differentials


def _make_classic_df(players: list[dict]) -> pd.DataFrame:
    """Build a Classic-style DataFrame (Player as column)."""
    return pd.DataFrame(players)


def _make_draft_df(players: list[dict]) -> pd.DataFrame:
    """Build a Draft-style DataFrame (Player as index)."""
    df = pd.DataFrame(players)
    df.set_index("Player", inplace=True)
    return df


class TestComputeKeyDifferentials:
    def test_shared_players_excluded(self):
        """Shared EPL players should not appear as differentials."""
        team1 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": "AVL (H)"},
            {"Player": "Haaland", "Team": "MCI", "Position": "F", "Points": 10.0, "Matchup": "CHE (A)"},
            {"Player": "Saka", "Team": "ARS", "Position": "M", "Points": 6.0, "Matchup": "BRE (H)"},
        ])
        team2 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": "AVL (H)"},
            {"Player": "Palmer", "Team": "CHE", "Position": "M", "Points": 7.0, "Matchup": "MCI (H)"},
            {"Player": "Watkins", "Team": "AVL", "Position": "F", "Points": 5.0, "Matchup": "LIV (A)"},
        ])
        d1, d2 = compute_key_differentials(team1, team2, "Team A", "Team B")
        # Salah shared — excluded from both
        d1_names = [d["player"] for d in d1]
        d2_names = [d["player"] for d in d2]
        assert "Salah" not in d1_names
        assert "Salah" not in d2_names
        assert "Haaland" in d1_names
        assert "Palmer" in d2_names

    def test_all_unique(self):
        """When no players are shared, all appear as differentials."""
        team1 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": ""},
            {"Player": "Haaland", "Team": "MCI", "Position": "F", "Points": 10.0, "Matchup": ""},
        ])
        team2 = _make_classic_df([
            {"Player": "Palmer", "Team": "CHE", "Position": "M", "Points": 7.0, "Matchup": ""},
            {"Player": "Watkins", "Team": "AVL", "Position": "F", "Points": 5.0, "Matchup": ""},
        ])
        d1, d2 = compute_key_differentials(team1, team2, "Team A", "Team B")
        assert len(d1) == 2
        assert len(d2) == 2

    def test_sorted_by_points_desc(self):
        """Differentials should be sorted by projected points descending."""
        team1 = _make_classic_df([
            {"Player": "Low", "Team": "LIV", "Position": "D", "Points": 2.0, "Matchup": ""},
            {"Player": "Mid", "Team": "MCI", "Position": "M", "Points": 5.0, "Matchup": ""},
            {"Player": "High", "Team": "ARS", "Position": "F", "Points": 9.0, "Matchup": ""},
        ])
        team2 = _make_classic_df([
            {"Player": "Other", "Team": "CHE", "Position": "M", "Points": 6.0, "Matchup": ""},
        ])
        d1, _ = compute_key_differentials(team1, team2, "A", "B")
        pts = [d["points"] for d in d1]
        assert pts == sorted(pts, reverse=True)

    def test_empty_squads(self):
        """Empty DataFrames should return empty lists."""
        empty = pd.DataFrame()
        team = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": ""},
        ])
        d1, d2 = compute_key_differentials(empty, team, "A", "B")
        assert d1 == []
        assert d2 == []

    def test_draft_index_format(self):
        """Should work with Draft-style DataFrames (Player as index)."""
        team1 = _make_draft_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": "AVL (H)"},
            {"Player": "Haaland", "Team": "MCI", "Position": "F", "Points": 10.0, "Matchup": "CHE (A)"},
        ])
        team2 = _make_draft_df([
            {"Player": "Palmer", "Team": "CHE", "Position": "M", "Points": 7.0, "Matchup": "MCI (H)"},
            {"Player": "Watkins", "Team": "AVL", "Position": "F", "Points": 5.0, "Matchup": "LIV (A)"},
        ])
        d1, d2 = compute_key_differentials(team1, team2, "A", "B")
        assert len(d1) == 2
        assert len(d2) == 2
        assert d1[0]["player"] == "Haaland"  # Higher points first

    def test_classic_column_format(self):
        """Should work with Classic-style DataFrames (Player as column)."""
        team1 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": ""},
        ])
        team2 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": ""},
        ])
        d1, d2 = compute_key_differentials(team1, team2, "A", "B")
        # Same player on both — no differentials
        assert d1 == []
        assert d2 == []

    def test_uses_custom_points_col(self):
        """Should use the specified points column when available."""
        team1 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Blended_Points": 12.0, "Matchup": ""},
        ])
        team2 = _make_classic_df([
            {"Player": "Palmer", "Team": "CHE", "Position": "M", "Points": 7.0, "Blended_Points": 9.0, "Matchup": ""},
        ])
        d1, d2 = compute_key_differentials(team1, team2, "A", "B", points_col="Blended_Points")
        assert d1[0]["points"] == 12.0
        assert d2[0]["points"] == 9.0

    def test_falls_back_to_points_if_custom_col_missing(self):
        """Should fall back to 'Points' if custom column doesn't exist."""
        team1 = _make_classic_df([
            {"Player": "Salah", "Team": "LIV", "Position": "M", "Points": 8.0, "Matchup": ""},
        ])
        team2 = _make_classic_df([
            {"Player": "Palmer", "Team": "CHE", "Position": "M", "Points": 7.0, "Matchup": ""},
        ])
        d1, d2 = compute_key_differentials(team1, team2, "A", "B", points_col="Blended_Points")
        assert d1[0]["points"] == 8.0
        assert d2[0]["points"] == 7.0
