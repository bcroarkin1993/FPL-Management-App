"""Tests for scripts/common/luck_analysis.py."""

import pandas as pd
import pytest

from scripts.common.luck_analysis import (
    extract_draft_gw_scores,
    extract_classic_h2h_gw_scores,
    calculate_all_play_standings,
)


class TestExtractDraftGwScores:
    def test_basic_extraction(self, mock_league_response):
        df = extract_draft_gw_scores(mock_league_response)
        assert not df.empty
        assert set(df.columns) == {"gameweek", "team", "score"}

    def test_skips_both_zero(self, mock_league_response):
        """Matches where BOTH teams scored 0 should be skipped (unplayed)."""
        df = extract_draft_gw_scores(mock_league_response)
        # GW1 match 3 vs 1 (both 0) should be skipped
        gw1_teams = df[df["gameweek"] == 1]["team"].tolist()
        assert "Team Gamma" not in gw1_teams

    def test_keeps_valid_zero(self, mock_league_response):
        """A match where one team scored 0 is valid and should be kept."""
        df = extract_draft_gw_scores(mock_league_response)
        # GW2: Team Alpha (45) vs Team Gamma (0) — should be kept
        gw2 = df[df["gameweek"] == 2]
        gamma_scores = gw2[gw2["team"] == "Team Gamma"]["score"].values
        assert len(gamma_scores) > 0
        assert 0 in gamma_scores  # At least one of Gamma's GW2 scores is 0

    def test_empty_matches(self):
        df = extract_draft_gw_scores({"matches": [], "league_entries": []})
        assert df.empty

    def test_team_name_mapping(self, mock_league_response):
        df = extract_draft_gw_scores(mock_league_response)
        teams = df["team"].unique()
        assert "Team Alpha" in teams
        assert "Team Beta" in teams


class TestExtractClassicH2hGwScores:
    def test_basic_extraction(self, mock_h2h_matches):
        df = extract_classic_h2h_gw_scores(mock_h2h_matches)
        assert not df.empty
        assert set(df.columns) == {"gameweek", "team", "team_id", "score"}

    def test_skips_unplayed(self, mock_h2h_matches):
        """Unplayed: finished=False AND both scores 0."""
        df = extract_classic_h2h_gw_scores(mock_h2h_matches)
        # GW2 B vs C (both 0, not finished) should be skipped
        gw2 = df[df["gameweek"] == 2]
        assert gw2.empty

    def test_keeps_finished_zero(self, mock_h2h_matches):
        """Finished match with one team scoring 0 is valid."""
        df = extract_classic_h2h_gw_scores(mock_h2h_matches)
        # GW1: Team A scored 0 against Team C, finished=True
        team_a_gw1 = df[(df["gameweek"] == 1) & (df["team"] == "Team A")]
        # Team A appears in both GW1 matches
        assert len(team_a_gw1) == 2  # once as entry_1, once as entry_2
        scores = team_a_gw1["score"].tolist()
        assert 0 in scores  # one of the scores is 0

    def test_empty_input(self):
        df = extract_classic_h2h_gw_scores([])
        assert df.empty


class TestCalculateAllPlayStandings:
    def test_three_team_scenario(self):
        """3 teams, 1 gameweek: check all-play wins/losses."""
        gw_scores = pd.DataFrame({
            "gameweek": [1, 1, 1],
            "team": ["Team A", "Team B", "Team C"],
            "score": [80, 60, 70],
        })
        result = calculate_all_play_standings(gw_scores)
        assert not result.empty
        # Team A (80) beats both -> 2W 0L
        team_a = result[result["Team"] == "Team A"].iloc[0]
        assert team_a["AP W"] == 2
        assert team_a["AP L"] == 0

        # Team B (60) loses to both -> 0W 2L
        team_b = result[result["Team"] == "Team B"].iloc[0]
        assert team_b["AP W"] == 0
        assert team_b["AP L"] == 2

        # Team C (70) beats B, loses to A -> 1W 1L
        team_c = result[result["Team"] == "Team C"].iloc[0]
        assert team_c["AP W"] == 1
        assert team_c["AP L"] == 1

    def test_tie_handling(self):
        """When two teams tie, each gets a draw."""
        gw_scores = pd.DataFrame({
            "gameweek": [1, 1],
            "team": ["Team A", "Team B"],
            "score": [50, 50],
        })
        result = calculate_all_play_standings(gw_scores)
        for _, row in result.iterrows():
            assert row["AP D"] == 1
            assert row["AP W"] == 0
            assert row["AP L"] == 0

    def test_empty_input(self):
        result = calculate_all_play_standings(pd.DataFrame())
        assert result.empty

    def test_fair_rank_ordering(self):
        """Higher AP Win% should get lower (better) Fair Rank."""
        gw_scores = pd.DataFrame({
            "gameweek": [1, 1, 1],
            "team": ["Best", "Middle", "Worst"],
            "score": [100, 50, 10],
        })
        result = calculate_all_play_standings(gw_scores)
        best = result[result["Team"] == "Best"].index[0]
        worst = result[result["Team"] == "Worst"].index[0]
        assert best < worst  # Fair Rank index: lower = better

    def test_with_actual_standings(self):
        """Verify Luck +/- column appears when actual_standings provided."""
        gw_scores = pd.DataFrame({
            "gameweek": [1, 1, 1],
            "team": ["A", "B", "C"],
            "score": [80, 60, 70],
        })
        actual = pd.DataFrame({
            "team": ["A", "B", "C"],
            "actual_rank": [2, 3, 1],
            "actual_pts": [10, 5, 15],
        })
        result = calculate_all_play_standings(gw_scores, actual_standings=actual)
        assert "Luck +/-" in result.columns
        assert "Actual Rank" in result.columns
