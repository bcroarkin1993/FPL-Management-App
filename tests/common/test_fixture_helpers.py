"""Unit tests for compute_key_differentials()."""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from scripts.common import fixture_helpers
from scripts.common.fixture_helpers import (
    compute_key_differentials,
    _bootstrap_teams_df,
    get_fixture_difficulty_grid,
)


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


class TestBootstrapTeamsDf:
    @patch("scripts.common.fpl_classic_api.get_classic_bootstrap_static")
    def test_returns_id_and_short_name(self, mock_bootstrap):
        mock_bootstrap.return_value = {
            "teams": [
                {"id": 1, "short_name": "ARS", "name": "Arsenal"},
                {"id": 2, "short_name": "AVL", "name": "Aston Villa"},
            ]
        }
        df = _bootstrap_teams_df()
        assert list(df["short_name"]) == ["ARS", "AVL"]

    @patch("scripts.common.fpl_classic_api.get_classic_bootstrap_static")
    def test_handles_unavailable_bootstrap(self, mock_bootstrap):
        """Regression: the Draft API's bootstrap-static goes offline (serves an
        HTML 'Game Updating' page) between seasons. If the Classic source we
        now use also fails, this must degrade to an empty DF, not raise."""
        mock_bootstrap.return_value = None
        df = _bootstrap_teams_df()
        assert df.empty
        assert list(df.columns) == ["id", "short_name"]


class TestGetFixtureDifficultyGrid:
    @patch("scripts.common.fixture_helpers._bootstrap_teams_df")
    @patch("scripts.common.fpl_draft_api.get_current_gameweek")
    @patch("scripts.common.fixture_helpers.requests.get")
    def test_unresolved_team_id_is_skipped_not_crashed(self, mock_get, mock_gw, mock_teams):
        """Regression for KeyError crash on the Gameweek Fixtures page: a fixture
        referencing a team id missing from the bootstrap map (e.g. incomplete/
        stale bootstrap data) must be skipped, not crash the whole grid."""
        mock_gw.return_value = 1
        mock_teams.return_value = pd.DataFrame({"id": [1, 2], "short_name": ["ARS", "AVL"]})

        fixture_resp = MagicMock()
        fixture_resp.raise_for_status.return_value = None
        # team_h=1 (known) vs team_a=99 (unknown -- not in the bootstrap map)
        fixture_resp.json.return_value = [
            {"team_h": 1, "team_a": 99, "team_h_difficulty": 2, "team_a_difficulty": 3}
        ]
        mock_get.return_value = fixture_resp

        disp, diffs, avg = get_fixture_difficulty_grid(weeks=1)
        assert "ARS" in disp.index
        assert "99" not in disp.index


class TestAttachMatchups:
    """`merge_fpl_players_and_projections` writes the literal string "N/A" as the
    Matchup for any player Rotowire did not list. Rotowire covers only the ~220
    players it expects to start, so once the blend started giving FFP-only
    players real projections they reached lineups and rendered "N/A" beside a
    perfectly good number.
    """

    MATCHUPS = {"MCI": "MCI at MUN", "BRE": "BRE at BOU", "CHE": "CHE v. HUL"}

    @pytest.fixture(autouse=True)
    def _stub_fixtures(self, monkeypatch):
        monkeypatch.setattr(fixture_helpers, "get_team_matchups",
                            lambda gw: dict(self.MATCHUPS))
        monkeypatch.setattr(fixture_helpers, "_all_clubs",
                            lambda gw: {"MCI", "BRE", "CHE", "EVE"})

    def _frame(self):
        return pd.DataFrame({
            "Player": ["O'Reilly", "Collins", "Haaland", "Pickford", "Ghost"],
            "Team": ["MCI", "BRE", "MCI", "EVE", "ZZZ"],
            "Matchup": ["N/A", "N/A", "MCI v. AVL", "N/A", "N/A"],
        })

    def test_placeholder_is_replaced_with_the_real_fixture(self):
        out = fixture_helpers.attach_matchups(self._frame(), 4)
        assert out.loc[0, "Matchup"] == "MCI at MUN"
        assert out.loc[1, "Matchup"] == "BRE at BOU"

    def test_a_stale_matchup_is_overridden(self):
        """A source's matchup is whatever gameweek that source was written for.
        Trusting it lets last week's opponent render under this week's heading --
        plausible, wrong, and undetectable by eye."""
        out = fixture_helpers.attach_matchups(self._frame(), 4)
        assert out.loc[2, "Matchup"] == "MCI at MUN"

    def test_a_club_with_no_fixture_says_so(self):
        """A blank gameweek is why the projection is zero. An empty cell reads
        as missing data instead."""
        out = fixture_helpers.attach_matchups(self._frame(), 4)
        assert out.loc[3, "Matchup"] == "No fixture"

    def test_an_unknown_club_is_left_blank_not_mislabelled(self):
        out = fixture_helpers.attach_matchups(self._frame(), 4)
        assert out.loc[4, "Matchup"] == ""

    def test_an_unreadable_fixture_list_leaves_the_frame_untouched(self, monkeypatch):
        """Cosmetic. This must never take a page down."""
        monkeypatch.setattr(fixture_helpers, "get_team_matchups", lambda gw: {})
        before = self._frame()
        out = fixture_helpers.attach_matchups(before, 4)
        pd.testing.assert_frame_equal(out, before)

    def test_a_frame_without_a_team_column_is_returned_unchanged(self):
        df = pd.DataFrame({"Player": ["X"]})
        pd.testing.assert_frame_equal(fixture_helpers.attach_matchups(df, 4), df)

    def test_missing_matchup_column_is_created(self):
        df = pd.DataFrame({"Player": ["O'Reilly"], "Team": ["MCI"]})
        out = fixture_helpers.attach_matchups(df, 4)
        assert out.loc[0, "Matchup"] == "MCI at MUN"


# =============================================================================
# BLANK AND DOUBLE GAMEWEEKS
# =============================================================================


class TestClubFixtureCounts:
    """Counts come from the fixture rows, so a double is two rows and a blank
    is an absent pair. Derived from a difficulty grid instead, a double's two
    FDRs are averaged into one cell and the count survives only inside the
    rendered string."""

    TEAMS = pd.DataFrame({"id": [1, 2, 3, 4], "short_name": ["AAA", "BBB", "CCC", "DDD"]})

    #: GW10 is ordinary. GW11 gives team 1 a double and leaves 4 blank.
    FIXTURES = pd.DataFrame([
        {"event": 10, "team_h": 1, "team_a": 2},
        {"event": 10, "team_h": 3, "team_a": 4},
        {"event": 11, "team_h": 1, "team_a": 2},
        {"event": 11, "team_h": 3, "team_a": 1},
    ])

    def _counts(self, gameweeks):
        with patch("scripts.common.fixture_helpers._bootstrap_teams_df",
                   return_value=self.TEAMS), \
             patch("scripts.common.fixture_helpers._all_fixtures_min",
                   return_value=self.FIXTURES):
            return fixture_helpers.club_fixture_counts(gameweeks)

    def test_an_ordinary_gameweek_is_one_each(self):
        counts = self._counts([10])
        assert list(counts[10]) == [1, 1, 1, 1]

    def test_a_double_counts_twice(self):
        counts = self._counts([11])
        assert counts.at[1, 11] == 2

    def test_a_blank_counts_zero(self):
        counts = self._counts([11])
        assert counts.at[4, 11] == 0

    def test_every_club_appears_even_when_it_is_not_playing(self):
        """A club absent from the frame cannot be reported as blanking."""
        counts = self._counts([11])
        assert set(counts.index) == {1, 2, 3, 4}

    def test_an_unreadable_fixture_list_gives_nothing_not_twenty_blanks(self):
        """All clubs showing 0 is what an outage looks like, and reporting it
        as a league-wide blank would zero every projection on the page."""
        with patch("scripts.common.fixture_helpers._bootstrap_teams_df",
                   return_value=self.TEAMS), \
             patch("scripts.common.fixture_helpers._all_fixtures_min",
                   return_value=pd.DataFrame()):
            assert fixture_helpers.club_fixture_counts([10]).empty

    def test_no_gameweeks_asked_for_is_empty(self):
        assert self._counts([]).empty


class TestBlankAndDoubleClubs:
    def test_it_names_only_the_interesting_gameweeks(self):
        counts = pd.DataFrame({10: [1, 1], 11: [2, 0]}, index=[1, 2])
        teams = pd.DataFrame({"id": [1, 2], "short_name": ["AAA", "BBB"]})
        found = fixture_helpers.blank_and_double_clubs(counts, teams)
        assert set(found) == {11}
        assert found[11] == {"blank": ["BBB"], "double": ["AAA"]}

    def test_an_all_ordinary_window_reports_nothing(self):
        """Today's real state, and it must read as "none" rather than empty."""
        counts = pd.DataFrame({10: [1, 1], 11: [1, 1]}, index=[1, 2])
        teams = pd.DataFrame({"id": [1, 2], "short_name": ["AAA", "BBB"]})
        assert fixture_helpers.blank_and_double_clubs(counts, teams) == {}

    def test_an_empty_frame_is_not_a_crash(self):
        assert fixture_helpers.blank_and_double_clubs(pd.DataFrame()) == {}
