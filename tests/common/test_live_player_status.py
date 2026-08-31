"""Unit tests for live player/fixture status.

These cover the bug where a 0-minute player whose match had already finished was
rendered as "Upcoming" all week, and kept his full projection in the blended team
score. Minutes alone cannot tell an unused substitute apart from a player whose
match has not kicked off -- the fixture's own state is the missing signal.
"""

import json
from unittest.mock import MagicMock, patch

import pandas as pd

from scripts.common.fixture_helpers import live_player_status
from scripts.common.fpl_classic_api import (
    get_gw_finished_teams,
    get_gw_team_fixture_status,
)


def _mock_response(json_data):
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = 200
    resp.content = json.dumps(json_data).encode()
    resp.raise_for_status.return_value = None
    return resp


class TestLivePlayerStatus:
    def test_minutes_played(self):
        assert live_player_status(True, fixture_finished=True) == "played"
        assert live_player_status(True, fixture_finished=False, fixture_started=True) == "played"

    def test_zero_minutes_in_a_finished_match_is_dnp_not_upcoming(self):
        # Senesi: named in the XI, match against TOT over, never came on.
        assert live_player_status(False, fixture_finished=True, fixture_started=True) == "dnp"

    def test_zero_minutes_in_a_running_match_can_still_come_on(self):
        assert live_player_status(False, fixture_finished=False, fixture_started=True) == "live"

    def test_match_not_kicked_off(self):
        assert live_player_status(False, fixture_finished=False, fixture_started=False) == "upcoming"

    def test_defaults_to_upcoming_without_fixture_info(self):
        """Callers with no fixture state must keep the old, safe reading."""
        assert live_player_status(False) == "upcoming"


class TestGwTeamFixtureStatus:
    @patch("scripts.common.fpl_classic_api.requests.get")
    def test_finished_provisional_counts_as_over(self, mock_get):
        """The API leaves `finished` False for hours while bonus is confirmed."""
        mock_get.return_value = _mock_response([
            {"team_h": 19, "team_a": 17, "started": True,
             "finished": False, "finished_provisional": True},
        ])
        status = get_gw_team_fixture_status(2)
        assert status[19] == {"started": True, "finished": True}
        assert status[17] == {"started": True, "finished": True}

    @patch("scripts.common.fpl_classic_api.requests.get")
    def test_unstarted_fixture(self, mock_get):
        mock_get.return_value = _mock_response([
            {"team_h": 2, "team_a": 1, "started": False,
             "finished": False, "finished_provisional": False},
        ])
        status = get_gw_team_fixture_status(2)
        assert status[1] == {"started": False, "finished": False}

    @patch("scripts.common.fpl_classic_api.requests.get")
    def test_double_gameweek_needs_every_fixture_over(self, mock_get):
        mock_get.return_value = _mock_response([
            {"team_h": 5, "team_a": 6, "started": True,
             "finished": True, "finished_provisional": True},
            {"team_h": 7, "team_a": 5, "started": False,
             "finished": False, "finished_provisional": False},
        ])
        status = get_gw_team_fixture_status(2)
        # Team 5 plays twice; one match still to come, so it is not finished.
        assert status[5] == {"started": True, "finished": False}
        assert status[6] == {"started": True, "finished": True}

    @patch("scripts.common.fpl_classic_api.requests.get")
    def test_finished_teams_uses_the_same_reading(self, mock_get):
        mock_get.return_value = _mock_response([
            {"team_h": 19, "team_a": 17, "started": True,
             "finished": False, "finished_provisional": True},
            {"team_h": 2, "team_a": 1, "started": False,
             "finished": False, "finished_provisional": False},
        ])
        # Without finished_provisional this was empty, and auto-subs never fired.
        assert get_gw_finished_teams(2) == {19, 17}

    @patch("scripts.common.fpl_classic_api.requests.get", side_effect=Exception("boom"))
    def test_unreachable_api_is_not_fatal(self, mock_get):
        assert get_gw_team_fixture_status(2) == {}
        assert get_gw_finished_teams(2) == set()


class TestClassicBlend:
    def _squad(self):
        return pd.DataFrame([
            {"element_id": 106, "Player": "Igor Thiago", "Points": 4.9},
            {"element_id": 498, "Player": "Marcos Senesi Barón", "Points": 3.1},
            {"element_id": 400, "Player": "Yet To Play", "Points": 5.0},
        ])

    def _live(self):
        return {
            106: {"points": 2, "minutes": 90, "has_played": True,
                  "fixture_started": True, "fixture_finished": True},
            498: {"points": 0, "minutes": 0, "has_played": False,
                  "fixture_started": True, "fixture_finished": True},
            400: {"points": 0, "minutes": 0, "has_played": False,
                  "fixture_started": False, "fixture_finished": False},
        }

    def test_dnp_drops_its_projection_from_the_blend(self):
        from scripts.classic.fixture_projections import _blend_live_with_squad

        out = _blend_live_with_squad(self._squad(), self._live()).set_index("Player")
        # Played: actual points.
        assert out.at["Igor Thiago", "Blended_Points"] == 2
        # Match over, never came on: 0 points can no longer become 3.1.
        assert out.at["Marcos Senesi Barón", "Blended_Points"] == 0
        assert out.at["Marcos Senesi Barón", "Fixture_Finished"]
        assert not out.at["Marcos Senesi Barón", "Has_Played"]
        # Still to play: keep the projection.
        assert out.at["Yet To Play", "Blended_Points"] == 5.0


class TestDraftBlend:
    def _team_df(self):
        """Frame as the Draft page builds it: names come from Rotowire, not FPL."""
        df = pd.DataFrame([
            # Rotowire's short name never reaches the bootstrap's full legal name.
            {"Player": "Igor Thiago", "Team": "BRE", "Points": 4.9, "Player_ID": 106},
            {"Player": "Marcos Senesi Barón", "Team": "TOT", "Points": 3.1, "Player_ID": 498},
        ])
        return df.set_index("Player")

    def _player_mapping(self):
        return {
            106: {"Player": "Igor Thiago Nascimento Rodrigues", "Web_Name": "Thiago",
                  "Team": "BRE", "Position": "F"},
            498: {"Player": "Marcos Senesi Barón", "Web_Name": "Senesi",
                  "Team": "TOT", "Position": "D"},
        }

    def _live(self):
        return {
            106: {"points": 2, "minutes": 90, "has_played": True,
                  "fixture_started": True, "fixture_finished": True},
            498: {"points": 0, "minutes": 0, "has_played": False,
                  "fixture_started": True, "fixture_finished": True},
        }

    def test_element_id_beats_name_matching(self):
        from scripts.draft.fixture_projections import _blend_live_with_projections

        out = _blend_live_with_projections(self._team_df(), self._live(), self._player_mapping())
        # "Igor Thiago" matches no bootstrap name key; the carried id does.
        assert out.at["Igor Thiago", "Has_Played"]
        assert out.at["Igor Thiago", "Live_Points"] == 2
        assert out.at["Igor Thiago", "Blended_Points"] == 2

    def test_dnp_drops_its_projection_from_the_blend(self):
        from scripts.draft.fixture_projections import _blend_live_with_projections

        out = _blend_live_with_projections(self._team_df(), self._live(), self._player_mapping())
        assert not out.at["Marcos Senesi Barón", "Has_Played"]
        assert out.at["Marcos Senesi Barón", "Fixture_Finished"]
        assert out.at["Marcos Senesi Barón", "Blended_Points"] == 0

    def test_name_matching_still_works_without_ids(self):
        df = self._team_df().drop(columns=["Player_ID"])
        from scripts.draft.fixture_projections import _blend_live_with_projections

        out = _blend_live_with_projections(df, self._live(), self._player_mapping())
        assert out.at["Marcos Senesi Barón", "Fixture_Finished"]


class TestMergeCarriesElementIds:
    """The merge takes matched names from the projection source, so the FPL element
    id has to be carried explicitly or the live lookup is left guessing at names."""

    def _projections(self):
        return pd.DataFrame([
            {"Player": "Igor Thiago", "Team": "BRE", "Position": "F",
             "Matchup": "BRE at LEE", "Points": 4.88, "Pos Rank": 9},
        ])

    def test_player_id_survives_a_matched_row(self):
        from scripts.common.player_matching import merge_fpl_players_and_projections

        fpl = pd.DataFrame([
            {"Player": "Igor Thiago Nascimento Rodrigues", "Team": "BRE",
             "Position": "F", "Player_ID": 106},
        ])
        out = merge_fpl_players_and_projections(
            fpl, self._projections(), carry_cols=["Player_ID"]
        )
        # Name comes from Rotowire, id from FPL.
        assert out.iloc[0]["Player"] == "Igor Thiago"
        assert out.iloc[0]["Player_ID"] == 106

    def test_player_id_survives_an_unmatched_row(self):
        from scripts.common.player_matching import merge_fpl_players_and_projections

        fpl = pd.DataFrame([
            {"Player": "Marcos Senesi Barón", "Team": "TOT",
             "Position": "D", "Player_ID": 498},
        ])
        out = merge_fpl_players_and_projections(
            fpl, self._projections(), carry_cols=["Player_ID"]
        )
        assert out.iloc[0]["Player_ID"] == 498

    def test_absent_carry_column_is_ignored(self):
        from scripts.common.player_matching import merge_fpl_players_and_projections

        fpl = pd.DataFrame([
            {"Player": "Igor Thiago Nascimento Rodrigues", "Team": "BRE", "Position": "F"},
        ])
        out = merge_fpl_players_and_projections(
            fpl, self._projections(), carry_cols=["Player_ID"]
        )
        assert "Player_ID" not in out.columns
