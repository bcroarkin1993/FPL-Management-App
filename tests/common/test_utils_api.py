"""Integration tests for API wrapper functions in scripts/common/utils.py.

All HTTP calls are mocked — no real network traffic.
"""

import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from scripts.common.utils import (
    get_fpl_player_mapping,
    get_current_gameweek,
    get_classic_bootstrap_static,
    get_league_entries,
    get_league_player_ownership,
    get_draft_league_details,
    get_classic_league_standings,
    get_classic_team_history,
    get_rotowire_player_projections,
    is_draft_league_reachable,
)


def _mock_response(json_data, status_code=200):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.status_code = status_code
    resp.content = json.dumps(json_data).encode()
    resp.raise_for_status.return_value = None
    return resp


class TestGetFplPlayerMapping:
    @patch("scripts.common.utils.requests.get")
    def test_returns_player_dict(self, mock_get, mock_bootstrap_data):
        mock_get.return_value = _mock_response(mock_bootstrap_data)
        result = get_fpl_player_mapping()
        assert isinstance(result, dict)
        assert len(result) == 6
        # Check player 4 (Haaland)
        assert result[4]["Player"] == "Erling Haaland"
        assert result[4]["Team"] == "MCI"
        assert result[4]["Position"] == "F"

    @patch("scripts.common.utils.requests.get")
    def test_handles_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_fpl_player_mapping()
        assert result == {}


class TestGetCurrentGameweek:
    @patch("scripts.common.utils.requests.get")
    def test_current_event_not_finished(self, mock_get):
        mock_get.return_value = _mock_response({
            "current_event": 25,
            "current_event_finished": False,
            "next_event": 26,
        })
        result = get_current_gameweek()
        assert result == 25

    @patch("scripts.common.utils.requests.get")
    def test_current_event_finished(self, mock_get):
        mock_get.return_value = _mock_response({
            "current_event": 25,
            "current_event_finished": True,
            "next_event": 26,
        })
        result = get_current_gameweek()
        assert result == 26

    @patch("scripts.common.utils.requests.get")
    def test_handles_error_falls_back(self, mock_get):
        mock_get.side_effect = Exception("timeout")
        # Should fall back to config.CURRENT_GAMEWEEK (set to 25 in conftest)
        result = get_current_gameweek()
        assert isinstance(result, int)


class TestGetClassicBootstrapStatic:
    @patch("scripts.common.utils.requests.get")
    def test_returns_dict(self, mock_get, mock_bootstrap_data):
        mock_get.return_value = _mock_response(mock_bootstrap_data)
        result = get_classic_bootstrap_static()
        assert isinstance(result, dict)
        assert "elements" in result
        assert "teams" in result

    @patch("scripts.common.utils.requests.get")
    def test_handles_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_classic_bootstrap_static()
        assert result is None


class TestGetLeagueEntries:
    @patch("scripts.common.utils.requests.get")
    def test_returns_entry_dict(self, mock_get, mock_league_response):
        mock_get.return_value = _mock_response(mock_league_response)
        result = get_league_entries(12345)
        assert isinstance(result, dict)
        assert result[101] == "Team Alpha"
        assert result[102] == "Team Beta"

    @patch("scripts.common.utils.requests.get")
    def test_handles_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_league_entries(12345)
        assert result == {}


class TestIsDraftLeagueReachable:
    def test_unset_league_id_is_unreachable(self):
        assert is_draft_league_reachable(0) is False
        assert is_draft_league_reachable(None) is False

    @patch("scripts.common.utils.requests.get")
    def test_valid_league_is_reachable(self, mock_get, mock_league_response):
        mock_get.return_value = _mock_response(mock_league_response)
        assert is_draft_league_reachable(12345) is True

    @patch("scripts.common.utils.requests.get")
    def test_stale_or_invalid_league_is_unreachable(self, mock_get):
        """A prior season's league ID (before the new season's league exists)
        typically fails to resolve -- this must surface as False, not raise."""
        mock_get.side_effect = Exception("Network error")
        assert is_draft_league_reachable(12345) is False


class TestGetLeaguePlayerOwnership:
    @patch("scripts.common.utils.requests.get")
    def test_handles_error(self, mock_get):
        """Regression test: a failed/non-JSON API response (e.g. an empty body from a
        stale or off-season league ID) must degrade to {} instead of raising."""
        mock_get.side_effect = Exception("Network error")
        result = get_league_player_ownership(12345)
        assert result == {}

    @patch("scripts.common.utils.requests.get")
    def test_handles_bad_json(self, mock_get):
        """Same regression, but for a response that returns 200 with an unparseable body
        (the actual failure mode reported: JSONDecodeError on an empty response)."""
        bad_resp = MagicMock()
        bad_resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_get.return_value = bad_resp
        result = get_league_player_ownership(12345)
        assert result == {}


class TestGetDraftLeagueDetails:
    @patch("scripts.common.utils.requests.get")
    def test_returns_full_details(self, mock_get, mock_league_response):
        mock_get.return_value = _mock_response(mock_league_response)
        result = get_draft_league_details(12345)
        assert isinstance(result, dict)
        assert "matches" in result
        assert "league_entries" in result
        assert "standings" in result

    def test_returns_none_for_falsy_id(self):
        result = get_draft_league_details(0)
        assert result is None

    @patch("scripts.common.utils.requests.get")
    def test_handles_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        result = get_draft_league_details(12345)
        assert result is None


class TestGetClassicLeagueStandings:
    @patch("scripts.common.utils.requests.get")
    def test_returns_standings(self, mock_get):
        mock_data = {
            "league": {"id": 99999, "name": "Test League"},
            "standings": {"results": [{"entry": 1, "total": 100}]},
        }
        mock_get.return_value = _mock_response(mock_data)
        result = get_classic_league_standings(99999)
        assert isinstance(result, dict)
        assert "standings" in result

    def test_returns_none_for_falsy_id(self):
        result = get_classic_league_standings(0)
        assert result is None


class TestGetClassicTeamHistory:
    @patch("scripts.common.utils.requests.get")
    def test_returns_history(self, mock_get):
        mock_data = {
            "current": [{"event": 1, "points": 50, "total_points": 50}],
            "chips": [],
            "past": [],
        }
        mock_get.return_value = _mock_response(mock_data)
        result = get_classic_team_history(11111)
        assert isinstance(result, dict)
        assert "current" in result

    def test_returns_none_for_falsy_id(self):
        result = get_classic_team_history(0)
        assert result is None


class TestGetRotowirePlayerProjections:
    @patch("scripts.common.utils.requests.get")
    def test_parses_table(self, mock_get):
        """Test parsing a minimal Rotowire HTML table."""
        # Columns: Overall Rank, FW Rank, MID Rank, DEF Rank, GK Rank,
        #          Player, Team, Matchup, Position, Price, TSB %, Points
        html = """
        <html><body>
        <table class="article-table__tablesorter article-table__standard article-table__figure">
        <tbody>
        <tr>
            <td>1</td><td>1</td><td>-</td><td>-</td><td>-</td>
            <td>Haaland</td><td>Man City</td><td>vs WHU (H)</td>
            <td>FWD</td><td>13.0</td><td>95%</td><td>10.0</td>
        </tr>
        </tbody>
        </table>
        </body></html>
        """
        resp = MagicMock()
        resp.content = html.encode()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        mock_get.return_value = resp

        result = get_rotowire_player_projections("https://example.com/rotowire")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.iloc[0]["Player"] == "Haaland"

    @patch("scripts.common.utils.requests.get")
    def test_handles_error(self, mock_get):
        import requests as req
        mock_get.side_effect = req.RequestException("Network error")
        result = get_rotowire_player_projections("https://example.com/bad")
        assert isinstance(result, pd.DataFrame)
        assert result.empty
