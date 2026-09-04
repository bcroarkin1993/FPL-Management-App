"""Live checks on the endpoints the Classic squad resolver depends on.

These pin the one fact the resolver's precedence actually rests on: the picks
endpoint 404s for a gameweek whose deadline has not passed, and my-team is
gated behind authentication. If either changes, we want a failing test rather
than a quietly wrong squad.

Deliberately NOT asserted: whether entry/{id}/transfers/ publishes a move
before its deadline. That is unresolved (see scripts/common/fpl_auth.py), and a
test that asserted either way would be pinning a guess.

Contract, per tests/live/conftest.py: unreachable → SKIP, reachable but
implausible → FAIL. FPL takes the API down for a "The game is being updated"
maintenance window around each deadline; that is an outage, not a failure.
"""

import pytest
import requests

import config
from scripts.common.fpl_auth import MY_TEAM_URL, STATUS_NO_AUTH, fetch_my_team
from tests.live.conftest import skip_if_unreachable

TIMEOUT = 30
_UA = {"User-Agent": "Mozilla/5.0"}


def _get(url):
    resp = requests.get(url, timeout=TIMEOUT, headers=_UA)
    if resp.status_code == 503:
        pytest.skip("FPL API is in its post-deadline maintenance window")
    return resp


@pytest.fixture(scope="module")
def team_id():
    tid = config.FPL_CLASSIC_TEAM_ID
    if not tid:
        pytest.skip("No Classic team configured")
    return int(tid)


class TestMyTeamRequiresAuth:
    def test_unauthenticated_my_team_is_rejected(self, team_id):
        """my-team is the only pre-deadline squad source and it is gated.

        If this ever returns 200 unauthenticated, the credential plumbing is
        unnecessary — and if it starts returning 200 with an empty body we must
        fail rather than silently trust it.
        """
        resp = _get(MY_TEAM_URL.format(team_id=team_id))
        assert resp.status_code in (401, 403), (
            "my-team returned HTTP %s unauthenticated; the resolver assumes it "
            "is always gated." % resp.status_code
        )

    def test_fetch_my_team_reports_no_auth_without_a_credential(self, team_id):
        """The absent-credential path must be a clean, named state — never an
        exception and never a silent None that reads as 'no squad'."""
        from scripts.common import fpl_auth

        if fpl_auth.has_auth():
            pytest.skip("A credential is configured on this machine")
        payload, status = fetch_my_team(team_id)
        assert payload is None
        assert status == STATUS_NO_AUTH


class TestPublicEndpointsCannotSeeAPendingSquad:
    """The three facts that make the authenticated path necessary."""

    def test_picks_404_for_the_upcoming_gameweek(self, team_id):
        bootstrap = skip_if_unreachable(
            lambda: _get("https://fantasy.premierleague.com/api/bootstrap-static/").json(),
            "FPL bootstrap",
        )
        next_gw = next((e["id"] for e in bootstrap.get("events", [])
                        if e.get("is_next")), None)
        if next_gw is None:
            pytest.skip("No upcoming gameweek (season complete)")

        resp = _get(
            "https://fantasy.premierleague.com/api/entry/%s/event/%s/picks/"
            % (team_id, next_gw)
        )
        assert resp.status_code == 404, (
            "picks for the upcoming GW%s returned HTTP %s, not 404. If FPL now "
            "publishes them, the resolver can read the pending squad without a "
            "credential." % (next_gw, resp.status_code)
        )

    def test_transfers_endpoint_is_a_list(self, team_id):
        """Shape check only. Contents are not asserted: this endpoint publishes
        nothing for an upcoming GW until its deadline passes, which is exactly
        why the squad cannot be reconstructed from it pre-deadline."""
        data = skip_if_unreachable(
            lambda: _get("https://fantasy.premierleague.com/api/entry/%s/transfers/"
                         % team_id).json(),
            "FPL transfers endpoint",
        )
        assert isinstance(data, list)
        for row in data[:5]:
            assert {"element_in", "element_out", "event"} <= set(row)
