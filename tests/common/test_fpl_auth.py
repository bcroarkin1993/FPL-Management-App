"""Tests for the FPL session credential and the authenticated my-team fetch.

The endpoint under test is the only source of a pre-deadline squad, so the
important behaviour here is that failures are *loud and distinguishable*: a
rejected cookie must report `expired`, never a silent None that lets a stale
squad render as if it were current.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from scripts.common import fpl_auth
from scripts.common.fpl_auth import (
    STATUS_ERROR,
    STATUS_EXPIRED,
    STATUS_NO_AUTH,
    STATUS_OK,
    build_headers,
    normalise_my_team,
)


@pytest.fixture
def auth_file(tmp_path, monkeypatch):
    path = tmp_path / ".fpl_auth.json"
    monkeypatch.setattr(fpl_auth, "AUTH_FILE", path)
    return path


# ── Storage ───────────────────────────────────────────────────────────────────

class TestStorage:
    def test_round_trip(self, auth_file):
        fpl_auth.save_auth("pl_profile=abc; sessionid=xyz")
        assert auth_file.exists()
        assert fpl_auth.load_auth()["cookie"] == "pl_profile=abc; sessionid=xyz"
        assert fpl_auth.has_auth() is True

    def test_missing_file_is_not_an_error(self, auth_file):
        assert fpl_auth.load_auth() == {}
        assert fpl_auth.has_auth() is False

    def test_corrupt_file_is_not_an_error(self, auth_file):
        auth_file.write_text("{not json", encoding="utf-8")
        assert fpl_auth.load_auth() == {}

    def test_saved_file_is_owner_only(self, auth_file):
        fpl_auth.save_auth("sessionid=xyz")
        assert (auth_file.stat().st_mode & 0o077) == 0

    def test_clear_is_idempotent(self, auth_file):
        fpl_auth.clear_auth()  # absent
        fpl_auth.save_auth("sessionid=xyz")
        fpl_auth.clear_auth()
        assert not auth_file.exists()

    def test_never_writes_league_settings(self, auth_file, tmp_path):
        """A credential must not land in league_settings.json — that file holds
        non-secret IDs and this repo is public."""
        league = tmp_path / "league_settings.json"
        fpl_auth.save_auth("sessionid=secret")
        assert not league.exists()
        assert auth_file.read_text(encoding="utf-8").count("sessionid=secret") == 1


# ── Headers ───────────────────────────────────────────────────────────────────

class TestBuildHeaders:
    def test_cookie_only(self):
        h = build_headers({"cookie": "sessionid=xyz"})
        assert h["Cookie"] == "sessionid=xyz"
        assert "X-API-Authorization" not in h
        assert "Mozilla" in h["User-Agent"]

    def test_bearer_added_when_present(self):
        h = build_headers({"cookie": "sessionid=xyz", "bearer": "tok123"})
        assert h["X-API-Authorization"] == "Bearer tok123"

    def test_bearer_prefix_not_doubled(self):
        h = build_headers({"cookie": "c", "bearer": "Bearer tok123"})
        assert h["X-API-Authorization"] == "Bearer tok123"


# ── Fetch outcomes ────────────────────────────────────────────────────────────

def _response(status_code=200, json_data=None, raise_value=False):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = 200 <= status_code < 300
    if raise_value:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = json_data or {}
    return resp


class TestFetchOutcomes:
    def test_no_credentials(self, auth_file):
        assert fpl_auth._fetch_my_team_uncached(1) == (None, STATUS_NO_AUTH)

    @pytest.mark.parametrize("code", [401, 403])
    def test_rejected_credentials_report_expired(self, auth_file, code):
        """Never a silent None — that is the bug this module exists to fix."""
        fpl_auth.save_auth("sessionid=stale")
        with patch("scripts.common.fpl_auth.requests.get",
                   return_value=_response(code)):
            assert fpl_auth._fetch_my_team_uncached(1) == (None, STATUS_EXPIRED)

    def test_html_login_page_with_200_reports_expired(self, auth_file):
        fpl_auth.save_auth("sessionid=half-valid")
        with patch("scripts.common.fpl_auth.requests.get",
                   return_value=_response(200, raise_value=True)):
            assert fpl_auth._fetch_my_team_uncached(1) == (None, STATUS_EXPIRED)

    def test_transport_failure_reports_error_not_expired(self, auth_file):
        fpl_auth.save_auth("sessionid=xyz")
        with patch("scripts.common.fpl_auth.requests.get",
                   side_effect=ConnectionError("down")):
            assert fpl_auth._fetch_my_team_uncached(1) == (None, STATUS_ERROR)

    def test_maintenance_503_reports_error(self, auth_file):
        fpl_auth.save_auth("sessionid=xyz")
        with patch("scripts.common.fpl_auth.requests.get",
                   return_value=_response(503)):
            assert fpl_auth._fetch_my_team_uncached(1) == (None, STATUS_ERROR)

    def test_success(self, auth_file):
        fpl_auth.save_auth("sessionid=xyz")
        with patch("scripts.common.fpl_auth.requests.get",
                   return_value=_response(200, {"picks": []})):
            data, status = fpl_auth._fetch_my_team_uncached(1)
        assert status == STATUS_OK and data == {"picks": []}

    def test_test_credentials_uses_the_argument_not_the_file(self, auth_file):
        """The League Setup page validates a pasted value before saving it."""
        with patch("scripts.common.fpl_auth.requests.get",
                   return_value=_response(200, {"picks": [{"element": 1, "position": 1}],
                                                "transfers": {"bank": 5, "value": 1000}})) as get:
            payload, status = fpl_auth.test_credentials(
                7, {"cookie": "sessionid=pasted"})
        assert status == STATUS_OK
        assert payload["entry_history"]["bank"] == 5
        assert get.call_args.kwargs["headers"]["Cookie"] == "sessionid=pasted"
        assert not auth_file.exists()  # testing must not persist anything


# ── Normalisation ─────────────────────────────────────────────────────────────

class TestNormaliseMyTeam:
    """my-team and the picks endpoint differ in three ways that all fail
    silently if ignored — bank/value location, chip location, and a missing
    multiplier that _build_squad_dataframe() defaults to 1."""

    def _raw(self, chips=None):
        return {
            "picks": [
                {"element": 10, "position": 1, "is_captain": False, "is_vice_captain": False},
                {"element": 11, "position": 2, "is_captain": True, "is_vice_captain": False},
                {"element": 12, "position": 12, "is_captain": False, "is_vice_captain": True},
            ],
            "chips": chips or [],
            "transfers": {"bank": 15, "value": 1006, "made": 2, "cost": 4, "limit": 1},
        }

    def test_bank_and_value_move_into_entry_history(self):
        out = normalise_my_team(self._raw())
        assert out["entry_history"] == {
            "value": 1006, "bank": 15, "event_transfers": 2, "event_transfers_cost": 4}

    def test_captain_gets_a_doubled_multiplier(self):
        out = normalise_my_team(self._raw())
        captain = next(p for p in out["picks"] if p["is_captain"])
        assert captain["multiplier"] == 2

    def test_bench_is_zeroed(self):
        out = normalise_my_team(self._raw())
        bench = next(p for p in out["picks"] if p["position"] > 11)
        assert bench["multiplier"] == 0

    def test_triple_captain_triples(self):
        raw = self._raw([{"name": "3xc", "status_for_entry": "active"}])
        out = normalise_my_team(raw)
        assert out["active_chip"] == "3xc"
        assert next(p for p in out["picks"] if p["is_captain"])["multiplier"] == 3

    def test_bench_boost_pays_the_bench(self):
        raw = self._raw([{"name": "bboost", "status_for_entry": "active"}])
        out = normalise_my_team(raw)
        assert next(p for p in out["picks"] if p["position"] > 11)["multiplier"] == 1

    def test_active_wildcard_is_surfaced(self):
        """The whole point: an active pre-deadline wildcard appears here and
        nowhere else in the public API."""
        raw = self._raw([
            {"name": "wildcard", "status_for_entry": "active"},
            {"name": "bboost", "status_for_entry": "available"},
        ])
        assert normalise_my_team(raw)["active_chip"] == "wildcard"

    def test_no_active_chip(self):
        raw = self._raw([{"name": "wildcard", "status_for_entry": "available"}])
        assert normalise_my_team(raw)["active_chip"] is None
