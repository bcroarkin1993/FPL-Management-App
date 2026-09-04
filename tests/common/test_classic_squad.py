"""Tests for the shared Classic squad resolver.

The bug that prompted this module: a manager played a wildcard before the GW3
deadline, and Fixture Projections rendered the pre-wildcard squad with no
indication it was stale. Verified live — FPL's picks endpoint 404s for an
upcoming GW, `transfers/` returns [], and an active chip is absent from
`history` — so the reconstruction had nothing to replay and silently returned
the old fifteen.

These tests pin the precedence order, the provenance the pages render, and the
two traps that make a wrong squad look right.
"""

from unittest.mock import patch

import pytest

from scripts.common import classic_squad
from scripts.common.classic_squad import (
    SOURCE_LOCAL_PENDING,
    SOURCE_MY_TEAM,
    SOURCE_PICKS,
    SOURCE_PICKS_REPLAY,
    SquadResolution,
    resolve_classic_squad,
)
from scripts.common.fpl_auth import STATUS_EXPIRED, STATUS_NO_AUTH, STATUS_OK

USER_TEAM = 6720205
OPPONENT = 999999

BOOTSTRAP = {
    "events": [
        {"id": 2, "is_next": False},
        {"id": 3, "is_next": True},
    ],
    "elements": [{"id": i, "web_name": f"P{i}", "team": 1, "now_cost": 50}
                 for i in range(1, 40)],
}


def _picks(ids):
    return {
        "picks": [{"element": i, "position": n + 1, "multiplier": 1}
                  for n, i in enumerate(ids)],
        "active_chip": None,
        "entry_history": {"bank": 10, "value": 1000},
    }


PRE_WILDCARD = list(range(1, 16))
POST_WILDCARD = list(range(20, 35))


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolate the pending-transfer file and pin the configured team id."""
    monkeypatch.setattr(classic_squad, "PENDING_FILE", tmp_path / ".pending.json")
    monkeypatch.setattr(classic_squad.config, "FPL_CLASSIC_TEAM_ID", USER_TEAM,
                        raising=False)
    return tmp_path


def _resolve(team_id=USER_TEAM, picks_by_gw=None, transfers=None,
             my_team=None, auth_status=STATUS_NO_AUTH, history=None,
             has_auth=False, **kwargs):
    picks_by_gw = picks_by_gw if picks_by_gw is not None else {2: _picks(PRE_WILDCARD)}
    with patch.object(classic_squad, "has_auth", return_value=has_auth), \
         patch.object(classic_squad, "fetch_my_team",
                      return_value=(my_team, auth_status)), \
         patch.object(classic_squad, "get_classic_team_picks",
                      side_effect=lambda tid, gw: picks_by_gw.get(gw)), \
         patch.object(classic_squad, "get_classic_transfers",
                      return_value=transfers or []), \
         patch.object(classic_squad, "get_classic_team_history",
                      return_value=history or {"chips": []}):
        return resolve_classic_squad(team_id, BOOTSTRAP, 3, **kwargs)


# ── Precedence ────────────────────────────────────────────────────────────────

class TestPrecedence:
    def test_authenticated_squad_wins(self, env):
        """The reported bug, fixed: the live squad, not the pre-wildcard one."""
        live = _picks(POST_WILDCARD)
        live["active_chip"] = "wildcard"
        res = _resolve(has_auth=True, my_team=live, auth_status=STATUS_OK)

        assert res.source == SOURCE_MY_TEAM
        assert [p["element"] for p in res.picks] == POST_WILDCARD
        assert res.active_chip == "wildcard"
        assert res.is_stale is False
        assert res.provenance == "Live squad from your FPL account."

    def test_falls_back_to_last_deadline_without_credentials(self, env):
        res = _resolve()
        assert res.source == SOURCE_PICKS
        assert [p["element"] for p in res.picks] == PRE_WILDCARD
        assert res.source_gw == 2 and res.target_gw == 3

    def test_expired_credentials_fall_back_but_stay_visible(self, env):
        """A stale cookie must not degrade into a silent stale squad."""
        res = _resolve(has_auth=True, my_team=None, auth_status=STATUS_EXPIRED)
        assert res.source == SOURCE_PICKS
        assert res.auth_status == STATUS_EXPIRED
        assert res.is_stale is True

    def test_confirmed_transfers_are_replayed(self, env):
        res = _resolve(transfers=[{"event": 3, "element_out": 1, "element_in": 99,
                                   "element_out_cost": 50, "element_in_cost": 50}])
        assert res.source == SOURCE_PICKS_REPLAY
        assert 99 in [p["element"] for p in res.picks]
        assert 1 not in [p["element"] for p in res.picks]

    def test_no_team_id(self, env):
        res = resolve_classic_squad(0, BOOTSTRAP, 3)
        assert res.ok is False

    def test_no_picks_anywhere(self, env):
        res = _resolve(picks_by_gw={})
        assert res.ok is False
        assert "Could not load" in res.provenance


# ── Auth is scoped to the user's own team ─────────────────────────────────────

class TestAuthScoping:
    def test_opponents_never_attempt_auth(self, env):
        """The leaderboard loop and H2H opponents run through this same
        function; a credential must never be reached for on their behalf."""
        with patch.object(classic_squad, "fetch_my_team") as fetch, \
             patch.object(classic_squad, "has_auth", return_value=True), \
             patch.object(classic_squad, "get_classic_team_picks",
                          side_effect=lambda tid, gw: {2: _picks(PRE_WILDCARD)}.get(gw)), \
             patch.object(classic_squad, "get_classic_transfers", return_value=[]), \
             patch.object(classic_squad, "get_classic_team_history",
                          return_value={"chips": []}):
            res = resolve_classic_squad(OPPONENT, BOOTSTRAP, 3)
        fetch.assert_not_called()
        assert res.source == SOURCE_PICKS

    def test_allow_auth_can_be_forced_off(self, env):
        with patch.object(classic_squad, "fetch_my_team") as fetch, \
             patch.object(classic_squad, "has_auth", return_value=True), \
             patch.object(classic_squad, "get_classic_team_picks",
                          side_effect=lambda tid, gw: {2: _picks(PRE_WILDCARD)}.get(gw)), \
             patch.object(classic_squad, "get_classic_transfers", return_value=[]), \
             patch.object(classic_squad, "get_classic_team_history",
                          return_value={"chips": []}):
            resolve_classic_squad(USER_TEAM, BOOTSTRAP, 3, allow_auth=False)
        fetch.assert_not_called()


# ── Free Hit ──────────────────────────────────────────────────────────────────

class TestFreeHitSkip:
    def test_free_hit_gameweek_is_never_the_base(self, env):
        """A Free Hit squad reverts afterwards — using it as the base rebuilds
        a squad the manager does not own. Fixture Projections never did this
        skip; the shared resolver gives it that fix for every team."""
        res = _resolve(
            picks_by_gw={2: _picks(POST_WILDCARD), 1: _picks(PRE_WILDCARD)},
            history={"chips": [{"name": "freehit", "event": 2}]},
        )
        assert res.source_gw == 1
        assert [p["element"] for p in res.picks] == PRE_WILDCARD


# ── Staleness ─────────────────────────────────────────────────────────────────

class TestStaleness:
    def test_stale_between_gameweeks(self, env):
        res = _resolve()
        assert res.is_stale is True
        assert "does not publish pending transfers" in res.provenance
        assert "GW3" in res.provenance

    def test_not_stale_once_the_target_gw_has_picks(self, env):
        res = _resolve(picks_by_gw={3: _picks(POST_WILDCARD)})
        assert res.is_stale is False
        assert res.source_gw == 3


# ── Locally-logged pending transfers ──────────────────────────────────────────

class TestLocalPending:
    def _log(self, env, entries):
        classic_squad.save_pending_file(entries)

    def test_local_transfers_are_replayed_and_reported(self, env):
        self._log(env, [{"team_id": USER_TEAM, "event": 3, "local": True,
                         "element_out": 1, "element_in": 77,
                         "element_out_cost": 50, "element_in_cost": 50}])
        res = _resolve()
        assert res.source == SOURCE_LOCAL_PENDING
        assert 77 in [p["element"] for p in res.picks]
        assert res.is_stale is False
        assert "locally-logged" in res.provenance

    def test_another_managers_log_is_ignored(self, env):
        self._log(env, [{"team_id": OPPONENT, "event": 3, "local": True,
                         "element_out": 1, "element_in": 77}])
        res = _resolve()
        assert 77 not in [p["element"] for p in res.picks]

    def test_authenticated_squad_retires_a_settled_log_entry(self, env):
        """Replaying a transfer the live squad already contains would
        double-count its bank adjustment — the pick swap no-ops but the bank
        line fires regardless. So a settled entry is dropped from the log."""
        self._log(env, [{"team_id": USER_TEAM, "event": 3, "local": True,
                         "element_out": 1, "element_in": 20,
                         "element_out_cost": 50, "element_in_cost": 50}])
        live = _picks(POST_WILDCARD)  # element 1 is gone, 20 is in
        res = _resolve(has_auth=True, my_team=live, auth_status=STATUS_OK)

        assert res.source == SOURCE_MY_TEAM
        assert res.local_stale == []
        assert res.entry_history["bank"] == 10  # untouched, not double-counted
        assert classic_squad.load_pending_file() == []

    def test_authenticated_squad_flags_a_contradicted_log_entry(self, env):
        """Logged in the app but element_out is still in the real squad: the
        move never went through, so applying it would invent a squad."""
        self._log(env, [{"team_id": USER_TEAM, "event": 3, "local": True,
                         "element_out": 20, "element_in": 88}])
        res = _resolve(has_auth=True, my_team=_picks(POST_WILDCARD),
                       auth_status=STATUS_OK)

        assert [t["element_in"] for t in res.local_stale] == [88]
        assert 88 not in [p["element"] for p in res.picks]
        assert classic_squad.load_pending_file() != []  # kept, not silently dropped


# ── Budget ────────────────────────────────────────────────────────────────────

class TestBudget:
    def test_uses_resolved_value_and_bank(self, env):
        with patch.object(classic_squad, "resolve_classic_squad",
                          return_value=SquadResolution(
                              picks=[{"element": 1}],
                              entry_history={"value": 1006, "bank": 15})):
            total, label = classic_squad.get_squad_budget(USER_TEAM, BOOTSTRAP, 3)
        assert total == pytest.approx(102.1)
        assert "£100.6m" in label and "£1.5m" in label

    def test_fallback_includes_the_bank(self, env):
        """last_deadline_value excludes the bank; reading it alone under-reports
        the budget by whatever is banked, which the optimizer then cannot spend."""
        with patch.object(classic_squad, "resolve_classic_squad",
                          return_value=SquadResolution()), \
             patch.object(classic_squad, "get_entry_details",
                          return_value={"last_deadline_value": 1000,
                                        "last_deadline_bank": 12}):
            total, label = classic_squad.get_squad_budget(USER_TEAM, BOOTSTRAP, 3)
        assert total == pytest.approx(101.2)
        assert label == "from last deadline value"

    def test_no_team(self, env):
        total, label = classic_squad.get_squad_budget(0, BOOTSTRAP, 3)
        assert total == 100.0 and "no team configured" in label
