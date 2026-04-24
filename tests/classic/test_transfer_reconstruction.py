"""Tests for squad reconstruction via _apply_pending_transfers.

Covers the real-world scenario: user used Free Hit in GW33, squad reverts to
GW32, then makes a permanent Bowen-for-Ekitike transfer before GW34's deadline.
FPL's API doesn't expose pre-deadline transfers, so we reconstruct via local
session-state entries.

Key facts from the user's live session:
  - picks_source_gw = 32
  - fh_gws = {18, 33}
  - Ekitike FPL element ID = 373
  - get_current_gameweek() may return 33 (the FH GW) between gameweeks via
    the Draft API — this is the root cause of the persistent bug: a locally-
    logged transfer saved with event=33 was filtered by the FH exclusion.
"""

import pytest
from scripts.classic.transfers import (
    _apply_pending_transfers,
    _add_pending_local,
    _get_pending_local,
    _remove_pending_local,
    _sync_pending_local,
    _PENDING_KEY,
)

EKITIKE_ID = 373
BOWEN_ID   = 627   # Jarrod Bowen — update if wrong
OTHER_IDS  = [101, 151, 5, 683, 82, 235, 449, 267, 136, 249, 67, 661, 547, 713]

FH_GWS       = {18, 33}
BASE_GW      = 32   # picks_source_gw
CURRENT_GW   = 34   # real upcoming GW
DRAFT_API_GW = 33   # what get_current_gameweek() may return between GWs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _picks(player_ids):
    return {
        "picks": [
            {"element": pid, "selling_price": 60, "multiplier": 1,
             "is_captain": False, "is_vice_captain": False, "position": i + 1}
            for i, pid in enumerate(player_ids)
        ],
        "entry_history": {"bank": 10, "value": 1000},
        "active_chip": None,
    }


def _bootstrap(*ids):
    return {"elements": [{"id": i, "web_name": f"P{i}", "now_cost": 60} for i in ids]}


def _api_transfer(event, element_out, element_in, time="2026-04-17T20:00:00Z"):
    return {"event": event, "element_out": element_out, "element_in": element_in,
            "element_out_cost": 60, "element_in_cost": 65, "time": time}


def _local_transfer(event, element_out, element_in):
    """Mirrors what _add_pending_local saves to session state."""
    return {"event": event, "element_out": element_out, "element_in": element_in,
            "element_out_cost": 60, "element_in_cost": 65,
            "time": "2026-04-24T09:00:00Z", "local": True}


def _player_ids(picks_data):
    return [p["element"] for p in picks_data["picks"]]


# ── Root-cause regression test ─────────────────────────────────────────────────

class TestLocalTransferBypassesFhFilter:
    """The core bug: event == FH GW causes transfer to be silently dropped."""

    def test_local_transfer_with_event_equal_to_fh_gw_is_applied(self):
        """
        REGRESSION: get_current_gameweek() returns 33 (the FH GW) between GWs.
        Local transfer saved with event=33.  Old code filtered it as a FH transfer.
        New code: local=True bypasses the FH event filter.
        """
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        data  = _picks(squad)
        local = _local_transfer(event=33, element_out=EKITIKE_ID, element_in=BOWEN_ID)
        bs    = _bootstrap(*squad, BOWEN_ID)

        result, applied = _apply_pending_transfers(data, BASE_GW, [local], FH_GWS, bs)

        assert BOWEN_ID in _player_ids(result), (
            "Bowen must replace Ekitike even when event=33 (FH GW). "
            "This was the root cause — local transfers must bypass the FH filter."
        )
        assert EKITIKE_ID not in _player_ids(result)
        assert len(applied) == 1

    def test_local_transfer_with_event_34_is_applied(self):
        """Happy path: Draft API correctly returns 34."""
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        data  = _picks(squad)
        local = _local_transfer(event=34, element_out=EKITIKE_ID, element_in=BOWEN_ID)
        bs    = _bootstrap(*squad, BOWEN_ID)

        result, applied = _apply_pending_transfers(data, BASE_GW, [local], FH_GWS, bs)

        assert BOWEN_ID in _player_ids(result)
        assert EKITIKE_ID not in _player_ids(result)
        assert len(applied) == 1

    def test_api_fh_transfer_still_excluded(self):
        """API FH transfers (no local flag) with event=33 must still be filtered."""
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        data  = _picks(squad)
        fh_api = _api_transfer(event=33, element_out=EKITIKE_ID, element_in=BOWEN_ID,
                               time="2026-04-17T20:44:00Z")
        bs = _bootstrap(*squad, BOWEN_ID)

        result, applied = _apply_pending_transfers(data, BASE_GW, [fh_api], FH_GWS, bs)

        assert EKITIKE_ID in _player_ids(result), "FH API transfer must be excluded"
        assert BOWEN_ID not in _player_ids(result)
        assert applied == []

    def test_local_and_api_fh_transfers_together(self):
        """FH API transfers (event=33) excluded; local transfer (event=33) applied."""
        P_OUT, P_IN = 500, 501
        squad = [EKITIKE_ID, P_OUT] + OTHER_IDS[:13]
        data  = _picks(squad)
        fh_api = _api_transfer(33, P_OUT, 999, time="2026-04-17T20:44:00Z")  # FH — excluded
        local  = _local_transfer(33, EKITIKE_ID, BOWEN_ID)                   # local — applied
        bs     = _bootstrap(*squad, BOWEN_ID, 999)

        result, applied = _apply_pending_transfers(data, BASE_GW, [fh_api, local], FH_GWS, bs)

        ids = _player_ids(result)
        assert BOWEN_ID in ids,   "local transfer must be applied"
        assert EKITIKE_ID not in ids
        assert 999 not in ids,    "FH API transfer must not be applied"
        assert P_OUT in ids,      "P_OUT stays (only FH API transfer targeted it)"
        assert len(applied) == 1


# ── Full squad reconstruction flow ────────────────────────────────────────────

class TestSquadReconstructionFlow:
    """Tests that mirror the complete page-load flow."""

    def test_ekitike_replaced_by_bowen_after_fh(self):
        """
        Full scenario:
          GW32 picks (from SQLite) include Ekitike.
          GW33 was Free Hit — many FH API transfers, all event=33.
          User makes Bowen-for-Ekitike permanent transfer, logged locally with event=33
          (because Draft API returns 33 between gameweeks).
        Expected: Bowen is in squad, Ekitike is out.
        """
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        gw32_picks = _picks(squad)

        # Simulate FH batch: 15 players swapped in GW33 via API (excluded)
        fh_api_transfers = [
            _api_transfer(33, squad[i], 900 + i, time=f"2026-04-17T20:44:{i:02d}Z")
            for i in range(10)
        ]
        # User's real transfer: logged locally (event=33 from Draft API lag)
        bowen_transfer = _local_transfer(33, EKITIKE_ID, BOWEN_ID)
        all_transfers = fh_api_transfers + [bowen_transfer]

        bs = _bootstrap(*squad, BOWEN_ID, *[900 + i for i in range(10)])

        result, applied = _apply_pending_transfers(
            gw32_picks, BASE_GW, all_transfers, FH_GWS, bs
        )

        ids = _player_ids(result)
        assert BOWEN_ID in ids,    "Bowen must be in squad"
        assert EKITIKE_ID not in ids, "Ekitike must be transferred out"
        # FH batch must NOT have been applied
        for i in range(10):
            assert 900 + i not in ids, f"FH player {900+i} must not be in squad"
        assert len(applied) == 1

    def test_no_local_transfer_shows_gw32_squad(self):
        """Without a local transfer, GW32 squad (Ekitike) is returned unchanged."""
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        gw32_picks = _picks(squad)
        fh_transfers = [_api_transfer(33, squad[i], 900 + i) for i in range(10)]
        bs = _bootstrap(*squad, *[900 + i for i in range(10)])

        result, applied = _apply_pending_transfers(
            gw32_picks, BASE_GW, fh_transfers, FH_GWS, bs
        )

        assert EKITIKE_ID in _player_ids(result), "Ekitike still in squad — no local transfer"
        assert applied == []

    def test_multiple_local_transfers(self):
        """Two local transfers both applied correctly."""
        P1_OUT, P1_IN = EKITIKE_ID, BOWEN_ID
        P2_OUT, P2_IN = OTHER_IDS[0], 800
        squad = [P1_OUT, P2_OUT] + OTHER_IDS[1:14]
        data  = _picks(squad)
        transfers = [
            _local_transfer(33, P1_OUT, P1_IN),
            _local_transfer(33, P2_OUT, P2_IN),
        ]
        bs = _bootstrap(*squad, P1_IN, P2_IN)

        result, applied = _apply_pending_transfers(data, BASE_GW, transfers, FH_GWS, bs)

        ids = _player_ids(result)
        assert P1_IN in ids and P1_OUT not in ids
        assert P2_IN in ids and P2_OUT not in ids
        assert len(applied) == 2


# ── Edge cases ─────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_transfers_returns_original_object(self):
        data = _picks(OTHER_IDS[:15])
        result, applied = _apply_pending_transfers(
            data, BASE_GW, [], FH_GWS, _bootstrap(*OTHER_IDS[:15])
        )
        assert result is data
        assert applied == []

    def test_does_not_mutate_original(self):
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        data  = _picks(squad)
        before = _player_ids(data)
        local = _local_transfer(33, EKITIKE_ID, BOWEN_ID)
        _apply_pending_transfers(data, BASE_GW, [local], FH_GWS, _bootstrap(*squad, BOWEN_ID))
        assert _player_ids(data) == before

    def test_local_transfer_event_equal_to_base_gw_is_applied(self):
        """Local transfers with event == picks_source_gw ARE applied (>= semantics).
        Scenario: FPL returns current-GW picks as the base; user logs a transfer
        also tagged with the current GW — we use >= so it's not silently dropped."""
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        data  = _picks(squad)
        local = _local_transfer(BASE_GW, EKITIKE_ID, BOWEN_ID)  # event=32, base=32
        result, applied = _apply_pending_transfers(
            data, BASE_GW, [local], FH_GWS, _bootstrap(*squad, BOWEN_ID)
        )
        assert BOWEN_ID in _player_ids(result)
        assert len(applied) == 1

    def test_out_player_not_in_squad_is_skipped(self):
        """If element_out isn't in picks (already transferred?), nothing blows up."""
        squad = OTHER_IDS[:15]  # Ekitike not present
        data  = _picks(squad)
        local = _local_transfer(33, EKITIKE_ID, BOWEN_ID)
        result, applied = _apply_pending_transfers(
            data, BASE_GW, [local], FH_GWS, _bootstrap(*squad, BOWEN_ID)
        )
        # No crash, no change to squad (out player wasn't there)
        assert BOWEN_ID not in _player_ids(result)
        # applied may or may not list it — just verify squad unchanged
        assert set(_player_ids(result)) == set(squad)


# ── Session state pipeline tests (no Streamlit needed) ────────────────────────

class TestSessionStatePipeline:
    """Tests for the _add / _sync / _get functions using an injected state dict.

    These replace the Streamlit session_state with a plain dict so we can test
    the full add → sync → apply pipeline without starting the app.
    """

    def _make_state(self):
        return {}  # stand-in for st.session_state

    def test_add_stores_transfer(self):
        state = self._make_state()
        _add_pending_local(team_id=123, event=33, element_out=EKITIKE_ID,
                           element_in=BOWEN_ID, out_cost=60, in_cost=65, _state=state)
        result = _get_pending_local(123, _state=state)
        assert len(result) == 1
        assert result[0]["element_out"] == EKITIKE_ID
        assert result[0]["element_in"] == BOWEN_ID
        assert result[0]["local"] is True

    def test_add_replaces_existing_for_same_out_player(self):
        state = self._make_state()
        _add_pending_local(123, 33, EKITIKE_ID, 999, 60, 65, _state=state)
        _add_pending_local(123, 33, EKITIKE_ID, BOWEN_ID, 60, 65, _state=state)
        result = _get_pending_local(123, _state=state)
        assert len(result) == 1
        assert result[0]["element_in"] == BOWEN_ID  # second entry wins

    def test_remove_deletes_entry(self):
        state = self._make_state()
        _add_pending_local(123, 33, EKITIKE_ID, BOWEN_ID, 60, 65, _state=state)
        _remove_pending_local(123, EKITIKE_ID, _state=state)
        assert _get_pending_local(123, _state=state) == []

    def test_sync_keeps_unconfirmed(self):
        state = self._make_state()
        _add_pending_local(123, 33, EKITIKE_ID, BOWEN_ID, 60, 65, _state=state)
        remaining = _sync_pending_local(123, api_transfers=[], _state=state)
        assert len(remaining) == 1

    def test_sync_removes_confirmed(self):
        """FPL API confirms a same-GW transfer → local entry cleared."""
        state = self._make_state()
        _add_pending_local(123, 33, EKITIKE_ID, BOWEN_ID, 60, 65, _state=state)
        # Must include event so the (out, in, event) triplet matches
        api_confirmed = [{"element_out": EKITIKE_ID, "element_in": BOWEN_ID, "event": 33}]
        remaining = _sync_pending_local(123, api_confirmed, _state=state)
        assert remaining == []

    def test_sync_does_not_remove_historical_transfer(self):
        """Historical API transfer (same players, old GW) must NOT purge a new local transfer.
        Root cause of the reported bug: FPL returns all-season transfers; a previous
        Ekitike→Bowen swap would incorrectly clear the new pending transfer."""
        state = self._make_state()
        _add_pending_local(123, 34, EKITIKE_ID, BOWEN_ID, 60, 65, _state=state)
        # Historical transfer from GW25 — same players but different event
        historical_api = [{"element_out": EKITIKE_ID, "element_in": BOWEN_ID, "event": 25}]
        remaining = _sync_pending_local(123, historical_api, _state=state)
        assert len(remaining) == 1, (
            "Historical same-player transfer (event=25) must NOT clear a new pending "
            "transfer (event=34). Old pair-only matching was the bug."
        )

    def test_sync_does_not_remove_different_in_player(self):
        """Logged (373→Bowen) must not be purged by API entry (373→260) from FH."""
        state = self._make_state()
        _add_pending_local(123, 33, EKITIKE_ID, BOWEN_ID, 60, 65, _state=state)
        fh_api = [{"element_out": EKITIKE_ID, "element_in": 260}]  # FH, different player
        remaining = _sync_pending_local(123, fh_api, _state=state)
        assert len(remaining) == 1  # NOT purged

    def test_full_pipeline_ekitike_to_bowen(self):
        """
        Simulates the complete app flow on a single page rerun:
          1. add_pending_local (what happens when user clicks the button)
          2. sync_pending_local (what happens at top of next page load)
          3. _apply_pending_transfers (squad reconstruction)
        Result: Bowen in squad, Ekitike out.
        """
        state = self._make_state()
        team_id = 123

        # Step 1: user clicks button — event=33 (Draft API lag)
        _add_pending_local(team_id, event=33, element_out=EKITIKE_ID,
                           element_in=BOWEN_ID, out_cost=56, in_cost=65, _state=state)

        # Step 2: page rerun — sync against FPL API (no confirmed transfers yet)
        fh_api_transfers = [
            {"element_out": squad_id, "element_in": 900 + i, "event": 33}
            for i, squad_id in enumerate([5, 82, 101, 136, 151, 235, 449, 267, 661, 713])
        ]
        local_pending = _sync_pending_local(team_id, fh_api_transfers, _state=state)
        confirmed_keys = {(t["element_out"], t["element_in"], t.get("event", 0))
                          for t in fh_api_transfers}
        extra_local = [t for t in local_pending
                       if (t["element_out"], t["element_in"], t.get("event", 0))
                       not in confirmed_keys]

        assert len(extra_local) == 1, "Pending transfer must survive sync"

        # Step 3: apply to GW32 picks
        squad = [EKITIKE_ID] + OTHER_IDS[:14]
        picks_data = _picks(squad)
        bs = _bootstrap(*squad, BOWEN_ID)
        effective = fh_api_transfers + extra_local

        result, applied = _apply_pending_transfers(
            picks_data, BASE_GW, effective, FH_GWS, bs
        )

        ids = _player_ids(result)
        assert BOWEN_ID in ids, "Bowen must be in squad after full pipeline"
        assert EKITIKE_ID not in ids, "Ekitike must be out after full pipeline"
        assert len(applied) == 1

    def test_team_id_isolation(self):
        """Transfers for one team don't appear for another."""
        state = self._make_state()
        _add_pending_local(team_id=111, event=33, element_out=EKITIKE_ID,
                           element_in=BOWEN_ID, out_cost=60, in_cost=65, _state=state)
        assert _get_pending_local(222, _state=state) == []
        assert len(_get_pending_local(111, _state=state)) == 1


# ── Live API helpers (skipped in CI, run manually to diagnose) ─────────────────

@pytest.mark.skip(reason="Live API — run manually: pytest tests/classic/test_transfer_reconstruction.py::test_live_transfers -s")
def test_live_transfers():
    """Print all transfers for the configured team, newest first.
    Shows whether the Bowen transfer appears and what event number it has.
    Run after the GW34 deadline to confirm it's now visible in the API.
    """
    import requests, config
    team_id = config.FPL_CLASSIC_TEAM_ID
    assert team_id, "Set FPL_CLASSIC_TEAM_ID in .env"
    resp = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/",
        timeout=30
    )
    resp.raise_for_status()
    transfers = sorted(resp.json(), key=lambda t: t.get("time", ""), reverse=True)
    print(f"\nTotal: {len(transfers)}")
    for t in transfers[:15]:
        print(f"  event={t['event']}  out={t['element_out']}  in={t['element_in']}  {t['time']}")
    gw34 = [t for t in transfers if t.get("event") == 34]
    print(f"\nGW34 transfers: {gw34}")
