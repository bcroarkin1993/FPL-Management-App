"""Tests for the Draft waiver priority model.

The fixtures here are trimmed from league 11347's real payloads (2026-09-24), so
a failure means the model stopped agreeing with an observation rather than with
an invention. The GW4/GW5 transaction rows in particular reproduce the exact
`index` ordering the round-order reconstruction has to recover.
"""

import pandas as pd
import pytest

from scripts.common.waiver_priority import (
    BAND_CONTESTED,
    BAND_LIKELY,
    BAND_LONG_SHOT,
    BAND_UNKNOWN,
    claim_history_stats,
    claim_outlook,
    expected_gone_before,
    managers_ahead,
    my_waiver_pick,
    parse_waiver_order,
    rank_claim_plan,
    reconstruct_round_order,
    rival_needs,
)


def _details():
    """The live league's entries, both id spaces intact."""
    rows = [
        (56182, 56086, "Stoned Squirrels", 7),
        (56183, 56087, "BigBobbyGolazoFC", 10),
        (56184, 56088, "RIP Gary's Boys", 5),
        (56185, 56089, "Top Drawer Balls", 6),
        (56186, 56090, "COYS", 9),
        (56187, 56091, "Chappy's Goats", 4),
        (56188, 56092, "Tofu Jo", 8),
        (56189, 56093, "Kekambas", 2),
        (56190, 56094, "Starboys", 1),
        (87129, 86925, "Chino FC", 3),
    ]
    return {
        "league_entries": [
            {"id": i, "entry_id": e, "entry_name": n, "waiver_pick": w,
             "player_first_name": "A", "player_last_name": "B"}
            for i, e, n, w in rows
        ],
        "standings": [
            {"league_entry": 56183, "rank": 1}, {"league_entry": 56186, "rank": 2},
            {"league_entry": 56188, "rank": 3}, {"league_entry": 56182, "rank": 4},
            {"league_entry": 56185, "rank": 5}, {"league_entry": 56184, "rank": 6},
            {"league_entry": 56187, "rank": 7}, {"league_entry": 87129, "rank": 8},
            {"league_entry": 56189, "rank": 9}, {"league_entry": 56190, "rank": 10},
        ],
    }


def _gw5_transactions():
    """GW5's waiver round, verbatim in processing order (`index`)."""
    raw = [
        (1, 1, 56094, 286, "a"), (2, 1, 86925, 286, "di"), (3, 2, 86925, 330, "a"),
        (4, 1, 56088, 223, "a"), (5, 1, 56086, 389, "a"), (6, 1, 56090, 389, "di"),
        (7, 2, 56090, 608, "a"), (8, 1, 56092, 330, "di"), (9, 2, 56092, 305, "a"),
        (10, 2, 56094, 330, "di"), (11, 3, 56094, 98, "a"), (12, 2, 56088, 136, "do"),
        (13, 3, 56088, 491, "do"), (14, 2, 56086, 136, "a"), (15, 3, 56090, 286, "di"),
        (16, 4, 56090, 155, "a"), (17, 4, 56094, 491, "a"), (18, 3, 56086, 233, "do"),
    ]
    return [
        {"index": i, "priority": p, "entry": e, "element_in": el, "element_out": 1,
         "result": r, "kind": "w", "event": 5}
        for i, p, e, el, r in raw
    ]


class TestParsingThePublishedQueue:
    def test_the_order_is_read_best_pick_first(self):
        order = parse_waiver_order(_details())
        assert [r["pick"] for r in order] == list(range(1, 11))
        assert order[0]["team_name"] == "Starboys"
        assert order[-1]["team_name"] == "BigBobbyGolazoFC"

    def test_both_id_spaces_are_carried(self):
        """entry_id keys element-status and transactions; id keys the standings.

        Crossing them produces an empty match rather than an error, which on the
        page reads as "you have no waiver pick" and disables the feature.
        """
        order = parse_waiver_order(_details())
        mine = next(r for r in order if r["team_name"] == "Stoned Squirrels")
        assert mine["entry_id"] == 56086
        assert mine["league_entry_id"] == 56182
        assert my_waiver_pick(order, 56086) == 7
        assert my_waiver_pick(order, 56182) is None   # the standings-space id

    def test_a_missing_pick_is_dropped_not_defaulted(self):
        """An absent waiver_pick is not pick 0, which would head the queue."""
        payload = _details()
        payload["league_entries"][0]["waiver_pick"] = None
        order = parse_waiver_order(payload)
        assert len(order) == 9
        assert all(r["pick"] is not None for r in order)

    def test_an_unreadable_payload_is_empty_not_a_guess(self):
        for bad in (None, {}, {"league_entries": "nope"}, {"league_entries": [1, 2]}):
            assert parse_waiver_order(bad) == []

    def test_managers_ahead_counts_only_better_picks(self):
        order = parse_waiver_order(_details())
        assert len(managers_ahead(order, 56086)) == 6      # pick 7 of 10
        assert managers_ahead(order, 56094) == []          # pick 1
        assert managers_ahead(order, 999) == []            # not in the league


class TestLeagueHistory:
    def test_round_order_is_recovered_from_first_claim_index(self):
        """Historical waiver_pick is not published, so the order has to be rebuilt."""
        assert reconstruct_round_order(_gw5_transactions(), 5) == [
            56094, 86925, 56088, 56086, 56090, 56092,
        ]

    def test_only_the_asked_for_gameweek_is_considered(self):
        assert reconstruct_round_order(_gw5_transactions(), 4) == []

    def test_participation_is_what_calibrates_the_queue(self):
        """Players gone before your turn is managers-ahead-who-claim, not pick-1.

        Six of ten managers claimed in GW5, so at pick 7 roughly four players go
        first — which is what was actually observed (four accepted claims were
        processed before Stoned Squirrels' first, at index 5).
        """
        stats = claim_history_stats(_gw5_transactions(), n_managers=10)
        assert stats["events_observed"] == 1
        assert stats["claimants_per_event"] == {5: 6}
        assert stats["participation_rate"] == pytest.approx(0.6)
        assert expected_gone_before(7, stats["participation_rate"]) == pytest.approx(3.6)

    def test_contention_counts_claims_not_declines(self):
        """Several `di` rows can trail one acceptance; counting them double-counts."""
        stats = claim_history_stats(_gw5_transactions(), n_managers=10)
        # 330 drew three claims (86925 won it, 56092 and 56094 were beaten).
        assert stats["max_contention"] == 3
        assert 0.0 < stats["contested_share"] <= 1.0

    def test_free_agency_rows_are_not_waiver_history(self):
        rows = _gw5_transactions() + [
            {"index": 99, "priority": 1, "entry": 56087, "element_in": 500,
             "element_out": 1, "result": "a", "kind": "f", "event": 5}
        ]
        assert claim_history_stats(rows, n_managers=10)["claimants_per_event"] == {5: 6}

    def test_no_history_gives_no_rate_rather_than_a_made_up_one(self):
        stats = claim_history_stats([], n_managers=10)
        assert stats["participation_rate"] is None
        assert expected_gone_before(7, None) is None
        assert expected_gone_before(None, 0.6) is None


class TestRivalNeeds:
    def _team_df(self):
        """Power-rankings output: Team_ID is the entry_id space."""
        return pd.DataFrame([
            {"Team_ID": 56094, "GK_Rank": 1, "DEF_Rank": 1, "MID_Rank": 10, "FWD_Rank": 2},
            {"Team_ID": 56093, "GK_Rank": 2, "DEF_Rank": 9, "MID_Rank": 9, "FWD_Rank": 3},
            {"Team_ID": 86925, "GK_Rank": 3, "DEF_Rank": 2, "MID_Rank": 1, "FWD_Rank": 4},
            {"Team_ID": 56086, "GK_Rank": 10, "DEF_Rank": 10, "MID_Rank": 2, "FWD_Rank": 10},
        ])

    def test_only_managers_ahead_of_you_count(self):
        order = parse_waiver_order(_details())
        needs = rival_needs(self._team_df(), order, 56086)
        # 56094 (pick 1) and 56093 (pick 2) are bottom-third at MID; 86925 is not.
        assert sorted(needs["M"]) == [56093, 56094]
        # Your own bottom-third positions are not a rival's need.
        assert 56086 not in needs["G"]
        assert 56086 not in needs["D"]

    def test_no_power_rankings_means_unknown_not_nobody(self):
        order = parse_waiver_order(_details())
        assert rival_needs(None, order, 56086) == {}
        assert rival_needs(pd.DataFrame(), order, 56086) == {}
        assert rival_needs(self._team_df(), order, 56094) == {}   # first pick


class TestClaimOutlook:
    def test_the_best_available_at_a_position_several_rivals_need_is_a_long_shot(self):
        out = claim_outlook("M", pos_rank=1, rivals_needing=4, n_ahead=6)
        assert out["band"] == BAND_LONG_SHOT
        assert "4 of the 6" in out["reason"]

    def test_just_past_the_rivals_is_contested(self):
        assert claim_outlook("M", 5, rivals_needing=4, n_ahead=6)["band"] == BAND_CONTESTED

    def test_deep_enough_to_survive_them_is_likely(self):
        out = claim_outlook("M", 9, rivals_needing=4, n_ahead=6)
        assert out["band"] == BAND_LIKELY

    def test_board_position_catches_competition_from_managers_with_no_need(self):
        """A standout target goes even when nobody ahead is short at his position."""
        out = claim_outlook("M", 9, rivals_needing=0, n_ahead=6,
                            expected_gone=4.6, overall_rank=2)
        assert out["band"] == BAND_CONTESTED
        out = claim_outlook("M", 9, rivals_needing=0, n_ahead=6,
                            expected_gone=4.6, overall_rank=30)
        assert out["band"] == BAND_LIKELY

    def test_the_first_pick_is_never_contested(self):
        assert claim_outlook("M", 1, rivals_needing=0, n_ahead=0)["band"] == BAND_LIKELY

    def test_missing_inputs_render_nothing_rather_than_a_guess(self):
        assert claim_outlook("M", None, 3, 6)["band"] == BAND_UNKNOWN
        assert claim_outlook("M", 1, 3, None)["band"] == BAND_UNKNOWN
        assert claim_outlook("M", None, 3, 6)["reason"] == ""


class TestClaimPlan:
    def test_a_long_shot_is_not_demoted(self):
        """The rule the whole plan rests on: only a *successful* claim costs your slot.

        A manager in GW4 was declined at priorities 2, 3 and 4 and still won at 5.
        So leading with a safe claim buys nothing and forfeits the better player —
        the plan is the honest preference order, gain descending.
        """
        plan = rank_claim_plan([
            {"drop_player": "A", "add_player": "Safe", "transaction_score": 0.10,
             "outlook_band": BAND_LIKELY},
            {"drop_player": "B", "add_player": "Star", "transaction_score": 0.40,
             "outlook_band": BAND_LONG_SHOT},
        ])
        assert [r["add_player"] for r in plan] == ["Star", "Safe"]
        assert [r["claim_priority"] for r in plan] == [1, 2]

    def test_the_same_swap_cannot_be_claimed_twice(self):
        plan = rank_claim_plan([
            {"drop_player": "A", "add_player": "X", "transaction_score": 0.2},
            {"drop_player": "A", "add_player": "X", "transaction_score": 0.9},
        ])
        assert len(plan) == 1

    def test_an_unscoreable_row_sinks_rather_than_raising(self):
        plan = rank_claim_plan([
            {"drop_player": "A", "add_player": "X", "transaction_score": None},
            {"drop_player": "B", "add_player": "Y", "transaction_score": 0.3},
        ])
        assert [r["add_player"] for r in plan] == ["Y", "X"]

    def test_nothing_in_gives_nothing_out(self):
        assert rank_claim_plan([]) == []
        assert rank_claim_plan(None) == []


class TestActionsImportPath:
    def test_the_model_is_importable_without_streamlit(self):
        """The Discord notifier runs in GitHub Actions and installs requirements
        best-effort (`|| true` in fpl-notifications.yml), so the deadline and
        priority code must not need Streamlit at import time.

        `waiver_alerts` already had this constraint and is where a regression
        would land: it is one `from scripts.common.fpl_draft_api import ...` away
        from pulling Streamlit into a workflow that may not have it.
        """
        import pathlib
        import subprocess
        import sys

        repo = pathlib.Path(__file__).resolve().parents[2]
        code = (
            "import sys;"
            "import scripts.common.waiver_priority;"
            "import scripts.common.waiver_alerts;"
            "import scripts.common.data_validation;"
            "assert 'streamlit' not in sys.modules, 'pulled in Streamlit';"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
