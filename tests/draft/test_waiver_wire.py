"""Waiver Wire locked-player handling.

The bug: the page suggested Oliver McBurnie hours after another manager dropped
him. A dropped player is *locked* — on nobody's roster, so an ownership anti-join
calls them available, but unclaimable until the next waiver round processes.

This is not a hypothetical ranking. Against the live league on 2026-08-27 the
single highest-scoring "available" player was locked, as were two others in the
top eleven, so without this filter the top DEF and MID suggestions were both
players the manager could not acquire.
"""

import pandas as pd
import pytest

from scripts.draft.waiver_wire import _compute_transfer_suggestions


def _avail(rows):
    """Available-player frame shaped the way the scoring pipeline leaves it."""
    df = pd.DataFrame(rows)
    df["_effective_proj"] = df["Points"]
    df["MultiGW_Proj"] = df["Points"] * 3
    df["Season_Points"] = 40
    df["Form"] = 4.0
    df["chance_of_playing_next_round"] = 100
    df["status"] = "a"
    df["news"] = ""
    return df


def _roster(rows):
    df = pd.DataFrame(rows)
    df["_effective_proj"] = df["Points"]
    df["MultiGW_Proj"] = df["Points"] * 3
    df["Form"] = 1.0
    df["chance_of_playing_next_round"] = 100
    df["status"] = "a"
    df["news"] = ""
    return df


ROSTER = [
    {"Player": "Weak Mid", "Team": "ARS", "Position": "M", "Points": 1.0,
     "Keep Score": 0.10, "Season_Points": 5},
    {"Player": "Solid Mid", "Team": "ARS", "Position": "M", "Points": 6.0,
     "Keep Score": 0.80, "Season_Points": 90},
]


class TestLockedPlayersAreNeverSuggested:
    def test_locked_top_candidate_is_skipped_for_the_next_available_one(self):
        """The exact live shape: the best candidate is locked, so the second-best
        available one must be suggested instead — not nothing, and not the locked one."""
        avail = _avail([
            {"Player": "Locked Star", "Team": "HUL", "Position": "M", "Points": 6.5,
             "Transfer Score": 0.90, "Draft_State": "l"},
            {"Player": "Free Agent", "Team": "NEW", "Position": "M", "Points": 6.0,
             "Transfer Score": 0.80, "Draft_State": "a"},
        ])

        suggestions, debug = _compute_transfer_suggestions(avail, _roster(ROSTER), top_n=3)

        added = {s["add_player"] for s in suggestions}
        assert "Locked Star" not in added
        assert "Free Agent" in added, "the available fallback should still be suggested"
        assert any(d.get("locked_excluded") == 1 for d in debug)

    def test_no_suggestion_when_every_candidate_is_locked(self):
        """Better to propose nothing than a transfer that cannot be made."""
        avail = _avail([
            {"Player": "Locked Star", "Team": "HUL", "Position": "M", "Points": 6.5,
             "Transfer Score": 0.90, "Draft_State": "l"},
        ])

        suggestions, _ = _compute_transfer_suggestions(avail, _roster(ROSTER), top_n=3)
        assert suggestions == []

    def test_available_players_are_unaffected(self):
        avail = _avail([
            {"Player": "Free Agent", "Team": "NEW", "Position": "M", "Points": 6.0,
             "Transfer Score": 0.80, "Draft_State": "a"},
        ])

        suggestions, debug = _compute_transfer_suggestions(avail, _roster(ROSTER), top_n=3)
        assert [s["add_player"] for s in suggestions] == ["Free Agent"]
        assert all(d.get("locked_excluded") == 0 for d in debug)

    def test_missing_state_column_falls_open(self):
        """If the element-status endpoint is down the column never appears. The
        page must keep working exactly as it did before states existed."""
        avail = _avail([
            {"Player": "Free Agent", "Team": "NEW", "Position": "M", "Points": 6.0,
             "Transfer Score": 0.80},
        ])

        suggestions, _ = _compute_transfer_suggestions(avail, _roster(ROSTER), top_n=3)
        assert [s["add_player"] for s in suggestions] == ["Free Agent"]


class TestDraftStateSurvivesScoring:
    """compute_player_scores() returns a rebuilt frame. If it ever stops carrying
    Draft_State through, the filter above silently stops firing and locked players
    are suggested again — with nothing failing."""

    def test_compute_player_scores_preserves_draft_state(self):
        from scripts.common.analytics import compute_player_scores

        df = _avail([
            {"Player": "P%d" % i, "Team": "ARS", "Position": "M", "Points": 3.0 + i * 0.1,
             "Draft_State": "l" if i == 0 else "a"}
            for i in range(12)
        ])
        df["starts"] = 5
        # compute_player_scores() resolves these by name; supply them so the test
        # exercises the real path rather than its missing-column handling.
        df["AvgFDRNextN"] = 3.0

        scored = compute_player_scores(df, df, current_gw=5, format_context="draft")

        assert "Draft_State" in scored.columns
        assert (scored["Draft_State"] == "l").sum() == 1
