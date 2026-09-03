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


# ---------------------------------------------------------------------------
# Suggestion breadth: the page used to hard-code a 2-roster x 5-available window
# and one move per position, so it could never show more than three cards no
# matter how many upgrades were on the board.
# ---------------------------------------------------------------------------

DEEP_ROSTER = [
    {"Player": f"Weak {pos}{i}", "Team": "ARS", "Position": pos, "Points": 1.0 + i,
     "Keep Score": 0.10 + 0.10 * i, "Season_Points": 10 + i}
    for pos in ("G", "D", "M", "F")
    for i in range(3)
]

DEEP_AVAIL = [
    {"Player": f"Star {pos}{i}", "Team": "NEW", "Position": pos, "Points": 7.0 - 0.2 * i,
     "Transfer Score": 0.90 - 0.05 * i, "Draft_State": "a"}
    for pos in ("G", "D", "M", "F")
    for i in range(6)
]


def _exhaustive(avail, roster, **kw):
    """The 'All improvements' view: no candidate window, no one-per-position cap."""
    return _compute_transfer_suggestions(
        avail, roster, top_n=None,
        roster_candidates=None, avail_candidates=None, one_per_position=False,
        **kw
    )


class TestSuggestionBreadth:
    def test_default_view_is_still_one_move_per_position(self):
        """The compact view is what the page shipped with — widening the search
        must not change it."""
        suggestions, _ = _compute_transfer_suggestions(
            _avail(DEEP_AVAIL), _roster(DEEP_ROSTER), top_n=None
        )
        positions = [s["drop_position"] for s in suggestions]
        assert sorted(positions) == ["D", "F", "G", "M"]

    def test_exhaustive_view_surfaces_every_upgradeable_player(self):
        """Nine weak players with a clear upgrade available must all be listed,
        not just the best one at each position."""
        suggestions, _ = _exhaustive(_avail(DEEP_AVAIL), _roster(DEEP_ROSTER))

        assert len(suggestions) == len(DEEP_ROSTER)
        assert len({s["drop_player"] for s in suggestions}) == len(DEEP_ROSTER)

    def test_results_stay_ranked_by_transaction_score(self):
        suggestions, _ = _exhaustive(_avail(DEEP_AVAIL), _roster(DEEP_ROSTER))
        scores = [s["transaction_score"] for s in suggestions]
        assert scores == sorted(scores, reverse=True)

    def test_position_filter_restricts_the_search(self):
        suggestions, debug = _exhaustive(
            _avail(DEEP_AVAIL), _roster(DEEP_ROSTER), positions=["M"]
        )
        assert {s["drop_position"] for s in suggestions} == {"M"}
        assert {d["pos"] for d in debug} == {"M"}

    def test_locked_players_are_still_excluded_when_scanning_the_whole_pool(self):
        """Widening the search must not widen it to unclaimable players."""
        avail = _avail([
            {"Player": "Locked Star", "Team": "HUL", "Position": "M", "Points": 9.0,
             "Transfer Score": 0.99, "Draft_State": "l"},
            {"Player": "Free Agent", "Team": "NEW", "Position": "M", "Points": 6.0,
             "Transfer Score": 0.80, "Draft_State": "a"},
        ])
        suggestions, _ = _exhaustive(avail, _roster(ROSTER))
        assert "Locked Star" not in {s["add_player"] for s in suggestions}

    def test_top_n_caps_the_list_and_none_does_not(self):
        """The 'Top 5' / 'Top 10' views cap the same full search that 'All' shows."""
        everything, _ = _exhaustive(_avail(DEEP_AVAIL), _roster(DEEP_ROSTER))
        capped, _ = _compute_transfer_suggestions(
            _avail(DEEP_AVAIL), _roster(DEEP_ROSTER), top_n=5,
            roster_candidates=None, avail_candidates=None, one_per_position=False,
        )
        assert len(everything) > 5
        assert capped == everything[:5]

    def test_debug_pairs_are_capped_on_a_full_pool_scan(self):
        """The transparency expander renders every pair; an exhaustive scan must
        not turn it into a wall of hundreds of rows."""
        _, debug = _exhaustive(_avail(DEEP_AVAIL), _roster(DEEP_ROSTER))
        assert all(len(d["pairs"]) <= 40 for d in debug)
