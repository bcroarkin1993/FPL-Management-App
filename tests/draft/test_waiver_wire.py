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


# ---------------------------------------------------------------------------
# Display names. The cards used to print whatever name the source published —
# "Savio Moreira de Oliveira (TOT)" for a player everyone calls Savio.
# ---------------------------------------------------------------------------

from unittest.mock import patch

BOOTSTRAP = {
    "elements": [
        {"id": 403, "first_name": "Sávio", "second_name": "Moreira de Oliveira",
         "web_name": "Sávio"},
        {"id": 10, "first_name": "Bruno", "second_name": "Borges Fernandes",
         "web_name": "B.Fernandes"},
        # Two players sharing a surname: the name-only fallback must refuse both
        # rather than print one player's name on the other's card.
        {"id": 20, "first_name": "Cole", "second_name": "Palmer", "web_name": "Palmer"},
        {"id": 21, "first_name": "Alex", "second_name": "Palmer", "web_name": "Palmer"},
    ]
}


def _with_bootstrap(fn, *args, **kwargs):
    """Run `fn` against a stubbed bootstrap.

    `_display_name_maps()` is `st.cache_data`-wrapped in the app; the test
    conftest replaces that with a pass-through, but clear the cache anyway if it
    is real so a memoized map from another test can't leak in.
    """
    from scripts.draft import waiver_wire as ww

    clear = getattr(ww._display_name_maps, "clear", None)
    if clear:
        clear()
    try:
        with patch.object(ww, "_load_bootstrap", return_value=BOOTSTRAP):
            return fn(*args, **kwargs)
    finally:
        if clear:
            clear()


class TestDisplayNames:
    def test_legal_name_renders_as_the_common_name(self):
        from scripts.draft.waiver_wire import _attach_display_names

        df = pd.DataFrame([
            {"Player": "Sávio Moreira de Oliveira", "Player_ID": 403, "Team": "TOT"},
        ])
        out = _with_bootstrap(_attach_display_names, df)

        assert out.loc[0, "Display_Name"] == "Sávio"
        # Player is what every merge on this page keys on — it must not change.
        assert out.loc[0, "Player"] == "Sávio Moreira de Oliveira"

    def test_resolves_by_name_when_the_element_id_is_missing(self):
        from scripts.draft.waiver_wire import _attach_display_names

        df = pd.DataFrame([{"Player": "Bruno Borges Fernandes", "Team": "MUN"}])
        out = _with_bootstrap(_attach_display_names, df)
        assert out.loc[0, "Display_Name"] == "Bruno Fernandes"

    def test_ambiguous_surname_does_not_borrow_another_players_name(self):
        """The display-side Alex/Cole Palmer trap: a shared key must resolve to
        nothing, not to whichever player was seen first."""
        from scripts.draft.waiver_wire import _attach_display_names

        df = pd.DataFrame([{"Player": "Palmer", "Team": "CHE"}])
        out = _with_bootstrap(_attach_display_names, df)
        assert out.loc[0, "Display_Name"] == "Palmer"

    def test_unknown_player_falls_back_to_the_source_name(self):
        from scripts.draft.waiver_wire import _attach_display_names

        df = pd.DataFrame([{"Player": "Rotowire Only Guy", "Team": "XXX"}])
        out = _with_bootstrap(_attach_display_names, df)
        assert out.loc[0, "Display_Name"] == "Rotowire Only Guy"

    def test_suggestion_cards_use_the_display_name(self):
        """The end-to-end point of all of the above."""
        roster = _roster([
            {"Player": "Sávio Moreira de Oliveira", "Team": "TOT", "Position": "M",
             "Points": 1.0, "Keep Score": 0.10, "Season_Points": 5, "Display_Name": "Sávio"},
            {"Player": "Solid Mid", "Team": "ARS", "Position": "M", "Points": 6.0,
             "Keep Score": 0.80, "Season_Points": 90, "Display_Name": "Solid Mid"},
        ])
        avail = _avail([
            {"Player": "Bruno Borges Fernandes", "Team": "MUN", "Position": "M",
             "Points": 6.0, "Transfer Score": 0.80, "Draft_State": "a",
             "Display_Name": "Bruno Fernandes"},
        ])

        suggestions, debug = _compute_transfer_suggestions(avail, roster, top_n=3)

        assert suggestions[0]["drop_player"] == "Savio"
        assert suggestions[0]["add_player"] == "Bruno Fernandes"
        assert debug[2]["pairs"][0]["drop"] == "Savio"

    def test_display_name_survives_scoring(self):
        """compute_player_scores() rebuilds the frame; if it stopped carrying
        Display_Name the cards would silently revert to legal names."""
        from scripts.common.analytics import compute_player_scores

        df = _avail([
            {"Player": "P%d" % i, "Team": "ARS", "Position": "M", "Points": 3.0 + i * 0.1,
             "Display_Name": "Nick %d" % i}
            for i in range(12)
        ])
        df["starts"] = 5
        df["AvgFDRNextN"] = 3.0

        scored = compute_player_scores(df, df, current_gw=5, format_context="draft")
        assert "Display_Name" in scored.columns


class TestOneAddCannotBeSuggestedTwice:
    """A suggestion list has to be a set of moves you can actually make.

    Reported from the app: with several weak defenders, the same available
    player was offered against each of them. You can add him once. The list read
    as a plan but was really the same move written three times.
    """

    def _roster(self):
        # Three droppable defenders of increasing value, plus a fourth so the
        # position is never reduced below a legal squad.
        return pd.DataFrame({
            "Player": ["Weakest", "Middle", "Strongest", "Anchor"],
            "Team": ["AVL", "MCI", "BRE", "ARS"],
            "Position": ["D"] * 4,
            "Keep Score": [0.05, 0.25, 0.40, 0.90],
            "Season_Points": [0, 10, 20, 60],
            "Form": [0.0, 1.0, 2.0, 5.0],
            "_effective_proj": [0.0, 2.0, 3.0, 5.0],
            "MultiGW_Proj": [0.0, 6.0, 9.0, 15.0],
            "chance_of_playing_next_round": [None, None, None, None],
            "status": ["a", "a", "a", "a"],
            "news": ["", "", "", ""],
        })

    def _avail(self):
        return pd.DataFrame({
            "Player": ["Best Target", "Second Target", "Third Target"],
            "Team": ["CRY", "EVE", "FUL"],
            "Position": ["D"] * 3,
            "Transfer Score": [0.80, 0.70, 0.60],
            "Season_Points": [30, 25, 20],
            "Form": [5.0, 4.0, 3.0],
            "_effective_proj": [5.0, 4.0, 3.5],
            "Points": [5.0, 4.0, 3.5],
            "MultiGW_Proj": [15.0, 12.0, 10.0],
            "chance_of_playing_next_round": [None, None, None],
            "status": ["a", "a", "a"],
            "news": ["", "", ""],
        })

    def _run(self):
        return _compute_transfer_suggestions(
            self._avail(), self._roster(), top_n=None, positions=["D"],
            roster_candidates=None, avail_candidates=None, one_per_position=False,
        )[0]

    def test_no_add_appears_more_than_once(self):
        adds = [s["add_player"] for s in self._run()]
        assert len(adds) == len(set(adds)), f"duplicate adds: {adds}"

    def test_no_drop_appears_more_than_once(self):
        drops = [s["drop_player"] for s in self._run()]
        assert len(drops) == len(set(drops)), f"duplicate drops: {drops}"

    def test_the_best_target_goes_to_the_player_you_most_want_to_replace(self):
        """Gain is add - drop, so for a fixed add it is largest against the
        weakest drop. That is the behaviour asked for, and it falls out of
        sorting by gain rather than needing a special case."""
        by_drop = {s["drop_player"]: s["add_player"] for s in self._run()}
        assert by_drop["Weakest"] == "Best Target"

    def test_a_losing_drop_still_gets_its_next_best_alternative(self):
        """Losing the contested target must not remove the drop from the list --
        it still wants replacing, just with someone else."""
        by_drop = {s["drop_player"]: s["add_player"] for s in self._run()}
        assert by_drop.get("Middle") == "Second Target"

    def test_compact_view_still_returns_a_single_move_per_position(self):
        out = _compute_transfer_suggestions(
            self._avail(), self._roster(), top_n=None, positions=["D"],
            roster_candidates=2, avail_candidates=5, one_per_position=True,
        )[0]
        assert len(out) == 1
        assert out[0]["drop_player"] == "Weakest"
        assert out[0]["add_player"] == "Best Target"

    def test_debug_rows_separate_clearing_the_bar_from_being_recommended(self):
        """An add can clear a drop's threshold and still lose it to a stronger
        pairing. The transparency expander has to show which happened."""
        _, debug = _compute_transfer_suggestions(
            self._avail(), self._roster(), top_n=None, positions=["D"],
            roster_candidates=None, avail_candidates=None, one_per_position=False,
        )
        pairs = debug[0]["pairs"]
        assert any(p["passed"] and not p["assigned"] for p in pairs), (
            "expected at least one pair that cleared its threshold but lost the "
            "add to a better pairing"
        )


class TestClaimReachabilityIsStampedInsideTheSearch:
    """A suggested claim you cannot win at your waiver priority is not a plan.

    The outlook is computed inside `_compute_transfer_suggestions()`, not at the
    callsite, for the same reason the locked filter above is: a caller that
    forgets it renders a page that looks identical and quietly recommends players
    six other managers see first.
    """

    def _avail_pair(self):
        return _avail([
            {"Player": "Top Mid", "Team": "NEW", "Position": "M", "Points": 6.5,
             "Transfer Score": 0.90, "Draft_State": "a"},
            {"Player": "Deep Mid", "Team": "BUR", "Position": "M", "Points": 6.0,
             "Transfer Score": 0.72, "Draft_State": "a"},
        ])

    def test_the_top_of_a_position_the_league_shops_at_is_a_long_shot(self):
        from scripts.common.waiver_priority import BAND_LONG_SHOT
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            # 6 managers ahead x 0.40 midfielders each = 2.4 expected gone.
            waiver_context={"n_ahead": 6, "claim_rates": {"M": 0.40},
                            "rival_needs": {"M": [1, 2, 3, 4]},
                            "expected_gone": 4.6},
        )
        top = next(s for s in suggestions if s["add_player"] == "Top Mid")
        assert top["outlook_band"] == BAND_LONG_SHOT
        assert "2.4 midfielders" in top["outlook_reason"]

    def test_the_band_comes_from_the_rates_and_not_from_rival_weakness(self):
        """Four rivals short at a position they never claim at is not contention.

        This is the over-call the page was reported for: `rival_needs` alone made
        the top four forwards long shots in a league whose log shows forwards are
        18 of 87 accepted claims.
        """
        from scripts.common.waiver_priority import BAND_LIKELY
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "claim_rates": {"M": 0.01},
                            "rival_needs": {"M": [1, 2, 3, 4]},
                            "expected_gone": 0.0},
        )
        assert {s["outlook_band"] for s in suggestions} == {BAND_LIKELY}

    def test_a_deeper_target_at_the_same_position_survives_them(self):
        from scripts.common.waiver_priority import BAND_LIKELY
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "claim_rates": {"M": 0.0},
                            "rival_needs": {"M": []}, "expected_gone": 0.0},
        )
        assert {s["outlook_band"] for s in suggestions} == {BAND_LIKELY}

    def test_a_league_with_no_measured_rates_is_not_banded(self):
        """Power Rankings alone is not evidence about a rival's intentions."""
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "claim_rates": {},
                            "rival_needs": {"M": [1, 2, 3, 4]},
                            "expected_gone": None},
        )
        assert suggestions
        assert all(s["outlook_band"] == "unknown" for s in suggestions)

    def test_no_waiver_context_leaves_the_band_blank_rather_than_neutral(self):
        """The order or the power rankings being unavailable is not a finding.

        A grey "Unknown" badge on every card is noise; the pill renders nothing.
        """
        from scripts.draft.waiver_wire import _compute_transfer_suggestions, _outlook_pill

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
        )
        assert suggestions
        assert all(s["outlook_band"] == "" for s in suggestions)
        assert _outlook_pill("") == ""

    def test_ranks_come_from_the_ordering_the_search_itself_walks(self):
        """The outlook must not disagree with the list it annotates.

        Position rank is taken from `avail_sorted` — the same injury-adjusted
        ordering the search iterates — so the "#1 available" the badge refers to
        is the player the search actually considered first.
        """
        from scripts.common.waiver_priority import BAND_LONG_SHOT
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        avail = self._avail_pair()
        # Make the nominally weaker player the best candidate *by the ordering
        # the search walks*. This used to be done with a 25% chance of playing,
        # back when availability was multiplied into `_adj_value`; that second
        # discount is gone -- it double-charged what `Transfer Score` already
        # prices -- so the divergence is created where it actually lives now,
        # between the score and the raw projection.
        avail.loc[avail["Player"] == "Top Mid", "Transfer Score"] = 0.60
        avail.loc[avail["Player"] == "Deep Mid", "Transfer Score"] = 0.88

        suggestions, _ = _compute_transfer_suggestions(
            avail, _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "claim_rates": {"M": 0.40},
                            "rival_needs": {"M": [1]}, "expected_gone": 4.6},
        )
        deep = next(s for s in suggestions if s["add_player"] == "Deep Mid")
        assert deep["outlook_band"] == BAND_LONG_SHOT, "he is now the #1 adjusted target"


class TestFreeAgencyFraming:
    """What the page says once the waiver round has processed.

    Free agency is a different mechanic: an available player is simply added, so
    there is no queue, no priority to spend and nothing to be beaten to. Every
    line that describes a queue has to stop describing one -- and the single
    exception has to be named, because a player dropped this week is locked and
    is the one way an "available" player still cannot be had today.
    """

    STATS = {
        "events_observed": 8,
        "recent_events": [5, 6, 7, 8],
        "participation_rate": 0.713,
        "participation_rate_recent": 0.63,
        "contested_share": 0.41,
        "max_contention": 8,
        "position_claim_rates": {"G": 0.025, "D": 0.25, "M": 0.475, "F": 0.1},
    }

    def _order(self, n=10):
        return [{"pick": i, "entry_id": 56085 + i, "league_entry_id": 56181 + i,
                 "team_name": "Team %d" % i} for i in range(1, n + 1)]

    def _suggestions(self):
        """Real suggestion payloads, from the real search.

        Hand-built dicts drift out of step with `_build_suggestion()` and then
        fail on a missing key rather than on the thing under test.
        """
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        avail = _avail([
            {"Player": "Top Mid", "Team": "NEW", "Position": "M", "Points": 6.5,
             "Transfer Score": 0.90, "Draft_State": "a"},
            {"Player": "Second Mid", "Team": "LIV", "Position": "M", "Points": 6.0,
             "Transfer Score": 0.85, "Draft_State": "a"},
        ])
        # Two droppable players, so two moves assign and the claim plan renders
        # (it returns early below two).
        roster = _roster(ROSTER + [
            {"Player": "Spare Mid", "Team": "BUR", "Position": "M", "Points": 1.2,
             "Keep Score": 0.12, "Season_Points": 8},
        ])
        suggestions, _ = _compute_transfer_suggestions(
            avail, roster, top_n=None, one_per_position=False,
            roster_candidates=None, avail_candidates=None)
        assert len(suggestions) >= 2, suggestions
        return suggestions

    def _captions(self, mock_streamlit):
        return " ".join(
            str(c.args[0]) if c.args else ""
            for c in mock_streamlit["caption"].call_args_list
        )

    @staticmethod
    def _expander_titles():
        """`mock_streamlit["expander"]` is the patcher, not the started mock."""
        from unittest.mock import MagicMock, patch

        spy = MagicMock()
        spy.return_value.__enter__ = MagicMock(return_value=None)
        spy.return_value.__exit__ = MagicMock(return_value=None)
        return patch("streamlit.expander", spy), spy

    def test_the_cards_say_a_dropped_player_is_still_locked(self, mock_streamlit):
        """The one exception to "adds process immediately"."""
        import scripts.draft.waiver_wire as ww

        ww._render_transfer_suggestions(self._suggestions(), current_gw=8,
                                        total_found=2, free_agency=True,
                                        waiver_gw=9)
        text = self._captions(mock_streamlit)
        assert "dropped this week is locked" in text
        assert "GW9 waiver round" in text

    def test_the_cards_say_nothing_about_free_agency_during_waivers(self, mock_streamlit):
        import scripts.draft.waiver_wire as ww

        ww._render_transfer_suggestions(self._suggestions(), current_gw=8,
                                        total_found=2, free_agency=False)
        assert "Free agency" not in self._captions(mock_streamlit)

    def test_the_claim_plan_becomes_a_pickup_order(self, mock_streamlit):
        """Presenting it as a ranked *request* list would describe a dead mechanic."""
        import scripts.draft.waiver_wire as ww

        sugg = self._suggestions()
        pat, spy = self._expander_titles()
        with pat:
            ww._render_claim_plan(sugg, free_agency=True)
        assert any("pickup order" in c.args[0] for c in spy.call_args_list)
        text = self._captions(mock_streamlit)
        assert "no waiver priority to spend" in text
        assert "failed claim" not in text

    def test_the_claim_plan_keeps_its_waiver_framing_during_waivers(self, mock_streamlit):
        import scripts.draft.waiver_wire as ww

        sugg = self._suggestions()
        pat, spy = self._expander_titles()
        with pat:
            ww._render_claim_plan(sugg, free_agency=False)
        assert any("claim order" in c.args[0] for c in spy.call_args_list)
        assert "failed claim" in self._captions(mock_streamlit)

    def test_the_queue_is_labelled_as_the_next_round(self, mock_streamlit):
        """`waiver_pick` is always the next round's order, never today's."""
        import scripts.draft.waiver_wire as ww

        pat, spy = self._expander_titles()
        with pat:
            ww._render_waiver_queue(self._order(), 56091, self.STATS,
                                    free_agency=True)
        assert any("Next waiver round" in c.args[0] for c in spy.call_args_list)
        assert "applies to the **next** waiver round" in self._captions(mock_streamlit)

    def test_the_queue_quotes_the_recent_window_and_the_position_split(
            self, mock_streamlit):
        """The figures the bands are actually calibrated on.

        63% over the last four rounds, not 71% over the season, and the
        per-position expectation that replaced the rival-need inference.
        """
        import scripts.draft.waiver_wire as ww

        ww._render_waiver_queue(self._order(), 56091, self.STATS)
        text = self._captions(mock_streamlit)
        assert "the last 4 waiver rounds" in text
        assert "63% of managers claimed" in text
        # 5 managers ahead of pick 6: MID 2.4, GK 0.1.
        assert "MID 2.4" in text and "GK 0.1" in text

    def test_the_window_card_names_the_locked_exception(self, mock_streamlit):
        import scripts.draft.waiver_wire as ww

        ww._render_transaction_window({"mode": "free-agency", "next_event": 9},
                                      locked_count=3, waiver_order=self._order(),
                                      my_entry_id=56091)
        html = " ".join(str(c.args[0]) if c.args else ""
                        for c in mock_streamlit["markdown"].call_args_list)
        assert "except players just dropped" in html
        # The waiver pick is next round's, and saying so is the point.
        assert "Next round" in html


class TestWaiverContextDegradesQuietly:
    def test_an_unknown_order_disables_the_outlook(self):
        from scripts.draft.waiver_wire import _build_waiver_context

        assert _build_waiver_context([], 56086, 5) == {}
        assert _build_waiver_context(None, 56086, 5) == {}

    def test_a_manager_absent_from_the_order_disables_it(self):
        """The two-id-space trap: matching on the standings id finds nothing."""
        from scripts.draft.waiver_wire import _build_waiver_context

        order = [{"pick": 1, "entry_id": 56094, "league_entry_id": 56190,
                  "team_name": "Starboys"}]
        assert _build_waiver_context(order, 56190, 5) == {}

    def test_the_first_pick_needs_no_power_rankings(self):
        """Nothing ahead of you can take anyone, so no network call is warranted."""
        from unittest.mock import patch
        import scripts.draft.waiver_wire as ww

        order = [{"pick": 1, "entry_id": 56094, "league_entry_id": 56190,
                  "team_name": "Starboys"},
                 {"pick": 2, "entry_id": 56086, "league_entry_id": 56182,
                  "team_name": "Stoned Squirrels"}]
        with patch.object(ww, "_waiver_claim_stats", side_effect=AssertionError):
            ctx = ww._build_waiver_context(order, 56094, 5)
        assert ctx == {"n_ahead": 0, "rival_needs": {}, "claim_rates": {},
                       "expected_gone": 0.0}

    def test_free_agency_disables_the_outlook_entirely(self):
        """There is no queue to be beaten in, so no claim is a long shot.

        An available player is simply added, so every band would be a statement
        about a mechanic that is not running. An empty context is the whole
        suppression: the pill, the reason line and the claim-plan parenthetical
        all key off it.
        """
        from unittest.mock import patch
        import scripts.draft.waiver_wire as ww

        order = [{"pick": 1, "entry_id": 56094, "league_entry_id": 56190,
                  "team_name": "Starboys"},
                 {"pick": 2, "entry_id": 56086, "league_entry_id": 56182,
                  "team_name": "Stoned Squirrels"}]
        # Neither the claim history nor Power Rankings should be reached for --
        # free agency is also the window that should cost the page the least.
        with patch.object(ww, "_waiver_claim_stats", side_effect=AssertionError), \
                patch("scripts.common.team_strength.build_league_strength",
                      side_effect=AssertionError):
            assert ww._build_waiver_context(
                order, 56086, 5, transaction_mode="free-agency") == {}

    def test_an_unreadable_window_keeps_the_outlook(self):
        """An outage must not be indistinguishable from free agency.

        `/league/{id}/details` failing leaves `mode` None, and suppressing on
        that would silently delete the feature for the duration.
        """
        import scripts.draft.waiver_wire as ww

        order = [{"pick": 1, "entry_id": 56094, "league_entry_id": 56190,
                  "team_name": "Starboys"},
                 {"pick": 2, "entry_id": 56086, "league_entry_id": 56182,
                  "team_name": "Stoned Squirrels"}]
        ctx = ww._build_waiver_context(order, 56094, 5, transaction_mode=None)
        assert ctx.get("n_ahead") == 0


class TestClaimsAreRankedInPoints:
    """A percentile gap is not the same quantity at two positions.

    Measured on the live GW6 pool, a 0.10 gain in Transfer/Keep Score was worth
    0.091 expected points at goalkeeper and 0.405 at forward -- a 4.4x spread,
    because the positions differ in depth and in scoring range. The suggestion
    list and the claim plan both ranked across positions on that gap, so a
    goalkeeper move and a forward move that looked equal were not. Since waiver
    processing stops at your first success, mis-ordering the plan costs you the
    claim you most wanted.
    """

    @staticmethod
    def _frames():
        """A GK swap with the larger percentile gap and the smaller points gain.

        The numbers are the live shape: goalkeepers are packed into a narrow
        scoring band, so a big move in rank buys very few points.
        """
        avail = _avail([
            {"Player": "Better Keeper", "Team": "NEW", "Position": "G", "Points": 3.6,
             "Transfer Score": 0.92, "Draft_State": "a"},
            {"Player": "Better Forward", "Team": "BUR", "Position": "F", "Points": 5.2,
             "Transfer Score": 0.74, "Draft_State": "a"},
        ])
        roster = _roster([
            {"Player": "My Keeper", "Team": "EVE", "Position": "G", "Points": 3.2,
             "Keep Score": 0.40, "Season_Points": 20},
            {"Player": "Spare Keeper", "Team": "AVL", "Position": "G", "Points": 2.9,
             "Keep Score": 0.35, "Season_Points": 18},
            {"Player": "My Forward", "Team": "WOL", "Position": "F", "Points": 2.6,
             "Keep Score": 0.45, "Season_Points": 22},
            {"Player": "Spare Forward", "Team": "BHA", "Position": "F", "Points": 2.4,
             "Keep Score": 0.42, "Season_Points": 21},
        ])
        return avail, roster

    def _suggestions(self):
        avail, roster = self._frames()
        suggestions, _ = _compute_transfer_suggestions(
            avail, roster, top_n=None, one_per_position=True,
            roster_candidates=None, avail_candidates=None,
        )
        return suggestions

    def test_the_bigger_points_move_leads_even_on_a_smaller_score_gap(self):
        suggestions = self._suggestions()
        assert len(suggestions) == 2
        lead, second = suggestions[0], suggestions[1]
        assert lead["add_player"] == "Better Forward"
        # ...and it leads *despite* the goalkeeper move looking better on the
        # currency the old ordering used.
        assert second["transaction_score"] > lead["transaction_score"]
        assert lead["points_gain"] > second["points_gain"]

    def test_the_claim_plan_uses_the_same_order(self):
        """The plan is what gets typed into FPL, in priority order."""
        from scripts.common.waiver_priority import rank_claim_plan

        plan = rank_claim_plan(self._suggestions())
        assert [p["add_player"] for p in plan] == ["Better Forward", "Better Keeper"]
        assert [p["claim_priority"] for p in plan] == [1, 2]

    def test_the_three_gameweek_window_is_preferred_to_one(self):
        """A claim is a lasting roster change, not a one-week rental."""
        from scripts.draft.waiver_wire import _claim_points_rate

        row = {"_effective_proj": 2.0, "Proj_Next3": 12.0}
        assert _claim_points_rate(row) == pytest.approx(4.0)

    def test_it_falls_back_to_the_single_gameweek_then_to_zero(self):
        from scripts.draft.waiver_wire import _claim_points_rate

        assert _claim_points_rate({"_effective_proj": 2.5}) == pytest.approx(2.5)
        assert _claim_points_rate({}) == 0.0

    def test_no_projections_anywhere_degrades_to_the_old_ordering(self):
        """A dead feed must reorder nothing, not order arbitrarily."""
        from scripts.common.waiver_priority import rank_claim_plan

        rows = [
            {"drop_player": "a", "add_player": "b", "transaction_score": 0.05},
            {"drop_player": "c", "add_player": "d", "transaction_score": 0.20},
        ]
        assert [r["add_player"] for r in rank_claim_plan(rows)] == ["d", "b"]

    def test_availability_is_not_charged_twice(self):
        """`Proj` is expected points -- start probability is already inside it.

        The percentile side applies its own availability multiplier; doing it
        again on the points side is the double-discount this codebase keeps
        paying for.
        """
        from scripts.draft.waiver_wire import _claim_points_rate

        doubtful = {"_effective_proj": 3.0, "chance_of_playing_next_round": 25,
                    "status": "d"}
        assert _claim_points_rate(doubtful) == pytest.approx(3.0)


class TestTheHoldFactorReadsTheCalendar:
    """Drop protection is bucketed by absence *duration*.

    `days // 7` overstates that across an international break, and the buckets
    turn the overstatement straight into less protection: a player back next
    gameweek falls from 0.70 to 0.40, which is the page telling you to drop
    someone it should be telling you to keep. Live at GW6, 19 players cross a
    bucket -- twelve of them from 0.70 to 1.00, because they miss nothing.
    """

    @staticmethod
    def _deadlines():
        from datetime import datetime, timedelta
        base = datetime.now() + timedelta(days=14)
        return [base, base + timedelta(days=7)]

    def test_back_before_the_next_deadline_keeps_full_protection(self):
        from datetime import datetime, timedelta
        from scripts.draft.waiver_wire import _roster_injury_factor

        back = (datetime.now() + timedelta(days=13)).strftime("%d %b")
        news = "Knock - Expected back %s" % back
        assert _roster_injury_factor(0, "i", news, 0.0,
                                     deadlines=self._deadlines()) == 1.0
        # Without the calendar the same player reads as a two-gameweek absence.
        assert _roster_injury_factor(0, "i", news, 0.0) < 1.0

    def test_a_fit_player_never_reads_the_calendar(self):
        """It is a cached HTTP fetch, and every offline test would pay for it."""
        from scripts.draft.waiver_wire import _roster_injury_factor

        def _boom():
            raise AssertionError("the calendar was read for a fit player")

        assert _roster_injury_factor(100, "a", "", 0.5, deadlines=_boom) == 1.0

    def test_a_callable_is_resolved_for_a_doubtful_one(self):
        from scripts.draft.waiver_wire import _roster_injury_factor

        calls = []

        def _deadlines():
            calls.append(1)
            return self._deadlines()

        _roster_injury_factor(25, "d", "Knock", 0.5, deadlines=_deadlines)
        assert calls == [1]


class TestAvailabilityIsChargedOnce:
    """`Transfer Score` already prices availability; multiplying again charged it twice.

    1GW is a percentile of the engine's `Proj` = `Proj_Start x Start_Pct`, and
    `Start_Pct` carries FPL's availability ceiling. So `Transfer Score x
    _availability_multiplier` discounted a 75%-doubtful player once inside the
    score and again on top of it -- the double-charge this codebase keeps paying
    for, in the one place it had not been looked at.
    """

    @staticmethod
    def _avail(rows):
        return _avail(rows)

    def test_a_doubtful_player_keeps_his_score(self):
        avail = _avail([
            {"Player": "Doubtful", "Team": "NEW", "Position": "M", "Points": 7.0,
             "Transfer Score": 0.90, "Draft_State": "a"},
        ])
        avail["chance_of_playing_next_round"] = 50
        avail["status"] = "d"
        suggestions, debug = _compute_transfer_suggestions(
            avail, _roster(ROSTER), top_n=None,
            roster_candidates=None, avail_candidates=None, one_per_position=False)
        assert suggestions, "a doubtful player is still a candidate"
        # 0.90, not 0.90 x 0.50: the doubt is already inside the score.
        assert suggestions[0]["add_value"] == pytest.approx(0.90)

    def test_a_player_who_cannot_play_is_excluded_not_discounted(self):
        """Unavailability is an eligibility question, like a locked player.

        The multiplier already made this an exclusion in effect -- `_adj_value`
        of 0 makes the gap negative, so he could never clear a threshold -- so
        this states the rule rather than changing a recommendation.
        """
        avail = _avail([
            {"Player": "Injured Star", "Team": "NEW", "Position": "M", "Points": 9.0,
             "Transfer Score": 0.99, "Draft_State": "a"},
        ])
        avail["status"] = "i"
        avail["chance_of_playing_next_round"] = 0
        suggestions, debug = _compute_transfer_suggestions(
            avail, _roster(ROSTER), top_n=None,
            roster_candidates=None, avail_candidates=None, one_per_position=False)
        assert not [s for s in suggestions if s["add_player"] == "Injured Star"]
        assert any(r.get("unavailable_excluded") for r in debug)


class TestTheHoldFactorCannotInventValue:
    """It multiplies a percentile Keep Score, so above 1.0 it manufactures value.

    Reachable the moment return dates were counted against the real calendar: a
    player back before the next deadline scores `gws_to_miss == 0`, which pairs
    a `duration_factor` of 1.0 with a quality boost of up to 0.25. Twelve live
    players were sitting at 1.25 and 1.125.
    """

    @staticmethod
    def _deadlines():
        """A fortnight's break, then weekly for the rest of the season.

        The list has to be long: `gameweeks_until` counts the deadlines it is
        given, so a truncated one understates a long absence — which is correct
        at season end, where the gameweeks really have run out, and misleading in
        a fixture meant to represent mid-season.
        """
        from datetime import datetime, timedelta
        base = datetime.now() + timedelta(days=14)
        return [base + timedelta(days=7 * i) for i in range(30)]

    def test_an_elite_player_back_immediately_is_capped_at_one(self):
        from datetime import datetime, timedelta
        from scripts.draft.waiver_wire import _roster_injury_factor

        back = (datetime.now() + timedelta(days=13)).strftime("%d %b")
        factor = _roster_injury_factor(0, "i", "Knock - Expected back %s" % back, 1.0,
                                       deadlines=self._deadlines())
        assert factor == 1.0

    def test_a_real_absence_still_discounts(self):
        from scripts.draft.waiver_wire import _roster_injury_factor

        assert _roster_injury_factor(
            0, "i", "ACL - Expected back 20 May", 0.0,
            deadlines=self._deadlines()) < 0.5


class TestTheDefaultViewDoesNotHideMoves:
    """Reported from the app: "default should not be Best per Position, since
    that misses a lot of valuable transfers to make".

    It is right, and measurably so. The compact view tries only the 2 weakest
    roster players at each position against the 5 best available, then emits a
    single move per position -- so a second good move at a position someone has
    already claimed vanishes, and vanishes *silently*, because a player who was
    never evaluated produces no debug row and so reads exactly like one that
    was evaluated and rejected. Measured on the live GW9 pool it showed 2 moves
    where the full scan found 5.
    """

    @staticmethod
    def _three_weak_mids():
        return _roster([
            {"Player": "Weak Mid A", "Team": "ARS", "Position": "M",
             "Points": 1.0, "Keep Score": 0.10, "Season_Points": 5},
            {"Player": "Weak Mid B", "Team": "ARS", "Position": "M",
             "Points": 1.2, "Keep Score": 0.15, "Season_Points": 6},
            {"Player": "Weak Mid C", "Team": "ARS", "Position": "M",
             "Points": 1.4, "Keep Score": 0.20, "Season_Points": 7},
        ])

    @staticmethod
    def _three_good_mids():
        return _avail([
            {"Player": "Good Mid A", "Team": "NEW", "Position": "M",
             "Points": 6.0, "Transfer Score": 0.90, "Draft_State": "a"},
            {"Player": "Good Mid B", "Team": "LIV", "Position": "M",
             "Points": 5.8, "Transfer Score": 0.85, "Draft_State": "a"},
            {"Player": "Good Mid C", "Team": "MCI", "Position": "M",
             "Points": 5.6, "Transfer Score": 0.80, "Draft_State": "a"},
        ])

    def test_the_compact_view_shows_one_move_and_hides_two(self):
        sugg, _ = _compute_transfer_suggestions(
            self._three_good_mids(), self._three_weak_mids(), top_n=None,
            roster_candidates=2, avail_candidates=5, one_per_position=True)
        assert len(sugg) == 1, (
            "this is the narrow view working as designed -- and why it is no "
            "longer the default"
        )

    def test_the_full_scan_finds_all_three(self):
        sugg, _ = _compute_transfer_suggestions(
            self._three_good_mids(), self._three_weak_mids(), top_n=None,
            roster_candidates=None, avail_candidates=None,
            one_per_position=False)
        assert len(sugg) == 3, (
            "every weak midfielder has a distinct upgrade available, and all "
            "three are moves the manager could actually make"
        )
        adds = {s["add_player"] for s in sugg}
        drops = {s["drop_player"] for s in sugg}
        assert len(adds) == 3 and len(drops) == 3, (
            "no player may be added or dropped twice -- the list has to be a "
            "set of moves that can all be made together"
        )


class TestTheDefaultView:
    def test_the_default_is_not_the_compact_view(self):
        from scripts.draft import waiver_wire as ww
        assert ww.SUGGESTION_VIEW_DEFAULT != ww.SUGGESTION_VIEW_BEST_PER_POS

    def test_the_default_is_a_real_option(self):
        from scripts.draft import waiver_wire as ww
        assert ww.SUGGESTION_VIEW_DEFAULT in ww.SUGGESTION_VIEW_OPTIONS

    def test_the_default_is_capped_so_the_list_stays_readable(self):
        """A full scan can return one move per droppable roster player."""
        from scripts.draft import waiver_wire as ww
        assert ww.SUGGESTION_VIEW_LIMITS.get(ww.SUGGESTION_VIEW_DEFAULT) == 10

    def test_the_selectbox_defaults_to_it_rather_than_to_index_zero(self):
        """A mocked selectbox hands back its first option whatever the index,
        so no page test can catch a regression to `index=0` here."""
        import inspect
        from scripts.draft import waiver_wire as ww
        src = inspect.getsource(ww.show_waiver_wire_page)
        assert "SUGGESTION_VIEW_OPTIONS.index(SUGGESTION_VIEW_DEFAULT)" in src, (
            "the Show selectbox must resolve its index from the default "
            "constant, not hard-code one"
        )


class TestWaiverDisplayNamesUseTheRightSpace:
    def test_the_map_is_built_from_the_classic_bootstrap(self):
        """`Player_ID` on every Waiver Wire frame comes from
        `pull_fpl_player_stats()`, i.e. the Classic space, so the display map
        has to be keyed the same way. Reading the Draft bootstrap here labelled
        a tenth of the pool with another player's name."""
        import inspect
        from scripts.draft import waiver_wire as ww
        src = inspect.getsource(ww._display_name_maps)
        assert "get_classic_bootstrap_static" in src
        assert "_load_bootstrap" not in src, (
            "_load_bootstrap is the Draft API, whose ids disagree with "
            "Player_ID on these frames"
        )


class TestEverySuggestionIsPositionLocked:
    """FPL's rule, not ours: a claim replaces a squad player with an unselected
    one at the *same* position.

    Asked from the app why the page did not suggest dropping Bobby Thomas (a
    Coventry defender) for Mathys Tel (a Spurs midfielder). It cannot: that
    swap is not submittable.

    Note each fixture carries **two** players per position on the roster and at
    least one available player at the position under test. A position with one
    rostered player, or none available, is skipped wholesale -- so a naive
    fixture returns an empty list and the test passes without ever exercising
    the rule it claims to pin.
    """

    @staticmethod
    def _roster_both_positions():
        return _roster([
            {"Player": "Weak Def", "Team": "COV", "Position": "D",
             "Points": 1.2, "Keep Score": 0.15, "Season_Points": 6},
            {"Player": "Spare Def", "Team": "COV", "Position": "D",
             "Points": 3.0, "Keep Score": 0.55, "Season_Points": 30},
            {"Player": "Weak Mid", "Team": "ARS", "Position": "M",
             "Points": 1.0, "Keep Score": 0.10, "Season_Points": 5},
            {"Player": "Spare Mid", "Team": "ARS", "Position": "M",
             "Points": 3.2, "Keep Score": 0.58, "Season_Points": 32},
        ])

    def test_an_elite_midfielder_is_never_offered_for_a_defender(self):
        """The reported shape: a weak defender on the roster, the best
        available player in the game, and no legal way to connect them."""
        avail = _avail([
            {"Player": "Elite Mid", "Team": "TOT", "Position": "M",
             "Points": 9.0, "Transfer Score": 0.99, "Draft_State": "a"},
        ])
        sugg, _ = _compute_transfer_suggestions(
            avail, self._roster_both_positions(), top_n=None,
            roster_candidates=None, avail_candidates=None,
            one_per_position=False)
        assert len(sugg) == 1, sugg
        assert sugg[0]["add_player"] == "Elite Mid"
        assert sugg[0]["drop_position"] == "M" == sugg[0]["add_position"]
        assert all(s["drop_position"] != "D" for s in sugg), (
            "the weak defender must not be upgraded with a midfielder, "
            "however much it would improve the squad"
        )

    def test_every_suggestion_matches_positions_on_both_sides(self):
        avail = _avail([
            {"Player": "Good Def", "Team": "AVL", "Position": "D",
             "Points": 5.0, "Transfer Score": 0.80, "Draft_State": "a"},
            {"Player": "Good Mid", "Team": "TOT", "Position": "M",
             "Points": 6.0, "Transfer Score": 0.90, "Draft_State": "a"},
        ])
        sugg, _ = _compute_transfer_suggestions(
            avail, self._roster_both_positions(), top_n=None,
            roster_candidates=None, avail_candidates=None,
            one_per_position=False)
        assert len(sugg) == 2, sugg
        for s in sugg:
            assert s["drop_position"] == s["add_position"]
        assert {(s["drop_player"], s["add_player"]) for s in sugg} == {
            ("Weak Def", "Good Def"), ("Weak Mid", "Good Mid")}

    def test_the_caption_says_so(self):
        """The reported confusion was a reasonable inference from a page that
        never stated the rule."""
        import inspect
        from scripts.draft import waiver_wire as ww
        src = inspect.getsource(ww._render_transfer_suggestions)
        assert "position-locked" in src
