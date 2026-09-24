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

    def test_the_best_target_at_a_position_rivals_need_is_a_long_shot(self):
        from scripts.common.waiver_priority import BAND_LONG_SHOT
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "rival_needs": {"M": [1, 2, 3, 4]},
                            "expected_gone": 4.6},
        )
        top = next(s for s in suggestions if s["add_player"] == "Top Mid")
        assert top["outlook_band"] == BAND_LONG_SHOT
        assert "4 of the 6" in top["outlook_reason"]

    def test_a_deeper_target_at_the_same_position_survives_them(self):
        from scripts.common.waiver_priority import BAND_LIKELY
        from scripts.draft.waiver_wire import _compute_transfer_suggestions

        suggestions, _ = _compute_transfer_suggestions(
            self._avail_pair(), _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "rival_needs": {"M": []},
                            "expected_gone": 0.0},
        )
        assert {s["outlook_band"] for s in suggestions} == {BAND_LIKELY}

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
        # Make the nominally weaker player the best *adjusted* candidate.
        avail.loc[avail["Player"] == "Top Mid", "chance_of_playing_next_round"] = 25

        suggestions, _ = _compute_transfer_suggestions(
            avail, _roster(ROSTER), top_n=3,
            one_per_position=False, roster_candidates=None, avail_candidates=None,
            waiver_context={"n_ahead": 6, "rival_needs": {"M": [1]}, "expected_gone": 4.6},
        )
        deep = next(s for s in suggestions if s["add_player"] == "Deep Mid")
        assert deep["outlook_band"] == BAND_LONG_SHOT, "he is now the #1 adjusted target"


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
        assert ctx == {"n_ahead": 0, "rival_needs": {}, "expected_gone": 0.0}
