"""Tests for the shared transfer sanity veto.

The veto's whole job is to remove indefensible suggestions, so the tests that
matter most are the ones showing it can still *see* -- a gate reading columns a
frame does not have passes everything while looking exactly like protection.
"""

import pandas as pd
import pytest

from scripts.common.transfer_sanity import (SANITY_TOLERANCE, is_seriously_injured,
                                            sanity_check_signals,
                                            sanity_check_suggestion)


def draft_row(proj=5.0, season=60, multi=15.0, status="a", chance=None):
    """A Draft frame row: season points live in ``Season_Points``."""
    return pd.Series({"_effective_proj": proj, "Season_Points": season,
                      "MultiGW_Proj": multi, "status": status,
                      "chance_of_playing_next_round": chance})


def classic_row(proj=5.0, season=60, multi=15.0, status="a", chance=None):
    """A Classic frame row: season points live in ``total_points``, and
    ``Season_Points`` is never defined at all."""
    return pd.Series({"_effective_proj": proj, "total_points": season,
                      "MultiGW_Proj": multi, "status": status,
                      "chance_of_playing_next_round": chance})


class TestSignalsAreVisibleInBothFormats:
    """The trap this helper exists to avoid.

    ``compute_player_scores`` resolves season points as "Season_Points if
    present else total_points" because Draft and Classic frames differ. A veto
    hard-coded to Draft's column would find nothing on a Classic frame, count
    zero signals, and pass every suggestion -- indistinguishable from working.
    """

    @pytest.mark.parametrize("make_row", [draft_row, classic_row])
    def test_all_three_signals_are_found(self, make_row):
        checks = sanity_check_signals(make_row(), make_row())
        assert [name for name, _ in checks] == ["proj_pts", "season_pts", "3gw_proj"]

    @pytest.mark.parametrize("make_row", [draft_row, classic_row])
    def test_a_clearly_worse_add_is_vetoed(self, make_row):
        drop = make_row(proj=6.0, season=80, multi=18.0)
        add = make_row(proj=1.0, season=5, multi=3.0)
        passes, reason = sanity_check_suggestion(drop, add)
        assert not passes
        assert "proj_pts" in reason and "season_pts" in reason

    @pytest.mark.parametrize("make_row", [draft_row, classic_row])
    def test_a_clearly_better_add_passes(self, make_row):
        passes, reason = sanity_check_suggestion(
            make_row(proj=2.0, season=20, multi=6.0),
            make_row(proj=7.0, season=90, multi=20.0))
        assert passes and reason == "ok"

    def test_a_classic_row_is_not_judged_on_a_missing_draft_column(self):
        """Season points must come from total_points here, not be skipped."""
        checks = sanity_check_signals(classic_row(season=80), classic_row(season=5))
        assert ("season_pts", False) in checks


class TestTolerance:
    def test_marginally_worse_is_allowed(self):
        """The composite score legitimately knows things raw points do not, so
        the band is loose on purpose. A veto that cries wolf gets switched off."""
        drop = draft_row(proj=5.0, season=100, multi=15.0)
        add = draft_row(proj=5.0 * SANITY_TOLERANCE, season=100 * SANITY_TOLERANCE,
                        multi=15.0 * SANITY_TOLERANCE)
        assert sanity_check_suggestion(drop, add)[0]

    def test_just_below_the_band_fails(self):
        drop = draft_row(proj=5.0, season=100, multi=15.0)
        add = draft_row(proj=5.0 * 0.5, season=100 * 0.5, multi=15.0 * 0.5)
        assert not sanity_check_suggestion(drop, add)[0]

    def test_a_majority_of_signals_decides_it(self):
        """Two of three good is enough; one of three is not."""
        drop = draft_row(proj=5.0, season=100, multi=15.0)
        two_good = draft_row(proj=6.0, season=120, multi=2.0)
        one_good = draft_row(proj=6.0, season=10, multi=2.0)
        assert sanity_check_suggestion(drop, two_good)[0]
        assert not sanity_check_suggestion(drop, one_good)[0]


class TestZeroVersusMissing:
    """0.0 and "no value" are different claims and must not be merged."""

    def test_an_add_not_expected_to_start_fails_the_projection(self):
        checks = sanity_check_signals(draft_row(proj=5.0), draft_row(proj=0.0))
        assert ("proj_pts", False) in checks

    def test_a_drop_not_expected_to_start_passes_it(self):
        checks = sanity_check_signals(draft_row(proj=0.0), draft_row(proj=4.0))
        assert ("proj_pts", True) in checks

    def test_neither_expected_to_start_says_nothing(self):
        checks = sanity_check_signals(draft_row(proj=0.0), draft_row(proj=0.0))
        assert not [c for c in checks if c[0] == "proj_pts"]

    def test_a_missing_projection_is_skipped_not_scored_as_zero(self):
        """A blank gameweek is an absence of information, not a zero. Scoring it
        as zero would veto every suggestion involving a blanking club."""
        drop = pd.Series({"_effective_proj": float("nan"), "Season_Points": 50,
                          "MultiGW_Proj": 12.0, "status": "a"})
        add = draft_row(proj=4.0, season=45, multi=11.0)
        assert not [c for c in sanity_check_signals(drop, add) if c[0] == "proj_pts"]

    def test_a_zero_season_total_is_skipped(self):
        """A new signing has no history; that is not evidence against him."""
        checks = sanity_check_signals(draft_row(season=80), draft_row(season=0))
        assert not [c for c in checks if c[0] == "season_pts"]


class TestInjuryOverride:
    @pytest.mark.parametrize("status", ["i", "s", "u"])
    def test_replacing_an_unavailable_player_is_always_allowed(self, status):
        drop = draft_row(proj=9.0, season=150, multi=27.0, status=status)
        add = draft_row(proj=1.0, season=2, multi=3.0)
        passes, reason = sanity_check_suggestion(drop, add)
        assert passes and reason == "injury override"

    def test_a_low_chance_of_playing_also_overrides(self):
        drop = draft_row(proj=9.0, season=150, multi=27.0, chance=25)
        assert sanity_check_suggestion(drop, draft_row(proj=1.0, season=2, multi=3.0))[0]

    def test_a_merely_doubtful_player_is_still_protected(self):
        """75% is a doubt, not an absence -- the veto should still apply."""
        drop = draft_row(proj=9.0, season=150, multi=27.0, chance=75)
        assert not sanity_check_suggestion(drop, draft_row(proj=1.0, season=2, multi=3.0))[0]

    @pytest.mark.parametrize("row", [
        pd.Series({"status": "a"}),
        pd.Series({"status": "a", "chance_of_playing_next_round": None}),
        pd.Series({}),
    ])
    def test_a_healthy_or_unknown_player_is_not_treated_as_injured(self, row):
        assert not is_seriously_injured(row)


class TestDegradation:
    def test_no_comparable_data_passes_rather_than_blocking_everything(self):
        """A veto firing on absent columns would suppress every suggestion on a
        degraded feed, which is worse than letting a questionable one through.
        The Classic callsite counts these so the state is visible."""
        blank = pd.Series({"status": "a"})
        passes, reason = sanity_check_suggestion(blank, blank)
        assert passes and reason == "no data"
        assert sanity_check_signals(blank, blank) == []

    def test_unparseable_values_do_not_raise(self):
        junk = pd.Series({"_effective_proj": "n/a", "Season_Points": "-",
                          "MultiGW_Proj": None, "status": "a"})
        passes, reason = sanity_check_suggestion(junk, junk)
        assert passes and reason == "no data"

    def test_plain_dicts_work_as_well_as_series(self):
        """The Classic loop passes Series; keeping dicts working means a caller
        building rows by hand cannot silently disable the gate."""
        drop = {"_effective_proj": 6.0, "total_points": 80, "MultiGW_Proj": 18.0}
        add = {"_effective_proj": 1.0, "total_points": 5, "MultiGW_Proj": 3.0}
        assert not sanity_check_suggestion(drop, add)[0]


class TestDraftDelegates:
    def test_the_draft_wrapper_is_the_shared_implementation(self):
        """Draft's veto moved here; it must not drift back into a second copy."""
        from scripts.draft.waiver_wire import _sanity_check_suggestion
        drop = draft_row(proj=6.0, season=80, multi=18.0)
        add = draft_row(proj=1.0, season=5, multi=3.0)
        assert _sanity_check_suggestion(drop, add) == sanity_check_suggestion(drop, add)


class TestBasisIsNotMixed:
    """_effective_proj carries start likelihood; Projected_Points does not.

    Comparing one against the other charges a rotation risk to exactly one side
    of the swap -- the basis confusion the projection engine exists to end.
    """

    def test_a_shared_fallback_column_is_used(self):
        drop = pd.Series({"Projected_Points": 6.0, "status": "a"})
        add = pd.Series({"Projected_Points": 1.0, "status": "a"})
        assert ("proj_pts", False) in sanity_check_signals(drop, add)

    def test_different_columns_on_each_side_are_not_compared(self):
        drop = pd.Series({"_effective_proj": 6.0, "status": "a"})
        add = pd.Series({"Projected_Points": 1.0, "status": "a"})
        assert not [c for c in sanity_check_signals(drop, add) if c[0] == "proj_pts"]

    def test_the_most_specific_shared_column_wins(self):
        """Both carry both columns, so the start-adjusted one is used -- and it
        is the one that says the add is fine."""
        drop = pd.Series({"_effective_proj": 2.0, "Projected_Points": 9.0, "status": "a"})
        add = pd.Series({"_effective_proj": 5.0, "Projected_Points": 1.0, "status": "a"})
        assert ("proj_pts", True) in sanity_check_signals(drop, add)

    def test_season_points_are_not_mixed_across_formats(self):
        draft = pd.Series({"Season_Points": 80, "status": "a"})
        classic = pd.Series({"total_points": 5, "status": "a"})
        assert not [c for c in sanity_check_signals(draft, classic) if c[0] == "season_pts"]

    def test_the_three_gameweek_signal_prefers_the_engines_horizon(self):
        """`MultiGW_Proj` is a mixture -- FFP's start-adjusted total where FFP
        matched the player, a conditional `x 3` fallback where it did not. Read
        raw, the two sides of a swap can be on different bases."""
        drop = pd.Series({"Proj_Next3": 12.0, "MultiGW_Proj": 12.0, "status": "a"})
        add = pd.Series({"Proj_Next3": 4.0, "MultiGW_Proj": 30.0, "status": "a"})
        assert ("3gw_proj", False) in sanity_check_signals(drop, add)

    def test_a_horizon_on_one_side_only_is_not_compared(self):
        drop = pd.Series({"Proj_Next3": 12.0, "status": "a"})
        add = pd.Series({"MultiGW_Proj": 30.0, "status": "a"})
        assert not [c for c in sanity_check_signals(drop, add) if c[0] == "3gw_proj"]
