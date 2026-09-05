"""Tests for the projection engine.

The engine exists to end a specific bug class: a projection that has already
been multiplied by a start probability being multiplied by it again, or two
copies of "the blend" quietly disagreeing. Every test here is written against a
real failure the app shipped, not an invented edge case.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common.projection_engine import (
    build_projections,
    attach_projections,
    DEFAULT_START_FLOORS,
)
from scripts.common.projection_sources import (
    BASIS_CONDITIONAL,
    BASIS_UNCONDITIONAL,
    COVERS_ALL,
    COVERS_STARTERS,
    SourceResult,
)


def _pool(n=4, **overrides):
    base = pd.DataFrame({
        "Player_ID": [1, 2, 3, 4][:n],
        "Player": ["Erling Haaland", "Bruno Borges Fernandes", "Alex Palmer", "Cole Palmer"][:n],
        "Web_Name": ["Haaland", "B.Fernandes", "A.Palmer", "Palmer"][:n],
        "Team": ["MCI", "MUN", "IPS", "CHE"][:n],
        "Position": ["F", "M", "G", "M"][:n],
    })
    for k, v in overrides.items():
        base[k] = v
    return base


def _src(name, basis, covers, rows, **kw):
    return SourceResult(name, pd.DataFrame(rows), basis, covers, **kw)


class TestBasisConversion:
    def test_conditional_source_is_multiplied_by_start_once(self):
        """Proj must be Proj_Start x Start_Pct -- exactly once."""
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1], "Proj_Start": [10.0], "Start_Pct": [0.5]})
        out = build_projections([ffp], gameweek=3, pool=_pool(),
                                weights={"ffp": 1.0})
        assert out.loc[1, "Proj_Start"] == pytest.approx(10.0)
        assert out.loc[1, "Start_Pct"] == pytest.approx(0.5)
        assert out.loc[1, "Proj"] == pytest.approx(5.0)

    def test_unconditional_source_is_converted_up_before_blending(self):
        """An expected-value source must be un-discounted to the conditional
        basis first. Averaging FPL's ep_next straight against Rotowire drags the
        blend down by the start probability -- the shape of the bug that ran the
        FFP term ~44% low."""
        ep = _src("fpl_ep", BASIS_UNCONDITIONAL, COVERS_ALL,
                  {"Player_ID": [1], "Proj": [5.0], "Start_Pct": [0.5]})
        out = build_projections([ep], gameweek=3, pool=_pool(),
                                weights={"fpl_ep": 1.0})
        # 5.0 expected at a 50% start rate means 10.0 if he starts...
        assert out.loc[1, "Proj_Start"] == pytest.approx(10.0)
        # ...and converting back must return the number we started with.
        assert out.loc[1, "Proj"] == pytest.approx(5.0)

    def test_round_trip_never_double_discounts(self):
        """Mixing both bases in one blend must not charge start twice."""
        cond = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                    {"Player_ID": [1], "Proj_Start": [10.0], "Start_Pct": [0.5]})
        uncond = _src("fpl_ep", BASIS_UNCONDITIONAL, COVERS_ALL,
                      {"Player_ID": [1], "Proj": [5.0]})
        out = build_projections([cond, uncond], gameweek=3, pool=_pool(),
                                weights={"ffp": 0.5, "fpl_ep": 0.5})
        # Both describe the same player identically; the blend must too.
        assert out.loc[1, "Proj_Start"] == pytest.approx(10.0)
        assert out.loc[1, "Proj"] == pytest.approx(5.0)


class TestWeightRenormalisation:
    def test_both_sources_present_uses_the_configured_split(self):
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1], "Proj_Start": [5.0], "Start_Pct": [1.0]})
        out = build_projections([rw, ffp], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[1, "Proj_Start"] == pytest.approx(0.6 * 10.0 + 0.4 * 5.0)

    def test_missing_source_renormalises_rather_than_shrinking(self):
        """With only Rotowire, the answer is Rotowire -- not 60% of it."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        out = build_projections([rw], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[1, "Proj_Start"] == pytest.approx(10.0)

    def test_zero_weight_source_is_carried_but_never_blended(self):
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        ep = _src("fpl_ep", BASIS_UNCONDITIONAL, COVERS_ALL,
                  {"Player_ID": [1], "Proj": [1.0], "Start_Pct": [1.0]})
        out = build_projections([rw, ep], gameweek=3, pool=_pool(),
                                weights={"rotowire": 1.0, "fpl_ep": 0.0})
        assert out.loc[1, "Proj_Start"] == pytest.approx(10.0)
        # ...but its value is still available for the Hub and the snapshot.
        assert out.loc[1, "Proj_Start__fpl_ep"] == pytest.approx(1.0)
        assert "xP" not in out.loc[1, "Proj_Src"]


class TestStartProbability:
    def test_chance_of_playing_is_used_when_ffp_is_silent(self):
        """The divergence this module exists to remove: compute_player_scores
        fell back to chance_of_playing, blend_fixture_projections did not, so
        the same player had two different 'blended' numbers."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Cole Palmer"], "Team": ["CHE"],
                   "Position": ["M"], "Proj_Start": [10.0]})
        pool = _pool(chance_of_playing_next_round=[None, None, None, 25])
        out = build_projections([rw], gameweek=3, pool=pool, weights={"rotowire": 1.0})
        # 25% chance, floored to the MID Rotowire floor because Rotowire still
        # lists him -- the floor is the expert-lineup signal, and it is applied
        # in exactly one place now.
        assert out.loc[4, "Start_Pct"] == pytest.approx(DEFAULT_START_FLOORS["M"])

    def test_rotowire_absence_lowers_start_rather_than_dropping_the_player(self):
        """Rotowire lists only expected starters, so absence is information."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1, 4], "Proj_Start": [9.0, 8.0],
                    "Start_Pct": [0.30, 0.30]})
        out = build_projections([rw, ffp], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        # Haaland: Rotowire prices him, so his 30% is floored to the FWD floor.
        assert out.loc[1, "Start_Pct"] == pytest.approx(DEFAULT_START_FLOORS["F"])
        # Cole Palmer: Rotowire does not, so FFP's 30% stands.
        assert out.loc[4, "Start_Pct"] == pytest.approx(0.30)


class TestGameweekGate:
    def test_a_source_published_for_another_gameweek_is_dropped(self):
        """A wrong gameweek is worse than a missing source: every value in it is
        individually plausible."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        stale = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                     {"Player_ID": [1], "Proj_Start": [99.0], "Start_Pct": [1.0]},
                     gameweek=2)
        out = build_projections([rw, stale], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[1, "Proj_Start"] == pytest.approx(10.0)
        assert out.loc[1, "Proj_Src"] == "RW"

    def test_an_unknown_gameweek_is_not_a_wrong_one(self):
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1], "Proj_Start": [6.0], "Start_Pct": [1.0]},
                   gameweek=None)
        out = build_projections([ffp], gameweek=3, pool=_pool(), weights={"ffp": 1.0})
        assert out.loc[1, "Proj_Start"] == pytest.approx(6.0)


class TestProvenanceAndSpread:
    def test_source_label_names_only_contributors(self):
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1, 4], "Proj_Start": [5.0, 4.0],
                    "Start_Pct": [1.0, 1.0]})
        out = build_projections([rw, ffp], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[1, "Proj_Src"] == "RW+FFP"
        assert out.loc[4, "Proj_Src"] == "FFP"
        assert out.loc[2, "Proj_Src"] == "None"

    def test_spread_measures_disagreement(self):
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1], "Proj_Start": [4.0], "Start_Pct": [1.0]})
        out = build_projections([rw, ffp], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[1, "Proj_Spread"] == pytest.approx(6.0)


class TestBlankGameweeks:
    def test_unpriced_but_available_player_is_unknown_not_zero(self):
        """Scoring a blank gameweek as 0 reads as 'drop him'."""
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1], "Proj_Start": [10.0], "Start_Pct": [1.0]})
        out = build_projections([ffp], gameweek=3, pool=_pool(), weights={"ffp": 1.0})
        assert pd.isna(out.loc[4, "Proj"])

    def test_unpriced_and_injured_player_scores_zero(self):
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1], "Proj_Start": [10.0], "Start_Pct": [1.0]})
        pool = _pool(status=["a", "a", "a", "i"])
        out = build_projections([ffp], gameweek=3, pool=pool, weights={"ffp": 1.0})
        assert out.loc[4, "Proj"] == 0.0


class TestNameResolution:
    def test_rotowire_common_name_resolves_to_the_bootstrap_legal_name(self):
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Bruno Fernandes"], "Team": ["MUN"],
                   "Position": ["M"], "Proj_Start": [7.0]})
        out = build_projections([rw], gameweek=3, pool=_pool(), weights={"rotowire": 1.0})
        assert out.loc[2, "Proj_Start"] == pytest.approx(7.0)

    def test_a_shared_surname_does_not_cross_positions(self):
        """Alex Palmer (backup GK) must never inherit Cole Palmer's numbers."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Cole Palmer"], "Team": ["CHE"],
                   "Position": ["M"], "Proj_Start": [9.0]})
        out = build_projections([rw], gameweek=3, pool=_pool(), weights={"rotowire": 1.0})
        assert out.loc[4, "Proj_Start"] == pytest.approx(9.0)
        assert pd.isna(out.loc[3, "Proj_Start"])


class TestAttachProjections:
    def test_joins_on_element_id_and_overwrites_stale_columns(self):
        proj = pd.DataFrame(
            {"Proj": [5.0], "Proj_Start": [10.0], "Start_Pct": [0.5],
             "Proj_Next3": [15.0], "Proj_Src": ["RW+FFP"],
             "Proj_Spread": [1.0], "Proj_GW": [3]},
            index=pd.Index([1], name="Player_ID"),
        )
        page = pd.DataFrame({"Player_ID": [1, 2], "Proj": [999.0, 999.0]})
        out = attach_projections(page, proj)
        assert out.loc[0, "Proj"] == pytest.approx(5.0)
        assert pd.isna(out.loc[1, "Proj"])

    def test_missing_key_column_is_a_no_op_not_a_crash(self):
        proj = pd.DataFrame({"Proj": [5.0]}, index=pd.Index([1], name="Player_ID"))
        page = pd.DataFrame({"Player": ["Haaland"]})
        assert attach_projections(page, proj).equals(page)


class TestPositionCodesAndFallbacks:
    """Two traps that make the blend quietly wrong rather than obviously broken."""

    def test_gkp_style_position_codes_still_get_the_rotowire_floor(self):
        """Draft pages carry GK/DEF/MID/FWD; analytics groups on G/D/M/F. Feeding
        the wrong codes in makes the start floors match nothing and silently do
        nothing -- the same failure that had every Power Rankings team at 50."""
        from scripts.common.analytics import blend_projections_onto

        df = pd.DataFrame({
            "Player": ["Erling Haaland"],
            "Team": ["MCI"],
            "Position": ["FWD"],          # not "F"
            "Points": [10.0],
            "chance_of_playing_next_round": [25],
        })
        out = blend_projections_onto(df, None)
        assert out.loc[0, "Start_Pct"] == pytest.approx(DEFAULT_START_FLOORS["F"])

    def test_ep_next_fills_only_where_nothing_else_priced_the_player(self):
        """FPL's expected points used to be written into the Rotowire column,
        taking Rotowire's 60% weight while reading as Rotowire downstream. As a
        declared fallback it fills the same gap and says so."""
        from scripts.common.analytics import blend_projections_onto

        df = pd.DataFrame({
            "Player": ["Erling Haaland", "Cole Palmer"],
            "Team": ["MCI", "CHE"],
            "Position": ["F", "M"],
            "Points": [10.0, 0.0],        # Rotowire priced only Haaland
            "ep_next": [3.0, 4.0],
        })
        out = blend_projections_onto(df, None)
        # Haaland keeps Rotowire's number -- the fallback cannot displace it.
        assert out.loc[0, "Proj_Start"] == pytest.approx(10.0)
        assert out.loc[0, "Proj_Src"] == "RW"
        # Palmer, whom nobody else priced, gets FPL's number under its own label.
        assert out.loc[1, "Proj_Start"] == pytest.approx(4.0)
        assert out.loc[1, "Proj_Src"] == "xP"


class TestSourcePositionCodesDoNotBreakMatching:
    """FFP publishes GK/DEF/MID/FWD; the FPL pool uses G/D/M/F.

    Every ReferenceMatcher tier below the first two is scoped by position, so
    mismatched encodings share no group and any name that is not an exact
    (name, team) hit falls straight through. FFP's site payload is saved by its
    integer Player_ID, so this stayed invisible -- it only bites where there is
    no id to join on, which is exactly the archived and spreadsheet tables.
    Measured live on the recovered GW3 archive: it cost 32 of 543 matches.
    """

    def test_ffp_style_codes_still_match_a_gdmf_pool(self):
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL, {
            # Common name + FFP's position spelling, and no Player_ID -- the
            # exact shape of an archived table.
            "Player": ["Bruno Fernandes"],
            "Team": ["MUN"],
            "Position": ["MID"],
            "Proj_Start": [7.0],
            "Start_Pct": [0.9],
        })
        out = build_projections([ffp], gameweek=3, pool=_pool(), weights={"ffp": 1.0})
        assert out.loc[2, "Proj_Start"] == pytest.approx(7.0)

    def test_position_scoping_still_keeps_the_palmers_apart(self):
        """Normalising encodings must not weaken the scoping itself."""
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL, {
            "Player": ["Cole Palmer"],
            "Team": ["CHE"],
            "Position": ["MID"],
            "Proj_Start": [9.0],
            "Start_Pct": [1.0],
        })
        out = build_projections([ffp], gameweek=3, pool=_pool(), weights={"ffp": 1.0})
        assert out.loc[4, "Proj_Start"] == pytest.approx(9.0)
        assert pd.isna(out.loc[3, "Proj_Start"])      # Alex Palmer, backup GK

    def test_web_name_is_tried_when_the_full_name_misses(self):
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL, {
            "Player": ["Not A Real Name"],
            "Web_Name": ["Haaland"],
            "Team": ["MCI"],
            "Position": ["FWD"],
            "Proj_Start": [8.0],
            "Start_Pct": [1.0],
        })
        out = build_projections([ffp], gameweek=3, pool=_pool(), weights={"ffp": 1.0})
        assert out.loc[1, "Proj_Start"] == pytest.approx(8.0)
