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
    blend_aligned,
    build_projections,
    attach_projections,
    DEFAULT_OMITTED_STARTS,
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

    def test_rotowire_presence_floors_the_start_probability(self):
        """Rotowire lists only expected starters, so presence is information."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player": ["Erling Haaland"], "Team": ["MCI"],
                   "Position": ["F"], "Proj_Start": [10.0]})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1, 4], "Proj_Start": [9.0, 8.0],
                    "Start_Pct": [0.30, 0.30]})
        out = build_projections([rw, ffp], gameweek=3, pool=_pool(),
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[1, "Start_Pct"] == pytest.approx(DEFAULT_START_FLOORS["F"])
        # Cole Palmer is omitted, but Rotowire priced nobody else at CHE, so
        # there is no evidence it covered the club. FFP's 30% stands.
        assert out.loc[4, "Start_Pct"] == pytest.approx(0.30)


def _covered_pool(club="CHE", n=8, position="M"):
    """A club Rotowire has clearly covered, plus one omitted player at it."""
    return pd.DataFrame({
        "Player_ID": list(range(1, n + 2)),
        "Player": [f"Player {i}" for i in range(1, n + 2)],
        "Team": [club] * (n + 1),
        "Position": [position] * (n + 1),
    })


def _covered_sources(n=8, omitted_ffp_start=0.70, position="M"):
    """Rotowire prices players 1..n; player n+1 is omitted but priced by FFP."""
    rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
              {"Player_ID": list(range(1, n + 1)),
               "Proj_Start": [5.0] * n})
    ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
               {"Player_ID": list(range(1, n + 2)),
                "Proj_Start": [5.0] * (n + 1),
                "Start_Pct": [0.80] * n + [omitted_ffp_start]})
    return rw, ffp


class TestRotowireOmission:
    """Absence from a covered club is a lineup call, not a missing value.

    Measured on the GW3 snapshot: players Rotowire listed started 90.5% of the
    time, players it omitted 4.2%. The engine used to renormalise Rotowire's
    weight away and let FFP's start probability stand alone, so a player FFP
    rated at 70% kept 70% -- and of the omitted players the engine gave >=80%,
    none at all started.
    """

    def _run(self, pool, sources, **kw):
        return build_projections(list(sources), gameweek=3, pool=pool,
                                 weights={"rotowire": 0.6, "ffp": 0.4}, **kw)

    def test_an_omitted_player_at_a_covered_club_is_discounted(self):
        out = self._run(_covered_pool(), _covered_sources(omitted_ffp_start=0.70))
        # 0.6 * implied(M) + 0.4 * 0.70
        expected = 0.6 * DEFAULT_OMITTED_STARTS["M"] + 0.4 * 0.70
        assert out.loc[9, "Start_Pct"] == pytest.approx(expected)
        assert out.loc[9, "Start_Pct"] < 0.70

    def test_listed_players_are_untouched(self):
        """Re-deriving the listed side as a blend was measured and is worse."""
        out = self._run(_covered_pool(), _covered_sources())
        assert out.loc[1, "Start_Pct"] == pytest.approx(0.80)

    def test_it_can_only_lower(self):
        """An omission is never evidence that a player *will* start."""
        out = self._run(_covered_pool(), _covered_sources(omitted_ffp_start=0.03))
        assert out.loc[9, "Start_Pct"] == pytest.approx(0.03)

    def test_ffp_still_orders_the_omitted_players(self):
        """The reason this blends instead of capping.

        A cap flattens an FFP-90% player and an FFP-20% player onto one number,
        discarding the only opinion left about which of them might play.
        """
        pool = _covered_pool(n=8)
        pool = pd.concat([pool, pd.DataFrame({
            "Player_ID": [10], "Player": ["Player 10"], "Team": ["CHE"],
            "Position": ["M"]})], ignore_index=True)
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player_ID": list(range(1, 9)), "Proj_Start": [5.0] * 8})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": list(range(1, 11)),
                    "Proj_Start": [5.0] * 10,
                    "Start_Pct": [0.80] * 8 + [0.90, 0.20]})
        out = self._run(pool, (rw, ffp))
        assert out.loc[9, "Start_Pct"] > out.loc[10, "Start_Pct"]
        assert out.loc[9, "Start_Pct"] - out.loc[10, "Start_Pct"] == pytest.approx(
            0.4 * (0.90 - 0.20))

    def test_a_goalkeeper_is_discounted_hardest(self):
        """Keepers do not rotate: 0 of 51 omitted GKs started in GW3."""
        gk = self._run(_covered_pool(position="G"),
                       _covered_sources(position="G", omitted_ffp_start=0.70))
        mid = self._run(_covered_pool(position="M"),
                        _covered_sources(position="M", omitted_ffp_start=0.70))
        assert gk.loc[9, "Start_Pct"] < mid.loc[9, "Start_Pct"]

    def test_an_uncovered_club_is_not_punished(self):
        """Below the coverage threshold, silence is an outage, not a lineup."""
        out = self._run(_covered_pool(n=3), _covered_sources(n=3, omitted_ffp_start=0.70))
        assert out.loc[4, "Start_Pct"] == pytest.approx(0.70)

    def test_without_team_labels_it_fails_open(self):
        pool = _covered_pool().drop(columns=["Team"])
        out = self._run(pool, _covered_sources(omitted_ffp_start=0.70))
        assert out.loc[9, "Start_Pct"] == pytest.approx(0.70)

    def test_coverage_comes_from_the_source_table_not_the_frame(self):
        """The frame is usually a 15-player squad, which can never show coverage.

        Counting Rotowire-priced players *within the frame* is only meaningful
        when the frame is the whole pool. A Classic squad holds two or three
        players per club, so the threshold was unreachable and the penalty
        silently did nothing on every per-squad page -- Fixture Projections, the
        lineup cards, Team Analysis. It looked correct because it was measured
        against a 652-row pool.
        """
        pool = pd.DataFrame({
            "Player_ID": [1, 2, 3],
            "Player": ["A", "B", "C"],
            "Team": ["LIV"] * 3,
            "Position": ["M"] * 3,
        })
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player_ID": [1, 2], "Proj_Start": [5.0, 5.0]})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": [1, 2, 3], "Proj_Start": [5.0] * 3,
                    "Start_Pct": [0.80, 0.80, 0.70]})

        # Two priced players in the frame: not enough to conclude anything.
        assert self._run(pool, (rw, ffp)).loc[3, "Start_Pct"] == pytest.approx(0.70)

        # Rotowire published 11 at LIV, which the frame simply cannot see.
        idx = pd.Index([1, 2, 3], name="Player_ID")
        out = blend_aligned(
            index=idx,
            per_source_raw={"rotowire": pd.Series([5.0, 5.0, np.nan], index=idx),
                            "ffp": pd.Series([5.0, 5.0, 5.0], index=idx)},
            per_source_basis={"rotowire": BASIS_CONDITIONAL, "ffp": BASIS_CONDITIONAL},
            per_source_startpct={"ffp": pd.Series([0.80, 0.80, 0.70], index=idx)},
            starters_only={"rotowire"},
            positions=pd.Series(["M"] * 3, index=idx),
            teams=pd.Series(["LIV"] * 3, index=idx),
            weights={"rotowire": 0.6, "ffp": 0.4},
            source_club_coverage={"rotowire": {"LIV": 11}},
        )
        assert out.loc[3, "Start_Pct"] == pytest.approx(
            0.6 * DEFAULT_OMITTED_STARTS["M"] + 0.4 * 0.70)

    def test_build_projections_derives_coverage_from_the_source_itself(self):
        """A source carrying Team labels needs no help from the caller."""
        pool = pd.DataFrame({
            "Player_ID": list(range(1, 8)),
            "Player": [f"P{i}" for i in range(1, 8)],
            "Team": ["LIV"] * 7,
            "Position": ["M"] * 7,
        })
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player_ID": list(range(1, 7)), "Team": ["LIV"] * 6,
                   "Proj_Start": [5.0] * 6})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": list(range(1, 8)), "Proj_Start": [5.0] * 7,
                    "Start_Pct": [0.80] * 6 + [0.70]})
        out = build_projections([rw, ffp], gameweek=3, pool=pool,
                                weights={"rotowire": 0.6, "ffp": 0.4})
        assert out.loc[7, "Start_Pct"] < 0.70

    def test_the_implied_value_is_recorded_for_the_harness(self):
        out = self._run(_covered_pool(), _covered_sources())
        assert out.loc[9, "Start_Pct__rotowire"] == pytest.approx(
            DEFAULT_OMITTED_STARTS["M"])
        assert out.loc[1, "Start_Pct__rotowire"] == pytest.approx(
            DEFAULT_START_FLOORS["M"])

    def test_with_no_other_start_opinion_the_implied_value_stands_alone(self):
        """start_pct would otherwise be the bare 1.0 default, which is not an opinion."""
        rw = _src("rotowire", BASIS_CONDITIONAL, COVERS_STARTERS,
                  {"Player_ID": list(range(1, 9)), "Proj_Start": [5.0] * 8})
        ffp = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                   {"Player_ID": list(range(1, 10)), "Proj_Start": [5.0] * 9})
        out = self._run(_covered_pool(), (rw, ffp))
        assert out.loc[9, "Start_Pct"] == pytest.approx(DEFAULT_OMITTED_STARTS["M"])


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


class TestUnpricedPlayersAreNotNeutral:
    """A player no source priced is a non-starter, not an unknown.

    Reported from the app: the Waiver Wire suggested dropping Nathan Collins
    (projected 2.9, but a 25%-chance calf injury) while keeping Taylor
    Harwood-Bellis, who was projected 0. The drop queue sorts on
    ``Keep Score x injury_factor``, and the two came out at 0.242 and 0.245 --
    a gap of 0.003, entirely created by Harwood-Bellis being handed a neutral
    1GW of 0.50 for having no projection at all.

    Measured on GW4: 120 players sat on that neutral, and all 20 clubs had a
    fixture, so not one of them was in a blank gameweek.
    """

    def _pool(self):
        return pd.DataFrame({
            "Player_ID": [1, 2, 3, 4],
            "Player": ["Priced One", "Priced Two", "Unpriced Sub", "Blank Club Star"],
            "Web_Name": ["One", "Two", "Sub", "Star"],
            "Team": ["MCI", "MCI", "MCI", "BLA"],
            "Position": ["M", "D", "D", "M"],
            "status": ["a", "a", "a", "a"],
        })

    def _source(self):
        # Two of MCI's players are priced; nobody at BLA is (a blank gameweek,
        # or that club missing from the feed).
        return _src("ffp", BASIS_CONDITIONAL, COVERS_ALL, {
            "Player_ID": [1, 2],
            "Proj_Start": [6.0, 4.0],
            "Start_Pct": [1.0, 1.0],
        })

    def test_unpriced_player_at_a_covered_club_scores_zero(self):
        """His club is clearly in the feed, so his absence from it means he is
        not expected to start -- which is information, not a gap."""
        out = build_projections([self._source()], gameweek=4,
                                pool=self._pool(), weights={"ffp": 1.0})
        assert out.loc[3, "Proj"] == 0.0

    def test_unpriced_player_at_an_uncovered_club_stays_unknown(self):
        """A club with no priced players at all is either blank or missing from
        the feeds. Scoring an elite asset 0 there reads as "drop him"."""
        out = build_projections([self._source()], gameweek=4,
                                pool=self._pool(), weights={"ffp": 1.0})
        assert pd.isna(out.loc[4, "Proj"])

    def test_a_dead_feed_zeroes_nobody(self):
        """If nothing was priced anywhere, the feed is down -- and no player
        should be marked a non-starter on the strength of data that is absent."""
        empty = _src("ffp", BASIS_CONDITIONAL, COVERS_ALL,
                     {"Player_ID": [], "Proj_Start": [], "Start_Pct": []})
        out = build_projections([empty], gameweek=4,
                                pool=self._pool(), weights={"ffp": 1.0})
        assert out["Proj"].isna().all()

    def test_an_injured_unpriced_player_scores_zero_even_at_a_blank_club(self):
        pool = self._pool()
        pool.loc[3, "status"] = "i"
        out = build_projections([self._source()], gameweek=4,
                                pool=pool, weights={"ffp": 1.0})
        assert out.loc[4, "Proj"] == 0.0
