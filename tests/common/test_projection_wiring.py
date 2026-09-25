"""The blend must exist in exactly one place, and pages must not re-derive it.

Plausibility checks cannot catch this class of bug: two hand-copied blends that
have drifted apart both produce individually plausible numbers. The only way to
see it is to compare the callsites, which is what this file does.
"""

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from scripts.common.analytics import (
    blend_fixture_projections,
    blend_projections_onto,
    compute_player_scores,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _frame():
    """A frame that exercises the divergence: a player FFP is silent about but
    whom the FPL bootstrap flags as doubtful."""
    return pd.DataFrame({
        "Player_ID": [1, 2, 3],
        "Player": ["Erling Haaland", "Cole Palmer", "Bukayo Saka"],
        "Team": ["MCI", "CHE", "ARS"],
        "Position": ["F", "M", "M"],
        "Points": [8.0, 6.0, 5.0],
        "total_points": [40, 30, 25],
        "form": [5.0, 4.0, 3.0],
        "chance_of_playing_next_round": [None, 25, 75],
    })


class TestTheTwoCallsitesAgree:
    def test_scoring_and_fixture_display_produce_the_same_number(self):
        """compute_player_scores and blend_fixture_projections were hand-copied
        twins that had drifted: only the first fell back to the FPL
        chance_of_playing when FFP published no start percentage. The same
        player therefore scored on one number and displayed as another, both
        labelled "the blend"."""
        df = _frame()
        pool = df.copy()

        scored = compute_player_scores(df.copy(), pool, current_gw=3)
        displayed = blend_fixture_projections(df.copy(), None)

        pd.testing.assert_series_equal(
            scored["_effective_proj"].reset_index(drop=True),
            displayed["Proj"].reset_index(drop=True),
            check_names=False,
        )

    def test_the_doubtful_player_is_discounted_on_both_paths(self):
        """The concrete case: chance_of_playing=25 used to mean 0.25 in the 1GW
        score and the position floor on the fixture pages."""
        df = _frame()
        scored = compute_player_scores(df.copy(), df.copy(), current_gw=3)
        displayed = blend_fixture_projections(df.copy(), None)
        assert scored.loc[1, "Start_Pct"] == displayed.loc[1, "Start_Pct"]

    def test_they_agree_on_the_horizon_too(self):
        """The multi-gameweek half of the blend was written into one twin only.

        `blend_projections_onto` split `MultiGW_Proj` by provenance and let the
        engine convert each half; `compute_player_scores` -- the one that
        percentiles the result for 40% of every ROS score -- never assembled a
        horizon at all. Same columns in, two different answers out.
        """
        df = _frame()
        df["MultiGW_Proj"] = [24.0, 18.0, 15.0]
        df["MultiGW_Src"] = ["ffp", "single_x3", "single_x3"]
        pool = df.copy()

        scored = compute_player_scores(df.copy(), pool, current_gw=3)
        displayed = blend_projections_onto(df.copy(), None)

        pd.testing.assert_series_equal(
            scored["Proj_Next3"].reset_index(drop=True),
            displayed["Proj_Next3"].reset_index(drop=True),
            check_names=False,
        )
        # And it is a real conversion, not a pass-through: the conditional
        # fallback is start-discounted before it lands.
        assert scored.loc[1, "Proj_Next3"] < 18.0
        # Deliberately *not* `18.0 x Start_Pct`. `Start_Pct` carries this player's
        # 25% chance of playing **this** gameweek; the horizon converts on his
        # ordinary start probability instead, because applying a one-week absence
        # uniformly across three weeks is the error FFP's single `start_pct`
        # makes. A stated absence reaches the horizon through the duration cap.
        assert scored.loc[1, "Proj_Next3"] > 18.0 * scored.loc[1, "Start_Pct"]

    def test_scoring_does_not_erase_a_horizon_it_was_given(self):
        """`blend_aligned` wrote `Proj_Next3` whether or not it had anything to
        write, so scoring a frame that had already been blended replaced its
        horizon with NaN. On the Classic page that is `squad_df`, and the
        planner then priced the legs it proposed selling over one gameweek
        while pricing the legs it proposed buying over three -- the solve was
        unaffected, the leg the user reads was not.
        """
        df = _frame()
        df["Proj_Next3"] = [24.0, 18.0, 15.0]

        scored = compute_player_scores(df.copy(), df.copy(), current_gw=3)

        assert scored["Proj_Next3"].notna().all()
        assert list(scored["Proj_Next3"]) == [24.0, 18.0, 15.0]

    def test_legacy_column_matches_the_canonical_one(self):
        """Proj_Blended is kept for the pages that still read it. If it ever
        stops equalling Proj, the app has two blends again."""
        out = blend_fixture_projections(_frame(), None)
        assert np.allclose(out["Proj_Blended"], out["Proj"].round(2))


class TestNobodyReimplementsTheBlend:
    """Grep-level guards. Crude, but the failure they prevent is a page quietly
    growing its own copy of arithmetic that took three bugs to get right."""

    #: Files allowed to multiply a projection by a start probability.
    ALLOWED = {
        "scripts/common/projection_engine.py",
        "scripts/common/projection_sources.py",
    }

    def _page_sources(self):
        for sub in ("draft", "classic", "fpl", "common"):
            for path in (REPO / "scripts" / sub).glob("*.py"):
                # Tracked duplicates holding unmerged work -- never edited, and
                # deliberately excluded rather than deleted.
                if " 2.py" in path.name:
                    continue
                yield path

    def test_no_page_hardcodes_the_blend_weights(self):
        """0.6/0.4 as bare literals in two functions, restated in six comments,
        was how the split came to be unreachable from config."""
        pattern = re.compile(r"0\.6\s*\*\s*\w*[rR]oto|0\.4\s*\*\s*\w*ffp", re.I)
        offenders = [
            str(p.relative_to(REPO)) for p in self._page_sources()
            if pattern.search(p.read_text())
        ]
        assert offenders == [], (
            "These files hardcode the Rotowire/FFP blend weights. The weights "
            "live in config.PROJECTION_SOURCE_WEIGHTS and are applied by "
            "projection_engine: %s" % offenders
        )

    def test_only_the_engine_multiplies_a_projection_by_a_start_probability(self):
        """Every double-discount bug in this app's history is one of these
        multiplications happening a second time, somewhere else."""
        pattern = re.compile(
            r"(blended\w*|proj\w*|predicted\w*)\s*\*\s*start_likelihood", re.I)
        offenders = [
            str(p.relative_to(REPO)) for p in self._page_sources()
            if pattern.search(p.read_text())
            and str(p.relative_to(REPO)) not in self.ALLOWED
        ]
        assert offenders == [], (
            "Applying start likelihood outside the projection engine is how the "
            "FFP term came to run ~44%% low. Use the engine's Proj (expected "
            "value) or Proj_Start (conditional): %s" % offenders
        )

    def test_only_one_place_assembles_the_per_source_dicts(self):
        """The two hand-copied assemblies are why the horizon reached one
        callsite and not the other. There is one now: `_frame_projection_sources`."""
        pattern = re.compile(r"per_source_next3\s*(\[|=)")
        offenders = [
            str(p.relative_to(REPO)) for p in self._page_sources()
            if pattern.search(p.read_text())
            and str(p.relative_to(REPO)) not in self.ALLOWED | {
                "scripts/common/analytics.py"}
        ]
        assert offenders == [], (
            "Only analytics._frame_projection_sources and the engine may build "
            "per-source horizons: %s" % offenders)

    def test_engine_is_importable_without_streamlit(self):
        """The Actions snapshot collector installs requirements best-effort
        (`|| true` in fpl-notifications.yml), so these modules must not need
        Streamlit at import time."""
        import subprocess
        import sys

        code = (
            "import sys;"
            "import scripts.common.projection_engine;"
            "import scripts.common.projection_sources;"
            "import scripts.common.name_matching;"
            "assert 'streamlit' not in sys.modules, 'pulled in Streamlit';"
            "print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO,
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr


class TestRotowireCoverageMemo:
    """The omission penalty needs Rotowire's club coverage, and no page passes it.

    Coverage is a property of Rotowire's published table, not of the frame being
    blended -- a 15-player squad can never show 5 priced players at one club. So
    it is memoised at the fetch and read by the two analytics entry points,
    rather than threaded through ten page callsites where one would be forgotten.
    """

    def test_the_fetch_records_club_counts(self):
        import pandas as pd
        from scripts.common import projection_sources

        projection_sources._ROTOWIRE_CLUB_COVERAGE.clear()
        projection_sources._record_club_coverage(
            pd.DataFrame({"Team": ["LIV"] * 11 + ["ARS"] * 11}), limit=None)
        assert projection_sources.rotowire_club_coverage() == {"LIV": 11, "ARS": 11}

    def test_a_truncated_table_is_not_recorded(self):
        """A `limit` under-counts every club, which reads as an outage."""
        import pandas as pd
        from scripts.common import projection_sources

        projection_sources._ROTOWIRE_CLUB_COVERAGE.clear()
        projection_sources._record_club_coverage(
            pd.DataFrame({"Team": ["LIV"] * 3}), limit=20)
        assert projection_sources.rotowire_club_coverage() == {}

    def test_an_empty_memo_means_no_penalty(self):
        """Offline, in tests, before the first fetch: fail open, never punish."""
        import pandas as pd
        from scripts.common import projection_sources
        from scripts.common.analytics import blend_fixture_projections

        squad = pd.DataFrame({
            "Player_ID": [1, 2, 3], "Team": ["LIV"] * 3, "Position": ["M"] * 3,
            "Points": [5.0, 5.0, None],
            "FFP_Starting_Predicted": [5.0] * 3, "FFP_Start": [80, 80, 70],
        })
        projection_sources._ROTOWIRE_CLUB_COVERAGE.clear()
        out = blend_fixture_projections(squad.copy(), None)
        # FFP's own 70% stands untouched: with no coverage memo there is no
        # evidence that Rotowire *left him out*, only that it did not price him.
        assert out.loc[2, "Start_Pct"] == pytest.approx(0.70)

    def test_a_warm_memo_reaches_a_squad_sized_frame(self):
        import pandas as pd
        from scripts.common import projection_sources
        from scripts.common.analytics import blend_fixture_projections

        squad = pd.DataFrame({
            "Player_ID": [1, 2, 3], "Team": ["LIV"] * 3, "Position": ["M"] * 3,
            "Points": [5.0, 5.0, None],
            "FFP_Starting_Predicted": [5.0] * 3, "FFP_Start": [80, 80, 70],
        })
        projection_sources._ROTOWIRE_CLUB_COVERAGE.clear()
        projection_sources._ROTOWIRE_CLUB_COVERAGE.update({"LIV": 11})
        try:
            out = blend_fixture_projections(squad.copy(), None)
            assert out.loc[2, "Start_Pct"] < 0.70
        finally:
            projection_sources._ROTOWIRE_CLUB_COVERAGE.clear()


class TestUnconditionalBasisRecovery:
    """An unconditional source is un-discounted by its OWN start probability.

    Joao Pedro, live on 2026-09-17: FPL rated him 75% to play and published 6.1
    expected points; Rotowire omitted him, so the omission penalty pulled the
    app's resolved Start_Pct to 0.33. The engine divided 6.1 by 0.33 and showed
    18.5 points "if he starts" -- more than any single gameweek can produce.

    Every existing invariant passed, which is why it survived: Proj is
    Proj_Start x Start_Pct, so dividing and then multiplying by the same wrong
    number is exactly self-consistent. Both halves were wrong together.
    """

    @staticmethod
    def _blend(*, source_start=None, resolved_chance=25.0):
        import pandas as pd
        from scripts.common import projection_engine as engine

        index = [0]
        kwargs = dict(
            index=index,
            per_source_raw={"xp": pd.Series([6.1], index=index)},
            per_source_basis={"xp": engine.BASIS_UNCONDITIONAL},
            positions=pd.Series(["F"], index=index),
            chance_of_playing=pd.Series([resolved_chance], index=index),
            fallback_names=["xp"],
        )
        if source_start is not None:
            kwargs["per_source_startpct"] = {
                "xp": pd.Series([source_start], index=index)}
        return engine.blend_aligned(**kwargs)

    def test_the_sources_own_start_probability_is_used(self):
        out = self._blend(source_start=0.75, resolved_chance=25.0)
        assert out["Proj_Start"].iloc[0] == pytest.approx(6.1 / 0.75, abs=0.01)

    def test_the_resolved_start_probability_is_not_used_to_un_discount(self):
        """The regression. The resolved value is other sources' opinion; using
        it here divides one source's number by another's pessimism."""
        out = self._blend(source_start=0.75, resolved_chance=25.0)
        assert out["Proj_Start"].iloc[0] != pytest.approx(6.1 / 0.25, abs=0.01)

    def test_the_recovery_is_capped_at_a_two_fold_inflation(self):
        """A source with no stated basis falls back to the resolved value, but
        the divisor is floored at BASIS_RECOVERY_FLOOR.

        The conversion assumes the source discounted its number by exactly that
        probability, and FPL's ep is a model output rather than chance_of_playing
        times something -- so at 12% the assumption is guesswork and dividing by
        it would manufacture 50 points from 6.1.
        """
        from scripts.common.projection_engine import BASIS_RECOVERY_FLOOR
        out = self._blend(source_start=None, resolved_chance=12.0)
        assert out["Proj_Start"].iloc[0] == pytest.approx(
            6.1 / BASIS_RECOVERY_FLOOR, abs=0.01)

    def test_a_high_stated_probability_is_used_exactly(self):
        """Inside the range where the assumption holds, nothing is capped."""
        out = self._blend(source_start=0.75, resolved_chance=75.0)
        assert out["Proj_Start"].iloc[0] == pytest.approx(6.1 / 0.75, abs=0.01)

    def test_recovery_never_inflates_past_plausibility(self):
        from scripts.common.data_validation import (MAX_PLAUSIBLE_PROJ_START,
                                                    check_blended_projections)
        for chance in (1.0, 5.0, 12.0, 25.0, 50.0):
            out = self._blend(source_start=None, resolved_chance=chance)
            assert out["Proj_Start"].iloc[0] <= MAX_PLAUSIBLE_PROJ_START
            assert not [i for i in check_blended_projections(out)
                        if i.severity == "error"], chance

    def test_the_identity_still_holds_after_the_fix(self):
        out = self._blend(source_start=0.75, resolved_chance=25.0)
        row = out.iloc[0]
        assert row["Proj"] == pytest.approx(row["Proj_Start"] * row["Start_Pct"],
                                            abs=0.01)


class TestImplausibleProjStartIsCaught:
    """The check that would have caught it, since the identity could not."""

    def test_an_inflated_conditional_projection_is_an_error(self):
        import pandas as pd
        from scripts.common.data_validation import check_blended_projections
        # Exactly the shape of the live bug: self-consistent, and absurd.
        df = pd.DataFrame({"Proj": [6.1], "Proj_Start": [18.48], "Start_Pct": [0.33]})
        issues = check_blended_projections(df)
        assert any(i.severity == "error" and "if they start" in i.message
                   for i in issues), [str(i) for i in issues]

    def test_a_normal_blend_raises_nothing(self):
        import pandas as pd
        from scripts.common.data_validation import check_blended_projections
        df = pd.DataFrame({"Proj": [6.45, 2.68], "Proj_Start": [7.16, 8.13],
                           "Start_Pct": [0.90, 0.33]})
        assert not [i for i in check_blended_projections(df) if i.severity == "error"]


class TestFplEpCarriesItsOwnStartProbability:
    """The Joao Pedro regression, at the callsite that actually had it.

    ``blend_projections_onto`` declared fpl_ep as BASIS_UNCONDITIONAL but never
    put its start probability in ``per_source_startpct``, so the engine fell
    back to the app's resolved value -- which the Rotowire omission penalty had
    already pushed down. FPL's opinion was divided by Rotowire's pessimism.

    ``build_projections`` passed the same player correctly the whole time, so
    the two entry points disagreed: 8.13 against 18.48 for the same man.
    """

    @staticmethod
    def _pool():
        """One club, priced by Rotowire except the player under test.

        The club needs enough priced players to clear
        ROTOWIRE_MIN_CLUB_COVERAGE, or the omission penalty does not fire and
        the bug cannot reproduce.
        """
        import pandas as pd
        rows = [{"Player_ID": i, "Player": "Priced %d" % i, "Team": "CHE",
                 "Position": "M", "Points": 4.0, "ep_next": 4.0,
                 "chance_of_playing_next_round": None, "status": "a"}
                for i in range(1, 9)]
        rows.append({"Player_ID": 99, "Player": "Omitted Forward", "Team": "CHE",
                     "Position": "F", "Points": 0.0, "ep_next": 6.1,
                     "chance_of_playing_next_round": 75, "status": "d"})
        return pd.DataFrame(rows)

    def _blended(self):
        import pandas as pd
        from scripts.common import analytics
        from scripts.common.projection_engine import DEFAULT_MIN_CLUB_COVERAGE
        pool = self._pool()
        assert (pool["Team"] == "CHE").sum() > DEFAULT_MIN_CLUB_COVERAGE
        out = analytics.blend_projections_onto(pool, None, expected_gw=5)
        return out[out["Player_ID"] == 99].iloc[0]

    def test_the_omission_penalty_still_lowers_the_start_probability(self):
        """Guard on the premise: without this the test proves nothing."""
        assert self._blended()["Start_Pct"] < 0.75

    def test_points_if_he_starts_stay_plausible(self):
        from scripts.common.data_validation import MAX_PLAUSIBLE_PROJ_START
        row = self._blended()
        assert row["Proj_Start"] <= MAX_PLAUSIBLE_PROJ_START, (
            "6.1 expected points became %.1f if-he-starts -- FPL's number was "
            "divided by Rotowire's pessimism" % row["Proj_Start"])

    def test_it_is_un_discounted_by_fpls_own_chance_of_playing(self):
        row = self._blended()
        assert row["Proj_Start"] == pytest.approx(6.1 / 0.75, abs=0.05)

    def test_the_whole_frame_passes_the_blend_checks(self):
        from scripts.common import analytics
        from scripts.common.data_validation import check_blended_projections
        out = analytics.blend_projections_onto(self._pool(), None, expected_gw=5)
        assert not [i for i in check_blended_projections(out) if i.severity == "error"]
