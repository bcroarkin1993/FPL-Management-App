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
