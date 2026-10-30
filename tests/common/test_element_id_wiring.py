"""Every frame that reaches the projection engine must carry the element id.

The engine's Rotowire projected-XI start signal is keyed on the FPL element id,
because that is the only thing an XI can be resolved to. A frame without one is
*silently* skipped -- `analytics._frame_starter_overrides` returns `{}`, start
likelihood falls back to the stale weekly rankings article, and the page renders
a perfectly ordinary-looking number.

It shipped that way twice in one sitting:

1. keyed on `Player_ID` alone, so the Classic squad builders' `element_id`
   frames were skipped -- fixed by `analytics._player_id_col`;
2. `get_team_composition_for_gameweek` had the element id in hand and dropped
   it, and the two Draft pages merging from it passed no `carry_cols`. Those are
   the *pre-deadline* paths, which is when a projection is read. Phil Foden
   rendered at 4.2 there while the engine had him at 0.45, which is exactly the
   bug the signal was built to fix.

This is a wiring test because plausibility cannot catch it: both numbers are
plausible, and the difference is the rotation risk the blend exists to price.
It is the same shape of failure, on the same two callsites, as
`test_matchup_wiring.py`.
"""

import ast
import pathlib

import pandas as pd
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
MERGE = "merge_fpl_players_and_projections"

#: Pages whose merged frame goes on to be blended, so the element id has to
#: survive the merge. `merge_fpl_players_and_projections` takes a matched row's
#: name from the *projection* source, so nothing survives unless asked for.
BLENDING_PAGES = [
    SCRIPTS / "draft" / "fixture_projections.py",
    SCRIPTS / "draft" / "team_analysis.py",
]


def _merge_calls(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) == MERGE:
            yield node


def _carries_player_id(call):
    for kw in call.keywords:
        if kw.arg != "carry_cols":
            continue
        if isinstance(kw.value, (ast.List, ast.Tuple)):
            return any(getattr(el, "value", None) == "Player_ID"
                       for el in kw.value.elts)
        return True            # a variable; assume the caller knows
    return False


@pytest.mark.parametrize("path", BLENDING_PAGES, ids=lambda p: p.name)
def test_every_merge_carries_the_element_id(path):
    tree = ast.parse(path.read_text())
    missing = [c.lineno for c in _merge_calls(tree) if not _carries_player_id(c)]
    assert missing == [], (
        "%s merges Rotowire projections without carry_cols=['Player_ID'] at "
        "line(s) %s. The blend cannot then apply anything keyed on the element "
        "id -- Rotowire's projected XIs included -- and nothing on the page "
        "shows it." % (path.name, missing))


def test_the_draft_roster_frame_publishes_the_element_id():
    """`get_team_composition_for_gameweek` is the pre-deadline roster for both
    Draft pages above, and it had `pid` in hand while returning three columns."""
    import inspect
    from scripts.common import fpl_draft_api

    src = inspect.getsource(fpl_draft_api.get_team_composition_for_gameweek)
    assert "'Player_ID': pid" in src, (
        "the composition frame must publish the element id it already resolves")
    # Membership, not an exact column list: pinning the list meant any
    # legitimate addition failed this test, which is a guard that punishes
    # unrelated work rather than catching the regression it names.
    for col in ("Player_ID", "Player_Code"):
        assert col in fpl_draft_api._COMPOSITION_COLUMNS, (
            "the composition frame must publish %s" % col)
    assert "Player_ID" in fpl_draft_api._ACTUAL_LINEUP_COLUMNS


class TestOverrideReachesBothFrameShapes:
    """The two real shapes, as the pages actually build them."""

    @staticmethod
    def _override(df):
        from unittest.mock import patch
        from scripts.common import analytics
        from scripts.common import rotowire_lineups as rlm
        # `_lineup_xis` is the seam: `_frame_starter_overrides` builds one
        # override per publication and the engine combines them.
        with patch.object(analytics, "_lineup_xis",
                          return_value={"rotowire": rlm.LineupXI({367}, {"MCI"}, 8)}):
            return analytics._frame_starter_overrides(df)

    def test_the_draft_shape(self):
        """Player_ID, as carried through merge_fpl_players_and_projections."""
        df = pd.DataFrame({"Player": ["Phil Foden", "Cody Gakpo"],
                           "Player_ID": [398, 367], "Team": ["MCI", "MCI"],
                           "Position": ["M", "M"]})
        ov = self._override(df)["rotowire"]
        assert list(ov) == [0.0, 1.0]

    def test_the_classic_shape(self):
        """element_id, as the Classic squad builders emit it."""
        df = pd.DataFrame({"Player": ["Phil Foden", "Cody Gakpo"],
                           "element_id": [398, 367], "Team": ["MCI", "MCI"],
                           "Position": ["M", "M"]})
        ov = self._override(df)["rotowire"]
        assert list(ov) == [0.0, 1.0]
