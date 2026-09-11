"""Every frame built by name-matching against Rotowire must have its fixture
backfilled from the fixture list.

`merge_fpl_players_and_projections()` writes the literal string "N/A" into
Matchup for any player Rotowire did not list -- which is every player it does
not expect to start. `attach_matchups()` replaces that from the real fixture
list, where all 20 clubs are known.

This is a wiring test because plausibility cannot catch it: "N/A" beside a real
projection is not an implausible *number*, and the page renders perfectly. It
shipped by covering four of six callsites -- the two it missed were on the
branch that runs pre-deadline, which is when anyone looks at a projection.
"""

import ast
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
MERGE = "merge_fpl_players_and_projections"


def _merge_targets(tree):
    """Variables assigned directly from the Rotowire name-merge."""
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        fn = node.value.func
        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) != MERGE:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out.append((target.id, node.lineno))
    return out


def _attached(tree):
    """Variables passed as the first argument to attach_matchups()."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) != "attach_matchups":
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            out.add(node.args[0].id)
    return out


#: Pages that render a Matchup beside a projection. Others merge for the points
#: alone and never surface the column -- `fpl_draft_api` returns three columns,
#: the Waiver Wire renders cards with no fixture on them -- so requiring the
#: backfill there would be noise. The unmatched branch of the merge now leaves
#: Matchup empty rather than "N/A", so a page missing from this list shows no
#: fixture instead of asserting a wrong one.
MATCHUP_PAGES = [
    SCRIPTS / "draft" / "fixture_projections.py",
    SCRIPTS / "draft" / "team_analysis.py",
]


def _page_files():
    return MATCHUP_PAGES


@pytest.mark.parametrize("path", _page_files(), ids=lambda p: str(p.name))
def test_every_merged_frame_gets_its_fixture_backfilled(path):
    tree = ast.parse(path.read_text())
    attached = _attached(tree)
    missing = [(name, line) for name, line in _merge_targets(tree) if name not in attached]
    assert not missing, (
        f"{path.name}: {missing} came out of {MERGE}() and never reached "
        f"attach_matchups(), so every player Rotowire did not list renders "
        f'Matchup as the literal string "N/A".'
    )


def test_the_test_can_actually_fail():
    """A merge with no backfill must be detected."""
    tree = ast.parse("frame = merge_fpl_players_and_projections(a, b)\n")
    assert _merge_targets(tree) == [("frame", 1)]
    assert _attached(tree) == set()
