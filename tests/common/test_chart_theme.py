"""The chart theme has one home, and a tenth copy must be a reviewed act.

Nine page-local copies of ``_DARK_CHART_LAYOUT`` had grown up across the app in
five variants. Seven of them -- three groups that were exactly duplicated --
now alias ``scripts/common/charts.py``; two had genuinely diverged and are
allowlisted below, by name and with the reason.

This is the same shape of guard as ``test_matchup_wiring.py`` and
``test_element_id_wiring.py``, and it exists for the same reason: the failure it
catches is not a wrong number but a *shared thing silently stopping being
shared*. A tenth copy would look entirely ordinary in review.
"""

import ast
import io
import pathlib

import pytest

from scripts.common import charts

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Modules that still define their own layout, and why. Adding an entry here is
#: the reviewed act; adding a copy *without* touching this list fails the test.
DIVERGED = {
    "scripts/draft/league_analysis.py": "#333 gridlines, hovermode='x unified'",
    "scripts/draft/trade_analyzer.py": "12px tick fonts",
}


def _own_layout_definitions():
    """Every module that assigns ``_DARK_CHART_LAYOUT`` a literal of its own.

    An assignment from an attribute (``charts.DARK_LAYOUT``) is the migrated
    form and does not count -- that is the whole point of the alias.
    """
    found = {}
    for path in sorted((ROOT / "scripts").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        # The tracked " 2.py" duplicates are deliberate unmerged work the
        # project's notes say never to delete; they are not live callsites.
        if " 2.py" in rel:
            continue
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "_DARK_CHART_LAYOUT" not in names:
                continue
            if isinstance(node.value, ast.Attribute):
                continue
            found[rel] = type(node.value).__name__
    return found


def test_only_the_known_diverged_pages_define_their_own_layout():
    offenders = sorted(set(_own_layout_definitions()) - set(DIVERGED))
    assert not offenders, (
        "These modules define their own _DARK_CHART_LAYOUT instead of aliasing "
        "scripts/common/charts.py: %s. If the divergence is deliberate, add it "
        "to DIVERGED with the reason; otherwise alias the shared constant."
        % offenders
    )


def test_the_allowlisted_pages_really_still_diverge():
    """So the allowlist cannot rot into a list of pages that have been fixed."""
    defining = _own_layout_definitions()
    stale = sorted(set(DIVERGED) - set(defining))
    assert not stale, (
        "DIVERGED names %s, which no longer define their own layout -- drop "
        "the entry rather than leaving a guard that permits a copy nobody "
        "intends to add back." % stale
    )


@pytest.mark.parametrize("layout", [
    charts.DARK_LAYOUT, charts.DARK_LAYOUT_BIG_TITLE, charts.DARK_LAYOUT_BARE,
])
def test_every_layout_paints_its_own_background(layout):
    """A transparent plot area takes Streamlit's ground, not the app's."""
    assert layout["paper_bgcolor"] == charts.PANEL
    assert layout["plot_bgcolor"] == charts.PANEL
    assert layout["font"]["color"] == charts.TEXT


def test_the_three_variants_stay_distinct():
    """They are three because the originals were three, not for tidiness.

    Collapsing two of them restyles real pages -- the title size and the
    presence of axis styling are the differences -- so a change that makes any
    two equal should be a deliberate one with a before/after, not a refactor.
    """
    assert charts.DARK_LAYOUT != charts.DARK_LAYOUT_BIG_TITLE
    assert charts.DARK_LAYOUT != charts.DARK_LAYOUT_BARE
    assert charts.DARK_LAYOUT_BIG_TITLE != charts.DARK_LAYOUT_BARE
    # The bare variant's defining property: it styles no axes, so a page using
    # it is expected to set its own.
    assert "xaxis" not in charts.DARK_LAYOUT_BARE
    assert "xaxis" in charts.DARK_LAYOUT


class TestDarkLayoutMerges:
    """``dark_layout`` exists to avoid the collision in the module docstring.

    ``update_layout(**DARK_LAYOUT, xaxis=...)`` raises ``TypeError: got
    multiple values for keyword argument 'xaxis'``, and the obvious workaround
    -- a dict literal -- throws the theme's gridcolour away silently. This is
    the behaviour that makes the helper worth having over either.
    """

    def test_an_axis_override_keeps_the_themes_own_axis_styling(self):
        got = charts.dark_layout(xaxis=dict(title="Points"))
        assert got["xaxis"]["title"] == "Points"
        assert got["xaxis"]["gridcolor"] == charts.GRID, (
            "merging must be one level deep -- replacing the axis dict "
            "outright is the silent half of the bug this helper avoids"
        )
        assert got["xaxis"]["tickfont"]["color"] == charts.TEXT

    def test_a_new_key_is_simply_added(self):
        got = charts.dark_layout(height=310)
        assert got["height"] == 310
        assert got["paper_bgcolor"] == charts.PANEL

    def test_a_nested_dict_is_replaced_not_recursed(self):
        """The shallow rule, asserted so it stays predictable."""
        got = charts.dark_layout(xaxis=dict(tickfont=dict(size=9)))
        assert got["xaxis"]["tickfont"] == {"size": 9}

    def test_the_module_constant_cannot_be_reached_through_the_result(self):
        got = charts.dark_layout()
        got["xaxis"]["gridcolor"] = "#ff0000"
        assert charts.DARK_LAYOUT["xaxis"]["gridcolor"] == charts.GRID

    def test_another_base_can_be_passed(self):
        got = charts.dark_layout(charts.DARK_LAYOUT_BARE, height=200)
        assert got["height"] == 200
        assert "xaxis" not in got
