"""Both scored frames must measure Form over the same window.

`_add_fdr_and_form(df, fpl_stats, current_gw, weeks, form_weeks=None)` falls
back to `config.FORM_LOOKBACK_WEEKS` when `form_weeks` is omitted. The Waiver
Wire has two callsites -- the available pool and the roster -- and for as long
as the "Form lookback GWs" control has existed only the pool was passed it.

So moving that slider off its default measured the pool's Form over N
gameweeks and the roster's over 4. The two are then compared *directly*:
`_build_rationale` sets `drop_form` against `add_form` on every suggestion
card, both frames render a `Form` column side by side, and the value feeds
`compute_player_scores`'s form term, which is 10-15% of ROS. A player's Form
is a different quantity on each side of a suggestion.

Nothing could see it. At the default slider value the two windows agree, so
every test and every ordinary page load produced identical numbers; the
divergence only appears once a user touches the control, and even then both
numbers stay entirely plausible.

This is a wiring property -- "two callsites of one function are fed the same
arguments" -- so it belongs offline, in the AST, next to
`test_element_id_wiring.py` and `test_matchup_wiring.py`, which exist for the
same shape of failure on the same page family.
"""

import ast
import pathlib

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
FN = "_add_fdr_and_form"

#: Pages that score a roster against an available pool, so the two frames'
#: Form columns are read together.
PAGES = [SCRIPTS / "draft" / "waiver_wire.py"]


def _calls(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if (getattr(fn, "id", None) or getattr(fn, "attr", None)) == name:
                yield node


def _passes(call, kwarg, positional_index):
    """True when the argument is supplied, by keyword or by position."""
    if any(kw.arg == kwarg for kw in call.keywords):
        return True
    return len(call.args) > positional_index


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_every_callsite_sets_the_form_window(path):
    tree = ast.parse(path.read_text())
    calls = list(_calls(tree, FN))
    assert len(calls) >= 2, (
        "expected the roster and the available pool to be scored separately; "
        "found %d %s call(s) in %s" % (len(calls), FN, path.name))

    # form_weeks is the 5th parameter: df, fpl_stats, current_gw, weeks, form_weeks
    missing = [c.lineno for c in calls if not _passes(c, "form_weeks", 4)]
    assert missing == [], (
        "%s calls %s without form_weeks at line(s) %s. That callsite silently "
        "falls back to config.FORM_LOOKBACK_WEEKS while the other uses the "
        "widget, so the two frames' Form columns are measured over different "
        "windows -- and they are compared directly in _build_rationale and in "
        "the ROS form term. Pass form_weeks=form_weeks."
        % (path.name, FN, missing))


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_every_callsite_sets_the_fixture_window(path):
    """The same argument, for the same reason: AvgFDRNextN is percentiled
    across both frames, so a 3-week pool against a 5-week roster is two
    populations compared as one."""
    tree = ast.parse(path.read_text())
    calls = list(_calls(tree, FN))
    missing = [c.lineno for c in calls if not _passes(c, "weeks", 3)]
    assert missing == [], (
        "%s calls %s without the fixture lookahead at line(s) %s"
        % (path.name, FN, missing))
