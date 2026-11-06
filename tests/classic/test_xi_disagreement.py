"""The badge that reconciles two projected XIs.

Reported from the app: Rayan Cherki appeared in the Classic Fixture Projections
XI on 5.20 projected, while the app's own Projected Lineups page -- which shows
**Rotowire** -- had him out of a complete Man City eleven.

Neither page was wrong. The engine combines the XIs *unanimous-to-bench*: any
source naming a player promotes him to the positional start floor, and he is
benched only where every source with an opinion leaves him out. So
Sportsgambler naming Cherki lifted him to the midfield floor of 0.86 while
Rotowire had him benched. Nothing on screen said they were different sources.
"""

import re

import pandas as pd
import pytest

from scripts.classic.fixture_projections import _XI_SOURCE_LABELS, _xi_split_badge


def _row(**kw):
    return pd.Series(kw)


class TestItFlagsOnlyRealDisagreement:
    def test_the_reported_case(self):
        got = _xi_split_badge(_row(Start_XI__rotowire=0.0,
                                   Start_XI__sportsgambler=1.0))
        assert got, "the sources disagree and nothing said so"
        assert "Rotowire" in got
        assert "xi-split" in got

    def test_it_names_whichever_source_left_him_out(self):
        got = _xi_split_badge(_row(Start_XI__rotowire=1.0,
                                   Start_XI__sportsgambler=0.0))
        assert "Sportsgambler" in got and "Rotowire" not in got

    def test_agreement_is_silent(self):
        assert _xi_split_badge(_row(Start_XI__rotowire=1.0,
                                    Start_XI__sportsgambler=1.0)) == ""

    def test_a_unanimous_benching_is_silent(self):
        """Both leave him out, so `Start_Pct` already reflects it. There is no
        disagreement to reconcile -- only a low projection, which is correct."""
        assert _xi_split_badge(_row(Start_XI__rotowire=0.0,
                                    Start_XI__sportsgambler=0.0)) == ""

    def test_one_source_benching_with_the_other_silent_is_silent(self):
        """**No opinion is not a vote.** Counting against the number of source
        labels rather than the number with an opinion flagged a player one
        source benched and the other had not published on -- where the engine
        benches him too, so there is nothing to show."""
        assert _xi_split_badge(_row(Start_XI__rotowire=0.0)) == ""
        assert _xi_split_badge(_row(Start_XI__sportsgambler=0.0)) == ""

    def test_no_xi_data_at_all_is_silent(self):
        assert _xi_split_badge(_row()) == ""
        assert _xi_split_badge(_row(Start_XI__rotowire=float("nan"),
                                    Start_XI__sportsgambler=float("nan"))) == ""

    def test_it_renders_one_balanced_element(self):
        """It is injected into a `components.html` card whose height is computed
        in Python, so it must stay inline and self-closing."""
        got = _xi_split_badge(_row(Start_XI__rotowire=0.0,
                                   Start_XI__sportsgambler=1.0))
        assert got.count("<span") == 1 and got.count("</span>") == 1
        assert "\n" not in got


def test_the_labels_match_the_sources_the_engine_is_actually_given():
    """`Start_XI__<source>` is named from the keys `analytics` passes as
    `starter_overrides`, so a renamed or added source would switch this badge
    off -- or leave a new source unreconciled -- silently rather than erroring.

    This is the same class of guard as `test_matchup_wiring`: no plausibility
    check can see it, because every number on the card stays correct.
    """
    import inspect
    from scripts.common import analytics as an

    src = inspect.getsource(an._lineup_xis)
    for key in _XI_SOURCE_LABELS:
        assert '"%s"' % key in src or "'%s'" % key in src, (
            "%r is not a source analytics feeds the engine, so Start_XI__%s is "
            "never written and the badge is dead code for it" % (key, key))

    # And the reverse: a source the engine is given but this badge does not know
    # about would disagree with the XI on screen and never be flagged.
    named = set(re.findall(r'\(\s*["\']([a-z_]+)["\']\s*,\s*["\']get_', src))
    assert named <= set(_XI_SOURCE_LABELS), (
        "projected-XI source(s) %s are blended but carry no label here, so a "
        "disagreement involving them renders no badge"
        % sorted(named - set(_XI_SOURCE_LABELS)))
