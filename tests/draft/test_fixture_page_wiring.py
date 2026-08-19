"""Wiring tests for the Draft Fixture Projections page.

The Fixtures Overview table and the Detailed Match Analysis below it are two
separate callsites of the same analysis function. They diverged in production:
the overview omitted `ffp_df`, so blend_fixture_projections took its no-FFP
early return, Proj_Blended came back equal to raw Rotowire Points, and the two
sections printed different projected scores (and different win probabilities)
for the same fixture.

A plausibility test can't catch that -- both numbers are individually plausible.
What catches it is asserting that both callsites are fed the same inputs, which
is a wiring property and belongs offline with mocks.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import scripts.draft.fixture_projections as fx


@pytest.fixture
def captured_analysis_calls(mock_streamlit):
    """Run show_fixtures_page() with everything stubbed, capturing every
    analyze_fixture_projections() call so the two callsites can be compared."""
    calls = []

    def _record(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return None  # every fixture "fails to resolve" -- we only want the call

    ffp_sentinel = pd.DataFrame({"Name": ["A"], "Team": ["ARS"], "Predicted": [4.0]})
    rotowire_sentinel = pd.DataFrame({
        "Player": ["A"], "Team": ["ARS"], "Position": ["M"], "Points": [4.0],
        "Matchup": ["ARS v CHE"], "Pos Rank": [1],
    })

    with patch.object(fx, "analyze_fixture_projections", side_effect=_record), \
         patch.object(fx, "get_rotowire_player_projections", return_value=rotowire_sentinel), \
         patch.object(fx, "get_ffp_projections_data", return_value=ffp_sentinel), \
         patch.object(fx, "get_gameweek_fixtures", return_value=["Team A vs Team B"]), \
         patch.object(fx, "get_current_gameweek", return_value=1), \
         patch.object(fx, "is_gameweek_live", return_value=False), \
         patch.object(fx, "_estimate_score_std", return_value=(15.0, 300)), \
         patch.object(fx, "get_live_gameweek_stats", return_value={}), \
         patch.object(fx, "get_fpl_player_mapping", return_value={}), \
         patch.object(fx, "compute_key_differentials", return_value=pd.DataFrame()), \
         patch.object(fx, "render_key_differentials", MagicMock()), \
         patch.object(fx, "components", MagicMock()):
        fx.show_fixtures_page()

    return calls, ffp_sentinel


class TestOverviewAndDetailUseTheSameInputs:
    def test_both_sections_analyse_the_fixture(self, captured_analysis_calls):
        calls, _ = captured_analysis_calls
        assert len(calls) >= 2, (
            "expected the overview and the detail view to each analyse the fixture, "
            "got %d call(s)" % len(calls)
        )

    def test_every_callsite_receives_the_ffp_projections(self, captured_analysis_calls):
        """The actual bug: the overview called this with no ffp_df at all."""
        calls, ffp_sentinel = captured_analysis_calls
        missing = object()
        for i, call in enumerate(calls):
            ffp = call["kwargs"].get("ffp_df", missing)
            assert ffp is not missing, (
                "callsite %d omitted ffp_df; it will silently fall back to raw "
                "Rotowire points while the other section uses the blend" % i
            )
            assert ffp is ffp_sentinel, (
                "callsite %d passed a different ffp_df than the page fetched" % i
            )

    def test_every_callsite_receives_the_same_projections_frame(self, captured_analysis_calls):
        calls, _ = captured_analysis_calls
        frames = [call["args"][2] if len(call["args"]) > 2 else call["kwargs"].get("projections_df")
                  for call in calls]
        first = frames[0]
        for i, frame in enumerate(frames[1:], start=1):
            assert frame is first, "callsite %d used a different projections frame" % i

    def test_every_callsite_agrees_on_live_mode(self, captured_analysis_calls):
        """Overview and detail must not disagree about whether the GW is live --
        one showing blended live scores while the other shows pre-match ones."""
        calls, _ = captured_analysis_calls
        modes = {call["kwargs"].get("use_actual_lineup") for call in calls}
        assert len(modes) == 1, "callsites disagree about use_actual_lineup: %s" % modes
