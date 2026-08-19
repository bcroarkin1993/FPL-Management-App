"""Tests for the Draft fixture win-probability inputs.

Regression coverage for a preseason bug: the Draft API returns a full 38-gameweek
score grid from day one, so before the season starts every historical score is 0.
_estimate_score_std returned std=0.0, the caller's `sigma > 0` guard substituted a
denominator of 1.0, and a 1.1-point projection edge (55.8 vs 54.7) rendered as an
85%/15% win probability.
"""

import numpy as np
import pandas as pd

from scripts.draft.fixture_projections import (
    _DEFAULT_SCORE_STD,
    _estimate_score_std,
    _normal_cdf,
    _weekly_score_std,
    _winprob_denom,
)


def _grid(played_events, n_teams=10, total_events=38, seed=0):
    """38-gameweek x n_teams score grid; unplayed gameweeks are all zeros,
    exactly as the Draft API reports them."""
    rng = np.random.default_rng(seed)
    rows = []
    for event in range(1, total_events + 1):
        for team in range(n_teams):
            pts = float(rng.normal(50, 14)) if event in played_events else 0.0
            rows.append({"event": event, "entry_id": team, "points": pts})
    df = pd.DataFrame(rows)
    df["total_points"] = df.groupby("entry_id")["points"].cumsum()
    return df


class TestWeeklyScoreStd:
    def test_preseason_all_zero_grid_yields_no_estimate(self):
        """The live bug: 380 rows of zeros is not a std of 0, it's no data."""
        assert _weekly_score_std(_grid(played_events=set())) is None

    def test_unplayed_gameweeks_do_not_deflate_the_estimate(self):
        """Only 3 of 38 gameweeks played -- the 35 all-zero weeks must not count."""
        df = _grid(played_events={1, 2, 3})
        std = _weekly_score_std(df)
        assert std is not None
        # Estimated from the 30 real scores only, so it lands near the true sigma of 14.
        assert 8 < std < 20
        # A naive std over all 380 rows is skewed by the 350 zeros sitting ~50 below
        # the mean; the filtered estimate must not equal it.
        naive = df["points"].std(ddof=1)
        assert abs(std - naive) > 0.5

    def test_prefers_weekly_points_over_cumulative_total(self):
        """total_points is a running season total; its spread is a different
        (much larger) quantity than the spread of one week's scores."""
        df = _grid(played_events=set(range(1, 39)))
        weekly = _weekly_score_std(df)
        cumulative = df["total_points"].std(ddof=1)
        assert weekly is not None
        assert weekly < cumulative / 2

    def test_recovers_weekly_scores_when_only_cumulative_is_available(self):
        df = _grid(played_events=set(range(1, 39)))[["event", "entry_id", "total_points"]]
        std = _weekly_score_std(df)
        assert std is not None
        assert 10 < std < 20  # true sigma is 14

    def test_empty_or_missing_frame_yields_no_estimate(self):
        assert _weekly_score_std(None) is None
        assert _weekly_score_std(pd.DataFrame()) is None
        assert _weekly_score_std(pd.DataFrame({"foo": [1, 2, 3]})) is None


class TestEstimateScoreStd:
    def test_falls_back_to_prior_when_history_is_unusable(self, monkeypatch):
        monkeypatch.setattr(
            "scripts.draft.fixture_projections.get_historical_team_scores",
            lambda _lid: _grid(played_events=set()),
        )
        monkeypatch.setattr(pd, "read_csv", lambda *_a, **_k: pd.DataFrame())
        sigma, n = _estimate_score_std(123)
        assert sigma == _DEFAULT_SCORE_STD
        assert n == 0  # signals "prior, not a real estimate" to the UI caption


class TestWinProbDenom:
    def test_degenerate_sigma_uses_the_prior_not_one(self):
        """A denominator of 1.0 turns a 1-point edge into an ~85% win call."""
        for bad in (0.0, -1.0, float("nan"), None):
            assert _winprob_denom(bad) == _winprob_denom(_DEFAULT_SCORE_STD)

    def test_close_matchup_is_close_to_a_coin_flip(self):
        """55.8 vs 54.7 -- the fixture that rendered as 85%/15%."""
        pct = _normal_cdf((55.8 - 54.7) / _winprob_denom(_DEFAULT_SCORE_STD)) * 100
        assert 50 < pct < 55

    def test_lopsided_matchup_still_reads_lopsided(self):
        pct = _normal_cdf((53.3 - 25.6) / _winprob_denom(_DEFAULT_SCORE_STD)) * 100
        assert pct > 85
