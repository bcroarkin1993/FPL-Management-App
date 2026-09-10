"""Tests for projection accuracy scoring.

The point of this module is to replace an assumption (60/40) with a
measurement, so the tests care most about the ways a measurement can lie:
scoring an if-he-starts number against a player who never started, comparing
sources across different populations, and letting a backfill move the weights.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common import projection_accuracy as acc


def _pre(**over):
    base = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "proj": [6.0, 4.0, 1.0, 0.5],
        "proj_start": [7.0, 5.0, 4.0, 3.0],
        "proj_start__rotowire": [8.0, 6.0, np.nan, np.nan],
        "proj_start__ffp": [6.0, 4.0, 4.0, 3.0],
        "proj_start__fpl_ep": [5.0, 5.0, 3.0, 2.0],
    })
    for k, v in over.items():
        base[k] = v
    return base


def _actual(**over):
    base = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        # Player 3 starts but Rotowire never priced him -- that gap between
        # sources is exactly what the common subset exists to neutralise.
        "points": [8, 2, 3, 0],
        "minutes": [90, 90, 90, 20],
        "started": [1, 1, 1, 0],
    })
    for k, v in over.items():
        base[k] = v
    return base


@pytest.fixture
def _archive(monkeypatch):
    """One scoreable gameweek, captured pre-deadline unless a test says otherwise."""
    state = {"meta": {"gameweek": 3, "captured_before_deadline": True}}
    monkeypatch.setattr(acc.projection_archive, "load_pre",
                        lambda gw: (_pre(), state["meta"]))
    monkeypatch.setattr(acc.projection_archive, "load_actuals",
                        lambda gw: (_actual(), {}))
    monkeypatch.setattr(acc.projection_archive, "scoreable_gameweeks", lambda: [3])
    return state


class TestSpearmanWithoutScipy:
    """pandas' spearman imports scipy, which is not a dependency here and which
    the scheduled workflow would then install on every run."""

    def test_matches_the_known_value(self):
        a = pd.Series([1, 2, 3, 4, 5.0])
        b = pd.Series([2, 1, 4, 3, 5.0])
        assert acc.spearman(a, b) == pytest.approx(0.8)

    def test_perfect_and_inverse_rank_agreement(self):
        a = pd.Series([1, 2, 3, 4.0])
        assert acc.spearman(a, a) == pytest.approx(1.0)
        assert acc.spearman(a, a[::-1].reset_index(drop=True)) == pytest.approx(-1.0)

    def test_degenerate_inputs_are_nan_not_a_crash(self):
        assert np.isnan(acc.spearman(pd.Series([1.0]), pd.Series([1.0])))
        assert np.isnan(acc.spearman(pd.Series([1, 1, 1.0]), pd.Series([1, 2, 3.0])))


class TestMetrics:
    def test_mae_rmse_and_bias(self):
        m = acc._metrics(pd.Series([3.0, 5.0]), pd.Series([1.0, 2.0]))
        assert m["mae"] == pytest.approx(2.5)
        assert m["rmse"] == pytest.approx(np.sqrt((4 + 9) / 2))
        assert m["bias"] == pytest.approx(2.5)

    def test_bias_is_signed(self):
        """A source 0.4 points high every week is a different problem from one
        that is noisy, and MAE cannot tell them apart."""
        under = acc._metrics(pd.Series([1.0, 1.0]), pd.Series([3.0, 3.0]))
        assert under["bias"] == pytest.approx(-2.0)
        assert under["mae"] == pytest.approx(2.0)

    def test_no_overlap_is_reported_as_zero_not_a_crash(self):
        m = acc._metrics(pd.Series([np.nan]), pd.Series([1.0]))
        assert m["n"] == 0 and np.isnan(m["mae"])


class TestScopes:
    def test_starters_scope_excludes_players_who_did_not_start(self, _archive):
        """proj_start is 'points if he starts'. Scoring it against someone who
        never started measures the minutes model, not the points model."""
        s = acc.score_gameweek(3)
        row = s[(s.source == "blend") & (s.scope == "starters")].iloc[0]
        assert row["n"] == 3          # players 1-3 started; player 4 did not

    def test_all_scope_keeps_the_full_population(self, _archive):
        """Including players who never played and scored zero -- that is a
        projection error like any other."""
        s = acc.score_gameweek(3)
        row = s[(s.source == "blend") & (s.scope == "all")].iloc[0]
        assert row["n"] == 4

    def test_only_the_blend_reports_on_the_all_scope(self, _archive):
        """The per-source columns are all conditional, so reporting them over
        every player would compare an if-he-starts number against non-starters."""
        s = acc.score_gameweek(3)
        assert set(s[s.scope == "all"]["source"]) == {"blend"}


class TestCommonSubsetMakesSourcesComparable:
    def test_without_it_sources_are_scored_on_different_populations(self, _archive):
        s = acc.score_gameweek(3, common_subset=False)
        starters = s[s.scope == "starters"].set_index("source")["n"]
        # Rotowire prices only two of these players; FFP prices all four.
        assert starters["rotowire"] != starters["ffp"]

    def test_with_it_every_source_is_scored_on_the_same_players(self, _archive):
        s = acc.score_gameweek(3, common_subset=True)
        starters = s[s.scope == "starters"].set_index("source")["n"]
        assert starters.nunique() == 1

    def test_the_all_scope_is_not_shrunk_by_the_common_subset(self, _archive):
        """Its whole point is the full population. Restricting it to players
        every source priced would drop the non-starters and quietly turn the
        end-to-end number into a starters-only one."""
        s = acc.score_gameweek(3, common_subset=True)
        row = s[(s.source == "blend") & (s.scope == "all")].iloc[0]
        assert row["n"] == 4

    def test_coverage_is_reported_against_the_whole_pool(self, _archive):
        s = acc.score_gameweek(3, common_subset=True)
        rw = s[(s.source == "rotowire")].iloc[0]
        assert rw["coverage"] == pytest.approx(0.5)   # 2 of 4 priced


class TestBackfillsNeverMoveTheWeights:
    def test_a_backfill_is_labelled(self, _archive):
        _archive["meta"]["captured_before_deadline"] = False
        s = acc.score_gameweek(3)
        assert (s["capture"] == "backfill").all()

    def test_weight_fitting_refuses_a_backfill(self, _archive):
        """A backfill can see team news -- in the limit, lineups -- that no
        manager had. Letting it set the weights the app runs on is exactly how a
        measurement harness makes things worse."""
        _archive["meta"]["captured_before_deadline"] = False
        out = acc.fit_blend_weights()
        assert out["fitted"] is None
        assert out["n"] == 0

    def test_weight_fitting_uses_a_pre_deadline_gameweek(self, _archive):
        out = acc.fit_blend_weights(step=0.5)
        assert out["fitted"] is not None
        assert sum(out["fitted"].values()) == pytest.approx(1.0)
        assert out["n"] == 2          # only the two who started
        # The fit cannot be worse than the configured weights on its own data.
        assert out["mae"] <= out["current_mae"] + 1e-9


class TestSimplex:
    def test_every_weight_vector_sums_to_one(self):
        for w in acc._simplex(3, 0.25):
            assert sum(w) == pytest.approx(1.0)

    def test_grid_is_complete(self):
        assert len(acc._simplex(2, 0.5)) == 3      # (0,1) (0.5,0.5) (1,0)


class TestSummaryAndConfidence:
    def test_summary_weights_gameweeks_by_sample_size(self):
        scored = pd.DataFrame({
            "gameweek": [1, 2], "source": ["ffp", "ffp"], "scope": ["starters"] * 2,
            "n": [10, 90], "mae": [1.0, 2.0], "rmse": [1.0, 2.0],
            "bias": [0.0, 0.0], "spearman": [0.5, 0.5], "coverage": [0.5, 0.5],
        })
        out = acc.summarise(scored)
        # A gameweek where 10 players were priced must not count as much as one
        # where 90 were: (10*1 + 90*2) / 100 = 1.9, not the unweighted 1.5.
        assert out.iloc[0]["mae"] == pytest.approx(1.9)

    def test_confidence_note_says_so_when_there_is_nothing(self):
        assert "No gameweek yet" in acc.confidence_note(pd.DataFrame())

    def test_confidence_note_warns_on_a_small_sample(self):
        scored = pd.DataFrame({"gameweek": [3]})
        note = acc.confidence_note(scored)
        assert "noise" in note and "1 gameweek" in note

    def test_confidence_note_is_plain_once_the_sample_is_adequate(self):
        scored = pd.DataFrame({"gameweek": list(range(1, 9))})
        assert acc.confidence_note(scored) == "8 gameweeks of history."
