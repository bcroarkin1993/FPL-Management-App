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


# ---------------------------------------------------------------------------
# The start model. `Proj = Proj_Start x Start_Pct`, so this half is on every
# board in the app and nothing scored it until now.
# ---------------------------------------------------------------------------


def _pre_start(self_consistent=False, **over):
    """A pre frame carrying start probabilities as well as projections.

    Ten players at one club, because the omission signal only fires once a
    source has priced enough of a club for its silence to mean anything -- and
    it is the engine's own `Start_Pct__rotowire` column, written on that path,
    that records which constants built a snapshot.

    `self_consistent=True` replaces the stored start columns with what the
    engine actually produces from these inputs, which is what an archived
    snapshot is. The fitter refuses to run on anything else, so a hand-written
    number here would only ever exercise the refusal.
    """
    base = pd.DataFrame({
        "player_id": list(range(1, 11)),
        "position": ["M", "M", "M", "D", "D", "G", "M", "M", "D", "F"],
        "team": ["ARS"] * 10,
        "proj": [6.0, 5.0, 4.0, 3.0, 3.0, 3.0, 1.0, 0.8, 0.6, 0.4],
        "proj_start": [7.0] * 6 + [4.0] * 4,
        # Players 1-6 are Rotowire's expected XI at this club; 7-10 are its
        # silence about the rest of the squad.
        "proj_start__rotowire": [8.0, 7.0, 6.0, 5.0, 5.0, 5.0] + [np.nan] * 4,
        "proj_start__ffp": [6.0, 6.0, 5.0, 4.0, 4.0, 4.0, 4.0, 3.0, 3.0, 2.0],
        "proj_start__fpl_ep": [5.0, 5.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0],
        # Every source has a start opinion about every player: the start scope's
        # common subset is drawn from these columns, not from the points ones.
        "start_pct__ffp": [0.90, 0.85, 0.80, 0.85, 0.80, 0.95,
                           0.40, 0.25, 0.20, 0.10],
        "start_pct__fpl_ep": [1.00] * 10,
        "start_pct": [0.90, 0.85, 0.80, 0.85, 0.80, 0.95,
                      0.20, 0.15, 0.12, 0.05],
    })
    for k, v in over.items():
        base[k] = v
    if self_consistent:
        engine = _engine_start_columns(base)
        base["start_pct"] = list(engine["Start_Pct"])
        # The engine writes this itself, and it is what the fitter reads back to
        # learn which constants built the snapshot. Hand-writing it would make
        # the fixture claim it was built by constants that did not build it.
        base["start_pct__rotowire"] = list(engine["Start_Pct__rotowire"])
    return base


def _actual_start(**over):
    """Six of the ten start: Rotowire's XI minus one, plus nobody it omitted."""
    base = pd.DataFrame({
        "player_id": list(range(1, 11)),
        "points": [8, 6, 2, 5, 1, 3, 0, 0, 1, 0],
        "minutes": [90, 90, 90, 90, 90, 90, 0, 0, 12, 0],
        "started": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    })
    for k, v in over.items():
        base[k] = v
    return base


def _engine_start_columns(pre: pd.DataFrame) -> pd.DataFrame:
    """What the engine resolves these inputs to, as an archived snapshot would.

    Built by hand rather than through the archive, which is what the fixture is
    in the middle of constructing.
    """
    from scripts.common import projection_engine
    from scripts.common.projection_sources import BASIS_CONDITIONAL

    one = pre.drop_duplicates(subset="player_id").set_index("player_id")
    raw = {n: one[f"proj_start__{n}"] for n in ("rotowire", "ffp", "fpl_ep")}
    spct = {n: one[f"start_pct__{n}"] for n in ("ffp", "fpl_ep")}
    priced = raw["rotowire"].gt(0).fillna(False)
    return projection_engine.blend_aligned(
        index=one.index,
        per_source_raw=raw,
        per_source_basis={n: BASIS_CONDITIONAL for n in raw},
        per_source_startpct=spct,
        starters_only={"rotowire"},
        positions=one["position"],
        teams=one["team"],
        source_club_coverage={"rotowire": {
            k: int(v) for k, v in one.loc[priced, "team"].astype(str)
            .value_counts().items()}},
        gameweek=4,
    )


@pytest.fixture
def _start_archive(monkeypatch):
    state = {"meta": {"gameweek": 4, "captured_before_deadline": True},
             "pre": _pre_start(self_consistent=True), "actual": _actual_start()}
    monkeypatch.setattr(acc.projection_archive, "load_pre",
                        lambda gw: (state["pre"], state["meta"]))
    monkeypatch.setattr(acc.projection_archive, "load_actuals",
                        lambda gw: (state["actual"], {}))
    monkeypatch.setattr(acc.projection_archive, "scoreable_gameweeks", lambda: [4])
    return state


class TestBinaryMetrics:
    def test_brier_is_mean_squared_error_of_the_probability(self):
        m = acc._binary_metrics(pd.Series([1.0, 0.0, 0.5, 0.5]),
                                pd.Series([1, 0, 1, 0]))
        assert m["brier"] == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)
        assert m["n"] == 4

    def test_bias_is_signed_so_too_high_is_distinguishable_from_noisy(self):
        high = acc._binary_metrics(pd.Series([0.6, 0.6]), pd.Series([0, 0]))
        noisy = acc._binary_metrics(pd.Series([1.0, 0.0]), pd.Series([0, 1]))
        assert high["bias"] > 0
        # Equally wrong on average, opposite in kind: one is correctable by
        # moving a constant, the other is not.
        assert noisy["bias"] == pytest.approx(0.0)

    def test_auc_reads_ordering_alone(self):
        perfect = acc.auc(pd.Series([0.9, 0.8, 0.2, 0.1]), pd.Series([1, 1, 0, 0]))
        backwards = acc.auc(pd.Series([0.1, 0.2, 0.8, 0.9]), pd.Series([1, 1, 0, 0]))
        assert perfect == pytest.approx(1.0)
        assert backwards == pytest.approx(0.0)

    def test_a_source_that_says_one_for_everybody_scores_half(self):
        """The GW3 failure's signature. Ties must not look like discrimination."""
        assert acc.auc(pd.Series([1.0] * 6),
                       pd.Series([1, 1, 0, 0, 0, 0])) == pytest.approx(0.5)

    def test_confident_misses_do_not_make_log_loss_infinite(self):
        m = acc._binary_metrics(pd.Series([1.0, 0.0]), pd.Series([0, 1]))
        assert np.isfinite(m["logloss"])

    def test_percentages_and_fractions_score_the_same(self):
        """FFP publishes 0-100 and the engine works in 0-1."""
        frac = acc._binary_metrics(pd.Series([0.9, 0.1]), pd.Series([1, 0]))
        pct = acc._binary_metrics(pd.Series([90.0, 10.0]), pd.Series([1, 0]))
        assert frac["brier"] == pytest.approx(pct["brier"])


class TestStartScope:
    def test_every_player_is_scored_not_just_the_starters(self, _start_archive):
        scored = acc.score_gameweek(4)
        row = scored[(scored["source"] == "blend")
                     & (scored["scope"] == acc.SCOPE_START)].iloc[0]
        # The whole question is who plays, so restricting to players who did
        # would be scoring the model against its own answer.
        assert row["n"] == 10

    def test_points_scopes_are_untouched_by_the_new_one(self, _start_archive):
        scored = acc.score_gameweek(4)
        starters = scored[(scored["source"] == "blend")
                          & (scored["scope"] == acc.SCOPE_STARTERS)].iloc[0]
        assert starters["n"] == 6          # the six who started
        assert pd.isna(starters["brier"])  # a points scope has no Brier

    def test_start_scope_carries_no_points_metrics(self, _start_archive):
        scored = acc.score_gameweek(4)
        row = scored[scored["scope"] == acc.SCOPE_START].iloc[0]
        assert pd.isna(row["mae"]) and pd.isna(row["spearman"])

    def test_summarise_keeps_each_metric_to_its_own_scope(self, _start_archive):
        summary = acc.summarise(acc.score_archive())
        start = summary[summary["scope"] == acc.SCOPE_START]
        points = summary[summary["scope"] == acc.SCOPE_STARTERS]
        assert start["brier"].notna().all() and start["mae"].isna().all()
        assert points["mae"].notna().all() and points["brier"].isna().all()

    def test_common_subset_does_not_narrow_the_start_scope_to_rotowire(
            self, _start_archive):
        """Rotowire prices its expected XI but the engine records a start
        decision for everybody. One mask over the points columns would throw
        most of the start evidence away."""
        scored = acc.score_gameweek(4, common_subset=True)
        start = scored[(scored["source"] == "blend")
                       & (scored["scope"] == acc.SCOPE_START)].iloc[0]
        starters = scored[(scored["source"] == "blend")
                          & (scored["scope"] == acc.SCOPE_STARTERS)].iloc[0]
        assert start["n"] == 10
        assert starters["n"] == 6   # only Rotowire's six are priced by everyone


class TestStartCalibration:
    def test_buckets_report_predicted_against_observed(self, _start_archive):
        calib = acc.start_calibration()
        assert {"bucket", "n", "predicted", "actual", "gap"} <= set(calib.columns)
        assert calib["n"].sum() == 10

    def test_it_surfaces_a_confident_claim_that_is_wrong(self, _start_archive):
        """The GW3 failure: 208 players rendered at exactly 100%, a quarter of
        whom started. No single row looks wrong; the bucket is unmissable."""
        _start_archive["pre"] = _pre_start(start_pct=[1.0] * 10)
        _start_archive["actual"] = _actual_start(started=[1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        calib = acc.start_calibration()
        top = calib[calib["predicted"] > 0.95].iloc[0]
        assert top["actual"] == pytest.approx(0.2)
        assert top["gap"] > 0.7

    def test_cohorts_split_by_position_and_rotowire(self, _start_archive):
        cohorts = acc.start_cohort_rates()
        assert set(cohorts["cohort"]) == {"listed", "omitted"}
        assert cohorts[cohorts["cohort"] == "listed"]["n"].sum() == 6
        assert cohorts[cohorts["cohort"] == "omitted"]["n"].sum() == 4

    def test_cohorts_exclude_backfills(self, _start_archive):
        """A snapshot taken after kickoff may have seen the team sheet."""
        _start_archive["meta"] = {"gameweek": 4, "captured_before_deadline": False}
        assert acc.start_cohort_rates().empty
        # ...but the headline calibration still shows it, labelled.
        assert not acc.start_calibration().empty

    def test_rotowire_cohort_uses_the_engines_own_priced_test(self, _start_archive):
        """`> 0`, not `notna` -- a zero projection is not a listing."""
        _start_archive["pre"] = _pre_start(
            proj_start__rotowire=[8.0, 7.0, 6.0, 5.0, 5.0, 0.0] + [np.nan] * 4)
        cohorts = acc.start_cohort_rates()
        assert cohorts[cohorts["cohort"] == "listed"]["n"].sum() == 5


class TestFittingTheStartConstants:
    def test_a_replay_that_does_not_reproduce_the_archive_is_refused(
            self, monkeypatch, _start_archive):
        """The load-bearing guard. Fitting constants against a replay that does
        not reproduce the app fits a different app -- and the archive really does
        span engine versions: GW3's snapshot predates both the omitted-start
        signal and FFP's start recovery, so replaying it today produces 0.02
        where it recorded 1.00."""
        monkeypatch.setattr(acc, "_replay_start_pct",
                            lambda job, floors=None, omitted=None:
                            pd.Series(0.0, index=job["index"]))
        fit = acc.fit_start_constants()
        assert fit["fitted_floors"] == fit["current_floors"]
        assert fit["fitted_omitted"] == fit["current_omitted"]
        assert "Refusing to fit" in fit["note"]

    def test_backfills_are_never_fitted_on(self, _start_archive):
        _start_archive["meta"] = {"gameweek": 4, "captured_before_deadline": False}
        assert acc.fit_start_constants()["gameweeks"] == []

    def test_rotowires_own_start_column_is_not_fed_back_in(self, _start_archive):
        """It is the engine's decision about a player, not anything Rotowire
        published. As an input the replay would reproduce itself perfectly and
        measure nothing."""
        jobs = acc._start_replay_inputs()
        assert jobs and "rotowire" not in jobs[0]["per_source_startpct"]

    def test_fidelity_is_judged_against_the_constants_that_built_the_snapshot(
            self, monkeypatch, _start_archive):
        """Otherwise the fitter is broken by its own last answer.

        Fidelity checked against *today's* constants fails on every archived
        gameweek the moment they are retuned -- the snapshot was written under
        the old ones -- so the fitter would refuse to fit for ever after the
        first time it was acted on. The snapshot records what built it, per
        player, in `start_pct__rotowire`.
        """
        import config
        monkeypatch.setattr(config, "ROTOWIRE_START_FLOORS",
                            {"G": 0.99, "D": 0.99, "M": 0.99, "F": 0.99})
        monkeypatch.setattr(config, "ROTOWIRE_OMITTED_START",
                            {"G": 0.30, "D": 0.30, "M": 0.30, "F": 0.30})
        fit = acc.fit_start_constants()
        assert fit["gameweeks"] == [4], fit["note"]

    def test_a_snapshot_that_cannot_prove_what_built_it_is_not_fitted_on(
            self, _start_archive):
        """GW3's predates the column, and predates two engine changes with it."""
        _start_archive["pre"] = _pre_start(self_consistent=True).drop(
            columns=["start_pct__rotowire"])
        fit = acc.fit_start_constants()
        assert fit["gameweeks"] == []
        assert "Refusing to fit" in fit["note"]

    def test_a_thin_cohort_is_never_retuned(self, _start_archive):
        fit = acc.fit_start_constants()
        cohorts = fit["cohorts"]
        assert not cohorts.empty
        # Ten players is nobody's evidence for anything.
        assert not cohorts["enough_rows"].any()
        assert not cohorts["moves"].any()
        assert fit["fitted_floors"] == fit["current_floors"]

    def test_the_implied_value_is_never_fitted_to_zero(self):
        """A zero says a player certainly will not start. Rotowire's silence is
        usually a benching and sometimes a name this app failed to match, and
        those are indistinguishable from here."""
        assert acc.MIN_IMPLIED_START > 0


class TestAgainstTheRealArchive:
    """The committed snapshots, not a fixture.

    Deterministic because `archive/projections/` is in the repo, and a real
    regression guard: if a future engine change or retune makes the app
    confidently wrong about who starts, this is what says so.
    """

    def _scoreable(self):
        from scripts.common import projection_archive
        return projection_archive.scoreable_gameweeks()

    def test_the_app_is_not_confidently_wrong_about_who_starts(self):
        from scripts.common.data_validation import check_start_calibration
        if not self._scoreable():
            pytest.skip("no gameweek has both a projection snapshot and actuals")
        errors = [i for i in check_start_calibration(
            acc.start_calibration(include_backfills=False),
            acc.start_cohort_rates()) if i.severity == "error"]
        assert not errors, "\n".join(str(i) for i in errors)

    def test_the_replay_reproduces_what_was_archived(self):
        """The fitter's own precondition, asserted rather than assumed.

        A snapshot that cannot say which constants built it will not replay --
        GW3's predates the column, and predates two engine changes with it. That
        is the refusal working. What must not happen is *every* gameweek failing,
        which would mean the archive and the engine have parted company and the
        constants are being fitted against nothing.
        """
        if not self._scoreable():
            pytest.skip("no gameweek has both a projection snapshot and actuals")
        jobs = acc._start_replay_inputs()
        if not jobs:
            pytest.skip("no pre-deadline snapshot to replay")
        fidelity = acc._replay_fidelity(jobs)
        assert fidelity["usable"], (
            "no archived gameweek replays to within %.2f: %s"
            % (acc.REPLAY_TOLERANCE, fidelity["per_gameweek"]))
