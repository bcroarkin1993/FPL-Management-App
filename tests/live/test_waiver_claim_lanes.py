"""Live plausibility for the rent/hold waiver lanes and the ROS terms they rank on.

Contract (see tests/live/conftest.py): **unreachable -> SKIP, reachable but
implausible -> FAIL.**

The two defects these guard were both silent, and both were a scoring term that
had quietly stopped being a percentile:

* `start consistency` fell through to the raw cumulative `starts` count, because
  FFP stopped publishing `LongStart` -- non-null for 0 of 670 reference rows.
* `fixture ease` had no reference column at all (`AvgFDRNextN` was written onto
  the scored frames and not the pool), so it min-maxed each frame against itself.

Neither is visible in a single value. The test that would have caught them is the
reference pool carrying the columns the terms rank against.
"""

import numpy as np
import pandas as pd
import pytest

import config
from scripts.common import claim_value as cv
from scripts.common.data_validation import check_claim_lanes, check_ros_terms, raise_on_error
from tests.live.conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def pool():
    """The live blended player pool, which is what every ROS term ranks against."""
    from scripts.common.player_pool import blended_pool
    df = skip_if_unreachable(
        lambda: blended_pool(config.CURRENT_GAMEWEEK, config.ROTOWIRE_URL),
        "blended player pool")
    if df is None or df.empty:
        pytest.skip("blended pool empty")
    return df


class TestTheReferencePoolCarriesWhatTheTermsRankAgainst:
    def test_start_pct_is_published_for_the_whole_pool(self):
        """`Start_Pct` is the start term's top rung and the reference side of its
        percentile. If it ever stops being populated the term silently drops to a
        `starts` rate, which is weaker but still comparable -- so this is the
        early warning rather than the failure."""
        from scripts.common.player_pool import blended_pool
        df = skip_if_unreachable(
            lambda: blended_pool(config.CURRENT_GAMEWEEK, config.ROTOWIRE_URL),
            "blended player pool")
        vals = pd.to_numeric(df["Start_Pct"], errors="coerce")
        assert vals.notna().mean() > 0.95, (
            "Start_Pct is populated for only %.0f%% of the pool" % (100 * vals.notna().mean()))
        assert vals.max() <= 1.0 and vals.min() >= 0.0

    def test_ffp_longstart_is_still_absent(self, pool):
        """Informational. FFP stopped publishing `LongStart` and the ladder keeps
        it as a rung so the term returns for free if they resume -- so this must
        **not** fail on good news."""
        if "FFP_LongStart" not in pool.columns:
            return
        n = int(pd.to_numeric(pool["FFP_LongStart"], errors="coerce").notna().sum())
        if n:
            print("\nFFP has resumed publishing LongStart (%d rows) -- the start "
                  "term will now prefer Start_Pct over it; revisit the rung order." % n)


class TestTheStartTermIsOnOneScale:
    def test_every_rung_is_a_rate(self, pool):
        """A raw count can never satisfy this, which is the defect's tripwire."""
        import scripts.common.analytics as an
        for rung in an.START_CONSISTENCY_RUNGS:
            vals = an._start_rate_on_rung(pool, rung, config.CURRENT_GAMEWEEK)
            if vals is None:
                continue
            v = pd.to_numeric(vals, errors="coerce").dropna()
            assert v.min() >= 0.0 and v.max() <= 1.0, (
                "rung %r is outside [0,1] (%.3f..%.3f)" % (rung, v.min(), v.max()))

    def test_the_starts_rung_is_a_rate_not_a_count(self, pool):
        """9 starts in 9 gameweeks must read 1.0, not 9."""
        import scripts.common.analytics as an
        if "starts" not in pool.columns:
            pytest.skip("no starts column")
        vals = an._start_rate_on_rung(pool, "starts", config.CURRENT_GAMEWEEK)
        raw = pd.to_numeric(pool["starts"], errors="coerce")
        assert raw.max() > 1.0, "nobody has started a game; cannot tell a rate from a count"
        assert pd.to_numeric(vals, errors="coerce").max() <= 1.0


class TestTheWaiverFloorIsReal:
    @pytest.fixture(scope="class")
    @classmethod
    def live(cls, pool):
        from scripts.common.fpl_draft_api import get_league_waiver_order
        import scripts.draft.waiver_wire as ww
        lid, me = config.FPL_DRAFT_LEAGUE_ID, config.FPL_DRAFT_TEAM_ID
        if not lid or not me:
            pytest.skip("no Draft league configured")
        order = skip_if_unreachable(lambda: get_league_waiver_order(lid), "waiver order")
        fc = ww._build_floor_context(order, int(me), lid)
        if not fc or not fc.get("claim_rates"):
            pytest.skip("league has no measurable claim history")
        return fc, pool

    def test_a_floor_is_never_drawn_from_a_thin_pool(self, live):
        """Live there was exactly one available goalkeeper, so a GK 'floor' would
        have been that player's own projection presented as a market rate."""
        fc, pool = live
        floors = cv.waiver_floors(pool, fc.get("n_ahead"), fc.get("claim_rates"))
        counts = pool["Position"].value_counts().to_dict()
        raise_on_error(check_claim_lanes([], floors=floors,
                                         available_by_position=counts,
                                         min_floor_pool=cv.MIN_FLOOR_POOL))

    def test_a_floor_is_a_plausible_per_gameweek_rate(self, live):
        """Bounded above by the best forward rate in the pool and below by zero.
        A floor above the pool maximum would mean the index walked off the end;
        one far above any real projection means the horizon basis was crossed --
        `Proj_Next3 / 3` instead of `(Proj_Next3 - Proj) / 2`."""
        fc, pool = live
        floors = cv.waiver_floors(pool, fc.get("n_ahead"), fc.get("claim_rates"))
        if not floors:
            pytest.skip("no measurable floors")
        for pos, floor in floors.items():
            best = cv.forward_rate(pool[pool["Position"] == pos]).max()
            assert 0.0 <= floor <= best + 1e-9, (
                "%s floor %.2f is outside [0, %.2f]" % (pos, floor, best))
            assert floor < 15.0, (
                "%s floor %.2f is not a per-gameweek rate" % (pos, floor))

    def test_release_cost_is_never_negative(self, live):
        """`max(0, rate - floor)` by construction, so a negative one means the
        rate and the floor were computed on different bases."""
        fc, pool = live
        floors = cv.waiver_floors(pool, fc.get("n_ahead"), fc.get("claim_rates"))
        rc = pd.to_numeric(cv.release_cost(pool, floors), errors="coerce").dropna()
        if rc.empty:
            pytest.skip("no measurable release costs")
        assert rc.min() >= 0.0


class TestTheRosTermsCarryInformation:
    def test_no_weighted_term_is_flat(self, pool):
        """The `check_team_strength` 'every team scores exactly 50' signature,
        one model over: a weighted term with no spread ranks nobody, and that is
        what a reference pool missing its column looks like."""
        import scripts.common.analytics as an
        import scripts.draft.waiver_wire as ww
        from scripts.common.fpl_draft_api import pull_fpl_player_stats
        from scripts.common.player_matching import normalize_fpl_players_to_rotowire_schema

        teams = ww._bootstrap_teams_df()
        ref = skip_if_unreachable(
            lambda: ww._enforce_rw_schema_fpl(
                normalize_fpl_players_to_rotowire_schema(pull_fpl_player_stats(),
                                                         teams_df=teams), teams),
            "FPL player stats")
        ref["AvgFDRNextN"] = pd.to_numeric(
            ref["Team"].astype(str).map(
                ww._avg_fdr_by_short_name(config.CURRENT_GAMEWEEK,
                                          config.UPCOMING_WEEKS_DEFAULT)),
            errors="coerce")
        assert ref["AvgFDRNextN"].notna().mean() > 0.9, (
            "the fixture term's reference column is unpopulated, which is the "
            "defect: the term then min-maxes each frame against itself")

        scored = an.compute_player_scores(pool.head(200).copy(), ref,
                                          config.CURRENT_GAMEWEEK,
                                          format_context="draft")
        raise_on_error(check_ros_terms({"ROS": scored["ROS"]},
                                       ros=scored["ROS"], weights={"ROS": 1.0}))
        spread = float(scored["ROS"].max() - scored["ROS"].min())
        assert spread > 0.05, "ROS is flat across 200 players (spread %.4f)" % spread
