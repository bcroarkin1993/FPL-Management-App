"""Season Monte Carlo.

Numbers in these fixtures come from league 11347. The points system is verified
against the live payload rather than assumed: `3 * matches_won + matches_drawn`
reproduced `total` for all ten teams.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.common import season_sim as ss


def _standings(overrides=None):
    """Four teams, three matches each, clearly separated on scoring."""
    base = [
        {"league_entry": 1, "total": 9, "points_for": 180,
         "matches_won": 3, "matches_drawn": 0, "matches_lost": 0},
        {"league_entry": 2, "total": 6, "points_for": 165,
         "matches_won": 2, "matches_drawn": 0, "matches_lost": 1},
        {"league_entry": 3, "total": 3, "points_for": 150,
         "matches_won": 1, "matches_drawn": 0, "matches_lost": 2},
        {"league_entry": 4, "total": 0, "points_for": 120,
         "matches_won": 0, "matches_drawn": 0, "matches_lost": 3},
    ]
    for row in base:
        row.update((overrides or {}).get(row["league_entry"], {}))
    return base


def _matches(n_future=2):
    out = [{"event": 1, "league_entry_1": 1, "league_entry_2": 2,
            "league_entry_1_points": 60, "league_entry_2_points": 55,
            "finished": True, "started": True}]
    for gw in range(4, 4 + n_future):
        out.append({"event": gw, "league_entry_1": 1, "league_entry_2": 3,
                    "league_entry_1_points": 0, "league_entry_2_points": 0,
                    "finished": False, "started": False})
        out.append({"event": gw, "league_entry_1": 2, "league_entry_2": 4,
                    "league_entry_1_points": 0, "league_entry_2_points": 0,
                    "finished": False, "started": False})
    return out


class TestScoringRates:
    def test_it_divides_by_matches_actually_contested(self):
        """**Not** by `matches_played`, which the live payload reports as 38 for
        every team -- the scheduled total, not the contested one. Dividing by it
        understates every mean by a factor of four."""
        rates = ss.team_scoring_rates(_standings())
        assert rates[1]["mean"] == pytest.approx(60.0)   # 180 / 3
        assert rates[1]["n"] == 3

    def test_a_team_with_no_matches_has_no_mean(self):
        s = _standings({4: {"matches_won": 0, "matches_drawn": 0,
                              "matches_lost": 0, "points_for": 0}})
        assert np.isnan(ss.team_scoring_rates(s)[4]["mean"])


class TestRemainingMatches:
    def test_it_keeps_only_unfinished_fixtures(self):
        rem = ss.remaining_matches(_matches(2))
        assert len(rem) == 4
        assert all(r["event"] >= 4 for r in rem)

    def test_an_unfinished_past_gameweek_is_excluded(self):
        """FPL leaves `finished` False for hours after full time while bonus
        settles. Replaying one would double-count it against standings that
        already include the result."""
        m = _matches(1) + [{"event": 2, "league_entry_1": 3, "league_entry_2": 4,
                            "finished": False, "started": True}]
        assert len(ss.remaining_matches(m, from_gw=4)) == 2

    def test_a_fixture_missing_a_side_is_dropped(self):
        m = [{"event": 5, "league_entry_1": 1, "league_entry_2": None,
              "finished": False}]
        assert ss.remaining_matches(m) == []


class TestSimulateSeason:
    def _run(self, **kw):
        kw.setdefault("n_sims", 400)
        kw.setdefault("seed", 7)
        kw.setdefault("from_gw", 4)
        return ss.simulate_season(_standings(), _matches(3), **kw)

    def test_the_probabilities_are_a_distribution(self):
        sim = self._run()
        assert sim["title"].sum() == pytest.approx(1.0, abs=1e-9)
        assert sim["bottom"].sum() == pytest.approx(1.0, abs=1e-9)

    def test_top_n_sums_to_n(self):
        sim = self._run(playoff_places=2)
        assert sim["top_n"].sum() == pytest.approx(2.0, abs=1e-9)

    def test_the_leader_is_favoured(self):
        sim = self._run().set_index("league_entry")
        assert sim.loc[1, "title"] > sim.loc[4, "title"]
        assert sim.loc[1, "mean_rank"] < sim.loc[4, "mean_rank"]

    def test_it_is_reproducible(self):
        """Streamlit reruns the page on every widget interaction, and an
        unseeded simulation returns different odds each time -- which reads as a
        broken page rather than as sampling error."""
        a, b = self._run(), self._run()
        pd.testing.assert_frame_equal(a, b)

    def test_the_same_league_and_gameweek_seed_themselves_identically(self):
        a = ss.simulate_season(_standings(), _matches(3), n_sims=300,
                               from_gw=4, league_id=11347, gameweek=10)
        b = ss.simulate_season(_standings(), _matches(3), n_sims=300,
                               from_gw=4, league_id=11347, gameweek=10)
        pd.testing.assert_frame_equal(a, b)

    def test_a_different_gameweek_reseeds(self):
        a = ss.simulate_season(_standings(), _matches(3), n_sims=300,
                               from_gw=4, league_id=11347, gameweek=10)
        b = ss.simulate_season(_standings(), _matches(3), n_sims=300,
                               from_gw=4, league_id=11347, gameweek=11)
        assert not a["title"].equals(b["title"])

    def test_estimation_uncertainty_widens_the_field(self):
        """The methodological point. A mean estimated from three matches is not
        a known quantity, and holding it fixed reports the league as more decided
        than the evidence supports -- live, 31.4% against 25.4% for the leader."""
        loose = self._run(propagate_estimation_error=True).set_index("league_entry")
        tight = self._run(propagate_estimation_error=False).set_index("league_entry")
        assert loose.loc[1, "title"] < tight.loc[1, "title"]

    def test_no_remaining_fixtures_returns_the_table_as_it_stands(self):
        sim = ss.simulate_season(_standings(), _matches(0), n_sims=50, seed=1,
                                 from_gw=4).set_index("league_entry")
        assert sim.loc[1, "title"] == 1.0
        assert sim.loc[4, "bottom"] == 1.0
        assert sim.loc[1, "mean_points"] == pytest.approx(9.0)

    def test_ties_break_on_points_for(self):
        """How the published standings are ordered."""
        s = _standings({2: {"total": 9, "points_for": 179}})
        sim = ss.simulate_season(s, _matches(0), n_sims=20, seed=3,
                                 from_gw=4).set_index("league_entry")
        assert sim.loc[1, "title"] == 1.0      # 180 beats 179 on equal points

    def test_a_thin_sample_falls_back_to_the_league_mean(self):
        """One or two results say almost nothing about a manager, and letting
        them set a mean makes the simulation confident about noise."""
        s = _standings({4: {"matches_won": 1, "matches_drawn": 0,
                              "matches_lost": 0, "points_for": 200}})
        sim = ss.simulate_season(s, _matches(2), n_sims=100, seed=5,
                                 from_gw=4).set_index("league_entry")
        assert sim.loc[4, "mean_score"] < 200, "a single 200 must not set his mean"

    def test_empty_input_is_an_empty_frame_not_a_crash(self):
        assert ss.simulate_season([], []).empty

    def test_a_degenerate_sigma_falls_back_to_the_prior(self):
        """A sigma of 0 makes a one-point edge a certainty -- the failure
        `check_score_std` exists for."""
        sim = self._run(sigma=0.0).set_index("league_entry")
        assert 0.0 < sim.loc[1, "title"] < 1.0


class TestSummarise:
    def test_it_names_teams_and_renders_percentages(self):
        sim = ss.simulate_season(_standings(), _matches(2), n_sims=100, seed=2,
                                 from_gw=4)
        out = ss.summarise(sim, {1: "Alpha", 2: "Beta"}, playoff_places=2)
        assert list(out.columns)[:4] == ["Team", "Title", "Top 2", "Last"]
        assert "Alpha" in out["Team"].tolist()
        assert "Entry 3" in out["Team"].tolist()   # unnamed entries still show

    def test_an_empty_simulation_summarises_to_empty(self):
        assert ss.summarise(pd.DataFrame(), {}).empty


def test_the_module_is_streamlit_free():
    import pathlib
    assert "import streamlit" not in pathlib.Path(ss.__file__).read_text()
