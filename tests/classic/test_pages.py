"""Smoke tests for Classic pages.

Each test calls the page's show_*() function with all dependencies mocked,
verifying no exception is raised.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

# Import eagerly (before any test patches scripts.common.utils.* via
# mock_all_utils) so this module's `from scripts.common.utils import
# position_converter, ...` binds the real functions, not a mock that
# happens to be active during a lazy first import inside a `with patch(...)`
# block later. See TestInitialSquadOptimizerPage's second test, which
# exercises this module's internals directly (unmocked) after test_smoke.
import scripts.classic.initial_squad  # noqa: F401
from scripts.common.data_validation import check_initial_squad


class TestClassicHomePage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.home.get_league_standings", return_value=None), \
             patch("scripts.classic.home.get_classic_team_history", return_value=None), \
             patch("scripts.classic.home.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.home.get_current_gameweek", return_value=25), \
             patch("scripts.classic.home.get_all_h2h_league_matches", return_value=[]), \
             patch("scripts.classic.home.extract_classic_h2h_gw_scores", return_value=pd.DataFrame()), \
             patch("scripts.classic.home.calculate_all_play_standings", return_value=pd.DataFrame()), \
             patch("scripts.classic.home.render_luck_adjusted_table"), \
             patch("scripts.classic.home.render_standings_table"):
            from scripts.classic.home import show_classic_home_page
            show_classic_home_page()


class TestClassicFixtureProjectionsPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.fixture_projections.get_current_gameweek", return_value=25), \
             patch("scripts.classic.fixture_projections.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.fixture_projections.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.fixture_projections.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.fixture_projections.find_optimal_lineup", return_value=pd.DataFrame()), \
             patch("scripts.classic.fixture_projections.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.fixture_projections.get_league_standings", return_value=None), \
             patch("scripts.classic.fixture_projections.get_h2h_league_matches", return_value=[]), \
             patch("scripts.classic.fixture_projections.get_classic_h2h_record", return_value={"wins": 0, "draws": 0, "losses": 0, "record_str": "0-0-0", "matches": []}), \
             patch("scripts.classic.fixture_projections.get_classic_transfers", return_value=[]), \
             patch("scripts.classic.fixture_projections.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.fixture_projections.is_gameweek_live", return_value=False), \
             patch("scripts.classic.fixture_projections.get_live_gameweek_stats", return_value={}), \
             patch("scripts.classic.fixture_projections.get_fpl_player_mapping", return_value={}), \
             patch("scripts.classic.fixture_projections.get_gw_finished_teams", return_value=set()), \
             patch("scripts.classic.fixture_projections.simulate_auto_subs", return_value=(pd.DataFrame(), [])), \
             patch("scripts.classic.fixture_projections.show_api_error"), \
             patch("scripts.classic.fixture_projections.compute_key_differentials", return_value=([], [])), \
             patch("scripts.classic.fixture_projections.render_key_differentials"):
            from scripts.classic.fixture_projections import show_classic_fixture_projections_page
            show_classic_fixture_projections_page()


class TestClassicTransfersPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.transfers.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.transfers.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.transfers.get_classic_team_history", return_value=None), \
             patch("scripts.classic.transfers.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.transfers.get_current_gameweek", return_value=25), \
             patch("scripts.classic.transfers.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.transfers.get_classic_transfers", return_value=[]), \
             patch("scripts.classic.transfers.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.transfers.show_api_error"), \
             patch("scripts.classic.transfers.compute_healthy_form", return_value=5.0), \
             patch("scripts.classic.transfers.get_ffp_projections_data", return_value=None), \
             patch("scripts.classic.transfers.blend_multi_gw_projections", side_effect=lambda df, *a, **kw: df), \
             patch("scripts.classic.transfers.compute_positional_depth", return_value={}):
            from scripts.classic.transfers import show_classic_transfers_page
            show_classic_transfers_page()


class TestFreeHitPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.free_hit.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.free_hit.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.free_hit.get_current_gameweek", return_value=25), \
             patch("scripts.classic.free_hit.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.free_hit.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.free_hit.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.free_hit.show_api_error"):
            from scripts.classic.free_hit import show_free_hit_page
            show_free_hit_page()


class TestWildcardPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.wildcard.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.wildcard.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.wildcard.get_current_gameweek", return_value=25), \
             patch("scripts.classic.wildcard.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.wildcard.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.wildcard.get_fixture_difficulty_grid", return_value=pd.DataFrame()), \
             patch("scripts.classic.wildcard.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.wildcard.show_api_error"):
            from scripts.classic.wildcard import show_wildcard_page
            show_wildcard_page()


class TestInitialSquadOptimizerPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.initial_squad.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.initial_squad.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.initial_squad.get_current_gameweek", return_value=25), \
             patch("scripts.classic.initial_squad.get_fixture_difficulty_grid", return_value=pd.DataFrame()), \
             patch("scripts.classic.initial_squad.get_rotowire_season_rankings", return_value=pd.DataFrame()), \
             patch("scripts.classic.initial_squad.get_ffp_projections_data", return_value=None), \
             patch("scripts.classic.initial_squad.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.initial_squad.show_api_error"):
            from scripts.classic.initial_squad import show_initial_squad_optimizer_page
            show_initial_squad_optimizer_page()

    def test_scoring_pipeline_builds_valid_squad(self):
        """Not a UI smoke test — exercises the real scoring + ILP pipeline end
        to end with a small synthetic player pool to catch integration bugs
        the mocked-empty-data UI smoke test above can't reach."""
        import numpy as np
        from scripts.classic.initial_squad import (
            CAPTAIN_BONUS_WEIGHT,
            DEFAULT_BENCH_WEIGHT,
            _build_full_player_pool,
            _apply_eligibility_filters,
            _compute_scores,
        )
        from scripts.common.optimization import solve_squad_ilp

        rng = np.random.default_rng(0)
        # Real-looking 3-letter codes: _to_short_team_code() passes those through
        # untouched, where "T1" trips its "unknown team, guessing" warning path.
        _CODES = ["ARS", "AVL", "BHA", "BOU", "BRE", "CHE", "CRY", "EVE"]
        teams = [{"id": i, "short_name": _CODES[i - 1]} for i in range(1, 9)]
        elements = []
        pid = 1
        for team in teams:
            for pos, count in [(1, 3), (2, 6), (3, 6), (4, 4)]:
                for _ in range(count):
                    elements.append({
                        "id": pid,
                        "web_name": f"P{pid}",
                        "first_name": "First",
                        "second_name": f"Last{pid}",
                        "team": team["id"],
                        "element_type": pos,
                        "now_cost": int(rng.uniform(40, 140)),
                        "chance_of_playing_next_round": 100,
                        "news": "",
                        "total_points": 0,
                        "form": 0.0,
                    })
                    pid += 1
        bootstrap = {"elements": elements, "teams": teams}

        gw1_df = pd.DataFrame([
            {
                "Player": f"First Last{e['id']}", "Team": next(t["short_name"] for t in teams if t["id"] == e["team"]),
                "Position": {1: "G", 2: "D", 3: "M", 4: "F"}[e["element_type"]],
                "Points": round(rng.uniform(0, 8), 2),
            }
            for e in elements[:150]
        ])
        season_df = pd.DataFrame([
            {
                "Player": f"First Last{e['id']}", "Team": next(t["short_name"] for t in teams if t["id"] == e["team"]),
                "Position": {1: "G", 2: "D", 3: "M", 4: "F"}[e["element_type"]],
                "Points": round(rng.uniform(20, 220), 1),
            }
            for e in elements
        ])
        fdr_avg = pd.Series({t["short_name"]: rng.uniform(1.5, 4.5) for t in teams})

        full_pool = _build_full_player_pool(bootstrap)
        scored = _compute_scores(
            full_pool, gw1_df, season_df, None, fdr_avg,
            current_gw=1, w_season=0.70, w_opening=0.30,
        )
        candidate = _apply_eligibility_filters(scored, exclude_injured=True, min_chance_of_playing=75)

        squad_df, totals = solve_squad_ilp(
            candidate, 100.0, score_col="ExpPts", formation="auto",
            bench_weight=DEFAULT_BENCH_WEIGHT,
            captain_score_col="CapPts", captain_bonus_weight=CAPTAIN_BONUS_WEIGHT,
        )

        assert squad_df is not None
        assert len(squad_df) == 15
        assert squad_df["Is_Starter"].sum() == 11
        assert squad_df["Price"].sum() <= 100.0 + 1e-6
        pos_counts = squad_df["Position"].value_counts().to_dict()
        assert pos_counts == {"D": 5, "M": 5, "F": 3, "G": 2}
        assert (squad_df["Team"].value_counts() <= 3).all()
        assert squad_df["Is_Captain"].sum() == 1
        assert not check_initial_squad(squad_df, 100.0)

    def test_premium_player_is_worth_its_price(self):
        """The regression test for the defect that made this page unusable.

        The objective used to be a weighted sum of positional *percentiles*.
        Percentile is scale-free and saturates, so under a budget it cannot
        trade points against pounds -- and worse, it inverts across positions.
        A forward at the very top of a 32-deep pool scores 31/32 = 0.969, while
        a midfielder at the top of a 48-deep pool scores 47/48 = 0.979. The
        cheaper midfielder therefore *outranks* the premium forward no matter
        how many more points the forward actually projects. That is exactly how
        the live page ended up refusing to buy the best asset in the game and
        banking the change instead.

        This pool reproduces that inversion: the star forward projects twice the
        points of anyone else but scores *below* a 4.5m midfielder on percentile,
        and prices are tight enough that the budget genuinely binds. Under the
        old percentile objective the solver leaves the star on the shelf; an
        expected-points objective must buy him and captain him.
        """
        import numpy as np
        from scripts.classic.initial_squad import (
            CAPTAIN_BONUS_WEIGHT,
            DEFAULT_BENCH_WEIGHT,
            _apply_eligibility_filters,
            _build_full_player_pool,
            _compute_scores,
        )
        from scripts.common.optimization import solve_squad_ilp

        rng = np.random.default_rng(7)
        _CODES = ["ARS", "AVL", "BHA", "BOU", "BRE", "CHE", "CRY", "EVE"]
        teams = [{"id": i, "short_name": _CODES[i - 1]} for i in range(1, 9)]
        elements, pid = [], 1
        for team in teams:
            for pos, count in [(1, 3), (2, 6), (3, 6), (4, 4)]:
                for _ in range(count):
                    elements.append({
                        "id": pid, "web_name": f"P{pid}", "first_name": "First",
                        "second_name": f"Last{pid}", "team": team["id"],
                        "element_type": pos,
                        # 6.0-7.8m: a baseline 15 costs ~100m, so the budget
                        # actually binds and the premium has to be paid for.
                        "now_cost": int(rng.uniform(60, 78)),
                        "chance_of_playing_next_round": 100, "news": "",
                        "total_points": 0, "form": 0.0,
                    })
                    pid += 1

        star = next(e for e in elements if e["element_type"] == 4)
        star["now_cost"] = 145
        # Top of the deeper midfield pool, priced as bench fodder -- the player
        # a percentile objective prefers over the star.
        ringer = next(e for e in elements if e["element_type"] == 3)
        ringer["now_cost"] = 45
        star_name = f"First Last{star['id']}"

        def _rows(season):
            out = []
            for e in elements:
                if e["id"] == star["id"]:
                    pts = 400.0
                elif e["id"] == ringer["id"]:
                    pts = 200.0
                else:
                    pts = round(rng.uniform(150, 175), 1)
                out.append({
                    "Player": f"First Last{e['id']}",
                    "Team": next(t["short_name"] for t in teams if t["id"] == e["team"]),
                    "Position": {1: "G", 2: "D", 3: "M", 4: "F"}[e["element_type"]],
                    "Points": pts if season else round(pts / 38, 2),
                })
            return pd.DataFrame(out)

        fdr_avg = pd.Series({t["short_name"]: 3.0 for t in teams})
        scored = _compute_scores(
            _build_full_player_pool({"elements": elements, "teams": teams}),
            _rows(season=False), _rows(season=True), None, fdr_avg,
            current_gw=1, w_season=0.70, w_opening=0.30,
        )
        candidate = _apply_eligibility_filters(scored, exclude_injured=True, min_chance_of_playing=75)

        # The inversion this test exists to defeat must actually be present,
        # otherwise the assertions below would pass for the wrong reason.
        star_row = candidate[candidate["Player"] == star_name].iloc[0]
        ringer_row = candidate[candidate["Player"] == f"First Last{ringer['id']}"].iloc[0]
        assert star_row["Player Score"] < ringer_row["Player Score"]
        assert star_row["ExpPts"] > ringer_row["ExpPts"] * 1.5

        squad_df, _ = solve_squad_ilp(
            candidate, 100.0, score_col="ExpPts", formation="auto",
            bench_weight=DEFAULT_BENCH_WEIGHT,
            captain_score_col="CapPts", captain_bonus_weight=CAPTAIN_BONUS_WEIGHT,
        )

        assert squad_df is not None
        assert star_name in squad_df["Player"].tolist(), (
            "the best player in the pool was priced out of the squad -- the "
            "percentile-scale regression is back"
        )
        assert bool(squad_df.loc[squad_df["Player"] == star_name, "Is_Starter"].iloc[0])
        assert bool(squad_df.loc[squad_df["Player"] == star_name, "Is_Captain"].iloc[0])
        # A scale-free objective also shows up as money left in the bank.
        assert squad_df["Price"].sum() >= 95.0


class TestInitialSquadWeightSliders:
    """The two scoring weights are one split, bound to always total 100%.

    They used to be free sliders normalized after the fact, which silently
    rescaled the input -- 70/70 rendered as 70 and 70 but scored as 50/50, so
    the numbers on screen stopped meaning what they said.
    """

    def _state(self, season, opening):
        import streamlit as st
        from scripts.classic.initial_squad import _W_SEASON_KEY, _W_OPENING_KEY
        st.session_state = {_W_SEASON_KEY: season, _W_OPENING_KEY: opening}
        return st.session_state, _W_SEASON_KEY, _W_OPENING_KEY

    def test_defaults_are_the_documented_split(self):
        import streamlit as st
        from scripts.classic.initial_squad import (
            DEFAULT_W_SEASON, DEFAULT_W_OPENING, _W_SEASON_KEY, _W_OPENING_KEY,
            _init_weight_state,
        )
        st.session_state = {}
        _init_weight_state()
        assert st.session_state[_W_SEASON_KEY] == int(round(DEFAULT_W_SEASON * 100))
        assert st.session_state[_W_OPENING_KEY] == int(round(DEFAULT_W_OPENING * 100))

    def test_existing_values_are_not_clobbered(self):
        """Re-running the page must not reset a split the user chose."""
        import streamlit as st
        from scripts.classic.initial_squad import (
            _W_SEASON_KEY, _W_OPENING_KEY, _init_weight_state,
        )
        st.session_state = {_W_SEASON_KEY: 40, _W_OPENING_KEY: 60}
        _init_weight_state()
        assert st.session_state[_W_SEASON_KEY] == 40

    @pytest.mark.parametrize("week1", [0, 30, 60, 100])
    def test_moving_fast_start_drives_season_down(self, week1):
        from scripts.classic.initial_squad import _sync_weight_from_opening
        state, season_key, opening_key = self._state(70, week1)
        _sync_weight_from_opening()
        assert state[opening_key] == week1
        assert state[season_key] == 100 - week1
        assert state[season_key] + state[opening_key] == 100

    @pytest.mark.parametrize("season", [0, 55, 85, 100])
    def test_moving_season_drives_fast_start_down(self, season):
        from scripts.classic.initial_squad import _sync_weight_from_season
        state, season_key, opening_key = self._state(season, 30)
        _sync_weight_from_season()
        assert state[season_key] == season
        assert state[opening_key] == 100 - season
        assert state[season_key] + state[opening_key] == 100


class TestPositionalColorRatios:
    """Grading a squad's expected points within position, for the colour scale.

    A goalkeeper projecting 4.3 is the best in the game; a midfielder
    projecting 4.3 is ordinary. One shared scale cannot say both, so it says
    the wrong thing about whichever position projects lower.
    """

    POOL = pd.DataFrame({
        "Position": ["G"] * 3 + ["M"] * 3,
        "ExpPts": [0.6, 2.5, 4.3, 0.3, 3.2, 6.1],
    })

    def _ratios(self, rows):
        from scripts.classic.initial_squad import _positional_color_ratios
        return _positional_color_ratios(rows, self.POOL)

    def test_best_in_each_position_both_grade_top(self):
        rows = pd.DataFrame({"Position": ["G", "M"], "ExpPts": [4.3, 6.1]})
        assert self._ratios(rows) == [1.0, 1.0]

    def test_same_raw_value_grades_differently_by_position(self):
        """4.3 is the best keeper in the pool but a mid-tier midfielder."""
        rows = pd.DataFrame({"Position": ["G", "M"], "ExpPts": [4.3, 4.3]})
        gk, mid = self._ratios(rows)
        assert gk == 1.0
        assert mid < gk

    def test_worst_in_position_grades_bottom(self):
        rows = pd.DataFrame({"Position": ["G"], "ExpPts": [0.6]})
        assert self._ratios(rows) == [0.0]

    def test_values_outside_the_pool_range_are_clamped(self):
        rows = pd.DataFrame({"Position": ["G", "G"], "ExpPts": [99.0, -5.0]})
        assert self._ratios(rows) == [1.0, 0.0]

    def test_unknown_position_yields_nan_not_a_wrong_colour(self):
        rows = pd.DataFrame({"Position": ["F"], "ExpPts": [4.0]})
        assert pd.isna(self._ratios(rows)[0])

    @pytest.mark.parametrize("rows", [
        pd.DataFrame(),
        pd.DataFrame({"ExpPts": [4.0]}),           # no Position column
        pd.DataFrame({"Position": ["G"]}),          # no value column
    ])
    def test_unusable_input_returns_none_to_trigger_fallback(self, rows):
        assert self._ratios(rows) is None

    def test_flat_pool_returns_none(self):
        """No spread means no meaningful grading; fall back to the default."""
        from scripts.classic.initial_squad import _positional_color_ratios
        flat = pd.DataFrame({"Position": ["G", "G"], "ExpPts": [3.0, 3.0]})
        assert _positional_color_ratios(flat, flat) is None


class TestClassicTeamAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.team_analysis.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.team_analysis.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.team_analysis.get_classic_team_history", return_value=None), \
             patch("scripts.classic.team_analysis.get_classic_team_position_data", return_value={}), \
             patch("scripts.classic.team_analysis.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.team_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.classic.team_analysis.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.team_analysis.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.team_analysis.render_season_highlights"), \
             patch("scripts.classic.team_analysis.compute_classic_bench_data", return_value=None), \
             patch("scripts.classic.team_analysis.render_bench_analysis"):
            from scripts.classic.team_analysis import show_classic_team_analysis_page
            show_classic_team_analysis_page()

    def test_smoke_no_current_season_gws_but_has_past_seasons(self, mock_all_utils):
        """Regression: a team with no gameweek data yet for the current season
        (e.g. joined right before a new season, or preseason) must still show
        Season History from `past` — it shouldn't go completely blank just
        because `current` is empty."""
        history = {
            "current": [],
            "past": [
                {"season_name": "2025/26", "total_points": 2187, "rank": 1037838, "rank_percentage": "8"},
                {"season_name": "2024/25", "total_points": 2125, "rank": 3932974, "rank_percentage": "34"},
            ],
            "chips": [],
        }
        with patch("scripts.classic.team_analysis.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.team_analysis.get_classic_team_picks", return_value=None), \
             patch("scripts.classic.team_analysis.get_classic_team_history", return_value=history), \
             patch("scripts.classic.team_analysis.get_classic_team_position_data", return_value={}), \
             patch("scripts.classic.team_analysis.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.team_analysis.get_current_gameweek", return_value=1), \
             patch("scripts.classic.team_analysis.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.team_analysis.position_converter", side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.team_analysis.render_season_highlights"), \
             patch("scripts.classic.team_analysis.compute_classic_bench_data", return_value=None), \
             patch("scripts.classic.team_analysis.render_bench_analysis"), \
             patch("scripts.classic.team_analysis.render_styled_table") as mock_render_table, \
             patch("scripts.classic.team_analysis.config.get_classic_season_notes", return_value={}), \
             patch("scripts.classic.team_analysis.config.get_classic_league_history_records", return_value=[]):
            from scripts.classic.team_analysis import show_classic_team_analysis_page
            show_classic_team_analysis_page()
        # Season History table must have rendered despite current==[].
        assert mock_render_table.called
        rendered_df = mock_render_table.call_args[0][0]
        assert list(rendered_df["Season"]) == ["2025/26", "2024/25"]
        assert rendered_df.iloc[0]["% Finish"] == "8%"


class TestClassicLeagueAnalysisPage:
    def test_smoke(self, mock_all_utils):
        with patch("scripts.classic.league_analysis.get_league_standings", return_value=None), \
             patch("scripts.classic.league_analysis.get_classic_team_history", return_value=None), \
             patch("scripts.classic.league_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.classic.league_analysis.get_classic_bootstrap_static", return_value={"elements": [], "teams": [], "events": []}), \
             patch("scripts.classic.league_analysis.get_classic_team_position_data", return_value={}), \
             patch("scripts.classic.league_analysis.show_api_error"), \
             patch("scripts.classic.league_analysis.compute_classic_league_bench_data", return_value=[]), \
             patch("scripts.classic.league_analysis.render_league_bench_analysis"):
            from scripts.classic.league_analysis import show_classic_league_analysis_page
            show_classic_league_analysis_page()


class TestClassicTeamAnalysisSeasonHistory:
    """Season History table extension (% Finish, League Placements) needs a
    populated `history` (current + past) and picks data to actually reach
    that section — the base smoke test above early-returns before it."""

    def test_smoke_with_season_history_and_placements(self, mock_all_utils):
        history = {
            "current": [{"event": 1, "points": 60, "total_points": 60, "rank": 100, "overall_rank": 100}],
            "past": [
                {"season_name": "2025/26", "total_points": 2187, "rank": 1037838},
                {"season_name": "2024/25", "total_points": 2125, "rank": 3932974},
            ],
            "chips": [],
        }
        picks_data = {
            "picks": [{"element": 1, "position": 1, "is_captain": True, "is_vice_captain": False, "multiplier": 2}],
            "active_chip": None,
            "entry_history": {"points": 60, "rank": 100, "value": 1000, "bank": 5},
        }
        bootstrap = {
            "elements": [{
                "id": 1, "web_name": "Salah", "first_name": "Mohamed", "second_name": "Salah",
                "team": 1, "element_type": 3,
            }],
            "teams": [{"id": 1, "short_name": "LIV"}],
            "events": [],
        }
        with patch("scripts.classic.team_analysis.get_classic_bootstrap_static", return_value=bootstrap), \
             patch("scripts.classic.team_analysis.get_classic_team_picks", return_value=picks_data), \
             patch("scripts.classic.team_analysis.get_classic_team_history", return_value=history), \
             patch("scripts.classic.team_analysis.get_classic_team_position_data", return_value={}), \
             patch("scripts.classic.team_analysis.get_entry_details", return_value={"name": "Test", "id": 1}), \
             patch("scripts.classic.team_analysis.get_current_gameweek", return_value=25), \
             patch("scripts.classic.team_analysis.get_rotowire_player_projections", return_value=pd.DataFrame()), \
             patch("scripts.classic.team_analysis.position_converter",
                   side_effect=lambda x: {1: "G", 2: "D", 3: "M", 4: "F"}.get(x, "M")), \
             patch("scripts.classic.team_analysis.render_season_highlights"), \
             patch("scripts.classic.team_analysis.compute_classic_bench_data", return_value=None), \
             patch("scripts.classic.team_analysis.render_bench_analysis"), \
             patch("scripts.classic.team_analysis.config.get_classic_season_notes",
                   return_value={"2025/26": {"pct_finish": 8.0}}), \
             patch("scripts.classic.team_analysis.config.get_classic_league_history_records", return_value=[
                 {"season": "2025/26", "league_id": 1161877, "league_name": "Super League DMV Starboys",
                  "manual_stats": {"rank": 4, "total_points": None}},
                 {"season": "2025/26", "league_id": 1555691, "league_name": "FAFO FPL",
                  "manual_stats": {"rank": 1, "total_points": None}},
             ]):
            from scripts.classic.team_analysis import show_classic_team_analysis_page
            show_classic_team_analysis_page()
