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
            _build_full_player_pool,
            _apply_eligibility_filters,
            _compute_scores,
        )
        from scripts.common.optimization import solve_squad_ilp

        rng = np.random.default_rng(0)
        teams = [{"id": i, "short_name": f"T{i}"} for i in range(1, 9)]
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
            current_gw=1, w_season=0.55, w_week1=0.30, w_fixture=0.15,
        )
        candidate = _apply_eligibility_filters(scored, exclude_injured=True, min_chance_of_playing=75)

        squad_df, totals = solve_squad_ilp(
            candidate, 100.0, score_col="Player Score", formation="auto", bench_weight=0.2,
            captain_score_col="Captain Score", captain_bonus_weight=0.5,
        )

        assert squad_df is not None
        assert len(squad_df) == 15
        assert squad_df["Is_Starter"].sum() == 11
        assert squad_df["Price"].sum() <= 100.0 + 1e-6
        pos_counts = squad_df["Position"].value_counts().to_dict()
        assert pos_counts == {"D": 5, "M": 5, "F": 3, "G": 2}
        assert (squad_df["Team"].value_counts() <= 3).all()
        assert squad_df["Is_Captain"].sum() == 1


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
