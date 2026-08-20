"""Plausibility tests against live upstream data.

These exist because the app's two worst bugs were invisible to a fully-mocked
test suite: Rotowire silently swapped in a 5-gameweek cumulative article, and the
Draft API's preseason all-zero score grid collapsed the win-probability model.
Both produced numbers no one could look at and call correct -- 18.6 projected
points for a goalkeeper, 85%/15% on a 1.1-point gap -- which is exactly what an
automated assertion is good at.

Nothing here asserts an exact value; upstream numbers change every week. They
assert that the numbers are *possible*.
"""

import pandas as pd
import pytest

from tests.live.conftest import skip_if_unreachable

from scripts.common.data_validation import (
    check_projected_team_total,
    check_score_std,
    check_team_strength,
    check_single_gw_projections,
    check_source_scale_agreement,
    check_win_probability,
    format_issues,
    raise_on_error,
)


class TestRotowireSource:
    def test_discovered_article_is_a_single_gameweek_article(self, rotowire_url, current_gw):
        """Discovery must not land on a multi-gameweek or season-long article.

        The GW1-5 "best picks" article is a legitimate fallback, but only when no
        single-gameweek article exists -- and the scraper then divides it down.
        """
        assert "top-400" not in rotowire_url, (
            "ROTOWIRE_URL resolved to the season-long rankings article: %s" % rotowire_url
        )

    def test_projections_are_single_gameweek_scale(self, rotowire_projections, rotowire_url):
        """The check that would have caught the original bug on its own."""
        issues = check_single_gw_projections(
            rotowire_projections, source="Rotowire (%s)" % rotowire_url
        )
        raise_on_error(issues, context="Rotowire weekly projections")

    def test_projection_table_has_the_columns_the_app_consumes(self, rotowire_projections):
        required = {"Player", "Team", "Position", "Points"}
        missing = required - set(rotowire_projections.columns)
        assert not missing, (
            "Rotowire parser lost column(s) %s -- the upstream table shape changed. "
            "Present: %s" % (sorted(missing), sorted(rotowire_projections.columns))
        )

    def test_positions_use_the_apps_scheme(self, rotowire_projections):
        unexpected = set(rotowire_projections["Position"].dropna().unique()) - {"G", "D", "M", "F"}
        assert not unexpected, "unexpected position codes from Rotowire: %s" % sorted(unexpected)


class TestFfpSource:
    def test_sheet_has_the_columns_the_app_consumes(self, ffp_projections):
        required = {"Name", "Team", "Position", "Predicted", "Start", "LongStart", "Next3GWs"}
        missing = required - set(ffp_projections.columns)
        assert not missing, (
            "FFP sheet lost column(s) %s -- the published sheet changed. "
            "Present: %s" % (sorted(missing), sorted(ffp_projections.columns))
        )

    def test_start_percentages_are_percentages(self, ffp_projections):
        start = pd.to_numeric(ffp_projections["Start"], errors="coerce").dropna()
        if start.empty:
            pytest.skip("FFP Start column is empty")
        assert start.between(0, 100).all(), (
            "FFP Start%% outside 0-100 (min %.1f, max %.1f) -- the app divides this "
            "by 100 to scale projections." % (start.min(), start.max())
        )

    def test_predicted_points_are_single_gameweek_scale(self, ffp_projections):
        """Skips while FFP has not published the gameweek (all zeros preseason)."""
        predicted = pd.to_numeric(ffp_projections["Predicted"], errors="coerce").dropna()
        if not (predicted > 0).any():
            pytest.skip("FFP has not published Predicted points for this gameweek yet")
        raise_on_error(
            check_single_gw_projections(
                pd.DataFrame({"Points": predicted[predicted > 0]}), source="FFP Predicted"
            ),
            context="FFP single-gameweek predictions",
        )


class TestCrossSourceAgreement:
    def test_rotowire_and_ffp_are_in_the_same_units(self, rotowire_projections, ffp_projections):
        """A systematic multiple between two independent sources means a unit
        mismatch, which is what a multi-gameweek total masquerading as one
        gameweek looks like from the outside."""
        ffp_predicted = pd.to_numeric(ffp_projections["Predicted"], errors="coerce").dropna()
        if not (ffp_predicted > 0).any():
            pytest.skip("FFP has not published Predicted points for this gameweek yet")
        raise_on_error(
            check_source_scale_agreement(
                rotowire_projections["Points"], ffp_predicted, "Rotowire", "FFP Predicted"
            ),
            context="Rotowire vs FFP",
        )


class TestDraftWinProbabilityInputs:
    def test_score_spread_is_usable(self, draft_league_id):
        """sigma == 0 is the preseason failure that produced 85%/15% on a
        1.1-point gap. The estimator must return the league prior instead."""
        from scripts.draft.fixture_projections import _estimate_score_std

        sigma, n_hist = _estimate_score_std(draft_league_id)
        raise_on_error(check_score_std(sigma, n_hist), context="Draft score spread")


class TestDraftFixtureProjections:
    """End-to-end over every fixture in the current gameweek.

    This is the test that fails instead of the user noticing something is off.
    """

    @pytest.fixture(scope="class")
    def analysed_fixtures(self, draft_league_id, current_gw, rotowire_projections, ffp_projections):
        from scripts.common.utils import get_gameweek_fixtures
        from scripts.draft.fixture_projections import analyze_fixture_projections

        fixtures = get_gameweek_fixtures(draft_league_id, current_gw)
        if not fixtures:
            pytest.skip("No Draft fixtures for GW %s" % current_gw)

        analysed = []
        for fixture in fixtures:
            result = analyze_fixture_projections(
                fixture, draft_league_id, rotowire_projections, ffp_df=ffp_projections
            )
            if result is not None:
                analysed.append((fixture, result))
        if not analysed:
            pytest.skip("No Draft fixture could be resolved to two rosters")
        return analysed

    @staticmethod
    def _total(team_df):
        col = "Proj_Blended" if "Proj_Blended" in team_df.columns else "Points"
        return float(team_df[col].sum())

    def test_every_team_total_is_plausible(self, analysed_fixtures):
        issues = []
        for fixture, result in analysed_fixtures:
            team1_df, team2_df, name1, name2 = result[0], result[1], result[2], result[3]
            for team_df, name in ((team1_df, name1), (team2_df, name2)):
                issues += check_projected_team_total(
                    self._total(team_df), len(team_df), label="%s (%s)" % (name, fixture)
                )
        raise_on_error(issues, context="Draft GW fixture projections")

    def test_every_win_probability_matches_its_scoreline(self, analysed_fixtures, draft_league_id):
        """The reported bug, asserted across every fixture rather than the one
        the user happened to look at."""
        from scripts.draft.fixture_projections import (
            _estimate_score_std,
            _normal_cdf,
            _winprob_denom,
        )

        sigma, _ = _estimate_score_std(draft_league_id)
        denom = _winprob_denom(sigma)

        issues = []
        for fixture, result in analysed_fixtures:
            score1, score2 = self._total(result[0]), self._total(result[1])
            prob1 = _normal_cdf((score1 - score2) / denom)
            issues += [
                i._replace(message="%s -- %s" % (fixture, i.message))
                for i in check_win_probability(prob1, score1, score2)
            ]
        raise_on_error(issues, context="Draft GW win probabilities")

    def test_every_lineup_is_a_legal_formation(self, analysed_fixtures):
        """A projected XI that isn't a legal FPL formation means the optimiser
        silently degraded -- which also makes the team total meaningless."""
        problems = []
        for fixture, result in analysed_fixtures:
            for team_df, name in ((result[0], result[2]), (result[1], result[3])):
                counts = team_df["Position"].value_counts().to_dict()
                n_gk, n_def = counts.get("G", 0), counts.get("D", 0)
                n_mid, n_fwd = counts.get("M", 0), counts.get("F", 0)
                if not (len(team_df) == 11 and n_gk == 1 and 3 <= n_def <= 5
                        and 2 <= n_mid <= 5 and 1 <= n_fwd <= 3):
                    problems.append(
                        "%s (%s): %d players, %dG/%dD/%dM/%dF"
                        % (name, fixture, len(team_df), n_gk, n_def, n_mid, n_fwd)
                    )
        assert not problems, "Illegal projected formation(s):\n  " + "\n  ".join(problems)

    def test_no_player_projects_an_impossible_single_gw_score(self, analysed_fixtures):
        """Catches the original symptom directly: a goalkeeper showing 18.6."""
        offenders = []
        for fixture, result in analysed_fixtures:
            for team_df in (result[0], result[1]):
                df = team_df.reset_index()
                col = "Proj_Blended" if "Proj_Blended" in df.columns else "Points"
                for _, row in df.iterrows():
                    if float(row[col]) > 20.0:
                        offenders.append(
                            "%s (%s, %s) projects %.1f in %s"
                            % (row.get("Player"), row.get("Team"), row.get("Position"),
                               row[col], fixture)
                        )
        assert not offenders, (
            "No player scores this much in a single gameweek:\n  " + "\n  ".join(offenders)
        )


class TestDraftTeamStrength:
    """Power rankings over the real league.

    The silent failure this guards against: analytics.py groups on position codes
    G/D/M/F while the FPL bootstrap supplies GKP/DEF/MID/FWD. Get that wrong and
    every percentile falls back to its 0.5 default, so every team scores exactly
    50.0 -- a table that looks entirely reasonable until you read it closely.
    """

    @pytest.fixture(scope="class")
    def league_strength(self, draft_league_id, current_gw):
        from scripts.common.team_strength import build_league_strength

        team_df, player_df = skip_if_unreachable(
            lambda: build_league_strength(draft_league_id, current_gw),
            "Draft league strength",
        )
        if team_df is None or team_df.empty:
            pytest.skip("league has no drafted rosters yet")
        return team_df, player_df

    def test_scores_are_plausible(self, league_strength):
        team_df, _ = league_strength
        raise_on_error(check_team_strength(team_df), context="Draft power rankings")

    def test_league_is_not_flat(self, league_strength):
        """Distinct rosters must produce distinct scores."""
        team_df, _ = league_strength
        assert team_df["Score"].nunique() > 1, (
            "every team scored identically -- the percentile join failed"
        )

    def test_positional_scores_differ_within_a_team(self, league_strength):
        """A team strong everywhere and weak nowhere means the split isn't working."""
        team_df, _ = league_strength
        spreads = (team_df[["GK", "DEF", "MID", "FWD"]].max(axis=1)
                   - team_df[["GK", "DEF", "MID", "FWD"]].min(axis=1))
        assert spreads.max() > 1.0, (
            "no team shows any positional variation -- positions are not being "
            "grouped separately"
        )

    def test_player_strengths_stay_in_range(self, league_strength):
        _, player_df = league_strength
        assert player_df["Player_Strength"].between(0.0, 1.0).all()
        assert player_df["Raw_Strength"].between(0.0, 1.0).all()
        assert (player_df["Player_Strength"] <= player_df["Raw_Strength"] + 1e-9).all(), (
            "the injury discount can only reduce a score"
        )
