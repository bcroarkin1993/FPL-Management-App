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

from scripts.common.analytics import merge_season_projections
from scripts.common.data_validation import (
    check_element_states,
    check_merge_match_rate,
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
        gameweek looks like from the outside.

        Compare against FFP's *StartingPredicted*, not `Predicted`. Rotowire
        projects a player's points if he starts; `Predicted` is already
        multiplied by start probability, so comparing the two conflates a
        difference of basis with a difference of units.
        """
        col = "StartingPredicted" if "StartingPredicted" in ffp_projections.columns else "Predicted"
        ffp_predicted = pd.to_numeric(ffp_projections[col], errors="coerce").dropna()
        if not (ffp_predicted > 0).any():
            pytest.skip("FFP has not published %s for this gameweek yet" % col)
        raise_on_error(
            check_source_scale_agreement(
                rotowire_projections["Points"], ffp_predicted, "Rotowire", "FFP %s" % col
            ),
            context="Rotowire vs FFP",
        )

    def test_ffp_predicted_is_still_the_start_weighted_column(self, ffp_projections):
        """Pin down the relationship the 1GW blend depends on.

        FFP publishes both bases: `Predicted` == `StartingPredicted` * `Start`/100.
        The blend uses the conditional column and applies start likelihood once
        itself. If FFP ever redefines either column, blending would silently
        charge the start probability twice (or stop charging it at all), which
        moved the FFP term ~44% at a 60% median start rate -- a shift no mocked
        test would notice.
        """
        needed = {"Predicted", "StartingPredicted", "Start"}
        if not needed.issubset(ffp_projections.columns):
            pytest.skip("FFP sheet is missing %s" % ", ".join(sorted(needed - set(ffp_projections.columns))))

        df = ffp_projections[list(needed)].apply(pd.to_numeric, errors="coerce").dropna()
        df = df[(df["Start"] > 0) & (df["StartingPredicted"] > 0)]
        if len(df) < 25:
            pytest.skip("FFP has not published predictions for this gameweek yet")

        implied = df["StartingPredicted"] * df["Start"] / 100.0
        correlation = float(df["Predicted"].corr(implied))
        assert correlation > 0.95, (
            "FFP's Predicted no longer equals StartingPredicted x Start%% "
            "(correlation %.3f). The 1GW blend assumes StartingPredicted is "
            "conditional on starting -- re-check compute_player_scores()."
            % correlation
        )
        assert (df["Predicted"] <= df["StartingPredicted"] + 0.5).mean() > 0.95, (
            "Predicted should not exceed StartingPredicted -- it is the same "
            "projection discounted by start probability."
        )


class TestFfpMerge:
    """The FFP merge against the real player pool.

    FFP's Predicted/Start/LongStart feed the 1GW score and Next3GWs is 40% of
    ROS, so a match that lands on the wrong player is a scoring error nobody can
    see. On 2026-09-03 the merge ended in two team-agnostic, position-free tiers
    and 53 of 652 pool rows resolved on a shared surname alone -- Kalvin Phillips
    (MID) was carrying Dillon Phillips' goalkeeper start rate.
    """

    @pytest.fixture(scope="class")
    def claims(self, classic_player_pool, ffp_projections):
        from scripts.common.analytics import _claim_reference_rows

        if ffp_projections.empty:
            pytest.skip("FFP sheet returned no rows")
        return _claim_reference_rows(
            classic_player_pool, ffp_projections,
            name_col="Player", ref_name_col="Name", ref_team_col="Team",
            allow_cross_team_exact=True, source_name="FFP live check",
        )

    def test_no_match_crosses_position(self, claims, classic_player_pool, ffp_projections):
        from scripts.common.analytics import _position_letters

        pool_pos = _position_letters(classic_player_pool)
        ffp_pos = _position_letters(ffp_projections)
        assert pool_pos is not None and ffp_pos is not None, (
            "one side lost its position column -- every loose matching tier is "
            "scoped to position, so without it the merge is surname roulette."
        )
        crossed = {
            classic_player_pool.at[i, "Player"]: (
                pool_pos.at[i], ffp_projections.at[j, "Name"], ffp_pos.at[j])
            for i, j in claims.items() if pool_pos.at[i] != ffp_pos.at[j]
        }
        assert not crossed, (
            "%d FFP match(es) cross position, i.e. one player is being scored on "
            "another's projection: %s" % (len(crossed), crossed)
        )

    def test_cross_club_matches_agree_on_the_whole_name(self, claims, classic_player_pool,
                                                        ffp_projections):
        """Two sources disagree on a club for weeks after a transfer, which is why
        a cross-club tier exists at all. It is safe only while it demands the
        entire name: the surname-only version gave Abu Kamara (HUL) Boubacar
        Kamara's (AVL) 90% start rate."""
        from scripts.common.player_matching import canonical_normalize
        from scripts.common.text_helpers import _to_short_team_code

        loose = {}
        for i, j in claims.items():
            pool_team = _to_short_team_code(classic_player_pool.at[i, "Team"])
            ffp_team = _to_short_team_code(ffp_projections.at[j, "Team"])
            if pool_team == ffp_team:
                continue
            pool_name = canonical_normalize(classic_player_pool.at[i, "Player"])
            ffp_name = canonical_normalize(ffp_projections.at[j, "Name"])
            if pool_name != ffp_name:
                loose[classic_player_pool.at[i, "Player"]] = ffp_projections.at[j, "Name"]
        assert not loose, (
            "%d match(es) crossed clubs on a partial name: %s" % (len(loose), loose)
        )

    def test_match_rate_holds_over_the_full_pool(self, claims, classic_player_pool,
                                                 ffp_projections):
        """FFP lists a subset of the pool, so the reference is what is matchable."""
        raise_on_error(
            check_merge_match_rate(
                len(claims), len(ffp_projections), "FFP -> player pool",
                min_rate=0.80, input_rows=len(classic_player_pool),
            ),
            context="FFP merge",
        )


class TestTeamNameMapping:
    """Every current Premier League club must be in TEAM_FULL_TO_SHORT.

    Promotion and relegation change three clubs a year, and a missing one fails
    quietly: _to_short_team_code() falls back to a naive first-three-letters
    guess. Leeds happened to guess right ("LEE") and merely spammed the log, but
    "Sheffield Utd" guesses "SHE" against the real "SHU" -- and since player
    matching is scoped by team, a wrong code silently drops every match for that
    club. This test is the thing that notices in August.
    """

    @pytest.fixture(scope="class")
    def bootstrap_teams(self):
        from scripts.common.fpl_classic_api import get_classic_bootstrap_static

        bootstrap = skip_if_unreachable(get_classic_bootstrap_static, "FPL bootstrap")
        teams = (bootstrap or {}).get("teams", [])
        if not teams:
            pytest.skip("FPL bootstrap returned no teams")
        return teams

    def test_every_current_club_is_mapped(self, bootstrap_teams):
        from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

        missing = {t["name"]: t["short_name"]
                   for t in bootstrap_teams if t["name"] not in TEAM_FULL_TO_SHORT}
        assert not missing, (
            "TEAM_FULL_TO_SHORT is missing %d current club(s): %s. Add them to "
            "scripts/common/text_helpers.py -- until then _to_short_team_code() "
            "guesses the code from the first three letters."
            % (len(missing), missing)
        )

    def test_mapped_codes_match_the_official_ones(self, bootstrap_teams):
        """A code that is present but wrong is worse than one that is absent."""
        from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

        wrong = {
            t["name"]: (TEAM_FULL_TO_SHORT[t["name"]], t["short_name"])
            for t in bootstrap_teams
            if t["name"] in TEAM_FULL_TO_SHORT
            and TEAM_FULL_TO_SHORT[t["name"]] != t["short_name"]
        }
        assert not wrong, (
            "TEAM_FULL_TO_SHORT disagrees with the FPL bootstrap "
            "(club: mapped vs official): %s" % wrong
        )

    def test_every_ffp_club_label_resolves(self, bootstrap_teams, ffp_projections):
        """FFP spells clubs its own way -- "Notts Forest", not "Nott'm Forest".

        An unmapped label is not cosmetic: matching is scoped by team, so all 28
        Forest rows missed the exact tiers and fell through to the loose ones.
        """
        from scripts.common.text_helpers import _to_short_team_code

        official = {t["short_name"] for t in bootstrap_teams}
        labels = {str(t).strip() for t in ffp_projections["Team"].dropna().unique()}
        unresolved = {label: _to_short_team_code(label)
                      for label in labels
                      if _to_short_team_code(label) not in official}
        assert not unresolved, (
            "FFP club label(s) do not resolve to a current club (label: guess): "
            "%s. Add them to TEAM_FULL_TO_SHORT." % unresolved
        )

    def test_no_current_club_falls_through_to_the_guess(self, bootstrap_teams):
        """End-to-end over the function the app actually calls."""
        from scripts.common.text_helpers import _to_short_team_code

        mismatched = {
            t["name"]: (_to_short_team_code(t["name"]), t["short_name"])
            for t in bootstrap_teams
            if _to_short_team_code(t["name"]) != t["short_name"]
        }
        assert not mismatched, (
            "_to_short_team_code() returns the wrong code for: %s" % mismatched
        )


class TestDraftElementStates:
    """The Waiver Wire decides what to suggest from these states.

    If Draft renames a status code or stops populating `owner`, every player
    silently reads as available and the page resumes suggesting players who were
    dropped an hour ago and cannot be picked up. Nothing crashes; the advice is
    just wrong. That is exactly the class of failure this suite exists for.
    """

    @pytest.fixture(scope="class")
    def element_states(self, draft_league_id):
        from scripts.common.fpl_draft_api import get_league_element_states

        states = skip_if_unreachable(
            lambda: get_league_element_states(draft_league_id), "Draft element-status"
        )
        if not states:
            pytest.skip("Draft league has no element states (pre-draft?)")
        return states

    @pytest.fixture(scope="class")
    def team_count(self, draft_league_id):
        from scripts.common.fpl_draft_api import get_league_entries

        entries = skip_if_unreachable(
            lambda: get_league_entries(draft_league_id), "Draft league entries"
        )
        if not entries:
            pytest.skip("Draft league has no entries")
        return len(entries)

    def test_states_are_plausible(self, element_states, team_count):
        raise_on_error(
            check_element_states(element_states, expected_teams=team_count),
            context="Draft element states",
        )

    def test_locked_players_exist_and_are_unowned(self, element_states):
        """Locked is the state the Waiver Wire fix turns on. Assert it is real
        and distinguishable — not merely absent from the payload."""
        locked = [s for s in element_states.values() if s.get("status") == "l"]
        for state in locked:
            assert state.get("owner") is None, (
                "A locked player belongs to nobody -- they were just dropped or "
                "added. An owner here means the status codes have shifted meaning."
            )

    def test_transaction_window_is_one_of_the_two_known_modes(self, draft_league_id):
        from scripts.common.fpl_draft_api import (
            get_draft_transaction_window,
            TRANSACTION_MODE_FREE_AGENCY,
            TRANSACTION_MODE_WAIVERS,
        )

        window = skip_if_unreachable(
            lambda: get_draft_transaction_window(draft_league_id), "Draft transaction window"
        )
        mode = window.get("mode")
        if mode is None:
            pytest.skip("transaction_mode not published for this league")
        assert mode in (TRANSACTION_MODE_FREE_AGENCY, TRANSACTION_MODE_WAIVERS), (
            "Unknown transaction_mode %r. The Waiver Wire banner labels anything "
            "that is not free-agency as a pending waiver round, so a third mode "
            "would be mislabelled." % mode
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


class TestSeasonRankingsMerge:
    """The merge that silently stopped matching.

    Rotowire publishes common names ("Bruno Fernandes"); the FPL bootstrap
    publishes full legal names ("Bruno Borges Fernandes"). A strict single-key
    merge matched 356 of 425 season-ranking rows and said nothing about the
    other 69 -- among them the #2 asset in the game -- each of which fell back
    to a neutral 0.5 percentile and rendered as an exactly average player.

    Nothing here asserts *which* players match; upstream squads change weekly.
    It asserts that the merge is still doing its job at all.
    """

    def test_season_rankings_match_the_fpl_pool(self, classic_player_pool,
                                                rotowire_season_rankings):
        stats = {}
        merge_season_projections(
            classic_player_pool, rotowire_season_rankings, stats=stats)
        issues = check_merge_match_rate(
            stats["matched"], stats["total"], "Rotowire season rankings")
        raise_on_error(issues, "season rankings merge")
        assert not issues, format_issues(issues)

    def test_no_reference_row_is_claimed_twice(self, classic_player_pool,
                                               rotowire_season_rankings):
        """Two FPL players sharing one ranking row means a false positive.

        A fuzzy tier once matched "Harrison" to team-mate "Harry Wilson" on
        character similarity alone, handing one player the other's season total.
        """
        stats = {}
        merge_season_projections(
            classic_player_pool, rotowire_season_rankings, stats=stats)
        assert stats["matched"] <= stats["total"], (
            "matched more players than there are reference rows -- some ranking "
            "row was claimed by two different players"
        )

    def test_top_ranked_players_all_resolve(self, classic_player_pool,
                                            rotowire_season_rankings):
        """The elite end is where a miss costs most, so hold it to a higher bar."""
        top = rotowire_season_rankings.nsmallest(50, "Overall Rank") \
            if "Overall Rank" in rotowire_season_rankings.columns \
            else rotowire_season_rankings.nlargest(50, "Points")
        stats = {}
        merge_season_projections(classic_player_pool, top, stats=stats)
        issues = check_merge_match_rate(
            stats["matched"], stats["total"],
            "Rotowire season rankings (top 50)", min_rate=0.95)
        raise_on_error(issues, "top-50 season rankings merge")


class TestLiveFixtureStatus:
    """The state the lineup cards read to say Played / Did not play / Upcoming.

    A 0-minute player whose match was over used to render as "Upcoming" for the
    rest of the week and keep his full projection in the team total. Minutes alone
    cannot see that, so the fixture's own state has to come through intact.
    """

    @pytest.fixture(scope="class")
    def fixture_status(self, current_gw):
        from scripts.common.utils import get_gw_team_fixture_status

        return skip_if_unreachable(
            lambda: get_gw_team_fixture_status(current_gw), "FPL fixtures endpoint")

    @pytest.fixture(scope="class")
    def live_stats(self, current_gw):
        from scripts.common.utils import get_live_gameweek_stats

        stats = skip_if_unreachable(
            lambda: get_live_gameweek_stats(current_gw), "FPL live endpoint")
        if not stats:
            pytest.skip("No live stats published for GW %s" % current_gw)
        return stats

    def test_every_club_has_a_fixture_state(self, fixture_status):
        """20 clubs, each either playing or not -- a blank map disables the whole
        played/upcoming distinction silently."""
        assert len(fixture_status) >= 18, (
            "only %d clubs carry a fixture state; blank/near-blank means the "
            "fixtures endpoint changed shape" % len(fixture_status)
        )
        for team_id, state in fixture_status.items():
            assert set(state) == {"started", "finished"}, (
                "team %s carries %s" % (team_id, sorted(state)))
            if state["finished"]:
                assert state["started"], "team %s finished without starting" % team_id

    def test_live_stats_carry_the_fixture_state(self, live_stats):
        keys_missing = [
            eid for eid, s in live_stats.items()
            if "fixture_started" not in s or "fixture_finished" not in s
        ]
        assert not keys_missing, (
            "%d players have no fixture state; the lineup cards fall back to "
            "minutes-only and unused subs read as Upcoming" % len(keys_missing)
        )

    def test_players_with_minutes_are_in_a_started_fixture(self, live_stats):
        """Minutes without a started fixture means the two feeds disagree."""
        contradictions = [
            eid for eid, s in live_stats.items()
            if s.get("minutes", 0) > 0 and not s.get("fixture_started")
        ]
        assert not contradictions, (
            "%d players logged minutes in a fixture reported as not started "
            "(e.g. element %s)" % (len(contradictions), contradictions[0])
        )

    def test_a_finished_fixture_produces_minutes(self, live_stats, fixture_status):
        """Every finished match must show players who played. If a whole club's
        squad reads 0 minutes after full time, the join broke and all of them
        would render as unused subs."""
        finished = {t for t, s in fixture_status.items() if s.get("finished")}
        if not finished:
            pytest.skip("No fixture finished yet in this gameweek")
        played = sum(
            1 for s in live_stats.values()
            if s.get("fixture_finished") and s.get("minutes", 0) > 0
        )
        # 22 starters per finished match, before substitutes.
        assert played >= 11 * len(finished), (
            "only %d players logged minutes across %d finished clubs"
            % (played, len(finished))
        )
