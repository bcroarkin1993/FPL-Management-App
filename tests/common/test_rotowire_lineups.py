"""Rotowire's lineups page as the projection engine's start signal.

Rotowire publishes its weekly numbers twice -- a rankings *article* carrying
projected points, and a lineups page carrying projected XIs and no points. The
engine took "expects him to start" from the article, which is written once a
week, and presence there clips start probability *up* to the positional floor.

Measured live for GW8 on 2026-10-21, with the article stamped the previous
morning: Phil Foden was benched on the lineups page, still in the article's XI,
rated 20% to start by FFP -- and rendered by this app at **86%** and 4.24
expected points, while its own Projected Lineups page showed him on the bench.
"""

import pandas as pd
import pytest

from scripts.common import projection_engine as pe
from scripts.common import rotowire_lineups as rl
from scripts.common.name_matching import ReferenceMatcher


def _pool():
    """Two clubs, enough players each to make an XI plus substitutes."""
    rows = []
    pid = 1
    for team in ("MCI", "LIV"):
        for i in range(14):
            pos = "G" if i == 0 else "D" if i < 5 else "M" if i < 11 else "F"
            rows.append({"Player_ID": pid, "Player": "First%d %s%d" % (pid, team, pid),
                         "Web_Name": "%s%d" % (team, pid), "Team": team,
                         "Position": pos})
            pid += 1
    return pd.DataFrame(rows)


def _xi_frame(pool, team, take):
    """A scraped-lineups frame naming ``take`` players of ``team``'s pool."""
    club = {"MCI": "Manchester City", "LIV": "Liverpool"}[team]
    rows = pool[pool["Team"] == team].head(take)
    tac = {"G": "GK", "D": "DC", "M": "MC", "F": "FW"}
    return pd.DataFrame([
        {"Team": club, "Position": tac[r["Position"]], "Player": r["Player"],
         "MatchupIndex": 0}
        for _, r in rows.iterrows()
    ])


class TestLineupStarters:
    def test_a_complete_xi_can_both_promote_and_bench(self):
        pool = _pool()
        xi = rl.lineup_starters(_xi_frame(pool, "MCI", 11), pool, gameweek=8)
        assert len(xi.player_ids) == 11
        assert xi.clubs == {"MCI"}

    def test_a_partial_xi_promotes_but_never_benches(self):
        """The asymmetry is the safety design.

        "He starts" needs only his own name to have resolved. "He does not
        start" is a claim about everyone the XI leaves out, so an unresolved
        starter is indistinguishable from a benched one -- and getting that
        wrong does not degrade his projection, it inverts it.
        """
        pool = _pool()
        frame = _xi_frame(pool, "MCI", 11)
        frame.loc[frame.index[-1], "Player"] = "Someone Unmatchable"
        xi = rl.lineup_starters(frame, pool, gameweek=8)
        assert len(xi.player_ids) == 10, "the ten that resolved stay promotable"
        assert xi.clubs == set(), "nobody may be benched off an incomplete XI"

    def test_an_unmapped_club_is_dropped_rather_than_benched(self):
        pool = _pool()
        frame = _xi_frame(pool, "MCI", 11)
        frame["Team"] = "Some New Club FC"
        xi = rl.lineup_starters(frame, pool, gameweek=8)
        assert xi.player_ids == set()
        assert xi.clubs == set()

    def test_no_lineups_is_no_opinion(self):
        xi = rl.lineup_starters(pd.DataFrame(), _pool(), gameweek=8)
        assert not xi.ok and xi.player_ids == set()

    def test_clubs_come_back_as_fpl_short_codes(self):
        """The engine compares them against the pool's own Team column."""
        pool = _pool()
        xi = rl.lineup_starters(_xi_frame(pool, "LIV", 11), pool, gameweek=8)
        assert xi.clubs == {"LIV"}


class TestTierAwareRetry:
    """The cross-position retry must prefer the stronger match, not abstain.

    Live at GW8 "B. Thomas-Asante" matched Brandon Thomas-Asante (COV, FWD) on
    a complete surname and his team-mate *Bobby Thomas* on a looser tier. Two
    answers read as ambiguity, so the retry abstained and threw away a match an
    exact token had earned -- leaving Coventry's XI one player short of the
    eleven the benching gate requires.
    """

    @staticmethod
    def _matcher():
        pool = pd.DataFrame({
            "Player_ID": [1, 2],
            "Player": ["Bobby Thomas", "Brandon Thomas-Asante"],
            "Web_Name": ["Thomas", "Thomas-Asante"],
            "Team": ["COV", "COV"],
            "Position": ["D", "F"],
        })
        return ReferenceMatcher(pool, name_col="Player", web_name_col="Web_Name",
                                team_col="Team", position_col="Position"), pool

    def test_the_stronger_tier_wins_the_retry(self):
        matcher, pool = self._matcher()
        # Rotowire files him in midfield, where FPL registers neither player.
        hit = rl.resolve_lineup_name(matcher, "B. Thomas-Asante", "COV", "M")
        assert hit is not None
        assert pool.loc[hit, "Player"] == "Brandon Thomas-Asante"

    def test_a_tie_at_one_tier_still_resolves_to_nothing(self):
        """Two team-mates whose surnames resolve at the *same* tier are a real
        ambiguity, and ambiguity must stay unresolved -- the Alex/Cole Palmer
        lesson. Preferring the stronger tier only breaks contention between
        tiers, never within one."""
        pool = pd.DataFrame({
            "Player_ID": [1, 2],
            "Player": ["Bobby Thomas", "Luke Thomas"],
            "Web_Name": ["B.Thomas", "L.Thomas"],
            "Team": ["COV", "COV"],
            "Position": ["D", "M"],
        })
        matcher = ReferenceMatcher(pool, name_col="Player", web_name_col="Web_Name",
                                   team_col="Team", position_col="Position")
        assert rl.resolve_lineup_name(matcher, "Thomas", "COV", "F") is None


class TestStarterOverride:
    """The tri-state the engine consumes."""

    @staticmethod
    def _args(index=(1, 2, 3, 4)):
        idx = pd.Index(index, name="Player_ID")
        return {
            "index": idx,
            "player_ids": pd.Series(list(index), index=idx),
            "teams": pd.Series(["MCI", "MCI", "LIV", "LIV"], index=idx),
        }

    def test_three_states(self):
        xi = rl.LineupXI({1, 3}, {"MCI"})
        ov = pe.starter_override(xi, **self._args())
        assert ov.loc[1] == 1.0, "in the XI"
        assert ov.loc[2] == 0.0, "MCI resolved completely and left him out"
        assert ov.loc[3] == 1.0, "in the XI, at a club that did not resolve fully"
        assert pd.isna(ov.loc[4]), "no opinion: LIV cannot bench anybody"

    def test_no_xi_is_no_override(self):
        assert pe.starter_override(None, **self._args()) is None
        assert pe.starter_override(rl.LineupXI(set(), set()), **self._args()) is None

    def test_without_team_labels_it_can_only_promote(self):
        args = self._args()
        args["teams"] = None
        ov = pe.starter_override(rl.LineupXI({1}, {"MCI"}), **args)
        assert ov.loc[1] == 1.0
        assert ov.loc[2:].isna().all()


class TestEngineReadsTheOverride:
    """End to end: the article prices him, the lineups page benches him."""

    @staticmethod
    def _blend(override=None):
        idx = pd.Index([1, 2], name="Player_ID")
        return pe.blend_aligned(
            index=idx,
            # Both players are in the rankings article at 5 points.
            per_source_raw={"rotowire": pd.Series([5.0, 5.0], index=idx),
                            "ffp": pd.Series([5.0, 5.0], index=idx)},
            per_source_basis={"rotowire": pe.BASIS_CONDITIONAL,
                              "ffp": pe.BASIS_CONDITIONAL},
            # FFP rates one of them 20% to start, as it rated Foden.
            per_source_startpct={"ffp": pd.Series([0.20, 0.90], index=idx)},
            starters_only={"rotowire"},
            positions=pd.Series(["M", "M"], index=idx),
            teams=pd.Series(["MCI", "MCI"], index=idx),
            source_club_coverage={"rotowire": {"MCI": 11}},
            starter_overrides=override or {},
        )

    def test_without_the_override_the_article_clips_him_up(self):
        out = self._blend()
        floor = pe._start_floors()["M"]
        assert out.loc[1, "Start_Pct"] == pytest.approx(floor)

    def test_the_lineups_page_overrules_the_stale_article(self):
        idx = pd.Index([1, 2], name="Player_ID")
        # Benched by a complete XI that includes player 2 but not player 1.
        override = {"rotowire": pd.Series([0.0, 1.0], index=idx)}
        out = self._blend(override)
        floor = pe._start_floors()["M"]
        assert out.loc[1, "Start_Pct"] < 0.20, (
            "a benched player must fall below even FFP's pessimistic 20%, to the "
            "blend of it with the omitted prior")
        # The floor only ever clips *upward*, so a player FFP already rates
        # above it keeps FFP's number.
        assert out.loc[2, "Start_Pct"] >= floor
        assert out.loc[1, "Proj"] < out.loc[2, "Proj"]

    def test_proj_start_is_untouched(self):
        """Only the start model moves. He would still score 5 if he played, and
        `team_strength` percentiles that number."""
        idx = pd.Index([1, 2], name="Player_ID")
        before = self._blend()
        after = self._blend({"rotowire": pd.Series([0.0, 1.0], index=idx)})
        pd.testing.assert_series_equal(before["Proj_Start"], after["Proj_Start"])

    def test_a_player_the_article_omitted_is_promoted_by_the_xi(self):
        """The signal runs both ways: Gvardiol and Gakpo moved *up* live."""
        idx = pd.Index([1], name="Player_ID")
        out = pe.blend_aligned(
            index=idx,
            per_source_raw={"rotowire": pd.Series([float("nan")], index=idx),
                            "ffp": pd.Series([5.0], index=idx)},
            per_source_basis={"rotowire": pe.BASIS_CONDITIONAL,
                              "ffp": pe.BASIS_CONDITIONAL},
            per_source_startpct={"ffp": pd.Series([0.30], index=idx)},
            starters_only={"rotowire"},
            positions=pd.Series(["M"], index=idx),
            teams=pd.Series(["MCI"], index=idx),
            source_club_coverage={"rotowire": {"MCI": 11}},
            starter_overrides={"rotowire": pd.Series([1.0], index=idx)},
        )
        assert out.loc[1, "Start_Pct"] == pytest.approx(pe._start_floors()["M"])


class TestRevisionIsStamped:
    def test_the_start_model_revision_moved(self):
        """A logic change is not recoverable from an archived snapshot the way a
        constant retune is: the snapshots record `Start_Pct__rotowire` but not
        which players the lineups page had in the XI. So the fitter must refuse
        older snapshots rather than replay them against a different signal."""
        assert pe.START_MODEL_REVISION >= 3


class TestTheTwoPagesShareOneResolver:
    """Projected Lineups and the engine must agree about who a name is.

    The page renders a card per lineup entry; the engine decides whether that
    same entry means "expects to start". Two resolvers would let one card show
    a player the engine had benched -- the conflicting-information failure this
    whole change exists to end, one level down.
    """

    def test_the_page_resolves_through_the_shared_function(self):
        import inspect
        from scripts.fpl import projected_lineups as pl
        src = inspect.getsource(pl.PlayerIndex.lookup)
        assert "resolve_lineup_name" in src
        # And it is the same object, not a second copy under the same name.
        assert pl.resolve_lineup_name is rl.resolve_lineup_name

    def test_the_page_does_not_reimplement_the_scraper(self):
        from scripts.fpl import projected_lineups as pl
        for name in ("scrape_lineups", "extract_players", "LINEUP_COLUMNS",
                     "ROTOWIRE_TACTICAL_TO_POSITION"):
            assert getattr(pl, name) is getattr(rl, name), (
                "%s must be the shared implementation, not a copy" % name)


class TestValidation:
    def test_an_empty_xi_is_an_error(self):
        from scripts.common.data_validation import check_rotowire_lineup_xi
        issues = check_rotowire_lineup_xi(rl.LineupXI(set(), set(), 8, "unreachable"))
        assert any(i.severity == "error" for i in issues)

    def test_a_healthy_xi_is_clean(self):
        from scripts.common.data_validation import check_rotowire_lineup_xi
        xi = rl.LineupXI(set(range(220)), {"T%d" % i for i in range(20)}, 8)
        assert check_rotowire_lineup_xi(xi) == []

    def test_two_gameweeks_stacked_is_an_error(self):
        """The failure the ordered (home, away) fixture filter exists for."""
        from scripts.common.data_validation import check_rotowire_lineup_xi
        xi = rl.LineupXI(set(range(440)), {"T%d" % i for i in range(20)}, 8)
        issues = check_rotowire_lineup_xi(xi)
        assert any(i.severity == "error" for i in issues)

    def test_losing_most_clubs_is_an_error(self):
        from scripts.common.data_validation import check_rotowire_lineup_xi
        xi = rl.LineupXI(set(range(55)), {"T%d" % i for i in range(5)}, 8)
        issues = check_rotowire_lineup_xi(xi)
        assert any(i.severity == "error" for i in issues)


class TestEveryFrameShapeCanReachTheOverride:
    """The id column is spelled differently on every page, and a lookup keyed
    on one name silently does nothing on the pages using another.

    That is not hypothetical twice over. It is how the Rotowire *omission*
    penalty came to be a no-op on every per-squad page, and it immediately
    repeated here: keyed on ``Player_ID`` alone, the lineup override was off on
    Classic Fixture Projections -- whose squad builders emit ``element_id`` --
    so Phil Foden kept rendering at 4.24 there while the engine had him at 0.45
    everywhere else. Reported from the app, after the fix was supposedly in.
    """

    import pytest as _pytest

    @_pytest.mark.parametrize("id_col", ["Player_ID", "element_id", "element",
                                         "player_id"])
    def test_each_spelling_of_the_element_id_resolves(self, id_col):
        from scripts.common import analytics
        df = pd.DataFrame({id_col: [398, 367], "Team": ["MCI", "LIV"],
                           "Position": ["M", "M"], "Points": [4.88, float("nan")]})
        ids = analytics._player_id_col(df)
        assert ids is not None and list(ids) == [398, 367]

    def test_a_bare_id_column_is_not_trusted(self):
        """On these pages ``id`` is as likely to be a league or fixture id, and
        resolving a start signal against the wrong id space is worse than not
        resolving it at all."""
        from scripts.common import analytics
        df = pd.DataFrame({"id": [1, 2], "Team": ["MCI", "LIV"]})
        assert analytics._player_id_col(df) is None

    def test_a_classic_squad_frame_gets_a_stated_opinion(self):
        from unittest.mock import patch
        from scripts.common import analytics
        from scripts.common import rotowire_lineups as rlm

        df = pd.DataFrame({"element_id": [398, 367], "Team": ["MCI", "MCI"],
                           "Position": ["M", "M"]})
        with patch.object(analytics, "_lineup_xis",
                          return_value={"rotowire": rlm.LineupXI({367}, {"MCI"}, 8)}):
            overrides = analytics._frame_starter_overrides(df)
        assert "rotowire" in overrides
        ov = overrides["rotowire"]
        assert ov.iloc[0] == 0.0, "benched: MCI's XI resolved and left him out"
        assert ov.iloc[1] == 1.0

    def test_a_frame_with_no_id_warns_rather_than_failing_silently(self, caplog):
        """The symptom of this going wrong is a stale projection that looks
        entirely ordinary, so it has to say something."""
        import logging
        from scripts.common import analytics

        df = pd.DataFrame({"Team": ["MCI"], "Position": ["M"]})
        with caplog.at_level(logging.WARNING):
            assert analytics._frame_starter_overrides(df) == {}
        assert any("carries no element id" in r.getMessage()
                   for r in caplog.records)
