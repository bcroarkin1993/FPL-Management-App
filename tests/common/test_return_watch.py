"""Unit tests for scripts/common/return_watch.py.

The detector exists because a player returning from a long absence is invisible
to every score the app has. That makes most of these tests structural: there is
no projection to corroborate a row against, so what has to be pinned is *which
players reach the board at all* and *what they are ranked against*.

Two of them pin cuts that measurement rejected -- a per-rung reference, and
multiplying the rate by the start share. Both produced boards that looked
entirely plausible and were wrong, so they are the tests most worth keeping.
"""

import numpy as np
import pandas as pd
import pytest

import config
from scripts.common.data_validation import check_return_watch
from scripts.common.return_watch import (
    BAND_CLEARED,
    BAND_IMMINENT,
    BAND_WATCH,
    RUNG_PRESEASON,
    RUNG_PRIOR,
    RUNG_THIS,
    THIS_SEASON_MIN_STARTS,
    WATCH_COLUMNS,
    build_return_watch,
    claim_phrase,
    return_phrase,
    role_phrase,
)

GW = 11
ELAPSED = GW - 1


def _pool(rows):
    """A pool frame with the columns the detector reads."""
    base = dict(news="", status="a", chance_of_playing_next_round=100,
                minutes=0, total_points=0, starts=0,
                Season_PP90=np.nan, SeasonProjection=np.nan)
    out = []
    for i, row in enumerate(rows, start=1):
        merged = dict(base)
        merged.update(row)
        merged.setdefault("Player_Code", 1000 + i)
        merged.setdefault("Player_ID", i)
        merged.setdefault("Player", merged.get("name", "Player %d" % i))
        merged.setdefault("Team", "ARS")
        merged.setdefault("Position", "D")
        merged.pop("name", None)
        out.append(merged)
    return pd.DataFrame(out)


def _starters(position="D", n=12, rate=3.0, team="ARS", first_code=5000):
    """Established starters, who are the percentile reference."""
    return [dict(Player_Code=first_code + i, Player_ID=900 + i,
                 Player="Starter %d %s" % (i, position), Team=team,
                 Position=position, starts=ELAPSED, minutes=90 * ELAPSED,
                 total_points=rate * ELAPSED)
            for i in range(n)]


def _states(pool, state="a"):
    return {int(c): state for c in pool["Player_Code"]}


# The reported case, as the live bootstrap published it.
SALIBA = dict(Player_Code=462424, Player_ID=6, Player="William Saliba",
              Team="ARS", Position="D", starts=0, minutes=0, total_points=0,
              status="a", chance_of_playing_next_round=100, news="")
SALIBA_PRIOR = pd.DataFrame([dict(
    Player_Code=462424, Prior_PPS=137 / 30, Prior_Role=30 / 38,
    Prior_Season="2025/26", Prior_Starts=30, Prior_Points=137)])


class TestTheReportedCase:

    def test_saliba_reaches_the_board(self):
        """Zero of everything this season, cleared by FPL, elite last season."""
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert list(out.columns) == WATCH_COLUMNS
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Band"] == BAND_CLEARED
        assert row["Pedigree_Rung"] == RUNG_PRIOR
        assert row["Pedigree_Rate"] == pytest.approx(137 / 30, abs=1e-6)
        assert row["Role"] == pytest.approx(30 / 38, abs=1e-6)

    def test_his_evidence_names_the_season_it_came_from(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        ev = out[out["Player_Code"] == 462424].iloc[0]["Evidence"]
        assert ev["rung"] == RUNG_PRIOR
        assert ev["season"] == "2025/26"
        assert ev["season_starts"] == 30
        assert ev["band"] == BAND_CLEARED
        assert "30 of 38" in role_phrase(
            out[out["Player_Code"] == 462424].iloc[0])
        assert "2025/26" in role_phrase(
            out[out["Player_Code"] == 462424].iloc[0])


class TestTheRungLadder:

    def test_this_season_wins_where_the_sample_is_adequate(self):
        """It is the only rung describing him now."""
        pool = _pool([dict(Player_Code=7, Player_ID=7, starts=THIS_SEASON_MIN_STARTS,
                           total_points=40, status="i", news="Knee injury")]
                     + _starters(n=12))
        prior = pd.DataFrame([dict(Player_Code=7, Prior_PPS=1.0, Prior_Role=0.9,
                                   Prior_Season="2025/26", Prior_Starts=34,
                                   Prior_Points=34)])
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior)
        row = out[out["Player_Code"] == 7].iloc[0]
        assert row["Pedigree_Rung"] == RUNG_THIS
        assert row["Pedigree_Rate"] == pytest.approx(10.0)

    def test_prior_season_covers_a_player_with_nothing_this_season(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert out[out["Player_Code"] == 462424].iloc[0]["Pedigree_Rung"] == RUNG_PRIOR

    def test_the_preseason_rate_covers_a_player_with_no_pl_history(self):
        """A new signing or a promoted-club player.

        His role comes from the nineties Rotowire's own two columns imply:
        `Points / PP90 / 38`. Here 152 / 4.0 = 38 nineties, i.e. a full season.
        """
        pool = _pool([dict(Player_Code=8, Player_ID=8, Season_PP90=4.0,
                           SeasonProjection=152.0)] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW)
        row = out[out["Player_Code"] == 8].iloc[0]
        assert row["Pedigree_Rung"] == RUNG_PRESEASON
        assert row["Pedigree_Rate"] == pytest.approx(4.0)
        assert row["Role"] == pytest.approx(1.0)

    def test_a_player_no_rung_can_rate_is_absent(self):
        pool = _pool([dict(Player_Code=9, Player_ID=9)] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW)
        assert 9 not in set(out["Player_Code"])


class TestTheReferenceIsEstablishedStarters:
    """The percentile's denominator, and the thing two failed cuts got wrong."""

    def test_the_reference_is_not_the_candidate_cohort(self):
        """Scoping it to the candidates makes it "elite among benchwarmers".

        The first cut percentiled each rung against its own rung, and rung 2
        only applies to players who are *not playing* -- so the reference was
        selected on the very thing being measured and a mediocre absentee came
        out at 0.92. Here the candidate's 3.0 is exactly the starters' rate, so
        an honest reference puts him mid-pack; a self-referential one would put
        the only candidate at the top.
        """
        pool = _pool([dict(Player_Code=11, Player_ID=11)] + _starters(rate=3.0))
        prior = pd.DataFrame([dict(Player_Code=11, Prior_PPS=3.0, Prior_Role=0.9,
                                   Prior_Season="2025/26", Prior_Starts=34,
                                   Prior_Points=102)])
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior, min_pctile=0.0)
        assert out.iloc[0]["Pedigree"] < 0.5

    def test_a_small_sample_starter_does_not_set_the_top(self):
        """Players with few starts are out of the denominator.

        Pooling everyone with four starts let a 4-start hot streak define the
        ceiling, and a genuine elite defender then landed at the 77th
        percentile. The 8.0-a-start player below has only four starts, so he is
        a candidate's *peer*, never part of the reference.
        """
        pool = _pool([dict(Player_Code=12, Player_ID=12, starts=4,
                           total_points=32, status="i", news="Knee injury")]
                     + _starters(rate=3.0))
        # `min_role=0` so the role gate cannot be what excludes him -- this test
        # is about the reference, and his 4-of-10 share would fail the gate.
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 min_pctile=0.0, min_role=0.0)
        assert out[out["Player_Code"] == 12].iloc[0]["Pedigree"] == pytest.approx(1.0)

    def test_a_thin_reference_is_reported(self):
        pool = _pool([SALIBA] + _starters(n=2))
        notes = []
        build_return_watch(pool, _states(pool), current_gw=GW,
                           prior_rates=SALIBA_PRIOR, notes=notes)
        assert any("established starter" in m for _s, m in notes)

    def test_positions_are_normalised_before_grouping(self):
        """Every percentile here is grouped by position.

        `pull_fpl_player_stats` emits GKP/DEF/MID/FWD while the app groups on
        G/D/M/F, and wrong codes match no group -- the failure that once had
        every Power Rankings team scoring exactly 50.
        """
        rows = [dict(SALIBA, Position="DEF")]
        rows += [dict(r, Position="DEF") for r in _starters()]
        pool = _pool(rows).rename(columns={"Position": "position_abbrv"})
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert set(out["Position"]) == {"D"}


class TestTheRoleGate:

    def test_a_high_rate_off_few_starts_is_pruned(self):
        """The Tomiyasu shape.

        7.00 points a start off 10 starts outranked every genuine elite
        defender until the role gate removed him. 10/38 is 0.26, well under the
        0.60 bar.
        """
        pool = _pool([dict(Player_Code=13, Player_ID=13)] + _starters())
        prior = pd.DataFrame([dict(Player_Code=13, Prior_PPS=7.0,
                                   Prior_Role=10 / 38, Prior_Season="2025/26",
                                   Prior_Starts=10, Prior_Points=70)])
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior)
        assert 13 not in set(out["Player_Code"])

    def test_the_rate_and_the_role_may_come_from_different_rungs(self):
        """A player hurt in GW3 has a depressed *this-season* share.

        His rate is best read from this season, but `starts / elapsed` is
        dragged down by the very absence that put him on the board -- so the
        role takes the most favourable rung and records which.
        """
        pool = _pool([dict(Player_Code=14, Player_ID=14, starts=4,
                           total_points=24, status="i",
                           news="Knee injury - Unknown return date")]
                     + _starters())
        prior = pd.DataFrame([dict(Player_Code=14, Prior_PPS=2.0, Prior_Role=0.95,
                                   Prior_Season="2025/26", Prior_Starts=36,
                                   Prior_Points=72)])
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior, min_pctile=0.0)
        row = out[out["Player_Code"] == 14].iloc[0]
        assert row["Pedigree_Rung"] == RUNG_THIS           # the rate
        assert row["Pedigree_Rate"] == pytest.approx(6.0)
        assert row["Role_Src"] == RUNG_PRIOR               # the role
        assert row["Role"] == pytest.approx(0.95)

    def test_the_rate_is_never_multiplied_by_the_role(self):
        """The second failed cut.

        `rate x share` mixes two denominators -- this season's share is over
        elapsed gameweeks and a prior season's over all 38 -- and it took the
        live board to zero rows. The published rate must be the rate.
        """
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Pedigree_Rate"] == pytest.approx(137 / 30, abs=1e-6)
        assert row["Pedigree_Rate"] != pytest.approx((137 / 30) * (30 / 38), abs=1e-3)


class TestThisSeasonOverrulesPriorRole:
    """Reported from the app: Marcos Senesi topped the board on last season's
    97% start share while having started **2 of 10** for his new club -- fit the
    whole time, simply not picked.

    The most favourable rung is right for a player whose *absence* depressed his
    share and wrong for one whose *selection* did, and what separates them is
    whether he is currently out. These are the real live numbers.
    """

    @staticmethod
    def _candidate(**kw):
        row = dict(Player_Code=462424, Player_ID=6, Player="Test Player",
                   Team="ARS", Position="D", starts=0, minutes=0, total_points=0,
                   status="a", chance_of_playing_next_round=100, news="")
        row.update(kw)
        return row

    @staticmethod
    def _prior(role=30 / 38, apps=0):
        return pd.DataFrame([dict(Player_Code=462424, Prior_PPS=137 / 30,
                                  Prior_Role=role, Prior_Season="2025/26",
                                  Prior_Starts=round(role * 38), Prior_Points=137,
                                  Appearances=apps)])

    def _run(self, candidate, prior):
        pool = _pool([candidate] + _starters())
        return build_return_watch(pool, _states(pool), current_gw=GW,
                                  prior_rates=prior, min_pctile=0.0)

    def test_senesi_is_dropped(self):
        """Fit, 178 minutes across 2 appearances, 2 starts of 10. Not a returnee."""
        out = self._run(self._candidate(minutes=178, starts=2, total_points=9),
                        self._prior(role=37 / 38, apps=2))
        assert 462424 not in set(out["Player_Code"])

    def test_saliba_is_kept(self):
        """Fit, but *zero* appearances -- this season has said nothing at all.

        This is the case the whole feature exists for, and it is why no
        appearances can never mean "not a starter": a man FPL never flagged and
        a man who is simply unused look identical from here.
        """
        out = self._run(self._candidate(minutes=0, starts=0),
                        self._prior(role=30 / 38, apps=0))
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Role"] == pytest.approx(30 / 38)
        assert row["Role_Src"] == RUNG_PRIOR
        assert row["Evidence"]["role_from_this_season"] is False

    def test_a_fit_substitute_is_dropped_on_appearances_alone(self):
        """Kulusevski: 42 minutes, but *four* appearances and no starts.

        Under a minutes-only threshold he would survive on 42 minutes, which is
        why the appearance count is the other half of the test.
        """
        out = self._run(self._candidate(minutes=42, starts=0, total_points=4),
                        self._prior(role=27 / 38, apps=4))
        assert 462424 not in set(out["Player_Code"])

    def test_a_player_just_back_keeps_his_old_role(self):
        """Minteh: 8 minutes in 1 appearance -- too little to conclude anything."""
        out = self._run(self._candidate(minutes=8, starts=0, total_points=1),
                        self._prior(role=26 / 38, apps=1))
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Role_Src"] == RUNG_PRIOR
        assert row["Evidence"]["role_from_this_season"] is False

    def test_an_injured_regular_keeps_his_prior_role(self):
        """Semenyo: 6 starts of 10 then a knee injury.

        His this-season share is depressed by the injury, which is evidence
        about the injury and not about whether his manager rates him. Without
        this branch a man hurt early enough would be dropped for being hurt.
        """
        out = self._run(
            self._candidate(minutes=553, starts=6, total_points=37, status="i",
                            chance_of_playing_next_round=0,
                            news="Knee injury - Unknown return date"),
            self._prior(role=37 / 38, apps=7))
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Role"] == pytest.approx(37 / 38)
        assert row["Evidence"]["role_from_this_season"] is False

    def test_a_fit_player_with_evidence_can_never_reach_the_board(self):
        """The override is purely exclusionary, and that is structural.

        Once this season decides, a fit player's role *is* his start rate -- and
        the not-playing gate has already capped that at 0.25, well under the
        0.60 role bar. So no fit player with real current-season evidence can
        satisfy both, whatever he did last year. That is the intended meaning:
        a man who is available and not being picked is not a returning asset.
        """
        for minutes, starts, apps in [(178, 2, 2), (126, 1, 8), (42, 0, 4),
                                      (227, 2, 4), (99, 1, 3)]:
            out = self._run(
                self._candidate(minutes=minutes, starts=starts, total_points=10),
                self._prior(role=37 / 38, apps=apps))
            assert 462424 not in set(out["Player_Code"]), \
                "fit, %d mins / %d apps / %d starts should be excluded" % (
                    minutes, apps, starts)

    def test_role_src_can_still_read_this_season_for_an_injured_player(self):
        """Via the best-rung branch, not the override.

        Jorrel Hato: 6 starts of 10 then injured, and a *weaker* prior season --
        so the best rung is this one. The override flag stays False, because he
        is out and this season's share was not what decided it.
        """
        out = self._run(
            self._candidate(minutes=412, starts=6, total_points=25, status="i",
                            chance_of_playing_next_round=0,
                            news="Unspecified injury - Unknown return date"),
            self._prior(role=0.3, apps=6))
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Role_Src"] == RUNG_THIS
        assert row["Role"] == pytest.approx(0.60)
        assert row["Evidence"]["role_from_this_season"] is False
        assert "this season" in role_phrase(row)

    def test_without_appearance_data_the_minutes_threshold_still_applies(self):
        """`Appearances` is absent whenever the history fetch failed.

        The rung must degrade rather than switch off, so minutes alone still
        decide -- a frame with no appearance column is an ordinary state.
        """
        prior = self._prior(role=37 / 38).drop(columns=["Appearances"])
        out = self._run(self._candidate(minutes=178, starts=2, total_points=9),
                        prior)
        assert 462424 not in set(out["Player_Code"])

    def test_thresholds_are_configurable(self):
        candidate = self._candidate(minutes=178, starts=2, total_points=9)
        pool = _pool([candidate] + _starters())
        # Raise the bar past his minutes and appearances: he is a returnee again.
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=self._prior(role=37 / 38, apps=2),
                                 min_pctile=0.0, evidence_minutes=500,
                                 evidence_apps=10)
        assert 462424 in set(out["Player_Code"])


class TestTheNotPlayingGate:

    def test_a_season_long_absentee_qualifies(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert 462424 in set(out["Player_Code"])

    def test_a_regular_who_is_now_out_qualifies(self):
        """The leg that did not exist, and could not have.

        Rung 1 needs four starts while a 0.25 season-long rate allows two, so
        the two conditions were mutually exclusive until GW17 and rung 1
        contributed nothing: measured, 15 of 15 live rows came from last season.
        """
        pool = _pool([dict(Player_Code=15, Player_ID=15, starts=6,
                           total_points=37, status="i",
                           news="Knee injury - Unknown return date")]
                     + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW, min_pctile=0.0)
        row = out[out["Player_Code"] == 15].iloc[0]
        assert row["Pedigree_Rung"] == RUNG_THIS
        assert row["Band"] == BAND_WATCH

    def test_a_merely_doubtful_regular_does_not(self):
        """`OUT_OF_SQUAD_STATUSES` excludes `d` for exactly this reason.

        A player who has started every gameweek and carries a 75% chance is
        fully visible to every score the app already has; admitting him turns
        this board into a list of the week's knocks. Measured, the
        `chance < 100` version admitted 17 players of whom 13 were that.
        """
        pool = _pool([dict(Player_Code=16, Player_ID=16, starts=ELAPSED,
                           total_points=50, status="d",
                           chance_of_playing_next_round=75,
                           news="Calf injury - 75% chance of playing")]
                     + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW, min_pctile=0.0)
        assert 16 not in set(out["Player_Code"])

    def test_an_ordinary_fit_starter_never_appears(self):
        pool = _pool(_starters(rate=9.0))
        out = build_return_watch(pool, _states(pool), current_gw=GW, min_pctile=0.0)
        assert out.empty


class TestClaimability:

    @pytest.mark.parametrize("state,expected", [("a", True), ("l", True),
                                                ("o", False), (None, False)])
    def test_only_claimable_states_reach_the_board(self, state, expected):
        pool = _pool([SALIBA] + _starters())
        states = _states(pool)
        states[462424] = state
        out = build_return_watch(pool, states, current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert (462424 in set(out["Player_Code"])) is expected

    def test_a_locked_player_is_kept_and_flagged(self):
        """Locked means claimable at the *next* waiver -- the round this is for."""
        pool = _pool([SALIBA] + _starters())
        states = _states(pool)
        states[462424] = "l"
        out = build_return_watch(pool, states, current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert out[out["Player_Code"] == 462424].iloc[0]["Draft_State"] == "l"

    def test_no_ownership_at_all_is_an_empty_board_with_a_reason(self):
        pool = _pool([SALIBA] + _starters())
        notes = []
        out = build_return_watch(pool, {}, current_gw=GW,
                                 prior_rates=SALIBA_PRIOR, notes=notes)
        assert out.empty
        assert any("ownership" in m for _s, m in notes)

    def test_a_pool_with_no_code_cannot_resolve_ownership(self):
        pool = _pool([SALIBA] + _starters()).drop(columns=["Player_Code"])
        notes = []
        out = build_return_watch(pool, {1: "a"}, current_gw=GW, notes=notes)
        assert out.empty
        assert any("player code" in m for _s, m in notes)


class TestThereMustBeAnAbsenceToReturnFrom:
    """Reported from the app: Nick Pope sat on this board with 0 minutes while
    perfectly fit -- benched behind Hornicek, with no injury to come back from.

    The band read "FPL has not flagged him as doubtful", which is true of every
    benched squad player in the game. What it should read is "FPL published news
    about him and has since cleared it", which is an explicit
    ``chance_of_playing`` of 100.

    Measured live, the field separates the two cohorts exactly: of 479 fit
    players, 183 carry an explicit 100 and **all 183** also carry a
    ``news_added`` stamp; 296 carry None and **none of them** does.
    """

    @staticmethod
    def _run(**kw):
        row = dict(Player_Code=462424, Player_ID=6, Player="Test Player",
                   Team="ARS", Position="D", starts=0, minutes=0, total_points=0,
                   status="a", chance_of_playing_next_round=100, news="")
        row.update(kw)
        prior = pd.DataFrame([dict(Player_Code=462424, Prior_PPS=137 / 30,
                                   Prior_Role=30 / 38, Prior_Season="2025/26",
                                   Prior_Starts=30, Prior_Points=137,
                                   Appearances=0)])
        pool = _pool([row] + _starters())
        return build_return_watch(pool, _states(pool), current_gw=GW,
                                  prior_rates=prior, min_pctile=0.0)

    def test_a_never_flagged_player_is_not_returning_from_anything(self):
        """Pope: fit, 0 minutes, and `chance_of_playing` is None.

        None is FPL never having said a word about him. Reading it as "cleared"
        turns "nobody mentioned him" into "he has recovered".
        """
        out = self._run(chance_of_playing_next_round=None)
        assert 462424 not in set(out["Player_Code"])

    def test_an_explicitly_cleared_player_is(self):
        """Saliba: the same 0 minutes, but FPL states 100 outright."""
        out = self._run(chance_of_playing_next_round=100)
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Band"] == BAND_CLEARED

    def test_a_currently_injured_player_is(self):
        out = self._run(status="i", chance_of_playing_next_round=0,
                        news="Knee injury - Unknown return date")
        assert 462424 in set(out["Player_Code"])

    def test_a_cleared_flag_needs_the_news_to_be_gone(self):
        """A 100 alongside live news text is not a clearance."""
        out = self._run(chance_of_playing_next_round=100,
                        news="Knee injury - 100% chance of playing")
        assert 462424 not in set(out["Player_Code"])

    def test_the_band_and_the_gate_read_the_same_flag(self):
        """They cannot disagree about who has been cleared.

        Two independent spellings of one test is how a band comes to say
        `cleared` about a player the gate admitted for a different reason.
        """
        out = self._run(chance_of_playing_next_round=100)
        assert out[out["Player_Code"] == 462424].iloc[0]["Band"] == BAND_CLEARED
        assert self._run(chance_of_playing_next_round=None).empty


class TestDeparturesAreExcluded:
    """7 of 18 raw live candidate rows had left the league."""

    @pytest.mark.parametrize("news", [
        "Has joined Al Hilal permanently",
        "Has joined Juventus on loan for the rest of the season",
        "Has joined Lincoln City on loan until January",
        "not included in squad.",
        "Has left the club",
    ])
    def test_a_departed_player_never_reaches_the_board(self, news):
        pool = _pool([dict(SALIBA, news=news, status="u",
                           chance_of_playing_next_round=0)] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert 462424 not in set(out["Player_Code"])

    def test_an_injury_is_not_a_departure(self):
        pool = _pool([dict(SALIBA, news="Groin injury - Expected back 21 Nov",
                           status="i", chance_of_playing_next_round=0)]
                     + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert 462424 in set(out["Player_Code"])


class TestBands:

    def test_cleared_is_fpl_rating_him_fit_while_he_has_not_featured(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert out[out["Player_Code"] == 462424].iloc[0]["Band"] == BAND_CLEARED

    def test_imminent_needs_a_date_the_news_actually_states(self):
        pool = _pool([dict(SALIBA, status="i", chance_of_playing_next_round=0,
                           news="Groin injury - Expected back 21 Nov")]
                     + _starters())
        deadlines = [pd.Timestamp("2026-11-21T18:00:00Z"),
                     pd.Timestamp("2026-11-28T18:00:00Z")]
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR, deadlines=deadlines,
                                 now=pd.Timestamp("2026-11-15").to_pydatetime())
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Band"] == BAND_IMMINENT
        assert row["Basis"] == "stated"
        assert row["Return_GW"] >= GW

    def test_an_unknown_return_is_watch_and_asserts_no_date(self):
        """`estimate_games_to_miss`' chance and status rungs infer a duration
        from a code, so "Unknown return date" must not become a gameweek."""
        pool = _pool([dict(SALIBA, status="i", chance_of_playing_next_round=0,
                           news="Knee injury - Unknown return date")]
                     + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        row = out[out["Player_Code"] == 462424].iloc[0]
        assert row["Band"] == BAND_WATCH
        assert pd.isna(row["Return_GW"])
        assert row["Basis"] == "no stated date"

    def test_a_return_beyond_the_horizon_is_not_imminent(self):
        pool = _pool([dict(SALIBA, status="i", chance_of_playing_next_round=0,
                           news="ACL - Expected back 20 May")] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR, horizon=3)
        assert out[out["Player_Code"] == 462424].iloc[0]["Band"] == BAND_WATCH


class TestRankingAndTruncation:

    def test_a_more_elite_player_must_be_claimed_earlier(self):
        """"The more elite a player is, the earlier I will need to pick him up."

        Held against an identical return date, so the only thing varying is
        pedigree. Saliba went before he was even scheduled back; a marginal
        player is still there the week he returns, because the rest of the
        league is less sure he walks straight into the side.
        """
        out = self._two_absentees(elite_rate=15.0, marginal_rate=8.0)
        elite = out[out["Player_Code"] == 31].iloc[0]
        marginal = out[out["Player_Code"] == 32].iloc[0]
        assert elite["Expected_Return_GW"] == marginal["Expected_Return_GW"]
        assert elite["Claim_Lead"] > marginal["Claim_Lead"]
        assert elite["Claim_By_GW"] < marginal["Claim_By_GW"]
        assert out.iloc[0]["Player_Code"] == 31

    def test_eliteness_shifts_urgency_rather_than_dominating_it(self):
        """A marginal player back *much* sooner still leads, and should.

        The lead time moves an elite player up the order; it does not suspend
        the calendar. Pinned because the opposite -- pedigree overriding the
        return date outright -- would put a man who is out for months above one
        who plays this weekend.
        """
        out = self._two_absentees(elite_rate=15.0, marginal_rate=8.0,
                                  marginal_news="Groin injury - Expected back 21 Nov",
                                  deadlines=[pd.Timestamp("2026-11-21T18:00:00Z")],
                                  now=pd.Timestamp("2026-11-16").to_pydatetime())
        marginal = out[out["Player_Code"] == 32].iloc[0]
        elite = out[out["Player_Code"] == 31].iloc[0]
        assert marginal["Expected_Return_GW"] < elite["Expected_Return_GW"]
        assert out.iloc[0]["Player_Code"] == 32

    @staticmethod
    def _two_absentees(*, elite_rate, marginal_rate,
                       marginal_news="Knee injury - Unknown return date",
                       deadlines=None, now=None):
        """Two out-injured players against a reference with a real spread.

        The spread matters: against a flat reference every candidate percentiles
        to 1.0, the lead times tie, and a test of this mechanism silently
        exercises nothing.
        """
        elite = dict(Player_Code=31, Player_ID=31, status="i",
                     chance_of_playing_next_round=0,
                     news="Knee injury - Unknown return date")
        marginal = dict(Player_Code=32, Player_ID=32, status="i",
                        chance_of_playing_next_round=0, news=marginal_news)
        prior = pd.DataFrame([
            dict(Player_Code=31, Prior_PPS=elite_rate, Prior_Role=0.95,
                 Prior_Season="2025/26", Prior_Starts=36,
                 Prior_Points=elite_rate * 36, Appearances=0),
            dict(Player_Code=32, Prior_PPS=marginal_rate, Prior_Role=0.75,
                 Prior_Season="2025/26", Prior_Starts=28,
                 Prior_Points=marginal_rate * 28, Appearances=0),
        ])
        spread = [dict(Player_Code=5000 + i, Player_ID=900 + i,
                       Player="Starter %d" % i, Team="ARS", Position="D",
                       starts=ELAPSED, minutes=90 * ELAPSED,
                       total_points=(i + 1.0) * ELAPSED)
                  for i in range(14)]
        pool = _pool([elite, marginal] + spread)
        return build_return_watch(pool, _states(pool), current_gw=GW,
                                  prior_rates=prior, min_pctile=0.0,
                                  claim_lead=4, deadlines=deadlines,
                                  now=now).reset_index(drop=True)

    def test_the_lead_time_scales_with_pedigree(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR, claim_lead=4)
        row = out.iloc[0]
        assert row["Claim_Lead"] == round(4 * float(row["Pedigree"]))
        assert row["Claim_By_GW"] == row["Expected_Return_GW"] - row["Claim_Lead"]

    def test_being_already_late_is_said_rather_than_clamped(self):
        """A negative margin is a different thing to say than "claim now"."""
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR, claim_lead=4)
        row = out.iloc[0]
        assert row["Claim_By_GW"] < GW
        assert "overdue" in claim_phrase(row, GW)

    def test_an_inferred_return_is_never_reported_as_a_stated_one(self):
        """`Return_GW` is a claim; `Expected_Return_GW` is an estimate.

        The second falls back to `absence_estimate`'s chance and status buckets,
        which are a guess -- fine to sort by, never to render as a date.
        """
        pool = _pool([dict(SALIBA, status="i", chance_of_playing_next_round=0,
                           news="Knee injury - Unknown return date")]
                     + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        row = out.iloc[0]
        assert pd.isna(row["Return_GW"])
        assert pd.notna(row["Expected_Return_GW"])
        assert "no return date" in return_phrase(row)

    def test_a_cleared_player_reads_as_available_now(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert return_phrase(out.iloc[0]) == "available now"

    def test_it_ranks_by_pedigree_and_truncates(self):
        rows = [dict(Player_Code=100 + i, Player_ID=100 + i) for i in range(6)]
        prior = pd.DataFrame([
            dict(Player_Code=100 + i, Prior_PPS=3.0 + i * 0.5, Prior_Role=0.9,
                 Prior_Season="2025/26", Prior_Starts=34, Prior_Points=100)
            for i in range(6)])
        pool = _pool(rows + _starters(rate=2.0))
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior, limit=3, min_pctile=0.0)
        assert len(out) == 3
        assert out["Pedigree_Rate"].tolist() == sorted(
            out["Pedigree_Rate"].tolist(), reverse=True)
        assert out.iloc[0]["Pedigree_Rate"] == pytest.approx(5.5)

    def test_the_floor_drops_the_tail(self):
        """Harwood-Bellis sat at 0.07 on the live board."""
        pool = _pool([dict(Player_Code=21, Player_ID=21)] + _starters(rate=5.0))
        prior = pd.DataFrame([dict(Player_Code=21, Prior_PPS=1.72, Prior_Role=0.84,
                                   Prior_Season="2025/26", Prior_Starts=32,
                                   Prior_Points=55)])
        notes = []
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior, min_pctile=0.50, notes=notes)
        assert out.empty
        assert any("median starter" in m for _s, m in notes)


class TestAbstention:
    """An all-clear must be derived, never defaulted."""

    @pytest.mark.parametrize("gw", [1, 2])
    def test_it_abstains_early_in_the_season(self, gw):
        """At GW1 every player has nothing played, so "not playing" says nothing."""
        pool = _pool([SALIBA] + _starters())
        notes = []
        out = build_return_watch(pool, _states(pool), current_gw=gw,
                                 prior_rates=SALIBA_PRIOR, notes=notes,
                                 min_gws=3)
        assert out.empty
        assert any("Too early" in m for _s, m in notes)

    def test_an_empty_pool_says_so(self):
        notes = []
        out = build_return_watch(pd.DataFrame(), {1: "a"}, current_gw=GW, notes=notes)
        assert out.empty
        assert any("pool" in m for _s, m in notes)

    def test_no_gameweek_says_so(self):
        pool = _pool([SALIBA] + _starters())
        notes = []
        out = build_return_watch(pool, _states(pool), current_gw=None, notes=notes)
        assert out.empty
        assert notes

    def test_missing_prior_history_is_reported_not_silent(self):
        pool = _pool([SALIBA] + _starters())
        notes = []
        build_return_watch(pool, _states(pool), current_gw=GW, prior_rates=None,
                           notes=notes)
        assert any("prior-season" in m for _s, m in notes)

    def test_every_empty_return_still_declares_the_columns(self):
        out = build_return_watch(None, {}, current_gw=GW)
        assert list(out.columns) == WATCH_COLUMNS


class TestItValidatesItsOwnOutput:

    def test_a_real_board_passes_its_own_validation(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        errors = [i for i in check_return_watch(out, current_gw=GW)
                  if i.severity == "error"]
        assert not errors, errors

    def test_config_defaults_are_used_when_nothing_is_passed(self):
        pool = _pool([SALIBA] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        assert len(out) <= config.RETURN_WATCH_LIMIT


class TestAClearedPlayerWhoIsOnlyGettingCameos:
    """Reported from the app: "Minteh is not an elite player and is back healthy
    now and only got 8 minutes in the last game. I'm not confident he'll break
    into the XI for Brighton when they've been doing well without him."

    An appearance that is not a start is counter-evidence against the one thing
    this board claims -- that he walks back into the XI. One cameo is weak, so
    it raises the pedigree bar rather than disqualifying outright.
    """

    @staticmethod
    def _run(*, apps, starts, minutes, pedigree_rate, status="a", news="",
             chance=100, benched_pctile=0.85):
        cand = dict(Player="Candidate", Player_Code=777, Player_ID=777,
                    Team="ARS", Position="M", starts=starts, minutes=minutes,
                    total_points=0, status=status, news=news,
                    chance_of_playing_next_round=chance)
        prior = pd.DataFrame([dict(Player_Code=777, Prior_PPS=pedigree_rate,
                                   Prior_Role=26 / 38, Prior_Season="2025/26",
                                   Prior_Starts=26, Prior_Points=117,
                                   Appearances=apps)])
        # A reference with a spread, so the pedigree percentile is meaningful.
        spread = [dict(Player_Code=5000 + i, Player_ID=900 + i,
                       Player="Starter %d" % i, Team="ARS", Position="M",
                       starts=ELAPSED, minutes=90 * ELAPSED,
                       total_points=(i + 1.0) * ELAPSED) for i in range(14)]
        pool = _pool([cand] + spread)
        notes = []
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=prior, min_pctile=0.0,
                                 benched_pctile=benched_pctile, notes=notes)
        return out, notes

    def test_a_sub_cameo_drops_a_non_elite_player(self):
        """Minteh: fit, one appearance, no starts, 8 minutes."""
        out, notes = self._run(apps=1, starts=0, minutes=8, pedigree_rate=4.5)
        assert 777 not in set(out["Player_Code"])
        assert any("without starting" in m for _s, m in notes), \
            "the exclusion must say why -- a silently shorter list is not an answer"

    def test_no_appearances_is_not_counter_evidence(self):
        """Saliba: fit, zero appearances.

        A man FPL never flagged and a man who is simply unused look identical
        from here, so absence of appearances can never be read as a benching --
        it is the case the whole feature exists for.
        """
        out, _notes = self._run(apps=0, starts=0, minutes=0, pedigree_rate=4.5)
        assert 777 in set(out["Player_Code"])

    def test_an_elite_player_gets_the_benefit_of_the_doubt(self):
        """One cameo is weak evidence, so it raises the bar rather than
        disqualifying."""
        out, _notes = self._run(apps=1, starts=0, minutes=8, pedigree_rate=20.0)
        row = out[out["Player_Code"] == 777]
        assert len(row) == 1
        assert float(row.iloc[0]["Pedigree"]) >= 0.85

    def test_a_player_still_out_is_unaffected(self):
        """His appearances predate the absence and say nothing about selection
        after he returns."""
        out, _notes = self._run(apps=2, starts=0, minutes=15, pedigree_rate=4.5,
                                status="i", chance=0,
                                news="Knee injury - Unknown return date")
        assert 777 in set(out["Player_Code"])

    def test_a_cleared_player_who_has_started_is_unaffected(self):
        """One appearance and it was a start -- nothing counter-evidential.

        Kept deliberately below the current-season evidence thresholds (90
        minutes / 3 appearances): past them the earlier "this season decides"
        rule takes over and a man who has started 2 of 10 while fit is a
        Senesi-shaped exclusion, which is a different rule doing its job.
        """
        out, _notes = self._run(apps=1, starts=1, minutes=60, pedigree_rate=4.5)
        assert 777 in set(out["Player_Code"])


class TestNoFabricatedReturnGameweek:
    """`Expected_Return_GW` falls back to `absence_estimate`'s chance and status
    buckets -- status `i` alone yields four gameweeks and a zero chance yields
    five, on no evidence about this player at all.

    Reported from the app: "How are we saying ~GW16 when it also says no stated
    date?" Fair -- it stated a gameweek and disclaimed it in the same breath.
    """

    @staticmethod
    def _row(news, status="i", chance=0):
        pool = _pool([dict(SALIBA, status=status, news=news,
                           chance_of_playing_next_round=chance)] + _starters())
        out = build_return_watch(pool, _states(pool), current_gw=GW,
                                 prior_rates=SALIBA_PRIOR)
        return out.iloc[0]

    def test_an_inferred_return_renders_no_gameweek(self):
        phrase = return_phrase(self._row("Knee injury - Unknown return date"))
        assert "GW" not in phrase
        assert "no return date" in phrase

    def test_the_estimate_still_exists_for_ordering(self):
        """Fit to sort by, never to render -- the ordering must still work."""
        row = self._row("Knee injury - Unknown return date")
        assert pd.notna(row["Expected_Return_GW"])
        assert pd.notna(row["Claim_By_GW"])
        assert pd.isna(row["Return_GW"])

    def test_a_stated_return_does_render_its_gameweek(self):
        pool = _pool([dict(SALIBA, status="i", chance_of_playing_next_round=0,
                           news="Groin injury - Expected back 21 Nov")]
                     + _starters())
        out = build_return_watch(
            pool, _states(pool), current_gw=GW, prior_rates=SALIBA_PRIOR,
            deadlines=[pd.Timestamp("2026-11-21T18:00:00Z"),
                       pd.Timestamp("2026-11-28T18:00:00Z")],
            now=pd.Timestamp("2026-11-16").to_pydatetime())
        assert "GW" in return_phrase(out.iloc[0])
