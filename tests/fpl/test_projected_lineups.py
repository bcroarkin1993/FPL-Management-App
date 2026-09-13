"""Tests for the Projected Lineups gameweek filter.

Rotowire's lineups page lists whatever matches it has lineups for, which runs
past the current gameweek. Observed 2026-09-10: eleven matchups, ten from GW4
and one -- "Brentford vs Chelsea", 18 September -- from GW5, rendered under a
GW4 heading as though Brentford were playing Chelsea that week. They were
playing Bournemouth.
"""

import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.common.text_helpers import TEAM_FULL_TO_SHORT
from scripts.fpl import projected_lineups as pl


GW4_PAIRS = {("BOU", "BRE"), ("CHE", "HUL"), ("LIV", "FUL")}


class TestMatchupIsInGameweek:
    def test_a_fixture_from_this_gameweek_is_kept(self):
        assert pl._matchup_is_in_gameweek("AFC Bournemouth", "Brentford", GW4_PAIRS)

    def test_a_fixture_from_a_later_gameweek_is_dropped(self):
        """The reported bug, exactly: Brentford host Chelsea in GW5."""
        assert not pl._matchup_is_in_gameweek("Brentford", "Chelsea", GW4_PAIRS)

    def test_reversed_home_and_away_is_a_different_fixture(self):
        """Chelsea host Hull in GW4; Hull hosting Chelsea is another week."""
        assert not pl._matchup_is_in_gameweek("Hull City", "Chelsea", GW4_PAIRS)

    def test_an_unmapped_club_label_keeps_the_matchup(self):
        """Fails open on purpose. Showing one extra match is an annoyance;
        silently dropping a real one is a functional loss, and club spellings
        are exactly what goes stale when a source changes."""
        assert pl._matchup_is_in_gameweek("Some New Club FC", "Brentford", GW4_PAIRS)

    def test_an_unreadable_fixture_list_disables_the_filter(self):
        assert pl._matchup_is_in_gameweek("Brentford", "Chelsea", set())


class TestRotowireLineupLabelsResolve:
    """An unmapped label here is not cosmetic -- it silently disables the filter
    for that matchup, which is how the extra fixture would come back."""

    @pytest.mark.parametrize("label", [
        "AFC Bournemouth", "Brighton & Hove Albion", "Newcastle United",
        "Tottenham Hotspur", "Nottingham Forest", "Coventry City",
        "Hull City", "Ipswich Town", "Leeds United", "Manchester United",
        "Manchester City", "Crystal Palace", "Aston Villa",
    ])
    def test_label_resolves_to_a_short_code(self, label):
        assert TEAM_FULL_TO_SHORT.get(label), f"{label!r} is unmapped"


_HTML = """
<html><body>
  <div class="lineup__time">September 12  10:00 AM ET</div>
  <div class="lineup__mteam is-home">Chelsea</div>
  <div class="lineup__mteam is-visit">Hull City</div>
  <div class="lineup__main"></div>
  <div class="lineup__time">September 18  3:00 PM ET</div>
  <div class="lineup__mteam is-home">Brentford</div>
  <div class="lineup__mteam is-visit">Chelsea</div>
  <div class="lineup__main"></div>
  <div class="lineup__time">September 12  10:00 AM ET</div>
  <div class="lineup__mteam is-home">Liverpool</div>
  <div class="lineup__mteam is-visit">Fulham</div>
  <div class="lineup__main"></div>
</body></html>
"""


class TestScrapeFiltersByGameweek:
    def _scrape(self):
        resp = MagicMock()
        resp.content = _HTML.encode("utf-8")
        # One player per side, so matchup indices are observable.
        def _extract(section, side, team, idx):
            return [(team, "M", f"{team} player", idx)]
        with patch.object(pl.requests, "get", return_value=resp), \
             patch.object(pl, "_gameweek_fixture_pairs", return_value=GW4_PAIRS), \
             patch.object(pl, "extract_players", side_effect=_extract):
            return pl.scrape_rotowire_lineups("https://example.com/lineups", gameweek=4)

    def test_the_later_gameweeks_matchup_is_excluded(self):
        df = self._scrape()
        assert "Brentford" not in set(df["Team"])
        assert set(df["Team"]) == {"Chelsea", "Hull City", "Liverpool", "Fulham"}

    def test_matchup_indices_stay_contiguous(self):
        """The renderer pairs home and away by MatchupIndex. A gap left by a
        filtered match would leave a matchup showing only one side."""
        df = self._scrape()
        assert sorted(df["MatchupIndex"].unique()) == [0, 1]

    def test_passing_zero_disables_the_filter(self):
        resp = MagicMock()
        resp.content = _HTML.encode("utf-8")
        def _extract(section, side, team, idx):
            return [(team, "M", f"{team} player", idx)]
        with patch.object(pl.requests, "get", return_value=resp), \
             patch.object(pl, "extract_players", side_effect=_extract):
            df = pl.scrape_rotowire_lineups("https://example.com/lineups", gameweek=0)
        assert sorted(df["MatchupIndex"].unique()) == [0, 1, 2]


class TestMatchupsShareTheIndexSpace:
    """The matchup list and the player frame must be one filtered pass.

    They were two functions doing two separate fetches and only one of them
    filtered, so the dropdown carried a next-gameweek fixture ("Brentford v
    Chelsea" under a GW4 heading) *and* the two halves were numbered
    differently: players 0..N-1 after filtering, matchups 0..N before it.

    Note where the stray fixture sits in ``_HTML`` -- second of three. That is
    the case that matters. Live, it happened to sort last, so the two index
    spaces coincided and the only symptom was one empty extra card. One
    position earlier and every subsequent matchup renders another club's
    players, with every name on screen still perfectly plausible.
    """

    def _scrape(self, gameweek=4):
        resp = MagicMock()
        resp.content = _HTML.encode("utf-8")

        def _extract(section, side, team, idx):
            return [(team, "M", f"{team} player", idx)]

        with patch.object(pl.requests, "get", return_value=resp) as get, \
             patch.object(pl, "_gameweek_fixture_pairs", return_value=GW4_PAIRS), \
             patch.object(pl, "extract_players", side_effect=_extract):
            return pl.scrape_lineups("https://example.com/lineups", gameweek), get

    def test_matchups_exclude_the_later_gameweek(self):
        scrape, _ = self._scrape()
        pairs = [(h, a) for h, a, _ in scrape.matchups]
        assert ("Brentford", "Chelsea") not in pairs
        assert pairs == [("Chelsea", "Hull City"), ("Liverpool", "Fulham")]

    def test_matchup_indices_are_contiguous(self):
        scrape, _ = self._scrape()
        assert [i for _, _, i in scrape.matchups] == [0, 1]

    def test_every_index_holds_exactly_its_own_two_clubs(self):
        """The assertion that would have caught the desync: for each matchup,
        the players filed under its index are that fixture's two clubs."""
        scrape, _ = self._scrape()
        for home, away, idx in scrape.matchups:
            teams = set(scrape.players.loc[
                scrape.players["MatchupIndex"] == idx, "Team"])
            assert teams == {home, away}, (
                "index %d is labelled %s v %s but holds %s"
                % (idx, home, away, sorted(teams)))

    def test_the_page_is_fetched_once(self):
        """Two fetches of the same page is not just waste -- Rotowire can
        publish between them, which is a second way the halves can disagree."""
        _, get = self._scrape()
        assert get.call_count == 1

    def test_scrape_matchups_filters_by_gameweek(self):
        """It had no gameweek parameter at all, which is how the stray fixture
        reached the dropdown."""
        resp = MagicMock()
        resp.content = _HTML.encode("utf-8")
        with patch.object(pl.requests, "get", return_value=resp), \
             patch.object(pl, "_gameweek_fixture_pairs", return_value=GW4_PAIRS), \
             patch.object(pl, "extract_players", side_effect=lambda *a: []):
            matchups = pl.scrape_matchups("https://example.com/lineups", gameweek=4)
        assert [(h, a) for h, a, _ in matchups] == [
            ("Chelsea", "Hull City"), ("Liverpool", "Fulham")]

    def test_the_two_wrappers_agree_with_the_single_pass(self):
        scrape, _ = self._scrape()
        resp = MagicMock()
        resp.content = _HTML.encode("utf-8")

        def _extract(section, side, team, idx):
            return [(team, "M", f"{team} player", idx)]

        with patch.object(pl.requests, "get", return_value=resp), \
             patch.object(pl, "_gameweek_fixture_pairs", return_value=GW4_PAIRS), \
             patch.object(pl, "extract_players", side_effect=_extract):
            df = pl.scrape_rotowire_lineups("https://example.com/lineups", gameweek=4)
            matchups = pl.scrape_matchups("https://example.com/lineups", gameweek=4)
        assert df.equals(scrape.players)
        assert matchups == scrape.matchups

    def test_a_failed_fetch_returns_both_halves_empty(self):
        with patch.object(pl.requests, "get", side_effect=RuntimeError("down")):
            scrape = pl.scrape_lineups("https://example.com/lineups", gameweek=4)
        assert scrape.players.empty
        assert list(scrape.players.columns) == pl.LINEUP_COLUMNS
        assert scrape.matchups == []


# ---------------------------------------------------------------------------
# Player stats lookup
# ---------------------------------------------------------------------------

def _element(pid, first, second, web, team, etype, **over):
    e = {"id": pid, "first_name": first, "second_name": second, "web_name": web,
         "team": team, "element_type": etype, "form": "1.0", "points_per_game": "1.0",
         "total_points": 10, "minutes": 90, "starts": 1, "goals_scored": 0,
         "assists": 0, "clean_sheets": 0, "chance_of_playing_this_round": None,
         "status": "a", "news": ""}
    e.update(over)
    return e


# Team ids: 1 CHE, 2 IPS, 3 MUN, 4 LEE
_BOOTSTRAP = {
    "teams": [{"id": 1, "short_name": "CHE"}, {"id": 2, "short_name": "IPS"},
              {"id": 3, "short_name": "MUN"}, {"id": 4, "short_name": "LEE"}],
    "elements": [
        # The collision this whole rewrite exists for: an elite midfielder and a
        # backup keeper at different clubs, sharing a surname.
        _element(1, "Cole", "Palmer", "Palmer", 1, 3, form="6.5", total_points=26),
        _element(2, "Alex", "Palmer", "Palmer", 2, 1, form="0.0", total_points=0),
        # Registered a midfielder, listed by Rotowire as a forward.
        _element(3, "Matheus", "Santos Carneiro da Cunha", "M.Cunha", 3, 3),
        # Two Wilsons at one club. FPL disambiguates them by web_name, which
        # is what the exact-web_name tier is for.
        _element(4, "Harry", "Wilson", "Wilson", 4, 3),
        _element(5, "Ben", "Wilson", "B.Wilson", 4, 2),
        # Two Fergusons at one club whose web_names are *both* initialled, so
        # a bare "Ferguson" matches neither exactly and reaches the last-word
        # tier, where it is genuinely ambiguous.
        _element(6, "Evan", "Ferguson", "E.Ferguson", 4, 3),
        _element(7, "Lewis", "Ferguson", "L.Ferguson", 4, 3),
        # Same club, same surname, *different* registered positions -- here the
        # position scoping is what separates them.
        _element(9, "Tom", "Doyle", "T.Doyle", 4, 2),
        _element(10, "Sam", "Doyle", "S.Doyle", 4, 4),
        _element(8, "Joao", "Palhinha", "Palhinha", 1, 3),
    ],
}

_AVAIL = pd.DataFrame([
    {"Player_ID": 1, "PlayPct": 100.0, "StatusBucket": "Available", "News": ""},
    {"Player_ID": 2, "PlayPct": 0.0, "StatusBucket": "Out", "News": "Knee injury"},
])


@pytest.fixture
def index():
    with patch.object(pl, "get_classic_bootstrap_static", return_value=_BOOTSTRAP), \
         patch.object(pl, "get_fpl_availability_df", return_value=_AVAIL):
        return pl.build_player_index()


class TestPlayerIndexLookup:
    """The matcher this replaced was a six-stage ladder, every stage of which
    was team- and position-agnostic, over a dict that also keyed players by
    bare surname and by web_name. Measured live: 24 surnames and 17 web_names
    were ambiguous league-wide (51 and 36 players), and a plain dict keeps
    whichever the bootstrap happened to list last.
    """

    def test_a_shared_surname_resolves_by_club(self, index):
        """Cole Palmer's form must not appear on Alex Palmer's card."""
        cole = index.lookup("Palmer", "Chelsea", "AMC")
        alex = index.lookup("Palmer", "Ipswich Town", "GK")
        assert cole["team"] == "CHE" and cole["total_points"] == 26
        assert alex["team"] == "IPS" and alex["total_points"] == 0

    def test_availability_rides_along_with_the_right_player(self, index):
        assert index.lookup("Palmer", "Ipswich Town", "GK")["status_bucket"] == "Out"
        assert index.lookup("Palmer", "Chelsea", "AMC")["status_bucket"] == "Available"

    def test_rotowire_role_may_disagree_with_the_registered_position(self, index):
        """Rotowire publishes a tactical role, FPL a registered position. Five
        of 66 starters differed in one gameweek -- wing-backs listed in
        midfield, Cunha listed as a forward. Position is a hint, not a filter."""
        assert index.lookup("Matheus Cunha", "Manchester United", "FW")["team"] == "MUN"

    def test_a_web_name_disambiguates_same_club_namesakes(self, index):
        """Two Wilsons at Leeds, and FPL names one of them "Wilson" precisely
        to tell them apart. The exact-web_name tier should take it."""
        assert index.lookup("Wilson", "Leeds United", "MC")["team"] == "LEE"

    def test_a_genuinely_ambiguous_name_resolves_to_nothing(self, index):
        """Two Fergusons at Leeds, neither of whom FPL calls plain "Ferguson",
        so the query reaches the last-word tier and matches both. A coin flip
        is worse than a blank card."""
        assert index.lookup("Ferguson", "Leeds United", "MC") == {}

    def test_ambiguity_survives_the_all_positions_retry(self, index):
        """The retry widens the search; it must not turn a tie into a winner.
        Both Fergusons are midfielders, so no position separates them and every
        pass must come back empty."""
        for code in ("MC", "FW", "GK", "DC"):
            assert index.lookup("Ferguson", "Leeds United", code) == {}, code

    def test_position_separates_same_club_namesakes(self, index):
        """Two Doyles at Leeds, one a defender and one a forward. Scoping by
        position is what makes each resolvable at all."""
        assert index.lookup("Doyle", "Leeds United", "DC")["team"] == "LEE"
        assert index.lookup("Doyle", "Leeds United", "FW")["team"] == "LEE"
        # ...and they must be different players, not the same one twice.
        assert (index.lookup("Doyle", "Leeds United", "DC")
                is not index.lookup("Doyle", "Leeds United", "FW"))

    def test_an_abbreviated_initial_with_no_space_still_resolves(self, index):
        """Rotowire writes "J.Palhinha"; canonical_normalize deletes the dot
        rather than splitting on it, collapsing the name to one meaningless
        token unless the space is restored first."""
        assert index.lookup("J.Palhinha", "Chelsea", "MC")["team"] == "CHE"

    def test_an_unmapped_club_fails_closed(self, index):
        """Without a club only whole-name tiers are safe, and a card showing
        the wrong player is worse than one showing no stats."""
        assert index.lookup("Cole Palmer", "Chelsea FC Football Club", "AMC") == {}

    def test_a_player_absent_from_fpl_returns_nothing(self, index):
        """Rotowire lists players who have left the league. They must come back
        empty, not attach to the nearest surname."""
        assert index.lookup("F. Kadioglu", "Chelsea", "DR") == {}

    def test_lookup_is_scoped_even_for_a_unique_surname(self, index):
        """Cunha exists only at MUN, but asking for him at Chelsea must still
        miss -- uniqueness in the pool is not a licence to cross clubs."""
        assert index.lookup("Matheus Cunha", "Chelsea", "FW") == {}


class TestPlayerIndexDegradation:
    def test_an_empty_bootstrap_yields_the_empty_index(self):
        with patch.object(pl, "get_classic_bootstrap_static", return_value={}):
            assert pl.build_player_index() is pl.EMPTY_PLAYER_INDEX

    def test_a_failing_bootstrap_does_not_raise(self):
        with patch.object(pl, "get_classic_bootstrap_static",
                          side_effect=RuntimeError("FPL down")):
            assert pl.build_player_index() is pl.EMPTY_PLAYER_INDEX

    def test_the_empty_index_looks_up_to_nothing(self):
        assert pl.EMPTY_PLAYER_INDEX.lookup("Cole Palmer", "Chelsea", "AMC") == {}

    def test_missing_availability_still_builds_the_index(self):
        """Availability is an enhancement; losing it must not lose the stats."""
        with patch.object(pl, "get_classic_bootstrap_static", return_value=_BOOTSTRAP), \
             patch.object(pl, "get_fpl_availability_df",
                          side_effect=RuntimeError("bootstrap down")):
            idx = pl.build_player_index()
        assert idx.lookup("Palmer", "Chelsea", "AMC")["total_points"] == 26


class TestTacticalPositionMap:
    def test_every_code_the_field_can_draw_is_mapped(self):
        """plot_soccer_field's position_mapping is the set of codes Rotowire
        publishes. One missing here silently drops position scoping for it."""
        import inspect
        source = inspect.getsource(pl.plot_soccer_field)
        drawn = set(re.findall(r"'([A-Z]{2,3})':\s*\(", source))
        assert drawn, "could not read position_mapping out of plot_soccer_field"
        missing = drawn - set(pl.ROTOWIRE_TACTICAL_TO_POSITION)
        assert not missing, "unmapped tactical codes: %s" % sorted(missing)

    def test_codes_map_to_the_apps_position_scheme(self):
        assert set(pl.ROTOWIRE_TACTICAL_TO_POSITION.values()) <= {"G", "D", "M", "F"}
