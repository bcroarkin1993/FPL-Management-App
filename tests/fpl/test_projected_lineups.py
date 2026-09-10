"""Tests for the Projected Lineups gameweek filter.

Rotowire's lineups page lists whatever matches it has lineups for, which runs
past the current gameweek. Observed 2026-09-10: eleven matchups, ten from GW4
and one -- "Brentford vs Chelsea", 18 September -- from GW5, rendered under a
GW4 heading as though Brentford were playing Chelsea that week. They were
playing Bournemouth.
"""

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
