"""Unit tests for the bookmaker odds model.

Every fixture is a real number the live feed actually published on 2026-09-04,
because the failures worth catching here are not ones you invent: the 157.6%
ladder, the ``bestOdds: "2"`` that means 2/1, and the five-month-old timestamp
that got this source rejected the first time.
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts.common.data_validation import check_transfer_odds, raise_on_error
from scripts.common.transfer_odds import (age_band, blend_odds_risk,
                                          classify_market, disjoint_ladder,
                                          exit_probability, group_ladder,
                                          implied_probability, ladder_overround,
                                          normalise_ladder, odds_age_days,
                                          odds_age_weight, parse_fractional)

#: Mohamed Salah's live ladder, verbatim.
SALAH_LADDER = [
    {"Destination": "Any Saudi club", "Fractional": "8/11"},
    {"Destination": "Al Ittihad", "Fractional": "7/4"},
    {"Destination": "Any MLS Team", "Fractional": "5/2"},
    {"Destination": "Al Hilal", "Fractional": "7/1"},
    {"Destination": "Any French club", "Fractional": "8/1"},
    {"Destination": "Any Italian club", "Fractional": "8/1"},
]

SALAH_UPDATED = "2026-03-25T19:56:27.509+00:00"
NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


class TestParseFractional:
    def test_live_prices(self):
        assert parse_fractional("8/11") == pytest.approx(1.7272, abs=1e-3)
        assert parse_fractional("1/2") == 1.5
        assert parse_fractional("7/4") == 2.75

    def test_bare_numerator_means_to_one(self):
        """The feed publishes bestOdds "2" alongside decimal 3, so "2" is 2/1.

        Reading it as decimal 2 would price a 25% shot at 50%.
        """
        assert parse_fractional("2") == 3.0

    def test_evens(self):
        assert parse_fractional("evens") == 2.0
        assert parse_fractional("1/1") == 2.0

    def test_junk_is_none_not_zero(self):
        for bad in (None, "", "garbage", "8/0", "-1/2"):
            assert parse_fractional(bad) is None

    def test_implied_probability_matches_the_published_column(self):
        # The site prints these percentages beside the prices; ours must agree.
        assert implied_probability(parse_fractional("8/11")) == pytest.approx(0.579, abs=1e-3)
        assert implied_probability(parse_fractional("7/4")) == pytest.approx(0.364, abs=1e-3)
        assert implied_probability(parse_fractional("8/1")) == pytest.approx(0.111, abs=1e-3)

    def test_implied_rejects_impossible_odds(self):
        assert implied_probability(0.5) is None
        assert implied_probability("nonsense") is None


class TestOverlap:
    """The bug this module exists to prevent."""

    def test_raw_ladder_is_not_a_distribution(self):
        raw = sum(implied_probability(parse_fractional(r["Fractional"]))
                  for r in SALAH_LADDER)
        assert raw == pytest.approx(1.576, abs=1e-3)

    def test_aggregate_absorbs_its_member_clubs(self):
        kept = [r["Destination"] for r in disjoint_ladder(SALAH_LADDER)]
        assert "Any Saudi club" in kept
        assert "Al Ittihad" not in kept, "Al Ittihad is inside Any Saudi club"
        assert "Al Hilal" not in kept

    def test_collapsed_overround_is_a_believable_book(self):
        assert ladder_overround(SALAH_LADDER) == pytest.approx(1.087, abs=1e-3)

    def test_normalising_the_raw_ladder_would_understate_saudi(self):
        """Dividing by 1.576 reports 37% where the market says 58%."""
        naive = implied_probability(parse_fractional("8/11")) / 1.576
        actual = normalise_ladder(SALAH_LADDER)[0]["Probability"]
        assert naive == pytest.approx(0.367, abs=1e-2)
        assert actual == pytest.approx(0.533, abs=1e-2)
        assert actual > naive

    def test_unresolvable_club_is_kept_not_dropped(self):
        """Unknown vocabulary must cost a visible overround, never a destination."""
        rows = SALAH_LADDER + [{"Destination": "Obscure FC", "Fractional": "20/1"}]
        assert "Obscure FC" in [r["Destination"] for r in disjoint_ladder(rows)]

    def test_classify_market(self):
        assert classify_market("Any Saudi club") == "aggregate"
        assert classify_market("Any MLS Team") == "aggregate"
        assert classify_market("Al Ittihad") == "club"
        assert classify_market("Nottingham Forest") == "club"


class TestDistribution:
    def test_shares_sum_to_one(self):
        assert sum(e["Probability"] for e in normalise_ladder(SALAH_LADDER)) == pytest.approx(1.0)

    def test_ordered_by_probability(self):
        shares = [e["Probability"] for e in normalise_ladder(SALAH_LADDER)]
        assert shares == sorted(shares, reverse=True)

    def test_members_ride_along_without_being_summed(self):
        grouped = group_ladder(SALAH_LADDER)
        saudi = next(e for e in grouped if e["Destination"] == "Any Saudi club")
        members = [m["Destination"] for m in saudi["Members"]]
        assert members == ["Al Ittihad", "Al Hilal"]
        assert sum(e["Probability"] for e in grouped) == pytest.approx(1.0)

    def test_exit_probability_is_the_shortest_price(self):
        assert exit_probability(SALAH_LADDER) == pytest.approx(0.579, abs=1e-3)

    def test_empty_ladder_is_zero_not_an_error(self):
        assert exit_probability([]) == 0.0
        assert normalise_ladder([]) == []


class TestFreshness:
    def test_the_real_stale_quote(self):
        age = odds_age_days(SALAH_UPDATED, NOW)
        assert age == pytest.approx(163, abs=1)
        assert age_band(age) == "archival"
        assert odds_age_weight(SALAH_UPDATED, NOW) < 0.1

    def test_a_fresh_quote_keeps_its_weight(self):
        assert odds_age_weight("2026-09-01T00:00:00+00:00", NOW) > 0.9
        assert age_band(3) == "live"

    def test_missing_timestamp_is_assumed_stale_not_fresh(self):
        """A parse failure must not read as confidence."""
        assert odds_age_days(None, NOW) == 30.0
        assert odds_age_weight(None, NOW) < 0.7


class TestBlending:
    def test_no_odds_returns_news_unchanged(self):
        assert blend_odds_risk(0.62, None, None) == 0.62
        assert blend_odds_risk(0.62, 0.0, 0.9) == 0.62

    def test_stale_odds_barely_move_the_score(self):
        weight = odds_age_weight(SALAH_UPDATED, NOW)
        assert blend_odds_risk(0.20, 0.579, weight) == pytest.approx(0.223, abs=0.01)

    def test_fresh_odds_pull_meaningfully(self):
        assert blend_odds_risk(0.20, 0.579, 0.95) > 0.36

    def test_result_stays_a_probability(self):
        assert 0.0 <= blend_odds_risk(1.0, 1.0, 1.0) <= 1.0
        assert 0.0 <= blend_odds_risk(-5, 5, 5) <= 1.0


class TestValidation:
    def test_correct_pipeline_is_clean(self):
        raise_on_error(check_transfer_odds(
            [dict(r, Implied=implied_probability(parse_fractional(r["Fractional"])))
             for r in SALAH_LADDER],
            normalised=normalise_ladder(SALAH_LADDER),
            overround=ladder_overround(SALAH_LADDER),
            age_days=30,
        ), context="salah ladder")

    def test_naive_overround_is_caught(self):
        """The whole point: summing overlapping rows must fail loudly."""
        rows = [dict(r, Implied=implied_probability(parse_fractional(r["Fractional"])))
                for r in SALAH_LADDER]
        issues = check_transfer_odds(rows, overround=sum(r["Implied"] for r in rows))
        assert any(i.severity == "error" and "disjoint" in i.message for i in issues)

    def test_empty_ladder_errors(self):
        assert any(i.severity == "error" for i in check_transfer_odds([]))

    def test_missing_favourite_shows_as_sub_unity_book(self):
        rows = [dict(r, Implied=implied_probability(parse_fractional(r["Fractional"])))
                for r in SALAH_LADDER[1:]]
        issues = check_transfer_odds(rows, overround=ladder_overround(SALAH_LADDER[1:]))
        assert any(i.severity == "error" and "below 1.0" in i.message for i in issues)

    def test_future_dated_quote_errors(self):
        rows = [dict(r, Implied=0.5) for r in SALAH_LADDER]
        assert any(i.severity == "error" for i in check_transfer_odds(rows, age_days=-3))

    def test_old_quote_warns_but_does_not_error(self):
        rows = [dict(r, Implied=0.5) for r in SALAH_LADDER]
        issues = check_transfer_odds(rows, age_days=163)
        assert issues and all(i.severity == "warning" for i in issues)


class TestOddsNameMatching:
    """Matching the odds feed to the FPL pool — the repo's recurring bug class."""

    @staticmethod
    def _odds(name, next_club="Atletico Madrid"):
        return pd.DataFrame([{
            "Player": name, "Slug": "x", "Next_Club": next_club,
            "Fractional": "6/4", "Decimal": 2.5, "Implied": 0.4,
            "Bookmaker": "William Hill", "Trending": "neutral", "Updated": None,
        }])

    def test_a_departed_player_does_not_donate_his_market_to_a_namesake(self):
        """Darwin Núñez has left the league; Marcelino Núñez has not.

        Uniqueness inside the pool is satisfied — only one Núñez remains — so
        the surname alone resolved, and Marcelino was priced on Darwin's market.
        """
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame({"Player": ["Marcelino Núñez", "Cole Palmer"]})
        assert attach_odds(pool, self._odds("Darwin Nunez")).empty

    def test_full_legal_name_still_matches_a_common_name(self):
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame({"Player": ["Bruno Borges Fernandes", "Cole Palmer"]})
        matched = attach_odds(pool, self._odds("Bruno Fernandes"))
        assert list(matched["Player"]) == ["Bruno Borges Fernandes"]

    def test_a_shared_surname_matches_neither(self):
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame({"Player": ["Alex Palmer", "Cole Palmer"]})
        assert attach_odds(pool, self._odds("Palmer")).empty

    def test_the_right_palmer_is_matched_on_a_full_name(self):
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame({"Player": ["Alex Palmer", "Cole Palmer"]})
        matched = attach_odds(pool, self._odds("Cole Palmer"))
        assert list(matched["Player"]) == ["Cole Palmer"]


class TestSettledMarkets:
    """A market whose destination is the player's current club has resolved.

    The feed keeps quoting "Bradley Barcola to Liverpool" after Liverpool signed
    him. Scored as exit risk that reads as a 25% chance a new arrival leaves —
    the inverse of the exclude_team trap in parse_destination.
    """

    @staticmethod
    def _odds(name, next_club):
        return pd.DataFrame([{
            "Player": name, "Slug": "x", "Next_Club": next_club,
            "Fractional": "3/1", "Decimal": 4.0, "Implied": 0.25,
            "Bookmaker": "William Hill", "Trending": "neutral", "Updated": None,
        }])

    def test_market_to_his_own_club_is_dropped(self):
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame([{"Player": "Bradley Barcola", "Team": "LIV"}])
        assert attach_odds(pool, self._odds("Bradley Barcola", "Liverpool")).empty

    def test_club_names_resolve_through_the_team_map(self):
        """The feed writes 'Manchester Utd', the bootstrap writes 'MUN'."""
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame([{"Player": "Carlos Baleba", "Team": "MUN"}])
        assert attach_odds(pool, self._odds("Carlos Baleba", "Manchester Utd")).empty

    def test_a_genuine_exit_market_survives(self):
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame([{"Player": "Cole Palmer", "Team": "CHE"}])
        matched = attach_odds(pool, self._odds("Cole Palmer", "Manchester United"))
        assert list(matched["Player"]) == ["Cole Palmer"]

    def test_a_non_league_destination_survives(self):
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame([{"Player": "Cole Palmer", "Team": "CHE"}])
        matched = attach_odds(pool, self._odds("Cole Palmer", "Any Saudi club"))
        assert len(matched) == 1

    def test_a_pool_without_teams_still_matches(self):
        """The settled-market guard must not become a hard requirement."""
        from scripts.common.transfer_risk_app import attach_odds
        pool = pd.DataFrame({"Player": ["Cole Palmer"]})
        assert len(attach_odds(pool, self._odds("Cole Palmer", "Chelsea"))) == 1
