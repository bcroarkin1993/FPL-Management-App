"""Live plausibility for the bookmaker odds feed.

Contract, as everywhere in this directory: unreachable -> SKIP, reachable but
implausible -> FAIL, broken in our own code -> FAIL.

This source was rejected once for being stale and re-admitted on the condition
that its staleness is measured rather than assumed. These tests hold that line:
they do not assert the quotes are fresh -- they are not, and the model handles
it -- they assert we can still tell how old they are and that the arithmetic
over them is sound.
"""

import pytest

from scripts.common.data_validation import (check_transfer_odds, format_issues,
                                            raise_on_error)
from scripts.common.transfer_odds import (implied_probability, ladder_overround,
                                          normalise_ladder, odds_age_days,
                                          parse_fractional)

from .conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def odds_index():
    """The live index, or a skip when the site did not serve us the odds page.

    An empty frame used to fail outright, on the reasoning that a
    server-rendered page cannot parse to nothing unless its shape changed. That
    is true only when the thing we parsed *was* the odds page. Observed
    2026-09-13: this failed with exactly that message while the parser was
    healthy -- the next request returned 57 rows -- because the site had handed
    back something else for one request. A test that reports weather as a code
    defect is the failure mode the contract at the top of this directory exists
    to prevent, so the two cases are now told apart at the source.
    """
    from scripts.common import odds_feeds

    result = skip_if_unreachable(odds_feeds.fetch_odds_index_with_status,
                                 "footballtransfers.co.uk odds index")
    df, status, note = result

    if status == odds_feeds.ODDS_UNREACHABLE:
        pytest.skip("odds index unreachable: %s" % note)
    if status == odds_feeds.ODDS_NOT_ODDS_PAGE:
        pytest.skip("odds index did not serve the odds page: %s" % note)
    if status == odds_feeds.ODDS_SHAPE_CHANGED or df is None or df.empty:
        pytest.fail(
            "The odds page was served but parsed to nothing, so its shape has "
            "changed -- see odds_feeds._parse_index, which reads the RSC JSON "
            "payload with a ticker-anchor fallback. Detail: %s" % note
        )
    return df


@pytest.fixture(scope="module")
def probe_ladder(odds_index):
    from scripts.common.odds_feeds import fetch_player_odds_ladder
    slug = str(odds_index.iloc[0]["Slug"])
    df = skip_if_unreachable(lambda: fetch_player_odds_ladder(slug),
                             "odds ladder for %s" % slug)
    if df is None or df.empty:
        pytest.skip("No ladder published for %s" % slug)
    return df


class TestOddsIndex:
    def test_index_has_players_with_prices(self, odds_index):
        assert len(odds_index) >= 5, (
            "Only %d markets found; the index normally carries dozens."
            % len(odds_index))
        assert odds_index["Player"].notna().all()

    def test_every_quoted_price_parses(self, odds_index):
        """A silently unparsed price becomes a missing signal, not an error."""
        unparsed = [str(r["Player"]) for _, r in odds_index.iterrows()
                    if parse_fractional(r["Fractional"]) is None]
        assert not unparsed, (
            "Fractional odds failed to parse for: %s. Check transfer_odds."
            "parse_fractional against the live format." % unparsed[:5])

    def test_implied_probabilities_are_probabilities(self, odds_index):
        for _, row in odds_index.iterrows():
            implied = implied_probability(row["Decimal"])
            assert implied is not None and 0.0 < implied <= 1.0, (
                "%s: decimal %r gave implied %r"
                % (row["Player"], row["Decimal"], implied))

    def test_likelihood_is_not_mistaken_for_probability(self, odds_index):
        """The feed's own 'likelihood' is a proprietary score, not a price.

        It pairs decimal 1.5 (66.7% implied) with likelihood 90. Nothing in the
        app may read it as a probability, so it must not appear as a column.
        """
        assert "Likelihood" not in odds_index.columns


class TestOddsLadder:
    def test_ladder_is_plausible_end_to_end(self, probe_ladder):
        rows = probe_ladder.to_dict("records")
        raise_on_error(
            check_transfer_odds(rows,
                                normalised=normalise_ladder(rows),
                                overround=ladder_overround(rows),
                                age_days=odds_age_days(rows[0].get("Updated"))),
            context="live odds ladder for %s" % rows[0].get("Player"),
        )

    def test_collapsing_overlap_lowers_the_book(self, probe_ladder):
        """If an aggregate is quoted with its members, collapsing must bite."""
        rows = probe_ladder.to_dict("records")
        raw = sum(p for p in (implied_probability(r.get("Decimal")) for r in rows)
                  if p is not None)
        assert ladder_overround(rows) <= raw + 1e-9

    def test_ladder_carries_a_timestamp(self, probe_ladder):
        """Staleness is the known weakness; losing the ability to measure it is
        the failure that matters, because then it looks fresh."""
        updated = probe_ladder.iloc[0].get("Updated")
        if not updated:
            pytest.fail(
                "No semanticOddsUpdatedAt on the ladder page. Without it every "
                "quote falls back to ODDS_ASSUMED_AGE_DAYS and a months-old price "
                "is scored as a month-old one.")

    def test_staleness_is_reported_not_hidden(self, probe_ladder):
        age = odds_age_days(probe_ladder.iloc[0].get("Updated"))
        assert age >= 0, "A future-dated quote means a timezone bug."
        issues = check_transfer_odds(probe_ladder.to_dict("records"), age_days=age)
        # An old quote is a warning by design; it must never be an error.
        assert not [i for i in issues if i.severity == "error"], format_issues(issues)
