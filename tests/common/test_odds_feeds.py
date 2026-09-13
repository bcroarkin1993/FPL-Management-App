"""Offline tests for the bookmaker odds fetcher.

This module had no offline coverage at all, which is how a one-off bad response
came to be reported by the live suite as "the page shape changed" while the
parser was perfectly healthy. The tests here pin the parse against a real
captured payload and, more importantly, pin the distinction between the site
changing and the site having a bad moment.
"""

import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.common import odds_feeds
from scripts.common.transfer_odds import ODDS_INDEX_COLUMNS

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "odds_index.html"


@pytest.fixture(scope="module")
def index_html():
    return FIXTURE.read_text()


class TestParseIndex:
    def test_the_real_payload_parses(self, index_html):
        rows = odds_feeds._parse_index(index_html)
        assert len(rows) >= 6
        first = rows[0]
        assert first["Player"] and first["Slug"]
        assert 0 < first["Implied"] <= 1

    def test_records_carry_every_column(self, index_html):
        df = pd.DataFrame(odds_feeds._parse_index(index_html),
                          columns=ODDS_INDEX_COLUMNS)
        assert list(df.columns) == ODDS_INDEX_COLUMNS
        assert df["Slug"].is_unique, "slugs are the join key and must not repeat"
        assert df["Decimal"].gt(1.0).all(), "decimal odds are always > 1"

    def test_likelihood_is_not_read_as_a_probability(self, index_html):
        """The feed pairs decimal 1.5 (66.7% implied) with likelihood 90. The
        field is parsed past, never used -- Implied must come from the odds."""
        rows = odds_feeds._parse_index(index_html)
        alvarez = next(r for r in rows if r["Slug"] == "julian-alvarez")
        assert alvarez["Implied"] == pytest.approx(1 / 1.5, abs=1e-6)


class TestOddsPageDetection:
    """Telling "the site changed" apart from "the site had a bad moment"."""

    def test_the_real_page_is_recognised(self, index_html):
        assert odds_feeds._looks_like_odds_page(index_html)

    @pytest.mark.parametrize("body", [
        "",
        "<html><body>Just a moment...</body></html>",
        "<html><body><h1>502 Bad Gateway</h1></body></html>",
    ])
    def test_something_else_is_not(self, body):
        assert not odds_feeds._looks_like_odds_page(body)


class TestFetchStatus:
    def test_a_healthy_fetch_reports_ok(self, index_html):
        with patch.object(odds_feeds, "_get", return_value=index_html):
            df, status, note = odds_feeds.fetch_odds_index_with_status()
        assert status == odds_feeds.ODDS_OK
        assert not df.empty and note == ""

    def test_a_transport_failure_is_unreachable_not_a_shape_change(self):
        with patch.object(odds_feeds, "_get", side_effect=RuntimeError("reset")):
            df, status, _ = odds_feeds.fetch_odds_index_with_status()
        assert status == odds_feeds.ODDS_UNREACHABLE
        assert df.empty

    def test_a_page_that_is_not_the_odds_page_says_so(self):
        """The case that misled the live suite: a 200 carrying something else.
        Reporting this as a shape change sends someone hunting a parser bug."""
        with patch.object(odds_feeds, "_get", return_value="<html>nope</html>"):
            df, status, note = odds_feeds.fetch_odds_index_with_status()
        assert status == odds_feeds.ODDS_NOT_ODDS_PAGE
        assert df.empty and "not the odds page" in note

    def test_the_odds_page_parsing_to_nothing_is_a_shape_change(self):
        """This one *is* a real defect and must stay loud."""
        served = '<a href="/odds/x">hotTransfers Next Club Odds</a>'
        with patch.object(odds_feeds, "_get", return_value=served):
            df, status, note = odds_feeds.fetch_odds_index_with_status()
        assert status == odds_feeds.ODDS_SHAPE_CHANGED
        assert df.empty and "neither the JSON payload" in note

    def test_every_status_still_yields_the_right_columns(self):
        for value in ("<html>nope</html>", '<a href="/odds/x">hotTransfers</a>'):
            with patch.object(odds_feeds, "_get", return_value=value):
                df, _, _ = odds_feeds.fetch_odds_index_with_status()
            assert list(df.columns) == ODDS_INDEX_COLUMNS

    def test_the_plain_fetch_never_raises_and_always_returns_a_frame(self):
        """The app-facing call: a dead odds feed must not take a page down."""
        with patch.object(odds_feeds, "_get", side_effect=RuntimeError("down")):
            df = odds_feeds.fetch_odds_index()
        assert isinstance(df, pd.DataFrame) and df.empty


class TestRetry:
    """``_get`` had no retry, alone among the app's feeds."""

    def test_a_transient_failure_is_retried(self, index_html):
        """One bad response used to empty the whole odds board."""
        with patch.object(odds_feeds, "_session") as session, \
             patch.object(odds_feeds.time, "sleep"):
            session.return_value.get.side_effect = [
                RuntimeError("transient"),
                type("R", (), {"text": index_html,
                               "raise_for_status": staticmethod(lambda: None)})(),
            ]
            assert odds_feeds._get(odds_feeds.ODDS_INDEX_URL) == index_html

    def test_it_gives_up_after_the_last_attempt(self):
        with patch.object(odds_feeds, "_session") as session, \
             patch.object(odds_feeds.time, "sleep"):
            session.return_value.get.side_effect = RuntimeError("always down")
            with pytest.raises(RuntimeError):
                odds_feeds._get(odds_feeds.ODDS_INDEX_URL)
            assert session.return_value.get.call_count == odds_feeds.DEFAULT_ATTEMPTS

    def test_a_healthy_fetch_does_not_retry(self, index_html):
        with patch.object(odds_feeds, "_session") as session:
            session.return_value.get.return_value = type(
                "R", (), {"text": index_html,
                          "raise_for_status": staticmethod(lambda: None)})()
            odds_feeds._get(odds_feeds.ODDS_INDEX_URL)
            assert session.return_value.get.call_count == 1
