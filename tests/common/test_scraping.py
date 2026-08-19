"""Tests for scripts/common/scraping.py's Rotowire projections parser.

Covers the header-driven parsing added to handle Rotowire's two known
table shapes (standard weekly rankings vs. preseason "best picks" preview),
plus the legacy positional fallback for pages with no <thead>.
"""

import contextlib
import logging
from unittest.mock import MagicMock, patch

import pytest

from scripts.common.scraping import (
    _map_rotowire_header_row,
    _rotowire_row_from_header_map,
    get_rotowire_player_projections,
)

# Real headers/rows captured live from Rotowire (2026-08), trimmed to a few rows.
WEEKLY_TABLE_HTML = """
<html><body>
<table class="ck-table-resized">
  <thead><tr>
    <th>Overall Rank</th><th>FW Rank</th><th>MID Rank</th><th>DEF Rank</th><th>GK Rank</th>
    <th>Player</th><th>Team</th><th>Matchup</th><th>Pos</th><th>Price</th><th>TSB%</th><th>Pts</th>
  </tr></thead>
  <tbody>
    <tr><td>1</td><td></td><td>1</td><td></td><td></td>
        <td>Jeremy Doku</td><td>MCI</td><td>MCI v. AVL</td><td>M</td><td>6.5</td><td>6.5</td><td>6.84</td></tr>
    <tr><td>2</td><td>1</td><td></td><td></td><td></td>
        <td>Erling Haaland</td><td>MCI</td><td>MCI v. AVL</td><td>F</td><td>15.0</td><td>60.1</td><td>7.15</td></tr>
  </tbody>
</table>
</body></html>
"""

PREVIEW_TABLE_HTML = """
<html><body>
<table class="ck-table-resized">
  <thead><tr>
    <th>Rank</th><th>Player</th><th>Team</th><th>Pos</th><th>Price</th><th>Adj Total</th>
  </tr></thead>
  <tbody>
    <tr><td>1</td><td>Erling Haaland</td><td>Man City</td><td>F</td><td>15.5</td><td>37.7</td></tr>
    <tr><td>2</td><td>Bruno Fernandes</td><td>Man Utd</td><td>M</td><td>12.0</td><td>31.1</td></tr>
  </tbody>
</table>
</body></html>
"""

# No <thead> at all -- exercises the legacy fixed 12-column positional fallback.
NO_THEAD_TABLE_HTML = """
<html><body>
<table class="article-table__tablesorter">
  <tbody>
    <tr><td>1</td><td></td><td>1</td><td></td><td></td>
        <td>Jeremy Doku</td><td>MCI</td><td>MCI v. AVL</td><td>M</td><td>6.5</td><td>6.5</td><td>6.84</td></tr>
  </tbody>
</table>
</body></html>
"""

NO_TABLE_HTML = "<html><body><p>No table here.</p></body></html>"


def _mock_response(html: str):
    resp = MagicMock()
    resp.content = html.encode("utf-8")
    resp.raise_for_status = MagicMock()
    return resp


class TestMapRotowireHeaderRow:
    def test_weekly_headers(self):
        headers = ["Overall Rank", "FW Rank", "MID Rank", "DEF Rank", "GK Rank",
                   "Player", "Team", "Matchup", "Pos", "Price", "TSB%", "Pts"]
        mapping = _map_rotowire_header_row(headers)
        assert mapping["Player"] == 5
        assert mapping["Team"] == 6
        assert mapping["Points"] == 11

    def test_preview_headers(self):
        headers = ["Rank", "Player", "Team", "Pos", "Price", "Adj Total"]
        mapping = _map_rotowire_header_row(headers)
        assert mapping["Player"] == 1
        assert mapping["Team"] == 2
        assert mapping["Points"] == 5
        assert "FW Rank" not in mapping

    def test_unrecognized_headers_return_empty_mapping(self):
        assert _map_rotowire_header_row(["Foo", "Bar"]) == {}


@contextlib.contextmanager
def caplog_at_error():
    """Collect fpl_app.scraping ERROR records (caplog fixture isn't available
    inside the plain-class tests below)."""
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    logger = logging.getLogger("fpl_app.scraping")
    handler = _Handler(level=logging.ERROR)
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)


def _safe_numeric(val, default=0):
    """Stand-in matching get_rotowire_player_projections' nested _safe_numeric
    (not separately importable) for isolated unit tests of the row mapper."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class TestRotowireRowFromHeaderMap:
    def test_missing_player_or_team_returns_none(self):
        header_map = {"Player": 0, "Team": 1}
        assert _rotowire_row_from_header_map(["", "MCI"], header_map, _safe_numeric) is None
        assert _rotowire_row_from_header_map(["Doku", ""], header_map, _safe_numeric) is None

    def test_missing_optional_fields_default_safely(self):
        header_map = {"Player": 0, "Team": 1, "Points": 2}
        row = _rotowire_row_from_header_map(["Doku", "MCI", "6.5"], header_map, _safe_numeric)
        assert row["Player"] == "Doku"
        assert row["Points"] == 6.5
        assert row["FW Rank"] == 0
        assert row["Matchup"] == ""


class TestGetRotowireProjections:
    def test_weekly_format_parses_correctly(self):
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(WEEKLY_TABLE_HTML)):
            df = get_rotowire_player_projections("https://example.com/weekly")
        assert len(df) == 2
        assert set(df["Player"]) == {"Jeremy Doku", "Erling Haaland"}
        doku = df[df["Player"] == "Jeremy Doku"].iloc[0]
        assert doku["Points"] == 6.84
        assert doku["Pos Rank"] == 1  # MID Rank contributes 1, others 0

    def test_preview_format_parses_correctly(self):
        """Regression test for the live bug: this table shape used to be
        silently dropped entirely (0 rows) because every row had fewer than
        the hardcoded 12 expected columns."""
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(PREVIEW_TABLE_HTML)):
            df = get_rotowire_player_projections("https://example.com/preview")
        assert len(df) == 2
        haaland = df[df["Player"] == "Erling Haaland"].iloc[0]
        assert haaland["Points"] == 37.7  # mapped from "Adj Total"
        assert haaland["Pos Rank"] == 0   # no per-position ranks in this table shape
        assert haaland["Team"] == "Man City"

    def test_no_thead_falls_back_to_legacy_positional_parsing(self):
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(NO_THEAD_TABLE_HTML)):
            df = get_rotowire_player_projections("https://example.com/legacy")
        assert len(df) == 1
        assert df.iloc[0]["Player"] == "Jeremy Doku"
        assert df.iloc[0]["Points"] == 6.84

    def test_no_table_on_page_returns_empty(self):
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(NO_TABLE_HTML)):
            df = get_rotowire_player_projections("https://example.com/empty")
        assert df.empty

    def test_request_failure_returns_empty(self):
        import requests
        with patch("scripts.common.scraping.requests.get", side_effect=requests.ConnectionError("network down")):
            df = get_rotowire_player_projections("https://example.com/fails")
        assert df.empty


class TestRotowireRangeArticleIsRefused:
    """Rotowire's "best picks for gameweeks X-Y" articles are not a projection
    source. Their column is headed "Adj Total" -- an adjusted value metric
    accumulated over the range, not points for any gameweek. Using one made
    Kelleher show 18.6 for a single gameweek instead of 3.31, and scaling it down
    by the range width would not recover a projection, only disguise a made-up
    number. The scraper refuses it so the app's "projections unavailable" warning
    shows instead."""

    def test_range_article_returns_nothing(self):
        url = (
            "https://www.rotowire.com/soccer/article/"
            "best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238"
        )
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(PREVIEW_TABLE_HTML)):
            df = get_rotowire_player_projections(url)
        assert df.empty, "a multi-gameweek 'best picks' article must not be used"

    def test_refusal_is_logged_loudly(self):
        url = (
            "https://www.rotowire.com/soccer/article/"
            "best-fpl-picks-for-gameweeks-2-6-fantasy-premier-league-2026-27-126999"
        )
        with caplog_at_error() as records:
            with patch("scripts.common.scraping.requests.get", return_value=_mock_response(PREVIEW_TABLE_HTML)):
                get_rotowire_player_projections(url)
        assert any("Refusing to use it" in r.message for r in records)

    def test_single_gw_article_points_untouched(self):
        url = (
            "https://www.rotowire.com/soccer/article/"
            "fpl-gameweek-1-best-players-captain-picks-2026-27-rankings-gw1-127487"
        )
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(WEEKLY_TABLE_HTML)):
            df = get_rotowire_player_projections(url)
        assert df[df["Player"] == "Erling Haaland"].iloc[0]["Points"] == 7.15

    def test_refusal_does_not_depend_on_the_range_width(self):
        """It is the article *type* that is wrong, not the width of its window --
        a one-gameweek "best picks" article still reports Adj Total, not points."""
        url = (
            "https://www.rotowire.com/soccer/article/"
            "best-fpl-picks-for-gameweeks-7-7-fantasy-premier-league-2026-27-130000"
        )
        with patch("scripts.common.scraping.requests.get", return_value=_mock_response(WEEKLY_TABLE_HTML)):
            df = get_rotowire_player_projections(url)
        assert df.empty

    def test_implausible_median_logs_a_warning(self, caplog):
        """Tripwire for the next slug shape Rotowire invents: a table that still
        doesn't look like single-GW data should say so loudly."""
        url = "https://www.rotowire.com/soccer/article/some-unrecognized-season-long-table-999999"
        with caplog.at_level(logging.WARNING, logger="fpl_app.scraping"):
            with patch("scripts.common.scraping.requests.get", return_value=_mock_response(PREVIEW_TABLE_HTML)):
                get_rotowire_player_projections(url)
        assert any("implausible for a single gameweek" in r.message for r in caplog.records)
