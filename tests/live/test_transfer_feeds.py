"""Live plausibility for the transfer-news feed.

Contract, as everywhere in tests/live/: unreachable -> SKIP, reachable but
implausible -> FAIL.  Google News going down must not block a push; Google News
silently changing shape must.

The feed is the app's only *predictive* transfer signal — the FPL bootstrap
resolves moves only after they complete — so a shape change here means the draft
board quietly stops discounting anybody.
"""

import pandas as pd
import pytest

from scripts.common.data_validation import (
    check_transfer_risk,
    check_transfer_windows,
    raise_on_error,
)
from scripts.common.transfer_feeds import NEWS_COLUMNS, fetch_player_transfer_news
from scripts.common.transfer_risk import (
    TRANSFER_WINDOWS,
    attach_transfer_risk,
    classify_headline,
)

from .conftest import skip_if_unreachable

# A player with a permanent, high-volume news presence, so the query is never
# legitimately empty.
_PROBE_PLAYER = ("Bruno Fernandes", "MUN")


@pytest.fixture(scope="module")
def probe_news():
    name, team = _PROBE_PLAYER
    df = skip_if_unreachable(
        lambda: fetch_player_transfer_news(name, team), "Google News RSS"
    )
    if df.empty:
        pytest.skip("Google News returned no items for %s" % name)
    return df


class TestTransferNewsFeed:
    def test_returns_the_columns_the_app_consumes(self, probe_news):
        missing = set(NEWS_COLUMNS) - set(probe_news.columns)
        assert not missing, (
            "Transfer news parser lost column(s) %s — the RSS shape changed. "
            "Present: %s" % (sorted(missing), sorted(probe_news.columns))
        )

    def test_headlines_are_populated(self, probe_news):
        filled = probe_news["Headline"].astype(str).str.strip().ne("").mean()
        assert filled > 0.95, (
            "Only %.0f%% of items carried a headline — <title> parsing has broken."
            % (100 * filled)
        )

    def test_publish_dates_parse(self, probe_news):
        parsed = pd.to_datetime(probe_news["Published"], errors="coerce", utc=True)
        rate = parsed.notna().mean()
        assert rate > 0.9, (
            "Only %.0f%% of pubDate values parsed. Recency decay silently treats "
            "undated headlines as half-weight, so a format change quietly halves "
            "every risk score." % (100 * rate)
        )

    def test_sources_are_attributed(self, probe_news):
        """Corroboration is counted in distinct outlets. If Source stops
        populating, every player collapses to the uncorroborated floor."""
        named = probe_news["Source"].astype(str).str.strip().ne("").mean()
        assert named > 0.8, (
            "Only %.0f%% of items had a source. The corroboration gate depends on "
            "this and would silently suppress every real signal." % (100 * named)
        )

    def test_some_headline_classifies_as_transfer_language(self, probe_news):
        """If the keyword tiers match nothing across a hundred transfer
        headlines, the tier patterns have drifted from how outlets write."""
        scored = [classify_headline(h) for h in probe_news["Headline"]]
        assert any(s > 0 for s in scored), (
            "No headline in %d matched any transfer keyword tier." % len(scored)
        )


class TestTransferRiskOutput:
    def test_end_to_end_output_is_plausible(self, probe_news):
        name, team = _PROBE_PLAYER
        players = pd.DataFrame([{"Player": name, "Team": team}])
        out = attach_transfer_risk(players, probe_news, ["Man Utd", "Arsenal", "Liverpool"])
        raise_on_error(check_transfer_risk(out), context="live transfer risk")

    def test_window_calendar_has_not_lapsed(self):
        """Hardcoded and season-specific: once past, every discount is silently 1.0."""
        issues = check_transfer_windows(TRANSFER_WINDOWS)
        if issues:
            pytest.fail(
                "Transfer window calendar needs updating for the new season:\n%s"
                % "\n".join(str(i) for i in issues)
            )
