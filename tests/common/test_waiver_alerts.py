"""Tests for data source alert logic in waiver_alerts.py."""

from datetime import datetime
from unittest.mock import patch, MagicMock
from zoneinfo import ZoneInfo

import pytest

from scripts.common.waiver_alerts import _check_data_source_alerts

TZ = ZoneInfo("America/New_York")


def _make_settings(rotowire_enabled=True, ffp_enabled=False, last_rw_gw=0, last_ffp_gw=0):
    return {
        "data_source_alerts": {
            "rotowire": {"enabled": rotowire_enabled},
            "ffp": {"enabled": ffp_enabled},
        },
        "alert_state": {
            "last_rotowire_alert_gw": last_rw_gw,
            "last_ffp_alert_gw": last_ffp_gw,
        },
    }


class TestDataSourceAlertSkipsAfterKickoff:
    """Alerts must never fire after the GW has started."""

    def test_skips_when_gw_started(self):
        kickoff = datetime(2025, 2, 1, 10, 0, tzinfo=TZ)
        after_kickoff = datetime(2025, 2, 1, 12, 0, tzinfo=TZ)

        with patch("scripts.common.waiver_alerts.datetime") as mock_dt:
            mock_dt.now.return_value = after_kickoff
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _check_data_source_alerts(
                "https://hook", "", gw=25,
                settings=_make_settings(rotowire_enabled=True),
                kickoff_et=kickoff,
            )
        assert result == 0

    def test_sends_when_before_kickoff(self):
        kickoff = datetime(2025, 2, 1, 15, 0, tzinfo=TZ)
        before_kickoff = datetime(2025, 2, 1, 10, 0, tzinfo=TZ)

        with patch("scripts.common.waiver_alerts.datetime") as mock_dt, \
             patch("scripts.common.data_source_checks.is_rotowire_available_for_gw", return_value=True), \
             patch("scripts.common.waiver_alerts.requests.post") as mock_post, \
             patch("scripts.common.waiver_alerts.update_alert_state"), \
             patch("scripts.common.alert_config.load_settings", return_value=_make_settings(last_rw_gw=0)):
            mock_dt.now.return_value = before_kickoff
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _check_data_source_alerts(
                "https://hook", "", gw=25,
                settings=_make_settings(rotowire_enabled=True),
                kickoff_et=kickoff,
            )
        assert result == 1
        mock_post.assert_called_once()


class TestDataSourceAlertDeduplication:
    """Each source should alert at most once per GW."""

    def test_skips_when_already_alerted(self):
        kickoff = datetime(2025, 2, 1, 15, 0, tzinfo=TZ)
        before_kickoff = datetime(2025, 2, 1, 10, 0, tzinfo=TZ)

        with patch("scripts.common.waiver_alerts.datetime") as mock_dt, \
             patch("scripts.common.alert_config.load_settings", return_value=_make_settings(last_rw_gw=25)):
            mock_dt.now.return_value = before_kickoff
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _check_data_source_alerts(
                "https://hook", "", gw=25,
                settings=_make_settings(rotowire_enabled=True, last_rw_gw=25),
                kickoff_et=kickoff,
            )
        assert result == 0

    def test_reads_fresh_state_from_disk(self):
        """State should be re-read from disk, not use stale in-memory settings."""
        kickoff = datetime(2025, 2, 1, 15, 0, tzinfo=TZ)
        before_kickoff = datetime(2025, 2, 1, 10, 0, tzinfo=TZ)

        # In-memory settings say last_rw_gw=0 (stale), but disk says 25 (fresh)
        stale_settings = _make_settings(rotowire_enabled=True, last_rw_gw=0)
        fresh_settings = _make_settings(last_rw_gw=25)

        with patch("scripts.common.waiver_alerts.datetime") as mock_dt, \
             patch("scripts.common.alert_config.load_settings", return_value=fresh_settings):
            mock_dt.now.return_value = before_kickoff
            mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
            result = _check_data_source_alerts(
                "https://hook", "", gw=25,
                settings=stale_settings,
                kickoff_et=kickoff,
            )
        # Should skip because disk state says already alerted for GW 25
        assert result == 0
