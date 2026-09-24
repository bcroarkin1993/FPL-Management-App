"""Tests for scripts/common/alert_config.py."""

import json
import pytest
from unittest.mock import patch

from scripts.common.alert_config import (
    _deep_merge,
    load_settings,
    save_settings,
    update_alert_state,
    DEFAULT_SETTINGS,
)


class TestDeepMerge:
    def test_flat_override(self):
        defaults = {"a": 1, "b": 2}
        overrides = {"b": 3}
        result = _deep_merge(defaults, overrides)
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self):
        defaults = {"outer": {"a": 1, "b": 2}}
        overrides = {"outer": {"b": 3}}
        result = _deep_merge(defaults, overrides)
        assert result == {"outer": {"a": 1, "b": 3}}

    def test_new_key_added(self):
        defaults = {"a": 1}
        overrides = {"b": 2}
        result = _deep_merge(defaults, overrides)
        assert result == {"a": 1, "b": 2}

    def test_no_overrides(self):
        defaults = {"a": 1, "nested": {"x": 10}}
        result = _deep_merge(defaults, {})
        assert result == defaults

    def test_deeply_nested(self):
        defaults = {"l1": {"l2": {"l3": "default"}}}
        overrides = {"l1": {"l2": {"l3": "override"}}}
        result = _deep_merge(defaults, overrides)
        assert result["l1"]["l2"]["l3"] == "override"


class TestLoadSettings:
    def test_missing_file_returns_defaults(self, tmp_path):
        """When config file doesn't exist, return defaults."""
        with patch("scripts.common.alert_config._find_config_path", return_value=tmp_path / "nonexistent.json"):
            settings = load_settings()
        assert settings["version"] == DEFAULT_SETTINGS["version"]
        assert "deadline_alerts" in settings

    def test_valid_file(self, tmp_path):
        """When config file exists with partial data, merge with defaults."""
        config_path = tmp_path / "alert_settings.json"
        config_path.write_text(json.dumps({
            "version": 2,
            "discord": {"mention_user_id": "12345"},
        }))
        with patch("scripts.common.alert_config._find_config_path", return_value=config_path):
            settings = load_settings()
        assert settings["version"] == 2
        assert settings["discord"]["mention_user_id"] == "12345"
        # Default keys should still be present
        assert "deadline_alerts" in settings
        assert "data_source_alerts" in settings

    def test_corrupt_json_returns_defaults(self, tmp_path):
        """When config file has invalid JSON, return defaults."""
        config_path = tmp_path / "alert_settings.json"
        config_path.write_text("{invalid json")
        with patch("scripts.common.alert_config._find_config_path", return_value=config_path):
            settings = load_settings()
        assert settings == DEFAULT_SETTINGS


class TestSaveSettings:
    def test_round_trip(self, tmp_path):
        """Save and load should produce identical results."""
        config_path = tmp_path / "alert_settings.json"
        with patch("scripts.common.alert_config._find_config_path", return_value=config_path):
            settings = dict(DEFAULT_SETTINGS)
            settings["discord"]["mention_user_id"] = "test_user_123"
            assert save_settings(settings) is True

            loaded = load_settings()
            assert loaded["discord"]["mention_user_id"] == "test_user_123"


class TestUpdateAlertState:
    def test_updates_state(self, tmp_path):
        config_path = tmp_path / "alert_settings.json"
        config_path.write_text(json.dumps(DEFAULT_SETTINGS))
        with patch("scripts.common.alert_config._find_config_path", return_value=config_path):
            update_alert_state("rotowire", 25)
            settings = load_settings()
            assert settings["alert_state"]["last_rotowire_alert_gw"] == 25

    def test_creates_state_key(self, tmp_path):
        config_path = tmp_path / "alert_settings.json"
        config_path.write_text(json.dumps(DEFAULT_SETTINGS))
        with patch("scripts.common.alert_config._find_config_path", return_value=config_path):
            update_alert_state("ffp", 10)
            settings = load_settings()
            assert settings["alert_state"]["last_ffp_alert_gw"] == 10


class TestTradeDeadlineAlertSettings:
    """Trades close with waivers, or a day earlier under approval — its own deadline."""

    def test_trade_defaults_exist(self):
        from scripts.common.alert_config import DEFAULT_SETTINGS

        trade = DEFAULT_SETTINGS["deadline_alerts"]["trade"]
        assert trade["enabled"] is False
        assert trade["alert_windows"] == [24, 6, 1]

    def test_backfilled_into_a_config_written_before_it_existed(self):
        """No migration: _deep_merge fills the key in for configs already on disk.

        alert_settings.json is committed and rewritten by the notifier workflow, so
        a config predating this feature is the normal case, not an edge one.
        """
        from scripts.common.alert_config import DEFAULT_SETTINGS, _deep_merge

        old_config = {
            "version": 1,
            "deadline_alerts": {
                "draft": {"enabled": True, "alert_windows": [12]},
                "classic": {"enabled": True, "alert_windows": [6]},
            },
        }
        merged = _deep_merge(DEFAULT_SETTINGS, old_config)

        assert merged["deadline_alerts"]["trade"] == {
            "enabled": False, "alert_windows": [24, 6, 1],
        }
        # The user's own settings survive the merge untouched.
        assert merged["deadline_alerts"]["draft"] == {"enabled": True, "alert_windows": [12]}


class TestTradeDeadlineDerivation:
    """Where accepted trades need approval, offers shut 24h before the waiver deadline."""

    def _kickoff(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        return datetime(2026, 9, 26, 10, 0, tzinfo=ZoneInfo("America/New_York"))

    def test_admin_approval_pulls_the_deadline_forward(self):
        from datetime import timedelta
        from scripts.common.waiver_alerts import (
            DRAFT_OFFSET_HOURS, TRADE_APPROVAL_LEAD_HOURS, trade_deadline_for,
        )

        kickoff = self._kickoff()
        deadline, assumed = trade_deadline_for(kickoff, "a")

        expected = kickoff - timedelta(
            hours=DRAFT_OFFSET_HOURS + TRADE_APPROVAL_LEAD_HOURS
        )
        assert deadline == expected
        assert assumed is False

    def test_an_unreadable_setting_takes_the_earlier_deadline_and_says_so(self):
        """Erring early is a mild annoyance; erring late costs the window entirely.

        The flag matters as much as the time: the message must not state a deadline
        this league never confirmed as though it had.
        """
        from scripts.common.waiver_alerts import trade_deadline_for

        kickoff = self._kickoff()
        approved, _ = trade_deadline_for(kickoff, "a")

        for unreadable in (None, "zzz"):
            deadline, assumed = trade_deadline_for(kickoff, unreadable)
            assert deadline == approved, unreadable
            assert assumed is True, unreadable


class TestPublishedDeadlines:
    """FPL states its own waiver and trade deadlines; the offsets are the fallback.

    The Draft bootstrap carries `waivers_time` and `trades_time` on every event.
    They agree with the hardcoded 25.5h/49.5h for all 38 gameweeks of this season,
    so nothing is visibly broken today — but `waivers_before_deadline_hours_event`
    exists to shorten a specific gameweek's window, and arithmetic over the fixture
    list cannot see that.
    """

    TZ_NAME = "America/New_York"

    def _events(self):
        """One gameweek, shaped as the live bootstrap shapes it."""
        return [{
            "id": 6,
            "deadline_time": "2026-10-10T10:00:00Z",
            "waivers_time": "2026-10-09T10:00:00Z",
            "trades_time": "2026-10-08T10:00:00Z",
        }]

    def _kickoff(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        # Deliberately inconsistent with the published times, so a test that passes
        # can only have read the published ones.
        return datetime(2026, 10, 10, 3, 0, tzinfo=ZoneInfo(self.TZ_NAME))

    def test_the_events_list_lives_under_data(self):
        """`events` is a dict with the list under `data`, not a bare list."""
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        class _Resp:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"events": {"current": 5, "next": 6, "data": [{"id": 6}]}}

        with patch.object(wa.requests, "get", return_value=_Resp()):
            assert wa._draft_events() == [{"id": 6}]

    def test_published_waiver_time_beats_the_kickoff_offset(self):
        import scripts.common.waiver_alerts as wa

        deadline, source = wa.waiver_deadline_for(6, self._kickoff(), events=self._events())
        assert source == "published"
        assert deadline == wa._iso_to_et("2026-10-09T10:00:00Z")

    def test_an_unreadable_bootstrap_falls_back_and_says_so(self):
        from datetime import timedelta
        import scripts.common.waiver_alerts as wa

        kickoff = self._kickoff()
        deadline, source = wa.waiver_deadline_for(6, kickoff, events=[])
        assert source == "offset"
        assert deadline == kickoff - timedelta(hours=wa.DRAFT_OFFSET_HOURS)

    def test_a_gameweek_absent_from_the_payload_is_not_a_zero(self):
        import scripts.common.waiver_alerts as wa
        assert wa.published_deadlines(99, events=self._events()) is None
        assert wa.published_deadlines(None, events=self._events()) is None

    def test_approval_takes_trades_time_and_instant_takes_waivers_time(self):
        """Which published time applies is a property of the league, not the game."""
        import scripts.common.waiver_alerts as wa

        events = self._events()
        approval, assumed = wa.trade_deadline_for(None, "a", gw=6, events=events)
        assert approval == wa._iso_to_et("2026-10-08T10:00:00Z")
        assert assumed is False

        # A recognised no-approval code would shut with waivers. None is confirmed
        # yet, so this asserts the branch through the constant rather than a code.
        with patch_known_setting(wa, "x"):
            instant, assumed = wa.trade_deadline_for(None, "x", gw=6, events=events)
        assert instant == wa._iso_to_et("2026-10-09T10:00:00Z")
        assert assumed is False

    def test_an_unreadable_setting_still_takes_the_earlier_published_time(self):
        import scripts.common.waiver_alerts as wa

        events = self._events()
        for unreadable in (None, "zzz"):
            deadline, assumed = wa.trade_deadline_for(None, unreadable, gw=6, events=events)
            assert deadline == wa._iso_to_et("2026-10-08T10:00:00Z"), unreadable
            assert assumed is True, unreadable


import contextlib


@contextlib.contextmanager
def patch_known_setting(wa, code):
    """Temporarily treat `code` as a recognised no-approval trade setting."""
    original = wa.KNOWN_TRADE_SETTINGS
    wa.KNOWN_TRADE_SETTINGS = original | {code}
    try:
        yield
    finally:
        wa.KNOWN_TRADE_SETTINGS = original


class TestTradeDeadlineGameweekTargeting:
    """A midweek gameweek's trade windows all fall inside the previous gameweek.

    main() derives every deadline from one gameweek's kickoff, and that gameweek is
    still N until N's last match finishes. Under approval, trades close 49.5h before
    kickoff — so for a Tuesday fixture all three windows (73.5h/55.5h/50.5h out) land
    while GW N is being played, `hours_left` reads negative, and the alert silently
    never fires. 7 of 38 gameweeks are midweek.

    Every test here stubs `_draft_events` to [] so it exercises the *fallback*
    kickoff arithmetic against fabricated gameweeks. Left live they would score
    against FPL's real published calendar, which is a different test (and a
    network call from an offline suite).
    """

    TZ_NAME = "America/New_York"

    def _tz(self):
        from zoneinfo import ZoneInfo
        return ZoneInfo(self.TZ_NAME)

    def test_midweek_gameweek_alerts_are_reachable(self):
        from datetime import datetime
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        tz = self._tz()
        gw8_kickoff = datetime(2026, 10, 3, 7, 30, tzinfo=tz)    # Sat
        gw9_kickoff = datetime(2026, 10, 6, 15, 0, tzinfo=tz)    # Tue — midweek
        kickoffs = {8: gw8_kickoff, 9: gw9_kickoff}

        # Each of the three windows, evaluated while GW8 is still in play.
        moments = [
            (24, datetime(2026, 10, 3, 13, 30, tzinfo=tz)),
            (6, datetime(2026, 10, 4, 7, 30, tzinfo=tz)),
            (1, datetime(2026, 10, 4, 12, 30, tzinfo=tz)),
        ]
        with patch.object(wa, "_earliest_kickoff_et", side_effect=lambda g: kickoffs[g]), \
             patch.object(wa, "_draft_events", return_value=[]):
            for window, now in moments:
                gw, deadline, _ = wa.resolve_trade_deadline(8, gw8_kickoff, "a", now)
                assert gw == 9, f"{window}h window still targeting the old gameweek"
                hours_left = (deadline - now).total_seconds() / 3600
                assert abs(hours_left - window) <= 0.5, (window, hours_left)

    def test_a_live_deadline_is_never_skipped_ahead(self):
        """The look-ahead fires only once this gameweek's window has actually shut."""
        from datetime import datetime
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        tz = self._tz()
        kickoff = datetime(2026, 10, 10, 10, 0, tzinfo=tz)       # Sat
        now = datetime(2026, 10, 7, 8, 30, tzinfo=tz)            # deadline still ahead

        with patch.object(wa, "_earliest_kickoff_et", side_effect=AssertionError), \
             patch.object(wa, "_draft_events", return_value=[]):
            gw, deadline, _ = wa.resolve_trade_deadline(10, kickoff, "a", now)

        assert gw == 10
        assert deadline > now

    def test_no_fixtures_for_the_next_gameweek_degrades_quietly(self):
        """End of season: there is no GW+1 to look ahead to."""
        from datetime import datetime
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        tz = self._tz()
        kickoff = datetime(2026, 10, 3, 7, 30, tzinfo=tz)
        now = datetime(2026, 10, 3, 12, 0, tzinfo=tz)            # deadline long past

        with patch.object(wa, "_earliest_kickoff_et",
                          side_effect=RuntimeError("no fixtures")), \
             patch.object(wa, "_draft_events", return_value=[]):
            gw, deadline, _ = wa.resolve_trade_deadline(38, kickoff, "a", now)

        assert gw == 38
        assert deadline < now          # stale, so no window matches and nothing sends


class TestNotifierLeagueIdResolution:
    """The notifier must see a league configured on the League Setup page.

    Reading FPL_DRAFT_LEAGUE_ID alone put "your league's trade setting could not be
    read" on every locally-run alert for a user who had never touched .env, with the
    setting sitting readable in league_settings.json.
    """

    def test_locked_local_settings_win_over_the_env(self):
        import os
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        locked = {"draft": {"locked": True, "league_id": 4242}}
        with patch("scripts.common.league_config.load_settings", return_value=locked), \
             patch.dict(os.environ, {"FPL_DRAFT_LEAGUE_ID": "9999"}):
            assert wa._resolve_draft_league_id() == "4242"

    def test_env_is_the_fallback_when_nothing_is_locked(self):
        import os
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        unlocked = {"draft": {"locked": False, "league_id": 4242}}
        with patch("scripts.common.league_config.load_settings", return_value=unlocked), \
             patch.dict(os.environ, {"FPL_DRAFT_LEAGUE_ID": "9999"}):
            assert wa._resolve_draft_league_id() == "9999"

    def test_unreadable_settings_fall_back_rather_than_raise(self):
        import os
        from unittest.mock import patch
        import scripts.common.waiver_alerts as wa

        with patch("scripts.common.league_config.load_settings",
                   side_effect=OSError("boom")), \
             patch.dict(os.environ, {"FPL_DRAFT_LEAGUE_ID": "9999"}):
            assert wa._resolve_draft_league_id() == "9999"
