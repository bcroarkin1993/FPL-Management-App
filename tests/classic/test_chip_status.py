"""Chip availability on the Classic Transfers page.

Two chips have two uses each, one per half of the season. "Available" therefore
has to mean *playable this gameweek*, not "you own one somewhere this season" --
the app listed Wildcard under Chips Available at GW4 for a manager who had
played it in GW3, because the second slot exists. It does not open until GW20.
"""

import pytest

from scripts.classic.transfers import CHIP_SLOT2_FIRST_GW, _parse_chip_status


def _history(*chips):
    return {"chips": [{"name": name, "event": gw} for name, gw in chips]}


class TestDoubleUseChips:
    def test_a_spent_first_wildcard_is_not_available_in_the_first_half(self):
        status = _parse_chip_status(_history(("wildcard", 3)), current_gw=4)
        assert "wildcard" not in status["available"]
        assert status["wildcard_available"] is False

    def test_it_is_reported_as_owned_but_later(self):
        """Dropping it silently would read as "your wildcard is gone"."""
        status = _parse_chip_status(_history(("wildcard", 3)), current_gw=4)
        assert status["available_later"] == ["wildcard"]
        assert status["slot2_first_gw"] == CHIP_SLOT2_FIRST_GW

    def test_the_second_slot_opens_at_gw20(self):
        before = _parse_chip_status(_history(("wildcard", 3)), current_gw=19)
        at = _parse_chip_status(_history(("wildcard", 3)), current_gw=20)
        assert "wildcard" not in before["available"]
        assert "wildcard" in at["available"]
        assert at["available_later"] == []

    def test_a_spent_second_wildcard_is_gone_for_good(self):
        status = _parse_chip_status(
            _history(("wildcard", 3), ("wildcard", 22)), current_gw=25)
        assert "wildcard" not in status["available"]
        assert status["available_later"] == []
        assert status["wildcard_1_used"] and status["wildcard_2_used"]

    def test_an_unused_first_slot_is_forfeited_after_gw20(self):
        """FPL does not carry an unplayed first-half chip into the second half."""
        status = _parse_chip_status(_history(("wildcard", 22)), current_gw=25)
        assert "wildcard" not in status["available"]

    def test_untouched_chips_are_available_now(self):
        status = _parse_chip_status(_history(), current_gw=4)
        assert set(status["available"]) == {"wildcard", "bboost", "freehit", "3xc"}
        assert status["available_later"] == []

    def test_bench_boost_follows_the_same_rule(self):
        status = _parse_chip_status(_history(("bboost", 5)), current_gw=6)
        assert "bboost" not in status["available"]
        assert status["available_later"] == ["bboost"]


class TestSingleUseChips:
    @pytest.mark.parametrize("chip", ["freehit", "3xc"])
    def test_used_once_is_used_forever(self, chip):
        status = _parse_chip_status(_history((chip, 7)), current_gw=8)
        assert chip not in status["available"]
        assert chip not in status["available_later"]

    @pytest.mark.parametrize("chip", ["freehit", "3xc"])
    def test_unused_stays_available(self, chip):
        status = _parse_chip_status(_history(), current_gw=25)
        assert chip in status["available"]


class TestAdvisorFlags:
    def test_wildcard_alert_cannot_fire_on_a_spent_wildcard(self):
        """The Chip Strategy advisor gates on this flag.

        Unscoped, it could recommend a wildcard rebuild sixteen weeks before the
        chip could be played.
        """
        status = _parse_chip_status(_history(("wildcard", 3)), current_gw=10)
        assert status["wildcard_available"] is False

    def test_no_history_is_not_a_crash(self):
        status = _parse_chip_status(None, current_gw=4)
        assert set(status["available"]) == {"wildcard", "bboost", "freehit", "3xc"}
