"""Free transfers available this gameweek.

The number gates the whole page: the hit verdict on every suggestion card, the
"-4 pts" warning, and whether the 2-Transfer Plan renders at all. It was
reconstructed from one gameweek of history and could not exceed 2.
"""

import pytest

from scripts.classic.transfers import (
    MAX_BANKED_FREE_TRANSFERS,
    _compute_free_transfers,
)


def _history(transfers_by_gw, chips=None):
    return {
        "current": [{"event": gw, "event_transfers": n}
                    for gw, n in sorted(transfers_by_gw.items())],
        "chips": chips or [],
    }


class TestAuthoritativeLimit:
    """FPL states the answer on the authenticated payload. Use it."""

    def test_the_stated_limit_wins(self):
        history = _history({1: 0, 2: 0, 3: 0})  # replay would say 4
        got = _compute_free_transfers(history, {"event_transfers_limit": 1}, current_gw=4)
        assert got == 1

    def test_zero_is_a_real_answer(self):
        assert _compute_free_transfers(
            _history({1: 0}), {"event_transfers_limit": 0}, current_gw=2) == 0

    def test_absent_limit_falls_back_to_the_replay(self):
        history = _history({1: 0, 2: 0})
        assert _compute_free_transfers(history, {}, current_gw=3) == 3


class TestReplay:
    def test_banks_beyond_two(self):
        """The reported ceiling: three quiet gameweeks is four transfers, not two."""
        history = _history({1: 0, 2: 0, 3: 0})
        assert _compute_free_transfers(history, {}, current_gw=4) == 4

    def test_caps_at_the_rules_limit(self):
        history = _history({gw: 0 for gw in range(1, 12)})
        assert _compute_free_transfers(history, {}, current_gw=12) == MAX_BANKED_FREE_TRANSFERS

    def test_spending_resets_the_bank(self):
        history = _history({1: 0, 2: 0, 3: 3})
        assert _compute_free_transfers(history, {}, current_gw=4) == 1

    def test_spending_part_of_a_bank_keeps_the_rest(self):
        history = _history({1: 0, 2: 0, 3: 1})  # held 3, used 1
        assert _compute_free_transfers(history, {}, current_gw=4) == 3

    def test_a_hit_cannot_drive_the_count_negative(self):
        history = _history({1: 5})
        assert _compute_free_transfers(history, {}, current_gw=2) == 1

    def test_transfers_already_made_this_gameweek_are_subtracted(self):
        # Banked to 3 by GW3, one already spent -> 2 still free.
        history = _history({1: 0, 2: 0, 3: 1})
        assert _compute_free_transfers(history, {}, current_gw=3) == 2

    def test_a_hit_taken_this_gameweek_leaves_none(self):
        """The old guard tested event_transfers_cost < 0, which never fires."""
        history = _history({1: 2, 2: 3})
        assert _compute_free_transfers(history, {}, current_gw=2) == 0

    def test_no_history_is_one(self):
        assert _compute_free_transfers(None, {}, current_gw=4) == 1
        assert _compute_free_transfers({}, {}, current_gw=4) == 1


class TestChipGameweeks:
    def test_a_wildcard_does_not_empty_the_bank(self):
        """A wildcard registers a dozen transfers that were never charged."""
        history = _history({1: 0, 2: 12, 3: 0},
                           chips=[{"name": "wildcard", "event": 2}])
        assert _compute_free_transfers(history, {}, current_gw=4) == 3

    def test_a_free_hit_does_not_either(self):
        history = _history({1: 0, 2: 11, 3: 0},
                           chips=[{"name": "freehit", "event": 2}])
        assert _compute_free_transfers(history, {}, current_gw=4) == 3

    def test_the_fh_gws_argument_still_works(self):
        history = _history({1: 0, 2: 11, 3: 0})
        assert _compute_free_transfers(history, {}, current_gw=4, fh_gws={2}) == 3

    def test_a_bench_boost_is_a_normal_gameweek(self):
        history = _history({1: 0, 2: 2}, chips=[{"name": "bboost", "event": 2}])
        assert _compute_free_transfers(history, {}, current_gw=3) == 1


class TestFutureGameweeks:
    def test_history_beyond_the_current_gameweek_is_ignored(self):
        history = _history({1: 0, 2: 0, 3: 0, 4: 3, 5: 3})
        assert _compute_free_transfers(history, {}, current_gw=3) == 3
