"""Free transfers available this gameweek.

The number gates the whole page: the hit verdict on every suggestion card, the
"-4 pts" warning, and how many legs the transfer planner may propose for free.

Two rules of FPL's the replay has to encode, both of which it once got wrong:

* **The first free transfer is for GW2.** Before the first deadline the squad
  is being picked and changes are unlimited, so GW1 grants nothing. Seeding the
  bank at 1 and then crediting GW1 as well reported one too many for the rest
  of the season -- 4 at GW5 against FPL's 3.
* **A wildcard or free hit week neither spends nor earns one.** The bank is
  retained across the chip, but no extra transfer is granted for that week.

And one of the authenticated payload's: `transfers.limit` is the gameweek's
*allowance*, not what is left. `{"limit": 1, "made": 2, "cost": 4}` is one free
transfer, two made and a four-point hit -- so the answer there is 0, not 1.
"""

import pytest

from scripts.classic.transfers import (
    MAX_BANKED_FREE_TRANSFERS,
    _compute_free_transfers,
    _ft_override_key,
    resolve_free_transfers,
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
        history = _history({1: 0, 2: 0, 3: 0})  # replay would say 3
        got = _compute_free_transfers(history, {"event_transfers_limit": 1}, current_gw=4)
        assert got == 1

    def test_the_limit_is_an_allowance_not_a_remainder(self):
        """`{"limit": 1, "made": 2, "cost": 4}` is one free, two made, -4."""
        got = _compute_free_transfers(
            _history({1: 0, 2: 0}),
            {"event_transfers_limit": 1, "event_transfers": 2},
            current_gw=3)
        assert got == 0

    def test_an_unspent_allowance_is_returned_whole(self):
        got = _compute_free_transfers(
            _history({1: 0, 2: 0}),
            {"event_transfers_limit": 3, "event_transfers": 0},
            current_gw=3)
        assert got == 3

    def test_zero_is_a_real_answer(self):
        assert _compute_free_transfers(
            _history({1: 0}), {"event_transfers_limit": 0}, current_gw=2) == 0

    def test_absent_limit_falls_back_to_the_replay(self):
        history = _history({1: 0, 2: 0})
        assert _compute_free_transfers(history, {}, current_gw=3) == 2


class TestReplay:
    def test_banks_beyond_two(self):
        """The reported ceiling: GW2 and GW3 quiet is three at GW4, not two."""
        history = _history({1: 0, 2: 0, 3: 0})
        assert _compute_free_transfers(history, {}, current_gw=4) == 3

    def test_gameweek_one_grants_nothing(self):
        """Changes before the first deadline are unlimited, not a free transfer."""
        assert _compute_free_transfers(_history({1: 0}), {}, current_gw=2) == 1

    def test_a_midseason_joiner_gets_one_not_two(self):
        """Their first deadline is unlimited too, however much they used."""
        history = _history({7: 14})
        assert _compute_free_transfers(history, {}, current_gw=8) == 1

    def test_caps_at_the_rules_limit(self):
        history = _history({gw: 0 for gw in range(1, 12)})
        assert _compute_free_transfers(history, {}, current_gw=12) == MAX_BANKED_FREE_TRANSFERS

    def test_spending_resets_the_bank(self):
        history = _history({1: 0, 2: 0, 3: 3})
        assert _compute_free_transfers(history, {}, current_gw=4) == 1

    def test_spending_part_of_a_bank_keeps_the_rest(self):
        history = _history({1: 0, 2: 0, 3: 1})  # held 2, used 1
        assert _compute_free_transfers(history, {}, current_gw=4) == 2

    def test_a_hit_cannot_drive_the_count_negative(self):
        history = _history({1: 5})
        assert _compute_free_transfers(history, {}, current_gw=2) == 1

    def test_transfers_already_made_this_gameweek_are_subtracted(self):
        # Banked to 2 by GW3, one already spent -> 1 still free.
        history = _history({1: 0, 2: 0, 3: 1})
        assert _compute_free_transfers(history, {}, current_gw=3) == 1

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
        assert _compute_free_transfers(history, {}, current_gw=4) == 2

    def test_a_free_hit_does_not_either(self):
        history = _history({1: 0, 2: 11, 3: 0},
                           chips=[{"name": "freehit", "event": 2}])
        assert _compute_free_transfers(history, {}, current_gw=4) == 2

    def test_the_fh_gws_argument_still_works(self):
        history = _history({1: 0, 2: 11, 3: 0})
        assert _compute_free_transfers(history, {}, current_gw=4, fh_gws={2}) == 2

    def test_a_chip_week_grants_no_extra_transfer(self):
        """The bank is retained across a wildcard, but nothing is added to it.

        This is the live account the off-by-one was found on: GW2 unused, a
        wildcard in GW3, GW4 unused. FPL's answer at GW5 is 3 -- limit(3) = 2,
        the wildcard adds nothing, limit(4) = 2, limit(5) = 3.
        """
        history = _history({1: 0, 2: 0, 3: 12, 4: 0},
                           chips=[{"name": "wildcard", "event": 3}])
        assert _compute_free_transfers(history, {}, current_gw=5) == 3

    def test_a_bench_boost_is_a_normal_gameweek(self):
        history = _history({1: 0, 2: 2}, chips=[{"name": "bboost", "event": 2}])
        assert _compute_free_transfers(history, {}, current_gw=3) == 1


class TestFutureGameweeks:
    def test_history_beyond_the_current_gameweek_is_ignored(self):
        history = _history({1: 0, 2: 0, 3: 0, 4: 3, 5: 3})
        assert _compute_free_transfers(history, {}, current_gw=3) == 2


class TestResolveFreeTransfers:
    """Precedence, and the logged-transfer deduction that was missing entirely.

    Logging a transfer in-app left the panel reading "N FTs Banked / All free
    this gameweek" with transfers already spent, because nothing subtracted
    them -- `apply_pending_transfers()` adjusts only the bank.
    """

    def test_the_replay_is_the_floor(self):
        got = resolve_free_transfers(_history({1: 0, 2: 0}), {}, 3,
                                     team_id=1, _state={})
        assert got["count"] == 2
        assert got["source"] == "replay"

    def test_a_logged_transfer_is_spent(self):
        got = resolve_free_transfers(_history({1: 0, 2: 0}), {}, 3,
                                     team_id=1, logged_pending=1, _state={})
        assert got["count"] == 1
        assert got["hits"] == 0

    def test_logging_past_the_allowance_reports_a_hit(self):
        got = resolve_free_transfers(_history({1: 0, 2: 0}), {}, 3,
                                     team_id=1, logged_pending=3, _state={})
        assert got["count"] == 0
        assert got["hits"] == 1  # 3 logged, 2 free

    def test_an_authenticated_squad_does_not_double_count_a_logged_move(self):
        """`my-team` already counts it in transfers.made, and the pending log
        entry is retired rather than replayed."""
        got = resolve_free_transfers(
            _history({1: 0, 2: 0}),
            {"event_transfers_limit": 2, "event_transfers": 1},
            3, team_id=1, logged_pending=1, squad_source="my_team", _state={})
        assert got["count"] == 1
        assert got["source"] == "my_team"

    def test_a_manual_override_wins(self):
        state = {_ft_override_key(1, 3): 1}
        got = resolve_free_transfers(_history({1: 0, 2: 0}), {}, 3,
                                     team_id=1, _state=state)
        assert got["count"] == 1
        assert got["source"] == "manual"
        assert got["computed"] == 2  # what it overrode, kept for the caption

    def test_an_override_is_scoped_to_its_gameweek(self):
        """It answers "how many do I have now"; carrying it forward would be a
        stale number presented as a stated one."""
        state = {_ft_override_key(1, 3): 1}
        got = resolve_free_transfers(_history({1: 0, 2: 0, 3: 0}), {}, 4,
                                     team_id=1, _state=state)
        assert got["source"] == "replay"
