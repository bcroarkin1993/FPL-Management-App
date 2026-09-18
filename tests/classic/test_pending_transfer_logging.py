"""Logging a transfer the FPL API has not confirmed yet.

`_add_pending_local()` stored whatever it was handed. A mis-click was persisted
as fact, `apply_pending_transfers()` replayed it onto the squad, and the first
thing to notice was `check_resolved_squad()` reporting an illegal squad --
several steps removed from the typo that caused it.

Checking at the point of entry says what is actually wrong, and every rule here
is one FPL enforces on a real transfer.
"""

import pytest

from scripts.classic.transfers import _validate_pending_transfer


def _el(pid, name, etype, team, cost):
    return {"id": pid, "web_name": name, "element_type": etype,
            "team": team, "now_cost": cost}


@pytest.fixture
def squad():
    """Fifteen players: three from club 1, the rest spread out."""
    els = {}
    picks = []
    for i in range(15):
        etype = 1 if i < 2 else 2 if i < 7 else 3 if i < 12 else 4
        team = 1 if i in (0, 5, 10) else 20 + i
        els[i] = _el(i, "P%d" % i, etype, team, 50)
        picks.append({"element": i, "position": i + 1})
    # Free agents
    els[100] = _el(100, "Cheap MID", 3, 30, 45)
    els[101] = _el(101, "Rich MID", 3, 31, 140)
    els[102] = _el(102, "A Forward", 4, 32, 70)
    els[103] = _el(103, "Club 1 MID", 3, 1, 60)
    return picks, els


class TestLegalSwaps:
    def test_a_like_for_like_swap_within_budget_passes(self, squad):
        picks, els = squad
        assert _validate_pending_transfer(els[7], els[100], picks, els, bank=0) is None


class TestPositionMustMatch:
    def test_a_midfielder_cannot_be_replaced_by_a_forward(self, squad):
        """FPL's squad is 2/5/5/3; a cross-position swap makes it illegal."""
        picks, els = squad
        problem = _validate_pending_transfer(els[7], els[102], picks, els, bank=500)
        assert problem and "midfielder" in problem and "forward" in problem


class TestBudget:
    def test_an_unaffordable_player_is_rejected(self, squad):
        picks, els = squad
        problem = _validate_pending_transfer(els[7], els[101], picks, els, bank=0)
        assert problem and "£14.0m" in problem

    def test_the_outgoing_player_funds_the_incoming_one(self, squad):
        """Budget is bank plus what the sale releases, not bank alone."""
        picks, els = squad
        assert _validate_pending_transfer(els[7], els[101], picks, els, bank=900) is None


class TestClubLimit:
    def test_a_fourth_player_from_one_club_is_rejected(self, squad):
        picks, els = squad
        problem = _validate_pending_transfer(els[7], els[103], picks, els, bank=500)
        assert problem and "same club" in problem

    def test_selling_from_that_club_makes_room(self, squad):
        """The limit is counted after the outgoing player has left."""
        picks, els = squad
        assert _validate_pending_transfer(els[10], els[103], picks, els, bank=500) is None


class TestDegenerateInput:
    def test_the_same_player_on_both_sides_is_rejected(self, squad):
        picks, els = squad
        assert _validate_pending_transfer(els[7], els[7], picks, els, bank=0)

    def test_a_player_already_owned_is_rejected(self, squad):
        picks, els = squad
        problem = _validate_pending_transfer(els[7], els[8], picks, els, bank=0)
        assert problem and "already in your squad" in problem

    def test_an_unknown_player_is_rejected_rather_than_crashing(self, squad):
        picks, els = squad
        assert _validate_pending_transfer({}, els[100], picks, els, bank=0)
        assert _validate_pending_transfer(els[7], {}, picks, els, bank=0)
