"""The Draft checklist: a different round, and three rows Classic has no analogue for.

Draft has no chips, no prices and no captain (`settings.squad.captains_disabled`
is true), and in their place sit the questions a Draft week actually turns on:
which transaction window is open, whether a claim is worth making, and whether
any of your players are tied up in a trade.

The rule the whole module exists for applies unchanged here: **an all-clear
must be derived, never defaulted**. "Nothing to do" and "we could not check"
render identically and are opposite claims, so every builder below returns
`unknown` -- carrying no evidence -- when its inputs are missing.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.common import week_check as W
from scripts.common.data_validation import check_week_plan

NOW = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)
SOON = NOW + timedelta(hours=6)
LATER = NOW + timedelta(days=4)


class TestUnknownIsNeverOk:
    @pytest.mark.parametrize("item", [
        W.build_draft_squad_item(None),
        W.build_draft_lineup_item(None),
        W.build_waivers_item(None),
        W.build_waivers_item({}),
        W.build_claim_item(checked=False),
        W.build_trades_item(None),
        W.build_trades_item({}),
    ], ids=["squad", "lineup", "waivers-none", "waivers-empty",
            "claim-unchecked", "trades-none", "trades-empty"])
    def test_a_missing_input_is_unknown_and_carries_no_evidence(self, item):
        assert item.status == W.STATUS_UNKNOWN
        assert item.evidence == {}
        assert item.verdict

    def test_unknown_outranks_ok_in_the_headline(self):
        ok = W.build_draft_squad_item(pd.DataFrame({"Player": ["a"] * 15}))
        assert W.worst_status([ok, W.build_claim_item(checked=False)]) == W.STATUS_UNKNOWN


class TestYourSquad:
    def test_it_never_warns_about_staleness(self):
        """A Draft roster is public and current -- element-status plus accepted
        transactions -- so there is no last-deadline caveat to carry."""
        item = W.build_draft_squad_item(pd.DataFrame({"Player": ["a"] * 15}), "Mine")
        assert item.status == W.STATUS_OK
        assert item.evidence["players"] == 15
        assert "stale" not in item.verdict.lower()

    def test_locked_players_are_mentioned_as_context(self):
        item = W.build_draft_squad_item(pd.DataFrame({"Player": ["a"] * 15}), "Mine", 4)
        assert "4 players" in item.detail and "waiver round" in item.detail
        assert item.status == W.STATUS_OK, "league-wide locks are not your task"


class TestTheXiYouHaveSet:
    def test_a_better_bench_player_is_an_action_naming_the_swap(self):
        item = W.build_draft_lineup_item(
            62.0, 58.0, swap={"in_name": "Bench Guy", "out_name": "Starter"})
        assert item.status == W.STATUS_ACT
        assert "Bench Guy" in item.verdict and "Starter" in item.verdict
        assert item.evidence["gain"] == pytest.approx(4.0)

    def test_an_optimal_xi_is_quiet(self):
        item = W.build_draft_lineup_item(60.0, 60.0)
        assert item.status == W.STATUS_OK and item.evidence

    def test_a_sub_floor_gain_is_ignored(self):
        item = W.build_draft_lineup_item(60.4, 60.0,
                                         swap={"in_name": "A", "out_name": "B"})
        assert item.status == W.STATUS_OK

    def test_bench_order_warns_rather_than_acts(self):
        """It costs points only if a starter does not play, so it is a
        conditional loss -- and keeping it a warning is what leaves
        `check_week_plan`'s "an act must name a gain" rule intact."""
        item = W.build_draft_lineup_item(
            60.0, 60.0,
            bench_issue={"better_name": "Good Sub", "better_proj": 4.2,
                         "ahead_name": "Weak Sub", "ahead_proj": 1.1})
        assert item.status == W.STATUS_WARN
        assert "Good Sub" in item.verdict and "Weak Sub" in item.verdict
        assert check_week_plan([item]) == [] or all(
            i.severity != "error" for i in check_week_plan([item]))

    def test_an_xi_fault_outranks_a_bench_fault(self):
        item = W.build_draft_lineup_item(
            62.0, 58.0, swap={"in_name": "A", "out_name": "B"},
            bench_issue={"better_name": "C", "better_proj": 1.0,
                         "ahead_name": "D", "ahead_proj": 0.5})
        assert item.status == W.STATUS_ACT

    def test_there_is_no_captain_anywhere(self):
        """Draft disables captains; a row mentioning one would be inventing a
        mechanic the format does not have."""
        item = W.build_draft_lineup_item(60.0, 55.0,
                                         swap={"in_name": "A", "out_name": "B"})
        assert "captain" not in item.verdict.lower()
        assert "captain" not in (item.detail or "").lower()
        assert "captain" not in item.evidence and "set_captain" not in item.evidence

    def test_no_confirm_flag_exists(self):
        """Classic needs one because it cannot see a pre-deadline lineup.
        Draft publishes picks without credentials, so there is nothing to be
        told -- and a confirm button here would let a real fault be dismissed."""
        import inspect
        assert "confirmed" not in inspect.signature(W.build_draft_lineup_item).parameters


class TestWhichWindowIsOpen:
    def test_waivers_leads_with_your_pick(self):
        item = W.build_waivers_item({"mode": "waivers"}, LATER, 6, 5, 0, now=NOW)
        assert item.status == W.STATUS_OK
        assert "pick 6" in item.verdict and item.evidence["waiver_pick"] == 6

    def test_free_agency_quotes_no_queue(self):
        """There is nobody to be beaten to a player by, so a pick number there
        describes a mechanic that is not running."""
        item = W.build_waivers_item({"mode": "free-agency"}, LATER, 6, 5, 0, now=NOW)
        assert item.evidence["free_agency"] is True
        assert "pick" not in item.verdict.lower()
        assert "immediately" in item.verdict

    def test_free_agency_still_warns_about_locked_players(self):
        item = W.build_waivers_item({"mode": "free-agency"}, LATER, None, None, 2, now=NOW)
        assert "locked" in item.detail.lower()

    def test_an_imminent_deadline_warns(self):
        item = W.build_waivers_item({"mode": "waivers"}, SOON, 1, 0, 0, now=NOW)
        assert item.status == W.STATUS_WARN
        assert item.evidence["hours_left"] == pytest.approx(6.0, abs=0.01)

    def test_an_unreadable_mode_is_unknown_not_free_agency(self):
        """An outage must not be indistinguishable from the permissive state --
        the rule `_build_waiver_context` already follows."""
        item = W.build_waivers_item({"mode": None}, LATER, now=NOW)
        assert item.status == W.STATUS_UNKNOWN and item.evidence == {}


class TestIsAClaimWorthMaking:
    def test_unchecked_is_unknown_and_says_so(self):
        item = W.build_claim_item(checked=False)
        assert item.status == W.STATUS_UNKNOWN
        assert item.evidence == {}
        assert "not checked" in item.verdict.lower()

    def test_a_search_that_found_nothing_is_a_derived_all_clear(self):
        """The distinction the whole module turns on: this `ok` has evidence
        behind it, where the unchecked row above has none."""
        item = W.build_claim_item(claim=None, checked=True, pool_size=96)
        assert item.status == W.STATUS_OK
        assert item.evidence["checked"] is True and item.evidence["pool"] == 96
        assert check_week_plan([item]) == [] or all(
            i.severity != "error" for i in check_week_plan([item]))

    def test_a_worthwhile_claim_is_an_action_naming_both_players(self):
        item = W.build_claim_item(
            {"add_player": "New Guy", "drop_player": "Old Guy",
             "points_gain": 3.4, "lane": "hold", "outlook_band": "likely"},
            checked=True, pool_size=96)
        assert item.status == W.STATUS_ACT
        assert "New Guy" in item.verdict and "Old Guy" in item.verdict
        assert item.evidence["best_gain"] == pytest.approx(3.4)

    def test_a_trivial_gain_stays_quiet(self):
        item = W.build_claim_item(
            {"add_player": "A", "drop_player": "B", "points_gain": 0.4},
            checked=True, pool_size=96)
        assert item.status == W.STATUS_OK
        assert "not worth" in item.verdict.lower()

    def test_a_failed_search_is_unknown_with_its_reason(self):
        item = W.build_claim_item(checked=True, reason="Projections unavailable.")
        assert item.status == W.STATUS_UNKNOWN
        assert item.verdict == "Projections unavailable."
        assert item.evidence == {}

    def test_the_lane_is_named(self):
        hold = W.build_claim_item({"add_player": "A", "drop_player": "B",
                                   "points_gain": 3.0, "lane": "hold"},
                                  checked=True)
        rent = W.build_claim_item({"add_player": "A", "drop_player": "B",
                                   "points_gain": 3.0, "lane": "rent"},
                                  checked=True)
        assert "rest of season" in hold.detail
        assert "just this week" in rent.detail


class TestTrades:
    def test_trades_off_is_a_quiet_fact(self):
        item = W.build_trades_item({"trades": "n"})
        assert item.status == W.STATUS_OK and item.evidence["enabled"] is False

    def test_your_players_in_an_accepted_trade_warn(self):
        item = W.build_trades_item({"trades": "a"}, LATER, pending_mine=2, now=NOW)
        assert item.status == W.STATUS_WARN
        assert "2 of your players" in item.verdict

    def test_an_imminent_deadline_warns(self):
        item = W.build_trades_item({"trades": "a"}, SOON, now=NOW)
        assert item.status == W.STATUS_WARN

    def test_an_assumed_approval_regime_says_so(self):
        """An unreadable setting resolves to the *earlier* deadline, and the
        row discloses that rather than presenting it as the league's rule."""
        item = W.build_trades_item({"trades": "a"}, LATER, approval_assumed=True, now=NOW)
        assert "assumes approval" in item.detail


class TestTheDraftChecklistItself:
    def _clean(self):
        return W.build_draft_week_checklist(
            deadline=W.build_deadline_item([LATER], 11, now=NOW),
            opponent=W.build_opponent_item("Rival", 50.0, 44.0, 0.61,
                                           section=W.SECTION_DRAFT),
            squad=W.build_draft_squad_item(pd.DataFrame({"Player": ["a"] * 15}), "Mine"),
            availability=W.build_availability_item([]),
            fixtures=W.build_fixtures_item(
                pd.DataFrame({"team_id": [1], "gameweek": [11], "fixtures": [1]}),
                {1}, [11]),
            lineup=W.build_draft_lineup_item(60.0, 60.0),
            waivers=W.build_waivers_item({"mode": "waivers"}, LATER, 6, 5, 0, now=NOW),
            claim=W.build_claim_item(claim=None, checked=True, pool_size=96),
            trades=W.build_trades_item({"trades": "a"}, LATER, now=NOW),
        )

    def test_it_renders_in_order(self):
        assert [i.key for i in self._clean()] == list(W.DRAFT_ITEM_ORDER)

    def test_a_clean_draft_week_passes_its_own_validation(self):
        issues = check_week_plan(self._clean(), squad_size=15, xi_size=11)
        assert [i for i in issues if i.severity == "error"] == []

    def test_an_omitted_builder_fabricates_no_row(self):
        items = W.build_draft_week_checklist(
            deadline=W.build_deadline_item([LATER], 11, now=NOW))
        assert [i.key for i in items] == ["deadline"]

    def test_it_has_no_chip_or_price_row(self):
        """Draft has neither. A shared order would put them here."""
        assert "chips" not in W.DRAFT_ITEM_ORDER
        assert "prices" not in W.DRAFT_ITEM_ORDER
        assert "transfers" not in W.DRAFT_ITEM_ORDER

    def test_classic_order_is_untouched(self):
        assert W.ITEM_ORDER == ("deadline", "opponent", "squad", "availability",
                                "fixtures", "lineup", "transfers", "chips", "prices")


class TestNavigationTargets:
    """A renamed page does not raise -- it silently turns every Open button
    into a no-op. Same guard as the Classic checklist's."""

    def test_every_draft_item_points_at_a_real_page(self):
        import main
        for item in TestTheDraftChecklistItself()._clean():
            if not item.page:
                continue
            assert item.section in main.SECTIONS, (item.key, item.section)
            assert item.page in main.SECTIONS[item.section], (item.key, item.page)

    def test_every_row_in_the_order_is_covered_by_this_guard(self):
        covered = {i.key for i in TestTheDraftChecklistItself()._clean()}
        assert covered == set(W.DRAFT_ITEM_ORDER), (
            "a row was added to DRAFT_ITEM_ORDER without a navigation check")

    def test_the_draft_section_label_matches_main(self):
        import main
        assert W.SECTION_DRAFT in main.SECTIONS
