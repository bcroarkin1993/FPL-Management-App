"""Live checks on what FPL Draft publishes about its own rules and queues.

Three things this app now reads rather than infers, each of which would fail
silently if the payload moved:

* `league_entries[].waiver_pick` — the claim queue the Waiver Wire's reachability
  bands rest on entirely, and the per-position claim rates read out of the
  transaction log that decide where those bands fall.
* `events[].waivers_time` / `trades_time` — the real deadlines, replacing the
  hardcoded offsets.
* `settings` — the rulebook this codebase compiles in as constants.

Contract per tests/live/conftest.py: unreachable SKIPs, implausible FAILs, and a
bug in our own code FAILs rather than being filed as an outage.
"""

import pytest
import requests

import config
from scripts.common.data_validation import (
    check_draft_game_settings,
    check_position_claim_rates,
    check_waiver_order,
    format_issues,
)
from scripts.common.waiver_priority import (
    claim_history_stats,
    expected_taken_at_position,
    managers_ahead,
    my_waiver_pick,
    parse_waiver_order,
    position_claim_rates,
)
from tests.live.conftest import skip_if_unreachable

DRAFT_API = "https://draft.premierleague.com/api"


def _errors(issues):
    return [i for i in issues if i.severity == "error"]


@pytest.fixture(scope="module")
def bootstrap():
    def _fetch():
        r = requests.get(f"{DRAFT_API}/bootstrap-static", timeout=30)
        r.raise_for_status()
        return r.json()
    return skip_if_unreachable(_fetch, "Draft bootstrap")


@pytest.fixture(scope="module")
def league_details():
    league_id = getattr(config, "FPL_DRAFT_LEAGUE_ID", None)
    if not league_id:
        pytest.skip("No Draft league configured")

    def _fetch():
        r = requests.get(f"{DRAFT_API}/league/{league_id}/details", timeout=30)
        r.raise_for_status()
        return r.json()
    return skip_if_unreachable(_fetch, "Draft league details")


@pytest.fixture(scope="module")
def transactions():
    league_id = getattr(config, "FPL_DRAFT_LEAGUE_ID", None)
    if not league_id:
        pytest.skip("No Draft league configured")

    def _fetch():
        r = requests.get(f"{DRAFT_API}/draft/league/{league_id}/transactions", timeout=30)
        r.raise_for_status()
        return r.json().get("transactions", [])
    return skip_if_unreachable(_fetch, "Draft transactions")


class TestWaiverOrderIsPublished:
    def test_the_queue_is_a_permutation_covering_every_manager(self, league_details):
        order = parse_waiver_order(league_details)
        assert order, "waiver_pick vanished from league_entries — the claim outlook dies silently"
        issues = _errors(check_waiver_order(
            order,
            n_entries=len(league_details.get("league_entries", [])),
            standings=league_details.get("standings"),
        ))
        assert not issues, format_issues(issues)

    def test_this_manager_resolves_in_the_entry_id_space(self, league_details):
        """The two id spaces: entry_id keys transactions, id keys the standings.

        Crossing them yields an empty match rather than an error, which reads on
        the page as "you have no waiver pick" and disables the whole feature.
        """
        team_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
        if not team_id:
            pytest.skip("No Draft team configured")

        order = parse_waiver_order(league_details)
        pick = my_waiver_pick(order, team_id)
        assert pick is not None, (
            "FPL_DRAFT_TEAM_ID does not resolve against entry_id. Either the "
            "configured team is not in this league, or the id spaces were crossed."
        )
        assert 1 <= pick <= len(order)
        assert len(managers_ahead(order, team_id)) == pick - 1


class TestTransactionLogStillDecodes:
    def test_a_declined_in_claim_means_the_player_was_taken(self, transactions):
        """`di` is the contention signal; `do` is not, and must not be read as one.

        Verified 2026-09-24 at 58/58. If this ever drops materially, the meaning
        of the code has changed and claim_history_stats is counting the wrong rows.
        """
        waivers = [t for t in transactions if t.get("kind") == "w"]
        if not waivers:
            pytest.skip("No waiver claims in this league's history yet")

        accepted = {(t["event"], t.get("element_in"))
                    for t in waivers if t.get("result") == "a"}
        declined_in = [t for t in waivers if t.get("result") == "di"]
        if not declined_in:
            pytest.skip("No contested claims in this league's history yet")

        taken = sum(1 for t in declined_in if (t["event"], t.get("element_in")) in accepted)
        rate = taken / len(declined_in)
        assert rate >= 0.9, (
            f"only {taken}/{len(declined_in)} 'di' rows had their player accepted "
            f"elsewhere that gameweek. 'di' no longer means 'beaten on priority'."
        )

    def test_history_yields_a_usable_participation_rate(self, transactions, league_details):
        stats = claim_history_stats(
            transactions, n_managers=len(league_details.get("league_entries", []))
        )
        if not stats["events_observed"]:
            pytest.skip("No waiver rounds yet this season")
        assert 0.0 < stats["participation_rate"] <= 1.0
        assert stats["max_contention"] >= 1

    def test_the_recent_window_is_measurable_and_differs_from_the_season(
            self, transactions, league_details):
        """The bands read the recent rate, so it has to exist.

        They disagreed by 8 points live (63% against 71%) because the rounds
        just after a draft are churn. Equality is legitimate for a short log, so
        this asserts the figure is usable rather than that it is lower.
        """
        stats = claim_history_stats(
            transactions, n_managers=len(league_details.get("league_entries", []))
        )
        if not stats["events_observed"]:
            pytest.skip("No waiver rounds yet this season")
        assert stats["recent_events"], "a log with rounds in it must name a window"
        assert 0.0 < stats["participation_rate_recent"] <= 1.0

    def test_per_position_claim_rates_are_plausible(
            self, transactions, league_details, bootstrap):
        """The only input to the reachability bands, so it is the thing to check.

        A wrong rate mislabels every pill at that position while each still reads
        as a plausible opinion — the silent failure `check_position_claim_rates()`
        exists for. Measured 2026-10-22: MID 0.475, DEF 0.25, FWD 0.10, GK 0.025
        accepted claims per manager per round.
        """
        pos_by_code = {1: "G", 2: "D", 3: "M", 4: "F"}
        positions = {
            e.get("id"): pos_by_code.get(e.get("element_type"))
            for e in bootstrap.get("elements", [])
        }
        assert positions, "the Draft bootstrap published no elements"

        n_managers = len(league_details.get("league_entries", []))
        stats = claim_history_stats(transactions, n_managers=n_managers)
        if not stats["events_observed"]:
            pytest.skip("No waiver rounds yet this season")

        rates = position_claim_rates(transactions, positions, n_managers=n_managers)
        assert rates, (
            "no per-position claim rates could be measured from a log that has "
            "waiver rounds in it — the bands would silently abstain on every card"
        )
        issues = check_position_claim_rates(rates, stats["events_observed"])
        assert not _errors(issues), format_issues(issues)

        # Every position resolved, so a claim cannot have been dropped on the
        # floor: an unresolved element deflates every rate at once.
        accepted = [t for t in transactions
                    if t.get("kind") == "w" and t.get("result") == "a"
                    and t.get("event") in set(stats["recent_events"])]
        resolved = sum(1 for t in accepted if positions.get(t.get("element_in")))
        assert resolved == len(accepted), (
            f"{len(accepted) - resolved} of {len(accepted)} accepted claims in "
            f"the recent window name a player the bootstrap does not carry"
        )

    def test_the_expected_sweep_cannot_exceed_the_board(
            self, transactions, league_details, bootstrap):
        """`n_ahead x rate` is a count of players, and it has to be a small one.

        The old model called the top four forwards unreachable at a mid-queue
        pick; a figure that large again would mean the rates had broken, since no
        league takes four of one position per round.
        """
        pos_by_code = {1: "G", 2: "D", 3: "M", 4: "F"}
        positions = {
            e.get("id"): pos_by_code.get(e.get("element_type"))
            for e in bootstrap.get("elements", [])
        }
        entries = league_details.get("league_entries", [])
        n_managers = len(entries)
        rates = position_claim_rates(transactions, positions, n_managers=n_managers)
        if not rates:
            pytest.skip("No waiver rounds yet this season")

        n_ahead = max(0, n_managers - 1)   # the worst case: you pick last
        for pos in ("G", "D", "M", "F"):
            expected = expected_taken_at_position(pos, n_ahead, rates)
            assert expected is not None
            assert 0.0 <= expected <= n_managers, (
                f"{expected:.1f} {pos} expected gone before the last pick in a "
                f"{n_managers}-team league — the rates have lost a divisor"
            )


class TestPublishedDeadlines:
    def test_every_gameweek_states_its_own_waiver_and_trade_times(self, bootstrap):
        events = bootstrap["events"]["data"]
        missing = [e["id"] for e in events if not e.get("waivers_time")]
        assert not missing, f"gameweeks with no published waivers_time: {missing}"

    def test_the_offsets_the_app_falls_back_to_still_match(self, bootstrap):
        """waivers = deadline - 24h, trades = deadline - 48h, for all 38.

        If FPL ever varies these, the *published* path stays correct and only the
        fallback constants go stale — so this is the prompt to revisit them, not
        evidence the app is currently wrong.
        """
        from scripts.common.waiver_alerts import _iso_to_et

        off = []
        for e in bootstrap["events"]["data"]:
            deadline = _iso_to_et(e.get("deadline_time"))
            waivers = _iso_to_et(e.get("waivers_time"))
            trades = _iso_to_et(e.get("trades_time"))
            if not (deadline and waivers and trades):
                continue
            w_h = (deadline - waivers).total_seconds() / 3600
            t_h = (deadline - trades).total_seconds() / 3600
            if abs(w_h - 24) > 0.01 or abs(t_h - 48) > 0.01:
                off.append((e["id"], w_h, t_h))
        assert not off, (
            "gameweeks whose windows are not the standard 24h/48h: %r. The "
            "published path handles this; DRAFT_OFFSET_HOURS / "
            "TRADE_APPROVAL_LEAD_HOURS in waiver_alerts.py no longer do." % off
        )

    def test_the_resolver_reads_the_live_calendar(self, bootstrap):
        from scripts.common.waiver_alerts import published_deadlines

        gw = bootstrap["events"]["data"][0]["id"]
        resolved = published_deadlines(gw, events=bootstrap["events"]["data"])
        assert resolved and resolved["waivers"] is not None


class TestGameSettings:
    def test_the_rules_this_app_hardcodes_are_still_the_rules(self, bootstrap):
        issues = check_draft_game_settings(bootstrap.get("settings"))
        errors = _errors(issues)
        assert not errors, format_issues(errors)

    def test_draft_rank_is_published_for_most_of_the_game(self, bootstrap):
        """It is FPL's own board and the auto-pick order for an absent manager.

        The Draft Helper renders it beside ours; a payload that stopped carrying
        it would leave the column silently blank.
        """
        elements = bootstrap["elements"]
        ranked = [e for e in elements if e.get("draft_rank") is not None]
        assert len(ranked) / len(elements) > 0.9, (
            f"only {len(ranked)}/{len(elements)} elements carry a draft_rank"
        )
