"""The My Week checklist's builders.

The rule under test throughout is the one a checklist uniquely gets wrong: an
all-clear has to be *derived*. Every builder must answer "unknown" when its
inputs are missing rather than returning the reassuring value, because the two
render identically and are opposite claims.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts.common import week_check as wc
from scripts.common.data_validation import check_week_plan

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)


class _Resolution:
    """The subset of SquadResolution the builders read."""

    def __init__(self, **kw):
        self.ok = kw.get("ok", True)
        self.picks = kw.get("picks", [{"element": i} for i in range(1, 16)])
        self.source = kw.get("source", "my_team")
        self.source_gw = kw.get("source_gw", 12)
        self.target_gw = kw.get("target_gw", 12)
        self.is_stale = kw.get("is_stale", False)
        self.active_chip = kw.get("active_chip")
        self.provenance = kw.get("provenance", "live from your account")


class TestUnknownIsNeverOk:
    """Each builder, starved of its input."""

    @pytest.mark.parametrize("item", [
        wc.build_deadline_item(None, 12),
        wc.build_squad_item(None),
        wc.build_availability_item(None),
        wc.build_fixtures_item(None, [1, 2], [12]),
        wc.build_fixtures_item(pd.DataFrame(), [1, 2], [12]),
        wc.build_lineup_item(None),
        wc.build_transfers_item(None),
        wc.build_chips_item(None),
        wc.build_prices_item(None),
    ])
    def test_a_missing_input_is_unknown_and_carries_no_evidence(self, item):
        assert item.status == wc.STATUS_UNKNOWN
        assert not item.evidence
        assert item.verdict.strip()

    def test_an_unresolvable_squad_is_unknown(self):
        assert wc.build_squad_item(_Resolution(ok=False)).status == wc.STATUS_UNKNOWN


class TestDeadline:
    def test_a_deadline_today_is_a_warning(self):
        item = wc.build_deadline_item([NOW + timedelta(hours=5)], 12, now=NOW)
        assert item.status == wc.STATUS_WARN
        assert "5 hours" in item.verdict

    def test_a_deadline_next_week_is_fine(self):
        item = wc.build_deadline_item([NOW + timedelta(days=6)], 12, now=NOW)
        assert item.status == wc.STATUS_OK
        assert item.evidence["hours_left"] > wc.DEADLINE_SOON_HOURS

    def test_past_deadlines_are_skipped(self):
        item = wc.build_deadline_item(
            [NOW - timedelta(days=3), NOW + timedelta(days=2)], 12, now=NOW)
        assert item.evidence["hours_left"] == pytest.approx(48.0)

    def test_a_live_gameweek_says_so_rather_than_counting_down(self):
        item = wc.build_deadline_item([NOW + timedelta(days=4)], 12,
                                      gameweek_started=True, now=NOW)
        assert item.status == wc.STATUS_OK
        assert "under way" in item.verdict
        assert "Nothing can be changed" in item.detail

    def test_a_finished_calendar_is_unknown_not_ok(self):
        item = wc.build_deadline_item([NOW - timedelta(days=1)], 38, now=NOW)
        assert item.status == wc.STATUS_UNKNOWN


class TestSquad:
    def test_a_live_squad_is_ok(self):
        item = wc.build_squad_item(_Resolution())
        assert item.status == wc.STATUS_OK
        assert item.evidence["players"] == 15

    def test_a_stale_squad_warns_rather_than_failing(self):
        """The normal state between gameweeks — readable, but disclosed."""
        item = wc.build_squad_item(_Resolution(is_stale=True, source_gw=11,
                                               source="picks"))
        assert item.status == wc.STATUS_WARN
        assert "GW11" in item.verdict

    def test_an_active_chip_is_surfaced(self):
        item = wc.build_squad_item(_Resolution(active_chip="wildcard"))
        assert "wildcard" in item.detail


class TestAvailability:
    def _squad(self):
        return pd.DataFrame({"Player_Code": [100, 200, 300],
                             "Display_Name": ["Fit Man", "Doubt", "Injured"]})

    def _avail(self, **over):
        rows = [
            {"Player_Code": 100, "Status": "a", "PlayPct": 100, "News": ""},
            {"Player_Code": 200, "Status": "d", "PlayPct": 50,
             "News": "Knock - 50% chance of playing"},
            {"Player_Code": 300, "Status": "i", "PlayPct": 0,
             "News": "Hamstring injury - Expected back 26 Nov"},
        ]
        return pd.DataFrame(rows)

    def test_a_fit_squad_is_an_empty_list_not_none(self):
        """Empty means "checked, all fine"; None means "could not check"."""
        avail = pd.DataFrame({"Player_Code": [100], "Status": ["a"],
                              "PlayPct": [100], "News": [""]})
        flags = wc.squad_availability(
            pd.DataFrame({"Player_Code": [100], "Display_Name": ["Fit"]}), avail)
        assert flags == []
        item = wc.build_availability_item(flags)
        assert item.status == wc.STATUS_OK
        assert item.evidence["checked"] is True

    def test_it_reports_duration_and_whose_authority(self):
        flags = wc.squad_availability(self._squad(), self._avail(), now=NOW)
        by_name = {f["name"]: f for f in flags}
        assert set(by_name) == {"Doubt", "Injured"}
        assert by_name["Injured"]["unavailable"] is True
        assert by_name["Injured"]["stated"] is True
        assert by_name["Doubt"]["stated"] is False

    def test_an_out_player_needs_action_a_doubt_only_a_warning(self):
        flags = wc.squad_availability(self._squad(), self._avail(), now=NOW)
        assert wc.build_availability_item(flags).status == wc.STATUS_ACT
        doubt_only = [f for f in flags if not f["unavailable"]]
        assert wc.build_availability_item(doubt_only).status == wc.STATUS_WARN

    def test_no_shared_id_column_is_unknown_rather_than_a_name_match(self):
        squad = pd.DataFrame({"Name": ["Fit Man"]})
        assert wc.squad_availability(squad, self._avail()) is None

    def test_the_hedge_is_rendered_as_a_hedge(self):
        flags = wc.squad_availability(self._squad(), self._avail(), now=NOW)
        inferred = [f for f in flags if not f["stated"] and f["out_for"]]
        for flag in inferred:
            assert "inferred" in wc._absence_phrase(flag)


class TestFixtures:
    def _counts(self, mapping):
        return pd.DataFrame(mapping).T

    def test_nothing_scheduled_says_how_far_it_looked(self):
        counts = pd.DataFrame({12: [1, 1], 13: [1, 1]}, index=[1, 2])
        item = wc.build_fixtures_item(counts, [1, 2], [12, 13], 12, {1: "ARS"})
        assert item.status == wc.STATUS_OK
        assert "through GW13" in item.verdict

    def test_a_blank_in_the_gameweek_being_picked_needs_action(self):
        counts = pd.DataFrame({12: [0, 1], 13: [1, 1]}, index=[1, 2])
        item = wc.build_fixtures_item(counts, [1, 2], [12, 13], 12, {1: "ARS"})
        assert item.status == wc.STATUS_ACT
        assert "ARS" in item.verdict

    def test_a_blank_later_is_information_not_an_action(self):
        counts = pd.DataFrame({12: [1, 1], 13: [0, 1]}, index=[1, 2])
        item = wc.build_fixtures_item(counts, [1, 2], [12, 13], 12, {1: "ARS"})
        assert item.status == wc.STATUS_WARN

    def test_a_double_is_reported(self):
        counts = pd.DataFrame({12: [2, 1]}, index=[1, 2])
        item = wc.build_fixtures_item(counts, [1, 2], [12], 12, {1: "ARS"})
        assert "double" in item.verdict

    def test_only_your_own_clubs_count(self):
        counts = pd.DataFrame({12: [0, 1]}, index=[1, 2])
        item = wc.build_fixtures_item(counts, [2], [12], 12, {2: "CHE"})
        assert item.status == wc.STATUS_OK


class TestTransfers:
    def test_the_count_and_the_lane_are_both_reported(self):
        item = wc.build_transfers_item({"count": 2, "source": "my_team"}, bank=1.3)
        assert item.status == wc.STATUS_OK
        assert "2 free transfers" in item.verdict and "£1.3m" in item.verdict
        assert item.evidence["source"] == "my_team"

    def test_a_reconstructed_count_says_so(self):
        item = wc.build_transfers_item({"count": 1, "source": "replay"})
        assert "Reconstructed" in item.detail

    def test_the_cap_is_a_warning_because_accrual_is_forfeited(self):
        item = wc.build_transfers_item({"count": 5, "source": "replay"})
        assert item.status == wc.STATUS_WARN
        assert "cap" in item.detail

    def test_a_worthwhile_plan_promotes_the_row(self):
        item = wc.build_transfers_item({"count": 2, "source": "my_team"},
                                       plan={"gain_net": 6.9, "n_changes": 2})
        assert item.status == wc.STATUS_ACT
        assert "+6.9" in item.verdict

    def test_a_trivial_plan_does_not(self):
        item = wc.build_transfers_item({"count": 2, "source": "my_team"},
                                       plan={"gain_net": 0.4, "n_changes": 1})
        assert item.status == wc.STATUS_OK
        assert "no transfer gains" in item.verdict


class TestLineupAndCaptain:
    def test_a_disagreement_on_a_live_squad_is_actionable(self):
        item = wc.build_lineup_item(52.4, "Haaland", 9.1, "Saka", comparable=True)
        assert item.status == wc.STATUS_ACT
        assert "Saka" in item.detail and "Haaland" in item.detail

    def test_a_stale_squad_never_recommends_a_captain_change(self):
        """Its armband is last deadline's, so the disagreement is with a choice
        the manager may already have revised."""
        item = wc.build_lineup_item(52.4, "Haaland", 9.1, "Saka", comparable=False)
        assert item.status == wc.STATUS_OK
        assert "not compared" in item.detail
        assert "Haaland" in item.verdict

    def test_agreement_is_quiet(self):
        item = wc.build_lineup_item(52.4, "Haaland", 9.1, "Haaland", comparable=True)
        assert item.status == wc.STATUS_OK
        assert not item.detail


class TestChips:
    class _State:
        def __init__(self, gws_left=None, next_window_opens=None, used_in=()):
            self.gws_left = gws_left
            self.next_window_opens = next_window_opens
            self.used_in = used_in
            self.playable_now = gws_left is not None

    def test_an_expiring_chip_is_a_warning(self):
        status = {"available": ["bboost"], "available_later": [],
                  "states": {"bboost": self._State(gws_left=2)}}
        item = wc.build_chips_item(status, {}, {"bboost": "Bench Boost"})
        assert item.status == wc.STATUS_WARN
        assert "2 gameweeks" in item.verdict

    def test_a_long_window_is_quiet(self):
        status = {"available": ["bboost"], "available_later": [],
                  "states": {"bboost": self._State(gws_left=9)}}
        assert wc.build_chips_item(status).status == wc.STATUS_OK

    def test_a_valuable_chip_this_week_is_named(self):
        status = {"available": ["bboost"], "available_later": [],
                  "states": {"bboost": self._State(gws_left=9)}}
        item = wc.build_chips_item(status, {"bboost": 7.4},
                                   {"bboost": "Bench Boost"})
        assert "7.4" in item.detail

    def test_nothing_playable_is_ok_with_evidence(self):
        status = {"available": [], "available_later": ["wildcard"],
                  "states": {"wildcard": self._State(next_window_opens=20)}}
        item = wc.build_chips_item(status)
        assert item.status == wc.STATUS_OK
        assert item.evidence["available_later"] == ["wildcard"]


class TestPrices:
    def test_a_faller_you_own_is_a_warning(self):
        item = wc.build_prices_item({"tracked": 15, "rising": 0, "falling": 1,
                                     "priced": 15,
                                     "nearest": [{"name": "DCL", "pct": -142.0,
                                                  "state": "fall", "hourly": -27000}]})
        assert item.status == wc.STATUS_WARN
        assert "DCL" in item.detail

    def test_the_nearest_is_shown_even_when_nothing_has_crossed(self):
        item = wc.build_prices_item({"tracked": 15, "rising": 0, "falling": 0,
                                     "priced": 15,
                                     "nearest": [{"name": "DCL", "pct": -95.0,
                                                  "state": "steady", "hourly": None}]})
        assert item.status == wc.STATUS_OK
        assert "-95%" in item.detail


class TestTheChecklistItself:
    def _full(self):
        return wc.build_week_checklist(
            deadline=wc.build_deadline_item([NOW + timedelta(days=4)], 12, now=NOW),
            squad=wc.build_squad_item(_Resolution()),
            availability=wc.build_availability_item([]),
            fixtures=wc.build_fixtures_item(
                pd.DataFrame({12: [1]}, index=[1]), [1], [12], 12, {1: "ARS"}),
            lineup=wc.build_lineup_item(52.4, "Haaland", 9.1, "Haaland", True),
            transfers=wc.build_transfers_item({"count": 1, "source": "my_team"}, 0.5),
            chips_item=wc.build_chips_item(
                {"available": [], "available_later": [], "states": {}}),
            prices=wc.build_prices_item({"tracked": 15, "rising": 0,
                                         "falling": 0, "priced": 15,
                                         "nearest": []}),
        )

    def test_it_renders_every_item_in_the_declared_order(self):
        items = self._full()
        assert [i.key for i in items] == list(wc.ITEM_ORDER)

    def test_every_builder_is_ordered(self):
        """A builder whose key is absent from ITEM_ORDER would be silently
        dropped from the page, which is the one thing worse than an unknown."""
        keys = {i.key for i in self._full()}
        assert keys <= set(wc.ITEM_ORDER)
        assert len(wc.ITEM_ORDER) == len(set(wc.ITEM_ORDER))

    def test_a_clean_week_passes_its_own_validation(self):
        """The cross-check on the whole rule: no builder may reach `ok` without
        recording what it derived that from."""
        issues = check_week_plan(self._full(), squad_size=15, xi_size=11)
        assert [i for i in issues if i.severity == "error"] == []

    def test_unknown_outranks_ok_in_the_headline(self):
        items = [wc.build_squad_item(_Resolution()), wc.build_availability_item(None)]
        assert wc.worst_status(items) == wc.STATUS_UNKNOWN

    def test_attention_is_the_urgent_rows_first(self):
        items = [
            wc.build_squad_item(_Resolution(is_stale=True)),          # warn
            wc.build_availability_item([{"unavailable": True, "name": "X",
                                         "out_for": 3, "stated": True,
                                         "play_pct": 0, "news": ""}]),  # act
            wc.build_squad_item(_Resolution()),                        # ok
        ]
        flagged = wc.attention_items(items)
        assert [i.status for i in flagged] == [wc.STATUS_ACT, wc.STATUS_WARN]

    def test_an_omitted_builder_does_not_fabricate_a_row(self):
        items = wc.build_week_checklist(squad=wc.build_squad_item(_Resolution()))
        assert [i.key for i in items] == ["squad"]
