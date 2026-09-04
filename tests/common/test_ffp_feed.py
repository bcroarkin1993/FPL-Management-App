"""Offline tests for the Fantasy Football Pundit feed.

These pin the three things that made FFP wrong in the app and would be silent
if they broke again: which gameweek a table is for, which of the two prediction
columns is conditional on starting, and where a cumulative window starts.
"""

import json

import pandas as pd
import pytest

from scripts.common import ffp_feed


def _flight_script(payload: str) -> str:
    """Wrap a payload the way Next.js serialises it into the page."""
    return '<script>self.__next_f.push([1,%s])</script>' % json.dumps(payload)


def _row(gw, code, web, first, second, element_type, team, opp,
         cond, start_pct, is_home=True):
    return {
        "gw": gw, "player_code": code, "web_name": web,
        "first_name": first, "second_name": second,
        "position": "Midfielder", "element_type": element_type,
        "team_name": team, "team_short": team[:3].upper(), "team_abbr": team[:3].upper(),
        "price": "9.5", "selected_by_percent": "10.80",
        "predicted_points": "%.3f" % cond,
        "predicted_points_start": "%.3f" % (cond * start_pct / 100.0),
        "start_pct": "%.2f" % start_pct, "fixture_count": 1, "source": "odds",
        "opponent_abbr": opp, "is_home": is_home, "difficulty": 3,
    }


@pytest.fixture
def payload_rows():
    """Two players over GW3-GW5, with a per-gameweek projection each."""
    rows = []
    for gw, saka, white in ((3, 5.889, 4.563), (4, 6.089, 5.163), (5, 5.500, 4.100)):
        rows.append(_row(gw, 223340, "Saka", "Bukayo", "Saka", 3, "Arsenal", "CHE", saka, 90))
        rows.append(_row(gw, 198869, "White", "Benjamin", "White", 2, "Arsenal", "CHE", white, 80))
    return rows


class TestFlightExtraction:
    def test_rows_survive_the_rsc_escaping(self, payload_rows):
        page = (
            "<html>"
            + _flight_script('13:["$","div",null,{"children":"noise"}]\n')
            + _flight_script('14:["$","$L1d",null,{"rows":%s}]\n' % json.dumps(payload_rows))
            + "</html>"
        )
        rows = ffp_feed.extract_rows(ffp_feed.flight_payload(page))
        assert len(rows) == len(payload_rows)
        assert rows[0]["web_name"] == "Saka"

    def test_a_page_with_no_payload_returns_nothing_rather_than_raising(self):
        assert ffp_feed.flight_payload("<html>nothing here</html>") == ""
        assert ffp_feed.extract_rows("") == []
        assert ffp_feed.extract_rows('"rows":[ not json') == []

    def test_updated_stamp_is_read_with_its_gameweek(self):
        gw, when = ffp_feed.parse_updated('x "Updated for GW",3," · 4 September at 16:18" y')
        assert gw == 3
        assert when is not None
        assert (when.day, when.hour, when.minute) == (4, 16, 18)

    def test_a_missing_stamp_is_not_an_error(self):
        assert ffp_feed.parse_updated("no stamp here") == (None, None)


class TestSheetSchema:
    def test_the_conditional_column_is_the_larger_one(self, payload_rows):
        """The migration's worst case, in one assertion.

        FFP's site calls the conditional value `predicted_points` and the
        start-weighted one `predicted_points_start` -- the opposite way round
        from the spreadsheet's `StartingPredicted` / `Predicted`. Mapping them
        across by name rather than by basis charges the start probability twice.
        """
        df = ffp_feed.to_sheet_schema(payload_rows, gw=3)
        assert (df["StartingPredicted"] >= df["Predicted"]).all()
        implied = df["StartingPredicted"] * df["Start"] / 100.0
        assert (df["Predicted"] - implied).abs().max() < 0.01

    def test_forward_gameweeks_are_relative_offsets(self, payload_rows):
        """`GW2` means the second week of the window, not gameweek 2."""
        df = ffp_feed.to_sheet_schema(payload_rows, gw=3).set_index("Web_Name")
        assert df.at["Saka", "GW2"] == pytest.approx(6.089)   # the GW4 row
        assert df.at["Saka", "GW3"] == pytest.approx(5.500)   # the GW5 row

    def test_cumulative_windows_include_the_current_gameweek(self, payload_rows):
        df = ffp_feed.to_sheet_schema(payload_rows, gw=3).set_index("Web_Name")
        saka = df.loc["Saka"]
        assert saka["Next2GWsStart"] == pytest.approx(saka["StartingPredicted"] + saka["GW2"], abs=0.01)
        assert saka["Next2GWs"] == pytest.approx(saka["Predicted"] + saka["GW2s"], abs=0.01)
        assert saka["Next3GWsStart"] == pytest.approx(
            saka["StartingPredicted"] + saka["GW2"] + saka["GW3"], abs=0.01)

    def test_long_start_is_not_invented(self, payload_rows):
        """The site has one start rate per player, so there is no long-run figure.

        Emitting a copy of `Start` would spend 10% of ROS on a signal 1GW
        already carries; leaving it out lets the FPL `starts` fallback stand.
        """
        df = ffp_feed.to_sheet_schema(payload_rows, gw=3)
        assert "LongStart" not in df.columns

    def test_element_ids_are_attached_when_the_map_is_available(self, payload_rows):
        df = ffp_feed.to_sheet_schema(payload_rows, gw=3, code_to_id={223340: 12, 198869: 10})
        assert set(df["Player_ID"].dropna()) == {12, 10}

    def test_fixture_string_matches_the_sheet_format(self, payload_rows):
        df = ffp_feed.to_sheet_schema(payload_rows, gw=3)
        assert set(df["Fixture"]) == {"CHE (H)"}

    def test_a_gameweek_with_no_rows_yields_nothing(self, payload_rows):
        assert ffp_feed.to_sheet_schema(payload_rows, gw=9).empty

    def test_an_empty_payload_is_survivable(self):
        assert ffp_feed.to_sheet_schema([]).empty
        assert ffp_feed.to_sheet_schema([{"nonsense": 1}]).empty


class TestGameweekResolution:
    """Ordered fixture pairs, because team *sets* cannot separate gameweeks.

    Every club plays every week, so the check this replaces scored ~0.95 for
    every gameweek and never fired.
    """

    FIXTURES = {
        2: {("AVL", "ARS"), ("LIV", "NFO"), ("BOU", "EVE"), ("COV", "HUL"), ("TOT", "NEW")},
        3: {("ARS", "CHE"), ("IPS", "LIV"), ("NEW", "BOU"), ("BRE", "SUN"), ("MCI", "COV")},
    }

    def _sheet(self, pairs_by_team):
        return pd.DataFrame([{"Team": t, "Fixture": f} for t, f in pairs_by_team.items()])

    def test_a_stale_sheet_resolves_to_the_week_it_actually_describes(self):
        sheet = self._sheet({
            "Arsenal": "Aston Villa (a)", "Liverpool": "Notts Forest (H)",
            "Bournemouth": "Everton (H)", "Coventry City": "Hull City (H)",
            "Spurs": "Newcastle (a)",
        })
        assert ffp_feed.resolve_ffp_gameweek(sheet, self.FIXTURES) == 2

    def test_a_current_sheet_resolves_to_the_current_week(self):
        sheet = self._sheet({
            "Arsenal": "Chelsea (H)", "Liverpool": "Ipswich Town (a)",
            "Newcastle": "Bournemouth (H)", "Brentford": "Sunderland (H)",
            "Man City": "Coventry City (H)",
        })
        assert ffp_feed.resolve_ffp_gameweek(sheet, self.FIXTURES) == 3

    def test_a_stated_gameweek_wins_without_a_fixture_lookup(self):
        df = pd.DataFrame({"FFP_GW": [7, 7, 7], "Team": ["Arsenal"] * 3,
                           "Fixture": ["Chelsea (H)"] * 3})
        assert ffp_feed.resolve_ffp_gameweek(df, self.FIXTURES) == 7

    def test_unrecognisable_fixtures_resolve_to_no_answer(self):
        """"Unknown" must never be reported as a gameweek."""
        sheet = self._sheet({"Arsenal": "Real Madrid (H)", "Chelsea": "Barcelona (a)"})
        assert ffp_feed.resolve_ffp_gameweek(sheet, self.FIXTURES) is None

    def test_a_club_label_the_app_does_not_know_is_skipped_not_guessed(self):
        """FFP writes "Notts Forest"; the bootstrap writes "Nott'm Forest"."""
        from scripts.common.text_helpers import TEAM_FULL_TO_SHORT
        assert TEAM_FULL_TO_SHORT.get("Notts Forest") == "NFO"

    def test_an_empty_frame_is_survivable(self):
        assert ffp_feed.resolve_ffp_gameweek(None) is None
        assert ffp_feed.resolve_ffp_gameweek(pd.DataFrame()) is None
