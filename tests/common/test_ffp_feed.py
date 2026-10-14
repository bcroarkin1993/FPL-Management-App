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

    def test_the_common_name_is_available_alongside_the_legal_one(self):
        """FFP republishes the bootstrap's legal name, which nobody says aloud.

        `Name` has to stay the legal one -- every merge keys on it, and the
        projection sources are matched against the full name. So the readable
        version rides alongside it and the UI renders that.
        """
        rows = [_row(3, 100, "Thiago", "Igor Thiago", "Nascimento Rodrigues",
                     4, "Brentford", "SUN", 6.5, 90),
                _row(3, 101, "B.Fernandes", "Bruno", "Borges Fernandes",
                     3, "Man Utd", "EVE", 6.1, 90),
                _row(3, 102, "Gabriel", "Gabriel", "dos Santos Magalhães",
                     2, "Arsenal", "CHE", 4.5, 90)]
        df = ffp_feed.to_sheet_schema(rows, gw=3).set_index("Web_Name")
        assert df.at["Thiago", "Name"] == "Igor Thiago Nascimento Rodrigues"
        assert df.at["Thiago", "Display_Name"] == "Igor Thiago"
        assert df.at["B.Fernandes", "Display_Name"] == "Bruno Fernandes"
        assert df.at["Gabriel", "Display_Name"] == "Gabriel"

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


class TestStartPctRecovery:
    """FFP stops publishing `start_pct` for the gameweek it is *currently* on.

    Measured 2026-09-25 with the app on GW6: null on all 372 current-week rows,
    published in full for GW7-GW11. That is the one week the app scores, and
    without it every player Rotowire listed resolved to a 100% start
    probability -- 208 of 220 on the Projections Hub, against an FFP view of the
    same players with a median of 70%. FFP never rates anybody above 95%, so
    every 100% there was the app's fallback rather than anyone's opinion.
    """

    @staticmethod
    def _rows(start_pct, published=True):
        rows = [_row(6, 223340, "Saka", "Bukayo", "Saka", 3, "Arsenal", "CHE",
                     6.0, start_pct)]
        if not published:
            rows[0]["start_pct"] = None
        return rows

    def test_it_is_recovered_from_the_two_point_columns(self):
        """`Predicted == StartingPredicted x Start/100`, so the ratio is it."""
        df = ffp_feed.to_sheet_schema(self._rows(70, published=False), gw=6)
        assert df["Start"].iloc[0] == pytest.approx(70.0, abs=0.05)

    def test_a_published_value_is_left_alone(self):
        df = ffp_feed.to_sheet_schema(self._rows(70), gw=6)
        assert df["Start"].iloc[0] == pytest.approx(70.0)

    def test_the_relation_still_holds_after_recovery(self):
        df = ffp_feed.to_sheet_schema(self._rows(35, published=False), gw=6)
        row = df.iloc[0]
        assert row["Predicted"] == pytest.approx(
            row["StartingPredicted"] * row["Start"] / 100.0, abs=0.01)

    def test_a_missing_column_entirely_does_not_raise(self):
        """`DataFrame.get` returns None for an absent column, not a Series."""
        rows = self._rows(70)
        rows[0].pop("start_pct")
        df = ffp_feed.to_sheet_schema(rows, gw=6)
        assert df["Start"].iloc[0] == pytest.approx(70.0, abs=0.05)

    def test_a_zero_projection_recovers_nothing_rather_than_dividing_by_it(self):
        rows = self._rows(70, published=False)
        rows[0]["predicted_points"] = "0.000"
        rows[0]["predicted_points_start"] = "0.000"
        df = ffp_feed.to_sheet_schema(rows, gw=6)
        assert pd.isna(df["Start"].iloc[0])


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


class TestLongSchema:
    """The long schema and the sheet schema must never disagree.

    These two are the same payload read two ways, and this codebase has three
    times paid for two implementations of one piece of arithmetic drifting
    apart. The sheet's forward columns are *relative offsets* and the long
    frame's ``FFP_GW`` is absolute, so a drift here is a grid whose columns are
    labelled with the wrong gameweeks -- and every number in it stays plausible.
    """

    def test_one_row_per_player_per_gameweek(self, payload_rows):
        long = ffp_feed.to_long_schema(payload_rows)
        assert len(long) == len(payload_rows)
        assert sorted(long["FFP_GW"].unique().tolist()) == [3, 4, 5]
        assert long.groupby("FFP_GW")["Player_Code"].nunique().tolist() == [2, 2, 2]

    def test_gameweek_is_absolute_not_an_offset(self, payload_rows):
        long = ffp_feed.to_long_schema(payload_rows)
        saka = long[long["Web_Name"] == "Saka"].set_index("FFP_GW")
        # The GW4 row must carry GW4's own projection, not GW3's.
        assert saka.loc[4, "StartingPredicted"] == pytest.approx(6.089, abs=1e-3)
        assert saka.loc[3, "StartingPredicted"] == pytest.approx(5.889, abs=1e-3)

    @pytest.mark.parametrize("offset", [1, 2])
    def test_sheet_offsets_resolve_to_the_same_absolute_gameweeks(self, payload_rows, offset):
        """``GW{n}s`` at offset n-1 is the long frame's ``Predicted`` at gw+n-1.

        This is the test that would catch the drift the shared ``_gameweek_block``
        exists to prevent. The sheet says "the 2nd week of my window"; the long
        frame says "GW4". If those two ever stop meaning the same thing, the
        horizon grid silently shows one week's numbers under another's heading.
        """
        gw = 3
        sheet = ffp_feed.to_sheet_schema(payload_rows, gw)
        long = ffp_feed.to_long_schema(payload_rows)

        label = "GW%d" % (offset + 1)
        target = gw + offset
        sheet_rows = sheet.set_index(sheet["Player_Code"].astype("int64"))
        long_rows = long[long["FFP_GW"] == target]
        long_rows = long_rows.set_index(long_rows["Player_Code"].astype("int64"))

        # Unconditional on both sides...
        assert sheet_rows[label + "s"].astype(float).round(4).tolist() == \
            long_rows["Predicted"].astype(float).round(4).tolist()
        # ...and conditional on both sides.
        assert sheet_rows[label].astype(float).round(4).tolist() == \
            long_rows["StartingPredicted"].astype(float).round(4).tolist()

    def test_both_schemas_invert_the_basis_identically(self, payload_rows):
        """``Predicted`` is the discounted one on both paths, for every week."""
        sheet = ffp_feed.to_sheet_schema(payload_rows, 3)
        long = ffp_feed.to_long_schema(payload_rows)
        assert (sheet["Predicted"] <= sheet["StartingPredicted"] + 1e-9).all()
        assert (long["Predicted"] <= long["StartingPredicted"] + 1e-9).all()
        gw3 = long[long["FFP_GW"] == 3].reset_index(drop=True)
        for col in ("Predicted", "StartingPredicted", "Start"):
            assert sheet[col].astype(float).round(4).tolist() == \
                gw3[col].astype(float).round(4).tolist()

    def test_start_pct_recovery_is_shared(self):
        """A week FFP publishes no start_pct for recovers on both paths alike."""
        rows = []
        for gw in (5, 6):
            row = _row(gw, 223340, "Saka", "Bukayo", "Saka", 3, "Arsenal", "CHE", 6.0, 75)
            if gw == 5:                     # the current week, as FFP publishes it
                row["start_pct"] = None
            rows.append(row)
        sheet = ffp_feed.to_sheet_schema(rows, 5)
        long = ffp_feed.to_long_schema(rows)
        assert sheet["Start"].iloc[0] == pytest.approx(75.0, abs=0.05)
        assert long[long["FFP_GW"] == 5]["Start"].iloc[0] == pytest.approx(75.0, abs=0.05)

    def test_difficulty_rides_the_long_schema_only(self, payload_rows):
        """The sheet schema is a compatibility seam; its columns do not move."""
        long = ffp_feed.to_long_schema(payload_rows)
        assert "Difficulty" in long.columns
        assert (long["Difficulty"] == 3).all()
        assert "Difficulty" not in ffp_feed.to_sheet_schema(payload_rows, 3).columns

    def test_an_unusable_payload_returns_empty_rather_than_raising(self):
        assert ffp_feed.to_long_schema([]).empty
        assert ffp_feed.to_long_schema([{"nope": 1}]).empty


class TestFetchMemo:
    """The in-process memo that stops one payload being downloaded twice.

    Two separately-cached Streamlit wrappers need the same FFP payload --
    ``fetch_ffp_table`` for the gameweek being scored, ``fetch_ffp_horizon`` for
    the window -- and this module cannot use ``@st.cache_data`` because it is
    Streamlit-free by hard constraint.
    """

    def setup_method(self):
        ffp_feed.clear_memo()

    def teardown_method(self):
        ffp_feed.clear_memo()

    def test_a_successful_result_is_fetched_once(self):
        calls = []

        def fetch():
            calls.append(1)
            return (["row"], 7, None)

        first = ffp_feed._memoised("k", fetch)
        second = ffp_feed._memoised("k", fetch)
        assert first == second
        assert len(calls) == 1, "the second caller must not re-fetch"

    def test_a_failure_is_never_memoised(self):
        """The rule this app has already paid for breaking.

        ``get_ffp_feed`` returning a bare ``None`` from ``@st.cache_data`` pinned
        a transient timeout for five minutes, and FFP read "temporarily
        unavailable" in the app while the website plainly worked. An empty result
        must be handed back and *not* stored, so the next caller retries.
        """
        calls = []

        def failing():
            calls.append(1)
            return ([], None, None)

        assert ffp_feed._memoised("k", failing) == ([], None, None)
        assert ffp_feed._memoised("k", failing) == ([], None, None)
        assert len(calls) == 2, "a failed fetch must be retried, not pinned"

    def test_a_later_success_replaces_the_unstored_failure(self):
        results = [([], None, None), (["row"], 7, None)]

        def flaky():
            return results.pop(0)

        assert ffp_feed._memoised("k", flaky)[0] == []
        assert ffp_feed._memoised("k", flaky)[0] == ["row"]
        # And that success is now held.
        assert ffp_feed._memoised("k", lambda: (["other"], 9, None))[0] == ["row"]

    def test_none_and_empty_dicts_count_as_failures(self):
        """`bootstrap_code_to_id` returns {} when the bootstrap is unreachable."""
        assert not ffp_feed._memo_worth_keeping(None)
        assert not ffp_feed._memo_worth_keeping({})
        assert not ffp_feed._memo_worth_keeping(([], None, None))
        assert ffp_feed._memo_worth_keeping({1: 2})
        assert ffp_feed._memo_worth_keeping((["row"], 7, None))

    def test_the_entry_expires(self):
        calls = []

        def fetch():
            calls.append(1)
            return (["row"], 7, None)

        ffp_feed._memoised("k", fetch, ttl=0)
        ffp_feed._memoised("k", fetch, ttl=0)
        assert len(calls) == 2

    def test_different_keys_do_not_share(self):
        assert ffp_feed._memoised("a", lambda: (["a"], 1, None))[0] == ["a"]
        assert ffp_feed._memoised("b", lambda: (["b"], 2, None))[0] == ["b"]

    def test_the_memo_ttl_sits_under_the_streamlit_cache_above_it(self):
        """What makes the memo invisible: it is strictly shorter-lived than the
        300s cache on get_ffp_feed, so it can never serve a staler payload than
        that cache would have served anyway."""
        assert ffp_feed.MEMO_TTL_SECONDS < 300
