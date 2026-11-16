"""Unit tests for scripts/common/player_history.py.

This is the app's first previous-season data. Two properties matter more than
the arithmetic: the frame is keyed on the **permanent player code** (element ids
are reassigned at season rollover, so a frame keyed on one cannot be joined to
anything in a later season), and a failed fetch returns an empty frame rather
than raising -- it feeds one rung of a three-rung ladder, and the other two must
keep working.
"""

from unittest import mock

import pandas as pd
import pytest

from scripts.common.player_history import (
    CURRENT_COLUMNS,
    HISTORY_COLUMNS,
    RATE_COLUMNS,
    SEASON_GWS,
    fetch_history_past,
    fetch_player_history,
    prior_season_rate,
)


def _history(rows):
    return pd.DataFrame(rows, columns=["Player_Code", "Player_ID", "Season_Name",
                                       "total_points", "starts", "minutes"])


# Saliba's real payload, read live on 2026-11-15. The 2020/21 and 2021/22 rows
# carry 0 starts because FPL did not track `starts` before 2022/23 -- which is
# why `min_starts` is the right filter and a `minutes` fallback would be wrong.
SALIBA = _history([
    (462424, 6, "2020/21", 0, 0, 0),
    (462424, 6, "2021/22", 0, 0, 0),
    (462424, 6, "2022/23", 117, 27, 2415),
    (462424, 6, "2023/24", 164, 38, 3420),
    (462424, 6, "2024/25", 130, 35, 3039),
    (462424, 6, "2025/26", 137, 30, 2614),
])


class TestPriorSeasonRate:

    def test_it_reads_the_reported_case(self):
        out = prior_season_rate(SALIBA, min_starts=20)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["Player_Code"] == 462424
        assert row["Prior_Season"] == "2025/26"
        assert row["Prior_PPS"] == pytest.approx(137 / 30, abs=1e-6)
        assert row["Prior_Role"] == pytest.approx(30 / SEASON_GWS, abs=1e-6)

    def test_it_takes_the_most_recent_season_not_the_best(self):
        """A four-year-old peak is not pedigree.

        The discriminating case: an elite season followed by a mediocre but
        still-regular one. "Best" would report 6.00; the answer is 3.00.
        """
        hist = _history([
            (99, 1, "2022/23", 228, 38, 3420),   # 6.00 a start -- the peak
            (99, 1, "2025/26", 114, 38, 3420),   # 3.00 a start -- who he is now
        ])
        out = prior_season_rate(hist, min_starts=20)
        assert out.iloc[0]["Prior_Season"] == "2025/26"
        assert out.iloc[0]["Prior_PPS"] == pytest.approx(3.0)

    def test_order_in_the_input_frame_does_not_matter(self):
        shuffled = SALIBA.iloc[::-1].reset_index(drop=True)
        assert prior_season_rate(shuffled, min_starts=20).iloc[0]["Prior_Season"] \
            == prior_season_rate(SALIBA, min_starts=20).iloc[0]["Prior_Season"]

    def test_min_starts_excludes_a_part_season(self):
        hist = _history([(5, 2, "2025/26", 70, 10, 900)])
        assert prior_season_rate(hist, min_starts=20).empty
        # ...but a low bar keeps him, because the *share* is what the caller
        # gates on and a part-season shows up there rather than inflating a rate.
        kept = prior_season_rate(hist, min_starts=10)
        assert len(kept) == 1
        assert kept.iloc[0]["Prior_PPS"] == pytest.approx(7.0)
        assert kept.iloc[0]["Prior_Role"] == pytest.approx(10 / SEASON_GWS)

    def test_role_never_exceeds_one(self):
        hist = _history([(5, 2, "2025/26", 200, 44, 3960)])
        assert prior_season_rate(hist, min_starts=10).iloc[0]["Prior_Role"] == 1.0

    def test_one_row_per_player(self):
        hist = pd.concat([SALIBA, _history([
            (118748, 1, "2025/26", 162, 37, 3330),
            (118748, 1, "2024/25", 142, 38, 3420),
        ])])
        out = prior_season_rate(hist, min_starts=20)
        assert len(out) == 2
        assert out["Player_Code"].is_unique

    @pytest.mark.parametrize("bad", [None, pd.DataFrame()])
    def test_nothing_in_nothing_out_with_the_columns_declared(self, bad):
        out = prior_season_rate(bad)
        assert out.empty
        assert list(out.columns) == RATE_COLUMNS

    def test_a_frame_missing_columns_degrades_rather_than_raising(self):
        out = prior_season_rate(pd.DataFrame({"Player_Code": [1]}))
        assert out.empty
        assert list(out.columns) == RATE_COLUMNS


class TestFetchHistoryPast:

    def test_both_halves_come_from_one_request(self):
        """The current season's usage costs no extra call.

        They answer different questions -- what he is worth, and whether he is
        being picked -- but they arrive in the same payload, and two fetchers
        hitting one endpoint is the twin-drift failure this codebase keeps
        paying for.
        """
        payload = {
            "history_past": [{"season_name": "2025/26", "total_points": 137,
                              "starts": 30, "minutes": 2614}],
            "history": [{"minutes": 90, "starts": 1}, {"minutes": 0, "starts": 0},
                        {"minutes": 88, "starts": 1}, {"minutes": 12, "starts": 0}],
        }
        with mock.patch("scripts.common.player_history.requests.Session") as sess:
            sess.return_value.get.return_value.json.return_value = payload
            past, current = fetch_player_history({6: 462424})
        assert sess.return_value.get.call_count == 1
        assert list(current.columns) == CURRENT_COLUMNS
        row = current.iloc[0]
        assert row["Player_Code"] == 462424
        # Three gameweeks with minutes on the pitch, two of them starts.
        assert row["Appearances"] == 3
        assert row["Current_Starts"] == 2
        assert row["Current_Minutes"] == 190
        assert len(past) == 1

    def test_an_appearance_is_a_gameweek_he_played_not_one_he_started(self):
        """The distinction the bootstrap cannot make.

        A fit substitute and a man in the treatment room both record 0 minutes
        in a given week; only the appearance count separates them, and that is
        the whole basis for deciding whether this season has answered anything.
        """
        payload = {"history_past": [],
                   "history": [{"minutes": 12, "starts": 0},
                               {"minutes": 3, "starts": 0},
                               {"minutes": 0, "starts": 0}]}
        with mock.patch("scripts.common.player_history.requests.Session") as sess:
            sess.return_value.get.return_value.json.return_value = payload
            _past, current = fetch_player_history({6: 462424})
        assert current.iloc[0]["Appearances"] == 2
        assert current.iloc[0]["Current_Starts"] == 0

    def test_it_keys_on_the_code_the_caller_supplies(self):
        """Not on the element id the endpoint was addressed by.

        Element ids are reassigned at season rollover, so the code is the only
        key a prior-season frame can carry. The mapping is an argument rather
        than something this function derives, so the id space stays the
        caller's decision -- `element-summary` is the *Classic* API, and a Draft
        id resolves to a real but wrong player for 64 of 669.
        """
        payload = {"history_past": [
            {"season_name": "2025/26", "total_points": 137, "starts": 30,
             "minutes": 2614},
        ]}
        with mock.patch("scripts.common.player_history.requests.Session") as sess:
            sess.return_value.get.return_value.json.return_value = payload
            out = fetch_history_past({6: 462424})
        assert list(out.columns) == HISTORY_COLUMNS
        assert out.iloc[0]["Player_Code"] == 462424
        assert out.iloc[0]["Player_ID"] == 6
        assert sess.return_value.get.call_args[0][0].endswith("/element-summary/6/")

    def test_a_failed_request_is_an_empty_frame_not_an_exception(self):
        with mock.patch("scripts.common.player_history.requests.Session") as sess:
            sess.return_value.get.side_effect = RuntimeError("down")
            out = fetch_history_past({6: 462424})
        assert out.empty
        assert list(out.columns) == HISTORY_COLUMNS

    def test_a_player_with_no_history_is_simply_absent(self):
        with mock.patch("scripts.common.player_history.requests.Session") as sess:
            sess.return_value.get.return_value.json.return_value = {"history_past": []}
            assert fetch_history_past({6: 462424}).empty

    @pytest.mark.parametrize("mapping", [None, {}, {None: 1}, {1: None}])
    def test_an_unusable_mapping_fetches_nothing(self, mapping):
        with mock.patch("scripts.common.player_history.requests.Session") as sess:
            past, current = fetch_player_history(mapping)
            sess.return_value.get.assert_not_called()
        assert past.empty and current.empty
        assert list(current.columns) == CURRENT_COLUMNS


class TestUsageIsAttachedToTheRate:

    def test_appearances_ride_along_with_the_prior_rate(self):
        current = pd.DataFrame([dict(Player_Code=462424, Appearances=2,
                                     Current_Starts=2, Current_Minutes=178)])
        out = prior_season_rate(SALIBA, current, min_starts=20)
        assert out.iloc[0]["Appearances"] == 2
        assert out.iloc[0]["Prior_PPS"] == pytest.approx(137 / 30)

    def test_a_player_with_no_prior_season_still_gets_his_appearances(self):
        """An outer join, because the two halves cover different players.

        The appearance count is what says whether this season has answered the
        question, and that matters just as much for a player with no prior
        season at all.
        """
        current = pd.DataFrame([dict(Player_Code=999, Appearances=5,
                                     Current_Starts=0, Current_Minutes=60)])
        out = prior_season_rate(SALIBA, current, min_starts=20)
        assert set(out["Player_Code"]) == {462424, 999}
        assert out[out["Player_Code"] == 999].iloc[0]["Appearances"] == 5
        assert pd.isna(out[out["Player_Code"] == 999].iloc[0]["Prior_PPS"])

    def test_no_usage_still_declares_the_column(self):
        out = prior_season_rate(SALIBA, None, min_starts=20)
        assert list(out.columns) == RATE_COLUMNS
        assert out["Appearances"].isna().all()
