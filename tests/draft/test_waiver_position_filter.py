"""The position filter on the Waiver Wire's Available Players board."""

import pandas as pd
import pytest

from scripts.draft.waiver_wire import POSITION_LABEL_TO_CODE, _filter_by_position


def _board(positions):
    return pd.DataFrame({
        "Player": [f"P{i}" for i in range(len(positions))],
        "Position": positions,
        "Transfer Score": [0.5] * len(positions),
    })


ALL = list(POSITION_LABEL_TO_CODE.keys())


class TestFiltering:
    def test_one_position(self):
        out = _filter_by_position(_board(["G", "D", "M", "F"]), ["MID"])
        assert list(out["Position"]) == ["M"]

    def test_several_positions(self):
        out = _filter_by_position(_board(["G", "D", "M", "F"]), ["DEF", "FWD"])
        assert list(out["Position"]) == ["D", "F"]

    def test_all_positions_is_the_whole_board(self):
        board = _board(["G", "D", "M", "F"])
        assert len(_filter_by_position(board, ALL)) == len(board)

    def test_no_positions_selected_is_an_empty_board(self):
        """Not the whole board: an empty multiselect means "show me nothing"."""
        assert _filter_by_position(_board(["G", "D", "M", "F"]), []).empty

    def test_row_order_is_preserved(self):
        """The board is pre-sorted by Transfer Score; filtering must not reorder."""
        board = _board(["M", "M", "M"])
        board["Transfer Score"] = [0.9, 0.6, 0.3]
        out = _filter_by_position(board, ["MID"])
        assert list(out["Transfer Score"]) == [0.9, 0.6, 0.3]


class TestPositionEncodings:
    """Frames here carry G/D/M/F or GK/DEF/MID/FWD depending on the merge."""

    def test_long_codes_match(self):
        out = _filter_by_position(_board(["GK", "DEF", "MID", "FWD"]), ["MID"])
        assert list(out["Position"]) == ["MID"]

    def test_mixed_encodings_in_one_frame(self):
        out = _filter_by_position(_board(["M", "MID", "D"]), ["MID"])
        assert list(out["Position"]) == ["M", "MID"]

    def test_case_and_whitespace(self):
        out = _filter_by_position(_board([" mid ", "d"]), ["MID"])
        assert len(out) == 1

    def test_an_unknown_code_is_dropped_not_kept(self):
        out = _filter_by_position(_board(["M", "???"]), ["MID"])
        assert list(out["Position"]) == ["M"]


class TestDegradesQuietly:
    @pytest.mark.parametrize("df", [None, pd.DataFrame()])
    def test_empty_input(self, df):
        assert _filter_by_position(df, ["MID"]) is df or _filter_by_position(df, ["MID"]).empty

    def test_frame_without_a_position_column(self):
        board = pd.DataFrame({"Player": ["a", "b"]})
        assert len(_filter_by_position(board, ["MID"])) == 2
