"""Tests for scripts/fpl/player_projections.py's pure helper functions."""

from scripts.fpl.player_projections import is_rotowire_url_stale


class TestIsRotowireUrlStale:
    def test_preview_range_covers_current_gw(self):
        url = "https://www.rotowire.com/soccer/article/best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238"
        assert is_rotowire_url_stale(url, 1) is False
        assert is_rotowire_url_stale(url, 5) is False

    def test_preview_range_does_not_cover_current_gw(self):
        url = "https://www.rotowire.com/soccer/article/best-fpl-picks-for-gameweeks-1-5-fantasy-premier-league-2026-27-126238"
        assert is_rotowire_url_stale(url, 6) is True

    def test_single_gw_article_matches(self):
        url = "https://www.rotowire.com/soccer/article/fantasy-premier-league-player-rankings-gameweek-26-fpl-gw26-arsenal-104779"
        assert is_rotowire_url_stale(url, 26) is False

    def test_single_gw_article_mismatch(self):
        url = "https://www.rotowire.com/soccer/article/fantasy-premier-league-player-rankings-gameweek-26-fpl-gw26-arsenal-104779"
        assert is_rotowire_url_stale(url, 27) is True

    def test_season_rankings_url_has_no_gw_marker(self):
        """A season-long rankings article (no gameweek in the URL) can't be
        judged stale by this check — that's handled elsewhere."""
        url = "https://www.rotowire.com/soccer/article/fantasy-premier-league-fpl-rankings-top-400-for-2026-27-season-124261"
        assert is_rotowire_url_stale(url, 1) is False


class TestGwList:
    """The Accuracy tab tells the reader which gameweeks it holds, so the empty
    state is informative rather than just blank."""

    def test_reads_as_a_sentence(self):
        from scripts.fpl.player_projections import _gw_list
        assert _gw_list([]) == "none"
        assert _gw_list([3]) == "GW3"
        assert _gw_list([1, 2]) == "GW1 and GW2"
        assert _gw_list([1, 2, 4]) == "GW1, GW2 and GW4"


class TestRenderAccuracyEmptyState:
    """With no scoreable gameweek the tab must explain *why*, not render blank.

    This is the state every user sees until a gameweek completes after snapshots
    began, so it is the most-seen state the page has.
    """

    def test_explains_which_halves_are_missing(self, monkeypatch):
        import pandas as pd
        from unittest.mock import MagicMock, patch
        import scripts.fpl.player_projections as pp

        monkeypatch.setattr(pp.projection_accuracy, "score_archive",
                            lambda **k: pd.DataFrame())
        monkeypatch.setattr(pp.projection_archive, "list_pre", lambda: [4])
        monkeypatch.setattr(pp.projection_archive, "list_actuals", lambda: [1, 2])

        said = []
        with patch.object(pp.st, "info", lambda t, *a, **k: said.append(t)), \
             patch.object(pp.st, "caption", lambda t, *a, **k: said.append(t)):
            pp.render_accuracy()

        blob = " ".join(said)
        assert "No gameweek yet" in blob
        assert "GW4" in blob and "GW1 and GW2" in blob


class TestPositionalRank:
    """The Pos RK column on the Hub's projection tables.

    A projection only means something against the players who could fill the
    same slot: 4.2 is a fine goalkeeper and a poor forward, and the overall
    ``#`` column cannot say which.
    """

    def _frame(self):
        import pandas as pd
        return pd.DataFrame({
            "Pos": ["M", "M", "M", "M", "F", "F", "G"],
            "Proj": [6.0, 4.0, 4.0, 2.0, 9.0, 1.0, 3.0],
        })

    def test_ranks_within_position_not_overall(self):
        from scripts.fpl.player_projections import _positional_rank
        rank = _positional_rank(self._frame(), "Proj")
        # The best forward is 1 despite out-scoring every midfielder, and the
        # lone goalkeeper is 1 despite the 3.0 being near the bottom overall.
        assert rank.tolist() == [1.0, 2.0, 2.0, 4.0, 1.0, 2.0, 1.0]

    def test_ties_share_a_rank_and_the_next_player_skips_one(self):
        """Matching analytics.positional_rank()'s "strictly better, plus one"."""
        from scripts.fpl.player_projections import _positional_rank
        rank = _positional_rank(self._frame(), "Proj")
        assert rank.iloc[1] == rank.iloc[2] == 2.0
        assert rank.iloc[3] == 4.0

    def test_no_value_ranks_nowhere(self):
        """NaN renders as an empty cell, never as a confident last place."""
        import pandas as pd
        from scripts.fpl.player_projections import _positional_rank
        df = pd.DataFrame({"Pos": ["M", "M"], "Proj": [5.0, float("nan")]})
        rank = _positional_rank(df, "Proj")
        assert rank.iloc[0] == 1.0
        assert pd.isna(rank.iloc[1])

    def test_a_missing_column_is_not_a_crash(self):
        import pandas as pd
        from scripts.fpl.player_projections import _positional_rank
        df = pd.DataFrame({"Pos": ["M"]})
        assert _positional_rank(df, "Proj").isna().all()

    def test_keeps_the_frames_index(self):
        """Callers insert this straight back into the frame it came from."""
        import pandas as pd
        from scripts.fpl.player_projections import _positional_rank
        df = self._frame().set_index(pd.Index(list("abcdefg")))
        assert list(_positional_rank(df, "Proj").index) == list("abcdefg")


class TestBlendedPosRankIsLeagueWide:
    """The rank must be taken over the whole board, before any filtering.

    A rank re-derived from the filtered rows ranks a selection against itself --
    search for one player and he reads "1" whatever he is worth -- which is the
    same trap render_styled_table's colour ranges are documented for. The only
    way to see it is to render the tab twice and compare.
    """

    @staticmethod
    def _pool():
        import pandas as pd
        return pd.DataFrame({
            "Player_ID": [1, 2, 3, 4],
            "Player": ["Big Mid", "Mid Mid", "Small Mid", "A Keeper"],
            "Display_Name": ["Big Mid", "Mid Mid", "Small Mid", "A Keeper"],
            "Team": ["ARS", "MCI", "LIV", "ARS"],
            "Position": ["M", "M", "M", "G"],
            "Proj": [7.0, 5.0, 3.0, 4.0],
            "Proj_Start": [8.0, 6.0, 4.0, 4.2],
            "Start_Pct": [0.9, 0.85, 0.75, 0.95],
            "Proj_Start__rotowire": [8.0, 6.0, 4.0, 4.2],
            "Proj_Start__ffp": [8.1, 5.9, 4.1, 4.3],
            "Proj_Spread": [0.1, 0.1, 0.1, 0.1],
            "Proj_Src": ["RW+FFP"] * 4,
        })

    def _render(self, search):
        import contextlib
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp

        captured = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(
                pp, "get_rotowire_player_projections",
                return_value=pd.DataFrame({"Player": ["Big Mid"], "Points": [8.0]})))
            stack.enter_context(patch.object(pp, "get_ffp_feed",
                                             return_value=_FeedStub()))
            stack.enter_context(patch.object(pp, "_blended_pool",
                                             return_value=self._pool()))
            stack.enter_context(patch.object(pp, "_blended_source_status"))
            stack.enter_context(patch.object(
                pp, "render_styled_table",
                side_effect=lambda df, **k: captured.append((df, k))))
            stack.enter_context(patch("streamlit.text_input", return_value=search))
            pp.render_blended_projections()
        assert captured, "the tab rendered no table"
        return captured[-1]

    def test_the_column_is_there(self, mock_streamlit):
        df, kwargs = self._render("")
        assert "Pos RK" in df.columns
        assert kwargs["col_formats"].get("Pos RK") == "{:.0f}"
        mids = df[df["Pos"] == "M"].sort_values("Proj", ascending=False)
        assert mids["Pos RK"].tolist() == [1.0, 2.0, 3.0]
        # Ranked within position, so the keeper leads his own position while
        # sitting third overall.
        assert df.loc[df["Player"] == "A Keeper", "Pos RK"].iloc[0] == 1.0

    def test_filtering_does_not_re_rank(self, mock_streamlit):
        df, _ = self._render("Small Mid")
        assert df["Player"].tolist() == ["Small Mid"]
        assert df["Pos RK"].iloc[0] == 3.0, (
            "Pos RK was re-derived from the filtered rows -- the third-best "
            "midfielder in the league must not read as #1 because he is the "
            "only row left"
        )


class _FeedStub:
    """The minimum of scraping.FFPFeed that render_blended_projections reads."""
    ok = True
    gameweek = 25
    provenance = "site"
    updated = None
    note = ""

    def __init__(self):
        import pandas as pd
        self.df = pd.DataFrame()

    def is_stale(self, gw):
        return False


class TestFFPDataPosRank:
    """Pos RK on the FFP Data table, ranked on whatever that table is about.

    FFP normally publishes point predictions and sometimes only betting-derived
    probabilities, and the table's primary column changes with it. The rank has
    to follow, and say which column it followed.
    """

    @staticmethod
    def _feed(with_predictions=True):
        import pandas as pd
        feed = _FeedStub()
        rows = {
            "Name": ["Big Mid", "Mid Mid", "Small Mid", "A Keeper"],
            "Display_Name": ["Big Mid", "Mid Mid", "Small Mid", "A Keeper"],
            "Team": ["Arsenal", "Man City", "Liverpool", "Arsenal"],
            "Position": ["MID", "MID", "MID", "GK"],
            "Fixture": ["Chelsea (h)"] * 4,
            "Price": [9.0, 7.0, 5.0, 5.5],
            "Ownership": [30.0, 10.0, 2.0, 8.0],
            "Start": [90.0, 80.0, 60.0, 95.0],
        }
        if with_predictions:
            rows.update({
                "Predicted": [7.0, 5.0, 3.0, 4.0],
                "StartingPredicted": [7.8, 6.3, 5.0, 4.2],
                "Next2GWs": [14.0, 10.0, 6.0, 8.0],
                "Next3GWs": [21.0, 15.0, 9.0, 12.0],
                "Next6GWs": [42.0, 30.0, 18.0, 24.0],
            })
        else:
            rows.update({
                "CS": [20.0, 25.0, 30.0, 30.0],
                "AnytimeGoal": [35.0, 20.0, 5.0, 0.0],
                "AnytimeAssist": [20.0, 15.0, 5.0, 0.0],
                "AnytimeReturn": [50.0, 30.0, 10.0, 0.0],
            })
        feed.df = pd.DataFrame(rows)
        return feed

    def _render(self, feed, search=""):
        import contextlib
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp

        captured, captions = [], []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(pp, "get_ffp_feed", return_value=feed))
            stack.enter_context(patch.object(
                pp, "render_styled_table",
                side_effect=lambda df, **k: captured.append((df, k))))
            stack.enter_context(patch("streamlit.text_input", return_value=search))
            stack.enter_context(patch("streamlit.caption",
                                      side_effect=lambda t, *a, **k: captions.append(t)))
            pp.render_ffp_data()
        assert captured, "the FFP tab rendered no table"
        return captured[-1][0], captured[-1][1], " ".join(captions)

    def test_ranks_on_the_prediction_when_there_is_one(self, mock_streamlit):
        df, kwargs, caption = self._render(self._feed())
        assert "Pos RK" in df.columns
        assert kwargs["col_formats"].get("Pos RK") == "{:.0f}"
        assert df.loc[df["Player"] == "Big Mid", "Pos RK"].iloc[0] == 1.0
        assert df.loc[df["Player"] == "Small Mid", "Pos RK"].iloc[0] == 3.0
        assert df.loc[df["Player"] == "A Keeper", "Pos RK"].iloc[0] == 1.0
        assert "Pred Pts" in caption

    def test_ranks_on_the_goal_odds_when_ffp_has_published_no_predictions(self, mock_streamlit):
        """The fallback table is about probabilities, so the rank must be too --
        and the caption has to name the column it used, or the number is a
        rank of nothing in particular."""
        df, _kwargs, caption = self._render(self._feed(with_predictions=False))
        assert "Pred Pts" not in df.columns
        assert df.loc[df["Player"] == "Big Mid", "Pos RK"].iloc[0] == 1.0
        assert "Goal %" in caption

    def test_filtering_does_not_re_rank(self, mock_streamlit):
        df, _kwargs, _caption = self._render(self._feed(), search="Small Mid")
        assert df["Player"].tolist() == ["Small Mid"]
        assert df["Pos RK"].iloc[0] == 3.0
