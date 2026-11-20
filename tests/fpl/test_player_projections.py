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
            # The source audit is now data (`_source_rows`) plus two
            # presentations. Stubbing the rows keeps `analytics._lineup_xis`
            # and the Rotowire article stamp out of a rank test, and stubbing
            # the detail renderer keeps its table out of `captured`.
            stack.enter_context(patch.object(pp, "_source_rows", return_value=[]))
            stack.enter_context(patch.object(pp, "_render_source_detail"))
            # The trust tile reads `archive/projections/`, which measured 0.61s
            # when this page rendered it eagerly. No test should pay for it.
            stack.enter_context(patch.object(
                pp, "_trust_figures",
                return_value={"ok": False, "reason": "stubbed in tests"}))
            stack.enter_context(patch.object(pp, "_sparkline_column",
                                             return_value={}))
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

    def test_the_overall_rank_is_league_wide_too(self, mock_streamlit):
        """The reported symptom: searching one player reported him as #1."""
        df, kwargs = self._render("")
        assert df["#"].tolist() == [1.0, 2.0, 3.0, 4.0]
        assert kwargs["col_formats"].get("#") == "{:.0f}"

        df, _ = self._render("Small Mid")
        assert df["Player"].tolist() == ["Small Mid"]
        assert df["#"].iloc[0] == 4.0, (
            "# was stamped on after filtering -- the worst player on the board "
            "must not read as #1 because he is the only row left"
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


# =============================================================================
# The reordered board: source strip, insight tiles, sparklines
# =============================================================================

def _src_row(short="Rotowire", ok=True, note="", weight=0.2, rows=200,
             updated=None, name=None):
    """One record in `_source_rows`' shape."""
    return {
        "name": name or short, "short": short, "ok": ok, "rows": rows,
        "note": note, "updated": updated,
        "weight": weight,
        "weight_label": ("%.0f%%" % (weight * 100)) if weight else "fallback only",
        "status": ("\U0001F534 Unavailable" if not ok
                   else ("\U0001F7E1 %s" % note if note else "\U0001F7E2 OK")),
    }


class TestSourceHealthLine:
    """The strip above the board and the table behind `Source detail` are two
    presentations of one record set, so they cannot disagree.

    That is the reason `_source_rows` exists as its own function. A strip that
    did its own reading of the same feeds would be a second implementation able
    to contradict the table a single click below it -- the twin-drift failure
    this codebase keeps paying for, in the one place a user would compare them
    directly.
    """

    def test_a_healthy_board_says_so(self):
        from scripts.fpl import player_projections as pp
        from scripts.common import cards
        html = pp._source_health_line([_src_row(), _src_row(short="FFP", weight=0.8)])
        assert "All 2 sources live" in html
        assert cards.ACCENT in html

    def test_a_down_source_is_named_and_turns_the_verdict_red(self):
        from scripts.fpl import player_projections as pp
        from scripts.common import cards
        html = pp._source_health_line([
            _src_row(),
            _src_row(short="FFP", weight=0.8, ok=False, rows=0,
                     note="no rankings table for this gameweek"),
        ])
        assert "1 of 2 sources down" in html
        assert cards.BAD in html
        assert "FFP" in html and "unavailable" in html

    def test_a_degraded_source_reads_as_degraded_not_as_down(self):
        """An archived FFP table is the normal state for most of every week."""
        from scripts.fpl import player_projections as pp
        from scripts.common import cards
        html = pp._source_health_line([
            _src_row(),
            _src_row(short="FFP", weight=0.8,
                     note="from our archive of their last publication"),
        ])
        assert "degraded" in html
        assert cards.WARN in html
        assert "down" not in html

    def test_a_healthy_zero_weight_fallback_is_left_off_the_strip(self):
        """Deliberate: it has nothing to say at a glance, and listing it pushes
        the two numbers that matter off the end of the line. It is still in the
        detail table, and it still appears here the moment it is unhealthy."""
        from scripts.fpl import player_projections as pp
        rows = [_src_row(), _src_row(short="FPL xP", weight=0.0, rows=29)]
        assert "FPL xP" not in pp._source_health_line(rows)

        rows[1] = _src_row(short="FPL xP", weight=0.0, ok=False, rows=0,
                           note="unavailable")
        assert "FPL xP" in pp._source_health_line(rows)

    def test_a_start_only_source_appears_when_its_coverage_is_short(self):
        from scripts.fpl import player_projections as pp
        rows = [_src_row(), _src_row(short="RW XIs", weight=None, rows=209,
                                     note="19/20 clubs complete")]
        html = pp._source_health_line(rows)
        assert "RW XIs" in html and "19/20" in html

    def test_no_rows_renders_nothing_rather_than_an_empty_card(self):
        from scripts.fpl import player_projections as pp
        assert pp._source_health_line([]) == ""

    def test_the_strip_and_the_table_agree_about_what_is_down(self, mock_streamlit):
        """The structural claim: both read the same records.

        Asserted by rendering both from one row set and checking that every
        source the table marks unavailable is also named as unavailable on the
        strip.
        """
        import contextlib
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp

        rows = [
            _src_row(short="Rotowire", weight=0.2),
            _src_row(short="FFP", weight=0.8, ok=False, rows=0, note="gone"),
            _src_row(short="RW XIs", weight=None, rows=0, ok=False,
                     note="unavailable - the other XI source stands alone"),
        ]
        captured = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(
                pp, "render_styled_table",
                side_effect=lambda df, **k: captured.append(df)))
            pp._render_source_detail(rows)
        table = captured[-1]
        down_in_table = {
            r["Source"] for _, r in table.iterrows()
            if "Unavailable" in str(r["Status"])
        }
        strip = pp._source_health_line(rows)
        assert down_in_table == {"FFP", "RW XIs"}
        for name in down_in_table:
            assert name in strip, (
                "%s is Unavailable in the detail table but absent from the "
                "status line -- the two have stopped sharing a derivation" % name
            )


class TestInsightTiles:
    """Three tiles above the board, each of which must be able to say it does
    not know.

    This is the `week_check.STATUS_UNKNOWN` discipline one surface over: a tile
    makes a claim at a glance, and "nothing to flag" and "we could not check"
    render identically unless one of them says so. Every test below has a
    partner asserting the unknown state, because the defaulted version of each
    of these would look entirely calm and correct.
    """

    @staticmethod
    def _board():
        import pandas as pd
        return pd.DataFrame({
            "#": [1.0, 2.0, 3.0],
            "Player": ["Erling Haaland", "Morgan Rogers", "Phil Foden"],
            "Team": ["MCI", "AVL", "MCI"],
            "Pos": ["F", "M", "M"],
            "Proj": [7.02, 4.41, 0.45],
            "Start %": [95.0, 90.0, 9.0],
            "Rotowire": [7.41, 3.88, 4.93],
            "FFP": [7.12, 5.41, 4.21],
            "Spread": [0.29, 1.53, 0.72],
        })

    # ---- top of the board ------------------------------------------------
    def test_the_top_tile_names_the_leader_and_his_projection(self):
        from scripts.fpl import player_projections as pp
        html = pp._tile_top(self._board())
        assert "Erling Haaland" in html
        assert "7.02" in html
        assert "95% to start" in html

    def test_the_top_tile_is_unknown_on_an_empty_board(self):
        import pandas as pd
        from scripts.fpl import player_projections as pp
        html = pp._tile_top(pd.DataFrame())
        assert "Not available" in html
        assert "no player is projected" in html

    def test_the_top_tile_is_unknown_when_the_leading_row_has_no_number(self):
        from scripts.fpl import player_projections as pp
        board = self._board()
        board.loc[0, "Proj"] = float("nan")
        html = pp._tile_top(board)
        assert "Not available" in html

    # ---- disagreement ----------------------------------------------------
    def test_the_disagreement_tile_finds_the_widest_spread(self):
        from scripts.fpl import player_projections as pp
        html = pp._tile_disagreement(self._board())
        assert "Morgan Rogers" in html, (
            "the tile must pick the largest Spread, not the top of the board"
        )
        assert "1.53" in html

    def test_the_disagreement_tile_says_which_source_is_higher(self):
        """The whole content of the tile: `Spread` is a magnitude, and a
        manager needs to know whose number they are taking."""
        from scripts.fpl import player_projections as pp
        html = pp._tile_disagreement(self._board())
        assert "FFP 5.41 against Rotowire 3.88" in html

    def test_the_disagreement_tile_counts_the_wide_ones(self):
        from scripts.fpl import player_projections as pp
        assert "1 player over 1.5" in pp._tile_disagreement(self._board())

    def test_the_disagreement_tile_is_unknown_with_only_one_source(self):
        from scripts.fpl import player_projections as pp
        board = self._board()
        board["Spread"] = float("nan")
        html = pp._tile_disagreement(board)
        assert "Not available" in html
        assert "only one source" in html

    def test_the_disagreement_tile_is_unknown_with_no_spread_column(self):
        from scripts.fpl import player_projections as pp
        board = self._board().drop(columns=["Spread"])
        assert "Not available" in pp._tile_disagreement(board)

    # ---- trust -----------------------------------------------------------
    def test_the_trust_tile_reports_the_skill_figure(self):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        with patch.object(pp, "_trust_figures", return_value={
                "ok": True, "skill": 0.618, "brier": 0.0845,
                "baseline": 0.2213, "n": 5308, "gameweeks": 8}):
            html = pp._tile_trust()
        assert "62%" in html
        assert "0.085" in html and "0.221" in html
        assert "5,308 player-gameweeks" in html, (
            "the sample size is what makes an error rate readable"
        )
        assert "8 gameweeks" in html

    def test_the_trust_tile_states_the_reason_when_it_cannot_answer(self):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        with patch.object(pp, "_trust_figures", return_value={
                "ok": False, "reason": "only 2 of 5 pre-deadline gameweeks "
                                       "scored so far"}):
            html = pp._tile_trust()
        assert "Not available" in html
        assert "2 of 5 pre-deadline gameweeks" in html, (
            "the tile must say why it is blank -- a silent tile reads as "
            "'nothing to report', which is the opposite claim"
        )

    def test_negative_skill_is_reported_rather_than_hidden(self):
        """A source worse than one constant for everybody is a real answer, and
        the Accuracy section already says so about MAE."""
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        with patch.object(pp, "_trust_figures", return_value={
                "ok": True, "skill": -0.08, "brier": 0.24, "baseline": 0.2215,
                "n": None, "gameweeks": 6}):
            html = pp._tile_trust()
        assert "-8%" in html
        from scripts.common import cards
        assert cards.BAD in html

    # ---- the gate --------------------------------------------------------
    def test_the_confidence_gate_is_the_accuracy_sections_own(self):
        """Not a second threshold.

        The Accuracy section refuses to fit on fewer than
        MIN_GAMEWEEKS_FOR_CONFIDENCE pre-deadline gameweeks. A tile printing a
        confident percentage off a history that section disclaims would have
        the page contradict itself two clicks apart.
        """
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        from scripts.common import projection_accuracy

        scored = pd.DataFrame({"source": ["blend"], "scope": ["start"],
                               "gameweek": [4], "n": [300], "brier": [0.07]})
        under = projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE - 1
        with patch.object(projection_accuracy, "score_archive", return_value=scored), \
             patch.object(projection_accuracy, "confident_gameweeks", return_value=under):
            figs = pp._trust_figures.__wrapped__() if hasattr(
                pp._trust_figures, "__wrapped__") else pp._trust_figures()
        assert figs["ok"] is False
        assert str(under) in figs["reason"]
        assert str(projection_accuracy.MIN_GAMEWEEKS_FOR_CONFIDENCE) in figs["reason"]

    def test_the_model_is_scored_on_the_same_population_as_its_baseline(self):
        """The defect this tile shipped with for one headless render.

        `no_skill_reference` is computed over every scored row. Scoring the
        blend on `common_subset=True` instead -- the players *all four* sources
        priced, which are mostly the doubtful ones FPL bothered to rate --
        compares a model on the hard subset against a baseline on everybody.
        Measured live that reported 39% skill where the like-for-like figure is
        62%, and both numbers look entirely plausible.
        """
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        from scripts.common import projection_accuracy
        seen = {}

        def _spy(**kwargs):
            seen.update(kwargs)
            import pandas as pd
            return pd.DataFrame()

        with patch.object(projection_accuracy, "score_archive", side_effect=_spy):
            pp._trust_figures()
        assert seen.get("common_subset") is False, (
            "the trust tile must score the blend over the full population, "
            "because that is the population its baseline is drawn from"
        )

    def test_an_unreadable_archive_is_unknown_rather_than_a_crash(self):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        from scripts.common import projection_accuracy
        with patch.object(projection_accuracy, "score_archive",
                          side_effect=OSError("no archive")):
            figs = pp._trust_figures()
        assert figs["ok"] is False
        assert "could not be read" in figs["reason"]


class TestSparklineColumn:
    """The horizon as a shape, on one scale for the whole board."""

    def test_it_draws_a_line_through_the_points(self):
        from scripts.fpl import player_projections as pp
        svg = pp._sparkline_svg([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 0.0, 7.0)
        assert svg.startswith("<svg")
        assert "polyline" in svg and svg.count(",") >= 6
        assert "circle" in svg, "the last gameweek should be marked"

    def test_one_point_renders_nothing(self):
        """An empty cell, not a flat line at zero -- a single forecast is not a
        trajectory, and a drawn line would assert a trend nobody published."""
        from scripts.fpl import player_projections as pp
        assert pp._sparkline_svg([3.0], 0.0, 7.0) == ""
        assert pp._sparkline_svg([float("nan")] * 6, 0.0, 7.0) == ""

    def test_the_scale_is_shared_so_two_flat_players_differ(self):
        """The trap this exists to avoid: a sparkline normalised to its own row
        makes every trajectory fill its cell, so a flat 0.3 and a flat 6.0
        would draw identically."""
        from scripts.fpl import player_projections as pp
        low = pp._sparkline_svg([0.3] * 6, 0.0, 7.0)
        high = pp._sparkline_svg([6.0] * 6, 0.0, 7.0)
        assert low != high
        assert 'points="' in low and 'points="' in high

    def test_the_colour_comes_from_the_shared_table_ramp(self):
        from scripts.fpl import player_projections as pp
        from scripts.common.styled_tables import value_rgb
        svg = pp._sparkline_svg([6.0] * 6, 0.0, 7.0)
        assert value_rgb(6.0, 0.0, 7.0) in svg, (
            "the line must use the same red-to-green ramp as the numbers "
            "beside it, not a second implementation of it"
        )

    def test_the_column_is_really_populated(self):
        """Populated, not merely present.

        The `test_build_pool_really_stamps_it` lesson: a column of empty
        strings satisfies "is the column there" while having exercised none of
        the code path.
        """
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp

        points = pd.DataFrame(
            {12: [7.0, 1.0], 13: [6.0, 1.2], 14: [5.0, 0.9]},
            index=pd.Index([1, 2], name="Player_ID"))

        class _Horizon:
            ok = True
            def gameweeks(self):
                return [12, 13, 14]

        with patch.object(pp, "get_ffp_horizon", return_value=_Horizon()), \
             patch.object(pp, "_horizon_frames",
                          return_value=(points, None, None, None)):
            got = pp._sparkline_column(pd.DataFrame(), 12)
        assert set(got) == {1, 2}
        assert all(v.startswith("<svg") for v in got.values())
        assert got[1] != got[2], "both rows rendered the same line"

    def test_an_unreadable_horizon_costs_the_column_and_nothing_else(self):
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        with patch.object(pp, "get_ffp_horizon", side_effect=OSError("down")):
            assert pp._sparkline_column(pd.DataFrame(), 12) == {}

    def test_a_single_gameweek_window_is_not_a_sparkline(self):
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp

        class _Horizon:
            ok = True
            def gameweeks(self):
                return [12]

        points = pd.DataFrame({12: [7.0]}, index=pd.Index([1], name="Player_ID"))
        with patch.object(pp, "get_ffp_horizon", return_value=_Horizon()), \
             patch.object(pp, "_horizon_frames",
                          return_value=(points, None, None, None)):
            assert pp._sparkline_column(pd.DataFrame(), 12) == {}


class TestBoardSparklineWiring:
    """The sparkline has to reach the rendered table, not just exist."""

    @staticmethod
    def _render(sparks):
        import contextlib
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp

        captured = []
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(
                pp, "get_rotowire_player_projections", return_value=pd.DataFrame()))
            stack.enter_context(patch.object(pp, "get_ffp_feed",
                                             return_value=_FeedStub()))
            stack.enter_context(patch.object(
                pp, "_blended_pool",
                return_value=TestBlendedPosRankIsLeagueWide._pool()))
            stack.enter_context(patch.object(pp, "_source_rows", return_value=[]))
            stack.enter_context(patch.object(pp, "_render_source_detail"))
            stack.enter_context(patch.object(
                pp, "_trust_figures", return_value={"ok": False, "reason": "stub"}))
            stack.enter_context(patch.object(pp, "_sparkline_column",
                                             return_value=sparks))
            stack.enter_context(patch.object(
                pp, "render_styled_table",
                side_effect=lambda df, **k: captured.append((df, k))))
            stack.enter_context(patch("streamlit.text_input", return_value=""))
            pp.render_blended_projections()
        return captured[-1]

    def test_the_column_lands_on_the_table_and_is_centred(self, mock_streamlit):
        df, kwargs = self._render({1: "<svg>big</svg>", 2: "<svg>mid</svg>",
                                   3: "<svg>small</svg>", 4: "<svg>gk</svg>"})
        assert "Next 6" in df.columns
        assert all(v.startswith("<svg") for v in df["Next 6"]), (
            "the column is present but empty -- populated is the assertion "
            "that matters here"
        )
        assert kwargs.get("text_align", {}).get("Next 6") == "center"

    def test_a_player_with_no_horizon_gets_an_empty_cell(self, mock_streamlit):
        """Not a flat line at zero: FFP legitimately prices fewer players than
        the pool holds, and an unpriced player's window is unknown rather than
        worthless."""
        df, _ = self._render({1: "<svg>big</svg>"})
        by_name = dict(zip(df["Player"], df["Next 6"]))
        assert by_name["Big Mid"].startswith("<svg")
        assert by_name["Small Mid"] == ""
        assert by_name["A Keeper"] == ""

    def test_no_horizon_at_all_omits_the_column_entirely(self, mock_streamlit):
        df, _ = self._render({})
        assert "Next 6" not in df.columns

    def test_the_hidden_join_and_search_columns_never_render(self, mock_streamlit):
        df, _ = self._render({1: "<svg>big</svg>"})
        assert "_pid" not in df.columns
        assert "_search_name" not in df.columns


class TestBoardCharts:
    """Both charts draw from the whole board, with the filter marked.

    The property under test is the one the artifact's rationale rests on: a
    chart re-derived from the filtered rows grades a selection against itself,
    which is the same trap the page's fixed colour ranges and pre-filter ranks
    each exist to avoid. It is also the only coverage of the plotly calls --
    `charts.dark_layout` exists because `update_layout(**DARK_LAYOUT,
    xaxis=...)` raises, and nothing but rendering catches that.
    """

    @staticmethod
    def _board():
        import pandas as pd
        n = 40
        return pd.DataFrame({
            "Player": ["P%d" % i for i in range(n)],
            "Team": (["ARS", "MCI", "LIV", "AVL"] * 10)[:n],
            "Pos": (["G", "D", "M", "F"] * 10)[:n],
            "Proj": [0.2 * i for i in range(n)],
            "Rotowire": [0.2 * i + 0.1 for i in range(n)],
            "FFP": [0.2 * i - 0.1 if i % 7 else 0.2 * i + 2.4 for i in range(n)],
        })

    def test_both_charts_render(self, mock_streamlit):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        with patch("streamlit.plotly_chart") as chart:
            pp._render_board_charts(board, board)
        assert chart.call_count == 2, "expected the scatter and the strips"

    def test_the_scatter_plots_the_whole_board_not_the_filter(self, mock_streamlit):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        shown = board[board["Team"] == "ARS"]
        with patch("streamlit.plotly_chart") as chart:
            pp._agreement_chart(board, shown)
        fig = chart.call_args[0][0]
        plotted = sum(len(t.x) for t in fig.data
                      if t.name and t.name.startswith(("Within", "Apart")))
        assert plotted == len(board), (
            "the scatter drew %d of %d players -- a field re-derived from the "
            "filter grades that selection against itself" % (plotted, len(board))
        )
        assert any(t.name == "Your filter" for t in fig.data), (
            "the filter must be marked on the full field, not substituted for it"
        )

    def test_the_scatter_draws_the_agreement_diagonal(self, mock_streamlit):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        with patch("streamlit.plotly_chart") as chart:
            pp._agreement_chart(board, board)
        fig = chart.call_args[0][0]
        line = next(t for t in fig.data if t.name == "They agree")
        assert line.x[0] == 0 and line.y[0] == 0
        assert line.x[1] == line.y[1], "y=x, or it is not an agreement line"

    def test_the_scatter_separates_the_players_the_sources_disagree_about(
            self, mock_streamlit):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        with patch("streamlit.plotly_chart") as chart:
            pp._agreement_chart(board, board)
        fig = chart.call_args[0][0]
        apart = next(t for t in fig.data if t.name and t.name.startswith("Apart"))
        assert len(apart.x) == sum(1 for i in range(len(board)) if i % 7 == 0)

    def test_too_few_paired_players_says_so_instead_of_drawing(self, mock_streamlit):
        import pandas as pd
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = pd.DataFrame({"Player": ["A"], "Pos": ["M"], "Proj": [3.0],
                              "Rotowire": [3.0], "FFP": [3.1]})
        with patch("streamlit.plotly_chart") as chart:
            pp._agreement_chart(board, board)
        assert chart.call_count == 0

    def test_the_strips_cover_every_position_on_the_board(self, mock_streamlit):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        with patch("streamlit.plotly_chart") as chart:
            pp._position_strip_chart(board, board)
        fig = chart.call_args[0][0]
        boxes = {t.name for t in fig.data if t.type == "box"}
        assert boxes == {"G", "D", "M", "F"}
        for trace in fig.data:
            if trace.type == "box":
                assert len(trace.x) == int((board["Pos"] == trace.name).sum())

    def test_the_strips_mark_a_small_filter(self, mock_streamlit):
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        shown = board.head(3)
        with patch("streamlit.plotly_chart") as chart:
            pp._position_strip_chart(board, shown)
        fig = chart.call_args[0][0]
        marker = next(t for t in fig.data if t.name == "Your filter")
        assert len(marker.x) == 3

    def test_an_unfiltered_board_is_not_marked_against_itself(self, mock_streamlit):
        """Every row ringed is the same as none ringed, and reads as noise."""
        from unittest.mock import patch
        from scripts.fpl import player_projections as pp
        board = self._board()
        with patch("streamlit.plotly_chart") as chart:
            pp._position_strip_chart(board, board)
        fig = chart.call_args[0][0]
        assert not any(t.name == "Your filter" for t in fig.data)
