"""Tests for the My Leagues section on the app home page.

Preseason, the Draft API returns a *full* standings array in which every rank
(and rank_sort, and last_rank) is null. `s.get("rank", 0)` returns None for
those rows -- the default only applies when the key is absent, not when its
value is null -- and the subsequent `rank <= 3` raised TypeError. The caller
wraps snapshot building in `except Exception`, so the entire Draft league
silently disappeared from My Leagues with nothing shown in its place.
"""

from unittest.mock import patch

import pytest

import main


def _draft_details(ranks, draft_status="post"):
    """Draft league payload with the given rank values, one entry per rank."""
    entries = [
        {"id": 100 + i, "entry_id": 900 + i, "entry_name": "Team %d" % i}
        for i in range(len(ranks))
    ]
    standings = [
        {
            "league_entry": 100 + i, "rank": rank, "rank_sort": rank, "last_rank": rank,
            "total": 0 if rank is None else 50 - i,
            "matches_won": 0, "matches_drawn": 0, "matches_lost": 0, "matches_played": 0,
        }
        for i, rank in enumerate(ranks)
    ]
    return {
        "league": {"name": "FPL Friends", "draft_status": draft_status},
        "league_entries": entries,
        "standings": standings,
    }


class TestHasRankedRows:
    def test_all_null_ranks_is_not_ranked(self):
        assert main._has_ranked_rows([{"rank": None}] * 10) is False

    def test_any_real_rank_is_ranked(self):
        assert main._has_ranked_rows([{"rank": None}, {"rank": 2}]) is True

    def test_empty_and_none_are_not_ranked(self):
        assert main._has_ranked_rows([]) is False
        assert main._has_ranked_rows(None) is False

    def test_rank_one_is_not_mistaken_for_falsy(self):
        """Guarding with `if r.get("rank")` instead of `is not None` would drop
        a legitimate 1st place."""
        assert main._has_ranked_rows([{"rank": 1}]) is True


class TestDraftSnapshot:
    def test_preseason_unranked_standings_render_the_info_card(self):
        """The reported bug: the Draft league vanished from My Leagues."""
        with patch.object(main, "get_draft_league_details", return_value=_draft_details([None] * 10)):
            html = main._build_draft_snapshot(11347)
        assert html is not None, "an unranked league must still appear under My Leagues"
        assert "FPL Friends" in html
        assert "10 teams" in html
        assert "Season not started yet" in html

    def test_ranked_standings_render_the_standings_card(self):
        with patch.object(main, "get_draft_league_details", return_value=_draft_details([1, 2, 3, 4])):
            html = main._build_draft_snapshot(11347)
        assert html is not None
        assert "standings-row" in html
        assert "🥇" in html

    def test_a_single_unranked_row_does_not_take_out_the_card(self):
        """Defense in depth: one null rank among real ones must not raise."""
        with patch.object(main, "get_draft_league_details", return_value=_draft_details([1, 2, None, 4])):
            html = main._build_draft_snapshot(11347)
        assert html is not None
        assert "standings-row" in html

    def test_no_entries_and_no_standings_returns_none(self):
        details = {"league": {"name": "Empty"}, "league_entries": [], "standings": []}
        with patch.object(main, "get_draft_league_details", return_value=details):
            assert main._build_draft_snapshot(11347) is None


class TestLeagueCardHtml:
    def test_null_rank_renders_without_raising(self):
        rows = [(None, "Team A", "0", "0W 0D 0L", False),
                (None, "Team B", "0", "0W 0D 0L", True)]
        html = main._build_league_card_html("League", rows, "📋")
        assert "Team A" in html and "Team B" in html

    def test_null_rank_gets_no_medal(self):
        html = main._build_league_card_html(
            "League", [(None, "Team A", "0", None, False)], "📋"
        )
        assert "🥇" not in html

    @pytest.mark.parametrize("rank,medal", [(1, "🥇"), (2, "🥈"), (3, "🥉")])
    def test_podium_ranks_get_medals(self, rank, medal):
        rows = [(r, "Team %d" % r, "0", None, False) for r in (1, 2, 3)]
        html = main._build_league_card_html("League", rows, "📋")
        assert medal in html


class TestRenderLeagueSnapshotsIsResilient:
    def test_a_broken_draft_league_still_shows_the_other_leagues(self, mock_streamlit):
        """The bare `except` around snapshot building must not be the only thing
        standing between one bad league and an empty My Leagues section."""
        with patch.object(main, "_build_draft_snapshot", side_effect=RuntimeError("boom")), \
             patch.object(main, "_build_classic_snapshot", return_value=("c", "<div>classic</div>")), \
             patch.object(main.config, "FPL_DRAFT_LEAGUE_ID", 11347), \
             patch.object(main.config, "FPL_CLASSIC_LEAGUE_IDS", [{"id": 1, "name": "L"}]):
            main._render_league_snapshots()
        rendered = " ".join(str(c) for c in mock_streamlit["markdown"].call_args_list)
        assert "classic" in rendered
