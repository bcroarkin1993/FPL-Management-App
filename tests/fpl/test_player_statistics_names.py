"""Player Statistics renders names; it must not *match* them.

Reported from the app: João Pedro was missing from the tables. He was not
missing -- he was rendering as **Maxim De Cuyper**, while three other players
rendered as "Joao Pedro".

The page used to fuzzy-match every FPL player against Rotowire's 220-row weekly
table and rename him to the best hit at a threshold of 80, scored with
`fuzzywuzzy.WRatio`. WRatio falls back to partial-token ratios once two strings
differ enough in length, and one shared token then carries it: "Maxim De
Cuyper" scores **86** against "João Pedro Junqueira de Jesus" on the word *de*.

Measured live, 87 of 667 players were renamed and 66 rows ended up sharing a
name with another player -- "Maxim De Cuyper" was applied to eleven of them,
"Bruno Fernandes" and "Gabriel" to three each, and Alex Palmer rendered as Cole
Palmer. Afterwards: zero duplicate rendered names.

Nothing needs a name match to display a name. `to_display_name()` builds the
common name out of the bootstrap's own fields.
"""

import pandas as pd
import pytest


#: (first_name, second_name, web_name, the name that should render).
BOOTSTRAP_NAMES = [
    # The reported player: a tilde, and a surname FPL files in full.
    ("João Pedro", "Junqueira de Jesus", "João Pedro", "João Pedro"),
    # The name he was being renamed to.
    ("Maxim", "De Cuyper", "De Cuyper", "Maxim De Cuyper"),
    # web_name too terse to stand alone.
    ("Bruno", "Borges Fernandes", "B.Fernandes", "Bruno Fernandes"),
    ("Alisson", "Ramses Becker", "A.Becker", "Alisson Becker"),
    # A mononym.
    ("Gabriel", "dos Santos Magalhães", "Gabriel", "Gabriel"),
    # The surname collision this app has paid for twice.
    ("Cole", "Palmer", "Palmer", "Cole Palmer"),
    ("Alex", "Palmer", "Palmer", "Alex Palmer"),
]


class TestPullFplPlayerStatsCarriesBothNames:
    """`player` is the matching key, `display_name` is what renders."""

    @pytest.mark.parametrize("first,second,web,expected", BOOTSTRAP_NAMES)
    def test_display_name_is_the_common_name(self, first, second, web, expected):
        from scripts.common.text_helpers import to_display_name
        assert to_display_name(first, second, web) == expected

    def test_the_two_palmers_keep_their_own_names(self):
        """A rendered name that resolves to two players is the whole bug."""
        from scripts.common.text_helpers import to_display_name
        assert (to_display_name("Cole", "Palmer", "Palmer")
                != to_display_name("Alex", "Palmer", "Palmer"))


class TestThePageRendersTheDisplayName:
    """The page reads `player` everywhere, so the display name is written into
    it. Asserted through the page function, because the defect was in the
    wiring rather than in `to_display_name`."""

    @staticmethod
    def _stats():
        return pd.DataFrame({
            "id": [165, 115],
            "web_name": ["João Pedro", "De Cuyper"],
            "player": ["João Pedro Junqueira de Jesus", "Maxim De Cuyper"],
            "display_name": ["João Pedro", "Maxim De Cuyper"],
            "position_abbrv": ["FWD", "DEF"],
            "team_name": ["Chelsea", "Brighton"],
            "team_name_abbrv": ["CHE", "BHA"],
            "minutes": [431, 490],
            "total_points": [45, 53],
        })

    def _render(self, stats):
        from unittest.mock import patch
        import scripts.fpl.player_statistics as ps

        seen = {}

        def _capture(df, *a, **k):
            seen["df"] = df.copy()

        with patch.object(ps, "pull_fpl_player_stats", return_value=stats), \
             patch.object(ps, "get_fixture_difficulty_grid",
                          return_value=(pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=float))), \
             patch.object(ps, "display_advanced_stats_table", side_effect=_capture), \
             patch.object(ps, "display_top_goal_scorers"), \
             patch.object(ps, "display_top_assisters"), \
             patch.object(ps, "display_top_clean_sheets"), \
             patch.object(ps, "display_expected_vs_actual_goals"), \
             patch.object(ps, "display_boxplot_point_distribution"), \
             patch.object(ps, "display_set_piece_takers"):
            ps.show_player_stats_page()
        return seen.get("df")

    def test_joao_pedro_renders_under_his_own_name(self, mock_streamlit):
        rendered = self._render(self._stats())
        assert rendered is not None, "the advanced table was never rendered"
        names = rendered["player"].tolist()
        assert "João Pedro" in names
        assert names.count("Maxim De Cuyper") == 1, (
            "the reported bug: João Pedro rendered as Maxim De Cuyper, and "
            "that name was applied to eleven different players")

    def test_no_two_players_render_under_one_name(self, mock_streamlit):
        rendered = self._render(self._stats())
        assert not rendered["player"].duplicated().any()

    def test_a_frame_with_no_display_name_still_renders(self, mock_streamlit):
        """The column is new; a cached or hand-built frame may predate it."""
        stats = self._stats().drop(columns=["display_name"])
        rendered = self._render(stats)
        assert rendered["player"].tolist() == [
            "João Pedro Junqueira de Jesus", "Maxim De Cuyper"]

    def test_a_blank_display_name_falls_back_to_the_legal_name(self, mock_streamlit):
        """Never render an empty cell where a name belongs."""
        stats = self._stats()
        stats.loc[0, "display_name"] = ""
        rendered = self._render(stats)
        assert "João Pedro Junqueira de Jesus" in rendered["player"].tolist()

    def test_the_page_no_longer_reads_a_projection_source_at_all(self):
        """It fetched Rotowire solely to rename players against it."""
        import scripts.fpl.player_statistics as ps
        assert not hasattr(ps, "clean_fpl_player_names")
        assert not hasattr(ps, "get_rotowire_player_projections")


class TestTheBrokenHelpersStayDeleted:
    def test_they_are_gone_from_player_matching(self):
        """Left deleted rather than deprecated so nothing reaches for them."""
        import scripts.common.player_matching as pm
        for name in ("clean_fpl_player_names", "find_best_match",
                     "_fuzzy_match_player"):
            assert not hasattr(pm, name), (
                "%s renamed players onto a fuzzy name match; display names "
                "come from to_display_name() and projections from "
                "analytics.attach_weekly_projections()" % name)
