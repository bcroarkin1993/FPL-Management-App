"""The shared dark card.

Four hand-rolled versions of this had drifted across the app, which is why
making two cards level needed a tuned pixel constant per page. These tests pin
the two properties that make a row line up, and the one that keeps a card from
breaking out of its own markup.
"""

from scripts.common import cards


class TestTheShell:
    def test_it_is_one_line(self):
        """A blank line ends an HTML block in Markdown, and what follows is
        parsed afresh -- which is how a card's closing tag reaches the screen as
        literal text. Every card goes through `compact_html` for that reason."""
        html = cards.card(cards.eyebrow("Label") + cards.headline("Title")
                          + cards.push(cards.note("")))
        assert "\n" not in html

    def test_an_empty_body_still_renders_a_card(self):
        html = cards.card("")
        assert html.startswith("<div") and html.endswith("</div>")

    def test_a_min_height_is_applied(self):
        assert "min-height:145px" in cards.card("x", min_height=145)

    def test_no_height_means_no_floor(self):
        assert "min-height:0px" in cards.card("x")

    def test_a_negative_height_is_clamped(self):
        """`_card_height` subtracts the selectbox, which can go negative on a
        one-row strip; a negative CSS length is simply ignored, so clamp."""
        assert "min-height:0px" in cards.card("x", min_height=-40)

    def test_an_accent_draws_a_left_edge(self):
        assert "border-left:4px solid #ff4757" in cards.card("x", accent="#ff4757")

    def test_it_is_a_flex_column_so_push_can_reach_the_bottom(self):
        """Without this a stretched card renders its content at the top and the
        rest as dead space -- the look this module exists to remove."""
        html = cards.card(cards.push("footer"), min_height=200)
        assert "flex-direction:column" in html
        assert "margin-top:auto" in html


class TestARowLinesUp:
    def test_every_card_in_a_row_gets_the_same_height(self):
        row = cards.card_row(["a", "bb", "ccc"], min_height=160)
        assert len(row) == 3
        assert all("min-height:160px" in html for html in row)

    def test_the_row_survives_an_empty_sequence(self):
        assert cards.card_row([]) == []


class TestTheBlocks:
    def test_a_stat_carries_its_value_and_label(self):
        html = cards.stat("742", "total pts")
        assert "742" in html and "total pts" in html

    def test_stats_wrap_rather_than_overflow(self):
        """Three stats in a third of the page is the actual layout."""
        assert "flex-wrap:wrap" in cards.stats([cards.stat("1", "a")])

    def test_a_badge_is_coloured_from_one_value(self):
        html = cards.badge("H2H", cards.ACCENT_ALT)
        assert html.count(cards.ACCENT_ALT) >= 2      # text and border

    def test_the_palette_is_shared_rather_than_restated(self):
        """The point of the module: one set of colours for every card."""
        for value in (cards.ACCENT, cards.WARN, cards.BAD, cards.MUTED):
            assert value.startswith("#")
