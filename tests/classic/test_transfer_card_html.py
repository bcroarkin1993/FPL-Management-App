"""The Classic transfer cards must survive their own optional fragments.

A blank line ends an HTML block in Markdown. An optional fragment that renders
to "" on a line of its own therefore splits the card in two, and the remainder
-- indented four spaces -- comes out as an indented code block: the user sees a
literal `</div>` under the player's name.

It depends entirely on the data, which is why it shipped: the same card renders
correctly for a 25%-owned player (who gets a "Template" badge) and breaks for a
9%-owned one (who gets "").
"""

import re

import pytest

from scripts.classic.transfers import (
    _render_multi_transfer_plan,
    _render_transfer_suggestions,
)


def _cards(mock_streamlit):
    """Every HTML string the renderer handed to st.markdown."""
    return [c.args[0] for c in mock_streamlit["markdown"].call_args_list
            if c.args and "<div" in str(c.args[0])]


def _assert_renderable(html: str):
    blank = [i for i, line in enumerate(html.splitlines()) if not line.strip()]
    assert not blank, (
        f"blank line(s) at {blank} split the card's HTML block — everything after "
        f"the first one renders as literal text"
    )
    assert html.count("<div") == html.count("</div>"), "unbalanced div tags"


def _plan_leg(**overrides):
    leg = {
        "position": "DEF", "score_diff": 0.472,
        "drop_player": "Ajayi", "drop_full_name": "Semi Ajayi", "drop_team": "HUL",
        "drop_price": "£4.1m", "drop_form": "8.3", "drop_injury": "✓", "drop_season_pts": 40,
        "add_player": "Mitchell", "add_full_name": "Tyrick Mitchell", "add_team": "CRY",
        "add_price": "£4.5m", "add_form": "6.0", "add_proj_pts": "4.2", "add_injury": "✓",
        "add_ownership_badge": "", "rationale": "Part of optimal 2-transfer plan",
        "urgency": "", "ep_delta": None, "price_trend": None, "hit_verdict": None,
        "plan_funds": 107.0, "plan_outlay": 95.0,
    }
    leg.update(overrides)
    return leg


class TestPlanCards:
    def test_survives_an_empty_ownership_badge(self, mock_streamlit):
        """The reported break: a mid-owned add gets no badge, so the line is empty."""
        _render_multi_transfer_plan([_plan_leg(), _plan_leg()], free_transfers=2)

        cards = _cards(mock_streamlit)
        assert len(cards) == 2
        for card in cards:
            _assert_renderable(card)

    def test_badge_still_reaches_the_card(self, mock_streamlit):
        badge = '<span class="t">Template</span>'
        _render_multi_transfer_plan(
            [_plan_leg(add_ownership_badge=badge), _plan_leg()], free_transfers=2)

        cards = _cards(mock_streamlit)
        assert any("Template" in c for c in cards), "the badge was flattened away"
        for card in cards:
            _assert_renderable(card)

    def test_player_text_is_not_run_together(self, mock_streamlit):
        """Flattening must not glue adjacent text onto the tags around it."""
        _render_multi_transfer_plan([_plan_leg(), _plan_leg()], free_transfers=2)

        card = _cards(mock_streamlit)[0]
        text = re.sub(r"<[^>]+>", " ", card)
        assert "Tyrick Mitchell (CRY)" in " ".join(text.split())
        assert "Semi Ajayi (HUL)" in " ".join(text.split())


class TestSuggestionCards:
    def test_survives_no_verdict_and_no_badges(self, mock_streamlit):
        """Both optional rows sit on their own line, and both can be empty."""
        suggestion = _plan_leg(
            hit_verdict=None, price_trend=None, add_ownership_badge="",
            add_ownership_pct=0, ep_delta=None,
        )
        _render_transfer_suggestions([suggestion], free_transfers=1)

        cards = _cards(mock_streamlit)
        assert cards
        for card in cards:
            _assert_renderable(card)
