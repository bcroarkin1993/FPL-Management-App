"""Price Changes — FPL's own prediction, and the states that make it readable.

The page reported only changes that had **already happened**, which is history
by the time a manager reads it. FPL publishes the forward view itself -- five
fields nothing in the app read -- so the useful half needed surfacing, not
modelling. Measured on 2026-11-06, all five were populated for all 670 players.
"""

import pandas as pd
import pytest

from scripts.fpl.price_changes import (
    _change_state,
    _format_change,
    _format_momentum,
    _format_price,
    _progress_width,
    _watchlist_html,
)


def _row(**kw):
    return pd.Series(kw)


class TestChangeState:
    def test_past_the_threshold_either_way(self):
        assert _change_state(_row(Change_Pct=105.5)) == "rise"
        assert _change_state(_row(Change_Pct=-139.5)) == "fall"

    def test_short_of_it_is_steady(self):
        assert _change_state(_row(Change_Pct=92.9)) == "steady"
        assert _change_state(_row(Change_Pct=-95.0)) == "steady"

    def test_locked_beats_the_threshold(self):
        """The one case where a bare percentage reads as the *opposite* of the
        truth: FPL allows one change per player per day, so a player who has
        already moved sits above 100% and cannot move again. 39 of 670 were in
        that state live."""
        assert _change_state(_row(Change_Pct=156.9,
                                  Locked_Until="2026-11-11T21:00:09Z")) == "locked"

    def test_calibrating_beats_the_threshold_too(self):
        """FPL is saying its own number is not yet usable, and a progress bar
        that does not say so is a confident claim the source disclaims."""
        assert _change_state(_row(Change_Pct=120.0, Calibrating=True)) == "calibrating"

    def test_no_published_figure_is_steady_not_a_guess(self):
        assert _change_state(_row()) == "steady"
        assert _change_state(_row(Change_Pct=float("nan"))) == "steady"


class TestMomentum:
    def test_it_renders_the_published_rate(self):
        assert _format_momentum(41801) == "+41.8k/h"
        assert _format_momentum(-10054) == "-10.1k/h"
        assert _format_momentum(430) == "+430/h"

    def test_nothing_to_say_renders_nothing(self):
        assert _format_momentum(0) == ""
        assert _format_momentum(None) == ""
        assert _format_momentum("nonsense") == ""

    def test_it_is_not_a_countdown(self):
        """Deliberately a rate, not an ETA. Converting it needs the threshold
        expressed in transfers, which FPL does not publish -- so a countdown
        would rest on an assumption this page cannot check."""
        import inspect
        from scripts.fpl import price_changes as pc
        assert "_eta" not in dir(pc), (
            "an ETA was removed on purpose; reinstating it needs a published "
            "transfers-to-threshold figure")


class TestProgressWidth:
    def test_it_clamps_the_bar_but_not_the_number(self):
        """A real -139% must not draw outside its track; the figure beside the
        bar still reports it."""
        assert _progress_width(-139.5) == 100.0
        assert _progress_width(156.9) == 100.0
        assert _progress_width(60.6) == pytest.approx(60.6)

    def test_missing_is_empty_not_full(self):
        assert _progress_width(None) == 0.0
        assert _progress_width(float("nan")) == 0.0


class TestWatchlistHtml:
    def _rows(self):
        return pd.DataFrame([
            {"Display_Name": "Gabriel", "Player": "Gabriel", "Team": "ARS",
             "Pos": "D", "now_cost": 80, "Change_Pct": 105.5, "Hourly": 7842,
             "cost_change_event": 0},
            {"Display_Name": "Wissa", "Player": "Wissa", "Team": "NEW",
             "Pos": "F", "now_cost": 60, "Change_Pct": -139.5, "Hourly": -10054,
             "cost_change_event": 0},
        ])

    def test_it_is_balanced_and_inline(self):
        html = _watchlist_html(self._rows())
        assert html.count("<div") == html.count("</div>")
        assert html.count("<span") == html.count("</span>")

    def test_a_rise_and_a_fall_are_styled_apart(self):
        html = _watchlist_html(self._rows())
        assert "pc-fill-rise" in html and "pc-fill-fall" in html

    def test_it_prints_the_real_percentage(self):
        html = _watchlist_html(self._rows())
        assert "+106%" in html or "+105%" in html
        assert "-140%" in html or "-139%" in html

    def test_a_player_who_already_moved_says_so(self):
        """Evanilson read +160% having already risen 0.1 this gameweek. A second
        change on a later day is legal, so the figure is not wrong -- but the row
        said nothing about it."""
        rows = self._rows()
        rows.loc[0, "cost_change_event"] = 1
        html = _watchlist_html(rows)
        assert "TODAY" in html and "+£0.1m" in html

    def test_locked_and_calibrating_are_flagged(self):
        rows = self._rows()
        rows.loc[0, "Locked_Until"] = "2026-11-11T21:00:09Z"
        rows.loc[1, "Calibrating"] = True
        html = _watchlist_html(rows)
        assert "LOCKED" in html and "CALIBRATING" in html


class TestFormatting:
    def test_price(self):
        assert _format_price(100) == "£10.0m"
        assert _format_price(45) == "£4.5m"

    def test_change_is_signed(self):
        assert _format_change(1) == "+£0.1m"
        assert _format_change(-2) == "-£0.2m"
