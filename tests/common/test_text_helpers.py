"""Tests for scripts/common/text_helpers.py."""

from datetime import datetime, timedelta

import pytest

from scripts.common.text_helpers import (
    TEAM_FULL_TO_SHORT,
    TZ_ET,
    _to_short_team_code,
    compact_html,
    format_deadline_stamp,
    format_last_updated,
    format_time_until,
    to_display_name,
)


class TestToDisplayName:
    """The app-wide player-name format.

    Neither raw FPL field is presentable: the full legal name is what nobody
    says out loud, and web_name is often abbreviated or too terse to stand
    alone. Every case below is a real player from the FPL bootstrap.
    """

    @pytest.mark.parametrize("first, second, web, expected", [
        # Abbreviated web_name -- expand the initial from first_name.
        ("Bruno", "Borges Fernandes", "B.Fernandes", "Bruno Fernandes"),
        ("Alisson", "Becker", "A.Becker", "Alisson Becker"),
        ("Benoît", "Badiashile Mukinayi", "B.Badiashile", "Benoît Badiashile"),
        # The ordinary case: web_name is the surname.
        ("David", "Raya Martín", "Raya", "David Raya"),
        ("Matheus", "Santos Carneiro da Cunha", "Cunha", "Matheus Cunha"),
        ("Dominic", "Solanke-Mitchell", "Solanke", "Dominic Solanke"),
        ("Erling", "Haaland", "Haaland", "Erling Haaland"),
        # Mononym -- web_name already is the whole name people use.
        ("Gabriel", "dos Santos Magalhães", "Gabriel", "Gabriel"),
        # Surname sits in first_name, so web_name alone would lose "Igor".
        ("Igor Thiago", "Nascimento Rodrigues", "Thiago", "Igor Thiago"),
    ])
    def test_returns_the_common_name(self, first, second, web, expected):
        assert to_display_name(first, second, web) == expected

    def test_missing_web_name_falls_back_to_the_full_name(self):
        assert to_display_name("Erling", "Haaland", None) == "Erling Haaland"

    def test_missing_first_name_falls_back_to_web_name(self):
        assert to_display_name(None, None, "Haaland") == "Haaland"

    def test_all_blank_is_empty_not_an_error(self):
        assert to_display_name(None, None, None) == ""




class TestFormatLastUpdated:
    """Rendering a source's publish time, with its age."""

    @staticmethod
    def _ago(**kwargs):
        return datetime.now(TZ_ET) - timedelta(**kwargs)

    def test_none_is_unknown(self):
        assert format_last_updated(None) == "Unknown"

    def test_includes_the_timestamp_and_zone(self):
        when = datetime(2026, 8, 20, 10, 54, tzinfo=TZ_ET)
        out = format_last_updated(when, include_age=False)
        assert out == "Aug 20, 2026 10:54 AM ET"

    def test_pm_renders_as_pm(self):
        when = datetime(2026, 8, 20, 17, 6, tzinfo=TZ_ET)
        assert "5:06 PM ET" in format_last_updated(when, include_age=False)

    @pytest.mark.parametrize("kwargs, expected", [
        ({"minutes": 8}, "8m ago"),
        ({"hours": 3}, "3h ago"),
        ({"days": 1}, "1 day ago"),
        ({"days": 9}, "9 days ago"),
    ])
    def test_age_is_reported(self, kwargs, expected):
        assert expected in format_last_updated(self._ago(**kwargs))

    def test_future_timestamp_does_not_render_negative_age(self):
        """Our clock and the source's can disagree by a few minutes."""
        future = datetime.now(TZ_ET) + timedelta(minutes=5)
        assert "just now" in format_last_updated(future)

    def test_age_can_be_suppressed(self):
        assert "ago" not in format_last_updated(self._ago(hours=3), include_age=False)


class TestToShortTeamCode:
    """Team-name -> short-code mapping.

    A missing club is not loud. Newly-promoted Leeds fell through to the naive
    3-letter guess, which was right by luck ("LEE") but logged a warning on every
    row of every player table. A club whose guess is *wrong* (the classic case is
    "Sheffield Utd" -> "SHE" rather than "SHU") would instead break name matching
    silently, since matching is scoped by team.
    """

    def test_leeds_maps_without_guessing(self):
        assert TEAM_FULL_TO_SHORT["Leeds"] == "LEE"
        assert _to_short_team_code("Leeds") == "LEE"

    def test_common_leeds_variants_map(self):
        """Rotowire and FFP each spell club names their own way."""
        for variant in ("Leeds", "Leeds United", "Leeds Utd"):
            assert _to_short_team_code(variant) == "LEE", variant

    def test_existing_short_codes_pass_through(self):
        assert _to_short_team_code("LEE") == "LEE"
        assert _to_short_team_code("MCI") == "MCI"

    def test_relegated_clubs_are_retained_for_historical_pages(self):
        """The dict is append-only across seasons; Season Wrapped reads back."""
        for club in ("Leicester", "Southampton", "West Ham", "Wolves"):
            assert club in TEAM_FULL_TO_SHORT, club

    def test_unknown_team_warns_only_once(self, caplog):
        """This is the reported symptom: one unmapped club, dozens of identical
        log lines per page load, because this runs per row."""
        import scripts.common.text_helpers as th

        th._UNKNOWN_TEAMS_WARNED.discard("Wrexham")
        with caplog.at_level("WARNING", logger="fpl_app.text_helpers"):
            for _ in range(10):
                _to_short_team_code("Wrexham")

        warnings = [r for r in caplog.records if "Wrexham" in r.getMessage()]
        assert len(warnings) == 1, "expected one warning, got %d" % len(warnings)

    def test_every_mapped_code_is_three_letters(self):
        bad = {k: v for k, v in TEAM_FULL_TO_SHORT.items() if len(v) != 3 or not v.isupper()}
        assert not bad, "malformed short codes: %s" % bad


class TestCompactHtml:
    """A blank line ends an HTML block in Markdown.

    Everything after it is parsed as fresh Markdown, and indented four spaces
    that is an indented code block -- which is how a card's closing `</div>`
    came to render as visible text whenever an optional fragment on its own
    line was empty.
    """

    def test_drops_blank_lines(self):
        assert compact_html("<div>\n\n  <p>hi</p>\n\n</div>") == "<div> <p>hi</p> </div>"

    def test_an_empty_fragment_leaves_no_gap(self):
        card = '<div class="c">\n    <span>x</span>\n    {}\n</div>'.format("")
        out = compact_html(card)
        assert "\n" not in out
        assert out == '<div class="c"> <span>x</span> </div>'

    def test_keeps_a_space_between_fragments(self):
        """A CSS declaration split across source lines must not be welded shut."""
        out = compact_html('<div style="border: 1px solid #444;\n  background: #000;">x</div>')
        assert "#444; background:" in out

    def test_single_line_input_is_unchanged(self):
        assert compact_html("<div>x</div>") == "<div>x</div>"

    def test_empty_input(self):
        assert compact_html("") == ""


class TestFormatDeadlineStamp:
    """Both deadline cards rendered the weekday alone.

    "Fri 6:00 AM ET" is read as *this* Friday. Measured 2026-09-26 the GW6 waiver
    deadline was Friday 9 October, thirteen days out across an international
    break -- and GW6 and GW7, a week apart, rendered identically, so the card
    could not distinguish the deadline you have from the one after it.
    """

    def test_it_carries_the_weekday_and_the_date(self):
        stamp = format_deadline_stamp(datetime(2026, 10, 9, 6, 0, tzinfo=TZ_ET))
        assert stamp == "Fri Oct 9, 6:00 AM ET"

    def test_a_single_digit_day_does_not_eat_the_hours_zero(self):
        """The hack this replaced was `replace(" 0", " ", 1)` over the whole
        string, which strips the first zero it finds *anywhere* -- on
        "Fri Oct 09, 06:00" that is the day, leaving the hour padded."""
        assert format_deadline_stamp(
            datetime(2026, 10, 19, 6, 0, tzinfo=TZ_ET)) == "Mon Oct 19, 6:00 AM ET"
        assert format_deadline_stamp(
            datetime(2026, 11, 1, 12, 5, tzinfo=TZ_ET)) == "Sun Nov 1, 12:05 PM ET"

    def test_midnight_and_noon_read_as_twelve(self):
        assert "12:00 AM" in format_deadline_stamp(
            datetime(2026, 10, 9, 0, 0, tzinfo=TZ_ET))
        assert "12:00 PM" in format_deadline_stamp(
            datetime(2026, 10, 9, 12, 0, tzinfo=TZ_ET))

    def test_nothing_usable_renders_empty(self):
        assert format_deadline_stamp(None) == ""
        assert format_deadline_stamp("not a datetime") == ""


class TestFormatTimeUntil:
    """The forward-looking sibling of `format_last_updated`'s "(3h ago)"."""

    NOW = datetime(2026, 9, 26, 14, 0, tzinfo=TZ_ET)

    def test_days_away(self):
        assert format_time_until(self.NOW + timedelta(days=13), now=self.NOW) == "in 13 days"

    def test_tomorrow_is_calendar_based_not_a_24_hour_block(self):
        """A 06:00 deadline tomorrow is "tomorrow" whether it is 16 hours away
        or 4 -- a manager reads the day, not the elapsed hours."""
        assert format_time_until(
            self.NOW + timedelta(hours=16), now=self.NOW) == "tomorrow"

    def test_hours_and_minutes(self):
        assert format_time_until(self.NOW + timedelta(hours=6), now=self.NOW) == "in 6 hours"
        assert format_time_until(self.NOW + timedelta(hours=1), now=self.NOW) == "in 1 hour"
        assert format_time_until(self.NOW + timedelta(minutes=40), now=self.NOW) == "in 40 min"

    def test_a_passed_deadline_says_so(self):
        assert format_time_until(self.NOW - timedelta(hours=2), now=self.NOW) == "passed"

    def test_nothing_usable_renders_empty(self):
        """Empty rather than a placeholder, so a caller can append it blind."""
        assert format_time_until(None) == ""
        assert format_time_until("not a datetime") == ""
