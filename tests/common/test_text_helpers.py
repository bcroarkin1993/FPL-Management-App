"""Tests for scripts/common/text_helpers.py."""

from datetime import datetime, timedelta

import pytest

from scripts.common.text_helpers import (
    TEAM_FULL_TO_SHORT,
    TZ_ET,
    _to_short_team_code,
    format_last_updated,
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
