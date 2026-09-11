"""The hit-verdict row on a Classic transfer suggestion.

It carries two different quantities that are easy to confuse: the gap between
two players' expected points this gameweek, and what a -4 hit leaves of it.
Reported from the app as "FREE  -3.0 pts net (free)", which reads as though
making the transfer costs three points.
"""

import re

from scripts.classic.transfers import _build_hit_verdict_row, _compute_hit_verdict


def _row_text(ep_delta, is_hit, ep_add=1.0, ep_drop=4.0):
    row = _build_hit_verdict_row({
        "hit_verdict": _compute_hit_verdict(ep_delta, is_hit),
        "ep_next_add": ep_add, "ep_next_drop": ep_drop, "ep_delta": ep_delta,
        "add_player": "Mitchell", "drop_player": "Ajayi",
    })
    text = re.sub(r"<[^>]+>", " ", row)
    text = text.replace("&minus;", "-").replace("&rarr;", "->").replace("&nbsp;", " ")
    return " ".join(text.split())


class TestFreeTransfer:
    def test_no_hit_arithmetic_is_offered(self):
        """"pts net (free)" netted off a hit that was never charged."""
        text = _row_text(-3.0, is_hit=False)
        assert "net" not in text.lower()
        assert "hit" not in text.lower()

    def test_the_delta_is_named_as_a_comparison(self):
        text = _row_text(-3.0, is_hit=False)
        assert "Mitchell 1.0 vs Ajayi 4.0" in text, (
            "a bare signed number gives the reader nothing to attach it to"
        )
        assert "-3.0 xPts this GW" in text

    def test_a_gain_reads_as_a_gain(self):
        text = _row_text(1.4, is_hit=False, ep_add=5.4, ep_drop=4.0)
        assert "+1.4 xPts this GW" in text

    def test_verdict_says_only_that_it_is_free(self):
        assert _compute_hit_verdict(-3.0, is_hit=False)["display_str"] == ""
        assert _compute_hit_verdict(-3.0, is_hit=False)["verdict"] == "FREE"


class TestHit:
    def test_the_number_is_labelled_as_post_hit(self):
        text = _row_text(6.5, is_hit=True, ep_add=10.5, ep_drop=4.0)
        assert "+6.5 xPts this GW" in text, "the raw comparison is still shown"
        assert "+2.5 xPts after the -4 hit" in text, "and what the hit leaves of it"

    def test_a_losing_hit_is_signed_not_silently_positive(self):
        text = _row_text(1.0, is_hit=True, ep_add=5.0, ep_drop=4.0)
        assert "-3.0 xPts after the -4 hit" in text
        assert "NO" in text

    def test_verdict_thresholds_are_unchanged(self):
        assert _compute_hit_verdict(6.5, True)["verdict"] == "YES"        # +2.5 net
        assert _compute_hit_verdict(4.5, True)["verdict"] == "MARGINAL"   # +0.5 net
        assert _compute_hit_verdict(1.0, True)["verdict"] == "NO"         # -3.0 net
        assert _compute_hit_verdict(6.5, True)["net_gain"] == 2.5


class TestColour:
    def test_a_loss_is_not_painted_green(self):
        row = _build_hit_verdict_row({
            "hit_verdict": _compute_hit_verdict(-3.0, False),
            "ep_next_add": 1.0, "ep_next_drop": 4.0, "ep_delta": -3.0,
            "add_player": "Mitchell", "drop_player": "Ajayi",
        })
        # The delta itself must carry the warning colour, whatever the badge says.
        assert "#f87171" in row

    def test_a_gain_is_green(self):
        row = _build_hit_verdict_row({
            "hit_verdict": _compute_hit_verdict(1.4, False),
            "ep_next_add": 5.4, "ep_next_drop": 4.0, "ep_delta": 1.4,
            "add_player": "Mitchell", "drop_player": "Ajayi",
        })
        assert "#4ecca3" in row


class TestAbsent:
    def test_no_verdict_renders_nothing(self):
        assert _build_hit_verdict_row({"hit_verdict": None}) == ""
