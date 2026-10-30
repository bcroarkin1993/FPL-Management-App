"""Every frame that publishes `Player_ID` must publish it in the Classic space.

**The failure these pin is invisible by construction.** The two FPL APIs
publish the same 669 players and number 64 of them differently, so an id taken
from a Draft-only endpoint and looked up in Classic-keyed data does not fail to
match -- it matches the *wrong player*, and every value on the resulting row
stays plausible. See `scripts/common/element_ids.py`.

`tests/live/test_element_id_spaces.py` catches it after the fact, by comparing a
real frame's ids against its own names. These are the offline half: they pin
*where the ids came from*, which is the thing a reviewer can actually check.
"""

import ast
import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

DRAFT_BOOTSTRAP = "draft.premierleague.com/api/bootstrap-static"

#: Pages whose frames are joined to Classic-keyed data, so they must not read
#: the Draft bootstrap at all. `fpl/injuries.py` is the one that regressed: it
#: built the availability frame from the Draft bootstrap for a year, and
#: `availability.py`, `projected_lineups.py` and its own `_attach_pl_injuries`
#: all join that frame's `Player_ID` against the Classic bootstrap.
#:
#: Deliberately not every module. `fpl_draft_api.py`, `element_ids.py`,
#: `waiver_alerts.py` and `draft/waiver_wire.py` read the Draft bootstrap
#: legitimately -- the first two to speak to the Draft API and bridge away from
#: it, the others for league and club data that carries no element id.
CLASSIC_KEYED_PAGES = [
    SCRIPTS / "fpl" / "injuries.py",
    SCRIPTS / "fpl" / "availability.py",
    SCRIPTS / "fpl" / "projected_lineups.py",
    SCRIPTS / "fpl" / "gameweek_review.py",
]


def _bootstrap(*, first="Christos", second="Tzolis", pid=572, code=500001):
    """A one-player bootstrap. The default is a real player the two APIs
    number differently, so a test using it is exercising the hazard."""
    return {
        "teams": [{"id": 1, "short_name": "ARS", "name": "Arsenal"}],
        "elements": [{
            "id": pid, "code": code, "first_name": first, "second_name": second,
            "web_name": second, "team": 1, "element_type": 3, "status": "a",
            "chance_of_playing_this_round": 100,
            "chance_of_playing_next_round": 100,
            "news": "", "news_added": "", "total_points": 10, "now_cost": 50,
        }],
        "events": [],
    }


class TestTheAvailabilityFrame:
    """`get_fpl_availability_df` -- the frame three pages join on."""

    def test_its_ids_come_from_the_classic_bootstrap(self):
        from scripts.fpl import injuries
        with patch.object(injuries, "get_classic_bootstrap_static",
                          return_value=_bootstrap(pid=572)) as boot:
            df = injuries.get_fpl_availability_df()
        assert boot.called, "the availability frame must read the Classic bootstrap"
        assert list(df["Player_ID"]) == [572]
        assert list(df["Player"]) == ["Christos Tzolis"]

    def test_an_unreachable_bootstrap_yields_an_empty_frame_not_a_crash(self):
        """It is wrapped in a try/except by `availability.py`, but a raise here
        would also take the Injuries page down, which has no such guard."""
        from scripts.fpl import injuries
        with patch.object(injuries, "get_classic_bootstrap_static", return_value=None):
            df = injuries.get_fpl_availability_df()
        assert df.empty


class TestTheDraftActualLineup:
    """`get_team_actual_lineup` -- its id reaches `simulate_auto_subs`."""

    def _run(self, mapping):
        from scripts.common import fpl_draft_api as api

        picks = {"picks": [{"element": 555, "position": 1}]}

        class _Resp:
            def raise_for_status(self): pass
            def json(self): return picks

        with patch.object(api, "get_fpl_player_mapping", return_value={
                    555: {"Player": "Antonio Silva", "Web_Name": "Silva",
                          "Team": "BOU", "Position": "D"}}), \
             patch.object(api.requests, "get", return_value=_Resp()), \
             patch("scripts.common.element_ids.draft_to_classic_ids",
                   return_value=mapping):
            return api.get_team_actual_lineup(1, 9)

    def test_the_published_id_is_translated_to_the_classic_space(self):
        """Draft 555 is Antonio Silva; Classic 555 is Remi Matthews. The row is
        named after Silva, so the id it carries must be Silva's Classic id."""
        out = self._run({555: 566})
        assert list(out["Player_ID"]) == [566]
        assert list(out["Player"]) == ["Antonio Silva"]

    def test_an_untranslatable_id_becomes_none_rather_than_passing_through(self):
        out = self._run({1: 1})
        assert out["Player_ID"].isna().all()

    def test_the_player_is_still_looked_up_in_the_draft_space(self):
        """The lookup must happen *before* translation -- the picks and the
        player map are both Draft-keyed. Translating first would name the row
        after whoever holds that id in the other API."""
        out = self._run({555: 566})
        assert list(out["Team"]) == ["BOU"]
        assert list(out["Position"]) == ["D"]


class TestTheDraftGameweekReview:
    def test_it_cannot_reach_the_classic_bootstrap(self):
        """It used to be handed the Classic bootstrap and to look Draft pick
        ids up in it, *preferring* that over the Draft-keyed player map. The fix
        is that the wrong-space object is no longer a parameter: a space cannot
        be crossed by code that cannot reach the other space."""
        import inspect
        from scripts.fpl import gameweek_review
        params = inspect.signature(gameweek_review._get_draft_gw_data).parameters
        assert "bootstrap" not in params, (
            "_get_draft_gw_data takes a bootstrap again -- it resolves Draft "
            "pick ids, and the only bootstrap its caller holds is Classic")


@pytest.mark.parametrize("path", CLASSIC_KEYED_PAGES, ids=lambda p: p.name)
def test_a_classic_keyed_page_does_not_read_the_draft_bootstrap(path):
    """The guard with teeth: it fails on the regression rather than on its
    symptom. Every field the availability frame reads is published by both
    APIs, so switching the URL back would change nothing visible and no
    assertion about the frame's *contents* could tell."""
    source = path.read_text()
    assert DRAFT_BOOTSTRAP not in source, (
        "%s reads the Draft bootstrap, but its frames are joined to "
        "Classic-keyed data. The two APIs number 64 of 669 players "
        "differently, so the ids would resolve to the wrong player -- "
        "silently. See scripts/common/element_ids.py." % path.name
    )


def test_the_guard_can_actually_fail(tmp_path):
    """A guard that cannot fail is indistinguishable from one that does not run."""
    bad = tmp_path / "offender.py"
    bad.write_text('URL = "https://draft.premierleague.com/api/bootstrap-static"\n')
    with pytest.raises(AssertionError):
        test_a_classic_keyed_page_does_not_read_the_draft_bootstrap(bad)


def test_the_registry_that_stamped_draft_ids_is_gone():
    """`PlayerRegistry` built from the Draft bootstrap and stamped its ids as
    `Player_ID`. Nothing called it, so it was not a live bug but a loaded one.
    Pinned as deleted so it cannot come back as a convenience."""
    import scripts.common.player_matching as pm
    for name in ("PlayerRegistry", "get_player_registry", "PlayerRecord",
                 "add_normalized_name_column", "normalize_for_merge"):
        assert not hasattr(pm, name), (
            "%s is back; if it is needed, resolve names with ReferenceMatcher "
            "and take ids from the Classic bootstrap" % name)
