"""The two FPL APIs' element id spaces, checked against the live feeds.

Contract for this directory: unreachable -> SKIP, reachable but implausible ->
FAIL. What is pinned here is the *bridge*, not the disagreement: if FPL ever
reconciles the ids, `code` still works and these stay green.
"""

import pytest
import requests

from tests.live.conftest import skip_if_unreachable  # noqa: F401

DRAFT = "https://draft.premierleague.com/api/bootstrap-static"


@pytest.fixture(scope="module")
def bootstraps():
    from scripts.common.fpl_classic_api import get_classic_bootstrap_static
    try:
        draft = requests.get(DRAFT, timeout=30).json()
        classic = get_classic_bootstrap_static()
    except Exception as exc:                              # transport only
        pytest.skip("FPL bootstrap unreachable: %s" % exc)
    if not draft.get("elements") or not classic.get("elements"):
        pytest.skip("a bootstrap returned no elements")
    return draft, classic


def _name(el):
    return ("%s %s" % (el.get("first_name", ""), el.get("second_name", ""))).strip()


def test_code_is_a_sound_bridge(bootstraps):
    """`code` must identify the same player in both APIs.

    This is the load-bearing assertion: every cross-API element join in the app
    goes through it. If it ever breaks there is no safe way to join the two.
    """
    draft, classic = bootstraps
    d = {int(e["code"]): _name(e) for e in draft["elements"] if e.get("code")}
    c = {int(e["code"]): _name(e) for e in classic["elements"] if e.get("code")}
    shared = set(d) & set(c)
    assert len(shared) > 500, "barely any shared codes: %d" % len(shared)
    bad = [k for k in shared if d[k] != c[k]]
    assert not bad, (
        "`code` no longer identifies the same player in both APIs (%d "
        "mismatches, e.g. %r) -- every cross-API id join in the app rests on "
        "this" % (len(bad), [(k, d[k], c[k]) for k in bad[:3]])
    )


def test_the_translation_map_covers_the_pool(bootstraps):
    from scripts.common.element_ids import draft_to_classic_ids
    draft, _ = bootstraps
    m = draft_to_classic_ids()
    assert len(m) > 500, "translation resolved only %d ids" % len(m)
    coverage = len(m) / len(draft["elements"])
    assert coverage > 0.95, "translated only %.1f%% of Draft elements" % (coverage * 100)


def test_the_id_spaces_still_disagree(bootstraps):
    """Documents *why* the translation layer exists.

    Not an error either way: if this ever reports 0 the APIs have been
    reconciled and the translation becomes an identity map, which is harmless.
    It is here so the change is noticed rather than assumed.
    """
    draft, classic = bootstraps
    d = {int(e["id"]): _name(e) for e in draft["elements"]}
    c = {int(e["id"]): _name(e) for e in classic["elements"]}
    shared = set(d) & set(c)
    disagree = [p for p in shared if d[p] != c[p]]
    print("\nelement ids whose meaning differs between the APIs: %d of %d"
          % (len(disagree), len(shared)))
    assert len(disagree) < len(shared) * 0.5, (
        "more than half the ids disagree (%d of %d) -- that is not the "
        "late-signing drift this is known for" % (len(disagree), len(shared))
    )


# ---------------------------------------------------------------------------
# Does any real frame carry an id for a player it is not named after?
#
# This is the layer that catches the *next* occurrence. Four separate surfaces
# had already shipped with a cross-space id before the check existed, and none
# was visible in the numbers -- the id resolves to somebody on both sides, so
# nothing raises and every value stays plausible.
# ---------------------------------------------------------------------------

def _names():
    from scripts.common.element_ids import classic_id_names
    ref = classic_id_names()
    if not ref:
        pytest.skip("Classic bootstrap unreachable")
    return ref


def _check(df, label, name_col="Player"):
    from scripts.common.data_validation import check_element_id_names, raise_on_error
    raise_on_error(check_element_id_names(df, _names(), name_col=name_col,
                                          source=label), context=label)


def test_the_classic_player_pool_is_self_consistent():
    from scripts.common.fpl_draft_api import pull_fpl_player_stats
    stats = skip_if_unreachable(pull_fpl_player_stats, "FPL player stats")
    if stats is None or stats.empty:
        pytest.skip("no FPL player stats")
    _check(stats, "pull_fpl_player_stats")


def test_the_blended_pool_is_self_consistent():
    import config
    from scripts.common.player_pool import blended_pool
    pool = skip_if_unreachable(
        lambda: blended_pool(config.CURRENT_GAMEWEEK, config.ROTOWIRE_URL),
        "blended pool")
    if pool is None or pool.empty:
        pytest.skip("no blended pool")
    _check(pool, "blended_pool")


def test_the_draft_roster_frame_is_self_consistent():
    """The reported failure: this frame resolved its ids through the Draft
    bootstrap and then every consumer joined it against Classic-keyed data."""
    import config
    from scripts.common.fpl_draft_api import get_team_composition_for_gameweek
    league_id = getattr(config, "FPL_DRAFT_LEAGUE_ID", None)
    team_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    if not league_id or not team_id:
        pytest.skip("no Draft league configured")
    comp = skip_if_unreachable(
        lambda: get_team_composition_for_gameweek(
            league_id, team_id, config.CURRENT_GAMEWEEK),
        "Draft roster")
    if comp is None or comp.empty:
        pytest.skip("no roster for this gameweek")
    _check(comp, "get_team_composition_for_gameweek")


def test_the_check_would_notice_an_untranslated_roster():
    """The negative control.

    A check that never fires is indistinguishable from one that cannot. This
    puts the same roster back into the Draft space and asserts it is caught --
    so if the id spaces are ever reconciled, or the check stops working, this
    is what says so.
    """
    import config
    import pandas as pd
    from scripts.common.data_validation import check_element_id_names
    from scripts.common.element_ids import draft_to_classic_ids
    from scripts.common.fpl_draft_api import get_team_composition_for_gameweek

    league_id = getattr(config, "FPL_DRAFT_LEAGUE_ID", None)
    team_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    if not league_id or not team_id:
        pytest.skip("no Draft league configured")
    comp = skip_if_unreachable(
        lambda: get_team_composition_for_gameweek(
            league_id, team_id, config.CURRENT_GAMEWEEK),
        "Draft roster")
    if comp is None or comp.empty:
        pytest.skip("no roster for this gameweek")

    inverse = {v: k for k, v in draft_to_classic_ids().items()}
    if not inverse:
        pytest.skip("no id translation available")
    rolled_back = comp.copy()
    rolled_back["Player_ID"] = [
        inverse.get(int(p)) if pd.notna(p) else None
        for p in rolled_back["Player_ID"]
    ]
    issues = check_element_id_names(rolled_back, _names(),
                                    source="untranslated roster")
    if not any(i.severity == "error" for i in issues):
        # Possible and not a defect: this roster may hold none of the ~64
        # players the two APIs disagree about. Say which it was.
        affected = sum(1 for p in comp["Player_ID"].dropna()
                       if inverse.get(int(p)) != int(p))
        assert affected == 0, (
            "%d roster player(s) have a different Draft id, so rolling them "
            "back should have been caught and was not" % affected
        )
        pytest.skip("this roster holds no player the two APIs number differently")


def test_the_availability_frame_is_self_consistent():
    """The reported failure on this frame: it was built from the *Draft*
    bootstrap and then joined to Classic-keyed data by three pages.

    Every field it reads is published identically by both APIs, so nothing
    looked wrong -- only `id` differs. Measured on 2026-10-30, 10 of the
    affected players carried news, so the Availability and Injuries pages were
    attributing ten injuries to the wrong player.
    """
    from scripts.fpl.injuries import get_fpl_availability_df
    pool = skip_if_unreachable(get_fpl_availability_df, "FPL availability")
    if pool is None or pool.empty:
        pytest.skip("no availability frame")
    _check(pool, "get_fpl_availability_df")


def test_the_draft_actual_lineup_is_self_consistent():
    """This frame's id reaches `simulate_auto_subs` beside Classic live stats
    and a Classic `element_to_team`, so a cross-space id there is an auto-sub
    decision made on another player's minutes."""
    import config
    from scripts.common.fpl_draft_api import get_team_actual_lineup
    team_id = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    if not team_id:
        pytest.skip("no Draft team configured")
    lineup = skip_if_unreachable(
        lambda: get_team_actual_lineup(team_id, config.CURRENT_GAMEWEEK),
        "Draft actual lineup")
    if lineup is None or lineup.empty:
        pytest.skip("no actual lineup for this gameweek")
    _check(lineup, "get_team_actual_lineup")
