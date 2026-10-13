"""Live plausibility for FFP's multi-gameweek horizon.

Contract, per ``tests/live/conftest.py``: **unreachable → SKIP, reachable but
implausible → FAIL.**

What this guards is a failure the single-gameweek fixtures cannot see. FFP
publishes six gameweeks per player and the app folds five of them into
*relative-offset* columns; the horizon grid un-folds them back into absolute
gameweeks. If that correspondence slips, one week's projections render under
another week's heading and every number on screen stays perfectly plausible --
the same class of silent error as the sheet being a gameweek behind.
"""

import config
import pandas as pd
import pytest

from scripts.common import ffp_feed, projection_sources
from scripts.common.data_validation import check_ffp_horizon, raise_on_error
from tests.live.conftest import skip_if_unreachable


@pytest.fixture(scope="module")
def horizon():
    result = skip_if_unreachable(
        lambda: projection_sources.fetch_ffp_horizon(start_gw=config.CURRENT_GAMEWEEK),
        "FFP horizon",
    )
    df = result[0]
    if df is None or df.empty:
        pytest.skip("FFP published no horizon we could read (provenance=%s)" % result[4])
    return result


def test_the_horizon_is_plausible(horizon):
    df, start_gw, _window_gw, _updated, _prov, per_gw, _note = horizon
    raise_on_error(check_ffp_horizon(df, start_gw, per_gw))


def test_it_covers_a_contiguous_run_of_gameweeks(horizon):
    df, start_gw = horizon[0], horizon[1]
    gws = sorted(int(g) for g in df["FFP_GW"].dropna().unique())
    assert gws[0] == int(start_gw), (
        "the horizon starts at GW%d but GW%d was asked for; the grid labels its "
        "first column with the gameweek being scored" % (gws[0], start_gw)
    )
    assert gws == list(range(gws[0], gws[-1] + 1)), (
        "gameweeks %s are not contiguous -- a gap shifts every later column one "
        "week left of its own heading" % gws
    )
    assert len(gws) >= 2, "a horizon of one gameweek is not a horizon"


def test_every_gameweek_prices_a_full_board(horizon):
    df = horizon[0]
    counts = df.groupby("FFP_GW")["Predicted"].count()
    assert counts.min() >= 150, (
        "GW%d prices only %d players; a short week means the payload parser "
        "captured a fragment" % (counts.idxmin(), counts.min())
    )
    # Every week in one window describes the same squad, so the weeks should
    # not differ wildly in size. A big gap means one week was assembled
    # differently from the rest.
    assert counts.max() - counts.min() <= 0.25 * counts.max(), (
        "per-gameweek row counts vary too much: %s" % counts.to_dict()
    )


def test_players_resolve_to_fpl_element_ids(horizon):
    """FFP joins on an integer, which is the whole reason to use its payload."""
    df = horizon[0]
    if "Player_ID" not in df.columns:
        pytest.skip("no bootstrap code map was available for this fetch")
    resolved = df["Player_ID"].notna().mean()
    assert resolved > 0.95, (
        "only %.0f%% of horizon rows resolved to an element id; the grid joins "
        "on it, so unresolved rows simply vanish" % (100 * resolved)
    )


def test_the_basis_relation_holds_in_every_gameweek(horizon):
    """``Predicted == StartingPredicted x Start/100``, week by week."""
    df = horizon[0]
    for gw, block in df.groupby("FFP_GW"):
        pred = pd.to_numeric(block["Predicted"], errors="coerce")
        cond = pd.to_numeric(block["StartingPredicted"], errors="coerce")
        start = pd.to_numeric(block["Start"], errors="coerce") / 100.0
        ok = pred.notna() & cond.notna() & start.notna() & (cond > 0)
        assert ok.sum() > 0, "GW%s carries no usable projection at all" % gw
        err = (pred[ok] - cond[ok] * start[ok]).abs().mean()
        assert err < 0.05, (
            "GW%s: Predicted is not StartingPredicted x Start%% (MAE %.3f). The "
            "two point columns have probably been mapped across by name rather "
            "than by basis." % (gw, err)
        )
        assert (pred[ok] <= cond[ok] + 0.05).mean() > 0.99, (
            "GW%s: the unconditional projection exceeds the conditional one" % gw
        )


def test_the_forecasts_actually_vary_by_gameweek(horizon):
    """A horizon where every week is identical is not telling you anything.

    This is the check that would catch the grid being filled with one week's
    numbers six times over -- which is both plausible-looking and useless, and
    is the exact shape of the bug the offset/absolute conversion could produce.
    """
    df = horizon[0]
    if "Player_ID" not in df.columns:
        pytest.skip("no element ids to pivot on")
    wide = df.dropna(subset=["Player_ID"]).pivot_table(
        index="Player_ID", columns="FFP_GW", values="Predicted")
    if wide.shape[1] < 2:
        pytest.skip("only one gameweek present")
    spread = (wide.max(axis=1) - wide.min(axis=1)).dropna()
    assert spread.median() > 0.10, (
        "the median player's projection varies by only %.3f points across the "
        "whole window; the weeks are probably duplicates of one another"
        % spread.median()
    )


def test_fixtures_and_difficulty_come_with_the_forecasts(horizon):
    """The fixtures view reads these, which saves six live fixture fetches."""
    df = horizon[0]
    assert "Fixture" in df.columns and "Difficulty" in df.columns
    fixtures = df["Fixture"].fillna("").astype(str)
    assert (fixtures.str.contains(r"\(")).mean() > 0.9, (
        "fixtures are not in the 'OPP (H)' shape the grid renders"
    )
    fdr = pd.to_numeric(df["Difficulty"], errors="coerce").dropna()
    assert not fdr.empty, "no difficulty published; the fixtures view loses its colour"
    assert fdr.between(1, 5).all(), (
        "difficulty outside 1-5 (min %s, max %s); the grid's colour scale is "
        "fixed to that range" % (fdr.min(), fdr.max())
    )


def test_the_long_and_sheet_schemas_agree_on_the_live_payload():
    """The offset/absolute correspondence, checked against the real payload.

    The offline test pins this on a fixture. This pins it on what FFP is
    actually serving, which is where a change to their payload would show up.
    """
    rows, window_gw, _updated = skip_if_unreachable(
        ffp_feed.fetch_points_predictor, "FFP points predictor")
    if not rows:
        pytest.skip("FFP's site payload carried no rows")

    sheet = ffp_feed.to_sheet_schema(rows, window_gw)
    long = ffp_feed.to_long_schema(rows)
    if sheet.empty or long.empty:
        pytest.skip("payload did not project onto either schema")

    sheet_rows = sheet.set_index(sheet["Player_Code"].astype("int64"))
    checked = 0
    for offset in range(1, ffp_feed._HORIZON + 1):
        target = window_gw + offset
        block = long[long["FFP_GW"] == target]
        if block.empty:
            continue
        block = block.set_index(block["Player_Code"].astype("int64"))
        label = "GW%d" % (offset + 1)
        common = sheet_rows.index.intersection(block.index)
        a = pd.to_numeric(sheet_rows.loc[common, label + "s"], errors="coerce")
        b = pd.to_numeric(block.loc[common, "Predicted"], errors="coerce")
        both = a.notna() & b.notna()
        assert both.sum() > 0, "no overlap for %s / GW%d" % (label, target)
        assert (a[both] - b[both]).abs().max() < 1e-6, (
            "the sheet's %s and the long frame's GW%d disagree: the relative "
            "offset no longer resolves to the absolute gameweek the horizon "
            "grid labels its column with" % (label, target)
        )
        checked += 1
    assert checked >= 1, "no forward gameweek was in both schemas to compare"
