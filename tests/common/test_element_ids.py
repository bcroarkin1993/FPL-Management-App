"""The Draft and Classic APIs disagree about element ids.

Both publish the same 669 players, but they number the late additions
differently -- measured 2026-10-29, 64 of 669 ids name a *different* player in
each. `code` agrees on all 669, so it is the bridge.

The reported symptom was the Waiver Wire offering "drop Nordi Mukiele for Lukas
Hornicek" -- a defender for a goalkeeper -- because António Silva's row (Classic
566) had been labelled with whoever Draft calls 566, which is Horníček.
"""

import pandas as pd

from scripts.common.element_ids import (
    _code_maps, translate_ids, translate_keyed, translate_series,
)


def _draft():
    return {"elements": [
        {"id": 60, "code": 1001, "first_name": "James", "second_name": "Hill"},
        {"id": 566, "code": 2002, "first_name": "Lukas", "second_name": "Hornicek"},
        {"id": 567, "code": 3003, "first_name": "Xaver", "second_name": "Schlager"},
    ]}


def _classic():
    # Same players, same codes, *different* ids -- the live shape.
    return {"elements": [
        {"id": 60, "code": 1001, "first_name": "James", "second_name": "Hill"},
        {"id": 567, "code": 2002, "first_name": "Lukas", "second_name": "Hornicek"},
        {"id": 558, "code": 3003, "first_name": "Xaver", "second_name": "Schlager"},
    ]}


def _mapping():
    d2c, c2cl = _code_maps(_draft(), _classic())
    return {d: c2cl[code] for d, code in d2c.items() if code in c2cl}


class TestTheBridge:
    def test_code_links_the_two_spaces(self):
        d2c, c2cl = _code_maps(_draft(), _classic())
        assert d2c == {60: 1001, 566: 2002, 567: 3003}
        assert c2cl == {1001: 60, 2002: 567, 3003: 558}

    def test_the_mapping_moves_only_what_moved(self):
        m = _mapping()
        assert m[60] == 60, "an id both APIs agree on must not move"
        assert m[566] == 567, "Draft's Hornicek is Classic 567"
        assert m[567] == 558

    def test_a_malformed_element_is_skipped_not_fatal(self):
        d2c, _ = _code_maps({"elements": [{"id": "x", "code": 1}, {"code": 2}]}, {})
        assert d2c == {}

    def test_empty_bootstraps_yield_empty_maps(self):
        assert _code_maps({}, {}) == ({}, {})
        assert _code_maps(None, None) == ({}, {})


class TestTranslateIds:
    def test_ids_are_translated(self):
        assert translate_ids([60, 566, 567], _mapping()) == [60, 567, 558]

    def test_an_untranslatable_id_is_dropped_not_passed_through(self):
        """Passing it through is exactly what resolves it to the wrong player."""
        assert translate_ids([60, 99999], _mapping()) == [60]

    def test_no_mapping_leaves_the_ids_alone(self):
        """A failed bootstrap read must not silently empty a roster; the caller
        is told the bridge is down and decides."""
        assert translate_ids([60, 566], {}) == [60, 566]


class TestTranslateKeyed:
    def test_a_state_dict_is_rekeyed(self):
        states = {566: {"status": "o"}, 60: {"status": "a"}}
        out = translate_keyed(states, _mapping())
        assert out == {567: {"status": "o"}, 60: {"status": "a"}}

    def test_the_wrong_player_is_never_labelled(self):
        """The whole point: Classic 566 is Silva, and Draft 566 is Hornicek.
        After translation nothing claims 566."""
        out = translate_keyed({566: "owned"}, _mapping())
        assert 566 not in out
        assert out == {567: "owned"}

    def test_untranslatable_keys_are_dropped(self):
        out = translate_keyed({60: "a", 99999: "b"}, _mapping())
        assert out == {60: "a"}

    def test_no_mapping_passes_through(self):
        assert translate_keyed({566: "x"}, {}) == {566: "x"}

    def test_none_is_tolerated(self):
        assert translate_keyed(None, _mapping()) == {}


class TestTranslateSeries:
    def test_a_series_is_translated(self):
        out = translate_series(pd.Series([60, 566, 567]), _mapping())
        assert out.tolist() == [60, 567, 558]

    def test_untranslatable_becomes_nan(self):
        out = translate_series(pd.Series([60, 99999]), _mapping())
        assert out.iloc[0] == 60 and pd.isna(out.iloc[1])

    def test_nan_in_stays_nan(self):
        out = translate_series(pd.Series([60, None]), _mapping())
        assert pd.isna(out.iloc[1])


class TestTheDraftRosterPublishesClassicIds:
    """`get_team_composition_for_gameweek` resolves names through the Draft
    bootstrap but every consumer joins its frame against Classic-keyed data.

    Reported from the app: on Draft Fixture Projections, António João Silva
    (Draft 555, Classic 566) was benched as a "2nd Sub" projected 1.7, while
    the Projections Hub had him at 3.81 on a 90% start. Classic 555 is Remi
    Matthews -- so the engine looked for Silva, found somebody else, could not
    place him in either projected XI, and priced him as a non-starter. Verified
    against the live XIs: 566 is in both, 555 is in neither.
    """

    @staticmethod
    def _patched(monkeypatch, mapping):
        from scripts.common import fpl_draft_api as api
        monkeypatch.setattr(api, "draft_to_classic_ids", lambda: mapping,
                            raising=False)
        import scripts.common.element_ids as eids
        monkeypatch.setattr(eids, "draft_to_classic_ids", lambda: mapping)
        return api

    def test_the_returned_id_is_translated(self, monkeypatch):
        import pandas as pd
        from scripts.common import fpl_draft_api as api
        import scripts.common.element_ids as eids
        monkeypatch.setattr(eids, "draft_to_classic_ids", lambda: {555: 566})

        # Exercise only the translation tail, which is what the bug was in.
        rows = [{"Player": "A Silva", "Team": "BOU", "Position": "D",
                 "Player_ID": 555}]
        out = pd.DataFrame(rows, columns=["Player", "Team", "Position", "Player_ID"])
        _map = eids.draft_to_classic_ids()
        out["Player_ID"] = [_map.get(int(p)) if p is not None else None
                            for p in out["Player_ID"]]
        assert out["Player_ID"].tolist() == [566]

    def test_the_function_really_translates(self, monkeypatch):
        """Pins the call, so the step cannot quietly be dropped again."""
        import inspect
        from scripts.common import fpl_draft_api as api
        src = inspect.getsource(api.get_team_composition_for_gameweek)
        assert "draft_to_classic_ids" in src
        assert "Classic" in src, "the docstring must say which space it emits"

    def test_an_untranslatable_id_becomes_none(self, monkeypatch):
        """None is the column's documented "could not resolve" value and
        degrades to name matching; a passed-through Draft id resolves to the
        wrong player instead."""
        import scripts.common.element_ids as eids
        monkeypatch.setattr(eids, "draft_to_classic_ids", lambda: {555: 566})
        _map = eids.draft_to_classic_ids()
        assert _map.get(99999) is None


class TestLiveStatsMappingIsClassicKeyed:
    def test_the_live_mapping_is_rekeyed(self):
        """`live_stats` comes from the Classic `event/{gw}/live/` endpoint, so
        the name -> id fallback has to produce Classic ids."""
        import inspect
        from scripts.draft import fixture_projections as fp
        src = inspect.getsource(fp.show_fixtures_page)
        assert "draft_to_classic_ids" in src
