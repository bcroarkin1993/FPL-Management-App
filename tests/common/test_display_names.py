"""One place answers "what do we call this player?".

Before this there were three bootstrap->display maps, four copies of the
"Display_Name, else Player" fallback, and two re-derivations that split a legal
name on whitespace to recover first/second. The last pair is the lossy one:
"Igor Thiago Nascimento Rodrigues" splits to first="Igor", and `to_display_name`
cannot then recover "Igor Thiago" the way it can from the real fields.

The two rules that must survive any future consolidation are the Palmer rule (a
name-only key resolves to one player or to nothing) and that `Player` is never
overwritten, because every cross-source merge keys on it.
"""

import pandas as pd
import pytest

from scripts.common.display_names import (
    DisplayMap,
    attach_display_names,
    build_display_map,
    display_of,
    display_series,
)

BOOTSTRAP = {
    "elements": [
        # The case the whole module exists for: a legal name nobody says.
        {"id": 17, "code": 141746, "first_name": "Bruno",
         "second_name": "Borges Fernandes", "web_name": "B.Fernandes"},
        # A mononym -- `to_display_name` must not double it.
        {"id": 5, "code": 100005, "first_name": "Gabriel",
         "second_name": "dos Santos Magalhaes", "web_name": "Gabriel"},
        # The surname sits in first_name.
        {"id": 106, "code": 482973, "first_name": "Igor Thiago",
         "second_name": "Nascimento Rodrigues", "web_name": "Igor Thiago"},
        # Two Palmers, so the bare surname is ambiguous.
        {"id": 200, "code": 200200, "first_name": "Cole",
         "second_name": "Palmer", "web_name": "Palmer"},
        {"id": 201, "code": 200201, "first_name": "Alex",
         "second_name": "Palmer", "web_name": "Palmer"},
        # Diacritics, which must survive to the rendered name.
        {"id": 300, "code": 300300, "first_name": "Martin",
         "second_name": "Ødegaard", "web_name": "Ødegaard"},
    ]
}


@pytest.fixture(scope="module")
def dmap():
    return build_display_map(BOOTSTRAP)


class TestBuildDisplayMap:
    def test_it_keys_on_code_and_id_and_name(self, dmap):
        assert dmap.by_code[141746] == "Bruno Fernandes"
        assert dmap.by_id[17] == "Bruno Fernandes"
        assert dmap.by_name["bruno borges fernandes"] == "Bruno Fernandes"

    def test_an_ambiguous_name_gets_no_entry(self, dmap):
        """The Palmer rule. A shared key must resolve to nothing, never to
        whichever player the bootstrap happened to list last."""
        assert "palmer" not in dmap.by_name
        # The unambiguous keys for each still exist.
        assert dmap.by_name["cole palmer"] == "Cole Palmer"
        assert dmap.by_name["alex palmer"] == "Alex Palmer"

    def test_a_malformed_element_is_skipped_not_fatal(self):
        m = build_display_map({"elements": [
            {"id": "x", "code": None},
            {"first_name": "No", "second_name": "Id", "web_name": "Id"},
        ]})
        assert m.by_code == {} and m.by_id == {}

    def test_an_empty_bootstrap_is_not_ok(self):
        assert not build_display_map({}).ok
        assert not build_display_map(None).ok


class TestAttachDisplayNames:
    def test_the_code_wins_over_the_frames_own_name(self, dmap):
        """The point of keying on the code: the source's name can be anything
        -- Rotowire's, an abbreviation, a stale spelling -- and the rendered
        name is still right."""
        df = pd.DataFrame([{"Player": "whatever the source said",
                            "Player_Code": 141746}])
        out = attach_display_names(df, dmap)
        assert out.loc[0, "Display_Name"] == "Bruno Fernandes"

    def test_it_falls_through_to_the_element_id(self, dmap):
        df = pd.DataFrame([{"Player": "x", "Player_ID": 17}])
        assert attach_display_names(df, dmap).loc[0, "Display_Name"] == "Bruno Fernandes"

    def test_the_code_is_preferred_over_the_element_id(self, dmap):
        """A frame can carry an id that crossed between the two FPL APIs; the
        code cannot. Where both are present the code must win."""
        df = pd.DataFrame([{"Player": "x", "Player_Code": 141746, "Player_ID": 201}])
        assert attach_display_names(df, dmap).loc[0, "Display_Name"] == "Bruno Fernandes"

    def test_it_resolves_by_name_when_there_is_no_id(self, dmap):
        df = pd.DataFrame([{"Player": "Bruno Borges Fernandes"}])
        assert attach_display_names(df, dmap).loc[0, "Display_Name"] == "Bruno Fernandes"

    def test_an_ambiguous_name_keeps_the_source_name(self, dmap):
        df = pd.DataFrame([{"Player": "Palmer"}])
        assert attach_display_names(df, dmap).loc[0, "Display_Name"] == "Palmer"

    def test_an_unknown_player_keeps_the_source_name(self, dmap):
        df = pd.DataFrame([{"Player": "Rotowire Only Guy"}])
        assert attach_display_names(df, dmap).loc[0, "Display_Name"] == "Rotowire Only Guy"

    def test_player_is_never_overwritten(self, dmap):
        """Every cross-source merge keys on `Player`. A match is for numbers,
        never for names."""
        df = pd.DataFrame([{"Player": "Bruno Borges Fernandes", "Player_Code": 141746}])
        out = attach_display_names(df, dmap)
        assert out.loc[0, "Player"] == "Bruno Borges Fernandes"

    def test_no_map_leaves_the_source_name(self):
        df = pd.DataFrame([{"Player": "Bruno Borges Fernandes"}])
        out = attach_display_names(df, DisplayMap({}, {}, {}))
        assert out.loc[0, "Display_Name"] == "Bruno Borges Fernandes"

    def test_an_empty_or_nameless_frame_is_returned_unchanged(self, dmap):
        assert attach_display_names(pd.DataFrame(), dmap).empty
        nameless = pd.DataFrame([{"Team": "ARS"}])
        assert "Display_Name" not in attach_display_names(nameless, dmap).columns

    def test_the_surname_in_first_name_case(self, dmap):
        """`to_display_name`'s hard case, and the one the whitespace-splitting
        re-derivations got wrong: the surname is in `first_name`."""
        df = pd.DataFrame([{"Player": "x", "Player_Code": 482973}])
        assert attach_display_names(df, dmap).loc[0, "Display_Name"] == "Igor Thiago"


class TestDisplayOf:
    def test_it_prefers_display_name(self):
        assert display_of({"Display_Name": "Sávio", "Player": "Sávio Moreira"}) == "Sávio"

    def test_it_falls_through_an_absent_or_nan_display_name(self):
        assert display_of({"Player": "Martin Ødegaard"}) == "Martin Ødegaard"
        assert display_of({"Display_Name": float("nan"),
                           "Player": "Martin Ødegaard"}) == "Martin Ødegaard"
        assert display_of({"Display_Name": "  ", "Player": "Martin Ødegaard"}) == "Martin Ødegaard"

    def test_web_name_is_the_last_resort_not_the_second(self):
        """An abbreviation must never be what we render while a real name is
        available, but it beats rendering nothing."""
        assert display_of({"Player": "Bruno Borges Fernandes",
                           "Web_Name": "B.Fernandes"}) == "Bruno Borges Fernandes"
        assert display_of({"Web_Name": "B.Fernandes"}) == "B.Fernandes"

    def test_diacritics_survive(self):
        """The Waiver Wire stripped them, for no recorded reason, so one page
        disagreed with every other about what a player is called."""
        assert display_of({"Display_Name": "Lukáš Horníček"}) == "Lukáš Horníček"

    def test_nothing_renderable_gives_the_default(self):
        assert display_of({}, default="—") == "—"
        assert display_of({"Player": None}, default="—") == "—"


class TestDisplaySeries:
    def test_it_renders_a_column(self):
        df = pd.DataFrame([{"Display_Name": "Sávio", "Player": "Sávio Moreira"},
                           {"Player": "Solid Mid"}])
        assert display_series(df).tolist() == ["Sávio", "Solid Mid"]

    def test_an_empty_frame_gives_a_series_not_a_frame(self):
        """`DataFrame.apply` over an empty frame returns a DataFrame, which
        cannot be assigned to a column."""
        out = display_series(pd.DataFrame(columns=["Player"]))
        assert isinstance(out, pd.Series)
        assert out.empty
