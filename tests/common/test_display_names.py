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
    resolve_display,
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


class TestResolveDisplay:
    """The row-level resolver, for frames with no name *column*.

    The Draft lineup cards are indexed by player name, so
    `attach_display_names` -- which needs a column -- does not fit. This is the
    same resolution for one row.
    """

    def test_it_resolves_by_code_then_id(self, dmap):
        assert resolve_display({"Player_Code": 141746}, dmap) == "Bruno Fernandes"
        assert resolve_display({"Player_ID": 17}, dmap) == "Bruno Fernandes"

    def test_the_code_wins(self, dmap):
        """An element id can have crossed between the two FPL APIs; a code
        cannot."""
        row = {"Player_Code": 141746, "Player_ID": 201}   # id 201 is Alex Palmer
        assert resolve_display(row, dmap) == "Bruno Fernandes"

    def test_it_falls_back_to_the_rows_own_name(self, dmap):
        assert resolve_display({"Player": "Bruno Borges Fernandes"}, dmap) == "Bruno Fernandes"

    def test_an_unknown_name_is_passed_through(self, dmap):
        assert resolve_display({"Player": "Rotowire Only Guy"}, dmap) == "Rotowire Only Guy"

    def test_an_ambiguous_name_is_passed_through(self, dmap):
        """The Palmer rule again: resolve to nothing rather than to one of two."""
        assert resolve_display({"Player": "Palmer"}, dmap) == "Palmer"

    def test_a_nan_id_is_skipped_not_crashed(self, dmap):
        row = {"Player_Code": float("nan"), "Player_ID": 17}
        assert resolve_display(row, dmap) == "Bruno Fernandes"

    def test_nothing_usable_gives_the_default(self, dmap):
        assert resolve_display({}, dmap, default="—") == "—"

    def test_no_map_leaves_the_rows_name(self):
        row = {"Player": "Bruno Borges Fernandes", "Player_Code": 141746}
        assert resolve_display(row, DisplayMap({}, {}, {})) == "Bruno Borges Fernandes"


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


# ---------------------------------------------------------------------------
# Guards: the two shapes that actually duplicated
# ---------------------------------------------------------------------------

import ast
import pathlib

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

#: `display_names.py` owns the fallback; nothing else may re-derive it.
FALLBACK_ALLOWED = {"display_names.py"}


def _display_name_fallbacks(tree):
    """Line numbers of `<expr>.get("Display_Name") or <expr>`.

    Four copies of this had drifted apart on their default ("N/A", "Unknown",
    "", a KeyError) and on whether they tolerated a non-string -- a `Display_Name`
    of `NaN` is truthy enough to slip through some of them and not others.
    """
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
            continue
        first = node.values[0]
        if not isinstance(first, ast.Call):
            continue
        fn = first.func
        if getattr(fn, "attr", None) != "get" or not first.args:
            continue
        arg = first.args[0]
        if isinstance(arg, ast.Constant) and arg.value == "Display_Name":
            out.append(node.lineno)
    return out


_SPLITTERS = ("split", "partition", "rpartition")


def _split_bound_names(scope):
    """Local names bound from a `.split()`/`.partition()` call in ``scope``.

    Needed because the real bug was two statements, not one:

        first, _, rest = player.partition(" ")
        return to_display_name(first, rest, row.get("Web_Name"))

    A detector that only looked inside the `to_display_name` call itself saw
    nothing -- which is what the self-check below caught.
    """
    names = set()
    for node in ast.walk(scope):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if value is None:
            continue
        calls = [n for n in ast.walk(value)
                 if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", None) in _SPLITTERS]
        if not calls:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
    return names


def _lossy_rederivations(tree):
    """Line numbers of `to_display_name(...)` fed from a split of one string.

    The bug: `first, _, rest = player.partition(" ")` to fake the bootstrap's
    name fields. It cannot recover "Igor Thiago", whose surname sits in
    `first_name`, and it silently produced a *plausible* wrong name.

    Scoped per function, plus module level, so a name split in one function
    cannot flag an unrelated call in another.
    """
    scopes = [tree] + [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out = []
    for scope in scopes:
        tainted = _split_bound_names(scope)
        for node in ast.walk(scope):
            if not isinstance(node, ast.Call):
                continue
            if (getattr(node.func, "id", None)
                    or getattr(node.func, "attr", None)) != "to_display_name":
                continue
            for arg in node.args:
                hit = False
                for inner in ast.walk(arg):
                    if isinstance(inner, ast.Call) and \
                            getattr(inner.func, "attr", None) in _SPLITTERS:
                        hit = True
                    elif isinstance(inner, ast.Name) and inner.id in tainted:
                        hit = True
                if hit:
                    out.append(node.lineno)
                    break
    return sorted(set(out))


def test_nothing_re_derives_the_display_name_fallback():
    offenders = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        if path.name in FALLBACK_ALLOWED:
            continue
        for lineno in _display_name_fallbacks(ast.parse(path.read_text())):
            offenders.append("%s:%d" % (path.relative_to(SCRIPTS), lineno))
    assert not offenders, (
        "a re-derived Display_Name fallback at %s -- use "
        "`display_names.display_of(row)`, which owns the "
        "Display_Name -> Player -> Web_Name order and the NaN handling"
        % offenders)


def test_no_display_name_is_built_by_splitting_a_legal_name():
    offenders = []
    for path in sorted(SCRIPTS.rglob("*.py")):
        for lineno in _lossy_rederivations(ast.parse(path.read_text())):
            offenders.append("%s:%d" % (path.relative_to(SCRIPTS), lineno))
    assert not offenders, (
        "to_display_name fed from a split string at %s -- it needs the "
        "bootstrap's real first_name/second_name/web_name. Splitting a legal "
        "name cannot recover 'Igor Thiago', whose surname is in first_name, "
        "and the wrong answer it gives is plausible. Resolve through "
        "`display_names.attach_display_names` instead." % offenders)


def test_the_guards_can_actually_fail(tmp_path):
    """A guard that cannot fail is indistinguishable from one that does not run."""
    bad = tmp_path / "offender.py"
    bad.write_text(
        'name = row.get("Display_Name") or row["Player"]\n'
        'first, _, rest = player.partition(" ")\n'
        'shown = to_display_name(first, rest, row.get("Web_Name"))\n')
    tree = ast.parse(bad.read_text())
    assert _display_name_fallbacks(tree) == [1]
    assert _lossy_rederivations(tree) == [3]
