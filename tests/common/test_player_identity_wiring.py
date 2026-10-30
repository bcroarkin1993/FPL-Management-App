"""Identity is resolved once, at the boundary, and carried from there.

**The invariant.** `code` is the app's cross-source join key: it is the only id
the Draft and Classic APIs agree on (669 of 669, against 605 for the element
id), and it is range-disjoint from the element id, so a mixup resolves to
nothing rather than to a real but wrong player. `Player_ID` remains a
Classic-space scalar for talking to the FPL API -- never a join key between two
frames of different provenance.

**Why these are guard tests and not plausibility checks.** Every failure in this
family is invisible in the numbers. A name match that silently misses leaves a
NaN that some default fills; an id that crosses API spaces resolves to another
player whose every value is plausible; a schema coercion that drops an id sends
a whole page back to name matching with nothing on screen to show it. Measured
examples, all of them real: two owned players offered as waiver targets because
"Oli McBurnie" does not reach "Oliver McBurnie"; ten injuries attributed to the
wrong player; four separate surfaces shipping a cross-space id.

So what is asserted here is *where identity comes from* and *that it survives*,
which is the part a reviewer can actually check.

Modelled on `test_element_id_wiring.py` and `test_matchup_wiring.py`: an
explicit, commented allowlist, small `ast.walk` collectors, and a self-check so
the guard cannot quietly stop working.
"""

import ast
import pathlib

import pandas as pd
import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _modules():
    return sorted(SCRIPTS.rglob("*.py"))


def _tree(path):
    return ast.parse(path.read_text())


# ---------------------------------------------------------------------------
# 1. Fuzzy string scoring has exactly one home
# ---------------------------------------------------------------------------

#: `fuzzywuzzy` may be imported here and nowhere else.
#:
#: `player_matching.merge_fpl_players_and_projections` is the last fuzzy
#: matcher in the app, and it is on its way out rather than endorsed: Rotowire
#: now carries an element id from the fetch boundary, so the merge has an exact
#: key available and the name scoring is the fallback.
#:
#: The pattern this forbids is specific and has bitten repeatedly. `WRatio`
#: falls back to partial-token ratios once two strings differ in length, so one
#: shared token carries it -- "Maxim De Cuyper" scores 86 against "João Pedro
#: Junqueira de Jesus" on the word *de*. That renamed 87 players and left 66
#: sharing a name. `ReferenceMatcher` exists so nothing needs to re-derive
#: this.
FUZZ_ALLOWED = {"player_matching.py"}


def test_fuzzy_scoring_is_confined_to_one_module():
    offenders = []
    for path in _modules():
        for node in ast.walk(_tree(path)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] in ("fuzzywuzzy", "rapidfuzz") for n in names):
                if path.name not in FUZZ_ALLOWED:
                    offenders.append("%s:%d" % (path.relative_to(SCRIPTS), node.lineno))
    assert not offenders, (
        "fuzzy string scoring outside %s: %s -- resolve names with "
        "ReferenceMatcher, whose every loose tier is scoped by team and "
        "position, and carry the resulting id rather than re-deriving it"
        % (sorted(FUZZ_ALLOWED), offenders))


# ---------------------------------------------------------------------------
# 2. Name resolution happens in a known, small set of places
# ---------------------------------------------------------------------------

#: Modules permitted to build a `ReferenceMatcher`. Each is a *boundary*: a
#: source that publishes names and no id, resolved once.
#:
#:   analytics.py          `_claim_reference_rows`, the shared claim protocol
#:                         every projection merge goes through
#:   projection_engine.py  `resolve_player_ids`, used by the engine and by
#:                         `scraping._attach_projection_ids` at the Rotowire
#:                         fetch boundary
#:   rotowire_lineups.py   `resolve_lineup_name`, shared with Sportsgambler
#:   projected_lineups.py  the lineup cards' club-scoped player index
#:   transfer_risk_app.py  the Rotowire Top-400 board against bootstrap news
#:
#: A new entry means a new place where a name becomes a player, which is the
#: single most frequent source of silent bugs in this app. Adding one should be
#: a deliberate, reviewed act -- which is what this list makes it.
MATCHER_ALLOWED = {
    "analytics.py",
    "projection_engine.py",
    "rotowire_lineups.py",
    "projected_lineups.py",
    "transfer_risk_app.py",
}


def _matcher_constructions(tree):
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (getattr(fn, "id", None) or getattr(fn, "attr", None)) == "ReferenceMatcher":
            out.append(node.lineno)
    return out


def test_only_the_boundary_modules_resolve_names():
    offenders = []
    for path in _modules():
        if path.name in MATCHER_ALLOWED:
            continue
        for lineno in _matcher_constructions(_tree(path)):
            offenders.append("%s:%d" % (path.relative_to(SCRIPTS), lineno))
    assert not offenders, (
        "a name resolver outside the boundary modules: %s -- resolve once at "
        "the fetch boundary and join on the id downstream" % offenders)


def test_the_guard_can_actually_fail(tmp_path):
    """A guard that cannot fail is indistinguishable from one that does not run."""
    bad = tmp_path / "offender.py"
    bad.write_text("from fuzzywuzzy import fuzz\nm = ReferenceMatcher(pool)\n")
    tree = ast.parse(bad.read_text())
    assert _matcher_constructions(tree) == [2]
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports and imports[0].module == "fuzzywuzzy"


# ---------------------------------------------------------------------------
# 3. Identity survives the places that used to drop it
# ---------------------------------------------------------------------------

class TestIdentitySurvives:
    """Each of these dropped the id at some point, and each was silent."""

    PROJ = pd.DataFrame([
        {"Player": "Bruno Fernandes", "Team": "MUN", "Position": "M",
         "Points": 6.0, "Pos Rank": 1, "Matchup": "MUN v ARS"},
    ])

    FPL = pd.DataFrame([
        {"Player": "Bruno Borges Fernandes", "Team": "MUN", "Position": "M",
         "Player_ID": 17, "Player_Code": 141746},
    ])

    def test_the_projection_merge_keeps_it_without_being_asked(self):
        """`carry_cols=['Player_ID']` was the workaround, and four of six
        callsites used it while two forgot -- the two that run pre-deadline."""
        from scripts.common.player_matching import merge_fpl_players_and_projections
        merged = merge_fpl_players_and_projections(self.FPL, self.PROJ)
        assert merged["Player_ID"].tolist() == [17]
        assert merged["Player_Code"].tolist() == [141746]
        # And the matched row still takes its *name* from the projection source.
        assert merged["Player"].tolist() == ["Bruno Fernandes"]

    def test_the_waiver_schema_coercion_keeps_it(self):
        """`_prepare_proj_for_merge` returned exactly the six columns the merge
        wants, so the ids resolved at the fetch boundary were discarded and the
        page went back to name matching."""
        from scripts.draft.waiver_wire import _prepare_proj_for_merge
        df = self.PROJ.copy()
        df["Player_ID"] = 17
        df["Player_Code"] = 141746
        out = _prepare_proj_for_merge(df)
        assert out["Player_ID"].tolist() == [17]
        assert out["Player_Code"].tolist() == [141746]

    def test_the_rotowire_boundary_stamps_it(self):
        """The keystone: Rotowire publishes names only, so if the boundary does
        not resolve them then six consumers each resolve them separately."""
        from unittest.mock import patch
        from scripts.common import scraping

        pool = pd.DataFrame([
            {"Player_ID": 17, "Player_Code": 141746,
             "Player": "Bruno Borges Fernandes", "Web_Name": "B.Fernandes",
             "Team": "MUN", "Position": "M"},
        ])
        with patch.object(scraping, "_bootstrap_player_pool", return_value=pool):
            out = scraping._attach_projection_ids(self.PROJ.copy())
        assert out["Player_ID"].tolist() == [17]
        assert out["Player_Code"].tolist() == [141746]

    def test_no_pool_degrades_rather_than_raising(self):
        """Every failure here must leave the app exactly as it was before any
        of this existed: name matching, which still works."""
        from unittest.mock import patch
        from scripts.common import scraping
        with patch.object(scraping, "_bootstrap_player_pool",
                          return_value=pd.DataFrame()):
            out = scraping._attach_projection_ids(self.PROJ.copy())
        assert len(out) == 1
        assert "Player_ID" not in out.columns


# ---------------------------------------------------------------------------
# 4. The deleted matchers stay deleted
# ---------------------------------------------------------------------------

def test_the_banned_matchers_are_gone():
    """Each of these was a name-keyed join with no caller, or a WRatio scorer
    whose scope reached the whole league. Pinned as deleted so none comes back
    as a convenience; the tombstones in the source say why each went."""
    import scripts.common.analytics as A
    import scripts.common.player_matching as pm

    for name in ("_backfill_player_ids", "PlayerRegistry", "get_player_registry",
                 "add_normalized_name_column", "normalize_for_merge"):
        assert not hasattr(pm, name), "player_matching.%s is back" % name
    for name in ("attach_availability", "team_optimizer", "_add_fdr_and_form"):
        assert not hasattr(A, name), "analytics.%s is back" % name
