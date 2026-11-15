"""A `mock.patch` that silently does nothing is worse than no mock at all.

Patching a name on a module only affects calls resolved from *that module's*
globals -- calls written inside a function that module defines. Move the
calling function elsewhere and the patch keeps succeeding (the attribute is
still there, via a re-export) while the real function runs.

That is not hypothetical. Splitting the claim engine out of the Waiver Wire
page detached eight patches at once:

- two were outright no-ops in the page's smoke test, one of them
  `compute_healthy_form`, which drives one `element-summary` HTTP request *per
  player* -- inside a test whose body is `except Exception: pass`;
- two were `side_effect=AssertionError` guards asserting a function is never
  reached. Detached, they assert nothing, and both tests still passed;
- one unit test went green *while reading the live API*, because the real
  element map happens to return the ids its fixture hardcodes.

Only the first of those eight announced itself. So the property is checked
directly: every name a test patches on a guarded module must actually be
called from a function that module defines.
"""

import ast
import collections
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Modules whose patch targets are guarded. The pair that split is the point;
#: add a module here when tests start patching it.
GUARDED = {
    "scripts.draft.waiver_wire": ROOT / "scripts" / "draft" / "waiver_wire.py",
    "scripts.draft.waiver_core": ROOT / "scripts" / "draft" / "waiver_core.py",
}


def _globals_used_in_functions(path: pathlib.Path) -> collections.Counter:
    """Names loaded inside function bodies -- exactly what a patch can reach."""
    used = collections.Counter()
    for fn in (n for n in ast.walk(ast.parse(path.read_text()))
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used[node.id] += 1
            elif isinstance(node, ast.Attribute):
                used[node.attr] += 1
    return used


def _alias_map(text: str) -> dict:
    """`import scripts.draft.waiver_core as wc` -> {"wc": "scripts.draft.waiver_core"}."""
    aliases = {}
    for m in re.finditer(
            r"import\s+(scripts\.draft\.waiver_(?:wire|core))\s+as\s+(\w+)", text):
        aliases[m.group(2)] = m.group(1)
    for m in re.finditer(
            r"from\s+scripts\.draft\s+import\s+waiver_(wire|core)\s+as\s+(\w+)", text):
        aliases[m.group(2)] = "scripts.draft.waiver_" + m.group(1)
    return aliases


def _patch_calls(tree, aliases):
    """Every guarded patch target in a file, read from the AST.

    AST rather than a text scan so that prose cannot trigger the guard: this
    module's docstring names a detached patch as its worked example, and a
    regex over the file duly flagged it.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        fn = node.func
        is_patch = isinstance(fn, ast.Name) and fn.id == "patch"
        is_object = (isinstance(fn, ast.Attribute) and fn.attr == "object"
                     and getattr(fn.value, "id", None) == "patch")
        first = node.args[0]
        if is_patch and isinstance(first, ast.Constant) and isinstance(first.value, str):
            mod, _, name = first.value.rpartition(".")
            if mod in GUARDED:
                yield mod, name, node.lineno
        elif is_object and len(node.args) >= 2:
            alias, second = getattr(first, "id", None), node.args[1]
            if (alias in aliases and isinstance(second, ast.Constant)
                    and isinstance(second.value, str)):
                yield aliases[alias], second.value, node.lineno


def _collect():
    reach = {mod: _globals_used_in_functions(p) for mod, p in GUARDED.items()}
    detached = []
    for f in sorted((ROOT / "tests").rglob("*.py")):
        text = f.read_text(errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        aliases = _alias_map(text)
        for mod, name, lineno in _patch_calls(tree, aliases):
            if reach[mod].get(name, 0) == 0:
                other = next((m for m in GUARDED
                              if m != mod and reach[m].get(name, 0) > 0), None)
                detached.append((f.relative_to(ROOT), lineno, mod, name, other))
    return detached


def test_no_patch_target_is_detached():
    detached = _collect()
    assert detached == [], "\n".join(
        ["%s:%d patches %s.%s, which that module never calls%s"
         % (f, ln, mod, name, (" -- it lives in %s now" % other) if other else "")
         for f, ln, mod, name, other in detached]
        + ["", "Such a patch succeeds and does nothing: the real function runs, "
              "possibly over the network, and the test still passes."])


def test_the_guard_can_actually_see_a_patch():
    """A guard whose parsing matches nothing is a permanent green tick."""
    reach = _globals_used_in_functions(GUARDED["scripts.draft.waiver_core"])
    assert reach, "parsed no function-body globals out of waiver_core"
    assert reach.get("compute_healthy_form", 0) > 0, (
        "expected waiver_core to call compute_healthy_form; if that moved, "
        "repoint this guard rather than deleting it")

    found = []
    for f in sorted((ROOT / "tests").rglob("*.py")):
        text = f.read_text(errors="ignore")
        found += list(_patch_calls(ast.parse(text), _alias_map(text)))
    assert len(found) >= 15, (
        "the AST walker found only %d guarded patch targets; the suite has "
        "many more, so the parsing has stopped matching" % len(found))
