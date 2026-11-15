"""`waiver_core` computes; it must never render or halt.

`st.stop()` is an instruction to Streamlit's script runner, not a return. A
helper that calls it can end a page it has never heard of -- and
`_available_from_projections`, a function whose job is to filter a DataFrame,
could do exactly that. So could `_align_roster_player_names_to_projections`,
via `st.warning`. That is why the boundary could not be drawn as a line range
through the page function, and why it is worth pinning afterwards.

`import streamlit` itself is allowed, and has to be: six of these functions are
memoised and `@st.cache_data` is how this codebase memoises. What is banned is
*rendering* and *control flow* -- the page decides what a reader sees and
whether to stop, from a status and a note the engine returns.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "scripts" / "draft" / "waiver_core.py"

#: Everything `st.` offers except memoisation. A library may not write to the
#: page, and above all may not end it.
BANNED = {
    "stop", "error", "warning", "info", "success", "caption", "write",
    "markdown", "dataframe", "table", "metric", "columns", "container",
    "expander", "tabs", "subheader", "header", "title", "spinner", "button",
    "selectbox", "multiselect", "slider", "checkbox", "radio", "number_input",
    "text_input", "form", "plotly_chart", "altair_chart", "rerun", "divider",
    "session_state", "empty", "sidebar", "toggle", "progress", "code", "json",
}
ALLOWED = {"cache_data", "cache_resource"}


def _streamlit_attrs(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and getattr(node.value, "id", None) == "st"):
            yield node.attr, node.lineno


def test_waiver_core_never_renders_or_stops():
    tree = ast.parse(CORE.read_text())
    offences = [(a, ln) for a, ln in _streamlit_attrs(tree) if a in BANNED]
    assert offences == [], (
        "waiver_core.py calls Streamlit rendering/control-flow at: "
        + ", ".join("st.%s (line %d)" % (a, ln) for a, ln in offences)
        + ". The engine must return a status and a note and let the page "
          "decide -- st.stop() in particular ends a page this module knows "
          "nothing about.")


def test_only_caching_uses_streamlit_here():
    tree = ast.parse(CORE.read_text())
    used = {a for a, _ in _streamlit_attrs(tree)}
    assert used, "no st.* usage found at all -- has the module moved?"
    unexpected = used - ALLOWED - BANNED
    assert unexpected == set(), (
        "unrecognised Streamlit attributes in waiver_core: %s. Decide whether "
        "each is memoisation (allowed) or rendering (banned) and list it."
        % sorted(unexpected))
    assert used <= ALLOWED, "expected only %s, found %s" % (sorted(ALLOWED), sorted(used))


def test_the_page_still_owns_the_rendering():
    """The counterpart: the split is only meaningful if the page kept it."""
    wire = ast.parse((ROOT / "scripts" / "draft" / "waiver_wire.py").read_text())
    used = {a for a, _ in _streamlit_attrs(wire)}
    assert "stop" in used, "waiver_wire should now own the st.stop() calls"
    assert used & {"error", "caption"}, "waiver_wire should own the messaging"
