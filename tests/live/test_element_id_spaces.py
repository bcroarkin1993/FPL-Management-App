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
