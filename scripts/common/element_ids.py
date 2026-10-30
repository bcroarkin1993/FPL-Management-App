"""Translating between the Draft and Classic element id spaces.

**The two FPL APIs do not agree about element ids.** Both publish the same 669
players, but `draft.premierleague.com/api/bootstrap-static` and
`fantasy.premierleague.com/api/bootstrap-static/` number the late additions
differently -- measured 2026-10-29, **64 of 669 ids (9.6%) name a different
player in each**:

    id 566   draft: Lukás Hornícek      classic: António João ... da Silva
    id 567   draft: Xaver Schlager      classic: Lukás Hornícek

The disagreement is confined to the high ids, which is what a mid-season
signing appended in a different order looks like. `code` -- FPL's permanent
per-player identifier -- agrees on **all 669**, so it is the bridge.

This matters because the Draft-only endpoints (`element-status`, league
rosters, `get_fpl_player_mapping`) speak the Draft space while every stats
frame in the app is built from `pull_fpl_player_stats()` / the Classic
bootstrap and so carries Classic ids. Joining the two directly is silent: the
id exists on both sides and resolves to *somebody*, just the wrong somebody.
Observed symptom -- the Waiver Wire offering "drop Nordi Mukiele for Lukas
Hornicek", a defender for a goalkeeper, because António Silva's row had been
labelled with Hornicek's name.

Measured against one live 10-team league, 7 of its 150 rostered players were
affected.
"""

from typing import Dict, Iterable, NamedTuple, Optional

import pandas as pd
import streamlit as st

from scripts.common.error_helpers import get_logger

_logger = get_logger(__name__)

_DRAFT_BOOTSTRAP = "https://draft.premierleague.com/api/bootstrap-static"


#: `code` and element ids occupy disjoint numeric ranges, and that is a
#: load-bearing safety property rather than a coincidence worth noting.
#: Measured live 2026-10-30: codes run 17761-700308 and element ids 1-669, with
#: **zero** overlap. So a code used where an element id is expected -- or the
#: reverse -- resolves to *nothing*, where a Draft id used as a Classic id
#: resolves to the wrong player. One failure is an empty join you can see; the
#: other is a plausible number you cannot. `tests/live/` pins the separation,
#: because the whole point of keying on `code` rests on it.
PLAYER_CODE_MIN = 10000

#: Generous: 669 live, and the pool grows through the season as players are
#: registered. Only the gap to PLAYER_CODE_MIN matters.
ELEMENT_ID_MAX = 2000


class CodeMaps(NamedTuple):
    """Every direction of the bridge, built from one pair of bootstraps.

    `code` is FPL's permanent per-player identifier and the only id the two
    APIs agree on, so it is the app's join key. The element-id maps exist to
    translate at a boundary, never to join across one.
    """

    draft_id_to_code: Dict[int, int]
    code_to_classic_id: Dict[int, int]
    classic_id_to_code: Dict[int, int]
    code_to_draft_id: Dict[int, int]

    @property
    def ok(self) -> bool:
        return bool(self.draft_id_to_code or self.classic_id_to_code)


def _code_maps(draft_bootstrap, classic_bootstrap) -> CodeMaps:
    """:class:`CodeMaps` from two bootstrap payloads. Pure."""
    draft_to_code, classic_to_code = {}, {}
    for el in (draft_bootstrap or {}).get("elements", []) or []:
        try:
            draft_to_code[int(el["id"])] = int(el["code"])
        except (KeyError, TypeError, ValueError):
            continue
    for el in (classic_bootstrap or {}).get("elements", []) or []:
        try:
            classic_to_code[int(el["id"])] = int(el["code"])
        except (KeyError, TypeError, ValueError):
            continue
    return CodeMaps(
        draft_id_to_code=draft_to_code,
        code_to_classic_id={c: i for i, c in classic_to_code.items()},
        classic_id_to_code=classic_to_code,
        code_to_draft_id={c: i for i, c in draft_to_code.items()},
    )


def _fetch_bootstraps():
    """``(draft, classic)``, either of which may be None."""
    import requests
    from scripts.common.fpl_classic_api import get_classic_bootstrap_static

    draft = classic = None
    try:
        draft = requests.get(_DRAFT_BOOTSTRAP, timeout=30).json()
    except Exception as exc:
        _logger.warning("Could not read the Draft bootstrap: %s", exc)
    try:
        classic = get_classic_bootstrap_static()
    except Exception as exc:
        _logger.warning("Could not read the Classic bootstrap: %s", exc)
    return draft, classic


@st.cache_data(ttl=3600, show_spinner=False)
def code_maps() -> CodeMaps:
    """The bridge, fetched once and shared.

    One cached reader so the two bootstraps are fetched once per hour rather
    than per consumer -- :func:`draft_to_classic_ids` and
    :func:`classic_id_names` previously each fetched their own, and every
    further consumer would have added another.

    Empty maps on failure; see :func:`draft_to_classic_ids` for what a caller
    must do with that.
    """
    return _code_maps(*_fetch_bootstraps())


def draft_to_classic_ids() -> Dict[int, int]:
    """``{draft element id: classic element id}``.

    Empty on any failure. A caller holding an empty bridge has no way to
    translate and must decide what that means for it: the `translate_*`
    helpers below keep the Draft ids and log, while
    `get_team_composition_for_gameweek` keeps them and warns. Neither is
    silent, because the one thing that must not happen is a Draft id being
    *presented* as a Classic one.
    """
    maps = code_maps()
    out = {}
    for draft_id, code in maps.draft_id_to_code.items():
        classic_id = maps.code_to_classic_id.get(code)
        if classic_id is not None:
            out[draft_id] = classic_id
    if not out:
        _logger.warning("Draft->Classic id translation resolved nothing.")
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def classic_id_names() -> Dict[int, tuple]:
    """``{classic element id: (full name, web name, display name)}``.

    The reference `data_validation.check_element_id_names` compares a frame's
    own names against, so that an id which has crossed from the Draft space can
    be caught by the one thing that reveals it: the row carries both an id and
    a name, and they disagree.

    All three name forms are returned because the app's frames carry whichever
    one their source publishes -- Rotowire the common name, the bootstrap the
    full legal name, and a handful of players (``Costinha`` for "João Pedro
    Loureiro da Costa") are only reachable through the web name.
    """
    from scripts.common.fpl_classic_api import get_classic_bootstrap_static
    from scripts.common.text_helpers import to_display_name

    try:
        bootstrap = get_classic_bootstrap_static()
    except Exception as exc:
        _logger.warning("Could not read the Classic bootstrap for id names: %s", exc)
        return {}

    out = {}
    for el in (bootstrap or {}).get("elements", []) or []:
        try:
            pid = int(el["id"])
        except (KeyError, TypeError, ValueError):
            continue
        first = el.get("first_name") or ""
        second = el.get("second_name") or ""
        web = el.get("web_name") or ""
        out[pid] = (
            ("%s %s" % (first, second)).strip(),
            web,
            to_display_name(first, second, web) or "",
        )
    return out


def translate_ids(ids: Iterable, mapping: Optional[Dict[int, int]] = None) -> list:
    """Draft ids as Classic ids, dropping any that cannot be translated.

    An untranslatable id is dropped rather than passed through: passing it
    through is exactly what resolves it to the wrong player.

    **With no bridge at all the ids are returned unchanged**, which is a
    deliberate trade rather than an oversight: emptying a roster on a transient
    bootstrap failure is a visible functional loss, where passing ids through
    is wrong for 9.6% of players. The cost is that the result is then
    indistinguishable from a successful translation, so the no-bridge path
    logs -- it must never be silent.
    """
    mapping = draft_to_classic_ids() if mapping is None else mapping
    if not mapping:
        _logger.warning(
            "No Draft->Classic bridge: %d id(s) stay in the Draft space and may "
            "resolve to the wrong player downstream.", len(list(ids) or []))
        return list(ids)
    out = []
    for raw in ids:
        try:
            out.append(mapping[int(raw)])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def translate_keyed(values: Dict, mapping: Optional[Dict[int, int]] = None) -> Dict:
    """Re-key a ``{draft element id: anything}`` dict into the Classic space.

    No bridge leaves the keys alone -- see :func:`translate_ids` for why, and
    for why that path is logged rather than silent.
    """
    mapping = draft_to_classic_ids() if mapping is None else mapping
    if not mapping:
        _logger.warning(
            "No Draft->Classic bridge: %d key(s) stay in the Draft space.",
            len(values or {}))
        return dict(values)
    out, dropped = {}, 0
    for raw, val in (values or {}).items():
        try:
            out[mapping[int(raw)]] = val
        except (KeyError, TypeError, ValueError):
            dropped += 1
    if dropped:
        _logger.info("Dropped %d element id(s) with no Classic counterpart.", dropped)
    return out


def translate_series(series: pd.Series,
                     mapping: Optional[Dict[int, int]] = None) -> pd.Series:
    """Draft ids in a Series as Classic ids, NaN where untranslatable.

    No bridge leaves the ids alone -- see :func:`translate_ids`.
    """
    mapping = draft_to_classic_ids() if mapping is None else mapping
    nums = pd.to_numeric(series, errors="coerce")
    if not mapping:
        _logger.warning(
            "No Draft->Classic bridge: %d id(s) stay in the Draft space.",
            int(nums.notna().sum()))
        return nums
    return nums.map(lambda v: mapping.get(int(v)) if pd.notna(v) else None)
