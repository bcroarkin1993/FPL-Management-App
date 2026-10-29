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

from typing import Dict, Iterable, Optional

import pandas as pd
import streamlit as st

from scripts.common.error_helpers import get_logger

_logger = get_logger(__name__)

_DRAFT_BOOTSTRAP = "https://draft.premierleague.com/api/bootstrap-static"


def _code_maps(draft_bootstrap, classic_bootstrap):
    """``(draft_id -> code, code -> classic_id)``, the two halves of the bridge."""
    draft_to_code, code_to_classic = {}, {}
    for el in (draft_bootstrap or {}).get("elements", []) or []:
        try:
            draft_to_code[int(el["id"])] = int(el["code"])
        except (KeyError, TypeError, ValueError):
            continue
    for el in (classic_bootstrap or {}).get("elements", []) or []:
        try:
            code_to_classic[int(el["code"])] = int(el["id"])
        except (KeyError, TypeError, ValueError):
            continue
    return draft_to_code, code_to_classic


@st.cache_data(ttl=3600, show_spinner=False)
def draft_to_classic_ids() -> Dict[int, str]:
    """``{draft element id: classic element id}``.

    Empty on any failure, which callers must read as "cannot translate" and
    handle by *not* joining -- never by assuming the spaces agree, which is the
    bug this module exists for.
    """
    import requests
    from scripts.common.fpl_classic_api import get_classic_bootstrap_static

    try:
        draft = requests.get(_DRAFT_BOOTSTRAP, timeout=30).json()
    except Exception as exc:
        _logger.warning("Could not read the Draft bootstrap for id translation: %s", exc)
        return {}
    try:
        classic = get_classic_bootstrap_static()
    except Exception as exc:
        _logger.warning("Could not read the Classic bootstrap for id translation: %s", exc)
        return {}

    draft_to_code, code_to_classic = _code_maps(draft, classic)
    out = {}
    for draft_id, code in draft_to_code.items():
        classic_id = code_to_classic.get(code)
        if classic_id is not None:
            out[draft_id] = classic_id
    if not out:
        _logger.warning("Draft->Classic id translation resolved nothing.")
    return out


def translate_ids(ids: Iterable, mapping: Optional[Dict[int, int]] = None) -> list:
    """Draft ids as Classic ids, dropping any that cannot be translated.

    An untranslatable id is dropped rather than passed through: passing it
    through is exactly what resolves it to the wrong player.
    """
    mapping = draft_to_classic_ids() if mapping is None else mapping
    if not mapping:
        return list(ids)        # no bridge -- caller must decide what that means
    out = []
    for raw in ids:
        try:
            out.append(mapping[int(raw)])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def translate_keyed(values: Dict, mapping: Optional[Dict[int, int]] = None) -> Dict:
    """Re-key a ``{draft element id: anything}`` dict into the Classic space."""
    mapping = draft_to_classic_ids() if mapping is None else mapping
    if not mapping:
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
    """Draft ids in a Series as Classic ids, NaN where untranslatable."""
    mapping = draft_to_classic_ids() if mapping is None else mapping
    nums = pd.to_numeric(series, errors="coerce")
    if not mapping:
        return nums
    return nums.map(lambda v: mapping.get(int(v)) if pd.notna(v) else None)
