"""One place that answers "what do we call this player?".

`to_display_name()` (`text_helpers`) turns the bootstrap's three name fields
into the common name — "Bruno Fernandes" out of "Bruno Borges Fernandes" and
"B.Fernandes". This module is everything around it: resolving that name for a
frame that carries only *some* of those fields, and rendering it.

**It exists because there were three bootstrap->display maps and four copies of
the fallback.** `waiver_wire._display_name_maps`, `element_ids.classic_id_names`
and `player_pool.build_pool` each built their own; `waiver_wire._display_of`,
`classic/fixture_projections._row_display_name` and two inline expressions each
re-implemented "Display_Name, else Player". Two more re-derived first/second
names by **splitting a legal name on whitespace**
(`pl_content.add_display_names`, `availability._display_name`), which is lossy:
"Igor Thiago Nascimento Rodrigues" splits to first="Igor", rest="Thiago
Nascimento Rodrigues", and `to_display_name` cannot recover "Igor Thiago" from
that the way it can from the real fields.

Two rules carried over from the implementations this replaces, both
load-bearing:

- **`Player` is never overwritten.** It is what every cross-source merge keys
  on. A match is for numbers, never for names — renaming a player to a matched
  source's name is what turns a bad match from a wrong number into a wrong
  person. See "Player Display Names" in CLAUDE.md.
- **A name-only key is kept only when it resolves to exactly one player.** A
  shared surname would otherwise print Cole Palmer's name on Alex Palmer's
  card: the display-side version of the match bug this codebase keeps paying
  for.

Resolution order is **code, then element id, then unique name**, which is the
app's join-key order (see `element_ids`). The code is the only id both FPL APIs
agree on, so a frame carrying one cannot be mislabelled by an id that crossed
API spaces -- the failure that gave a tenth of the pool another player's name.

Pure except for the cached `display_map()` accessor, so `pl_content` and the
other Streamlit-free modules can use `build_display_map` with a bootstrap they
already hold.
"""

from typing import Dict, NamedTuple, Optional

import pandas as pd

from scripts.common.error_helpers import get_logger
from scripts.common.text_helpers import canonical_normalize, to_display_name

_logger = get_logger(__name__)

#: Where a render-ready name is looked for, in order. `Web_Name` is last
#: because it is frequently an abbreviation ("B.Fernandes", "A.Becker") and an
#: abbreviation must never be the thing we render -- but it beats nothing.
DISPLAY_COLUMNS = ("Display_Name", "Player", "Web_Name")

#: Identity columns a frame may carry, in resolution order.
_ID_COLUMNS = (("Player_Code", "by_code"), ("Player_ID", "by_id"))


class DisplayMap(NamedTuple):
    """Bootstrap -> display name, keyed three ways."""

    by_code: Dict[int, str]
    by_id: Dict[int, str]
    by_name: Dict[str, str]

    @property
    def ok(self) -> bool:
        return bool(self.by_code or self.by_id or self.by_name)


def build_display_map(bootstrap) -> DisplayMap:
    """A :class:`DisplayMap` from a bootstrap payload. Pure.

    Pass the **Classic** bootstrap unless you know the frame's ids are Draft
    ids. The two APIs number 64 of 669 players differently, and labelling a row
    from the wrong one is silent -- it prints a real player's name onto another
    player's row. Keying on `Player_Code` avoids the question entirely.
    """
    by_code: Dict[int, str] = {}
    by_id: Dict[int, str] = {}
    seen: Dict[str, set] = {}

    for el in (bootstrap or {}).get("elements", []) or []:
        display = to_display_name(
            el.get("first_name"), el.get("second_name"), el.get("web_name"))
        if not display:
            continue
        for field, target in (("code", by_code), ("id", by_id)):
            try:
                target[int(el[field])] = display
            except (KeyError, TypeError, ValueError):
                pass
        full = " ".join(str(el.get(k) or "")
                        for k in ("first_name", "second_name")).strip()
        for raw in (full, el.get("web_name"), display):
            key = canonical_normalize(str(raw or ""))
            if key:
                seen.setdefault(key, set()).add(display)

    # Ambiguity resolves to *no* entry, never to whichever was seen first.
    by_name = {k: next(iter(v)) for k, v in seen.items() if len(v) == 1}
    return DisplayMap(by_code=by_code, by_id=by_id, by_name=by_name)


def display_map() -> DisplayMap:
    """:func:`build_display_map` over the cached Classic bootstrap.

    Empty on failure, which :func:`attach_display_names` reads as "could not
    resolve" and handles by leaving the source name in place. A missing display
    name is cosmetic; a wrong one is not.
    """
    try:
        import streamlit as st
    except Exception:                                  # pragma: no cover
        return _display_map_uncached()
    return st.cache_data(ttl=3600, show_spinner=False)(_display_map_uncached)()


def _display_map_uncached() -> DisplayMap:
    try:
        from scripts.common.fpl_classic_api import get_classic_bootstrap_static
        return build_display_map(get_classic_bootstrap_static())
    except Exception as exc:
        _logger.warning("Could not build the display-name map: %s", exc)
        return DisplayMap({}, {}, {})


def attach_display_names(df: pd.DataFrame,
                         dmap: Optional[DisplayMap] = None,
                         name_col: str = "Player") -> pd.DataFrame:
    """Add a ``Display_Name`` column, leaving ``name_col`` untouched.

    Resolves by player code, then element id, then a name that identifies
    exactly one player, and finally falls back to the frame's own name so a
    player the bootstrap cannot place still renders something sane rather than
    blank.
    """
    if df is None or getattr(df, "empty", True) or name_col not in df.columns:
        return df

    dmap = display_map() if dmap is None else dmap
    out = df.copy()

    lookups = []
    for col, attr in _ID_COLUMNS:
        if col in out.columns:
            lookups.append((pd.to_numeric(out[col], errors="coerce"),
                            getattr(dmap, attr)))

    raw_names = out[name_col].astype(str)
    resolved = []
    for pos, raw in enumerate(raw_names):
        hit = None
        for values, table in lookups:
            value = values.iloc[pos]
            if pd.notna(value):
                hit = table.get(int(value))
                if hit:
                    break
        if not hit:
            hit = dmap.by_name.get(canonical_normalize(raw))
        resolved.append(hit or raw)

    out["Display_Name"] = resolved
    return out


def display_of(row, default: str = "") -> str:
    """The name to render for one row.

    Accents are **kept**. The common name is the one with its diacritics in it
    -- "Martin Ødegaard", "Lukáš Horníček" -- and every other surface in the app
    renders them; the Waiver Wire stripped them for no recorded reason, so one
    page disagreed with the rest about what a player is called.
    """
    for col in DISPLAY_COLUMNS:
        value = row.get(col) if hasattr(row, "get") else None
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return default


def display_series(df: pd.DataFrame, default: str = "") -> pd.Series:
    """Render-ready names for ``df``.

    Empty-frame safe: ``DataFrame.apply`` over an empty frame returns a
    *DataFrame*, which cannot be assigned to a column.
    """
    if df is None or getattr(df, "empty", True):
        return pd.Series(dtype=str, index=getattr(df, "index", None))
    return df.apply(lambda row: display_of(row, default), axis=1)
