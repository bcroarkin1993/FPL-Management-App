"""Price-change signals, as FPL itself publishes them.

Fetch-free: the frame is built from a bootstrap payload the caller has already
fetched, so two pages can read the same states without either owning them.
``scripts/fpl/price_changes.py`` re-exports the two private names it used to own,
so that page is unchanged. (Not import-free of Streamlit, via
``display_names`` -> ``error_helpers``; nothing in GitHub Actions reads this.)

Lifted out for the same reason ``rotowire_lineups.py`` came out of
``projected_lineups.py``: a second consumer arrived (the Classic **My Week**
checklist), and the alternative -- importing a page's private helper from
another page -- is how two copies of one rule start.
"""

from typing import Iterable, List, Optional

import pandas as pd

from scripts.common.display_names import display_of
from scripts.common.text_helpers import to_display_name

#: FPL ``element_type`` -> the position letter the rest of the app uses.
POS_LETTER = {1: "G", 2: "D", 3: "M", 4: "F"}

STATE_RISE = "rise"
STATE_FALL = "fall"
STATE_LOCKED = "locked"
STATE_CALIBRATING = "calibrating"
STATE_STEADY = "steady"


def build_price_data(bootstrap: Optional[dict]) -> pd.DataFrame:
    """Every player's price-change state, from the Classic bootstrap.

    Keyed on ``Player_Code`` as well as the element id, because this frame is
    joined to squads that may have been resolved from either API and ``code`` is
    the only identifier the two agree on.
    """
    bootstrap = bootstrap or {}
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    rows = []
    for p in bootstrap.get("elements", []):
        rows.append({
            "Player_Code": p.get("code"),
            "Player_ID": p.get("id"),
            "Player": p.get("web_name", ""),
            "Display_Name": to_display_name(
                p.get("first_name"), p.get("second_name"), p.get("web_name")),
            "Team": teams.get(p.get("team"), ""),
            "Pos": POS_LETTER.get(p.get("element_type"), "?"),
            "now_cost": p.get("now_cost", 0),
            "cost_change_event": p.get("cost_change_event", 0),
            "cost_change_start": p.get("cost_change_start", 0),
            "cost_change_start_fall": p.get("cost_change_start_fall", 0),
            "transfers_in_event": p.get("transfers_in_event", 0),
            "transfers_out_event": p.get("transfers_out_event", 0),
            "transfers_in": p.get("transfers_in", 0),
            "transfers_out": p.get("transfers_out", 0),
            "selected_by_percent": p.get("selected_by_percent", "0"),
            # FPL's own price-change prediction. Strings on the wire.
            "Change_Pct": pd.to_numeric(p.get("price_change_percent"), errors="coerce"),
            "Hourly": pd.to_numeric(p.get("price_change_hourly_rate"), errors="coerce"),
            "Locked_Until": p.get("price_change_locked_until"),
            "Calibrating": bool(p.get("price_change_calibrating")),
            "Projections": p.get("price_change_projections") or [],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Net_Event"] = (pd.to_numeric(df["transfers_in_event"], errors="coerce").fillna(0)
                       - pd.to_numeric(df["transfers_out_event"], errors="coerce").fillna(0))
    return df


def change_state(row) -> str:
    """``rise`` / ``fall`` / ``locked`` / ``calibrating`` / ``steady``.

    Order matters. **Locked is checked before the threshold**, because a player
    who has already changed today sits above 100% and is the one case where a
    bare percentage reads as the opposite of the truth: FPL allows one change per
    player per day. Calibrating comes next, since FPL is telling us its own
    number is not yet usable.
    """
    pct = pd.to_numeric(row.get("Change_Pct"), errors="coerce")
    # **`bool(float("nan"))` is True**, so a null-unsafe truth test reads a
    # missing timestamp as a lock. The live frame holds `None` and so happens to
    # work, but any float round-trip of the column would mark every player
    # locked -- and "locked" is the state that suppresses the whole signal.
    locked = row.get("Locked_Until")
    if locked is not None and pd.notna(locked) and str(locked).strip():
        return STATE_LOCKED
    calibrating = row.get("Calibrating")
    if calibrating is not None and pd.notna(calibrating) and bool(calibrating):
        return STATE_CALIBRATING
    if pd.isna(pct):
        return STATE_STEADY
    if pct >= 100:
        return STATE_RISE
    if pct <= -100:
        return STATE_FALL
    return STATE_STEADY


def squad_price_pressure(price_df: Optional[pd.DataFrame],
                         codes: Iterable, limit: int = 3) -> Optional[dict]:
    """What tonight's price moves do to one squad.

    Returns ``None`` when the question cannot be answered at all -- no frame, no
    codes, or a frame that matches none of them -- which a caller must report as
    *unknown* rather than as "nothing is moving". The two are opposite claims and
    look identical on a card.

    ``nearest`` is ranked by **proximity to a change, not by whether the
    threshold is already crossed**, for the reason the Price Changes page records:
    filtering to rise/fall printed "nothing close" while a squad player sat at
    -95% and accelerating. Fallers lead, since a fall costs money you already
    have where a rise on a player you own is free.
    """
    codes = {c for c in (codes or []) if c is not None}
    if price_df is None or price_df.empty or not codes:
        return None
    mine = price_df[price_df["Player_Code"].isin(codes)].copy()
    if mine.empty:
        return None

    mine["_state"] = mine.apply(change_state, axis=1)
    near = mine.dropna(subset=["Change_Pct"]).copy()
    nearest: List[dict] = []
    if not near.empty:
        near["_prox"] = near["Change_Pct"].abs()
        near = near.nlargest(min(limit, len(near)), "_prox").sort_values("Change_Pct")
        nearest = [{
            # `display_of` owns the Display_Name -> Player -> Web_Name order and
            # the NaN handling; four hand-rolled copies of it had drifted.
            "name": display_of(r),
            "pct": float(r["Change_Pct"]),
            "state": r["_state"],
            "hourly": (float(r["Hourly"]) if pd.notna(r.get("Hourly")) else None),
        } for _, r in near.iterrows()]

    return {
        "tracked": int(len(mine)),
        "rising": int((mine["_state"] == STATE_RISE).sum()),
        "falling": int((mine["_state"] == STATE_FALL).sum()),
        "priced": int(mine["Change_Pct"].notna().sum()),
        "nearest": nearest,
    }
