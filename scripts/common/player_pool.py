"""The canonical player universe, and the blend over it.

Both functions used to live in ``scripts/fpl/player_projections.py``, private to
the Projections Hub. They are here because a second page now needs the same
frame, and re-deriving a blend is the twin-drift failure this codebase has paid
for three times -- see "Projection Engine -- one blend, one contract" in
CLAUDE.md. One implementation, one cache key, two readers.

Not Streamlit-free: ``blended_pool`` owns the cache. Pure consumers (such as
``mini_league``) take the frame as an argument rather than importing this.
"""

import pandas as pd
import streamlit as st

from scripts.common.analytics import (
    blend_multi_gw_projections,
    blend_projections_onto,
    merge_season_projections,
)
from scripts.common.fpl_classic_api import get_classic_bootstrap_static
from scripts.common.scraping import get_ffp_feed
from scripts.common.text_helpers import to_display_name
from scripts.common.utils import get_rotowire_player_projections


def build_pool(bootstrap) -> pd.DataFrame:
    """The canonical player universe for the blend, from the FPL bootstrap.

    Carries both name forms on purpose: ``Player`` is the full legal name that
    projection sources are matched against, ``Display_Name`` is what a human
    calls him. Match on Player, display Display_Name -- swapping them silently
    degrades match rates.
    """
    teams = {t["id"]: t["short_name"] for t in bootstrap.get("teams", [])}
    pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    rows = []
    for e in bootstrap.get("elements", []):
        first, second = e.get("first_name", ""), e.get("second_name", "")
        rows.append({
            "Player_ID": e.get("id"),
            # FPL's permanent per-player id, and the only one the Draft and
            # Classic APIs agree on -- 669 of 669, against 605 for the element
            # id. Carried so a cross-source join has an id it cannot
            # misinterpret: codes and element ids occupy disjoint ranges, so a
            # mixup resolves to nothing rather than to the wrong player. See
            # `scripts/common/element_ids.py`.
            "Player_Code": e.get("code"),
            "Player": f"{first} {second}".strip(),
            "Display_Name": to_display_name(first, second, e.get("web_name", "")),
            "Web_Name": e.get("web_name", ""),
            "Team": teams.get(e.get("team"), ""),
            # The club *id*, because every fixture join is keyed on it:
            # `club_fixture_counts` is indexed by team id, and a short
            # name is a label rather than a key.
            "Team_ID": e.get("team"),
            "Position": pos_map.get(e.get("element_type"), "M"),
            "ep_next": pd.to_numeric(e.get("ep_next"), errors="coerce"),
            # Purely additive, and needed by anything that has to *buy* a
            # squad rather than only rank one -- the chip page solves a Free
            # Hit and a Wildcard against this pool, and both are budgeted.
            "Price": pd.to_numeric(e.get("now_cost"), errors="coerce") / 10.0,
            # Tenths, as FPL publishes it. `Price` above is the same
            # number in millions; the solvers budget in millions and
            # the transfer planner reads tenths, so both are carried
            # rather than re-derived at a callsite.
            "now_cost": pd.to_numeric(e.get("now_cost"), errors="coerce"),
            "status": e.get("status"),
            # Buy-eligibility in the transfer planner: a player with no
            # minutes all season is not a candidate.
            "minutes": pd.to_numeric(e.get("minutes"), errors="coerce"),
            "chance_of_playing_next_round": e.get("chance_of_playing_next_round"),
            # The player's own start count, which floors the omitted prior:
            # a man every published XI leaves out who has nonetheless been
            # starting is not the never-starter that prior was fitted on.
            # See `config.OMITTED_START_HISTORY_WEIGHT`.
            "starts": pd.to_numeric(e.get("starts"), errors="coerce"),
            # Carries the return date the horizon model reads -- "Expected back
            # 11 Oct" is three gameweeks at GW6.
            "news": e.get("news"),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def blended_pool(current_gw: int, rotowire_url: str):
    """The blended projection for ``current_gw``, built once per page load.

    Several surfaces need this frame -- the Projections Hub renders it directly,
    Horizon takes its first column from it, and the Classic Rival Tracker prices
    every squad in a mini-league with it -- and ``st.tabs`` runs every tab body
    on every rerun, so without a cache the same blend is assembled repeatedly on
    each interaction. Measured warm, that was 4.4s for the first assembly and a
    further ~0.9s for the second; the name-matched ``merge_season_projections``
    is most of it.

    Cached on ``(gameweek, rotowire_url)`` rather than on the frames it reads:
    both are small and hashable, and the fetches inside are themselves cached,
    so a miss here is cheap. The TTL matches ``get_ffp_feed``'s so the blend and
    the feed it was built from roll over together rather than the blend pinning
    a feed that has since been refreshed.

    ``st.cache_data`` returns a fresh object per call, so a caller mutating the
    result cannot corrupt the other's copy -- which ``cache_resource`` would.
    """
    bootstrap = get_classic_bootstrap_static()
    if not bootstrap:
        return pd.DataFrame()
    pool = build_pool(bootstrap)
    if pool.empty:
        return pd.DataFrame()

    rotowire_df = pd.DataFrame()
    if rotowire_url:
        rotowire_df = get_rotowire_player_projections(rotowire_url)
    feed = get_ffp_feed()

    # Rotowire values land on the pool by name (merge_season_projections is the
    # app's tiered name-matched merge); FFP joins on the element id inside
    # blend_projections_onto.
    pool = merge_season_projections(pool, rotowire_df, output_col="Points")

    # The multi-gameweek merge has to happen *before* the blend, because the
    # engine converts the horizon's basis along with everything else -- see
    # "A multi-gameweek total has a basis too" in CLAUDE.md. Without it
    # `Proj_Next3` came back all-zero on this pool: no source supplied a
    # horizon, so the only non-null values were the players the engine had
    # zeroed outright. Nothing rendered it, so nothing caught it.
    #
    # Verified on the live GW7 pool: `Proj`, `Proj_Start` and `Start_Pct` are
    # byte-identical with and without this, and the horizon goes from max 0.00
    # to a median (Proj_Next3 / 3) / Proj of 1.04 -- the population ratio
    # check_blended_projections() tests, which is 1.0 when the two share a
    # basis.
    pool = blend_multi_gw_projections(pool, feed.df, single_gw_col="Points",
                                      expected_gw=current_gw)
    return blend_projections_onto(pool, feed.df, expected_gw=current_gw)


def squad_frame(picks, pool) -> pd.DataFrame:
    """A manager's fifteen as a frame carrying position, projection and price.

    Joined on the element id, which is sound here because both sides come from
    the Classic API: the picks are a Classic endpoint's and the pool is built
    from the Classic bootstrap. Carries **both** ``element_id`` and
    ``Player_ID``, because Classic frames spell the id four different ways and a
    gate keyed on the wrong spelling fails silently -- which is how a benched
    player went on projecting 4.24 on one page and 0.45 everywhere else.

    Lifted out of the Chip Strategy page when the My Week checklist became a
    second reader; that page re-exports it under its old private name.
    """
    rows = []
    by_id = {}
    if pool is not None and not pool.empty and "Player_ID" in pool.columns:
        ids = pd.to_numeric(pool["Player_ID"], errors="coerce")
        by_id = {int(i): r for i, (_, r) in zip(ids, pool.iterrows())
                 if pd.notna(i)}
    for pick in picks or []:
        element = pick.get("element")
        if element is None:
            continue
        ref = by_id.get(int(element))
        rows.append({
            "element_id": int(element),
            "Player_ID": int(element),
            "Player_Code": (ref.get("Player_Code") if ref is not None else None),
            "Display_Name": (ref.get("Display_Name") if ref is not None else ""),
            "Team": (ref.get("Team") if ref is not None else ""),
            "Team_ID": (ref.get("Team_ID") if ref is not None else None),
            "Position": (ref.get("Position") if ref is not None else ""),
            "Proj": (ref.get("Proj") if ref is not None else 0.0),
            "Proj_Next3": (ref.get("Proj_Next3") if ref is not None else None),
            "Price": (ref.get("Price") if ref is not None else 0.0),
            "now_cost": (ref.get("now_cost") if ref is not None else None),
            "status": (ref.get("status") if ref is not None else ""),
            # FPL publishes a selling price only to an authenticated request, so
            # on the public path this is absent and the planner falls back to
            # `now_cost` -- which overstates funds. See CLAUDE.md,
            # "Pre-deadline Classic squad".
            "selling_price": pick.get("selling_price"),
            "multiplier": pick.get("multiplier", 1),
            "is_captain": bool(pick.get("is_captain")),
            # 1-11 is the starting XI, 12-15 the bench in order. Carried so a
            # caller can check the lineup a manager has actually *set*, rather
            # than only the best one available -- the multiplier cannot say,
            # because `my-team` publishes none at all.
            "pick_position": pick.get("position"),
        })
    return pd.DataFrame(rows)
