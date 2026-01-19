# waiver_wire.py
import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List

import config
from openai import OpenAI

from scripts.common.utils import (
    get_current_gameweek,
    merge_fpl_players_and_projections,
    get_league_player_ownership,
    get_rotowire_player_projections,
    pull_fpl_player_stats,
    normalize_fpl_players_to_rotowire_schema,
    normalize_rotowire_players,
    _bootstrap_teams_df,
    _norm_text,
    _backfill_player_ids
)

# ---------------------------
# FPL API READS
# ---------------------------

@st.cache_data(show_spinner=False)
def _load_bootstrap() -> Dict[str, Any]:
    """bootstrap-static with players & teams (cached)."""
    url = "https://draft.premierleague.com/api/bootstrap-static"
    return requests.get(url, timeout=30).json()

def _teams_map_short() -> Dict[int, str]:
    """Map FPL team id -> short_name (e.g., 13 -> 'MCI')."""
    data = _load_bootstrap()
    return {t['id']: t['short_name'] for t in data.get('teams', [])}

def _team_id_by_short(short_name: str) -> Optional[int]:
    """Reverse lookup: short_name -> team id."""
    m = {v: k for k, v in _teams_map_short().items()}
    return m.get(short_name)

@st.cache_data(show_spinner=False)
def _load_future_fixtures() -> pd.DataFrame:
    """
    Returns future fixtures with difficulties.
    Columns: event, kickoff_time, team_h, team_a, team_h_difficulty, team_a_difficulty
    """
    url = "https://fantasy.premierleague.com/api/fixtures/?future=1"
    fx = requests.get(url, timeout=30).json()
    df = pd.DataFrame(fx)
    keep = ["event", "kickoff_time", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    df = df[[c for c in keep if c in df.columns]].copy()
    return df

@st.cache_data(show_spinner=False)
def _element_history_df(player_id: int) -> Optional[pd.DataFrame]:
    """Return per-GW history for a player (None on failure)."""
    try:
        url = f"https://fantasy.premierleague.com/api/element-summary/{player_id}/"
        js = requests.get(url, timeout=30).json()
        hist = pd.DataFrame(js.get("history", []))
        return hist if not hist.empty else None
    except Exception:
        return None

# ---------------------------
# HARDENING HELPERS
# ---------------------------

_POS_NUM_TO_LETTER = {1: "G", 2: "D", 3: "M", 4: "F"}
_POS_WORD_TO_LETTER = {
    "GK": "G", "GKP": "G", "Goalkeeper": "G",
    "DEF": "D", "Defender": "D",
    "MID": "M", "Midfielder": "M",
    "FWD": "F", "FW": "F", "Forward": "F"
}

def _to_position_letter(val) -> Optional[str]:
    """Map whatever we get to G/D/M/F."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)) and not pd.isna(val):
        return _POS_NUM_TO_LETTER.get(int(val))
    v = str(val).strip()
    if v in {"G", "D", "M", "F"}:
        return v
    return _POS_WORD_TO_LETTER.get(v)

def _series_to_short_team(s: pd.Series, teams_df: pd.DataFrame) -> pd.Series:
    """
    Convert a Series of team identifiers to short codes (e.g., 'MCI').
    Handles numeric ids or strings; falls back gracefully.
    """
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s):
        id2short = dict(zip(teams_df["id"], teams_df["short_name"]))
        return s.map(id2short)
    s = s.astype(str)
    mask_short = s.str.len() == 3
    out = s.where(mask_short)
    team_map = getattr(config, "TEAM_FULL_TO_SHORT", None)
    if isinstance(team_map, dict):
        out = out.fillna(s.map(team_map))
    return out.fillna(s)

def _enforce_rw_schema_fpl(fpl_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure FPL stats DF has: Player, Team (short), Position (G/D/M/F), Player_ID, Season_Points.
    """
    df = fpl_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Player_ID", "Season_Points"])

    idx = {c.lower(): c for c in df.columns}

    # --- Player ---
    if "player" in idx:
        df["Player"] = df[idx["player"]]
    elif "player_name" in idx:
        df["Player"] = df[idx["player_name"]]
    elif "web_name" in idx:
        df["Player"] = df[idx["web_name"]]
    elif "first_name" in idx and "second_name" in idx:
        df["Player"] = df[idx["first_name"]].astype(str) + " " + df[idx["second_name"]].astype(str)
    elif "name" in idx:
        df["Player"] = df[idx["name"]]
    else:
        df["Player"] = df.index.astype(str)

    # --- Team (short) ---
    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    elif "team_name" in idx:
        df["Team"] = _series_to_short_team(df[idx["team_name"]], teams_df)
    else:
        df["Team"] = np.nan

    # --- Position (G/D/M/F) ---
    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "position_abbrv" in idx:  # <— NEW: FPL stats has 'FWD/MID/DEF/GK'
        df["Position"] = df[idx["position_abbrv"]].map(_to_position_letter)
    elif "element_type" in idx:
        df["Position"] = df[idx["element_type"]].map(_to_position_letter)
    elif "pos" in idx:
        df["Position"] = df[idx["pos"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    # --- Player_ID ---
    if "player_id" in idx:
        df["Player_ID"] = pd.to_numeric(df[idx["player_id"]], errors="coerce")
    elif "id" in idx:
        df["Player_ID"] = pd.to_numeric(df[idx["id"]], errors="coerce")
    else:
        df["Player_ID"] = np.nan

    # --- Season_Points ---
    if "season_points" in idx:
        df["Season_Points"] = pd.to_numeric(df[idx["season_points"]], errors="coerce")
    elif "total_points" in idx:
        df["Season_Points"] = pd.to_numeric(df[idx["total_points"]], errors="coerce")
    else:
        df["Season_Points"] = 0

    df["Team"] = df["Team"].astype(str)
    df["Position"] = df["Position"].astype(str)

    keep = ["Player", "Team", "Position", "Player_ID", "Season_Points"]
    extras = [c for c in df.columns if c not in keep]
    return df[keep + extras]

def _enforce_rw_schema_proj(proj_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure projections DF has Player, Team (short), Position (G/D/M/F), Points.
    """
    df = proj_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])

    idx = {c.lower(): c for c in df.columns}

    # Player
    df["Player"] = df[idx["player"]] if "player" in idx else (
        df[idx["name"]] if "name" in idx else df.index.astype(str)
    )

    # Team
    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    else:
        df["Team"] = np.nan

    # Position
    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "pos" in idx:
        df["Position"] = df[idx["pos"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    # Points
    df["Points"] = pd.to_numeric(df[idx["points"]], errors="coerce") if "points" in idx else np.nan

    return df

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

def _avg_fdr_for_team(short_name: str, current_gw: int, n_weeks: int) -> Optional[float]:
    """Average FDR over next n_weeks for a team (short_name)."""
    team_id = _team_id_by_short(short_name)
    if team_id is None:
        return None

    fixtures = _load_future_fixtures()
    fixtures = fixtures.dropna(subset=["event"])
    fixtures["event"] = fixtures["event"].astype(int)

    future_slice = fixtures[(fixtures["event"] >= current_gw) &
                            (fixtures["event"] < current_gw + n_weeks)].copy()
    if future_slice.empty:
        return None

    def _row_team_fdr(row):
        if row.get("team_h") == team_id:
            return row.get("team_h_difficulty")
        if row.get("team_a") == team_id:
            return row.get("team_a_difficulty")
        return None

    future_slice["team_fdr"] = future_slice.apply(_row_team_fdr, axis=1)
    d = pd.to_numeric(future_slice["team_fdr"], errors="coerce").dropna()
    return float(d.mean()) if not d.empty else None

def _avg_form_last_n(player_id: int, last_n: int = config.FORM_LOOKBACK_WEEKS) -> Optional[float]:
    """Average of last N GWs' points for a player using element-summary history."""
    hist = _element_history_df(player_id)
    if hist is None or hist.empty:
        return None
    gw_col = "round" if "round" in hist.columns else "event"
    pts_col = "total_points" if "total_points" in hist.columns else "points"

    try:
        hist = hist.sort_values(gw_col)
        tail = hist.tail(last_n)
        vals = pd.to_numeric(tail[pts_col], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else None
    except Exception:
        return None

def _min_max_norm(series: pd.Series) -> pd.Series:
    """Min-max to [0,1], safe for constants/NaN."""
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - lo) / (hi - lo)

# ---------------------------
# OWNERSHIP / AVAILABLES
# ---------------------------

def _flatten_owned_names(league_ownership: Dict[int, Dict[str, Any]]) -> List[str]:
    """
    Supports structures like:
      { team_id: { 'team_name': str, 'players': {'G':[...],'D':[...],'M':[...],'F':[...]} } }
    or { team_id: { 'team_name': str, 'players': [<names>...] } }
    or even { team_id: { 'G':[...],'D':[...],'M':[...],'F':[...]} }  (older variants)
    Falls back to scanning any nested iterables of strings.
    """
    out: List[str] = []

    def _extend_from(obj):
        if obj is None:
            return
        # dict of positions -> list of names
        if isinstance(obj, dict):
            for v in obj.values():
                _extend_from(v)
        # iterable of names
        elif isinstance(obj, (list, set, tuple)):
            for x in obj:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, (list, set, tuple, dict)):
                    _extend_from(x)
        # single string
        elif isinstance(obj, str):
            out.append(obj)

    for _, team_blob in league_ownership.items():
        if not isinstance(team_blob, dict):
            continue
        # primary path
        if "players" in team_blob:
            _extend_from(team_blob["players"])
            continue
        # fallback: maybe positions live at top level
        pos_like = {k: v for k, v in team_blob.items() if k in {"G", "D", "M", "F"}}
        if pos_like:
            _extend_from(pos_like)
            continue
        # absolute fallback: scan everything in blob
        _extend_from(team_blob)

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def _prepare_proj_for_merge(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce projections to the exact schema the fuzzy merge expects:
    ['Player','Team','Matchup','Position','Points','Pos Rank'].

    - Renames common variants
    - Creates missing columns with safe defaults
    - Returns only the expected columns (and correct dtypes)
    """
    df = df_in.copy()

    # Rename common variants -> canonical
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("player", "name", "player_name"):         rename_map[c] = "Player"
        elif lc in ("team", "team_short", "teamname", "team_name"): rename_map[c] = "Team"
        elif lc in ("matchup", "fixture", "opp", "opponent"):       rename_map[c] = "Matchup"
        elif lc in ("position", "pos"):                           rename_map[c] = "Position"
        elif lc in ("points", "point", "proj", "projection"):       rename_map[c] = "Points"
        elif lc in ("pos rank", "pos_rank", "position rank", "position_rank"): rename_map[c] = "Pos Rank"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["Player", "Team", "Matchup", "Position", "Points", "Pos Rank"]
    defaults = {"Player": None, "Team": None, "Matchup": "", "Position": None, "Points": np.nan, "Pos Rank": "NA"}
    for k in required:
        if k not in df.columns:
            df[k] = defaults[k]

    # Dtypes
    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")

    # Only the columns used by the merge helper
    return df[required]

def _available_from_projections(
    projections_df: pd.DataFrame,
    fpl_player_statistics_df: pd.DataFrame,
    league_ownership: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Anti-join to remove league-owned players using your fuzzy merge helper.
    If ownership is empty/unavailable (preseason, API issues), return normalized projections unchanged.
    """
    if projections_df is None or projections_df.empty:
        raise ValueError("projections_df is empty or None.")

    # Normalize projections to the exact schema the fuzzy merge expects
    proj_for_merge = _prepare_proj_for_merge(projections_df)

    # 1) Try to build owned names
    owned_names = _flatten_owned_names(league_ownership)
    if len(owned_names) == 0:
        # Nothing owned -> everyone available
        st.info("⚠️ No ownership data detected (preseason or API). Treating all players as available.")
        return proj_for_merge

    # 2) Enrich owned with Team/Position (for better fuzzy accuracy)
    fpl_cols = {c.lower(): c for c in fpl_player_statistics_df.columns}
    need = {"player", "team", "position"}
    miss = need - set(fpl_cols.keys())
    if miss:
        raise ValueError(
            "fpl_player_statistics_df must contain ['Player','Team','Position'] after normalization. "
            f"Missing (case-insensitive): {sorted(miss)}. "
            f"Got: {list(fpl_player_statistics_df.columns)}"
        )

    owned_fpl = pd.DataFrame({"Player": owned_names}).merge(
        fpl_player_statistics_df[
            [fpl_cols["player"], fpl_cols["team"], fpl_cols["position"]]
        ].rename(columns={
            fpl_cols["player"]:   "Player",
            fpl_cols["team"]:     "Team",
            fpl_cols["position"]: "Position"
        }),
        on="Player", how="left"
    )

    # 3) Fuzzy map FPL -> RotoWire canonical names
    try:
        mapped_owned = merge_fpl_players_and_projections(owned_fpl, proj_for_merge)
    except Exception as e:
        raise RuntimeError(
            "merge_fpl_players_and_projections failed during ownership mapping. "
            f"owned_fpl cols={list(owned_fpl.columns)} projections_df cols={list(proj_for_merge.columns)} "
            f"Error: {e}"
        )

    # 4) Anti-join
    owned_rw_names = set(mapped_owned["Player"].dropna().unique().tolist())
    avail = proj_for_merge[~proj_for_merge["Player"].isin(owned_rw_names)].copy()
    return avail


# ---------------------------
# DRAFT CAPITAL (OPTIONAL)
# ---------------------------

@st.cache_data(show_spinner=False)
def _draft_picks_df() -> Optional[pd.DataFrame]:
    """Return draft picks as a DataFrame with columns: ['pick','player_id','team_id','team_name'] or None."""
    try:
        url = f"https://draft.premierleague.com/api/draft/{config.FPL_DRAFT_LEAGUE_ID}/choices"
        js = requests.get(url, timeout=30).json()
        choices = js.get("choices", [])
        if not choices:
            return None
        out = []
        for c in choices:
            out.append({
                "pick": c.get("pick"),
                "player_id": c.get("element"),
                "team_id": c.get("entry"),
                "team_name": c.get("entry_name")
            })
        df = pd.DataFrame(out).dropna(subset=["pick"])
        df["pick"] = pd.to_numeric(df["pick"], errors="coerce")
        return df
    except Exception:
        return None

# ---------------------------
# SCORING
# ---------------------------

def _ensure_team_col(df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a single 'Team' column exists on df.
    - Prefer existing 'Team' if present.
    - If only Team_x/Team_y exist, coalesce.
    - Else, pull 'Team' from fpl_stats by ['Player','Position'].
    """
    base = df.copy()

    # If already present, try to coalesce with any *_y leftovers
    if "Team" in base.columns:
        if "Team_y" in base.columns:
            base["Team"] = base["Team"].fillna(base["Team_y"])
        if "Team_x" in base.columns:
            base["Team"] = base["Team"].fillna(base["Team_x"])
    else:
        # No 'Team' column — try to derive
        src = None
        if "Team_x" in base.columns or "Team_y" in base.columns:
            base["Team"] = base.get("Team_x")
            base["Team"] = base["Team"].fillna(base.get("Team_y"))
        else:
            # Join from fpl_stats by Player+Position
            try:
                add = (fpl_stats[["Player", "Position", "Team"]]
                       .drop_duplicates())
                base = base.merge(add, on=["Player", "Position"], how="left", suffixes=("", "_from_stats"))
                if "Team_from_stats" in base.columns and "Team" not in base.columns:
                    base.rename(columns={"Team_from_stats":"Team"}, inplace=True)
            except Exception:
                base["Team"] = np.nan

    # Clean up any leftovers
    for col in ("Team_x", "Team_y", "Team_from_stats"):
        if col in base.columns:
            base.drop(columns=[col], inplace=True)

    # Ensure it's stringy (FDR call expects a short code string)
    base["Team"] = base["Team"].astype(str)
    return base

def _align_roster_player_names_to_projections(
    roster_df: pd.DataFrame,
    projections_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Replace FPL-style names on the roster with the RotoWire canonical names,
    preserving row order, using your merge_fpl_players_and_projections().
    """
    try:
        # prepare inputs for the fuzzy function
        for_merge = roster_df[["Player", "Team", "Position"]].copy()
        norm_proj = _prepare_proj_for_merge(projections_df)  # already in this file

        mapped = merge_fpl_players_and_projections(for_merge, norm_proj)
        # mapped['Player'] is the canonical (RotoWire) name in the same order as input
        canon = mapped["Player"].reset_index(drop=True)

        out = roster_df.copy()
        out["Player"] = canon
        return out
    except Exception as e:
        st.warning(f"⚠️ Name alignment to projections failed; keeping FPL names. Reason: {e}")
        return roster_df

def _add_fdr_and_form(
    df: pd.DataFrame,
    fpl_player_statistics_df: pd.DataFrame,
    current_gw: int,
    weeks: int
) -> pd.DataFrame:
    """
    Join AvgFDRNextN and Form onto df.

    - Requires df to have Player/Team/Position (will create if missing).
    - Merges in Player_ID, then computes Form via element-summary.
    - Fallback chain for Form: element-summary -> FPL 'form' -> 'points_per_game' -> 0.
    """
    base = df.copy()
    # Ensure join keys exist
    for col in ("Player", "Team", "Position"):
        if col not in base.columns:
            base[col] = np.nan

    # Safely select merge cols from stats
    stats = fpl_player_statistics_df.copy()
    for col in ("Player", "Team", "Position", "Player_ID", "form", "points_per_game"):
        if col not in stats.columns:
            stats[col] = np.nan

    # Merge in Player_ID + FPL fallback stats
    base = base.merge(
        stats[["Player", "Team", "Position", "Player_ID", "form", "points_per_game"]],
        on=["Player", "Team", "Position"],
        how="left"
    )

    # Ensure Player_ID exists (for downstream)
    if "Player_ID" not in base.columns:
        base["Player_ID"] = np.nan

    # Compute Avg FDR next N GWs
    base["AvgFDRNextN"] = base["Team"].apply(lambda t: _avg_fdr_for_team(str(t), current_gw, weeks))
    base["AvgFDRNextN"] = pd.to_numeric(base["AvgFDRNextN"], errors="coerce")

    # Robust Form calculation with fallbacks
    def _safe_form(pid, fallback_form, fallback_ppg):
        # element-summary average of last N
        val = None
        if pd.notna(pid):
            try:
                val = _avg_form_last_n(int(pid), config.FORM_LOOKBACK_WEEKS)
            except Exception:
                val = None
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_form, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = pd.to_numeric(fallback_ppg, errors="coerce")
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = 0.0
        return float(val)

    base["Form"] = base.apply(
        lambda r: _safe_form(r.get("Player_ID"), r.get("form"), r.get("points_per_game")),
        axis=1
    )
    base["Form"] = pd.to_numeric(base["Form"], errors="coerce").fillna(0.0)

    return base

def _compute_waiver_score(df: pd.DataFrame,
                          w_proj: float,
                          w_form: float,
                          w_fdr: float) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Proj_norm"] = _min_max_norm(tmp["Points"]).fillna(0.5)
    tmp["Form_norm"] = _min_max_norm(tmp["Form"]).fillna(0.5)

    tmp["FDREase"] = 6 - pd.to_numeric(tmp["AvgFDRNextN"], errors="coerce")
    tmp["FDREase_norm"] = _min_max_norm(tmp["FDREase"]).fillna(0.5)

    denom = max(w_proj + w_form + w_fdr, 1e-9)
    tmp["Waiver Score"] = (
        w_proj * tmp["Proj_norm"] +
        w_form * tmp["Form_norm"] +
        w_fdr  * tmp["FDREase_norm"]
    ) / denom

    return tmp.drop(columns=["Proj_norm", "Form_norm", "FDREase", "FDREase_norm"])

def _compute_keep_score(roster_df: pd.DataFrame,
                        draft_df: Optional[pd.DataFrame],
                        w_season: float,
                        w_form: float,
                        w_fdr: float,
                        w_draft: float) -> pd.DataFrame:
    df = roster_df.copy()

    # Coerce numerics
    df["Season_Points"] = pd.to_numeric(df.get("Season_Points"), errors="coerce")
    df["Form"] = pd.to_numeric(df.get("Form"), errors="coerce")
    df["AvgFDRNextN"] = pd.to_numeric(df.get("AvgFDRNextN"), errors="coerce")

    # Draft pick mapping (optional)
    if draft_df is not None and "Player_ID" in df.columns:
        dd = draft_df.dropna(subset=["player_id", "pick"]).copy()
        dd["player_id"] = pd.to_numeric(dd["player_id"], errors="coerce")
        dd["pick"] = pd.to_numeric(dd["pick"], errors="coerce")
        best_pick = dd.groupby("player_id", as_index=False)["pick"].min()
        df = df.merge(best_pick.rename(columns={"player_id": "Player_ID", "pick": "DraftPick"}),
                      on="Player_ID", how="left")
    else:
        df["DraftPick"] = np.nan

    # Normalized components (fill NaN -> 0.5 neutral)
    df["Season_norm"] = _min_max_norm(df["Season_Points"]).fillna(0.5)
    df["Form_norm"]   = _min_max_norm(df["Form"]).fillna(0.5)
    df["FDREase"]     = 6 - df["AvgFDRNextN"]
    df["FDREase_norm"]= _min_max_norm(df["FDREase"]).fillna(0.5)

    # Draft norm
    if df["DraftPick"].notna().any():
        max_pick = pd.to_numeric(df["DraftPick"], errors="coerce").max()
        df["DraftValueRaw"] = max_pick - pd.to_numeric(df["DraftPick"], errors="coerce")
        df["Draft_norm"] = _min_max_norm(df["DraftValueRaw"]).fillna(0.5)
    else:
        df["Draft_norm"] = 0.5

    denom = max(w_season + w_form + w_fdr + w_draft, 1e-9)
    df["Keep Score"] = (
        w_season * df["Season_norm"] +
        w_form   * df["Form_norm"] +
        w_fdr    * df["FDREase_norm"] +
        w_draft  * df["Draft_norm"]
    ) / denom

    drop_cols = ["Season_norm", "Form_norm", "FDREase", "FDREase_norm", "Draft_norm", "DraftValueRaw"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols)

# ---------------------------
# AZURE OPENAI (OPTIONAL)
# ---------------------------

def _azure_suggest_moves(available_df: pd.DataFrame,
                         roster_df: pd.DataFrame,
                         top_k: int = 5) -> Optional[str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not (endpoint and api_key and deployment):
        return None

    try:
        client = OpenAI(
            base_url=f"{endpoint}/openai/deployments/{deployment}",
            api_key=api_key,
        )
        top_adds = available_df.head(top_k)[["Player", "Team", "Position", "Points", "Form", "AvgFDRNextN", "Waiver Score"]]
        top_drops = roster_df.nsmallest(top_k, "Keep Score")[["Player", "Team", "Position", "Season_Points", "Form", "AvgFDRNextN", "Keep Score"]]

        prompt = (
            "You are an FPL Draft assistant. Based on Waiver Score (adds) and Keep Score (drops), "
            "suggest the best 3 add/drop pairs. Prefer players with higher Waiver Score to add, and "
            "players with lower Keep Score to drop. Keep positional balance where possible.\n\n"
            f"Top Adds:\n{top_adds.to_string(index=False)}\n\n"
            f"Potential Drops:\n{top_drops.to_string(index=False)}\n\n"
            "Return 3 bullet points like: "
            "'Add <Player, Team, Pos> for <Player, Team, Pos> — quick rationale.'"
        )

        rsp = client.chat.completions.create(
            model=deployment,
            messages=[{"role":"user", "content":prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        text = rsp.choices[0].message.content.strip()
        return text
    except Exception as e:
        return f"(Azure suggestion unavailable: {e})"

# ---------------------------
# PAGE
# ---------------------------

def show_waiver_wire_page():
    st.header("🔁 Waiver Wire Assistant")
    st.caption("Blends weekly projections, recent form (last 3 GWs), and upcoming fixture difficulty (FDR).")

    # Controls
    with st.expander("Filters & Weights", expanded=True):
        colA, colB, colC = st.columns(3)
        pos_filter = colA.multiselect("Positions", ["G", "D", "M", "F"], default=["G", "D", "M", "F"])
        lookahead = int(colB.number_input("Upcoming GWs to average FDR", min_value=1, max_value=8, value=config.UPCOMING_WEEKS_DEFAULT))
        form_weeks = int(colC.number_input("Form lookback GWs", min_value=1, max_value=5, value=config.FORM_LOOKBACK_WEEKS))

        st.markdown("**Waiver Score Weights** (available player ranking):")
        wcol1, wcol2, wcol3 = st.columns(3)
        w_proj = float(wcol1.slider("Projected Points", 0.0, 1.0, 0.5, 0.05, key="w_proj"))
        w_form = float(wcol2.slider("Recent Form (adds)", 0.0, 1.0, 0.3, 0.05, key="w_form"))
        w_fdr  = float(wcol3.slider("Fixture Ease (adds)", 0.0, 1.0, 0.2, 0.05, key="w_fdr"))

        st.markdown("**Keep Score Weights** (roster protection):")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        k_season = float(kcol1.slider("Season Points (keep)", 0.0, 1.0, 0.5, 0.05, key="k_season"))
        k_form   = float(kcol2.slider("Recent Form (keep)", 0.0, 1.0, 0.2, 0.05, key="k_form"))
        k_fdr    = float(kcol3.slider("Fixture Ease (keep)", 0.0, 1.0, 0.2, 0.05, key="k_fdr"))
        k_draft  = float(kcol4.slider("Draft Capital (keep)", 0.0, 1.0, 0.1, 0.05, key="k_draft"))

    # Load ownership
    try:
        ownership = get_league_player_ownership(config.FPL_DRAFT_LEAGUE_ID)
    except Exception as e:
        st.error(f"Unable to load league ownership: {e}")
        ownership = {}

    # Quick visibility
    owned_sample = _flatten_owned_names(ownership)
    st.caption(f"Ownership snapshot: {len(owned_sample)} owned player names found.")
    if len(owned_sample) == 0:
        st.info(
            "No owned players detected. If the season hasn't started or the API returns no element owners, this is expected.")

    draft_df = _draft_picks_df()

    # Projections & FPL stats
    projections_raw = get_rotowire_player_projections(config.ROTOWIRE_URL)
    fpl_stats_raw = pull_fpl_player_stats()

    # Normalize both to a unified schema (and enforce minimal columns)
    teams_df = _bootstrap_teams_df()
    try:
        fpl_stats_norm = normalize_fpl_players_to_rotowire_schema(fpl_stats_raw, teams_df=teams_df)
    except Exception:
        fpl_stats_norm = fpl_stats_raw
    try:
        projections_norm = normalize_rotowire_players(projections_raw)
    except Exception:
        projections_norm = projections_raw

    fpl_stats = _enforce_rw_schema_fpl(fpl_stats_norm, teams_df)
    proj = _enforce_rw_schema_proj(projections_norm, teams_df)

    # Guard-rails
    req = {"Player", "Team", "Position"}
    miss_stats = req - set(fpl_stats.columns)
    if miss_stats:
        st.error(f"Failed to compute available players: fpl_player_statistics_df missing {sorted(miss_stats)} after normalization.")
        st.stop()

    miss_proj = req - set(proj.columns)
    if miss_proj:
        st.error(f"Projections table missing {sorted(miss_proj)} after normalization.")
        st.stop()

    if "Season_Points" not in fpl_stats.columns:
        fpl_stats["Season_Points"] = 0

    # Apply position filter to projections
    if pos_filter:
        proj = proj[proj["Position"].isin(pos_filter)]

    # Current GW
    try:
        current_gw = int(get_current_gameweek() or 1)
    except Exception:
        current_gw = 1

    # AVAILABLE PLAYERS (anti-join using your robust fuzzy merge)
    try:
        avail = _available_from_projections(proj, fpl_stats, ownership)
    except Exception as e:
        st.error(f"Failed to compute available players: {e}")
        # Diagnostics
        try:
            st.caption("**Projections columns:** " + ", ".join(list(proj.columns)))
            st.dataframe(proj.head(5), use_container_width=True)
        except Exception:
            pass
        try:
            owned_names = _flatten_owned_names(ownership)
            st.caption(f"**Sample owned (FPL) names (first 10 of {len(owned_names)}):** {owned_names[:10]}")
            st.caption("**FPL stats columns:** " + ", ".join(list(fpl_stats.columns)))
        except Exception:
            pass
        st.stop()

    # Add FDR & Form, compute Waiver Score
    global FORM_LOOKBACK_WEEKS
    FORM_LOOKBACK_WEEKS = form_weeks
    avail = _add_fdr_and_form(avail, fpl_stats, current_gw, lookahead)
    avail = _compute_waiver_score(avail, w_proj, w_form, w_fdr)
    avail = avail.sort_values("Waiver Score", ascending=False).reset_index(drop=True)

    _display_avail = avail.copy()
    for col in _display_avail.select_dtypes(include=[np.number]).columns:
        _display_avail[col] = _display_avail[col].round(2)

    st.subheader("Available Players (ranked)")
    st.dataframe(
        _display_avail[["Player", "Team", "Position", "Points", "Form", "AvgFDRNextN", "Waiver Score"]],
        use_container_width=True
    )

    # --- Team picker (dropdown) ---
    team_options = []
    for tid, blob in ownership.items():
        tname = blob.get("team_name", f"Team {tid}")
        team_options.append((int(tid), f"{tname} ({tid})"))

    # Sort by label
    team_options = sorted(team_options, key=lambda x: x[1].lower())

    # Default to config.FPL_DRAFT_TEAM_ID if available
    default_tid = getattr(config, "FPL_DRAFT_TEAM_ID", None)
    default_idx = 0
    if default_tid is not None:
        for i, (tid, label) in enumerate(team_options):
            if str(tid) == str(default_tid):
                default_idx = i
                break

    choice_label = st.selectbox(
        "Your Team",
        options=[label for _, label in team_options],
        index=default_idx
    )
    my_team_id = next(tid for tid, label in team_options if label == choice_label)
    my_team = ownership.get(my_team_id, {})

    # Build MY ROSTER table (existing code)
    my_players = []
    for pos, names in my_team.get("players", {}).items():
        for nm in names:
            my_players.append({"Player": nm, "Team": None, "Position": pos})
    my_roster = pd.DataFrame(my_players)

    # Attach Team/Player_ID/Season_Points from FPL master (existing)
    my_roster = my_roster.merge(
        fpl_stats[["Player", "Team", "Position", "Player_ID", "Season_Points"]],
        on=["Player", "Position"], how="left"
    )

    # Ensure Team present for exact joins and align roster names to RotoWire canonical names
    my_roster = _ensure_team_col(my_roster, fpl_stats)
    my_roster = _align_roster_player_names_to_projections(my_roster, proj)

    # Add next-GW projected points (match on canonical Player/Team/Position)
    proj_points_df = proj[["Player", "Team", "Position", "Points"]].rename(
        columns={"Points": "Projected_Points"}
    )
    my_roster = my_roster.merge(proj_points_df, on=["Player", "Team", "Position"], how="left")

    # Ensure any remaining missing Player_IDs are backfilled one more time
    my_roster = _backfill_player_ids(my_roster, fpl_stats)

    # Recompute form/FDR with any new IDs (optional but safe)
    my_roster = _add_fdr_and_form(my_roster, fpl_stats, current_gw, lookahead)

    # Compute Keep Score (unchanged)
    my_roster = _compute_keep_score(my_roster, draft_df, k_season, k_form, k_fdr, k_draft)
    my_roster = my_roster.sort_values("Keep Score", ascending=False).reset_index(drop=True)

    # Round numeric columns to 2 decimals for display
    _display_roster = my_roster.copy()
    for col in _display_roster.select_dtypes(include=[np.number]).columns:
        _display_roster[col] = _display_roster[col].round(2)

    st.subheader(f"My Roster — {my_team.get('team_name', '(unknown)')}")
    st.dataframe(
        _display_roster[
            ["Player", "Team", "Position", "Season_Points", "Projected_Points", "Form", "AvgFDRNextN", "Keep Score"]],
        use_container_width=True
    )

    # Optional Azure suggestions
    # if st.checkbox("Ask Azure AI to suggest moves", value=False):
    #     suggestion = _azure_suggest_moves(avail, my_roster, top_k=5)
    #     if suggestion:
    #         st.markdown("**Azure AI Suggestions**")
    #         st.write(suggestion)
    #     else:
    #         st.info("Azure OpenAI not configured or request failed.")
