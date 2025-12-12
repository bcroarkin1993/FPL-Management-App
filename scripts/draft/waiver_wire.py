import os
import requests
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List

import config
from openai import OpenAI

# ------------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------------
# Import API fetchers from the common API module
from scripts.common.api import (
    get_league_player_ownership,
    get_rotowire_player_projections,
    pull_fpl_player_stats,
    get_current_gameweek
)
# Import the shared fuzzy merge utility
from scripts.common.utils import merge_fpl_players_and_projections


# ------------------------------------------------------------------
# LOCAL HELPERS & NORMALIZATION
# (Kept local to ensure this script is self-contained)
# ------------------------------------------------------------------

def _bootstrap_teams_df() -> pd.DataFrame:
    """Fetches teams from bootstrap-static and returns a DataFrame [id, name, short_name]."""
    try:
        url = "https://draft.premierleague.com/api/bootstrap-static"
        data = requests.get(url, timeout=30).json()
        teams = data.get("teams", [])
        return pd.DataFrame(teams)[["id", "name", "short_name"]]
    except Exception:
        return pd.DataFrame(columns=["id", "name", "short_name"])


def normalize_rotowire_players(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes Rotowire projection columns."""
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])

    # Map common column variations
    rename_map = {
        "player_name": "Player", "name": "Player",
        "team_short": "Team", "team": "Team",
        "pos": "Position", "position": "Position",
        "points": "Points", "proj_points": "Points"
    }

    out = df.rename(columns=rename_map)
    # Ensure columns exist
    for col in ["Player", "Team", "Position", "Points"]:
        if col not in out.columns:
            out[col] = np.nan

    return out[["Player", "Team", "Position", "Points"]]


def normalize_fpl_players_to_rotowire_schema(fpl_df: pd.DataFrame, teams_df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes FPL Stats columns to match Rotowire schema."""
    if fpl_df.empty:
        return pd.DataFrame(
            columns=["Player", "Team", "Position", "Player_ID", "Season_Points", "form", "points_per_game"])

    out = fpl_df.copy()

    # map team ID to short name
    if "team" in out.columns and not teams_df.empty:
        id_to_short = dict(zip(teams_df["id"], teams_df["short_name"]))
        out["Team"] = out["team"].map(id_to_short)

    # map element_type to Position (1=G, 2=D, 3=M, 4=F)
    type_map = {1: "G", 2: "D", 3: "M", 4: "F"}
    if "element_type" in out.columns:
        out["Position"] = out["element_type"].map(type_map)

    # Rename common columns
    rename_map = {
        "web_name": "Player",
        "id": "Player_ID",
        "total_points": "Season_Points"
    }
    out = out.rename(columns=rename_map)

    cols_needed = ["Player", "Team", "Position", "Player_ID", "Season_Points", "form", "points_per_game"]
    for c in cols_needed:
        if c not in out.columns:
            out[c] = np.nan

    return out[cols_needed]


def _backfill_player_ids(df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Tries to fill missing Player_IDs by matching names with the master stats DF."""
    if "Player_ID" not in df.columns:
        df["Player_ID"] = np.nan

    mask_missing = df["Player_ID"].isna()
    if not mask_missing.any():
        return df

    # Create lookup
    lookup = stats_df.set_index("Player")["Player_ID"].to_dict()

    # Apply
    df.loc[mask_missing, "Player_ID"] = df.loc[mask_missing, "Player"].map(lookup)
    return df


@st.cache_data(show_spinner=False)
def _load_future_fixtures() -> pd.DataFrame:
    """
    Returns future fixtures with difficulties.
    Columns: event, kickoff_time, team_h, team_a, team_h_difficulty, team_a_difficulty
    """
    url = "https://fantasy.premierleague.com/api/fixtures/?future=1"
    try:
        fx = requests.get(url, timeout=30).json()
        df = pd.DataFrame(fx)
        keep = ["event", "kickoff_time", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
        df = df[[c for c in keep if c in df.columns]].copy()
        return df
    except Exception:
        return pd.DataFrame()


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


# ------------------------------------------------------------------
# SCHEMA ENFORCERS & HELPERS
# ------------------------------------------------------------------

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
    """Convert a Series of team identifiers to short codes (e.g., 'MCI')."""
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
    """Ensure FPL stats DF has: Player, Team (short), Position, Player_ID, Season_Points."""
    df = fpl_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Player_ID", "Season_Points"])

    idx = {c.lower(): c for c in df.columns}

    # --- Player ---
    if "player" in idx:
        df["Player"] = df[idx["player"]]
    elif "web_name" in idx:
        df["Player"] = df[idx["web_name"]]
    elif "name" in idx:
        df["Player"] = df[idx["name"]]
    else:
        df["Player"] = df.index.astype(str)

    # --- Team ---
    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    else:
        df["Team"] = np.nan

    # --- Position ---
    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "element_type" in idx:
        df["Position"] = df[idx["element_type"]].map(_to_position_letter)
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
    """Ensure projections DF has Player, Team, Position, Points."""
    df = proj_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["Player", "Team", "Position", "Points"])

    idx = {c.lower(): c for c in df.columns}

    df["Player"] = df[idx["player"]] if "player" in idx else df.index.astype(str)

    if "team_short" in idx:
        df["Team"] = df[idx["team_short"]]
    elif "team" in idx:
        df["Team"] = _series_to_short_team(df[idx["team"]], teams_df)
    else:
        df["Team"] = np.nan

    if "position" in idx:
        df["Position"] = df[idx["position"]].map(_to_position_letter)
    elif "pos" in idx:
        df["Position"] = df[idx["pos"]].map(_to_position_letter)
    else:
        df["Position"] = np.nan

    df["Points"] = pd.to_numeric(df[idx["points"]], errors="coerce") if "points" in idx else np.nan

    return df


# ------------------------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------------------------

def _load_bootstrap_for_mapping() -> Dict[int, str]:
    """Helper to map Team ID to Short Name."""
    df = _bootstrap_teams_df()
    if df.empty: return {}
    return dict(zip(df["id"], df["short_name"]))


def _team_id_by_short(short_name: str) -> Optional[int]:
    """Reverse lookup: short_name -> team id."""
    df = _bootstrap_teams_df()
    if df.empty: return None
    m = dict(zip(df["short_name"], df["id"]))
    return m.get(short_name)


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


# ------------------------------------------------------------------
# OWNERSHIP & MATCHING LOGIC
# ------------------------------------------------------------------

def _flatten_owned_names(league_ownership: Dict[int, Dict[str, Any]]) -> List[str]:
    """Flatten ownership dict to list of names."""
    out: List[str] = []

    def _extend_from(obj):
        if obj is None: return
        if isinstance(obj, dict):
            for v in obj.values(): _extend_from(v)
        elif isinstance(obj, (list, set, tuple)):
            for x in obj:
                if isinstance(x, str):
                    out.append(x)
                elif isinstance(x, (list, set, tuple, dict)):
                    _extend_from(x)
        elif isinstance(obj, str):
            out.append(obj)

    for _, team_blob in league_ownership.items():
        if not isinstance(team_blob, dict): continue
        if "players" in team_blob:
            _extend_from(team_blob["players"])
            continue
        _extend_from(team_blob)

    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _prepare_proj_for_merge(df_in: pd.DataFrame) -> pd.DataFrame:
    """Prepare projections for fuzzy merge."""
    df = df_in.copy()
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in ("player", "name", "player_name"):
            rename_map[c] = "Player"
        elif lc in ("team", "team_short"):
            rename_map[c] = "Team"
        elif lc in ("position", "pos"):
            rename_map[c] = "Position"
        elif lc in ("points", "point", "proj"):
            rename_map[c] = "Points"

    if rename_map: df = df.rename(columns=rename_map)

    required = ["Player", "Team", "Matchup", "Position", "Points", "Pos Rank"]
    defaults = {"Player": None, "Team": None, "Matchup": "", "Position": None, "Points": np.nan, "Pos Rank": "NA"}
    for k in required:
        if k not in df.columns: df[k] = defaults[k]

    df["Points"] = pd.to_numeric(df["Points"], errors="coerce")
    return df[required]


def _available_from_projections(
        projections_df: pd.DataFrame,
        fpl_player_statistics_df: pd.DataFrame,
        league_ownership: Dict[int, Dict[str, Any]]
) -> pd.DataFrame:
    """Anti-join to remove league-owned players."""
    if projections_df is None or projections_df.empty:
        raise ValueError("projections_df is empty.")

    proj_for_merge = _prepare_proj_for_merge(projections_df)
    owned_names = _flatten_owned_names(league_ownership)

    if not owned_names:
        st.info("⚠️ No ownership data detected. Treating all as available.")
        return proj_for_merge

    # Enrich owned with Team/Pos for better matching
    fpl_cols = {c.lower(): c for c in fpl_player_statistics_df.columns}

    owned_fpl = pd.DataFrame({"Player": owned_names}).merge(
        fpl_player_statistics_df.rename(columns={
            fpl_cols.get("player", "Player"): "Player",
            fpl_cols.get("team", "Team"): "Team",
            fpl_cols.get("position", "Position"): "Position"
        }),
        on="Player", how="left"
    )

    try:
        mapped_owned = merge_fpl_players_and_projections(owned_fpl, proj_for_merge)
    except Exception as e:
        st.error(f"Merge error: {e}")
        return proj_for_merge

    owned_rw_names = set(mapped_owned["Player"].dropna().unique().tolist())
    avail = proj_for_merge[~proj_for_merge["Player"].isin(owned_rw_names)].copy()
    return avail


@st.cache_data(show_spinner=False)
def _draft_picks_df() -> Optional[pd.DataFrame]:
    """Return draft picks."""
    try:
        url = f"https://draft.premierleague.com/api/draft/{config.FPL_DRAFT_LEAGUE_ID}/choices"
        js = requests.get(url, timeout=30).json()
        choices = js.get("choices", [])
        if not choices: return None
        out = []
        for c in choices:
            out.append({
                "pick": c.get("pick"),
                "player_id": c.get("element"),
                "team_id": c.get("entry")
            })
        df = pd.DataFrame(out).dropna(subset=["pick"])
        return df
    except Exception:
        return None


# ------------------------------------------------------------------
# SCORING & LOGIC
# ------------------------------------------------------------------

def _ensure_team_col(df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """Ensure a single 'Team' column exists."""
    base = df.copy()
    if "Team" not in base.columns:
        if "Team_x" in base.columns:
            base["Team"] = base["Team_x"].fillna(base.get("Team_y", np.nan))
        else:
            try:
                add = fpl_stats[["Player", "Position", "Team"]].drop_duplicates()
                base = base.merge(add, on=["Player", "Position"], how="left", suffixes=("", "_stats"))
                if "Team_stats" in base.columns: base["Team"] = base["Team_stats"]
            except:
                base["Team"] = np.nan

    if "Team" in base.columns:
        base["Team"] = base["Team"].astype(str)
    return base


def _align_roster_player_names_to_projections(roster_df: pd.DataFrame, projections_df: pd.DataFrame) -> pd.DataFrame:
    """Align roster names to Rotowire canonical names."""
    try:
        for_merge = roster_df[["Player", "Team", "Position"]].copy()
        norm_proj = _prepare_proj_for_merge(projections_df)
        mapped = merge_fpl_players_and_projections(for_merge, norm_proj)

        out = roster_df.copy()
        out["Player"] = mapped["Player"].reset_index(drop=True)
        return out
    except:
        return roster_df


def _add_fdr_and_form(df: pd.DataFrame, fpl_player_statistics_df: pd.DataFrame, current_gw: int,
                      weeks: int) -> pd.DataFrame:
    """Join AvgFDRNextN and Form onto df."""
    base = df.copy()
    for col in ("Player", "Team", "Position"):
        if col not in base.columns: base[col] = np.nan

    stats = fpl_player_statistics_df.copy()
    for col in ("Player", "Team", "Position", "Player_ID", "form", "points_per_game"):
        if col not in stats.columns: stats[col] = np.nan

    base = base.merge(
        stats[["Player", "Team", "Position", "Player_ID", "form", "points_per_game"]],
        on=["Player", "Team", "Position"], how="left"
    )

    base["AvgFDRNextN"] = base["Team"].apply(lambda t: _avg_fdr_for_team(str(t), current_gw, weeks))
    base["AvgFDRNextN"] = pd.to_numeric(base["AvgFDRNextN"], errors="coerce")

    def _safe_form(pid, fallback_form, fallback_ppg):
        val = None
        if pd.notna(pid):
            try:
                val = _avg_form_last_n(int(pid), config.FORM_LOOKBACK_WEEKS)
            except:
                pass
        if val is None: val = pd.to_numeric(fallback_form, errors="coerce")
        if val is None: val = pd.to_numeric(fallback_ppg, errors="coerce")
        return float(val) if val is not None else 0.0

    base["Form"] = base.apply(lambda r: _safe_form(r.get("Player_ID"), r.get("form"), r.get("points_per_game")), axis=1)
    base["Form"] = pd.to_numeric(base["Form"], errors="coerce").fillna(0.0)
    return base


def _compute_waiver_score(df: pd.DataFrame, w_proj: float, w_form: float, w_fdr: float) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Proj_norm"] = _min_max_norm(tmp["Points"]).fillna(0.5)
    tmp["Form_norm"] = _min_max_norm(tmp["Form"]).fillna(0.5)
    tmp["FDREase"] = 6 - pd.to_numeric(tmp["AvgFDRNextN"], errors="coerce")
    tmp["FDREase_norm"] = _min_max_norm(tmp["FDREase"]).fillna(0.5)

    denom = max(w_proj + w_form + w_fdr, 1e-9)
    tmp["Waiver Score"] = (w_proj * tmp["Proj_norm"] + w_form * tmp["Form_norm"] + w_fdr * tmp["FDREase_norm"]) / denom
    return tmp


def _compute_keep_score(roster_df: pd.DataFrame, draft_df: Optional[pd.DataFrame], w_season: float, w_form: float,
                        w_fdr: float, w_draft: float) -> pd.DataFrame:
    df = roster_df.copy()
    df["Season_Points"] = pd.to_numeric(df.get("Season_Points"), errors="coerce")
    df["Form"] = pd.to_numeric(df.get("Form"), errors="coerce")
    df["AvgFDRNextN"] = pd.to_numeric(df.get("AvgFDRNextN"), errors="coerce")

    if draft_df is not None and "Player_ID" in df.columns:
        dd = draft_df.dropna(subset=["player_id", "pick"]).copy()
        dd["player_id"] = pd.to_numeric(dd["player_id"], errors="coerce")
        dd["pick"] = pd.to_numeric(dd["pick"], errors="coerce")
        best_pick = dd.groupby("player_id", as_index=False)["pick"].min()
        df = df.merge(best_pick.rename(columns={"player_id": "Player_ID", "pick": "DraftPick"}), on="Player_ID",
                      how="left")
    else:
        df["DraftPick"] = np.nan

    df["Season_norm"] = _min_max_norm(df["Season_Points"]).fillna(0.5)
    df["Form_norm"] = _min_max_norm(df["Form"]).fillna(0.5)
    df["FDREase"] = 6 - df["AvgFDRNextN"]
    df["FDREase_norm"] = _min_max_norm(df["FDREase"]).fillna(0.5)

    if df["DraftPick"].notna().any():
        max_pick = pd.to_numeric(df["DraftPick"], errors="coerce").max()
        df["DraftValueRaw"] = max_pick - pd.to_numeric(df["DraftPick"], errors="coerce")
        df["Draft_norm"] = _min_max_norm(df["DraftValueRaw"]).fillna(0.5)
    else:
        df["Draft_norm"] = 0.5

    denom = max(w_season + w_form + w_fdr + w_draft, 1e-9)
    df["Keep Score"] = (
                               w_season * df["Season_norm"] + w_form * df["Form_norm"] +
                               w_fdr * df["FDREase_norm"] + w_draft * df["Draft_norm"]
                       ) / denom
    return df


# ------------------------------------------------------------------
# MAIN PAGE
# ------------------------------------------------------------------

def show_waiver_wire_page():
    st.header("🔁 Waiver Wire Assistant")
    st.caption("Blends weekly projections, recent form (last 3 GWs), and upcoming fixture difficulty (FDR).")

    with st.expander("Filters & Weights", expanded=True):
        colA, colB, colC = st.columns(3)
        pos_filter = colA.multiselect("Positions", ["G", "D", "M", "F"], default=["G", "D", "M", "F"])
        lookahead = int(colB.number_input("Upcoming GWs (FDR)", 1, 8, config.UPCOMING_WEEKS_DEFAULT))
        form_weeks = int(colC.number_input("Form Lookback", 1, 5, config.FORM_LOOKBACK_WEEKS))

        st.markdown("**Waiver Score Weights** (Adds):")
        wcol1, wcol2, wcol3 = st.columns(3)
        w_proj = float(wcol1.slider("Projections", 0.0, 1.0, 0.5, 0.05, key="w_proj"))
        w_form = float(wcol2.slider("Form", 0.0, 1.0, 0.3, 0.05, key="w_form"))
        w_fdr = float(wcol3.slider("FDR", 0.0, 1.0, 0.2, 0.05, key="w_fdr"))

        st.markdown("**Keep Score Weights** (Drops):")
        kcol1, kcol2, kcol3, kcol4 = st.columns(4)
        k_season = float(kcol1.slider("Season Pts", 0.0, 1.0, 0.5, 0.05, key="k_season"))
        k_form = float(kcol2.slider("Form", 0.0, 1.0, 0.2, 0.05, key="k_form"))
        k_fdr = float(kcol3.slider("FDR", 0.0, 1.0, 0.2, 0.05, key="k_fdr"))
        k_draft = float(kcol4.slider("Draft Capital", 0.0, 1.0, 0.1, 0.05, key="k_draft"))

    try:
        ownership = get_league_player_ownership(config.FPL_DRAFT_LEAGUE_ID)
    except Exception as e:
        st.error(f"Ownership load failed: {e}")
        ownership = {}

    draft_df = _draft_picks_df()
    projections_raw = get_rotowire_player_projections(config.ROTOWIRE_URL)
    fpl_stats_raw = pull_fpl_player_stats()

    teams_df = _bootstrap_teams_df()
    fpl_stats_norm = normalize_fpl_players_to_rotowire_schema(fpl_stats_raw, teams_df=teams_df)
    projections_norm = normalize_rotowire_players(projections_raw)

    fpl_stats = _enforce_rw_schema_fpl(fpl_stats_norm, teams_df)
    proj = _enforce_rw_schema_proj(projections_norm, teams_df)

    if pos_filter:
        proj = proj[proj["Position"].isin(pos_filter)]

    current_gw = get_current_gameweek() or 1
    config.FORM_LOOKBACK_WEEKS = form_weeks

    # AVAILABLE
    try:
        avail = _available_from_projections(proj, fpl_stats, ownership)
    except Exception as e:
        st.error(f"Availability calc failed: {e}")
        st.stop()

    avail = _add_fdr_and_form(avail, fpl_stats, current_gw, lookahead)
    avail = _compute_waiver_score(avail, w_proj, w_form, w_fdr)
    avail = avail.sort_values("Waiver Score", ascending=False).reset_index(drop=True)

    st.subheader("Available Players (ranked)")
    st.dataframe(
        avail[["Player", "Team", "Position", "Points", "Form", "AvgFDRNextN", "Waiver Score"]].style.format("{:.2f}",
                                                                                                            subset=[
                                                                                                                "Points",
                                                                                                                "Form",
                                                                                                                "AvgFDRNextN",
                                                                                                                "Waiver Score"]),
        use_container_width=True)

    # MY ROSTER
    team_options = sorted([(int(tid), f"{blob.get('team_name')} ({tid})") for tid, blob in ownership.items()],
                          key=lambda x: x[1].lower())
    default_idx = next(
        (i for i, (tid, _) in enumerate(team_options) if str(tid) == str(getattr(config, "FPL_DRAFT_TEAM_ID", ""))), 0)

    if team_options:
        choice_label = st.selectbox("Your Team", [label for _, label in team_options], index=default_idx)
        my_team_id = next(tid for tid, label in team_options if label == choice_label)
        my_team = ownership.get(my_team_id, {})

        my_players = [{"Player": nm, "Team": None, "Position": pos} for pos, names in my_team.get("players", {}).items()
                      for nm in names]
        my_roster = pd.DataFrame(my_players)

        my_roster = my_roster.merge(fpl_stats[["Player", "Team", "Position", "Player_ID", "Season_Points"]],
                                    on=["Player", "Position"], how="left")
        my_roster = _ensure_team_col(my_roster, fpl_stats)
        my_roster = _align_roster_player_names_to_projections(my_roster, proj)

        my_roster = my_roster.merge(
            proj[["Player", "Team", "Position", "Points"]].rename(columns={"Points": "Projected_Points"}),
            on=["Player", "Team", "Position"], how="left")
        my_roster = _backfill_player_ids(my_roster, fpl_stats)
        my_roster = _add_fdr_and_form(my_roster, fpl_stats, current_gw, lookahead)
        my_roster = _compute_keep_score(my_roster, draft_df, k_season, k_form, k_fdr, k_draft)
        my_roster = my_roster.sort_values("Keep Score", ascending=False).reset_index(drop=True)

        st.subheader(f"My Roster — {my_team.get('team_name')}")
        st.dataframe(my_roster[
                         ["Player", "Team", "Position", "Season_Points", "Projected_Points", "Form", "AvgFDRNextN",
                          "Keep Score"]].style.format("{:.2f}", subset=["Season_Points", "Projected_Points", "Form",
                                                                        "AvgFDRNextN", "Keep Score"]),
                     use_container_width=True)
    else:
        st.info("No teams found in ownership data.")