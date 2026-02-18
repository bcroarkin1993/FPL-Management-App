"""
Player Matching Module

Provides canonical name normalization, a PlayerRegistry for consistent
player matching across FPL and Rotowire data sources, and fuzzy matching
functions for merging FPL rosters with projection data.

The key insight is that FPL names often contain accents (e.g., "Raúl Jiménez")
while Rotowire uses ASCII names (e.g., "Raul Jimenez"). By normalizing both
to a canonical form, we can match players reliably.
"""

import unicodedata
import re
from typing import Dict, Optional, NamedTuple

from fuzzywuzzy import process, fuzz
import numpy as np
import pandas as pd
import requests
import streamlit as st

from scripts.common.text_helpers import (
    PLAYER_ALIASES,
    _clean_player_name,
    _map_position_to_rw,
    _norm_text,
    _strip_accents,
    _to_short_team_code,
)


class PlayerRecord(NamedTuple):
    """Immutable record for a player from FPL bootstrap data."""
    player_id: int
    name: str
    web_name: str
    team_short: str
    position: str
    norm_name: str  # Canonical normalized name for matching


def canonical_normalize(name: str) -> str:
    """
    Single source of truth for name normalization.

    Converts player names to a canonical form for matching:
    - Manual substitution for special characters (ø, æ, ð, etc.)
    - NFKD unicode normalize (decomposes accented characters)
    - ASCII encode (strips accents)
    - Lowercase
    - Remove non-alphanumeric characters
    - Collapse whitespace

    Examples:
        "Raúl Jiménez" -> "raul jimenez"
        "Bruno Fernandes" -> "bruno fernandes"
        "Heung-Min Son" -> "heungmin son"
        "N'Golo Kanté" -> "ngolo kante"
        "Rasmus Højlund" -> "rasmus hojlund"

    Args:
        name: Player name to normalize

    Returns:
        Canonical normalized name string
    """
    if pd.isna(name) or name is None:
        return ""

    s = str(name).strip()

    # Manual substitution for special characters that don't decompose cleanly
    special_chars = {
        'ø': 'o', 'Ø': 'O',
        'æ': 'ae', 'Æ': 'AE',
        'œ': 'oe', 'Œ': 'OE',
        'ð': 'd', 'Ð': 'D',
        'þ': 'th', 'Þ': 'Th',
        'ł': 'l', 'Ł': 'L',
        'đ': 'd', 'Đ': 'D',
        'ß': 'ss',
    }
    for char, replacement in special_chars.items():
        s = s.replace(char, replacement)

    # NFKD decomposition separates accents from base characters
    s = unicodedata.normalize("NFKD", s)

    # Encode to ASCII (drops accent characters) then decode back
    s = s.encode("ascii", "ignore").decode("ascii")

    # Lowercase
    s = s.lower()

    # Remove non-alphanumeric except spaces
    s = re.sub(r"[^a-z0-9 ]", "", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s


class PlayerRegistry:
    """
    Centralized player lookup using FPL Player_ID as primary key.

    Provides O(1) lookups by:
    - Player ID
    - Normalized name (optionally filtered by team/position)

    Usage:
        registry = get_player_registry()

        # Lookup by ID
        player = registry.lookup_by_id(123)

        # Lookup by name
        player = registry.lookup_by_name("Raul Jimenez", team="FUL", position="F")

        # Enrich a DataFrame with Player_ID
        df = registry.enrich_dataframe(df, name_col="Player")
    """

    def __init__(self):
        self._by_id: Dict[int, PlayerRecord] = {}
        self._by_norm_name: Dict[str, list] = {}  # norm_name -> [PlayerRecord, ...]
        self._built = False

    def build_from_bootstrap(self, bootstrap_data: dict) -> None:
        """
        Build indices from FPL bootstrap-static data.

        Args:
            bootstrap_data: Response from FPL bootstrap-static API
        """
        self._by_id.clear()
        self._by_norm_name.clear()

        players = bootstrap_data.get("elements", [])
        teams = bootstrap_data.get("teams", [])

        # Build team ID -> short_name mapping
        team_map = {t["id"]: t["short_name"] for t in teams}

        # Position mapping
        pos_map = {1: "G", 2: "D", 3: "M", 4: "F"}

        for p in players:
            player_id = p.get("id")
            if player_id is None:
                continue

            first_name = p.get("first_name", "")
            second_name = p.get("second_name", "")
            full_name = f"{first_name} {second_name}".strip()
            web_name = p.get("web_name", full_name)

            team_id = p.get("team", 0)
            team_short = team_map.get(team_id, "???")

            element_type = p.get("element_type", 0)
            position = pos_map.get(element_type, "?")

            norm_name = canonical_normalize(full_name)

            record = PlayerRecord(
                player_id=player_id,
                name=full_name,
                web_name=web_name,
                team_short=team_short,
                position=position,
                norm_name=norm_name
            )

            self._by_id[player_id] = record

            # Index by normalized name (handle collisions)
            if norm_name not in self._by_norm_name:
                self._by_norm_name[norm_name] = []
            self._by_norm_name[norm_name].append(record)

            # Also index by normalized web_name if different
            norm_web = canonical_normalize(web_name)
            if norm_web and norm_web != norm_name:
                if norm_web not in self._by_norm_name:
                    self._by_norm_name[norm_web] = []
                self._by_norm_name[norm_web].append(record)

        self._built = True

    def lookup_by_id(self, player_id: int) -> Optional[PlayerRecord]:
        """
        Lookup a player by their FPL ID.

        Args:
            player_id: FPL player ID

        Returns:
            PlayerRecord or None if not found
        """
        return self._by_id.get(int(player_id))

    def lookup_by_name(
        self,
        name: str,
        team: str = None,
        position: str = None
    ) -> Optional[PlayerRecord]:
        """
        Lookup a player by normalized name, optionally filtered by team/position.

        Args:
            name: Player name (will be canonically normalized)
            team: Optional team short code (e.g., "MCI") to filter matches
            position: Optional position (G/D/M/F) to filter matches

        Returns:
            Best matching PlayerRecord or None if not found
        """
        norm = canonical_normalize(name)
        if not norm:
            return None

        candidates = self._by_norm_name.get(norm, [])
        if not candidates:
            return None

        # Filter by team and/or position if provided
        if team or position:
            filtered = []
            for c in candidates:
                team_match = (team is None) or (c.team_short == team)
                pos_match = (position is None) or (c.position == position)
                if team_match and pos_match:
                    filtered.append(c)
            if filtered:
                return filtered[0]

        # Return first match if no filter or no filtered matches
        return candidates[0] if candidates else None

    def get_player_id(
        self,
        name: str,
        team: str = None,
        position: str = None
    ) -> Optional[int]:
        """
        Get a player's ID by name lookup.

        Convenience method that wraps lookup_by_name.

        Args:
            name: Player name
            team: Optional team short code
            position: Optional position

        Returns:
            Player ID or None if not found
        """
        record = self.lookup_by_name(name, team, position)
        return record.player_id if record else None

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        name_col: str = "Player",
        team_col: str = "Team",
        position_col: str = "Position",
        add_cols: bool = True
    ) -> pd.DataFrame:
        """
        Enrich a DataFrame with Player_ID from the registry.

        For each row, looks up the player by normalized name (optionally
        constrained by team and position) and adds the Player_ID.

        Args:
            df: DataFrame to enrich
            name_col: Name of the column containing player names
            team_col: Name of the column containing team codes (optional)
            position_col: Name of the column containing positions (optional)
            add_cols: If True, also add __norm_name column

        Returns:
            DataFrame with Player_ID column added/updated
        """
        result = df.copy()

        # Add normalized name column
        if add_cols and name_col in result.columns:
            result["__norm_name"] = result[name_col].apply(canonical_normalize)

        # Prepare Player_ID column
        if "Player_ID" not in result.columns:
            result["Player_ID"] = pd.NA

        has_team = team_col in result.columns
        has_pos = position_col in result.columns

        for idx in result.index:
            # Skip if already has valid Player_ID
            if pd.notna(result.at[idx, "Player_ID"]):
                continue

            name = result.at[idx, name_col] if name_col in result.columns else None
            if pd.isna(name):
                continue

            team = result.at[idx, team_col] if has_team else None
            pos = result.at[idx, position_col] if has_pos else None

            player_id = self.get_player_id(str(name), team, pos)
            if player_id is not None:
                result.at[idx, "Player_ID"] = player_id

        return result

    @property
    def is_built(self) -> bool:
        """Check if the registry has been built."""
        return self._built

    def __len__(self) -> int:
        """Return the number of players in the registry."""
        return len(self._by_id)


@st.cache_resource(show_spinner=False)
def get_player_registry() -> PlayerRegistry:
    """
    Get the cached PlayerRegistry singleton.

    Uses Streamlit's cache_resource for session-level caching.
    The registry is built from FPL bootstrap-static data on first access.

    Returns:
        Initialized PlayerRegistry instance
    """
    registry = PlayerRegistry()

    try:
        url = "https://draft.premierleague.com/api/bootstrap-static"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        registry.build_from_bootstrap(data)
    except Exception as e:
        # Log but don't fail - registry will just be empty
        st.warning(f"Failed to build player registry: {e}")

    return registry


def add_normalized_name_column(
    df: pd.DataFrame,
    name_col: str = "Player",
    norm_col: str = "__norm_name"
) -> pd.DataFrame:
    """
    Add a normalized name column to a DataFrame.

    Utility function for preparing DataFrames for normalized matching.

    Args:
        df: Input DataFrame
        name_col: Column containing player names
        norm_col: Name for the new normalized column

    Returns:
        DataFrame with normalized name column added
    """
    result = df.copy()
    if name_col in result.columns:
        result[norm_col] = result[name_col].apply(canonical_normalize)
    return result


# =============================================================================
# FUZZY MATCHING & MERGE FUNCTIONS (merged from utils.py Section 8)
# =============================================================================

def _backfill_player_ids(roster_df: pd.DataFrame, fpl_stats: pd.DataFrame) -> pd.DataFrame:
    """
    For rows where Player_ID is NaN, fill via fuzzy match against fpl_stats
    constrained by Team and Position (prefer exact team+pos matches).
    Safe if any expected columns are missing.
    """
    df = roster_df.copy()

    # Ensure required columns exist
    for col in ("Player", "Team", "Position"):
        if col not in df.columns:
            df[col] = np.nan
    if "Player_ID" not in df.columns:
        df["Player_ID"] = np.nan

    # Prepare candidate table safely
    cand = fpl_stats.copy()
    for col in ("Player", "Team", "Position", "Player_ID"):
        if col not in cand.columns:
            cand[col] = np.nan

    # Normalized names for matching (uses module-level _norm_text)
    cand["__name_norm"] = cand["Player"].apply(_norm_text)
    df["__name_norm"] = df["Player"].apply(_norm_text)

    # Indices to backfill
    try:
        missing_idx = df[df["Player_ID"].isna()].index
    except KeyError:
        # Safety: create it and mark all as missing
        df["Player_ID"] = np.nan
        missing_idx = df.index

    for i in missing_idx:
        name_norm = df.at[i, "__name_norm"]
        team = df.at[i, "Team"]
        pos = df.at[i, "Position"]

        # Try team+pos scope; then pos; then all
        scope = cand[(cand["Team"] == team) & (cand["Position"] == pos)]
        if scope.empty:
            scope = cand[cand["Position"] == pos]
        if scope.empty:
            scope = cand

        if scope.empty or scope["__name_norm"].isna().all():
            continue

        match = process.extractOne(name_norm, scope["__name_norm"].dropna().tolist(), scorer=fuzz.WRatio)
        if match:
            m_name, score = match[0], match[1]
            if score >= 85:
                pid = scope.loc[scope["__name_norm"] == m_name, "Player_ID"]
                if not pid.empty and pd.notna(pid.iloc[0]):
                    df.at[i, "Player_ID"] = float(pid.iloc[0])

    return df.drop(columns=["__name_norm"], errors="ignore")


def _fuzzy_match_player(fpl_player, fpl_team, fpl_position, candidates, projections_df,
                        fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Consolidated fuzzy matching function for player names.

    Finds the best match for a player using fuzzy matching with context-aware thresholds.
    If team and position match, uses a lower threshold; otherwise requires higher confidence.

    Parameters:
    - fpl_player: Player name to match.
    - fpl_team: Player's team.
    - fpl_position: Player's position.
    - candidates: List of candidate player names.
    - projections_df: DataFrame with projection data containing 'Player', 'Team', 'Position'.
    - fuzzy_threshold: Default threshold for matches (default: 80).
    - lower_fuzzy_threshold: Threshold when team+position match (default: 60).

    Returns:
    - Matched player name or None if no good match found.
    """
    if not candidates:
        return None

    result = process.extractOne(str(fpl_player), candidates)
    if not result:
        return None

    match, score = result[0], result[1]

    matched_row = projections_df[projections_df['Player'] == match]
    if not matched_row.empty:
        match_team = matched_row.iloc[0]['Team']
        match_position = matched_row.iloc[0]['Position']

        # Lower threshold if team and position match
        if match_team == fpl_team and match_position == fpl_position and score >= lower_fuzzy_threshold:
            return match

    # Higher threshold for general matches
    if score >= fuzzy_threshold:
        return match

    return None


def clean_fpl_player_names(fpl_players_df, projections_df, fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Cleans the player names in the FPL DataFrame by replacing them with their best matches from the projections DataFrame.

    Parameters:
    - fpl_players_df: DataFrame with FPL players ['Player', 'Team', 'Position'].
    - projections_df: DataFrame with Rotowire projections ['Player', 'Team', 'Position'].
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.

    Returns:
    - fpl_players_df: Updated FPL DataFrame with cleaned player names.
    """
    # Extract candidate names from projections
    projection_names = projections_df['Player'].tolist()

    def find_best_match_row(row):
        result = _fuzzy_match_player(
            row['player'], row['team_name'], row['position_abbrv'],
            projection_names, projections_df,
            fuzzy_threshold, lower_fuzzy_threshold
        )
        return result if result else row['player']

    # Update FPL DataFrame with cleaned player names
    fpl_players_df['player'] = fpl_players_df.apply(find_best_match_row, axis=1)

    return fpl_players_df


def find_best_match(fpl_player, fpl_team, fpl_position, candidates, projections_df,
                    fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Finds the best match for a player using fuzzy matching.

    Note: This is a public wrapper around _fuzzy_match_player for backward compatibility.

    Parameters:
    - fpl_player: Player name to match.
    - fpl_team: Player's team.
    - fpl_position: Player's position.
    - candidates: List of candidate names.
    - projections_df: DataFrame with Rotowire projections.
    - fuzzy_threshold: Default fuzzy matching threshold for player names.
    - lower_fuzzy_threshold: Lower threshold if team and position match.

    Returns:
    - Matched player name or None.
    """
    return _fuzzy_match_player(fpl_player, fpl_team, fpl_position, candidates, projections_df,
                               fuzzy_threshold, lower_fuzzy_threshold)


def merge_fpl_players_and_projections(fpl_players_df, projections_df,
                                      fuzzy_threshold=80, lower_fuzzy_threshold=60):
    """
    Robust merge of FPL players (Player/Team/Position) with projections.
    - Normalizes projections_df to RotoWire schema inside the function.
    - Uses canonical name normalization (strips accents) for reliable matching.
    - Tries exact match on normalized names first, then falls back to fuzzy matching.
    - Returns a table with players that *did* or *did not* match; unmatched get NA fields.
    """

    # -------- normalize projections to RW schema --------
    def _normalize_proj(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # case-insensitive rename - avoid duplicates by tracking which targets are used
        rename = {}
        targets_used = set()
        for c in df.columns:
            lc = c.strip().lower()
            target = None
            if lc in ('player', 'name', 'player_name'):                     target = 'Player'
            elif lc in ('team', 'team_short', 'teamname', 'team_name'):     target = 'Team'
            elif lc in ('matchup', 'fixture', 'opp', 'opponent'):           target = 'Matchup'
            elif lc in ('position', 'pos'):                                 target = 'Position'
            elif lc in ('points', 'point', 'proj', 'projection'):           target = 'Points'
            elif lc in ('pos rank','pos_rank','position rank','position_rank'):
                target = 'Pos Rank'
            elif lc in ('price',):                                          target = 'Price'
            elif lc in ('tsb','tsb%','tsb %','ownership'):                  target = 'TSB %'
            # Only rename if target not already used (avoid duplicate columns)
            if target and target not in targets_used:
                rename[c] = target
                targets_used.add(target)
        if rename:
            df = df.rename(columns=rename)

        # ensure required columns exist
        req_defaults = {
            'Player': None,
            'Team': None,
            'Matchup': '',
            'Position': None,
            'Points': np.nan,
            'Pos Rank': 'NA'
        }
        for k, v in req_defaults.items():
            if k not in df.columns:
                df[k] = v

        # optional (don't fail if they don't exist)
        if 'Price' not in df.columns:
            df['Price'] = np.nan
        if 'TSB %' not in df.columns:
            df['TSB %'] = np.nan

        # numeric coercions
        df['Points'] = pd.to_numeric(df['Points'], errors='coerce')
        # leave Pos Rank as-is; we'll handle at the end

        # return only the columns we use/expect
        keep = ['Player','Team','Matchup','Position','Points','Pos Rank','Price','TSB %']
        return df[[c for c in keep if c in df.columns]]

    proj_norm = _normalize_proj(projections_df)

    # guard: candidates for fuzzy
    if 'Player' not in proj_norm.columns:
        raise ValueError(f"Normalized projections missing 'Player' column. Have: {list(proj_norm.columns)}")

    # Add normalized name column for matching (strips accents, lowercase, etc.)
    proj_norm['__norm_name'] = proj_norm['Player'].apply(canonical_normalize)

    # Build lookup dict: normalized_name -> list of original Player names
    norm_to_players = {}
    for _, row in proj_norm.iterrows():
        norm = row['__norm_name']
        player = row['Player']
        if norm and pd.notna(player):
            if norm not in norm_to_players:
                norm_to_players[norm] = []
            if player not in norm_to_players[norm]:
                norm_to_players[norm].append(player)

    # Normalized candidates for fuzzy matching fallback
    normalized_candidates = list(norm_to_players.keys())

    # Build team-filtered lookup for prioritized matching
    # Maps (norm_name, team) -> list of original Player names
    norm_team_to_players = {}
    for _, row in proj_norm.iterrows():
        norm = row['__norm_name']
        player = row['Player']
        team = str(row.get('Team', ''))
        if norm and pd.notna(player):
            key = (norm, team)
            if key not in norm_team_to_players:
                norm_team_to_players[key] = []
            if player not in norm_team_to_players[key]:
                norm_team_to_players[key].append(player)

    # -------- matching strategy --------
    def _has_significant_token_overlap(fpl_name: str, proj_name: str) -> bool:
        """
        Check if the projection name is a valid match for the FPL name.
        Prevents false matches like "Pedro Porro Sauceda" -> "Joao Pedro"
        where only one common token "Pedro" causes high fuzzy scores.

        Valid matches:
        - "João Pedro Junqueira de Jesus" -> "Joao Pedro" (proj tokens subset of fpl)
        - "Carlos Henrique Casimiro" -> "Casemiro" (proj is single token ~= casimiro)
        - "Pedro Lomba Neto" -> "Pedro Neto" (proj tokens subset of fpl)

        Invalid matches:
        - "Pedro Porro Sauceda" -> "Joao Pedro" (proj has "joao" not in fpl)
        """
        fpl_tokens = set(fpl_name.lower().split())
        proj_tokens = set(proj_name.lower().split())
        if not fpl_tokens or not proj_tokens:
            return False

        def token_matches_any(token: str, token_set: set) -> bool:
            """Check if token matches any token in the set (exact or fuzzy)."""
            if token in token_set:
                return True
            # Allow fuzzy match for similar tokens (e.g., "casemiro" vs "casimiro")
            for t in token_set:
                # Use ratio for single token comparison - require 85%+ similarity
                if fuzz.ratio(token, t) >= 85:
                    return True
            return False

        def all_tokens_match(source_tokens: set, target_tokens: set) -> bool:
            """Check if all source tokens have a match in target tokens."""
            return all(token_matches_any(t, target_tokens) for t in source_tokens)

        # For a valid match, all tokens in the projection name should be in the FPL name
        # This handles: "Joao Pedro" in "João Pedro Junqueira de Jesus" ✓
        # But rejects: "Joao Pedro" from "Pedro Porro Sauceda" ✗ (no "joao" in fpl)
        # Also handles: "Casemiro" matching "Casimiro" with fuzzy matching
        if all_tokens_match(proj_tokens, fpl_tokens):
            return True

        # Also allow if FPL tokens are subset of proj (shorter FPL name)
        # E.g., "Pedro Neto" matching "Pedro Lomba Neto"
        if all_tokens_match(fpl_tokens, proj_tokens):
            return True

        # Allow if there's significant overlap (both share most tokens)
        # Require at least 2 common tokens or >66% of smaller set
        # Use fuzzy matching for overlap count
        fuzzy_overlap = sum(1 for t in fpl_tokens if token_matches_any(t, proj_tokens))
        smaller_len = min(len(fpl_tokens), len(proj_tokens))
        return fuzzy_overlap >= 2 and fuzzy_overlap >= smaller_len * 0.66

    def _best_match(fpl_player, fpl_team, fpl_position):
        """
        Match strategy:
        0. Check PLAYER_ALIASES for known difficult matches (Brazilian players, etc.)
        1. Try exact match on canonically normalized name (O(1) lookup)
        2. Try fuzzy match WITHIN the same team first (prioritize team context)
        3. Fall back to fuzzy match across all players
        Uses lower threshold when team+position agree for context-aware matching.
        Cross-team matches also require significant token overlap to prevent false matches.
        """
        if not normalized_candidates:
            return None

        fpl_player_str = str(fpl_player)
        fpl_team_str = str(fpl_team) if fpl_team else ''

        # Step 0: Check manual alias mapping for known difficult matches
        # Try exact match first, then normalized match
        alias_name = PLAYER_ALIASES.get(fpl_player_str)
        if not alias_name:
            # Try with stripped accents
            fpl_stripped = _strip_accents(fpl_player_str)
            alias_name = PLAYER_ALIASES.get(fpl_stripped)
        if alias_name:
            # Look up the alias in projections
            alias_norm = canonical_normalize(alias_name)
            if alias_norm in norm_to_players:
                return norm_to_players[alias_norm][0]

        # Normalize the FPL player name
        fpl_norm = canonical_normalize(fpl_player_str)
        if not fpl_norm:
            return None

        # Step 1: Try exact match on normalized name
        if fpl_norm in norm_to_players:
            exact_players = norm_to_players[fpl_norm]
            # If only one match, use it
            if len(exact_players) == 1:
                return exact_players[0]
            # Multiple matches: prefer one with matching team+position
            for player in exact_players:
                rows = proj_norm[proj_norm['Player'] == player]
                if rows.empty:
                    continue
                row = rows.iloc[0]
                if str(row.get('Team')) == fpl_team_str and str(row.get('Position')) == str(fpl_position):
                    return player
            # No team+position match, return first
            return exact_players[0]

        # Step 2: Try fuzzy match WITHIN the same team first
        # This ensures "Carlos Henrique Casimiro" (MUN) matches "Casemiro" (MUN) not "Carlos Baleba" (BHA)
        # Also validate token overlap to prevent "Pedro Lomba Neto" matching "Joao Pedro"
        same_team_candidates = [
            norm for norm in normalized_candidates
            if (norm, fpl_team_str) in norm_team_to_players
        ]

        if same_team_candidates:
            # Get multiple matches to evaluate token overlap
            matches = process.extract(fpl_norm, same_team_candidates, limit=5)
            for match_result in matches:
                matched_norm, score = match_result[0], match_result[1]
                if score >= lower_fuzzy_threshold:
                    # Validate token overlap before accepting
                    if _has_significant_token_overlap(fpl_norm, matched_norm):
                        original_players = norm_team_to_players.get((matched_norm, fpl_team_str), [])
                        if original_players:
                            return original_players[0]

        # Step 3: Fall back to fuzzy match across all players
        res = process.extractOne(fpl_norm, normalized_candidates)
        if not res:
            return None
        matched_norm, score = res[0], res[1]

        # Get the original player name(s) for this normalized name
        original_players = norm_to_players.get(matched_norm, [])
        if not original_players:
            return None

        # Find best matching player considering team+position
        for player in original_players:
            rows = proj_norm[proj_norm['Player'] == player]
            if rows.empty:
                continue
            row = rows.iloc[0]
            match_team = row.get('Team')
            match_pos = row.get('Position')

            # If team+position agrees, allow lower threshold
            if (str(match_team) == fpl_team_str) and (str(match_pos) == str(fpl_position)):
                if score >= lower_fuzzy_threshold:
                    return player

        # No team+position match - use stricter threshold AND require token overlap
        # This prevents "Pedro Porro Sauceda" from matching "Joao Pedro" where
        # the only common token is "Pedro"
        if score >= fuzzy_threshold:
            if _has_significant_token_overlap(fpl_norm, matched_norm):
                return original_players[0]

        return None

    # -------- iterate FPL rows and assemble output --------
    out = []
    # case-insensitive accessors for FPL columns
    fpl_cols = {c.lower(): c for c in fpl_players_df.columns}
    need = {'player','team','position'}
    missing = need - set(fpl_cols.keys())
    if missing:
        raise ValueError(
            "fpl_players_df must include columns ['Player','Team','Position'] "
            f"(case-insensitive). Missing: {sorted(missing)}. "
            f"Columns seen: {list(fpl_players_df.columns)}"
        )

    for _, r in fpl_players_df.iterrows():
        fpl_player = r[fpl_cols['player']]
        fpl_team = r[fpl_cols['team']]
        fpl_position = r[fpl_cols['position']]

        match = _best_match(fpl_player, fpl_team, fpl_position)

        if match:
            mrow = proj_norm.loc[proj_norm['Player'] == match].iloc[0]
            out.append({
                'Player': mrow.get('Player'),
                'Team': mrow.get('Team'),
                'Matchup': mrow.get('Matchup', ''),
                'Position': mrow.get('Position'),
                'Price': mrow.get('Price', np.nan),
                'TSB %': mrow.get('TSB %', np.nan),
                'Points': mrow.get('Points'),
                'Pos Rank': mrow.get('Pos Rank', 'NA')
            })
        else:
            # keep the original FPL row but with NA projections
            out.append({
                'Player': fpl_player,
                'Team': fpl_team,
                'Matchup': 'N/A',
                'Position': fpl_position,
                'Price': np.nan,
                'TSB %': np.nan,
                'Points': np.nan,
                'Pos Rank': 'NA'
            })

    merged = pd.DataFrame(out)

    # clean Pos Rank => ints or 'NA'
    pr = pd.to_numeric(merged['Pos Rank'], errors='coerce')
    pr = pr.round().astype('Int64')  # pandas nullable int
    # convert to object with 'NA' for missing
    merged['Pos Rank'] = pr.astype(object).where(pr.notna(), 'NA')

    # final column order (don't fail if some are missing)
    order = ['Player','Team','Matchup','Position','Price','TSB %','Points','Pos Rank']
    merged = merged[[c for c in order if c in merged.columns]]

    # 1-based index
    merged.index = pd.RangeIndex(start=1, stop=len(merged) + 1, step=1)
    return merged


def normalize_for_merge(fpl_df: pd.DataFrame,
                        rotowire_df: pd.DataFrame,
                        teams_df: pd.DataFrame = None):
    """
    Convenience wrapper: returns (fpl_norm, rw_norm) aligned to the same
    schema so downstream code can rely on ['Player','Team','Position'] and helpers.
    """
    fpl_norm = normalize_fpl_players_to_rotowire_schema(fpl_df, teams_df=teams_df)
    rw_norm = normalize_rotowire_players(rotowire_df)

    # IMPORTANT: align Team to the same representation (short codes).
    # RotoWire often uses full names; we add Team_Short there too, and then
    # overwrite 'Team' to be the short code for both dataframes to make
    # equality checks reliable.
    fpl_norm["Team"] = fpl_norm["Team_Short"]
    rw_norm["Team"] = rw_norm["Team_Short"]

    return fpl_norm, rw_norm


def normalize_fpl_players_to_rotowire_schema(fpl_df: pd.DataFrame,
                                             teams_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Convert an FPL players DataFrame (from bootstrap-static or your in-app tables) into
    the RotoWire-aligned schema.

    Expected columns (we handle flexible inputs):
      - Player name source: ('Player') OR ('first_name' + 'second_name') OR ('web_name')
      - Team source: ('Team') OR numeric `team` id (with teams_df) OR already short code
      - Position source: ('Position') OR numeric `element_type` OR text variants.

    Returns DF with at least: ['Player','Team','Position','Player_ID','Team_Short','Player_Clean']
    (we keep other columns you had, too).
    """
    df = fpl_df.copy()

    # Column finder (case-insensitive)
    cmap = {c.lower(): c for c in df.columns}
    def _c(name):
        return cmap.get(name.lower())

    # --- Player name ---
    player_col = _c("Player")
    if player_col is None:
        fn, sn, wn = _c("first_name"), _c("second_name"), _c("web_name")
        if fn and sn:
            df["Player"] = (df[fn].astype(str).str.strip() + " " + df[sn].astype(str).str.strip()).str.strip()
        elif wn:
            df["Player"] = df[wn].astype(str).str.strip()
        else:
            raise ValueError("FPL df needs 'Player' or ('first_name' and 'second_name') or 'web_name'.")
    elif player_col != "Player":
        # Rename to canonical 'Player' column
        df.rename(columns={player_col: "Player"}, inplace=True)

    # --- Team ---
    # Prefer short codes; if the frame has numeric 'team' id + teams_df (bootstrap teams), we can map.
    team_col = _c("Team")
    if team_col is None:
        # Check for various team column names: 'team' (numeric ID) or 'team_name' (full name)
        team_src = _c("team") or _c("team_name") or _c("team_short")
        if team_src is not None:
            # Convert to short code format
            if teams_df is None or not {"id", "short_name"}.issubset(set(teams_df.columns)):
                df["Team"] = df[team_src].apply(lambda v: _to_short_team_code(v, teams_df=None))
            else:
                df["Team"] = df[team_src].apply(lambda v: _to_short_team_code(v, teams_df=teams_df))
        else:
            # If there's no team col at all, create empty; you can fill later
            df["Team"] = ""
    elif team_col != "Team":
        # Rename to canonical 'Team' column and ensure it's converted to short codes
        df.rename(columns={team_col: "Team"}, inplace=True)
        df["Team"] = df["Team"].apply(lambda v: _to_short_team_code(v, teams_df=teams_df))

    # --- Position ---
    pos_col = _c("Position")
    if pos_col is None:
        pos_src = _c("element_type") or _c("pos") or _c("position_abbrv") or _c("position")
        if pos_src is None:
            raise ValueError("FPL df needs 'Position' or a mappable source like 'element_type' or 'pos'.")
        df["Position"] = df[pos_src].apply(_map_position_to_rw)
    elif pos_col != "Position":
        # Rename to canonical 'Position' column
        df.rename(columns={pos_col: "Position"}, inplace=True)

    # --- Player_ID (optional if present) ---
    pid_col = _c("id") or _c("player_id")
    if pid_col:
        df.rename(columns={pid_col: "Player_ID"}, inplace=True)

    # Normalize values / helpers
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Team_Short"] = df["Team"].apply(_to_short_team_code)
    df["Player_Clean"] = df["Player"].map(_clean_player_name)

    # Reorder helpful basics at front if present
    front = [c for c in ["Player_ID","Player","Team","Position","Team_Short","Player_Clean"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df


def normalize_rotowire_players(rotowire_df: pd.DataFrame) -> pd.DataFrame:
    """
    Take your RotoWire projections DataFrame in any reasonable form and return:
    columns ['Player','Team','Position', ...], plus helper columns:
    - 'Team_Short' (3-letter code)
    - 'Player_Clean' (accent- and punctuation-stripped lower key)
    No columns are dropped; we just add/fix and standardize casing.
    """
    df = rotowire_df.copy()

    # Unify column names (case-insensitive)
    colmap = {c.lower(): c for c in df.columns}
    def _get(col):
        for k in colmap:
            if k == col.lower():
                return colmap[k]
        return None

    # Ensure the three core columns exist
    pcol = _get("player") or _get("name")
    tcol = _get("team")
    ccol = _get("position") or _get("pos")

    if pcol is None or tcol is None or ccol is None:
        raise ValueError("RotoWire df must contain columns for player, team, and position.")

    # Standardize names
    if pcol != "Player":
        df.rename(columns={pcol: "Player"}, inplace=True)
    if tcol != "Team":
        df.rename(columns={tcol: "Team"}, inplace=True)
    if ccol != "Position":
        df.rename(columns={ccol: "Position"}, inplace=True)

    # Normalize values
    df["Player"] = df["Player"].astype(str).map(lambda s: s.strip())
    df["Team_Short"] = df["Team"].apply(_to_short_team_code)
    df["Position"] = df["Position"].apply(_map_position_to_rw)
    df["Player_Clean"] = df["Player"].map(_clean_player_name)

    return df
