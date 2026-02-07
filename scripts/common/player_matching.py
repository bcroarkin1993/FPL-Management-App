"""
Player Matching Module

Provides canonical name normalization and a PlayerRegistry for consistent
player matching across FPL and Rotowire data sources.

The key insight is that FPL names often contain accents (e.g., "Raúl Jiménez")
while Rotowire uses ASCII names (e.g., "Raul Jimenez"). By normalizing both
to a canonical form, we can match players reliably.
"""

import unicodedata
import re
from typing import Dict, Optional, NamedTuple
import pandas as pd
import requests
import streamlit as st


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
