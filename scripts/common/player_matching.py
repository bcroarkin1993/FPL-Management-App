"""
Player Matching Module

Provides canonical name normalization and the fuzzy merge that attaches
Rotowire projections to an FPL roster.

For resolving a name to a player, prefer ``ReferenceMatcher``
(``scripts/common/name_matching.py``): every tier below the first is scoped by
team and position, and one reference row is given to at most one player.

The key insight is that FPL names often contain accents (e.g., "Raúl Jiménez")
while Rotowire uses ASCII names (e.g., "Raul Jimenez"). By normalizing both
to a canonical form, we can match players reliably.
"""

from fuzzywuzzy import process, fuzz
import numpy as np
import pandas as pd

from scripts.common.text_helpers import (
    PLAYER_ALIASES,
    canonical_normalize,
    _clean_player_name,
    _map_position_to_rw,
    _norm_text,
    _strip_accents,
    _to_short_team_code,
)


# ``PlayerRecord``, ``PlayerRegistry``, ``get_player_registry`` and
# ``add_normalized_name_column`` were removed on 2026-10-30.
#
# The registry was built from the **Draft** bootstrap and stamped its ids as
# ``Player_ID``, which is the one id space the rest of the app does not speak:
# `draft.premierleague.com` and `fantasy.premierleague.com` number 64 of 669
# players differently, so an id from it resolves to a real but wrong player
# downstream, silently. Nothing in the app ever called it -- only its own tests
# did -- so it was not a live bug but a loaded one: the first caller would have
# reintroduced the failure `scripts/common/element_ids.py` exists for.
#
# ``add_normalized_name_column`` and ``normalize_for_merge`` had no callers at
# all. Left deleted rather than deprecated so a future caller cannot reach for
# them; `ReferenceMatcher` (`name_matching.py`) is the supported way to resolve
# a name, and the element id is carried, never re-derived.


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


# ``_fuzzy_match_player()``, ``find_best_match()`` and
# ``clean_fpl_player_names()`` were removed here. They fuzzy-matched a player
# against a projection source's name list and *renamed* him to the best hit,
# scored with ``fuzzywuzzy.WRatio`` at a threshold of 80. WRatio falls back to
# partial-token ratios once two strings differ enough in length, so one shared
# token carries it: "Maxim De Cuyper" scores 86 against "João Pedro Junqueira
# de Jesus" on the word *de*. Measured live on their only remaining callsite,
# 87 of 667 players were renamed and 66 ended up sharing a name with another
# player -- eleven of them rendered as "Maxim De Cuyper", and Alex Palmer
# rendered as Cole Palmer.
#
# Nothing needs a name match to *display* a name: ``to_display_name()`` builds
# the common name from the bootstrap's own fields. Nothing needs a hand-rolled
# ladder to match one either: ``ReferenceMatcher`` /
# ``analytics._claim_reference_rows()`` scope every loose tier by team and
# position and give one reference row to at most one player. They are left
# deleted rather than deprecated so a future caller cannot reach for them.


def merge_fpl_players_and_projections(fpl_players_df, projections_df,
                                      fuzzy_threshold=80, lower_fuzzy_threshold=60,
                                      carry_cols=None):
    """
    Robust merge of FPL players (Player/Team/Position) with projections.
    - Normalizes projections_df to RotoWire schema inside the function.
    - Uses canonical name normalization (strips accents) for reliable matching.
    - Tries exact match on normalized names first, then falls back to fuzzy matching.
    - Returns a table with players that *did* or *did not* match; unmatched get NA fields.

    carry_cols: FPL-side columns to copy onto the output rows (e.g. ['Player_ID']).
    A matched row takes its 'Player' from the *projection* source, so anything keyed
    on the FPL name is lost here -- carry the element id instead and join on that.
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
            row = {
                'Player': mrow.get('Player'),
                'Team': mrow.get('Team'),
                'Matchup': mrow.get('Matchup', ''),
                'Position': mrow.get('Position'),
                'Price': mrow.get('Price', np.nan),
                'TSB %': mrow.get('TSB %', np.nan),
                'Points': mrow.get('Points'),
                'Pos Rank': mrow.get('Pos Rank', 'NA')
            }
        else:
            # keep the original FPL row but with NA projections
            row = {
                'Player': fpl_player,
                'Team': fpl_team,
                # Empty, not "N/A". Rotowire lists only expected starters, so
                # this branch is every squad player it left out -- and a fixture
                # is a property of the club, which is perfectly well known. The
                # pages backfill it from the fixture list via attach_matchups();
                # where one forgets, an empty cell renders as nothing, while
                # "N/A" renders as a confident statement that the club has no
                # fixture. The module's own other default here is already ''.
                'Matchup': '',
                'Position': fpl_position,
                'Price': np.nan,
                'TSB %': np.nan,
                'Points': np.nan,
                'Pos Rank': 'NA'
            }

        for col in (carry_cols or []):
            if col in fpl_players_df.columns:
                row[col] = r[col]

        out.append(row)

    merged = pd.DataFrame(out)

    # clean Pos Rank => ints or 'NA'
    pr = pd.to_numeric(merged['Pos Rank'], errors='coerce')
    pr = pr.round().astype('Int64')  # pandas nullable int
    # convert to object with 'NA' for missing
    merged['Pos Rank'] = pr.astype(object).where(pr.notna(), 'NA')

    # final column order (don't fail if some are missing)
    order = ['Player','Team','Matchup','Position','Price','TSB %','Points','Pos Rank']
    order += [c for c in (carry_cols or []) if c in merged.columns]
    merged = merged[[c for c in order if c in merged.columns]]

    # 1-based index
    merged.index = pd.RangeIndex(start=1, stop=len(merged) + 1, step=1)
    return merged



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



# ---------------------------------------------------------------------------
# Tiered reference matching
# ---------------------------------------------------------------------------
#
# ReferenceMatcher and its helpers now live in ``name_matching.py``, which is
# Streamlit-free so the projection engine and the GitHub Actions snapshot
# collector can import it. Re-exported here because this module is the
# documented home of cross-source matching and every existing callsite (and
# ``tests/common/test_player_matching.py``) imports them from here.
from scripts.common.name_matching import (  # noqa: F401  (re-export)
    ReferenceMatcher,
    _hyphen_split_normalize,
    _name_tokens,
)
