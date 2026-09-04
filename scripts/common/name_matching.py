"""
Tiered cross-source player name matching — pure, Streamlit-free.

This module holds :class:`ReferenceMatcher` and its helpers. They used to live
in ``player_matching.py``, which imports Streamlit for the cached
``PlayerRegistry``. Matching itself needs nothing from Streamlit, and the
projection engine and the GitHub Actions snapshot collector must be able to
import it in an environment where ``pip install -r requirements.txt`` was
best-effort (see ``.github/workflows/fpl-notifications.yml``, which installs
with ``|| true``). Same reasoning as ``transfer_feeds.py`` and
``data_source_checks.py`` avoiding Streamlit.

``player_matching.py`` re-exports everything here, so every existing import
site keeps working unchanged.
"""

import difflib
from typing import Dict, Optional

import pandas as pd

from scripts.common.text_helpers import canonical_normalize, _to_short_team_code

# ---------------------------------------------------------------------------
# Tiered reference matching
# ---------------------------------------------------------------------------
#
# Rotowire (and other projection sources) publish *common* names — "Bruno
# Fernandes", "Gabriel", "David Raya" — while the FPL bootstrap publishes full
# legal names — "Bruno Borges Fernandes", "Gabriel dos Santos Magalhaes",
# "David Raya Martin". A single exact (name, team) key therefore misses ~16% of
# a 425-row season-rankings table, and the misses are silent: the caller sees
# NaN and substitutes a neutral default, so an elite asset scores as average
# with no error anywhere.
#
# ReferenceMatcher applies progressively looser tiers, and **every** loose tier
# is scoped to both Team and Position. There is deliberately no team-agnostic
# tier: a surname-only match once attached Cole Palmer's stats (elite MID) to
# Alex Palmer (backup GK) and pushed him into the top 10 overall.
#
# Ambiguity always resolves to "no match" rather than an arbitrary pick.


def _hyphen_split_normalize(name) -> str:
    """canonical_normalize, but hyphens/apostrophes become word breaks.

    canonical_normalize *deletes* them ("Solanke-Mitchell" -> "solankemitchell"),
    which is right for whole-string equality but wrong for token comparison,
    where we need {"solanke", "mitchell"}.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""
    return canonical_normalize(str(name).replace("-", " ").replace("'", " "))


def _name_tokens(name) -> frozenset:
    """Token set of a name, hyphen-aware. Empty set for blanks."""
    return frozenset(_hyphen_split_normalize(name).split())


class ReferenceMatcher:
    """Match player names against a reference table using tiered fallbacks.

    Build once over the reference frame, then call :meth:`match` per player.

    Tiers, in order:
      1. exact (normalized name, team)
      2. exact (normalized web name, team)
      3. last word, scoped to (team, position)
      4. token subset in either direction, scoped to (team, position)
         -- token sets are unordered, so this also covers reversed name order
            ("Kaoru Mitoma" vs "Mitoma Kaoru")
      5. difflib ratio >= fuzzy_threshold, scoped to (team, position)
      6. exact name + position, team ignored -- opt-in only

    Tier 6 exists because two sources disagree about a club for weeks after a
    transfer: FPL lists Nicolas Jackson at Aston Villa while FFP still has him
    at Chelsea, and every team-scoped tier misses him. It is off by default,
    and safe only because it demands the *whole* name plus position: the
    surname-only version of this idea once gave Alex Palmer (backup GK) Cole
    Palmer's stats. Enable it for reference tables that lag on transfers, not
    for one-shot rankings where a club disagreement means the wrong player.

    Args:
        reference_df: Frame to match against.
        name_col: Primary name column in the reference.
        web_name_col: Optional short/display name column. Tier skipped if absent.
        team_col: Team column. Required -- without it only ambiguity-free
            global name matches are possible, so all tiers degrade to tier 1.
        position_col: Position column. Tiers 3-6 are skipped if absent, since
            they rely on it as the corroborating signal.
        fuzzy_threshold: Minimum difflib ratio for tier 5.
        allow_cross_team_exact: Enable tier 6.
    """

    def __init__(
        self,
        reference_df: pd.DataFrame,
        name_col: str = "Player",
        web_name_col: Optional[str] = "Web_Name",
        team_col: Optional[str] = "Team",
        position_col: Optional[str] = "Position",
        fuzzy_threshold: float = 0.78,
        allow_cross_team_exact: bool = False,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.allow_cross_team_exact = allow_cross_team_exact
        self._exact: Dict[tuple, list] = {}
        self._exact_web: Dict[tuple, list] = {}
        self._groups: Dict[tuple, list] = {}
        self._cross_team: Dict[tuple, list] = {}
        self.tier_counts: Dict[str, int] = {}

        if reference_df is None or reference_df.empty or name_col not in reference_df.columns:
            return

        has_web = bool(web_name_col) and web_name_col in reference_df.columns
        has_team = bool(team_col) and team_col in reference_df.columns
        has_pos = bool(position_col) and position_col in reference_df.columns
        self._has_pos = has_pos

        for idx, row in reference_df.iterrows():
            team = _to_short_team_code(row[team_col]) if has_team else ""
            pos = str(row[position_col]) if has_pos else ""

            norm = _hyphen_split_normalize(row[name_col])
            if norm:
                self._exact.setdefault((norm, team), []).append(idx)

            web_norm = ""
            if has_web:
                web_norm = _hyphen_split_normalize(row[web_name_col])
                if web_norm:
                    self._exact_web.setdefault((web_norm, team), []).append(idx)

            if has_pos:
                self._groups.setdefault((team, pos), []).append(
                    (idx, norm, web_norm, _name_tokens(row[name_col]),
                     _name_tokens(row[web_name_col]) if has_web else frozenset())
                )
                if allow_cross_team_exact and norm:
                    self._cross_team.setdefault((norm, pos), []).append(idx)

    @staticmethod
    def _sole(hits) -> Optional[object]:
        """Unique hit, or None. Ambiguity is treated as no match."""
        return hits[0] if hits and len(hits) == 1 else None

    def match(self, name, team=None, position=None) -> Optional[object]:
        """Return the reference index label for `name`, or None.

        Args:
            name: Player name to look up.
            team: Team code/name. Normalized to a short code internally.
            position: Single-letter position (G/D/M/F). Required for tiers 3-5.
        """
        return self.match_with_tier(name, team, position)[0]

    def match_with_tier(self, name, team=None, position=None, cross_team=True):
        """Like :meth:`match`, but returns ``(index, tier_rank)``.

        tier_rank is 1 (strongest) to 6 (weakest), or None when unmatched.
        Callers resolving contention for the same reference row should prefer
        the lower rank.

        Set ``cross_team=False`` to suppress tier 6 for this query alone. Callers
        retry with a short display name when the full name misses, and a short
        name is too weak a key to also drop the team: "Savio" is one word that
        several players could answer to, so matching it at a club he does not
        play for rests on nothing but the word itself.
        """
        norm = _hyphen_split_normalize(name)
        if not norm:
            return None, None
        team_code = _to_short_team_code(team) if team is not None else ""

        hit = self._sole(self._exact.get((norm, team_code)))
        if hit is not None:
            self.tier_counts["exact_name"] = self.tier_counts.get("exact_name", 0) + 1
            return hit, 1

        hit = self._sole(self._exact_web.get((norm, team_code)))
        if hit is not None:
            self.tier_counts["exact_web_name"] = self.tier_counts.get("exact_web_name", 0) + 1
            return hit, 2

        # Tiers 3-5 require position as a corroborating signal.
        if not getattr(self, "_has_pos", False) or position is None:
            return None, None
        tokens = _name_tokens(name)
        last_word = norm.split()[-1]

        # A transferred player's new club may have no reference rows at all, so
        # an empty group must fall through to tier 6 rather than end the search.
        group = self._groups.get((team_code, str(position))) or []

        hits = [g[0] for g in group if g[1] and g[1].split()[-1] == last_word]
        hit = self._sole(hits)
        if hit is not None:
            self.tier_counts["last_word"] = self.tier_counts.get("last_word", 0) + 1
            return hit, 3

        hits = [
            g[0] for g in group
            if (g[3] and (tokens <= g[3] or g[3] <= tokens))
            or (g[4] and (tokens <= g[4] or g[4] <= tokens))
        ]
        hit = self._sole(hits)
        if hit is not None:
            self.tier_counts["token_subset"] = self.tier_counts.get("token_subset", 0) + 1
            return hit, 4

        # A fuzzy match must still agree on at least one *complete* name part.
        # Character-level similarity alone is not enough: "Harrison" scores 0.79
        # against team-mate "Harry Wilson" while sharing no token, and quietly
        # hands one player another's projection. The real variants this tier
        # exists for -- "Yarmoliuk"/"Yarmolyuk", "Kadoglu"/"Kadioglu" -- all
        # share a full token and score above 0.93.
        best_idx, best_score, runner_up = None, 0.0, 0.0
        for g in group:
            if not (tokens & g[3] or tokens & g[4]):
                continue
            for target in (g[1], g[2]):
                if not target:
                    continue
                score = difflib.SequenceMatcher(None, norm, target).ratio()
                if score > best_score:
                    best_idx, runner_up, best_score = g[0], best_score, score
                elif score > runner_up:
                    runner_up = score
        # Require a clear winner, not a coin flip between two similar names.
        if best_score >= self.fuzzy_threshold and best_score > runner_up:
            self.tier_counts["fuzzy"] = self.tier_counts.get("fuzzy", 0) + 1
            return best_idx, 5

        # Tier 6: the same player, filed under a club he has already left.
        if not cross_team:
            return None, None
        hit = self._sole(self._cross_team.get((norm, str(position))))
        if hit is not None:
            self.tier_counts["cross_team_exact"] = (
                self.tier_counts.get("cross_team_exact", 0) + 1)
            return hit, 6
        return None, None
