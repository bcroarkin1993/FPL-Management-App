"""Tests for scripts/common/player_matching.py."""

import pandas as pd
import pytest

from scripts.common.player_matching import (
    canonical_normalize,
    ReferenceMatcher,
)


# ---- canonical_normalize tests ----

class TestCanonicalNormalize:
    def test_basic_ascii(self):
        assert canonical_normalize("Bruno Fernandes") == "bruno fernandes"

    def test_accented_characters(self):
        assert canonical_normalize("Raúl Jiménez") == "raul jimenez"

    def test_nordic_o(self):
        assert canonical_normalize("Rasmus Højlund") == "rasmus hojlund"

    def test_nordic_o_stroke(self):
        assert canonical_normalize("Martin Ødegaard") == "martin odegaard"

    def test_hyphen_removal(self):
        assert canonical_normalize("Heung-Min Son") == "heungmin son"

    def test_apostrophe_removal(self):
        assert canonical_normalize("N'Golo Kanté") == "ngolo kante"

    def test_polish_l(self):
        assert canonical_normalize("Robert Lewandowski") == "robert lewandowski"
        assert canonical_normalize("Łukasz Fabiański") == "lukasz fabianski"

    def test_german_ss(self):
        assert canonical_normalize("Straße") == "strasse"

    def test_ae_ligature(self):
        assert canonical_normalize("Præst") == "praest"

    def test_none_returns_empty(self):
        assert canonical_normalize(None) == ""

    def test_nan_returns_empty(self):
        assert canonical_normalize(float("nan")) == ""

    def test_empty_string(self):
        assert canonical_normalize("") == ""

    def test_whitespace_collapse(self):
        assert canonical_normalize("  Bruno   Fernandes  ") == "bruno fernandes"

    def test_numeric_preserved(self):
        assert canonical_normalize("Player 9") == "player 9"

    def test_icelandic_eth(self):
        assert canonical_normalize("Guðmundsson") == "gudmundsson"


class TestReferenceMatcher:
    """Tiered matching against a projection source.

    Every fixture below is a real name pair from the Rotowire season rankings
    that the previous strict (name, team) merge silently dropped. Rotowire
    publishes common names, the FPL bootstrap publishes full legal names, so a
    single exact key missed 69 of 425 rows -- including the #2 asset in the game
    -- and every miss rendered as a neutral 0.5 default with no error anywhere.
    """

    REFERENCE = pd.DataFrame([
        # name,               web_name,    team,  position
        ("Bruno Fernandes",   "Fernandes", "MUN", "M"),
        ("Gabriel",           "Gabriel",   "ARS", "D"),
        ("David Raya",        "Raya",      "ARS", "G"),
        ("Ezri Konsa",        "Konsa",     "AVL", "D"),
        ("Dominic Solanke",   "Solanke",   "TOT", "F"),
        ("Djordje Petrovic",  "Petrovic",  "BOU", "G"),
        ("Ferdi Kadioglu",    "Kadioglu",  "BHA", "D"),
        ("Kaoru Mitoma",      "Mitoma",    "BHA", "M"),
        ("Harry Wilson",      "Wilson",    "LEE", "M"),
        ("Cole Palmer",       "Palmer",    "CHE", "M"),
    ], columns=["Player", "Web_Name", "Team", "Position"])

    def _matcher(self):
        return ReferenceMatcher(self.REFERENCE)

    @pytest.mark.parametrize("fpl_name, team, position, expected", [
        # Tier 1 -- exact.
        ("Bruno Fernandes", "MUN", "M", "Bruno Fernandes"),
        # Full legal name vs common name.
        ("Bruno Borges Fernandes", "MUN", "M", "Bruno Fernandes"),
        # Token subset: FPL appends further surnames.
        ("David Raya Martin", "ARS", "G", "David Raya"),
        ("Ezri Konsa Ngoyo", "AVL", "D", "Ezri Konsa"),
        # Hyphenated surname -- canonical_normalize deletes the hyphen, so this
        # only matches if tokens are split on it first.
        ("Dominic Solanke-Mitchell", "TOT", "F", "Dominic Solanke"),
        # Transliteration that canonical_normalize folds to ASCII.
        ("Đorđe Petrović", "BOU", "G", "Djordje Petrovic"),
        # Reversed name order -- token sets are unordered, so tier 4 covers it.
        ("Mitoma Kaoru", "BHA", "M", "Kaoru Mitoma"),
        # Spelling variant: fuzzy, but shares the complete token "Ferdi".
        ("Ferdi Kadoglu", "BHA", "D", "Ferdi Kadioglu"),
    ])
    def test_matches_known_name_variants(self, fpl_name, team, position, expected):
        idx = self._matcher().match(fpl_name, team, position)
        assert idx is not None, "%r should have matched %r" % (fpl_name, expected)
        assert self.REFERENCE.at[idx, "Player"] == expected

    @pytest.mark.parametrize("fpl_name", ["Gabriel dos Santos Magalhaes", "Gabriel"])
    def test_mononym_matches_from_either_name_column(self, fpl_name):
        """Rotowire publishes bare "Gabriel"; FPL carries both a long legal name
        and the web_name. Token-subset resolves the long form, so the caller
        matches whether it passes Player or Web_Name."""
        idx = self._matcher().match(fpl_name, "ARS", "D")
        assert idx is not None and self.REFERENCE.at[idx, "Player"] == "Gabriel"

    def test_fuzzy_requires_a_shared_complete_token(self):
        """The guard against handing one player another's projection.

        "Harrison" scores ~0.79 against team-mate "Harry Wilson" on raw character
        similarity while sharing no name part. Without a shared-token rule this
        silently gave Jack Harrison Harry Wilson's season points.
        """
        assert self._matcher().match("Harrison", "LEE", "M") is None

    def test_never_matches_across_position(self):
        """Alex Palmer (backup GK) must never inherit Cole Palmer's (elite MID) stats.

        A team-agnostic surname tier did exactly this once and pushed a backup
        goalkeeper into the top 10 overall, which is why every loose tier here is
        scoped to team *and* position.
        """
        assert self._matcher().match("Alex Palmer", "CHE", "G") is None

    def test_never_matches_across_team(self):
        assert self._matcher().match("Bruno Fernandes", "ARS", "M") is None

    def test_ambiguity_resolves_to_no_match(self):
        """Two equally good candidates is a reason to abstain, not to guess."""
        ref = pd.DataFrame([
            ("Danny Ings", "Ings", "AVL", "F"),
            ("Danny Ings", "Ings", "AVL", "F"),
        ], columns=["Player", "Web_Name", "Team", "Position"])
        assert ReferenceMatcher(ref).match("Danny Ings", "AVL", "F") is None

    def test_degrades_without_position_column(self):
        """Callers lacking Position keep the exact tiers and lose the loose ones."""
        ref = self.REFERENCE.drop(columns=["Position"])
        m = ReferenceMatcher(ref, position_col=None)
        assert m.match("Bruno Fernandes", "MUN") is not None
        assert m.match("Bruno Borges Fernandes", "MUN") is None

    def test_empty_and_blank_inputs_are_safe(self):
        assert ReferenceMatcher(pd.DataFrame()).match("Anyone", "MUN", "M") is None
        assert self._matcher().match("", "MUN", "M") is None
        assert self._matcher().match(None, "MUN", "M") is None

    def test_reports_which_tier_matched(self):
        m = self._matcher()
        m.match("Bruno Fernandes", "MUN", "M")
        m.match("David Raya Martin", "ARS", "G")
        assert m.tier_counts["exact_name"] == 1
        assert m.tier_counts["token_subset"] == 1


class TestCrossTeamExactTier:
    """Tier 6: the same player, filed under a club he has already left.

    Sources disagree about a club for weeks after a transfer -- FPL had Nicolas
    Jackson at Aston Villa on 2026-09-03 while FFP still listed him at Chelsea --
    so every team-scoped tier misses him. The tier is opt-in and demands the
    whole name plus position, which is what separates it from the surname-only
    fallback it replaced.
    """

    REFERENCE = pd.DataFrame([
        ("Nicolas Jackson", "CHE", "F"),
        ("Cole Palmer",     "CHE", "M"),
        ("Dillon Phillips", "HUL", "G"),
    ], columns=["Player", "Team", "Position"])

    def _matcher(self, **kwargs):
        return ReferenceMatcher(self.REFERENCE, web_name_col=None, **kwargs)

    def test_off_by_default(self):
        assert self._matcher().match("Nicolas Jackson", "AVL", "F") is None

    def test_matches_a_transferred_player(self):
        idx = self._matcher(allow_cross_team_exact=True).match("Nicolas Jackson", "AVL", "F")
        assert idx is not None
        assert self.REFERENCE.at[idx, "Player"] == "Nicolas Jackson"

    def test_still_requires_the_whole_name(self):
        """Alex Palmer must not inherit Cole Palmer's stats through this tier."""
        m = self._matcher(allow_cross_team_exact=True)
        assert m.match("Alex Palmer", "IPS", "G") is None

    def test_still_requires_position(self):
        """Kalvin Phillips (MID) against Dillon Phillips (GK) -- a shared surname
        and nothing else, which is how a goalkeeper's start rate reached a
        midfielder in the live data this tier was written for."""
        m = self._matcher(allow_cross_team_exact=True)
        assert m.match("Kalvin Phillips", "MCI", "M") is None
        assert m.match("Dillon Phillips", "MCI", "M") is None

    def test_ambiguity_resolves_to_no_match(self):
        ref = pd.DataFrame([
            ("Danny Ings", "AVL", "F"),
            ("Danny Ings", "WHU", "F"),
        ], columns=["Player", "Team", "Position"])
        m = ReferenceMatcher(ref, web_name_col=None, allow_cross_team_exact=True)
        assert m.match("Danny Ings", "BRE", "F") is None

    def test_a_short_display_name_may_not_cross_clubs(self):
        """Callers retry on web_name when the full name misses, and a one-word
        display name is too weak a key to also drop the team -- "Savio" matched
        a Man City row while playing for Spurs on nothing but the word itself."""
        m = self._matcher(allow_cross_team_exact=True)
        assert m.match_with_tier("Nicolas Jackson", "AVL", "F", cross_team=False) == (None, None)

    def test_reports_its_tier_rank(self):
        m = self._matcher(allow_cross_team_exact=True)
        assert m.match_with_tier("Nicolas Jackson", "AVL", "F")[1] == 6
        assert m.match_with_tier("Nicolas Jackson", "CHE", "F")[1] == 1
