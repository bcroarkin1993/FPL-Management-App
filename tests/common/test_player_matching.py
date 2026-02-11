"""Tests for scripts/common/player_matching.py."""

import pandas as pd
import pytest

from scripts.common.player_matching import canonical_normalize, PlayerRegistry


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


# ---- PlayerRegistry tests ----

class TestPlayerRegistry:
    @pytest.fixture
    def registry(self, mock_bootstrap_data):
        """Build a registry from mock bootstrap data."""
        reg = PlayerRegistry()
        reg.build_from_bootstrap(mock_bootstrap_data)
        return reg

    def test_is_built(self, registry):
        assert registry.is_built is True

    def test_len(self, registry):
        assert len(registry) == 6

    def test_lookup_by_id(self, registry):
        player = registry.lookup_by_id(4)
        assert player is not None
        assert player.name == "Erling Haaland"
        assert player.team_short == "MCI"
        assert player.position == "F"

    def test_lookup_by_id_missing(self, registry):
        assert registry.lookup_by_id(999) is None

    def test_lookup_by_name(self, registry):
        player = registry.lookup_by_name("Mohamed Salah")
        assert player is not None
        assert player.player_id == 5
        assert player.team_short == "LIV"

    def test_lookup_by_name_with_team_filter(self, registry):
        player = registry.lookup_by_name("William Saliba", team="ARS")
        assert player is not None
        assert player.player_id == 2

    def test_lookup_by_name_normalized(self, registry):
        """Accented or varied input should still match."""
        player = registry.lookup_by_name("kevin de bruyne")
        assert player is not None
        assert player.player_id == 3

    def test_lookup_by_web_name(self, registry):
        """Should be able to look up by web_name too."""
        player = registry.lookup_by_name("Haaland")
        assert player is not None
        assert player.player_id == 4

    def test_get_player_id(self, registry):
        pid = registry.get_player_id("Aaron Ramsdale")
        assert pid == 1

    def test_get_player_id_missing(self, registry):
        assert registry.get_player_id("Nonexistent Player") is None

    def test_enrich_dataframe(self, registry):
        df = pd.DataFrame({
            "Player": ["Erling Haaland", "Mohamed Salah"],
            "Team": ["MCI", "LIV"],
        })
        enriched = registry.enrich_dataframe(df, team_col="Team")
        assert "Player_ID" in enriched.columns
        assert enriched.loc[0, "Player_ID"] == 4
        assert enriched.loc[1, "Player_ID"] == 5

    def test_empty_registry(self):
        reg = PlayerRegistry()
        assert reg.is_built is False
        assert len(reg) == 0
        assert reg.lookup_by_id(1) is None
        assert reg.lookup_by_name("anyone") is None
