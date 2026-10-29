"""Resolving a projection source's short name back to an FPL element id.

The reported symptom was "James Hill recommended over António João Silva".
Measured on the live 101-row Draft available pool, 12 players lost their
element id at an exact-equality join -- including the 4th-best available
defender -- and everything downstream is keyed on that id, so their season
points, starts, injury status and start signal all silently read as absent.
"""

import numpy as np
import pandas as pd

from scripts.common.analytics import attach_reference_ids


def _ref():
    """The FPL pool: full legal names, plus a web_name and an element id."""
    return pd.DataFrame({
        "Player": [
            "António João Pereira de Albuquerque Tavares da Silva",
            "James Hill",
            "Joelinton Cássio Apolinário de Lira",
            "Tanaka Ao",
            "João Pedro Loureiro da Costa",
            "Alex Palmer",
            "Cole Palmer",
        ],
        "web_name": ["A.Silva", "Hill", "Joelinton", "Tanaka", "Costinha",
                     "A.Palmer", "Palmer"],
        "Team": ["BOU", "BOU", "NEW", "LEE", "BHA", "IPS", "CHE"],
        "Position": ["D", "D", "M", "M", "D", "G", "M"],
        "Player_ID": [566, 60, 458, 345, 119, 401, 402],
        "form": [3.1, 2.0, 4.4, 1.5, 0.9, 0.2, 7.7],
        "points_per_game": [3.0, 2.2, 4.0, 1.1, 1.0, 0.5, 6.9],
    })


class TestTheReportedBug:
    def test_a_long_legal_name_resolves_from_the_common_name(self):
        """The exact join's blind spot, and the whole reason this exists."""
        pool = pd.DataFrame({
            "Player": ["Antonio Silva", "James Hill"],
            "Team": ["BOU", "BOU"], "Position": ["D", "D"],
            "Player_ID": [np.nan, 60.0],
        })
        out = attach_reference_ids(pool, _ref())
        assert out.loc[0, "Player_ID"] == 566, (
            "Rotowire publishes 'Antonio Silva' and FPL the full legal name; "
            "an exact join misses him and he loses every id-keyed signal"
        )
        assert out.loc[1, "Player_ID"] == 60, "an exact match must be untouched"

    def test_the_two_players_end_up_distinguished(self):
        """Hill and Silva: same club, same position, 0.02 points apart."""
        pool = pd.DataFrame({
            "Player": ["Antonio Silva", "James Hill"],
            "Team": ["BOU", "BOU"], "Position": ["D", "D"],
            "Player_ID": [np.nan, np.nan],
        })
        out = attach_reference_ids(pool, _ref())
        assert out["Player_ID"].tolist() == [566.0, 60.0]
        assert out["Player_ID"].nunique() == 2


class TestWhatItMustNotDo:
    def test_an_existing_id_is_never_overwritten(self):
        pool = pd.DataFrame({
            "Player": ["Antonio Silva"], "Team": ["BOU"], "Position": ["D"],
            "Player_ID": [999.0],
        })
        out = attach_reference_ids(pool, _ref())
        assert out.loc[0, "Player_ID"] == 999.0

    def test_a_shared_surname_does_not_cross_positions(self):
        """The Alex/Cole Palmer rule: a backup GK must not take an elite MID."""
        pool = pd.DataFrame({
            "Player": ["Palmer"], "Team": ["IPS"], "Position": ["G"],
            "Player_ID": [np.nan],
        })
        out = attach_reference_ids(pool, _ref())
        got = out.loc[0, "Player_ID"]
        assert pd.isna(got) or got == 401, (
            "a goalkeeper must never be handed Cole Palmer's id"
        )

    def test_one_reference_row_is_never_shared(self):
        """Two players cannot both be the same element."""
        pool = pd.DataFrame({
            "Player": ["Antonio Silva", "Silva"],
            "Team": ["BOU", "BOU"], "Position": ["D", "D"],
            "Player_ID": [np.nan, np.nan],
        })
        out = attach_reference_ids(pool, _ref())
        ids = out["Player_ID"].dropna()
        assert ids.nunique() == len(ids), "an element id was claimed twice"

    def test_a_wrong_club_does_not_match(self):
        pool = pd.DataFrame({
            "Player": ["Antonio Silva"], "Team": ["ARS"], "Position": ["D"],
            "Player_ID": [np.nan],
        })
        out = attach_reference_ids(pool, _ref())
        assert pd.isna(out.loc[0, "Player_ID"])


class TestTheOtherTiers:
    def test_a_reversed_name_resolves(self):
        """Rotowire "Ao Tanaka" against FPL "Tanaka Ao" -- token sets are
        unordered, which is why tier 4 covers this for free."""
        pool = pd.DataFrame({
            "Player": ["Ao Tanaka"], "Team": ["LEE"], "Position": ["M"],
            "Player_ID": [np.nan],
        })
        out = attach_reference_ids(pool, _ref())
        assert out.loc[0, "Player_ID"] == 345

    def test_a_nickname_resolves_through_web_name(self):
        """"Costinha" shares no token with "João Pedro Loureiro da Costa", so
        only the exact (web_name, team) tier can reach him."""
        pool = pd.DataFrame({
            "Player": ["Costinha"], "Team": ["BHA"], "Position": ["D"],
            "Player_ID": [np.nan],
        })
        out = attach_reference_ids(pool, _ref(), ref_web_col="web_name")
        assert out.loc[0, "Player_ID"] == 119


class TestItFillsTheOtherSignalsToo:
    def test_form_and_ppg_come_with_the_id(self):
        """These are read straight off the same matched row, and a player whose
        form reads NaN is scored as though he had none."""
        pool = pd.DataFrame({
            "Player": ["Antonio Silva"], "Team": ["BOU"], "Position": ["D"],
            "Player_ID": [np.nan], "form": [np.nan],
            "points_per_game": [np.nan],
        })
        out = attach_reference_ids(
            pool, _ref(), fill_cols=("Player_ID", "form", "points_per_game"))
        assert out.loc[0, "form"] == 3.1
        assert out.loc[0, "points_per_game"] == 3.0


class TestDegradedInputs:
    def test_an_empty_pool_is_returned_unchanged(self):
        out = attach_reference_ids(pd.DataFrame(), _ref())
        assert out.empty

    def test_an_empty_reference_changes_nothing(self):
        pool = pd.DataFrame({
            "Player": ["Antonio Silva"], "Team": ["BOU"], "Position": ["D"],
            "Player_ID": [np.nan],
        })
        out = attach_reference_ids(pool, pd.DataFrame())
        assert pd.isna(out.loc[0, "Player_ID"])

    def test_a_missing_fill_column_is_skipped(self):
        pool = pd.DataFrame({
            "Player": ["Antonio Silva"], "Team": ["BOU"], "Position": ["D"],
            "Player_ID": [np.nan],
        })
        out = attach_reference_ids(pool, _ref(), fill_cols=("Player_ID", "nope"))
        assert out.loc[0, "Player_ID"] == 566
