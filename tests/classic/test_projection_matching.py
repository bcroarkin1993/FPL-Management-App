"""Attaching Rotowire's weekly projection to the Classic player pool.

What this replaced scanned the whole projections frame per player, scored every
row with `fuzz.ratio` and accepted any hit at 60, with team and position
contributing a +15 nudge rather than scoping the search. Nothing stopped one
reference row being claimed over and over.

Measured live against a 659-player pool and Rotowire's 220 rows: **323 players
were given a projection from a 220-row table**, 172 of them wearing another
player's numbers. David Raya (ARS, GK) shared a row with Rayan, Allan and Gray;
Ødegaard with Merino, Martinelli and Nørgaard; eight goalkeepers shared one row.
Every value on screen was plausible, which is why it went unnoticed -- and
`Projected_Points` feeds the suggestion cards, the sanity veto and the blend.
"""

import pandas as pd

from scripts.classic.transfers import _add_projections


def _pool(rows):
    return pd.DataFrame(rows)


def _projections(rows):
    return pd.DataFrame(rows)


class TestOneRowOnePlayer:
    def test_a_reference_row_is_never_shared(self):
        """The live failure: similar surnames all claiming one row."""
        pool = _pool([
            {"Player": "Raya", "Full Name": "David Raya Martin",
             "Team": "ARS", "Position": "G"},
            {"Player": "Rayan", "Full Name": "Rayan Ait-Nouri",
             "Team": "MCI", "Position": "D"},
            {"Player": "Gray", "Full Name": "Archie Gray",
             "Team": "TOT", "Position": "D"},
        ])
        proj = _projections([
            {"Player": "David Raya", "Team": "ARS", "Position": "G",
             "Points": 4.14, "Pos Rank": 2},
        ])
        out = _add_projections(pool, proj)
        assert out["Projected_Points"].notna().sum() == 1
        assert out.loc[0, "Projected_Points"] == 4.14

    def test_never_more_matches_than_reference_rows(self):
        pool = _pool([
            {"Player": "P%d" % i, "Full Name": "Player Number %d" % i,
             "Team": "ARS", "Position": "M"} for i in range(20)
        ])
        proj = _projections([
            {"Player": "Player Number 3", "Team": "ARS", "Position": "M",
             "Points": 5.0, "Pos Rank": 1},
        ])
        out = _add_projections(pool, proj)
        assert out["Projected_Points"].notna().sum() <= len(proj)


class TestMatchesOnTheFullName:
    def test_the_bootstrap_legal_name_reaches_rotowires_common_name(self):
        """FPL files Bruno Fernandes as "Bruno Borges Fernandes"."""
        pool = _pool([{"Player": "B.Fernandes",
                       "Full Name": "Bruno Borges Fernandes",
                       "Team": "MUN", "Position": "M"}])
        proj = _projections([{"Player": "Bruno Fernandes", "Team": "MUN",
                              "Position": "M", "Points": 6.05, "Pos Rank": 2}])
        out = _add_projections(pool, proj)
        assert out.loc[0, "Projected_Points"] == 6.05

    def test_a_mononym_resolves_off_the_short_name(self):
        pool = _pool([{"Player": "Gabriel",
                       "Full Name": "Gabriel dos Santos Magalhaes",
                       "Team": "ARS", "Position": "D"}])
        proj = _projections([{"Player": "Gabriel", "Team": "ARS",
                              "Position": "D", "Points": 4.9, "Pos Rank": 3}])
        out = _add_projections(pool, proj)
        assert out.loc[0, "Projected_Points"] == 4.9


class TestPositionScoping:
    def test_a_goalkeeper_does_not_take_a_midfielders_row(self):
        """Alex Palmer (GK) wearing Cole Palmer's (MID) numbers, restated."""
        pool = _pool([{"Player": "A.Palmer", "Full Name": "Alex Palmer",
                       "Team": "CHE", "Position": "G"}])
        proj = _projections([{"Player": "Cole Palmer", "Team": "CHE",
                              "Position": "M", "Points": 6.5, "Pos Rank": 1}])
        out = _add_projections(pool, proj)
        assert pd.isna(out.loc[0, "Projected_Points"])


class TestAbsence:
    def test_an_unlisted_player_gets_no_projection(self):
        """Rotowire lists 20 clubs x 11. Absence is the "not starting" signal,
        so a blank here is correct -- inventing a number is the error."""
        pool = _pool([{"Player": "Benchwarmer", "Full Name": "Some Reserve",
                       "Team": "ARS", "Position": "M"}])
        proj = _projections([{"Player": "Declan Rice", "Team": "ARS",
                              "Position": "M", "Points": 5.0, "Pos Rank": 4}])
        out = _add_projections(pool, proj)
        assert pd.isna(out.loc[0, "Projected_Points"])

    def test_an_empty_projections_frame_leaves_the_columns_present(self):
        pool = _pool([{"Player": "X", "Full Name": "X Y", "Team": "ARS",
                       "Position": "M"}])
        out = _add_projections(pool, pd.DataFrame())
        assert "Projected_Points" in out.columns and "Pos_Rank" in out.columns

    def test_a_frame_without_full_name_still_matches(self):
        """Not every caller carries one; it falls back to the short name."""
        pool = _pool([{"Player": "Haaland", "Team": "MCI", "Position": "F"}])
        proj = _projections([{"Player": "Erling Haaland", "Team": "MCI",
                              "Position": "F", "Points": 7.02, "Pos Rank": 1}])
        out = _add_projections(pool, proj)
        assert out.loc[0, "Projected_Points"] == 7.02
