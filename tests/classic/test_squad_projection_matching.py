"""Every Classic squad page attaches Rotowire's weekly numbers the same way.

`classic/transfers.py` was fixed to use the shared tiered matcher and the other
four pages were not, so the bug it was fixed for stayed live on them. Reported
from the app: **the Arsenal goalkeeper rendered as "Rayan", on 4.72 points,
against Bournemouth's fixture.**

Two independent faults produced that one card, and both are pinned here:

* The ladder accepted `fuzz.ratio("raya", "rayan") == 89` while David Raya's
  own Rotowire row scored 74 on his full legal name -- and nothing stopped one
  reference row being claimed by several players, so the better score simply
  won.
* Fixture Projections then **overwrote the player's name with the matched
  source's name**, "matching Draft behavior". That is what turned a wrong
  number into a wrong person.

`tests/classic/test_projection_matching.py` covers the same matcher through the
Transfers callsite; this file covers the two display pages that had drifted,
and the AST guard that found the other two -- the Free Hit and Wildcard pools,
where a phantom projection does not merely render wrongly but *buys a player*.
"""

import ast
import pathlib

import pandas as pd

from scripts.classic.fixture_projections import (
    _add_projections_to_squad as fixture_attach,
    _build_squad_dataframe as fixture_squad,
)
from scripts.classic.team_analysis import (
    _add_projections_to_squad as analysis_attach,
    _build_squad_dataframe as analysis_squad,
)

REPO = pathlib.Path(__file__).resolve().parents[2]

#: The live Rotowire GW7 rows for the reported bug, verbatim.
RAYA_ROWS = pd.DataFrame([
    {"Player": "Rayan", "Team": "BOU", "Position": "M",
     "Points": 4.72, "Pos Rank": 29, "Matchup": "BOU v. SUN"},
    {"Player": "David Raya", "Team": "ARS", "Position": "G",
     "Points": 4.16, "Pos Rank": 1, "Matchup": "ARS at NFO"},
])

#: The bootstrap as these pages read it: a full legal name, a web name, and the
#: tilde that made João Pedro hard to match on his rendered name.
BOOTSTRAP = {
    "teams": [{"id": 1, "short_name": "ARS"}, {"id": 2, "short_name": "BOU"},
              {"id": 3, "short_name": "CHE"}],
    "elements": [
        {"id": 1, "first_name": "David", "second_name": "Raya Martín",
         "web_name": "Raya", "team": 1, "element_type": 1},
        {"id": 2, "first_name": "Rayan", "second_name": "Vitor Simplício Rocha",
         "web_name": "Rayan", "team": 2, "element_type": 3},
        {"id": 3, "first_name": "João Pedro",
         "second_name": "Junqueira de Jesus", "web_name": "João Pedro",
         "team": 3, "element_type": 4},
    ],
}


def _picks(*element_ids):
    return [{"element": e, "position": i + 1, "multiplier": 1}
            for i, e in enumerate(element_ids)]


class TestFixtureProjections:
    def _squad(self, projections=RAYA_ROWS):
        return fixture_attach(fixture_squad(_picks(1, 2), BOOTSTRAP),
                              projections.copy())

    def test_raya_gets_rayas_row_and_rayan_gets_rayans(self):
        squad = self._squad().set_index("element_id")
        assert squad.at[1, "Points"] == 4.16
        assert squad.at[1, "Pos Rank"] == 1
        assert squad.at[1, "Matchup"] == "ARS at NFO"
        assert squad.at[2, "Points"] == 4.72

    def test_the_players_name_is_not_replaced_by_the_sources(self):
        """The reported symptom. A match is a number, never a rename."""
        squad = self._squad().set_index("element_id")
        assert squad.at[1, "Display_Name"] == "David Raya"
        assert squad.at[1, "Player"] == "David Raya Martín"
        assert "Rayan" not in str(squad.at[1, "Display_Name"])

    def test_a_tilde_in_the_legal_name_is_no_obstacle(self):
        """Matching normalises accents, so the frame keeps the real spelling."""
        squad = fixture_attach(
            fixture_squad(_picks(3), BOOTSTRAP),
            pd.DataFrame([{"Player": "Joao Pedro", "Team": "CHE",
                           "Position": "F", "Points": 5.15, "Pos Rank": 6,
                           "Matchup": "CHE at EVE"}]),
        ).set_index("element_id")
        assert squad.at[3, "Points"] == 5.15
        assert squad.at[3, "Display_Name"] == "João Pedro"

    def test_an_unlisted_player_keeps_his_own_name_and_gets_no_number(self):
        """Rotowire lists 20 clubs x 11, so absence is the "not starting"
        signal -- and a blank matchup must render as nothing, never as a
        confident claim about a fixture."""
        squad = fixture_attach(fixture_squad(_picks(3), BOOTSTRAP),
                               RAYA_ROWS.copy()).set_index("element_id")
        assert pd.isna(squad.at[3, "Points"])
        assert squad.at[3, "Matchup"] == ""
        assert squad.at[3, "Display_Name"] == "João Pedro"

    def test_an_empty_projections_frame_leaves_every_column_present(self):
        squad = fixture_attach(fixture_squad(_picks(1, 2), BOOTSTRAP),
                               pd.DataFrame())
        for col in ("Points", "Pos Rank", "Matchup"):
            assert col in squad.columns
        assert squad["Points"].isna().all()


class TestTeamAnalysis:
    """The loosest of the three ladders: it scored the *web* name at a
    threshold of 60 with no team or position scoping at all."""

    def _squad(self):
        return analysis_attach(analysis_squad(_picks(1, 2), BOOTSTRAP),
                               RAYA_ROWS.copy()).set_index("element_id")

    def test_raya_gets_rayas_row(self):
        squad = self._squad()
        assert squad.at[1, "Points"] == 4.16
        assert squad.at[1, "Pos Rank"] == 1

    def test_rayan_keeps_his_own(self):
        assert self._squad().at[2, "Points"] == 4.72

    def test_the_rendered_name_is_the_common_name(self):
        """`web_name` alone is often too terse to stand on a card."""
        assert self._squad().at[1, "Display_Name"] == "David Raya"


class TestNoFourthLadder:
    def test_no_page_scores_a_projection_name_with_fuzz_ratio(self):
        """The grep guard. Five callsites had each grown their own copy.

        Walks the AST rather than the text, so the prose in the docstrings
        that record what each ladder did -- which is the part worth keeping --
        does not read as another ladder.

        `player_matching` owns the fuzzy tier (inside `ReferenceMatcher`, where
        it is scoped by team and position and requires a shared whole token);
        `transfer_risk*` and the odds modules match headline prose, which has
        no reference frame to claim rows from.
        """
        owners = {
            "scripts/common/player_matching.py",
            "scripts/common/name_matching.py",
            "scripts/common/transfer_risk.py",
            "scripts/common/transfer_risk_app.py",
            "scripts/common/transfer_feeds.py",
            "scripts/common/odds_feeds.py",
            "scripts/common/transfer_odds.py",
        }
        offenders = []
        for path in sorted((REPO / "scripts").rglob("*.py")):
            rel = str(path.relative_to(REPO))
            if rel in owners:
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in ("fuzz", "process")):
                    offenders.append("%s:%d: %s.%s"
                                     % (rel, node.lineno, node.value.id, node.attr))
        assert offenders == [], (
            "Attach a projection through analytics.attach_weekly_projections / "
            "_claim_reference_rows instead of scoring names at the page: %s"
            % offenders)
