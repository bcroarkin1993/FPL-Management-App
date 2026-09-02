"""Unit tests for how the Draft Helper board combines its two discounts.

The board answers "who should I draft". Two separate signals reduce a player's
value: he may leave the league (outbound), and his club may be signing someone to
take his minutes (inbound). They are applied together but toggled separately, so
the column and the ordering must never disagree about what was actually applied.
"""

import pandas as pd

from scripts.common.transfer_risk import MAX_MINUTES_IMPACT, STATUS_AT_RISK
from scripts.common.transfer_risk_app import build_inbound_competition
from scripts.draft.draft_helper import _merge_risk

PL_TEAMS = ["Aston Villa", "Arsenal", "Chelsea", "Liverpool", "Man City"]
FRESH = "Fri, 28 Aug 2026 10:00:00 GMT"


def _rankings():
    return pd.DataFrame({
        "Rank": [1, 2, 3],
        "Player": ["Ollie Watkins", "Jhon Duran", "Bukayo Saka"],
        "Team": ["AVL", "AVL", "ARS"],
        "Position": ["F", "F", "M"],
        "Points": [100.0, 80.0, 120.0],
    })


def _scored(risk_mult=0.5):
    return pd.DataFrame({
        "Player": ["Ollie Watkins", "Jhon Duran", "Bukayo Saka"],
        "Team": ["AVL", "AVL", "ARS"],
        "Position": ["F", "F", "M"],
        "Transfer_Risk": [0.6, 0.0, 0.0],
        "Transfer_Mult": [risk_mult, 1.0, 1.0],
        "Transfer_Destination": ["Al Hilal", "", ""],
        "Transfer_Outlets": [4, 0, 0],
        "Transfer_Note": ["Al Hilal (4 outlets)", "", ""],
    })


def _minutes(mult=0.9):
    return pd.DataFrame({
        "Player": ["Ollie Watkins", "Jhon Duran", "Bukayo Saka"],
        "Team": ["AVL", "AVL", "ARS"],
        "Position": ["F", "F", "M"],
        "Minutes_Mult": [1.0, mult, 1.0],
        "Competition": ["", "Nicolas Jackson (£51m)", ""],
    })


class TestMergeRisk:
    def test_both_discounts_apply_together(self):
        out = _merge_risk(_rankings(), _scored(0.5), _minutes(0.9),
                          use_risk=True, use_minutes=True).set_index("Player")
        assert out.at["Ollie Watkins", "Adj Points"] == 50.0   # 100 * 0.5
        assert out.at["Jhon Duran", "Adj Points"] == 72.0      # 80 * 0.9
        assert out.at["Bukayo Saka", "Adj Points"] == 120.0

    def test_minutes_toggle_off_removes_only_that_factor(self):
        out = _merge_risk(_rankings(), _scored(0.5), _minutes(0.9),
                          use_risk=True, use_minutes=False).set_index("Player")
        assert out.at["Jhon Duran", "Adj Points"] == 80.0
        assert out.at["Ollie Watkins", "Adj Points"] == 50.0

    def test_risk_toggle_off_removes_only_that_factor(self):
        out = _merge_risk(_rankings(), _scored(0.5), _minutes(0.9),
                          use_risk=False, use_minutes=True).set_index("Player")
        assert out.at["Ollie Watkins", "Adj Points"] == 100.0
        assert out.at["Jhon Duran", "Adj Points"] == 72.0

    def test_both_off_is_the_raw_projection(self):
        out = _merge_risk(_rankings(), _scored(0.5), _minutes(0.9),
                          use_risk=False, use_minutes=False).set_index("Player")
        assert list(out["Adj Points"]) == [100.0, 80.0, 120.0]

    def test_unscanned_tail_fills_neutrally(self):
        """A player nobody checked is not a player with no news."""
        rankings = _rankings()
        rankings.loc[3] = {"Rank": 4, "Player": "Cole Palmer", "Team": "CHE",
                           "Position": "M", "Points": 150.0}
        out = _merge_risk(rankings, _scored(0.5), _minutes(0.9)).set_index("Player")
        assert out.at["Cole Palmer", "Transfer_Mult"] == 1.0
        assert out.at["Cole Palmer", "Minutes_Mult"] == 1.0
        assert out.at["Cole Palmer", "Adj Points"] == 150.0
        assert out.at["Cole Palmer", "Competition"] == ""

    def test_missing_minutes_frame_is_neutral(self):
        """The board renders before the club scan has ever run."""
        out = _merge_risk(_rankings(), _scored(0.5), None).set_index("Player")
        assert (out["Minutes_Mult"] == 1.0).all()
        assert out.at["Ollie Watkins", "Adj Points"] == 50.0


class TestBuildInboundCompetition:
    def _club_news(self):
        return pd.DataFrame([
            {"Club": "Aston Villa", "Headline": "Aston Villa complete signing of "
             "striker Nicolas Jackson for £51m", "URL": "", "Published": FRESH,
             "Source": "BBC"},
            {"Club": "Aston Villa", "Headline": "Aston Villa agree deal to sign "
             "striker Nicolas Jackson", "URL": "", "Published": FRESH,
             "Source": "Sky Sports"},
        ])

    def _pool(self):
        return pd.DataFrame({
            "Player": ["Ollie Watkins", "Jhon Duran", "Bukayo Saka"],
            "Team": ["AVL", "AVL", "ARS"],
            "Position": ["F", "F", "M"],
            "Points": [100.0, 80.0, 120.0],
            "Transfer_Status": [STATUS_AT_RISK, "", ""],
        })

    def test_arrival_discounts_the_incumbent(self):
        from datetime import date

        arrivals, out = build_inbound_competition(
            self._pool(), self._club_news(), PL_TEAMS, today=date(2026, 8, 30))
        assert len(arrivals) == 1
        assert arrivals.iloc[0]["Player"] == "Nicolas Jackson"
        out = out.set_index("Player")
        assert out.at["Jhon Duran", "Minutes_Mult"] < 1.0
        assert out.at["Bukayo Saka", "Minutes_Mult"] == 1.0

    def test_a_player_leaving_is_exempt(self):
        """Jackson arrives *because* Watkins is going — one move, charged once."""
        from datetime import date

        _arrivals, out = build_inbound_competition(
            self._pool(), self._club_news(), PL_TEAMS, today=date(2026, 8, 30))
        assert out.set_index("Player").at["Ollie Watkins", "Minutes_Mult"] == 1.0

    def test_discount_never_exceeds_its_cap(self):
        from datetime import date

        _arrivals, out = build_inbound_competition(
            self._pool(), self._club_news(), PL_TEAMS, today=date(2026, 8, 30))
        assert out["Minutes_Mult"].min() >= 1.0 - MAX_MINUTES_IMPACT

    def test_no_club_news_is_neutral(self):
        arrivals, out = build_inbound_competition(self._pool(), None, PL_TEAMS)
        assert arrivals.empty
        assert (out["Minutes_Mult"] == 1.0).all()

    def test_broken_input_degrades_instead_of_raising(self):
        """A noisy club feed must not take the draft board down."""
        junk = pd.DataFrame([{"Club": None, "Headline": None, "Published": None,
                              "Source": None}])
        arrivals, out = build_inbound_competition(self._pool(), junk, PL_TEAMS)
        assert arrivals.empty
        assert (out["Minutes_Mult"] == 1.0).all()
