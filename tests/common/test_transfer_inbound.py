"""Unit tests for the inbound side of the transfer model.

The outbound model asks "will he still be here?". This asks the two questions an
arrival raises: who is coming, and whose minutes does he take.

This path is noisier than the outbound one by construction -- a per-club query is
about the club, so the arriving player's name has to be pulled out of prose -- so
these tests lean on real headline shapes rather than invented ones.
"""

from datetime import date

import pandas as pd

from scripts.common.transfer_risk import (
    INCUMBENT_TOP_SHARE,
    MAX_MINUTES_IMPACT,
    MINUTES_FLOOR,
    STATUS_AT_RISK,
    TIER_A,
    TIER_B,
    TIER_C,
    apply_minutes_competition,
    build_inbound_watchlist,
    classify_headline,
    classify_signing,
    extract_signing,
    parse_fee,
    parse_fee_for_player,
    position_from_text,
)

PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
    "Coventry City", "Crystal Palace", "Everton", "Fulham", "Hull City",
    "Ipswich Town", "Leeds", "Liverpool", "Man City", "Man Utd", "Newcastle",
    "Nott'm Forest", "Sunderland", "Spurs",
]

TODAY = date(2026, 8, 30)
FRESH = "Fri, 28 Aug 2026 10:00:00 GMT"
STALE = "Mon, 02 Mar 2026 10:00:00 GMT"


def _news(rows):
    return pd.DataFrame(
        [{"Club": c, "Headline": h, "URL": "", "Published": p, "Source": s}
         for c, h, p, s in rows],
        columns=["Club", "Headline", "URL", "Published", "Source"],
    )


class TestParseFee:
    def test_reads_the_common_magnitudes(self):
        assert parse_fee("Chelsea agree £51m deal for striker") == 51.0
        assert parse_fee("Arsenal bid €40 million") == 40.0
        assert parse_fee("a deal worth up to £51.5m") == 51.5

    def test_bare_amount_is_not_a_fee(self):
        """Transfer fees are always written with a magnitude."""
        assert parse_fee("£10 boots") is None
        assert parse_fee("tickets from £45") is None

    def test_takeover_money_is_not_a_transfer_fee(self):
        assert parse_fee("£1.2bn takeover completed") is None

    def test_largest_fee_wins(self):
        assert parse_fee("bid of £30m rejected, £45m accepted") == 45.0

    def test_no_fee(self):
        assert parse_fee("Villa sign striker on a free") is None
        assert parse_fee(None) is None


class TestFeeAttribution:
    HEADLINE = "Liverpool agree £123m Barcola deal as Gakpo decides to join Man City"

    def test_fee_goes_to_the_player_it_sits_beside(self):
        assert parse_fee_for_player(self.HEADLINE, "Bradley Barcola", {"gakpo"}) == 123.0

    def test_other_players_fee_is_not_claimed(self):
        """Same failure as reading a player's own club as his destination, and it
        matters more: fee is the evidence for how big a role a signing takes."""
        assert parse_fee_for_player(self.HEADLINE, "Cody Gakpo", {"barcola"}) is None

    def test_player_absent_from_the_headline_gets_nothing(self):
        assert parse_fee_for_player(self.HEADLINE, "Erling Haaland", set()) is None

    def test_distant_fee_is_not_attributed(self):
        far = ("Chelsea complete £80m rebuild of their defence over the summer "
               "window in a busy month before finally turning to Wharton")
        assert parse_fee_for_player(far, "Adam Wharton", set()) is None


class TestPositionFromText:
    def test_winger_is_a_midfielder(self):
        """FPL classifies the overwhelming majority of them that way."""
        assert position_from_text("Palace sign winger from Union SG") == "M"

    def test_longest_keyword_wins(self):
        """'attacking midfielder' must not be read as a forward."""
        assert position_from_text("Arsenal land attacking midfielder") == "M"

    def test_defender_and_keeper(self):
        assert position_from_text("Forest complete signing of centre back") == "D"
        assert position_from_text("Spurs sign goalkeeper") == "G"

    def test_no_description(self):
        assert position_from_text("Spurs sign someone") is None


class TestClassifySigning:
    def test_completed_signing_is_tier_a(self):
        """The regression this classifier exists for.

        The exit-side classifier is written for the question "is he leaving", so
        the single strongest inbound sentence there is scores nothing in it -- and
        a confirmed arrival was dropped from the watchlist while a rumour lived.
        """
        headline = "Aston Villa complete signing of striker Nicolas Jackson for £51m"
        assert classify_signing(headline) == TIER_A
        assert classify_headline(headline) == 0.0

    def test_active_pursuit_is_tier_b(self):
        assert classify_signing("Chelsea in talks to sign midfielder Adam Wharton") == TIER_B

    def test_interest_is_tier_c(self):
        assert classify_signing("Arsenal linked with move for Morgan Rogers") == TIER_C

    def test_denial_is_capped(self):
        """"Chelsea rule out move" must not score on the word "move"."""
        assert classify_signing("Chelsea rule out January move for Wharton") <= TIER_C

    def test_collapsed_deal_is_capped(self):
        assert classify_signing("Villa deal for Jackson collapses") <= TIER_C

    def test_nothing_transfer_shaped(self):
        assert classify_signing("Aston Villa training ground reopens") == 0.0


class TestExtractSigning:
    def test_buyer_comes_from_the_sentence_not_the_feed(self):
        """A raid on a club surfaces under *that* club's query.

        Trusting the queried club would discount the selling club's squad, which
        is exactly backwards.
        """
        got = extract_signing(
            "Nottingham Forest sign Marc Guehi from Crystal Palace",
            PL_TEAMS, queried_club="Crystal Palace")
        assert got is not None
        player, club, _pos, _fee = got
        assert player == "Marc Guehi"
        assert "Forest" in club

    def test_leading_subject_form(self):
        got = extract_signing(
            "Anan Khalaili transfer news: Crystal Palace sign winger from "
            "Union Saint-Gilloise", PL_TEAMS)
        assert got == ("Anan Khalaili", "Crystal Palace", "M", None)

    def test_player_between_club_prefix_and_verb(self):
        got = extract_signing(
            "Crystal Palace transfer news: Axel Disasi signs on loan from Chelsea",
            PL_TEAMS)
        assert got is not None
        assert got[0] == "Axel Disasi"
        assert got[1] == "Crystal Palace"

    def test_name_after_the_verb(self):
        got = extract_signing(
            "Chelsea complete signing of Nicolas Jackson for £51m", PL_TEAMS)
        assert got == ("Nicolas Jackson", "Chelsea", None, 51.0)

    def test_joins_form_states_both_ends(self):
        got = extract_signing("Marc Guehi joins Liverpool in £35m deal", PL_TEAMS)
        assert got == ("Marc Guehi", "Liverpool", None, 35.0)

    def test_no_signing_verb_is_not_a_signing(self):
        assert extract_signing(
            "Nottingham Forest attempt double Crystal Palace raid", PL_TEAMS) is None

    def test_headline_boilerplate_is_not_a_player(self):
        """Without a stopword list the extractor signs "Transfer News"."""
        assert extract_signing(
            "Premier League transfer news: latest gossip and rumours", PL_TEAMS) is None

    def test_a_player_staying_is_not_an_arrival(self):
        assert extract_signing(
            "Chelsea star signs new deal and will stay at the club", PL_TEAMS) is None


class TestBuildInboundWatchlist:
    def test_corroborated_signing_is_listed(self):
        news = _news([
            ("Aston Villa", "Aston Villa complete signing of striker Nicolas Jackson for £51m", FRESH, "BBC"),
            ("Aston Villa", "Aston Villa agree deal to sign striker Nicolas Jackson", FRESH, "Sky Sports"),
        ])
        out = build_inbound_watchlist(news, PL_TEAMS, today=TODAY)
        assert len(out) == 1
        row = out.iloc[0]
        assert row["Player"] == "Nicolas Jackson"
        assert row["Club"] == "Aston Villa"
        assert row["Position"] == "F"
        assert row["Fee"] == 51.0
        assert row["Outlets"] == 2
        assert 0.0 < row["Confidence"] <= 1.0

    def test_single_outlet_is_not_a_signing(self):
        """Same corroboration gate as the outbound side: one paper is speculation."""
        news = _news([
            ("Chelsea", "Chelsea in talks to sign midfielder Adam Wharton", FRESH, "The Athletic"),
        ])
        assert build_inbound_watchlist(news, PL_TEAMS, today=TODAY).empty

    def test_stale_news_is_ignored(self):
        news = _news([
            ("Aston Villa", "Aston Villa complete signing of striker Nicolas Jackson", STALE, "BBC"),
            ("Aston Villa", "Aston Villa agree deal to sign striker Nicolas Jackson", STALE, "Sky Sports"),
        ])
        assert build_inbound_watchlist(news, PL_TEAMS, today=TODAY).empty

    def test_known_player_position_beats_inference(self):
        """An intra-PL mover is already in the game, so his position is known."""
        news = _news([
            ("Liverpool", "Marc Guehi joins Liverpool in £35m deal", FRESH, "BBC"),
            ("Liverpool", "Liverpool complete signing of Marc Guehi", FRESH, "Sky Sports"),
        ])
        known = pd.DataFrame([{"Player": "Marc Guehi", "Position": "D"}])
        out = build_inbound_watchlist(news, PL_TEAMS, today=TODAY, known_players=known)
        assert out.iloc[0]["Position"] == "D"

    def test_empty_and_missing_inputs_are_safe(self):
        assert build_inbound_watchlist(None, PL_TEAMS, today=TODAY).empty
        assert build_inbound_watchlist(_news([]), PL_TEAMS, today=TODAY).empty


class TestApplyMinutesCompetition:
    def _players(self):
        return pd.DataFrame([
            {"Player": "Ollie Watkins", "Team": "AVL", "Position": "F",
             "Points": 180.0, "Transfer_Status": STATUS_AT_RISK},
            {"Player": "Jhon Duran", "Team": "AVL", "Position": "F",
             "Points": 90.0, "Transfer_Status": ""},
            {"Player": "Donyell Malen", "Team": "AVL", "Position": "F",
             "Points": 60.0, "Transfer_Status": ""},
            {"Player": "Morgan Rogers", "Team": "AVL", "Position": "M",
             "Points": 150.0, "Transfer_Status": ""},
            {"Player": "Cole Palmer", "Team": "CHE", "Position": "M",
             "Points": 200.0, "Transfer_Status": ""},
        ])

    def _arrivals(self):
        return pd.DataFrame([{
            "Player": "Nicolas Jackson", "Club": "Aston Villa", "Position": "F",
            "Fee": 51.0, "Outlets": 3, "Confidence": 0.8,
            "Headline": "Aston Villa complete signing of Nicolas Jackson",
        }])

    def _result(self):
        return apply_minutes_competition(self._players(), self._arrivals()).set_index("Player")

    def test_same_position_incumbents_are_discounted(self):
        out = self._result()
        assert out.at["Jhon Duran", "Minutes_Mult"] < 1.0
        assert out.at["Donyell Malen", "Minutes_Mult"] < 1.0
        assert "Nicolas Jackson" in out.at["Jhon Duran", "Competition"]

    def test_first_choice_absorbs_less_than_the_players_behind_him(self):
        out = self._result()
        assert out.at["Jhon Duran", "Minutes_Mult"] > out.at["Donyell Malen", "Minutes_Mult"]

    def test_a_player_leaving_is_exempt(self):
        """Jackson arrives *because* Watkins is going -- charging Watkins for his
        own replacement double-counts one move."""
        assert self._result().at["Ollie Watkins", "Minutes_Mult"] == 1.0

    def test_other_positions_and_clubs_are_untouched(self):
        out = self._result()
        assert out.at["Morgan Rogers", "Minutes_Mult"] == 1.0
        assert out.at["Cole Palmer", "Minutes_Mult"] == 1.0

    def test_discount_stays_within_its_cap(self):
        """A signing is a tiebreak between similar players, never a verdict."""
        arrivals = self._arrivals()
        arrivals.loc[0, "Confidence"] = 1.0
        arrivals.loc[0, "Fee"] = 200.0
        out = apply_minutes_competition(self._players(), arrivals).set_index("Player")
        assert out["Minutes_Mult"].min() >= MINUTES_FLOOR
        assert MINUTES_FLOOR == 1.0 - MAX_MINUTES_IMPACT
        # Even at maximum threat the first choice keeps most of his value.
        expected_top = 1.0 - MAX_MINUTES_IMPACT * INCUMBENT_TOP_SHARE
        assert out.at["Jhon Duran", "Minutes_Mult"] >= expected_top - 1e-9

    def test_no_arrivals_leaves_everyone_alone(self):
        out = apply_minutes_competition(self._players(), pd.DataFrame())
        assert (out["Minutes_Mult"] == 1.0).all()
        assert (out["Minutes_Note"] == "").all()

    def test_arrival_without_a_position_is_not_applied(self):
        """Without both club and position there is nobody to attribute it to."""
        arrivals = self._arrivals()
        arrivals.loc[0, "Position"] = None
        out = apply_minutes_competition(self._players(), arrivals)
        assert (out["Minutes_Mult"] == 1.0).all()

    def test_missing_columns_are_safe(self):
        bare = pd.DataFrame([{"Player": "Someone", "Points": 10.0}])
        out = apply_minutes_competition(bare, self._arrivals())
        assert (out["Minutes_Mult"] == 1.0).all()
