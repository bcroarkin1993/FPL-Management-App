"""Unit tests for scripts/common/transfer_risk.py.

Every fixture here is real: the Watkins headlines are the ones Google News
actually returned on 29 Aug 2026, and the bootstrap news strings are copied
verbatim from live FPL data.  The bug this module exists to prevent was a top-32
draft pick who had already agreed a move to Saudi Arabia.
"""

from datetime import date

import pandas as pd
import pytest

from scripts.common.transfer_risk import (
    STATUS_DEPARTED,
    STATUS_MOVING_PL,
    TIER_A,
    TIER_C,
    TRANSFER_FLOOR,
    WEIGHT_INTRA_PL,
    WEIGHT_LEAVES_PL,
    attach_transfer_risk,
    build_ambiguous_tokens,
    classify_headline,
    headline_mentions_player,
    is_premier_league_club,
    next_window_close,
    parse_destination,
    resolve_from_bootstrap,
    score_headlines,
    team_code,
    transfer_exposure,
    transfer_multiplier,
)

# The 2026/27 Premier League, as the bootstrap spells it.  Note Leicester, Stoke
# and Leeds' absence/presence: membership is the entire basis of the destination
# weighting, so the list has to be the real one.
PL_TEAMS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton", "Chelsea",
    "Coventry City", "Crystal Palace", "Everton", "Fulham", "Hull City",
    "Ipswich Town", "Leeds", "Liverpool", "Man City", "Man Utd", "Newcastle",
    "Nott'm Forest", "Spurs", "Sunderland",
]

TODAY = date(2026, 8, 29)

WATKINS_HEADLINES = [
    {"Headline": "Aston Villa Transfer Latest: Ollie Watkins undergoes medical at Al Hilal",
     "Source": "Sky Sports", "Published": "Fri, 28 Aug 2026 18:21:08 GMT"},
    {"Headline": "Al-Hilal close to Ollie Watkins deal",
     "Source": "Transfermarkt", "Published": "Thu, 27 Aug 2026 08:35:00 GMT"},
    {"Headline": "Al Hilal agree £51million Ollie Watkins transfer",
     "Source": "The Sun", "Published": "Wed, 26 Aug 2026 18:55:00 GMT"},
    {"Headline": "Al-Hilal agree deal to sign Aston Villa striker Ollie Watkins",
     "Source": "SANA", "Published": "Thu, 27 Aug 2026 15:01:35 GMT"},
    {"Headline": "Transfer latest: Emery sheds tears over Watkins exit",
     "Source": "The Telegraph", "Published": "Fri, 28 Aug 2026 10:17:00 GMT"},
    # Returned by the same query but about nobody relevant.
    {"Headline": "Exeter City transfer latest as things will happen but nothing close",
     "Source": "Devon Live", "Published": "Thu, 27 Aug 2026 08:20:14 GMT"},
]


class TestClassifyHeadline:
    def test_medical_is_top_tier(self):
        assert classify_headline("Watkins undergoes medical at Al Hilal") == TIER_A

    def test_agreed_fee_is_top_tier_without_the_word_deal(self):
        assert classify_headline("Al Hilal agree £51million Watkins transfer") == TIER_A

    def test_talks_are_middle_tier(self):
        assert 0 < classify_headline("Villa in talks over Watkins exit") < TIER_A

    def test_linked_is_bottom_tier(self):
        assert classify_headline("Arsenal linked with Watkins") == TIER_C

    def test_denial_caps_at_bottom_tier(self):
        """'Rule out a sale' must not score as Tier A on the word 'sale'."""
        assert classify_headline("Villa rule out Watkins sale") <= TIER_C

    def test_new_contract_caps_the_score(self):
        assert classify_headline("Watkins signs new contract at Villa") <= TIER_C

    def test_irrelevant_headline_scores_nothing(self):
        assert classify_headline("Watkins scores twice against Everton") == 0.0

    def test_empty_input_is_safe(self):
        assert classify_headline("") == 0.0
        assert classify_headline(None) == 0.0


class TestHeadlineMentionsPlayer:
    def test_surname_alone_counts(self):
        """Headlines routinely drop the first name; requiring it lost real outlets."""
        assert headline_mentions_player("Mateta set to sign for Forest",
                                        "Jean-Philippe Mateta", "Crystal Palace")

    def test_unrelated_headline_is_rejected(self):
        assert not headline_mentions_player("Exeter City transfer latest",
                                            "Ollie Watkins", "Aston Villa")

    def test_mononym_needs_the_club_to_corroborate(self):
        """A bare 'Gabriel' collected every Gabriel in the league and discounted
        Arsenal's by 71% off other players' news."""
        ambiguous = build_ambiguous_tokens(["Gabriel", "Gabriel Martinelli"])
        assert not headline_mentions_player(
            "Gabriel linked with Saudi move", "Gabriel", "Arsenal", ambiguous)
        assert headline_mentions_player(
            "Arsenal defender Gabriel eyed by Al Hilal", "Gabriel", "Arsenal", ambiguous)

    def test_shared_surname_needs_corroboration(self):
        ambiguous = build_ambiguous_tokens(["Joao Pedro", "Pedro Neto"])
        assert not headline_mentions_player(
            "Pedro joins Real Madrid", "Joao Pedro", "Chelsea", ambiguous)


class TestDestinationWeighting:
    """The single most important distinction in the module."""

    def test_saudi_move_is_a_total_loss(self):
        _club, weight = parse_destination(
            "Watkins undergoes medical at Al Hilal", PL_TEAMS, exclude_team="Aston Villa")
        assert weight == WEIGHT_LEAVES_PL

    def test_hyphenated_saudi_club_still_matches(self):
        """Normalisation strips hyphens, so 'Al-Hilal' arrives as 'alhilal'."""
        club, weight = parse_destination("Al-Hilal agree deal for Watkins", PL_TEAMS)
        assert weight == WEIGHT_LEAVES_PL
        assert club

    def test_championship_move_is_as_bad_as_saudi(self):
        """The line is Premier League membership, not geography.  A season-long
        loan to Leicester scores you exactly as many points as Al-Hilal: zero."""
        _club, weight = parse_destination(
            "Has joined Leicester City on loan for the rest of the season", PL_TEAMS)
        assert weight == WEIGHT_LEAVES_PL

    def test_intra_pl_move_is_a_small_cost(self):
        _club, weight = parse_destination(
            "Mateta set to sign for Nottingham Forest", PL_TEAMS, exclude_team="Crystal Palace")
        assert weight == WEIGHT_INTRA_PL

    def test_players_own_club_is_never_the_destination(self):
        """'Al-Hilal agree deal to sign Aston Villa striker Watkins' names two
        clubs and only one is where he is going."""
        club, weight = parse_destination(
            "Al-Hilal agree deal to sign Aston Villa striker Ollie Watkins",
            PL_TEAMS, exclude_team="Aston Villa")
        assert weight == WEIGHT_LEAVES_PL
        assert "villa" not in str(club).lower()

    def test_own_club_excluded_by_short_code(self):
        """Frames carry short codes ('MCI'), the bootstrap carries names."""
        club, _w = parse_destination("Cherki stunning goal for Man City",
                                     PL_TEAMS, exclude_team="MCI")
        assert club is None

    def test_destination_capture_stops_at_connectives(self):
        """Unbounded, this swallowed the sentence: 'Tottenham amid Arsenal ...'."""
        club, weight = parse_destination(
            "Sandro Tonali joined Tottenham amid Arsenal interest", PL_TEAMS)
        assert club == "Tottenham"
        assert weight == WEIGHT_INTRA_PL

    def test_free_agent_has_left_the_league(self):
        club, weight = parse_destination(
            "has departed the club as a free agent.", PL_TEAMS)
        assert weight == WEIGHT_LEAVES_PL
        assert "free agent" in club.lower()

    def test_bootstrap_spelling_variants_are_premier_league(self):
        for name in ("Nottingham Forest", "Nott'm Forest", "Man Utd",
                     "Manchester United", "Spurs", "Tottenham Hotspur"):
            assert is_premier_league_club(name, PL_TEAMS), name

    def test_relegated_and_foreign_clubs_are_not(self):
        for name in ("Leicester City", "Stoke City", "Al Hilal", "Getafe",
                     "Bolton Wanderers"):
            assert not is_premier_league_club(name, PL_TEAMS), name

    def test_team_code_accepts_names_and_codes(self):
        assert team_code("Man City") == team_code("MCI") == "MCI"


class TestScoreHeadlines:
    def test_watkins_scores_high(self):
        risk, dest, weight, outlets, evidence, _fee = score_headlines(
            WATKINS_HEADLINES, "Ollie Watkins", "Aston Villa", PL_TEAMS, today=TODAY)
        assert risk >= 0.7, "the case this module exists for must score high"
        assert weight == WEIGHT_LEAVES_PL
        assert outlets >= 4
        assert evidence

    def test_unrelated_item_is_not_counted_as_an_outlet(self):
        _r, _d, _w, outlets, evidence, _fee = score_headlines(
            WATKINS_HEADLINES, "Ollie Watkins", "Aston Villa", PL_TEAMS, today=TODAY)
        assert not any("Devon Live" in e["Source"] for e in evidence)
        assert outlets <= 5

    def test_single_outlet_rumour_stays_low(self):
        risk, _d, _w, _o, _e, _fee = score_headlines(
            [{"Headline": "Arsenal linked with move for Morgan Rogers",
              "Source": "Daily Star", "Published": "Fri, 28 Aug 2026 10:00:00 GMT"}],
            "Morgan Rogers", "Aston Villa", PL_TEAMS, today=TODAY)
        assert risk < 0.10

    def test_corroboration_increases_risk(self):
        one = [{"Headline": "Watkins agrees deal to join Al Hilal", "Source": "A",
                "Published": "Fri, 28 Aug 2026 10:00:00 GMT"}]
        many = one + [dict(one[0], Source=s) for s in ("B", "C", "D", "E")]
        r_one, *_ = score_headlines(one, "Ollie Watkins", "Aston Villa", PL_TEAMS, today=TODAY)
        r_many, *_ = score_headlines(many, "Ollie Watkins", "Aston Villa", PL_TEAMS, today=TODAY)
        assert r_many > r_one

    def test_denial_produces_no_meaningful_risk(self):
        risk, _d, _w, _o, _e, _fee = score_headlines(
            [{"Headline": "Villa rule out Rogers sale, he is not for sale",
              "Source": "BBC", "Published": "Fri, 28 Aug 2026 10:00:00 GMT"}],
            "Morgan Rogers", "Aston Villa", PL_TEAMS, today=TODAY)
        assert risk < 0.10

    def test_stale_news_is_ignored(self):
        """A six-month-old rumour is what made the betting source useless."""
        risk, _d, _w, _o, _e, _fee = score_headlines(
            [{"Headline": "Rogers undergoes medical at Real Madrid", "Source": "AS",
              "Published": "Mon, 02 Mar 2026 10:00:00 GMT"}],
            "Morgan Rogers", "Aston Villa", PL_TEAMS, today=TODAY)
        assert risk == 0.0

    def test_no_news_is_no_risk(self):
        assert score_headlines([], "Erling Haaland", "Man City", PL_TEAMS, today=TODAY)[0] == 0.0


# Real Google News headlines for Bruno Fernandes, 29 Aug 2026. Galatasaray made
# an offer; United rejected it and opened contract talks. He never moved. Scored
# 0.33 risk and cost him a third of his season projection -- from rank 2 to 39 --
# purely because "offer" and "approach" were being read as departure evidence.
BRUNO_HEADLINES = [
    {"Headline": "Man Utd face Fernandes transfer challenge with 'offer prepared'",
     "Source": "London Evening Standard", "Published": "Thu, 27 Aug 2026 10:00:00 GMT"},
    {"Headline": "Manchester United in contract talks with Bruno Fernandes as Galatasaray make lucrative offer",
     "Source": "The New York Times", "Published": "Wed, 26 Aug 2026 10:00:00 GMT"},
    {"Headline": "Man United make Bruno Fernandes transfer stance clear amid 'audacious' offer",
     "Source": "Manchester Evening News", "Published": "Wed, 26 Aug 2026 10:00:00 GMT"},
    {"Headline": "Man Utd issue instant response to shock Bruno Fernandes transfer approach",
     "Source": "Daily Mirror", "Published": "Wed, 26 Aug 2026 10:00:00 GMT"},
    {"Headline": "Bruno Fernandes transfer news: Manchester United captain not for sale at any price despite Galatasaray interest",
     "Source": "Sky Sports", "Published": "Tue, 25 Aug 2026 10:00:00 GMT"},
    {"Headline": "Bruno Fernandes transfer news: Manchester United will reject any offer for captain amid Galatasaray interest",
     "Source": "BBC", "Published": "Tue, 25 Aug 2026 10:00:00 GMT"},
]


class TestClubRefusingToSell:
    """A bid is the selling club's problem, not evidence the player leaves."""

    def test_bruno_fernandes_is_not_meaningfully_at_risk(self):
        risk, _dest, _w, _outlets, _ev, _fee = score_headlines(
            BRUNO_HEADLINES, "Bruno Fernandes", "Man Utd", PL_TEAMS, today=TODAY)
        assert risk < 0.15, (
            "A rejected offer plus contract talks must not discount a squad staple; "
            "got %.3f" % risk
        )

    def test_bare_offer_and_bid_are_bottom_tier(self):
        """Reported constantly, and mostly come to nothing."""
        for headline in ("Galatasaray make lucrative offer for Fernandes",
                         "Newcastle prepare bid for Fernandes",
                         "Man Utd respond to transfer approach"):
            assert classify_headline(headline) == TIER_C, headline

    def test_accepted_bid_is_still_top_tier(self):
        """Demoting 'bid' must not demote 'bid accepted'."""
        assert classify_headline("Villa accept £50m bid for Watkins") == TIER_A
        assert classify_headline("Al Hilal have bid accepted for Watkins") == TIER_A

    def test_contract_talks_are_a_stay_signal_not_talks(self):
        """'In contract talks' is the club tying him down — the opposite signal."""
        assert classify_headline("Man Utd in contract talks with Bruno Fernandes") <= TIER_C

    def test_refusal_across_outlets_caps_the_score(self):
        rejections = [
            {"Headline": "Man Utd reject Galatasaray offer for Bruno Fernandes",
             "Source": "BBC", "Published": "Wed, 26 Aug 2026 10:00:00 GMT"},
            {"Headline": "Bruno Fernandes not for sale at any price, say Man Utd",
             "Source": "Sky Sports", "Published": "Wed, 26 Aug 2026 10:00:00 GMT"},
            {"Headline": "Bruno Fernandes in talks over a move away",
             "Source": "Daily Star", "Published": "Wed, 26 Aug 2026 10:00:00 GMT"},
        ]
        risk, *_ = score_headlines(rejections, "Bruno Fernandes", "Man Utd",
                                   PL_TEAMS, today=TODAY)
        assert risk <= 0.10

    def test_a_medical_overrides_the_clubs_denials(self):
        """The club's position last week does not survive him having a medical."""
        mixed = [
            {"Headline": "Villa say Watkins is not for sale at any price",
             "Source": "BBC", "Published": "Mon, 24 Aug 2026 10:00:00 GMT"},
            {"Headline": "Villa reject Al Hilal offer for Watkins",
             "Source": "Sky Sports", "Published": "Mon, 24 Aug 2026 10:00:00 GMT"},
            {"Headline": "Ollie Watkins undergoes medical at Al Hilal",
             "Source": "The Athletic", "Published": "Fri, 28 Aug 2026 10:00:00 GMT"},
        ]
        risk, *_ = score_headlines(mixed, "Ollie Watkins", "Aston Villa",
                                   PL_TEAMS, today=TODAY)
        assert risk > 0.3


class TestResolveFromBootstrap:
    def test_completed_move_abroad_is_total(self):
        risk, dest = resolve_from_bootstrap("u", "Has joined Al Qadsiah permanently", PL_TEAMS)
        assert risk == WEIGHT_LEAVES_PL
        assert "Qadsiah" in dest

    def test_completed_loan_to_championship_is_total(self):
        risk, _dest = resolve_from_bootstrap(
            "u", "Has joined Leicester City on loan for the rest of the season", PL_TEAMS)
        assert risk == WEIGHT_LEAVES_PL

    def test_completed_intra_pl_move_is_minor(self):
        risk, _dest = resolve_from_bootstrap("u", "Has joined Nott'm Forest permanently", PL_TEAMS)
        assert risk == WEIGHT_INTRA_PL

    def test_injury_is_not_a_transfer(self):
        assert resolve_from_bootstrap("i", "Knee injury - expected back 20 Sep", PL_TEAMS) is None

    def test_available_player_resolves_to_nothing(self):
        assert resolve_from_bootstrap("a", "", PL_TEAMS) is None

    def test_unavailable_without_transfer_wording_is_not_a_transfer(self):
        assert resolve_from_bootstrap("u", "Lack of match fitness", PL_TEAMS) is None


class TestExposure:
    def test_preseason_exposure_is_near_total(self):
        assert transfer_exposure("Real Madrid", today=date(2026, 8, 1)) > 0.85

    def test_saudi_window_outlives_the_english_one(self):
        """The Watkins trap: the English window shut on 1 Sep, the Saudi one ran
        to 12 Oct, so a player could still be sold five gameweeks into the season."""
        after_english_deadline = date(2026, 9, 15)
        assert next_window_close("Saudi Pro League", after_english_deadline) == date(2026, 10, 12)
        assert transfer_exposure("Saudi Pro League", today=after_english_deadline) > 0.5

    def test_exposure_falls_between_windows(self):
        early = transfer_exposure("Real Madrid", today=date(2026, 8, 20))
        mid = transfer_exposure("Real Madrid", today=date(2026, 11, 15))
        assert mid < early

    def test_exposure_is_zero_after_the_january_deadline(self):
        assert transfer_exposure("Real Madrid", today=date(2027, 2, 5)) == 0.0
        assert transfer_exposure("Saudi Pro League", today=date(2027, 3, 1)) == 0.0

    def test_multiplier_returns_to_one_after_the_window_shuts(self):
        """The feature must switch itself off, not linger as a stale discount."""
        exposure = transfer_exposure("Saudi Pro League", today=date(2027, 2, 5))
        assert transfer_multiplier(0.85, exposure) == 1.0


class TestTransferMultiplier:
    def test_no_risk_is_exactly_one(self):
        assert transfer_multiplier(0.0, 0.9) == 1.0

    def test_never_below_the_floor(self):
        assert transfer_multiplier(1.0, 1.0) == TRANSFER_FLOOR

    def test_never_above_one(self):
        for risk in (0.0, 0.3, 0.85, 1.0):
            assert transfer_multiplier(risk, 0.8) <= 1.0

    def test_monotonic_in_risk(self):
        vals = [transfer_multiplier(r, 0.8) for r in (0.1, 0.3, 0.6, 0.9)]
        assert vals == sorted(vals, reverse=True)

    def test_junk_input_degrades_to_no_op(self):
        for bad in (None, "abc", float("nan")):
            assert transfer_multiplier(bad, 0.8) == 1.0
            assert transfer_multiplier(0.5, bad) == 1.0


class TestAttachTransferRisk:
    def _players(self):
        return pd.DataFrame([
            {"Player": "Ollie Watkins", "Team": "Aston Villa", "status": "a", "news": ""},
            {"Player": "Erling Haaland", "Team": "Man City", "status": "a", "news": ""},
            {"Player": "Ben Watson", "Team": "Nott'm Forest", "status": "u",
             "news": "Has joined Leicester City on loan for the rest of the season"},
        ])

    def _news(self):
        return pd.DataFrame([dict(h, Player="Ollie Watkins") for h in WATKINS_HEADLINES])

    def test_attaches_expected_columns(self):
        out = attach_transfer_risk(self._players(), self._news(), PL_TEAMS, today=TODAY)
        for col in ("Transfer_Risk", "Transfer_Exposure", "Transfer_Mult",
                    "Transfer_Destination", "Transfer_Outlets", "Transfer_Note"):
            assert col in out.columns

    def test_watkins_is_heavily_discounted(self):
        out = attach_transfer_risk(self._players(), self._news(), PL_TEAMS, today=TODAY)
        row = out[out["Player"] == "Ollie Watkins"].iloc[0]
        assert row["Transfer_Risk"] >= 0.7
        assert row["Transfer_Mult"] < 0.5

    def test_player_with_no_news_is_untouched(self):
        out = attach_transfer_risk(self._players(), self._news(), PL_TEAMS, today=TODAY)
        row = out[out["Player"] == "Erling Haaland"].iloc[0]
        assert row["Transfer_Risk"] == 0.0
        assert row["Transfer_Mult"] == 1.0

    def test_completed_departure_costs_the_whole_remaining_season(self):
        """He has already gone — there is no window to wait on."""
        out = attach_transfer_risk(self._players(), self._news(), PL_TEAMS, today=TODAY)
        row = out[out["Player"] == "Ben Watson"].iloc[0]
        assert row["Transfer_Exposure"] == 1.0
        assert row["Transfer_Mult"] == pytest.approx(TRANSFER_FLOOR)
        assert row["Transfer_Status"] == STATUS_DEPARTED
        assert "Departed" in row["Transfer_Note"]

    def test_bootstrap_ground_truth_overrides_news(self):
        """Speculation must never outrank a deal that already happened.

        Watkins carries Tier-A Al-Hilal headlines in the news frame. Once the
        bootstrap says he has joined Nott'm Forest, that is where he is: the
        destination is the completed one and the Saudi risk is gone.
        """
        players = self._players()
        players.loc[players["Player"] == "Ollie Watkins", "status"] = "u"
        players.loc[players["Player"] == "Ollie Watkins", "news"] = \
            "Has joined Nott'm Forest permanently"
        out = attach_transfer_risk(players, self._news(), PL_TEAMS, today=TODAY)
        row = out[out["Player"] == "Ollie Watkins"].iloc[0]
        assert row["Transfer_Risk"] == WEIGHT_INTRA_PL
        assert "Forest" in row["Transfer_Destination"]

    def test_completed_intra_pl_move_is_flagged_not_discounted(self):
        """He is still in the game, so in Draft you simply keep him.

        The note is the whole signal here — no multiplier attaches — so it must
        say where he went and must not read as a departure.
        """
        players = self._players()
        players.loc[players["Player"] == "Ollie Watkins", "status"] = "u"
        players.loc[players["Player"] == "Ollie Watkins", "news"] = \
            "Has joined Nott'm Forest permanently"
        out = attach_transfer_risk(players, self._news(), PL_TEAMS, today=TODAY)
        row = out[out["Player"] == "Ollie Watkins"].iloc[0]
        assert row["Transfer_Status"] == STATUS_MOVING_PL
        assert row["Transfer_Mult"] == 1.0
        assert "Departed" not in row["Transfer_Note"]
        assert "Forest" in row["Transfer_Note"]

    def test_no_news_frame_is_safe(self):
        out = attach_transfer_risk(self._players(), None, PL_TEAMS, today=TODAY)
        assert (out["Transfer_Mult"] <= 1.0).all()

    def test_empty_frame_is_safe(self):
        out = attach_transfer_risk(pd.DataFrame(columns=["Player", "Team"]), None, PL_TEAMS)
        assert out.empty
        assert "Transfer_Mult" in out.columns
