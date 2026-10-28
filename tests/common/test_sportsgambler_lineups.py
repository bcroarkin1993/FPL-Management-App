"""Tests for the second projected-XI source.

The fixture HTML is trimmed from the live page (2026-10-22), so a failure means
the scraper stopped agreeing with an observation rather than with an invention.
The structure that matters: an index of `a.lineups-toggle-button` anchors whose
`id` the AJAX endpoint takes, and per fixture a `.lineups-home` / `.lineups-away`
pair of `.player-name` lists.

What is deliberately *not* tested here is name resolution or the benching gate:
both are imported from `rotowire_lineups` and tested there. One gate for every
XI source is the design.
"""

from unittest.mock import patch

from scripts.common import sportsgambler_lineups as sg


def _index_html(rows):
    """The lineups index. `rows` is ``[(fixture_id, home, away)]``."""
    blocks = []
    for fixture_id, home, away in rows:
        blocks.append(f"""
        <div class="table-row-loneups"><div class="lineup-row">
          <div class="fxs-info"><span class="fxs-time">15:00</span></div>
          <div class="fxs-game">
            <span class="fxs-team home">{home}</span>
            <span class="vs-teams"> vs </span>
            <span class="fxs-team">{away}</span>
          </div>
          <div class="fxs-btn">
            <a href="#" class="view-lineups lineups-toggle-button confirmed yellow-button"
               rel="#lineup{fixture_id}" id="{fixture_id}">
              <span class="h-sm">Predicted Lineups</span></a>
          </div>
        </div></div>""")
    return "<html><body>" + "".join(blocks) + "</body></html>"


def _fixture_html(home_names, away_names):
    def _side(cls, names):
        players = "".join(
            f'<div class="lineups-player"><span class="player-profile">{i + 1}</span>'
            f'<span class="player-name">{n}</span></div>'
            for i, n in enumerate(names))
        return f'<div class="{cls}"><div class="players-line">{players}</div></div>'
    return ("<div class='lineups-formation'><h3><span>Predicted Lineup</span></h3></div>"
            "<div class='lineups'>"
            + _side("lineups-home", home_names)
            + _side("lineups-away", away_names)
            + "</div>")


ELEVEN = [f"Player {i}" for i in range(1, 12)]


class TestParsingTheIndex:
    def test_every_fixture_yields_its_id_and_both_clubs(self):
        html = _index_html([("5795486", "Aston Villa", "Man City"),
                            ("5795485", "Arsenal", "Everton")])
        assert sg.parse_index(html) == [
            ("5795486", "Aston Villa", "Man City"),
            ("5795485", "Arsenal", "Everton"),
        ]

    def test_home_comes_first(self):
        """The AJAX fragment is home-then-away, so the index order is load-bearing."""
        rows = sg.parse_index(_index_html([("1", "Hull", "Brentford")]))
        assert rows[0][1:] == ("Hull", "Brentford")

    def test_an_anchor_with_no_id_is_dropped(self):
        html = _index_html([("5795486", "Aston Villa", "Man City")])
        assert sg.parse_index(html.replace('id="5795486"', "")) == []

    def test_nothing_in_gives_nothing_out(self):
        assert sg.parse_index("") == []
        assert sg.parse_index("<html><body>no fixtures</body></html>") == []


class TestParsingAFixture:
    def test_both_elevens_come_back_in_order(self):
        home, away = sg.parse_fixture_xi(_fixture_html(ELEVEN, ELEVEN[::-1]))
        assert home == ELEVEN
        assert away == ELEVEN[::-1]

    def test_a_missing_side_is_empty_not_an_error(self):
        html = _fixture_html(ELEVEN, ELEVEN).replace("lineups-away", "lineups-other")
        home, away = sg.parse_fixture_xi(html)
        assert home == ELEVEN and away == []

    def test_an_unparseable_fragment_is_empty(self):
        assert sg.parse_fixture_xi("") == ([], [])
        assert sg.parse_fixture_xi("<div>maintenance</div>") == ([], [])


class TestTheGameweekFilter:
    """The page carries about five gameweeks, and every one is a full set of ten."""

    INDEX = [("1", "Aston Villa", "Man City"),      # GW8
             ("2", "Arsenal", "Everton"),           # GW8
             ("3", "Liverpool", "Arsenal")]         # a later gameweek

    def test_only_this_gameweeks_fixtures_survive(self):
        kept = sg._fixtures_in_gameweek(self.INDEX, {("AVL", "MCI"), ("ARS", "EVE")})
        assert [f[0] for f in kept] == ["1", "2"]

    def test_a_page_wide_vote_would_not_have_worked(self):
        """Why this is a lookup and not a vote.

        Every gameweek on the page is a complete set of ten fixtures, so a vote
        over all of them scores each gameweek equally and cannot choose. An
        ordered pair occurs at most once a season, so the lookup is exact.
        """
        gw8 = sg._fixtures_in_gameweek(self.INDEX, {("AVL", "MCI"), ("ARS", "EVE")})
        later = sg._fixtures_in_gameweek(self.INDEX, {("LIV", "ARS")})
        assert [f[0] for f in gw8] == ["1", "2"]
        assert [f[0] for f in later] == ["3"]

    def test_an_unreadable_fixture_list_fails_closed(self):
        """**Opposite to the Rotowire matchup filter, and deliberately.**

        There, an extra matchup costs one rendered card, so it fails open. Here
        the XIs feed the start model and the page holds five gameweeks: the
        union of them would promote half the league and is not a lineup call at
        all.
        """
        assert sg._fixtures_in_gameweek(self.INDEX, set()) == []
        assert sg._fixtures_in_gameweek(self.INDEX, None) == []

    def test_an_unmapped_club_label_is_skipped(self):
        kept = sg._fixtures_in_gameweek(
            [("1", "Aston Villa", "Nowhere United")], {("AVL", "MCI")})
        assert kept == []


class TestScrapingEndToEnd:
    def _patched(self, index_rows, per_fixture, gameweek=8, pairs=None):
        if pairs is None:
            pairs = {("AVL", "MCI"), ("ARS", "EVE")}

        class _Resp:
            def __init__(self, text):
                self.text = text

        def _get(url, session=None, **kwargs):
            if url.endswith("lineups-load2.php"):
                fid = (kwargs.get("params") or {}).get("id")
                return _Resp(per_fixture.get(fid, ""))
            return _Resp(_index_html(index_rows))

        with patch.object(sg, "_get", _get), \
                patch.object(sg, "_gameweek_fixture_pairs", lambda gw: pairs):
            return sg.scrape_lineups("https://host/lineups/football/england-premier-league/",
                                     gameweek)

    def test_the_frame_is_shaped_the_way_lineup_starters_wants_it(self):
        """Shared shape is what lets one benching gate serve both sources."""
        scrape = self._patched(
            [("1", "Aston Villa", "Man City")],
            {"1": _fixture_html(ELEVEN, ELEVEN)},
            pairs={("AVL", "MCI")})
        assert list(scrape.players.columns) == ["Team", "Position", "Player", "MatchupIndex"]
        assert len(scrape.players) == 22
        assert set(scrape.players["Team"]) == {"Aston Villa", "Man City"}
        assert scrape.matchups == [("Aston Villa", "Man City")]

    def test_position_is_emitted_empty_so_the_resolver_retries_all_four(self):
        """Sportsgambler publishes no position; the resolver takes it as a hint."""
        scrape = self._patched([("1", "Aston Villa", "Man City")],
                               {"1": _fixture_html(ELEVEN, ELEVEN)},
                               pairs={("AVL", "MCI")})
        assert set(scrape.players["Position"]) == {""}

    def test_a_later_gameweeks_fixture_is_never_fetched(self):
        """Filtering before fetching is what keeps this at ten requests, not fifty."""
        asked = []

        class _Resp:
            def __init__(self, text):
                self.text = text

        def _get(url, session=None, **kwargs):
            if url.endswith("lineups-load2.php"):
                asked.append((kwargs.get("params") or {}).get("id"))
                return _Resp(_fixture_html(ELEVEN, ELEVEN))
            return _Resp(_index_html([("1", "Aston Villa", "Man City"),
                                      ("2", "Liverpool", "Arsenal")]))

        with patch.object(sg, "_get", _get), \
                patch.object(sg, "_gameweek_fixture_pairs", lambda gw: {("AVL", "MCI")}):
            sg.scrape_lineups("https://host/lineups/", 8)
        assert asked == ["1"]

    def test_matchup_indices_stay_contiguous_after_filtering(self):
        scrape = self._patched(
            [("1", "Aston Villa", "Man City"), ("9", "Liverpool", "Arsenal"),
             ("2", "Arsenal", "Everton")],
            {"1": _fixture_html(ELEVEN, ELEVEN), "2": _fixture_html(ELEVEN, ELEVEN)})
        assert sorted(scrape.players["MatchupIndex"].unique().tolist()) == [0, 1]

    def test_a_fixture_that_parses_to_nothing_is_dropped_not_rendered(self):
        scrape = self._patched(
            [("1", "Aston Villa", "Man City"), ("2", "Arsenal", "Everton")],
            {"1": _fixture_html(ELEVEN, ELEVEN), "2": ""})
        assert scrape.matchups == [("Aston Villa", "Man City")]
        assert len(scrape.players) == 22

    def test_an_unreachable_index_yields_an_empty_frame(self):
        with patch.object(sg, "_get", lambda *a, **k: None):
            scrape = sg.scrape_lineups("https://host/lineups/", 8)
        assert scrape.players.empty and scrape.matchups == []
        assert list(scrape.players.columns) == ["Team", "Position", "Player", "MatchupIndex"]


class TestFetchNeverRaises:
    """These reach for the **real** function on purpose.

    `tests/conftest.py` stubs `fetch_lineup_xi` to an empty XI across the whole
    offline suite so nothing touches the network. That stub also returns an
    empty XI for every input, so a test of the real function's failure handling
    would pass against it without exercising a single line -- which is what
    happened when these were first written.
    """

    @staticmethod
    def _real():
        from tests import conftest as root_conftest
        real = (getattr(root_conftest, "REAL_LINEUP_FETCHERS", None)
                or {}).get(sg.__name__)
        assert real is not None, "the conftest stub registry lost this module"
        return real

    def test_a_failure_is_an_empty_xi_not_an_exception(self):
        """The engine reads an empty XI as "no opinion" and the other source stands."""
        with patch.object(sg, "scrape_lineups", side_effect=RuntimeError("boom")):
            sg.clear_memo()
            xi = self._real()("https://host/lineups/", 8)
        assert xi.player_ids == set() and xi.clubs == set()
        assert xi.ok is False

    def test_no_configured_url_says_so(self):
        with patch.object(sg.config, "SPORTSGAMBLER_LINEUPS_URL", None, create=True):
            xi = self._real()(gameweek=8)
        assert not xi.ok and "URL" in xi.note


class TestTheConfirmedClassIsNotTrusted:
    def test_a_confirmed_class_does_not_appear_anywhere_in_the_module(self):
        """Every fixture button carries `confirmed` while the heading says
        "Predicted Lineup". Reading it would promote a prediction to a team
        sheet, so nothing here may branch on it."""
        import inspect
        src = inspect.getsource(sg)
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        assert '"confirmed"' not in code and "'confirmed'" not in code


class TestMemoKeepsOnlySuccesses:
    def test_a_failure_is_not_cached(self):
        sg.clear_memo()
        calls = []

        def _fail():
            calls.append(1)
            return sg.LineupXI(set(), set(), 8, "down")

        sg._memoised("k", _fail)
        sg._memoised("k", _fail)
        assert len(calls) == 2, "memoising a failure pins an outage after it clears"

    def test_a_success_is_cached(self):
        sg.clear_memo()
        calls = []

        def _ok():
            calls.append(1)
            return sg.LineupXI({1}, {"MCI"}, 8, "fine")

        sg._memoised("k2", _ok)
        sg._memoised("k2", _ok)
        assert len(calls) == 1
