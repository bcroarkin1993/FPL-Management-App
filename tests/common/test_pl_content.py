"""Offline tests for the Premier League content parser.

Every fixture here is a real payload, and every assertion is a shape the live
feed was actually observed to have on 2026-09-11 — the parse traps in
particular are not hypothetical, they are failures the first draft of this
parser had.
"""

import json
import pathlib

import pandas as pd
import pytest

from scripts.common import pl_content
from scripts.common.data_validation import check_pl_content
from scripts.common.text_helpers import TEAM_FULL_TO_SHORT

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

#: Every club in the live MW4 article, by the label the PL publishes.
EXPECTED_CLUBS = {
    "ARS", "AVL", "BHA", "BOU", "BRE", "CHE", "COV", "CRY", "EVE", "FUL",
    "HUL", "IPS", "LEE", "LIV", "MCI", "MUN", "NEW", "NFO", "SUN", "TOT",
}

#: The four clubs whose <strong> wraps the entire paragraph rather than just
#: the "Club:" label. Taking strong.get_text() whole drops exactly these.
WHOLE_STRONG_CLUBS = {"BOU", "LIV", "COV", "LEE"}


@pytest.fixture(scope="module")
def article():
    return json.loads((FIXTURES / "pl_article_mw4.json").read_text())


@pytest.fixture(scope="module")
def lineups(article):
    # resolve_graphics=False keeps this offline; graphic mapping is tested
    # separately against a stubbed photo lookup.
    return pl_content.parse_predicted_lineups(article, resolve_graphics=False)


# =============================================================================
# Article parsing
# =============================================================================

def test_all_twenty_clubs_carry_team_news(lineups):
    """The <strong> trap: the label is the text *before the first colon inside*
    the <strong>, not the whole <strong>. Getting this wrong yields 16 clubs."""
    assert set(lineups.club_news) == EXPECTED_CLUBS
    assert len(lineups.club_news) == 20


@pytest.mark.parametrize("club", sorted(WHOLE_STRONG_CLUBS))
def test_clubs_whose_strong_wraps_the_paragraph(lineups, club):
    """These four are the regression guard: their markup bolds the whole
    paragraph, so a naive parse silently loses them and nothing else changes."""
    assert club in lineups.club_news
    text = lineups.club_news[club]
    assert len(text) > 80, "prose was truncated to the label"
    assert not text.lower().startswith(("bournemouth", "liverpool", "coventry", "leeds")), \
        "the club label leaked into the body"


def test_team_news_excludes_see_also_links(lineups):
    """'See: <club> team news' paragraphs are also <p><strong>...</strong>."""
    for club, text in lineups.club_news.items():
        assert not text.lower().startswith("see"), club
        assert "team news" not in text[:20].lower(), club


def test_every_published_club_label_resolves(lineups):
    """An unmapped label is not cosmetic -- matching and gameweek voting are
    both scoped by club, so it drops that club entirely (the FFP 'Notts
    Forest' lesson)."""
    assert lineups.unresolved_labels == ()


def test_fixtures_are_ordered_home_away_pairs(lineups):
    assert len(lineups.fixtures) == 10
    assert ("AVL", "NFO") in lineups.fixtures
    assert ("NFO", "AVL") not in lineups.fixtures, "home/away order was lost"
    clubs = {c for pair in lineups.fixtures for c in pair}
    assert clubs == EXPECTED_CLUBS, "every club plays exactly once"


def test_gameweek_comes_from_the_title(lineups):
    assert lineups.gameweek == 4
    assert "Matchweek 4" in lineups.title


def test_gameweek_is_none_when_title_and_fixtures_disagree(article):
    """A source that cannot prove its own gameweek must not state one."""
    pairs = {9: {("AVL", "NFO"), ("BOU", "BRE"), ("CHE", "HUL"), ("CRY", "IPS"),
                 ("LIV", "FUL"), ("TOT", "EVE"), ("SUN", "ARS"), ("COV", "BHA"),
                 ("MUN", "MCI"), ("LEE", "NEW")}}
    parsed = pl_content.parse_predicted_lineups(
        article, fixture_pairs=pairs, resolve_graphics=False)
    assert parsed.gameweek is None
    assert "MW4" in parsed.note and "MW9" in parsed.note


def test_fixture_vote_stands_in_when_the_title_has_no_matchweek(article):
    pairs = {4: {("AVL", "NFO"), ("BOU", "BRE"), ("CHE", "HUL"), ("CRY", "IPS"),
                 ("LIV", "FUL"), ("TOT", "EVE"), ("SUN", "ARS"), ("COV", "BHA"),
                 ("MUN", "MCI"), ("LEE", "NEW")}}
    retitled = dict(article, title="Predicted line-ups for every Premier League team")
    parsed = pl_content.parse_predicted_lineups(
        retitled, fixture_pairs=pairs, resolve_graphics=False)
    assert parsed.gameweek == 4
    assert "fixture vote" in parsed.note


def test_updated_timestamp_is_timezone_aware(lineups):
    assert lineups.updated is not None
    assert lineups.updated.tzinfo is not None


def test_bad_payloads_return_empty_rather_than_raising():
    for payload in (None, {}, {"title": "x", "body": None}, {"body": "<p>no clubs</p>"}):
        parsed = pl_content.parse_predicted_lineups(payload, resolve_graphics=False)
        assert not parsed.ok
        assert dict(parsed.club_news) == {}


# =============================================================================
# Gameweek voting
# =============================================================================

def test_ordered_pairs_separate_adjacent_gameweeks():
    """A *set* of club names cannot: all 20 clubs play every week."""
    gw4 = {("AVL", "NFO"), ("CHE", "HUL"), ("MUN", "MCI")}
    gw5 = {("NFO", "AVL"), ("HUL", "CHE"), ("MCI", "MUN")}
    assert pl_content.resolve_gameweek_from_fixtures(
        tuple(gw4), {4: gw4, 5: gw5}) == 4


def test_vote_returns_none_when_ambiguous():
    pairs = {("AVL", "NFO")}
    assert pl_content.resolve_gameweek_from_fixtures(
        tuple(pairs), {4: pairs, 5: pairs}) is None


def test_vote_returns_none_below_threshold():
    assert pl_content.resolve_gameweek_from_fixtures(
        (("AVL", "NFO"), ("CHE", "HUL"), ("MUN", "MCI")),
        {4: {("AVL", "NFO")}}) is None


def test_vote_returns_none_without_fixtures():
    assert pl_content.resolve_gameweek_from_fixtures((), {4: {("A", "B")}}) is None
    assert pl_content.resolve_gameweek_from_fixtures((("A", "B"),), {}) is None


# =============================================================================
# XI graphics
# =============================================================================

def _photo(title, url="https://example.test/x.png"):
    return {"title": title, "onDemandUrl": url}


def test_on_demand_url_gets_a_width():
    """onDemandUrl is a resizer, not a file: bare it answers 400 'At least one
    of width or height parameters must be specified'. Without this the page
    renders 20 broken images whose only symptom is the alt text."""
    url = pl_content._graphic_url({"onDemandUrl": "https://x/a.png"})
    assert url == "https://x/a.png?width=%d" % pl_content.GRAPHIC_WIDTH


def test_on_demand_url_with_existing_query_keeps_it():
    url = pl_content._graphic_url({"onDemandUrl": "https://x/a.png?v=2"})
    assert url == "https://x/a.png?v=2&width=%d" % pl_content.GRAPHIC_WIDTH


def test_image_url_is_the_fallback_and_needs_no_parameters():
    """imageUrl is a plain file; adding a width to it would be wrong."""
    assert pl_content._graphic_url({"imageUrl": "https://x/b.png"}) == "https://x/b.png"
    assert pl_content._graphic_url({}) is None


def test_resolved_graphics_carry_a_width(monkeypatch):
    monkeypatch.setattr(pl_content, "fetch_photo",
                        lambda pid, **kw: _photo("Aston Villa Matchweek 4 line up",
                                                 "https://x/avl.png"))
    out = pl_content._resolve_graphics((("AVL", ""),), [["1", "2"]], 4, 20)
    assert all("width=" in url for url in out.values())


def test_graphics_map_to_clubs_and_verify_against_the_photo_title(monkeypatch):
    photos = {
        "1": _photo("Aston Villa Matchweek 4 line up", "https://x/avl.png"),
        "2": _photo("Nottm Forest Matchweek 4 line up", "https://x/nfo.png"),
    }
    monkeypatch.setattr(pl_content, "fetch_photo",
                        lambda pid, **kw: photos.get(str(pid)))
    out = pl_content._resolve_graphics((("AVL", "NFO"),), [["1", "2"]], 4, 20)
    width = pl_content.GRAPHIC_WIDTH
    assert out == {"AVL": "https://x/avl.png?width=%d" % width,
                   "NFO": "https://x/nfo.png?width=%d" % width}


def test_graphic_titled_for_another_club_is_dropped(monkeypatch):
    """Position proposes, the title confirms. Rendering one club's XI under
    another club's name is worse than rendering nothing."""
    photos = {"1": _photo("Aston Villa Matchweek 4 line up"),
              "2": _photo("Chelsea Matchweek 4 line up")}
    monkeypatch.setattr(pl_content, "fetch_photo",
                        lambda pid, **kw: photos.get(str(pid)))
    out = pl_content._resolve_graphics((("AVL", "NFO"),), [["1", "2"]], 4, 20)
    assert set(out) == {"AVL"}


def test_graphic_from_another_matchweek_is_dropped(monkeypatch):
    photos = {"1": _photo("Aston Villa Matchweek 3 line up"),
              "2": _photo("Nottm Forest Matchweek 4 line up")}
    monkeypatch.setattr(pl_content, "fetch_photo",
                        lambda pid, **kw: photos.get(str(pid)))
    out = pl_content._resolve_graphics((("AVL", "NFO"),), [["1", "2"]], 4, 20)
    assert set(out) == {"NFO"}


def test_unexpected_graphic_count_skips_the_fixture(monkeypatch):
    monkeypatch.setattr(pl_content, "fetch_photo",
                        lambda pid, **kw: _photo("Aston Villa Matchweek 4 line up"))
    assert pl_content._resolve_graphics((("AVL", "NFO"),), [["1", "2", "3"]], 4, 20) == {}


def test_nottm_forest_photo_label_resolves():
    """The PL spells it without an apostrophe only in the graphic titles."""
    assert TEAM_FULL_TO_SHORT.get("Nottm Forest") == "NFO"
    assert pl_content._club_from_photo_title("Nottm Forest Matchweek 4 line up") == "NFO"
    assert pl_content._club_from_photo_title("Some FC Matchweek 4 line up") is None


# =============================================================================
# Injury table
# =============================================================================

@pytest.fixture(scope="module")
def injury_payloads():
    return json.loads((FIXTURES / "pl_injuries.json").read_text())


@pytest.fixture
def injuries(injury_payloads, monkeypatch):
    def fake(url, **kwargs):
        if "4509826" in url:
            return injury_payloads["hub"]
        for club_id, payload in injury_payloads["clubs"].items():
            if "/%s?" % club_id in url:
                return payload
        return None

    monkeypatch.setattr(pl_content, "_get_json", fake)
    return pl_content.fetch_pl_injuries()


def test_injury_table_shape(injuries):
    assert list(injuries.columns) == pl_content.INJURY_COLUMNS
    assert not injuries.empty
    assert injuries["Team"].notna().all(), "a club label failed to resolve"
    assert injuries["Player"].str.len().gt(0).all()


def test_injury_placeholder_becomes_blank_not_a_claim(injuries):
    """The PL writes '-' where it has no detail; rendering that verbatim reads
    as a stated injury type."""
    assert "-" not in set(injuries["Injury"])


def test_injury_fetch_returns_empty_frame_on_failure(monkeypatch):
    monkeypatch.setattr(pl_content, "_get_json", lambda url, **kw: None)
    out = pl_content.fetch_pl_injuries()
    assert out.empty
    assert list(out.columns) == pl_content.INJURY_COLUMNS


# =============================================================================
# Matching
# =============================================================================

def _pool(rows):
    return pd.DataFrame(rows)


def test_match_prefers_exact_display_name_over_legal_name():
    pool = _pool([
        {"Player_ID": 1, "Player": "Cody Gakpo", "Web_Name": "Gakpo", "Team": "LIV"},
        {"Player_ID": 2, "Player": "Bruno Borges Fernandes", "Web_Name": "B.Fernandes",
         "Team": "MUN"},
    ])
    pool = pl_content.add_display_names(pool)
    inj = _pool([{"Team": "LIV", "Player": "Cody Gakpo", "Injury": "Adductor", "Link": None},
                 {"Team": "MUN", "Player": "Bruno Fernandes", "Injury": "Knock", "Link": None}])
    out = pl_content.attach_pl_injuries(pool, inj)
    assert set(out["Player_ID"]) == {1, 2}


def test_match_is_scoped_to_the_club():
    """The same surname at another club must not match."""
    pool = _pool([{"Player_ID": 1, "Player": "Alex Palmer", "Web_Name": "Palmer",
                   "Team": "WBA"}])
    inj = _pool([{"Team": "CHE", "Player": "Cole Palmer", "Injury": "Groin", "Link": None}])
    assert pl_content.attach_pl_injuries(pool, inj).empty


def test_ambiguity_resolves_to_no_match():
    """Two players who both token-subset the query cancel out rather than one
    winning a coin flip."""
    pool = _pool([
        {"Player_ID": 1, "Player": "Gabriel Fernando de Jesus", "Web_Name": "Jesus",
         "Team": "ARS"},
        {"Player_ID": 2, "Player": "Gabriel dos Santos Magalhaes", "Web_Name": "Gabriel",
         "Team": "ARS"},
    ])
    inj = _pool([{"Team": "ARS", "Player": "Gabriel", "Injury": "Knock", "Link": None}])
    out = pl_content.attach_pl_injuries(pool, inj)
    assert out.empty or 1 not in set(out["Player_ID"])


def test_bare_surname_alone_does_not_match():
    """A token subset, never a surname -- the Darwin/Marcelino Nunez lesson."""
    pool = _pool([{"Player_ID": 1, "Player": "Marcelino Nunez", "Web_Name": "Nunez",
                   "Team": "IPS"}])
    inj = _pool([{"Team": "IPS", "Player": "Darwin Nunez", "Injury": "Knee", "Link": None}])
    assert pl_content.attach_pl_injuries(pool, inj).empty


def test_attach_handles_missing_columns_and_empties():
    empty = pd.DataFrame()
    assert pl_content.attach_pl_injuries(empty, empty).empty
    assert pl_content.attach_pl_injuries(_pool([{"Player": "x"}]), empty).empty


def test_cms_item_date_is_not_carried_into_the_match():
    """It is a CMS authoring date, not an injury date -- 20 live rows shared a
    single value eight months old. Surfacing it as 'reported' would be a
    confident claim the data does not support."""
    pool = pl_content.add_display_names(
        _pool([{"Player_ID": 1, "Player": "Cody Gakpo", "Web_Name": "Gakpo", "Team": "LIV"}]))
    inj = _pool([{"Team": "LIV", "Player": "Cody Gakpo", "Injury": "Adductor",
                  "Link": None, "Item_Date": "2026-01-19"}])
    out = pl_content.attach_pl_injuries(pool, inj)
    assert not any("Date" in c for c in out.columns)


# =============================================================================
# Validation
# =============================================================================

def test_healthy_payload_raises_no_issues(lineups, injuries):
    issues = check_pl_content(lineups=lineups, injuries=injuries,
                              matched_rows=len(injuries), expected_gw=4)
    assert issues == [], [str(i) for i in issues]


def test_dropped_clubs_are_an_error(lineups):
    crippled = lineups._replace(club_news={"ARS": "x"})
    issues = check_pl_content(lineups=crippled, expected_gw=4)
    assert any(i.severity == "error" and "team news" in i.message for i in issues)


def test_a_part_played_gameweek_is_not_an_error(lineups):
    """The PL rewrites the edition in place as matches kick off, dropping the
    fixtures already played -- the same article id went from 10 fixtures and 20
    clubs on the Friday to 3 and 6 by Saturday afternoon. Judged against a flat
    20 that fails every weekend; judged against its own fixtures it is fine."""
    trimmed = lineups._replace(
        fixtures=lineups.fixtures[:3],
        club_news={c: lineups.club_news[c]
                   for pair in lineups.fixtures[:3] for c in pair},
    )
    issues = check_pl_content(lineups=trimmed, expected_gw=4)
    assert issues == [], [str(i) for i in issues]


def test_unresolved_label_is_an_error(lineups):
    issues = check_pl_content(lineups=lineups._replace(
        unresolved_labels=("Notts County",)), expected_gw=4)
    assert any(i.severity == "error" and "Notts County" in i.message for i in issues)


def test_wrong_gameweek_is_an_error(lineups):
    issues = check_pl_content(lineups=lineups, expected_gw=7)
    assert any(i.severity == "error" and "MW4" in i.message for i in issues)


def test_empty_injury_table_is_an_error():
    issues = check_pl_content(injuries=pd.DataFrame())
    assert any(i.severity == "error" for i in issues)


def test_collapsed_match_rate_is_an_error(injuries):
    issues = check_pl_content(injuries=injuries, matched_rows=1)
    assert any(i.severity == "error" and "matched" in i.message for i in issues)


def test_module_is_importable_without_streamlit():
    """Same constraint as the projection modules: the caching wrappers live in
    scraping.py so this half stays importable where Streamlit is not installed."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c",
         "import sys;"
         "import scripts.common.pl_content;"
         "assert 'streamlit' not in sys.modules, 'pulled in Streamlit';"
         "print('ok')"],
        cwd=pathlib.Path(__file__).resolve().parents[2],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
