"""
App-side bridge between the pure transfer-risk model and the app's data sources.

``transfer_risk.py`` and ``transfer_feeds.py`` stay free of Streamlit and of
``player_matching`` so GitHub Actions can import them.  The cross-source name
matching that Draft and Classic pages need lives here instead, because
``ReferenceMatcher`` pulls in Streamlit.
"""

import pandas as pd

from scripts.common.error_helpers import get_logger
from scripts.common.player_matching import ReferenceMatcher
from scripts.common.transfer_odds import ODDS_COLUMNS
from scripts.common.transfer_risk import (
    RISK_COLUMNS,
    apply_minutes_competition,
    attach_transfer_risk,
    build_inbound_watchlist,
)

_logger = get_logger("fpl_app.transfer_risk_app")


def get_pl_team_names(bootstrap) -> list:
    """Full club names of the current Premier League, from the live bootstrap.

    Used to decide whether a destination keeps a player in the game.  Read from
    live data, never hardcoded, so promotion and relegation need no maintenance —
    which matters because a Championship move costs a manager exactly as much as
    a Saudi one.
    """
    try:
        teams = (bootstrap or {}).get("teams") or []
        return [t.get("name") for t in teams if t.get("name")]
    except AttributeError:
        _logger.warning("Bootstrap had no usable teams list for transfer risk")
        return []


def attach_bootstrap_availability(rankings_df: pd.DataFrame,
                                  availability_df: pd.DataFrame) -> pd.DataFrame:
    """Merge FPL ``status`` / ``news`` onto a Rotowire-shaped frame.

    Rotowire publishes common names ("Bruno Fernandes"); the bootstrap publishes
    full legal names ("Bruno Borges Fernandes").  A strict key misses a sixth of
    the table *silently*, so this goes through the shared tiered matcher rather
    than a merge.

    Players with no match keep empty status/news, which reads as "no completed
    transfer" — the safe default, since news scoring still applies.
    """
    result = rankings_df.copy()
    result["status"] = ""
    result["news"] = ""

    if availability_df is None or availability_df.empty:
        _logger.warning("No FPL availability data — transfer ground truth unavailable")
        return result

    matcher = ReferenceMatcher(
        availability_df, name_col="Player", web_name_col="Web_Name",
        team_col="Team", position_col="Position",
    )

    statuses, newses = [], []
    matched = 0
    for _, row in result.iterrows():
        idx = matcher.match(row.get("Player"), row.get("Team"), row.get("Position"))
        if idx is None:
            statuses.append("")
            newses.append("")
            continue
        matched += 1
        ref = availability_df.loc[idx]
        statuses.append(ref.get("Status") or "")
        newses.append(ref.get("News") or "")

    result["status"] = statuses
    result["news"] = newses
    _logger.info("Transfer risk: matched %d/%d players to FPL availability",
                 matched, len(result))
    return result


def build_transfer_risk(rankings_df: pd.DataFrame,
                        availability_df: pd.DataFrame,
                        news_df: pd.DataFrame,
                        pl_teams,
                        today=None,
                        odds_df: pd.DataFrame = None) -> pd.DataFrame:
    """Full pipeline: ground truth + news (+ odds) -> risk columns.

    ``odds_df`` is the raw odds index; it is matched to the pool here and passed
    down.  Odds only ever soften or sharpen a *speculative* score — a completed
    move found in the bootstrap still overrides everything downstream.

    Never raises. Any failure degrades to a neutral multiplier of 1.0 for every
    player, which is the pre-existing behaviour — a broken news feed must not
    take the draft board down.
    """
    try:
        enriched = attach_bootstrap_availability(rankings_df, availability_df)
        matched_odds = None
        if odds_df is not None and not odds_df.empty:
            matched_odds = attach_odds(enriched, odds_df)
        return attach_transfer_risk(enriched, news_df, pl_teams, today=today,
                                    odds_df=matched_odds)
    except Exception as e:
        _logger.warning("Transfer risk pipeline failed: %s", e, exc_info=True)
        fallback = rankings_df.copy()
        # Every declared column, or a caller that reads one unconditionally
        # crashes on the degraded path instead of just losing the feature.
        for col in RISK_COLUMNS:
            if col in ("Transfer_Mult",):
                fallback[col] = 1.0
            elif col in ("Transfer_Destination", "Transfer_Note", "Transfer_Status"):
                fallback[col] = ""
            elif col == "Transfer_Fee":
                fallback[col] = float("nan")
            else:
                fallback[col] = 0.0
        for col in ODDS_COLUMNS:
            if col in ("Odds_Destination", "Odds_Fractional", "Odds_Bookmaker",
                       "Odds_Updated"):
                fallback[col] = ""
            elif col == "Odds_Age_Days":
                fallback[col] = float("nan")
            else:
                fallback[col] = 0.0
        return fallback


def build_inbound_competition(pool_df: pd.DataFrame,
                              club_news_df: pd.DataFrame,
                              pl_teams,
                              today=None,
                              min_outlets: int = 2):
    """Arrivals watchlist plus the minutes discount it implies.

    Returns ``(watchlist, discounted_pool)``. Never raises: on any failure every
    player keeps ``Minutes_Mult`` 1.0, which is the behaviour before this existed
    — a noisy club feed must not move a draft board by accident.

    ``pool_df`` doubles as the known-player reference, so an intra-PL mover gets
    his real position rather than one inferred from a headline's prose.
    """
    neutral = pool_df.copy()
    neutral["Minutes_Mult"] = 1.0
    neutral["Minutes_Note"] = ""
    neutral["Competition"] = ""
    empty = pd.DataFrame(
        columns=["Player", "Club", "Position", "Fee", "Outlets", "Confidence", "Headline"])

    if club_news_df is None or getattr(club_news_df, "empty", True):
        return empty, neutral

    try:
        watchlist = build_inbound_watchlist(
            club_news_df, pl_teams, today=today, min_outlets=min_outlets,
            known_players=pool_df,
        )
        if watchlist.empty:
            return watchlist, neutral
        return watchlist, apply_minutes_competition(pool_df, watchlist)
    except Exception as e:
        _logger.warning("Inbound competition pipeline failed: %s", e, exc_info=True)
        return empty, neutral


def attach_odds(pool_df: pd.DataFrame, odds_index_df: pd.DataFrame) -> pd.DataFrame:
    """Match bookmaker odds onto a player pool by name.

    Returns a small frame keyed on the pool's own ``Player`` values with
    ``Odds_Exit``, ``Odds_Destination``, ``Odds_Fractional``, ``Odds_Bookmaker``
    and ``Odds_Updated`` — the contract ``attach_transfer_risk(odds_df=...)``
    reads.  Players with no market are simply absent.

    Why not ``ReferenceMatcher``: the odds feed publishes a bare name and the
    club the player is *going to*, never the club he is at, and no position.
    Every tier of the shared matcher below the first is scoped to team or
    position, so it has nothing to work with and would degrade to an exact
    ``(name, team)`` key that never fires.

    A name-only key is the weak kind that attached Cole Palmer's stats to Alex
    Palmer, so two rules apply together:

    1. **A key is kept only when it resolves to exactly one player** — the same
       rule the Waiver Wire's display names use.
    2. **The fallback is a token subset, never a bare surname.**  Uniqueness
       inside the pool is not enough on its own, because the odds feed quotes
       players who have *left* the league: with Darwin Núñez gone from the FPL
       pool, the surname "nunez" resolved uniquely to Marcelino Núñez and handed
       him Darwin's market.  Requiring one name's tokens to contain the other's
       ("bruno fernandes" ⊂ "bruno borges fernandes") keeps the legal-name case
       that matters and rejects two different people who merely share a surname.

    Losing a match costs one odds quote; a wrong match prices a real player on
    someone else's market.
    """
    from scripts.common.text_helpers import canonical_normalize
    from scripts.common.transfer_odds import implied_probability
    from scripts.common.transfer_risk import team_code

    columns = ["Player", "Odds_Exit", "Odds_Destination", "Odds_Fractional",
               "Odds_Bookmaker", "Odds_Updated", "Odds_Slug"]
    if (pool_df is None or pool_df.empty
            or odds_index_df is None or odds_index_df.empty
            or "Player" not in pool_df.columns):
        return pd.DataFrame(columns=columns)

    full_index = {}
    token_index = []
    for name in pool_df["Player"].dropna().astype(str):
        key = canonical_normalize(name)
        if not key:
            continue
        full_index.setdefault(key, set()).add(name)
        tokens = {t for t in key.split() if len(t) > 1}
        if tokens:
            token_index.append((tokens, name))

    team_of = {}
    if "Team" in pool_df.columns:
        team_of = {str(r["Player"]): team_code(r["Team"])
                   for _, r in pool_df[["Player", "Team"]].dropna().iterrows()}

    rows, ambiguous, settled = [], 0, 0
    for _, odds_row in odds_index_df.iterrows():
        raw = odds_row.get("Player")
        key = canonical_normalize(str(raw)) if raw else ""
        if not key:
            continue

        candidates = full_index.get(key)
        if not candidates:
            query = {t for t in key.split() if len(t) > 1}
            if query:
                candidates = {name for tokens, name in token_index
                              if query <= tokens or tokens <= query}
        if not candidates:
            continue
        if len(candidates) > 1:
            # Ambiguous resolves to no match, exactly as ReferenceMatcher does.
            ambiguous += 1
            continue

        player = next(iter(candidates))

        # A market whose destination is the player's *current* club has already
        # settled: he signed, and the quote is left over from before he did.
        # Reading it as exit risk says a player who has just arrived is 40%
        # likely to leave — the inverse of the exclude_team trap in
        # parse_destination, and just as expensive. Resolve through
        # TEAM_FULL_TO_SHORT (team_code), never by comparing raw strings.
        destination_code = team_code(odds_row.get("Next_Club"))
        if destination_code and destination_code == team_of.get(player):
            settled += 1
            continue

        decimal = odds_row.get("Decimal")
        implied = odds_row.get("Implied")
        if implied is None or (isinstance(implied, float) and pd.isna(implied)):
            implied = implied_probability(decimal)
        if not implied:
            continue

        rows.append({
            "Player": player,
            "Odds_Exit": float(implied),
            "Odds_Destination": odds_row.get("Next_Club") or "",
            "Odds_Fractional": odds_row.get("Fractional") or "",
            "Odds_Bookmaker": odds_row.get("Bookmaker") or "",
            "Odds_Updated": odds_row.get("Updated") or "",
            # The site's own slug, so a ladder request never has to guess a URL
            # from an FPL legal name ("gabriel-martinelli-silva" is not a page).
            "Odds_Slug": odds_row.get("Slug") or "",
        })

    _logger.info("Transfer odds: matched %d/%d markets to the player pool "
                 "(%d ambiguous, %d already settled)",
                 len(rows), len(odds_index_df), ambiguous, settled)
    return pd.DataFrame(rows, columns=columns)
