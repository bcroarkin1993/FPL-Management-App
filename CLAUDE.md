# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit-based Fantasy Premier League (FPL) management app supporting both **Draft** and **Classic** formats. Integrates FPL APIs with Rotowire projections to provide analytics, transfer suggestions, and optimization tools for FPL managers.

## Commands

```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Then edit with your league/team IDs
git config core.hooksPath .githooks  # Enable pre-push test hook

# Run the app
streamlit run main.py

# Run Discord waiver alerts (used by GitHub Actions)
python -m scripts.common.waiver_alerts

# Run tests
pytest                        # All tests (includes live plausibility checks)
pytest tests/common/          # Unit tests only
pytest tests/draft/ tests/classic/ tests/fpl/  # Smoke tests only
pytest tests/live/            # Live data plausibility only
pytest -m "not live"          # Skip anything that hits the network
FPL_SKIP_LIVE_TESTS=1 pytest  # Same, via env var
pytest -x                     # Stop on first failure
```

## Architecture

### Data Flow
```
FPL Draft API  ──────────┐
FPL Classic API ─────────┤
Rotowire scrape ─────────┼──> scripts/common/utils.py (normalize/merge) ──> Page scripts ──> Streamlit UI
Fantasy Football Pundit ─┤
The Odds API ────────────┘
```

### Key Components

**config.py** - Configuration with lazy loading via PEP 562 `__getattr__`. `CURRENT_GAMEWEEK` and `ROTOWIRE_URL` resolve on first access to avoid import-time network calls.

**scripts/common/** - Shared utilities:
- `utils.py` - FPL API fetching, Rotowire scraping, player matching, fixture analysis
- `player_matching.py` - `canonical_normalize()`, `PlayerRegistry` for centralized player lookups
- `analytics.py` - `compute_player_scores()` (shared Keep/Transfer scoring via positional percentiles — see "Transfer Scoring Model" section), `compute_dynamic_alpha()`, `merge_ffp_single_gw_data()`, `positional_percentile()`, `positional_rank()`, form dampening, multi-GW blending, season projection merging
- `waiver_alerts.py` - Discord notification system for Draft waiver and Classic transfer deadlines
- `team_strength.py` - Draft power rankings (`build_league_strength()`, `compute_player_strength()`, `aggregate_team_strength()`) — 0-100 team + positional scores from positional percentiles
- `injury_helpers.py` - `estimate_games_to_miss()` (parses expected-return dates out of FPL news text) and `injury_multiplier()` (season-aware availability discount)

**main.py** - Streamlit entry point with three-section navigation:
- FPL App Home: Cross-format tools (fixtures, lineups, stats, injuries)
- Draft: League-specific analysis, waiver wire, team projections
- Classic: Full Classic FPL support (home with standings/charts, fixture projections, transfers, free hit optimizer, team analysis)

**Page scripts** - Organized by section, each implements a `show_*_page()` function:
- `scripts/draft/` - home.py, waiver_wire.py, fixture_projections.py, team_analysis.py, league_analysis.py, draft_helper.py
- `scripts/classic/` - home.py, team_analysis.py, fixture_projections.py, transfers.py, free_hit.py, league_analysis.py
- `scripts/fpl/` - fixtures.py, player_statistics.py, projected_lineups.py, availability.py (transfer news + odds + injuries), injuries.py

### External Data Sources

| Source | Usage |
|--------|-------|
| `draft.premierleague.com/api/` | League data, rosters, transactions |
| `fantasy.premierleague.com/api/` | Player stats, fixtures, FDR |
| `rotowire.com/soccer/` | Player projections, EPL lineups, article publish times |
| `fantasyfootballpundit.com` | Points predictions, goal/assist odds, clean sheet odds (site payload; see "Fantasy Football Pundit feed") |
| `api.the-odds-api.com` | Match betting odds (h2h, BTTS, totals) |

### Player Matching

Players are matched across sources using a two-step approach:
1. **Canonical normalization** via `canonical_normalize()` strips accents and normalizes names (e.g., "Raúl Jiménez Rodríguez" → "raul jimenez rodriguez")
2. **Team-prioritized fuzzy matching** tries same-team matches first, then falls back to cross-team matching

Key modules:
- `scripts/common/player_matching.py` - `canonical_normalize()`, `PlayerRegistry` class for centralized lookups, `ReferenceMatcher` for tiered source matching
- `scripts/common/utils.py` - `merge_fpl_players_and_projections()` with 80% threshold (60% if team+position match)

**`ReferenceMatcher` — the shared tiered matcher.** Projection sources publish
common names ("Bruno Fernandes"); the FPL bootstrap publishes full legal names
("Bruno Borges Fernandes"). A strict `(name, team)` key misses ~16% of a 425-row
table, and the misses are *silent* — the caller sees NaN, substitutes a neutral
default, and the #2 asset in the game renders as exactly average.

Tiers, loosest last: (1) exact `(name, team)`, (2) exact `(web_name, team)`,
(3) last word, (4) token subset either direction — token sets are unordered so
this also covers reversed names like "Mitoma Kaoru", (5) difflib ≥ 0.78,
(6) exact full name + position with **team ignored**, opt-in per caller.

Three rules exist because breaking them caused real bugs:
- **Every tier below the first two is scoped to Position, and every tier below
  the first two except tier 6 is scoped to Team as well.** A surname-only,
  team-agnostic match once gave Alex Palmer (backup GK) Cole Palmer's (elite MID)
  stats and ranked him top 10.
- **The fuzzy tier requires a shared complete token.** Character similarity
  alone matched "Harrison" to team-mate "Harry Wilson" at 0.79.
- **Tier 6 demands the *whole* name, and only from the full-name query.** It
  exists because two sources disagree about a club for weeks after a transfer —
  FPL had Nicolas Jackson at Villa while FFP still had him at Chelsea — and every
  team-scoped tier misses him. Retrying it on a short `Web_Name` gives the tier
  a one-word key with the team dropped too, which matched Spurs' "Sávio" to a Man
  City row on nothing but the word; pass `cross_team=False` on that retry.

Ambiguity (>1 candidate at a tier) resolves to *no match*. `match_with_tier()`
returns the tier rank so callers can resolve two players contending for one
reference row in favour of the stronger tier — without that, a fuzzy guess can
steal a row an exact match already earned. Two players tying at tier 6 for one
row cancel out rather than one winning a coin flip.

`_claim_reference_rows()` (`analytics.py`) is that whole protocol in one place —
build the matcher, query on `Player` then `Web_Name`, resolve contention by tier
— and all three projection merges go through it.

**Team labels are part of matching.** Every source spells clubs its own way, and
an unmapped label is not cosmetic: matching is scoped by team, so FFP's "Notts
Forest" (absent from `TEAM_FULL_TO_SHORT` until 2026-09-03) sent all 28 Forest
rows past the exact tiers into the loose ones. `tests/live/` now fails on an FFP
label that does not resolve, as well as on a missing current club.

### Fantasy Football Pundit feed

`scripts/common/ffp_feed.py` (pure, Streamlit-free), `scraping.get_ffp_feed()`
(cached, provenance) and `analytics.ffp_gameweek_matches()` (the gate).

**FFP publishes the same numbers twice and the two disagree.** The published
Google Sheet the app read for a year has stopped keeping pace: measured
2026-09-04 with the app and Rotowire both on GW3, the sheet's `Fixture` column
still described **GW2** — and not consistently, since Aston Villa and Man City
each carried a leftover GW1 fixture string alongside the GW2 one. The site said
"Updated for GW3 · 4 September at 16:18". The app was blending Rotowire GW3 at
60% with FFP GW2 at 40% and applying GW2 start probabilities, silently, because
every individual value was plausible.

The site is a Next.js app that server-embeds its tables in the RSC flight
payload (`self.__next_f.push([1,"…"])`). Concatenating those chunks,
JSON-unescaping each and `raw_decode`-ing from `"rows":[` yields records that
beat the sheet in four ways: an explicit `gw` on every row; `player_code`, which
is the FPL bootstrap `code` (368/368 resolved live, so **FFP joins on an integer
id and never on a name**); `fixture_count`, so doubles and blanks are
representable; and six gameweeks of forecasts instead of five relative-offset
columns.

**The two point columns are named the opposite way round from the sheet.**
Verified over all 2208 live rows: `predicted_points_start == predicted_points *
start_pct/100` at MAE 0.0003, against 2.31 for the reverse. So the site's
`predicted_points` is the *conditional* value (the sheet's `StartingPredicted`)
and `predicted_points_start` is the *unconditional* one (the sheet's
`Predicted`). Mapping these by name rather than by basis re-creates the
double-discount bug under "FFP has two prediction bases"; `check_ffp_feed()`
errors on it, since an inversion makes `Predicted` exceed `StartingPredicted`,
which the relation forbids.

`to_sheet_schema()` is the compatibility seam — it emits the legacy sheet column
names, so all seven page callsites and every merge in `analytics.py` are
unchanged. Sheet semantics it reproduces, pinned live: `GW2…GW6` are **relative
offsets** (the 2nd…6th gameweek of the window, conditional basis), `GW2s…GW6s`
are those start-discounted, and the cumulative columns **include** the current
gameweek — `Next2GWsStart == StartingPredicted + GW2` (MAE 0.03) rather than
`GW2 + GW3` (MAE 0.45). So `Next3GWs` as a 3-gameweek start-adjusted total is
correct as the app already used it.

**`LongStart` has no site equivalent** and is deliberately not emitted: the site
publishes one `start_pct` per player, identical across all six forecast weeks
(`nunique() == 1` for all 368). `compute_player_scores()` already falls back to
the FPL `starts` count for `_start_consistency`; a copy of `Start` would instead
spend 10% of ROS on a signal 1GW already carries.

**A set of team names cannot identify a gameweek.** All 20 clubs play every
week, so the 50%-overlap check this replaced scored 18/19 for GW2, GW3 *and*
GW4, never fired, and had `is_ffp_available_for_gw(3)` announcing "FFP GW3
projections are now available" against a GW2 table — burning the once-per-GW
alert guard so the real publication went unannounced. `resolve_ffp_gameweek()`
votes **ordered `(home, away)` pairs** against the real fixture list instead: on
the live stale sheet that scores GW2 at 0.83 and GW3 at exactly 0.0. Club labels
resolve through `TEAM_FULL_TO_SHORT`, never raw strings. It requires a unique
winner at ≥0.60 and otherwise returns `None` — "unknown" must never be reported
as a gameweek.

**The gate lives inside the merges and defaults itself.** `expected_gw` on
`merge_ffp_single_gw_data()`, `blend_multi_gw_projections()` and
`blend_fixture_projections()` defaults to `config.CURRENT_GAMEWEEK`, so no
future caller can forget it — the same reasoning as locked-player filtering
living inside `_compute_transfer_suggestions()`. A mismatch leaves the FFP
columns NaN, which is already the "unmatched" contract, so every consumer
degrades to Rotowire-only unchanged. An *unknown* gameweek is not a wrong one
and does not gate: refusing to blend on "we could not tell" would remove FFP
from every page whenever the fallback's vote is inconclusive.

**A failure is never cached as a success.** `get_ffp_feed()` is
`@st.cache_data(ttl=300)`, and returning a bare `None` from it pinned a
transient timeout for five minutes — which is how FFP came to read "temporarily
unavailable" in the app while the website plainly worked. On failure it returns
`provenance="none"` with the reason in `note`. Fetches retry three times with
backoff at `timeout=20`; a real sheet read was observed to exceed the old 15s.

`blend_fixture_projections()` also blended `FFP_Predicted` and then multiplied
by start likelihood again — the double discount `compute_player_scores()` was
fixed for. It now blends `FFP_Starting_Predicted` with the same un-discount
recovery. Measured across 163 affected players this raised `Proj_Blended` by
7.2% on average, never lowered it, and moved low-start players most (Matheus
Nunes at 40% start: 3.25 → 4.20).

Risk accepted: the parser depends on FFP's frontend, which they have just
rebuilt. The sheet stays as an automatic fallback, a parse failure degrades to
Rotowire-only rather than taking a page down, and `tests/live/` fails on a
payload that changes shape, states no gameweek, states the wrong one, or whose
fixtures contradict its own gameweek claim.

### Source Freshness

`get_rotowire_article_updated()` (`scripts/common/scraping.py`) scrapes an
article's "Updated on ..." stamp from its `div.article__date`; render it with
`format_last_updated()` (`text_helpers.py`), which appends the age
("Aug 20, 2026 10:54 AM ET (3h ago)").

Show this wherever projections are displayed. A weekly rankings table written
before the last team-news cycle is materially less reliable than one written
after it, and nothing else on the page distinguishes them. Currently surfaced in
the Initial Squad Optimizer's Data Sources panel and the Projections Hub source
banner. FFP now carries one too — its site stamps "Updated for GW3 · 4 September
at 16:18", read by `ffp_feed.parse_updated()`. Only the spreadsheet fallback
lacks a revision time, and shows "—" rather than a fabricated one.

A missing timestamp is cosmetic: the scraper returns `None` on any failure and
must never take a page down.

### Live Gameweek Player Status

**Minutes played cannot tell an unused substitute from a player whose match has not
kicked off.** Both are `minutes == 0`, and reading only that rendered Senesi — named
in the XI, 0 minutes, match long over — as "Upcoming" for the rest of the week, with
his full projection still counted in the team total.

`get_live_gameweek_stats()` therefore attaches each player's own fixture state
(`fixture_started`, `fixture_finished`) alongside `has_played`, and
`live_player_status()` (`fixture_helpers.py`) turns the pair into one of
`played` / `dnp` / `live` / `upcoming`. A `dnp` player scores his actual 0 in
`Blended_Points`: no more points can arrive for him this week.

**`finished` alone is not "the match is over".** The API leaves it `False` for hours
after full time while bonus is confirmed, publishing `finished_provisional` first —
during that window every completed match read as unfinished, which also suppressed
auto-subs entirely. `get_gw_team_fixture_status()` accepts either, and in a double
gameweek a club is finished only once *every* one of its fixtures is.

**Join live stats on the element id, not the name.** `merge_fpl_players_and_projections()`
takes a matched row's `Player` from the *projection* source, so the frame the Draft
lineup renders is keyed on Rotowire names: "Igor Thiago" never reaches the bootstrap's
"Igor Thiago Nascimento Rodrigues", and he showed as Upcoming through a full 90
minutes. Pass `carry_cols=['Player_ID']` so the element id survives the merge; name
matching is the fallback, not the plan.

Card heights in these lineups are computed in Python and the iframe does not scroll
(`components.html(..., scrolling=False)`), so every line-height in the CSS is
load-bearing — underestimate one and the last card is silently clipped.

### Player Display Names

**Always render player names via `to_display_name()` (`scripts/common/text_helpers.py`).
This is the preferred format for player names on every page.**

Neither raw FPL field is presentable on its own. The bootstrap's full legal name
is what nobody says out loud ("Bruno Borges Fernandes", "Rúben dos Santos Gato
Alves Dias"), and `web_name` is frequently abbreviated or too terse to stand
alone ("B.Fernandes", "A.Becker"). `to_display_name(first_name, second_name,
web_name)` returns the common name — "Bruno Fernandes", "Alisson Becker",
"David Raya", "Matheus Cunha" — handling dotted initials, mononyms
("Gabriel"), and players whose surname sits in the `first_name` field
("Igor Thiago").

Keep the full legal name in a `Player` column for matching, and put the rendered
value in `Display_Name`. **Match on `Player`, display `Display_Name`** — swapping
them silently degrades match rates, since projection sources are matched against
the full name.

### Caching

Two-tier caching strategy for fast page navigation:

1. **Startup preload** (`main.py`): Uses `@st.cache_resource` to preload core data once per session:
   - Player mappings, bootstrap static, league entries, Rotowire projections
   - Runs on first page load with "Loading app data..." spinner

2. **Function-level caching** (`utils.py`): Uses `@st.cache_data` with TTL values:
   - 1 hour: player mappings, Rotowire projections, draft picks
   - 10 minutes: league entries, team compositions, H2H records
   - 5 minutes: bootstrap static, league standings, ownership data

Gameweek is cached at module level in config.py via lazy loading.

## Data Plausibility Testing

The app's worst bugs have not been logic errors. They were upstream data quietly
changing shape while every mocked unit test stayed green:

- Rotowire published a *"Best FPL Picks for Gameweeks 1-5"* article whose `Points`
  column was a cumulative 5-week total. Discovery selected it, nothing crashed, and
  every projection app-wide was 5x too big (a goalkeeper showing 18.6 for one GW).
- The Draft API returns a full 38-gameweek score grid from day one, so preseason
  every historical score is `0`. `_estimate_score_std` returned `0.0`, the caller
  substituted a denominator of `1.0`, and a 1.1-point projection edge rendered as
  an 85% win probability.

Neither is detectable by asserting on mocked inputs. What they share is that the
*output* was far outside any plausible norm — cheap to assert automatically.

### `scripts/common/data_validation.py`

Pure, network-free predicates over real numbers. Each returns a list of `Issue`
tuples (`check`, `severity`, `message`, `hint`); `raise_on_error()` turns errors
into an `AssertionError` with the hint attached. No Streamlit, no requests — safe
to import from GitHub Actions.

| Check | Catches |
|-------|---------|
| `check_single_gw_projections()` | Multi-GW or season-long tables masquerading as one gameweek; truncated/unparsed tables; all-zero sources |
| `check_score_std()` | Degenerate sigma (0, NaN, negative) and sigma taken over cumulative season totals |
| `check_win_probability()` | Near-ties reported as near-certainties; extreme calls on small gaps; probability that contradicts the scoreline |
| `check_projected_team_total()` | Inflated XI totals; illegal lineup sizes |
| `check_source_scale_agreement()` | Two sources denominated in different units (the original bug's signature) |
| `check_merge_match_rate()` | A name-based merge quietly ceasing to match (the 356/425 season-rankings regression). Pass `input_rows`: the check **abstains** when the merged frame is smaller than the reference, since a subset can never claim every reference row — judged against the reference alone it logged an ERROR on every Waiver Wire load while matching 100% of its input |
| `check_initial_squad()` | Illegal or implausibly-priced Classic squads; a scale-free objective, whose signature is unspent budget |
| `check_team_strength()` | Degenerate power rankings — every team scoring ~50 because position codes were `GKP/DEF/MID/FWD` instead of `G/D/M/F`, short squads, impossible injury costs |
| `check_element_states()` | Draft player states changing shape — an unknown `status` code, `owner` disagreeing with the status, an owned count that isn't teams x 15. Every one makes locked players read as available, so the Waiver Wire suggests players who cannot be picked up |

Ranges are deliberately wide — these are "this cannot be right" boundaries, not
"this looks unusual" ones. A check that cries wolf gets muted. Note the XI floor is
permissive on purpose: a squad with several players not expected to start
legitimately projects in the high teens, because **absence from Rotowire's weekly
table is itself the "not starting" signal** (it lists 20 teams x 11 = 220 projected
starters, so an unlisted player scoring 0 is correct, not a matching failure).

### `DataFrame.get()` is not a safe accessor

`df.get("Missing")` returns **None** and `df.get("Missing", 0)` returns the scalar
`0` — neither is a Series. So the natural-looking

```python
pd.to_numeric(df.get(col), errors="coerce").fillna(0)   # WRONG
```

raises `AttributeError: 'numpy.float64' object has no attribute 'fillna'` the
moment the column is absent, which is the exact case the `.fillna()` was written
for. Use `numeric_col(df, col, default)` (`analytics.py`) instead — it returns a
Series aligned to the frame's index either way.

This is a *latent* crash: the column is normally present, so it only fires on a
degraded upstream or an early-season frame — precisely when the page can least
afford it. It took down Draft Power Rankings, and the live suite filed it as
"Draft league strength **unreachable**".

### Test layers

| Layer | Location | Runs | Catches |
|-------|----------|------|---------|
| Unit tests for the checks | `tests/common/test_data_validation.py` | Always, offline | A check silently ceasing to fire. Every "bad data" fixture is a real number the app actually displayed. |
| Live plausibility | `tests/live/` | Default run; skips offline | Upstream sources changing shape. Contract: **unreachable → SKIP, reachable but implausible → FAIL.** |
| Page wiring | `tests/draft/test_fixture_page_wiring.py` | Always, offline | Two callsites of the same function being fed different inputs (the Fixtures Overview omitting `ffp_df`). Plausibility can't catch this — both numbers are individually plausible. |

`tests/live/conftest.py` clears the `FPL_CURRENT_GAMEWEEK` / `ROTOWIRE_URL` pins that
the root conftest sets, so live tests resolve real config. Everything there is
auto-marked `live`.

**Only transport failures skip.** `skip_if_unreachable()` used to catch bare
`Exception`, so the `AttributeError` above was reported as an outage and four
Power Rankings checks stood down while the page was broken. It now inspects the
exception chain — a wrapped `requests` error is still an outage, anything else
fails with "this is a bug in our code". A test that skips itself when the code is
wrong is worse than no test: it is a green tick over a failure.

**When adding a data source or a derived metric, add a plausibility check for it.**
The question to answer is not "is this value correct" (untestable against live data)
but "is this value *possible*".

## Transfer Scoring Model — Design Reference

This section documents the philosophy, reasoning, and specification behind the transfer/keep scoring system used in both Draft Waiver Wire and Classic Transfers. It serves as a reference for future modifications.

### Core Philosophy

The scoring model answers two questions: **"Who should I pick up?"** (Transfer Score) and **"Who should I hold?"** (Keep Score). Rather than a single opaque number, the system exposes four transparent columns: **1GW** (this-gameweek value), **ROS** (rest-of-season value), **Transfer Score**, and **Keep Score** (dynamic blends of 1GW + ROS that adapt to context).

### 1GW Score — Pure Expected Value

**Principle**: 1GW should reflect *expected points this gameweek* and nothing else. Form, season points, and FDR are intentionally excluded because Rotowire and FFP projections already incorporate those signals — adding them again would double-count.

```
blended_projection = 0.6 × Rotowire + 0.4 × FFP Predicted  — use whichever is available
start_likelihood   = FFP Start% (primary) | FPL chance_of_playing (fallback) | 100%
effective_proj     = blended_projection × start_likelihood

1GW = positional_percentile(effective_proj)
```

**Key design decisions**:
- **Start likelihood is critical**: A player with 5.0 projected but 50% start chance is really worth 2.5. The old model ignored this entirely.
- **FFP Start% is primary** because it provides continuous 0-100 granularity. FPL's `chance_of_playing` is coarse and often missing. Rotowire is binary (in/out).
- **No form weight**: Projections already embed form. Adding it separately inflates recent-hot players beyond what the data supports.
- **No FDR weight**: Projections already adjust for opponent strength.
- **Rotowire + FFP blend**: Rotowire covers projected starters only (expert opinion); FFP covers everyone who might play. The absence of a Rotowire projection is itself a signal. When both exist, average them; when only one exists, use it.

### ROS Score — Multi-GW Dominant, Dynamic Weights

**Principle**: ROS represents long-term player value. Multi-GW projections (FFP Next3GWs) are the strongest forward-looking signal and should be the dominant input, especially early season. As the season progresses, actual performance data (season points) becomes more trustworthy.

```
p = season_progress_weight(current_gw)  // 0.10 at GW1 → 0.95 at GW38

season_quality = p × season_pts_pctile + (1-p) × season_proj_pctile

w_mgw   = 0.40 - 0.10×p   // 40% → 30%  (multi-GW projections — FFP Next3GWs)
w_sq    = 0.30 + 0.15×p   // 30% → 45%  (season quality — actual + projected blend)
w_form  = 0.15 - 0.05×p   // 15% → 10%  (trajectory indicator)
w_start = 0.10            // 10% constant (start consistency — nailed-on starters)
w_fdr   = 0.05            //  5% constant (supplementary fixture signal)

ROS = w_mgw × multigw_pctile + w_sq × season_quality
    + w_form × form_dampened_pctile + w_start × start_consistency_pctile
    + w_fdr × fdr_ease_pctile
```

**Key design decisions**:
- **Multi-GW at ~40%** (was 20%): FFP's 3-week window captures upcoming fixture runs and is the most actionable forward-looking signal. 3 GWs is the sweet spot — long enough to capture a fixture run, short enough to be reliable.
- **Season quality grows over time**: Early season, trust preseason projections; late season, trust actual performance. The `season_progress_weight` concave curve shifts this trust faster than linear.
- **Form is a trajectory indicator** (15→10%): A player at #10 in their position on strong form is likely heading to #8 soon. Form matters *more* for ROS than 1GW because it signals where positional ranking is heading. Dampened by starts to avoid overvaluing small-sample hot streaks.
- **Start consistency at 10%** (constant): Uses FFP `LongStart` (long-term start %) to reward nailed-on starters. A rotation player who starts 50% of games should be worth less for ROS even if per-game stats are good. Critical for Draft where dropped players go to the waiver wire.
- **FDR at 5%** (small, constant): Fixtures beyond the multi-GW window are supplementary — most of the fixture signal is already captured by multi-GW projections.

### Transfer Score / Keep Score — Dynamic Alpha Blend

**Principle**: The optimal blend of 1GW and ROS depends on context — format, position, player quality, and squad depth. Rather than fixed weights or the old TILT mechanism, a dynamic alpha adapts the blend per player.

```
Score = α × 1GW + (1-α) × ROS
```

**Alpha adjustments** (applied in order, then clamped to [0.15, 0.75]):

| Factor | Adjustment | Reasoning |
|--------|-----------|-----------|
| **Format baseline** | Draft α=0.35, Classic α=0.55 | Draft players are harder to replace — ROS stability matters more. Classic allows easy weekly transfers, so 1GW impact matters more. |
| **Position: GK** | α -= 0.10 | GK waiver wire is extremely thin. Dropping a starting GK for a short-term upgrade is dangerous — the sub will lose their spot when the injured player returns, and finding a new starting GK is nearly impossible. |
| **Position: FWD** | α -= 0.05 | Similar but less extreme depth concern as GK. Only 3 FWD slots. |
| **Position: DEF/MID** | No change | Deeper pools, more roster flexibility. |
| **Elite player (ROS > 0.80)** | α -= 0.10 | Elite positional players are irreplaceable. We hold them through injuries and bad fixtures. A top-5 MID having one bad GW is not a reason to drop. |
| **Above avg (ROS > 0.60)** | α -= 0.05 | Good players deserve some protection but less than elite. |
| **Below avg (ROS < 0.40)** | α += 0.05 | Replacement-level players — 1GW matters more. If they aren't performing now, there's no long-term value to protect. |
| **Critical squad depth** | α += 0.15 | Urgency — we need someone who plays THIS week. |
| **Low squad depth** | α += 0.10 | Moderate urgency to fill the gap. |

**Why TILT was removed**: The old TILT mechanism explicitly tilted add/drop scoring differently (adds favored projections, drops favored season points). The dynamic alpha blend achieves the same protective behavior naturally — an elite ROS player on your roster gets high ROS weight (protected from panic drops), while a low-depth position triggers high 1GW weight (targets someone who plays NOW). The dynamic blend is more elegant and handles more edge cases than the fixed asymmetry.

### Data Source Hierarchy

| Signal | Primary Source | Fallback | Used In |
|--------|---------------|----------|---------|
| Single-GW projection | Rotowire + FFP Predicted (blended) | Rotowire only or FFP only | 1GW |
| Start likelihood | FFP Start % | FPL chance_of_playing → 100% | 1GW |
| Multi-GW projection | FFP Next3GWs | single_gw × 3 | ROS |
| Season projection | Rotowire Season Rankings | Season points (actuals) | ROS (season_quality) |
| Start consistency | FFP LongStart | FPL starts count | ROS |
| Form | HealthyForm (element-summary) | FPL form → points_per_game | ROS |
| FDR | AvgFDRNextN / AvgFDR | Default 3.0 | ROS |

### Implementation Reference

| Function | File | Purpose |
|----------|------|---------|
| `compute_player_scores()` | `scripts/common/analytics.py` | Core scoring — computes all 4 columns + `_effective_proj` |
| `compute_dynamic_alpha()` | `scripts/common/analytics.py` | Per-player alpha based on context |
| `merge_ffp_single_gw_data()` | `scripts/common/analytics.py` | Merges FFP Predicted/Start/LongStart onto player DataFrames |
| `blend_multi_gw_projections()` | `scripts/common/analytics.py` | Merges FFP Next3GWs (falls back to PPG×3 if unpublished) |
| `blend_fixture_projections()` | `scripts/common/analytics.py` | Lightweight fixture display blend: Rotowire 60% + FFP 40% × start likelihood → `Proj_Blended` column. No percentile computation. Uses `Proj_Blended` (not `Blended_Points`) to avoid collision with the live-blending column. |
| `positional_percentile()` | `scripts/common/analytics.py` | Within-position percentile against full FPL pool |
| `season_progress_weight()` | `scripts/common/analytics.py` | Concave GW→weight curve for season quality blend |

All scores are **positional percentiles** (0-1) computed against the full FPL player pool (~700 players). A score of 0.85 means "top 15% at this position" — immediately interpretable regardless of position.

**FFP has two prediction bases — blend the conditional one.** FFP publishes
`StartingPredicted` (points *if he starts*, the same basis as a Rotowire
projection) and `Predicted`, which is that number already multiplied by start
probability: verified live at `Predicted == StartingPredicted * Start/100`,
r=0.9998. `compute_player_scores()` blends `StartingPredicted` and applies start
likelihood once itself. Blending `Predicted` instead charges the start
probability twice and ran the FFP term ~44% low at a 60% median start rate —
a silent, everyone-slightly-too-low distortion. When only `Predicted` is
available the conditional value is recovered by dividing the start rate back
out. `tests/live/` pins the relationship so a change at FFP surfaces as a
failure rather than a quiet re-scaling.

### Waiver Wire names come from the bootstrap, not the frame

Every frame on that page carries whichever name its source publishes — the Draft
roster and FPL stats use the full legal name ("Sávio Moreira de Oliveira"),
projections use Rotowire's — and the cards used to render it raw.
`_attach_display_names()` resolves a `Display_Name` per row through
`to_display_name()`, by element id first and normalized name second, and every
rendered name (cards, roster table, available table) goes through `_display_of()`.
`Player` is untouched: it is what the merges on that page key on.

The name-only fallback keeps a key **only when it resolves to exactly one
player**. A shared surname would otherwise print Cole Palmer's name on Alex
Palmer's card — the display-side version of the match bug in "Player Matching".

### Suggestion breadth — Draft Waiver Wire

`_compute_transfer_suggestions()` searches a *window*, not the board: by default
the two weakest roster players at a position against the five strongest available
ones, stopping at the first move it finds per position. That window is the
compact "Best per position" view and is deliberately narrow — a wide search
surfaces the same standout target against every mediocre player you own.

The window is parameterised (`roster_candidates`, `avail_candidates`,
`one_per_position`, `positions`), and the page's **Show** control lifts it: with
both limits `None` the search runs the whole roster against the whole available
pool and returns **at most one move per droppable roster player** — the best add
that clears that player's threshold. So one target can head several cards; the
caption says so rather than letting it read as a repeat.

Two things make the full scan cheap. Available players are sorted by adjusted
value descending, so once one misses a drop's threshold no lower-ranked player
can clear it and the inner loop breaks. And `DEBUG_PAIR_CAP` bounds the
transparency expander, which otherwise renders every evaluated pair — 15 roster
players against 150 available ones is a wall of rows, not transparency.

**`_effective_proj` column**: `compute_player_scores()` retains `_effective_proj` (blended_proj × start_likelihood) in its output. Consumers (Waiver Wire suggestion engine, card rendering) rely on it for GW projection display and sanity checking. Do not drop it from the result.

**FFP name matching goes through `ReferenceMatcher`.** `merge_ffp_single_gw_data()`,
`blend_multi_gw_projections()` and `merge_season_projections()` all call
`_claim_reference_rows()` — see "Player Matching" for the tiers.

They used to carry a hand-rolled 4-step ladder each, whose last two tiers were
team-agnostic **and position-free**. Measured live on 2026-09-03, 53 of 652 pool
rows resolved on a shared surname alone: Kalvin Phillips (MID) held Dillon
Phillips' goalkeeper start rate, Abu Kamara held Boubacar Kamara's 90%, and João
Pedro's `Next3GWs` was Gabriel Jesus's. Since `FFP_Start`/`FFP_Starting_Predicted`
drive **1GW** and `MultiGW_Proj` is **40% of ROS**, that scored real players on
other players' data across the Waiver Wire, Classic Transfers, Initial Squad and
both fixture pages — silently, because every value involved was plausible.

Positions are normalized on both sides (`POS_MAP_TO_RW`, accepting `Position`,
`position_abbrv` or `element_type`); FFP publishes `GK/DEF/MID/FWD` and the app
uses `G/D/M/F`. A frame with no position column still merges but gets exact
`(name, team)` only, which costs matches rather than inventing wrong ones.


## Initial Squad Model — Classic FPL

`scripts/classic/initial_squad.py`. Answers "what 15 should I start the season with?"

**The objective is denominated in expected points per gameweek, not percentiles.**
This is the single most important thing about this page, and the opposite of how
Draft scoring works.

```
SeasonPG  = SeasonProjection / 38          # Rotowire Top-400, as a per-GW rate
GW1PG     = _effective_proj                # blended GW1 projection x start likelihood
OpeningPG = GW1PG * (1 + 0.10*(3.0 - Team_AvgFDR))   # "fast start"
ExpPts    = 0.70*SeasonPG + 0.30*OpeningPG
CapPts    = 0.85*SeasonPG + 0.15*GW1PG     # armband goes on a week-in producer
```

**Fixtures modify the opening term only, never the blend.** Multiplying the whole
expression -- as this once did -- inflates the *season-long* projection for a soft
opening month, but that projection already prices in all 38 fixtures. The player
will have been transferred around a bad opening run long before it costs a
season. Confining the tilt to the opening term also lets it be meaningful: at 10%
per FDR point it is worth ~0.25 expected points across the league's actual FDR
spread, against a ~1.9-point gap between the best and 10th-best midfielder -- a
tiebreaker between similar players, never an override of quality.

The UI therefore exposes **one** trade-off, not three: season-long quality versus
a fast start, as two sliders bound to total 100%.

Percentiles are the right currency for Draft, where you compare rank and there is
no budget. They are actively wrong for Classic, where the budget makes you buy
points per pound — and they fail in two compounding ways:

1. **They saturate.** Haaland projects 213.7 season points against a mid-price
   midfielder's 178.3 (+20%), but as percentiles that is 0.974 vs 0.977. The
   premium ranks *lower*, because percentile has no headroom above 1.0.
2. **They invert across positions.** Percentile is computed within position, so
   the best of 32 forwards scores 31/32 = 0.969 while the best of 48 midfielders
   scores 47/48 = 0.979. Depth of pool, not quality, decides the ranking.

The live symptom was a page that refused to buy any premium, captained a £7.0m
midfielder, spent £99.0 of £100, and put £5–7m players on the bench.
`check_initial_squad()` catches all of it; **underspend is the tell.**

Percentile columns (`Season Score`, `Week1 Score`, `Player Score`) are still
computed and displayed — they read far better than raw points — but nothing
optimizes on them. The captain bonus is also in points, so `captain_bonus_weight
= 1.0` models what the armband is actually worth (a doubled score) rather than a
rounding error on a percentile.

**ROS is deliberately unused** here — it depends on live-season signals (form,
starts, multi-GW FFP data) that don't exist before GW1.

**Unranked players are floored below every ranked player at their position.** A
player outside a 400-deep season ranking is not average; absence is itself the
signal, the same way absence from Rotowire's weekly table means "not starting".

The floor must anchor to the position's *ranked minimum* (times
`UNRANKED_FLOOR_FRACTION`), **not** a quantile inside the ranked distribution.
Rotowire does not sample positions evenly — it lists roughly one goalkeeper per
club, so only ~21 of 67 GKs are ranked and all of them are starters. A
10th-percentile-of-ranked floor therefore paid every backup keeper a starter's
rate (2.65 against a backup defender's 0.40) and bought them onto the bench.

**Dead weight** — players in neither the season rankings nor the GW1 table — is
dropped from the candidate pool by default (`_drop_dead_weight()`). Bench Boost
makes all four bench slots live, and a ranked replacement costs the same at
DEF/MID/FWD and £0.5m more at GK. This is a different lever from bench weight:
bench weight asks how much bench *points* count, this asks whether a slot can
score at all. Positions that would be left too thin keep their dead weight — an
unsolvable squad is worse than one weak slot.

Both Rotowire URLs (season and GW1) are pinned in `config.py` — neither slug is
auto-discoverable — and both are editable in the page's **Data Sources** panel,
which shows fetch status, row counts and **match rate** per source. That panel
exists because every fetch degrades to an empty frame on failure: without it, a
broken season URL renders a page identical to a working one, just with every
Season score silently at 0.50.

## Transfer Risk Model — "will he still be here?"

`scripts/common/transfer_risk.py` (pure), `transfer_feeds.py` (fetch),
`transfer_risk_app.py` (name matching). Answers the question that let Ollie
Watkins be drafted at rank 32 weeks before moving to Al-Hilal: a player who
cannot score a point all season was priced as if he would play 38 games.

```
transfer_multiplier = max(TRANSFER_FLOOR, 1 - risk * exposure)
```

Applied the way `injury_multiplier()` is, and returning exactly `1.0` for a
player with no news, so it is safe to apply unconditionally.

### Destination weight is the whole model

**The line is Premier League membership, not geography.** These two live
bootstrap rows cost a manager exactly the same — everything:

```
'u' Reijnders | Has joined Al Qadsiah permanently
'u' Watson    | Has joined Leicester City on loan for the rest of the season
```

A Championship loan and a Saudi transfer both score zero. An *intra*-PL move
scores `0.20` — in Draft you simply keep the player. Membership is read from the
bootstrap `teams` list, never hardcoded, so promotion and relegation need no
maintenance. An unparsed destination gets `0.60`: most rumoured exits leave the
league, so a parse failure must not read as safety.

Two traps this code exists to avoid:

1. **The player's own club is not his destination.** "Al-Hilal agree deal to sign
   *Aston Villa* striker Ollie Watkins" names two clubs. `parse_destination`
   takes `exclude_team`, and `team_code()` resolves it whether the frame carries
   a short code (`MCI`) or a name (`Man City`) — they disagree across sources.
2. **Club spellings differ.** The bootstrap says `Nott'm Forest`, headlines say
   `Nottingham Forest`. Resolve through `TEAM_FULL_TO_SHORT`, never raw strings,
   or an intra-PL move reads as an exit — the expensive direction of error.

### Sources, and why the obvious one was rejected

| Source | Role |
|---|---|
| Google News RSS, one quoted query per player | **Predictive.** Free, no API key, timely — it carried the Watkins medical days before completion. |
| FPL bootstrap `status='u'` + `news` | **Ground truth.** Resolves a move outright and overrides all news scoring. |
| Bookmaker next-club odds | **Rejected.** `footballtransfers.co.uk/odds/ollie-watkins` was stale to 06 Mar 2026, listed his club as "Unknown", and priced Any Saudi Club at 15/1 (6.3%). As the primary signal it would have left him top of the board. The Odds API carries no transfer markets at all. |

Speculation never outranks a completed deal, and a *completed* departure sets
`exposure = 1.0` — there is no window left to wait on.

### Scoring the news

Confidence comes from breadth of agreement, not any single headline:

```
base    = strongest keyword tier seen, decayed (10-day half-life, 30-day window)
outlets = distinct sources carrying a Tier-A or Tier-B headline
risk    = base * (0.30 + 0.70 * min(1, outlets/4)) * destination_weight
```

Tier A (0.85) is language that commits — "medical", "agree deal", an agreed fee.
Tier B (0.50) is "bid", "in talks", "exit". Tier C (0.25) is "linked",
"monitoring". **Denials cap the score at Tier C**, so "Villa rule out Watkins
sale" cannot score Tier A on the word "sale".

**Ambiguous names need a corroborating signal.** A query for the mononym
"Gabriel" returns news about every Gabriel in the league, and discounted
Arsenal's by 71% off other players' stories. `build_ambiguous_tokens()` derives
the ambiguous set *from the pool itself*, and those names additionally require
the club or first name in the headline. Same lesson as the Alex/Cole Palmer
match: a weak key needs corroboration. Mononyms also get the club added to their
search query, since quoting a single word is not a search.

### Exposure — why it switches itself off

Exposure is the fraction of the remaining season a completed move would cost,
measured in **days to season end** rather than gameweeks, because windows are
dates:

```
exposure = days_left_after_next_window_close / days_left_now
```

**Windows are per destination region.** This is the Watkins trap precisely: the
English window shut 1 Sep 2026 but the Saudi one ran to 12 Oct, so a player could
be sold out of a squad five gameweeks into a season whose window had "closed".

Pre-draft that is ~0.92 for a European move and ~0.84 for a Saudi one. After the
January deadline no window remains, exposure is 0, and every multiplier returns
to exactly 1.0 for the rest of the season — the feature turns itself off rather
than lingering as a stale discount.

**`TRANSFER_WINDOWS` is hardcoded and season-specific.** It cannot be discovered
and must be updated each season; `check_transfer_windows()` warns once it lapses,
and a live test fails on it, because a lapsed calendar makes the whole feature a
silent no-op indistinguishable from "nobody is at risk".

### Inbound — "will he still play?"

The outbound model asks whether a player will still be here. `build_inbound_watchlist()`
and `apply_minutes_competition()` ask the other half: who is *arriving*, and whose
minutes that costs. Arrivals are worth knowing twice over — they are waiver targets
the moment they enter the game, and they are the reason an incumbent is about to lose
his place.

Queries are per **club**, not per player, because the interesting arrivals are not in
the FPL pool yet — there is no name to query with. That makes this path structurally
noisier than the outbound one: a per-player query is already about that player, while
a per-club query is about the club and the name has to be pulled out of prose.
Everything is therefore conservative — corroboration required, discount capped at
`MAX_MINUTES_IMPACT` (0.25), unparsed names dropped rather than guessed.

**The buying club comes from the sentence, never from whose feed the headline arrived
on.** "Nottingham Forest sign Marc Guehi from Crystal Palace" surfaces under a *Palace*
query and describes *Forest* buying. Trusting the query would discount the selling
club's squad — precisely backwards. `_club_before()` takes the club preceding the
signing verb.

**Arrivals need their own tier vocabulary.** `classify_headline()` is written for
"is he leaving": its Tier A is a player departing, and the strongest inbound sentence
there is — "Villa complete signing of Nicolas Jackson" — matches nothing in it and
scores `0.0`. Reusing it dropped confirmed signings from the watchlist while rumours
survived. `classify_signing()` shares the three tiers and the denial cap so both sides
stay comparable, but nothing else.

**A fee belongs to whoever is nearest it.** "Liverpool agree £123m Barcola deal as
Gakpo decides to join Man City" names two players and one fee, and the fee is not
Gakpo's. `parse_fee_for_player()` requires the amount to sit nearer this player's
surname than any other player's in the pool — the same lesson as a player's own club
not being his destination, and it matters more here because fee is the evidence for
how big a role a signing takes. A bare "£10" is never a fee: transfer fees are always
written with a magnitude.

**The two effects are one event seen from both ends.** Nicolas Jackson arrives at Villa
*because* Ollie Watkins is going to Al-Hilal, so a player who is himself leaving is
exempt from minutes competition — charging Watkins for his own replacement counts one
move twice. `Minutes_Mult` is kept separate from `Transfer_Mult` for the same reason:
one answers "will he be here", the other "will he still play".

The established first choice at a club and position absorbs only
`INCUMBENT_TOP_SHARE` of the threat — a signing displaces the players behind him long
before it displaces a starter.

**A signing must not compete with himself.** Once he is in the ranked pool he
matches his own club and position: "James Trafford completes club-record £40m move
to Leeds" discounted Leeds' new goalkeeper *for arriving*. `_same_player()`
excludes him — deliberately narrow, since the two sources differ in accents
("Emiliano Martínez" vs "Emiliano Martinez") and in completeness (a headline
routinely prints the surname alone). He remains competition for everyone else at
the club.

**A deal that fell through reads exactly like one that happened.** "Monaco pull
out of selling midfielder to Chelsea in £47m deal" scored Tier A on "£47m deal"
and discounted four Chelsea midfielders. Withdrawal language (`pull out`,
`withdraw`, `priced out`, `off the table`, `ends interest`) lives in
`_NEGATION_PATTERNS`, which caps both directions at Tier C. Both of these were
found by running the pipeline against live feeds, not by unit tests — these are
not shapes you invent.

**`WEIGHT_INTRA_PL` is 0, deliberately.** A move inside the league costs a Draft
manager nothing, and the sign is genuinely ambiguous: Gakpo leaving a crowded
Liverpool front line for a starting role elsewhere is plausibly an upgrade. A 20%
discount asserted a direction the evidence does not support. The move is surfaced
through `Transfer_Status` (`At risk` / `Moving` / `Departed`) instead, so it can be
judged by eye — and a completed intra-PL move reads "stays in EPL", not "Departed".

### Wiring

`draft_helper.py` discounts Rotowire's season `Points` into `Adj Points` and
re-ranks on it, behind an "Adjust for transfer risk" toggle. `Risk` sits directly
after `Position` — it explains the ordering, so it must be readable without
scrolling.

The inbound side rides the same page under its own "Adjust for incoming signings"
toggle: an **Incoming signings** expander lists the arrivals, and `Minutes_Mult`
multiplies into `Adj Points` alongside `Transfer_Mult`. **`Adj Points` shows
exactly the adjustments that are switched on** — computing the full discount and
then only re-sorting conditionally would let the column and the ordering disagree
about what was applied.

Club news is 20 requests against the player scan's 150 (~1.6s for ~2,000
headlines), so it rides the same **Scan** button and the same background prefetch
rather than getting controls of its own.

**Fetching is the slow part, and three things keep it usable:**

- **Concurrency, not pacing.** The work is pure network latency, so it
  parallelises: `ThreadPoolExecutor` over a shared `requests.Session` does 150
  players in ~8s where a serialised loop with a sleep took ~2 minutes.
- **The cache is keyed per player, not per batch.** Scans are therefore
  incremental — widening the depth from 150 to 175 fetches 25 players, not 175.
  **A player with no news caches an empty list, which is a cache *hit***; treating
  it as a miss refetches the quiet majority of the board every single time.
- **It starts before anyone asks.** `start_transfer_news_prefetch()` warms the
  cache on a daemon thread at app startup (gated on a configured Draft league, so
  Classic-only users don't pay the ~1.3s board scrape) and again on page load. The
  worker touches only Streamlit-free code — calling `st.*` from a thread with no
  ScriptRunContext would fail.

**Do not put `@st.cache_data` on `get_transfer_news`.** It was there once and
memoized the *empty* "nothing cached yet" result from first page load; after a
scan, every rerun — searching, filtering, toggling — was served that stale empty
frame and the risk columns silently vanished mid-draft. The SQLite layer is the
cache. The computed frame is also parked in `st.session_state` as a backstop,
since every Streamlit interaction reruns the whole page function.

Import constraints, which are load-bearing: `transfer_risk.py` and
`transfer_feeds.py` use plain `logging` and avoid `error_helpers`, `cache` and
`player_matching` — all three reach Streamlit, and GitHub Actions must be able to
import these. Cross-source name matching therefore lives in `transfer_risk_app.py`.

Validation: `check_transfer_risk()` errors on an empty frame (a broken feed
renders a page identical to a working one), on a multiplier above 1.0 (which
would *inflate* a projection), and on more than 10% of a 50+ player pool being at
risk — that signature means the matcher broke, not that the league is emptying.

## Transfer Odds Model — the market as a third opinion

`scripts/common/transfer_odds.py` (pure), `odds_feeds.py` (fetch),
`transfer_risk_app.attach_odds()` (name matching). Surfaced on the Availability
page. News tells you a story is being written; a price tells you what somebody is
willing to be wrong about, and a market can exist for a player no headline names.

**Bookmaker odds were rejected once and that verdict was half right.** The
rejection judged `footballtransfers.co.uk/odds/<slug>` — a page that still shows
the player's club as "Unknown" behind JavaScript loading spinners. What it missed
is that the *numbers* on that page are server-rendered in a plain HTML table, and
that `/odds` carries a live index of ~57 markets in a React payload. A stale price
is usable the moment it is **labelled** stale, which is the whole design here.
`oddschecker.com` returns 403 to automated fetches; `bettingodds.com` publishes
prices inside prose and its FAQ still says the window shuts in February 2024.

### A ladder is not a probability distribution

The single most important thing in the module. A live ladder:

```
Any Saudi club   8/11   57.9%     <- contains Al Ittihad and Al Hilal
Al Ittihad        7/4   36.4%
Any MLS Team      5/2   28.6%
Al Hilal          7/1   12.5%
Any French club   8/1   11.1%
Any Italian club  8/1   11.1%
                        157.6%
```

That excess is mostly **overlap**, not margin. Normalising it whole reports Saudi
at 37% where the market says 58%, understating every row by counting one outcome
three times. `disjoint_ladder()` keeps each outcome once — the aggregate wins,
being the broader market — which takes Salah from 1.58 to 1.09. `group_ladder()`
then re-attaches the member clubs for display *without* summing them, because the
specific clubs are the interesting part.

A club whose region cannot be resolved is **kept**, not dropped: unknown
vocabulary must cost a visible overround, never a silent lost destination.

**Do not read the excess as margin or back a P(leaves) out of it.** These are
independent binary bets ("Barcola to Liverpool"), each carrying its own margin,
not one coupled book over a partition — the Mateta ladder totals 1.75 across six
clubs that do not overlap at all.

### What the two numbers mean

No bookmaker prices "stays at Liverpool", so normalising cannot yield P(leaves) —
it would force it to 1.0 by construction. Hence two different questions:

| Output | Question |
|---|---|
| `normalise_ladder()` | *Given that he moves*, where to. Sums to 1.0. |
| `exit_probability()` | The shortest quoted price on a departure. A floor on leaving at all. |

### Staleness is measured, never assumed

```
odds_weight = 0.5 ** (age_days / 45)      # vs 10-day half-life for news
```

Prices move slower than headlines, hence the longer half-life. The live Salah
quote is stamped `semanticOddsUpdatedAt = 2026-03-25` — 163 days old, weight
0.08. It renders, banded 🔴 archival, and barely moves the score. **A missing
timestamp is treated as 30 days old, not fresh**: a parse failure must not read
as confidence, the same reasoning as `WEIGHT_UNKNOWN`.

### Blending

```
blended = (news*W_NEWS + odds*odds_weight*W_ODDS) / (W_NEWS + odds_weight*W_ODDS)
```

Returns `news_risk` exactly when there is no usable quote, so it applies
unconditionally. Note a zero `news_risk` is an *observation* — the model looked
and found nothing — so a live 58% market with no headlines blends to ~0.25, not
0.58. Damping that way is deliberate: the opposite error prices a name-matching
failure as a transfer saga.

**Odds are consulted only where the bootstrap has not resolved the move.** A
completed deal is ground truth and a price is speculation; letting a stale quote
reopen a settled question is exactly how Watkins came to be priced at 6% months
after he had gone.

### Two traps found by running it against live data

1. **A market whose destination is the player's own club has already settled.**
   The feed still quotes "Bradley Barcola to Liverpool" after Liverpool signed
   him; scored as exit risk that says a new arrival is 25% likely to leave. Four
   of 25 matched markets were in this state. `attach_odds` drops them, resolving
   club names through `team_code()`/`TEAM_FULL_TO_SHORT` — never raw strings.
   This is the inverse of `parse_destination`'s `exclude_team` trap.
2. **Uniqueness inside the pool is not enough for a name-only key.** The odds
   feed quotes players who have *left* the league. With Darwin Núñez gone from
   the FPL pool, the surname "nunez" resolved uniquely — to Marcelino Núñez, who
   was handed Darwin's market. The fallback is therefore a **token subset** in
   either direction (`"bruno fernandes" ⊂ "bruno borges fernandes"`), never a
   bare surname, and ambiguity still resolves to no match. Same lesson as
   Alex/Cole Palmer, one step further out.

`attach_odds` cannot use `ReferenceMatcher`: the feed publishes a bare name and
the club the player is going *to*, with no position, and every tier of the shared
matcher below the first is scoped to team or position.

### Validation

`check_transfer_odds()` errors on an empty ladder, an implied probability outside
`(0, 1]`, a disjoint total below 1.0 (a bookmaker does not offer an arbitrage, so
a row is missing — usually the favourite), and normalised shares not summing to
1.0. The precise overlap detector is a **comparison against `ladder_overround()`
itself**, because a threshold cannot separate 1.58 from 1.09 — both are inside
any honest band. A high total and an old quote are warnings, not errors.

`check_transfer_risk()` now excludes bootstrap-resolved departures from its
at-risk *fraction* check. Confirmed departures score 1.0 by construction, and the
Availability tracker deliberately lists them — judged against the whole frame it
failed at 82% while working perfectly.


## Team Strength Model — Draft Power Rankings

`scripts/common/team_strength.py`. Answers "how good is each roster in my league?"

**Everything is a positional percentile against the full FPL pool**, so 85 means "top
15% at this position" regardless of position. Draft enforces 2 GK / 5 DEF / 5 MID /
3 FWD, so a flat mean across all 15 players is directly comparable between teams.

```
quality = p*actuals + (1-p)*pedigree      # p = season_progress_weight(gw)
actuals = 0.70*pctile(points_per_START) + 0.30*pctile(form)
          -> pedigree alone when starts < _MIN_STARTS_FOR_ACTUALS (4)
          -> points-per-start alone when form is flat across the pool (preseason)

raw = 0.60*quality + 0.25*pctile(Proj_Blended) + 0.10*start_security + 0.05*fixture_ease
player_strength = raw * injury_multiplier(gws_missed, current_gw)

Team Score = 100 * mean(player_strength over all 15)
Injury Cost = Healthy Score - Team Score
```

**Points per START, not total points**: a star returning from six weeks out has fewer
total points than an ever-present squad player but is plainly the better asset. Total
points ranks them backwards.

**Injuries scale by fraction of the *remaining* season missed** (`injury_helpers.py`),
so the same five-week absence costs ~14% at GW3 and hits the floor at GW34. Return
dates come from parsing the FPL `news` text.

### Two traps this code exists to avoid

1. **Position codes.** `pull_fpl_player_stats()` returns `GKP/DEF/MID/FWD`; everything
   in `analytics.py` groups on `G/D/M/F`. Feed it the wrong ones and
   `positional_percentile()` matches no group, returns its 0.5 default for every
   player, and the page renders a plausible table where every team scores exactly 50.
   Convert with `_map_position_to_rw()` first. `check_team_strength()` catches it.
2. **Reference-pool contamination.** `positional_percentile()` silently falls back to
   min-max over the *input* frame when the reference lacks the value column — turning
   an absolute percentile into a within-roster one, with no error. And a 2-start
   player scoring 17.5/start will outrank Haaland unless excluded from the
   denominator. `attach_reference_pps()` handles both; do not bypass it.

## Draft Transaction Rules — FPL Platform Reference

The rules FPL Draft actually enforces on adds, drops and trades. Anything the app
suggests must be something the manager can really do; two features shipped advice
that was impossible to act on before this was written down.

### Player states

Every player is in exactly one state, published per-league on
`GET /api/league/{id}/element-status` as `status`, alongside `owner` and
`in_accepted_trade`:

| Code | State | Meaning |
|------|-------|---------|
| `o` | **Owned** | already in a squad (`owner` is the entry id) |
| `a` | **Available** | addable now if free agency is active, otherwise requestable at the next waiver |
| `l` | **Locked** | *cannot be selected yet* |

A player locks when they are **removed from another squad**, added to the game within
24h of a draft/waiver deadline, or added while free agency was active. **Locked players
become available for waiver requests at the next deadline** — they are not gone, they
are next week's waiver targets.

**Ownership data alone cannot see this.** A locked player is on nobody's roster, so an
anti-join against rosters marks them available. That is how the Waiver Wire came to
suggest Oliver McBurnie hours after another manager dropped him.
`get_league_element_states()` (`fpl_draft_api.py`) reads the states;
`_available_from_projections()` flags them; `_compute_transfer_suggestions()` excludes
them — **inside the function, not at the callsite**, so no future caller can forget.

### Waivers vs free agency

Two windows alternate every gameweek, and which one is open decides whether an
available player can be taken *today*:

1. After the draft, unselected players go to waivers.
2. Waivers process ~24h before the gameweek deadline (less when gameweeks are close
   together). Lowest-ranked team picks first; a successful claim sends that team to the
   back of the queue. Claims are position-locked: you propose replacing a squad player
   with an unselected one **in the same position**. Multiple claims must be ranked.
3. Free agency then runs until the gameweek deadline — adds process immediately.
4. At the deadline, all unowned players return to waivers and the cycle repeats.

The active window is `league.transaction_mode` on `/api/league/{id}/details`
(`"waivers"` | `"free-agency"`); `/api/game` carries the global cycle state
(`waivers_processed`, `current_event`, `next_event`, `current_event_finished`,
`trades_time_for_approval`). `get_draft_transaction_window()` merges the two.

### Trades

**A trade swaps the same number of players with identical position composition on both
sides.** 1 MID + 2 FWD for 1 MID + 2 FWD is legal; the same three for 2 MID + 1 FWD is
not, and neither is any unequal shape. This is the rule the Trade Analyzer used to
break — it searched cross-position 1-for-1 swaps and had a whole 2-for-1 finder, both
producing trades that cannot be submitted. `_is_legal_trade()` now gates every proposal
from inside `_score_proposal()`, so only 1-for-1 and 2-for-2 are discoverable.

Offers can be made until the waiver deadline, or 24h earlier where approval is required.
A player may appear in several offers; accepting one invalidates the rest. Accepted
trades cannot be cancelled, and trades process *before* waivers.

League setting lives at `league.trades` on `/api/league/{id}/details`. The four options
are no trades / all trades (immediate) / administrator approval / manager approval
(fails on 50%+ objection); where approval is required, an un-vetoed trade counts as
approved at the waiver deadline. **Only one code is verified: `"a"` = administrator
approval**, confirmed against a league whose admin reported the setting. Do not guess
the others — label an unrecognised code as unknown rather than inventing a mapping.

Trade states: proposed, withdrawn, rejected, accepted, invalid, vetoed, expired,
processed.

### Verified live payload

League 11347, 2026-08-27 — the numbers `check_element_states()` asserts against:

```
element-status : 616 elements -> 446 'a', 150 'o' (10 teams x 15), 20 'l'
league details : transaction_mode "free-agency", trades "a"
game           : waivers_processed true, current_event 1, next_event 2
```

## Environment Variables

Required in `.env` (or set/locked via the in-app **🆔 League Setup** page under FPL App Home,
which validates IDs against the live FPL APIs and takes priority over these env vars —
see `scripts/common/league_config.py` and `scripts/fpl/league_setup.py`):
- `FPL_DRAFT_LEAGUE_ID` - Your draft league ID (from URL)
- `FPL_DRAFT_TEAM_ID` - Your team ID (from URL)

Optional (Notifications):
- `DISCORD_WEBHOOK_URL` - For deadline notifications
- `DISCORD_MENTION_USER_ID` - Discord user ID to mention in alerts
- `DISCORD_MENTION_ROLE_ID` - Discord role ID to mention in alerts
- `FPL_DRAFT_ALERTS_ENABLED` - Enable Draft waiver alerts (default: false)
- `FPL_DEADLINE_OFFSET_HOURS` - Hours before kickoff for Draft deadline (default: 25.5)
- `FPL_CLASSIC_ALERTS_ENABLED` - Enable Classic transfer alerts (default: false)
- `FPL_CLASSIC_DEADLINE_OFFSET_HOURS` - Hours before kickoff for Classic deadline (default: 1.5)

Optional (Classic):
- `FPL_CLASSIC_LEAGUE_IDS` - Comma-separated list of `league_id:League Name` pairs (e.g., `123456:My League,789012:Friends`)
- `FPL_CLASSIC_TEAM_ID` - Your Classic FPL team ID

Optional (External APIs):
- `ODDS_API_KEY` - The Odds API key for match betting odds (free tier: 500 requests/month)

Optional (Development):
- `FPL_CURRENT_GAMEWEEK` - Override for offline development
- `ROTOWIRE_URL` - Pin specific Rotowire article URL

## Adding New Features or Fixing Bugs

### Git Workflow (CRITICAL)

**NEVER commit directly to `main`.** This is a strict requirement. All work must follow this branching workflow:

1. **Create a feature branch FIRST** - Before writing any code, create a branch from `main`
   - Use naming convention: `feature/description` or `fix/description`
   - Examples: `feature/h2h-history`, `fix/player-matching`
2. **Do ALL work on the feature branch** - All commits, testing, and iterations happen here
3. **Test thoroughly on the feature branch** before merging
4. **ASK USER TO TEST before merging** - Before any merge to `main`, prompt the user to run `streamlit run main.py` and verify the changes work correctly. Wait for user confirmation before proceeding with the merge.
5. **Merge to `main` only when complete** - Feature must be tested and working, AND user has confirmed

```bash
# CORRECT workflow - always start with a branch
git checkout main
git pull origin main                     # Get latest changes
git checkout -b feature/my-feature       # Create feature branch BEFORE any work
# ... do work, commit changes ...
# ASK USER: "Please test with `streamlit run main.py` and confirm the changes work"
# ... wait for user confirmation ...
git checkout main && git merge feature/my-feature   # Merge when complete
git push origin main
```

```bash
# WRONG - never do this
git checkout main
# ... make changes and commit directly to main ...  # DON'T DO THIS
```

Note: The `dev` branch exists but is optional for integration testing when working on multiple features simultaneously.

## Roadmap

### High Priority

| Task | Status | Notes |
|------|--------|-------|
| Multi-GW Transfer Planner | Completed (polish available) | FFP Next3GWs blended into ROS scoring (40% weight) and displayed on waiver/transfer suggestion cards. Gaps: only Next3GWs used (Next2/4–6 fetched but ignored); Classic Transfers lacks sanity-check gate that Draft has. |
| Set Piece Takers Dashboard | Completed | New tab on Player Statistics page. Surface FPL bootstrap set piece data (penalties_order, direct_freekicks_order, corners_and_indirect_freekicks_order) grouped by team with penalty stats context. |
| Gameweek Review/Recap | Completed | New tab on Home page covering both Draft and Classic. Post-GW summary: top/bottom performers, bench points missed, captain vs best-captain analysis, rank movement, optimal lineup what-if. Leverage existing bench_analysis.py and live stats. |

### Medium Priority

| Task | Status | Notes |
|------|--------|-------|
| Fixture Projections Enhancements | Completed | Key differentials section, captain comparison (Classic), H2H layout fix (Classic now matches Draft order). Blended projections (Rotowire 60% + FFP 40% × start likelihood) added to Draft and Classic fixture pages; blend weight unified app-wide at 60/40. |

### Low Priority

| Task | Status | Notes |
|------|--------|-------|
| Transfer Risk Tracking | Phases 1-3 complete | Outbound (Google News RSS + bootstrap ground truth + per-region windows), inbound (per-club feeds, arrivals watchlist, minutes competition, fee attribution, `Transfer_Status`) and bookmaker odds are wired into the Draft Helper board and the Availability page. Remaining: Initial Squad `ExpPts` discount, `compute_player_scores()` ROS discount, roster-only Discord alerts. See "Transfer Risk Model" and "Transfer Odds Model". |
| Mini-League Rival Tracker | Not Started | Tab on League Analysis pages. Show differential players, projected points gap, effective ownership within mini-league. Data available via get_league_player_ownership (Draft) and team picks (Classic). No transfer advice (handled elsewhere). |
| Player Trade Analyzer | Completed | Trade Value model (season pts, regression, form, FDR, minutes), positional needs analysis, 1-for-1/2-for-2 trade discovery (position-matched — see "Draft Transaction Rules"; cross-position and 2-for-1 shapes were removed as FPL forbids them), acceptance likelihood scoring, Explore Teams comparison, Regression Watch (buy-low/sell-high) |
| Historical Data Analysis | Completed | Season History section on Classic Team Analysis (rank chart, points chart, data table); League Standing metrics on Draft Team Analysis |
| Split utils.py | Completed | Split into 7 focused modules (`text_helpers`, `fpl_draft_api`, `fpl_classic_api`, `scraping`, `fixture_helpers`, `analytics`, `optimization`); merged matching functions into `player_matching.py`; `utils.py` is now a thin re-export shim |

### Completed

| Task | Notes |
|------|-------|
| Draft Team Power Rankings | New "💪 Power Rankings" tab on Draft League Analysis. Team Score 0-100 plus GK/DEF/MID/FWD scores with league rank badges, all positional percentiles against the full ~700-player FPL pool. Player strength = 60% quality (preseason Rotowire pedigree blended into in-season actuals via `season_progress_weight()`) + 25% this-GW blended projection + 10% start security + 5% FDR, times a season-aware injury discount. Actuals use **points per start** (not total points) so a player returning from a long injury isn't penalised for missed games; below 4 starts the sample is too thin and pedigree is used alone. Rosters join to stats on integer element ID via new `get_league_rosters_with_ids()` — no name matching in this path. See "Team Strength Model" section below. |
| League Setup Admin Page | In-app "🆔 League Setup" page (FPL App Home) to set/validate/lock Draft and Classic league & team IDs instead of hand-editing `.env`. Draft: enter league ID, look it up, pick your team from a resolved dropdown (no need to know your entry ID). Classic: add/remove multiple leagues, resolve team via entry ID lookup. Locked read-only view with two-step "Unlock to Edit" confirmation. Persisted to gitignored `league_settings.json` (repo is public — not committed, unlike `alert_settings.json`). `config.py`'s 4 ID attributes (`FPL_DRAFT_LEAGUE_ID`, `FPL_DRAFT_TEAM_ID`, `FPL_CLASSIC_TEAM_ID`, `FPL_CLASSIC_LEAGUE_IDS`) converted from eager to lazy PEP 562 resolution (same pattern as `CURRENT_GAMEWEEK`), prioritizing locked JSON settings over `.env`, so saved changes take effect immediately without an app restart. |
| Gameweek Fixtures GW38 Cap | Auto-constrain FDR Horizon and "How many weeks?" sliders to never exceed GW38; defensive cap added inside `get_fixture_difficulty_grid()`; fetch range hard-capped at `end_gw = min(start_gw + weeks - 1, 38)` |
| Team Difficulty Visualizations | FDR heatmap, defensive stats, attack vs defense ratings (inspired by fpl.page/team-dds) |
| Projections Hub | Unified projections page with 5 data source tabs: Rotowire expert rankings, FFP Points Predictor (start %, multi-GW forecasts), Goal/Assist Odds, Clean Sheet Odds, Match Odds (The Odds API). Each tab has data source attribution, filters, and usage tips. |
| Live Score Integration & Gameweek Refresh | TTL-based gameweek caching (5 min) with manual refresh; live points from FPL API blended with Rotowire projections; actual starting 11 from Draft picks API (not optimal projections); styled overview table showing live/blended/original scores; player cards with played/upcoming status; win probability updates in real-time |
| Enhanced Lineup Visualizations | Fixed duplicate team bug (matchup index tracking); start likelihood indicator (opacity + border color based on injury status, FPL chance_of_playing, historical starts); robust player name matching (abbreviated names, nicknames, Nordic characters); team name mapping (Rotowire → FPL); Squad Details cards with form, points, goals/assists |
| Rotowire scraping robustness | Fallback table selectors (exact → partial → any); row validation before indexing; multiple regex patterns for URL discovery; proper logging throughout; replaced bare except clauses |
| Performance optimizations | Added `@st.cache_data` to 9 uncached API functions; startup preload with `@st.cache_resource`; refactored Draft home to eliminate 4 redundant `/league/details` calls; 50-60% faster page loads after initial startup |
| Season Highlights for Team Analysis | Best XI (optimal formation from top scorers), Team MVP (with starts/goals/assists/captain stats), Best Clubs (top 3 contributing EPL clubs); shared `team_analysis_helpers.py` module for Draft and Classic |
| Advanced Player Statistics Table | 40+ columns with 8 presets (Essential, Attacking, Defensive, Per 90, ICT Focus, Fixture Focus, GK Stats, Regression); green-white-red color gradients; regression metrics (G-xG, A-xA, GI-xGI) to identify over/under performers; switched to Classic FPL API for price/ownership data |
| Waiver Wire Transfer Suggestions | Top-3 position-locked swap suggestions with unified Player Value scoring, injury-aware hold logic, raised score-gap thresholds (elite/above-avg/weak tiers), multi-signal sanity check (vetoes ADD clearly worse on proj/season/3GW), 4-level FFP name fallback, and inline GW+3GW+season stats on each suggestion card |
| Error logging & better error messages | Added `error_helpers.py` module with structured logging and user-facing error display; added `timeout=30` to ~12 unprotected `requests.get()` calls; added `_logger.warning()` to ~15 silent `except` blocks; replaced ~13 generic error messages with actionable hints |
| Luck-Adjusted Standings (All-Play Record) | Replaced simplistic average-based model with industry-standard All-Play Record (every team vs every other each GW); fixed 0-score filter bug; shared `luck_analysis.py` module for Draft and Classic H2H; color-styled standings tables with auto-sized height; added toggle to Classic H2H standings |
| Data Source Update Alerts | Discord notifications when Rotowire/FFP publish new GW data; unified Alert Settings page in FPL App Home with configurable alert windows, test buttons, and live data source status checks; JSON config (`alert_settings.json`) with GitHub Actions commit-back for state persistence |
| Improve H2H Visuals | Better styling for H2H history sections with match history cards, icons, etc. |
| Add basic tests | pytest framework with 136 tests: unit tests for pure functions (player matching, luck analysis, alert config, team analysis helpers, utils), integration tests for API wrappers (mocked HTTP), and smoke tests for all 19 Streamlit pages |
| Season History for Team Analysis | Classic: Season History section with tabbed Plotly charts (Overall Rank, Total Points by season) and formatted data table; Draft: League Standing section with position, record, points for/against, and league points metrics |
| Styled Tables UI Refresh | Dark-themed HTML tables via shared `styled_tables.py` replacing ~35 `st.dataframe()` calls; dark gradient cards replacing `st.metric()` across all pages; Plotly charts with consistent `_DARK_CHART_LAYOUT` (dark bg, white text, green accents); FDR heatmap with distinct 5-level palette and continuous Avg FDR color interpolation; Match Odds proportional bars; side-by-side Points by Position charts; muted Injury Watchlist; sort-by-column on Advanced Stats and Rotowire Projections |
| Gameweek Review/Recap | Cross-format GW review page under FPL App Home. GW selector (defaults to last completed), top 10 scorers and notable blankers, Classic review (summary cards, squad table with captain, captain vs best-captain analysis, optimal lineup with best captaincy), Draft review (squad table, optimal lineup). Reuses `find_optimal_gw_lineup()` from `bench_analysis.py`. |
| Keep/Transfer Score Redesign | Replaced min-max normalization + user weight sliders with shared `compute_player_scores()` using positional percentiles against full FPL pool (~700 players). 1GW: fixed weights (0.55 proj + 0.25 form + 0.20 season). ROS: GW-dynamic weights shifting toward season quality. Scores directly interpretable (0.85 = top 15% at position). Removed `POSITIONAL_SCARCITY`, `ros_rebalanced_weights`, and 4-5 weight sliders from both Draft and Classic UI. |
| Transfer Scoring Model Redesign v2 | Pure EV-based 1GW (blended projections × start likelihood), multi-GW dominant ROS (40% MGW, 30% season quality, 15% form, 10% start consistency, 5% FDR), dynamic alpha blend for Transfer/Keep Score (adapts to format, position, player quality tier, squad depth). Removed TILT mechanism. Added `merge_ffp_single_gw_data()`, `compute_dynamic_alpha()`. 4-column output: 1GW, ROS, Transfer Score, Keep Score. See "Transfer Scoring Model — Design Reference" section above for full specification. |
