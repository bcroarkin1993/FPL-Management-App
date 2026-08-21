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
- `scripts/fpl/` - fixtures.py, player_statistics.py, projected_lineups.py, injuries.py

### External Data Sources

| Source | Usage |
|--------|-------|
| `draft.premierleague.com/api/` | League data, rosters, transactions |
| `fantasy.premierleague.com/api/` | Player stats, fixtures, FDR |
| `rotowire.com/soccer/` | Player projections, EPL lineups, article publish times |
| `fantasyfootballpundit.com` | Points predictions, goal/assist odds, clean sheet odds (via Google Sheets) |
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
this also covers reversed names like "Mitoma Kaoru", (5) difflib ≥ 0.78.

Two rules exist because breaking them caused real bugs:
- **Every tier below the first two is scoped to Team *and* Position.** There is
  no team-agnostic tier — a surname-only match once gave Alex Palmer (backup GK)
  Cole Palmer's (elite MID) stats and ranked him top 10.
- **The fuzzy tier requires a shared complete token.** Character similarity
  alone matched "Harrison" to team-mate "Harry Wilson" at 0.79.

Ambiguity (>1 candidate at a tier) resolves to *no match*. `match_with_tier()`
returns the tier rank so callers can resolve two players contending for one
reference row in favour of the stronger tier — without that, a fuzzy guess can
steal a row an exact match already earned.

### Source Freshness

`get_rotowire_article_updated()` (`scripts/common/scraping.py`) scrapes an
article's "Updated on ..." stamp from its `div.article__date`; render it with
`format_last_updated()` (`text_helpers.py`), which appends the age
("Aug 20, 2026 10:54 AM ET (3h ago)").

Show this wherever projections are displayed. A weekly rankings table written
before the last team-news cycle is materially less reliable than one written
after it, and nothing else on the page distinguishes them. Currently surfaced in
the Initial Squad Optimizer's Data Sources panel and the Projections Hub source
banner. FFP is a live Google Sheet with no published revision time, so it shows
"—" rather than a fabricated one.

A missing timestamp is cosmetic: the scraper returns `None` on any failure and
must never take a page down.

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
| `check_merge_match_rate()` | A name-based merge quietly ceasing to match (the 356/425 season-rankings regression) |
| `check_initial_squad()` | Illegal or implausibly-priced Classic squads; a scale-free objective, whose signature is unspent budget |
| `check_team_strength()` | Degenerate power rankings — every team scoring ~50 because position codes were `GKP/DEF/MID/FWD` instead of `G/D/M/F`, short squads, impossible injury costs |

Ranges are deliberately wide — these are "this cannot be right" boundaries, not
"this looks unusual" ones. A check that cries wolf gets muted. Note the XI floor is
permissive on purpose: a squad with several players not expected to start
legitimately projects in the high teens, because **absence from Rotowire's weekly
table is itself the "not starting" signal** (it lists 20 teams x 11 = 220 projected
starters, so an unlisted player scoring 0 is correct, not a matching failure).

### Test layers

| Layer | Location | Runs | Catches |
|-------|----------|------|---------|
| Unit tests for the checks | `tests/common/test_data_validation.py` | Always, offline | A check silently ceasing to fire. Every "bad data" fixture is a real number the app actually displayed. |
| Live plausibility | `tests/live/` | Default run; skips offline | Upstream sources changing shape. Contract: **unreachable → SKIP, reachable but implausible → FAIL.** |
| Page wiring | `tests/draft/test_fixture_page_wiring.py` | Always, offline | Two callsites of the same function being fed different inputs (the Fixtures Overview omitting `ffp_df`). Plausibility can't catch this — both numbers are individually plausible. |

`tests/live/conftest.py` clears the `FPL_CURRENT_GAMEWEEK` / `ROTOWIRE_URL` pins that
the root conftest sets, so live tests resolve real config. Everything there is
auto-marked `live`.

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

**`_effective_proj` column**: `compute_player_scores()` retains `_effective_proj` (blended_proj × start_likelihood) in its output. Consumers (Waiver Wire suggestion engine, card rendering) rely on it for GW projection display and sanity checking. Do not drop it from the result.

**FFP name matching — 4-level fallback**: Both `merge_ffp_single_gw_data()` and `blend_multi_gw_projections()` use a 4-step lookup to handle name mismatches between FFP short names ("Eze") and FPL full names ("Eberechi Eze"), as well as FFP team name variants: (1) exact `(norm_name, team_short)`, (2) `(last_word, team_short)`, (3) `norm_name` only, (4) `last_word` only.


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

**Unranked players get a positional floor, not the median.** A player outside a
400-deep season ranking is not average; absence is itself the signal, the same
way absence from Rotowire's weekly table means "not starting". They take the
10th-percentile `SeasonPG` for their position. Letting `positional_percentile()`
hand its 0.5 default to 243 of 599 players made genuine fodder look mid-table.

Both Rotowire URLs (season and GW1) are pinned in `config.py` — neither slug is
auto-discoverable — and both are editable in the page's **Data Sources** panel,
which shows fetch status, row counts and **match rate** per source. That panel
exists because every fetch degrades to an empty frame on failure: without it, a
broken season URL renders a page identical to a working one, just with every
Season score silently at 0.50.

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
| Mini-League Rival Tracker | Not Started | Tab on League Analysis pages. Show differential players, projected points gap, effective ownership within mini-league. Data available via get_league_player_ownership (Draft) and team picks (Classic). No transfer advice (handled elsewhere). |
| Player Trade Analyzer | Completed | Trade Value model (season pts, regression, form, FDR, minutes), positional needs analysis, 1-for-1/2-for-2/2-for-1 trade discovery, acceptance likelihood scoring, Explore Teams comparison, Regression Watch (buy-low/sell-high) |
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
