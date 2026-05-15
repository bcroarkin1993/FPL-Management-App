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
pytest                        # All tests
pytest tests/common/          # Unit tests only
pytest tests/draft/ tests/classic/ tests/fpl/  # Smoke tests only
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
| `rotowire.com/soccer/` | Player projections, EPL lineups |
| `fantasyfootballpundit.com` | Points predictions, goal/assist odds, clean sheet odds (via Google Sheets) |
| `api.the-odds-api.com` | Match betting odds (h2h, BTTS, totals) |

### Player Matching

Players are matched across sources using a two-step approach:
1. **Canonical normalization** via `canonical_normalize()` strips accents and normalizes names (e.g., "Raúl Jiménez Rodríguez" → "raul jimenez rodriguez")
2. **Team-prioritized fuzzy matching** tries same-team matches first, then falls back to cross-team matching

Key modules:
- `scripts/common/player_matching.py` - `canonical_normalize()`, `PlayerRegistry` class for centralized lookups
- `scripts/common/utils.py` - `merge_fpl_players_and_projections()` with 80% threshold (60% if team+position match)

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

## Environment Variables

Required in `.env`:
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
