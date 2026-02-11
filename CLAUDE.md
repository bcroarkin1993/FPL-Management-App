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

# Run the app
streamlit run main.py

# Run Discord waiver alerts (used by GitHub Actions)
python -m scripts.common.waiver_alerts
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

### Low Priority

| Task | Status | Notes |
|------|--------|-------|
| Player Trade Analyzer | Not started | Evaluate potential trades between teams (Draft mode) |
| Historical Data Analysis | Not started | Past season trends and performance analysis |
| Split utils.py | In progress | Created `player_matching.py`; more modules could be extracted |
| Add basic tests | Not started | No test infrastructure currently |

### Completed

| Task | Notes |
|------|-------|
| Team Difficulty Visualizations | FDR heatmap, defensive stats, attack vs defense ratings (inspired by fpl.page/team-dds) |
| Projections Hub | Unified projections page with 5 data source tabs: Rotowire expert rankings, FFP Points Predictor (start %, multi-GW forecasts), Goal/Assist Odds, Clean Sheet Odds, Match Odds (The Odds API). Each tab has data source attribution, filters, and usage tips. |
| Live Score Integration & Gameweek Refresh | TTL-based gameweek caching (5 min) with manual refresh; live points from FPL API blended with Rotowire projections; actual starting 11 from Draft picks API (not optimal projections); styled overview table showing live/blended/original scores; player cards with played/upcoming status; win probability updates in real-time |
| Enhanced Lineup Visualizations | Fixed duplicate team bug (matchup index tracking); start likelihood indicator (opacity + border color based on injury status, FPL chance_of_playing, historical starts); robust player name matching (abbreviated names, nicknames, Nordic characters); team name mapping (Rotowire → FPL); Squad Details cards with form, points, goals/assists |
| Rotowire scraping robustness | Fallback table selectors (exact → partial → any); row validation before indexing; multiple regex patterns for URL discovery; proper logging throughout; replaced bare except clauses |
| Performance optimizations | Added `@st.cache_data` to 9 uncached API functions; startup preload with `@st.cache_resource`; refactored Draft home to eliminate 4 redundant `/league/details` calls; 50-60% faster page loads after initial startup |
| Season Highlights for Team Analysis | Best XI (optimal formation from top scorers), Team MVP (with starts/goals/assists/captain stats), Best Clubs (top 3 contributing EPL clubs); shared `team_analysis_helpers.py` module for Draft and Classic |
| Advanced Player Statistics Table | 40+ columns with 8 presets (Essential, Attacking, Defensive, Per 90, ICT Focus, Fixture Focus, GK Stats, Regression); green-white-red color gradients; regression metrics (G-xG, A-xA, GI-xGI) to identify over/under performers; switched to Classic FPL API for price/ownership data |
| Waiver Wire Transfer Suggestions | Top-3 position-locked swap suggestions with unified Player Value scoring, injury-aware hold logic, asymmetric add/drop weights, and styled suggestion cards |
| Error logging & better error messages | Added `error_helpers.py` module with structured logging and user-facing error display; added `timeout=30` to ~12 unprotected `requests.get()` calls; added `_logger.warning()` to ~15 silent `except` blocks; replaced ~13 generic error messages with actionable hints |
| Luck-Adjusted Standings (All-Play Record) | Replaced simplistic average-based model with industry-standard All-Play Record (every team vs every other each GW); fixed 0-score filter bug; shared `luck_analysis.py` module for Draft and Classic H2H; color-styled standings tables with auto-sized height; added toggle to Classic H2H standings |
| Data Source Update Alerts | Discord notifications when Rotowire/FFP publish new GW data; unified Alert Settings page in FPL App Home with configurable alert windows, test buttons, and live data source status checks; JSON config (`alert_settings.json`) with GitHub Actions commit-back for state persistence |
| Improve H2H Visuals | Better styling for H2H history sections with match history cards, icons, etc. |
