# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Streamlit-based Fantasy Premier League (FPL) Draft management app that integrates FPL APIs with Rotowire projections to provide analytics for draft league managers.

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
FPL Draft API  ──┐
                 ├──> scripts/common/utils.py (normalize/merge) ──> Page scripts ──> Streamlit UI
Rotowire scrape ─┘
```

### Key Components

**config.py** - Configuration with lazy loading via PEP 562 `__getattr__`. `CURRENT_GAMEWEEK` and `ROTOWIRE_URL` resolve on first access to avoid import-time network calls.

**scripts/common/** - Shared utilities:
- `utils.py` - FPL API fetching, Rotowire scraping, player matching, fixture analysis
- `player_matching.py` - `canonical_normalize()`, `PlayerRegistry` for centralized player lookups
- `waiver_alerts.py` - Discord notification system for waiver deadlines

**main.py** - Streamlit entry point with three-section navigation:
- FPL App Home: Cross-format tools (fixtures, lineups, stats, injuries)
- Draft: League-specific analysis, waiver wire, team projections
- Classic: League standings and team analysis for Classic FPL leagues

**Page scripts** - Organized by section, each implements a `show_*_page()` function:
- `scripts/draft/` - home.py, waiver_wire.py, fixture_projections.py, team_analysis.py
- `scripts/classic/` - league_standings.py, team_analysis.py
- `scripts/fpl/` - fixtures.py, player_statistics.py, projected_lineups.py, injuries.py

### External Data Sources

| Source | Usage |
|--------|-------|
| `draft.premierleague.com/api/` | League data, rosters, transactions |
| `fantasy.premierleague.com/api/` | Player stats, fixtures, FDR |
| `rotowire.com/soccer/` | Player projections, EPL lineups |

### Player Matching

Players are matched across sources using a two-step approach:
1. **Canonical normalization** via `canonical_normalize()` strips accents and normalizes names (e.g., "Raúl Jiménez Rodríguez" → "raul jimenez rodriguez")
2. **Team-prioritized fuzzy matching** tries same-team matches first, then falls back to cross-team matching

Key modules:
- `scripts/common/player_matching.py` - `canonical_normalize()`, `PlayerRegistry` class for centralized lookups
- `scripts/common/utils.py` - `merge_fpl_players_and_projections()` with 80% threshold (60% if team+position match)

### Caching

Streamlit's `@st.cache_data` decorator is used for expensive API calls. Gameweek is cached at module level in config.py.

## Environment Variables

Required in `.env`:
- `FPL_DRAFT_LEAGUE_ID` - Your draft league ID (from URL)
- `FPL_DRAFT_TEAM_ID` - Your team ID (from URL)

Optional:
- `DISCORD_WEBHOOK_URL` - For waiver deadline notifications
- `FPL_DEADLINE_OFFSET_HOURS` - Hours before kickoff for deadline (default: 25.5)
- `FPL_CURRENT_GAMEWEEK` - Override for offline development
- `ROTOWIRE_URL` - Pin specific Rotowire article URL

## Adding New Features or Fixing Bugs

**IMPORTANT**: When you work on a new feature or bug, create a new git branch first and then work on
that branch for the remainder of the session.

## Roadmap

### High Priority

| Task | Status | Notes |
|------|--------|-------|
| FPL Classic Compatibility | Partial | League standings and team analysis pages done; transfers and free hit pages pending |

### Medium Priority

| Task | Status | Notes |
|------|--------|-------|
| Matchup Insights - H2H History | Not started | Fixture projections done, head-to-head history pending |
| Rotowire scraping robustness | Not started | URL discovery depends on HTML structure; add fallbacks |
| Add error logging | Not started | Many silent try/except blocks; harder to debug |
| Better error messages | Not started | Surface clearer feedback when APIs fail |

### Low Priority

| Task | Status | Notes |
|------|--------|-------|
| Player Trade Analyzer | Not started | Evaluate potential trades between teams |
| Live Score Integration | Not started | Real-time score tracking during gameweeks |
| Enhanced Lineup Visualizations | Not started | Add player form, injury status to lineup views |
| Historical Data Analysis | Not started | Past season trends and performance analysis |
| Split utils.py | In progress | Created `player_matching.py`; more modules could be extracted |
| Add basic tests | Not started | No test infrastructure currently |
| Gameweek refresh logic | Not started | Cached at module level, doesn't auto-update during day |
