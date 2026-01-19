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
python -m scripts.waiver_alerts
```

## Architecture

### Data Flow
```
FPL Draft API  ──┐
                 ├──> scripts/utils.py (normalize/merge) ──> Page scripts ──> Streamlit UI
Rotowire scrape ─┘
```

### Key Components

**config.py** - Configuration with lazy loading via PEP 562 `__getattr__`. `CURRENT_GAMEWEEK` and `ROTOWIRE_URL` resolve on first access to avoid import-time network calls.

**scripts/utils.py** (~1,800 lines) - Central utility module containing:
- FPL API fetching (`pull_fpl_player_stats()`, `get_league_entries()`, `get_transaction_data()`)
- Player name normalization and fuzzy matching (`_backfill_player_ids()`, `merge_fpl_players_and_projections()`)
- Rotowire scraping (`get_rotowire_player_projections()`, `get_rotowire_season_rankings()`)
- Fixture analysis (`get_fixture_difficulty_grid()`, `get_earliest_kickoff_et()`)

**main.py** - Streamlit entry point with three-section navigation:
- FPL App Home: Cross-format tools (fixtures, lineups, stats, injuries)
- Draft: League-specific analysis, waiver wire, team projections
- Classic: Placeholder structure for future FPL Classic support

**Page scripts** (scripts/*.py) - Each implements a `show_*_page()` function:
- `home.py` - Draft league standings and trends
- `waiver_wire.py` - Available player rankings
- `fixture_projections.py` - Draft matchup analysis
- `player_statistics.py` - EPL player stats

### External Data Sources

| Source | Usage |
|--------|-------|
| `draft.premierleague.com/api/` | League data, rosters, transactions |
| `fantasy.premierleague.com/api/` | Player stats, fixtures, FDR |
| `rotowire.com/soccer/` | Player projections, EPL lineups |

### Player Matching

Players are matched across sources using fuzzywuzzy with an 85% threshold (60% if team+position match). Key functions: `_norm_text()`, `_clean_player_name()`, `merge_fpl_players_and_projections()`.

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
| FPL Classic Compatibility | Not started | Add Classic league support (navigation structure exists) |
| Waiver Wire Improvements | Needs refinement | Basic system works, needs tuning |
| ~~Fix README entry point~~ | Done | ~~README says `app.py`, actual entry is `main.py`~~ |

### Medium Priority

| Task | Status | Notes |
|------|--------|-------|
| Matchup Insights - H2H History | Not started | Fixture projections done, head-to-head history pending |
| Improve player matching accuracy | Not started | 85% fuzzy threshold may miss players; tune or add manual mappings |
| Rotowire scraping robustness | Not started | URL discovery depends on HTML structure; add fallbacks |
| Add error logging | Not started | Many silent try/except blocks; harder to debug |
| Better error messages | Not started | Surface clearer feedback when APIs fail |
| Notifications improvements | Partial | Discord waiver alerts + injuries page exist; could improve |

### Low Priority

| Task | Status | Notes |
|------|--------|-------|
| Player Trade Analyzer | Not started | Evaluate potential trades between teams |
| Live Score Integration | Not started | Real-time score tracking during gameweeks |
| Enhanced Lineup Visualizations | Not started | Add player form, injury status to lineup views |
| Historical Data Analysis | Not started | Past season trends and performance analysis |
| Split utils.py | Not started | ~1,800 lines; could separate into api.py, matching.py, etc. |
| Add basic tests | Not started | No test infrastructure currently |
| Gameweek refresh logic | Not started | Cached at module level, doesn't auto-update during day |
