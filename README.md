# Fantasy Premier League (FPL) Management App

A **Streamlit-based Fantasy Premier League management app** for both **Draft** and **Classic** formats. The app integrates data from official FPL APIs and Rotowire projections to provide actionable insights for FPL managers.

---

## Features

### Cross-Format Tools (FPL App Home)

1. **Gameweek Fixtures**
   - Browse upcoming Premier League fixtures with fixture difficulty ratings (FDR)
   - Color-coded difficulty grid showing each team's schedule

2. **Projected EPL Lineups**
   - Fetch projected lineups from Rotowire for all upcoming fixtures
   - Visualize lineups on a soccer pitch representation by position

3. **Player Projections**
   - View Rotowire's gameweek player projections and rankings
   - Compare projected points across positions

4. **EPL Player Statistics**
   - Track top performers by category (goals, assists, clean sheets, xG, xA)
   - Advanced data visualization graphs

5. **Player Injuries**
   - View current injury news and availability status across all EPL players

### Draft League Tools

6. **League Standings & Trends**
   - Current standings with wins, draws, losses, and total points
   - Luck-adjusted statistics showing which teams have been fortunate/unfortunate
   - Interactive graphs: Total Points and League Points over gameweeks

7. **Fixture Projections**
   - View upcoming draft league matchups with projected scores
   - Analyze team compositions for all league fixtures

8. **Waiver Wire**
   - Identify the best available players in your league
   - Player rankings with projected points, form, and fixture difficulty

9. **Team Analysis**
   - Track composition of all teams in your draft league
   - Compare strengths and weaknesses against opponents

10. **Draft Helper**
    - Tools to assist during the draft process

11. **League Analysis**
    - Head-to-head records matrix showing win/draw/loss against each opponent
    - Scoring distribution and consistency metrics
    - Strength of schedule analysis
    - League records (highest/lowest scores, biggest wins, closest matches)
    - Win/loss streak tracking

### Classic League Tools

12. **League Home & Standings**
    - League standings with support for multiple Classic leagues via dropdown
    - Points progression charts (Total Points, GW Points, Rank Progression)
    - Supports both Classic scoring and H2H league formats

13. **Fixture Projections**
    - View projected scores for all teams in your Classic league
    - H2H mode: Win probability analysis for head-to-head matchups
    - Classic mode: Projected leaderboard with standings movement indicators

14. **Transfer Suggestions**
    - Smart transfer recommendations ranked by projections, form, FDR, and price
    - Position-specific tabs with transfer-in and transfer-out candidates
    - Transfer comparison tool to evaluate swaps

15. **Free Hit Optimizer**
    - Linear programming-based squad optimizer for Free Hit chip
    - Formation selector, budget controls, and fixture difficulty display
    - Respects budget, squad size (15 players), and max 3 per team limits
    - Filters out injured/doubtful players automatically

16. **Team Analysis**
    - Detailed breakdown of your Classic FPL team performance
    - Season trends, chip usage, and transfer history

17. **League Analysis**
    - Chip usage analysis across the league (who played what chip when)
    - Rank movement tracking (biggest risers and fallers)
    - Points behind leader over time
    - Team value comparison
    - Gameweek scoring distribution and consistency metrics

### Notifications

18. **Discord Waiver Alerts**
    - Automated reminders for waiver wire deadlines via Discord webhook
    - Runs on schedule via GitHub Actions

---

## Project Structure

```
scripts/
├── common/          # Shared utilities
│   ├── utils.py     # API calls, data transforms, player matching
│   ├── player_matching.py  # Canonical normalization, PlayerRegistry
│   └── waiver_alerts.py
├── draft/           # Draft league features
│   ├── home.py, waiver_wire.py, fixture_projections.py
│   ├── team_analysis.py, draft_helper.py, league_analysis.py
├── fpl/             # Cross-format features
│   ├── fixtures.py, injuries.py, player_projections.py
│   ├── player_statistics.py, projected_lineups.py
└── classic/         # Classic league features
    ├── home.py              # League standings + progression charts
    ├── fixture_projections.py  # H2H matchups or projected leaderboard
    ├── transfers.py         # Transfer suggestions
    ├── free_hit.py          # Free Hit chip optimizer
    ├── league_analysis.py   # Chip usage, rank movement, team values
    └── team_analysis.py     # Team performance analysis
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/bcroarkin1993/FPL-Management-App.git
cd FPL-Management-App
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Your Environment

The app requires your FPL league and team IDs, which must be set up in a `.env` file:

1. Copy the example file: `cp .env.example .env`
2. Edit `.env` with your information:
   - `FPL_DRAFT_LEAGUE_ID`: Found in the URL at `https://draft.premierleague.com/league/{ID}`
   - `FPL_DRAFT_TEAM_ID`: Found in the URL at `https://draft.premierleague.com/entry/{ID}`

Example `.env` file:
```bash
FPL_DRAFT_LEAGUE_ID=123456
FPL_DRAFT_TEAM_ID=123
```

Optional settings:
```bash
# Draft notifications
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...  # For waiver alerts
FPL_DEADLINE_OFFSET_HOURS=25.5  # Hours before kickoff for deadline reminders
TZ_NAME=America/New_York  # Your timezone

# Classic FPL (optional)
FPL_CLASSIC_LEAGUE_IDS=123456:My League,789012:Friends League  # Multiple leagues supported
FPL_CLASSIC_TEAM_ID=456789  # Your Classic team ID
```

### 4. Run the App

```bash
streamlit run main.py
```

---

## App Navigation

The app has three main sections accessible from the sidebar:

- **FPL App Home**: Cross-format tools (fixtures, lineups, projections, stats, injuries)
- **Draft**: Draft league-specific features (standings, waiver wire, team analysis)
- **Classic**: Full Classic FPL support (standings, fixture projections, transfers, free hit optimizer)

---

## Roadmap

### Completed
- **FPL Classic Compatibility** - Full Classic league support with standings, fixture projections, transfers, and free hit optimizer

### Medium Priority
- **Team Difficulty Visualizations** - FDR heatmap, defensive stats, attack vs defense ratings
- **Head-to-Head History** - Add historical matchup data to fixture projections
- **Error Handling** - Better logging and user-facing error messages

### Low Priority
- Player Trade Analyzer (Draft mode)
- Live Score Integration
- Enhanced Lineup Visualizations
- Historical Data Analysis

---

## Contributing

Feedback and suggestions are welcome! Feel free to open an issue on the GitHub repository.
