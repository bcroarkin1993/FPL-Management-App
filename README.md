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

### Classic League Tools (Coming Soon)

11. **Free Hit Optimizer**
    - Linear programming-based squad optimizer for Free Hit chip
    - Respects budget, squad size, and team limits

12. **Transfer Recommendations**
    - Suggested transfers based on projected point gains

### Notifications

13. **Discord Waiver Alerts**
    - Automated reminders for waiver wire deadlines via Discord webhook
    - Runs on schedule via GitHub Actions

---

## Project Structure

```
scripts/
├── common/          # Shared utilities
│   ├── utils.py     # API calls, data transforms, player matching
│   └── waiver_alerts.py
├── draft/           # Draft league features
│   ├── home.py, waiver_wire.py, fixture_projections.py
│   ├── team_analysis.py, draft_helper.py
├── fpl/             # Cross-format features
│   ├── fixtures.py, injuries.py, player_projections.py
│   ├── player_statistics.py, projected_lineups.py
└── classic/         # Classic league features
    ├── free_hit.py, transfers.py
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
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...  # For waiver alerts
FPL_DEADLINE_OFFSET_HOURS=25.5  # Hours before kickoff for deadline reminders
TZ_NAME=America/New_York  # Your timezone
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
- **Classic**: FPL Classic league support (in development)

---

## Roadmap

### High Priority
- **FPL Classic Compatibility** - Complete Classic league support
- **Waiver Wire Improvements** - Refine recommendation accuracy

### Medium Priority
- **Head-to-Head History** - Add historical matchup data to fixture projections
- **Improved Player Matching** - Better fuzzy matching between data sources
- **Error Handling** - Better logging and user-facing error messages

### Low Priority
- Player Trade Analyzer
- Live Score Integration
- Enhanced Lineup Visualizations
- Historical Data Analysis

---

## Contributing

Feedback and suggestions are welcome! Feel free to open an issue on the GitHub repository.
