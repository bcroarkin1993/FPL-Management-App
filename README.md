# Fantasy Premier League (FPL) Draft Team Management App

This repository contains a **Fantasy Premier League (FPL) Draft Team Management App** built with **Streamlit**. The app provides a variety of tools to help manage and optimize your FPL Draft team.

The app integrates data from official FPL APIs and Rotowire projected lineups and player projections to provide actionable insights for FPL managers.

---

## Features

1. **League Standings**
   - View the current standings of your league, including matches won, drawn, and lost, as well as total points scored.
   - Toggle to an advanced view to see luck-adjusted statistics in the league table, offering insights into which teams have been fortunate or unfortunate based on weekly performances.
   - Analyze team performance trends with interactive graphs:
     - **Total Points Over Gameweeks**: Visualize the total points scored by each team across all gameweeks.
     - **League Points Over Gameweeks**: Track how each team accumulates league points (3 for a win, 1 for a draw) throughout the season.

2. **Fixture Analysis**  
   - View upcoming fixtures for your team and league.
   - Analyze team compositions and view match projections for all league matchups.

3. **Waiver Wire Suggestions**  
   - Identify the best available players in your league to improve your team.

4. **Projected EPL Lineups**  
   - Automatically fetch projected EPL lineups from Rotowire for all upcoming Premier League fixtures.
   - Visualize the lineups with a soccer pitch representation, clearly showing players by their positions (e.g., GK, DEF, MID, FWD).
   - Includes player names and positions, with special adjustments for formations (e.g., multiple central players) for an accurate display.

5. **Team Composition Tracking**  
   - Track the composition of all teams in your FPL draft league.
   - Compare your team's strengths and weaknesses against opponents.
   
---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/FPL-Draft-App.git
cd FPL-Draft-App
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 3: Configure Your `config.py` File

The app uses sensitive information, such as your FPL league ID and team ID, which must be set up in a `.env` file. Follow the steps below:

1. In the root directory of the project, create a file named `.env` (or rename the included `.env.example` file to `.env`).
2. Open the `.env` file and update the values with your information:
   - `FPL_LEAGUE_ID`: The league ID for your draft league. You can find this in the URL when viewing your league on the FPL website (e.g., `https://draft.premierleague.com/league/{FPL_LEAGUE_ID}`).
   - `FPL_TEAM_ID`: Your team ID, found in the URL when viewing your team (e.g., `https://draft.premierleague.com/team/{FPL_TEAM_ID}`).
   - `GAMEWEEK`: Optionally set the current gameweek if needed.

Example `.env` file:
```bash
FPL_DRAFT_LEAGUE_ID = 123456
FPL_DRAFT_TEAM_ID = 123
FPL_CLASSICAL_TEAM_ID = 654321
```

3. Save the .env file. The app will automatically load these variables during runtime.
Ensure `.env` is listed in the `.gitignore` file to avoid accidentally pushing sensitive information to GitHub.

### Step 4: Run the App

Launch the Streamlit app:
```bash
streamlit run app.py
```
