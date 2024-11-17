# FPL-Management-App
=======
# Fantasy Premier League (FPL) Draft Team Management App

This repository contains a **Fantasy Premier League (FPL) Draft Team Management App** built with **Streamlit**. The app provides a variety of tools to help manage and optimize your FPL Draft team, including:

- **Fixture analysis** to evaluate matchups and projections.
- **Waiver wire suggestions** to find the best available players.
- **Projected lineups** for all Premier League teams.
- **Team composition tracking** for your league.
- **Visualizations** of player positions on the soccer field.

The app integrates data from official FPL APIs and Rotowire projections to provide actionable insights for FPL managers.

---

## Features

1. **Fixture Analysis**  
   - View upcoming fixtures for your team and league.
   - Analyze team compositions and optimize starting lineups.

2. **Waiver Wire Suggestions**  
   - Identify the best available players in your league using fuzzy matching and projection data.

3. **Projected Lineups**  
   - View projected lineups for all Premier League teams, pulled directly from Rotowire.

4. **Team Composition Tracking**  
   - Track the composition of all teams in your FPL draft league.
   - Compare your team's strengths and weaknesses against opponents.

5. **Interactive Visualizations**  
   - Visualize player positions on a soccer field for both your team and projected lineups.

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

1. Open the `config.py` file in the root of the project.
2. Update the `FPL_TEAM_ID` variable with your personal FPL team ID retrieved in Step 2. The `FPL_TEAM_ID` should be an integer.

Example configuration in `config.py`:
```python
FPL_TEAM_ID = 123456  # Replace 123456 with your actual FPL team ID
```

3. Save the file after making your changes.
4. You're now ready to run the app with your personalized configuration!

### Step 4: Run the App

Launch the Streamlit app:
```bash
streamlit run app.py
```

>>>>>>> 9c86a7e (Cleaning repo)
