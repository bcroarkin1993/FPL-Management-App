# Configuration settings for the Streamlit app
from utils import get_current_gameweek, get_rotowire_rankings_url

# Rotowire URL for player rankings
ROTOWIRE_URL = get_rotowire_rankings_url()

# Rotowire URL for projected lineups
ROTOWIRE_LINEUPS_URL = 'https://www.rotowire.com/soccer/lineups.php'

# Team colors dictionary
TEAM_COLORS = {
    "AFC Bournemouth": {"primary": "#B50E12", "secondary": "#000000"},  # Dark Red / Black,
    "Arsenal": {"primary": "#EF0107", "secondary": "#9C824A"},  # Red / Gold,
    "Aston Villa": {"primary": "#670E36", "secondary": "#95BFE5"},  # Claret / Light Blue,
    "Brentford": {"primary": "#0057B8", "secondary": "#FFFFFF"},  # Blue / White
    "Brighton & Hove Albion": {"primary": "#0057B8", "secondary": "#FFFFFF"},  # Blue / White
    "Chelsea": {"primary": "#034694", "secondary": "#FFFFFF"},  # Royal Blue / White
    "Crystal Palace": {"primary": "#1B458F", "secondary": "#A7A5A6"},  # Blue / Gray
    "Everton": {"primary": "#003399", "secondary": "#FFFFFF"},  # Blue / White
    "Fulham": {"primary": "#FFFFFF", "secondary": "#000000"},  # White / Black
    "Ipswich Town": {"primary": "#0E00F7", "secondary": "#FFFFFF"},  # Blue / White
    "Leicester City": {"primary": "#003090", "secondary": "#FDBE11"},  # Blue / Gold,
    "Liverpool": {"primary": "#C8102E", "secondary": "#00B2A9"},  # Red / Green
    "Manchester City": {"primary": "#6CABDD", "secondary": "#1C2C5B"},  # Sky Blue / Blue
    "Manchester United": {"primary": "#DA291C", "secondary": "#FBE122"},  # Red / Yellow
    "Newcastle United": {"primary": "#241F20", "secondary": "#FFFFFF"},  # Black / White
    "Nottingham Forest": {"primary": "#DA291C", "secondary": "#FBE122"},  # Red / White
    "Southampton": {"primary": "#D71920", "secondary": "#130C0E"},  # Red / Black
    "Tottenham Hotspur": {"primary": "#132257", "secondary": "#FFFFFF"},  # Navy Blue / White
    "West Ham United": {"primary": "#7A263A", "secondary": "#1BB1E7"},  # Maroon / Light Blue
    "Wolverhampton Wanderers": {"primary": "#FDB913", "secondary": "#231F20"},  # Yellow / Black
}

# Any other global settings or variables can be added here
BRANDON_DRAFT_LEAGUE_ID = 49249
BRANDON_DRAFT_TEAM_ID = 189880
BRANDON_CLASSICAL_TEAM_ID = 6005298

# Set the API endpoints
team_gw_endpoint = f'https://fantasy.premierleague.com/api/entry/{BRANDON_CLASSICAL_TEAM_ID}/event/{CURRENT_GAMEWEEK}/picks/'
team_history_endpoint = f'https://fantasy.premierleague.com/api/entry/{BRANDON_CLASSICAL_TEAM_ID}/history/'
game_status_endpoint = f'https://draft.premierleague.com/api/game'

# Global variables to hold fetched data
CURRENT_GAMEWEEK = get_current_gameweek()
PLAYER_DATA = None  # This will store the player data from bootstrap-static API
LEAGUE_DATA = None  # This will store the league details
TRANSACTION_DATA = None  # This will store the transaction data
