"""
Constants and configuration data for NFL coaching analysis.

This module contains all the static dictionaries, lists, and mappings
used throughout the coaching data analysis pipeline.
"""

from typing import Dict, List, Union


# Team franchise abbreviation mappings for historical name changes and relocations
TEAM_FRANCHISE_MAPPINGS = {
    "ind": ["ind", "clt"], 
    "ari": ["ari", "crd"],
    "hou": ["hou", "htx", "oti"],
    "ten": ["ten", "oti"],
    "oak": ["oak", "rai"], 
    "stl": ["stl", "ram", "sla", "gun"],
    "lar": ["lar", "ram"],
    "ram": ["lar", "ram"],
    "bal": ["bal", "rav", "clt"],
    "lac": ["lac", "sdg"],
    "chr": ["chr", "cra"],
    "frn": ["frn", "fyj"],
    "nyt": ["nyt", "nyj"],
    "can": ["can", "cbd"],
    "bos": ["bos", "byk", "was", "ptb"],
    "pot": ["pot", "ptb"],
    "lad": ["lad", "lda"],
    "evn": ["evn", "ecg"],
    "pho": ["pho", "crd"],
    "prt": ["prt", "det"],
    "lvr": ["lvr", "rai"],
    "chh": ["chh", "cra"],
    "htx": ["htx", "oti"],
    "buf": ["buf", "bff", "bba"],
    "cle": ["cle", "cti", "cib", "cli"],
    "min": ["min", "mnn"],
    "kan": ["kan", "kcb"],
    "det": ["det", "dwl", "dpn", "dhr", "dti"],
    "nyy": ["nyy", "naa", "nya"],
    "cin": ["cin", "red", "ccl"],
    "mia": ["mia", "msa"],
    "was": ["was", "sen"],
    "dtx": ["kan", "dtx"],
    "nyg": ["nyg", "ng1"],
    "cli": ["cli", "cib"],
    "nyb": ["nyb", "nyy"]
}

# Current team name to abbreviation mapping for new hires
CURRENT_TEAM_ABBREVIATIONS = {
    "Chicago Bears": "chi",
    "Jacksonville Jaguars": "jax",
    "New Orleans Saints": "nor",
    "New York Jets": "nyj",
    "Dallas Cowboys": "dal",
    "New England Patriots": "nwe",
    "Las Vegas Raiders": "rai",
    "San Francisco 49ers": "sfo",
    "New York Giants": "nyg",
    "Los Angeles Chargers": "lac", 
    "Green Bay Packers": "gnb",
    "Los Angeles Rams": "ram"
}

# Spotrac to Pro Football Reference team abbreviation mappings
SPOTRAC_TO_PFR_MAPPINGS = {
    # AFC East
    "BUFBUF": "buf",  # Buffalo Bills
    "MIAMIA": "mia",  # Miami Dolphins  
    "NENE": "nwe",    # New England Patriots
    "NYJNYJ": "nyj",  # New York Jets
    
    # AFC North
    "BALBAL": "rav",  # Baltimore Ravens (PFR uses 'rav')
    "CINCIN": "cin",  # Cincinnati Bengals
    "CLECLE": "cle",  # Cleveland Browns
    "PITPIT": "pit",  # Pittsburgh Steelers
    
    # AFC South
    "HOUHOU": "htx",  # Houston Texans (PFR uses 'htx')
    "INDIND": "clt",  # Indianapolis Colts (PFR uses 'clt')
    "JAXJAX": "jax",  # Jacksonville Jaguars
    "TENTEN": "oti",  # Tennessee Titans (PFR uses 'oti')
    
    # AFC West
    "DENDEN": "den",  # Denver Broncos
    "KCKC": "kan",    # Kansas City Chiefs (PFR uses 'kan')
    "LACLAC": "sdg",  # Los Angeles Chargers (PFR uses 'sdg')
    "LVLV": "rai",    # Las Vegas Raiders (PFR uses 'rai')
    "OAKOAK": "rai",  # Oakland Raiders (historical, maps to 'rai')
    
    # NFC East
    "DALDAL": "dal",  # Dallas Cowboys
    "NYGNYG": "nyg",  # New York Giants
    "PHIPHI": "phi",  # Philadelphia Eagles
    "WASWAS": "was",  # Washington (PFR uses 'was')
    
    # NFC North
    "CHICHI": "chi",  # Chicago Bears
    "DETDET": "det",  # Detroit Lions
    "GBGB": "gnb",    # Green Bay Packers (PFR uses 'gnb')
    "MINMIN": "min",  # Minnesota Vikings
    
    # NFC South
    "ATLATL": "atl",  # Atlanta Falcons
    "CARCAR": "car",  # Carolina Panthers
    "NONO": "nor",    # New Orleans Saints (PFR uses 'nor')
    "TBTB": "tam",    # Tampa Bay Buccaneers (PFR uses 'tam')
    
    # NFC West
    "ARIARI": "crd",  # Arizona Cardinals (PFR uses 'crd')
    "LARLAR": "ram",  # Los Angeles Rams
    "SFSF": "sfo",    # San Francisco 49ers (PFR uses 'sfo')
    "SEASEA": "sea",  # Seattle Seahawks
    
    # Historical team names (before relocations)
    "STLSTL": "ram",  # St. Louis Rams (2011-2015) → Los Angeles Rams
    "SDSD": "sdg"     # San Diego Chargers (2011-2016) → Los Angeles Chargers
}

# Core coaching experience features
CORE_COACHING_FEATURES = [
    "age",
    "num_times_hc",
    "num_yr_col_pos",
    "num_yr_col_coor",
    "num_yr_col_hc",
    "num_yr_nfl_pos",
    "num_yr_nfl_coor",
    "num_yr_nfl_hc"
]

# Base team statistics that get role-specific suffixes
BASE_TEAM_STATISTICS = [
    "PF (Points For)",
    "Yds",
    "Y/P",
    "TO",
    "1stD",
    "Cmp Passing",
    "Att Passing",
    "Yds Passing",
    "TD Passing",
    "Int Passing",
    "NY/A Passing",
    "1stD Passing",
    "Att Rushing",
    "Yds Rushing",
    "TD Rushing",
    "Y/A Rushing",
    "1stD Rushing",
    "Pen",
    "Yds Penalties",
    "1stPy",
    "#Dr",
    "Sc%",
    "TO%",
    "Time Average Drive",
    "Plays Average Drive",
    "Yds Average Drive",
    "Pts Average Drive",
    "3DAtt",
    "3D%",
    "4DAtt",
    "4D%",
    "RZAtt",
    "RZPct"
]

# Role suffixes for team statistics
ROLE_SUFFIXES = {
    "offensive_coordinator": "__oc",
    "defensive_coordinator": "__dc", 
    "head_coach": "__hc",
    "head_coach_opponent": "__opp__hc"
}

# Hiring team context features
HIRING_TEAM_FEATURES = [
    "hiring_team_win_pct",
    "hiring_team_points_scored", 
    "hiring_team_points_allowed",
    "hiring_team_yards_offense",
    "hiring_team_yards_allowed",
    "hiring_team_yards_per_play",
    "hiring_team_yards_per_play_allowed",
    "hiring_team_turnovers_forced",
    "hiring_team_turnovers_committed",
    "hiring_team_num_playoff_appearances"
]

# Words/phrases that exclude a coaching role from consideration
EXCLUDED_ROLE_KEYWORDS = [
    "Consultant",
    "Scout", 
    "Analyst",
    "Athletic Director",
    "Advisor",
    "Intern",
    "Sports Science",
    "Quality Control",
    "Emeritus",
    "Freshman ",
    "/Freshman",
    "Passing Game Coordinator",
    "Pass Gm. Coord.",
    "Recruiting",
    "Reserve",
    "earnings",
    "Strength and Conditioning",
    "Strength & Conditioning", 
    "Video",
    "Senior Assistant",
    "Associate Head Coach"
]

# Coaching tenure classification thresholds
TENURE_CLASSIFICATIONS = {
    "short": (0, 2),      # 0-2 years
    "medium": (3, 4),     # 3-4 years  
    "long": (5, float('inf'))  # 5+ years
}

# Data file configurations
DATA_FILES = {
    "coach_tables": ["all_coaching_results", "all_coaching_ranks", "all_coaching_history"],
    "league_tables": ["league_team_data_normalized", "league_opponent_data_normalized"],
    "team_record": "team_record.csv",
    "team_playoff": "team_playoff_record.csv"
}

# Current analysis parameters
ANALYSIS_CONFIG = {
    "cutoff_year": 2022,
    "current_year": 2025,
    "expected_feature_count": 154,
    "hiring_context_years": [1, 2]  # Look back 1-2 years for team context
}

# Coaches who were fired/resigned and should get actual tenure classification
# (not marked as -1 for insufficient data)
FIRED_COACHES = [
    "Doug Pederson",
    "Frank Reich",
    "Antonio Pierce",
    "Jerod Mayo"
]

# Coach-year hiring instances to exclude from final dataset
# Format: (coach_name, hire_year) tuples
EXCLUDED_HIRING_INSTANCES = [
    ("Sean Payton", 2013)  # Interim/temporary hire that should not be included in analysis
]


def get_all_feature_names() -> List[str]:
    """Generate complete list of feature names in correct order matching Excel file"""
    
    # Core coaching experience features (Features 1-8)
    core_features = [
        "age",
        "num_times_hc", 
        "num_yr_col_pos",
        "num_yr_col_coor",
        "num_yr_col_hc",
        "num_yr_nfl_pos",
        "num_yr_nfl_coor", 
        "num_yr_nfl_hc"
    ]
    
    # NFL OC team statistics (Features 9-41)
    oc_features = []
    for stat in BASE_TEAM_STATISTICS:
        oc_features.append(f"{stat}__oc")
    
    # NFL DC opponent statistics (Features 42-74) 
    dc_features = []
    for stat in BASE_TEAM_STATISTICS:
        dc_features.append(f"{stat}__dc")
    
    # NFL HC team statistics (Features 75-107)
    hc_features = []
    for stat in BASE_TEAM_STATISTICS:
        hc_features.append(f"{stat}__hc")
    
    # NFL HC opponent statistics (Features 108-140)
    hc_opp_features = []
    for stat in BASE_TEAM_STATISTICS:
        hc_opp_features.append(f"{stat}__opp__hc")
    
    # Combine all in exact Excel order
    feature_names = core_features + oc_features + dc_features + hc_features + hc_opp_features
    
    return feature_names


def get_feature_dict() -> Dict[str, Union[int, List]]:
    """Create feature dictionary with appropriate default values"""
    feature_dict = {}
    
    # Core features start as integers
    for feature in CORE_COACHING_FEATURES:
        feature_dict[feature] = 0
    
    # Team statistics features start as lists
    for stat in BASE_TEAM_STATISTICS:
        for suffix in ROLE_SUFFIXES.values():
            feature_dict[f"{stat}{suffix}"] = []
    
    return feature_dict


def get_hiring_team_stat_dict() -> Dict[str, List]:
    """Create hiring team statistics dictionary"""
    return {feature: [] for feature in HIRING_TEAM_FEATURES}


def get_output_column_names() -> List[str]:
    """Generate output CSV column names"""
    columns = ['Coach Name', 'Year']
    
    # Add numbered feature columns
    total_features = len(get_all_feature_names()) + len(HIRING_TEAM_FEATURES)
    for i in range(1, total_features + 1):
        columns.append(f'Feature {i}')
    
    columns.extend(['Avg 2Y Win Pct', 'Coach Tenure Class'])
    return columns


# NFL Salary Cap Maximum Values by Year
# Use this dictionary to convert salary cap data to percentages of max cap
SALARY_CAP_MAX_BY_YEAR = {
    # Format: year: max_salary_cap_amount
    2011: 120375000,  # Enter max cap for 2011
    2012: 120600000,  # Enter max cap for 2012
    2013: 123600000,  # Enter max cap for 2013
    2014: 133000000,  # Enter max cap for 2014
    2015: 143280000,  # Enter max cap for 2015
    2016: 155270000,  # Enter max cap for 2016
    2017: 167000000,  # Enter max cap for 2017
    2018: 177200000,  # Enter max cap for 2018
    2019: 188200000,  # Enter max cap for 2019
    2020: 198200000,  # Enter max cap for 2020
    2021: 182500000,  # Enter max cap for 2021
    2022: 208200000,  # Enter max cap for 2022
    2023: 224800000,  # Enter max cap for 2023
    2024: 255400000,  # Enter max cap for 2024
}


def get_games_in_season(year: int) -> int:
    """
    Get the number of regular season games for a given NFL season year.
    
    Args:
        year: The NFL season year
        
    Returns:
        Number of regular season games (16 for 2022 and earlier, 17 for 2023 and later)
    """
    if year >= 2023:
        return 17
    else:
        return 16