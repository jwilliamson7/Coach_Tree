import os
import pandas as pd
import math
from pathlib import Path
from numpy import mean, nan, nanmean
from typing import List, Dict, Tuple, Union, Optional

from data_constants import (
    TEAM_FRANCHISE_MAPPINGS,
    CURRENT_TEAM_ABBREVIATIONS,
    EXCLUDED_ROLE_KEYWORDS,
    TENURE_CLASSIFICATIONS,
    DATA_FILES,
    ANALYSIS_CONFIG,
    FIRED_COACHES,
    EXCLUDED_HIRING_INSTANCES,
    get_all_feature_names,
    get_feature_dict,
    get_hiring_team_stat_dict,
    get_output_column_names
)


class CoachingDataProcessor:
    """Processes NFL coaching career data into feature-engineered datasets"""
    
    def __init__(self, coaches_dir: str = "Coaches", teams_dir: str = "Teams", 
                 league_dir: str = "League Data"):
        """Initialize with data directory paths"""
        self.coaches_dir = Path(coaches_dir)
        self.teams_dir = Path(teams_dir)
        self.league_dir = Path(league_dir)
        
    def _classify_coach_tenure(self, years: int) -> int:
        """Classify coaching tenure into categories"""
        if years <= 0:
            return -1
        elif years <= TENURE_CLASSIFICATIONS["short"][1]:
            return 0
        elif years <= TENURE_CLASSIFICATIONS["medium"][1]:
            return 1
        else:
            return 2
    
    def _classify_coaching_role(self, role: str) -> str:
        """Classify coaching role, excluding interim and non-coaching positions"""
        if not role or not isinstance(role, str):
            return "None"
        
        # Check for excluded keywords
        for keyword in EXCLUDED_ROLE_KEYWORDS:
            if keyword in role:
                return "None"
        
        # Exclude generic assistant roles
        if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
            return "None"
        
        # Classify specific roles
        if "Head Coach" in role and "Ass" not in role and "Interim" not in role:
            return "Head Coach"
        
        if "Coordinator" in role:
            if "Offensive Coordinator" in role and "Interim O" not in role:
                return "Offensive Coordinator"
            elif "Defensive Coordinator" in role and "Interim D" not in role:
                return "Defensive Coordinator"
            elif "Special" in role and "Interim S" not in role:
                return "Special Teams Coordinator"
            else:
                return "Position"
        
        return "Position"
    
    def _classify_coaching_level(self, level: str) -> str:
        """Classify coaching level (College, NFL, or None)"""
        if not level:
            return "None"
        if "College" in level:
            return "College"
        if "NFL" in level and level != "NFL Europe":
            return "NFL"
        return "None"
    
    def _safe_mean(self, values: Union[List, float, int]) -> float:
        """Calculate mean of list, handling NaN values and non-list inputs"""
        if not isinstance(values, list):
            return values
        
        clean_values = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
        if len(clean_values) == 0:
            return nan
        return nanmean(clean_values)
    
    def _get_team_abbreviation(self, team_name: str) -> str:
        """Get team abbreviation for current team names"""
        return CURRENT_TEAM_ABBREVIATIONS.get(team_name, team_name.lower()[:3])
    
    def _resolve_team_franchise(self, team_abbrev: str) -> List[str]:
        """Resolve team abbreviation to all historical variants"""
        franchise_list = TEAM_FRANCHISE_MAPPINGS.get(team_abbrev, team_abbrev)
        return franchise_list if isinstance(franchise_list, list) else [franchise_list]
    
    def _load_league_data(self, year: int, team_list: List[str], role: str, 
                         feature_dict: Dict) -> int:
        """Load and process league data for a specific year and role"""
        
        year_dir = self.league_dir / str(year)
        if not year_dir.exists():
            return 0
        
        try:
            # Load team and opponent data
            team_file = year_dir / f"{DATA_FILES['league_tables'][0]}.csv"
            opponent_file = year_dir / f"{DATA_FILES['league_tables'][1]}.csv"
            
            team_df = pd.read_csv(team_file)
            opponent_df = pd.read_csv(opponent_file)
            
        except FileNotFoundError:
            return 0
        
        # Find team in data
        team_found = False
        team_index = 0
        
        for i, team_abbrev in enumerate(team_list):
            team_row = team_df[team_df['Team Abbreviation'] == team_abbrev]
            if not team_row.empty:
                team_found = True
                team_index = i
                break
        
        if not team_found:
            print(f"Error: {team_list[0]} not found in year {year}")
            return -1
        
        # Get corresponding opponent row
        team_abbrev = team_list[team_index]
        team_row = team_df[team_df['Team Abbreviation'] == team_abbrev]
        opponent_row = opponent_df[opponent_df['Team Abbreviation'] == team_abbrev]
        
        # Process data based on role
        if role == "HC":
            # Head coach gets both team and opponent stats
            for col in team_row.columns:
                if col != 'Team Abbreviation':
                    feature_key = f"{col}__hc"
                    if feature_key in feature_dict:
                        feature_dict[feature_key].append(team_row[col].values[0])
            
            for col in opponent_row.columns:
                if col != 'Team Abbreviation':
                    feature_key = f"{col}__opp__hc"
                    if feature_key in feature_dict:
                        feature_dict[feature_key].append(opponent_row[col].values[0])
        
        elif role == "OC":
            # Offensive coordinator gets team offensive stats
            for col in team_row.columns:
                if col != 'Team Abbreviation':
                    feature_key = f"{col}__oc"
                    if feature_key in feature_dict:
                        feature_dict[feature_key].append(team_row[col].values[0])
        
        elif role == "DC":
            # Defensive coordinator gets opponent offensive stats (team's defensive performance)
            for col in opponent_row.columns:
                if col != 'Team Abbreviation':
                    feature_key = f"{col}__dc"
                    if feature_key in feature_dict:
                        feature_dict[feature_key].append(opponent_row[col].values[0])
        
        return team_index
    
    def _get_hiring_team_context(self, franchise: str, hire_year: int) -> List[float]:
        """Get hiring team's recent performance context"""
        
        team_dir = self.teams_dir / franchise
        if not team_dir.exists():
            return [nan] * len(get_hiring_team_stat_dict())
        
        # Load team record
        record_file = team_dir / DATA_FILES['team_record']
        if not record_file.exists():
            return [nan] * len(get_hiring_team_stat_dict())
        
        team_record = pd.read_csv(record_file)
        feature_dict = get_hiring_team_stat_dict()
        
        # Look at previous years
        for prev_year in [hire_year - 1, hire_year - 2]:
            year_records = team_record[team_record['Year'] == prev_year]
            if year_records.empty:
                continue
            
            record = year_records.iloc[0]
            wins = record['W']
            losses = record['L'] 
            ties = record.get('T', 0)
            
            # Calculate win percentage
            total_games = wins + losses + ties
            win_pct = (wins + 0.5 * ties) / total_games if total_games > 0 else 0
            feature_dict["hiring_team_win_pct"].append(win_pct)
            
            # Get league performance data
            year_dir = self.league_dir / str(prev_year)
            if year_dir.exists():
                try:
                    team_file = year_dir / f"{DATA_FILES['league_tables'][0]}.csv"
                    opponent_file = year_dir / f"{DATA_FILES['league_tables'][1]}.csv"
                    
                    team_df = pd.read_csv(team_file)
                    opponent_df = pd.read_csv(opponent_file)
                    
                    team_row = team_df[team_df['Team Abbreviation'] == franchise]
                    opponent_row = opponent_df[opponent_df['Team Abbreviation'] == franchise]
                    
                    if not team_row.empty and not opponent_row.empty:
                        feature_dict["hiring_team_points_scored"].append(team_row['PF (Points For)'].values[0])
                        feature_dict["hiring_team_points_allowed"].append(opponent_row['PF (Points For)'].values[0])
                        feature_dict["hiring_team_yards_offense"].append(team_row['Yds'].values[0])
                        feature_dict["hiring_team_yards_allowed"].append(opponent_row['Yds'].values[0])
                        feature_dict["hiring_team_yards_per_play"].append(team_row['Y/P'].values[0])
                        feature_dict["hiring_team_yards_per_play_allowed"].append(opponent_row['Y/P'].values[0])
                        feature_dict["hiring_team_turnovers_forced"].append(opponent_row['TO'].values[0])
                        feature_dict["hiring_team_turnovers_committed"].append(team_row['TO'].values[0])
                        
                except FileNotFoundError:
                    continue
        
        # Check playoff appearances
        playoff_file = team_dir / DATA_FILES['team_playoff']
        if playoff_file.exists():
            playoff_df = pd.read_csv(playoff_file)
            for prev_year in [hire_year - 1, hire_year - 2]:
                playoff_games = playoff_df[playoff_df['Year'] == prev_year]
                if prev_year == hire_year - 1:
                    feature_dict["hiring_team_num_playoff_appearances"].append(1 if not playoff_games.empty else 0)
                elif not feature_dict["hiring_team_num_playoff_appearances"]:
                    feature_dict["hiring_team_num_playoff_appearances"].append(1 if not playoff_games.empty else 0)
                elif not playoff_games.empty:
                    feature_dict["hiring_team_num_playoff_appearances"][0] += 1
        
        return [self._safe_mean(values) for values in feature_dict.values()]
    
    def _process_coach_career(self, coach_name: str) -> List[List]:
        """Process a single coach's career into training instances"""
        
        coach_dir = self.coaches_dir / coach_name
        if not coach_dir.exists():
            return []
        
        # Load coach data files
        dfs = []
        for table_name in DATA_FILES['coach_tables']:
            file_path = coach_dir / f"{table_name}.csv"
            if file_path.exists():
                dfs.append(pd.read_csv(file_path))
            else:
                dfs.append(pd.DataFrame())
        
        if len(dfs) < 3 or dfs[2].empty:  # Need coaching history
            return []
        
        results_df, ranks_df, history_df = dfs
        
        # Track coaching progression
        career_instances = []
        feature_dict = get_feature_dict()
        is_head_coach = False
        prev_franchise = None
        prev_year_check = None
        last_hc_year = None  # Track last year as head coach for gap detection
        
        # Process each year of coaching history
        for _, row in history_df.iterrows():
            year = row['Year']
            level = self._classify_coaching_level(row.get('Level', ''))
            role = self._classify_coaching_role(row.get('Role', ''))
            age = row.get('Age', 0)
            
            # Handle demotion from head coach
            if is_head_coach and (level != 'NFL' or role != "Head Coach"):
                if career_instances:
                    # Calculate tenure for previous instance
                    prev_instance = career_instances[-1]
                    prev_hire_year = prev_instance[1]
                    tenure_years = (prev_year_check + 1) - prev_hire_year if prev_year_check else year - prev_hire_year
                    prev_instance.append(self._classify_coach_tenure(tenure_years))
                is_head_coach = False
            
            # Update experience counters
            if level == "College":
                if role == "Position":
                    feature_dict["num_yr_col_pos"] += 1
                elif role in ["Offensive Coordinator", "Defensive Coordinator", "Special Teams Coordinator"]:
                    feature_dict["num_yr_col_coor"] += 1
                elif role == "Head Coach":
                    feature_dict["num_yr_col_hc"] += 1
                    
            elif level == "NFL":
                if role == "Position":
                    feature_dict["num_yr_nfl_pos"] += 1
                elif role != "None":
                    # Get team abbreviation from ranks data or derive for current year hires
                    franchise = None
                    franchise_list = None
                    
                    if not ranks_df.empty:
                        team_rows = ranks_df[ranks_df['Year'] == year]
                        if not team_rows.empty:
                            franchise = team_rows.iloc[0]['Tm'].lower()
                            franchise_list = self._resolve_team_franchise(franchise)
                    
                    # For current year head coach hires without ranking data, derive team from role
                    # This applies to NEW hires and team changes in current year
                    if (franchise is None and role == "Head Coach" and 
                        year >= ANALYSIS_CONFIG['current_year']):
                        # Parse team from history data for current year hires/moves
                        employer = row.get('Employer', '')
                        if employer:
                            current_franchise = self._get_team_abbreviation(employer)
                            current_franchise_list = self._resolve_team_franchise(current_franchise)
                            
                            # Only set franchise if this is genuinely a new hire or team change
                            if not is_head_coach or current_franchise_list != prev_franchise:
                                franchise = current_franchise
                                franchise_list = current_franchise_list
                    
                    if franchise_list:
                        
                        # FIRST: Load performance data for ALL roles (before creating hiring instances)
                        if role in ["Offensive Coordinator", "Defensive Coordinator", "Special Teams Coordinator"]:
                            feature_dict["num_yr_nfl_coor"] += 1
                            
                            # Load performance data
                            if role == "Offensive Coordinator":
                                self._load_league_data(year, franchise_list, "OC", feature_dict)
                            elif role == "Defensive Coordinator":
                                self._load_league_data(year, franchise_list, "DC", feature_dict)
                                
                        elif role == "Head Coach":
                            # FIRST: Check if this is a new head coaching hire
                            # Conditions: first time HC, team change, OR gap > 1 year since last HC role
                            is_new_hire = (not is_head_coach or 
                                         franchise_list != prev_franchise or
                                         (last_hc_year is not None and year - last_hc_year > 1))
                            
                            if is_new_hire:
                                # Create new hiring instance with ONLY prior experience data
                                is_head_coach = True
                                feature_dict["age"] = age
                                prev_franchise = franchise_list
                                
                                # Calculate 2-year winning percentage
                                win_results = []
                                for result_year in [year, year + 1]:
                                    if not results_df.empty:
                                        result_rows = results_df[results_df['Year'] == result_year]
                                        if not result_rows.empty:
                                            result = result_rows.iloc[0]
                                            wins = result.get('W', 0)
                                            losses = result.get('L', 0)
                                            ties = result.get('T', 0)
                                            total = wins + losses + ties
                                            if total > 0:
                                                win_pct = (wins + 0.5 * ties) / total
                                                win_results.append(win_pct)
                                
                                # Finalize previous instance if exists
                                if career_instances and len(career_instances[-1]) == ANALYSIS_CONFIG['expected_feature_count'] - 1:
                                    prev_instance = career_instances[-1]
                                    prev_hire_year = prev_instance[1]
                                    
                                    # Calculate actual tenure based on when the previous stint ended
                                    # If there's a gap, the previous stint ended at last_hc_year + 1
                                    # If it's a team change with no gap, use the year before current hire
                                    if last_hc_year is not None and year - last_hc_year > 1:
                                        # Gap detected - previous stint ended after last_hc_year
                                        tenure_years = (last_hc_year + 1) - prev_hire_year
                                    else:
                                        # No gap - previous stint ended just before current hire
                                        tenure_years = year - prev_hire_year
                                    
                                    prev_instance.append(self._classify_coach_tenure(tenure_years))
                                
                                # Create new instance with PREVIOUS experience only
                                new_instance = [coach_name, year]
                                
                                # Ensure feature order matches get_all_feature_names() order
                                feature_names = get_all_feature_names()
                                feature_values = []
                                for feature_name in feature_names:
                                    if feature_name in feature_dict:
                                        feature_values.append(self._safe_mean(feature_dict[feature_name]))
                                    else:
                                        feature_values.append(0)  # Default for missing features
                                
                                new_instance.extend(feature_values)
                                
                                # Get team index for hiring context
                                team_index = 0
                                for i, team_abbrev in enumerate(franchise_list):
                                    year_dir = self.league_dir / str(year)
                                    if year_dir.exists():
                                        try:
                                            team_file = year_dir / f"{DATA_FILES['league_tables'][0]}.csv"
                                            team_df = pd.read_csv(team_file)
                                            team_row = team_df[team_df['Team Abbreviation'] == team_abbrev]
                                            if not team_row.empty:
                                                team_index = i
                                                break
                                        except FileNotFoundError:
                                            continue
                                    
                                new_instance.extend(self._get_hiring_team_context(franchise_list[team_index], year))
                                new_instance.append(self._safe_mean(win_results))
                                
                                # Check if this instance should be excluded from the dataset
                                if (coach_name, year) not in EXCLUDED_HIRING_INSTANCES:
                                    career_instances.append(new_instance)
                                    feature_dict["num_times_hc"] += 1
                                else:
                                    print(f"\tExcluded hiring instance: {coach_name} ({year})")
                                    # Do NOT count excluded instances in prior HC count for future hires
                            
                            # NOW add current year's HC performance for future hiring instances
                            self._load_league_data(year, franchise_list, "HC", feature_dict)
                            feature_dict["num_yr_nfl_hc"] += 1
                            
                            # Update last HC year for gap detection (for all HC years, not just new hires)
                            last_hc_year = year
            
            prev_year_check = year
        
        # Handle final instance
        if career_instances and len(career_instances[-1]) == ANALYSIS_CONFIG['expected_feature_count'] - 1:
            final_instance = career_instances[-1]
            hire_year = final_instance[1]
            
            # Determine tenure classification
            if hire_year > ANALYSIS_CONFIG['current_year']:
                # Future hire - exclude completely
                career_instances.pop()
                print(f'\tExcluded future hire: {coach_name}, hire year: {hire_year}')
            elif hire_year >= ANALYSIS_CONFIG['cutoff_year']:
                # Recent hire - check if still active or was fired
                # Special handling for known fired coaches vs still active
                # For coaches hired 2022+, check if we have evidence they were fired
                if coach_name in FIRED_COACHES or prev_year_check < ANALYSIS_CONFIG['current_year'] - 1:
                    # Was fired/resigned - we know the actual tenure
                    tenure_years = (prev_year_check + 1) - hire_year
                    final_instance.append(self._classify_coach_tenure(tenure_years))
                    print(f'\tCalculated tenure for fired coach: {coach_name}, tenure: {tenure_years} years')
                else:
                    # Still active - insufficient data for final classification
                    final_instance.append(-1)
                    print(f'\tExcluded tenure classification: {coach_name}')
            else:
                # Calculate final tenure
                tenure_years = (prev_year_check + 1) - hire_year
                final_instance.append(self._classify_coach_tenure(tenure_years))
        
        # Validate instance lengths
        for i, instance in enumerate(career_instances):
            if len(instance) != ANALYSIS_CONFIG['expected_feature_count']:
                print(f'Error: {coach_name}, instance {i}, length {len(instance)} != {ANALYSIS_CONFIG["expected_feature_count"]}')
        
        return career_instances
    
    def process_all_coaches(self) -> pd.DataFrame:
        """Process all coaches and return combined dataset"""
        
        coach_dirs = [d for d in self.coaches_dir.iterdir() if d.is_dir()]
        all_instances = []
        
        print(f"Processing {len(coach_dirs)} coaches...")
        
        for i, coach_dir in enumerate(coach_dirs, 1):
            coach_name = coach_dir.name
            print(f"Parsing coach {i}: {coach_name}")
            
            coach_instances = self._process_coach_career(coach_name)
            all_instances.extend(coach_instances)
            
            
            
        
        # Create DataFrame
        columns = get_output_column_names()
        df = pd.DataFrame(data=all_instances, columns=columns)
        
        
        print(f'Processed {len(all_instances)} coaching instances')
        return df


def main():
    """Main execution function"""
    print("Starting Coaching Data Processing...")
    
    # Initialize processor
    processor = CoachingDataProcessor()
    
    # Process all coaching data
    master_df = processor.process_all_coaches()
    
    # Save results
    output_file = "master_data.csv"
    master_df.to_csv(output_file, index=True)
    
    print(f"\nProcessing completed!")
    print(f"Dataset saved to: {output_file}")
    print(f"Total coaching instances: {len(master_df)}")


if __name__ == "__main__":
    main()