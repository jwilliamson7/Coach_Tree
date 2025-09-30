#!/usr/bin/env python3
"""
Create Yearly Coach Performance Data

This script creates yearly performance metrics for each coach's career,
generating one data point per year instead of just at hiring instances.
Filters for 1970 onwards and uses the current project structure.

Usage:
    python scripts/create_yearly_coach_performance_data.py
    python scripts/create_yearly_coach_performance_data.py --output-dir data/processed
"""

import os
import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# Add parent directory to path to import constants
sys.path.append(str(Path(__file__).parent.parent))
from crawlers.utils.data_constants import (
    SPOTRAC_TO_PFR_MAPPINGS,
    BASE_TEAM_STATISTICS,
    ROLE_SUFFIXES,
    CORE_COACHING_FEATURES,
    EXCLUDED_ROLE_KEYWORDS,
    TENURE_CLASSIFICATIONS,
    get_all_feature_names,
    get_feature_dict
)

class YearlyCoachPerformanceProcessor:
    """Processes NFL coaching data into yearly performance metrics"""
    
    def __init__(self, coaches_dir: str = "data/raw/Coaches", 
                 league_dir: str = "data/processed/League Data",
                 output_dir: str = "data/processed/Coaching"):
        """Initialize with data directory paths"""
        self.coaches_dir = Path(coaches_dir)
        self.league_dir = Path(league_dir) 
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Team mappings for standardization
        self.team_mappings = {v: v for v in SPOTRAC_TO_PFR_MAPPINGS.values()}
        
        # Historical team corrections
        self.team_corrections = {
            'BAL': 'RAV',  # Baltimore Ravens
            'HOU': 'HTX',  # Houston Texans  
            'LAC': 'SDG',  # LA Chargers
            'LAS': 'RAI',  # Las Vegas Raiders
            'TEN': 'OTI',  # Tennessee Titans
            'IND': 'CLT',  # Indianapolis Colts
            'ARI': 'CRD',  # Arizona Cardinals
            'GB': 'GNB',   # Green Bay Packers
            'KC': 'KAN',   # Kansas City Chiefs
            'NE': 'NWE',   # New England Patriots
            'NO': 'NOR',   # New Orleans Saints
            'SF': 'SFO',   # San Francisco 49ers
            'TB': 'TAM',   # Tampa Bay Buccaneers
            'WAS': 'WAS',  # Washington
            'LV': 'RAI',   # Las Vegas Raiders
            'OAK': 'RAI',  # Oakland Raiders -> Las Vegas Raiders
            'STL': 'RAM',  # St. Louis Rams -> Los Angeles Rams
            'LAR': 'RAM',  # Los Angeles Rams
            'PHO': 'CRD',  # Phoenix Cardinals -> Arizona Cardinals
            'BOS': 'NWE',  # Boston Patriots -> New England Patriots
        }
    
    def standardize_team_name(self, team: str) -> str:
        """Standardize team abbreviation to current PFR format"""
        if not team:
            return team
            
        team = team.upper().strip()
        
        # Apply corrections if needed
        if team in self.team_corrections:
            team = self.team_corrections[team]
            
        # Return lowercase for consistency with data files
        return team.lower()
    
    def classify_coaching_role(self, role: str) -> str:
        """Classify coaching role, excluding interim, suspended, and non-coaching positions"""
        if not role or not isinstance(role, str):
            return "None"
        
        # Check for excluded keywords
        for keyword in EXCLUDED_ROLE_KEYWORDS:
            if keyword in role:
                return "None"
        
        # Exclude suspended seasons - these don't count toward experience
        if "Suspended" in role or "suspended" in role:
            return "None"
        
        # Exclude generic assistant roles
        if ("Assistant" in role or "Asst" in role) and "/" not in role and "\\" not in role:
            return "None"
        
        # Classify specific roles
        if "Head Coach" in role and "Ass" not in role and "Interim" not in role and "Suspended" not in role and "suspended" not in role:
            return "HC"
        
        if "Coordinator" in role:
            if "Offensive Coordinator" in role and "Interim O" not in role:
                return "OC"
            elif "Defensive Coordinator" in role and "Interim D" not in role:
                return "DC"
            elif "Special" in role and "Interim S" not in role:
                return "STC"
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
    
    def _resolve_team_franchise(self, team_abbrev: str) -> List[str]:
        """Resolve team abbreviation to all historical variants"""
        if not team_abbrev:
            return []
        # For this simplified version, just return the team as-is
        return [team_abbrev]
    
    def _load_league_data(self, year: int, team_list: List[str], role: str, feature_dict: Dict) -> int:
        """Load and process normalized league data for a specific year and role"""
        
        year_dir = self.league_dir / str(year)
        if not year_dir.exists():
            return 0
        
        try:
            # Load normalized team and opponent data
            team_file = year_dir / "league_team_data_normalized.csv"
            opponent_file = year_dir / "league_opponent_data_normalized.csv"
            
            if not team_file.exists() or not opponent_file.exists():
                return 0
            
            team_df = pd.read_csv(team_file)
            opponent_df = pd.read_csv(opponent_file)
            
        except FileNotFoundError:
            return 0
        
        # Find team in data using Team Abbreviation column
        team_found = False
        team_abbrev = None
        
        for abbrev in team_list:
            team_row = team_df[team_df['Team Abbreviation'] == abbrev]
            if not team_row.empty:
                team_found = True
                team_abbrev = abbrev
                break
        
        if not team_found:
            return -1
        
        # Get corresponding opponent row
        team_row = team_df[team_df['Team Abbreviation'] == team_abbrev]
        opponent_row = opponent_df[opponent_df['Team Abbreviation'] == team_abbrev]
        
        # Process data based on role - append ALL normalized values
        if role == "HC":
            # Head coach gets both team and opponent stats
            for stat in BASE_TEAM_STATISTICS:
                # Try to find the column in the data
                if stat in team_row.columns:
                    feature_key = f"{stat}__hc"
                    if feature_key in feature_dict:
                        value = team_row[stat].iloc[0]
                        if pd.notna(value):  # Only append non-NaN values
                            feature_dict[feature_key].append(value)
                    
                if stat in opponent_row.columns:
                    feature_key = f"{stat}__opp__hc"
                    if feature_key in feature_dict:
                        value = opponent_row[stat].iloc[0]
                        if pd.notna(value):
                            feature_dict[feature_key].append(value)
        
        elif role == "OC":
            # Offensive coordinator gets team offensive stats
            for stat in BASE_TEAM_STATISTICS:
                if stat in team_row.columns:
                    feature_key = f"{stat}__oc"
                    if feature_key in feature_dict:
                        value = team_row[stat].iloc[0]
                        if pd.notna(value):
                            feature_dict[feature_key].append(value)
        
        elif role == "DC":
            # Defensive coordinator gets opponent offensive stats (team's defensive performance)
            for stat in BASE_TEAM_STATISTICS:
                if stat in opponent_row.columns:
                    feature_key = f"{stat}__dc"
                    if feature_key in feature_dict:
                        value = opponent_row[stat].iloc[0]
                        if pd.notna(value):
                            feature_dict[feature_key].append(value)
        
        return 0

    
    def calculate_career_metrics(self, coach_history: pd.DataFrame, coach_ranks: pd.DataFrame, current_year: int) -> Dict:
        """Calculate cumulative career metrics up to (but not including) current year using full feature set"""
        
        # Initialize feature dictionary with all 150+ features
        feature_dict = get_feature_dict()
        
        # Core experience metrics
        core_metrics = {
            'num_times_hc': 0,
            'num_yr_col_pos': 0,
            'num_yr_col_coor': 0,
            'num_yr_col_hc': 0,
            'num_yr_nfl_pos': 0,
            'num_yr_nfl_coor': 0,
            'num_yr_nfl_hc': 0
        }
        
        # Filter to years before current year and 1970+
        prior_years = coach_history[
            (coach_history['Year'] < current_year) & 
            (coach_history['Year'] >= 1970)
        ]
        
        if prior_years.empty:
            # Return feature dict with default values
            result_metrics = {}
            feature_names = get_all_feature_names()
            for feature_name in feature_names:
                # Skip 'age' since we use 'Age' from the actual row data
                if feature_name == 'age':
                    continue
                    
                if feature_name in core_metrics:
                    result_metrics[feature_name] = core_metrics[feature_name]
                else:
                    result_metrics[feature_name] = np.nan
            return result_metrics
        
        # Track HC hiring instances for num_times_hc
        hc_years = []
        hc_hires = []  # Track actual hiring events
        prev_franchise = None
        last_hc_year = None
        
        # Process each year of prior experience
        for _, row in prior_years.iterrows():
            year = row['Year']
            level = row.get('Level', '')
            role = row.get('Role', '')
            
            # Get team abbreviation from ranks data if available
            team = None
            if not coach_ranks.empty:
                rank_row = coach_ranks[coach_ranks['Year'] == year]
                if not rank_row.empty:
                    team = self.standardize_team_name(rank_row.iloc[0].get('Tm', ''))
            
            classified_role = self.classify_coaching_role(role)
            
            # Count experience by level and role using improved level classification
            classified_level = self._classify_coaching_level(level)
            
            if classified_level == "College":
                if classified_role == "Position":
                    core_metrics["num_yr_col_pos"] += 1
                elif classified_role in ["OC", "DC", "STC"]:
                    core_metrics["num_yr_col_coor"] += 1
                elif classified_role == "HC":
                    core_metrics["num_yr_col_hc"] += 1
                    
            elif classified_level == "NFL":
                if classified_role == "Position":
                    core_metrics["num_yr_nfl_pos"] += 1
                elif classified_role in ["OC", "DC", "STC"]:
                    core_metrics["num_yr_nfl_coor"] += 1
                    
                    # Load performance data for coordinator roles
                    franchise_list = self._resolve_team_franchise(team) if team else []
                    if franchise_list:
                        if classified_role == "OC":
                            self._load_league_data(year, franchise_list, "OC", feature_dict)
                        elif classified_role == "DC":
                            self._load_league_data(year, franchise_list, "DC", feature_dict)
                            
                elif classified_role == "HC":
                    core_metrics["num_yr_nfl_hc"] += 1
                    hc_years.append(year)
                    
                    # Load performance data for HC role
                    franchise_list = self._resolve_team_franchise(team) if team else []
                    if franchise_list:
                        self._load_league_data(year, franchise_list, "HC", feature_dict)
                        
                        # Check if this is a new HC hire
                        is_new_hire = (len(hc_years) == 1 or  # First HC job
                                     franchise_list != prev_franchise)  # Team change only
                        # Note: Removed gap detection to avoid counting returns from suspension as new hires
                        
                        if is_new_hire:
                            hc_hires.append(year)
                            prev_franchise = franchise_list
                        
                        last_hc_year = year
        
        # Set num_times_hc to the number of prior HC hires
        core_metrics["num_times_hc"] = len(hc_hires)
        
        # Convert feature lists to averages
        result_metrics = {}
        feature_names = get_all_feature_names()
        
        for feature_name in feature_names:
            # Skip 'age' since we use 'Age' from the actual row data
            if feature_name == 'age':
                continue
                
            if feature_name in core_metrics:
                result_metrics[feature_name] = core_metrics[feature_name]
            elif feature_name in feature_dict:
                # Calculate mean of accumulated performance data
                values = feature_dict[feature_name]
                if isinstance(values, list) and len(values) > 0:
                    clean_values = [x for x in values if not (isinstance(x, float) and np.isnan(x))]
                    result_metrics[feature_name] = np.mean(clean_values) if clean_values else np.nan
                else:
                    result_metrics[feature_name] = np.nan
            else:
                result_metrics[feature_name] = np.nan
        
        return result_metrics
    
    def process_coach_career(self, coach_name: str) -> List[Dict]:
        """Process a single coach's career into yearly performance records for HEAD COACH years only"""
        coach_dir = self.coaches_dir / coach_name
        
        if not coach_dir.exists():
            return []
        
        # Load coaching history and ranks
        history_file = coach_dir / "all_coaching_history.csv"
        ranks_file = coach_dir / "all_coaching_ranks.csv"
        
        if not history_file.exists():
            return []
        
        try:
            history_df = pd.read_csv(history_file)
            # Load ranks data if available for team abbreviations
            if ranks_file.exists():
                ranks_df = pd.read_csv(ranks_file)
            else:
                ranks_df = pd.DataFrame()
        except Exception as e:
            self.logger.warning(f"Error loading data for {coach_name}: {e}")
            return []
        
        if history_df.empty:
            return []
        
        # Filter for 1970 onwards and NFL only
        nfl_history = history_df[
            (history_df['Year'] >= 1970) & 
            (history_df['Level'].apply(lambda x: self._classify_coaching_level(x) == 'NFL'))
        ].copy()
        
        if nfl_history.empty:
            return []
        
        yearly_records = []
        
        # Process each year of NFL coaching - but only create records for HC years
        for _, row in nfl_history.iterrows():
            year = row['Year']
            role = row.get('Role', '')
            age = row.get('Age', np.nan)
            
            # Get team abbreviation from ranks data
            team = None
            if not ranks_df.empty:
                rank_row = ranks_df[ranks_df['Year'] == year]
                if not rank_row.empty:
                    team = self.standardize_team_name(rank_row.iloc[0].get('Tm', ''))
            
            classified_role = self.classify_coaching_role(role)
            
            # ONLY create records for HEAD COACH years
            if classified_role != 'HC':
                continue
            
            # Calculate career metrics up to this year (from all prior experience)
            career_metrics = self.calculate_career_metrics(history_df, ranks_df, year)
            
            # Note: Raw team performance metrics removed - only normalized features included
            
            # Create yearly record
            yearly_record = {
                'Coach': coach_name,
                'Year': year,
                'Team': team,
                'Role': classified_role,
                'Age': age,
                **career_metrics
            }
            
            yearly_records.append(yearly_record)
        
        return yearly_records
    
    def process_all_coaches(self, specific_coach=None) -> pd.DataFrame:
        """Process all coaches (or a specific coach) and return combined yearly dataset"""
        if specific_coach:
            # Process only the specified coach
            coach_dirs = [self.coaches_dir / specific_coach]
            if not coach_dirs[0].exists():
                self.logger.error(f"Coach directory not found: {specific_coach}")
                return pd.DataFrame()
        else:
            # Process all coaches
            coach_dirs = [d for d in self.coaches_dir.iterdir() if d.is_dir()]
        
        all_records = []
        
        self.logger.info(f"Processing {len(coach_dirs)} coaches...")
        
        for i, coach_dir in enumerate(coach_dirs, 1):
            coach_name = coach_dir.name
            self.logger.info(f"Processing coach {i}/{len(coach_dirs)}: {coach_name}")
            
            coach_records = self.process_coach_career(coach_name)
            all_records.extend(coach_records)
        
        if not all_records:
            self.logger.warning("No coaching records found")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(all_records)
        
        # Sort by coach name and year
        df = df.sort_values(['Coach', 'Year'], ignore_index=True)
        
        self.logger.info(f"Processed {len(df)} yearly coaching records")
        self.logger.info(f"Years covered: {df['Year'].min()}-{df['Year'].max()}")
        self.logger.info(f"Unique coaches: {df['Coach'].nunique()}")
        
        return df
    
    def save_data(self, df: pd.DataFrame) -> bool:
        """Save yearly coaching performance data"""
        try:
            # Main output file
            output_file = self.output_dir / "yearly_coach_performance.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved yearly coaching data to {output_file}")
            
            # Create metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Records': len(df),
                'Unique_Coaches': df['Coach'].nunique(),
                'Unique_Teams': df['Team'].nunique(),
                'Year_Range': f"{df['Year'].min()}-{df['Year'].max()}",
                'HC_Records': len(df[df['Role'] == 'HC']),
                'Coordinator_Records': len(df[df['Role'].isin(['OC', 'DC', 'STC'])]),
                'Position_Coach_Records': len(df[df['Role'] == 'Position']),
                'Description': 'Yearly coaching performance metrics for each coach-year combination from 1970 onwards'
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "yearly_coach_performance_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False


def main():
    """Main function to create yearly coach performance data"""
    parser = argparse.ArgumentParser(
        description='Create yearly coach performance data from coaching histories'
    )
    parser.add_argument(
        '--coaches-dir',
        type=str,
        default='data/raw/Coaches',
        help='Directory containing coach data (default: data/raw/Coaches)'
    )
    parser.add_argument(
        '--league-dir',
        type=str,
        default='data/processed/League Data',
        help='Directory containing league data (default: data/processed/League Data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/Coaching',
        help='Output directory for processed data (default: data/processed/Coaching)'
    )
    parser.add_argument(
        '--coach',
        type=str,
        default=None,
        help='Process only a specific coach (e.g., "Andy Reid")'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = YearlyCoachPerformanceProcessor(
        coaches_dir=args.coaches_dir,
        league_dir=args.league_dir,
        output_dir=args.output_dir
    )
    
    print("Creating yearly coach performance data...")
    print(f"Coaches directory: {args.coaches_dir}")
    print(f"League directory: {args.league_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Process coaching data (all or specific coach)
    yearly_df = processor.process_all_coaches(specific_coach=args.coach)
    
    if yearly_df.empty:
        print("Error: No coaching data processed")
        sys.exit(1)
    
    # Save the data
    if processor.save_data(yearly_df):
        print(f"\nProcessing completed successfully!")
        print(f"Generated {len(yearly_df)} yearly coaching records")
        print(f"Covering {yearly_df['Coach'].nunique()} unique coaches")
        print(f"Years: {yearly_df['Year'].min()}-{yearly_df['Year'].max()}")
        print(f"Results saved to: {args.output_dir}")
    else:
        print("Error: Failed to save results")
        sys.exit(1)


if __name__ == "__main__":
    main()