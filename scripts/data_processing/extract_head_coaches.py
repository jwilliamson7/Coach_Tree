#!/usr/bin/env python3
"""
Extract Head Coach Information from Coaching Data

This script processes all coaching data in data/raw/Coaches/ to extract team-year
combinations where the coach's role was "HC" (Head Coach). Creates a mapping
table of team-year to head coach for use in the final dataset combination.

Usage:
    python scripts/extract_head_coaches.py
    python scripts/extract_head_coaches.py --output-dir data/processed
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import glob

# Add parent directory to path to import constants
sys.path.append(str(Path(__file__).parent.parent))
from crawlers.utils.data_constants import SPOTRAC_TO_PFR_MAPPINGS

class HeadCoachExtractor:
    """Extracts head coach information from coaching data"""
    
    def __init__(self, coaches_dir: str = "data/raw/Coaches", 
                 output_dir: str = "data/processed/Coaching"):
        """
        Initialize the extractor
        
        Args:
            coaches_dir: Directory containing coach data
            output_dir: Directory to save processed data
        """
        self.coaches_dir = Path(coaches_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Team mapping for standardization
        self.team_mappings = {v: v for v in SPOTRAC_TO_PFR_MAPPINGS.values()}
        
        # Common team abbreviation corrections
        self.team_corrections = {
            'BAL': 'RAV',  # Baltimore Ravens
            'HOU': 'HTX',  # Houston Texans  
            'LAC': 'SDG',  # LA Chargers (formerly San Diego)
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
            'WAS': 'WAS',  # Washington (already correct)
            'LV': 'RAI',   # Las Vegas Raiders
            'OAK': 'RAI'   # Oakland Raiders -> Las Vegas Raiders
        }
        
    def standardize_team_name(self, team: str) -> str:
        """
        Standardize team abbreviation to PFR format
        
        Args:
            team: Team abbreviation from coaching data
            
        Returns:
            Standardized team abbreviation
        """
        team = team.upper().strip()
        
        # Apply corrections if needed
        if team in self.team_corrections:
            team = self.team_corrections[team]
            
        return team
    
    def process_coach_file(self, coach_name: str, ranks_file: Path) -> Optional[pd.DataFrame]:
        """
        Process a single coach's ranks file to extract HC years
        
        Args:
            coach_name: Name of the coach
            ranks_file: Path to the coaching ranks file
            
        Returns:
            DataFrame with HC years or None if no HC data
        """
        try:
            df = pd.read_csv(ranks_file)
            
            # Check if required columns exist
            if 'Role' not in df.columns or 'Year' not in df.columns or 'Tm' not in df.columns:
                self.logger.warning(f"Missing required columns in {ranks_file}")
                return None
            
            # Filter for Head Coach role only
            hc_df = df[df['Role'] == 'HC'].copy()
            
            if hc_df.empty:
                self.logger.debug(f"No HC role found for {coach_name}")
                return None
            
            # Try to load results file to determine if coach started season
            results_file = ranks_file.parent / "all_coaching_results.csv"
            is_starter = False
            if results_file.exists():
                try:
                    results_df = pd.read_csv(results_file)
                    # Check if coach has games from week 1 (G column should be high, or W+L+T should be near full season)
                    if 'G' in results_df.columns:
                        hc_df['Games'] = hc_df['Year'].apply(
                            lambda y: results_df[results_df['Year'] == y]['G'].iloc[0] 
                            if len(results_df[results_df['Year'] == y]) > 0 else 0
                        )
                        # If coach has 10+ games, they likely started the season
                        hc_df['Is_Starter'] = hc_df['Games'] >= 10
                    elif all(col in results_df.columns for col in ['W', 'L', 'T']):
                        hc_df['Total_Games'] = hc_df['Year'].apply(
                            lambda y: (results_df[results_df['Year'] == y][['W', 'L', 'T']].sum(axis=1)).iloc[0]
                            if len(results_df[results_df['Year'] == y]) > 0 else 0
                        )
                        hc_df['Is_Starter'] = hc_df['Total_Games'] >= 10
                    else:
                        hc_df['Is_Starter'] = True  # Default to true if we can't determine
                except:
                    hc_df['Is_Starter'] = True  # Default to true if error reading
            else:
                hc_df['Is_Starter'] = True  # Default to true if no results file
            
            # Extract relevant columns
            result_df = hc_df[['Year', 'Tm']].copy()
            result_df['Coach'] = coach_name
            result_df['Is_Starter'] = hc_df.get('Is_Starter', True)
            
            # Standardize team names
            result_df['Team'] = result_df['Tm'].apply(self.standardize_team_name)
            result_df = result_df.drop(columns=['Tm'])
            
            # Ensure Year is integer
            result_df['Year'] = result_df['Year'].astype(int)
            
            # Filter for 1970 onwards (as requested) and reasonable upper bound
            result_df = result_df[
                (result_df['Year'] >= 1970) & 
                (result_df['Year'] <= datetime.now().year + 1)
            ]
            
            if not result_df.empty:
                self.logger.info(f"Found {len(result_df)} HC years for {coach_name}")
                return result_df[['Team', 'Year', 'Coach', 'Is_Starter']]
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing {ranks_file}: {e}")
            return None
    
    def extract_all_head_coaches(self) -> Optional[pd.DataFrame]:
        """
        Extract head coach information from all coaching files
        
        Returns:
            Combined DataFrame with all team-year-coach combinations
        """
        all_hc_data = []
        processed_coaches = 0
        failed_coaches = 0
        
        # Get all coach directories
        coach_dirs = [d for d in self.coaches_dir.iterdir() if d.is_dir()]
        
        self.logger.info(f"Processing {len(coach_dirs)} coach directories")
        
        for coach_dir in coach_dirs:
            coach_name = coach_dir.name
            ranks_file = coach_dir / "all_coaching_ranks.csv"
            
            if not ranks_file.exists():
                self.logger.warning(f"No coaching ranks file found for {coach_name}")
                failed_coaches += 1
                continue
            
            hc_data = self.process_coach_file(coach_name, ranks_file)
            
            if hc_data is not None:
                all_hc_data.append(hc_data)
                processed_coaches += 1
            else:
                failed_coaches += 1
        
        if not all_hc_data:
            self.logger.error("No head coach data extracted from any files")
            return None
        
        # Combine all data
        combined_df = pd.concat(all_hc_data, ignore_index=True)
        
        # Sort by Team, Year, Coach for consistency
        combined_df = combined_df.sort_values(['Team', 'Year', 'Coach'], ignore_index=True)
        
        self.logger.info(f"Successfully processed {processed_coaches} coaches")
        self.logger.info(f"Failed to process {failed_coaches} coaches")
        self.logger.info(f"Total HC team-year combinations: {len(combined_df)}")
        
        return combined_df
    
    def handle_duplicate_coaches(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle cases where multiple coaches have the same team-year
        (e.g., interim coaches, mid-season changes)
        
        Args:
            df: DataFrame with potential duplicates
            
        Returns:
            DataFrame with duplicates resolved
        """
        # Check for duplicates
        duplicates = df.groupby(['Team', 'Year']).size()
        duplicate_team_years = duplicates[duplicates > 1]
        
        if len(duplicate_team_years) > 0:
            self.logger.info(f"Found {len(duplicate_team_years)} team-years with multiple coaches")
            
            # For duplicates, we'll keep all coaches but mark them
            df_with_order = df.copy()
            df_with_order['Coach_Order'] = df_with_order.groupby(['Team', 'Year']).cumcount() + 1
            df_with_order['Total_Coaches'] = df_with_order.groupby(['Team', 'Year'])['Coach'].transform('count')
            
            # Determine primary coach - the one who STARTED the season
            # Logic: If a coach was HC in the prior year and is listed for current year, they started
            def get_primary_coach(group):
                if len(group) == 1:
                    # Only one coach, they're primary
                    return group.iloc[0]['Coach']
                
                # Multiple coaches - check who was HC in prior year
                current_year = group.iloc[0]['Year']
                prior_year = current_year - 1
                
                # Get coaches who were HC in prior year for this team
                prior_year_coaches = df_with_order[
                    (df_with_order['Year'] == prior_year) & 
                    (df_with_order['Team'] == group.iloc[0]['Team'])
                ]['Coach'].unique()
                
                # Check if any current year coaches were also HC in prior year
                for coach in group['Coach'].unique():
                    if coach in prior_year_coaches:
                        return coach
                
                # If no prior year match, return first coach alphabetically for consistency
                return sorted(group['Coach'].unique())[0]
            
            # Get primary coach for each team-year group
            primary_coaches = df_with_order.groupby(['Team', 'Year']).apply(get_primary_coach)
            # Map back to all rows
            df_with_order['Primary_Coach'] = df_with_order.set_index(['Team', 'Year']).index.map(primary_coaches.to_dict()).values
            
            # For cases with multiple coaches, concatenate names
            multiple_coach_mask = df_with_order['Total_Coaches'] > 1
            if multiple_coach_mask.any():
                # Create combined coach names for team-years with multiple coaches
                combined_coaches = df_with_order[multiple_coach_mask].groupby(['Team', 'Year'])['Coach'].apply(
                    lambda x: ' / '.join(sorted(x.unique()))
                ).reset_index()
                combined_coaches = combined_coaches.rename(columns={'Coach': 'Combined_Coach'})
                
                # Merge back
                df_with_order = df_with_order.merge(combined_coaches, on=['Team', 'Year'], how='left')
                df_with_order['Combined_Coach'] = df_with_order['Combined_Coach'].fillna(df_with_order['Coach'])
            else:
                df_with_order['Combined_Coach'] = df_with_order['Coach']
            
            # Return one row per team-year with primary coach and combined coach info
            result_df = df_with_order.groupby(['Team', 'Year']).agg({
                'Primary_Coach': 'first',
                'Combined_Coach': 'first',
                'Total_Coaches': 'first'
            }).reset_index()
            
            return result_df
        else:
            # No duplicates, simple case
            df['Primary_Coach'] = df['Coach']
            df['Combined_Coach'] = df['Coach']
            df['Total_Coaches'] = 1
            return df[['Team', 'Year', 'Primary_Coach', 'Combined_Coach', 'Total_Coaches']]
    
    def save_head_coach_data(self, df: pd.DataFrame) -> bool:
        """
        Save head coach data to CSV file
        
        Args:
            df: DataFrame with head coach data
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            output_file = self.output_dir / "team_year_head_coaches.csv"
            df.to_csv(output_file, index=False)
            self.logger.info(f"Saved head coach data to {output_file}")
            
            # Create summary statistics
            total_team_years = len(df)
            unique_coaches = df['Primary_Coach'].nunique()
            unique_teams = df['Team'].nunique()
            year_range = f"{df['Year'].min()}-{df['Year'].max()}"
            multiple_coaches = df[df['Total_Coaches'] > 1]
            
            # Save metadata
            metadata = {
                'Creation_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Total_Team_Years': total_team_years,
                'Unique_Coaches': unique_coaches,
                'Unique_Teams': unique_teams,
                'Year_Range': year_range,
                'Team_Years_Multiple_Coaches': len(multiple_coaches),
                'Description': 'Team-year to head coach mapping extracted from coaching ranks data'
            }
            
            metadata_df = pd.DataFrame([metadata])
            metadata_file = self.output_dir / "team_year_head_coaches_metadata.csv"
            metadata_df.to_csv(metadata_file, index=False)
            self.logger.info(f"Saved metadata to {metadata_file}")
            
            # Log summary
            self.logger.info(f"Summary:")
            self.logger.info(f"  - Total team-years: {total_team_years}")
            self.logger.info(f"  - Unique coaches: {unique_coaches}")
            self.logger.info(f"  - Unique teams: {unique_teams}")
            self.logger.info(f"  - Year range: {year_range}")
            self.logger.info(f"  - Team-years with multiple coaches: {len(multiple_coaches)}")
            
            if len(multiple_coaches) > 0:
                self.logger.info("Team-years with multiple coaches:")
                for _, row in multiple_coaches.head(10).iterrows():
                    self.logger.info(f"  {row['Team']} {row['Year']}: {row['Combined_Coach']}")
                if len(multiple_coaches) > 10:
                    self.logger.info(f"  ... and {len(multiple_coaches) - 10} more")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving head coach data: {e}")
            return False


def main():
    """Main function to extract head coach data"""
    parser = argparse.ArgumentParser(
        description='Extract head coach information from coaching data'
    )
    parser.add_argument(
        '--coaches-dir',
        type=str,
        default='data/raw/Coaches',
        help='Directory containing coach data (default: data/raw/Coaches)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/Coaching',
        help='Output directory for processed data (default: data/processed/Coaching)'
    )
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = HeadCoachExtractor(
        coaches_dir=args.coaches_dir,
        output_dir=args.output_dir
    )
    
    print("Extracting head coach information...")
    print(f"Source directory: {args.coaches_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Extract head coach data
    hc_df = extractor.extract_all_head_coaches()
    
    if hc_df is None or hc_df.empty:
        print("Error: No head coach data extracted")
        sys.exit(1)
    
    # Handle duplicate coaches
    final_df = extractor.handle_duplicate_coaches(hc_df)
    
    # Save the data
    if extractor.save_head_coach_data(final_df):
        print(f"\nSuccessfully extracted head coach data!")
        print(f"Results saved to: {args.output_dir}")
        print("Generated files:")
        print("  - team_year_head_coaches.csv (main data)")
        print("  - team_year_head_coaches_metadata.csv (metadata)")
    else:
        print("Error: Failed to save head coach data")
        sys.exit(1)


if __name__ == "__main__":
    main()