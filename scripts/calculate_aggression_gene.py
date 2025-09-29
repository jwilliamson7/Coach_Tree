#!/usr/bin/env python3
"""
Calculate Coaching Aggression Gene

This script calculates an "aggression" gene for NFL coaches based on their tendency
to make aggressive play-calling decisions relative to model predictions. The aggression
score combines three dimensions:

1. 4th Down Aggression: Going for it on 4th down more than predicted
2. Pass-Heavy Aggression: Passing more than predicted in run/pass situations  
3. Deep Pass Aggression: Targeting beyond the sticks more than predicted

For each coach, we calculate the difference between actual and predicted rates
for these three decision types to create a composite aggression profile.

Usage:
    python calculate_aggression_gene.py [--start_year 2006] [--end_year 2024]
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys
import warnings
import pickle
import json
import xgboost as xgb
from datetime import datetime
warnings.filterwarnings('ignore')

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from utils.model_features import (
    get_fourth_down_predictor_features,
    get_run_pass_predictor_features, 
    get_pass_target_predictor_features,
    get_categorical_features
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressionCalculator:
    """Calculate aggression scores for NFL coaches based on play-calling patterns"""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data/raw/play_by_play", 
                 coaching_dir: str = "data/processed/Coaching"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.coaching_dir = Path(coaching_dir)
        
        # Model containers
        self.fourth_down_model = None
        self.run_pass_model = None
        self.pass_target_model = None
        
        # Encoder containers
        self.fourth_down_encoders = None
        self.run_pass_encoders = None
        self.pass_target_encoders = None
        
        # Feature lists
        self.fourth_down_features = get_fourth_down_predictor_features()
        self.run_pass_features = get_run_pass_predictor_features()
        self.pass_target_features = get_pass_target_predictor_features()
        self.categorical_features = get_categorical_features()
        
        # Coach mapping
        self.coach_mapping = None
        
    def load_models(self) -> None:
        """Load all three trained XGBoost models and their encoders"""
        logger.info("Loading trained models...")
        
        # Load 4th down model
        fourth_down_path = self.models_dir / "fourth_down" / "fourth_down_decision_model.json"
        if fourth_down_path.exists():
            self.fourth_down_model = xgb.XGBClassifier()
            self.fourth_down_model.load_model(str(fourth_down_path))
            
            # Load encoders
            encoders_path = self.models_dir / "fourth_down" / "fourth_down_decision_model_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.fourth_down_encoders = pickle.load(f)
            logger.info("✓ Loaded 4th down decision model")
        else:
            logger.warning("4th down model not found")
            
        # Load run/pass model
        run_pass_path = self.models_dir / "run_pass" / "run_pass_prediction_model.json"
        if run_pass_path.exists():
            self.run_pass_model = xgb.XGBClassifier()
            self.run_pass_model.load_model(str(run_pass_path))
            
            # Load encoders
            encoders_path = self.models_dir / "run_pass" / "run_pass_prediction_model_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.run_pass_encoders = pickle.load(f)
            logger.info("✓ Loaded run/pass prediction model")
        else:
            logger.warning("Run/pass model not found")
            
        # Load pass target model
        pass_target_path = self.models_dir / "pass_target" / "pass_target_prediction_model.json"
        if pass_target_path.exists():
            self.pass_target_model = xgb.XGBClassifier()
            self.pass_target_model.load_model(str(pass_target_path))
            
            # Load encoders
            encoders_path = self.models_dir / "pass_target" / "pass_target_prediction_model_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.pass_target_encoders = pickle.load(f)
            logger.info("✓ Loaded pass target prediction model")
        else:
            logger.warning("Pass target model not found")
    
    def load_coach_mappings(self) -> None:
        """Load the mapping of team-year to head coaches"""
        logger.info("Loading coach mappings...")
        
        coach_file = self.coaching_dir / "team_year_head_coaches.csv"
        if coach_file.exists():
            self.coach_mapping = pd.read_csv(coach_file)
            # Create a dictionary for fast lookup: (team, year) -> coach
            self.coach_dict = {}
            for _, row in self.coach_mapping.iterrows():
                # Use Primary_Coach as the main coach
                self.coach_dict[(row['Team'], int(row['Year']))] = row['Primary_Coach']
            logger.info(f"✓ Loaded coach mappings for {len(self.coach_dict)} team-years")
        else:
            logger.warning(f"Coach mapping file not found: {coach_file}")
            self.coach_dict = {}
    
    def normalize_team_abbr(self, pbp_team: str) -> str:
        """
        Normalize team abbreviations from play-by-play data to match coach mapping format.
        
        Args:
            pbp_team: Team abbreviation from play-by-play data
            
        Returns:
            Normalized team abbreviation matching coach mapping format
        """
        # Mapping from play-by-play format to coach mapping format
        team_mapping = {
            # Current teams with different abbreviations
            'GB': 'GNB',      # Green Bay Packers
            'KC': 'KAN',      # Kansas City Chiefs
            'LA': 'LAR',      # Los Angeles Rams
            'LV': 'LVR',      # Las Vegas Raiders
            'NO': 'NOR',      # New Orleans Saints
            'NE': 'NWE',      # New England Patriots
            'TEN': 'OTI',     # Tennessee Titans (was OTI for Oilers/Titans)
            'ARI': 'PHO',     # Arizona Cardinals (Phoenix/Arizona)
            'LAC': 'SDG',     # Los Angeles Chargers (was San Diego)
            'SF': 'SFO',      # San Francisco 49ers
            'TB': 'TAM',      # Tampa Bay Buccaneers
            'IND': 'CLT',     # Indianapolis Colts (was Baltimore Colts CLT)
            'BAL': 'RAV',     # Baltimore Ravens
            'HOU': 'HTX',     # Houston Texans
            
            # Historical teams (for older data if needed)
            'OAK': 'RAI',     # Oakland Raiders (before Las Vegas)
            'SD': 'SDG',      # San Diego Chargers (before LA)
            'STL': 'STL',     # St. Louis Rams (before LA)
            
            # Teams that stay the same
            'ATL': 'ATL', 'BUF': 'BUF', 'CAR': 'CAR', 'CHI': 'CHI',
            'CIN': 'CIN', 'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN',
            'DET': 'DET', 'JAX': 'JAX', 'MIA': 'MIA', 'MIN': 'MIN',
            'NYG': 'NYG', 'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT',
            'SEA': 'SEA', 'WAS': 'WAS'
        }
        
        return team_mapping.get(pbp_team, pbp_team)
    
    def load_play_data(self, start_year: int = 2006, end_year: int = 2024) -> pd.DataFrame:
        """
        Load play-by-play data for the specified years with coach information.
        
        Args:
            start_year: First year to include (default 2006 for air_yards)
            end_year: Last year to include
            
        Returns:
            DataFrame with play-by-play data including coach columns
        """
        logger.info(f"Loading play-by-play data from {start_year} to {end_year}...")
        
        all_plays = []
        
        for year in range(start_year, end_year + 1):
            file_path = self.data_dir / f"play_by_play_{year}.csv"
            
            if not file_path.exists():
                logger.warning(f"No data found for {year}")
                continue
                
            try:
                # Read the CSV
                df = pd.read_csv(file_path)
                
                # Keep only necessary columns to reduce memory
                needed_cols = list(set(
                    self.fourth_down_features + 
                    self.run_pass_features + 
                    self.pass_target_features +
                    ['play_type', 'punt_attempt', 'field_goal_attempt', 'air_yards', 
                     'qb_scramble',  # Need this for proper pass classification
                     'posteam', 'season',  # Need these for coach mapping
                     'desc',  # Need for analyzing no_play situations
                     'game_id', 'play_id']
                ))
                
                # Keep only columns that exist
                keep_cols = [col for col in needed_cols if col in df.columns]
                df = df[keep_cols]
                
                all_plays.append(df)
                logger.info(f"Loaded {len(df):,} plays from {year}")
                
            except Exception as e:
                logger.error(f"Error loading {year}: {e}")
                continue
        
        if not all_plays:
            raise ValueError("No play data loaded")
            
        combined = pd.concat(all_plays, ignore_index=True)
        
        # Map posteam and season to head coach using our mapping
        # First normalize team abbreviations to match coach mapping format
        combined['offensive_coach'] = combined.apply(
            lambda row: self.coach_dict.get(
                (self.normalize_team_abbr(row['posteam']), int(row['season'])), np.nan
            ) if pd.notna(row['posteam']) and pd.notna(row['season'])
              else np.nan,
            axis=1
        )
        
        # Log how many plays we could map to coaches
        mapped_count = combined['offensive_coach'].notna().sum()
        total_with_posteam = combined['posteam'].notna().sum()
        logger.info(f"Total plays loaded: {len(combined):,}")
        logger.info(f"Mapped {mapped_count:,}/{total_with_posteam:,} plays to coaches ({mapped_count/total_with_posteam*100:.1f}%)")
        
        return combined
    
    def prepare_features_for_model(self, df: pd.DataFrame, features: List[str], 
                                  encoders: Dict) -> np.ndarray:
        """
        Prepare features for model prediction, handling missing values and encoding.
        
        Args:
            df: DataFrame with raw features
            features: List of feature names needed by model
            encoders: Dictionary of label encoders for categorical features
            
        Returns:
            Numpy array ready for model prediction
        """
        # Select features
        df_features = df[features].copy()
        
        # Encode categorical features
        if encoders:
            for col in features:
                if col in encoders and col in self.categorical_features:
                    le = encoders[col]
                    # Handle missing/unseen categories
                    df_features[col] = df_features[col].astype(str)
                    mask = df_features[col].isin(le.classes_)
                    df_features.loc[~mask, col] = 'nan'  # Use 'nan' as unknown category
                    
                    # Add 'nan' to encoder classes if not present
                    if 'nan' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'nan')
                    
                    df_features[col] = le.transform(df_features[col])
        
        # Handle missing values with median imputation
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        features_imputed = imputer.fit_transform(df_features)
        
        return features_imputed
    
    def calculate_fourth_down_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 4th down aggression for each coach.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach and their 4th down aggression score
        """
        logger.info("Calculating 4th down aggression...")
        
        # Filter for 4th down plays
        fourth_downs = plays[plays['down'] == 4].copy()
        
        # Remove kickoffs, extra points, and QB kneels (not decision plays)
        fourth_downs = fourth_downs[
            ~fourth_downs['play_type'].isin(['kickoff', 'extra_point', 'qb_kneel'])
        ]
        
        if len(fourth_downs) == 0:
            logger.warning("No 4th down plays found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(fourth_downs):,} 4th down decision plays")
        
        # Prepare features for prediction
        features = self.prepare_features_for_model(
            fourth_downs, 
            self.fourth_down_features,
            self.fourth_down_encoders
        )
        
        # Generate predictions (probability of going for it)
        predictions = self.fourth_down_model.predict_proba(features)[:, 1]
        fourth_downs['predicted_go_rate'] = predictions
        
        # Actual decisions - match the model's target creation exactly
        # Initialize as go-for-it (1)
        fourth_downs['actual_decision'] = 1
        
        # Set to not go-for-it (0) for punts and field goals
        if 'punt_attempt' in fourth_downs.columns and 'field_goal_attempt' in fourth_downs.columns:
            punt_fg_conditions = (
                (fourth_downs['punt_attempt'] == 1) |
                (fourth_downs['field_goal_attempt'] == 1)
            )
            fourth_downs.loc[punt_fg_conditions, 'actual_decision'] = 0
        else:
            # Fallback to play_type if attempt columns not available
            fourth_downs['actual_decision'] = (
                ~fourth_downs['play_type'].isin(['punt', 'field_goal'])
            ).astype(int)
        
        # For no_play situations, analyze the description to determine the intent
        if 'desc' in fourth_downs.columns:
            no_play_mask = fourth_downs['play_type'] == 'no_play'
            if no_play_mask.any():
                # Get the indices of no_play rows
                no_play_indices = fourth_downs[no_play_mask].index
                
                # Check descriptions for punt/FG attempts that were nullified
                desc_lower = fourth_downs.loc[no_play_indices, 'desc'].str.lower()
                
                # Identify punt attempts in no_play situations
                punt_patterns = ['punt', 'punts']
                is_punt_attempt = desc_lower.str.contains('|'.join(punt_patterns), na=False)
                
                # Identify field goal attempts in no_play situations  
                fg_patterns = ['field goal', 'fg ']
                is_fg_attempt = desc_lower.str.contains('|'.join(fg_patterns), na=False)
                
                # Update actual_decision for no_play situations using the indices
                punt_indices = no_play_indices[is_punt_attempt]
                fg_indices = no_play_indices[is_fg_attempt]
                
                fourth_downs.loc[punt_indices, 'actual_decision'] = 0
                fourth_downs.loc[fg_indices, 'actual_decision'] = 0
                
                logger.info(f"Processed {no_play_mask.sum()} no_play situations using play descriptions")
        
        # Group by coach and calculate aggression
        coach_aggression = fourth_downs.groupby('offensive_coach').agg({
            'actual_decision': 'mean',  # Actual go-for-it rate
            'predicted_go_rate': 'mean',  # Expected go-for-it rate
            'play_id': 'count'  # Number of 4th down decisions
        }).rename(columns={'play_id': 'fourth_down_plays'})
        
        # Calculate aggression score (actual - predicted)
        coach_aggression['fourth_down_aggression'] = (
            coach_aggression['actual_decision'] - coach_aggression['predicted_go_rate']
        )
        
        return coach_aggression.reset_index()
    
    def calculate_pass_heavy_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pass-heavy aggression for each coach.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach and their pass-heavy aggression score
        """
        logger.info("Calculating pass-heavy aggression...")
        
        # Start with clear run/pass plays
        run_pass_plays = plays[
            plays['play_type'].isin(['run', 'pass'])
        ].copy()
        
        # Process no_play plays to identify runs/passes from penalties
        if 'desc' in plays.columns:
            no_play_plays = plays[plays['play_type'] == 'no_play'].copy()
            
            if len(no_play_plays) > 0:
                logger.info(f"Processing {len(no_play_plays):,} no_play plays for run/pass classification...")
                
                # Define pass play keywords - be specific to avoid false positives
                pass_keywords = [
                    ' pass ', ' passes ', ' passed ', ' passing ',
                    'incomplete', 'complete pass', 'sacked', ' sack ',
                    'scrambles', 'scrambled', 'throw', 'throws',
                    'intercepted', 'interception'
                ]
                
                # Define run play keywords
                run_keywords = [
                    ' run ', ' runs ', ' ran ', ' rush ', ' rushes ', ' rushed ',
                    'up the middle', 'left end', 'right end', 
                    'left tackle', 'right tackle', 'left guard', 'right guard'
                ]
                
                # Convert descriptions to lowercase for case-insensitive matching
                desc_lower = no_play_plays['desc'].str.lower()
                
                # Check for pass keywords
                pass_pattern = '|'.join(pass_keywords)
                pass_matches = desc_lower.str.contains(pass_pattern, na=False, regex=True)
                
                # Check for run keywords  
                run_pattern = '|'.join(run_keywords)
                run_matches = desc_lower.str.contains(run_pattern, na=False, regex=True)
                
                # Create dataframes for classified plays
                classified_passes = no_play_plays[pass_matches].copy()
                classified_passes['play_type'] = 'pass'
                
                classified_runs = no_play_plays[run_matches & ~pass_matches].copy()
                classified_runs['play_type'] = 'run'
                
                # Add classified plays to run_pass_plays
                run_pass_plays = pd.concat([
                    run_pass_plays,
                    classified_passes,
                    classified_runs
                ], ignore_index=True)
                
                classified_count = len(classified_passes) + len(classified_runs)
                logger.info(f"  Classified {classified_count:,} no_play plays ({classified_count/len(no_play_plays)*100:.1f}%)")
                logger.info(f"    - {len(classified_passes):,} as pass plays")
                logger.info(f"    - {len(classified_runs):,} as run plays")
        
        if len(run_pass_plays) == 0:
            logger.warning("No run/pass plays found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(run_pass_plays):,} run/pass plays")
        
        # Prepare features for prediction
        features = self.prepare_features_for_model(
            run_pass_plays,
            self.run_pass_features,
            self.run_pass_encoders
        )
        
        # Generate predictions (probability of pass)
        predictions = self.run_pass_model.predict_proba(features)[:, 1]
        run_pass_plays['predicted_pass_rate'] = predictions
        
        # Actual decisions - match the model's target creation exactly
        # Pass plays include play_type="pass" OR qb_scramble=1
        run_pass_plays['actual_pass'] = (
            (run_pass_plays['play_type'] == 'pass') | 
            (run_pass_plays.get('qb_scramble', 0) == 1)
        ).astype(int)
        
        # Group by coach and calculate aggression
        coach_aggression = run_pass_plays.groupby('offensive_coach').agg({
            'actual_pass': 'mean',  # Actual pass rate
            'predicted_pass_rate': 'mean',  # Expected pass rate
            'play_id': 'count'  # Number of plays
        }).rename(columns={'play_id': 'run_pass_plays'})
        
        # Calculate aggression score (actual - predicted)
        coach_aggression['pass_heavy_aggression'] = (
            coach_aggression['actual_pass'] - coach_aggression['predicted_pass_rate']
        )
        
        return coach_aggression.reset_index()
    
    def calculate_deep_pass_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate deep pass aggression for each coach.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach and their deep pass aggression score
        """
        logger.info("Calculating deep pass aggression...")
        
        # Filter for pass plays with air_yards data
        pass_plays = plays[
            (plays['play_type'] == 'pass') &
            (plays['air_yards'].notna()) &
            (plays['ydstogo'].notna())
        ].copy()
        
        if len(pass_plays) == 0:
            logger.warning("No pass plays with air_yards found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(pass_plays):,} pass plays with air_yards")
        
        # Prepare features for prediction
        features = self.prepare_features_for_model(
            pass_plays,
            self.pass_target_features,
            self.pass_target_encoders
        )
        
        # Generate predictions (probability of targeting beyond sticks)
        predictions = self.pass_target_model.predict_proba(features)[:, 1]
        pass_plays['predicted_beyond_rate'] = predictions
        
        # Actual decisions (1 = beyond sticks, 0 = at/behind)
        pass_plays['actual_beyond'] = (
            pass_plays['air_yards'] > pass_plays['ydstogo']
        ).astype(int)
        
        # Group by coach and calculate aggression
        coach_aggression = pass_plays.groupby('offensive_coach').agg({
            'actual_beyond': 'mean',  # Actual beyond-sticks rate
            'predicted_beyond_rate': 'mean',  # Expected beyond-sticks rate
            'play_id': 'count'  # Number of pass plays
        }).rename(columns={'play_id': 'pass_plays'})
        
        # Calculate aggression score (actual - predicted)
        coach_aggression['deep_pass_aggression'] = (
            coach_aggression['actual_beyond'] - coach_aggression['predicted_beyond_rate']
        )
        
        return coach_aggression.reset_index()
    
    def calculate_composite_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite aggression scores for all coaches.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with all aggression metrics for each coach
        """
        logger.info("Calculating composite aggression scores...")
        
        # Calculate individual aggression components
        fourth_down_agg = self.calculate_fourth_down_aggression(plays)
        pass_heavy_agg = self.calculate_pass_heavy_aggression(plays)
        deep_pass_agg = self.calculate_deep_pass_aggression(plays)
        
        # Merge all components
        aggression_df = fourth_down_agg[['offensive_coach', 'fourth_down_aggression', 
                                        'fourth_down_plays', 'actual_decision', 
                                        'predicted_go_rate']]
        
        if not pass_heavy_agg.empty:
            aggression_df = aggression_df.merge(
                pass_heavy_agg[['offensive_coach', 'pass_heavy_aggression', 
                              'run_pass_plays', 'actual_pass', 'predicted_pass_rate']],
                on='offensive_coach',
                how='outer'
            )
        
        if not deep_pass_agg.empty:
            aggression_df = aggression_df.merge(
                deep_pass_agg[['offensive_coach', 'deep_pass_aggression',
                             'pass_plays', 'actual_beyond', 'predicted_beyond_rate']],
                on='offensive_coach',
                how='outer'
            )
        
        # Calculate composite aggression score (average of three components)
        aggression_components = []
        if 'fourth_down_aggression' in aggression_df.columns:
            aggression_components.append('fourth_down_aggression')
        if 'pass_heavy_aggression' in aggression_df.columns:
            aggression_components.append('pass_heavy_aggression')
        if 'deep_pass_aggression' in aggression_df.columns:
            aggression_components.append('deep_pass_aggression')
        
        if aggression_components:
            aggression_df['composite_aggression'] = aggression_df[aggression_components].mean(axis=1)
            
            # Standardize scores (z-scores) for better interpretation
            for col in aggression_components + ['composite_aggression']:
                mean_val = aggression_df[col].mean()
                std_val = aggression_df[col].std()
                aggression_df[f'{col}_zscore'] = (aggression_df[col] - mean_val) / std_val
        
        # Add total plays coached
        aggression_df['total_plays'] = aggression_df[
            [col for col in ['fourth_down_plays', 'run_pass_plays', 'pass_plays'] 
             if col in aggression_df.columns]
        ].sum(axis=1)
        
        # Sort by composite aggression
        if 'composite_aggression' in aggression_df.columns:
            aggression_df = aggression_df.sort_values('composite_aggression', ascending=False)
        
        return aggression_df
    
    def save_results(self, aggression_df: pd.DataFrame, output_dir: str = "data/processed/coaching_genes"):
        """
        Save aggression gene results to files.
        
        Args:
            aggression_df: DataFrame with aggression scores
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results as CSV
        csv_path = output_path / f"aggression_gene_{datetime.now().strftime('%Y%m%d')}.csv"
        aggression_df.to_csv(csv_path, index=False)
        logger.info(f"Saved aggression data to {csv_path}")
        
        # Save summary as JSON for easy access
        summary = {
            'generated_date': datetime.now().isoformat(),
            'num_coaches': len(aggression_df),
            'metrics': {
                'fourth_down_aggression': {
                    'mean': float(aggression_df['fourth_down_aggression'].mean()) 
                            if 'fourth_down_aggression' in aggression_df.columns else None,
                    'std': float(aggression_df['fourth_down_aggression'].std())
                           if 'fourth_down_aggression' in aggression_df.columns else None,
                },
                'pass_heavy_aggression': {
                    'mean': float(aggression_df['pass_heavy_aggression'].mean())
                            if 'pass_heavy_aggression' in aggression_df.columns else None,
                    'std': float(aggression_df['pass_heavy_aggression'].std())
                           if 'pass_heavy_aggression' in aggression_df.columns else None,
                },
                'deep_pass_aggression': {
                    'mean': float(aggression_df['deep_pass_aggression'].mean())
                            if 'deep_pass_aggression' in aggression_df.columns else None,
                    'std': float(aggression_df['deep_pass_aggression'].std())
                           if 'deep_pass_aggression' in aggression_df.columns else None,
                }
            },
            'most_aggressive_coaches': aggression_df.head(10)['offensive_coach'].tolist()
                                      if 'offensive_coach' in aggression_df.columns else [],
            'least_aggressive_coaches': aggression_df.tail(10)['offensive_coach'].tolist()
                                       if 'offensive_coach' in aggression_df.columns else []
        }
        
        json_path = output_path / f"aggression_gene_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {json_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate coaching aggression genes')
    parser.add_argument('--start_year', type=int, default=2006,
                       help='Start year for analysis (default: 2006 for air_yards)')
    parser.add_argument('--end_year', type=int, default=2024,
                       help='End year for analysis (default: 2024)')
    parser.add_argument('--output_dir', type=str, default='data/processed/coaching_genes',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COACHING AGGRESSION GENE CALCULATOR")
    print("=" * 80)
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")
    
    try:
        # Initialize calculator
        calculator = AggressionCalculator()
        
        # Load models
        logger.info("Step 1: Loading predictive models...")
        calculator.load_models()
        
        # Check if all models loaded
        if not all([calculator.fourth_down_model, calculator.run_pass_model, 
                   calculator.pass_target_model]):
            logger.error("Not all models loaded. Please train all models first.")
            return
        
        # Load coach mappings
        logger.info("Step 2: Loading coach mappings...")
        calculator.load_coach_mappings()
        
        # Load play data
        logger.info("Step 3: Loading play-by-play data...")
        plays = calculator.load_play_data(args.start_year, args.end_year)
        
        # Calculate aggression scores
        logger.info("Step 4: Calculating aggression scores...")
        aggression_df = calculator.calculate_composite_aggression(plays)
        
        # Save results
        logger.info("Step 5: Saving results...")
        calculator.save_results(aggression_df, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 80)
        print("AGGRESSION GENE SUMMARY")
        print("=" * 80)
        print(f"Total coaches analyzed: {len(aggression_df)}")
        
        if 'composite_aggression' in aggression_df.columns:
            print("\nMost Aggressive Coaches (Composite Score):")
            for i, row in aggression_df.head(10).iterrows():
                print(f"  {row['offensive_coach']}: {row['composite_aggression']:.3f}")
            
            print("\nLeast Aggressive Coaches (Composite Score):")
            for i, row in aggression_df.tail(10).iterrows():
                print(f"  {row['offensive_coach']}: {row['composite_aggression']:.3f}")
        
        print("\nAggression Components (mean ± std):")
        for component in ['fourth_down_aggression', 'pass_heavy_aggression', 'deep_pass_aggression']:
            if component in aggression_df.columns:
                mean_val = aggression_df[component].mean()
                std_val = aggression_df[component].std()
                print(f"  {component}: {mean_val:.3f} ± {std_val:.3f}")
        
        print("=" * 80)
        logger.info("Aggression gene calculation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during aggression calculation: {e}")
        raise


if __name__ == "__main__":
    main()