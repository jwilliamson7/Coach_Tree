#!/usr/bin/env python3
"""
Calculate Coaching Aggression Gene by Coach-Year

This script calculates an "aggression" gene for NFL coaches based on their tendency
to make aggressive play-calling decisions relative to model predictions. The aggression
score combines four dimensions:

1. 4th Down Aggression: Going for it on 4th down more than predicted
2. Pass-Heavy Aggression: Passing more than predicted in run/pass situations  
3. Deep Pass Aggression: Targeting beyond the sticks more than predicted
4. Two-Point Aggression: Attempting two-point conversions more than predicted

For each coach-year, we calculate the difference between actual and predicted rates
for these four decision types to create a composite aggression profile. This enables
temporal analysis of how coaching genes evolve over time.

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

# Add project root to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_features import (
    get_fourth_down_predictor_features,
    get_run_pass_predictor_features, 
    get_pass_target_predictor_features,
    get_two_point_predictor_features,
    get_categorical_features
)
from utils import model_pipeline as mp

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
        self.two_point_model = None
        
        # Encoder containers
        self.fourth_down_encoders = None
        self.run_pass_encoders = None
        self.pass_target_encoders = None
        self.two_point_encoders = None

        # Imputer containers (SimpleImputer + TruncatedSVD persisted at train time)
        self.fourth_down_imputers = None
        self.run_pass_imputers = None
        self.pass_target_imputers = None
        self.two_point_imputers = None

        # Feature lists
        self.fourth_down_features = get_fourth_down_predictor_features()
        self.run_pass_features = get_run_pass_predictor_features()
        self.pass_target_features = get_pass_target_predictor_features()
        self.two_point_features = get_two_point_predictor_features()
        self.categorical_features = get_categorical_features()
        
        # Coach mapping
        self.coach_mapping = None
        
    def load_models(self) -> None:
        """Load trained models with their encoders and imputers via the shared
        pipeline. Feature lists are taken from each model's metadata so gene-time
        features match exactly what the model was trained on (train/serve)."""
        logger.info("Loading trained models...")
        specs = {
            'fourth_down': ('fourth_down/fourth_down_decision_model', xgb.XGBClassifier),
            'run_pass': ('run_pass/run_pass_prediction_model', xgb.XGBClassifier),
            'pass_target': ('pass_target/pass_target_prediction_model', xgb.XGBClassifier),
            'two_point': ('two_point/two_point_conversion_model', xgb.XGBClassifier),
        }
        for name, (relstem, model_cls) in specs.items():
            stem = str(self.models_dir / relstem)
            if not Path(f"{stem}.json").exists():
                logger.warning(f"{name} model not found at {stem}.json")
                continue
            bundle = mp.load_inference_bundle(stem, model_cls)
            setattr(self, f"{name}_model", bundle['model'])
            setattr(self, f"{name}_encoders", bundle['encoders'])
            setattr(self, f"{name}_imputers", bundle['imputers'])
            if bundle['feature_names']:
                setattr(self, f"{name}_features", bundle['feature_names'])
            if bundle['imputers'] is None:
                logger.warning(f"{name}: no imputer artifact found; using fallback median "
                               f"impute (retrain the model to restore train/serve consistency)")
            logger.info(f"Loaded {name} model")
    
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
            'ARI': 'CRD',     # Arizona Cardinals (uses CRD in coach data)
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
                    self.two_point_features +
                    ['play_type', 'punt_attempt', 'field_goal_attempt', 'air_yards', 
                     'qb_scramble',  # Need this for proper pass classification
                     'extra_point_attempt', 'two_point_attempt',  # Need for conversion analysis
                     'posteam', 'season',  # Need these for coach mapping
                     'home_team', 'away_team', 'home_coach', 'away_coach',  # Need actual coach names
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
        
        # Map posteam to the actual head coach for that play
        # Use home_coach/away_coach fields which correctly handle interim coaches
        def get_head_coach(row):
            if pd.isna(row['posteam']):
                return np.nan
            
            # Check if we have coach data in the play-by-play
            # This gives us exact coach attribution for each play, handling interim coaches
            if all(col in row.index for col in ['home_coach', 'away_coach', 'home_team', 'away_team']):
                if pd.notna(row['home_coach']) and row['posteam'] == row['home_team']:
                    return row['home_coach']
                elif pd.notna(row['away_coach']) and row['posteam'] == row['away_team']:
                    return row['away_coach']
            
            # Fallback to team-year mapping if coach fields not available
            # (for older data or missing fields)
            if pd.notna(row.get('season')):
                return self.coach_dict.get(
                    (self.normalize_team_abbr(row['posteam']), int(row['season'])), np.nan
                )
            
            return np.nan
        
        combined['head_coach'] = combined.apply(get_head_coach, axis=1)
        
        # Log how many plays we could map to coaches
        mapped_count = combined['head_coach'].notna().sum()
        total_with_posteam = combined['posteam'].notna().sum()
        logger.info(f"Total plays loaded: {len(combined):,}")
        logger.info(f"Mapped {mapped_count:,}/{total_with_posteam:,} plays to coaches ({mapped_count/total_with_posteam*100:.1f}%)")
        
        return combined
    
    def prepare_features_for_model(self, df: pd.DataFrame, features: List[str],
                                  encoders: Dict, imputers: Dict = None) -> np.ndarray:
        """Transform-only feature prep via the shared pipeline.

        Encoders and the imputer (SimpleImputer + TruncatedSVD) were fit at train
        time and are reused here, so gene-time features land in the exact same
        space the model was trained on. If `imputers` is None (model predates the
        persistence fix) a fallback median impute is used and a warning logged.
        """
        if imputers is None:
            logger.warning("No persisted imputer for this model; using fallback median "
                           "impute. Retrain the model to restore train/serve consistency.")
        return mp.prepare_features_for_inference(
            df, features, self.categorical_features, encoders, imputers)

    def calculate_fourth_down_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate 4th down aggression for each coach-year.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach-year and their 4th down aggression score
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
            self.fourth_down_encoders,
            self.fourth_down_imputers
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
        
        # Group by coach and season for coach-year analysis
        coach_aggression = fourth_downs.groupby(['head_coach', 'season']).agg({
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
        Calculate pass-heavy aggression for each coach-year.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach-year and their pass-heavy aggression score
        """
        logger.info("Calculating pass-heavy aggression...")
        
        # Filter for run/pass decision plays (match model training data)
        # Include downs 1-3 and non-special teams 4th downs
        run_pass_plays = plays[
            (plays['play_type'].isin(['run', 'pass'])) &
            (
                (plays['down'].isin([1, 2, 3])) |  # All 1st-3rd downs
                ((plays['down'] == 4) & 
                 (plays.get('punt_attempt', 0) != 1) &      # Not punts
                 (plays.get('field_goal_attempt', 0) != 1)  # Not field goals
                )
            )
        ].copy()
        
        # Process no_play plays to identify runs/passes from penalties
        # Apply same down/situation filter as above
        if 'desc' in plays.columns:
            no_play_plays = plays[
                (plays['play_type'] == 'no_play') &
                (
                    (plays['down'].isin([1, 2, 3])) |  # All 1st-3rd downs
                    ((plays['down'] == 4) & 
                     (plays.get('punt_attempt', 0) != 1) &      # Not punts
                     (plays.get('field_goal_attempt', 0) != 1)  # Not field goals
                    )
                )
            ].copy()
            
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
            self.run_pass_encoders,
            self.run_pass_imputers
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
        
        # Group by coach and season for coach-year analysis
        coach_aggression = run_pass_plays.groupby(['head_coach', 'season']).agg({
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
        Calculate deep pass aggression for each coach-year.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach-year and their deep pass aggression score
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
            self.pass_target_encoders,
            self.pass_target_imputers
        )
        
        # Generate predictions (probability of targeting beyond sticks)
        predictions = self.pass_target_model.predict_proba(features)[:, 1]
        pass_plays['predicted_beyond_rate'] = predictions
        
        # Actual decisions (1 = beyond sticks, 0 = at/behind)
        pass_plays['actual_beyond'] = (
            pass_plays['air_yards'] > pass_plays['ydstogo']
        ).astype(int)
        
        # Group by coach and season for coach-year analysis
        coach_aggression = pass_plays.groupby(['head_coach', 'season']).agg({
            'actual_beyond': 'mean',  # Actual beyond-sticks rate
            'predicted_beyond_rate': 'mean',  # Expected beyond-sticks rate
            'play_id': 'count'  # Number of pass plays
        }).rename(columns={'play_id': 'pass_plays'})
        
        # Calculate aggression score (actual - predicted)
        coach_aggression['deep_pass_aggression'] = (
            coach_aggression['actual_beyond'] - coach_aggression['predicted_beyond_rate']
        )
        
        return coach_aggression.reset_index()
    
    def calculate_two_point_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate two-point conversion aggression for each coach-year.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach-year and their two-point aggression score
        """
        logger.info("Calculating two-point conversion aggression...")
        
        # Filter for conversion attempts (both extra point and two-point)
        conversion_plays = plays[
            (plays['extra_point_attempt'] == 1) |
            (plays['two_point_attempt'] == 1)
        ].copy()
        
        if len(conversion_plays) == 0:
            logger.warning("No conversion attempts found")
            return pd.DataFrame()
        
        logger.info(f"Found {len(conversion_plays):,} conversion attempts")
        
        # Prepare features for prediction
        features = self.prepare_features_for_model(
            conversion_plays,
            self.two_point_features,
            self.two_point_encoders,
            self.two_point_imputers
        )
        
        # Generate predictions (probability of attempting two-point conversion)
        predictions = self.two_point_model.predict_proba(features)[:, 1]
        conversion_plays['predicted_two_point_rate'] = predictions
        
        # Actual decisions (1 = two-point attempt, 0 = extra point)
        conversion_plays['actual_two_point'] = (
            conversion_plays['two_point_attempt'] == 1
        ).astype(int)
        
        # Group by coach and season for coach-year analysis
        coach_aggression = conversion_plays.groupby(['head_coach', 'season']).agg({
            'actual_two_point': 'mean',  # Actual two-point rate
            'predicted_two_point_rate': 'mean',  # Expected two-point rate
            'play_id': 'count'  # Number of conversion attempts
        }).rename(columns={'play_id': 'conversion_attempts'})
        
        # Calculate aggression score (actual - predicted)
        coach_aggression['two_point_aggression'] = (
            coach_aggression['actual_two_point'] - coach_aggression['predicted_two_point_rate']
        )
        
        return coach_aggression.reset_index()
    
    def calculate_composite_aggression(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite aggression scores for all coach-years.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with all aggression metrics for each coach-year
        """
        logger.info("Calculating composite aggression scores...")
        
        # Calculate individual aggression components
        fourth_down_agg = self.calculate_fourth_down_aggression(plays)
        pass_heavy_agg = self.calculate_pass_heavy_aggression(plays)
        deep_pass_agg = self.calculate_deep_pass_aggression(plays)
        two_point_agg = self.calculate_two_point_aggression(plays)
        
        # Merge all components on coach and season
        aggression_df = fourth_down_agg.reset_index()
        
        if not pass_heavy_agg.empty:
            pass_heavy_reset = pass_heavy_agg.reset_index()
            aggression_df = aggression_df.merge(
                pass_heavy_reset,
                on=['head_coach', 'season'],
                how='outer'
            )
        
        if not deep_pass_agg.empty:
            deep_pass_reset = deep_pass_agg.reset_index()
            aggression_df = aggression_df.merge(
                deep_pass_reset,
                on=['head_coach', 'season'],
                how='outer'
            )
        
        if not two_point_agg.empty:
            two_point_reset = two_point_agg.reset_index()
            aggression_df = aggression_df.merge(
                two_point_reset,
                on=['head_coach', 'season'],
                how='outer'
            )
        
        # Calculate composite aggression score (average of four components)
        aggression_components = []
        if 'fourth_down_aggression' in aggression_df.columns:
            aggression_components.append('fourth_down_aggression')
        if 'pass_heavy_aggression' in aggression_df.columns:
            aggression_components.append('pass_heavy_aggression')
        if 'deep_pass_aggression' in aggression_df.columns:
            aggression_components.append('deep_pass_aggression')
        if 'two_point_aggression' in aggression_df.columns:
            aggression_components.append('two_point_aggression')
        
        if aggression_components:
            aggression_df['composite_aggression'] = aggression_df[aggression_components].mean(axis=1)
            
            # Standardize scores (z-scores) for better interpretation
            for col in aggression_components + ['composite_aggression']:
                mean_val = aggression_df[col].mean()
                std_val = aggression_df[col].std()
                aggression_df[f'{col}_zscore'] = (aggression_df[col] - mean_val) / std_val
        
        # Add total plays coached
        aggression_df['total_plays'] = aggression_df[
            [col for col in ['fourth_down_plays', 'run_pass_plays', 'pass_plays', 'conversion_attempts'] 
             if col in aggression_df.columns]
        ].sum(axis=1)
        
        # Sort by season, then composite aggression
        if 'composite_aggression' in aggression_df.columns:
            aggression_df = aggression_df.sort_values(['season', 'composite_aggression'], ascending=[True, False])
        else:
            aggression_df = aggression_df.sort_values(['season', 'head_coach'])
        
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
        csv_path = output_path / "aggression_gene_by_year.csv"
        aggression_df.to_csv(csv_path, index=False)
        logger.info(f"Saved coach-year aggression data to {csv_path}")
        
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
            'most_aggressive_coach_years': [(row['head_coach'], int(row['season'])) 
                                           for _, row in aggression_df.head(10).iterrows()]
                                          if all(col in aggression_df.columns for col in ['head_coach', 'season']) else [],
            'least_aggressive_coach_years': [(row['head_coach'], int(row['season'])) 
                                            for _, row in aggression_df.tail(10).iterrows()]
                                           if all(col in aggression_df.columns for col in ['head_coach', 'season']) else [],
            'unique_coaches': int(aggression_df['head_coach'].nunique()) if 'head_coach' in aggression_df.columns else 0,
            'years_covered': f"{int(aggression_df['season'].min())}-{int(aggression_df['season'].max())}" if 'season' in aggression_df.columns else None
        }
        
        json_path = output_path / "aggression_gene_by_year_summary.json"
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
                   calculator.pass_target_model, calculator.two_point_model]):
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
        print(f"Total coach-years analyzed: {len(aggression_df)}")
        print(f"Unique coaches: {aggression_df['head_coach'].nunique()}")
        print(f"Years covered: {int(aggression_df['season'].min())} - {int(aggression_df['season'].max())}")
        
        if 'composite_aggression' in aggression_df.columns:
            print("\nMost Aggressive Coach-Years (Composite Score):")
            for i, row in aggression_df.head(10).iterrows():
                print(f"  {row['head_coach']} ({int(row['season'])}): {row['composite_aggression']:.3f}")
            
            print("\nLeast Aggressive Coach-Years (Composite Score):")
            for i, row in aggression_df.tail(10).iterrows():
                print(f"  {row['head_coach']} ({int(row['season'])}): {row['composite_aggression']:.3f}")
        
        print("\nAggression Components (mean ± std):")
        for component in ['fourth_down_aggression', 'pass_heavy_aggression', 'deep_pass_aggression', 'two_point_aggression']:
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