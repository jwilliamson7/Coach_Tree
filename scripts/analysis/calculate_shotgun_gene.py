#!/usr/bin/env python3
"""
Calculate Coaching Shotgun Gene by Coach-Year

This script calculates a "shotgun gene" for NFL head coaches based on their tendency
to use shotgun formation relative to model predictions. The shotgun score measures
whether a coach uses shotgun formation more or less than expected given the 
game context (down, distance, score, time, etc.).

A positive shotgun gene indicates a coach who uses shotgun more than expected,
while a negative gene indicates a more traditional, under-center approach.

For each coach-year, we calculate the difference between actual and predicted 
shotgun usage rates. This enables temporal analysis of how coaching formation 
preferences evolve over time.

Usage:
    python calculate_shotgun_gene.py [--start_year 1999] [--end_year 2024]
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
from utils.model_features import get_shotgun_predictor_features, get_categorical_features
from utils import model_pipeline as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShotgunGeneCalculator:
    """Calculate shotgun formation gene for NFL coaches based on formation patterns"""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data/raw/play_by_play", 
                 coaching_dir: str = "data/processed/Coaching"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.coaching_dir = Path(coaching_dir)
        
        # Model containers
        self.shotgun_model = None
        self.shotgun_encoders = None
        self.shotgun_imputers = None
        
        # Feature lists - use centralized feature definitions
        self.shotgun_features = get_shotgun_predictor_features()
        self.categorical_features = get_categorical_features()
        
        # Coach mapping
        self.coach_mapping = None
        
    def load_model(self) -> None:
        """Load the trained XGBoost shotgun model with its encoders and imputer via
        the shared pipeline. Feature list is taken from the model's metadata so
        gene-time features match exactly what the model was trained on (train/serve)."""
        logger.info("Loading trained shotgun model...")
        specs = {
            'shotgun': ('shotgun/shotgun_prediction_model', xgb.XGBClassifier),
        }
        for name, (relstem, model_cls) in specs.items():
            stem = str(self.models_dir / relstem)
            if not Path(f"{stem}.json").exists():
                raise FileNotFoundError(f"{name} model not found at {stem}.json")
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
    
    def load_play_data(self, start_year: int = 1999, end_year: int = 2024) -> pd.DataFrame:
        """
        Load play-by-play data for the specified years with coach information.
        
        Args:
            start_year: First year to include (default 1999 when shotgun data starts)
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
                
                # Check if shotgun column exists
                if 'shotgun' not in df.columns:
                    logger.warning(f"No 'shotgun' column in {year} data - skipping")
                    continue
                
                # Keep only necessary columns to reduce memory
                needed_cols = list(set(
                    self.shotgun_features + 
                    ['play_type', 'shotgun', 'punt_attempt', 'field_goal_attempt',
                     'down', 'posteam', 'season',  # Need these for filtering and coach mapping
                     'home_team', 'away_team', 'home_coach', 'away_coach',  # Need actual coach names
                     'game_id', 'play_id']
                ))
                
                # Keep only columns that exist
                keep_cols = [col for col in needed_cols if col in df.columns]
                df = df[keep_cols]
                
                # Filter for relevant plays (1st-3rd downs and non-special teams 4th downs)
                df = df[
                    (df['play_type'].isin(['run', 'pass'])) &  # Only offensive plays
                    (
                        (df['down'].isin([1, 2, 3])) |  # All 1st-3rd downs
                        ((df['down'] == 4) & 
                         (df.get('punt_attempt', 0) != 1) &      # Not punts
                         (df.get('field_goal_attempt', 0) != 1)  # Not field goals
                        )
                    ) &
                    (df['shotgun'].notna())  # Must have shotgun data
                ]
                
                all_plays.append(df)
                logger.info(f"Loaded {len(df):,} plays from {year}")
                
            except Exception as e:
                logger.error(f"Error loading {year}: {e}")
                continue
        
        if not all_plays:
            raise ValueError("No play data loaded")
            
        combined = pd.concat(all_plays, ignore_index=True)
        
        # Map posteam to the actual coach for that play
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
    
    def prepare_features_for_model(self, df: pd.DataFrame) -> np.ndarray:
        """Transform-only feature prep via the shared pipeline.

        Encoders and the imputer (SimpleImputer + TruncatedSVD) were fit at train
        time and are reused here, so gene-time features land in the exact same
        space the model was trained on. If the imputer is None (model predates the
        persistence fix) a fallback median impute is used and a warning logged.
        """
        if self.shotgun_imputers is None:
            logger.warning("No persisted imputer for this model; using fallback median "
                           "impute. Retrain the model to restore train/serve consistency.")
        return mp.prepare_features_for_inference(
            df, self.shotgun_features, self.categorical_features,
            self.shotgun_encoders, self.shotgun_imputers)
    
    def calculate_shotgun_gene(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate shotgun formation gene for each coach.
        
        Args:
            plays: DataFrame with play data
            
        Returns:
            DataFrame with coach and their shotgun gene score
        """
        logger.info("Calculating shotgun formation gene...")
        
        if len(plays) == 0:
            logger.warning("No plays found for shotgun analysis")
            return pd.DataFrame()
        
        logger.info(f"Analyzing {len(plays):,} offensive plays with shotgun data")
        
        # Prepare features for prediction
        features = self.prepare_features_for_model(plays)
        
        # Generate predictions (probability of using shotgun)
        predictions = self.shotgun_model.predict_proba(features)[:, 1]
        plays['predicted_shotgun_rate'] = predictions
        
        # Actual shotgun usage
        plays['actual_shotgun'] = plays['shotgun'].astype(int)
        
        # Group by coach and season for coach-year analysis
        coach_shotgun = plays.groupby(['head_coach', 'season']).agg({
            'actual_shotgun': 'mean',  # Actual shotgun rate
            'predicted_shotgun_rate': 'mean',  # Expected shotgun rate
            'play_id': 'count'  # Number of plays
        }).rename(columns={'play_id': 'total_plays'})
        
        # Calculate shotgun gene (actual - predicted)
        coach_shotgun['shotgun_gene'] = (
            coach_shotgun['actual_shotgun'] - coach_shotgun['predicted_shotgun_rate']
        )
        
        # Reset index to make head_coach and season regular columns
        coach_shotgun = coach_shotgun.reset_index()
        
        # Calculate z-score for better interpretation
        mean_gene = coach_shotgun['shotgun_gene'].mean()
        std_gene = coach_shotgun['shotgun_gene'].std()
        coach_shotgun['shotgun_gene_zscore'] = (coach_shotgun['shotgun_gene'] - mean_gene) / std_gene
        
        # Sort by season, then shotgun gene
        coach_shotgun = coach_shotgun.sort_values(['season', 'shotgun_gene'], ascending=[True, False])
        
        return coach_shotgun
    
    def analyze_by_era(self, plays: pd.DataFrame, coach_shotgun: pd.DataFrame) -> Dict:
        """
        Analyze shotgun usage trends by era.
        
        Args:
            plays: DataFrame with play data
            coach_shotgun: DataFrame with coach shotgun genes
            
        Returns:
            Dictionary with era-based analysis
        """
        logger.info("Analyzing shotgun usage by era...")
        
        # Define eras
        eras = {
            'Early (1999-2005)': (1999, 2005),
            'Mid (2006-2012)': (2006, 2012),
            'Modern (2013-2019)': (2013, 2019),
            'Current (2020-2024)': (2020, 2024)
        }
        
        era_analysis = {}
        
        for era_name, (start_year, end_year) in eras.items():
            era_plays = plays[(plays['season'] >= start_year) & (plays['season'] <= end_year)]
            
            if len(era_plays) > 0:
                # Get coaches active in this era
                era_coaches = era_plays.groupby('head_coach').agg({
                    'shotgun': 'mean',
                    'play_id': 'count'
                }).rename(columns={'shotgun': 'era_shotgun_rate', 'play_id': 'era_plays'})
                
                # For era analysis, we'll use the average gene across years for each coach
                avg_coach_genes = coach_shotgun.groupby('head_coach')[['shotgun_gene', 'shotgun_gene_zscore']].mean()
                era_coaches = era_coaches.merge(
                    avg_coach_genes,
                    left_index=True,
                    right_index=True
                ).reset_index()  # Reset index to make head_coach a column
                
                era_analysis[era_name] = {
                    'avg_shotgun_rate': float(era_plays['shotgun'].mean()),
                    'total_plays': len(era_plays),
                    'num_coaches': len(era_coaches),
                    'top_genes': era_coaches.nlargest(5, 'shotgun_gene')[
                        ['head_coach', 'shotgun_gene', 'era_shotgun_rate']
                    ].to_dict('records'),
                    'bottom_genes': era_coaches.nsmallest(5, 'shotgun_gene')[
                        ['head_coach', 'shotgun_gene', 'era_shotgun_rate']
                    ].to_dict('records')
                }
        
        return era_analysis
    
    def save_results(self, coach_shotgun: pd.DataFrame, era_analysis: Dict, 
                    output_dir: str = "data/processed/coaching_genes"):
        """
        Save shotgun gene results to files.
        
        Args:
            coach_shotgun: DataFrame with shotgun genes
            era_analysis: Dictionary with era-based analysis
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results as CSV
        csv_path = output_path / "shotgun_gene.csv"
        coach_shotgun.to_csv(csv_path, index=False)
        logger.info(f"Saved shotgun gene data to {csv_path}")
        
        # Save summary as JSON for easy access
        summary = {
            'generated_date': datetime.now().isoformat(),
            'num_coaches': len(coach_shotgun),
            'metrics': {
                'shotgun_gene': {
                    'mean': float(coach_shotgun['shotgun_gene'].mean()),
                    'std': float(coach_shotgun['shotgun_gene'].std()),
                    'min': float(coach_shotgun['shotgun_gene'].min()),
                    'max': float(coach_shotgun['shotgun_gene'].max())
                },
                'actual_shotgun_rate': {
                    'mean': float(coach_shotgun['actual_shotgun'].mean()),
                    'std': float(coach_shotgun['actual_shotgun'].std()),
                    'min': float(coach_shotgun['actual_shotgun'].min()),
                    'max': float(coach_shotgun['actual_shotgun'].max())
                }
            },
            'most_shotgun_heavy_coach_years': coach_shotgun.head(10)[
                ['head_coach', 'season', 'shotgun_gene', 'actual_shotgun']
            ].to_dict('records'),
            'most_traditional_coach_years': coach_shotgun.tail(10)[
                ['head_coach', 'season', 'shotgun_gene', 'actual_shotgun']
            ].to_dict('records'),
            'unique_coaches': int(coach_shotgun['head_coach'].nunique()),
            'years_covered': f"{int(coach_shotgun['season'].min())}-{int(coach_shotgun['season'].max())}",
            'era_analysis': era_analysis
        }
        
        json_path = output_path / "shotgun_gene_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {json_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate coaching shotgun formation genes')
    parser.add_argument('--start_year', type=int, default=1999,
                       help='Start year for analysis (default: 1999 when shotgun data begins)')
    parser.add_argument('--end_year', type=int, default=2024,
                       help='End year for analysis (default: 2024)')
    parser.add_argument('--output_dir', type=str, default='data/processed/coaching_genes',
                       help='Output directory for results')
    parser.add_argument('--min_plays', type=int, default=100,
                       help='Minimum plays required for a coach to be included (default: 100)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("COACHING SHOTGUN GENE CALCULATOR")
    print("=" * 80)
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum plays threshold: {args.min_plays}")
    print("=" * 80 + "\n")
    
    try:
        # Initialize calculator
        calculator = ShotgunGeneCalculator()
        
        # Load model
        logger.info("Step 1: Loading predictive model...")
        calculator.load_model()
        
        # Load coach mappings
        logger.info("Step 2: Loading coach mappings...")
        calculator.load_coach_mappings()
        
        # Load play data
        logger.info("Step 3: Loading play-by-play data...")
        plays = calculator.load_play_data(args.start_year, args.end_year)
        
        # Calculate shotgun gene
        logger.info("Step 4: Calculating shotgun gene...")
        coach_shotgun = calculator.calculate_shotgun_gene(plays)
        
        # Filter by minimum plays
        coach_shotgun = coach_shotgun[coach_shotgun['total_plays'] >= args.min_plays]
        logger.info(f"Analyzing {len(coach_shotgun)} coaches with at least {args.min_plays} plays")
        
        # Analyze by era
        logger.info("Step 5: Analyzing trends by era...")
        era_analysis = calculator.analyze_by_era(plays, coach_shotgun)
        
        # Save results
        logger.info("Step 6: Saving results...")
        calculator.save_results(coach_shotgun, era_analysis, args.output_dir)
        
        # Print summary
        print("\n" + "=" * 80)
        print("SHOTGUN GENE SUMMARY")
        print("=" * 80)
        print(f"Total coach-years analyzed: {len(coach_shotgun)}")
        print(f"Unique coaches: {coach_shotgun['head_coach'].nunique()}")
        print(f"Years covered: {int(coach_shotgun['season'].min())} - {int(coach_shotgun['season'].max())}")
        
        print("\nMost Shotgun-Heavy Coach-Years (Positive Gene):")
        for i, row in coach_shotgun.head(10).iterrows():
            print(f"  {row['head_coach']:25} ({int(row['season'])}) {row['shotgun_gene']:+.3f} "
                  f"(Actual: {row['actual_shotgun']:.1%}, Expected: {row['predicted_shotgun_rate']:.1%})")
        
        print("\nMost Traditional Coach-Years (Negative Gene):")
        for i, row in coach_shotgun.tail(10).iterrows():
            print(f"  {row['head_coach']:25} ({int(row['season'])}) {row['shotgun_gene']:+.3f} "
                  f"(Actual: {row['actual_shotgun']:.1%}, Expected: {row['predicted_shotgun_rate']:.1%})")
        
        print("\nShotgun Gene Statistics:")
        print(f"  Mean: {coach_shotgun['shotgun_gene'].mean():.3f}")
        print(f"  Std Dev: {coach_shotgun['shotgun_gene'].std():.3f}")
        print(f"  Range: [{coach_shotgun['shotgun_gene'].min():.3f}, {coach_shotgun['shotgun_gene'].max():.3f}]")
        
        print("\nEra Analysis:")
        for era_name, era_data in era_analysis.items():
            print(f"\n{era_name}:")
            print(f"  Average shotgun rate: {era_data['avg_shotgun_rate']:.1%}")
            print(f"  Top innovator: {era_data['top_genes'][0]['head_coach'] if era_data['top_genes'] else 'N/A'}")
            print(f"  Most traditional: {era_data['bottom_genes'][0]['head_coach'] if era_data['bottom_genes'] else 'N/A'}")
        
        print("=" * 80)
        logger.info("Shotgun gene calculation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during shotgun gene calculation: {e}")
        raise


if __name__ == "__main__":
    main()