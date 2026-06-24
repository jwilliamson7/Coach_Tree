#!/usr/bin/env python3
"""
Calculate Coaching Tempo Gene by Coach-Year

This script calculates a composite "tempo gene" for NFL head coaches based on
two sub-components:

1. No-Huddle Gene (classification): How much more/less a coach uses no-huddle
   than expected given game context. Positive = more no-huddle than expected.
2. Pace Gene (regression): How much faster/slower a coach snaps the ball than
   expected given game context. Positive = faster than expected.

The composite tempo gene is the mean of the z-scores of both sub-components,
putting them on the same scale despite different units.

Usage:
    python calculate_tempo_gene.py [--start_year 2006] [--end_year 2024]
    python calculate_tempo_gene.py --min_plays 100
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
import sys
import warnings
import pickle
import json
import xgboost as xgb
from datetime import datetime
import gc
warnings.filterwarnings('ignore')

# Add project root to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.model_features import (
    get_no_huddle_predictor_features,
    get_pace_predictor_features,
    get_categorical_features
)
from utils import model_pipeline as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TempoGeneCalculator:
    """Calculate composite tempo gene for NFL coaches based on pace and no-huddle patterns"""

    def __init__(self, models_dir: str = "models", data_dir: str = "data/raw/play_by_play",
                 coaching_dir: str = "data/processed/Coaching"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.coaching_dir = Path(coaching_dir)

        # Model containers
        self.no_huddle_model = None
        self.no_huddle_encoders = None
        self.no_huddle_imputers = None
        self.pace_model = None
        self.pace_encoders = None
        self.pace_imputers = None

        # Feature lists
        self.no_huddle_features = get_no_huddle_predictor_features()
        self.pace_features = get_pace_predictor_features()
        self.categorical_features = get_categorical_features()

        # Coach mapping
        self.coach_mapping = None
        self.coach_dict = {}

    def load_models(self) -> None:
        """Load trained models with their encoders and imputers via the shared
        pipeline. Feature lists are taken from each model's metadata so gene-time
        features match exactly what the model was trained on (train/serve)."""
        logger.info("Loading trained models...")
        specs = {
            'no_huddle': ('no_huddle/no_huddle_prediction_model', xgb.XGBClassifier),
            'pace': ('pace/pace_prediction_model', xgb.XGBRegressor),
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
            self.coach_dict = {}
            for _, row in self.coach_mapping.iterrows():
                self.coach_dict[(row['Team'], int(row['Year']))] = row['Primary_Coach']
            logger.info(f"Loaded coach mappings for {len(self.coach_dict)} team-years")
        else:
            logger.warning(f"Coach mapping file not found: {coach_file}")
            self.coach_dict = {}

    def normalize_team_abbr(self, pbp_team: str) -> str:
        """Normalize team abbreviations from play-by-play data to coach mapping format."""
        team_mapping = {
            'GB': 'GNB', 'KC': 'KAN', 'LA': 'LAR', 'LV': 'LVR',
            'NO': 'NOR', 'NE': 'NWE', 'TEN': 'OTI', 'ARI': 'CRD',
            'LAC': 'SDG', 'SF': 'SFO', 'TB': 'TAM', 'IND': 'CLT',
            'BAL': 'RAV', 'HOU': 'HTX', 'OAK': 'RAI', 'SD': 'SDG',
            'STL': 'STL',
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
        Computes seconds_between_plays for pace analysis.

        Args:
            start_year: First year to include
            end_year: Last year to include

        Returns:
            DataFrame with play-by-play data including coach and pace columns
        """
        logger.info(f"Loading play-by-play data from {start_year} to {end_year}...")

        all_plays = []

        # Columns needed across both models
        all_features = list(set(self.no_huddle_features + self.pace_features))
        key_columns = ['play_type', 'down', 'no_huddle', 'shotgun', 'punt_attempt',
                       'field_goal_attempt', 'posteam', 'season',
                       'home_team', 'away_team', 'home_coach', 'away_coach',
                       'game_id', 'play_id', 'drive', 'game_seconds_remaining']
        needed_cols = list(set(all_features + key_columns))

        for year in range(start_year, end_year + 1):
            file_path = self.data_dir / f"play_by_play_{year}.csv"

            if not file_path.exists():
                logger.warning(f"No data found for {year}")
                continue

            try:
                header_df = pd.read_csv(file_path, nrows=0)
                keep_cols = [col for col in needed_cols if col in header_df.columns]
                df = pd.read_csv(file_path, usecols=keep_cols, low_memory=False)

                # Filter for relevant offensive plays
                df = df[
                    (df['play_type'].isin(['run', 'pass'])) &
                    (
                        (df['down'].isin([1, 2, 3])) |
                        ((df['down'] == 4) &
                         (df.get('punt_attempt', 0) != 1) &
                         (df.get('field_goal_attempt', 0) != 1))
                    )
                ].copy()

                # Sort for pace computation
                df = df.sort_values(['game_id', 'play_id'], ascending=[True, True])

                # Compute seconds between consecutive plays within same game+drive
                df['prev_game_seconds'] = df.groupby(
                    ['game_id', 'drive'])['game_seconds_remaining'].shift(1)
                df['seconds_between_plays'] = (
                    df['prev_game_seconds'] - df['game_seconds_remaining']
                )
                df = df.drop(columns=['prev_game_seconds'], errors='ignore')

                all_plays.append(df)
                logger.info(f"Loaded {len(df):,} plays from {year}")

                del df
                gc.collect()

            except Exception as e:
                logger.error(f"Error loading {year}: {e}")
                continue

        if not all_plays:
            raise ValueError("No play data loaded")

        combined = pd.concat(all_plays, ignore_index=True)
        del all_plays
        gc.collect()

        # Map plays to head coaches
        def get_head_coach(row):
            if pd.isna(row['posteam']):
                return np.nan

            if all(col in row.index for col in ['home_coach', 'away_coach', 'home_team', 'away_team']):
                if pd.notna(row.get('home_coach')) and row['posteam'] == row.get('home_team'):
                    return row['home_coach']
                elif pd.notna(row.get('away_coach')) and row['posteam'] == row.get('away_team'):
                    return row['away_coach']

            if pd.notna(row.get('season')):
                return self.coach_dict.get(
                    (self.normalize_team_abbr(row['posteam']), int(row['season'])), np.nan
                )
            return np.nan

        combined['head_coach'] = combined.apply(get_head_coach, axis=1)

        mapped_count = combined['head_coach'].notna().sum()
        total_with_posteam = combined['posteam'].notna().sum()
        logger.info(f"Total plays loaded: {len(combined):,}")
        logger.info(f"Mapped {mapped_count:,}/{total_with_posteam:,} plays to coaches "
                     f"({mapped_count/total_with_posteam*100:.1f}%)")

        return combined

    def prepare_features_for_model(self, df: pd.DataFrame, features: List[str],
                                   encoders: Optional[Dict],
                                   imputers: Optional[Dict] = None) -> np.ndarray:
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

    def calculate_no_huddle_gene(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate no-huddle gene for each coach-year.

        Gene = mean(actual_no_huddle) - mean(predicted_no_huddle_prob)
        Positive = uses no-huddle more than expected.

        Args:
            plays: DataFrame with play data (must have head_coach and no_huddle)

        Returns:
            DataFrame with coach-year no-huddle gene scores
        """
        logger.info("Calculating no-huddle gene...")

        # Filter to plays with no-huddle data and mapped coaches
        nh_plays = plays[
            (plays['no_huddle'].notna()) &
            (plays['head_coach'].notna())
        ].copy()

        if len(nh_plays) == 0:
            logger.warning("No plays found for no-huddle analysis")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(nh_plays):,} plays for no-huddle gene")

        # Prepare features and predict
        features = self.prepare_features_for_model(nh_plays, self.no_huddle_features,
                                                    self.no_huddle_encoders,
                                                    self.no_huddle_imputers)
        predictions = self.no_huddle_model.predict_proba(features)[:, 1]
        nh_plays['predicted_no_huddle_rate'] = predictions
        nh_plays['actual_no_huddle'] = nh_plays['no_huddle'].astype(int)

        # Group by coach-year
        coach_nh = nh_plays.groupby(['head_coach', 'season']).agg({
            'actual_no_huddle': 'mean',
            'predicted_no_huddle_rate': 'mean',
            'play_id': 'count'
        }).rename(columns={'play_id': 'no_huddle_plays'})

        coach_nh['no_huddle_gene'] = (
            coach_nh['actual_no_huddle'] - coach_nh['predicted_no_huddle_rate']
        )

        return coach_nh.reset_index()

    def calculate_pace_gene(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pace gene for each coach-year.

        Gene = -(mean(actual_seconds) - mean(predicted_seconds))
        Negated so positive = faster than expected.

        Args:
            plays: DataFrame with play data (must have seconds_between_plays)

        Returns:
            DataFrame with coach-year pace gene scores
        """
        logger.info("Calculating pace gene...")

        # Filter to plays with valid pace data and mapped coaches
        pace_plays = plays[
            (plays['seconds_between_plays'].notna()) &
            (plays['seconds_between_plays'] > 0) &
            (plays['seconds_between_plays'] <= 60) &
            (plays['head_coach'].notna())
        ].copy()

        if len(pace_plays) == 0:
            logger.warning("No plays found for pace analysis")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(pace_plays):,} plays for pace gene")

        # Prepare features and predict
        features = self.prepare_features_for_model(pace_plays, self.pace_features,
                                                    self.pace_encoders,
                                                    self.pace_imputers)
        predictions = self.pace_model.predict(features)
        pace_plays['predicted_pace'] = predictions
        pace_plays['actual_pace'] = pace_plays['seconds_between_plays'].astype(float)

        # Group by coach-year
        coach_pace = pace_plays.groupby(['head_coach', 'season']).agg({
            'actual_pace': 'mean',
            'predicted_pace': 'mean',
            'play_id': 'count'
        }).rename(columns={'play_id': 'pace_plays'})

        # Negate so positive = faster than expected
        # (fewer seconds than predicted = faster = positive gene)
        coach_pace['pace_gene'] = -(
            coach_pace['actual_pace'] - coach_pace['predicted_pace']
        )

        return coach_pace.reset_index()

    def calculate_composite_tempo(self, plays: pd.DataFrame,
                                  min_plays: int = 100) -> pd.DataFrame:
        """
        Calculate composite tempo gene from no-huddle and pace sub-components.

        The composite is the mean of the z-scored sub-components, putting both
        on the same scale despite different units.

        Args:
            plays: DataFrame with all play data
            min_plays: Minimum plays required per sub-component

        Returns:
            DataFrame with all tempo gene columns per coach-year
        """
        # Calculate both sub-genes
        no_huddle_df = self.calculate_no_huddle_gene(plays)
        pace_df = self.calculate_pace_gene(plays)

        if no_huddle_df.empty and pace_df.empty:
            raise ValueError("No data available for either tempo sub-component")

        # Merge on (head_coach, season) with outer join
        if not no_huddle_df.empty and not pace_df.empty:
            tempo_df = no_huddle_df.merge(
                pace_df,
                on=['head_coach', 'season'],
                how='outer'
            )
        elif not no_huddle_df.empty:
            tempo_df = no_huddle_df.copy()
        else:
            tempo_df = pace_df.copy()

        # Calculate total plays (sum of available sub-component plays)
        play_cols = [c for c in ['no_huddle_plays', 'pace_plays'] if c in tempo_df.columns]
        tempo_df['total_plays'] = tempo_df[play_cols].sum(axis=1)

        # Filter by minimum plays per sub-component
        if 'no_huddle_plays' in tempo_df.columns:
            tempo_df.loc[tempo_df['no_huddle_plays'] < min_plays, 'no_huddle_gene'] = np.nan
        if 'pace_plays' in tempo_df.columns:
            tempo_df.loc[tempo_df['pace_plays'] < min_plays, 'pace_gene'] = np.nan

        # Z-score each sub-gene independently
        if 'no_huddle_gene' in tempo_df.columns:
            nh_valid = tempo_df['no_huddle_gene'].dropna()
            if len(nh_valid) > 1:
                nh_mean = nh_valid.mean()
                nh_std = nh_valid.std()
                if nh_std > 0:
                    tempo_df['no_huddle_gene_zscore'] = (
                        (tempo_df['no_huddle_gene'] - nh_mean) / nh_std
                    )
                else:
                    tempo_df['no_huddle_gene_zscore'] = 0.0
            else:
                tempo_df['no_huddle_gene_zscore'] = np.nan

        if 'pace_gene' in tempo_df.columns:
            pace_valid = tempo_df['pace_gene'].dropna()
            if len(pace_valid) > 1:
                pace_mean = pace_valid.mean()
                pace_std = pace_valid.std()
                if pace_std > 0:
                    tempo_df['pace_gene_zscore'] = (
                        (tempo_df['pace_gene'] - pace_mean) / pace_std
                    )
                else:
                    tempo_df['pace_gene_zscore'] = 0.0
            else:
                tempo_df['pace_gene_zscore'] = np.nan

        # Composite = mean of available z-scores (NaN-tolerant)
        zscore_cols = [c for c in ['no_huddle_gene_zscore', 'pace_gene_zscore']
                       if c in tempo_df.columns]
        if zscore_cols:
            tempo_df['composite_tempo'] = tempo_df[zscore_cols].mean(axis=1)

            # Track how many sub-components contributed
            tempo_df['tempo_components'] = tempo_df[zscore_cols].notna().sum(axis=1)

            # Z-score the composite
            ct_valid = tempo_df['composite_tempo'].dropna()
            if len(ct_valid) > 1:
                ct_mean = ct_valid.mean()
                ct_std = ct_valid.std()
                if ct_std > 0:
                    tempo_df['composite_tempo_zscore'] = (
                        (tempo_df['composite_tempo'] - ct_mean) / ct_std
                    )
                else:
                    tempo_df['composite_tempo_zscore'] = 0.0
            else:
                tempo_df['composite_tempo_zscore'] = np.nan

        # Sort by season then composite tempo
        sort_cols = ['season']
        if 'composite_tempo' in tempo_df.columns:
            sort_cols.append('composite_tempo')
        tempo_df = tempo_df.sort_values(sort_cols, ascending=[True, False])

        # Drop rows where both sub-genes are NaN (no usable data)
        gene_cols = [c for c in ['no_huddle_gene', 'pace_gene'] if c in tempo_df.columns]
        if gene_cols:
            tempo_df = tempo_df.dropna(subset=gene_cols, how='all')

        return tempo_df

    def analyze_by_era(self, tempo_df: pd.DataFrame) -> Dict:
        """Analyze tempo gene trends by era."""
        logger.info("Analyzing tempo trends by era...")

        eras = {
            'Mid (2006-2012)': (2006, 2012),
            'Modern (2013-2019)': (2013, 2019),
            'Current (2020-2024)': (2020, 2024)
        }

        era_analysis = {}

        for era_name, (start_year, end_year) in eras.items():
            era_data = tempo_df[
                (tempo_df['season'] >= start_year) &
                (tempo_df['season'] <= end_year)
            ]

            if len(era_data) > 0:
                era_info = {
                    'num_coach_years': len(era_data),
                    'num_coaches': int(era_data['head_coach'].nunique()),
                }

                if 'no_huddle_gene' in era_data.columns:
                    era_info['avg_no_huddle_gene'] = float(era_data['no_huddle_gene'].mean())
                if 'pace_gene' in era_data.columns:
                    era_info['avg_pace_gene'] = float(era_data['pace_gene'].mean())
                if 'composite_tempo' in era_data.columns:
                    valid = era_data.dropna(subset=['composite_tempo'])
                    if len(valid) > 0:
                        era_info['top_tempo'] = valid.nlargest(5, 'composite_tempo')[
                            ['head_coach', 'season', 'composite_tempo']
                        ].to_dict('records')
                        era_info['bottom_tempo'] = valid.nsmallest(5, 'composite_tempo')[
                            ['head_coach', 'season', 'composite_tempo']
                        ].to_dict('records')

                era_analysis[era_name] = era_info

        return era_analysis

    def save_results(self, tempo_df: pd.DataFrame, era_analysis: Dict,
                     output_dir: str = "data/processed/coaching_genes") -> None:
        """Save tempo gene results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full results as CSV
        csv_path = output_path / "tempo_gene.csv"
        tempo_df.to_csv(csv_path, index=False)
        logger.info(f"Saved tempo gene data to {csv_path}")

        # Build summary
        summary = {
            'generated_date': datetime.now().isoformat(),
            'num_coach_years': len(tempo_df),
            'unique_coaches': int(tempo_df['head_coach'].nunique()),
            'years_covered': f"{int(tempo_df['season'].min())}-{int(tempo_df['season'].max())}",
            'metrics': {},
            'era_analysis': era_analysis
        }

        # Add metrics for each available gene component
        for col, label in [('no_huddle_gene', 'no_huddle_gene'),
                           ('pace_gene', 'pace_gene'),
                           ('composite_tempo', 'composite_tempo')]:
            if col in tempo_df.columns:
                valid = tempo_df[col].dropna()
                if len(valid) > 0:
                    summary['metrics'][label] = {
                        'mean': float(valid.mean()),
                        'std': float(valid.std()),
                        'min': float(valid.min()),
                        'max': float(valid.max())
                    }

        # Top/bottom coach-years by composite
        if 'composite_tempo' in tempo_df.columns:
            valid = tempo_df.dropna(subset=['composite_tempo']).sort_values(
                'composite_tempo', ascending=False)
            summary['fastest_tempo_coach_years'] = valid.head(10)[
                ['head_coach', 'season', 'composite_tempo', 'composite_tempo_zscore']
            ].to_dict('records')
            summary['slowest_tempo_coach_years'] = valid.tail(10)[
                ['head_coach', 'season', 'composite_tempo', 'composite_tempo_zscore']
            ].to_dict('records')

        json_path = output_path / "tempo_gene_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {json_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate coaching tempo genes')
    parser.add_argument('--start_year', type=int, default=2006,
                        help='Start year for analysis (default: 2006)')
    parser.add_argument('--end_year', type=int, default=2024,
                        help='End year for analysis (default: 2024)')
    parser.add_argument('--output_dir', type=str, default='data/processed/coaching_genes',
                        help='Output directory for results')
    parser.add_argument('--min_plays', type=int, default=100,
                        help='Minimum plays per sub-component for a coach to be included (default: 100)')

    args = parser.parse_args()

    print("=" * 80)
    print("COACHING TEMPO GENE CALCULATOR")
    print("=" * 80)
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum plays threshold: {args.min_plays}")
    print("=" * 80 + "\n")

    try:
        calculator = TempoGeneCalculator()

        logger.info("Step 1: Loading predictive models...")
        calculator.load_models()

        logger.info("Step 2: Loading coach mappings...")
        calculator.load_coach_mappings()

        logger.info("Step 3: Loading play-by-play data...")
        plays = calculator.load_play_data(args.start_year, args.end_year)

        logger.info("Step 4: Calculating composite tempo gene...")
        tempo_df = calculator.calculate_composite_tempo(plays, min_plays=args.min_plays)

        logger.info("Step 5: Analyzing trends by era...")
        era_analysis = calculator.analyze_by_era(tempo_df)

        logger.info("Step 6: Saving results...")
        calculator.save_results(tempo_df, era_analysis, args.output_dir)

        # Print summary
        print("\n" + "=" * 80)
        print("TEMPO GENE SUMMARY")
        print("=" * 80)
        print(f"Total coach-years analyzed: {len(tempo_df)}")
        print(f"Unique coaches: {tempo_df['head_coach'].nunique()}")
        print(f"Years covered: {int(tempo_df['season'].min())} - {int(tempo_df['season'].max())}")

        if 'composite_tempo' in tempo_df.columns:
            valid = tempo_df.dropna(subset=['composite_tempo'])

            print("\nFastest-Tempo Coach-Years (Positive Composite):")
            for _, row in valid.nlargest(10, 'composite_tempo').iterrows():
                nh_str = f"NH: {row.get('no_huddle_gene', float('nan')):+.3f}" if pd.notna(row.get('no_huddle_gene')) else "NH: N/A"
                pace_str = f"Pace: {row.get('pace_gene', float('nan')):+.3f}" if pd.notna(row.get('pace_gene')) else "Pace: N/A"
                print(f"  {row['head_coach']:25} ({int(row['season'])}) "
                      f"Composite: {row['composite_tempo']:+.3f}  ({nh_str}, {pace_str})")

            print("\nSlowest-Tempo Coach-Years (Negative Composite):")
            for _, row in valid.nsmallest(10, 'composite_tempo').iterrows():
                nh_str = f"NH: {row.get('no_huddle_gene', float('nan')):+.3f}" if pd.notna(row.get('no_huddle_gene')) else "NH: N/A"
                pace_str = f"Pace: {row.get('pace_gene', float('nan')):+.3f}" if pd.notna(row.get('pace_gene')) else "Pace: N/A"
                print(f"  {row['head_coach']:25} ({int(row['season'])}) "
                      f"Composite: {row['composite_tempo']:+.3f}  ({nh_str}, {pace_str})")

        if 'no_huddle_gene' in tempo_df.columns:
            nh_valid = tempo_df['no_huddle_gene'].dropna()
            print(f"\nNo-Huddle Gene: mean={nh_valid.mean():.4f}, std={nh_valid.std():.4f}")
        if 'pace_gene' in tempo_df.columns:
            pace_valid = tempo_df['pace_gene'].dropna()
            print(f"Pace Gene: mean={pace_valid.mean():.4f}, std={pace_valid.std():.4f}")
        if 'composite_tempo' in tempo_df.columns:
            ct_valid = tempo_df['composite_tempo'].dropna()
            print(f"Composite Tempo: mean={ct_valid.mean():.4f}, std={ct_valid.std():.4f}")

        # Correlation between sub-components
        if all(c in tempo_df.columns for c in ['no_huddle_gene', 'pace_gene']):
            both_valid = tempo_df.dropna(subset=['no_huddle_gene', 'pace_gene'])
            if len(both_valid) > 2:
                corr = both_valid['no_huddle_gene'].corr(both_valid['pace_gene'])
                print(f"\nCorrelation (no-huddle vs pace): {corr:.3f}")

        print("\nEra Analysis:")
        for era_name, era_data in era_analysis.items():
            print(f"\n{era_name}:")
            print(f"  Coach-years: {era_data['num_coach_years']}")
            if 'top_tempo' in era_data and era_data['top_tempo']:
                print(f"  Fastest: {era_data['top_tempo'][0]['head_coach']}")
            if 'bottom_tempo' in era_data and era_data['bottom_tempo']:
                print(f"  Slowest: {era_data['bottom_tempo'][0]['head_coach']}")

        print("=" * 80)
        logger.info("Tempo gene calculation completed successfully!")

    except Exception as e:
        logger.error(f"Error during tempo gene calculation: {e}")
        raise


if __name__ == "__main__":
    main()
