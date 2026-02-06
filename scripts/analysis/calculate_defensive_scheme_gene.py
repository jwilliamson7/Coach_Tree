#!/usr/bin/env python3
"""
Calculate Defensive Scheme Gene by Team-Year

This script calculates a composite "defensive scheme gene" for NFL teams based
on three sub-components:

1. Box Stacking Gene (regression): How many more/fewer defenders a team puts
   in the box than expected. Positive = more aggressive box loading.
   Available 2016+.
2. Pass Rush Gene (regression): How many more/fewer pass rushers a team sends
   than expected. Positive = sends more rushers (blitz-heavy).
   Available 2016+.
3. Man Coverage Gene (classification): How much more/less a team plays man
   coverage than expected. Positive = plays more man than expected.
   Available 2018+ only.

The composite is the mean of the z-scores of available sub-components.
For 2016-2017, only box stacking and pass rush contribute (2 components).
For 2018+, all three contribute (3 components).

Attribution: plays are grouped by defteam (the defending team) per season.
The HC of the defending team is also recorded for inheritance analysis.

Usage:
    python calculate_defensive_scheme_gene.py [--start_year 2016] [--end_year 2024]
    python calculate_defensive_scheme_gene.py --min_plays 100
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
    get_defensive_scheme_predictor_features,
    get_categorical_features
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DefensiveSchemeGeneCalculator:
    """Calculate composite defensive scheme gene per team-year"""

    def __init__(self, models_dir: str = "models", data_dir: str = "data/raw/play_by_play"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)

        # Model containers
        self.box_model = None
        self.box_encoders = None
        self.rush_model = None
        self.rush_encoders = None
        self.man_model = None
        self.man_encoders = None

        # Feature list (shared across all 3 defensive models)
        self.features = get_defensive_scheme_predictor_features()
        self.categorical_features = get_categorical_features()

    def load_models(self) -> None:
        """Load all three trained defensive models and their encoders"""
        logger.info("Loading trained defensive models...")

        # Load box stacking regressor
        box_path = self.models_dir / "box_stacking" / "box_stacking_prediction_model.json"
        if box_path.exists():
            self.box_model = xgb.XGBRegressor()
            self.box_model.load_model(str(box_path))
            encoders_path = self.models_dir / "box_stacking" / "box_stacking_prediction_model_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.box_encoders = pickle.load(f)
            logger.info("Loaded box stacking prediction model")
        else:
            raise FileNotFoundError(f"Box stacking model not found at {box_path}")

        # Load pass rush regressor
        rush_path = self.models_dir / "pass_rush" / "pass_rush_prediction_model.json"
        if rush_path.exists():
            self.rush_model = xgb.XGBRegressor()
            self.rush_model.load_model(str(rush_path))
            encoders_path = self.models_dir / "pass_rush" / "pass_rush_prediction_model_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.rush_encoders = pickle.load(f)
            logger.info("Loaded pass rush prediction model")
        else:
            raise FileNotFoundError(f"Pass rush model not found at {rush_path}")

        # Load man coverage classifier (optional - only available from 2018+)
        man_path = self.models_dir / "man_coverage" / "man_coverage_prediction_model.json"
        if man_path.exists():
            self.man_model = xgb.XGBClassifier()
            self.man_model.load_model(str(man_path))
            encoders_path = self.models_dir / "man_coverage" / "man_coverage_prediction_model_encoders.pkl"
            if encoders_path.exists():
                with open(encoders_path, 'rb') as f:
                    self.man_encoders = pickle.load(f)
            logger.info("Loaded man coverage prediction model")
        else:
            logger.warning(f"Man coverage model not found at {man_path} - will skip man coverage gene")
            self.man_model = None

    def load_play_data(self, start_year: int = 2016, end_year: int = 2024) -> pd.DataFrame:
        """
        Load play-by-play data for the specified years.

        Uses defteam to identify the defending team and derives the HC
        of the defending team from home_coach/away_coach.

        Args:
            start_year: First year to include
            end_year: Last year to include

        Returns:
            DataFrame with play-by-play data including defteam_coach
        """
        logger.info(f"Loading play-by-play data from {start_year} to {end_year}...")

        all_plays = []

        key_columns = ['play_type', 'down', 'shotgun', 'no_huddle',
                       'punt_attempt', 'field_goal_attempt',
                       'posteam', 'defteam', 'season',
                       'home_team', 'away_team', 'home_coach', 'away_coach',
                       'game_id', 'play_id',
                       'defenders_in_box', 'number_of_pass_rushers',
                       'defense_man_zone_type']
        needed_cols = list(set(self.features + key_columns))

        for year in range(start_year, end_year + 1):
            file_path = self.data_dir / f"play_by_play_{year}.csv"

            if not file_path.exists():
                logger.warning(f"No data found for {year}")
                continue

            try:
                header_df = pd.read_csv(file_path, nrows=0)
                keep_cols = [col for col in needed_cols if col in header_df.columns]
                df = pd.read_csv(file_path, usecols=keep_cols, low_memory=False)

                # Filter for relevant offensive plays (run or pass)
                df = df[
                    (df['play_type'].isin(['run', 'pass'])) &
                    (
                        (df['down'].isin([1, 2, 3])) |
                        ((df['down'] == 4) &
                         (df.get('punt_attempt', 0) != 1) &
                         (df.get('field_goal_attempt', 0) != 1))
                    )
                ].copy()

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

        # Derive HC of the defending team from PBP coach fields
        logger.info("Attributing plays to defending team HC...")

        def get_def_coach(row):
            if pd.isna(row.get('defteam')):
                return np.nan
            if pd.notna(row.get('home_coach')) and row['defteam'] == row.get('home_team'):
                return row['home_coach']
            elif pd.notna(row.get('away_coach')) and row['defteam'] == row.get('away_team'):
                return row['away_coach']
            return np.nan

        combined['defteam_coach'] = combined.apply(get_def_coach, axis=1)

        # Report attribution statistics
        total = len(combined)
        has_defteam = combined['defteam'].notna().sum()
        has_coach = combined['defteam_coach'].notna().sum()

        logger.info(f"Total plays: {total:,}")
        logger.info(f"Plays with defteam: {has_defteam:,}")
        logger.info(f"Plays with defteam HC: {has_coach:,} ({has_coach/has_defteam*100:.1f}%)")

        return combined

    def prepare_features_for_model(self, df: pd.DataFrame, features: List[str],
                                   encoders: Optional[Dict]) -> np.ndarray:
        """
        Prepare features for model prediction, handling missing values and encoding.

        Args:
            df: DataFrame with raw features
            features: List of feature column names
            encoders: Label encoders for categorical features

        Returns:
            Numpy array ready for model prediction
        """
        df_features = df[features].copy()

        if encoders:
            for col in features:
                if col in encoders and col in self.categorical_features:
                    le = encoders[col]
                    df_features[col] = df_features[col].astype(str)
                    mask = df_features[col].isin(le.classes_)
                    df_features.loc[~mask, col] = 'nan'
                    if 'nan' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'nan')
                    df_features[col] = le.transform(df_features[col])

        from sklearn.impute import SimpleImputer
        numeric_cols = [col for col in df_features.columns
                        if df_features[col].dtype in ['int64', 'float64']]
        if numeric_cols:
            imputer = SimpleImputer(strategy='median')
            df_features[numeric_cols] = imputer.fit_transform(df_features[numeric_cols])

        return df_features.values

    def calculate_box_stacking_gene(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate box stacking gene for each team-year.

        Gene = mean(actual_defenders_in_box) - mean(predicted_defenders_in_box)
        Positive = more aggressive box loading than expected.
        """
        logger.info("Calculating box stacking gene...")

        box_plays = plays[
            (plays['defenders_in_box'].notna()) &
            (plays['defenders_in_box'] > 0) &
            (plays['defteam'].notna())
        ].copy()

        if len(box_plays) == 0:
            logger.warning("No plays found for box stacking analysis")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(box_plays):,} plays for box stacking gene")

        features = self.prepare_features_for_model(box_plays, self.features, self.box_encoders)
        predictions = self.box_model.predict(features)
        box_plays['predicted_box'] = predictions
        box_plays['actual_box'] = box_plays['defenders_in_box'].astype(float)

        team_box = box_plays.groupby(['defteam', 'season']).agg({
            'actual_box': 'mean',
            'predicted_box': 'mean',
            'play_id': 'count'
        }).rename(columns={'play_id': 'box_plays'})

        team_box['box_stacking_gene'] = (
            team_box['actual_box'] - team_box['predicted_box']
        )

        return team_box.reset_index()

    def calculate_pass_rush_gene(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pass rush gene for each team-year.

        Gene = mean(actual_rushers) - mean(predicted_rushers)
        Positive = sends more pass rushers than expected.
        """
        logger.info("Calculating pass rush gene...")

        rush_plays = plays[
            (plays['play_type'] == 'pass') &
            (plays['number_of_pass_rushers'].notna()) &
            (plays['number_of_pass_rushers'] > 0) &
            (plays['defteam'].notna())
        ].copy()

        if len(rush_plays) == 0:
            logger.warning("No plays found for pass rush analysis")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(rush_plays):,} plays for pass rush gene")

        features = self.prepare_features_for_model(rush_plays, self.features, self.rush_encoders)
        predictions = self.rush_model.predict(features)
        rush_plays['predicted_rushers'] = predictions
        rush_plays['actual_rushers'] = rush_plays['number_of_pass_rushers'].astype(float)

        team_rush = rush_plays.groupby(['defteam', 'season']).agg({
            'actual_rushers': 'mean',
            'predicted_rushers': 'mean',
            'play_id': 'count'
        }).rename(columns={'play_id': 'rush_plays'})

        team_rush['pass_rush_gene'] = (
            team_rush['actual_rushers'] - team_rush['predicted_rushers']
        )

        return team_rush.reset_index()

    def calculate_man_coverage_gene(self, plays: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate man coverage gene for each team-year.

        Gene = mean(actual_man_rate) - mean(predicted_man_prob)
        Positive = plays more man coverage than expected.
        Only available for 2018+ when defense_man_zone_type data exists.
        """
        if self.man_model is None:
            logger.info("Skipping man coverage gene (model not loaded)")
            return pd.DataFrame()

        logger.info("Calculating man coverage gene...")

        man_plays = plays[
            (plays['play_type'] == 'pass') &
            (plays['defense_man_zone_type'].isin(['MAN_COVERAGE', 'ZONE_COVERAGE'])) &
            (plays['defteam'].notna())
        ].copy()

        if len(man_plays) == 0:
            logger.warning("No plays found for man coverage analysis")
            return pd.DataFrame()

        logger.info(f"Analyzing {len(man_plays):,} plays for man coverage gene")

        features = self.prepare_features_for_model(man_plays, self.features, self.man_encoders)
        predictions = self.man_model.predict_proba(features)[:, 1]
        man_plays['predicted_man_prob'] = predictions
        man_plays['actual_man'] = (man_plays['defense_man_zone_type'] == 'MAN_COVERAGE').astype(int)

        team_man = man_plays.groupby(['defteam', 'season']).agg({
            'actual_man': 'mean',
            'predicted_man_prob': 'mean',
            'play_id': 'count'
        }).rename(columns={'play_id': 'man_plays'})

        team_man['man_coverage_gene'] = (
            team_man['actual_man'] - team_man['predicted_man_prob']
        )

        return team_man.reset_index()

    def calculate_composite_scheme(self, plays: pd.DataFrame,
                                   min_plays: int = 100) -> pd.DataFrame:
        """
        Calculate composite defensive scheme gene from all sub-components.

        The composite is the mean of the z-scored sub-components. For 2016-2017,
        only box stacking and pass rush contribute (2 components). For 2018+,
        all three contribute (3 components).

        Args:
            plays: DataFrame with all play data
            min_plays: Minimum plays required per sub-component

        Returns:
            DataFrame with all scheme gene columns per team-year
        """
        # Calculate all sub-genes
        box_df = self.calculate_box_stacking_gene(plays)
        rush_df = self.calculate_pass_rush_gene(plays)
        man_df = self.calculate_man_coverage_gene(plays)

        if box_df.empty and rush_df.empty:
            raise ValueError("No data available for any defensive scheme sub-component")

        # Merge sub-genes on (defteam, season) with outer joins
        scheme_df = box_df if not box_df.empty else pd.DataFrame(columns=['defteam', 'season'])

        if not rush_df.empty:
            scheme_df = scheme_df.merge(
                rush_df,
                on=['defteam', 'season'],
                how='outer'
            )

        if not man_df.empty:
            scheme_df = scheme_df.merge(
                man_df,
                on=['defteam', 'season'],
                how='outer'
            )

        # Calculate total plays (sum of available sub-component plays)
        play_cols = [c for c in ['box_plays', 'rush_plays', 'man_plays'] if c in scheme_df.columns]
        scheme_df['total_plays'] = scheme_df[play_cols].sum(axis=1)

        # Filter by minimum plays per sub-component
        if 'box_plays' in scheme_df.columns:
            scheme_df.loc[scheme_df['box_plays'] < min_plays, 'box_stacking_gene'] = np.nan
        if 'rush_plays' in scheme_df.columns:
            scheme_df.loc[scheme_df['rush_plays'] < min_plays, 'pass_rush_gene'] = np.nan
        if 'man_plays' in scheme_df.columns:
            scheme_df.loc[scheme_df['man_plays'] < min_plays, 'man_coverage_gene'] = np.nan

        # Z-score each sub-gene independently
        gene_zscore_cols = []
        for gene_col, zscore_col in [
            ('box_stacking_gene', 'box_stacking_gene_zscore'),
            ('pass_rush_gene', 'pass_rush_gene_zscore'),
            ('man_coverage_gene', 'man_coverage_gene_zscore')
        ]:
            if gene_col in scheme_df.columns:
                valid = scheme_df[gene_col].dropna()
                if len(valid) > 1:
                    g_mean = valid.mean()
                    g_std = valid.std()
                    if g_std > 0:
                        scheme_df[zscore_col] = (scheme_df[gene_col] - g_mean) / g_std
                    else:
                        scheme_df[zscore_col] = 0.0
                else:
                    scheme_df[zscore_col] = np.nan
                gene_zscore_cols.append(zscore_col)

        # Composite = mean of available z-scores (NaN-tolerant)
        if gene_zscore_cols:
            scheme_df['composite_scheme'] = scheme_df[gene_zscore_cols].mean(axis=1)

            # Track how many sub-components contributed
            scheme_df['scheme_components'] = scheme_df[gene_zscore_cols].notna().sum(axis=1)

            # Z-score the composite
            cs_valid = scheme_df['composite_scheme'].dropna()
            if len(cs_valid) > 1:
                cs_mean = cs_valid.mean()
                cs_std = cs_valid.std()
                if cs_std > 0:
                    scheme_df['composite_scheme_zscore'] = (
                        (scheme_df['composite_scheme'] - cs_mean) / cs_std
                    )
                else:
                    scheme_df['composite_scheme_zscore'] = 0.0
            else:
                scheme_df['composite_scheme_zscore'] = np.nan

        # Add HC of defending team (most common per team-year from PBP data)
        if 'defteam_coach' in plays.columns:
            hc_info = plays.dropna(subset=['defteam', 'defteam_coach']).groupby(
                ['defteam', 'season']
            )['defteam_coach'].agg(lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan)
            hc_info = hc_info.reset_index().rename(columns={'defteam_coach': 'head_coach'})
            scheme_df = scheme_df.merge(hc_info, on=['defteam', 'season'], how='left')

        # Sort by season then composite
        sort_cols = ['season']
        if 'composite_scheme' in scheme_df.columns:
            sort_cols.append('composite_scheme')
            scheme_df = scheme_df.sort_values(sort_cols, ascending=[True, False])
        else:
            scheme_df = scheme_df.sort_values(sort_cols)

        # Drop rows where all sub-genes are NaN
        gene_cols = [c for c in ['box_stacking_gene', 'pass_rush_gene', 'man_coverage_gene']
                     if c in scheme_df.columns]
        if gene_cols:
            scheme_df = scheme_df.dropna(subset=gene_cols, how='all')

        return scheme_df

    def analyze_by_era(self, scheme_df: pd.DataFrame) -> Dict:
        """Analyze defensive scheme gene trends by era."""
        logger.info("Analyzing defensive scheme trends by era...")

        eras = {
            'Early Tracking (2016-2017)': (2016, 2017),
            'Full Tracking (2018-2021)': (2018, 2021),
            'Current (2022-2024)': (2022, 2024)
        }

        era_analysis = {}

        for era_name, (start_year, end_year) in eras.items():
            era_data = scheme_df[
                (scheme_df['season'] >= start_year) &
                (scheme_df['season'] <= end_year)
            ]

            if len(era_data) > 0:
                era_info = {
                    'num_team_years': len(era_data),
                    'num_teams': int(era_data['defteam'].nunique()),
                }

                if 'box_stacking_gene' in era_data.columns:
                    era_info['avg_box_stacking_gene'] = float(era_data['box_stacking_gene'].mean())
                if 'pass_rush_gene' in era_data.columns:
                    era_info['avg_pass_rush_gene'] = float(era_data['pass_rush_gene'].mean())
                if 'man_coverage_gene' in era_data.columns:
                    valid_man = era_data['man_coverage_gene'].dropna()
                    if len(valid_man) > 0:
                        era_info['avg_man_coverage_gene'] = float(valid_man.mean())

                if 'composite_scheme' in era_data.columns:
                    valid = era_data.dropna(subset=['composite_scheme'])
                    if len(valid) > 0:
                        display_cols = ['defteam', 'season', 'composite_scheme']
                        if 'head_coach' in valid.columns:
                            display_cols.insert(2, 'head_coach')
                        era_info['top_scheme'] = valid.nlargest(5, 'composite_scheme')[
                            display_cols
                        ].to_dict('records')
                        era_info['bottom_scheme'] = valid.nsmallest(5, 'composite_scheme')[
                            display_cols
                        ].to_dict('records')

                era_analysis[era_name] = era_info

        return era_analysis

    def save_results(self, scheme_df: pd.DataFrame, era_analysis: Dict,
                     output_dir: str = "data/processed/coaching_genes") -> None:
        """Save defensive scheme gene results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save full results as CSV
        csv_path = output_path / "defensive_scheme_gene.csv"
        scheme_df.to_csv(csv_path, index=False)
        logger.info(f"Saved defensive scheme gene data to {csv_path}")

        # Build summary
        summary = {
            'generated_date': datetime.now().isoformat(),
            'num_team_years': len(scheme_df),
            'unique_teams': int(scheme_df['defteam'].nunique()),
            'years_covered': f"{int(scheme_df['season'].min())}-{int(scheme_df['season'].max())}",
            'metrics': {},
            'era_analysis': era_analysis
        }

        # Add metrics for each available gene component
        for col, label in [('box_stacking_gene', 'box_stacking_gene'),
                           ('pass_rush_gene', 'pass_rush_gene'),
                           ('man_coverage_gene', 'man_coverage_gene'),
                           ('composite_scheme', 'composite_scheme')]:
            if col in scheme_df.columns:
                valid = scheme_df[col].dropna()
                if len(valid) > 0:
                    summary['metrics'][label] = {
                        'mean': float(valid.mean()),
                        'std': float(valid.std()),
                        'min': float(valid.min()),
                        'max': float(valid.max())
                    }

        # Component count distribution
        if 'scheme_components' in scheme_df.columns:
            summary['component_distribution'] = scheme_df['scheme_components'].value_counts().sort_index().to_dict()

        # Top/bottom team-years by composite
        if 'composite_scheme' in scheme_df.columns:
            valid = scheme_df.dropna(subset=['composite_scheme']).sort_values(
                'composite_scheme', ascending=False)
            cols_for_summary = ['defteam', 'season', 'composite_scheme',
                               'composite_scheme_zscore']
            if 'head_coach' in valid.columns:
                cols_for_summary.insert(2, 'head_coach')
            cols_for_summary = [c for c in cols_for_summary if c in valid.columns]
            summary['most_aggressive_team_years'] = valid.head(10)[cols_for_summary].to_dict('records')
            summary['least_aggressive_team_years'] = valid.tail(10)[cols_for_summary].to_dict('records')

        json_path = output_path / "defensive_scheme_gene_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {json_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Calculate defensive scheme genes per team-year')
    parser.add_argument('--start_year', type=int, default=2016,
                        help='Start year for analysis (default: 2016)')
    parser.add_argument('--end_year', type=int, default=2024,
                        help='End year for analysis (default: 2024)')
    parser.add_argument('--output_dir', type=str, default='data/processed/coaching_genes',
                        help='Output directory for results')
    parser.add_argument('--min_plays', type=int, default=100,
                        help='Minimum plays per sub-component (default: 100)')

    args = parser.parse_args()

    print("=" * 80)
    print("DEFENSIVE SCHEME GENE CALCULATOR")
    print("=" * 80)
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum plays threshold: {args.min_plays}")
    print("=" * 80 + "\n")

    try:
        calculator = DefensiveSchemeGeneCalculator()

        logger.info("Step 1: Loading predictive models...")
        calculator.load_models()

        logger.info("Step 2: Loading play-by-play data...")
        plays = calculator.load_play_data(args.start_year, args.end_year)

        logger.info("Step 3: Calculating composite defensive scheme gene...")
        scheme_df = calculator.calculate_composite_scheme(plays, min_plays=args.min_plays)

        logger.info("Step 4: Analyzing trends by era...")
        era_analysis = calculator.analyze_by_era(scheme_df)

        logger.info("Step 5: Saving results...")
        calculator.save_results(scheme_df, era_analysis, args.output_dir)

        # Print summary
        print("\n" + "=" * 80)
        print("DEFENSIVE SCHEME GENE SUMMARY")
        print("=" * 80)
        print(f"Total team-years analyzed: {len(scheme_df)}")
        print(f"Unique teams: {scheme_df['defteam'].nunique()}")
        print(f"Years covered: {int(scheme_df['season'].min())} - {int(scheme_df['season'].max())}")

        # Component distribution
        if 'scheme_components' in scheme_df.columns:
            comp_dist = scheme_df['scheme_components'].value_counts().sort_index()
            print(f"\nComponents per team-year:")
            for n_comp, count in comp_dist.items():
                print(f"  {int(n_comp)} components: {count} team-years")

        if 'composite_scheme' in scheme_df.columns:
            valid = scheme_df.dropna(subset=['composite_scheme'])

            hc_col = 'head_coach' in valid.columns

            print("\nMost Aggressive Defensive Schemes (Positive Composite):")
            for _, row in valid.nlargest(10, 'composite_scheme').iterrows():
                box_str = f"Box: {row.get('box_stacking_gene', float('nan')):+.3f}" if pd.notna(row.get('box_stacking_gene')) else "Box: N/A"
                rush_str = f"Rush: {row.get('pass_rush_gene', float('nan')):+.3f}" if pd.notna(row.get('pass_rush_gene')) else "Rush: N/A"
                man_str = f"Man: {row.get('man_coverage_gene', float('nan')):+.3f}" if pd.notna(row.get('man_coverage_gene')) else "Man: N/A"
                hc_str = f" HC: {row['head_coach']}" if hc_col and pd.notna(row.get('head_coach')) else ""
                print(f"  {row['defteam']:4} ({int(row['season'])}) "
                      f"Composite: {row['composite_scheme']:+.3f}  ({box_str}, {rush_str}, {man_str}){hc_str}")

            print("\nLeast Aggressive Defensive Schemes (Negative Composite):")
            for _, row in valid.nsmallest(10, 'composite_scheme').iterrows():
                box_str = f"Box: {row.get('box_stacking_gene', float('nan')):+.3f}" if pd.notna(row.get('box_stacking_gene')) else "Box: N/A"
                rush_str = f"Rush: {row.get('pass_rush_gene', float('nan')):+.3f}" if pd.notna(row.get('pass_rush_gene')) else "Rush: N/A"
                man_str = f"Man: {row.get('man_coverage_gene', float('nan')):+.3f}" if pd.notna(row.get('man_coverage_gene')) else "Man: N/A"
                hc_str = f" HC: {row['head_coach']}" if hc_col and pd.notna(row.get('head_coach')) else ""
                print(f"  {row['defteam']:4} ({int(row['season'])}) "
                      f"Composite: {row['composite_scheme']:+.3f}  ({box_str}, {rush_str}, {man_str}){hc_str}")

        # Sub-gene summaries
        if 'box_stacking_gene' in scheme_df.columns:
            valid = scheme_df['box_stacking_gene'].dropna()
            print(f"\nBox Stacking Gene: mean={valid.mean():.4f}, std={valid.std():.4f}")
        if 'pass_rush_gene' in scheme_df.columns:
            valid = scheme_df['pass_rush_gene'].dropna()
            print(f"Pass Rush Gene: mean={valid.mean():.4f}, std={valid.std():.4f}")
        if 'man_coverage_gene' in scheme_df.columns:
            valid = scheme_df['man_coverage_gene'].dropna()
            if len(valid) > 0:
                print(f"Man Coverage Gene: mean={valid.mean():.4f}, std={valid.std():.4f}")
        if 'composite_scheme' in scheme_df.columns:
            valid = scheme_df['composite_scheme'].dropna()
            print(f"Composite Scheme: mean={valid.mean():.4f}, std={valid.std():.4f}")

        # Correlation between sub-components
        corr_pairs = [
            ('box_stacking_gene', 'pass_rush_gene'),
            ('box_stacking_gene', 'man_coverage_gene'),
            ('pass_rush_gene', 'man_coverage_gene'),
        ]
        print("\nSub-gene Correlations:")
        for col_a, col_b in corr_pairs:
            if all(c in scheme_df.columns for c in [col_a, col_b]):
                both_valid = scheme_df.dropna(subset=[col_a, col_b])
                if len(both_valid) > 2:
                    corr = both_valid[col_a].corr(both_valid[col_b])
                    print(f"  {col_a} vs {col_b}: {corr:.3f}")

        print("\nEra Analysis:")
        for era_name, era_data in era_analysis.items():
            print(f"\n{era_name}:")
            print(f"  Team-years: {era_data['num_team_years']}")
            if 'top_scheme' in era_data and era_data['top_scheme']:
                top = era_data['top_scheme'][0]
                label = f"{top['defteam']} ({top.get('head_coach', '?')})" if 'head_coach' in top else top['defteam']
                print(f"  Most aggressive: {label}")
            if 'bottom_scheme' in era_data and era_data['bottom_scheme']:
                bot = era_data['bottom_scheme'][0]
                label = f"{bot['defteam']} ({bot.get('head_coach', '?')})" if 'head_coach' in bot else bot['defteam']
                print(f"  Least aggressive: {label}")

        print("=" * 80)
        logger.info("Defensive scheme gene calculation completed successfully!")

    except Exception as e:
        logger.error(f"Error during gene calculation: {e}")
        raise


if __name__ == "__main__":
    main()
