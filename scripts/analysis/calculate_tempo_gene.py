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
from utils import parsimony
from utils.coach_attribution import build_game_coach_map, attach_head_coach
from crawlers.utils.data_constants import standardize_team_abbreviation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TempoGeneCalculator:
    """Calculate composite tempo gene for NFL coaches based on pace and no-huddle patterns"""

    def __init__(self, models_dir: str = "models", data_dir: str = "data/raw/play_by_play",
                 coaching_dir: str = "data/processed/Coaching", rel_floor: float = 0.1):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.coaching_dir = Path(coaching_dir)

        # Reliability floor for the weighted composite (see utils.parsimony).
        self.rel_floor = rel_floor
        self.component_reliability = {}

        # WS1 cross-fitting: leave-coach-out OOF predictions so a coach is never
        # scored by a model trained on that same coach (removes attenuation).
        self.use_crossfit = True
        self.cv_splits = 5

        # Model containers
        self.no_huddle_model = None
        self.no_huddle_encoders = None
        self.no_huddle_imputers = None
        self.no_huddle_params = None
        self.pace_model = None
        self.pace_encoders = None
        self.pace_imputers = None
        self.pace_params = None

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
            setattr(self, f"{name}_params", bundle['model'].get_params())
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
                # Year-aware standardized PFR key (matches the standardized lookup).
                yr = int(row['Year'])
                self.coach_dict[(standardize_team_abbreviation(row['Team'], yr), yr)] = \
                    row['Primary_Coach']
            logger.info(f"Loaded coach mappings for {len(self.coach_dict)} team-years")
        else:
            logger.warning(f"Coach mapping file not found: {coach_file}")
            self.coach_dict = {}

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

        # Authoritative game-level head-coach attribution shared across all gene
        # calculators (utils/coach_attribution.py): PBP per-game coach, mid-season
        # changes PBP missed corrected from coach game-count records, names
        # canonicalized to the coaching-tree identity, NOR 2012 dropped.
        gcmap = build_game_coach_map(
            start_year, end_year, self.data_dir,
            self.data_dir.parent / "Coaches",
            drop_team_seasons=[("NO", 2012)], logger=logger)
        combined = attach_head_coach(combined, gcmap, "posteam", "head_coach")

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

    def _predict_component(self, df: pd.DataFrame, features: List[str], target_col: str,
                           model, encoders, imputers, params,
                           objective: str, is_classifier: bool) -> np.ndarray:
        """Per-play predictions for one tempo sub-gene.

        Default (use_crossfit): leave-coach-out OOF predictions via GroupKFold on
        head_coach, refitting encode/impute/SVD + a fresh XGB (tuned `params`, no
        re-search) per fold. `df` must already carry `target_col` and a non-null
        `head_coach` for every row. Fallback uses the persisted all-data model.
        """
        if self.use_crossfit:
            return mp.crossfit_predict(
                df, features, target_col, self.categorical_features,
                df["head_coach"], params, objective, is_classifier,
                n_splits=self.cv_splits, logger=logger)
        feats = self.prepare_features_for_model(df, features, encoders, imputers)
        if is_classifier:
            return model.predict_proba(feats)[:, 1]
        return model.predict(feats)

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

        # Actual no-huddle (target) must exist before cross-fit prediction
        nh_plays['actual_no_huddle'] = nh_plays['no_huddle'].astype(int)

        # Predictions (leave-coach-out cross-fit by default; in-sample fallback)
        predictions = self._predict_component(
            nh_plays, self.no_huddle_features, 'actual_no_huddle',
            self.no_huddle_model, self.no_huddle_encoders, self.no_huddle_imputers,
            self.no_huddle_params, objective='binary:logistic', is_classifier=True)
        nh_plays['predicted_no_huddle_rate'] = predictions
        # Per-play Bernoulli noise phat(1-phat); summed -> binomial var of gene mean
        nh_plays['phat_var'] = predictions * (1 - predictions)

        # Group by coach-year
        coach_nh = nh_plays.groupby(['head_coach', 'season']).agg({
            'actual_no_huddle': 'mean',
            'predicted_no_huddle_rate': 'mean',
            'phat_var': 'sum',
            'play_id': 'count'
        }).rename(columns={'play_id': 'no_huddle_plays',
                           'phat_var': 'no_huddle_phat_var_sum'})

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

        # Actual pace (target) must exist before cross-fit prediction
        pace_plays['actual_pace'] = pace_plays['seconds_between_plays'].astype(float)

        # Predictions (leave-coach-out cross-fit by default; in-sample fallback)
        predictions = self._predict_component(
            pace_plays, self.pace_features, 'actual_pace',
            self.pace_model, self.pace_encoders, self.pace_imputers,
            self.pace_params, objective='reg:squarederror', is_classifier=False)
        pace_plays['predicted_pace'] = predictions
        # Per-play squared residual; summed -> sampling var of the gene mean
        # (regression analogue of the binomial phat(1-phat) noise term).
        pace_plays['resid_var'] = (pace_plays['actual_pace'] - predictions) ** 2

        # Group by coach-year
        coach_pace = pace_plays.groupby(['head_coach', 'season']).agg({
            'actual_pace': 'mean',
            'predicted_pace': 'mean',
            'resid_var': 'sum',
            'play_id': 'count'
        }).rename(columns={'play_id': 'pace_plays',
                           'resid_var': 'pace_resid_var_sum'})

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

        # No hard min-plays gate: low-n coach-years carry high sampling variance
        # and are automatically down-weighted by the reliability weighting below
        # (data-driven, replacing the old arbitrary play-count cutoff).

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

        # Reliability-weighted composite (WS4): weight each sub-gene's z-score by
        # its reliability rel = tau2/(tau2+samp_var) with tau2 by DerSimonian-
        # Laird (shared helper in utils.parsimony). Reliability is scale-invariant
        # so it is computed from the raw genes; the weighted mean runs over the
        # z-scores so the rate-based no-huddle and seconds-based pace components
        # stay comparable.
        comp_specs = [
            ('no_huddle_gene', 'no_huddle_plays', 'no_huddle_phat_var_sum'),
            ('pace_gene', 'pace_plays', 'pace_resid_var_sum'),
        ]
        composite, self.component_reliability, present = (
            parsimony.reliability_weighted_composite(
                tempo_df, comp_specs, rel_floor=self.rel_floor,
                value_suffix='_zscore', logger=logger))
        if present:
            tempo_df['composite_tempo'] = composite

            # Track how many sub-components contributed
            zscore_cols = [f'{g}_zscore' for g in present
                           if f'{g}_zscore' in tempo_df.columns]
            tempo_df['tempo_components'] = tempo_df[zscore_cols].notna().sum(axis=1)

            # Z-score the composite
            ct_valid = tempo_df['composite_tempo'].dropna()
            if len(ct_valid) > 1:
                ct_mean = ct_valid.mean()
                ct_std = ct_valid.std()
                tempo_df['composite_tempo_zscore'] = (
                    (tempo_df['composite_tempo'] - ct_mean) / ct_std
                    if ct_std > 0 else 0.0
                )
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
            'component_reliability': getattr(self, 'component_reliability', {}),
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
                        help='(deprecated; reliability weighting now handles low-n, no hard gate)')
    parser.add_argument('--rel_floor', type=float, default=0.1,
                        help='Reliability floor below which a sub-gene gets zero weight in the composite')
    parser.add_argument('--no_crossfit', action='store_true',
                        help='Score in-sample with the persisted all-data model instead of '
                             'leave-coach-out cross-fit (faster; for debugging only)')
    parser.add_argument('--cv_splits', type=int, default=5,
                        help='GroupKFold splits for cross-fit scoring (default 5)')

    args = parser.parse_args()

    print("=" * 80)
    print("COACHING TEMPO GENE CALCULATOR")
    print("=" * 80)
    print(f"Years: {args.start_year} - {args.end_year}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")

    try:
        calculator = TempoGeneCalculator(rel_floor=args.rel_floor)
        calculator.use_crossfit = not args.no_crossfit
        calculator.cv_splits = args.cv_splits
        logger.info("Scoring mode: %s",
                    "in-sample (all-data model)" if args.no_crossfit
                    else f"leave-coach-out cross-fit (k={args.cv_splits})")

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
