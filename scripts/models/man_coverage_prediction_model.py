#!/usr/bin/env python3
"""
Man Coverage Prediction Model

Builds an XGBoost classifier to predict whether a defense plays man or zone
coverage based on game context and situational factors. Uses play-by-play data
from 2018-2024 (when defense_man_zone_type tracking data became available).

Target Variable:
- 0: Zone coverage
- 1: Man coverage

Includes only pass plays with valid defense_man_zone_type data.

Usage:
    python man_coverage_prediction_model.py [--test_size 0.2] [--random_state 42]
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import warnings
import pickle
import json
import gc
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.model_features import get_defensive_scheme_predictor_features, get_categorical_features, validate_features
from utils import model_pipeline as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# defense_man_zone_type available from 2018+
START_YEAR = 2018


class ManCoverageDataProcessor:
    """Processes play-by-play data for man vs zone coverage classification"""

    def __init__(self, data_dir: str = "data/raw/play_by_play"):
        self.data_dir = Path(data_dir)
        self.categorical_features = get_categorical_features()
        self.label_encoders = {}

    def load_and_filter_plays(self) -> pd.DataFrame:
        """
        Load play-by-play files from 2018+ and filter for pass plays
        with valid man/zone coverage data.

        Returns:
            Combined DataFrame with relevant plays
        """
        logger.info(f"Loading and filtering plays for man/zone analysis from {START_YEAR}+...")

        all_data = []

        pbp_files = sorted(self.data_dir.glob("play_by_play_*.csv"))
        if not pbp_files:
            raise FileNotFoundError(f"No play-by-play files found in {self.data_dir}")

        needed_features = get_defensive_scheme_predictor_features()
        key_columns = ['play_type', 'down', 'defense_man_zone_type', 'punt_attempt',
                       'field_goal_attempt', 'qb_scramble', 'desc',
                       'game_id', 'play_id']
        columns_to_keep = list(set(needed_features + key_columns + ['posteam', 'defteam', 'season']))

        for file_path in pbp_files:
            year = int(file_path.stem.split('_')[-1])
            if year < START_YEAR:
                continue

            logger.info(f"Processing {year} season data...")

            try:
                header_df = pd.read_csv(file_path, nrows=0)

                if 'defense_man_zone_type' not in header_df.columns:
                    logger.warning(f"No 'defense_man_zone_type' column in {year} data - skipping")
                    continue

                available_cols = [col for col in columns_to_keep if col in header_df.columns]

                season_df = pd.read_csv(file_path, usecols=available_cols, low_memory=False)

                # Filter: pass plays with MAN_COVERAGE or ZONE_COVERAGE
                filtered = season_df[
                    (season_df['play_type'] == 'pass') &
                    (
                        (season_df['down'].isin([1, 2, 3])) |
                        ((season_df['down'] == 4) &
                         (season_df['punt_attempt'] != 1) &
                         (season_df['field_goal_attempt'] != 1))
                    ) &
                    (season_df['defense_man_zone_type'].isin(['MAN_COVERAGE', 'ZONE_COVERAGE']))
                ].copy()

                if not filtered.empty:
                    logger.info(f"Found {len(filtered):,} pass plays with man/zone data in {year}")
                    man_count = (filtered['defense_man_zone_type'] == 'MAN_COVERAGE').sum()
                    zone_count = (filtered['defense_man_zone_type'] == 'ZONE_COVERAGE').sum()
                    logger.info(f"  Man: {man_count:,} ({man_count/len(filtered)*100:.1f}%), "
                              f"Zone: {zone_count:,} ({zone_count/len(filtered)*100:.1f}%)")
                    all_data.append(filtered)
                else:
                    logger.warning(f"No valid man/zone coverage data found in {year}")

                del season_df
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No defense_man_zone_type data found in any files")

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total pass plays with man/zone data: {len(combined):,}")

        del all_data
        gc.collect()

        return combined

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable for man vs zone coverage.

        Target:
        - 0: Zone coverage
        - 1: Man coverage
        """
        logger.info("Creating target variable for man/zone classification...")

        df['is_man'] = (df['defense_man_zone_type'] == 'MAN_COVERAGE').astype(int)

        target_counts = df['is_man'].value_counts().sort_index()
        logger.info(f"Target distribution:")
        logger.info(f"  Zone (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"  Man (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling."""
        logger.info("Preparing features for modeling...")

        basic_features = get_defensive_scheme_predictor_features()
        validation = validate_features(df.columns.tolist())
        available_features = [f for f in basic_features if f in validation['available']]

        logger.info(f"Using {len(available_features)} features out of {len(basic_features)} possible")

        if validation['missing']:
            logger.info(f"Missing features: {validation['missing'][:10]}...")

        feature_df = df[available_features + ['is_man']].copy()
        return feature_df, available_features

    # Categorical encoding and SVD imputation are provided by
    # utils.model_pipeline (split_encode_impute) so that training and gene
    # calculation share one leakage-free, train/serve-consistent implementation.


class ManCoverageModel:
    """XGBoost classifier for man vs zone coverage prediction"""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def get_param_distributions(self) -> Dict:
        """Get parameter distributions for RandomizedSearchCV"""
        return {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.01, 0.1, 1],
            'reg_lambda': [1, 1.5, 2, 5],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.5]
        }

    def train_with_hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                                         n_iter: int = 100, cv_folds: int = 3) -> None:
        """Train classifier with hyperparameter tuning."""
        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=self.random_state,
            n_jobs=-1
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=self.get_param_distributions(),
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_

        logger.info(f"Best cross-validation AUC: {random_search.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str]) -> Dict:
        """Evaluate classifier performance and return metrics."""
        logger.info("Evaluating model performance...")

        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        auc_score = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"Test AUC Score: {auc_score:.4f}")

        report = classification_report(y_test, y_pred, output_dict=True)

        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(self.model.feature_importances_)],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return {
            'auc_score': auc_score,
            'classification_report': report,
            'feature_importance': feature_importance,
            'predictions_proba': y_pred_proba,
            'predictions': y_pred
        }

    def save_model(self, filepath, feature_names, label_encoders=None, metrics=None, imputers=None):
        """Persist model + metadata + encoders + imputer via the shared pipeline."""
        metadata_extra = {
            'model_type': 'XGBoost Man Coverage Classification',
            'target_encoding': {
                '0': 'Zone Coverage',
                '1': 'Man Coverage'
            },
        }
        mp.persist_model(self.model, filepath, feature_names, metadata_extra,
                         label_encoders=label_encoders, metrics=metrics,
                         imputers=imputers, logger=logger)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train man vs zone coverage classification model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--n_iter', type=int, default=100,
                        help='Number of hyperparameter combinations to try (default: 100)')
    parser.add_argument('--cv_folds', type=int, default=3,
                        help='Number of CV folds (default: 3)')

    args = parser.parse_args()

    print("=" * 80)
    print("MAN VS ZONE COVERAGE CLASSIFICATION MODEL TRAINING")
    print("=" * 80)
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Hyperparameter search iterations: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print(f"Data range: {START_YEAR}-2024 (pass plays only)")
    print("=" * 80 + "\n")

    try:
        processor = ManCoverageDataProcessor()

        logger.info("Step 1: Loading man/zone coverage data...")
        data = processor.load_and_filter_plays()

        logger.info("Step 2: Creating target variable...")
        data_with_target = processor.create_target_variable(data)

        logger.info("Step 3: Preparing features...")
        feature_data, feature_names = processor.prepare_features(data_with_target)

        feature_data = feature_data.dropna(subset=['is_man'])
        logger.info(f"Final dataset size: {len(feature_data):,} plays")

        # Apply any persisted stability-selected feature subset (parsimony).
        feature_names = mp.apply_feature_selection(feature_names, "models/man_coverage/man_coverage_prediction_model")

        # Leakage-free split + encode + impute (encoders/imputer fit on train only).
        logger.info("Split, encode, and impute (fit on train only)...")
        split = mp.split_encode_impute(
            feature_data, feature_names, target_col='is_man',
            categorical_features=processor.categorical_features,
            test_size=args.test_size, random_state=args.random_state, stratify=True)
        X_train_imputed, X_test_imputed = split.X_train, split.X_test
        y_train, y_test = split.y_train, split.y_test
        feature_names = split.feature_names
        processor.label_encoders = split.label_encoders
        imputers = split.imputers
        logger.info(f"Training set: {len(X_train_imputed):,} samples")
        logger.info(f"Test set: {len(X_test_imputed):,} samples")

        logger.info("Step 7: Training model with hyperparameter tuning...")
        model = ManCoverageModel(random_state=args.random_state)
        model.train_with_hyperparameter_tuning(
            X_train_imputed, y_train,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds
        )

        logger.info("Step 8: Evaluating model...")
        results = model.evaluate(X_test_imputed, y_test, feature_names)

        logger.info("Step 9: Saving model...")
        model_dir = Path("models/man_coverage")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = str(model_dir / "man_coverage_prediction_model")
        model.save_model(model_filepath, feature_names, processor.label_encoders, metrics=results, imputers=imputers)

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Test AUC Score: {results['auc_score']:.4f}")
        print(f"Best Parameters: {model.best_params}")
        print("\nTop 10 Feature Importance:")
        for i, row in results['feature_importance'].head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        print(f"\nModel saved to: {model_filepath}")
        print("=" * 80)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise


if __name__ == "__main__":
    main()
