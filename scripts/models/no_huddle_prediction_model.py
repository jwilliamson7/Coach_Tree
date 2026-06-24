#!/usr/bin/env python3
"""
No-Huddle Formation Prediction Model

Builds an XGBoost model to predict whether NFL teams will use no-huddle
based on game context and situational factors. Uses play-by-play data from 1999-2024
to train the model with basic features only (no advanced analytics).

Target Variable:
- 0: Huddle (standard)
- 1: No-huddle

Includes all 1st, 2nd, and 3rd downs plus 4th downs where the play was not
a special teams play or fake punt/field goal.

Usage:
    python no_huddle_prediction_model.py [--test_size 0.2] [--random_state 42]
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
from utils.model_features import get_no_huddle_predictor_features, get_categorical_features, validate_features
from utils import model_pipeline as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NoHuddleDataProcessor:
    """Processes play-by-play data for no-huddle prediction modeling"""

    def __init__(self, data_dir: str = "data/raw/play_by_play"):
        self.data_dir = Path(data_dir)
        self.categorical_features = get_categorical_features()
        self.label_encoders = {}

    def load_and_filter_plays(self) -> pd.DataFrame:
        """
        Load all play-by-play files and filter for plays where no-huddle decision is relevant.

        Includes:
        - All 1st, 2nd, and 3rd down plays (run or pass)
        - 4th down plays that are not special teams or fake punts/field goals

        Returns:
            Combined DataFrame with relevant plays from all seasons
        """
        logger.info("Loading and filtering plays for no-huddle analysis from all seasons...")

        all_data = []

        pbp_files = list(self.data_dir.glob("play_by_play_*.csv"))
        if not pbp_files:
            raise FileNotFoundError(f"No play-by-play files found in {self.data_dir}")

        pbp_files.sort()

        needed_features = get_no_huddle_predictor_features()
        key_columns = ['play_type', 'down', 'no_huddle', 'punt_attempt', 'field_goal_attempt',
                       'qb_scramble', 'desc']
        columns_to_keep = list(set(needed_features + key_columns + ['posteam', 'defteam', 'season']))

        for file_path in pbp_files:
            year = file_path.stem.split('_')[-1]
            logger.info(f"Processing {year} season data...")

            try:
                chunk_size = 25000
                season_data = []

                header_df = pd.read_csv(file_path, nrows=0)

                if 'no_huddle' not in header_df.columns:
                    logger.warning(f"No 'no_huddle' column in {year} data - skipping")
                    continue

                available_cols = [col for col in columns_to_keep if col in header_df.columns]

                for chunk in pd.read_csv(file_path, usecols=available_cols, chunksize=chunk_size, low_memory=False):
                    filtered = chunk[
                        (chunk['play_type'].isin(['run', 'pass'])) &
                        (
                            (chunk['down'].isin([1, 2, 3])) |
                            ((chunk['down'] == 4) &
                             (chunk['punt_attempt'] != 1) &
                             (chunk['field_goal_attempt'] != 1))
                        ) &
                        (chunk['no_huddle'].notna())
                    ].copy()

                    if not filtered.empty:
                        season_data.append(filtered)

                if season_data:
                    season_df = pd.concat(season_data, ignore_index=True)
                    logger.info(f"Found {len(season_df):,} plays with no-huddle data in {year}")
                    all_data.append(season_df)
                    del season_data, season_df
                else:
                    logger.warning(f"No plays with no-huddle data found in {year}")

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No no-huddle data found in any files")

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total plays with no-huddle data across all seasons: {len(combined):,}")

        del all_data
        gc.collect()

        return combined

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable for no-huddle.

        Target:
        - 0: Huddle (standard)
        - 1: No-huddle
        """
        logger.info("Creating target variable for no-huddle classification...")

        df['is_no_huddle'] = df['no_huddle'].astype(int)

        target_counts = df['is_no_huddle'].value_counts().sort_index()
        logger.info(f"Target distribution:")
        logger.info(f"  Huddle (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"  No-Huddle (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling using basic features only."""
        logger.info("Preparing features for modeling...")

        basic_features = get_no_huddle_predictor_features()
        validation = validate_features(df.columns.tolist())
        available_features = [f for f in basic_features if f in validation['available']]

        logger.info(f"Using {len(available_features)} features out of {len(basic_features)} possible")

        if validation['missing']:
            logger.info(f"Missing features: {validation['missing'][:10]}...")

        feature_df = df[available_features + ['is_no_huddle']].copy()
        return feature_df, available_features

    # Categorical encoding and SVD imputation are provided by
    # utils.model_pipeline (split_encode_impute) so that training and gene
    # calculation share one leakage-free, train/serve-consistent implementation.


class NoHuddleModel:
    """XGBoost model for no-huddle prediction"""

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
                                         n_iter: int = 20, cv_folds: int = 3) -> None:
        """Train model with hyperparameter tuning using RandomizedSearchCV."""
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
        """Evaluate model performance and return metrics."""
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
            'model_type': 'XGBoost No-Huddle Prediction',
            'target_encoding': {
                '0': 'Huddle',
                '1': 'No-Huddle'
            },
        }
        mp.persist_model(self.model, filepath, feature_names, metadata_extra,
                         label_encoders=label_encoders, metrics=metrics,
                         imputers=imputers, logger=logger)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train no-huddle prediction model')
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
    print("NO-HUDDLE PREDICTION MODEL TRAINING")
    print("=" * 80)
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Hyperparameter search iterations: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print("=" * 80 + "\n")

    try:
        processor = NoHuddleDataProcessor()

        logger.info("Step 1: Loading no-huddle data...")
        data = processor.load_and_filter_plays()

        logger.info("Step 2: Creating target variable...")
        data_with_target = processor.create_target_variable(data)

        logger.info("Step 3: Preparing features...")
        feature_data, feature_names = processor.prepare_features(data_with_target)

        feature_data = feature_data.dropna(subset=['is_no_huddle'])
        logger.info(f"Final dataset size: {len(feature_data):,} plays with no-huddle data")

        # Apply any persisted stability-selected feature subset (parsimony).
        feature_names = mp.apply_feature_selection(feature_names, "models/no_huddle/no_huddle_prediction_model")

        # Leakage-free split + encode + impute (encoders/imputer fit on train only).
        logger.info("Split, encode, and impute (fit on train only)...")
        split = mp.split_encode_impute(
            feature_data, feature_names, target_col='is_no_huddle',
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
        model = NoHuddleModel(random_state=args.random_state)
        model.train_with_hyperparameter_tuning(
            X_train_imputed, y_train,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds
        )

        logger.info("Step 8: Evaluating model...")
        results = model.evaluate(X_test_imputed, y_test, feature_names)

        logger.info("Step 9: Saving model...")
        model_dir = Path("models/no_huddle")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = str(model_dir / "no_huddle_prediction_model")
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
