#!/usr/bin/env python3
"""
Pace (Snap Timing) Prediction Model

Builds an XGBoost regression model to predict the expected seconds between
consecutive plays within a drive, based on game context and situational factors.
Uses play-by-play data from 1999-2024.

This is the first regression model in the project -- all other models are classifiers.

Target Variable:
- seconds_between_plays (continuous, typically 5-60 seconds)

Includes all 1st, 2nd, and 3rd downs plus 4th downs where the play was not
a special teams play. Excludes first play of each drive and timing outliers.

Usage:
    python pace_prediction_model.py [--test_size 0.2] [--random_state 42]
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.model_features import get_pace_predictor_features, get_categorical_features, validate_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaceDataProcessor:
    """Processes play-by-play data for pace (snap timing) regression modeling"""

    def __init__(self, data_dir: str = "data/raw/play_by_play"):
        self.data_dir = Path(data_dir)
        self.categorical_features = get_categorical_features()
        self.label_encoders = {}

    def load_and_filter_plays(self) -> pd.DataFrame:
        """
        Load all play-by-play files and filter for plays where pace can be measured.

        Loads full seasons to compute inter-play timing, then filters to
        offensive plays with valid timing intervals.

        Returns:
            Combined DataFrame with relevant plays and seconds_between_plays computed
        """
        logger.info("Loading and filtering plays for pace analysis from all seasons...")

        all_data = []

        pbp_files = list(self.data_dir.glob("play_by_play_*.csv"))
        if not pbp_files:
            raise FileNotFoundError(f"No play-by-play files found in {self.data_dir}")

        pbp_files.sort()

        needed_features = get_pace_predictor_features()
        # Extra columns needed for timing computation and filtering
        key_columns = ['play_type', 'down', 'punt_attempt', 'field_goal_attempt',
                       'qb_scramble', 'desc', 'game_id', 'play_id', 'drive',
                       'game_seconds_remaining']
        columns_to_keep = list(set(needed_features + key_columns))

        for file_path in pbp_files:
            year = file_path.stem.split('_')[-1]
            logger.info(f"Processing {year} season data...")

            try:
                header_df = pd.read_csv(file_path, nrows=0)

                if 'game_seconds_remaining' not in header_df.columns:
                    logger.warning(f"No 'game_seconds_remaining' column in {year} data - skipping")
                    continue

                available_cols = [col for col in columns_to_keep if col in header_df.columns]

                # Load full season (needed for inter-play timing computation)
                season_df = pd.read_csv(file_path, usecols=available_cols, low_memory=False)

                # Filter to offensive plays first
                offensive = season_df[
                    (season_df['play_type'].isin(['run', 'pass'])) &
                    (
                        (season_df['down'].isin([1, 2, 3])) |
                        ((season_df['down'] == 4) &
                         (season_df['punt_attempt'] != 1) &
                         (season_df['field_goal_attempt'] != 1))
                    )
                ].copy()

                if offensive.empty:
                    logger.warning(f"No offensive plays found in {year}")
                    del season_df
                    continue

                # Sort by game and play order
                offensive = offensive.sort_values(['game_id', 'play_id'], ascending=[True, True])

                # Compute seconds between consecutive plays within same game+drive
                offensive['prev_game_seconds'] = offensive.groupby(
                    ['game_id', 'drive'])['game_seconds_remaining'].shift(1)
                offensive['seconds_between_plays'] = (
                    offensive['prev_game_seconds'] - offensive['game_seconds_remaining']
                )

                # Filter to valid timing intervals
                valid_pace = offensive[
                    (offensive['seconds_between_plays'].notna()) &   # Not first play of drive
                    (offensive['seconds_between_plays'] > 0) &       # Positive interval
                    (offensive['seconds_between_plays'] <= 60)       # Within play clock limit
                ].copy()

                if not valid_pace.empty:
                    # Drop computation columns we no longer need
                    valid_pace = valid_pace.drop(columns=['prev_game_seconds'], errors='ignore')
                    logger.info(f"Found {len(valid_pace):,} plays with valid pace data in {year}")
                    all_data.append(valid_pace)
                else:
                    logger.warning(f"No valid pace data found in {year}")

                del season_df, offensive
                gc.collect()

            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue

        if not all_data:
            raise ValueError("No pace data found in any files")

        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total plays with pace data across all seasons: {len(combined):,}")

        del all_data
        gc.collect()

        return combined

    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for pace regression.

        Target: seconds_between_plays (already computed during loading)
        """
        logger.info("Creating target variable for pace regression...")

        df['pace_target'] = df['seconds_between_plays'].astype(float)

        logger.info(f"Target distribution:")
        logger.info(f"  Mean: {df['pace_target'].mean():.1f} seconds")
        logger.info(f"  Median: {df['pace_target'].median():.1f} seconds")
        logger.info(f"  Std: {df['pace_target'].std():.1f} seconds")
        logger.info(f"  Min: {df['pace_target'].min():.1f} seconds")
        logger.info(f"  Max: {df['pace_target'].max():.1f} seconds")

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for modeling using basic features only."""
        logger.info("Preparing features for modeling...")

        basic_features = get_pace_predictor_features()
        validation = validate_features(df.columns.tolist())
        available_features = [f for f in basic_features if f in validation['available']]

        logger.info(f"Using {len(available_features)} features out of {len(basic_features)} possible")

        if validation['missing']:
            logger.info(f"Missing features: {validation['missing'][:10]}...")

        feature_df = df[available_features + ['pace_target']].copy()
        return feature_df, available_features

    def encode_categorical_features(self, df: pd.DataFrame, feature_names: List[str],
                                    fit: bool = True) -> pd.DataFrame:
        """Encode categorical features using label encoding."""
        df = df.copy()

        categorical_cols = [col for col in feature_names if col in self.categorical_features and col in df.columns]

        if categorical_cols:
            logger.info(f"Encoding {len(categorical_cols)} categorical features...")

            for col in categorical_cols:
                if fit:
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        mask = df[col].isin(le.classes_)
                        df.loc[~mask, col] = 'unknown'
                        if 'unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'unknown')
                        df[col] = le.transform(df[col])

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = set(df.columns) - set(numeric_cols)

        if 'pace_target' in non_numeric_cols:
            non_numeric_cols.remove('pace_target')

        if non_numeric_cols:
            logger.warning(f"Found non-numeric columns: {non_numeric_cols}")
            df = df.drop(columns=list(non_numeric_cols))

        return df

    def impute_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Impute missing values using SVD-based imputation."""
        logger.info("Imputing missing values using SVD...")

        simple_imputer = SimpleImputer(strategy='median')
        X_train_simple = simple_imputer.fit_transform(X_train)
        X_test_simple = simple_imputer.transform(X_test)

        n_components = min(50, X_train.shape[1] - 1, X_train.shape[0] - 1)

        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            X_train_svd = svd.fit_transform(X_train_simple)
            X_train_reconstructed = svd.inverse_transform(X_train_svd)
            X_test_svd = svd.transform(X_test_simple)
            X_test_reconstructed = svd.inverse_transform(X_test_svd)

            logger.info(f"SVD imputation completed with {n_components} components")
            logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")

            return X_train_reconstructed, X_test_reconstructed
        else:
            logger.warning("SVD imputation skipped due to insufficient dimensions")
            return X_train_simple, X_test_simple


class PaceModel:
    """XGBoost regression model for pace (snap timing) prediction"""

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
        """
        Train regression model with hyperparameter tuning.

        Uses KFold (not StratifiedKFold) and neg_mean_squared_error scoring
        since this is a regression task.
        """
        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")

        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=self.random_state,
            n_jobs=-1
        )

        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=self.get_param_distributions(),
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_

        best_rmse = np.sqrt(-random_search.best_score_)
        logger.info(f"Best cross-validation RMSE: {best_rmse:.4f}")
        logger.info(f"Best parameters: {self.best_params}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray,
                 feature_names: List[str]) -> Dict:
        """Evaluate regression model performance and return metrics."""
        logger.info("Evaluating model performance...")

        y_pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(f"Test RMSE: {rmse:.4f} seconds")
        logger.info(f"Test MAE: {mae:.4f} seconds")
        logger.info(f"Test R-squared: {r2:.4f}")

        feature_importance = pd.DataFrame({
            'feature': feature_names[:len(self.model.feature_importances_)],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'feature_importance': feature_importance,
            'predictions': y_pred
        }

    def save_model(self, filepath: str, feature_names: List[str], label_encoders: Dict = None,
                   metrics: Dict = None) -> None:
        """Save the trained model and metadata."""
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_file = f"{filepath}.json"
        self.model.save_model(model_file)

        metadata = {
            'best_params': self.best_params,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'model_type': 'XGBoost Pace Regression',
            'target_encoding': {
                'unit': 'seconds_between_plays',
                'type': 'continuous'
            }
        }

        if metrics:
            metadata['performance_metrics'] = {
                k: v for k, v in metrics.items()
                if k not in ('predictions_proba', 'predictions', 'feature_importance')
            }

        metadata_file = f"{filepath}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        if label_encoders:
            encoders_file = f"{filepath}_encoders.pkl"
            with open(encoders_file, 'wb') as f:
                pickle.dump(label_encoders, f)

        logger.info(f"Model saved to {model_file}")
        logger.info(f"Metadata saved to {metadata_file}")
        if label_encoders:
            logger.info(f"Label encoders saved to {encoders_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train pace (snap timing) regression model')
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
    print("PACE (SNAP TIMING) REGRESSION MODEL TRAINING")
    print("=" * 80)
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Hyperparameter search iterations: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print("=" * 80 + "\n")

    try:
        processor = PaceDataProcessor()

        logger.info("Step 1: Loading pace data...")
        data = processor.load_and_filter_plays()

        logger.info("Step 2: Creating target variable...")
        data_with_target = processor.create_target_variable(data)

        logger.info("Step 3: Preparing features...")
        feature_data, feature_names = processor.prepare_features(data_with_target)

        feature_data = feature_data.dropna(subset=['pace_target'])
        logger.info(f"Final dataset size: {len(feature_data):,} plays with pace data")

        logger.info("Step 4: Encoding categorical features...")
        feature_data_encoded = processor.encode_categorical_features(
            feature_data, feature_names + ['pace_target'], fit=True)

        X = feature_data_encoded.drop('pace_target', axis=1)
        y = feature_data_encoded['pace_target'].astype(float)

        # No stratify for regression (continuous target)
        logger.info("Step 5: Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state
        )

        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")

        logger.info("Step 6: Imputing missing values...")
        X_train_imputed, X_test_imputed = processor.impute_missing_values(X_train, X_test)

        logger.info("Step 7: Training model with hyperparameter tuning...")
        model = PaceModel(random_state=args.random_state)
        model.train_with_hyperparameter_tuning(
            X_train_imputed, y_train,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds
        )

        logger.info("Step 8: Evaluating model...")
        results = model.evaluate(X_test_imputed, y_test, feature_names)

        logger.info("Step 9: Saving model...")
        model_dir = Path("models/pace")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = str(model_dir / "pace_prediction_model")
        model.save_model(model_filepath, feature_names, processor.label_encoders, metrics=results)

        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Test RMSE: {results['rmse']:.4f} seconds")
        print(f"Test MAE: {results['mae']:.4f} seconds")
        print(f"Test R-squared: {results['r2_score']:.4f}")
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
