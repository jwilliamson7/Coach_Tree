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
        columns_to_keep = list(set(needed_features + key_columns))

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

        if 'is_man' in non_numeric_cols:
            non_numeric_cols.remove('is_man')

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
            'model_type': 'XGBoost Man Coverage Classification',
            'target_encoding': {
                '0': 'Zone Coverage',
                '1': 'Man Coverage'
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

        logger.info("Step 4: Encoding categorical features...")
        feature_data_encoded = processor.encode_categorical_features(
            feature_data, feature_names + ['is_man'], fit=True)

        X = feature_data_encoded.drop('is_man', axis=1)
        y = feature_data_encoded['is_man'].astype(int)

        logger.info("Step 5: Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y
        )

        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")

        logger.info("Step 6: Imputing missing values...")
        X_train_imputed, X_test_imputed = processor.impute_missing_values(X_train, X_test)

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
        model.save_model(model_filepath, feature_names, processor.label_encoders, metrics=results)

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
