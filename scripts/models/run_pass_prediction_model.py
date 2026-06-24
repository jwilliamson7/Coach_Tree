#!/usr/bin/env python3
"""
Run vs Pass Prediction Model

Builds an XGBoost model to predict whether NFL teams will run or pass based on
game context and situational factors. Uses play-by-play data from 1999-2024
to train the model with basic features only (no advanced analytics).

Target Variable:
- 0: Run play (rush attempt)
- 1: Pass play (pass attempt)

Includes all 1st, 2nd, and 3rd downs plus 4th downs where the play was not
a special teams play or fake punt/field goal.

Usage:
    python run_pass_prediction_model.py [--test_size 0.2] [--random_state 42]
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
import gc  # For garbage collection
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.model_features import get_run_pass_predictor_features, get_categorical_features, validate_features
from utils import model_pipeline as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RunPassDataProcessor:
    """Processes play-by-play data for run vs pass prediction modeling"""
    
    def __init__(self, data_dir: str = "data/raw/play_by_play"):
        self.data_dir = Path(data_dir)
        self.categorical_features = get_categorical_features()
        self.label_encoders = {}
        
    def load_and_filter_run_pass_plays(self) -> pd.DataFrame:
        """
        Load all play-by-play files and filter for run/pass plays only.
        
        Includes:
        - All 1st, 2nd, and 3rd down plays
        - 4th down plays that are not special teams or fake punts/field goals
        
        Filters each file individually before combining to handle large dataset size.
        
        Returns:
            Combined DataFrame with run/pass plays from all seasons
        """
        logger.info("Loading and filtering run/pass plays from all seasons...")
        
        run_pass_data = []
        
        # Get all play-by-play CSV files
        pbp_files = list(self.data_dir.glob("play_by_play_*.csv"))
        if not pbp_files:
            raise FileNotFoundError(f"No play-by-play files found in {self.data_dir}")
        
        pbp_files.sort()  # Process in chronological order
        
        # Get feature columns we need plus key columns for filtering
        from utils.model_features import get_run_pass_predictor_features
        needed_features = get_run_pass_predictor_features()
        key_columns = ['play_type', 'down', 'qb_scramble', 'punt_attempt', 'field_goal_attempt', 'desc']
        columns_to_keep = list(set(needed_features + key_columns + ['posteam', 'defteam', 'season']))
        
        for file_path in pbp_files:
            year = file_path.stem.split('_')[-1]
            logger.info(f"Processing {year} season data...")
            
            try:
                # Read file in smaller chunks to handle memory efficiently
                chunk_size = 25000  # Reduced chunk size
                season_run_pass = []
                
                # First read just the header to see available columns
                header_df = pd.read_csv(file_path, nrows=0)
                available_cols = [col for col in columns_to_keep if col in header_df.columns]
                
                for chunk in pd.read_csv(file_path, usecols=available_cols, chunksize=chunk_size, low_memory=False):
                    # Filter for run/pass plays
                    # Include downs 1-3 and non-special teams 4th downs
                    run_pass_chunk = chunk[
                        (chunk['down'].isin([1, 2, 3])) |  # All 1st-3rd downs
                        ((chunk['down'] == 4) & 
                         (chunk['punt_attempt'] != 1) &      # Not punts
                         (chunk['field_goal_attempt'] != 1)  # Not field goals
                        )
                    ].copy()
                    
                    if not run_pass_chunk.empty:
                        season_run_pass.append(run_pass_chunk)
                
                if season_run_pass:
                    season_data = pd.concat(season_run_pass, ignore_index=True)
                    logger.info(f"Found {len(season_data):,} run/pass plays in {year}")
                    run_pass_data.append(season_data)
                    
                    # Clear intermediate data to save memory
                    del season_run_pass
                    del season_data
                else:
                    logger.warning(f"No run/pass plays found in {year}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not run_pass_data:
            raise ValueError("No run/pass data found in any files")
        
        # Combine all seasons
        combined_data = pd.concat(run_pass_data, ignore_index=True)
        logger.info(f"Total run/pass plays across all seasons: {len(combined_data):,}")
        
        # Clear the list to free memory
        del run_pass_data
        gc.collect()
        
        return combined_data
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable for run vs pass decision.
        
        Target:
        - 0: Run play (play_type="run" or identified from no_play description)
        - 1: Pass play (play_type="pass", qb_scramble=1, or identified from no_play description)
        
        Args:
            df: DataFrame with play-by-play data
            
        Returns:
            DataFrame with 'is_pass' target column added
        """
        logger.info("Creating target variable for run vs pass classification...")
        
        # Work with the original dataframe to avoid unnecessary copy
        # We'll only copy at the end after filtering
        
        # Initialize as unknown (-1)
        df['is_pass'] = -1
        
        # Identify pass plays (play_type="pass" includes sacks, or QB scrambles)
        pass_conditions = (
            (df['play_type'] == 'pass') | 
            (df['qb_scramble'] == 1)
        )
        df.loc[pass_conditions, 'is_pass'] = 1
        
        # Identify run plays (play_type="run" excludes scrambles)
        run_conditions = (
            (df['play_type'] == 'run')
        )
        df.loc[run_conditions, 'is_pass'] = 0
        
        # Process no_play plays with penalty descriptions
        if 'desc' in df.columns:
            no_play_mask = (df['play_type'] == 'no_play') & (df['is_pass'] == -1)
            no_play_count = no_play_mask.sum()
            
            if no_play_count > 0:
                logger.info(f"Processing {no_play_count:,} no_play plays for run/pass classification...")
                
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
                desc_lower = df.loc[no_play_mask, 'desc'].str.lower()
                
                # Check for pass keywords
                pass_pattern = '|'.join(pass_keywords)
                pass_matches = desc_lower.str.contains(pass_pattern, na=False, regex=True)
                
                # Check for run keywords  
                run_pattern = '|'.join(run_keywords)
                run_matches = desc_lower.str.contains(run_pattern, na=False, regex=True)
                
                # Set is_pass based on keyword matches
                # Use the indices from the original mask
                pass_indices = df.index[no_play_mask][pass_matches]
                run_indices = df.index[no_play_mask][run_matches & ~pass_matches]
                
                df.loc[pass_indices, 'is_pass'] = 1
                df.loc[run_indices, 'is_pass'] = 0
                
                # Log how many no_play plays were classified
                classified_count = len(pass_indices) + len(run_indices)
                logger.info(f"  Classified {classified_count:,} no_play plays ({classified_count/no_play_count*100:.1f}%)")
                logger.info(f"    - {len(pass_indices):,} as pass plays")
                logger.info(f"    - {len(run_indices):,} as run plays")
        
        # Remove plays that are neither clear runs nor passes
        # Now make a copy with only the rows we need
        df_filtered = df[df['is_pass'] != -1].copy()
        
        # Log distribution
        target_counts = df_filtered['is_pass'].value_counts().sort_index()
        logger.info(f"Target distribution:")
        logger.info(f"  Run (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df_filtered)*100:.1f}%)")
        logger.info(f"  Pass (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df_filtered)*100:.1f}%)")
        
        return df_filtered
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for modeling using basic features only.
        
        Args:
            df: DataFrame with play-by-play data
            
        Returns:
            Tuple of (processed_df, feature_names)
        """
        logger.info("Preparing features for modeling...")
        
        # Get run vs pass specific predictor features (pre-play context only)
        basic_features = get_run_pass_predictor_features()
        
        # Validate which features are available
        validation = validate_features(df.columns.tolist())
        available_features = [f for f in basic_features if f in validation['available']]
        
        logger.info(f"Using {len(available_features)} features out of {len(basic_features)} possible")
        
        if validation['missing']:
            logger.info(f"Missing features: {validation['missing'][:10]}...")  # Show first 10
        
        # Select available features
        feature_df = df[available_features + ['is_pass']].copy()
        
        return feature_df, available_features
    
    # Categorical encoding and SVD imputation are provided by
    # utils.model_pipeline (split_encode_impute) so that training and gene
    # calculation share one leakage-free, train/serve-consistent implementation.


class RunPassModel:
    """XGBoost model for run vs pass prediction"""
    
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
                                       n_iter: int = 50, cv_folds: int = 3) -> None:
        """
        Train model with hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training targets
            n_iter: Number of parameter combinations to try
            cv_folds: Number of cross-validation folds
        """
        logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
        
        # Base XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Randomized search
        param_distributions = self.get_param_distributions()
        
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        # Fit the model
        random_search.fit(X_train, y_train)
        
        # Store best model and parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        logger.info(f"Best cross-validation AUC: {random_search.best_score_:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                feature_names: List[str]) -> Dict:
        """
        Evaluate model performance and return metrics.
        
        Args:
            X_test: Test features
            y_test: Test targets  
            feature_names: Names of features for importance analysis
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Test AUC Score: {auc_score:.4f}")
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Feature importance
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
                   metrics: Dict = None, imputers: Dict = None) -> None:
        """Persist model + metadata + encoders + imputer via the shared pipeline."""
        metadata_extra = {
            'model_type': 'XGBoost Run vs Pass Prediction',
            'target_encoding': {'0': 'Run', '1': 'Pass'},
        }
        mp.persist_model(self.model, filepath, feature_names, metadata_extra,
                         label_encoders=label_encoders, metrics=metrics,
                         imputers=imputers, logger=logger)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train run vs pass prediction model')
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
    print("RUN vs PASS PREDICTION MODEL TRAINING")
    print("=" * 80)
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Hyperparameter search iterations: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print("=" * 80 + "\n")
    
    try:
        # Initialize data processor
        processor = RunPassDataProcessor()
        
        # Load and filter run/pass data
        logger.info("Step 1: Loading run/pass data...")
        run_pass_data = processor.load_and_filter_run_pass_plays()
        
        # Create target variable
        logger.info("Step 2: Creating target variable...")
        data_with_target = processor.create_target_variable(run_pass_data)
        
        # Prepare features
        logger.info("Step 3: Preparing features...")
        feature_data, feature_names = processor.prepare_features(data_with_target)
        
        # Remove rows with missing target
        feature_data = feature_data.dropna(subset=['is_pass'])
        logger.info(f"Final dataset size: {len(feature_data):,} run/pass plays")
        
        # Apply any persisted stability-selected feature subset (parsimony).
        feature_names = mp.apply_feature_selection(feature_names, "models/run_pass/run_pass_prediction_model")

        # Leakage-free split + encode + impute (encoders/imputer fit on train only).
        logger.info("Step 4-6: Split, encode, and impute (fit on train only)...")
        split = mp.split_encode_impute(
            feature_data, feature_names, target_col='is_pass',
            categorical_features=processor.categorical_features,
            test_size=args.test_size, random_state=args.random_state, stratify=True)
        X_train_imputed, X_test_imputed = split.X_train, split.X_test
        y_train, y_test = split.y_train, split.y_test
        feature_names = split.feature_names
        processor.label_encoders = split.label_encoders
        imputers = split.imputers
        logger.info(f"Training set: {len(X_train_imputed):,} samples")
        logger.info(f"Test set: {len(X_test_imputed):,} samples")

        # Initialize and train model
        logger.info("Step 7: Training model with hyperparameter tuning...")
        model = RunPassModel(random_state=args.random_state)
        model.train_with_hyperparameter_tuning(
            X_train_imputed, y_train,
            n_iter=args.n_iter,
            cv_folds=args.cv_folds
        )
        
        # Evaluate model
        logger.info("Step 8: Evaluating model...")
        results = model.evaluate(X_test_imputed, y_test, feature_names)
        
        # Save model
        logger.info("Step 9: Saving model...")
        model_dir = Path("models/run_pass")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = str(model_dir / "run_pass_prediction_model")
        model.save_model(model_filepath, feature_names, processor.label_encoders,
                         metrics=results, imputers=imputers)
        
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