#!/usr/bin/env python3
"""
Two-Point Conversion Decision Model

Builds an XGBoost model to predict whether NFL teams will attempt a two-point 
conversion vs. an extra point after scoring a touchdown. Uses play-by-play data 
from 1999-2024 to train the model with basic features only (no advanced analytics).

Target Variable:
- 0: Extra point attempt
- 1: Two-point conversion attempt

Includes all touchdown situations where teams have a choice between 1 or 2 points.

Usage:
    python two_point_conversion_model.py [--test_size 0.2] [--random_state 42]
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
from utils.model_features import get_two_point_predictor_features, get_categorical_features, validate_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TwoPointDataProcessor:
    """Processes play-by-play data for two-point conversion decision modeling"""
    
    def __init__(self, data_dir: str = "data/raw/play_by_play"):
        self.data_dir = Path(data_dir)
        self.categorical_features = get_categorical_features()
        self.label_encoders = {}
        
    def load_and_filter_touchdown_plays(self) -> pd.DataFrame:
        """
        Load all play-by-play files and filter for touchdown situations where 
        teams have a choice between extra point and two-point conversion.
        
        Filters each file individually before combining to handle large dataset size.
        
        Returns:
            Combined DataFrame with touchdown conversion attempts from all seasons
        """
        logger.info("Loading and filtering touchdown conversion attempts from all seasons...")
        
        conversion_data = []
        
        # Get all play-by-play CSV files
        pbp_files = list(self.data_dir.glob("play_by_play_*.csv"))
        if not pbp_files:
            raise FileNotFoundError(f"No play-by-play files found in {self.data_dir}")
        
        pbp_files.sort()  # Process in chronological order
        
        # Get feature columns we need plus key columns for filtering
        from utils.model_features import get_two_point_predictor_features
        needed_features = get_two_point_predictor_features()
        key_columns = ['extra_point_attempt', 'two_point_attempt', 'touchdown', 
                       'play_type', 'desc']
        columns_to_keep = list(set(needed_features + key_columns))
        
        for file_path in pbp_files:
            year = file_path.stem.split('_')[-1]
            logger.info(f"Processing {year} season data...")
            
            try:
                # Read file in smaller chunks to handle memory efficiently
                chunk_size = 25000  # Reduced chunk size for memory efficiency
                season_conversions = []
                
                # First read just the header to see available columns
                header_df = pd.read_csv(file_path, nrows=0)
                
                # Check if conversion columns exist
                if 'extra_point_attempt' not in header_df.columns or 'two_point_attempt' not in header_df.columns:
                    logger.warning(f"No conversion attempt columns in {year} data - skipping")
                    continue
                
                available_cols = [col for col in columns_to_keep if col in header_df.columns]
                
                for chunk in pd.read_csv(file_path, usecols=available_cols, chunksize=chunk_size, low_memory=False):
                    # Filter for conversion attempts (both extra point and two-point)
                    conversion_chunk = chunk[
                        (chunk['extra_point_attempt'] == 1) |  # Extra point attempts
                        (chunk['two_point_attempt'] == 1)     # Two-point attempts
                    ].copy()
                    
                    if not conversion_chunk.empty:
                        season_conversions.append(conversion_chunk)
                
                if season_conversions:
                    season_data = pd.concat(season_conversions, ignore_index=True)
                    logger.info(f"Found {len(season_data):,} conversion attempts in {year}")
                    conversion_data.append(season_data)
                    
                    # Clear intermediate data to save memory
                    del season_conversions
                    del season_data
                else:
                    logger.warning(f"No conversion attempts found in {year}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
        
        if not conversion_data:
            raise ValueError("No conversion attempt data found in any files")
        
        # Combine all seasons
        combined_data = pd.concat(conversion_data, ignore_index=True)
        logger.info(f"Total conversion attempts across all seasons: {len(combined_data):,}")
        
        # Clear the list to free memory
        del conversion_data
        gc.collect()
        
        return combined_data
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary target variable for two-point conversion decision.
        
        Target:
        - 0: Extra point attempt
        - 1: Two-point conversion attempt
        
        Args:
            df: DataFrame with play-by-play data
            
        Returns:
            DataFrame with 'is_two_point' target column added
        """
        logger.info("Creating target variable for two-point conversion decision...")
        
        df = df.copy()
        
        # Create target variable
        # 1 = two-point attempt, 0 = extra point attempt
        df['is_two_point'] = (df['two_point_attempt'] == 1).astype(int)
        
        # Verify we have valid data
        valid_attempts = (df['extra_point_attempt'] == 1) | (df['two_point_attempt'] == 1)
        df = df[valid_attempts].copy()
        
        # Log distribution
        target_counts = df['is_two_point'].value_counts().sort_index()
        logger.info(f"Target distribution:")
        logger.info(f"  Extra Point (0): {target_counts.get(0, 0):,} ({target_counts.get(0, 0)/len(df)*100:.1f}%)")
        logger.info(f"  Two-Point (1): {target_counts.get(1, 0):,} ({target_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for modeling using basic features only.
        
        Args:
            df: DataFrame with play-by-play data
            
        Returns:
            Tuple of (processed_df, feature_names)
        """
        logger.info("Preparing features for modeling...")
        
        # Get two-point specific predictor features (pre-play context only)
        basic_features = get_two_point_predictor_features()
        
        # Validate which features are available
        validation = validate_features(df.columns.tolist())
        available_features = [f for f in basic_features if f in validation['available']]
        
        logger.info(f"Using {len(available_features)} features out of {len(basic_features)} possible")
        
        if validation['missing']:
            logger.info(f"Missing features: {validation['missing'][:10]}...")  # Show first 10
        
        # Select available features
        feature_df = df[available_features + ['is_two_point']].copy()
        
        return feature_df, available_features
    
    def encode_categorical_features(self, df: pd.DataFrame, feature_names: List[str], 
                                  fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: DataFrame with features
            feature_names: List of feature column names
            fit: Whether to fit new encoders or use existing ones
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        categorical_cols = [col for col in feature_names if col in self.categorical_features and col in df.columns]
        
        if categorical_cols:
            logger.info(f"Encoding {len(categorical_cols)} categorical features...")
            
            for col in categorical_cols:
                if fit:
                    # Fit new encoder
                    le = LabelEncoder()
                    # Handle NaN values by treating them as a separate category
                    df[col] = df[col].astype(str)
                    df[col] = le.fit_transform(df[col])
                    self.label_encoders[col] = le
                else:
                    # Use existing encoder
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        df[col] = df[col].astype(str)
                        # Handle unseen categories
                        mask = df[col].isin(le.classes_)
                        df.loc[~mask, col] = 'unknown'
                        
                        # Add 'unknown' to encoder if not present
                        if 'unknown' not in le.classes_:
                            le.classes_ = np.append(le.classes_, 'unknown')
                        
                        df[col] = le.transform(df[col])
        
        # Check for any remaining non-numeric columns and handle them
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        non_numeric_cols = set(df.columns) - set(numeric_cols)
        
        # Remove is_two_point from non-numeric check if present
        if 'is_two_point' in non_numeric_cols:
            non_numeric_cols.remove('is_two_point')
        
        if non_numeric_cols:
            logger.warning(f"Found non-numeric columns: {non_numeric_cols}")
            # Drop non-numeric columns
            cols_to_drop = list(non_numeric_cols)
            if cols_to_drop:
                logger.info(f"Dropping non-numeric columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
        
        return df
    
    def impute_missing_values(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Impute missing values using SVD-based imputation.
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            Tuple of (imputed_X_train, imputed_X_test) as numpy arrays
        """
        logger.info("Imputing missing values using SVD...")
        
        # First, use simple imputation to handle any remaining NaNs
        simple_imputer = SimpleImputer(strategy='median')
        X_train_simple = simple_imputer.fit_transform(X_train)
        X_test_simple = simple_imputer.transform(X_test)
        
        # Then apply SVD imputation for more sophisticated handling
        # Use fewer components for efficiency with large dataset
        n_components = min(50, X_train.shape[1] - 1, X_train.shape[0] - 1)
        
        if n_components > 0:
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Fit SVD on training data
            X_train_svd = svd.fit_transform(X_train_simple)
            X_train_reconstructed = svd.inverse_transform(X_train_svd)
            
            # Transform test data
            X_test_svd = svd.transform(X_test_simple)
            X_test_reconstructed = svd.inverse_transform(X_test_svd)
            
            logger.info(f"SVD imputation completed with {n_components} components")
            logger.info(f"Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
            
            return X_train_reconstructed, X_test_reconstructed
        else:
            logger.warning("SVD imputation skipped due to insufficient dimensions")
            return X_train_simple, X_test_simple


class TwoPointModel:
    """XGBoost model for two-point conversion decision prediction"""
    
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
    
    def save_model(self, filepath: str, feature_names: List[str], label_encoders: Dict = None) -> None:
        """
        Save the trained model and metadata.
        
        Args:
            filepath: Path to save the model (without extension)
            feature_names: List of feature names used in training
            label_encoders: Dictionary of label encoders for categorical features
        """
        model_path = Path(filepath)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the XGBoost model in native format
        model_file = f"{filepath}.json"
        self.model.save_model(model_file)
        
        # Save model metadata
        metadata = {
            'best_params': self.best_params,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'model_type': 'XGBoost Two-Point Conversion Decision',
            'target_encoding': {
                '0': 'Extra Point',
                '1': 'Two-Point Conversion'
            }
        }
        
        metadata_file = f"{filepath}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save label encoders if provided
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
    parser = argparse.ArgumentParser(description='Train two-point conversion decision model')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    parser.add_argument('--n_iter', type=int, default=20,
                       help='Number of hyperparameter combinations to try (default: 20)')
    parser.add_argument('--cv_folds', type=int, default=3,
                       help='Number of CV folds (default: 3)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TWO-POINT CONVERSION DECISION MODEL TRAINING")
    print("=" * 80)
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Hyperparameter search iterations: {args.n_iter}")
    print(f"Cross-validation folds: {args.cv_folds}")
    print("=" * 80 + "\n")
    
    try:
        # Initialize data processor
        processor = TwoPointDataProcessor()
        
        # Load and filter conversion data
        logger.info("Step 1: Loading two-point conversion data...")
        conversion_data = processor.load_and_filter_touchdown_plays()
        
        # Create target variable
        logger.info("Step 2: Creating target variable...")
        data_with_target = processor.create_target_variable(conversion_data)
        
        # Prepare features
        logger.info("Step 3: Preparing features...")
        feature_data, feature_names = processor.prepare_features(data_with_target)
        
        # Remove rows with missing target
        feature_data = feature_data.dropna(subset=['is_two_point'])
        logger.info(f"Final dataset size: {len(feature_data):,} conversion attempts")
        
        # Encode categorical features before splitting
        logger.info("Step 4: Encoding categorical features...")
        feature_data_encoded = processor.encode_categorical_features(feature_data, feature_names + ['is_two_point'], fit=True)
        
        # Separate features and target
        X = feature_data_encoded.drop('is_two_point', axis=1)
        y = feature_data_encoded['is_two_point'].astype(int)
        
        # Train-test split (stratified)
        logger.info("Step 5: Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=args.test_size, 
            random_state=args.random_state,
            stratify=y
        )
        
        logger.info(f"Training set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Impute missing values
        logger.info("Step 6: Imputing missing values...")
        X_train_imputed, X_test_imputed = processor.impute_missing_values(X_train, X_test)
        
        # Initialize and train model
        logger.info("Step 7: Training model with hyperparameter tuning...")
        model = TwoPointModel(random_state=args.random_state)
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
        model_dir = Path("models/two_point")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_filepath = str(model_dir / "two_point_conversion_model")
        model.save_model(model_filepath, feature_names, processor.label_encoders)
        
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