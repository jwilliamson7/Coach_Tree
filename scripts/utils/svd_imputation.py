#!/usr/bin/env python3
"""
SVD Imputation Script for Combined Final Dataset

This script performs SVD-based matrix completion on the combined_final_dataset.csv
to handle missing values, based on the implementation from Cell 10 of 
analysis/Williamson-Jon-CoachTenureModels-v5.ipynb.

Usage:
    python scripts/svd_imputation.py
    python scripts/svd_imputation.py --input data/final/combined_final_dataset.csv --output data/final/imputed_final_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SVDImputer:
    """SVD-based matrix completion for missing value imputation"""
    
    def __init__(self, n_components=50, max_iter=100, tol=1e-4, verbose=True):
        """
        Initialize SVD Imputer
        
        Args:
            n_components: Number of SVD components to use
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
            verbose: Print progress information
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
    def fit_transform(self, X):
        """
        Apply SVD-based matrix completion for missing value imputation
        
        Args:
            X: Input matrix (pandas DataFrame or numpy array)
            
        Returns:
            Imputed matrix (same type as input)
        """
        # Convert to numpy array for processing
        X_array = X.values if hasattr(X, 'values') else X
        X_filled = X_array.copy()
        
        # Get mask of missing values
        missing_mask = np.isnan(X_array)
        
        if not missing_mask.any():
            if self.verbose:
                print("No missing values found, returning original data")
            return X
        
        total_missing = missing_mask.sum()
        if self.verbose:
            print(f"Found {total_missing:,} missing values ({total_missing/X_array.size*100:.1f}% of data)")
        
        # Initialize missing values with column means
        col_means = np.nanmean(X_array, axis=0)
        for j in range(X_array.shape[1]):
            X_filled[missing_mask[:, j], j] = col_means[j]
        
        # Determine number of components
        n_components = min(self.n_components, X_filled.shape[0] - 1, X_filled.shape[1] - 1)
        if self.verbose:
            print(f"Using {n_components} SVD components")
        
        # Iterative SVD imputation
        for iteration in range(self.max_iter):
            # Store previous iteration for convergence check
            X_old = X_filled[missing_mask].copy()
            
            # Standardize for SVD
            X_mean = np.mean(X_filled, axis=0)
            X_std = np.std(X_filled, axis=0)
            X_std[X_std == 0] = 1  # Avoid division by zero
            X_standardized = (X_filled - X_mean) / X_std
            
            # Apply SVD
            try:
                U, s, Vt = np.linalg.svd(X_standardized, full_matrices=False)
                
                # Keep only top components
                U_k = U[:, :n_components]
                s_k = s[:n_components]
                Vt_k = Vt[:n_components, :]
                
                # Reconstruct matrix
                X_reconstructed = U_k @ np.diag(s_k) @ Vt_k
                
                # Transform back to original scale
                X_reconstructed = X_reconstructed * X_std + X_mean
                
                # Update only missing values
                X_filled[missing_mask] = X_reconstructed[missing_mask]
                
            except np.linalg.LinAlgError:
                if self.verbose:
                    print("SVD convergence issue, using mean imputation fallback")
                # Fallback to mean imputation if SVD fails
                for j in range(X_array.shape[1]):
                    col_mean = np.nanmean(X_array[:, j])
                    X_filled[missing_mask[:, j], j] = col_mean
                break
            
            # Check convergence
            if iteration > 0:
                diff = np.mean((X_filled[missing_mask] - X_old) ** 2)
                if diff < self.tol:
                    if self.verbose:
                        print(f"SVD imputation converged after {iteration + 1} iterations")
                    break
                elif self.verbose and (iteration + 1) % 10 == 0:
                    print(f"  Iteration {iteration + 1}: MSE = {diff:.6f}")
        else:
            if self.verbose:
                print(f"SVD imputation completed {self.max_iter} iterations (max reached)")
        
        # Return same type as input
        if hasattr(X, 'values'):
            return pd.DataFrame(X_filled, columns=X.columns, index=X.index)
        else:
            return X_filled


def preprocess_features(df, target_col='Win_Pct', exclude_cols=None, n_components=50, verbose=True):
    """
    Preprocess features with SVD imputation and standardization
    
    Args:
        df: Input DataFrame
        target_col: Target column name to exclude from features
        exclude_cols: Additional columns to exclude from features
        n_components: Number of SVD components
        verbose: Print progress information
        
    Returns:
        processed_df: DataFrame with imputed and standardized features
        scaler: Fitted StandardScaler for non-normalized features
        imputer: Fitted SVDImputer
    """
    if exclude_cols is None:
        exclude_cols = ['Team', 'Year']
    
    # Remove target and exclude columns from features
    all_exclude = exclude_cols + [target_col] if target_col in df.columns else exclude_cols
    feature_cols = [col for col in df.columns if col not in all_exclude]
    
    if verbose:
        print(f"Processing {len(feature_cols)} features")
    
    # Extract features
    X = df[feature_cols].copy()
    
    # Identify normalized vs non-normalized features
    normalized_cols = [col for col in X.columns if col.endswith('_Norm')]
    non_normalized_cols = [col for col in X.columns if not col.endswith('_Norm')]
    
    if verbose:
        print(f"  Already normalized features: {len(normalized_cols)}")
        print(f"  Features needing standardization: {len(non_normalized_cols)}")
        
        # Check missing value distribution
        missing_counts = X.isnull().sum()
        features_with_missing = missing_counts[missing_counts > 0]
        print(f"  Features with missing values: {len(features_with_missing)}")
    
    # Apply SVD imputation
    if verbose:
        print(f"\\nStarting SVD-based imputation...")
    
    imputer = SVDImputer(n_components=n_components, verbose=verbose)
    X_imputed = imputer.fit_transform(X)
    
    # Apply standardization only to non-normalized features
    scaler = StandardScaler()
    X_processed = X_imputed.copy()
    
    if non_normalized_cols:
        if verbose:
            print(f"\\nStandardizing {len(non_normalized_cols)} non-normalized features...")
        X_processed[non_normalized_cols] = scaler.fit_transform(X_imputed[non_normalized_cols])
    
    # Verify no missing values remain
    missing_after = X_processed.isnull().sum().sum()
    if verbose:
        print(f"\\nPreprocessing completed:")
        print(f"  Missing values remaining: {missing_after}")
        print(f"  Features standardized: {len(non_normalized_cols)}")
        print(f"  Features kept as-is: {len(normalized_cols)}")
    
    # Reconstruct full DataFrame
    processed_df = df.copy()
    processed_df[feature_cols] = X_processed
    
    return processed_df, scaler, imputer


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Apply SVD imputation to combined final dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/svd_imputation.py
  python scripts/svd_imputation.py --input data/final/combined_final_dataset.csv --output data/final/imputed_final_data.csv
  python scripts/svd_imputation.py --components 100 --max-iter 200
        """
    )
    
    parser.add_argument('--input', '-i', 
                       default='data/final/combined_final_dataset.csv',
                       help='Input CSV file path (default: data/final/combined_final_dataset.csv)')
    parser.add_argument('--output', '-o', 
                       default='data/final/imputed_final_data.csv',
                       help='Output CSV file path (default: data/final/imputed_final_data.csv)')
    parser.add_argument('--components', '-c', type=int, default=50,
                       help='Number of SVD components (default: 50)')
    parser.add_argument('--max-iter', '-m', type=int, default=100,
                       help='Maximum iterations for SVD (default: 100)')
    parser.add_argument('--tolerance', '-t', type=float, default=1e-4,
                       help='Convergence tolerance (default: 1e-4)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 80)
        print("SVD IMPUTATION FOR COMBINED FINAL DATASET")
        print("=" * 80)
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"SVD components: {args.components}")
        print(f"Max iterations: {args.max_iter}")
        print(f"Tolerance: {args.tolerance}")
        print()
    
    # Load data
    if verbose:
        print("Loading data...")
    
    try:
        df = pd.read_csv(input_path)
        if verbose:
            print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    
    # Remove rows with missing target values
    if 'Win_Pct' in df.columns:
        initial_rows = len(df)
        df = df[df['Win_Pct'].notna()]
        if verbose and len(df) != initial_rows:
            print(f"Removed {initial_rows - len(df)} rows with missing Win_Pct values")
    
    # Apply preprocessing
    try:
        processed_df, scaler, imputer = preprocess_features(
            df, 
            target_col='Win_Pct',
            exclude_cols=['Team', 'Year'],
            n_components=args.components,
            verbose=verbose
        )
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)
    
    # Save results
    if verbose:
        print(f"\\nSaving imputed data to {output_path}...")
    
    try:
        processed_df.to_csv(output_path, index=False)
        if verbose:
            print("Successfully saved imputed dataset!")
            
            # Create summary file
            summary_path = output_path.with_suffix('.txt')
            with open(summary_path, 'w') as f:
                f.write("SVD Imputation Summary\\n")
                f.write("=" * 40 + "\\n\\n")
                f.write(f"Input file: {input_path}\\n")
                f.write(f"Output file: {output_path}\\n")
                f.write(f"Dataset shape: {processed_df.shape[0]:,} rows × {processed_df.shape[1]:,} columns\\n")
                f.write(f"SVD components used: {args.components}\\n")
                f.write(f"Max iterations: {args.max_iter}\\n")
                f.write(f"Convergence tolerance: {args.tolerance}\\n")
                
                # Feature summary
                feature_cols = [col for col in processed_df.columns if col not in ['Team', 'Year', 'Win_Pct']]
                normalized_cols = [col for col in feature_cols if col.endswith('_Norm')]
                non_normalized_cols = [col for col in feature_cols if not col.endswith('_Norm')]
                
                f.write(f"\\nFeature Summary:\\n")
                f.write(f"  Total features processed: {len(feature_cols)}\\n")
                f.write(f"  Already normalized: {len(normalized_cols)}\\n")
                f.write(f"  Standardized during processing: {len(non_normalized_cols)}\\n")
                f.write(f"  Missing values after imputation: {processed_df.isnull().sum().sum()}\\n")
            
            print(f"Summary saved to {summary_path}")
            
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)
    
    if verbose:
        print("\\n" + "=" * 80)
        print("SVD IMPUTATION COMPLETED SUCCESSFULLY")
        print("=" * 80)


if __name__ == "__main__":
    main()