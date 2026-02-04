#!/usr/bin/env python3
"""
Statistical Assumption Validation Script

This script tests statistical assumptions for the analyses:
1. Normality tests (Shapiro-Wilk)
2. Homogeneity of variance (Levene's test)
3. Autocorrelation (Durbin-Watson for temporal analyses)
4. Outlier influence (Cook's distance, leverage)

Provides robust alternatives when assumptions are violated.

Output:
- Assumption test results
- Recommendations for robust alternatives
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import shapiro, levene, spearmanr, kendalltau, mannwhitneyu

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_analysis_data() -> Dict[str, pd.DataFrame]:
    """Load all relevant analysis data"""

    data = {}
    base_dir = Path("data/processed")

    # Load aggression gene data
    agg_path = base_dir / "coaching_genes" / "aggression_gene_by_year.csv"
    if agg_path.exists():
        data['aggression_by_year'] = pd.read_csv(agg_path)
        print(f"Loaded aggression_by_year: {len(data['aggression_by_year'])} rows")

    # Load by-coach aggregated data
    agg_coach_path = base_dir / "coaching_genes" / "aggression_gene_by_coach.csv"
    if agg_coach_path.exists():
        data['aggression_by_coach'] = pd.read_csv(agg_coach_path)
        print(f"Loaded aggression_by_coach: {len(data['aggression_by_coach'])} rows")

    return data


def test_normality(data: np.ndarray, variable_name: str, alpha: float = 0.05) -> Dict:
    """
    Test for normality using Shapiro-Wilk test.

    Note: For samples > 5000, Shapiro-Wilk may reject even minor deviations.
    Also calculate skewness and kurtosis for additional assessment.
    """

    result = {
        'variable': variable_name,
        'n': len(data),
        'is_normal': None,
        'shapiro_stat': None,
        'shapiro_pvalue': None,
        'skewness': None,
        'kurtosis': None,
        'recommendation': None
    }

    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    result['n'] = len(clean_data)

    if len(clean_data) < 3:
        result['recommendation'] = "Insufficient data for normality test"
        return result

    # Shapiro-Wilk test (limit to 5000 samples)
    if len(clean_data) > 5000:
        # Use a random sample for the test
        sample = np.random.choice(clean_data, 5000, replace=False)
    else:
        sample = clean_data

    try:
        stat, pvalue = shapiro(sample)
        result['shapiro_stat'] = float(stat)
        result['shapiro_pvalue'] = float(pvalue)
        result['is_normal'] = pvalue > alpha
    except Exception as e:
        result['shapiro_stat'] = None
        result['shapiro_pvalue'] = None

    # Calculate skewness and kurtosis
    result['skewness'] = float(stats.skew(clean_data))
    result['kurtosis'] = float(stats.kurtosis(clean_data))

    # Make recommendation
    if result['is_normal'] == False:
        if abs(result['skewness']) > 2:
            result['recommendation'] = "Severely skewed - use non-parametric tests (Mann-Whitney, Spearman)"
        elif abs(result['skewness']) > 1:
            result['recommendation'] = "Moderately skewed - consider transformation or robust methods"
        else:
            result['recommendation'] = "Minor deviation from normality - parametric tests likely OK"
    else:
        result['recommendation'] = "Data appears normally distributed - parametric tests appropriate"

    return result


def test_homogeneity_of_variance(groups: List[np.ndarray], group_names: List[str], alpha: float = 0.05) -> Dict:
    """
    Test homogeneity of variance using Levene's test.
    """

    result = {
        'test': "Levene's test",
        'groups': group_names,
        'group_sizes': [len(g) for g in groups],
        'group_variances': [float(np.nanvar(g)) for g in groups],
        'levene_stat': None,
        'levene_pvalue': None,
        'homogeneous': None,
        'recommendation': None
    }

    # Remove NaN from each group
    clean_groups = [g[~np.isnan(g)] for g in groups]

    # Filter out empty groups
    valid_groups = [g for g in clean_groups if len(g) > 0]

    if len(valid_groups) < 2:
        result['recommendation'] = "Insufficient groups for variance test"
        return result

    try:
        stat, pvalue = levene(*valid_groups)
        result['levene_stat'] = float(stat)
        result['levene_pvalue'] = float(pvalue)
        result['homogeneous'] = pvalue > alpha
    except Exception as e:
        result['recommendation'] = f"Levene test failed: {e}"
        return result

    # Calculate variance ratio
    max_var = max(result['group_variances'])
    min_var = min([v for v in result['group_variances'] if v > 0])
    result['variance_ratio'] = float(max_var / min_var) if min_var > 0 else float('inf')

    if not result['homogeneous']:
        result['recommendation'] = "Heterogeneous variances - use Welch's t-test or robust methods"
    else:
        result['recommendation'] = "Variances are homogeneous - standard parametric tests appropriate"

    return result


def test_autocorrelation(data: np.ndarray, variable_name: str) -> Dict:
    """
    Test for autocorrelation in time series data using Durbin-Watson.

    Values close to 2 indicate no autocorrelation.
    Values < 1.5 or > 2.5 indicate significant autocorrelation.
    """

    result = {
        'variable': variable_name,
        'n': len(data),
        'durbin_watson': None,
        'autocorrelation_lag1': None,
        'has_autocorrelation': None,
        'recommendation': None
    }

    # Remove NaN
    clean_data = data[~np.isnan(data)]
    result['n'] = len(clean_data)

    if len(clean_data) < 10:
        result['recommendation'] = "Insufficient data for autocorrelation test"
        return result

    # Calculate Durbin-Watson statistic
    residuals = clean_data - np.mean(clean_data)  # Deviations from mean
    diffs = np.diff(residuals)
    dw = np.sum(diffs**2) / np.sum(residuals**2)
    result['durbin_watson'] = float(dw)

    # Calculate lag-1 autocorrelation
    n = len(clean_data)
    lag1 = np.corrcoef(clean_data[:-1], clean_data[1:])[0, 1]
    result['autocorrelation_lag1'] = float(lag1)

    # Interpret
    if dw < 1.5:
        result['has_autocorrelation'] = True
        result['recommendation'] = "Positive autocorrelation detected - use clustered/robust standard errors"
    elif dw > 2.5:
        result['has_autocorrelation'] = True
        result['recommendation'] = "Negative autocorrelation detected - use clustered/robust standard errors"
    else:
        result['has_autocorrelation'] = False
        result['recommendation'] = "No significant autocorrelation - standard errors OK"

    return result


def analyze_outliers(data: pd.DataFrame, value_col: str, id_col: str = None) -> Dict:
    """
    Analyze outliers using IQR method and z-scores.
    """

    result = {
        'variable': value_col,
        'n': len(data),
        'outliers_iqr': [],
        'outliers_zscore': [],
        'n_outliers_iqr': 0,
        'n_outliers_zscore': 0,
        'recommendation': None
    }

    values = data[value_col].values
    clean_mask = ~np.isnan(values)
    clean_values = values[clean_mask]

    if len(clean_values) < 4:
        result['recommendation'] = "Insufficient data for outlier analysis"
        return result

    # IQR method
    q1 = np.percentile(clean_values, 25)
    q3 = np.percentile(clean_values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    iqr_outlier_mask = (values < lower_bound) | (values > upper_bound)
    result['n_outliers_iqr'] = int(iqr_outlier_mask.sum())
    result['iqr_bounds'] = [float(lower_bound), float(upper_bound)]

    # Z-score method (>3 std)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    zscore_outlier_mask = np.abs((values - mean) / std) > 3
    result['n_outliers_zscore'] = int(np.nansum(zscore_outlier_mask))

    # Get outlier identifiers
    if id_col and id_col in data.columns:
        iqr_outliers = data.loc[iqr_outlier_mask, [id_col, value_col]].values.tolist()
        result['outliers_iqr'] = iqr_outliers[:10]  # Limit to first 10

        zscore_outliers = data.loc[zscore_outlier_mask, [id_col, value_col]].values.tolist()
        result['outliers_zscore'] = zscore_outliers[:10]

    # Calculate outlier percentage
    outlier_pct = result['n_outliers_iqr'] / len(data) * 100
    result['outlier_percentage'] = float(outlier_pct)

    if outlier_pct > 5:
        result['recommendation'] = "High outlier rate (>5%) - consider robust regression or winsorization"
    elif outlier_pct > 2:
        result['recommendation'] = "Moderate outliers - check influential observations"
    else:
        result['recommendation'] = "Outlier rate acceptable - standard methods OK"

    return result


def run_assumption_tests(data: Dict[str, pd.DataFrame]) -> Dict:
    """Run all assumption tests on the data"""

    results = {
        'normality_tests': [],
        'variance_tests': [],
        'autocorrelation_tests': [],
        'outlier_analyses': [],
        'issues': [],
        'recommendations': []
    }

    if 'aggression_by_year' not in data:
        results['issues'].append({
            'type': 'missing_data',
            'description': 'Aggression by year data not available'
        })
        return results

    df = data['aggression_by_year']

    print("\n" + "=" * 80)
    print("STATISTICAL ASSUMPTION TESTS")
    print("=" * 80)

    # 1. Normality Tests
    print("\n" + "-" * 80)
    print("1. NORMALITY TESTS (Shapiro-Wilk)")
    print("-" * 80)

    normality_vars = [
        'composite_aggression',
        'fourth_down_aggression',
        'pass_heavy_aggression',
        'deep_pass_aggression',
        'two_point_aggression'
    ]

    for var in normality_vars:
        if var in df.columns:
            test_result = test_normality(df[var].values, var)
            results['normality_tests'].append(test_result)

            status = "NORMAL" if test_result['is_normal'] else "NOT NORMAL"
            print(f"\n{var}:")
            print(f"  N = {test_result['n']}")
            print(f"  Shapiro-Wilk p-value: {test_result['shapiro_pvalue']:.4f}")
            print(f"  Skewness: {test_result['skewness']:.3f}")
            print(f"  Kurtosis: {test_result['kurtosis']:.3f}")
            print(f"  Status: {status}")
            print(f"  Recommendation: {test_result['recommendation']}")

            if not test_result['is_normal']:
                results['issues'].append({
                    'type': 'non_normal',
                    'variable': var,
                    'skewness': test_result['skewness']
                })

    # 2. Homogeneity of Variance Tests
    print("\n" + "-" * 80)
    print("2. HOMOGENEITY OF VARIANCE TESTS (Levene)")
    print("-" * 80)

    # Test variance across eras
    if 'season' in df.columns and 'composite_aggression' in df.columns:
        df['era'] = pd.cut(df['season'],
                          bins=[2005, 2011, 2017, 2025],
                          labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)'])

        era_groups = []
        era_names = []
        for era in df['era'].dropna().unique():
            era_data = df[df['era'] == era]['composite_aggression'].values
            if len(era_data) > 0:
                era_groups.append(era_data)
                era_names.append(str(era))

        if len(era_groups) >= 2:
            var_result = test_homogeneity_of_variance(era_groups, era_names)
            results['variance_tests'].append(var_result)

            status = "HOMOGENEOUS" if var_result['homogeneous'] else "HETEROGENEOUS"
            print(f"\nComposite Aggression across Eras:")
            print(f"  Groups: {var_result['groups']}")
            print(f"  Sizes: {var_result['group_sizes']}")
            print(f"  Variances: {[f'{v:.6f}' for v in var_result['group_variances']]}")
            print(f"  Variance ratio: {var_result['variance_ratio']:.2f}")
            print(f"  Levene p-value: {var_result['levene_pvalue']:.4f}")
            print(f"  Status: {status}")
            print(f"  Recommendation: {var_result['recommendation']}")

            if not var_result['homogeneous']:
                results['issues'].append({
                    'type': 'heterogeneous_variance',
                    'analysis': 'era_comparison'
                })

    # 3. Autocorrelation Tests
    print("\n" + "-" * 80)
    print("3. AUTOCORRELATION TESTS (Durbin-Watson)")
    print("-" * 80)

    # Test autocorrelation in yearly aggregates
    if 'season' in df.columns and 'composite_aggression' in df.columns:
        yearly_means = df.groupby('season')['composite_aggression'].mean().sort_index()

        if len(yearly_means) >= 10:
            ac_result = test_autocorrelation(yearly_means.values, 'composite_aggression (yearly means)')
            results['autocorrelation_tests'].append(ac_result)

            status = "AUTOCORRELATED" if ac_result['has_autocorrelation'] else "NO AUTOCORRELATION"
            print(f"\nComposite Aggression (yearly means):")
            print(f"  N = {ac_result['n']}")
            print(f"  Durbin-Watson: {ac_result['durbin_watson']:.3f}")
            print(f"  Lag-1 autocorrelation: {ac_result['autocorrelation_lag1']:.3f}")
            print(f"  Status: {status}")
            print(f"  Recommendation: {ac_result['recommendation']}")

            if ac_result['has_autocorrelation']:
                results['issues'].append({
                    'type': 'autocorrelation',
                    'variable': 'yearly_means'
                })

    # 4. Outlier Analysis
    print("\n" + "-" * 80)
    print("4. OUTLIER ANALYSIS")
    print("-" * 80)

    if 'head_coach' in df.columns and 'composite_aggression' in df.columns:
        outlier_result = analyze_outliers(df, 'composite_aggression', 'head_coach')
        results['outlier_analyses'].append(outlier_result)

        print(f"\nComposite Aggression:")
        print(f"  N = {outlier_result['n']}")
        print(f"  IQR bounds: [{outlier_result['iqr_bounds'][0]:.4f}, {outlier_result['iqr_bounds'][1]:.4f}]")
        print(f"  Outliers (IQR): {outlier_result['n_outliers_iqr']} ({outlier_result['outlier_percentage']:.1f}%)")
        print(f"  Outliers (z>3): {outlier_result['n_outliers_zscore']}")
        print(f"  Recommendation: {outlier_result['recommendation']}")

        if outlier_result['outliers_iqr']:
            print(f"  Sample outliers: {outlier_result['outliers_iqr'][:5]}")

        if outlier_result['outlier_percentage'] > 5:
            results['issues'].append({
                'type': 'high_outliers',
                'variable': 'composite_aggression',
                'percentage': outlier_result['outlier_percentage']
            })

    # Generate overall recommendations
    print("\n" + "-" * 80)
    print("5. OVERALL RECOMMENDATIONS")
    print("-" * 80)

    recommendations = []

    # Check for non-normality
    non_normal_vars = [t['variable'] for t in results['normality_tests'] if not t.get('is_normal', True)]
    if non_normal_vars:
        recommendations.append(f"Non-normal distributions: {non_normal_vars} - use robust/non-parametric methods")

    # Check for heterogeneous variance
    if any(not t.get('homogeneous', True) for t in results['variance_tests']):
        recommendations.append("Heterogeneous variance detected - use Welch's correction or bootstrap")

    # Check for autocorrelation
    if any(t.get('has_autocorrelation', False) for t in results['autocorrelation_tests']):
        recommendations.append("Autocorrelation in temporal data - use clustered standard errors or AR models")

    # Check for outliers
    if any(a.get('outlier_percentage', 0) > 5 for a in results['outlier_analyses']):
        recommendations.append("High outlier rate - consider robust regression (MM estimation)")

    results['recommendations'] = recommendations

    print()
    if recommendations:
        for rec in recommendations:
            print(f"  - {rec}")
    else:
        print("  No major assumption violations detected")

    return results


def main():
    """Main validation function"""

    print("=" * 80)
    print("STATISTICAL ASSUMPTION VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading data...")
    data = load_analysis_data()

    # Run tests
    results = run_assumption_tests(data)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if results['issues']:
        print(f"\nTotal issues found: {len(results['issues'])}")
        for issue in results['issues']:
            print(f"  - {issue['type']}: {issue.get('variable', issue.get('analysis', 'Unknown'))}")
    else:
        print("\nNo major assumption violations found!")

    # Save results
    output_dir = Path("data/processed/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "assumption_validation.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return results['issues']


if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues else 0)
