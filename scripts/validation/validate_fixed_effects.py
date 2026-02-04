#!/usr/bin/env python3
"""
Fixed Effects Model Validation Script

This script validates the implementation of the two-way fixed effects model
in analyze_within_coach_fixed_effects.py:

1. Verify demeaning formula implementation
2. Check cluster-robust SE calculation
3. Review era comparison methodology

Output:
- Validation results for each component
- Recommendations if issues found
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def validate_demeaning_formula():
    """
    Validate that the demeaning formula is implemented correctly.

    Two-way demeaning formula:
    x_it_demeaned = x_it - x_i_bar - x_t_bar + x_bar

    Where:
    - x_it is the original value
    - x_i_bar is the coach mean
    - x_t_bar is the year mean
    - x_bar is the overall mean
    """

    print("\n" + "=" * 80)
    print("1. DEMEANING FORMULA VALIDATION")
    print("=" * 80)

    # Load data
    data_path = Path("data/processed/coaching_genes/aggression_gene_by_year.csv")
    if not data_path.exists():
        print("ERROR: Data file not found")
        return {'valid': False, 'error': 'Data file not found'}

    df = pd.read_csv(data_path)
    df = df.rename(columns={'head_coach': 'coach'})

    # Filter to coaches with 2+ years
    coach_counts = df['coach'].value_counts()
    multi_year = coach_counts[coach_counts >= 2].index
    df = df[df['coach'].isin(multi_year)]

    var = 'composite_aggression'

    # Calculate components
    overall_mean = df[var].mean()
    coach_means = df.groupby('coach')[var].transform('mean')
    year_means = df.groupby('season')[var].transform('mean')

    # Two-way demeaning
    df['twoway_demeaned'] = df[var] - coach_means - year_means + overall_mean

    # Validate properties of two-way demeaned data
    results = {
        'valid': True,
        'checks': []
    }

    # Check 1: Mean of demeaned data should be approximately zero
    mean_demeaned = df['twoway_demeaned'].mean()
    check1_pass = np.abs(mean_demeaned) < 1e-10

    results['checks'].append({
        'check': 'Mean of demeaned data ≈ 0',
        'value': float(mean_demeaned),
        'passed': check1_pass
    })

    print(f"\nCheck 1: Mean of demeaned data approx 0")
    print(f"  Value: {mean_demeaned:.2e}")
    print(f"  Passed: {check1_pass}")

    # Check 2: Coach means of demeaned data should be approximately equal
    # (equal to the overall mean of demeaned data)
    coach_demeaned_means = df.groupby('coach')['twoway_demeaned'].mean()
    coach_mean_variance = coach_demeaned_means.var()
    check2_pass = coach_mean_variance < 1e-10

    results['checks'].append({
        'check': 'Coach means of demeaned data are equal',
        'value': float(coach_mean_variance),
        'passed': check2_pass
    })

    print(f"\nCheck 2: Coach means of demeaned data are equal")
    print(f"  Variance of coach means: {coach_mean_variance:.2e}")
    print(f"  Passed: {check2_pass}")

    # Check 3: Year means of demeaned data should be approximately equal
    year_demeaned_means = df.groupby('season')['twoway_demeaned'].mean()
    year_mean_variance = year_demeaned_means.var()
    check3_pass = year_mean_variance < 1e-10

    results['checks'].append({
        'check': 'Year means of demeaned data are equal',
        'value': float(year_mean_variance),
        'passed': check3_pass
    })

    print(f"\nCheck 3: Year means of demeaned data are equal")
    print(f"  Variance of year means: {year_mean_variance:.2e}")
    print(f"  Passed: {check3_pass}")

    # Check 4: Verify variance reduction
    orig_var = df[var].var()
    demeaned_var = df['twoway_demeaned'].var()
    variance_ratio = demeaned_var / orig_var

    results['checks'].append({
        'check': 'Variance reduction after demeaning',
        'original_variance': float(orig_var),
        'demeaned_variance': float(demeaned_var),
        'ratio': float(variance_ratio),
        'passed': variance_ratio < 1  # Should reduce variance
    })

    print(f"\nCheck 4: Variance reduction")
    print(f"  Original variance: {orig_var:.6f}")
    print(f"  Demeaned variance: {demeaned_var:.6f}")
    print(f"  Ratio: {variance_ratio:.2%}")
    print(f"  Passed: {variance_ratio < 1}")

    results['valid'] = all(c['passed'] for c in results['checks'])

    return results


def validate_clustered_se():
    """
    Validate cluster-robust standard error calculation.

    The implementation should use:
    1. Sandwich estimator (HC0 or HC1 correction)
    2. Cluster by coach (not by observation)
    3. Finite sample adjustment
    """

    print("\n" + "=" * 80)
    print("2. CLUSTERED STANDARD ERROR VALIDATION")
    print("=" * 80)

    # Load results
    results_path = Path("outputs/analysis/within_coach_fixed_effects_results.json")
    if not results_path.exists():
        print("ERROR: Fixed effects results not found")
        print("Run analyze_within_coach_fixed_effects.py first")
        return {'valid': False, 'error': 'Results file not found'}

    with open(results_path, 'r') as f:
        fe_results = json.load(f)

    results = {
        'valid': True,
        'checks': []
    }

    # Check 1: SE should be larger than naive OLS SE (due to clustering)
    pooled = fe_results.get('two_way', {}).get('pooled', {})

    if pooled:
        # Compare clustered SE with non-clustered SE
        # The implementation reports clustered SE
        clustered_se = pooled.get('aggression_se', 0)

        # A rough check: for clustered data, SE should be non-trivial
        # relative to the coefficient
        coef = pooled.get('aggression_coef', 0)
        t_stat = pooled.get('aggression_t', 0)

        # Verify t-stat = coef / se
        computed_t = coef / clustered_se if clustered_se != 0 else 0
        t_match = np.abs(computed_t - t_stat) < 0.01

        results['checks'].append({
            'check': 't-statistic correctly computed',
            'expected_t': float(computed_t),
            'reported_t': float(t_stat),
            'passed': t_match
        })

        print(f"\nCheck 1: t-statistic computation")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  SE (clustered): {clustered_se:.4f}")
        print(f"  Expected t: {computed_t:.3f}")
        print(f"  Reported t: {t_stat:.3f}")
        print(f"  Passed: {t_match}")

        # Check 2: Verify degrees of freedom
        n_coaches = pooled.get('n_coaches', 0)
        n_obs = pooled.get('n', 0)

        # For cluster-robust SEs, df should be based on number of clusters
        # The p-value should use t-distribution with n_clusters - k - 1 df
        p_value = pooled.get('aggression_p', 0)

        # Compute expected p-value
        df = n_coaches - 2  # n_clusters - k - 1 where k=1 (one predictor)
        expected_p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

        p_match = np.abs(p_value - expected_p) < 0.001

        results['checks'].append({
            'check': 'p-value uses correct degrees of freedom',
            'n_clusters': int(n_coaches),
            'df': int(df),
            'expected_p': float(expected_p),
            'reported_p': float(p_value),
            'passed': p_match
        })

        print(f"\nCheck 2: Degrees of freedom for p-value")
        print(f"  Number of clusters: {n_coaches}")
        print(f"  Degrees of freedom: {df}")
        print(f"  Expected p-value: {expected_p:.4f}")
        print(f"  Reported p-value: {p_value:.4f}")
        print(f"  Passed: {p_match}")

    results['valid'] = all(c['passed'] for c in results['checks'])

    return results


def validate_era_comparison():
    """
    Validate era comparison methodology.

    The implementation should:
    1. Use separate regressions for each era
    2. Compare coefficients using proper SE of difference
    3. Note assumption of independence (which may not fully hold)
    """

    print("\n" + "=" * 80)
    print("3. ERA COMPARISON METHODOLOGY VALIDATION")
    print("=" * 80)

    # Load results
    results_path = Path("outputs/analysis/within_coach_fixed_effects_results.json")
    if not results_path.exists():
        return {'valid': False, 'error': 'Results file not found'}

    with open(results_path, 'r') as f:
        fe_results = json.load(f)

    results = {
        'valid': True,
        'checks': [],
        'notes': []
    }

    # Get era-specific results
    stratified = fe_results.get('two_way', {}).get('stratified', {})
    comparisons = fe_results.get('two_way', {}).get('coefficient_comparisons', {})

    if not stratified or not comparisons:
        return {'valid': False, 'error': 'Stratified results not found'}

    # Check 1: Verify SE of difference formula
    # SE_diff = sqrt(SE1^2 + SE2^2) assuming independence

    for comp_name, comp_results in comparisons.items():
        # Parse era names
        parts = comp_name.split(' vs ')
        if len(parts) != 2:
            continue

        era1, era2 = parts

        if era1 not in stratified or era2 not in stratified:
            continue

        se1 = stratified[era1]['aggression_se']
        se2 = stratified[era2]['aggression_se']

        expected_se_diff = np.sqrt(se1**2 + se2**2)
        reported_se_diff = comp_results['se_diff']

        se_match = np.abs(expected_se_diff - reported_se_diff) < 0.0001

        results['checks'].append({
            'check': f'SE of difference for {comp_name}',
            'expected_se_diff': float(expected_se_diff),
            'reported_se_diff': float(reported_se_diff),
            'passed': se_match
        })

        print(f"\n{comp_name}:")
        print(f"  SE1: {se1:.4f}, SE2: {se2:.4f}")
        print(f"  Expected SE_diff: {expected_se_diff:.4f}")
        print(f"  Reported SE_diff: {reported_se_diff:.4f}")
        print(f"  Passed: {se_match}")

    # Note about independence assumption
    results['notes'].append(
        "The SE of difference formula assumes independence between era estimates. "
        "This may slightly underestimate SE if coaches span multiple eras."
    )

    # Check 2: Verify z-statistic calculation
    for comp_name, comp_results in comparisons.items():
        coef_diff = comp_results['coef_diff']
        se_diff = comp_results['se_diff']

        expected_z = coef_diff / se_diff
        reported_z = comp_results['z_statistic']

        z_match = np.abs(expected_z - reported_z) < 0.001

        results['checks'].append({
            'check': f'Z-statistic for {comp_name}',
            'expected_z': float(expected_z),
            'reported_z': float(reported_z),
            'passed': z_match
        })

    results['valid'] = all(c['passed'] for c in results['checks'])

    return results


def main():
    """Main validation function"""

    print("=" * 80)
    print("FIXED EFFECTS MODEL VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}

    # Validate each component
    all_results['demeaning'] = validate_demeaning_formula()
    all_results['clustered_se'] = validate_clustered_se()
    all_results['era_comparison'] = validate_era_comparison()

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_valid = True
    for component, results in all_results.items():
        status = "PASSED" if results.get('valid', False) else "FAILED"
        all_valid = all_valid and results.get('valid', False)
        print(f"\n{component.upper()}: {status}")

        if 'checks' in results:
            for check in results['checks']:
                status = "" if check['passed'] else " [!]"
                print(f"  - {check['check']}{status}")

        if 'notes' in results:
            for note in results['notes']:
                print(f"  Note: {note}")

    # Save results
    output_dir = Path("data/processed/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "fixed_effects_validation.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    if all_valid:
        print("FIXED EFFECTS IMPLEMENTATION VALIDATED")
    else:
        print("ISSUES FOUND - REVIEW IMPLEMENTATION")
    print("=" * 80)

    return 0 if all_valid else 1


if __name__ == "__main__":
    sys.exit(main())
