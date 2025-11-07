#!/usr/bin/env python3
"""
Apply Benjamini-Hochberg correction to all p-values from the aggression analyses.

This script compiles all hypothesis tests conducted in the paper and applies
the Benjamini-Hochberg False Discovery Rate (FDR) correction to control for
multiple comparisons.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def benjamini_hochberg(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction.

    Parameters:
    -----------
    p_values : array-like
        List of p-values
    alpha : float
        Desired FDR level (default 0.05)

    Returns:
    --------
    pd.DataFrame with original p-values, adjusted significance, and ranks
    """
    n = len(p_values)

    # Create dataframe and sort by p-value
    df = pd.DataFrame({
        'original_p': p_values,
        'test_id': range(len(p_values))
    })
    df = df.sort_values('original_p').reset_index(drop=True)

    # Calculate BH critical values
    df['rank'] = df.index + 1
    df['bh_threshold'] = (df['rank'] / n) * alpha
    df['significant_raw'] = df['original_p'] < 0.05
    df['significant_bh'] = df['original_p'] <= df['bh_threshold']

    return df

# Compile all p-values from analyses
tests = []

# ============================================================================
# 1. OVERALL WAR RELATIONSHIPS (5 tests)
# ============================================================================
tests.extend([
    {'category': 'Overall WAR', 'test': 'Composite Aggression', 'p': 0.0315},
    {'category': 'Overall WAR', 'test': '4th Down Aggression', 'p': 0.1709},
    {'category': 'Overall WAR', 'test': 'Pass-Heavy Aggression', 'p': 0.0047},
    {'category': 'Overall WAR', 'test': 'Deep Pass Aggression', 'p': 0.7537},
    {'category': 'Overall WAR', 'test': '2-Point Aggression', 'p': 0.8867},
])

# ============================================================================
# 2. TEMPORAL TREND ANALYSIS (6 tests)
# ============================================================================
tests.extend([
    {'category': 'Temporal Trend', 'test': 'Linear regression', 'p': 6.31e-08},
    {'category': 'Temporal Trend', 'test': 'ANOVA across eras', 'p': 3.26e-07},
    {'category': 'Temporal Trend', 'test': 'Early vs Middle (t-test)', 'p': 0.0653},
    {'category': 'Temporal Trend', 'test': 'Early vs Late (t-test)', 'p': 2.71e-06},
    {'category': 'Temporal Trend', 'test': 'Middle vs Late (t-test)', 'p': 5.28e-05},
    {'category': 'Temporal Trend', 'test': 'Early vs Late (Mann-Whitney)', 'p': 0.0012},
])

# ============================================================================
# 3. ERA-SPECIFIC WAR RELATIONSHIPS (6 tests)
# ============================================================================
tests.extend([
    {'category': 'Era Analysis', 'test': 'Early: Composite', 'p': 0.0056},
    {'category': 'Era Analysis', 'test': 'Early: Pass-Heavy', 'p': 0.1488},
    {'category': 'Era Analysis', 'test': 'Middle: Composite', 'p': 0.1450},
    {'category': 'Era Analysis', 'test': 'Middle: Pass-Heavy', 'p': 0.0976},
    {'category': 'Era Analysis', 'test': 'Late: Composite', 'p': 0.7203},
    {'category': 'Era Analysis', 'test': 'Late: Pass-Heavy', 'p': 0.1518},
])

# ============================================================================
# 4. COACH TYPE OVERALL (4 tests)
# ============================================================================
tests.extend([
    {'category': 'Coach Type (Overall)', 'test': 'Offensive: Composite', 'p': 0.0922},
    {'category': 'Coach Type (Overall)', 'test': 'Offensive: Pass-Heavy', 'p': 0.0114},
    {'category': 'Coach Type (Overall)', 'test': 'Defensive: Composite', 'p': 0.0892},
    {'category': 'Coach Type (Overall)', 'test': 'Defensive: Pass-Heavy', 'p': 0.0433},
])

# ============================================================================
# 5. COACH TYPE BY ERA (6 tests)
# ============================================================================
tests.extend([
    {'category': 'Coach Type by Era', 'test': 'Offensive Early', 'p': 0.0310},
    {'category': 'Coach Type by Era', 'test': 'Offensive Middle', 'p': 0.7112},
    {'category': 'Coach Type by Era', 'test': 'Offensive Late', 'p': 0.8226},
    {'category': 'Coach Type by Era', 'test': 'Defensive Early', 'p': 0.0422},
    {'category': 'Coach Type by Era', 'test': 'Defensive Middle', 'p': 0.0068},
    {'category': 'Coach Type by Era', 'test': 'Defensive Late', 'p': 0.2925},
])

# ============================================================================
# 6. PERSISTENCE OVERALL (15 tests: 5 measures × 3 lags)
# ============================================================================
# All have p < 0.0001, represented as 0.0000 in logs
for lag in [1, 2, 3]:
    for measure in ['Composite', '4th Down', 'Pass-Heavy', 'Deep Pass', '2-Point']:
        tests.append({
            'category': f'Persistence Lag {lag}',
            'test': measure,
            'p': 0.0001  # Conservative since logs show 0.0000
        })

# ============================================================================
# 7. PERSISTENCE BY COACH TYPE (30 tests: 2 types × 5 measures × 3 lags)
# ============================================================================
# Offensive coaches - all p < 0.0001 except Deep Pass Lag 3
tests.extend([
    {'category': 'Persistence Offensive', 'test': '4th Down Lag 1', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': '4th Down Lag 2', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': '4th Down Lag 3', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Pass-Heavy Lag 1', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Pass-Heavy Lag 2', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Pass-Heavy Lag 3', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Deep Pass Lag 1', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Deep Pass Lag 2', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Deep Pass Lag 3', 'p': 0.0007},
    {'category': 'Persistence Offensive', 'test': '2-Point Lag 1', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': '2-Point Lag 2', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': '2-Point Lag 3', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Composite Lag 1', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Composite Lag 2', 'p': 0.0001},
    {'category': 'Persistence Offensive', 'test': 'Composite Lag 3', 'p': 0.0001},
])

# Defensive coaches
tests.extend([
    {'category': 'Persistence Defensive', 'test': '4th Down Lag 1', 'p': 0.0001},
    {'category': 'Persistence Defensive', 'test': '4th Down Lag 2', 'p': 0.0037},
    {'category': 'Persistence Defensive', 'test': '4th Down Lag 3', 'p': 0.0039},
    {'category': 'Persistence Defensive', 'test': 'Pass-Heavy Lag 1', 'p': 0.0001},
    {'category': 'Persistence Defensive', 'test': 'Pass-Heavy Lag 2', 'p': 0.0017},
    {'category': 'Persistence Defensive', 'test': 'Pass-Heavy Lag 3', 'p': 0.0125},
    {'category': 'Persistence Defensive', 'test': 'Deep Pass Lag 1', 'p': 0.0001},
    {'category': 'Persistence Defensive', 'test': 'Deep Pass Lag 2', 'p': 0.0022},
    {'category': 'Persistence Defensive', 'test': 'Deep Pass Lag 3', 'p': 0.0008},
    {'category': 'Persistence Defensive', 'test': '2-Point Lag 1', 'p': 0.0001},
    {'category': 'Persistence Defensive', 'test': '2-Point Lag 2', 'p': 0.0206},
    {'category': 'Persistence Defensive', 'test': '2-Point Lag 3', 'p': 0.0524},
    {'category': 'Persistence Defensive', 'test': 'Composite Lag 1', 'p': 0.0001},
    {'category': 'Persistence Defensive', 'test': 'Composite Lag 2', 'p': 0.0001},
    {'category': 'Persistence Defensive', 'test': 'Composite Lag 3', 'p': 0.0001},
])

# ============================================================================
# 8. INHERITANCE BY MENTOR BACKGROUND (10 tests: 2 types × 5 measures)
# ============================================================================
tests.extend([
    {'category': 'Inheritance: Offensive Mentors', 'test': '4th Down', 'p': 0.0001},
    {'category': 'Inheritance: Offensive Mentors', 'test': 'Pass-Heavy', 'p': 0.4077},
    {'category': 'Inheritance: Offensive Mentors', 'test': 'Deep Pass', 'p': 0.8003},
    {'category': 'Inheritance: Offensive Mentors', 'test': '2-Point', 'p': 0.1387},
    {'category': 'Inheritance: Offensive Mentors', 'test': 'Composite', 'p': 0.2512},

    {'category': 'Inheritance: Defensive Mentors', 'test': '4th Down', 'p': 0.7343},
    {'category': 'Inheritance: Defensive Mentors', 'test': 'Pass-Heavy', 'p': 0.7161},
    {'category': 'Inheritance: Defensive Mentors', 'test': 'Deep Pass', 'p': 0.1718},
    {'category': 'Inheritance: Defensive Mentors', 'test': '2-Point', 'p': 0.1589},
    {'category': 'Inheritance: Defensive Mentors', 'test': 'Composite', 'p': 0.4698},
])

# ============================================================================
# 9. INHERITANCE BY COORDINATOR TYPE (10 tests: 2 types × 5 measures)
# ============================================================================
tests.extend([
    {'category': 'Inheritance: OC->HC', 'test': '4th Down', 'p': 0.0040},
    {'category': 'Inheritance: OC->HC', 'test': 'Pass-Heavy', 'p': 0.0006},
    {'category': 'Inheritance: OC->HC', 'test': 'Deep Pass', 'p': 0.9936},
    {'category': 'Inheritance: OC->HC', 'test': '2-Point', 'p': 0.0914},
    {'category': 'Inheritance: OC->HC', 'test': 'Composite', 'p': 0.1416},

    {'category': 'Inheritance: DC->HC', 'test': '4th Down', 'p': 0.7184},
    {'category': 'Inheritance: DC->HC', 'test': 'Pass-Heavy', 'p': 0.7535},
    {'category': 'Inheritance: DC->HC', 'test': 'Deep Pass', 'p': 0.2188},
    {'category': 'Inheritance: DC->HC', 'test': '2-Point', 'p': 0.1683},
    {'category': 'Inheritance: DC->HC', 'test': 'Composite', 'p': 0.4536},
])

# ============================================================================
# 10. WITHIN-COACH FIXED EFFECTS (4 tests: pooled + 3 eras)
# ============================================================================
tests.extend([
    {'category': 'Two-Way Fixed Effects', 'test': 'Pooled (2006-2024)', 'p': 0.0130},
    {'category': 'Two-Way Fixed Effects', 'test': 'Early (2006-2011)', 'p': 0.0640},
    {'category': 'Two-Way Fixed Effects', 'test': 'Middle (2012-2017)', 'p': 0.3030},
    {'category': 'Two-Way Fixed Effects', 'test': 'Late (2018-2024)', 'p': 0.4240},
])

# ============================================================================
# 11. TEMPORAL ROBUSTNESS (2 tests: continuous year + structural break)
# ============================================================================
tests.extend([
    {'category': 'Temporal Robustness', 'test': 'Aggression x Year interaction', 'p': 0.0195},
    {'category': 'Temporal Robustness', 'test': 'Chow test (2011 breakpoint)', 'p': 0.0258},
])

# ============================================================================
# CREATE DATAFRAME AND APPLY CORRECTION
# ============================================================================
df = pd.DataFrame(tests)
print(f"\nTotal number of hypothesis tests: {len(df)}")
print(f"\nBy category:")
print(df['category'].value_counts().sort_index())

# Apply Benjamini-Hochberg correction
p_values = df['p'].values
n = len(p_values)

# Sort by p-value
df_sorted = df.sort_values('p').reset_index(drop=True)
df_sorted['rank'] = df_sorted.index + 1
df_sorted['bh_threshold'] = (df_sorted['rank'] / n) * 0.05
df_sorted['sig_raw'] = df_sorted['p'] < 0.05
df_sorted['sig_bh'] = df_sorted['p'] <= df_sorted['bh_threshold']

# Find largest rank where p <= threshold (this is the cutoff)
sig_tests = df_sorted[df_sorted['sig_bh']]
if len(sig_tests) > 0:
    cutoff_rank = sig_tests['rank'].max()
    cutoff_p = sig_tests['p'].max()
else:
    cutoff_rank = 0
    cutoff_p = 0

print(f"\n{'='*80}")
print("BENJAMINI-HOCHBERG CORRECTION RESULTS")
print(f"{'='*80}")
print(f"\nFDR level (alpha): 0.05")
print(f"Total tests: {n}")
print(f"Tests significant at raw p < 0.05: {df_sorted['sig_raw'].sum()}")
print(f"Tests significant after BH correction: {df_sorted['sig_bh'].sum()}")
print(f"\nBH cutoff: p <= {cutoff_p:.6f} (rank {cutoff_rank}/{n})")

# Show tests that changed significance
print(f"\n{'='*80}")
print("TESTS THAT LOST SIGNIFICANCE AFTER CORRECTION:")
print(f"{'='*80}")
lost_sig = df_sorted[(df_sorted['sig_raw']) & (~df_sorted['sig_bh'])]
if len(lost_sig) > 0:
    for idx, row in lost_sig.iterrows():
        print(f"\n{row['category']}: {row['test']}")
        print(f"  p = {row['p']:.4f}, threshold = {row['bh_threshold']:.6f}")
else:
    print("\nNone - all tests that were significant remain significant!")

# Show strongest results
print(f"\n{'='*80}")
print("TOP 20 MOST SIGNIFICANT RESULTS (after BH correction):")
print(f"{'='*80}")
top20 = df_sorted[df_sorted['sig_bh']].head(20)
for idx, row in top20.iterrows():
    sig_marker = 'Y' if row['sig_bh'] else ' '
    print(f"{sig_marker} {row['category']:30s} | {row['test']:25s} | p={row['p']:.6f}")

# Save results
output_dir = Path('outputs/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

df_sorted.to_csv(output_dir / 'benjamini_hochberg_results.csv', index=False)
print(f"\n\nFull results saved to: {output_dir / 'benjamini_hochberg_results.csv'}")

# Summary statistics by category
print(f"\n{'='*80}")
print("SIGNIFICANCE BY CATEGORY:")
print(f"{'='*80}")
summary = df_sorted.groupby('category').agg({
    'sig_raw': 'sum',
    'sig_bh': 'sum',
    'p': 'count'
}).rename(columns={'p': 'total'})
summary['pct_sig_raw'] = (summary['sig_raw'] / summary['total'] * 100).round(1)
summary['pct_sig_bh'] = (summary['sig_bh'] / summary['total'] * 100).round(1)
print(summary.to_string())
