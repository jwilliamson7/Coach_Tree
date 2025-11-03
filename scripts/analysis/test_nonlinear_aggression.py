#!/usr/bin/env python3
"""
Test for non-linear relationship between aggression and WAR

Hypothesis: Extreme aggression now hurts performance, creating diminishing
or negative returns at high levels.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data
war_file = 'outputs/analysis/aggression_war_merged_data.csv'
merged_data = pd.read_csv(war_file)

# Define eras
merged_data['era'] = pd.cut(
    merged_data['year'],
    bins=[2005, 2011, 2017, 2025],
    labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
)

print("="*80)
print("TESTING FOR NON-LINEAR AGGRESSION-WAR RELATIONSHIP")
print("="*80)

# Test 1: Quadratic regression
print("\n1. QUADRATIC REGRESSION (aggression + aggression^2):")
print("-"*80)

for era in ['Early (2006-2011)', 'Late (2018-2024)']:
    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    if len(era_data) < 20:
        continue

    x = era_data['composite_aggression'].values
    y = era_data['annual_war'].values

    # Linear model
    slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(x, y)

    # Quadratic model
    coeffs = np.polyfit(x, y, 2)
    y_pred_quad = np.polyval(coeffs, x)
    r_quad = np.corrcoef(y, y_pred_quad)[0, 1]

    # Compare
    print(f"\n{era}:")
    print(f"  Linear:    r={r_lin:.3f}, p={p_lin:.4f}")
    print(f"  Quadratic: r={r_quad:.3f}")
    print(f"  Quadratic coeff (x^2): {coeffs[0]:.2f}")
    if coeffs[0] < 0:
        print(f"  -> INVERTED U-SHAPE (diminishing returns)")
    elif coeffs[0] > 0:
        print(f"  -> U-SHAPE (extreme aggression beneficial)")

# Test 2: Performance by aggression quintiles
print("\n\n2. PERFORMANCE BY AGGRESSION QUINTILES:")
print("-"*80)

for era in ['Early (2006-2011)', 'Late (2018-2024)']:
    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    if len(era_data) < 20:
        continue

    # Create quintiles
    era_data['quintile'] = pd.qcut(
        era_data['composite_aggression'],
        q=5,
        labels=['Q1 (Most Conservative)', 'Q2', 'Q3', 'Q4', 'Q5 (Most Aggressive)'],
        duplicates='drop'
    )

    print(f"\n{era}:")
    for q in era_data['quintile'].unique():
        q_data = era_data[era_data['quintile'] == q]['annual_war']
        print(f"  {q}: mean WAR = {q_data.mean():6.3f}, n={len(q_data)}")

# Test 3: Top 10% vs Bottom 10%
print("\n\n3. EXTREME AGGRESSION ANALYSIS (Top 10% vs Bottom 10%):")
print("-"*80)

for era in ['Early (2006-2011)', 'Late (2018-2024)']:
    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    if len(era_data) < 20:
        continue

    # Get top and bottom 10%
    p90 = era_data['composite_aggression'].quantile(0.90)
    p10 = era_data['composite_aggression'].quantile(0.10)

    top10 = era_data[era_data['composite_aggression'] >= p90]['annual_war']
    bottom10 = era_data[era_data['composite_aggression'] <= p10]['annual_war']
    middle = era_data[
        (era_data['composite_aggression'] > p10) &
        (era_data['composite_aggression'] < p90)
    ]['annual_war']

    print(f"\n{era}:")
    print(f"  Bottom 10% (Most Conservative): mean WAR = {bottom10.mean():6.3f}, n={len(bottom10)}")
    print(f"  Middle 80%:                     mean WAR = {middle.mean():6.3f}, n={len(middle)}")
    print(f"  Top 10% (Most Aggressive):      mean WAR = {top10.mean():6.3f}, n={len(top10)}")

    # Statistical test
    t_stat, p_val = stats.ttest_ind(top10, middle)
    print(f"  Top 10% vs Middle: t={t_stat:.2f}, p={p_val:.4f}")
    if p_val < 0.05:
        if top10.mean() < middle.mean():
            print(f"  -> SIGNIFICANT: Extreme aggression HURTS!")
        else:
            print(f"  -> SIGNIFICANT: Extreme aggression HELPS!")

print("\n" + "="*80)
