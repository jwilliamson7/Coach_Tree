#!/usr/bin/env python3
"""
Compare linear vs quadratic model fit using adjusted R²
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

# Load data
war_file = 'outputs/analysis/aggression_war_merged_data.csv'
merged_data = pd.read_csv(war_file)

# Define eras
merged_data['era'] = pd.cut(
    merged_data['year'],
    bins=[2005, 2011, 2017, 2025],
    labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
)

def adjusted_r2(r2, n, p):
    """Calculate adjusted R²

    Args:
        r2: R² value
        n: sample size
        p: number of predictors (excluding intercept)
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("="*80)
print("MODEL COMPARISON: LINEAR vs QUADRATIC")
print("="*80)

for era in ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']:
    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    if len(era_data) < 20:
        continue

    x = era_data['composite_aggression'].values
    y = era_data['annual_war'].values
    n = len(x)

    # Linear model (1 predictor)
    slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(x, y)
    y_pred_lin = slope_lin * x + intercept_lin
    r2_lin = r2_score(y, y_pred_lin)
    adj_r2_lin = adjusted_r2(r2_lin, n, p=1)

    # Quadratic model (2 predictors: x and x²)
    coeffs = np.polyfit(x, y, 2)
    y_pred_quad = np.polyval(coeffs, x)
    r2_quad = r2_score(y, y_pred_quad)
    adj_r2_quad = adjusted_r2(r2_quad, n, p=2)

    print(f"\n{era} (n={n}):")
    print("-"*80)
    print(f"LINEAR MODEL:")
    print(f"  R²:          {r2_lin:.4f}")
    print(f"  Adjusted R²: {adj_r2_lin:.4f}")
    print(f"  p-value:     {p_lin:.4f}")

    print(f"\nQUADRATIC MODEL:")
    print(f"  R²:          {r2_quad:.4f}")
    print(f"  Adjusted R²: {adj_r2_quad:.4f}")

    print(f"\nCOMPARISON:")
    improvement = adj_r2_quad - adj_r2_lin
    print(f"  Change in Adjusted R2: {improvement:+.4f}")

    if adj_r2_quad > adj_r2_lin:
        print(f"  >> Quadratic model is BETTER (even after penalty for extra parameter)")
    else:
        print(f"  >> Linear model is better (quadratic doesn't justify extra parameter)")

    # F-test for nested models
    ss_res_lin = np.sum((y - y_pred_lin)**2)
    ss_res_quad = np.sum((y - y_pred_quad)**2)

    f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / (n - 3))
    p_f = 1 - stats.f.cdf(f_stat, 1, n - 3)

    print(f"\nF-TEST (is quadratic term significant?):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value:     {p_f:.4f}")
    if p_f < 0.05:
        print(f"  >> Quadratic term is SIGNIFICANT")
    else:
        print(f"  >> Quadratic term is not significant")

print("\n" + "="*80)
