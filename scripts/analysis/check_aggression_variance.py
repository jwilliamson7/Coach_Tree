#!/usr/bin/env python3
"""
Check variance in aggression over time

Tests whether correlation erosion is due to:
1. Saturation (everyone became aggressive, variance decreased)
2. Equalization (diverse behavior persists, but returns equalized)
"""

import pandas as pd
import numpy as np

# Load aggression data
agg_data = pd.read_csv('data/processed/coaching_genes/aggression_gene_by_year.csv')

# Calculate variance by era
agg_data['era'] = pd.cut(
    agg_data['season'],
    bins=[2005, 2011, 2017, 2025],
    labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
)

print("VARIANCE IN AGGRESSION OVER TIME")
print("="*60)

for component in ['fourth_down_aggression', 'pass_heavy_aggression',
                  'two_point_aggression', 'composite_aggression']:
    print(f"\n{component}:")
    print("-"*60)

    for era in ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']:
        era_data = agg_data[agg_data['era'] == era][component].dropna()

        mean_val = era_data.mean()
        std_val = era_data.std()
        n = len(era_data)

        print(f"{era}:")
        print(f"  Mean: {mean_val:7.4f}")
        print(f"  Std:  {std_val:7.4f}")
        print(f"  N:    {n}")

print("\n" + "="*60)
print("INTERPRETATION:")
print("If variance DECREASES -> saturation (everyone became aggressive)")
print("If variance STAYS SAME -> equalization (diverse behavior, similar outcomes)")
print("="*60)
