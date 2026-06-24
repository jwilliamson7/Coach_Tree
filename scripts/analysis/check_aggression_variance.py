#!/usr/bin/env python3
"""
Check variance in aggression over time

Tests whether correlation erosion is due to:
1. Saturation (everyone became aggressive, variance decreased)
2. Equalization (diverse behavior persists, but returns equalized)

Outputs:
    outputs/analysis/aggression_variance_by_era.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Load aggression data
agg_data = pd.read_csv('data/processed/coaching_genes/aggression_gene_by_year.csv')

# Calculate variance by era
agg_data['era'] = pd.cut(
    agg_data['season'],
    bins=[2005, 2011, 2017, 2025],
    labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
)

results = {'generated_date': datetime.now().isoformat(), 'components': {}}

print("VARIANCE IN AGGRESSION OVER TIME")
print("="*60)

components = ['fourth_down_aggression', 'pass_heavy_aggression',
              'deep_pass_aggression', 'two_point_aggression',
              'composite_aggression']

for component in components:
    print(f"\n{component}:")
    print("-"*60)

    comp_results = {}
    for era in ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']:
        era_data = agg_data[agg_data['era'] == era][component].dropna()

        mean_val = era_data.mean()
        std_val = era_data.std()
        n = len(era_data)

        print(f"{era}:")
        print(f"  Mean: {mean_val:7.4f}")
        print(f"  Std:  {std_val:7.4f}")
        print(f"  N:    {n}")

        comp_results[era] = {
            'mean': float(mean_val),
            'std': float(std_val),
            'n': int(n),
        }

    # Compute early-to-late SD change
    early_std = comp_results['Early (2006-2011)']['std']
    late_std = comp_results['Late (2018-2024)']['std']
    pct_change = 100 * (late_std - early_std) / early_std

    comp_results['sd_change_early_to_late_pct'] = round(float(pct_change), 1)
    results['components'][component] = comp_results

    print(f"  SD change (Early -> Late): {pct_change:+.1f}%")

print("\n" + "="*60)
print("INTERPRETATION:")
print("If variance DECREASES -> saturation (everyone became aggressive)")
print("If variance STAYS SAME -> equalization (diverse behavior, similar outcomes)")
print("="*60)

# Save results
output_dir = Path("outputs/analysis")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "aggression_variance_by_era.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {output_file}")
