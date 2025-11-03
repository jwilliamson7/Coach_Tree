#!/usr/bin/env python3
"""
Visualize WAR by aggression quintiles across eras
"""

import pandas as pd
import numpy as np
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

plt.rcParams['font.family'] = 'Cambria'

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

eras = ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
colors = ['#2E86AB', '#6A4C93', '#A23B72']

for idx, era in enumerate(eras):
    ax = axes[idx]

    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    # Create quintiles
    era_data['quintile'] = pd.qcut(
        era_data['composite_aggression'],
        q=5,
        labels=['Q1\n(Most\nConservative)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most\nAggressive)'],
        duplicates='drop'
    )

    # Calculate mean WAR for each quintile
    quintile_means = []
    quintile_sems = []
    quintile_ns = []

    for q in ['Q1\n(Most\nConservative)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most\nAggressive)']:
        q_data = era_data[era_data['quintile'] == q]['annual_war']
        if len(q_data) > 0:
            quintile_means.append(q_data.mean())
            quintile_sems.append(q_data.sem())
            quintile_ns.append(len(q_data))
        else:
            quintile_means.append(0)
            quintile_sems.append(0)
            quintile_ns.append(0)

    # Bar plot
    x_pos = np.arange(len(quintile_means))
    bars = ax.bar(x_pos, quintile_means, color=colors[idx], alpha=0.7,
                  edgecolor='black', linewidth=1.5)

    # Error bars (standard error)
    ax.errorbar(x_pos, quintile_means, yerr=quintile_sems,
               fmt='none', ecolor='black', capsize=5, capthick=2, alpha=0.7)

    # Highlight the best quintile
    best_q = np.argmax(quintile_means)
    bars[best_q].set_edgecolor('gold')
    bars[best_q].set_linewidth(4)

    # Add sample sizes on bars
    for i, (mean, n) in enumerate(zip(quintile_means, quintile_ns)):
        if n > 0:
            ax.text(i, mean + (0.01 if mean >= 0 else -0.01), f'n={n}',
                   ha='center', va='bottom' if mean >= 0 else 'top',
                   fontsize=8, fontweight='bold')

    # Reference line at 0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.5)

    # Labels
    ax.set_xlabel('Aggression Quintile', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Annual WAR', fontsize=11, fontweight='bold')
    ax.set_title(f'{era}', fontsize=12, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Q1\n(Most\nConservative)', 'Q2', 'Q3', 'Q4', 'Q5\n(Most\nAggressive)'],
                       fontsize=9)

    # Set consistent y-axis range
    ax.set_ylim(-0.06, 0.06)

    # Add pattern annotation
    if idx == 0:
        # Early era - monotonic increase
        ax.annotate('', xy=(4, 0.03), xytext=(0, -0.04),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkgreen'))
        ax.text(2, -0.055, 'Linear increase', ha='center', fontsize=9,
               style='italic', color='darkgreen', fontweight='bold')
    elif idx == 2:
        # Late era - inverted U
        ax.annotate('', xy=(3, 0.053), xytext=(1, 0.03),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkred',
                                 connectionstyle='arc3,rad=0.3'))
        ax.annotate('', xy=(4, 0.01), xytext=(3, 0.053),
                   arrowprops=dict(arrowstyle='->', lw=2, color='darkred',
                                 connectionstyle='arc3,rad=-0.3'))
        ax.text(2.5, -0.055, 'Inverted U\n(Diminishing returns)', ha='center', fontsize=9,
               style='italic', color='darkred', fontweight='bold')

    ax.grid(True, alpha=0.3, linestyle=':', axis='y')

fig.suptitle('Performance by Aggression Quintile: The Emergence of Diminishing Returns\n(Gold border = best quintile)',
            fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_dir = 'outputs/visualizations/performance'
output_file = output_dir + '/quintile_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_file}")

plt.close()

# Also print the actual values
print("\n" + "="*80)
print("QUINTILE ANALYSIS SUMMARY")
print("="*80)

for era in eras:
    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    era_data['quintile'] = pd.qcut(
        era_data['composite_aggression'],
        q=5,
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
        duplicates='drop'
    )

    print(f"\n{era}:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        q_data = era_data[era_data['quintile'] == q]['annual_war']
        if len(q_data) > 0:
            print(f"  {q}: mean={q_data.mean():7.4f}, sem={q_data.sem():7.4f}, n={len(q_data)}")

    best = era_data.groupby('quintile')['annual_war'].mean().idxmax()
    print(f"  BEST QUINTILE: {best}")

print("\n" + "="*80)
