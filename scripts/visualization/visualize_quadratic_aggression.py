#!/usr/bin/env python3
"""
Visualize quadratic relationship between aggression and WAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

# Load data
war_file = 'outputs/analysis/aggression_war_merged_data.csv'
merged_data = pd.read_csv(war_file)

# Define eras
merged_data['era'] = pd.cut(
    merged_data['year'],
    bins=[2005, 2011, 2017, 2025],
    labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
)

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 13  # Base font size

fig, axes = plt.subplots(1, 3, figsize=(22, 7))

eras = ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
colors = ['#2E86AB', '#F18F01', '#A23B72']

def percent_formatter(x, pos):
    return f"{x*100:+.1f}%"

def war_formatter(x, pos):
    return f"{x:+.1f}"

for idx, era in enumerate(eras):
    ax = axes[idx]

    era_data = merged_data[merged_data['era'] == era].dropna(subset=['composite_aggression', 'annual_war'])

    x = era_data['composite_aggression'].values
    y = era_data['annual_war'].values * 16  # Convert from percentage to games

    # Scatter plot
    ax.scatter(x, y, c=colors[idx], alpha=0.4, s=80, edgecolors='black', linewidth=0.5)

    # Linear regression
    slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope_lin * x_line + intercept_lin
    ax.plot(x_line, y_line, '--', color='gray', linewidth=2, alpha=0.7, label=f'Linear: r={r_lin:.3f}')

    # Quadratic regression
    coeffs = np.polyfit(x, y, 2)
    y_quad = np.polyval(coeffs, x_line)
    r_quad = np.corrcoef(y, np.polyval(coeffs, x))[0, 1]
    ax.plot(x_line, y_quad, '-', color=colors[idx], linewidth=3,
           label=f'Quadratic: r={r_quad:.3f}')

    # Find peak of quadratic (vertex)
    vertex_x = -coeffs[1] / (2 * coeffs[0])
    vertex_y = np.polyval(coeffs, vertex_x)

    # Mark peak if within data range
    if x.min() <= vertex_x <= x.max():
        ax.plot(vertex_x, vertex_y, '*', color='red', markersize=20,
               markeredgecolor='black', markeredgewidth=2, label='Optimal point')

    # Reference lines
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Labels
    ax.set_xlabel('Composite Aggression (POE)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Annual WAR (Games)', fontsize=15, fontweight='bold')
    ax.set_title(f'{era}', fontsize=16, fontweight='bold', pad=15)

    # Format axes
    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(war_formatter))

    # Calculate quadratic p-value using F-test
    # Residual sum of squares for linear model
    y_pred_lin = slope_lin * x + intercept_lin
    ss_res_lin = np.sum((y - y_pred_lin) ** 2)

    # Residual sum of squares for quadratic model
    y_pred_quad = np.polyval(coeffs, x)
    ss_res_quad = np.sum((y - y_pred_quad) ** 2)

    # F-test for improvement
    n = len(x)
    f_stat = ((ss_res_lin - ss_res_quad) / 1) / (ss_res_quad / (n - 3))
    p_quad = 1 - stats.f.cdf(f_stat, 1, n - 3)

    # Stats box
    stats_text = f'Linear: r={r_lin:.3f}, p={p_lin:.4f}\n'
    stats_text += f'Quadratic: r={r_quad:.3f}, p={p_quad:.4f}\n'
    stats_text += f'Coeff (x²): {coeffs[0]:.2f}\n'
    stats_text += f'n = {len(era_data)}'

    if coeffs[0] < 0:
        shape_text = 'Inverted U (diminishing returns)'
    else:
        shape_text = 'U-shape'

    ax.text(0.05, 0.95, stats_text,
           transform=ax.transAxes,
           fontsize=13,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

    ax.text(0.95, 0.05, shape_text,
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='bottom',
           horizontalalignment='right',
           style='italic',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_dir = 'outputs/visualizations/performance'
output_file = output_dir + '/quadratic_aggression_war.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_file}")

plt.close()
