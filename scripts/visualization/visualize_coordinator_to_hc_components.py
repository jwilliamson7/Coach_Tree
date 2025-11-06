#!/usr/bin/env python3
"""
Visualize Aggression Component Inheritance: Coordinator → Head Coach Only

This script creates a 2x2 grid showing inheritance for each of the four aggression
components, filtered to only coordinator→head coach relationships.

Usage:
    python visualize_coordinator_to_hc_components.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Creating Coordinator→HC component inheritance visualization...")

    inheritance_file = Path("outputs/visualizations/inheritance/aggression_inheritance_data.csv")
    if not inheritance_file.exists():
        logger.error(f"Inheritance data not found: {inheritance_file}")
        logger.error("Please run visualize_aggression_inheritance.py first")
        return

    df = pd.read_csv(inheritance_file)
    logger.info(f"Loaded {len(df):,} total relationships")

    coord_df = df[df['relationship_type'] == 'coordinator_to_hc'].copy()
    logger.info(f"Filtered to {len(coord_df):,} Coordinator→HC relationships")

    if len(coord_df) == 0:
        logger.error("No coordinator_to_hc relationships found!")
        return

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 13  # Base font size

    components = [
        ('fourth_down', '4th Down Aggression', '#A23B72'),
        ('pass_heavy', 'Pass-Heavy Aggression', '#F18F01'),
        ('deep_pass', 'Deep Pass Aggression', '#C73E1D'),
        ('two_point', '2-Point Conversion Aggression', '#6A994E')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes = axes.flatten()

    def percent_formatter(x, pos):
        return f"{x*100:+.1f}%"

    for idx, (comp_name, comp_label, color) in enumerate(components):
        ax = axes[idx]

        parent_col = f'parent_{comp_name}'
        child_col = f'child_{comp_name}'

        ax.scatter(coord_df[child_col], coord_df[parent_col],
                  c=color, alpha=0.6, s=80, edgecolors='black', linewidth=0.5)

        corr, p_val = stats.pearsonr(coord_df[child_col], coord_df[parent_col])
        logger.info(f"{comp_label}: r={corr:.3f}, p={p_val:.4f}")

        x_range = np.array([coord_df[child_col].min(), coord_df[child_col].max()])
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            coord_df[child_col], coord_df[parent_col]
        )
        ax.plot(x_range, slope * x_range + intercept,
               'k--', linewidth=2, alpha=0.7)

        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Head Coach POE', fontsize=14, fontweight='bold')
        ax.set_ylabel('Coordinator POE', fontsize=14, fontweight='bold')
        ax.set_title(comp_label, fontsize=15, fontweight='bold', pad=10)

        ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

        significance = "* SIG" if p_val < 0.05 else "n.s."
        stats_text = f'r = {corr:.3f}\np = {p_val:.4f}\n{significance}\nn = {len(coord_df)}'

        ax.text(0.05, 0.95, stats_text,
               transform=ax.transAxes,
               fontsize=13,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

        ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    output_dir = Path("outputs/visualizations/inheritance")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "coordinator_to_hc_components.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved visualization: {png_path}")

    plt.close()

    logger.info("Component analysis complete!")


if __name__ == "__main__":
    main()
