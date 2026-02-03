#!/usr/bin/env python3
"""
Visualize Aggression Gene Inheritance: Coordinator → Head Coach Only

This script creates a focused analysis of aggression inheritance specifically for
the coordinator→head coach relationship, which is the most direct mentorship path.

Usage:
    python visualize_coordinator_to_hc_inheritance.py
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
    logger.info("Creating Coordinator→HC inheritance visualization...")

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

    from plot_config import configure_plots
    configure_plots()

    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        coord_df['child_aggression'],
        coord_df['parent_aggression'],
        c='#A23B72',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

    corr, p_val = stats.pearsonr(coord_df['child_aggression'], coord_df['parent_aggression'])
    logger.info(f"Correlation: r={corr:.3f}, p={p_val:.4f}")

    x_range = np.array([coord_df['child_aggression'].min(), coord_df['child_aggression'].max()])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        coord_df['child_aggression'], coord_df['parent_aggression']
    )
    ax.plot(x_range, slope * x_range + intercept,
            'k--', linewidth=2.5, alpha=0.8, label=f'Linear fit (r={corr:.3f})')

    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

    ax.set_xlabel('Head Coach Aggression (POE)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Coordinator Aggression (POE)', fontsize=16, fontweight='bold')

    def percent_formatter(x, pos):
        return f"{x*100:+.1f}%"

    ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    significance = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
    stats_text = f"Correlation: r = {corr:.3f}\n"
    stats_text += f"P-value: {p_val:.4f} ({significance})\n"
    stats_text += f"Sample size: n = {len(coord_df)}\n"
    stats_text += f"Linear model: y = {slope:.3f}x + {intercept:.4f}"

    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           fontsize=14,
           verticalalignment='top',
           horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray', linewidth=1.5),
           family='serif')

    ax.legend(loc='lower right', framealpha=0.95, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()

    output_dir = Path("outputs/visualizations/inheritance")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "coordinator_to_hc_inheritance.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved visualization: {png_path}")

    plt.close()

    logger.info(f"Analysis complete! Coordinator→HC correlation: r={corr:.3f}, p={p_val:.4f}")


if __name__ == "__main__":
    main()
