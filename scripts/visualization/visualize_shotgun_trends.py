#!/usr/bin/env python3
"""
Visualize Shotgun Gene Trends Over Time

This script creates a line chart showing how shotgun formation usage
(relative to model predictions) has evolved across NFL seasons.

Usage:
    python visualize_shotgun_trends.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Creating shotgun gene trends visualization...")

    # Load shotgun gene data
    shotgun_file = Path("data/processed/coaching_genes/shotgun_gene.csv")
    if not shotgun_file.exists():
        logger.error(f"Shotgun gene data not found: {shotgun_file}")
        logger.error("Please run calculate_shotgun_gene.py first")
        return

    df = pd.read_csv(shotgun_file)
    logger.info(f"Loaded {len(df):,} coach-year records")

    # Calculate yearly averages
    yearly = df.groupby('season').agg({
        'shotgun_gene': 'mean',
        'head_coach': 'count'
    }).reset_index()
    yearly.columns = ['season', 'avg_shotgun_gene', 'num_coaches']

    logger.info(f"Calculated trends for {len(yearly)} seasons")

    # Create visualization
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams['font.family'] = 'Helvetica'
    plt.rcParams['font.size'] = 13  # Base font size

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot shotgun gene trend
    ax.plot(yearly['season'], yearly['avg_shotgun_gene'],
            linewidth=2.5, color='#1E88E5', marker='o', markersize=6,
            label='Shotgun Formation Usage')

    # Add zero reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Format y-axis as percentage
    def percent_formatter(x, pos):
        """Format y-axis as +/-XX.X%"""
        return f"{x*100:+.1f}%"

    ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

    # Labels
    ax.set_xlabel('Season', fontsize=15, fontweight='bold')
    ax.set_ylabel('POE (Percent over Expected)', fontsize=15, fontweight='bold')

    # Set x-axis to show every other year
    years = yearly['season'].astype(int)
    ax.set_xticks(years[::2])
    ax.set_xticklabels(years[::2])

    # Legend and grid
    ax.legend(loc='upper left', framealpha=0.9, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle=':')

    # Add sample size annotation
    min_coaches = yearly['num_coaches'].min()
    max_coaches = yearly['num_coaches'].max()
    ax.text(0.98, 0.02, f'Sample size: {min_coaches}-{max_coaches} coaches per season',
            transform=ax.transAxes,
            fontsize=12,
            horizontalalignment='right',
            verticalalignment='bottom',
            style='italic',
            color='gray')

    plt.tight_layout()

    # Save visualization
    output_dir = Path("outputs/visualizations/trends")
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "shotgun_trends.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved visualization: {png_path}")

    plt.close()

    # Print summary statistics
    first_year = yearly.iloc[0]
    last_year = yearly.iloc[-1]
    total_change = last_year['avg_shotgun_gene'] - first_year['avg_shotgun_gene']

    logger.info(f"\nShotgun Gene Trends Summary:")
    logger.info(f"  {first_year['season']}: {first_year['avg_shotgun_gene']*100:+.2f}% over expected")
    logger.info(f"  {last_year['season']}: {last_year['avg_shotgun_gene']*100:+.2f}% over expected")
    logger.info(f"  Total change: {total_change*100:+.2f} percentage points")

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
