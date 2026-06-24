#!/usr/bin/env python3
"""
Visualize Gene Inheritance: Coordinator -> Head Coach

Creates figures for the paper showing how coaching genes persist (or don't)
when coordinators become head coaches. Uses gene_inheritance.csv data.

Produces:
  1. 2x2 scatter panel: coordinator-era vs HC-era gene for all 4 types
  2. Summary bar chart: correlation strength and direction retention

Usage:
    python scripts/visualization/visualize_gene_inheritance.py
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

# Must be imported before matplotlib to set up fonts
import sys
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import configure_plots
configure_plots()

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter


# Gene display configuration
GENE_CONFIG = {
    'defensive_scheme': {
        'label': 'Defensive Scheme',
        'short': 'Def. Scheme',
        'color': '#2E86AB',
        'transition': 'DC->HC',
        'unit': 'Composite Z-score',
    },
    'shotgun': {
        'label': 'Shotgun Formation',
        'short': 'Shotgun',
        'color': '#A23B72',
        'transition': 'OC->HC',
        'unit': 'Z-score',
    },
    'aggression': {
        'label': 'Aggression',
        'short': 'Aggression',
        'color': '#F18F01',
        'transition': 'OC->HC',
        'unit': 'POE',
    },
    'tempo': {
        'label': 'Tempo',
        'short': 'Tempo',
        'color': '#C73E1D',
        'transition': 'OC->HC',
        'unit': 'Composite Z-score',
    },
}


def load_data():
    """Load gene inheritance CSV."""
    path = Path("data/processed/coaching_genes/gene_inheritance.csv")
    if not path.exists():
        raise FileNotFoundError(f"Gene inheritance data not found: {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} inheritance records")
    return df


def aggregate_coordinator_stints(df):
    """Average all coordinator stints per coach, weighted by years of data.

    Coaches with multiple coordinator stints (e.g., Brian Daboll was OC at
    CLE, MIA, KAN, BUF before becoming HC at NYG) have their coordinator-era
    gene averaged across all stints, weighted by coord_years_with_data. This
    avoids pseudo-replication of the HC-era outcome while using all data years.
    """
    results = []
    for (coach, gene), group in df.groupby(['coach_name', 'gene_type']):
        weights = group['coord_years_with_data'].values
        coord_gene = np.average(group['coord_era_gene'].values, weights=weights)
        hc_gene = group['hc_era_gene'].iloc[0]
        results.append({
            'coach_name': coach,
            'gene_type': gene,
            'coord_era_gene': coord_gene,
            'hc_era_gene': hc_gene,
            'total_coord_years': weights.sum(),
            'num_stints': len(group),
            'gene_change': hc_gene - coord_gene,
        })
    return pd.DataFrame(results)


def create_scatter_panel(df, output_dir):
    """Create 2x2 scatter panel: coordinator-era vs HC-era gene."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    gene_order = ['defensive_scheme', 'shotgun', 'aggression', 'tempo']

    # Aggregate all stints up front
    agg_df = aggregate_coordinator_stints(df)

    for idx, gene_key in enumerate(gene_order):
        ax = axes[idx]
        config = GENE_CONFIG[gene_key]

        gene_df = agg_df[agg_df['gene_type'] == gene_key].copy()

        if len(gene_df) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        x = gene_df['coord_era_gene'].values
        y = gene_df['hc_era_gene'].values

        # Scatter
        ax.scatter(x, y, c=config['color'], s=80, alpha=0.6,
                   edgecolors='black', linewidth=0.5, zorder=3)

        # Regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept,
                'k--', linewidth=2, alpha=0.7)

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Pearson correlation
        r, p = stats.pearsonr(x, y)
        n = len(gene_df)

        # Direction retention
        same_sign = np.sum(np.sign(x) == np.sign(y))
        dir_pct = 100 * same_sign / n

        # Significance marker
        sig = ''
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'

        # Stats annotation
        stats_text = f"r = {r:.3f}{sig}\np = {p:.4f}\nn = {n}"
        ax.text(0.03, 0.97, stats_text,
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          alpha=0.9, edgecolor='gray'))

        # Labels
        ax.set_xlabel(f'Coordinator-Era Gene ({config["unit"]})', fontsize=12)
        ax.set_ylabel(f'HC-Era Gene ({config["unit"]})', fontsize=12)
        ax.set_title(f'{config["label"]} ({config["transition"]}, n={n})',
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, linestyle=':')

        # Label notable coaches
        gene_df['abs_change'] = np.abs(gene_df['gene_change'])
        for _, row in gene_df.nlargest(3, 'abs_change').iterrows():
            # Shorten name for label
            name = row['coach_name'].split()[-1]
            ax.annotate(name,
                        (row['coord_era_gene'], row['hc_era_gene']),
                        fontsize=8, alpha=0.7,
                        xytext=(5, 5), textcoords='offset points')

    plt.tight_layout(pad=2.0)

    out_path = output_dir / "gene_inheritance_scatter.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {out_path}")
    plt.close()
    return out_path


def create_summary_chart(df, output_dir):
    """Create summary bar chart comparing inheritance across gene types."""
    gene_order = ['defensive_scheme', 'shotgun', 'aggression', 'tempo']

    correlations = []
    p_values = []
    dir_retentions = []
    sample_sizes = []
    labels = []
    colors = []

    agg_df = aggregate_coordinator_stints(df)

    for gene_key in gene_order:
        config = GENE_CONFIG[gene_key]
        gene_df = agg_df[agg_df['gene_type'] == gene_key].copy()

        if len(gene_df) < 5:
            continue

        x = gene_df['coord_era_gene'].values
        y = gene_df['hc_era_gene'].values
        r, p = stats.pearsonr(x, y)
        n = len(gene_df)
        same_sign = np.sum(np.sign(x) == np.sign(y))
        dir_pct = 100 * same_sign / n

        correlations.append(r)
        p_values.append(p)
        dir_retentions.append(dir_pct)
        sample_sizes.append(n)
        labels.append(config['short'])
        colors.append(config['color'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    x_pos = np.arange(len(labels))
    bar_width = 0.6

    # Left panel: Correlation coefficients
    bars1 = ax1.bar(x_pos, correlations, bar_width, color=colors,
                    edgecolor='black', linewidth=0.8, alpha=0.85)

    # Add significance markers
    for i, (r, p, n) in enumerate(zip(correlations, p_values, sample_sizes)):
        sig = ''
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'

        y_offset = 0.02 if r >= 0 else -0.04
        ax1.text(i, r + y_offset, f'{r:.3f}{sig}',
                 ha='center', va='bottom' if r >= 0 else 'top',
                 fontsize=12, fontweight='bold')
        ax1.text(i, -0.02, f'n={n}', ha='center', va='top',
                 fontsize=10, color='gray')

    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_ylabel('Pearson Correlation (r)', fontsize=13)
    ax1.set_title('Inheritance Strength', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.1, max(correlations) + 0.15)
    ax1.grid(True, axis='y', alpha=0.2, linestyle=':')

    # Right panel: Direction retention
    bars2 = ax2.bar(x_pos, dir_retentions, bar_width, color=colors,
                    edgecolor='black', linewidth=0.8, alpha=0.85)

    # 50% chance line
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1.5,
                label='Chance (50%)')

    for i, (pct, n) in enumerate(zip(dir_retentions, sample_sizes)):
        ax2.text(i, pct + 1.5, f'{pct:.0f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.set_ylabel('Direction Retention (%)', fontsize=13)
    ax2.set_title('Consistency of Sign', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, axis='y', alpha=0.2, linestyle=':')

    plt.tight_layout(pad=2.0)

    out_path = output_dir / "gene_inheritance_summary.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved: {out_path}")
    plt.close()
    return out_path


def main():
    logger.info("Creating gene inheritance visualizations...")

    df = load_data()

    output_dir = Path("outputs/visualizations/inheritance")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 2x2 scatter panel
    scatter_path = create_scatter_panel(df, output_dir)

    # Figure 2: Summary comparison
    summary_path = create_summary_chart(df, output_dir)

    logger.info("Gene inheritance visualizations complete!")
    logger.info(f"  Scatter panel: {scatter_path}")
    logger.info(f"  Summary chart: {summary_path}")


if __name__ == "__main__":
    main()
