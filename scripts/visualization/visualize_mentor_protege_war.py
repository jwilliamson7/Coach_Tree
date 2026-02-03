#!/usr/bin/env python3
"""
Visualize Mentor WAR → Protégé WAR Relationship

Creates scatter plot showing the relationship between mentor (HC) performance
and protégé (coordinator who later became HC) performance.

Usage:
    python scripts/visualization/visualize_mentor_protege_war.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import stats
from scipy.stats import linregress

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_visualization():
    """Create mentor-protégé WAR scatter plot"""

    # Load data
    data_file = Path("outputs/analysis/mentor_protege_war_pairs.csv")
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Please run: python scripts/analysis/analyze_mentor_war_protege_war.py"
        )

    df = pd.read_csv(data_file)
    logger.info(f"Loaded {len(df)} mentor-protégé pairs")

    # Set up matplotlib style
    from plot_config import configure_plots
    configure_plots()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Separate by coordinator type for coloring
    oc_mask = df['protege_role_under_mentor'] == 'Offensive Coordinator'
    dc_mask = df['protege_role_under_mentor'] == 'Defensive Coordinator'

    # Plot scatter points
    ax.scatter(
        df[oc_mask]['avg_war_mentor'],
        df[oc_mask]['avg_war_protege'],
        alpha=0.6,
        s=80,
        color='#F18F01',
        edgecolors='black',
        linewidth=0.5,
        label=f'Offensive Coordinator (n={oc_mask.sum()})',
        zorder=3
    )

    ax.scatter(
        df[dc_mask]['avg_war_mentor'],
        df[dc_mask]['avg_war_protege'],
        alpha=0.6,
        s=80,
        color='#6A4C93',
        edgecolors='black',
        linewidth=0.5,
        label=f'Defensive Coordinator (n={dc_mask.sum()})',
        zorder=3
    )

    # Calculate regression line
    slope, intercept, r_value, p_value, std_err = linregress(
        df['avg_war_mentor'], df['avg_war_protege']
    )

    # Plot regression line
    x_range = np.linspace(df['avg_war_mentor'].min(), df['avg_war_mentor'].max(), 100)
    y_pred = slope * x_range + intercept

    ax.plot(
        x_range,
        y_pred,
        color='#2E86AB',
        linewidth=2.5,
        linestyle='--',
        label=f'Regression Line (R² = {r_value**2:.3f})',
        zorder=2
    )

    # Add reference lines at 0
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, zorder=1)

    # Add text box with statistics
    r_overall, p_overall = stats.pearsonr(df['avg_war_mentor'], df['avg_war_protege'])

    if p_overall < 0.001:
        sig = "***"
    elif p_overall < 0.01:
        sig = "**"
    elif p_overall < 0.05:
        sig = "*"
    else:
        sig = "n.s."

    stats_text = (
        f'n = {len(df)} pairs\n'
        f'r = {r_overall:.3f} {sig}\n'
        f'p = {p_overall:.4f}\n'
        f'Slope = {slope:.3f}'
    )

    ax.text(
        0.05, 0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
    )

    # Labels and title
    ax.set_xlabel('Mentor Average WAR (as Head Coach)', fontsize=15, fontweight='bold')
    ax.set_ylabel('Protégé Average WAR (as Head Coach)', fontsize=15, fontweight='bold')
    ax.set_title('Does Mentor Performance Predict Protégé Performance?',
                 fontsize=17, fontweight='bold', pad=20)

    # Add subtitle
    ax.text(
        0.5, 1.02,
        'Head Coach → Coordinator relationships where coordinator later became HC',
        transform=ax.transAxes,
        fontsize=12,
        ha='center',
        style='italic'
    )

    # Legend
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)

    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

    # Format axes to show as games (multiply by 16)
    from matplotlib.ticker import FuncFormatter

    def war_formatter(x, pos):
        """Format WAR as +/- games"""
        games = x * 16
        return f"{games:+.1f}"

    ax.xaxis.set_major_formatter(FuncFormatter(war_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(war_formatter))

    # Add secondary axis labels showing it's in games
    ax.text(
        1.01, 0.5,
        'Games',
        transform=ax.transAxes,
        fontsize=11,
        rotation=270,
        verticalalignment='center'
    )

    ax.text(
        0.5, -0.08,
        'Games',
        transform=ax.transAxes,
        fontsize=11,
        horizontalalignment='center'
    )

    plt.tight_layout()

    # Save
    output_dir = Path("outputs/visualizations/performance")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "mentor_protege_war.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved visualization: {output_file}")

    plt.close()

    # Also create a version by era
    create_by_era_visualization(df)


def create_by_era_visualization(df):
    """Create visualization split by era"""

    logger.info("Creating by-era visualization...")

    # Define eras
    df['era'] = pd.cut(
        df['relationship_year'],
        bins=[0, 2000, 2010, 2020, 3000],
        labels=['Pre-2000', '2000-2009', '2010-2019', '2020+']
    )

    # Set up matplotlib style
    from plot_config import configure_plots
    configure_plots()

    # Create subplots
    eras = ['Pre-2000', '2000-2009', '2010-2019', '2020+']
    colors = ['#A23B72', '#F18F01', '#2E86AB', '#6A994E']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (era, color) in enumerate(zip(eras, colors)):
        ax = axes[idx]
        era_data = df[df['era'] == era].dropna(subset=['avg_war_mentor', 'avg_war_protege'])

        if len(era_data) < 5:
            ax.text(0.5, 0.5, f'{era}\nInsufficient Data (n={len(era_data)})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue

        # Scatter plot
        ax.scatter(
            era_data['avg_war_mentor'],
            era_data['avg_war_protege'],
            alpha=0.6,
            s=80,
            color=color,
            edgecolors='black',
            linewidth=0.5,
            zorder=3
        )

        # Regression line
        if len(era_data) >= 10:
            slope, intercept, r_value, p_value, std_err = linregress(
                era_data['avg_war_mentor'], era_data['avg_war_protege']
            )

            x_range = np.linspace(era_data['avg_war_mentor'].min(),
                                 era_data['avg_war_mentor'].max(), 100)
            y_pred = slope * x_range + intercept

            ax.plot(
                x_range,
                y_pred,
                color='black',
                linewidth=2,
                linestyle='--',
                alpha=0.7,
                zorder=2
            )

            # Statistics
            r, p = stats.pearsonr(era_data['avg_war_mentor'], era_data['avg_war_protege'])

            if p < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            stats_text = f'n = {len(era_data)}\nr = {r:.3f} {sig}\np = {p:.3f}'

            ax.text(
                0.05, 0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
            )

        # Reference lines
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.4, zorder=1)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.4, zorder=1)

        # Labels
        ax.set_xlabel('Mentor WAR (Games)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Protégé WAR (Games)', fontsize=12, fontweight='bold')
        ax.set_title(era, fontsize=14, fontweight='bold', pad=10)

        # Format as games
        from matplotlib.ticker import FuncFormatter

        def war_formatter(x, pos):
            return f"{x*16:+.0f}"

        ax.xaxis.set_major_formatter(FuncFormatter(war_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(war_formatter))

        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', zorder=0)

    plt.suptitle('Mentor-Protégé WAR Relationship by Era',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    # Save
    output_dir = Path("outputs/visualizations/performance")
    output_file = output_dir / "mentor_protege_war_by_era.png"

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved by-era visualization: {output_file}")

    plt.close()


def main():
    """Main execution"""
    logger.info("Creating mentor-protégé WAR visualizations...")
    create_visualization()
    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
