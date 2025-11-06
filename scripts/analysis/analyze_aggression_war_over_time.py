#!/usr/bin/env python3
"""
Analyze How Aggression-WAR Relationship Changed Over Time

This script examines whether the relationship between coaching aggression
and performance (WAR) has strengthened, weakened, or shifted over time.

Focuses on the two significant aggression measures:
- Composite Aggression
- Pass-Heavy Aggression

Usage:
    python analyze_aggression_war_over_time.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TemporalAggressionAnalyzer:
    """Analyze how aggression-WAR relationship evolved over time"""

    def __init__(self):
        self.data = None

    def load_data(self):
        """Load the merged aggression-WAR dataset"""
        logger.info("Loading merged aggression-WAR data...")

        data_file = Path("outputs/analysis/aggression_war_merged_data.csv")
        if not data_file.exists():
            raise FileNotFoundError(
                f"Merged data not found: {data_file}\n"
                "Please run analyze_aggression_war_relationship.py first"
            )

        self.data = pd.read_csv(data_file)
        logger.info(f"Loaded {len(self.data)} coach-year observations")
        logger.info(f"Year range: {self.data['year'].min()}-{self.data['year'].max()}")

    def analyze_by_era(self):
        """Analyze correlations by era (early/mid/late)"""
        logger.info("\nAnalyzing by era...")

        # Define eras
        self.data['era'] = pd.cut(
            self.data['year'],
            bins=[2005, 2011, 2017, 2025],
            labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
        )

        results = {}

        for era in ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']:
            era_data = self.data[self.data['era'] == era].copy()

            if len(era_data) < 10:
                continue

            results[era] = {}

            # Composite aggression
            comp_data = era_data[['composite_aggression', 'annual_war']].dropna()
            if len(comp_data) > 0:
                corr, p_val = stats.pearsonr(comp_data['composite_aggression'], comp_data['annual_war'])
                results[era]['composite'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(comp_data)),
                    'significant': bool(p_val < 0.05)
                }

            # Pass-heavy aggression
            pass_data = era_data[['pass_heavy_aggression', 'annual_war']].dropna()
            if len(pass_data) > 0:
                corr, p_val = stats.pearsonr(pass_data['pass_heavy_aggression'], pass_data['annual_war'])
                results[era]['pass_heavy'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(pass_data)),
                    'significant': bool(p_val < 0.05)
                }

            logger.info(f"\n{era}:")
            logger.info(f"  Composite:  r={results[era]['composite']['correlation']:7.4f}, "
                       f"p={results[era]['composite']['p_value']:.4f}, "
                       f"n={results[era]['composite']['n']}")
            logger.info(f"  Pass-Heavy: r={results[era]['pass_heavy']['correlation']:7.4f}, "
                       f"p={results[era]['pass_heavy']['p_value']:.4f}, "
                       f"n={results[era]['pass_heavy']['n']}")

        return results

    def analyze_rolling_correlation(self, window=3):
        """Calculate rolling correlation over time"""
        logger.info(f"\nCalculating rolling {window}-year correlations...")

        years = sorted(self.data['year'].unique())

        rolling_results = {
            'composite': [],
            'pass_heavy': []
        }

        for year in years:
            # Get data for this year +/- window
            year_data = self.data[
                (self.data['year'] >= year - window) &
                (self.data['year'] <= year + window)
            ].copy()

            if len(year_data) < 20:  # Need minimum sample size
                continue

            # Composite aggression
            comp_data = year_data[['composite_aggression', 'annual_war']].dropna()
            if len(comp_data) >= 20:
                corr, p_val = stats.pearsonr(comp_data['composite_aggression'], comp_data['annual_war'])
                rolling_results['composite'].append({
                    'year': int(year),
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(comp_data))
                })

            # Pass-heavy aggression
            pass_data = year_data[['pass_heavy_aggression', 'annual_war']].dropna()
            if len(pass_data) >= 20:
                corr, p_val = stats.pearsonr(pass_data['pass_heavy_aggression'], pass_data['annual_war'])
                rolling_results['pass_heavy'].append({
                    'year': int(year),
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(pass_data))
                })

        return rolling_results

    def create_era_visualization(self, era_results):
        """Create scatter plots by era"""
        logger.info("Creating era comparison visualization...")

        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 13  # Base font size

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        eras = ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
        colors = ['#3498db', '#e67e22', '#e74c3c']

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        def war_formatter(x, pos):
            return f"{x:+.1f}"

        # Row 1: Composite aggression
        for idx, (era, color) in enumerate(zip(eras, colors)):
            ax = axes[0, idx]

            era_data = self.data[self.data['era'] == era].copy()
            clean_data = era_data[['composite_aggression', 'annual_war']].dropna()

            if len(clean_data) == 0:
                continue

            x = clean_data['composite_aggression']
            y = clean_data['annual_war'] * 16  # Convert from percentage to games

            # Scatter plot
            ax.scatter(x, y, c=color, alpha=0.5, s=60, edgecolors='black', linewidth=0.5)

            # Regression line
            corr, p_val = stats.pearsonr(x, y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_range = np.array([x.min(), x.max()])
            ax.plot(x_range, slope * x_range + intercept, 'k--', linewidth=2, alpha=0.7)

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            # Labels
            ax.set_xlabel('Composite Aggression POE', fontsize=13, fontweight='bold')
            ax.set_ylabel('Annual WAR (Games)', fontsize=13, fontweight='bold')
            ax.set_title(f'{era}', fontsize=14, fontweight='bold', pad=10)

            # Format x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

            # Format y-axis
            ax.yaxis.set_major_formatter(FuncFormatter(war_formatter))

            # Statistics box
            significance = "SIG" if p_val < 0.05 else "n.s."
            stats_text = f'r = {corr:.3f}\np = {p_val:.4f}\n{significance}\nn = {len(clean_data)}'

            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes,
                   fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

            ax.grid(True, alpha=0.3, linestyle=':')

        # Row 2: Pass-heavy aggression
        for idx, (era, color) in enumerate(zip(eras, colors)):
            ax = axes[1, idx]

            era_data = self.data[self.data['era'] == era].copy()
            clean_data = era_data[['pass_heavy_aggression', 'annual_war']].dropna()

            if len(clean_data) == 0:
                continue

            x = clean_data['pass_heavy_aggression']
            y = clean_data['annual_war'] * 16  # Convert from percentage to games

            # Scatter plot
            ax.scatter(x, y, c=color, alpha=0.5, s=60, edgecolors='black', linewidth=0.5)

            # Regression line
            corr, p_val = stats.pearsonr(x, y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_range = np.array([x.min(), x.max()])
            ax.plot(x_range, slope * x_range + intercept, 'k--', linewidth=2, alpha=0.7)

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            # Labels
            ax.set_xlabel('Pass-Heavy Aggression POE', fontsize=13, fontweight='bold')
            ax.set_ylabel('Annual WAR (Games)', fontsize=13, fontweight='bold')
            ax.set_title(f'{era}', fontsize=14, fontweight='bold', pad=10)

            # Format x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

            # Format y-axis
            ax.yaxis.set_major_formatter(FuncFormatter(war_formatter))

            # Statistics box
            significance = "SIG" if p_val < 0.05 else "n.s."
            stats_text = f'r = {corr:.3f}\np = {p_val:.4f}\n{significance}\nn = {len(clean_data)}'

            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes,
                   fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

            ax.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save
        output_dir = Path("outputs/visualizations/performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "aggression_war_by_era.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def create_rolling_correlation_plot(self, rolling_results):
        """Create plot showing how correlation changed over time"""
        logger.info("Creating rolling correlation visualization...")

        plt.rcParams['font.family'] = 'Helvetica'
        plt.rcParams['font.size'] = 13  # Base font size

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Composite aggression
        comp_df = pd.DataFrame(rolling_results['composite'])

        ax1.plot(comp_df['year'], comp_df['correlation'],
                linewidth=2.5, color='#2E86AB', marker='o', markersize=6,
                label='Correlation Coefficient')

        # Mark significant periods
        sig_periods = comp_df[comp_df['p_value'] < 0.05]
        if len(sig_periods) > 0:
            ax1.scatter(sig_periods['year'], sig_periods['correlation'],
                       s=100, color='#e74c3c', marker='*', zorder=5,
                       label='Significant (p<0.05)')

        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Year (center of 3-year window)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Correlation Coefficient (r)', fontsize=14, fontweight='bold')
        ax1.set_title('Composite Aggression vs WAR', fontsize=15, fontweight='bold', pad=10)
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Pass-heavy aggression
        pass_df = pd.DataFrame(rolling_results['pass_heavy'])

        ax2.plot(pass_df['year'], pass_df['correlation'],
                linewidth=2.5, color='#F18F01', marker='o', markersize=6,
                label='Correlation Coefficient')

        # Mark significant periods
        sig_periods = pass_df[pass_df['p_value'] < 0.05]
        if len(sig_periods) > 0:
            ax2.scatter(sig_periods['year'], sig_periods['correlation'],
                       s=100, color='#e74c3c', marker='*', zorder=5,
                       label='Significant (p<0.05)')

        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Year (center of 3-year window)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Correlation Coefficient (r)', fontsize=14, fontweight='bold')
        ax2.set_title('Pass-Heavy Aggression vs WAR', fontsize=15, fontweight='bold', pad=10)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle=':')
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Save
        output_dir = Path("outputs/visualizations/performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "aggression_war_rolling_correlation.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def save_results(self, era_results, rolling_results):
        """Save temporal analysis results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'by_era': era_results,
            'rolling_correlation': rolling_results
        }

        output_file = output_dir / "aggression_war_temporal_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved temporal analysis results: {output_file}")

    def print_summary(self, era_results, rolling_results):
        """Print summary of temporal findings"""
        print("\n" + "="*80)
        print("TEMPORAL ANALYSIS: HOW AGGRESSION-WAR LINK CHANGED OVER TIME")
        print("="*80)

        print("\nERA ANALYSIS:")
        print("-" * 80)

        for era in ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']:
            if era not in era_results:
                continue

            print(f"\n{era}:")

            comp = era_results[era]['composite']
            print(f"  Composite Aggression:")
            print(f"    r = {comp['correlation']:.4f}, p = {comp['p_value']:.4f} "
                  f"({'SIGNIFICANT' if comp['significant'] else 'not significant'}), n = {comp['n']}")

            pass_h = era_results[era]['pass_heavy']
            print(f"  Pass-Heavy Aggression:")
            print(f"    r = {pass_h['correlation']:.4f}, p = {pass_h['p_value']:.4f} "
                  f"({'SIGNIFICANT' if pass_h['significant'] else 'not significant'}), n = {pass_h['n']}")

        print("\n" + "-" * 80)
        print("KEY INSIGHTS:")
        print("-" * 80)

        # Analyze trends
        eras = ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
        comp_corrs = [era_results[era]['composite']['correlation'] for era in eras if era in era_results]
        pass_corrs = [era_results[era]['pass_heavy']['correlation'] for era in eras if era in era_results]

        if len(comp_corrs) >= 2:
            if comp_corrs[-1] > comp_corrs[0]:
                print("\nComposite Aggression: STRENGTHENING relationship over time")
                print(f"  Early: r={comp_corrs[0]:.3f} -> Late: r={comp_corrs[-1]:.3f} "
                      f"(change: {comp_corrs[-1]-comp_corrs[0]:+.3f})")
            else:
                print("\nComposite Aggression: WEAKENING relationship over time")
                print(f"  Early: r={comp_corrs[0]:.3f} -> Late: r={comp_corrs[-1]:.3f} "
                      f"(change: {comp_corrs[-1]-comp_corrs[0]:+.3f})")

        if len(pass_corrs) >= 2:
            if pass_corrs[-1] > pass_corrs[0]:
                print("\nPass-Heavy Aggression: STRENGTHENING relationship over time")
                print(f"  Early: r={pass_corrs[0]:.3f} -> Late: r={pass_corrs[-1]:.3f} "
                      f"(change: {pass_corrs[-1]-pass_corrs[0]:+.3f})")
            else:
                print("\nPass-Heavy Aggression: WEAKENING relationship over time")
                print(f"  Early: r={pass_corrs[0]:.3f} -> Late: r={pass_corrs[-1]:.3f} "
                      f"(change: {pass_corrs[-1]-pass_corrs[0]:+.3f})")

        print("\n" + "="*80)


def main():
    analyzer = TemporalAggressionAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Analyze by era
        era_results = analyzer.analyze_by_era()

        # Rolling correlation
        rolling_results = analyzer.analyze_rolling_correlation(window=3)

        # Create visualizations
        analyzer.create_era_visualization(era_results)
        analyzer.create_rolling_correlation_plot(rolling_results)

        # Save results
        analyzer.save_results(era_results, rolling_results)

        # Print summary
        analyzer.print_summary(era_results, rolling_results)

        logger.info("\nTemporal analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
