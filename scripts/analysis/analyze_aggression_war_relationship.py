#!/usr/bin/env python3
"""
Analyze Relationship Between Coaching Aggression and WAR

This script analyzes whether coaching aggression (and its components) predicts
coaching performance as measured by Wins Above Replacement (WAR).

Usage:
    python analyze_aggression_war_relationship.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import coach_war_trajectories_path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressionWARAnalyzer:
    """Analyze the relationship between coaching aggression and performance"""

    def __init__(self):
        self.aggression_data = None
        self.war_data = None
        self.merged_data = None

    def load_data(self):
        """Load aggression and WAR datasets"""
        logger.info("Loading datasets...")

        # Load aggression data
        aggression_file = Path("data/processed/coaching_genes/aggression_gene_by_year.csv")
        if not aggression_file.exists():
            raise FileNotFoundError(f"Aggression data not found: {aggression_file}")

        self.aggression_data = pd.read_csv(aggression_file)
        logger.info(f"Loaded {len(self.aggression_data)} coach-year aggression records")

        # Load WAR data
        war_file = coach_war_trajectories_path()
        if not war_file.exists():
            raise FileNotFoundError(f"WAR data not found: {war_file}")

        self.war_data = pd.read_csv(war_file)
        logger.info(f"Loaded {len(self.war_data)} coach-year WAR records")

    def merge_datasets(self):
        """Merge aggression and WAR data on coach name and year"""
        logger.info("Merging datasets...")

        # Standardize column names for merging
        aggression = self.aggression_data.copy()
        aggression.columns = aggression.columns.str.lower()
        aggression = aggression.rename(columns={'head_coach': 'coach', 'season': 'year'})

        war = self.war_data.copy()
        war.columns = war.columns.str.lower()

        # Merge on coach name and year
        self.merged_data = pd.merge(
            aggression,
            war,
            on=['coach', 'year'],
            how='inner'
        )

        logger.info(f"Merged dataset: {len(self.merged_data)} coach-year records")
        logger.info(f"Unique coaches: {self.merged_data['coach'].nunique()}")
        logger.info(f"Year range: {self.merged_data['year'].min()}-{self.merged_data['year'].max()}")

        # Check for missing values
        missing = self.merged_data[['annual_war', 'composite_aggression']].isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected:\n{missing}")

    def run_regressions(self):
        """Run regression analysis for each aggression component vs WAR"""
        logger.info("Running regression analyses...")

        results = {}

        # Define aggression measures to analyze
        measures = {
            'composite_aggression': 'Composite Aggression',
            'fourth_down_aggression': '4th Down Aggression',
            'pass_heavy_aggression': 'Pass-Heavy Aggression',
            'deep_pass_aggression': 'Deep Pass Aggression',
            'two_point_aggression': '2-Point Aggression'
        }

        for col, label in measures.items():
            # Remove any rows with missing values for this analysis
            clean_data = self.merged_data[[col, 'annual_war']].dropna()

            if len(clean_data) == 0:
                logger.warning(f"No valid data for {label}")
                continue

            x = clean_data[col]
            y = clean_data['annual_war'] * 16  # Convert from percentage to games

            # Pearson correlation
            corr, p_val = stats.pearsonr(x, y)

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            results[label] = {
                'correlation': float(corr),
                'p_value': float(p_val),
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value ** 2),
                'std_err': float(std_err),
                'n': int(len(clean_data)),
                'significant': bool(p_val < 0.05)
            }

            sig_marker = "✓ SIGNIFICANT" if p_val < 0.05 else "not significant"
            logger.info(f"{label:25s}: r={corr:7.4f}, p={p_val:.4f} ({sig_marker}), n={len(clean_data)}")

        return results

    def create_visualizations(self, results):
        """Create scatter plots with regression lines for each aggression measure"""
        logger.info("Creating visualizations...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        # Create 2x3 subplot grid (5 aggression measures + 1 for win pct)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes = axes.flatten()

        measures = [
            ('composite_aggression', 'Composite Aggression', '#2E86AB'),
            ('fourth_down_aggression', '4th Down Aggression', '#A23B72'),
            ('pass_heavy_aggression', 'Pass-Heavy Aggression', '#F18F01'),
            ('deep_pass_aggression', 'Deep Pass Aggression', '#C73E1D'),
            ('two_point_aggression', '2-Point Aggression', '#6A994E'),
            ('composite_aggression', 'Composite Agg vs Win %', '#2E86AB')  # Special case for win pct
        ]

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        def war_formatter(x, pos):
            return f"{x:+.1f}"

        for idx, (col, label, color) in enumerate(measures):
            ax = axes[idx]

            # Special handling for win percentage plot
            if idx == 5:
                y_col = 'win_pct'
                y_label = 'Win Percentage'
                clean_data = self.merged_data[[col, y_col]].dropna()
            else:
                y_col = 'annual_war'
                y_label = 'Annual WAR (Games)'
                clean_data = self.merged_data[[col, y_col]].dropna()

            if len(clean_data) == 0:
                ax.text(0.5, 0.5, 'No data available',
                       transform=ax.transAxes,
                       ha='center', va='center',
                       fontsize=15, style='italic', color='gray')
                continue

            x = clean_data[col]
            y = clean_data[y_col] * 16 if idx != 5 else clean_data[y_col]  # Convert WAR to games

            # Scatter plot
            ax.scatter(x, y, c=color, alpha=0.5, s=60, edgecolors='black', linewidth=0.5)

            # Calculate regression
            corr, p_val = stats.pearsonr(x, y)
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Plot regression line
            x_range = np.array([x.min(), x.max()])
            ax.plot(x_range, slope * x_range + intercept,
                   'k--', linewidth=2, alpha=0.7)

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            # Labels
            ax.set_xlabel('Aggression POE', fontsize=13, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
            ax.set_title(label, fontsize=14, fontweight='bold', pad=10)

            # Format x-axis as percentage
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))

            # Format y-axis (win pct is already 0-1, WAR is not)
            if idx == 5:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(war_formatter))

            # Statistics box
            significance = "* SIG" if p_val < 0.05 else "n.s."
            stats_text = f'r = {corr:.3f}\np = {p_val:.4f}\n{significance}\nn = {len(clean_data)}'

            ax.text(0.05, 0.95, stats_text,
                   transform=ax.transAxes,
                   fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

            ax.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout(rect=[0, 0, 1, 0.99])

        # Save visualization
        output_dir = Path("outputs/visualizations/performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "aggression_war_analysis.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def save_results(self, results):
        """Save regression results to JSON"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "aggression_war_regression_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved regression results: {output_file}")

        # Also save merged dataset for further analysis
        data_file = output_dir / "aggression_war_merged_data.csv"
        self.merged_data.to_csv(data_file, index=False)
        logger.info(f"Saved merged dataset: {data_file}")

    def print_summary(self, results):
        """Print summary of findings"""
        print("\n" + "="*80)
        print("AGGRESSION vs WAR ANALYSIS SUMMARY")
        print("="*80)

        print(f"\nDataset: {len(self.merged_data)} coach-year observations")
        print(f"Coaches: {self.merged_data['coach'].nunique()}")
        print(f"Years: {self.merged_data['year'].min()}-{self.merged_data['year'].max()}")

        print("\nKey Findings:")
        print("-" * 80)

        # Sort by correlation strength
        sorted_results = sorted(results.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)

        for measure, stats in sorted_results:
            sig = "SIGNIFICANT" if stats['significant'] else "not significant"
            direction = "positive" if stats['correlation'] > 0 else "negative"

            print(f"\n{measure}:")
            print(f"  Correlation: r = {stats['correlation']:.4f} ({direction})")
            print(f"  P-value: {stats['p_value']:.4f} ({sig})")
            print(f"  R²: {stats['r_squared']:.4f} ({stats['r_squared']*100:.1f}% of variance explained)")
            print(f"  Sample size: n = {stats['n']}")

            if stats['significant']:
                # Interpret the slope
                games_per_10pct = stats['slope'] * 0.1  # Games change per 10% aggression increase
                print(f"  Impact: +10% aggression -> {games_per_10pct:+.3f} games per year")

        print("\n" + "="*80)


def main():
    analyzer = AggressionWARAnalyzer()

    try:
        # Load and merge data
        analyzer.load_data()
        analyzer.merge_datasets()

        # Run regression analyses
        results = analyzer.run_regressions()

        # Create visualizations
        analyzer.create_visualizations(results)

        # Save results
        analyzer.save_results(results)

        # Print summary
        analyzer.print_summary(results)

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
