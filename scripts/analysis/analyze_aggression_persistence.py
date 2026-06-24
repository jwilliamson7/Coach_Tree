#!/usr/bin/env python3
"""
Analyze Aggression Persistence Over Time

This script examines whether coaching aggression is a stable trait by analyzing
whether a coach's aggression in year N predicts their aggression in year N+1.

Usage:
    python analyze_aggression_persistence.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import cluster_bootstrap_corr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressionPersistenceAnalyzer:
    """Analyze year-to-year persistence of coaching aggression"""

    def __init__(self):
        self.aggression_data = None
        self.persistence_data = None

    def load_data(self):
        """Load aggression data by year"""
        logger.info("Loading aggression data...")

        data_file = Path("data/processed/coaching_genes/aggression_gene_by_year.csv")
        if not data_file.exists():
            raise FileNotFoundError(f"Aggression data not found: {data_file}")

        self.aggression_data = pd.read_csv(data_file)
        logger.info(f"Loaded {len(self.aggression_data)} coach-year records")

    def create_lagged_dataset(self, max_lag=3):
        """Create dataset matching year N to year N+lag for each coach"""
        logger.info(f"Creating lagged dataset (up to {max_lag} years)...")

        results = []

        for coach in self.aggression_data['head_coach'].unique():
            coach_data = self.aggression_data[
                self.aggression_data['head_coach'] == coach
            ].sort_values('season').copy()

            # Need at least 2 years for persistence analysis
            if len(coach_data) < 2:
                continue

            for lag in range(1, max_lag + 1):
                # Match year N with year N+lag
                for idx in range(len(coach_data) - lag):
                    year_n = coach_data.iloc[idx]
                    year_n_plus = coach_data.iloc[idx + lag]

                    # Check if years are actually consecutive (or lag years apart)
                    if year_n_plus['season'] - year_n['season'] != lag:
                        continue

                    results.append({
                        'coach': coach,
                        'year_n': year_n['season'],
                        'year_n_plus': year_n_plus['season'],
                        'lag': lag,
                        # Composite
                        'composite_n': year_n['composite_aggression'],
                        'composite_n_plus': year_n_plus['composite_aggression'],
                        # 4th down
                        'fourth_down_n': year_n['fourth_down_aggression'],
                        'fourth_down_n_plus': year_n_plus['fourth_down_aggression'],
                        # Pass-heavy
                        'pass_heavy_n': year_n['pass_heavy_aggression'],
                        'pass_heavy_n_plus': year_n_plus['pass_heavy_aggression'],
                        # Deep pass
                        'deep_pass_n': year_n['deep_pass_aggression'],
                        'deep_pass_n_plus': year_n_plus['deep_pass_aggression'],
                        # 2-point
                        'two_point_n': year_n['two_point_aggression'],
                        'two_point_n_plus': year_n_plus['two_point_aggression'],
                    })

        self.persistence_data = pd.DataFrame(results)
        logger.info(f"Created {len(self.persistence_data)} year-to-year comparisons")

        for lag in range(1, max_lag + 1):
            n = len(self.persistence_data[self.persistence_data['lag'] == lag])
            logger.info(f"  Lag {lag}: {n} comparisons")

    def analyze_persistence(self):
        """Calculate persistence correlations for each lag"""
        logger.info("\nAnalyzing persistence by lag...")

        results = {}

        measures = {
            'composite': 'Composite Aggression',
            'fourth_down': '4th Down Aggression',
            'pass_heavy': 'Pass-Heavy Aggression',
            'deep_pass': 'Deep Pass Aggression',
            'two_point': '2-Point Aggression'
        }

        for lag in sorted(self.persistence_data['lag'].unique()):
            lag_data = self.persistence_data[self.persistence_data['lag'] == lag].copy()
            results[f'lag_{lag}'] = {}

            logger.info(f"\nLag {lag} (Year N to Year N+{lag}):")

            for measure, label in measures.items():
                col_n = f'{measure}_n'
                col_n_plus = f'{measure}_n_plus'

                clean_data = lag_data[[col_n, col_n_plus, 'coach']].dropna()

                if len(clean_data) < 10:
                    continue

                corr, p_val = stats.pearsonr(clean_data[col_n], clean_data[col_n_plus])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    clean_data[col_n], clean_data[col_n_plus]
                )
                # The same coach contributes many year-N/year-N+lag pairs, so
                # cluster the bootstrap on coach rather than treating pairs as
                # independent.
                boot = cluster_bootstrap_corr(
                    clean_data[col_n].values, clean_data[col_n_plus].values,
                    clean_data['coach'].values, n_boot=2000, seed=42,
                )

                results[f'lag_{lag}'][measure] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'ci_low': boot['ci_low'],
                    'ci_high': boot['ci_high'],
                    'p_bootstrap_coach_clustered': boot['p_bootstrap'],
                    'n_coaches': boot['n_clusters'],
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_squared': float(r_value ** 2),
                    'n': int(len(clean_data)),
                    'significant': bool(p_val < 0.05)
                }

                sig = "✓ SIG" if p_val < 0.05 else "n.s."
                logger.info(f"  {label:25s}: r={corr:6.3f}, p={p_val:.4f} ({sig}), n={len(clean_data)}")

        return results

    def create_persistence_visualization(self, results):
        """Create scatter plots showing year-to-year persistence"""
        logger.info("\nCreating persistence visualization...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        # Create 3x5 grid: 3 lags x 5 measures
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))

        measures = [
            ('composite', 'Composite', '#2E86AB'),
            ('fourth_down', '4th Down', '#A23B72'),
            ('pass_heavy', 'Pass-Heavy', '#F18F01'),
            ('deep_pass', 'Deep Pass', '#C73E1D'),
            ('two_point', '2-Point', '#6A994E')
        ]

        def percent_formatter(x, pos):
            return f"{x*100:+.0f}%"

        for lag_idx, lag in enumerate([1, 2, 3]):
            lag_data = self.persistence_data[self.persistence_data['lag'] == lag].copy()

            for measure_idx, (measure, label, color) in enumerate(measures):
                ax = axes[lag_idx, measure_idx]

                col_n = f'{measure}_n'
                col_n_plus = f'{measure}_n_plus'

                clean_data = lag_data[[col_n, col_n_plus]].dropna()

                if len(clean_data) < 10:
                    ax.text(0.5, 0.5, 'Insufficient data',
                           transform=ax.transAxes,
                           ha='center', va='center',
                           fontsize=13, style='italic', color='gray')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                x = clean_data[col_n]
                y = clean_data[col_n_plus]

                # Scatter plot
                ax.scatter(x, y, c=color, alpha=0.4, s=40, edgecolors='black', linewidth=0.3)

                # Regression line
                corr, p_val = stats.pearsonr(x, y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_range = np.array([x.min(), x.max()])
                ax.plot(x_range, slope * x_range + intercept, 'k--', linewidth=2, alpha=0.7)

                # Reference lines
                ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.4)
                ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.4)

                # Add diagonal line (perfect persistence)
                diag_min = max(x.min(), y.min())
                diag_max = min(x.max(), y.max())
                ax.plot([diag_min, diag_max], [diag_min, diag_max],
                       'r:', linewidth=1.5, alpha=0.5, label='Perfect persistence')

                # Labels
                if lag_idx == 2:  # Bottom row
                    ax.set_xlabel(f'Year N', fontsize=12, fontweight='bold')
                if measure_idx == 0:  # Left column
                    ax.set_ylabel(f'Year N+{lag}', fontsize=12, fontweight='bold')

                # Title on top row only
                if lag_idx == 0:
                    ax.set_title(label, fontsize=13, fontweight='bold', pad=8)

                # Format axes
                ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
                ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

                # Statistics box
                significance = "* SIG" if p_val < 0.05 else "n.s."
                stats_text = f'r={corr:.2f}\n{significance}'

                ax.text(0.05, 0.95, stats_text,
                       transform=ax.transAxes,
                       fontsize=11,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5))

                # Add n to bottom right
                ax.text(0.95, 0.05, f'n={len(clean_data)}',
                       transform=ax.transAxes,
                       fontsize=10,
                       horizontalalignment='right',
                       verticalalignment='bottom',
                       color='gray')

                ax.grid(True, alpha=0.2, linestyle=':')

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Save
        output_dir = Path("outputs/visualizations/persistence")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "aggression_persistence_analysis.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def create_persistence_decay_plot(self, results):
        """Create plot showing how persistence decays with lag"""
        logger.info("Creating persistence decay visualization...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        fig, ax = plt.subplots(figsize=(12, 8))

        measures = {
            'composite': ('Composite Aggression', '#2E86AB', 'o'),
            'fourth_down': ('4th Down Aggression', '#A23B72', 's'),
            'pass_heavy': ('Pass-Heavy Aggression', '#F18F01', '^'),
            'deep_pass': ('Deep Pass Aggression', '#C73E1D', 'D'),
            'two_point': ('2-Point Aggression', '#6A994E', 'v')
        }

        for measure, (label, color, marker) in measures.items():
            lags = []
            corrs = []

            for lag in [1, 2, 3]:
                lag_key = f'lag_{lag}'
                if lag_key in results and measure in results[lag_key]:
                    lags.append(lag)
                    corrs.append(results[lag_key][measure]['correlation'])

            if len(lags) > 0:
                ax.plot(lags, corrs, linewidth=2.5, color=color, marker=marker,
                       markersize=10, label=label, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Lag (Years)', fontsize=15, fontweight='bold')
        ax.set_ylabel('Correlation Coefficient (r)', fontsize=15, fontweight='bold')
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['N -> N+1', 'N -> N+2', 'N -> N+3'])
        ax.legend(loc='best', framealpha=0.95, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=':')

        # Add interpretation zones
        ax.axhspan(0.7, 1.0, alpha=0.05, color='green', zorder=0)
        ax.axhspan(0.3, 0.7, alpha=0.05, color='yellow', zorder=0)
        ax.axhspan(0.0, 0.3, alpha=0.05, color='red', zorder=0)

        ax.text(3.1, 0.85, 'Strong\nPersistence', fontsize=11, color='darkgreen', style='italic')
        ax.text(3.1, 0.5, 'Moderate\nPersistence', fontsize=11, color='orange', style='italic')
        ax.text(3.1, 0.15, 'Weak\nPersistence', fontsize=11, color='darkred', style='italic')

        plt.tight_layout()

        # Save
        output_dir = Path("outputs/visualizations/persistence")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "aggression_persistence_decay.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def save_results(self, results):
        """Save persistence analysis results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "aggression_persistence_results.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved persistence results: {output_file}")

    def print_summary(self, results):
        """Print summary of persistence findings"""
        print("\n" + "="*80)
        print("AGGRESSION PERSISTENCE ANALYSIS")
        print("="*80)
        print("\nQuestion: Is aggression a stable trait, or does it vary year-to-year?")
        print("-" * 80)

        measures = {
            'composite': 'Composite Aggression',
            'fourth_down': '4th Down Aggression',
            'pass_heavy': 'Pass-Heavy Aggression',
            'deep_pass': 'Deep Pass Aggression',
            'two_point': '2-Point Aggression'
        }

        for lag in [1, 2, 3]:
            lag_key = f'lag_{lag}'
            if lag_key not in results:
                continue

            print(f"\n{'='*80}")
            print(f"LAG {lag}: Year N -> Year N+{lag}")
            print('='*80)

            for measure, label in measures.items():
                if measure not in results[lag_key]:
                    continue

                stats = results[lag_key][measure]
                sig = "SIGNIFICANT" if stats['significant'] else "not significant"

                print(f"\n{label}:")
                print(f"  Correlation: r = {stats['correlation']:.4f} ({sig})")
                print(f"  P-value: {stats['p_value']:.4f}")
                print(f"  R²: {stats['r_squared']:.4f} ({stats['r_squared']*100:.1f}% of variance explained)")
                print(f"  Sample size: n = {stats['n']}")

                # Interpret the strength
                r = abs(stats['correlation'])
                if r >= 0.7:
                    strength = "VERY STRONG"
                elif r >= 0.5:
                    strength = "STRONG"
                elif r >= 0.3:
                    strength = "MODERATE"
                elif r >= 0.1:
                    strength = "WEAK"
                else:
                    strength = "NEGLIGIBLE"

                print(f"  Persistence: {strength}")

        # Overall conclusions
        print("\n" + "="*80)
        print("KEY FINDINGS:")
        print("="*80)

        # Get lag 1 results for summary
        lag1_results = results.get('lag_1', {})

        if lag1_results:
            print("\nYear-to-Year Stability (N -> N+1):")
            for measure, label in measures.items():
                if measure in lag1_results:
                    r = lag1_results[measure]['correlation']
                    sig = "SIG" if lag1_results[measure]['significant'] else "n.s."
                    print(f"  {label:25s}: r = {r:.3f} ({sig})")

            print("\nInterpretation:")
            # Check if aggression is stable
            stable_count = sum(1 for m in lag1_results.values() if m['correlation'] > 0.5 and m['significant'])
            total_count = len(lag1_results)

            if stable_count >= 4:
                print("  >> Aggression is a STABLE TRAIT - coaches maintain their tendencies")
            elif stable_count >= 2:
                print("  >> Aggression is MODERATELY STABLE - some persistence but variation exists")
            else:
                print("  >> Aggression is SITUATIONAL - coaches adapt significantly year-to-year")

        print("\n" + "="*80)


def main():
    analyzer = AggressionPersistenceAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Create lagged dataset
        analyzer.create_lagged_dataset(max_lag=3)

        # Analyze persistence
        results = analyzer.analyze_persistence()

        # Create visualizations
        analyzer.create_persistence_visualization(results)
        analyzer.create_persistence_decay_plot(results)

        # Save results
        analyzer.save_results(results)

        # Print summary
        analyzer.print_summary(results)

        logger.info("\nPersistence analysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
