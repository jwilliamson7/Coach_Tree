#!/usr/bin/env python3
"""
Analyze Aggression Persistence by Coach Background Type

This script examines whether aggression persistence (year-to-year stability)
differs between offensive coaches and defensive coaches.

Usage:
    python analyze_persistence_by_coach_type.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from scipy import stats
import matplotlib.pyplot as plt
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import cluster_bootstrap_corr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PersistenceByTypeAnalyzer:
    """Analyze aggression persistence by coach background type"""

    def __init__(self):
        self.data = None

    def load_data(self):
        """Load aggression data with coach type classifications"""
        logger.info("Loading aggression data with coach types...")

        data_file = Path("outputs/analysis/aggression_war_with_coach_type.csv")
        if not data_file.exists():
            raise FileNotFoundError(
                f"Coach type data not found: {data_file}\n"
                "Please run analyze_aggression_by_coach_type.py first"
            )

        self.data = pd.read_csv(data_file)
        logger.info(f"Loaded {len(self.data)} coach-year observations")

        # Filter to only coaches with multiple years
        coach_years = self.data.groupby('coach')['year'].count()
        multi_year_coaches = coach_years[coach_years >= 2].index
        self.data = self.data[self.data['coach'].isin(multi_year_coaches)]

        logger.info(f"Filtered to {len(self.data)} observations from coaches with 2+ years")

    def calculate_persistence_by_type(self, max_lag=3):
        """Calculate year-to-year persistence for each coach type"""
        logger.info(f"\nCalculating persistence (up to {max_lag}-year lag) by coach type...")

        aggression_vars = [
            'fourth_down_aggression',
            'pass_heavy_aggression',
            'deep_pass_aggression',
            'two_point_aggression',
            'composite_aggression'
        ]

        results = {}

        for bg_type in ['Offensive', 'Defensive']:
            logger.info(f"\n{bg_type} Coaches:")
            results[bg_type] = {}

            bg_data = self.data[self.data['Background'] == bg_type].copy()
            bg_data = bg_data.sort_values(['coach', 'year'])

            for var in aggression_vars:
                results[bg_type][var] = []

                for lag in range(1, max_lag + 1):
                    pairs = []

                    for coach in bg_data['coach'].unique():
                        coach_data = bg_data[bg_data['coach'] == coach].sort_values('year')

                        if len(coach_data) < lag + 1:
                            continue

                        # Match year N with year N+lag
                        for idx in range(len(coach_data) - lag):
                            year_n = coach_data.iloc[idx]
                            year_n_plus = coach_data.iloc[idx + lag]

                            # Verify consecutive years
                            if year_n_plus['year'] - year_n['year'] != lag:
                                continue

                            # Get values
                            val_n = year_n[var]
                            val_n_plus = year_n_plus[var]

                            if pd.notna(val_n) and pd.notna(val_n_plus):
                                pairs.append({
                                    'year_n': val_n,
                                    'year_n_plus': val_n_plus,
                                    'coach': coach,
                                    'lag': lag
                                })

                    if len(pairs) >= 10:
                        pairs_df = pd.DataFrame(pairs)
                        corr, p_val = stats.pearsonr(pairs_df['year_n'], pairs_df['year_n_plus'])
                        # Cluster the bootstrap on coach (same coach -> many pairs).
                        boot = cluster_bootstrap_corr(
                            pairs_df['year_n'].values, pairs_df['year_n_plus'].values,
                            pairs_df['coach'].values, n_boot=2000, seed=42,
                        )

                        results[bg_type][var].append({
                            'lag': int(lag),
                            'correlation': float(corr),
                            'p_value': float(p_val),
                            'ci_low': boot['ci_low'],
                            'ci_high': boot['ci_high'],
                            'p_bootstrap_coach_clustered': boot['p_bootstrap'],
                            'n_coaches': boot['n_clusters'],
                            'n_pairs': int(len(pairs)),
                            'significant': bool(p_val < 0.05)
                        })

                        sig_marker = "SIG" if p_val < 0.05 else "n.s."
                        logger.info(f"  {var} (lag={lag}): r={corr:.3f}, p={p_val:.4f} ({sig_marker}), n={len(pairs)}")

        return results

    def compare_types(self, results):
        """Compare persistence between offensive and defensive coaches"""
        logger.info("\n\nCOMPARING OFFENSIVE vs DEFENSIVE COACHES:")
        logger.info("="*80)

        aggression_vars = [
            'fourth_down_aggression',
            'pass_heavy_aggression',
            'deep_pass_aggression',
            'two_point_aggression',
            'composite_aggression'
        ]

        comparison = {}

        for var in aggression_vars:
            logger.info(f"\n{var}:")

            comparison[var] = {}

            for lag in [1, 2, 3]:
                off_data = [x for x in results['Offensive'][var] if x['lag'] == lag]
                def_data = [x for x in results['Defensive'][var] if x['lag'] == lag]

                if len(off_data) > 0 and len(def_data) > 0:
                    off_corr = off_data[0]['correlation']
                    def_corr = def_data[0]['correlation']
                    diff = off_corr - def_corr

                    comparison[var][f'lag_{lag}'] = {
                        'offensive_corr': float(off_corr),
                        'defensive_corr': float(def_corr),
                        'difference': float(diff)
                    }

                    logger.info(f"  Lag {lag}: Offensive r={off_corr:.3f}, Defensive r={def_corr:.3f}, "
                               f"Diff={diff:+.3f}")

        return comparison

    def create_visualization(self, results):
        """Create visualization comparing persistence by coach type"""
        logger.info("\nCreating persistence comparison visualization...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        aggression_vars = [
            ('fourth_down_aggression', '4th Down'),
            ('pass_heavy_aggression', 'Pass-Heavy'),
            ('deep_pass_aggression', 'Deep Pass'),
            ('two_point_aggression', 'Two-Point'),
            ('composite_aggression', 'Composite')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        colors = {'Offensive': '#FF6B35', 'Defensive': '#004E89'}
        markers = {'Offensive': 'o', 'Defensive': 's'}

        for idx, (var, label) in enumerate(aggression_vars):
            ax = axes[idx]

            for bg_type in ['Offensive', 'Defensive']:
                if var not in results[bg_type] or len(results[bg_type][var]) == 0:
                    continue

                data = results[bg_type][var]
                lags = [x['lag'] for x in data]
                corrs = [x['correlation'] for x in data]

                ax.plot(lags, corrs,
                       marker=markers[bg_type],
                       markersize=10,
                       linewidth=2.5,
                       color=colors[bg_type],
                       label=bg_type,
                       alpha=0.8)

                # Mark significant points
                for point in data:
                    if point['significant']:
                        ax.scatter([point['lag']], [point['correlation']],
                                 s=200, marker='*', color=colors[bg_type],
                                 zorder=5, edgecolors='black', linewidth=0.5)

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_xlabel('Lag (Years)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Correlation (r)', fontsize=14, fontweight='bold')
            ax.set_title(f'{label} Aggression', fontsize=15, fontweight='bold', pad=10)
            ax.set_xticks([1, 2, 3])
            ax.set_ylim(-0.1, 0.8)
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='best', framealpha=0.95)

        # Hide extra subplot
        axes[5].axis('off')

        fig.suptitle('Aggression Persistence by Coach Background Type\n(Stars indicate p<0.05)',
                    fontsize=19, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Save
        output_dir = Path("outputs/visualizations/persistence")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "persistence_by_coach_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def create_comparison_heatmap(self, comparison):
        """Create heatmap showing difference in persistence (Offensive - Defensive)"""
        logger.info("Creating difference heatmap...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        aggression_vars = [
            ('fourth_down_aggression', '4th Down'),
            ('pass_heavy_aggression', 'Pass-Heavy'),
            ('deep_pass_aggression', 'Deep Pass'),
            ('two_point_aggression', 'Two-Point'),
            ('composite_aggression', 'Composite')
        ]

        # Create matrix
        matrix = []
        labels = []

        for var, label in aggression_vars:
            if var not in comparison:
                continue
            labels.append(label)
            row = []
            for lag in [1, 2, 3]:
                key = f'lag_{lag}'
                if key in comparison[var]:
                    row.append(comparison[var][key]['difference'])
                else:
                    row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)

        # Set ticks
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['1 Year', '2 Years', '3 Years'])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Difference in Correlation\n(Offensive - Defensive)',
                      rotation=270, labelpad=25, fontweight='bold')

        # Add text annotations
        for i in range(len(labels)):
            for j in range(3):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:+.3f}',
                                 ha='center', va='center',
                                 color='white' if abs(matrix[i, j]) > 0.08 else 'black',
                                 fontsize=14, fontweight='bold')

        ax.set_xlabel('Lag Period', fontsize=15, fontweight='bold')
        ax.set_ylabel('Aggression Type', fontsize=15, fontweight='bold')
        ax.set_title('Persistence Difference: Offensive vs Defensive Coaches\n(Positive = More Persistent for Offensive)',
                    fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout()

        # Save
        output_dir = Path("outputs/visualizations/persistence")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "persistence_difference_heatmap.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved heatmap: {png_path}")

        plt.close()

    def save_results(self, results, comparison):
        """Save results to JSON"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        all_results = {
            'by_type': results,
            'comparison': comparison
        }

        output_file = output_dir / "persistence_by_coach_type_results.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Saved results: {output_file}")

    def print_summary(self, results, comparison):
        """Print summary of findings"""
        print("\n" + "="*80)
        print("AGGRESSION PERSISTENCE BY COACH BACKGROUND TYPE")
        print("="*80)

        print("\nAVERAGE 1-YEAR PERSISTENCE:")
        print("-" * 80)

        aggression_vars = [
            'composite_aggression',
            'fourth_down_aggression',
            'pass_heavy_aggression',
            'deep_pass_aggression',
            'two_point_aggression'
        ]

        for var in aggression_vars:
            print(f"\n{var}:")

            for bg_type in ['Offensive', 'Defensive']:
                if var in results[bg_type] and len(results[bg_type][var]) > 0:
                    lag1 = [x for x in results[bg_type][var] if x['lag'] == 1]
                    if len(lag1) > 0:
                        r = lag1[0]['correlation']
                        p = lag1[0]['p_value']
                        n = lag1[0]['n_pairs']
                        sig = "SIG" if lag1[0]['significant'] else "n.s."
                        print(f"  {bg_type}: r={r:.3f}, p={p:.4f} ({sig}), n={n}")

        print("\n" + "-" * 80)
        print("KEY INSIGHTS:")
        print("-" * 80)

        # Find biggest differences
        max_diff = 0
        max_diff_var = None
        max_diff_lag = None

        for var in aggression_vars:
            if var in comparison:
                for lag in [1, 2, 3]:
                    key = f'lag_{lag}'
                    if key in comparison[var]:
                        diff = abs(comparison[var][key]['difference'])
                        if diff > max_diff:
                            max_diff = diff
                            max_diff_var = var
                            max_diff_lag = lag

        if max_diff_var:
            diff_data = comparison[max_diff_var][f'lag_{max_diff_lag}']
            print(f"\nLargest difference: {max_diff_var} at {max_diff_lag}-year lag")
            print(f"  Offensive: r={diff_data['offensive_corr']:.3f}")
            print(f"  Defensive: r={diff_data['defensive_corr']:.3f}")
            print(f"  Difference: {diff_data['difference']:+.3f}")

        print("\n" + "="*80)


def main():
    analyzer = PersistenceByTypeAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Calculate persistence
        results = analyzer.calculate_persistence_by_type(max_lag=3)

        # Compare types
        comparison = analyzer.compare_types(results)

        # Create visualizations
        analyzer.create_visualization(results)
        analyzer.create_comparison_heatmap(comparison)

        # Save results
        analyzer.save_results(results, comparison)

        # Print summary
        analyzer.print_summary(results, comparison)

        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
