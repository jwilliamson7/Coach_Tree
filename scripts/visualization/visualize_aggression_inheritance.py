#!/usr/bin/env python3
"""
Visualize Aggression Gene Inheritance: Mentor vs Protégé Analysis

This script tests the core hypothesis: Do aggressive coaches breed aggressive coaches?
Creates a scatter plot showing the relationship between mentor (parent) and protégé
(child) aggression scores, broken down by relationship type.

Usage:
    python visualize_aggression_inheritance.py [--output_dir outputs/visualizations/inheritance]
"""

import argparse
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


class AggressionInheritanceAnalyzer:
    """Analyze aggression gene inheritance through coaching relationships"""

    def __init__(self, tree_dir: str = "data/processed/coaching_tree",
                 gene_dir: str = "data/processed/coaching_genes",
                 output_dir: str = "outputs/visualizations/inheritance"):
        self.tree_dir = Path(tree_dir)
        self.gene_dir = Path(gene_dir)
        self.output_dir = Path(output_dir)
        self.relationships_df = None
        self.aggression_data = None
        self.inheritance_data = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> None:
        """Load coaching relationships and aggression gene data"""
        logger.info("Loading data...")

        relationships_file = self.tree_dir / "relationships.csv"
        if not relationships_file.exists():
            raise FileNotFoundError(f"Relationships file not found: {relationships_file}")

        self.relationships_df = pd.read_csv(relationships_file)
        logger.info(f"Loaded {len(self.relationships_df):,} coaching relationships")

        aggression_file = self.gene_dir / "aggression_gene_by_year.csv"
        if not aggression_file.exists():
            raise FileNotFoundError(f"Aggression gene file not found: {aggression_file}")

        self.aggression_data = pd.read_csv(aggression_file)
        logger.info(f"Loaded {len(self.aggression_data):,} coach-year aggression records")

    def calculate_average_aggression(self) -> pd.DataFrame:
        """Calculate average aggression scores per coach across all years"""
        logger.info("Calculating average aggression per coach...")

        coach_avg = self.aggression_data.groupby('head_coach').agg({
            'composite_aggression': 'mean',
            'fourth_down_aggression': 'mean',
            'pass_heavy_aggression': 'mean',
            'deep_pass_aggression': 'mean',
            'two_point_aggression': 'mean',
            'season': 'count'
        }).rename(columns={'season': 'num_seasons'})

        logger.info(f"Calculated averages for {len(coach_avg)} coaches")
        return coach_avg.reset_index()

    def create_inheritance_dataset(self) -> pd.DataFrame:
        """Create dataset matching parent and child aggression scores"""
        logger.info("Creating parent-child aggression dataset...")

        coach_avg = self.calculate_average_aggression()

        parent_agg = coach_avg.rename(columns={
            'head_coach': 'parent_name',
            'composite_aggression': 'parent_aggression',
            'fourth_down_aggression': 'parent_fourth_down',
            'pass_heavy_aggression': 'parent_pass_heavy',
            'deep_pass_aggression': 'parent_deep_pass',
            'two_point_aggression': 'parent_two_point',
            'num_seasons': 'parent_seasons'
        })

        child_agg = coach_avg.rename(columns={
            'head_coach': 'child_name',
            'composite_aggression': 'child_aggression',
            'fourth_down_aggression': 'child_fourth_down',
            'pass_heavy_aggression': 'child_pass_heavy',
            'deep_pass_aggression': 'child_deep_pass',
            'two_point_aggression': 'child_two_point',
            'num_seasons': 'child_seasons'
        })

        unique_relationships = self.relationships_df.drop_duplicates(
            subset=['parent_name', 'child_name', 'relationship_type']
        )

        inheritance = unique_relationships.merge(
            parent_agg, on='parent_name', how='inner'
        ).merge(
            child_agg, on='child_name', how='inner'
        )

        logger.info(f"Created {len(inheritance):,} parent-child pairs with aggression data")
        logger.info(f"Relationship types: {inheritance['relationship_type'].value_counts().to_dict()}")

        self.inheritance_data = inheritance
        return inheritance

    def plot_inheritance_analysis(self) -> None:
        """Create scatter plot of parent vs child aggression"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        logger.info("Creating inheritance visualization...")

        from plot_config import configure_plots
        configure_plots()

        df = self.inheritance_data

        relationship_types = {
            'coordinator_to_hc': {'label': 'Coordinator → HC', 'color': '#A23B72', 'marker': 'o'},
            'position_to_coordinator': {'label': 'Position → Coordinator', 'color': '#F18F01', 'marker': 's'},
            'position_to_hc': {'label': 'Position → HC', 'color': '#2E86AB', 'marker': '^'}
        }

        fig, ax = plt.subplots(figsize=(12, 10))

        for rel_type, style in relationship_types.items():
            subset = df[df['relationship_type'] == rel_type]
            if len(subset) > 0:
                ax.scatter(
                    subset['parent_aggression'],
                    subset['child_aggression'],
                    c=style['color'],
                    marker=style['marker'],
                    s=80,
                    alpha=0.6,
                    edgecolors='black',
                    linewidth=0.5,
                    label=f"{style['label']} (n={len(subset)})"
                )

        overall_corr, overall_p = stats.pearsonr(df['parent_aggression'], df['child_aggression'])
        logger.info(f"Overall correlation: r={overall_corr:.3f}, p={overall_p:.4f}")

        x_range = np.array([df['parent_aggression'].min(), df['parent_aggression'].max()])
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            df['parent_aggression'], df['child_aggression']
        )
        ax.plot(x_range, slope * x_range + intercept,
                'k--', linewidth=2, alpha=0.7, label=f'Overall fit (r={overall_corr:.3f})')

        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Mentor Aggression (POE)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Protégé Aggression (POE)', fontsize=16, fontweight='bold')

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        from matplotlib.ticker import FuncFormatter
        ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

        ax.legend(loc='upper left', framealpha=0.95, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle=':')

        stats_text = f"Overall: r = {overall_corr:.3f}, p = {overall_p:.4f}\n"
        stats_text += f"Sample size: {len(df)} relationships\n"

        for rel_type, style in relationship_types.items():
            subset = df[df['relationship_type'] == rel_type]
            if len(subset) >= 3:
                corr, p = stats.pearsonr(subset['parent_aggression'], subset['child_aggression'])
                stats_text += f"{style['label']}: r = {corr:.3f}, p = {p:.4f}\n"

        ax.text(0.98, 0.02, stats_text,
               transform=ax.transAxes,
               fontsize=13,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

        plt.tight_layout()

        png_path = self.output_dir / "aggression_inheritance.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved PNG image: {png_path}")

        plt.close()

    def create_component_analysis(self) -> None:
        """Create faceted plot showing inheritance for each aggression component"""
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        logger.info("Creating component-wise inheritance analysis...")

        from plot_config import configure_plots
        configure_plots()

        df = self.inheritance_data

        components = [
            ('fourth_down', '4th Down Aggression'),
            ('pass_heavy', 'Pass-Heavy Aggression'),
            ('deep_pass', 'Deep Pass Aggression'),
            ('two_point', '2-Point Conversion Aggression')
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes = axes.flatten()

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        for idx, (comp_name, comp_label) in enumerate(components):
            ax = axes[idx]

            parent_col = f'parent_{comp_name}'
            child_col = f'child_{comp_name}'

            ax.scatter(df[parent_col], df[child_col],
                      c='#2E86AB', alpha=0.5, s=60, edgecolors='black', linewidth=0.5)

            corr, p_val = stats.pearsonr(df[parent_col], df[child_col])

            x_range = np.array([df[parent_col].min(), df[parent_col].max()])
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df[parent_col], df[child_col]
            )
            ax.plot(x_range, slope * x_range + intercept,
                   'r--', linewidth=2, alpha=0.7)

            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            ax.set_xlabel('Mentor POE', fontsize=14, fontweight='bold')
            ax.set_ylabel('Protégé POE', fontsize=14, fontweight='bold')
            ax.set_title(comp_label, fontsize=15, fontweight='bold', pad=10)

            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

            ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}\nn = {len(df)}',
                   transform=ax.transAxes,
                   fontsize=13,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

            ax.grid(True, alpha=0.3, linestyle=':')

        plt.tight_layout()

        png_path = self.output_dir / "aggression_inheritance_components.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved component analysis: {png_path}")

        plt.close()

    def create_summary_stats(self) -> None:
        """Create summary statistics file"""
        logger.info("Creating summary statistics...")

        df = self.inheritance_data

        overall_corr, overall_p = stats.pearsonr(df['parent_aggression'], df['child_aggression'])

        summary = {
            'overall': {
                'correlation': float(overall_corr),
                'p_value': float(overall_p),
                'sample_size': int(len(df)),
                'significant': bool(overall_p < 0.05)
            },
            'by_relationship_type': {},
            'by_component': {}
        }

        for rel_type in df['relationship_type'].unique():
            subset = df[df['relationship_type'] == rel_type]
            if len(subset) >= 3:
                corr, p = stats.pearsonr(subset['parent_aggression'], subset['child_aggression'])
                summary['by_relationship_type'][rel_type] = {
                    'correlation': float(corr),
                    'p_value': float(p),
                    'sample_size': int(len(subset)),
                    'significant': bool(p < 0.05)
                }

        components = [
            ('fourth_down', '4th Down'),
            ('pass_heavy', 'Pass-Heavy'),
            ('deep_pass', 'Deep Pass'),
            ('two_point', '2-Point')
        ]

        for comp_name, comp_label in components:
            parent_col = f'parent_{comp_name}'
            child_col = f'child_{comp_name}'
            corr, p = stats.pearsonr(df[parent_col], df[child_col])
            summary['by_component'][comp_label] = {
                'correlation': float(corr),
                'p_value': float(p),
                'significant': bool(p < 0.05)
            }

        import json
        summary_path = self.output_dir / "aggression_inheritance_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary statistics: {summary_path}")

        csv_path = self.output_dir / "aggression_inheritance_data.csv"
        self.inheritance_data.to_csv(csv_path, index=False)
        logger.info(f"Saved inheritance data: {csv_path}")

    def run(self) -> None:
        """Execute the full analysis pipeline"""
        logger.info("Starting aggression inheritance analysis...")

        self.load_data()
        self.create_inheritance_dataset()
        self.plot_inheritance_analysis()
        self.create_component_analysis()
        self.create_summary_stats()

        logger.info("Analysis complete!")
        logger.info(f"Output directory: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze aggression gene inheritance in coaching relationships'
    )
    parser.add_argument(
        '--tree_dir',
        type=str,
        default='data/processed/coaching_tree',
        help='Directory containing coaching tree data'
    )
    parser.add_argument(
        '--gene_dir',
        type=str,
        default='data/processed/coaching_genes',
        help='Directory containing aggression gene data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/visualizations/inheritance',
        help='Output directory for visualizations'
    )

    args = parser.parse_args()

    analyzer = AggressionInheritanceAnalyzer(
        tree_dir=args.tree_dir,
        gene_dir=args.gene_dir,
        output_dir=args.output_dir
    )

    analyzer.run()


if __name__ == "__main__":
    main()
