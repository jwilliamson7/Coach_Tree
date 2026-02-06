#!/usr/bin/env python3
"""
Analyze Shotgun Gene Inheritance by Coach Background Type

This script examines whether inheritance of shotgun formation usage differs
between offensive and defensive coaches. Do offensive coaches pass on their
shotgun preferences more than defensive coaches?

Usage:
    python analyze_shotgun_inheritance_by_coach_type.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ShotgunInheritanceAnalyzer:
    """Analyze shotgun gene inheritance by coach background type"""

    def __init__(self):
        self.shotgun_data = None
        self.coach_types = None
        self.relationships = None
        self.coach_coordinator_types = None

    def load_data(self):
        """Load all necessary data"""
        logger.info("Loading data...")

        # Load shotgun gene data
        shotgun_file = Path("data/processed/coaching_genes/shotgun_gene.csv")
        if not shotgun_file.exists():
            raise FileNotFoundError(
                f"Shotgun gene data not found: {shotgun_file}\n"
                "Please run calculate_shotgun_gene.py first"
            )

        self.shotgun_data = pd.read_csv(shotgun_file)
        logger.info(f"Loaded {len(self.shotgun_data)} coach-year observations")

        # Load coach background types from aggression data
        type_file = Path("outputs/analysis/aggression_war_with_coach_type.csv")
        if not type_file.exists():
            raise FileNotFoundError(
                f"Coach type data not found: {type_file}\n"
                "Please run analyze_aggression_by_coach_type.py first"
            )

        coach_type_data = pd.read_csv(type_file)

        # Create mapping of coach -> background
        coach_backgrounds = coach_type_data.groupby('coach')['Background'].first().to_dict()

        # Add background to shotgun data
        self.shotgun_data['Background'] = self.shotgun_data['head_coach'].map(coach_backgrounds)

        # Log coverage
        has_background = self.shotgun_data['Background'].notna().sum()
        logger.info(f"Matched background for {has_background}/{len(self.shotgun_data)} observations")

        # Calculate average shotgun gene for each coach (across their career)
        coach_avg = self.shotgun_data.groupby('head_coach').agg({
            'shotgun_gene': 'mean',
            'shotgun_gene_zscore': 'mean',
            'Background': 'first'
        }).reset_index()
        coach_avg.columns = ['coach', 'shotgun_gene', 'shotgun_gene_zscore', 'Background']

        self.coach_types = coach_avg
        logger.info(f"Calculated average shotgun gene for {len(self.coach_types)} coaches")

        # Load coaching relationships
        rel_file = Path("data/processed/coaching_tree/relationships.csv")
        if not rel_file.exists():
            raise FileNotFoundError(f"Relationships file not found: {rel_file}")

        self.relationships = pd.read_csv(rel_file)
        logger.info(f"Loaded {len(self.relationships)} coaching relationships")

        # Identify coordinator types from all relationships
        self.identify_coordinator_types()

    def identify_coordinator_types(self):
        """Identify which coaches were OC vs DC by looking at their roles"""
        logger.info("Identifying coordinator types from role history...")

        # Look at all relationships where someone was a coordinator
        all_roles = self.relationships[['parent_name', 'parent_role']].copy()
        all_roles.columns = ['coach', 'role']

        child_roles = self.relationships[['child_name', 'child_role']].copy()
        child_roles.columns = ['coach', 'role']

        all_roles = pd.concat([all_roles, child_roles], ignore_index=True)

        # Filter to coordinator roles
        coordinators = all_roles[
            all_roles['role'].str.contains('Coordinator', case=False, na=False)
        ].copy()

        # Classify as OC or DC
        coord_types = {}
        for coach in coordinators['coach'].unique():
            coach_roles = coordinators[coordinators['coach'] == coach]['role'].values

            has_oc = any('Offensive' in str(role) for role in coach_roles)
            has_dc = any('Defensive' in str(role) for role in coach_roles)

            if has_oc and not has_dc:
                coord_types[coach] = 'OC'
            elif has_dc and not has_oc:
                coord_types[coach] = 'DC'
            elif has_oc and has_dc:
                coord_types[coach] = 'Both'
            else:
                coord_types[coach] = 'Other'

        self.coach_coordinator_types = coord_types
        logger.info(f"Identified coordinator types for {len(coord_types)} coaches")

        # Print distribution
        type_counts = pd.Series(coord_types).value_counts()
        logger.info("\nCoordinator type distribution:")
        for coord_type, count in type_counts.items():
            logger.info(f"  {coord_type}: {count}")

    def create_mentor_protege_pairs(self):
        """Create mentor-protege pairs with shotgun gene data for both"""
        logger.info("\nCreating mentor-protege pairs...")

        # Filter to coordinator -> head coach relationships
        coord_to_hc = self.relationships[
            self.relationships['relationship_type'] == 'coordinator_to_hc'
        ].copy()

        logger.info(f"Found {len(coord_to_hc)} coordinator -> head coach transitions")

        pairs = []

        for _, rel in coord_to_hc.iterrows():
            mentor_name = rel['parent_name']
            protege_name = rel['child_name']

            # Get mentor shotgun gene
            mentor_data = self.coach_types[self.coach_types['coach'] == mentor_name]
            if len(mentor_data) == 0:
                continue

            # Get protege shotgun gene
            protege_data = self.coach_types[self.coach_types['coach'] == protege_name]
            if len(protege_data) == 0:
                continue

            mentor = mentor_data.iloc[0]
            protege = protege_data.iloc[0]

            # Get coordinator type from our mapping
            coord_type = self.coach_coordinator_types.get(mentor_name, 'Other')

            pair = {
                'mentor_name': mentor_name,
                'protege_name': protege_name,
                'mentor_background': mentor['Background'],
                'protege_background': protege['Background'],
                'mentor_coordinator_type': coord_type,
                'shotgun_mentor': mentor['shotgun_gene'],
                'shotgun_protege': protege['shotgun_gene'],
                'shotgun_zscore_mentor': mentor['shotgun_gene_zscore'],
                'shotgun_zscore_protege': protege['shotgun_gene_zscore']
            }

            pairs.append(pair)

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"Created {len(pairs_df)} complete mentor-protege pairs")

        # Show distribution by mentor type
        logger.info("\nMentor background distribution:")
        for bg, count in pairs_df['mentor_background'].value_counts(dropna=False).items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {bg}: {count} ({pct:.1f}%)")

        logger.info("\nMentor coordinator type distribution:")
        for coord_type, count in pairs_df['mentor_coordinator_type'].value_counts().items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {coord_type}: {count} ({pct:.1f}%)")

        return pairs_df

    def analyze_overall(self, pairs_df):
        """Analyze overall shotgun gene inheritance"""
        logger.info("\n\nANALYZING OVERALL SHOTGUN INHERITANCE:")
        logger.info("="*80)

        clean_data = pairs_df[['shotgun_mentor', 'shotgun_protege']].dropna()

        if len(clean_data) >= 10:
            corr, p_val = stats.pearsonr(clean_data['shotgun_mentor'], clean_data['shotgun_protege'])

            result = {
                'correlation': float(corr),
                'p_value': float(p_val),
                'n': int(len(clean_data)),
                'significant': bool(p_val < 0.05)
            }

            sig_marker = "SIG" if p_val < 0.05 else "n.s."
            logger.info(f"Overall: r={corr:.3f}, p={p_val:.4f} ({sig_marker}), n={len(clean_data)}")

            return result
        else:
            logger.info(f"Insufficient data (n={len(clean_data)})")
            return None

    def analyze_by_mentor_type(self, pairs_df):
        """Analyze inheritance by mentor's background type"""
        logger.info("\n\nANALYZING INHERITANCE BY MENTOR TYPE:")
        logger.info("="*80)

        results = {}

        for bg_type in ['Offensive', 'Defensive']:
            logger.info(f"\n{bg_type} Mentors:")
            results[bg_type] = {}

            bg_pairs = pairs_df[pairs_df['mentor_background'] == bg_type]

            if len(bg_pairs) < 10:
                logger.info(f"  Insufficient data (n={len(bg_pairs)})")
                continue

            clean_data = bg_pairs[['shotgun_mentor', 'shotgun_protege']].dropna()

            if len(clean_data) >= 10:
                corr, p_val = stats.pearsonr(clean_data['shotgun_mentor'], clean_data['shotgun_protege'])

                results[bg_type] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(clean_data)),
                    'significant': bool(p_val < 0.05)
                }

                sig_marker = "SIG" if p_val < 0.05 else "n.s."
                logger.info(f"  Shotgun: r={corr:.3f}, p={p_val:.4f} ({sig_marker}), n={len(clean_data)}")

        return results

    def analyze_by_coordinator_type(self, pairs_df):
        """Analyze separately for OC vs DC mentors"""
        logger.info("\n\nANALYZING BY COORDINATOR TYPE:")
        logger.info("="*80)

        results = {}

        for coord_type in ['OC', 'DC']:
            logger.info(f"\n{coord_type} mentors:")
            results[coord_type] = {}

            coord_pairs = pairs_df[pairs_df['mentor_coordinator_type'] == coord_type]

            if len(coord_pairs) < 10:
                logger.info(f"  Insufficient data (n={len(coord_pairs)})")
                continue

            logger.info(f"  Sample size: {len(coord_pairs)} pairs")

            clean_data = coord_pairs[['shotgun_mentor', 'shotgun_protege']].dropna()

            if len(clean_data) >= 10:
                corr, p_val = stats.pearsonr(clean_data['shotgun_mentor'], clean_data['shotgun_protege'])

                results[coord_type] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(clean_data)),
                    'significant': bool(p_val < 0.05)
                }

                sig_marker = "SIG" if p_val < 0.05 else "n.s."
                logger.info(f"  Shotgun: r={corr:.3f}, p={p_val:.4f} ({sig_marker}), n={len(clean_data)}")

        return results

    def create_visualization(self, pairs_df, overall_results, mentor_type_results, coord_type_results):
        """Create scatter plots and bar charts"""
        logger.info("\nCreating visualizations...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        try:
            from plot_config import configure_plots
            configure_plots()
        except ImportError:
            pass

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        colors = {'Offensive': '#FF6B35', 'Defensive': '#004E89'}

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        # Panel 1: Overall scatter
        ax1 = axes[0]
        clean_data = pairs_df[['shotgun_mentor', 'shotgun_protege', 'mentor_background']].dropna()

        for bg_type in ['Offensive', 'Defensive']:
            bg_data = clean_data[clean_data['mentor_background'] == bg_type]
            if len(bg_data) > 0:
                ax1.scatter(bg_data['shotgun_mentor'], bg_data['shotgun_protege'],
                           c=colors.get(bg_type, 'gray'), alpha=0.5, s=60,
                           edgecolors='black', linewidth=0.5, label=bg_type)

        # Overall regression line
        if overall_results:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_data['shotgun_mentor'], clean_data['shotgun_protege']
            )
            x_range = np.array([clean_data['shotgun_mentor'].min(), clean_data['shotgun_mentor'].max()])
            ax1.plot(x_range, slope * x_range + intercept, color='black',
                    linestyle='--', linewidth=2, alpha=0.7, label='Overall fit')

        ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Mentor Shotgun Gene', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Protege Shotgun Gene', fontsize=13, fontweight='bold')
        ax1.set_title('Shotgun Gene Inheritance', fontsize=14, fontweight='bold', pad=10)
        ax1.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
        ax1.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

        if overall_results:
            stats_text = f"Overall: r={overall_results['correlation']:.3f}\n"
            stats_text += f"p={overall_results['p_value']:.4f}, n={overall_results['n']}"
            ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, fontsize=11,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

        ax1.grid(True, alpha=0.3, linestyle=':')
        ax1.legend(loc='lower right', framealpha=0.95)

        # Panel 2: By mentor background bar chart
        ax2 = axes[1]
        backgrounds = ['Offensive', 'Defensive']
        correlations = []
        for bg in backgrounds:
            if bg in mentor_type_results and mentor_type_results[bg]:
                correlations.append(mentor_type_results[bg].get('correlation', 0))
            else:
                correlations.append(0)

        bars = ax2.bar(backgrounds, correlations, color=[colors['Offensive'], colors['Defensive']],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Mark significant bars
        for i, bg in enumerate(backgrounds):
            if bg in mentor_type_results and mentor_type_results[bg]:
                if mentor_type_results[bg].get('significant', False):
                    ax2.text(i, correlations[i], '*', ha='center', va='bottom',
                            fontsize=19, fontweight='bold')

        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax2.set_xlabel('Mentor Background', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Mentor-Protege Correlation (r)', fontsize=13, fontweight='bold')
        ax2.set_title('By Mentor Background', fontsize=14, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax2.set_ylim(-0.1, 0.4)

        # Panel 3: By coordinator type bar chart
        ax3 = axes[2]
        coord_types = ['OC', 'DC']
        coord_correlations = []
        for ct in coord_types:
            if ct in coord_type_results and coord_type_results[ct]:
                coord_correlations.append(coord_type_results[ct].get('correlation', 0))
            else:
                coord_correlations.append(0)

        bars = ax3.bar(coord_types, coord_correlations, color=[colors['Offensive'], colors['Defensive']],
                      alpha=0.8, edgecolor='black', linewidth=1)

        # Mark significant bars
        for i, ct in enumerate(coord_types):
            if ct in coord_type_results and coord_type_results[ct]:
                if coord_type_results[ct].get('significant', False):
                    ax3.text(i, coord_correlations[i], '*', ha='center', va='bottom',
                            fontsize=19, fontweight='bold')

        ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax3.set_xlabel('Coordinator Type', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Mentor-Protege Correlation (r)', fontsize=13, fontweight='bold')
        ax3.set_title('By Coordinator Type (OC/DC -> HC)', fontsize=14, fontweight='bold', pad=10)
        ax3.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax3.set_ylim(-0.1, 0.4)

        plt.tight_layout()

        # Save
        output_dir = Path("outputs/visualizations/inheritance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "shotgun_inheritance_by_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def save_results(self, pairs_df, overall_results, mentor_type_results, coord_type_results):
        """Save results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pairs
        pairs_file = output_dir / "shotgun_mentor_protege_pairs.csv"
        pairs_df.to_csv(pairs_file, index=False)
        logger.info(f"Saved pairs data: {pairs_file}")

        # Save results
        results = {
            'overall': overall_results,
            'by_mentor_background': mentor_type_results,
            'by_coordinator_type': coord_type_results
        }

        results_file = output_dir / "shotgun_inheritance_by_type_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_file}")

    def print_summary(self, overall_results, mentor_type_results, coord_type_results):
        """Print summary"""
        print("\n" + "="*80)
        print("SHOTGUN GENE INHERITANCE BY COACH TYPE")
        print("="*80)

        print("\nOVERALL:")
        print("-" * 80)
        if overall_results:
            sig = "SIG" if overall_results['significant'] else "n.s."
            print(f"  r={overall_results['correlation']:.3f}, p={overall_results['p_value']:.4f} "
                  f"({sig}), n={overall_results['n']}")

        print("\nBY MENTOR BACKGROUND:")
        print("-" * 80)

        for bg_type in ['Offensive', 'Defensive']:
            if bg_type not in mentor_type_results or not mentor_type_results[bg_type]:
                continue

            result = mentor_type_results[bg_type]
            sig = "SIG" if result['significant'] else "n.s."
            print(f"\n{bg_type} Mentors:")
            print(f"  r={result['correlation']:.3f}, p={result['p_value']:.4f} "
                  f"({sig}), n={result['n']}")

        print("\n" + "-" * 80)
        print("BY COORDINATOR TYPE:")
        print("-" * 80)

        for coord_type in ['OC', 'DC']:
            if coord_type not in coord_type_results or not coord_type_results[coord_type]:
                continue

            result = coord_type_results[coord_type]
            sig = "SIG" if result['significant'] else "n.s."
            print(f"\n{coord_type} -> HC:")
            print(f"  r={result['correlation']:.3f}, p={result['p_value']:.4f} "
                  f"({sig}), n={result['n']}")

        print("\n" + "="*80)


def main():
    analyzer = ShotgunInheritanceAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Create pairs
        pairs_df = analyzer.create_mentor_protege_pairs()

        # Analyze overall
        overall_results = analyzer.analyze_overall(pairs_df)

        # Analyze by mentor type
        mentor_type_results = analyzer.analyze_by_mentor_type(pairs_df)

        # Analyze by coordinator type
        coord_type_results = analyzer.analyze_by_coordinator_type(pairs_df)

        # Create visualizations
        analyzer.create_visualization(pairs_df, overall_results, mentor_type_results, coord_type_results)

        # Save results
        analyzer.save_results(pairs_df, overall_results, mentor_type_results, coord_type_results)

        # Print summary
        analyzer.print_summary(overall_results, mentor_type_results, coord_type_results)

        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
