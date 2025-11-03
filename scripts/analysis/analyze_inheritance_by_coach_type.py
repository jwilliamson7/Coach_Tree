#!/usr/bin/env python3
"""
Analyze Aggression Inheritance by Coach Background Type

This script examines whether inheritance of aggression differs between
offensive and defensive coaches. Do offensive coaches pass on their
philosophies more than defensive coaches?

Usage:
    python analyze_inheritance_by_coach_type.py
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


class InheritanceByTypeAnalyzer:
    """Analyze aggression inheritance by coach background type"""

    def __init__(self):
        self.aggression_data = None
        self.coach_types = None
        self.relationships = None
        self.coach_coordinator_types = None

    def load_data(self):
        """Load all necessary data"""
        logger.info("Loading data...")

        # Load aggression data with coach types
        type_file = Path("outputs/analysis/aggression_war_with_coach_type.csv")
        if not type_file.exists():
            raise FileNotFoundError(
                f"Coach type data not found: {type_file}\n"
                "Please run analyze_aggression_by_coach_type.py first"
            )

        self.aggression_data = pd.read_csv(type_file)
        logger.info(f"Loaded {len(self.aggression_data)} coach-year observations")

        # Calculate average aggression for each coach (across their career)
        aggression_vars = [
            'fourth_down_aggression',
            'pass_heavy_aggression',
            'deep_pass_aggression',
            'two_point_aggression',
            'composite_aggression'
        ]

        coach_avg = self.aggression_data.groupby('coach')[aggression_vars + ['Background']].agg({
            **{var: 'mean' for var in aggression_vars},
            'Background': 'first'
        }).reset_index()

        self.coach_types = coach_avg
        logger.info(f"Calculated average aggression for {len(self.coach_types)} coaches")

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
        """Create mentor-protégé pairs with aggression data for both"""
        logger.info("\nCreating mentor-protégé pairs...")

        # Filter to coordinator → head coach relationships
        coord_to_hc = self.relationships[
            self.relationships['relationship_type'] == 'coordinator_to_hc'
        ].copy()

        logger.info(f"Found {len(coord_to_hc)} coordinator → head coach transitions")

        pairs = []

        for _, rel in coord_to_hc.iterrows():
            mentor_name = rel['parent_name']
            protege_name = rel['child_name']

            # Get mentor aggression
            mentor_data = self.coach_types[self.coach_types['coach'] == mentor_name]
            if len(mentor_data) == 0:
                continue

            # Get protégé aggression
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
                'fourth_down_mentor': mentor['fourth_down_aggression'],
                'fourth_down_protege': protege['fourth_down_aggression'],
                'pass_heavy_mentor': mentor['pass_heavy_aggression'],
                'pass_heavy_protege': protege['pass_heavy_aggression'],
                'deep_pass_mentor': mentor['deep_pass_aggression'],
                'deep_pass_protege': protege['deep_pass_aggression'],
                'two_point_mentor': mentor['two_point_aggression'],
                'two_point_protege': protege['two_point_aggression'],
                'composite_mentor': mentor['composite_aggression'],
                'composite_protege': protege['composite_aggression']
            }

            pairs.append(pair)

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"Created {len(pairs_df)} complete mentor-protégé pairs")

        # Show distribution by mentor type
        logger.info("\nMentor background distribution:")
        for bg, count in pairs_df['mentor_background'].value_counts().items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {bg}: {count} ({pct:.1f}%)")

        logger.info("\nMentor coordinator type distribution:")
        for coord_type, count in pairs_df['mentor_coordinator_type'].value_counts().items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {coord_type}: {count} ({pct:.1f}%)")

        return pairs_df

    def analyze_by_mentor_type(self, pairs_df):
        """Analyze inheritance by mentor's background type"""
        logger.info("\n\nANALYZING INHERITANCE BY MENTOR TYPE:")
        logger.info("="*80)

        aggression_types = [
            ('fourth_down', '4th Down'),
            ('pass_heavy', 'Pass-Heavy'),
            ('deep_pass', 'Deep Pass'),
            ('two_point', 'Two-Point'),
            ('composite', 'Composite')
        ]

        results = {}

        for bg_type in ['Offensive', 'Defensive']:
            logger.info(f"\n{bg_type} Mentors:")
            results[bg_type] = {}

            bg_pairs = pairs_df[pairs_df['mentor_background'] == bg_type]

            if len(bg_pairs) < 10:
                logger.info(f"  Insufficient data (n={len(bg_pairs)})")
                continue

            for var, label in aggression_types:
                mentor_col = f'{var}_mentor'
                protege_col = f'{var}_protege'

                clean_data = bg_pairs[[mentor_col, protege_col]].dropna()

                if len(clean_data) >= 10:
                    corr, p_val = stats.pearsonr(clean_data[mentor_col], clean_data[protege_col])

                    results[bg_type][var] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'n': int(len(clean_data)),
                        'significant': bool(p_val < 0.05)
                    }

                    sig_marker = "SIG" if p_val < 0.05 else "n.s."
                    logger.info(f"  {label}: r={corr:.3f}, p={p_val:.4f} ({sig_marker}), n={len(clean_data)}")

        return results

    def analyze_by_coordinator_type(self, pairs_df):
        """Analyze separately for OC vs DC mentors"""
        logger.info("\n\nANALYZING BY COORDINATOR TYPE:")
        logger.info("="*80)

        aggression_types = [
            ('fourth_down', '4th Down'),
            ('pass_heavy', 'Pass-Heavy'),
            ('deep_pass', 'Deep Pass'),
            ('two_point', 'Two-Point'),
            ('composite', 'Composite')
        ]

        results = {}

        for coord_type in ['OC', 'DC']:
            logger.info(f"\n{coord_type} mentors:")
            results[coord_type] = {}

            coord_pairs = pairs_df[pairs_df['mentor_coordinator_type'] == coord_type]

            if len(coord_pairs) < 10:
                logger.info(f"  Insufficient data (n={len(coord_pairs)})")
                continue

            logger.info(f"  Sample size: {len(coord_pairs)} pairs")

            for var, label in aggression_types:
                mentor_col = f'{var}_mentor'
                protege_col = f'{var}_protege'

                clean_data = coord_pairs[[mentor_col, protege_col]].dropna()

                if len(clean_data) >= 10:
                    corr, p_val = stats.pearsonr(clean_data[mentor_col], clean_data[protege_col])

                    results[coord_type][var] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'n': int(len(clean_data)),
                        'significant': bool(p_val < 0.05)
                    }

                    sig_marker = "SIG" if p_val < 0.05 else "n.s."
                    logger.info(f"  {label}: r={corr:.3f}, p={p_val:.4f} ({sig_marker}), n={len(clean_data)}")

        return results

    def create_visualization_by_type(self, pairs_df, mentor_type_results):
        """Create scatter plots by mentor type"""
        logger.info("\nCreating visualization by mentor type...")

        plt.rcParams['font.family'] = 'Cambria'

        aggression_types = [
            ('fourth_down', '4th Down'),
            ('pass_heavy', 'Pass-Heavy'),
            ('deep_pass', 'Deep Pass'),
            ('two_point', 'Two-Point'),
            ('composite', 'Composite')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        colors = {'Offensive': '#FF6B35', 'Defensive': '#004E89'}

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        for idx, (var, label) in enumerate(aggression_types):
            ax = axes[idx]

            mentor_col = f'{var}_mentor'
            protege_col = f'{var}_protege'

            for bg_type in ['Offensive', 'Defensive']:
                bg_pairs = pairs_df[pairs_df['mentor_background'] == bg_type]
                clean_data = bg_pairs[[mentor_col, protege_col]].dropna()

                if len(clean_data) < 10:
                    continue

                x = clean_data[mentor_col]
                y = clean_data[protege_col]

                # Scatter plot
                ax.scatter(x, y, c=colors[bg_type], alpha=0.5, s=60,
                          edgecolors='black', linewidth=0.5, label=bg_type)

                # Regression line
                corr, p_val = stats.pearsonr(x, y)
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                x_range = np.array([x.min(), x.max()])
                ax.plot(x_range, slope * x_range + intercept,
                       color=colors[bg_type], linestyle='--', linewidth=2, alpha=0.7)

            # Reference lines
            ax.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

            # Labels
            ax.set_xlabel('Mentor Aggression POE', fontsize=10, fontweight='bold')
            ax.set_ylabel('Protégé Aggression POE', fontsize=10, fontweight='bold')
            ax.set_title(f'{label} Aggression', fontsize=11, fontweight='bold', pad=10)

            # Format axes
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))

            # Add statistics
            stats_text = []
            for bg_type in ['Offensive', 'Defensive']:
                if bg_type in mentor_type_results and var in mentor_type_results[bg_type]:
                    result = mentor_type_results[bg_type][var]
                    sig = "SIG" if result['significant'] else "n.s."
                    stats_text.append(f"{bg_type}: r={result['correlation']:.3f} ({sig})")

            if stats_text:
                ax.text(0.05, 0.95, '\n'.join(stats_text),
                       transform=ax.transAxes,
                       fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='gray'))

            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='lower right', framealpha=0.95)

        # Hide extra subplot
        axes[5].axis('off')

        fig.suptitle('Aggression Inheritance by Mentor Background Type\n(Coordinator → Head Coach)',
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout(rect=[0, 0, 1, 0.985])

        # Save
        output_dir = Path("outputs/visualizations/inheritance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "inheritance_by_mentor_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def create_comparison_bar_chart(self, mentor_type_results, coord_type_results):
        """Create bar chart comparing inheritance strength"""
        logger.info("Creating comparison bar chart...")

        plt.rcParams['font.family'] = 'Cambria'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left panel: By mentor background
        aggression_types = ['fourth_down', 'pass_heavy', 'deep_pass', 'two_point', 'composite']
        labels = ['4th Down', 'Pass-Heavy', 'Deep Pass', 'Two-Point', 'Composite']

        x = np.arange(len(labels))
        width = 0.35

        off_corrs = []
        def_corrs = []

        for var in aggression_types:
            off_val = mentor_type_results.get('Offensive', {}).get(var, {}).get('correlation', 0)
            def_val = mentor_type_results.get('Defensive', {}).get(var, {}).get('correlation', 0)
            off_corrs.append(off_val)
            def_corrs.append(def_val)

        bars1 = ax1.bar(x - width/2, off_corrs, width, label='Offensive Mentors',
                       color='#FF6B35', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, def_corrs, width, label='Defensive Mentors',
                       color='#004E89', alpha=0.8, edgecolor='black', linewidth=1)

        # Mark significant bars
        for i, var in enumerate(aggression_types):
            if 'Offensive' in mentor_type_results and var in mentor_type_results['Offensive']:
                if mentor_type_results['Offensive'][var]['significant']:
                    ax1.text(i - width/2, off_corrs[i], '*', ha='center', va='bottom',
                            fontsize=16, fontweight='bold')
            if 'Defensive' in mentor_type_results and var in mentor_type_results['Defensive']:
                if mentor_type_results['Defensive'][var]['significant']:
                    ax1.text(i + width/2, def_corrs[i], '*', ha='center', va='bottom',
                            fontsize=16, fontweight='bold')

        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax1.set_xlabel('Aggression Type', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Mentor-Protégé Correlation (r)', fontsize=11, fontweight='bold')
        ax1.set_title('By Mentor Background Type', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend(loc='best', framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax1.set_ylim(-0.05, 0.35)

        # Right panel: By coordinator type
        oc_corrs = []
        dc_corrs = []

        for var in aggression_types:
            oc_val = coord_type_results.get('OC', {}).get(var, {}).get('correlation', 0)
            dc_val = coord_type_results.get('DC', {}).get(var, {}).get('correlation', 0)
            oc_corrs.append(oc_val)
            dc_corrs.append(dc_val)

        bars3 = ax2.bar(x - width/2, oc_corrs, width, label='OC → HC',
                       color='#FF6B35', alpha=0.8, edgecolor='black', linewidth=1)
        bars4 = ax2.bar(x + width/2, dc_corrs, width, label='DC → HC',
                       color='#004E89', alpha=0.8, edgecolor='black', linewidth=1)

        # Mark significant bars
        for i, var in enumerate(aggression_types):
            if 'OC' in coord_type_results and var in coord_type_results['OC']:
                if coord_type_results['OC'][var]['significant']:
                    ax2.text(i - width/2, oc_corrs[i], '*', ha='center', va='bottom',
                            fontsize=16, fontweight='bold')
            if 'DC' in coord_type_results and var in coord_type_results['DC']:
                if coord_type_results['DC'][var]['significant']:
                    ax2.text(i + width/2, dc_corrs[i], '*', ha='center', va='bottom',
                            fontsize=16, fontweight='bold')

        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax2.set_xlabel('Aggression Type', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Mentor-Protégé Correlation (r)', fontsize=11, fontweight='bold')
        ax2.set_title('By Coordinator Type', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend(loc='best', framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax2.set_ylim(-0.05, 0.35)

        fig.suptitle('Aggression Inheritance Strength by Coach Type\n(Stars indicate p<0.05)',
                    fontsize=14, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        output_dir = Path("outputs/visualizations/inheritance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "inheritance_comparison_by_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved bar chart: {png_path}")

        plt.close()

    def save_results(self, pairs_df, mentor_type_results, coord_type_results):
        """Save results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pairs
        pairs_file = output_dir / "mentor_protege_pairs_with_types.csv"
        pairs_df.to_csv(pairs_file, index=False)
        logger.info(f"Saved pairs data: {pairs_file}")

        # Save results
        results = {
            'by_mentor_background': mentor_type_results,
            'by_coordinator_type': coord_type_results
        }

        results_file = output_dir / "inheritance_by_type_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_file}")

    def print_summary(self, mentor_type_results, coord_type_results):
        """Print summary"""
        print("\n" + "="*80)
        print("AGGRESSION INHERITANCE BY COACH TYPE")
        print("="*80)

        print("\nBY MENTOR BACKGROUND:")
        print("-" * 80)

        for bg_type in ['Offensive', 'Defensive']:
            if bg_type not in mentor_type_results:
                continue

            print(f"\n{bg_type} Mentors:")
            for var in ['fourth_down', 'pass_heavy', 'composite']:
                if var in mentor_type_results[bg_type]:
                    result = mentor_type_results[bg_type][var]
                    sig = "SIG" if result['significant'] else "n.s."
                    print(f"  {var}: r={result['correlation']:.3f}, p={result['p_value']:.4f} "
                          f"({sig}), n={result['n']}")

        print("\n" + "-" * 80)
        print("BY COORDINATOR TYPE:")
        print("-" * 80)

        for coord_type in ['OC', 'DC']:
            if coord_type not in coord_type_results:
                continue

            print(f"\n{coord_type} -> HC:")
            for var in ['fourth_down', 'pass_heavy', 'composite']:
                if var in coord_type_results[coord_type]:
                    result = coord_type_results[coord_type][var]
                    sig = "SIG" if result['significant'] else "n.s."
                    print(f"  {var}: r={result['correlation']:.3f}, p={result['p_value']:.4f} "
                          f"({sig}), n={result['n']}")

        print("\n" + "="*80)


def main():
    analyzer = InheritanceByTypeAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Create pairs
        pairs_df = analyzer.create_mentor_protege_pairs()

        # Analyze by mentor type
        mentor_type_results = analyzer.analyze_by_mentor_type(pairs_df)

        # Analyze by coordinator type
        coord_type_results = analyzer.analyze_by_coordinator_type(pairs_df)

        # Create visualizations
        analyzer.create_visualization_by_type(pairs_df, mentor_type_results)
        analyzer.create_comparison_bar_chart(mentor_type_results, coord_type_results)

        # Save results
        analyzer.save_results(pairs_df, mentor_type_results, coord_type_results)

        # Print summary
        analyzer.print_summary(mentor_type_results, coord_type_results)

        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
