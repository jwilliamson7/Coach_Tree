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
import sys
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import (
    cluster_bootstrap_corr,
    corr_with_small_cluster_guard,
    within_group_demean,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _map_coord_role(role):
    """Map a relationships.csv child_role string to OC / DC / STC / Other."""
    r = str(role)
    if 'Offensive' in r:
        return 'OC'
    if 'Defensive' in r:
        return 'DC'
    if 'Special' in r:
        return 'STC'
    return 'Other'


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

        # Era-adjusted (contemporary-group) shotgun gene: within-season demean at the
        # coach-year level before averaging over a coach's seasons. Shotgun has the
        # largest league-wide temporal drift of any gene (~49% between-season
        # variance), so this control matters most here. The published gene CSV stays
        # absolute; the control lives only in the inference.
        self.shotgun_data['shotgun_gene_eradj'] = within_group_demean(
            self.shotgun_data, 'shotgun_gene', 'season')

        # Calculate average shotgun gene for each coach (across their career)
        coach_avg = self.shotgun_data.groupby('head_coach').agg({
            'shotgun_gene': 'mean',
            'shotgun_gene_eradj': 'mean',
            'shotgun_gene_zscore': 'mean',
            'Background': 'first'
        }).reset_index()
        coach_avg.columns = ['coach', 'shotgun_gene', 'shotgun_gene_eradj',
                             'shotgun_gene_zscore', 'Background']

        self.coach_types = coach_avg
        logger.info(f"Calculated average shotgun gene (raw + era-adjusted) for {len(self.coach_types)} coaches")

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

        # Deduplicate to unique (mentor, protege) pairs. relationships.csv stores one
        # row per overlapping year/team/role, but the inherited shotgun gene is a
        # career average (one fixed value per coach), so multiple rows for the same
        # pair are identical points that pseudo-replicate the data and inflate n.
        coord_to_hc = coord_to_hc.drop_duplicates(subset=['parent_name', 'child_name'])

        logger.info(f"Found {len(coord_to_hc)} unique coordinator -> head coach pairs")

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

            # Protege's role UNDER THIS MENTOR (child_role on the relationship row)
            # is the apprenticeship channel we split on -- not the mentor's own
            # coordinator background. Mentor coordinator type kept for reference only.
            protege_role = _map_coord_role(rel.get('child_role'))
            mentor_coord_type = self.coach_coordinator_types.get(mentor_name, 'Other')

            pair = {
                'mentor_name': mentor_name,
                'protege_name': protege_name,
                'mentor_background': mentor['Background'],
                'protege_background': protege['Background'],
                'protege_role': protege_role,
                'mentor_coordinator_type': mentor_coord_type,
                'shotgun_mentor': mentor['shotgun_gene'],
                'shotgun_protege': protege['shotgun_gene'],
                'shotgun_mentor_eradj': mentor['shotgun_gene_eradj'],
                'shotgun_protege_eradj': protege['shotgun_gene_eradj'],
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

        logger.info("\nProtege role (under mentor) distribution:")
        for role, count in pairs_df['protege_role'].value_counts().items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {role}: {count} ({pct:.1f}%)")

        return pairs_df

    def _shotgun_stats(self, data):
        """Shotgun mentor-protege correlation on a subset, era-adjusted + raw.

        Era-adjusted (contemporary-group controlled) values occupy the canonical
        (unsuffixed) keys so the BH harvester and the paper use the era-clean
        inference; raw values are kept under *_raw. Returns {} if < 10 pairs.
        """
        out = {}
        specs = [
            ('', 'shotgun_mentor_eradj', 'shotgun_protege_eradj'),
            ('_raw', 'shotgun_mentor', 'shotgun_protege'),
        ]
        for tag, mcol, pcol in specs:
            if mcol not in data.columns or pcol not in data.columns:
                continue
            d = data[[mcol, pcol, 'mentor_name']].dropna()
            if len(d) < 10:
                continue
            corr, p_val = stats.pearsonr(d[mcol], d[pcol])
            boot = corr_with_small_cluster_guard(
                d[mcol].values, d[pcol].values, d['mentor_name'].values,
                min_clusters=40, n_boot=2000, seed=42,
            )
            out[f'correlation{tag}'] = float(corr)
            out[f'p_value{tag}'] = float(p_val)
            out[f'ci_low{tag}'] = boot['ci_low']
            out[f'ci_high{tag}'] = boot['ci_high']
            out[f'p_bootstrap_mentor_clustered{tag}'] = boot['p_bootstrap']
            out[f'p_wild_cluster{tag}'] = boot.get('p_wild_cluster')
            out[f'n_mentors{tag}'] = boot['n_clusters']
            out[f'small_cluster{tag}'] = boot.get('small_cluster', False)
            out[f'n{tag}'] = int(len(d))
        if 'correlation' in out:
            out['significant'] = bool(out.get('p_value', 1.0) < 0.05)
        return out

    def _by(self, pairs_df, group_col, levels, header):
        """Run _shotgun_stats for each level of group_col."""
        logger.info(f"\n\n{header}")
        logger.info("=" * 80)
        results = {}
        for level in levels:
            sub = pairs_df[pairs_df[group_col] == level]
            s = self._shotgun_stats(sub)
            results[level] = s if s else {}
            if s:
                logger.info(f"  {level} (n={s.get('n', 0)}): era-adj "
                            f"r={s.get('correlation', float('nan')):.3f} "
                            f"(raw {s.get('correlation_raw', float('nan')):.3f})")
        return results

    def analyze_overall(self, pairs_df):
        """Analyze overall shotgun gene inheritance (era-adjusted primary)."""
        logger.info("\n\nANALYZING OVERALL SHOTGUN INHERITANCE:")
        logger.info("=" * 80)
        s = self._shotgun_stats(pairs_df)
        if s:
            logger.info(f"Overall (n={s.get('n', 0)}): era-adj "
                        f"r={s.get('correlation', float('nan')):.3f} "
                        f"(raw {s.get('correlation_raw', float('nan')):.3f})")
            return s
        logger.info("Insufficient data")
        return None

    def analyze_by_mentor_type(self, pairs_df):
        """Inheritance by mentor's background (Offensive/Defensive)."""
        return self._by(pairs_df, 'mentor_background', ['Offensive', 'Defensive'],
                        "ANALYZING SHOTGUN INHERITANCE BY MENTOR BACKGROUND:")

    def analyze_by_protege_role(self, pairs_df):
        """Inheritance by the protege's role under the mentor (OC/DC).

        Replaces the prior mentor-keyed 'by coordinator type' split.
        """
        return self._by(pairs_df, 'protege_role', ['OC', 'DC'],
                        "ANALYZING SHOTGUN INHERITANCE BY PROTEGE ROLE:")

    def analyze_two_by_two(self, pairs_df):
        """Full mentor-background x protege-role 2x2 (the apprenticeship cells)."""
        logger.info("\n\nANALYZING SHOTGUN 2x2 (mentor background x protege role):")
        logger.info("=" * 80)
        results = {}
        for bg in ['Offensive', 'Defensive']:
            for role in ['OC', 'DC']:
                cell = f'{bg}|{role}'
                sub = pairs_df[(pairs_df['mentor_background'] == bg)
                               & (pairs_df['protege_role'] == role)]
                s = self._shotgun_stats(sub)
                results[cell] = s if s else {}
                if s:
                    logger.info(f"  {cell} (n={s.get('n', 0)}): era-adj "
                                f"r={s.get('correlation', float('nan')):.3f} "
                                f"(raw {s.get('correlation_raw', float('nan')):.3f})")
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

        # Panel 1: Overall scatter (era-adjusted gene, matching the reported r)
        ax1 = axes[0]
        clean_data = pairs_df[['shotgun_mentor_eradj', 'shotgun_protege_eradj',
                               'mentor_background']].dropna()

        for bg_type in ['Offensive', 'Defensive']:
            bg_data = clean_data[clean_data['mentor_background'] == bg_type]
            if len(bg_data) > 0:
                ax1.scatter(bg_data['shotgun_mentor_eradj'], bg_data['shotgun_protege_eradj'],
                           c=colors.get(bg_type, 'gray'), alpha=0.5, s=60,
                           edgecolors='black', linewidth=0.5, label=bg_type)

        # Overall regression line
        if overall_results:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_data['shotgun_mentor_eradj'], clean_data['shotgun_protege_eradj']
            )
            x_range = np.array([clean_data['shotgun_mentor_eradj'].min(),
                                clean_data['shotgun_mentor_eradj'].max()])
            ax1.plot(x_range, slope * x_range + intercept, color='black',
                    linestyle='--', linewidth=2, alpha=0.7, label='Overall fit')

        ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Mentor Shotgun Gene (era-adjusted)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Protege Shotgun Gene (era-adjusted)', fontsize=13, fontweight='bold')
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

        # Panel 3: By protege role bar chart (era-adjusted)
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
        ax3.set_xlabel('Protege Role (under mentor)', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Mentor-Protege Correlation (r)', fontsize=13, fontweight='bold')
        ax3.set_title('By Protege Role (era-adjusted)', fontsize=14, fontweight='bold', pad=10)
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

    def save_results(self, pairs_df, overall_results, mentor_type_results,
                     protege_role_results, two_by_two_results):
        """Save results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pairs
        pairs_file = output_dir / "shotgun_mentor_protege_pairs.csv"
        pairs_df.to_csv(pairs_file, index=False)
        logger.info(f"Saved pairs data: {pairs_file}")

        # Save results. Canonical correlation/p keys are era-adjusted; *_raw keys hold
        # the raw estimate. The 2x2 is the apprenticeship decomposition; the marginals
        # feed the FDR family (2x2 cells reported descriptively to avoid double-count).
        results = {
            'overall': overall_results,
            'by_mentor_background': mentor_type_results,
            'by_protege_role': protege_role_results,
            'two_by_two': two_by_two_results,
        }

        results_file = output_dir / "shotgun_inheritance_by_type_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_file}")

    def print_summary(self, overall_results, mentor_type_results, protege_role_results,
                      two_by_two_results):
        """Print summary (era-adjusted r primary; raw in parentheses)."""
        print("\n" + "=" * 80)
        print("SHOTGUN GENE INHERITANCE (era-adjusted; raw in parens)")
        print("=" * 80)

        def _line(d):
            return (f"r={d.get('correlation', float('nan')):.3f} "
                    f"(raw {d.get('correlation_raw', float('nan')):.3f}), "
                    f"p={d.get('p_value', float('nan')):.4f}, n={d.get('n', 0)}")

        print("\nOVERALL:")
        print("-" * 80)
        if overall_results:
            print(f"  {_line(overall_results)}")

        print("\nBY MENTOR BACKGROUND:")
        print("-" * 80)
        for bg_type in ['Offensive', 'Defensive']:
            if mentor_type_results.get(bg_type):
                print(f"\n{bg_type} Mentors:")
                print(f"  {_line(mentor_type_results[bg_type])}")

        print("\n" + "-" * 80)
        print("BY PROTEGE ROLE (under mentor):")
        print("-" * 80)
        for role in ['OC', 'DC']:
            if protege_role_results.get(role):
                print(f"\nProtege {role}:")
                print(f"  {_line(protege_role_results[role])}")

        print("\n" + "-" * 80)
        print("2x2 (mentor background x protege role):")
        print("-" * 80)
        for cell in ['Offensive|OC', 'Offensive|DC', 'Defensive|OC', 'Defensive|DC']:
            if two_by_two_results.get(cell):
                print(f"  {cell}: {_line(two_by_two_results[cell])}")

        print("\n" + "=" * 80)


def main():
    analyzer = ShotgunInheritanceAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Create pairs
        pairs_df = analyzer.create_mentor_protege_pairs()

        # Analyze overall
        overall_results = analyzer.analyze_overall(pairs_df)

        # Analyze by mentor background and by protege role (the two marginals)
        mentor_type_results = analyzer.analyze_by_mentor_type(pairs_df)
        protege_role_results = analyzer.analyze_by_protege_role(pairs_df)

        # Full mentor-background x protege-role 2x2 (apprenticeship decomposition)
        two_by_two_results = analyzer.analyze_two_by_two(pairs_df)

        # Create visualizations
        analyzer.create_visualization(pairs_df, overall_results, mentor_type_results,
                                      protege_role_results)

        # Save results
        analyzer.save_results(pairs_df, overall_results, mentor_type_results,
                              protege_role_results, two_by_two_results)

        # Print summary
        analyzer.print_summary(overall_results, mentor_type_results,
                               protege_role_results, two_by_two_results)

        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
