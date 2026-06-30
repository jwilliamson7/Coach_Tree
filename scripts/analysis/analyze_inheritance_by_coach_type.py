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

# Aggression components (column stem, display label).
AGG_TYPES = [
    ('fourth_down', '4th Down'),
    ('pass_heavy', 'Pass-Heavy'),
    ('deep_pass', 'Deep Pass'),
    ('two_point', 'Two-Point'),
    ('composite', 'Composite'),
]


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

        # Era-adjusted (contemporary-group) career means: within-season demean each
        # component at the coach-year level before averaging over a coach's seasons.
        # This removes the league-wide temporal drift that would otherwise act as a
        # shared-era confounder in the mentor-protege correlation. The published gene
        # CSVs stay absolute; the control lives only here in the inference.
        adj = self.aggression_data.copy()
        for var in aggression_vars:
            adj[f'{var}_eradj'] = within_group_demean(adj, var, 'year')
        coach_avg_e = adj.groupby('coach')[[f'{var}_eradj' for var in aggression_vars]].mean().reset_index()
        coach_avg = coach_avg.merge(coach_avg_e, on='coach', how='left')

        self.coach_types = coach_avg
        logger.info(f"Calculated average aggression (raw + era-adjusted) for {len(self.coach_types)} coaches")

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

        # Deduplicate to unique (mentor, protege) pairs. relationships.csv stores one
        # row per overlapping year/team/role, but the inherited genes are career
        # averages (one fixed value per coach), so multiple rows for the same pair are
        # identical points. Keeping them pseudo-replicates the data (e.g. Belichick ->
        # McDaniels appeared 13 times), inflating n ~2.7x and over-weighting long
        # overlaps. The mentor-WAR inheritance analysis already dedupes this way.
        coord_to_hc = coord_to_hc.drop_duplicates(subset=['parent_name', 'child_name'])

        logger.info(f"Found {len(coord_to_hc)} unique coordinator -> head coach pairs")

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

            # Protege's role UNDER THIS MENTOR (child_role on the relationship row)
            # is the apprenticeship channel we split on -- not the mentor's own
            # coordinator background, which was the prior (incorrect) key. Mentor
            # coordinator type is kept for reference only.
            protege_role = _map_coord_role(rel.get('child_role'))
            mentor_coord_type = self.coach_coordinator_types.get(mentor_name, 'Other')

            pair = {
                'mentor_name': mentor_name,
                'protege_name': protege_name,
                'mentor_background': mentor['Background'],
                'protege_background': protege['Background'],
                'protege_role': protege_role,
                'mentor_coordinator_type': mentor_coord_type,
            }
            for stem, _label in AGG_TYPES:
                src = f'{stem}_aggression'
                pair[f'{stem}_mentor'] = mentor[src]
                pair[f'{stem}_protege'] = protege[src]
                pair[f'{stem}_mentor_eradj'] = mentor[f'{src}_eradj']
                pair[f'{stem}_protege_eradj'] = protege[f'{src}_eradj']

            pairs.append(pair)

        pairs_df = pd.DataFrame(pairs)
        logger.info(f"Created {len(pairs_df)} complete mentor-protégé pairs")

        # Show distribution by mentor type
        logger.info("\nMentor background distribution:")
        for bg, count in pairs_df['mentor_background'].value_counts().items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {bg}: {count} ({pct:.1f}%)")

        logger.info("\nProtege role (under mentor) distribution:")
        for role, count in pairs_df['protege_role'].value_counts().items():
            pct = (count / len(pairs_df)) * 100
            logger.info(f"  {role}: {count} ({pct:.1f}%)")

        return pairs_df

    def _pair_stats(self, data, var):
        """Mentor-protege correlation stats for one component on a subset.

        Computes both the era-adjusted (contemporary-group controlled) estimate
        and the raw estimate. The era-adjusted values occupy the canonical
        (unsuffixed) keys so the BH harvester and the paper use the era-clean
        inference; raw values are kept under *_raw for the era-inflation
        comparison. Returns {} if the subset has < 10 usable pairs.
        """
        out = {}
        specs = [
            ('', f'{var}_mentor_eradj', f'{var}_protege_eradj'),
            ('_raw', f'{var}_mentor', f'{var}_protege'),
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

    def _analyze_subsets(self, pairs_df, group_col, levels, header):
        """Run _pair_stats for every component within each level of group_col."""
        logger.info(f"\n\n{header}")
        logger.info("=" * 80)
        results = {}
        for level in levels:
            sub = pairs_df[pairs_df[group_col] == level]
            results[level] = {}
            for var, label in AGG_TYPES:
                s = self._pair_stats(sub, var)
                if s:
                    results[level][var] = s
            comp = results[level].get('composite', {})
            if comp:
                logger.info(f"  {level} (n={comp.get('n', 0)}): composite era-adj "
                            f"r={comp.get('correlation', float('nan')):.3f} "
                            f"(raw {comp.get('correlation_raw', float('nan')):.3f})")
        return results

    def analyze_overall(self, pairs_df):
        """Analyze inheritance pooling across all mentor-protege pairs (overall table)."""
        logger.info("\n\nANALYZING OVERALL INHERITANCE (all pairs):")
        logger.info("=" * 80)
        results = {}
        for var, label in AGG_TYPES:
            s = self._pair_stats(pairs_df, var)
            if s:
                results[var] = s
                logger.info(f"  {label}: era-adj r={s.get('correlation', float('nan')):.3f} "
                            f"(raw {s.get('correlation_raw', float('nan')):.3f}), n={s.get('n', 0)}")
        return results

    def analyze_by_mentor_type(self, pairs_df):
        """Inheritance by mentor's background (Offensive/Defensive)."""
        return self._analyze_subsets(
            pairs_df, 'mentor_background', ['Offensive', 'Defensive'],
            "ANALYZING INHERITANCE BY MENTOR BACKGROUND:")

    def analyze_by_protege_role(self, pairs_df):
        """Inheritance by the protege's role under the mentor (OC/DC).

        Replaces the prior mentor-keyed 'by coordinator type' split, which keyed
        the apprenticeship channel on the mentor's own coordinator background
        rather than the protege's role under that mentor.
        """
        return self._analyze_subsets(
            pairs_df, 'protege_role', ['OC', 'DC'],
            "ANALYZING INHERITANCE BY PROTEGE ROLE (under mentor):")

    def analyze_two_by_two(self, pairs_df):
        """Full mentor-background x protege-role 2x2 (the apprenticeship cells)."""
        logger.info("\n\nANALYZING 2x2 (mentor background x protege role):")
        logger.info("=" * 80)
        results = {}
        for bg in ['Offensive', 'Defensive']:
            for role in ['OC', 'DC']:
                cell = f'{bg}|{role}'
                sub = pairs_df[(pairs_df['mentor_background'] == bg)
                               & (pairs_df['protege_role'] == role)]
                results[cell] = {}
                for var, label in AGG_TYPES:
                    s = self._pair_stats(sub, var)
                    if s:
                        results[cell][var] = s
                comp = results[cell].get('composite', {})
                if comp:
                    logger.info(f"  {cell} (n={comp.get('n', 0)}): composite era-adj "
                                f"r={comp.get('correlation', float('nan')):.3f} "
                                f"(raw {comp.get('correlation_raw', float('nan')):.3f})")
        return results

    def create_visualization_by_type(self, coord_type_results):
        """Forest plot of OC vs DC coordinator-to-HC inheritance r with 95% CIs per component."""
        logger.info("\nCreating coordinator-type forest plot...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        # Top-to-bottom display order (composite first)
        comps = [
            ('composite', 'Composite'),
            ('fourth_down', '4th Down'),
            ('pass_heavy', 'Pass-Heavy'),
            ('deep_pass', 'Deep Pass'),
            ('two_point', 'Two-Point'),
        ]
        colors = {'OC': '#FF6B35', 'DC': '#004E89'}
        offsets = {'OC': 0.15, 'DC': -0.15}
        labels = {'OC': 'Protege OC', 'DC': 'Protege DC'}

        fig, ax = plt.subplots(figsize=(9, 6))
        yticks, ylabels = [], []
        seen_legend = set()

        for i, (var, label) in enumerate(comps):
            y = len(comps) - i
            yticks.append(y)
            ylabels.append(label)
            for ct in ['OC', 'DC']:
                res = coord_type_results.get(ct, {}).get(var)
                if not res:
                    continue
                r = res['correlation']
                lo = res.get('ci_low', r)
                hi = res.get('ci_high', r)
                yy = y + offsets[ct]
                ax.errorbar(
                    r, yy, xerr=[[max(0.0, r - lo)], [max(0.0, hi - r)]],
                    fmt='o', color=colors[ct], ecolor=colors[ct], elinewidth=2,
                    capsize=3, markersize=8, markeredgecolor='black', markeredgewidth=0.6,
                    label=labels[ct] if ct not in seen_legend else None,
                )
                seen_legend.add(ct)

        ax.axvline(0, color='gray', linestyle=':', linewidth=1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
        ax.set_ylim(0.4, len(comps) + 0.7)
        ax.set_xlabel('Mentor-protege correlation r, era-adjusted (95% CI)',
                      fontsize=13, fontweight='bold')
        ax.set_title('Offensive-aggression inheritance by protege role',
                     fontsize=14, fontweight='bold', pad=10)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':')
        plt.tight_layout()

        output_dir = Path("outputs/visualizations/inheritance")
        output_dir.mkdir(parents=True, exist_ok=True)
        png_path = output_dir / "inheritance_by_mentor_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def create_comparison_bar_chart(self, mentor_type_results, coord_type_results):
        """Create bar chart comparing inheritance strength"""
        logger.info("Creating comparison bar chart...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

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
                            fontsize=19, fontweight='bold')
            if 'Defensive' in mentor_type_results and var in mentor_type_results['Defensive']:
                if mentor_type_results['Defensive'][var]['significant']:
                    ax1.text(i + width/2, def_corrs[i], '*', ha='center', va='bottom',
                            fontsize=19, fontweight='bold')

        ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax1.set_xlabel('Aggression Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Mentor-Protégé Correlation (r)', fontsize=14, fontweight='bold')
        ax1.set_title('By Mentor Background Type', fontsize=15, fontweight='bold', pad=10)
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

        bars3 = ax2.bar(x - width/2, oc_corrs, width, label='Protege OC',
                       color='#FF6B35', alpha=0.8, edgecolor='black', linewidth=1)
        bars4 = ax2.bar(x + width/2, dc_corrs, width, label='Protege DC',
                       color='#004E89', alpha=0.8, edgecolor='black', linewidth=1)

        # Mark significant bars
        for i, var in enumerate(aggression_types):
            if 'OC' in coord_type_results and var in coord_type_results['OC']:
                if coord_type_results['OC'][var]['significant']:
                    ax2.text(i - width/2, oc_corrs[i], '*', ha='center', va='bottom',
                            fontsize=19, fontweight='bold')
            if 'DC' in coord_type_results and var in coord_type_results['DC']:
                if coord_type_results['DC'][var]['significant']:
                    ax2.text(i + width/2, dc_corrs[i], '*', ha='center', va='bottom',
                            fontsize=19, fontweight='bold')

        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax2.set_xlabel('Aggression Type', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Mentor-Protégé Correlation (r)', fontsize=14, fontweight='bold')
        ax2.set_title('By Protege Role', fontsize=15, fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend(loc='best', framealpha=0.95)
        ax2.grid(True, alpha=0.3, linestyle=':', axis='y')
        ax2.set_ylim(-0.05, 0.35)

        plt.tight_layout()

        # Save
        output_dir = Path("outputs/visualizations/inheritance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "inheritance_comparison_by_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved bar chart: {png_path}")

        plt.close()

    def save_results(self, pairs_df, overall_results, mentor_type_results,
                     protege_role_results, two_by_two_results):
        """Save results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save pairs
        pairs_file = output_dir / "mentor_protege_pairs_with_types.csv"
        pairs_df.to_csv(pairs_file, index=False)
        logger.info(f"Saved pairs data: {pairs_file}")

        # Save results. Canonical (unsuffixed) correlation/p keys are era-adjusted
        # (contemporary-group controlled); *_raw keys hold the raw estimate. The
        # 2x2 is the apprenticeship decomposition; the two marginals feed the FDR
        # family (the 2x2 cells are reported descriptively to avoid double-counting).
        results = {
            'overall': overall_results,
            'by_mentor_background': mentor_type_results,
            'by_protege_role': protege_role_results,
            'two_by_two': two_by_two_results,
        }

        results_file = output_dir / "inheritance_by_type_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results: {results_file}")

    def print_summary(self, mentor_type_results, protege_role_results):
        """Print summary (era-adjusted r primary; raw in parentheses)."""
        print("\n" + "=" * 80)
        print("AGGRESSION INHERITANCE (era-adjusted; raw in parens)")
        print("=" * 80)

        def _line(d):
            return (f"r={d.get('correlation', float('nan')):.3f} "
                    f"(raw {d.get('correlation_raw', float('nan')):.3f}), "
                    f"p={d.get('p_value', float('nan')):.4f}, n={d.get('n', 0)}")

        print("\nBY MENTOR BACKGROUND:")
        print("-" * 80)
        for bg_type in ['Offensive', 'Defensive']:
            if bg_type not in mentor_type_results:
                continue
            print(f"\n{bg_type} Mentors:")
            for var in ['fourth_down', 'pass_heavy', 'composite']:
                if var in mentor_type_results[bg_type]:
                    print(f"  {var}: {_line(mentor_type_results[bg_type][var])}")

        print("\n" + "-" * 80)
        print("BY PROTEGE ROLE (under mentor):")
        print("-" * 80)
        for role in ['OC', 'DC']:
            if role not in protege_role_results:
                continue
            print(f"\nProtege {role}:")
            for var in ['fourth_down', 'pass_heavy', 'composite']:
                if var in protege_role_results[role]:
                    print(f"  {var}: {_line(protege_role_results[role][var])}")

        print("\n" + "=" * 80)


def main():
    analyzer = InheritanceByTypeAnalyzer()

    try:
        # Load data
        analyzer.load_data()

        # Create pairs
        pairs_df = analyzer.create_mentor_protege_pairs()

        # Analyze overall (pooled across all pairs)
        overall_results = analyzer.analyze_overall(pairs_df)

        # Analyze by mentor background and by protege role (the two marginals)
        mentor_type_results = analyzer.analyze_by_mentor_type(pairs_df)
        protege_role_results = analyzer.analyze_by_protege_role(pairs_df)

        # Full mentor-background x protege-role 2x2 (apprenticeship decomposition)
        two_by_two_results = analyzer.analyze_two_by_two(pairs_df)

        # Create visualizations
        analyzer.create_visualization_by_type(protege_role_results)
        analyzer.create_comparison_bar_chart(mentor_type_results, protege_role_results)

        # Save results
        analyzer.save_results(pairs_df, overall_results, mentor_type_results,
                              protege_role_results, two_by_two_results)

        # Print summary
        analyzer.print_summary(mentor_type_results, protege_role_results)

        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
