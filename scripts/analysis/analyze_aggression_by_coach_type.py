#!/usr/bin/env python3
"""
Analyze Aggression Effects by Coach Background Type

This script examines whether the aggression-WAR relationship differs
between offensive coaches, defensive coaches, and other coaches.

Uses actual coaching history to classify head coaches by their coordinator
and position coach background.

Usage:
    python analyze_aggression_by_coach_type.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import corr_with_small_cluster_guard
from utils.data_paths import canonicalize_coach_name

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoachBackgroundClassifier:
    """Classify coaches by their offensive/defensive background"""

    def __init__(self):
        self.coach_backgrounds = None

    def load_coach_backgrounds(self):
        """Load pre-classified coach backgrounds from CSV"""
        background_file = 'data/processed/Coaching/coach_backgrounds_from_history.csv'

        if not os.path.exists(background_file):
            raise FileNotFoundError(f"Coach backgrounds file not found: {background_file}")

        logger.info(f"Loading pre-classified coach backgrounds...")
        self.coach_backgrounds = pd.read_csv(background_file)

        logger.info(f"Loaded backgrounds for {len(self.coach_backgrounds)} coaches")

        # Print distribution
        logger.info("\nCoach Background Distribution:")
        for bg, count in self.coach_backgrounds['Background'].value_counts().items():
            pct = (count / len(self.coach_backgrounds)) * 100
            logger.info(f"  {bg}: {count} ({pct:.1f}%)")

        return self.coach_backgrounds

    def match_with_aggression_data(self, aggression_war_data):
        """Match coach backgrounds with aggression-WAR data"""
        # Need to match coach names from aggression data with coach directories

        matched_data = []
        unmatched = []

        # Canonical name column for a robust exact match (handles punctuation, case,
        # and suffix differences) -- replaces the old unanchored last-name substring
        # match, which silently took the first row and so could attach the WRONG
        # coach's background for shared surnames (the Ryan brothers, the Harbaughs,
        # the two Jim Moras).
        bg = self.coach_backgrounds.copy()
        bg['_canon'] = bg['Coach_Name'].map(canonicalize_coach_name)

        for _, agg_row in aggression_war_data.iterrows():
            coach_name = agg_row['coach']

            # Try exact match on coach name
            matches = bg[bg['Coach_Name'] == coach_name]

            # Try directory-style name (spaces to underscores)
            if len(matches) == 0:
                coach_dir = coach_name.replace(' ', '_')
                matches = bg[bg['Coach_Directory'] == coach_dir]

            # Try canonical-name match (case/punctuation/suffix-insensitive, exact)
            if len(matches) == 0:
                matches = bg[bg['_canon'] == canonicalize_coach_name(coach_name)]

            # Require a UNIQUE match. An ambiguous result must not silently take the
            # first row -- skip it and warn, so a mismatch never propagates downstream.
            if len(matches) == 1:
                match = matches.iloc[0]
                row_with_bg = agg_row.to_dict()
                row_with_bg['Background'] = match['Background']
                row_with_bg['Offensive_Years'] = match['Offensive_Years']
                row_with_bg['Defensive_Years'] = match['Defensive_Years']
                matched_data.append(row_with_bg)
            else:
                if len(matches) > 1:
                    logger.warning(f"  ambiguous background match for '{coach_name}' "
                                   f"({len(matches)} candidates); skipping")
                unmatched.append(coach_name)

        matched_df = pd.DataFrame(matched_data)

        logger.info(f"\nMatching results:")
        logger.info(f"  Matched: {len(matched_df)} coach-year observations")
        logger.info(f"  Unmatched: {len(unmatched)} observations")

        if len(matched_df) > 0:
            logger.info("\nMatched data distribution:")
            for bg, count in matched_df['Background'].value_counts().items():
                pct = (count / len(matched_df)) * 100
                logger.info(f"  {bg}: {count} ({pct:.1f}%)")

        return matched_df


class AggressionByCoachTypeAnalyzer:
    """Analyze aggression-WAR relationship by coach background type"""

    def __init__(self):
        self.data = None

    def load_data(self):
        """Load and classify aggression-WAR data by coach type"""
        logger.info("Loading aggression-WAR data...")

        data_file = Path("outputs/analysis/aggression_war_merged_data.csv")
        if not data_file.exists():
            raise FileNotFoundError(
                f"Merged data not found: {data_file}\n"
                "Please run analyze_aggression_war_relationship.py first"
            )

        aggression_data = pd.read_csv(data_file)
        logger.info(f"Loaded {len(aggression_data)} coach-year observations")

        # Load pre-classified coach backgrounds
        classifier = CoachBackgroundClassifier()
        classifier.load_coach_backgrounds()

        # Match with aggression data
        self.data = classifier.match_with_aggression_data(aggression_data)

        logger.info(f"Final dataset: {len(self.data)} observations with coach background")

    def analyze_overall_by_type(self):
        """Analyze overall aggression-WAR relationship by coach type"""
        logger.info("\nAnalyzing overall relationships by coach type...")

        results = {}

        for bg_type in ['Offensive', 'Defensive', 'Other']:
            bg_data = self.data[self.data['Background'] == bg_type].copy()

            if len(bg_data) < 10:
                continue

            results[bg_type] = {}

            # Composite aggression
            comp_data = bg_data[['composite_aggression', 'annual_war', 'coach']].dropna()
            if len(comp_data) > 0:
                corr, p_val = stats.pearsonr(comp_data['composite_aggression'], comp_data['annual_war'])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    comp_data['composite_aggression'], comp_data['annual_war']
                )
                boot = corr_with_small_cluster_guard(
                    comp_data['composite_aggression'].to_numpy(),
                    comp_data['annual_war'].to_numpy(),
                    comp_data['coach'].to_numpy(),
                    n_boot=2000, seed=0,
                )
                results[bg_type]['composite'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(comp_data)),
                    'slope': float(slope),
                    'significant': bool(p_val < 0.05),
                    'ci_low': boot['ci_low'],
                    'ci_high': boot['ci_high'],
                    'n_coaches': boot['n_clusters'],
                    'p_clustered': boot.get('p_bootstrap'),
                    'small_cluster': boot.get('small_cluster'),
                    'p_wild_cluster': boot.get('p_wild_cluster'),
                }

            # Pass-heavy aggression
            pass_data = bg_data[['pass_heavy_aggression', 'annual_war', 'coach']].dropna()
            if len(pass_data) > 0:
                corr, p_val = stats.pearsonr(pass_data['pass_heavy_aggression'], pass_data['annual_war'])
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    pass_data['pass_heavy_aggression'], pass_data['annual_war']
                )
                boot = corr_with_small_cluster_guard(
                    pass_data['pass_heavy_aggression'].to_numpy(),
                    pass_data['annual_war'].to_numpy(),
                    pass_data['coach'].to_numpy(),
                    n_boot=2000, seed=0,
                )
                results[bg_type]['pass_heavy'] = {
                    'correlation': float(corr),
                    'p_value': float(p_val),
                    'n': int(len(pass_data)),
                    'slope': float(slope),
                    'significant': bool(p_val < 0.05),
                    'ci_low': boot['ci_low'],
                    'ci_high': boot['ci_high'],
                    'n_coaches': boot['n_clusters'],
                    'p_clustered': boot.get('p_bootstrap'),
                    'small_cluster': boot.get('small_cluster'),
                    'p_wild_cluster': boot.get('p_wild_cluster'),
                }

            logger.info(f"\n{bg_type} Coaches:")
            logger.info(f"  Composite:  r={results[bg_type]['composite']['correlation']:7.4f}, "
                       f"p={results[bg_type]['composite']['p_value']:.4f}, "
                       f"n={results[bg_type]['composite']['n']}")
            logger.info(f"  Pass-Heavy: r={results[bg_type]['pass_heavy']['correlation']:7.4f}, "
                       f"p={results[bg_type]['pass_heavy']['p_value']:.4f}, "
                       f"n={results[bg_type]['pass_heavy']['n']}")

        return results

    def analyze_by_type_and_era(self):
        """Analyze by coach type AND era"""
        logger.info("\nAnalyzing by coach type and era...")

        # Define eras
        self.data['era'] = pd.cut(
            self.data['year'],
            bins=[2005, 2011, 2017, 2025],
            labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
        )

        results = {}

        for bg_type in ['Offensive', 'Defensive', 'Other']:
            results[bg_type] = {}

            for era in ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']:
                era_bg_data = self.data[
                    (self.data['Background'] == bg_type) &
                    (self.data['era'] == era)
                ].copy()

                if len(era_bg_data) < 5:
                    continue

                results[bg_type][era] = {}

                # Composite aggression
                comp_data = era_bg_data[['composite_aggression', 'annual_war']].dropna()
                if len(comp_data) >= 5:
                    corr, p_val = stats.pearsonr(comp_data['composite_aggression'], comp_data['annual_war'])
                    results[bg_type][era]['composite'] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'n': int(len(comp_data)),
                        'significant': bool(p_val < 0.05)
                    }

                # Pass-heavy aggression
                pass_data = era_bg_data[['pass_heavy_aggression', 'annual_war']].dropna()
                if len(pass_data) >= 5:
                    corr, p_val = stats.pearsonr(pass_data['pass_heavy_aggression'], pass_data['annual_war'])
                    results[bg_type][era]['pass_heavy'] = {
                        'correlation': float(corr),
                        'p_value': float(p_val),
                        'n': int(len(pass_data)),
                        'significant': bool(p_val < 0.05)
                    }

        return results

    def create_visualization_by_type(self, overall_results):
        """Create scatter plots by coach type"""
        logger.info("Creating visualization by coach type...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        # Determine which coach types have sufficient data
        bg_types = ['Offensive', 'Defensive', 'Other']
        colors_map = {'Offensive': '#FF6B35', 'Defensive': '#004E89', 'Other': '#808080'}

        valid_types = []
        for bg_type in bg_types:
            bg_data = self.data[self.data['Background'] == bg_type].copy()
            clean_data = bg_data[['composite_aggression', 'annual_war']].dropna()
            if len(clean_data) >= 10:  # Require at least 10 observations
                valid_types.append(bg_type)

        if len(valid_types) == 0:
            logger.warning("No coach types have sufficient data for visualization")
            return

        # Create subplot grid based on number of valid coach types
        n_cols = len(valid_types)
        fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 12))

        # Handle case where only 1 column exists (axes won't be 2D)
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        def percent_formatter(x, pos):
            return f"{x*100:+.1f}%"

        def war_formatter(x, pos):
            return f"{x:+.1f}"

        # Row 1: Composite aggression
        for idx, bg_type in enumerate(valid_types):
            ax = axes[0, idx]
            color = colors_map[bg_type]

            bg_data = self.data[self.data['Background'] == bg_type].copy()
            clean_data = bg_data[['composite_aggression', 'annual_war']].dropna()

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
            ax.set_title(f'{bg_type} Coaches', fontsize=14, fontweight='bold', pad=10)

            # Format x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

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
        for idx, bg_type in enumerate(valid_types):
            ax = axes[1, idx]
            color = colors_map[bg_type]

            bg_data = self.data[self.data['Background'] == bg_type].copy()
            clean_data = bg_data[['pass_heavy_aggression', 'annual_war']].dropna()

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
            ax.set_title(f'{bg_type} Coaches', fontsize=14, fontweight='bold', pad=10)

            # Format x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(percent_formatter))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

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

        png_path = output_dir / "aggression_war_by_coach_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def create_temporal_visualization_by_type(self, era_results):
        """Create visualization showing temporal trends by coach type"""
        logger.info("Creating temporal visualization by coach type...")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))
        from plot_config import configure_plots
        configure_plots()

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        eras = ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
        bg_types = ['Offensive', 'Defensive', 'Other']
        bg_colors = {'Offensive': '#FF6B35', 'Defensive': '#004E89', 'Other': '#808080'}

        # Row 1: Composite Aggression over time
        for idx, era in enumerate(eras):
            ax = axes[0, idx]

            for bg_type in bg_types:
                if bg_type not in era_results or era not in era_results[bg_type]:
                    continue
                if 'composite' not in era_results[bg_type][era]:
                    continue

                result = era_results[bg_type][era]['composite']

                # Plot correlation as bar
                ax.barh(bg_type, result['correlation'],
                       color=bg_colors[bg_type], alpha=0.7,
                       edgecolor='black', linewidth=1)

                # Add significance marker
                if result['significant']:
                    ax.text(result['correlation'], bg_type, ' *',
                           fontsize=17, fontweight='bold',
                           verticalalignment='center')

            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Correlation (r)', fontsize=13, fontweight='bold')
            ax.set_title(f'{era}', fontsize=14, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle=':', axis='x')

            if idx == 0:
                ax.set_ylabel('Coach Type', fontsize=13, fontweight='bold')

        # Row 2: Pass-Heavy Aggression over time
        for idx, era in enumerate(eras):
            ax = axes[1, idx]

            for bg_type in bg_types:
                if bg_type not in era_results or era not in era_results[bg_type]:
                    continue
                if 'pass_heavy' not in era_results[bg_type][era]:
                    continue

                result = era_results[bg_type][era]['pass_heavy']

                # Plot correlation as bar
                ax.barh(bg_type, result['correlation'],
                       color=bg_colors[bg_type], alpha=0.7,
                       edgecolor='black', linewidth=1)

                # Add significance marker
                if result['significant']:
                    ax.text(result['correlation'], bg_type, ' *',
                           fontsize=17, fontweight='bold',
                           verticalalignment='center')

            ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Correlation (r)', fontsize=13, fontweight='bold')
            ax.set_title(f'{era}', fontsize=14, fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3, linestyle=':', axis='x')

            if idx == 0:
                ax.set_ylabel('Coach Type', fontsize=13, fontweight='bold')

        # Add row labels
        fig.text(0.01, 0.75, 'Composite Aggression', rotation=90,
                fontsize=15, fontweight='bold', verticalalignment='center')
        fig.text(0.01, 0.25, 'Pass-Heavy Aggression', rotation=90,
                fontsize=15, fontweight='bold', verticalalignment='center')

        plt.tight_layout(rect=[0.02, 0, 1, 0.99])

        # Save
        output_dir = Path("outputs/visualizations/performance")
        output_dir.mkdir(parents=True, exist_ok=True)

        png_path = output_dir / "aggression_war_temporal_by_coach_type.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved visualization: {png_path}")

        plt.close()

    def save_results(self, overall_results, era_results):
        """Save analysis results"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'overall_by_type': overall_results,
            'by_type_and_era': era_results
        }

        output_file = output_dir / "aggression_by_coach_type_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results: {output_file}")

        # Save the matched data
        data_file = output_dir / "aggression_war_with_coach_type.csv"
        self.data.to_csv(data_file, index=False)
        logger.info(f"Saved matched data: {data_file}")

    def print_summary(self, overall_results, era_results):
        """Print summary of findings"""
        print("\n" + "="*80)
        print("AGGRESSION ANALYSIS BY COACH BACKGROUND TYPE")
        print("="*80)

        print("\nOVERALL RELATIONSHIPS (2006-2024):")
        print("-" * 80)

        for bg_type in ['Offensive', 'Defensive', 'Other']:
            if bg_type not in overall_results:
                continue

            print(f"\n{bg_type} Coaches:")

            comp = overall_results[bg_type]['composite']
            print(f"  Composite Aggression:")
            print(f"    r = {comp['correlation']:.4f}, p = {comp['p_value']:.4f} "
                  f"({'SIG' if comp['significant'] else 'n.s.'}), n = {comp['n']}")

            pass_h = overall_results[bg_type]['pass_heavy']
            print(f"  Pass-Heavy Aggression:")
            print(f"    r = {pass_h['correlation']:.4f}, p = {pass_h['p_value']:.4f} "
                  f"({'SIG' if pass_h['significant'] else 'n.s.'}), n = {pass_h['n']}")

        print("\n" + "-" * 80)
        print("TEMPORAL TRENDS BY COACH TYPE:")
        print("-" * 80)

        for bg_type in ['Offensive', 'Defensive', 'Other']:
            if bg_type not in era_results:
                continue

            print(f"\n{bg_type} Coaches - Composite Aggression:")

            eras = ['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
            for era in eras:
                if era not in era_results[bg_type]:
                    continue
                if 'composite' not in era_results[bg_type][era]:
                    continue

                result = era_results[bg_type][era]['composite']
                sig = "SIG" if result['significant'] else "n.s."
                print(f"  {era}: r={result['correlation']:.3f}, p={result['p_value']:.4f} "
                      f"({sig}), n={result['n']}")

        print("\n" + "="*80)


def main():
    analyzer = AggressionByCoachTypeAnalyzer()

    try:
        # Load and classify data
        analyzer.load_data()

        # Analyze overall by type
        overall_results = analyzer.analyze_overall_by_type()

        # Analyze by type and era
        era_results = analyzer.analyze_by_type_and_era()

        # Create visualizations
        analyzer.create_visualization_by_type(overall_results)
        analyzer.create_temporal_visualization_by_type(era_results)

        # Save results
        analyzer.save_results(overall_results, era_results)

        # Print summary
        analyzer.print_summary(overall_results, era_results)

        logger.info("\nAnalysis complete!")

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
