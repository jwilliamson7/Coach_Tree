#!/usr/bin/env python3
"""
Analyze Mentor WAR → Protégé WAR Relationship

Examines whether a head coach's performance (measured by average WAR)
predicts their coordinators' later performance when those coordinators
become head coaches themselves.

Only considers head coach → coordinator relationships where the
coordinator later becomes a head coach.

Usage:
    python scripts/analysis/analyze_mentor_war_protege_war.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import sys
from scipy import stats
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import cluster_bootstrap_corr, corr_with_small_cluster_guard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MentorWARAnalyzer:
    """Analyze mentor WAR to protégé WAR relationships"""

    def __init__(self,
                 relationships_file: str = "data/processed/coaching_tree/relationships.csv",
                 war_file: str = "outputs/analysis/aggression_war_merged_data.csv",
                 output_dir: str = "outputs/analysis"):
        self.relationships_file = Path(relationships_file)
        self.war_file = Path(war_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.relationships = None
        self.war_data = None
        self.coach_avg_war = None
        self.mentor_protege_pairs = None

    def load_data(self) -> None:
        """Load relationships and WAR data"""
        logger.info("Loading data...")

        # Load relationships
        if not self.relationships_file.exists():
            raise FileNotFoundError(
                f"Relationships file not found: {self.relationships_file}\n"
                "Please run: python scripts/data_processing/build_coaching_tree.py"
            )
        self.relationships = pd.read_csv(self.relationships_file)
        logger.info(f"Loaded {len(self.relationships):,} relationships")

        # Load WAR data
        if not self.war_file.exists():
            raise FileNotFoundError(
                f"WAR data file not found: {self.war_file}\n"
                "Please run: python scripts/analysis/analyze_aggression_war_relationship.py"
            )
        self.war_data = pd.read_csv(self.war_file)
        logger.info(f"Loaded {len(self.war_data):,} coach-year WAR records")

        # Ensure we have the columns we need
        required_cols = ['coach', 'year', 'annual_war']
        missing_cols = [col for col in required_cols if col not in self.war_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in WAR data: {missing_cols}")

    def calculate_average_war(self) -> pd.DataFrame:
        """Calculate average WAR for each coach across their HC career"""
        logger.info("Calculating average WAR for each coach...")

        # Group by coach and calculate mean WAR
        coach_avg = self.war_data.groupby('coach').agg({
            'annual_war': ['mean', 'std', 'count']
        }).reset_index()

        coach_avg.columns = ['coach', 'avg_war', 'std_war', 'n_years']

        # Calculate standard error
        coach_avg['se_war'] = coach_avg['std_war'] / np.sqrt(coach_avg['n_years'])

        self.coach_avg_war = coach_avg
        logger.info(f"Calculated average WAR for {len(coach_avg)} coaches")

        return coach_avg

    def build_mentor_protege_pairs(self) -> pd.DataFrame:
        """Build mentor-protégé pairs with their average WARs"""
        logger.info("Building mentor-protégé pairs...")

        # Filter for coordinator_to_hc relationships
        # (coordinator who worked under a head coach)
        hc_to_coord = self.relationships[
            self.relationships['relationship_type'] == 'coordinator_to_hc'
        ].copy()

        logger.info(f"Found {len(hc_to_coord):,} HC→coordinator relationships")

        # For each coordinator, check if they later became a head coach
        # (i.e., appear in the WAR data)
        hc_to_coord['protege_became_hc'] = hc_to_coord['child_name'].isin(
            self.coach_avg_war['coach']
        )

        # Also check if mentor has WAR data
        hc_to_coord['mentor_has_war'] = hc_to_coord['parent_name'].isin(
            self.coach_avg_war['coach']
        )

        # Keep only pairs where both have HC WAR data
        valid_pairs = hc_to_coord[
            hc_to_coord['protege_became_hc'] & hc_to_coord['mentor_has_war']
        ].copy()

        logger.info(f"Found {len(valid_pairs):,} pairs where both became HCs")

        # Get unique mentor-protégé pairs (may have worked together multiple years)
        # Use the earliest year of their relationship
        unique_pairs = valid_pairs.groupby(['parent_name', 'child_name']).agg({
            'year': 'min',
            'child_role': 'first',
            'team': 'first'
        }).reset_index()

        logger.info(f"Unique mentor-protégé pairs: {len(unique_pairs)}")

        # Merge with mentor WAR
        pairs_with_war = unique_pairs.merge(
            self.coach_avg_war,
            left_on='parent_name',
            right_on='coach',
            how='inner',
            suffixes=('', '_mentor')
        ).drop('coach', axis=1)

        # Merge with protégé WAR
        pairs_with_war = pairs_with_war.merge(
            self.coach_avg_war,
            left_on='child_name',
            right_on='coach',
            how='inner',
            suffixes=('_mentor', '_protege')
        ).drop('coach', axis=1)

        # Rename columns for clarity
        pairs_with_war = pairs_with_war.rename(columns={
            'parent_name': 'mentor_name',
            'child_name': 'protege_name',
            'child_role': 'protege_role_under_mentor',
            'year': 'relationship_year'
        })

        self.mentor_protege_pairs = pairs_with_war
        logger.info(f"Final dataset: {len(pairs_with_war)} mentor-protégé pairs with WAR data")

        return pairs_with_war

    def analyze_correlation(self) -> Dict:
        """Analyze correlation between mentor and protégé WAR"""
        logger.info("Analyzing mentor-protégé WAR correlation...")

        df = self.mentor_protege_pairs

        # Overall correlation. A single mentor appears in many pairs, so the
        # pairs are not independent; cluster the bootstrap on mentor_name.
        r_overall, p_overall = stats.pearsonr(df['avg_war_mentor'], df['avg_war_protege'])
        boot_overall = cluster_bootstrap_corr(
            df['avg_war_mentor'].values, df['avg_war_protege'].values,
            df['mentor_name'].values, n_boot=2000, seed=42,
        )

        results = {
            'overall': {
                'n': len(df),
                'correlation': float(r_overall),
                'p_value': float(p_overall),
                'ci_low': boot_overall['ci_low'],
                'ci_high': boot_overall['ci_high'],
                'p_bootstrap_mentor_clustered': boot_overall['p_bootstrap'],
                'n_mentors': boot_overall['n_clusters'],
                'mentor_war_mean': float(df['avg_war_mentor'].mean()),
                'mentor_war_std': float(df['avg_war_mentor'].std()),
                'protege_war_mean': float(df['avg_war_protege'].mean()),
                'protege_war_std': float(df['avg_war_protege'].std())
            }
        }

        # Linear regression
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(
            df['avg_war_mentor'], df['avg_war_protege']
        )

        results['regression'] = {
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'std_err': float(std_err)
        }

        # By coordinator type (OC vs DC)
        results['by_coordinator_type'] = {}

        for coord_type in ['Offensive Coordinator', 'Defensive Coordinator']:
            subset = df[df['protege_role_under_mentor'] == coord_type]
            if len(subset) >= 10:  # Minimum sample size
                r, p = stats.pearsonr(subset['avg_war_mentor'], subset['avg_war_protege'])
                boot = corr_with_small_cluster_guard(
                    subset['avg_war_mentor'].values, subset['avg_war_protege'].values,
                    subset['mentor_name'].values, min_clusters=40, n_boot=2000, seed=42,
                )
                results['by_coordinator_type'][coord_type] = {
                    'n': len(subset),
                    'correlation': float(r),
                    'p_value': float(p),
                    'ci_low': boot['ci_low'],
                    'ci_high': boot['ci_high'],
                    'p_bootstrap_mentor_clustered': boot['p_bootstrap'],
                    'n_mentors': boot['n_clusters'],
                    'small_cluster': boot.get('small_cluster', False),
                    'p_wild_cluster': boot.get('p_wild_cluster'),
                    'mentor_war_mean': float(subset['avg_war_mentor'].mean()),
                    'protege_war_mean': float(subset['avg_war_protege'].mean())
                }
            else:
                logger.info(f"Insufficient data for {coord_type}: n={len(subset)}")

        # By era (split by relationship year)
        df['era'] = pd.cut(
            df['relationship_year'],
            bins=[0, 2000, 2010, 2020, 3000],
            labels=['Pre-2000', '2000-2009', '2010-2019', '2020+']
        )

        results['by_era'] = {}
        for era in df['era'].dropna().unique():
            subset = df[df['era'] == era]
            if len(subset) >= 10:
                r, p = stats.pearsonr(subset['avg_war_mentor'], subset['avg_war_protege'])
                # Era splits were previously naive Pearson (ignored mentor clustering
                # AND small cluster counts). Route through the same clustered guard:
                # small eras (e.g. 2020+, ~few mentors) get a wild cluster bootstrap p.
                boot = corr_with_small_cluster_guard(
                    subset['avg_war_mentor'].values, subset['avg_war_protege'].values,
                    subset['mentor_name'].values, min_clusters=40, n_boot=2000, seed=42,
                )
                results['by_era'][str(era)] = {
                    'n': len(subset),
                    'correlation': float(r),
                    'p_value': float(p),
                    'p_bootstrap_mentor_clustered': boot['p_bootstrap'],
                    'n_mentors': boot['n_clusters'],
                    'small_cluster': boot.get('small_cluster', False),
                    'p_wild_cluster': boot.get('p_wild_cluster'),
                }

        return results

    def save_results(self, analysis_results: Dict) -> None:
        """Save analysis results"""
        logger.info("Saving results...")

        # Save mentor-protégé pairs
        pairs_file = self.output_dir / "mentor_protege_war_pairs.csv"
        self.mentor_protege_pairs.to_csv(pairs_file, index=False)
        logger.info(f"Saved pairs data: {pairs_file}")

        # Save analysis results
        results_file = self.output_dir / "mentor_protege_war_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        logger.info(f"Saved analysis results: {results_file}")

        # Create summary log
        log_file = self.output_dir / "mentor_protege_war_analysis.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MENTOR WAR → PROTÉGÉ WAR ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("Research Question:\n")
            f.write("Does a head coach's performance predict the later performance\n")
            f.write("of their coordinators when those coordinators become head coaches?\n\n")

            f.write("Relationships: Head Coach → Coordinator (later became HC)\n\n")

            f.write("-" * 80 + "\n")
            f.write("OVERALL RESULTS\n")
            f.write("-" * 80 + "\n\n")

            overall = analysis_results['overall']
            f.write(f"Sample Size: {overall['n']} mentor-protégé pairs\n\n")

            f.write(f"Mentor Average WAR: {overall['mentor_war_mean']:.4f} ± {overall['mentor_war_std']:.4f}\n")
            f.write(f"Protégé Average WAR: {overall['protege_war_mean']:.4f} ± {overall['protege_war_std']:.4f}\n\n")

            f.write(f"Correlation: r = {overall['correlation']:.3f}\n")
            f.write(f"P-value: p = {overall['p_value']:.4f}\n")

            if overall['p_value'] < 0.001:
                sig = "***"
            elif overall['p_value'] < 0.01:
                sig = "**"
            elif overall['p_value'] < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            f.write(f"Significance: {sig}\n\n")

            reg = analysis_results['regression']
            f.write(f"Linear Regression:\n")
            f.write(f"  Protégé WAR = {reg['intercept']:.4f} + {reg['slope']:.4f} * Mentor WAR\n")
            f.write(f"  R² = {reg['r_squared']:.4f}\n")
            f.write(f"  Standard Error = {reg['std_err']:.4f}\n\n")

            # Interpretation
            if overall['correlation'] > 0 and overall['p_value'] < 0.05:
                f.write("INTERPRETATION: Significant positive relationship detected.\n")
                f.write("Coordinators who worked under successful head coaches tend to\n")
                f.write("become more successful head coaches themselves.\n\n")
            elif overall['correlation'] < 0 and overall['p_value'] < 0.05:
                f.write("INTERPRETATION: Significant negative relationship detected.\n")
                f.write("Coordinators who worked under successful head coaches tend to\n")
                f.write("become less successful head coaches themselves.\n\n")
            else:
                f.write("INTERPRETATION: No significant relationship detected.\n")
                f.write("Mentor performance does not predict protégé performance.\n\n")

            # By coordinator type
            if analysis_results.get('by_coordinator_type'):
                f.write("-" * 80 + "\n")
                f.write("BY COORDINATOR TYPE\n")
                f.write("-" * 80 + "\n\n")

                for coord_type, results in analysis_results['by_coordinator_type'].items():
                    f.write(f"{coord_type}:\n")
                    f.write(f"  n = {results['n']}\n")
                    f.write(f"  r = {results['correlation']:.3f}, p = {results['p_value']:.4f}\n")
                    f.write(f"  Mentor WAR: {results['mentor_war_mean']:.4f}\n")
                    f.write(f"  Protégé WAR: {results['protege_war_mean']:.4f}\n\n")

            # By era
            if analysis_results.get('by_era'):
                f.write("-" * 80 + "\n")
                f.write("BY ERA (Relationship Year)\n")
                f.write("-" * 80 + "\n\n")

                for era, results in sorted(analysis_results['by_era'].items()):
                    f.write(f"{era}:\n")
                    f.write(f"  n = {results['n']}\n")
                    f.write(f"  r = {results['correlation']:.3f}, p = {results['p_value']:.4f}\n\n")

            f.write("=" * 80 + "\n")

        logger.info(f"Saved summary log: {log_file}")

    def run(self) -> None:
        """Run the complete analysis"""
        logger.info("Starting mentor WAR → protégé WAR analysis...")

        self.load_data()
        self.calculate_average_war()
        self.build_mentor_protege_pairs()
        results = self.analyze_correlation()
        self.save_results(results)

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nKey Result:")
        logger.info(f"  n = {results['overall']['n']} pairs")
        logger.info(f"  r = {results['overall']['correlation']:.3f}")
        logger.info(f"  p = {results['overall']['p_value']:.4f}")
        logger.info(f"\nOutput files saved to: {self.output_dir}")


def main():
    """Main execution"""
    analyzer = MentorWARAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
