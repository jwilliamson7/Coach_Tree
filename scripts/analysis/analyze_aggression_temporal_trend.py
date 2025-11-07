#!/usr/bin/env python3
"""
Analyze Temporal Trend in Average Aggression

This script tests whether the league-wide increase in average coaching aggression
over time is statistically significant. This supports the claim that aggressive
tactics diffused through the league as analytics-driven decision-making spread.

Statistical tests performed:
1. Linear regression: Tests if there's a significant linear trend over time
2. One-way ANOVA: Tests if mean aggression differs across three eras
3. Pairwise comparisons: Tests specific era-to-era increases

Usage:
    python analyze_aggression_temporal_trend.py [--gene_dir data/processed/coaching_genes]
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from scipy import stats
from scipy.stats import f_oneway, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressionTemporalTrendAnalyzer:
    """Analyze statistical significance of temporal trends in aggression"""

    def __init__(self, gene_dir: str = "data/processed/coaching_genes",
                 output_dir: str = "outputs/analysis"):
        self.gene_dir = Path(gene_dir)
        self.output_dir = Path(output_dir)
        self.aggression_data = None
        self.results = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define eras (same as WAR analysis)
        self.eras = {
            'Early (2006-2011)': (2006, 2011),
            'Middle (2012-2017)': (2012, 2017),
            'Late (2018-2024)': (2018, 2024)
        }

    def load_data(self) -> None:
        """Load aggression gene data"""
        logger.info("Loading aggression gene data...")

        aggression_file = self.gene_dir / "aggression_gene_by_year.csv"
        if not aggression_file.exists():
            raise FileNotFoundError(
                f"Aggression gene file not found: {aggression_file}\n"
                "Please run: python scripts/analysis/calculate_aggression_gene.py"
            )

        self.aggression_data = pd.read_csv(aggression_file)
        logger.info(f"Loaded {len(self.aggression_data):,} coach-year records")

        required_cols = ['season', 'composite_aggression']
        missing_cols = [col for col in required_cols if col not in self.aggression_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def test_linear_trend(self) -> dict:
        """Test for linear trend over time using regression"""
        logger.info("Testing linear trend over time...")

        # Calculate yearly means
        yearly_means = self.aggression_data.groupby('season')['composite_aggression'].mean()
        years = yearly_means.index.values
        aggression = yearly_means.values

        # Linear regression: aggression ~ year
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, aggression)

        result = {
            'slope': float(slope),
            'slope_percent_per_year': float(slope * 100),  # Convert to percentage
            'intercept': float(intercept),
            'r': float(r_value),
            'r_squared': float(r_value**2),
            'p_value': float(p_value),
            'std_err': float(std_err),
            'n_years': int(len(years)),
            'year_range': f"{int(years.min())}-{int(years.max())}",
            'significant': bool(p_value < 0.05),
            'interpretation': self._interpret_trend(slope, p_value)
        }

        logger.info(f"Linear trend: slope={slope:.6f} ({slope*100:+.3f}% per year), "
                   f"r²={r_value**2:.4f}, p={p_value:.4f}")

        return result

    def test_era_differences(self) -> dict:
        """Test if mean aggression differs significantly across eras"""
        logger.info("Testing era differences using ANOVA...")

        # Assign eras to data
        self.aggression_data['era'] = self.aggression_data['season'].apply(self._assign_era)

        # Calculate yearly means for each era
        yearly_means = self.aggression_data.groupby('season')['composite_aggression'].mean()

        era_groups = {}
        era_stats = {}

        for era_name, (start, end) in self.eras.items():
            era_years = yearly_means[(yearly_means.index >= start) & (yearly_means.index <= end)]
            era_groups[era_name] = era_years.values
            era_stats[era_name] = {
                'mean': float(era_years.mean()),
                'std': float(era_years.std()),
                'n_years': int(len(era_years)),
                'year_range': f"{int(start)}-{int(end)}"
            }
            logger.info(f"{era_name}: mean={era_years.mean():.6f}, std={era_years.std():.6f}, n={len(era_years)}")

        # One-way ANOVA
        f_stat, p_value = f_oneway(
            era_groups['Early (2006-2011)'],
            era_groups['Middle (2012-2017)'],
            era_groups['Late (2018-2024)']
        )

        result = {
            'anova': {
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05)
            },
            'era_statistics': era_stats,
            'interpretation': self._interpret_anova(era_stats, p_value)
        }

        logger.info(f"ANOVA: F={f_stat:.4f}, p={p_value:.4f}")

        return result

    def test_pairwise_comparisons(self) -> dict:
        """Test specific era-to-era increases"""
        logger.info("Testing pairwise era comparisons...")

        # Calculate yearly means
        yearly_means = self.aggression_data.groupby('season')['composite_aggression'].mean()

        # Get era groups
        era_data = {}
        for era_name, (start, end) in self.eras.items():
            era_data[era_name] = yearly_means[(yearly_means.index >= start) & (yearly_means.index <= end)].values

        comparisons = [
            ('Early (2006-2011)', 'Middle (2012-2017)'),
            ('Early (2006-2011)', 'Late (2018-2024)'),
            ('Middle (2012-2017)', 'Late (2018-2024)')
        ]

        results = {}

        for era1, era2 in comparisons:
            # Two-sample t-test (two-tailed)
            t_stat, p_value_ttest = stats.ttest_ind(era_data[era1], era_data[era2])

            # Mann-Whitney U test (non-parametric alternative)
            u_stat, p_value_mw = mannwhitneyu(era_data[era1], era_data[era2],
                                              alternative='two-sided')

            # Calculate effect size (Cohen's d)
            mean1, mean2 = era_data[era1].mean(), era_data[era2].mean()
            pooled_std = np.sqrt((era_data[era1].std()**2 + era_data[era2].std()**2) / 2)
            cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0

            results[f"{era1} vs {era2}"] = {
                'mean_difference': float(mean2 - mean1),
                'mean_difference_percent': float((mean2 - mean1) * 100),
                't_test': {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value_ttest),
                    'significant': p_value_ttest < 0.05
                },
                'mann_whitney': {
                    'u_statistic': float(u_stat),
                    'p_value': float(p_value_mw),
                    'significant': p_value_mw < 0.05
                },
                'effect_size': {
                    'cohens_d': float(cohens_d),
                    'interpretation': self._interpret_cohens_d(cohens_d)
                }
            }

            logger.info(f"{era1} vs {era2}: "
                       f"Δ={mean2-mean1:+.6f} ({(mean2-mean1)*100:+.3f}%), "
                       f"t={t_stat:.3f}, p={p_value_ttest:.4f}, d={cohens_d:.3f}")

        return results

    def _assign_era(self, year: int) -> str:
        """Assign a year to an era"""
        for era_name, (start, end) in self.eras.items():
            if start <= year <= end:
                return era_name
        return "Unknown"

    def _interpret_trend(self, slope: float, p_value: float) -> str:
        """Interpret the linear trend result"""
        if p_value >= 0.05:
            return "No significant linear trend detected."

        direction = "increasing" if slope > 0 else "decreasing"
        magnitude = abs(slope * 18)  # Total change over 18 years (2006-2024)

        return (f"Significant {direction} linear trend: average aggression changed by "
                f"{magnitude*100:+.2f}% over the 19-year period (p={p_value:.4f}).")

    def _interpret_anova(self, era_stats: dict, p_value: float) -> str:
        """Interpret ANOVA results"""
        if p_value >= 0.05:
            return "No significant difference in mean aggression across eras."

        means = {era: stats['mean'] for era, stats in era_stats.items()}
        early = means['Early (2006-2011)']
        middle = means['Middle (2012-2017)']
        late = means['Late (2018-2024)']

        return (f"Significant differences across eras (p={p_value:.4f}): "
                f"Early mean={early:.6f}, Middle mean={middle:.6f}, Late mean={late:.6f}. "
                f"Overall increase from Early to Late: {(late-early)*100:+.2f}%.")

    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"

    def save_results(self) -> None:
        """Save results to JSON file"""
        output_file = self.output_dir / "aggression_temporal_trend_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"Results saved to: {output_file}")

    def print_summary(self) -> None:
        """Print human-readable summary"""
        logger.info("\n" + "="*80)
        logger.info("TEMPORAL TREND ANALYSIS SUMMARY")
        logger.info("="*80)

        # Linear trend
        trend = self.results['linear_trend']
        logger.info(f"\nLinear Trend Test:")
        logger.info(f"  Slope: {trend['slope_percent_per_year']:+.3f}% per year")
        logger.info(f"  R²: {trend['r_squared']:.4f}")
        logger.info(f"  P-value: {trend['p_value']:.4f}")
        logger.info(f"  Significant: {trend['significant']}")
        logger.info(f"  {trend['interpretation']}")

        # ANOVA
        anova = self.results['era_comparison']['anova']
        logger.info(f"\nANOVA Test (Comparing Three Eras):")
        logger.info(f"  F-statistic: {anova['f_statistic']:.4f}")
        logger.info(f"  P-value: {anova['p_value']:.4f}")
        logger.info(f"  Significant: {anova['significant']}")
        logger.info(f"  {self.results['era_comparison']['interpretation']}")

        # Pairwise comparisons
        logger.info(f"\nPairwise Era Comparisons:")
        for comparison, result in self.results['pairwise_comparisons'].items():
            logger.info(f"\n  {comparison}:")
            logger.info(f"    Mean difference: {result['mean_difference_percent']:+.3f}%")
            logger.info(f"    T-test p-value: {result['t_test']['p_value']:.4f}")
            logger.info(f"    Significant: {result['t_test']['significant']}")
            logger.info(f"    Effect size (Cohen's d): {result['effect_size']['cohens_d']:.3f} "
                       f"({result['effect_size']['interpretation']})")

        logger.info("\n" + "="*80)

    def run(self) -> None:
        """Execute the full analysis pipeline"""
        logger.info("Starting temporal trend analysis...")

        self.load_data()

        # Run all tests
        self.results['linear_trend'] = self.test_linear_trend()
        self.results['era_comparison'] = self.test_era_differences()
        self.results['pairwise_comparisons'] = self.test_pairwise_comparisons()

        # Save and summarize
        self.save_results()
        self.print_summary()

        logger.info("\nAnalysis complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Test statistical significance of temporal trends in aggression'
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
        default='outputs/analysis',
        help='Output directory for results'
    )

    args = parser.parse_args()

    analyzer = AggressionTemporalTrendAnalyzer(
        gene_dir=args.gene_dir,
        output_dir=args.output_dir
    )

    analyzer.run()


if __name__ == "__main__":
    main()
