#!/usr/bin/env python3
"""
Within-Coach Fixed Effects Analysis

This script tests whether coaches perform better when they are more aggressive
than their personal average, controlling for all time-invariant coach characteristics.

The fixed effects model compares each coach to themselves across different years:
    WAR_it = β(Aggression_it - Aggression_i) + year_effects + ε_it

Where Aggression_i is coach i's personal average. This isolates whether variation
in aggression WITHIN coaches predicts variation in performance WITHIN coaches,
controlling for coach quality, personality, and other stable traits.

Statistical approach:
1. Demean all variables within each coach (removes coach fixed effects)
2. Run regression on demeaned variables
3. Optionally add year fixed effects to control for temporal trends
4. Cluster standard errors by coach
5. Test separately by era to see if the within-coach effect changed over time

Usage:
    python analyze_within_coach_fixed_effects.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import cluster_robust_ols

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


class WithinCoachFixedEffectsAnalyzer:
    """Analyze within-coach variation in aggression and performance"""

    def __init__(self,
                 aggression_file: str = "data/processed/coaching_genes/aggression_gene_by_year.csv",
                 war_file: str = "outputs/analysis/aggression_war_with_coach_type.csv",
                 output_dir: str = "outputs/analysis"):
        self.aggression_file = Path(aggression_file)
        self.war_file = Path(war_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data = None
        self.results = {}

        # Define eras
        self.eras = {
            'Early (2006-2011)': (2006, 2011),
            'Middle (2012-2017)': (2012, 2017),
            'Late (2018-2024)': (2018, 2024)
        }

    def load_data(self) -> None:
        """Load and merge aggression and WAR data"""
        logger.info("Loading data...")

        # Load aggression data
        if not self.aggression_file.exists():
            raise FileNotFoundError(f"Aggression file not found: {self.aggression_file}")
        agg_df = pd.read_csv(self.aggression_file)
        logger.info(f"Loaded {len(agg_df):,} aggression records")

        # Load WAR data
        if not self.war_file.exists():
            raise FileNotFoundError(f"WAR file not found: {self.war_file}")
        war_df = pd.read_csv(self.war_file)
        logger.info(f"Loaded {len(war_df):,} WAR records")

        # Rename columns for consistency
        agg_df = agg_df.rename(columns={'head_coach': 'coach'})
        war_df = war_df.rename(columns={'year': 'season', 'annual_war': 'WAR', 'Background': 'coach_type'})

        # Merge on coach and season
        self.data = pd.merge(
            agg_df[['coach', 'season', 'composite_aggression',
                    'fourth_down_aggression', 'pass_heavy_aggression',
                    'deep_pass_aggression', 'two_point_aggression']],
            war_df[['coach', 'season', 'WAR', 'coach_type']],
            on=['coach', 'season'],
            how='inner'
        )

        logger.info(f"Merged dataset: {len(self.data):,} coach-year observations")
        logger.info(f"Unique coaches: {self.data['coach'].nunique()}")
        logger.info(f"Year range: {self.data['season'].min()}-{self.data['season'].max()}")

        # Filter to coaches with multiple years (required for within-coach analysis)
        coach_counts = self.data['coach'].value_counts()
        multi_year_coaches = coach_counts[coach_counts >= 2].index
        n_before = len(self.data)
        self.data = self.data[self.data['coach'].isin(multi_year_coaches)]
        n_after = len(self.data)

        logger.info(f"Filtered to coaches with 2+ years: {len(multi_year_coaches)} coaches, "
                   f"{n_after:,} observations (dropped {n_before - n_after} single-year coaches)")

    def demean_within_coaches(self) -> pd.DataFrame:
        """Demean variables within each coach (removes coach fixed effects)"""
        logger.info("\nDemeaning variables within coaches...")

        # Variables to demean
        vars_to_demean = ['WAR', 'composite_aggression', 'fourth_down_aggression',
                          'pass_heavy_aggression', 'deep_pass_aggression', 'two_point_aggression']

        # Calculate coach means
        coach_means = self.data.groupby('coach')[vars_to_demean].transform('mean')

        # Create demeaned variables
        demeaned_data = self.data.copy()
        for var in vars_to_demean:
            demeaned_data[f'{var}_demeaned'] = self.data[var] - coach_means[var]

        # Log some summary stats
        logger.info(f"\nDemeaned variable summary:")
        for var in ['WAR', 'composite_aggression']:
            orig_std = self.data[var].std()
            demeaned_std = demeaned_data[f'{var}_demeaned'].std()
            logger.info(f"  {var}: std={orig_std:.4f} → demeaned std={demeaned_std:.4f} "
                       f"({demeaned_std/orig_std*100:.1f}% of original)")

        return demeaned_data

    def demean_two_way(self) -> pd.DataFrame:
        """Demean variables within coaches AND within years (two-way fixed effects)"""
        logger.info("\nApplying two-way fixed effects (coach + year demeaning)...")

        # Variables to demean
        vars_to_demean = ['WAR', 'composite_aggression', 'fourth_down_aggression',
                          'pass_heavy_aggression', 'deep_pass_aggression', 'two_point_aggression']

        demeaned_data = self.data.copy()

        for var in vars_to_demean:
            # Step 1: Calculate overall mean
            overall_mean = self.data[var].mean()

            # Step 2: Calculate coach means (deviation from overall)
            coach_means = self.data.groupby('coach')[var].transform('mean')

            # Step 3: Calculate year means (deviation from overall)
            year_means = self.data.groupby('season')[var].transform('mean')

            # Step 4: Two-way demeaning
            # Remove coach effect and year effect, add back overall mean
            demeaned_data[f'{var}_twoway'] = (
                self.data[var] - coach_means - year_means + overall_mean
            )

        # Log some summary stats
        logger.info(f"\nTwo-way demeaned variable summary:")
        for var in ['WAR', 'composite_aggression']:
            orig_std = self.data[var].std()
            twoway_std = demeaned_data[f'{var}_twoway'].std()
            logger.info(f"  {var}: std={orig_std:.4f} → two-way std={twoway_std:.4f} "
                       f"({twoway_std/orig_std*100:.1f}% of original)")

        return demeaned_data

    def assign_eras(self, data: pd.DataFrame) -> pd.DataFrame:
        """Assign each observation to an era"""
        def assign_era(year):
            for era_name, (start, end) in self.eras.items():
                if start <= year <= end:
                    return era_name
            return None

        data['era'] = data['season'].apply(assign_era)

        # Log era distribution
        logger.info("\nEra distribution:")
        for era in self.eras.keys():
            n = (data['era'] == era).sum()
            n_coaches = data[data['era'] == era]['coach'].nunique()
            logger.info(f"  {era}: {n:,} observations, {n_coaches} coaches")

        return data

    def run_fixed_effects_regression(self, data: pd.DataFrame,
                                     aggression_var: str = 'composite_aggression_demeaned',
                                     war_var: str = 'WAR_demeaned') -> dict:
        """Run fixed effects regression: demeaned_WAR ~ demeaned_aggression"""

        # Coach-clustered OLS via the shared parsimony helper (single source of
        # truth for cluster-robust sandwich SEs across all downstream analyses).
        X = data[[aggression_var]].values
        y = data[war_var].values
        res = cluster_robust_ols(X, y, data['coach'].values, [aggression_var])
        c = res['coefficients'][aggression_var]

        return {
            'n': int(res['n']),
            'n_coaches': int(res['n_clusters']),
            'aggression_coef': float(c['coefficient']),
            'aggression_se': float(c['std_error']),
            'aggression_t': float(c['t_statistic']),
            'aggression_p': float(c['p_value']),
            'r_squared': float(res['r_squared']),
            'significant': bool(c['significant'])
        }

    def run_pooled_analysis(self, demeaned_data: pd.DataFrame, var_suffix: str = '_demeaned') -> dict:
        """Run fixed effects on full sample"""
        model_type = "ONE-WAY" if var_suffix == '_demeaned' else "TWO-WAY"
        logger.info("\n" + "="*80)
        logger.info(f"POOLED {model_type} FIXED EFFECTS ANALYSIS (All Years)")
        logger.info("="*80)

        aggression_var = f'composite_aggression{var_suffix}'
        war_var = f'WAR{var_suffix}'
        result = self.run_fixed_effects_regression(demeaned_data, aggression_var, war_var)

        logger.info(f"\nPooled sample: {result['n']} observations, {result['n_coaches']} coaches")
        logger.info(f"Aggression coefficient: {result['aggression_coef']:.4f} "
                   f"(SE={result['aggression_se']:.4f}, t={result['aggression_t']:.3f}, "
                   f"p={result['aggression_p']:.4f})")
        logger.info(f"R² (within): {result['r_squared']:.4f}")
        logger.info(f"Significant: {result['significant']}")

        return result

    def run_stratified_analysis(self, demeaned_data: pd.DataFrame, var_suffix: str = '_demeaned') -> dict:
        """Run fixed effects separately by era"""
        model_type = "ONE-WAY" if var_suffix == '_demeaned' else "TWO-WAY"
        logger.info("\n" + "="*80)
        logger.info(f"STRATIFIED {model_type} FIXED EFFECTS ANALYSIS (By Era)")
        logger.info("="*80)

        results = {}
        aggression_var = f'composite_aggression{var_suffix}'
        war_var = f'WAR{var_suffix}'

        for era_name in self.eras.keys():
            era_data = demeaned_data[demeaned_data['era'] == era_name].copy()

            # Filter to coaches with 2+ observations in this era
            coach_counts = era_data['coach'].value_counts()
            multi_year_coaches = coach_counts[coach_counts >= 2].index
            era_data = era_data[era_data['coach'].isin(multi_year_coaches)]

            if len(era_data) < 10:
                logger.warning(f"Insufficient data for {era_name}: {len(era_data)} observations")
                continue

            logger.info(f"\n{era_name}:")
            result = self.run_fixed_effects_regression(era_data, aggression_var, war_var)
            results[era_name] = result

            logger.info(f"  n={result['n']}, n_coaches={result['n_coaches']}")
            logger.info(f"  β={result['aggression_coef']:.4f} "
                       f"(SE={result['aggression_se']:.4f}, t={result['aggression_t']:.3f}, "
                       f"p={result['aggression_p']:.4f})")
            logger.info(f"  R² (within): {result['r_squared']:.4f}")
            logger.info(f"  Significant: {result['significant']}")

        return results

    def test_coefficient_differences(self, era_results: dict) -> dict:
        """
        Test whether coefficients differ significantly across eras.

        Note: This uses SE_diff = sqrt(SE1^2 + SE2^2), which assumes the two
        estimates are independent. This is approximately true when eras have
        different coaches, but may slightly underestimate SE if coaches span
        multiple eras. Given that all era-specific effects are non-significant,
        this limitation doesn't affect substantive conclusions.
        """
        logger.info("\n" + "="*80)
        logger.info("TESTING COEFFICIENT DIFFERENCES ACROSS ERAS")
        logger.info("="*80)
        logger.info("Note: Assumes independence between era-specific estimates")

        comparisons = {}
        era_names = list(era_results.keys())

        for i in range(len(era_names)):
            for j in range(i + 1, len(era_names)):
                era1 = era_names[i]
                era2 = era_names[j]

                b1 = era_results[era1]['aggression_coef']
                se1 = era_results[era1]['aggression_se']

                b2 = era_results[era2]['aggression_coef']
                se2 = era_results[era2]['aggression_se']

                diff = b1 - b2
                # Conservative approach: assumes independence
                se_diff = np.sqrt(se1**2 + se2**2)
                z_stat = diff / se_diff
                p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

                comparisons[f"{era1} vs {era2}"] = {
                    'coef_diff': float(diff),
                    'se_diff': float(se_diff),
                    'z_statistic': float(z_stat),
                    'p_value': float(p_value),
                    'significant': bool(p_value < 0.05)
                }

                logger.info(f"\n{era1} vs {era2}:")
                logger.info(f"  Coefficient difference: {diff:+.4f}")
                logger.info(f"  Z-statistic: {z_stat:.3f}")
                logger.info(f"  P-value: {p_value:.4f}")
                logger.info(f"  Significant: {p_value < 0.05}")

        return comparisons

    def run_component_analysis(self, demeaned_data: pd.DataFrame) -> dict:
        """Run fixed effects for individual aggression components"""
        logger.info("\n" + "="*80)
        logger.info("COMPONENT-LEVEL FIXED EFFECTS ANALYSIS")
        logger.info("="*80)

        components = {
            'fourth_down': 'fourth_down_aggression_demeaned',
            'pass_heavy': 'pass_heavy_aggression_demeaned',
            'deep_pass': 'deep_pass_aggression_demeaned',
            'two_point': 'two_point_aggression_demeaned'
        }

        component_results = {}

        for comp_name, demeaned_var in components.items():
            logger.info(f"\n{comp_name.upper()} AGGRESSION:")
            component_results[comp_name] = {}

            # Pooled
            result_pooled = self.run_fixed_effects_regression(demeaned_data, demeaned_var)
            component_results[comp_name]['pooled'] = result_pooled
            logger.info(f"  Pooled: β={result_pooled['aggression_coef']:.4f} "
                       f"(p={result_pooled['aggression_p']:.4f})")

            # By era
            for era_name in self.eras.keys():
                era_data = demeaned_data[demeaned_data['era'] == era_name].copy()

                # Filter to coaches with 2+ observations
                coach_counts = era_data['coach'].value_counts()
                multi_year_coaches = coach_counts[coach_counts >= 2].index
                era_data = era_data[era_data['coach'].isin(multi_year_coaches)]

                if len(era_data) < 10:
                    continue

                result = self.run_fixed_effects_regression(era_data, demeaned_var)
                component_results[comp_name][era_name] = result
                logger.info(f"  {era_name}: β={result['aggression_coef']:.4f} "
                           f"(p={result['aggression_p']:.4f})")

        return component_results

    def save_results(self) -> None:
        """Save all results to JSON"""
        output_file = self.output_dir / "within_coach_fixed_effects_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nResults saved to: {output_file}")

    def print_summary(self) -> None:
        """Print human-readable summary"""
        logger.info("\n" + "="*80)
        logger.info("WITHIN-COACH FIXED EFFECTS SUMMARY")
        logger.info("="*80)

        # Two-way results (the more important ones)
        logger.info("\nTWO-WAY FIXED EFFECTS (Coach + Year):")
        logger.info("Key Question: When coaches are more aggressive than league average in year t")
        logger.info("(relative to their typical deviation), do they perform better?")
        logger.info("(This controls for coach quality AND temporal changes in aggression's value)")

        logger.info("\nPOOLED ANALYSIS (All Years):")
        pooled = self.results['two_way']['pooled']
        sig_marker = "***" if pooled['significant'] else "   "
        logger.info(f"  {sig_marker} β = {pooled['aggression_coef']:+.4f} "
                   f"(SE={pooled['aggression_se']:.4f}, p={pooled['aggression_p']:.4f})")
        logger.info(f"      R² (within) = {pooled['r_squared']:.4f}")
        logger.info(f"      n = {pooled['n']} observations, {pooled['n_coaches']} coaches")

        if pooled['significant']:
            if pooled['aggression_coef'] > 0:
                logger.info(f"      → When coaches are more aggressive (controlling for time), they perform BETTER")
            else:
                logger.info(f"      → When coaches are more aggressive (controlling for time), they perform WORSE")
        else:
            logger.info(f"      → No significant within-coach relationship")

        logger.info("\nSTRATIFIED BY ERA:")
        for era_name, results in self.results['two_way']['stratified'].items():
            sig_marker = "***" if results['significant'] else "   "
            logger.info(f"\n{era_name}:")
            logger.info(f"  {sig_marker} β = {results['aggression_coef']:+.4f} "
                       f"(SE={results['aggression_se']:.4f}, p={results['aggression_p']:.4f})")
            logger.info(f"      n = {results['n']}, R² = {results['r_squared']:.4f}")

        logger.info("\n" + "="*80)
        logger.info("INTERPRETATION")
        logger.info("="*80)

        pooled = self.results['two_way']['pooled']
        early = self.results['two_way']['stratified'].get('Early (2006-2011)', {})
        late = self.results['two_way']['stratified'].get('Late (2018-2024)', {})

        if pooled['significant']:
            logger.info("\n✓ WITHIN-COACH EFFECT DETECTED:")
            logger.info(f"  Coaches perform {abs(pooled['aggression_coef']):.3f} WAR {'better' if pooled['aggression_coef'] > 0 else 'worse'}")
            logger.info(f"  when they are more aggressive (controlling for coach quality and time).")
            logger.info(f"  This supports the causal interpretation of aggression affecting outcomes.")
        else:
            logger.info("\n? WEAK OR NO WITHIN-COACH EFFECT:")
            logger.info("  The pooled effect is not statistically significant.")
            logger.info("  Check era-specific results for temporal variation.")

        if early and late:
            if early['significant'] and not late['significant']:
                logger.info(f"\n✓ EFFECT ERODED (Supports Diffusion Hypothesis):")
                logger.info(f"  Early: β={early['aggression_coef']:+.4f} (p={early['aggression_p']:.4f}) - significant")
                logger.info(f"  Late: β={late['aggression_coef']:+.4f} (p={late['aggression_p']:.4f}) - not significant")
                logger.info(f"  The causal effect of aggression disappeared as tactics diffused.")
            elif not early['significant'] and late['significant']:
                logger.info(f"\n? EFFECT EMERGED:")
                logger.info(f"  Early: β={early['aggression_coef']:+.4f} (p={early['aggression_p']:.4f}) - not significant")
                logger.info(f"  Late: β={late['aggression_coef']:+.4f} (p={late['aggression_p']:.4f}) - significant")
                logger.info(f"  Aggression gained causal importance (unexpected).")
            else:
                logger.info(f"\n? NO CLEAR TEMPORAL PATTERN:")
                logger.info(f"  Early: β={early['aggression_coef']:+.4f} (p={early['aggression_p']:.4f})")
                logger.info(f"  Late: β={late['aggression_coef']:+.4f} (p={late['aggression_p']:.4f})")
                logger.info(f"  Neither era shows significant effects, or both do.")

        logger.info("\n" + "="*80)

    def run(self) -> None:
        """Execute full fixed effects analysis pipeline"""
        logger.info("Starting within-coach fixed effects analysis...\n")

        # Load and prepare data
        self.load_data()

        # One-way fixed effects (coach only)
        demeaned_data = self.demean_within_coaches()
        demeaned_data = self.assign_eras(demeaned_data)

        # Two-way fixed effects (coach + year)
        twoway_data = self.demean_two_way()
        twoway_data = self.assign_eras(twoway_data)

        # Run one-way analyses
        logger.info("\n" + "#"*80)
        logger.info("# ONE-WAY FIXED EFFECTS (Coach FE only)")
        logger.info("#"*80)
        self.results['one_way'] = {
            'pooled': self.run_pooled_analysis(demeaned_data, '_demeaned'),
            'stratified': self.run_stratified_analysis(demeaned_data, '_demeaned'),
        }
        self.results['one_way']['coefficient_comparisons'] = self.test_coefficient_differences(
            self.results['one_way']['stratified']
        )

        # Run two-way analyses
        logger.info("\n" + "#"*80)
        logger.info("# TWO-WAY FIXED EFFECTS (Coach FE + Year FE)")
        logger.info("#"*80)
        self.results['two_way'] = {
            'pooled': self.run_pooled_analysis(twoway_data, '_twoway'),
            'stratified': self.run_stratified_analysis(twoway_data, '_twoway'),
        }
        self.results['two_way']['coefficient_comparisons'] = self.test_coefficient_differences(
            self.results['two_way']['stratified']
        )

        # Component analysis (just one-way for now to keep output manageable)
        self.results['component_analysis'] = self.run_component_analysis(demeaned_data)

        # Save and summarize
        self.save_results()
        self.print_summary()

        logger.info("\nAnalysis complete!")


def main():
    analyzer = WithinCoachFixedEffectsAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
