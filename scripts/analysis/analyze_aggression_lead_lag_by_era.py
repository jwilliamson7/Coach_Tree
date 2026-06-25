#!/usr/bin/env python3
"""
Analyze Lead-Lag Relationship Between Aggression and WAR by Era

This script tests whether aggression in year t predicts WAR in year t+1,
controlling for year t WAR. We stratify by era to test whether the predictive
power of aggression eroded over time alongside the cross-sectional correlation.

Statistical approach:
1. Create lagged variables (aggression_t-1, WAR_t-1)
2. Run regression: WAR_t ~ aggression_t-1 + WAR_t-1 separately for each era
3. Test whether coefficients differ significantly across eras
4. Interpret: Does aggression predict future performance? Did this erode?

Usage:
    python analyze_aggression_lead_lag_by_era.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

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


class LeadLagAnalyzer:
    """Analyze lead-lag relationships between aggression and WAR by era"""

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

    def create_lagged_variables(self) -> None:
        """Create lagged aggression and WAR variables"""
        logger.info("Creating lagged variables...")

        # Sort by coach and season
        self.data = self.data.sort_values(['coach', 'season'])

        # Create lagged variables (shift by 1 within each coach)
        self.data['aggression_lag1'] = self.data.groupby('coach')['composite_aggression'].shift(1)
        self.data['war_lag1'] = self.data.groupby('coach')['WAR'].shift(1)
        # Lag the season too, to verify the pair is actually consecutive (see below)
        self.data['season_lag1'] = self.data.groupby('coach')['season'].shift(1)

        # Also lag the individual components
        self.data['fourth_down_lag1'] = self.data.groupby('coach')['fourth_down_aggression'].shift(1)
        self.data['pass_heavy_lag1'] = self.data.groupby('coach')['pass_heavy_aggression'].shift(1)
        self.data['deep_pass_lag1'] = self.data.groupby('coach')['deep_pass_aggression'].shift(1)
        self.data['two_point_lag1'] = self.data.groupby('coach')['two_point_aggression'].shift(1)

        # Remove rows with missing lags (first year for each coach)
        n_before = len(self.data)
        self.data = self.data.dropna(subset=['aggression_lag1', 'war_lag1'])
        n_after = len(self.data)

        # Gap guard: keep ONLY contiguous-season pairs (t-1 -> t). Without this, a
        # coach's non-consecutive stints (e.g. McDaniels 2010 then 2022) get paired
        # as if adjacent, which is not a real one-year lag.
        self.data = self.data[self.data['season'] - self.data['season_lag1'] == 1].copy()
        n_contig = len(self.data)

        logger.info(f"Dropped {n_before - n_after:,} observations with missing lags")
        logger.info(f"Dropped {n_after - n_contig:,} non-contiguous-season lag pairs (gap guard)")
        logger.info(f"Final sample: {n_contig:,} observations ({self.data['coach'].nunique()} coaches)")

    def assign_eras(self) -> None:
        """Assign each observation to an era"""
        def assign_era(year):
            for era_name, (start, end) in self.eras.items():
                if start <= year <= end:
                    return era_name
            return None

        self.data['era'] = self.data['season'].apply(assign_era)

        # Log era distribution
        logger.info("\nEra distribution:")
        for era in self.eras.keys():
            n = (self.data['era'] == era).sum()
            logger.info(f"  {era}: {n:,} observations")

    def run_stratified_regressions(self) -> dict:
        """Run separate regressions for each era"""
        logger.info("\nRunning stratified regressions by era...")

        results = {}

        for era_name in self.eras.keys():
            era_data = self.data[self.data['era'] == era_name].copy()

            if len(era_data) < 10:
                logger.warning(f"Insufficient data for {era_name}: {len(era_data)} observations")
                continue

            logger.info(f"\n{era_name}: {len(era_data)} observations")

            # Main regression: WAR_t ~ aggression_t-1 + WAR_t-1, with coach-
            # clustered (sandwich) SEs -- repeated coach-years are not independent.
            try:
                from utils.parsimony import cluster_robust_ols
                X = era_data[['aggression_lag1', 'war_lag1']].values
                y = era_data['WAR'].values
                n = len(y)
                k = X.shape[1]

                res = cluster_robust_ols(X, y, era_data['coach'].values,
                                         ['aggression_lag1', 'war_lag1'])
                ac = res['coefficients']['aggression_lag1']
                wc = res['coefficients']['war_lag1']
                r2 = res['r_squared']
                adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n - k - 1 > 0 else float('nan')

                results[era_name] = {
                    'n': int(n),
                    'n_coaches': int(res['n_clusters']),
                    'aggression_coef': float(ac['coefficient']),
                    'aggression_se': float(ac['std_error']),
                    'aggression_t': float(ac['t_statistic']),
                    'aggression_p': float(ac['p_value']),
                    'war_lag_coef': float(wc['coefficient']),
                    'war_lag_se': float(wc['std_error']),
                    'war_lag_t': float(wc['t_statistic']),
                    'war_lag_p': float(wc['p_value']),
                    'intercept': float(res['intercept']),
                    'r_squared': float(r2),
                    'adj_r_squared': float(adj_r2),
                    'se_type': 'cluster_robust_by_coach',
                    'significant': bool(ac['p_value'] < 0.05)
                }

                logger.info(f"  Aggression_t-1 coefficient: {ac['coefficient']:.4f} "
                           f"(clustered SE={ac['std_error']:.4f}, "
                           f"t={ac['t_statistic']:.3f}, "
                           f"p={ac['p_value']:.4f})")
                logger.info(f"  WAR_t-1 coefficient: {wc['coefficient']:.4f} "
                           f"(p={wc['p_value']:.4f})")
                logger.info(f"  R2={r2:.4f}, Adj R2={adj_r2:.4f}")

            except Exception as e:
                logger.error(f"Error fitting model for {era_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

        return results

    def test_coefficient_differences(self, era_results: dict) -> dict:
        """Test whether aggression coefficients differ significantly across eras"""
        logger.info("\nTesting coefficient differences across eras...")

        comparisons = {}
        era_names = list(era_results.keys())

        for i in range(len(era_names)):
            for j in range(i + 1, len(era_names)):
                era1 = era_names[i]
                era2 = era_names[j]

                # Extract coefficients and standard errors
                b1 = era_results[era1]['aggression_coef']
                se1 = era_results[era1]['aggression_se']

                b2 = era_results[era2]['aggression_coef']
                se2 = era_results[era2]['aggression_se']

                # Test difference: (b1 - b2) / sqrt(se1^2 + se2^2)
                diff = b1 - b2
                se_diff = np.sqrt(se1**2 + se2**2)
                z_stat = diff / se_diff
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

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

    def run_component_analysis(self) -> dict:
        """Run lead-lag analysis for individual aggression components"""
        logger.info("\n" + "="*80)
        logger.info("COMPONENT-LEVEL ANALYSIS")
        logger.info("="*80)

        components = {
            'fourth_down': 'fourth_down_lag1',
            'pass_heavy': 'pass_heavy_lag1',
            'deep_pass': 'deep_pass_lag1',
            'two_point': 'two_point_lag1'
        }

        component_results = {}

        for comp_name, lag_var in components.items():
            logger.info(f"\n{comp_name.upper()} AGGRESSION:")
            component_results[comp_name] = {}

            for era_name in self.eras.keys():
                era_data = self.data[self.data['era'] == era_name].copy()

                if len(era_data) < 10:
                    continue

                try:
                    # Prepare data
                    X = era_data[[lag_var, 'war_lag1']].values
                    y = era_data['WAR'].values

                    # Fit model
                    model = LinearRegression()
                    model.fit(X, y)

                    # Calculate standard errors
                    n = len(y)
                    k = X.shape[1]
                    residuals = y - model.predict(X)
                    mse = np.sum(residuals**2) / (n - k - 1)
                    var_coef = mse * np.linalg.inv(X.T @ X).diagonal()
                    se = np.sqrt(var_coef)
                    t_stat = model.coef_[0] / se[0]
                    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - k - 1))

                    component_results[comp_name][era_name] = {
                        'coef': float(model.coef_[0]),
                        'se': float(se[0]),
                        'p': float(p_value),
                        'significant': bool(p_value < 0.05)
                    }

                    logger.info(f"  {era_name}: β={model.coef_[0]:.4f} "
                               f"(p={p_value:.4f})")

                except Exception as e:
                    logger.error(f"Error for {comp_name} in {era_name}: {e}")

        return component_results

    def save_results(self) -> None:
        """Save all results to JSON"""
        output_file = self.output_dir / "lead_lag_by_era_results.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2, cls=NumpyEncoder)

        logger.info(f"\nResults saved to: {output_file}")

    def print_summary(self) -> None:
        """Print human-readable summary"""
        logger.info("\n" + "="*80)
        logger.info("LEAD-LAG ANALYSIS SUMMARY")
        logger.info("="*80)

        logger.info("\nKey Question: Does aggression in year t predict WAR in year t+1?")
        logger.info("(controlling for WAR in year t)")

        logger.info("\nCOMPOSITE AGGRESSION BY ERA:")
        for era_name, results in self.results['stratified_regressions'].items():
            sig_marker = "***" if results['significant'] else "   "
            logger.info(f"\n{era_name}:")
            logger.info(f"  {sig_marker} β = {results['aggression_coef']:+.4f} "
                       f"(SE={results['aggression_se']:.4f}, p={results['aggression_p']:.4f})")
            logger.info(f"      R² = {results['r_squared']:.4f}, n = {results['n']}")

            if results['significant']:
                logger.info(f"      → Aggression predicts future performance!")
            else:
                logger.info(f"      → Aggression does NOT predict future performance")

        logger.info("\nCOEFFICIENT COMPARISONS:")
        for comparison, results in self.results['coefficient_comparisons'].items():
            sig_marker = "***" if results['significant'] else "   "
            logger.info(f"\n{comparison}:")
            logger.info(f"  {sig_marker} Difference = {results['coef_diff']:+.4f} "
                       f"(z={results['z_statistic']:.3f}, p={results['p_value']:.4f})")

        logger.info("\n" + "="*80)
        logger.info("INTERPRETATION")
        logger.info("="*80)

        early = self.results['stratified_regressions'].get('Early (2006-2011)', {})
        late = self.results['stratified_regressions'].get('Late (2018-2024)', {})

        if early and late:
            if early['significant'] and not late['significant']:
                logger.info("\n✓ EROSION CONFIRMED:")
                logger.info(f"  Early era: Aggression predicted future WAR (β={early['aggression_coef']:+.4f}, p={early['aggression_p']:.4f})")
                logger.info(f"  Late era: Aggression no longer predictive (β={late['aggression_coef']:+.4f}, p={late['aggression_p']:.4f})")
                logger.info("\n  This strengthens the causal interpretation: aggressive coaching")
                logger.info("  once CAUSED better future performance, but this advantage eroded.")
            elif not early['significant'] and not late['significant']:
                logger.info("\n✗ NO PREDICTIVE POWER:")
                logger.info("  Aggression never predicted future performance in any era.")
                logger.info("  The cross-sectional correlations may reflect reverse causation")
                logger.info("  or selection effects rather than aggression causing success.")
            else:
                logger.info("\n? MIXED RESULTS:")
                logger.info("  See detailed coefficients above for interpretation.")

        logger.info("\n" + "="*80)

    def run(self) -> None:
        """Execute full lead-lag analysis pipeline"""
        logger.info("Starting lead-lag analysis by era...\n")

        # Load and prepare data
        self.load_data()
        self.create_lagged_variables()
        self.assign_eras()

        # Run analyses
        self.results['stratified_regressions'] = self.run_stratified_regressions()
        self.results['coefficient_comparisons'] = self.test_coefficient_differences(
            self.results['stratified_regressions']
        )
        self.results['component_analysis'] = self.run_component_analysis()

        # Save and summarize
        self.save_results()
        self.print_summary()

        logger.info("\nAnalysis complete!")


def main():
    analyzer = LeadLagAnalyzer()
    analyzer.run()


if __name__ == "__main__":
    main()
