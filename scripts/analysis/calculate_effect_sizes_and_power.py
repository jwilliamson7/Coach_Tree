#!/usr/bin/env python3
"""
Effect Size and Power Analysis for Within-Coach Fixed Effects

This script enhances the fixed effects analysis with:
1. Effect size metrics (Cohen's d, standardized betas)
2. Confidence intervals for all estimates
3. WAR distribution statistics for contextualization
4. Post-hoc power analysis for era-stratified results
5. Minimum detectable effect sizes given sample sizes

Usage:
    python calculate_effect_sizes_and_power.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from scipy import stats
from sklearn.linear_model import LinearRegression
from typing import Dict, Tuple
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import coach_war_trajectories_path
from utils.parsimony import cluster_robust_ols
warnings.filterwarnings('ignore')


class EffectSizePowerAnalyzer:
    """Calculate effect sizes and power for within-coach fixed effects"""

    def __init__(self, aggression_file: str, war_file: str):
        """Initialize with data file paths"""
        self.aggression_file = Path(aggression_file)
        self.war_file = Path(war_file)
        self.data = None
        self.results = {}

        # Era definitions
        self.eras = {
            'Early (2006-2011)': (2006, 2011),
            'Middle (2012-2017)': (2012, 2017),
            'Late (2018-2024)': (2018, 2024)
        }

    def load_and_merge_data(self) -> None:
        """Load and merge aggression and WAR data"""
        print("Loading data...")

        # Load aggression data
        agg_df = pd.read_csv(self.aggression_file)

        # Load WAR data
        war_df = pd.read_csv(self.war_file)

        # Rename columns for consistency
        if 'head_coach' in agg_df.columns:
            agg_df = agg_df.rename(columns={'head_coach': 'coach'})
        if 'Year' in war_df.columns:
            war_df = war_df.rename(columns={'Year': 'season'})
        if 'Coach' in war_df.columns:
            war_df = war_df.rename(columns={'Coach': 'coach'})
        # Use Annual_Games (games per season) instead of Annual_WAR (per-game)
        if 'Annual_Games' in war_df.columns:
            war_df = war_df.rename(columns={'Annual_Games': 'WAR'})

        # Merge
        self.data = pd.merge(
            agg_df,
            war_df[['coach', 'season', 'WAR']],
            on=['coach', 'season'],
            how='inner'
        )

        # Filter to 2006-2024
        self.data = self.data[(self.data['season'] >= 2006) & (self.data['season'] <= 2024)]

        # Remove missing
        self.data = self.data.dropna(subset=['composite_aggression', 'WAR', 'season'])

        # Filter to coaches with 2+ seasons -- the SAME within-coach sample as
        # analyze_within_coach_fixed_effects.py. Single-season coaches contribute
        # only zeros after two-way demeaning and otherwise inflate n while diluting
        # the panel, so both scripts must use the multi-season sample for the effect
        # sizes and power to describe the same estimand.
        coach_counts = self.data['coach'].value_counts()
        multi = coach_counts[coach_counts >= 2].index
        n_before = len(self.data)
        self.data = self.data[self.data['coach'].isin(multi)]
        print(f"Filtered to {len(multi)} coaches with 2+ seasons "
              f"({len(self.data)} obs; dropped {n_before - len(self.data)} single-season rows)")

        # Add era labels
        self.data['era'] = pd.cut(
            self.data['season'],
            bins=[2005, 2011, 2017, 2025],
            labels=['Early (2006-2011)', 'Middle (2012-2017)', 'Late (2018-2024)']
        )

        print(f"\nMerged dataset: {len(self.data)} coach-years")
        print(f"Seasons: {self.data['season'].min()}-{self.data['season'].max()}")
        print(f"Unique coaches: {self.data['coach'].nunique()}")

    def calculate_war_distribution_stats(self) -> Dict:
        """Calculate WAR distribution statistics for contextualization"""
        print("\n" + "="*80)
        print("WAR DISTRIBUTION STATISTICS")
        print("="*80)

        war_values = self.data['WAR'].values

        stats_dict = {
            'n': int(len(war_values)),
            'mean': float(np.mean(war_values)),
            'median': float(np.median(war_values)),
            'std': float(np.std(war_values, ddof=1)),
            'min': float(np.min(war_values)),
            'max': float(np.max(war_values)),
            'percentiles': {
                '10th': float(np.percentile(war_values, 10)),
                '25th': float(np.percentile(war_values, 25)),
                '50th': float(np.percentile(war_values, 50)),
                '75th': float(np.percentile(war_values, 75)),
                '90th': float(np.percentile(war_values, 90)),
            },
            'iqr': float(np.percentile(war_values, 75) - np.percentile(war_values, 25))
        }

        print(f"\nN = {stats_dict['n']}")
        print(f"Mean:   {stats_dict['mean']:.4f}")
        print(f"Median: {stats_dict['median']:.4f}")
        print(f"SD:     {stats_dict['std']:.4f}")
        print(f"Min:    {stats_dict['min']:.4f}")
        print(f"Max:    {stats_dict['max']:.4f}")
        print(f"\nPercentiles:")
        print(f"  10th: {stats_dict['percentiles']['10th']:.4f}")
        print(f"  25th: {stats_dict['percentiles']['25th']:.4f}")
        print(f"  50th: {stats_dict['percentiles']['50th']:.4f}")
        print(f"  75th: {stats_dict['percentiles']['75th']:.4f}")
        print(f"  90th: {stats_dict['percentiles']['90th']:.4f}")
        print(f"IQR:    {stats_dict['iqr']:.4f}")

        self.results['war_distribution'] = stats_dict
        return stats_dict

    def demean_two_way(self) -> pd.DataFrame:
        """Demean within coaches AND within years (two-way fixed effects)"""
        print("\nDemeaning variables (two-way: coach + year)...")

        data = self.data.copy()
        vars_to_demean = ['WAR', 'composite_aggression']

        for var in vars_to_demean:
            # Calculate overall mean
            overall_mean = data[var].mean()

            # Calculate coach and year means
            coach_means = data.groupby('coach')[var].transform('mean')
            year_means = data.groupby('season')[var].transform('mean')

            # Two-way demeaning: remove both coach and year effects
            data[f'{var}_twoway'] = (
                data[var] - coach_means - year_means + overall_mean
            )

        return data

    def calculate_effect_sizes(self, beta: float, se: float, n: int, n_clusters: int,
                              data: pd.DataFrame, aggression_var: str, war_var: str) -> Dict:
        """
        Calculate comprehensive effect size metrics

        Returns:
        - Cohen's d (standardized mean difference)
        - Standardized beta (correlation-like metric)
        - 95% confidence interval
        - Percentile impact (what percentile shift does beta represent)
        """
        # Standard deviations on the WITHIN-demeaned columns (the two-way-demeaned
        # gene and WAR actually used in the regression), NOT the raw columns. The
        # beta is a within-coach slope, so the matching standardized effect must use
        # the within (demeaned) SDs; using raw SDs mixed a within slope with a
        # between+within spread (the prior bug).
        sd_aggression = data[aggression_var].std()
        sd_war = data[war_var].std()

        # Cohen's d: within-coach effect size in within-SD units, d = beta * SD_X/SD_Y
        cohens_d = beta * (sd_aggression / sd_war)

        # 95% Confidence interval
        # Use t-distribution with n_clusters - k - 1 degrees of freedom
        t_critical = stats.t.ppf(0.975, n_clusters - 2)
        ci_lower = beta - t_critical * se
        ci_upper = beta + t_critical * se

        # Calculate effect in absolute terms (games)
        # Effect of 1 SD increase in aggression
        war_sd = self.results['war_distribution']['std']
        war_iqr = self.results['war_distribution']['iqr']
        effect_1sd_aggression = beta * sd_aggression  # games

        # What percentile is 0 + effect_1sd in the WAR distribution?
        percentile_impact = stats.norm.cdf(effect_1sd_aggression / war_sd) * 100

        # Proportion of IQR
        proportion_iqr = effect_1sd_aggression / war_iqr

        return {
            'cohens_d': float(cohens_d),
            'ci_95_lower': float(ci_lower),
            'ci_95_upper': float(ci_upper),
            'sd_war': float(sd_war),
            'sd_aggression': float(sd_aggression),
            'effect_1sd_aggression_games': float(effect_1sd_aggression),
            'proportion_iqr': float(proportion_iqr),
            'percentile_impact': float(percentile_impact)
        }

    def calculate_power(self, beta: float, se: float, n: int, n_clusters: int, alpha: float = 0.05) -> Dict:
        """
        Calculate post-hoc power for observed effect

        Power = P(reject H0 | H1 is true with effect = beta)

        Also calculates minimum detectable effect (MDE) at 80% power
        """
        # Degrees of freedom
        df = n_clusters - 2

        # Non-centrality parameter for observed effect
        ncp = beta / se

        # Critical t-value for two-tailed test
        t_critical = stats.t.ppf(1 - alpha/2, df)

        # Power = P(|t| > t_crit | ncp)
        # For two-tailed test, this is:
        power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)

        # Minimum detectable effect at 80% power
        # For 80% power, we need ncp such that power = 0.80
        # Approximate: MDE = t_crit * SE / sqrt(power_target)
        # More accurate: solve for ncp where power = 0.80

        # Use approximation: for 80% power with alpha=0.05 (two-tailed)
        # ncp_80 ≈ 2.8 (depends on df, but roughly this)
        z_power = stats.norm.ppf(0.80)  # ~0.84
        z_alpha = stats.norm.ppf(1 - alpha/2)  # ~1.96
        mde_80 = (z_alpha + z_power) * se

        # Alternative MDE using t-distribution
        t_alpha = stats.t.ppf(1 - alpha/2, df)
        t_beta = stats.t.ppf(0.80, df)  # Not quite right, but approximation
        mde_80_t = (t_alpha + z_power) * se

        return {
            'power': float(power),
            'mde_80_power': float(mde_80_t),
            'observed_effect': float(beta),
            'se': float(se),
            'n': int(n),
            'n_clusters': int(n_clusters),
            'df': int(df),
            'sufficient_power': bool(power >= 0.80)
        }

    def run_fixed_effects_with_metrics(self, data: pd.DataFrame,
                                       aggression_var: str,
                                       war_var: str,
                                       label: str) -> Dict:
        """Run fixed effects regression with full effect size and power metrics"""

        # Prepare data
        X = data[[aggression_var]].values
        y = data[war_var].values

        n = len(y)

        # Coach-clustered (sandwich) SEs via the shared implementation in
        # utils.parsimony -- single source of truth (df = n_clusters - k - 1,
        # finite-sample G/(G-1) adjustment). Inference is on the coach count, so
        # the effective n is n_clusters (~123), not the 606 coach-year rows.
        coaches = data['coach'].values
        cr = cluster_robust_ols(X, y, coaches, [aggression_var])
        n_clusters = cr['n_clusters']
        coef = cr['coefficients'][aggression_var]
        beta = coef['coefficient']
        se = coef['std_error']
        t_stat = coef['t_statistic']
        p_value = coef['p_value']

        # Calculate effect sizes
        effect_sizes = self.calculate_effect_sizes(beta, se, n, n_clusters, data, aggression_var, war_var)

        # Calculate power
        power_analysis = self.calculate_power(beta, se, n, n_clusters)

        # Combine results
        results = {
            'label': label,
            'n': int(n),
            'n_coaches': int(n_clusters),
            'beta': float(beta),
            'se': float(se),
            't': float(t_stat),
            'p': float(p_value),
            'effect_sizes': effect_sizes,
            'power_analysis': power_analysis
        }

        return results

    def run_pooled_analysis(self, demeaned_data: pd.DataFrame) -> Dict:
        """Run pooled two-way fixed effects with full metrics"""
        print("\n" + "="*80)
        print("POOLED TWO-WAY FIXED EFFECTS WITH EFFECT SIZES")
        print("="*80)

        result = self.run_fixed_effects_with_metrics(
            demeaned_data,
            'composite_aggression_twoway',
            'WAR_twoway',
            'Pooled (2006-2024)'
        )

        # Print comprehensive results
        print(f"\n{result['label']}")
        print(f"  N = {result['n']} observations, {result['n_coaches']} coaches")
        print(f"\nRegression Coefficient:")
        print(f"  Beta = {result['beta']:.4f}")
        print(f"  SE   = {result['se']:.4f}")
        print(f"  t    = {result['t']:.3f}")
        print(f"  p    = {result['p']:.4f}")
        print(f"  95% CI: [{result['effect_sizes']['ci_95_lower']:.4f}, {result['effect_sizes']['ci_95_upper']:.4f}]")

        print(f"\nEffect Sizes (within-coach SDs, WAR in games):")
        print(f"  Cohen's d (within):  {result['effect_sizes']['cohens_d']:.4f}")
        print(f"  Effect (1 within-SD aggr): {result['effect_sizes']['effect_1sd_aggression_games']:.3f} games")
        print(f"  Proportion of IQR:   {result['effect_sizes']['proportion_iqr']:.2%}")
        print(f"  Percentile shift:    {result['effect_sizes']['percentile_impact']:.1f}th percentile")

        print(f"\nPower Analysis:")
        print(f"  Observed power:    {result['power_analysis']['power']:.4f} ({result['power_analysis']['power']*100:.1f}%)")
        print(f"  Sufficient power:  {result['power_analysis']['sufficient_power']}")
        print(f"  MDE (80% power):   {result['power_analysis']['mde_80_power']:.4f}")

        return result

    def run_stratified_analysis(self, demeaned_data: pd.DataFrame) -> Dict:
        """Run era-stratified two-way fixed effects with full metrics"""
        print("\n" + "="*80)
        print("ERA-STRATIFIED TWO-WAY FIXED EFFECTS WITH EFFECT SIZES & POWER")
        print("="*80)

        results = {}

        for era_name in self.eras.keys():
            era_data = demeaned_data[demeaned_data['era'] == era_name].copy()

            if len(era_data) < 20:
                print(f"\n{era_name}: Skipped (insufficient data)")
                continue

            result = self.run_fixed_effects_with_metrics(
                era_data,
                'composite_aggression_twoway',
                'WAR_twoway',
                era_name
            )

            results[era_name] = result

            # Print results
            print(f"\n{result['label']}")
            print(f"  N = {result['n']} observations, {result['n_coaches']} coaches")
            print(f"\nRegression Coefficient:")
            print(f"  Beta = {result['beta']:.4f}")
            print(f"  SE   = {result['se']:.4f}")
            print(f"  t    = {result['t']:.3f}")
            print(f"  p    = {result['p']:.4f}")
            print(f"  95% CI: [{result['effect_sizes']['ci_95_lower']:.4f}, {result['effect_sizes']['ci_95_upper']:.4f}]")

            print(f"\nEffect Sizes:")
            print(f"  Cohen's d:           {result['effect_sizes']['cohens_d']:.4f}")
            print(f"  Effect (1 SD aggr): {result['effect_sizes']['effect_1sd_aggression_games']:.3f} games")
            print(f"  Proportion of IQR:   {result['effect_sizes']['proportion_iqr']:.2%}")

            print(f"\nPower Analysis:")
            print(f"  Observed power:    {result['power_analysis']['power']:.4f} ({result['power_analysis']['power']*100:.1f}%)")
            print(f"  Sufficient power:  {result['power_analysis']['sufficient_power']}")
            print(f"  MDE (80% power):   {result['power_analysis']['mde_80_power']:.4f}")

            # Interpretation
            if result['p'] >= 0.05:
                if result['power_analysis']['power'] < 0.80:
                    print(f"  ** Underpowered: Cannot rule out effects as large as {result['power_analysis']['mde_80_power']:.2f} WAR")
                else:
                    print(f"  ** Well-powered null: Effect likely < {result['power_analysis']['mde_80_power']:.2f} WAR")

        return results

    def compare_confidence_intervals(self, results: Dict) -> None:
        """Check if era-stratified CIs overlap"""
        print("\n" + "="*80)
        print("CONFIDENCE INTERVAL OVERLAP ANALYSIS")
        print("="*80)

        era_names = list(results.keys())

        for i, era1 in enumerate(era_names):
            for era2 in era_names[i+1:]:
                ci1_lower = results[era1]['effect_sizes']['ci_95_lower']
                ci1_upper = results[era1]['effect_sizes']['ci_95_upper']
                ci2_lower = results[era2]['effect_sizes']['ci_95_lower']
                ci2_upper = results[era2]['effect_sizes']['ci_95_upper']

                # Check overlap
                overlap = not (ci1_upper < ci2_lower or ci2_upper < ci1_lower)

                print(f"\n{era1} vs {era2}:")
                print(f"  {era1}: [{ci1_lower:.4f}, {ci1_upper:.4f}]")
                print(f"  {era2}: [{ci2_lower:.4f}, {ci2_upper:.4f}]")
                print(f"  Overlap: {overlap}")

                if overlap:
                    print(f"  ** CIs overlap: Cannot conclude effects differ significantly")
                else:
                    print(f"  ** CIs do NOT overlap: Strong evidence effects differ")

    def save_results(self, output_dir: str = 'outputs/analysis') -> None:
        """Save all results to JSON"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / 'effect_sizes_and_power_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")

    def run_all_analyses(self) -> None:
        """Run complete effect size and power analysis pipeline"""
        self.load_and_merge_data()
        self.calculate_war_distribution_stats()

        # Demean data
        demeaned_data = self.demean_two_way()

        # Run pooled analysis
        pooled_results = self.run_pooled_analysis(demeaned_data)
        self.results['pooled'] = pooled_results

        # Run stratified analysis
        stratified_results = self.run_stratified_analysis(demeaned_data)
        self.results['stratified'] = stratified_results

        # Compare CIs
        self.compare_confidence_intervals(stratified_results)

        # Save results
        self.save_results()


def main():
    """Main execution"""
    aggression_file = 'data/processed/coaching_genes/aggression_gene_by_year.csv'
    war_file = coach_war_trajectories_path()

    analyzer = EffectSizePowerAnalyzer(aggression_file, war_file)
    analyzer.run_all_analyses()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
