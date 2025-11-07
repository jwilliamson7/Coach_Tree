#!/usr/bin/env python3
"""
Temporal Robustness Checks for Aggression-Performance Relationship

This script tests whether the erosion of aggression's performance benefit is robust to:
1. Continuous year specification (Aggression × Year interaction)
2. Structural break tests (Chow test at various candidate breakpoints)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class TemporalRobustnessAnalyzer:
    """Analyzes robustness of temporal erosion findings"""

    def __init__(self, aggression_file: str, war_file: str):
        """Initialize with data file paths"""
        self.aggression_file = Path(aggression_file)
        self.war_file = Path(war_file)
        self.data = None
        self.results = {}

    def load_and_merge_data(self) -> None:
        """Load aggression and WAR data, merge on coach and season"""
        print("Loading data...")

        # Load aggression data
        agg_df = pd.read_csv(self.aggression_file)
        print(f"Loaded {len(agg_df)} aggression records")

        # Load WAR data
        war_df = pd.read_csv(self.war_file)
        print(f"Loaded {len(war_df)} WAR records")

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

        # Merge on coach and season
        self.data = pd.merge(
            agg_df,
            war_df[['coach', 'season', 'WAR']],
            on=['coach', 'season'],
            how='inner'
        )

        # Filter to 2006-2024
        self.data = self.data[(self.data['season'] >= 2006) & (self.data['season'] <= 2024)]

        # Remove missing values
        self.data = self.data.dropna(subset=['composite_aggression', 'WAR', 'season'])

        print(f"\nMerged dataset: {len(self.data)} coach-years")
        print(f"Seasons: {self.data['season'].min()}-{self.data['season'].max()}")
        print(f"Unique coaches: {self.data['coach'].nunique()}")

    def continuous_year_analysis(self) -> Dict:
        """
        Test if aggression effect changes linearly over time using interaction.

        Model: WAR = b0 + b1(Aggression) + b2(Year) + b3(Aggression x Year) + e

        If b3 < 0 and significant -> erosion
        """
        print("\n" + "="*80)
        print("CONTINUOUS YEAR ANALYSIS")
        print("="*80)

        # Center year at 2015 (midpoint) for interpretability
        self.data['year_centered'] = self.data['season'] - 2015

        # Create interaction term
        self.data['agg_x_year'] = self.data['composite_aggression'] * self.data['year_centered']

        # Prepare data
        X = self.data[['composite_aggression', 'year_centered', 'agg_x_year']].values
        y = self.data['WAR'].values

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Get predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred

        # Calculate cluster-robust standard errors (clustered by coach)
        n, k = X.shape
        X_with_const = np.column_stack([np.ones(n), X])
        k_full = k + 1

        # Coefficient estimates
        coef_full = np.concatenate([[model.intercept_], model.coef_])

        # Cluster-robust standard errors (clustered by coach)
        coaches = self.data['coach'].values
        unique_coaches = np.unique(coaches)
        n_clusters = len(unique_coaches)

        bread = np.linalg.pinv(X_with_const.T @ X_with_const)

        # Build meat matrix using cluster sandwich
        meat = np.zeros((k_full, k_full))
        for coach in unique_coaches:
            mask = coaches == coach
            X_c = X_with_const[mask]
            e_c = residuals[mask]
            meat += X_c.T @ (e_c[:, None] @ e_c[None, :]) @ X_c

        # Finite sample adjustment
        meat *= n_clusters / (n_clusters - 1)

        vcov = bread @ meat @ bread
        se_robust = np.sqrt(np.diag(vcov))

        # Calculate statistics (use n_clusters - k_full for clustered SEs)
        t_stats = coef_full / se_robust
        df = n_clusters - k_full
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

        # Model fit
        r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)

        # Extract key coefficients
        results = {
            'model': 'WAR = b0 + b1(Aggression) + b2(Year) + b3(Aggression x Year)',
            'n_obs': int(n),
            'n_clusters': int(n_clusters),
            'r_squared': float(r_squared),
            'coefficients': {
                'intercept': {
                    'beta': float(coef_full[0]),
                    'se': float(se_robust[0]),
                    't': float(t_stats[0]),
                    'p': float(p_values[0])
                },
                'aggression': {
                    'beta': float(coef_full[1]),
                    'se': float(se_robust[1]),
                    't': float(t_stats[1]),
                    'p': float(p_values[1]),
                    'interpretation': 'Main effect of aggression at year 2015'
                },
                'year': {
                    'beta': float(coef_full[2]),
                    'se': float(se_robust[2]),
                    't': float(t_stats[2]),
                    'p': float(p_values[2]),
                    'interpretation': 'Temporal trend in WAR'
                },
                'aggression_x_year': {
                    'beta': float(coef_full[3]),
                    'se': float(se_robust[3]),
                    't': float(t_stats[3]),
                    'p': float(p_values[3]),
                    'interpretation': 'Change in aggression effect per year'
                }
            }
        }

        # Calculate implied effects at specific years
        years_to_test = [2006, 2011, 2017, 2024]
        implied_effects = {}

        for year in years_to_test:
            year_centered = year - 2015
            # Effect of aggression at this year
            effect = coef_full[1] + coef_full[3] * year_centered

            # Standard error (delta method)
            # Var(b1 + b3*t) = Var(b1) + t^2*Var(b3) + 2t*Cov(b1,b3)
            var_effect = vcov[1,1] + (year_centered**2)*vcov[3,3] + 2*year_centered*vcov[1,3]
            se_effect = np.sqrt(var_effect)

            t_effect = effect / se_effect
            p_effect = 2 * (1 - stats.t.cdf(np.abs(t_effect), df))

            implied_effects[str(year)] = {
                'effect_size': float(effect),
                'se': float(se_effect),
                't': float(t_effect),
                'p': float(p_effect)
            }

        results['implied_effects_by_year'] = implied_effects

        # Print results
        print(f"\nModel: {results['model']}")
        print(f"N = {n} observations, {n_clusters} coaches (clustered SEs)")
        print(f"R2 = {r_squared:.4f}")
        print("\nCoefficients (with cluster-robust standard errors):")
        print(f"  Intercept:           b0 = {coef_full[0]:7.4f}, SE = {se_robust[0]:.4f}, p = {p_values[0]:.4f}")
        print(f"  Aggression:          b1 = {coef_full[1]:7.4f}, SE = {se_robust[1]:.4f}, p = {p_values[1]:.4f}")
        print(f"  Year (centered):     b2 = {coef_full[2]:7.4f}, SE = {se_robust[2]:.4f}, p = {p_values[2]:.4f}")
        print(f"  Aggression x Year:   b3 = {coef_full[3]:7.4f}, SE = {se_robust[3]:.4f}, p = {p_values[3]:.4f}")

        if p_values[3] < 0.05:
            direction = "DECREASING" if coef_full[3] < 0 else "INCREASING"
            print(f"\n*** INTERACTION IS SIGNIFICANT: Aggression effect is {direction} over time (p={p_values[3]:.4f}) ***")
        else:
            print(f"\nInteraction is not significant (p={p_values[3]:.4f})")

        print("\nImplied Aggression Effects by Year:")
        for year in years_to_test:
            e = implied_effects[str(year)]
            sig_marker = "**" if e['p'] < 0.01 else "*" if e['p'] < 0.05 else ""
            print(f"  {year}: b = {e['effect_size']:7.4f}, SE = {e['se']:.4f}, p = {e['p']:.4f} {sig_marker}")

        self.results['continuous_year'] = results
        return results

    def chow_test(self, breakpoint: int) -> Dict:
        """
        Perform Chow test for structural break at given year.

        Tests null hypothesis: b1_early = b1_late (coefficients equal)
        against alternative: b1_early != b1_late
        """
        # Split data
        early = self.data[self.data['season'] <= breakpoint]
        late = self.data[self.data['season'] > breakpoint]

        if len(early) < 20 or len(late) < 20:
            return None  # Skip if too few observations

        # Pooled model (restricted)
        X_pooled = self.data[['composite_aggression']].values
        y_pooled = self.data['WAR'].values
        model_pooled = LinearRegression()
        model_pooled.fit(X_pooled, y_pooled)
        rss_pooled = np.sum((y_pooled - model_pooled.predict(X_pooled))**2)

        # Early model
        X_early = early[['composite_aggression']].values
        y_early = early['WAR'].values
        model_early = LinearRegression()
        model_early.fit(X_early, y_early)
        rss_early = np.sum((y_early - model_early.predict(X_early))**2)

        # Late model
        X_late = late[['composite_aggression']].values
        y_late = late['WAR'].values
        model_late = LinearRegression()
        model_late.fit(X_late, y_late)
        rss_late = np.sum((y_late - model_late.predict(X_late))**2)

        # Chow test statistic
        # F = [(RSS_pooled - (RSS_1 + RSS_2)) / k] / [(RSS_1 + RSS_2) / (n - 2k)]
        # where k = number of parameters (2: intercept + slope)
        k = 2
        n = len(self.data)

        rss_unrestricted = rss_early + rss_late
        numerator = (rss_pooled - rss_unrestricted) / k
        denominator = rss_unrestricted / (n - 2*k)

        if denominator == 0:
            return None

        f_stat = numerator / denominator
        p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)

        # Store results
        result = {
            'breakpoint': int(breakpoint),
            'n_early': int(len(early)),
            'n_late': int(len(late)),
            'beta_early': float(model_early.coef_[0]),
            'beta_late': float(model_late.coef_[0]),
            'change': float(model_late.coef_[0] - model_early.coef_[0]),
            'pct_change': float(100 * (model_late.coef_[0] - model_early.coef_[0]) / model_early.coef_[0]) if model_early.coef_[0] != 0 else None,
            'rss_pooled': float(rss_pooled),
            'rss_early': float(rss_early),
            'rss_late': float(rss_late),
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'reject_null': bool(p_value < 0.05)
        }

        return result

    def structural_break_tests(self, candidate_years: List[int] = None) -> Dict:
        """
        Test for structural breaks at multiple candidate years using Chow test.
        """
        print("\n" + "="*80)
        print("STRUCTURAL BREAK TESTS (Chow Test)")
        print("="*80)

        if candidate_years is None:
            # Test every year from 2009 to 2019
            candidate_years = list(range(2009, 2020))

        results = {}

        for year in candidate_years:
            print(f"\nTesting breakpoint at {year}...")
            result = self.chow_test(year)

            if result is None:
                print(f"  Skipped (insufficient data)")
                continue

            results[str(year)] = result

            print(f"  Early period (<={year}): n={result['n_early']}, b={result['beta_early']:.4f}")
            print(f"  Late period (>{year}):  n={result['n_late']}, b={result['beta_late']:.4f}")
            print(f"  Change: {result['change']:.4f} ({result['pct_change']:.1f}%)")
            print(f"  F-statistic: {result['f_statistic']:.4f}")
            print(f"  P-value: {result['p_value']:.6f}")

            if result['reject_null']:
                print(f"  *** SIGNIFICANT STRUCTURAL BREAK at {year} (p<0.05) ***")

        # Find most significant break
        if results:
            best_break = min(results.items(), key=lambda x: x[1]['p_value'])
            print("\n" + "="*80)
            print(f"MOST SIGNIFICANT BREAK: {best_break[0]} (p={best_break[1]['p_value']:.6f})")
            print("="*80)

            results['best_breakpoint'] = {
                'year': int(best_break[0]),
                'p_value': best_break[1]['p_value'],
                'f_statistic': best_break[1]['f_statistic'],
                'beta_early': best_break[1]['beta_early'],
                'beta_late': best_break[1]['beta_late'],
                'change_pct': best_break[1]['pct_change']
            }

        self.results['structural_breaks'] = results
        return results

    def save_results(self, output_dir: str = 'outputs/analysis') -> None:
        """Save results to JSON file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        output_file = output_path / 'temporal_robustness_results.json'

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")

    def run_all_analyses(self) -> None:
        """Run all temporal robustness analyses"""
        self.load_and_merge_data()
        self.continuous_year_analysis()
        self.structural_break_tests()
        self.save_results()


def main():
    """Main execution"""
    # File paths
    aggression_file = 'data/processed/coaching_genes/aggression_gene_by_year.csv'
    war_file = 'data/processed/Coaching/coach_war_trajectories.csv'

    # Initialize analyzer
    analyzer = TemporalRobustnessAnalyzer(aggression_file, war_file)

    # Run all analyses
    analyzer.run_all_analyses()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
