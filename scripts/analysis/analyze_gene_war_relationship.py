#!/usr/bin/env python3
"""
Analyze Relationship Between Coaching Genes and WAR

Tests whether coaching genes (shotgun, tempo, defensive scheme) predict
coaching performance as measured by Wins Above Replacement (WAR).

Extends the aggression-WAR analysis to all gene dimensions.

Usage:
    python scripts/analysis/analyze_gene_war_relationship.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import coach_war_trajectories_path, merge_gene_war
from utils.parsimony import cluster_robust_ols, cluster_bootstrap_ci, cluster_bootstrap_corr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Gene definitions: (csv_path, coach_col, year_col, gene_columns_to_test)
GENE_CONFIGS = {
    'shotgun': {
        'path': 'data/processed/coaching_genes/shotgun_gene.csv',
        'coach_col': 'head_coach',
        'year_col': 'season',
        'measures': {
            'shotgun_gene_zscore': 'Shotgun Formation',
        },
        'min_year': 2006,
    },
    'tempo': {
        'path': 'data/processed/coaching_genes/tempo_gene.csv',
        'coach_col': 'head_coach',
        'year_col': 'season',
        'measures': {
            'composite_tempo_zscore': 'Composite Tempo',
            'no_huddle_gene_zscore': 'No-Huddle',
            'pace_gene_zscore': 'Pace',
        },
        'min_year': 2006,
    },
    'defensive_scheme': {
        'path': 'data/processed/coaching_genes/defensive_scheme_gene.csv',
        'coach_col': 'head_coach',
        'year_col': 'season',
        'measures': {
            'composite_scheme_zscore': 'Defensive Scheme',
            'box_stacking_gene_zscore': 'Box Stacking',
            'pass_rush_gene_zscore': 'Pass Rush',
        },
        'min_year': 2016,
    },
    'aggression': {
        'path': 'data/processed/coaching_genes/aggression_gene_by_year.csv',
        'coach_col': 'head_coach',
        'year_col': 'season',
        'measures': {
            'composite_aggression': 'Composite Aggression',
        },
        'min_year': 2006,
    },
}


def load_war_data():
    """Load WAR data."""
    war_file = coach_war_trajectories_path()
    if not war_file.exists():
        raise FileNotFoundError(f"WAR data not found: {war_file}")
    war = pd.read_csv(war_file)
    war.columns = war.columns.str.lower()
    logger.info(f"Loaded {len(war)} WAR records")
    return war


def load_and_merge_gene(gene_key, war_data):
    """Load a gene CSV and merge with WAR data."""
    config = GENE_CONFIGS[gene_key]
    path = Path(config['path'])
    if not path.exists():
        logger.warning(f"Gene file not found: {path}")
        return None

    gene_df = pd.read_csv(path)
    gene_df = gene_df.rename(columns={
        config['coach_col']: 'coach',
        config['year_col']: 'year',
    })

    # WS8: join on canonicalized coach name (+ year) with attrition logging, so
    # name-format mismatches are recovered and true drops are surfaced.
    merged = merge_gene_war(gene_df, war_data, 'coach', 'coach',
                            year_cols=('year', 'year'), how='inner', logger=logger)
    logger.info(f"{gene_key}: merged {len(merged)} coach-year records "
                f"({merged['coach'].nunique()} coaches)")
    return merged


def analyze_correlations(merged, measures, gene_key):
    """Run correlation analysis for each measure vs WAR."""
    results = {}

    for col, label in measures.items():
        clean = merged[[col, 'annual_war', 'coach']].dropna()
        if len(clean) < 10:
            logger.warning(f"  {label}: insufficient data (n={len(clean)})")
            continue

        x = clean[col]
        y = clean['annual_war'] * 16  # Convert to games

        r, p = stats.pearsonr(x, y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Coach-clustered bootstrap p/CI: coach-years are repeated measures, so the
        # naive Pearson p overstates significance. This is the primary inference.
        boot = cluster_bootstrap_corr(
            x.values, y.values, clean['coach'].values, n_boot=2000, seed=0,
        )

        results[label] = {
            'column': col,
            'correlation': float(r),
            'p_value': float(p),
            'ci_low': boot['ci_low'],
            'ci_high': boot['ci_high'],
            'p_bootstrap_coach_clustered': boot['p_bootstrap'],
            'n_coaches': boot['n_clusters'],
            'slope': float(slope),
            'intercept': float(intercept),
            'r_squared': float(r_value ** 2),
            'std_err': float(std_err),
            'n': int(len(clean)),
            'significant': bool(p < 0.05),
        }

        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "n.s.")
        cp = boot['p_bootstrap']
        logger.info(f"  {label:25s}: r={r:7.4f}, p={p:.4f} ({sig}), "
                    f"clust_p={cp:.4f}, n={len(clean)}")

    return results


def analyze_by_era(merged, measures, gene_key):
    """Analyze correlations by era."""
    merged = merged.copy()
    merged['era'] = pd.cut(
        merged['year'],
        bins=[2005, 2011, 2017, 2025],
        labels=['2006-2011', '2012-2017', '2018-2024']
    )

    era_results = {}
    for era_label in ['2006-2011', '2012-2017', '2018-2024']:
        era_data = merged[merged['era'] == era_label]
        if len(era_data) < 10:
            continue

        era_results[era_label] = {}
        for col, label in measures.items():
            clean = era_data[[col, 'annual_war']].dropna()
            if len(clean) < 10:
                continue

            x = clean[col]
            y = clean['annual_war'] * 16

            r, p = stats.pearsonr(x, y)
            era_results[era_label][label] = {
                'correlation': float(r),
                'p_value': float(p),
                'n': int(len(clean)),
                'significant': bool(p < 0.05),
            }

    return era_results


def run_multiple_regression(war_data):
    """Run multiple linear regression: WAR ~ all composite genes.

    Two models:
      1. Offensive genes only (aggression, shotgun, tempo): 2006-2024, ~606 obs
      2. All genes including defensive scheme: 2016-2024, ~282 obs
    """
    logger.info("\n" + "=" * 70)
    logger.info("MULTIPLE LINEAR REGRESSION: WAR ~ GENES")
    logger.info("=" * 70)

    # Load all gene data
    gene_dfs = {}
    for gene_key in ['aggression', 'shotgun', 'tempo', 'defensive_scheme']:
        config = GENE_CONFIGS[gene_key]
        path = Path(config['path'])
        if not path.exists():
            logger.warning(f"Gene file not found: {path}")
            continue
        df = pd.read_csv(path)
        df = df.rename(columns={
            config['coach_col']: 'coach',
            config['year_col']: 'year',
        })
        gene_dfs[gene_key] = df

    # Define composite columns for each gene
    composite_cols = {
        'aggression': 'composite_aggression',
        'shotgun': 'shotgun_gene_zscore',
        'tempo': 'composite_tempo_zscore',
        'defensive_scheme': 'composite_scheme_zscore',
    }

    results = {}

    # --- Model 1: Offensive genes only (2006-2024) ---
    logger.info("\n--- Model 1: Offensive Genes (2006-2024) ---")
    offensive_genes = ['aggression', 'shotgun', 'tempo']
    merged = war_data.copy()
    for gene_key in offensive_genes:
        cols_to_keep = ['coach', 'year', composite_cols[gene_key]]
        gene_subset = gene_dfs[gene_key][cols_to_keep].drop_duplicates()
        merged = pd.merge(merged, gene_subset, on=['coach', 'year'], how='inner')

    feature_cols = [composite_cols[g] for g in offensive_genes]
    clean = merged[feature_cols + ['annual_war']].dropna()
    y = clean['annual_war'].values * 16
    X = clean[feature_cols].values
    coaches1 = merged.loc[clean.index, 'coach'].values

    model1_results = _fit_and_report(X, y, feature_cols,
                                     ['Aggression', 'Shotgun', 'Tempo'],
                                     "Offensive Genes (2006-2024)",
                                     clusters=coaches1)
    model1_results['n'] = len(clean)
    model1_results['n_coaches'] = int(merged.loc[clean.index, 'coach'].nunique())
    model1_results['year_range'] = '2006-2024'
    results['offensive_only'] = model1_results

    # --- Model 2: All genes (2016-2024) ---
    logger.info("\n--- Model 2: All Genes (2016-2024) ---")
    all_genes = ['aggression', 'shotgun', 'tempo', 'defensive_scheme']
    merged2 = war_data[war_data['year'] >= 2016].copy()
    for gene_key in all_genes:
        cols_to_keep = ['coach', 'year', composite_cols[gene_key]]
        gene_subset = gene_dfs[gene_key][cols_to_keep].drop_duplicates()
        merged2 = pd.merge(merged2, gene_subset, on=['coach', 'year'], how='inner')

    feature_cols2 = [composite_cols[g] for g in all_genes]
    clean2 = merged2[feature_cols2 + ['annual_war']].dropna()
    y2 = clean2['annual_war'].values * 16
    X2 = clean2[feature_cols2].values
    coaches2 = merged2.loc[clean2.index, 'coach'].values

    model2_results = _fit_and_report(X2, y2, feature_cols2,
                                     ['Aggression', 'Shotgun', 'Tempo', 'Def. Scheme'],
                                     "All Genes (2016-2024)",
                                     clusters=coaches2)
    model2_results['n'] = len(clean2)
    model2_results['n_coaches'] = int(merged2.loc[clean2.index, 'coach'].nunique())
    model2_results['year_range'] = '2016-2024'
    results['all_genes'] = model2_results

    # --- Multicollinearity check ---
    logger.info("\n--- Multicollinearity Check (VIF) ---")
    corr_matrix = clean2[feature_cols2].corr()
    logger.info("Pairwise correlations between genes:")
    gene_labels = ['Aggression', 'Shotgun', 'Tempo', 'Def. Scheme']
    for i in range(len(feature_cols2)):
        for j in range(i + 1, len(feature_cols2)):
            r = corr_matrix.iloc[i, j]
            logger.info(f"  {gene_labels[i]} <-> {gene_labels[j]}: r={r:.4f}")

    # VIF calculation
    from numpy.linalg import inv
    corr_vals = clean2[feature_cols2].corr().values
    try:
        vif_diag = np.diag(inv(corr_vals))
        vif_results = {}
        for i, label in enumerate(gene_labels):
            logger.info(f"  VIF({label}): {vif_diag[i]:.3f}")
            vif_results[label] = float(vif_diag[i])
        results['vif'] = vif_results
    except Exception as e:
        logger.warning(f"  VIF calculation failed: {e}")

    results['pairwise_correlations'] = {}
    for i in range(len(feature_cols2)):
        for j in range(i + 1, len(feature_cols2)):
            pair = f"{gene_labels[i]} <-> {gene_labels[j]}"
            results['pairwise_correlations'][pair] = float(corr_matrix.iloc[i, j])

    return results


def _fit_and_report(X, y, feature_cols, labels, model_name, clusters=None):
    """Fit OLS and report statistics.

    Point estimates (coefficients, R^2, F) come from OLS. When `clusters` is given
    (coach id per row), coefficient SEs / t / p are coach-CLUSTER-ROBUST and a
    coach-block bootstrap 95% CI is attached -- the WAR panel has many coach-years
    from few coaches (e.g. ~600 obs from ~123 coaches), so treating rows as
    independent understates uncertainty. The F-test stays classical (reported for
    reference only; not valid under clustering).
    """
    n, k = X.shape

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)

    ms_reg = (ss_tot - ss_res) / k
    ms_res = ss_res / (n - k - 1)
    f_stat = ms_reg / ms_res
    f_p_value = 1 - stats.f.cdf(f_stat, k, n - k - 1)

    # classical OLS SEs (fallback / when no clusters provided)
    X_with_const = np.column_stack([np.ones(n), X])
    try:
        cov_matrix = ms_res * np.linalg.inv(X_with_const.T @ X_with_const)
        ols_se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        ols_se = np.full(k + 1, np.nan)

    # coach-clustered inference (preferred when clusters available)
    cl = boot = None
    if clusters is not None:
        cl = cluster_robust_ols(X, y, np.asarray(clusters), labels)
        boot = cluster_bootstrap_ci(X, y, np.asarray(clusters), labels, n_boot=2000, seed=0)
        se_type = 'cluster_robust_by_coach'
        intercept_se = cl['intercept_se']
        dof = cl['n_clusters'] - k - 1
    else:
        se_type = 'classical_ols'
        intercept_se = ols_se[0]
        dof = n - k - 1

    logger.info(f"\n  {model_name}: n={n}, R2={r_squared:.4f}, "
                f"Adj-R2={adj_r_squared:.4f}  [SE: {se_type}]")
    logger.info(f"  F({k},{n-k-1})={f_stat:.3f}, p={f_p_value:.6f}")
    logger.info(f"  Intercept: {model.intercept_:.4f} (SE={intercept_se:.4f})")

    coefficients = {}
    for i, (col, label) in enumerate(zip(feature_cols, labels)):
        coef = float(model.coef_[i])
        if cl is not None:
            c = cl['coefficients'][label]
            se_i, t_stat, p_val = c['std_error'], c['t_statistic'], c['p_value']
        else:
            se_i = ols_se[i + 1]
            t_stat = coef / se_i if not np.isnan(se_i) else np.nan
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof)) if not np.isnan(t_stat) else np.nan
        sig = "**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s.")
        logger.info(f"  {label:15s}: beta={coef:8.4f}, SE={se_i:.4f}, "
                     f"t={t_stat:.3f}, p={p_val:.4f} ({sig})")

        entry = {
            'column': col,
            'coefficient': coef,
            'std_error': float(se_i),
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(np.isfinite(p_val) and p_val < 0.05),
        }
        if boot is not None:
            entry['ci_low'] = boot[label]['ci_low']
            entry['ci_high'] = boot[label]['ci_high']
        coefficients[label] = entry

    result = {
        'model_name': model_name,
        'r_squared': float(r_squared),
        'adj_r_squared': float(adj_r_squared),
        'f_statistic': float(f_stat),
        'f_p_value': float(f_p_value),
        'intercept': float(model.intercept_),
        'intercept_se': float(intercept_se),
        'se_type': se_type,
        'coefficients': coefficients,
    }
    if cl is not None:
        result['n_clusters'] = cl['n_clusters']
        result['bootstrap_n'] = boot['_meta']['n_boot']
    return result


def main():
    logger.info("=" * 70)
    logger.info("GENE vs WAR CORRELATION ANALYSIS")
    logger.info("=" * 70)

    war_data = load_war_data()

    all_results = {}

    for gene_key, config in GENE_CONFIGS.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing: {gene_key}")
        logger.info(f"{'='*50}")

        merged = load_and_merge_gene(gene_key, war_data)
        if merged is None:
            continue

        # Overall correlations
        overall = analyze_correlations(merged, config['measures'], gene_key)

        # By era
        by_era = analyze_by_era(merged, config['measures'], gene_key)

        all_results[gene_key] = {
            'overall': overall,
            'by_era': by_era,
            'n_total': int(len(merged)),
            'n_coaches': int(merged['coach'].nunique()),
            'year_range': f"{int(merged['year'].min())}-{int(merged['year'].max())}",
        }

    # Save results
    output_dir = Path("outputs/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "gene_war_correlation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved results: {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Gene vs WAR Correlations (Overall)")
    print("=" * 80)
    print(f"{'Gene':<25} {'r':>8} {'p':>10} {'n':>6} {'Sig?':>6}")
    print("-" * 60)

    for gene_key, gene_results in all_results.items():
        for label, stats_dict in gene_results['overall'].items():
            sig = "**" if stats_dict['p_value'] < 0.01 else (
                "*" if stats_dict['p_value'] < 0.05 else "")
            print(f"{label:<25} {stats_dict['correlation']:8.4f} "
                  f"{stats_dict['p_value']:10.4f} {stats_dict['n']:6d} "
                  f"{sig:>6}")

    print("\n" + "=" * 80)
    print("BY ERA")
    print("=" * 80)

    for gene_key, gene_results in all_results.items():
        for era, era_stats in gene_results.get('by_era', {}).items():
            for label, s in era_stats.items():
                sig = "*" if s['p_value'] < 0.05 else ""
                print(f"{label:<25} {era:<12} r={s['correlation']:7.4f} "
                      f"p={s['p_value']:.4f} n={s['n']:4d} {sig}")

    # Multiple linear regression
    mlr_results = run_multiple_regression(war_data)

    # Save MLR results
    mlr_file = output_dir / "gene_war_multiple_regression.json"
    with open(mlr_file, 'w') as f:
        json.dump(mlr_results, f, indent=2)
    logger.info(f"\nSaved MLR results: {mlr_file}")

    # Print MLR summary
    print("\n" + "=" * 80)
    print("MULTIPLE LINEAR REGRESSION SUMMARY")
    print("=" * 80)

    for model_key in ['offensive_only', 'all_genes']:
        m = mlr_results[model_key]
        print(f"\n{m['model_name']} (n={m['n']}, {m['n_coaches']} coaches)")
        print(f"  R2={m['r_squared']:.4f}, Adj-R2={m['adj_r_squared']:.4f}, "
              f"F={m['f_statistic']:.3f}, p={m['f_p_value']:.6f}")
        print(f"  {'Predictor':<15} {'Beta':>8} {'SE':>8} {'t':>8} {'p':>10}")
        print(f"  {'-'*55}")
        for label, coef_info in m['coefficients'].items():
            sig = "**" if coef_info['p_value'] < 0.01 else (
                "*" if coef_info['p_value'] < 0.05 else "")
            print(f"  {label:<15} {coef_info['coefficient']:8.4f} "
                  f"{coef_info['std_error']:8.4f} "
                  f"{coef_info['t_statistic']:8.3f} "
                  f"{coef_info['p_value']:10.4f} {sig}")

    if 'vif' in mlr_results:
        print(f"\n  VIF values:")
        for label, vif in mlr_results['vif'].items():
            flag = " (HIGH)" if vif > 5 else ""
            print(f"    {label}: {vif:.3f}{flag}")

    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
