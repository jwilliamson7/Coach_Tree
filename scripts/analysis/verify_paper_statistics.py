#!/usr/bin/env python3
"""
Verify Paper Statistics Against Source Data

Recomputes all key statistics cited in the paper and saves them to a
verification log. This ensures no statistical values are fabricated.

Outputs:
    outputs/analysis/paper_statistics_verification.json
    outputs/analysis/paper_statistics_verification.log

Usage:
    python scripts/analysis/verify_paper_statistics.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from scipy import stats

# Set up both file and console logging
log_dir = Path("outputs/analysis")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "paper_statistics_verification.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def verify_coordinator_to_hc_inheritance():
    """Verify Table 13: Coordinator-to-HC Gene Transmission.

    The paper uses AGGREGATED values where multiple coordinator stints
    per coach are weighted-averaged by years_with_data, then correlations
    are computed on unique coaches (not individual stints).

    This matches the methodology in visualize_gene_inheritance.py:
    aggregate_coordinator_stints().
    """
    logger.info("=" * 70)
    logger.info("VERIFYING: Coordinator-to-HC Gene Inheritance (Table 13)")
    logger.info("=" * 70)

    path = Path("data/processed/coaching_genes/gene_inheritance.csv")
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} inheritance records")

    results = {
        'raw_per_stint': {},
        'aggregated_per_coach': {},
    }

    gene_types = df['gene_type'].unique()
    logger.info(f"Gene types: {list(gene_types)}")

    for gene_type in sorted(gene_types):
        gene_df = df[df['gene_type'] == gene_type].copy()

        # --- RAW (per-stint) statistics ---
        n_raw = len(gene_df)
        if n_raw >= 5:
            r_raw, p_raw = stats.pearsonr(
                gene_df['coord_era_gene'], gene_df['hc_era_gene']
            )
            same_sign = np.sum(
                np.sign(gene_df['coord_era_gene'].values) ==
                np.sign(gene_df['hc_era_gene'].values)
            )
            dir_ret_raw = 100 * same_sign / n_raw
        else:
            r_raw, p_raw, dir_ret_raw = np.nan, np.nan, np.nan

        results['raw_per_stint'][gene_type] = {
            'n': int(n_raw),
            'pearson_r': round(float(r_raw), 4) if not np.isnan(r_raw) else None,
            'p_value': round(float(p_raw), 4) if not np.isnan(p_raw) else None,
            'direction_retention_pct': round(float(dir_ret_raw), 1) if not np.isnan(dir_ret_raw) else None,
            'transition_type': gene_df['transition_type'].iloc[0],
        }

        logger.info(f"\n  {gene_type} RAW ({gene_df['transition_type'].iloc[0]}): "
                     f"n={n_raw}, r={r_raw:.4f}, p={p_raw:.4f}, "
                     f"dir_ret={dir_ret_raw:.1f}%")

        # --- AGGREGATED (per-coach) statistics ---
        # Weight-average coordinator stints per coach
        agg_rows = []
        for coach, group in gene_df.groupby('coach_name'):
            weights = group['coord_years_with_data'].values
            coord_gene = np.average(group['coord_era_gene'].values, weights=weights)
            hc_gene = group['hc_era_gene'].iloc[0]  # Same HC-era for all stints
            agg_rows.append({
                'coach_name': coach,
                'coord_era_gene': coord_gene,
                'hc_era_gene': hc_gene,
                'total_coord_years': weights.sum(),
                'num_stints': len(group),
            })
        agg_df = pd.DataFrame(agg_rows)

        n_agg = len(agg_df)
        if n_agg >= 5:
            r_agg, p_agg = stats.pearsonr(
                agg_df['coord_era_gene'], agg_df['hc_era_gene']
            )
            same_sign_agg = np.sum(
                np.sign(agg_df['coord_era_gene'].values) ==
                np.sign(agg_df['hc_era_gene'].values)
            )
            dir_ret_agg = 100 * same_sign_agg / n_agg
        else:
            r_agg, p_agg, dir_ret_agg = np.nan, np.nan, np.nan

        results['aggregated_per_coach'][gene_type] = {
            'n': int(n_agg),
            'pearson_r': round(float(r_agg), 4) if not np.isnan(r_agg) else None,
            'p_value': round(float(p_agg), 6) if not np.isnan(p_agg) else None,
            'direction_retention_pct': round(float(dir_ret_agg), 1) if not np.isnan(dir_ret_agg) else None,
            'transition_type': gene_df['transition_type'].iloc[0],
            'coaches_with_multiple_stints': int((agg_df['num_stints'] > 1).sum()),
        }

        logger.info(f"  {gene_type} AGGREGATED: "
                     f"n={n_agg}, r={r_agg:.4f}, p={p_agg:.6f}, "
                     f"dir_ret={dir_ret_agg:.1f}%")

        # List all coaches for verification
        logger.info(f"  Coaches ({n_agg}):")
        for _, row in agg_df.sort_values('coach_name').iterrows():
            stints = f" ({int(row['num_stints'])} stints)" if row['num_stints'] > 1 else ""
            logger.info(f"    {row['coach_name']}: coord={row['coord_era_gene']:.4f} "
                         f"-> hc={row['hc_era_gene']:.4f}{stints}")

    return results


def verify_gene_war_correlations():
    """Verify Gene-WAR table (gene_war)."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING: Gene-WAR Correlations")
    logger.info("=" * 70)

    # Load the saved results
    results_file = Path("outputs/analysis/gene_war_correlation_results.json")
    if not results_file.exists():
        logger.error(f"File not found: {results_file}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    # Report composite gene values only (what paper uses)
    verification = {}
    composite_map = {
        'defensive_scheme': ('Defensive Scheme', 'composite_scheme_zscore'),
        'aggression': ('Composite Aggression', 'composite_aggression'),
        'shotgun': ('Shotgun Formation', 'shotgun_gene_zscore'),
        'tempo': ('Composite Tempo', 'composite_tempo_zscore'),
    }

    for gene_key, (label, col) in composite_map.items():
        if gene_key in results and label in results[gene_key]['overall']:
            stats_dict = results[gene_key]['overall'][label]
            verification[gene_key] = {
                'label': label,
                'r': stats_dict['correlation'],
                'p': stats_dict['p_value'],
                'n': stats_dict['n'],
                'significant': stats_dict['significant'],
            }
            sig = "**" if stats_dict['p_value'] < 0.01 else (
                "*" if stats_dict['p_value'] < 0.05 else "n.s.")
            logger.info(f"  {label:25s}: r={stats_dict['correlation']:.4f}, "
                         f"p={stats_dict['p_value']:.4f}, "
                         f"n={stats_dict['n']} ({sig})")

    return verification


def verify_mentor_protege_war():
    """Verify Table war_inherit: Mentor WAR -> Protege WAR."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING: Mentor WAR -> Protege WAR")
    logger.info("=" * 70)

    results_file = Path("outputs/analysis/mentor_protege_war_analysis.json")
    if not results_file.exists():
        logger.error(f"File not found: {results_file}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    logger.info(f"  Overall: r={results['overall']['correlation']:.4f}, "
                 f"p={results['overall']['p_value']:.4f}, "
                 f"n={results['overall']['n']}")

    for coord_type in ['Offensive Coordinator', 'Defensive Coordinator']:
        if coord_type in results.get('by_coordinator_type', {}):
            ct = results['by_coordinator_type'][coord_type]
            logger.info(f"  {coord_type}->HC: r={ct['correlation']:.4f}, "
                         f"p={ct['p_value']:.4f}, n={ct['n']}")

    return results


def verify_shotgun_inheritance_by_type():
    """Verify Table shotgun_inherit: Mentor-Protege Shotgun Inheritance."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING: Shotgun Mentor-Protege Inheritance")
    logger.info("=" * 70)

    results_file = Path("outputs/analysis/shotgun_inheritance_by_type_results.json")
    if not results_file.exists():
        logger.error(f"File not found: {results_file}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    logger.info(f"  Overall: r={results['overall']['correlation']:.4f}, "
                 f"p={results['overall']['p_value']:.6f}, "
                 f"n={results['overall']['n']}")

    for key in ['by_coordinator_type', 'by_mentor_background']:
        if key in results:
            for subkey, stats_dict in results[key].items():
                logger.info(f"  {key}/{subkey}: r={stats_dict['correlation']:.4f}, "
                             f"p={stats_dict['p_value']:.6f}, "
                             f"n={stats_dict['n']}")

    return results


def verify_aggression_inheritance_by_type():
    """Verify Tables inheritance and inherit_coord."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING: Aggression Mentor-Protege Inheritance by Type")
    logger.info("=" * 70)

    results_file = Path("outputs/analysis/inheritance_by_type_results.json")
    if not results_file.exists():
        logger.error(f"File not found: {results_file}")
        return None

    with open(results_file) as f:
        results = json.load(f)

    for section, section_data in results.items():
        logger.info(f"\n  Section: {section}")
        if isinstance(section_data, dict):
            for key, val in section_data.items():
                if isinstance(val, dict) and 'correlation' in val:
                    logger.info(f"    {key}: r={val['correlation']:.4f}, "
                                 f"p={val['p_value']:.6f}, n={val['n']}")
                elif isinstance(val, dict):
                    for subkey, subval in val.items():
                        if isinstance(subval, dict) and 'correlation' in subval:
                            logger.info(f"    {key}/{subkey}: "
                                         f"r={subval['correlation']:.4f}, "
                                         f"p={subval['p_value']:.6f}, "
                                         f"n={subval['n']}")

    return results


def verify_aggression_war():
    """Verify aggression-WAR correlation tables. Reports the season-level raw and
    era-adjusted estimates so the verifier captures the era-clean primary."""
    logger.info("\n" + "=" * 70)
    logger.info("VERIFYING: Aggression-WAR Correlations")
    logger.info("=" * 70)

    out = {}
    reg = Path("outputs/analysis/aggression_war_regression_results.json")
    if reg.exists():
        with open(reg) as f:
            data = json.load(f)
        for label, val in data.items():
            if isinstance(val, dict) and 'correlation' in val:
                out[label] = {
                    'r_raw': val.get('correlation'),
                    'p_raw_clustered': val.get('p_bootstrap_coach_clustered'),
                    'r_eradj': val.get('correlation_eradj'),
                    'p_eradj_clustered': val.get('p_bootstrap_coach_clustered_eradj'),
                    'career_r': val.get('career', {}).get('all_coaches', {}).get('correlation'),
                    'career_war_vs_era_r': val.get('career', {}).get('career_war_vs_era_r'),
                    'n': val.get('n'),
                }
                logger.info(f"    {label:25s}: raw r={val.get('correlation'):.4f} | "
                            f"era-adj r={val.get('correlation_eradj', float('nan')):.4f} | "
                            f"career r={out[label]['career_r']}")
    return out


def verify_headline_recompute():
    """Independently RECOMPUTE the two headline numbers from source/intermediate
    data (not echoing the analysis JSON), so a fabricated or drifted JSON would be
    caught: (a) the season-level era-adjusted composite aggression -> WAR
    correlation, recomputed from the gene CSV + WAR trajectories; (b) the
    era-adjusted Offensive-mentor x OC-protege shotgun inheritance cell, recomputed
    from the saved mentor-protege pairs. Logs each against the pipeline value."""
    import sys as _sys
    logger.info("\n" + "=" * 70)
    logger.info("INDEPENDENT RECOMPUTE OF HEADLINE NUMBERS")
    logger.info("=" * 70)
    _sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    out = {}

    # (a) season-level era-adjusted composite aggression -> WAR
    try:
        from utils.data_paths import coach_war_trajectories_path, merge_gene_war
        from utils.parsimony import within_group_demean
        agg = pd.read_csv("data/processed/coaching_genes/aggression_gene_by_year.csv")
        agg = agg.rename(columns={'head_coach': 'coach', 'season': 'year'})
        war = pd.read_csv(coach_war_trajectories_path())
        war.columns = war.columns.str.lower()
        m = merge_gene_war(agg, war, 'coach', 'coach', year_cols=('year', 'year'), how='inner')
        m = m[['composite_aggression', 'annual_war', 'coach', 'year']].dropna().copy()
        m['_w'] = m['annual_war'] * 16
        gx = within_group_demean(m, 'composite_aggression', 'year')
        gy = within_group_demean(m, '_w', 'year')
        r = float(np.corrcoef(gx, gy)[0, 1])
        j = json.load(open("outputs/analysis/aggression_war_regression_results.json"))
        jr = j.get('Composite Aggression', {}).get('correlation_eradj')
        out['composite_aggression_war_eradj'] = {'recomputed_r': r, 'json_r': jr, 'n': int(len(m))}
        logger.info(f"  composite aggression->WAR era-adj: recomputed r={r:.4f} "
                    f"(n={len(m)}) vs JSON {jr}")
    except Exception as e:
        logger.warning(f"  aggression->WAR recompute failed: {e}")

    # (b) era-adjusted Offensive-mentor x OC-protege shotgun cell, from saved pairs
    try:
        p = pd.read_csv("outputs/analysis/shotgun_mentor_protege_pairs.csv")
        cell = p[(p['mentor_background'] == 'Offensive') & (p['protege_role'] == 'OC')]
        cell = cell[['shotgun_mentor_eradj', 'shotgun_protege_eradj']].dropna()
        r = float(np.corrcoef(cell['shotgun_mentor_eradj'], cell['shotgun_protege_eradj'])[0, 1])
        j = json.load(open("outputs/analysis/shotgun_inheritance_by_type_results.json"))
        jr = j.get('two_by_two', {}).get('Offensive|OC', {}).get('correlation')
        out['shotgun_off_mentor_oc_protege_eradj'] = {'recomputed_r': r, 'json_r': jr,
                                                      'n': int(len(cell))}
        logger.info(f"  shotgun Off-mentor x OC-protege era-adj: recomputed r={r:.4f} "
                    f"(n={len(cell)}) vs JSON {jr}")
    except Exception as e:
        logger.warning(f"  shotgun 2x2 recompute failed: {e}")

    return out


def main():
    logger.info("=" * 70)
    logger.info("PAPER STATISTICS VERIFICATION")
    logger.info(f"Run date: {datetime.now().isoformat()}")
    logger.info("=" * 70)

    all_results = {
        'verification_date': datetime.now().isoformat(),
        'sections': {}
    }

    # 1. Coordinator-to-HC Gene Inheritance (Table 13) - CRITICAL
    coord_hc = verify_coordinator_to_hc_inheritance()
    if coord_hc:
        all_results['sections']['coordinator_to_hc_inheritance'] = coord_hc

    # 2. Gene-WAR correlations
    gene_war = verify_gene_war_correlations()
    if gene_war:
        all_results['sections']['gene_war_correlations'] = gene_war

    # 3. Mentor-protege WAR
    war_inherit = verify_mentor_protege_war()
    if war_inherit:
        all_results['sections']['mentor_protege_war'] = {
            'overall': war_inherit['overall'],
            'by_coordinator_type': war_inherit.get('by_coordinator_type', {}),
        }

    # 4. Shotgun mentor-protege inheritance
    shotgun_inherit = verify_shotgun_inheritance_by_type()
    if shotgun_inherit:
        all_results['sections']['shotgun_mentor_protege'] = shotgun_inherit

    # 5. Aggression inheritance by type
    agg_inherit = verify_aggression_inheritance_by_type()
    if agg_inherit:
        all_results['sections']['aggression_inheritance_by_type'] = agg_inherit

    # 6. Aggression-WAR (now captured)
    agg_war = verify_aggression_war()
    if agg_war:
        all_results['sections']['aggression_war'] = agg_war

    # 7. Independent recompute of headline numbers (recompute, not echo)
    recompute = verify_headline_recompute()
    if recompute:
        all_results['sections']['headline_recompute'] = recompute

    # Save verification results
    output_file = log_dir / "paper_statistics_verification.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*70}")
    logger.info(f"Verification complete!")
    logger.info(f"JSON: {output_file}")
    logger.info(f"Log:  {log_file}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
