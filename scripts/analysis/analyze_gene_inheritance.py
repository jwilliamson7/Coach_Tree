#!/usr/bin/env python3
"""
Analyze Gene Inheritance: Coordinator -> Head Coach

Tests whether coaching philosophies propagate when coordinators are promoted to HC.

For DC->HC transitions:
  - DC-era gene: team defensive scheme gene during their DC years
  - HC-era gene: team defensive scheme gene during their HC years
  - Test: do these correlate?

For OC->HC transitions:
  - OC-era gene: team offensive genes (aggression, tempo, shotgun) during OC years
    (attributed to the HC at the time, since offensive genes are HC-keyed)
  - HC-era gene: same offensive genes during their HC years
  - Test: do these correlate?

Usage:
    python analyze_gene_inheritance.py
    python analyze_gene_inheritance.py --min_years 1
"""

import argparse
import pandas as pd
import numpy as np
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.parsimony import cluster_bootstrap_corr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PFR -> PBP Team Abbreviation Mapping
# =============================================================================

# Static mappings for teams whose PBP code doesn't change in the modern era
PFR_TO_PBP_STATIC = {
    'atl': 'ATL', 'buf': 'BUF', 'car': 'CAR', 'chi': 'CHI',
    'cin': 'CIN', 'cle': 'CLE', 'clt': 'IND', 'crd': 'ARI',
    'dal': 'DAL', 'den': 'DEN', 'det': 'DET', 'gnb': 'GB',
    'htx': 'HOU', 'jax': 'JAX', 'kan': 'KC', 'mia': 'MIA',
    'min': 'MIN', 'nor': 'NO', 'nwe': 'NE', 'nyg': 'NYG',
    'nyj': 'NYJ', 'oti': 'TEN', 'phi': 'PHI', 'pit': 'PIT',
    'rav': 'BAL', 'sea': 'SEA', 'sfo': 'SF', 'tam': 'TB', 'was': 'WAS',
}


def pfr_to_pbp(pfr_code: str, year: int) -> str:
    """Convert PFR lowercase team code to PBP format for a given year.

    Handles year-dependent relocations (Raiders, Chargers, Rams).
    """
    if pfr_code in PFR_TO_PBP_STATIC:
        return PFR_TO_PBP_STATIC[pfr_code]
    # Year-dependent relocations
    if pfr_code in ('rai', 'oak', 'lvr'):
        return 'LV' if year >= 2020 else 'OAK'
    if pfr_code in ('sdg', 'lac', 'sd'):
        return 'LAC' if year >= 2017 else 'SD'
    if pfr_code in ('ram', 'lar', 'stl'):
        return 'LA' if year >= 2016 else 'STL'
    return pfr_code.upper()  # fallback


# =============================================================================
# Main Analysis Class
# =============================================================================

class InheritanceAnalyzer:
    """Analyze gene inheritance from coordinator to head coach."""

    def __init__(self,
                 coaching_tree_dir: str = "data/processed/coaching_tree",
                 genes_dir: str = "data/processed/coaching_genes",
                 coaching_dir: str = "data/processed/Coaching",
                 min_years: int = 1):
        self.coaching_tree_dir = Path(coaching_tree_dir)
        self.genes_dir = Path(genes_dir)
        self.coaching_dir = Path(coaching_dir)
        self.min_years = min_years

        self.coaches = {}
        self.defensive_genes = None
        self.aggression_genes = None
        self.tempo_genes = None
        self.shotgun_genes = None
        self.hc_mapping = None

    # -----------------------------------------------------------------
    # Data Loading
    # -----------------------------------------------------------------

    def load_data(self):
        """Load all required data files."""
        # coaches.json — career paths
        coaches_path = self.coaching_tree_dir / "coaches.json"
        logger.info(f"Loading coaches from {coaches_path}")
        with open(coaches_path, 'r') as f:
            self.coaches = json.load(f)
        logger.info(f"Loaded {len(self.coaches)} coaches")

        # Defensive scheme gene (team-year, 2016-2024)
        def_path = self.genes_dir / "defensive_scheme_gene.csv"
        if def_path.exists():
            self.defensive_genes = pd.read_csv(def_path)
            logger.info(f"Loaded defensive genes: {len(self.defensive_genes)} team-years")
        else:
            logger.warning(f"Defensive gene file not found: {def_path}")

        # Offensive genes (HC-keyed, 2006-2024)
        for attr, filename in [
            ('aggression_genes', 'aggression_gene_by_year.csv'),
            ('tempo_genes', 'tempo_gene.csv'),
            ('shotgun_genes', 'shotgun_gene.csv'),
        ]:
            path = self.genes_dir / filename
            if path.exists():
                setattr(self, attr, pd.read_csv(path))
                logger.info(f"Loaded {attr}: {len(getattr(self, attr))} coach-years")
            else:
                logger.warning(f"{filename} not found")

        # HC mapping — team_year_head_coaches.csv (uppercase PFR team codes)
        hc_path = self.coaching_dir / "team_year_head_coaches.csv"
        if hc_path.exists():
            self.hc_mapping = pd.read_csv(hc_path)
            logger.info(f"Loaded HC mapping: {len(self.hc_mapping)} team-years")
        else:
            logger.warning(f"HC mapping not found: {hc_path}")

    # -----------------------------------------------------------------
    # Stint Extraction & Transition Building
    # -----------------------------------------------------------------

    def _extract_stints(self, career: dict, role_category: str) -> List[dict]:
        """Extract consecutive stints at the same team in the same role.

        A stint = consecutive years at one team with the given role_category.
        Returns list of {team: str, years: [int]}.
        """
        stints = []
        current = None

        for year_str in sorted(career.keys(), key=lambda y: int(y)):
            entry = career[year_str]
            year = int(year_str)

            if entry.get('role_category') == role_category and entry.get('level') == 'NFL':
                team = entry.get('team', '')
                if current and current['team'] == team:
                    current['years'].append(year)
                else:
                    if current:
                        stints.append(current)
                    current = {'team': team, 'years': [year]}
            else:
                if current:
                    stints.append(current)
                    current = None

        if current:
            stints.append(current)

        return stints

    def build_transitions(self) -> List[dict]:
        """Build coordinator->HC transitions from coaches.json.

        Each coordinator stint is paired with the first HC stint that starts
        after the coordinator stint ends (temporal ordering enforced).
        """
        transitions = []

        for coach_id, coach_data in self.coaches.items():
            name = coach_data.get('name', '')
            career = coach_data.get('career', {})
            if not career:
                continue

            dc_stints = self._extract_stints(career, 'DC')
            oc_stints = self._extract_stints(career, 'OC')
            hc_stints = self._extract_stints(career, 'HC')

            if not hc_stints:
                continue

            # DC -> HC transitions
            for coord_stint in dc_stints:
                coord_end = max(coord_stint['years'])
                for hc_stint in hc_stints:
                    hc_start = min(hc_stint['years'])
                    if hc_start > coord_end:
                        transitions.append({
                            'coach_name': name,
                            'coach_id': coach_id,
                            'transition_type': 'DC->HC',
                            'coord_team': coord_stint['team'],
                            'coord_years': coord_stint['years'],
                            'hc_team': hc_stint['team'],
                            'hc_years': hc_stint['years'],
                            'years_gap': hc_start - coord_end,
                        })
                        break  # Pair with first eligible HC stint only

            # OC -> HC transitions
            for coord_stint in oc_stints:
                coord_end = max(coord_stint['years'])
                for hc_stint in hc_stints:
                    hc_start = min(hc_stint['years'])
                    if hc_start > coord_end:
                        transitions.append({
                            'coach_name': name,
                            'coach_id': coach_id,
                            'transition_type': 'OC->HC',
                            'coord_team': coord_stint['team'],
                            'coord_years': coord_stint['years'],
                            'hc_team': hc_stint['team'],
                            'hc_years': hc_stint['years'],
                            'years_gap': hc_start - coord_end,
                        })
                        break

        dc_count = sum(1 for t in transitions if t['transition_type'] == 'DC->HC')
        oc_count = sum(1 for t in transitions if t['transition_type'] == 'OC->HC')
        logger.info(f"Found {len(transitions)} total transitions (DC->HC: {dc_count}, OC->HC: {oc_count})")

        return transitions

    # -----------------------------------------------------------------
    # Gene Lookups
    # -----------------------------------------------------------------

    def _get_defensive_gene_by_team(self, team_pfr: str, year: int) -> Optional[float]:
        """Look up defensive scheme gene for a team-year via PFR code."""
        if self.defensive_genes is None:
            return None
        pbp_code = pfr_to_pbp(team_pfr, year)
        match = self.defensive_genes[
            (self.defensive_genes['defteam'] == pbp_code) &
            (self.defensive_genes['season'] == year)
        ]
        if len(match) == 1:
            return float(match.iloc[0]['composite_scheme'])
        return None

    def _get_defensive_gene_by_hc(self, coach_name: str, year: int) -> Optional[float]:
        """Look up defensive scheme gene by HC name and season."""
        if self.defensive_genes is None:
            return None
        match = self.defensive_genes[
            (self.defensive_genes['head_coach'] == coach_name) &
            (self.defensive_genes['season'] == year)
        ]
        if len(match) == 1:
            return float(match.iloc[0]['composite_scheme'])
        return None

    def _get_offensive_gene(self, coach_name: str, year: int, gene_type: str) -> Optional[float]:
        """Look up an offensive gene by HC name and season.

        gene_type: 'aggression', 'tempo', or 'shotgun'
        """
        gene_map = {
            'aggression': (self.aggression_genes, 'composite_aggression'),
            'tempo': (self.tempo_genes, 'composite_tempo'),
            'shotgun': (self.shotgun_genes, 'shotgun_gene_zscore'),
        }
        df, col = gene_map.get(gene_type, (None, None))
        if df is None:
            return None
        match = df[(df['head_coach'] == coach_name) & (df['season'] == year)]
        if len(match) == 1:
            val = match.iloc[0][col]
            if pd.notna(val):
                return float(val)
        return None

    def _get_hc_name(self, team_pfr: str, year: int) -> Optional[str]:
        """Get HC name for a team-year from team_year_head_coaches.csv.

        team_year_head_coaches.csv uses uppercase PFR codes (CRD, GNB, KAN).
        """
        if self.hc_mapping is None:
            return None
        team_upper = team_pfr.upper()
        match = self.hc_mapping[
            (self.hc_mapping['Team'] == team_upper) &
            (self.hc_mapping['Year'] == year)
        ]
        if len(match) == 1:
            return match.iloc[0]['Primary_Coach']
        return None

    # -----------------------------------------------------------------
    # Inheritance Analysis
    # -----------------------------------------------------------------

    def analyze_defensive_inheritance(self, transitions: List[dict]) -> pd.DataFrame:
        """Analyze DC->HC defensive gene inheritance.

        DC-era: look up defensive_scheme_gene by (defteam=DC's team, season)
        HC-era: look up defensive_scheme_gene by (head_coach=name, season)
        """
        results = []
        dc_transitions = [t for t in transitions if t['transition_type'] == 'DC->HC']

        for t in dc_transitions:
            # DC-era genes
            dc_genes = []
            for year in t['coord_years']:
                gene = self._get_defensive_gene_by_team(t['coord_team'], year)
                if gene is not None:
                    dc_genes.append(gene)

            # HC-era genes
            hc_genes = []
            for year in t['hc_years']:
                gene = self._get_defensive_gene_by_hc(t['coach_name'], year)
                if gene is not None:
                    hc_genes.append(gene)

            if len(dc_genes) >= self.min_years and len(hc_genes) >= self.min_years:
                coord_avg = float(np.mean(dc_genes))
                hc_avg = float(np.mean(hc_genes))
                results.append({
                    'coach_name': t['coach_name'],
                    'coach_id': t['coach_id'],
                    'transition_type': 'DC->HC',
                    'gene_type': 'defensive_scheme',
                    'coord_team': t['coord_team'],
                    'coord_years': f"{min(t['coord_years'])}-{max(t['coord_years'])}",
                    'coord_years_with_data': len(dc_genes),
                    'coord_era_gene': coord_avg,
                    'hc_team': t['hc_team'],
                    'hc_years': f"{min(t['hc_years'])}-{max(t['hc_years'])}",
                    'hc_years_with_data': len(hc_genes),
                    'hc_era_gene': hc_avg,
                    'gene_change': hc_avg - coord_avg,
                    'years_gap': t['years_gap'],
                })

        logger.info(f"Defensive inheritance: {len(results)} valid transitions "
                     f"(from {len(dc_transitions)} total DC->HC)")
        return pd.DataFrame(results)

    def analyze_offensive_inheritance(self, transitions: List[dict]) -> pd.DataFrame:
        """Analyze OC->HC offensive gene inheritance.

        OC-era: find HC of team during OC years, look up their offensive gene
        HC-era: look up offensive gene under the OC's own name (now HC)
        """
        results = []
        oc_transitions = [t for t in transitions if t['transition_type'] == 'OC->HC']
        gene_types = ['aggression', 'tempo', 'shotgun']

        for t in oc_transitions:
            for gene_type in gene_types:
                # OC-era: find HC of team, look up their gene
                oc_genes = []
                for year in t['coord_years']:
                    hc_name = self._get_hc_name(t['coord_team'], year)
                    if hc_name:
                        gene = self._get_offensive_gene(hc_name, year, gene_type)
                        if gene is not None:
                            oc_genes.append(gene)

                # HC-era: look up directly by own name
                hc_genes = []
                for year in t['hc_years']:
                    gene = self._get_offensive_gene(t['coach_name'], year, gene_type)
                    if gene is not None:
                        hc_genes.append(gene)

                if len(oc_genes) >= self.min_years and len(hc_genes) >= self.min_years:
                    coord_avg = float(np.mean(oc_genes))
                    hc_avg = float(np.mean(hc_genes))
                    results.append({
                        'coach_name': t['coach_name'],
                        'coach_id': t['coach_id'],
                        'transition_type': 'OC->HC',
                        'gene_type': gene_type,
                        'coord_team': t['coord_team'],
                        'coord_years': f"{min(t['coord_years'])}-{max(t['coord_years'])}",
                        'coord_years_with_data': len(oc_genes),
                        'coord_era_gene': coord_avg,
                        'hc_team': t['hc_team'],
                        'hc_years': f"{min(t['hc_years'])}-{max(t['hc_years'])}",
                        'hc_years_with_data': len(hc_genes),
                        'hc_era_gene': hc_avg,
                        'gene_change': hc_avg - coord_avg,
                        'years_gap': t['years_gap'],
                    })

        n_coaches = len(set(r['coach_id'] for r in results))
        logger.info(f"Offensive inheritance: {len(results)} transition-gene pairs "
                     f"({n_coaches} coaches, from {len(oc_transitions)} total OC->HC)")
        return pd.DataFrame(results)

    # -----------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------

    def _compute_statistics(self, df: pd.DataFrame) -> dict:
        """Compute inheritance statistics per gene type."""
        results = {}

        for gene_type in sorted(df['gene_type'].unique()):
            subset = df[df['gene_type'] == gene_type]
            n = len(subset)

            if n < 3:
                results[gene_type] = {
                    'n': n,
                    'note': 'Too few transitions for statistical analysis'
                }
                continue

            coord_vals = subset['coord_era_gene'].values
            hc_vals = subset['hc_era_gene'].values
            changes = subset['gene_change'].values

            # Pearson correlation. These are raw per-stint rows: a coach with
            # multiple coordinator stints appears more than once, so cluster the
            # bootstrap on coach_id rather than treating stints as independent.
            r, p = stats.pearsonr(coord_vals, hc_vals)
            boot = cluster_bootstrap_corr(
                coord_vals, hc_vals, subset['coach_id'].values,
                n_boot=2000, seed=42,
            )

            # Direction retention: same sign (exclude zeros)
            nonzero = (coord_vals != 0) & (hc_vals != 0)
            if nonzero.sum() > 0:
                same_sign = np.sum(np.sign(coord_vals[nonzero]) == np.sign(hc_vals[nonzero]))
                direction_pct = round(same_sign / nonzero.sum() * 100, 1)
            else:
                direction_pct = None

            results[gene_type] = {
                'n': n,
                'pearson_r': round(float(r), 4),
                'pearson_p': round(float(p), 4),
                'significant': p < 0.05,
                'ci_low': round(boot['ci_low'], 4),
                'ci_high': round(boot['ci_high'], 4),
                'p_bootstrap_coach_clustered': round(boot['p_bootstrap'], 4),
                'n_coaches': boot['n_clusters'],
                'mean_coord_gene': round(float(np.mean(coord_vals)), 4),
                'mean_hc_gene': round(float(np.mean(hc_vals)), 4),
                'mean_change': round(float(np.mean(changes)), 4),
                'mean_abs_change': round(float(np.mean(np.abs(changes))), 4),
                'direction_retention_pct': direction_pct,
                'mean_years_gap': round(float(subset['years_gap'].mean()), 1),
            }

        return results

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------

    def _generate_summary(self, all_results: pd.DataFrame, statistics: dict) -> dict:
        """Generate summary JSON with statistics and notable examples."""
        summary = {
            'generated_date': datetime.now().isoformat(),
            'total_transitions': len(all_results),
            'unique_coaches': int(all_results['coach_id'].nunique()),
            'min_years_required': self.min_years,
            'statistics': statistics,
            'notable_examples': {},
        }

        for gene_type in sorted(all_results['gene_type'].unique()):
            subset = all_results[all_results['gene_type'] == gene_type].copy()
            if len(subset) == 0:
                continue

            subset = subset.copy()
            subset['abs_change'] = subset['gene_change'].abs()

            display_cols = ['coach_name', 'coord_team', 'coord_years',
                            'coord_era_gene', 'hc_team', 'hc_years',
                            'hc_era_gene', 'gene_change']

            faithful = subset.nsmallest(min(5, len(subset)), 'abs_change')
            transformers = subset.nlargest(min(5, len(subset)), 'abs_change')

            summary['notable_examples'][gene_type] = {
                'faithful_inheritors': faithful[display_cols].to_dict('records'),
                'biggest_transformers': transformers[display_cols].to_dict('records'),
            }

        return summary

    def _print_results(self, all_results: pd.DataFrame, statistics: dict):
        """Print analysis results to console."""
        print("\n" + "=" * 80)
        print("GENE INHERITANCE ANALYSIS: Coordinator -> Head Coach")
        print("=" * 80)

        for gene_type, stat in statistics.items():
            transition_type = 'DC->HC' if gene_type == 'defensive_scheme' else 'OC->HC'
            print(f"\n--- {gene_type.upper()} ({transition_type}) ---")
            n = stat['n']

            if n < 3:
                print(f"  Only {n} transition(s) — insufficient for analysis")
                continue

            sig = " *" if stat.get('significant') else ""
            print(f"  Transitions: {n}")
            print(f"  Pearson r: {stat['pearson_r']:.3f} (p={stat['pearson_p']:.4f}){sig}")
            print(f"  Mean coordinator-era gene: {stat['mean_coord_gene']:.3f}")
            print(f"  Mean HC-era gene: {stat['mean_hc_gene']:.3f}")
            print(f"  Mean absolute change: {stat['mean_abs_change']:.3f}")
            if stat.get('direction_retention_pct') is not None:
                print(f"  Direction retention: {stat['direction_retention_pct']:.0f}%")
            print(f"  Mean years gap: {stat['mean_years_gap']:.1f}")

            # Show individual transitions
            subset = all_results[all_results['gene_type'] == gene_type].sort_values(
                'gene_change', key=abs, ascending=False
            )
            print(f"\n  {'Coach':<25} {'Coord':<6} {'Years':<10} {'C.Gene':>7} "
                  f"{'HC':<6} {'Years':<10} {'HC.Gene':>7} {'Change':>8}")
            print(f"  {'-'*25} {'-'*6} {'-'*10} {'-'*7} {'-'*6} {'-'*10} {'-'*7} {'-'*8}")
            for _, row in subset.iterrows():
                print(f"  {row['coach_name']:<25} {row['coord_team']:<6} "
                      f"{row['coord_years']:<10} {row['coord_era_gene']:>7.3f} "
                      f"{row['hc_team']:<6} {row['hc_years']:<10} "
                      f"{row['hc_era_gene']:>7.3f} {row['gene_change']:>+8.3f}")

        print("\n" + "=" * 80)

    # -----------------------------------------------------------------
    # Main Runner
    # -----------------------------------------------------------------

    def run(self, output_dir: str = "data/processed/coaching_genes"):
        """Run the full inheritance analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.load_data()

        transitions = self.build_transitions()

        # Analyze both sides
        def_results = self.analyze_defensive_inheritance(transitions)
        off_results = self.analyze_offensive_inheritance(transitions)

        all_results = pd.concat([def_results, off_results], ignore_index=True)

        if len(all_results) == 0:
            logger.warning("No valid inheritance transitions found!")
            return

        # Statistics
        statistics = self._compute_statistics(all_results)

        # Print to console
        self._print_results(all_results, statistics)

        # Save CSV
        csv_path = output_path / "gene_inheritance.csv"
        all_results.to_csv(csv_path, index=False)
        logger.info(f"\nSaved inheritance data to {csv_path}")

        # Save summary JSON
        summary = self._generate_summary(all_results, statistics)
        json_path = output_path / "gene_inheritance_summary.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved summary to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gene inheritance from coordinators to head coaches"
    )
    parser.add_argument('--min_years', type=int, default=1,
                        help='Minimum coordinator/HC years with gene data required (default: 1)')
    parser.add_argument('--output_dir', type=str, default='data/processed/coaching_genes',
                        help='Output directory for results')
    args = parser.parse_args()

    analyzer = InheritanceAnalyzer(min_years=args.min_years)
    analyzer.run(output_dir=args.output_dir)


if __name__ == '__main__':
    main()
