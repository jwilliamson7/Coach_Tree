#!/usr/bin/env python3
"""
Team Abbreviation Validation Script

This script audits all team abbreviation mappings across the NFL Coaching Tree project
to identify inconsistencies between data sources and ensure complete coverage.

Data Sources Analyzed:
1. data_constants.py - TEAM_FRANCHISE_MAPPINGS, SPOTRAC_TO_PFR_MAPPINGS
2. calculate_aggression_gene.py - normalize_team_abbr() function
3. extract_head_coaches.py - team_corrections mapping
4. build_coaching_tree.py - truncation logic for team abbreviations
5. relationships.csv - actual team abbreviations in use

Output:
- Master mapping table for all 32 current NFL teams
- Inconsistencies between mapping sources
- Coverage analysis for play-by-play to coach matching
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from crawlers.utils.data_constants import (
    TEAM_FRANCHISE_MAPPINGS,
    SPOTRAC_TO_PFR_MAPPINGS,
    CURRENT_TEAM_ABBREVIATIONS
)


# Define the current 32 NFL teams with all their common abbreviation formats
CURRENT_NFL_TEAMS = {
    # AFC East
    'Buffalo Bills': {'standard': 'BUF', 'pfr': 'buf', 'nflfastR': 'BUF'},
    'Miami Dolphins': {'standard': 'MIA', 'pfr': 'mia', 'nflfastR': 'MIA'},
    'New England Patriots': {'standard': 'NE', 'pfr': 'nwe', 'nflfastR': 'NE'},
    'New York Jets': {'standard': 'NYJ', 'pfr': 'nyj', 'nflfastR': 'NYJ'},

    # AFC North
    'Baltimore Ravens': {'standard': 'BAL', 'pfr': 'rav', 'nflfastR': 'BAL'},
    'Cincinnati Bengals': {'standard': 'CIN', 'pfr': 'cin', 'nflfastR': 'CIN'},
    'Cleveland Browns': {'standard': 'CLE', 'pfr': 'cle', 'nflfastR': 'CLE'},
    'Pittsburgh Steelers': {'standard': 'PIT', 'pfr': 'pit', 'nflfastR': 'PIT'},

    # AFC South
    'Houston Texans': {'standard': 'HOU', 'pfr': 'htx', 'nflfastR': 'HOU'},
    'Indianapolis Colts': {'standard': 'IND', 'pfr': 'clt', 'nflfastR': 'IND'},
    'Jacksonville Jaguars': {'standard': 'JAX', 'pfr': 'jax', 'nflfastR': 'JAX'},
    'Tennessee Titans': {'standard': 'TEN', 'pfr': 'oti', 'nflfastR': 'TEN'},

    # AFC West
    'Denver Broncos': {'standard': 'DEN', 'pfr': 'den', 'nflfastR': 'DEN'},
    'Kansas City Chiefs': {'standard': 'KC', 'pfr': 'kan', 'nflfastR': 'KC'},
    'Las Vegas Raiders': {'standard': 'LV', 'pfr': 'rai', 'nflfastR': 'LV'},
    'Los Angeles Chargers': {'standard': 'LAC', 'pfr': 'sdg', 'nflfastR': 'LAC'},

    # NFC East
    'Dallas Cowboys': {'standard': 'DAL', 'pfr': 'dal', 'nflfastR': 'DAL'},
    'New York Giants': {'standard': 'NYG', 'pfr': 'nyg', 'nflfastR': 'NYG'},
    'Philadelphia Eagles': {'standard': 'PHI', 'pfr': 'phi', 'nflfastR': 'PHI'},
    'Washington Commanders': {'standard': 'WAS', 'pfr': 'was', 'nflfastR': 'WAS'},

    # NFC North
    'Chicago Bears': {'standard': 'CHI', 'pfr': 'chi', 'nflfastR': 'CHI'},
    'Detroit Lions': {'standard': 'DET', 'pfr': 'det', 'nflfastR': 'DET'},
    'Green Bay Packers': {'standard': 'GB', 'pfr': 'gnb', 'nflfastR': 'GB'},
    'Minnesota Vikings': {'standard': 'MIN', 'pfr': 'min', 'nflfastR': 'MIN'},

    # NFC South
    'Atlanta Falcons': {'standard': 'ATL', 'pfr': 'atl', 'nflfastR': 'ATL'},
    'Carolina Panthers': {'standard': 'CAR', 'pfr': 'car', 'nflfastR': 'CAR'},
    'New Orleans Saints': {'standard': 'NO', 'pfr': 'nor', 'nflfastR': 'NO'},
    'Tampa Bay Buccaneers': {'standard': 'TB', 'pfr': 'tam', 'nflfastR': 'TB'},

    # NFC West
    'Arizona Cardinals': {'standard': 'ARI', 'pfr': 'crd', 'nflfastR': 'ARI'},
    'Los Angeles Rams': {'standard': 'LAR', 'pfr': 'ram', 'nflfastR': 'LA'},
    'San Francisco 49ers': {'standard': 'SF', 'pfr': 'sfo', 'nflfastR': 'SF'},
    'Seattle Seahawks': {'standard': 'SEA', 'pfr': 'sea', 'nflfastR': 'SEA'},
}


# Mapping from calculate_aggression_gene.py normalize_team_abbr()
AGGRESSION_GENE_MAPPING = {
    'GB': 'GNB',
    'KC': 'KAN',
    'LA': 'LAR',
    'LV': 'LVR',
    'NO': 'NOR',
    'NE': 'NWE',
    'TEN': 'OTI',
    'ARI': 'CRD',
    'LAC': 'SDG',
    'SF': 'SFO',
    'TB': 'TAM',
    'IND': 'CLT',
    'BAL': 'RAV',
    'HOU': 'HTX',
    'OAK': 'RAI',
    'SD': 'SDG',
    'STL': 'STL',
    # Teams that stay the same
    'ATL': 'ATL', 'BUF': 'BUF', 'CAR': 'CAR', 'CHI': 'CHI',
    'CIN': 'CIN', 'CLE': 'CLE', 'DAL': 'DAL', 'DEN': 'DEN',
    'DET': 'DET', 'JAX': 'JAX', 'MIA': 'MIA', 'MIN': 'MIN',
    'NYG': 'NYG', 'NYJ': 'NYJ', 'PHI': 'PHI', 'PIT': 'PIT',
    'SEA': 'SEA', 'WAS': 'WAS'
}


# Mapping from extract_head_coaches.py team_corrections
EXTRACT_HC_MAPPING = {
    'BAL': 'RAV',
    'HOU': 'HTX',
    'LAC': 'SDG',
    'LAS': 'RAI',
    'TEN': 'OTI',
    'IND': 'CLT',
    'ARI': 'CRD',
    'GB': 'GNB',
    'KC': 'KAN',
    'NE': 'NWE',
    'NO': 'NOR',
    'SF': 'SFO',
    'TB': 'TAM',
    'WAS': 'WAS',
    'LV': 'RAI',
    'OAK': 'RAI'
}


def validate_team_mappings():
    """Main validation function for team abbreviation mappings"""

    print("=" * 80)
    print("TEAM ABBREVIATION VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    issues = []

    # 1. Build master mapping table
    print("1. MASTER MAPPING TABLE FOR ALL 32 NFL TEAMS")
    print("-" * 80)
    print(f"{'Team Name':<30} {'nflfastR':<10} {'PFR':<10} {'Agg. Gene':<10} {'Extract HC':<10}")
    print("-" * 80)

    for team_name, abbrevs in CURRENT_NFL_TEAMS.items():
        nflfastr = abbrevs['nflfastR']
        pfr = abbrevs['pfr'].upper()

        # Get aggression gene mapping
        agg_result = AGGRESSION_GENE_MAPPING.get(nflfastr, nflfastr)

        # Get extract_head_coaches mapping
        ext_result = EXTRACT_HC_MAPPING.get(nflfastr, nflfastr)

        # Check for consistency
        consistent = (agg_result.upper() == pfr.upper() == ext_result.upper())
        marker = "" if consistent else " [!]"

        print(f"{team_name:<30} {nflfastr:<10} {pfr:<10} {agg_result:<10} {ext_result:<10}{marker}")

        if not consistent:
            issues.append({
                'team': team_name,
                'issue': 'Inconsistent mappings',
                'details': f"nflfastR={nflfastr}, PFR={pfr}, AggGene={agg_result}, ExtractHC={ext_result}"
            })

    print()

    # 2. Check relationships.csv for truncation issues
    print("2. RELATIONSHIPS.CSV TEAM ABBREVIATION ANALYSIS")
    print("-" * 80)

    relationships_path = Path("data/processed/coaching_tree/relationships.csv")
    if relationships_path.exists():
        rel_df = pd.read_csv(relationships_path)
        unique_teams = rel_df['team'].unique()

        print(f"Total unique team abbreviations in relationships.csv: {len(unique_teams)}")
        print()

        # Group by length
        truncated_teams = [t for t in unique_teams if len(str(t)) == 3]
        proper_teams = [t for t in unique_teams if len(str(t)) > 3]

        print(f"3-character (truncated) teams: {len(truncated_teams)}")
        print(f"Longer abbreviations: {len(proper_teams)}")
        print()

        # Show truncated teams
        print("Truncated team abbreviations found:")
        truncated_sorted = sorted(truncated_teams)
        for i, team in enumerate(truncated_sorted):
            if i > 0 and i % 8 == 0:
                print()
            print(f"  {team}", end="")
        print("\n")

        # Identify problematic truncations
        print("Problematic truncations (ambiguous or non-standard):")
        problematic = {
            'new': 'Could be NYG, NYJ, NWE, or NOR',
            'los': 'Could be LAR or LAC (now)',
            'tam': 'Tampa Bay - non-standard',
            'kan': 'Kansas City - non-standard',
            'san': 'Could be SF or SD',
            'gnb': 'Green Bay - non-standard short form',
            'nor': 'New Orleans - non-standard short form',
        }

        found_problems = []
        for team in truncated_sorted:
            if team.lower() in problematic:
                print(f"  {team}: {problematic[team.lower()]}")
                found_problems.append(team)
                issues.append({
                    'team': team,
                    'issue': 'Ambiguous truncation',
                    'details': problematic[team.lower()]
                })

        if not found_problems:
            print("  None found")
        print()

        # Check for 'new' team occurrences (most problematic)
        if 'new' in [t.lower() for t in unique_teams]:
            new_rows = rel_df[rel_df['team'].str.lower() == 'new']
            print("Analysis of 'new' team abbreviation:")
            print(f"  Total rows with 'new': {len(new_rows)}")

            # Try to identify which team based on coaches
            if 'parent_name' in new_rows.columns:
                coaches = new_rows['parent_name'].unique()[:10]
                print(f"  Sample head coaches: {list(coaches)}")
            print()
    else:
        print(f"WARNING: relationships.csv not found at {relationships_path}")
        issues.append({
            'team': 'N/A',
            'issue': 'Missing file',
            'details': f'relationships.csv not found at {relationships_path}'
        })
    print()

    # 3. Check team_year_head_coaches.csv for consistency
    print("3. HEAD COACH MAPPING TEAM ABBREVIATION ANALYSIS")
    print("-" * 80)

    hc_mapping_path = Path("data/processed/Coaching/team_year_head_coaches.csv")
    if hc_mapping_path.exists():
        hc_df = pd.read_csv(hc_mapping_path)
        unique_hc_teams = hc_df['Team'].unique()

        print(f"Total unique team abbreviations in head coach mapping: {len(unique_hc_teams)}")

        # Show sample
        print("Sample team abbreviations (sorted):")
        sorted_teams = sorted(unique_hc_teams)
        for i, team in enumerate(sorted_teams[:20]):
            if i > 0 and i % 8 == 0:
                print()
            print(f"  {team}", end="")
        print("\n")

        # Check which format they use
        three_char = [t for t in unique_hc_teams if len(str(t)) == 3]
        other = [t for t in unique_hc_teams if len(str(t)) != 3]

        print(f"3-character abbreviations: {len(three_char)}")
        print(f"Other length abbreviations: {len(other)}")

        if other:
            print(f"Non-3-char teams: {sorted(other)}")
    else:
        print(f"WARNING: team_year_head_coaches.csv not found at {hc_mapping_path}")
    print()

    # 4. Validate SPOTRAC_TO_PFR_MAPPINGS covers all teams
    print("4. SPOTRAC_TO_PFR_MAPPINGS COVERAGE CHECK")
    print("-" * 80)

    # Extract all target PFR abbreviations
    pfr_targets = set(v.lower() for v in SPOTRAC_TO_PFR_MAPPINGS.values())

    # Expected PFR abbreviations from master list
    expected_pfr = set(team['pfr'].lower() for team in CURRENT_NFL_TEAMS.values())

    missing_in_spotrac = expected_pfr - pfr_targets
    extra_in_spotrac = pfr_targets - expected_pfr

    print(f"PFR abbreviations in SPOTRAC mapping: {len(pfr_targets)}")
    print(f"Expected current team PFR abbreviations: {len(expected_pfr)}")

    if missing_in_spotrac:
        print(f"\nMISSING from SPOTRAC mapping: {missing_in_spotrac}")
        for team in missing_in_spotrac:
            issues.append({
                'team': team,
                'issue': 'Missing from SPOTRAC_TO_PFR_MAPPINGS',
                'details': f'PFR abbreviation {team} not covered in spotrac mapping'
            })
    else:
        print("All current teams covered in SPOTRAC mapping")

    if extra_in_spotrac:
        print(f"\nExtra in SPOTRAC mapping (historical teams): {extra_in_spotrac}")
    print()

    # 5. Cross-validate aggression gene mapping
    print("5. AGGRESSION GENE MAPPING VALIDATION")
    print("-" * 80)

    # Check that all nflfastR abbreviations are mapped
    nflfastr_abbrevs = set(team['nflfastR'] for team in CURRENT_NFL_TEAMS.values())
    mapped_in_agg = set(AGGRESSION_GENE_MAPPING.keys())

    unmapped = nflfastr_abbrevs - mapped_in_agg
    if unmapped:
        print(f"WARNING: nflfastR abbreviations not mapped in aggression gene: {unmapped}")
        for team in unmapped:
            issues.append({
                'team': team,
                'issue': 'Missing from normalize_team_abbr()',
                'details': f'nflfastR abbreviation {team} not covered in aggression gene mapping'
            })
    else:
        print("All current nflfastR abbreviations are mapped")

    # Verify output format consistency
    agg_outputs = set(AGGRESSION_GENE_MAPPING.values())
    print(f"\nAggression gene output abbreviations: {len(agg_outputs)} unique values")

    # Check case consistency
    uppercase_outputs = [o for o in agg_outputs if o.isupper()]
    lowercase_outputs = [o for o in agg_outputs if o.islower()]
    mixed_outputs = [o for o in agg_outputs if not o.isupper() and not o.islower()]

    print(f"  Uppercase: {len(uppercase_outputs)}")
    print(f"  Lowercase: {len(lowercase_outputs)}")
    print(f"  Mixed case: {len(mixed_outputs)}")

    if uppercase_outputs and lowercase_outputs:
        print("WARNING: Mixed case output format in aggression gene mapping!")
        issues.append({
            'team': 'All',
            'issue': 'Case inconsistency',
            'details': 'Aggression gene mapping outputs both upper and lower case abbreviations'
        })
    print()

    # 6. Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if issues:
        print(f"\nTotal issues found: {len(issues)}")
        print("\nIssues by type:")
        issue_types = defaultdict(list)
        for issue in issues:
            issue_types[issue['issue']].append(issue)

        for issue_type, items in issue_types.items():
            print(f"\n  {issue_type}: {len(items)}")
            for item in items[:3]:  # Show first 3
                print(f"    - {item['team']}: {item['details']}")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")
    else:
        print("\nNo issues found!")

    print()

    # 7. Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("1. CRITICAL: Fix 3-character truncation in build_coaching_tree.py")
    print("   - Line 130 truncates team names to 3 characters")
    print("   - This creates ambiguous teams like 'new' (NYG, NYJ, NWE, NOR)")
    print("   - Recommendation: Use full PFR-style abbreviations")
    print()
    print("2. Standardize case: Use uppercase for all abbreviations")
    print("   - PFR uses lowercase internally, but uppercase is more standard")
    print("   - Update AGGRESSION_GENE_MAPPING outputs to uppercase")
    print()
    print("3. Create single source of truth:")
    print("   - Add MASTER_TEAM_MAPPING to data_constants.py")
    print("   - All scripts should reference this single mapping")
    print()
    print("4. For relationships.csv:")
    print("   - Need to rebuild with correct abbreviations")
    print("   - Or add mapping layer to disambiguate 'new', 'los', etc.")
    print()

    return issues


def create_master_mapping_csv(output_path: str = "data/processed/validation/master_team_mapping.csv"):
    """Create a master mapping CSV file"""

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for team_name, abbrevs in CURRENT_NFL_TEAMS.items():
        row = {
            'team_name': team_name,
            'nflfastR': abbrevs['nflfastR'],
            'pfr_full': abbrevs['pfr'].upper(),
            'pfr_3char': abbrevs['pfr'][:3].lower() if len(abbrevs['pfr']) >= 3 else abbrevs['pfr'].lower(),
            'aggression_gene_output': AGGRESSION_GENE_MAPPING.get(abbrevs['nflfastR'], abbrevs['nflfastR']),
            'extract_hc_output': EXTRACT_HC_MAPPING.get(abbrevs['nflfastR'], abbrevs['nflfastR'])
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Master mapping saved to: {output_path}")

    return df


if __name__ == "__main__":
    # Run validation
    issues = validate_team_mappings()

    # Create master mapping CSV
    print()
    master_df = create_master_mapping_csv()

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    # Exit with error code if issues found
    sys.exit(1 if issues else 0)
