#!/usr/bin/env python3
"""
Play-by-Play Coverage Validation Script

This script validates the play-by-play to coach attribution rate
and examines the quality of the aggression gene data.

Validation areas:
1. Coach-year coverage analysis
2. Play volume per coach-season
3. Missing data patterns
4. Data quality checks

Output:
- Coverage statistics
- Potential data quality issues
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


def load_aggression_data():
    """Load aggression gene data files"""

    data = {}

    # Load by-year data
    by_year_path = Path("data/processed/coaching_genes/aggression_gene_by_year.csv")
    if by_year_path.exists():
        data['by_year'] = pd.read_csv(by_year_path)
        print(f"Loaded by_year data: {len(data['by_year'])} rows")
    else:
        print(f"WARNING: aggression_gene_by_year.csv not found")
        data['by_year'] = pd.DataFrame()

    # Load by-coach data
    by_coach_path = Path("data/processed/coaching_genes/aggression_gene_by_coach.csv")
    if by_coach_path.exists():
        data['by_coach'] = pd.read_csv(by_coach_path)
        print(f"Loaded by_coach data: {len(data['by_coach'])} rows")
    else:
        print(f"WARNING: aggression_gene_by_coach.csv not found")
        data['by_coach'] = pd.DataFrame()

    # Load head coach mapping
    hc_mapping_path = Path("data/processed/Coaching/team_year_head_coaches.csv")
    if hc_mapping_path.exists():
        data['hc_mapping'] = pd.read_csv(hc_mapping_path)
        print(f"Loaded head coach mapping: {len(data['hc_mapping'])} rows")
    else:
        print(f"WARNING: team_year_head_coaches.csv not found")
        data['hc_mapping'] = pd.DataFrame()

    return data


def analyze_coach_year_coverage(data: Dict) -> Dict:
    """Analyze coverage of coach-years in the aggression data"""

    results = {
        'issues': [],
        'statistics': {}
    }

    if data['by_year'].empty:
        results['issues'].append({
            'type': 'missing_data',
            'description': 'No aggression gene by-year data available'
        })
        return results

    df = data['by_year']

    # Basic statistics
    results['statistics']['total_coach_years'] = len(df)
    results['statistics']['unique_coaches'] = df['head_coach'].nunique()
    results['statistics']['years_covered'] = sorted(df['season'].unique().tolist())
    results['statistics']['min_year'] = int(df['season'].min())
    results['statistics']['max_year'] = int(df['season'].max())

    # Expected coach-years (32 teams * years)
    num_years = len(results['statistics']['years_covered'])
    expected_coach_years = 32 * num_years  # 32 teams per year

    results['statistics']['expected_coach_years'] = expected_coach_years
    results['statistics']['coverage_rate'] = len(df) / expected_coach_years

    print("\n" + "=" * 80)
    print("COACH-YEAR COVERAGE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal coach-years in data: {len(df)}")
    print(f"Unique coaches: {df['head_coach'].nunique()}")
    print(f"Years covered: {results['statistics']['min_year']} - {results['statistics']['max_year']}")
    print(f"Expected coach-years (32 teams x {num_years} years): {expected_coach_years}")
    print(f"Coverage rate: {results['statistics']['coverage_rate']:.1%}")

    # Check for years with low coverage
    print("\nCoach-years per season:")
    season_counts = df.groupby('season').size()
    for season, count in season_counts.items():
        expected = 32
        pct = count / expected * 100
        marker = " [!]" if count < 28 else ""
        print(f"  {int(season)}: {count} coaches ({pct:.0f}% of 32){marker}")

        if count < 28:
            results['issues'].append({
                'type': 'low_coverage',
                'season': int(season),
                'coach_count': count,
                'expected': 32
            })

    return results


def analyze_play_volumes(data: Dict) -> Dict:
    """Analyze play volumes per coach-season"""

    results = {
        'issues': [],
        'statistics': {}
    }

    if data['by_year'].empty:
        return results

    df = data['by_year']

    print("\n" + "=" * 80)
    print("PLAY VOLUME ANALYSIS")
    print("=" * 80)

    # Check available play count columns
    play_columns = [
        'fourth_down_plays',
        'run_pass_plays',
        'pass_plays',
        'conversion_attempts',
        'total_plays'
    ]

    available_cols = [col for col in play_columns if col in df.columns]
    print(f"\nAvailable play count columns: {available_cols}")

    for col in available_cols:
        if col in df.columns:
            print(f"\n{col}:")
            print(f"  Min: {df[col].min():.0f}")
            print(f"  Max: {df[col].max():.0f}")
            print(f"  Mean: {df[col].mean():.1f}")
            print(f"  Median: {df[col].median():.1f}")

            # Flag coach-seasons with very low play counts
            if col == 'fourth_down_plays':
                threshold = 30  # At least 30 4th down decisions
            elif col == 'run_pass_plays':
                threshold = 200  # At least 200 run/pass plays
            elif col == 'pass_plays':
                threshold = 150  # At least 150 passes
            elif col == 'conversion_attempts':
                threshold = 20  # At least 20 conversion attempts
            else:
                threshold = 500  # At least 500 total plays

            low_count = df[df[col] < threshold]
            if len(low_count) > 0:
                print(f"  Coach-seasons below {threshold}: {len(low_count)}")
                for _, row in low_count.head(5).iterrows():
                    print(f"    {row['head_coach']} ({int(row['season'])}): {int(row[col])}")

                results['issues'].append({
                    'type': 'low_play_count',
                    'column': col,
                    'threshold': threshold,
                    'count': len(low_count)
                })

    # Total plays summary
    if 'total_plays' in df.columns:
        results['statistics']['total_plays_analyzed'] = int(df['total_plays'].sum())
        print(f"\nTotal plays analyzed across all coach-years: {results['statistics']['total_plays_analyzed']:,}")

    return results


def analyze_missing_patterns(data: Dict) -> Dict:
    """Analyze patterns of missing data"""

    results = {
        'issues': [],
        'statistics': {}
    }

    if data['by_year'].empty:
        return results

    df = data['by_year']

    print("\n" + "=" * 80)
    print("MISSING DATA PATTERNS")
    print("=" * 80)

    # Check for missing values in key columns
    key_columns = [
        'fourth_down_aggression',
        'pass_heavy_aggression',
        'deep_pass_aggression',
        'two_point_aggression',
        'composite_aggression'
    ]

    available_key_cols = [col for col in key_columns if col in df.columns]

    print("\nMissing values in aggression columns:")
    for col in available_key_cols:
        missing = df[col].isna().sum()
        missing_pct = missing / len(df) * 100
        print(f"  {col}: {missing} ({missing_pct:.1f}%)")

        if missing > 0:
            results['issues'].append({
                'type': 'missing_values',
                'column': col,
                'count': int(missing),
                'percentage': missing_pct
            })

    # Check for infinite values
    print("\nInfinite values in aggression columns:")
    for col in available_key_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            print(f"  {col}: {inf_count}")
            results['issues'].append({
                'type': 'infinite_values',
                'column': col,
                'count': int(inf_count)
            })
        else:
            print(f"  {col}: 0")

    return results


def analyze_data_quality(data: Dict) -> Dict:
    """Analyze overall data quality"""

    results = {
        'issues': [],
        'statistics': {}
    }

    if data['by_year'].empty:
        return results

    df = data['by_year']

    print("\n" + "=" * 80)
    print("DATA QUALITY CHECKS")
    print("=" * 80)

    # Check for duplicate coach-season combinations
    duplicates = df.duplicated(subset=['head_coach', 'season'], keep=False)
    if duplicates.any():
        dup_count = duplicates.sum()
        print(f"\nDuplicate coach-season combinations: {dup_count}")
        dup_df = df[duplicates].sort_values(['head_coach', 'season'])
        print(dup_df[['head_coach', 'season']].head(10))
        results['issues'].append({
            'type': 'duplicate_entries',
            'count': int(dup_count)
        })
    else:
        print("\nNo duplicate coach-season combinations found")

    # Check aggression score distribution
    if 'composite_aggression' in df.columns:
        print("\nComposite aggression score distribution:")
        print(f"  Mean: {df['composite_aggression'].mean():.4f}")
        print(f"  Std: {df['composite_aggression'].std():.4f}")
        print(f"  Min: {df['composite_aggression'].min():.4f}")
        print(f"  Max: {df['composite_aggression'].max():.4f}")

        # Check for extreme outliers (>4 std from mean)
        mean = df['composite_aggression'].mean()
        std = df['composite_aggression'].std()
        outliers = df[abs(df['composite_aggression'] - mean) > 4 * std]

        if len(outliers) > 0:
            print(f"\n  Extreme outliers (>4 std): {len(outliers)}")
            for _, row in outliers.iterrows():
                print(f"    {row['head_coach']} ({int(row['season'])}): {row['composite_aggression']:.4f}")

            results['issues'].append({
                'type': 'extreme_outliers',
                'count': len(outliers)
            })

    # Check rate columns are in valid range [0, 1]
    rate_columns = ['actual_decision', 'predicted_go_rate', 'actual_pass',
                    'predicted_pass_rate', 'actual_beyond', 'predicted_beyond_rate',
                    'actual_two_point', 'predicted_two_point_rate']

    print("\nRate column validation (should be in [0, 1]):")
    for col in rate_columns:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range > 0:
                print(f"  {col}: {out_of_range} values out of range [!]")
                results['issues'].append({
                    'type': 'invalid_rate',
                    'column': col,
                    'count': int(out_of_range)
                })
            else:
                print(f"  {col}: OK")

    return results


def cross_validate_with_hc_mapping(data: Dict) -> Dict:
    """Cross-validate aggression data against head coach mapping"""

    results = {
        'issues': [],
        'statistics': {}
    }

    if data['by_year'].empty or data['hc_mapping'].empty:
        return results

    agg_df = data['by_year']
    hc_df = data['hc_mapping']

    print("\n" + "=" * 80)
    print("CROSS-VALIDATION WITH HEAD COACH MAPPING")
    print("=" * 80)

    # Get the year range overlap
    agg_years = set(agg_df['season'].unique())
    hc_years = set(hc_df['Year'].unique())
    common_years = agg_years & hc_years

    print(f"\nAggression data years: {min(agg_years)} - {max(agg_years)}")
    print(f"HC mapping years: {min(hc_years)} - {max(hc_years)}")
    print(f"Common years: {len(common_years)}")

    # For each common year, check coach coverage
    print("\nCoverage by year (aggression vs HC mapping):")
    for year in sorted(common_years):
        agg_coaches = set(agg_df[agg_df['season'] == year]['head_coach'].unique())
        hc_coaches = set(hc_df[hc_df['Year'] == year]['Primary_Coach'].unique())

        in_agg_only = agg_coaches - hc_coaches
        in_hc_only = hc_coaches - agg_coaches
        common = agg_coaches & hc_coaches

        if in_hc_only:
            print(f"  {int(year)}: {len(agg_coaches)} agg, {len(hc_coaches)} hc, "
                  f"{len(in_hc_only)} missing from agg")

            if len(in_hc_only) <= 5:
                for coach in in_hc_only:
                    print(f"    - {coach}")

            results['issues'].append({
                'type': 'missing_coaches_in_agg',
                'year': int(year),
                'coaches': list(in_hc_only)[:10]
            })

    # Overall statistics
    agg_coaches_all = set(agg_df['head_coach'].unique())
    hc_coaches_all = set(hc_df[hc_df['Year'].isin(agg_years)]['Primary_Coach'].unique())

    results['statistics']['coaches_in_agg'] = len(agg_coaches_all)
    results['statistics']['coaches_in_hc_mapping'] = len(hc_coaches_all)
    results['statistics']['coaches_in_both'] = len(agg_coaches_all & hc_coaches_all)
    results['statistics']['coaches_only_in_agg'] = len(agg_coaches_all - hc_coaches_all)
    results['statistics']['coaches_only_in_hc'] = len(hc_coaches_all - agg_coaches_all)

    print(f"\nOverall coach comparison:")
    print(f"  Coaches in aggression data: {results['statistics']['coaches_in_agg']}")
    print(f"  Coaches in HC mapping: {results['statistics']['coaches_in_hc_mapping']}")
    print(f"  Coaches in both: {results['statistics']['coaches_in_both']}")
    print(f"  Only in aggression data: {results['statistics']['coaches_only_in_agg']}")
    print(f"  Only in HC mapping: {results['statistics']['coaches_only_in_hc']}")

    return results


def main():
    """Main validation function"""

    print("=" * 80)
    print("PLAY-BY-PLAY COVERAGE VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    print("Loading data...")
    data = load_aggression_data()

    all_results = {
        'coverage': None,
        'play_volumes': None,
        'missing_patterns': None,
        'data_quality': None,
        'cross_validation': None
    }

    # Run analyses
    all_results['coverage'] = analyze_coach_year_coverage(data)
    all_results['play_volumes'] = analyze_play_volumes(data)
    all_results['missing_patterns'] = analyze_missing_patterns(data)
    all_results['data_quality'] = analyze_data_quality(data)
    all_results['cross_validation'] = cross_validate_with_hc_mapping(data)

    # Compile all issues
    all_issues = []
    for category, results in all_results.items():
        if results and 'issues' in results:
            for issue in results['issues']:
                issue['category'] = category
                all_issues.append(issue)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if all_issues:
        print(f"\nTotal issues found: {len(all_issues)}")

        # Group by category
        issue_by_category = defaultdict(list)
        for issue in all_issues:
            issue_by_category[issue['category']].append(issue)

        print("\nIssues by category:")
        for category, items in issue_by_category.items():
            print(f"\n  {category}: {len(items)}")
            for item in items[:3]:
                issue_type = item.get('type', 'unknown')
                print(f"    - {issue_type}")
    else:
        print("\nNo critical issues found!")

    # Save results
    output_dir = Path("data/processed/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "pbp_coverage_validation.json"
    with open(results_path, 'w') as f:
        json.dump({
            'issues': all_issues,
            'statistics': {k: v.get('statistics', {}) for k, v in all_results.items() if v}
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return all_issues


if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues else 0)
