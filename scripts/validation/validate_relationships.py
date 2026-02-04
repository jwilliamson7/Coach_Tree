#!/usr/bin/env python3
"""
Coaching Tree Relationship Validation Script

This script validates the integrity of the coaching tree relationship data:
1. Identifies orphan coaches (position coaches with no coordinator/HC parent)
2. Checks for duplicate coach-year-role-parent combinations
3. Verifies interim role exclusion is consistent
4. Audits sample relationships manually against known coaching data

Output:
- Relationship integrity report
- List of potential issues for manual review
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


def load_coaching_tree_data():
    """Load all coaching tree data files"""

    data = {}

    # Load relationships
    rel_path = Path("data/processed/coaching_tree/relationships.csv")
    if rel_path.exists():
        data['relationships'] = pd.read_csv(rel_path)
        print(f"Loaded relationships: {len(data['relationships'])} rows")
    else:
        print(f"WARNING: relationships.csv not found")
        data['relationships'] = pd.DataFrame()

    # Load coaches
    coaches_path = Path("data/processed/coaching_tree/coaches.json")
    if coaches_path.exists():
        with open(coaches_path, 'r') as f:
            data['coaches'] = json.load(f)
        print(f"Loaded coaches: {len(data['coaches'])} coaches")
    else:
        print(f"WARNING: coaches.json not found")
        data['coaches'] = {}

    # Load team rosters
    rosters_path = Path("data/processed/coaching_tree/team_rosters.json")
    if rosters_path.exists():
        with open(rosters_path, 'r') as f:
            data['rosters'] = json.load(f)
        print(f"Loaded rosters: {len(data['rosters'])} years")
    else:
        print(f"WARNING: team_rosters.json not found")
        data['rosters'] = {}

    return data


def check_orphan_coaches(data: Dict) -> List[Dict]:
    """
    Identify orphan coaches - position coaches who don't have an expected
    coordinator or head coach parent in their team for that year.
    """
    issues = []

    if data['relationships'].empty or not data['rosters']:
        return issues

    rel_df = data['relationships']

    # Get all position coaches who report to coordinators
    position_to_coordinator = rel_df[rel_df['relationship_type'] == 'position_to_coordinator']
    position_to_hc = rel_df[rel_df['relationship_type'] == 'position_to_hc']

    # Find position coaches who only have position_to_hc relationships but no coordinator
    pos_hc_only_coaches = set(position_to_hc['child_id'].unique()) - set(position_to_coordinator['child_id'].unique())

    # Check these coaches - are they offensive/defensive coaches without a coordinator?
    if pos_hc_only_coaches:
        print(f"\nPosition coaches without coordinator parent: {len(pos_hc_only_coaches)}")

        # Sample a few to understand
        sample_coaches = list(pos_hc_only_coaches)[:5]
        for coach_id in sample_coaches:
            coach_records = position_to_hc[position_to_hc['child_id'] == coach_id]
            sample_row = coach_records.iloc[0]
            issues.append({
                'type': 'orphan_coach',
                'coach_id': coach_id,
                'coach_name': sample_row['child_name'],
                'sample_role': sample_row['child_role'],
                'sample_year': sample_row['year'],
                'sample_team': sample_row['team'],
                'total_records': len(coach_records)
            })

    return issues


def check_duplicate_relationships(data: Dict) -> List[Dict]:
    """
    Check for duplicate relationships (same child-parent-year-type combo)
    """
    issues = []

    if data['relationships'].empty:
        return issues

    rel_df = data['relationships']

    # Check for exact duplicates
    dup_cols = ['year', 'child_id', 'parent_id', 'relationship_type']
    duplicates = rel_df[rel_df.duplicated(subset=dup_cols, keep=False)]

    if len(duplicates) > 0:
        print(f"\nExact duplicate relationships found: {len(duplicates)}")

        # Group by the duplicate columns
        dup_groups = duplicates.groupby(dup_cols).size().reset_index(name='count')
        dup_groups = dup_groups[dup_groups['count'] > 1]

        for _, row in dup_groups.head(10).iterrows():
            issues.append({
                'type': 'duplicate_relationship',
                'year': row['year'],
                'child_id': row['child_id'],
                'parent_id': row['parent_id'],
                'relationship_type': row['relationship_type'],
                'count': row['count']
            })
    else:
        print("\nNo exact duplicate relationships found")

    return issues


def check_interim_exclusion(data: Dict) -> List[Dict]:
    """
    Check that interim coaches are properly excluded from parent roles
    """
    issues = []

    if data['relationships'].empty:
        return issues

    rel_df = data['relationships']

    # Check for 'interim' in any role fields (case insensitive)
    interim_in_child = rel_df[rel_df['child_role'].str.lower().str.contains('interim', na=False)]
    interim_in_parent = rel_df[rel_df['parent_role'].str.lower().str.contains('interim', na=False)]

    if len(interim_in_child) > 0:
        print(f"\nInterim roles found in child_role: {len(interim_in_child)}")
        # Sample
        for _, row in interim_in_child.head(5).iterrows():
            issues.append({
                'type': 'interim_not_excluded',
                'field': 'child_role',
                'year': row['year'],
                'coach_name': row['child_name'],
                'role': row['child_role'],
                'team': row['team']
            })
    else:
        print("\nNo interim roles in child_role field - good")

    if len(interim_in_parent) > 0:
        print(f"\nInterim roles found in parent_role: {len(interim_in_parent)}")
        for _, row in interim_in_parent.head(5).iterrows():
            issues.append({
                'type': 'interim_not_excluded',
                'field': 'parent_role',
                'year': row['year'],
                'coach_name': row['parent_name'],
                'role': row['parent_role'],
                'team': row['team']
            })
    else:
        print("\nNo interim roles in parent_role field - good")

    return issues


def check_relationship_consistency(data: Dict) -> List[Dict]:
    """
    Check for logical consistency in relationships
    """
    issues = []

    if data['relationships'].empty:
        return issues

    rel_df = data['relationships']

    # Check 1: Coordinator should not be parent to another coordinator
    coord_to_coord = rel_df[
        (rel_df['relationship_type'] == 'coordinator_to_hc') &
        (rel_df['child_role'].str.contains('Coordinator', case=False, na=False)) &
        (rel_df['parent_role'].str.contains('Coordinator', case=False, na=False))
    ]

    if len(coord_to_coord) > 0:
        print(f"\nCoordinator-to-coordinator relationships (unexpected): {len(coord_to_coord)}")
        for _, row in coord_to_coord.head(3).iterrows():
            issues.append({
                'type': 'coordinator_to_coordinator',
                'year': row['year'],
                'child': f"{row['child_name']} ({row['child_role']})",
                'parent': f"{row['parent_name']} ({row['parent_role']})",
                'team': row['team']
            })

    # Check 2: A coach should not be their own parent
    self_refs = rel_df[rel_df['child_id'] == rel_df['parent_id']]
    if len(self_refs) > 0:
        print(f"\nSelf-referential relationships found: {len(self_refs)}")
        for _, row in self_refs.head(3).iterrows():
            issues.append({
                'type': 'self_reference',
                'year': row['year'],
                'coach_id': row['child_id'],
                'coach_name': row['child_name'],
                'team': row['team']
            })
    else:
        print("\nNo self-referential relationships - good")

    # Check 3: HC should not report to anyone
    hc_as_child = rel_df[
        (rel_df['child_role'] == 'Head Coach') &
        (rel_df['relationship_type'] != 'coordinator_to_hc')  # Should only be coordinators to HC
    ]

    if len(hc_as_child) > 0:
        print(f"\nHead Coach as child in non-coordinator relationships: {len(hc_as_child)}")
        for _, row in hc_as_child.head(3).iterrows():
            issues.append({
                'type': 'hc_as_child',
                'year': row['year'],
                'coach': row['child_name'],
                'parent': row['parent_name'],
                'relationship_type': row['relationship_type'],
                'team': row['team']
            })
    else:
        print("\nNo HC as child in unexpected relationships - good")

    return issues


def audit_sample_relationships(data: Dict) -> List[Dict]:
    """
    Manually audit a sample of relationships against known coaching history.
    These are well-documented relationships that should appear in our data.
    """
    issues = []

    if data['relationships'].empty:
        return issues

    rel_df = data['relationships']

    # Known relationships to verify (based on public knowledge)
    known_relationships = [
        {
            'description': 'Kyle Shanahan under Mike Shanahan (father)',
            'child': 'kyle_shanahan',
            'parent': 'mike_shanahan',
            'years': [2010, 2011, 2012, 2013],  # Washington years; 2008-09 Kyle was at Houston
            'relationship_type': 'coordinator_to_hc'
        },
        {
            'description': 'Sean McVay under Jay Gruden (Washington)',
            'child': 'sean_mcvay',
            'parent': 'jay_gruden',
            'years': [2014, 2015, 2016],
            'relationship_type': 'coordinator_to_hc'
        },
        {
            'description': 'Matt LaFleur under Sean McVay (Rams)',
            'child': 'matt_lafleur',
            'parent': 'sean_mcvay',
            'years': [2017],
            'relationship_type': 'coordinator_to_hc'
        },
        {
            'description': 'Bill Belichick under Bill Parcells (Giants/Patriots)',
            'child': 'bill_belichick',
            'parent': 'bill_parcells',
            'years': [1988, 1989, 1990, 1996],
            'relationship_type': 'coordinator_to_hc'
        },
        {
            'description': 'Nick Saban under Bill Belichick (Browns)',
            'child': 'nick_saban',
            'parent': 'bill_belichick',
            'years': [1991, 1992, 1993, 1994],
            'relationship_type': 'coordinator_to_hc'
        },
        {
            'description': 'Mike Tomlin under Tony Dungy (Bucs)',
            'child': 'mike_tomlin',
            'parent': 'tony_dungy',
            'years': [2001],  # Dungy fired after 2001; Gruden was HC 2002-2005
            'relationship_type': 'position_to_hc'
        },
        {
            'description': 'Andy Reid under Mike Holmgren (Packers)',
            'child': 'andy_reid',
            'parent': 'mike_holmgren',
            'years': [1992, 1993, 1994, 1995, 1996, 1997, 1998],
            'relationship_type': 'position_to_hc'
        }
    ]

    print("\n" + "=" * 80)
    print("SAMPLE RELATIONSHIP AUDIT")
    print("=" * 80)

    verified_count = 0
    missing_count = 0

    for known in known_relationships:
        print(f"\nChecking: {known['description']}")

        # Look for this relationship
        matches = rel_df[
            (rel_df['child_id'] == known['child']) &
            (rel_df['parent_id'] == known['parent'])
        ]

        if len(matches) > 0:
            found_years = sorted(matches['year'].unique())
            expected_years = known['years']

            matching_years = set(found_years) & set(expected_years)
            missing_years = set(expected_years) - set(found_years)

            if missing_years:
                print(f"  PARTIAL: Found {len(matching_years)}/{len(expected_years)} expected years")
                print(f"  Missing years: {sorted(missing_years)}")
                issues.append({
                    'type': 'partial_relationship',
                    'description': known['description'],
                    'found_years': found_years,
                    'expected_years': expected_years,
                    'missing_years': sorted(missing_years)
                })
                missing_count += 1
            else:
                print(f"  VERIFIED: Found all {len(expected_years)} expected years")
                verified_count += 1

            # Check relationship type
            rel_types = matches['relationship_type'].unique()
            if known['relationship_type'] not in rel_types:
                print(f"  WARNING: Expected type {known['relationship_type']}, found {rel_types}")
        else:
            print(f"  MISSING: No relationship found between {known['child']} and {known['parent']}")
            issues.append({
                'type': 'missing_relationship',
                'description': known['description'],
                'child': known['child'],
                'parent': known['parent'],
                'expected_years': known['years']
            })
            missing_count += 1

    print(f"\n\nSample Audit Summary:")
    print(f"  Verified: {verified_count}")
    print(f"  Issues: {missing_count}")

    return issues


def analyze_relationship_statistics(data: Dict):
    """Print overall statistics about relationships"""

    if data['relationships'].empty:
        return

    rel_df = data['relationships']

    print("\n" + "=" * 80)
    print("RELATIONSHIP STATISTICS")
    print("=" * 80)

    print(f"\nTotal relationships: {len(rel_df)}")

    # By type
    print("\nRelationships by type:")
    type_counts = rel_df['relationship_type'].value_counts()
    for rel_type, count in type_counts.items():
        print(f"  {rel_type}: {count:,}")

    # By decade
    rel_df['decade'] = (rel_df['year'] // 10) * 10
    print("\nRelationships by decade:")
    decade_counts = rel_df['decade'].value_counts().sort_index()
    for decade, count in decade_counts.items():
        print(f"  {int(decade)}s: {count:,}")

    # Unique coaches
    unique_children = rel_df['child_id'].nunique()
    unique_parents = rel_df['parent_id'].nunique()
    print(f"\nUnique children (mentees): {unique_children}")
    print(f"Unique parents (mentors): {unique_parents}")

    # Most connected mentors
    print("\nTop 10 Mentors (by number of relationships):")
    top_mentors = rel_df.groupby(['parent_id', 'parent_name']).size().sort_values(ascending=False).head(10)
    for (parent_id, parent_name), count in top_mentors.items():
        print(f"  {parent_name}: {count} relationships")

    # Most prolific protege makers (by unique children)
    print("\nTop 10 Mentors (by unique mentees):")
    mentor_children = rel_df.groupby(['parent_id', 'parent_name'])['child_id'].nunique().sort_values(ascending=False).head(10)
    for (parent_id, parent_name), count in mentor_children.items():
        print(f"  {parent_name}: {count} unique mentees")


def main():
    """Main validation function"""

    print("=" * 80)
    print("COACHING TREE RELATIONSHIP VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    print("Loading data...")
    data = load_coaching_tree_data()

    all_issues = []

    # Run validations
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    print("\n1. Checking for orphan coaches...")
    issues = check_orphan_coaches(data)
    all_issues.extend(issues)

    print("\n2. Checking for duplicate relationships...")
    issues = check_duplicate_relationships(data)
    all_issues.extend(issues)

    print("\n3. Checking interim role exclusion...")
    issues = check_interim_exclusion(data)
    all_issues.extend(issues)

    print("\n4. Checking relationship consistency...")
    issues = check_relationship_consistency(data)
    all_issues.extend(issues)

    print("\n5. Auditing sample relationships...")
    issues = audit_sample_relationships(data)
    all_issues.extend(issues)

    # Print statistics
    analyze_relationship_statistics(data)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if all_issues:
        print(f"\nTotal issues found: {len(all_issues)}")

        # Group by type
        issue_types = defaultdict(list)
        for issue in all_issues:
            issue_types[issue['type']].append(issue)

        print("\nIssues by type:")
        for issue_type, items in issue_types.items():
            print(f"\n  {issue_type}: {len(items)}")
            for item in items[:3]:  # Show first 3
                desc = item.get('description', item.get('coach_name', item.get('coach_id', 'Unknown')))
                print(f"    - {desc}")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more")
    else:
        print("\nNo issues found!")

    # Save issues to file
    output_dir = Path("data/processed/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    issues_path = output_dir / "relationship_validation_issues.json"
    with open(issues_path, 'w') as f:
        json.dump(all_issues, f, indent=2, default=str)
    print(f"\nIssues saved to: {issues_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return all_issues


if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues else 0)
