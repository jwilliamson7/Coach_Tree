#!/usr/bin/env python3
"""
Run All Post-Aggression Statistical Analyses and Visualizations

This script runs all statistical analyses on the aggression gene data
and all non-HTML visualization scripts, saving console output to log files in outputs/logs/.

Usage:
    python scripts/analysis/run_all_analyses.py
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

def run_script(script_path, log_name, description):
    """Run a single script and save output to log file"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"Log: {log_name}")
    print(f"{'='*80}\n")

    log_path = Path("outputs/logs") / log_name

    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Run the script and capture output
    try:
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True
            )

        if result.returncode == 0:
            print(f"[SUCCESS] Log saved to {log_path}\n")
            return True
        else:
            print(f"[FAILED] Exit code {result.returncode}")
            print(f"  Check log file for details: {log_path}\n")
            return False

    except Exception as e:
        print(f"[ERROR] {e}\n")
        return False


def main():
    """Run all analyses and visualizations in sequence"""
    start_time = datetime.now()

    print("\n" + "="*80)
    print("RUNNING ALL POST-AGGRESSION STATISTICAL ANALYSES AND VISUALIZATIONS")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Define all scripts to run
    scripts = [
        # Analysis scripts
        {
            'path': Path("scripts/analysis/analyze_gene_war_relationship.py"),
            'log': 'gene_war_relationship.log',
            'description': 'Gene-WAR Multiple Regression (coach-clustered SEs)'
        },
        {
            'path': Path("scripts/analysis/analyze_aggression_war_relationship.py"),
            'log': 'aggression_war_relationship.log',
            'description': 'Aggression-WAR Relationship Analysis'
        },
        {
            'path': Path("scripts/analysis/analyze_aggression_war_over_time.py"),
            'log': 'aggression_war_over_time.log',
            'description': 'Temporal Analysis: Aggression-WAR Over Time'
        },
        {
            'path': Path("scripts/analysis/analyze_aggression_persistence.py"),
            'log': 'aggression_persistence.log',
            'description': 'Aggression Persistence Analysis (Year-to-Year Stability)'
        },
        {
            'path': Path("scripts/analysis/analyze_aggression_by_coach_type.py"),
            'log': 'aggression_by_coach_type.log',
            'description': 'Aggression by Coach Type (Offensive vs Defensive)'
        },
        {
            'path': Path("scripts/analysis/analyze_persistence_by_coach_type.py"),
            'log': 'persistence_by_coach_type.log',
            'description': 'Persistence by Coach Type Analysis'
        },
        {
            'path': Path("scripts/analysis/analyze_inheritance_by_coach_type.py"),
            'log': 'inheritance_by_coach_type.log',
            'description': 'Inheritance by Coach Type (Mentor-Protégé Transmission)'
        },
        {
            'path': Path("scripts/analysis/check_aggression_variance.py"),
            'log': 'aggression_variance.log',
            'description': 'Aggression Variance Over Time Analysis'
        },
        {
            'path': Path("scripts/analysis/analyze_temporal_robustness.py"),
            'log': 'temporal_robustness.log',
            'description': 'Temporal Robustness Analysis'
        },
        {
            'path': Path("scripts/analysis/calculate_effect_sizes_and_power.py"),
            'log': 'effect_sizes_and_power.log',
            'description': 'Effect Sizes and Statistical Power Analysis'
        },
        {
            'path': Path("scripts/analysis/analyze_mentor_war_protege_war.py"),
            'log': 'mentor_war_protege_war.log',
            'description': 'Mentor-Protégé WAR Relationship Analysis'
        },
        # Multiple-comparison correction MUST run last: it reads every analysis
        # JSON above and applies a single global Benjamini-Hochberg FDR.
        {
            'path': Path("scripts/analysis/benjamini_hochberg_correction.py"),
            'log': 'benjamini_hochberg_correction.log',
            'description': 'Benjamini-Hochberg FDR Correction (global, clustered p where available)'
        },
        # Visualization scripts (non-HTML)
        {
            'path': Path("scripts/visualization/visualize_aggression_trends.py"),
            'log': 'viz_aggression_trends.log',
            'description': 'Aggression Trends Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_aggression_inheritance.py"),
            'log': 'viz_aggression_inheritance.log',
            'description': 'Aggression Inheritance Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_coordinator_to_hc_inheritance.py"),
            'log': 'viz_coordinator_to_hc_inheritance.log',
            'description': 'Coordinator to Head Coach Inheritance Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_coordinator_to_hc_components.py"),
            'log': 'viz_coordinator_to_hc_components.log',
            'description': 'Coordinator to Head Coach Components Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_shotgun_trends.py"),
            'log': 'viz_shotgun_trends.log',
            'description': 'Shotgun Formation Trends Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_quadratic_aggression.py"),
            'log': 'viz_quadratic_aggression.log',
            'description': 'Quadratic Aggression Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_quintile_comparison.py"),
            'log': 'viz_quintile_comparison.log',
            'description': 'Quintile Comparison Visualization'
        },
        {
            'path': Path("scripts/visualization/visualize_mentor_protege_war.py"),
            'log': 'viz_mentor_protege_war.log',
            'description': 'Mentor-Protégé WAR Visualization'
        }
    ]

    # Track results
    results = []

    # Run each script
    for script in scripts:
        success = run_script(
            script['path'],
            script['log'],
            script['description']
        )
        results.append({
            'name': script['description'],
            'success': success
        })

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("RUN SUMMARY")
    print("="*80)
    print(f"Start time:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:    {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:    {duration}")
    print()

    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful

    print(f"Results: {successful}/{len(results)} successful, {failed} failed")
    print()

    if successful > 0:
        print("Successful scripts:")
        for result in results:
            if result['success']:
                print(f"  [OK] {result['name']}")

    if failed > 0:
        print("\nFailed scripts:")
        for result in results:
            if not result['success']:
                print(f"  [FAIL] {result['name']}")

    print("\n" + "="*80)
    print(f"All log files saved to: outputs/logs/")
    print(f"Visualization outputs saved to: outputs/visualizations/")
    print("="*80 + "\n")

    # Exit with error code if any failed
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
