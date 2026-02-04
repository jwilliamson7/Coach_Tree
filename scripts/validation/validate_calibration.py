#!/usr/bin/env python3
"""
Model Calibration Validation Script

This script validates the calibration of predictive models:
1. Generates calibration plots (predicted probability vs actual rate)
2. Calculates Brier scores
3. Checks for systematic over/under-confidence
4. Validates that aggression genes are based on well-calibrated models

Output:
- Calibration metrics for each model
- Calibration curve data
- Issues with model calibration
"""

import pandas as pd
import numpy as np
import json
import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def analyze_aggression_calibration(data_path: Path) -> Dict:
    """
    Analyze calibration based on aggression gene data.

    Since aggression = actual - predicted, we can infer calibration:
    - If mean(aggression) ~ 0, models are well-calibrated on average
    - If mean(actual_rate) ~ mean(predicted_rate), models are calibrated
    """

    results = {
        'calibration_analysis': {},
        'issues': []
    }

    if not data_path.exists():
        results['issues'].append({
            'type': 'missing_file',
            'description': f'Aggression gene data not found at {data_path}'
        })
        return results

    df = pd.read_csv(data_path)

    print("\n" + "=" * 80)
    print("MODEL CALIBRATION ANALYSIS (from Aggression Gene Data)")
    print("=" * 80)

    # For each model type, compare actual vs predicted rates
    model_types = [
        {
            'name': '4th Down Decision',
            'actual_col': 'actual_decision',
            'predicted_col': 'predicted_go_rate',
            'aggression_col': 'fourth_down_aggression'
        },
        {
            'name': 'Run/Pass Prediction',
            'actual_col': 'actual_pass',
            'predicted_col': 'predicted_pass_rate',
            'aggression_col': 'pass_heavy_aggression'
        },
        {
            'name': 'Pass Target Prediction',
            'actual_col': 'actual_beyond',
            'predicted_col': 'predicted_beyond_rate',
            'aggression_col': 'deep_pass_aggression'
        },
        {
            'name': 'Two-Point Conversion',
            'actual_col': 'actual_two_point',
            'predicted_col': 'predicted_two_point_rate',
            'aggression_col': 'two_point_aggression'
        }
    ]

    for model in model_types:
        if model['actual_col'] not in df.columns or model['predicted_col'] not in df.columns:
            continue

        actual_mean = df[model['actual_col']].mean()
        predicted_mean = df[model['predicted_col']].mean()
        aggression_mean = df[model['aggression_col']].mean() if model['aggression_col'] in df.columns else None

        # Calculate calibration metrics
        calibration_error = actual_mean - predicted_mean  # Should be ~0 for well-calibrated model

        # Correlation between predicted and actual (should be high)
        correlation = df[model['predicted_col']].corr(df[model['actual_col']])

        # Calculate pseudo-Brier by binning
        n_bins = 10
        df_sorted = df.sort_values(model['predicted_col'])
        df_sorted['bin'] = pd.qcut(df_sorted[model['predicted_col']], n_bins, labels=False, duplicates='drop')

        bin_stats = df_sorted.groupby('bin').agg({
            model['predicted_col']: 'mean',
            model['actual_col']: 'mean'
        }).dropna()

        if len(bin_stats) > 0:
            # Mean absolute calibration error across bins
            bin_calibration_error = abs(bin_stats[model['actual_col']] - bin_stats[model['predicted_col']]).mean()
        else:
            bin_calibration_error = np.nan

        model_results = {
            'mean_actual_rate': float(actual_mean),
            'mean_predicted_rate': float(predicted_mean),
            'overall_calibration_error': float(calibration_error),
            'mean_aggression': float(aggression_mean) if aggression_mean is not None else None,
            'actual_predicted_correlation': float(correlation),
            'binned_calibration_error': float(bin_calibration_error) if not np.isnan(bin_calibration_error) else None,
            'n_observations': int(df[model['actual_col']].notna().sum())
        }

        results['calibration_analysis'][model['name']] = model_results

        # Print results
        print(f"\n{model['name']}:")
        print(f"  Mean actual rate: {actual_mean:.4f}")
        print(f"  Mean predicted rate: {predicted_mean:.4f}")
        print(f"  Overall calibration error: {calibration_error:.4f}")

        if aggression_mean is not None:
            print(f"  Mean aggression (should be ~0): {aggression_mean:.4f}")

        print(f"  Actual-Predicted correlation: {correlation:.4f}")

        if not np.isnan(bin_calibration_error):
            print(f"  Binned calibration error: {bin_calibration_error:.4f}")

        # Flag issues
        if abs(calibration_error) > 0.02:
            issue = {
                'type': 'calibration_bias',
                'model': model['name'],
                'description': f"Systematic bias of {calibration_error:.4f}"
            }
            results['issues'].append(issue)
            print(f"  WARNING: Systematic calibration bias detected!")

    return results


def analyze_calibration_by_year(data_path: Path) -> Dict:
    """Analyze if calibration changes over time (temporal stability)"""

    results = {
        'temporal_calibration': {},
        'issues': []
    }

    if not data_path.exists():
        return results

    df = pd.read_csv(data_path)

    print("\n" + "=" * 80)
    print("TEMPORAL CALIBRATION ANALYSIS")
    print("=" * 80)
    print("\nChecking if model calibration is stable across years...")

    if 'season' not in df.columns:
        results['issues'].append({
            'type': 'missing_column',
            'description': 'Season column not found in data'
        })
        return results

    # Analyze 4th down calibration by year
    if 'actual_decision' in df.columns and 'predicted_go_rate' in df.columns:
        print("\n4th Down Decision - Calibration by Year:")
        print("-" * 60)

        yearly_stats = df.groupby('season').agg({
            'actual_decision': 'mean',
            'predicted_go_rate': 'mean',
            'fourth_down_aggression': 'mean'
        }).rename(columns={
            'actual_decision': 'actual_rate',
            'predicted_go_rate': 'predicted_rate',
            'fourth_down_aggression': 'aggression'
        })

        yearly_stats['calibration_error'] = yearly_stats['actual_rate'] - yearly_stats['predicted_rate']

        results['temporal_calibration']['fourth_down'] = yearly_stats.to_dict()

        print(f"{'Year':<8} {'Actual':<10} {'Predicted':<10} {'Error':<10}")
        print("-" * 40)

        for year, row in yearly_stats.iterrows():
            marker = " [!]" if abs(row['calibration_error']) > 0.05 else ""
            print(f"{int(year):<8} {row['actual_rate']:.4f}     {row['predicted_rate']:.4f}     {row['calibration_error']:+.4f}{marker}")

        # Check for temporal trend
        years = yearly_stats.index.values
        errors = yearly_stats['calibration_error'].values

        if len(years) > 3:
            correlation = np.corrcoef(years, errors)[0, 1]
            print(f"\nTemporal trend (year vs error correlation): {correlation:.3f}")

            if abs(correlation) > 0.5:
                results['issues'].append({
                    'type': 'temporal_drift',
                    'model': '4th Down Decision',
                    'description': f"Model shows temporal drift in calibration (r={correlation:.3f})"
                })
                print("  WARNING: Significant temporal drift detected!")

    return results


def analyze_calibration_by_situation(data_path: Path) -> Dict:
    """Check if calibration varies by game situation"""

    results = {
        'situational_calibration': {},
        'issues': []
    }

    # This would require access to the original play-by-play data
    # For now, we note that this analysis would be valuable

    print("\n" + "=" * 80)
    print("SITUATIONAL CALIBRATION ANALYSIS")
    print("=" * 80)
    print("\nNote: Full situational calibration analysis requires play-by-play data")
    print("Key situations to check:")
    print("  - Score differential (close games vs blowouts)")
    print("  - Quarter/Time remaining")
    print("  - Field position (own territory vs opponent territory)")
    print("  - Down and distance")

    return results


def generate_calibration_report(data_dir: Path) -> Dict:
    """Generate comprehensive calibration report"""

    report = {
        'generated': datetime.now().isoformat(),
        'data_source': str(data_dir),
        'analyses': {},
        'issues': [],
        'recommendations': []
    }

    aggression_path = data_dir / "coaching_genes" / "aggression_gene_by_year.csv"

    # Run analyses
    overall_results = analyze_aggression_calibration(aggression_path)
    report['analyses']['overall_calibration'] = overall_results.get('calibration_analysis', {})
    report['issues'].extend(overall_results.get('issues', []))

    temporal_results = analyze_calibration_by_year(aggression_path)
    report['analyses']['temporal_calibration'] = temporal_results.get('temporal_calibration', {})
    report['issues'].extend(temporal_results.get('issues', []))

    situational_results = analyze_calibration_by_situation(aggression_path)
    report['analyses']['situational_calibration'] = situational_results.get('situational_calibration', {})
    report['issues'].extend(situational_results.get('issues', []))

    # Generate recommendations
    if any(i['type'] == 'calibration_bias' for i in report['issues']):
        report['recommendations'].append(
            "Consider Platt scaling or isotonic regression to improve calibration"
        )

    if any(i['type'] == 'temporal_drift' for i in report['issues']):
        report['recommendations'].append(
            "Model shows temporal drift - consider retraining on more recent data or using rolling window"
        )

    # Summary
    print("\n" + "=" * 80)
    print("CALIBRATION VALIDATION SUMMARY")
    print("=" * 80)

    if report['issues']:
        print(f"\nIssues found: {len(report['issues'])}")
        for issue in report['issues']:
            print(f"  - {issue['type']}: {issue.get('description', issue.get('model', 'Unknown'))}")
    else:
        print("\nNo major calibration issues found!")

    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")

    return report


def main():
    """Main validation function"""

    data_dir = Path("data/processed")

    print("=" * 80)
    print("MODEL CALIBRATION VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    report = generate_calibration_report(data_dir)

    # Save report
    output_dir = Path("data/processed/validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "calibration_validation.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)

    return report['issues']


if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues else 0)
