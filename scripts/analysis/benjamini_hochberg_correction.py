#!/usr/bin/env python3
"""
Apply a single global Benjamini-Hochberg FDR correction to every hypothesis
test reported in the Coach Aggression paper.

This script is DATA-DRIVEN: it reads p-values from the regenerated analysis
JSONs in outputs/analysis/ rather than hardcoding them, so it cannot drift from
the actual results. Wherever a coach- or mentor-clustered bootstrap p-value
exists (the repeated-measures correlations/regressions), it is preferred over
the naive p-value; tests with no cluster structure (linear trend, ANOVA, the
era t-tests/Mann-Whitney, the Chow test) fall back to their reported p-value.

The whole paper is treated as ONE FDR family (the most conservative, most
defensible choice against the garden-of-forking-paths). Run as part of
run_all_analyses.py so the correction stays in sync with the analyses.

Outputs:
  outputs/analysis/benjamini_hochberg_results.csv
  outputs/analysis/benjamini_hochberg_results.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_DIR = Path("outputs/analysis")

# Keys that hold a clustered (preferred) p-value, in priority order.
CLUSTERED_KEYS = ("p_bootstrap_coach_clustered", "p_bootstrap_mentor_clustered")
# Keys that hold a naive p-value, in priority order.
NAIVE_KEYS = ("p_value", "pearson_p", "aggression_p", "p")


def _load(name):
    path = ANALYSIS_DIR / name
    if not path.exists():
        logger.warning("missing source JSON: %s (its tests are skipped)", path)
        return None
    with open(path) as f:
        return json.load(f)


def _pick_p(node, prefer_clustered=True):
    """Return (p, source) from a result dict. Clustered p wins when present."""
    if not isinstance(node, dict):
        return None, None
    if prefer_clustered:
        for k in CLUSTERED_KEYS:
            if node.get(k) is not None:
                return float(node[k]), "clustered"
    for k in NAIVE_KEYS:
        if node.get(k) is not None:
            return float(node[k]), "naive"
    return None, None


def _add(tests, category, test, node, prefer_clustered=True):
    p, source = _pick_p(node, prefer_clustered)
    if p is None:
        logger.warning("no p-value for %s / %s", category, test)
        return
    tests.append({"category": category, "test": test, "p": p, "p_source": source})


def collect_tests():
    """Walk the regenerated JSONs and build the full list of paper tests."""
    tests = []

    # 1. Overall aggression -> WAR (clustered by coach)
    d = _load("aggression_war_regression_results.json")
    if d:
        for label in ["Composite Aggression", "4th Down Aggression",
                      "Pass-Heavy Aggression", "Deep Pass Aggression",
                      "2-Point Aggression"]:
            if label in d:
                _add(tests, "Overall WAR", label, d[label])

    # 2. Temporal trend (no cluster structure -> naive)
    d = _load("aggression_temporal_trend_results.json")
    if d:
        if "linear_trend" in d:
            _add(tests, "Temporal Trend", "Linear regression", d["linear_trend"], False)
        anova = d.get("era_comparison", {}).get("anova")
        if anova:
            _add(tests, "Temporal Trend", "ANOVA across eras", anova, False)
        for pair, node in d.get("pairwise_comparisons", {}).items():
            if "t_test" in node:
                _add(tests, "Temporal Trend", f"{pair} (t-test)", node["t_test"], False)
        # one Mann-Whitney (Early vs Late) to mirror the reported set
        for pair, node in d.get("pairwise_comparisons", {}).items():
            if "Late" in pair and "Early" in pair and "mann_whitney" in node:
                _add(tests, "Temporal Trend", f"{pair} (Mann-Whitney)", node["mann_whitney"], False)

    # 3. Era-specific aggression -> WAR (naive; era subset correlations)
    d = _load("aggression_war_temporal_analysis.json")
    if d:
        for era, measures in d.get("by_era", {}).items():
            for measure in ("composite", "pass_heavy"):
                if measure in measures:
                    _add(tests, "Era Analysis", f"{era}: {measure}", measures[measure], False)

    # 4. & 5. Coach type overall and by era (naive)
    d = _load("aggression_by_coach_type_results.json")
    if d:
        for ctype, measures in d.get("overall_by_type", {}).items():
            for measure in ("composite", "pass_heavy"):
                if measure in measures:
                    _add(tests, "Coach Type (Overall)", f"{ctype}: {measure}", measures[measure], False)
        for ctype, eras in d.get("by_type_and_era", {}).items():
            if ctype not in ("Offensive", "Defensive"):
                continue
            for era, measures in eras.items():
                if isinstance(measures, dict) and "composite" in measures:
                    _add(tests, "Coach Type by Era", f"{ctype} {era}", measures["composite"], False)

    # 6. Persistence overall (clustered by coach)
    d = _load("aggression_persistence_results.json")
    if d:
        for lag_key in ("lag_1", "lag_2", "lag_3"):
            for measure, node in d.get(lag_key, {}).items():
                _add(tests, f"Persistence {lag_key}", measure, node)

    # 7. Persistence by coach type (clustered by coach)
    d = _load("persistence_by_coach_type_results.json")
    if d:
        for ctype, measures in d.get("by_type", {}).items():
            for measure, lag_list in measures.items():
                if not isinstance(lag_list, list):
                    continue
                for node in lag_list:
                    lag = node.get("lag", "?")
                    _add(tests, f"Persistence {ctype}", f"{measure} Lag {lag}", node)

    # 8. & 9. Inheritance by mentor background / coordinator type (clustered by mentor)
    d = _load("inheritance_by_type_results.json")
    if d:
        for ctype, measures in d.get("by_mentor_background", {}).items():
            for measure, node in measures.items():
                _add(tests, f"Inheritance: {ctype} Mentors", measure, node)
        for ctype, measures in d.get("by_coordinator_type", {}).items():
            for measure, node in measures.items():
                _add(tests, f"Inheritance: {ctype}->HC", measure, node)

    # 10. Within-coach two-way fixed effects (cluster-robust aggression_p)
    d = _load("within_coach_fixed_effects_results.json")
    if d:
        tw = d.get("two_way", {})
        if "pooled" in tw:
            _add(tests, "Two-Way Fixed Effects", "Pooled (2006-2024)", tw["pooled"], False)
        for era, node in tw.get("stratified", {}).items():
            _add(tests, "Two-Way Fixed Effects", era, node, False)

    # 11. Temporal robustness: aggression x year interaction + Chow test (naive)
    d = _load("temporal_robustness_results.json")
    if d:
        inter = d.get("continuous_year", {}).get("coefficients", {}).get("aggression_x_year")
        if inter:
            _add(tests, "Temporal Robustness", "Aggression x Year interaction", inter, False)
        breaks = d.get("structural_breaks", {})
        # prefer a 2011 breakpoint to mirror the reported Chow test; else first
        chosen = breaks.get("2011") or (next(iter(breaks.values())) if breaks else None)
        if chosen:
            yr = chosen.get("breakpoint", "?")
            _add(tests, "Temporal Robustness", f"Chow test ({yr} breakpoint)", chosen, False)

    return tests


def benjamini_hochberg(df, alpha=0.05):
    """Single global BH-FDR. Adds rank, bh_threshold, sig_raw, sig_bh columns.

    A test is BH-significant if its p is <= the largest BH threshold among all
    tests with p below their own threshold (the standard step-up cutoff), not
    merely p <= its own threshold.
    """
    n = len(df)
    df = df.sort_values("p").reset_index(drop=True)
    df["rank"] = df.index + 1
    df["bh_threshold"] = df["rank"] / n * alpha
    df["sig_raw"] = df["p"] < 0.05
    below = df[df["p"] <= df["bh_threshold"]]
    cutoff_rank = int(below["rank"].max()) if len(below) else 0
    df["sig_bh"] = df["rank"] <= cutoff_rank
    return df, cutoff_rank


def main():
    tests = collect_tests()
    df = pd.DataFrame(tests)
    if df.empty:
        logger.error("no tests collected; are the analysis JSONs present?")
        raise SystemExit(1)

    n = len(df)
    n_clustered = int((df["p_source"] == "clustered").sum())
    df, cutoff_rank = benjamini_hochberg(df)
    cutoff_p = float(df.loc[df["rank"] == cutoff_rank, "p"].iloc[0]) if cutoff_rank else 0.0

    print("=" * 80)
    print("BENJAMINI-HOCHBERG FDR CORRECTION (single global family)")
    print("=" * 80)
    print(f"Total tests:                     {n}")
    print(f"  using clustered p-value:       {n_clustered}")
    print(f"  using naive p-value:           {n - n_clustered}")
    print(f"Significant at raw p < 0.05:     {int(df['sig_raw'].sum())}")
    print(f"Significant after BH (FDR 0.05): {int(df['sig_bh'].sum())}")
    print(f"BH cutoff: p <= {cutoff_p:.6f} (rank {cutoff_rank}/{n})")

    lost = df[(df["sig_raw"]) & (~df["sig_bh"])]
    print("\n" + "=" * 80)
    print(f"LOST SIGNIFICANCE AFTER CORRECTION ({len(lost)}):")
    print("=" * 80)
    if len(lost):
        for _, r in lost.iterrows():
            print(f"  {r['category']:28s} | {r['test']:28s} | p={r['p']:.4f} ({r['p_source']}) "
                  f"> thr={r['bh_threshold']:.5f}")
    else:
        print("  None.")

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = ANALYSIS_DIR / "benjamini_hochberg_results.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "method": "Benjamini-Hochberg FDR",
        "family": "single_global",
        "alpha": 0.05,
        "n_tests": n,
        "n_using_clustered_p": n_clustered,
        "n_using_naive_p": n - n_clustered,
        "n_sig_raw": int(df["sig_raw"].sum()),
        "n_sig_bh": int(df["sig_bh"].sum()),
        "bh_cutoff_p": cutoff_p,
        "bh_cutoff_rank": cutoff_rank,
        "lost_significance": lost[["category", "test", "p", "p_source", "bh_threshold"]].to_dict("records"),
        "tests": df[["category", "test", "p", "p_source", "rank", "bh_threshold",
                     "sig_raw", "sig_bh"]].to_dict("records"),
    }
    json_path = ANALYSIS_DIR / "benjamini_hochberg_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("SIGNIFICANCE BY CATEGORY (raw -> BH):")
    print("=" * 80)
    by_cat = df.groupby("category").agg(
        total=("p", "count"), sig_raw=("sig_raw", "sum"), sig_bh=("sig_bh", "sum")
    )
    print(by_cat.to_string())
    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
