#!/usr/bin/env python3
"""
Full regeneration cascade for the leakage-free / parsimony refactor.

Runs, in order:
  1. stability selection for all 10 gene models  -> *_selected_features.json
  2. retrain all 10 models                        -> model + encoders + imputer
  3. recompute all 4 genes                        -> data/processed/coaching_genes/*
  4. run_all_analyses                             -> outputs/analysis/*
  5. verify_paper_statistics                      -> consistency check

It STOPS before any paper/report edits by design (the report is regenerated and
discussed separately). Each step logs to outputs/logs/regen_*.log; by default a
failing step aborts the cascade (use --keep_going to continue).

Usage:
  python scripts/run_full_regeneration.py                       # full, n_iter=100
  python scripts/run_full_regeneration.py --n_iter 30           # faster retrain
  python scripts/run_full_regeneration.py --skip_selection      # reuse existing JSONs
  python scripts/run_full_regeneration.py --start_step genes    # resume partway
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = REPO_ROOT / "outputs" / "logs"

MODELS = [
    "fourth_down_decision_model", "run_pass_prediction_model",
    "pass_target_prediction_model", "two_point_conversion_model",
    "shotgun_prediction_model", "no_huddle_prediction_model",
    "pace_prediction_model", "box_stacking_prediction_model",
    "pass_rush_prediction_model", "man_coverage_prediction_model",
]
GENES = [
    "calculate_aggression_gene", "calculate_tempo_gene",
    "calculate_shotgun_gene", "calculate_defensive_scheme_gene",
]
STEPS = ["selection", "training", "genes", "analyses", "verify"]


def run(cmd, log_name, keep_going):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    print(f"  -> {' '.join(cmd)}")
    print(f"     logging to {log_path}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f,
                              stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else f"FAILED (exit {proc.returncode})"
    print(f"     {status} in {dt/60:.1f} min")
    if not ok and not keep_going:
        print(f"\nABORTING: {log_name} failed. See {log_path}. Use --keep_going to continue past failures.")
        sys.exit(1)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Full regeneration cascade")
    ap.add_argument("--n_iter", type=int, default=100, help="RandomizedSearchCV iterations per model")
    ap.add_argument("--cv_folds", type=int, default=3)
    ap.add_argument("--stability_n_boot", type=int, default=50)
    ap.add_argument("--skip_selection", action="store_true")
    ap.add_argument("--start_step", choices=STEPS, default="selection")
    ap.add_argument("--keep_going", action="store_true", help="continue past a failing step")
    args = ap.parse_args()

    py = sys.executable
    start_idx = STEPS.index(args.start_step)
    results = []

    print("=" * 72)
    print(f"FULL REGENERATION CASCADE  ({datetime.now():%Y-%m-%d %H:%M:%S})")
    print(f"n_iter={args.n_iter} cv_folds={args.cv_folds} "
          f"stability_n_boot={args.stability_n_boot} start_step={args.start_step}")
    print("STOPS before paper/report edits by design.")
    print("=" * 72)

    # 1. stability selection
    if start_idx <= STEPS.index("selection") and not args.skip_selection:
        print("\n[1/5] STABILITY SELECTION (all models)")
        ok = run([py, "scripts/models/run_stability_selection.py",
                  "--n_boot", str(args.stability_n_boot)],
                 "regen_1_stability_selection.log", args.keep_going)
        results.append(("stability_selection", ok))

    # 2. retrain models
    if start_idx <= STEPS.index("training"):
        print("\n[2/5] RETRAIN MODELS")
        for mod in MODELS:
            ok = run([py, f"scripts/models/{mod}.py",
                      "--n_iter", str(args.n_iter), "--cv_folds", str(args.cv_folds)],
                     f"regen_2_train_{mod}.log", args.keep_going)
            results.append((f"train_{mod}", ok))

    # 3. recompute genes
    if start_idx <= STEPS.index("genes"):
        print("\n[3/5] RECOMPUTE GENES")
        for gene in GENES:
            ok = run([py, f"scripts/analysis/{gene}.py"],
                     f"regen_3_{gene}.log", args.keep_going)
            results.append((gene, ok))

    # 4. run all analyses
    if start_idx <= STEPS.index("analyses"):
        print("\n[4/5] RUN ALL ANALYSES")
        ok = run([py, "scripts/analysis/run_all_analyses.py"],
                 "regen_4_run_all_analyses.log", args.keep_going)
        results.append(("run_all_analyses", ok))

    # 5. verify paper statistics
    if start_idx <= STEPS.index("verify"):
        print("\n[5/5] VERIFY PAPER STATISTICS")
        ok = run([py, "scripts/analysis/verify_paper_statistics.py"],
                 "regen_5_verify_paper_statistics.log", args.keep_going)
        results.append(("verify_paper_statistics", ok))

    print("\n" + "=" * 72)
    print("CASCADE SUMMARY")
    for name, ok in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name}")
    print("=" * 72)
    print("\nDONE. Paper/report NOT modified -- review the before/after deltas, then")
    print("discuss paper direction before editing outputs/reports/.")


if __name__ == "__main__":
    main()
