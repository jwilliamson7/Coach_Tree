#!/usr/bin/env python3
"""WS3: calibration check for the classifier gene models.

Gene = actual - predicted_prob, so a miscalibrated classifier would bias the
residual. AUC measures discrimination, not calibration; this measures calibration
directly via the Expected Calibration Error (ECE) and Brier score, computed on
LEAKAGE-FREE cross-fit out-of-fold predictions (GroupKFold by team-year, the same
machinery the genes use). In-sample calibration would be meaningless.

A constant calibration bias cancels in the gene (it is differenced then z-scored
across coaches), so the risk is NON-uniform error concentrated in sparse buckets
(4th-and-long, two-point). We report ECE per model and the largest per-bin gap;
only a model with high ECE (> ~0.03-0.05) AND a visibly non-uniform reliability
curve would warrant isotonic recalibration. Default expectation: XGBoost on
log-loss is well-calibrated out of the box.

(Supersedes the earlier coach-year-aggregated calibration check, which could not
see per-play sparse-bucket miscalibration.)

Writes outputs/analysis/calibration_metrics.json. ASCII only (Windows console).

Usage:
  python scripts/validation/validate_calibration.py
  python scripts/validation/validate_calibration.py --bins 15 --n_splits 5
"""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))

from utils import model_pipeline as mp
from run_stability_selection import MODELS, _group_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"

CLASSIFIERS = ["fourth_down", "run_pass", "pass_target", "two_point",
               "shotgun", "no_huddle", "man_coverage"]


def expected_calibration_error(y_true, y_prob, n_bins=15):
    """ECE with equal-width bins, plus the per-bin reliability curve."""
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    curve = []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        conf = float(y_prob[m].mean())
        acc = float(y_true[m].mean())
        ece += abs(acc - conf) * nb / n
        curve.append({"bin": b, "n": nb, "conf": conf, "acc": acc})
    return float(ece), curve


def evaluate(name, cfg, n_splits, n_bins):
    logger.info("=" * 60)
    logger.info("CALIBRATION: %s", name)
    mod = importlib.import_module(cfg["module"])
    proc = getattr(mod, cfg["proc"])()
    raw = getattr(proc, cfg["load"])()
    raw = proc.create_target_variable(raw)
    group = _group_key(raw, cfg["side"])
    feature_data, feature_names = proc.prepare_features(raw)
    feature_data = feature_data.dropna(subset=[cfg["target"]])
    group = group.loc[feature_data.index]

    selected = mp.apply_feature_selection(feature_names, cfg["stem"])

    params = None
    meta_path = Path(f"{cfg['stem']}_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            params = json.load(f).get("best_params")

    oof = mp.crossfit_predict(
        feature_data, selected, cfg["target"], proc.categorical_features,
        group, params, cfg["objective"], is_classifier=True, n_splits=n_splits)

    yv = feature_data[cfg["target"]].astype(int).to_numpy()
    ece, curve = expected_calibration_error(yv, oof, n_bins=n_bins)
    brier = float(np.mean((oof - yv) ** 2))
    base = float(yv.mean())
    out = {"n": int(len(yv)), "base_rate": base, "ece": ece, "brier": brier,
           "n_bins": n_bins, "n_splits": n_splits,
           "max_bin_gap": max((abs(c["acc"] - c["conf"]) for c in curve), default=0.0),
           "reliability_curve": curve}
    flag = "OK" if ece <= 0.03 else ("REVIEW" if ece <= 0.05 else "RECALIBRATE?")
    out["flag"] = flag
    logger.info("%s: ECE=%.4f Brier=%.4f base=%.3f max_bin_gap=%.3f -> %s",
                name, ece, brier, base, out["max_bin_gap"], flag)
    return out


def main():
    ap = argparse.ArgumentParser(description="WS3 calibration (ECE) for classifier gene models")
    ap.add_argument("--models", default="all")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    which = CLASSIFIERS if args.models == "all" else [m.strip() for m in args.models.split(",")]
    summary = {}
    for name in which:
        try:
            summary[name] = evaluate(name, MODELS[name], args.n_splits, args.bins)
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)
            summary[name] = {"error": str(e)}

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "calibration_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("CALIBRATION (ECE on cross-fit OOF; lower is better)")
    print("=" * 60)
    for name, out in summary.items():
        if "error" in out:
            print(f"  {name:14s}: ERROR {out['error']}")
        else:
            print(f"  {name:14s}: ECE={out['ece']:.4f}  Brier={out['brier']:.4f}  "
                  f"max_gap={out['max_bin_gap']:.3f}  [{out['flag']}]")
    print(f"\nSaved: {ANALYSIS_DIR / 'calibration_metrics.json'}")


if __name__ == "__main__":
    main()
