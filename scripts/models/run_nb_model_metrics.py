#!/usr/bin/env python3
"""Nadeau-Bengio corrected resampling CIs for the 10 gene models' performance.

The grouped-CV point estimates (run_grouped_cv_metrics.py) report a mean and a
naive across-fold std. For JQAS we want a proper 95% confidence interval on each
model metric. The Nadeau-Bengio (2003) corrected resampled-t accounts for the
dependence between overlapping training sets across resamples, which a naive
resample std ignores (and badly understates).

Procedure, per model:
  - Draw J random train/test resamples with GroupShuffleSplit grouped by team-year
    (posteam+season offense, defteam+season defense) so no team-year leaks across
    train and test -- the same leakage guard as the grouped-CV metrics.
  - test fraction f = rho / (1 + rho) so that n_test / n_train = rho (= 0.2),
    matching the CoachingProject NB convention (J=50, rho=0.2).
  - Refit the model (tuned hyperparameters, stability-selected features, encoders +
    imputer fit on the train fold only) on each resample and score the held-out test.
  - Corrected variance of the mean score: (1/J + rho) * S^2, where S^2 is the plain
    sample variance of the J scores. 95% CI = mean +/- t_{J-1, .975} * sqrt of that.

Writes per-model results into each model's *_metadata.json under 'nb_metrics' and a
combined summary to outputs/analysis/nb_model_metrics.json. ASCII only.

Usage:
  python scripts/models/run_nb_model_metrics.py
  python scripts/models/run_nb_model_metrics.py --models run_pass,shotgun --J 50 --rho 0.2
"""

import argparse
import gc
import importlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))

from utils import model_pipeline as mp
from run_stability_selection import MODELS, _group_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def nb_corrected_ci(scores, rho=0.2, alpha=0.05):
    """Nadeau-Bengio corrected 95% CI from J resample scores.

    se_corrected = sqrt((1/J + rho) * sample_variance); CI uses t with J-1 dof.
    """
    s = np.asarray([x for x in scores if np.isfinite(x)], float)
    J = int(len(s))
    if J == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "J": 0, "rho": rho}
    mean = float(s.mean())
    if J < 2:
        return {"mean": mean, "ci_low": float("nan"), "ci_high": float("nan"),
                "std": 0.0, "se_corrected": float("nan"), "J": J, "rho": rho}
    var = float(s.var(ddof=1))
    se = float(np.sqrt((1.0 / J + rho) * var))
    tcrit = float(stats.t.ppf(1 - alpha / 2, J - 1))
    return {"mean": mean, "ci_low": mean - tcrit * se, "ci_high": mean + tcrit * se,
            "std": float(np.sqrt(var)), "se_corrected": se, "J": J, "rho": rho}


def _prep(Xtr_df, Xte_df, feature_names, categorical):
    """Fit encoders+imputer on TRAIN only, transform both. Mirrors grouped CV."""
    le = {}
    tr = mp.encode_categoricals(Xtr_df, feature_names, categorical, le, fit=True)
    te = mp.encode_categoricals(Xte_df, feature_names, categorical, le, fit=False)
    te = te.reindex(columns=list(tr.columns))
    tri, tei, _ = mp.fit_impute(tr, te)
    return tri, tei


def evaluate_model(name, cfg, J, rho):
    logger.info("=" * 70)
    logger.info("NADEAU-BENGIO RESAMPLING: %s (J=%d, rho=%.2f)", name, J, rho)
    mod = importlib.import_module(cfg["module"])
    proc = getattr(mod, cfg["proc"])()
    raw = getattr(proc, cfg["load"])()
    raw = proc.create_target_variable(raw)
    group = _group_key(raw, cfg["side"])

    feature_data, feature_names = proc.prepare_features(raw)
    feature_data = feature_data.dropna(subset=[cfg["target"]])
    group = group.loc[feature_data.index]

    selected = mp.apply_feature_selection(feature_names, cfg["stem"])

    is_reg = cfg["objective"].startswith("reg:")
    y = feature_data[cfg["target"]]
    y = y if is_reg else y.astype(int)

    params = None
    meta_path = Path(f"{cfg['stem']}_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            params = json.load(f).get("best_params")

    X_df = feature_data[selected].reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    groups = pd.Series(np.asarray(group)).reset_index(drop=True)

    test_frac = rho / (1.0 + rho)
    gss = GroupShuffleSplit(n_splits=J, test_size=test_frac, random_state=0)

    auc_scores, r2_scores, rmse_scores = [], [], []
    for i, (tr, te) in enumerate(gss.split(X_df, y, groups)):
        Xtr, Xte = _prep(X_df.iloc[tr], X_df.iloc[te], selected, proc.categorical_features)
        est = mp.build_xgb_estimator(params, cfg["objective"], not is_reg)
        est.fit(Xtr, y.iloc[tr])
        if is_reg:
            pred = est.predict(Xte)
            r2_scores.append(float(r2_score(y.iloc[te], pred)))
            rmse_scores.append(float(np.sqrt(mean_squared_error(y.iloc[te], pred))))
        else:
            proba = est.predict_proba(Xte)[:, 1]
            auc_scores.append(float(roc_auc_score(y.iloc[te], proba)))
        if (i + 1) % 10 == 0:
            logger.info("  %s: %d/%d resamples done", name, i + 1, J)

    out = {"n": int(len(X_df)), "n_groups": int(groups.nunique()),
           "J": J, "rho": rho, "test_frac": test_frac,
           "group_col": "team_year_" + cfg["side"], "n_features": len(selected)}
    if is_reg:
        out["nb_r2"] = nb_corrected_ci(r2_scores, rho=rho)
        out["nb_rmse"] = nb_corrected_ci(rmse_scores, rho=rho)
        logger.info("%s NB R2=%.4f [%.4f, %.4f]  RMSE=%.4f [%.4f, %.4f]", name,
                    out["nb_r2"]["mean"], out["nb_r2"]["ci_low"], out["nb_r2"]["ci_high"],
                    out["nb_rmse"]["mean"], out["nb_rmse"]["ci_low"], out["nb_rmse"]["ci_high"])
    else:
        out["nb_auc"] = nb_corrected_ci(auc_scores, rho=rho)
        logger.info("%s NB AUC=%.4f [%.4f, %.4f]", name,
                    out["nb_auc"]["mean"], out["nb_auc"]["ci_low"], out["nb_auc"]["ci_high"])

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["nb_metrics"] = out
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    del raw, feature_data, X_df, group, groups
    gc.collect()
    return name, out


def main():
    ap = argparse.ArgumentParser(description="Nadeau-Bengio corrected resampling CIs for gene models")
    ap.add_argument("--models", default="all")
    ap.add_argument("--J", type=int, default=50)
    ap.add_argument("--rho", type=float, default=0.2)
    args = ap.parse_args()

    which = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    summary = {}
    for name in which:
        try:
            _, out = evaluate_model(name, MODELS[name], args.J, args.rho)
            summary[name] = out
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)
            summary[name] = {"error": str(e)}

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "nb_model_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("NADEAU-BENGIO CORRECTED RESAMPLING CIs (J=%d, rho=%.2f)" % (args.J, args.rho))
    print("=" * 70)
    for name, out in summary.items():
        if "error" in out:
            print(f"  {name:14s}: ERROR {out['error']}")
        elif "nb_auc" in out:
            a = out["nb_auc"]
            print(f"  {name:14s}: AUC={a['mean']:.4f} [{a['ci_low']:.4f}, {a['ci_high']:.4f}]")
        else:
            r, rm = out["nb_r2"], out["nb_rmse"]
            print(f"  {name:14s}: R2={r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]  "
                  f"RMSE={rm['mean']:.4f} [{rm['ci_low']:.4f}, {rm['ci_high']:.4f}]")
    print(f"\nSaved: {ANALYSIS_DIR / 'nb_model_metrics.json'}")


if __name__ == "__main__":
    main()
