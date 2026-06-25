#!/usr/bin/env python3
"""WS6: honest grouped-CV metrics for the 10 gene models.

The metadata `performance_metrics` come from a random play-level split, which is
optimistic: plays from the same team-year/coach leak across train and test. This
recomputes each model's metric under GroupKFold grouped by team-year
(posteam+season for offense, defteam+season for defense), reusing the model's
tuned hyperparameters and stability-selected features, with encoders + imputer +
SVD fit on the training folds only.

Writes per-model results into each model's *_metadata.json under
'grouped_cv_metrics' and a combined summary to
outputs/analysis/grouped_cv_metrics.json. ASCII only (Windows console).

Usage:
  python scripts/models/run_grouped_cv_metrics.py
  python scripts/models/run_grouped_cv_metrics.py --models run_pass,shotgun --n_splits 5
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))

from utils import model_pipeline as mp
from utils import parsimony
from run_stability_selection import MODELS, _group_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_DIR = REPO_ROOT / "outputs" / "analysis"


def _prepare_fn_factory(feature_names, categorical):
    """Return prepare_fn(Xtr_df, Xte_df) that fits encode+impute on TRAIN only."""
    def prep(Xtr_df, Xte_df):
        le = {}
        tr = mp.encode_categoricals(Xtr_df, feature_names, categorical, le, fit=True)
        te = mp.encode_categoricals(Xte_df, feature_names, categorical, le, fit=False)
        cols = list(tr.columns)
        te = te.reindex(columns=cols)
        tri, tei, _ = mp.fit_impute(tr, te)
        return tri, tei
    return prep


def evaluate_model(name, cfg, n_splits):
    logger.info("=" * 70)
    logger.info("GROUPED CV: %s", name)
    mod = importlib.import_module(cfg["module"])
    proc = getattr(mod, cfg["proc"])()
    raw = getattr(proc, cfg["load"])()
    raw = proc.create_target_variable(raw)
    group = _group_key(raw, cfg["side"])

    feature_data, feature_names = proc.prepare_features(raw)
    feature_data = feature_data.dropna(subset=[cfg["target"]])
    group = group.loc[feature_data.index]

    # stability-selected feature subset (same set train and gene time)
    selected = mp.apply_feature_selection(feature_names, cfg["stem"])

    is_reg = cfg["objective"].startswith("reg:")
    y = feature_data[cfg["target"]]
    y = y if is_reg else y.astype(int)

    # tuned hyperparameters from metadata (fallback: defaults)
    params = None
    meta_path = Path(f"{cfg['stem']}_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            params = json.load(f).get("best_params")

    X_df = feature_data[selected].reset_index(drop=True)
    y = pd.Series(np.asarray(y)).reset_index(drop=True)
    groups = pd.Series(np.asarray(group)).reset_index(drop=True)

    prep = _prepare_fn_factory(selected, proc.categorical_features)
    make_est = lambda: mp.build_xgb_estimator(params, cfg["objective"], not is_reg)

    out = {"n": int(len(X_df)), "n_groups": int(groups.nunique()),
           "n_splits": n_splits, "group_col": "team_year_" + cfg["side"],
           "n_features": len(selected)}
    if is_reg:
        for metric in ("r2", "rmse"):
            res = parsimony.grouped_cv_score(make_est, prep, X_df, y, groups,
                                             n_splits=n_splits, scoring=metric)
            out[f"grouped_{metric}"] = res["mean"]
            out[f"grouped_{metric}_std"] = res["std"]
        logger.info("%s grouped R2=%.4f RMSE=%.4f (random-split was the metadata value)",
                    name, out["grouped_r2"], out["grouped_rmse"])
    else:
        res = parsimony.grouped_cv_score(make_est, prep, X_df, y, groups,
                                         n_splits=n_splits, scoring="auc")
        out["grouped_auc"] = res["mean"]
        out["grouped_auc_std"] = res["std"]
        logger.info("%s grouped AUC=%.4f", name, out["grouped_auc"])

    # persist into the model metadata
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["grouped_cv_metrics"] = out
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

    del raw, feature_data, X_df, group, groups
    gc.collect()
    return name, out


def main():
    ap = argparse.ArgumentParser(description="WS6 grouped-CV metrics for gene models")
    ap.add_argument("--models", default="all")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    which = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    summary = {}
    for name in which:
        try:
            _, out = evaluate_model(name, MODELS[name], args.n_splits)
            summary[name] = out
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)
            summary[name] = {"error": str(e)}

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANALYSIS_DIR / "grouped_cv_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("GROUPED-CV METRICS (team-year grouped; vs optimistic random split)")
    print("=" * 70)
    for name, out in summary.items():
        if "error" in out:
            print(f"  {name:14s}: ERROR {out['error']}")
        elif "grouped_auc" in out:
            print(f"  {name:14s}: AUC={out['grouped_auc']:.4f} (n={out['n']}, {out['n_groups']} team-years)")
        else:
            print(f"  {name:14s}: R2={out.get('grouped_r2',float('nan')):.4f} "
                  f"RMSE={out.get('grouped_rmse',float('nan')):.4f} (n={out['n']})")
    print(f"\nSaved: {ANALYSIS_DIR / 'grouped_cv_metrics.json'}")


if __name__ == "__main__":
    main()
