#!/usr/bin/env python3
"""
Stability-selection producer for the play-level gene models (upstream parsimony).

For each model this:
  1. loads + filters that model's plays with the model's own DataProcessor,
  2. builds a group key (team-year) so subsampling is group-aware,
  3. encodes categoricals and prunes redundant/collinear features,
  4. runs Meinshausen-Buhlmann stability selection (xgb gain importance) over
     several K from one bootstrap pass,
  5. writes models/<name>/<stem>_selected_features.json.

The 10 model scripts and the 4 gene calculators consume that JSON via
utils.model_pipeline.apply_feature_selection / metadata, so the selected set is
used identically at train and gene time.

Group unit: team-year (posteam+season for offense, defteam+season for defense).
A play-level model has hundreds of team-years and huge n, so ROW-level
subsampling would barely perturb feature rankings (frequencies ~1.0, useless).
Dropping half the team-years per bootstrap injects the real heterogeneity the
selection frequency is meant to measure.

Usage:
  python scripts/models/run_stability_selection.py
  python scripts/models/run_stability_selection.py --models run_pass,shotgun --n_boot 50
  python scripts/models/run_stability_selection.py --sample_frac 0.1 --n_boot 5   # quick smoke
"""

import argparse
import gc
import importlib
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "models"))

from utils import model_pipeline as mp
from utils import parsimony
from utils import model_features as mf

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# model -> data-prep wiring (discovered from each model script)
MODELS = {
    "fourth_down": dict(module="fourth_down_decision_model", proc="FourthDownDataProcessor",
                        load="load_and_filter_fourth_downs", target="go_for_it",
                        side="offense", objective="binary:logistic",
                        stem="models/fourth_down/fourth_down_decision_model"),
    "run_pass": dict(module="run_pass_prediction_model", proc="RunPassDataProcessor",
                     load="load_and_filter_run_pass_plays", target="is_pass",
                     side="offense", objective="binary:logistic",
                     stem="models/run_pass/run_pass_prediction_model"),
    "pass_target": dict(module="pass_target_prediction_model", proc="PassTargetDataProcessor",
                        load="load_and_filter_pass_plays", target="targets_ahead",
                        side="offense", objective="binary:logistic",
                        stem="models/pass_target/pass_target_prediction_model"),
    "two_point": dict(module="two_point_conversion_model", proc="TwoPointDataProcessor",
                      load="load_and_filter_touchdown_plays", target="is_two_point",
                      side="offense", objective="binary:logistic",
                      stem="models/two_point/two_point_conversion_model"),
    "shotgun": dict(module="shotgun_prediction_model", proc="ShotgunDataProcessor",
                    load="load_and_filter_shotgun_plays", target="is_shotgun",
                    side="offense", objective="binary:logistic",
                    stem="models/shotgun/shotgun_prediction_model"),
    "no_huddle": dict(module="no_huddle_prediction_model", proc="NoHuddleDataProcessor",
                      load="load_and_filter_plays", target="is_no_huddle",
                      side="offense", objective="binary:logistic",
                      stem="models/no_huddle/no_huddle_prediction_model"),
    "pace": dict(module="pace_prediction_model", proc="PaceDataProcessor",
                 load="load_and_filter_plays", target="pace_target",
                 side="offense", objective="reg:squarederror",
                 stem="models/pace/pace_prediction_model"),
    "box_stacking": dict(module="box_stacking_prediction_model", proc="BoxStackingDataProcessor",
                         load="load_and_filter_plays", target="box_target",
                         side="defense", objective="reg:squarederror",
                         stem="models/box_stacking/box_stacking_prediction_model"),
    "pass_rush": dict(module="pass_rush_prediction_model", proc="PassRushDataProcessor",
                      load="load_and_filter_plays", target="rush_target",
                      side="defense", objective="reg:squarederror",
                      stem="models/pass_rush/pass_rush_prediction_model"),
    "man_coverage": dict(module="man_coverage_prediction_model", proc="ManCoverageDataProcessor",
                         load="load_and_filter_plays", target="is_man",
                         side="defense", objective="binary:logistic",
                         stem="models/man_coverage/man_coverage_prediction_model"),
}


def _group_key(raw: pd.DataFrame, side: str) -> pd.Series:
    """Team-year group key. Falls back to team-only if 'season' is absent."""
    team_col = "posteam" if side == "offense" else "defteam"
    if team_col not in raw.columns:
        raise KeyError(f"group column '{team_col}' not in loaded data; "
                       f"ensure the loader keeps posteam/defteam/season")
    team = raw[team_col].astype(str)
    if "season" in raw.columns:
        return team + "_" + raw["season"].astype(str)
    logger.warning("no 'season' column; grouping by %s only", team_col)
    return team


def select_for_model(name, cfg, args):
    logger.info("=" * 70)
    logger.info("STABILITY SELECTION: %s", name)
    logger.info("=" * 70)

    mod = importlib.import_module(cfg["module"])
    proc = getattr(mod, cfg["proc"])()

    raw = getattr(proc, cfg["load"])()
    raw = proc.create_target_variable(raw)

    group = _group_key(raw, cfg["side"])

    feature_data, feature_names = proc.prepare_features(raw)
    feature_data = feature_data.dropna(subset=[cfg["target"]])
    group = group.loc[feature_data.index]

    # optional row subsample for a quick smoke test
    if args.sample_frac and args.sample_frac < 1.0:
        feature_data = feature_data.sample(frac=args.sample_frac, random_state=args.seed)
        group = group.loc[feature_data.index]
        logger.info("sampled down to %d rows (sample_frac=%.3f)", len(feature_data), args.sample_frac)

    is_reg = cfg["objective"].startswith("reg:")
    y = feature_data[cfg["target"]]
    y = y if is_reg else y.astype(int)

    # encode categoricals to numeric (fit on the whole selection set is fine -- we
    # are selecting feature NAMES, not estimating a deployed model here)
    enc = {}
    X = mp.encode_categoricals(feature_data[feature_names].copy(), feature_names,
                               proc.categorical_features, enc, fit=True)

    # WS2: protected game-state core. Restrict the pre-specified core to columns
    # that are structurally valid for THIS model -- present and non-constant -- so
    # `down` (constant on 4th-down-only data) and two-point's undefined
    # down/distance/field block drop out automatically. The core is then shielded
    # from the redundancy prune and force-included in the final selected set
    # regardless of its measured selection frequency (frequency-blind by design).
    core = mf.get_protected_core_features(cfg["side"])
    core_present = [c for c in core if c in X.columns and X[c].nunique(dropna=False) > 1]
    core_excluded = [c for c in core if c not in core_present]
    logger.info("protected core (%d): %s", len(core_present), core_present)
    if core_excluded:
        logger.info("core excluded as structurally invalid (absent/constant): %s", core_excluded)

    X_pruned, dropped = parsimony.drop_redundant_features(
        X, threshold=args.redundancy_threshold, protect=core_present)
    logger.info("features: %d -> %d after redundancy prune (dropped %s)",
                X.shape[1], X_pruned.shape[1], dropped)

    n_groups = group.nunique()
    logger.info("%d rows, %d team-year groups, target=%s (%s)",
                len(feature_data), n_groups, cfg["target"],
                "regression" if is_reg else "classification")

    df_grp = pd.DataFrame({"grp": group.values})
    freq = parsimony.stability_selection_multi(
        df_grp, X_pruned.reset_index(drop=True),
        pd.Series(np.asarray(y)),
        group_col="grp", Ks=args.ks, n_boot=args.n_boot,
        subsample=args.subsample, seed=args.seed, objective=cfg["objective"],
    )
    selected = parsimony.select_stable_features(freq, pi=args.pi)

    # guard: never select zero features -- fall back to the top max(K) by frequency
    if not selected:
        topk = max(args.ks)
        selected = list(freq[topk].head(topk).index)
        logger.warning("no feature reached pi=%.2f; falling back to top-%d by frequency", args.pi, topk)

    # WS2: force the protected core in (regardless of selection frequency).
    forced_in = [c for c in core_present if c not in selected]
    selected = selected + forced_in
    if forced_in:
        logger.info("forced %d protected-core features not reaching pi: %s", len(forced_in), forced_in)
    else:
        logger.info("protected core already selected by stability selection; no features forced")

    payload = {
        "selected_features": selected,
        "method": "xgb_gain_stability",
        "pi_threshold": args.pi,
        "Ks": list(args.ks),
        "n_boot": args.n_boot,
        "subsample": args.subsample,
        "group_col": "team_year_" + cfg["side"],
        "n_groups": int(n_groups),
        "full_feature_count": int(X.shape[1]),
        "dropped_redundant": dropped,
        "protected_core": core_present,
        "protected_core_excluded": core_excluded,
        "forced_in": forced_in,
        "selection_frequency": {K: freq[K].round(3).to_dict() for K in args.ks},
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = parsimony.save_selected_features(cfg["stem"], payload)
    full_count = int(X.shape[1])
    logger.info("selected %d/%d features -> %s", len(selected), full_count, path)
    logger.info("selected: %s", selected)

    del raw, feature_data, X, X_pruned, group
    gc.collect()
    return name, len(selected), full_count


def main():
    ap = argparse.ArgumentParser(description="Stability selection for gene models")
    ap.add_argument("--models", default="all",
                    help="comma-separated subset (default all): " + ",".join(MODELS))
    ap.add_argument("--n_boot", type=int, default=50)
    ap.add_argument("--subsample", type=float, default=0.5)
    ap.add_argument("--pi", type=float, default=0.6)
    ap.add_argument("--ks", default="8,12,16", help="comma-separated K values")
    ap.add_argument("--redundancy_threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample_frac", type=float, default=None,
                    help="row subsample fraction for a quick smoke test (e.g. 0.1)")
    args = ap.parse_args()
    args.ks = tuple(int(k) for k in str(args.ks).split(","))

    which = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    unknown = [m for m in which if m not in MODELS]
    if unknown:
        ap.error(f"unknown models: {unknown}; choose from {list(MODELS)}")

    print("=" * 70)
    print("STABILITY SELECTION PRODUCER")
    print(f"models={which} n_boot={args.n_boot} subsample={args.subsample} "
          f"pi={args.pi} Ks={args.ks}")
    print("=" * 70)

    results = []
    for name in which:
        try:
            results.append(select_for_model(name, MODELS[name], args))
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)

    print("\n" + "=" * 70)
    print("SUMMARY")
    for name, n_sel, _ in results:
        print(f"  {name:14s}: {n_sel} features selected")
    print("=" * 70)


if __name__ == "__main__":
    main()
