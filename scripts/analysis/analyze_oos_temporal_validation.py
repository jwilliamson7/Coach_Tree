#!/usr/bin/env python3
"""Out-of-sample temporal validation of gene -> WAR.

Every gene->WAR estimate so far is in-sample (the correlation is computed on the
same coach-years it is reported for). This asks a stricter question: does the
relationship learned on EARLY seasons predict WAR in LATER, unseen seasons? Fit
WAR ~ gene on a train window (default 2006-2018), apply the frozen train slope to
the held-out test window (2019-2024), and report:
  - test-period correlation of gene with actual WAR
  - predictive SKILL: RMSE of the train-fit model on test vs a naive baseline that
    predicts the train-mean WAR (skill = 1 - RMSE_model/RMSE_baseline; >0 means the
    early relationship carries real out-of-sample predictive content).
Runs for each composite gene available. Writes
outputs/analysis/oos_temporal_validation_results.json. ASCII only.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import coach_war_trajectories_path, merge_gene_war

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GAMES = 16
GENES = [
    ("Composite Aggression", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "composite_aggression"),
    ("Shotgun", "data/processed/coaching_genes/shotgun_gene.csv", "shotgun_gene_zscore"),
    ("Tempo", "data/processed/coaching_genes/tempo_gene.csv", "composite_tempo_zscore"),
    ("Defensive Scheme", "data/processed/coaching_genes/defensive_scheme_gene.csv",
     "composite_scheme_zscore"),
]


def oos(csv, gene_col, war, split_year):
    g = pd.read_csv(csv).rename(columns={"head_coach": "coach", "season": "year"})
    if gene_col not in g.columns:
        return None
    m = merge_gene_war(g, war, "coach", "coach", year_cols=("year", "year"),
                       how="inner", logger=logger)
    m = m[[gene_col, "annual_war", "year"]].dropna()
    m["war_games"] = m["annual_war"] * GAMES
    train = m[m["year"] <= split_year]
    test = m[m["year"] > split_year]
    if len(train) < 30 or len(test) < 30:
        return {"insufficient": True, "n_train": int(len(train)), "n_test": int(len(test))}

    # fit WAR ~ gene on train
    sl, ic, r_tr, p_tr, _ = stats.linregress(train[gene_col], train["war_games"])
    # apply frozen train model to test
    pred = ic + sl * test[gene_col].to_numpy()
    actual = test["war_games"].to_numpy()
    r_te, p_te = stats.pearsonr(test[gene_col], actual)
    rmse_model = float(np.sqrt(np.mean((actual - pred) ** 2)))
    rmse_base = float(np.sqrt(np.mean((actual - train["war_games"].mean()) ** 2)))
    skill = 1 - rmse_model / rmse_base if rmse_base > 0 else float("nan")
    return {
        "split_year": split_year,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "train_r": float(r_tr), "train_slope": float(sl),
        "test_r": float(r_te), "test_p": float(p_te),
        "rmse_model_on_test": rmse_model, "rmse_baseline_on_test": rmse_base,
        "predictive_skill": float(skill),
        "note": ("frozen train-window slope applied to unseen test window; "
                 "predictive_skill > 0 means the early gene->WAR relation generalizes"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_year", type=int, default=2018)
    args = ap.parse_args()

    war = pd.read_csv(coach_war_trajectories_path())
    war.columns = war.columns.str.lower()

    results = {}
    print("=" * 78)
    print(f"OUT-OF-SAMPLE TEMPORAL VALIDATION (train <= {args.split_year}, test > {args.split_year})")
    print("=" * 78)
    for label, csv, gene_col in GENES:
        if not Path(csv).exists():
            continue
        r = oos(csv, gene_col, war, args.split_year)
        if not r:
            continue
        results[label] = r
        if r.get("insufficient"):
            print(f"  {label:22s} insufficient (train={r['n_train']}, test={r['n_test']})")
        else:
            print(f"  {label:22s} train_r={r['train_r']:+.3f} -> test_r={r['test_r']:+.3f} "
                  f"(p={r['test_p']:.3f}); skill={r['predictive_skill']:+.3f} "
                  f"(n_test={r['n_test']})")

    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "oos_temporal_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'oos_temporal_validation_results.json'}")


if __name__ == "__main__":
    main()
