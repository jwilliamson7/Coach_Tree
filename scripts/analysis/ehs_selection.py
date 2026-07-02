#!/usr/bin/env python3
"""EHS confirmatory selection S (Section 3, mechanical extension pre-specified).

Selection S is the era-adjusted, inverse-variance-weighted correlation of a trait
with Coach WAR: within-season demean the standardized gene and WAR (the single
contemporary-group control), weight each coach-season by its games-based WAR
precision, and take the coach-clustered weighted Pearson correlation. This is the
identical estimator already used for the composites and defensive components
(utils.war_noise.war_noise_robustness -> r_ivw_eradj); here it is applied to each
of the ten frozen sub-traits and the four gene-level composites.

Coach WAR is pinned at Coach_WAR commit dffe2f1 (Section 5). Writes
outputs/analysis/ehs_selection_results.json and ehs_selection_draws.npz (coach-
block bootstrap draw vectors, consumed by the H1 family-contrast step). ASCII only.
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import coach_war_trajectories_path, merge_gene_war
from utils.war_noise import war_noise_robustness, _ensure_precision, GAMES_PER_SEASON
from utils.parsimony import within_group_demean, weighted_pearson

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GENES = Path("data/processed/coaching_genes")
OUT_DIR = Path("outputs/analysis")
N_BOOT = 4000

# trait -> (csv, committed z-score column)
SUBTRAIT_Z = {
    "fourth_down": ("aggression_gene_by_year.csv", "fourth_down_aggression_zscore"),
    "pass_heavy": ("aggression_gene_by_year.csv", "pass_heavy_aggression_zscore"),
    "deep_pass": ("aggression_gene_by_year.csv", "deep_pass_aggression_zscore"),
    "two_point": ("aggression_gene_by_year.csv", "two_point_aggression_zscore"),
    "no_huddle": ("tempo_gene.csv", "no_huddle_gene_zscore"),
    "pace": ("tempo_gene.csv", "pace_gene_zscore"),
    "box_stacking": ("defensive_scheme_gene.csv", "box_stacking_gene_zscore"),
    "pass_rush": ("defensive_scheme_gene.csv", "pass_rush_gene_zscore"),
    "man_coverage": ("defensive_scheme_gene.csv", "man_coverage_gene_zscore"),
    "shotgun": ("shotgun_gene.csv", "shotgun_gene_zscore"),
}
COMPOSITE_Z = {
    "aggression": ("aggression_gene_by_year.csv", "composite_aggression_zscore"),
    "tempo": ("tempo_gene.csv", "composite_tempo_zscore"),
    "defensive": ("defensive_scheme_gene.csv", "composite_scheme_zscore"),
    "shotgun": ("shotgun_gene.csv", "shotgun_gene_zscore"),
}


def load_war():
    war = pd.read_csv(coach_war_trajectories_path())
    war.columns = war.columns.str.lower()
    return war


def _merge(csv, zcol, war):
    gene = pd.read_csv(GENES / csv).rename(columns={"head_coach": "coach", "season": "year"})
    gene = gene[["coach", "year", zcol]].dropna(subset=[zcol])
    return merge_gene_war(gene, war, "coach", "coach", year_cols=("year", "year"),
                          how="inner", logger=None)


def _eradj_ivw_draws(merged, zcol, n_boot=N_BOOT, seed=0):
    """Coach-block bootstrap of the era-adjusted IVW correlation; returns
    (point, ci_low, ci_high, p, n, n_coaches, draws)."""
    df = _ensure_precision(merged, coach_col="coach", year_col="year")
    clean = df.dropna(subset=[zcol, "annual_war", "coach", "war_weight", "year"]).copy()
    clean["_w16"] = clean["annual_war"].astype(float) * GAMES_PER_SEASON
    gx = within_group_demean(clean, zcol, "year").to_numpy(float)
    gy = within_group_demean(clean, "_w16", "year").to_numpy(float)
    w = clean["war_weight"].to_numpy(float)
    coaches = clean["coach"].to_numpy()
    point = weighted_pearson(gx, gy, w)

    uniq = np.unique(coaches)
    idx_by = {g: np.where(coaches == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[g] for g in samp])
        r = weighted_pearson(gx[rows], gy[rows], w[rows])
        if np.isfinite(r):
            draws.append(r)
    draws = np.asarray(draws)
    if point >= 0:
        p = 2.0 * float(np.mean(draws <= 0))
    else:
        p = 2.0 * float(np.mean(draws >= 0))
    return (float(point), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float(min(1.0, p)),
            int(len(gx)), int(len(uniq)), draws)


def compute_S(csv, zcol, war):
    merged = _merge(csv, zcol, war)
    point, lo, hi, p, n, ncoach, draws = _eradj_ivw_draws(merged, zcol)
    # cross-check against the shared estimator (should match to bootstrap noise)
    wn = war_noise_robustness(merged, zcol, war_col="annual_war",
                              coach_col="coach", year_col="year")
    return {
        "S": point, "ci_low": lo, "ci_high": hi, "p_bootstrap": p,
        "n": n, "n_coaches": ncoach,
        "S_shared_estimator": float(wn.get("r_ivw_eradj", float("nan"))),
    }, draws


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    war = load_war()
    results = {"subtraits": {}, "composites": {},
               "estimator": "era-adjusted inverse-variance-weighted coach-clustered Pearson r",
               "war_pin": "Coach_WAR dffe2f1", "n_boot": N_BOOT}
    draws_store = {}

    logger.info("=== component-level selection S (sub-traits) ===")
    for k, (csv, zcol) in SUBTRAIT_Z.items():
        s, draws = compute_S(csv, zcol, war)
        results["subtraits"][k] = s
        draws_store[f"S__{k}"] = draws
        logger.info("%-14s S=%+.3f [%+.3f,%+.3f] p=%.3f n=%d (shared=%+.3f)",
                    k, s["S"], s["ci_low"], s["ci_high"], s["p_bootstrap"], s["n"],
                    s["S_shared_estimator"])

    logger.info("=== composite selection S ===")
    for k, (csv, zcol) in COMPOSITE_Z.items():
        s, draws = compute_S(csv, zcol, war)
        results["composites"][k] = s
        draws_store[f"S__composite_{k}"] = draws
        logger.info("%-14s S=%+.3f [%+.3f,%+.3f] p=%.3f n=%d",
                    k, s["S"], s["ci_low"], s["ci_high"], s["p_bootstrap"], s["n"])

    with open(OUT_DIR / "ehs_selection_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(OUT_DIR / "ehs_selection_draws.npz", **draws_store)
    logger.info("Wrote %s and ehs_selection_draws.npz", OUT_DIR / "ehs_selection_results.json")


if __name__ == "__main__":
    main()
