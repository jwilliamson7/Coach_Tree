#!/usr/bin/env python3
"""Defensive-aggression -> WAR by coach background type.

Parallel to analyze_aggression_by_coach_type.py, but for the team-level defensive
aggression gene (composite_scheme_zscore, 2016-2024). Splits the gene-WAR
relationship by the head coach's offensive/defensive background and reports
coach-clustered inference (wild cluster bootstrap for small subgroups). Writes
outputs/analysis/defensive_aggression_by_coach_type_results.json. ASCII only.
"""

import json
import logging
from pathlib import Path
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import coach_war_trajectories_path, merge_gene_war
from utils.parsimony import corr_with_small_cluster_guard

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def attach_background(df, bg):
    """Map each coach to an offensive/defensive/other background (same matching
    rules as analyze_aggression_by_coach_type.py)."""
    def lookup(name):
        m = bg[bg["Coach_Name"] == name]
        if len(m) == 0:
            m = bg[bg["Coach_Directory"] == str(name).replace(" ", "_")]
        if len(m) == 0:
            last = name.split()[-1] if " " in str(name) else name
            m = bg[bg["Coach_Name"].str.contains(str(last), case=False, na=False)]
        return m.iloc[0]["Background"] if len(m) else None

    df = df.copy()
    df["Background"] = df["coach"].map(lookup)
    return df


def main():
    war = pd.read_csv(coach_war_trajectories_path())
    war.columns = war.columns.str.lower()

    gene = pd.read_csv("data/processed/coaching_genes/defensive_scheme_gene.csv")
    gene = gene.rename(columns={"head_coach": "coach", "season": "year"})

    merged = merge_gene_war(gene, war, "coach", "coach", year_cols=("year", "year"),
                            how="inner", logger=logger)
    bg = pd.read_csv("data/processed/Coaching/coach_backgrounds_from_history.csv")
    merged = attach_background(merged, bg)

    results = {}
    for bg_type in ["Offensive", "Defensive"]:
        sub = merged[merged["Background"] == bg_type][
            ["composite_scheme_zscore", "annual_war", "coach"]].dropna()
        if len(sub) < 10:
            logger.info(f"{bg_type}: insufficient data (n={len(sub)})")
            continue
        x = sub["composite_scheme_zscore"].to_numpy()
        y = sub["annual_war"].to_numpy()
        r, p = stats.pearsonr(x, y)
        boot = corr_with_small_cluster_guard(
            x, y, sub["coach"].to_numpy(), min_clusters=40, n_boot=2000, seed=0)
        results[bg_type] = {
            "correlation": float(r),
            "p_value": float(p),
            "n": int(len(sub)),
            "ci_low": boot["ci_low"],
            "ci_high": boot["ci_high"],
            "n_coaches": boot["n_clusters"],
            "p_clustered": boot["p_bootstrap"],
            "small_cluster": boot.get("small_cluster"),
            "p_wild_cluster": boot.get("p_wild_cluster"),
        }
        logger.info(
            f"{bg_type}: r={r:.3f} CI[{boot['ci_low']:.2f}, {boot['ci_high']:.2f}] "
            f"p_clust={boot['p_bootstrap']} (wild={boot.get('p_wild_cluster')}) "
            f"n={len(sub)} ({boot['n_clusters']} coaches)")

    out = Path("outputs/analysis/defensive_aggression_by_coach_type_results.json")
    out.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved: {out}")


if __name__ == "__main__":
    main()
