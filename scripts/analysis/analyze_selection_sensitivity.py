#!/usr/bin/env python3
"""WS15: selection / filtering sensitivity for the headline aggression -> WAR result.

The gene -> WAR analyses inner-join the gene panel to WAR and (downstream) apply
min-plays / min-years filters, which silently drop interim and one-season coaches.
This script makes that attrition VISIBLE and tests whether the headline depends on
it:

  1. Attrition: how many gene coach-years match WAR, how many drop, and whether the
     dropped (unmatched / short-tenure) coaches differ systematically from the kept
     ones on observable dimensions (seasons coached, mean composite gene).
  2. Sensitivity of composite aggression -> WAR (coach-clustered) across samples:
       - all matched coach-years (the headline)
       - excluding one-season coaches
       - excluding partial seasons (< 8 games)
       - career level (one row per coach; WAR is reliable here)
     A result that only survives in the full sample would be selection-driven; one
     that holds across all cuts is robust.

Writes outputs/analysis/selection_sensitivity_results.json. ASCII only.
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import (coach_war_trajectories_path, merge_gene_war,
                              add_war_precision, load_coach_year_games,
                              canonicalize_coach_name)
from utils.parsimony import cluster_bootstrap_corr
from utils.war_noise import career_level_corr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GENE_PATH = Path("data/processed/coaching_genes/aggression_gene_by_year.csv")
GENE_COL = "composite_aggression"
GAMES_PER_SEASON = 16


def _attach_games(df, coach_col="coach", year_col="year"):
    g = load_coach_year_games()
    out = df.copy()
    if g.empty:
        out["games_coached"] = np.nan
        return out
    out["_canon"] = out[coach_col].map(canonicalize_coach_name)
    out["_yr"] = pd.to_numeric(out[year_col], errors="coerce")
    gg = g.rename(columns={"year": "_yr"})
    out = out.merge(gg, left_on=["_canon", "_yr"], right_on=["coach_canon", "_yr"],
                    how="left").drop(columns=["_canon", "_yr", "coach_canon"], errors="ignore")
    return out


def _clustered(df):
    """Coach-clustered composite aggression -> WAR (wins) on a coach-year frame."""
    clean = df[[GENE_COL, "annual_war", "coach"]].dropna()
    if len(clean) < 10:
        return {"insufficient": True, "n": int(len(clean))}
    b = cluster_bootstrap_corr(clean[GENE_COL].to_numpy(float),
                               clean["annual_war"].to_numpy(float) * GAMES_PER_SEASON,
                               clean["coach"].to_numpy(), n_boot=2000, seed=0)
    return {"r": b["r"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
            "p_bootstrap_coach_clustered": b["p_bootstrap"],
            "n": int(len(clean)), "n_coaches": b["n_clusters"]}


def main():
    gene = pd.read_csv(GENE_PATH).rename(columns={"head_coach": "coach", "season": "year"})
    war = pd.read_csv(coach_war_trajectories_path())
    war.columns = war.columns.str.lower()

    results = {}

    # --- 1. Attrition: who drops in the gene<->WAR join ---
    matched = merge_gene_war(gene, war, "coach", "coach",
                             year_cols=("year", "year"), how="inner", logger=logger)
    left = merge_gene_war(gene, war, "coach", "coach",
                          year_cols=("year", "year"), how="left", logger=logger)
    unmatched = left[left["annual_war"].isna()]

    seasons_per_coach = gene.groupby("coach")["year"].nunique()
    matched_coaches = set(matched["coach"].unique())
    unmatched_coaches = set(unmatched["coach"].unique()) - matched_coaches

    def _grp_stats(coach_set):
        sp = seasons_per_coach[seasons_per_coach.index.isin(coach_set)]
        gsub = gene[gene["coach"].isin(coach_set)]
        return {
            "n_coaches": int(len(coach_set)),
            "n_coach_years": int(len(gsub)),
            "mean_seasons": float(sp.mean()) if len(sp) else float("nan"),
            "median_seasons": float(sp.median()) if len(sp) else float("nan"),
            "mean_composite_gene": float(gsub[GENE_COL].mean()) if len(gsub) else float("nan"),
        }

    results["attrition"] = {
        "gene_coach_years_total": int(len(gene)),
        "matched_to_war": int(len(matched)),
        "unmatched_coach_years": int(len(unmatched)),
        "kept_coaches": _grp_stats(matched_coaches),
        "dropped_coaches": _grp_stats(unmatched_coaches),
        "note": ("dropped coaches are those with no WAR baseline -- predominantly "
                 "interims / very short stints; compare seasons + mean gene to check "
                 "they are not a biased slice on the gene dimension"),
    }
    logger.info("Attrition: %d gene coach-years, %d matched, %d dropped (%d coaches)",
                len(gene), len(matched), len(unmatched), len(unmatched_coaches))
    logger.info("  kept coaches: mean %.2f seasons, mean gene %.3f; dropped: mean %.2f seasons, mean gene %.3f",
                results["attrition"]["kept_coaches"]["mean_seasons"],
                results["attrition"]["kept_coaches"]["mean_composite_gene"],
                results["attrition"]["dropped_coaches"]["mean_seasons"],
                results["attrition"]["dropped_coaches"]["mean_composite_gene"])

    # --- 2. Sensitivity across sample definitions ---
    panel = _attach_games(matched)
    panel["n_seasons"] = panel["coach"].map(panel.groupby("coach")["year"].nunique())

    sens = {}
    sens["all_matched"] = _clustered(panel)
    sens["exclude_one_season_coaches"] = _clustered(panel[panel["n_seasons"] >= 2])
    if panel["games_coached"].notna().any():
        full = panel[panel["games_coached"].fillna(0) >= 8]
        sens["exclude_partial_seasons"] = _clustered(full)
        sens["exclude_partial_seasons"]["n_dropped"] = int(
            panel["games_coached"].notna().sum() - len(full))
    sens["career_level"] = career_level_corr(panel, GENE_COL)
    results["sensitivity"] = sens

    print("\n" + "=" * 78)
    print("SELECTION / FILTERING SENSITIVITY: composite aggression -> WAR")
    print("=" * 78)
    for k, v in sens.items():
        if "r" in v:
            print(f"  {k:30s} r={v['r']:+.4f}  clust_p={v.get('p_bootstrap_coach_clustered'):.4f}  "
                  f"n={v.get('n')}  coaches={v.get('n_coaches')}")
        elif "all_coaches" in v:  # career_level dict
            c = v["all_coaches"]
            if "correlation" in c:
                print(f"  {k:30s} r={c['correlation']:+.4f}  p={c['p_value']:.4f}  "
                      f"coaches={c['n_coaches']}")

    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "selection_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved: %s", out_path)


if __name__ == "__main__":
    main()
