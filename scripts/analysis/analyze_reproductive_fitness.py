#!/usr/bin/env python3
"""
Reproductive Fitness of NFL Head Coaches (coaching-tree offspring)

The evolutionary-reading companion analysis. In quantitative genetics an
individual's fitness has two faces:

    ecological fitness   -- how well it performs (here: career WAR), and
    reproductive fitness -- how many offspring it leaves.

A coaching "offspring" is a protege who served on a coach's staff and later
became an NFL head coach himself. A coach's reproductive fitness R_i is the
number of DISTINCT such head-coaching offspring. This is the quantity the
popular "coaching tree" tracks (the Walsh tree, the Belichick tree, ...).

We ask three questions:
  1. How is reproductive success distributed? Is it heavy-tailed -- a few
     "supersires" producing a disproportionate share of the league's head
     coaches?
  2. Does winning beget offspring? corr(R_i, career WAR), and -- because both R
     and lifetime WAR grow with tenure -- the partial correlation net of the
     number of head-coaching seasons.
  3. Do particular coaching genes travel with reproductive success? corr(R_i,
     career-mean gene) for each gene.

Offspring must postdate the apprenticeship: a protege counts only if his first
head-coaching season is later than the first season he served under the mentor.
Recent mentors are right-censored (their proteges have not yet had time to
become head coaches); this is reported as a caveat and partly absorbed by the
tenure control.

All inputs are committed. ASCII only. Writes
outputs/analysis/reproductive_fitness_results.json and (via the companion
visualizer) a figure.

Usage:
    python scripts/analysis/analyze_reproductive_fitness.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import canonicalize_coach_name, coach_war_trajectories_path
from utils.parsimony import cluster_bootstrap_corr

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
TREE_DIR = REPO / "data/processed/coaching_tree"
GENE_DIR = REPO / "data/processed/coaching_genes"
OUT_DIR = REPO / "outputs/analysis"

# Gene CSV -> (career-mean column, friendly label)
GENE_SPECS = {
    "aggression": ("aggression_gene_by_year.csv", "composite_aggression_zscore",
                   "head_coach", "Composite Aggression"),
    "shotgun": ("shotgun_gene.csv", "shotgun_gene_zscore",
                "head_coach", "Shotgun Formation"),
    "tempo": ("tempo_gene.csv", "composite_tempo_zscore",
              "head_coach", "Composite Tempo"),
    "defensive_scheme": ("defensive_scheme_gene.csv", "composite_scheme_zscore",
                         "head_coach", "Defensive Scheme"),
}


# --------------------------------------------------------------------------- #
# Coaching tree -> reproductive fitness
# --------------------------------------------------------------------------- #
def load_hc_first_year(start_year=None):
    """Map coach_id -> first NFL head-coaching season, for every coach with an
    NFL HC stint in coaches.json (role_category == 'HC', level == 'NFL').

    When start_year is given, restrict the head-coach universe to coaches whose
    first NFL head-coaching season is in the modern era (>= start_year). Because
    both mentors and offspring are drawn from this map, the whole reproductive-
    fitness tree is then confined to the modern era, matching the window over
    which the genes are measured (offspring postdate the mentor's HC seasons, so
    a modern mentor can only have modern offspring)."""
    with open(TREE_DIR / "coaches.json") as f:
        coaches = json.load(f)
    first_hc = {}
    n_hc_seasons = {}
    id_to_name = {}
    for cid, c in coaches.items():
        id_to_name[cid] = c.get("name", cid)
        hc_years = [int(y) for y, info in c["career"].items()
                    if info.get("role_category") == "HC"
                    and info.get("level") == "NFL"]
        if hc_years and (start_year is None or min(hc_years) >= start_year):
            first_hc[cid] = min(hc_years)
            n_hc_seasons[cid] = len(hc_years)
    logger.info("NFL head coaches in tree%s: %d",
                "" if start_year is None else f" (first HC season >= {start_year})",
                len(first_hc))
    return first_hc, id_to_name, n_hc_seasons


def compute_reproductive_fitness(first_hc, id_to_name, n_hc_seasons):
    """R_i = number of distinct proteges of mentor i (mentor served as Head Coach)
    who later became NFL head coaches.

    Edges: relationships.csv rows whose parent_role is Head Coach
    (position_to_hc + coordinator_to_hc). Deduped to (mentor, protege) pairs,
    keeping the FIRST shared season. A protege counts only if he is an NFL HC and
    his first HC season is strictly after that first shared season.
    """
    rel = pd.read_csv(TREE_DIR / "relationships.csv")
    hc_edges = rel[rel["relationship_type"].isin(
        ["position_to_hc", "coordinator_to_hc"])].copy()

    # First shared season per (mentor=parent, protege=child) pair.
    pairs = (hc_edges.groupby(["parent_id", "child_id"], as_index=False)["year"]
             .min().rename(columns={"year": "first_shared_year"}))

    # Keep pairs whose protege became an NFL HC after the apprenticeship began.
    pairs["child_first_hc"] = pairs["child_id"].map(first_hc)
    offspring = pairs[
        (pairs["child_id"] != pairs["parent_id"])
        & pairs["child_first_hc"].notna()
        & (pairs["child_first_hc"] > pairs["first_shared_year"])
    ].copy()

    # Every NFL HC is a potential mentor (R = 0 if none qualify).
    repro = {cid: 0 for cid in first_hc}
    offspring_map = {cid: [] for cid in first_hc}
    for _, row in offspring.iterrows():
        m = row["parent_id"]
        if m in repro:                       # mentor must himself be an NFL HC
            repro[m] += 1
            offspring_map[m].append(id_to_name.get(row["child_id"], row["child_id"]))

    df = pd.DataFrame({
        "coach_id": list(repro.keys()),
        "coach_name": [id_to_name.get(c, c) for c in repro],
        "first_hc_year": [first_hc[c] for c in repro],
        "n_hc_seasons_tree": [n_hc_seasons.get(c, 0) for c in repro],
        "repro_fitness": [repro[c] for c in repro],
    })
    df["offspring"] = df["coach_id"].map(lambda c: "; ".join(sorted(offspring_map[c])))
    logger.info("Mentors (NFL HCs): %d | total HC-offspring edges: %d",
                len(df), int(df["repro_fitness"].sum()))
    return df.sort_values("repro_fitness", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Career WAR + genes
# --------------------------------------------------------------------------- #
def load_career_war():
    """Per-coach career WAR summary: mean annual WAR (quality), total WAR
    (lifetime), and number of HC seasons (tenure). Keyed by canonical name."""
    war = pd.read_csv(coach_war_trajectories_path())
    war["coach_canon"] = war["Coach"].map(canonicalize_coach_name)
    agg = war.groupby("coach_canon").agg(
        career_mean_war=("Annual_WAR", "mean"),
        career_total_war=("Annual_WAR", "sum"),
        n_hc_seasons=("Annual_WAR", "size"),
    ).reset_index()
    return agg


def load_career_genes():
    """Per-coach career-mean of each gene's z-score, keyed by canonical name."""
    out = None
    for key, (fname, col, name_col, _label) in GENE_SPECS.items():
        g = pd.read_csv(GENE_DIR / fname, usecols=[name_col, col])
        g["coach_canon"] = g[name_col].map(canonicalize_coach_name)
        gm = g.groupby("coach_canon")[col].mean().reset_index()
        gm = gm.rename(columns={col: f"gene_{key}"})
        out = gm if out is None else out.merge(gm, on="coach_canon", how="outer")
    return out


def partial_corr(x, y, z):
    """Partial Pearson correlation of x and y controlling for z (residual method)."""
    x, y, z = np.asarray(x, float), np.asarray(y, float), np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 4:
        return float("nan"), 0
    Z = np.column_stack([np.ones(len(z)), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), len(x)


def corr_block(x, y, ids):
    """Coach-level bootstrap correlation (each coach its own cluster -> ordinary
    nonparametric bootstrap over coaches)."""
    res = cluster_bootstrap_corr(np.asarray(x, float), np.asarray(y, float),
                                 np.asarray(ids), n_boot=2000, seed=0)
    return res


def gini(values):
    """Gini coefficient of a non-negative distribution."""
    v = np.sort(np.asarray(values, float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start_year", type=int, default=1970,
                    help="restrict the head-coach tree to coaches whose first NFL "
                         "HC season is >= this year (default 1970, the post-merger "
                         "modern era; pass 0 for all-time).")
    args = ap.parse_args()
    start_year = args.start_year if args.start_year and args.start_year > 0 else None

    first_hc, id_to_name, n_hc_seasons = load_hc_first_year(start_year)
    repro = compute_reproductive_fitness(first_hc, id_to_name, n_hc_seasons)
    repro["coach_canon"] = repro["coach_name"].map(canonicalize_coach_name)

    war = load_career_war()
    genes = load_career_genes()

    merged = repro.merge(war, on="coach_canon", how="left").merge(
        genes, on="coach_canon", how="left")

    R = repro["repro_fitness"].values
    total_offspring = int(R.sum())

    # ---- (1) distribution --------------------------------------------------- #
    order = np.sort(R)[::-1]
    decile_n = max(1, int(round(0.10 * len(R))))
    top_decile_share = float(order[:decile_n].sum() / total_offspring) if total_offspring else 0.0
    distribution = {
        "n_mentors": int(len(R)),
        "n_mentors_with_offspring": int((R > 0).sum()),
        "pct_with_offspring": float((R > 0).mean() * 100),
        "total_offspring_edges": total_offspring,
        "mean": float(R.mean()),
        "median": float(np.median(R)),
        "max": int(R.max()),
        "std": float(R.std(ddof=1)),
        "cv": float(R.std(ddof=1) / R.mean()) if R.mean() else float("nan"),
        "gini": gini(R),
        "top_decile_share_of_offspring": top_decile_share,
        "top_decile_n_mentors": decile_n,
    }
    supersires = [
        {"coach": r.coach_name, "repro_fitness": int(r.repro_fitness),
         "first_hc_year": int(r.first_hc_year),
         "offspring": r.offspring}
        for r in repro.head(15).itertuples()
    ]

    # ---- (2) does winning beget offspring? ---------------------------------- #
    war_block = merged.dropna(subset=["career_mean_war"]).copy()
    r_mean = corr_block(war_block["career_mean_war"], war_block["repro_fitness"],
                        war_block["coach_id"])
    r_total = corr_block(war_block["career_total_war"], war_block["repro_fitness"],
                         war_block["coach_id"])
    r_tenure = corr_block(war_block["n_hc_seasons"], war_block["repro_fitness"],
                          war_block["coach_id"])
    pr_quality, n_pr = partial_corr(war_block["career_mean_war"],
                                    war_block["repro_fitness"],
                                    war_block["n_hc_seasons"])
    winning = {
        "n_coaches_with_war": int(len(war_block)),
        "repro_vs_career_mean_war": r_mean,
        "repro_vs_career_total_war": r_total,
        "repro_vs_n_hc_seasons": r_tenure,
        "repro_vs_mean_war_partial_tenure": {"r": pr_quality, "n": n_pr},
        "note": ("career-mean WAR is quality (tenure-adjusted); total WAR and "
                 "tenure both grow mechanically with seasons coached, so the "
                 "partial correlation of R on mean WAR net of tenure isolates "
                 "whether winning per se -- not merely lasting -- begets offspring."),
    }

    # ---- (2b) preregistered exposure-normalized rate (Section 4) ------------- #
    # Rate = HC offspring per head-coaching season. Each head-coaching season
    # fields a coordinator staff, so head-coaching seasons are the coordinator-slot
    # exposure that generates future-HC offspring. A minimum-exposure floor keeps
    # one- and two-season coaches from producing unstable 0/1 rate spikes. The
    # registered question is whether coach quality predicts this rate NET of
    # exposure (partial correlation controlling head-coaching seasons).
    EXPOSURE_FLOOR = 3
    rate_all = merged[merged["n_hc_seasons_tree"] >= EXPOSURE_FLOOR].copy()
    rate_all["repro_rate"] = rate_all["repro_fitness"] / rate_all["n_hc_seasons_tree"]
    rate_war = rate_all.dropna(subset=["career_mean_war"]).copy()
    r_rate_quality = corr_block(rate_war["career_mean_war"], rate_war["repro_rate"],
                                rate_war["coach_id"])
    pr_rate_quality, n_rq = partial_corr(rate_war["career_mean_war"],
                                         rate_war["repro_rate"],
                                         rate_war["n_hc_seasons_tree"])
    reproductive_rate = {
        "definition": ("HC offspring per head-coaching season; exposure = number "
                       "of NFL head-coaching seasons (the coordinator-slot exposure "
                       "that generates future-HC proteges)."),
        "exposure_floor_hc_seasons": EXPOSURE_FLOOR,
        "n_mentors_over_floor": int(len(rate_all)),
        "mean_rate": float(rate_all["repro_rate"].mean()),
        "median_rate": float(rate_all["repro_rate"].median()),
        "rate_vs_career_mean_war": r_rate_quality,
        "rate_vs_mean_war_partial_exposure": {"r": pr_rate_quality, "n": n_rq},
        "note": ("exposure-normalized reproductive rate; the quality test is net "
                 "of exposure (partial correlation controlling HC seasons)."),
    }

    # ---- (3) do genes travel with reproductive success? --------------------- #
    # The career-mean gene carries an era signal (it correlates with a coach's
    # career midpoint) and R is right-censored for recent coaches, so the raw
    # gene-R correlation is confounded by longevity/era. We therefore also report
    # the partial correlation net of tenure (number of HC seasons), the dominant
    # driver of R; a gene effect that survives the tenure control is substantive,
    # one that collapses was a longevity/era artifact.
    gene_corrs = {}
    for key in GENE_SPECS:
        col = f"gene_{key}"
        sub = merged.dropna(subset=[col, "n_hc_seasons"]).copy()
        if len(sub) >= 5:
            pr, npr = partial_corr(sub[col], sub["repro_fitness"], sub["n_hc_seasons"])
            gene_corrs[key] = {
                **corr_block(sub[col], sub["repro_fitness"], sub["coach_id"]),
                "label": GENE_SPECS[key][3],
                "partial_tenure_r": pr,
                "partial_tenure_n": npr,
            }

    results = {
        "era": ("all-time" if start_year is None
                else f"modern era, first HC season >= {start_year}"),
        "start_year": start_year,
        "definition": ("reproductive fitness R_i = number of distinct proteges "
                       "who served under head coach i and later became NFL head "
                       "coaches (first HC season after first shared season).edges "
                       "from relationships.csv parent_role=Head Coach."),
        "censoring_caveat": ("mentors active recently are right-censored: their "
                             "proteges may not yet have become head coaches, so "
                             "their R is a lower bound. The tenure control and "
                             "career-mean (vs total) WAR partly absorb this."),
        "distribution": distribution,
        "supersires": supersires,
        "winning_begets_offspring": winning,
        "reproductive_rate": reproductive_rate,
        "gene_vs_reproductive_fitness": gene_corrs,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "reproductive_fitness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)

    # Persist the per-coach table for the visualizer.
    merged.to_csv(OUT_DIR / "reproductive_fitness_coaches.csv", index=False)
    logger.info("Wrote %s", OUT_DIR / "reproductive_fitness_coaches.csv")

    # ---- console summary ---------------------------------------------------- #
    print("\n" + "=" * 70)
    print("REPRODUCTIVE FITNESS OF NFL HEAD COACHES")
    print("=" * 70)
    d = distribution
    print(f"Mentors (NFL HCs):           {d['n_mentors']}")
    print(f"  with >=1 HC offspring:     {d['n_mentors_with_offspring']} "
          f"({d['pct_with_offspring']:.0f}%)")
    print(f"Total HC-offspring edges:    {d['total_offspring_edges']}")
    print(f"Mean / median / max R:       {d['mean']:.2f} / {d['median']:.0f} / {d['max']}")
    print(f"CV / Gini:                   {d['cv']:.2f} / {d['gini']:.2f}")
    print(f"Top-decile ({d['top_decile_n_mentors']} mentors) share "
          f"of all offspring: {d['top_decile_share_of_offspring']*100:.0f}%")
    print("\nTop supersires:")
    for s in supersires[:10]:
        print(f"  {s['coach']:22s} R={s['repro_fitness']:2d}  (since {s['first_hc_year']})")
    print("\nDoes winning beget offspring?  (n=%d coaches with WAR)"
          % winning["n_coaches_with_war"])
    print(f"  R vs career-mean WAR:   r={r_mean['r']:.3f} "
          f"[{r_mean['ci_low']:.3f},{r_mean['ci_high']:.3f}] p={r_mean['p_bootstrap']:.3f}")
    print(f"  R vs career-total WAR:  r={r_total['r']:.3f} "
          f"[{r_total['ci_low']:.3f},{r_total['ci_high']:.3f}] p={r_total['p_bootstrap']:.3f}")
    print(f"  R vs # HC seasons:      r={r_tenure['r']:.3f} "
          f"[{r_tenure['ci_low']:.3f},{r_tenure['ci_high']:.3f}] p={r_tenure['p_bootstrap']:.3f}")
    print(f"  R vs mean WAR | tenure: r={pr_quality:.3f} (partial, n={n_pr})")
    rr = reproductive_rate
    print(f"\nExposure-normalized rate (>= {rr['exposure_floor_hc_seasons']} HC seasons, "
          f"n={rr['n_mentors_over_floor']}):  mean={rr['mean_rate']:.3f}/season")
    print(f"  rate vs mean WAR:       r={rr['rate_vs_career_mean_war']['r']:.3f} "
          f"[{rr['rate_vs_career_mean_war']['ci_low']:.3f},"
          f"{rr['rate_vs_career_mean_war']['ci_high']:.3f}] "
          f"p={rr['rate_vs_career_mean_war']['p_bootstrap']:.3f}")
    print(f"  rate vs WAR | exposure: r={rr['rate_vs_mean_war_partial_exposure']['r']:.3f} "
          f"(partial, n={rr['rate_vs_mean_war_partial_exposure']['n']})")
    print("\nGene vs reproductive fitness  (raw | partial net of tenure):")
    for key, gc in gene_corrs.items():
        print(f"  {gc['label']:22s} r={gc['r']:.3f} "
              f"[{gc['ci_low']:.3f},{gc['ci_high']:.3f}] p={gc['p_bootstrap']:.3f} "
              f"| partial={gc['partial_tenure_r']:.3f}  (n={gc['n']})")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
