#!/usr/bin/env python3
"""
Phylogenetic Signal of Coaching Genes on the Mentor Network (Moran's I)

In comparative biology, phylogenetic signal measures whether related species
resemble each other in a trait more than random pairs do. The coaching analog:
do network-adjacent coaches (a head coach and a protege who served under him and
later became a head coach) resemble each other in a gene more than random coach
pairs?

We quantify this with Moran's I, the network autocorrelation statistic, using a
row-normalized adjacency built from mentor-protege links (coordinator_to_hc +
position_to_hc, undirected). Each node is a head coach; its trait is the career
mean of the era-adjusted gene (within-season demeaned, then averaged). Because
genes exist only for head coaches, the gene-valued subgraph is exactly the
head-coach mentor-protege lineage network.

Significance is assessed by a within-era label-permutation null (preregistered,
osf.io/y2kr5 Section 4): each node's era is its career midpoint, and trait values
are shuffled only among nodes that share an era block (the network held fixed) so
era composition is preserved and shared-era resemblance is not counted as
network signal. Moran's I is recomputed on each of many shuffles. A one-sided p
(share of permutations at least as positive as observed) is the natural test for
positive phylogenetic signal; the null expectation is E[I] = -1/(N-1).

Expected pattern, mirroring the transmission results: positive, significant
signal for shotgun (and defensive scheme); near zero for offensive aggression.

ASCII only. Writes outputs/analysis/phylogenetic_signal_results.json.

Usage:
    python scripts/analysis/analyze_phylogenetic_signal.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import canonicalize_coach_name
from utils.parsimony import within_group_demean

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
TREE_DIR = REPO / "data/processed/coaching_tree"
GENE_DIR = REPO / "data/processed/coaching_genes"
OUT_DIR = REPO / "outputs/analysis"

N_PERM = 5000

# gene key -> (csv, zscore column, name column, label)
GENE_SPECS = {
    "aggression": ("aggression_gene_by_year.csv", "composite_aggression_zscore",
                   "head_coach", "Composite Aggression"),
    "shotgun": ("shotgun_gene.csv", "shotgun_gene_zscore", "head_coach",
                "Shotgun Formation"),
    "tempo": ("tempo_gene.csv", "composite_tempo_zscore", "head_coach",
              "Composite Tempo"),
    "defensive_scheme": ("defensive_scheme_gene.csv", "composite_scheme_zscore",
                         "head_coach", "Defensive Scheme"),
}


def build_edges():
    """Undirected mentor-protege edges as canonical-name pairs (deduped).
    Uses parent_role = Head Coach links (position_to_hc + coordinator_to_hc)."""
    rel = pd.read_csv(TREE_DIR / "relationships.csv")
    hc = rel[rel["relationship_type"].isin(["position_to_hc", "coordinator_to_hc"])]
    edges = set()
    for _, r in hc.iterrows():
        a = canonicalize_coach_name(r["parent_name"])
        b = canonicalize_coach_name(r["child_name"])
        if a and b and a != b:
            edges.add(frozenset((a, b)))
    return [tuple(e) for e in edges]


def career_mean_eradj(spec):
    """Per-coach career mean of the era-adjusted (within-season demeaned) gene
    z-score and the career midpoint (mean season with gene data), keyed by
    canonical name. The midpoint gives each node its era for the within-era
    permutation null (Section 4)."""
    fname, col, name_col, _label = spec
    g = pd.read_csv(GENE_DIR / fname, usecols=[name_col, "season", col])
    g = g.dropna(subset=[col])
    g["eradj"] = within_group_demean(g, col, "season")
    g["coach_canon"] = g[name_col].map(canonicalize_coach_name)
    agg = g.groupby("coach_canon").agg(eradj=("eradj", "mean"),
                                       mid_year=("season", "mean"))
    return agg


def _era_bin(mid_years):
    """Assign each node's career midpoint to an era block (the project's three
    six-season windows), used to hold era fixed under permutation."""
    return pd.cut(mid_years, bins=[2005, 2011, 2017, 2025],
                  labels=["2006-2011", "2012-2017", "2018-2024"]).astype(str).to_numpy()


def _within_era_permute(values, era, rng):
    """Shuffle trait values only within era blocks (era composition held fixed)."""
    out = values.copy()
    for e in np.unique(era):
        idx = np.where(era == e)[0]
        if len(idx) > 1:
            out[idx] = values[rng.permutation(idx)]
    return out


def morans_i(values, W):
    """Moran's I for trait `values` on weight matrix W (rows already normalized).
    I = (N / sum(W)) * (z' W z) / (z' z), z = values - mean."""
    z = values - values.mean()
    n = len(values)
    wsum = W.sum()
    num = z @ (W @ z)
    den = (z * z).sum()
    if den == 0 or wsum == 0:
        return float("nan")
    return float((n / wsum) * (num / den))


def analyze_gene(key, spec, edges):
    agg = career_mean_eradj(spec)
    trait = agg["eradj"]
    valued = set(trait.index)

    # Subgraph among coaches that have this gene; drop isolated nodes.
    adj = {}
    for a, b in edges:
        if a in valued and b in valued:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    nodes = sorted(adj.keys())
    if len(nodes) < 10:
        logger.warning("%s: only %d connected gene-valued nodes; skipping",
                       key, len(nodes))
        return None

    idx = {c: i for i, c in enumerate(nodes)}
    n = len(nodes)
    n_edges = sum(len(v) for v in adj.values()) // 2
    values = np.array([trait[c] for c in nodes], float)
    era = _era_bin(pd.Series([agg.loc[c, "mid_year"] for c in nodes]))

    # Row-normalized weight matrix.
    W = np.zeros((n, n))
    for c, nbrs in adj.items():
        for d in nbrs:
            W[idx[c], idx[d]] = 1.0
    W = W / W.sum(axis=1, keepdims=True)        # every node has >=1 neighbor

    I_obs = morans_i(values, W)
    expected = -1.0 / (n - 1)

    # Within-era permutation null (Section 4): shuffle trait labels only among
    # nodes sharing an era block, so era composition is held fixed and the test
    # asks whether network adjacency predicts resemblance BEYOND shared era.
    rng = np.random.default_rng(0)
    perms = np.empty(N_PERM)
    for i in range(N_PERM):
        perms[i] = morans_i(_within_era_permute(values, era, rng), W)
    p_one = (1 + np.sum(perms >= I_obs)) / (N_PERM + 1)
    p_two = (1 + np.sum(np.abs(perms - expected) >= abs(I_obs - expected))) / (N_PERM + 1)

    era_counts = {e: int(np.sum(era == e)) for e in sorted(set(era))}
    res = {
        "label": spec[3],
        "n_nodes": int(n),
        "n_edges": int(n_edges),
        "morans_I": I_obs,
        "expected_I_null": float(expected),
        "perm_mean": float(perms.mean()),
        "perm_sd": float(perms.std(ddof=1)),
        "z_score": float((I_obs - perms.mean()) / perms.std(ddof=1)),
        "p_perm_one_sided": float(p_one),
        "p_perm_two_sided": float(p_two),
        "n_perm": N_PERM,
        "null": "within-era label permutation (career-midpoint era blocks)",
        "era_block_counts": era_counts,
    }
    logger.info("%-16s I=%.3f (E=%.3f) z=%.2f p1=%.4f  N=%d edges=%d",
                key, I_obs, expected, res["z_score"], p_one, n, n_edges)
    return res


def main():
    edges = build_edges()
    logger.info("Undirected mentor-protege edges: %d", len(edges))
    results = {"network": {"total_edges": len(edges),
                           "edge_types": "position_to_hc + coordinator_to_hc",
                           "directed": False}, "genes": {}}
    for key, spec in GENE_SPECS.items():
        r = analyze_gene(key, spec, edges)
        if r:
            results["genes"][key] = r

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phylogenetic_signal_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", out_path)

    print("\n" + "=" * 70)
    print("PHYLOGENETIC SIGNAL ON THE MENTOR NETWORK (Moran's I)")
    print("=" * 70)
    print(f"{'Gene':18s} {'I':>7s} {'E[I]':>7s} {'z':>6s} {'p(1)':>7s} "
          f"{'N':>4s} {'edges':>6s}")
    for key, r in results["genes"].items():
        print(f"{r['label']:18s} {r['morans_I']:7.3f} {r['expected_I_null']:7.3f} "
              f"{r['z_score']:6.2f} {r['p_perm_one_sided']:7.4f} "
              f"{r['n_nodes']:4d} {r['n_edges']:6d}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
