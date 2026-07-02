#!/usr/bin/env python3
"""Benjamini-Hochberg FDR across the EHS EXPLORATORY analyses only (Section 4).

The confirmatory tests (C1-C3, H1-H3) are judged by their pre-specified interval
rules and are NOT in any multiple-comparison family. The two declared exploratory
analyses -- the within-era Moran's I on the mentor network and the exposure-
normalized reproductive-fitness rate (and its supporting correlations) -- are
corrected among themselves here under a single BH FDR at q = 0.05.

Reads phylogenetic_signal_results.json and reproductive_fitness_results.json;
writes ehs_exploratory_bh.json. ASCII only.
"""

import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/analysis")
Q = 0.05


def bh(pvals):
    """Benjamini-Hochberg: return (rejected bool array, adjusted-cutoff p)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * Q
    passed = ranked <= thresh
    if not passed.any():
        cutoff = 0.0
    else:
        kmax = np.max(np.where(passed)[0])
        cutoff = ranked[kmax]
    rej = p <= cutoff
    return rej, float(cutoff)


def main():
    tests = []  # (name, p)

    phylo = json.load(open(OUT_DIR / "phylogenetic_signal_results.json"))
    for k, v in phylo["genes"].items():
        tests.append((f"moransI:{k}", v["p_perm_one_sided"]))

    repro = json.load(open(OUT_DIR / "reproductive_fitness_results.json"))
    w = repro["winning_begets_offspring"]
    tests.append(("repro:R_vs_mean_war", w["repro_vs_career_mean_war"]["p_bootstrap"]))
    tests.append(("repro:R_vs_total_war", w["repro_vs_career_total_war"]["p_bootstrap"]))
    rr = repro["reproductive_rate"]
    tests.append(("repro:rate_vs_mean_war", rr["rate_vs_career_mean_war"]["p_bootstrap"]))
    for k, gc in repro["gene_vs_reproductive_fitness"].items():
        tests.append((f"repro:gene_{k}_vs_R", gc["p_bootstrap"]))

    names = [t[0] for t in tests]
    pvals = [t[1] for t in tests]
    rej, cutoff = bh(pvals)

    results = {
        "q": Q, "n_tests": len(tests), "bh_cutoff_p": cutoff,
        "n_significant": int(np.sum(rej)),
        "family": "exploratory only (Moran's I + reproductive-fitness rate/correlations)",
        "tests": [{"name": n, "p": float(p), "bh_significant": bool(r)}
                  for n, p, r in zip(names, pvals, rej)],
    }
    with open(OUT_DIR / "ehs_exploratory_bh.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 64)
    print(f"EXPLORATORY BH-FDR (q={Q}): {results['n_significant']}/{len(tests)} "
          f"significant, cutoff p<={cutoff:.4f}")
    print("=" * 64)
    for t in sorted(results["tests"], key=lambda d: d["p"]):
        print(f"  {'*' if t['bh_significant'] else ' '} {t['name']:28s} p={t['p']:.4f}")
    print("=" * 64 + "\n")
    logger.info("Wrote %s", OUT_DIR / "ehs_exploratory_bh.json")


if __name__ == "__main__":
    main()
