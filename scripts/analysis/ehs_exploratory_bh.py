#!/usr/bin/env python3
"""Benjamini-Hochberg FDR across the EHS EXPLORATORY tests only (Section 4).

The confirmatory tests (C1-C3, H1-H3) are judged by their pre-specified interval
rules and are NOT in any multiple-comparison family.

The family here is exactly the exploratory hypothesis tests the paper reports and
interprets -- nothing more:
  * the within-era Moran's I on the mentor network, one per gene (4 tests), and
  * the preregistered reproductive-fitness test: does coach quality predict the
    exposure-normalized reproductive rate, net of exposure (1 test).
That is 5 tests. The supporting reproductive-fitness quantities (the zero-order
R-vs-career-WAR and R-vs-tenure correlations, and the gene-vs-R correlations) are
reported as descriptive effect sizes, not as FDR-corrected hypothesis tests, so
they are deliberately excluded from the family -- adding unreported tests would
only make the family arbitrary. Including or excluding them does not change the
qualitative verdict (defensive-aggression and tempo Moran's I survive; shotgun
does not).

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

    # (1) Moran's I on the mentor network, one test per gene.
    phylo = json.load(open(OUT_DIR / "phylogenetic_signal_results.json"))
    for k, v in phylo["genes"].items():
        tests.append((f"moransI:{k}", v["p_perm_one_sided"]))

    # (2) Preregistered reproductive-fitness test: does quality predict the
    # exposure-normalized rate? (The registered form is net of exposure; the
    # partial has r ~ 0.07 and is likewise null, so we carry the rate-vs-quality
    # bootstrap p, the only inferential p stored for the rate.)
    repro = json.load(open(OUT_DIR / "reproductive_fitness_results.json"))
    rr = repro["reproductive_rate"]
    tests.append(("repro:rate_vs_mean_war", rr["rate_vs_career_mean_war"]["p_bootstrap"]))

    names = [t[0] for t in tests]
    pvals = [t[1] for t in tests]
    rej, cutoff = bh(pvals)

    results = {
        "q": Q, "n_tests": len(tests),
        "n_significant": int(np.sum(rej)),
        "largest_rejected_p": cutoff,
        "largest_rejected_p_note": (
            "BH is a step-up procedure that rejects every test up to the highest "
            "rank whose p-value clears its step-up bound. This is the LARGEST "
            "p-value among the rejected tests, not an alpha threshold; it can sit "
            "well below q when the p-values have a gap (the next-smallest p fails "
            "its own bound, so the step-up stops early)."),
        "family": "exploratory only (Moran's I per gene + reproductive-rate quality test)",
        "tests": [{"name": n, "p": float(p), "bh_significant": bool(r)}
                  for n, p, r in zip(names, pvals, rej)],
    }
    with open(OUT_DIR / "ehs_exploratory_bh.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 68)
    print(f"EXPLORATORY BH-FDR (q={Q}): {results['n_significant']} of {len(tests)} rejected")
    print(f"  largest rejected p (BH step-up): {cutoff:.4f}")
    print( "  (this is the biggest rejected p, NOT an alpha; a gap in the p-values")
    print( "   can leave it below q even though the step-up bounds rise to q)")
    print("=" * 68)
    for t in sorted(results["tests"], key=lambda d: d["p"]):
        tag = "reject" if t["bh_significant"] else "  --  "
        print(f"  [{tag}] {t['name']:28s} p={t['p']:.4f}")
    print("=" * 68 + "\n")
    logger.info("Wrote %s", OUT_DIR / "ehs_exploratory_bh.json")


if __name__ == "__main__":
    main()
