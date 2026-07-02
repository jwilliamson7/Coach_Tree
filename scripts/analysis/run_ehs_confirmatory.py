#!/usr/bin/env python3
"""One-command reproducible bundle for the EHS confirmatory + exploratory analyses
(frozen protocol osf.io/y2kr5).

Runs, in dependency order:
  1. ehs_heritability.py      C1 (h^2) + C2 (repeatability), Bayesian, PyMC   [slow]
  2. ehs_selection.py         component + composite selection S (era-adj IVW)
  3. ehs_cross_trait.py       C3 Spearman(h^2,S) + H1/H2/H3 decision rules
  4. analyze_phylogenetic_signal.py   exploratory within-era Moran's I
  5. analyze_reproductive_fitness.py  exploratory exposure-normalized rate
  6. ehs_exploratory_bh.py    BH-FDR across the exploratory family only
  7. visualize_gene_fitness_heritability.py   h^2 x S map figure
  8. visualize_reproductive_fitness.py         reproductive-fitness figure

--skip_heritability reuses the cached posterior draws (steps 1 is the only slow
one; the Bayesian fits take ~25 min). ASCII only.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ANALYSIS = REPO / "scripts/analysis"
VIZ = REPO / "scripts/visualization"


def run(script, args=None):
    cmd = [sys.executable, str(script)] + (args or [])
    print(f"\n>>> {' '.join(str(c) for c in cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {script} (exit {r.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_heritability", action="store_true",
                    help="reuse cached ehs_posterior_draws.npz (skip the slow Bayesian fits)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    args = ap.parse_args()

    if not args.skip_heritability:
        run(ANALYSIS / "ehs_heritability.py",
            ["--draws", str(args.draws), "--tune", str(args.tune)])
    run(ANALYSIS / "ehs_selection.py")
    run(ANALYSIS / "ehs_cross_trait.py")
    run(ANALYSIS / "analyze_phylogenetic_signal.py")
    run(ANALYSIS / "analyze_reproductive_fitness.py")
    run(ANALYSIS / "ehs_exploratory_bh.py")
    run(VIZ / "visualize_gene_fitness_heritability.py")
    run(VIZ / "visualize_reproductive_fitness.py")
    print("\nEHS bundle complete.")


if __name__ == "__main__":
    main()
