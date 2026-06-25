#!/usr/bin/env python3
"""Face validity: do the gene rankings match coaches' known reputations?

A simple, reproducible credibility check for the paper -- the top and bottom
career-mean coaches on each gene should line up with expert/public perception
(e.g. famously aggressive vs conservative coaches). Career mean over coaches with
>= min_seasons seasons. Writes outputs/analysis/gene_face_validity_results.json
and a readable table. ASCII only.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# (label, csv, gene_col, coach_col, higher_means)
GENES = [
    ("Composite Aggression", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "composite_aggression", "head_coach", "more aggressive"),
    ("4th-Down Aggression", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "fourth_down_aggression", "head_coach", "goes for it more"),
    ("Pass-Heavy", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "pass_heavy_aggression", "head_coach", "passes more than expected"),
    ("Deep-Pass", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "deep_pass_aggression", "head_coach", "throws deep more"),
    ("2-Point", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "two_point_aggression", "head_coach", "goes for 2 more"),
    ("Shotgun", "data/processed/coaching_genes/shotgun_gene.csv",
     "shotgun_gene_zscore", "head_coach", "more shotgun"),
    ("Tempo", "data/processed/coaching_genes/tempo_gene.csv",
     "composite_tempo_zscore", "head_coach", "faster / more no-huddle"),
    # Recorded against the HC (consistent with the gene's attribution throughout).
    # Play-calling responsibility varies (some HCs call plays, most delegate to a
    # coordinator); that is a noted LIMITATION, not engineered around here.
    ("Defensive Scheme", "data/processed/coaching_genes/defensive_scheme_gene.csv",
     "composite_scheme_zscore", "head_coach", "more exotic/aggressive D"),
]


def rank_gene(csv, gene_col, coach_col, min_seasons, top_n):
    g = pd.read_csv(csv)
    if gene_col not in g.columns or coach_col not in g.columns:
        return None
    career = (g.dropna(subset=[gene_col, coach_col])
              .groupby(coach_col)
              .agg(gene=(gene_col, "mean"), seasons=(gene_col, "count")))
    career = career[career["seasons"] >= min_seasons].sort_values("gene", ascending=False)
    if career.empty:
        return None
    top = [{"coach": c, "gene": round(float(r.gene), 4), "seasons": int(r.seasons)}
           for c, r in career.head(top_n).iterrows()]
    bottom = [{"coach": c, "gene": round(float(r.gene), 4), "seasons": int(r.seasons)}
              for c, r in career.tail(top_n).iloc[::-1].iterrows()]
    return {"n_coaches": int(len(career)), "top": top, "bottom": bottom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_seasons", type=int, default=3)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    results = {}
    print("=" * 78)
    print("GENE FACE VALIDITY: career-mean leaders / laggards (>= %d seasons)" % args.min_seasons)
    print("=" * 78)
    for label, csv, gene_col, coach_col, direction in GENES:
        if not Path(csv).exists():
            continue
        r = rank_gene(csv, gene_col, coach_col, args.min_seasons, args.top_n)
        if not r:
            continue
        results[label] = {"direction_high": direction, **r}
        print(f"\n{label}  (high = {direction}; n={r['n_coaches']} coaches)")
        print("  TOP:    " + ", ".join(f"{x['coach']} ({x['gene']:+.3f})" for x in r["top"][:6]))
        print("  BOTTOM: " + ", ".join(f"{x['coach']} ({x['gene']:+.3f})" for x in r["bottom"][:6]))

    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gene_face_validity_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_dir / 'gene_face_validity_results.json'}")


if __name__ == "__main__":
    main()
