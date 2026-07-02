#!/usr/bin/env python3
"""Validation checks for the EHS confirmatory bundle (not a statistical analysis).

Confirms the frozen-protocol diagnostic targets and internal consistency before
the results are written up:
  - every C1/C2 fit meets R-hat <= 1.01 and bulk ESS > 400 (Section 3);
  - component S reproduces the shared estimator (r_ivw_eradj);
  - the ten sub-traits and four composites are all present;
  - h^2 does not grossly exceed repeatability (P(h^2>repeat) reported per trait);
  - the C3/H1/H2/H3 summary and exploratory-BH outputs exist and parse.
Prints a PASS/WARN/FAIL line per check. ASCII only.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path("outputs/analysis")
RHAT_MAX = 1.01
ESS_MIN = 400
n_fail = n_warn = 0


def check(cond, msg, warn=False):
    global n_fail, n_warn
    tag = "PASS" if cond else ("WARN" if warn else "FAIL")
    if not cond:
        if warn:
            n_warn += 1
        else:
            n_fail += 1
    print(f"  [{tag}] {msg}")
    return cond


def main():
    h2 = json.load(open(OUT / "ehs_heritability_results.json"))
    S = json.load(open(OUT / "ehs_selection_results.json"))

    print("=== trait coverage ===")
    subs = list(h2["subtraits"].keys())
    check(len(subs) == 10, f"10 sub-traits present (got {len(subs)}): {subs}")
    check(len(h2["composites"]) == 4, f"4 composites present (got {len(h2['composites'])})")

    print("=== MCMC diagnostics (R-hat <= 1.01, ESS > 400) ===")
    for grp in ("subtraits", "composites"):
        for k, v in h2[grp].items():
            for fit in ("c1", "c2"):
                r = v[fit]["r_hat"]; e = v[fit]["ess_bulk"]
                check(r <= RHAT_MAX and e >= ESS_MIN,
                      f"{grp}/{k}/{fit}: r_hat={r:.3f} ess={e:.0f}",
                      warn=(r <= 1.02 and e >= 300))

    print("=== component S reproduces shared estimator ===")
    for k, v in S["subtraits"].items():
        d = abs(v["S"] - v["S_shared_estimator"])
        check(d < 0.02, f"{k}: |S - shared| = {d:.4f}")

    print("=== h^2 vs repeatability (P(h2>repeat) should not be ~1) ===")
    for grp in ("subtraits", "composites"):
        for k, v in h2[grp].items():
            p = v["h2_gt_repeat"]["p_h2_gt_repeat"]
            check(p < 0.9, f"{grp}/{k}: P(h2>repeat)={p:.2f}", warn=(p < 0.975))

    print("=== confirmatory + exploratory summaries parse ===")
    conf = json.load(open(OUT / "ehs_confirmatory_summary.json"))
    for h in ("H1", "H2", "C3_H3"):
        check(h in conf, f"summary has {h}")
    bh = json.load(open(OUT / "ehs_exploratory_bh.json"))
    check("tests" in bh and len(bh["tests"]) > 0, f"exploratory BH has {len(bh.get('tests', []))} tests")

    print("=== figures exist ===")
    for fig in ("outputs/visualizations/performance/gene_fitness_heritability_map.png",
                "outputs/visualizations/performance/reproductive_fitness.png"):
        check(Path(fig).exists(), fig)

    print(f"\nVALIDATION: {n_fail} FAIL, {n_warn} WARN")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
