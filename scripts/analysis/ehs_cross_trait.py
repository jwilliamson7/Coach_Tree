#!/usr/bin/env python3
"""EHS confirmatory C3 and the H1-H3 decision rules (frozen protocol osf.io/y2kr5).

Reads the C1/C2 posterior draws (ehs_posterior_draws.npz + ehs_heritability_
results.json) and the selection draws (ehs_selection_draws.npz + ehs_selection_
results.json) and evaluates the three preregistered hypotheses by their single
pre-specified interval rules:

  C3 / H3  Spearman rank correlation between h^2 and S across the ten sub-traits,
           with a 95% bootstrap interval that resamples the traits and draws h^2
           from its posterior and S from its bootstrap (so both the small-sample
           and the estimation uncertainty enter). H3 holds if the correlation is
           negative with the interval excluding 0.

  H1       Approach family {offensive aggression, defensive aggression} vs
           identity family {shotgun, tempo}: (a) approach mean S minus identity
           mean S, and (b) identity mean h^2 minus approach mean h^2, each a 95%
           interval that must exclude 0 (and be positive) for support.

  H2       Offensive-aggression repeatability minus its h^2; supported if the 95%
           HDI excludes 0 (and is positive): repeatable but not heritable.

Writes outputs/analysis/ehs_confirmatory_summary.json. ASCII only.
"""

import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/analysis")
SUBTRAITS = ["fourth_down", "pass_heavy", "deep_pass", "two_point",
             "no_huddle", "pace", "box_stacking", "pass_rush", "man_coverage", "shotgun"]
APPROACH = ["composite_aggression", "composite_defensive"]
IDENTITY = ["shotgun", "composite_tempo"]  # shotgun composite == shotgun sub-trait


def _hdi(draws, prob=0.95):
    d = np.sort(np.asarray(draws))
    n = len(d)
    k = int(np.floor(prob * n))
    widths = d[k:] - d[:n - k]
    i = int(np.argmin(widths))
    return float(d[i]), float(d[i + k])


def _pick(rng, arr, size):
    return arr[rng.integers(0, len(arr), size)]


def main():
    h2_json = json.load(open(OUT_DIR / "ehs_heritability_results.json"))
    S_json = json.load(open(OUT_DIR / "ehs_selection_results.json"))
    h2npz = np.load(OUT_DIR / "ehs_posterior_draws.npz")
    Snpz = np.load(OUT_DIR / "ehs_selection_draws.npz")

    def h2_draws(ent):
        return h2npz[f"h2__{ent}"]

    def rep_draws(ent):
        return h2npz[f"rep__{ent}"]

    def S_draws(ent):
        return Snpz[f"S__{ent}"]

    # per-entity point estimates
    def h2_med(ent):
        grp = "composites" if ent.startswith("composite_") else "subtraits"
        key = ent.replace("composite_", "")
        return h2_json[grp][key]["c1"]["h2_median"]

    def S_pt(ent):
        grp = "composites" if ent.startswith("composite_") else "subtraits"
        key = ent.replace("composite_", "")
        return S_json[grp][key]["S"]

    rng = np.random.default_rng(0)
    summary = {"protocol": "osf.io/y2kr5"}

    # ------------------------------------------------------------------ C3 / H3
    h2_points = np.array([h2_med(k) for k in SUBTRAITS])
    S_points = np.array([S_pt(k) for k in SUBTRAITS])
    rho_point = float(stats.spearmanr(h2_points, S_points).correlation)

    n = len(SUBTRAITS)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, n, n)
        hh = np.array([_pick(rng, h2_draws(SUBTRAITS[i]), 1)[0] for i in idx])
        ss = np.array([_pick(rng, S_draws(SUBTRAITS[i]), 1)[0] for i in idx])
        if np.std(hh) == 0 or np.std(ss) == 0:
            continue
        r = stats.spearmanr(hh, ss).correlation
        if np.isfinite(r):
            boot.append(r)
    boot = np.asarray(boot)
    c3_lo, c3_hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    h3_supported = bool(rho_point < 0 and c3_hi < 0)
    summary["C3_H3"] = {
        "spearman_rho": rho_point, "ci_low": c3_lo, "ci_high": c3_hi,
        "p_negative": float(np.mean(boot < 0)),
        "H3_supported": h3_supported,
        "subtrait_table": {k: {"h2": float(h2_med(k)), "S": float(S_pt(k))} for k in SUBTRAITS},
    }
    logger.info("C3/H3 Spearman(h2,S) = %.3f [%.3f, %.3f]  H3 supported=%s",
                rho_point, c3_lo, c3_hi, h3_supported)

    # --------------------------------------------------------------------- H1
    K = 4000

    def family_draws(ents, fn):
        cols = [fn(e)[:] for e in ents]
        m = min(len(c) for c in cols)
        idx = [rng.permutation(len(c))[:m] for c in cols]
        stacked = np.vstack([c[i] for c, i in zip(cols, idx)])
        return stacked.mean(axis=0)  # family mean per draw

    # S side: approach mean S - identity mean S (> 0)
    appS = family_draws(APPROACH, S_draws)
    idS = family_draws(IDENTITY, S_draws)
    mS = min(len(appS), len(idS))
    dS = appS[:mS] - idS[:mS]
    s_lo, s_hi = float(np.percentile(dS, 2.5)), float(np.percentile(dS, 97.5))

    # h2 side: identity mean h2 - approach mean h2 (> 0)
    appH = family_draws(APPROACH, h2_draws)
    idH = family_draws(IDENTITY, h2_draws)
    mH = min(len(appH), len(idH))
    dH = idH[:mH] - appH[:mH]
    h_lo, h_hi = float(np.percentile(dH, 2.5)), float(np.percentile(dH, 97.5))

    h1_supported = bool(s_lo > 0 and h_lo > 0)
    summary["H1"] = {
        "approach": APPROACH, "identity": IDENTITY,
        "S_diff_approach_minus_identity": {
            "median": float(np.median(dS)), "ci_low": s_lo, "ci_high": s_hi,
            "p_positive": float(np.mean(dS > 0))},
        "h2_diff_identity_minus_approach": {
            "median": float(np.median(dH)), "ci_low": h_lo, "ci_high": h_hi,
            "p_positive": float(np.mean(dH > 0))},
        "H1_supported": h1_supported,
        "family_means": {
            "approach_mean_S": float(np.mean([S_pt(e) for e in APPROACH])),
            "identity_mean_S": float(np.mean([S_pt(e) for e in IDENTITY])),
            "approach_mean_h2": float(np.mean([h2_med(e) for e in APPROACH])),
            "identity_mean_h2": float(np.mean([h2_med(e) for e in IDENTITY]))},
    }
    logger.info("H1  S diff (approach-identity)=%.3f [%.3f,%.3f]; "
                "h2 diff (identity-approach)=%.3f [%.3f,%.3f]  H1 supported=%s",
                np.median(dS), s_lo, s_hi, np.median(dH), h_lo, h_hi, h1_supported)

    # --------------------------------------------------------------------- H2
    rep = rep_draws("composite_aggression")
    h2a = h2_draws("composite_aggression")
    m = min(len(rep), len(h2a))
    contrast = rep[:m] - h2a[:m]  # repeatability minus heritability
    c_lo, c_hi = _hdi(contrast)
    h2_supported = bool(c_lo > 0)
    summary["H2"] = {
        "trait": "offensive aggression (composite)",
        "repeatability_median": float(np.median(rep)),
        "h2_median": float(np.median(h2a)),
        "repeat_minus_h2_median": float(np.median(contrast)),
        "hdi_low": c_lo, "hdi_high": c_hi,
        "p_repeat_gt_h2": float(np.mean(contrast > 0)),
        "H2_supported": h2_supported,
    }
    logger.info("H2  repeatability-h2 (aggression) = %.3f [%.3f,%.3f]  H2 supported=%s",
                np.median(contrast), c_lo, c_hi, h2_supported)

    with open(OUT_DIR / "ehs_confirmatory_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote %s", OUT_DIR / "ehs_confirmatory_summary.json")

    print("\n" + "=" * 72)
    print("EHS CONFIRMATORY DECISIONS (frozen rules, osf.io/y2kr5)")
    print("=" * 72)
    print(f"H1 approach-selected / identity-inherited : {'SUPPORTED' if h1_supported else 'not supported'}")
    print(f"   S(approach)-S(identity)   = {np.median(dS):+.3f} [{s_lo:+.3f},{s_hi:+.3f}]")
    print(f"   h2(identity)-h2(approach) = {np.median(dH):+.3f} [{h_lo:+.3f},{h_hi:+.3f}]")
    print(f"H2 aggression repeatable not heritable    : {'SUPPORTED' if h2_supported else 'not supported'}")
    print(f"   repeatability-h2          = {np.median(contrast):+.3f} [{c_lo:+.3f},{c_hi:+.3f}]")
    print(f"H3 cross-trait h2 vs S negative           : {'SUPPORTED' if h3_supported else 'not supported'}")
    print(f"   Spearman(h2,S)            = {rho_point:+.3f} [{c3_lo:+.3f},{c3_hi:+.3f}]")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
