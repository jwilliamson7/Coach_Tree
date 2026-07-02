#!/usr/bin/env python3
"""EHS confirmatory C1 (heritability) and C2 (repeatability).

Frozen protocol osf.io/y2kr5, Sections 3-4.

C1  For each trait, transmissibility h^2 is the coefficient of a Bayesian
    multilevel parent-offspring regression: the protege's own head-coaching
    phenotype (outcome) on the phenotype of the head coach they apprenticed under
    (predictor), the latter measured on the mentor's team over the overlapping
    coordinator seasons. The mentor predictor enters as a latent variable
    carrying its known sampling variance (errors-in-variables), correcting the
    downward attenuation that predictor noise would otherwise cause. Because both
    phenotypes are already contemporary-group adjusted and z-scored, the slope is
    the standardized transmissibility, with no Mendelian doubling. Priors:
    Normal(0,1) on intercept and slope, half-Normal(0,1) on the residual SD.
    Robustness: (a) a mentor varying intercept (half-Normal(0,1) on the mentor
    SD), (b) a frequentist mentor-clustered OLS slope.

C2  Per-trait repeatability is a variance-components ICC on the full head-coach
    coach-season panel, with the per-season sampling variance carried as an
    explicit measurement-error level so the ICC is the latent-trait repeatability
    (between-coach variance over between plus true within-coach variance), the
    heritability ceiling.

Sampling: 4 chains, non-centered, target R-hat <= 1.01 and bulk ESS > 400.

Writes outputs/analysis/ehs_heritability_results.json (posterior summaries +
diagnostics) and outputs/analysis/ehs_posterior_draws.npz (h^2 and repeatability
draw vectors, for the H1/H2/H3 combination step). ASCII only.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pymc as pm
import arviz as az

from utils.ehs_traits import (
    SUBTRAITS, COMPOSITES, build_all_subtrait_panels, build_composite_panel,
    get_transitions, build_pairs, build_composite_pairs,
    build_subtrait_obs, build_composite_obs,
)
from utils.parsimony import cluster_robust_ols

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/analysis")
DRAWS = 2000
TUNE = 2000
CHAINS = 4
SEED = 42


def _diag(idata, var):
    arr = idata.posterior[var].values  # (chains, draws) for a scalar parameter
    return float(az.rhat(arr)), float(az.ess(arr))


def _hdi(draws, prob=0.95):
    h = az.hdi(np.asarray(draws).reshape(-1), prob=prob)
    return float(h[0]), float(h[1])


def fit_c1(obs, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED, mentor_re=False):
    """Unified season-grain errors-in-variables parent-offspring model.

    Each protege i has a latent mentor level m_i and a latent own level a_i, with
    a_i = alpha + beta*m_i + residual. Individual mentor-overlap seasons and
    protege head-coaching seasons load on those latents with variance
    tau^2 (true within-coach season-to-season) + se^2 (known measurement).
    h^2 is the parent-offspring regression slope at the coach-season grain,
        h2 = beta * sigma_m^2 / (sigma_m^2 + tau^2)  =  beta_latent * repeatability,
    which is bounded by the mentor repeatability sigma_m^2/(sigma_m^2+tau^2).
    Returns (h2_draws, summary dict).
    """
    obs = obs.copy()
    codes = {p: i for i, p in enumerate(sorted(obs["pidx"].unique()))}
    obs["i"] = obs["pidx"].map(codes)
    N = len(codes)
    M = obs[obs["side"] == "M"]
    P = obs[obs["side"] == "P"]
    xm = M["val"].to_numpy(float); sem = np.maximum(M["se"].to_numpy(float), 1e-3)
    im = M["i"].to_numpy(int)
    yp = P["val"].to_numpy(float); sep = np.maximum(P["se"].to_numpy(float), 1e-3)
    ip = P["i"].to_numpy(int)
    prot_mentor = obs.groupby("i")["mentor_id"].first().reindex(range(N)).astype(str).to_numpy()
    m_levels, m_of_prot = np.unique(prot_mentor, return_inverse=True)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        sigma_m = pm.HalfNormal("sigma_m", 1.0)
        tau = pm.HalfNormal("tau", 1.0)
        m_raw = pm.Normal("m_raw", 0.0, 1.0, shape=N)
        mlat = mu + sigma_m * m_raw

        alpha = pm.Normal("alpha", 0.0, 1.0)
        beta = pm.Normal("beta", 0.0, 1.0)
        sigma_a = pm.HalfNormal("sigma_a", 1.0)
        a_mu = alpha + beta * mlat
        if mentor_re and len(m_levels) < N:
            sigma_u = pm.HalfNormal("sigma_u", 1.0)
            u_raw = pm.Normal("u_raw", 0.0, 1.0, shape=len(m_levels))
            a_mu = a_mu + (u_raw * sigma_u)[m_of_prot]
        a_raw = pm.Normal("a_raw", 0.0, 1.0, shape=N)
        alat = a_mu + sigma_a * a_raw

        pm.Normal("xm", mu=mlat[im], sigma=pm.math.sqrt(tau**2 + sem**2), observed=xm)
        pm.Normal("yp", mu=alat[ip], sigma=pm.math.sqrt(tau**2 + sep**2), observed=yp)

        Vp = sigma_m**2 + tau**2
        pm.Deterministic("h2", beta * sigma_m**2 / Vp)
        pm.Deterministic("R_mentor", sigma_m**2 / Vp)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=0.99, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": False})

    h2_draws = idata.posterior["h2"].values.reshape(-1)
    beta_draws = idata.posterior["beta"].values.reshape(-1)
    rhat, ess = _diag(idata, "h2")
    lo, hi = _hdi(h2_draws)
    return h2_draws, {
        "n_proteges": int(N), "n_mentors": int(len(m_levels)),
        "n_mentor_seasons": int(len(xm)), "n_protege_seasons": int(len(yp)),
        "h2_median": float(np.median(h2_draws)),
        "h2_mean": float(np.mean(h2_draws)),
        "hdi_low": lo, "hdi_high": hi,
        "p_positive": float(np.mean(h2_draws > 0)),
        "raw_latent_slope_median": float(np.median(beta_draws)),
        "mentor_repeatability_median": float(np.median(idata.posterior["R_mentor"].values)),
        "r_hat": rhat, "ess_bulk": ess,
    }


def freq_clustered_slope(pairs):
    """Frequentist mentor-clustered OLS slope of y on observed x (sanity cross-check)."""
    x = pairs["x"].to_numpy(float).reshape(-1, 1)
    y = pairs["y"].to_numpy(float)
    clusters = pairs["mentor_id"].astype(str).to_numpy()
    res = cluster_robust_ols(x, y, clusters, ["slope"])
    c = res["coefficients"]["slope"]
    return {"slope": c["coefficient"], "std_error": c["std_error"],
            "p_value": c["p_value"], "n_clusters": res["n_clusters"]}


def fit_c2(panel, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED):
    """Measurement-error variance-components ICC (latent repeatability)."""
    z = panel["z"].to_numpy(float)
    se = np.maximum(np.sqrt(panel["samp_var"].to_numpy(float)), 1e-3)
    coaches, c_idx = np.unique(panel["coach"].to_numpy(), return_inverse=True)
    n = len(z)

    with pm.Model() as model:
        mu = pm.Normal("mu", 0.0, 1.0)
        sigma_between = pm.HalfNormal("sigma_between", 1.0)
        a_raw = pm.Normal("a_raw", 0.0, 1.0, shape=len(coaches))
        a = mu + sigma_between * a_raw
        sigma_within = pm.HalfNormal("sigma_within", 1.0)
        z_true = pm.Normal("z_true", mu=a[c_idx], sigma=sigma_within, shape=n)
        pm.Normal("z_obs", mu=z_true, sigma=se, observed=z)
        repeat = pm.Deterministic(
            "repeat", sigma_between**2 / (sigma_between**2 + sigma_within**2))
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=1,
                          target_accept=0.99, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": False})

    rep_draws = idata.posterior["repeat"].values.reshape(-1)
    rhat, ess = _diag(idata, "repeat")
    lo, hi = _hdi(rep_draws)
    return rep_draws, {
        "n_obs": int(n), "n_coaches": int(len(coaches)),
        "repeatability_median": float(np.median(rep_draws)),
        "hdi_low": lo, "hdi_high": hi,
        "r_hat": rhat, "ess_bulk": ess,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--tune", type=int, default=TUNE)
    ap.add_argument("--traits", nargs="*", default=None,
                    help="subset of trait keys to run (default: all 10 + 4 composites)")
    ap.add_argument("--no_mentor_re", action="store_true",
                    help="skip the mentor varying-intercept robustness fit")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panels = build_all_subtrait_panels()
    transitions, analyzer = get_transitions(min_years=1)

    entities = list(SUBTRAITS.keys()) + [f"composite_{k}" for k in COMPOSITES]
    if args.traits:
        entities = args.traits

    results = {"subtraits": {}, "composites": {}, "meta": {
        "draws": args.draws, "tune": args.tune, "chains": CHAINS,
        "protocol": "osf.io/y2kr5"}}
    h2_draws_store, rep_draws_store = {}, {}

    for ent in entities:
        is_comp = ent.startswith("composite_")
        key = ent.replace("composite_", "")
        logger.info("=== %s ===", ent)

        if is_comp:
            obs = build_composite_obs(key, panels, transitions, analyzer)
            pairs = build_composite_pairs(key, panels, transitions, analyzer)
            panel = build_composite_panel(key, panels)
        else:
            obs = build_subtrait_obs(key, panels[key], transitions, analyzer)
            pairs = build_pairs(key, panels[key], transitions, analyzer)
            panel = panels[key]

        entry = {}
        # C1 main (unified season-grain EIV)
        beta_draws, c1 = fit_c1(obs, draws=args.draws, tune=args.tune)
        entry["c1"] = c1
        h2_draws_store[ent] = beta_draws
        logger.info("  C1 h2=%.3f [%.3f,%.3f] p+=%.3f rhat=%.3f ess=%.0f proteges=%d",
                    c1["h2_median"], c1["hdi_low"], c1["hdi_high"], c1["p_positive"],
                    c1["r_hat"], c1["ess_bulk"], c1["n_proteges"])
        # C1 mentor-RE robustness
        if not args.no_mentor_re:
            try:
                _, c1re = fit_c1(obs, draws=args.draws, tune=args.tune, mentor_re=True)
                entry["c1_mentor_re"] = c1re
            except Exception as e:
                entry["c1_mentor_re"] = {"error": str(e)}
        # C1 frequentist clustered (on the averaged pairs, sanity cross-check)
        entry["c1_freq_clustered"] = freq_clustered_slope(pairs)

        # C2 repeatability
        rep_draws, c2 = fit_c2(panel, draws=args.draws, tune=args.tune)
        entry["c2"] = c2
        rep_draws_store[ent] = rep_draws
        logger.info("  C2 repeat=%.3f [%.3f,%.3f] rhat=%.3f ess=%.0f coaches=%d",
                    c2["repeatability_median"], c2["hdi_low"], c2["hdi_high"],
                    c2["r_hat"], c2["ess_bulk"], c2["n_coaches"])

        # P(h2 > repeatability): independent posteriors, paired elementwise
        k = min(len(beta_draws), len(rep_draws))
        contrast = beta_draws[:k] - rep_draws[:k]
        entry["h2_gt_repeat"] = {
            "p_h2_gt_repeat": float(np.mean(contrast > 0)),
            "repeat_minus_h2_median": float(np.median(-contrast)),
            "repeat_minus_h2_hdi": list(_hdi(-contrast)),
        }

        (results["composites"] if is_comp else results["subtraits"])[key] = entry

    with open(OUT_DIR / "ehs_heritability_results.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(OUT_DIR / "ehs_posterior_draws.npz",
             **{f"h2__{k}": v for k, v in h2_draws_store.items()},
             **{f"rep__{k}": v for k, v in rep_draws_store.items()})
    logger.info("Wrote %s and ehs_posterior_draws.npz", OUT_DIR / "ehs_heritability_results.json")


if __name__ == "__main__":
    main()
