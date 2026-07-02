#!/usr/bin/env python3
"""EHS exploratory: horizontal (peer-to-peer) transmission of coaching genes.

The confirmatory pipeline measures VERTICAL transmission (mentor->protege h^2) on
era-adjusted phenotypes, which by construction remove the league-wide drift. This
script measures the complementary HORIZONTAL channel, which lives IN that drift:
do head coaches converge each season on the prevailing league level of a trait,
beyond their own persistence and their own baseline? That convergence is the
signature of frequency-biased (conformist) social learning -- copying what is
common -- and, unlike a payoff-biased test, it does not use (noisy) WAR at all.

We deliberately use the ABSOLUTE standardized phenotype (z-scored over the data
window, NOT within-season demeaned), because the season-to-season drift is the
horizontal signal, not a confound to be removed here.

Model (Bayesian, for consistency with the confirmatory C1/C2 PyMC models). For a
coach i, phenotype p, and season t with a contiguous next season t+1:

    z_{i,p,t+1} ~ Normal( a_p + u_{i,p} + phi_p * z_{i,p,t}
                          + kappa_p * Mloo_{p,t} , sigma_p )

  u_{i,p} ~ Normal(0, sigma_u)         coach-phenotype baseline (own level)
  kappa_p ~ Normal(mu_kappa, sigma_kappa)   conformity/frequency bias (partial
                                            pooling across traits IS the
                                            multiplicity control -- no BH)
  Mloo_{p,t}                           leave-one-out league mean in season t

kappa_p > 0 net of z_now and the coach baseline = the coach tracks the crowd =
frequency-biased horizontal transmission. mu_kappa is the population-level
conformity across traits.

Primary pooled fit is over the ten frozen sub-traits (the independent set, as in
C3); the four gene-level composites are fit singly for the map/forest. A robustness
fit adds a per-trait linear season trend so kappa is convergence-on-the-crowd
beyond a common linear drift (a conservative floor; conformity and the drift it
produces are collinear, so kappa attenuates but its sign is the check).

Writes outputs/analysis/horizontal_transmission_results.json. ASCII only.
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

from utils.ehs_traits import SUBTRAITS, COMPOSITES, GENES_DIR
from utils.data_paths import canonicalize_coach_name
from utils.parsimony import cluster_robust_ols

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/analysis")
DRAWS = 1000
TUNE = 1000
CHAINS = 4
SEED = 42
MIN_SEASON_COACHES = 4   # need >3 peers to form a leave-one-out crowd mean


# --------------------------------------------------------------------------- #
# Absolute (un-era-adjusted) coach-season panels
# --------------------------------------------------------------------------- #
def build_absolute_panel(key):
    """Coach-season panel carrying the ABSOLUTE standardized phenotype z_abs
    (z-scored over the trait window, NOT within-season demeaned). Play-weighted
    over any duplicate coach-season rows. Columns: coach, head_coach, season,
    z_abs."""
    spec = SUBTRAITS[key]
    df = pd.read_csv(GENES_DIR / spec["csv"]).copy()
    y0, y1 = spec["window"]
    df = df[(df["season"] >= y0) & (df["season"] <= y1)]
    raw, plays = spec["raw"], spec["plays"]
    df = df.dropna(subset=[raw, plays])
    df = df[df[plays].astype(float) > 0]
    if df.empty:
        return pd.DataFrame(columns=["coach", "head_coach", "season", "z_abs"])
    sd = float(df[raw].std(ddof=0)) or 1.0
    mean = float(df[raw].mean())
    df["z_abs"] = (df[raw].astype(float) - mean) / sd
    df["coach"] = df["head_coach"].map(canonicalize_coach_name)
    w = df["total_plays"].astype(float) if "total_plays" in df else df[plays].astype(float)
    df["_w"] = np.where(w > 0, w, 1.0)
    df["_wz"] = df["z_abs"] * df["_w"]
    # play-weighted mean per coach-season (collapses any mid-season splits)
    g = df.groupby(["coach", "season"], as_index=False).agg(
        head_coach=("head_coach", "first"), _wz=("_wz", "sum"), _w=("_w", "sum"))
    g["z_abs"] = g["_wz"] / g["_w"]
    return g[["coach", "head_coach", "season", "z_abs"]].reset_index(drop=True)


def build_absolute_composite_panel(key, subpanels):
    """Absolute composite panel: mean of the available component z_abs on the
    shared coach-season grain."""
    comp = COMPOSITES[key]
    frames = []
    for c in comp["components"]:
        p = subpanels[c][["coach", "season", "z_abs"]].rename(columns={"z_abs": f"z_{c}"})
        frames.append(p)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["coach", "season"], how="outer")
    zcols = [f"z_{c}" for c in comp["components"]]
    out["z_abs"] = out[zcols].mean(axis=1, skipna=True)
    out = out.dropna(subset=["z_abs"])
    out["head_coach"] = out["coach"]
    return out[["coach", "head_coach", "season", "z_abs"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Transition table (z_now, crowd mean, z_next)
# --------------------------------------------------------------------------- #
def build_transitions(panel, pheno):
    """From an absolute coach-season panel, build contiguous t->t+1 transitions
    with the leave-one-out league mean of season t. Requires > MIN_SEASON_COACHES
    coaches in season t to form the crowd mean."""
    d = panel.copy()
    ssum = d.groupby("season")["z_abs"].transform("sum")
    scnt = d.groupby("season")["z_abs"].transform("count")
    d = d.assign(Mloo=(ssum - d["z_abs"]) / (scnt - 1), scnt=scnt)
    d = d[d["scnt"] >= MIN_SEASON_COACHES]
    nxt = d[["coach", "season", "z_abs"]].rename(columns={"season": "ps", "z_abs": "z_next"})
    nxt["season"] = nxt["ps"] - 1
    p = d.merge(nxt[["coach", "season", "z_next"]], on=["coach", "season"], how="inner")
    p = p.rename(columns={"z_abs": "z_now", "Mloo": "M_crowd"})
    p["pheno"] = pheno
    return p[["coach", "pheno", "season", "z_now", "M_crowd", "z_next"]].dropna()


# --------------------------------------------------------------------------- #
# Bayesian fits
# --------------------------------------------------------------------------- #
def _hdi(draws, prob=0.95):
    h = az.hdi(np.asarray(draws).reshape(-1), prob=prob)
    return float(h[0]), float(h[1])


def _scalar_diag(idata, var, idx=None):
    arr = idata.posterior[var]
    a = arr.values if idx is None else arr.values[..., idx]
    return float(az.rhat(a)), float(az.ess(a))


def fit_pooled(T, phenos, with_trend=False, draws=DRAWS, tune=TUNE, seed=SEED):
    """Hierarchical conformity model across `phenos` (partial pooling of kappa)."""
    d = T[T["pheno"].isin(phenos)].copy()
    pcode = {p: i for i, p in enumerate(phenos)}
    d["pi"] = d["pheno"].map(pcode)
    cp = (d["coach"] + "||" + d["pheno"]).to_numpy()
    cp_levels, cp_idx = np.unique(cp, return_inverse=True)
    pidx = d["pi"].to_numpy(int)
    znow = d["z_now"].to_numpy(float)
    mcr = d["M_crowd"].to_numpy(float)
    y = d["z_next"].to_numpy(float)
    scen = d["season"].to_numpy(float) - d["season"].to_numpy(float).mean()
    P = len(phenos)

    with pm.Model():
        mu_k = pm.Normal("mu_kappa", 0.0, 0.5)
        sig_k = pm.HalfNormal("sigma_kappa", 0.5)
        k_raw = pm.Normal("k_raw", 0.0, 1.0, shape=P)
        kappa = pm.Deterministic("kappa", mu_k + sig_k * k_raw)
        mu_phi = pm.Normal("mu_phi", 0.3, 0.5)
        sig_phi = pm.HalfNormal("sigma_phi", 0.5)
        phi = mu_phi + sig_phi * pm.Normal("phi_raw", 0.0, 1.0, shape=P)
        a = pm.Normal("a", 0.0, 1.0, shape=P)
        sig_u = pm.HalfNormal("sigma_u", 1.0)
        u = sig_u * pm.Normal("u_raw", 0.0, 1.0, shape=len(cp_levels))
        sigma = pm.HalfNormal("sigma", 1.0, shape=P)
        mu = a[pidx] + u[cp_idx] + phi[pidx] * znow + kappa[pidx] * mcr
        if with_trend:
            tau = mu_phi * 0 + pm.Normal("tau", 0.0, 0.5, shape=P)
            mu = mu + tau[pidx] * scen
        pm.Normal("y", mu=mu, sigma=sigma[pidx], observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=CHAINS, cores=1,
                          target_accept=0.95, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": False})

    muk = idata.posterior["mu_kappa"].values.reshape(-1)
    rhat, ess = _scalar_diag(idata, "mu_kappa")
    lo, hi = _hdi(muk)
    out = {
        "mu_kappa_median": float(np.median(muk)),
        "hdi_low": lo, "hdi_high": hi,
        "p_positive": float(np.mean(muk > 0)),
        "r_hat": rhat, "ess_bulk": ess,
        "n_transitions": int(len(d)), "n_phenotypes": P,
    }
    per = {}
    kd_all = idata.posterior["kappa"].values  # (chains, draws, P)
    for p, i in pcode.items():
        kd = kd_all[..., i].reshape(-1)
        klo, khi = _hdi(kd)
        rh, es = _scalar_diag(idata, "kappa", idx=i)
        sub = d[d["pi"] == i]
        per[p] = {
            "kappa_median": float(np.median(kd)),
            "hdi_low": klo, "hdi_high": khi,
            "p_positive": float(np.mean(kd > 0)),
            "n_transitions": int(len(sub)), "n_coaches": int(sub["coach"].nunique()),
            "r_hat": rh, "ess_bulk": es, "source": "pooled_shrunk",
        }
    return out, per


def fit_single(sub, draws=DRAWS, tune=TUNE, seed=SEED):
    """Single-phenotype conformity model (coach random intercepts)."""
    coach_levels, cidx = np.unique(sub["coach"].to_numpy(), return_inverse=True)
    znow = sub["z_now"].to_numpy(float)
    mcr = sub["M_crowd"].to_numpy(float)
    y = sub["z_next"].to_numpy(float)
    with pm.Model():
        a = pm.Normal("a", 0.0, 1.0)
        phi = pm.Normal("phi", 0.3, 0.5)
        kappa = pm.Normal("kappa", 0.0, 0.5)
        sig_u = pm.HalfNormal("sigma_u", 1.0)
        u = sig_u * pm.Normal("u_raw", 0.0, 1.0, shape=len(coach_levels))
        sigma = pm.HalfNormal("sigma", 1.0)
        mu = a + u[cidx] + phi * znow + kappa * mcr
        pm.Normal("y", mu=mu, sigma=sigma, observed=y)
        idata = pm.sample(draws=draws, tune=tune, chains=CHAINS, cores=1,
                          target_accept=0.95, random_seed=seed, progressbar=False,
                          idata_kwargs={"log_likelihood": False})
    kd = idata.posterior["kappa"].values.reshape(-1)
    rhat, ess = _scalar_diag(idata, "kappa")
    lo, hi = _hdi(kd)
    return {
        "kappa_median": float(np.median(kd)),
        "hdi_low": lo, "hdi_high": hi,
        "p_positive": float(np.mean(kd > 0)),
        "n_transitions": int(len(sub)), "n_coaches": int(len(coach_levels)),
        "r_hat": rhat, "ess_bulk": ess, "source": "single",
    }


def freq_crosscheck(sub):
    """Frequentist coach-clustered anchor: z_next ~ z_now + mu_excl + M_crowd,
    with mu_excl the coach baseline computed leave-outcome-out. Reports the
    M_crowd coefficient and its coach-clustered p."""
    d = sub.copy()
    csum = d.groupby("coach")["z_next"].transform("sum")
    ccnt = d.groupby("coach")["z_next"].transform("count")
    d["mu_excl"] = np.where(ccnt > 1, (csum - d["z_next"]) / (ccnt - 1), 0.0)
    d = d[ccnt > 1]
    if len(d) < 25 or d["coach"].nunique() < 5:
        return {"M_crowd_beta": None, "p_value": None, "n": int(len(d))}
    X = np.column_stack([d["z_now"], d["mu_excl"], d["M_crowd"]])
    res = cluster_robust_ols(X, d["z_next"].to_numpy(float), d["coach"].to_numpy(),
                             ["z_now", "mu_excl", "M_crowd"])
    c = res["coefficients"]["M_crowd"]
    return {"M_crowd_beta": c["coefficient"], "p_value": c["p_value"],
            "n": int(len(d)), "n_coaches": res["n_clusters"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--tune", type=int, default=TUNE)
    ap.add_argument("--no_trend_robustness", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subpanels = {k: build_absolute_panel(k) for k in SUBTRAITS}
    comp_panels = {k: build_absolute_composite_panel(k, subpanels)
                   for k in COMPOSITES if k != "shotgun"}  # shotgun composite == subtrait

    # transition tables
    sub_T = pd.concat([build_transitions(subpanels[k], k) for k in SUBTRAITS],
                      ignore_index=True)
    comp_T = {k: build_transitions(comp_panels[k], f"composite_{k}") for k in comp_panels}

    subtraits = list(SUBTRAITS.keys())
    logger.info("Pooled hierarchical fit over %d sub-traits (%d transitions)...",
                len(subtraits), len(sub_T))
    pooled, per_sub = fit_pooled(sub_T, subtraits, with_trend=False,
                                 draws=args.draws, tune=args.tune)
    logger.info("  mu_kappa=%.3f [%.3f,%.3f] p+=%.3f rhat=%.3f",
                pooled["mu_kappa_median"], pooled["hdi_low"], pooled["hdi_high"],
                pooled["p_positive"], pooled["r_hat"])

    pooled_detrended = None
    if not args.no_trend_robustness:
        logger.info("Robustness: pooled fit WITH per-trait linear season trend...")
        pooled_detrended, _ = fit_pooled(sub_T, subtraits, with_trend=True,
                                         draws=args.draws, tune=args.tune)
        logger.info("  mu_kappa (detrended)=%.3f [%.3f,%.3f] p+=%.3f",
                    pooled_detrended["mu_kappa_median"], pooled_detrended["hdi_low"],
                    pooled_detrended["hdi_high"], pooled_detrended["p_positive"])

    # per-phenotype record (10 sub-traits from pooled; 3 composites single)
    per_pheno = {}
    fam = {k: "sub" for k in subtraits}
    for k in subtraits:
        rec = per_sub[k]
        rec["freq"] = freq_crosscheck(sub_T[sub_T["pheno"] == k])
        rec["label"] = SUBTRAITS[k]["label"]
        rec["family"] = COMPOSITES.get(  # family of the parent composite
            next((ck for ck, cv in COMPOSITES.items() if k in cv["components"]), ""),
            {}).get("family", SUBTRAITS[k].get("side", ""))
        rec["is_composite"] = False
        per_pheno[k] = rec
    for k in comp_panels:
        logger.info("Single fit: composite %s (%d transitions)...", k, len(comp_T[k]))
        rec = fit_single(comp_T[k], draws=args.draws, tune=args.tune)
        rec["freq"] = freq_crosscheck(comp_T[k])
        rec["label"] = COMPOSITES[k]["label"]
        rec["family"] = COMPOSITES[k]["family"]
        rec["is_composite"] = True
        logger.info("  composite_%s kappa=%.3f [%.3f,%.3f] p+=%.3f",
                    k, rec["kappa_median"], rec["hdi_low"], rec["hdi_high"],
                    rec["p_positive"])
        per_pheno[f"composite_{k}"] = rec

    results = {
        "meta": {
            "draws": args.draws, "tune": args.tune, "chains": CHAINS,
            "phenotype": "absolute (z-scored over window, NOT era-demeaned)",
            "model": "z_next ~ a_p + u_{coach,pheno} + phi_p z_now + kappa_p Mloo",
            "kappa_meaning": "frequency-biased convergence on the league level; "
                             ">0 = conformist horizontal transmission",
            "multiplicity": "partial pooling of kappa across the 10 sub-traits "
                            "regularizes; reported by posterior HDI / P(kappa>0), "
                            "a separate family from the era/network exploratory tests",
            "pooled_over": subtraits,
        },
        "pooled": pooled,
        "pooled_detrended_robustness": pooled_detrended,
        "per_phenotype": per_pheno,
    }
    with open(OUT_DIR / "horizontal_transmission_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", OUT_DIR / "horizontal_transmission_results.json")

    # console summary
    print("\n" + "=" * 74)
    print("HORIZONTAL TRANSMISSION (conformist convergence, kappa)")
    print(f"  POOLED mu_kappa = {pooled['mu_kappa_median']:+.3f} "
          f"[{pooled['hdi_low']:+.3f},{pooled['hdi_high']:+.3f}]  "
          f"P(>0)={pooled['p_positive']:.3f}")
    if pooled_detrended:
        print(f"  detrended robustness mu_kappa = {pooled_detrended['mu_kappa_median']:+.3f} "
              f"[{pooled_detrended['hdi_low']:+.3f},{pooled_detrended['hdi_high']:+.3f}]  "
              f"P(>0)={pooled_detrended['p_positive']:.3f}")
    print("-" * 74)
    for k, r in sorted(per_pheno.items(), key=lambda kv: -kv[1]["kappa_median"]):
        star = "*" if (r["hdi_low"] > 0 or r["hdi_high"] < 0) else " "
        print(f" {star} {r['label']:22s} kappa={r['kappa_median']:+.3f} "
              f"[{r['hdi_low']:+.3f},{r['hdi_high']:+.3f}] P(>0)={r['p_positive']:.3f} "
              f"n={r['n_transitions']}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
