#!/usr/bin/env python3
"""WS11: WAR-measurement-noise-aware robustness for gene -> WAR analyses.

Single-season coach WAR is a noisy estimate: Coach_WAR reports a global
single-season SE of ~2.48 wins against a ~1.94-win binomial sampling floor, so
roughly half of a single season's WAR is irreducible luck, and the noise is
heteroskedastic in games coached. Classical mean-zero noise in the DEPENDENT
variable does not bias the slope -- it ATTENUATES the observed correlation -- so
the reported r is a conservative floor, not an inflated number. This module adds
three honest robustness views WITHOUT manufacturing precision:

  1. inverse-variance weighting  -- precise full-season coach-years count more
     (weights from the transparent games-based proxy in data_paths.add_war_precision);
  2. partial-season sensitivity  -- drop sub-half seasons (the noisiest, also the
     ones Coach_WAR warns amplify negative WAR) and confirm the result holds;
  3. disattenuation bracket      -- r corrected for WAR's measurement noise,
     reported as an assumption-flagged upper anchor so the observed r reads as a
     floor. The reliability is estimated by variance decomposition (approximate),
     so this is a bracket, not a headline.

A fourth view lives here too, now SECONDARY:
  4. career-level correlation    -- collapse to one row per coach (career-mean
     gene vs precision-weighted career-mean WAR). A coach's mean WAR is reliable
     even when any single season is not, so this sidesteps season noise. BUT it
     carries a window-selection era artifact: within a fixed observation window,
     career-mean WAR correlates with the coach's career midpoint (~0.29 here,
     because still-employed recent coaches skew positive), even though annual WAR
     is replacement-relative and ~era-flat by construction. So the career-level
     correlation BRACKETS ABOVE the season-level estimate rather than refining it,
     and the season-level (coach-clustered, era-controlled) grain is the primary,
     era-clean answer. The career number is reported with this caveat.

ASCII only (Windows console).
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from utils.data_paths import (
    add_war_precision,
    load_coach_year_games,
    canonicalize_coach_name,
)
from utils.parsimony import (
    cluster_bootstrap_corr, cluster_bootstrap_corr_weighted, within_group_demean,
)

GAMES_PER_SEASON = 16          # matches Coach_WAR's *16 WAR scaling
PARTIAL_SEASON_GAMES = 8       # "half season" cutoff for the sensitivity drop

_GAMES_CACHE = {"df": None}


def _games_table() -> pd.DataFrame:
    if _GAMES_CACHE["df"] is None:
        _GAMES_CACHE["df"] = load_coach_year_games()
    return _GAMES_CACHE["df"]


def _ensure_precision(df: pd.DataFrame, coach_col: str = "coach",
                      year_col: str = "year") -> pd.DataFrame:
    """Attach real games-coached (reconstructed from raw PFR coaching results, the
    same W/L source Coach_WAR uses for win pct) and derive WAR precision. The WAR
    trajectory file has no games column, so without this the precision proxy and
    partial-season sensitivity have no input."""
    if "war_weight" in df.columns:
        return df
    out = df.copy()
    # Attach games only if not already present (avoid a double-merge collision when
    # the caller pre-attached games_coached).
    if "games_coached" not in out.columns:
        g = _games_table()
        if not g.empty and coach_col in out.columns and year_col in out.columns:
            out["_canon"] = out[coach_col].map(canonicalize_coach_name)
            out["_yr"] = pd.to_numeric(out[year_col], errors="coerce")
            gg = g.rename(columns={"year": "_yr"})
            out = out.merge(gg, left_on=["_canon", "_yr"], right_on=["coach_canon", "_yr"],
                            how="left").drop(columns=["_canon", "_yr", "coach_canon"],
                                             errors="ignore")
    return add_war_precision(out, games_col="games_coached")


def _war_test_retest_reliability(df: pd.DataFrame, war_col: str,
                                 coach_col: str, year_col: str) -> Dict:
    """Empirical reliability of a single-season WAR: the within-coach lag-1
    autocorrelation of annual WAR across CONSECUTIVE seasons. Under the same
    stable-quality assumption the gene-persistence analysis uses, this is a
    test-retest reliability -- a defensible, data-driven number (no binomial
    modeling) that quantifies how much of one season's WAR is signal vs luck.
    Reported for CONTEXT (it frames why the season-level r is attenuated); we do
    NOT disattenuate by it, since with a low/uncertain reliability that would
    manufacture precision."""
    if year_col not in df.columns:
        return {}
    d = df[[coach_col, year_col, war_col]].dropna().copy()
    d[year_col] = pd.to_numeric(d[year_col], errors="coerce")
    d = d.sort_values([coach_col, year_col])
    d["prev"] = d.groupby(coach_col)[war_col].shift(1)
    d["prev_yr"] = d.groupby(coach_col)[year_col].shift(1)
    pairs = d[(d[year_col] - d["prev_yr"]) == 1]
    if len(pairs) < 10 or pairs[war_col].std() == 0 or pairs["prev"].std() == 0:
        return {"war_test_retest_reliability": float("nan"), "n_pairs": int(len(pairs))}
    r = float(np.corrcoef(pairs[war_col], pairs["prev"])[0, 1])
    return {"war_test_retest_reliability": r, "n_pairs": int(len(pairs))}


def war_noise_robustness(
    merged: pd.DataFrame,
    gene_col: str,
    war_col: str = "annual_war",
    coach_col: str = "coach",
    year_col: str = "year",
    games_col: str = "games_coached",
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict:
    """Noise-aware robustness fields for one gene measure vs WAR.

    Returns a dict (to merge into the measure's results) with:
      r_ivw, ci/p_ivw, n_ivw            -- inverse-variance-weighted clustered r
                                           (down-weights noisy short seasons)
      r_partial_season, p_partial, n_partial, n_dropped_partial
                                           -- drop sub-half seasons (Coach_WAR
                                              warns these amplify negative WAR)
      war_test_retest_reliability, n_pairs -- empirical single-season WAR reliability
      note                              -- one-line interpretation guard
    The OBSERVED (unweighted, full-sample) r is intentionally NOT recomputed here;
    callers keep their existing primary estimate so headline numbers are unchanged.
    We deliberately do NOT report a disattenuated r: single-season WAR is mostly
    luck (reliability is low and uncertain), so dividing by sqrt(reliability) would
    manufacture precision. Instead, season-level WAR noise is handled by the
    inverse-variance-weighted estimate (r_ivw, and its era-clean variant
    r_ivw_eradj), which down-weights the noisy short seasons; the career-level
    correlation is a secondary cross-check that additionally carries a window-
    selection era artifact.
    """
    df = _ensure_precision(merged, coach_col=coach_col, year_col=year_col)
    cols = [gene_col, war_col, coach_col, "war_weight"]
    if games_col in df.columns:
        cols.append(games_col)
    if year_col in df.columns and year_col not in cols:
        cols.append(year_col)
    clean = df[cols].dropna(subset=[gene_col, war_col, coach_col, "war_weight"]).copy()
    out: Dict = {}
    if len(clean) < 10:
        return out

    x = clean[gene_col].to_numpy(float)
    y = clean[war_col].to_numpy(float) * GAMES_PER_SEASON
    coaches = clean[coach_col].to_numpy()
    w = clean["war_weight"].to_numpy(float)

    # 1. inverse-variance weighted (by games-coached precision), coach-clustered
    bw = cluster_bootstrap_corr_weighted(x, y, coaches, w=w, n_boot=n_boot, seed=seed)
    out["r_ivw"] = bw["r"]
    out["ci_low_ivw"] = bw["ci_low"]
    out["ci_high_ivw"] = bw["ci_high"]

    # 1b. era-adjusted IVW: the noise-aware AND era-clean season-level estimate.
    # Within-season demean gene and WAR (contemporary-group control), then
    # inverse-variance weight (down-weighting noisy short seasons), coach-clustered.
    # This is the primary season-level number when both WAR noise and league-wide
    # drift must be handled at once.
    if year_col in clean.columns:
        cc = clean.dropna(subset=[year_col]).copy()
        cc["_w16"] = cc[war_col].astype(float) * GAMES_PER_SEASON
        gdm = within_group_demean(cc, gene_col, year_col).to_numpy(float)
        wdm = within_group_demean(cc, "_w16", year_col).to_numpy(float)
        bwe = cluster_bootstrap_corr_weighted(
            gdm, wdm, cc[coach_col].to_numpy(), w=cc["war_weight"].to_numpy(float),
            n_boot=n_boot, seed=seed)
        out["r_ivw_eradj"] = bwe["r"]
        out["ci_low_ivw_eradj"] = bwe["ci_low"]
        out["ci_high_ivw_eradj"] = bwe["ci_high"]
        out["p_ivw_eradj_coach_clustered"] = bwe["p_bootstrap"]
        out["n_ivw_eradj"] = bwe["n"]
    out["p_ivw_coach_clustered"] = bw["p_bootstrap"]
    out["n_ivw"] = bw["n"]

    # 2. partial-season sensitivity (drop sub-half seasons)
    if games_col in clean.columns and clean[games_col].notna().any():
        full = clean[clean[games_col].astype(float) >= PARTIAL_SEASON_GAMES]
        n_dropped = int(clean[games_col].notna().sum() - len(full))
        if len(full) >= 10:
            bp = cluster_bootstrap_corr(
                full[gene_col].to_numpy(float),
                full[war_col].to_numpy(float) * GAMES_PER_SEASON,
                full[coach_col].to_numpy(), n_boot=n_boot, seed=seed)
            out["r_partial_season"] = bp["r"]
            out["p_partial_coach_clustered"] = bp["p_bootstrap"]
            out["n_partial"] = bp["n"]
            out["n_dropped_partial"] = n_dropped
        out["n_games_matched"] = int(clean[games_col].notna().sum())

    # 3. empirical single-season WAR reliability (context, not used to disattenuate)
    out.update(_war_test_retest_reliability(df, war_col, coach_col, year_col))
    out["note"] = ("season-level coach-clustered r is the era-clean PRIMARY grain "
                   "(annual WAR is ~era-flat, ~3% between-season variance, so a season "
                   "control barely moves it); r_ivw down-weights noisy short seasons. "
                   "The career-level correlation is a SECONDARY anchor that carries a "
                   "window-selection era artifact (career-mean WAR correlates with era), "
                   "so it brackets above rather than refining the season estimate.")
    return out


def career_level_corr(
    merged: pd.DataFrame,
    gene_col: str,
    war_col: str = "annual_war",
    coach_col: str = "coach",
    year_col: str = "year",
    min_seasons_variants: Sequence[int] = (1, 3),
) -> Dict:
    """Career-level robustness anchor: one row per coach, career-mean gene vs
    precision-weighted career-mean WAR (weights = games-based WAR precision).

    Because a coach's mean WAR is reliable even when any single season is not, this
    is the assumption-light answer to "single-season WAR is mostly noise". No
    clustering is needed (one row per coach). Reports the correlation for each
    minimum-seasons cutoff in min_seasons_variants (e.g. all coaches, and coaches
    with >=3 seasons where career WAR is most reliable).
    """
    df = _ensure_precision(merged, coach_col=coach_col, year_col=year_col)
    keep = [gene_col, war_col, coach_col, "war_weight"]
    if year_col in df.columns:
        keep.append(year_col)
    sub = df[keep].dropna(subset=[gene_col, war_col, coach_col, "war_weight"]).copy()
    if sub.empty:
        return {}
    sub["war_games"] = sub[war_col].astype(float) * GAMES_PER_SEASON

    recs = []
    for coach, g in sub.groupby(coach_col):
        w = g["war_weight"].to_numpy(float)
        war_career = (float(np.average(g["war_games"].to_numpy(float), weights=w))
                      if w.sum() > 0 else float(g["war_games"].mean()))
        mid_year = (float(pd.to_numeric(g[year_col], errors="coerce").mean())
                    if year_col in g.columns else float("nan"))
        recs.append((coach, float(g[gene_col].mean()), war_career, int(len(g)), mid_year))
    cdf = pd.DataFrame(recs, columns=["coach", "gene", "war", "n_seasons", "mid_year"])

    def _report(d: pd.DataFrame) -> Dict:
        if len(d) < 10 or d["gene"].std() == 0 or d["war"].std() == 0:
            return {"insufficient": True, "n_coaches": int(len(d))}
        r, p = stats.pearsonr(d["gene"], d["war"])
        # One row per coach (independent), so an ordinary nonparametric bootstrap
        # over coaches gives the 95% CI; no clustering needed at the career level.
        gene = d["gene"].to_numpy(float)
        war = d["war"].to_numpy(float)
        n = len(d)
        rng = np.random.default_rng(0)
        draws = []
        for _ in range(2000):
            idx = rng.integers(0, n, n)
            if np.std(gene[idx]) > 0 and np.std(war[idx]) > 0:
                draws.append(np.corrcoef(gene[idx], war[idx])[0, 1])
        draws = np.asarray(draws)
        return {"correlation": float(r), "p_value": float(p),
                "ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "n_coaches": int(len(d)), "significant": bool(p < 0.05)}

    out = {}
    # Window-selection era artifact: correlation of career-mean WAR with the coach's
    # career midpoint. Annual WAR is era-flat by construction, so a non-zero value
    # here is an observation-window selection effect (still-employed recent coaches
    # skew positive) that inflates the career-level gene->WAR correlation. Reported
    # so the career anchor is read as bracketing-above, not refining.
    valid_my = cdf.dropna(subset=["mid_year"])
    if len(valid_my) >= 10 and valid_my["war"].std() > 0 and valid_my["mid_year"].std() > 0:
        out["career_war_vs_era_r"] = float(
            np.corrcoef(valid_my["war"], valid_my["mid_year"])[0, 1])
    for m in min_seasons_variants:
        key = "all_coaches" if m <= 1 else f"min{m}_seasons"
        out[key] = _report(cdf[cdf["n_seasons"] >= m])
    return out
