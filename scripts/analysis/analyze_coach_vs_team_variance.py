#!/usr/bin/env python3
"""Coach-vs-team variance decomposition: is the gene a COACH trait or a team/roster
artifact? This is the core validation of the "coaching gene" premise.

Three complementary views per gene:
  1. Variance explained by coach identity vs team identity vs season (one-way eta^2
     -- descriptive; coach and team are confounded since a coach mostly maps to one
     team, so this brackets but does not identify).
  2. Cross-team travel: for coaches who changed teams, correlation of their mean
     gene at the OLD team vs the NEW team. If the gene travels with the coach, this
     is high (and comparable to / above the within-team year-to-year persistence).
     Mentor-... no: clustered? one value per mover, so a plain correlation + a
     coach-resampling bootstrap CI.
  3. Mover regression (the identification): regress the coach's mean gene at the new
     team on (a) their mean gene at the old team [coach effect] and (b) the new
     team's mean gene in the seasons BEFORE they arrived, under the prior regime
     [team effect]. A large coach coefficient with a small team coefficient means
     the gene is attached to the coach, not the franchise.

Writes outputs/analysis/coach_vs_team_variance_results.json. ASCII only.
"""

import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import canonicalize_coach_name
from utils.parsimony import cluster_robust_ols, cluster_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TEAM_MAP = Path("data/processed/Coaching/team_year_head_coaches.csv")

# (label, csv, gene_col, coach_col, year_col)
GENES = [
    ("composite_aggression", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "composite_aggression", "head_coach", "season"),
    ("fourth_down", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "fourth_down_aggression", "head_coach", "season"),
    ("pass_heavy", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "pass_heavy_aggression", "head_coach", "season"),
    ("deep_pass", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "deep_pass_aggression", "head_coach", "season"),
    ("two_point", "data/processed/coaching_genes/aggression_gene_by_year.csv",
     "two_point_aggression", "head_coach", "season"),
    ("shotgun", "data/processed/coaching_genes/shotgun_gene.csv",
     "shotgun_gene_zscore", "head_coach", "season"),
    ("tempo", "data/processed/coaching_genes/tempo_gene.csv",
     "composite_tempo_zscore", "head_coach", "season"),
]


def _eta2(df, value, group):
    """Share of variance of `value` explained by `group` identity (one-way eta^2)."""
    d = df[[value, group]].dropna()
    if d[group].nunique() < 2:
        return float("nan")
    grand = d[value].mean()
    ss_tot = float(((d[value] - grand) ** 2).sum())
    ss_bet = float(d.groupby(group)[value].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum())
    return ss_bet / ss_tot if ss_tot > 0 else float("nan")


def _attach_team(g, coach_col, year_col):
    tm = pd.read_csv(TEAM_MAP)
    tm["canon"] = tm["Primary_Coach"].map(canonicalize_coach_name)
    g = g.copy()
    g["canon"] = g[coach_col].map(canonicalize_coach_name)
    g = g.merge(tm[["canon", "Year", "Team"]].rename(columns={"Year": year_col}),
                on=["canon", year_col], how="left")
    return g


def _bootstrap_corr_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    if n < 5 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.corrcoef(a, b)[0, 1])
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(a[idx]) > 0 and np.std(b[idx]) > 0:
            draws.append(np.corrcoef(a[idx], b[idx])[0, 1])
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyze_gene(label, csv, gene_col, coach_col, year_col):
    g = pd.read_csv(csv)
    if gene_col not in g.columns:
        return None
    g = _attach_team(g, coach_col, year_col)
    g = g.dropna(subset=[gene_col])
    out = {
        "n_coach_years": int(len(g)),
        "n_coaches": int(g["canon"].nunique()),
        "n_teams": int(g["Team"].nunique()),
        "team_match_rate": float(g["Team"].notna().mean()),
    }

    # 1. variance explained by coach vs team vs season (descriptive one-way eta^2;
    #    coach and team are confounded so these bracket rather than identify -- the
    #    inferential CI lives on the cross-team travel r and the mover regression)
    out["var_explained_coach"] = _eta2(g, gene_col, "canon")
    out["var_explained_team"] = _eta2(g, gene_col, "Team")
    out["var_explained_season"] = _eta2(g, gene_col, year_col)

    # 2 & 3. movers
    gt = g.dropna(subset=["Team"])
    per_team = gt.groupby("canon")["Team"].nunique()
    movers = per_team[per_team > 1].index
    out["n_movers"] = int(len(movers))

    old_means, new_means, team_priors = [], [], []
    for c in movers:
        sub = gt[gt["canon"] == c].sort_values(year_col)
        # first team vs the team they move to (second distinct team in time order)
        teams_in_order = list(dict.fromkeys(sub["Team"].tolist()))
        if len(teams_in_order) < 2:
            continue
        t_old, t_new = teams_in_order[0], teams_in_order[1]
        new_first_year = sub[sub["Team"] == t_new][year_col].min()
        old_mean = sub[sub["Team"] == t_old][gene_col].mean()
        new_mean = sub[sub["Team"] == t_new][gene_col].mean()
        # team_new's gene BEFORE this coach arrived (any prior coach in the panel)
        prior = gt[(gt["Team"] == t_new) & (gt[year_col] < new_first_year)][gene_col]
        old_means.append(old_mean)
        new_means.append(new_mean)
        team_priors.append(prior.mean() if len(prior) else np.nan)

    old_means = np.array(old_means); new_means = np.array(new_means)
    team_priors = np.array(team_priors)

    r, lo, hi = _bootstrap_corr_ci(old_means, new_means)
    out["cross_team_travel"] = {
        "r_old_vs_new_team": r, "ci_low": lo, "ci_high": hi, "n": int(len(old_means)),
        "note": "coach's mean gene at old team vs new team; high = gene travels with coach",
    }

    # mover regression: new ~ coach_pre + team_new_prior
    mask = np.isfinite(old_means) & np.isfinite(new_means) & np.isfinite(team_priors)
    if mask.sum() >= 8 and np.std(old_means[mask]) > 0 and np.std(team_priors[mask]) > 0:
        X = np.column_stack([old_means[mask], team_priors[mask]])
        y = new_means[mask]
        # clusters = each mover is its own cluster (independent moves) -> use index
        res = cluster_robust_ols(X, y, np.arange(mask.sum()),
                                 ["coach_pre", "team_new_prior"])
        # Bootstrap 95% CIs over movers (each move is an independent observation).
        mboot = cluster_bootstrap_ci(X, y, np.arange(mask.sum()),
                                     ["coach_pre", "team_new_prior"], n_boot=2000, seed=0)
        coach_pre = dict(res["coefficients"]["coach_pre"])
        coach_pre["ci_low"] = mboot["coach_pre"]["ci_low"]
        coach_pre["ci_high"] = mboot["coach_pre"]["ci_high"]
        team_prior = dict(res["coefficients"]["team_new_prior"])
        team_prior["ci_low"] = mboot["team_new_prior"]["ci_low"]
        team_prior["ci_high"] = mboot["team_new_prior"]["ci_high"]
        out["mover_regression"] = {
            "n": int(mask.sum()),
            "coach_pre": coach_pre,
            "team_new_prior": team_prior,
            "r_squared": res["r_squared"],
            "note": ("new-team gene ~ coach's old-team gene + new-team's pre-arrival "
                     "gene; large coach_pre vs small team_new_prior = coach trait"),
        }
    else:
        out["mover_regression"] = {"insufficient": True, "n": int(mask.sum())}
    return out


def main():
    results = {}
    for label, csv, gene_col, coach_col, year_col in GENES:
        if not Path(csv).exists():
            logger.warning("missing %s", csv)
            continue
        r = analyze_gene(label, csv, gene_col, coach_col, year_col)
        if r:
            results[label] = r
            mv = r.get("cross_team_travel", {})
            mr = r.get("mover_regression", {})
            logger.info("%-20s var(coach)=%.2f var(team)=%.2f | travel r=%.3f (n=%d) | "
                        "mover coach_pre b=%s",
                        label, r["var_explained_coach"], r["var_explained_team"],
                        mv.get("r_old_vs_new_team", float("nan")), mv.get("n", 0),
                        (f"{mr['coach_pre']['coefficient']:.2f}(p={mr['coach_pre']['p_value']:.3f})"
                         if "coach_pre" in mr else "n/a"))

    out_dir = Path("outputs/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "coach_vs_team_variance_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: outputs/analysis/coach_vs_team_variance_results.json")


if __name__ == "__main__":
    main()
