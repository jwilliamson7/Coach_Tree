#!/usr/bin/env python3
"""EHS confirmatory trait scaffolding (frozen protocol osf.io/y2kr5).

Single source of truth for the ten frozen sub-traits (and the four gene-level
composites used as H1/H2/map exemplars) and for turning the committed gene CSVs
into the two data structures the confirmatory analyses consume:

  1. a per-trait coach-season PANEL  -- head-coach-keyed, carrying the
     contemporary-group-adjusted (within-season demeaned) standardized phenotype
     and its known per-observation sampling variance on the same standardized
     scale. This feeds C2 repeatability and, aggregated, the mentor/protege
     values for C1.
  2. per-trait mentor->protege PAIRS -- the protege's own head-coaching phenotype
     (outcome) paired with the phenotype of the head coach they apprenticed under,
     measured on that mentor's team over their overlapping coordinator seasons
     (predictor), each carrying its sampling variance for the errors-in-variables
     level of C1.

Standardization and sampling variance. Each sub-trait's raw coach-season gene has
a known sampling-variance numerator in the CSV (`*_phat_var_sum` for the rate/
probability genes, `*_resid_var_sum` for the regression genes); the per-season
sampling variance is that numerator divided by the squared play count. We z-score
the raw gene over its data window and rescale the sampling variance by 1/sd^2 so
that phenotype and measurement error live on one standardized scale, then subtract
the season mean (the single era control). Demeaning shifts by a season constant
and leaves the sampling variance unchanged.

Composites are rebuilt from their demeaned standardized sub-traits (mean of the
available components) so the composite phenotype and its propagated sampling
variance are internally consistent with the parts; this is the "reliability /
variance is not cleanly estimable per composite -> propagate from components"
branch of Section 3.

Team/coach attribution and transition building are reused verbatim from
InheritanceAnalyzer so the mentor-of-team lookups and franchise mappings stay in
one tested place. ASCII only (Windows console).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.parsimony import within_group_demean
from utils.data_paths import canonicalize_coach_name
from crawlers.utils.data_constants import pfr_to_pbp

GENES_DIR = Path("data/processed/coaching_genes")

# --------------------------------------------------------------------------- #
# Frozen trait definitions (Section 2 of the preregistration)
# --------------------------------------------------------------------------- #
# side: 'off' -> OC->HC transition, offensive/HC-keyed genes
#       'def' -> DC->HC transition, defensive genes (team-keyed mentor side)
# var_kind: 'binom' (phat_var_sum) or 'resid' (resid_var_sum); both are the
#           sampling-variance numerator, divided by plays^2 for the per-season var.
SUBTRAITS = {
    "fourth_down": dict(gene="aggression", csv="aggression_gene_by_year.csv",
                        raw="fourth_down_aggression", var="fourth_down_phat_var_sum",
                        plays="fourth_down_plays", rel="fourth_down_aggression_reliability",
                        side="off", window=(2006, 2024), label="Fourth down"),
    "pass_heavy": dict(gene="aggression", csv="aggression_gene_by_year.csv",
                       raw="pass_heavy_aggression", var="pass_heavy_phat_var_sum",
                       plays="run_pass_plays", rel="pass_heavy_aggression_reliability",
                       side="off", window=(2006, 2024), label="Pass-heavy"),
    "deep_pass": dict(gene="aggression", csv="aggression_gene_by_year.csv",
                      raw="deep_pass_aggression", var="deep_pass_phat_var_sum",
                      plays="pass_plays", rel="deep_pass_aggression_reliability",
                      side="off", window=(2006, 2024), label="Deep pass"),
    "two_point": dict(gene="aggression", csv="aggression_gene_by_year.csv",
                      raw="two_point_aggression", var="two_point_phat_var_sum",
                      plays="conversion_attempts", rel="two_point_aggression_reliability",
                      side="off", window=(2006, 2024), label="Two-point"),
    "no_huddle": dict(gene="tempo", csv="tempo_gene.csv",
                      raw="no_huddle_gene", var="no_huddle_phat_var_sum",
                      plays="no_huddle_plays", rel="no_huddle_gene_reliability",
                      side="off", window=(2006, 2024), label="No-huddle"),
    "pace": dict(gene="tempo", csv="tempo_gene.csv",
                 raw="pace_gene", var="pace_resid_var_sum",
                 plays="pace_plays", rel="pace_gene_reliability",
                 side="off", window=(2006, 2024), label="Pace"),
    "box_stacking": dict(gene="defensive", csv="defensive_scheme_gene.csv",
                         raw="box_stacking_gene", var="box_resid_var_sum",
                         plays="box_plays", rel="box_stacking_gene_reliability",
                         side="def", window=(2016, 2024), label="Box stacking"),
    "pass_rush": dict(gene="defensive", csv="defensive_scheme_gene.csv",
                      raw="pass_rush_gene", var="rush_resid_var_sum",
                      plays="rush_plays", rel="pass_rush_gene_reliability",
                      side="def", window=(2016, 2024), label="Pass rush"),
    "man_coverage": dict(gene="defensive", csv="defensive_scheme_gene.csv",
                         raw="man_coverage_gene", var="man_phat_var_sum",
                         plays="man_plays", rel="man_coverage_gene_reliability",
                         side="def", window=(2018, 2024), label="Man coverage"),
    "shotgun": dict(gene="shotgun", csv="shotgun_gene.csv",
                    raw="shotgun_gene", var="shotgun_phat_var_sum",
                    plays="total_plays", rel="shotgun_gene_reliability",
                    side="off", window=(2006, 2024), label="Shotgun"),
}

# Gene-level composites (H1 families, H2, and map exemplars); NOT counted as
# independent points in the C3 cross-trait test.
COMPOSITES = {
    "aggression": dict(components=["fourth_down", "pass_heavy", "deep_pass", "two_point"],
                       family="approach", side="off", label="Offensive aggression"),
    "tempo": dict(components=["no_huddle", "pace"],
                  family="identity", side="off", label="Tempo"),
    "defensive": dict(components=["box_stacking", "pass_rush", "man_coverage"],
                      family="approach", side="def", label="Defensive aggression"),
    "shotgun": dict(components=["shotgun"],
                    family="identity", side="off", label="Shotgun"),
}


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _load_gene_csv(name):
    return pd.read_csv(GENES_DIR / name)


def build_subtrait_panel(key):
    """Head-coach-keyed coach-season panel for one sub-trait.

    Columns: coach (canonical), head_coach (raw), season, defteam (def only),
    z (demeaned standardized phenotype), samp_var (sampling variance on the z
    scale), plays, reliability.
    """
    spec = SUBTRAITS[key]
    df = _load_gene_csv(spec["csv"]).copy()
    y0, y1 = spec["window"]
    df = df[(df["season"] >= y0) & (df["season"] <= y1)]

    raw, var, plays = spec["raw"], spec["var"], spec["plays"]
    df = df.dropna(subset=[raw, plays])
    df = df[df[plays].astype(float) > 0]
    if df.empty:
        return df

    sd = float(df[raw].std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    mean = float(df[raw].mean())
    df["z_abs"] = (df[raw].astype(float) - mean) / sd
    # per-season sampling variance of the raw gene, rescaled to the z scale
    df["samp_var"] = (df[var].astype(float) / (df[plays].astype(float) ** 2)) / (sd ** 2)
    df["plays"] = df[plays].astype(float)
    df["reliability"] = df[spec["rel"]].astype(float) if spec["rel"] in df else np.nan

    # contemporary-group adjustment: subtract the season mean (single era control)
    df["z"] = within_group_demean(df, "z_abs", "season")

    df["coach"] = df["head_coach"].map(canonicalize_coach_name)
    keep = ["coach", "head_coach", "season", "z", "samp_var", "plays", "reliability"]
    if spec["side"] == "def":
        keep.append("defteam")
    return df[keep].reset_index(drop=True)


def build_all_subtrait_panels():
    return {k: build_subtrait_panel(k) for k in SUBTRAITS}


def build_composite_panel(key, subpanels):
    """Composite coach-season panel rebuilt from its demeaned standardized
    sub-trait panels: composite z = mean of available component z; composite
    sampling variance = (1/m^2) * sum of component sampling variances (equal-
    weight noise propagation). Averaged over the coach-season grain."""
    comp = COMPOSITES[key]
    frames = []
    for c in comp["components"]:
        p = subpanels[c][["coach", "head_coach", "season", "z", "samp_var"]].copy()
        p = p.rename(columns={"z": f"z_{c}", "samp_var": f"v_{c}"})
        frames.append(p)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["coach", "head_coach", "season"], how="outer")
    zcols = [f"z_{c}" for c in comp["components"]]
    vcols = [f"v_{c}" for c in comp["components"]]
    out["z"] = out[zcols].mean(axis=1, skipna=True)
    m = out[zcols].notna().sum(axis=1).replace(0, np.nan)
    out["samp_var"] = out[vcols].sum(axis=1, min_count=1) / (m ** 2)
    out = out.dropna(subset=["z"])
    return out[["coach", "head_coach", "season", "z", "samp_var"]].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Transitions -> mentor/protege pairs
# --------------------------------------------------------------------------- #
def get_transitions(min_years=1):
    """Reuse InheritanceAnalyzer's tested transition builder (one arc per coach
    per coordinator role: most recent coordinator stint preceding a head-coaching
    stint). Returns the raw transition dicts plus a loaded analyzer for the
    mentor-of-team lookups."""
    from scripts.analysis.analyze_gene_inheritance import InheritanceAnalyzer
    az = InheritanceAnalyzer(min_years=min_years)
    az.load_data()
    return az.build_transitions(), az


def _weighted(values, weights):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    if weights.sum() <= 0:
        return float(values.mean())
    return float(np.average(values, weights=weights))


def _panel_lookup_off(panel, coach_name, season):
    """One HC-keyed row (coach's own gene) for an offensive sub-trait."""
    key = canonicalize_coach_name(coach_name)
    m = panel[(panel["coach"] == key) & (panel["season"] == season)]
    if len(m) == 0:
        return None
    if len(m) == 1:
        return float(m.iloc[0]["z"]), float(m.iloc[0]["samp_var"])
    w = m["plays"].to_numpy(float)
    return _weighted(m["z"], w), float(np.sum((w ** 2) * m["samp_var"]) / (w.sum() ** 2))


def _panel_lookup_def_team(panel, team_pfr, season):
    """Team whole-season defensive value (mentor side): play-weighted over any
    mid-season head-coach split rows."""
    pbp = pfr_to_pbp(team_pfr, season)
    m = panel[(panel["defteam"] == pbp) & (panel["season"] == season)]
    if len(m) == 0:
        return None
    if len(m) == 1:
        return float(m.iloc[0]["z"]), float(m.iloc[0]["samp_var"])
    w = m["plays"].to_numpy(float)
    return _weighted(m["z"], w), float(np.sum((w ** 2) * m["samp_var"]) / (w.sum() ** 2))


def build_pairs(key, panel, transitions, analyzer):
    """Mentor->protege pairs for one sub-trait.

    Returns DataFrame: coach, mentor_id, y (protege HC-era mean z), y_se,
    x (mentor mean z over the overlapping coordinator seasons), x_se, n_hc,
    n_coord. Mentor id is the mentor coach for the offensive side (the head coach
    of the team during the overlap) or the team-season group for defense; used
    only for the varying-intercept non-independence check.
    """
    spec = SUBTRAITS[key]
    side = spec["side"]
    ttype = "OC->HC" if side == "off" else "DC->HC"
    rows = []
    for t in transitions:
        if t["transition_type"] != ttype:
            continue
        # protege HC-era value (own name)
        yv, yvar = [], []
        for yr in t["hc_years"]:
            r = _panel_lookup_off(panel, t["coach_name"], yr)
            if r is not None:
                yv.append(r[0]); yvar.append(r[1])
        if len(yv) < 1:
            continue
        # mentor value over overlap
        xv, xvar, mentor_id = [], [], None
        for yr in t["coord_years"]:
            if side == "off":
                hc = analyzer._get_hc_name(t["coord_team"], yr)
                if not hc:
                    continue
                r = _panel_lookup_off(panel, hc, yr)
                if r is not None:
                    xv.append(r[0]); xvar.append(r[1])
                    if mentor_id is None:
                        mentor_id = canonicalize_coach_name(hc)
            else:
                r = _panel_lookup_def_team(panel, t["coord_team"], yr)
                if r is not None:
                    xv.append(r[0]); xvar.append(r[1])
        if len(xv) < 1:
            continue
        if mentor_id is None:  # defensive: mentor identity is the team-season block
            mentor_id = f"{t['coord_team']}"
        ny, nx = len(yv), len(xv)
        rows.append({
            "coach": canonicalize_coach_name(t["coach_name"]),
            "coach_name": t["coach_name"],
            "mentor_id": mentor_id,
            "y": float(np.mean(yv)),
            "y_se": float(np.sqrt(np.sum(yvar) / (ny ** 2))),
            "x": float(np.mean(xv)),
            "x_se": float(np.sqrt(np.sum(xvar) / (nx ** 2))),
            "n_hc": ny,
            "n_coord": nx,
        })
    return pd.DataFrame(rows)


def build_subtrait_obs(key, panel, transitions, analyzer):
    """Per-season observations for the unified season-grain C1 model.

    One row per (protege, side, season): side 'M' = a mentor-team overlap season
    (predictor), side 'P' = a protege head-coaching season (outcome). Each row
    carries the standardized phenotype `val` and its measurement standard error
    `se`. Proteges are kept only if they have at least one M and one P season.
    Keeping individual seasons (rather than pre-averaging) lets the model share a
    true within-coach season variance, so h^2 is estimated at the coach-season
    grain and is properly bounded by repeatability.
    """
    spec = SUBTRAITS[key]
    side = spec["side"]
    ttype = "OC->HC" if side == "off" else "DC->HC"
    rows, pidx = [], 0
    for t in transitions:
        if t["transition_type"] != ttype:
            continue
        prot = []
        for yr in t["hc_years"]:
            r = _panel_lookup_off(panel, t["coach_name"], yr)
            if r is not None:
                prot.append((yr, r[0], np.sqrt(r[1])))
        ment, mentor_id = [], None
        for yr in t["coord_years"]:
            if side == "off":
                hc = analyzer._get_hc_name(t["coord_team"], yr)
                if not hc:
                    continue
                r = _panel_lookup_off(panel, hc, yr)
                if r is not None:
                    ment.append((yr, r[0], np.sqrt(r[1])))
                    if mentor_id is None:
                        mentor_id = canonicalize_coach_name(hc)
            else:
                r = _panel_lookup_def_team(panel, t["coord_team"], yr)
                if r is not None:
                    ment.append((yr, r[0], np.sqrt(r[1])))
        if not prot or not ment:
            continue
        if mentor_id is None:
            mentor_id = str(t["coord_team"])
        for yr, v, se in ment:
            rows.append(dict(pidx=pidx, coach=canonicalize_coach_name(t["coach_name"]),
                             coach_name=t["coach_name"], mentor_id=mentor_id,
                             side="M", season=yr, val=v, se=se))
        for yr, v, se in prot:
            rows.append(dict(pidx=pidx, coach=canonicalize_coach_name(t["coach_name"]),
                             coach_name=t["coach_name"], mentor_id=mentor_id,
                             side="P", season=yr, val=v, se=se))
        pidx += 1
    return pd.DataFrame(rows)


def build_composite_obs(key, subpanels, transitions, analyzer):
    """Per-season composite observations: average the available component per-
    season observations on the shared (protege, side, season) grain, propagating
    the measurement variance as (1/m^2) sum."""
    comp = COMPOSITES[key]
    keys = ["coach", "coach_name", "mentor_id", "side", "season"]
    frames = []
    for c in comp["components"]:
        o = build_subtrait_obs(c, subpanels[c], transitions, analyzer)
        if o.empty:
            continue
        o = o[keys + ["val", "se"]].rename(columns={"val": f"val_{c}", "se": f"se_{c}"})
        frames.append(o)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=keys, how="outer")
    vcols = [f"val_{c}" for c in comp["components"] if f"val_{c}" in out]
    scols = [f"se_{c}" for c in comp["components"] if f"se_{c}" in out]
    out["val"] = out[vcols].mean(axis=1, skipna=True)
    m = out[vcols].notna().sum(axis=1).replace(0, np.nan)
    out["se"] = np.sqrt((out[scols] ** 2).sum(axis=1, min_count=1)) / m
    out = out.dropna(subset=["val"])
    # renumber proteges 0..N-1 on the coach identity
    codes = {c: i for i, c in enumerate(sorted(out["coach"].unique()))}
    out["pidx"] = out["coach"].map(codes)
    return out[["pidx", "coach", "coach_name", "mentor_id", "side", "season", "val", "se"]].reset_index(drop=True)


def build_composite_pairs(key, subpanels, transitions, analyzer):
    """Mentor->protege pairs for a gene-level composite, built by averaging the
    per-component pairs on the shared coach set (so x/y are the component-mean
    phenotype and the sampling variances propagate as (1/m^2) sum)."""
    comp = COMPOSITES[key]
    per = {}
    for c in comp["components"]:
        pr = build_pairs(c, subpanels[c], transitions, analyzer)
        if not pr.empty:
            per[c] = pr.set_index("coach")
    if not per:
        return pd.DataFrame()
    # Union of coaches: average whatever components a coach has (a 2016-2017
    # defensive protege with box+rush but no man still contributes a 2-component
    # composite, matching the published composite's window handling).
    coaches = set().union(*[set(v.index) for v in per.values()])
    rows = []
    for coach in sorted(coaches):
        ys, yvs, xs, xvs, mid, nm = [], [], [], [], None, None
        for c in comp["components"]:
            if coach not in per[c].index:
                continue
            r = per[c].loc[coach]
            ys.append(r["y"]); yvs.append(r["y_se"] ** 2)
            xs.append(r["x"]); xvs.append(r["x_se"] ** 2)
            mid = r["mentor_id"]; nm = r["coach_name"]
        if not ys:
            continue
        m = len(ys)
        rows.append({
            "coach": coach, "coach_name": nm, "mentor_id": mid,
            "y": float(np.mean(ys)), "y_se": float(np.sqrt(np.sum(yvs) / (m ** 2))),
            "x": float(np.mean(xs)), "x_se": float(np.sqrt(np.sum(xvs) / (m ** 2))),
        })
    return pd.DataFrame(rows)
