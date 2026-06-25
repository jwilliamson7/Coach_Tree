#!/usr/bin/env python3
"""
Centralized resolution of external data inputs for Coach_Tree.

The coach WAR trajectories are produced and corrected in the sibling Coach_WAR
repo. Rather than copy the CSV into this repo (which goes stale), downstream
analyses reference the canonical Coach_WAR output through this resolver.

Resolution order for the WAR trajectories file:
  1. env var COACH_WAR_TRAJECTORIES, if set and the path exists;
  2. sibling repo  ../Coach_WAR/data/final/coach_war_trajectories.csv
     (both projects live side-by-side under Documents/Projects/);
  3. the legacy local copy data/processed/Coaching/coach_war_trajectories.csv,
     kept only as a safety net.

ASCII only (Windows console).
"""

import os
import re
import logging
from pathlib import Path

# repo root = parent of this utils/ directory
REPO_ROOT = Path(__file__).resolve().parent.parent

_logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Coach-name canonicalization for gene <-> WAR joins (WS8)
# --------------------------------------------------------------------------- #
# Verified same-person aliases ONLY (gene-side name -> WAR-side name, or vice
# versa, keyed by canonical form). We deliberately do NOT blanket-strip
# generational suffixes: 'Jim Mora' (Sr.) and 'Jim Mora' (Jr.) are DIFFERENT
# NFL head coaches, so a blanket strip would merge distinct people. Suffix
# mismatches are resolved here case-by-case after inspecting the attrition log.
_COACH_ALIASES = {
    # gene-side canonical -> WAR-side canonical, verified same person.
    # The gene data names the 2006 (ATL) and 2009 (SEA) coach "Jim Mora"; WAR
    # disambiguates Sr (1986-2001) from Jr (2004-2009). Those gene years are the
    # younger Mora, so route "jim mora" to "jim mora jr". Year-safe: Sr and Jr
    # have no overlapping seasons, and no "Jim Mora Sr" appears in the gene era.
    "jim mora": "jim mora jr",
}


def canonicalize_coach_name(name) -> str:
    """Canonical join key for a coach name: lowercase, strip punctuation, collapse
    whitespace. Does NOT strip Jr/Sr/II/III (those distinguish real people).

    'Mike McCarthy', 'mike mccarthy', 'Mike  McCarthy' -> 'mike mccarthy';
    "Sean McVay" / "Sean Mcvay" -> 'sean mcvay'. Returns '' for NaN/empty.
    """
    try:
        import pandas as pd
        if name is None or (isinstance(name, float) and pd.isna(name)):
            return ""
    except Exception:
        if name is None:
            return ""
    s = str(name).strip().lower()
    s = re.sub(r"[.\-']", " ", s)          # punctuation -> space
    s = re.sub(r"[^a-z0-9 ]", "", s)       # drop anything else
    s = re.sub(r"\s+", " ", s).strip()     # collapse whitespace
    return _COACH_ALIASES.get(s, s)


def add_coach_canon(df, name_col: str, out_col: str = "coach_canon"):
    """Return df with a canonical-name column added (for joining on coach name)."""
    df = df.copy()
    df[out_col] = df[name_col].map(canonicalize_coach_name)
    return df


def merge_gene_war(gene_df, war_df, gene_name_col, war_name_col,
                   year_cols=("year", "year"), how="inner", logger=None):
    """Merge a gene frame with WAR on canonicalized coach name + year, logging
    attrition (which gene-side coaches fail to match, and rows dropped).

    year_cols = (gene_year_col, war_year_col). Returns the merged frame. Adds a
    transient 'coach_canon' key on both sides and drops it from the result.
    """
    log = logger or _logger
    g = add_coach_canon(gene_df, gene_name_col)
    w = add_coach_canon(war_df, war_name_col)
    gy, wy = year_cols
    if wy != gy:
        w = w.rename(columns={wy: gy})
    merged = g.merge(w, on=["coach_canon", gy], how=how, suffixes=("", "_war"))

    gene_names = set(g["coach_canon"].unique())
    war_names = set(w["coach_canon"].unique())
    unmatched = sorted(n for n in (gene_names - war_names) if n)
    matched_rows = merged["coach_canon"].notna().sum() if how != "inner" else len(merged)
    log.info("gene<->WAR merge: %d gene coach-years, %d matched; %d gene coaches "
             "unmatched in WAR", len(gene_df), matched_rows, len(unmatched))
    if unmatched:
        log.info("  unmatched gene coaches (no WAR row): %s",
                 ", ".join(unmatched[:40]) + (" ..." if len(unmatched) > 40 else ""))
    return merged.drop(columns=["coach_canon"])

# --------------------------------------------------------------------------- #
# Per-coach-year WAR precision (WS11)
# --------------------------------------------------------------------------- #
# Coach_WAR persists only ONE global single-season WAR standard error: the test
# RMSE (0.15521 in win-fraction units) scaled to a 16-game season = 2.4833 wins,
# which sits just above the ~1.94-win binomial sampling floor (so a single-season
# WAR is roughly half irreducible luck). Coach_WAR does NOT publish a per-row SE.
# But the dominant noise source is season length: a season win fraction has
# binomial sampling variance ~ p(1-p)/games, i.e. variance scales ~ 1/games. So
# we derive a transparent per-row precision from games coached rather than invent
# a model-based SE. full_games=16 matches Coach_WAR's *16 WAR scaling convention.
WAR_SE_FULL_SEASON = 2.4833   # wins; Coach_WAR global single-season WAR SE


def load_coach_year_games(coaches_dir=None):
    """Build a [coach_canon, year, games_coached] table from the raw PFR coaching
    results (data/raw/Coaches/<name>/all_coaching_results.csv), which carry a real
    per-season regular-season game count `G`.

    The WAR trajectory file does NOT contain games coached (its 'annual_games'
    column is actually WAR expressed in games, = annual_war*16), so games -- needed
    for WAR-precision weighting and the partial-season sensitivity -- are
    reconstructed here. NFL rows only; G summed within (coach, year) to handle
    mid-season team changes. Coach key is canonicalized to match gene/WAR joins.
    Returns an empty frame if the directory is absent.
    """
    import pandas as pd
    base = Path(coaches_dir) if coaches_dir else (REPO_ROOT / "data" / "raw" / "Coaches")
    if not base.exists():
        _logger.warning("load_coach_year_games: %s absent; games unavailable", base)
        return pd.DataFrame(columns=["coach_canon", "year", "games_coached"])
    rows = []
    for d in base.iterdir():
        f = d / "all_coaching_results.csv"
        if not f.exists():
            continue
        try:
            r = pd.read_csv(f, usecols=lambda c: c in ("Year", "Lg", "G"))
        except Exception:
            continue
        if "Year" not in r.columns or "G" not in r.columns:
            continue
        if "Lg" in r.columns:
            r = r[r["Lg"] == "NFL"]
        r = r.assign(coach_canon=canonicalize_coach_name(d.name))
        r["year"] = pd.to_numeric(r["Year"], errors="coerce")
        r["games_coached"] = pd.to_numeric(r["G"], errors="coerce")
        rows.append(r[["coach_canon", "year", "games_coached"]].dropna())
    if not rows:
        return pd.DataFrame(columns=["coach_canon", "year", "games_coached"])
    out = pd.concat(rows, ignore_index=True)
    out["year"] = out["year"].astype(int)
    return out.groupby(["coach_canon", "year"], as_index=False)["games_coached"].sum()


def add_war_precision(df, games_col="games_coached",
                      war_se_full: float = WAR_SE_FULL_SEASON,
                      full_games: float = 16.0):
    """Add per-row WAR precision columns from a games-based sampling-noise proxy.

    WAR is a single-season win-fraction residual; its sampling variance scales
    ~ 1/games_coached, so

        war_var_row = war_se_full^2 * (full_games / games),
        war_se_row  = sqrt(war_var_row),
        war_weight  = 1 / war_var_row   (proportional to games coached).

    This is a heteroskedasticity model, NOT a claim of a per-row model SE: it
    down-weights short/partial seasons (the noisiest WAR estimates, which
    Coach_WAR warns also amplify negative values) without manufacturing precision
    Coach_WAR does not provide. Rows with missing or non-positive games fall back
    to a full-season SE. Returns a copy with war_se_row, war_var_row, war_weight.
    """
    import numpy as np
    import pandas as pd
    df = df.copy()
    if games_col in df.columns:
        g = pd.to_numeric(df[games_col], errors="coerce")
        g = g.where(g > 0, full_games)          # missing / 0 -> treat as full season
    else:
        _logger.warning("add_war_precision: '%s' absent; using full-season SE for all rows",
                        games_col)
        g = pd.Series(full_games, index=df.index)
    df["war_se_row"] = war_se_full * np.sqrt(full_games / g)
    df["war_var_row"] = df["war_se_row"] ** 2
    df["war_weight"] = 1.0 / df["war_var_row"]
    return df


_ENV_VAR = "COACH_WAR_TRAJECTORIES"
_SIBLING = REPO_ROOT.parent / "Coach_WAR" / "data" / "final" / "coach_war_trajectories.csv"
_LOCAL_FALLBACK = REPO_ROOT / "data" / "processed" / "Coaching" / "coach_war_trajectories.csv"


def coach_war_trajectories_path(must_exist: bool = True) -> Path:
    """Return the path to the canonical coach WAR trajectories CSV.

    See module docstring for resolution order. With must_exist=True (default) a
    FileNotFoundError is raised if none of the candidates resolve, with the list
    of locations tried.
    """
    candidates = []

    env = os.environ.get(_ENV_VAR)
    if env:
        candidates.append(Path(env))
    candidates.append(_SIBLING)
    candidates.append(_LOCAL_FALLBACK)

    for cand in candidates:
        if cand.exists():
            return cand

    if must_exist:
        tried = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "coach_war_trajectories.csv not found. Tried (in order):\n  "
            + tried
            + f"\nSet {_ENV_VAR} to override, or ensure the Coach_WAR repo is a "
            "sibling of Coach_Tree."
        )
    # must_exist=False: return the preferred (sibling) location even if absent
    return _SIBLING


if __name__ == "__main__":
    p = coach_war_trajectories_path(must_exist=False)
    print(f"Resolved coach WAR trajectories -> {p}")
    print(f"Exists: {p.exists()}")
