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
