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
from pathlib import Path

# repo root = parent of this utils/ directory
REPO_ROOT = Path(__file__).resolve().parent.parent

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
