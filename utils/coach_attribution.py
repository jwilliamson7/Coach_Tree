#!/usr/bin/env python3
"""
Authoritative game-level head-coach attribution.

Single source of truth for "which head coach led team T in game G", shared by
every gene calculator so offense and defense attribute plays to the SAME coach on
the SAME per-game basis.

Why this exists
---------------
The raw play-by-play `home_coach`/`away_coach` fields are correct for the vast
majority of team-seasons, but they have two failure modes that corrupted gene
attribution:

  Bug A - a mid-season head-coaching change is captured per game by PBP, so the
          offensive genes (grouped by per-play coach) split it correctly, but the
          defensive gene grouped the whole (defteam, season) and tagged it with
          the plurality coach, blending two coaches' defenses into one row.
  Bug B - for some team-seasons PBP lists ONLY the starter for the whole year and
          silently misses the change entirely (e.g. MIA 2015 credits Philbin all
          16 games when Campbell coached 12), so BOTH offense and defense credit
          the fired coach for the successor's games.

Resolution
----------
Build a per-(game_id, team) coach map:

  * Default = the PBP per-game coach. This is already correct for single-coach
    seasons and for Bug A seasons (PBP captured the change), so those genes are
    unchanged.
  * Override = for team-seasons where the coach RECORDS (data/raw/Coaches/*/
    all_coaching_results.csv, which carry each coach's partial-season game count G
    and are the source PFR's team_year_head_coaches.csv is built from) show MORE
    distinct coaches than PBP, rebuild the split: the starter is the coach of the
    team's earliest-week game (PBP is always right about who STARTED, only wrong
    about the change), and the team's games in week order are assigned starter for
    the first G_starter games and the successor thereafter (trailing playoff games
    go to the successor, the end-of-season coach). `sum(G)` is asserted to equal
    the regular-season game count.

Names are canonicalized to the coaching-tree identity (the raw Coaches/ directory
names, which equal coaches.json), so downstream joins (inheritance, gene<->WAR)
see one consistent name; this also repairs the "Jim Mora" -> "Jim Mora Jr" and the
"Jay Rosburg" -> "Jerry Rosburg" (PBP typo) orphans.

Team-seasons in `drop_team_seasons` are removed entirely (used to drop NOR 2012,
the Payton-suspension co-head-coach year whose two coaches were never full-time
head coaches).

The starter is taken from the PBP week-1 game rather than PFR's Primary_Coach on
purpose: Primary_Coach is wrong for NOR 2012 (it names the successor), and while
NOR 2012 is dropped, keying off PBP week-1 keeps the logic robust to any future
season with the same shape.

ASCII only (Windows console).
"""

import sys
from pathlib import Path
from collections import defaultdict
import logging

import pandas as pd

# Allow standalone execution (python utils/coach_attribution.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crawlers.utils.data_constants import standardize_team_abbreviation, pfr_to_pbp
from utils.data_paths import canonicalize_coach_name

_logger = logging.getLogger(__name__)

# Known PBP coach-name typos that canonicalization cannot bridge (different first
# name), mapped to the coaching-tree canonical key. Same-person, verified.
_PBP_NAME_ALIASES = {
    "jay rosburg": "jerry rosburg",   # DEN 2022 2-game interim, PBP mislabel
}

_PBP_COLS = {"game_id", "season", "week", "home_team", "away_team",
             "home_coach", "away_coach"}


def _canon(name):
    k = canonicalize_coach_name(name)
    return _PBP_NAME_ALIASES.get(k, k)


def _load_pbp_game_teams(start, end, pbp_dir):
    """One row per (game_id, team) with that team's PBP coach, week-datable."""
    frames = []
    for yr in range(start, end + 1):
        fp = Path(pbp_dir) / f"play_by_play_{yr}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp, usecols=lambda c: c in _PBP_COLS, low_memory=False)
        g = df.drop_duplicates("game_id")
        home = g[["game_id", "season", "week", "home_team", "home_coach"]].rename(
            columns={"home_team": "team", "home_coach": "coach"})
        away = g[["game_id", "season", "week", "away_team", "away_coach"]].rename(
            columns={"away_team": "team", "away_coach": "coach"})
        frames.append(pd.concat([home, away], ignore_index=True))
    gt = pd.concat(frames, ignore_index=True)
    gt = gt.dropna(subset=["team"])
    gt["season"] = gt["season"].astype(int)
    gt["week"] = gt["week"].astype(int)
    return gt


def _load_results_coach_games(start, end, coaches_dir):
    """List of (team_pbp_code, season, dir_name, G) for NFL coach-of-record rows."""
    recs = []
    for d in sorted(Path(coaches_dir).iterdir()):
        if not d.is_dir():
            continue
        fp = d / "all_coaching_results.csv"
        if not fp.exists():
            continue
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        if "Lg" in df.columns:
            df = df[df["Lg"] == "NFL"]
        for _, r in df.iterrows():
            yr = r.get("Year")
            if pd.isna(yr) or not (start <= int(yr) <= end):
                continue
            g = r.get("G")
            if pd.isna(g) or int(g) <= 0:
                continue
            tm = r.get("Tm")
            if pd.isna(tm):
                continue
            pbp = pfr_to_pbp(standardize_team_abbreviation(tm, int(yr)), int(yr))
            recs.append((pbp, int(yr), d.name, int(g)))
    return recs


def build_game_coach_map(start_year, end_year, pbp_dir, coaches_dir,
                         drop_team_seasons=None, logger=None):
    """Return DataFrame[game_id, team, season, head_coach] (tree-canonical name).

    team is the PBP/nflfastR code (join to plays on posteam or defteam).
    """
    log = logger or _logger
    gt = _load_pbp_game_teams(start_year, end_year, pbp_dir)
    recs = _load_results_coach_games(start_year, end_year, coaches_dir)

    res = defaultdict(list)                      # (team, season) -> [(dir_name, G)]
    for pbp, yr, name, g in recs:
        res[(pbp, yr)].append((name, g))

    # canonical key -> tree/dir display name (for emitting a consistent identity)
    canon2dir = {}
    for _, _, name, _ in recs:
        canon2dir.setdefault(canonicalize_coach_name(name), name)

    def to_dir(name):
        return canon2dir.get(_canon(name), name)

    drop = {(t, int(s)) for t, s in (drop_team_seasons or [])}
    n_override = n_dropped = n_warn = 0
    rows = []

    for (team, season), grp in gt.groupby(["team", "season"]):
        if (team, int(season)) in drop:
            n_dropped += 1
            continue
        grp = grp.sort_values("week")
        pbp_canons = {_canon(c) for c in grp["coach"].dropna()}
        rcoaches = res.get((team, int(season)), [])
        rcanon = {canonicalize_coach_name(n) for n, _ in rcoaches}

        # Regular-season games from PBP (playoffs are higher week numbers).
        reg_week_max = 17 if int(season) <= 2020 else 18
        reg_games_pbp = int((grp["week"] <= reg_week_max).sum())
        reg_games_res = int(sum(g for _, g in rcoaches))

        override = len(rcoaches) >= 2 and len(pbp_canons) < len(rcanon)
        # Guard: a medical-leave / suspension season where PFR credits the HC of
        # record the FULL year AND the interim their games double-counts G
        # (e.g. IND 2012 Pagano 16 + Arians 12 = 28 vs a 16-game season). We
        # cannot recover a clean boundary from G there, and the HC of record is
        # who PBP already shows, so fall back to the PBP per-game coach.
        if override and reg_games_res != reg_games_pbp:
            n_warn += 1
            log.warning(f"[coach_attr] {team} {season}: results sum G={reg_games_res} "
                        f"!= {reg_games_pbp} reg games (HC-of-record double-count?); "
                        f"using PBP per-game coach")
            override = False

        if override:
            starter_canon = _canon(grp.iloc[0]["coach"])
            gmap = {canonicalize_coach_name(n): (n, g) for n, g in rcoaches}
            if starter_canon not in gmap:
                n_warn += 1
                log.warning(f"[coach_attr] {team} {season}: PBP starter "
                            f"{grp.iloc[0]['coach']} not in results {list(gmap)}; "
                            f"falling back to PBP per-game coach")
                for _, r in grp.iterrows():
                    rows.append((r["game_id"], team, int(season), to_dir(r["coach"])))
                continue
            starter = gmap[starter_canon]
            successors = [v for k, v in gmap.items() if k != starter_canon]
            g_starter = starter[1]
            games = grp["game_id"].tolist()      # week-sorted (reg then post)
            # first g_starter games -> starter; remainder (incl playoffs) ->
            # successor, the end-of-season coach
            successor_name = successors[0][0] if successors else starter[0]
            for i, gid in enumerate(games):
                name = starter[0] if i < g_starter else successor_name
                rows.append((gid, team, int(season), to_dir(name)))
            n_override += 1
        else:
            for _, r in grp.iterrows():
                rows.append((r["game_id"], team, int(season), to_dir(r["coach"])))

    out = pd.DataFrame(rows, columns=["game_id", "team", "season", "head_coach"])
    log.info(f"[coach_attr] game-team rows={len(out)}; Bug-B overrides={n_override}; "
             f"team-seasons dropped={n_dropped}; warnings={n_warn}")
    return out


def attach_head_coach(plays, game_coach_map, team_col, out_col="head_coach"):
    """Merge the authoritative coach onto plays via (game_id, team_col).

    Returns a copy of `plays` with `out_col` set to the tree-canonical coach
    (NaN where team_col is NaN or the team-season was dropped).
    """
    m = game_coach_map.rename(columns={"team": team_col, "head_coach": out_col})[
        ["game_id", team_col, out_col]]
    merged = plays.merge(m, on=["game_id", team_col], how="left")
    return merged


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Diagnostics for game-level coach map")
    ap.add_argument("--start_year", type=int, default=2006)
    ap.add_argument("--end_year", type=int, default=2024)
    ap.add_argument("--pbp_dir", default="data/raw/play_by_play")
    ap.add_argument("--coaches_dir", default="data/raw/Coaches")
    args = ap.parse_args()

    m = build_game_coach_map(args.start_year, args.end_year, args.pbp_dir,
                             args.coaches_dir, drop_team_seasons=[("NO", 2012)])
    # Show every team-season split into >1 coach (the mid-season universe)
    per = m.groupby(["team", "season"])["head_coach"].agg(
        lambda s: s.value_counts().to_dict())
    multi = {k: v for k, v in per.items() if len(v) > 1}
    print(f"\nTeam-seasons split across >1 coach: {len(multi)}")
    for (team, season), counts in sorted(multi.items()):
        parts = ", ".join(f"{c} [{n}g]" for c, n in counts.items())
        print(f"  {team} {season}: {parts}")
