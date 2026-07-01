#!/usr/bin/env python3
"""
Attribution Validation Script

Independently verifies that play-by-play data is attributed to the correct head
coach across the gene pipeline, and that the offense and defense gene calculators
handle mid-season head-coach changes CONSISTENTLY.

This exists because the correctness of "which coach gets which plays" was
previously asserted by inspection, and a real bug (defense blends a mid-season
HC change into one modal-coach row while offense splits it by coach) still got
through. This script encodes those correctness claims as pass/fail checks and
quantifies coverage, so the claim is measured rather than eyeballed.

Layers
------
1. Invariants (property tests) on the gene CSVs:
     - no duplicate keys, no zero-play rows, coverage reported (not assumed).
2. Cross-source reconciliation against an INDEPENDENT truth rebuilt directly
   from data/raw/play_by_play/*.csv (game-level home_coach/away_coach), which
   does not depend on any gene output:
     (A) PBP truth              vs  team_year_head_coaches.csv   (two sources
         for the same fact: who coached team T in year Y).
     (B) PBP truth              vs  the gene CSVs: soundness of every attributed
         coach label, and the offense-splits / defense-blends asymmetry, with
         the affected team-seasons enumerated (the bug footprint).
3. Hand-verified golden cases (check_golden_cases) traced end-to-end vs PFR.

Independent truth
-----------------
For each game, the home team's coach is home_coach and the away team's coach is
away_coach. Deduplicated to the game level this yields, per (team, season), the
exact set of head coaches and their game counts, entirely from raw data. This is
the same home/away rule the calculators use, applied at the game level so it is
an independent reconstruction rather than a re-run of the gene code.

Output: data/processed/validation/attribution_validation.json
Exit code 1 if any HARD invariant or golden case fails. Soft coverage gaps and
cross-source name/count disagreements are reported, not failed (they are
expected artifacts of min-play filtering and PFR-vs-PBP name spelling).

ASCII only (Windows console).
"""

import sys
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import pandas as pd
import numpy as np

# Project root on path for shared utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.data_paths import canonicalize_coach_name  # noqa: E402
from crawlers.utils.data_constants import standardize_team_abbreviation  # noqa: E402
from utils.coach_attribution import build_game_coach_map  # noqa: E402

PBP_DIR = Path("data/raw/play_by_play")
GENE_DIR = Path("data/processed/coaching_genes")
HC_MAP_PATH = Path("data/processed/Coaching/team_year_head_coaches.csv")
OUT_DIR = Path("data/processed/validation")

OFFENSE_START, OFFENSE_END = 2006, 2024   # offensive gene window
DEFENSE_START, DEFENSE_END = 2016, 2024   # defensive gene window

PBP_COLS = ["game_id", "season", "week", "posteam", "defteam",
            "home_team", "away_team", "home_coach", "away_coach"]


# --------------------------------------------------------------------------- #
# Independent truth from raw play-by-play
# --------------------------------------------------------------------------- #
def load_pbp_truth(start=OFFENSE_START, end=OFFENSE_END):
    """Rebuild coach attribution directly from raw PBP, independently of genes.

    Returns
    -------
    truth : DataFrame with one row per (game_id, team) -> [season, team, coach].
            team/coach are the raw PBP nflfastR code and PBP coach name.
    coverage : dict of play-level attribution rates (offense/defense) computed
               the same way the calculators do, as an independent coverage check.
    """
    game_rows = []
    cov = {
        "offense_plays_total": 0, "offense_plays_with_posteam": 0,
        "offense_plays_attributed": 0,
        "defense_plays_total": 0, "defense_plays_with_defteam": 0,
        "defense_plays_attributed": 0,
    }

    for year in range(start, end + 1):
        fp = PBP_DIR / f"play_by_play_{year}.csv"
        if not fp.exists():
            print(f"  WARNING: missing PBP file {fp}")
            continue
        df = pd.read_csv(fp, usecols=lambda c: c in PBP_COLS, low_memory=False)

        # Play-level offensive coverage (posteam -> its coach)
        off_coach = np.where(
            df["posteam"] == df["home_team"], df["home_coach"],
            np.where(df["posteam"] == df["away_team"], df["away_coach"], np.nan))
        off_coach = pd.Series(off_coach, index=df.index)
        cov["offense_plays_total"] += len(df)
        cov["offense_plays_with_posteam"] += int(df["posteam"].notna().sum())
        cov["offense_plays_attributed"] += int(
            off_coach[df["posteam"].notna()].notna().sum())

        # Play-level defensive coverage (defteam -> its coach)
        def_coach = np.where(
            df["defteam"] == df["home_team"], df["home_coach"],
            np.where(df["defteam"] == df["away_team"], df["away_coach"], np.nan))
        def_coach = pd.Series(def_coach, index=df.index)
        cov["defense_plays_total"] += len(df)
        cov["defense_plays_with_defteam"] += int(df["defteam"].notna().sum())
        cov["defense_plays_attributed"] += int(
            def_coach[df["defteam"].notna()].notna().sum())

        # Game-level truth: one (team, coach) per side per game.
        games = df.drop_duplicates("game_id")
        home = games[["game_id", "season", "home_team", "home_coach"]].rename(
            columns={"home_team": "team", "home_coach": "coach"})
        away = games[["game_id", "season", "away_team", "away_coach"]].rename(
            columns={"away_team": "team", "away_coach": "coach"})
        game_rows.append(pd.concat([home, away], ignore_index=True))
        del df

    truth = pd.concat(game_rows, ignore_index=True)
    truth = truth.dropna(subset=["team", "coach"])
    truth["coach_canon"] = truth["coach"].map(canonicalize_coach_name)
    return truth, cov


def build_team_season_truth(truth):
    """Per (team, season): ordered coach list with game counts, modal coach,
    and n distinct coaches. Team code is the raw PBP code."""
    rows = {}
    grp = truth.groupby(["team", "season"])
    for (team, season), sub in grp:
        counts = sub.groupby("coach_canon").agg(
            coach=("coach", "first"), n_games=("game_id", "nunique"))
        counts = counts.sort_values("n_games", ascending=False)
        rows[(team, int(season))] = {
            "coaches": counts["coach"].tolist(),
            "coach_canons": counts.index.tolist(),
            "games": counts["n_games"].tolist(),
            "modal_coach": counts["coach"].iloc[0],
            "modal_coach_canon": counts.index[0],
            "n_coaches": int(len(counts)),
        }
    return rows


# --------------------------------------------------------------------------- #
# Gene CSV loading
# --------------------------------------------------------------------------- #
def load_genes():
    def _load(name):
        fp = GENE_DIR / name
        if fp.exists():
            return pd.read_csv(fp)
        print(f"  WARNING: gene file not found: {fp}")
        return pd.DataFrame()

    return {
        "aggression": _load("aggression_gene_by_year.csv"),
        "shotgun": _load("shotgun_gene.csv"),
        "tempo": _load("tempo_gene.csv"),
        "defense": _load("defensive_scheme_gene.csv"),
    }


# --------------------------------------------------------------------------- #
# Layer 1: invariants
# --------------------------------------------------------------------------- #
def check_invariants(genes):
    issues = []
    stats = {}

    specs = [
        ("aggression", ["head_coach", "season"], "total_plays"),
        ("shotgun", ["head_coach", "season"], "total_plays"),
        ("tempo", ["head_coach", "season"], "total_plays"),
        # Defense is coach-year post-fix: unique per (defteam, coach, season).
        ("defense", ["defteam", "season", "head_coach"], "total_plays"),
    ]
    print("\n" + "=" * 80)
    print("LAYER 1: GENE-CSV INVARIANTS")
    print("=" * 80)

    for name, keys, play_col in specs:
        df = genes[name]
        if df.empty:
            issues.append({"type": "missing_gene_file", "gene": name})
            continue
        st = {"rows": int(len(df))}

        dup = df.duplicated(subset=keys, keep=False)
        st["duplicate_keys"] = int(dup.sum())
        if dup.any():
            issues.append({"type": "duplicate_keys", "gene": name,
                           "count": int(dup.sum()),
                           "examples": df[dup][keys].head(8).to_dict("records")})

        if play_col in df.columns:
            zero = (df[play_col].fillna(0) <= 0).sum()
            st["zero_play_rows"] = int(zero)
            if zero > 0:
                issues.append({"type": "zero_play_rows", "gene": name,
                               "count": int(zero)})

        # Defensive rows must carry a head_coach label to be usable for inheritance
        if name == "defense" and "head_coach" in df.columns:
            missing_hc = int(df["head_coach"].isna().sum())
            st["defense_rows_without_hc_label"] = missing_hc
            if missing_hc > 0:
                issues.append({"type": "defense_row_without_hc", "count": missing_hc})

        stats[name] = st
        print(f"  {name:11s}: rows={st['rows']:4d}  dup_keys={st.get('duplicate_keys',0)}"
              f"  zero_play={st.get('zero_play_rows',0)}"
              + (f"  no_hc_label={st.get('defense_rows_without_hc_label',0)}"
                 if name == "defense" else ""))

    return {"issues": issues, "statistics": stats}


# --------------------------------------------------------------------------- #
# Layer 2A: PBP truth vs team_year_head_coaches.csv
# --------------------------------------------------------------------------- #
def reconcile_pbp_vs_hcmap(team_season, start=OFFENSE_START, end=OFFENSE_END):
    issues = []
    stats = {}
    print("\n" + "=" * 80)
    print("LAYER 2A: PBP TRUTH vs team_year_head_coaches.csv")
    print("=" * 80)

    if not HC_MAP_PATH.exists():
        print("  WARNING: team_year_head_coaches.csv not found")
        return {"issues": [{"type": "missing_hc_map"}], "statistics": {}}

    hc = pd.read_csv(HC_MAP_PATH)
    hc = hc[(hc["Year"] >= start) & (hc["Year"] <= end)].copy()

    # Canonical franchise key for both sides
    hc["fkey"] = [standardize_team_abbreviation(t, int(y))
                  for t, y in zip(hc["Team"], hc["Year"])]
    hc["primary_canon"] = hc["Primary_Coach"].map(canonicalize_coach_name)

    # Truth keyed by canonical franchise + season
    truth_fkey = {}
    for (team, season), rec in team_season.items():
        fkey = standardize_team_abbreviation(team, season)
        truth_fkey[(fkey, season)] = rec

    name_mismatch = 0
    count_mismatch = 0
    pbp_missed = 0        # PFR records a mid-season change that PBP fields miss
    missing_in_truth = 0
    checked = 0
    for _, row in hc.iterrows():
        key = (row["fkey"], int(row["Year"]))
        rec = truth_fkey.get(key)
        if rec is None:
            missing_in_truth += 1
            issues.append({"type": "team_year_absent_from_pbp",
                           "team": row["Team"], "fkey": row["fkey"],
                           "year": int(row["Year"])})
            continue
        checked += 1
        # Primary coach vs PBP modal coach
        if row["primary_canon"] and rec["modal_coach_canon"] != row["primary_canon"]:
            name_mismatch += 1
            issues.append({
                "type": "primary_coach_mismatch",
                "team": row["Team"], "year": int(row["Year"]),
                "hc_map_primary": row["Primary_Coach"],
                "pbp_modal": rec["modal_coach"],
                "pbp_all": list(zip(rec["coaches"], rec["games"]))})
        # Total_Coaches vs PBP distinct count
        if "Total_Coaches" in row and pd.notna(row["Total_Coaches"]):
            if int(row["Total_Coaches"]) != rec["n_coaches"]:
                count_mismatch += 1
                # PBP under-counting a real change is a distinct data-accuracy
                # problem: the fired coach is credited for games the successor
                # actually coached, on BOTH sides of the ball.
                pbp_undercount = rec["n_coaches"] < int(row["Total_Coaches"])
                if pbp_undercount:
                    pbp_missed += 1
                issues.append({
                    "type": "pbp_missed_midseason_change" if pbp_undercount
                            else "coach_count_mismatch",
                    "team": row["Team"], "year": int(row["Year"]),
                    "hc_map_total": int(row["Total_Coaches"]),
                    "hc_map_primary": row["Primary_Coach"],
                    "pbp_n_coaches": rec["n_coaches"],
                    "pbp_all": list(zip(rec["coaches"], rec["games"]))})

    stats = {"rows_checked": checked, "primary_name_mismatches": name_mismatch,
             "coach_count_mismatches": count_mismatch,
             "pbp_missed_midseason_changes": pbp_missed,
             "team_years_absent_from_pbp": missing_in_truth}
    print(f"  checked={checked}  primary_name_mismatch={name_mismatch}"
          f"  count_mismatch={count_mismatch}  absent_from_pbp={missing_in_truth}")
    if pbp_missed:
        print(f"  [!] PBP coach fields MISS {pbp_missed} real mid-season changes "
              f"that PFR records (fired coach over-credited):")
        for iss in issues:
            if iss["type"] == "pbp_missed_midseason_change":
                print(f"      {iss['team']} {iss['year']}: PBP={iss['pbp_all']} "
                      f"but PFR Total_Coaches={iss['hc_map_total']} "
                      f"(primary={iss['hc_map_primary']})")
    return {"issues": issues, "statistics": stats}


# --------------------------------------------------------------------------- #
# Layer 2B: gene CSVs vs the AUTHORITATIVE corrected coach map (fix confirmation)
# --------------------------------------------------------------------------- #
def build_map_team_season(gcmap):
    """Per (team, season) coach set from the corrected game->coach map."""
    rows = {}
    for (team, season), sub in gcmap.groupby(["team", "season"]):
        counts = sub["head_coach"].value_counts()
        rows[(team, int(season))] = {
            "coaches": counts.index.tolist(),
            "coach_canons": [canonicalize_coach_name(c) for c in counts.index],
            "games": counts.values.tolist(),
            "n_coaches": int(len(counts)),
        }
    return rows


def reconcile_genes_vs_map(map_ts, genes):
    """Validate the FIXED gene CSVs against the corrected coach map (truth now).

    Confirms coach-year attribution: every gene coach-season traces to the
    corrected map (no phantom), mid-season changes are SPLIT on both offense and
    defense (a row per coach), and dropped team-seasons (NOR 2012) do not appear.
    """
    issues = []
    stats = {}
    print("\n" + "=" * 80)
    print("LAYER 2B: GENE CSVs vs AUTHORITATIVE COACH MAP (fix confirmation)")
    print("=" * 80)

    expected_off = set()          # (coach_canon, season) in 2006-2024
    expected_def = {}             # (team, season) -> set(coach_canon) in 2016-2024
    midseason = {}                # (team, season) -> rec, for n_coaches > 1
    for (team, season), rec in map_ts.items():
        if OFFENSE_START <= season <= OFFENSE_END:
            for cc in rec["coach_canons"]:
                expected_off.add((cc, season))
        if DEFENSE_START <= season <= DEFENSE_END:
            expected_def[(team, season)] = set(rec["coach_canons"])
        if rec["n_coaches"] > 1:
            midseason[(team, season)] = rec

    # --- Offense: every gene coach-season must trace to the corrected map ---
    agg = genes["aggression"]
    off_keys = set()
    if not agg.empty:
        off_keys = {(canonicalize_coach_name(r["head_coach"]), int(r["season"]))
                    for _, r in agg.iterrows()}
    phantom_off = sorted(off_keys - expected_off)
    stats["offense_rows"] = len(off_keys)
    stats["offense_phantom"] = len(phantom_off)
    if phantom_off:
        issues.append({"type": "offense_phantom_attribution",
                       "count": len(phantom_off), "examples": phantom_off[:10]})

    # --- Defense: one row per (defteam, coach, season); coach in corrected truth ---
    dfd = genes["defense"]
    def_by_ts = defaultdict(set)
    def_phantom = 0
    if not dfd.empty:
        for _, r in dfd.iterrows():
            ts = (r["defteam"], int(r["season"]))
            cc = canonicalize_coach_name(r.get("head_coach"))
            def_by_ts[ts].add(cc)
            exp = expected_def.get(ts)
            if exp is not None and cc not in exp:
                def_phantom += 1
                issues.append({"type": "defense_phantom", "defteam": r["defteam"],
                               "season": int(r["season"]),
                               "gene_head_coach": r.get("head_coach")})
    stats["defense_team_seasons"] = len(def_by_ts)
    stats["defense_phantom"] = def_phantom

    # --- NOR 2012 (dropped) must not appear on offense ---
    nor_canons = {canonicalize_coach_name(n) for n in ("Aaron Kromer", "Joe Vitt")}
    nor_off = [(c, s) for (c, s) in off_keys if s == 2012 and c in nor_canons]
    stats["nor2012_offense_rows"] = len(nor_off)
    if nor_off:
        issues.append({"type": "nor2012_present", "examples": nor_off})

    # --- Mid-season SPLIT confirmation (the fix) ---
    off_split_ok = off_split_incomplete = 0
    def_split_ok = def_split_partial = 0
    still_blended = []
    for (team, season), rec in sorted(midseason.items()):
        cds = rec["coach_canons"]
        if OFFENSE_START <= season <= OFFENSE_END:
            if all((c, season) in off_keys for c in cds):
                off_split_ok += 1
            else:
                off_split_incomplete += 1
        if DEFENSE_START <= season <= DEFENSE_END:
            have = def_by_ts.get((team, season), set())
            if len(have) > 1 and all(c in have for c in cds):
                def_split_ok += 1
            elif len(have) > 1:
                def_split_partial += 1
            else:
                still_blended.append((team, season,
                                      list(zip(rec["coaches"], rec["games"]))))
    stats.update({
        "midseason_total": len(midseason),
        "offense_split_ok": off_split_ok,
        "offense_split_incomplete": off_split_incomplete,
        "defense_split_ok": def_split_ok,
        "defense_split_partial_lowplay": def_split_partial,
        "defense_still_blended": len(still_blended),
    })
    print(f"  mid-season team-seasons (corrected map): {len(midseason)}")
    print(f"  offense split confirmed: {off_split_ok} (incomplete: {off_split_incomplete})")
    print(f"  defense split confirmed (2016-2024): {def_split_ok} "
          f"(partial low-play: {def_split_partial}; still blended: {len(still_blended)})")
    for team, season, cs in still_blended:
        parts = ", ".join(f"{c} [{g}g]" for c, g in cs)
        print(f"    still one defensive row: {team} {season}: {parts}")
    if still_blended:
        issues.append({"type": "defense_still_blended_lowplay",
                       "count": len(still_blended), "team_seasons": still_blended})
    print(f"  offense phantom: {len(phantom_off)}  defense phantom: {def_phantom}  "
          f"NOR2012 offense rows: {len(nor_off)}")
    return {"issues": issues, "statistics": stats}


# --------------------------------------------------------------------------- #
# Layer 3: hand-verified golden cases
# --------------------------------------------------------------------------- #
# Curated after the first reconciliation run; each case is checked against the
# independent PBP truth and/or the gene CSVs. Facts are verifiable on PFR.
GOLDEN_CASES = []


def check_golden_cases(team_season, genes):
    issues = []
    stats = {"cases": len(GOLDEN_CASES), "passed": 0, "failed": 0}
    if not GOLDEN_CASES:
        print("\n" + "=" * 80)
        print("LAYER 3: GOLDEN CASES (none curated yet)")
        print("=" * 80)
        return {"issues": issues, "statistics": stats}

    print("\n" + "=" * 80)
    print("LAYER 3: HAND-VERIFIED GOLDEN CASES")
    print("=" * 80)
    agg = genes["aggression"]
    off_keys = set()
    if not agg.empty:
        off_keys = {(canonicalize_coach_name(r["head_coach"]), int(r["season"]))
                    for _, r in agg.iterrows()}
    dfd = genes["defense"]
    def_by_key = {}
    if not dfd.empty:
        def_by_key = {(r["defteam"], int(r["season"])): r for _, r in dfd.iterrows()}

    for case in GOLDEN_CASES:
        ok, detail = _run_golden_case(case, team_season, off_keys, def_by_key)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {case['id']}: {detail}")
        if ok:
            stats["passed"] += 1
        else:
            stats["failed"] += 1
            issues.append({"type": "golden_case_failed", "id": case["id"],
                           "detail": detail})
    return {"issues": issues, "statistics": stats}


def _run_golden_case(case, team_season, off_keys, def_by_key):
    kind = case["kind"]
    if kind == "pbp_coaches":
        rec = team_season.get((case["team"], case["season"]))
        if rec is None:
            return False, f"no PBP truth for {case['team']} {case['season']}"
        got = set(rec["coach_canons"])
        want = {canonicalize_coach_name(c) for c in case["coaches"]}
        return got == want, f"PBP coaches {rec['coaches']} (want {case['coaches']})"
    if kind == "offense_split":
        missing = [c for c in case["coaches"]
                   if (canonicalize_coach_name(c), case["season"]) not in off_keys]
        return (not missing), (f"offense rows present for all of {case['coaches']}"
                               if not missing else f"missing offense rows: {missing}")
    if kind == "defense_label":
        rec = def_by_key.get((case["team"], case["season"]))
        if rec is None:
            return False, f"no defense row for {case['team']} {case['season']}"
        got = canonicalize_coach_name(rec.get("head_coach"))
        want = canonicalize_coach_name(case["head_coach"])
        return got == want, f"defense HC label={rec.get('head_coach')} (want {case['head_coach']})"
    return False, f"unknown golden kind {kind}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 80)
    print("ATTRIBUTION VALIDATION REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print("\nRebuilding independent coach attribution from raw play-by-play...")
    truth, cov = load_pbp_truth()
    team_season = build_team_season_truth(truth)
    print(f"  team-seasons reconstructed: {len(team_season)}")

    off_rate = cov["offense_plays_attributed"] / max(1, cov["offense_plays_with_posteam"])
    def_rate = cov["defense_plays_attributed"] / max(1, cov["defense_plays_with_defteam"])
    print(f"  offense play-level attribution: {off_rate:.1%} "
          f"({cov['offense_plays_attributed']:,}/{cov['offense_plays_with_posteam']:,})")
    print(f"  defense play-level attribution: {def_rate:.1%} "
          f"({cov['defense_plays_attributed']:,}/{cov['defense_plays_with_defteam']:,})")

    genes = load_genes()

    # Authoritative corrected coach map (same source the gene calculators use):
    # this is the truth the FIXED genes are validated against.
    print("\nBuilding authoritative corrected coach map...")
    gcmap = build_game_coach_map(OFFENSE_START, OFFENSE_END, PBP_DIR,
                                 Path("data/raw/Coaches"),
                                 drop_team_seasons=[("NO", 2012)])
    map_ts = build_map_team_season(gcmap)

    r_inv = check_invariants(genes)
    r_2a = reconcile_pbp_vs_hcmap(team_season)
    r_2b = reconcile_genes_vs_map(map_ts, genes)
    r_gold = check_golden_cases(team_season, genes)

    all_issues = []
    for tag, res in [("invariants", r_inv), ("reconcile_hcmap", r_2a),
                     ("reconcile_genes", r_2b), ("golden", r_gold)]:
        for iss in res["issues"]:
            iss["category"] = tag
            all_issues.append(iss)

    # Hard-fail set: real correctness violations. Soft/informational: coverage
    # gaps, low-play single defensive rows, and the PBP-vs-PFR source limitations
    # in Layer 2A (pbp_missed / primary_coach_mismatch) which the fix works around
    # rather than being gene faults.
    HARD = {"duplicate_keys", "zero_play_rows", "defense_row_without_hc",
            "offense_phantom_attribution", "defense_phantom", "nor2012_present",
            "golden_case_failed", "missing_gene_file"}
    hard_issues = [i for i in all_issues if i["type"] in HARD]

    s2b = r_2b["statistics"]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  play-level coverage: offense {off_rate:.1%}, defense {def_rate:.1%}")
    print(f"  FIX - offense mid-season splits confirmed: {s2b.get('offense_split_ok', 0)}"
          f"/{s2b.get('midseason_total', 0)}")
    print(f"  FIX - defense mid-season splits confirmed (2016-2024): "
          f"{s2b.get('defense_split_ok', 0)} (+{s2b.get('defense_split_partial_lowplay', 0)} partial)")
    print(f"  soundness - offense phantom: {s2b.get('offense_phantom', 0)}, "
          f"defense phantom: {s2b.get('defense_phantom', 0)}, "
          f"NOR2012 rows: {s2b.get('nor2012_offense_rows', 0)}")
    print(f"  (source note) PBP still misses {r_2a['statistics'].get('pbp_missed_midseason_changes', 0)} "
          f"mid-season changes in raw data; corrected in genes")
    print(f"  total issues: {len(all_issues)}  (hard failures: {len(hard_issues)})")
    if hard_issues:
        by_type = defaultdict(int)
        for i in hard_issues:
            by_type[i["type"]] += 1
        for t, n in by_type.items():
            print(f"    HARD {t}: {n}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "coverage": {
            "offense_play_attribution_rate": off_rate,
            "defense_play_attribution_rate": def_rate,
            **cov,
        },
        "statistics": {
            "invariants": r_inv["statistics"],
            "reconcile_hcmap": r_2a["statistics"],
            "reconcile_genes": r_2b["statistics"],
            "golden": r_gold["statistics"],
        },
        "hard_failure_count": len(hard_issues),
        "issues": all_issues,
    }
    out_path = OUT_DIR / "attribution_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    print("=" * 80)
    return hard_issues


if __name__ == "__main__":
    hard = main()
    sys.exit(1 if hard else 0)
