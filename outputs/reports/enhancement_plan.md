# Coach_Tree Enhancement Plan (post-audit, 2026-06-24)

Comprehensive remediation of the upstream-of-paper audit (data transformation, leakage,
methodology) plus recommended enhancements. Goal: correct every substantive finding, then run
ONE regeneration cascade, then re-thread the paper. ASCII only.

## Status today (already done, not yet cascaded or committed)

- **Reliability-weighted aggression composite** implemented in `calculate_aggression_gene.py`.
  Each component is weighted per coach-year by `rel = tau2 / (tau2 + sampling_var)`, where
  `sampling_var = sum(phat(1-phat))/n^2` and `tau2` (true between-coach variance) is estimated by
  DerSimonian-Laird. Components below `rel_floor` (0.1) get zero weight. Replaces the equal-weight
  mean. Validated against split-half reliability.
  - Measured reliabilities: pass_heavy 0.90, deep_pass 0.69, fourth_down 0.41, two_point 0.14
    (two_point effectively drops out; it has ~no stable between-coach signal).
  - Headline composite gene -> WAR: r 0.172 -> **0.181** (p<1e-4, n=606). Strengthened.
  - OLD vs NEW composite z correlation 0.808 (a real re-centering -> downstream must be rerun).

## Locked design decisions

- **Gene stays absolute and season-agnostic.** Models deliberately exclude season; the gene
  measures deviation from a time-invariant, game-state-conditioned baseline, not from the
  contemporaneous peer average. The upward drift over 2006-2024 is the intended diffusion signal,
  not an artifact. We will NOT add season to the models. (Audit finding #1 gene-side: rejected by
  design, on purpose.)
- **Clustered (coach/mentor bootstrap) inference remains primary**; single global BH-FDR family.
- **Reliability weighting is outcome-blind** (weights never see WAR), so it is not circular.

---

## Workstreams

Legend: effort S/M/L; "retrain" = needs model refit; "cascade" = needs analysis rerun.

### WS1 - Cross-fitting / no in-sample residual  (Medium-High severity; L; retrain+cascade)  [IMPLEMENTED 2026-06-25]
Implemented: `mp.crossfit_predict` + `build_xgb_estimator` (GroupKFold leave-coach-out for offense /
leave-team-out for defense, reuses tuned params, no re-search); all 3 gene calculators score via a
`_predict_component` method with `--no_crossfit`/`--cv_splits` flags. Key result: cross-fit lifted the
small-sample component reliabilities sharply (in-sample had been memorizing each coach's own rare-
situation tendencies) -- aggression two_point 0.14 -> 0.67, fourth_down 0.41 -> 0.65, pass_heavy ~0.92.
Original notes below.


Problem: genes are scored on the same plays the models trained on (~80% in-sample), so the
"expected" baseline partly fits each coach's own behavior and attenuates the gene toward zero.
Fix: out-of-group (cross-fitted) predictions. Add `crossfit_predict(df, y, feature_names,
categorical, groups, params, k=5)` to `utils/model_pipeline.py`: GroupKFold by coach (offense) /
defteam (defense); for each fold fit the pipeline (encode/impute/SVD + selected features) on the
other folds using the already-tuned hyperparameters from metadata `best_params` (no re-tuning),
predict the held-out fold; concatenate OOF predictions. The 4 offensive + 3 defensive gene
calculators call this instead of loading the single persisted model for scoring. The
all-data model stays as the reported/metadata model.
Files: `utils/model_pipeline.py`, all 4 `calculate_*_gene.py`.
Cost: ~5 plain fits x 7 models (no RandomizedSearch) ~ 30-90 min. Note: changes phat -> recompute
reliability weights (WS depends-on order below). Fallback if too costly: document attenuation as a
limitation instead.

### WS2 - Protected game-state core  (Medium; M; retrain+cascade)  [IMPLEMENTED 2026-06-25]
Problem: stability selection optimizes parsimonious prediction, not residual purity, so a real
game-state confounder can be dropped (for redundancy or borderline stability) and leak into the
gene. A scan of all 10 selection JSONs showed selection was mostly clean -- the headline component
(pass_heavy / run_pass) lost only zero-gain features -- with ONE real gap: pass_rush dropped
`score_differential` (freq 0.44).
Decision (agreed with Jon): protect a pre-specified granular game-state core, defined on
substantive grounds, NOT tuned on measured frequencies (that would reintroduce a forking path).
- Core (`utils/model_features.get_protected_core_features`): clock (quarter+half seconds + qtr),
  score (posteam/defteam + margin), yardline_100, down/ydstogo/goal_to_go, both timeouts (12 for
  offense; +shotgun,no_huddle = 14 for defense). Granular member of each axis: continuous clocks
  over qtr buckets, raw scores + margin, yardline_100 over side_of_field.
- Deliberately excluded: game_seconds_remaining (>0.95 redundant casualty), game_half (coarsening
  of qtr), side_of_field (coarsening of yardline_100), and drive context (drive_play_count,
  drive_first_downs, ydsnet) -- the last because it is ENDOGENOUS to the coach's own play-calling
  and conditioning on it would shrink the gene.
- Structural restriction (frequency-blind): the consumer keeps only core members that are PRESENT
  and NON-CONSTANT for that model, so `down` drops from the 4th-down model (constant) and the
  down/distance/field block drops from two_point (undefined post-TD) automatically.
Implementation: `run_stability_selection.py` passes the restricted core to
`drop_redundant_features(protect=...)` and force-unions it into the selected set; JSON now records
`protected_core`, `protected_core_excluded`, `forced_in`. Validated end-to-end on fourth_down
(excludes `down`) and pass_rush (gains `score_differential`).
Impact: 5/10 models are bit-identical (run_pass, no_huddle, pace, box_stacking, two_point -- the
core was already fully selected). The headline pass_heavy model does NOT move. Real change is
concentrated in pass_rush (+score_differential), with minor borderline adds in pass_target /
shotgun / man_coverage / fourth_down.
Separate lever (NOT WS2): the AUC drop on shotgun/no_huddle/box from pi=0.6 -- forcing the core in
may recover some of it; re-check retrained AUCs before deciding whether a pi=0.5 re-selection of
those three is still needed.
Files: `utils/model_features.py`, `scripts/models/run_stability_selection.py`.

### WS3 - Calibration of decision models  (VERIFY-ONLY; S; rides on WS1, no extra retrain)  [IMPLEMENTED 2026-06-25 - all 7 classifiers OK, no recalibration]
RESULT: ECE on cross-fit OOF (outputs/analysis/calibration_metrics.json): fourth_down .0030,
run_pass .0025, pass_target .0033, two_point .0065, shotgun .0103, man_coverage .0177, no_huddle
.0276 (all flag OK, <=.03). No isotonic recalibration warranted. Two non-uniform tails to note in
paper limitations: no_huddle over-confident at the high end (max-bin-gap .289, but mass region
calibrated; feeds only the NS tempo composite) and two_point sparse high-conf buckets (gap .116,
already reliability-downweighted). Original plan below.

Rationale: gene = actual - predicted_prob; a miscalibrated classifier could bias the residual.
Decision (agreed with Jon): downgraded from recalibrate-everything to verify-only. AUC measures
discrimination, not calibration, but XGBoost on log-loss is usually well-calibrated out of the box;
the residual structure also cancels a CONSTANT calibration bias (gene is differenced then z-scored
across coaches), so only non-uniform error in sparse buckets (4th-and-long, two-point) is a risk --
and those are already the low-reliability components we downweight.
Plan: compute ECE (10-15 bins) + a reliability curve per classifier on the WS1 OUT-OF-FOLD
predictions (in-sample calibration is meaningless). Report ECE in metadata + one appendix figure.
Recalibrate (isotonic) ONLY a model with ECE > ~0.03-0.05 AND non-uniform error. Default
expectation: all pass, report and move on. No separate model fit -- rides entirely on WS1 OOF.
Files: `scripts/validation/validate_calibration.py` (reads WS1 OOF predictions).

### WS4 - Reliability weighting for tempo + defensive composites  (enhancement; M; cascade)  [IMPLEMENTED 2026-06-25]
Implemented in `parsimony.reliability_weighted_composite` (+ `dersimonian_laird_tau2`,
`reliability_weights`); aggression refactored onto it. `value_suffix` param weights raw genes
(aggression, same unit) vs z-scores (tempo/defensive, mixed unit; reliability is scale-invariant so
computed from raw genes either way). Noise term: classifiers sum phat(1-phat); regressors (pace, box,
rush) sum squared residual. Arbitrary min_plays gates removed (reliability handles low-n). Original
notes below.


Make all composite genes use the same DL reliability weighting as aggression, for methodological
uniformity. Generalize the noise term: classifier components use `phat(1-phat)`; regression
components (pace, box, pass_rush) use the model residual variance. no_huddle (cls) + pace (reg) ->
tempo; box (reg) + pass_rush (reg) + man (cls) -> defensive scheme.
Files: `calculate_tempo_gene.py`, `calculate_defensive_scheme_gene.py`, shared helper extracted
from `calculate_aggression_gene.py` (move `_dersimonian_laird` + weighting into `utils/parsimony.py`).

### WS5 - Lead-lag gap guard + clustered SEs  (High; S; cascade)  [IMPLEMENTED 2026-06-25]
Problem: `analyze_aggression_lead_lag_by_era.py` pairs `shift(1)` without a contiguous-season
check (e.g. McDaniels 2010 -> 2022 treated as t-1; 25/483 pairs span gaps), and uses plain OLS SEs.
Fix: also shift `season`, drop pairs where `season - season_lag != lag`; route era regressions
through `cluster_robust_ols(clusters=coach)`.
Files: `scripts/analysis/analyze_aggression_lead_lag_by_era.py`.

### WS6 - Grouped CV for reported model metrics  (High for paper honesty; M; no gene change)  [IMPLEMENTED 2026-06-25 - run_grouped_cv_metrics.py]
RESULT (grouped-CV vs documented random-split): most match; real drops where random split was
optimistic -> no_huddle AUC .766 vs .858, shotgun .806 vs .821, man_coverage .682 vs .710, pass_rush
R2 .055 vs .068. Paper model table should report grouped numbers.
Problem: reported AUC/RMSE/R2 are from a random play-level split (inflated); `grouped_cv_score`
exists but is never called.
Fix: compute and persist grouped-CV metrics (group = game_id or team-year) for all 10 models into
metadata; report these in the paper's model table (or alongside the random-split number).
Files: all `scripts/models/*.py` (or a one-off evaluator), `utils/parsimony.grouped_cv_score`.

### WS7 - Head-coach primary selection tiebreak  (Medium; S; cascade)  [IMPLEMENTED 2026-06-25 - only 1 pre-gene-era cell changed]
Problem: `extract_head_coaches.py` breaks multi-HC team-year ties alphabetically; the computed
`Is_Starter` (>=10 games) signal is ignored. Feeds the OC->HC inheritance fallback attribution.
Fix: pick the season-opening HC by games played before any alphabetical fallback.
Files: `scripts/data_processing/extract_head_coaches.py`.

### WS8 - WAR join canonicalization + attrition logging  (Medium; S; cascade)  [IMPLEMENTED 2026-06-25]
NOTE: did NOT blanket-strip Jr/Sr (Mora Sr 1986-2001 vs Jr 2004-2009 are distinct people, both in
WAR). Used a verified alias "jim mora"->"jim mora jr" + punctuation/whitespace canon + attrition
logging. Recovered Jim Mora (n 606->608); remaining unmatched are legit interims. Wired into the two
gene<->WAR correlation/regression consumers; FE/temporal/power/mentor still raw-merge (would only
gain Mora's 2 rows).
Problem: 5 gene<->WAR merges key on raw coach names with no normalization and no logging; ~35/641
coach-years drop (mostly legit interims, but e.g. "Jim Mora" vs "Jim Mora Sr/Jr" is a true loss).
Fix: a shared name-canonicalizer (strip ` Jr/Sr/II/III`, normalize whitespace/punctuation) +
log `set(gene_names) - set(war_names)` and dropped-row counts on every merge.
Files: `utils/data_paths.py` (or new `utils/names.py`), the 5 consumers.

### WS9 - Team-abbreviation + tree hygiene  (Low; S; cascade for B2/B3)
- B2 [IMPLEMENTED 2026-06-25, verified result-neutral]: deleted the triplicated local
  `normalize_team_abbr` dicts in all 3 gene calculators; coach_dict keys and the fallback lookup
  both route through year-aware `standardize_team_abbreviation(team, year)`. Verified neutral on
  2006-2024 (641/641 coach-years identical, 0 play-count cells changed - the fallback was dead),
  so no gene re-run needed; the value is removing the trap, not changing results.
- B3/B4 NOT done (deferred): build_coaching_tree role-priority and download_play_by_play max_year
  are out of the gene/paper critical path.
- B3: `build_coaching_tree.py:164` keep highest-ranked role per coach-year (HC > OC/DC/STC >
  position) instead of first-seen.
- B4 (informational): parameterize `download_play_by_play.py` max_year.

### WS10 - Paper re-thread  (after the single cascade; M)
Re-thread every number; add a methods paragraph on reliability weighting (DL tau2 + split-half
validation) and the absolute-gene rationale (diffusion by design). Optional framing: present the
within-year-demeaned / two-way-FE estimate as the conservative anchor for gene->WAR (the pooled r
has a between-year component because WAR also trends with year; FE p=0.003 isolates the within-era
effect). Update `paper_delta_map.md`.

---

## Robustness pass v2 (WS11-WS15) - 2026-06-25 [IMPLEMENTED, pre-paper]

Second methodological pass (after the WS1-WS9 audit) focused on what survives a clean
pipeline: what the gene->WAR claim means, not whether the code is correct. NO model
retraining anywhere; WS13/WS14 re-run gene CALCULATORS (inference + changed aggregation) only.

### WS11 - WAR-measurement-noise-aware gene->WAR  [IMPLEMENTED]
Finding: a single-season WAR is only ~24% reliable (year-over-year test-retest r=0.24); the rest
is luck (binomial floor). Classical noise in the DEPENDENT variable ATTENUATES the correlation, so
the season-level r=0.189 is a FLOOR, not inflated. Also: the WAR file's `annual_games` is mislabeled
(it is WAR-in-games, not games coached); real games reconstructed from raw PFR results (`G`, 100%
match) in `data_paths.load_coach_year_games`.
- New: `utils/war_noise.py` (war_noise_robustness + career_level_corr); `parsimony.weighted_pearson`,
  `cluster_bootstrap_corr_weighted`, `reliability_from_variance`, `disattenuate_r`;
  `data_paths.add_war_precision` (games-based proxy SE = 2.4833*sqrt(16/games)).
- Views added to gene_war + aggression_war: inverse-variance-weighted r, partial-season-drop
  sensitivity, empirical WAR reliability, and the CAREER-LEVEL anchor (one row/coach, WAR reliable).
- DELIBERATELY NO disattenuated point estimate (rel~0.24 -> dividing by sqrt(rel) is false precision).
- RESULT: composite aggression->WAR season 0.189 -> CAREER 0.303 (p=.0006, n=124); 4th-down 0.095 ->
  career 0.187; pass-heavy 0.183 -> 0.251; deep/2pt weak everywhere. IVW + partial-season hold.

### WS12 - Small-cluster subgroup inference  [IMPLEMENTED]
Added `parsimony.wild_cluster_bootstrap_corr` (Cameron-Gelbach-Miller restricted, Rademacher) +
`corr_with_small_cluster_guard` (auto WCB + `small_cluster` flag when n_clusters < 40, textbook).
Wired into gene_inheritance, inheritance_by_coach_type, shotgun_inheritance_by_coach_type,
mentor_war (era splits were naive Pearson -> now clustered), persistence_by_coach_type. BH now
prefers `p_wild_cluster` for small-cluster tests.
- SURVIVE WCB: OC->HC shotgun (p=.003), OC->HC pass-heavy (.022), DC->HC defensive scheme (.0155,
  n=17 so suggestive), gene-persistence subgroups (strong).
- DEMOTED (naive p was anti-conservative): OC->HC 4th-down (.001->.187), OC->HC composite aggression
  (.012->.209), offensive-mentor 4th-down. [percentile clustered p already demoted these; WCB confirms]

### WS13 - Defensive composite 2018 structural break  [IMPLEMENTED]
The composite is a reliability-weighted mean of k component z-scores; k=2 (box, rush) for 2016-17 and
k=3 (adds man) for 2018+, and the mean of k unit-variance terms has k-dependent variance -> a scale
jump at 2018. Fix: z-score the composite WITHIN each `scheme_components` regime so 2017 and 2019 genes
are comparable. Verified: each regime now mean 0, std 1. Defensive gene recomputed.

### WS14 - Shotgun reliability  [IMPLEMENTED]
Folded shotgun (the one gene outside the WS4 framework) in via empirical-Bayes SHRINKAGE
(gene_shrunk = mean + rel*(gene-mean), rel = tau2/(tau2+samp_var) from per-play phat(1-phat)).
Practical impact negligible: HC coach-years have ample plays so reliability is 0.95-0.996 (mean .993),
shrinkage moves the gene ~0.001. Honest finding: shotgun was never at risk; framework now consistent.

### WS15 - Selection / filtering sensitivity  [IMPLEMENTED]
New `analyze_selection_sensitivity.py`. Attrition: 641 gene coach-years, 608 matched to WAR, 33
dropped (14 coaches), all short-tenure interims (mean 1.07 seasons) with mean gene -0.002 vs kept
+0.003 -> NOT a biased slice. Headline holds across all cuts: all .189, excl-one-season .189,
excl-partial .211, career .305. Added to run_all_analyses.

### Deferred (documented limitation, not implemented)
Opponent-strength adjustment in the upstream play models (would require retraining all offensive
models; arguably part of the coach's signature). Pooled cross-era z-scoring is NOT a flaw (settled
absolute/season-agnostic design).

---

## Execution order (one cascade at the end)

1. Model-level changes together (they all force refits): **WS2** (feature set) -> **WS3**
   (calibration) -> retrain the 10 models once.
2. Attribution/data fixes: **WS7**, **WS9 B2/B3** (no retrain).
3. Gene recompute with **WS1** (cross-fit) + **WS4** (reliability weighting on all composites);
   aggression already done, re-run it under cross-fit too so all genes are consistent.
4. Downstream fixes: **WS5**, **WS8**, plus **WS6** (model metrics).
5. **Single full cascade**: `run_all_analyses.py` + viz + BH + `verify_paper_statistics.py`.
   Quantify before/after deltas (gene stability corr, gene->WAR, inheritance, persistence, era).
6. **WS10** paper edits. Commit only when asked.

## Verification

- `python utils/parsimony.py` self-test still passes after moving the DL/weighting helper there.
- Per gene: assert reliability weights sum sensibly; spot-check anchors (Chip Kelly fastest tempo).
- Cross-fit sanity: OOF AUC/R2 close to grouped-CV; gene magnitudes should INCREASE vs in-sample
  (attenuation removed).
- `verify_paper_statistics.py` clean; BH family recomputed; no undefined refs in the rebuilt PDF.

## Open questions for tomorrow

1. WS2: protect-core-context vs pi=0.5 re-selection? (recommend protect `location`, `side_of_field`.)
2. WS1 cross-fit grouping: coach for offense, defteam for defense - confirm; k=5 ok?
3. WS3 calibration: report-only, or actually recalibrate the probabilities used in the gene?
4. WS6: replace the random-split metric in the paper table, or report both?
5. Any appetite for the parked survival-analysis project (genes -> firing hazard)?
