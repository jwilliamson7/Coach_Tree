# Coach Aggression Paper: Claim-by-Claim Delta Map

## 2026-06-25 v2/v3 RE-THREAD (CURRENT) -- paper rewritten against the WS11-WS15 + V1-V3 outputs

Paper restructured and re-threaded after two further passes (WS11-15 robustness: cross-fit genes,
WAR-noise/career-level, small-cluster wild bootstrap, defensive 2018-break fix, shotgun reliability,
selection sensitivity; V1-V3: coach-vs-team variance, face validity, OOS). Compiles clean (26 pp,
no undefined refs). Key changes from the committed (2026-06-24) draft:

- NEW LEAD Results section "Coaching Genes Are Coach Traits": coach variance ~5x team (composite
  0.51 vs 0.13); gene travels across team changes (composite r=0.61, shotgun 0.55, tempo 0.54;
  mover regression coach_pre b=0.53 p<.001); face-validity table. Two new tables (tab:variance,
  tab:face_validity). This is now the opening result (Jon's directive).
- Gene->WAR refreshed: composite aggression 0.172->0.189 (clust p<.001); shotgun 0.139->0.142
  (.004); defensive 0.135->0.151 (.015); tempo 0.032->0.045 (ns). CAREER-LEVEL anchor added:
  aggression career r=0.303 (p<.001, n=124); the season-level is framed as a conservative floor
  because single-season WAR is ~24% reliable. IVW (0.211) + partial-season (0.212) robustness noted.
- 4th-down->WAR: 0.085/0.051 -> 0.095/0.033 (clustered), but BH-LOST (career-level 0.187 robust).
- Erosion SOFTENED: era composite 0.242/0.159/0.039 -> 0.239/0.137/0.101 (late now 0.10, not ~0.04);
  continuous interaction -1.59/0.055 -> -0.884/0.237 (clearly NS); Chow 2012 4.42/0.012 ->
  3.41/0.034 (conventional-sig, but BH-LOST at cutoff 0.022). Framed as "real but partial, suggestive".
- Coordinator-to-HC: shotgun 0.519->0.473, defensive 0.616->0.572 (wild p .0155, n=17 suggestive),
  TEMPO 0.359->0.091 (NO LONGER transmits after cross-fit; claim dropped), aggression 0.093->0.017.
- Aggression inheritance overall composite 0.101->0.076; OC pass-heavy 0.326->0.316 (clust .016,
  wild .022, survives). Shotgun OC inheritance 0.427->0.418.
- Within-coach FE: 0.39 games/SD/d.20 -> 0.44/d.23, power 97->99%, two-way p=0.003 (table reformatted
  to within-coach betas). Coach-type composite: off 0.178->0.211, def 0.195->0.184.
- Persistence lag1 composite 0.501->0.516; by-type table refreshed (off 4th-down 0.692->0.639;
  def 4th-down lag2 now ns).
- Model table -> grouped-CV (no_huddle 0.858->0.766, shotgun 0.820->0.807, man 0.702->0.682, etc).
- BH family: 101 tests, cutoff 0.028->0.022, 63 BH-sig. Now prefers wild-cluster p for small subgroups.
- METHODS added: cross-fitting, reliability-weighted/formative composites, calibration (ECE<=.03),
  WAR-noise/career anchor, wild cluster bootstrap (Cameron-Gelbach-Miller cite added), coach-vs-team
  variance method, defensive 2018-block z-scoring. New limitations: WAR noise, play-calling
  attribution (HC vs coordinator), formative construct, defensive team-attribution.

The 2026-06-24 map below is retained as history (it documented the FIRST rewrite, now superseded).

---

# (HISTORY) 2026-06-24 map

Generated 2026-06-24 from the regenerated outputs (leakage-free pipeline, stability-selected
models, external WAR, coach/mentor-clustered inference, global BH-FDR).

Legend: OK = unchanged or rounds the same | MINOR = small shift, same conclusion |
NARRATIVE = changes a stated story or significance call. "naive p" = OLS/Pearson p as the paper
currently reports; "clust p" = coach/mentor cluster-bootstrap p (the honest repeated-measures p).

---

## A. Headline narrative-level changes (read these first)

1. NARRATIVE - Composite & pass-heavy aggression -> WAR got STRONGER.
   Composite r 0.118 -> 0.172 (naive p 0.004 -> 2.2e-5; clust p = 0.001).
   Pass-heavy r 0.107 -> 0.170 (clust p = 0.001). Both survive BH. Good for the core claim.

2. NARRATIVE - 4th-down -> WAR is now NON-significant under clustering.
   r 0.085 unchanged; naive p 0.037; clustered p = 0.051; BH rank 66 (not significant).
   Paper currently stars it (Table 10) and leans on it as a co-driver. Must demote.

3. NARRATIVE - The "erosion" robustness is softer.
   Continuous Aggression x Year interaction: beta3 -1.905 -> -1.595, p 0.029 -> 0.0552 (now NS).
   Implied effect 2024: -4.73 (p 0.537) -> +3.46 (p 0.662) (now POSITIVE, not negative).
   Era split still declines (composite 0.242 -> 0.159 -> 0.039) BUT the Middle era is now
   SIGNIFICANT (0.159, p 0.028) and the Late era is near-zero-positive, not negative.
   Chow at 2012 still sig (F 4.42, p 0.012) but weaker (was F 6.01, p 0.003); 2011 break lost.
   Net: "disappearing advantage" holds as a decline, but "collapsed to negative / complete
   erosion" overstates it now.

4. NARRATIVE - Defensive-coach composite persistence jumped; the offensive-vs-defensive
   composite gap nearly vanished.
   Defensive composite lag-1 0.424 -> 0.511; offensive 0.521 -> 0.537. Gap 0.097 -> 0.026.
   The 4th-down gap still holds strongly (off 0.692 vs def 0.391). So "offensive coaches are
   stickier" is true for 4th down, not really for composite anymore.

5. NARRATIVE - Coordinator-to-HC schematic inheritance got STRONGER.
   Def scheme DC->HC r 0.552 -> 0.616 (p 0.022 -> 0.0084). Shotgun OC->HC 0.493 -> 0.519
   (p 0.001 -> 0.0006). Tempo 0.353 -> 0.359. Aggression 0.131 -> 0.093 (still NS). Reinforces
   "trees transmit systems, not risk tolerance."

6. ORPHAN - Overall aggression-inheritance table (Table 5, N=517) has NO regenerated source.
   No script in run_all_analyses produces a pooled overall aggression inheritance anymore; only
   by-mentor-background and by-coordinator-type are regenerated. Decision needed: regenerate an
   overall pooled version, or drop Table 5 and rely on by-coordinator-type (Table 6).

7. METHODS GAP - new methodology is undescribed: leakage-free train/serve pipeline, stability
   selection (feature counts dropped from 21-28 to 13-16), external WAR reference, coach/
   mentor-clustered SEs + cluster bootstrap CIs, single global BH-FDR. Tables 1 and 2 and the
   "26 features" prose are stale.

8. BH claim mismatch - paper says coordinator-to-HC results "survive BH (p<=0.026)". Those four
   tests (shotgun/tempo/defscheme/aggression coordinator-to-HC) are NOT in the current global BH
   family. Current global BH cutoff is p<=0.0277. Either fold them in or rephrase.

---

## B. Abstract (line 55)

| Claim | OLD | NEW | Status |
|---|---|---|---|
| Plays | 643,290 (2006-2024) | verify (likely unchanged) | OK? |
| Def scheme -> WAR | r=0.152, p=0.010 | r=0.135, p=0.023 (n=282) | MINOR |
| Shotgun -> WAR | r=0.125, p=0.002 | r=0.139, p=0.0006 | MINOR |
| Composite aggression -> WAR | r=0.118, p=0.004 | r=0.172, naive 2.2e-5 / clust 0.001 | NARRATIVE (stronger) |
| Tempo -> WAR | "does not" | r=0.032, p=0.426 | OK |
| Aggression erosion | 0.211 (p=0.003) -> -0.017 (p=0.804) | 0.242 (p=0.0008) -> 0.039 (p=0.565) | NARRATIVE |
| Mentor->protege WAR | r=0.112, p=0.118, n=196 | r=0.112, naive 0.118 / clust 0.127 | OK |
| Shotgun OC inheritance | r=0.375, p<0.001, n=242 | r=0.427, naive 3.9e-12 / clust 0.0 | MINOR (stronger) |
| Shotgun DC inheritance | r=0.033, p=0.584 | r=0.091, naive 0.128 / clust 0.391 | OK (still NS) |
| Coord->HC shotgun | r=0.493, p=0.001 | r=0.519, p=0.0006 | MINOR |
| Coord->HC def scheme | r=0.552, p=0.022 | r=0.616, p=0.0084 | MINOR |
| Coord->HC aggression | r=0.131, p=0.420 | r=0.093, p=0.567 | OK (still NS) |

## C. Methods

| Location | OLD | NEW | Status |
|---|---|---|---|
| 4.1 plays | 643,290 | verify | OK? |
| 4.1 WAR rows | 1,637 | 1,637 (external file) | OK |
| 4.1 merged n | 606 off / 282 def / 123 coaches | 606 / 282 / 123 | OK |
| 4.1 background split | 605: 318 off (52.6%) / 287 def (47.4%) | data is 310 off / 251 def / 45 both (UNCHANGED since before regen; verified across HEAD..HEAD~2). Methods text was already inconsistent with its own data and with Table 12 (310/251). PRE-EXISTING paper bug, not a regen change. Reconcile during edit. | FIX (latent) |
| 4.2 / Table 1 features | hand-curated 26/26/26/21 | stability-selected 15/16/14/16 | METHODS rewrite |
| 4.2 AUCs (prose) | 0.977 / 0.784 / 0.732 (+0.926) | 0.977 / 0.783 / 0.732 / 0.930 | OK |
| 4.3 model selection | "RandomizedSearchCV (100 iter, 3-fold)" | 100 iter, 5-fold (cv changed) | MINOR |
| 4.3 defensive features | "28 features" | stability-selected 13-15 | METHODS rewrite |
| 4.4.1 shotgun | "22-feature ... 638 coach-years" | 14 features; 638 coach-years | METHODS |
| 4.4.2 tempo no-huddle AUC | 0.887 | 0.858 | MINOR (down) |
| 4.4.2 pace | RMSE 11.04 / R2 0.428 | RMSE 11.06 / R2 0.427 | OK |
| 4.4.3 box stacking | RMSE 0.778 / R2 0.467 | RMSE 0.805 / R2 0.428 | MINOR (down) |
| 4.4.3 pass rush | RMSE 0.789 / R2 0.068 | RMSE 0.793 / R2 0.059 | OK |
| 4.4.3 man coverage AUC | 0.710 | 0.702 | OK |
| 4.4.3 def team-years | 288 (2 comp 2016-17, 3 comp 2018+) | 288 total / 282 merged | OK |
| 4.6 Multiple comparison | BH described, "survive ... p<=0.026" | global BH, cutoff p<=0.0277, 98 tests, 61 BH-sig | NARRATIVE |
| NEW methods needed | - | leakage-free pipeline; stability selection; external WAR; clustered SEs + bootstrap CIs | ADD |

### Table 2 (Predictive Model Performance) - current values
| Model | OLD | NEW |
|---|---|---|
| 4th Down (AUC) | 0.977 | 0.977 |
| Run vs Pass (AUC) | 0.784 | 0.783 |
| Pass Target (AUC) | 0.732 | 0.732 |
| Two-Point (AUC) | 0.926 | 0.930 |
| Shotgun (AUC) | 0.864 | 0.820 |
| No-Huddle (AUC) | 0.887 | 0.858 |
| Pace (RMSE / R2) | 11.04 / 0.428 | 11.06 / 0.427 |
| Box Stacking (RMSE / R2) | 0.778 / 0.467 | 0.805 / 0.428 |
| Pass Rush (RMSE / R2) | 0.789 / 0.068 | 0.793 / 0.059 |
| Man Coverage (AUC) | 0.710 | 0.702 |

Feature counts (was -> now): 4th down 26->15, run/pass 26->16, pass target 26->14,
two-point 21->16, shotgun 22->14, no-huddle 26->15, pace 26->15, box 28->15, pass rush 28->13,
man coverage 28->15.

## D. Results tables

### Table 3 (Genes vs WAR overall)
| Gene | OLD r,p | NEW r, naive p, clust p |
|---|---|---|
| Defensive Scheme | 0.152, 0.010 | 0.135, 0.023, (n=282) |
| Shotgun | 0.125, 0.002 | 0.139, 0.0006 |
| Composite Aggression | 0.118, 0.004 | 0.172, 2.2e-5, clust 0.001 |
| Composite Tempo | 0.039, 0.333 | 0.032, 0.426 |
Note: clustered p available for aggression (multiple-regression file). For shotgun/tempo/def
the multiple-regression (cluster-robust) gives: shotgun coef p=0.091, tempo p=0.86, def p=0.044.

### Table 4 (Mentor WAR vs Protege WAR) - all OK (external WAR unchanged)
Overall 0.112 / naive 0.118 / clust 0.127 / n=196; OC->HC 0.087 / 0.390 / n=101;
DC->HC 0.163 / naive 0.119 / clust 0.082 / n=93.

### Table 5 (Aggression inheritance overall, N=517) - ORPHANED, no current source. See A.6.
OLD: Composite 0.050 (0.261); 4th Down 0.150 (0.0006); Pass-Heavy 0.023 (0.596);
Deep Pass 0.072 (0.101); Two-Point 0.085 (0.053).

### Table 6 (Aggression inheritance by coordinator type) - from inheritance_by_type
| Component | OC OLD r,p | OC NEW r, naive p, clust p | DC OLD | DC NEW |
|---|---|---|---|---|
| Composite | 0.079, 0.276 | 0.183, 0.011, clust 0.11 | 0.041, 0.526 | 0.061, 0.347, clust 0.625 |
| 4th Down | 0.191, 0.008 | 0.205, 0.0043, clust 0.185 | -0.016, 0.808 | 0.050, 0.443 |
| Pass-Heavy | 0.200, 0.005 | 0.326, 3.8e-6, clust 0.013 | 0.040, 0.533 | 0.020, 0.758 |
| Deep Pass | 0.062, 0.390 | 0.126, 0.081, clust 0.218 | 0.064, 0.325 | 0.094, 0.145 |
| Two-Point | 0.101, 0.162 | 0.091, 0.209, clust 0.339 | 0.084, 0.193 | 0.076, 0.240 |
n: OC=193, DC=241 (unchanged). KEY: under clustering only OC pass-heavy is significant.

### Table 7 (Shotgun inheritance)
| Relationship | OLD r,p,n | NEW r, naive p, clust p, n |
|---|---|---|
| Overall | 0.245, <0.001, 612 | 0.303, 1.8e-14, clust 0.0, 612 |
| OC->HC | 0.375, <0.001, 242 | 0.427, 3.9e-12, clust 0.0, 242 |
| DC->HC | 0.033, 0.584, 278 | 0.091, 0.128, clust 0.391, 278 |

### Table 8 (Tree summary)
Performance 0.112 No (OK); Composite Aggression 0.050 No (ORPHAN - see Table 5);
4th Down 0.150 Weakly (ORPHAN; by-coord OC 4th down now 0.205 naive-sig but clust NS);
Shotgun OC 0.375 -> 0.427 Strongly.

### Table 9 (Coordinator-to-HC gene inheritance, aggregated_per_coach)
| Gene | OLD r,p,dirret | NEW r,p,dirret |
|---|---|---|
| Def Scheme DC->HC (n=17) | 0.552, 0.022, 71% | 0.616, 0.0084, 64.7% |
| Shotgun OC->HC (n=40) | 0.493, 0.001, 75% | 0.519, 0.0006, 70.0% |
| Tempo OC->HC (n=40) | 0.353, 0.025, 62% | 0.359, 0.023, 60.0% |
| Aggression OC->HC (n=40) | 0.131, 0.420, 60% | 0.093, 0.567, 55.0% |
BH caveat (A.8): these are not in the global BH family; current cutoff is p<=0.0277.

### Table 10 (Aggression vs WAR overall)
| Measure | OLD r,p | NEW r, naive p, clust p, BH |
|---|---|---|
| Composite | 0.118, 0.004 | 0.172, 2.2e-5, clust 0.001, BH-sig |
| Pass-Heavy | 0.107, 0.008 | 0.170, 2.6e-5, clust 0.001, BH-sig |
| 4th Down | 0.085, 0.037 | 0.085, 0.037, clust 0.051, BH NOT sig |
| Deep Pass | 0.020, 0.623 | 0.036, 0.378, clust 0.41 |
| 2-Point | 0.031, 0.444 | 0.042, 0.304, clust 0.369 |
n=606.

### Table 11 (Aggression vs WAR by era)
| Era | Composite OLD | Composite NEW | Pass-Heavy OLD | Pass-Heavy NEW | n |
|---|---|---|---|---|---|
| 2006-2011 | 0.211, 0.003 | 0.242, 0.0008 | 0.108, 0.139 | 0.156, 0.032 | 190 |
| 2012-2017 | 0.102, 0.160 | 0.159, 0.028 | 0.107, 0.139 | 0.161, 0.026 | 192 |
| 2018-2024 | -0.017, 0.804 | 0.039, 0.565 | 0.063, 0.352 | 0.124, 0.064 | 224 |
BH: Middle composite (0.028) and Middle pass-heavy (0.026) now BH-sig; Early pass-heavy (0.032)
raw-sig but BH-lost.

### Temporal robustness (4.5.1)
Interaction beta3: -1.905 (SE 0.862, p 0.029) -> -1.595 (SE 0.824, p 0.0552, NS).
Implied 2006: 29.57 (p 0.012) -> 32.16 (p 0.0040). Implied 2024: -4.73 (p 0.537) -> +3.46 (p 0.662).
Chow best break 2012: F 6.01, p 0.003, 87% drop -> F 4.42, p 0.012, beta 31.99 -> 9.19 (71% drop).
Breaks 2011 (0.006 -> 0.033, BH-lost), 2013 (0.005 -> 0.019).

### Table 12 (Aggression vs WAR by coach type)
| Measure | Off OLD | Off NEW | Def OLD | Def NEW |
|---|---|---|---|---|
| Composite | 0.117, 0.039 | 0.178, 0.0016 | 0.152, 0.016 | 0.195, 0.0020 |
n: Off=310, Def=251 (unchanged). Both stronger; both BH-sig.

### Table 13 (Within-coach fixed effects) + prose
Two sources are conflated in the current table:
- within_coach_fixed_effects (two-way FE): pooled coef 1.166, p 0.0029 (the genuine FE p, ~unchanged).
- effect_sizes_and_power (cluster-robust OLS): the "Effect (1 SD)", "% IQR", "Power" columns.
| Row | OLD effect/%IQR/p/power | NEW effect/%IQR/p/power |
|---|---|---|
| Pooled | 0.314 / 12.5% / 0.003 / 84% | 0.391 / 15.6% / (FE 0.0029; OLS 0.0002) / 96.7% |
| Early | 0.247 / 9.8% / 0.218 / 23% | 0.292 / 11.6% / 0.141 / 31.1% |
| Middle | 0.237 / 9.4% / 0.150 / 30% | 0.286 / 11.4% / 0.081 / 41.5% |
| Late | 0.127 / 5.1% / 0.408 / 13% | 0.245 / 9.8% / 0.114 / 35.1% |
Cohen's d pooled 0.16 -> 0.20. IQR 2.512 -> 2.512. Prose "0.314 games per SD" (Discussion,
Conclusion) -> 0.391. Recommend separating the FE estimate from the OLS effect-size to remove
the conflation.

### Table 14 (Year-to-year persistence, lag 1)
| Component | OLD r / R2 | NEW r / R2 |
|---|---|---|
| Composite | 0.458 / 0.210 | 0.501 / 0.251 |
| 4th Down | 0.556 / 0.309 | 0.564 / 0.318 |
| Pass-Heavy | 0.473 / 0.224 | 0.548 / 0.300 |
| Deep Pass | 0.455 / 0.207 | 0.472 / 0.223 |
| 2-Point | 0.393 / 0.154 | 0.387 / 0.150 |
n=467. All clustered p ~0.0, all BH-sig. Prose "0.39 to 0.56" -> "0.39 to 0.56" (0.387-0.564);
"15-31% variance" -> "15-32%". Lag-2 range "0.23-0.44" still OK; lag-3 "0.28-0.40" -> "0.30-0.41".

### Table 15 (Persistence by coach background, lags 1-3)
Offensive: Composite 0.521/0.460/0.355 -> 0.537/0.448/0.330; 4th Down 0.690/0.619/0.549 ->
0.692/0.604/0.520; Pass-Heavy 0.544/0.422/0.425 -> 0.621/0.483/0.439; Two-Point 0.424/0.417/0.488
-> 0.421/0.397/0.492.
Defensive: Composite 0.424/0.387/0.361 -> 0.511/0.410/0.391; 4th Down 0.373/0.236/0.269 ->
0.391/0.268/0.346; Pass-Heavy 0.448/0.254/0.234 -> 0.516/0.340/0.276; Two-Point 0.425/0.189/0.183
-> 0.415/0.149/0.183 (lag-2 and lag-3 NS clustered).
n unchanged (Off 228/160/110; Def 194/150/113). NARRATIVE: composite off-vs-def gap collapses
(lag1 0.097 -> 0.026); 4th-down gap holds (0.692 vs 0.391).

## E. Discussion / Conclusion

| Claim | OLD | NEW | Status |
|---|---|---|---|
| Erosion 0.211 -> -0.017 | as stated | 0.242 -> 0.039 (+ Middle now sig) | NARRATIVE |
| Mean aggression trend | +0.125%/yr, R2 0.83, p<0.001 | +0.125%/yr, R2 0.829, p 6.3e-8 | OK |
| Variance increase | +23% | +16.7% (composite SD early->late) | MINOR |
| Within-coach effect | 0.314 games/SD, p 0.003, d 0.16 | 0.391 games/SD, d 0.20 | NARRATIVE |
| Scheme vs situational | shotgun OC 0.375; coord shotgun 0.493 / def 0.552 | 0.427; 0.519 / 0.616 | MINOR |
| Aggression -> WAR | 0.118, 0.004 | 0.172, clust 0.001 | NARRATIVE (stronger) |
| Composite inheritance | 0.050, 0.261 | ORPHAN (Table 5) | see A.6 |
| 4th down most heritable | "0.19-0.22 offensive mentors" | 0.218 by-mentor / 0.205 OC (clust NS) | MINOR |
| Mentor WAR | 0.112, 0.118 | 0.112, clust 0.127 | OK |
| Coord->HC survive BH p<=0.026 | as stated | not in BH family; global cutoff 0.0277 | NARRATIVE |
| Conclusion totals | 643,290 plays / 606 coach-yrs / 612 pairs | verify play count; 606; 612 | OK? |
