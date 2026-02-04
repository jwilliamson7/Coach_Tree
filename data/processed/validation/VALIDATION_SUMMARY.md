# NFL Coaching Tree Analysis - Validation Summary

Generated: 2026-02-03

## Overview

This report summarizes the systematic validation of all analyses in the NFL Coaching Tree project.

---

## Phase 1: Data Pipeline Validation

### 1.1 Team Abbreviation Audit (CRITICAL)

**Status: ISSUES FOUND**

**Key Findings:**
- 7 inconsistencies identified across mapping sources
- **Critical**: The `relationships.csv` file uses 3-character truncated team names
  - 34 unique teams, all 3 characters
  - Problematic truncations: 'new' (NYG/NYJ/NWE/NOR), 'los' (LAR/LAC), 'tam', 'kan', 'san'
  - 763 rows have 'new' which is ambiguous across 4 teams
- Las Vegas Raiders: Inconsistent mapping (LVR vs RAI)
- Los Angeles Rams: Inconsistent mapping (LAR vs RAM vs LA)

**Recommendation**:
- Fix truncation logic in `build_coaching_tree.py` (line 130)
- Create single source of truth mapping in `data_constants.py`
- Rebuild `relationships.csv` with proper abbreviations

### 1.2 Coaching Tree Relationship Integrity

**Status: ISSUES FOUND**

**Key Findings:**
- 4,836 total relationships documented
- 536 coaches, 104 years (1922-2025)
- 13 issues identified:
  - 5 orphan coaches (position coaches without expected coordinator parent)
  - 5 interim roles not fully excluded
  - 2 partial relationships (missing expected years)
  - 1 missing known relationship (Belichick-Parcells)

**Sample Relationship Audit:**
- Kyle Shanahan under Mike Shanahan: PARTIAL (1/3 years found)
- Sean McVay under Jay Gruden: VERIFIED
- Matt LaFleur under Sean McVay: VERIFIED
- Bill Belichick under Bill Parcells: MISSING
- Mike Tomlin under Tony Dungy: PARTIAL (1/5 years found)

### 1.3 Play-by-Play Data Coverage

**Status: GOOD**

**Key Findings:**
- 641 coach-year observations
- 138 unique coaches
- 105.4% coverage (more than 32 per year due to mid-season changes)
- 1,124,935 total plays analyzed
- All years (2006-2024) have adequate coverage (32-36 coaches/year)

**Low play count coaches (exploratory):**
- 11 coach-seasons with <30 4th down decisions
- 8 coach-seasons with <200 run/pass plays
- These are primarily interim coaches or partial seasons

---

## Phase 2: Predictive Model Validation

### 2.1 Train/Test Verification

**Status: VERIFIED**

- All 4 models use stratified 80/20 train/test split
- Imputation fit on training data only
- No outcome features included
- Season/week excluded to prevent temporal confounding
- 26 features for 4th down, run/pass, pass target models
- 21 features for two-point model

### 2.2 Temporal Validation

**Status: CANNOT FULLY VALIDATE (data not present)**

- Play-by-play data files not available for re-training
- Recommendation: Re-train on 2006-2019, test on 2020-2024

### 2.3 Model Calibration

**Status: ISSUES FOUND**

**Overall Calibration (from aggression data):**
| Model | Mean Actual | Mean Predicted | Calibration Error |
|-------|-------------|----------------|-------------------|
| 4th Down | 0.1607 | 0.1610 | -0.0003 |
| Run/Pass | 0.6110 | 0.5917 | +0.0192 |
| Pass Target | 0.3736 | 0.3804 | -0.0068 |
| Two-Point | 0.0727 | 0.0712 | +0.0015 |

**CRITICAL: Temporal Drift in 4th Down Model**
- Year-error correlation: r = 0.732 (significant temporal drift)
- Early years (2006-2017): Model over-predicts go-for-it rate
- Recent years (2018-2024): Model under-predicts go-for-it rate
- 2024 error: +0.0308 (actual 22.4% vs predicted 19.4%)

**Interpretation**: Teams have become significantly more aggressive on 4th down over time. The model trained on all years doesn't fully capture this trend.

**Recommendation**: Consider time-varying model or rolling window approach.

---

## Phase 3: Statistical Analysis Validation

### 3.1 Assumption Testing

**Status: ISSUES FOUND (but not critical)**

**Normality Tests (Shapiro-Wilk):**
- All 5 aggression variables: NOT NORMAL
- Composite aggression: Skewness=0.47, Kurtosis=1.34 (minor deviation)
- Fourth down aggression: Skewness=1.16, Kurtosis=3.55 (moderate skew)
- Recommendation: Use robust/non-parametric methods for confirmation

**Homogeneity of Variance (Levene's):**
- Aggression across eras: HETEROGENEOUS (p=0.003)
- Variance ratio: 1.50 (Late era has 50% more variance than Early)
- Recommendation: Use Welch's correction for era comparisons

**Autocorrelation (Durbin-Watson):**
- Yearly means: DW=0.36 (strong positive autocorrelation)
- Lag-1 correlation: r=0.81
- Recommendation: Use clustered standard errors or AR models

**Outliers:**
- 15 coach-seasons (2.3%) are IQR outliers
- 8 coach-seasons have |z| > 3
- Notable outliers: Jim Tomsula, John Fox, Eric Studesville
- Recommendation: Check influential observations in key analyses

### 3.2 Multiple Comparison Corrections (Benjamini-Hochberg)

**Status: PROPERLY APPLIED**

**Summary:**
- 98 total hypothesis tests across all analyses
- 63 significant at raw p < 0.05
- 59 significant after BH correction
- 4 tests lost significance after correction:
  1. Coach Type by Era: Offensive Early (p=0.031)
  2. Overall WAR: Composite Aggression (p=0.032)
  3. Coach Type by Era: Defensive Early (p=0.042)
  4. Coach Type (Overall): Defensive Pass-Heavy (p=0.043)

**Key findings that remain significant:**
- Temporal trend in aggression (p < 0.001)
- Persistence correlations (all p < 0.001)
- OC->HC inheritance for 4th down and pass-heavy (p < 0.01)
- Two-way fixed effects pooled (p = 0.013)

### 3.3 Sample Size Assessment

**Status: ADEQUATE FOR MAIN ANALYSES**

| Analysis | N | Unit | Status |
|----------|---|------|--------|
| Overall | 641 | coach-years | OK |
| Early Era | 204 | coach-years | OK |
| Middle Era | 199 | coach-years | OK |
| Late Era | 238 | coach-years | OK |
| Offensive coaches | 310 | coach-years | OK |
| Defensive coaches | 251 | coach-years | OK |
| Both backgrounds | 45 | coach-years | OK |

**Exploratory (underpowered):**
- Both x Early: N=12
- Both x Middle: N=15
- Both x Late: N=18

### 3.4 Fixed Effects Model Validation

**Status: PARTIALLY VALID**

**Verified:**
- Clustered SE calculation: CORRECT
- t-statistic computation: CORRECT
- p-value degrees of freedom: CORRECT
- Era comparison SE of difference: CORRECT

**Issue Found:**
- Two-way demeaning: Coach and year means not perfectly equal after demeaning
- Variance of coach means after demeaning: 4.61e-05 (should be ~0)
- Variance of year means after demeaning: 3.36e-05 (should be ~0)

**Note**: This is likely due to the iterative nature of two-way demeaning. The current implementation uses single-pass demeaning which may not fully converge. However, the deviation is small relative to the original variance.

---

## Key Recommendations

### Critical (Should Address)

1. **Team Abbreviation Mapping**: Create unified mapping and rebuild relationships.csv
2. **4th Down Model Temporal Drift**: Document this limitation clearly in the paper; consider era-specific models

### Important (Should Consider)

3. **Missing Belichick-Parcells relationship**: Investigate why this known relationship wasn't captured
4. **Use Robust Methods**: Given non-normality and heterogeneity, report non-parametric alternatives alongside parametric tests
5. **Note BH Corrections**: The "Overall WAR: Composite Aggression" finding loses significance after BH correction (p=0.032 vs threshold 0.031)

### Minor (For Completeness)

6. **Document Exploratory Analyses**: Flag "Both" coach type analyses as exploratory due to small N
7. **Clustered Standard Errors**: Already implemented correctly
8. **Two-Way FE Iteration**: Consider iterative demeaning for perfect convergence

---

## Validation Scripts Created

All validation scripts are in `scripts/validation/`:

1. `validate_team_abbreviations.py` - Team mapping audit
2. `validate_relationships.py` - Coaching tree integrity
3. `validate_pbp_coverage.py` - Play-by-play coverage
4. `validate_models_temporal.py` - Temporal holdout design
5. `validate_calibration.py` - Model calibration
6. `validate_assumptions.py` - Statistical assumptions
7. `validate_sample_sizes.py` - Sample size documentation
8. `validate_fixed_effects.py` - Fixed effects implementation

---

## Phase 4: Results Consistency Validation (COMPLETED)

### 4.1 Figure-to-Analysis Reconciliation

**Status: VERIFIED**

All figures match their source analysis JSON files:
- aggression_war_analysis.png ↔ aggression_war_regression_results.json ✓
- aggression_war_by_era.png ↔ aggression_war_temporal_analysis.json ✓
- aggression_persistence_analysis.png ↔ aggression_persistence_results.json ✓
- inheritance_by_mentor_type.png ↔ inheritance_by_type_results.json ✓
- aggression_trends.png ↔ aggression_temporal_trend_results.json ✓

### 4.2 LaTeX Report Accuracy

**Status: REQUIRES UPDATE**

Most statistics verified correct:
- Overall correlations: ✓ Verified
- Era correlations: ✓ Verified
- Persistence values: ✓ Verified
- Fixed effects (from effect_sizes_and_power_results.json): ✓ Verified
- BH correction counts (98 tests, 63 raw, 59 BH): ✓ Verified

**ISSUE FOUND: Inheritance analysis values need updating**

After fixing team abbreviation mapping and rebuilding relationships.csv, the inheritance analysis results changed:

| Metric | Paper Value | Updated Value |
|--------|-------------|---------------|
| Offensive mentors 4th Down r | 0.232 | 0.217 |
| Offensive mentors n | 284 | 257 |
| OC→HC 4th Down r | 0.201 | 0.191 |
| OC→HC Pass-Heavy r | 0.239 | 0.200 |
| OC→HC n | 204 | 193 |

**Note**: Core conclusions remain valid - offensive coaches still show significant inheritance for 4th down and pass-heavy aggression, but with slightly smaller effect sizes.

---

## Team Abbreviation Fix (COMPLETED)

**Date: 2026-02-03**

Added `standardize_team_abbreviation()` function to `data_constants.py` with:
- Year-based franchise logic (BAL, HOU, STL)
- Full team name to PFR abbreviation mapping (50 teams)
- Handles all 32 current NFL franchises for 2006+ data

**Note**: Mapping optimized for 2006+ play-by-play data era. Pre-2006 historical teams may not be fully mapped.

---

## Final Recommendations

### Critical (Paper Update Required)

1. **Update inheritance analysis values** in LaTeX report to reflect corrected team mapping

### Important (Methodology Notes)

2. **4th Down Model Temporal Drift**: Document calibration drift (r=0.732 year-error correlation)
3. **Use Robust Methods**: Report non-parametric alternatives for non-normal data
4. **Note BH Corrections**: Flag that "Overall WAR: Composite Aggression" loses significance after correction

### Completed

- [x] Team abbreviation mapping fixed (2006+ era)
- [x] Relationships.csv rebuilt with proper abbreviations
- [x] Figure-analysis reconciliation verified
- [x] Inheritance analysis re-run with corrected data
