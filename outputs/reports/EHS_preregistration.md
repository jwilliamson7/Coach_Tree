# Preregistration: Heritability and selection of strategic traits in the NFL coaching tree

Preregistration of new analyses of an existing dataset; draft for OSF. Drafted 2026-06-30 by Jon
Williamson (Independent Researcher).

---

## 0. Study type and honest-disclosure statement

This work began as exploratory analysis of an existing dataset. As it matured into a set of
confirmatory hypotheses, prompted by Mesoudi's (2020) study "Cultural evolution of football tactics:
strategic social learning in managers' choice of formation," which names lineage transmission along
mentor-protege ties as open work, I paused before running those confirmatory tests to register them
here. That paper was itself preregistered (osf.io/er4dx). The account that follows states what the
preliminary work had established, so hindsight cannot blur the confirmatory line.

This preregistration covers new analyses of a dataset that already exists and that we have already
explored in a preliminary, proof-of-concept phase. Integrity requires stating what that preliminary
work established, so the confirmatory predictions below are understood as motivated by it but not yet
formally tested.

The preliminary phase built the four composite coaching genes (offensive aggression, shotgun, tempo,
defensive aggression) and their sub-components for 2006-2024, with the defensive-aggression genes from
2016 and man coverage from 2018, and examined as a proof of concept how they transmit from coordinator
to head coach and how they relate to Coach WAR (wins above replacement, not raw win percentage; the
selection index S). That work suggested a qualitative pattern: the schematic-identity traits (shotgun,
tempo) appeared to transmit from coordinator to head coach more strongly than they predicted winning,
whereas the approach-to-scenarios traits (offensive and defensive aggression) appeared to predict Coach
WAR more strongly than they transmitted. Those impressions motivated the hypotheses below. The
confirmatory analyses re-estimate every quantity with the frozen, leakage-free pipeline; no preliminary
point estimate is carried forward.

Three analyses are novel and preregistered here. C1 estimates transmission as a variance-components
heritability (h^2) through a Bayesian multilevel model with era at its own level, in place of a raw
correlation. C2 estimates per-trait repeatability as a variance-components ICC, tabulated as the h^2
ceiling. C3 extends h^2 and S from the four composites to the ten sub-traits and tests the cross-trait
h^2-versus-S relationship. None of the confirmatory quantities has been computed on the frozen
pipeline.

We commit to not inspecting any C1-C3 output until this document is posted.

---

## 1. Confirmatory hypotheses

Each trait is a z-scored, head-coach-season phenotype. The offensive, shotgun, and tempo genes are
measured on the plays of the coach's own offense; the defensive-aggression genes are measured on the
defending team's plays and attributed to that team's head coach. Every play is assigned to the coach
who led the team for that game, so a within-season head-coaching change splits the season between the
two coaches at the game of the change. In the coordinator-to-head-coach transmission tests, a coach's
coordinator-era value is the
corresponding team gene for the team and seasons in which they held that coordinator role. All inference
is contemporary-group (within-season) adjusted so that shared league-wide drift is not counted as
transmission or selection (Section 3). h^2 is the between-lineage variance share (transmission), and S
is the selection index, a trait's correlation with Coach WAR.

H1 (approach is selected, identity is inherited). Coaching traits split into two families that behave
differently. The approach-to-scenarios traits, offensive aggression (going for it on fourth down,
pass-heavy play-calling, targeting downfield, two-point tries) and defensive aggression (loading the
box, sending extra rushers, playing man), are more strongly selected, meaning they predict Coach WAR,
than the schematic-identity traits. The schematic-identity traits, shotgun and tempo, are more strongly
inherited, meaning they transmit down the tree, than the approach traits, and their apparent link to
winning disappears under contemporary-group adjustment.

H2 (repeatability is not heritability). Offensive aggression is a repeatable individual trait that is
nonetheless not heritable; its within-coach repeatability clearly exceeds its heritability.

H3 (cross-trait relationship). Across the ten sub-traits, heritability and selection are negatively
related; the traits that transmit most strongly are not the ones that predict winning. An orthogonal or
null relationship would be weaker, partial support.

---

## 2. Frozen trait set

The primary cross-trait test uses ten sub-traits, frozen here.

| Family | Sub-trait | Transition tested | Data window |
|---|---|---|---|
| Offensive aggression | fourth_down | OC->HC | 2006-2024 |
| Offensive aggression | pass_heavy | OC->HC | 2006-2024 |
| Offensive aggression | deep_pass | OC->HC | 2006-2024 |
| Offensive aggression | two_point | OC->HC | 2006-2024 |
| Tempo | no_huddle | OC->HC | 2006-2024 |
| Tempo | pace | OC->HC | 2006-2024 |
| Defensive aggression | box_stacking | DC->HC | 2016-2024 |
| Defensive aggression | pass_rush | DC->HC | 2016-2024 |
| Defensive aggression | man_coverage | DC->HC | 2018-2024 |
| Schematic | shotgun | OC->HC | 2006-2024 |

Three composites, offensive aggression, tempo, and defensive aggression, appear on the h^2-versus-S map
as labeled exemplars but are not counted as independent points in the cross-trait test; shotgun is
already a single sub-trait.

All ten sub-traits enter the primary cross-trait test. As a pre-specified sensitivity check, the test is
re-run excluding sub-traits with below-median reliability, and the conclusion should be qualitatively
stable.

Each coach contributes at most one transition per coordinator role: the most recent coordinator stint
that precedes a head-coaching stint, paired with the first head-coaching stint beginning after it.
Coaches with more than one coordinator-then-head-coach arc are represented by their most recent arc,
which avoids pseudo-replication.

---

## 3. Analysis plan (frozen)

Analyses are in Python, with the Bayesian models in PyMC to keep the whole pipeline in one language and
a single-command reproducible bundle. Frequentist REML (statsmodels) is reported only as a sanity
cross-check.

Every trait is contemporary-group adjusted (within-season demeaning) before modeling, and era
additionally enters the multilevel models as its own variance level, so lineage variance is estimated
net of era. Coach WAR is already a within-season, replacement-relative phenotype, with a between-season
variance share of about 3%, and receives no further era adjustment.

Each gene is an estimate with a known per-coach-season sampling variance, from a reliability
decomposition applied to every sub-trait. Because unmodeled
noise attenuates variance-component ratios, and does so unevenly across traits of differing
reliability, C1 and C2 carry this sampling variance as a fixed observation-level measurement variance,
so that h^2 and repeatability are estimated for the latent trait rather than its noisy observation.
Where a per-observation variance is not cleanly estimable for a sub-trait, the estimate is
disattenuated by its reliability and both raw and disattenuated values are reported. Reliability is
estimated and reported for every sub-trait. Selection S is inverse-variance weighted, so the
cross-trait test compares h^2 and S on the
same disattenuated footing.

For each trait, C1 fits a Bayesian multilevel model with a mentor/lineage random intercept and an era
level, and defines h^2 as sigma^2_lineage / (sigma^2_lineage + sigma^2_residual), reported net of era
as a posterior median with a 95% HDI. Priors are half-Normal(0, 1) on all standard-deviation
components (traits are z-scored) and Normal(0, 1) on the fixed intercept. Sampling uses 4 chains with
2000 post-warmup draws each; convergence is checked with the standard diagnostics, treating R-hat at or
below 1.01 and bulk effective sample size above 400 as the target, using a non-centered
parameterization with additional draws for variance components near the boundary. The multilevel
estimate is cross-checked against a classical parent-offspring dyadic regression, an era-adjusted
regression of protege trait on mentor trait; the two are reported side by side and their agreement is
the evidence. As an assumption-dependent robustness check, a pedigree (relatedness-matrix) animal model
is also fit, recognizing that coaching relatedness can only be assumed by analogy rather than derived
from a Mendelian process.

Selection S is the era-adjusted, inverse-variance-weighted correlation of a trait with Coach WAR.
Sub-trait S is computed with the identical estimator already used for the composites and the
defensive-aggression components, a mechanical extension pre-specified here.

C2 estimates a variance-components ICC (coach random effect) per trait, net of era, as a posterior
median with a 95% HDI, tabulated against h^2. As an ordering guard, h^2 should not exceed repeatability
for any trait, and a violation flags a modeling error to fix before interpretation.

C3 assembles the (h^2, S) pair for the ten sub-traits and computes the Spearman rank correlation
between them with a 95% bootstrap interval. As a robustness check it is recomputed excluding sub-traits
with below-median reliability. The ten sub-traits index distinct on-field behaviors, for example
fourth-down decisions versus deep targeting, rather than re-scalings of one measure, and the full
inter-sub-trait correlation matrix is reported alongside the result so readers can judge their
independence.

The decision rules are one per hypothesis. H1 is supported if, comparing the approach family (offensive
aggression, defensive aggression) with the identity family (shotgun, tempo), the approach family's mean
S exceeds the identity family's and the identity family's mean h^2 exceeds the approach family's, each
difference with a 95% interval excluding 0. H2 is supported if the quantity (aggression repeatability
minus aggression h^2) has a 95% HDI excluding 0. H3 is supported if the Spearman rank correlation
between h^2 and S across the sub-traits is negative with a 95% bootstrap interval excluding 0.

Confirmatory results are reported with intervals, not bright-line p-values.

---

## 4. Confirmatory vs exploratory boundary

The confirmatory core is C1 (per-trait h^2), C2 (repeatability versus h^2), C3 (the cross-trait
h^2-versus-S relationship), and the H1 approach-versus-identity family contrast. The genes feeding
these tests come from the leakage-free, coach-held-out pipeline.

The following analyses are declared exploratory, reported with confidence intervals or permutation
p-values, and held outside the confirmatory family under a single global Benjamini-Hochberg FDR. The
first is an era-residualized Moran's I on the mentor network with a within-era permutation null, taking
node era as career midpoint and using at least 1000 within-era shuffles. The second is a Price
decomposition per trait, with the selection term as the within-generation covariance of fitness and
trait and the transmission term as the mentor-to-protege trait change, carrying a
vertical-versus-horizontal caveat bounded by the era-residualization above. The third is reproductive
fitness as an exposure-normalized rate, offspring per coordinator-season with a minimum-exposure floor,
together with the test of whether quality predicts that rate net of exposure.

---

## 5. Robustness / leave-out (pre-specified)

The confirmatory conclusions are re-run under two leave-outs and reported as sensitivity analyses:
dropping shared-mentor protege pairs, and holding out the most recent coaching generation. The
direction of the cross-trait relationship is expected to be stable under both. Because these subsamples
are small, a sign change is read as a bound on robustness rather than a falsification of the main
result.

---

## 6. Data and code availability

Everything is public, seed-fixed, and runnable end-to-end: this repository plus the sibling Coach_WAR
repository, which is the source of the WAR phenotype. Selection S is computed on Coach WAR at commit
dffe2f1 (2026-06-18). If the WAR estimator is revised, for example in response to methodological
review, S is recomputed on the revised version as a pre-specified robustness check and both are
reported; the confirmatory predictions concern the sign of the contrasts, not the exact value of S.

---

## 7. Limitations acknowledged up front

The transmission samples are small, about 40 for OC->HC and about 17 for DC->HC, so the heritability
posteriors will be wide; the main result rests on the cross-trait pattern (H3) and on agreement across
estimators, not on any single precise h^2. Coaching pedigrees are also non-tree, with multiple
simultaneous mentors and overlapping careers, which is why the lineage random effect, not the pedigree,
is primary. Genes attribute team play-calling to the head coach and do not isolate a coordinator's
individual contribution within a staff; the coordinator-to-head-coach tests address transmission by
comparing each coach's coordinator-era team gene with their head-coach-era gene.
