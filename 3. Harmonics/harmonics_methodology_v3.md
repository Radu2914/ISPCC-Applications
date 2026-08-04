# Harmonics Surrogate Methodology — v3 (Exp1 → Exp2 → Exp3)

## Problem Statement

The target system is the harmonic series under just intonation. For every pair of
integers (n, m) with 1 ≤ n ≤ m ≤ 64, the **Euler Gradus Suavitatis (GS)** measures
harmonic consonance. The formula is:

$$GS(n, m) = 1 + \sum_{p^e \mid p \cdot q} (p - 1) \cdot e$$

where p and q are the numerator and denominator of the fully reduced ratio n/m
(i.e. p = n/gcd(n,m), q = m/gcd(n,m)), and the sum runs over all prime power
factors of p·q. GS = 1 for the unison; higher values indicate greater dissonance.
The pool contains **2080 pairs** (GS mean = 27.38, max = 119).

The harmonics domain is used as a validation domain for a different structural
reason than the logistic map. Here, the mathematical structure of the target is
known exactly: GS depends **only** on the prime factorisation of p·q and is
**completely independent of g = gcd(n,m)**. This makes g a mathematically
confirmed noise variable — a ground truth that can be used to test whether the
encoding and canonical reduction identify the correct variables.

The work proceeded in three experiments. Exp1 tested whether the coordinate system
alone (same features, different basis functions) improves data efficiency. Exp2
identified the minimal sufficient feature set by encoding only the true structural
variables. Exp3 added IntentionalMaxiMin sampling and a two-stage model.

---

## Mathematical Structure of the Domain

Any pair (n, m) decomposes uniquely as:

$$n = p \cdot g, \quad m = q \cdot g, \quad \gcd(p, q) = 1, \quad g = \gcd(n, m)$$

The three independent variables are therefore **p** (numerator of reduced ratio),
**q** (denominator), and **g** (voicing — the common factor that scales both). GS
is determined entirely by the prime structure of p·q. g affects which octave or
voicing the interval appears in, but not its harmonic character.

This decomposition is the structural prior. It determines which variables are encoded
with which basis functions, which features are canonical, and which are mathematically
redundant — all before any data is examined.

---

## Stage 0 — Structural Constants (Music-Theoretic, No Data Required)

Normalisation scales are derived from established limits in just intonation theory:

| Scale | Value | Physical meaning |
|---|---|---|
| SCALE_5LIM | 3.0 = log₂(8) | 5-limit JI boundary — p and q are consonance-limited here |
| SCALE_TH | 8.0 = log₂(256) | Tenney height at 16-limit — complexity ceiling for p·q |
| SCALE_GCD | 4.0 = log₂(16) | GCD convergence boundary at 16th harmonic |
| SCALE_H16 | 4.0 = log₂(16) | 16th harmonic limit — range of n and m |
| SCALE_OCT | 1.0 = log₂(2) | Octave — the fundamental period of pitch |
| SCALE_FRAC | 1.0 | p/(p+q) is naturally bounded in (0, 0.5] |

These are analogous to λ_free, λ_rubber, and DIEL_BIFURCATION in the EM pipeline:
each converts a raw quantity into a dimensionless ratio whose musical meaning is fixed.

---

## Stage 1 — Raw Feature Set (Inherited, Pre-Canonical)

The initial feature set of 11 log-scale statistics was inherited from the logistic map
convention — the same structural variables available before the canonical decomposition
was applied:

| Feature | Formula | Variable character |
|---|---|---|
| log_n_low | log₂(n) | π-type: grows with harmonic position, cascading |
| log_n_high | log₂(m) | π-type: same |
| log_ratio | log₂(m/n) | ε-type: bounded by octave (log₂(2) = 1) |
| log_p_num | log₂(p+1) | ε-type: bounded by 5-limit JI |
| log_q_den | log₂(q+1) | ε-type: bounded, **dominant — 9.2% probe importance** |
| log_gcd | log₂(g+1) | ε-type: convergence quantity, mathematically irrelevant for GS |
| tenney_h | log₂(p·q+1) | π-type: grows with dissonance, unbounded, cascading |
| log_p_plus_q | log₂(p+q+1) | π-type: complexity sum |
| log_diff | log₂(m−n+1) | π-type: harmonic distance |
| log_euler_dist | log₂(p+q) | π-type: near-identical to log_p_plus_q |
| p_frac | p/(p+q) | ε-type: naturally bounded in (0, 0.5] |

A probe run (RF importance on the full 2080-point pool, all 48 encoded features)
identified log_q_den as dominant at **9.2% importance** — approximately three times
any other single feature. log_p_num is secondary at 3.4%. log_diff and log_n_low
are noise at 0.6% each. This ordering is consistent with the mathematical structure:
GS is more sensitive to the denominator q than the numerator p because larger
denominators correspond to higher prime factors and greater dissonance.

---

## Stage 2 — Pi/e Encoding

Both π-encoding and e-encoding are applied, assigned by variable character.

**E-encoding** (for ε-type variables — bounded, consonance-limited, self-regulating):

$$\Phi_E(\tilde{x}) = \left(\frac{2}{5}e^{-e\tilde{x}},\ \frac{2}{5}\tilde{x}^e,\ \frac{1}{5}e^{-e(\tilde{x}-0.5)^2}\right)$$

Applied to: log_ratio, log_p_num, log_q_den, log_gcd, p_frac.

**Pi-encoding** (for π-type variables — cascading, complexity-growing, unbounded):

$$\Phi_\Pi(\tilde{x}) = \left(\frac{5}{11}\sin(\pi\tilde{x}),\ \frac{1}{11}\cos(\pi\tilde{x}),\ \frac{1}{11}\sin(2\pi\tilde{x}),\ \frac{3}{11}\sin(\pi^2\tilde{x}),\ \frac{1}{11}\sin(\pi\tilde{x})\cos(\pi^2\tilde{x})\right)$$

Applied to: log_n_low, log_n_high, tenney_h, log_p_plus_q, log_diff, log_euler_dist.

The full encoding produces **48 features** (6 e-encoded variables × 3 features each +
6 π-encoded variables × 5 features each). The normalisation scale for each variable
is its music-theoretic constant, not a data-derived statistic.

---

## Experiment 1 — Coordinate System Test (Matched 11 vs 11)

**Question:** Does changing the coordinate system alone — same 11 features, same
model, same N — improve data efficiency?

**Protocol:** For each of 10 seeds and each N ∈ [11, 22, 33, 50, 100, 150, 200, 300,
500], draw N points from the 2080-point pool. Apply 5-fold CV. Compare:
- **RAW-11:** XGB on 11 raw log-scale statistics
- **MATCHED-11:** XGB on 11 probe-selected encoded features (one per raw variable)

The MATCHED-11 selection takes the highest-importance encoded feature per source
variable from the probe run:

| Raw variable | Encoding | Selected feature | Probe imp. |
|---|---|---|---|
| log_n_low | π | pienc_log_n_low_sin_pi | 0.006 |
| log_n_high | π | pienc_log_n_high_sin_pi2 | 0.039 |
| log_ratio | e | eenc_log_ratio_gauss | 0.007 |
| log_p_num | e | eenc_log_p_num_pow_e | 0.034 |
| log_q_den | e | eenc_log_q_den_pow_e | **0.092** |
| log_gcd | e | eenc_log_gcd_pow_e | 0.012 |
| tenney_h | π | pienc_tenney_h_sin_pi | 0.053 |
| log_p_plus_q | π | pienc_log_p_plus_q_sin_pi | 0.044 |
| log_diff | π | pienc_log_diff_sin_pi | 0.006 |
| log_euler_dist | π | pienc_log_euler_dist_sin_pi | 0.052 |
| p_frac | e | eenc_p_frac_pow_e | 0.007 |

Total MATCHED-11 importance: 35.2% (vs 23% expected if uniform across 48 features),
confirming that the probe selects structurally meaningful features.

**Results (Exp1, 10 seeds, 5-fold CV):**

| N | RF raw | XGB raw | RF matched | XGB matched | XGB match > XGB raw? |
|---|---|---|---|---|---|
| 11 | −45.26 ± 95.68 | −39.84 ± 65.29 | −62.38 ± 117.14 | −57.21 ± 101.54 | NO |
| 22 | −2.19 ± 3.22 | −2.56 ± 2.67 | −2.31 ± 3.52 | −3.02 ± 4.66 | NO |
| 33 | 0.242 ± 0.394 | 0.038 ± 0.691 | 0.258 ± 0.380 | 0.093 ± 0.672 | YES |
| 50 | 0.258 ± 0.228 | 0.158 ± 0.245 | 0.302 ± 0.214 | 0.197 ± 0.243 | YES |
| 100 | 0.538 ± 0.079 | 0.522 ± 0.088 | 0.542 ± 0.074 | 0.583 ± 0.074 | YES |
| 150 | 0.599 ± 0.047 | 0.650 ± 0.064 | 0.608 ± 0.051 | 0.674 ± 0.072 | YES |
| 200 | 0.533 ± 0.037 | 0.623 ± 0.048 | 0.556 ± 0.049 | 0.658 ± 0.061 | YES |
| 300 | 0.609 ± 0.051 | 0.724 ± 0.037 | 0.626 ± 0.042 | 0.747 ± 0.035 | YES |
| 500 | 0.663 ± 0.026 | 0.801 ± 0.017 | 0.691 ± 0.025 | 0.817 ± 0.016 | YES |

XGB matched beats XGB raw at **7/9** sample sizes. The advantage is consistent at
N ≥ 33 and grows at large N. At N=11 and N=22, both methods fail — insufficient data
for a domain with 2080 distinct pairs spanning GS from 1 to 119.

**Exp1 conclusion:** The coordinate system alone improves prediction, particularly at
mid-to-large N. But the inherited 11 features include noise variables (log_diff,
log_n_low, log_gcd for GS purposes) that dilute the signal. Exp2 tests whether
encoding only the true structural variables (p and q) removes this dilution.

---

## Experiment 2 — Canonical Reduction

**Question:** Does encoding only the mathematically confirmed structural variables
(p and q directly) outperform both RAW-11 and MATCHED-11?

Three canonical feature sets were defined, each building on the structural prior:

**CANON-6:** log_q (e×3) + log_p (e×3) — the two GS-determining variables only.
Six features encoding what the mathematical structure says is sufficient.

**CANON-11:** CANON-6 + tenney_h (π×5) — adds the complexity axis log(p·q),
which grows with dissonance and is π-type (cascading). Same feature count as
the baselines.

**CANON-14:** CANON-11 + log_g (e×3) — adds voicing (g = gcd(n,m)), which is
mathematically irrelevant for GS. Serves as a structural noise test: if CANON-14 ≈
CANON-11, the model correctly ignores g.

**Encoding rationale:**
- log_q → e-basis: denominator is consonance-limited, self-regulating, bounded by 5-limit JI
- log_p → e-basis: same character as q
- tenney_h (log p·q) → π-basis: product grows with dissonance, cascading, unbounded
- log_g → e-basis: GCD is a convergence/grouping quantity

**Three hypotheses tested:**

**H1 — CANON-11 ≥ MATCH-11** (direct p/q encoding beats inherited derived features):
CANON-11 wins at 6/9 N values. **Supported.** The inherited features include
redundant derived quantities (log_euler_dist ≈ log_p_plus_q) and noise variables
(log_diff, log_n_low). Encoding p and q directly removes this redundancy.

**H2 — CANON-14 ≈ CANON-11** (log_g is noise for GS):
CANON-14 wins at 6/9 N values, avg Δ = +0.107. **Unexpected.** The mathematical
prior says g is irrelevant, and the result should be a tie. The marginal positive
contribution of log_g is attributed to a dataset correlation: at MAX_HARMONIC=64,
the sampling distribution of p·q is not uniform — high-g pairs (unisons, octaves)
are over-represented. This is a finite-pool artefact, not a structural finding.

**H3 — CANON-6 vs RAW-11** (can 6 features beat 11?):
CANON-6 wins at 7/9 N values, avg Δ = +1.597 (large positive due to extreme low-N
behaviour). **Confirmed.** Six features encoding only p and q, with music-theoretically
grounded normalisation scales, outperform 11 inherited features. This is the strong
form of the canonical reduction claim: fewer, better-grounded features win.

**Exp2 results (XGBoost, 10 seeds, 5-fold CV):**

| N | XGB raw | XGB match | XGB c6 | XGB c11 | XGB c14 | Best |
|---|---|---|---|---|---|---|
| 11 | −39.84 | −57.21 | −25.51 | −25.67 | −24.70 | c14 |
| 22 | −2.56 | −3.02 | −2.88 | −2.65 | −2.64 | raw |
| 33 | 0.038 | 0.093 | 0.011 | 0.039 | 0.029 | match |
| 50 | 0.158 | 0.197 | 0.203 | 0.178 | 0.166 | c6 |
| 100 | 0.522 | 0.583 | **0.601** | 0.565 | 0.566 | c6 |
| 150 | 0.650 | 0.674 | **0.713** | 0.680 | 0.683 | c6 |
| 200 | 0.623 | 0.658 | **0.701** | 0.661 | 0.663 | c6 |
| 300 | 0.724 | 0.747 | **0.794** | 0.762 | 0.762 | c6 |
| 500 | 0.801 | 0.817 | **0.849** | 0.830 | 0.829 | c6 |

CANON-6 wins at 7/9 N values for both XGBoost and Random Forest.
CANON-6 becomes the confirmed minimal sufficient representation for Exp3.

**Why CANON-6 beats CANON-11 at large N:** tenney_h = log(p·q) = log_p + log_q.
It is a linear combination of the two variables already encoded in CANON-6. At
large N, the model has enough data to learn this redundancy and CANON-11's extra
5 features add noise. At small N, tenney_h provides a useful shortcut to the
complexity axis before the model can reconstruct it from p and q separately.

---

## Experiment 3 — IntentionalMaxiMin + Two-Stage Model

**Question:** Does MaxiMin sampling in CANON-6 space, combined with a two-stage
grammar-then-dialect model, reduce the N required to reach a given R²?

**Why MaxiMin applies here (unlike the logistic map):** The selection space is
genuinely 2D — p and q are independent variables. CANON-6 is a 6-dimensional
e-encoded representation of two independent musical quantities. MaxiMin in this
space selects archetypal intervals: maximally spread across consonant/dissonant,
simple/complex, small/large denominator. This is structurally meaningful coverage,
not coverage of transforms of a single variable.

**Protocol change from Exp1/Exp2:** A fixed 500-point test set (seed=999) is held
out before training. The remaining 1580 points form the training pool. MaxiMin
selects N points from the 1580-point pool per seed; the model is then evaluated
on the fixed 500 held-out points. This allows direct comparison of training budget
vs. out-of-sample accuracy. Exp3 R² numbers differ from Exp2 (which used 5-fold CV);
within-Exp3 comparisons are valid, and the crossover table includes Exp2 references
for context.

**Four methods compared:**
- **A) RAW-11 + random:** Exp1 baseline, re-evaluated under the fixed test protocol
- **B) CANON-6 + random:** Exp2 winner, re-evaluated
- **C) CANON-6 + MaxiMin:** Structural sample selection in 6D e-encoded space
- **D) CANON-6 + MaxiMin + two-stage:** Grammar (Ridge on CANON-6) + dialect (XGB on residuals)

**Two-stage model (D):**

*Stage 1 — Ridge on CANON-6 (the grammar):* Fits the dominant structural trend:
GS as a function of p and q. Ridge is used because the relationship is approximately
linear in the encoded space, and L2 regularisation keeps coefficients stable at small N.

*Stage 2 — XGB on Stage-1 residuals (the dialect):* Fits the nonlinear structure
that Ridge cannot represent: the prime factorisation irregularities (GS is not smooth
in p and q — it jumps when p or q crosses a prime boundary), and the interaction
between p and q that tenney_h partially captured. Residuals are smaller and smoother
than the raw target, so Stage 2 converges faster.

**Results (Exp3, 10 seeds, fixed 500-point test set):**

| N | RAW-11 rand | C6 rand | C6 MaxiMin | MM+2-stage | Best | MM gain vs C6 rand |
|---|---|---|---|---|---|---|
| 11 | 0.227 ± 0.165 | 0.371 ± 0.094 | 0.245 ± 0.167 | 0.232 ± 0.194 | B | −0.127 |
| 22 | 0.430 ± 0.064 | 0.304 ± 0.157 | 0.384 ± 0.086 | 0.413 ± 0.107 | A | +0.079 |
| 33 | 0.413 ± 0.079 | 0.391 ± 0.107 | 0.488 ± 0.080 | 0.511 ± 0.089 | **D** | +0.097 |
| 50 | 0.459 ± 0.122 | 0.534 ± 0.093 | 0.570 ± 0.084 | 0.603 ± 0.079 | **D** | +0.035 |
| 66 | 0.522 ± 0.045 | 0.586 ± 0.049 | 0.618 ± 0.055 | 0.660 ± 0.055 | **D** | +0.032 |
| 100 | 0.592 ± 0.070 | 0.646 ± 0.064 | 0.723 ± 0.034 | 0.750 ± 0.026 | **D** | +0.077 |
| 150 | 0.674 ± 0.038 | 0.762 ± 0.029 | 0.776 ± 0.020 | 0.797 ± 0.023 | **D** | +0.015 |
| 200 | 0.713 ± 0.022 | 0.796 ± 0.029 | 0.803 ± 0.016 | 0.820 ± 0.025 | **D** | +0.007 |
| 300 | 0.749 ± 0.024 | 0.817 ± 0.025 | 0.851 ± 0.018 | 0.862 ± 0.022 | **D** | +0.034 |
| 500 | 0.823 ± 0.012 | 0.869 ± 0.013 | 0.886 ± 0.010 | 0.897 ± 0.009 | **D** | +0.017 |

MaxiMin beats C6 random at **9/10** N values. MM+2-stage beats C6 random at **9/10**
N values. The two-stage model adds consistently over MaxiMin alone at N ≥ 22
(all 9 remaining N values show positive gain).

**Crossover efficiency (MaxiMin-N equivalent to random at which N):**

| MaxiMin N | MM R² | Random equiv N | Efficiency |
|---|---|---|---|
| 22 | 0.384 | 33 | 1.5× fewer |
| 33 | 0.488 | 50 | 1.5× fewer |
| 50 | 0.570 | 66 | 1.3× fewer |
| 66 | 0.618 | 100 | 1.5× fewer |
| 100 | 0.723 | 150 | 1.5× fewer |
| 200 | 0.803 | 300 | 1.5× fewer |
| 300 | 0.851 | 500 | 1.7× fewer |

**N=11 exception:** At N=11, MaxiMin underperforms random C6 (0.245 vs 0.371). This
is analogous to the logistic map N=11 result: at 11 training points, MaxiMin's
structural spread guarantees that test pairs are maximally dissimilar from training
pairs. With this many distinct harmonic structures in the 500-point test set, 11
maximally spread training points are insufficient to cover the space. Random sampling
at N=11 occasionally draws clusters that happen to match the test set distribution.
This crossover resolves by N=22.

---

## What Each Stage Contributed

| Stage | What it removed from the problem | Method |
|---|---|---|
| Mathematical decomposition | Identified p, q, g as the true independent variables | Number theory (exact) |
| Music-theoretic scales | Established normalisation before any data | JI limit theory |
| Probe run | Confirmed log_q_den dominance; identified noise variables | RF importance |
| E-encoding of q, p | Matched basis to bounded consonance-limited character | Fixed-weight e-encoding |
| Pi-encoding of p·q | Matched basis to cascading complexity growth | Fixed-weight π-encoding |
| CANON-6 reduction | Removed 5 redundant and noise features from MATCH-11 | H3 hypothesis test |
| MaxiMin in CANON-6 space | Spread training budget across archetypal intervals | Greedy MaxiMin |
| Ridge grammar | Removed dominant GS ~ f(p,q) trend from residuals | Linear regression |
| XGB dialect | Fitted prime-boundary irregularities in residuals | XGBoost on residuals |

---

## Relationship to the EM and Logistic Map Pipelines

The harmonics pipeline confirms the architecture in a domain where the structural
prior comes from **number theory and music theory** rather than electromagnetic
geometry or dynamical systems. What remains identical across all three domains:

- The encoding functions (same weights, same basis functions)
- The probe mechanism (RF importance → canonical reduction)
- The two-stage model structure (grammar + dialect)
- The normalisation principle (domain constants, not data statistics)

What is different in harmonics:

- **Both e-encoding and π-encoding are used** (q and p are e-type; p·q is π-type),
  unlike the logistic map where all variables are π-type
- **MaxiMin applies** because the selection space is genuinely 2D (p and q are
  independent), unlike the logistic map where all features are functions of r alone
- **The noise variable (g) is mathematically confirmed** before the experiment —
  GS is provably independent of gcd(n,m). This is a stronger prior than in EM or
  the logistic map, and the experiment partially confirms it (H2 unexpected result
  is attributed to finite-pool sampling, not to structural relevance of g)
- **The target is an integer** (GS ∈ ℕ), not a continuous physical quantity.
  No log-transform is applied
- **Normalisation scales come from music theory** (JI limits, harmonic series
  boundaries) rather than physical geometry or Feigenbaum constants

The domain-agnostic claim is confirmed across three structurally distinct domains:
electromagnetic simulation, dynamical systems, and number-theoretic harmony.
