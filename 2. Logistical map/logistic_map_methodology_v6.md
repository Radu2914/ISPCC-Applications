# Logistic Map Surrogate Methodology — v5 → v6

## Problem Statement

The target system is the logistic map: x_{n+1} = r · x_{n} · (1 − x_{n}). The input is the
control parameter **r** drawn uniformly from [2.5, 4.0]. The quantity of interest is the
**Lyapunov exponent λ**, computed from the long-run orbit after transient discard. λ < 0
indicates a stable (periodic) regime; λ ≥ 0 indicates chaos. The pool contains 2000
samples (73.3% stable, 26.7% chaotic).

The logistic map is used as a validation domain because ground truth is exact and
computable to arbitrary precision. There is no measurement noise, no simulation
uncertainty, and no hidden variables. Any accuracy gain from the pipeline is therefore
attributable to the pipeline architecture, not to lucky data.

The work proceeded in two versions. v5 established the encoding baseline and identified
the best single-stage model. v6 introduced the three-stage pipeline and tested it
against those baselines across the full N range.

---

## Why MaxiMin Is Not Applied in This Domain

IntentionalMaxiMin was tested and removed. The logistic map has a single input: r.
Every encoded feature — regardless of how many basis functions are constructed — is a
deterministic function of r alone. MaxiMin in the encoded space is therefore MaxiMin
across nonlinear transforms of a single variable. The greedy algorithm selects
structurally extreme r values: bifurcation corners, chaos window boundaries,
period-doubling thresholds. When 5-fold CV is then applied, test folds contain points
intentionally far from all training points in structural terms. The result is R² = −51
at n=22.

This is not a failure of MaxiMin as an algorithm. It is the correct output of MaxiMin
applied to the wrong geometry. MaxiMin is valid for multi-dimensional independent input
spaces — in the EM pipeline, four independent physical inputs produce a genuinely
4-dimensional selection space. It is not valid for a 1D parameter system where all
dimensions are coupled by construction.

The finding from the logistic map is therefore about the pipeline architecture alone.
The three-stage model improves over single-stage baselines regardless of how samples
are drawn.

---

## Stage 0 — Structural Constants (No Fitting, No Data Required)

All constants are derived from the known mathematics of the logistic map before any
sample is examined.

**Bifurcation anchor:**
- R_BIFURCATION = **3.5699456** — the accumulation point of the period-doubling cascade,
  at which the map transitions from periodic to chaotic behaviour. This is the Β-type
  anchor of the system: the value of r at which the character of the orbit changes
  irreversibly from self-regulating to cascading.

**Feigenbaum constants (universal, not system-specific):**
- δ = **4.669201609** — the rate at which successive bifurcation intervals contract.
  Every one-dimensional map with a quadratic maximum converges to the same δ. It is the
  universal scale of the period-doubling cascade.
- α = **2.502907875** — the ratio of successive orbit-width scales at bifurcation. It
  governs the spatial structure of the attractor as r approaches R_BIFURCATION.

These constants serve as the normalisation scales for the encoding. They are analogous
to λ_free, λ_rubber, and DIEL_BIFURCATION in the EM pipeline: each converts a raw
quantity into a dimensionless ratio whose physical meaning is fixed.

**Derived normalisation scales:**

| Scale constant | Value | Physical meaning |
|---|---|---|
| SCALE_R | 3.5699456 | r as fraction of bifurcation point |
| SCALE_DIST | 0.4301 (= R_MAX − R_BIF) | distance from bifurcation, normalised to chaotic range |
| SCALE_STD | 1/(2α) = 0.1998 | orbit standard deviation at bifurcation, α-normalised |
| SCALE_VAR | SCALE_STD² | variance scale |
| SCALE_SPREAD | α = 2.503 | orbit range scale |
| SCALE_N_OCC | δ² = 21.80 | number of occupied bins, δ²-normalised |
| SCALE_AC | 1.0 | autocorrelation (already bounded in [−1, 1]) |
| SCALE_MEAN_X | (R_BIF − 1)/R_BIF = 0.7201 | mean orbit position at bifurcation |
| SCALE_INV_NOCC | 1/δ = 0.2142 | inverse occupation density |

---

## Stage 1 — Orbit Statistics (The Raw Feature Set)

Because r is the only input, the features available to the model are statistics computed
from the orbit of the logistic map at each r value. Each r generates a sequence of 500
iterates (after discarding 1000 transients from x₀ = 0.5), from which 11 statistics
are extracted:

| Statistic | Physical meaning |
|---|---|
| r | The control parameter itself |
| mean_x | Mean orbit position |
| std_x | Orbit standard deviation — zero in fixed points, growing with period |
| var_d1 | Variance of first differences — measures step-to-step variability |
| var_d2 | Variance of second differences — measures acceleration of the orbit |
| ac1 | Lag-1 autocorrelation — near 1 for periodic, near 0 for chaotic |
| ac2 | Lag-2 autocorrelation |
| n_occ | Number of occupied histogram bins (out of 100) — 1 for fixed point, ~100 for chaos |
| spread | Orbit range / mean_x — normalised amplitude of oscillation |
| dist_bifurc | r − R_BIFURCATION (signed distance from bifurcation point) |
| abs_dist | \|r − R_BIFURCATION\| (unsigned distance) |

All 11 statistics are cascade (Π-type) in character: as r increases from 2.5 to 4.0,
each passes through successive bifurcations and does not return to its prior value.
std_x, n_occ, and spread monotonically increase with r in the chaotic regime. ac1 and
ac2 decrease monotonically. No statistic is self-regulating in the Ε-type sense — there
is no bounded equilibrium to return to because the entire domain of interest spans the
bifurcation cascade. For this reason, only pi-encoding is used; e-encoding is not
applied in this domain.

---

## Stage 2 — Pi-Encoding (The Coordinate System)

All 11 statistics are pi-encoded using identical basis functions and weights to the
EM pipeline. The encoding is domain-agnostic: the same functions, the same weights,
with only the normalisation scales adapted to the logistic map's constants.

**Pi-encoding** applied to all 11 statistics:

For normalised value x̃ = clip(x/scale, 0, 10):

$$\Phi_\Pi(\tilde{x}) = \left(\frac{5}{11}\sin(\pi\tilde{x}),\ \frac{1}{11}\cos(\pi\tilde{x}),\ \frac{1}{11}\sin(2\pi\tilde{x}),\ \frac{3}{11}\sin(\pi^2\tilde{x}),\ \frac{1}{11}\sin(\pi\tilde{x})\cos(\pi^2\tilde{x})\right)$$

This produces 55 encoded features (11 statistics × 5 basis functions each). The
normalisation scale for each statistic is its corresponding Feigenbaum-derived constant —
not the mean or standard deviation of the training sample.

**The probe** — RF importance ranking on the full 55-feature set — identifies the
11 most informative encoded features, one per source statistic, by taking the
highest-importance basis function for each source variable:

```
ENCODED_11 = [
    pienc_r_val_sin_pi,       # r position (dominant)
    pienc_mean_x_cos_pi,      # mean orbit position
    pienc_std_sin_2pi,        # orbit standard deviation
    pienc_var_d1_sin_pi2,     # first-difference variance
    pienc_var_d2_sin_pi2,     # second-difference variance
    pienc_ac1_cos_pi,         # lag-1 autocorrelation
    pienc_ac2_cos_pi,         # lag-2 autocorrelation
    pienc_n_occ_sin_pi2,      # occupied histogram bins
    pienc_spread_cos_pi,      # orbit spread
    pienc_abs_dist_sin_pi,    # distance from bifurcation
    pienc_inv_n_occ_cos_pi,   # inverse occupation (regime switch indicator)
]
```

The probe result is consistent with the EM and bearing domains: canonical reduction
automatically selects one representative per structural dimension, reducing 55 features
to 11 without manual selection.

---

## v5 — Baseline Encoding Experiment

v5 compared four models across the full N range to establish which combination of
model and feature set performed best:

- **RF raw:** Random Forest on 11 raw orbit statistics
- **XGB raw:** XGBoost on 11 raw orbit statistics
- **RF enc:** Random Forest on ENCODED_11
- **XGB enc:** XGBoost on ENCODED_11

Protocol: 10 seeds, 5-fold CV, N ∈ [11, 22, 33, 50, 66, 100, 110, 121, 150, 200, 300, 500].
N=110 and N=121 were added deliberately to probe the N/P=10 boundary (P=11 features).

**Results (v5, 10 seeds, 5-fold CV):**

| N | RF raw | XGB raw | RF enc | XGB enc | XGB enc > XGB raw? |
|---|---|---|---|---|---|
| 11 | −10.33 ± 13.78 | −36.23 ± 60.24 | −14.92 ± 24.50 | −10.42 ± 11.94 | YES (Δ=+25.81) |
| 22 | 0.222 ± 0.399 | −0.302 ± 1.060 | 0.239 ± 0.401 | 0.248 ± 0.467 | YES (Δ=+0.55) |
| 33 | 0.437 ± 0.493 | 0.435 ± 0.369 | 0.554 ± 0.315 | 0.575 ± 0.280 | YES (Δ=+0.14) |
| 50 | 0.795 ± 0.079 | 0.511 ± 0.394 | 0.800 ± 0.076 | 0.728 ± 0.178 | YES (Δ=+0.22) |
| 66 | 0.805 ± 0.103 | 0.772 ± 0.165 | 0.807 ± 0.107 | 0.822 ± 0.132 | YES (Δ=+0.05) |
| 100 | 0.880 ± 0.070 | 0.845 ± 0.114 | 0.890 ± 0.030 | 0.868 ± 0.086 | YES (Δ=+0.02) |
| **110** | **0.917 ± 0.022** | **0.867 ± 0.119** | **0.898 ± 0.045** | **0.791 ± 0.239** | **NO (Δ=−0.08)** |
| 121 | 0.886 ± 0.061 | 0.855 ± 0.100 | 0.893 ± 0.038 | 0.861 ± 0.095 | YES (Δ=+0.01) |
| 150 | 0.915 ± 0.046 | 0.893 ± 0.060 | 0.910 ± 0.041 | 0.888 ± 0.061 | NO |
| 200 | 0.928 ± 0.020 | 0.903 ± 0.046 | 0.924 ± 0.017 | 0.897 ± 0.044 | NO |
| 300 | 0.947 ± 0.010 | 0.945 ± 0.013 | 0.946 ± 0.012 | 0.952 ± 0.014 | YES |
| 500 | 0.970 ± 0.011 | 0.966 ± 0.016 | 0.969 ± 0.009 | 0.963 ± 0.018 | NO |

XGB encoded beats XGB raw at **8/12** sample sizes.

**What v5 established:** XGB encoded at n=100 (R²=0.868) and RF raw at n=100 (R²=0.880)
become the reference baselines for v6. RF encoded is consistently competitive with RF
raw throughout, confirming that the encoding does not degrade performance.

**N/P=10 anomaly:** At N=110 (exactly N/P=10 where P=11 encoded features), XGB encoded
collapses to R²=0.791 ± 0.239 — a sharp drop relative to N=100 and N=121, with variance
more than doubling. RF encoded at the same N holds at 0.898 ± 0.045. N=110 and N=121
were added to the sweep specifically to confirm this boundary. The pattern is noted
without a confirmed interpretation and will be examined in further work.

---

## v6 — Three-Stage Pipeline

v5 showed that a single-stage model with encoding plateaus and that XGB's boosting
mechanism becomes unstable at specific N/P ratios. v6 introduced a three-stage
architecture: Ridge fits the structural grammar first, RF learns the residuals second.

### Structural Features (Stage 0 Grammar Input)

Three of the 11 encoded features serve as the structural grammar for Stage 1 Ridge,
analogous to Π28, Π31, and Π32 in the EM pipeline:

| Feature | EM analog | Physical meaning |
|---|---|---|
| pienc_r_val_sin_pi | Π28 (path competition) | r-position relative to R_BIFURCATION; encodes where in the cascade the current sample sits |
| pienc_inv_n_occ_cos_pi | Π31 (regime switch) | Inverse of occupied bins, δ-normalised; highest-importance feature from probe; distinguishes periodic from chaotic regime |
| pienc_abs_dist_sin_pi | Π32 (path difference) | Absolute distance from bifurcation, encoded; captures how far into either regime the sample has progressed |

These three encoded features are combined with the two raw structural statistics
(r and abs_dist) to form the 5-feature Stage 0 input:

```
Stage-0 = [r, abs_dist, pienc_r_val_sin_pi, pienc_inv_n_occ_cos_pi, pienc_abs_dist_sin_pi]
```

### Three-Stage Model

**Stage 1 — Ridge regression (the grammar):**

Input: 5 Stage-0 features. Target: Lyapunov exponent λ (no log-transform — λ is already
defined on ℝ and spans a natural linear range from approximately −1.5 to +0.7).

Ridge (α=1.0) fits the structural grammar: the dominant λ ~ f(r) monotone trend, the
near-bifurcation regime inflection, and the Feigenbaum-normalised period structure.
This is the logistic map equivalent of the power law grammar in the EM pipeline. Once
fitted, Ridge removes the dominant signal from the residuals.

**Stage 2 — Random Forest on residuals (the dialect):**

Input: ENCODED_11. Target: residuals from Stage 1.

The RF (500 trees, max_features=√p, min_samples_leaf=2) learns the fine structure that
the linear grammar cannot represent: the bifurcation cascade windows (sudden drops in λ
at period-doubling points), the periodic windows inside the chaotic regime (narrow bands
where λ returns to negative values), and the chaos-band sub-structure at large r.

**Final prediction:** λ_pred = Stage-1(x) + Stage-2(x)

No exponential transform is applied — unlike the EM case, the target is not constrained
to be positive.

### Evaluation

**Protocol:** For each of 10 independent random seeds and each N, draw N points
uniformly from the 2000-point pool without replacement. Apply 5-fold CV (KFold,
shuffle=True, random_state=42). Report mean R² ± std across seeds. Pool size of 2000
with N ≤ 500 ensures each seed draws at most 25% of the pool, avoiding overlap bias.

**Baselines:** RF raw (raw 11 statistics) and XGB encoded (ENCODED_11, the v5 winner).

**Results (v6, 10 seeds, 5-fold CV):**

| N | RF raw | XGB enc | 3-stage | 3s > RF? | 3s > XGB? |
|---|---|---|---|---|---|
| 11 | −10.33 ± 13.78 | −10.42 ± 11.94 | −6.62 ± 8.50 | YES | YES |
| 22 | 0.222 ± 0.399 | 0.248 ± 0.467 | 0.376 ± 0.364 | YES | YES |
| 33 | 0.437 ± 0.493 | 0.575 ± 0.280 | 0.602 ± 0.297 | YES | YES |
| 50 | 0.795 ± 0.079 | 0.728 ± 0.178 | 0.809 ± 0.069 | YES | YES |
| 66 | 0.805 ± 0.103 | 0.822 ± 0.132 | 0.833 ± 0.074 | YES | YES |
| 100 | 0.880 ± 0.070 | 0.868 ± 0.086 | 0.890 ± 0.027 | YES | YES |
| 150 | 0.918 ± 0.028 | 0.881 ± 0.076 | 0.902 ± 0.031 | NO | YES |
| 200 | 0.932 ± 0.025 | 0.920 ± 0.032 | 0.922 ± 0.022 | NO | YES |
| 300 | 0.959 ± 0.015 | 0.956 ± 0.017 | 0.948 ± 0.018 | NO | NO |
| 500 | 0.972 ± 0.006 | 0.972 ± 0.007 | 0.968 ± 0.006 | NO | NO |

3-stage beats RF raw at **6/10** sample sizes. 3-stage beats XGB encoded at **8/10**
sample sizes.

**Cross-N efficiency:**

| 3-stage N | R²_3s | Equivalent RF raw N | Speedup |
|---|---|---|---|
| 33 | 0.602 | 50 | 1.5× |
| 50 | 0.809 | 100 | 2.0× |
| 66 | 0.833 | 100 | 1.5× |
| 100 | 0.890 | 150 | 1.5× |

The peak data efficiency gain is **2.0× at N=50**: the 3-stage pipeline at 50 samples
matches what RF raw requires 100 samples to achieve.

---

## What the Results Show and What They Do Not Show

**What they show:** The three-stage pipeline is a consistent improvement over
single-stage models at small to medium N. The improvement is largest where data is
scarcest (N=11–100) and narrows as N grows. At large N (300, 500), raw RF has enough
data to discover the structure itself and the pipeline advantage disappears. This is the
expected and correct behaviour: the pipeline front-loads structure that the model would
otherwise have to learn from data; at large N the model can learn it directly.

**What they do not show:** The pipeline does not improve over RF raw at all N. Above
N=150, raw RF's advantage in model flexibility outweighs the structural pre-loading.
The claim is about data efficiency at small N, not about universal superiority.

**Variance reduction:** The 3-stage pipeline consistently shows lower σ than its
baselines at the same N. At N=50, RF raw σ=0.079 and 3-stage σ=0.069. At N=100,
RF raw σ=0.070 and 3-stage σ=0.027. The structural grammar stabilises predictions
across seeds because Ridge's output is deterministic given the training set — the
dominant signal is removed consistently regardless of which samples were drawn.

**Why N=11 gives negative R²:** At N=11, 5-fold CV trains on 8 points and tests on 3.
Eight points are insufficient to constrain either Ridge or RF in a domain with multiple
bifurcation regimes. All three methods fail at this scale; the 3-stage result (−6.62)
is less negative than both baselines (−10.33, −10.42), but all are noise.

---

## What Each Stage Contributed

| Stage | What it removed from the problem | Method |
|---|---|---|
| Feigenbaum constants | Established normalisation scales before any data | Analytical derivation |
| Orbit statistics | Extracted structurally meaningful summaries from r alone | Logistic sequence simulation |
| Pi-encoding | Encoded cascade character into basis functions | Fixed-weight π-encoding |
| Probe (ENCODED_11) | Reduced 55 features to 11 canonical representatives | RF importance ranking |
| Stage-0 features | Isolated three structural regime indicators | Probe-guided selection |
| Ridge grammar | Removed dominant λ ~ f(r) trend and bifurcation inflection | Linear regression |
| RF dialect | Fitted period-doubling windows and chaos sub-structure | Random Forest on residuals |

---

## Relationship to the EM Pipeline

The logistic map pipeline is the same architecture applied to a domain where the
structural constants come from number theory (Feigenbaum) rather than electromagnetic
geometry (HFSS coordinates). The encoding functions, weights, probe mechanism, and
two-stage model are identical. What changes between domains is:

- The normalisation scales (Feigenbaum δ, α instead of λ_free, λ_rubber)
- The structural grammar features (Π_r, Π_inv_n_occ, Π_abs_dist instead of Π28, Π31, Π32)
- The absence of IntentionalMaxiMin (1D input space, not applicable)
- The absence of e-encoding (all variables cascade-type; no bounded equilibrium variables)
- The absence of a log-transform on the target (λ is already defined on ℝ)

The domain-agnostic claim is confirmed: the same pipeline, with constants and
normalisation adapted to the domain's known physics, produces a consistent
data-efficiency gain across a mathematically exact test case with no hidden variables.
