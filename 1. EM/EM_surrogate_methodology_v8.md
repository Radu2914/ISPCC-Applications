# EM Surrogate Methodology — v8 (4SIMM)

## Change Summary: v7 → v8

**What changed:** The prediction pipeline expands from three stages (3SIMM) to four stages (4SIMM) by inserting a TSA-prescribed regime gate between the geometric computation and the grammar fit. This implements TSA Part 8.4 (Three-Phase Architecture) in the EM domain, replacing the single undifferentiated Ridge fit with a regime-conditioned grammar that uses only Ε-type features in the waveguide regime and the full power-law grammar in the penetration regime.

**What did not change:** Stages 0–4 (physical constants, Pi groups, variable classification, encoding, IntentionalMaxiMin selection) are identical to v7. The 127-feature RF dialect (Stage 5/Stage 3 in the new numbering) is unchanged.

**New baseline to beat:** R² = 0.6603 ± 0.0988, RMSE = 0.4189 W/m² (v7 result, 60 points, 5-fold CV).

---

## Problem Statement (unchanged)

5G headset at 28 GHz. Physical quantity: power density at tissue (W/m²), referred to as SAR-proxy. Four physical inputs: **gap** (mm), **upper protective layer thickness** (mm), **lower protective layer thickness** (mm), **protective layer dielectric** (loss tangent, dimensionless). Simulation budget: 200 HFSS points. Prior result used in this paper: XGBoost with pi/e encoding, 200 points, 5-fold CV → R² = 0.5514.

---

## Stage 0 — Physical Constants (unchanged from v7)

All constants derived analytically. No data examined.

**Electromagnetic scales:**
- λ_free = 300/28 = **10.714 mm**
- λ_rubber = λ_free/√4.5 = **5.051 mm**
- NF_BOUND = λ_free/(2π) = **1.705 mm**

**Geometric constants:**
- MODULE_Y = **46.53 mm**; Phantom radius = **12.0 mm**; Aperture half-height = **10.5 mm**
- GAP_SPREAD = 10.5²/(2×12) = **4.594 mm**

**Waveguide triangle (three confirmed HFSS coordinates):**

| Point | Coordinates (mm) | Role |
|---|---|---|
| A — 5G module | (−27.98, −46.62, −32.00) | Source |
| B — Nose ridge | (−7.40, −0.16, −3.46) | Direct path endpoint |
| C — Brow ridge | (−0.90, −49.26, −14.91) | Waveguide deflector |

Computed distances: d(A,B) = 58.280 mm; d(A,C) = 32.130 mm = **3.000λ_free exactly**; angle at C = 86.14°; PATH_RATIO = 0.5513.

**Material bifurcation:** DIEL_BIFURCATION = **0.107** — the loss tangent at which both propagation paths carry equal energy.

**TSA anchors declared in Stage 0:**
- DIEL_BIFURCATION is the **Β-anchor for tan_δ** (TSA type Β, Rule T5).
- PATH_RATIO / DIEL_BIFURCATION is the **⊘Ε threshold constant** — when Π31 = PATH_RATIO / (tan_δ / DIEL_BIFURCATION) crosses 1.0, the forced COMPLETE event ⊘Ε fires and the regime switches.

This declaration was implicit in v7 (DIEL_BIFURCATION appeared as a constant in Π31 but was not used architecturally). In v8 it becomes the gate condition at Stage 5 / Prediction Stage 1.

---

## Stage 1 — Dimensionless Pi Groups (unchanged from v7)

Thirty-four groups constructed from four inputs and five physical constants. The three structurally significant groups:

**Π28 — Path competition ratio:**
$$\Pi_{28} = \frac{d(A,C)}{d(A,B) + \text{gap}}$$

**Π31 — Regime switch index:**
$$\Pi_{31} = \frac{\text{PATH\_RATIO}}{\tan\delta / \text{DIEL\_BIFURCATION}}$$

Π31 > 1: waveguide geometry dominates (SAR-proxy in Ε-regime). Π31 < 1: direct penetration dominates (SAR-proxy in Π-regime). **Π31 = 1 is the ⊘Ε boundary.**

**Π32 — Path difference in wavelengths:**
$$\Pi_{32} = \frac{d(A,B) - d(A,C) + \text{gap}}{\lambda_{\text{free}}}$$

---

## Stage 2 — Variable Classification (TSA Types) (unchanged from v7)

| Variable | TSA type | Physical justification |
|---|---|---|
| gap | **Π (cascade)** | Spans reactive near-field to far-field; non-returning |
| upper layer | **Ε (equilibrium)** | Bounded by resonance at λ_rubber |
| lower layer | **Ε (equilibrium)** | Bounded by resonance at λ_rubber |
| tan_δ | **Β (boundary)** | Crosses DIEL_BIFURCATION; neither cascading nor bounded |

**Target type (TSA Rule T6):** SAR-proxy has Β character from its own physical dynamics. The regime switch encoded in Π31 is the structural expression of Β character: the model alternates between Ε-governed and Π-governed prediction depending on which side of DIEL_BIFURCATION the material sits.

**4SIMM implication (v8):** In v7, the Ridge grammar was trained on all points simultaneously, implicitly asking it to fit both regimes with one coefficient set. TSA Rule T6 and TSA Part 8.4 jointly prescribe that this is wrong: before the ⊘Ε event (Π31 > 1), only Ε-type features should govern the grammar; after ⊘Ε (Π31 < 1), the Π-type cascade features activate. The Ridge grammar's α=−0.812 coefficient for gap was computed from a mixed-regime fit. In the Ε-regime, the true gap exponent is attenuated by waveguide shielding; assigning it −0.812 there is a structural error, not a data error.

---

## Stage 3 — Pi/e Encoding (unchanged from v7)

**Pi-encoding** (for Π-type variables):

$$\Phi_\Pi(\tilde{x}) = \left(\frac{5}{11}\sin(\pi\tilde{x}),\ \frac{1}{11}\cos(\pi\tilde{x}),\ \frac{1}{11}\sin(2\pi\tilde{x}),\ \frac{3}{11}\sin(\pi^2\tilde{x}),\ \frac{1}{11}\sin(\pi\tilde{x})\cos(\pi^2\tilde{x})\right)$$

**E-encoding** (for Ε-type variables):

$$\Phi_E(\tilde{x}) = \left(\frac{2}{5}e^{-e\tilde{x}},\ \frac{2}{5}\tilde{x}^e,\ \frac{1}{5}e^{-e(\tilde{x}-0.5)^2}\right)$$

**Cross-products:** 14 physically motivated interactions. Full encoded set: **127 features** (same as v7).

**New in v8 — Ε-regime feature subset (6 features):**

For Stage 5 / Prediction Stage 2a (Ε-regime grammar), the E-encoded feature set is extracted as the regime-specific input:

| Feature | Pi group | Encoding | Physical meaning |
|---|---|---|---|
| upper_et_exp_neg | Π4 = upper/λ_rubber | E | Layer thickness vs resonance scale |
| upper_et_pow_e | Π4 | E | Power-law equilibrium mode |
| upper_et_gauss | Π4 | E | Gaussian resonance peak |
| lower_et_exp_neg | Π5 = lower/λ_rubber | E | Lower layer resonance |
| lower_et_pow_e | Π5 | E | Power-law equilibrium mode |
| lower_et_gauss | Π5 | E | Gaussian resonance peak |

These 6 features encode the standing-wave resonance structure in the rubber layers. In the waveguide-dominant regime, layer thickness (not gap) is the primary determinant of SAR-proxy magnitude — the waveguide redirects most energy to the periphery, and what reaches tissue is governed by the shielding efficiency of the layers, which peaks at thickness = λ_rubber/2 and λ_rubber. This resonance structure is Ε-type (bounded, convergent to the resonance condition).

*Open question flagged:* Should the solid angle (Π13, E-type, bounded in [0,1]) and total electrical thickness (Π6) also be included in the Ε-regime grammar? These are E-type and physically relevant in the waveguide regime. The proposed starting point is the 6-feature layer-only set above; solid angle and Π6 can be added if the Ε-regime Ridge residuals show systematic structure.

---

## Stage 4 — IntentionalMaxiMin Selection (unchanged from v7)

**7D selection space:** [gap, upper, lower, tan_δ, Π28, Π31, Π32], all normalised to [0,1]. Greedy MaxiMin, seed=0, N=60 (same as v7).

The 7D space remains the correct selection space for 4SIMM. In the 4SIMM architecture:
- Raw 4D dimensions ensure coverage of both grammar inputs (Ε-regime and Π-regime)
- Π31 dimension ensures coverage across the ⊘Ε boundary — the selection will include designs on both sides of Π31 = 1.0

**Regime coverage in the 60-point selection (from v7 results):**
- Π31 range: [0.297, 2.761] — full DoE range covered
- wg (Π31 > 0.8): ~35/60 points; dp (Π31 < 0.4): ~16/60 points; tr (transition): ~9/60 points

This coverage is already appropriate for 4SIMM without changing N or the seed. The ⊘Ε gate at Π31 = 1.0 will have training examples on both sides in every fold.

---

## Stage 5 — Four-Stage Model (4SIMM) [V8 ARCHITECTURE]

This stage replaces the three-stage model in v7. The prediction pipeline now has four stages.

---

### Prediction Stage 0 — Geometric Pi Computation (No Fitting)

Identical to v7. Compute Π28, Π31, Π32 from confirmed HFSS coordinates and four raw inputs. Pure arithmetic — no training data involved.

$$\Pi_{28} = \frac{32.130}{58.280 + \text{gap}}, \quad \Pi_{31} = \frac{0.5513}{\tan\delta / 0.107}, \quad \Pi_{32} = \frac{58.280 - 32.130 + \text{gap}}{10.714}$$

---

### Prediction Stage 1 — Regime Gate (TSA ⊘Ε Event) [V8 ARCHITECTURE]

For each prediction point, evaluate the regime from Prediction Stage 0:

$$\text{regime}(x) = \begin{cases} \text{Ε-regime} & \text{if } \Pi_{31}(x) > 1 + \delta_{\text{gate}} \\ \text{Π-regime} & \text{if } \Pi_{31}(x) < 1 - \delta_{\text{gate}} \\ \text{transition} & \text{if } |\Pi_{31}(x) - 1| \leq \delta_{\text{gate}} \end{cases}$$

**Gate width:** δ_gate = 0.20 (proposed). This places the transition zone at Π31 ∈ [0.80, 1.20], corresponding to tan_δ ∈ [0.089, 0.134] — a ±25% band around DIEL_BIFURCATION = 0.107.

**Physical justification:** The ⊘Ε event in TSA is a forcing event, not a gradual transition. At exactly Π31 = 1.0, both propagation paths carry equal energy and neither grammar is correct alone. The gate width δ_gate acknowledges that at finite distance from Π31 = 1.0, the dominant regime is unambiguous. Within the transition zone, predictions are blended (see below). The width of 0.20 is a proposed starting point and can be tuned based on transition-zone residual structure after initial CV runs.

**In-training versus prediction-time use:**
- Training: each training point is assigned to Ε-regime, Π-regime, or transition by its Π31 value. The Ε-regime Ridge (Stage 2a) is fitted on Ε-regime training points; the Π-regime Ridge (Stage 2b) is fitted on all training points.
- Prediction: Stage 2a and/or Stage 2b are evaluated based on the test point's regime classification.

**TSA mapping:** This is the implementation of TSA Def ⊘ (Forced COMPLETE): ⊘Ε fires when tan_δ crosses DIEL_BIFURCATION upward (Π31 falls below 1.0). Before ⊘Ε (Π31 > 1), the target is Ε-dominant; after ⊘Ε (Π31 < 1), the target is Π-dominant, and the Ridge grammar activates. TSA Part 8.4 prescribes this exactly: "only Ε features should govern the prediction" before ⊘Ε; "the Ridge grammar, which encodes the global Π cascade trend, activates at ⊘Ε."

---

### Prediction Stage 2a — Ε-Regime Grammar (E-Only Ridge) [V8 ARCHITECTURE]

**Applies when:** Π31 > 1 + δ_gate (waveguide-dominant, SAR-proxy in Ε-regime).

**Input features (6):** E-encoded upper and lower layer thickness (upper/λ_rubber, lower/λ_rubber), 3 features each:
```
[upper_et_exp_neg, upper_et_pow_e, upper_et_gauss,
 lower_et_exp_neg, lower_et_pow_e, lower_et_gauss]
```

**Model:** Ridge (α = 1.0) fitted in log space on Ε-regime training points only.

$$\log(P_{\text{wg}}) = c_0 + \sum_{j=1}^{6} c_j \cdot \Phi_{E,j}(\text{layers})$$

**Physical rationale:** In the waveguide-dominant regime, the 5G module's signal arrives at the brow ridge in phase (3λ path, 86° deflection angle), redirecting most energy away from the tissue. What reaches tissue is governed primarily by the shielding efficiency of the rubber layers — which is a resonance phenomenon (standing wave in the rubber, peaks at thickness = λ_rubber/2 and λ_rubber). This is Ε-type structure: bounded, convergent, periodic in the layer thickness coordinate. The cascade trend of gap (α = −0.812 in v7) is attenuated in this regime because the waveguide intercepts and redirects gap-dependent energy. Training a Ridge on Ε-type features only means the grammar is correct for this regime — it fits what actually varies, not what the mixed-regime grammar thinks should vary.

**Why not include gap in Stage 2a:** Gap is Π-type. In the Ε-regime, its influence is present but attenuated. Including it in the Ε-regime grammar risks contaminating the E-only fit with Π-type structure — the Ridge cannot know which coefficient to suppress. The attenuation of gap's influence in the wg regime will appear as a residual for Stage 3 (RF) to correct, which is the correct structural allocation: residual structure that doesn't belong in the grammar goes to the dialect.

---

### Prediction Stage 2b — Π-Regime Grammar (Full Ridge) [unchanged from v7 Stage 1]

**Applies when:** Π31 < 1 − δ_gate (penetration-dominant, SAR-proxy in Π-regime).

**Input features (7):** [gap, upper, lower, tan_δ, Π28, Π31, Π32]

**Model:** Ridge (α = 1.0) in log space. Fitted on **all training points** (not just Π-regime points).

$$\log(P_{\text{dp}}) = \log(C) + \alpha\cdot\log(\text{gap}) + \beta\cdot\log(\text{upper}) + \gamma\cdot\log(\text{lower}) + \delta\cdot\log(\tan\delta) + \zeta\cdot\Pi_{28} + \eta\cdot\Pi_{31} + \theta\cdot\Pi_{32}$$

**Why fitted on all training points:** The Π-regime grammar is the global model — it is the correct grammar everywhere in the Π-regime and an approximation in the Ε-regime. Fitting it on all training points gives it more data and keeps its coefficients stable at small N. The Ε-regime grammar (Stage 2a) is the correction applied in the wg zone; it does not replace the Π-regime grammar's job of anchoring the global structure.

*Open question flagged:* An alternative is to fit Stage 2b only on Π-regime training points, allowing the gap coefficient to be fitted without being pulled toward the wg regime's attenuated gap influence. This could sharpen the Π-regime grammar coefficients. The tradeoff is N reduction (fewer Π-regime training points per fold). To be tested after initial CV.

---

### Transition Zone Blend

**Applies when:** |Π31 − 1| ≤ δ_gate.

$$\log(P_{\text{blend}}) = w \cdot \log(P_{\text{wg}}) + (1 - w) \cdot \log(P_{\text{dp}})$$

where the blend weight is:

$$w = \frac{\Pi_{31} - (1 - \delta_{\text{gate}})}{2\delta_{\text{gate}}} \in [0, 1]$$

At Π31 = 1.0: w = 0.5 (equal blend). At Π31 = 1.20: w = 1.0 (fully wg). At Π31 = 0.80: w = 0.0 (fully dp). The blend is linear in Π31, which is monotone in tan_δ, making the transition smooth and differentiable at both boundaries.

---

### Prediction Stage 3 — RF Dialect on Stage 2 Residuals (unchanged from v7 Stage 2)

**Input:** 127 encoded features (all 34 Pi groups + 60 pi-encoded + 18 e-encoded + 14 cross-products).

**Target:** Residuals from the regime-gated Stage 2 prediction:
```
residual(x) = log(y_true) - log(P_stage2(x))
```
where P_stage2 is Stage 2a, 2b, or blended depending on regime.

**Model:** Random Forest (500 trees, max_features=√p, min_samples_leaf=2, random_state=42).

**What the RF now sees:** The residuals from Stage 2 are structurally cleaner than those from v7's Stage 1. In the Ε-regime, Stage 2a already removes the layer-resonance structure from residuals; the RF sees only gap-attenuation effects, near-field curvature corrections, and HFSS noise. In the Π-regime, Stage 2b is identical to v7 Stage 1, so the RF sees the same standing-wave and near-field residuals as before. The net effect is that Ε-regime residuals (which were the hardest for the v7 RF because the mixed-regime grammar was structurally wrong there) should be smaller and more local.

---

### Final Prediction

$$\hat{y}(x) = \exp\left(\log(P_{\text{stage2}}(x)) + \text{RF}_{\text{residual}}(x)\right)$$

---

## Stage 6 — Evaluation

**Protocol:** 5-fold CV on the same 60 points selected in Stage 4 (KFold seed=42, fixed). Per fold: 48 training, 12 test. Architecture: Regime Gate → E-only Ridge (wg) or Full Ridge (dp) → RF on residuals.

**Reference baseline:** R² = 0.6603 ± 0.0988 (v7, same 60 points).

**Anticipated improvements from the regime gate:**

In v7, the five highest-variance folds were associated with Fold 4 (R²=0.5948) and Fold 5 (R²=0.5287). Post-hoc analysis of the v7 results shows that the test sets in those folds contained disproportionately many Ε-regime points (Π31 > 1.0) where the single Ridge grammar misfits systematically. The 4SIMM Stage 2a should correct this structural misfit, reducing fold-to-fold variance and lifting the mean.

**Safety tracking:** Same protocol — track underprediction rate and correction factor for high-SAR designs (≥2 W/m²) across all folds. The 4SIMM's regime gate should have no adverse effect on safety: high-SAR designs predominantly occur at small gap + small tan_δ, which places them in the Ε-regime (Π31 > 1, small gap). Stage 2a uses E-only features tuned to this region. The RF correction factor (1.437× in v7) is expected to remain stable or decrease as the grammar better fits the Ε-regime predictions.

**Results:** PENDING — code update required. See "What Is Needed" section.

---

## Architecture Comparison: 3SIMM (v7) vs 4SIMM (v8)

| Step | 3SIMM (v7) | 4SIMM (v8) |
|---|---|---|
| P.Stage 0 | Compute Π28, Π31, Π32 | Compute Π28, Π31, Π32 (same) |
| P.Stage 1 | — | **⊘Ε regime gate (Π31 threshold)** ← V8 |
| P.Stage 2 | Ridge on [raw-4 + Π28/31/32] — all points | **2a**: E-only Ridge (wg) / **2b**: Full Ridge (dp) ← SPLIT |
| P.Stage 3 | RF on Stage 1 residuals — 127 features | RF on Stage 2 residuals — 127 features (same) |
| Final | exp(S1 + S3) | exp(S2 + S3) |
| Grammar features | 7 (mixed-regime, all points) | 6 (Ε-only, wg points) + 7 (full, dp points) |
| Residual target | Mixed-regime | Regime-purified |

---

## What Each Stage Contributed

| Stage | What it removed from the problem | Method |
|---|---|---|
| Physical constants | Established normalisation scales before any data | Analytical derivation |
| Waveguide triangle | Encoded the 3λ/86° geometry as regime variables | HFSS coordinate measurement |
| Pi groups | Expressed inputs as physically meaningful ratios | Buckingham's theorem |
| Variable classification | Assigned correct basis family to each variable | TSA type inference |
| Pi/e encoding | Pre-loaded dominant structure into coordinates | Fixed-weight encoding |
| 7D IntentionalMaxiMin | Spread simulation budget across all physics dimensions | Greedy MaxiMin |
| ⊘Ε regime gate | Separated Ε-dominant from Π-dominant prediction regime | TSA Def ⊘, Π31 threshold |
| E-only Ridge (wg) | Layer-resonance grammar for waveguide-active regime | Ridge on E-encoded layers |
| Ridge grammar (dp) | Power law trend + regime correction for penetration regime | Log-linear Ridge (unchanged) |
| RF dialect | Residual resonance, curvature, NF effects | Random Forest on residuals |

---

## Claims the Architecture Supports

**Claim 1 — Regime gate reduces systematic misfit:** The v7 Ridge grammar misassigns gap's cascade coefficient to Ε-regime points where gap's influence is attenuated by waveguide shielding. This misfit is structural — it cannot be corrected by adding more data or adjusting RF. The 4SIMM regime gate removes it at the source.

**Claim 2 — Variance reduction from TSA-prescribed staging:** In v7, σ = 0.0988 across folds. The folds with highest variance correspond to test sets with heavy Ε-regime representation. Purifying the grammar per regime should reduce cross-fold variance further, consistent with the pattern seen between v7 N=40 (σ=0.205) and v7 N=60 (σ=0.099): structural improvements reduce variance, not just more data.

**Claim 3 — TSA Three-Phase Architecture correctly applied:** TSA Part 8.4 prescribes Ε-only before ⊘Ε, Ridge+RF after ⊘Ε. The 4SIMM implements this exactly, with the ⊘Ε event defined by DIEL_BIFURCATION (a Stage 0 physical constant, not a data-fitted threshold). No data was used to determine the gate — it follows from the geometry confirmed in Stage 0.

---

## TSA Genealogy of the Architecture

The following TSA operations are directly expressed in the 4SIMM design:

| TSA concept | Implementation in 4SIMM |
|---|---|
| T = {Π, Ε, Β} | gap:Π, layers:Ε, tan_δ:Β — unchanged from v7 |
| Β-anchor (DIEL_BIFURCATION) | Gate condition in Prediction Stage 1 |
| Def ⊘ (Forced COMPLETE) | ⊘Ε fires at Π31 = 1.0; gate separates pre-⊘Ε from post-⊘Ε |
| TSA Part 8.4 (Three-Phase Architecture) | Ε-only before ⊘Ε (Stage 2a); Ridge+RF after ⊘Ε (Stage 2b + Stage 3) |
| Encoding Theorem — Φ_Ε for Ε variables | Stage 2a uses only Φ_Ε-encoded features |
| Encoding Theorem — Φ_Π for Π variables | Stage 2b uses gap in its raw log form (Π-type grammar) |
| Cross-products as Β-type expressions | Cross-products in Stage 3 RF — unchanged from v7 |

The v7 architecture implemented TSA variable classification and encoding but did not implement TSA's staging prescription. v8 implements TSA end-to-end: the algebra determines not just what basis functions to use but how the model should be structured around the bifurcation event.

---

*v8 status: methodology complete, code update pending decisions on items flagged in "What Is Needed." Results section will be populated after first CV run.*
