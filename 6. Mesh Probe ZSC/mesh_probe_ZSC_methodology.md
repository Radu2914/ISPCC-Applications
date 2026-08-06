# ISPCC Mesh Probe — Zone Structural Classifier (ZSC) — CAD Zone Classification Methodology

## Purpose and Framing

The mesh probe is a standalone ISPCC instrument for classifying parametric CAD mesh
zones by their TSA dynamical character before any solver is run. It takes geometry
descriptors exported from a CAD package and returns a TSA type {Π, Ε, Β} per zone,
a refinement recommendation, and a structured description field suitable for both
human review and automated pipeline consumption.

This application is structurally different from all prior ISPCC domains in one
important respect: the classification target is not a measured physical quantity.
It is derived from a confirmed engineering constant (the h = R/6 meshing rule)
applied to local geometry — a theoretically grounded score, computed from geometry
alone, before any simulation data exists. The probe confirms that the TSA encoding
correctly captures the structural character of mesh refinement decisions without
requiring solver output to validate against.

The downstream use is direct: the classification output feeds a meshing API call.
The TSA type per zone determines the refinement action (REFINE / COARSE OK / REVIEW
/ INSPECT), and the description field carries the quantitative justification in a
form both an engineer and an automated system can consume.

---

## The Engineering Problem

A CAD model is partitioned into parametric zones before meshing. The mesh quality in
each zone determines solver accuracy. The decision of where to refine is currently
made by the engineer from visual inspection and experience — a time-consuming step
that is domain-specific, non-transferable, and difficult to automate.

The fundamental distinction the probe makes is the same distinction experienced
meshing engineers make implicitly: some zones have geometry where mesh error
**compounds** if the mesh is too coarse (curvature-driven, aspect-ratio-driven,
proximity-to-boundary-condition-driven), and some zones have geometry where mesh
error **self-corrects** (flat surfaces, smoothly varying normals, elements far from
stress concentrations). The first class requires refinement. The second tolerates
coarse mesh without accuracy loss.

TSA names these two classes Π and Ε respectively. Zones at the structural transition
between them — where the engineer genuinely cannot decide without solver feedback —
are Β.

---

## Stage 0 — Structural Constants (No Data, No Solver Required)

All normalisation scales are derived from meshing standards and solver accuracy
requirements before any zone is examined.

**The bifurcation constant — h = R/6:**

The central engineering rule in curvature-driven meshing is that the element size h
at a surface of radius of curvature R must satisfy h ≤ R/6 for the solver to
accurately represent the curved geometry. At h = R/6 exactly, the mesh is at the
bifurcation between acceptable and unacceptable representation. This gives the
KAPPA_BIFURCATION anchor:

$$\text{KAPPA\_BIFURCATION} = \frac{1}{6 \cdot h_{\text{median}}}$$

where h_median is the median element size across all zones — making the threshold
robust to extreme outlier zones. This constant is computed from the mesh geometry
itself, analogous to RMS_BIFURCATION in the bearing CSP and DIEL_BIFURCATION in
the EM surrogate.

**Fixed engineering thresholds:**

| Constant | Value | Physical meaning |
|---|---|---|
| KAPPA_FACTOR | 6.0 | h = R/6 engineering rule |
| ASPECT_BIFURCATION | 5.0 | Element quality threshold — solver accuracy degrades above |
| SKEW_BIFURCATION | 0.85 | Skewness limit — solver stability boundary |
| DIST_NF_FACTOR | 0.10 | BC near-field radius = 10% of characteristic length |
| NORMAL_DEV_SCALE | 90.0° | Maximum meaningful normal deviation (geometric bound) |

These constants are not calibrated from data. They are established meshing practice
encoded as ISPCC structural constants — the same role as physical constants in the
other domains.

**The refinement score:**

$$S = 0.50 \cdot (\kappa_{\max} \cdot h_{\text{local}} \cdot 6) + 0.25 \cdot \frac{\text{aspect}}{\text{ASPECT\_BIF}} + 0.15 \cdot \frac{\text{skewness}}{\text{SKEW\_BIF}} + 0.10 \cdot \text{bc\_proximity}$$

The curvature term uses local element size per zone (not the global median), so
each zone is judged by its own current resolution. The score equals 1.0 exactly
when the local element is at the h = R/6 bifurcation condition — making 1.0 the
Β-anchor of the classification.

Score > 1.5 → Π (refinement necessary). Score < 0.60 → Ε (coarse mesh acceptable).
Score between 0.60 and 1.5 → Β (boundary regime, inspect).

Curvature receives weight 0.50 because it is the primary driver of mesh error in
CAE. Aspect ratio and skewness are secondary quality metrics. BC proximity encodes
the physical observation that stress gradients near boundary condition attachment
points require local refinement independent of surface geometry.

---

## Stage 1 — Variable Classification (From Meshing Physics)

**π-type (cascade — mesh error compounds non-repeatingly):**

| Variable | Physical justification |
|---|---|
| kappa_max | High curvature → field gradients grow without bound if element is too coarse; refining one level does not restore accuracy without further refinement |
| kappa_mean | Mean curvature of zone — same cascade character as kappa_max |
| aspect_ratio | Element quality degrades non-repeatingly past ASPECT_BIFURCATION; no geometric self-correction |
| skewness | Past SKEW_BIFURCATION, solver accuracy does not recover; cascade toward numerical instability |
| dist_to_bc | Proximity to BC drives field gradient cascade; the closer to the attachment, the more the field concentrates |

**ε-type (equilibrium — bounded, self-regulating, coarse mesh acceptable):**

| Variable | Physical justification |
|---|---|
| normal_deviation | Surface normal variation is bounded by geometry — a flat face cannot exceed 0°, a hemisphere cannot exceed 90°; bounded by construction |
| area_fraction | Zone area fraction is bounded in [0, 1] by definition; self-regulating in the sense that zones compete for a fixed total area |
| edge_length_min | Minimum element size is bounded by mesh generation constraints and the characteristic length; self-limited |

**Cross-products (Π × Ε interaction):**

Three cross-products encode physically motivated regime interactions:
- sin(π × kappa_n) × exp(−e × normal_n): curvature cascade modulated by surface flatness
- sin(π × aspect_n) × exp(−e × area_n): element quality cascade modulated by zone size
- sin(π × dist_n) × exp(−e × edge_n): BC near-field cascade modulated by local element density

Cross-products are maximally active when the cascade variable is elevated and the
equilibrium variable is at its mid-range — the structural condition where both
regimes are simultaneously present, characteristic of the Β zone.

---

## Stage 2 — Pi/e Encoding

The same encoding functions and weights as all prior ISPCC domains. No domain-specific
tuning.

**Pi-encoding** (for Π-type variables — cascade, non-periodic, non-repeating):

$$\Phi_\Pi(\tilde{x}) = \left(\frac{5}{11}\sin(\pi\tilde{x}),\ \frac{1}{11}\cos(\pi\tilde{x}),\ \frac{1}{11}\sin(2\pi\tilde{x}),\ \frac{3}{11}\sin(\pi^2\tilde{x}),\ \frac{1}{11}\sin(\pi\tilde{x})\cos(\pi^2\tilde{x})\right)$$

Applied to: kappa_max, kappa_mean (normalised by local element size × KAPPA_FACTOR),
aspect_ratio (normalised by ASPECT_BIFURCATION), skewness (normalised by
SKEW_BIFURCATION), dist_to_bc (normalised by DIST_NF).

**E-encoding** (for Ε-type variables — bounded, self-regulating):

$$\Phi_E(\tilde{x}) = \left(\frac{2}{5}e^{-e\tilde{x}},\ \frac{2}{5}\tilde{x}^e,\ \frac{1}{5}e^{-e(\tilde{x}-0.5)^2}\right)$$

Applied to: normal_deviation (normalised by 90°), area_fraction (normalised by 1.0),
edge_length_min (normalised by L_CHAR/6).

The full encoded feature set contains 37 features: 25 Π-encoded (5 variables × 5
basis functions each), 9 Ε-encoded (3 variables × 3 basis functions each), and 3
cross-products.

---

## Stage 3 — Probe (Structural Confirmation)

The RF probe is run against the geometry-derived refinement score — not against
solver output. It confirms that the variable classification is structurally correct
before any simulation is performed.

Expected result: Π-encoded features dominate importance, because the refinement
score is primarily driven by curvature (a cascade variable). If Ε-encoded features
dominate, the variable classification requires review.

**Probe results across test shapes:**

| Shape | Π importance | Ε importance | × importance | Dominant | Ratio |
|---|---|---|---|---|---|
| Sphere (all Ε zones) | 79.6% | 9.2% | 11.2% | Π | 8.7× |
| Cylinder (Π/Ε/Β) | 88.6% | 5.8% | 5.6% | Π | 15.3× |
| Box (Π/Ε/Β) | 58.3% | 37.2% | 4.5% | Π | 1.6× |
| Cone (Π/Ε/Β) | 61.1% | 32.3% | 6.6% | Π | 1.9× |

Π dominates in all four cases. The sphere result (79.6% Π despite all zones being
Ε-classified) is correct: the probe is measuring which *feature type* drives the
refinement score, not which zones require refinement. Curvature features drive
the score even in low-curvature geometries — the score is near zero for a sphere
because the curvature is low, but curvature is still the structural determinant.

The box result (58.3% Π, 37.2% Ε) shows the closest split. This is expected:
the box has a large proportion of flat-face Ε zones where area_fraction and
normal_deviation contribute meaningfully to distinguishing those zones from
the corner Π zones. The Ε features are doing real work in the box geometry.

---

## Stage 4 — Zone Classification

Each zone receives a TSA type through a priority-ordered decision:

**Priority 1 — Physics-driven load zone (--load flag):** Force Π regardless of
geometry. A flat surface with low curvature scores Ε on geometry alone, but
stress concentrations at load application points are solver-driven and the probe
cannot see them. The --load override encodes the engineer's physical knowledge
that a zone requires refinement independent of its shape.

**Priority 2 — User-flagged zone (--flag flag):** Force Β regardless of geometry.
The engineer is asserting uncertainty about this zone. The score and Π/Ε ratio are
still computed and displayed, so the engineer can see what the probe would have
said, but the output classification is always Β and the recommendation is always
INSPECT.

**Priority 3–5 — Geometry-driven classification:**
- Score > 1.5 → Π: cascade regime, curvature or quality has crossed the
  bifurcation; refinement required
- Score < 0.60 → Ε: equilibrium regime, geometry is self-regulating; coarse
  mesh acceptable
- Otherwise → Β: boundary regime, at or near bifurcation; moderate refinement
  and visual inspection recommended

The distinction between --load and --flag is architecturally significant. --load
encodes a physical assertion: the engineer knows this zone needs refinement. --flag
encodes epistemic uncertainty: the engineer is not sure. Both override the geometry
classification, but for different reasons, and they produce different downstream
actions in the meshing pipeline.

**Per-zone output:**

Each zone produces five output fields:
- `tsa_type`: Π, Ε, or Β
- `score`: the geometry-derived refinement score (bifurcation = 1.0)
- `pi_e_ratio`: ratio of mean Π-encoded to mean Ε-encoded feature values
- `rec`: short recommendation (REFINE / COARSE OK / REVIEW / INSPECT)
- `desc`: full description including curvature values, threshold comparison,
  and recommended element size (h ≤ R/6 for Π zones)

The `desc` field is designed for dual consumption: an engineer reads it directly;
an automated pipeline parses it to construct the meshing API call.

---

## Validation

**Unit test results across four basic shapes (all zones, all shapes):**

| Shape | Expected pattern | Accuracy |
|---|---|---|
| Sphere | All Ε (uniform κ, below threshold everywhere) | 8/8 ✓ |
| Cylinder | Curved surface → Π, flat ends → Ε, edge transitions → Β | 8/8 ✓ |
| Box | Sharp edges/corners → Π, flat faces → Ε, face-edge transitions → Β | 8/8 ✓ |
| Cone | Tip/upper → Π, mid/lower/base-edge → Β, base flat/far → Ε | 8/8 ✓ |

100% classification accuracy across all shapes. Expected TSA types are derived from
the h = R/6 rule applied locally per zone — not from running the probe. The probe
output matches the physics-derived expectation in every case.

**Validation boundaries — what is and is not confirmed:**

| Claim | Status |
|---|---|
| Formula implemented correctly | ✓ Confirmed by unit tests |
| Π-encoding structurally appropriate for curvature-driven refinement | ✓ Confirmed by probe dominance across all shapes |
| Score threshold 1.5 (Π/Β boundary) matches solver behaviour | ⚠ Engineering rule — not yet validated against reference mesher |
| Score threshold 0.60 (Β/Ε boundary) matches solver behaviour | ⚠ Engineering rule — not yet validated against reference mesher |
| Unit test parameters chosen independently of the formula | ⚠ Parameters are consistent with the formula by construction — unit test, not physical validation |

The threshold validation step requires running the same zones through a reference
mesher and comparing where it places refinement boundaries. This is the standard
solver calibration step and is independent of the ISPCC framework.

---

## What Is Structurally New in This Domain

Relative to all prior ISPCC domains, the mesh probe introduces three new elements:

**1. A theoretically derived target.** In every prior domain, the probe target was
a measured or simulated quantity (power density, Lyapunov exponent, Euler GS, rul_norm,
substitution offset). Here the target is a formula — the geometry-derived refinement
score — computed entirely from engineering constants. The probe therefore confirms
structural alignment between the encoding and the theory, not between the encoding
and data. This is the strongest form of the structural confirmation claim.

**2. Epistemic uncertainty as a typed output.** The --flag override produces a Β
classification not because the geometry is at the bifurcation, but because the
engineer is uncertain. This is the first explicit encoding in ISPCC of the difference
between *structural* Β (the geometry is at the transition) and *epistemic* Β (the
geometry may or may not be at the transition — human review required). Both produce
the same output type for the same downstream reason: the decision is non-obvious
and requires attention.

**3. Physical assertion as a typed override.** The --load override produces a Π
classification not from geometry but from physical knowledge that the probe cannot
access from shape descriptors alone. Load application points require refinement
because of stress concentration physics that is solver-dependent and
geometry-independent. This encodes the boundary between what TSA can determine
from structural typing (geometry-driven cascade or equilibrium) and what it cannot
(physics-driven requirements that require domain knowledge to assert).

---

## Relationship to Prior ISPCC Domains

The mesh probe uses identical encoding functions, weights, and probe mechanism to
the EM surrogate, bearing CSP, logistic map, and harmonics pipelines. What changes:

- **Normalisation scales** come from meshing engineering constants (h = R/6,
  ASPECT_BIFURCATION, SKEW_BIFURCATION) rather than physical geometry or
  dynamical systems constants
- **No surrogate model trained on data** — classification follows directly from
  the score formula without a fitted Ridge or RF stage
- **No IntentionalMaxiMin** — there is no simulation budget to allocate; zones
  are classified from existing geometry descriptors
- **Direct operational output** — the classification feeds a meshing API call
  without further processing, making this the most operationally direct
  ISPCC application to date
- **Both Π and Ε encoding used**, with cross-products — same structure as
  harmonics and EM

The domain-agnostic claim is extended to a domain where neither training data nor
solver output is available — pure geometry classification from structural typing
alone, using constants derived from engineering standards rather than physics
experiments.
