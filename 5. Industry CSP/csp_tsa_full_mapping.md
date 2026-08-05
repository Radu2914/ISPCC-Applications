# Annex — Full TSA Combination Mapping: Bearing Degradation Domain

## Preface

This annex exhaustively maps the TSA calculus across all meaningful variable
combinations in the CSP bearing application. The purpose is to show that the
algebra is not applied selectively — every combination of input variables produces
a typed result, every model operation corresponds to a TSA operation, and every
operational state of the deployment system has a precise algebraic description.

**TSA operation rules:**

| Operation | Rule | Meaning |
|---|---|---|
| Π ⊕ Π | = Π | Cascade combined with cascade remains cascade |
| Ε ⊕ Ε | = Ε | Equilibrium combined with equilibrium remains equilibrium |
| Π ⊕ Ε | = Β | Cascade combined with equilibrium produces boundary |
| Β ⊕ Π | = Π | Cascade dominates boundary |
| Β ⊕ Ε | = Β | Boundary absorbs equilibrium |
| Β ⊕ Β | = Β | Boundary combined with boundary remains boundary |
| Π ⊗ Β | = Ε | Resolving cascade from boundary leaves equilibrium residuals |
| Ε ⊗ Β | = Π | Resolving equilibrium from boundary leaves cascade residuals |
| ¬Π | = Ε | Inverting cascade gives equilibrium |
| ¬Ε | = Π | Inverting equilibrium gives cascade |
| ¬Β | = Β | Boundary inverted remains boundary |
| ↓Ε | = Π | Equilibrium completing gives cascade (bifurcation event) |
| ↓Π | = ∅ | Cascade completing gives failure |

---

## B1. Single Variable Typings (Complete)

| Variable | Type | Physical state |
|---|---|---|
| RMS_h | Π | Horizontal vibration energy — cascading |
| RMS_v | Π | Vertical vibration energy — cascading |
| Kurt_h | Π | Horizontal impulsiveness — cascading |
| Kurt_v | Π | Vertical impulsiveness — cascading |
| Peak_h | Π | Horizontal shock amplitude — cascading |
| Peak_v | Π | Vertical shock amplitude — cascading |
| life_frac | Π | Time elapsed fraction — strictly monotone |
| rms_bif_dist | Π | Distance past RMS bifurcation — zero then growing |
| temperature | Ε | Thermal state — self-regulating |
| temp_dev | Ε | Thermal deviation — mean-reverting |
| crest_h | Ε | Horizontal peak/RMS ratio — geometry-bounded |
| crest_v | Ε | Vertical peak/RMS ratio — geometry-bounded |

---

## B2. Pairwise Combinations — Same Type (Π × Π and Ε × Ε)

All same-type pairs reduce to the shared type. The 28 Π-pairs all produce Π;
the 6 Ε-pairs all produce Ε. Selected physically meaningful pairs shown:

**Π ⊕ Π = Π (cascade reinforcement):**

| Combination | Result | Physical meaning |
|---|---|---|
| RMS_h ⊕ RMS_v | Π | Total vibration energy — both axes cascading together |
| Kurt_h ⊕ Kurt_v | Π | Combined impulsiveness — fault visible in both axes |
| RMS_h ⊕ Kurt_h | Π | Energy + impulsiveness — envelope of horizontal fault severity |
| Peak_h ⊕ Peak_v | Π | Shock amplitude — maximum instantaneous force |
| life_frac ⊕ rms_bif_dist | Π | Time + cascade distance — the power law grammar |
| RMS_h ⊕ life_frac | Π | Energy growth over time — primary degradation signal |
| Kurt_h ⊕ rms_bif_dist | Π | Impulsiveness past bifurcation — confirmed fault growth |
| Peak_h ⊕ life_frac | Π | Shock growth over time — structural damage accumulation |
| RMS_h ⊕ RMS_v ⊕ Kurt_h ⊕ Kurt_v | Π | Full mechanical cascade group |
| All Π variables (8) | Π | Complete cascade feature set |

**Ε ⊕ Ε = Ε (equilibrium reinforcement):**

| Combination | Result | Physical meaning |
|---|---|---|
| temperature ⊕ temp_dev | Ε | Full thermal state — both absolute and relative bounded |
| crest_h ⊕ crest_v | Ε | Combined crest — geometry-bounded in both axes |
| temperature ⊕ crest_h | Ε | Thermal + geometric equilibrium — self-regulation confirmed |
| temp_dev ⊕ crest_v | Ε | Deviation metrics — both mean-reverting |
| All Ε variables (4) | Ε | Complete equilibrium feature set |

---

## B3. Pairwise Combinations — Cross Type (Π × Ε = Β)

All 32 cross-type pairs produce Β. Selected physically meaningful pairs:

| Combination | Result | Physical interpretation |
|---|---|---|
| RMS_h ⊕ temperature | Β | Vibration cascade vs thermal equilibrium — the core bearing state |
| RMS_v ⊕ temp_dev | Β | Vertical cascade vs thermal deviation boundary |
| Kurt_h ⊕ crest_h | Β | Impulsiveness vs geometry ratio — fault vs healthy structure competing |
| Kurt_v ⊕ crest_v | Β | Same in vertical axis |
| rms_bif_dist ⊕ temp_dev | Β | Cascade indicator vs thermal boundary — most direct Β expression |
| life_frac ⊕ temperature | Β | Time elapsed vs thermal self-regulation — aging vs stability |
| life_frac ⊕ crest_h | Β | Time elapsed vs geometric health — the wear boundary |
| Peak_h ⊕ crest_h | Β | Shock vs geometry ratio — peak fault event vs healthy structure |
| rms_bif_dist ⊕ crest_h | Β | Cascade onset vs geometric health — most operationally critical pair |
| RMS_h ⊕ crest_h | Β | Energy vs geometry — the transition from crest-governed to RMS-governed |

**Note on rms_bif_dist ⊕ crest_h:** This pair is the most diagnostically precise
Β expression in the domain. crest_h is Ε until the rolling element surface begins
to fail (at which point it plateaus and then drops as RMS rises faster than peak);
rms_bif_dist is zero until 0.5g is crossed and then grows. Their combination is Β
from the moment the bearing leaves the purely healthy state.

---

## B4. Three-Variable Combinations (Selected)

| Combination | Result | Interpretation |
|---|---|---|
| RMS_h ⊕ RMS_v ⊕ temperature | Β | Both vibration axes + thermal: full bearing state (minimal) |
| Kurt_h ⊕ Kurt_v ⊕ crest_h | Β | Impulsiveness + geometry: fault character |
| life_frac ⊕ rms_bif_dist ⊕ temp_dev | Β | Time + cascade onset + thermal boundary: Stage-0 grammar core |
| RMS_h ⊕ Kurt_h ⊕ crest_h | Β | Energy + impulsiveness + geometry: full horizontal fault picture |
| rms_bif_dist ⊕ temp_dev ⊕ life_frac | Β | The three structural grammar features: Β confirmed |
| temperature ⊕ temp_dev ⊕ crest_h | Ε | All three Ε: self-regulation group (healthy bearing signature) |
| RMS_h ⊕ Kurt_h ⊕ Peak_h | Π | All three horizontal cascade features: fault severity in one axis |

---

## B5. Feature Group Combinations (Operational)

| Group | Variables | Type | Operational meaning |
|---|---|---|---|
| Mechanical cascade | RMS_h, RMS_v, Kurt_h, Kurt_v, Peak_h, Peak_v | Π | Full fault severity state |
| Life position | life_frac, rms_bif_dist | Π | Time and cascade distance |
| Full Π group | All 8 Π variables | Π | Complete cascade feature set |
| Thermal | temperature, temp_dev | Ε | Thermal self-regulation state |
| Geometric | crest_h, crest_v | Ε | Geometric health of rolling surface |
| Full Ε group | All 4 Ε variables | Ε | Complete equilibrium feature set |
| Stage-0 grammar | life_frac, rms_bif_dist, temp_dev + encoded | Β | Grammar input: Π ⊕ Ε = Β |
| All 12 variables | All | Β | Complete bearing state = Β (target type confirmed) |

---

## B6. RESOLVE Operations ⊗ (The Two-Stage Model in TSA)

The two-stage model is a sequential application of RESOLVE:

**Stage 1 — Ridge resolves Π from Β:**

$$\Pi(\text{life\_frac, rms\_bif\_dist, ...}) \otimes \text{B}(\text{rul\_norm}) = \varepsilon_{\text{residuals}}$$

Ridge fitted on Π features, subtracted from the Β target, leaves Ε residuals.
By ¬Π = Ε: removing the cascade component exposes the equilibrium texture.

**Stage 2 — RF fits the Ε residuals:**

$$\text{RF}(\varepsilon_\text{E}) \rightarrow \hat{\varepsilon}$$

Random Forest on Ε residuals captures bearing-specific degradation texture:
kurtosis structure within the healthy phase, crest factor plateau before
cascade onset, and the smooth degradation curve between bifurcation events.

**Reconstruction:**

$$\hat{\text{B}} = \text{Ridge}(\Pi) + \text{RF}(\varepsilon_\text{E})$$

$$= \Pi + \neg\Pi = \text{B}$$

The sum of a Π prediction and its complement (Ε residuals) reconstructs the Β
target. This is not a modeling choice — it is the algebraic identity for Β.

---

## B7. INVERT Operations ¬ (Diagnostic Interpretation)

The INVERT operation ¬ applied to diagnostic outputs produces their structural
complement. In the bearing domain:

| Expression | Result | Operational interpretation |
|---|---|---|
| ¬(Π cascade in RMS) | Ε | RMS self-regulation restored — not physically possible once ↓Ε reached |
| ¬(Ε boundary in temperature) | Π | Thermal cascade — not reached in any deployment bearing |
| ¬(Ε boundary in crest) | Π | Geometric failure — rolling element surface breakdown |
| ¬(Β target = rul_norm) | Β | Complement of remaining life is elapsed life; same Β type |
| ¬(rms_bif_dist > 0) | Ε | Return below bifurcation threshold — physically impossible post-↓Ε |
| ¬(life_frac advancing) | Π | Time reversal — not physically possible; confirms Π irreversibility |

The INVERT of all Π diagnostic signals is physically impossible in the bearing
domain — once a cascade variable has crossed its COMPLETE threshold, ¬ cannot
be realised. This is the algebraic expression of irreversibility: Π variables
cannot be inverted under normal operating conditions.

---

## B8. COMPLETE Events ↓ (Bifurcation Constants)

The COMPLETE operation ↓ marks regime transitions. Four COMPLETE events are
defined in the bearing domain, at constants derived from physics:

| Variable | COMPLETE event | Constant | Physical meaning |
|---|---|---|---|
| RMS | ↓Ε(RMS) = Π(RMS) | 0.5g (RMS_BIFURCATION) | Vibration leaves self-regulating regime — detectable by surrogate |
| temperature | ↓Ε(temp) = Π(temp) | 25 + 5 = 30°C (TEMP_BIFURCATION) | Thermal equilibrium fails — not reached in deployment |
| crest | ↓Ε(crest) = Π(crest) | geometry-dependent | Rolling surface breakdown precedes ↓Π(RMS) |
| RMS | ↓Π(RMS) = ∅ | 20g (FAILURE_G) | Cascade completes — physical failure, test stops |

**The cascade detection window** is the interval between ↓Ε(RMS) and ↓Π(RMS):

$$\Delta t = t_{↓\Pi} - t_{↓\varepsilon} = \text{lead time}$$

Measured values across deployment bearings:

| Bearing | ↓Ε detected | ↓Π (failure) | Δt (lead time) |
|---|---|---|---|
| Bearing1_3 | 294.8 min | 395.8 min | **68.0 min** |
| Bearing2_4 | 95.0 min | 125.2 min | **30.3 min** |
| Bearing3_3 | 58.8 min | 72.3 min | **17.5 min** |

The lead time is not a model parameter — it is the physical duration between
two TSA events (↓Ε and ↓Π) that the surrogate identifies from sensor data.

---

## B9. Per-State TSA Expression (Four Operational States)

Each deployment state corresponds to a specific TSA algebraic expression evaluated
on the current snapshot's encoded features:

**HEALTHY — Ε dominant, Π not yet active:**

$$\text{State} = \varepsilon(\text{temperature}) \oplus \varepsilon(\text{crest}) \oplus \varepsilon(\text{RMS below bifurcation})$$
$$= \varepsilon$$

All variables self-regulating. rms_bif_dist = 0 (COMPLETE not reached).
rul_norm > 0.50. Signal: `e-regul.`

**EARLY — Β transitional, both types present:**

$$\text{State} = \Pi(\text{life\_frac growing}) \oplus \varepsilon(\text{RMS still bounded}) \oplus \varepsilon(\text{temperature intact})$$
$$= \Pi \oplus \varepsilon = \text{B}$$

Π emerging through life_frac and early RMS growth. Ε boundary still intact
in temperature and crest. rul_norm 0.25–0.50. Signal: `π/e mixed`

**CASCADE — Π dominant, ↓Ε reached:**

$$\text{State} = \Pi(\text{rms\_bif\_dist} > 0) \oplus \varepsilon(\text{temperature intact})$$
$$= \Pi \oplus \varepsilon = \text{B with } \Pi \text{ dominant}$$

↓Ε(RMS) has been reached. rms_bif_dist actively growing. Temperature and crest
still Ε (e-type boundary intact). rul_norm 0.08–0.25. Signal: `π-cascade`

**CRITICAL — Π accelerating toward ↓Π:**

$$\text{State} = \Pi(\text{rms\_bif\_dist} \gg 0) \oplus \varepsilon(\text{temperature intact})$$
$$= \text{B approaching } \downarrow\Pi$$

Cascade accelerating; rul_norm < 0.08. ↓Π imminent. Temperature Ε boundary
still intact in all observed cases. Signal: `π-cascade`

**Note:** Temperature remained Ε in all four states across all three deployment
bearings. The thermal COMPLETE event (↓Ε(temperature)) was never triggered —
the test protocol stops at mechanical ↓Π before thermal cascade develops. The
CSP correctly identifies this: `self-regulating ✓ (e-type boundary intact)` is
reported at every CASCADE and CRITICAL snapshot in the deployment output.

---

## B10. Full TSA Expression of the Deployment System

The complete CSP deployment, stated as a single TSA expression:

$$\text{CSP} = \left[\text{Ridge}(\Pi) \otimes \text{B}\right] + \text{RF}(\neg\Pi)$$

$$= \text{B}$$

Evaluated at each 10-second snapshot, producing:

$$\hat{\text{rul\_norm}} = \exp\left(\hat{\text{B}}\right)$$

$$\text{State} = \begin{cases} \text{HEALTHY} & \hat{\text{B}} \text{ Ε-dominant} \\ \text{EARLY} & \hat{\text{B}} \text{ transitional} \\ \text{CASCADE} & \hat{\text{B}} \text{ Π-dominant, } \downarrow\varepsilon \text{ reached} \\ \text{CRITICAL} & \hat{\text{B}} \to \downarrow\Pi \end{cases}$$

The deployment system is a real-time evaluator of the TSA expression for the
Β-type rul_norm target, streaming at 10-second intervals, with threshold crossings
corresponding directly to TSA COMPLETE events. The algebra is not a post-hoc
description of what the model does — it is the structure from which the model
architecture, the feature encoding, the threshold values, and the diagnostic
outputs were all derived before training began.
