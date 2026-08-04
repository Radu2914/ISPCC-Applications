# Annex — TSA Calculus for the Bearing Degradation Domain

## A1. Variable Typing

Each extracted feature is assigned a TSA state from bearing physics before
any data is examined. The classification follows from the physical character
of each variable, not from data statistics.

| Variable | TSA type | Physical justification |
|---|---|---|
| RMS_h, RMS_v | Π | Accumulates toward FAILURE_G; no self-correcting force after fault onset |
| Kurt_h, Kurt_v | Π | Impulsiveness grows non-repeatingly as fault severity increases |
| Peak_h, Peak_v | Π | Shock levels cascade upward at each fault contact event |
| life_frac | Π | Strictly monotone; never returns to a prior value |
| rms_bif_dist | Π | Zero in healthy phase; positive and growing after cascade onset |
| temperature | Ε | Self-regulates around TEMP_AMBIENT under constant load |
| temp_dev | Ε | Bounded deviation; mean-reverting until thermal event |
| crest_h, crest_v | Ε | Peak/RMS ratio; self-corrects in healthy state; bounded by geometry |

---

## A2. Target Type Derivation

The CASCADE operation ⊕ combines TSA types. Rules: Π ⊕ Π = Π, Ε ⊕ Ε = Ε,
Π ⊕ Ε = Β.

Combining all input variables:

$$(\text{RMS} \oplus \text{Kurt} \oplus \text{Peak} \oplus \text{life\_frac} \oplus \text{rms\_bif\_dist}) \oplus (\text{temperature} \oplus \text{temp\_dev} \oplus \text{crest})$$

$$= \Pi \oplus \text{E} = \textbf{B}$$

**rul_norm is Β-type.** This is physically correct: remaining useful life begins
in the Ε regime (slow, bounded degradation — the bearing is self-regulating) and
crosses into the Π regime (accelerating cascade — damage accumulates faster than
the bearing can dissipate) at fault onset. The boundary between these two regimes
is exactly what the cascade state detection identifies.

This typed derivation explains why a single-stage model underperforms on bearing
degradation: a model trained only on Π features cannot represent the Ε residuals,
and a model trained only on Ε features cannot represent the Π trend. A Β target
requires both.

---

## A3. Two-Stage Model as TSA Operations

The two-stage model is a direct implementation of the TSA RESOLVE operation ⊗.

**Stage 1 — Ridge grammar (RESOLVE of Π):**

Ridge is fitted to the Stage-0 features, which are Π-dominant (life_frac,
rms_bif_dist, and their encoded forms). Subtracting its prediction is the
RESOLVE operation:

$$\text{Ridge}(\Pi) \otimes \text{target} \rightarrow \text{residuals}$$

By the TSA axiom ¬Π = Ε (inverting the cascade removes the cascade component):
the residuals remaining after Ridge are Ε-type — the bounded, mean-reverting
texture that linear grammar cannot represent.

**Stage 2 — RF dialect (fit of Ε residuals):**

Random Forest fits the Ε residuals — the bearing-specific degradation texture,
kurtosis structure, and crest factor patterns that survive after the power-law
grammar is removed.

**Full prediction in TSA notation:**

$$\hat{y} = \text{Ridge}(\Pi) + \text{RF}(\varepsilon_\text{E}) = \hat{\text{B}}$$

The sum of a Π prediction and an Ε correction produces a Β estimate — the correct
structural type for the target.

---

## A4. The Bifurcation Constant as a TSA COMPLETE Operation

RMS_BIFURCATION = 0.5g is the COMPLETE (↓) of the Ε state for the RMS dimension.
In TSA notation:

$$\downarrow\varepsilon(\text{RMS}) = \Pi(\text{RMS}) \quad \text{at RMS} = 0.5\text{g}$$

The COMPLETE operation ↓ marks the transition from bounded equilibrium to irreversible
cascade. The feature `rms_bif_dist = max(0, rms_env − 0.5)` encodes this precisely:

- rms_bif_dist = 0 → Ε state intact; COMPLETE not yet reached
- rms_bif_dist > 0 → Π state active; COMPLETE triggered; cascade in progress

This is the structural reason rms_bif_dist dominates probe importance at fault onset:
it is the only feature that directly encodes the COMPLETE event.

---

## A5. The Four Deployment States as TSA Regimes

The four operational states produced by the CSP deployment have direct TSA
interpretations:

| State | TSA interpretation | Physical meaning |
|---|---|---|
| HEALTHY | Β with Ε dominant | Equilibrium controls; cascade not yet initiated |
| EARLY | Β in transition | Π beginning to emerge; ↓Ε approaching |
| CASCADE | Β with Π dominant | ↓Ε reached; cascade has taken over the RMS dimension |
| CRITICAL | Π approaching ↓Π | COMPLETE imminent; FAILURE_G boundary approaching |

The cascade detection event is the practical identification of ↓Ε being reached in
the RMS dimension — before the system reaches ↓Π at the failure criterion
(FAILURE_G = 20g). The lead time (17–68 minutes across deployment bearings) is the
temporal gap between ↓Ε (detectable by the surrogate) and ↓Π (physical failure).

---

## A6. Temperature as Confirmed Ε Throughout

In all three deployment bearings, temperature never crossed TEMP_BIFURCATION = 5°C
above ambient in any sustained phase. In TSA:

$$\downarrow\varepsilon(\text{temperature}) \text{ was never reached}$$

Temperature remained Ε-type throughout the full bearing life in every deployment run.
The CSP correctly reports "self-regulating ✓ (e-type boundary intact)" at every
cascade snapshot — the Ε boundary was not completed in the thermal dimension even
as the mechanical Π cascade reached completion. This is consistent with the test
protocol: PRONOSTIA stops at mechanical failure, before thermal cascade can develop.

The TSA calculus therefore produces three independent typed results simultaneously
at every snapshot: the Π-dominant cascade confirmed in the RMS dimension, the Ε
boundary intact in the thermal dimension, and the overall Β prediction of rul_norm.
These are not post-hoc labels — they are the direct output of the variable
classification applied to the current encoded feature state.
