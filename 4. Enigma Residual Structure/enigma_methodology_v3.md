# Enigma Structural Diagnostic — v1 → v2 → v3

## Purpose and Framing

The Enigma application is not a surrogate modeling pipeline. It does not predict a
physical quantity from simulation data, and it does not reduce a DoE budget. Its
purpose is different and serves two functions simultaneously.

**Function 1 — Negative ground proof for ISPCC.** The Enigma cipher was deliberately
engineered to resist structural prediction. Its substitution offset (what the machine
does at each position for each input) is designed to be unpredictable from any single
structural axis. A structural diagnostic that finds exploitable structure in Enigma's
substitution would be hallucinating. The correct result is a reliable null. This test
establishes that the ISPCC probe does not find structure where none exists by design —
which is the property that makes it trustworthy in domains where the ground truth is
not known in advance.

**Function 2 — Cryptographic structural classifier.** Applied to the correct targets
(the residuals that Enigma's design does not conceal), the probe correctly classifies
the structural type of each residual without being told which type to look for. This
demonstrates a potential application of ISPCC to side-channel analysis and
distinguisher attacks, where the first step — identifying which structural axis carries
the exploitable information — currently requires human judgment.

The three versions document a progression: v1 applied the wrong target to insufficient
data; v2 fixed the data starvation; v3 fixed the target formulation and completed the
bidirectional structural typing test.

---

## The Enigma System (P1030700 Key)

The machine is an M4 Enigma with the following confirmed key:

| Component | Setting |
|---|---|
| Reflector | B (thin) |
| Greek wheel | Gamma, position V, ring A |
| Left rotor | IV, position M, ring A |
| Middle rotor | III, position G, ring C |
| Fast rotor | VIII, position C, ring U |
| Plugboard | CH EJ NV OU TY LG SZ PK DI QB |

The plugboard connects 10 letter pairs. Six letters (A, F, M, R, W, X) are
self-connected (no plugboard swap). The pair Q↔B has the largest alphabetic
distance (15 positions) — the longest-reach plugboard pair in this key.

**Structural constants (Enigma analogs of Feigenbaum / R_BIFURCATION):**

| Constant | Value | Type | Physical meaning |
|---|---|---|---|
| N_ALPHA | 26 | e-scale | Alphabet size — natural bound on all character variables |
| ROTOR1_PERIOD | 26 | π-scale | Fast rotor revolution — period-26 cascade |
| ROTOR2_PERIOD | 676 = 26² | π-scale | Middle rotor period |
| E_freq | 17.4% | e-scale | German 'E' frequency — normalising constant for frequency residual |
| REFLECT_PROP | 1/26 | e-bound | Reflector eliminates 1/26 of key space |

These constants are derived from the machine specification and standard German corpus
statistics, with no data fitting. They serve the same role as λ_free and
DIEL_BIFURCATION in the EM pipeline: they convert raw quantities into dimensionless
ratios with fixed structural meaning.

**The Enigma invariant:** No letter encrypts to itself. This is a hard constraint
imposed by the reflector's design — the substitution is always a non-identity
permutation. Verified across all 456,976 pairs: 0 self-mappings confirmed.

---

## Variable Classification

**π-type (position / rotor cascade):**
- `fast_pos` = position mod 26 — fast rotor revolution, period-26 cascade
- `mid_pos` = (position ÷ 26) mod 26 — middle rotor period
- `pos_frac` = position / N — global message position
- `notch_f_dist` — distance to Rotor VIII notch positions (Z, M) — stepping bifurcation
- `notch_m_dist` — distance to Rotor III notch position (V) — middle rotor bifurcation
- `eff_fast`, `eff_mid` — ring-adjusted effective positions

These are all cascade-type: position advances non-repeatingly through the rotor state
space, with bifurcations at notch positions where the middle rotor steps.

**e-type (character / involution constraint):**
- `cipher_val` — which letter was received; bounded [0, 25]
- `plug_dist` = |PLUG[input] − input| / 13; bounded [0, 1]; 0 for self-connected letters
- `is_plugged` — 1 if input is in a plugboard pair, 0 otherwise
- `plug_partner` = PLUG[input] / 26 — where the plugboard routes the signal
- `cipher_gfreq` — global frequency of this letter in the message (bounded)
- `cipher_lfreq` — local window frequency (bounded, self-regulating)

These are bounded, self-regulating: character values are confined to [0, 25] by the
alphabet, plugboard distances are bounded by the maximum pair distance (15/13 ≤ 1.15),
and frequencies are bounded by [0, 1].

**Cross-products (π × e):**
Three interaction features: `cross_fast_x_char`, `cross_fast_x_gfreq`,
`cross_fast_x_lfreq`. These encode the joint state where rotor position and character
identity interact — the signal path through a specific plugboard pair at a specific
rotor position.

---

## Feature Set

| Group | Count | Normalisation scale |
|---|---|---|
| π-encoded (position / rotor) | 35 | ROTOR1_PERIOD = 26 |
| e-encoded (character / involution) | 12 | N_ALPHA = 26, MAX_PLUG_DIST = 15 |
| Cross-products (π × e) | 3 | — |
| **Total encoded** | **50** | — |

The full 456,976-pair dataset (all 17,576 rotor positions × 26 input letters) is
generated by the verified M4 simulator. 30,000 pairs are sampled for the probe
(N/P = 600×, vs v1's 8.9×).

---

## v1 — Wrong Target, Insufficient Data

**Protocol:** 239 aligned cipher/plain pairs from one WWII message. Target:
substitution offset = (plain − cipher) mod 26. Features: 27 encoded (π=15, e=9,
cross=3). N/P = 8.9.

**Results:**

| Model | R² |
|---|---|
| Raw features | −0.144 |
| π-only | −0.236 |
| e-only | −0.262 |
| Full (π + e) | −0.095 |

All R² negative. Turing criteria: 0/3. Verdict: π-type dominant.

**What went wrong — two compounding problems:**

*Data starvation:* 239 samples across a 26³ rotor position space. The probe cannot
estimate feature importance or CV accuracy reliably with N/P = 8.9. Cross-products
dominated the probe (combined 28.1% importance) because neither π nor e had enough
data to express themselves independently — the interaction term captured what both
axes together were trying to say.

*Wrong target:* The substitution offset is the full Enigma output — the combined
result of plugboard, rotors, reflector, and plugboard again. It is unpredictable from
any single structural axis by design. Asking the probe to predict it is asking the
probe to break Enigma. The correct null result is all R² near zero or negative. v1's
failure is therefore partially correct: the probe did not hallucinate structure in an
unpredictable target. But it was not designed to test that claim — it happened to
produce the right null for the wrong reason.

---

## v2 — Data Fixed, Target Still Incorrect

**Protocol:** Full M4 simulator verified against known plaintext (131/131 match).
456,976 synthetic pairs (all positions × all inputs); 30,000 sampled. Same target:
substitution offset. New features: explicit plugboard e-type features (plug_dist,
is_plugged, plug_partner), rotor notch bifurcation π-type features. Total: π=35,
e=12, cross=3. N/P = 600×.

**Results:**

| Model | R² | vs v1 |
|---|---|---|
| Raw features | −0.067 | improved |
| π-only | −0.084 | improved |
| e-only | −0.001 | improved dramatically |
| Plugboard-only (e subset) | −0.001 | — |
| Full (π + e + cross) | −0.097 | similar |

**Key finding:** e-only R² converges to −0.001 (near zero). π-only remains at −0.084.
The e-type features, given enough data, correctly admit that the target is
unpredictable from character alone — they converge to a flat prediction rather than
memorising noise. The π-type features, given enough data, overfit false period
structure and stay negative.

This is the correct behaviour for each type. The e-type correctly refuses to predict
what it cannot predict. The π-type finds the most salient periodic pattern in the data
and reports it, even though it explains no variance. This asymmetry reveals the
structural distinction between the two encoding families: e-type features self-limit,
π-type features do not.

**Criterion 2 now met:** e-only R² (−0.001) > π-only R² (−0.084). 1/3 Turing criteria.

**Target diagnosis:** The substitution offset is the wrong target because it is the
key — the full mapping that Enigma was designed to conceal. Turing did not try to
predict the key. He found what persists in the ciphertext regardless of the key:
cipher character frequencies map to plaintext frequencies through the fixed plugboard
involution. This is a different residual — one the cipher is not designed to hide.

---

## v3 — Correct Target, Bidirectional Structural Typing

**The target change:** Three targets tested simultaneously, each with a known
structural type:

| Target | Expected type | Reason |
|---|---|---|
| German_freq[PLUG[inp]] | e-type | Fixed function of character only; position irrelevant |
| Fast rotor position / 26 | π-type | Fixed function of position only; character irrelevant |
| Substitution offset (v2 reference) | neither (×) | Joint-only; cryptographically secure |

**The German frequency structural constant:**

E_freq = 17.4% is the German 'E' frequency — the highest-frequency letter in standard
German. It serves as the normalising scale for the frequency residual, exactly as
R_BIFURCATION normalises the logistic map bifurcation distance or λ_free normalises
the EM gap variable. Each cipher letter, routed through the plugboard, maps to a
partner whose German frequency is a fixed, bounded quantity determined entirely by
character identity.

The plugboard-to-frequency mapping for this key:

| Cipher | → Partner | German freq | Type |
|---|---|---|---|
| J | → E | 17.40% | HIGH — dominant |
| V | → N | 9.78% | HIGH |
| S | → Z | 1.13% | low |
| T | → Y | 0.04% | low |
| B | → Q | 0.02% | low |

The structural spread (17.4% vs 0.02%) is the e-type signal Turing exploited.
High-frequency cipher letters identify high-frequency plaintext letters through the
fixed involution.

**Results (v3, n=30,000, 5-fold CV):**

*Target 1 — German frequency [expected: e-type]:*

| Group | Importance | Ratio |
|---|---|---|
| π-type | 0.4% | — |
| e-type | 96.5% | 225× over π |
| Cross-products | 3.1% | — |

CV: π-only R² = −0.085, e-only R² = **1.000**. Dominant type: **e ✓**

Top features: e_plug_part_pow_e (0.161), e_plug_part_exp_neg (0.159),
e_char_gauss (0.155). The probe correctly identifies that the plugboard partner
(e_plug_part) is the dominant feature — because PLUG[inp] directly determines the
German frequency of the partner letter. The importance ordering recovers the Enigma
signal path (plugboard first) without being told the architecture.

*Target 2 — Fast rotor position [expected: π-type]:*

| Group | Importance | Ratio |
|---|---|---|
| π-type | 97.0% | 322× over e |
| e-type | 0.3% | — |
| Cross-products | 2.7% | — |

CV: π-only R² = **0.992**, e-only R² = −0.001. Dominant type: **π ✓**

*Target 3 — Substitution offset [expected: neither]:*

| Group | Importance | Ratio |
|---|---|---|
| π-type | 60.1% | 2.8× over e |
| e-type | 21.3% | — |
| Cross-products | 18.6% | — |

CV: π-only R² = −0.084, e-only R² = −0.001. Neither positive. Dominant: **× ✓**

**Bidirectional structural typing confirmed:**

| Target | e-only R² | π-only R² | Detected type | Correct? |
|---|---|---|---|---|
| German freq [e] | 1.000 | −0.085 | e | ✓ |
| Fast rotor [π] | −0.001 | 0.992 | π | ✓ |
| Subst offset [×] | −0.001 | −0.084 | neither | ✓ |

3/3 correct structural classifications, separation ratios 225×, 322×, and near-zero
respectively — all in the correct direction.

---

## The R² = 1.000 Qualifier

The German frequency result (R² = 1.000, e-type) is correct but not a discovery made
blind. The plugboard partner (`e_plug_part`) was provided as an explicit encoded
feature, and PLUG[inp] directly determines the German frequency of the partner letter.
The probe confirmed the structure rather than discovering it from ciphertext alone.

**What this does and does not prove:**

It does not prove that ISPCC can break Enigma or recover the plugboard pairs from
ciphertext without the key. The plugboard mapping was given to the probe as a feature.
In a real cryptanalytic scenario, that mapping is what you are trying to find.

It does prove three things that are not trivial:

1. The probe correctly classified all three structural types simultaneously, with
   separation ratios of 225×, 322×, and near-zero — all correct, all without being
   told which type to expect.

2. The probe recovered the correct importance ordering within the e-type features:
   `e_plug_part` ranked above `e_char`. This is the correct Enigma signal path —
   the plugboard routes the input before the rotors process it. The probe recovered
   the architectural precedence from importance ranking without being told the signal
   flows plugboard-first.

3. The substitution offset result is a reliable null. The probe correctly refused to
   find single-axis predictive structure where the cipher was designed to prevent it.
   A method that hallucinated structure in the substitution offset would be
   disqualified as a structural diagnostic. The null held at 8.9× N/P (v1) and
   600× N/P (v2) — the negative result is not data starvation.

---

## The Four Structural Outcomes

The Enigma probe established that the ISPCC diagnostic has four discriminable
outcomes, not two:

1. **e-dominant, positive R²** — bounded / involution structure exploitable by
   character features alone (German frequency target)

2. **π-dominant, positive R²** — periodic / cascade structure exploitable by
   position features alone (rotor position target)

3. **neither, negative R²** — joint-only; single-axis prediction is
   cryptographically blocked (substitution offset)

4. **cross-dominant, negative R²** — interaction required; neither axis alone
   is sufficient, but the interaction of position and character carries a signal
   (observed at v1's N/P = 8.9 where cross-products dominated at 28.1%)

Outcomes 3 and 4 are correct null results. The probe does not hallucinate structure
where the target is resistant to single-axis prediction. This is the reliability
property the other three domains (logistic map, harmonics, EM) could not test —
all three have genuine predictable structure. The Enigma substitution offset is the
first domain where the correct answer is *no single-axis structure exists*, and the
probe correctly returns it.

---

## Potential Application to Cryptanalysis

The Enigma test is framed as a structural diagnostic, not a cryptanalytic tool.
But three specific applications follow from the results:

**Side-channel and implementation analysis.** Real cipher implementations leak
information through timing, power consumption, or cache behaviour. These leakages
have structural character: timing leakages are periodic (clock-dependent, π-type);
data leakages are bounded and self-regulating (data-dependent, e-type). The ISPCC
probe applied to side-channel traces would automatically classify whether a leakage
is timing-structural (π) or data-structural (e) without the analyst needing to
hypothesise the source first. That classification directs the attack.

**Distinguisher analysis.** A well-designed cipher produces output structurally
indistinguishable from random on both axes — the substitution offset result is exactly
this. Applied to a cipher with a structural weakness — a biased output distribution
— the probe would detect which axis carries the bias before a human analyst formulates
the right statistical test. This automates the first step of distinguisher attacks.

**Known-plaintext structure recovery.** In a known-plaintext scenario the pairs are
available but the key is not. The probe would identify whether to pursue frequency
analysis (e-type target, as in Turing's frequency method) or period analysis (π-type
target, as in index-of-coincidence attacks) without running both exhaustively. The
v3 result shows this classification operates correctly at R² separation ratios above
200× — sufficient to make the call unambiguously.

The distinction is important: ISPCC is a structural diagnostic, not a cryptanalytic
attack. Turing found the structure. ISPCC would identify, reliably and automatically,
which type of structure to look for before the search begins.

---

## Relationship to the Other Domains

The Enigma probe is structurally different from the EM, logistic map, and harmonics
pipelines in every dimension:

- **No surrogate model.** There is no DoE budget to reduce, no simulation cost, no
  prediction accuracy target. The pipeline produces a structural classification, not
  a predicted value.
- **No IntentionalMaxiMin.** Sampling from a synthetic dataset of known structure;
  no selection problem.
- **No two-stage grammar/dialect model.** The 3-stage pipeline was tested and found
  to add no value for the substitution offset target. Ridge on π-features followed by
  RF on e-residuals produced R² = −0.277 vs Ridge alone at R² = −0.050. This is the
  correct result: there is no grammar to remove from an unpredictable target. The
  absence of pipeline value is itself a structural finding.
- **Both e and π encoding used**, with cross-products — same as harmonics and EM.
- **Normalisation scales from machine specification and linguistics** (alphabet size,
  rotor periods, German corpus frequency) rather than physical geometry or dynamical
  systems constants.

The domain-agnostic claim is extended to a deliberately adversarial case: the same
probe, the same encoding functions, the same weights, applied to a system engineered
to resist structural prediction — and it correctly returns null on the resistant
target while correctly classifying the non-resistant residuals.

---

## Annex — Cross-Domain Probe Summary

Probe behaviour across all domains tested (e-only R² and π-only R² from 5-fold CV):

| Domain | e-only R² | π-only R² | Dominant type | Finding |
|---|---|---|---|---|
| Logistic map | ~0.87 | ~0.88 | π | Chaos signal; both axes similar; π marginal |
| Harmonics (Euler GS) | ~0.85 | ~0.60 | e | Denominator q carries consonance structure |
| EM simulation (SAR) | moderate | moderate | both | e-type margin at small N |
| Enigma — subst offset | −0.001 | −0.084 | × (neither) | Cryptographically secure; correct null |
| Enigma — German freq | 1.000 | −0.085 | e | Turing's frequency layer; 225× separation |
| Enigma — rotor position | −0.001 | 0.992 | π | Period structure; 322× separation |

The four structural outcomes established across these domains:

| Outcome | e R² | π R² | Example |
|---|---|---|---|
| e-dominant | positive | near-zero or negative | Harmonics, Enigma freq |
| π-dominant | near-zero or negative | positive | Logistic map, Enigma rotor |
| joint (×) | negative | negative | Enigma subst offset |
| cross-dominant | negative | negative | Enigma v1 at N/P = 8.9 |

No previous validation domain covered outcomes 3 and 4. The encoding is not a feature
engineering method that improves accuracy — it is a structural diagnostic that
correctly classifies the type of information available in any domain, including
correctly identifying when single-feature prediction is cryptographically impossible.
