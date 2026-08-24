# Phosphorus Material Classifier — Reference

## What this tool does

Given a set of material properties, the classifier places the material at a point in a compact 3D space and finds the closest known elements. The three coordinates of that space are **Π**, **Ε**, and **Β**. Each coordinate captures a different kind of physical character.

This document explains what those coordinates mean, how they are computed, and what they imply for materials used in nuclear applications.

---

## The three coordinates

### Π — Cascade character

A material with high Π score has properties that are large in scale, non-periodic, and accumulating. Think of it as measuring how far a material sits from the stable centre of the periodic table and how strongly it participates in processes that are irreversible and directional.

In nuclear terms: materials with high Π character tend to be strong neutron absorbers. Their nuclear cross-sections are large, often spanning orders of magnitude relative to lighter elements, and their interaction with radiation is dominated by capture rather than scattering.

### Ε — Equilibrium character

A material with high Ε score has properties that are bounded, self-regulating, and periodic. These are materials whose behaviour returns to a stable state after perturbation — structurally, thermally, and chemically.

In nuclear terms: materials with high Ε character tend to be moderators or structural materials. They slow neutrons through repeated elastic collisions (self-correcting energy transfer) or provide mechanical integrity that maintains its character under irradiation.

### Β — Boundary character

A material with high Β score sits at a structural transition — between metallic and non-metallic behaviour, between stable and unstable isotope configurations, or between neutron-absorbing and neutron-transparent regimes. The Β score peaks when a material is precisely at a separatrix: neither fully one thing nor another.

In nuclear terms: materials with high Β character are relevant wherever a controlled balance is required — criticality management, burnable poisons, or structural components at the neutron flux boundary. The boundary condition is not a defect; it is the design target.

---

## Nuclear role lookup

| Role | Π | Ε | Β | Examples from the classifier |
|---|---|---|---|---|
| Neutron absorber | High | Low–Mid | Mid | Gd, B, Cd, Hf |
| Neutron moderator | Low | High | Low–Mid | H, C (graphite), D₂O (via H/O) |
| Structural material | Low–Mid | High | Mid | Fe, Ni, Zr, Al |
| Fuel | High | Mid | High | U, Pu (via neighbours) |
| Reflector | Low | High | Low | Be, C, Pb |
| Burnable poison | Mid | Mid | High | Gd, B (at operating conditions) |
| Coolant carrier | Low | High | Low | O (water), Na, He |

These are indicative, not prescriptive. A material's role depends on geometry, temperature, and neutron spectrum — not on its TSA fingerprint alone. The fingerprint is a fast first filter, not a qualification.

---

## How each coordinate is computed

### Input properties and their type

Every property fed to the classifier is assigned a type before any computation. The type determines which mathematical basis is used to encode it.

| Property | Type | Physical reasoning |
|---|---|---|
| Atomic mass | Π | Increases monotonically across the periodic table — cascade character |
| Electronegativity | Ε | Bounded between 0.7 (Cs) and 4.0 (F), periodic — equilibrium character |
| First ionisation energy | Ε | Bounded, resets at each noble gas — equilibrium character |
| Atomic radius | Ε | Bounded, periodic within each shell — equilibrium character |
| Melting point | Ε | Bounded by W at 3695 K — equilibrium character |
| Boiling point | Ε | Bounded by W at 5555 K — equilibrium character |
| Density | Ε | Bounded by Os at 22.59 g/cm³ — equilibrium character |
| Electron affinity | Β | Zero-crossing between metals and non-metals — boundary character |
| Period (1–7) | Ε | Bounded count of electron shells — equilibrium character |
| Group (1–18) | Ε | Bounded valence column — equilibrium character |
| N/Z ratio (computed) | Β | Valley of stability is a separatrix — boundary character |

### Normalisation

Each property is divided by a physical constant before encoding. These constants are derived from physics, not from the dataset.

| Property | Normalisation constant | Source |
|---|---|---|
| Atomic mass | 56.0 u | Iron peak — the nuclear binding energy attractor |
| Electronegativity | 4.0 | Fluorine Pauling maximum |
| Ionisation energy | 24.59 eV | Helium first IE — noble gas ceiling |
| Atomic radius | 298.0 pm | Caesium — largest s-block radius |
| Melting point | 3695.0 K | Tungsten melting point |
| Boiling point | 5555.0 K | Tungsten boiling point |
| Density | 22.59 g/cm³ | Osmium — densest element |
| Electron affinity | ±3.617 eV | Chlorine maximum — symmetric about zero |
| Period | 7 | Total number of periods |
| Group | 18 | Total number of groups |

### Encoding functions

After normalisation, each property value passes through one of two encoding functions.

**Π-encoding** (for cascade-type properties):

```
φ_Π(x) = (5/11)·sin(π·x)  +  (1/11)·cos(π·x)  +  (1/11)·sin(2π·x)
        + (3/11)·sin(π²·x) +  (1/11)·sin(π·x)·cos(π²·x)
```

The weights (5, 1, 1, 3, 1) are fixed. They are not tuned to the data. The dominant term sin(π·x) carries cascade character. The sin(π²·x) term carries weight 3 because π² is irrational — sin(π²·x) never repeats at any integer period, making it the strongest non-returning basis function available.

**Ε-encoding** (for equilibrium-type properties):

```
φ_E(x) = (2/5)·exp(−e·x)  +  (2/5)·x^e  +  (1/5)·exp(−e·(x − 0.5)²)
```

Weights (2, 2, 1), near-uniform. The base of the natural exponential e = 2.718… governs all three terms. Near-uniform weighting is structurally correct for bounded properties — no single mode dominates.

**Β-encoding** (for boundary-type properties):

```
φ_B(x) = exp(−e·(x − 0.5)²)
```

A Gaussian peaked at x = 0.5. Properties are normalised so that the physical boundary condition maps to x = 0.5. The score is highest when the material sits exactly at the separatrix and falls toward zero at either extreme.

### Aggregation to three scores

Once all available properties are encoded, the scores are averaged within each type:

```
Π = mean of φ_Π(x) across all Π-type properties
Ε = mean of φ_E(x) across all Ε-type properties
Β = mean of φ_B(x) across all Β-type properties
```

Missing properties are skipped. The score is computed from whatever properties are available. An element with fewer known properties still receives a valid fingerprint — with lower resolution, but not an invalid result.

### Search

Given a query (either an element symbol or a partial property set), the classifier computes (Π, Ε, Β) for the query and finds the k nearest elements by Euclidean distance in 3D space:

```
distance = sqrt( (Π_query − Π_target)²  +  (Ε_query − Ε_target)²  +  (Β_query − Β_target)² )
```

Dimensions for which the query has no information (N/A) are excluded from the distance calculation. A query with only Ε-type properties searches in 1D Ε-space. Adding Π or Β properties sharpens the localisation into 2D or 3D.

---

## What the classifier does not do

It does not replace neutron transport calculations, regulatory qualification, or material certification. It does not account for irradiation damage, fabrication constraints, or coolant compatibility. It is a fast structural filter — a way to narrow a large materials space to a small candidate set before detailed analysis begins.

---

## Command reference

```
python phosphorus.py                             # landmark nuclear elements
python phosphorus.py --all                       # all elements with fingerprints
python phosphorus.py --query Fe                  # nearest 5 to iron
python phosphorus.py --query U --k 10            # nearest 10 to uranium
python phosphorus.py --props density=7.87 electronegativity=1.83
python phosphorus.py --props group=17 period=3   # search by periodic position
python phosphorus.py --csv my_elements.csv --query Gd
```

Valid property keys for `--props`: `mass`, `electronegativity`, `ie`, `radius`, `mp`, `bp`, `density`, `ea`, `period`, `group`. Aliases: `atomic_mass`, `eneg`, `ionization_energy`, `atomic_radius`, `melting_point`, `boiling_point`, `electron_affinity`.
