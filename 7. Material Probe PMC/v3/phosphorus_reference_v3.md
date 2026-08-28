# Phosphorus Material Classifier — Reference (v3)

## What this tool does

Given a set of material properties, the classifier places the material at a point in a 2D search space and finds the closest known elements from a curated nuclear subset. The two search coordinates are **Π** (cascade character) and **Ε** (equilibrium character). A third coordinate **Β** (boundary character) is computed and shown in every result but is not used in the search distance — it is a diagnostic.

The classifier operates on 38 nuclear-relevant elements only. Non-nuclear elements are not loaded.

---

## The three coordinates

### Π — Cascade character

A material with high Π score has properties that are large in scale, non-periodic, and accumulating — far from the nuclear binding energy centre and participating in processes that are irreversible and directional.

In nuclear terms: heavy structural metals, fuel elements (U, Th, Pu), and materials with large nuclear cross-sections tend toward high Π. Light moderators (H, Be, C) sit at different Π positions depending on their mass encoding.

### Ε — Equilibrium character

A material with high Ε score has properties that are bounded, self-regulating, and periodic — thermodynamically moderate, structurally stable. The score is highest when all normalised properties are far below their physical maximums.

In nuclear terms: light moderators and low-period structural materials have high Ε. Heavy refractory metals (W, Mo) and dense actinides have low Ε because their thermodynamic properties approach their physical ceilings.

### Β — Boundary character

A material with high Β score sits at a structural separatrix — near the valley of nuclear stability, or at the electronic boundary between metal and non-metal character. The score peaks at the separatrix and falls toward zero at either extreme.

**Β is shown in all outputs but is not used in search distance.** It is a secondary diagnostic: read it from the results table to assess how close a candidate is to a controlled boundary condition. When nuclear cross-section data is added in a future version, Β will re-enter the distance calculation.

---

## Nuclear role reference

| Role | Π | Ε | Β | Examples |
|---|---|---|---|---|
| Neutron absorber | Mid–High | Low–Mid | Mid | Gd, B, Cd, Hf, Sm, Eu, Dy, Er |
| Neutron moderator | Low–Mid | High | Mid | H, Be, C, Na |
| Structural | Low–Mid | Mid | Mid–High | Fe, Ni, Cr, Zr, Mo, W, Ti, V, Nb |
| Fuel | High | Low | High | U, Th, Pu |
| Coolant | Low–Mid | Mid | Mid | Pb, Bi, Na |

These are indicative. Role depends on geometry, temperature, and neutron spectrum. The fingerprint is a fast first filter, not a qualification.

---

## How the coordinates are computed

### Property types and roles

| Property | Type | Weight | Physical reasoning |
|---|---|---|---|
| Atomic mass | Π | 5 | Cascade: increases monotonically, normalised by iron peak |
| Atomic number Z | Π | 3 | Cascade: linear, separates elements that share a mass encoding |
| Electronegativity | Ε | equal | Bounded [0.7–4.0], periodic — equilibrium |
| First ionisation energy | Ε | equal | Bounded, resets at noble gases — equilibrium |
| Atomic radius | Ε | equal | Bounded, periodic within each shell — equilibrium |
| Melting point | Ε | equal | Bounded by W at 3695 K — equilibrium |
| Boiling point | Ε | equal | Bounded by W at 5555 K — equilibrium |
| Density | Ε | equal | Bounded by Os at 22.59 g/cm³ — equilibrium |
| Period (1–7) | Ε | equal | Bounded shell count — equilibrium |
| Group (1–18) | Ε | equal | Bounded valence column — equilibrium |
| Electron affinity | Β | — | Zero-crossing boundary; shown only, not queried |
| N/Z ratio (computed) | Β | — | Valley of stability separatrix; shown only, not queried |

### Normalisation constants

| Property | Constant | Source |
|---|---|---|
| Atomic mass | 56.0 u | Iron peak — nuclear binding energy maximum |
| Atomic number | 118 | Total known elements |
| Electronegativity | 4.0 | Fluorine Pauling maximum |
| Ionisation energy | 24.59 eV | Helium first IE |
| Atomic radius | 298.0 pm | Caesium maximum |
| Melting point | 3695.0 K | Tungsten maximum |
| Boiling point | 5555.0 K | Tungsten maximum |
| Density | 22.59 g/cm³ | Osmium maximum |
| Electron affinity | ±3.617 eV | Chlorine maximum, symmetric about zero |
| Period | 7 | Total periods |
| Group | 18 | Total groups |

### Encoding functions

**Π-encoding** (cascade character — applied to atomic mass):

```
φ_Π(x) = (5/11)·sin(π·x)  +  (1/11)·cos(π·x)  +  (1/11)·sin(2π·x)
        + (3/11)·sin(π²·x) +  (1/11)·sin(π·x)·cos(π²·x)
```

sin(π²·x) carries weight 3 because π² is irrational — it never repeats at any integer period.

**Ε-encoding** (equilibrium character — applied to all Ε-type properties):

```
φ_E(x) = ( exp(−e·x) + (1 − x)^e ) / 2
```

Monotone decay from 1.0 at x=0 to 0.033 at x=1. Higher score means the property is further below its physical ceiling — more bounded, more self-regulating.

**Β-encoding** (boundary character):

```
φ_B(x) = exp(−e·(x − 0.5)²)
```

Gaussian peaked at x=0.5 (the separatrix). For N/Z: the valley of stability maps to x=0.5. For electron affinity: EA=0 maps to x=0.5.

### Aggregation

**Π score** (weighted 5:3):
```
Π = (5 · φ_Π(mass/56) + 3 · Z/118) / 8
```

Atomic mass passes through φ_Π (cascade encoding). Z is added linearly at weight 3 to break degeneracy cases where different masses encode to the same φ_Π value. When Granta data provides neutron cross-section or thermal conductivity, Z is replaced by that property in the weight-3 slot.

**Ε score** (equal weight):
```
Ε = mean of φ_E(x) across all available Ε-type properties
```

**Β score** (shown, not searched):
```
Β = mean of φ_B(x_ea) and φ_B(x_nz)  [where available]
```

x_nz = max(0, 0.5 − |nz − nz_stability(Z)|)

### Search distance

Distance is computed over Π and Ε only:

```
distance = sqrt( ((Π_query − Π_target)² + (Ε_query − Ε_target)²) / 2 )
```

Β is excluded from the distance. It is shown in every result row for diagnostic reading.

---

## Query rules

Property queries must supply **exactly**:
- 1 Π primary property: `mass`
- 1 Π secondary property: `z` (or `atomic_number`)
- 2 Ε properties (any two from the Ε list)
- 0 Β properties (not accepted as query input)

Supplying fewer or more than these counts is rejected with an error that names the missing or excess properties and gives a corrected example.

The group + period combination triggers a different search mode — Manhattan distance on the periodic table grid — instead of the TSA fingerprint distance. This is indicated in the output header.

---

## What the classifier does not do

It does not replace neutron transport calculations, regulatory qualification, or material certification. It does not account for irradiation damage, fabrication constraints, or coolant compatibility. It is a fast structural filter — a way to narrow a large materials space to a small candidate set before detailed analysis begins.

The current property set (bulk elemental properties from PubChem) does not include thermal conductivity, specific heat capacity, or neutron cross-section. These are the properties that will most improve nuclear-specific discrimination and are planned for the next version using Granta institute data.

---

## Command reference

```
python phosphorus_v3.py                    # landmark nuclear elements (default)
python phosphorus_v3.py --list             # all 38 nuclear elements with fingerprints
python phosphorus_v3.py --query Fe         # nearest 5 to iron
python phosphorus_v3.py --query U --k 8   # nearest 8 to uranium

# Property query — requires exactly mass + z + 2 E properties
python phosphorus_v3.py --props mass=238.03 z=92 electronegativity=1.38 density=18.95

# Grid search — group + period triggers Manhattan distance
python phosphorus_v3.py --props group=8 period=4

python phosphorus_v3.py --csv granta_export.csv --query Fe
```

**Valid Π keys:** `mass` (or `atomic_mass`), `z` (or `atomic_number`)

**Valid Ε keys:** `electronegativity` (`eneg`), `ie` (`ionization_energy`), `radius` (`atomic_radius`), `mp` (`melting_point`), `bp` (`boiling_point`), `density`, `period`, `group`

**Β keys are not accepted in queries.** Β is read from the results table.
