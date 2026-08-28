#!/usr/bin/env python3
"""
Phosphorus Material Classifier — v3 (nuclear scope)
TSA/ISPCC element fingerprinting: (Π, Ε, Β) 3D space.

Default search scope: 38 nuclear-relevant elements covering structural
materials, absorbers, moderators, coolants, and fuels for Gen-IV LFR/SFR.
Use --all to search across all 118 elements.

Changes from v1:
  - φ_e rewritten: monotone decay, range [0.033, 1.0].
  - N/Z normalisation corrected: stability line maps to x=0.5.
  - Period and group stored in db for grid search.
  - Group + period triggers Manhattan distance on periodic table grid.
  - Nuclear subset filter applied by default.

Data source (auto-downloaded on first run):
    PubChem Periodic Table CSV
    https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV

Usage:
    python phosphorus_v3.py                              # nuclear landmark elements
    python phosphorus_v3.py --query Fe                   # nearest in nuclear scope
    python phosphorus_v3.py --query Fe --all             # nearest across all 118
    python phosphorus_v3.py --query U --k 10
    python phosphorus_v3.py --props density=7.87 electronegativity=1.83
    python phosphorus_v3.py --props group=8 period=4
    python phosphorus_v3.py --all                        # fingerprints, all 118
    python phosphorus_v3.py --csv my_elements.csv --query Gd

Requirements: Python 3.7+, standard library only.
"""

import csv
import math
import os
import sys
import urllib.request
import argparse

E  = math.e
PI = math.pi

# ── NUCLEAR ELEMENT SUBSET ────────────────────────────────────────────────────
# 38 elements spanning the materials universe of Gen-IV lead/sodium fast reactors.
# Structural | Absorber | Moderator/Reflector | Coolant | Fuel
#
# Structural:  Al, Si, Ti, V, Cr, Mn, Fe, Co, Ni, Cu, Zr, Nb, Mo, Ag, W
# Absorber:    B, Cd, In, Sm, Eu, Gd, Dy, Er, Hf, Ta, Ir
# Moderator:   H, He, Be, C, O
# Coolant:     Na, Pb, Bi
# Fuel:        Th, U, Pu
# Fission prd: Xe (major neutron poison)

NUCLEAR_Z = {
    1,            # H   — moderator, tritium
    2,            # He  — coolant (gas-cooled), moderator
    4,            # Be  — moderator, reflector
    5,            # B   — absorber, control
    6,            # C   — moderator (graphite)
    8,            # O   — moderator (water/oxide fuel)
    11,           # Na  — coolant (SFR)
    13,           # Al  — structural (low-temp)
    14,           # Si  — structural (ODS steels)
    22,           # Ti  — structural
    23,           # V   — structural (V-alloys)
    24,           # Cr  — structural (steels)
    25,           # Mn  — structural (steels)
    26,           # Fe  — structural (primary)
    27,           # Co  — structural (activation product)
    28,           # Ni  — structural (steels, Ni-alloys)
    29,           # Cu  — structural
    40,           # Zr  — structural (cladding, LWR)
    41,           # Nb  — structural (steels)
    42,           # Mo  — structural (steels, refractory)
    47,           # Ag  — control (Ag-In-Cd rods)
    48,           # Cd  — absorber, control
    49,           # In  — absorber (Ag-In-Cd rods)
    54,           # Xe  — fission product poison (Xe-135)
    62,           # Sm  — absorber poison (Sm-149)
    63,           # Eu  — absorber
    64,           # Gd  — absorber, burnable poison
    66,           # Dy  — burnable poison
    68,           # Er  — burnable poison
    72,           # Hf  — absorber, control rods
    73,           # Ta  — structural, control
    74,           # W   — structural, refractory
    77,           # Ir  — absorber, neutron source
    82,           # Pb  — coolant (LFR/ALFRED)
    83,           # Bi  — coolant (LBE)
    90,           # Th  — fuel (Th-U cycle)
    92,           # U   — fuel (primary)
    94,           # Pu  — fuel (MOX, fast reactor)
}

NUCLEAR_ROLE = {
    1:"moderator",  2:"moderator",  4:"moderator",   5:"absorber",
    6:"moderator",  8:"moderator",  11:"coolant",    13:"structural",
    14:"structural",22:"structural",23:"structural",  24:"structural",
    25:"structural",26:"structural",27:"structural",  28:"structural",
    29:"structural",40:"structural",41:"structural",  42:"structural",
    47:"absorber",  48:"absorber",  49:"absorber",   54:"other",
    62:"absorber",  63:"absorber",  64:"absorber",   66:"absorber",
    68:"absorber",  72:"absorber",  73:"structural",  74:"structural",
    77:"absorber",  82:"coolant",   83:"coolant",    90:"fuel",
    92:"fuel",      94:"fuel",
}

# ── STRUCTURAL CONSTANTS ──────────────────────────────────────────────────────

PI_SCALE_MASS    = 56.0    # u       Iron peak
E_SCALE_ENEG     = 4.0     # —       Fluorine maximum
E_SCALE_IE       = 24.59   # eV      Helium first IE
E_SCALE_RADIUS   = 298.0   # pm      Caesium maximum
E_SCALE_MP       = 3695.0  # K       Tungsten melting point
E_SCALE_BP       = 5555.0  # K       Tungsten boiling point
E_SCALE_DENSITY  = 22.59   # g/cm³   Osmium maximum
B_SCALE_EA       = 3.617   # eV      Chlorine EA
E_SCALE_PERIOD   = 7.0     # —       7 periods
E_SCALE_GROUP    = 18.0    # —       18 groups

# ── TSA PROPERTY SCHEMA ───────────────────────────────────────────────────────

PROP_SCHEMA = [
    # PI5: primary Π input, weight 5 — encoded through phi_pi
    ("mass",              "AtomicMass",         "PI5", PI_SCALE_MASS),
    # PI3: secondary Π input, weight 3 — linear Z/118 (no phi_pi, avoids cancellation)
    ("z",                 "AtomicNumber",        "PI3", 118.0),
    # E: exactly 2 required for query, equal weight (2+2)/4
    ("electronegativity", "Electronegativity",   "E",   E_SCALE_ENEG),
    ("ie",                "IonizationEnergy",    "E",   E_SCALE_IE),
    ("radius",            "AtomicRadius",        "E",   E_SCALE_RADIUS),
    ("mp",                "MeltingPoint",        "E",   E_SCALE_MP),
    ("bp",                "BoilingPoint",        "E",   E_SCALE_BP),
    ("density",           "Density",             "E",   E_SCALE_DENSITY),
    # B: shown in results, not accepted as query input
    ("ea",                "ElectronAffinity",    "B",   B_SCALE_EA),
    ("period",            None,                  "E",   E_SCALE_PERIOD),
    ("group",             None,                  "E",   E_SCALE_GROUP),
    ("nz",                None,                  "B",   None),
]

PROP_ALIASES = {
    "atomic_mass":        "mass",
    "electronegativity":  "electronegativity",
    "eneg":               "electronegativity",
    "ionization_energy":  "ie",
    "ionisation_energy":  "ie",
    "atomic_radius":      "radius",
    "melting_point":      "mp",
    "boiling_point":      "bp",
    "electron_affinity":  "ea",
    "period":             "period",
    "group":              "group",
    "z":                  "z",
    "atomic_number":      "z",
    "atomic_z":           "z",
}

# ── ENCODING FUNCTIONS ────────────────────────────────────────────────────────

def phi_pi(x: float) -> float:
    """
    Π-basis: cascade / non-returning character.
    sin(π²x) term is irrational-period — never repeats. Range ≈ [−0.9, 0.7].
    """
    x = max(0.0, min(float(x), 10.0))
    return (
          (5/11) * math.sin(PI * x)
        + (1/11) * math.cos(PI * x)
        + (1/11) * math.sin(2 * PI * x)
        + (3/11) * math.sin(PI**2 * x)
        + (1/11) * math.sin(PI * x) * math.cos(PI**2 * x)
    )

def phi_e(x: float) -> float:
    """
    Ε-basis: equilibrium / bounded character.
    Monotone decay from 1.0 at x=0 to 0.033 at x=1.
    x=0: property far below its physical ceiling (maximally bounded).
    x=1: property at its physical ceiling (edge of equilibrium regime).
    Range ≈ [0.033, 1.0] — 6× wider than v1's [0.36, 0.53].
    """
    x = max(0.0, min(float(x), 1.0))
    return (math.exp(-E * x) + (1.0 - x) ** E) / 2.0

def phi_b(x: float) -> float:
    """
    Β-basis: boundary / separatrix character.
    Gaussian peaked at x=0.5. Score = 1 at the separatrix, decays to ~0.5
    at x=0 or x=1.
    """
    x = max(0.0, min(float(x), 1.0))
    return math.exp(-E * (x - 0.5) ** 2)

# ── DATA LOADING ──────────────────────────────────────────────────────────────

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV"
LOCAL_CSV   = "elements_pubchem.csv"

def load_rows(path: str = None) -> list:
    target = path if (path and os.path.exists(path)) else LOCAL_CSV
    if not os.path.exists(target):
        print(f"Downloading periodic table → {LOCAL_CSV}")
        urllib.request.urlretrieve(PUBCHEM_URL, LOCAL_CSV)
        print("Saved.\n")
    with open(target, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def fget(row: dict, key: str):
    v = row.get(key, "").strip()
    if not v or v.lower() in ("n/a", "na", "none", "-", "unknown", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ── PERIODIC TABLE GEOMETRY ───────────────────────────────────────────────────

def period_and_group(Z: int):
    if Z == 1:   return 1, 1
    if Z == 2:   return 1, 18
    if Z <= 4:   return 2, Z - 2
    if Z <= 10:  return 2, Z + 8
    if Z <= 12:  return 3, Z - 10
    if Z <= 18:  return 3, Z
    if Z <= 36:  return 4, Z - 18
    if Z <= 54:  return 5, Z - 36
    if Z <= 56:  return 6, Z - 54
    if Z <= 71:  return 6, 3
    if Z <= 86:  return 6, Z - 68
    if Z <= 88:  return 7, Z - 86
    if Z <= 103: return 7, 3
    return       7, Z - 100

def nz_stability_line(Z: int) -> float:
    if Z <= 20:
        return 1.0
    return 1.0 + 0.4 * (Z - 20) / (83 - 20)

# ── FINGERPRINT ───────────────────────────────────────────────────────────────

def compute_fp(row: dict) -> dict:
    """
    Compute (Π, Ε, Β) fingerprint.
    Π: weighted combination — mass at weight 5 (phi_pi), Z at weight 3 (linear).
    Ε: mean of all available Ε properties.
    Β: shown in results, not used in search distance.
    """
    Z    = fget(row, "AtomicNumber")
    mass = fget(row, "AtomicMass")

    # ── Π: (5 · phi_pi(mass/56) + 3 · Z/118) / 8 ────────────────────────────
    # mass dominates (weight 5) — iron peak normalization preserves nuclear meaning.
    # Z/118 is linear (weight 3) — avoids phi_pi cancellation for heavy elements,
    # slots out when Granta provides thermal conductivity / neutron cross-section.
    if mass is not None and Z is not None and int(Z) > 0:
        pi_score = (5.0 * phi_pi(mass / PI_SCALE_MASS) + 3.0 * float(Z) / 118.0) / 8.0
        n_pi = 2
    else:
        pi_score = None
        n_pi = 0

    # ── Ε: mean of all available equilibrium properties ───────────────────────
    e_v = []
    for (key, col, tsa_type, scale) in PROP_SCHEMA:
        if col is None or tsa_type != "E":
            continue
        v = fget(row, col)
        if v is None:
            continue
        e_v.append(phi_e(v / scale))
    if Z is not None and int(Z) > 0:
        period, group = period_and_group(int(Z))
        e_v.append(phi_e(period / E_SCALE_PERIOD))
        e_v.append(phi_e(group  / E_SCALE_GROUP))

    # ── Β: shown in results, not used in search distance ─────────────────────
    b_v = []
    ea = fget(row, "ElectronAffinity")
    if ea is not None:
        x_ea = (ea + B_SCALE_EA) / (2.0 * B_SCALE_EA)
        b_v.append(phi_b(max(0.0, min(x_ea, 1.0))))
    if Z is not None and mass is not None and int(Z) > 0:
        N = round(mass) - int(Z)
        if N > 0:
            nz  = N / int(Z)
            dev = abs(nz - nz_stability_line(int(Z)))
            x_nz = max(0.0, 0.5 - dev)
            b_v.append(phi_b(x_nz))

    return {
        "PI":   pi_score,
        "E":    sum(e_v) / len(e_v) if e_v else None,
        "B":    sum(b_v) / len(b_v) if b_v else None,
        "n_pi": n_pi,
        "n_e":  len(e_v),
        "n_b":  len(b_v),
    }

def build_db(rows: list) -> list:
    """Build fingerprint db — nuclear subset only (NUCLEAR_Z elements)."""
    db = []
    for row in rows:
        Z = fget(row, "AtomicNumber")
        if Z is None:
            continue
        Zi = int(Z)
        if Zi not in NUCLEAR_Z:
            continue
        fp = compute_fp(row)
        period, group = period_and_group(Zi)
        db.append({
            "Symbol": row.get("Symbol", "?").strip(),
            "Name":   row.get("Name",   "?").strip(),
            "Z":      Zi,
            "period": period,
            "group":  group,
            "role":   NUCLEAR_ROLE.get(Zi, "other"),
            **fp,
        })
    return sorted(db, key=lambda el: el["Z"])

# ── SEARCH ────────────────────────────────────────────────────────────────────

def tsa_dist(a: dict, b: dict) -> float:
    """
    Euclidean distance in Π × Ε space.
    Β is shown in results but not used in search distance —
    it's a diagnostic, not a search coordinate.
    """
    sq, n = 0.0, 0
    for dim in ("PI", "E"):
        va, vb = a.get(dim), b.get(dim)
        if va is not None and vb is not None:
            sq += (va - vb) ** 2
            n  += 1
    return math.sqrt(sq / n) if n > 0 else float("inf")

def find_nearest(target: dict, db: list, k: int = 5, exclude: str = None) -> list:
    scored = [
        (tsa_dist(target, el), el)
        for el in db
        if exclude is None or el["Symbol"].upper() != exclude.upper()
    ]
    scored.sort(key=lambda x: x[0])
    return scored[:k]

def find_by_grid(g: int, p: int, db: list, k: int = 5, exclude: str = None) -> list:
    """Manhattan distance on periodic table grid: |Δgroup| + |Δperiod|."""
    scored = []
    for el in db:
        if exclude and el["Symbol"].upper() == exclude.upper():
            continue
        eg, ep = el.get("group"), el.get("period")
        if eg is None or ep is None:
            continue
        scored.append((abs(g - eg) + abs(p - ep), el))
    scored.sort(key=lambda x: (x[0], x[1]["Z"]))
    return scored[:k]

# ── DISPLAY ───────────────────────────────────────────────────────────────────

def fv(v):
    return f"{v:>7.4f}" if v is not None else "  N/A  "

def show_element(el: dict):
    n    = el.get("n_pi", 0) + el.get("n_e", 0) + el.get("n_b", 0)
    role = el.get("role", "")
    print(f"\n  ┌─ {el['Symbol']:<4} {el['Name']:<20} Z={el['Z']:<4}  "
          f"period={el.get('period','?')}  group={el.get('group','?')}  "
          f"[{role}]  [{n} props]")
    print(f"  │  Π (cascade)     {fv(el['PI'])}  ← search")
    print(f"  │  Ε (equilibrium) {fv(el['E'])}  ← search")
    print(f"  └  Β (boundary)    {fv(el['B'])}  [shown · not in distance]")

def show_table(results: list, header: str, mode: str = "TSA distance"):
    W = 76
    print(f"\n  {header}  [{mode}]")
    print(f"  {'─'*W}")
    print(f"  {'Sym':<5} {'Name':<18} {'Z':>3}  {'Role':<12} {'Π':>7}  {'Ε':>7}  {'Β':>7}  {'Dist':>8}")
    print(f"  {'─'*W}")
    for dist, el in results:
        print(f"  {el['Symbol']:<5} {el['Name']:<18} {el['Z']:>3}  "
              f"{el.get('role',''):<12} "
              f"{fv(el['PI'])}  {fv(el['E'])}  {fv(el['B'])}  {dist:>8.4f}")
    print(f"  {'─'*W}\n")

def show_grid_table(results: list, header: str):
    W = 80
    print(f"\n  {header}  [Manhattan distance]")
    print(f"  {'─'*W}")
    print(f"  {'Sym':<5} {'Name':<18} {'Z':>3}  {'Role':<12} {'Grp':>4}  {'Per':>4}  "
          f"{'Π':>7}  {'Ε':>7}  {'Β':>7}  {'ΔMhtn':>6}")
    print(f"  {'─'*W}")
    for dist, el in results:
        print(f"  {el['Symbol']:<5} {el['Name']:<18} {el['Z']:>3}  "
              f"{el.get('role',''):<12} "
              f"{el.get('group',0):>4}  {el.get('period',0):>4}  "
              f"{fv(el['PI'])}  {fv(el['E'])}  {fv(el['B'])}  {int(dist):>6}")
    print(f"  {'─'*W}\n")

def show_list(db: list):
    """Print fingerprints for all elements in the nuclear subset."""
    print(f"\n  Phosphorus v3 — nuclear subset ({len(db)} elements)")
    print(f"\n  {'Sym':<5} {'Name':<18} {'Z':>3}  {'Role':<12} {'Π':>7}  {'Ε':>7}  {'Β':>7}  props")
    print(f"  {'─'*75}")
    for el in db:
        n = el["n_pi"] + el["n_e"] + el["n_b"]
        print(f"  {el['Symbol']:<5} {el['Name']:<18} {el['Z']:>3}  "
              f"{el.get('role',''):<12} "
              f"{fv(el['PI'])}  {fv(el['E'])}  {fv(el['B'])}  {n}")
    print()

# ── PROPERTY QUERY ────────────────────────────────────────────────────────────

def props_to_fp(props: dict) -> dict:
    """
    Build query fingerprint from user-supplied properties.
    Π = (5·phi_pi(mass/56) + 3·z/118) / 8   — weighted 5:3
    Ε = (phi_e(x1) + phi_e(x2)) / 2          — equal weight 2:2
    Β = None                                   — not used in search
    """
    pi5_v = []   # mass → phi_pi, weight 5
    pi3_v = []   # z    → linear,  weight 3
    e_v   = []

    schema_by_key = {row[0]: row for row in PROP_SCHEMA}

    for raw_key, val in props.items():
        key   = PROP_ALIASES.get(raw_key.lower(), raw_key.lower())
        entry = schema_by_key.get(key)
        if entry is None:
            print(f"  Unknown property '{raw_key}'.")
            continue
        _, col, tsa_type, scale = entry
        if tsa_type == "PI5":
            pi5_v.append(phi_pi(val / scale))
        elif tsa_type == "PI3":
            pi3_v.append(val / scale)          # linear, no phi_pi
        elif tsa_type == "E":
            if col is not None:
                e_v.append(phi_e(val / scale))
            elif key == "period":
                e_v.append(phi_e(val / E_SCALE_PERIOD))
            elif key == "group":
                e_v.append(phi_e(val / E_SCALE_GROUP))
        # B: rejected at validation — never reaches here

    pi_score = (5.0 * pi5_v[0] + 3.0 * pi3_v[0]) / 8.0 \
               if pi5_v and pi3_v else None
    e_score  = sum(e_v) / len(e_v) if e_v else None

    return {"PI": pi_score, "E": e_score, "B": None}

# ── MAIN ──────────────────────────────────────────────────────────────────────

LANDMARKS = ["H", "Be", "B", "C", "Na", "Fe", "Ni", "Zr", "Mo", "Gd", "W", "Pb", "U"]

def main():
    ap = argparse.ArgumentParser(
        description="Phosphorus v3 — TSA nuclear material classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scope: nuclear subset only (38 elements). Non-nuclear elements are not loaded.

Search modes:
  --query SYMBOL            TSA fingerprint nearest neighbours
  --props KEY=VAL ...       TSA fingerprint from partial properties
  --props group=G period=P  Manhattan distance on periodic table grid
  --list                    Print fingerprints for all 38 nuclear elements

Examples:
  python phosphorus_v3.py --query Fe
  python phosphorus_v3.py --query U --k 8
  python phosphorus_v3.py --props density=7.87 electronegativity=1.83
  python phosphorus_v3.py --props density=7.87 electronegativity=1.83 mass=55.84
  python phosphorus_v3.py --props group=8 period=4
  python phosphorus_v3.py --list
  python phosphorus_v3.py --csv granta_export.csv --query Fe
        """
    )
    ap.add_argument("--csv",   default=None,  help="Elements CSV (default: PubChem auto-download)")
    ap.add_argument("--query", default=None,  help="Element symbol, e.g. Fe")
    ap.add_argument("--k",     default=5, type=int, help="Neighbours to return (default: 5)")
    ap.add_argument("--list",  action="store_true", help="List fingerprints for all 38 nuclear elements")
    ap.add_argument("--props", nargs="+", metavar="KEY=VAL",
                    help="Property query. group + period → Manhattan grid search.")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    db   = build_db(rows)           # always nuclear subset — 38 elements maximum

    if len(db) == 0:
        print("No nuclear elements found in CSV. Check that AtomicNumber column is present.")
        sys.exit(1)

    # ── --list ────────────────────────────────────────────────────────────────
    if args.list:
        show_list(db)
        return

    # ── --query ───────────────────────────────────────────────────────────────
    if args.query:
        sym   = args.query.strip()
        match = next((el for el in db if el["Symbol"].lower() == sym.lower()), None)
        if match is None:
            available = ", ".join(el["Symbol"] for el in db)
            print(f"'{sym}' not in nuclear subset. Available: {available}")
            sys.exit(1)
        show_element(match)
        results = find_nearest(match, db, k=args.k, exclude=match["Symbol"])
        show_table(results,
                   header=f"Nearest {args.k} to {match['Name']} ({match['Symbol']})",
                   mode=f"TSA distance · {len(db)} nuclear elements")
        return

    # ── --props ───────────────────────────────────────────────────────────────
    if args.props:
        try:
            props = {}
            for item in args.props:
                k_str, v_str = item.split("=", 1)
                props[k_str.strip()] = float(v_str.strip())
        except (ValueError, IndexError) as exc:
            print(f"Error parsing --props: {exc}")
            sys.exit(1)

        normalized = {PROP_ALIASES.get(k.lower(), k.lower()): v for k, v in props.items()}

        # Grid branch
        if "group" in normalized and "period" in normalized:
            tg, tp = int(round(normalized["group"])), int(round(normalized["period"]))
            print(f"\n  Grid query  group={tg}  period={tp}  [{len(db)} nuclear elements]")
            results = find_by_grid(tg, tp, db, k=args.k)
            show_grid_table(results, header=f"Nearest {args.k} to (group={tg}, period={tp})")
            return

        # TSA branch
        fp = props_to_fp(props)

        # ── Structural query validation ────────────────────────────────────────
        # Valid query: exactly 1 PI5 (mass) + 1 PI3 (z) + 2 E properties, 0 B.
        # Enforces the (5:3) Π weighting and (2:2) Ε equal-weighting structure.
        schema_type = {key: tsa_type for (key, _, tsa_type, _) in PROP_SCHEMA}
        for alias, key in PROP_ALIASES.items():
            if key in schema_type:
                schema_type[alias] = schema_type[key]

        type_keys = {"PI5": [], "PI3": [], "E": [], "B": []}
        for raw_key in props:
            nkey = PROP_ALIASES.get(raw_key.lower(), raw_key.lower())
            t    = schema_type.get(nkey)
            if t in type_keys:
                type_keys[t].append(raw_key)

        E_OPTS  = "electronegativity, ie, radius, mp, bp, density, period, group"
        errors  = []
        if len(type_keys["PI5"]) != 1:
            errors.append( "  Π primary  (weight 5): exactly one of: mass")
        if len(type_keys["PI3"]) != 1:
            errors.append( "  Π secondary (weight 3): exactly one of: z, atomic_number")
        if len(type_keys["E"]) != 2:
            errors.append(f"  Ε (weight 2+2): exactly two of: {E_OPTS}"
                          f"  (got {len(type_keys['E'])})")
        if type_keys["B"]:
            errors.append(f"  Β properties not accepted in query: "
                          f"{', '.join(type_keys['B'])}")

        if errors:
            print(f"\n  Error: query must be exactly mass + z + 2 Ε properties.")
            for e in errors:
                print(e)
            print(f"\n  Example for uranium:")
            print(f"    --props mass=238.03 z=92 electronegativity=1.38 density=18.95")
            sys.exit(1)

        print(f"\n  Query fingerprint   Π={fv(fp['PI'])}  Ε={fv(fp['E'])}"
              f"  Β=N/A  [{len(db)} nuclear elements]")
        results = find_nearest(fp, db, k=args.k)
        show_table(results, header=f"Nearest {args.k} to property query",
                   mode="Π × Ε distance")
        return

    # ── default: landmark elements ────────────────────────────────────────────
    print(f"\n  Phosphorus v3 — nuclear landmark elements  [{len(db)} elements in scope]")
    for sym in LANDMARKS:
        el = next((e for e in db if e["Symbol"] == sym), None)
        if el:
            show_element(el)
    print()

if __name__ == "__main__":
    main()