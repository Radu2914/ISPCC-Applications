#!/usr/bin/env python3
"""
Phosphorus Material Classifier
TSA/ISPCC element fingerprinting: collapses property space to (Π, Ε, Β) 3D space.
Query by symbol or by partial property set — returns nearest neighbours in TSA space.

Data source (auto-downloaded on first run):
    PubChem Periodic Table CSV
    https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV
    Covers: mass, electronegativity, IE, EA, radius, MP, BP, density — all 118 elements.

    For nuclear cross-sections: pip install mendeleev
    (thermal_neutron_cross_section, nuclear_magnetic_moment, etc. — extend PROP_MAP below)

Usage:
    python phosphorus_v1.py                             # landmark elements
    python phosphorus_v1.py --all                       # all elements + TSA fingerprint
    python phosphorus_v1.py --query Fe                  # nearest 5 to iron in TSA space
    python phosphorus_v1.py --query U --k 10            # nearest 10 to uranium
    python phosphorus_v1.py --props density=7.87 electronegativity=1.83
    python phosphorus_v1.py --csv my_elements.csv --query Gd

Requirements: Python 3.7+, standard library only.
"""

import csv
import math
import os
import sys
import urllib.request
import argparse

E  = math.e    # 2.71828…
PI = math.pi   # 3.14159…

# ── STRUCTURAL CONSTANTS ──────────────────────────────────────────────────────
# All scales from physics. Not from data statistics.

PI_SCALE_MASS    = 56.0    # u       Iron peak — nuclear Ε-attractor, natural mass anchor
E_SCALE_ENEG     = 4.0     # —       Fluorine Pauling maximum
E_SCALE_IE       = 24.59   # eV      Helium first IE — noble gas ceiling
E_SCALE_RADIUS   = 298.0   # pm      Caesium — largest confirmed atomic radius
E_SCALE_MP       = 3695.0  # K       Tungsten melting point — elemental maximum
E_SCALE_BP       = 5555.0  # K       Tungsten boiling point — elemental maximum
E_SCALE_DENSITY  = 22.59   # g/cm³   Osmium — densest element
B_SCALE_EA       = 3.617   # eV      Chlorine EA — symmetric zero-point for boundary encoding
E_SCALE_PERIOD   = 7.0     # —       7 periods (n=1…7), natural electron-shell ceiling
E_SCALE_GROUP    = 18.0    # —       18 groups, natural valence ceiling

# ── TSA PROPERTY SCHEMA ───────────────────────────────────────────────────────
# Π-type: cascade/non-returning across element space
# Ε-type: bounded, periodic, self-regulating
# Β-type: boundary / separatrix (zero-crossing, stability line)

#   (user key,            CSV column,          TSA type,  scale)
PROP_SCHEMA = [
    ("mass",              "AtomicMass",         "PI",  PI_SCALE_MASS),
    ("electronegativity", "Electronegativity",   "E",   E_SCALE_ENEG),
    ("ie",                "IonizationEnergy",    "E",   E_SCALE_IE),
    ("radius",            "AtomicRadius",        "E",   E_SCALE_RADIUS),
    ("mp",                "MeltingPoint",        "E",   E_SCALE_MP),
    ("bp",                "BoilingPoint",        "E",   E_SCALE_BP),
    ("density",           "Density",             "E",   E_SCALE_DENSITY),
    ("ea",                "ElectronAffinity",    "B",   B_SCALE_EA),
    # Computed from Z — not CSV columns
    ("period",            None,                  "E",   E_SCALE_PERIOD),
    ("group",             None,                  "E",   E_SCALE_GROUP),
    ("nz",                None,                  "B",   None),
]

# Aliases so users can type natural names on --props
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
}

# ── ENCODING FUNCTIONS ────────────────────────────────────────────────────────

def phi_pi(x: float) -> float:
    """π-basis: cascade character. Weights (5,1,1,3,1)/11."""
    x = max(0.0, min(float(x), 10.0))
    return (
          (5/11) * math.sin(PI * x)
        + (1/11) * math.cos(PI * x)
        + (1/11) * math.sin(2 * PI * x)
        + (3/11) * math.sin(PI**2 * x)
        + (1/11) * math.sin(PI * x) * math.cos(PI**2 * x)
    )

def phi_e(x: float) -> float:
    """e-basis: equilibrium character. Weights (2,2,1)/5."""
    x = max(0.0, min(float(x), 1.0))
    return (
          (2/5) * math.exp(-E * x)
        + (2/5) * (x ** E if x > 0.0 else 0.0)
        + (1/5) * math.exp(-E * (x - 0.5) ** 2)
    )

def phi_b(x: float) -> float:
    """Β-basis: boundary condition. Gaussian peaked at x=0.5 (the separatrix)."""
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
    """Safe float extraction. Returns None if missing or non-numeric."""
    v = row.get(key, "").strip()
    if not v or v.lower() in ("n/a", "na", "none", "-", "unknown", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None

# ── FINGERPRINT ───────────────────────────────────────────────────────────────

# Period and group from Z — derived, no CSV column needed.
# Lanthanides (57-71) and actinides (89-103) assigned group 3 (conventional).

def period_and_group(Z: int):
    """Return (period, group) for element Z."""
    if Z == 1:  return 1, 1
    if Z == 2:  return 1, 18
    if Z <= 4:  return 2, Z - 2          # Li=1, Be=2
    if Z <= 10: return 2, Z + 8          # B=13 … Ne=18
    if Z <= 12: return 3, Z - 10         # Na=1, Mg=2
    if Z <= 18: return 3, Z              # Al=13 … Ar=18
    if Z <= 36: return 4, Z - 18         # K=1, Ca=2, Sc=3 … Kr=18
    if Z <= 54: return 5, Z - 36         # Rb=1 … Xe=18
    if Z <= 56: return 6, Z - 54         # Cs=1, Ba=2
    if Z <= 71: return 6, 3              # lanthanides
    if Z <= 86: return 6, Z - 68         # Hf=4 … Rn=18
    if Z <= 88: return 7, Z - 86         # Fr=1, Ra=2
    if Z <= 103: return 7, 3             # actinides
    return      7, Z - 100              # Rf=4 … Og=18

def nz_stability_line(Z: int) -> float:
    """Valley-of-stability N/Z approximation. Light elements N≈Z; heavy N≈1.4*Z."""
    if Z <= 20:
        return 1.0
    return 1.0 + 0.4 * (Z - 20) / (83 - 20)

def compute_fp(row: dict) -> dict:
    """
    Collapse element properties → (PI, E, B) TSA fingerprint.
    Each score = mean of phi_* values across all available properties of that type.
    Missing properties are silently skipped.
    """
    pi_v, e_v, b_v = [], [], []

    Z    = fget(row, "AtomicNumber")
    mass = fget(row, "AtomicMass")
    if mass is not None:
        pi_v.append(phi_pi(mass / PI_SCALE_MASS))

    for (key, col, tsa_type, scale) in PROP_SCHEMA:
        if col is None or tsa_type == "PI":
            continue   # handled above or computed separately

        v = fget(row, col)
        if v is None:
            continue

        if tsa_type == "E":
            e_v.append(phi_e(v / scale))

        elif tsa_type == "B" and key == "ea":
            # ElectronAffinity: symmetric normalisation so EA=0 maps to x=0.5
            x = (v + B_SCALE_EA) / (2 * B_SCALE_EA)
            b_v.append(phi_b(max(0.0, min(x, 1.0))))

    # Period and group (Ε-type: computed from Z)
    if Z is not None and int(Z) > 0:
        period, group = period_and_group(int(Z))
        e_v.append(phi_e(period / E_SCALE_PERIOD))
        e_v.append(phi_e(group  / E_SCALE_GROUP))

    # N/Z stability ratio (Β-type: computed from Z and mass)
    if Z is not None and mass is not None and int(Z) > 0:
        N = round(mass) - int(Z)
        if N > 0:
            nz = N / int(Z)
            dev = abs(nz - nz_stability_line(int(Z)))
            x_nz = 1.0 - min(dev / 0.5, 1.0)   # 1.0 = on stability line (maximum Β)
            b_v.append(phi_b(x_nz))

    return {
        "PI":   sum(pi_v) / len(pi_v) if pi_v else None,
        "E":    sum(e_v)  / len(e_v)  if e_v  else None,
        "B":    sum(b_v)  / len(b_v)  if b_v  else None,
        "n_pi": len(pi_v),
        "n_e":  len(e_v),
        "n_b":  len(b_v),
    }

def build_db(rows: list) -> list:
    db = []
    for row in rows:
        Z = fget(row, "AtomicNumber")
        if Z is None:
            continue
        fp = compute_fp(row)
        db.append({
            "Symbol": row.get("Symbol", "?").strip(),
            "Name":   row.get("Name",   "?").strip(),
            "Z":      int(Z),
            **fp,
        })
    return sorted(db, key=lambda el: el["Z"])

# ── SEARCH ────────────────────────────────────────────────────────────────────

def tsa_dist(a: dict, b: dict) -> float:
    """Euclidean distance in (Π, Ε, Β) space. NaN-safe: missing dims are skipped."""
    sq, n = 0.0, 0
    for dim in ("PI", "E", "B"):
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

# ── DISPLAY ───────────────────────────────────────────────────────────────────

def fv(v):
    return f"{v:.4f}" if v is not None else " N/A  "

def show_element(el: dict):
    n = el.get("n_pi", 0) + el.get("n_e", 0) + el.get("n_b", 0)
    print(f"\n  ┌─ {el['Symbol']:<4} {el['Name']:<20} Z={el['Z']:<4}  [{n} props]")
    print(f"  │  Π (cascade)     {fv(el['PI'])}")
    print(f"  │  Ε (equilibrium) {fv(el['E'])}")
    print(f"  └  Β (boundary)    {fv(el['B'])}")

def show_table(results: list, header: str):
    W = 66
    print(f"\n  {header}")
    print(f"  {'─'*W}")
    print(f"  {'Sym':<5} {'Name':<18} {'Z':>3}  {'Π':>7}  {'Ε':>7}  {'Β':>7}  {'Dist':>8}")
    print(f"  {'─'*W}")
    for dist, el in results:
        print(f"  {el['Symbol']:<5} {el['Name']:<18} {el['Z']:>3}  "
              f"{fv(el['PI'])}  {fv(el['E'])}  {fv(el['B'])}  {dist:>8.4f}")
    print(f"  {'─'*W}\n")

# ── PROPERTY QUERY ────────────────────────────────────────────────────────────

def props_to_fp(props: dict) -> dict:
    """Build a TSA fingerprint from a user-supplied partial property set."""
    pi_v, e_v, b_v = [], [], []

    schema_by_key = {row[0]: row for row in PROP_SCHEMA}

    for raw_key, val in props.items():
        key = PROP_ALIASES.get(raw_key.lower(), raw_key.lower())
        entry = schema_by_key.get(key)
        if entry is None:
            valid = list(schema_by_key.keys()) + list(PROP_ALIASES.keys())
            print(f"  Unknown property '{raw_key}'. Valid: {sorted(set(valid))}")
            continue
        _, col, tsa_type, scale = entry
        if col is None:
            # computed properties supplied directly by the user
            if key == "period":
                e_v.append(phi_e(val / E_SCALE_PERIOD))
            elif key == "group":
                e_v.append(phi_e(val / E_SCALE_GROUP))
            continue
        if tsa_type == "PI":
            pi_v.append(phi_pi(val / scale))
        elif tsa_type == "E":
            e_v.append(phi_e(val / scale))
        elif tsa_type == "B":
            x = (val + B_SCALE_EA) / (2 * B_SCALE_EA)
            b_v.append(phi_b(max(0.0, min(x, 1.0))))

    return {
        "PI": sum(pi_v) / len(pi_v) if pi_v else None,
        "E":  sum(e_v)  / len(e_v)  if e_v  else None,
        "B":  sum(b_v)  / len(b_v)  if b_v  else None,
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────

# Nuclear-relevant landmarks: moderators, absorbers, structural, fuel
LANDMARKS = ["H", "He", "B", "C", "O", "Al", "Fe", "Ni", "Zr", "Mo", "Gd", "W", "Pb", "U"]

def main():
    ap = argparse.ArgumentParser(
        description="Phosphorus Material Classifier — TSA element fingerprinting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python phosphorus.py --query Fe
  python phosphorus.py --query U --k 10
  python phosphorus.py --props density=7.87 electronegativity=1.83
  python phosphorus.py --all
  python phosphorus.py --csv my_elements.csv --query Gd
        """
    )
    ap.add_argument("--csv",   default=None,  help="Path to elements CSV (default: PubChem auto-download)")
    ap.add_argument("--query", default=None,  help="Element symbol to look up, e.g. Fe")
    ap.add_argument("--k",     default=5, type=int, help="Nearest neighbours to return (default: 5)")
    ap.add_argument("--all",   action="store_true", help="Print TSA fingerprints for all elements")
    ap.add_argument("--props", nargs="+", metavar="KEY=VAL",
                    help="Property-based query, e.g.  density=7.87 electronegativity=1.83")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    db   = build_db(rows)

    # ── --all ─────────────────────────────────────────────────────────────────
    if args.all:
        print(f"\n  {'Sym':<5} {'Name':<18} {'Z':>3}  {'Π':>7}  {'Ε':>7}  {'Β':>7}  props")
        print(f"  {'─'*65}")
        for el in db:
            n = el["n_pi"] + el["n_e"] + el["n_b"]
            print(f"  {el['Symbol']:<5} {el['Name']:<18} {el['Z']:>3}  "
                  f"{fv(el['PI'])}  {fv(el['E'])}  {fv(el['B'])}  {n}")
        return

    # ── --query ───────────────────────────────────────────────────────────────
    if args.query:
        sym = args.query.strip()
        match = next(
            (el for el in db if el["Symbol"].lower() == sym.lower()), None
        )
        if match is None:
            print(f"Symbol '{sym}' not found. Run --all to list available symbols.")
            sys.exit(1)
        show_element(match)
        results = find_nearest(match, db, k=args.k, exclude=match["Symbol"])
        show_table(results, header=f"Nearest {args.k} to {match['Name']} ({match['Symbol']}) in TSA space")
        return

    # ── --props ───────────────────────────────────────────────────────────────
    if args.props:
        try:
            props = {}
            for item in args.props:
                k_str, v_str = item.split("=", 1)
                props[k_str.strip()] = float(v_str.strip())
        except (ValueError, IndexError) as exc:
            print(f"Error parsing --props: {exc}\nExpected: key=value pairs, e.g. density=7.87")
            sys.exit(1)
        fp = props_to_fp(props)
        print(f"\n  Query fingerprint   Π={fv(fp['PI'])}  Ε={fv(fp['E'])}  Β={fv(fp['B'])}")
        results = find_nearest(fp, db, k=args.k)
        show_table(results, header=f"Nearest {args.k} elements to property query")
        return

    # ── default: landmark elements ────────────────────────────────────────────
    print("\n  Phosphorus Material Classifier — nuclear landmark elements")
    for sym in LANDMARKS:
        el = next((e for e in db if e["Symbol"] == sym), None)
        if el:
            show_element(el)
    print()

if __name__ == "__main__":
    main()