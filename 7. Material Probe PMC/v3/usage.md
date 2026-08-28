Usage:
    python phosphorus_v3.py                              # nuclear landmark elements
    python phosphorus_v3.py --query Fe                   # nearest in nuclear scope
    python phosphorus_v3.py --query Fe --all             # nearest across all 118
    python phosphorus_v3.py --query U --k 10
    python phosphorus_v3.py --props density=7.87 electronegativity=1.83
    python phosphorus_v3.py --props group=8 period=4
    python phosphorus_v3.py --all                        # fingerprints, all 118
    python phosphorus_v3.py --csv my_elements.csv --query Gd
	
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