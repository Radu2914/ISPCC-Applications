PS D:\ISPCC\6. Mesh Probe ZSC\v3> python mesh_to_probe.py --nodes "D:\ISPCC\6. Mesh Probe ZSC\v3\_files\mesh_file.txt" --stl "D:\ISPCC\6. Mesh Probe ZSC\v2\_files\stl_mesh_file.stl" --load_nodes 3 2 --normal_tol 5.0

============================================================
  MESH TO PROBE PREPROCESSOR
============================================================

  Nodes loaded : 81
  Node bbox    : X[-10.00, 10.00] Y[-10.00, 10.00] Z[0.00, 20.00]

  STL triangles: 192
  Unit scale   : ×1000 (STL → node file units)

  Zones detected: 6  (normal_tol = 5.0°)
   Zone  Triangles  Normal direction
  ─────  ─────────  ──────────────────────────────
      1         32  (-1.000, +0.000, +0.000)
      2         32  (+0.000, +0.000, -1.000)
      3         32  (+0.000, +0.000, +1.000)
      4         32  (+0.000, +1.000, +0.000)
      5         32  (+0.000, -1.000, +0.000)
      6         32  (+1.000, +0.000, +0.000)

  Resolving physics-concentration node IDs to zones...
  Node    3 at (-10.00, +0.00, +10.00) → zone 1
  Node    2 at (+10.00, +0.00, +10.00) → zone 6
  Physics zones resolved (1-based): [1, 6]

  Computing zone statistics...

   Zone    κ_mean     κ_max     AR   Skew   NrmDev   AreaFrac   h_min  dist_load  load
  ─────  ────────  ────────  ─────  ─────  ───────  ─────────  ──────  ─────────  ────
      1    0.0000    0.0000   1.41  0.250     0.00     0.1667   5.000       0.00     1
      2    0.0000    0.0000   1.41  0.250     0.00     0.1667   5.000      14.14     0
      3    0.0000    0.0000   1.41  0.250     0.00     0.1667   5.000      14.14     0
      4    0.0000    0.0000   1.41  0.250     0.00     0.1667   5.000      14.14     0
      5    0.0000    0.0000   1.41  0.250     0.00     0.1667   5.000      14.14     0
      6    0.0000    0.0000   1.41  0.250     0.00     0.1667   5.000       0.00     1

  Extracting zone boundary edges...
  Zone boundary edges: 12
    Edge  1: Zone 1 ↔ Zone 2  dihedral=90.0°  length=20.00mm
    Edge  2: Zone 1 ↔ Zone 3  dihedral=90.0°  length=20.00mm
    Edge  3: Zone 1 ↔ Zone 4  dihedral=90.0°  length=20.00mm
    Edge  4: Zone 1 ↔ Zone 5  dihedral=90.0°  length=20.00mm
    Edge  5: Zone 2 ↔ Zone 4  dihedral=90.0°  length=20.00mm
    Edge  6: Zone 2 ↔ Zone 5  dihedral=90.0°  length=20.00mm
    Edge  7: Zone 2 ↔ Zone 6  dihedral=90.0°  length=20.00mm
    Edge  8: Zone 3 ↔ Zone 4  dihedral=90.0°  length=20.00mm
    Edge  9: Zone 3 ↔ Zone 5  dihedral=90.0°  length=20.00mm
    Edge 10: Zone 3 ↔ Zone 6  dihedral=90.0°  length=20.00mm
    Edge 11: Zone 4 ↔ Zone 6  dihedral=90.0°  length=20.00mm
    Edge 12: Zone 5 ↔ Zone 6  dihedral=90.0°  length=20.00mm

  Extracting corner points...
  Corner points (3+ zones): 8

  Saved:
    probe_input.csv         (face zones)
    probe_input_edges.csv  (zone boundary edges)
    probe_input_corners.csv  (corner points)

  Next step:
    python mesh_probe.py --zones probe_input.csv
    (physics zones [1, 6] resolved from node IDs [3, 2] — read automatically by probe)
    Add --flag ZONE_ID for geometric uncertainty, --load ZONE_ID to override zone assignment manually.

============================================================