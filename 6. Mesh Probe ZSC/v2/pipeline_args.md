# Pipeline Argument Reference

Two scripts. Run in order.

```
mesh_to_probe.py  →  probe_input.csv  →  mesh_probe.py
```

---

## mesh_to_probe.py

Reads the STL and node file exported from the mesher, clusters triangles into zones, computes per-zone geometry statistics, and writes `probe_input.csv` ready for the probe.

### `--nodes PATH` *(required)*

Tab-separated node coordinate file exported from the mesher. Required columns: `Node Number`, `X Location (mm)`, `Y Location (mm)`, `Z Location (mm)`. Column names are matched flexibly — any column containing `node` or `number` maps to the ID, any containing `x`, `y`, `z` maps to coordinates. Units must be consistent with the STL (scale factor is auto-detected).

### `--stl PATH` *(required)*

Binary STL file exported from the same mesher as the node file. Coordinate units are auto-detected relative to the node file (common case: STL in metres, node file in mm → scale ×1000 applied automatically). ASCII STL is not supported — export as binary.

### `--out PATH`

Output CSV path. Default: `probe_input.csv`. This file is the direct input to `mesh_probe.py --zones`.

### `--normal_tol DEGREES`

Angular tolerance in degrees for clustering STL triangles into zones. Default: `5.0`.

**User input required.** The correct value depends on the geometry. After running, check that the number of zones detected matches the number of physical faces you expect.

| Symptom | Cause | Fix |
|---|---|---|
| More zones than expected | Tolerance too tight — one face fragmented into multiple zones | Increase (try 10–15°) |
| Fewer zones than expected | Tolerance too loose — adjacent faces merged | Decrease (try 2–3°) |

Basic shapes with axis-aligned faces work correctly at the default 5°. Organic or off-axis geometry may need tuning.

### `--bc_nodes NODE_ID [NODE_ID ...]`

Node IDs (from the node file) where structural constraints are applied — fixed supports, symmetry planes, pinned joints.

**User input required.** This information does not exist in the geometry files. It is defined in the solver setup and must be entered manually.

The script resolves each node ID to its surface zone automatically by matching coordinates to the nearest STL triangle vertex. The resolved zones are used to compute `dist_to_bc` — the Euclidean distance from each zone centroid to the nearest BC zone centroid. Zones close to a constraint score higher on the BC proximity term in the refinement score.

Multiple nodes across multiple zones are handled correctly — all resolved zones contribute to the `dist_to_bc`calculation and the nearest one wins per zone. Nodes that don't match any STL vertex (interior nodes) produce a warning and are skipped.

If omitted: `dist_to_bc` is set to a large value (1e6) for all zones. The BC proximity term contributes nothing to the score. Safe default, but the score is incomplete.

```bash
--bc_nodes 12 13 14
```

### `--load_nodes NODE_ID [NODE_ID ...]`

Node IDs (from the node file) where forces are applied.

**User input required.** Not derivable from geometry files.

The script resolves each node ID to its surface zone automatically by matching the node's coordinates to the nearest STL triangle vertex. The matched zone is written as `is_load_zone = 1` in the output CSV. `mesh_probe.py` reads this column automatically and forces those zones to Π without needing a --load argument at the probe step.

If a node ID has no matching STL vertex within 0.01mm, a warning is printed and that node is skipped. This happens when the node is interior to the mesh (not on the surface) — force application nodes should always be surface nodes.

Multiple nodes that resolve to the same zone produce one `is_load_zone = 1` entry for that zone (no duplication).


If omitted: `is_load_zone = 0` for all zones.

```bash
--load_nodes 18
--load_nodes 18 22 23
```

## `Note`

`Specify surface nodes only for --bc_nodes and --load_nodes`

---

## mesh_probe.py

Reads the CSV produced by `mesh_to_probe.py` (or a hand-crafted CSV with the required columns), classifies each zone by TSA type, runs the RF importance probe, and resolves boundary zones interactively.

### `--zones PATH`

CSV file with zone descriptors. One row per zone. Required columns:

| Column | Description |
|---|---|
| `zone_id` | Integer zone identifier |
| `kappa_mean` | Mean curvature in zone |
| `kappa_max` | Maximum curvature in zone |
| `aspect_ratio` | Average element aspect ratio (1.0 = equilateral) |
| `skewness` | Maximum element skewness (0 = perfect, 1 = degenerate) |
| `dist_to_bc` | Distance from zone centroid to nearest BC attachment |
| `normal_deviation` | Angular deviation of surface normals across zone (degrees) |
| `area_fraction` | Zone area / total surface area |
| `edge_length_min` | Minimum element edge length in zone |

Optional columns added automatically by `mesh_to_probe.py`:

| Column | Effect when present |
|---|---|
| `is_load_zone` | Zones with value `1` are auto-forced to Π (no `--load` needed) |
| `neighbor_ids` | Enables neighbor-aware reclassification of flat zones |

Mutually exclusive with `--test`.

### `--load ZONE_ID`

Force a zone to **Π** (REFINE — physics-driven) regardless of geometry. Repeat up to 4 times.

Used as a manual override when running the probe directly on a hand-crafted CSV (without `is_load_zone` column), or to add a load zone after the fact without re-running the preprocessor. When `is_load_zone` is present in the CSV, both sources are merged — `--load` is additive.

```bash
--load 3 --load 7
```

### `--flag ZONE_ID`

Force a zone to **Β** (INSPECT — user attention) regardless of geometry. Repeat up to 4 times.

Distinct from `--load`: this signals geometric or contextual uncertainty, not a known physics condition. The zone is not refined automatically — it is held for human inspection. Used when the engineer is unsure whether a zone's geometry was simplified correctly, or suspects the mesh quality is locally unreliable.

```bash
--flag 9
```

### `--test {sphere,cylinder,box,cone}`

Run the probe on built-in synthetic zone data for a basic shape. No `--zones` needed. Prints a validation table comparing expected vs actual TSA type per zone at the end.

Use to verify the probe is working correctly after any code change, or to demonstrate the output before real mesh data is available.

### `--no_probe`

Skip the RF importance step. Produces only the zone classification table and descriptions. Faster. Use when structural confirmation has already been run, or when fewer than 4 zones are present (RF requires at least 4 samples).

### `--no_interactive`

Print boundary zone resolution options but skip the input prompt. Β zones stay as Β in the output. Use in scripted or pipeline contexts where interactive input is not available.

### `--seed INTEGER`

Random seed for the stochastic boundary zone option generator. Default: time-based (different options each run).

Set a fixed seed to reproduce the same set of intermediate options — useful for sharing a specific run with a colleague or for debugging. The two endpoint options (Coarse and Fine) are always exact physics-derived bounds regardless of seed; only the Light and Moderate intermediate options are affected.

```bash
--seed 42
```

### `--save PATH`

Save classification results to a CSV at the given path. One row per zone. Columns: `zone_id`, `tsa_type`, `confidence`, `score`, `pi_e_ratio`, `rec`, `desc`.

This file feeds the next step — the library query and Ansys instruction generator. The `confidence` field records user choices for boundary zones. The `desc` field contains the full physical reasoning per zone.

### `--L_CHAR VALUE`

Override the characteristic length in the same units as the curvature input. Default: estimated from `max(edge_length_min) × 10`.

Supply the CAD bounding box diagonal when available — it is more accurate than the estimate. Controls `DIST_NF` (BC influence radius = 10% of L_CHAR) and the display value of `KAPPA_BIFURCATION`. The curvature refinement score uses local element size per zone and is not sensitive to this value, but the BC proximity term and printed descriptions reference it.

```bash
--L_CHAR 142.3
```

---

## Arguments requiring user input — summary

These cannot be derived from the geometry files and must be entered manually.

| Script | Argument | Why it cannot be auto-detected |
|---|---|---|
| `mesh_to_probe.py` | `--bc_zones` | Constraint locations are defined in the solver, not the geometry export |
| `mesh_to_probe.py` | `--load_zones` | Force application is a physics decision, not visible in the STL |
| `mesh_to_probe.py` | `--normal_tol` | Correct tolerance depends on expected zone count, which only the engineer knows |
| `mesh_probe.py` | `--flag` | Geometric uncertainty is an engineering judgment, not computable |

All other arguments have safe defaults and are optional.

---

## Full pipeline example

```bash
# Step 1 — preprocess
python mesh_to_probe.py \
    --nodes mesh_file.txt \
    --stl   stl_mesh_file.stl \
    --bc_zones 2 \
    --load_zones 3 \
    --normal_tol 5 \
    --out probe_input.csv

# Step 2 — probe (load zones read automatically from CSV)
python mesh_probe.py \
    --zones probe_input.csv \
    --flag 9 \
    --seed 42 \
    --save results.csv
```

## Dependencies

```bash
pip install numpy pandas scikit-learn scipy
```
