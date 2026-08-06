# mesh_probe.py — Argument Reference

## `--zones path/to/file.csv`

The main input. A CSV with one row per mesh zone containing the nine required columns:

| Column | Description |
|---|---|
| `zone_id` | Integer zone identifier |
| `kappa_mean` | Mean curvature in zone (1/mm or 1/m — consistent units throughout) |
| `kappa_max` | Maximum curvature in zone |
| `aspect_ratio` | Average element aspect ratio (1.0 = equilateral) |
| `skewness` | Maximum element skewness in zone (0 = perfect, 1 = degenerate) |
| `dist_to_bc` | Distance from zone centroid to nearest boundary condition attachment |
| `normal_deviation` | Angular deviation of surface normals across zone (degrees) |
| `area_fraction` | Zone area / total surface area (dimensionless, sums to 1 across all zones) |
| `edge_length_min` | Minimum element edge length in zone (same units as curvature) |

This is how you run the probe on real CAD export data. Mutually exclusive with `--test` — use one or the other, never both.

---

## `--flag ZONE_ID`

Marks a zone ID as **Β** regardless of what the geometry says. Repeat up to four times for multiple zones.

This is the user's assertion that *"I am not sure about this zone — force it to boundary status and show me why."* The refinement score and Π/Ε feature ratio are still computed and displayed for flagged zones (so you can see what the probe would have said), but the classification output is always **Β** and the recommendation is always **INSPECT**.

The accuracy count in `--test` validation mode excludes flagged zones.

**Example:**
```bash
python mesh_probe.py --zones my_mesh.csv --flag 3 --flag 7 --flag 12
```

---

--load ZONE_ID

Marks a zone ID as Π — forced refinement regardless of local geometry. Repeat up to four times for multiple zones.

This is the user's assertion that "physics requires refinement here independent of shape." The canonical use case is a load application point: a flat surface with low curvature scores Ε on geometry alone, but stress concentrations at the attachment are solver-driven and the probe cannot see them. --load overrides the geometry classification and writes a physics-specific justification to the rec and desc output fields.

This is distinct from --flag in a way that matters for the downstream mini program:

Argument	Output tsa_type	Output rec	Downstream action
--load	Π	REFINE — physics-driven. Load application zone...	Triggers refinement unconditionally
--flag	Β	INSPECT — user-flagged zone...	Pauses pipeline, requests human review

The score and Π/Ε ratio are still computed and displayed for load zones (so you can see the geometry assessment alongside the physics override). The confidence field reads physics-driven load zone rather than a score, distinguishing it from geometry-classified Π zones in the saved CSV.

Example:

bash
python mesh_probe.py --zones my_mesh.csv --load 3 --load 7
python mesh_probe.py --zones my_mesh.csv --load 3 --flag 9

---

## `--test {sphere,cylinder,box,cone}`

Runs the probe on built-in synthetic zone data for one of the four basic shapes instead of reading a CSV. No `--zones` needed.

Each shape has eight zones with physically motivated geometry parameters and known expected TSA types derived from the `h = R/6` engineering rule applied locally per zone. At the end of the run the probe prints a validation table comparing expected vs actual type per zone and an overall accuracy percentage.

Use this to:
- Verify the probe is working correctly after any code change
- Show a colleague what the output looks like before real mesh data is available
- Confirm that the encoding is structurally consistent with the domain physics

**Available shapes and their expected classification patterns:**

| Shape | Π zones | Β zones | Ε zones |
|---|---|---|---|
| `sphere` | none | none | all (uniform κ, below threshold everywhere) |
| `cylinder` | curved surface | edge transitions | flat ends, interior |
| `box` | sharp edges, corners | face-to-edge transitions | flat faces |
| `cone` | tip, upper cone | mid cone, lower cone, base edge | base flat, far zones |

---

## `--no_probe`

Skips the RF importance step entirely.

Without this flag the probe fits a 500-tree random forest on the encoded features vs the geometry-derived refinement score and prints:
- Grouped importance: % Π-encoded, % Ε-encoded, % cross-product features
- Top-10 features by importance with type label
- Dominant type and ratio
- Structural typing verdict (CONFIRMED or review flag)

This step confirms the variable classification is structurally correct. If Π-encoded features dominate (expected for curvature-driven mesh refinement), cascade encoding was the right choice for this domain. If Ε dominates, review the variable classification.

With `--no_probe` you get only the zone classification table and descriptions. Faster, and does not require the RF to run. Use it when:
- You only want the recommendations and have already confirmed the probe structurally
- You have fewer than 4 zones (RF requires at least 4 samples)
- You are running in a time-constrained pipeline step

---

## `--save path/to/output.csv`

Writes the classification results to a CSV at the given path. Each row is one zone.

| Column | Description |
|---|---|
| `zone_id` | Zone identifier |
| `tsa_type` | Classification: `Π`, `Ε`, or `Β` |
| `confidence` | Score and Π/Ε ratio, or `user-flagged` |
| `score` | Geometry-derived refinement urgency score (bifurcation = 1.0) |
| `pi_e_ratio` | Ratio of mean Π-encoded to mean Ε-encoded feature values |
| `rec` | Short recommendation string (REFINE / COARSE OK / REVIEW / INSPECT) |
| `desc` | Full description including curvature values, threshold comparison, and recommended element size |

This file feeds the next step in the pipeline — the mini program that calls the meshing API with refinement arguments derived from the probe output. The `desc` column is also what the skill-set AI layer reads to inform the user at each step.

---

## `--L_CHAR value`

Override for the characteristic length, in the same units as your curvature input (mm if curvature is 1/mm).

By default the probe estimates `L_CHAR` from the mesh data as `max(edge_length_min) × 10`, which is a rough approximation sufficient for basic shapes. If the CAD bounding box diagonal is available from the export — which it should be for any real job — pass it here.

`L_CHAR` controls:
- `DIST_NF` — the BC near-field influence radius (`= 0.10 × L_CHAR`)
- The display value of `KAPPA_BIFURCATION` shown in the constants summary
- The descriptions' reference to the h = R/6 bifurcation threshold

The curvature score itself uses local element size per zone (`κ × h_local × 6`) so it is not sensitive to `L_CHAR`. The BC proximity term and the printed descriptions are.

**Example:**
```bash
python mesh_probe.py --zones my_mesh.csv --flag 5 --L_CHAR 142.3
```

---

## Quick reference

```
python mesh_probe.py --zones FILE --flag ID [--flag ID] [--no_probe] [--save FILE] [--L_CHAR VALUE]
python mesh_probe.py --test {sphere,cylinder,box,cone} [--flag ID] [--no_probe] [--save FILE]
```

| Argument | Type | Default | Required |
|---|---|---|---|
| `--zones` | string (path) | None | Yes, unless `--test` |
| `--test` | choice | None | Yes, unless `--zones` |
| `--flag` | integer (repeatable, max 4) | empty | No |
| `--no_probe` | flag | off | No |
| `--save` | string (path) | None | No |
| `--L_CHAR` | float | estimated from data | No |

---

## Dependencies

```bash
pip install numpy pandas scikit-learn
```
