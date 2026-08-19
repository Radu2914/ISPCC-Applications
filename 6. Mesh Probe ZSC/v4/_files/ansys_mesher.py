"""
ansys_mesher.py — PyMechanical Mesh Control Application
========================================================
Reads probe output CSVs and applies mesh controls to a running
Ansys Mechanical session via ansys-mechanical-core.

Methodology grounding (from APA reference library):
  Sizing strategy  : Zienkiewicz & Zhu (1987, 1992) — adaptive error estimator
  Error bounds     : Babuška & Rheinboldt (1978) — a posteriori error estimates
  Element quality  : Knupp (2001) — algebraic mesh quality metrics
  Mesh generation  : Blacker & Stephenson (1991) — paving / quadrilateral meshing
  ANSYS commands   : ANSYS Inc. (2025) — Mechanical APDL meshing guide 2025 R1

Requires:
    Ansys Mechanical 2023 R1 or later, running with project and geometry open.
    pip install ansys-mechanical-core

Probe pipeline prerequisites (run in order):
    mesh_to_probe.py  →  probe_input.csv
                         probe_input_edges.csv
                         probe_input_corners.csv
    interior_probe.py →  probe_input_interior.csv

Usage:
    python ansys_mesher.py --zones probe_input.csv
    python ansys_mesher.py --zones probe_input.csv --port 10000
    python ansys_mesher.py --zones probe_input.csv --dry_run
    python ansys_mesher.py --zones probe_input.csv --no_generate

Controls applied in insertion order (later = higher priority in Ansys):
    Step 1  Body Sizing          — global baseline, Soft
    Step 2  Inflation            — 2-2-1 B transition from each Π face
    Step 3  Edge Sizing          — Π/Π and Π/Ε zone boundaries
    Step 4  Face Sizing          — Π zones (load/constraint faces), Hard
    Step 5  Sphere of Influence  — corners (Π/Ε and Π/Π)
    Step 6  Sphere of Influence  — interior B_sphere correction (load midpoint)
"""

import numpy as np
import pandas as pd
import argparse
import os
import sys
import tempfile

EPS = 1e-9


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_probe_data(zones_path):
    """
    Load all probe output CSVs from the zones file path stem.
    Derives edge, corner, and interior paths automatically.
    Returns dict of DataFrames.
    """
    base        = zones_path.replace('.csv', '')
    edge_path   = base + '_edges.csv'
    corner_path = base + '_corners.csv'
    interior_path = base + '_interior.csv'

    data = {}
    data['face'] = pd.read_csv(zones_path)

    data['edge'] = pd.read_csv(edge_path) \
                   if os.path.exists(edge_path) else pd.DataFrame()
    data['corner'] = pd.read_csv(corner_path) \
                     if os.path.exists(corner_path) else pd.DataFrame()
    data['interior'] = pd.read_csv(interior_path) \
                       if os.path.exists(interior_path) else pd.DataFrame()

    return data


def derive_zone_centroids(face_df, edge_df):
    """
    Compute zone face centroids from edge midpoints.
    Zone centroid = mean of all edge midpoints attached to that zone.
    """
    centroids = {}
    for z in face_df['zone_id'].astype(int):
        if len(edge_df) > 0:
            mask = (edge_df['zone_a'] == z) | (edge_df['zone_b'] == z)
            if mask.any():
                pts = edge_df.loc[mask, ['mid_x','mid_y','mid_z']].values
                centroids[z] = pts.mean(axis=0).tolist()
    return centroids


def derive_edge_regime(face_df, edge_df):
    """
    Compute regime and h_edge per edge from face types.
    Returns edge_df with regime and h_edge columns added.
    """
    if len(edge_df) == 0 or 'regime' in edge_df.columns:
        return edge_df

    type_map = {int(r['zone_id']): ('Pi' if r['is_load_zone']==1 else 'E')
                for _, r in face_df.iterrows()}
    h_map    = {int(r['zone_id']): float(r['edge_length_min'])
                for _, r in face_df.iterrows()}

    regimes, h_edges = [], []
    for _, r in edge_df.iterrows():
        ta = type_map.get(int(r['zone_a']), 'E')
        tb = type_map.get(int(r['zone_b']), 'E')
        ha = h_map.get(int(r['zone_a']), 5.0)
        hb = h_map.get(int(r['zone_b']), 5.0)
        pair = tuple(sorted([ta, tb]))
        if pair == ('Pi', 'Pi'):
            regime = 'Pi_Pi'; h_edge = float(np.sqrt(ha * hb))
        elif 'Pi' in pair:
            regime = 'Pi_E';  h_edge = float(np.sqrt(ha * hb))
        else:
            regime = 'E_E';   h_edge = max(ha, hb)
        regimes.append(regime)
        h_edges.append(round(h_edge, 4))

    edge_df = edge_df.copy()
    edge_df['regime'] = regimes
    edge_df['h_edge'] = h_edges
    return edge_df


def derive_corner_regime(face_df, corner_df):
    """Compute regime per corner from adjacent zone types."""
    if len(corner_df) == 0 or 'regime' in corner_df.columns:
        return corner_df

    type_map = {int(r['zone_id']): ('Pi' if r['is_load_zone']==1 else 'E')
                for _, r in face_df.iterrows()}
    h_map    = {int(r['zone_id']): float(r['edge_length_min'])
                for _, r in face_df.iterrows()}

    regimes, h_corners, soi_radii = [], [], []
    for _, r in corner_df.iterrows():
        adj   = [int(z) for z in str(r['adjacent_zones']).split(',')]
        types = [type_map.get(z, 'E') for z in adj]
        h_vals= [h_map.get(z, 5.0) for z in adj]
        n_pi  = types.count('Pi')

        if n_pi >= 2:
            regime   = 'Pi_Pi'
            h_corner = min(h for h,t in zip(h_vals,types) if t=='Pi') * 0.5
        elif n_pi == 1:
            regime   = 'Pi_E'
            h_pi     = min(h for h,t in zip(h_vals,types) if t=='Pi')
            h_e      = max(h for h,t in zip(h_vals,types) if t=='E')
            h_corner = float(np.sqrt(h_pi * h_e)) * 0.5
        else:
            regime   = 'E_E'
            h_corner = max(h_vals)

        regimes.append(regime)
        h_corners.append(round(h_corner, 4))
        soi_radii.append(round(h_corner * 2.0, 4))

    corner_df = corner_df.copy()
    corner_df['regime']     = regimes
    corner_df['h_corner']   = h_corners
    corner_df['soi_radius'] = soi_radii
    return corner_df


# ═══════════════════════════════════════════════════════════════════════════════
# MECHANICAL SCRIPT BUILDER
# Generates the Python script that runs INSIDE Mechanical's Python environment.
# Data is embedded as Python literals — the generated script is self-contained
# and can also be pasted directly into Mechanical's Scripting console.
# ═══════════════════════════════════════════════════════════════════════════════

def build_mechanical_script(data, sv_depth, unit_scale=1.0):
    """
    Build the complete Python script to execute inside Ansys Mechanical.

    unit_scale: multiply probe coordinates by this to get Mechanical units.
                1.0 if Mechanical project uses mm (typical).
                0.001 if Mechanical uses metres (SI strict mode).

    Returns script string. Embed into app.run_python_script() or save to file.

    Geometry selection strategy:
      Faces  — nearest face centroid to probe zone centroid (by distance)
      Edges  — nearest model edge midpoint to probe edge midpoint
      Vertices — nearest vertex to probe corner/sphere coordinate

    All selections create Named Selections before applying controls.
    This keeps the Mechanical outline tree readable and auditable.
    """
    face_df    = data['face']
    edge_df    = data['edge']
    corner_df  = data['corner']
    interior_df= data['interior']

    # Probe values
    h_global = float(face_df['edge_length_min'].max())
    pi_faces = face_df[face_df['is_load_zone'] == 1]
    h_pi     = float(pi_faces['edge_length_min'].mean()) \
               if len(pi_faces) > 0 else h_global
    h_e      = h_global
    h_b      = float(np.sqrt(h_pi * h_e))

    # Zone centroids
    centroids = {}
    for z in face_df['zone_id'].astype(int):
        if len(edge_df) > 0:
            mask = (edge_df['zone_a'] == z) | (edge_df['zone_b'] == z)
            if mask.any():
                pts = edge_df.loc[mask, ['mid_x','mid_y','mid_z']].values
                centroids[z] = [round(v * unit_scale, 6) for v in pts.mean(axis=0)]

    # Active edges (non Ε/Ε)
    active_edges = edge_df[edge_df['regime'] != 'E_E'] \
                   if len(edge_df) > 0 and 'regime' in edge_df.columns \
                   else pd.DataFrame()

    # Active corners (non Ε/Ε)
    active_corners = corner_df[corner_df['regime'] != 'E_E'] \
                     if len(corner_df) > 0 and 'regime' in corner_df.columns \
                     else pd.DataFrame()

    # Interior B_sphere nodes
    b_sphere_nodes = interior_df[interior_df['zone_type'] == 'B_sphere'] \
                     if len(interior_df) > 0 and 'zone_type' in interior_df.columns \
                     else pd.DataFrame()
    h_b_sphere     = float(b_sphere_nodes['h_target'].mean()) \
                     if len(b_sphere_nodes) > 0 else h_b * 0.5
    r_b_sphere     = round(h_b_sphere * 2.0 * unit_scale, 6)

    # ── Generate script ───────────────────────────────────────────────────────
    s = []
    s.append('# ═══════════════════════════════════════════════════════════════')
    s.append('# Auto-generated by ansys_mesher.py — ISPCC Mesh Probe Pipeline')
    s.append('# Methodology: Zienkiewicz-Zhu (1987, 1992), Babuška-Rheinboldt (1978)')
    s.append('# Insert controls in order — later entries take higher priority')
    s.append('# ═══════════════════════════════════════════════════════════════')
    s.append('')
    s.append('import System')
    s.append('')
    s.append('# ── Model references ────────────────────────────────────────────')
    s.append('model  = ExtAPI.DataModel.Project.Model')
    s.append('mesh   = model.Mesh')
    s.append('geo    = model.Geometry')
    s.append('')
    s.append('# Get first body (extend for multi-body geometry)')
    s.append('try:')
    s.append('    body = geo.Children[0].Children[0]')
    s.append('except:')
    s.append('    body = geo.Children[0]')
    s.append('')

    # ── Geometry helper functions ──────────────────────────────────────────────
    s.append('# ── Geometry helpers ───────────────────────────────────────────')
    s.append('def dist3(ax, ay, az, bx, by, bz):')
    s.append('    return ((ax-bx)**2 + (ay-by)**2 + (az-bz)**2)**0.5')
    s.append('')
    s.append('def find_face(cx, cy, cz, tol=10.0):')
    s.append('    """Find face whose centroid is nearest (cx,cy,cz). tol in model units."""')
    s.append('    best, best_d = None, float("inf")')
    s.append('    for f in body.Faces:')
    s.append('        c = f.Centroid')
    s.append('        d = dist3(c.X, c.Y, c.Z, cx, cy, cz)')
    s.append('        if d < best_d:')
    s.append('            best_d, best = d, f')
    s.append('    return best if best_d < tol else None')
    s.append('')
    s.append('def find_edge(mx, my, mz, tol=10.0):')
    s.append('    """Find edge whose midpoint is nearest (mx,my,mz)."""')
    s.append('    best, best_d = None, float("inf")')
    s.append('    for e in body.Edges:')
    s.append('        mid = e.StartVertex.Location')
    s.append('        end = e.EndVertex.Location')
    s.append('        c = ((mid.X+end.X)/2, (mid.Y+end.Y)/2, (mid.Z+end.Z)/2)')
    s.append('        d = dist3(c[0], c[1], c[2], mx, my, mz)')
    s.append('        if d < best_d:')
    s.append('            best_d, best = d, e')
    s.append('    return best if best_d < tol else None')
    s.append('')
    s.append('def find_vertex(vx, vy, vz, tol=2.0):')
    s.append('    """Find vertex nearest (vx,vy,vz)."""')
    s.append('    best, best_d = None, float("inf")')
    s.append('    for v in body.Vertices:')
    s.append('        loc = v.Location')
    s.append('        d   = dist3(loc.X, loc.Y, loc.Z, vx, vy, vz)')
    s.append('        if d < best_d:')
    s.append('            best_d, best = d, v')
    s.append('    return best if best_d < tol else None')
    s.append('')
    s.append('def make_ns(name, ids, entity_type):')
    s.append('    """Create Named Selection from a list of entity IDs."""')
    s.append('    ns   = model.AddNamedSelection()')
    s.append('    ns.Name = name')
    s.append('    sel  = ExtAPI.SelectionManager.CreateSelectionInfo(')
    s.append('               SelectionTypeEnum.GeometryEntities)')
    s.append('    sel.Ids = ids')
    s.append('    ns.Location = sel')
    s.append('    return ns')
    s.append('')
    s.append('def make_sizing(ns, h_mm, behavior="Soft", sizing_type="ElementSize"):')
    s.append('    """Add a sizing control on a named selection."""')
    s.append('    sz = mesh.AddSizing()')
    s.append('    sz.Location = ns')
    s.append('    if sizing_type == "ElementSize":')
    s.append('        sz.Type = SizingType.ElementSize')
    s.append('        sz.ElementSize = Quantity(str(round(h_mm, 4)) + " [mm]")')
    s.append('    if behavior == "Hard":')
    s.append('        sz.Behavior = SizingBehavior.Hard')
    s.append('    else:')
    s.append('        sz.Behavior = SizingBehavior.Soft')
    s.append('    return sz')
    s.append('')

    # ── Step 1: Global body sizing ─────────────────────────────────────────────
    s.append('# ════════════════════════════════════════════════════════════════')
    s.append('# Step 1 — Global Body Sizing (baseline, Soft)')
    s.append('# Babuška & Rheinboldt (1978): lower-bound element size for acceptable error')
    s.append('# ════════════════════════════════════════════════════════════════')
    s.append('body_ns = make_ns("ISPCC_Body_Global", [body.Id], "Body")')
    s.append('sz_global = mesh.AddSizing()')
    s.append('sz_global.Location = body_ns')
    s.append('sz_global.Type = SizingType.ElementSize')
    s.append(f'sz_global.ElementSize = Quantity("{h_global:.4f} [mm]")')
    s.append('sz_global.Behavior = SizingBehavior.Soft')
    s.append('print("Step 1 done — global body sizing {:.4f} mm".format({:.4f}))'.format(
        h_global, h_global))
    s.append('')

    # ── Step 2: Inflation from each Π face (2-2-1) ─────────────────────────────
    if len(pi_faces) > 0:
        s.append('# ════════════════════════════════════════════════════════════════')
        s.append('# Step 2 — Inflation: 2-2-1 B transition from each Pi face')
        s.append('# Zienkiewicz & Zhu (1992): superconvergent patch — layer grading')
        s.append('# 5 total layers: 2 Pi-weighted + 2 mixed + 1 B proper')
        s.append('# ════════════════════════════════════════════════════════════════')

        for _, row in pi_faces.iterrows():
            zid = int(row['zone_id'])
            cx, cy, cz = centroids.get(zid, [0.0, 0.0, 0.0])
            s.append(f'# Zone {zid} — Π face inflation')
            s.append(f'face_z{zid} = find_face({cx}, {cy}, {cz})')
            s.append(f'if face_z{zid} is not None:')
            s.append(f'    ns_z{zid} = make_ns("ISPCC_Pi_Face_{zid}", [face_z{zid}.Id], "Face")')
            s.append(f'    infl_z{zid} = mesh.AddInflation()')
            s.append(f'    infl_z{zid}.Location = ns_z{zid}')
            s.append(f'    infl_z{zid}.InflationOption = InflationOption.TotalThickness')
            s.append(f'    infl_z{zid}.TotalThickness = Quantity("{sv_depth * unit_scale:.4f} [mm]")')
            s.append(f'    infl_z{zid}.NumberOfLayers = 5')
            s.append(f'    infl_z{zid}.GrowthRate = 1.2')
            s.append(f'    print("  Zone {zid} inflation added")')
            s.append(f'else:')
            s.append(f'    print("  [WARN] Zone {zid} face not found at centroid ({cx:.2f},{cy:.2f},{cz:.2f})")')
            s.append('')

        s.append(f'print("Step 2 done — inflation from {len(pi_faces)} Pi face(s)")')
        s.append('')

    # ── Step 3: Edge sizing (Π/Ε and Π/Π edges) ───────────────────────────────
    if len(active_edges) > 0:
        s.append('# ════════════════════════════════════════════════════════════════')
        s.append('# Step 3 — Edge Sizing: zone boundary Β entities')
        s.append('# TSA theorem: h_edge = sqrt(h_Pi × h_E) — geometric mean')
        s.append('# Blacker & Stephenson (1991): boundary layer seeding')
        s.append('# ════════════════════════════════════════════════════════════════')

        for _, row in active_edges.iterrows():
            eid  = int(row['edge_id'])
            mx   = float(row['mid_x']) * unit_scale
            my   = float(row['mid_y']) * unit_scale
            mz   = float(row['mid_z']) * unit_scale
            h    = float(row['h_edge'])
            reg  = row['regime']
            beh  = 'Hard' if reg == 'Pi_Pi' else 'Soft'

            s.append(f'# Edge {eid} — {reg}')
            s.append(f'edge_{eid} = find_edge({mx:.4f}, {my:.4f}, {mz:.4f})')
            s.append(f'if edge_{eid} is not None:')
            s.append(f'    ns_e{eid} = make_ns("ISPCC_Edge_{eid}_{reg}", [edge_{eid}.Id], "Edge")')
            s.append(f'    make_sizing(ns_e{eid}, {h:.4f}, behavior="{beh}")')
            s.append(f'else:')
            s.append(f'    print("  [WARN] Edge {eid} not found at midpoint ({mx:.2f},{my:.2f},{mz:.2f})")')
            s.append('')

        s.append(f'print("Step 3 done — {len(active_edges)} edge sizing control(s)")')
        s.append('')

    # ── Step 4: Face sizing (Π faces, Hard) ────────────────────────────────────
    if len(pi_faces) > 0:
        s.append('# ════════════════════════════════════════════════════════════════')
        s.append('# Step 4 — Face Sizing: Pi zones, Hard (overrides global)')
        s.append('# Zienkiewicz & Zhu (1987): element size at high-gradient regions')
        s.append('# ════════════════════════════════════════════════════════════════')

        for _, row in pi_faces.iterrows():
            zid  = int(row['zone_id'])
            h    = float(row['edge_length_min'])
            cx, cy, cz = centroids.get(zid, [0.0, 0.0, 0.0])

            s.append(f'# Zone {zid} — Pi face sizing')
            s.append(f'if face_z{zid} is not None:')
            s.append(f'    ns_fs{zid} = make_ns("ISPCC_Pi_FaceSizing_{zid}", [face_z{zid}.Id], "Face")')
            s.append(f'    make_sizing(ns_fs{zid}, {h:.4f}, behavior="Hard")')
            s.append(f'    print("  Zone {zid} face sizing {h:.4f} mm Hard")')
            s.append('')

        s.append(f'print("Step 4 done — face sizing on {len(pi_faces)} Pi face(s)")')
        s.append('')

    # ── Step 5: Sphere of influence — corners ──────────────────────────────────
    if len(active_corners) > 0:
        s.append('# ════════════════════════════════════════════════════════════════')
        s.append('# Step 5 — Sphere of Influence: corner Β entities')
        s.append('# Knupp (2001): Jacobian quality at corner singularities')
        s.append('# h_corner = geometric_mean × 0.5 — halved for corner gradation')
        s.append('# ════════════════════════════════════════════════════════════════')

        for _, row in active_corners.iterrows():
            cid  = int(row['corner_id'])
            vx   = float(row['x']) * unit_scale
            vy   = float(row['y']) * unit_scale
            vz   = float(row['z']) * unit_scale
            h    = float(row['h_corner'])
            r    = float(row['soi_radius']) * unit_scale
            reg  = row['regime']

            s.append(f'# Corner {cid} — {reg} at ({vx:.2f},{vy:.2f},{vz:.2f})')
            s.append(f'vert_{cid} = find_vertex({vx:.4f}, {vy:.4f}, {vz:.4f})')
            s.append(f'if vert_{cid} is not None:')
            s.append(f'    ns_c{cid} = make_ns("ISPCC_Corner_{cid}_{reg}", [vert_{cid}.Id], "Vertex")')
            s.append(f'    sz_c{cid} = mesh.AddSizing()')
            s.append(f'    sz_c{cid}.Location = ns_c{cid}')
            s.append(f'    sz_c{cid}.Type = SizingType.SphereOfInfluence')
            s.append(f'    sz_c{cid}.ElementSize = Quantity("{h:.4f} [mm]")')
            s.append(f'    sz_c{cid}.SphereRadius = Quantity("{r:.4f} [mm]")')
            s.append(f'    print("  Corner {cid} sphere h={h:.4f} r={r:.4f}")')
            s.append(f'else:')
            s.append(f'    print("  [WARN] Corner {cid} vertex not found at ({vx:.2f},{vy:.2f},{vz:.2f})")')
            s.append('')

        s.append(f'print("Step 5 done — {len(active_corners)} corner sphere(s)")')
        s.append('')

    # ── Step 6: Sphere of influence — interior B_sphere ────────────────────────
    if len(b_sphere_nodes) > 0:
        ctr = b_sphere_nodes[['x','y','z']].values.mean(axis=0) * unit_scale
        s.append('# ════════════════════════════════════════════════════════════════')
        s.append('# Step 6 — Sphere of Influence: interior B_sphere correction')
        s.append('# Load midpoint — equidistant from both Pi faces.')
        s.append('# Sphere mesh distributes load radially. Planar gradient incorrect.')
        s.append('# Applied to nearest vertex; covers midpoint via SOI radius.')
        s.append('# ════════════════════════════════════════════════════════════════')
        s.append(f'vert_bsph = find_vertex({ctr[0]:.4f}, {ctr[1]:.4f}, {ctr[2]:.4f}, tol=15.0)')
        s.append('if vert_bsph is not None:')
        s.append('    ns_bsph = make_ns("ISPCC_Interior_BSphere", [vert_bsph.Id], "Vertex")')
        s.append('    sz_bsph = mesh.AddSizing()')
        s.append('    sz_bsph.Location = ns_bsph')
        s.append('    sz_bsph.Type = SizingType.SphereOfInfluence')
        s.append(f'    sz_bsph.ElementSize = Quantity("{h_b_sphere:.4f} [mm]")')
        s.append(f'    sz_bsph.SphereRadius = Quantity("{r_b_sphere:.4f} [mm]")')
        s.append(f'    print("Interior B_sphere correction: h={h_b_sphere:.4f} r={r_b_sphere:.4f}")')
        s.append('else:')
        s.append('    print("[WARN] Interior B_sphere vertex not found — check coordinate")')
        s.append('')
        s.append('print("Step 6 done — interior B_sphere correction")')
        s.append('')

    # ── Mesh generation ────────────────────────────────────────────────────────
    s.append('# ════════════════════════════════════════════════════════════════')
    s.append('# Generate mesh')
    s.append('# ════════════════════════════════════════════════════════════════')
    s.append('# mesh.GenerateMesh()  # uncomment to generate automatically')
    s.append('# Leave commented for manual review of controls before meshing.')
    s.append('print("All mesh controls applied. Review in outline tree.")')
    s.append('print("Run mesh.GenerateMesh() or use Generate Mesh in the GUI.")')
    s.append('')
    s.append('# ═══════════════════════════════════════════════════════════════')
    s.append('# End of ISPCC auto-generated mesh script')
    s.append('# ═══════════════════════════════════════════════════════════════')

    return '\n'.join(s)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='ansys_mesher — apply ISPCC probe mesh controls to Ansys Mechanical')
    ap.add_argument('--zones',      required=True,
                    help='Face zone CSV (probe_input.csv)')
    ap.add_argument('--port',       type=int, default=None,
                    help='Mechanical gRPC port (default: auto-detect running instance)')
    ap.add_argument('--sv_depth',   type=float, default=None,
                    help='St. Venant depth in mm (default: read from interior CSV '
                         'or fallback 10mm). Override when running without interior_probe.')
    ap.add_argument('--unit_scale', type=float, default=1.0,
                    help='Scale factor: probe coords × scale = Mechanical units. '
                         '1.0 for mm projects (default). '
                         '0.001 for SI-metre Mechanical projects.')
    ap.add_argument('--dry_run',    action='store_true',
                    help='Generate and print the Mechanical script without connecting')
    ap.add_argument('--save_script', type=str, default=None,
                    help='Save generated Mechanical script to this .py path')
    ap.add_argument('--no_generate', action='store_true',
                    help='Apply controls but do not call GenerateMesh()')
    args = ap.parse_args()

    SEP = '=' * 68

    print(f'\n{SEP}')
    print(f'  ANSYS MESHER — ISPCC Probe Pipeline')
    print(f'{SEP}')

    # ── Load probe data ─────────────────────────────────────────────
    print(f'\n  Loading probe data from: {args.zones}')
    data = load_probe_data(args.zones)
    data['edge']   = derive_edge_regime(data['face'], data['edge'])
    data['corner'] = derive_corner_regime(data['face'], data['corner'])

    print(f'  Face zones    : {len(data["face"])}')
    print(f'  Edge entries  : {len(data["edge"])}')
    print(f'  Corner points : {len(data["corner"])}')
    print(f'  Interior nodes: {len(data["interior"])}')

    # ── St. Venant depth ─────────────────────────────────────────────
    if args.sv_depth is not None:
        sv_depth = args.sv_depth
    elif len(data['interior']) > 0 and 'dist_surf' in data['interior'].columns:
        b_nodes  = data['interior'][data['interior']['zone_type'].isin(['B','B_sphere'])]
        sv_depth = float(b_nodes['dist_surf'].max()) if len(b_nodes) > 0 else 10.0
    else:
        # Fallback: 2 × min active h_edge
        if len(data['edge']) > 0 and 'h_edge' in data['edge'].columns:
            active   = data['edge'][data['edge']['regime'] != 'E_E']['h_edge']
            sv_depth = float(active.min()) * 2.0 if len(active) > 0 else 10.0
        else:
            sv_depth = 10.0

    print(f'  St. Venant depth: {sv_depth:.4f} mm')
    print(f'  Unit scale      : ×{args.unit_scale}')

    # ── Build Mechanical script ──────────────────────────────────────
    print(f'\n  Building Mechanical Python script...')
    script = build_mechanical_script(data, sv_depth, unit_scale=args.unit_scale)

    # ── Save script ──────────────────────────────────────────────────
    script_path = args.save_script or args.zones.replace('.csv', '_mech_script.py')
    with open(script_path, 'w') as f:
        f.write(script)
    print(f'  Script saved to: {script_path}')
    print(f'  (Can be pasted directly into Mechanical Scripting console)')

    if args.dry_run:
        print(f'\n  ── DRY RUN — script content ──────────────────────────────')
        print(script)
        print(f'\n{SEP}')
        print(f'  Dry run complete. No Mechanical connection made.')
        print(f'{SEP}\n')
        return

    # ── Connect to Mechanical ────────────────────────────────────────
    try:
        import ansys.mechanical.core as mech
    except ImportError:
        print('\n  [ERROR] ansys-mechanical-core not installed.')
        print('  Install: pip install ansys-mechanical-core')
        print(f'  Script saved to {script_path} — paste into Mechanical scripting console.')
        sys.exit(1)

    print(f'\n  Connecting to Ansys Mechanical...')
    try:
        if args.port:
            app = mech.App(port=args.port)
        else:
            app = mech.App()
        print(f'  Connected. Mechanical version: {app.version}')
    except Exception as e:
        print(f'\n  [ERROR] Could not connect to Mechanical: {e}')
        print(f'  Ensure Mechanical is running with remote scripting enabled.')
        print(f'  Start Mechanical with: ansys-mechanical -grpc')
        print(f'  Script saved to {script_path} — paste manually if needed.')
        sys.exit(1)

    # ── Execute script ────────────────────────────────────────────────
    print(f'\n  Executing mesh controls in Mechanical...')
    print(f'  (Watch outline tree — Named Selections and controls appearing)')

    try:
        result = app.run_python_script_from_file(script_path)
        if result:
            print(f'\n  Mechanical output:')
            for line in str(result).split('\n'):
                if line.strip():
                    print(f'    {line}')
    except AttributeError:
        # Fallback: run_python_script (string method)
        try:
            result = app.run_python_script(script)
            if result:
                print(f'\n  Mechanical output:')
                for line in str(result).split('\n'):
                    if line.strip():
                        print(f'    {line}')
        except Exception as e:
            print(f'\n  [ERROR] Script execution failed: {e}')
            print(f'  Try pasting {script_path} manually into Mechanical scripting console.')
            sys.exit(1)

    print(f'\n{SEP}')
    print(f'  Done. Controls applied in Mechanical outline tree.')
    print(f'  Review Named Selections and mesh controls, then generate mesh.')
    print(f'  To generate: run mesh.GenerateMesh() in Mechanical scripting console')
    print(f'  or click Generate Mesh in the Mesh branch of the outline.')
    print(f'{SEP}\n')


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Full pipeline (run in order):
#   python mesh_to_probe.py  --nodes mesh_file.txt --stl stl_mesh_file.stl
#                            --load_nodes 2 --out probe_input.csv
#   python mesh_probe.py     --zones probe_input.csv --no_probe
#   python interior_probe.py --zones probe_input.csv
#                            --nodes mesh_file.txt --stl stl_mesh_file.stl
#   python ansys_mesher.py   --zones probe_input.csv
#
# Connect to specific port:
#   python ansys_mesher.py --zones probe_input.csv --port 10000
#
# Review script without connecting:
#   python ansys_mesher.py --zones probe_input.csv --dry_run
#
# Save generated script only (no connection):
#   python ansys_mesher.py --zones probe_input.csv --dry_run
#                          --save_script my_mesh_controls.py
#
# SI-metre Mechanical project:
#   python ansys_mesher.py --zones probe_input.csv --unit_scale 0.001
#
# Arguments:
#   --zones       Face zone CSV (required)
#   --port        Mechanical gRPC port (optional, auto-detect if omitted)
#   --sv_depth    St. Venant depth mm override (optional)
#   --unit_scale  Coordinate scale factor (default 1.0 = mm project)
#   --dry_run     Print script, no connection
#   --save_script Path to save generated script (default: zones_stem_mech_script.py)
#   --no_generate Apply controls but skip GenerateMesh call
#
# Dependencies:
#   pip install ansys-mechanical-core numpy pandas
#
# Mechanical prerequisites:
#   Ansys Mechanical 2023 R1 or later
#   Project open with geometry loaded
#   Remote scripting enabled (ansys-mechanical --grpc or enabled in Workbench)
