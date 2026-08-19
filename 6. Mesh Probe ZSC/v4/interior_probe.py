"""
interior_probe.py — Interior Node Classification and Topological Entropy
========================================================================
Standalone. Run after mesh_probe.py has produced face/edge/corner CSVs.

Reads:
    probe_input.csv          face zones + is_load_zone + edge_length_min
    probe_input_edges.csv    edge regime + h_edge + midpoints
    probe_input_corners.csv  corner points + regime
    mesh_file.txt            all nodes including interior
    stl_mesh_file.stl        surface node identification

Produces:
    probe_input_interior.csv    per-node classification
    console                     entropy scalar + Ansys interior instructions

Load direction:
    Inferred from Π zone edge midpoint centroids.
    Load zones are orthogonal to their face normals by assumption.
    Principal axis = vector between the two Π zone centroids.

Interior seeding:
    Pi from center outward : net-3 weight (5 forward, -2 back)
    E as stretched cylinders: 4 nodes per line minimum, period-2
    B transition (2-2-1)   : 2 Pi layers + 2 mixed layers + 1 B layer
                             applied from each Π face inward along load axis

Sphere correction for B nodes:
    B nodes equidistant (within sv_depth × 0.3) from both Π faces
    are resolved with a Sphere of Influence rather than a planar gradient.
    Physical meaning: at the load midpoint, load distributes radially.
    A sphere mesh distributes the load correctly; a planar gradient does not.

Topological entropy:
    H = -Σ p_i × log2(p_i) over weighted entity distribution
    B layers weighted by 2-2-1 sequence: [0.4, 0.4, 0.2]
    Computed across surface + interior entities combined.
    Low H (<1.5 bits): strong Π/Ε separation — large intentional mesh advantage.
    High H (>2.5 bits): complex load path — advantage present, needs validation.

Usage:
    python interior_probe.py --zones probe_input.csv
                             --nodes mesh_file.txt
                             --stl   stl_mesh_file.stl

    python interior_probe.py --zones probe_input.csv
                             --nodes mesh_file.txt
                             --stl   stl_mesh_file.stl
                             --sv_factor 2.5
                             --out   my_interior.csv

Dependencies:
    pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
import struct
import argparse
import os
import sys
from scipy.spatial import cKDTree

EPS = 1e-9
BETA_SEQ = (2, 2, 1)   # 2-2-1 transition layers — fixed by TSA theorem


# ═══════════════════════════════════════════════════════════════════════════════
# STL READER
# ═══════════════════════════════════════════════════════════════════════════════

def read_stl_binary(path):
    with open(path, 'rb') as f:
        f.read(80)
        n_tri   = struct.unpack('<I', f.read(4))[0]
        normals = np.zeros((n_tri, 3))
        verts   = np.zeros((n_tri, 3, 3))
        for i in range(n_tri):
            data        = f.read(50)
            normals[i]  = struct.unpack('<fff', data[0:12])
            verts[i, 0] = struct.unpack('<fff', data[12:24])
            verts[i, 1] = struct.unpack('<fff', data[24:36])
            verts[i, 2] = struct.unpack('<fff', data[36:48])
    return normals, verts


def detect_unit_scale(stl_verts, node_coords):
    stl_diag  = np.linalg.norm(
        stl_verts.reshape(-1,3).max(axis=0) -
        stl_verts.reshape(-1,3).min(axis=0))
    node_diag = np.linalg.norm(
        node_coords.max(axis=0) - node_coords.min(axis=0))
    if stl_diag < EPS:
        return 1.0
    ratio = node_diag / stl_diag
    if   ratio > 500:  return 1000.0
    elif ratio > 50:   return 100.0
    elif ratio > 5:    return 10.0
    elif ratio > 0.5:  return 1.0
    elif ratio > 0.05: return 0.1
    else:              return 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DIRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def infer_load_direction(face_df, edge_df):
    """
    Infer principal load axis from Π zone centroids.
    Π zone centroid = mean midpoint of all edges attached to that zone.
    Load direction = vector from first Π centroid to second.
    """
    pi_zones = sorted(
        face_df[face_df['is_load_zone'] == 1]['zone_id'].astype(int).tolist())

    if len(pi_zones) < 2:
        print('  [WARN] Fewer than 2 Π zones — defaulting to Z axis.')
        return np.array([0.0, 0.0, 1.0])

    centroids = []
    for z in pi_zones:
        mask = (edge_df['zone_a'] == z) | (edge_df['zone_b'] == z)
        if mask.any():
            centroids.append(
                edge_df.loc[mask, ['mid_x','mid_y','mid_z']].values.mean(axis=0))

    if len(centroids) < 2:
        print('  [WARN] Cannot compute load direction from edges — defaulting to Z.')
        return np.array([0.0, 0.0, 1.0])

    v   = centroids[1] - centroids[0]
    mag = np.linalg.norm(v)
    return v / mag if mag > EPS else np.array([0.0, 0.0, 1.0])


# ═══════════════════════════════════════════════════════════════════════════════
# SURFACE / INTERIOR SPLIT
# ═══════════════════════════════════════════════════════════════════════════════

def classify_surface_interior(node_df, verts_mm, tol=0.01):
    """
    Split all nodes into surface (STL vertex match) and interior.
    Surface match: nearest STL vertex within tol mm.
    """
    tree       = cKDTree(verts_mm.reshape(-1, 3))
    coords     = node_df[['x','y','z']].values.astype(float)
    dists, _   = tree.query(coords, k=1)
    is_surface = dists <= tol

    df              = node_df.copy()
    df['is_surface']= is_surface

    return df[is_surface].copy(), df[~is_surface].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERIOR ZONE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_interior_zones(interior_df, surface_df, face_df,
                              edge_df, load_dir, sv_factor):
    """
    Classify each interior node by zone type.

    Zone types:
      Pi        — near center of load axis, net-3 seeding from center
      E         — bulk interior away from Pi and transition; cylinder pattern
      B         — 2-2-1 transition layers from Π face inward
      B_sphere  — at load midpoint, equidistant from both Π faces;
                   resolved by Sphere of Influence instead of planar gradient

    2-2-1 layer assignment (from nearest Π face):
      Layer 1: within depth × 0.4   (Pi-weighted, ×2 repetitions)
      Layer 2: within depth × 0.8   (mixed, ×2 repetitions)
      Layer 3: within depth × 1.0   (B proper, ×1 repetition)

    Sphere criterion:
      |d_to_face_A - d_to_face_B| < sv_depth × 0.3
      AND node is within sv_depth of at least one Π face
      → load path intersection, gradient undefined → sphere
    """
    int_coords  = interior_df[['x','y','z']].values.astype(float)
    sur_coords  = surface_df[['x','y','z']].values.astype(float)

    # Load projections of all interior nodes
    load_proj   = int_coords @ load_dir

    # Distance to nearest surface node
    sur_tree    = cKDTree(sur_coords)
    dist_surf, _= sur_tree.query(int_coords, k=1)

    # Π face proxy centroids from edge midpoints
    pi_zones    = set(face_df[face_df['is_load_zone'] == 1]['zone_id'].astype(int))
    pi_centroids= []
    for z in sorted(pi_zones):
        m = (edge_df['zone_a'] == z) | (edge_df['zone_b'] == z)
        if m.any():
            pi_centroids.append(
                edge_df.loc[m, ['mid_x','mid_y','mid_z']].values.mean(axis=0))

    if not pi_centroids:
        pi_centroids = [sur_coords.mean(axis=0)]

    pi_tree     = cKDTree(np.array(pi_centroids))
    dist_pi, _  = pi_tree.query(int_coords, k=1)

    # St. Venant depth
    h_vals       = edge_df[edge_df['regime'] != 'Ε/Ε']['h_edge'].values \
                   if len(edge_df) > 0 else np.array([5.0])
    h_edge_min   = float(h_vals.min()) if len(h_vals) > 0 else 5.0
    sv_depth     = sv_factor * h_edge_min

    # 2-2-1 cumulative depth fractions
    beta_w       = [b / sum(BETA_SEQ) for b in BETA_SEQ]  # [0.4, 0.4, 0.2]
    d1           = sv_depth * beta_w[0]            # 0.4 × depth
    d2           = sv_depth * (beta_w[0]+beta_w[1])# 0.8 × depth
    d3           = sv_depth                        # 1.0 × depth

    # Π face load projections (their centroids projected onto load axis)
    pi_load      = [c @ load_dir for c in pi_centroids]
    pi_load_min  = min(pi_load)
    pi_load_max  = max(pi_load)
    center_proj  = (pi_load_min + pi_load_max) / 2.0

    dist_front   = np.abs(load_proj - pi_load_min)
    dist_back    = np.abs(load_proj - pi_load_max)
    dist_nearest = np.minimum(dist_front, dist_back)

    # h values
    h_pi  = float(face_df[face_df['is_load_zone']==1]['edge_length_min'].mean()) \
            if (face_df['is_load_zone']==1).any() else h_edge_min
    h_e   = float(face_df['edge_length_min'].max())
    h_b   = float(np.sqrt(h_pi * h_e))   # TSA geometric mean

    zone_types, layers, h_targets, patterns = [], [], [], []

    for i in range(len(int_coords)):
        lp  = load_proj[i]
        dn  = dist_nearest[i]
        df_ = dist_front[i]
        db_ = dist_back[i]

        # B_sphere: equidistant from both Π faces AND within sv_depth
        if dn <= sv_depth and abs(df_ - db_) < sv_depth * 0.3:
            zone_types.append('B_sphere')
            layers.append(None)
            h_targets.append(round(h_b * 0.5, 4))
            patterns.append('b_sphere')
            continue

        # 2-2-1 transition from nearest Π face
        if dn <= d1:
            zone_types.append('B')
            layers.append(1)
            h_targets.append(round(h_pi * 1.2, 4))
            patterns.append('b_transition')
        elif dn <= d2:
            zone_types.append('B')
            layers.append(2)
            h_targets.append(round(h_b, 4))
            patterns.append('b_transition')
        elif dn <= d3:
            zone_types.append('B')
            layers.append(3)
            h_targets.append(round(h_b * 1.1, 4))
            patterns.append('b_transition')
        else:
            # Bulk interior
            if abs(lp - center_proj) < h_edge_min:
                zone_types.append('Pi')
                layers.append(None)
                h_targets.append(round(h_pi, 4))
                patterns.append('pi_center')
            else:
                zone_types.append('E')
                layers.append(None)
                h_targets.append(round(h_e, 4))
                patterns.append('e_cylinder')

    out             = interior_df.copy()
    out['load_proj']= np.round(load_proj, 4)
    out['dist_surf']= np.round(dist_surf, 4)
    out['dist_pi']  = np.round(dist_pi, 4)
    out['zone_type']= zone_types
    out['layer']    = layers
    out['h_target'] = h_targets
    out['pattern']  = patterns

    return out, sv_depth, h_edge_min, h_pi, h_e, h_b


# ═══════════════════════════════════════════════════════════════════════════════
# CYLINDER E PATTERN — period-2, 4 nodes per line minimum
# ═══════════════════════════════════════════════════════════════════════════════

def apply_cylinder_pattern(interior_df, load_dir, h_e):
    """
    Group E-classified nodes into lines along load axis.
    Active at period-2 (even line index), minimum 4 nodes per line.
    """
    interior_df = interior_df.copy()
    interior_df['cyl_active']  = False
    interior_df['cyl_line_id'] = -1

    e_mask = interior_df['zone_type'] == 'E'
    if not e_mask.any():
        return interior_df

    e_idx    = interior_df.index[e_mask]
    coords   = interior_df.loc[e_idx, ['x','y','z']].values.astype(float)

    # Two transverse axes perpendicular to load_dir
    perp1 = np.array([load_dir[1], -load_dir[0], 0.0])
    if np.linalg.norm(perp1) < EPS:
        perp1 = np.array([0.0, load_dir[2], -load_dir[1]])
    perp1 /= np.linalg.norm(perp1) + EPS
    perp2  = np.cross(load_dir, perp1)
    perp2 /= np.linalg.norm(perp2) + EPS

    t1       = coords @ perp1
    t2       = coords @ perp2
    snap     = h_e * 0.8
    g1       = np.round(t1 / snap).astype(int)
    g2       = np.round(t2 / snap).astype(int)
    keys     = list(zip(g1, g2))
    unique   = sorted(set(keys))
    lid_map  = {k: i for i, k in enumerate(unique)}
    lids     = np.array([lid_map[k] for k in keys])

    # Minimum 4 nodes and period-2
    counts       = pd.Series(lids).value_counts()
    active_lines = {lid for lid, cnt in counts.items()
                    if lid % 2 == 0 and cnt >= 4}

    interior_df.loc[e_idx, 'cyl_active']  = [lid in active_lines for lid in lids]
    interior_df.loc[e_idx, 'cyl_line_id'] = lids

    return interior_df


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGICAL ENTROPY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_topological_entropy(face_df, edge_df, corner_df, interior_df):
    """
    H = -Σ p_i × log2(p_i) over weighted entity type distribution.

    Surface entities:
      face_Π, face_Ε, edge_Π/Π, edge_Π/Ε, corner_Π/Π, corner_Π/Ε
      (Ε/Ε edges and corners excluded — no physics concentration)

    Interior entities:
      Pi, E, B layers weighted by BETA_SEQ / sum = [0.4, 0.4, 0.2]
      B_sphere → weight 1.0 (fully resolved — same complexity as Π)
    """
    beta_w = [b / sum(BETA_SEQ) for b in BETA_SEQ]
    counts = {}

    # Surface faces
    for _, r in face_df.iterrows():
        t = 'Π' if r['is_load_zone'] == 1 else 'Ε'
        counts[f'face_{t}'] = counts.get(f'face_{t}', 0) + 1

    # Surface edges
    if edge_df is not None and len(edge_df) > 0:
        for _, r in edge_df.iterrows():
            if r['regime'] != 'Ε/Ε':
                k = f'edge_{r["regime"]}'
                counts[k] = counts.get(k, 0) + 1

    # Surface corners
    if corner_df is not None and len(corner_df) > 0:
        for _, r in corner_df.iterrows():
            if r['regime'] != 'Ε/Ε corner':
                k = f'corner_{r["regime"].replace(" ","_")}'
                counts[k] = counts.get(k, 0) + 1

    # Interior
    for _, r in interior_df.iterrows():
        zt = r['zone_type']
        if zt == 'B':
            layer = r['layer']
            w     = beta_w[int(layer)-1] if layer and 1 <= int(layer) <= 3 else 0.2
            counts['interior_B'] = counts.get('interior_B', 0) + w
        elif zt == 'B_sphere':
            counts['interior_B_sphere'] = counts.get('interior_B_sphere', 0) + 1.0
        elif zt in ('Pi', 'E'):
            counts[f'interior_{zt}'] = counts.get(f'interior_{zt}', 0) + 1.0

    total   = sum(counts.values()) + EPS
    probs   = {k: v / total for k, v in counts.items()}
    entropy = float(-sum(p * np.log2(p + EPS) for p in probs.values()))

    return entropy, counts, probs


# ═══════════════════════════════════════════════════════════════════════════════
# ANSYS INTERIOR INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_interior_instructions(interior_df, face_df,
                                 sv_depth, h_edge_min,
                                 h_pi, h_e, h_b,
                                 entropy, counts, has_e_interior):
    SEP  = '═' * 68
    SEP2 = '─' * 68

    print(f'\n  {SEP}')
    print(f'  INTERIOR MESH INSTRUCTIONS')
    print(f'  {SEP}')
    print(f'  St. Venant depth   : {sv_depth:.4f} mm')
    print(f'  Interior E zone    : '
          f'{"EXISTS — pure E region between two transition fronts"    if has_e_interior else "ABSENT — transition fronts overlap (thin geometry or large sv_factor)"}')
    print(f'  Topological entropy: {entropy:.4f} bits')
    if entropy < 1.5:
        print(f'  → LOW entropy. Strong Π/Ε separation. '
              f'Intentional mesh advantage: substantial.')
    elif entropy < 2.5:
        print(f'  → MODERATE entropy. Mixed load path. '
              f'Intentional mesh advantage: moderate.')
    else:
        print(f'  → HIGH entropy. Complex load path. '
              f'Intentional mesh advantage: significant — validate against reference.')

    vc = interior_df['zone_type'].value_counts().to_dict()
    print(f'\n  Interior node distribution:')
    for t, n in sorted(vc.items()):
        print(f'    {t:>10}: {n:>5} nodes')

    step = 1

    # 1. Global body sizing
    print(f'\n  {SEP2}')
    print(f'  Step {step} — Body Sizing  (global interior baseline)')
    print(f'  {SEP2}')
    print(f'  Body Sizing: Element Size = {h_e:.4f}, Behavior = Soft')
    print(f'  Applies to: entire body volume')
    step += 1

    # 2. B transition — inflation from each Π face
    if (interior_df['zone_type'] == 'B').any():
        print(f'\n  {SEP2}')
        print(f'  Step {step} — Inflation (2-2-1 B transition from each Π face)')
        print(f'  {SEP2}')
        print(f'  Inflation from each Π face inward along load axis:')
        print(f'    Layer group 1  Pi-weighted  (×2 layers): '
              f'h = {round(h_pi*1.2,4):.4f}')
        print(f'    Layer group 2  mixed        (×2 layers): '
              f'h = {round(h_b,4):.4f}')
        print(f'    Layer group 3  B proper     (×1 layer) : '
              f'h = {round(h_b*1.1,4):.4f}')
        print(f'  Total layers per Π face: 5  (2 + 2 + 1)')
        step += 1

    # 3. Pi center seeding
    if (interior_df['zone_type'] == 'Pi').any():
        pi_nodes = interior_df[interior_df['zone_type'] == 'Pi']
        ctr      = pi_nodes[['x','y','z']].values.mean(axis=0)
        print(f'\n  {SEP2}')
        print(f'  Step {step} — Sphere of Influence  (Pi center zone)')
        print(f'  {SEP2}')
        print(f'  Centre of Pi zone  : '
              f'({ctr[0]:.3f}, {ctr[1]:.3f}, {ctr[2]:.3f})')
        print(f'  Sphere of Influence: '
              f'radius = {h_edge_min:.4f}, h = {h_pi:.4f}, Hard')
        print(f'  Seeding strategy   : '
              f'net-3 outward (5 forward — 2 back) along load axis')
        print(f'  ({len(pi_nodes)} nodes classified as Pi interior)')
        step += 1

    # 4. E cylinder lines
    if 'cyl_active' in interior_df.columns:
        active_e = interior_df[interior_df['cyl_active'] == True]
        if len(active_e) > 0:
            n_lines = int(active_e['cyl_line_id'].nunique())
            print(f'\n  {SEP2}')
            print(f'  Step {step} — Edge Sizing  (E cylinder lines, period-2)')
            print(f'  {SEP2}')
            print(f'  Active cylinder lines: {n_lines}  '
                  f'(every other transverse line)')
            print(f'  Edge Sizing: '
                  f'Element Size = {h_e:.4f}, min 4 divisions, Soft')
            print(f'  Inactive lines carry global body size — no local control')
            step += 1

    # 5. B sphere corrections
    b_sph = interior_df[interior_df['zone_type'] == 'B_sphere']
    if len(b_sph) > 0:
        ctr_s = b_sph[['x','y','z']].values.mean(axis=0)
        h_s   = float(b_sph['h_target'].mean())
        r_s   = round(h_s * 2.0, 4)
        print(f'\n  {SEP2}')
        print(f'  Step {step} — Sphere of Influence  (B sphere correction)')
        print(f'  {SEP2}')
        print(f'  {len(b_sph)} node(s) at load path midpoint — '
              f'equidistant from both Π faces.')
        print(f'  Planar gradient replaced by spherical load distribution.')
        print(f'  Centre  : '
              f'({ctr_s[0]:.3f}, {ctr_s[1]:.3f}, {ctr_s[2]:.3f})')
        print(f'  Sphere of Influence: radius = {r_s:.4f}, h = {h_s:.4f}, Soft')
        print(f'  Physical basis: at the load midpoint no preferred direction')
        print(f'  exists — load distributes radially. Sphere mesh correct;')
        print(f'  planar gradient would impose a false directionality.')
        step += 1

    print(f'\n  {SEP}')
    print(f'  Entity count summary (for entropy verification):')
    for k, v in sorted(counts.items()):
        print(f'    {k:>30}: {v:.2f}')
    print(f'  {SEP}\n')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='interior_probe — interior classification + topological entropy')
    ap.add_argument('--zones',     required=True,
                    help='Face zone CSV (probe_input.csv)')
    ap.add_argument('--nodes',     required=True,
                    help='Full node file (mesh_file.txt)')
    ap.add_argument('--stl',       required=True,
                    help='Binary STL (stl_mesh_file.stl)')
    ap.add_argument('--sv_factor', type=float, default=2.0,
                    help='St. Venant depth multiplier (default 2.0). '
                         'Depth = sv_factor × min(h_edge). '
                         'Controls how far 2-2-1 transition extends inward.')
    ap.add_argument('--out',       default=None,
                    help='Output CSV (default: zones_stem + _interior.csv)')
    args = ap.parse_args()

    SEP = '=' * 68

    base        = args.zones.replace('.csv', '')
    edge_path   = base + '_edges.csv'
    corner_path = base + '_corners.csv'
    out_path    = args.out or base + '_interior.csv'

    print(f'\n{SEP}')
    print(f'  INTERIOR PROBE')
    print(f'{SEP}')

    # ── Load CSVs ──────────────────────────────────────────────────
    face_df   = pd.read_csv(args.zones)

    edge_df   = pd.read_csv(edge_path) if os.path.exists(edge_path) \
                else pd.DataFrame(columns=[
                    'edge_id','zone_a','zone_b','regime',
                    'h_edge','mid_x','mid_y','mid_z'])

    corner_df = pd.read_csv(corner_path) if os.path.exists(corner_path) \
                else pd.DataFrame()

    print(f'\n  Face zones     : {len(face_df)}')
    print(f'  Edge boundaries: {len(edge_df)}')
    print(f'  Corner points  : {len(corner_df)}')

    # ── Load nodes ─────────────────────────────────────────────────
    node_df  = pd.read_csv(args.nodes, sep='\t')
    node_df.columns = [c.strip() for c in node_df.columns]
    col_map  = {}
    for c in node_df.columns:
        cl = c.lower()
        if 'node' in cl or 'number' in cl: col_map[c] = 'node_id'
        elif cl.strip() == 'x location (mm)' or cl == 'x': col_map[c] = 'x'
        elif cl.strip() == 'y location (mm)' or cl == 'y': col_map[c] = 'y'
        elif cl.strip() == 'z location (mm)' or cl == 'z': col_map[c] = 'z'
    node_df  = node_df.rename(columns=col_map)
    for c in ['node_id','x','y','z']:
        if c not in node_df.columns:
            # fallback: positional
            node_df.columns = ['node_id','x','y','z'] + \
                              list(node_df.columns[4:])
            break
    node_df  = node_df[['node_id','x','y','z']].copy()
    node_df[['x','y','z']] = node_df[['x','y','z']].astype(float)
    node_coords = node_df[['x','y','z']].values
    print(f'  Nodes total    : {len(node_df)}')

    # ── STL ────────────────────────────────────────────────────────
    _, stl_verts = read_stl_binary(args.stl)
    scale        = detect_unit_scale(stl_verts, node_coords)
    verts_mm     = stl_verts * scale

    # ── Surface / interior split ───────────────────────────────────
    surface_df, interior_df = classify_surface_interior(node_df, verts_mm)
    print(f'  Surface nodes  : {len(surface_df)}')
    print(f'  Interior nodes : {len(interior_df)}')

    if len(interior_df) == 0:
        print('\n  [WARN] No interior nodes found. '
              'Mesh may be surface-only. Exiting.')
        sys.exit(0)

    # ── Load direction ─────────────────────────────────────────────
    load_dir = infer_load_direction(face_df, edge_df)
    print(f'\n  Load direction : '
          f'({load_dir[0]:+.3f}, {load_dir[1]:+.3f}, {load_dir[2]:+.3f})')

    # ── Derive edge regime and h_edge from face types ─────────────
    type_map = {int(r['zone_id']): ('Π' if r['is_load_zone']==1 else 'Ε')
                for _, r in face_df.iterrows()}
    h_map    = {int(r['zone_id']): float(r['edge_length_min'])
                for _, r in face_df.iterrows()}

    if len(edge_df) > 0 and 'regime' not in edge_df.columns:
        regimes, h_edges = [], []
        for _, r in edge_df.iterrows():
            ta = type_map.get(int(r['zone_a']), 'Ε')
            tb = type_map.get(int(r['zone_b']), 'Ε')
            ha = h_map.get(int(r['zone_a']), 5.0)
            hb = h_map.get(int(r['zone_b']), 5.0)
            pair = tuple(sorted([ta, tb]))
            if pair == ('Π', 'Π'):
                regime = 'Π/Π'; h_edge = float(np.sqrt(ha * hb))
            elif 'Π' in pair:
                regime = 'Π/Ε'; h_edge = float(np.sqrt(ha * hb))
            else:
                regime = 'Ε/Ε'; h_edge = max(ha, hb)
            regimes.append(regime)
            h_edges.append(round(h_edge, 4))
        edge_df = edge_df.copy()
        edge_df['regime'] = regimes
        edge_df['h_edge'] = h_edges

    if len(corner_df) > 0 and 'regime' not in corner_df.columns:
        c_regimes = []
        for _, r in corner_df.iterrows():
            adj   = [int(z) for z in str(r['adjacent_zones']).split(',')]
            types = [type_map.get(z, 'Ε') for z in adj]
            n_pi  = types.count('Π')
            if n_pi >= 2:   c_regimes.append('Π/Π corner')
            elif n_pi == 1: c_regimes.append('Π/Ε corner')
            else:           c_regimes.append('Ε/Ε corner')
        corner_df = corner_df.copy()
        corner_df['regime'] = c_regimes

    # ── Interior classification ────────────────────────────────────
    print(f'\n  Classifying interior nodes  '
          f'(sv_factor = {args.sv_factor})...')
    interior_df, sv_depth, h_edge_min, h_pi, h_e, h_b = \
        classify_interior_zones(
            interior_df, surface_df, face_df,
            edge_df, load_dir, args.sv_factor)

    # ── Cylinder pattern ───────────────────────────────────────────
    interior_df  = apply_cylinder_pattern(interior_df, load_dir, h_e)
    has_e        = (interior_df['zone_type'] == 'E').any()

    # ── Entropy ────────────────────────────────────────────────────
    entropy, counts, probs = compute_topological_entropy(
        face_df,
        edge_df   if len(edge_df)   > 0 else None,
        corner_df if len(corner_df) > 0 else None,
        interior_df)

    # ── Print instructions ─────────────────────────────────────────
    print_interior_instructions(
        interior_df, face_df,
        sv_depth, h_edge_min, h_pi, h_e, h_b,
        entropy, counts, has_e)

    # ── Save ───────────────────────────────────────────────────────
    surface_df   = surface_df.copy()
    surface_df['is_surface']  = True
    surface_df['zone_type']   = 'surface'
    surface_df['pattern']     = 'surface'
    for col in ['load_proj','dist_surf','dist_pi',
                'layer','h_target','cyl_active','cyl_line_id']:
        surface_df[col] = np.nan

    cols = ['node_id','x','y','z','is_surface','zone_type',
            'pattern','layer','h_target','load_proj',
            'dist_surf','dist_pi','cyl_active','cyl_line_id']
    for col in cols:
        if col not in interior_df.columns:
            interior_df[col] = np.nan
        if col not in surface_df.columns:
            surface_df[col]  = np.nan

    all_nodes = pd.concat([
        surface_df[cols], interior_df[cols]
    ], ignore_index=True)

    all_nodes.to_csv(out_path, index=False)
    print(f'  Saved: {out_path}')
    print(f'\n{SEP}\n')


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE
# ═══════════════════════════════════════════════════════════════════════════════
#
#   python interior_probe.py --zones probe_input.csv
#                            --nodes mesh_file.txt
#                            --stl   stl_mesh_file.stl
#
#   python interior_probe.py --zones probe_input.csv
#                            --nodes mesh_file.txt
#                            --stl   stl_mesh_file.stl
#                            --sv_factor 3.0
#
# Full pipeline:
#   python mesh_to_probe.py  --nodes mesh_file.txt --stl stl_mesh_file.stl
#                            --load_nodes 2 --out probe_input.csv
#   python mesh_probe.py     --zones probe_input.csv --no_probe
#   python interior_probe.py --zones probe_input.csv
#                            --nodes mesh_file.txt
#                            --stl   stl_mesh_file.stl
#
# Arguments:
#   --zones      Face zone CSV from mesh_to_probe.py         [required]
#   --nodes      Full node file from mesher                  [required]
#   --stl        Binary STL from same mesher export          [required]
#   --sv_factor  St. Venant depth multiplier (default 2.0)   [optional]
#   --out        Output CSV path                             [optional]
#
# Dependencies:
#   pip install numpy pandas scipy
