"""
mesh_to_probe.py — Preprocessing: STL + node file → probe_input.csv
=====================================================================
Standalone. Run this before mesh_probe.py.

Reads:
    mesh_file.txt     — tab-separated: Node Number, X, Y, Z (mm)
    stl_mesh_file.stl — binary STL exported from the same mesher
                        (coordinates may be in metres; auto-detected)

Produces:
    probe_input.csv   — one row per zone, ready for mesh_probe.py --zones

Zone detection:
    Clusters STL triangles by face normal direction (angular tolerance
    configurable, default 5°). Each unique normal direction = one zone.
    Works directly for basic shapes (box, cylinder, cone, flat plate).
    For curved surfaces, triangles with smoothly varying normals are
    clustered by proximity of normal direction on the unit sphere.

Curvature estimation:
    Computed from dihedral angle between adjacent triangle pairs sharing
    an edge, divided by shared edge length.
    κ ≈ dihedral_angle (rad) / edge_length
    For flat faces: dihedral = 0 → κ = 0 (correct).
    For curved faces: dihedral > 0 → κ > 0 (approximate, sufficient
    for TSA classification — not a substitute for quadric fitting).

dist_to_bc:
    If --bc_zones is provided, computed as Euclidean distance from each
    zone centroid to the nearest BC zone centroid.
    If omitted, defaults to a large value (BC term contributes nothing
    to the probe score — conservative, safe default).

Usage:
    python mesh_to_probe.py --nodes mesh_file.txt --stl stl_mesh_file.stl
    python mesh_to_probe.py --nodes mesh_file.txt --stl stl_mesh_file.stl --out my_input.csv
    python mesh_to_probe.py --nodes mesh_file.txt --stl stl_mesh_file.stl --bc_zones 1 3
    python mesh_to_probe.py --nodes mesh_file.txt --stl stl_mesh_file.stl --normal_tol 10

Dependencies:
    pip install numpy pandas scipy
"""

import numpy as np
import pandas as pd
import struct
import argparse
import sys
from scipy.spatial import cKDTree


# ═══════════════════════════════════════════════════════════════════════════════
# STL READER — binary format (Ansys Mechanical default export)
# ═══════════════════════════════════════════════════════════════════════════════

def read_stl_binary(path):
    """
    Read binary STL. Returns:
        normals : (N, 3) float — face normals as exported
        verts   : (N, 3, 3) float — triangle vertices in STL units
    """
    with open(path, 'rb') as f:
        f.read(80)                                   # header
        n_tri = struct.unpack('<I', f.read(4))[0]
        normals = np.zeros((n_tri, 3), dtype=np.float64)
        verts   = np.zeros((n_tri, 3, 3), dtype=np.float64)
        for i in range(n_tri):
            data          = f.read(50)
            normals[i]    = struct.unpack('<fff', data[0:12])
            verts[i, 0]   = struct.unpack('<fff', data[12:24])
            verts[i, 1]   = struct.unpack('<fff', data[24:36])
            verts[i, 2]   = struct.unpack('<fff', data[36:48])
            # last 2 bytes: attribute byte count (ignored)
    return normals, verts


def detect_unit_scale(stl_verts, node_coords):
    """
    Auto-detect coordinate scale factor between STL and node file.
    Compares bounding box diagonal of each. Returns scale to multiply
    STL coordinates by to match node file units.
    Common case: STL in metres, node file in mm → scale = 1000.
    """
    stl_flat  = stl_verts.reshape(-1, 3)
    stl_diag  = np.linalg.norm(stl_flat.max(axis=0) - stl_flat.min(axis=0))
    node_diag = np.linalg.norm(node_coords.max(axis=0) - node_coords.min(axis=0))

    if stl_diag < 1e-9:
        return 1.0

    ratio = node_diag / stl_diag
    # Round to nearest power of 10
    if   ratio > 500:   return 1000.0
    elif ratio > 50:    return 100.0
    elif ratio > 5:     return 10.0
    elif ratio > 0.5:   return 1.0
    elif ratio > 0.05:  return 0.1
    else:               return 0.001


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE DETECTION — cluster triangles by face normal direction
# ═══════════════════════════════════════════════════════════════════════════════

def cluster_by_normal(normals, tol_deg=5.0):
    """
    Cluster triangles into zones by normal direction.
    Each unique normal direction (within tol_deg) = one zone.

    Returns:
        zone_ids : (N,) int — zone index per triangle (0-based)
        zone_normals : (K, 3) — representative normal per zone
    """
    tol_rad   = np.deg2rad(tol_deg)
    n_tri     = len(normals)
    zone_ids  = -np.ones(n_tri, dtype=int)
    zone_reps = []                               # representative normals

    # Normalise (re-normalise exported normals to be safe)
    norms_mag = np.linalg.norm(normals, axis=1, keepdims=True)
    norms_mag = np.where(norms_mag < 1e-12, 1.0, norms_mag)
    normals_n = normals / norms_mag

    for i in range(n_tri):
        n = normals_n[i]
        matched = False
        for k, rep in enumerate(zone_reps):
            # angle between normals: cos(θ) = dot product (both unit vectors)
            dot   = np.clip(np.dot(n, rep), -1.0, 1.0)
            angle = np.arccos(dot)
            if angle < tol_rad:
                zone_ids[i] = k
                matched = True
                break
        if not matched:
            zone_ids[i] = len(zone_reps)
            zone_reps.append(n.copy())

    return zone_ids, np.array(zone_reps)


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE GEOMETRY — aspect ratio, skewness, area, edge lengths
# ═══════════════════════════════════════════════════════════════════════════════

def triangle_metrics(verts):
    """
    Compute per-triangle geometry metrics.
    verts : (N, 3, 3) — triangle vertices

    Returns dict of arrays, each length N:
        area, edge_min, edge_max, aspect_ratio, skewness, centroid (N,3)
    """
    v0, v1, v2 = verts[:, 0], verts[:, 1], verts[:, 2]

    e0 = v1 - v0    # edge opposite vertex 2
    e1 = v2 - v1    # edge opposite vertex 0
    e2 = v0 - v2    # edge opposite vertex 1

    l0 = np.linalg.norm(e0, axis=1)
    l1 = np.linalg.norm(e1, axis=1)
    l2 = np.linalg.norm(e2, axis=1)

    cross    = np.cross(e0, -e2)
    area     = 0.5 * np.linalg.norm(cross, axis=1)

    edge_min = np.minimum(np.minimum(l0, l1), l2)
    edge_max = np.maximum(np.maximum(l0, l1), l2)

    # Aspect ratio: longest / shortest edge (simple, practical)
    eps = 1e-12
    aspect = edge_max / np.maximum(edge_min, eps)

    # Skewness: equilateral angle deviation
    # For each triangle compute angles at each vertex
    # angle at v0: between e_v0v1 and e_v0v2
    def safe_angle(a, b):
        dot = np.einsum('ij,ij->i', a, b)
        na  = np.linalg.norm(a, axis=1)
        nb  = np.linalg.norm(b, axis=1)
        cos = np.clip(dot / np.maximum(na * nb, eps), -1, 1)
        return np.arccos(cos)

    ang0 = safe_angle( e0,  -e2)    # at v0
    ang1 = safe_angle(-e0,  e1)     # at v1
    ang2 = safe_angle(-e1,  e2)     # at v2

    theta_eq  = np.deg2rad(60.0)
    theta_max = np.maximum(np.maximum(ang0, ang1), ang2)
    theta_min = np.minimum(np.minimum(ang0, ang1), ang2)
    skewness  = np.maximum(
        (theta_max - theta_eq) / (np.pi - theta_eq),
        (theta_eq  - theta_min) / theta_eq
    )
    skewness  = np.clip(skewness, 0, 1)

    centroid  = (v0 + v1 + v2) / 3.0

    return {
        'area':         area,
        'edge_min':     edge_min,
        'edge_max':     edge_max,
        'aspect_ratio': aspect,
        'skewness':     skewness,
        'centroid':     centroid,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CURVATURE ESTIMATION — from dihedral angles between adjacent triangles
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_curvature_per_zone(verts, normals_n, zone_ids, n_zones, scale):
    """
    Estimate kappa_mean and kappa_max per zone.

    For each pair of triangles in the same zone sharing an edge:
        κ = dihedral_angle (rad) / shared_edge_length
    kappa_mean = mean κ across all shared edges in zone
    kappa_max  = max κ across all shared edges in zone

    For flat faces: dihedral = 0 → κ = 0.
    Units: 1/mm (matching node file units after scaling).
    """
    eps           = 1e-12
    kappa_means   = np.zeros(n_zones)
    kappa_maxs    = np.zeros(n_zones)

    # Build edge → triangle lookup per zone
    # Represent each edge as a sorted tuple of vertex coordinate tuples
    # (rounded to avoid float equality issues)

    def round_coord(v, decimals=4):
        return tuple(np.round(v * scale, decimals))

    for z in range(n_zones):
        mask     = (zone_ids == z)
        idx      = np.where(mask)[0]
        if len(idx) < 2:
            continue

        edge_map = {}    # edge → list of (tri_idx, local_edge_idx)
        for i in idx:
            tri = verts[i]    # (3, 3)
            for e in range(3):
                va = round_coord(tri[e])
                vb = round_coord(tri[(e+1) % 3])
                key = tuple(sorted([va, vb]))
                edge_map.setdefault(key, []).append(i)

        kappas = []
        for key, tris in edge_map.items():
            if len(tris) != 2:
                continue    # boundary edge or non-manifold
            i0, i1 = tris
            n0 = normals_n[i0]
            n1 = normals_n[i1]
            dot      = np.clip(np.dot(n0, n1), -1.0, 1.0)
            dihedral = np.arccos(dot)   # radians

            # shared edge length in mm
            va = np.array(key[0]) / scale * scale   # already scaled by round_coord
            vb = np.array(key[1]) / scale * scale
            # Simpler: recompute from original verts (already in mm after caller scales)
            # key stores scaled mm coords
            va_mm = np.array(key[0])
            vb_mm = np.array(key[1])
            edge_len = np.linalg.norm(va_mm - vb_mm)

            if edge_len > eps:
                kappas.append(dihedral / edge_len)

        if kappas:
            kappa_means[z] = float(np.mean(kappas))
            kappa_maxs[z]  = float(np.max(kappas))

    return kappa_means, kappa_maxs


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE ADJACENCY — which zones share an edge in the STL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_zone_adjacency(verts_mm, zone_ids, n_zones):
    """
    Find which zones are adjacent (share at least one edge in the STL).

    Two zones are adjacent if they have a pair of triangles — one from
    each zone — that share an edge (same two vertex coordinates).

    Returns:
        adjacency : dict {zone_index_0based: set of adjacent zone_index_0based}
    """
    def round_v(v, decimals=4):
        return tuple(np.round(v, decimals))

    edge_to_zones = {}
    for i, (tri, z) in enumerate(zip(verts_mm, zone_ids)):
        for e in range(3):
            va  = round_v(tri[e])
            vb  = round_v(tri[(e + 1) % 3])
            key = tuple(sorted([va, vb]))
            edge_to_zones.setdefault(key, set()).add(int(z))

    adjacency = {z: set() for z in range(n_zones)}
    for zones in edge_to_zones.values():
        zlist = list(zones)
        for i in range(len(zlist)):
            for j in range(i + 1, len(zlist)):
                adjacency[zlist[i]].add(zlist[j])
                adjacency[zlist[j]].add(zlist[i])

    return adjacency


# ═══════════════════════════════════════════════════════════════════════════════
# NODE → ZONE RESOLUTION
# Finds which zone(s) a set of node IDs belong to by matching node
# coordinates to STL triangle vertices.
# Both files share the same coordinate system after unit scaling —
# the match is exact within floating point tolerance.
# ═══════════════════════════════════════════════════════════════════════════════

def find_zones_for_nodes(node_ids, nodes_df, verts_mm, zone_ids_arr, tol=0.01):
    """
    Given a list of node IDs, return the set of zone IDs (1-based) that
    contain those nodes as STL triangle vertices.

    node_ids    : list of int — node IDs from the node file (1-based)
    nodes_df    : DataFrame with columns node_id, x, y, z (mm)
    verts_mm    : (N_tri, 3, 3) float — STL triangle vertices in mm
    zone_ids_arr: (N_tri,) int — 0-based zone index per triangle
    tol         : float — coordinate match tolerance in mm (default 0.01)

    Returns:
        matched_zones : set of int — 1-based zone IDs
        unmatched     : list of int — node IDs that had no matching STL vertex
    """
    # Build KD-tree of all unique STL vertices
    all_verts  = verts_mm.reshape(-1, 3)               # (N_tri×3, 3)
    tri_idx    = np.repeat(np.arange(len(verts_mm)), 3) # which triangle each vertex belongs to
    tree       = cKDTree(all_verts)

    matched_zones = set()
    unmatched     = []

    node_lookup = nodes_df.set_index('node_id')[['x', 'y', 'z']]

    for nid in node_ids:
        if nid not in node_lookup.index:
            print(f'  [WARN] Node ID {nid} not found in node file — skipped.')
            unmatched.append(nid)
            continue

        coord = node_lookup.loc[nid].values.astype(np.float64)
        dist, idx = tree.query(coord, k=1)

        if dist > tol:
            print(f'  [WARN] Node {nid} at {coord} has no STL vertex within '
                  f'{tol}mm (nearest: {dist:.4f}mm). '
                  f'Node may be interior (not on surface) — skipped.')
            unmatched.append(nid)
            continue

        tri_i     = int(tri_idx[idx])
        zone_1based = int(zone_ids_arr[tri_i]) + 1
        matched_zones.add(zone_1based)
        print(f'  Node {nid:>4} at ({coord[0]:+.2f}, {coord[1]:+.2f}, {coord[2]:+.2f}) '
              f'→ zone {zone_1based}')

    return matched_zones, unmatched


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AGGREGATION — per-zone statistics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_zone_stats(verts_mm, normals_n, zone_ids, n_zones, bc_zone_ids):
    """
    Aggregate triangle-level metrics to zone-level statistics.

    Returns DataFrame with probe_input.csv columns.
    """
    metrics   = triangle_metrics(verts_mm)
    total_area = metrics['area'].sum()
    eps        = 1e-12

    scale = 1.0
    kappa_means, kappa_maxs = estimate_curvature_per_zone(
        verts_mm, normals_n, zone_ids, n_zones, scale=1.0)

    # Zone adjacency — needed for flat zone reclassification in the probe
    adjacency = compute_zone_adjacency(verts_mm, zone_ids, n_zones)

    # Zone centroids (area-weighted average of triangle centroids)
    zone_centroids = np.zeros((n_zones, 3))
    zone_areas     = np.zeros(n_zones)
    for z in range(n_zones):
        mask = (zone_ids == z)
        a    = metrics['area'][mask]
        c    = metrics['centroid'][mask]
        zone_areas[z]     = a.sum()
        if a.sum() > eps:
            zone_centroids[z] = np.average(c, weights=a, axis=0)

    # dist_to_bc: if BC zones specified, distance to nearest BC zone centroid
    if bc_zone_ids:
        bc_centroids = zone_centroids[np.array(bc_zone_ids)]
        tree         = cKDTree(bc_centroids)
        dist_to_bc, _ = tree.query(zone_centroids)
    else:
        dist_to_bc = np.full(n_zones, 1e6)    # large → BC term = 0 in score

    rows = []
    for z in range(n_zones):
        mask = (zone_ids == z)

        # normal deviation: angular spread of face normals within zone
        zone_normals = normals_n[mask]
        if len(zone_normals) > 1:
            mean_n   = zone_normals.mean(axis=0)
            mean_n  /= max(np.linalg.norm(mean_n), eps)
            dots     = np.clip(zone_normals @ mean_n, -1, 1)
            angles   = np.degrees(np.arccos(dots))
            norm_dev = float(angles.max())
        else:
            norm_dev = 0.0

        # neighbor_ids: 1-based zone IDs of adjacent zones (comma-separated)
        neighbors_1based = sorted([n + 1 for n in adjacency[z]])
        neighbor_str     = ','.join(str(n) for n in neighbors_1based)

        rows.append({
            'zone_id':          z + 1,
            'kappa_mean':       round(kappa_means[z], 6),
            'kappa_max':        round(kappa_maxs[z],  6),
            'aspect_ratio':     round(float(metrics['aspect_ratio'][mask].mean()), 4),
            'skewness':         round(float(metrics['skewness'][mask].max()),  4),
            'dist_to_bc':       round(float(dist_to_bc[z]), 4),
            'normal_deviation': round(norm_dev, 4),
            'area_fraction':    round(float(zone_areas[z] / (total_area + eps)), 6),
            'edge_length_min':  round(float(metrics['edge_min'][mask].min()), 4),
            'neighbor_ids':     neighbor_str,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='mesh_to_probe — combine STL + node file → probe_input.csv')
    ap.add_argument('--nodes',      required=True,
                    help='Node coordinate file (tab-separated: Node Number, X, Y, Z)')
    ap.add_argument('--stl',        required=True,
                    help='Binary STL file exported from the same mesher')
    ap.add_argument('--out',        default='probe_input.csv',
                    help='Output CSV path (default: probe_input.csv)')
    ap.add_argument('--normal_tol', type=float, default=5.0,
                    help='Angular tolerance in degrees for zone clustering (default: 5). '
                         'USER INPUT REQUIRED: check zone count in output matches expected '
                         'number of physical faces. Too tight = zone fragmentation. '
                         'Too loose = adjacent faces merge.')
    ap.add_argument('--load_nodes', type=int, nargs='*', default=[],
                    help='Node IDs (from node file) where physics concentrates — '
                         'applied forces AND constrained surfaces (fixed supports, '
                         'pinned joints). Both create stress concentrations requiring '
                         'the same mesh refinement. '
                         'USER INPUT REQUIRED: not derivable from geometry files. '
                         'Script resolves each node ID to its surface zone. '
                         'Resolved zones are written as is_load_zone=1 in output CSV '
                         'and used to compute dist_to_load (proximity score). '
                         'mesh_probe.py forces those zones to Π automatically.')
    args = ap.parse_args()

    SEP = '=' * 60

    # ── Read node file ─────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  MESH TO PROBE PREPROCESSOR')
    print(f'{SEP}')

    nodes_df = pd.read_csv(args.nodes, sep='\t')
    nodes_df.columns = [c.strip() for c in nodes_df.columns]
    # Normalise column names flexibly
    col_map = {}
    for c in nodes_df.columns:
        cl = c.lower()
        if 'node' in cl or 'number' in cl:  col_map[c] = 'node_id'
        elif 'x' in cl:                      col_map[c] = 'x'
        elif 'y' in cl:                      col_map[c] = 'y'
        elif 'z' in cl:                      col_map[c] = 'z'
    nodes_df = nodes_df.rename(columns=col_map)[['node_id','x','y','z']]
    node_coords = nodes_df[['x','y','z']].values.astype(np.float64)
    print(f'\n  Nodes loaded : {len(nodes_df)}')
    print(f'  Node bbox    : X[{node_coords[:,0].min():.2f}, {node_coords[:,0].max():.2f}] '
          f'Y[{node_coords[:,1].min():.2f}, {node_coords[:,1].max():.2f}] '
          f'Z[{node_coords[:,2].min():.2f}, {node_coords[:,2].max():.2f}]')

    # ── Read STL ───────────────────────────────────────────────────
    stl_normals, stl_verts = read_stl_binary(args.stl)
    scale = detect_unit_scale(stl_verts, node_coords)
    verts_mm = stl_verts * scale
    print(f'\n  STL triangles: {len(stl_verts)}')
    print(f'  Unit scale   : ×{scale:.0f} (STL → node file units)')

    # Re-normalise exported normals
    mag = np.linalg.norm(stl_normals, axis=1, keepdims=True)
    mag = np.where(mag < 1e-12, 1.0, mag)
    normals_n = stl_normals / mag

    # ── Zone detection ─────────────────────────────────────────────
    zone_ids, zone_normals = cluster_by_normal(normals_n, tol_deg=args.normal_tol)
    n_zones = len(zone_normals)
    print(f'\n  Zones detected: {n_zones}  (normal_tol = {args.normal_tol}°)')
    print(f'  {"Zone":>5}  {"Triangles":>9}  Normal direction')
    print(f'  {"─"*5}  {"─"*9}  {"─"*30}')
    for z in range(n_zones):
        n_tri_z = int((zone_ids == z).sum())
        nrm     = zone_normals[z]
        print(f'  {z+1:>5}  {n_tri_z:>9}  '
              f'({nrm[0]:+.3f}, {nrm[1]:+.3f}, {nrm[2]:+.3f})')

    # ── Load nodes → zone resolution (covers both forces and constraints) ──────
    # Active loads and reactive constraints (fixed supports, pinned joints) both
    # create stress concentrations requiring the same mesh refinement.
    # One concept: where does the physics concentrate?
    load_zone_ids = set()
    bc_zone_ids_0 = []   # fed to compute_zone_stats for dist_to_bc computation

    if args.load_nodes:
        print(f'\n  Resolving physics-concentration node IDs to zones...')
        matched, unmatched = find_zones_for_nodes(
            args.load_nodes, nodes_df, verts_mm, zone_ids)
        load_zone_ids = matched
        bc_zone_ids_0 = [z - 1 for z in matched]   # 0-based for dist computation
        if unmatched:
            print(f'  [WARN] {len(unmatched)} node(s) could not be matched '
                  f'to a surface zone: {unmatched}')
        if load_zone_ids:
            print(f'  Physics zones resolved (1-based): {sorted(load_zone_ids)}')
        else:
            print(f'  [WARN] No nodes matched any surface zone. '
                  f'is_load_zone = 0 and dist_to_bc = large value for all zones.')
    else:
        print(f'\n  Load nodes: none specified — '
              f'is_load_zone = 0, dist_to_bc = large value for all zones')

    # ── Compute per-zone statistics ────────────────────────────────
    print(f'\n  Computing zone statistics...')
    df = compute_zone_stats(verts_mm, normals_n, zone_ids, n_zones, bc_zone_ids_0)

    # Add is_load_zone column
    df['is_load_zone'] = df['zone_id'].apply(
        lambda zid: 1 if int(zid) in load_zone_ids else 0)

    # ── Print summary ──────────────────────────────────────────────
    print(f'\n  {"Zone":>5}  {"κ_mean":>8}  {"κ_max":>8}  '
          f'{"AR":>5}  {"Skew":>5}  {"NrmDev":>7}  {"AreaFrac":>9}  {"h_min":>6}  '
          f'{"dist_load":>9}  {"load":>4}')
    print(f'  {"─"*5}  {"─"*8}  {"─"*8}  {"─"*5}  {"─"*5}  {"─"*7}  {"─"*9}  {"─"*6}  '
          f'{"─"*9}  {"─"*4}')
    for _, r in df.iterrows():
        print(f'  {int(r.zone_id):>5}  {r.kappa_mean:>8.4f}  {r.kappa_max:>8.4f}  '
              f'{r.aspect_ratio:>5.2f}  {r.skewness:>5.3f}  '
              f'{r.normal_deviation:>7.2f}  {r.area_fraction:>9.4f}  '
              f'{r.edge_length_min:>6.3f}  {r.dist_to_bc:>9.2f}  '
              f'{int(r.is_load_zone):>4}')

    # ── Save ───────────────────────────────────────────────────────
    df.to_csv(args.out, index=False)
    print(f'\n  Saved: {args.out}')
    print(f'\n  Next step:')
    print(f'    python mesh_probe.py --zones {args.out}')
    if load_zone_ids:
        print(f'    (physics zones {sorted(load_zone_ids)} resolved from '
              f'node IDs {args.load_nodes} — read automatically by probe)')
    print(f'    Add --flag ZONE_ID for geometric uncertainty, '
          f'--load ZONE_ID to override zone assignment manually.')
    print(f'\n{SEP}\n')


if __name__ == '__main__':
    main()
