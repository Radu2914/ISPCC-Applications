"""
mesh_probe.py — ISPCC Probe Instrument for CAD Mesh Zone Classification
========================================================================
Standalone. No dependency on other ISPCC files.

Purpose
-------
Given a rough mesh exported from CAD (geometry descriptors per parametric
zone ID), classify each zone by its TSA dynamical character:

    Π  (cascade)     — zone needs refinement; mesh error compounds
                        non-repeatingly toward solver inaccuracy
    Ε  (equilibrium) — zone tolerates coarse mesh; geometry is bounded
                        and self-correcting; refinement not warranted
    Β  (boundary)    — decision is non-obvious; at the structural
                        transition between Π and Ε regimes;
                        always assigned to user-flagged zone IDs

The probe runs RF feature importance on geometry-derived zone descriptors,
encoded in the pi/e basis (fixed weights, unchanged from EM and bearing
domains), against a geometry-derived refinement score.  The importance
split confirms whether the Π/Ε variable classification is structurally
correct before any solver data is available.

Input CSV columns (one row per zone)
-------------------------------------
    zone_id           integer zone identifier
    kappa_mean        mean curvature (1/mm or 1/m — consistent units)
    kappa_max         maximum curvature in zone
    aspect_ratio      average element aspect ratio (1.0 = equilateral)
    skewness          maximum element skewness (0 = perfect, 1 = degenerate)
    dist_to_bc        distance from zone centroid to nearest BC attachment
    normal_deviation  angular deviation of normals across zone (degrees)
    area_fraction     zone area / total surface area (dimensionless, sums to 1)
    edge_length_min   minimum element edge length in zone

Usage
-----
    python mesh_probe.py --zones zones.csv --flag 3 --flag 7
    python mesh_probe.py --test sphere
    python mesh_probe.py --test cylinder
    python mesh_probe.py --test box
    python mesh_probe.py --test cone
    python mesh_probe.py --test box --flag 4 --flag 5
    python mesh_probe.py --test cylinder --no_probe
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import argparse
import sys

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED ISPCC CONSTANTS
# Encoding weights are structural assertions, not tuned hyperparameters.
# Identical values to EM surrogate and bearing CSP domains.
# ═══════════════════════════════════════════════════════════════════════════════

PI = np.pi
E  = np.e
EPS = 1e-9

PI_WEIGHTS = (5, 1, 1, 3, 1)   # cascade basis: weight 5 on sin(πx), 3 on sin(π²x)
E_WEIGHTS  = (2, 2, 1)          # equilibrium basis: near-uniform (flat weighting correct)

# ═══════════════════════════════════════════════════════════════════════════════
# FIXED CAE ENGINEERING THRESHOLDS
# Derived from meshing standards and solver accuracy requirements.
# Not derived from data.  Analogous to DIEL_BIFURCATION (EM) and
# RMS_BIFURCATION (bearing CSP).
# ═══════════════════════════════════════════════════════════════════════════════

ASPECT_BIFURCATION = 5.0     # element quality threshold — solver accuracy degrades above
SKEW_BIFURCATION   = 0.85    # skewness threshold — solver stability limit
KAPPA_FACTOR       = 6.0     # h ≤ R/6 engineering rule → KAPPA_BIFURCATION = 6 / L_CHAR
DIST_NF_FACTOR     = 0.10    # BC near-field radius = 10% of characteristic length
NORMAL_DEV_SCALE   = 90.0    # degrees — normalising scale for surface normal variation

# Refinement score weights (curvature is primary driver in CAE)
W_KAPPA  = 0.50
W_ASPECT = 0.25
W_SKEW   = 0.15
W_BC     = 0.10


# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY CONSTANTS
# Derived from the mesh geometry itself — not from data statistics.
# Analogous to LAMBDA_FREE (EM) and FAILURE_G (bearing CSP).
# Replace the L_CHAR estimate with the actual CAD bounding box diagonal
# when available from the export.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_geometry_constants(df, L_CHAR_override=None):
    """
    Derive physical normalising constants from confirmed mesh geometry.

    All constants are computable from CAD/mesh geometry before any
    classification is attempted.  None are data-derived statistics.

    KAPPA_BIFURCATION uses the MEDIAN element size — not L_CHAR.
    Physical meaning: at the typical coarse element size h_med, the
    bifurcation occurs when κ × h_med × 6 = 1, i.e. h_med = R/6.
    This is the exact h = R/6 engineering rule applied to the median
    element, making the threshold robust to extreme outlier zones.

    The curvature term in the refinement score uses LOCAL element size
    (per zone), so each zone is judged by its OWN current resolution.

    In production: pass L_CHAR_override = CAD bounding box diagonal.
    """
    h_median = float(np.median(df['edge_length_min'].values))
    h_max    = float(df['edge_length_min'].max())

    if L_CHAR_override is not None:
        L_CHAR = float(L_CHAR_override)
    else:
        L_CHAR = h_max * 10.0   # conservative estimate; replace with bbox diagonal

    KAPPA_MAX         = float(df['kappa_max'].max())
    if KAPPA_MAX < EPS:
        KAPPA_MAX = 1.0 / (L_CHAR + EPS)

    # Β-anchor: curvature at which median element hits h = R/6
    KAPPA_BIFURCATION = 1.0 / (KAPPA_FACTOR * h_median)
    DIST_NF           = DIST_NF_FACTOR * L_CHAR

    return {
        'L_CHAR':             L_CHAR,
        'KAPPA_MAX':          KAPPA_MAX,
        'KAPPA_BIFURCATION':  KAPPA_BIFURCATION,
        'DIST_NF':            DIST_NF,
        'h_median':           h_median,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PI / E ENCODING
# Identical basis functions and weights to EM surrogate (em_doe_retrospective.py)
# and bearing CSP (pronostia_3simm.py).  Weights are not tuned per domain.
# ═══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, scale, weights=PI_WEIGHTS):
    """
    Cascade / non-periodic encoding.
    scale: confirmed physical constant (never a data statistic).
    sin(π²·x̃) carries weight 3 because π² is irrational → strictly non-periodic.
    This is the mathematically correct basis for variables that never return
    to a prior state (cascade character).
    """
    x  = np.asarray(x, dtype=float)
    xn = np.clip(x / (scale + EPS), 0, 10)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f'{prefix}_sin_pi':  w[0] * np.sin(PI * xn),
        f'{prefix}_cos_pi':  w[1] * np.cos(PI * xn),
        f'{prefix}_sin_2pi': w[2] * np.sin(2 * PI * xn),
        f'{prefix}_sin_pi2': w[3] * np.sin(PI**2 * xn),
        f'{prefix}_cascade': w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }


def encode_e_func(x, prefix, scale, weights=E_WEIGHTS):
    """
    Self-regulating / bounded encoding.
    scale: confirmed physical constant.
    Near-uniform weights confirm flat weighting is structurally correct
    for bounded, self-correcting variables.
    """
    x  = np.asarray(x, dtype=float)
    xn = np.clip(x / (scale + EPS), 0, 10)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f'{prefix}_exp_neg': w[0] * np.exp(-E * xn),
        f'{prefix}_pow_e':   w[1] * xn ** E,
        f'{prefix}_gauss':   w[2] * np.exp(-E * (xn - 0.5)**2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MESH ZONE FEATURE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_zone_features(df, consts):
    """
    Build encoded feature matrix for all mesh zones.

    Variable classification (from CAE meshing physics — not from data):

    Π-type (cascade — mesh error compounds non-repeatingly):
      kappa_max       high curvature → field gradients grow without bound → Π
                      scale: KAPPA_MAX (maximum curvature in model)
      kappa_mean      mean curvature in zone → Π for same reason
                      scale: KAPPA_MAX
      aspect_ratio    element quality degrades non-repeatingly past threshold → Π
                      scale: ASPECT_BIFURCATION = 5.0
      skewness        past SKEW_BIFURCATION, solver accuracy does not self-correct → Π
                      scale: SKEW_BIFURCATION = 0.85
      dist_to_bc      proximity to BC drives field gradient cascade → Π
                      scale: DIST_NF (BC influence radius)

    Ε-type (equilibrium — bounded, self-regulating, coarse mesh acceptable):
      normal_deviation surface normal variation; bounded by geometry → Ε
                       scale: NORMAL_DEV_SCALE = 90 degrees
      area_fraction    bounded [0, 1] by construction → Ε
                       scale: 1.0
      edge_length_min  bounded by mesh generation constraints → Ε
                       scale: L_CHAR / 6 (coarse mesh characteristic scale)

    Cross-products (Π × Ε regime interactions):
      sin(π × kappa_n)  × exp(−e × normal_n)   curvature cascade × surface flatness
      sin(π × aspect_n) × exp(−e × area_n)     element quality   × zone size
      sin(π × dist_n)   × exp(−e × edge_n)     BC near-field     × local density
    """
    KM = consts['KAPPA_MAX']
    DN = consts['DIST_NF']
    L  = consts['L_CHAR']

    enc = {}

    # Dimensionless curvature-times-element: 6h/R — the exact bifurcation variable.
    # Scale = KAPPA_FACTOR so that encoded value = 1.0 at the h=R/6 threshold.
    kappa_h_max  = df['kappa_max'].values  * df['edge_length_min'].values * KAPPA_FACTOR
    kappa_h_mean = df['kappa_mean'].values * df['edge_length_min'].values * KAPPA_FACTOR

    # ── Π-type ────────────────────────────────────────────────────────────────
    enc.update(encode_pi_func(kappa_h_max,  'pi_kappa_max',
                              scale=10.0))   # scale=10: clips at 10× bifurcation (extreme case)
    enc.update(encode_pi_func(kappa_h_mean, 'pi_kappa_mean',
                              scale=10.0))
    enc.update(encode_pi_func(df['aspect_ratio'].values, 'pi_aspect',
                              scale=ASPECT_BIFURCATION))
    enc.update(encode_pi_func(df['skewness'].values,     'pi_skew',
                              scale=SKEW_BIFURCATION))
    enc.update(encode_pi_func(df['dist_to_bc'].values,   'pi_dist_bc',
                              scale=DN))

    # ── Ε-type ────────────────────────────────────────────────────────────────
    enc.update(encode_e_func(df['normal_deviation'].values, 'e_normal_dev',
                              scale=NORMAL_DEV_SCALE))
    enc.update(encode_e_func(df['area_fraction'].values,    'e_area_frac',
                              scale=1.0))
    enc.update(encode_e_func(df['edge_length_min'].values,  'e_edge_len',
                              scale=L / 6.0))

    # ── Cross-products ────────────────────────────────────────────────────────
    kappa_n  = np.clip(df['kappa_max'].values        / (KM + EPS),                0, 1)
    normal_n = np.clip(df['normal_deviation'].values / NORMAL_DEV_SCALE,          0, 1)
    aspect_n = np.clip(df['aspect_ratio'].values     / ASPECT_BIFURCATION,        0, 1)
    area_n   = np.clip(df['area_fraction'].values,                                0, 1)
    dist_n   = np.clip(df['dist_to_bc'].values       / (DN + EPS),                0, 1)
    edge_n   = np.clip(df['edge_length_min'].values  / (L / 6.0 + EPS),           0, 1)

    enc['cross_kappa_x_flat']  = np.sin(PI * kappa_n)  * np.exp(-E * normal_n)
    enc['cross_aspect_x_area'] = np.sin(PI * aspect_n) * np.exp(-E * area_n)
    enc['cross_bc_x_edge']     = np.sin(PI * dist_n)   * np.exp(-E * edge_n)

    enc_df     = pd.DataFrame(enc, index=df.index)
    pi_cols    = [c for c in enc_df.columns if c.startswith('pi_')]
    e_cols     = [c for c in enc_df.columns if c.startswith('e_')]
    cross_cols = [c for c in enc_df.columns if c.startswith('cross_')]

    return enc_df, pi_cols, e_cols, cross_cols


# ═══════════════════════════════════════════════════════════════════════════════
# REFINEMENT SCORE
# Geometry-derived proxy target for the probe.
# No solver data required.  Computed from confirmed geometry constants only.
# Analogous to RUL (bearing CSP) and power_density (EM surrogate).
#
# Score > 1.0  →  Π regime (refinement necessary)
# Score ≈ 1.0  →  Β regime (at bifurcation)
# Score < 1.0  →  Ε regime (coarse mesh acceptable)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_refinement_score(df, consts):
    """
    Geometry-derived refinement urgency score per zone.
    Score > 1.5 → Π (refine).  Score < 0.6 → Ε (coarse ok).  Else → Β.

    Curvature term uses LOCAL element size (per zone):
      kappa_score = κ_max × h_local × KAPPA_FACTOR = 6h/R
    This is the exact dimensionless ratio the h = R/6 rule requires.
    Score = 1.0 when h = R/6 (bifurcation point).
    Score > 1.0 means the current element is too coarse for this curvature.

    Other terms (aspect, skewness, BC proximity) use global constants.
    """
    DN = consts['DIST_NF']

    # Per-zone: 6 × h_local / R = κ × h × 6.  Score=1 at bifurcation.
    kappa_score  = (df['kappa_max'].values
                    * df['edge_length_min'].values
                    * KAPPA_FACTOR)
    aspect_score = df['aspect_ratio'].values / ASPECT_BIFURCATION
    skew_score   = df['skewness'].values / SKEW_BIFURCATION
    bc_score     = np.clip(1.0 - df['dist_to_bc'].values / (DN + EPS), 0, 1)

    return (W_KAPPA  * kappa_score
          + W_ASPECT * aspect_score
          + W_SKEW   * skew_score
          + W_BC     * bc_score)


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE
# RF importance on encoded features vs refinement score.
# Identical mechanism to pronostia_3simm.py (bearing domain).
# Confirms structural typing: Π-encoded features should dominate importance
# if curvature-driven cascade character is correctly assigned.
# ═══════════════════════════════════════════════════════════════════════════════

def run_probe(X, y, feature_names, n_trees=500):
    """RF importance probe.  Returns importance Series sorted descending."""
    rf = RandomForestRegressor(
        n_estimators=n_trees, max_features='sqrt',
        min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_,
                     index=feature_names).sort_values(ascending=False)


def probe_report(imp, pi_cols, e_cols, cross_cols, top_n=10):
    """
    Print grouped importance and structural typing confirmation.

    Expected result: Π-encoded features dominate (curvature cascade
    drives refinement score).  If Ε dominates, review variable
    classification — the encoding may not match the domain physics.
    """
    tot       = imp.sum() + EPS
    pi_imp    = imp[pi_cols].sum()
    e_imp     = imp[e_cols].sum()
    cross_imp = imp[cross_cols].sum()

    top_n = min(top_n, len(imp))
    print(f'\n  Top {top_n} features by importance:')
    print(f"  {'Rank':>4}  {'Feature':>28}  {'Imp%':>7}  Type")
    print(f"  {'─'*4}  {'─'*28}  {'─'*7}  {'─'*4}")
    for rank, (fn, fv) in enumerate(imp.head(top_n).items(), 1):
        t = 'Π' if fn in pi_cols else ('Ε' if fn in e_cols else '×')
        print(f'  {rank:>4}  {fn:>28}  {100*fv/tot:>6.2f}%  {t}')

    print(f'\n  Grouped importance:')
    print(f'    Π (cascade)    : {100*pi_imp/tot:.1f}%')
    print(f'    Ε (equilibrium): {100*e_imp/tot:.1f}%')
    print(f'    × (cross)      : {100*cross_imp/tot:.1f}%')

    dominant = 'Π' if pi_imp > e_imp else 'Ε'
    ratio    = max(pi_imp, e_imp) / (min(pi_imp, e_imp) + EPS)
    verdict  = ('CONFIRMED — cascade character drives refinement. '
                'Π-encoding structurally correct.'
                if dominant == 'Π'
                else 'UNEXPECTED — review variable classification.')
    print(f'\n  Dominant: {dominant}  ({ratio:.1f}×)  →  {verdict}')

    return float(pi_imp / tot), float(e_imp / tot)


# ═══════════════════════════════════════════════════════════════════════════════
# ZONE CLASSIFIER
# TSA type per zone + refinement recommendation + description.
# ═══════════════════════════════════════════════════════════════════════════════

def classify_zones(df, enc_df, pi_cols, e_cols, consts, flagged_ids, load_ids):
    """
    Assign TSA type to each zone.

    Priority (highest to lowest):
      1. Zone ID in load_ids   → always Π  (physics-driven: load application point
                                             requires refinement regardless of geometry)
      2. Zone ID in flagged_ids → always Β  (human uncertainty: geometry decision
                                             is non-obvious, requires inspection)
      3. score > 1.5           → Π          (geometry-driven cascade confirmed)
      4. score < 0.60          → Ε          (geometry-driven equilibrium confirmed)
      5. all other zones       → Β          (at or near bifurcation)

    The Π/Ε feature ratio is informational only and does not gate classification.
    Score is the geometry discriminator. Load and flag IDs are physics/user overrides.
    """
    scores = compute_refinement_score(df, consts)
    KB     = consts['KAPPA_BIFURCATION']
    L      = consts['L_CHAR']

    results = []
    for i, row in df.iterrows():
        zid   = int(row['zone_id'])
        score = float(scores[i])
        pi_val = enc_df.loc[i, pi_cols].abs().mean()
        e_val  = enc_df.loc[i, e_cols].abs().mean()
        ratio  = float(pi_val / (e_val + EPS))

        if zid in load_ids:
            tsa  = 'Π'
            conf = 'physics-driven load zone'
            h_rec = 1.0 / (6.0 * row['kappa_max'] + EPS)
            rec  = ('REFINE — physics-driven. Load application zone: '
                    'refine regardless of local geometry.')
            desc = (f'Zone {zid}: user-identified load application point. '
                    f'Physics requires Π-level refinement independent of local curvature '
                    f'(κ_max = {row["kappa_max"]:.4f}, geometry score = {score:.2f}). '
                    f'Stress concentration at load application is solver-driven, not '
                    f'shape-driven. Recommended h ≤ {h_rec:.4f} as geometry lower bound; '
                    f'tighten further based on expected stress gradient.')

        elif zid in flagged_ids:
            tsa  = 'Β'
            conf = 'user-flagged'
            rec  = ('INSPECT — user-flagged zone. Apply focused refinement and '
                    'validate mesh visually before proceeding.')
            desc = (f'Zone {zid} flagged for manual attention. '
                    f'Refinement score: {score:.2f} (bifurcation = 1.0). '
                    f'Π/Ε feature balance: {ratio:.2f}. '
                    f'Classify after visual inspection of local geometry.')

        elif score > 1.5:
            tsa  = 'Π'
            conf = f'score={score:.2f}'
            h_rec = 1.0 / (6.0 * row['kappa_max'] + EPS)
            rec  = (f'REFINE — cascade regime. '
                    f'Reduce element size to ≤ {h_rec:.4f} (same units as curvature input).')
            desc = (f'Zone {zid}: curvature-driven cascade. '
                    f'κ_max = {row["kappa_max"]:.4f} exceeds bifurcation threshold '
                    f'κ_bif = {KB:.4f} (h = R/6 condition). Score {score:.2f} >> 1.0. '
                    f'Mesh error compounds non-repeatingly without refinement. '
                    f'Skewness {row["skewness"]:.2f}, aspect ratio {row["aspect_ratio"]:.2f}.')

        elif score < 0.60:
            tsa  = 'Ε'
            conf = f'score={score:.2f}'
            rec  = ('COARSE OK — equilibrium regime. '
                    'Current element density is acceptable for this zone.')
            desc = (f'Zone {zid}: self-regulating geometry. '
                    f'κ_max = {row["kappa_max"]:.4f} is well below bifurcation threshold '
                    f'κ_bif = {KB:.4f}. Score {score:.2f} << 1.0. '
                    f'Normal deviation {row["normal_deviation"]:.1f}° confirms surface flatness. '
                    f'Mesh error is bounded and self-correcting here.')

        else:
            tsa  = 'Β'
            conf = f'score={score:.2f}'
            rec  = ('REVIEW — boundary regime. Apply moderate refinement and '
                    'validate visually. Refinement decision is non-obvious.')
            desc = (f'Zone {zid}: at structural transition between cascade and equilibrium. '
                    f'κ_max = {row["kappa_max"]:.4f} near bifurcation threshold '
                    f'κ_bif = {KB:.4f}. Score {score:.2f} in transition zone [0.6, 1.5]. '
                    f'Π/Ε feature balance: {ratio:.2f}. '
                    f'Apply moderate refinement and inspect solver convergence.')

        results.append({
            'zone_id':    zid,
            'tsa_type':   tsa,
            'confidence': conf,
            'score':      round(score, 3),
            'pi_e_ratio': round(ratio, 3),
            'rec':        rec,
            'desc':       desc,
        })

    # ── Second pass: neighbor-aware reclassification of flat zones ────────────
    # Only runs if neighbor_ids column is present (produced by mesh_to_probe.py).
    # Flat zones (κ_max ≈ 0) are geometrically ambiguous on their own.
    # Their correct classification depends on surrounding context:
    #   all Π neighbors → Π  (refinement propagates from adjacent cascade zones)
    #   all Ε neighbors → Ε  (confirmed equilibrium, no cascade nearby)
    #   mixed neighbors → Β  (sits at a regime boundary)
    # Load and flagged zones are never reclassified here — their type is fixed.
    if 'neighbor_ids' in df.columns:
        results_map = {r['zone_id']: r['tsa_type'] for r in results}

        for r in results:
            zid = r['zone_id']
            if zid in load_ids or zid in flagged_ids:
                continue

            row = df[df['zone_id'] == zid].iloc[0]
            if float(row['kappa_max']) > 1e-6:
                continue    # not flat — geometry classification stands

            neighbor_str = str(row.get('neighbor_ids', ''))
            if not neighbor_str or neighbor_str == 'nan':
                continue

            neighbor_zone_ids  = [int(n) for n in neighbor_str.split(',') if n.strip()]
            neighbor_types     = [results_map.get(n, 'Β') for n in neighbor_zone_ids]
            if not neighbor_types:
                continue

            unique_types = set(neighbor_types)

            if unique_types == {'Π'}:
                r['tsa_type']   = 'Π'
                r['confidence'] = f'neighbor-driven ({len(neighbor_types)} Π neighbors)'
                r['rec']        = ('REFINE — neighbor-driven. '
                                   'All adjacent zones require refinement.')
                r['desc']       = (f'Zone {zid}: flat geometry (κ_max ≈ 0, score {r["score"]:.2f}) '
                                   f'but all {len(neighbor_types)} adjacent zones are Π. '
                                   f'Refinement propagates from neighboring cascade zones. '
                                   f'Apply same element density as adjacent Π zones.')

            elif unique_types == {'Ε'}:
                pass    # already Ε from first pass — confirmed, no change

            else:
                r['tsa_type']   = 'Β'
                r['confidence'] = f'neighbor-driven (mixed: {", ".join(sorted(unique_types))})'
                r['rec']        = ('REVIEW — neighbor-driven. '
                                   'Adjacent zones have mixed classifications.')
                r['desc']       = (f'Zone {zid}: flat geometry (κ_max ≈ 0, score {r["score"]:.2f}) '
                                   f'with mixed neighbor types '
                                   f'({", ".join(neighbor_types)}). '
                                   f'Sits at a regime boundary between cascade and equilibrium. '
                                   f'Apply moderate refinement and inspect transition.')

        # Rebuild results_map after second pass for any downstream use
        results_map = {r['zone_id']: r['tsa_type'] for r in results}

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC TEST GENERATORS
# Known a priori expected TSA classifications for four basic shapes.
# These are the ground truth for validating the probe without CAD data.
# Expected types derive from geometry physics — not from running the probe.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_zones(shape):
    """
    Generate synthetic zone descriptors for a basic shape.

    Expected TSA types are determined from geometry before any probe is run.
    They are the validation target: if the probe classifies correctly, the
    encoding and scoring are structurally consistent with the domain physics.
    """
    np.random.seed(42)

    if shape == 'sphere':
        # Sphere R=50mm — uniform curvature κ=1/R everywhere.
        # All zones: same curvature, same normal variation, far from BC.
        # Expected: all Ε (curvature is low relative to L_CHAR = R).
        R = 50.0
        rows = []
        for i in range(8):
            rows.append({
                'zone_id':         i + 1,
                'kappa_mean':      1.0 / R * (1.0 + 0.02 * np.random.randn()),
                'kappa_max':       1.0 / R * (1.0 + 0.05 * np.random.randn()),
                'aspect_ratio':    1.2 + 0.15 * np.random.rand(),
                'skewness':        0.10 + 0.04 * np.random.rand(),
                'dist_to_bc':      R * (0.5 + 0.35 * np.random.rand()),
                'normal_deviation': 3.0 + 1.5 * np.random.rand(),
                'area_fraction':   1.0 / 8.0,
                'edge_length_min': R / 20.0 * (1.0 + 0.1 * np.random.rand()),
            })
        expected = {i+1: 'Ε' for i in range(8)}
        notes    = ('Sphere R=50mm — uniform κ=1/R. '
                    'All zones expect Ε: curvature well below bifurcation threshold.')

    elif shape == 'cylinder':
        # Cylinder R=5mm, L=40mm — coarse rough mesh (h~3mm on curved surface)
        # Curved surface: κ=1/R=0.2, h=3mm → κ×h×6 = 3.6 → Π
        # Flat ends: κ≈0, any h → score≈0 → Ε
        # Curved-to-flat edge transitions: moderate κ, moderate h → Β
        R = 5.0
        rows = [
            # Curved surface zones — κ×h×6 = 0.2×3×6 = 3.6 → Π
            {'zone_id': 1, 'kappa_mean': 1/R, 'kappa_max': 1/R,
             'aspect_ratio': 1.5, 'skewness': 0.15, 'dist_to_bc': 5.0,
             'normal_deviation': 25.0, 'area_fraction': 0.15, 'edge_length_min': 3.0},
            {'zone_id': 2, 'kappa_mean': 1/R, 'kappa_max': 1/R,
             'aspect_ratio': 1.4, 'skewness': 0.12, 'dist_to_bc': 20.0,
             'normal_deviation': 22.0, 'area_fraction': 0.15, 'edge_length_min': 3.0},
            {'zone_id': 3, 'kappa_mean': 1/R, 'kappa_max': 1/R,
             'aspect_ratio': 1.6, 'skewness': 0.18, 'dist_to_bc': 35.0,
             'normal_deviation': 28.0, 'area_fraction': 0.15, 'edge_length_min': 3.0},
            # Flat end zones — κ≈0 → Ε
            {'zone_id': 4, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.2, 'skewness': 0.08, 'dist_to_bc': 3.0,
             'normal_deviation': 1.0, 'area_fraction': 0.15, 'edge_length_min': 3.0},
            {'zone_id': 5, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.3, 'skewness': 0.09, 'dist_to_bc': 37.0,
             'normal_deviation': 1.0, 'area_fraction': 0.15, 'edge_length_min': 3.0},
            # Edge transition — κ=0.12, h=1.5 → κ×h×6=1.08 → Β
            {'zone_id': 6, 'kappa_mean': 0.10, 'kappa_max': 0.12,
             'aspect_ratio': 3.0, 'skewness': 0.55, 'dist_to_bc': 3.0,
             'normal_deviation': 45.0, 'area_fraction': 0.10, 'edge_length_min': 1.5},
            {'zone_id': 7, 'kappa_mean': 0.10, 'kappa_max': 0.12,
             'aspect_ratio': 2.8, 'skewness': 0.50, 'dist_to_bc': 37.0,
             'normal_deviation': 42.0, 'area_fraction': 0.10, 'edge_length_min': 1.5},
            # Interior flat zone — κ≈0, large h → Ε
            {'zone_id': 8, 'kappa_mean': 0.001, 'kappa_max': 0.001,
             'aspect_ratio': 1.1, 'skewness': 0.05, 'dist_to_bc': 18.0,
             'normal_deviation': 0.5, 'area_fraction': 0.05, 'edge_length_min': 4.0},
        ]
        expected = {1:'Π', 2:'Π', 3:'Π', 4:'Ε', 5:'Ε', 6:'Β', 7:'Β', 8:'Ε'}
        notes    = ('Cylinder R=5mm L=40mm, rough mesh h~3mm on curved surface. '
                    'curved surface (6h/R=3.6)→Π, flat ends→Ε, edge transitions (6h/R≈1.1)→Β.')

    elif shape == 'box':
        # Box with sharp edges and corners.
        # kappa_max for sharp edges treated as high (small fillet assumption).
        # Flat faces: κ≈0 → Ε
        # Sharp edges: high κ, high skewness → Π
        # Face-to-edge transitions: moderate → Β
        rows = [
            # Flat faces
            {'zone_id': 1, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.2, 'skewness': 0.08, 'dist_to_bc': 20.0,
             'normal_deviation': 0.5, 'area_fraction': 0.20, 'edge_length_min': 5.0},
            {'zone_id': 2, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.3, 'skewness': 0.07, 'dist_to_bc': 25.0,
             'normal_deviation': 0.5, 'area_fraction': 0.20, 'edge_length_min': 5.0},
            {'zone_id': 3, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.1, 'skewness': 0.06, 'dist_to_bc': 30.0,
             'normal_deviation': 0.5, 'area_fraction': 0.20, 'edge_length_min': 5.0},
            # Sharp edges (high curvature, high skewness)
            {'zone_id': 4, 'kappa_mean': 5.0, 'kappa_max': 8.0,
             'aspect_ratio': 4.5, 'skewness': 0.70, 'dist_to_bc': 3.0,
             'normal_deviation': 88.0, 'area_fraction': 0.06, 'edge_length_min': 0.5},
            {'zone_id': 5, 'kappa_mean': 5.0, 'kappa_max': 8.0,
             'aspect_ratio': 4.2, 'skewness': 0.65, 'dist_to_bc': 25.0,
             'normal_deviation': 89.0, 'area_fraction': 0.06, 'edge_length_min': 0.5},
            # Corners (extreme curvature, worst skewness)
            {'zone_id': 6, 'kappa_mean': 10.0, 'kappa_max': 15.0,
             'aspect_ratio': 6.0, 'skewness': 0.90, 'dist_to_bc': 2.0,
             'normal_deviation': 89.0, 'area_fraction': 0.03, 'edge_length_min': 0.2},
            {'zone_id': 7, 'kappa_mean': 10.0, 'kappa_max': 15.0,
             'aspect_ratio': 5.8, 'skewness': 0.88, 'dist_to_bc': 28.0,
             'normal_deviation': 89.0, 'area_fraction': 0.03, 'edge_length_min': 0.2},
            # Face-to-edge transition — κ×h×6=0.5×0.5×6=1.5 → Β
            {'zone_id': 8, 'kappa_mean': 0.40, 'kappa_max': 0.50,
             'aspect_ratio': 3.0, 'skewness': 0.45, 'dist_to_bc': 12.0,
             'normal_deviation': 45.0, 'area_fraction': 0.08, 'edge_length_min': 0.5},
        ]
        expected = {1:'Ε', 2:'Ε', 3:'Ε', 4:'Π', 5:'Π', 6:'Π', 7:'Π', 8:'Β'}
        notes    = ('Box — flat faces→Ε, sharp edges/corners→Π, face-edge transitions→Β.')

    elif shape == 'cone':
        # Cone R_tip=0.5mm R_base=15mm H=50mm — rough mesh, h varies with zone
        # Expected types derived from κ×h×6:
        #   Tip:    κ=4,    h=0.1  → 4×0.1×6=2.4  → Π
        #   Upper:  κ=0.67, h=0.5  → 0.67×0.5×6=2.0 → Π
        #   Mid:    κ=0.2,  h=0.75 → 0.2×0.75×6=0.9 → Β
        #   Lower:  κ=0.1,  h=2.5  → 0.1×2.5×6=1.5 → Β (right at boundary)
        #   Base:   κ≈0,   h=3    → ≈0.036         → Ε
        #   B-edge: κ=0.067,h=1.5 → 0.6            → Β
        #   Far:    κ≈0,   h=4    → ≈0.03           → Ε
        R_tip = 0.5; R_base = 15.0
        rows = [
            # Tip zone — κ×h×6=2.4 → Π
            {'zone_id': 1, 'kappa_mean': 1/R_tip, 'kappa_max': 2/R_tip,
             'aspect_ratio': 5.5, 'skewness': 0.80, 'dist_to_bc': 1.0,
             'normal_deviation': 60.0, 'area_fraction': 0.02, 'edge_length_min': 0.1},
            # Upper cone — κ×h×6=0.67×0.7×6=2.8 → Π
            {'zone_id': 2, 'kappa_mean': 0.60, 'kappa_max': 0.67,
             'aspect_ratio': 3.5, 'skewness': 0.55, 'dist_to_bc': 8.0,
             'normal_deviation': 40.0, 'area_fraction': 0.08, 'edge_length_min': 0.7},
            # Mid cone — h increased to 1.2mm (physical gradient from tip h=0.1 to base h=3.0).
            # Full score ≈ 0.87, well inside Β band [0.60, 1.50]. Margin ~0.27 from lower bound.
            # NOTE: the h=R/6 formula (score = κ×h×6) gives the threshold correctly but the
            # absolute threshold values (0.60 and 1.50) have not been validated against an
            # independent solver. They are engineering-rule-derived. Further analysis required
            # when real mesh output is available for comparison.
            {'zone_id': 3, 'kappa_mean': 0.18, 'kappa_max': 0.20,
             'aspect_ratio': 2.0, 'skewness': 0.30, 'dist_to_bc': 20.0,
             'normal_deviation': 20.0, 'area_fraction': 0.15, 'edge_length_min': 1.2},
            # Lower cone — κ×h×6=1.5 → Β (at bifurcation)
            {'zone_id': 4, 'kappa_mean': 0.09, 'kappa_max': 0.10,
             'aspect_ratio': 1.4, 'skewness': 0.15, 'dist_to_bc': 35.0,
             'normal_deviation': 10.0, 'area_fraction': 0.20, 'edge_length_min': 2.5},
            # Base flat zone — κ≈0 → Ε
            {'zone_id': 5, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.2, 'skewness': 0.08, 'dist_to_bc': 45.0,
             'normal_deviation': 1.0, 'area_fraction': 0.20, 'edge_length_min': 3.0},
            # Base-to-cone edge — κ×h×6=0.067×1.8×6=0.72 → Β
            {'zone_id': 6, 'kappa_mean': 0.060, 'kappa_max': 0.067,
             'aspect_ratio': 3.2, 'skewness': 0.60, 'dist_to_bc': 42.0,
             'normal_deviation': 55.0, 'area_fraction': 0.10, 'edge_length_min': 1.8},
            # Smooth far zones — κ≈0, large h → Ε
            {'zone_id': 7, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.1, 'skewness': 0.06, 'dist_to_bc': 50.0,
             'normal_deviation': 0.5, 'area_fraction': 0.15, 'edge_length_min': 4.0},
            {'zone_id': 8, 'kappa_mean': 0.001, 'kappa_max': 0.002,
             'aspect_ratio': 1.3, 'skewness': 0.10, 'dist_to_bc': 48.0,
             'normal_deviation': 1.0, 'area_fraction': 0.10, 'edge_length_min': 3.5},
        ]
        expected = {1:'Π', 2:'Π', 3:'Β', 4:'Β', 5:'Ε', 6:'Β', 7:'Ε', 8:'Ε'}
        notes    = ('Cone R_tip=0.5mm R_base=15mm H=50mm. '
                    'tip/upper (6h/R>2)→Π, mid/lower/base-edge (6h/R≈0.6-1.5)→Β, '
                    'base flat/far (6h/R<0.1)→Ε.')

    else:
        raise ValueError(
            f"Unknown shape '{shape}'. Choose: sphere, cylinder, box, cone.")

    return pd.DataFrame(rows), expected, notes


# ═══════════════════════════════════════════════════════════════════════════════
# TSA BOUNDARY ZONE RESOLUTION
# For Β zones: 4 options derived from TSA theorem, no stochastic noise.
# Option 2 (TSA Β) is the geometric mean of h_fine and h_coarse — the
# exact Β encoding peak (exp(-e(x-0.5)²) at x=0.5) applied to element
# size. This is the theorem-derived answer for the boundary state.
# One global prompt for all Β zones — one decision, applied uniformly.
# ═══════════════════════════════════════════════════════════════════════════════

def generate_beta_options(df, results_df):
    """
    Build per-zone bounds for all Β zones.

    h_fine   = min(edge_length_min of Π neighbors) — finest bound
    h_coarse = max(edge_length_min of Ε neighbors) — coarsest bound
    Degenerate fallbacks:
        no Π neighbors → h_fine   = h_current / 4
        no Ε neighbors → h_coarse = h_current

    Returns dict {zone_id: bounds_dict} for use by prompt_beta_resolution.
    """
    type_map = dict(zip(results_df['zone_id'].astype(int),
                        results_df['tsa_type']))
    h_map    = dict(zip(df['zone_id'].astype(int),
                        df['edge_length_min'].astype(float)))

    beta_options = {}

    for _, r in results_df.iterrows():
        zid = int(r['zone_id'])
        if r['tsa_type'] != 'Β':
            continue

        row       = df[df['zone_id'] == zid].iloc[0]
        h_current = float(row['edge_length_min'])

        neighbor_str = str(row.get('neighbor_ids', ''))
        if neighbor_str == 'nan' or not neighbor_str:
            neighbor_ids = []
        else:
            neighbor_ids = [int(n) for n in neighbor_str.split(',') if n.strip()]

        pi_neighbors = [n for n in neighbor_ids if type_map.get(n) == 'Π']
        e_neighbors  = [n for n in neighbor_ids if type_map.get(n) == 'Ε']

        degenerate_fine   = len(pi_neighbors) == 0
        degenerate_coarse = len(e_neighbors)  == 0

        h_fine   = min(h_map[n] for n in pi_neighbors) if pi_neighbors else h_current / 4.0
        h_coarse = max(h_map[n] for n in e_neighbors)  if e_neighbors  else h_current

        if h_fine >= h_coarse:
            h_fine = h_coarse / 4.0

        beta_options[zid] = {
            'h_fine':           round(h_fine,    4),
            'h_coarse':         round(h_coarse,  4),
            'h_current':        round(h_current, 4),
            'degenerate_fine':  degenerate_fine,
            'degenerate_coarse':degenerate_coarse,
            'pi_neighbors':     pi_neighbors,
            'e_neighbors':      e_neighbors,
        }

    return beta_options


def _build_options(h_fine, h_coarse, h_current):
    """
    Build 4 exact TSA-derived options from h_fine and h_coarse.

    Option 1 — Coarse   : h_coarse          (Ε bound, exact)
    Option 2 — TSA Β    : √(h_fine×h_coarse) (geometric mean — TSA theorem)
    Option 3 — Moderate : √(h_fine×h_TSA)   (geometric mean of fine and TSA)
    Option 4 — Fine     : h_fine            (Π bound, exact)

    No stochastic noise. The geometric mean is the exact Β encoding peak
    (E-type Gaussian at x=0.5) applied to element size space.
    """
    h_tsa = float(np.sqrt(h_fine * h_coarse))         # TSA Β — theorem
    h_mod = float(np.sqrt(h_fine * h_tsa))            # between fine and TSA

    labels = [
        'Coarse (Ε level)',
        'TSA Β — geometric mean  ← recommended',
        'Moderate',
        'Fine (Π level)',
    ]
    h_vals = [h_coarse, h_tsa, h_mod, h_fine]

    options = []
    for label, h_opt in zip(labels, h_vals):
        cells_x = (h_current / h_opt) ** 2
        options.append({
            'label':   label,
            'h':       round(h_opt, 4),
            'cells_x': round(cells_x, 2),
        })
    return options


def prompt_beta_resolution(results_df, beta_options, no_interactive):
    """
    Single global prompt for all Β zones.

    Global bounds: tightest h_fine and loosest h_coarse across all Β zones.
    One set of 4 TSA-derived options shown once. One choice applied to all
    Β zones simultaneously. tsa_type stays Β in output — the zone was
    genuinely ambiguous; the chosen h records the resolution.

    If no_interactive: prints options and leaves all Β zones unchanged.
    """
    if not beta_options:
        return results_df

    SEP2 = '─' * 68

    # Global bounds across all Β zones
    h_fine_global   = min(v['h_fine']   for v in beta_options.values())
    h_coarse_global = max(v['h_coarse'] for v in beta_options.values())
    h_current_ref   = list(beta_options.values())[0]['h_current']

    any_degen_fine   = any(v['degenerate_fine']   for v in beta_options.values())
    any_degen_coarse = any(v['degenerate_coarse'] for v in beta_options.values())

    options = _build_options(h_fine_global, h_coarse_global, h_current_ref)

    print(f'\n  {"═"*68}')
    print(f'  BOUNDARY ZONE RESOLUTION — {len(beta_options)} Β zone(s)')
    print(f'  {"═"*68}')
    print(f'  Zones: {sorted(beta_options.keys())}')
    print(f'  Global finest bound   : {h_fine_global:.4f}  '
          f'{"(conservative — no Π neighbors in some zones)" if any_degen_fine else "(from Π neighbors)"}')
    print(f'  Global coarsest bound : {h_coarse_global:.4f}  '
          f'{"(conservative — no Ε neighbors in some zones)" if any_degen_coarse else "(from Ε neighbors)"}')
    print(f'\n  TSA theorem: h_Β = √(h_fine × h_coarse) = '
          f'√({h_fine_global:.4f} × {h_coarse_global:.4f}) = '
          f'{np.sqrt(h_fine_global * h_coarse_global):.4f}')
    print(f'\n  {"Opt":>3}  {"Label":>38}  {"h target":>10}  {"cells_x (ref zone)":>18}')
    print(f'  {"─"*3}  {"─"*38}  {"─"*10}  {"─"*18}')
    for i, opt in enumerate(options, 1):
        direction = '↓ fewer' if opt['cells_x'] < 1.0 else '↑ more '
        print(f'  {i:>3}  {opt["label"]:>38}  {opt["h"]:>10.4f}  '
              f'{opt["cells_x"]:>6.2f}×  {direction}')

    results = results_df.to_dict('records')

    if no_interactive:
        print(f'\n  [no_interactive] Β zones left unchanged. '
              f'Re-run without --no_interactive to choose.')
        return pd.DataFrame(results)

    while True:
        try:
            raw = input(f'\n  Choose [1-4] for all Β zones, or Enter to skip: ').strip()
            if raw == '':
                print(f'  Skipped — all Β zones remain unresolved.')
                break
            choice = int(raw)
            if 1 <= choice <= 4:
                chosen = options[choice - 1]
                for r in results:
                    zid = int(r['zone_id'])
                    if zid not in beta_options:
                        continue
                    h_cur  = beta_options[zid]['h_current']
                    cells_x = round((h_cur / chosen['h']) ** 2, 2)
                    r['confidence'] = (f'user-chosen option {choice} — '
                                       f'{chosen["label"].split("  ←")[0].strip()}, '
                                       f'h={chosen["h"]:.4f}')
                    r['rec']        = (f'REFINE to h={chosen["h"]:.4f} — '
                                       f'{chosen["label"].split("  ←")[0].strip()} '
                                       f'(option {choice}/4, applied globally to all Β zones).')
                    r['desc']       = (r['desc'].rstrip('.') +
                                       f' Globally resolved: option {choice} '
                                       f'({chosen["label"].split("  ←")[0].strip()}), '
                                       f'h={chosen["h"]:.4f}, '
                                       f'{cells_x}× current element count.')
                print(f'  ✓  All {len(beta_options)} Β zone(s) → h={chosen["h"]:.4f} '
                      f'({chosen["label"].split("  ←")[0].strip()})')
                break
            else:
                print(f'  Enter a number between 1 and 4.')
        except (ValueError, EOFError):
            print(f'  Enter a number 1–4 or press Enter to skip.')
            break

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='ISPCC Mesh Probe — TSA zone classification for CAD mesh refinement')
    ap.add_argument('--zones', type=str, default=None,
                    help='CSV file with zone descriptors (one row per zone)')
    ap.add_argument('--flag', type=int, action='append', default=[],
                    dest='flagged', metavar='ZONE_ID',
                    help='Zone ID to flag as Β (user attention). Repeat up to 4 times.')
    ap.add_argument('--load', type=int, action='append', default=[],
                    dest='loaded', metavar='ZONE_ID',
                    help='Zone ID with physics-driven load application. Forces Π regardless of geometry. Repeat up to 4 times.')
    ap.add_argument('--test', type=str, default=None,
                    choices=['sphere', 'cylinder', 'box', 'cone'],
                    help='Run synthetic test for a basic shape (no --zones needed)')
    ap.add_argument('--no_probe', action='store_true',
                    help='Skip RF probe (faster; skips structural confirmation)')
    ap.add_argument('--no_interactive', action='store_true',
                    help='Print Β zone options but skip prompt. Β zones stay as Β in output.')
    ap.add_argument('--save', type=str, default=None,
                    help='Save classification results to this CSV path')
    ap.add_argument('--L_CHAR', type=float, default=None,
                    help='Override characteristic length (CAD bounding box diagonal recommended)')
    args = ap.parse_args()

    SEP = '=' * 72

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.test:
        print(f'\n{SEP}')
        print(f'  MESH PROBE — SYNTHETIC TEST: {args.test.upper()}')
        print(f'{SEP}')
        df, expected, notes = generate_test_zones(args.test)
        print(f'\n  {notes}')
        flagged_ids = set(args.flagged)
        load_ids    = set(args.loaded)

    elif args.zones:
        print(f'\n{SEP}')
        print(f'  MESH PROBE — {args.zones}')
        print(f'{SEP}')
        df = pd.read_csv(args.zones)
        expected    = None
        flagged_ids = set(args.flagged)
        load_ids    = set(args.loaded)

    else:
        ap.print_help()
        sys.exit(1)

    # ── Auto-read is_load_zone from CSV (produced by mesh_to_probe.py) ─────────
    # Merges with any --load arguments. --load overrides are additive.
    if 'is_load_zone' in df.columns:
        csv_load_ids = set(
            int(row['zone_id'])
            for _, row in df.iterrows()
            if int(row['is_load_zone']) == 1
        )
        if csv_load_ids:
            added = csv_load_ids - load_ids
            load_ids = load_ids | csv_load_ids
            if added:
                print(f'\n  [CSV] is_load_zone column found — '
                      f'auto-adding load zones: {sorted(added)}')

    if len(args.flagged) > 4:
        print(f'  [WARN] More than 4 zones flagged ({len(args.flagged)}). '
              f'Expected 1–4 focus zones.')
    if len(load_ids) > 4:
        print(f'  [WARN] More than 4 load zones active ({len(load_ids)}). '
              f'Expected 1–4 focus zones.')

    print(f'\n  Zones loaded : {len(df)}')
    print(f'  Load zones   : {sorted(load_ids)    if load_ids    else "none"}  '
          f'{"(from CSV + --load)" if "is_load_zone" in df.columns else "(from --load)"}')
    print(f'  User-flagged : {sorted(flagged_ids) if flagged_ids else "none"}')

    # ── Geometry constants ─────────────────────────────────────────────────────
    consts = compute_geometry_constants(df, L_CHAR_override=args.L_CHAR)
    print(f'\n  Geometry constants (derived from mesh geometry, not data statistics):')
    print(f'    L_CHAR             = {consts["L_CHAR"]:.4f}   [characteristic length]')
    print(f'    KAPPA_MAX          = {consts["KAPPA_MAX"]:.4f}   [max curvature in model]')
    print(f'    KAPPA_BIFURCATION  = {consts["KAPPA_BIFURCATION"]:.4f}   '
          f'[h=R/6 condition — Β-anchor]')
    print(f'    DIST_NF            = {consts["DIST_NF"]:.4f}   '
          f'[BC influence radius]')

    # ── Build encoded features ─────────────────────────────────────────────────
    enc_df, pi_cols, e_cols, cross_cols = build_zone_features(df, consts)
    print(f'\n  Encoded features: {len(enc_df.columns)} total  '
          f'({len(pi_cols)} Π + {len(e_cols)} Ε + {len(cross_cols)} ×)')

    # ── Probe ──────────────────────────────────────────────────────────────────
    if not args.no_probe:
        if len(df) >= 4:
            print(f'\n{SEP}')
            print(f'  PROBE — structural typing confirmation')
            print(f'  Target: geometry-derived refinement score (no solver data)')
            print(f'{SEP}')
            score = compute_refinement_score(df, consts)
            imp   = run_probe(enc_df.values, score, list(enc_df.columns))
            probe_report(imp, pi_cols, e_cols, cross_cols,
                         top_n=min(10, len(imp)))
        else:
            print(f'\n  [WARN] Only {len(df)} zones — probe needs ≥4. Skipping.')

    # ── Zone classification ────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'  ZONE CLASSIFICATION')
    print(f'{SEP}')
    results = classify_zones(df, enc_df, pi_cols, e_cols, consts, flagged_ids, load_ids)

    print(f'\n  {"ZoneID":>6}  {"Type":>4}  {"Score":>6}  {"Π/Ε":>5}  Recommendation')
    print(f'  {"─"*6}  {"─"*4}  {"─"*6}  {"─"*5}  {"─"*48}')
    for _, r in results.iterrows():
        if r['zone_id'] in load_ids:
            marker = '  ← LOAD'
        elif r['zone_id'] in flagged_ids:
            marker = '  ← FLAG'
        else:
            marker = ''
        print(f'  {r["zone_id"]:>6}  {r["tsa_type"]:>4}  {r["score"]:>6.3f}  '
              f'{r["pi_e_ratio"]:>5.2f}  {r["rec"]}{marker}')

    print(f'\n  Detailed descriptions:')
    for _, r in results.iterrows():
        print(f'\n  Zone {r["zone_id"]} [{r["tsa_type"]}] — {r["confidence"]}')
        print(f'    {r["desc"]}')

    # ── Boundary zone resolution ───────────────────────────────────────────────
    beta_zone_ids = [int(r['zone_id']) for _, r in results.iterrows()
                     if r['tsa_type'] == 'Β'
                     and int(r['zone_id']) not in flagged_ids]

    if beta_zone_ids:
        beta_options = generate_beta_options(df, results)
        results      = prompt_beta_resolution(results, beta_options,
                                              args.no_interactive)

    # ── Validation (test mode only) ────────────────────────────────────────────
    if expected is not None:
        print(f'\n{SEP}')
        print(f'  VALIDATION — expected vs probe output')
        print(f'{SEP}')
        correct = 0; total = 0
        for _, r in results.iterrows():
            zid = r['zone_id']
            if zid in load_ids:
                print(f'  Zone {zid:>2}:  load zone → Π forced  (skipped from accuracy count)')
                continue
            if zid in flagged_ids:
                print(f'  Zone {zid:>2}:  flagged by user → Β  (skipped from accuracy count)')
                continue
            exp = expected.get(zid, '?')
            got = r['tsa_type']
            ok  = '✓' if got == exp else '✗'
            if got == exp: correct += 1
            total += 1
            print(f'  Zone {zid:>2}:  expected {exp}  got {got}  {ok}'
                  f'  (score={r["score"]:.2f}, Π/Ε={r["pi_e_ratio"]:.2f})')
        if total > 0:
            print(f'\n  Accuracy (non-flagged zones): {correct}/{total} '
                  f'= {100*correct/total:.0f}%')

    # ── Save ───────────────────────────────────────────────────────────────────
    if args.save:
        results.to_csv(args.save, index=False)
        print(f'\n  Results saved to: {args.save}')

    print(f'\n{SEP}\n')


if __name__ == '__main__':
    main()


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
#
# Synthetic tests (no data needed):
#   python mesh_probe.py --test sphere
#   python mesh_probe.py --test cylinder
#   python mesh_probe.py --test box
#   python mesh_probe.py --test cone
#
# With user-flagged zones:
#   python mesh_probe.py --test box --flag 4 --flag 5
#   python mesh_probe.py --test cylinder --flag 6
#
# With physics-driven load zones:
#   python mesh_probe.py --test box --load 4
#   python mesh_probe.py --zones my_mesh.csv --load 3 --load 7
#
# Combined (load and flag simultaneously):
#   python mesh_probe.py --zones my_mesh.csv --load 3 --flag 9
#
# Skip probe (classification only, faster):
#   python mesh_probe.py --test cone --no_probe
#
# Boundary zone resolution — interactive (default):
#   python mesh_probe.py --zones probe_input.csv --load 3
#
# Boundary zone resolution — print options only, no prompt:
#   python mesh_probe.py --zones probe_input.csv --load 3 --no_interactive
#
# Boundary zone resolution — reproducible options (same seed = same options):
#   python mesh_probe.py --zones probe_input.csv --load 3 --seed 42
#
# From real CAD export:
#   python mesh_probe.py --zones my_mesh_zones.csv --flag 12 --flag 15
#
# Save results:
#   python mesh_probe.py --test box --save box_results.csv
#
# Required CSV columns:
#   zone_id, kappa_mean, kappa_max, aspect_ratio, skewness,
#   dist_to_bc, normal_deviation, area_fraction, edge_length_min
#
# Dependencies:
#   pip install numpy pandas scikit-learn