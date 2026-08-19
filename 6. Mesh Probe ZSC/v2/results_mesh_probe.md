python mesh_probe.py --zones probe_input.csv                                                                                  
                                                                                               
========================================================================
  MESH PROBE — probe_input.csv
========================================================================

  [CSV] is_load_zone column found — auto-adding load zones: [1, 6]

  Zones loaded : 6
  Load zones   : [1, 6]  (from CSV + --load)
  User-flagged : none

  Geometry constants (derived from mesh geometry, not data statistics):
    L_CHAR             = 50.0000   [characteristic length]
    KAPPA_MAX          = 0.0200   [max curvature in model]
    KAPPA_BIFURCATION  = 0.0333   [h=R/6 condition — Β-anchor]
    DIST_NF            = 5.0000   [BC influence radius]

  Encoded features: 37 total  (25 Π + 9 Ε + 3 ×)

========================================================================
  PROBE — structural typing confirmation
  Target: geometry-derived refinement score (no solver data)
========================================================================

  Top 10 features by importance:
  Rank                       Feature     Imp%  Type
  ────  ────────────────────────────  ───────  ────
     1             pi_dist_bc_sin_pi   20.26%  Π
     2               cross_bc_x_edge   18.97%  ×
     3            pi_dist_bc_sin_pi2   17.67%  Π
     4             pi_dist_bc_cos_pi   15.09%  Π
     5            pi_dist_bc_sin_2pi   14.66%  Π
     6            pi_dist_bc_cascade   13.36%  Π
     7          pi_kappa_max_cascade    0.00%  Π
     8          pi_kappa_max_sin_2pi    0.00%  Π
     9           pi_kappa_max_cos_pi    0.00%  Π
    10           pi_kappa_max_sin_pi    0.00%  Π

  Grouped importance:
    Π (cascade)    : 81.0%
    Ε (equilibrium): 0.0%
    × (cross)      : 19.0%

  Dominant: Π  (810344827.6×)  →  CONFIRMED — cascade character drives refinement. Π-encoding structurally correct.

========================================================================
  ZONE CLASSIFICATION
========================================================================

  ZoneID  Type   Score    Π/Ε  Recommendation
  ──────  ────  ──────  ─────  ────────────────────────────────────────────────
       1     Π   0.215   0.44  REFINE — physics-driven. Load application zone: refine regardless of local geometry.  ← LOAD
       2     Β   0.115   0.57  REVIEW — neighbor-driven. Adjacent zones have mixed classifications.
       3     Β   0.115   0.57  REVIEW — neighbor-driven. Adjacent zones have mixed classifications.
       4     Β   0.115   0.57  REVIEW — neighbor-driven. Adjacent zones have mixed classifications.
       5     Β   0.115   0.57  REVIEW — neighbor-driven. Adjacent zones have mixed classifications.
       6     Π   0.215   0.44  REFINE — physics-driven. Load application zone: refine regardless of local geometry.  ← LOAD

  Detailed descriptions:

  Zone 1 [Π] — physics-driven load zone
    Zone 1: user-identified load application point. Physics requires Π-level refinement independent of local curvature (κ_max = 0.0000, geometry score = 0.21). Stress concentration at load application is solver-driven, not shape-driven. Recommended h ≤ 1000000000.0000 as geometry lower bound; tighten further based on expected stress gradient.

  Zone 2 [Β] — neighbor-driven (mixed: Ε, Π)
    Zone 2: flat geometry (κ_max ≈ 0, score 0.12) with mixed neighbor types (Π, Ε, Ε, Π). Sits at a regime boundary between cascade and equilibrium. Apply moderate refinement and inspect transition.

  Zone 3 [Β] — neighbor-driven (mixed: Ε, Π)
    Zone 3: flat geometry (κ_max ≈ 0, score 0.12) with mixed neighbor types (Π, Ε, Ε, Π). Sits at a regime boundary between cascade and equilibrium. Apply moderate refinement and inspect transition.

  Zone 4 [Β] — neighbor-driven (mixed: Ε, Π)
    Zone 4: flat geometry (κ_max ≈ 0, score 0.12) with mixed neighbor types (Π, Ε, Ε, Π). Sits at a regime boundary between cascade and equilibrium. Apply moderate refinement and inspect transition.

  Zone 5 [Β] — neighbor-driven (mixed: Ε, Π)
    Zone 5: flat geometry (κ_max ≈ 0, score 0.12) with mixed neighbor types (Π, Ε, Ε, Π). Sits at a regime boundary between cascade and equilibrium. Apply moderate refinement and inspect transition.

  Zone 6 [Π] — physics-driven load zone
    Zone 6: user-identified load application point. Physics requires Π-level refinement independent of local curvature (κ_max = 0.0000, geometry score = 0.21). Stress concentration at load application is solver-driven, not shape-driven. Recommended h ≤ 1000000000.0000 as geometry lower bound; tighten further based on expected stress gradient.

  ════════════════════════════════════════════════════════════════════
  BOUNDARY ZONE RESOLUTION — 4 Β zone(s) require a decision
  ════════════════════════════════════════════════════════════════════
  Options are generated in log space between Π-neighbor (finest)
  and Ε-neighbor (coarsest) element sizes. Middle two options
  include a bounded stochastic perturbation (±~35%) — re-run
  without --seed to see alternative intermediate values.

  ────────────────────────────────────────────────────────────────────
  Zone 2 [Β]
  ────────────────────────────────────────────────────────────────────
  Current h      : 5.0000
  Finest bound   : 1.2500  (from Π neighbors: [1, 6])
  Coarsest bound : 5.0000  (conservative — no Ε neighbors)

  Opt               Label    h target    Surface elements
  ───  ──────────────────  ──────────  ──────────────────
    1    Coarse (Ε level)      5.0000    1.00×  ↑ more 
    2               Light      3.2024    2.44×  ↑ more 
    3            Moderate      1.9897    6.31×  ↑ more 
    4      Fine (Π level)      1.2500   16.00×  ↑ more 

  Zone 2 — choose [1-4] or Enter to skip: 1
  ✓  Zone 2 → h=5.0000 (Coarse (Ε level), 1.00× elements)

  ────────────────────────────────────────────────────────────────────
  Zone 3 [Β]
  ────────────────────────────────────────────────────────────────────
  Current h      : 5.0000
  Finest bound   : 1.2500  (from Π neighbors: [1, 6])
  Coarsest bound : 5.0000  (conservative — no Ε neighbors)

  Opt               Label    h target    Surface elements
  ───  ──────────────────  ──────────  ──────────────────
    1    Coarse (Ε level)      5.0000    1.00×  ↑ more 
    2               Light      2.7739    3.25×  ↑ more 
    3            Moderate      1.8520    7.29×  ↑ more 
    4      Fine (Π level)      1.2500   16.00×  ↑ more 

  Zone 3 — choose [1-4] or Enter to skip: 1
  ✓  Zone 3 → h=5.0000 (Coarse (Ε level), 1.00× elements)

  ────────────────────────────────────────────────────────────────────
  Zone 4 [Β]
  ────────────────────────────────────────────────────────────────────
  Current h      : 5.0000
  Finest bound   : 1.2500  (from Π neighbors: [1, 6])
  Coarsest bound : 5.0000  (conservative — no Ε neighbors)

  Opt               Label    h target    Surface elements
  ───  ──────────────────  ──────────  ──────────────────
    1    Coarse (Ε level)      5.0000    1.00×  ↑ more 
    2               Light      2.7486    3.31×  ↑ more 
    3            Moderate      1.8188    7.56×  ↑ more 
    4      Fine (Π level)      1.2500   16.00×  ↑ more 

  Zone 4 — choose [1-4] or Enter to skip: 1
  ✓  Zone 4 → h=5.0000 (Coarse (Ε level), 1.00× elements)

  ────────────────────────────────────────────────────────────────────
  Zone 5 [Β]
  ────────────────────────────────────────────────────────────────────
  Current h      : 5.0000
  Finest bound   : 1.2500  (from Π neighbors: [1, 6])
  Coarsest bound : 5.0000  (conservative — no Ε neighbors)

  Opt               Label    h target    Surface elements
  ───  ──────────────────  ──────────  ──────────────────
    1    Coarse (Ε level)      5.0000    1.00×  ↑ more 
    2               Light      2.9333    2.91×  ↑ more 
    3            Moderate      1.7404    8.25×  ↑ more 
    4      Fine (Π level)      1.2500   16.00×  ↑ more 

  Zone 5 — choose [1-4] or Enter to skip: 1
  ✓  Zone 5 → h=5.0000 (Coarse (Ε level), 1.00× elements)

========================================================================