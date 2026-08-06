PS D:\ISPCC\6. Mesh Probe> python mesh_probe.py --test sphere

========================================================================
  MESH PROBE — SYNTHETIC TEST: SPHERE
========================================================================

  Sphere R=50mm — uniform κ=1/R. All zones expect Ε: curvature well below bifurcation threshold.

  Zones loaded : 8
  User-flagged : none

  Geometry constants (derived from mesh geometry, not data statistics):
    L_CHAR             = 27.4240   [characteristic length]
    KAPPA_MAX          = 0.0210   [max curvature in model]
    KAPPA_BIFURCATION  = 0.0635   [h=R/6 condition — Β-anchor]
    DIST_NF            = 2.7424   [BC influence radius]

  Encoded features: 37 total  (25 Π + 9 Ε + 3 ×)

========================================================================
  PROBE — structural typing confirmation
  Target: geometry-derived refinement score (no solver data)
========================================================================

  Top 10 features by importance:
  Rank                       Feature     Imp%  Type
  ────  ────────────────────────────  ───────  ────
     1           pi_kappa_max_sin_pi   12.10%  Π
     2          pi_kappa_max_sin_pi2   10.15%  Π
     3          pi_kappa_max_sin_2pi    9.89%  Π
     4            cross_kappa_x_flat    9.84%  ×
     5          pi_kappa_max_cascade    8.11%  Π
     6           pi_kappa_max_cos_pi    6.68%  Π
     7                pi_skew_sin_pi    4.51%  Π
     8               pi_skew_sin_2pi    3.93%  Π
     9             pi_aspect_sin_2pi    2.93%  Π
    10               pi_skew_sin_pi2    2.92%  Π

  Grouped importance:
    Π (cascade)    : 79.6%
    Ε (equilibrium): 9.2%
    × (cross)      : 11.2%

  Dominant: Π  (8.7×)  →  CONFIRMED — cascade character drives refinement. Π-encoding structurally correct.

========================================================================
  ZONE CLASSIFICATION
========================================================================

  ZoneID  Type   Score    Π/Ε  Recommendation
  ──────  ────  ──────  ─────  ────────────────────────────────────────────────
       1     Ε   0.237   0.61  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       2     Ε   0.244   0.60  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       3     Ε   0.239   0.62  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       4     Ε   0.235   0.61  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       5     Ε   0.249   0.69  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       6     Ε   0.248   0.62  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       7     Ε   0.238   0.62  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       8     Ε   0.247   0.60  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.

  Detailed descriptions:

  Zone 1 [Ε] — score=0.24
    Zone 1: self-regulating geometry. κ_max = 0.0199 is well below bifurcation threshold κ_bif = 0.0635. Score 0.24 << 1.0. Normal deviation 3.2° confirms surface flatness. Mesh error is bounded and self-correcting here.

    Zone 2: self-regulating geometry. κ_max = 0.0210 is well below bifurcation threshold κ_bif = 0.0635. Score 0.24 << 1.0. Normal deviation 4.2° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 3 [Ε] — score=0.24
    Zone 3: self-regulating geometry. κ_max = 0.0195 is well below bifurcation threshold κ_bif = 0.0635. Score 0.24 << 1.0. Normal deviation 3.4° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 4 [Ε] — score=0.24
    Zone 4: self-regulating geometry. κ_max = 0.0193 is well below bifurcation threshold κ_bif = 0.0635. Score 0.24 << 1.0. Normal deviation 3.3° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 5 [Ε] — score=0.25
    Zone 5: self-regulating geometry. κ_max = 0.0201 is well below bifurcation threshold κ_bif = 0.0635. Score 0.25 << 1.0. Normal deviation 4.4° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 6 [Ε] — score=0.25
    Zone 6: self-regulating geometry. κ_max = 0.0209 is well below bifurcation threshold κ_bif = 0.0635. Score 0.25 << 1.0. Normal deviation 3.2° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 7 [Ε] — score=0.24
    Zone 7: self-regulating geometry. κ_max = 0.0188 is well below bifurcation threshold κ_bif = 0.0635. Score 0.24 << 1.0. Normal deviation 3.3° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 8 [Ε] — score=0.25
    Zone 8: self-regulating geometry. κ_max = 0.0209 is well below bifurcation threshold κ_bif = 0.0635. Score 0.25 << 1.0. Normal deviation 3.1° confirms surface flatness. Mesh error is bounded and self-correcting here.

========================================================================
  VALIDATION — expected vs probe output
========================================================================
  Zone  1:  expected Ε  got Ε  ✓  (score=0.24, Π/Ε=0.61)
  Zone  2:  expected Ε  got Ε  ✓  (score=0.24, Π/Ε=0.60)
  Zone  3:  expected Ε  got Ε  ✓  (score=0.24, Π/Ε=0.62)
  Zone  4:  expected Ε  got Ε  ✓  (score=0.23, Π/Ε=0.61)
  Zone  5:  expected Ε  got Ε  ✓  (score=0.25, Π/Ε=0.69)
  Zone  6:  expected Ε  got Ε  ✓  (score=0.25, Π/Ε=0.62)
  Zone  7:  expected Ε  got Ε  ✓  (score=0.24, Π/Ε=0.62)
  Zone  8:  expected Ε  got Ε  ✓  (score=0.25, Π/Ε=0.60)

  Accuracy (non-flagged zones): 8/8 = 100%

========================================================================

PS D:\ISPCC\6. Mesh Probe> python mesh_probe.py --test cylinder

========================================================================
  MESH PROBE — SYNTHETIC TEST: CYLINDER
========================================================================

  Cylinder R=5mm L=40mm, rough mesh h~3mm on curved surface. curved surface (6h/R=3.6)→Π, flat ends→Ε, edge transitions (6h/R≈1.1)→Β.

  Zones loaded : 8
  User-flagged : none

  Geometry constants (derived from mesh geometry, not data statistics):
    L_CHAR             = 40.0000   [characteristic length]
    KAPPA_MAX          = 0.2000   [max curvature in model]
    KAPPA_BIFURCATION  = 0.0556   [h=R/6 condition — Β-anchor]
    DIST_NF            = 4.0000   [BC influence radius]

  Encoded features: 37 total  (25 Π + 9 Ε + 3 ×)

========================================================================
  PROBE — structural typing confirmation
  Target: geometry-derived refinement score (no solver data)
========================================================================

  Top 10 features by importance:
  Rank                       Feature     Imp%  Type
  ────  ────────────────────────────  ───────  ────
     1          pi_kappa_max_sin_2pi    7.18%  Π
     2          pi_kappa_mean_sin_pi    6.70%  Π
     3           pi_kappa_max_sin_pi    6.49%  Π
     4          pi_kappa_mean_cos_pi    6.20%  Π
     5         pi_kappa_mean_sin_2pi    5.99%  Π
     6               pi_skew_sin_pi2    5.93%  Π
     7          pi_kappa_max_sin_pi2    5.62%  Π
     8               pi_skew_sin_2pi    5.22%  Π
     9               pi_skew_cascade    5.07%  Π
    10         pi_kappa_mean_cascade    4.94%  Π

  Grouped importance:
    Π (cascade)    : 88.6%
    Ε (equilibrium): 5.8%
    × (cross)      : 5.6%

  Dominant: Π  (15.3×)  →  CONFIRMED — cascade character drives refinement. Π-encoding structurally correct.

========================================================================
  ZONE CLASSIFICATION
========================================================================

  ZoneID  Type   Score    Π/Ε  Recommendation
  ──────  ────  ──────  ─────  ────────────────────────────────────────────────
       1     Π   1.901   1.04  REFINE — cascade regime. Reduce element size to ≤ 0.8333 (same units as curvature input).
       2     Π   1.891   0.94  REFINE — cascade regime. Reduce element size to ≤ 0.8333 (same units as curvature input).
       3     Π   1.912   1.09  REFINE — cascade regime. Reduce element size to ≤ 0.8333 (same units as curvature input).
       4     Ε   0.117   0.62  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       5     Ε   0.099   0.58  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       6     Β   0.812   0.94  REVIEW — boundary regime. Apply moderate refinement and validate visually. Refinement decision is non-obvious.
       7     Β   0.768   0.94  REVIEW — boundary regime. Apply moderate refinement and validate visually. Refinement decision is non-obvious.
       8     Ε   0.076   0.52  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.

  Detailed descriptions:

  Zone 1 [Π] — score=1.90
    Zone 1: curvature-driven cascade. κ_max = 0.2000 exceeds bifurcation threshold κ_bif = 0.0556 (h = R/6 condition). Score 1.90 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.15, aspect ratio 1.50.

  Zone 2 [Π] — score=1.89
    Zone 2: curvature-driven cascade. κ_max = 0.2000 exceeds bifurcation threshold κ_bif = 0.0556 (h = R/6 condition). Score 1.89 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.12, aspect ratio 1.40.

  Zone 3 [Π] — score=1.91
    Zone 3: curvature-driven cascade. κ_max = 0.2000 exceeds bifurcation threshold κ_bif = 0.0556 (h = R/6 condition). Score 1.91 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.18, aspect ratio 1.60.

  Zone 4 [Ε] — score=0.12
    Zone 4: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.0556. Score 0.12 << 1.0. Normal deviation 1.0° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 5 [Ε] — score=0.10
    Zone 5: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.0556. Score 0.10 << 1.0. Normal deviation 1.0° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 6 [Β] — score=0.81
    Zone 6: at structural transition between cascade and equilibrium. κ_max = 0.1200 near bifurcation threshold κ_bif = 0.0556. Score 0.81 in transition zone [0.6, 1.5]. Π/Ε feature balance: 0.94. Apply moderate refinement and inspect solver convergence.

  Zone 7 [Β] — score=0.77
    Zone 7: at structural transition between cascade and equilibrium. κ_max = 0.1200 near bifurcation threshold κ_bif = 0.0556. Score 0.77 in transition zone [0.6, 1.5]. Π/Ε feature balance: 0.94. Apply moderate refinement and inspect solver convergence.

  Zone 8 [Ε] — score=0.08
    Zone 8: self-regulating geometry. κ_max = 0.0010 is well below bifurcation threshold κ_bif = 0.0556. Score 0.08 << 1.0. Normal deviation 0.5° confirms surface flatness. Mesh error is bounded and self-correcting here.

========================================================================
  VALIDATION — expected vs probe output
========================================================================
  Zone  1:  expected Π  got Π  ✓  (score=1.90, Π/Ε=1.04)
  Zone  2:  expected Π  got Π  ✓  (score=1.89, Π/Ε=0.94)
  Zone  3:  expected Π  got Π  ✓  (score=1.91, Π/Ε=1.09)
  Zone  4:  expected Ε  got Ε  ✓  (score=0.12, Π/Ε=0.62)
  Zone  5:  expected Ε  got Ε  ✓  (score=0.10, Π/Ε=0.58)
  Zone  6:  expected Β  got Β  ✓  (score=0.81, Π/Ε=0.94)
  Zone  7:  expected Β  got Β  ✓  (score=0.77, Π/Ε=0.94)
  Zone  8:  expected Ε  got Ε  ✓  (score=0.08, Π/Ε=0.52)

  Accuracy (non-flagged zones): 8/8 = 100%
========================================================================

PS D:\ISPCC\6. Mesh Probe> python mesh_probe.py --test box     

========================================================================
  MESH PROBE — SYNTHETIC TEST: BOX
========================================================================

  Box — flat faces→Ε, sharp edges/corners→Π, face-edge transitions→Β.

  Zones loaded : 8
  User-flagged : none

  Geometry constants (derived from mesh geometry, not data statistics):
    L_CHAR             = 50.0000   [characteristic length]
    KAPPA_MAX          = 15.0000   [max curvature in model]
    KAPPA_BIFURCATION  = 0.3333   [h=R/6 condition — Β-anchor]
    DIST_NF            = 5.0000   [BC influence radius]

  Encoded features: 37 total  (25 Π + 9 Ε + 3 ×)

========================================================================
  PROBE — structural typing confirmation
  Target: geometry-derived refinement score (no solver data)
========================================================================

  Top 10 features by importance:
  Rank                       Feature     Imp%  Type
  ────  ────────────────────────────  ───────  ────
     1          pi_kappa_mean_cos_pi    7.15%  Π
     2           e_area_frac_exp_neg    6.69%  Ε
     3          e_normal_dev_exp_neg    6.58%  Ε
     4          pi_kappa_mean_sin_pi    6.48%  Π
     5              pi_aspect_sin_pi    6.44%  Π
     6              pi_aspect_cos_pi    6.22%  Π
     7                pi_skew_cos_pi    5.91%  Π
     8             e_area_frac_pow_e    5.48%  Ε
     9             e_area_frac_gauss    5.17%  Ε
    10          pi_kappa_max_sin_pi2    5.15%  Π

  Grouped importance:
    Π (cascade)    : 58.3%
    Ε (equilibrium): 37.2%
    × (cross)      : 4.5%

  Dominant: Π  (1.6×)  →  CONFIRMED — cascade character drives refinement. Π-encoding structurally correct.

========================================================================
  ZONE CLASSIFICATION
========================================================================

  ZoneID  Type   Score    Π/Ε  Recommendation
  ──────  ────  ──────  ─────  ────────────────────────────────────────────────
       1     Ε   0.104   0.51  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       2     Ε   0.107   0.48  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       3     Ε   0.096   0.45  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       4     Π  12.389   0.84  REFINE — cascade regime. Reduce element size to ≤ 0.0208 (same units as curvature input).
       5     Π  12.325   0.81  REFINE — cascade regime. Reduce element size to ≤ 0.0208 (same units as curvature input).
       6     Π   9.519   0.78  REFINE — cascade regime. Reduce element size to ≤ 0.0111 (same units as curvature input).
       7     Π   9.445   0.76  REFINE — cascade regime. Reduce element size to ≤ 0.0111 (same units as curvature input).
       8     Β   0.979   0.98  REVIEW — boundary regime. Apply moderate refinement and validate visually. Refinement decision is non-obvious.

  Detailed descriptions:

  Zone 1 [Ε] — score=0.10
    Zone 1: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.3333. Score 0.10 << 1.0. Normal deviation 0.5° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 2 [Ε] — score=0.11
    Zone 2: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.3333. Score 0.11 << 1.0. Normal deviation 0.5° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 3 [Ε] — score=0.10
    Zone 3: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.3333. Score 0.10 << 1.0. Normal deviation 0.5° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 4 [Π] — score=12.39
    Zone 4: curvature-driven cascade. κ_max = 8.0000 exceeds bifurcation threshold κ_bif = 0.3333 (h = R/6 condition). Score 12.39 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.70, aspect ratio 4.50.

  Zone 5 [Π] — score=12.32
    Zone 5: curvature-driven cascade. κ_max = 8.0000 exceeds bifurcation threshold κ_bif = 0.3333 (h = R/6 condition). Score 12.32 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.65, aspect ratio 4.20.

  Zone 6 [Π] — score=9.52
    Zone 6: curvature-driven cascade. κ_max = 15.0000 exceeds bifurcation threshold κ_bif = 0.3333 (h = R/6 condition). Score 9.52 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.90, aspect ratio 6.00.

  Zone 7 [Π] — score=9.45
    Zone 7: curvature-driven cascade. κ_max = 15.0000 exceeds bifurcation threshold κ_bif = 0.3333 (h = R/6 condition). Score 9.45 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.88, aspect ratio 5.80.

  Zone 8 [Β] — score=0.98
    Zone 8: at structural transition between cascade and equilibrium. κ_max = 0.5000 near bifurcation threshold κ_bif = 0.3333. Score 0.98 in transition zone [0.6, 1.5]. Π/Ε feature balance: 0.98. Apply moderate refinement and inspect solver convergence.

========================================================================
  VALIDATION — expected vs probe output
========================================================================
  Zone  1:  expected Ε  got Ε  ✓  (score=0.10, Π/Ε=0.51)
  Zone  2:  expected Ε  got Ε  ✓  (score=0.11, Π/Ε=0.48)
  Zone  3:  expected Ε  got Ε  ✓  (score=0.10, Π/Ε=0.45)
  Zone  4:  expected Π  got Π  ✓  (score=12.39, Π/Ε=0.84)
  Zone  5:  expected Π  got Π  ✓  (score=12.32, Π/Ε=0.81)
  Zone  6:  expected Π  got Π  ✓  (score=9.52, Π/Ε=0.78)
  Zone  7:  expected Π  got Π  ✓  (score=9.45, Π/Ε=0.76)
  Zone  8:  expected Β  got Β  ✓  (score=0.98, Π/Ε=0.98)

  Accuracy (non-flagged zones): 8/8 = 100%
========================================================================

PS D:\ISPCC\6. Mesh Probe> python mesh_probe.py --test cone

========================================================================
  MESH PROBE — SYNTHETIC TEST: CONE
========================================================================

  Cone R_tip=0.5mm R_base=15mm H=50mm. tip/upper (6h/R>2)→Π, mid/lower/base-edge (6h/R≈0.6-1.5)→Β, base flat/far (6h/R<0.1)→Ε.

  Zones loaded : 8
  User-flagged : none

  Geometry constants (derived from mesh geometry, not data statistics):
    L_CHAR             = 40.0000   [characteristic length]
    KAPPA_MAX          = 4.0000   [max curvature in model]
    KAPPA_BIFURCATION  = 0.0775   [h=R/6 condition — Β-anchor]
    DIST_NF            = 4.0000   [BC influence radius]

  Encoded features: 37 total  (25 Π + 9 Ε + 3 ×)

========================================================================
  PROBE — structural typing confirmation
  Target: geometry-derived refinement score (no solver data)
========================================================================

  Top 10 features by importance:
  Rank                       Feature     Imp%  Type
  ────  ────────────────────────────  ───────  ────
     1           pi_kappa_max_sin_pi    6.10%  Π
     2          pi_kappa_max_sin_2pi    5.80%  Π
     3              e_edge_len_pow_e    4.91%  Ε
     4          pi_kappa_max_cascade    4.46%  Π
     5          pi_kappa_mean_cos_pi    4.45%  Π
     6            e_edge_len_exp_neg    4.35%  Ε
     7              pi_aspect_cos_pi    4.23%  Π
     8           e_area_frac_exp_neg    4.19%  Ε
     9           pi_kappa_max_cos_pi    3.98%  Π
    10          pi_kappa_mean_sin_pi    3.84%  Π

  Grouped importance:
    Π (cascade)    : 61.1%
    Ε (equilibrium): 32.3%
    × (cross)      : 6.6%

  Dominant: Π  (1.9×)  →  CONFIRMED — cascade character drives refinement. Π-encoding structurally correct.

========================================================================
  ZONE CLASSIFICATION
========================================================================

  ZoneID  Type   Score    Π/Ε  Recommendation
  ──────  ────  ──────  ─────  ────────────────────────────────────────────────
       1     Π   1.691   0.74  REFINE — cascade regime. Reduce element size to ≤ 0.0417 (same units as curvature input).
       2     Π   1.679   0.88  REFINE — cascade regime. Reduce element size to ≤ 0.2488 (same units as curvature input).
       3     Β   0.603   0.80  REVIEW — boundary regime. Apply moderate refinement and validate visually. Refinement decision is non-obvious.
       4     Β   0.846   1.01  REVIEW — boundary regime. Apply moderate refinement and validate visually. Refinement decision is non-obvious.
       5     Ε   0.092   0.51  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       6     Β   0.628   0.77  REVIEW — boundary regime. Apply moderate refinement and validate visually. Refinement decision is non-obvious.
       7     Ε   0.090   0.48  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.
       8     Ε   0.104   0.51  COARSE OK — equilibrium regime. Current element density is acceptable for this zone.

  Detailed descriptions:

  Zone 1 [Π] — score=1.69
    Zone 1: curvature-driven cascade. κ_max = 4.0000 exceeds bifurcation threshold κ_bif = 0.0775 (h = R/6 condition). Score 1.69 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.80, aspect ratio 5.50.

  Zone 2 [Π] — score=1.68
    Zone 2: curvature-driven cascade. κ_max = 0.6700 exceeds bifurcation threshold κ_bif = 0.0775 (h = R/6 condition). Score 1.68 >> 1.0. Mesh error compounds non-repeatingly without refinement. Skewness 0.55, aspect ratio 3.50.

  Zone 3 [Β] — score=0.60
    Zone 3: at structural transition between cascade and equilibrium. κ_max = 0.2000 near bifurcation threshold κ_bif = 0.0775. Score 0.60 in transition zone [0.6, 1.5]. Π/Ε feature balance: 0.80. Apply moderate refinement and inspect solver convergence.

  Zone 4 [Β] — score=0.85
    Zone 4: at structural transition between cascade and equilibrium. κ_max = 0.1000 near bifurcation threshold κ_bif = 0.0775. Score 0.85 in transition zone [0.6, 1.5]. Π/Ε feature balance: 1.01. Apply moderate refinement and inspect solver convergence.

  Zone 5 [Ε] — score=0.09
    Zone 5: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.0775. Score 0.09 << 1.0. Normal deviation 1.0° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 6 [Β] — score=0.63
    Zone 6: at structural transition between cascade and equilibrium. κ_max = 0.0670 near bifurcation threshold κ_bif = 0.0775. Score 0.63 in transition zone [0.6, 1.5]. Π/Ε feature balance: 0.76. Apply moderate refinement and inspect solver convergence.

  Zone 7 [Ε] — score=0.09
    Zone 7: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.0775. Score 0.09 << 1.0. Normal deviation 0.5° confirms surface flatness. Mesh error is bounded and self-correcting here.

  Zone 8 [Ε] — score=0.10
    Zone 8: self-regulating geometry. κ_max = 0.0020 is well below bifurcation threshold κ_bif = 0.0775. Score 0.10 << 1.0. Normal deviation 1.0° confirms surface flatness. Mesh error is bounded and self-correcting here.

========================================================================
  VALIDATION — expected vs probe output
========================================================================
  Zone  1:  expected Π  got Π  ✓  (score=1.69, Π/Ε=0.74)
  Zone  2:  expected Π  got Π  ✓  (score=1.68, Π/Ε=0.88)
  Zone  3:  expected Β  got Β  ✓  (score=0.60, Π/Ε=0.80)
  Zone  4:  expected Β  got Β  ✓  (score=0.85, Π/Ε=1.01)
  Zone  5:  expected Ε  got Ε  ✓  (score=0.09, Π/Ε=0.51)
  Zone  6:  expected Β  got Β  ✓  (score=0.63, Π/Ε=0.77)
  Zone  7:  expected Ε  got Ε  ✓  (score=0.09, Π/Ε=0.48)
  Zone  8:  expected Ε  got Ε  ✓  (score=0.10, Π/Ε=0.51)

  Accuracy (non-flagged zones): 8/8 = 100%

========================================================================