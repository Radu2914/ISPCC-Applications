PS C:\Users\Radu\Desktop\ISPCC\4. Enigma Residual Structure\v2> python enigma_probe_v2.py

========================================================================
  SIMULATOR VERIFICATION
========================================================================
  Encrypting known plaintext → ciphertext
  Match: 131/131  ✓ Simulator verified
  First 10: plain=KOMXBDMXUU  cipher=QBHEWTDFEQ  simulated=QBHEWTDFEQ

  Plugboard structure (e-type constants):
    Letter   Partner    Dist   Plugged
  ────────  ────────  ──────  ────────
         B         Q      15       YES
         C         H       5       YES
         D         I       5       YES
         E         J       5       YES
         G         L       5       YES
         K         P       5       YES
         N         V       8       YES
         O         U       6       YES
         S         Z       7       YES
         T         Y       5       YES
  Self-connected (dist=0): ['A', 'F', 'M', 'R', 'W', 'X']
  Max plugboard distance: 15 (Q↔B — the 'longest reach' pair)
[INFO] Generating substitution table (17576 positions × 26 inputs = 456,976 pairs)...
[INFO] Generated 456,976 pairs in 1.1s
[VERIFY] Self-mappings (inp==out): 0 (Enigma: must be 0)
[INFO] Sampled 30,000 pairs for probe

  Substitution offset distribution:
  mean=13.04  std=7.22  min=1  max=25
  Value 0 present: False (False = Enigma invariant OK)

  Features: π=35, e=12, cross=3, raw=7
  N/p ratio: 600× (vs v1: 8.9×)

========================================================================
  PROBE — RF importance  |  n=30,000, 500 trees
  Target: (output − input) mod 26  [what Enigma does at each position]
========================================================================

  Top 20 features:
  Rank                   Feature       Imp  Type  Layer
  ────  ────────────────────────  ────────  ────  ────────────────────────────
     1         cross_fast_x_char    0.0738     ×  position × involution interaction
     2          cross_mid_x_plug    0.0620     ×  position × involution interaction
     3         cross_fast_x_plug    0.0507     ×  position × involution interaction
     4         e_plug_part_pow_e    0.0311     e  plugboard partner value
     5       e_plug_part_exp_neg    0.0310     e  plugboard partner value
     6           pi_left_sin_pi2    0.0307     π  left rotor position
     7            e_char_exp_neg    0.0304     e  input letter value (bounded)
     8           pi_left_cascade    0.0303     π  left rotor position
     9              e_char_pow_e    0.0301     e  input letter value (bounded)
    10         e_plug_part_gauss    0.0293     e  plugboard partner value
    11              e_char_gauss    0.0290     e  input letter value (bounded)
    12            pi_left_cos_pi    0.0288     π  left rotor position
    13        pi_notch_m_cascade    0.0285     π  Rotor III notch distance  ← bifurcation
    14            pi_left_sin_pi    0.0285     π  left rotor position
    15        pi_notch_m_sin_pi2    0.0277     π  Rotor III notch distance  ← bifurcation
    16        pi_notch_m_sin_2pi    0.0268     π  Rotor III notch distance  ← bifurcation
    17         pi_notch_m_cos_pi    0.0259     π  Rotor III notch distance  ← bifurcation
    18           pi_left_sin_2pi    0.0250     π  left rotor position
    19        pi_notch_f_cascade    0.0242     π  Rotor VIII notch distance ← bifurcation
    20        pi_notch_f_sin_pi2    0.0238     π  Rotor VIII notch distance ← bifurcation

  ── Grouped importances ──
  π-type (position / rotor)     : 0.6013  ( 60.1%)
  e-type (char / plugboard)     : 0.2121  ( 21.2%)
  Cross-products (π × e)        : 0.1865  ( 18.7%)

  Plugboard features only (e-type subset):
  e_plug_dist_exp_neg, e_plug_dist_pow_e, e_plug_dist_gauss, e_is_plugged_exp_neg, e_is_plugged_pow_e, e_is_plugged_gauss, e_plug_part_exp_neg, e_plug_part_pow_e, e_plug_part_gauss
  Combined importance: 0.1226  (12.3%)
  ← Position still dominates

========================================================================
  CV COMPARISON  (5-fold, n=30,000)
  v1 reference: all R² negative (data starvation, 239 samples)
========================================================================

  Model                               R²       vs v1  Note
  ────────────────────────────  ────────  ──────────  ──────────────────────────────
  Raw features                   -0.0667         N/A  
  π-only (position)              -0.0837         N/A  
  e-only (character)             -0.0010         N/A  
  Plugboard only (e↓)            -0.0010         N/A  ← TURING: char > position
  Full (π + e + cross)           -0.0965         N/A  

========================================================================
  3-STAGE PIPELINE  (Enigma-specific decomposition)
  Stage 1 Ridge  : π features  (rotor period grammar)
  Stage 2 RF     : e features  → Stage-1 residuals (involution dialect)
  If ΔR² > 0: the plugboard involution explains variance above rotor period
========================================================================

  Stage 1 only (Ridge, π features)     : R² = -0.0008
  3-stage      (π grammar + e dialect) : R² = -0.0017
  Stage 2 ΔR²  (e dialect contribution): -0.0009
  → π alone sufficient: rotor period dominates

========================================================================
  VERDICT: Does the probe detect Turing's insight?
  (v2: 30,000 samples, explicit plugboard features, full M4 key)
========================================================================

  Criteria:
  [✗] e-type importance (21.2%) > π-type importance (60.1%)
  [✓] Plugboard-only R² (-0.0010) > π-only R² (-0.0837)
  [✗] Stage-2 e-dialect adds ΔR²=-0.0009 above Stage-1 π-grammar

  1/3 criteria met  (↑ from v1 0/3)

  → PARTIAL DETECTION (1/3)

  One criterion confirms the e-type (involution) layer.
  The full 456,976-sample dataset was sufficient to see it partially.
  Criterion met: see above.


========================================================================
  v1 → v2 COMPARISON
========================================================================
  Metric                                 v1 (239 samples)         v2 (30,000)
  ───────────────────────────────────  ──────────────────  ──────────────────
  Dataset size                                        239              30,000
  π-type importance                                 39.1%               60.1%
  e-type importance                                 32.8%               21.2%
  Cross-product importance                          28.1%               18.7%
  Full encoded R²                                 -0.0953             -0.0965
  π-only R²                                       -0.2356             -0.0837
  e-only R²                                       -0.2620             -0.0010
  3-stage ΔR²                                     -0.2247             -0.0009
  Turing criteria met                                 0/3                  1/3

[TIMING] Total: 26.6s