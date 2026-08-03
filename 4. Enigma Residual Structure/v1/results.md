PS C:\Users\Radu\Desktop\ISPCC\4. Enigma Residual Structure\v1> python enigma_probe.py

[INFO] Dataset: 239 aligned pairs  (Part 1: 131, Part 2: 108)
[VERIFY] Self-mappings (cipher == plain): 0/239  ✓ CONFIRMED — Enigma invariant holds

[INFO] Substitution offset distribution:
  mean=13.08  std=7.49  min=1  max=25
  Value 0 present: False  (False = Enigma no-self-map confirmed)

[INFO] Features:
  π-encoded (position/rotor)    :  15  ['pi_fast_sin_pi', 'pi_fast_cos_pi', 'pi_fast_sin_2pi', 'pi_fast_sin_pi2', 'pi_fast_cascade', 'pi_mid_sin_pi', 'pi_mid_cos_pi', 'pi_mid_sin_2pi', 'pi_mid_sin_pi2', 'pi_mid_cascade', 'pi_pos_sin_pi', 'pi_pos_cos_pi', 'pi_pos_sin_2pi', 'pi_pos_sin_pi2', 'pi_pos_cascade']
  e-encoded (character/invol.)  :   9  ['e_char_exp_neg', 'e_char_pow_e', 'e_char_gauss', 'e_gfreq_exp_neg', 'e_gfreq_pow_e', 'e_gfreq_gauss', 'e_lfreq_exp_neg', 'e_lfreq_pow_e', 'e_lfreq_gauss']
  Cross-products (π × e)        :   3  ['cross_fast_x_char', 'cross_fast_x_gfreq', 'cross_fast_x_lfreq']
  Raw baseline                  :   6
  Total encoded                 :  27
  N/p ratio (full encoded)      : 8.9

========================================================================
  PROBE — RF importance  |  target: (plain − cipher) mod 26
  [n=1000 trees for stable importance rankings]
========================================================================

  Top 15 features:
  Rank                     Feature       Imp  Type  Interpretation
  ────  ──────────────────────────  ────────  ────  ──────────────────────────────
     1          cross_fast_x_gfreq    0.0998     ×  position × character
     2           cross_fast_x_char    0.0976     ×  position × character
     3          cross_fast_x_lfreq    0.0833     ×  position × character
     4                e_char_gauss    0.0573     e  involution/char
     5              e_char_exp_neg    0.0508     e  cipher character value [bounded]
     6                e_char_pow_e    0.0489     e  cipher character value [bounded]
     7              pi_pos_cascade    0.0367     π  rotor period
     8              pi_mid_cascade    0.0341     π  rotor period
     9              pi_mid_sin_pi2    0.0329     π  middle rotor position
    10               e_gfreq_gauss    0.0322     e  involution/char
    11               e_gfreq_pow_e    0.0317     e  cipher char global frequency
    12             e_gfreq_exp_neg    0.0315     e  cipher char global frequency
    13              pi_pos_sin_2pi    0.0309     π  global message position
    14               pi_mid_sin_pi    0.0294     π  middle rotor position
    15               pi_mid_cos_pi    0.0288     π  middle rotor position

  ── Grouped importances ──
  π-type (position / rotor period)  : 0.3909  ( 39.1%)
  e-type (character / involution)   : 0.3284  ( 32.8%)
  Cross-products (π × e)            : 0.2807  ( 28.1%)

========================================================================
  CV COMPARISON  (5-fold, n=239)
  Isolates whether character (e) or position (π) predicts better
========================================================================

  Model                         R²  Interpretation
  ──────────────────────  ────────  ────────────────────────────────────────
  Raw features             -0.1441  baseline (raw position + char values)
  π-only (position)        -0.2356  rotor period alone
  e-only (character)       -0.2620  involution constraint alone
  Full (π + e)             -0.0953  both combined

  e-only > π-only? NO → position (rotor period) is stronger signal

========================================================================
  3-STAGE PIPELINE  (Enigma-specific)
  Stage 0  : π-encoded position features (no fitting)
  Stage 1  : Ridge on π features  → removes rotor-period grammar
  Stage 2  : RF on e features     → captures involution dialect
  Final    : Stage-1 + Stage-2 residual correction
  Turing's layer = Stage 2 contribution above Stage 1 alone
========================================================================

  Stage 1 only (Ridge, π features) : R² = -0.0503
  3-stage      (π grammar + e dial): R² = -0.2764
  Stage 2 contribution (ΔR²)       : -0.2261
  Stage 2 adds value: NO → position alone is sufficient

========================================================================
  VERDICT: Does the ISPCC probe detect Turing's insight?
========================================================================

  Enigma invariant (no self-mapping)    : CONFIRMED ✓
  π-type total importance               :  39.1%
  e-type total importance               :  32.8%
  e-only R²                             : -0.2620
  π-only R²                             : -0.2356
  Stage-2 (e dialect) ΔR²               : -0.2261

  Turing criteria met:
    [✗] e-type importance > π-type importance
    [✗] e-only R² > π-only R²
    [✗] Stage-2 (e dialect) adds R² above Stage-1 (π grammar)

  0/3 criteria met

  → π-TYPE DOMINANT (0/3 Turing criteria)

  Position (rotor period) is the stronger structural signal in this message.
  This means the probe finds the naive periodicity, not the deeper involution.
  With this message length (239 chars), the period-26 rotor structure dominates.
  More messages or longer text would likely expose the e-type layer.