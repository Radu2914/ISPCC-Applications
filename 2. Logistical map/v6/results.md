PS C:\Users\Radu\Desktop\ISPcC\2. Logistical map\v6> python logistical_map_norm_pi_v6.py
[INFO] building pool of 2000 samples (computed once)...
[INFO] stable  (λ<0): 1466/2000  (73.3%)
[INFO] chaotic (λ≥0): 534/2000  (26.7%)

[INFO] feature spaces:
  raw stats  : 11
  ENCODED_11 : 11
  Stage-0    : 5  [r, abs_dist, pienc_r, pienc_inv_n, pienc_abs_dist]

[INFO] v4 reference (random, 10 seeds, 5-fold CV, n=100):
  RF raw : 0.8798   XGB enc : 0.868

[INFO] sweep n∈[11, 22, 33, 50, 66, 100, 150, 200, 300, 500]  seeds=10  5-fold CV

  seed 1/10 done
  seed 2/10 done
  seed 3/10 done
  seed 4/10 done
  seed 5/10 done
  seed 6/10 done
  seed 7/10 done
  seed 8/10 done
  seed 9/10 done
  seed 10/10 done

============================================================================
  3-STAGE PIPELINE vs BASELINES  (10 seeds, 5-fold CV)
  Stage 1 Ridge : [r, abs_dist, pienc_r, pienc_inv_n, pienc_abs_dist]
  Stage 2 RF    : ENCODED_11 → Stage-1 residuals
============================================================================
      n          RF raw         XGB enc         3-stage   3s > RF?   3s > XGB?
  ─────  ──────────────  ──────────────  ──────────────  ─────────  ──────────
     11  -10.3309±13.784  -10.4238±11.936  -6.6187±8.495          YES         YES
     22  0.2220±0.399  0.2481±0.467  0.3760±0.364          YES         YES
     33  0.4369±0.493  0.5749±0.280  0.6021±0.297          YES         YES
     50  0.7947±0.079  0.7277±0.178  0.8094±0.069          YES         YES
     66  0.8052±0.103  0.8218±0.132  0.8332±0.074          YES         YES
    100  0.8798±0.070  0.8680±0.086  0.8902±0.027        YES         YES
    150  0.9176±0.028  0.8807±0.076  0.9023±0.031        NO          YES
    200  0.9316±0.025  0.9195±0.032  0.9220±0.022        NO          YES
    300  0.9591±0.015  0.9558±0.017  0.9476±0.018        NO          NO 
    500  0.9720±0.006  0.9715±0.007  0.9681±0.006        NO          NO 
────────────────────────────────────────────────────────────────────────────

  3-stage > RF raw : 6/10
  3-stage > XGB enc: 8/10

============================================================================
  CROSS-N EFFICIENCY — 3-stage N equivalent to baselines
============================================================================
   3-stage n     R²_3s   equiv RF raw   equiv XGB enc  speedup vs RF
  ──────────  ────────  ─────────────  ──────────────  ─────────────
          33    0.6021             50              50  1.5×
          50    0.8094            100              66  2.0×
          66    0.8332            100             100  1.5×
         100    0.8902            150             200  1.5×
         150    0.9023            150             200  1.0×
         200    0.9220            200             300  1.0×

============================================================================
  PIPELINE CONTRIBUTION (mean ΔR² vs XGB enc, all N)
============================================================================
  Mean  Δ across all N : +0.4088
  Median Δ             : +0.0219
  Min Δ  (largest N)   : -0.0082  at n=300
  Max Δ  (smallest N)  : +3.8051  at n=11
  Sign consistent      : NO — mixed sign

[TIMING] Total: 558.1s

[NOTE] To extend to N=1000 cleanly:
  Set N_POOL = 4000 and add 1000 to N_VALUES.
  Pool generation will take approximately 2× longer.
  At N=1000 with pool=4000 each seed draws 25% of pool — clean.