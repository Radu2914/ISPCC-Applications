PS C:\Users\Radu\Desktop\ISPCC\5. Industry CSP\v1> python pronostia_3simm.py --data "C:/Users/Radu/Desktop/ISPCC/_files/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set" --n_intentional 2400

[INFO] Loading Full_Test_Set: C:/Users/Radu/Desktop/ISPCC/_files/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set
  Loading Bearing1_3 ... 2375 snapshots  life=395.8 min (10.0s)
  Loading Bearing1_4 ... 1428 snapshots  life=238.0 min (6.6s)
  Loading Bearing1_5 ... 2463 snapshots  life=410.5 min (11.1s)
  Loading Bearing1_6 ... 2448 snapshots  life=408.0 min (10.7s)
  Loading Bearing1_7 ... 2259 snapshots  life=376.5 min (9.6s)
  Loading Bearing2_3 ... 1955 snapshots  life=325.8 min (8.2s)
  Loading Bearing2_4 ...  751 snapshots  life=125.2 min (3.5s)
  Loading Bearing2_5 ... 2311 snapshots  life=385.2 min (10.0s)
  Loading Bearing2_6 ...  701 snapshots  life=116.8 min (3.2s)
  Loading Bearing2_7 ...  230 snapshots  life= 38.3 min (1.1s)
  Loading Bearing3_3 ...  434 snapshots  life= 72.3 min (2.0s)
[INFO] 11 bearings loaded


========================================================================
  PROBE — structural typing on Bearing1_3
  Target: RUL in seconds
========================================================================

  Top 15 features:
  Rank                       Feature     Imp%  Type
  ────  ────────────────────────────  ───────  ────
     1                pi_life_cos_pi   11.43%  π
     2               pi_life_sin_2pi   10.03%  π
     3               pi_rms_h_cos_pi    8.93%  π
     4               pi_rms_h_sin_pi    8.36%  π
     5              pi_rms_h_sin_2pi    6.22%  π
     6              cross_rms_x_temp    6.12%  ×
     7             pi_rms_bif_sin_pi    5.01%  π
     8             pi_rms_bif_cos_pi    4.72%  π
     9              cross_bif_x_temp    4.04%  ×
    10                pi_life_sin_pi    3.48%  π
    11            pi_rms_bif_sin_pi2    3.15%  π
    12              cross_life_x_rms    3.14%  ×
    13            pi_rms_bif_sin_2pi    2.87%  π
    14               pi_life_sin_pi2    2.70%  π
    15              pi_rms_h_sin_pi2    2.54%  π

  Grouped:
    π  : 86.5%  (cascade — expected dominant for RUL)
    e  : 0.2%  (self-regulating temperature)
    ×  : 13.3%  (regime interaction)
  Dominant: π  (478.4× over the other)

  Structural typing result:
  π dominant at 86.5% → vibration cascade is the
    primary RUL signal.  Confirms π-encoding is structurally correct.
  e contribution at 0.2% → temperature self-regulation
    carries complementary healthy-life information.

  Bifurcation constants confirmed as structural (not data-derived):
    RMS_BIFURCATION  = 0.5g    (vibration leaves equilibrium)
    TEMP_BIFURCATION = 5.0°C above ambient  (temperature loses self-regulation)
    FAILURE_G        = 20.0g    (test-stop threshold = normalising scale)

  Analogy to prior domains:
    RMS_BIFURCATION  ↔  R_BIFURCATION  (logistic map)
    TEMP_BIFURCATION ↔  DIEL_BIFURCATION (EM surrogate)
    FAILURE_G        ↔  LAMBDA_FREE (EM normalising scale)

========================================================================
  LEAVE-ONE-BEARING-OUT  |  MaxiMin budget = 2400  (min 6/bearing)
========================================================================
  Bearing         Snaps  Life(min)   RMSE(min)       R²       PHM
  --------------  -----  ---------  ----------  -------  --------

  Selecting for test=Bearing1_3:
      Bearing1_4: 230 snapshots selected (pool=1428)
      Bearing1_5: 391 snapshots selected (pool=2463)
      Bearing1_6: 389 snapshots selected (pool=2448)
      Bearing1_7: 359 snapshots selected (pool=2259)
      Bearing2_3: 312 snapshots selected (pool=1955)
      Bearing2_4: 124 snapshots selected (pool=751)
      Bearing2_5: 366 snapshots selected (pool=2311)
      Bearing2_6: 115 snapshots selected (pool=701)
      Bearing2_7: 41 snapshots selected (pool=230)
      Bearing3_3: 73 snapshots selected (pool=434)
  Bearing1_3       2375      395.8        40.6    0.874  770068.958

  Selecting for test=Bearing1_4:
      Bearing1_3: 355 snapshots selected (pool=2375)
      Bearing1_5: 368 snapshots selected (pool=2463)
      Bearing1_6: 366 snapshots selected (pool=2448)
      Bearing1_7: 338 snapshots selected (pool=2259)
      Bearing2_3: 294 snapshots selected (pool=1955)
      Bearing2_4: 117 snapshots selected (pool=751)
      Bearing2_5: 346 snapshots selected (pool=2311)
      Bearing2_6: 108 snapshots selected (pool=701)
      Bearing2_7: 39 snapshots selected (pool=230)
      Bearing3_3: 69 snapshots selected (pool=434)
  Bearing1_4       1428      238.0       137.3   -2.996  167345485.280

  Selecting for test=Bearing1_5:
      Bearing1_3: 380 snapshots selected (pool=2375)
      Bearing1_4: 231 snapshots selected (pool=1428)
      Bearing1_6: 391 snapshots selected (pool=2448)
      Bearing1_7: 360 snapshots selected (pool=2259)
      Bearing2_3: 313 snapshots selected (pool=1955)
      Bearing2_4: 124 snapshots selected (pool=751)
      Bearing2_5: 369 snapshots selected (pool=2311)
      Bearing2_6: 116 snapshots selected (pool=701)
      Bearing2_7: 42 snapshots selected (pool=230)
      Bearing3_3: 74 snapshots selected (pool=434)
  Bearing1_5       2463      410.5        10.9    0.991  196982.871

  Selecting for test=Bearing1_6:
      Bearing1_3: 379 snapshots selected (pool=2375)
      Bearing1_4: 231 snapshots selected (pool=1428)
      Bearing1_5: 393 snapshots selected (pool=2463)
      Bearing1_7: 361 snapshots selected (pool=2259)
      Bearing2_3: 313 snapshots selected (pool=1955)
      Bearing2_4: 123 snapshots selected (pool=751)
      Bearing2_5: 368 snapshots selected (pool=2311)
      Bearing2_6: 116 snapshots selected (pool=701)
      Bearing2_7: 42 snapshots selected (pool=230)
      Bearing3_3: 74 snapshots selected (pool=434)
  Bearing1_6       2448      408.0         9.8    0.993  198269.867

  Selecting for test=Bearing1_7:
      Bearing1_3: 375 snapshots selected (pool=2375)
      Bearing1_4: 228 snapshots selected (pool=1428)
      Bearing1_5: 388 snapshots selected (pool=2463)
      Bearing1_6: 386 snapshots selected (pool=2448)
      Bearing2_3: 309 snapshots selected (pool=1955)
      Bearing2_4: 122 snapshots selected (pool=751)
      Bearing2_5: 364 snapshots selected (pool=2311)
      Bearing2_6: 114 snapshots selected (pool=701)
      Bearing2_7: 41 snapshots selected (pool=230)
      Bearing3_3: 73 snapshots selected (pool=434)
  Bearing1_7       2259      376.5        14.4    0.982  217964.406

  Selecting for test=Bearing2_3:
      Bearing1_3: 367 snapshots selected (pool=2375)
      Bearing1_4: 223 snapshots selected (pool=1428)
      Bearing1_5: 381 snapshots selected (pool=2463)
      Bearing1_6: 378 snapshots selected (pool=2448)
      Bearing1_7: 350 snapshots selected (pool=2259)
      Bearing2_4: 121 snapshots selected (pool=751)
      Bearing2_5: 357 snapshots selected (pool=2311)
      Bearing2_6: 112 snapshots selected (pool=701)
      Bearing2_7: 40 snapshots selected (pool=230)
      Bearing3_3: 71 snapshots selected (pool=434)
  Bearing2_3       1955      325.8        45.0    0.771  11943994.289

  Selecting for test=Bearing2_4:
      Bearing1_3: 341 snapshots selected (pool=2375)
      Bearing1_4: 208 snapshots selected (pool=1428)
      Bearing1_5: 354 snapshots selected (pool=2463)
      Bearing1_6: 351 snapshots selected (pool=2448)
      Bearing1_7: 325 snapshots selected (pool=2259)
      Bearing2_3: 281 snapshots selected (pool=1955)
      Bearing2_5: 331 snapshots selected (pool=2311)
      Bearing2_6: 104 snapshots selected (pool=701)
      Bearing2_7: 38 snapshots selected (pool=230)
      Bearing3_3: 67 snapshots selected (pool=434)
  Bearing2_4        751      125.2         8.4    0.946    11.189

  Selecting for test=Bearing2_5:
      Bearing1_3: 376 snapshots selected (pool=2375)
      Bearing1_4: 229 snapshots selected (pool=1428)
      Bearing1_5: 390 snapshots selected (pool=2463)
      Bearing1_6: 387 snapshots selected (pool=2448)
      Bearing1_7: 357 snapshots selected (pool=2259)
      Bearing2_3: 310 snapshots selected (pool=1955)
      Bearing2_4: 122 snapshots selected (pool=751)
      Bearing2_6: 115 snapshots selected (pool=701)
      Bearing2_7: 41 snapshots selected (pool=230)
      Bearing3_3: 73 snapshots selected (pool=434)
  Bearing2_5       2311      385.2        29.9    0.927  936460.847

  Selecting for test=Bearing2_6:
      Bearing1_3: 340 snapshots selected (pool=2375)
      Bearing1_4: 207 snapshots selected (pool=1428)
      Bearing1_5: 353 snapshots selected (pool=2463)
      Bearing1_6: 350 snapshots selected (pool=2448)
      Bearing1_7: 324 snapshots selected (pool=2259)
      Bearing2_3: 281 snapshots selected (pool=1955)
      Bearing2_4: 111 snapshots selected (pool=751)
      Bearing2_5: 330 snapshots selected (pool=2311)
      Bearing2_7: 38 snapshots selected (pool=230)
      Bearing3_3: 66 snapshots selected (pool=434)
  Bearing2_6        701      116.8        35.1   -0.084  748187.693

  Selecting for test=Bearing2_7:
      Bearing1_3: 331 snapshots selected (pool=2375)
      Bearing1_4: 202 snapshots selected (pool=1428)
      Bearing1_5: 343 snapshots selected (pool=2463)
      Bearing1_6: 341 snapshots selected (pool=2448)
      Bearing1_7: 315 snapshots selected (pool=2259)
      Bearing2_3: 273 snapshots selected (pool=1955)
      Bearing2_4: 108 snapshots selected (pool=751)
      Bearing2_5: 321 snapshots selected (pool=2311)
      Bearing2_6: 101 snapshots selected (pool=701)
      Bearing3_3: 65 snapshots selected (pool=434)
  Bearing2_7        230       38.3         5.6    0.742  364685.703

  Selecting for test=Bearing3_3:
      Bearing1_3: 335 snapshots selected (pool=2375)
      Bearing1_4: 204 snapshots selected (pool=1428)
      Bearing1_5: 347 snapshots selected (pool=2463)
      Bearing1_6: 345 snapshots selected (pool=2448)
      Bearing1_7: 319 snapshots selected (pool=2259)
      Bearing2_3: 277 snapshots selected (pool=1955)
      Bearing2_4: 109 snapshots selected (pool=751)
      Bearing2_5: 325 snapshots selected (pool=2311)
      Bearing2_6: 102 snapshots selected (pool=701)
      Bearing2_7: 37 snapshots selected (pool=230)
  Bearing3_3        434       72.3         2.5    0.986     8.680

  --------------  -----  ---------  ----------  -------  --------
  MEAN             1578      263.0        30.9    0.467  16611101.798

  MaxiMin N = 2400  (min 6/bearing × 10 training bearings)
========================================================================

[INFO] Results saved to lobo_results.csv
[TIMING] Total: 91.0s