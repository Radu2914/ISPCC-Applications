python csp_v2_deploy.py --model csp_model_v2.pkl --bearing "C:/Users/Radu/Desktop/ISPCC/_files/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set/Bearing3_3" --every 50
[CSP v2 DEPLOY]  Loading csp_model_v2.pkl...
  Thresholds: cascade<0.250  critical<0.080

════════════════════════════════════════════════════════════════════════
  CSP LIVE DEPLOYMENT  —  Bearing3_3
  Model : csp_model_v2.pkl
  Life  : 72.3 min estimated  (from snapshot count)
  Thresholds (rul_norm):  EARLY<0.50  CASCADE<0.25  CRITICAL<0.08
════════════════════════════════════════════════════════════════════════
    Snap     Time  Regime       RUL(min)   RMS(g)     Kurt   Temp(°C)  Signal
  ──────  ───────  ──────────  ─────────  ───────  ───────  ─────────  ────────
       0     0.0m    ●  HEALTHY          71.2   0.2970     3.35       67.1  e-regul.
      50     8.3m    ●  HEALTHY          69.9   0.3334     3.75      114.7  e-regul.
     100    16.7m    ●  HEALTHY          72.3   0.4425     4.89       25.0  e-regul.
     150    25.0m    ●  HEALTHY          62.1   0.3907     3.40       25.0  e-regul.
     200    33.3m    ●  HEALTHY          49.9   0.3698     3.61       25.0  e-regul.
     250    41.7m    ●  HEALTHY          39.0   0.3471     3.18       25.0  e-regul.
     261    43.5m   ◐   EARLY            36.1   0.3623     3.32       25.0  π/e mixed
     300    50.0m   ◐   EARLY            26.5   0.3583     4.37       25.0  π/e mixed
     350    58.3m   ◐   EARLY            17.8   0.8152     3.59       25.0  π/e mixed
     352    58.7m  *** ⚠    CASCADE          17.5   0.8515     3.90       25.0  π-cascade
     353    58.8m  *** ⚠    CASCADE          17.5   0.9340     3.76       25.0  π-cascade ◄ CASCADE CONFIRMED

  ────────────────────────────────────────────────────────────────────────
  ⚠  CASCADE STATE DETECTED  —  CSP structural diagnosis
  ────────────────────────────────────────────────────────────────────────
     Snapshot          : 353  (+58.8 min elapsed)
     Est. RUL          : 17.5 minutes
     Lead time         : 17.5 min before predicted failure

     STRUCTURAL EVIDENCE:
       RMS envelope    : 0.9340g  (2.9× healthy baseline)  → π-cascade  ✗
       Kurtosis        : 3.76  (1.1× healthy baseline)  → bounded  ✓
       Temperature     : 25.0°C  (dev +0.0°C)  →  self-regulating  ✓  (e-type boundary intact)

     DIAGNOSIS         : π-type emerging, monitoring recommended
     Recommended       : schedule maintenance within 7 min
  ────────────────────────────────────────────────────────────────────────

     354    59.0m  *** ⚠    CASCADE          17.2   0.9529     3.63       25.0  π-cascade ◄ CASCADE CONFIRMED
     355    59.2m  *** ⚠    CASCADE          17.0   0.8611     3.85       25.0  π-cascade ◄ CASCADE CONFIRMED
     356    59.3m  *** ⚠    CASCADE          17.2   0.9041     3.48       25.0  π-cascade ◄ CASCADE CONFIRMED
     357    59.5m  *** ⚠    CASCADE          16.9   0.9073     3.64       25.0  π-cascade ◄ CASCADE CONFIRMED
     358    59.7m  *** ⚠    CASCADE          17.0   0.9292     3.50       25.0  π-cascade ◄ CASCADE CONFIRMED
     359    59.8m  *** ⚠    CASCADE          16.9   0.9658     3.74       25.0  π-cascade ◄ CASCADE CONFIRMED
     360    60.0m  *** ⚠    CASCADE          16.8   0.9990     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     361    60.2m  *** ⚠    CASCADE          16.4   0.9576     3.63       25.0  π-cascade ◄ CASCADE CONFIRMED
     362    60.3m  *** ⚠    CASCADE          16.3   0.9049     3.79       25.0  π-cascade ◄ CASCADE CONFIRMED
     363    60.5m  *** ⚠    CASCADE          16.2   0.9435     3.51       25.0  π-cascade ◄ CASCADE CONFIRMED
     364    60.7m  *** ⚠    CASCADE          16.0   0.9285     3.86       25.0  π-cascade ◄ CASCADE CONFIRMED
     366    61.0m  *** ⚠    CASCADE          15.3   0.9328     3.62       25.0  π-cascade ◄ CASCADE CONFIRMED
     367    61.2m  *** ⚠    CASCADE          15.7   0.9815     3.58       25.0  π-cascade ◄ CASCADE CONFIRMED
     369    61.5m  *** ⚠    CASCADE          15.4   0.9858     3.59       25.0  π-cascade ◄ CASCADE CONFIRMED
     370    61.7m  *** ⚠    CASCADE          15.4   0.9920     3.39       25.0  π-cascade ◄ CASCADE CONFIRMED
     371    61.8m  *** ⚠    CASCADE          15.3   1.0467     3.38       25.0  π-cascade ◄ CASCADE CONFIRMED
     372    62.0m  *** ⚠    CASCADE          15.1   1.0281     3.57       25.0  π-cascade ◄ CASCADE CONFIRMED
     373    62.2m  *** ⚠    CASCADE          15.0   0.9841     3.19       25.0  π-cascade ◄ CASCADE CONFIRMED
     374    62.3m  *** ⚠    CASCADE          14.7   0.9941     3.30       25.0  π-cascade ◄ CASCADE CONFIRMED
     375    62.5m  *** ⚠    CASCADE          14.8   0.9999     3.11       25.0  π-cascade ◄ CASCADE CONFIRMED
     376    62.7m  *** ⚠    CASCADE          14.4   1.0614     3.41       25.0  π-cascade ◄ CASCADE CONFIRMED
     377    62.8m  *** ⚠    CASCADE          14.4   1.0579     3.28       25.0  π-cascade ◄ CASCADE CONFIRMED
     378    63.0m  *** ⚠    CASCADE          14.3   1.0161     3.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     379    63.2m  *** ⚠    CASCADE          14.1   1.0604     3.43       25.0  π-cascade ◄ CASCADE CONFIRMED
     380    63.3m  *** ⚠    CASCADE          13.8   1.0684     3.44       25.0  π-cascade ◄ CASCADE CONFIRMED
     381    63.5m  *** ⚠    CASCADE          13.8   1.0580     3.33       25.0  π-cascade ◄ CASCADE CONFIRMED
     382    63.7m  *** ⚠    CASCADE          13.7   1.0831     3.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     383    63.8m  *** ⚠    CASCADE          13.6   1.0789     3.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     384    64.0m  *** ⚠    CASCADE          13.3   0.9738     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     385    64.2m  *** ⚠    CASCADE          13.1   1.1321     3.51       25.0  π-cascade ◄ CASCADE CONFIRMED
     386    64.3m  *** ⚠    CASCADE          13.1   1.0673     3.15       25.0  π-cascade ◄ CASCADE CONFIRMED
     387    64.5m  *** ⚠    CASCADE          12.8   1.1540     3.51       25.0  π-cascade ◄ CASCADE CONFIRMED
     388    64.7m  *** ⚠    CASCADE          13.0   1.0196     3.09       25.0  π-cascade ◄ CASCADE CONFIRMED
     389    64.8m  *** ⚠    CASCADE          12.7   1.1162     3.27       25.0  π-cascade ◄ CASCADE CONFIRMED
     390    65.0m  *** ⚠    CASCADE          12.4   1.0841     3.50       25.0  π-cascade ◄ CASCADE CONFIRMED
     391    65.2m  *** ⚠    CASCADE          12.5   1.0923     3.20       25.0  π-cascade ◄ CASCADE CONFIRMED
     392    65.3m  *** ⚠    CASCADE          12.3   1.1155     3.08       25.0  π-cascade ◄ CASCADE CONFIRMED
     393    65.5m  *** ⚠    CASCADE          11.8   1.1178     3.31       25.0  π-cascade ◄ CASCADE CONFIRMED
     394    65.7m  *** ⚠    CASCADE          11.3   1.1803     3.52       25.0  π-cascade ◄ CASCADE CONFIRMED
     395    65.8m  *** ⚠    CASCADE          11.3   1.0760     3.40       25.0  π-cascade ◄ CASCADE CONFIRMED
     396    66.0m  *** ⚠    CASCADE          11.1   1.1930     3.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     397    66.2m  *** ⚠    CASCADE          11.0   1.1562     3.18       25.0  π-cascade ◄ CASCADE CONFIRMED
     398    66.3m  *** ⚠    CASCADE          10.7   1.1692     3.13       25.0  π-cascade ◄ CASCADE CONFIRMED
     399    66.5m  *** ⚠    CASCADE          10.7   1.1144     3.25       25.0  π-cascade ◄ CASCADE CONFIRMED
     400    66.7m  *** ⚠    CASCADE           9.9   1.1107     3.50       25.0  π-cascade ◄ CASCADE CONFIRMED
     401    66.8m  *** ⚠    CASCADE           9.5   1.1559     3.52       25.0  π-cascade ◄ CASCADE CONFIRMED
     402    67.0m  *** ⚠    CASCADE           9.4   1.2223     3.37       25.0  π-cascade ◄ CASCADE CONFIRMED
     403    67.2m  *** ⚠    CASCADE           8.7   1.0798     3.66       25.0  π-cascade ◄ CASCADE CONFIRMED
     404    67.3m  *** ⚠    CASCADE           8.6   1.1793     3.72       25.0  π-cascade ◄ CASCADE CONFIRMED
     405    67.5m  *** ⚠    CASCADE           8.2   1.2335     3.55       25.0  π-cascade ◄ CASCADE CONFIRMED
     406    67.7m  *** ⚠    CASCADE           8.5   1.1773     3.15       25.0  π-cascade ◄ CASCADE CONFIRMED
     407    67.8m  *** ⚠    CASCADE           8.2   1.1922     3.47       25.0  π-cascade ◄ CASCADE CONFIRMED
     408    68.0m  *** ⚠    CASCADE           8.2   1.1312     3.38       25.0  π-cascade ◄ CASCADE CONFIRMED
     409    68.2m  *** ⚠    CASCADE           7.7   1.2228     3.34       25.0  π-cascade ◄ CASCADE CONFIRMED
     410    68.3m  *** ⚠    CASCADE           7.5   1.1636     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     411    68.5m  *** ⚠    CASCADE           7.5   1.1733     3.35       25.0  π-cascade ◄ CASCADE CONFIRMED
     412    68.7m  *** ⚠    CASCADE           7.3   1.1882     3.39       25.0  π-cascade ◄ CASCADE CONFIRMED
     413    68.8m  *** ⚠    CASCADE           7.4   1.1864     3.23       25.0  π-cascade ◄ CASCADE CONFIRMED
     414    69.0m  *** ⚠    CASCADE           7.1   1.1242     3.45       25.0  π-cascade ◄ CASCADE CONFIRMED
     415    69.2m  *** ⚠    CASCADE           6.9   1.2012     3.29       25.0  π-cascade ◄ CASCADE CONFIRMED
     416    69.3m  *** ⚠    CASCADE           6.8   1.1489     3.17       25.0  π-cascade ◄ CASCADE CONFIRMED
     417    69.5m  *** ⚠    CASCADE           6.8   1.2433     3.21       25.0  π-cascade ◄ CASCADE CONFIRMED
     418    69.7m  *** ⚠    CASCADE           6.6   1.2248     3.27       25.0  π-cascade ◄ CASCADE CONFIRMED
     419    69.8m  *** ⚠    CASCADE           6.3   1.1457     3.30       25.0  π-cascade ◄ CASCADE CONFIRMED
     420    70.0m  *** ⚠    CASCADE           6.1   1.2191     3.07       25.0  π-cascade ◄ CASCADE CONFIRMED
     421    70.2m  *** ⚠    CASCADE           5.9   1.2025     2.98       25.0  π-cascade ◄ CASCADE CONFIRMED
     422    70.3m  *** ⚠    CASCADE           5.0   1.2264     3.91       25.0  π-cascade ◄ CASCADE CONFIRMED
     423    70.5m  *** ⚠    CASCADE           4.4   1.3699     3.94       25.0  π-cascade ◄ CASCADE CONFIRMED
     424    70.7m  !!! 🔴   CRITICAL          4.1   1.3781     4.75       25.0  π-cascade ◄ CASCADE CONFIRMED
     425    70.8m  !!! 🔴   CRITICAL          3.4   1.7418     6.13       25.0  π-cascade ◄ CASCADE CONFIRMED
     426    71.0m  !!! 🔴   CRITICAL          3.2   1.7672     7.94       25.0  π-cascade ◄ CASCADE CONFIRMED
     427    71.2m  !!! 🔴   CRITICAL          3.1   1.7876     7.39       25.0  π-cascade ◄ CASCADE CONFIRMED
     428    71.3m  !!! 🔴   CRITICAL          2.9   1.8095     7.74       25.0  π-cascade ◄ CASCADE CONFIRMED
     429    71.5m  !!! 🔴   CRITICAL          2.7   1.7985     8.38       25.0  π-cascade ◄ CASCADE CONFIRMED
     430    71.7m  !!! 🔴   CRITICAL          2.6   1.8165     7.77       25.0  π-cascade ◄ CASCADE CONFIRMED
     431    71.8m  !!! 🔴   CRITICAL          2.4   1.8809     6.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     432    72.0m  !!! 🔴   CRITICAL          1.8   2.0593     7.68       25.0  π-cascade ◄ CASCADE CONFIRMED
     433    72.2m  !!! 🔴   CRITICAL          0.9   2.2271     7.43       25.0  π-cascade ◄ CASCADE CONFIRMED

════════════════════════════════════════════════════════════════════════
  CSP RUN SUMMARY  —  Bearing3_3
  Total bearing life : 72.3 min  (434 snapshots  ×  10s each)
  Snapshots shown    : 89
  False alarms       : 0  (CASCADE → back to HEALTHY/EARLY transitions)

  CASCADE DETECTION RESULT:
    Detected at      : +58.8 min  (76% through bearing life)
    Lead time        : 17.5 min  = 24% of life remaining
    Action window    : 17.5 min to act before predicted failure

  CSP CLAIM VERIFIED: cascade state identified 17.5 min
  before failure with structural π-type diagnosis.
════════════════════════════════════════════════════════════════════════