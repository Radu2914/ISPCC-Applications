═══════════════════════════════════════════════════python csp_v2_deploy.py --model csp_model_v2.pkl --bearing "C:/Users/Radu/Desktop/ISPCC/_files/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set/Bearing1_3" --every 50
[CSP v2 DEPLOY]  Loading csp_model_v2.pkl...
  Thresholds: cascade<0.250  critical<0.080

════════════════════════════════════════════════════════════════════════
  CSP LIVE DEPLOYMENT  —  Bearing1_3
  Model : csp_model_v2.pkl
  Life  : 395.8 min estimated  (from snapshot count)
  Thresholds (rul_norm):  EARLY<0.50  CASCADE<0.25  CRITICAL<0.08
════════════════════════════════════════════════════════════════════════
    Snap     Time  Regime       RUL(min)   RMS(g)     Kurt   Temp(°C)  Signal
  ──────  ───────  ──────────  ─────────  ───────  ───────  ─────────  ────────
       0     0.0m    ●  HEALTHY         395.8   0.4156     3.07       25.0  e-regul.
      50     8.3m    ●  HEALTHY         395.8   0.3886     3.26       25.0  e-regul.
     100    16.7m    ●  HEALTHY         395.8   0.3781     2.98       25.0  e-regul.
     150    25.0m    ●  HEALTHY         395.8   0.4111     3.37       25.0  e-regul.
     200    33.3m    ●  HEALTHY         395.8   0.3812     3.10       25.0  e-regul.
     250    41.7m    ●  HEALTHY         395.8   0.3971     3.51       25.0  e-regul.
     300    50.0m    ●  HEALTHY         395.8   0.4061     3.38       25.0  e-regul.
     350    58.3m    ●  HEALTHY         395.8   0.3989     3.70       25.0  e-regul.
     400    66.7m    ●  HEALTHY         395.8   0.3704     3.35       25.0  e-regul.
     450    75.0m    ●  HEALTHY         395.8   0.3718     3.37       25.0  e-regul.
     500    83.3m    ●  HEALTHY         395.8   0.3529     2.90       25.0  e-regul.
     550    91.7m    ●  HEALTHY         384.4   0.4097     3.31       25.0  e-regul.
     600   100.0m    ●  HEALTHY         368.0   0.3748     3.00       25.0  e-regul.
     650   108.3m    ●  HEALTHY         350.9   0.3515     3.05       25.0  e-regul.
     700   116.7m    ●  HEALTHY         326.9   0.3647     3.55       25.0  e-regul.
     750   125.0m    ●  HEALTHY         317.6   0.3603     3.56       25.0  e-regul.
     800   133.3m    ●  HEALTHY         318.5   0.3267     3.19       25.0  e-regul.
     850   141.7m    ●  HEALTHY         294.1   0.3415     3.41       25.0  e-regul.
     900   150.0m    ●  HEALTHY         249.7   0.3849     6.49       25.0  e-regul.
     950   158.3m    ●  HEALTHY         266.9   0.3564     3.61       25.0  e-regul.
    1000   166.7m    ●  HEALTHY         235.3   0.3480     8.54       25.0  e-regul.
    1050   175.0m    ●  HEALTHY         247.0   0.3560     3.16       25.0  e-regul.
    1100   183.3m    ●  HEALTHY         214.3   0.4488    11.94       25.0  e-regul.
    1150   191.7m    ●  HEALTHY         207.2   0.4166     3.40       25.0  e-regul.
    1200   200.0m    ●  HEALTHY         201.7   0.3624     3.33       25.0  e-regul.
    1227   204.5m   ◐   EARLY           196.7   0.4000     4.39       25.0  π/e mixed
    1250   208.3m   ◐   EARLY           182.5   0.4098    16.78       25.0  π/e mixed
    1300   216.7m   ◐   EARLY           189.0   0.4137     4.17       25.0  π/e mixed
    1350   225.0m   ◐   EARLY           174.1   0.4480     3.09       25.0  π/e mixed
    1400   233.3m   ◐   EARLY           159.8   0.4711     4.39       25.0  π/e mixed
    1450   241.7m   ◐   EARLY           152.6   0.5159     3.34       25.0  π/e mixed
    1500   250.0m   ◐   EARLY           141.4   0.5199     3.30       25.0  π/e mixed
    1550   258.3m   ◐   EARLY           139.4   0.5797     3.65       25.0  π/e mixed
    1600   266.7m   ◐   EARLY           129.3   0.6017     3.51       25.0  π/e mixed
    1650   275.0m   ◐   EARLY           119.7   0.7867   172.77       25.0  π/e mixed
    1700   283.3m   ◐   EARLY           109.6   1.7079    81.63       25.0  π/e mixed
    1750   291.7m   ◐   EARLY           105.3   2.2544    45.53       25.0  π/e mixed
    1756   292.7m  *** ⚠    CASCADE          98.6   0.8122    39.55       25.0  π-cascade
    1757   292.8m   ◐   EARLY           106.3   0.9058    46.86       25.0  π/e mixed
    1768   294.7m  *** ⚠    CASCADE          92.5   4.3608     8.23       25.0  π-cascade
    1769   294.8m  *** ⚠    CASCADE          68.0   5.8465     5.08       25.0  π-cascade ◄ CASCADE CONFIRMED

  ────────────────────────────────────────────────────────────────────────
  ⚠  CASCADE STATE DETECTED  —  CSP structural diagnosis
  ────────────────────────────────────────────────────────────────────────
     Snapshot          : 1769  (+294.8 min elapsed)
     Est. RUL          : 68.0 minutes
     Lead time         : 68.0 min before predicted failure

     ENCODED FEATURE STATE (π/e structural typing):
       RMS encoded     : xn=0.292  (raw 5.846g = 15.0× baseline)  → e-regime
       Kurtosis encoded: xn=0.169  (raw 5.08)  → π-active
       Life fraction   : 0.745  (75% of expected life elapsed)  → π-active
       Temperature     : 25.0°C  (dev +0.0°C)  →  self-regulating  ✓  (e-type boundary intact)

     DIAGNOSIS         : π-dominant cascade confirmed (sensor spike + encoded signal)
     Recommended       : schedule maintenance within 58 min
  ────────────────────────────────────────────────────────────────────────

    1773   295.5m   ◐   EARLY           100.0   3.6013    15.01       25.0  π/e mixed
    1778   296.3m  *** ⚠    CASCADE          93.4   1.5421    94.37       25.0  π-cascade
    1800   300.0m  *** ⚠    CASCADE          93.9   4.2010     8.36       25.0  π-cascade
    1850   308.3m  *** ⚠    CASCADE          83.9   0.8431   157.08       25.0  π-cascade
    1900   316.7m  *** ⚠    CASCADE          78.4   1.6081    26.63       25.0  π-cascade
    1950   325.0m  *** ⚠    CASCADE          60.9   4.7169     9.31       25.0  π-cascade
    2000   333.3m  *** ⚠    CASCADE          57.8   3.5677    20.82       25.0  π-cascade
    2050   341.7m  *** ⚠    CASCADE          52.4   1.8254    67.75       25.0  π-cascade
    2100   350.0m  *** ⚠    CASCADE          44.2   4.0991    18.66       25.0  π-cascade
    2150   358.3m  *** ⚠    CASCADE          35.6   1.8242   153.02       25.0  π-cascade
    2185   364.2m  !!! 🔴   CRITICAL         30.1   1.7047    12.03       25.0  π-cascade
    2191   365.2m  *** ⚠    CASCADE          32.0   1.4438    92.61       25.0  π-cascade
    2192   365.3m  !!! 🔴   CRITICAL         30.8   2.6584    34.08       25.0  π-cascade
    2200   366.7m  !!! 🔴   CRITICAL         25.9   1.7326    11.28       25.0  π-cascade
    2250   375.0m  !!! 🔴   CRITICAL         22.5   1.3806    99.91       25.0  π-cascade
    2300   383.3m  !!! 🔴   CRITICAL         13.3   2.1424     8.53       25.0  π-cascade
    2350   391.7m  !!! 🔴   CRITICAL          4.4   4.9058    10.78       25.0  π-cascade

════════════════════════════════════════════════════════════════════════
  CSP RUN SUMMARY  —  Bearing1_3
  Total bearing life : 395.8 min  (2375 snapshots  ×  10s each)
  Snapshots shown    : 58
  False alarms       : 2  (CASCADE → back to HEALTHY/EARLY transitions)

  CASCADE DETECTION RESULT:
    Detected at      : +294.8 min  (83% through bearing life)
    Lead time        : 68.0 min  = 17% of life remaining
    Action window    : 68.0 min to act before predicted failure

  CSP CLAIM VERIFIED: cascade state identified 68.0 min
  before failure with structural π-type diagnosis.
════════════════════════════════════════════════════════════════════════