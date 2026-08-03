python csp_v2_deploy.py --model csp_model_v2.pkl --bearing "C:/Users/Radu/Desktop/ISPCC/_files/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set/Bearing2_4" --every 50
[CSP v2 DEPLOY]  Loading csp_model_v2.pkl...
  Thresholds: cascade<0.250  critical<0.080

════════════════════════════════════════════════════════════════════════
  CSP LIVE DEPLOYMENT  —  Bearing2_4
  Model : csp_model_v2.pkl
  Life  : 125.2 min estimated  (from snapshot count)
  Thresholds (rul_norm):  EARLY<0.50  CASCADE<0.25  CRITICAL<0.08
════════════════════════════════════════════════════════════════════════
    Snap     Time  Regime       RUL(min)   RMS(g)     Kurt   Temp(°C)  Signal
  ──────  ───────  ──────────  ─────────  ───────  ───────  ─────────  ────────
       0     0.0m    ●  HEALTHY         123.7   0.3413     3.03       63.8  e-regul.
      50     8.3m    ●  HEALTHY         112.5   0.3612     3.13       98.4  e-regul.
     100    16.7m    ●  HEALTHY         102.9   0.3778     3.04      104.9  e-regul.
     150    25.0m    ●  HEALTHY         118.0   0.3751     3.10       25.0  e-regul.
     200    33.3m    ●  HEALTHY         104.4   0.3489     3.10       25.0  e-regul.
     250    41.7m    ●  HEALTHY          92.8   0.3529     3.32       25.0  e-regul.
     300    50.0m    ●  HEALTHY          85.3   0.3343     3.22       25.0  e-regul.
     350    58.3m    ●  HEALTHY          68.5   0.3554     5.86       25.0  e-regul.
     384    64.0m   ◐   EARLY            61.8   0.3442     7.10       25.0  π/e mixed
     385    64.2m    ●  HEALTHY          64.8   0.3252     4.43       25.0  e-regul.
     388    64.7m   ◐   EARLY            60.6   0.4044    12.39       25.0  π/e mixed
     400    66.7m   ◐   EARLY            59.8   0.3392     5.23       25.0  π/e mixed
     450    75.0m   ◐   EARLY            52.7   0.2983     3.14       25.0  π/e mixed
     500    83.3m   ◐   EARLY            43.3   0.3156     3.48       25.0  π/e mixed
     550    91.7m   ◐   EARLY            33.3   0.3117     3.03       25.0  π/e mixed
     569    94.8m  *** ⚠    CASCADE          30.5   0.3211     3.17       25.0  π-cascade
     570    95.0m  *** ⚠    CASCADE          30.3   0.3239     3.16       25.0  π-cascade ◄ CASCADE CONFIRMED

  ────────────────────────────────────────────────────────────────────────
  ⚠  CASCADE STATE DETECTED  —  CSP structural diagnosis
  ────────────────────────────────────────────────────────────────────────
     Snapshot          : 570  (+95.0 min elapsed)
     Est. RUL          : 30.3 minutes
     Lead time         : 30.3 min before predicted failure

     STRUCTURAL EVIDENCE:
       RMS envelope    : 0.3239g  (0.9× healthy baseline)  → bounded  ✓
       Kurtosis        : 3.16  (1.0× healthy baseline)  → bounded  ✓
       Temperature     : 25.0°C  (dev +0.0°C)  →  self-regulating  ✓  (e-type boundary intact)

     DIAGNOSIS         : π-type emerging, monitoring recommended
     Recommended       : schedule maintenance within 20 min
  ────────────────────────────────────────────────────────────────────────

     573    95.5m  *** ⚠    CASCADE          30.2   0.2848     3.11       25.0  π-cascade ◄ CASCADE CONFIRMED
     574    95.7m  *** ⚠    CASCADE          29.7   0.3118     3.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     576    96.0m  *** ⚠    CASCADE          29.5   0.3111     3.11       25.0  π-cascade ◄ CASCADE CONFIRMED
     577    96.2m  *** ⚠    CASCADE          29.3   0.3054     3.19       25.0  π-cascade ◄ CASCADE CONFIRMED
     578    96.3m  *** ⚠    CASCADE          28.7   0.2981     3.34       25.0  π-cascade ◄ CASCADE CONFIRMED
     579    96.5m  *** ⚠    CASCADE          29.0   0.2991     3.23       25.0  π-cascade ◄ CASCADE CONFIRMED
     580    96.7m  *** ⚠    CASCADE          28.4   0.2844     3.15       25.0  π-cascade ◄ CASCADE CONFIRMED
     581    96.8m  *** ⚠    CASCADE          28.3   0.2718     3.21       25.0  π-cascade ◄ CASCADE CONFIRMED
     584    97.3m  *** ⚠    CASCADE          27.7   0.2971     4.20       25.0  π-cascade ◄ CASCADE CONFIRMED
     585    97.5m  *** ⚠    CASCADE          27.7   0.3226     3.54       25.0  π-cascade ◄ CASCADE CONFIRMED
     587    97.8m  *** ⚠    CASCADE          27.4   0.3117     3.35       25.0  π-cascade ◄ CASCADE CONFIRMED
     588    98.0m  *** ⚠    CASCADE          27.4   0.3008     3.13       25.0  π-cascade ◄ CASCADE CONFIRMED
     589    98.2m  *** ⚠    CASCADE          27.1   0.3165     3.24       25.0  π-cascade ◄ CASCADE CONFIRMED
     591    98.5m  *** ⚠    CASCADE          26.9   0.2819     3.05       25.0  π-cascade ◄ CASCADE CONFIRMED
     592    98.7m  *** ⚠    CASCADE          26.3   0.3062     2.99       25.0  π-cascade ◄ CASCADE CONFIRMED
     593    98.8m  *** ⚠    CASCADE          25.8   0.3138     3.94       25.0  π-cascade ◄ CASCADE CONFIRMED
     596    99.3m  *** ⚠    CASCADE          26.0   0.2901     3.10       25.0  π-cascade ◄ CASCADE CONFIRMED
     599    99.8m  *** ⚠    CASCADE          25.8   0.2989     3.05       25.0  π-cascade ◄ CASCADE CONFIRMED
     600   100.0m  *** ⚠    CASCADE          25.7   0.2811     3.08       25.0  π-cascade ◄ CASCADE CONFIRMED
     601   100.2m  *** ⚠    CASCADE          25.2   0.3037     3.05       25.0  π-cascade ◄ CASCADE CONFIRMED
     602   100.3m  *** ⚠    CASCADE          24.5   0.3244     3.84       25.0  π-cascade ◄ CASCADE CONFIRMED
     603   100.5m  *** ⚠    CASCADE          24.7   0.3055     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     604   100.7m  *** ⚠    CASCADE          24.2   0.3150     4.02       25.0  π-cascade ◄ CASCADE CONFIRMED
     605   100.8m  *** ⚠    CASCADE          23.9   0.2973     3.79       25.0  π-cascade ◄ CASCADE CONFIRMED
     606   101.0m  *** ⚠    CASCADE          23.3   0.3285     4.07       25.0  π-cascade ◄ CASCADE CONFIRMED
     607   101.2m  *** ⚠    CASCADE          21.9   0.3657     6.02       25.0  π-cascade ◄ CASCADE CONFIRMED
     608   101.3m  *** ⚠    CASCADE          20.2   0.4183     9.67       25.0  π-cascade ◄ CASCADE CONFIRMED
     611   101.8m  *** ⚠    CASCADE          22.4   0.3236     4.42       25.0  π-cascade ◄ CASCADE CONFIRMED
     612   102.0m  *** ⚠    CASCADE          22.5   0.3330     3.85       25.0  π-cascade ◄ CASCADE CONFIRMED
     615   102.5m  *** ⚠    CASCADE          22.7   0.3137     3.35       25.0  π-cascade ◄ CASCADE CONFIRMED
     616   102.7m  *** ⚠    CASCADE          22.7   0.3061     3.36       25.0  π-cascade ◄ CASCADE CONFIRMED
     619   103.2m  *** ⚠    CASCADE          22.5   0.3031     3.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     620   103.3m  *** ⚠    CASCADE          22.7   0.2879     3.08       25.0  π-cascade ◄ CASCADE CONFIRMED
     621   103.5m  *** ⚠    CASCADE          22.2   0.2893     3.34       25.0  π-cascade ◄ CASCADE CONFIRMED
     622   103.7m  *** ⚠    CASCADE          22.0   0.3109     3.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     625   104.2m  *** ⚠    CASCADE          22.1   0.2832     2.99       25.0  π-cascade ◄ CASCADE CONFIRMED
     627   104.5m  *** ⚠    CASCADE          21.5   0.3044     2.96       25.0  π-cascade ◄ CASCADE CONFIRMED
     628   104.7m  *** ⚠    CASCADE          21.2   0.2979     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     629   104.8m  *** ⚠    CASCADE          21.0   0.2966     3.25       25.0  π-cascade ◄ CASCADE CONFIRMED
     630   105.0m  *** ⚠    CASCADE          21.1   0.2803     3.30       25.0  π-cascade ◄ CASCADE CONFIRMED
     631   105.2m  *** ⚠    CASCADE          20.8   0.2736     3.42       25.0  π-cascade ◄ CASCADE CONFIRMED
     632   105.3m  *** ⚠    CASCADE          20.8   0.2736     3.17       25.0  π-cascade ◄ CASCADE CONFIRMED
     633   105.5m  *** ⚠    CASCADE          20.7   0.2881     3.29       25.0  π-cascade ◄ CASCADE CONFIRMED
     634   105.7m  *** ⚠    CASCADE          20.6   0.2730     3.34       25.0  π-cascade ◄ CASCADE CONFIRMED
     635   105.8m  *** ⚠    CASCADE          20.2   0.2700     3.56       25.0  π-cascade ◄ CASCADE CONFIRMED
     636   106.0m  *** ⚠    CASCADE          20.4   0.2830     3.24       25.0  π-cascade ◄ CASCADE CONFIRMED
     638   106.3m  *** ⚠    CASCADE          20.3   0.2925     3.10       25.0  π-cascade ◄ CASCADE CONFIRMED
     639   106.5m  *** ⚠    CASCADE          20.3   0.3184     3.07       25.0  π-cascade ◄ CASCADE CONFIRMED
     640   106.7m  *** ⚠    CASCADE          20.1   0.2963     3.02       25.0  π-cascade ◄ CASCADE CONFIRMED
     641   106.8m  *** ⚠    CASCADE          19.8   0.2779     3.03       25.0  π-cascade ◄ CASCADE CONFIRMED
     642   107.0m  *** ⚠    CASCADE          19.5   0.2896     3.19       25.0  π-cascade ◄ CASCADE CONFIRMED
     643   107.2m  *** ⚠    CASCADE          19.4   0.3029     3.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     644   107.3m  *** ⚠    CASCADE          19.3   0.3153     3.33       25.0  π-cascade ◄ CASCADE CONFIRMED
     645   107.5m  *** ⚠    CASCADE          18.4   0.3314     4.68       25.0  π-cascade ◄ CASCADE CONFIRMED
     646   107.7m  *** ⚠    CASCADE          19.3   0.2830     3.05       25.0  π-cascade ◄ CASCADE CONFIRMED
     648   108.0m  *** ⚠    CASCADE          19.2   0.2997     3.21       25.0  π-cascade ◄ CASCADE CONFIRMED
     650   108.3m  *** ⚠    CASCADE          19.0   0.3003     2.98       25.0  π-cascade ◄ CASCADE CONFIRMED
     651   108.5m  *** ⚠    CASCADE          18.3   0.3049     4.28       25.0  π-cascade ◄ CASCADE CONFIRMED
     652   108.7m  *** ⚠    CASCADE          18.9   0.3197     3.14       25.0  π-cascade ◄ CASCADE CONFIRMED
     654   109.0m  *** ⚠    CASCADE          18.7   0.3148     3.20       25.0  π-cascade ◄ CASCADE CONFIRMED
     655   109.2m  *** ⚠    CASCADE          18.3   0.3073     3.38       25.0  π-cascade ◄ CASCADE CONFIRMED
     656   109.3m  *** ⚠    CASCADE          18.4   0.2834     3.17       25.0  π-cascade ◄ CASCADE CONFIRMED
     657   109.5m  *** ⚠    CASCADE          18.1   0.3154     3.63       25.0  π-cascade ◄ CASCADE CONFIRMED
     658   109.7m  *** ⚠    CASCADE          16.9   0.3321     6.84       25.0  π-cascade ◄ CASCADE CONFIRMED
     659   109.8m  *** ⚠    CASCADE          18.0   0.3146     3.12       25.0  π-cascade ◄ CASCADE CONFIRMED
     661   110.2m  *** ⚠    CASCADE          16.7   0.3156     4.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     662   110.3m  *** ⚠    CASCADE          16.6   0.3079     4.35       25.0  π-cascade ◄ CASCADE CONFIRMED
     665   110.8m  *** ⚠    CASCADE          17.3   0.3322     3.10       25.0  π-cascade ◄ CASCADE CONFIRMED
     667   111.2m  *** ⚠    CASCADE          17.0   0.3080     3.18       25.0  π-cascade ◄ CASCADE CONFIRMED
     668   111.3m  *** ⚠    CASCADE          16.9   0.3024     3.24       25.0  π-cascade ◄ CASCADE CONFIRMED
     669   111.5m  *** ⚠    CASCADE          16.8   0.2759     3.06       25.0  π-cascade ◄ CASCADE CONFIRMED
     670   111.7m  *** ⚠    CASCADE          16.5   0.3171     3.40       25.0  π-cascade ◄ CASCADE CONFIRMED
     671   111.8m  *** ⚠    CASCADE          16.4   0.3079     3.29       25.0  π-cascade ◄ CASCADE CONFIRMED
     672   112.0m  *** ⚠    CASCADE          16.3   0.2945     3.04       25.0  π-cascade ◄ CASCADE CONFIRMED
     673   112.2m  *** ⚠    CASCADE          15.9   0.2921     3.41       25.0  π-cascade ◄ CASCADE CONFIRMED
     674   112.3m  *** ⚠    CASCADE          15.9   0.2807     3.28       25.0  π-cascade ◄ CASCADE CONFIRMED
     676   112.7m  *** ⚠    CASCADE          15.8   0.2863     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     677   112.8m  *** ⚠    CASCADE          15.7   0.2788     3.33       25.0  π-cascade ◄ CASCADE CONFIRMED
     678   113.0m  *** ⚠    CASCADE          15.7   0.2827     2.97       25.0  π-cascade ◄ CASCADE CONFIRMED
     679   113.2m  *** ⚠    CASCADE          15.3   0.3067     3.40       25.0  π-cascade ◄ CASCADE CONFIRMED
     680   113.3m  *** ⚠    CASCADE          13.0   0.4220     9.63       25.0  π-cascade ◄ CASCADE CONFIRMED
     681   113.5m  *** ⚠    CASCADE          15.0   0.2908     3.29       25.0  π-cascade ◄ CASCADE CONFIRMED
     683   113.8m  *** ⚠    CASCADE          13.9   0.3290     3.57       25.0  π-cascade ◄ CASCADE CONFIRMED
     688   114.7m  *** ⚠    CASCADE          13.3   0.3376     4.06       25.0  π-cascade ◄ CASCADE CONFIRMED
     689   114.8m  *** ⚠    CASCADE          13.0   0.3728     3.99       25.0  π-cascade ◄ CASCADE CONFIRMED
     692   115.3m  *** ⚠    CASCADE          13.6   0.3102     3.05       25.0  π-cascade ◄ CASCADE CONFIRMED
     693   115.5m  *** ⚠    CASCADE          13.5   0.2791     2.99       25.0  π-cascade ◄ CASCADE CONFIRMED
     694   115.7m  *** ⚠    CASCADE          13.3   0.2687     3.21       25.0  π-cascade ◄ CASCADE CONFIRMED
     695   115.8m  *** ⚠    CASCADE          13.3   0.2810     3.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     696   116.0m  *** ⚠    CASCADE          12.4   0.3062     3.93       25.0  π-cascade ◄ CASCADE CONFIRMED
     697   116.2m  *** ⚠    CASCADE          12.2   0.2810     3.59       25.0  π-cascade ◄ CASCADE CONFIRMED
     699   116.5m  *** ⚠    CASCADE          11.8   0.3115     3.96       25.0  π-cascade ◄ CASCADE CONFIRMED
     700   116.7m  *** ⚠    CASCADE          12.2   0.3010     3.30       25.0  π-cascade ◄ CASCADE CONFIRMED
     702   117.0m  *** ⚠    CASCADE          12.0   0.2914     3.49       25.0  π-cascade ◄ CASCADE CONFIRMED
     703   117.2m  *** ⚠    CASCADE          11.9   0.2830     3.17       25.0  π-cascade ◄ CASCADE CONFIRMED
     704   117.3m  *** ⚠    CASCADE          11.6   0.2991     3.07       25.0  π-cascade ◄ CASCADE CONFIRMED
     705   117.5m  *** ⚠    CASCADE          11.7   0.2932     3.36       25.0  π-cascade ◄ CASCADE CONFIRMED
     706   117.7m  *** ⚠    CASCADE          11.5   0.2900     2.99       25.0  π-cascade ◄ CASCADE CONFIRMED
     707   117.8m  *** ⚠    CASCADE          11.5   0.2709     3.07       25.0  π-cascade ◄ CASCADE CONFIRMED
     708   118.0m  *** ⚠    CASCADE          11.0   0.2914     3.02       25.0  π-cascade ◄ CASCADE CONFIRMED
     709   118.2m  *** ⚠    CASCADE          11.0   0.2624     3.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     710   118.3m  *** ⚠    CASCADE          10.8   0.2772     3.08       25.0  π-cascade ◄ CASCADE CONFIRMED
     711   118.5m  *** ⚠    CASCADE          10.5   0.2677     3.17       25.0  π-cascade ◄ CASCADE CONFIRMED
     712   118.7m  *** ⚠    CASCADE          10.1   0.2805     3.10       25.0  π-cascade ◄ CASCADE CONFIRMED
     713   118.8m  *** ⚠    CASCADE           9.7   0.3040     3.01       25.0  π-cascade ◄ CASCADE CONFIRMED
     714   119.0m  *** ⚠    CASCADE           9.9   0.2832     3.08       25.0  π-cascade ◄ CASCADE CONFIRMED
     715   119.2m  !!! 🔴   CRITICAL          9.1   0.2952     3.35       25.0  π-cascade ◄ CASCADE CONFIRMED
     716   119.3m  !!! 🔴   CRITICAL          9.2   0.2888     3.03       25.0  π-cascade ◄ CASCADE CONFIRMED
     717   119.5m  !!! 🔴   CRITICAL          8.9   0.2756     3.77       25.0  π-cascade ◄ CASCADE CONFIRMED
     718   119.7m  !!! 🔴   CRITICAL          8.9   0.2811     3.32       25.0  π-cascade ◄ CASCADE CONFIRMED
     719   119.8m  !!! 🔴   CRITICAL          8.4   0.3102     3.79       25.0  π-cascade ◄ CASCADE CONFIRMED
     720   120.0m  !!! 🔴   CRITICAL          8.4   0.3039     3.58       25.0  π-cascade ◄ CASCADE CONFIRMED
     721   120.2m  !!! 🔴   CRITICAL          8.4   0.3114     3.19       25.0  π-cascade ◄ CASCADE CONFIRMED
     723   120.5m  !!! 🔴   CRITICAL          8.0   0.2697     3.14       25.0  π-cascade ◄ CASCADE CONFIRMED
     724   120.7m  !!! 🔴   CRITICAL          7.8   0.2830     3.60       25.0  π-cascade ◄ CASCADE CONFIRMED
     725   120.8m  !!! 🔴   CRITICAL          7.4   0.2829     3.22       25.0  π-cascade ◄ CASCADE CONFIRMED
     726   121.0m  !!! 🔴   CRITICAL          7.4   0.2862     3.19       25.0  π-cascade ◄ CASCADE CONFIRMED
     727   121.2m  !!! 🔴   CRITICAL          7.0   0.2835     3.29       25.0  π-cascade ◄ CASCADE CONFIRMED
     728   121.3m  !!! 🔴   CRITICAL          7.0   0.2849     3.24       25.0  π-cascade ◄ CASCADE CONFIRMED
     729   121.5m  !!! 🔴   CRITICAL          6.8   0.2869     3.15       25.0  π-cascade ◄ CASCADE CONFIRMED
     730   121.7m  !!! 🔴   CRITICAL          6.7   0.3035     3.08       25.0  π-cascade ◄ CASCADE CONFIRMED
     731   121.8m  !!! 🔴   CRITICAL          6.4   0.2991     3.92       25.0  π-cascade ◄ CASCADE CONFIRMED
     732   122.0m  !!! 🔴   CRITICAL          6.5   0.3066     3.26       25.0  π-cascade ◄ CASCADE CONFIRMED
     733   122.2m  !!! 🔴   CRITICAL          6.3   0.3090     3.53       25.0  π-cascade ◄ CASCADE CONFIRMED
     734   122.3m  !!! 🔴   CRITICAL          6.1   0.3233     3.35       25.0  π-cascade ◄ CASCADE CONFIRMED
     735   122.5m  !!! 🔴   CRITICAL          5.8   0.3192     3.86       25.0  π-cascade ◄ CASCADE CONFIRMED
     736   122.7m  !!! 🔴   CRITICAL          5.8   0.3378     3.62       25.0  π-cascade ◄ CASCADE CONFIRMED
     737   122.8m  !!! 🔴   CRITICAL          5.2   0.3334     4.07       25.0  π-cascade ◄ CASCADE CONFIRMED
     738   123.0m  !!! 🔴   CRITICAL          5.3   0.3531     4.21       25.0  π-cascade ◄ CASCADE CONFIRMED
     739   123.2m  !!! 🔴   CRITICAL          4.6   0.3501     4.90       25.0  π-cascade ◄ CASCADE CONFIRMED
     740   123.3m  !!! 🔴   CRITICAL          4.5   0.3557     5.97       25.0  π-cascade ◄ CASCADE CONFIRMED
     741   123.5m  !!! 🔴   CRITICAL          4.4   0.3711     5.36       25.0  π-cascade ◄ CASCADE CONFIRMED
     742   123.7m  !!! 🔴   CRITICAL          3.9   0.4178     6.66       25.0  π-cascade ◄ CASCADE CONFIRMED
     745   124.2m  !!! 🔴   CRITICAL          4.7   1.1064     7.93       25.0  π-cascade ◄ CASCADE CONFIRMED
     746   124.3m  !!! 🔴   CRITICAL          4.4   1.0856     6.14       25.0  π-cascade ◄ CASCADE CONFIRMED
     747   124.5m  !!! 🔴   CRITICAL          4.3   1.1741     4.50       25.0  π-cascade ◄ CASCADE CONFIRMED
     748   124.7m  !!! 🔴   CRITICAL          3.5   1.1976     5.00       25.0  π-cascade ◄ CASCADE CONFIRMED
     749   124.8m  !!! 🔴   CRITICAL          2.3   1.5342     5.99       25.0  π-cascade ◄ CASCADE CONFIRMED
     750   125.0m  !!! 🔴   CRITICAL          1.4   1.5900     5.58       25.0  π-cascade ◄ CASCADE CONFIRMED

════════════════════════════════════════════════════════════════════════
  CSP RUN SUMMARY  —  Bearing2_4
  Total bearing life : 125.2 min  (751 snapshots  ×  10s each)
  Snapshots shown    : 156
  False alarms       : 0  (CASCADE → back to HEALTHY/EARLY transitions)

  CASCADE DETECTION RESULT:
    Detected at      : +95.0 min  (76% through bearing life)
    Lead time        : 30.3 min  = 24% of life remaining
    Action window    : 30.3 min to act before predicted failure

  CSP CLAIM VERIFIED: cascade state identified 30.3 min
  before failure with structural π-type diagnosis.
════════════════════════════════════════════════════════════════════════