import numpy as np
import pandas as pd
from math import gcd, log2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

# ══════════════════════════════════════════════════════════════════════════════
# HARMONICS CONSONANCE — Pi-encoding test (parallel to logistic map)
#
# Domain transfer test: does the Pi-encoding method generalise beyond the
# logistic map to a structurally different system (harmonic consonance)?
#
# Dataset:
#   All unique pairs (n, m) with 1 ≤ n ≤ m ≤ MAX_HARMONIC.
#   MAX_HARMONIC=64 → 2080 pairs. Deterministic pool, subsampled per seed.
#
# Target:
#   Euler Gradus Suavitatis (GS). Integer consonance metric.
#   GS = 1 + Σ (prime − 1) × exponent  over all prime factors of p×q
#   where p/q = n/m in lowest terms. Lower = more consonant.
#   Unison=1, Octave=2, P5=4, P4=5, M3=7, m3=8, M2=10, m2=16, Tritone≈24.
#
# Raw features (11, log-scale):
#   All log-transformed so values stay in [0, ~6], compatible with
#   encode_pi_func's internal clip at 10.
#
# Structural scales (music-theoretic, analogous to Feigenbaum constants):
#   SCALE_OCT  = log2(2) = 1.0   → octave (fundamental harmonic period)
#   SCALE_H16  = log2(16) = 4.0  → 16th harmonic prime-limit boundary
#   SCALE_5LIM = log2(8)  = 3.0  → 5-limit JI (primes 2,3,5)
#   SCALE_TH   = 8.0             → Tenney height at 16-limit (log2(16²))
#
# Verification:
#   Same sweep structure as logistic map — 11 raw vs 11 encoded,
#   identical subsamples per (seed, n), 5-fold CV, console output only.
# ══════════════════════════════════════════════════════════════════════════════

MAX_HARMONIC = 64
N_VALUES     = [11, 22, 33, 50, 100, 150, 200, 300, 500]
N_SEEDS      = 10

PI  = np.pi
EPS = 1e-9

# ── Structural normalization scales ────────────────────────────────────────
SCALE_H16  = 4.0   # log2(16)  — 16th harmonic prime-limit boundary
SCALE_OCT  = 1.0   # log2(2)   — octave, the fundamental period
SCALE_5LIM = 3.0   # log2(8)   — 5-limit JI boundary
SCALE_GCD  = 4.0   # same as H16 (gcd bounded by harmonic number)
SCALE_TH   = 8.0   # log2(16²) — Tenney height at 16-limit
SCALE_SUM  = 4.0   # log2(16)  — reduced-sum complexity
SCALE_DIFF = 4.0   # log2(16)  — harmonic distance
SCALE_EDST = 4.0   # log2(16)  — Euler complexity measure
SCALE_FRAC = 1.0   # p/(p+q) ∈ (0, 0.5] — naturally bounded


# ══════════════════════════════════════════════════════════════════════════════
# CONSONANCE METRIC
# ══════════════════════════════════════════════════════════════════════════════

def prime_factors(n):
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def euler_gradus(n, m):
    """
    Euler's Gradus Suavitatis for ratio n:m.
    GS(n,m) = 1 + Σ (p−1)·e  over prime factors p^e of (n/gcd)×(m/gcd).
    Lower = more consonant. Unison = 1.
    """
    g = gcd(n, m)
    p, q = n // g, m // g
    return 1 + sum((prime - 1) * exp for prime, exp in prime_factors(p * q).items())


# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset(max_h):
    """
    All unique harmonic pairs (n, m), 1 ≤ n ≤ m ≤ max_h.
    Features are log-transformed to stay naturally in [0, ~6],
    compatible with encode_pi_func's internal clip at 10.
    """
    rows = []
    for n in range(1, max_h + 1):
        for m in range(n, max_h + 1):
            g     = gcd(n, m)
            p, q  = n // g, m // g       # reduced ratio (p ≤ q)
            ratio = m / n

            rows.append({
                # ── Raw features (11) — log-scale ────────────────────────
                "log_n_low":      log2(n),                  # lower harmonic
                "log_n_high":     log2(m),                  # upper harmonic
                "log_ratio":      log2(ratio),              # = 0 unison, 1 octave, ...
                "log_p_num":      log2(p + 1),              # reduced numerator
                "log_q_den":      log2(q + 1),              # reduced denominator
                "log_gcd":        log2(g + 1),              # gcd: high → simple ratio
                "tenney_h":       log2(p * q + 1),          # Tenney height (log scale)
                "log_p_plus_q":   log2(p + q + 1),          # complexity sum
                "log_diff":       log2(m - n + 1),          # harmonic distance
                "log_euler_dist": log2(p + q),              # Euler complexity: log2(p+q), p,q coprime → always ≥ log2(2)=1
                "p_frac":         p / (p + q + EPS),        # reduced fraction ∈ (0, 0.5]
                # ── Target ───────────────────────────────────────────────
                "euler_gs":       float(euler_gradus(n, m)),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PI-ENCODING  (identical function to logistic map — same weights, same clip)
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    """
    Fourier + power-pi basis — copied verbatim from logistic map version.
    weights=(5,1,1,3,1) unchanged.  clip at 10 unchanged.
    scale: music-theoretic normalization constant (not data-derived).
    """
    x  = np.clip(x, 0, 10)
    if scale is not None:
        xn = x / (scale + EPS)
    else:
        xn = x / (x.max() + EPS)
    w = np.array(weights, dtype=float) / np.sum(weights)
    d = {}
    d[f"{prefix}_sin_pi"]  = w[0] * np.sin(PI * xn)
    d[f"{prefix}_cos_pi"]  = w[1] * np.cos(PI * xn)
    d[f"{prefix}_sin_2pi"] = w[2] * np.sin(2 * PI * xn)
    d[f"{prefix}_sin_pi2"] = w[3] * np.sin(PI**2 * xn)
    d[f"{prefix}_cascade"] = w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn)
    return d


RAW_COLS = [
    "log_n_low", "log_n_high", "log_ratio", "log_p_num", "log_q_den",
    "log_gcd", "tenney_h", "log_p_plus_q", "log_diff", "log_euler_dist", "p_frac",
]

VAR_SCALES = {
    "log_n_low":      SCALE_H16,   # 16th harmonic boundary
    "log_n_high":     SCALE_H16,
    "log_ratio":      SCALE_OCT,   # octave = 1 in log2 space
    "log_p_num":      SCALE_5LIM,  # 5-limit JI (log2(8)=3)
    "log_q_den":      SCALE_5LIM,
    "log_gcd":        SCALE_GCD,
    "tenney_h":       SCALE_TH,    # log2(16²) — Tenney at 16-limit
    "log_p_plus_q":   SCALE_SUM,
    "log_diff":       SCALE_DIFF,
    "log_euler_dist": SCALE_EDST,
    "p_frac":         SCALE_FRAC,  # naturally in (0, 0.5]
}


def build_encoded_features(df):
    pi_enc = {}
    for col in RAW_COLS:
        pi_enc.update(encode_pi_func(df[col].values, f"pienc_{col}",
                                     scale=VAR_SCALES[col]))
    return pd.concat([df, pd.DataFrame(pi_enc, index=df.index)], axis=1)


# ── One encoded feature per source variable ────────────────────────────────
# Mirrors logistic map ENCODED_11 selection logic:
#   - Period/ratio variables (log_ratio, tenney_h): sin_pi peaks at tritone
#     (half-octave = maximum tension) and at the 16-limit midpoint
#   - Harmonic numbers: sin_pi2 captures sub-octave fine structure (w=3)
#   - Integer-lattice variables (p_num, q_den, gcd): cos_pi — unison (p=1)
#     maps to cos(π*log2(2)/3) ≈ cos(1.05) ≈ 0.5, rising toward 1 at p→0
#   - Complexity/distance measures: sin_pi — monotone over relevant range
ENCODED_11 = [
    "pienc_log_n_low_sin_pi2",       # lower harmonic — sub-octave structure
    "pienc_log_n_high_sin_pi2",      # upper harmonic
    "pienc_log_ratio_sin_pi",        # log ratio — peaks at tritone (half-octave)
    "pienc_log_p_num_cos_pi",        # reduced numerator — integer lattice
    "pienc_log_q_den_cos_pi",        # reduced denominator
    "pienc_log_gcd_cos_pi",          # gcd — high gcd → simple ratio → consonant
    "pienc_tenney_h_sin_pi",         # Tenney height — peaks at 16-limit midpoint
    "pienc_log_p_plus_q_sin_pi",     # complexity sum
    "pienc_log_diff_sin_pi",         # harmonic distance
    "pienc_log_euler_dist_sin_pi",   # Euler complexity
    "pienc_p_frac_sin_pi",           # p/(p+q) — smaller = more consonant
]


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(X, y, model, kf):
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


def main():
    t0 = time.time()

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"[INFO] generating harmonic pairs 1..{MAX_HARMONIC}...")
    df     = build_dataset(MAX_HARMONIC)
    N_POOL = len(df)
    gs     = df["euler_gs"]
    print(f"[INFO] pool : {N_POOL} unique pairs  (harmonics {MAX_HARMONIC}×{MAX_HARMONIC+1}/2)")
    print(f"[INFO] Euler GS — min={int(gs.min())}  max={int(gs.max())}  mean={gs.mean():.2f}\n")

    # Reference table
    examples = [
        (1,1,"Unison"),(2,1,"Octave"),(3,2,"P5"),(4,3,"P4"),
        (5,4,"M3"),(6,5,"m3"),(5,3,"M6"),(9,8,"M2"),(16,15,"m2"),(45,32,"Tritone"),
    ]
    print(f"  {'Interval':<12} {'Ratio':>7}  GS")
    for n, m, name in examples:
        print(f"  {name:<12} {n}/{m:<5}   {euler_gradus(n, m)}")
    print()

    # ── Feature matrices ──────────────────────────────────────────────────
    target     = gs.values.astype(float)
    X_raw_pool = df[RAW_COLS].values

    full_df    = build_encoded_features(df)
    feat_cols  = [c for c in full_df.columns if c.startswith("pienc_")]
    enc_idx    = [feat_cols.index(f) for f in ENCODED_11]
    X_enc_pool = full_df[feat_cols].values[:, enc_idx]

    print(f"[INFO] raw features    : {X_raw_pool.shape[1]}  {RAW_COLS}")
    print(f"[INFO] encoded features: {X_enc_pool.shape[1]}  {ENCODED_11}")
    print(f"[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds, 5-fold CV\n")

    # ── Sweep ─────────────────────────────────────────────────────────────
    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    names   = ["RF raw", "XGB raw", "RF encoded", "XGB encoded"]
    res     = {k: {n: [] for n in N_VALUES} for k in names}
    valid_n = [n for n in N_VALUES if n < N_POOL]

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in valid_n:
            idx = rng.choice(N_POOL, size=n, replace=False)
            Xr  = X_raw_pool[idx]
            Xf  = X_enc_pool[idx]
            y   = target[idx]

            models = {
                "RF raw":      (RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=2, random_state=42, n_jobs=-1), Xr),
                "XGB raw":     (xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                                    max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=0), Xr),
                "RF encoded":  (RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=2, random_state=42, n_jobs=-1), Xf),
                "XGB encoded": (xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                                    max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=0), Xf),
            }
            for name, (model, X) in models.items():
                res[name][n].append(run_cv(X, y, model, kf))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Console output ────────────────────────────────────────────────────
    W = 74
    print(f"\n{'='*W}")
    print(f"  HARMONICS — 11 RAW vs 11 ENCODED  "
          f"({N_SEEDS} seeds, 5-fold CV, target=Euler GS)")
    print(f"  pool={N_POOL} pairs, MAX_HARMONIC={MAX_HARMONIC}")
    print(f"{'='*W}")
    print(f"  {'n':>5}  {'RF raw':>12}  {'XGB raw':>12}  "
          f"{'RF enc':>12}  {'XGB enc':>12}  XGB enc > XGB raw?")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*18}")

    for n in valid_n:
        m    = {k: np.mean(res[k][n]) for k in names}
        s    = {k: np.std( res[k][n]) for k in names}
        d    = m["XGB encoded"] - m["XGB raw"]
        flag = f"YES  (Δ={d:+.4f})" if d > 0 else f"NO   (Δ={d:+.4f})"
        print(f"  {n:>5}  "
              f"{m['RF raw']:6.4f}±{s['RF raw']:.3f}  "
              f"{m['XGB raw']:6.4f}±{s['XGB raw']:.3f}  "
              f"{m['RF encoded']:6.4f}±{s['RF encoded']:.3f}  "
              f"{m['XGB encoded']:6.4f}±{s['XGB encoded']:.3f}  "
              f"{flag}")

    wins = sum(np.mean(res["XGB encoded"][n]) > np.mean(res["XGB raw"][n])
               for n in valid_n)
    print(f"{'='*W}")
    print(f"  XGB encoded beats XGB raw at {wins}/{len(valid_n)} sample sizes")
    print(f"{'='*W}")
    print(f"\n[TIMING] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""
python harmonics_consonance.py
"""