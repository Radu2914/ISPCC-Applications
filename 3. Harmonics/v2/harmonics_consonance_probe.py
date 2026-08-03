import numpy as np
import pandas as pd
from math import gcd, log2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

# ══════════════════════════════════════════════════════════════════════════════
# HARMONICS CONSONANCE — Pi + E ENCODING PROBE
# (parallel to logistical_map_norm.py)
#
# Regime analogy (mirrors logistic map stable/chaotic split):
#   Logistic:  stable (λ<0)  → e-encoding (self-regulating)
#   Logistic:  chaotic (λ≥0) → pi-encoding (cascading)
#   Harmonics: consonant (GS ≤ GS_THRESHOLD) → e-encoding (periodic, bounded)
#   Harmonics: dissonant (GS >  GS_THRESHOLD) → pi-encoding (cascading complexity)
#
# Variable assignment:
#   Pi-encoded (cascading — complexity grows with dissonance):
#     log_n_low, log_n_high  — harmonic numbers cascade outward
#     tenney_h               — log2(p*q) cascades with prime complexity
#     log_p_plus_q           — total numerator+denominator complexity grows
#     log_diff               — harmonic distance cascades
#     log_euler_dist         — Euler complexity log2(p+q) cascades
#
#   E-encoded (self-regulating — bounded by harmonic period/consonance):
#     log_ratio    — octave-periodic (log2(2)=1 in log2 space), self-regulating
#     log_p_num    — reduced numerator converges in 5-limit JI (2,3,5 primes)
#     log_q_den    — reduced denominator, same
#     log_gcd      — high gcd → simple ratio → naturally self-regulates
#     p_frac       — p/(p+q) ∈ (0, 0.5], naturally bounded
#
#   Cross-products (3):
#     sin(π · tenney_h)    × exp(−e · log_ratio)   — complexity × period
#     sin(π · log_n_high)  × exp(−e · p_frac)       — outer harmonic × fraction
#     sin(π · log_diff)    × exp(−e · log_gcd)      — distance × simplicity
#
# Verification (PASS/FAIL):
#   RF on consonant subset  → E-encoded importance should dominate
#   RF on dissonant subset  → Pi-encoded importance should dominate
#
# Encoding functions:
#   encode_pi_func and encode_e_func copied UNCHANGED from logistical_map_norm.py.
#   Same weights. No domain modification.
# ══════════════════════════════════════════════════════════════════════════════

MAX_HARMONIC  = 64
GS_THRESHOLD  = 10        # Euler GS ≤ 10 → consonant; > 10 → dissonant
N_VALUES      = [11, 22, 33, 50, 100, 150, 200, 300, 500]
N_SEEDS       = 10

PI  = np.pi
E   = np.e
EPS = 1e-9

# ── Structural normalization scales (music-theoretic, analogous to Feigenbaum) ─
# These are not derived from data — they are intrinsic to harmonic number theory.
#   log2(2) = 1.0  octave: the fundamental period of pitch space
#   log2(8) = 3.0  5-limit JI: primes 2,3,5 — where consonant intervals live
#   log2(16)= 4.0  16th harmonic prime-limit boundary
#   log2(256)=8.0  Tenney height at the 16-limit (log2(16^2))
#
# Analogous to EM:  sin(π · gap/λ_free)      →  sin(π · tenney/SCALE_TH)
#                   exp(−e · layer/λ_rubber)  →  exp(−e · log_ratio/SCALE_OCT)
# The argument to every trig/exponential is structurally meaningful.
# ────────────────────────────────────────────────────────────────────────────
SCALE_H16  = 4.0   # log2(16)  — 16th harmonic prime-limit boundary
SCALE_OCT  = 1.0   # log2(2)   — octave (fundamental period)
SCALE_5LIM = 3.0   # log2(8)   — 5-limit JI boundary (primes 2, 3, 5)
SCALE_GCD  = 4.0   # log2(16)  — gcd convergence boundary
SCALE_TH   = 8.0   # log2(256) — Tenney height at 16-limit
SCALE_SUM  = 4.0   # log2(16)  — reduced-sum complexity boundary
SCALE_DIFF = 4.0   # log2(16)  — harmonic distance boundary
SCALE_EDST = 4.0   # log2(16)  — Euler complexity boundary
SCALE_FRAC = 1.0   # p/(p+q) ∈ (0, 0.5] — naturally bounded


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING FUNCTIONS — IDENTICAL TO logistical_map_norm.py, ZERO MODIFICATION
# Same weights. Same clip. These carry no domain knowledge.
# They do not know what a harmonic is.
# They do not know what a logistic map is.
# They operate on any numerical array.
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    """
    Fourier + power-pi basis for non-periodic cascading variables.
    weights=(5,1,1,3,1) unchanged.  clip at 10 unchanged.
    scale: music-theoretic normalization (structurally grounded, not data-derived).
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


def encode_e_func(x, prefix, scale=None, weights=(2, 2, 1)):
    """
    Exponential basis for self-regulating / periodic variables.
    weights=(2,2,1) unchanged.  clip at 10 unchanged.
    Copied unchanged from logistical_map_norm.py.
    """
    x  = np.clip(x, 0, 10)
    if scale is not None:
        xn = x / (scale + EPS)
    else:
        xn = x / (x.max() + EPS)
    w = np.array(weights, dtype=float) / np.sum(weights)
    d = {}
    d[f"{prefix}_exp_neg"] = w[0] * np.exp(-E * xn)
    d[f"{prefix}_pow_e"]   = w[1] * xn ** E
    d[f"{prefix}_gauss"]   = w[2] * np.exp(-E * (xn - 0.5)**2)
    return d


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
    Features are log-transformed to stay in [0, ~6],
    compatible with encode_pi_func's internal clip at 10.
    """
    rows = []
    for n in range(1, max_h + 1):
        for m in range(n, max_h + 1):
            g     = gcd(n, m)
            p, q  = n // g, m // g
            ratio = m / n

            rows.append({
                # ── Raw features (11) — log-scale ─────────────────────────
                "log_n_low":      log2(n),
                "log_n_high":     log2(m),
                "log_ratio":      log2(ratio),
                "log_p_num":      log2(p + 1),
                "log_q_den":      log2(q + 1),
                "log_gcd":        log2(g + 1),
                "tenney_h":       log2(p * q + 1),
                "log_p_plus_q":   log2(p + q + 1),
                "log_diff":       log2(m - n + 1),
                "log_euler_dist": log2(p + q),
                "p_frac":         p / (p + q + EPS),
                # ── Target ────────────────────────────────────────────────
                "euler_gs":       float(euler_gradus(n, m)),
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDING WITH PI + E ENCODING
# Encoding assignment rationale:
#
# Pi-encoded (cascading / dissonance-growing):
#   log_n_low, log_n_high — harmonic number grows outward with no fixed period
#   tenney_h              — log2(p*q) accumulates prime factors → cascades
#   log_p_plus_q          — total complexity grows monotonically
#   log_diff              — |m−n| increases as harmonics diverge
#   log_euler_dist        — log2(p+q) is Euler's non-periodic complexity measure
#
# E-encoded (self-regulating / consonance-bounded):
#   log_ratio   — in log2 space: unison=0, octave=1, tritone≈0.5; periodic,
#                 bounded within [0, log2(MAX_HARMONIC)]; octave-periodic
#   log_p_num   — reduced numerator: for consonant intervals stays in {1,2,3,4,5};
#                 converges to 5-limit (log2(8)=3) — self-regulating
#   log_q_den   — same logic for denominator
#   log_gcd     — high gcd = shared factor = simple ratio; convergent measure
#   p_frac      — p/(p+q) ∈ (0, 0.5] by construction: naturally bounded
# ══════════════════════════════════════════════════════════════════════════════

RAW_COLS = [
    "log_n_low", "log_n_high", "log_ratio", "log_p_num", "log_q_den",
    "log_gcd", "tenney_h", "log_p_plus_q", "log_diff", "log_euler_dist", "p_frac",
]

# Pi-encoded variables: cascading character
PI_VARS = {
    "log_n_low":      SCALE_H16,   # n / 16th harmonic boundary
    "log_n_high":     SCALE_H16,   # m / 16th harmonic boundary
    "tenney_h":       SCALE_TH,    # log2(p*q) / log2(256) — Tenney at 16-limit
    "log_p_plus_q":   SCALE_SUM,   # sum complexity / log2(16)
    "log_diff":       SCALE_DIFF,  # harmonic distance / log2(16)
    "log_euler_dist": SCALE_EDST,  # Euler complexity / log2(16)
}

# E-encoded variables: self-regulating character
E_VARS = {
    "log_ratio":  SCALE_OCT,   # ratio / octave — bounded periodic measure
    "log_p_num":  SCALE_5LIM,  # reduced numerator / 5-limit boundary
    "log_q_den":  SCALE_5LIM,  # reduced denominator / 5-limit boundary
    "log_gcd":    SCALE_GCD,   # gcd / 16th harmonic boundary
    "p_frac":     SCALE_FRAC,  # naturally ∈ (0, 0.5] — no scaling needed
}


def build_encoded_features(df):
    pi_enc = {}
    for col, scale in PI_VARS.items():
        pi_enc.update(encode_pi_func(df[col].values, f"pienc_{col}", scale=scale))

    e_enc = {}
    for col, scale in E_VARS.items():
        e_enc.update(encode_e_func(df[col].values, f"eenc_{col}", scale=scale))

    # ── Cross-products: cascading × self-regulating ────────────────────────
    # Analogous to logistic map cross_std_x_ac1 etc.
    # Each cross-product = sin(π · cascade_var) × exp(−e · regulate_var)
    th_n   = np.clip(df["tenney_h"].values   / (SCALE_TH   + EPS), 0, 1)
    rat_n  = np.clip(df["log_ratio"].values  / (SCALE_OCT  + EPS), 0, 1)
    nh_n   = np.clip(df["log_n_high"].values / (SCALE_H16  + EPS), 0, 1)
    pf_n   = np.clip(df["p_frac"].values     / (SCALE_FRAC + EPS), 0, 1)
    diff_n = np.clip(df["log_diff"].values   / (SCALE_DIFF + EPS), 0, 1)
    gcd_n  = np.clip(df["log_gcd"].values    / (SCALE_GCD  + EPS), 0, 1)

    cross = {
        "cross_tenney_ratio":  np.sin(PI * th_n)   * np.exp(-E * rat_n),
        "cross_harmonic_frac": np.sin(PI * nh_n)   * np.exp(-E * pf_n),
        "cross_diff_gcd":      np.sin(PI * diff_n)  * np.exp(-E * gcd_n),
    }

    return pd.concat([
        df,
        pd.DataFrame(pi_enc,  index=df.index),
        pd.DataFrame(e_enc,   index=df.index),
        pd.DataFrame(cross,   index=df.index),
    ], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# CV HELPER
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(X, y, model, kf):
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # ── Dataset ───────────────────────────────────────────────────────────
    print(f"[INFO] generating harmonic pairs 1..{MAX_HARMONIC}...")
    df     = build_dataset(MAX_HARMONIC)
    N_POOL = len(df)
    gs     = df["euler_gs"].values.astype(float)
    print(f"[INFO] pool : {N_POOL} unique pairs  (harmonics {MAX_HARMONIC}×{MAX_HARMONIC+1}/2)")
    print(f"[INFO] Euler GS — min={int(gs.min())}  max={int(gs.max())}  mean={gs.mean():.2f}")

    # ── Regime masks — analogous to stable/chaotic in logistic map ─────────
    cons_mask = gs <= GS_THRESHOLD   # consonant: GS ≤ 10, self-regulating
    diss_mask = gs >  GS_THRESHOLD   # dissonant: GS > 10, cascading
    print(f"\n[INFO] Consonance threshold  : GS = {GS_THRESHOLD}")
    print(f"[INFO] Consonant (GS≤{GS_THRESHOLD}) : {cons_mask.sum():5d} / {N_POOL}  "
          f"(analogous to stable λ<0)")
    print(f"[INFO] Dissonant (GS>{GS_THRESHOLD})  : {diss_mask.sum():5d} / {N_POOL}  "
          f"(analogous to chaotic λ≥0)")

    # Reference interval table
    examples = [
        (1, 1,  "Unison"),
        (2, 1,  "Octave"),
        (3, 2,  "P5"),
        (4, 3,  "P4"),
        (5, 4,  "M3"),
        (6, 5,  "m3"),
        (5, 3,  "M6"),
        (9, 8,  "M2"),
        (16, 15, "m2"),
        (45, 32, "Tritone"),
    ]
    print(f"\n  {'Interval':<12} {'Ratio':>7}  GS   Regime")
    for n, m, name in examples:
        g_val  = euler_gradus(n, m)
        regime = f"consonant  (GS ≤ {GS_THRESHOLD})" if g_val <= GS_THRESHOLD else \
                 f"dissonant  (GS > {GS_THRESHOLD})"
        print(f"  {name:<12} {n}/{m:<5}   {g_val:<4d}  {regime}")

    # ── Structural scales ──────────────────────────────────────────────────
    print(f"\n[INFO] Structural normalization scales (music-theoretic):")
    print(f"  ── Pi-encoded (cascading / dissonance-growing) ──────────────────")
    print(f"  log_n_low/high  ÷ log2(16) = {SCALE_H16:.1f}  (16th harmonic limit)")
    print(f"  tenney_h        ÷ log2(256)= {SCALE_TH:.1f}  (Tenney height at 16-limit)")
    print(f"  log_p_plus_q    ÷ log2(16) = {SCALE_SUM:.1f}  (sum complexity boundary)")
    print(f"  log_diff        ÷ log2(16) = {SCALE_DIFF:.1f}  (harmonic distance boundary)")
    print(f"  log_euler_dist  ÷ log2(16) = {SCALE_EDST:.1f}  (Euler complexity boundary)")
    print(f"  ── E-encoded (self-regulating / consonance-bounded) ─────────────")
    print(f"  log_ratio       ÷ log2(2)  = {SCALE_OCT:.1f}  (octave = fundamental period)")
    print(f"  log_p_num/q_den ÷ log2(8)  = {SCALE_5LIM:.1f}  (5-limit JI boundary)")
    print(f"  log_gcd         ÷ log2(16) = {SCALE_GCD:.1f}  (gcd convergence boundary)")
    print(f"  p_frac          ÷ 1.0      = {SCALE_FRAC:.1f}  (naturally ∈ (0, 0.5])")

    # ── Build feature matrices ─────────────────────────────────────────────
    X_raw_pool = df[RAW_COLS].values

    full_df    = build_encoded_features(df)
    feat_cols  = [c for c in full_df.columns
                  if c.startswith("pienc_") or
                     c.startswith("eenc_")  or
                     c.startswith("cross_")]
    X_enc_pool = full_df[feat_cols].values

    pi_feats = [c for c in feat_cols if c.startswith("pienc_")]
    e_feats  = [c for c in feat_cols if c.startswith("eenc_")]
    cr_feats = [c for c in feat_cols if c.startswith("cross_")]

    print(f"\n[INFO] Feature sets:")
    print(f"  Raw (baseline)   : {X_raw_pool.shape[1]}  {RAW_COLS}")
    print(f"  Encoded total    : {X_enc_pool.shape[1]}")
    print(f"    Pi-encoded     : {len(pi_feats)}  ({len(PI_VARS)} vars × 5 Fourier terms)")
    print(f"    E-encoded      : {len(e_feats)}  ({len(E_VARS)} vars × 3 exp terms)")
    print(f"    Cross-products : {len(cr_feats)}  (pi × e interactions)")
    print(f"\n[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds, 5-fold CV\n")

    # ── Sweep over sample sizes ────────────────────────────────────────────
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
            y   = gs[idx]

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

    # ── Sweep results ──────────────────────────────────────────────────────
    W = 78
    print(f"\n{'='*W}")
    print(f"  HARMONICS PROBE — 11 RAW vs Pi+E ENCODED  "
          f"({N_SEEDS} seeds, 5-fold CV, target=Euler GS)")
    print(f"  pool={N_POOL} pairs, MAX_HARMONIC={MAX_HARMONIC}, "
          f"GS_THRESHOLD={GS_THRESHOLD}")
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

    # ── Full-dataset feature importance ────────────────────────────────────
    print(f"\n[INFO] feature importance (full encoded set, all data)...")
    rf_imp = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_imp.fit(X_enc_pool, gs)
    print(f"[TIMING] importance fit: {time.time()-t0:.1f}s")
    imps = rf_imp.feature_importances_

    def gimp(arr, keywords):
        return sum(imp for col, imp in zip(feat_cols, arr)
                   if any(k in col for k in keywords))

    pi_all = gimp(imps, ["pienc_"])
    e_all  = gimp(imps, ["eenc_"])
    cr_all = gimp(imps, ["cross_"])

    pairs = sorted(zip(feat_cols, imps), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 features (full dataset):")
    for col, imp in pairs[:15]:
        print(f"  {col:<42}: {imp:.4f}  {'#'*int(imp*80)}")

    print(f"\n  Category importances (full dataset):")
    print(f"  Pi-encoded (Fourier)    : {pi_all:.4f}  ({pi_all*100:.1f}%)")
    print(f"  E-encoded (exponential) : {e_all:.4f}  ({e_all*100:.1f}%)")
    print(f"  Cross-products          : {cr_all:.4f}  ({cr_all*100:.1f}%)")

    # ── Regime-split feature importance — THE CORE VERIFICATION ───────────
    print(f"\n[INFO] regime-split feature importance (core verification)...")
    print(f"  Consonant (GS≤{GS_THRESHOLD}) → E-encoding should dominate  "
          f"[mirrors: stable → E in logistic map]")
    print(f"  Dissonant (GS>{GS_THRESHOLD})  → Pi-encoding should dominate  "
          f"[mirrors: chaotic → Pi in logistic map]")

    rf_cons = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_cons.fit(X_enc_pool[cons_mask], gs[cons_mask])
    imps_cons = rf_cons.feature_importances_

    rf_diss = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_diss.fit(X_enc_pool[diss_mask], gs[diss_mask])
    imps_diss = rf_diss.feature_importances_

    pi_cons = gimp(imps_cons, ["pienc_"])
    e_cons  = gimp(imps_cons, ["eenc_"])
    cr_cons = gimp(imps_cons, ["cross_"])
    pi_diss = gimp(imps_diss, ["pienc_"])
    e_diss  = gimp(imps_diss, ["eenc_"])
    cr_diss = gimp(imps_diss, ["cross_"])

    print(f"\n  Consonant regime (GS≤{GS_THRESHOLD}, n={cons_mask.sum()}):")
    print(f"    Pi-encoded importance  : {pi_cons:.4f}  ({pi_cons*100:.1f}%)")
    print(f"    E-encoded importance   : {e_cons:.4f}  ({e_cons*100:.1f}%)")
    print(f"    Cross importance       : {cr_cons:.4f}  ({cr_cons*100:.1f}%)")
    print(f"    Dominant encoding      : {'E (correct)' if e_cons > pi_cons else 'Pi (incorrect)'}")

    print(f"\n  Dissonant regime (GS>{GS_THRESHOLD}, n={diss_mask.sum()}):")
    print(f"    Pi-encoded importance  : {pi_diss:.4f}  ({pi_diss*100:.1f}%)")
    print(f"    E-encoded importance   : {e_diss:.4f}  ({e_diss*100:.1f}%)")
    print(f"    Cross importance       : {cr_diss:.4f}  ({cr_diss*100:.1f}%)")
    print(f"    Dominant encoding      : {'Pi (correct)' if pi_diss > e_diss else 'E (incorrect)'}")

    # ── Verification verdict ───────────────────────────────────────────────
    cons_correct = e_cons  > pi_cons
    diss_correct = pi_diss > e_diss

    print(f"\n{'='*65}")
    if cons_correct and diss_correct:
        print("  VERIFICATION: PASS")
        print("  E-encoding dominates consonant regime  (self-regulating)")
        print("  Pi-encoding dominates dissonant regime (cascading)")
        print("  Discrimination tracks GS_THRESHOLD consonance boundary")
        print("  Same weights, zero domain modification, correct result")
        print("  encode_pi_func / encode_e_func are domain-agnostic")
        print("  structural discriminators — confirmed in 3rd domain")
    elif cons_correct:
        print("  VERIFICATION: PARTIAL")
        print("  E-encoding correctly dominates consonant regime")
        print("  Pi-encoding does NOT dominate dissonant regime")
        print("  Pi weights may need revision or GS_THRESHOLD adjustment")
    elif diss_correct:
        print("  VERIFICATION: PARTIAL")
        print("  Pi-encoding correctly dominates dissonant regime")
        print("  E-encoding does NOT dominate consonant regime")
        print("  E weights may need revision or GS_THRESHOLD adjustment")
    else:
        print("  VERIFICATION: FAIL")
        print("  Encoding does not discriminate by consonance regime")
        print("  Variable assignment or GS_THRESHOLD may need revision")
    print(f"{'='*65}")

    print(f"\n[TIMING] Total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
python harmonics_consonance_probe.py
"""
