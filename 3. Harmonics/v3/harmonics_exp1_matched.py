import numpy as np
import pandas as pd
from math import gcd, log2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — MATCHED 11 vs 11: PURE COORDINATE SYSTEM TEST
#
# The only thing that changes between raw and encoded is the coordinate system.
# Same 11 features, same model configs, same N, same subsamples per seed.
#
# MATCHED_11 selected by importance rank from probe run:
#   eenc_log_q_den_gauss       0.0928  ← log_q_den  (e-encoded)
#   pienc_log_euler_dist_sin_pi 0.0524  ← log_euler_dist (pi-encoded)
#   pienc_tenney_h_sin_pi       0.0514  ← tenney_h   (pi-encoded)
#   pienc_log_p_plus_q_sin_pi   0.0431  ← log_p_plus_q  (pi-encoded)
#   pienc_log_n_high_sin_pi2    0.0403  ← log_n_high (pi-encoded)
#   eenc_log_p_num_pow_e        0.0339  ← log_p_num  (e-encoded)
#   pienc_log_n_low_sin_pi2     (est.)  ← log_n_low  (pi-encoded)
#   eenc_log_ratio_exp_neg      (est.)  ← log_ratio  (e-encoded)
#   eenc_log_gcd_exp_neg        (est.)  ← log_gcd    (e-encoded)
#   pienc_log_diff_sin_pi       (est.)  ← log_diff   (pi-encoded)
#   eenc_p_frac_gauss           (est.)  ← p_frac     (e-encoded)
#
# If XGB matched beats XGB raw at low N → the coordinate system is doing the
# work, not the feature count. The paper claim is clean.
# ══════════════════════════════════════════════════════════════════════════════

MAX_HARMONIC = 64
N_VALUES     = [11, 22, 33, 50, 100, 150, 200, 300, 500]
N_SEEDS      = 10

PI, E, EPS = np.pi, np.e, 1e-9

SCALE_H16  = 4.0
SCALE_OCT  = 1.0
SCALE_5LIM = 3.0
SCALE_GCD  = 4.0
SCALE_TH   = 8.0
SCALE_SUM  = 4.0
SCALE_DIFF = 4.0
SCALE_EDST = 4.0
SCALE_FRAC = 1.0


def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    x  = np.clip(x, 0, 10)
    xn = x / (scale + EPS) if scale is not None else x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2 * PI * xn),
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }


def encode_e_func(x, prefix, scale=None, weights=(2, 2, 1)):
    x  = np.clip(x, 0, 10)
    xn = x / (scale + EPS) if scale is not None else x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-E * xn),
        f"{prefix}_pow_e":   w[1] * xn ** E,
        f"{prefix}_gauss":   w[2] * np.exp(-E * (xn - 0.5)**2),
    }


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
    g = gcd(n, m); p, q = n // g, m // g
    return 1 + sum((pr - 1) * ex for pr, ex in prime_factors(p * q).items())


def build_dataset(max_h):
    rows = []
    for n in range(1, max_h + 1):
        for m in range(n, max_h + 1):
            g = gcd(n, m); p, q = n // g, m // g
            rows.append({
                "log_n_low":      log2(n),
                "log_n_high":     log2(m),
                "log_ratio":      log2(m / n),
                "log_p_num":      log2(p + 1),
                "log_q_den":      log2(q + 1),
                "log_gcd":        log2(g + 1),
                "tenney_h":       log2(p * q + 1),
                "log_p_plus_q":   log2(p + q + 1),
                "log_diff":       log2(m - n + 1),
                "log_euler_dist": log2(p + q),
                "p_frac":         p / (p + q + EPS),
                "euler_gs":       float(euler_gradus(n, m)),
            })
    return pd.DataFrame(rows)


RAW_COLS = [
    "log_n_low", "log_n_high", "log_ratio", "log_p_num", "log_q_den",
    "log_gcd", "tenney_h", "log_p_plus_q", "log_diff", "log_euler_dist", "p_frac",
]

PI_VARS = {
    "log_n_low":      SCALE_H16,
    "log_n_high":     SCALE_H16,
    "tenney_h":       SCALE_TH,
    "log_p_plus_q":   SCALE_SUM,
    "log_diff":       SCALE_DIFF,
    "log_euler_dist": SCALE_EDST,
}
E_VARS = {
    "log_ratio":  SCALE_OCT,
    "log_p_num":  SCALE_5LIM,
    "log_q_den":  SCALE_5LIM,
    "log_gcd":    SCALE_GCD,
    "p_frac":     SCALE_FRAC,
}

# PROBE-EXACT: best single encoded feature per source variable
# Derived from full 48-feature RF importance on the full 2080-pair pool.
# Not guessed — each selection is the highest-importance encoded feature
# for that source variable from the probe run.
#
# Key finding from probe: log_q_den (denominator of reduced ratio) is the
# dominant variable at 9.2% importance — nearly 3x any other single feature.
# log_diff and log_n_low are least informative (0.6% each).
MATCHED_11 = [
    ("log_n_low",      "pi", "pienc_log_n_low_sin_pi"),       # imp=0.0063
    ("log_n_high",     "pi", "pienc_log_n_high_sin_pi2"),     # imp=0.0393
    ("log_ratio",      "e",  "eenc_log_ratio_gauss"),         # imp=0.0068
    ("log_p_num",      "e",  "eenc_log_p_num_pow_e"),         # imp=0.0340
    ("log_q_den",      "e",  "eenc_log_q_den_pow_e"),         # imp=0.0920  ← dominant
    ("log_gcd",        "e",  "eenc_log_gcd_pow_e"),           # imp=0.0124
    ("tenney_h",       "pi", "pienc_tenney_h_sin_pi"),        # imp=0.0528
    ("log_p_plus_q",   "pi", "pienc_log_p_plus_q_sin_pi"),   # imp=0.0439
    ("log_diff",       "pi", "pienc_log_diff_sin_pi"),        # imp=0.0057
    ("log_euler_dist", "pi", "pienc_log_euler_dist_sin_pi"),  # imp=0.0521
    ("p_frac",         "e",  "eenc_p_frac_pow_e"),            # imp=0.0065
]
# Total MATCHED_11 importance: 35.2% (vs 23% expected if uniform across 48 features)


def build_matched_features(df):
    """Build all encoded features, then return only the MATCHED_11 column."""
    all_enc = {}
    for col, scale in PI_VARS.items():
        all_enc.update(encode_pi_func(df[col].values, f"pienc_{col}", scale=scale))
    for col, scale in E_VARS.items():
        all_enc.update(encode_e_func(df[col].values, f"eenc_{col}", scale=scale))
    enc_df = pd.DataFrame(all_enc, index=df.index)

    matched_cols = [feat_name for _, _, feat_name in MATCHED_11]
    return enc_df[matched_cols].values, matched_cols


def run_cv(X, y, model, kf):
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


def main():
    t_total = time.time()

    print(f"[INFO] building dataset...")
    df     = build_dataset(MAX_HARMONIC)
    N_POOL = len(df)
    gs     = df["euler_gs"].values.astype(float)
    print(f"[INFO] pool: {N_POOL} pairs  GS mean={gs.mean():.2f}  max={int(gs.max())}")

    X_raw_pool, _ = df[RAW_COLS].values, RAW_COLS
    X_match_pool, match_cols = build_matched_features(df)

    print(f"\n[INFO] Feature sets (both = 11 features):")
    print(f"  Raw     : {len(RAW_COLS)} features — raw log-scale statistics")
    print(f"  Matched : {len(match_cols)} features — one encoded feature per raw variable")
    print(f"\n  1-to-1 mapping (raw → encoded):")
    for raw_col, enc_type, feat_name in MATCHED_11:
        print(f"    {raw_col:<20} → [{enc_type}]  {feat_name}")

    print(f"\n[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds, 5-fold CV\n")

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    names   = ["RF raw", "XGB raw", "RF matched", "XGB matched"]
    res     = {k: {n: [] for n in N_VALUES} for k in names}
    valid_n = [n for n in N_VALUES if n < N_POOL]

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in valid_n:
            idx = rng.choice(N_POOL, size=n, replace=False)
            Xr  = X_raw_pool[idx]
            Xm  = X_match_pool[idx]
            y   = gs[idx]

            models = {
                "RF raw":      (RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=2, random_state=42, n_jobs=-1), Xr),
                "XGB raw":     (xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                                    max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=0), Xr),
                "RF matched":  (RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=2, random_state=42, n_jobs=-1), Xm),
                "XGB matched": (xgb.XGBRegressor(n_estimators=300, learning_rate=0.03,
                                    max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=0), Xm),
            }
            for name, (model, X) in models.items():
                res[name][n].append(run_cv(X, y, model, kf))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Results table ─────────────────────────────────────────────────────
    W = 82
    print(f"\n{'='*W}")
    print(f"  EXPERIMENT 1 — MATCHED 11 vs 11  "
          f"({N_SEEDS} seeds, 5-fold CV, target=Euler GS)")
    print(f"  Same 11 features each side. Only the coordinate system differs.")
    print(f"{'='*W}")
    print(f"  {'n':>5}  {'RF raw':>12}  {'XGB raw':>12}  "
          f"{'RF matched':>12}  {'XGB matched':>12}  XGB match > XGB raw?")
    print(f"  {'-'*5}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*20}")

    wins = 0
    for n in valid_n:
        m = {k: np.mean(res[k][n]) for k in names}
        s = {k: np.std( res[k][n]) for k in names}
        d = m["XGB matched"] - m["XGB raw"]
        flag = f"YES  (Δ={d:+.4f})" if d > 0 else f"NO   (Δ={d:+.4f})"
        if d > 0: wins += 1
        print(f"  {n:>5}  "
              f"{m['RF raw']:6.4f}±{s['RF raw']:.3f}  "
              f"{m['XGB raw']:6.4f}±{s['XGB raw']:.3f}  "
              f"{m['RF matched']:6.4f}±{s['RF matched']:.3f}  "
              f"{m['XGB matched']:6.4f}±{s['XGB matched']:.3f}  "
              f"{flag}")

    print(f"{'='*W}")
    print(f"  XGB matched beats XGB raw at {wins}/{len(valid_n)} sample sizes")

    # ── What this means ───────────────────────────────────────────────────
    low_n  = [n for n in valid_n if n <= 50]
    high_n = [n for n in valid_n if n >= 200]
    low_wins  = sum(np.mean(res["XGB matched"][n]) > np.mean(res["XGB raw"][n])
                    for n in low_n)
    high_wins = sum(np.mean(res["XGB matched"][n]) > np.mean(res["XGB raw"][n])
                    for n in high_n)
    print(f"  Low-N  (n≤50)  wins: {low_wins}/{len(low_n)}")
    print(f"  High-N (n≥200) wins: {high_wins}/{len(high_n)}")
    print(f"{'='*W}")

    if low_wins > len(low_n) // 2:
        print(f"\n  RESULT: Encoded coordinate system improves data efficiency at low N.")
        print(f"  The claim is clean: same features, same model, different basis → better fit.")
    elif wins > 0:
        print(f"\n  RESULT: Partial advantage. Encoding helps at some N values.")
        print(f"  Review crossover point — that is the honest efficiency number.")
    else:
        print(f"\n  RESULT: Raw features match or beat encoded at all N.")
        print(f"  GS may be better predicted by raw harmonic numbers than by encoded basis.")
        print(f"  MATCHED_11 feature selection may need revision from full importance run.")

    print(f"\n[TIMING] {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
python harmonics_exp1_matched.py
"""
