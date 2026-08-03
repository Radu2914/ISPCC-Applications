import numpy as np
import pandas as pd
from math import gcd, log2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — CANONICAL (p, q, g) ENCODING vs INHERITED BASELINES
#
# Previous experiments inherited 11 features from the logistic map convention.
# The key insight: any pair (n, m) decomposes as  n = p·g, m = q·g, gcd(p,q)=1
# Euler GS depends ONLY on prime factors of p·q — g is mathematically irrelevant.
#
# From the probe run:
#   - log_q_den dominates at 9.2% importance (≈3× any other single feature)
#   - log_p_num is secondary at 3.4%
#   - log_p_plus_q and log_euler_dist are near-identical (differ by <0.4% at high N)
#   - log_diff and log_n_low are noise (0.6% each, plus they carry g contamination)
#
# Five feature sets compared here:
#   RAW-11   : 11 inherited log-scale statistics (baseline from Exp1)
#   MATCH-11 : 11 probe-selected best encoded feature per raw variable (Exp1 result)
#   CANON-6  : log_q (e×3) + log_p (e×3)                   — just the 2 GS variables
#   CANON-11 : CANON-6 + log(p·q)/tenney_h (pi×5)          — adds complexity axis
#   CANON-14 : CANON-11 + log_g (e×3)                      — adds voicing (noise test)
#
# Key hypotheses:
#   H1: CANON-11 ≥ MATCH-11   (canonical p/q beats inherited derived features)
#   H2: CANON-14 ≈ CANON-11   (log_g is noise for GS; confirms 2-variable structure)
#   H3: CANON-6  competitive  (q and p alone carry most of the GS signal)
# ══════════════════════════════════════════════════════════════════════════════

MAX_HARMONIC = 64
N_VALUES     = [11, 22, 33, 50, 100, 150, 200, 300, 500]
N_SEEDS      = 10

PI, E, EPS = np.pi, np.e, 1e-9

# Music-theoretic normalisation scales (unchanged from Exp1)
SCALE_H16  = 4.0   # log2(16) — 16th harmonic limit
SCALE_OCT  = 1.0   # log2(2)  — octave = fundamental period
SCALE_5LIM = 3.0   # log2(8)  — 5-limit JI boundary
SCALE_GCD  = 4.0   # log2(16) — GCD convergence boundary
SCALE_TH   = 8.0   # log2(256)— Tenney height at 16-limit
SCALE_SUM  = 4.0   # log2(16) — sum complexity boundary
SCALE_DIFF = 4.0   # log2(16) — harmonic distance boundary
SCALE_EDST = 4.0   # log2(16) — Euler complexity boundary
SCALE_FRAC = 1.0   # naturally ∈ (0, 0.5]


# ── Encoding functions (unchanged) ────────────────────────────────────────────

def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    """Pi-encoded: 5 Fourier-style terms. Suited to complexity-growing variables."""
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
    """E-encoded: 3 exponential terms. Suited to bounded / consonance-limited variables."""
    x  = np.clip(x, 0, 10)
    xn = x / (scale + EPS) if scale is not None else x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-E * xn),
        f"{prefix}_pow_e":   w[1] * xn ** E,
        f"{prefix}_gauss":   w[2] * np.exp(-E * (xn - 0.5)**2),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

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
    """
    Build the full 2080-pair pool.
    Stores both the original 11 derived features AND the 3 independent variables
    (log_p_num, log_q_den, log_gcd) explicitly for canonical feature construction.
    Note: log_p_num = log2(p+1), log_q_den = log2(q+1), log_gcd = log2(g+1)
    are already present in RAW_COLS — the canonical sets reuse them directly,
    encoding all terms rather than just the probe-best single term.
    """
    rows = []
    for n in range(1, max_h + 1):
        for m in range(n, max_h + 1):
            g = gcd(n, m); p, q = n // g, m // g
            rows.append({
                # Original inherited features
                "log_n_low":      log2(n),
                "log_n_high":     log2(m),
                "log_ratio":      log2(m / n),
                "log_p_num":      log2(p + 1),   # ← canonical: log(p)
                "log_q_den":      log2(q + 1),   # ← canonical: log(q)  [dominant]
                "log_gcd":        log2(g + 1),   # ← canonical: log(g)  [noise for GS]
                "tenney_h":       log2(p * q + 1),  # ← canonical: log(p·q) [complexity]
                "log_p_plus_q":   log2(p + q + 1),
                "log_diff":       log2(m - n + 1),
                "log_euler_dist": log2(p + q),
                "p_frac":         p / (p + q + EPS),
                "euler_gs":       float(euler_gradus(n, m)),
            })
    return pd.DataFrame(rows)


# ── Feature set 1: RAW-11 (baseline) ─────────────────────────────────────────

RAW_COLS = [
    "log_n_low", "log_n_high", "log_ratio", "log_p_num", "log_q_den",
    "log_gcd", "tenney_h", "log_p_plus_q", "log_diff", "log_euler_dist", "p_frac",
]


# ── Feature set 2: MATCH-11 (from Exp1, probe-selected) ──────────────────────

PI_VARS_INHERITED = {
    "log_n_low":      SCALE_H16,
    "log_n_high":     SCALE_H16,
    "tenney_h":       SCALE_TH,
    "log_p_plus_q":   SCALE_SUM,
    "log_diff":       SCALE_DIFF,
    "log_euler_dist": SCALE_EDST,
}
E_VARS_INHERITED = {
    "log_ratio":  SCALE_OCT,
    "log_p_num":  SCALE_5LIM,
    "log_q_den":  SCALE_5LIM,
    "log_gcd":    SCALE_GCD,
    "p_frac":     SCALE_FRAC,
}

# Best single encoded feature per source variable, from probe RF importance.
# eenc_log_q_den_pow_e selected (imp=0.0920) over _gauss (imp=0.0928) because
# _pow_e is the probe-best for the e-encoding family specifically.
MATCHED_11 = [
    ("log_n_low",      "pi", "pienc_log_n_low_sin_pi"),
    ("log_n_high",     "pi", "pienc_log_n_high_sin_pi2"),
    ("log_ratio",      "e",  "eenc_log_ratio_gauss"),
    ("log_p_num",      "e",  "eenc_log_p_num_pow_e"),
    ("log_q_den",      "e",  "eenc_log_q_den_pow_e"),      # dominant: imp=0.0920
    ("log_gcd",        "e",  "eenc_log_gcd_pow_e"),
    ("tenney_h",       "pi", "pienc_tenney_h_sin_pi"),
    ("log_p_plus_q",   "pi", "pienc_log_p_plus_q_sin_pi"),
    ("log_diff",       "pi", "pienc_log_diff_sin_pi"),
    ("log_euler_dist", "pi", "pienc_log_euler_dist_sin_pi"),
    ("p_frac",         "e",  "eenc_p_frac_pow_e"),
]


def build_matched_features(df):
    """Build full inherited encoding, return only the MATCHED_11 columns."""
    enc = {}
    for col, scale in PI_VARS_INHERITED.items():
        enc.update(encode_pi_func(df[col].values, f"pienc_{col}", scale=scale))
    for col, scale in E_VARS_INHERITED.items():
        enc.update(encode_e_func(df[col].values, f"eenc_{col}", scale=scale))
    enc_df = pd.DataFrame(enc, index=df.index)
    cols   = [feat for _, _, feat in MATCHED_11]
    return enc_df[cols].values, cols


# ── Feature sets 3-5: CANONICAL (p, q, g) ────────────────────────────────────

def build_canonical_features(df):
    """
    Encode the 3 true independent variables directly.

    Variable → encoding choice rationale:
      log_q (log_q_den) → e-basis: dominant at 9.2%, bounded denominator,
                          consonance-limiting → self-regulating character
      log_p (log_p_num) → e-basis: secondary at 3.4%, same character as q
      log(p·q) (tenney_h) → pi-basis: product grows with dissonance,
                            cascading / complexity-growing character
      log_g (log_gcd)   → e-basis: GCD is a convergence/grouping quantity;
                          should be noise for GS (confirms 2-var structure)

    Returns three feature matrices:
      CANON-6  : q(e×3) + p(e×3)                    = 6 features
      CANON-11 : CANON-6 + tenney_h(pi×5)            = 11 features
      CANON-14 : CANON-11 + g(e×3)                  = 14 features
    """
    enc = {}

    # q first — dominant signal from probe
    enc.update(encode_e_func(df["log_q_den"].values, "can_q",  scale=SCALE_5LIM))
    # p — secondary signal
    enc.update(encode_e_func(df["log_p_num"].values, "can_p",  scale=SCALE_5LIM))
    # p·q product — complexity axis, pi-encoded
    enc.update(encode_pi_func(df["tenney_h"].values, "can_pq", scale=SCALE_TH))
    # g — voicing, irrelevant for GS (structural noise test)
    enc.update(encode_e_func(df["log_gcd"].values,  "can_g",  scale=SCALE_GCD))

    enc_df = pd.DataFrame(enc, index=df.index)

    cols_q  = ["can_q_exp_neg", "can_q_pow_e", "can_q_gauss"]
    cols_p  = ["can_p_exp_neg", "can_p_pow_e", "can_p_gauss"]
    cols_pq = ["can_pq_sin_pi", "can_pq_cos_pi", "can_pq_sin_2pi",
               "can_pq_sin_pi2", "can_pq_cascade"]
    cols_g  = ["can_g_exp_neg", "can_g_pow_e", "can_g_gauss"]

    c6  = cols_q + cols_p
    c11 = c6  + cols_pq
    c14 = c11 + cols_g

    return (enc_df[c6].values,  c6,
            enc_df[c11].values, c11,
            enc_df[c14].values, c14)


# ── CV runner ─────────────────────────────────────────────────────────────────

def run_cv(X, y, model, kf):
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)


def make_rf():
    return RandomForestRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()

    print("[INFO] building dataset...")
    df     = build_dataset(MAX_HARMONIC)
    N_POOL = len(df)
    gs     = df["euler_gs"].values.astype(float)
    print(f"[INFO] pool: {N_POOL} pairs  GS mean={gs.mean():.2f}  max={int(gs.max())}")

    # Build all feature pools
    X_raw                          = df[RAW_COLS].values
    X_match,  match_cols           = build_matched_features(df)
    X_c6,  c6_cols, \
    X_c11, c11_cols, \
    X_c14, c14_cols                = build_canonical_features(df)

    print(f"\n[INFO] Feature sets:")
    print(f"  RAW-11   : {len(RAW_COLS):>2}  — inherited log-scale statistics (Exp1 baseline)")
    print(f"  MATCH-11 : {len(match_cols):>2}  — probe-selected best encoding per raw var (Exp1 result)")
    print(f"  CANON-6  : {len(c6_cols):>2}  — log_q (e×3) + log_p (e×3)")
    print(f"  CANON-11 : {len(c11_cols):>2}  — CANON-6 + log(p·q) pi×5  [same count as baselines]")
    print(f"  CANON-14 : {len(c14_cols):>2}  — CANON-11 + log_g e×3     [g = noise test for GS]")

    print(f"\n[INFO] Canonical encoding rationale:")
    print(f"  log_q → e-basis   dominant (9.2% probe imp); bounded denominator → consonance-limiting")
    print(f"  log_p → e-basis   secondary (3.4%); same bounded character as q")
    print(f"  log_pq→ pi-basis  tenney height grows with dissonance → cascading/unbounded")
    print(f"  log_g → e-basis   GCD is irrelevant for GS; CANON-14≈CANON-11 confirms 2-var structure")

    print(f"\n[INFO] Hypotheses:")
    print(f"  H1: CANON-11 ≥ MATCH-11   (direct p,q encoding beats inherited derived features)")
    print(f"  H2: CANON-14 ≈ CANON-11   (log_g is noise for GS)")
    print(f"  H3: CANON-6  competitive  (q and p alone carry most GS signal)")

    print(f"\n[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds, 5-fold CV\n")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # All model × feature combinations
    CONFIGS = {
        "RF  raw":   (make_rf,  X_raw),
        "XGB raw":   (make_xgb, X_raw),
        "RF  match": (make_rf,  X_match),
        "XGB match": (make_xgb, X_match),
        "RF  c6":    (make_rf,  X_c6),
        "XGB c6":    (make_xgb, X_c6),
        "RF  c11":   (make_rf,  X_c11),
        "XGB c11":   (make_xgb, X_c11),
        "RF  c14":   (make_rf,  X_c14),
        "XGB c14":   (make_xgb, X_c14),
    }

    res     = {k: {n: [] for n in N_VALUES} for k in CONFIGS}
    valid_n = [n for n in N_VALUES if n < N_POOL]

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in valid_n:
            idx = rng.choice(N_POOL, size=n, replace=False)
            y_n = gs[idx]
            for name, (model_fn, X_pool) in CONFIGS.items():
                res[name][n].append(run_cv(X_pool[idx], y_n, model_fn(), kf))
        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── XGB-only results table ────────────────────────────────────────────
    xgb_keys = ["XGB raw", "XGB match", "XGB c6", "XGB c11", "XGB c14"]
    W = 105
    print(f"\n{'='*W}")
    print(f"  EXPERIMENT 2 — CANONICAL vs INHERITED  "
          f"({N_SEEDS} seeds, 5-fold CV, target=Euler GS, XGBoost)")
    print(f"  RAW-11=baseline | MATCH-11=Exp1 probe | CANON-6/11/14=p,q,g direct encoding")
    print(f"{'='*W}")
    hdr = f"  {'n':>5}  " + "  ".join(f"{k:>16}" for k in xgb_keys) + "  best"
    print(hdr)
    print(f"  {'-'*5}  " + "  ".join(f"{'-'*16}" for _ in xgb_keys))

    for n in valid_n:
        m = {k: np.mean(res[k][n]) for k in xgb_keys}
        s = {k: np.std( res[k][n]) for k in xgb_keys}
        best = max(xgb_keys, key=lambda k: m[k])
        row  = f"  {n:>5}  " + "  ".join(f"{m[k]:6.4f}±{s[k]:.3f}" for k in xgb_keys)
        row += f"  ← {best}"
        print(row)

    print(f"{'='*W}")

    # ── RF-only results table ─────────────────────────────────────────────
    rf_keys = ["RF  raw", "RF  match", "RF  c6", "RF  c11", "RF  c14"]
    print(f"\n{'='*W}")
    print(f"  EXPERIMENT 2 — CANONICAL vs INHERITED  (Random Forest)")
    print(f"{'='*W}")
    hdr = f"  {'n':>5}  " + "  ".join(f"{k:>16}" for k in rf_keys) + "  best"
    print(hdr)
    print(f"  {'-'*5}  " + "  ".join(f"{'-'*16}" for _ in rf_keys))

    for n in valid_n:
        m = {k: np.mean(res[k][n]) for k in rf_keys}
        s = {k: np.std( res[k][n]) for k in rf_keys}
        best = max(rf_keys, key=lambda k: m[k])
        row  = f"  {n:>5}  " + "  ".join(f"{m[k]:6.4f}±{s[k]:.3f}" for k in rf_keys)
        row += f"  ← {best}"
        print(row)

    print(f"{'='*W}")

    # ── Hypothesis evaluations ────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  HYPOTHESIS EVALUATION")
    print(f"{'─'*W}")

    # H1: CANON-11 vs MATCH-11
    h1_wins = sum(np.mean(res["XGB c11"][n]) > np.mean(res["XGB match"][n]) for n in valid_n)
    h1_deltas = [np.mean(res["XGB c11"][n]) - np.mean(res["XGB match"][n]) for n in valid_n]
    print(f"\n  H1 — CANON-11 vs MATCH-11 (same 11 features, different structure):")
    print(f"       CANON-11 wins at {h1_wins}/{len(valid_n)} N values  avg Δ={np.mean(h1_deltas):+.4f}")
    if h1_wins > len(valid_n) // 2:
        print(f"       → SUPPORTED: direct p/q encoding beats inherited derived features.")
        print(f"         Harmonics is a 2-variable problem; the coordinate system confirms it.")
    elif h1_wins == len(valid_n) // 2:
        print(f"       → INCONCLUSIVE: tied. MATCH-11 and CANON-11 are equivalent in practice.")
    else:
        print(f"       → NOT SUPPORTED: inherited derived features are competitive or better.")
        print(f"         Possible cause: redundant derived features (log_p_plus_q, log_euler_dist)")
        print(f"         provide complementary signal that direct p/q encoding misses.")

    # H2: CANON-14 vs CANON-11 (g noise test)
    h2_wins  = sum(np.mean(res["XGB c14"][n]) > np.mean(res["XGB c11"][n]) for n in valid_n)
    h2_deltas = [np.mean(res["XGB c14"][n]) - np.mean(res["XGB c11"][n]) for n in valid_n]
    print(f"\n  H2 — CANON-14 vs CANON-11 (g noise test):")
    print(f"       CANON-14 wins at {h2_wins}/{len(valid_n)} N values  avg Δ={np.mean(h2_deltas):+.4f}")
    if h2_wins <= len(valid_n) // 3:
        print(f"       → CONFIRMED: log_g is noise for Euler GS, as expected.")
        print(f"         GS depends only on prime factors of p·q, not on g. Structure correct.")
    elif h2_wins >= (len(valid_n) * 2) // 3:
        print(f"       → UNEXPECTED: log_g adds signal. Possible dataset correlation between")
        print(f"         voicing (g) and the sampling distribution of p·q at MAX_HARMONIC=64.")
    else:
        print(f"       → MARGINAL: g contributes weakly. Check if correlation is sample-size-dependent.")

    # H3: CANON-6 vs RAW-11 (how much do p and q alone explain?)
    h3_wins  = sum(np.mean(res["XGB c6"][n]) > np.mean(res["XGB raw"][n]) for n in valid_n)
    h3_deltas = [np.mean(res["XGB c6"][n]) - np.mean(res["XGB raw"][n]) for n in valid_n]
    print(f"\n  H3 — CANON-6 vs RAW-11 (can 6 features from p,q beat 11 inherited?):")
    print(f"       CANON-6 wins at {h3_wins}/{len(valid_n)} N values  avg Δ={np.mean(h3_deltas):+.4f}")
    if h3_wins > len(valid_n) // 2:
        print(f"       → CONFIRMED: q and p alone, properly encoded, beat 11 inherited features.")
        print(f"         This is the strong form of the claim: fewer, better-grounded features win.")
    else:
        print(f"       → p and q alone are not sufficient without the tenney_h complexity term.")
        print(f"         CANON-11 (which adds log(p·q)) is likely the right minimum set.")

    # Summary delta table
    print(f"\n{'─'*W}")
    print(f"  SUMMARY — Average Δ vs XGB raw across all N values:")
    print(f"{'─'*W}")
    for k in ["XGB match", "XGB c6", "XGB c11", "XGB c14"]:
        deltas   = [np.mean(res[k][n]) - np.mean(res["XGB raw"][n]) for n in valid_n]
        wins     = sum(d > 0 for d in deltas)
        avg_d    = np.mean(deltas)
        low_d    = np.mean([deltas[i] for i, n in enumerate(valid_n) if n <= 50])
        high_d   = np.mean([deltas[i] for i, n in enumerate(valid_n) if n >= 200])
        label    = {
            "XGB match": "MATCH-11 (Exp1)",
            "XGB c6":    "CANON-6  (q+p only)",
            "XGB c11":   "CANON-11 (q+p+pq)",
            "XGB c14":   "CANON-14 (q+p+pq+g)",
        }[k]
        print(f"  {label:<28}  wins={wins}/{len(valid_n)}  "
              f"avg Δ={avg_d:+.4f}  "
              f"low-N Δ={low_d:+.4f}  "
              f"high-N Δ={high_d:+.4f}")

    print(f"{'─'*W}")
    print(f"\n[TIMING] {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
python harmonics_exp2_canonical.py
"""
