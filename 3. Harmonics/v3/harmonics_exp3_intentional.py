import numpy as np
import pandas as pd
from math import gcd, log2
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, pairwise_distances
import time

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — INTENTIONAL MAXIMIN SAMPLING + TWO-STAGE RESIDUAL MODEL
#
# Builds directly on Exp2 results and code:
#   - CANON-6 is confirmed as the minimal sufficient representation
#   - 6 features (log_q e×3 + log_p e×3) beat RAW-11 at 7/9 N values
#   - log_q_den dominates at 9.2%; problem reduces to 2 canonical variables
#   - CANON-6 > CANON-11 at high N: tenney_h = log_p + log_q → redundant
#
# Exp3 hypothesis:
#   IntentionalMaxiMin in CANON-6 space at N achieves the same R² as
#   random CANON-6 at 2N to 3N. Sampling in the structurally meaningful
#   space (encoded p and q) selects archetypal intervals — not geometric
#   spread in raw space.
#
# Protocol change from Exp1/Exp2:
#   Exp1/2 : 5-fold CV on N random draws from full 2080-point pool
#   Exp3   : fixed 500-point test set (seed=999), train from remaining 1580
#   → Exp3 R² numbers differ from Exp2; crossover table shows Exp2 references
#   → within-Exp3 comparisons are fair (same test set for all methods)
#
# Four methods:
#   A) RAW-11  + random sampling             (Exp1 baseline, re-evaluated)
#   B) CANON-6 + random sampling             (Exp2 winner, re-evaluated)
#   C) CANON-6 + MaxiMin sampling            (structural sample selection)
#   D) CANON-6 + MaxiMin + two-stage model   (grammar + dialect)
#
# MaxiMin in CANON-6 space:
#   Greedy: at each step, add the unselected point with maximum min-distance
#   to the already-selected set. O(N_pool × n_select) — fast on 1580 points.
#   Starting point varied across seeds → variance estimate.
#   Distance = Euclidean in 6D e-encoded (q, p) space → structurally meaningful.
#
# Two-stage model (D):
#   Stage 1 — Ridge on CANON-6: captures structural grammar (p,q → GS trend)
#   Stage 2 — XGB on Stage-1 residuals: captures nonlinear dialect
#   Prediction = Stage1(x) + Stage2(x)
#   Rationale: residuals are smaller and smoother → Stage2 converges faster
# ══════════════════════════════════════════════════════════════════════════════

MAX_HARMONIC = 64
N_VALUES     = [11, 22, 33, 50, 66, 100, 150, 200, 300, 500]
N_SEEDS      = 10
TEST_SIZE    = 500
TEST_SEED    = 999

PI, E, EPS = np.pi, np.e, 1e-9

# Normalization scales — inherited unchanged from Exp1/Exp2
SCALE_5LIM = 3.0   # log2(8)   — 5-limit JI: p and q bounded here
SCALE_TH   = 8.0   # log2(256) — Tenney height at 16-limit
SCALE_GCD  = 4.0   # log2(16)  — GCD convergence boundary


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING FUNCTIONS — unchanged from Exp1/Exp2, zero modification
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# DATASET — unchanged from Exp2
# ══════════════════════════════════════════════════════════════════════════════

def prime_factors(n):
    factors = {}; d = 2
    while d * d <= n:
        while n % d == 0: factors[d] = factors.get(d, 0) + 1; n //= d
        d += 1
    if n > 1: factors[n] = factors.get(n, 0) + 1
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


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS — inherited from Exp2 (build_canonical_features verbatim)
# ══════════════════════════════════════════════════════════════════════════════

def build_canonical_features(df):
    """
    Verbatim from Exp2. Returns CANON-6, CANON-11, CANON-14.
    Only CANON-6 is used in Exp3 (confirmed winner from Exp2).
    """
    enc = {}
    enc.update(encode_e_func(df["log_q_den"].values, "can_q",  scale=SCALE_5LIM))
    enc.update(encode_e_func(df["log_p_num"].values, "can_p",  scale=SCALE_5LIM))
    enc.update(encode_pi_func(df["tenney_h"].values, "can_pq", scale=SCALE_TH))
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


# ══════════════════════════════════════════════════════════════════════════════
# MODEL FACTORIES — inherited from Exp2 (make_xgb, make_rf verbatim)
# ══════════════════════════════════════════════════════════════════════════════

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)


def make_rf():
    return RandomForestRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)


# ══════════════════════════════════════════════════════════════════════════════
# NEW: MAXIMIN SELECTION IN CANON-6 SPACE
# ══════════════════════════════════════════════════════════════════════════════

def maximin_select(D, n_select, seed=0):
    """
    Greedy MaxiMin on a pre-computed N×N distance matrix D.

    At each step: select the unselected point with the maximum minimum-distance
    to the already-selected set. O(N_pool × n_select).

    Why CANON-6 space, not raw space:
      Raw distances spread across 11 dimensions (many redundant for GS).
      CANON-6 distances are in 6D (p,q) encoded space — structurally meaningful.
      MaxiMin here selects maximally diverse intervals in consonance character:
      unison (p=q=1), P5 (p=2,q=3), complex dissonances (large p,q), etc.
      This is "structural coverage" rather than geometric spread in raw coordinates.

    seed: controls starting point — varied across 10 seeds for variance estimate.
    """
    rng = np.random.default_rng(seed)
    n_pool = D.shape[0]

    # Random start (varied by seed)
    first = int(rng.integers(0, n_pool))
    selected = [first]
    selected_mask = np.zeros(n_pool, dtype=bool)
    selected_mask[first] = True

    # min_dists[i] = minimum distance from point i to any selected point
    min_dists = D[first].copy().astype(float)
    min_dists[first] = -np.inf

    for _ in range(n_select - 1):
        # Mask out already-selected
        candidate_dists = np.where(~selected_mask, min_dists, -np.inf)
        next_pt = int(np.argmax(candidate_dists))
        selected.append(next_pt)
        selected_mask[next_pt] = True
        # Incremental update: new min dist = min(old min, dist to next_pt)
        np.minimum(min_dists, D[next_pt], out=min_dists)
        min_dists[next_pt] = -np.inf

    return np.array(selected)


# ══════════════════════════════════════════════════════════════════════════════
# NEW: TWO-STAGE RESIDUAL MODEL
# ══════════════════════════════════════════════════════════════════════════════

def two_stage_predict(X_train, y_train, X_test):
    """
    Stage 1 — Ridge(CANON-6): learns the structural grammar.
      Linear fit in e-encoded (p,q) space → captures broad GS trend.
      Why Ridge and not OLS: stable at small N; same features as XGB.

    Stage 2 — XGB(CANON-6) on Stage-1 residuals: learns the dialect.
      Target = GS_actual - GS_stage1 → smaller, smoother than raw GS.
      Converges faster because structure is already removed.

    Final prediction: Stage1(x) + Stage2(x)

    For very small N (< 10): Ridge may underfit but XGB compensates.
    For large N: Stage1 captures the trend; Stage2 refines corners.
    """
    s1 = Ridge(alpha=1.0)
    s1.fit(X_train, y_train)
    residuals = y_train - s1.predict(X_train)

    s2 = make_xgb()
    s2.fit(X_train, residuals)

    return s1.predict(X_test) + s2.predict(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_total = time.time()

    # ── Dataset ───────────────────────────────────────────────────────────
    print("[INFO] building dataset...")
    df = build_dataset(MAX_HARMONIC)
    N_ALL = len(df)
    gs    = df["euler_gs"].values.astype(float)
    print(f"[INFO] pool: {N_ALL} pairs  GS mean={gs.mean():.2f}  max={int(gs.max())}")

    # ── Feature matrices ───────────────────────────────────────────────────
    X_raw = df[RAW_COLS].values
    X_c6, c6_cols, X_c11, _, X_c14, _ = build_canonical_features(df)

    print(f"\n[INFO] Feature sets in use:")
    print(f"  RAW-11  : {X_raw.shape[1]}  (reference baseline from Exp1)")
    print(f"  CANON-6 : {X_c6.shape[1]}   (Exp2 winner: log_q e×3 + log_p e×3)")
    print(f"  CANON-6 columns: {c6_cols}")

    # ── Fixed test set — held out before MaxiMin touches anything ─────────
    rng_test     = np.random.default_rng(TEST_SEED)
    test_idx     = rng_test.choice(N_ALL, size=TEST_SIZE, replace=False)
    test_mask    = np.zeros(N_ALL, dtype=bool)
    test_mask[test_idx] = True
    pool_idx     = np.where(~test_mask)[0]   # 1580 training candidates
    N_POOL       = len(pool_idx)

    X_raw_test   = X_raw[test_idx];    X_c6_test   = X_c6[test_idx]
    X_raw_pool   = X_raw[pool_idx];    X_c6_pool   = X_c6[pool_idx]
    y_test       = gs[test_idx];       y_pool       = gs[pool_idx]

    print(f"\n[INFO] Fixed test set : {TEST_SIZE} points (seed={TEST_SEED})")
    print(f"[INFO] Training pool  : {N_POOL} points (2080 - {TEST_SIZE})")
    print(f"[INFO] Protocol: train on N → predict on fixed {TEST_SIZE} held-out points")
    print(f"[INFO] Exp2 used 5-fold CV; Exp3 uses fixed test → numbers differ, ratios matter")

    # ── Precompute CANON-6 distance matrix for MaxiMin (once, all seeds/N) ─
    print(f"\n[INFO] computing CANON-6 pairwise distances ({N_POOL}×{N_POOL})...")
    t0 = time.time()
    D_c6 = pairwise_distances(X_c6_pool)   # 1580×1580, Euclidean in 6D
    print(f"[TIMING] distance matrix: {time.time()-t0:.2f}s")

    # ── EXP2 reference values (CANON-6 XGB, 5-fold CV protocol) ───────────
    # Used in crossover table for context. Not directly comparable (different protocol).
    exp2_c6_rand = {11: -25.51, 22: -2.88, 33: 0.011, 50: 0.203,
                    100: 0.601, 150: 0.713, 200: 0.701, 300: 0.794, 500: 0.849}

    print(f"\n[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds, fixed test set\n")

    # ── Results storage ───────────────────────────────────────────────────
    METHODS = ["A_raw_rand", "B_c6_rand", "C_c6_mm", "D_c6_mm_2s"]
    res     = {m: {n: [] for n in N_VALUES} for m in METHODS}
    valid_n = [n for n in N_VALUES if n <= N_POOL]

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)

        for n in valid_n:
            # A — RAW-11 random XGB (reference baseline)
            rand_idx = rng.choice(N_POOL, size=n, replace=False)
            m = make_xgb()
            m.fit(X_raw_pool[rand_idx], y_pool[rand_idx])
            res["A_raw_rand"][n].append(r2_score(y_test, m.predict(X_raw_test)))

            # B — CANON-6 random XGB (Exp2 winner re-evaluated on fixed test set)
            m = make_xgb()
            m.fit(X_c6_pool[rand_idx], y_pool[rand_idx])
            res["B_c6_rand"][n].append(r2_score(y_test, m.predict(X_c6_test)))

            # C — CANON-6 MaxiMin XGB (structural sampling)
            mm_idx = maximin_select(D_c6, n, seed=seed)
            m = make_xgb()
            m.fit(X_c6_pool[mm_idx], y_pool[mm_idx])
            res["C_c6_mm"][n].append(r2_score(y_test, m.predict(X_c6_test)))

            # D — CANON-6 MaxiMin + two-stage (grammar + dialect)
            y_hat = two_stage_predict(X_c6_pool[mm_idx], y_pool[mm_idx], X_c6_test)
            res["D_c6_mm_2s"][n].append(r2_score(y_test, y_hat))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Results table ─────────────────────────────────────────────────────
    W = 102
    print(f"\n{'='*W}")
    print(f"  EXPERIMENT 3 — IntentionalMaxiMin + Two-Stage Residual")
    print(f"  train on N → test on fixed {TEST_SIZE}-point set  |  {N_SEEDS} seeds  |  pool={N_POOL}")
    print(f"{'='*W}")
    print(f"  {'n':>5}  {'RAW-11 rand':>14}  {'C6 rand':>14}  "
          f"{'C6 MaxiMin':>14}  {'MM+2-stage':>14}  best  MM gain vs C6rand")
    print(f"  {'─'*5}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*6}  {'─'*20}")

    for n in valid_n:
        μ = {m: np.mean(res[m][n]) for m in METHODS}
        σ = {m: np.std(res[m][n])  for m in METHODS}
        best = max(μ, key=μ.get)
        mm_gain = μ["C_c6_mm"] - μ["B_c6_rand"]
        best_labels = {"A_raw_rand": "A", "B_c6_rand": "B", "C_c6_mm": "C", "D_c6_mm_2s": "D"}
        print(f"  {n:>5}  "
              f"{μ['A_raw_rand']:>6.4f}±{σ['A_raw_rand']:.3f}  "
              f"{μ['B_c6_rand']:>6.4f}±{σ['B_c6_rand']:.3f}  "
              f"{μ['C_c6_mm']:>6.4f}±{σ['C_c6_mm']:.3f}  "
              f"{μ['D_c6_mm_2s']:>6.4f}±{σ['D_c6_mm_2s']:.3f}  "
              f"  {best_labels[best]}     Δ={mm_gain:+.4f}")

    wins_mm = sum(np.mean(res["C_c6_mm"][n]) > np.mean(res["B_c6_rand"][n]) for n in valid_n)
    wins_2s = sum(np.mean(res["D_c6_mm_2s"][n]) > np.mean(res["B_c6_rand"][n]) for n in valid_n)
    print(f"{'='*W}")
    print(f"  MaxiMin vs C6 random : wins at {wins_mm}/{len(valid_n)} N values")
    print(f"  MM+2stage vs C6 rand : wins at {wins_2s}/{len(valid_n)} N values")
    print(f"{'='*W}")

    # ── Crossover analysis ────────────────────────────────────────────────
    # For each MaxiMin-N, find the smallest random N that achieves the same R²
    # on the same fixed test set.
    print(f"\n{'─'*W}")
    print(f"  CROSSOVER TABLE — MaxiMin-N equivalent to random at which N?")
    print(f"  (within Exp3 protocol; Exp2 references shown for context)")
    print(f"{'─'*W}")
    print(f"  {'MaxiMin N':>10}  {'MM R²':>8}  {'C6rand R²':>10}  "
          f"{'rand equiv':>11}  efficiency  Exp2-C6 ref")
    print(f"  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*11}  {'─'*10}  {'─'*11}")

    rand_means = {n: np.mean(res["B_c6_rand"][n]) for n in valid_n}
    mm_means   = {n: np.mean(res["C_c6_mm"][n])   for n in valid_n}

    for n in valid_n:
        mm_r2   = mm_means[n]
        cr_r2   = rand_means[n]
        exp2ref = exp2_c6_rand.get(n, float("nan"))
        # Smallest random N in sweep whose R² >= MaxiMin R²
        rand_equiv = next((rn for rn in valid_n if rand_means[rn] >= mm_r2), None)
        if rand_equiv is not None:
            ratio  = rand_equiv / n
            eff    = f"{ratio:.1f}× fewer"
        else:
            rand_equiv_str = "> max"
            ratio  = float("inf")
            eff    = "exceeds rand ceiling"

        rand_equiv_str = str(rand_equiv) if rand_equiv else "> max"
        ratio_str      = f"{ratio:.1f}×" if np.isfinite(ratio) else "∞"
        print(f"  {n:>10}  {mm_r2:>8.4f}  {cr_r2:>10.4f}  "
              f"{rand_equiv_str:>11}  {ratio_str:>6} fewer  {exp2ref:>8.4f}")

    # ── Two-stage residual analysis ───────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  TWO-STAGE RESIDUAL ANALYSIS")
    print(f"  Does the grammar-then-dialect structure add over MaxiMin alone?")
    print(f"{'─'*W}")
    print(f"  {'n':>5}  {'MM XGB (C)':>12}  {'MM+2stg (D)':>12}  "
          f"{'2stg gain':>10}  interpretation")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*35}")

    for n in valid_n:
        mm  = np.mean(res["C_c6_mm"][n])
        ts  = np.mean(res["D_c6_mm_2s"][n])
        gain = ts - mm
        if gain > 0.005:
            interp = "two-stage adds signal"
        elif gain < -0.005:
            interp = "XGB alone sufficient at this N"
        else:
            interp = "effectively equal"
        print(f"  {n:>5}  {mm:>12.4f}  {ts:>12.4f}  {gain:>+10.4f}  {interp}")

    print(f"\n[TIMING] Total: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
python harmonics_exp3_intentional.py
"""
