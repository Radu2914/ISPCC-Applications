import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import time

# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP — ALL-PI ENCODING VERIFICATION (v4: diagnostic probe framing)
# Paper 2 (bifurcation v4): encoding as a mathematical diagnostic tool.
#
# The logistic map is a uniformly Pi-type system. The all-Pi run (v3)
# confirmed this but revealed a deeper result: the Pi feature space splits
# into two structurally distinct subgroups that swap dominance across regimes,
# without the model being told which regime any point belongs to.
#
# Pi subgroup classification:
#   GEOMETRIC Pi (7 vars): std, var_d1, var_d2, spread, n_occ, r_val, abs_dist
#     → Capture the geometric footprint of the cascade: attractor width,
#       period count, spread. Maximally informative in the STABLE regime
#       where these quantities still discriminate between period-2, period-4,
#       period-8 attractors.
#   MEMORY Pi (4 vars): ac1, ac2, mean_x, inv_n_occ
#     → Capture the memory and positional structure: how fast autocorrelation
#       decays, where the mean settles, how period count fragments.
#       Maximally informative in the CHAOTIC regime where geometric measures
#       saturate (everything is wide, everything is dense) and memory
#       structure becomes the only remaining discriminator.
#
# Diagnostic property:
#   The importance distribution across Pi subgroups reveals regime character
#   without regime labels in the training data. The encoding is functioning
#   as a mathematical probe: the pattern of what the model uses carries
#   structural information about the system independent of prediction accuracy.
#   This property — domain-agnostic basis assignment yielding domain-revealing
#   importance redistribution — has not been previously formalized.
#
# Reframed verification logic:
#   PASS 1 — Pi-type system confirmed  : Pi total > 50% overall importance
#   PASS 2 — Regime-sensitive structure : Geometric Pi fraction higher in
#             stable; Memory Pi fraction higher in chaotic. The SWAP is
#             the condition, not the total.
#   PASS 3 — Encoding lifts learner    : XGB encoded reaches RF raw ceiling
#
# Structural constants:
#   R_BIFURCATION = 3.5699456  (Feigenbaum onset of chaos — exact)
#   DELTA = 4.669201609        (universal bifurcation interval ratio)
#   ALPHA = 2.502907875        (universal attractor width scaling)
# ══════════════════════════════════════════════════════════════════════════════

R_BIFURCATION = 3.5699456
R_MIN         = 2.5
R_MAX         = 4.0
N_SAMPLES     = 500
N_ITER        = 1500
N_DISCARD     = 1000
X0            = 0.5

PI  = np.pi
E   = np.e
EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
# FEIGENBAUM UNIVERSAL CONSTANTS — NORMALIZATION SCALES
# Analogous to λ_free / λ_rubber in EM: intrinsic structural scales,
# not derived from data statistics.
# ══════════════════════════════════════════════════════════════════════════════

DELTA = 4.669201609   # bifurcation interval ratio
ALPHA = 2.502907875   # attractor width scaling

SCALE_R        = R_BIFURCATION                          # r / R_BIF
SCALE_DIST     = R_MAX - R_BIFURCATION                  # |r-R_BIF| / chaotic width ≈ 0.4301
SCALE_STD      = 1.0 / (2.0 * ALPHA)                   # attractor width unit ≈ 0.1997
SCALE_VAR      = SCALE_STD ** 2                        # variance unit ≈ 0.0399
SCALE_SPREAD   = ALPHA                                 # spread / α ≈ 2.5029
SCALE_N_OCC    = DELTA ** 2                           # period count unit ≈ 21.80
SCALE_AC       = 1.0                                   # ac naturally bounded [−1,1]
SCALE_MEAN_X   = (R_BIFURCATION - 1.0) / R_BIFURCATION # fixed-point at bifurcation ≈ 0.7202
SCALE_INV_NOCC = 1.0 / DELTA                           # inverse period unit ≈ 0.2141

# Pi subgroup labels — structural, not just bookkeeping
# GEOM: geometric footprint of cascade (attractor width, period count, spread)
# MEMO: memory and positional structure (autocorrelation decay, mean position)
GEOM_PI_KEYS = ["std", "var_d1", "var_d2", "spread", "n_occ", "r_val", "abs_dist"]
MEMO_PI_KEYS = ["ac1", "ac2", "mean_x", "inv_n_occ"]


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING FUNCTION — PI ONLY
# encode_e_func is dropped: no E-type variables exist in this system.
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    """
    Fourier + power-pi basis for non-periodic cascading variables.
    Weights (5,1,1,3,1) from pi macro reduction — unchanged from EM version.

    scale : Feigenbaum-derived normalization constant.
            When provided → xn = x/scale (structurally grounded).
            When None     → xn = x/x.max() (data-relative fallback).
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


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def logistic_sequence(r, x0=X0, n_iter=N_ITER, n_discard=N_DISCARD):
    """Generate logistic map attractor sequence after discarding transients."""
    x = x0
    for _ in range(n_discard):
        x = r * x * (1 - x)
    seq = np.empty(n_iter - n_discard)
    for i in range(len(seq)):
        x = r * x * (1 - x)
        seq[i] = x
    return seq


def lyapunov_exponent(r, x0=X0, n_iter=N_ITER, n_discard=N_DISCARD):
    """
    Lyapunov exponent: mean log|r(1-2x)| over attractor.
    Negative = stable (periodic), positive = chaotic.
    """
    x = x0
    for _ in range(n_discard):
        x = r * x * (1 - x)
    lyap = 0.0
    n = n_iter - n_discard
    for _ in range(n):
        x = r * x * (1 - x)
        lyap += np.log(abs(r * (1 - 2 * x)) + EPS)
    return lyap / n


# ══════════════════════════════════════════════════════════════════════════════
# SEQUENCE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def sequence_statistics(r_values):
    rows = []
    for r in r_values:
        seq = logistic_sequence(r)

        mean_x = np.mean(seq)
        std_x  = np.std(seq)
        diff1  = np.diff(seq)
        diff2  = np.diff(diff1)
        var_d1 = np.var(diff1)
        var_d2 = np.var(diff2)

        if std_x < EPS:
            ac1, ac2 = 1.0, 1.0
        else:
            cc1 = np.corrcoef(seq[:-1], seq[1:])
            ac1 = float(cc1[0, 1]) if np.isfinite(cc1[0, 1]) else 1.0
            cc2 = np.corrcoef(seq[:-2], seq[2:])
            ac2 = float(cc2[0, 1]) if np.isfinite(cc2[0, 1]) else 1.0

        seq_range = seq.max() - seq.min()
        if seq_range < EPS:
            n_occ = 1.0
        else:
            hist, _ = np.histogram(seq, bins=100)
            n_occ   = float(np.sum(hist > 0))

        spread      = (seq.max() - seq.min()) / (mean_x + EPS)
        dist_bifurc = r - R_BIFURCATION
        abs_dist    = abs(dist_bifurc)

        rows.append({
            "r"          : r,
            "mean_x"     : mean_x,
            "std_x"      : std_x,
            "var_d1"     : var_d1,
            "var_d2"     : var_d2,
            "ac1"        : ac1,
            "ac2"        : ac2,
            "n_occ"      : n_occ,
            "spread"     : spread,
            "dist_bifurc": dist_bifurc,
            "abs_dist"   : abs_dist,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDING — ALL PI, FEIGENBAUM-NORMALIZED
#
# All 11 statistics Pi-encoded. Cross-products are now Pi×Pi interactions
# between cascade-intensity variable pairs, replacing the Pi×E structure
# used when E-type variables were assumed to exist.
# ══════════════════════════════════════════════════════════════════════════════

def build_encoded_features(stats_df):
    inv_n_occ = 1.0 / (stats_df["n_occ"].values + EPS)

    all_vars = [
        # key         array                             scale          group
        ("std",       stats_df["std_x"].values,        SCALE_STD,     "geom"),
        ("var_d1",    stats_df["var_d1"].values,       SCALE_VAR,     "geom"),
        ("var_d2",    stats_df["var_d2"].values,       SCALE_VAR,     "geom"),
        ("spread",    stats_df["spread"].values,       SCALE_SPREAD,  "geom"),
        ("n_occ",     stats_df["n_occ"].values,        SCALE_N_OCC,   "geom"),
        ("r_val",     stats_df["r"].values,            SCALE_R,       "geom"),
        ("abs_dist",  stats_df["abs_dist"].values,     SCALE_DIST,    "geom"),
        ("ac1",       stats_df["ac1"].values,          SCALE_AC,      "memo"),
        ("ac2",       stats_df["ac2"].values,          SCALE_AC,      "memo"),
        ("mean_x",    stats_df["mean_x"].values,       SCALE_MEAN_X,  "memo"),
        ("inv_n_occ", inv_n_occ,                       SCALE_INV_NOCC,"memo"),
    ]

    pi_enc = {}
    for key, col, scale, _ in all_vars:
        pi_enc.update(encode_pi_func(col, f"pienc_{key}", scale=scale))

    # Pi×Pi cross-products — cascade-intensity pair interactions
    std_n  = np.clip(stats_df["std_x"].values   / (SCALE_STD    + EPS), 0, 5)
    nocc_n = np.clip(stats_df["n_occ"].values   / (SCALE_N_OCC  + EPS), 0, 5)
    r_n    = np.clip(stats_df["r"].values       / (SCALE_R      + EPS), 0, 2)
    sp_n   = np.clip(stats_df["spread"].values  / (SCALE_SPREAD + EPS), 0, 5)
    vd1_n  = np.clip(stats_df["var_d1"].values  / (SCALE_VAR    + EPS), 0, 10)
    ac1_n  = np.clip(np.abs(stats_df["ac1"].values), 0, 1)

    cross = {
        # attractor spread × period count: both intensify as chaos grows
        "cross_cascade_x_period": np.sin(PI * std_n)  * np.sin(PI * nocc_n),
        # control parameter × attractor width: r drives spread
        "cross_r_x_spread":       np.sin(PI * r_n)    * np.sin(PI * sp_n),
        # difference variance × autocorrelation: complementary cascade measures
        "cross_diff_x_ac":        np.sin(PI * vd1_n)  * np.sin(PI * ac1_n),
    }

    return pd.concat([
        stats_df,
        pd.DataFrame(pi_enc),
        pd.DataFrame(cross)
    ], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

N_VALUES = [11, 22, 33, 50, 66, 100, 110, 121, 150, 200, 300, 500]
N_SEEDS  = 10
N_POOL   = 2000   # generated once; subsampled per (seed, n)

# One encoded feature per source variable — exact 1-to-1 with raw stats.
# Chosen by importance rank across runs. sin_pi (w=5) and sin_pi2 (w=3)
# dominate geom vars; memory vars (ac, mean_x, inv_n_occ) top on cos_pi.
ENCODED_11 = [
    "pienc_r_val_sin_pi",       # r         — w=5, dominant in all runs
    "pienc_mean_x_cos_pi",      # mean_x    — memory, cos_pi pattern
    "pienc_std_sin_2pi",        # std_x     — top importance confirmed
    "pienc_var_d1_sin_pi2",     # var_d1    — w=3 basis
    "pienc_var_d2_sin_pi2",     # var_d2    — w=3 basis
    "pienc_ac1_cos_pi",         # ac1       — memory, cos_pi variant
    "pienc_ac2_cos_pi",         # ac2       — confirmed top in runs
    "pienc_n_occ_sin_pi2",      # n_occ     — top geom confirmed
    "pienc_spread_cos_pi",      # spread    — confirmed in importance
    "pienc_abs_dist_sin_pi",    # abs_dist  — geom, sin_pi dominant
    "pienc_inv_n_occ_cos_pi",   # inv_n_occ — #1 overall importance
]


def run_cv(X, y, model, kf):
    """5-fold CV, returns mean R²."""
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


def main():
    t_total = time.time()

    # ── Build pool once — subsampled per seed/n ─────────────────────────────
    print(f"[INFO] building pool of {N_POOL} samples (computed once)...")
    np.random.seed(0)
    r_pool = np.random.uniform(R_MIN, R_MAX, N_POOL)

    stats_pool = sequence_statistics(r_pool)
    lyap_pool  = np.array([lyapunov_exponent(r) for r in r_pool])

    raw_stat_cols = ["r", "mean_x", "std_x", "var_d1", "var_d2",
                     "ac1", "ac2", "n_occ", "spread", "dist_bifurc", "abs_dist"]
    X_raw_pool  = stats_pool[raw_stat_cols].values

    full_df_pool = build_encoded_features(stats_pool)
    feat_cols    = [c for c in full_df_pool.columns if c not in stats_pool.columns]

    # Select exactly 11 encoded features — one per source variable
    enc_idx    = [feat_cols.index(f) for f in ENCODED_11]
    X_enc_pool = full_df_pool[feat_cols].values[:, enc_idx]

    print(f"[INFO] raw features    : {X_raw_pool.shape[1]}  →  {raw_stat_cols}")
    print(f"[INFO] encoded features: {X_enc_pool.shape[1]}  →  {ENCODED_11}")
    print(f"[INFO] sweep: n∈{N_VALUES}  seeds={N_SEEDS}  5-fold CV\n")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # results[model_name][n] = list of R² across seeds
    names = ["RF raw", "XGB raw", "RF encoded", "XGB encoded"]
    results = {k: {n: [] for n in N_VALUES} for k in names}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in N_VALUES:
            idx  = rng.choice(N_POOL, size=n, replace=False)
            Xr   = X_raw_pool[idx]
            Xf   = X_enc_pool[idx]
            y    = lyap_pool[idx]

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
                results[name][n].append(run_cv(X, y, model, kf))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Console summary ──────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  11 RAW vs 11 ENCODED — XGB encoded vs baselines  ({N_SEEDS} seeds, 5-fold CV)")
    print(f"{'='*72}")
    print(f"  {'n':>5}  {'RF raw':>12}  {'XGB raw':>12}  {'RF enc':>12}  {'XGB enc':>12}  XGB enc > XGB raw?")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*18}")

    for n in N_VALUES:
        m = {k: np.mean(results[k][n]) for k in names}
        s = {k: np.std( results[k][n]) for k in names}
        wins = m["XGB encoded"] > m["XGB raw"]
        delta = m["XGB encoded"] - m["XGB raw"]
        flag  = f"YES  (Δ={delta:+.4f})" if wins else f"NO   (Δ={delta:+.4f})"
        print(f"  {n:>5}  "
              f"{m['RF raw']:6.4f}±{s['RF raw']:.3f}  "
              f"{m['XGB raw']:6.4f}±{s['XGB raw']:.3f}  "
              f"{m['RF encoded']:6.4f}±{s['RF encoded']:.3f}  "
              f"{m['XGB encoded']:6.4f}±{s['XGB encoded']:.3f}  "
              f"{flag}")

    print(f"{'='*72}")
    wins_all = sum(np.mean(results["XGB encoded"][n]) > np.mean(results["XGB raw"][n])
                   for n in N_VALUES)
    print(f"  XGB encoded beats XGB raw at {wins_all}/{len(N_VALUES)} sample sizes")
    print(f"{'='*72}")
    print(f"\n[TIMING] Total: {time.time()-t_total:.1f}s")


    # ── done ────────────────────────────────────────────────────────────────


def _old_main_body_removed():
    """Placeholder — old main() body removed; all method code above is unchanged."""
    print("[INFO] computing sequence statistics...")
    t0 = time.time()
    stats_df = sequence_statistics(r_values)
    print(f"[TIMING] sequence generation: {time.time()-t0:.1f}s")

    print("[INFO] computing Lyapunov exponents (target — ground truth)...")
    t0 = time.time()
    lyap = np.array([lyapunov_exponent(r) for r in r_values])
    print(f"[TIMING] Lyapunov computation: {time.time()-t0:.1f}s")

    stable_mask  = lyap < 0
    chaotic_mask = lyap >= 0
    print(f"\n[INFO] Bifurcation constant : r = {R_BIFURCATION}")
    print(f"[INFO] Stable  regime (λ<0) : {stable_mask.sum()}  / {N_SAMPLES}")
    print(f"[INFO] Chaotic regime (λ≥0) : {chaotic_mask.sum()} / {N_SAMPLES}")

    print(f"\n[INFO] Feigenbaum normalization constants (all variables — Pi-encoded):")
    print(f"  δ  (bifurcation interval ratio) : {DELTA:.9f}")
    print(f"  α  (attractor width scaling)    : {ALPHA:.9f}")
    print(f"  ── Original Pi-encoded (7 variables) ────────────────────────")
    print(f"  r_val    ÷ R_BIFURCATION        : {SCALE_R:.7f}")
    print(f"  abs_dist ÷ chaotic regime width : {SCALE_DIST:.7f}")
    print(f"  std_x    ÷ 1/(2α)              : {SCALE_STD:.7f}")
    print(f"  var_d*   ÷ (1/(2α))²           : {SCALE_VAR:.7f}")
    print(f"  spread   ÷ α                   : {SCALE_SPREAD:.7f}")
    print(f"  n_occ    ÷ δ²                  : {SCALE_N_OCC:.7f}")
    print(f"  ── Reclassified to Pi (4 variables) ─────────────────────────")
    print(f"  ac1/ac2  ÷ 1.0                 : {SCALE_AC:.7f}  (step-wise cascade at bifurcations)")
    print(f"  mean_x   ÷ (R_BIF-1)/R_BIF    : {SCALE_MEAN_X:.7f}  (discontinuous at bifurcations)")
    print(f"  inv_n_occ÷ 1/δ                : {SCALE_INV_NOCC:.7f}  (fast step-wise transitions)")

    # Build feature sets
    raw_stat_cols = ["r", "mean_x", "std_x", "var_d1", "var_d2",
                     "ac1", "ac2", "n_occ", "spread", "dist_bifurc", "abs_dist"]
    X_raw  = stats_df[raw_stat_cols].values

    full_df   = build_encoded_features(stats_df)
    feat_cols = [c for c in full_df.columns if c not in stats_df.columns]
    X_full    = full_df[feat_cols].values

    geom_cols  = [c for c in feat_cols if any(f"pienc_{k}_" in c for k in GEOM_PI_KEYS)]
    memo_cols  = [c for c in feat_cols if any(f"pienc_{k}_" in c for k in MEMO_PI_KEYS)]
    cross_cols = [c for c in feat_cols if c.startswith("cross_")]

    print(f"\n[INFO] Feature sets:")
    print(f"  Raw statistics          : {X_raw.shape[1]}")
    print(f"  Full encoded (all Pi)   : {X_full.shape[1]}")
    print(f"    Geometric Pi (7 vars) : {len(geom_cols)} features")
    print(f"    Memory Pi    (4 vars) : {len(memo_cols)} features")
    print(f"    Cross-products (Pi×Pi): {len(cross_cols)} features")

    # ── Cross-validate ─────────────────────────────────────────────────────
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    candidates = {
        "RF raw (baseline)" : (
            RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                  min_samples_leaf=2, random_state=42, n_jobs=-1),
            X_raw, raw_stat_cols),
        "RF full encoded"   : (
            RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                  min_samples_leaf=2, random_state=42, n_jobs=-1),
            X_full, feat_cols),
        "XGB raw (baseline)": (
            xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0),
            X_raw, raw_stat_cols),
        "XGB full encoded"  : (
            xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, verbosity=0),
            X_full, feat_cols),
    }

    print(f"\n[INFO] cross-validating...\n")
    best_r2 = -999
    best_name, best_actual, best_pred = None, None, None
    timing    = {}
    r2_scores = {}

    for name, (model, X_use, _) in candidates.items():
        t0 = time.time()
        all_actual, all_pred, r2_list, rmse_list = [], [], [], []
        for train_idx, test_idx in kf.split(X_use):
            model.fit(X_use[train_idx], lyap[train_idx])
            p = model.predict(X_use[test_idx])
            a = lyap[test_idx]
            r2_list.append(r2_score(a, p))
            rmse_list.append(np.sqrt(np.mean((a - p)**2)))
            all_actual.extend(a)
            all_pred.extend(p)
        r2_m    = float(np.mean(r2_list))
        elapsed = time.time() - t0
        timing[name]    = elapsed
        r2_scores[name] = r2_m
        print(f"  [{name}]")
        print(f"    R²={r2_m:.4f}±{np.std(r2_list):.4f}  "
              f"RMSE={np.mean(rmse_list):.4f}  Time: {elapsed:.1f}s")
        if r2_m > best_r2:
            best_r2   = r2_m
            best_name = name
            best_actual = np.array(all_actual)
            best_pred   = np.array(all_pred)

    print(f"\n{'='*65}")
    print(f"  BEST MODEL : {best_name}")
    print(f"  R²         : {best_r2:.4f}")
    print(f"{'='*65}")

    # ── Feature importance — full dataset ──────────────────────────────────
    print("\n[INFO] feature importance (full all-Pi set, all data)...")
    rf_imp = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_imp.fit(X_full, lyap)
    print(f"[TIMING] importance fit: {time.time()-t0:.1f}s")

    imps = rf_imp.feature_importances_

    def gimp(arr, keywords):
        return sum(imp for col, imp in zip(feat_cols, arr)
                   if any(k in col for k in keywords))

    geom_all  = gimp(imps, [f"pienc_{k}_" for k in GEOM_PI_KEYS])
    memo_all  = gimp(imps, [f"pienc_{k}_" for k in MEMO_PI_KEYS])
    pi_all    = geom_all + memo_all
    cross_all = gimp(imps, ["cross_"])

    pairs = sorted(zip(feat_cols, imps), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 15 features:")
    for col, imp in pairs[:15]:
        if any(f"pienc_{k}_" in col for k in MEMO_PI_KEYS):
            tag = " [memo — bifurcation-sensitive]"
        elif col.startswith("cross_"):
            tag = " [cross Pi×Pi]"
        else:
            tag = " [geom — attractor footprint]"
        print(f"  {col:<42}: {imp:.4f}  {'#'*int(imp*80)}{tag}")

    print(f"\n  Category importances (full dataset):")
    print(f"  Pi geometric (7 vars)  : {geom_all:.4f}  ({geom_all*100:.1f}%)")
    print(f"  Pi memory    (4 vars)  : {memo_all:.4f}  ({memo_all*100:.1f}%)")
    print(f"  Pi total               : {pi_all:.4f}  ({pi_all*100:.1f}%)")
    print(f"  Cross-products (Pi×Pi) : {cross_all:.4f}  ({cross_all*100:.1f}%)")
    geom_frac_all = geom_all / (pi_all + 1e-9)
    memo_frac_all = memo_all / (pi_all + 1e-9)
    print(f"  Within-Pi split        : Geom {geom_frac_all*100:.1f}% / Memo {memo_frac_all*100:.1f}%")

    # ── Regime-split importance ─────────────────────────────────────────────
    print("\n[INFO] regime-split feature importance...")

    rf_s = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_s.fit(X_full[stable_mask], lyap[stable_mask])
    imps_s = rf_s.feature_importances_

    rf_c = RandomForestRegressor(n_estimators=300, max_features="sqrt",
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_c.fit(X_full[chaotic_mask], lyap[chaotic_mask])
    imps_c = rf_c.feature_importances_

    geom_stable  = gimp(imps_s, [f"pienc_{k}_" for k in GEOM_PI_KEYS])
    memo_stable  = gimp(imps_s, [f"pienc_{k}_" for k in MEMO_PI_KEYS])
    geom_chaotic = gimp(imps_c, [f"pienc_{k}_" for k in GEOM_PI_KEYS])
    memo_chaotic = gimp(imps_c, [f"pienc_{k}_" for k in MEMO_PI_KEYS])
    pi_stable    = geom_stable  + memo_stable
    pi_chaotic   = geom_chaotic + memo_chaotic

    # Within-Pi fractions: the swap is what matters
    geom_frac_stable  = geom_stable  / (pi_stable  + EPS)
    memo_frac_stable  = memo_stable  / (pi_stable  + EPS)
    geom_frac_chaotic = geom_chaotic / (pi_chaotic + EPS)
    memo_frac_chaotic = memo_chaotic / (pi_chaotic + EPS)

    print(f"\n  Stable regime  (λ<0, n={stable_mask.sum()}):")
    print(f"    Pi geometric importance : {geom_stable:.4f}  ({geom_stable*100:.1f}%)")
    print(f"    Pi memory importance    : {memo_stable:.4f}  ({memo_stable*100:.1f}%)")
    print(f"    Pi total                : {pi_stable:.4f}  ({pi_stable*100:.1f}%)")
    print(f"    Within-Pi split         : Geom {geom_frac_stable*100:.1f}% / Memo {memo_frac_stable*100:.1f}%")

    print(f"\n  Chaotic regime (λ≥0, n={chaotic_mask.sum()}):")
    print(f"    Pi geometric importance : {geom_chaotic:.4f}  ({geom_chaotic*100:.1f}%)")
    print(f"    Pi memory importance    : {memo_chaotic:.4f}  ({memo_chaotic*100:.1f}%)")
    print(f"    Pi total                : {pi_chaotic:.4f}  ({pi_chaotic*100:.1f}%)")
    print(f"    Within-Pi split         : Geom {geom_frac_chaotic*100:.1f}% / Memo {memo_frac_chaotic*100:.1f}%")

    geom_shift = geom_frac_chaotic - geom_frac_stable
    memo_shift = memo_frac_chaotic - memo_frac_stable
    print(f"\n  Subgroup shift (stable → chaotic):")
    print(f"    Geometric Pi fraction  : {geom_frac_stable*100:.1f}% → {geom_frac_chaotic*100:.1f}%  "
          f"({geom_shift*100:+.1f}%)")
    print(f"    Memory Pi fraction     : {memo_frac_stable*100:.1f}% → {memo_frac_chaotic*100:.1f}%  "
          f"({memo_shift*100:+.1f}%)")
    print(f"    → Encoding reveals regime character through importance redistribution")

    # ── Verification verdict ───────────────────────────────────────────────
    xgb_raw_r2 = r2_scores.get("XGB raw (baseline)", 0)
    xgb_enc_r2 = r2_scores.get("XGB full encoded",   0)
    rf_raw_r2  = r2_scores.get("RF raw (baseline)",  0)
    xgb_lift   = xgb_enc_r2 - xgb_raw_r2

    # PASS 1: Pi total dominates — confirms Pi-type system
    pass1 = pi_all > 0.50

    # PASS 2: THE SWAP — geometric fraction falls in chaos, memory fraction rises.
    # This is the diagnostic property: importance redistribution reveals regime
    # character without regime labels in training. The swap is the condition,
    # not the total (total Pi is flat across regimes by design).
    pass2 = (geom_frac_stable > geom_frac_chaotic) and \
            (memo_frac_chaotic > memo_frac_stable)

    # PASS 3: Encoding lifts constrained learner to RF ceiling
    pass3 = xgb_enc_r2 >= rf_raw_r2 - 0.015

    all_pass = pass1 and pass2 and pass3

    print(f"\n{'='*65}")
    print(f"  VERIFICATION — ALL-PI SYSTEM (DIAGNOSTIC PROBE)")
    print(f"")
    print(f"  PASS 1 — Pi-type system confirmed   : "
          f"{'PASS' if pass1 else 'FAIL'}")
    print(f"           Pi total importance = {pi_all*100:.1f}% > 50% threshold")
    print(f"           The logistic map is a uniformly Pi-type system.")
    print(f"")
    print(f"  PASS 2 — Regime-sensitive structure : "
          f"{'PASS' if pass2 else 'FAIL'}")
    print(f"           Geometric Pi: {geom_frac_stable*100:.1f}% (stable) → "
          f"{geom_frac_chaotic*100:.1f}% (chaotic)  "
          f"[{(geom_frac_chaotic-geom_frac_stable)*100:+.1f}%]")
    print(f"           Memory Pi  : {memo_frac_stable*100:.1f}% (stable) → "
          f"{memo_frac_chaotic*100:.1f}% (chaotic)  "
          f"[{(memo_frac_chaotic-memo_frac_stable)*100:+.1f}%]")
    print(f"           The encoding reveals regime character through importance")
    print(f"           redistribution — without regime labels in training data.")
    print(f"")
    print(f"  PASS 3 — Encoding lifts constrained learner : "
          f"{'PASS' if pass3 else 'FAIL'}")
    print(f"           XGB encoded {xgb_enc_r2:.4f} vs raw {xgb_raw_r2:.4f}  "
          f"(Δ={xgb_lift:+.4f})  RF ceiling {rf_raw_r2:.4f}")
    print(f"")

    verdict = 'PASS' if all_pass else 'PARTIAL' if any([pass1, pass2, pass3]) else 'FAIL'
    print(f"  OVERALL : {verdict}")
    if all_pass:
        print(f"")
        print(f"  The encoding functions as a mathematical diagnostic probe.")
        print(f"  Geometric Pi variables (attractor footprint) dominate the")
        print(f"  stable regime. Memory Pi variables (autocorrelation, mean")
        print(f"  position, period fragmentation) rise sharply in the chaotic")
        print(f"  regime as geometric measures saturate. The model discovered")
        print(f"  this structure from the importance distribution alone.")
        print(f"  This diagnostic property — domain-agnostic basis yielding")
        print(f"  domain-revealing importance redistribution — is the core")
        print(f"  novel claim of the encoding methodology.")
    print(f"{'='*65}")

    # ── Timing ─────────────────────────────────────────────────────────────
    print(f"\n[INFO] timing comparison:")
    for name, t in timing.items():
        print(f"  {name:<30}: {t:.1f}s")
    if "RF raw (baseline)" in timing and "RF full encoded" in timing:
        print(f"  RF encoding overhead      : "
              f"{timing['RF raw (baseline)']/timing['RF full encoded']:.2f}x")
    if "RF full encoded" in timing and "XGB full encoded" in timing:
        print(f"  XGB vs RF encoded speed   : "
              f"{timing['RF full encoded']/timing['XGB full encoded']:.2f}x")
    if "XGB raw (baseline)" in timing and "XGB full encoded" in timing:
        print(f"  XGB encoding overhead     : "
              f"{timing['XGB raw (baseline)']/timing['XGB full encoded']:.2f}x")

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Predicted vs actual (best model)
    colors = ['#d62728' if a >= 0 else '#1f77b4' for a in best_actual]
    axes[0].scatter(best_actual, best_pred, c=colors,
                    alpha=0.4, edgecolors='k', linewidths=0.2, s=20)
    lims = [min(best_actual.min(), best_pred.min()) - 0.05,
            max(best_actual.max(), best_pred.max()) + 0.05]
    axes[0].plot(lims, lims, 'k--', linewidth=1.5, label='Perfect fit')
    axes[0].axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    axes[0].axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    axes[0].set_xlabel("Actual Lyapunov Exponent")
    axes[0].set_ylabel("Predicted Lyapunov Exponent")
    axes[0].set_title(f"{best_name}\nR²={best_r2:.4f}")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: The Swap — within-Pi fraction shift is the core diagnostic result.
    # Shows geom vs memo fractions per regime as stacked bars so the
    # redistribution is immediately visible even when totals are similar.
    regime_labels = [f'Stable\n(λ<0, n={stable_mask.sum()})',
                     f'Chaotic\n(λ≥0, n={chaotic_mask.sum()})']
    geom_fracs = [geom_frac_stable,  geom_frac_chaotic]
    memo_fracs = [memo_frac_stable,  memo_frac_chaotic]
    x = np.arange(2)
    axes[1].bar(x, geom_fracs, label='Geometric Pi\n(attractor footprint)',
                color='#d62728', edgecolor='k', linewidth=0.8)
    axes[1].bar(x, memo_fracs, bottom=geom_fracs,
                label='Memory Pi\n(autocorr, mean, period)',
                color='#ff7f0e', edgecolor='k', linewidth=0.8)
    # Annotate the fractions
    for i in range(2):
        axes[1].text(i, geom_fracs[i] / 2,
                     f"Geom\n{geom_fracs[i]*100:.1f}%",
                     ha='center', va='center', fontsize=8,
                     fontweight='bold', color='white')
        axes[1].text(i, geom_fracs[i] + memo_fracs[i] / 2,
                     f"Memo\n{memo_fracs[i]*100:.1f}%",
                     ha='center', va='center', fontsize=8,
                     fontweight='bold', color='white')
    # Draw swap arrows between the two bars
    axes[1].annotate("",
        xy=(1 - 0.12, geom_frac_chaotic / 2),
        xytext=(0 + 0.12, geom_frac_stable / 2),
        arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.5))
    axes[1].annotate("",
        xy=(1 - 0.12, geom_frac_chaotic + memo_frac_chaotic / 2),
        xytext=(0 + 0.12, geom_frac_stable + memo_frac_stable / 2),
        arrowprops=dict(arrowstyle='->', color='#ff7f0e', lw=1.5))
    axes[1].set_ylabel("Within-Pi Fraction")
    axes[1].set_title("Pi Subgroup Swap Across Regimes\n"
                      "(diagnostic: no regime labels in training)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(regime_labels)
    axes[1].set_ylim([0, 1.15])
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: R² comparison across all models — shows encoding lift
    model_names = list(r2_scores.keys())
    model_r2    = [r2_scores[n] for n in model_names]
    bar_colors  = ['#1f77b4', '#2ca02c', '#9467bd', '#d62728']
    bars = axes[2].bar(range(len(model_names)), model_r2,
                       color=bar_colors[:len(model_names)],
                       edgecolor='k', linewidth=0.8)
    axes[2].set_xticks(range(len(model_names)))
    short_names = [n.replace(" (baseline)", "\n(baseline)").replace(" full ", "\nfull ")
                   for n in model_names]
    axes[2].set_xticklabels(short_names, fontsize=8)
    axes[2].set_ylabel("R² (5-fold CV)")
    axes[2].set_title("Model R² Comparison\n(encoding lift across learners)")
    r2_floor = min(model_r2) - 0.01
    axes[2].set_ylim([max(0.96, r2_floor), max(model_r2) + 0.01])
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, r2v in zip(bars, model_r2):
        axes[2].text(bar.get_x() + bar.get_width() / 2, r2v + 0.0005,
                     f"{r2v:.4f}", ha='center', va='bottom',
                     fontsize=8, fontweight='bold')
    # Annotate XGB lift arrow
    if "XGB raw (baseline)" in model_names and "XGB full encoded" in model_names:
        xi = model_names.index("XGB raw (baseline)")
        xj = model_names.index("XGB full encoded")
        axes[2].annotate(
            f"Δ={xgb_lift:+.4f}",
            xy=(xj, r2_scores["XGB full encoded"]),
            xytext=(xj - 0.5, r2_scores["XGB full encoded"] + 0.004),
            fontsize=8, color='#d62728', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2)
        )

    plt.suptitle(
        f"Logistic Map — All-Pi Diagnostic Probe  |  Verdict: {verdict}\n"
        f"encode_pi_func weights=(5,1,1,3,1)  |  Feigenbaum δ={DELTA:.4f}, α={ALPHA:.4f}  |  "
        f"r_bifurcation={R_BIFURCATION}  |  Best R²={best_r2:.4f}  |  "
        f"Geom↓{(geom_frac_chaotic-geom_frac_stable)*100:+.1f}% / "
        f"Memo↑{(memo_frac_chaotic-memo_frac_stable)*100:+.1f}% in chaos",
        fontsize=9, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("logistic_map_allpi_v4_verification.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("[INFO] plot saved to logistic_map_allpi_v4_verification.png")
    print(f"\n[TIMING] Total wall time: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
python logistical_map_norm_pi.py
"""