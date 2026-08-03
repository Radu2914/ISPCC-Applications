import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import time

# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP — 3-STAGE PIPELINE COMPARISON (v7)
# Paper 2, bifurcation series.
#
# MaxiMin REMOVED. Reason: the logistic map has a single input r.
# All encoded features are functions of r alone. MaxiMin in the 4D encoded
# space is therefore MaxiMin across four nonlinear transforms of the same
# 1D variable — it selects structurally extreme r values (bifurcation corners,
# chaos windows) and then 5-fold CV tests at points intentionally far from
# training. Result: R² = −51 at n=22. It is not a failure of MaxiMin as an
# algorithm — it is the wrong tool for a 1D parameter system.
# MaxiMin is valid for multi-dimensional independent design spaces (EM: 4
# physical inputs). It is not valid here.
#
# WHAT THIS CODE SHOWS:
#   The 3-stage pipeline (Ridge structural grammar + RF residuals) is the real
#   finding. It beats both single-stage baselines at 8/8 sample sizes up to
#   n=200 (from v6) and should hold at larger N. This is the domain-agnostic
#   claim: the pipeline architecture improves over single-stage regardless of
#   selection strategy.
#
# ── 3 TRACKS ────────────────────────────────────────────────────────────────
#   A  RF raw         (baseline — raw 11 statistics, RF)
#   B  XGB encoded    (previous best — ENCODED_11, XGB)
#   C  3-stage        (NEW — Ridge structural + RF residuals, ENCODED_11)
#
# ── 3-STAGE PIPELINE ────────────────────────────────────────────────────────
#   Stage 0  Structural features (no fitting):
#              [r, abs_dist, pienc_r_val_sin_pi,
#               pienc_inv_n_occ_cos_pi, pienc_abs_dist_sin_pi]
#            These are the Feigenbaum-derived regime indicators —
#            analogous to EM Pi28/Pi31/Pi32 geometric groups.
#
#   Stage 1  Ridge on Stage-0 features.
#            Captures: dominant λ ~ f(r) monotone trend
#                      near-bifurcation regime inflection
#                      Feigenbaum-normalised period structure
#            Removes these from residuals so Stage 2 has a simpler target.
#
#   Stage 2  RF on ENCODED_11 → Stage-1 residuals.
#            Captures: bifurcation cascade windows
#                      period-doubling fine structure
#                      chaos-band sub-structure
#            Final: λ_pred = Stage-1 output + Stage-2 residual correction
#
# ── N RANGE ────────────────────────────────────────────────────────────────
#   Pool = 2000. Clean limit: N ≤ 500 (25% of pool per seed, no overlap bias).
#   To go to N=1000 cleanly: increase N_POOL to 4000–5000.
#   N_VALUES = [11, 22, 33, 50, 66, 100, 150, 200, 300, 500]
#
# ── EXPECTED CLAIM ──────────────────────────────────────────────────────────
#   3-stage at n=X achieves what RF raw needs n=Y > X to achieve.
#   The pipeline is a data-efficiency multiplier independent of selection.
# ══════════════════════════════════════════════════════════════════════════════

R_BIFURCATION = 3.5699456
R_MIN         = 2.5
R_MAX         = 4.0
N_ITER        = 1500
N_DISCARD     = 1000
X0            = 0.5

PI  = np.pi
EPS = 1e-9

DELTA = 4.669201609
ALPHA = 2.502907875

SCALE_R        = R_BIFURCATION
SCALE_DIST     = R_MAX - R_BIFURCATION
SCALE_STD      = 1.0 / (2.0 * ALPHA)
SCALE_VAR      = SCALE_STD ** 2
SCALE_SPREAD   = ALPHA
SCALE_N_OCC    = DELTA ** 2
SCALE_AC       = 1.0
SCALE_MEAN_X   = (R_BIFURCATION - 1.0) / R_BIFURCATION
SCALE_INV_NOCC = 1.0 / DELTA


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING — unchanged from v4/v6
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    x  = np.clip(x, 0, 10)
    xn = x / (scale + EPS) if scale is not None else x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / np.sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2 * PI * xn),
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }


# ══════════════════════════════════════════════════════════════════════════════
# LOGISTIC MAP GENERATION — unchanged from v4/v6
# ══════════════════════════════════════════════════════════════════════════════

def logistic_sequence(r, x0=X0, n_iter=N_ITER, n_discard=N_DISCARD):
    x = x0
    for _ in range(n_discard):
        x = r * x * (1 - x)
    seq = np.empty(n_iter - n_discard)
    for i in range(len(seq)):
        x = r * x * (1 - x)
        seq[i] = x
    return seq


def lyapunov_exponent(r, x0=X0, n_iter=N_ITER, n_discard=N_DISCARD):
    x = x0
    for _ in range(n_discard):
        x = r * x * (1 - x)
    lyap = 0.0
    n    = n_iter - n_discard
    for _ in range(n):
        x     = r * x * (1 - x)
        lyap += np.log(abs(r * (1 - 2 * x)) + EPS)
    return lyap / n


def sequence_statistics(r_values):
    rows = []
    for r in r_values:
        seq    = logistic_sequence(r)
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
        sr    = seq.max() - seq.min()
        n_occ = 1.0 if sr < EPS else float(
            np.sum(np.histogram(seq, bins=100)[0] > 0))
        spread      = sr / (mean_x + EPS)
        dist_bifurc = r - R_BIFURCATION
        abs_dist    = abs(dist_bifurc)
        rows.append({
            "r": r, "mean_x": mean_x, "std_x": std_x,
            "var_d1": var_d1, "var_d2": var_d2,
            "ac1": ac1, "ac2": ac2, "n_occ": n_occ,
            "spread": spread, "dist_bifurc": dist_bifurc, "abs_dist": abs_dist,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE BUILDING — unchanged from v4/v6
# ══════════════════════════════════════════════════════════════════════════════

def build_encoded_features(stats_df):
    inv_n_occ = 1.0 / (stats_df["n_occ"].values + EPS)
    all_vars = [
        ("std",       stats_df["std_x"].values,    SCALE_STD,     ),
        ("var_d1",    stats_df["var_d1"].values,    SCALE_VAR,     ),
        ("var_d2",    stats_df["var_d2"].values,    SCALE_VAR,     ),
        ("spread",    stats_df["spread"].values,    SCALE_SPREAD,  ),
        ("n_occ",     stats_df["n_occ"].values,     SCALE_N_OCC,   ),
        ("r_val",     stats_df["r"].values,         SCALE_R,       ),
        ("abs_dist",  stats_df["abs_dist"].values,  SCALE_DIST,    ),
        ("ac1",       stats_df["ac1"].values,       SCALE_AC,      ),
        ("ac2",       stats_df["ac2"].values,       SCALE_AC,      ),
        ("mean_x",    stats_df["mean_x"].values,    SCALE_MEAN_X,  ),
        ("inv_n_occ", inv_n_occ,                    SCALE_INV_NOCC,),
    ]
    pi_enc = {}
    for key, col, scale in all_vars:
        pi_enc.update(encode_pi_func(col, f"pienc_{key}", scale=scale))
    return pd.DataFrame(pi_enc, index=stats_df.index)


# Probe-ranked: one encoded feature per source variable
ENCODED_11 = [
    "pienc_r_val_sin_pi",
    "pienc_mean_x_cos_pi",
    "pienc_std_sin_2pi",
    "pienc_var_d1_sin_pi2",
    "pienc_var_d2_sin_pi2",
    "pienc_ac1_cos_pi",
    "pienc_ac2_cos_pi",
    "pienc_n_occ_sin_pi2",
    "pienc_spread_cos_pi",
    "pienc_abs_dist_sin_pi",
    "pienc_inv_n_occ_cos_pi",
]

# Stage-0 structural encoded features
# Pi28 analog → pienc_r_val_sin_pi    (r-position; sin(π × r/R_bif))
# Pi31 analog → pienc_inv_n_occ_cos_pi (#1 importance; period/regime switch)
# Pi32 analog → pienc_abs_dist_sin_pi  (bifurcation distance encoding)
STAGE0_ENC = [
    "pienc_r_val_sin_pi",
    "pienc_inv_n_occ_cos_pi",
    "pienc_abs_dist_sin_pi",
]


def build_stage0_features(stats_df, enc_df):
    """5-feature structural space for Ridge Stage 1.
    [r, abs_dist, pienc_r_val_sin_pi, pienc_inv_n_occ_cos_pi, pienc_abs_dist_sin_pi]
    """
    return np.hstack([
        stats_df[["r", "abs_dist"]].values,
        enc_df[STAGE0_ENC].values,
    ])


# ══════════════════════════════════════════════════════════════════════════════
# CV FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(X, y, model, kf):
    """5-fold CV, returns mean R²."""
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


def run_3stage_cv(X_s0, X_enc11, y, kf):
    """5-fold CV using the 3-stage pipeline.
    Stage 1  Ridge(X_s0) — structural grammar, removes dominant λ ~ f(r) trend
    Stage 2  RF(X_enc11) → residuals — cascade dialect, bifurcation windows
    Final    λ_pred = Stage-1 + Stage-2
    """
    r2s = []
    for tr, te in kf.split(X_s0):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_s0[tr], y[tr])
        resid_tr = y[tr] - ridge.predict(X_s0[tr])

        rf = RandomForestRegressor(
            n_estimators=500, max_features="sqrt",
            min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf.fit(X_enc11[tr], resid_tr)

        y_pred = ridge.predict(X_s0[te]) + rf.predict(X_enc11[te])
        r2s.append(r2_score(y[te], y_pred))
    return float(np.mean(r2s))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# N_POOL = 2000 supports N up to ~500 without seed-overlap bias (25% rule).
# To extend to N=1000 cleanly, set N_POOL = 4000 and add 1000 to N_VALUES.
N_POOL   = 2000
N_SEEDS  = 10
N_VALUES = [11, 22, 33, 50, 66, 100, 150, 200, 300, 500]

TRACKS = ["RF raw", "XGB enc", "3-stage"]

# Reference from v4 (random sampling, 10 seeds, 5-fold CV)
PREV_BEST_RF_RAW_N100  = 0.8798
PREV_BEST_XGB_ENC_N100 = 0.8680


def main():
    t_total = time.time()

    # ── Build pool once ──────────────────────────────────────────────────────
    print(f"[INFO] building pool of {N_POOL} samples (computed once)...")
    np.random.seed(0)
    r_pool = np.random.uniform(R_MIN, R_MAX, N_POOL)

    stats_pool = sequence_statistics(r_pool)
    lyap_pool  = np.array([lyapunov_exponent(r) for r in r_pool])

    stable_mask  = lyap_pool < 0
    chaotic_mask = ~stable_mask
    print(f"[INFO] stable  (λ<0): {stable_mask.sum()}/{N_POOL}  "
          f"({100*stable_mask.mean():.1f}%)")
    print(f"[INFO] chaotic (λ≥0): {chaotic_mask.sum()}/{N_POOL}  "
          f"({100*chaotic_mask.mean():.1f}%)")

    # ── Build feature matrices ───────────────────────────────────────────────
    enc_df_pool  = build_encoded_features(stats_pool)

    raw_stat_cols = ["r", "mean_x", "std_x", "var_d1", "var_d2",
                     "ac1", "ac2", "n_occ", "spread", "dist_bifurc", "abs_dist"]
    X_raw_pool   = stats_pool[raw_stat_cols].values
    X_enc11_pool = enc_df_pool[ENCODED_11].values
    X_s0_pool    = build_stage0_features(stats_pool, enc_df_pool)

    print(f"\n[INFO] feature spaces:")
    print(f"  raw stats  : {X_raw_pool.shape[1]}")
    print(f"  ENCODED_11 : {X_enc11_pool.shape[1]}")
    print(f"  Stage-0    : {X_s0_pool.shape[1]}  "
          f"[r, abs_dist, pienc_r, pienc_inv_n, pienc_abs_dist]")
    print(f"\n[INFO] v4 reference (random, 10 seeds, 5-fold CV, n=100):")
    print(f"  RF raw : {PREV_BEST_RF_RAW_N100}   XGB enc : {PREV_BEST_XGB_ENC_N100}")
    print(f"\n[INFO] sweep n∈{N_VALUES}  seeds={N_SEEDS}  5-fold CV\n")

    kf      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {k: {n: [] for n in N_VALUES} for k in TRACKS}

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in N_VALUES:
            idx = rng.choice(N_POOL, size=n, replace=False)
            y   = lyap_pool[idx]

            # Track A — RF raw
            rf_raw = RandomForestRegressor(
                n_estimators=500, max_features="sqrt",
                min_samples_leaf=2, random_state=42, n_jobs=-1)
            results["RF raw"][n].append(
                run_cv(X_raw_pool[idx], y, rf_raw, kf))

            # Track B — XGB encoded
            xgb_enc = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0)
            results["XGB enc"][n].append(
                run_cv(X_enc11_pool[idx], y, xgb_enc, kf))

            # Track C — 3-stage pipeline
            results["3-stage"][n].append(
                run_3stage_cv(X_s0_pool[idx], X_enc11_pool[idx], y, kf))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Summary table ────────────────────────────────────────────────────────
    SEP = "=" * 76
    print(f"\n{SEP}")
    print(f"  3-STAGE PIPELINE vs BASELINES  ({N_SEEDS} seeds, 5-fold CV)")
    print(f"  Stage 1 Ridge : [r, abs_dist, pienc_r, pienc_inv_n, pienc_abs_dist]")
    print(f"  Stage 2 RF    : ENCODED_11 → Stage-1 residuals")
    print(f"{SEP}")
    print(f"  {'n':>5}  {'RF raw':>14}  {'XGB enc':>14}  {'3-stage':>14}  "
          f"{'3s > RF?':>9}  {'3s > XGB?':>10}")
    print(f"  {'─'*5}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*9}  {'─'*10}")

    wins_rf  = 0
    wins_xgb = 0

    for n in N_VALUES:
        m = {k: np.mean(results[k][n]) for k in TRACKS}
        s = {k: np.std( results[k][n]) for k in TRACKS}
        beat_rf  = m["3-stage"] > m["RF raw"]
        beat_xgb = m["3-stage"] > m["XGB enc"]
        if beat_rf:  wins_rf  += 1
        if beat_xgb: wins_xgb += 1
        prev_marker = "" if m["3-stage"] >= PREV_BEST_RF_RAW_N100 else "  "
        print(f"  {n:>5}  "
              f"{m['RF raw']:6.4f}±{s['RF raw']:.3f}  "
              f"{m['XGB enc']:6.4f}±{s['XGB enc']:.3f}  "
              f"{m['3-stage']:6.4f}±{s['3-stage']:.3f}{prev_marker}  "
              f"{'YES' if beat_rf else 'NO ':>9}  "
              f"{'YES' if beat_xgb else 'NO ':>10}")

    print(f"{'─'*76}")
    # print(f"  ✓ = 3-stage ≥ v4 RF raw ceiling @ n=100 ({PREV_BEST_RF_RAW_N100})")
    print(f"\n  3-stage > RF raw : {wins_rf}/{len(N_VALUES)}")
    print(f"  3-stage > XGB enc: {wins_xgb}/{len(N_VALUES)}")

    # ── Cross-N efficiency ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  CROSS-N EFFICIENCY — 3-stage N equivalent to baselines")
    print(f"{SEP}")
    print(f"  {'3-stage n':>10}  {'R²_3s':>8}  "
          f"{'equiv RF raw':>13}  {'equiv XGB enc':>14}  speedup vs RF")
    print(f"  {'─'*10}  {'─'*8}  {'─'*13}  {'─'*14}  {'─'*13}")

    for n_3s in [33, 50, 66, 100, 150, 200]:
        r2_3s = np.mean(results["3-stage"][n_3s])
        rf_eq  = next((n for n in N_VALUES if np.mean(results["RF raw"][n])  >= r2_3s), ">500")
        xgb_eq = next((n for n in N_VALUES if np.mean(results["XGB enc"][n]) >= r2_3s), ">500")
        speedup = (f"{rf_eq/n_3s:.1f}×" if isinstance(rf_eq, int)
                   else f">{500/n_3s:.1f}×")
        print(f"  {n_3s:>10}  {r2_3s:>8.4f}  "
              f"{str(rf_eq):>13}  {str(xgb_eq):>14}  {speedup}")

    # ── Pipeline decomposition ───────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PIPELINE CONTRIBUTION (mean ΔR² vs XGB enc, all N)")
    print(f"{SEP}")
    deltas = [np.mean(results["3-stage"][n]) - np.mean(results["XGB enc"][n])
              for n in N_VALUES]
    print(f"  Mean  Δ across all N : {np.mean(deltas):+.4f}")
    print(f"  Median Δ             : {np.median(deltas):+.4f}")
    print(f"  Min Δ  (largest N)   : {min(deltas):+.4f}  "
          f"at n={N_VALUES[deltas.index(min(deltas))]}")
    print(f"  Max Δ  (smallest N)  : {max(deltas):+.4f}  "
          f"at n={N_VALUES[deltas.index(max(deltas))]}")
    print(f"  Sign consistent      : {'YES — 3-stage always positive Δ' if all(d > 0 for d in deltas) else 'NO — mixed sign'}")

    print(f"\n[TIMING] Total: {time.time()-t_total:.1f}s")

    print(f"""
[NOTE] To extend to N=1000 cleanly:
  Set N_POOL = 4000 and add 1000 to N_VALUES.
  Pool generation will take approximately 2× longer.
  At N=1000 with pool=4000 each seed draws 25% of pool — clean.
""")


if __name__ == "__main__":
    main()


"""
python logistical_map_norm_pi_v6.py
"""
