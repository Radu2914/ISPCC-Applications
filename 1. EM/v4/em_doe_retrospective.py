import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, pairwise_distances
import xgboost as xgb
import argparse
import time

# ══════════════════════════════════════════════════════════════════════════════
# EM DOE RETROSPECTIVE — MAXIMIN SAMPLING vs RANDOM (LHS-like)
#
# Core claim: MaxiMin sample selection achieves the same R² as random (LHS)
# at 1.5–2× fewer simulation points. This challenges the literature rule
# (10×d minimum) not by changing the model but by choosing better points.
#
# Honest position from the harmonics chain:
#   Harmonics proved: encoding finds canonical structure, MaxiMin in encoded
#   space gives 1.5× efficiency. The 3-experiment chain established the method.
#
#   For EM specifically: encode_surrogate.py showed RF on raw inputs (R²=0.5441)
#   is better than any encoded model. The 57% ceiling is a physics limitation
#   (near-field resonance not fully captured by 4 inputs), not a method failure.
#   The encoding coordinate system improvement and the DoE efficiency improvement
#   are two separate claims. This file proves the DoE claim only.
#
# Probe finding (from encode_surrogate importance):
#   Gap dominates (Π2, Π11, Π12, Π13 all gap-functions = top 10 features)
#   → gap is the primary DoE axis; layers and tan_δ are secondary
#   → MaxiMin must cover the gap dimension well to work
#   → Raw 4D MaxiMin achieves this without encoding complications
#
# Two MaxiMin spaces tested:
#   4D raw      : MaxiMin in [gap, upper, lower, tan_δ] input space
#                 Directly comparable to what OptiSLang LHS does
#   5D canonical: MaxiMin in [Π2, Π13, Π6, Π9, upper/total] dimensionless space
#                 Gap-normalised, physically meaningful distances
#                 Avoids Π11 = GAP_SPREAD/gap (clips at high curvature)
#
# Models: RF only (encode_surrogate confirmed best at R²=0.5441)
# Target: log(y) → exp for R² (matches encode_surrogate.py convention)
# Protocol: fixed 40-point test set (seed=999), train from remaining 160 points
# ══════════════════════════════════════════════════════════════════════════════

LAMBDA_FREE      = 300.0 / 28.0
ER_RUBBER        = 4.5
LAMBDA_RUBBER    = LAMBDA_FREE / np.sqrt(ER_RUBBER)
NF_BOUND         = LAMBDA_FREE / (2 * np.pi)
MODULE_Y         = 46.53
DIEL_BIFURCATION = 0.107
EPS = 1e-9; PI = np.pi; E = np.e

N_VALUES  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
N_SEEDS   = 10
TEST_SIZE = 40
TEST_SEED = 999


def encode_pi_func(x, prefix, scale, weights=(5, 1, 1, 3, 1)):
    """Pi-encode: clip then normalise by physics-grounded scale (not data max)."""
    x  = np.clip(x, 0, scale)           # clip at scale, not at 10
    xn = x / (scale + EPS)              # xn ∈ [0, 1]
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2 * PI * xn),
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }


def encode_e_func(x, prefix, scale, weights=(2, 2, 1)):
    """E-encode: clip then normalise by physics-grounded scale."""
    x  = np.clip(x, 0, scale)
    xn = x / (scale + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-E * xn),
        f"{prefix}_pow_e":   w[1] * xn ** E,
        f"{prefix}_gauss":   w[2] * np.exp(-E * (xn - 0.5)**2),
    }


def load_data(path):
    return pd.read_csv(
        path, sep=r'\s+', comment='#', header=None,
        names=["index","gap","upper_protective_layer","lower_protective_layer",
               "protective_layer_dielectric","variable_E","variable_H","variable_Power",
               "constr_variable_E","constr_variable_H","constr_variable_Power",
               "obj_variable_Power"])


def build_canon_em(df):
    """
    5D canonical dimensionless space for MaxiMin distance metric.
    All variables well-behaved (no Π11=GAP_SPREAD/gap — diverges at small gap).

    Scale choices — physically grounded, not data-derived:
      Π2  = gap/NF_bound    → scale = 10  (max ~8.2; gap up to ~14mm, NF=1.7mm)
      Π13 = solid_angle     → scale = 0.1 (max ~0.083; tiny but monotone in gap)
      Π6  = total_et        → scale = 1.2 (max layer = 1.18λ_rubber confirmed)
      Π9  = tan_δ           → scale = 0.2 (DoE range [0.021, 0.19])
      upper_frac = u/(u+l)  → scale = 1.0 (bounded [0,1] by definition)

    Upper fraction captures the asymmetry between layers — not in top importance
    but provides geometric variety for MaxiMin that total_et alone misses.
    """
    g   = df["gap"].values
    upl = df["upper_protective_layer"].values
    low = df["lower_protective_layer"].values
    td  = df["protective_layer_dielectric"].values
    tot = upl + low

    pi2  = g / NF_BOUND                           # gap in NF lengths
    pi13 = g**2 / (g**2 + MODULE_Y**2 + EPS)      # solid angle (bounded)
    pi6  = tot / LAMBDA_RUBBER                    # total electrical thickness
    pi9  = td                                     # loss tangent directly
    ufrac = upl / (tot + EPS)                     # upper layer fraction [0,1]

    enc = {}
    enc.update(encode_pi_func(pi2,   "can_gap_nf",   scale=10.0))
    enc.update(encode_e_func( pi13,  "can_solidang",  scale=0.1))
    enc.update(encode_e_func( pi6,   "can_total_et",  scale=1.2))
    enc.update(encode_pi_func(pi9,   "can_tand",      scale=0.2))
    enc.update(encode_e_func( ufrac, "can_upfrac",    scale=1.0))

    return pd.DataFrame(enc).values


def maximin_select(D, n_select, seed=0):
    """Greedy MaxiMin — identical to Exp3."""
    rng = np.random.default_rng(seed)
    n_pool = D.shape[0]
    first  = int(rng.integers(0, n_pool))
    selected = [first]
    sel_mask = np.zeros(n_pool, dtype=bool); sel_mask[first] = True
    min_dists = D[first].copy().astype(float); min_dists[first] = -np.inf
    for _ in range(n_select - 1):
        cands = np.where(~sel_mask, min_dists, -np.inf)
        nxt   = int(np.argmax(cands))
        selected.append(nxt); sel_mask[nxt] = True
        np.minimum(min_dists, D[nxt], out=min_dists); min_dists[nxt] = -np.inf
    return np.array(selected)


def make_rf():
    """RF config from encode_surrogate.py — confirmed best model."""
    return RandomForestRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)


def eval_r2(y_true, y_log_pred):
    """R² on original scale (exponentiate log prediction)."""
    return r2_score(y_true, np.exp(y_log_pred) - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    args = vars(ap.parse_args())
    t_total = time.time()

    print("[INFO] loading EM dataset...")
    df    = load_data(args["dataset"])
    N_ALL = len(df)
    y     = df["obj_variable_Power"].values
    y_log = np.log(y + EPS)
    print(f"[INFO] {N_ALL} HFSS design points")
    print(f"[INFO] y range=[{y.min():.3f}, {y.max():.3f}] W/m²")
    print(f"[INFO] encode_surrogate best (5-fold CV, full 200): R²=0.5514")

    input_cols = ["gap","upper_protective_layer","lower_protective_layer",
                  "protective_layer_dielectric"]
    X_raw   = df[input_cols].values
    X_canon = build_canon_em(df)

    print(f"\n[INFO] Feature spaces for MaxiMin distance metric:")
    print(f"  RAW-4    : {X_raw.shape[1]}D  [gap, upper, lower, tan_δ]")
    print(f"  CANON-5  : {X_canon.shape[1]}  (Π2, Π13, Π6, Π9, upper_frac — encoded)")

    # Fixed test set
    rng_test  = np.random.default_rng(TEST_SEED)
    test_idx  = rng_test.choice(N_ALL, size=TEST_SIZE, replace=False)
    test_mask = np.zeros(N_ALL, dtype=bool); test_mask[test_idx] = True
    pool_idx  = np.where(~test_mask)[0]; N_POOL = len(pool_idx)

    X_raw_test   = X_raw[test_idx];    X_raw_pool   = X_raw[pool_idx]
    X_can_test   = X_canon[test_idx];  X_can_pool   = X_canon[pool_idx]
    y_test       = y[test_idx]
    y_log_pool   = y_log[pool_idx]

    print(f"\n[INFO] Fixed test set  : {TEST_SIZE} points (seed={TEST_SEED})")
    print(f"[INFO] Training pool   : {N_POOL} points")
    print(f"[INFO] Literature min  : 10×d = {10*4} pts (raw 4D)")
    print(f"[INFO] Model           : RF (n_estimators=500) — encode_surrogate best")

    # Pre-compute distance matrices
    print(f"\n[INFO] computing distance matrices...")
    t0 = time.time()
    D_raw   = pairwise_distances(X_raw_pool)
    D_canon = pairwise_distances(X_can_pool)
    print(f"[TIMING] raw {D_raw.shape} + canon {D_canon.shape}: {time.time()-t0:.2f}s")

    print(f"\n[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds\n")

    METHODS = ["A_raw_rand", "B_raw_mm4d", "C_canon_mm", "D_raw_mm4d_2s"]
    res     = {m: {n: [] for n in N_VALUES} for m in METHODS}
    valid_n = [n for n in N_VALUES if n <= N_POOL]

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in valid_n:
            rand_idx   = rng.choice(N_POOL, size=n, replace=False)
            mm4d_idx   = maximin_select(D_raw,   n, seed=seed)
            mm_can_idx = maximin_select(D_canon,  n, seed=seed)

            # A — RAW-4 random RF (LHS-like baseline; matches literature standard)
            m = make_rf()
            m.fit(X_raw_pool[rand_idx], y_log_pool[rand_idx])
            res["A_raw_rand"][n].append(eval_r2(y_test, m.predict(X_raw_test)))

            # B — RAW-4 MaxiMin RF (structural DoE in raw input space)
            m = make_rf()
            m.fit(X_raw_pool[mm4d_idx], y_log_pool[mm4d_idx])
            res["B_raw_mm4d"][n].append(eval_r2(y_test, m.predict(X_raw_test)))

            # C — CANON-5 MaxiMin RF (structural DoE in dimensionless space)
            m = make_rf()
            m.fit(X_raw_pool[mm_can_idx], y_log_pool[mm_can_idx])
            res["C_canon_mm"][n].append(eval_r2(y_test, m.predict(X_raw_test)))

            # D — RAW-4 MaxiMin + two-stage (Ridge grammar + RF residual)
            s1 = Ridge(alpha=1.0)
            s1.fit(X_raw_pool[mm4d_idx], y_log_pool[mm4d_idx])
            resid = y_log_pool[mm4d_idx] - s1.predict(X_raw_pool[mm4d_idx])
            s2 = make_rf()
            s2.fit(X_raw_pool[mm4d_idx], resid)
            y_hat = s1.predict(X_raw_test) + s2.predict(X_raw_test)
            res["D_raw_mm4d_2s"][n].append(eval_r2(y_test, y_hat))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Results table ──────────────────────────────────────────────────────
    W = 108
    print(f"\n{'='*W}")
    print(f"  EM DOE RETROSPECTIVE — {N_ALL} HFSS points  |  RF (n=500)  |  "
          f"fixed {TEST_SIZE}-pt test  |  {N_SEEDS} seeds")
    print(f"  encode_surrogate reference (5-fold CV on all 200): R²=0.5514 (previous best)")
    print(f"{'='*W}")
    print(f"  {'n':>5}  {'RAW rand (A)':>14}  {'Raw MM-4D (B)':>15}  "
          f"{'Canon MM (C)':>14}  {'MM+2stg (D)':>13}  best  MM4D gain vs rand")
    print(f"  {'─'*5}  {'─'*14}  {'─'*15}  {'─'*14}  {'─'*13}  {'─'*5}  {'─'*20}")

    for n in valid_n:
        μ = {m: np.mean(res[m][n]) for m in METHODS}
        σ = {m: np.std( res[m][n]) for m in METHODS}
        best = max(μ, key=μ.get)
        mm_gain = μ["B_raw_mm4d"] - μ["A_raw_rand"]
        lbl = {"A_raw_rand":"A","B_raw_mm4d":"B","C_canon_mm":"C","D_raw_mm4d_2s":"D"}
        lit = "  ← 10×d literature min" if n == 40 else ""
        print(f"  {n:>5}  "
              f"{μ['A_raw_rand']:>6.4f}±{σ['A_raw_rand']:.3f}  "
              f"{μ['B_raw_mm4d']:>7.4f}±{σ['B_raw_mm4d']:.3f}  "
              f"{μ['C_canon_mm']:>6.4f}±{σ['C_canon_mm']:.3f}  "
              f"{μ['D_raw_mm4d_2s']:>5.4f}±{σ['D_raw_mm4d_2s']:.3f}  "
              f"  {lbl[best]}    Δ={mm_gain:+.4f}{lit}")

    wins_b = sum(np.mean(res["B_raw_mm4d"][n]) > np.mean(res["A_raw_rand"][n]) for n in valid_n)
    wins_c = sum(np.mean(res["C_canon_mm"][n]) > np.mean(res["A_raw_rand"][n]) for n in valid_n)
    print(f"{'='*W}")
    print(f"  Raw MaxiMin-4D vs random  : wins at {wins_b}/{len(valid_n)} N values")
    print(f"  Canon MaxiMin vs random   : wins at {wins_c}/{len(valid_n)} N values")

    # ── Crossover table ────────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  CROSSOVER — Raw MaxiMin-N equivalent to random at which N?")
    print(f"  Claim: MaxiMin selects better design points → fewer runs needed")
    print(f"{'─'*W}")
    print(f"  {'MaxiMin N':>10}  {'MM R²':>8}  {'Rand R²':>9}  "
          f"{'rand equiv':>11}  efficiency  DoE note")
    print(f"  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*11}  {'─'*10}  {'─'*25}")

    rand_means = {n: np.mean(res["A_raw_rand"][n]) for n in valid_n}
    mm_means   = {n: np.mean(res["B_raw_mm4d"][n]) for n in valid_n}

    for n in valid_n:
        mm_r2  = mm_means[n]; rand_r2 = rand_means[n]
        rand_equiv = next((rn for rn in valid_n if rand_means[rn] >= mm_r2), None)
        ratio = rand_equiv / n if rand_equiv else float("inf")
        eff   = f"{ratio:.1f}×" if np.isfinite(ratio) else "> max"
        note  = "← 10×d literature min" if n == 40 else ""
        rand_str = str(rand_equiv) if rand_equiv else "> max"
        print(f"  {n:>10}  {mm_r2:>8.4f}  {rand_r2:>9.4f}  "
              f"{rand_str:>11}  {eff:>7} fewer  {note}")

    # ── Safety bias check ──────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  SAFETY BIAS CHECK — does MaxiMin cover high-SAR points?")
    pool_high = np.sum(y[pool_idx] >= 2.0)
    pool_frac = pool_high / N_POOL
    print(f"  High-SAR (≥2 W/m²) in pool: {pool_high}/{N_POOL} = {pool_frac*100:.0f}%")
    print(f"  {'n':>5}  {'high-SAR in MM':>16}  bias vs pool")
    for n in valid_n[:6]:
        fracs = [np.sum(y[pool_idx][maximin_select(D_raw, n, seed=s)] >= 2.0)/n
                 for s in range(N_SEEDS)]
        mf = np.mean(fracs)
        bias = "balanced ✓" if abs(mf - pool_frac) < 0.12 else \
               "under-covers high-SAR ⚠" if mf < pool_frac - 0.12 else "over-covers"
        print(f"  {n:>5}  {mf*100:>10.0f}% (pool={pool_frac*100:.0f}%)  {bias}")

    print(f"\n[TIMING] Total: {time.time()-t_total:.1f}s")

if __name__ == "__main__":
    main()

"""
python em_doe_retrospective.py --dataset last_run_designs.csv
"""
