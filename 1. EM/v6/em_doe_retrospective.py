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
# Combined: em_doe_retrospective.py + geometric_pi_groups.py
#
# Three MaxiMin spaces compared:
#   4D raw      : [gap, upper, lower, tan_δ] — directly comparable to LHS
#   5D canonical: [Π2, Π13, Π6, Π9, upper_frac] — physically normalised
#   3D geometric: [Π28, Π31, Π32] — waveguide regime triangle
#                  path_competition × regime_switch × path_interference
#                  selects designs maximally spread in EM regime space
#
# Seven methods:
#   A  raw random RF              — LHS-like baseline
#   B  raw MaxiMin-4D RF          — structural DoE in raw input space
#   C  canon MaxiMin-5D RF        — dimensionless canonical space
#   D  raw MaxiMin-4D + 2stg RF   — Ridge grammar + RF residual
#   E  geom MaxiMin-3D RF         — geometric regime space only
#   F  geom MaxiMin-3D + 3stg XGB — geom selection + Ridge(raw+geom) + XGB residual
#                                    (tests whether XGB beats RF in Stage 2)
#   G  7D MaxiMin + 3stg RF       — intentional: [raw-4D + Pi28+Pi31+Pi32] selection
#                                    + Ridge(raw+geom) + RF residual
#                                    fixes E/F layer-blindness; adds regime axis to D
#
# F vs D: same selection (raw MaxiMin... actually F uses geom) → isolates XGB vs RF
# G vs D: same model (Ridge+RF) → isolates 7D MaxiMin vs raw-4D MaxiMin
# G vs F: same selection (geom+raw coverage) → isolates RF vs XGB in Stage 2
#
# Protocol: fixed 40-pt test (seed=999), train from remaining 160 points
# Model: RF (n=500) — confirmed best in encode_surrogate.py
# Target: log(y) → exp for R²
# ══════════════════════════════════════════════════════════════════════════════

# ── EM constants ──────────────────────────────────────────────────────────────
LAMBDA_FREE      = 300.0 / 28.0          # 10.714 mm
ER_RUBBER        = 4.5
LAMBDA_RUBBER    = LAMBDA_FREE / np.sqrt(ER_RUBBER)   # 5.051 mm
NF_BOUND         = LAMBDA_FREE / (2 * np.pi)           # 1.705 mm
MODULE_Y         = 46.53
DIEL_BIFURCATION = 0.107

# ── Waveguide triangle — confirmed from HFSS coordinate references ─────────────
# 5G module    : (-27.98, -46.62, -32.00) mm
# Nose ridge   : ( -7.40,  -0.16,  -3.46) mm  ← direct path endpoint
# Brow ridge   : ( -0.90, -49.26, -14.91) mm  ← waveguide deflector
#
# d_AC / λ_free = 2.999 ≈ 3.000 — EXACTLY 3λ at 28GHz (engineered waveguide)
# Angle at brow = 86.14° — near-perfect right-angle deflector (cos=0.0673)
# PATH_RATIO = d_AC/d_AB = 0.551 — waveguide is 55% of direct path length
# DIEL_BIFURCATION = 0.107 — geometrically confirmed: Pi31=PATH_RATIO at this tan_δ
D_MODULE_NOSE  = 58.280   # mm — module to nose ridge
D_MODULE_BROW  = 32.130   # mm — module to brow ridge  (= 3.000λ at 28GHz)
D_NOSE_BROW    = 50.835   # mm — nose to brow aperture span
COS_BROW       = 0.06732  # cos(86.14°) — near-perpendicular deflector
PATH_RATIO     = D_MODULE_BROW / D_MODULE_NOSE   # = 0.55131

EPS = 1e-9
PI  = np.pi
E   = np.e

N_VALUES  = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
             110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
N_SEEDS   = 10
TEST_SIZE = 40
TEST_SEED = 999


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING — physics-grounded scale (not data max)
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, scale, weights=(5, 1, 1, 3, 1)):
    x  = np.clip(x, 0, scale)
    xn = x / (scale + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2 * PI * xn),
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }


def encode_e_func(x, prefix, scale, weights=(2, 2, 1)):
    x  = np.clip(x, 0, scale)
    xn = x / (scale + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-E * xn),
        f"{prefix}_pow_e":   w[1] * xn ** E,
        f"{prefix}_gauss":   w[2] * np.exp(-E * (xn - 0.5)**2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(path):
    return pd.read_csv(
        path, sep=r'\s+', comment='#', header=None,
        names=["index", "gap", "upper_protective_layer", "lower_protective_layer",
               "protective_layer_dielectric", "variable_E", "variable_H",
               "variable_Power", "constr_variable_E", "constr_variable_H",
               "constr_variable_Power", "obj_variable_Power"])


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL SPACE — 5D, from em_doe_retrospective.py
# [Π2, Π13, Π6, Π9, upper_frac] — physically normalised, no clipping issues
# ══════════════════════════════════════════════════════════════════════════════

def build_canon_em(df):
    """
    5D canonical dimensionless space.
    Avoids Π11=GAP_SPREAD/gap (diverges at small gap).
    Scale choices physically grounded:
      Π2  = gap/NF_bound  → scale=10   (max ~8.2)
      Π13 = solid_angle   → scale=0.1  (max ~0.083)
      Π6  = total_et      → scale=1.2  (max 1.18λ_rubber)
      Π9  = tan_δ         → scale=0.2  (DoE range [0.021, 0.19])
      upper_frac          → scale=1.0  (bounded [0,1])
    """
    g    = df["gap"].values
    upl  = df["upper_protective_layer"].values
    low  = df["lower_protective_layer"].values
    td   = df["protective_layer_dielectric"].values
    tot  = upl + low
    enc  = {}
    enc.update(encode_pi_func(g / NF_BOUND,
                              "can_gap_nf",   scale=10.0))
    enc.update(encode_e_func( g**2 / (g**2 + MODULE_Y**2 + EPS),
                              "can_solidang", scale=0.1))
    enc.update(encode_e_func( tot / LAMBDA_RUBBER,
                              "can_total_et", scale=1.2))
    enc.update(encode_pi_func(td,
                              "can_tand",     scale=0.2))
    enc.update(encode_e_func( upl / (tot + EPS),
                              "can_upfrac",   scale=1.0))
    return pd.DataFrame(enc).values


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRIC Pi GROUPS — from geometric_pi_groups.py
#
# Encodes the waveguide/direct-path regime switch as three variables:
#   Π28 (path_competition): d_AC/(d_AB+gap) — varies 0.458→0.551 with gap
#                           cascading↓ as gap increases
#   Π31 (regime_switch):    PATH_RATIO/(tan_δ/DIEL_BIFURC)
#                           cascading: 2.36 (waveguide) → 0.31 (direct) with tan_δ
#                           = PATH_RATIO exactly at bifurcation point
#   Π32 (path_diff_wl):     (d_AB - d_AC + gap)/λ_free
#                           interference condition between paths, grows↑ with gap
#
# MaxiMin in this 3D space selects designs maximally spread in regime terms:
# one design with geometry-active low-loss (Pi31 high, Pi28 high),
# one with direct-path high-loss (Pi31 low, Pi32 large), etc.
# This is structural coverage of the EM regime space, not just geometric spread.
# ══════════════════════════════════════════════════════════════════════════════

def build_geom_pi(df):
    """
    Raw Π28, Π31, Π32 values for MaxiMin distance computation.
    Normalised to [0,1] per variable so all three dimensions contribute equally.
    Returns (n_points, 3) array.
    """
    g  = df["gap"].values
    td = df["protective_layer_dielectric"].values

    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE

    # Min-max normalise so distances are comparable across the three axes
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + EPS)

    return np.column_stack([norm(pi28), norm(pi31), norm(pi32)])


def build_geom_encoded(df):
    """
    Pi-encoded Π28, Π31, Π32 — for use as training features if wanted.
    Pi-encode all three: all are cascading / non-periodic variables.
    Returns (n_points, 15) array.
    """
    g  = df["gap"].values
    td = df["protective_layer_dielectric"].values

    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE

    # Cross-products from geometric_pi_groups.py
    pi28_n = pi28 / (pi28.max() + EPS)
    pi31_n = np.clip(pi31 / 3.0, 0, 1)     # max Pi31 ~ 2.77, scale by 3
    pi32_n = pi32 / (pi32.max() + EPS)

    enc = {}
    enc.update(encode_pi_func(pi28, "pienc_path_comp",  scale=pi28.max()))
    enc.update(encode_pi_func(pi31, "pienc_regime_sw",  scale=max(pi31.max(), 3.0)))
    enc.update(encode_pi_func(pi32, "pienc_path_diff",  scale=pi32.max()))

    # Cross 1: waveguide activation — high gap small AND tan_δ low
    enc["cross_wg_activation"] = (
        np.sin(PI * np.clip(pi28_n, 0, 1)) *
        np.exp(-E * np.clip(1 - pi31_n, 0, 1))
    )

    # Cross 2: direct dominance — gap large AND tan_δ high
    enc["cross_direct_dom"] = (
        np.sin(PI * pi32_n) *
        (td / (td.max() + EPS))
    )

    return pd.DataFrame(enc).values


# ══════════════════════════════════════════════════════════════════════════════
# MAXIMIN — greedy, O(N_pool × n_select), same as Exp3
# ══════════════════════════════════════════════════════════════════════════════

def maximin_select(D, n_select, seed=0):
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


# ══════════════════════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════════════════════

def make_rf():
    return RandomForestRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0)


def geom_stage1_features(X_raw):
    """
    Stage 1 (Ridge) feature set: raw 4D + Pi28 + Pi31 + Pi32.
    X_raw columns: [0=gap, 1=upper, 2=lower, 3=tan_δ]
    Appending the three geometric Pi groups makes the waveguide
    regime switch explicit to Ridge, removing it from the residuals
    before Stage 2 sees them.
    """
    g    = X_raw[:, 0]
    td   = X_raw[:, 3]
    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO    / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE
    return np.column_stack([X_raw, pi28, pi31, pi32])


def build_7d_maximin(X_raw):
    """
    7D IntentionalMaxiMin space: [gap, upper, lower, tan_δ, Pi28, Pi31, Pi32]
    all normalised to [0,1] using pool statistics.

    Why 7D beats geom-3D (E, F):
      Geom-3D only sees gap and tan_δ — upper/lower layers are invisible.
      MaxiMin in 3D can select points clustered in layer-thickness space,
      giving Stage 2 poor resonance coverage.

    Why 7D beats raw-4D (D):
      Raw-4D covers layers and gap but has no regime axis.
      Two points far apart in input space can share the same Pi31 value
      (both in the transition zone), missing waveguide coverage.

    7D ensures simultaneous coverage of:
      raw 4D → layers (resonance) + full input range (Stage 1 grammar)
      Pi28   → path competition (gap-dependent waveguide activation)
      Pi31   → regime switch (tan_δ-dependent — the key missing axis)
      Pi32   → path interference (gap-dependent constructive/destructive)
    """
    g    = X_raw[:, 0]
    td   = X_raw[:, 3]
    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO    / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + EPS)

    return np.column_stack([
        norm(X_raw[:, 0]),          # gap
        norm(X_raw[:, 1]),          # upper layer
        norm(X_raw[:, 2]),          # lower layer
        norm(X_raw[:, 3]),          # tan_δ
        norm(pi28), norm(pi31), norm(pi32),
    ])


def eval_r2(y_true, y_log_pred):
    return r2_score(y_true, np.exp(y_log_pred) - EPS)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

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

    # Waveguide geometry summary
    print(f"\n[INFO] Waveguide triangle geometry:")
    print(f"  d_module→nose = {D_MODULE_NOSE:.3f} mm  ({D_MODULE_NOSE/LAMBDA_FREE:.3f}λ)")
    print(f"  d_module→brow = {D_MODULE_BROW:.3f} mm  ({D_MODULE_BROW/LAMBDA_FREE:.4f}λ ≈ 3λ ← engineered)")
    print(f"  Brow angle    = 86.14°  cos={COS_BROW:.4f}  (near-perfect deflector)")
    print(f"  PATH_RATIO    = {PATH_RATIO:.5f}")
    print(f"  Pi31 at DIEL_BIFURC={DIEL_BIFURCATION}: {PATH_RATIO/(1.0):.5f} = PATH_RATIO ✓")

    # ── Feature matrices ───────────────────────────────────────────────────
    input_cols = ["gap", "upper_protective_layer",
                  "lower_protective_layer", "protective_layer_dielectric"]
    X_raw   = df[input_cols].values
    X_canon = build_canon_em(df)
    X_geom  = build_geom_pi(df)       # 3D normalised [Pi28, Pi31, Pi32] for MaxiMin

    print(f"\n[INFO] MaxiMin distance spaces:")
    print(f"  RAW-4    : {X_raw.shape[1]}D  [gap, upper, lower, tan_δ]")
    print(f"  CANON-5  : {X_canon.shape[1]}D  (Π2, Π13, Π6, Π9, upper_frac — encoded)")
    print(f"  GEOM-3   : {X_geom.shape[1]}D  (Π28 path_comp, Π31 regime_sw, Π32 path_diff)")
    print(f"             selects designs spread across waveguide regime space")

    # ── Fixed test set ─────────────────────────────────────────────────────
    rng_test  = np.random.default_rng(TEST_SEED)
    test_idx  = rng_test.choice(N_ALL, size=TEST_SIZE, replace=False)
    test_mask = np.zeros(N_ALL, dtype=bool); test_mask[test_idx] = True
    pool_idx  = np.where(~test_mask)[0]; N_POOL = len(pool_idx)

    X_raw_test  = X_raw[test_idx];   X_raw_pool  = X_raw[pool_idx]
    X_geom_test = X_geom[test_idx];  X_geom_pool = X_geom[pool_idx]
    y_test      = y[test_idx]
    y_log_pool  = y_log[pool_idx]

    # Canon only needed for MaxiMin distances (train always on raw)
    X_can_pool  = X_canon[pool_idx]

    print(f"\n[INFO] Fixed test set  : {TEST_SIZE} points (seed={TEST_SEED})")
    print(f"[INFO] Training pool   : {N_POOL} points")
    print(f"[INFO] Literature min  : 10×d = {10*4} pts (raw 4D)")
    print(f"[INFO] Model           : RF (n_estimators=500) — encode_surrogate best")
    print(f"[INFO] All methods train on raw 4D inputs; MaxiMin space controls selection only")

    # ── Pre-compute all distance matrices ──────────────────────────────────
    print(f"\n[INFO] computing distance matrices...")
    t0 = time.time()
    D_raw   = pairwise_distances(X_raw_pool)
    D_canon = pairwise_distances(X_can_pool)
    D_geom  = pairwise_distances(X_geom_pool)
    X_7d_pool = build_7d_maximin(X_raw_pool)   # 7D intentional space
    D_7d    = pairwise_distances(X_7d_pool)
    print(f"[TIMING] raw{D_raw.shape} + canon{D_canon.shape} + "
          f"geom{D_geom.shape} + 7d{D_7d.shape}: {time.time()-t0:.2f}s")

    print(f"\n[INFO] Pi31 (regime switch) in pool:")
    g_pool = df["gap"].values[pool_idx]
    td_pool = df["protective_layer_dielectric"].values[pool_idx]
    pi31_pool = PATH_RATIO / (td_pool / DIEL_BIFURCATION + EPS)
    print(f"  range=[{pi31_pool.min():.3f}, {pi31_pool.max():.3f}]  "
          f"(low=direct-path dominant, high=waveguide dominant)")

    print(f"\n[INFO] sweep n∈{N_VALUES}, {N_SEEDS} seeds\n")

    # ── Sweep ──────────────────────────────────────────────────────────────
    METHODS = ["A_raw_rand", "B_raw_mm4d", "C_canon_mm",
               "D_raw_mm4d_2s", "E_geom_mm",
               "F_geom_3s_xgb", "G_7d_3s_rf"]
    res     = {m: {n: [] for n in N_VALUES} for m in METHODS}
    valid_n = [n for n in N_VALUES if n <= N_POOL]

    # Stage 1 test features — constant across seeds and N
    X_s1_test = geom_stage1_features(X_raw_test)

    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        for n in valid_n:
            rand_idx   = rng.choice(N_POOL, size=n, replace=False)
            mm4d_idx   = maximin_select(D_raw,   n, seed=seed)
            mm_can_idx = maximin_select(D_canon,  n, seed=seed)
            mm_geo_idx = maximin_select(D_geom,   n, seed=seed)
            mm_7d_idx  = maximin_select(D_7d,     n, seed=seed)

            # A — RAW-4 random RF
            m = make_rf()
            m.fit(X_raw_pool[rand_idx], y_log_pool[rand_idx])
            res["A_raw_rand"][n].append(eval_r2(y_test, m.predict(X_raw_test)))

            # B — RAW-4 MaxiMin RF
            m = make_rf()
            m.fit(X_raw_pool[mm4d_idx], y_log_pool[mm4d_idx])
            res["B_raw_mm4d"][n].append(eval_r2(y_test, m.predict(X_raw_test)))

            # C — CANON-5 MaxiMin RF (selection in canonical space, train on raw)
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

            # E — GEOM-3 MaxiMin RF
            m = make_rf()
            m.fit(X_raw_pool[mm_geo_idx], y_log_pool[mm_geo_idx])
            res["E_geom_mm"][n].append(eval_r2(y_test, m.predict(X_raw_test)))

            # F — GEOM-3 MaxiMin + three-stage XGB
            #   Exposes XGB-vs-RF question in Stage 2
            #   Still suffers from layer-blindness in selection
            X_s1_tr = geom_stage1_features(X_raw_pool[mm_geo_idx])
            f1 = Ridge(alpha=1.0)
            f1.fit(X_s1_tr, y_log_pool[mm_geo_idx])
            resid_f = y_log_pool[mm_geo_idx] - f1.predict(X_s1_tr)
            f2 = make_xgb()
            f2.fit(X_raw_pool[mm_geo_idx], resid_f)
            y_hat_f = f1.predict(X_s1_test) + f2.predict(X_raw_test)
            res["F_geom_3s_xgb"][n].append(eval_r2(y_test, y_hat_f))

            # G — 7D IntentionalMaxiMin + three-stage RF
            #   Hypothesis: 7D MaxiMin fixes layer-blindness of E/F
            #   while adding regime axis missing from D.
            #   Ridge sees (raw 4D + Pi28 + Pi31 + Pi32) — same as F.
            #   RF on residuals — same as D.
            #   If G > D: 7D regime coverage adds value over raw-4D.
            #   If G > F: RF beats XGB for Stage 2 (confirms result).
            X_s1_tr7 = geom_stage1_features(X_raw_pool[mm_7d_idx])
            g1 = Ridge(alpha=1.0)
            g1.fit(X_s1_tr7, y_log_pool[mm_7d_idx])
            resid_g = y_log_pool[mm_7d_idx] - g1.predict(X_s1_tr7)
            g2 = make_rf()
            g2.fit(X_raw_pool[mm_7d_idx], resid_g)
            y_hat_g = g1.predict(X_s1_test) + g2.predict(X_raw_test)
            res["G_7d_3s_rf"][n].append(eval_r2(y_test, y_hat_g))

        print(f"  seed {seed+1}/{N_SEEDS} done")

    # ── Results table ──────────────────────────────────────────────────────
    W = 148
    print(f"\n{'='*W}")
    print(f"  EM DOE RETROSPECTIVE — {N_ALL} HFSS points  |  "
          f"fixed {TEST_SIZE}-pt test  |  {N_SEEDS} seeds")
    print(f"  Reference: encode_surrogate 5-fold CV all 200 pts → R²=0.5514")
    print(f"  F = geom-3D MaxiMin + Ridge(raw+geom) + XGB residual  "
          f"[isolates: XGB vs RF in Stage 2]")
    print(f"  G = 7D MaxiMin + Ridge(raw+geom) + RF residual         "
          f"[isolates: 7D vs raw-4D MaxiMin, fixes layer-blindness of E/F]")
    print(f"{'='*W}")
    print(f"  {'n':>5}  {'A rand RF':>12}  {'B MM-4D RF':>12}  {'C canon RF':>12}  "
          f"{'D MM+2s RF':>12}  {'E geom RF':>12}  {'F geo+XGB':>12}  "
          f"{'G 7D+RF':>12}  best")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}  {'─'*12}  "
          f"{'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*4}")

    lbl = {"A_raw_rand":"A","B_raw_mm4d":"B","C_canon_mm":"C",
           "D_raw_mm4d_2s":"D","E_geom_mm":"E",
           "F_geom_3s_xgb":"F","G_7d_3s_rf":"G"}

    for n in valid_n:
        μ = {m: np.mean(res[m][n]) for m in METHODS}
        σ = {m: np.std( res[m][n]) for m in METHODS}
        best = max(μ, key=μ.get)
        lit  = "  ← 10×d" if n == 40 else ""
        prev = " ← >full200" if μ[best] > 0.5514 else ""
        print(f"  {n:>5}  "
              f"{μ['A_raw_rand']:>4.4f}±{σ['A_raw_rand']:.3f}  "
              f"{μ['B_raw_mm4d']:>4.4f}±{σ['B_raw_mm4d']:.3f}  "
              f"{μ['C_canon_mm']:>4.4f}±{σ['C_canon_mm']:.3f}  "
              f"{μ['D_raw_mm4d_2s']:>4.4f}±{σ['D_raw_mm4d_2s']:.3f}  "
              f"{μ['E_geom_mm']:>4.4f}±{σ['E_geom_mm']:.3f}  "
              f"{μ['F_geom_3s_xgb']:>4.4f}±{σ['F_geom_3s_xgb']:.3f}  "
              f"{μ['G_7d_3s_rf']:>4.4f}±{σ['G_7d_3s_rf']:.3f}  "
              f"  {lbl[best]}{lit}{prev}")

    for m, label in [("B_raw_mm4d",    "Raw MM-4D RF   "),
                     ("C_canon_mm",    "Canon MM-5D RF "),
                     ("D_raw_mm4d_2s", "MM+2stg RF     "),
                     ("E_geom_mm",     "Geom MM-3D RF  "),
                     ("F_geom_3s_xgb", "Geom 3stg XGB  "),
                     ("G_7d_3s_rf",    "7D 3stg RF     ")]:
        wins = sum(np.mean(res[m][n]) > np.mean(res["A_raw_rand"][n]) for n in valid_n)
        g_vs_d = sum(np.mean(res["G_7d_3s_rf"][n]) > np.mean(res["D_raw_mm4d_2s"][n])
                     for n in valid_n) if m == "G_7d_3s_rf" else None
        extra = f"  G>D: {g_vs_d}/{len(valid_n)}" if g_vs_d is not None else ""
        print(f"  {label} vs random: wins {wins}/{len(valid_n)}{extra}")
    print(f"{'='*W}")

    # ── Crossover table — D, F, G vs random ───────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  CROSSOVER — method-N equivalent to random RF at which N?")
    print(f"  (D) raw-4D MaxiMin + Ridge+RF  "
          f"(F) geom-3D + Ridge+XGB  (G) 7D MaxiMin + Ridge+RF")
    print(f"{'─'*W}")
    print(f"  {'n':>5}  {'D R²':>8}  {'F R²':>8}  {'G R²':>8}  "
          f"{'Rand R²':>8}  {'D→rand':>10}  {'F→rand':>10}  {'G→rand':>10}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}")

    rand_means = {n: np.mean(res["A_raw_rand"][n]) for n in valid_n}
    d_means    = {n: np.mean(res["D_raw_mm4d_2s"][n]) for n in valid_n}
    f_means    = {n: np.mean(res["F_geom_3s_xgb"][n]) for n in valid_n}
    g_means    = {n: np.mean(res["G_7d_3s_rf"][n]) for n in valid_n}

    for n in valid_n:
        d_r2 = d_means[n]; f_r2 = f_means[n]
        g_r2 = g_means[n]; ra   = rand_means[n]
        d_eq = next((rn for rn in valid_n if rand_means[rn] >= d_r2), None)
        f_eq = next((rn for rn in valid_n if rand_means[rn] >= f_r2), None)
        g_eq = next((rn for rn in valid_n if rand_means[rn] >= g_r2), None)
        d_s  = f"{d_eq}({d_eq/n:.1f}×)" if d_eq else ">max"
        f_s  = f"{f_eq}({f_eq/n:.1f}×)" if f_eq else ">max"
        g_s  = f"{g_eq}({g_eq/n:.1f}×)" if g_eq else ">max"
        lit  = " ←10×d" if n == 40 else ""
        print(f"  {n:>5}  {d_r2:>8.4f}  {f_r2:>8.4f}  {g_r2:>8.4f}  "
              f"{ra:>8.4f}  {d_s:>10}  {f_s:>10}  {g_s:>10}{lit}")

    # ── Safety bias check ──────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  SAFETY BIAS CHECK — do MaxiMin methods cover high-SAR designs?")
    pool_high = np.sum(y[pool_idx] >= 2.0)
    pool_frac = pool_high / N_POOL
    print(f"  High-SAR (≥2 W/m²) in pool: {pool_high}/{N_POOL} = {pool_frac*100:.0f}%")
    print(f"  {'n':>5}  {'B raw MM%':>12}  {'E geom MM%':>13}  "
          f"{'G 7D MM%':>11}  pool%")
    for n in valid_n[:6]:
        b_fracs = [np.sum(y[pool_idx][maximin_select(D_raw,  n, seed=s)] >= 2.0)/n
                   for s in range(N_SEEDS)]
        e_fracs = [np.sum(y[pool_idx][maximin_select(D_geom, n, seed=s)] >= 2.0)/n
                   for s in range(N_SEEDS)]
        g_fracs = [np.sum(y[pool_idx][maximin_select(D_7d,   n, seed=s)] >= 2.0)/n
                   for s in range(N_SEEDS)]
        bf = np.mean(b_fracs); ef = np.mean(e_fracs); gf = np.mean(g_fracs)
        b_bias = "✓" if abs(bf - pool_frac) < 0.12 else "⚠"
        e_bias = "✓" if abs(ef - pool_frac) < 0.12 else "⚠"
        g_bias = "✓" if abs(gf - pool_frac) < 0.12 else "⚠"
        print(f"  {n:>5}  {bf*100:>7.0f}% ({b_bias})  "
              f"{ef*100:>8.0f}% ({e_bias})  "
              f"{gf*100:>7.0f}% ({g_bias})  {pool_frac*100:.0f}%")

    print(f"\n[TIMING] Total: {time.time()-t_total:.1f}s")


if __name__ == "__main__":
    main()

"""
python em_doe_retrospective.py --dataset last_run_designs.csv
"""
