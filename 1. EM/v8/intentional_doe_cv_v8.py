#!/usr/bin/env python3
"""
intentional_doe_cv_v8.py — 4SIMM comparison harness
=====================================================
Models compared at N = [10, 20, 30, 40, 50, 60]:

  RAW     — random selection, RF on 4 raw inputs, no encoding
             (10-seed average for stable estimate of random-selection noise)

  v7      — 7D MaxiMin, Ridge([raw-4 + Pi28/31/32]) → RF(127 encoded)
             (3SIMM, unchanged from published result)

  4SIMM-A — hard gate, E-narrow(6 features), Stage2b fitted on ALL train pts
  4SIMM-B — hard gate, E-wide(15 features),  Stage2b fitted on ALL train pts
  4SIMM-C — hard gate, E-narrow(6 features), Stage2b fitted on DP-ONLY train pts
  4SIMM-D — hard gate, E-wide(15 features),  Stage2b fitted on DP-ONLY train pts

Hard gate: Π31 > 1.0 → Ε-regime (waveguide); Π31 ≤ 1.0 → Π-regime (direct)

E-narrow (6):  eenc_upper_et (3) + eenc_lower_et (3)
E-wide   (15): eenc_upper_et + eenc_lower_et + eenc_total_et
               + eenc_solid_ang + eenc_path_phase

Usage:
    python intentional_doe_cv_v8.py --dataset last_run_designs.csv
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, pairwise_distances
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PHYSICAL CONSTANTS (Stage 0 — no data required)
# ══════════════════════════════════════════════════════════════════════════════

FREQ_GHZ         = 28.0
LAMBDA_FREE      = 300.0 / FREQ_GHZ          # 10.714 mm
ER_RUBBER        = 4.5
LAMBDA_RUBBER    = LAMBDA_FREE / np.sqrt(ER_RUBBER)   # 5.051 mm
NF_BOUND         = LAMBDA_FREE / (2 * np.pi) # 1.705 mm
MODULE_Y         = 46.53
PHANTOM_RADIUS   = 12.0
APERTURE_HALF_Z  = 10.5
GAP_SPREAD       = APERTURE_HALF_Z**2 / (2 * PHANTOM_RADIUS)  # 4.594 mm
DIEL_BIFURCATION = 0.107   # Β-anchor for tan_δ; ⊘Ε threshold constant
D_MODULE_NOSE    = 58.280  # d(A,B)
D_MODULE_BROW    = 32.130  # d(A,C) = 3.000λ_free exactly
COS_BROW         = 0.06732
PATH_RATIO       = D_MODULE_BROW / D_MODULE_NOSE   # 0.5513

# ── Experiment parameters ─────────────────────────────────────────────────────
N_VALUES  = [60, 100, 140, 180, 200]
MM_SEED   = 0          # MaxiMin starting point — fixed, deterministic
CV_SEED   = 42         # KFold — fixed
RAW_SEEDS = list(range(10))   # 10 random seeds for raw baseline
PREV_BEST_V7_60 = 0.6603      # v7 N=60, published result

EPS = 1e-9
PI  = np.pi
E   = np.e

RAW_COLS = ["gap", "upper_protective_layer",
            "lower_protective_layer", "protective_layer_dielectric"]


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING — identical to v7
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, weights=(5, 1, 1, 3, 1)):
    x  = np.clip(x, 0, 10)
    xn = x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / np.sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2*PI * xn),
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }


def encode_e_func(x, prefix, weights=(2, 2, 1)):
    x  = np.clip(x, 0, 10)
    xn = x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / np.sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-E * xn),
        f"{prefix}_pow_e":   w[1] * xn ** E,
        f"{prefix}_gauss":   w[2] * np.exp(-E * (xn - 0.5)**2),
    }


def pi_groups(df):
    g   = df["gap"].values
    upl = df["upper_protective_layer"].values
    low = df["lower_protective_layer"].values
    td  = df["protective_layer_dielectric"].values
    tot = upl + low
    d   = {}
    d["Pi1_gap_wl_free"]        = g / LAMBDA_FREE
    d["Pi2_gap_nf_ratio"]       = g / NF_BOUND
    d["Pi3_gap_module_ratio"]   = g / MODULE_Y
    d["Pi4_upper_elec_thick"]   = upl / LAMBDA_RUBBER
    d["Pi5_lower_elec_thick"]   = low / LAMBDA_RUBBER
    d["Pi6_total_elec_thick"]   = tot / LAMBDA_RUBBER
    d["Pi7_upper_module"]       = upl / MODULE_Y
    d["Pi8_lower_module"]       = low / MODULE_Y
    d["Pi9_tan_delta"]          = td
    d["Pi10_tan_norm"]          = td / DIEL_BIFURCATION
    d["Pi10b_tan_signed"]       = td - DIEL_BIFURCATION
    d["Pi11_curvature_ratio"]   = GAP_SPREAD / (g + EPS)
    d["Pi12_path_ratio"]        = np.sqrt(1 + (MODULE_Y / (g + EPS))**2)
    d["Pi13_solid_angle"]       = g**2 / (g**2 + MODULE_Y**2 + EPS)
    d["Pi14_shield_upper"]      = upl / (g + EPS)
    d["Pi15_shield_lower"]      = low / (g + EPS)
    d["Pi16_shield_total"]      = tot / (g + EPS)
    d["Pi17_pyramid_ctrl"]      = d["Pi3_gap_module_ratio"] / (td + EPS)
    d["Pi18_source_coupling"]   = td / (d["Pi3_gap_module_ratio"] + EPS)
    d["Pi19_regime_ratio"]      = d["Pi18_source_coupling"] / (d["Pi17_pyramid_ctrl"] + EPS)
    d["Pi20_upper_sw_phase"]    = np.cos(2*PI * d["Pi4_upper_elec_thick"])
    d["Pi21_lower_sw_phase"]    = np.cos(2*PI * d["Pi5_lower_elec_thick"])
    d["Pi22_total_sw_phase"]    = np.cos(2*PI * d["Pi6_total_elec_thick"])
    d["Pi23_gap_phase_free"]    = np.cos(2*PI * d["Pi1_gap_wl_free"])
    d["Pi24_loss_x_thick"]      = td * d["Pi6_total_elec_thick"]
    d["Pi25_loss_x_gap_wl"]     = td * d["Pi1_gap_wl_free"]
    d["Pi26_shield_x_loss"]     = d["Pi16_shield_total"] * td
    d["Pi27_curvature_nf"]      = d["Pi11_curvature_ratio"] / (d["Pi2_gap_nf_ratio"] + EPS)
    d["Pi28_path_competition"]  = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    d["Pi29_direct_path_wl"]    = (D_MODULE_NOSE + g) / LAMBDA_FREE
    d["Pi30_wg_path_wl"]        = np.full(len(df), D_MODULE_BROW / LAMBDA_FREE)
    d["Pi31_regime_switch"]     = PATH_RATIO / (td / DIEL_BIFURCATION + EPS)
    d["Pi32_path_diff_wl"]      = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE
    d["Pi33_deflect_effect"]    = (COS_BROW * PATH_RATIO) / \
                                   (td / DIEL_BIFURCATION + EPS) / \
                                   (1.0 + g / (D_MODULE_NOSE + EPS))
    d["Pi34_path_interf_phase"] = np.cos(2*PI * d["Pi32_path_diff_wl"])
    return pd.DataFrame(d)


def build_encoded_features(df):
    """Full 127-feature encoded set (Stage 3 input)."""
    Pi = pi_groups(df)
    pi_enc = {}
    for key, col in [
        ("gap_wl",      Pi["Pi1_gap_wl_free"].values),
        ("gap_nf",      Pi["Pi2_gap_nf_ratio"].values),
        ("gap_mod",     Pi["Pi3_gap_module_ratio"].values),
        ("tan_d",       Pi["Pi9_tan_delta"].values),
        ("regime",      Pi["Pi19_regime_ratio"].values),
        ("curv",        Pi["Pi11_curvature_ratio"].values),
        ("sw_gap",      Pi["Pi23_gap_phase_free"].values),
        ("path_comp",   Pi["Pi28_path_competition"].values),
        ("direct_wl",   Pi["Pi29_direct_path_wl"].values),
        ("regime_sw",   Pi["Pi31_regime_switch"].values),
        ("path_diff",   Pi["Pi32_path_diff_wl"].values),
        ("deflect_eff", Pi["Pi33_deflect_effect"].values),
    ]:
        pi_enc.update(encode_pi_func(col, f"pienc_{key}"))

    e_enc = {}
    for key, col in [
        ("upper_et",   Pi["Pi4_upper_elec_thick"].values),
        ("lower_et",   Pi["Pi5_lower_elec_thick"].values),
        ("total_et",   Pi["Pi6_total_elec_thick"].values),
        ("shield",     Pi["Pi16_shield_total"].values),
        ("solid_ang",  Pi["Pi13_solid_angle"].values),
        ("path_phase", Pi["Pi34_path_interf_phase"].values),
    ]:
        e_enc.update(encode_e_func(col, f"eenc_{key}"))

    cross = {
        "cross_gapwl_x_upper"    : (np.sin(PI*np.clip(Pi["Pi1_gap_wl_free"]/5, 0, 1)) *
                                     np.exp(-E*np.clip(Pi["Pi4_upper_elec_thick"]/2, 0, 1))),
        "cross_gapwl_x_lower"    : (np.sin(PI*np.clip(Pi["Pi1_gap_wl_free"]/5, 0, 1)) *
                                     np.exp(-E*np.clip(Pi["Pi5_lower_elec_thick"]/2, 0, 1))),
        "cross_tand_x_upper"     : (np.sin(PI*Pi["Pi10_tan_norm"].values/4) *
                                     np.exp(-E*np.clip(Pi["Pi4_upper_elec_thick"]/2, 0, 1))),
        "cross_tand_x_lower"     : (np.sin(PI*Pi["Pi10_tan_norm"].values/4) *
                                     np.exp(-E*np.clip(Pi["Pi5_lower_elec_thick"]/2, 0, 1))),
        "cross_regime_x_shield"  : (np.sin(PI*np.clip(Pi["Pi19_regime_ratio"]/10, 0, 1)) *
                                     np.exp(-E*np.clip(Pi["Pi16_shield_total"]/10, 0, 1))),
        "cross_curv_x_nf"        : (Pi["Pi11_curvature_ratio"].values *
                                     Pi["Pi2_gap_nf_ratio"].values),
        "cross_sw_upper_x_loss"  : (Pi["Pi20_upper_sw_phase"].values * Pi["Pi9_tan_delta"].values),
        "cross_sw_lower_x_loss"  : (Pi["Pi21_lower_sw_phase"].values * Pi["Pi9_tan_delta"].values),
        "cross_sw_total_x_loss"  : (Pi["Pi22_total_sw_phase"].values * Pi["Pi9_tan_delta"].values),
        "cross_path_x_tand"      : (Pi["Pi12_path_ratio"].values * Pi["Pi9_tan_delta"].values),
        "cross_solidang_x_loss"  : (Pi["Pi13_solid_angle"].values *
                                     Pi["Pi24_loss_x_thick"].values),
        "cross_wg_activation"    : (np.sin(PI*np.clip(
                                        Pi["Pi28_path_competition"].values /
                                        (Pi["Pi28_path_competition"].values.max()+EPS), 0, 1)) *
                                     np.exp(-E*np.clip(
                                        Pi["Pi9_tan_delta"].values /
                                        (Pi["Pi9_tan_delta"].values.max()+EPS), 0, 1))),
        "cross_direct_dominance" : (np.sin(PI*np.clip(
                                        Pi["Pi29_direct_path_wl"].values /
                                        (Pi["Pi29_direct_path_wl"].values.max()+EPS), 0, 1)) *
                                     (Pi["Pi10_tan_norm"].values /
                                      (Pi["Pi10_tan_norm"].values.max()+EPS))),
        "cross_path_interference": (Pi["Pi34_path_interf_phase"].values *
                                     np.clip(Pi["Pi31_regime_switch"].values /
                                             (Pi["Pi31_regime_switch"].values.max()+EPS), 0, 1)),
    }
    return pd.concat([Pi, pd.DataFrame(pi_enc),
                      pd.DataFrame(e_enc), pd.DataFrame(cross)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 FEATURES — unencoded-4D + geometric Pi groups (v7 Ridge input)
# ══════════════════════════════════════════════════════════════════════════════

def stage1_features(X_unenc):
    """7 features for Ridge: [gap, upper, lower, tan_δ, Pi28, Pi31, Pi32]"""
    g    = X_unenc[:, 0]
    td   = X_unenc[:, 3]
    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO    / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE
    return np.column_stack([X_unenc, pi28, pi31, pi32])


# ══════════════════════════════════════════════════════════════════════════════
# 4SIMM — REGIME GATE AND E-FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_pi31(X_unenc):
    """Compute Π31 regime switch index for array of raw inputs."""
    td = X_unenc[:, 3]
    return PATH_RATIO / (td / DIEL_BIFURCATION + EPS)


def wg_mask(X_unenc):
    """
    Hard gate (TSA ⊘Ε event):
    True  → Ε-regime (waveguide active, Π31 > 1.0)
    False → Π-regime (direct penetration, Π31 ≤ 1.0)
    """
    return compute_pi31(X_unenc) > 1.0


def extract_e_narrow(X_enc_df):
    """
    6 E-features: upper_et (3) + lower_et (3).
    Standing-wave resonance structure — pure Ε-type, bounded by λ_rubber.
    """
    cols = [c for c in X_enc_df.columns
            if c.startswith("eenc_upper_et") or c.startswith("eenc_lower_et")]
    return X_enc_df[cols].values  # shape (N, 6)


def extract_e_wide(X_enc_df):
    """
    15 E-features: upper_et + lower_et + total_et + solid_ang + path_phase.
    Adds total resonance condition, geometric coupling, interference phase.
    """
    cols = [c for c in X_enc_df.columns
            if any(c.startswith(f"eenc_{k}")
                   for k in ["upper_et", "lower_et", "total_et",
                              "solid_ang", "path_phase"])]
    return X_enc_df[cols].values  # shape (N, 15)


# ══════════════════════════════════════════════════════════════════════════════
# 7D INTENTIONAL MAXIMIN
# ══════════════════════════════════════════════════════════════════════════════

def build_7d_space(X_unenc):
    """[gap, upper, lower, tan_δ, Pi28, Pi31, Pi32] all normalised to [0,1]."""
    g    = X_unenc[:, 0]
    td   = X_unenc[:, 3]
    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO    / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + EPS)

    return np.column_stack([
        norm(X_unenc[:, 0]), norm(X_unenc[:, 1]),
        norm(X_unenc[:, 2]), norm(X_unenc[:, 3]),
        norm(pi28), norm(pi31), norm(pi32),
    ])


def maximin_select(D, n_select, seed=0):
    """Greedy MaxiMin — deterministic given seed (controls starting point only)."""
    rng   = np.random.default_rng(seed)
    n_pool = D.shape[0]
    first  = int(rng.integers(0, n_pool))
    sel    = [first]
    mask   = np.zeros(n_pool, dtype=bool); mask[first] = True
    dmin   = D[first].copy().astype(float); dmin[first] = -np.inf
    for _ in range(n_select - 1):
        nxt = int(np.argmax(np.where(~mask, dmin, -np.inf)))
        sel.append(nxt); mask[nxt] = True
        np.minimum(dmin, D[nxt], out=dmin); dmin[nxt] = -np.inf
    return np.array(sel)


def make_rf():
    return RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                 min_samples_leaf=2, random_state=42, n_jobs=-1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_raw_baseline(X_unenc, y, y_log, n_select, seeds=RAW_SEEDS):
    """
    RAW: random selection, RF on 4 unencoded inputs, 5-fold CV.
    Averaged over `seeds` random selections for noise-stable estimate.
    Returns (mean_r2, std_r2) across all folds × seeds.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    all_r2 = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), n_select, replace=False)
        Xs, ys, yls = X_unenc[idx], y[idx], y_log[idx]
        for tr, te in kf.split(Xs):
            rf = make_rf()
            rf.fit(Xs[tr], yls[tr])
            y_pred = np.exp(rf.predict(Xs[te]))
            all_r2.append(r2_score(ys[te], y_pred))
    return float(np.mean(all_r2)), float(np.std(all_r2))


def run_v7_3simm(X_unenc, X_enc_arr, y, y_log, n_select):
    """
    v7 3SIMM: 7D MaxiMin selection,
    Ridge([raw-4 + Pi28 + Pi31 + Pi32]) → RF(127 encoded, residuals).
    """
    X_7d = build_7d_space(X_unenc)
    D_7d = pairwise_distances(X_7d)
    sel  = maximin_select(D_7d, n_select, seed=MM_SEED)

    Xu = X_unenc[sel]; Xe = X_enc_arr[sel]
    ys = y[sel];       yls = y_log[sel]

    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    r2s = []
    for tr, te in kf.split(Xu):
        ridge = Ridge(alpha=1.0)
        ridge.fit(stage1_features(Xu[tr]), yls[tr])
        resid = yls[tr] - ridge.predict(stage1_features(Xu[tr]))
        rf = make_rf(); rf.fit(Xe[tr], resid)
        yp = np.exp(ridge.predict(stage1_features(Xu[te])) + rf.predict(Xe[te]))
        r2s.append(r2_score(ys[te], yp))
    return float(np.mean(r2s)), float(np.std(r2s))


def run_4simm(X_unenc, X_enc_arr, X_enc_df, y, y_log,
              n_select, e_width="narrow", stage2b_scope="all"):
    """
    4SIMM: hard gate (Π31 threshold = 1.0),
    Stage 2a: E-only Ridge on wg training points,
    Stage 2b: Ridge([raw-4 + Pi28/31/32]) on all OR dp-only training points,
    Stage 3:  RF on 127 encoded features → blended Stage-2 residuals.

    e_width:      "narrow" (6 features) | "wide" (15 features)
    stage2b_scope: "all"  | "dp_only"
    """
    X_7d = build_7d_space(X_unenc)
    D_7d = pairwise_distances(X_7d)
    sel  = maximin_select(D_7d, n_select, seed=MM_SEED)

    Xu  = X_unenc[sel]
    Xe  = X_enc_arr[sel]
    Xed = X_enc_df.iloc[sel].reset_index(drop=True)
    ys  = y[sel]
    yls = y_log[sel]

    # E-regime feature matrix (narrow or wide)
    X_e = extract_e_narrow(Xed) if e_width == "narrow" else extract_e_wide(Xed)

    # Regime mask for the selected N points (hard gate)
    is_wg = wg_mask(Xu)   # True = Ε-regime (waveguide)

    kf  = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)
    r2s = []

    for tr, te in kf.split(Xu):
        wg_tr = is_wg[tr]
        dp_tr = ~is_wg[tr]

        # ── Stage 2a: E-only Ridge on wg training points ──────────────────
        ridge_wg = Ridge(alpha=1.0)
        n_wg = int(wg_tr.sum())
        n_e_feats = X_e.shape[1]
        if n_wg >= 2:
            ridge_wg.fit(X_e[tr][wg_tr], yls[tr][wg_tr])
        else:
            # Fallback: train on all points if too few wg examples
            ridge_wg.fit(X_e[tr], yls[tr])

        # ── Stage 2b: Ridge on all or dp-only training points ─────────────
        ridge_dp = Ridge(alpha=1.0)
        X_s1_tr  = stage1_features(Xu[tr])
        if stage2b_scope == "dp_only":
            n_dp = int(dp_tr.sum())
            if n_dp >= 2:
                ridge_dp.fit(X_s1_tr[dp_tr], yls[tr][dp_tr])
            else:
                ridge_dp.fit(X_s1_tr, yls[tr])   # fallback
        else:  # "all"
            ridge_dp.fit(X_s1_tr, yls[tr])

        # ── Stage 2 predictions on training set (for residuals) ───────────
        log_s2_tr = np.where(
            wg_tr,
            ridge_wg.predict(X_e[tr]),
            ridge_dp.predict(X_s1_tr),
        )
        resid_tr = yls[tr] - log_s2_tr

        # ── Stage 3: RF on full encoded features → residuals ──────────────
        rf = make_rf()
        rf.fit(Xe[tr], resid_tr)

        # ── Test-set prediction ───────────────────────────────────────────
        wg_te    = is_wg[te]
        X_s1_te  = stage1_features(Xu[te])
        log_s2_te = np.where(
            wg_te,
            ridge_wg.predict(X_e[te]),
            ridge_dp.predict(X_s1_te),
        )
        yp = np.exp(log_s2_te + rf.predict(Xe[te]))
        r2s.append(r2_score(ys[te], yp))

    return float(np.mean(r2s)), float(np.std(r2s))


# ══════════════════════════════════════════════════════════════════════════════
# REGIME DIAGNOSTICS — per-N breakdown of wg / dp coverage
# ══════════════════════════════════════════════════════════════════════════════

def regime_summary(X_unenc, y, sel_idx):
    is_wg   = wg_mask(X_unenc[sel_idx])
    n_wg    = int(is_wg.sum())
    n_dp    = int((~is_wg).sum())
    n_hi    = int((y[sel_idx] >= 2.0).sum())
    pi31v   = compute_pi31(X_unenc[sel_idx])
    return n_wg, n_dp, n_hi, float(pi31v.min()), float(pi31v.max())


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    args = vars(ap.parse_args())
    t0   = time.time()

    # ── Load ─────────────────────────────────────────────────────────────────
    df = pd.read_csv(
        args["dataset"], sep=r"\s+", comment="#", header=None,
        names=["index", "gap", "upper_protective_layer", "lower_protective_layer",
               "protective_layer_dielectric", "variable_E", "variable_H",
               "variable_Power", "constr_variable_E", "constr_variable_H",
               "constr_variable_Power", "obj_variable_Power"])
    y      = df["obj_variable_Power"].values
    y_log  = np.log(y + EPS)
    N_ALL  = len(df)

    print(f"[INFO] {N_ALL} HFSS points  y=[{y.min():.3f}, {y.max():.3f}] W/m²")
    print(f"[INFO] Pool: {int((y >= 2.0).sum())} high-SAR (≥2 W/m²) = "
          f"{100*int((y>=2.0).sum())/N_ALL:.0f}%")
    print(f"[INFO] Pool wg (Π31>1): {int(wg_mask(df[RAW_COLS].values).sum())} / {N_ALL}")
    print(f"[INFO] Published v7 N=60 best: R²={PREV_BEST_V7_60}")

    # ── Build full feature matrices once ────────────────────────────────────
    X_unenc   = df[RAW_COLS].values
    X_enc_df  = build_encoded_features(df)
    X_enc_arr = X_enc_df.values
    print(f"[INFO] Feature sets: raw=4  ridge-s1=7  encoded=127")

    # ── Precompute 7D distance matrix (used by all MaxiMin models) ──────────
    print("[INFO] Precomputing 7D MaxiMin distance matrix...")
    X_7d = build_7d_space(X_unenc)
    D_7d = pairwise_distances(X_7d)

    # ── Model labels ─────────────────────────────────────────────────────────
    # Each entry: (label, e_width, stage2b_scope)
    # v7 and RAW handled separately
    simm_variants = [
        ("4SIMM-A", "narrow", "all"),
        ("4SIMM-B", "wide",   "all"),
        ("4SIMM-C", "narrow", "dp_only"),
        ("4SIMM-D", "wide",   "dp_only"),
    ]

    # ── Results store ────────────────────────────────────────────────────────
    # results[N][model_label] = (mean_r2, std_r2)
    results = {n: {} for n in N_VALUES}

    print("\n" + "="*72)
    print("  RUNNING ALL MODELS × ALL N VALUES")
    print("="*72)

    for n in N_VALUES:
        print(f"\n── N={n} {'─'*60}")

        # Regime coverage for this N (MaxiMin selection is same across SIMM variants)
        sel = maximin_select(D_7d, n, seed=MM_SEED)
        n_wg, n_dp, n_hi, pi31_min, pi31_max = regime_summary(X_unenc, y, sel)
        print(f"   MaxiMin selection: wg={n_wg}  dp={n_dp}  "
              f"high-SAR≥2={n_hi}  Π31=[{pi31_min:.3f},{pi31_max:.3f}]")

        # RAW baseline
        m, s = run_raw_baseline(X_unenc, y, y_log, n)
        results[n]["RAW"] = (m, s)
        print(f"   RAW   (avg {len(RAW_SEEDS)} seeds): R²={m:+.4f} ± {s:.4f}")

        # v7 3SIMM
        m, s = run_v7_3simm(X_unenc, X_enc_arr, y, y_log, n)
        results[n]["v7"]  = (m, s)
        print(f"   v7 3SIMM          : R²={m:+.4f} ± {s:.4f}")

        # 4SIMM variants
        for label, ew, s2b in simm_variants:
            m, s = run_4simm(X_unenc, X_enc_arr, X_enc_df, y, y_log,
                             n, e_width=ew, stage2b_scope=s2b)
            results[n][label] = (m, s)
            e_tag  = f"E-{ew[:3]}({6 if ew=='narrow' else 15})"
            s2_tag = f"S2b-{'all' if s2b=='all' else 'dp '}"
            print(f"   {label} [{e_tag} {s2_tag}]: R²={m:+.4f} ± {s:.4f}")

    # ── Summary table ────────────────────────────────────────────────────────
    models  = ["RAW", "v7", "4SIMM-A", "4SIMM-B", "4SIMM-C", "4SIMM-D"]
    hdesc   = {
        "RAW":     "raw-4, rand sel, RF only",
        "v7":      "7D MaxiMin, Ridge(7)+RF(127)",
        "4SIMM-A": "hard gate, E-narrow(6), S2b-all",
        "4SIMM-B": "hard gate, E-wide(15),  S2b-all",
        "4SIMM-C": "hard gate, E-narrow(6), S2b-dp",
        "4SIMM-D": "hard gate, E-wide(15),  S2b-dp",
    }

    print("\n\n" + "="*90)
    print("  COMPARISON TABLE  —  Mean R² (± std across folds)")
    print("  4SIMM gate: hard (Π31 = 1.0)  |  Δ columns: vs v7 at same N")
    print("="*90)
    header = f"  {'Model':<12}  {'Description':<38}"
    for n in N_VALUES:
        header += f"  N={n:02d}"
    print(header)
    print("  " + "─"*88)

    for mdl in models:
        row = f"  {mdl:<12}  {hdesc[mdl]:<38}"
        for n in N_VALUES:
            m, _ = results[n][mdl]
            row += f"  {m:+.3f}"
        print(row)

    print("\n  Std deviations (σ):")
    print("  " + "─"*88)
    for mdl in models:
        row = f"  {mdl:<12}  {'':<38}"
        for n in N_VALUES:
            _, s = results[n][mdl]
            row += f"   {s:.3f}"
        print(row)

    print("\n  Δ R² vs v7 at each N (positive = better than v7):")
    print("  " + "─"*88)
    for mdl in ["RAW", "4SIMM-A", "4SIMM-B", "4SIMM-C", "4SIMM-D"]:
        row = f"  {mdl:<12}  {'':<38}"
        for n in N_VALUES:
            delta = results[n][mdl][0] - results[n]["v7"][0]
            row += f"  {delta:+.3f}"
        print(row)

    print("\n  Best model per N:")
    print("  " + "─"*50)
    for n in N_VALUES:
        best_mdl = max(models, key=lambda m: results[n][m][0])
        best_r2  = results[n][best_mdl][0]
        best_std = results[n][best_mdl][1]
        print(f"  N={n:2d}: {best_mdl:<12} R²={best_r2:+.4f} ± {best_std:.4f}")

    # ── v7 N=60 reference line ────────────────────────────────────────────────
    print(f"\n  Reference: published v7 N=60 → R²={PREV_BEST_V7_60:.4f} ± 0.0988")
    print(f"  [v7 N=60 this run: R²={results[60]['v7'][0]:.4f} ± {results[60]['v7'][1]:.4f}]")

    print(f"\n[TIMING] {time.time()-t0:.1f}s")
    print("="*90)


if __name__ == "__main__":
    main()

"""
python intentional_doe_cv_v8.py --dataset last_run_designs.csv
"""
