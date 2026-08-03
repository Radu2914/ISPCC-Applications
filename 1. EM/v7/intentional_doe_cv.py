import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, pairwise_distances
import argparse
import time

# ══════════════════════════════════════════════════════════════════════════════
# INTENTIONAL DOE — 40-POINT MAXIMIN + 5-FOLD CV
#
# Takes the best architecture from the retrospective (method G) and shows it
# cleanly on 40 points, no seed comparison, no tuning, no method competition.
#
# Pipeline:
#   Selection : 7D IntentionalMaxiMin — [unencoded-4D + Pi28 + Pi31 + Pi32]
#               selects 40 points covering both input range AND waveguide regime
#   Stage 1   : Ridge on [unencoded-4D + Pi28 + Pi31 + Pi32]
#               recovers power-law grammar + geometric regime correction
#   Stage 2   : RF on full encoded feature set (127 features) → Stage-1 residuals
#               learns resonance / curvature / NF transition dialect
#
# Evaluation : 5-fold CV on the 40 selected points (train 32, test 8 per fold)
#   Answers: "given this intentional budget, how well does the pipeline generalise?"
#   Deterministic — seed fixed at 0 for MaxiMin, 42 for KFold splits.
#
# Note on terminology used throughout this project:
#   "unencoded" = 4 DoE inputs as-is from optiSLang (gap, upper, lower, tan_δ)
#   "encoded"   = full pi/e feature set (127 features from Pi groups + Fourier/exp)
#   "canonical" = probe-reduced minimal feature set (Canon-5 or Canon-6)
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
FREQ_GHZ         = 28.0
LAMBDA_FREE      = 300.0 / FREQ_GHZ
ER_RUBBER        = 4.5
LAMBDA_RUBBER    = LAMBDA_FREE / np.sqrt(ER_RUBBER)
NF_BOUND         = LAMBDA_FREE / (2 * np.pi)
MODULE_Y         = 46.53
PHANTOM_RADIUS   = 12.0
APERTURE_HALF_Z  = 10.5
GAP_SPREAD       = APERTURE_HALF_Z**2 / (2 * PHANTOM_RADIUS)
DIEL_BIFURCATION = 0.107
D_MODULE_NOSE    = 58.280
D_MODULE_BROW    = 32.130
COS_BROW         = 0.06732
PATH_RATIO       = D_MODULE_BROW / D_MODULE_NOSE

N_SELECT = 60          # intentional simulation budget
MM_SEED  = 0           # MaxiMin starting point — fixed, not swept
CV_SEED  = 42          # KFold — fixed, not swept
PREV_BEST = 0.5514     # encode_surrogate: 5-fold CV on full 200 points

EPS = 1e-9
PI  = np.pi
E   = np.e

RAW_COLS = ["gap", "upper_protective_layer",
            "lower_protective_layer", "protective_layer_dielectric"]


# ══════════════════════════════════════════════════════════════════════════════
# ENCODING (from encode_surrogate.py — unchanged)
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


# ══════════════════════════════════════════════════════════════════════════════
# PI GROUPS + FULL ENCODED FEATURES (from encode_surrogate.py — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

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
    """Full 127-feature encoded set (Stage 2 input)."""
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
        "cross_gapwl_x_upper"   : (np.sin(PI*np.clip(Pi["Pi1_gap_wl_free"]/5,0,1)) *
                                    np.exp(-E*np.clip(Pi["Pi4_upper_elec_thick"]/2,0,1))),
        "cross_gapwl_x_lower"   : (np.sin(PI*np.clip(Pi["Pi1_gap_wl_free"]/5,0,1)) *
                                    np.exp(-E*np.clip(Pi["Pi5_lower_elec_thick"]/2,0,1))),
        "cross_tand_x_upper"    : (np.sin(PI*Pi["Pi10_tan_norm"].values/4) *
                                    np.exp(-E*np.clip(Pi["Pi4_upper_elec_thick"]/2,0,1))),
        "cross_tand_x_lower"    : (np.sin(PI*Pi["Pi10_tan_norm"].values/4) *
                                    np.exp(-E*np.clip(Pi["Pi5_lower_elec_thick"]/2,0,1))),
        "cross_regime_x_shield" : (np.sin(PI*np.clip(Pi["Pi19_regime_ratio"]/10,0,1)) *
                                    np.exp(-E*np.clip(Pi["Pi16_shield_total"]/10,0,1))),
        "cross_curv_x_nf"       : (Pi["Pi11_curvature_ratio"].values *
                                    Pi["Pi2_gap_nf_ratio"].values),
        "cross_sw_upper_x_loss" : (Pi["Pi20_upper_sw_phase"].values * Pi["Pi9_tan_delta"].values),
        "cross_sw_lower_x_loss" : (Pi["Pi21_lower_sw_phase"].values * Pi["Pi9_tan_delta"].values),
        "cross_sw_total_x_loss" : (Pi["Pi22_total_sw_phase"].values * Pi["Pi9_tan_delta"].values),
        "cross_path_x_tand"     : (Pi["Pi12_path_ratio"].values * Pi["Pi9_tan_delta"].values),
        "cross_solidang_x_loss" : (Pi["Pi13_solid_angle"].values *
                                    Pi["Pi24_loss_x_thick"].values),
        "cross_wg_activation"   : (np.sin(PI*np.clip(
                                       Pi["Pi28_path_competition"].values /
                                       (Pi["Pi28_path_competition"].values.max()+EPS), 0,1)) *
                                    np.exp(-E*np.clip(
                                       Pi["Pi9_tan_delta"].values /
                                       (Pi["Pi9_tan_delta"].values.max()+EPS), 0,1))),
        "cross_direct_dominance": (np.sin(PI*np.clip(
                                       Pi["Pi29_direct_path_wl"].values /
                                       (Pi["Pi29_direct_path_wl"].values.max()+EPS), 0,1)) *
                                    (Pi["Pi10_tan_norm"].values /
                                     (Pi["Pi10_tan_norm"].values.max()+EPS))),
        "cross_path_interference": (Pi["Pi34_path_interf_phase"].values *
                                     np.clip(Pi["Pi31_regime_switch"].values /
                                             (Pi["Pi31_regime_switch"].values.max()+EPS), 0,1)),
    }
    return pd.concat([Pi, pd.DataFrame(pi_enc),
                      pd.DataFrame(e_enc), pd.DataFrame(cross)], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 FEATURES — unencoded-4D + geometric Pi groups
# ══════════════════════════════════════════════════════════════════════════════

def stage1_features(X_unenc):
    """
    7 features for Ridge:
      [gap, upper, lower, tan_δ, Pi28, Pi31, Pi32]
    Ridge fits the log-linear grammar:
      log(P) ≈ a0 + a1·gap + ... + a5·Pi28 + a6·Pi31 + a7·Pi32
    The three geometric Pi groups make the waveguide regime switch
    explicit at Stage 1, removing it from the Stage 2 residuals.
    """
    g    = X_unenc[:, 0]
    td   = X_unenc[:, 3]
    pi28 = D_MODULE_BROW / (D_MODULE_NOSE + g + EPS)
    pi31 = PATH_RATIO    / (td / DIEL_BIFURCATION + EPS)
    pi32 = (D_MODULE_NOSE - D_MODULE_BROW + g) / LAMBDA_FREE
    return np.column_stack([X_unenc, pi28, pi31, pi32])


# ══════════════════════════════════════════════════════════════════════════════
# 7D INTENTIONAL MAXIMIN
# ══════════════════════════════════════════════════════════════════════════════

def build_7d_space(X_unenc):
    """
    [gap, upper, lower, tan_δ, Pi28, Pi31, Pi32] all normalised to [0,1].
    Normalisation uses the statistics of the array passed in (the full pool).
    """
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
    rng = np.random.default_rng(seed)
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


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    args = vars(ap.parse_args())
    t0 = time.time()

    # ── Load ───────────────────────────────────────────────────────────────────
    df = pd.read_csv(
        args["dataset"], sep=r'\s+', comment='#', header=None,
        names=["index", "gap", "upper_protective_layer", "lower_protective_layer",
               "protective_layer_dielectric", "variable_E", "variable_H",
               "variable_Power", "constr_variable_E", "constr_variable_H",
               "constr_variable_Power", "obj_variable_Power"])
    y     = df["obj_variable_Power"].values
    y_log = np.log(y + EPS)
    N_ALL = len(df)

    print(f"[INFO] {N_ALL} HFSS design points  "
          f"y=[{y.min():.3f}, {y.max():.3f}] W/m²")
    print(f"[INFO] Previous best: encode_surrogate 5-fold CV all {N_ALL} pts "
          f"→ R²={PREV_BEST}")

    # ── Build feature matrices ─────────────────────────────────────────────────
    X_unenc = df[RAW_COLS].values           # 4 features — unencoded DoE inputs
    X_enc   = build_encoded_features(df).values  # 127 features — encoded

    print(f"\n[INFO] Feature sets:")
    print(f"  Unencoded (Stage 1 base) : {X_unenc.shape[1]}")
    print(f"  Stage 1 Ridge input      : 7  [unencoded-4 + Pi28 + Pi31 + Pi32]")
    print(f"  Encoded  (Stage 2 RF)    : {X_enc.shape[1]}")

    # ── 7D IntentionalMaxiMin — select N_SELECT points ─────────────────────────
    print(f"\n[INFO] 7D IntentionalMaxiMin — selecting {N_SELECT} points "
          f"(seed={MM_SEED}, deterministic)...")
    X_7d = build_7d_space(X_unenc)
    D_7d = pairwise_distances(X_7d)
    sel_idx = maximin_select(D_7d, N_SELECT, seed=MM_SEED)

    X_unenc_sel = X_unenc[sel_idx]
    X_enc_sel   = X_enc[sel_idx]
    y_sel       = y[sel_idx]
    y_log_sel   = y_log[sel_idx]

    # ── Selected design summary ────────────────────────────────────────────────
    df_sel = df.iloc[sel_idx][RAW_COLS + ["obj_variable_Power"]].reset_index(drop=True)
    print(f"\n[INFO] Selected {N_SELECT} designs:")
    print(f"  {'idx':>4}  {'gap':>7}  {'upper':>7}  {'lower':>7}  "
          f"{'tan_δ':>7}  {'power W/m²':>11}  Pi31")
    print(f"  {'─'*4}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*11}  {'─'*6}")
    for i, (orig_idx, row) in enumerate(zip(sel_idx, df_sel.itertuples())):
        g  = row.gap; td = row.protective_layer_dielectric
        pi31 = PATH_RATIO / (td / DIEL_BIFURCATION + EPS)
        regime = "wg" if pi31 > 0.8 else ("tr" if pi31 > 0.4 else "dp")
        print(f"  {orig_idx:>4}  {g:>7.3f}  {row.upper_protective_layer:>7.3f}  "
              f"{row.lower_protective_layer:>7.3f}  {td:>7.4f}  "
              f"{row.obj_variable_Power:>11.4f}  {pi31:.3f} ({regime})")

    high_sar = np.sum(y_sel >= 2.0)
    print(f"\n  High-SAR (≥2 W/m²) in selected : {high_sar}/{N_SELECT} "
          f"({100*high_sar/N_SELECT:.0f}%)  "
          f"[pool avg = {100*np.sum(y>=2.0)/N_ALL:.0f}%]")
    print(f"  Pi31 range in selected          : "
          f"[{PATH_RATIO/(X_unenc_sel[:,3].max()/DIEL_BIFURCATION):.3f}, "
          f"{PATH_RATIO/(X_unenc_sel[:,3].min()/DIEL_BIFURCATION):.3f}]  "
          f"(full pool: [0.297, 2.766])")

    # ── 5-fold CV on the 40 selected points ───────────────────────────────────
    print(f"\n[INFO] 5-fold CV on {N_SELECT} selected points (CV seed={CV_SEED})...")
    print(f"  Architecture: Ridge(unencoded-4 + Pi28+Pi31+Pi32)"
          f" → RF(encoded-127, residuals)")
    kf = KFold(n_splits=5, shuffle=True, random_state=CV_SEED)

    fold_r2s = []; fold_rmses = []; fold_rels = []
    all_actual = []; all_pred   = []

    for fold_i, (tr, te) in enumerate(kf.split(X_unenc_sel)):
        # Stage 0 — geometric Pi groups embedded in stage1_features (no fitting)
        X_s1_tr = stage1_features(X_unenc_sel[tr])
        X_s1_te = stage1_features(X_unenc_sel[te])

        # Stage 1 — Ridge: power law grammar + waveguide regime correction
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_s1_tr, y_log_sel[tr])
        resid_tr = y_log_sel[tr] - ridge.predict(X_s1_tr)

        # Stage 2 — RF on full encoded features → residuals
        rf = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf.fit(X_enc_sel[tr], resid_tr)

        # Prediction
        y_log_pred = ridge.predict(X_s1_te) + rf.predict(X_enc_sel[te])
        y_pred     = np.exp(y_log_pred) - EPS
        y_true     = y_sel[te]

        r2   = r2_score(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        rel  = np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100

        fold_r2s.append(r2); fold_rmses.append(rmse); fold_rels.append(rel)
        all_actual.extend(y_true); all_pred.extend(y_pred)

        print(f"  Fold {fold_i+1}  R²={r2:+.4f}  RMSE={rmse:.4f} W/m²  "
              f"RelErr={rel:.1f}%  (n_train={len(tr)}, n_test={len(te)})")

    # ── Summary ────────────────────────────────────────────────────────────────
    mean_r2   = np.mean(fold_r2s)
    std_r2    = np.std(fold_r2s)
    mean_rmse = np.mean(fold_rmses)
    mean_rel  = np.mean(fold_rels)
    delta     = mean_r2 - PREV_BEST

    print(f"\n{'='*65}")
    print(f"  RESULT — 7D MaxiMin, {N_SELECT} points, 5-fold CV")
    print(f"{'='*65}")
    print(f"  R²               : {mean_r2:.4f} ± {std_r2:.4f}")
    print(f"  RMSE             : {mean_rmse:.4f} W/m²")
    print(f"  Mean rel error   : {mean_rel:.1f}%")
    print(f"  vs previous best : {delta:+.4f}  "
          f"({'IMPROVEMENT' if delta > 0.005 else 'within margin' if abs(delta) < 0.005 else 'below'})")
    print(f"\n  Interpretation:")
    print(f"  {N_SELECT} intentionally selected simulations, 5-fold CV within that budget.")
    print(f"  Previous best used all {N_ALL} simulations with 5-fold CV.")
    if mean_r2 > PREV_BEST:
        print(f"  {N_SELECT} intentional points ≥ {N_ALL} random points.")
    print(f"{'='*65}")

    # ── Safety check on full predictions ──────────────────────────────────────
    all_actual = np.array(all_actual); all_pred = np.array(all_pred)
    high_mask  = all_actual >= 2.0; n_high = high_mask.sum()
    if n_high > 0:
        n_under = np.sum(all_pred[high_mask] < all_actual[high_mask])
        corr    = np.mean(all_actual[high_mask] / (all_pred[high_mask] + EPS))
        print(f"\n  Safety (high-SAR ≥2 W/m²):")
        print(f"  Underprediction : {n_under}/{n_high}  ({100*n_under/n_high:.0f}%)")
        print(f"  Correction factor: {corr:.3f}×")

    print(f"\n[TIMING] {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

"""
python intentional_doe_cv.py --dataset last_run_designs.csv
"""
