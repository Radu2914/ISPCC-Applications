import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import joblib
import argparse
import time

# ══════════════════════════════════════════════════════════════════════════════
# ALL CONSTANTS CONFIRMED FROM HFSS GEOMETRY AND MATERIAL PROPERTIES
# No assumptions — every value derived from measurement or HFSS data
# ══════════════════════════════════════════════════════════════════════════════

# ── EM constants ──────────────────────────────────────────────────────────────
FREQ_GHZ        = 28.0
LAMBDA_FREE     = 300.0 / FREQ_GHZ            # 10.714 mm — free-space wavelength
ER_RUBBER       = 4.5                          # confirmed: rubber_hard εr (HFSS)
                                               # NOTE: measured at 9.4 GHz in HFSS
                                               # εr may be slightly lower at 28 GHz
LAMBDA_RUBBER   = LAMBDA_FREE / np.sqrt(ER_RUBBER)  # 5.051 mm — wave in rubber
NF_BOUND        = LAMBDA_FREE / (2 * np.pi)   # 1.705 mm — reactive near-field boundary

# ── Confirmed geometry (all from HFSS measurements) ──────────────────────────
MODULE_Y        = 46.53    # mm — 5G module centre, global Y (confirmed HFSS)
PHANTOM_RADIUS  = 12.0     # mm — cylinder radius (confirmed HFSS)
APERTURE_HALF_Z = 10.5     # mm — half-height of face toward phantom in Z
                           #      confirmed: Edge_39592 Length=21mm → half=10.5mm
APERTURE_HALF_Y = 62.82   # mm — half-width of upper pyramid exit face in Y
                           #      Y_cav=152.9mm → lower pyramid: -2×8.38 → upper: -2×5.25
                           #      Y_upper = 125.64mm → half = 62.82mm

# ── Curvature correction — phantom curves in X-Z plane ONLY ──────────────────
# Cylinder axis is in Y direction → no Y curvature → gap spread only in Z
# GAP_SPREAD = APERTURE_HALF_Z² / (2 × PHANTOM_RADIUS)
GAP_SPREAD      = APERTURE_HALF_Z**2 / (2 * PHANTOM_RADIUS)  # 4.594 mm
# NOTE: this is 0.43 wavelengths — nearly half a wavelength
# For small gaps: peripheral tissue (edge in Z) sees gap + 4.594mm
# For gap=0.088mm: peripheral gap = 4.68mm (53× larger than centre)

# ── Resonance thresholds in rubber layer ──────────────────────────────────────
HALF_WAVE_RUBBER = LAMBDA_RUBBER / 2.0   # 2.526 mm — half-wave resonance
FULL_WAVE_RUBBER = LAMBDA_RUBBER         # 5.051 mm — full-wave resonance
# Layer range: 0.044–5.956mm → max layer = 1.18λ_rubber
# Layers above 2.526mm enter standing wave territory
# Layers above 5.051mm complete a full cycle

# ── Bifurcation point ─────────────────────────────────────────────────────────
DIEL_BIFURCATION = 0.107   # loss tangent — midpoint of 0.025–0.190 range

EPS = 1e-9
PI  = np.pi
E   = np.e


# ══════════════════════════════════════════════════════════════════════════════
# DIMENSIONLESS GROUPS (Buckingham Pi theorem)
# Reference lengths: λ_free (EM scale), λ_rubber (material scale), MODULE_Y (geometry)
# All features are ratios or angles — no absolute dimensions except MODULE_Y
# ══════════════════════════════════════════════════════════════════════════════

def pi_groups(df):
    """
    27 dimensionless groups from confirmed physical constants.

    EM scale: λ_free = 10.714mm, λ_rubber = 5.051mm, NF_BOUND = 1.705mm
    Geometric scale: MODULE_Y = 46.53mm (only confirmed absolute position)
    Material: εr = 4.5, loss tangent = DoE variable (0.021–0.199)
    Curvature: GAP_SPREAD = 4.594mm (Z only, confirmed phantom r=12mm, aperture Z=21mm)

    Group    Expression                   Physical meaning
    ------   ----------                   ----------------
    Π1       gap / λ_free                 gap in free-space wavelengths
    Π2       gap / NF_BOUND               gap relative to reactive NF boundary
    Π3       gap / MODULE_Y               regime competition indicator
    Π4       upper / λ_rubber             upper layer electrical thickness
    Π5       lower / λ_rubber             lower layer electrical thickness
    Π6       (upper+lower) / λ_rubber     total electrical thickness
    Π7       upper / MODULE_Y             upper layer to geometry ratio
    Π8       lower / MODULE_Y             lower layer to geometry ratio
    Π9       tan_δ                        loss tangent (dimensionless)
    Π10      tan_δ / DIEL_BIFURC          relative to bifurcation point
    Π10b     tan_δ - DIEL_BIFURC          signed distance from bifurcation
    Π11      GAP_SPREAD / gap             curvature correction (pure ratio)
    Π12      sqrt(1+(MODULE_Y/gap)²)      oblique path ratio
    Π13      gap²/(gap²+MODULE_Y²)        direct coupling solid angle
    Π14      upper / gap                  shielding-to-gap ratio
    Π15      lower / gap                  shielding-to-gap ratio
    Π16      total / gap                  total shielding-to-gap ratio
    Π17      (g/M) / tan_δ               pyramid control (dimensionless)
    Π18      tan_δ / (g/M)               source coupling (dimensionless)
    Π19      Π18 / Π17                   regime competition ratio
    Π20      cos(2π·Π4)                  upper layer standing wave phase
    Π21      cos(2π·Π5)                  lower layer standing wave phase
    Π22      cos(2π·Π6)                  total standing wave phase
    Π23      cos(2π·Π1)                  gap standing wave phase (free space)
    Π24      tan_δ × Π6                  loss × electrical thickness
    Π25      tan_δ × Π1                  loss × gap in wavelengths
    Π26      Π16 × tan_δ                 total shielding × loss
    Π27      Π11 / Π2                    curvature/NF interaction
    """
    g   = df["gap"].values
    upl = df["upper_protective_layer"].values
    low = df["lower_protective_layer"].values
    td  = df["protective_layer_dielectric"].values
    tot = upl + low

    d = {}

    # ── Gap dimensionless groups ───────────────────────────────────────────────
    d["Pi1_gap_wl_free"]       = g / LAMBDA_FREE
    d["Pi2_gap_nf_ratio"]      = g / NF_BOUND
    d["Pi3_gap_module_ratio"]  = g / MODULE_Y

    # ── Layer electrical thickness (in rubber wavelengths — confirmed εr=4.5) ──
    d["Pi4_upper_elec_thick"]  = upl / LAMBDA_RUBBER
    d["Pi5_lower_elec_thick"]  = low / LAMBDA_RUBBER
    d["Pi6_total_elec_thick"]  = tot / LAMBDA_RUBBER

    # ── Layer-to-geometry ratios ───────────────────────────────────────────────
    d["Pi7_upper_module"]      = upl / MODULE_Y
    d["Pi8_lower_module"]      = low / MODULE_Y

    # ── Loss tangent (inherently dimensionless) ────────────────────────────────
    d["Pi9_tan_delta"]         = td
    d["Pi10_tan_norm"]         = td / DIEL_BIFURCATION
    d["Pi10b_tan_signed"]      = td - DIEL_BIFURCATION

    # ── Pure geometric ratios ─────────────────────────────────────────────────
    d["Pi11_curvature_ratio"]  = GAP_SPREAD / (g + EPS)
    d["Pi12_path_ratio"]       = np.sqrt(1 + (MODULE_Y / (g + EPS))**2)
    d["Pi13_solid_angle"]      = g**2 / (g**2 + MODULE_Y**2 + EPS)
    d["Pi14_shield_upper"]     = upl / (g + EPS)
    d["Pi15_shield_lower"]     = low / (g + EPS)
    d["Pi16_shield_total"]     = tot / (g + EPS)

    # ── Regime competition (dimensionless by construction) ────────────────────
    d["Pi17_pyramid_ctrl"]     = d["Pi3_gap_module_ratio"] / (td + EPS)
    d["Pi18_source_coupling"]  = td / (d["Pi3_gap_module_ratio"] + EPS)
    d["Pi19_regime_ratio"]     = d["Pi18_source_coupling"] / \
                                  (d["Pi17_pyramid_ctrl"] + EPS)

    # ── Standing wave phase (dimensionless angles in radians) ──────────────────
    # These capture resonance effects at λ_rubber/2 and λ_rubber
    # Half-wave resonance: Π4 = 0.5 → cos = -1 (destructive)
    # Full-wave resonance: Π4 = 1.0 → cos = +1 (constructive)
    d["Pi20_upper_sw_phase"]   = np.cos(2*PI * d["Pi4_upper_elec_thick"])
    d["Pi21_lower_sw_phase"]   = np.cos(2*PI * d["Pi5_lower_elec_thick"])
    d["Pi22_total_sw_phase"]   = np.cos(2*PI * d["Pi6_total_elec_thick"])
    d["Pi23_gap_phase_free"]   = np.cos(2*PI * d["Pi1_gap_wl_free"])

    # ── Combined dimensionless products ───────────────────────────────────────
    d["Pi24_loss_x_thick"]     = td * d["Pi6_total_elec_thick"]
    d["Pi25_loss_x_gap_wl"]    = td * d["Pi1_gap_wl_free"]
    d["Pi26_shield_x_loss"]    = d["Pi16_shield_total"] * td
    d["Pi27_curvature_nf"]     = d["Pi11_curvature_ratio"] / \
                                  (d["Pi2_gap_nf_ratio"] + EPS)

    return pd.DataFrame(d)


# ══════════════════════════════════════════════════════════════════════════════
# PI/E ENCODING on dimensionless groups
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi_func(x, prefix, weights=(5, 1, 1, 3, 1)):
    """
    Fourier + power-pi basis for non-periodic cascading variables.
    Direction column weights (5,1,1,3,1) applied explicitly —
    the cascade structure is told to the model, not discovered from data.
    Weight 5: sin(π·x)   — fundamental cascade mode
    Weight 1: cos(π·x)   — quadrature component
    Weight 1: sin(2π·x)  — second harmonic
    Weight 3: sin(π²·x)  — irrational harmonic, strongest non-repeating character
    Weight 1: cascade     — cross-frequency product
    """
    x  = np.clip(x, 0, 10)
    xn = x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / np.sum(weights)  # normalise to sum=1
    d  = {}
    d[f"{prefix}_sin_pi"]  = w[0] * np.sin(PI * xn)
    d[f"{prefix}_cos_pi"]  = w[1] * np.cos(PI * xn)
    d[f"{prefix}_sin_2pi"] = w[2] * np.sin(2*PI * xn)
    d[f"{prefix}_sin_pi2"] = w[3] * np.sin(PI**2 * xn)
    d[f"{prefix}_cascade"] = w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn)
    return d

def encode_e_func(x, prefix, weights=(2, 2, 1)):
    """
    Exponential basis for self-regulating boundary-condition variables.
    Direction column weights (2,2,1) — nearly uniform, confirming that
    flat weighting is structurally correct, not an approximation.
    Weight 2: exp(-e·x)  — primary natural decay
    Weight 2: x^e        — power-e growth (symmetric partner)
    Weight 1: Gaussian   — symmetric self-regulation at midpoint
    """
    x  = np.clip(x, 0, 10)
    xn = x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / np.sum(weights)
    d  = {}
    d[f"{prefix}_exp_neg"] = w[0] * np.exp(-E * xn)
    d[f"{prefix}_pow_e"]   = w[1] * xn ** E
    d[f"{prefix}_gauss"]   = w[2] * np.exp(-E * (xn - 0.5)**2)
    return d

def build_dimensionless_features(df):
    """
    Full dimensionless feature set:
    1. 27 raw Pi groups (dimensionless by construction)
    2. Pi-encoding of cascading groups (gap, regime, loss tangent)
    3. E-encoding of self-regulating groups (electrical thickness)
    4. Cross-products (resonance × shielding)
    """
    Pi = pi_groups(df)

    # Pi-encode: cascading/non-periodic dimensionless groups
    pi_enc = {}
    for key, col in [
        ("gap_wl",   Pi["Pi1_gap_wl_free"].values),
        ("gap_nf",   Pi["Pi2_gap_nf_ratio"].values),
        ("gap_mod",  Pi["Pi3_gap_module_ratio"].values),
        ("tan_d",    Pi["Pi9_tan_delta"].values),
        ("regime",   Pi["Pi19_regime_ratio"].values),
        ("curv",     Pi["Pi11_curvature_ratio"].values),
        ("sw_gap",   Pi["Pi23_gap_phase_free"].values),
    ]:
        pi_enc.update(encode_pi_func(col, f"pienc_{key}"))

    # E-encode: self-regulating dimensionless groups
    e_enc = {}
    for key, col in [
        ("upper_et",  Pi["Pi4_upper_elec_thick"].values),
        ("lower_et",  Pi["Pi5_lower_elec_thick"].values),
        ("total_et",  Pi["Pi6_total_elec_thick"].values),
        ("shield",    Pi["Pi16_shield_total"].values),
        ("solid_ang", Pi["Pi13_solid_angle"].values),
    ]:
        e_enc.update(encode_e_func(col, f"eenc_{key}"))

    # Cross-products: pi-encoded gap × e-encoded layer thickness
    # All cross-products are dimensionless since both inputs are dimensionless
    cross = {
        "cross_gapwl_x_upper"  : (np.sin(PI * np.clip(Pi["Pi1_gap_wl_free"]/5, 0,1)) *
                                   np.exp(-E * np.clip(Pi["Pi4_upper_elec_thick"]/2,0,1))),
        "cross_gapwl_x_lower"  : (np.sin(PI * np.clip(Pi["Pi1_gap_wl_free"]/5,0,1)) *
                                   np.exp(-E * np.clip(Pi["Pi5_lower_elec_thick"]/2,0,1))),
        "cross_tand_x_upper"   : (np.sin(PI * Pi["Pi10_tan_norm"].values/4) *
                                   np.exp(-E * np.clip(Pi["Pi4_upper_elec_thick"]/2,0,1))),
        "cross_tand_x_lower"   : (np.sin(PI * Pi["Pi10_tan_norm"].values/4) *
                                   np.exp(-E * np.clip(Pi["Pi5_lower_elec_thick"]/2,0,1))),
        "cross_regime_x_shield": (np.sin(PI * np.clip(Pi["Pi19_regime_ratio"]/10,0,1)) *
                                   np.exp(-E * np.clip(Pi["Pi16_shield_total"]/10,0,1))),
        "cross_curv_x_nf"      : (Pi["Pi11_curvature_ratio"].values *
                                   Pi["Pi2_gap_nf_ratio"].values),
        "cross_sw_upper_x_loss": (Pi["Pi20_upper_sw_phase"].values *
                                   Pi["Pi9_tan_delta"].values),
        "cross_sw_lower_x_loss": (Pi["Pi21_lower_sw_phase"].values *
                                   Pi["Pi9_tan_delta"].values),
        "cross_sw_total_x_loss": (Pi["Pi22_total_sw_phase"].values *
                                   Pi["Pi9_tan_delta"].values),
        "cross_path_x_tand"    : (Pi["Pi12_path_ratio"].values *
                                   Pi["Pi9_tan_delta"].values),
        "cross_solidang_x_loss": (Pi["Pi13_solid_angle"].values *
                                   Pi["Pi24_loss_x_thick"].values),
    }

    return pd.concat([
        Pi,
        pd.DataFrame(pi_enc),
        pd.DataFrame(e_enc),
        pd.DataFrame(cross)
    ], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-o", "--output", default="obj_variable_Power")
    args = vars(ap.parse_args())

    t_total = time.time()
    print("[INFO] loading dataset...")
    df = pd.read_csv(
        args["dataset"], sep=r'\s+', comment='#', header=None,
        names=[
            "index", "gap", "upper_protective_layer", "lower_protective_layer",
            "protective_layer_dielectric", "variable_E", "variable_H",
            "variable_Power", "constr_variable_E", "constr_variable_H",
            "constr_variable_Power", "obj_variable_Power"
        ]
    )
    print(f"[INFO] loaded {len(df)} design points")

    print(f"\n[INFO] Confirmed physical constants:")
    print(f"  Frequency         : {FREQ_GHZ} GHz")
    print(f"  λ_free            : {LAMBDA_FREE:.3f} mm")
    print(f"  εr_rubber         : {ER_RUBBER}  (confirmed HFSS, measured at 9.4GHz)")
    print(f"  λ_rubber          : {LAMBDA_RUBBER:.3f} mm  (wave in protective layer)")
    print(f"  Half-wave rubber  : {HALF_WAVE_RUBBER:.3f} mm  (resonance threshold)")
    print(f"  Full-wave rubber  : {FULL_WAVE_RUBBER:.3f} mm  (resonance threshold)")
    print(f"  NF boundary       : {NF_BOUND:.3f} mm  (λ/2π at 28GHz)")
    print(f"  MODULE_Y          : {MODULE_Y} mm  (confirmed HFSS geometry)")
    print(f"  Phantom radius    : {PHANTOM_RADIUS} mm  (confirmed)")
    print(f"  Aperture Z half   : {APERTURE_HALF_Z} mm  (confirmed Edge_39592=21mm)")
    print(f"  Aperture Y half   : {APERTURE_HALF_Y:.2f} mm  (computed from cavity)")
    print(f"  GAP_SPREAD        : {GAP_SPREAD:.3f} mm  "
          f"= {APERTURE_HALF_Z}²/(2×{PHANTOM_RADIUS})  [Z only — cylinder in Y]")
    print(f"  GAP_SPREAD/λ_free : {GAP_SPREAD/LAMBDA_FREE:.3f}  "
          f"(nearly half a wavelength)")
    print(f"  Max layer/λ_rubber: {5.956/LAMBDA_RUBBER:.3f}  "
          f"(layers up to 1.18 wavelengths)")
    print(f"  Curvature in Y    : NONE  (cylinder axis in Y, flat in Y)")

    y     = df[args["output"]].values
    y_log = np.log(y + EPS)

    input_cols = ["gap", "upper_protective_layer",
                  "lower_protective_layer", "protective_layer_dielectric"]
    X_raw = df[input_cols].values

    X_pi_df      = pi_groups(df)
    X_pi         = X_pi_df.values
    X_full_df    = build_dimensionless_features(df)
    X_full       = X_full_df.values
    full_cols    = X_full_df.columns.tolist()

    print(f"\n[INFO] Feature sets:")
    print(f"  Raw inputs        : {X_raw.shape[1]}")
    print(f"  Pi groups only    : {X_pi.shape[1]}")
    print(f"  Full encoded      : {X_full.shape[1]}")

    # ── Cross-validate ─────────────────────────────────────────────────────────
    kf        = KFold(n_splits=5, shuffle=True, random_state=42)
    PREV_BEST = 0.5514

    candidates = {
        "RF raw (baseline)"     : (RandomForestRegressor(n_estimators=500,
                                    max_features="sqrt", min_samples_leaf=2,
                                    random_state=42, n_jobs=-1),   X_raw),
        "RF Pi groups only"     : (RandomForestRegressor(n_estimators=500,
                                    max_features="sqrt", min_samples_leaf=2,
                                    random_state=42, n_jobs=-1),   X_pi),
        "RF full encoded"       : (RandomForestRegressor(n_estimators=500,
                                    max_features="sqrt", min_samples_leaf=2,
                                    random_state=42, n_jobs=-1),   X_full),
        "XGB full encoded"      : (xgb.XGBRegressor(n_estimators=300,
                                    learning_rate=0.03, max_depth=4,
                                    subsample=0.8, colsample_bytree=0.8,
                                    random_state=42, verbosity=0), X_full),
    }

    print(f"\n[INFO] cross-validating (previous best R² = {PREV_BEST})...\n")
    best_r2, best_name = -999, None
    best_actual, best_pred = None, None
    best_model_obj, best_X = None, None

    for name, (model, X_use) in candidates.items():
        t0 = time.time()
        all_actual, all_pred = [], []
        rmse_list, r2_list   = [], []

        for train_idx, test_idx in kf.split(X_use):
            model.fit(X_use[train_idx], y_log[train_idx])
            p = np.exp(model.predict(X_use[test_idx])) - EPS
            a = y[test_idx]
            rmse_list.append(np.sqrt(np.mean((a - p)**2)))
            r2_list.append(r2_score(a, p))
            all_actual.extend(a)
            all_pred.extend(p)

        r2_m   = np.mean(r2_list)
        rmse_m = np.mean(rmse_list)
        rel    = np.mean(np.abs(np.array(all_actual) - np.array(all_pred)) /
                         np.abs(np.array(all_actual))) * 100
        delta  = r2_m - PREV_BEST
        tag    = f"  <<< +{delta:.4f} IMPROVEMENT" if delta > 0.005 \
                 else f"  (Δ={delta:+.4f})"

        print(f"  [{name}]")
        print(f"    RMSE={rmse_m:.4f} W/m²  R²={r2_m:.4f}±{np.std(r2_list):.4f}"
              f"  RelErr={rel:.1f}%{tag}")
        print(f"    Time: {time.time()-t0:.1f}s")

        if r2_m > best_r2:
            best_r2        = r2_m
            best_name      = name
            best_actual    = np.array(all_actual)
            best_pred      = np.array(all_pred)
            best_model_obj = model
            best_X         = X_use

    print(f"\n{'='*65}")
    print(f"  BEST MODEL       : {best_name}")
    print(f"  R²               : {best_r2:.4f}")
    print(f"  vs previous best : {best_r2 - PREV_BEST:+.4f}")
    print(f"{'='*65}")

    # ── Feature importance ─────────────────────────────────────────────────────
    print("\n[INFO] feature importance in full encoded set...")
    rf_imp = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf_imp.fit(X_full, y_log)
    print(f"[TIMING] Feature importance RF fit: {time.time()-t0:.1f}s")
    imps  = rf_imp.feature_importances_
    pairs = sorted(zip(full_cols, imps), key=lambda x: x[1], reverse=True)

    print(f"\n  Top 20 features:")
    for col, imp in pairs[:20]:
        bar = "#" * int(imp * 80)
        print(f"  {col:<36}: {imp:.4f}  {bar}")

    def gimp(keywords):
        return sum(imp for col, imp in zip(full_cols, imps)
                   if any(k in col for k in keywords))

    raw_pi  = gimp([f"Pi{i}" for i in range(1, 28)])
    pi_enc  = gimp(["pienc_"])
    e_enc   = gimp(["eenc_"])
    cross   = gimp(["cross_"])

    sw_imp  = gimp(["Pi20","Pi21","Pi22","Pi23","sw_"])
    curv_imp= gimp(["Pi11","Pi27","curv","cross_curv"])
    reg_imp = gimp(["Pi17","Pi18","Pi19","regime","pyramid","source"])

    print(f"\n  Category importances:")
    print(f"  Raw Pi groups               : {raw_pi:.4f}  ({raw_pi*100:.1f}%)")
    print(f"  Pi-encoded (Fourier)        : {pi_enc:.4f}  ({pi_enc*100:.1f}%)")
    print(f"  E-encoded (exponential)     : {e_enc:.4f}  ({e_enc*100:.1f}%)")
    print(f"  Cross-products              : {cross:.4f}  ({cross*100:.1f}%)")
    print(f"\n  Physics mechanism importances:")
    print(f"  Standing wave (rubber layer): {sw_imp:.4f}  ({sw_imp*100:.1f}%)")
    print(f"  Curvature correction        : {curv_imp:.4f}  ({curv_imp*100:.1f}%)")
    print(f"  Regime competition          : {reg_imp:.4f}  ({reg_imp*100:.1f}%)")

    # ── Safety analysis ────────────────────────────────────────────────────────
    high_mask   = best_actual >= 2.0
    n_high      = high_mask.sum()
    n_under     = np.sum(best_pred[high_mask] < best_actual[high_mask])
    corr_factor = float(np.mean(
        best_actual[high_mask] / (best_pred[high_mask] + EPS)))
    print(f"\n[SAFETY ANALYSIS]")
    print(f"  High-SAR underprediction : {n_under}/{n_high}"
          f"  ({100*n_under/max(n_high,1):.0f}%)")
    print(f"  Correction factor        : {corr_factor:.3f}x")

    # ── Diagnostic plots ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Actual vs Predicted
    axes[0].scatter(best_actual, best_pred, alpha=0.5,
                    edgecolors="k", linewidths=0.3, s=40)
    lims = [min(best_actual.min(), best_pred.min()) - 0.1,
            max(best_actual.max(), best_pred.max()) + 0.1]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Power Density (W/m²)")
    axes[0].set_ylabel("Predicted Power Density (W/m²)")
    axes[0].set_title(f"{best_name}\nR²={best_r2:.4f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Category importances
    cats   = ["Raw Pi\ngroups", "Pi-enc\n(Fourier)", "E-enc\n(exp)", "Cross\nproducts"]
    vals   = [raw_pi, pi_enc, e_enc, cross]
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    axes[1].bar(cats, vals, color=colors, edgecolor="k", linewidth=0.8)
    axes[1].set_ylabel("Feature Importance")
    axes[1].set_title("Encoding Category Importances")
    axes[1].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(vals):
        axes[1].text(i, v+0.003, f"{v*100:.1f}%",
                     ha="center", fontsize=9, fontweight="bold")

    # Plot 3: Physics mechanism importances
    mech_cats   = ["Standing\nwave", "Curvature\ncorrection", "Regime\ncompetition"]
    mech_vals   = [sw_imp, curv_imp, reg_imp]
    mech_colors = ["#ff7f0e", "#17becf", "#bcbd22"]
    axes[2].bar(mech_cats, mech_vals, color=mech_colors, edgecolor="k", linewidth=0.8)
    axes[2].set_ylabel("Feature Importance")
    axes[2].set_title("Physics Mechanism Importances\n(confirmed geometry)")
    axes[2].grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mech_vals):
        axes[2].text(i, v+0.001, f"{v*100:.1f}%",
                     ha="center", fontsize=9, fontweight="bold")

    plt.suptitle(
        f"Dimensionless Surrogate — Confirmed Geometry Encoding\n"
        f"εr=4.5, λ_rubber=5.051mm, GAP_SPREAD=4.594mm, r_phantom=12mm"
        f"  |  Best R²={best_r2:.4f}",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("surrogate_dimensionless_diagnostic.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] plot saved to surrogate_dimensionless_diagnostic.png")

    # ── Save ───────────────────────────────────────────────────────────────────
    best_model_obj.fit(best_X, y_log)
    joblib.dump({
        "model"            : best_model_obj,
        "model_name"       : best_name,
        "log_target"       : True,
        "dimensionless"    : True,
        "MODULE_Y"         : MODULE_Y,
        "LAMBDA_FREE"      : LAMBDA_FREE,
        "LAMBDA_RUBBER"    : LAMBDA_RUBBER,
        "ER_RUBBER"        : ER_RUBBER,
        "GAP_SPREAD"       : GAP_SPREAD,
        "PHANTOM_RADIUS"   : PHANTOM_RADIUS,
        "APERTURE_HALF_Z"  : APERTURE_HALF_Z,
        "DIEL_BIFURCATION" : DIEL_BIFURCATION,
        "correction_factor": corr_factor,
        "safety_threshold" : 1.5,
        "note"             : (
            "All features dimensionless via Buckingham Pi theorem. "
            "Constants confirmed from HFSS: εr=4.5 (rubber_hard), "
            "r_phantom=12mm, aperture_Z=21mm, MODULE_Y=46.53mm. "
            "GAP_SPREAD=4.594mm (Z-only, cylinder axis in Y)."
        )
    }, "headset_em_surrogate_dimensionless.pkl")
    print("[INFO] saved to headset_em_surrogate_dimensionless.pkl")

    # ── Print resonance thresholds for paper ──────────────────────────────────
    print(f"\n[INFO] Resonance thresholds (for paper Methods section):")
    print(f"  Half-wave in rubber : {HALF_WAVE_RUBBER:.3f} mm")
    print(f"  Full-wave in rubber : {FULL_WAVE_RUBBER:.3f} mm")
    print(f"  Layer range in DoE  : 0.044 – 5.956 mm")
    n_above_half = np.sum(
        (df["upper_protective_layer"].values > HALF_WAVE_RUBBER) |
        (df["lower_protective_layer"].values > HALF_WAVE_RUBBER))
    print(f"  Designs with at least one layer > half-wave: "
          f"{n_above_half}/200 ({100*n_above_half/200:.0f}%)")
    n_above_full = np.sum(
        (df["upper_protective_layer"].values > FULL_WAVE_RUBBER) |
        (df["lower_protective_layer"].values > FULL_WAVE_RUBBER))
    print(f"  Designs with at least one layer > full-wave: "
          f"{n_above_full}/200 ({100*n_above_full/200:.0f}%)")
    print(f"\n[TIMING] Total wall time: {time.time()-t_total:.1f}s")

if __name__ == "__main__":
    main()

"""
Usage:
python encode_surrogate.py --dataset "C:/Users/Radu/Desktop/ml project/v3/last_run_designs.csv"

All physical constants confirmed from HFSS:
  εr_rubber = 4.5 (rubber_hard material, measured 9.4 GHz)
  loss tangent = $protective_layer_dielectric (DoE variable, 0.021–0.199)
  phantom radius = 12mm (cylinder, axis in Y)
  aperture Z = 21mm → half = 10.5mm (Edge_39592)
  MODULE_Y = 46.53mm (5G module centre)
  GAP_SPREAD = 4.594mm (Z-direction only, no Y curvature)
"""
