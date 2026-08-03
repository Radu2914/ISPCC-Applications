import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import argparse
import time

# ── Physical constants (confirmed from HFSS geometry) ─────────────────────────
MODULE_Y              = 46.53   # mm — 5G module centre, global coords (Y=0 = nose)
PYRAMID_Y             =  0.0    # mm — pyramid centre (symmetric, Y=0 by design)
DIEL_BIFURCATION      =  0.107  # midpoint between tan_d=0.025 and tan_d=0.190
SAFETY_FLAG_THRESHOLD =  1.5    # W/m²
HIGH_SAR_THRESHOLD    =  2.0    # W/m²

# ── Feature engineering ────────────────────────────────────────────────────────
def engineer_features(df, module_y=MODULE_Y):
    """
    Physics-informed features encoding the competition between:
    - Pyramid redistribution regime (low dielectric → hotspot near Y=0)
    - Direct source coupling regime (high dielectric → hotspot near module Y)
    Confirmed by HFSS: 34.6mm hotspot migration between tan_d=0.025 and 0.190.
    """
    d   = df.copy()
    eps = 1e-6

    # ── Standard geometric features ───────────────────────────────────────────
    d["inv_gap"]         = 1.0 / (d["gap"] + eps)
    d["log_gap"]         = np.log(d["gap"] + eps)
    d["gap_sq"]          = d["gap"] ** 2
    d["total_layer"]     = d["upper_protective_layer"] + d["lower_protective_layer"]
    d["inv_total_layer"] = 1.0 / (d["total_layer"] + eps)
    d["log_total_layer"] = np.log(d["total_layer"] + eps)
    d["diel_x_gap"]      = d["protective_layer_dielectric"] * d["gap"]
    d["diel_x_layer"]    = d["protective_layer_dielectric"] * d["total_layer"]
    d["shield_total"]    = d["total_layer"] * d["protective_layer_dielectric"]

    # ── Regime competition features (NEW — requires MODULE_Y) ─────────────────
    diel = d["protective_layer_dielectric"]
    gap  = d["gap"]

    # 3D path from module to direct-coupling tissue point
    # When source regime dominates, field travels this oblique path
    d["lateral_path"]    = np.sqrt(gap**2 + module_y**2)
    d["inv_lateral"]     = 1.0 / (d["lateral_path"] + eps)

    # Pyramid control proxy:
    # Strong when dielectric is LOW (transparent layer) and gap is LARGE
    d["pyramid_control"] = gap / (diel + eps)

    # Source coupling proxy:
    # Strong when dielectric is HIGH and gap is SMALL
    d["source_coupling"] = diel / (gap + eps)

    # Regime ratio: >1 means source dominates, <1 means pyramid dominates
    d["regime_ratio"]    = d["source_coupling"] / (d["pyramid_control"] + eps)
    d["log_regime"]      = np.log(d["regime_ratio"] + eps)

    # Signed distance from bifurcation point
    # Negative = pyramid regime, Positive = source regime
    d["diel_from_bifurc"] = diel - DIEL_BIFURCATION

    # Proximity to bifurcation (where prediction uncertainty is highest)
    d["bifurc_proximity"] = np.abs(diel - DIEL_BIFURCATION)

    # Source-regime attenuation: field strength at lateral module position
    # relative to field strength at pyramid centre
    d["source_vs_pyramid"] = (1.0 / (d["lateral_path"] + eps)) / (1.0 / (gap + eps))

    # Combined shielding effectiveness in both regimes
    d["shield_pyramid"]  = d["total_layer"] * diel * (1.0 / (gap + eps))
    d["shield_lateral"]  = d["total_layer"] * diel * d["inv_lateral"]

    return d


def log_features(df):
    """Log-space features for power law base layer of stacked model."""
    eps = 1e-6
    return np.column_stack([
        np.log(df["gap"] + eps),
        np.log(df["upper_protective_layer"] + eps),
        np.log(df["lower_protective_layer"] + eps),
        np.log(df["protective_layer_dielectric"] + eps),
        np.log(df["upper_protective_layer"] + df["lower_protective_layer"] + eps),
        np.log(df["gap"] + eps) * np.log(df["protective_layer_dielectric"] + eps),
        np.log(df["gap"] + eps) * np.log(
            df["upper_protective_layer"] + df["lower_protective_layer"] + eps),
    ])


def plot_diagnostics(actual, predicted, title, filename):
    residuals = actual - predicted
    r2 = r2_score(actual, predicted)
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    ax1 = fig.add_subplot(gs[0])
    ax1.scatter(actual, predicted, alpha=0.5, edgecolors="k", linewidths=0.3, s=40)
    lims = [min(actual.min(), predicted.min()) - 0.1,
            max(actual.max(), predicted.max()) + 0.1]
    ax1.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    ax1.set_xlabel("Actual Power Density (W/m²)")
    ax1.set_ylabel("Predicted Power Density (W/m²)")
    ax1.set_title(f"Actual vs Predicted\nR²={r2:.4f}")
    ax1.legend()

    ax2 = fig.add_subplot(gs[1])
    ax2.scatter(predicted, residuals, alpha=0.5, edgecolors="k", linewidths=0.3, s=40)
    ax2.axhline(0, color="r", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Predicted Power Density (W/m²)")
    ax2.set_ylabel("Residual (actual - predicted) (W/m²)")
    ax2.set_title("Residuals vs Predicted")

    ax3 = fig.add_subplot(gs[2])
    ax3.hist(residuals, bins=20, edgecolor="k", alpha=0.7, color="steelblue")
    ax3.axvline(0, color="r", linestyle="--", linewidth=1.5)
    ax3.set_xlabel("Residual (W/m²)")
    ax3.set_ylabel("Count")
    ax3.set_title("Residual Distribution")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[INFO] plot saved to {filename}")


def cross_validate(model, X, y, log_target=True, n_splits=5):
    kf    = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_fit = np.log(y + 1e-6) if log_target else y
    all_actual, all_pred = [], []
    rmse_list, r2_list   = [], []

    for train_idx, test_idx in kf.split(X):
        model.fit(X[train_idx], y_fit[train_idx])
        p = model.predict(X[test_idx])
        if log_target:
            p = np.exp(p) - 1e-6
        a = y[test_idx]
        rmse_list.append(np.sqrt(np.mean((a - p) ** 2)))
        r2_list.append(r2_score(a, p))
        all_actual.extend(a)
        all_pred.extend(p)

    return (np.array(all_actual), np.array(all_pred),
            np.mean(rmse_list), np.std(rmse_list),
            np.mean(r2_list),   np.std(r2_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True)
    ap.add_argument("-o", "--output", default="obj_variable_Power")
    args = vars(ap.parse_args())
    t_total = time.time()

    # ── Load data ──────────────────────────────────────────────────────────────
    print("[INFO] loading dataset...")
    df = pd.read_csv(
        args["dataset"], sep=r'\s+', comment='#', header=None,
        names=[
            "index",
            "gap", "upper_protective_layer", "lower_protective_layer",
            "protective_layer_dielectric",
            "variable_E", "variable_H", "variable_Power",
            "constr_variable_E", "constr_variable_H", "constr_variable_Power",
            "obj_variable_Power"
        ]
    )
    print(f"[INFO] loaded {len(df)} design points")
    print(f"[INFO] 5G module centre: Y = {MODULE_Y} mm (confirmed from HFSS geometry)")
    print(f"[INFO] bifurcation point: tan_d = {DIEL_BIFURCATION}")

    input_cols = ["gap", "upper_protective_layer", "lower_protective_layer",
                  "protective_layer_dielectric"]
    y     = df[args["output"]].values
    y_log = np.log(y + 1e-6)

    X_raw  = df[input_cols].values
    X_eng  = engineer_features(df[input_cols]).values
    X_logf = log_features(df)
    eng_cols = engineer_features(df[input_cols]).columns.tolist()

    print(f"[INFO] engineered features ({len(eng_cols)}): {eng_cols}")

    # ── Fixed-param models (no tuning) ────────────────────────────────────────
    # Same hyperparameters as encode_surrogate.py — identical compute budget,
    # different feature space. This isolates encoding from tuning.
    best_rf = RandomForestRegressor(
        n_estimators=500, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)

    best_xgb = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0)

    # ── Compare all candidates ─────────────────────────────────────────────────
    print("\n[INFO] cross-validating all models...\n")

    candidates = {
        "RF raw (baseline)"      : (RandomForestRegressor(n_estimators=500,
                                    max_features="sqrt", min_samples_leaf=2,
                                    random_state=42, n_jobs=-1),  X_raw,  True),
        "RF physics-informed"    : (best_rf,                       X_eng,  True),
        "XGB physics-informed"   : (best_xgb,                      X_eng,  True),
        "Ensemble (RF+XGB) phys" : (VotingRegressor([
                                        ("rf",  best_rf),
                                        ("xgb", best_xgb)]),       X_eng,  True),
    }

    best_r2, best_name = -999, None
    best_actual, best_pred = None, None
    best_model_obj, best_X_use = None, None

    t0 = time.time()
    for name, (model, X_use, log_t) in candidates.items():
        actual, pred, rmse_m, rmse_s, r2_m, r2_s = cross_validate(
            model, X_use, y, log_target=log_t)
        rel = np.mean(np.abs(actual - pred) / np.abs(actual)) * 100
        print(f"  [{name}]")
        print(f"    RMSE={rmse_m:.4f}±{rmse_s:.4f} W/m²  "
              f"R²={r2_m:.4f}±{r2_s:.4f}  RelErr={rel:.1f}%")

        if r2_m > best_r2:
            best_r2        = r2_m
            best_name      = name
            best_actual    = actual
            best_pred      = pred
            best_model_obj = model
            best_X_use     = X_use
    print(f"[TIMING] Candidate CV: {time.time()-t0:.1f}s")

    # ── Stacked model ──────────────────────────────────────────────────────────
    print("\n[INFO] building stacked model (power law base → RF on physics features)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stack_actual, stack_pred = [], []

    t0 = time.time()    
    for train_idx, test_idx in kf.split(X_logf):
        pl = Ridge(alpha=1.0)
        pl.fit(X_logf[train_idx], y_log[train_idx])
        pl_train = pl.predict(X_logf[train_idx])
        pl_test  = pl.predict(X_logf[test_idx])

        # Stack: physics-informed features + power law prediction
        X_stack_train = np.hstack([X_eng[train_idx], pl_train.reshape(-1, 1)])
        X_stack_test  = np.hstack([X_eng[test_idx],  pl_test.reshape(-1, 1)])
        rf_s = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                     min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf_s.fit(X_stack_train, y_log[train_idx])
        p = np.exp(rf_s.predict(X_stack_test)) - 1e-6
        stack_actual.extend(y[test_idx])
        stack_pred.extend(p)
    print(f"[TIMING] Stacked model CV: {time.time()-t0:.1f}s")

    stack_actual = np.array(stack_actual)
    stack_pred   = np.array(stack_pred)
    r2_stack     = r2_score(stack_actual, stack_pred)
    rmse_stack   = np.sqrt(np.mean((stack_actual - stack_pred)**2))
    rel_stack    = np.mean(np.abs(stack_actual - stack_pred) /
                           np.abs(stack_actual)) * 100
    print(f"  [Stacked (PowerLaw + physics features -> RF)]")
    print(f"    RMSE={rmse_stack:.4f} W/m²  R²={r2_stack:.4f}  RelErr={rel_stack:.1f}%")

    if r2_stack > best_r2:
        best_r2      = r2_stack
        best_name    = "Stacked (PowerLaw + physics -> RF)"
        best_actual  = stack_actual
        best_pred    = stack_pred

    # ── Quantile model (conservative safety bound) ─────────────────────────────────
    print("\n[INFO] training conservative quantile surrogate (85th percentile)...")

    # Train in log space — raw y is too skewed for quantile regression
    # Exponentiate predictions back to W/m²
    q_model = GradientBoostingRegressor(
        loss="quantile", alpha=0.85,
        n_estimators=500, learning_rate=0.03,
        max_depth=4, subsample=0.8, random_state=42
    )

    kf       = KFold(n_splits=5, shuffle=True, random_state=42)
    q_actual, q_pred = [], []

    t0 = time.time()
    for train_idx, test_idx in kf.split(X_eng):
        q_model.fit(X_eng[train_idx], y_log[train_idx])   # fit on log(y)
        p = np.exp(q_model.predict(X_eng[test_idx]))       # exponentiate back
        q_actual.extend(y[test_idx])
        q_pred.extend(p)
    print(f"[TIMING] Quantile CV: {time.time()-t0:.1f}s")

    q_actual = np.array(q_actual)
    q_pred   = np.array(q_pred)
    q_r2     = r2_score(q_actual, q_pred)

    # High-SAR coverage
    high_mask   = q_actual >= HIGH_SAR_THRESHOLD
    n_high      = high_mask.sum()
    n_bounded   = np.sum(q_pred[high_mask] >= q_actual[high_mask])
    coverage    = 100 * n_bounded / max(n_high, 1)

    # Safe zone penalty — how much does it overpredict in the safe region
    safe_mask   = ~high_mask
    mean_over   = np.mean(q_pred[safe_mask] - q_actual[safe_mask])

    # Mean relative error
    rel_err = np.mean(np.abs(q_actual - q_pred) / np.abs(q_actual)) * 100

    print(f"\n  [Quantile surrogate — 85th percentile, log-space trained]")
    print(f"    R²                              : {q_r2:.4f}")
    print(f"    Mean relative error             : {rel_err:.1f}%")
    print(f"    High-SAR cases bounded (>=2.0)  : {n_bounded}/{n_high} ({coverage:.0f}%)")
    print(f"    Mean overestimate in safe zone  : {mean_over:.4f} W/m²")
    print(f"    Interpretation: conservative upper bound for compliance screening")

    # Retrain on all data and save
    q_model.fit(X_eng, y_log)
    joblib.dump({"model": q_model, "input_cols": input_cols,
                "log_target": True, "model_name": "Quantile GB (85th pct, log-space)",
                "type": "quantile", "alpha": 0.85, "module_Y": MODULE_Y},
                "headset_em_surrogate_conservative.pkl")
    print("[INFO] conservative quantile model saved to headset_em_surrogate_conservative.pkl")

    # Save quantile CV predictions separately
    q_df = pd.DataFrame({
        "actual_Wm2"    : q_actual,
        "predicted_Wm2" : q_pred,
        "abs_error_Wm2" : np.abs(q_actual - q_pred),
        "rel_error_pct" : np.abs(q_actual - q_pred) / np.abs(q_actual) * 100,
        "bounded"       : q_pred >= q_actual,
        "safety_flag"   : ["WARNING" if p >= SAFETY_FLAG_THRESHOLD else "OK"
                        for p in q_pred]
    })
    q_df.to_csv("surrogate_cv_quantile.csv", index=False)
    print("[INFO] quantile CV predictions saved to surrogate_cv_quantile.csv")

    # Plot: quantile vs mean surrogate comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(q_actual, q_pred, alpha=0.5, edgecolors="k",
                    linewidths=0.3, s=40, color="darkorange")
    lims = [min(q_actual.min(), q_pred.min()) - 0.1,
            max(q_actual.max(), q_pred.max()) + 0.1]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].axhline(SAFETY_FLAG_THRESHOLD, color="purple", linestyle=":",
                    linewidth=1.5, label=f"Safety flag ({SAFETY_FLAG_THRESHOLD} W/m²)")
    axes[0].set_xlabel("Actual Power Density (W/m²)")
    axes[0].set_ylabel("Predicted Power Density (W/m²)")
    axes[0].set_title(f"Quantile Surrogate (85th pct)\nR²={q_r2:.4f}  "
                    f"High-SAR coverage={coverage:.0f}%")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(q_actual[high_mask],  q_pred[high_mask],
                    color="red",   alpha=0.7, s=60, label=f"High-SAR (>={HIGH_SAR_THRESHOLD} W/m²)",
                    edgecolors="k", linewidths=0.3)
    axes[1].scatter(q_actual[safe_mask],  q_pred[safe_mask],
                    color="steelblue", alpha=0.4, s=30, label=f"Safe (<{HIGH_SAR_THRESHOLD} W/m²)")
    axes[1].plot(lims, lims, "r--", linewidth=1.5)
    axes[1].axhline(HIGH_SAR_THRESHOLD, color="red",    linestyle=":", linewidth=1.2)
    axes[1].axvline(HIGH_SAR_THRESHOLD, color="red",    linestyle=":", linewidth=1.2)
    axes[1].set_xlabel("Actual Power Density (W/m²)")
    axes[1].set_ylabel("Predicted Power Density (W/m²)")
    axes[1].set_title(f"High-SAR vs Safe Zone\n"
                    f"{n_bounded}/{n_high} high-SAR cases bounded above actual")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Conservative Quantile Surrogate — 85th Percentile\n"
                "5G Medical Headset Peak Spatial Power Density",
                fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("surrogate_quantile_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] quantile diagnostic plot saved to surrogate_quantile_diagnostic.png")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BEST QUANTITATIVE MODEL : {best_name}")
    print(f"  R²                      : {best_r2:.4f}")
    print(f"  RMSE                    : "
          f"{np.sqrt(np.mean((best_actual-best_pred)**2)):.4f} W/m²")
    print(f"  Mean RelErr             : "
          f"{np.mean(np.abs(best_actual-best_pred)/np.abs(best_actual))*100:.1f}%")
    print(f"{'='*60}")

    # ── Feature importances ────────────────────────────────────────────────────
    print("\n[INFO] training final RF on all data for feature importances...")
    rf_imp = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_imp.fit(X_eng, y_log)
    print("\n[INFO] Feature importances (physics-informed set):")
    pairs = sorted(zip(eng_cols, rf_imp.feature_importances_),
                   key=lambda x: x[1], reverse=True)
    for col, imp in pairs:
        bar = "#" * int(imp * 50)
        print(f"  {col:<28}: {imp:.4f}  {bar}")

    # Check where regime features rank
    regime_feats = ["regime_ratio", "log_regime", "source_coupling",
                    "pyramid_control", "diel_from_bifurc",
                    "bifurc_proximity", "lateral_path", "source_vs_pyramid"]
    regime_imp = sum(imp for col, imp in pairs if col in regime_feats)
    print(f"\n  Combined regime feature importance: {regime_imp:.4f} "
          f"({regime_imp*100:.1f}% of total)")

    # ── Safety analysis ────────────────────────────────────────────────────────
    print(f"\n[SAFETY ANALYSIS]")
    high_mask   = best_actual >= HIGH_SAR_THRESHOLD
    high_actual = best_actual[high_mask]
    high_pred   = best_pred[high_mask]
    n_high      = len(high_actual)
    n_under     = np.sum(high_pred < high_actual)
    corr_factor = float(np.mean(high_actual / (high_pred + 1e-9)))
    print(f"  Cases with actual >= {HIGH_SAR_THRESHOLD} W/m²  : {n_high}")
    print(f"  Underpredicted                   : {n_under} ({100*n_under/max(n_high,1):.0f}%)")
    print(f"  Mean underprediction             : {np.mean(high_actual-high_pred):.3f} W/m²")
    print(f"  Correction factor                : {corr_factor:.3f}x")
    print(f"  Safety flag threshold            : {SAFETY_FLAG_THRESHOLD} W/m²")
    print(f"\n  [NOTE] Predictions >= {SAFETY_FLAG_THRESHOLD} W/m² require HFSS verification.")

    # ── Retrain best model on all data ─────────────────────────────────────────
    print(f"\n[INFO] retraining {best_name} on all data...")
    if "Stacked" in best_name:
        pl_f = Ridge(alpha=1.0)
        pl_f.fit(X_logf, y_log)
        pl_all = pl_f.predict(X_logf)
        X_stack_all = np.hstack([X_eng, pl_all.reshape(-1, 1)])
        rf_f = RandomForestRegressor(n_estimators=500, max_features="sqrt",
                                     min_samples_leaf=2, random_state=42, n_jobs=-1)
        rf_f.fit(X_stack_all, y_log)
        model_to_save = {"type": "stacked", "pl": pl_f, "rf": rf_f,
                         "input_cols": input_cols,
                         "model_name": best_name, "log_target": True,
                         "correction_factor": corr_factor,
                         "safety_flag_threshold": SAFETY_FLAG_THRESHOLD,
                         "module_Y": MODULE_Y}
    else:
        best_model_obj.fit(best_X_use, y_log)
        model_to_save = {"type": "single", "model": best_model_obj,
                         "input_cols": input_cols,
                         "model_name": best_name, "log_target": True,
                         "correction_factor": corr_factor,
                         "safety_flag_threshold": SAFETY_FLAG_THRESHOLD,
                         "module_Y": MODULE_Y}

    joblib.dump(model_to_save, "headset_em_surrogate.pkl")
    q_model.fit(X_eng, y)
    joblib.dump({"model": q_model, "input_cols": input_cols,
                 "model_name": "Quantile (85th pct)", "type": "quantile",
                 "module_Y": MODULE_Y},
                "headset_em_surrogate_conservative.pkl")
    print("[INFO] model saved to headset_em_surrogate.pkl")
    print("[INFO] conservative model saved to headset_em_surrogate_conservative.pkl")

    # ── Save CV predictions ────────────────────────────────────────────────────
    cv_df = pd.DataFrame({
        "actual_Wm2"   : best_actual,
        "predicted_Wm2": best_pred,
        "abs_error_Wm2": np.abs(best_actual - best_pred),
        "rel_error_pct": np.abs(best_actual - best_pred) / np.abs(best_actual) * 100,
        "safety_flag"  : ["WARNING" if p >= SAFETY_FLAG_THRESHOLD else "OK"
                          for p in best_pred]
    })
    cv_df.to_csv("surrogate_cv_predictions.csv", index=False)
    print("[INFO] CV predictions saved to surrogate_cv_predictions.csv")

    # ── Diagnostic plot ────────────────────────────────────────────────────────
    plot_diagnostics(
        best_actual, best_pred,
        f"Best Surrogate: {best_name} — R²={best_r2:.4f}\n"
        f"5G Medical Headset — Peak Spatial Power Density",
        "surrogate_diagnostics_final.png"
    )

    # ── Design variable ranges ─────────────────────────────────────────────────
    print("\n[INFO] Design variable ranges (for Methods section):")
    for col in input_cols:
        print(f"  {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")
    print(f"\n[INFO] Output range : [{y.min():.4f}, {y.max():.4f}] W/m²")
    print(f"[INFO] ICNIRP 2020 limit (>6 GHz): 20 W/m²")
    print(f"[INFO] Design target: 10 W/m²  (safety factor 2x)")
    print(f"[INFO] 5G module Y  : {MODULE_Y} mm (confirmed HFSS geometry)")
    
    print(f"\n[TIMING] Total wall time: {time.time()-t_total:.1f}s")

if __name__ == "__main__":
    main()

"""
Usage:
python surrogate_test.py --dataset "C:/Users/Radu/Desktop/ml project/v2t/last_run_designs.csv"
python validate_equation.py --dataset "C:/Users/Radu/Desktop/ml project/last_run_designs.csv"
"""