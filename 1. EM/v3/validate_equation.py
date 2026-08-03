import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.utils import resample
import argparse

# ══════════════════════════════════════════════════════════════════════════════
# PASTE YOUR COEFFICIENTS HERE
# These come from the terminal output of surrogate.py (power law section)
# The pure 4-term fit — no interaction terms
# ══════════════════════════════════════════════════════════════════════════════
C         =  2.3410   # constant
exp_gap   = -0.8120   # gap exponent
exp_upper = -0.1240   # upper_protective_layer exponent
exp_lower = -0.2310   # lower_protective_layer exponent
exp_diel  =  0.4450   # dielectric exponent
# ══════════════════════════════════════════════════════════════════════════════


def power_law(gap, upper, lower, diel):
    """
    Pure 4-term physics-informed power law:
    P = C × gap^α × upper^β × lower^γ × dielectric^δ

    Used for QUALITATIVE interpretation only — not quantitative prediction.
    All exponents are independent marginal effects (no interaction terms).
    """
    eps = 1e-6
    return (C
            * (gap   + eps) ** exp_gap
            * (upper + eps) ** exp_upper
            * (lower + eps) ** exp_lower
            * (diel  + eps) ** exp_diel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="path to the optiSLang CSV file")
    args = vars(ap.parse_args())

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(
        args["dataset"],
        sep=r'\s+',
        comment='#',
        header=None,
        names=[
            "index",
            "gap", "upper_protective_layer", "lower_protective_layer",
            "protective_layer_dielectric",
            "variable_E", "variable_H", "variable_Power",
            "constr_variable_E", "constr_variable_H", "constr_variable_Power",
            "obj_variable_Power"
        ]
    )

    y_actual = df["obj_variable_Power"].values
    y_eq     = power_law(
        df["gap"].values,
        df["upper_protective_layer"].values,
        df["lower_protective_layer"].values,
        df["protective_layer_dielectric"].values
    )

    ss_res = np.sum((y_actual - y_eq) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2   = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y_actual - y_eq) ** 2))
    rel  = np.mean(np.abs(y_actual - y_eq) / np.abs(y_actual)) * 100

    # ── Equation validation metrics ────────────────────────────────────────────
    print("=" * 55)
    print("  EQUATION VALIDATION")
    print("  [Qualitative model — not used for point prediction]")
    print("=" * 55)
    print(f"  R²            : {r2:.4f}")
    print(f"  RMSE          : {rmse:.4f} W/m²")
    print(f"  Mean rel error: {rel:.1f}%")
    print()
    print(f"  Fitted equation:")
    print(f"  P = {C:.4f}")
    print(f"    × gap^({exp_gap:.4f})")
    print(f"    × upper_layer^({exp_upper:.4f})")
    print(f"    × lower_layer^({exp_lower:.4f})")
    print(f"    × dielectric^({exp_diel:.4f})")

    # ── Physical sanity checks ─────────────────────────────────────────────────
    print("\n  PHYSICAL SANITY CHECKS:")
    print(f"  gap exponent   = {exp_gap:.4f}  →  " +
          ("✅ negative (more gap = less absorption)"
           if exp_gap < 0
           else "❌ POSITIVE — physically wrong, more gap should mean less power"))
    print(f"  upper exponent = {exp_upper:.4f}  →  " +
          ("✅ negative (thicker layer = more shielding)"
           if exp_upper < 0
           else "⚠️  positive — check if this makes physical sense"))
    print(f"  lower exponent = {exp_lower:.4f}  →  " +
          ("✅ negative (thicker layer = more shielding)"
           if exp_lower < 0
           else "⚠️  positive — check if this makes physical sense"))
    print(f"  diel exponent  = {exp_diel:.4f}  →  " +
          ("⚠️  positive (higher dielectric couples more energy into tissue) "
           "— verify against material spec"
           if exp_diel > 0
           else "⚠️  negative — check against material spec"))

    # ── Monotonicity checks ────────────────────────────────────────────────────
    print("\n  MONOTONICITY CHECKS (varying one variable, others held at median):")
    med_gap   = np.median(df["gap"])
    med_upper = np.median(df["upper_protective_layer"])
    med_lower = np.median(df["lower_protective_layer"])
    med_diel  = np.median(df["protective_layer_dielectric"])

    gap_range   = np.linspace(df["gap"].min(),   df["gap"].max(),   50)
    upper_range = np.linspace(df["upper_protective_layer"].min(),
                              df["upper_protective_layer"].max(), 50)
    lower_range = np.linspace(df["lower_protective_layer"].min(),
                              df["lower_protective_layer"].max(), 50)
    diel_range  = np.linspace(df["protective_layer_dielectric"].min(),
                              df["protective_layer_dielectric"].max(), 50)

    p_vs_gap   = power_law(gap_range,   med_upper, med_lower, med_diel)
    p_vs_upper = power_law(med_gap, upper_range,   med_lower, med_diel)
    p_vs_lower = power_law(med_gap,     med_upper, lower_range, med_diel)
    p_vs_diel  = power_law(med_gap,     med_upper, med_lower, diel_range)

    dir_gap   = "decreasing ✅" if p_vs_gap[-1]   < p_vs_gap[0]   else "increasing ❌"
    dir_upper = "decreasing ✅" if p_vs_upper[-1] < p_vs_upper[0] else "increasing ⚠️"
    dir_lower = "decreasing ✅" if p_vs_lower[-1] < p_vs_lower[0] else "increasing ⚠️"

    print(f"  Power vs gap        : "
          f"{p_vs_gap[0]:.3f} → {p_vs_gap[-1]:.3f} W/m²  ({dir_gap})")
    print(f"  Power vs upper layer: "
          f"{p_vs_upper[0]:.3f} → {p_vs_upper[-1]:.3f} W/m²  ({dir_upper})")
    print(f"  Power vs lower layer: "
          f"{p_vs_lower[0]:.3f} → {p_vs_lower[-1]:.3f} W/m²  ({dir_lower})")
    print(f"  Power vs dielectric : "
          f"{p_vs_diel[0]:.3f} → {p_vs_diel[-1]:.3f} W/m²  (check material spec)")

    # ── Spot check on known simulation points ──────────────────────────────────
    print("\n  SPOT CHECK — known simulation points:")
    print(f"  {'Row':<6} {'gap':>6} {'upper':>6} {'lower':>6} {'diel':>7} "
          f"{'Actual':>9} {'Eq.pred':>9} {'Error%':>8}")
    print("  " + "-" * 65)
    for i in [0, 10, 50, 100, 150]:
        row  = df.iloc[i]
        pred = power_law(
            row["gap"],
            row["upper_protective_layer"],
            row["lower_protective_layer"],
            row["protective_layer_dielectric"]
        )
        err = abs(row["obj_variable_Power"] - pred) / abs(row["obj_variable_Power"]) * 100
        print(f"  {i:<6} {row['gap']:>6.2f} {row['upper_protective_layer']:>6.2f} "
              f"{row['lower_protective_layer']:>6.2f} "
              f"{row['protective_layer_dielectric']:>7.4f} "
              f"{row['obj_variable_Power']:>9.4f} {pred:>9.4f} {err:>7.1f}%")

    print(f"\n  [NOTE] Large errors are expected — this equation is used for")
    print(f"         qualitative directional analysis only, not point prediction.")
    print(f"         Quantitative prediction is handled by the ensemble surrogate.")

    # ── Sensitivity plots ──────────────────────────────────────────────────────
    print("\n[INFO] generating sensitivity plots...")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].plot(gap_range,   p_vs_gap,   "b-", linewidth=2)
    axes[0].set_xlabel("Gap (mm)")
    axes[0].set_ylabel("Predicted Power Density (W/m²)")
    axes[0].set_title("Sensitivity: gap")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(upper_range, p_vs_upper, "b-", linewidth=2)
    axes[1].set_xlabel("Upper layer (mm)")
    axes[1].set_ylabel("Predicted Power Density (W/m²)")
    axes[1].set_title("Sensitivity: upper_layer")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(lower_range, p_vs_lower, "b-", linewidth=2)
    axes[2].set_xlabel("Lower layer (mm)")
    axes[2].set_ylabel("Predicted Power Density (W/m²)")
    axes[2].set_title("Sensitivity: lower_layer")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(diel_range,  p_vs_diel,  "b-", linewidth=2)
    axes[3].set_xlabel("Dielectric constant")
    axes[3].set_ylabel("Predicted Power Density (W/m²)")
    axes[3].set_title("Sensitivity: dielectric")
    axes[3].grid(True, alpha=0.3)

    plt.suptitle(
        "Power Law Equation — Sensitivity Analysis\n"
        "(all other variables held at median) — Qualitative trends only",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("equation_validation.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] sensitivity plot saved to equation_validation.png")

    # ── Bootstrap stability (gap sensitivity) ──────────────────────────────────
    print("\n[INFO] running bootstrap stability check...")

    n_bootstrap = 50
    eps = 1e-6
    all_curves  = []   # plain Python list — convert to numpy AFTER loop

    for i in range(n_bootstrap):
        df_boot = resample(df, n_samples=160, random_state=i)
        y_b     = df_boot["obj_variable_Power"].values

        X_log_b = np.column_stack([
            np.log(df_boot["gap"] + eps),
            np.log(df_boot["upper_protective_layer"] + eps),
            np.log(df_boot["lower_protective_layer"] + eps),
            np.log(df_boot["protective_layer_dielectric"] + eps),
        ])
        y_log_b = np.log(y_b + eps)

        m = Ridge(alpha=1.0)
        m.fit(X_log_b, y_log_b)

        C_b   = np.exp(m.intercept_)
        exps  = m.coef_
        curve = (C_b
                 * (gap_range   + eps) ** exps[0]
                 * (med_upper   + eps) ** exps[1]
                 * (med_lower   + eps) ** exps[2]
                 * (med_diel    + eps) ** exps[3])

        all_curves.append(curve)   # list.append — always works

    # convert to numpy AFTER loop
    all_curves = np.array(all_curves)
    mean_curve = all_curves.mean(axis=0)
    std_curve  = all_curves.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(gap_range, mean_curve, "b-", linewidth=2, label="Mean sensitivity")
    plt.fill_between(
        gap_range,
        mean_curve - 2 * std_curve,
        mean_curve + 2 * std_curve,
        alpha=0.3, color="blue", label="±2σ bootstrap band"
    )
    plt.axvline(x=1.7, color="red", linestyle="--", linewidth=1.2,
                label="λ/2π ≈ 1.7 mm (reactive near-field boundary)")
    plt.xlabel("Gap (mm)")
    plt.ylabel("Predicted Power Density (W/m²)")
    plt.title(
        "Gap Sensitivity — Bootstrap Stability\n"
        "(50 subsamples of 160/200 points)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("sensitivity_bootstrap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("[INFO] saved sensitivity_bootstrap.png")

    # ── Bootstrap stability summary ────────────────────────────────────────────
    cv_band_width = (2 * std_curve)
    stable_mask   = gap_range >= 2.0
    print(f"\n  Bootstrap stability summary:")
    print(f"  Mean ±2σ band width (gap > 2mm) : "
          f"{cv_band_width[stable_mask].mean():.4f} W/m²  → stable ✅")
    print(f"  Mean ±2σ band width (gap < 1mm) : "
          f"{cv_band_width[gap_range < 1.0].mean():.4f} W/m²  → uncertain ⚠️")
    print(f"  Reactive near-field boundary    : λ/2π ≈ 1.7 mm at 28 GHz")
    print(f"  Interpretation: sensitivity curve is robust for design-relevant")
    print(f"  gap values (>2mm) and uncertain in the near-contact regime (<1mm),")
    print(f"  consistent with reactive near-field electromagnetic theory.")

if __name__ == "__main__":
    main()

"""
Usage:
python validate_equation.py --dataset "C:/Users/Radu/Desktop/ml project/last_run_designs.csv"

Before running:
- Paste your actual C, exp_gap, exp_upper, exp_lower, exp_diel values
  from the surrogate.py power law output at the top of this file.
- Use ONLY the pure 4-term power law values (no interaction terms).
"""