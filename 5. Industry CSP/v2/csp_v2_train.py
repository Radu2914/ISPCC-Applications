"""
csp_v2_train.py  —  CSP Surrogate Training  (offline, runs once)
══════════════════════════════════════════════════════════════════════════════
Role in architecture:
  OFFLINE — runs once on historical bearing data.
  Produces csp_model.pkl which the deployment model loads.
  Never runs again unless new bearing data is added to the fleet.

What this does that v1 (pronostia_3simm.py) does not:
  1. Trains on ALL 11 bearings (v1 used LOBO for validation — 10 train, 1 test)
  2. Derives cascade detection thresholds from RMS bifurcation analysis
  3. Saves the complete model to disk as a single .pkl artifact
  4. Produces a threshold report so the operator knows what the thresholds mean

Cascade threshold derivation (data-driven, not hardcoded):
  For each bearing: find the first sustained crossing of RMS_BIFURCATION
  (8 of 10 consecutive snapshots above 0.5g).
  The rul_norm at that crossing is the cascade onset for that bearing.
  CASCADE_THRESHOLD = median onset across all bearings.
  EARLY_THRESHOLD   = 2× CASCADE_THRESHOLD (early warning zone).
  CRITICAL_THRESHOLD = 0.35× CASCADE_THRESHOLD (last window before failure).

Output:
  csp_model.pkl — contains ridge, rf, thresholds, feature config, training summary

Usage:
  python csp_v2_train.py --data "C:/path/to/Full_Test_Set"
  python csp_v2_train.py --data "C:/path/to/Full_Test_Set" --n_intentional 2400
  python csp_v2_train.py --data "C:/path/to/Full_Test_Set" --output my_model.pkl
"""

import os
import sys
import glob
import pickle
import argparse
import time

import numpy as np
import pandas as pd

# ── Import all pipeline functions from v1 ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pronostia_3simm import (
    # Constants
    FAILURE_G, TEMP_AMBIENT, TEMP_SCALE, TEMP_BIFURCATION,
    KURT_GAUSSIAN, KURT_SCALE, RMS_BIFURCATION, SNAPSHOT_DT,
    SAMPLES_PER, OP_COND, PI, E, EPS,
    # Data
    load_all_bearings,
    # Features
    build_encoded_features, build_stage1_features, build_intentional_space,
    # Selection
    stratified_maximin_select, maximin_select,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


# ══════════════════════════════════════════════════════════════════════════════
# DEPLOYMENT FIT — tighter parameters than v1 LOBO validation
#
# v1 used alpha=1.0 and min_samples_leaf=2 for cross-bearing stability.
# Deployment trains on ONE operating condition at ONE facility.
# The next bearing will fail through the same mechanism.
# Tight fitting of the local degradation texture is CORRECT, not overfitting.
#
#   Ridge alpha       : 1.0  → 0.01  (fit local power law precisely)
#   RF min_samples_leaf: 2   → 1     (fit factory residual texture exactly)
#   RF n_estimators   : 500  → 1000  (tighter ensemble, cost paid once offline)
# ══════════════════════════════════════════════════════════════════════════════

def three_stage_fit_deployment(X_s1_tr, X_enc_tr, y_log_tr):
    """
    Deployment-tuned fit.  Alpha and leaf size tightened vs v1 LOBO version.
    Same architecture (Ridge grammar + RF dialect), different regularisation.
    """
    X_s1_tr  = np.nan_to_num(X_s1_tr,  nan=0.0, posinf=0.0, neginf=0.0)
    X_enc_tr = np.nan_to_num(X_enc_tr, nan=0.0, posinf=0.0, neginf=0.0)
    y_log_tr = np.nan_to_num(y_log_tr, nan=0.0, posinf=0.0, neginf=0.0)

    ridge = Ridge(alpha=0.01)           # was 1.0 — fit local power law tightly
    ridge.fit(X_s1_tr, y_log_tr)
    resid = y_log_tr - ridge.predict(X_s1_tr)

    rf = RandomForestRegressor(
        n_estimators=1000,              # was 500 — tighter ensemble
        max_features='sqrt',
        min_samples_leaf=1,             # was 2 — fit factory residual texture
        random_state=42, n_jobs=-1)
    rf.fit(X_enc_tr, resid)
    return ridge, rf


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD DERIVATION
# ══════════════════════════════════════════════════════════════════════════════

def filter_by_condition(bearings, condition=None):
    """
    Filter bearing pool to a single operating condition.

    condition: '1' (1800RPM/4000N), '2' (1650RPM/4200N), '3' (1500RPM/5000N)
               None = use all conditions (cross-condition, less precise)

    Rationale: a factory runs one bearing type under one load/speed condition.
    Cross-condition data adds noise to a precision fit. Filter to the condition
    matching the deployment target so Ridge fits the local degradation exponents
    exactly rather than averaging across all operating regimes.
    """
    if condition is None:
        return bearings

    filtered = {
        name: df for name, df in bearings.items()
        if name.startswith(f'Bearing{condition}_')
    }

    if not filtered:
        print(f'[WARN] No bearings found for condition {condition}. '
              f'Using all bearings.')
        return bearings

    excluded = [n for n in bearings if n not in filtered]
    print(f'  Condition filter: keeping {list(filtered.keys())}')
    print(f'  Excluded        : {excluded}')
    return filtered


def derive_thresholds(bearings=None, ridge=None, rf=None,
                      early=0.50, cascade=0.25, critical=0.08):
    """
    Operational thresholds in rul_norm space.

    These are OPERATIONAL DECISIONS, not statistical derivations.
    Previous attempts failed because:
      - RMS bifurcation crossing: PRONOSTIA healthy RMS already > 0.5g
        under extreme accelerated test load → fires at 0-3% of life.
      - Gradient acceleration: model R²=0.9976 → predicted rul_norm is
        nearly linear from 1.0 to 0.0. No 3× acceleration point exists
        in a straight line. Algorithm returns spurious warmup artifacts.

    Correct framing: the client decides how much warning they need.
      - CRITICAL < critical: last 8%  of life → act immediately
      - CASCADE  < cascade : last 25% of life → schedule maintenance now
      - EARLY    < early   : last 50% of life → monitor closely
      - HEALTHY  ≥ early   : >50% life remaining → normal operation

    All three values are configurable via --early, --cascade, --critical args.
    Defaults give: 50% / 25% / 8% of life as the three alert boundaries.

    On a 400-minute bearing (typical Condition 1):
      EARLY    fires at ~200 min remaining  →  3.3 hours warning
      CASCADE  fires at ~100 min remaining  →  1.7 hours warning
      CRITICAL fires at  ~32 min remaining  →  30 minutes to act
    """
    # Guard: strict ordering required
    critical = float(np.clip(critical, 0.01, 0.15))
    cascade  = float(np.clip(cascade,  critical + 0.05, 0.50))
    early    = float(np.clip(early,    cascade  + 0.10, 0.90))

    thresholds = {
        'early':    round(early,    3),
        'cascade':  round(cascade,  3),
        'critical': round(critical, 3),
    }

    print(f'\n[INFO] Operational thresholds (rul_norm space):')
    print(f'  HEALTHY  : rul_norm ≥ {thresholds["early"]:.2f}  ' +
          f'(≥{thresholds["early"]*100:.0f}% life remaining — normal operation)')
    print(f'  EARLY    : {thresholds["cascade"]:.2f} ≤ rul_norm ' +
          f'< {thresholds["early"]:.2f}  — monitor closely')
    print(f'  CASCADE  : {thresholds["critical"]:.2f} ≤ rul_norm ' +
          f'< {thresholds["cascade"]:.2f}  — schedule maintenance')
    print(f'  CRITICAL : rul_norm < {thresholds["critical"]:.2f}  ' +
          f'(<{thresholds["critical"]*100:.0f}% life remaining — act immediately)')
    return thresholds
def train_full(bearings, n_intentional=2400, min_per=6, mm_seed=0):
    """
    Train on ALL bearings simultaneously.
    This is what gets deployed — the model has seen every available bearing.

    Returns (ridge, rf, training_meta).
    """
    print(f'\n[INFO] Training on all {len(bearings)} bearings '
          f'(N_intentional={n_intentional}, min_per={min_per})...')

    all_df = pd.concat(list(bearings.values()), ignore_index=True)

    # ── Stratified MaxiMin across all bearings ────────────────────────────────
    print('\n[INFO] Stratified MaxiMin selection:')
    sel = stratified_maximin_select(
        bearings, n_total=n_intentional, min_per=min_per, seed=mm_seed)
    print(f'  Total selected: {len(sel)} snapshots from '
          f'{all_df.shape[0]:,} pool')

    # ── Feature matrices ───────────────────────────────────────────────────────
    enc_all, pi_cols, e_cols, cross_cols = build_encoded_features(all_df)
    X_s1_all   = build_stage1_features(all_df)
    X_enc_all  = enc_all.values

    # Normalised target
    y_log_norm = np.log(all_df['rul_norm'].values + EPS)

    X_s1_tr  = X_s1_all[sel]
    X_enc_tr = X_enc_all[sel]
    y_tr     = y_log_norm[sel]

    # ── Fit ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    ridge, rf = three_stage_fit_deployment(X_s1_tr, X_enc_tr, y_tr)
    print(f'  Fit complete ({time.time()-t0:.1f}s)  '
          f'[alpha=0.01, n_est=1000, leaf=1]')

    # ── In-sample sanity check ────────────────────────────────────────────────
    from sklearn.metrics import r2_score
    log_pred  = ridge.predict(X_s1_tr) + rf.predict(X_enc_tr)
    norm_pred = np.clip(np.exp(log_pred), 0, 1)
    y_true_n  = all_df['rul_norm'].values[sel]
    r2_in     = r2_score(y_true_n, norm_pred)
    print(f'  In-sample R² (rul_norm): {r2_in:.4f}  '
          f'[sanity check — not generalisation]')

    meta = {
        'bearing_names':   sorted(bearings.keys()),
        'n_bearings':      len(bearings),
        'n_pool':          all_df.shape[0],
        'n_intentional':   n_intentional,
        'n_selected':      len(sel),
        'pi_cols':         pi_cols,
        'e_cols':          e_cols,
        'cross_cols':      cross_cols,
        'r2_insample':     r2_in,
    }
    return ridge, rf, meta


# ══════════════════════════════════════════════════════════════════════════════
# SAVE MODEL ARTIFACT
# ══════════════════════════════════════════════════════════════════════════════

def save_model(ridge, rf, thresholds, meta, output_path='csp_model.pkl'):
    """
    Save complete model artifact.

    csp_model.pkl contains everything the deployment script needs:
      ridge       — Stage 1 Ridge (structural grammar)
      rf          — Stage 2 RF (dialect residuals)
      thresholds  — {early, cascade, critical} in rul_norm space
      constants   — physical constants used in encoding
      meta        — training summary
    """
    artifact = {
        'ridge':      ridge,
        'rf':         rf,
        'thresholds': thresholds,
        'constants': {
            'FAILURE_G':        FAILURE_G,
            'TEMP_AMBIENT':     TEMP_AMBIENT,
            'TEMP_SCALE':       TEMP_SCALE,
            'TEMP_BIFURCATION': TEMP_BIFURCATION,
            'KURT_GAUSSIAN':    KURT_GAUSSIAN,
            'KURT_SCALE':       KURT_SCALE,
            'RMS_BIFURCATION':  RMS_BIFURCATION,
            'SNAPSHOT_DT':      SNAPSHOT_DT,
            'EPS':              EPS,
        },
        'meta': meta,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(artifact, f)
    size_kb = os.path.getsize(output_path) / 1024
    print(f'\n[INFO] Model saved → {output_path}  ({size_kb:.0f} KB)')
    print(f'  Bearings trained on : {meta["bearing_names"]}')
    print(f'  Snapshots used      : {meta["n_selected"]} / {meta["n_pool"]:,}')
    print(f'  In-sample R²        : {meta["r2_insample"]:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='CSP v2 — offline surrogate training')
    ap.add_argument('--data',          required=True,
                    help='Path to Full_Test_Set folder')
    ap.add_argument('--n_intentional', type=int,   default=2400,
                    help='MaxiMin budget (default 2400)')
    ap.add_argument('--min_per',       type=int,   default=6,
                    help='Minimum snapshots per bearing (default 6)')
    ap.add_argument('--output',        default='csp_model.pkl',
                    help='Output model file (default csp_model.pkl)')
    ap.add_argument('--condition',     default=None,
                    choices=['1','2','3'],
                    help='Filter to one operating condition: '
                         '1=1800RPM/4000N  2=1650RPM/4200N  3=1500RPM/5000N '
                         '(default: use all conditions)')
    ap.add_argument('--early',    type=float, default=0.50,
                    help='EARLY threshold in rul_norm (default 0.50 = 50%% life)')
    ap.add_argument('--cascade',  type=float, default=0.25,
                    help='CASCADE threshold in rul_norm (default 0.25 = 25%% life)')
    ap.add_argument('--critical', type=float, default=0.08,
                    help='CRITICAL threshold in rul_norm (default 0.08 = 8%% life)')
    args = ap.parse_args()

    t_total = time.time()

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f'\n[CSP v2 TRAINING]')
    print(f'  Data : {args.data}')
    print(f'  IMM  : {args.n_intentional}  (min {args.min_per}/bearing)')
    bearings = load_all_bearings(args.data)
    print(f'[INFO] {len(bearings)} bearings loaded')

    # ── Condition filter ──────────────────────────────────────────────────────
    if args.condition:
        print(f'\n[INFO] Filtering to operating condition {args.condition}...')
        bearings = filter_by_condition(bearings, args.condition)
        print(f'[INFO] {len(bearings)} bearings after filter')
    else:
        print('[INFO] No condition filter — using all bearings')

    # ── Train first (thresholds derived from model predictions) ──────────────
    ridge, rf, meta = train_full(
        bearings,
        n_intentional=args.n_intentional,
        min_per=args.min_per)

    # ── Thresholds — operational decision, configurable via CLI ─────────────
    thresholds = derive_thresholds(
        early=args.early,
        cascade=args.cascade,
        critical=args.critical)

    # ── Save ──────────────────────────────────────────────────────────────────
    meta['condition'] = args.condition
    save_model(ridge, rf, thresholds, meta, output_path=args.output)

    print(f'\n[TIMING] Total: {time.time()-t_total:.1f}s')
    print(f'\n  Next step:')
    print(f'  python csp_v2_deploy.py --model {args.output} '
          f'--bearing "path/to/BearingX_Y"')


if __name__ == '__main__':
    main()


# ══════════════════════════════════════════════════════════════════════════════
# USAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# Standard run (2400 IMM budget):
#   python csp_v2_train.py \
#     --data "C:/Users/Radu/Desktop/ISPCC/_files/phm-ieee-2012-data-challenge-dataset-master/Full_Test_Set"
#
# Custom budget and output:
#   python csp_v2_train.py \
#     --data "C:/path/to/Full_Test_Set" \
#     --n_intentional 2400 \
#     --output csp_model_v2.pkl
#
# Dependencies: pronostia_3simm.py must be in the same folder.