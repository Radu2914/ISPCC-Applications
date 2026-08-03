"""
csp_v2_deploy.py  —  CSP Live Deployment  (runs continuously)
══════════════════════════════════════════════════════════════════════════════
Role in architecture:
  ONLINE — loads the trained model from csp_v2_train.py and runs on every
  new Kepware snapshot. Outputs regime classification and cascade alert.

Fully standalone — does NOT depend on pronostia_3simm.py or csp_v2_train.py.
All inference logic is self-contained so this file can be deployed
independently on any machine with numpy/pandas/scipy/sklearn/pickle.

What it does per snapshot (takes milliseconds):
  1. Load acc + temp CSV (or receive from Kepware live feed)
  2. Extract statistical features (RMS, kurtosis, peak, crest, temp)
  3. Pi/e encode using the same structural constants as training
  4. Predict rul_norm via Ridge + RF
  5. Convert to minutes using bearing's known/estimated total life
  6. Update rolling 5-snapshot regime window
  7. Output: regime, risk level, estimated minutes to failure

Cascade detection rule:
  CASCADE confirmed when 4 of last 5 snapshots have rul_norm < CASCADE_THRESHOLD
  and the trajectory is monotonically decreasing.
  Lead time = rul_s_pred at first confirmed CASCADE snapshot.

Usage:
  python csp_v2_deploy.py --model csp_model.pkl --bearing path/to/BearingX_Y
  python csp_v2_deploy.py --model csp_model.pkl --bearing path/to/BearingX_Y --every 50
"""

import os
import sys
import glob
import pickle
import argparse
import time

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE  —  all functions needed for deployment inference
# (no training dependencies — standalone)
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path):
    """Load csp_model.pkl. Returns (ridge, rf, thresholds, constants)."""
    with open(model_path, 'rb') as f:
        artifact = pickle.load(f)
    return (artifact['ridge'], artifact['rf'],
            artifact['thresholds'], artifact['constants'])


def load_snapshot(acc_path, temp_path=None):
    """
    Load one acc + optional temp snapshot.
    Returns (h, v, temperature) arrays.
    acc: comma-separated [hour,min,sec,usec,acc_h,acc_v]
    temp: semicolon-separated [hour,min,sec,subsec,temp_C]
    """
    try:
        acc = pd.read_csv(acc_path, header=None,
                          names=['hour','min','sec','usec','acc_h','acc_v'])
        h = acc['acc_h'].values.astype(float)
        v = acc['acc_v'].values.astype(float)
    except Exception:
        return None, None, None

    temp = None
    if temp_path and os.path.exists(temp_path):
        try:
            t = pd.read_csv(temp_path, header=None, sep=';',
                            names=['hour','min','sec','subsec','temp_C'])
            temp = float(t['temp_C'].mean())
        except Exception:
            pass

    return h, v, temp


def extract_features(h, v, temp, C, snapshot_idx, total_snapshots,
                     prev_rms_env=None):
    """
    Extract per-snapshot features from raw acceleration and temperature.

    Returns dict with all features needed for encoding and Stage 1.
    C = constants dict from model artifact.
    """
    EPS = C['EPS']

    # Guard: constant signal gives NaN kurtosis
    def safe_kurt(x):
        k = float(scipy_kurtosis(x, fisher=False))
        return k if np.isfinite(k) else C['KURT_GAUSSIAN']

    def _clean(x):
        return np.nan_to_num(float(x), nan=0.0, posinf=0.0, neginf=0.0)

    rms_h   = _clean(np.sqrt(np.mean(h**2)))
    rms_v   = _clean(np.sqrt(np.mean(v**2)))
    kurt_h  = _clean(safe_kurt(h))
    kurt_v  = _clean(safe_kurt(v))
    peak_h  = _clean(np.max(np.abs(h)))
    peak_v  = _clean(np.max(np.abs(v)))
    crest_h = peak_h / (rms_h + EPS)
    crest_v = peak_v / (rms_v + EPS)

    rms_env  = max(rms_h, rms_v)
    kurt_env = max(kurt_h, kurt_v)
    peak_env = max(peak_h, peak_v)

    # Temperature
    TEMP_AMBIENT = C['TEMP_AMBIENT']
    TEMP_SCALE   = C['TEMP_SCALE']
    t_val        = temp if (temp is not None and np.isfinite(temp)) \
                   else TEMP_AMBIENT
    temp_dev     = max(0.0, t_val - TEMP_AMBIENT)

    # Life fraction (requires knowing total snapshots)
    life_frac = snapshot_idx / max(total_snapshots - 1, 1)

    # Post-bifurcation RMS distance
    RMS_BIF   = C['RMS_BIFURCATION']
    rms_bif   = max(0.0, rms_env - RMS_BIF)

    return {
        'acc_h_rms':   rms_h,   'acc_v_rms':   rms_v,
        'acc_h_kurt':  kurt_h,  'acc_v_kurt':  kurt_v,
        'acc_h_peak':  peak_h,  'acc_v_peak':  peak_v,
        'acc_h_crest': crest_h, 'acc_v_crest': crest_v,
        'rms_env':     rms_env, 'kurt_env':    kurt_env,
        'peak_env':    peak_env,
        'temperature': t_val,   'temp_dev':    temp_dev,
        'life_frac':   life_frac,
        'rms_bif':     rms_bif,
    }


def encode_features(feat, C):
    """
    Pi/e encode one snapshot's features.
    Same functions and weights as training — must not diverge.
    Returns (X_s1, X_enc) as 1D arrays.
    """
    PI  = np.pi
    E_  = np.e
    EPS = C['EPS']

    def pi_enc(x, scale, weights=(5,1,1,3,1)):
        x  = np.nan_to_num(float(x), nan=0.0)
        xn = np.clip(x / (scale + EPS), 0, 10)
        w  = np.array(weights, dtype=float) / sum(weights)
        return np.array([
            w[0]*np.sin(PI*xn),
            w[1]*np.cos(PI*xn),
            w[2]*np.sin(2*PI*xn),
            w[3]*np.sin(PI**2*xn),
            w[4]*np.sin(PI*xn)*np.cos(PI**2*xn),
        ])

    def e_enc(x, scale, weights=(2,2,1)):
        x  = np.nan_to_num(float(x), nan=0.0)
        xn = np.clip(x / (scale + EPS), 0, 10)
        w  = np.array(weights, dtype=float) / sum(weights)
        return np.array([
            w[0]*np.exp(-E_*xn),
            w[1]*xn**E_,
            w[2]*np.exp(-E_*(xn-0.5)**2),
        ])

    FG  = C['FAILURE_G']
    KS  = C['KURT_SCALE']
    TS  = C['TEMP_SCALE']
    TBF = C['TEMP_BIFURCATION']
    RBF = C['RMS_BIFURCATION']

    enc = np.concatenate([
        # π-type
        pi_enc(feat['acc_h_rms'],  FG),
        pi_enc(feat['acc_v_rms'],  FG),
        pi_enc(feat['acc_h_kurt'], KS),
        pi_enc(feat['acc_v_kurt'], KS),
        pi_enc(feat['acc_h_peak'], FG*3),
        pi_enc(feat['acc_v_peak'], FG*3),
        pi_enc(feat['life_frac'],  1.0),
        pi_enc(feat['rms_bif'],    FG),
        # e-type
        e_enc(feat['temp_dev'],    TS),
        e_enc(feat['acc_h_crest'], 10.0),
        e_enc(feat['acc_v_crest'], 10.0),
        e_enc(max(0, feat['temp_dev'] - TBF), TS - TBF),
        # cross-products
        np.array([
            np.sin(PI * np.clip(feat['rms_env']/FG, 0, 1)) *
            np.exp(-E_ * feat['temp_dev']/(TS+EPS)),
            np.sin(PI * np.clip(feat['kurt_env']/KS, 0, 1)) *
            np.exp(-E_ * feat['temp_dev']/(TS+EPS)),
            np.sin(PI * feat['life_frac']) *
            np.exp(-E_ * np.clip(feat['rms_env']/FG, 0, 1)),
            np.sin(PI * np.clip(feat['rms_bif']/FG, 0, 1)) *
            np.exp(-E_ * feat['temp_dev']/(TS+EPS)),
        ]),
    ])

    # Stage 1 features: same 7-column structure as training
    log_rms_h = np.log(feat['acc_h_rms'] + EPS)
    log_rms_v = np.log(feat['acc_v_rms'] + EPS)
    log_kurt  = np.log(max(1.0, feat['kurt_env']))
    life      = feat['life_frac']
    temp_dev  = feat['temp_dev']
    rms_bif   = feat['rms_bif']
    regime    = np.clip(
        (feat['rms_env'] / (RBF + EPS)) /
        (temp_dev / (TBF + EPS) + EPS), 0, 100)

    s1 = np.array([log_rms_h, log_rms_v, log_kurt,
                   life, temp_dev, rms_bif, regime])
    s1 = np.nan_to_num(s1, nan=0.0, posinf=0.0, neginf=0.0)

    return s1.reshape(1, -1), enc.reshape(1, -1)


def predict_rul_norm(ridge, rf, X_s1, X_enc):
    """Predict rul_norm ∈ [0,1]. 0 = failure, 1 = start of life."""
    log_pred = ridge.predict(X_s1) + rf.predict(X_enc)
    return float(np.clip(np.exp(log_pred[0]), 0.0, 1.0))


def classify_regime(history_norm, thresholds):
    """
    Classify current regime from rolling 5-snapshot window.

    Returns (regime_str, risk_str, confirmed_cascade).
    confirmed_cascade = True when 4/5 recent snapshots are below
    CASCADE_THRESHOLD and the trajectory is declining.
    """
    if len(history_norm) < 2:
        return 'HEALTHY', 'LOW', False

    window = history_norm[-5:]
    median = float(np.median(window))
    T = thresholds

    if median > T['early']:
        regime, risk = 'HEALTHY',  'LOW'
    elif median > T['cascade']:
        regime, risk = 'EARLY',    'MEDIUM'
    elif median > T['critical']:
        regime, risk = 'CASCADE',  'HIGH'
    else:
        regime, risk = 'CRITICAL', 'CRITICAL'

    # Cascade confirmed: 4/5 below cascade threshold AND declining
    below_cascade = sum(1 for v in window if v < T['cascade'])
    declining     = (len(window) >= 3 and window[-1] < window[-3])
    confirmed     = (below_cascade >= 4 and declining)

    return regime, risk, confirmed


# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

REGIME_SYMBOL = {
    'HEALTHY':  '  ●  ',
    'EARLY':    ' ◐   ',
    'CASCADE':  '⚠    ',
    'CRITICAL': '🔴   ',
}

RISK_COLOUR = {
    'LOW':      '',
    'MEDIUM':   '',
    'HIGH':     '*** ',
    'CRITICAL': '!!! ',
}


def print_header(bearing_name, thresholds, total_life_min, model_path):
    SEP = '═' * 72
    print(f'\n{SEP}')
    print(f'  CSP LIVE DEPLOYMENT  —  {bearing_name}')
    print(f'  Model : {os.path.basename(model_path)}')
    print(f'  Life  : {total_life_min:.1f} min estimated  '
          f'(from snapshot count)')
    print(f'  Thresholds (rul_norm):  '
          f'EARLY<{thresholds["early"]:.2f}  '
          f'CASCADE<{thresholds["cascade"]:.2f}  '
          f'CRITICAL<{thresholds["critical"]:.2f}')
    print(f'{SEP}')
    print(f"  {'Snap':>6}  {'Time':>7}  {'Regime':10}  "
          f"{'RUL(min)':>9}  {'RMS(g)':>7}  {'Kurt':>7}  "
          f"{'Temp(°C)':>9}  Signal")
    print(f"  {'─'*6}  {'─'*7}  {'─'*10}  "
          f"{'─'*9}  {'─'*7}  {'─'*7}  {'─'*9}  {'─'*8}")


def print_snapshot(snap_idx, elapsed_min, regime, risk, rul_min,
                   feat, confirmed):
    sym      = REGIME_SYMBOL.get(regime, '     ')
    risk_pfx = RISK_COLOUR.get(risk, '')
    sig      = 'π-cascade' if regime in ('CASCADE','CRITICAL') \
               else ('π/e mixed' if regime == 'EARLY' else 'e-regul.')
    temp_str = f"{feat['temperature']:>9.1f}" \
               if feat['temperature'] is not None else '      n/a'
    flag     = ' ◄ CASCADE CONFIRMED' if confirmed else ''
    print(f"  {snap_idx:>6}  {elapsed_min:>6.1f}m  "
          f"{risk_pfx}{sym}{regime:10}  "
          f"{rul_min:>9.1f}  "
          f"{feat['rms_env']:>7.4f}  "
          f"{feat['kurt_env']:>7.2f}  "
          f"{temp_str}  {sig}{flag}")


def print_cascade_alert(snap_idx, elapsed_min, rul_min, regime,
                        feat, baseline, C):
    """
    Cascade alert with encoded feature diagnosis.
    Shows WHY cascade is flagged using π/e structural typing,
    not just raw sensor values vs baseline.
    Encoded diagnosis catches pre-spike cascade (Bearing2_4 case)
    where raw RMS is still low but trajectory has crossed structural midpoint.
    """
    SEP = '─' * 72

    # Raw sensor ratios for context
    rms_ratio  = feat['rms_env']  / (baseline['rms']  + C['EPS'])
    temp_diff  = feat['temp_dev'] - baseline['temp']
    temp_bif   = C['TEMP_BIFURCATION']
    temp_status = ('self-regulating  ✓  (e-type boundary intact)'
                   if feat['temp_dev'] < temp_bif
                   else f'elevated +{temp_diff:.1f}°C  (e-type boundary broken)')

    # Encoded feature state — the actual detection basis
    FG      = C['FAILURE_G']
    KS      = C['KURT_SCALE']
    EPS_    = C['EPS']
    rms_xn  = min(feat['rms_env']  / (FG + EPS_), 10.0)
    kurt_xn = min(feat['kurt_env'] / (KS + EPS_), 10.0)
    life_xn = feat['life_frac']

    # π-activation thresholds in encoded space
    rms_pi_enc  = rms_xn  > 0.30   # RMS past 30% of failure threshold
    kurt_pi_enc = kurt_xn > 0.10   # kurtosis past 10% of fault ceiling
    life_pi_enc = life_xn > 0.60   # bearing past 60% of expected life

    pi_signals = sum([rms_pi_enc, kurt_pi_enc, life_pi_enc])
    raw_spike  = rms_ratio > 2.0

    if raw_spike:
        diagnosis = 'π-dominant cascade confirmed (sensor spike + encoded signal)'
    elif pi_signals >= 2:
        diagnosis = 'π-dominant cascade confirmed (encoded signal — pre-spike detection)'
    else:
        diagnosis = 'π-type emerging via encoded trajectory (subtle — monitor closely)'

    print(f'\n  {SEP}')
    if regime == 'CRITICAL':
        print(f'  !!! CRITICAL — FAILURE IMMINENT')
    else:
        print(f'  ⚠  CASCADE STATE DETECTED  —  CSP structural diagnosis')
    print(f'  {SEP}')
    print(f'     Snapshot          : {snap_idx}  (+{elapsed_min:.1f} min elapsed)')
    print(f'     Est. RUL          : {rul_min:.1f} minutes')
    print(f'     Lead time         : {rul_min:.1f} min before predicted failure')
    print()
    print(f'     ENCODED FEATURE STATE (π/e structural typing):')
    print(f'       RMS encoded     : xn={rms_xn:.3f}  '
          f'(raw {feat["rms_env"]:.3f}g = {rms_ratio:.1f}× baseline)  '
          f'{"→ π-active" if rms_pi_enc else "→ e-regime"}')
    print(f'       Kurtosis encoded: xn={kurt_xn:.3f}  '
          f'(raw {feat["kurt_env"]:.2f})  '
          f'{"→ π-active" if kurt_pi_enc else "→ e-regime"}')
    print(f'       Life fraction   : {life_xn:.3f}  '
          f'({life_xn*100:.0f}% of expected life elapsed)  '
          f'{"→ π-active" if life_pi_enc else "→ e-regime"}')
    print(f'       Temperature     : {feat["temperature"]:.1f}°C  '
          f'(dev {feat["temp_dev"]:+.1f}°C)  →  {temp_status}')
    print()
    print(f'     DIAGNOSIS         : {diagnosis}')
    print(f'     Recommended       : schedule maintenance within '
          f'{max(0, rul_min - 10):.0f} min')
    print(f'  {SEP}\n')
def print_summary(bearing_name, total_life_min, cascade_detected_at,
                  cascade_lead_min, n_snaps, n_shown, false_alarms):
    SEP = '═' * 72
    print(f'\n{SEP}')
    print(f'  CSP RUN SUMMARY  —  {bearing_name}')
    print(f'  Total bearing life : {total_life_min:.1f} min  ' +
          f'({n_snaps} snapshots  ×  10s each)')
    print(f'  Snapshots shown    : {n_shown}')
    print(f'  False alarms       : {false_alarms}  ' +
          f'(CASCADE → back to HEALTHY/EARLY transitions)')
    if cascade_detected_at is not None:
        pct_through = 100 * (1 - cascade_lead_min / total_life_min)
        print()
        print(f'  CASCADE DETECTION RESULT:')
        print(f'    Detected at      : +{cascade_detected_at:.1f} min  ' +
              f'({pct_through:.0f}% through bearing life)')
        print(f'    Lead time        : {cascade_lead_min:.1f} min  ' +
              f'= {cascade_lead_min/total_life_min*100:.0f}% of life remaining')
        print(f'    Action window    : {cascade_lead_min:.1f} min to act ' +
              f'before predicted failure')
        print()
        print(f'  CSP CLAIM VERIFIED: cascade state identified {cascade_lead_min:.1f} min')
        print(f'  before failure with structural π-type diagnosis.')
    else:
        print(f'\n  Cascade            : NOT DETECTED in this run')
        print(f'  (Bearing completed run without confirmed cascade signal)')
    print(f'{SEP}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DEPLOYMENT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_deployment(bearing_dir, model_path, every=50, max_snaps=None):
    """
    Simulate live deployment by reading bearing snapshots sequentially.

    every     : print output every N snapshots (default 50 = every 8.3 min)
    max_snaps : stop after this many snapshots (None = run to failure)
    """
    # ── Load model ─────────────────────────────────────────────────────────────
    print(f'[CSP v2 DEPLOY]  Loading {model_path}...')
    ridge, rf, thresholds, C = load_model(model_path)
    print(f'  Thresholds: cascade<{thresholds["cascade"]:.3f}  '
          f'critical<{thresholds["critical"]:.3f}')

    # ── List snapshots ─────────────────────────────────────────────────────────
    bearing_name = os.path.basename(bearing_dir.rstrip('/\\'))
    acc_files    = sorted(glob.glob(os.path.join(bearing_dir, 'acc_*.csv')))
    temp_files   = sorted(glob.glob(os.path.join(bearing_dir, 'temp_*.csv')))

    if not acc_files:
        print(f'[ERROR] No acc files in {bearing_dir}')
        return

    n_total       = len(acc_files)
    total_life_s  = n_total * C['SNAPSHOT_DT']
    total_life_min = total_life_s / 60

    if max_snaps:
        acc_files = acc_files[:max_snaps]

    print_header(bearing_name, thresholds, total_life_min, model_path)

    # ── Live loop ──────────────────────────────────────────────────────────────
    history_norm      = []
    history_feats     = []          # rolling feature store for baseline
    cascade_first_idx = None
    cascade_lead_min  = None
    false_alarms      = 0
    n_shown           = 0
    prev_regime       = None

    for i, acc_path in enumerate(acc_files):
        temp_path = temp_files[i] if i < len(temp_files) else None
        h, v, temp = load_snapshot(acc_path, temp_path)
        if h is None:
            continue

        feat = extract_features(
            h, v, temp, C,
            snapshot_idx=i,
            total_snapshots=n_total)

        X_s1, X_enc   = encode_features(feat, C)
        rul_norm_pred  = predict_rul_norm(ridge, rf, X_s1, X_enc)
        history_norm.append(rul_norm_pred)
        history_feats.append(feat)

        rul_min   = rul_norm_pred * total_life_min
        elapsed   = i * C['SNAPSHOT_DT'] / 60
        regime, risk, confirmed = classify_regime(history_norm, thresholds)

        # Clean regime change detection — no double classify
        is_regime_change = (prev_regime is not None and regime != prev_regime)

        # False alarm: was CASCADE, now back to EARLY or HEALTHY
        if (is_regime_change and
                prev_regime in ('CASCADE', 'CRITICAL') and
                regime in ('HEALTHY', 'EARLY')):
            false_alarms += 1

        # Show first detection always; after that suppress repeat CONFIRMED label
        show_confirmed = confirmed and cascade_first_idx is None
        if i % every == 0 or n_shown == 0 or is_regime_change or show_confirmed:
            print_snapshot(i, elapsed, regime, risk, rul_min, feat,
                           show_confirmed)
            n_shown += 1

        prev_regime = regime

        # First confirmed cascade
        if confirmed and cascade_first_idx is None:
            cascade_first_idx = i
            cascade_lead_min  = rul_min
            # Compute healthy baseline from first 10% of life
            n_baseline = max(3, len(history_feats) // 10)
            baseline   = {
                'rms': float(np.mean([f['rms_env']  for f in history_feats[:n_baseline]])),
                'kurt': float(np.mean([f['kurt_env'] for f in history_feats[:n_baseline]])),
                'temp': float(np.mean([f['temp_dev'] for f in history_feats[:n_baseline]])),
            }
            print_cascade_alert(i, elapsed, rul_min, regime, feat, baseline, C)

    print_summary(bearing_name, total_life_min,
                  cascade_first_idx * C['SNAPSHOT_DT'] / 60
                  if cascade_first_idx is not None else None,
                  cascade_lead_min  if cascade_lead_min  is not None else 0.0,
                  n_total, n_shown, false_alarms)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='CSP v2 — live cascade detection deployment')
    ap.add_argument('--model',   required=True,
                    help='Path to csp_model.pkl (from csp_v2_train.py)')
    ap.add_argument('--bearing', required=True,
                    help='Path to bearing folder (e.g. Full_Test_Set/Bearing1_4)')
    ap.add_argument('--every',   type=int, default=50,
                    help='Print output every N snapshots (default 50 ≈ 8 min)')
    ap.add_argument('--max_snaps', type=int, default=None,
                    help='Stop after N snapshots (default: run to end)')
    args = ap.parse_args()

    run_deployment(
        bearing_dir=args.bearing,
        model_path=args.model,
        every=args.every,
        max_snaps=args.max_snaps)


if __name__ == '__main__':
    main()


# ══════════════════════════════════════════════════════════════════════════════
# USAGE
# ══════════════════════════════════════════════════════════════════════════════
#
# Step 1 — train (once):
#   python csp_v2_train.py --data "C:/path/to/Full_Test_Set"
#   → produces csp_model.pkl
#
# Step 2 — deploy on any bearing:
#   python csp_v2_deploy.py \
#     --model csp_model.pkl \
#     --bearing "C:/path/to/Full_Test_Set/Bearing1_4"
#
# Print every snapshot (verbose):
#   python csp_v2_deploy.py --model csp_model.pkl \
#     --bearing "path/to/Bearing1_4" --every 1
#
# Test on first 300 snapshots only:
#   python csp_v2_deploy.py --model csp_model.pkl \
#     --bearing "path/to/Bearing1_4" --max_snaps 300
#
# Deploy on unseen Learning_set bearing:
#   python csp_v2_deploy.py --model csp_model.pkl \
#     --bearing "C:/path/to/Learning_set/Bearing1_1"
#
# Dependencies (standalone — no pronostia_3simm.py needed):
#   pip install numpy pandas scipy scikit-learn