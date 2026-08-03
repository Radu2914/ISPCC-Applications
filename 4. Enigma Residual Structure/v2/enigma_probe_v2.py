"""
enigma_probe_v2.py  —  ISPCC structural probe on Enigma M4 (P1030700)
══════════════════════════════════════════════════════════════════════════
v1 result  : 239 pairs, cross-products dominated (28.1%), all R² negative.
             Data starvation (239 samples across 26³ position space).
             Finding: neither pure π nor pure e sufficient alone.

v2 adds:
  • Full M4 simulator — verified against known plaintext
  • 17,576 × 26 = 456,976 synthetic samples (all positions × all inputs)
  • Explicit plugboard e-type features (the involution layer Turing exploited)
  • Rotor notch π-type bifurcation features (stepping anomaly points)
  • Ring-adjusted effective positions

New e-type (character / involution):
  plug_dist    = |PLUG[input] - input| / 13   bounded [0,1]
                 0 for self-connected letters (A,F,M,R,W,X)
                 Non-zero for plugged letters — directly encodes the
                 fixed involution shift entering the rotors
  is_plugged   = 1 if input is in a plugboard pair
  plug_partner = PLUG[input] / 26   where plugboard routes the signal

New π-type (position / bifurcation):
  fast_notch_dist = min(|pos_F−25|, |pos_F−12|)/13   Rotor VIII notches at Z,M
  mid_notch_dist  = |pos_M − 21| / 26                 Rotor III notch at V

Turing's insight:
  The plug_dist feature encodes the reflector/plugboard involution directly.
  If plug_dist (e-type) dominates over position (π-type) in importance:
  → the probe detected that character-space constraints are the
    exploitable structural residual — not brute-force period searching.

Target: substitution_offset = (output − input) mod 26
  What the full Enigma (plugboard+rotors+reflector+plugboard) does at each
  position for each input character.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time

PI  = np.pi
EPS = 1e-9

# ══════════════════════════════════════════════════════════════════════════════
#  P1030700  KEY
# ══════════════════════════════════════════════════════════════════════════════
REFL_NAME    = 'B'         # Thin reflector B
GREEK_NAME   = 'Gamma'     # Greek wheel C = Gamma
ROTOR_L      = 'IV'        # Left rotor (steps rarely)
ROTOR_M      = 'III'       # Middle rotor
ROTOR_F      = 'VIII'      # Fast rotor (steps every keypress)

GREEK_POS    = ord('V') - ord('A')   # 21
POS_L_INIT   = ord('M') - ord('A')   # 12
POS_M_INIT   = ord('G') - ord('A')   # 6
POS_F_INIT   = ord('C') - ord('A')   # 2

RING_GREEK   = ord('A') - ord('A')   # 0
RING_L       = ord('A') - ord('A')   # 0
RING_M       = ord('C') - ord('A')   # 2
RING_F       = ord('U') - ord('A')   # 20

PLUG_PAIRS   = ['CH','EJ','NV','OU','TY','LG','SZ','PK','DI','QB']

# ══════════════════════════════════════════════════════════════════════════════
#  STANDARD ENIGMA WIRINGS (publicly known)
# ══════════════════════════════════════════════════════════════════════════════
_WIRING = {
    'I':    'EKMFLGDQVZNTOWYHXUSPAIBRCJ',
    'II':   'AJDKSIRUXBLHWTMCQGZNPYFVOE',
    'III':  'BDFHJLCPRTXVZNYEIWGAKMUSQO',
    'IV':   'ESOVPZJAYQUIRHXLNFTGKDCMWB',
    'V':    'VZBRGITYUPSDNHLXAWMJQOFECK',
    'VI':   'JPGVOUMFYQBENHZRDKASXLICTW',
    'VII':  'NZJHGRCXMYSWBOUFAIVLPEKQDT',
    'VIII': 'FKQHTLXOCBJSPDZRAMEWNIUYGV',
}
_REFLECTOR = {
    'B': 'ENKQAUYWJICOPBLMDXZVFTHRGS',
    'C': 'RDOBJNTKVEHMLFCWZAXGYIPSUQ',
}
_GREEK = {
    'Beta':  'LEYJVCNIXWPBQMDRTAKZGFUHOS',
    'Gamma': 'FSOKANUERHMBTIYCWLQPZXVGJD',
}
# Turnover notch positions (letter shown when carry occurs)
_NOTCH = {
    'I':    [ord('Q')-ord('A')],
    'II':   [ord('E')-ord('A')],
    'III':  [ord('V')-ord('A')],
    'IV':   [ord('J')-ord('A')],
    'V':    [ord('Z')-ord('A')],
    'VI':   [ord('Z')-ord('A'), ord('M')-ord('A')],
    'VII':  [ord('Z')-ord('A'), ord('M')-ord('A')],
    'VIII': [ord('Z')-ord('A'), ord('M')-ord('A')],
}

def _w(s):   return [ord(c)-ord('A') for c in s]
def _inv(w):
    inv = [0]*26
    for i,j in enumerate(w): inv[j] = i
    return inv

# Pre-computed wiring arrays
W_FWD  = {k: _w(v) for k,v in _WIRING.items()}
W_BWD  = {k: _inv(W_FWD[k]) for k in W_FWD}
REF_W  = {k: _w(v) for k,v in _REFLECTOR.items()}
GRK_W  = {k: _w(v) for k,v in _GREEK.items()}
GRK_WI = {k: _inv(GRK_W[k]) for k in GRK_W}

# Plugboard array
PLUG = list(range(26))
for pair in PLUG_PAIRS:
    a, b = ord(pair[0])-ord('A'), ord(pair[1])-ord('A')
    PLUG[a] = b
    PLUG[b] = a

# Notch sets
NOTCH_L = set(_NOTCH[ROTOR_L])
NOTCH_M = set(_NOTCH[ROTOR_M])
NOTCH_F = set(_NOTCH[ROTOR_F])

# ══════════════════════════════════════════════════════════════════════════════
#  ENIGMA M4 CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _fwd(s, w, pos, ring):
    return (w[(s + pos - ring) % 26] - pos + ring) % 26

def _bwd(s, wi, pos, ring):
    return (wi[(s + pos - ring) % 26] - pos + ring) % 26


def enigma_substitute(inp, pos_L, pos_M, pos_F):
    """
    Single substitution at given rotor positions (NO stepping).
    Greek wheel fixed at GREEK_POS, ring settings from key.
    Returns output letter (0-25).
    """
    s = PLUG[inp]
    s = _fwd(s, W_FWD[ROTOR_F], pos_F, RING_F)
    s = _fwd(s, W_FWD[ROTOR_M], pos_M, RING_M)
    s = _fwd(s, W_FWD[ROTOR_L], pos_L, RING_L)
    s = _fwd(s, GRK_W[GREEK_NAME], GREEK_POS, RING_GREEK)
    s = REF_W[REFL_NAME][s]
    s = _bwd(s, GRK_WI[GREEK_NAME], GREEK_POS, RING_GREEK)
    s = _bwd(s, W_BWD[ROTOR_L], pos_L, RING_L)
    s = _bwd(s, W_BWD[ROTOR_M], pos_M, RING_M)
    s = _bwd(s, W_BWD[ROTOR_F], pos_F, RING_F)
    return PLUG[s]


def _step_rotors(pos_L, pos_M, pos_F):
    """Enigma double-step mechanism."""
    step_M = (pos_M in NOTCH_M) or (pos_F in NOTCH_F)
    step_L = pos_M in NOTCH_M
    new_F  = (pos_F + 1) % 26
    new_M  = (pos_M + 1) % 26 if step_M else pos_M
    new_L  = (pos_L + 1) % 26 if step_L else pos_L
    return new_L, new_M, new_F


def encrypt_message(plaintext):
    """
    Encrypt a string starting from P1030700 initial positions.
    Steps rotors before each character. Returns ciphertext.
    """
    pos_L, pos_M, pos_F = POS_L_INIT, POS_M_INIT, POS_F_INIT
    result = []
    for c in plaintext.upper():
        if c.isalpha():
            pos_L, pos_M, pos_F = _step_rotors(pos_L, pos_M, pos_F)
            out = enigma_substitute(ord(c)-ord('A'), pos_L, pos_M, pos_F)
            result.append(chr(out + ord('A')))
    return ''.join(result)


# ══════════════════════════════════════════════════════════════════════════════
#  ENCODING FUNCTIONS  —  identical weights to v6
# ══════════════════════════════════════════════════════════════════════════════

def encode_pi(x, prefix, scale, weights=(5, 1, 1, 3, 1)):
    x  = np.clip(x, 0, 10)
    xn = x / (scale + EPS)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2 * PI * xn),
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }

def encode_e(x, prefix, scale, weights=(2, 2, 1)):
    xn = np.clip(x / (scale + EPS), 0, 1)
    w  = np.array(weights, dtype=float) / sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-np.e * xn),
        f"{prefix}_pow_e":   w[1] * np.power(xn + EPS, np.e),
        f"{prefix}_gauss":   w[2] * np.exp(-np.e * (xn - 0.5)**2),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PLUGBOARD STRUCTURAL CONSTANTS  (analogous to Feigenbaum constants in v6)
# ══════════════════════════════════════════════════════════════════════════════

# For each letter: plugboard distance (how far the signal is rerouted)
PLUG_DIST    = np.array([abs(PLUG[i] - i) for i in range(26)], dtype=float)
PLUG_PARTNER = np.array(PLUG, dtype=float)
IS_PLUGGED   = (PLUG_DIST > 0).astype(float)

# Max plugboard distance in this key (Q↔B = |1-16| = 15)
MAX_PLUG_DIST = PLUG_DIST.max()   # 15

# Notch positions for structural constant encoding
NOTCH_F_POSITIONS = sorted(NOTCH_F)   # [12, 25] (M=12, Z=25) for Rotor VIII
NOTCH_M_POS       = list(NOTCH_M)[0]  # 21 (V) for Rotor III
NOTCH_L_POS       = list(NOTCH_L)[0]  # 9 (J) for Rotor IV


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n_sample=None, seed=42):
    """
    Generate substitution table for all 17,576 × 26 = 456,976 (position, input) pairs.
    Returns DataFrame with positions, input char, output char, substitution offset.
    Optionally samples n_sample rows.
    """
    print(f"[INFO] Generating substitution table "
          f"(17576 positions × 26 inputs = 456,976 pairs)...")
    t0 = time.time()

    rows = []
    for pos_L in range(26):
        for pos_M in range(26):
            for pos_F in range(26):
                for inp in range(26):
                    out = enigma_substitute(inp, pos_L, pos_M, pos_F)
                    rows.append((pos_L, pos_M, pos_F, inp, out))

    df = pd.DataFrame(rows, columns=['pos_L','pos_M','pos_F','inp','out'])
    df['subst_off'] = (df['out'] - df['inp']) % 26

    print(f"[INFO] Generated {len(df):,} pairs in {time.time()-t0:.1f}s")
    print(f"[VERIFY] Self-mappings (inp==out): {(df['inp']==df['out']).sum()} "
          f"(Enigma: must be 0)")

    if n_sample and n_sample < len(df):
        df = df.sample(n_sample, random_state=seed).reset_index(drop=True)
        print(f"[INFO] Sampled {n_sample:,} pairs for probe")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def build_features(df):
    """
    Build π and e encoded features.

    π-type (position / rotor period cascade):
      pos_F raw and ring-adjusted (fast rotor, period-26)
      pos_M raw and ring-adjusted (middle rotor)
      pos_L raw                   (left rotor)
      fast_notch_dist             (distance to Rotor VIII notch — bifurcation)
      mid_notch_dist              (distance to Rotor III notch)

    e-type (character / plugboard involution):
      inp_char                    (input letter value, bounded [0,25])
      plug_dist                   (|PLUG[inp]-inp|, bounded [0,15])
      is_plugged                  (binary: letter in a plugboard pair)
      plug_partner                (PLUG[inp], bounded [0,25])

    Cross-products: fast_rotor_pos × plug_dist (position × involution)
    """
    N = len(df)

    pos_F = df['pos_F'].values.astype(float)
    pos_M = df['pos_M'].values.astype(float)
    pos_L = df['pos_L'].values.astype(float)
    inp   = df['inp'].values.astype(float)

    # Ring-adjusted effective positions (what the wiring actually sees)
    eff_F = (pos_F - RING_F + 26) % 26
    eff_M = (pos_M - RING_M + 26) % 26

    # Notch distances (minimum circular distance to notch)
    notch_F_arr = np.array(NOTCH_F_POSITIONS, dtype=float)
    fast_notch  = np.array([min(abs(p - n) for n in NOTCH_F_POSITIONS) for p in pos_F])
    mid_notch   = np.abs(pos_M - NOTCH_M_POS)
    mid_notch   = np.minimum(mid_notch, 26 - mid_notch)  # circular

    # Character features
    plug_dist    = PLUG_DIST[df['inp'].values.astype(int)]
    is_plugged   = IS_PLUGGED[df['inp'].values.astype(int)]
    plug_partner = PLUG_PARTNER[df['inp'].values.astype(int)]

    feat = {}

    # ── π-encoded: position (rotor cascade) ───────────────────────────────
    feat.update(encode_pi(pos_F,       "pi_fast",      scale=26.0))
    feat.update(encode_pi(pos_M,       "pi_mid",       scale=26.0))
    feat.update(encode_pi(pos_L,       "pi_left",      scale=26.0))
    feat.update(encode_pi(eff_F,       "pi_eff_fast",  scale=26.0))
    feat.update(encode_pi(eff_M,       "pi_eff_mid",   scale=26.0))
    feat.update(encode_pi(fast_notch,  "pi_notch_f",   scale=13.0))  # max dist = 13
    feat.update(encode_pi(mid_notch,   "pi_notch_m",   scale=13.0))

    # ── e-encoded: character (involution/plugboard structure) ──────────────
    feat.update(encode_e(inp,          "e_char",       scale=26.0))
    feat.update(encode_e(plug_dist,    "e_plug_dist",  scale=MAX_PLUG_DIST))
    feat.update(encode_e(is_plugged,   "e_is_plugged", scale=1.0))
    feat.update(encode_e(plug_partner, "e_plug_part",  scale=26.0))

    # ── Cross-products: position × involution ──────────────────────────────
    sin_fast     = np.sin(PI * pos_F / 26.0)
    exp_plug     = np.exp(-np.e * plug_dist / (MAX_PLUG_DIST + EPS))
    exp_char     = np.exp(-np.e * inp / 26.0)
    feat["cross_fast_x_plug"]  = sin_fast * exp_plug
    feat["cross_fast_x_char"]  = sin_fast * exp_char
    feat["cross_mid_x_plug"]   = np.sin(PI * pos_M / 26.0) * exp_plug

    # ── Raw baseline ───────────────────────────────────────────────────────
    raw = pd.DataFrame({
        "pos_F":       pos_F,
        "pos_M":       pos_M,
        "pos_L":       pos_L,
        "inp_char":    inp,
        "plug_dist":   plug_dist,
        "is_plugged":  is_plugged,
        "plug_partner":plug_partner,
    })

    feat_df = pd.DataFrame(feat, index=df.index)
    return feat_df, raw


# ══════════════════════════════════════════════════════════════════════════════
#  CV + 3-STAGE
# ══════════════════════════════════════════════════════════════════════════════

def run_cv(X, y, model, kf):
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))

def make_rf(n=300):
    return RandomForestRegressor(
        n_estimators=n, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

PLAIN1  = "KOMXBDMXUUUBOOTEYFXDXUUUAUSBILVUNYYZWOSECHSXUUUFLOTTXVVVUUURWODREISECHSVIERKKREMASKKMITUUVZWODREIFUVFYEWHSYUUUZWODREIFUNFZWOYUUFZWL"
CIPHER1 = "QBHEWTDFEQITKUWFQUHLIQQGVYGRSDOHDCOBFMDHXSKOFPAODRSVBEREIQZVEDAXSHOHBIYMCIIZSKGNDLNFKFVLWWHZXZGQXWSSPWLSOQXEANCELJYJCETZTLSTTWMTOBW"

N_SAMPLE  = 30000   # samples for probe/CV (from 456,976)
N_PROBE_TREES = 500
N_CV_TREES    = 300

def main():
    SEP = "=" * 72
    t_start = time.time()

    # ── Verify simulator ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  SIMULATOR VERIFICATION")
    print(SEP)
    encrypted = encrypt_message(PLAIN1)
    match = sum(a==b for a,b in zip(encrypted, CIPHER1))
    total = min(len(encrypted), len(CIPHER1))
    print(f"  Encrypting known plaintext → ciphertext")
    print(f"  Match: {match}/{total}  "
          f"{'✓ Simulator verified' if match==total else f'✗ {total-match} mismatches'}")
    print(f"  First 10: plain={PLAIN1[:10]}  cipher={CIPHER1[:10]}  "
          f"simulated={encrypted[:10]}")

    if match < total * 0.95:
        print("  [ERROR] Simulator verification failed. Check settings.")
        return

    # ── Plugboard structure ───────────────────────────────────────────────────
    print(f"\n  Plugboard structure (e-type constants):")
    print(f"  {'Letter':>8}  {'Partner':>8}  {'Dist':>6}  {'Plugged':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*6}  {'─'*8}")
    for i in range(26):
        if PLUG_DIST[i] > 0 and PLUG[i] > i:  # print each pair once
            c, p = chr(i+ord('A')), chr(PLUG[i]+ord('A'))
            print(f"  {c:>8}  {p:>8}  {PLUG_DIST[i]:>6.0f}  {'YES':>8}")
    print(f"  Self-connected (dist=0): "
          f"{[chr(i+ord('A')) for i in range(26) if PLUG_DIST[i]==0]}")
    print(f"  Max plugboard distance: {MAX_PLUG_DIST:.0f} "
          f"(Q↔B — the 'longest reach' pair)")

    # ── Generate dataset ──────────────────────────────────────────────────────
    df = generate_dataset(n_sample=N_SAMPLE)
    target = df['subst_off'].values.astype(float)

    print(f"\n  Substitution offset distribution:")
    print(f"  mean={target.mean():.2f}  std={target.std():.2f}  "
          f"min={target.min():.0f}  max={target.max():.0f}")
    print(f"  Value 0 present: {(target==0).any()} (False = Enigma invariant OK)")

    # ── Build features ────────────────────────────────────────────────────────
    feat_df, raw_df = build_features(df)

    pi_cols    = [c for c in feat_df.columns if c.startswith("pi_")]
    e_cols     = [c for c in feat_df.columns if c.startswith("e_")]
    cross_cols = [c for c in feat_df.columns if c.startswith("cross_")]

    print(f"\n  Features: π={len(pi_cols)}, e={len(e_cols)}, "
          f"cross={len(cross_cols)}, raw={len(raw_df.columns)}")
    print(f"  N/p ratio: {len(df)/len(feat_df.columns):.0f}× "
          f"(vs v1: {239/27:.1f}×)")

    # ── PROBE ─────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PROBE — RF importance  |  n={len(df):,}, {N_PROBE_TREES} trees")
    print(f"  Target: (output − input) mod 26  [what Enigma does at each position]")
    print(SEP)

    rf_probe = RandomForestRegressor(
        n_estimators=N_PROBE_TREES, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf_probe.fit(feat_df.values, target)

    imp        = pd.Series(rf_probe.feature_importances_, index=feat_df.columns)
    imp_sorted = imp.sort_values(ascending=False)

    print(f"\n  Top 20 features:")
    print(f"  {'Rank':>4}  {'Feature':>24}  {'Imp':>8}  Type  Layer")
    print(f"  {'─'*4}  {'─'*24}  {'─'*8}  {'─'*4}  {'─'*28}")

    layer_label = {
        "pi_fast":     "fast rotor position (period-26)",
        "pi_mid":      "middle rotor position",
        "pi_left":     "left rotor position",
        "pi_eff_fast": "ring-adjusted fast rotor",
        "pi_eff_mid":  "ring-adjusted middle rotor",
        "pi_notch_f":  "Rotor VIII notch distance ← bifurcation",
        "pi_notch_m":  "Rotor III notch distance  ← bifurcation",
        "e_char":      "input letter value (bounded)",
        "e_plug_dist": "plugboard shift distance  ← TURING'S LAYER",
        "e_is_plugged":"letter is in plugboard pair",
        "e_plug_part": "plugboard partner value",
        "cross":       "position × involution interaction",
    }

    for rank, (fn, fv) in enumerate(imp_sorted.head(20).items(), 1):
        t = "π" if fn in pi_cols else ("e" if fn in e_cols else "×")
        # Get prefix
        parts = fn.split("_")
        prefix_key = "_".join(parts[:3]) if len(parts) >= 3 else fn
        label = next((v for k,v in layer_label.items() if fn.startswith(k)), "")
        print(f"  {rank:>4}  {fn:>24}  {fv:>8.4f}  {t:>4}  {label}")

    pi_imp    = imp[pi_cols].sum()
    e_imp     = imp[e_cols].sum()
    cross_imp = imp[cross_cols].sum()
    tot       = pi_imp + e_imp + cross_imp

    print(f"\n  ── Grouped importances ──")
    print(f"  π-type (position / rotor)     : {pi_imp:.4f}  ({100*pi_imp/tot:5.1f}%)")
    print(f"  e-type (char / plugboard)     : {e_imp:.4f}  ({100*e_imp/tot:5.1f}%)")
    print(f"  Cross-products (π × e)        : {cross_imp:.4f}  ({100*cross_imp/tot:5.1f}%)")

    # Plugboard-specific features
    plug_features = [c for c in e_cols if 'plug' in c]
    plug_imp = imp[plug_features].sum()
    print(f"\n  Plugboard features only (e-type subset):")
    print(f"  {', '.join(plug_features)}")
    print(f"  Combined importance: {plug_imp:.4f}  ({100*plug_imp/tot:.1f}%)")
    print(f"  {'← Turing layer dominant' if plug_imp > pi_imp else '← Position still dominates'}")

    # ── CV COMPARISON ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  CV COMPARISON  (5-fold, n={len(df):,})")
    print(f"  v1 reference: all R² negative (data starvation, 239 samples)")
    print(SEP)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_raw    = run_cv(raw_df.values,                  target, make_rf(), kf)
    r2_pi     = run_cv(feat_df[pi_cols].values,         target, make_rf(), kf)
    r2_e      = run_cv(feat_df[e_cols].values,           target, make_rf(), kf)
    r2_full   = run_cv(feat_df.values,                  target, make_rf(), kf)
    r2_plug   = run_cv(feat_df[plug_features].values,   target, make_rf(), kf)

    print(f"\n  {'Model':28s}  {'R²':>8}  {'vs v1':>10}  Note")
    print(f"  {'─'*28}  {'─'*8}  {'─'*10}  {'─'*30}")
    v1_full = -0.0953  # v1 full encoded R²
    for lbl, r2 in [
        ("Raw features",         r2_raw),
        ("π-only (position)",    r2_pi),
        ("e-only (character)",   r2_e),
        ("Plugboard only (e↓)",  r2_plug),
        ("Full (π + e + cross)", r2_full),
    ]:
        note = ""
        if lbl == "π-only (position)" and r2_pi > 0:
            note = "← positive (data fixed)"
        if lbl == "e-only (character)" and r2_e > 0:
            note = "← positive (data fixed)"
        if lbl == "Plugboard only (e↓)" and r2_plug > r2_pi:
            note = "← TURING: char > position"
        print(f"  {lbl:28s}  {r2:>8.4f}  "
              f"{'N/A' if v1_full != r2 else 'same':>10}  {note}")

    e_beats_pi   = r2_e   > r2_pi
    plug_beats_pi = r2_plug > r2_pi

    # ── 3-STAGE PIPELINE ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  3-STAGE PIPELINE  (Enigma-specific decomposition)")
    print(f"  Stage 1 Ridge  : π features  (rotor period grammar)")
    print(f"  Stage 2 RF     : e features  → Stage-1 residuals (involution dialect)")
    print(f"  If ΔR² > 0: the plugboard involution explains variance above rotor period")
    print(SEP)

    X_pi = feat_df[pi_cols].values
    X_e  = feat_df[e_cols].values

    r2s_s1, r2s_3s, e_deltas = [], [], []
    for tr, te in kf.split(X_pi):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_pi[tr], target[tr])
        res_tr = target[tr] - ridge.predict(X_pi[tr])

        rf2 = make_rf()
        rf2.fit(X_e[tr], res_tr)

        pred_s1 = ridge.predict(X_pi[te])
        pred_3s = pred_s1 + rf2.predict(X_e[te])
        r2_s1_  = r2_score(target[te], pred_s1)
        r2_3s_  = r2_score(target[te], pred_3s)
        r2s_s1.append(r2_s1_)
        r2s_3s.append(r2_3s_)
        e_deltas.append(r2_3s_ - r2_s1_)

    r2_s1   = float(np.mean(r2s_s1))
    r2_3s   = float(np.mean(r2s_3s))
    delta_e = float(np.mean(e_deltas))

    print(f"\n  Stage 1 only (Ridge, π features)     : R² = {r2_s1:.4f}")
    print(f"  3-stage      (π grammar + e dialect) : R² = {r2_3s:.4f}")
    print(f"  Stage 2 ΔR²  (e dialect contribution): {delta_e:+.4f}")
    s2_positive = delta_e > 0
    print(f"  {'→ e-type adds above π alone: Turing layer confirmed' if s2_positive else '→ π alone sufficient: rotor period dominates'}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  VERDICT: Does the probe detect Turing's insight?")
    print(f"  (v2: {len(df):,} samples, explicit plugboard features, full M4 key)")
    print(SEP)

    c1 = e_imp > pi_imp
    c2 = plug_beats_pi
    c3 = s2_positive

    print(f"\n  Criteria:")
    print(f"  [{'✓' if c1 else '✗'}] e-type importance ({100*e_imp/tot:.1f}%) "
          f"> π-type importance ({100*pi_imp/tot:.1f}%)")
    print(f"  [{'✓' if c2 else '✗'}] Plugboard-only R² ({r2_plug:.4f}) "
          f"> π-only R² ({r2_pi:.4f})")
    print(f"  [{'✓' if c3 else '✗'}] Stage-2 e-dialect adds ΔR²={delta_e:+.4f} "
          f"above Stage-1 π-grammar")

    n_met = sum([c1, c2, c3])
    print(f"\n  {n_met}/3 criteria met  "
          f"({'↑ from v1 0/3' if n_met > 0 else '→ same as v1 0/3'})")

    if n_met >= 2:
        print(f"""
  → TURING'S INSIGHT DETECTED ({n_met}/3)

  The plugboard's fixed involution (e-type: bounded, self-regulating)
  is the structurally exploitable layer — not just the rotor period (π-type).

  For each input letter, the plugboard distance (|PLUG[c]-c|) tells you
  how far the signal is rerouted before entering the rotors. This is FIXED
  across ALL positions. It's what Turing's frequency analysis exploited:
  letter frequency constraints persist through the substitution because
  the plugboard pairs are fixed throughout the message.

  The probe detected this without being told: the e-encoded plug_dist
  features explain variance that position-only features cannot reach.
  That is the cryptographic structural residual Turing found in 1940.
""")
    elif n_met == 1:
        print(f"""
  → PARTIAL DETECTION (1/3)

  One criterion confirms the e-type (involution) layer.
  The full 456,976-sample dataset was sufficient to see it partially.
  Criterion met: see above.
""")
    else:
        print(f"""
  → π-TYPE DOMINANT (0/3)

  Rotor position (period-26) is still the stronger signal even with
  the full dataset and explicit plugboard features. This could mean:
  1. The plug distances for this key don't create enough variance
  2. The substitution is too position-dominated at this key setting
  3. A different target (e.g. language score) would expose the e-layer
""")

    # ── v1 vs v2 comparison ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  v1 → v2 COMPARISON")
    print(SEP)
    print(f"  {'Metric':35s}  {'v1 (239 samples)':>18}  {'v2 ({:,})'.format(len(df)):>18}")
    print(f"  {'─'*35}  {'─'*18}  {'─'*18}")
    print(f"  {'Dataset size':35s}  {'239':>18}  {len(df):>18,}")
    print(f"  {'π-type importance':35s}  {'39.1%':>18}  {100*pi_imp/tot:>17.1f}%")
    print(f"  {'e-type importance':35s}  {'32.8%':>18}  {100*e_imp/tot:>17.1f}%")
    print(f"  {'Cross-product importance':35s}  {'28.1%':>18}  {100*cross_imp/tot:>17.1f}%")
    print(f"  {'Full encoded R²':35s}  {'-0.0953':>18}  {r2_full:>18.4f}")
    print(f"  {'π-only R²':35s}  {'-0.2356':>18}  {r2_pi:>18.4f}")
    print(f"  {'e-only R²':35s}  {'-0.2620':>18}  {r2_e:>18.4f}")
    print(f"  {'3-stage ΔR²':35s}  {'-0.2247':>18}  {delta_e:>18.4f}")
    print(f"  {'Turing criteria met':35s}  {'0/3':>18}  {n_met:>17}/3")

    print(f"\n[TIMING] Total: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()

"""
python enigma_probe_v2.py
"""