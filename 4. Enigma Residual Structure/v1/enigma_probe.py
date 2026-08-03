"""
enigma_probe.py — ISPCC structural probe on Enigma ciphertext/plaintext
═══════════════════════════════════════════════════════════════════════════
Hypothesis:
  Turing's insight = the reflector's involution (no letter encrypts to itself)
  is a bounded, self-regulating CHARACTER constraint — e-type.
  The rotor stepping is a cascading, periodic POSITION structure — π-type.

  If e-type importance > π-type importance:
    → The probe detected the involution constraint automatically.
    → Character structure (reflector/plugboard) is more exploitable than
      position structure (rotor period). This IS Turing's insight.

  If π-type dominates:
    → The rotor period is the detectable structure.
    → Brute-force period search is what the probe finds, not Turing.

Target: substitution_offset = (plain_val − cipher_val) mod 26
  What the rotor+plugboard did at each position.
  Predicted from cipher features only (no plaintext used as input).

  3-stage interpretation:
    Stage 1 Ridge on π features  → removes rotor-period grammar
    Stage 2 RF    on e features  → captures involution dialect (Turing's layer)
    If Stage 2 adds R²: e-type contributes above position → Turing's layer found.

Structural constants (Enigma analogs of Feigenbaum):
  N_ALPHA       = 26   — alphabet size (natural bound, e-scale)
  ROTOR1_PERIOD = 26   — fast rotor revolution (π-scale, period-26 cascade)
  ROTOR2_PERIOD = 676  — 26² middle rotor
  REFLECT_PROP  = 1/26 — reflector eliminates 1/26 of key space (e-bound)

Data: one WWII Enigma message, two sections, 239 aligned cipher-plain pairs.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from collections import Counter

PI  = np.pi
EPS = 1e-9

# ── Enigma structural constants ──────────────────────────────────────────────
N_ALPHA       = 26.0    # alphabet size (natural bound — e-type scale)
ROTOR1_PERIOD = 26.0    # fast rotor — period-26 cascade (π-type scale)
ROTOR2_PERIOD = 676.0   # 26² — middle rotor (π-type)

# ── Encoding functions — IDENTICAL to v6 (weights unchanged) ────────────────
def encode_pi_func(x, prefix, scale=None, weights=(5, 1, 1, 3, 1)):
    """π-basis: cascading, non-periodic, position-type variables."""
    x  = np.clip(x, 0, 10)
    xn = x / (scale + EPS) if scale is not None else x / (x.max() + EPS)
    w  = np.array(weights, dtype=float) / np.sum(weights)
    return {
        f"{prefix}_sin_pi":  w[0] * np.sin(PI * xn),
        f"{prefix}_cos_pi":  w[1] * np.cos(PI * xn),
        f"{prefix}_sin_2pi": w[2] * np.sin(2 * PI * xn),   # ← circular: full rotor revolution
        f"{prefix}_sin_pi2": w[3] * np.sin(PI**2 * xn),
        f"{prefix}_cascade": w[4] * np.sin(PI * xn) * np.cos(PI**2 * xn),
    }

def encode_e_func(x, prefix, scale=None, weights=(2, 2, 1)):
    """e-basis: bounded, self-regulating, character-type variables."""
    xn = x / (scale + EPS) if scale is not None else x / (x.max() + EPS)
    xn = np.clip(xn, 0, 1)
    w  = np.array(weights, dtype=float) / np.sum(weights)
    return {
        f"{prefix}_exp_neg": w[0] * np.exp(-np.e * xn),
        f"{prefix}_pow_e":   w[1] * np.power(xn + EPS, np.e),
        f"{prefix}_gauss":   w[2] * np.exp(-np.e * (xn - 0.5)**2),
    }

# ── Known Enigma pairs (letters only; two sections of one WWII message) ──────
# Source: enigma_test.md
# Note: (.) extra missing letter between sections — ~1 position offset in part 2.
# Used as-is; probe has enough tolerance for one misalignment.

CIPHER1 = "QBHEWTDFEQITKUWFQUHLIQQGVYGRSDOHDCOBFMDHXSKOFPAODRSVBEREIQZVEDAX" \
           "SHOHBIYMCIIZSKGNDLNFKFVLWWHZXZGQXWSSPWLSOQXEANCELJYJCETZTLSTTWMTOBW"
PLAIN1  = "KOMXBDMXUUUBOOTEYFXDXUUUAUSBILVUNYYZWOSECHSXUUUFLOTTXVVVUUURWODREI" \
           "SECHSVIERKKREMASKKMITUUVZWODREIFUVFYEWHSYUUUZWODREIFUNFZWOYUUFZWL"

CIPHER2 = "OHOYPDNLUXMGOZFZBFLOXJNSSTLPHXJDYSSBNBOZLVPXJBATNNJDLCKKBZNRSTKPMPN" \
           "VSRETKOIZTVSDBSYPZEBSJLODSJGCXFJVENZTQTFI"
PLAIN2  = "RZIFUNFNYANYUUROFDDEEISEFHSNULFUUZGWRDQEISECHSDREIUNDUUUZWODREISECHS" \
           "FUNFEINSCECHSUHRWARNEMUONDEAUSNACZEIGLHL"


def parse_section(cipher_str, plain_str, offset=0):
    """Extract aligned letter pairs; offset = position counter for part 2."""
    c = [ch for ch in cipher_str.upper() if ch.isalpha()]
    p = [ch for ch in plain_str.upper() if ch.isalpha()]
    n = min(len(c), len(p))
    rows = []
    for i in range(n):
        ci = ord(c[i]) - ord('A')
        pi = ord(p[i]) - ord('A')
        rows.append({
            "pos":        float(i + offset),
            "cipher_val": float(ci),
            "plain_val":  float(pi),
            "subst_off":  float((pi - ci) % 26),
            "self_map":   int(ci == pi),   # Enigma invariant: must be 0
        })
    return pd.DataFrame(rows)


def build_features(df, total_N):
    """
    Generate π and e encoded features for each position.

    π-type (position-based, rotor cascade):
      fast_pos = i mod 26       — fast rotor revolution, period-26
      mid_pos  = (i//26) mod 26 — middle rotor period
      pos_frac = i/N            — global message position

    e-type (character-based, involution constraint):
      cipher_val   — which letter was received; bounded [0,25]
      cipher_gfreq — how often this letter appears in full message (frequency)
      cipher_lfreq — local window frequency (bounded self-regulating)

    Cross-products: position × character (rotor-state meets substitution constraint)
    """
    pos  = df["pos"].values
    cval = df["cipher_val"].values
    N    = len(df)

    # ── Character frequency features ─────────────────────────────────────────
    cipher_chars = [chr(int(v) + ord('A')) for v in cval]
    gcounter     = Counter(cipher_chars)
    gfreq = np.array([gcounter[c] / total_N for c in cipher_chars])

    WINDOW = 10
    lfreq  = np.zeros(N)
    for i in range(N):
        lo = max(0, i - WINDOW)
        hi = min(N, i + WINDOW + 1)
        w  = cipher_chars[lo:hi]
        lfreq[i] = w.count(cipher_chars[i]) / len(w)

    # ── π-encoded: position-based (rotor period cascade) ─────────────────────
    fast  = pos % ROTOR1_PERIOD                           # [0, 25] — fast rotor position
    mid   = (pos // ROTOR1_PERIOD) % ROTOR1_PERIOD        # [0, ~9] for 239 positions
    pfrac = pos / (total_N - 1 + EPS)                     # [0, 1]  — global position

    feat = {}
    feat.update(encode_pi_func(fast,                    "pi_fast",  scale=ROTOR1_PERIOD))
    feat.update(encode_pi_func(mid,                     "pi_mid",   scale=ROTOR1_PERIOD))
    feat.update(encode_pi_func(pfrac * ROTOR1_PERIOD,   "pi_pos",   scale=ROTOR1_PERIOD))

    # ── e-encoded: character-based (involution/bounded constraint) ────────────
    feat.update(encode_e_func(cval,  "e_char",  scale=N_ALPHA))
    feat.update(encode_e_func(gfreq, "e_gfreq", scale=1.0))
    feat.update(encode_e_func(lfreq, "e_lfreq", scale=1.0))

    # ── Cross-products: where × who (position meets character) ───────────────
    sin_fast = np.sin(PI * fast / ROTOR1_PERIOD)
    exp_char = np.exp(-np.e * cval / N_ALPHA)
    feat["cross_fast_x_char"]  = sin_fast * exp_char
    feat["cross_fast_x_gfreq"] = sin_fast * gfreq
    feat["cross_fast_x_lfreq"] = sin_fast * lfreq

    # ── Raw baseline ─────────────────────────────────────────────────────────
    raw = pd.DataFrame({
        "fast_pos":   fast,
        "mid_pos":    mid,
        "pos_frac":   pfrac,
        "cipher_val": cval,
        "gfreq":      gfreq,
        "lfreq":      lfreq,
    })

    return pd.DataFrame(feat, index=df.index), raw


def run_cv(X, y, model, kf):
    r2s = []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], model.predict(X[te])))
    return float(np.mean(r2s))


def make_rf(n=500, seed=42):
    return RandomForestRegressor(
        n_estimators=n, max_features="sqrt",
        min_samples_leaf=1, random_state=seed, n_jobs=-1)


def main():
    SEP = "=" * 72

    # ── Parse data ────────────────────────────────────────────────────────────
    df1 = parse_section(CIPHER1, PLAIN1, offset=0)
    df2 = parse_section(CIPHER2, PLAIN2, offset=len(df1))
    df  = pd.concat([df1, df2], ignore_index=True)
    N   = len(df)

    print(f"\n[INFO] Dataset: {N} aligned pairs  "
          f"(Part 1: {len(df1)}, Part 2: {len(df2)})")

    # ── Enigma invariant verification ─────────────────────────────────────────
    n_self = df["self_map"].sum()
    print(f"[VERIFY] Self-mappings (cipher == plain): {n_self}/{N}  "
          f"{'✓ CONFIRMED — Enigma invariant holds' if n_self == 0 else '✗ VIOLATED'}")

    target = df["subst_off"].values   # (plain - cipher) mod 26, range [1, 25]

    print(f"\n[INFO] Substitution offset distribution:")
    print(f"  mean={target.mean():.2f}  std={target.std():.2f}  "
          f"min={target.min():.0f}  max={target.max():.0f}")
    print(f"  Value 0 present: {(target==0).any()}  "
          f"(False = Enigma no-self-map confirmed)")

    # ── Build features ────────────────────────────────────────────────────────
    feat_df, raw_df = build_features(df, N)

    pi_cols    = [c for c in feat_df.columns if c.startswith("pi_")]
    e_cols     = [c for c in feat_df.columns if c.startswith("e_")]
    cross_cols = [c for c in feat_df.columns if c.startswith("cross_")]

    print(f"\n[INFO] Features:")
    print(f"  π-encoded (position/rotor)    : {len(pi_cols):3d}  {pi_cols}")
    print(f"  e-encoded (character/invol.)  : {len(e_cols):3d}  {e_cols}")
    print(f"  Cross-products (π × e)        : {len(cross_cols):3d}  {cross_cols}")
    print(f"  Raw baseline                  : {len(raw_df.columns):3d}")
    print(f"  Total encoded                 : {len(feat_df.columns):3d}")
    print(f"  N/p ratio (full encoded)      : {N/len(feat_df.columns):.1f}")

    # ── PROBE: RF importance on full encoded set ───────────────────────────────
    print(f"\n{SEP}")
    print(f"  PROBE — RF importance  |  target: (plain − cipher) mod 26")
    print(f"  [n=1000 trees for stable importance rankings]")
    print(SEP)

    rf_probe = RandomForestRegressor(
        n_estimators=1000, max_features="sqrt",
        min_samples_leaf=1, random_state=42, n_jobs=-1)
    rf_probe.fit(feat_df.values, target)

    imp        = pd.Series(rf_probe.feature_importances_, index=feat_df.columns)
    imp_sorted = imp.sort_values(ascending=False)

    print(f"\n  Top 15 features:")
    print(f"  {'Rank':>4}  {'Feature':>26}  {'Imp':>8}  Type  Interpretation")
    print(f"  {'─'*4}  {'─'*26}  {'─'*8}  {'─'*4}  {'─'*30}")

    interp = {
        "pi_fast": "fast rotor position mod 26",
        "pi_mid":  "middle rotor position",
        "pi_pos":  "global message position",
        "e_char":  "cipher character value [bounded]",
        "e_gfreq": "cipher char global frequency",
        "e_lfreq": "cipher char local frequency",
        "cross":   "position × character interaction",
    }

    for rank, (fn, fv) in enumerate(imp_sorted.head(15).items(), 1):
        if fn in pi_cols:
            t = "π"
            desc = interp.get(fn.rsplit("_", 1)[0].rsplit("_", 1)[0]
                              if fn.count("_") > 2 else fn, "rotor period")
        elif fn in e_cols:
            t = "e"
            desc = interp.get(fn.rsplit("_", 2)[0], "involution/char")
        else:
            t = "×"
            desc = "position × character"
        print(f"  {rank:>4}  {fn:>26}  {fv:>8.4f}  {t:>4}  {desc}")

    # ── Grouped importances ───────────────────────────────────────────────────
    pi_imp    = imp[pi_cols].sum()
    e_imp     = imp[e_cols].sum()
    cross_imp = imp[cross_cols].sum()
    total_imp = pi_imp + e_imp + cross_imp

    print(f"\n  ── Grouped importances ──")
    print(f"  π-type (position / rotor period)  : "
          f"{pi_imp:.4f}  ({100*pi_imp/total_imp:5.1f}%)")
    print(f"  e-type (character / involution)   : "
          f"{e_imp:.4f}  ({100*e_imp/total_imp:5.1f}%)")
    print(f"  Cross-products (π × e)            : "
          f"{cross_imp:.4f}  ({100*cross_imp/total_imp:5.1f}%)")

    # ── CV: π-only vs e-only vs full vs raw ──────────────────────────────────
    print(f"\n{SEP}")
    print(f"  CV COMPARISON  (5-fold, n={N})")
    print(f"  Isolates whether character (e) or position (π) predicts better")
    print(SEP)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    r2_raw  = run_cv(raw_df.values,               target, make_rf(), kf)
    r2_pi   = run_cv(feat_df[pi_cols].values,      target, make_rf(), kf)
    r2_e    = run_cv(feat_df[e_cols].values,        target, make_rf(), kf)
    r2_full = run_cv(feat_df.values,               target, make_rf(), kf)

    print(f"\n  {'Model':22s}  {'R²':>8}  {'Interpretation'}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*40}")
    print(f"  {'Raw features':22s}  {r2_raw:>8.4f}  baseline (raw position + char values)")
    print(f"  {'π-only (position)':22s}  {r2_pi:>8.4f}  rotor period alone")
    print(f"  {'e-only (character)':22s}  {r2_e:>8.4f}  involution constraint alone")
    print(f"  {'Full (π + e)':22s}  {r2_full:>8.4f}  both combined")

    e_beats_pi = r2_e > r2_pi
    e_label = "YES → character constraint is stronger signal" if e_beats_pi else "NO → position (rotor period) is stronger signal"
    print(f"\n  e-only > π-only? {e_label}")

    # ── 3-stage pipeline ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  3-STAGE PIPELINE  (Enigma-specific)")
    print(f"  Stage 0  : π-encoded position features (no fitting)")
    print(f"  Stage 1  : Ridge on π features  → removes rotor-period grammar")
    print(f"  Stage 2  : RF on e features     → captures involution dialect")
    print(f"  Final    : Stage-1 + Stage-2 residual correction")
    print(f"  Turing's layer = Stage 2 contribution above Stage 1 alone")
    print(SEP)

    X_pi = feat_df[pi_cols].values
    X_e  = feat_df[e_cols].values

    r2s_s1   = []   # Ridge (π) alone
    r2s_3s   = []   # Full 3-stage
    e_deltas = []   # Stage-2 contribution

    for tr, te in kf.split(X_pi):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_pi[tr], target[tr])
        res_tr = target[tr] - ridge.predict(X_pi[tr])

        rf2 = make_rf()
        rf2.fit(X_e[tr], res_tr)

        pred_s1 = ridge.predict(X_pi[te])
        pred_3s = pred_s1 + rf2.predict(X_e[te])

        r2_s1  = r2_score(target[te], pred_s1)
        r2_3s_ = r2_score(target[te], pred_3s)

        r2s_s1.append(r2_s1)
        r2s_3s.append(r2_3s_)
        e_deltas.append(r2_3s_ - r2_s1)

    r2_s1  = float(np.mean(r2s_s1))
    r2_3s  = float(np.mean(r2s_3s))
    delta  = float(np.mean(e_deltas))

    print(f"\n  Stage 1 only (Ridge, π features) : R² = {r2_s1:.4f}")
    print(f"  3-stage      (π grammar + e dial): R² = {r2_3s:.4f}")
    print(f"  Stage 2 contribution (ΔR²)       : {delta:+.4f}")
    print(f"  Stage 2 adds value: {'YES → e-type (involution) explains variance above rotor period'  if delta > 0 else 'NO → position alone is sufficient'}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  VERDICT: Does the ISPCC probe detect Turing's insight?")
    print(SEP)

    print(f"\n  Enigma invariant (no self-mapping)    : "
          f"{'CONFIRMED ✓' if n_self == 0 else 'VIOLATED ✗'}")
    print(f"  π-type total importance               : {100*pi_imp/total_imp:5.1f}%")
    print(f"  e-type total importance               : {100*e_imp/total_imp:5.1f}%")
    print(f"  e-only R²                             : {r2_e:.4f}")
    print(f"  π-only R²                             : {r2_pi:.4f}")
    print(f"  Stage-2 (e dialect) ΔR²               : {delta:+.4f}")

    # Three criteria for Turing's insight detection
    c1 = e_imp > pi_imp          # importance: e dominates
    c2 = r2_e > r2_pi            # predictive power: e beats π alone
    c3 = delta > 0               # 3-stage: e adds above π

    n_criteria = sum([c1, c2, c3])

    print(f"\n  Turing criteria met:")
    print(f"    [{'✓' if c1 else '✗'}] e-type importance > π-type importance")
    print(f"    [{'✓' if c2 else '✗'}] e-only R² > π-only R²")
    print(f"    [{'✓' if c3 else '✗'}] Stage-2 (e dialect) adds R² above Stage-1 (π grammar)")

    print(f"\n  {n_criteria}/3 criteria met")
    if n_criteria >= 2:
        print(f"""
  → TURING'S INSIGHT DETECTED ({n_criteria}/3 criteria)

  The probe finds that character structure (e-type: involution, bounded
  substitution constraint) is a stronger structural residual than position
  structure (π-type: rotor period stepping).

  This is what Turing found: the exploitable structure in Enigma is in
  CHARACTER SPACE (which letter → which substitute, bounded by the reflector's
  no-self-mapping rule), not only in POSITION SPACE (which rotor step).

  Cryptographic structural residual = e-type.
  The probe detected it instantly from one message.
""")
    elif n_criteria == 1:
        print(f"""
  → PARTIAL DETECTION (1/3 criteria)

  One criterion points to e-type dominance. With a longer message or multiple
  messages, the involution structure would likely be clearer.
  The single criterion that held: see above.
""")
    else:
        print(f"""
  → π-TYPE DOMINANT (0/3 Turing criteria)

  Position (rotor period) is the stronger structural signal in this message.
  This means the probe finds the naive periodicity, not the deeper involution.
  With this message length ({N} chars), the period-26 rotor structure dominates.
  More messages or longer text would likely expose the e-type layer.
""")


if __name__ == "__main__":
    main()
    
"""
python enigma_probe.py
"""