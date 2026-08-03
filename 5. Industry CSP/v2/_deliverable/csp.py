"""
csp.py  —  Cascade State Predictor  CLI
═══════════════════════════════════════════════════════════════════════════════
ISPCC Framework  —  Intentional Symbolic Pre-Cognitive Computing
Version 2.0

The ISPCC framework establishes all discoverable structure before the learning
algorithm engages.  Variables are classified by their dynamical character:

  π-type  (cascade, non-repeating, accumulating):
          encoded with sin(π·xn), sin(π²·xn) — basis functions that capture
          monotone accumulation toward a failure state.

  e-type  (equilibrium, bounded, self-regulating):
          encoded with exp(−e·xn), xn^e — basis functions that capture
          mean-reverting bounded behaviour.

  The bifurcation constant is the physical boundary between regimes.
  It is derived from domain physics, not from data statistics.

Pipeline (3SIMM — 3-Stage Intentional MaxiMin):
  Stage 0  Feature extraction from raw sensor snapshots
  Stage 1  Pi/e symbolic encoding (weights fixed across all domains)
  Stage 2  Probe — RF importance confirms structural variable typing
  Stage 3  Canonical reduction via probe importance rankings
  Stage 4  IntentionalMaxiMin — selects structurally representative snapshots
  Stage 5  Ridge grammar (structural power law) + RF dialect (residuals)

Validated domains:
  Logistic map        — recovered bifurcation constant R_BIFURCATION
  Harmonic intervals  — recovered canonical variables p and q
  EM surrogate        — beat PINN ceiling at 1/5 the simulation budget
  Industrial bearings — cascade detected 17-68 min before failure, 0 false alarms

Commands:
  csp train   --data PATH [--condition 1|2|3] [--n INT] [--output FILE]
  csp deploy  --model FILE --bearing PATH [--every INT]
  csp probe   --data PATH [--bearing NAME]
  csp status  --model FILE
  csp help
  csp         (no arguments → interactive menu)

Install:
  Run install.bat once to register 'csp' as a command available system-wide.
  After that, type 'csp' from any directory in Command Prompt.

Academic use:
  This file is the complete entry point for the CSP system.
  csp_v2_train.py  — offline surrogate training
  csp_v2_deploy.py — live deployment inference
  pronostia_3simm.py — LOBO validation and probe (v1)
  All four files are required in the same directory.

Reference dataset:
  FEMTO/PRONOSTIA (PHM IEEE 2012 Challenge)
  github.com/wkzs111/phm-ieee-2012-data-challenge-dataset
"""

import sys
import os
import argparse
import pickle
import glob
import textwrap
import time

# ── Path setup: sibling modules must be importable ────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

__version__ = '2.0'
LOG_DIR = 'csp_logs'   # all logs written here relative to working directory


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING — tee stdout to file simultaneously
# Every command produces a timestamped plain-text log.
# Logs are an audit trail: what was run, with what parameters, and what resulted.
# For academic use: attach logs to supplementary materials.
# ══════════════════════════════════════════════════════════════════════════════

class TeeLogger:
    """
    Redirects stdout to both terminal and a log file simultaneously.
    All print() calls — including those inside imported modules — are captured.
    """
    def __init__(self, log_path):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.log      = open(log_path, 'w', encoding='utf-8')
        self.log_path = log_path

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.flush()
        self.log.close()
        sys.stdout = self.terminal

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *args):
        self.close()


def make_log_path(command, tag=''):
    """Generate timestamped log path inside LOG_DIR."""
    ts  = time.strftime('%Y%m%d_%H%M%S')
    tag = ('_' + tag.replace('/', '_').replace('\\', '_').replace(' ', '_')
           if tag else '')
    name = f'csp_{command}{tag}_{ts}.log'
    return os.path.join(LOG_DIR, name)


def log_header(logger, command, params):
    """Write a structured header to the log file."""
    import platform
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    sep72  = '=' * 72
    sep72d = '-' * 72
    hdr = (
        sep72  + '\n'
        '  CSP — Cascade State Predictor  v' + __version__ + '\n'
        '  Command   : ' + command + '\n'
        '  Timestamp : ' + ts     + '\n'
        '  Platform  : ' + platform.system() + ' ' + platform.release() +
        ', Python ' + platform.python_version() + '\n' +
        sep72d + '\n'
        '  Parameters:\n'
    )
    for k, v in params.items():
        hdr += f'    {k:<16}: {v}\n'
    hdr += sep72 + '\n'
    logger.log.write(hdr)
    logger.log.flush()


def log_footer(logger):
    """Write log location footer to terminal (not log — it would be circular)."""
    logger.terminal.write(
        f'\n  Log saved → {os.path.abspath(logger.log_path)}\n')

# ══════════════════════════════════════════════════════════════════════════════
# DISPLAY CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

W = 72   # terminal width

BANNER = f"""
{'═' * W}
  CSP — Cascade State Predictor                            v{__version__}
  ISPCC Framework — Intentional Symbolic Pre-Cognitive Computing
{'─' * W}
  π  cascade variables   →  non-repeating, accumulating, sin(π·xn) basis
  e  equilibrium vars    →  bounded, self-regulating, exp(−e·xn) basis
  Β  bifurcation point   →  structural boundary between regimes (physics-derived)
{'═' * W}
"""

MENU = f"""
  Commands:

    [1]  train    Train CSP surrogate on bearing fleet data
    [2]  deploy   Run live cascade detection on a bearing
    [3]  probe    Structural typing probe on a dataset
    [4]  status   Show model artifact information
    [5]  help     Command reference
    [0]  exit
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sep(char='─'):
    print(char * W)


def header(title):
    print(f'\n{"═" * W}')
    print(f'  {title}')
    print(f'{"═" * W}')


def ask(prompt, default=None):
    """Prompt user for input with optional default."""
    if default is not None:
        display = f'  {prompt} [{default}]: '
    else:
        display = f'  {prompt}: '
    val = input(display).strip()
    if not val and default is not None:
        return str(default)
    return val


def ask_path(prompt, must_exist=True):
    """Ask for a path, validate existence."""
    while True:
        val = ask(prompt)
        if not val:
            print('  [error] path cannot be empty.')
            continue
        val = val.strip('"').strip("'")
        if must_exist and not os.path.exists(val):
            print(f'  [error] not found: {val}')
            continue
        return val


def confirm(prompt):
    """Yes/no confirmation."""
    ans = ask(f'{prompt} [y/n]', default='y').lower()
    return ans in ('y', 'yes')


def check_siblings():
    """Verify required sibling modules exist."""
    here   = os.path.dirname(os.path.abspath(__file__))
    needed = ['pronostia_3simm.py', 'csp_v2_train.py', 'csp_v2_deploy.py']
    missing = [f for f in needed if not os.path.exists(os.path.join(here, f))]
    if missing:
        print(f'\n  [error] missing required files in {here}:')
        for f in missing:
            print(f'    {f}')
        print('\n  All four .py files must be in the same directory.')
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(data=None, condition=None, n_intentional=2400,
              output='csp_model.pkl', min_per=6,
              early=0.50, cascade=0.25, critical=0.08):
    """
    Train CSP surrogate on all bearings in Full_Test_Set.
    Runs once offline.  Produces a model artifact (.pkl) for deployment.
    """
    header('TRAIN — CSP Surrogate')

    if not check_siblings():
        return

    from csp_v2_train import (load_all_bearings, filter_by_condition,
                               train_full, derive_thresholds, save_model)

    # ── Interactive prompts if not supplied ───────────────────────────────────
    if data is None:
        print('\n  Provide the path to the Full_Test_Set folder.')
        print('  Example: C:/path/to/phm-ieee.../Full_Test_Set\n')
        data = ask_path('Full_Test_Set path')

    if condition is None:
        print('\n  Operating condition filter:')
        print('    1  =  1800 RPM / 4000N')
        print('    2  =  1650 RPM / 4200N')
        print('    3  =  1500 RPM / 5000N')
        print('    all  =  use all conditions (cross-condition)')
        cond_in = ask('Condition [all]', default='all').strip()
        condition = None if cond_in.lower() in ('all', '') else cond_in

    n_str = ask('MaxiMin budget (snapshots selected for training)', default=n_intentional)
    n_intentional = int(n_str)

    output = ask('Output model filename', default=output)
    if not output.endswith('.pkl'):
        output += '.pkl'

    # ── Run with logging ──────────────────────────────────────────────────────
    log_path = make_log_path('train', os.path.basename(data or 'unknown'))
    params   = {
        'data':       data,
        'condition':  condition or 'all',
        'n_intentional': n_intentional,
        'output':     output,
        'early':      early,
        'cascade':    cascade,
        'critical':   critical,
    }
    print()
    sep()
    t0 = time.time()

    with TeeLogger(log_path) as logger:
        log_header(logger, 'train', params)

        bearings = load_all_bearings(data)
        print(f'\n  {len(bearings)} bearings loaded.')

        if condition:
            print(f'\n  Filtering to condition {condition}...')
            bearings = filter_by_condition(bearings, condition)
            print(f'  {len(bearings)} bearings after filter.')

        ridge, rf, meta = train_full(
            bearings, n_intentional=n_intentional, min_per=min_per)

        thresholds = derive_thresholds(early=early, cascade=cascade,
                                       critical=critical)

        meta['condition'] = condition
        save_model(ridge, rf, thresholds, meta, output_path=output)

        print(f'\n  [done]  {time.time() - t0:.1f}s')
        print(f'  Model → {os.path.abspath(output)}')
        print(f'\n  Next: csp deploy --model {output} --bearing <path>')

    log_footer(logger)
    sep()


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: DEPLOY
# ══════════════════════════════════════════════════════════════════════════════

def cmd_deploy(model=None, bearing=None, every=50):
    """
    Run live cascade detection on a bearing folder.
    Reads snapshots sequentially, outputs regime + cascade alert.
    """
    header('DEPLOY — Live Cascade Detection')

    if not check_siblings():
        return

    from csp_v2_deploy import run_deployment

    if model is None:
        print('\n  Provide the trained model file (.pkl from csp train).\n')
        model = ask_path('Model file (.pkl)')

    if bearing is None:
        print('\n  Provide the bearing folder to run detection on.')
        print('  Example: C:/path/to/Full_Test_Set/Bearing1_3\n')
        bearing = ask_path('Bearing folder')

    every_str = ask('Print every N snapshots', default=every)
    every = int(every_str)

    bearing_tag = os.path.basename(bearing.rstrip('/\\'))
    log_path    = make_log_path('deploy', bearing_tag)
    params      = {'model': model, 'bearing': bearing, 'every': every}

    print()
    sep()
    with TeeLogger(log_path) as logger:
        log_header(logger, 'deploy', params)
        run_deployment(bearing_dir=bearing, model_path=model, every=every)

    log_footer(logger)
    sep()


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: PROBE
# ══════════════════════════════════════════════════════════════════════════════

def cmd_probe(data=None, bearing_name='Bearing1_3'):
    """
    Run the ISPCC structural typing probe on one bearing.

    The probe trains a Random Forest on all encoded features and reports
    grouped importance by variable type (π / e / cross-product).
    A result of >80% π-dominant confirms the encoding is structurally correct
    for this domain — the same confirmation obtained on the logistic map,
    harmonics, and EM surrogate domains.
    """
    header('PROBE — Structural Typing')

    if not check_siblings():
        return

    from pronostia_3simm import (load_all_bearings, build_encoded_features,
                                  run_probe, probe_report)

    if data is None:
        print('\n  Provide the Full_Test_Set folder path.\n')
        data = ask_path('Full_Test_Set path')

    available = sorted([
        os.path.basename(d)
        for d in glob.glob(os.path.join(data, 'Bearing*'))
        if os.path.isdir(d)
    ])

    if not available:
        print(f'\n  [error] no Bearing* folders found in {data}')
        return

    print(f'\n  Available bearings: {available}')
    bearing_name = ask('Bearing to probe', default=bearing_name)

    if bearing_name not in available:
        print(f'\n  [error] {bearing_name} not found.')
        return

    print(f'\n  Loading {bearing_name}...')
    bearings = load_all_bearings(data)

    if bearing_name not in bearings:
        print(f'\n  [error] failed to load {bearing_name}.')
        return

    df = bearings[bearing_name]
    enc, pi_cols, e_cols, cross_cols = build_encoded_features(df)

    log_path = make_log_path('probe', bearing_name)
    csv_path = log_path.replace('.log', '_importance.csv')
    params   = {'data': data, 'bearing': bearing_name,
                'n_snapshots': len(df), 'n_features': len(enc.columns)}

    print(f'\n  Running probe on {len(df)} snapshots, {len(enc.columns)} encoded features...')
    sep()

    with TeeLogger(log_path) as logger:
        log_header(logger, 'probe', params)

        imp = run_probe(enc.values, df['rul_s'].values, list(enc.columns))
        pi_pct, e_pct = probe_report(imp, pi_cols, e_cols, cross_cols)

        print(f'\n  Interpretation:')
        if pi_pct > 0.70:
            print(f'  π-dominant at {pi_pct*100:.1f}% → encoding structurally confirmed.')
            print(f'  Cascade character is the primary RUL signal in this domain.')
        elif e_pct > 0.50:
            print(f'  e-dominant at {e_pct*100:.1f}% → equilibrium character dominates.')
            print(f'  Review variable classification before training.')
        else:
            print(f'  Mixed result — review variable encoding.')

        print(f'\n  Bifurcation analogy:')
        print(f'    RMS_BIFURCATION  ↔  R_BIFURCATION  (logistic map)')
        print(f'    TEMP_BIFURCATION ↔  DIEL_BIFURCATION (EM surrogate)')

        # Save importance table as CSV (machine-readable for analysis)
        imp_df = imp.reset_index()
        imp_df.columns = ['feature', 'importance']
        imp_df['type'] = imp_df['feature'].apply(
            lambda f: 'pi' if f in pi_cols else
                      ('e'  if f in e_cols  else 'cross'))
        imp_df['importance_pct'] = imp_df['importance'] * 100
        imp_df.to_csv(csv_path, index=False, float_format='%.6f')
        print(f'\n  Importance table → {os.path.abspath(csv_path)}')

    log_footer(logger)
    sep()


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: STATUS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_status(model=None):
    """
    Show information about a trained model artifact.
    """
    header('STATUS — Model Artifact')

    if model is None:
        # Try to find a .pkl in the current directory
        pkls = glob.glob('*.pkl')
        if pkls:
            model = ask('Model file', default=pkls[0])
        else:
            model = ask_path('Model file (.pkl)')

    if not os.path.exists(model):
        print(f'\n  [error] not found: {model}')
        return

    try:
        with open(model, 'rb') as f:
            artifact = pickle.load(f)
    except Exception as e:
        print(f'\n  [error] could not load model: {e}')
        return

    meta   = artifact.get('meta', {})
    thresh = artifact.get('thresholds', {})
    consts = artifact.get('constants', {})

    size_mb = os.path.getsize(model) / (1024 * 1024)

    print(f'\n  File      : {os.path.abspath(model)}  ({size_mb:.0f} MB)')
    print(f'  Framework : ISPCC 3SIMM  v{__version__}')
    print()
    print(f'  Training:')
    print(f'    Bearings     : {meta.get("bearing_names", "—")}')
    print(f'    Condition    : {meta.get("condition", "all")}')
    print(f'    Pool size    : {meta.get("n_pool", "—"):,} snapshots')
    print(f'    Selected     : {meta.get("n_selected", "—")} intentional snapshots')
    print(f'    In-sample R² : {meta.get("r2_insample", 0):.4f}')
    print()
    print(f'  Thresholds (rul_norm space):')
    print(f'    HEALTHY  : rul_norm ≥ {thresh.get("early", 0.50):.2f}')
    print(f'    EARLY    : {thresh.get("cascade", 0.25):.2f} ≤ rul_norm < '
          f'{thresh.get("early", 0.50):.2f}')
    print(f'    CASCADE  : {thresh.get("critical", 0.08):.2f} ≤ rul_norm < '
          f'{thresh.get("cascade", 0.25):.2f}')
    print(f'    CRITICAL : rul_norm < {thresh.get("critical", 0.08):.2f}')
    print()
    print(f'  Physical constants:')
    print(f'    FAILURE_G        : {consts.get("FAILURE_G", "—")} g')
    print(f'    RMS_BIFURCATION  : {consts.get("RMS_BIFURCATION", "—")} g')
    print(f'    TEMP_BIFURCATION : {consts.get("TEMP_BIFURCATION", "—")} °C above ambient')
    print(f'    KURT_SCALE       : {consts.get("KURT_SCALE", "—")}')

    log_path = make_log_path('status', os.path.basename(model))
    with TeeLogger(log_path) as logger:
        log_header(logger, 'status', {'model': model})
        # Status already printed above — write a reference only
        logger.log.write(f'  [see terminal output above — status is non-destructive]\n')

    log_footer(logger)
    sep()


# ══════════════════════════════════════════════════════════════════════════════
# COMMAND: HELP
# ══════════════════════════════════════════════════════════════════════════════

def cmd_help():
    """Print command reference."""
    print(textwrap.dedent(f"""
    {'═' * W}
      CSP COMMAND REFERENCE
    {'─' * W}

      TRAIN — build the surrogate from historical bearing data (run once)
        csp train
        csp train --data PATH --condition 1 --n 2400 --output model.pkl

        --data        Full_Test_Set folder path
        --condition   1 | 2 | 3  (filter by operating condition)
                      omit to train on all conditions
        --n           MaxiMin budget (default 2400)
        --output      output .pkl filename (default csp_model.pkl)
        --early       EARLY threshold in rul_norm (default 0.50)
        --cascade     CASCADE threshold in rul_norm (default 0.25)
        --critical    CRITICAL threshold in rul_norm (default 0.08)

      DEPLOY — run live cascade detection on a bearing folder
        csp deploy
        csp deploy --model model.pkl --bearing PATH --every 50

        --model       trained model .pkl file
        --bearing     bearing folder (contains acc_*.csv, temp_*.csv)
        --every       print every N snapshots (default 50 ≈ 8 min)

      PROBE — structural typing diagnostic (confirms encoding is correct)
        csp probe
        csp probe --data PATH --bearing Bearing1_3

        --data        Full_Test_Set folder path
        --bearing     which bearing to run probe on

      STATUS — show model artifact information
        csp status
        csp status --model model.pkl

      HELP
        csp help

    {'─' * W}
      ISPCC FRAMEWORK — ISPCC establishes all discoverable structure
      before the learning algorithm engages.  The model inherits physics
      as its coordinate system and learns only what physics cannot explain.
    {'─' * W}
      π encoding  :  sin(π·xn), sin(π²·xn)  weights (5,1,1,3,1)
      e encoding  :  exp(−e·xn), xn^e       weights (2,2,1)
      Normalising constants: derived from domain physics, not data.
    {'═' * W}
    """))


# ══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE MENU
# ══════════════════════════════════════════════════════════════════════════════

def interactive_menu():
    """Run the interactive command menu."""
    print(BANNER)

    dispatch = {
        '1': ('train',  cmd_train),
        '2': ('deploy', cmd_deploy),
        '3': ('probe',  cmd_probe),
        '4': ('status', cmd_status),
        '5': ('help',   cmd_help),
    }

    while True:
        print(MENU)
        choice = input('  CSP > ').strip().lower()

        if choice in ('0', 'exit', 'quit', 'q'):
            print('\n  Exiting CSP.\n')
            break
        elif choice in dispatch:
            _, fn = dispatch[choice]
            try:
                fn()
            except KeyboardInterrupt:
                print('\n  [cancelled]\n')
            except Exception as e:
                print(f'\n  [error] {e}\n')
        elif choice in ('train', 'deploy', 'probe', 'status', 'help'):
            # Accept command names directly
            for k, (name, fn) in dispatch.items():
                if name == choice:
                    try:
                        fn()
                    except KeyboardInterrupt:
                        print('\n  [cancelled]\n')
                    except Exception as e:
                        print(f'\n  [error] {e}\n')
                    break
        elif choice == 'help':
            cmd_help()
        elif choice == '':
            continue
        else:
            print(f'\n  Unknown command: {choice!r}   (type 0 to exit)\n')


# ══════════════════════════════════════════════════════════════════════════════
# DIRECT COMMAND MODE (argparse)
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        prog='csp',
        description='Cascade State Predictor — ISPCC Framework v2.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Run without arguments for interactive menu.\n'
               'Run "csp help" for full command reference.')

    sub = parser.add_subparsers(dest='command')

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser('train', help='Train CSP surrogate')
    p_train.add_argument('--data',       required=False,
                         help='Full_Test_Set folder path')
    p_train.add_argument('--condition',  choices=['1', '2', '3'], default=None,
                         help='Operating condition filter')
    p_train.add_argument('--n',          type=int, default=2400,
                         dest='n_intentional',
                         help='MaxiMin budget (default 2400)')
    p_train.add_argument('--output',     default='csp_model.pkl',
                         help='Output model filename')
    p_train.add_argument('--min_per',    type=int, default=6,
                         help='Min snapshots per bearing (default 6)')
    p_train.add_argument('--early',      type=float, default=0.50)
    p_train.add_argument('--cascade',    type=float, default=0.25)
    p_train.add_argument('--critical',   type=float, default=0.08)

    # ── deploy ────────────────────────────────────────────────────────────────
    p_dep = sub.add_parser('deploy', help='Live cascade detection')
    p_dep.add_argument('--model',   required=False,
                       help='Trained model .pkl file')
    p_dep.add_argument('--bearing', required=False,
                       help='Bearing folder path')
    p_dep.add_argument('--every',   type=int, default=50,
                       help='Print every N snapshots (default 50)')

    # ── probe ─────────────────────────────────────────────────────────────────
    p_probe = sub.add_parser('probe', help='Structural typing probe')
    p_probe.add_argument('--data',    required=False,
                         help='Full_Test_Set folder path')
    p_probe.add_argument('--bearing', default='Bearing1_3',
                         help='Bearing to probe (default Bearing1_3)')

    # ── status ────────────────────────────────────────────────────────────────
    p_status = sub.add_parser('status', help='Model artifact info')
    p_status.add_argument('--model', required=False,
                          help='Model .pkl file')

    # ── help ──────────────────────────────────────────────────────────────────
    sub.add_parser('help', help='Command reference')

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) == 1:
        # No arguments → interactive menu
        interactive_menu()
        return

    parser = build_parser()
    args   = parser.parse_args()

    if args.command == 'train':
        cmd_train(data=args.data,
                  condition=args.condition,
                  n_intentional=args.n_intentional,
                  output=args.output,
                  min_per=args.min_per,
                  early=args.early,
                  cascade=args.cascade,
                  critical=args.critical)

    elif args.command == 'deploy':
        cmd_deploy(model=args.model,
                   bearing=args.bearing,
                   every=args.every)

    elif args.command == 'probe':
        cmd_probe(data=args.data,
                  bearing_name=args.bearing)

    elif args.command == 'status':
        cmd_status(model=args.model)

    elif args.command == 'help':
        print(BANNER)
        cmd_help()

    else:
        print(BANNER)
        parser.print_help()


if __name__ == '__main__':
    main()
