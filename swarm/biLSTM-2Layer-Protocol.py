import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
import json
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization,
    Conv1D, MaxPooling1D, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true     = tf.cast(y_true, tf.float32)
        bce        = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t        = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_term = tf.pow((1 - p_t), gamma)
        return alpha * focal_term * bce
    return loss


ABSOLUTE_TIME_COLS = [
    'timestamp', 'time_boot_ms', 'time_usec',
    'udp_time_relative', 'tcp_time_relative'
]

DROP_COLS = [
    'mav_packet_type', 'sim_uuid',
    'ip_src', 'ip_addr', 'ip_src_host', 'ip_host', 'ip_dst', 'ip_dst_host',
    'ip_checksum', 'ip_checksum_status',
    'udp_checksum', 'udp_checksum_status',
    'tcp_checksum', 'tcp_checksum_status',
    'tcp_flags_str', 'udp_text', 'tcp_text',
    'tcp_options', 'tcp_options_nop', 'tcp_options_timestamp',
    'udp_payload', 'ip_payload'
]

ATTACK_FEATURE_MAP = {
    'ATTITUDE':    ['roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed'],
    'GPS_RAW_INT': ['lat', 'lon', 'alt', 'alt_ellipsoid', 'cog',
                    'vel', 'eph', 'epv', 'satellites_visible'],
    'VFR_HUD':     ['airspeed', 'groundspeed', 'heading', 'throttle', 'alt', 'climb']
}

# ── Per-CSV frozen detection config ───────────────────────────────────────────
FROZEN_CONFIG = {
    'ATTITUDE': {
        'cols':      ['roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed'],
        'threshold': 1e-6
    },
    'GPS_RAW_INT': {
        'cols':      ['lat', 'lon', 'alt', 'vel'],
        'threshold': 1e-6
    },
    'VFR_HUD': {
        'cols':      ['airspeed', 'groundspeed', 'alt', 'climb'],
        'threshold': 1e-6
    }
}


def get_feature_cols(csv_name):
    key = csv_name.replace('.csv', '').upper()
    for k, cols in ATTACK_FEATURE_MAP.items():
        if k in key:
            return cols
    return None


def get_frozen_config(csv_name):
    key = csv_name.replace('.csv', '').upper()
    for k, cfg in FROZEN_CONFIG.items():
        if k in key:
            return cfg
    return None


# ── Stage 1: Rule-based flat-line detector ────────────────────────────────────
def is_flat_line(df, frozen_cols, threshold=1e-6):
    """
    Returns True if any core column has near-zero variance.
    A real drone NEVER has frozen sensor values.
    """
    for col in frozen_cols:
        if col in df.columns:
            if df[col].var() < threshold:
                return True
    return False


def classify_sim(sim_path, primary_csv):
    """Classify a simulation as flat-line or subtle attack."""
    file_path = os.path.join(sim_path, primary_csv)
    if not os.path.exists(file_path):
        return 'missing'
    df  = pd.read_csv(file_path, low_memory=False)
    cfg = get_frozen_config(primary_csv)
    if cfg is None:
        return 'subtle'
    return 'flat' if is_flat_line(df, cfg['cols'], cfg['threshold']) else 'subtle'


# ── Feature engineering ───────────────────────────────────────────────────────
def rolling_autocorr(series, window, lag=1):
    s1  = series
    s2  = series.shift(lag)
    w   = window
    mp  = w // 2
    num = (
        (s1 - s1.rolling(w, min_periods=mp).mean()) *
        (s2 - s2.rolling(w, min_periods=mp).mean())
    ).rolling(w, min_periods=mp).mean()
    std1 = s1.rolling(w, min_periods=mp).std().replace(0, 1e-9)
    std2 = s2.rolling(w, min_periods=mp).std().replace(0, 1e-9)
    return (num / (std1 * std2)).fillna(0)


def engineer_attitude_features(df):
    new_cols = {}
    for col in ['roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed']:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f'{col}_d1'] = d1
        new_cols[f'{col}_d2'] = d1.diff().fillna(0)
        for w in [20, 50]:
            roll = df[col].rolling(w, min_periods=w // 2)
            new_cols[f'{col}_autocorr_{w}'] = rolling_autocorr(df[col], w)
            new_cols[f'{col}_kurt_{w}']     = roll.kurt().fillna(0)
            new_cols[f'{col}_skew_{w}']     = roll.skew().fillna(0)
            new_cols[f'{col}_std_{w}']      = roll.std().fillna(0)
            new_cols[f'{col}_range_{w}']    = (roll.max() - roll.min()).fillna(0)

    if all(c in df.columns for c in ['rollspeed', 'pitchspeed', 'yawspeed']):
        rate_mag = np.sqrt(
            df['rollspeed']**2 + df['pitchspeed']**2 + df['yawspeed']**2)
        new_cols['rate_mag']        = rate_mag
        new_cols['rate_mag_std_20'] = rate_mag.rolling(20, min_periods=10).std().fillna(0)

    if all(c in df.columns for c in ['roll', 'pitch', 'yaw']):
        new_cols['attitude_mag'] = np.sqrt(
            df['roll']**2 + df['pitch']**2 + df['yaw']**2)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def engineer_gps_features(df):
    new_cols = {}
    for col in ['lat', 'lon', 'alt', 'alt_ellipsoid', 'cog', 'vel', 'eph', 'epv']:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f'{col}_d1'] = d1
        new_cols[f'{col}_d2'] = d1.diff().fillna(0)
        for w in [20, 50]:
            roll = df[col].rolling(w, min_periods=w // 2)
            new_cols[f'{col}_autocorr_{w}'] = rolling_autocorr(df[col], w)
            new_cols[f'{col}_kurt_{w}']     = roll.kurt().fillna(0)
            new_cols[f'{col}_std_{w}']      = roll.std().fillna(0)
            new_cols[f'{col}_range_{w}']    = (roll.max() - roll.min()).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def engineer_vfr_features(df):
    new_cols = {}
    for col in ['airspeed', 'groundspeed', 'heading', 'throttle', 'alt', 'climb']:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f'{col}_d1'] = d1
        new_cols[f'{col}_d2'] = d1.diff().fillna(0)
        for w in [20, 50]:
            roll = df[col].rolling(w, min_periods=w // 2)
            new_cols[f'{col}_autocorr_{w}'] = rolling_autocorr(df[col], w)
            new_cols[f'{col}_kurt_{w}']     = roll.kurt().fillna(0)
            new_cols[f'{col}_std_{w}']      = roll.std().fillna(0)
            new_cols[f'{col}_range_{w}']    = (roll.max() - roll.min()).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def engineer_features(df, core_cols, csv_name):
    key = csv_name.replace('.csv', '').upper()

    if 'ATTITUDE' in key:
        attitude_cols = ['roll', 'pitch', 'yaw', 'rollspeed', 'pitchspeed', 'yawspeed']
        if all(c in df.columns for c in attitude_cols):
            df = engineer_attitude_features(df)
    elif 'GPS_RAW_INT' in key:
        df = engineer_gps_features(df)
    elif 'VFR_HUD' in key:
        df = engineer_vfr_features(df)

    new_cols = {}
    for col in core_cols:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f'{col}_d1'] = d1
        new_cols[f'{col}_d2'] = d1.diff().fillna(0)
        for w in [5, 10, 20]:
            roll = df[col].rolling(w, min_periods=1)
            new_cols[f'{col}_std_{w}']   = roll.std().fillna(0)
            new_cols[f'{col}_range_{w}'] = (roll.max() - roll.min()).fillna(0)
            new_cols[f'{col}_mean_{w}']  = roll.mean().fillna(0)
        local_std  = df[col].rolling(20, min_periods=1).std().replace(0, 1e-9)
        local_mean = df[col].rolling(20, min_periods=1).mean()
        new_cols[f'{col}_zscore'] = ((df[col] - local_mean) / local_std).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    extra = ['rate_mag', 'rate_mag_std_20', 'attitude_mag']
    keep  = [c for c in df.columns if any(
        c == col or c.startswith(col + '_') for col in core_cols
    ) or c in extra]

    return df[keep].fillna(0)


def preprocess_df(df, core_cols, csv_name):
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    for col in ABSOLUTE_TIME_COLS:
        if col in df.columns:
            df[col] = df[col].diff().fillna(0)
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(0)
    df = engineer_features(df, core_cols, csv_name)
    return df


def normalize_simulation(features):
    scaler   = StandardScaler()
    reshaped = features.reshape(-1, features.shape[-1])
    scaled   = scaler.fit_transform(reshaped)
    return scaled.reshape(features.shape)


def load_window_normalize_sim(sim_path, primary_csv, label,
                               window_size, stride, core_cols):
    file_path = os.path.join(sim_path, primary_csv)
    if not os.path.exists(file_path):
        return None
    df = pd.read_csv(file_path, low_memory=False)
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    df       = preprocess_df(df, core_cols, primary_csv)
    features = df.values
    if len(features) <= window_size:
        return None
    features = normalize_simulation(features)
    windows  = [
        features[i:i + window_size]
        for i in range(0, len(features) - window_size + 1, stride)
    ]
    if not windows:
        return None
    print(f"  Loaded {len(windows):>5} windows ← {sim_path}")
    return (windows, label)


def _configure_gpus(mixed_precision: bool) -> None:
    """Detect GPUs, enable memory growth, optionally enable mixed precision.

    Loud-prints what TF will train on so a silent CPU fallback can't waste
    hours of wall time. Raises nothing — if GPUs aren't detected the script
    still runs (just slowly).
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("⚠  No GPU detected — training will run on CPU (slow).")
        print("   If this is a GPU box, check: nvidia-smi, CUDA install, "
              "and that TensorFlow was built with GPU support.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            # Already initialised — non-fatal.
            print(f"   note: could not enable memory growth on {gpu.name}: {exc}")

    logical = tf.config.list_logical_devices("GPU")
    print(f"✓ TensorFlow sees {len(gpus)} physical GPU(s), {len(logical)} logical:")
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            name = details.get("device_name", "unknown")
            cc = details.get("compute_capability", "?")
            print(f"   - {gpu.name}  ({name}, compute {cc})")
        except Exception:
            print(f"   - {gpu.name}")

    if mixed_precision:
        from tensorflow.keras import mixed_precision as mp
        mp.set_global_policy("mixed_float16")
        print(f"✓ Mixed precision enabled (policy: {mp.global_policy().name})")


def _run_label_from_metadata(run_dir: str) -> str | None:
    """Return ``'benign'`` or ``'attack'`` based on the run's metadata.json.

    Training-set rules:
      - Runs with ``attack == 'none'`` contribute every row as a benign sample.
      - Runs with any other ``attack`` value contribute only their attack-tagged
        rows; null-tagged rows inside an attack run are contaminated broadcasts
        (couldn't be attributed to a drone) and must be excluded.

    Returns ``None`` when metadata.json is missing or unreadable so the caller
    can skip the run rather than mislabel its data.
    """
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return "benign" if str(meta.get("attack", "")).lower() == "none" else "attack"


def load_window_normalize_log(
    log_path: str,
    primary_type: str,
    window_size: int,
    stride: int,
    core_cols: list[str],
    *,
    run_label: str,
    label_policy: str = "any",  # retained for API compat; ignored when run_label is set
) -> tuple[list[np.ndarray], list[int]] | None:
    """Load one per-drone log file, slide windows, return per-window labels.

    Each ``<run_dir>/csv/drone_<NNN>.csv`` produced by DVD-swarm contains
    every MAVLink event flowing through one drone's mavlink-routerd. The
    file is therefore physically one drone's perspective — no groupby
    needed — and we just filter to primary_type, sort by timestamp, and
    slide windows.

    Labels come from the *run's* ``attack`` field (via ``run_label``), not
    from per-row ``attack_type``:
      - ``run_label='benign'``: keep every row, label every window 0.
      - ``run_label='attack'``: drop rows where ``attack_type == 'null'``
        (contaminated broadcasts), label every remaining window 1.

    Args:
        log_path: Path to one drone's CSV (e.g. ``.../csv/drone_005.csv``).
        primary_type: MAVLink message type to retain, e.g. ``"ATTITUDE"``.
        window_size: Number of timesteps per window.
        stride: Step between successive windows.
        core_cols: Feature columns used by engineer_features / preprocess_df.
        run_label: ``'benign'`` or ``'attack'`` — typically supplied by
            ``_run_label_from_metadata`` for the run dir containing log_path.
        label_policy: Retained for backward compatibility, no longer used.

    Returns:
        ``(windows, labels)`` lists, or ``None`` if the file has fewer than
        ``window_size`` qualifying rows of ``primary_type``.
    """
    if run_label not in ("benign", "attack"):
        raise ValueError(f"run_label must be 'benign' or 'attack', got {run_label!r}")
    del label_policy  # explicitly unused

    # Prefer flight.parquet over the per-drone CSV — predicate pushdown
    # filters by drone_id + mav_packet_type at read time so we only touch
    # the relevant rows. 10-100x faster than reading every CSV in full.
    run_dir = os.path.dirname(os.path.dirname(log_path))
    pq_path = os.path.join(run_dir, "flight.parquet")
    used_parquet = False
    if os.path.exists(pq_path):
        try:
            drone_id = int(
                os.path.basename(log_path).replace("drone_", "").replace(".csv", "")
            )
            df = pd.read_parquet(
                pq_path,
                filters=[
                    ("drone_id", "==", drone_id),
                    ("mav_packet_type", "==", primary_type),
                ],
            )
            # analyze.py writes parquet with all_varchar=true (handles the
            # mixed-type checksum column). Coerce numerics back here.
            keep_str = {"mav_packet_type", "attack_type"}
            for col in df.columns:
                if col not in keep_str:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            used_parquet = True
        except Exception as exc:
            print(f"  parquet read failed for {pq_path} ({exc!s}); falling back to CSV")

    if not used_parquet:
        df = pd.read_csv(log_path, low_memory=False)
        df = df[df["mav_packet_type"] == primary_type].copy()

    # Apply the per-run filter rule before windowing.
    if run_label == "attack":
        df = df[df["attack_type"] != "null"].copy()

    if len(df) < window_size:
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)

    binary_label = 1 if run_label == "attack" else 0

    pseudo_csv = primary_type + ".csv"
    df_processed = preprocess_df(df, core_cols, pseudo_csv)
    features = normalize_simulation(df_processed.values)
    if len(features) < window_size:
        return None

    windows: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(0, len(features) - window_size + 1, stride):
        windows.append(features[i : i + window_size])
        labels.append(binary_label)

    if not windows:
        return None

    print(f"  Loaded {len(windows):>5} {run_label} windows <- {log_path}")
    return (windows, labels)


def balance_sims(sims):
    min_windows = min(len(w) for w, _ in sims)
    balanced    = []
    for windows, label in sims:
        if len(windows) > min_windows:
            idx = np.random.choice(len(windows), min_windows, replace=False)
            balanced.append(([windows[i] for i in idx], label))
        else:
            balanced.append((windows, label))
    return balanced


def balance_classes(sims):
    benign = [s for s in sims if s[1] == 0]
    attack = [s for s in sims if s[1] == 1]
    n      = min(len(benign), len(attack))
    return benign[:n] + attack[:n]


def flatten_sims(sims):
    X, y = [], []
    for windows, label in sims:
        X.extend(windows)
        y.extend([label] * len(windows))
    return np.array(X), np.array(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',      type=str, required=True,
                        help='Root directory; all **/log.csv files under it are used.')
    parser.add_argument('--primary-type',  type=str, required=True,
                        help='MAVLink message type to train on, e.g. ATTITUDE.')
    parser.add_argument('--label-policy',  type=str, default='any',
                        choices=['any', 'majority', 'all'],
                        help='Per-window label aggregation (default: any).')
    parser.add_argument('--window-size',   type=int, default=80)
    parser.add_argument('--stride',        type=int, default=2)
    parser.add_argument('--output',        type=str, default=None)
    parser.add_argument('--test-frac', type=float, default=0.2,
                        help='Fraction of runs reserved for the held-out test set '
                             '(default 0.2 = nested 80/20: train 64%%, val 16%%, test 20%%).')
    parser.add_argument('--val-frac', type=float, default=0.2,
                        help='Fraction of non-test runs used for validation '
                             '(default 0.2 of the remaining 80%% = 16%% of total).')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable float16 mixed precision (≈2× faster on Volta+ GPUs).')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU-only execution (hide all GPUs from TF).')
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("CUDA_VISIBLE_DEVICES=-1 set — TF will run on CPU.")
    _configure_gpus(args.mixed_precision)

    tf.random.set_seed(42)
    np.random.seed(42)

    # primary_type is used as the dispatch key; reuse get_feature_cols via
    # the pseudo csv-name convention already established in ATTACK_FEATURE_MAP.
    pseudo_csv = args.primary_type + ".csv"
    core_cols  = get_feature_cols(pseudo_csv)
    if core_cols is None:
        raise ValueError(f"No feature map for primary type '{args.primary_type}'. "
                         f"Add it to ATTACK_FEATURE_MAP.")

    print(f"\nPrimary type : {args.primary_type}")
    print(f"Label policy : {args.label_policy}")
    print(f"Core features: {core_cols}")

    # ── Glob all per-drone log files ──────────────────────────────────────────
    drone_files = sorted(glob.glob(
        os.path.join(args.data_dir, '**', 'csv', 'drone_*.csv'), recursive=True
    ))
    if not drone_files:
        raise FileNotFoundError(
            f"No csv/drone_*.csv files found under '{args.data_dir}'. "
            "Check --data-dir and that DVD-swarm has been run with the "
            "per-drone PacketWriter."
        )

    # Group files by their parent run directory so the train/val split is
    # at the RUN level — drones from one run never span both folds.
    runs: dict[str, list[str]] = {}
    for f in drone_files:
        # .../output/run_X/csv/drone_NNN.csv  ->  .../output/run_X
        run_dir = os.path.dirname(os.path.dirname(f))
        runs.setdefault(run_dir, []).append(f)

    run_paths = sorted(runs.keys())
    print(f"Found {len(drone_files)} drone file(s) across {len(run_paths)} run(s)")

    # ── Run-level class assignment via metadata.json ─────────────────────────
    # benign  = rows from runs where the run's `attack` field is 'none'
    # attack  = rows where attack_type != 'null' inside an attack run
    # Runs missing metadata.json are skipped (can't tell which class).
    run_labels: dict[str, str] = {}
    for run_path in run_paths:
        rl = _run_label_from_metadata(run_path)
        if rl is None:
            print(f"  WARN: skipping {run_path} — no readable metadata.json")
            continue
        run_labels[run_path] = rl

    if not run_labels:
        raise RuntimeError("No runs had readable metadata.json — cannot label data.")

    runs = {k: v for k, v in runs.items() if k in run_labels}
    run_paths = sorted(runs.keys())
    drone_files = [f for r in run_paths for f in runs[r]]
    n_benign_runs = sum(1 for v in run_labels.values() if v == "benign")
    n_attack_runs = sum(1 for v in run_labels.values() if v == "attack")
    print(f"After metadata check: {len(drone_files)} drone file(s) across "
          f"{len(run_paths)} run(s) — {n_benign_runs} benign / {n_attack_runs} attack")

    # ── Per-run 3-way split — train / val / test, stratified by class ───────
    # Run-level isolation: drones from one run never span folds. The test
    # set is held out from BOTH fit and callbacks — only used once at the end.
    if len(run_paths) < 3:
        raise RuntimeError(
            f"Only {len(run_paths)} run(s) found; need at least 3 for "
            f"train/val/test. Run more sims before training."
        )

    stratify = [run_labels[r] for r in run_paths]

    def _safe_stratified_split(items, test_size, strat, label):
        try:
            return train_test_split(
                items, test_size=test_size, random_state=42, stratify=strat
            )
        except ValueError as exc:
            print(f"  WARN: stratified {label} split failed ({exc}); "
                  f"falling back to unstratified.")
            return train_test_split(items, test_size=test_size, random_state=42)

    # Split off test first so it stays sacrosanct.
    trainval_runs, test_runs = _safe_stratified_split(
        run_paths, args.test_frac, stratify, "test"
    )
    # Then split the remaining into train/val.
    trainval_stratify = [run_labels[r] for r in trainval_runs]
    train_runs, val_runs = _safe_stratified_split(
        trainval_runs, args.val_frac, trainval_stratify, "val"
    )

    train_files = [f for r in train_runs for f in runs[r]]
    val_files   = [f for r in val_runs   for f in runs[r]]
    test_files  = [f for r in test_runs  for f in runs[r]]

    def _class_summary(rs):
        c = Counter(run_labels[r] for r in rs)
        return f"{c.get('benign', 0)}B/{c.get('attack', 0)}A"

    print(f"Train: {len(train_runs)} run(s) ({_class_summary(train_runs)}, "
          f"{len(train_files)} drones)")
    print(f"Val:   {len(val_runs)} run(s) ({_class_summary(val_runs)}, "
          f"{len(val_files)} drones)")
    print(f"Test:  {len(test_runs)} run(s) ({_class_summary(test_runs)}, "
          f"{len(test_files)} drones)  [held out — only evaluated post-fit]")

    # ── Save Stage 1 config (used by live detector at inference time) ─────────
    # TODO: per-window Stage 1 flat-line filtering during inference
    attack_id  = args.primary_type
    model_name = args.output or f"aeroshield_{attack_id}.keras"
    cfg        = FROZEN_CONFIG.get(args.primary_type)

    stage1_config = {
        'frozen_cols'     : cfg['cols']      if cfg else [],
        'frozen_threshold': cfg['threshold'] if cfg else 1e-6
    }
    with open(f"stage1_{attack_id}.pkl", 'wb') as f:
        pickle.dump(stage1_config, f)
    print(f"\n✓ Stage 1 config saved → stage1_{attack_id}.pkl")

    # ── Load windows — one file = one drone = one sim ─────────────────────────
    def _load_split(files, name):
        windows: list[np.ndarray] = []
        labels: list[int] = []
        n_sims = 0
        print(f"\n── Loading {name} sims ────────────────────────────────────────────")
        for f in files:
            run_dir = os.path.dirname(os.path.dirname(f))
            out = load_window_normalize_log(
                f, args.primary_type,
                args.window_size, args.stride, core_cols,
                run_label=run_labels[run_dir],
                label_policy=args.label_policy,
            )
            if out is None:
                continue
            ws, ls = out
            windows.extend(ws)
            labels.extend(ls)
            n_sims += 1
        return windows, labels, n_sims

    all_train_windows, all_train_labels, n_train_sims = _load_split(train_files, "train")
    all_val_windows,   all_val_labels,   n_val_sims   = _load_split(val_files,   "val")
    all_test_windows,  all_test_labels,  n_test_sims  = _load_split(test_files,  "test")

    print(f"\nTrain sims (drones): {n_train_sims}  |  "
          f"Val sims: {n_val_sims}  |  Test sims: {n_test_sims}")

    if not all_train_windows:
        raise RuntimeError("No training windows produced — check --data-dir and --primary-type.")

    X_train = np.array(all_train_windows)
    y_train = np.array(all_train_labels)
    X_val   = np.array(all_val_windows)
    y_val   = np.array(all_val_labels)
    X_test  = np.array(all_test_windows) if all_test_windows else np.empty((0, args.window_size, 0))
    y_test  = np.array(all_test_labels)

    raw_train_dist = Counter(y_train.tolist())
    print(f"\nRaw train class distribution : {raw_train_dist}")
    print(f"Val class distribution        : {Counter(y_val.tolist())}")
    print(f"Test class distribution       : {Counter(y_test.tolist())}")

    # ── Window-level class balancing (train only) ─────────────────────────────
    n_minority = min(raw_train_dist.get(0, 0), raw_train_dist.get(1, 0))
    if n_minority == 0:
        print("WARNING: only one class in training set — skipping balancing.")
    else:
        idx0 = np.where(y_train == 0)[0]
        idx1 = np.where(y_train == 1)[0]
        rng  = np.random.default_rng(42)
        idx0 = rng.choice(idx0, size=n_minority, replace=False)
        idx1 = rng.choice(idx1, size=n_minority, replace=False)
        balanced_idx = np.concatenate([idx0, idx1])
        rng.shuffle(balanced_idx)
        X_train = X_train[balanced_idx]
        y_train = y_train[balanced_idx]

    print(f"Post-balance train distribution: {Counter(y_train.tolist())}")
    print(f"\nTrain windows     : {len(X_train)}")
    print(f"Val windows       : {len(X_val)}")
    print(f"Features/timestep : {X_train.shape[2]}")

    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    total    = len(y_train)
    n_benign = int(np.sum(y_train == 0))
    n_attack = int(np.sum(y_train == 1))
    # Guard against a fully-one-class train set after balancing.
    class_weight = {
        0: total / (2 * n_benign) if n_benign else 1.0,
        1: total / (2 * n_attack) if n_attack else 1.0,
    }
    print(f"Class weights     : {class_weight}")

    seq_len    = X_train.shape[1]   # window_size after possible MaxPooling shrinkage
    n_features = X_train.shape[2]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Sequential([
        Input(shape=(seq_len, n_features)),

        Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Bidirectional(LSTM(32, return_sequences=True,
                           kernel_regularizer=l2(1e-5),
                           recurrent_regularizer=l2(1e-5))),
        BatchNormalization(),
        Dropout(0.2),

        GlobalAveragePooling1D(),

        Dense(32, activation='relu', kernel_regularizer=l2(1e-5)),
        Dropout(0.2),
        # dtype='float32' keeps the loss in fp32 even when mixed_precision
        # is enabled — avoids underflow in focal_loss.
        Dense(1, activation='sigmoid', dtype='float32')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    model.summary()

    callbacks = [
        ReduceLROnPlateau(
            monitor='val_auc', factor=0.5, patience=3,
            min_lr=1e-6, mode='max', verbose=1
        ),
        EarlyStopping(
            monitor='val_auc', patience=6,
            restore_best_weights=True, mode='max', verbose=1
        )
    ]

    model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks
    )

    # ── Validation evaluation (tuning) ────────────────────────────────────────
    # Use the val set to pick the operating threshold (max F1). Test set
    # results below use this SAME threshold — never re-tuned on test.
    y_pred_prob = model.predict(X_val, verbose=0)
    prec, rec, thresh = precision_recall_curve(y_val, y_pred_prob)
    f1          = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_thresh = thresh[np.argmax(f1)]
    print(f"\nOptimal threshold (from val) : {best_thresh:.4f}")

    y_pred = (y_pred_prob >= best_thresh).astype(int)
    print("\n── Val classification report ─────────────────────────────────────")
    print(classification_report(y_val, y_pred, target_names=['Benign', 'Attack'], digits=4))
    cm = confusion_matrix(y_val, y_pred)
    print(f"""Val Confusion Matrix:
      Predicted Benign  Predicted Attack
Actual Benign    {cm[0][0]:<6}           {cm[0][1]:<6}   (False Alarms)
Actual Attack    {cm[1][0]:<6}           {cm[1][1]:<6}   (Missed Attacks)
""")

    # ── Held-out test evaluation (final, untouched until now) ─────────────────
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        y_test_prob = model.predict(X_test, verbose=0)
        y_test_pred = (y_test_prob >= best_thresh).astype(int)
        print("\n── TEST classification report (held-out runs) ────────────────────")
        print(classification_report(y_test, y_test_pred,
                                    target_names=['Benign', 'Attack'], digits=4))
        cm_t = confusion_matrix(y_test, y_test_pred)
        print(f"""Test Confusion Matrix:
      Predicted Benign  Predicted Attack
Actual Benign    {cm_t[0][0]:<6}           {cm_t[0][1]:<6}   (False Alarms)
Actual Attack    {cm_t[1][0]:<6}           {cm_t[1][1]:<6}   (Missed Attacks)
""")
        # Per-run breakdown so we can see which held-out runs the model
        # nailed vs which it struggled on.
        print("Test run-level summary:")
        for r in sorted(test_runs):
            print(f"  {run_labels[r]:<7}  {os.path.basename(r)}")
    elif len(X_test) == 0:
        print("\n⚠  No test windows produced — test set may have been too small.")
    else:
        print(f"\n⚠  Test set has only one class ({Counter(y_test.tolist())}); "
              f"skipping test eval. Consider adjusting --test-frac or run set.")

    print(f"\n── Combined system ──")
    print(f"Stage 1 (flat-line rule): see stage1_{attack_id}.pkl for live inference")
    print(f"Stage 2 (ML): trained on per-window labels, primary type '{args.primary_type}'")

    model.save(model_name)
    print(f"\n✓ Stage 2 model saved → {model_name}")
    print(f"✓ Stage 1 config saved → stage1_{attack_id}.pkl")
