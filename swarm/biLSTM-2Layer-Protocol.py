import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
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
    df  = pd.read_csv(file_path)
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
    df = pd.read_csv(file_path)
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


def load_window_normalize_log(
    log_path: str,
    primary_type: str,
    window_size: int,
    stride: int,
    core_cols: list[str],
    label_policy: str = "any",
) -> tuple[list[np.ndarray], list[int]] | None:
    """Load a unified log.csv, filter to primary_type rows, and slide windows.

    The attack flag is captured from the attack_type column before preprocess_df
    strips non-numeric columns.  Per-window labels are derived according to
    label_policy so that a single sim can contain both benign and attack windows.

    Args:
        log_path: Path to a unified log.csv produced by DVD-swarm.
        primary_type: MAVLink message type to retain, e.g. "ATTITUDE".
        window_size: Number of timesteps per window.
        stride: Step between successive windows.
        core_cols: Feature columns used by engineer_features / preprocess_df.
        label_policy: How to assign a label to each window:
            "any"      - 1 if any row in the window is an attack.
            "majority" - 1 if more than half the rows are attacks.
            "all"      - 1 if every row in the window is an attack.

    Returns:
        (windows, labels) where windows is a list of np.ndarray of shape
        (window_size, n_features) and labels is a list of int (0 or 1).
        Returns None if there are fewer rows than window_size after filtering.
    """
    df = pd.read_csv(log_path)

    # Filter to the message type we care about and sort chronologically.
    df = df[df["mav_packet_type"] == primary_type].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)

    if len(df) < window_size:
        return None

    # Capture attack flag BEFORE preprocess_df removes the string column.
    attack_flag: np.ndarray = (df["attack_type"] != "null").astype(int).to_numpy()

    # Reuse the existing pipeline — pass the type name as the pseudo csv_name
    # so engineer_features / get_feature_cols dispatch correctly.
    pseudo_csv = primary_type + ".csv"
    df = preprocess_df(df, core_cols, pseudo_csv)
    features = normalize_simulation(df.values)

    if len(features) < window_size:
        return None

    windows: list[np.ndarray] = []
    labels: list[int] = []
    for i in range(0, len(features) - window_size + 1, stride):
        window_flags = attack_flag[i : i + window_size]
        if label_policy == "majority":
            label = int(window_flags.mean() > 0.5)
        elif label_policy == "all":
            label = int(window_flags.all())
        else:  # "any" (default)
            label = int(window_flags.any())
        windows.append(features[i : i + window_size])
        labels.append(label)

    if not windows:
        return None

    n_attack = sum(labels)
    print(f"  Loaded {len(windows):>5} windows ({n_attack} attack, "
          f"{len(windows) - n_attack} benign) <- {log_path}")
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
    args = parser.parse_args()

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

    # ── Glob all unified log files ────────────────────────────────────────────
    log_files = sorted(glob.glob(
        os.path.join(args.data_dir, '**', 'log.csv'), recursive=True
    ))
    if not log_files:
        raise FileNotFoundError(
            f"No log.csv files found under '{args.data_dir}'. "
            "Check --data-dir and that DVD-swarm has been run."
        )
    print(f"Found {len(log_files)} sim log(s)")

    # ── Per-sim 80/20 split — no window leakage across the boundary ──────────
    train_logs, val_logs = train_test_split(
        log_files, test_size=0.2, random_state=42
    )
    print(f"Train sims: {len(train_logs)}  |  Val sims: {len(val_logs)}")

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

    # ── Load windows from each sim — flat-vs-subtle split is NOT applied ──────
    # With per-window labels every attack window is used directly for Stage 2.
    all_train_windows: list[np.ndarray] = []
    all_train_labels:  list[int]        = []
    all_val_windows:   list[np.ndarray] = []
    all_val_labels:    list[int]        = []

    for log_path in train_logs:
        out = load_window_normalize_log(
            log_path, args.primary_type,
            args.window_size, args.stride, core_cols,
            label_policy=args.label_policy,
        )
        if out is None:
            print(f"  Skipped (too few rows): {log_path}")
            continue
        ws, ls = out
        all_train_windows.extend(ws)
        all_train_labels.extend(ls)

    for log_path in val_logs:
        out = load_window_normalize_log(
            log_path, args.primary_type,
            args.window_size, args.stride, core_cols,
            label_policy=args.label_policy,
        )
        if out is None:
            print(f"  Skipped (too few rows): {log_path}")
            continue
        ws, ls = out
        all_val_windows.extend(ws)
        all_val_labels.extend(ls)

    if not all_train_windows:
        raise RuntimeError("No training windows produced — check --data-dir and --primary-type.")

    X_train = np.array(all_train_windows)
    y_train = np.array(all_train_labels)
    X_val   = np.array(all_val_windows)
    y_val   = np.array(all_val_labels)

    raw_train_dist = Counter(y_train.tolist())
    print(f"\nRaw train class distribution : {raw_train_dist}")
    print(f"Val class distribution        : {Counter(y_val.tolist())}")

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
        Dense(1, activation='sigmoid')
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

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_val, verbose=0)
    prec, rec, thresh = precision_recall_curve(y_val, y_pred_prob)
    f1          = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_thresh = thresh[np.argmax(f1)]
    print(f"\nOptimal threshold (subtle) : {best_thresh:.4f}")

    y_pred = (y_pred_prob >= best_thresh).astype(int)
    print("\nClassification Report (subtle attacks only):")
    print(classification_report(y_val, y_pred, target_names=['Benign', 'Attack'], digits=4))

    cm = confusion_matrix(y_val, y_pred)
    print(f"""
Confusion Matrix (subtle attacks):
      Predicted Benign  Predicted Attack
Actual Benign    {cm[0][0]:<6}           {cm[0][1]:<6}   (False Alarms)
Actual Attack    {cm[1][0]:<6}           {cm[1][1]:<6}   (Missed Attacks)
    """)

    print(f"\n── Combined system ──")
    print(f"Stage 1 (flat-line rule): see stage1_{attack_id}.pkl for live inference")
    print(f"Stage 2 (ML): trained on per-window labels via policy '{args.label_policy}'")

    model.save(model_name)
    print(f"\n✓ Stage 2 model saved → {model_name}")
    print(f"✓ Stage 1 config saved → stage1_{attack_id}.pkl")
