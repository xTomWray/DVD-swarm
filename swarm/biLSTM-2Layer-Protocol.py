import argparse
import glob
import json
import os
import pickle
import sys
from collections import Counter

# Ensure the sibling module ``preprocessing_cache`` is importable even when
# this file is invoked via an unusual sys.path (e.g. ``python -m`` or a
# wrapped launcher). When run as ``python swarm/biLSTM-2Layer-Protocol.py``
# the script's directory is already on sys.path, but this is defensive.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import preprocessing_cache as pc  # noqa: E402  -- sibling module

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def focal_loss(alpha=0.25, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_term = tf.pow((1 - p_t), gamma)
        return alpha * focal_term * bce

    return loss


# Columns dropped before feature engineering. The downstream
# engineer_features() filter is the final safety net — only columns
# matching core_cols + a small whitelist survive — but dropping these
# explicitly makes the preprocessing cheap and the intent obvious.
DROP_COLS = [
    # Identity / labels — not features
    "mav_packet_type",
    "sim_uuid",
    "attack_type",
    # Time columns — absolute and relative; carry no per-protocol signal
    # for ATTITUDE/GPS/VFR_HUD attack detection. Drop wholesale.
    "timestamp",
    "frame_timestamp",
    "time_boot_ms",
    "time_usec",
    "udp_time_relative",
    "tcp_time_relative",
    # MAVLink packet headers — transport-layer, not message content
    "magic",
    "payloadLength",
    "incompatibilityFlags",
    "compatibilityFlags",
    "seq",
    "sysid",
    "compid",
    "msgid",
    "checksum",
    "signature",
    # IP header fields
    "ip_version",
    "ip_hdr_len",
    "ip_tos",
    "ip_len",
    "ip_id",
    "ip_flags",
    "ip_frag_offset",
    "ip_ttl",
    "ip_proto",
    "ip_src",
    "ip_addr",
    "ip_src_host",
    "ip_host",
    "ip_dst",
    "ip_dst_host",
    "ip_checksum",
    "ip_checksum_status",
    "ip_payload",
    # UDP header fields
    "udp_srcport",
    "udp_dstport",
    "udp_length",
    "udp_checksum",
    "udp_checksum_status",
    "udp_payload",
    "udp_text",
    # TCP header fields
    "tcp_srcport",
    "tcp_dstport",
    "tcp_seq",
    "tcp_ack",
    "tcp_hdr_len",
    "tcp_flags",
    "tcp_flags_str",
    "tcp_window_size",
    "tcp_urgent_pointer",
    "tcp_checksum",
    "tcp_checksum_status",
    "tcp_options",
    "tcp_options_nop",
    "tcp_options_timestamp",
    "tcp_payload",
    "tcp_text",
]

ATTACK_FEATURE_MAP = {
    "ATTITUDE": ["roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"],
    "GPS_RAW_INT": [
        "lat",
        "lon",
        "alt",
        "alt_ellipsoid",
        "cog",
        "vel",
        "eph",
        "epv",
        "satellites_visible",
    ],
    "VFR_HUD": ["airspeed", "groundspeed", "heading", "throttle", "alt", "climb"],
}

# ── Per-CSV frozen detection config ───────────────────────────────────────────
FROZEN_CONFIG = {
    "ATTITUDE": {
        "cols": ["roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"],
        "threshold": 1e-6,
    },
    "GPS_RAW_INT": {"cols": ["lat", "lon", "alt", "vel"], "threshold": 1e-6},
    "VFR_HUD": {"cols": ["airspeed", "groundspeed", "alt", "climb"], "threshold": 1e-6},
}


def get_feature_cols(csv_name):
    key = csv_name.replace(".csv", "").upper()
    for k, cols in ATTACK_FEATURE_MAP.items():
        if k in key:
            return cols
    return None


# ── Feature engineering ───────────────────────────────────────────────────────
# Single-shot guard for the one-time feature-name log inside engineer_features.
_LOGGED_FEATURES = False


def rolling_autocorr(series, window, lag=1):
    s1 = series
    s2 = series.shift(lag)
    w = window
    mp = w // 2
    num = (
        ((s1 - s1.rolling(w, min_periods=mp).mean()) * (s2 - s2.rolling(w, min_periods=mp).mean()))
        .rolling(w, min_periods=mp)
        .mean()
    )
    std1 = s1.rolling(w, min_periods=mp).std().replace(0, 1e-9)
    std2 = s2.rolling(w, min_periods=mp).std().replace(0, 1e-9)
    return (num / (std1 * std2)).fillna(0)


def engineer_attitude_features(df):
    new_cols = {}
    for col in ["roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"]:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f"{col}_d1"] = d1
        new_cols[f"{col}_d2"] = d1.diff().fillna(0)
        for w in [20, 50]:
            roll = df[col].rolling(w, min_periods=w // 2)
            new_cols[f"{col}_autocorr_{w}"] = rolling_autocorr(df[col], w)
            new_cols[f"{col}_kurt_{w}"] = roll.kurt().fillna(0)
            new_cols[f"{col}_skew_{w}"] = roll.skew().fillna(0)
            new_cols[f"{col}_std_{w}"] = roll.std().fillna(0)
            new_cols[f"{col}_range_{w}"] = (roll.max() - roll.min()).fillna(0)

    if all(c in df.columns for c in ["rollspeed", "pitchspeed", "yawspeed"]):
        rate_mag = np.sqrt(df["rollspeed"] ** 2 + df["pitchspeed"] ** 2 + df["yawspeed"] ** 2)
        new_cols["rate_mag"] = rate_mag
        new_cols["rate_mag_std_20"] = rate_mag.rolling(20, min_periods=10).std().fillna(0)

    if all(c in df.columns for c in ["roll", "pitch", "yaw"]):
        new_cols["attitude_mag"] = np.sqrt(df["roll"] ** 2 + df["pitch"] ** 2 + df["yaw"] ** 2)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def engineer_gps_features(df):
    new_cols = {}
    for col in ["lat", "lon", "alt", "alt_ellipsoid", "cog", "vel", "eph", "epv"]:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f"{col}_d1"] = d1
        new_cols[f"{col}_d2"] = d1.diff().fillna(0)
        for w in [20, 50]:
            roll = df[col].rolling(w, min_periods=w // 2)
            new_cols[f"{col}_autocorr_{w}"] = rolling_autocorr(df[col], w)
            new_cols[f"{col}_kurt_{w}"] = roll.kurt().fillna(0)
            new_cols[f"{col}_std_{w}"] = roll.std().fillna(0)
            new_cols[f"{col}_range_{w}"] = (roll.max() - roll.min()).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def engineer_vfr_features(df):
    new_cols = {}
    for col in ["airspeed", "groundspeed", "heading", "throttle", "alt", "climb"]:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f"{col}_d1"] = d1
        new_cols[f"{col}_d2"] = d1.diff().fillna(0)
        for w in [20, 50]:
            roll = df[col].rolling(w, min_periods=w // 2)
            new_cols[f"{col}_autocorr_{w}"] = rolling_autocorr(df[col], w)
            new_cols[f"{col}_kurt_{w}"] = roll.kurt().fillna(0)
            new_cols[f"{col}_std_{w}"] = roll.std().fillna(0)
            new_cols[f"{col}_range_{w}"] = (roll.max() - roll.min()).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def engineer_features(df, core_cols, csv_name):
    key = csv_name.replace(".csv", "").upper()

    if "ATTITUDE" in key:
        attitude_cols = ["roll", "pitch", "yaw", "rollspeed", "pitchspeed", "yawspeed"]
        if all(c in df.columns for c in attitude_cols):
            df = engineer_attitude_features(df)
    elif "GPS_RAW_INT" in key:
        df = engineer_gps_features(df)
    elif "VFR_HUD" in key:
        df = engineer_vfr_features(df)

    new_cols = {}
    for col in core_cols:
        if col not in df.columns:
            continue
        d1 = df[col].diff().fillna(0)
        new_cols[f"{col}_d1"] = d1
        new_cols[f"{col}_d2"] = d1.diff().fillna(0)
        for w in [5, 10, 20]:
            roll = df[col].rolling(w, min_periods=1)
            new_cols[f"{col}_std_{w}"] = roll.std().fillna(0)
            new_cols[f"{col}_range_{w}"] = (roll.max() - roll.min()).fillna(0)
            new_cols[f"{col}_mean_{w}"] = roll.mean().fillna(0)
        local_std = df[col].rolling(20, min_periods=1).std().replace(0, 1e-9)
        local_mean = df[col].rolling(20, min_periods=1).mean()
        new_cols[f"{col}_zscore"] = ((df[col] - local_mean) / local_std).fillna(0)

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    # The specialised engineer_*_features() above can create columns like
    # roll_d1, roll_std_20 that the generic loop also creates — concat would
    # otherwise leave duplicated column labels. Keep the last occurrence
    # (the generic loop uses min_periods=1, which is what we want).
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    extra = ["rate_mag", "rate_mag_std_20", "attitude_mag"]
    keep = [
        c
        for c in df.columns
        if any(c == col or c.startswith(col + "_") for col in core_cols) or c in extra
    ]

    global _LOGGED_FEATURES
    if not _LOGGED_FEATURES:
        dropped = sorted(c for c in df.columns if c not in keep)
        print(f"\n── Feature set ({csv_name}) ──")
        print(f"  Kept ({len(keep)}): {sorted(keep)}")
        if dropped:
            preview = dropped if len(dropped) <= 12 else dropped[:12] + ["…"]
            print(f"  Dropped ({len(dropped)}): {preview}")
        _LOGGED_FEATURES = True

    return df[keep].fillna(0)


def preprocess_df(df, core_cols, csv_name):
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    # Keep only numeric columns. String/object cols (mav_packet_type, etc.)
    # already in DROP_COLS, but this guards against new MAVLink string fields.
    df = df.select_dtypes(include=[np.number])
    df = df.fillna(0)
    df = engineer_features(df, core_cols, csv_name)
    return df


def _artifact_paths(output: str | None, primary_type: str) -> tuple[str, str, str]:
    """Return (scaler_path, manifest_path, stage1_path) all derived from --output.

    With parallel training processes sharing one --primary-type, fixed
    `stage1_<type>.pkl` paths collided; routing it through --output keeps
    each sweep child's artifacts isolated.
    """
    if output:
        output_dir = os.path.dirname(output)
        stem = os.path.splitext(os.path.basename(output))[0]
        prefix = os.path.join(output_dir, stem) if output_dir else stem
        return (
            f"{prefix}_scaler.joblib",
            f"{prefix}_split_manifest.json",
            f"{prefix}_stage1.pkl",
        )
    return (
        f"scaler_{primary_type}.joblib",
        "model_split_manifest.json",
        f"stage1_{primary_type}.pkl",
    )


def _configure_gpus(mixed_precision: bool) -> None:
    """Detect GPUs, enable memory growth, optionally enable mixed precision.

    Loud-prints what TF will train on so a silent CPU fallback can't waste
    hours of wall time. Raises nothing — if GPUs aren't detected the script
    still runs (just slowly).
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("⚠  No GPU detected — training will run on CPU (slow).")
        print(
            "   If this is a GPU box, check: nvidia-smi, CUDA install, "
            "and that TensorFlow was built with GPU support."
        )
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


def load_preprocessed_rows(
    log_path: str,
    primary_type: str,
    core_cols: list[str],
    *,
    run_label: str,
) -> dict[str, object] | None:
    """Load one per-drone log file and return preprocessed rows plus metadata.

    Each ``<run_dir>/csv/drone_<NNN>.csv`` produced by DVD-swarm contains
    every MAVLink event flowing through one drone's mavlink-routerd. The
    file is therefore physically one drone's perspective. This function
    filters to primary_type, sorts causally, assigns row labels, preserves
    timestamps for metadata, and engineers features without fitting a scaler.

    Row labels come from both the run label and per-row ``attack_type``:
      - ``run_label='benign'``: keep every row, label every window 0.
      - ``run_label='attack'``: drop rows where ``attack_type == 'null'``
        after computing labels, preserving the existing attack-row-only behavior.
    """
    if run_label not in ("benign", "attack"):
        raise ValueError(f"run_label must be 'benign' or 'attack', got {run_label!r}")

    # Prefer flight.parquet over the per-drone CSV — predicate pushdown
    # filters by drone_id + mav_packet_type at read time and column
    # projection limits the read to the ~8 cols we actually use (out of
    # ~1100 in the union schema). 100-500x faster than reading every
    # row × every column then coercing.
    run_dir = os.path.dirname(os.path.dirname(log_path))
    pq_path = os.path.join(run_dir, "flight.parquet")
    used_parquet = False
    if os.path.exists(pq_path):
        try:
            drone_id = int(os.path.basename(log_path).replace("drone_", "").replace(".csv", ""))
            # Only the columns we actually need downstream: timestamp for
            # sort, attack_type for the run-level filter, and core_cols for
            # feature engineering. Everything else is union-schema noise.
            wanted_cols = ["timestamp", "attack_type", *core_cols]
            try:
                df = pd.read_parquet(
                    pq_path,
                    filters=[
                        ("drone_id", "==", drone_id),
                        ("mav_packet_type", "==", primary_type),
                    ],
                    columns=wanted_cols,
                )
            except (KeyError, ValueError):
                # Some requested column missing from the parquet schema —
                # fall back to reading everything (rare; covered for safety).
                df = pd.read_parquet(
                    pq_path,
                    filters=[
                        ("drone_id", "==", drone_id),
                        ("mav_packet_type", "==", primary_type),
                    ],
                )
            # analyze.py writes parquet with all_varchar=true (handles the
            # mixed-type checksum column). Coerce numerics back here —
            # tiny loop now that we only have ~8 columns.
            for col in df.columns:
                if col != "attack_type":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            used_parquet = True
        except Exception as exc:
            print(f"  parquet read failed for {pq_path} ({exc!s}); falling back to CSV")

    if not used_parquet:
        # mav_packet_type is needed for the row filter, then dropped.
        wanted_cols = ["mav_packet_type", "timestamp", "attack_type", *core_cols]
        try:
            df = pd.read_csv(log_path, low_memory=False, usecols=wanted_cols)
        except (ValueError, KeyError):
            # Older runs may have a narrower CSV header — fall back to full read.
            df = pd.read_csv(log_path, low_memory=False)
        df = df[df["mav_packet_type"] == primary_type].copy()

    if df.empty:
        return None

    if "timestamp" not in df.columns:
        raise ValueError(f"{log_path} is missing required timestamp column")

    df = df.sort_values("timestamp").reset_index(drop=True)
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).to_numpy()

    if run_label == "benign":
        row_labels = np.zeros(len(df), dtype=np.int64)
    else:
        if "attack_type" not in df.columns:
            raise ValueError(f"{log_path} is an attack run but has no attack_type column")
        row_labels = (df["attack_type"].fillna("null").astype(str).str.lower() != "null").astype(
            np.int64
        ).to_numpy()

    # Preserve current behavior: attack runs contribute only attack-tagged rows.
    if run_label == "attack":
        keep_mask = row_labels == 1
        df = df.loc[keep_mask].reset_index(drop=True)
        timestamps = timestamps[keep_mask]
        row_labels = row_labels[keep_mask]

    if len(df) == 0:
        return None

    pseudo_csv = primary_type + ".csv"
    df_processed = preprocess_df(df, core_cols, pseudo_csv)
    if len(df_processed) == 0:
        return None

    return {
        "features": df_processed,
        "row_labels": row_labels,
        "timestamps": timestamps,
        "run_dir": run_dir,
        "drone_file": os.path.basename(log_path),
        "run_label": run_label,
        "log_path": log_path,
    }


def make_windows_from_scaled_rows(
    row_data: dict[str, object],
    scaled_features: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[list[np.ndarray], list[int], list[dict[str, object]]] | None:
    """Create right-edge causal windows from already-scaled rows."""
    row_labels = np.asarray(row_data["row_labels"], dtype=np.int64)
    timestamps = np.asarray(row_data["timestamps"])
    if len(scaled_features) < window_size:
        return None

    windows: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[dict[str, object]] = []
    for window_index, end_idx in enumerate(range(window_size, len(scaled_features) + 1, stride)):
        start_idx = end_idx - window_size
        windows.append(scaled_features[start_idx:end_idx])
        labels.append(int(row_labels[end_idx - 1]))
        metadata.append(
            {
                "run_dir": row_data["run_dir"],
                "drone_file": row_data["drone_file"],
                "window_index": window_index,
                "row_start_index": int(start_idx),
                "row_end_index": int(end_idx),
                "window_start_timestamp": float(timestamps[start_idx]),
                "window_end_timestamp": float(timestamps[end_idx - 1]),
                "run_label": row_data["run_label"],
            }
        )

    if not windows:
        return None

    return (windows, labels, metadata)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:.6f}"


def _max_false_negative_streak(records: list[dict[str, object]]) -> int:
    max_streak = 0
    current_streak = 0
    for record in sorted(records, key=lambda r: (r["window_end_timestamp"], r["window_index"])):
        if record["y_pred"] == 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak


def print_detection_metrics(split_name, y_true, y_prob, y_pred, metadata) -> None:
    """Print run/drone-aware timing and false-negative metrics for biLSTM predictions."""
    del y_prob

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(metadata):
        raise ValueError(
            f"{split_name} metadata length ({len(metadata)}) does not match labels ({len(y_true)})"
        )

    records = [
        {
            **metadata[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
        }
        for i in range(len(metadata))
    ]
    attack_records = [r for r in records if r["y_true"] == 1]

    print(f"\n── {split_name} detection timing metrics ─────────────────────────")
    if not attack_records:
        print("No attack windows in this split; detection-latency metrics are not applicable.")
        return

    records_by_run: dict[str, list[dict[str, object]]] = {}
    records_by_drone: dict[str, list[dict[str, object]]] = {}
    for record in attack_records:
        run_dir = str(record["run_dir"])
        drone_key = f"{run_dir}/{record['drone_file']}"
        records_by_run.setdefault(run_dir, []).append(record)
        records_by_drone.setdefault(drone_key, []).append(record)

    run_fn_streaks = {
        run_dir: _max_false_negative_streak(run_records)
        for run_dir, run_records in records_by_run.items()
    }
    drone_fn_streaks = {
        drone_key: _max_false_negative_streak(drone_records)
        for drone_key, drone_records in records_by_drone.items()
    }
    max_run_fn_streak = max(run_fn_streaks.values(), default=0)
    max_drone_fn_streak = max(drone_fn_streaks.values(), default=0)

    first_detection_delay_per_run: dict[str, float | None] = {}
    detected_delays: list[float] = []
    undetected_attack_runs = 0
    false_negative_positions = {"onset": 0, "middle": 0, "end": 0}
    position_names = ("onset", "middle", "end")

    for run_dir, run_records in sorted(records_by_run.items()):
        run_records = sorted(
            run_records,
            key=lambda r: (r["window_end_timestamp"], r["drone_file"], r["window_index"]),
        )
        attack_onset = min(float(r["window_start_timestamp"]) for r in run_records)
        detections = [r for r in run_records if r["y_pred"] == 1]
        if detections:
            first_detection = min(float(r["window_end_timestamp"]) for r in detections)
            delay = first_detection - attack_onset
            first_detection_delay_per_run[run_dir] = delay
            detected_delays.append(delay)
        else:
            first_detection_delay_per_run[run_dir] = None
            undetected_attack_runs += 1

        n_run_attack_windows = len(run_records)
        for position, record in enumerate(run_records):
            if record["y_pred"] == 0:
                bucket = min((position * 3) // n_run_attack_windows, 2)
                false_negative_positions[position_names[bucket]] += 1

    mean_time_to_first_detection = float(np.mean(detected_delays)) if detected_delays else None
    median_time_to_first_detection = float(np.median(detected_delays)) if detected_delays else None
    total_false_negatives = sum(false_negative_positions.values())

    print(f"max_consecutive_false_negatives_by_run_global: {max_run_fn_streak}")
    print(f"max_consecutive_false_negatives_by_drone_global: {max_drone_fn_streak}")
    print(f"mean_time_to_first_detection: {_format_seconds(mean_time_to_first_detection)}")
    print(f"median_time_to_first_detection: {_format_seconds(median_time_to_first_detection)}")
    print(f"undetected_attack_runs: {undetected_attack_runs}")
    print("max_consecutive_false_negatives_per_run:")
    for run_dir, streak in sorted(run_fn_streaks.items()):
        print(f"  {os.path.basename(run_dir)}: {streak}")
    print("max_consecutive_false_negatives_per_drone:")
    for drone_key, streak in sorted(drone_fn_streaks.items()):
        run_dir, drone_file = os.path.split(drone_key)
        print(f"  {os.path.basename(run_dir)}/{drone_file}: {streak}")
    print("first_detection_delay_per_run:")
    for run_dir, delay in sorted(first_detection_delay_per_run.items()):
        print(f"  {os.path.basename(run_dir)}: {_format_seconds(delay)}")
    print("false_negative_positions:")
    for name in position_names:
        count = false_negative_positions[name]
        pct = (count / total_false_negatives * 100.0) if total_false_negatives else 0.0
        print(f"  {name}: {count} ({pct:.2f}%)")


def print_classification_breakdown(split_name, y_true, y_pred, metadata) -> None:
    """Per-run and per-drone attack-class precision/recall/F1, sorted worst-first.

    The global classification_report shows aggregate performance; this breakdown
    surfaces *which* run or drone the model fails on. NaN F1 (e.g. a run with
    only false positives) is sorted to the top so it is not lost.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) != len(metadata):
        raise ValueError(
            f"{split_name} metadata length ({len(metadata)}) does not match labels ({len(y_true)})"
        )

    by_run: dict[str, list[int]] = {}
    by_drone: dict[str, list[int]] = {}
    for i, m in enumerate(metadata):
        run_dir = str(m["run_dir"])
        by_run.setdefault(run_dir, []).append(i)
        by_drone.setdefault(f"{run_dir}/{m['drone_file']}", []).append(i)

    def _stats(idxs: list[int]) -> tuple[float, float, float, int]:
        idx_arr = np.asarray(idxs, dtype=np.int64)
        yt = y_true[idx_arr]
        yp = y_pred[idx_arr]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        support = int((yt == 1).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        if tp and (tp + fp) and (tp + fn):
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = float("nan")
        return prec, rec, f1, support

    def _sort_key(row: tuple[str, float, float, float, int]) -> tuple[int, float]:
        # NaN F1 sorts to the top (0), then real F1 ascending (worst first).
        _name, _p, _r, f1, _s = row
        return (0, 0.0) if f1 != f1 else (1, f1)

    print(f"\n── {split_name} per-run attack-class breakdown ───────────────────")
    run_rows = [
        (os.path.basename(r), *_stats(idxs))
        for r, idxs in by_run.items()
        if (np.asarray(y_true)[np.asarray(idxs, dtype=np.int64)] == 1).any()
    ]
    run_rows.sort(key=_sort_key)
    print(f"  {'run':<48} {'prec':>8} {'rec':>8} {'F1':>8} {'support':>10}")
    for name, p, r, f1, support in run_rows:
        print(f"  {name:<48} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {support:>10d}")
    if not run_rows:
        print("  (no runs with attack windows in this split)")

    print(f"\n── {split_name} per-drone attack-class breakdown ─────────────────")
    drone_rows = [
        (f"{os.path.basename(os.path.dirname(k))}/{os.path.basename(k)}", *_stats(idxs))
        for k, idxs in by_drone.items()
        if (np.asarray(y_true)[np.asarray(idxs, dtype=np.int64)] == 1).any()
    ]
    drone_rows.sort(key=_sort_key)
    print(f"  {'drone':<64} {'prec':>8} {'rec':>8} {'F1':>8} {'support':>10}")
    for name, p, r, f1, support in drone_rows:
        print(f"  {name:<64} {p:>8.4f} {r:>8.4f} {f1:>8.4f} {support:>10d}")
    if not drone_rows:
        print("  (no drones with attack windows in this split)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory; all **/log.csv files under it are used.",
    )
    parser.add_argument(
        "--primary-type",
        type=str,
        required=True,
        help="MAVLink message type to train on, e.g. ATTITUDE.",
    )
    parser.add_argument("--window-size", type=int, default=80)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--scaler-fit-scope",
        choices=("train_all", "train_benign"),
        default="train_all",
        help="Rows used to fit the StandardScaler: all training rows or only benign training rows.",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.2,
        help="Fraction of runs reserved for the held-out test set "
        "(default 0.2 = nested 80/20: train 64%%, val 16%%, test 20%%).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of non-test runs used for validation "
        "(default 0.2 of the remaining 80%% = 16%% of total).",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable float16 mixed precision (≈2× faster on Volta+ GPUs).",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU-only execution (hide all GPUs from TF)."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="If set, cache (or load) preprocessed + scaled rows under this directory. "
        "Lets a hyperparameter sweep over --window-size/--stride skip the slow CSV→scaler stages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Populate the cache (if --cache-dir is set) and exit before windowing/model fit. "
        "Used by sweep launchers to warm the cache on CPU before launching GPU children.",
    )
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
    core_cols = get_feature_cols(pseudo_csv)
    if core_cols is None:
        raise ValueError(
            f"No feature map for primary type '{args.primary_type}'. Add it to ATTACK_FEATURE_MAP."
        )

    print(f"\nPrimary type : {args.primary_type}")
    print(f"Core features: {core_cols}")

    # ── Glob all per-drone log files ──────────────────────────────────────────
    drone_files = sorted(
        glob.glob(os.path.join(args.data_dir, "**", "csv", "drone_*.csv"), recursive=True)
    )
    if not drone_files:
        raise FileNotFoundError(
            f"No csv/drone_*.csv files found under '{args.data_dir}'. "
            "Check --data-dir and that DVD-swarm has been run with the "
            "per-drone PacketWriter."
        )

    # Group files by their parent run directory. Used only for metadata
    # lookup (run-level attack label); the actual split is per-drone below.
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
    print(
        f"After metadata check: {len(drone_files)} drone file(s) across "
        f"{len(run_paths)} run(s) — {n_benign_runs} benign / {n_attack_runs} attack"
    )

    # ── Run-level 3-way split — train / val / test ───────────────────────────
    # Runs are the atomic unit. This prevents drones from the same simulation
    # run leaking across train/validation/test through shared timing and attack
    # conditions.
    if len(run_paths) < 3:
        raise RuntimeError(
            f"Only {len(run_paths)} run(s) found; need at least 3 for train/val/test split."
        )

    def _safe_run_split(items, test_size, label):
        strat = [run_labels[r] for r in items]
        counts = Counter(strat)
        use_stratify = len(counts) > 1 and min(counts.values()) >= 2
        try:
            return train_test_split(
                items,
                test_size=test_size,
                random_state=42,
                stratify=strat if use_stratify else None,
            )
        except ValueError as exc:
            print(f"  WARN: stratified {label} split failed ({exc}); falling back to unstratified.")
            return train_test_split(items, test_size=test_size, random_state=42)

    trainval_runs, test_runs = _safe_run_split(run_paths, args.test_frac, "test")
    train_runs, val_runs = _safe_run_split(trainval_runs, args.val_frac, "val")

    train_files = [f for r in sorted(train_runs) for f in runs[r]]
    val_files = [f for r in sorted(val_runs) for f in runs[r]]
    test_files = [f for r in sorted(test_runs) for f in runs[r]]

    train_set, val_set, test_set = set(train_runs), set(val_runs), set(test_runs)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise AssertionError("Run-level split overlap detected")

    def _class_summary(run_items):
        c = Counter(run_labels[r] for r in run_items)
        return f"{c.get('benign', 0)}B/{c.get('attack', 0)}A"

    print("Split unit: run")
    print("Train/Val/Test run overlap: none")
    print(f"Train: {len(train_runs)} run(s), {len(train_files)} drone file(s) ({_class_summary(train_runs)})")
    print(f"Val:   {len(val_runs)} run(s), {len(val_files)} drone file(s) ({_class_summary(val_runs)})")
    print(
        f"Test:  {len(test_runs)} run(s), {len(test_files)} drone file(s) ({_class_summary(test_runs)})  "
        f"[held out — only evaluated post-fit]"
    )

    can_have_both_classes = n_benign_runs >= 3 and n_attack_runs >= 3
    if can_have_both_classes:
        for split_name, split_runs in (("train", train_runs), ("val", val_runs), ("test", test_runs)):
            split_classes = {run_labels[r] for r in split_runs}
            if split_classes != {"benign", "attack"}:
                print(
                    f"  WARN: feasible mixed-class split expected, but {split_name} "
                    f"has classes {sorted(split_classes)}"
                )

    # ── Save Stage 1 config (used by live detector at inference time) ─────────
    # TODO: per-window Stage 1 flat-line filtering during inference
    attack_id = args.primary_type
    model_name = args.output or f"aeroshield_{attack_id}.keras"
    scaler_path, manifest_path, stage1_path = _artifact_paths(args.output, attack_id)
    manifest = {
        "train_runs": sorted(train_runs),
        "val_runs": sorted(val_runs),
        "test_runs": sorted(test_runs),
        "train_files": sorted(train_files),
        "val_files": sorted(val_files),
        "test_files": sorted(test_files),
        "n_train_runs": len(train_runs),
        "n_val_runs": len(val_runs),
        "n_test_runs": len(test_runs),
        "n_train_files": len(train_files),
        "n_val_files": len(val_files),
        "n_test_files": len(test_files),
        "split_seed": 42,
        "split_unit": "run",
        "scaler_fit": "train_only",
        "scaler_fit_scope": args.scaler_fit_scope,
        "window_size": args.window_size,
        "stride": args.stride,
        "primary_type": args.primary_type,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✓ Split manifest saved → {manifest_path}")

    cfg = FROZEN_CONFIG.get(args.primary_type)

    stage1_config = {
        "frozen_cols": cfg["cols"] if cfg else [],
        "frozen_threshold": cfg["threshold"] if cfg else 1e-6,
    }
    with open(stage1_path, "wb") as f:
        pickle.dump(stage1_config, f)
    print(f"\n✓ Stage 1 config saved → {stage1_path}")

    # ── Cache key (data fingerprint + preprocessing knobs) ────────────────────
    cache_key = None
    cache_subdir = None
    if args.cache_dir:
        cache_key = pc.compute_cache_key(
            pc.CacheKeyInputs(
                drone_files=tuple(sorted(drone_files)),
                primary_type=args.primary_type,
                scaler_fit_scope=args.scaler_fit_scope,
                test_frac=args.test_frac,
                val_frac=args.val_frac,
                core_cols=tuple(core_cols),
                split_seed=42,
                script_identity="2layer",
            )
        )
        cache_subdir = os.path.join(args.cache_dir, cache_key)

    cache_hit = cache_subdir is not None and pc.cache_exists(cache_subdir)

    # ── Load/preprocess rows before fitting one train-only scaler ─────────────
    def _load_rows_split(files, name):
        row_items: list[dict[str, object]] = []
        print(f"\n── Loading {name} rows ────────────────────────────────────────────")
        for f in files:
            run_dir = os.path.dirname(os.path.dirname(f))
            out = load_preprocessed_rows(
                f,
                args.primary_type,
                core_cols,
                run_label=run_labels[run_dir],
            )
            if out is None:
                continue
            row_items.append(out)
            print(f"  Loaded {len(out['features']):>7} {run_labels[run_dir]} rows <- {f}")
        return row_items

    if cache_hit:
        print(f"[cache] hit {cache_key}  ({cache_subdir})")
        cached = pc.load_cache(cache_subdir)
        train_rows = cached["train_rows"]
        val_rows = cached["val_rows"]
        test_rows = cached["test_rows"]
        feature_columns = cached["feature_columns"]
        scaler_bundle = cached["scaler_bundle"]
        scaler = scaler_bundle["scaler"]
        fit_scope = scaler_bundle.get("fit_scope", "unknown")
        # Mirror cached artifacts to this run's --output-derived paths so
        # downstream tooling (inference, archive scripts) sees them.
        pc.copy_artifacts_from_cache(
            cache_subdir,
            scaler_dst=scaler_path,
            manifest_dst=manifest_path,
            stage1_dst=stage1_path,
        )
        print(f"  scaler  → {scaler_path}")
        print(f"  manifest→ {manifest_path}")
        print(f"  stage1  → {stage1_path}")
        print(f"  features: {len(feature_columns)} columns; scaler fit scope: {fit_scope}")

        def _aligned_features(item):  # noqa: ARG001 -- unused on hit path
            raise RuntimeError("_aligned_features should not be called on cache-hit path")
    else:
        if cache_subdir is not None:
            print(f"[cache] miss {cache_key}; will populate at {cache_subdir}")

        train_rows = _load_rows_split(train_files, "train")
        val_rows = _load_rows_split(val_files, "val")
        test_rows = _load_rows_split(test_files, "test")

        if not train_rows:
            raise RuntimeError("No training rows produced — check --data-dir and --primary-type.")

        train_feature_sets = [set(item["features"].columns) for item in train_rows]
        missing_core_by_file = [
            (item["log_path"], sorted(set(core_cols) - set(item["features"].columns)))
            for item in train_rows
            if set(core_cols) - set(item["features"].columns)
        ]
        if missing_core_by_file:
            details = "\n".join(
                f"  {path}: {missing}" for path, missing in missing_core_by_file[:10]
            )
            raise RuntimeError(f"Required core feature columns missing from training data:\n{details}")

        feature_columns = sorted(set().union(*train_feature_sets))
        timestamp_like = [
            c for c in feature_columns if "timestamp" in c.lower() or c.startswith("time_")
        ]
        if timestamp_like:
            raise RuntimeError(f"Timestamp-like columns leaked into features: {timestamp_like}")

        def _aligned_features(item):
            frame = item["features"]
            missing_core = sorted(set(core_cols) - set(frame.columns))
            if missing_core:
                raise RuntimeError(
                    f"{item['log_path']} is missing core feature columns {missing_core}"
                )
            return frame.reindex(columns=feature_columns, fill_value=0).fillna(0)

        fit_rows = train_rows
        fit_scope = "train_rows_only"
        if args.scaler_fit_scope == "train_benign":
            fit_rows = [item for item in train_rows if item["run_label"] == "benign"]
            fit_scope = "train_benign_rows_only"
            if not fit_rows:
                raise RuntimeError(
                    "--scaler-fit-scope train_benign requested, but no benign train rows exist."
                )

        scaler = StandardScaler()
        scaler.fit(
            pd.concat([_aligned_features(item) for item in fit_rows], ignore_index=True).values
        )
        joblib.dump(
            {
                "scaler": scaler,
                "feature_columns": feature_columns,
                "primary_type": args.primary_type,
                "window_size": args.window_size,
                "stride": args.stride,
                "fit_scope": fit_scope,
                "created_by_script": os.path.basename(__file__),
                "feature_count": len(feature_columns),
                "model_output": model_name,
                "split_manifest_path": manifest_path,
            },
            scaler_path,
        )
        print(f"✓ Scaler bundle saved → {scaler_path}")
        print(f"Scaler fit scope: {fit_scope}; feature_count={len(feature_columns)}")

        # Pre-scale every row once — caching captures the scaled tensor so
        # window-sweep children skip both _aligned_features and scaler.transform.
        for items in (train_rows, val_rows, test_rows):
            for item in items:
                aligned = _aligned_features(item)
                item["scaled_features"] = scaler.transform(aligned.values).astype(np.float32)
                # The unscaled DataFrame is now redundant and large — drop it
                # so the cache write stays compact and downstream code can't
                # accidentally re-scale.
                item.pop("features", None)

        if cache_subdir is not None:
            print(f"[cache] populating {cache_subdir}")
            pc.write_cache(
                cache_subdir,
                key=cache_key,
                train_rows=train_rows,
                val_rows=val_rows,
                test_rows=test_rows,
                feature_columns=feature_columns,
                scaler_src=scaler_path,
                manifest_src=manifest_path,
                stage1_src=stage1_path,
            )
            print(f"[cache] populated {cache_subdir}")

    if args.dry_run:
        print("\n--dry-run set; skipping windowing + model fit.")
        sys.exit(0)

    def _load_split(row_items, name):
        windows: list[np.ndarray] = []
        labels: list[int] = []
        metadata: list[dict[str, object]] = []
        n_sims = 0
        print(f"\n── Windowing {name} rows ─────────────────────────────────────────")
        for item in row_items:
            out = make_windows_from_scaled_rows(
                item,
                item["scaled_features"],
                args.window_size,
                args.stride,
            )
            if out is None:
                continue
            ws, ls, meta = out
            windows.extend(ws)
            labels.extend(ls)
            metadata.extend(meta)
            n_sims += 1
            print(f"  Created {len(ws):>5} windows <- {item['log_path']}")
        return windows, labels, metadata, n_sims

    all_train_windows, all_train_labels, _, n_train_sims = _load_split(train_rows, "train")
    all_val_windows, all_val_labels, all_val_metadata, n_val_sims = _load_split(
        val_rows, "val"
    )
    all_test_windows, all_test_labels, all_test_metadata, n_test_sims = _load_split(
        test_rows, "test"
    )

    print(
        f"\nTrain sims (drones): {n_train_sims}  |  "
        f"Val sims: {n_val_sims}  |  Test sims: {n_test_sims}"
    )

    if not all_train_windows:
        raise RuntimeError("No training windows produced — check --data-dir and --primary-type.")

    X_train = np.array(all_train_windows)
    y_train = np.array(all_train_labels)
    X_val = np.array(all_val_windows)
    y_val = np.array(all_val_labels)
    X_test = np.array(all_test_windows) if all_test_windows else np.empty((0, args.window_size, 0))
    y_test = np.array(all_test_labels)

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
        rng = np.random.default_rng(42)
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

    total = len(y_train)
    n_benign = int(np.sum(y_train == 0))
    n_attack = int(np.sum(y_train == 1))
    # Guard against a fully-one-class train set after balancing.
    class_weight = {
        0: total / (2 * n_benign) if n_benign else 1.0,
        1: total / (2 * n_attack) if n_attack else 1.0,
    }
    print(f"Class weights     : {class_weight}")

    seq_len = X_train.shape[1]  # window_size after possible MaxPooling shrinkage
    n_features = X_train.shape[2]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Sequential(
        [
            Input(shape=(seq_len, n_features)),
            Conv1D(32, kernel_size=5, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Bidirectional(
                LSTM(
                    32,
                    return_sequences=True,
                    kernel_regularizer=l2(1e-5),
                    recurrent_regularizer=l2(1e-5),
                )
            ),
            BatchNormalization(),
            Dropout(0.2),
            GlobalAveragePooling1D(),
            Dense(32, activation="relu", kernel_regularizer=l2(1e-5)),
            Dropout(0.2),
            # dtype='float32' keeps the loss in fp32 even when mixed_precision
            # is enabled — avoids underflow in focal_loss.
            Dense(1, activation="sigmoid", dtype="float32"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    callbacks = [
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, mode="min", verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, mode="min", verbose=1
        ),
    ]

    model.fit(
        X_train,
        y_train,
        epochs=60,
        batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # ── Validation evaluation (tuning) ────────────────────────────────────────
    # Use the val set to pick the operating threshold (max F1). Test set
    # results below use this SAME threshold — never re-tuned on test.
    y_pred_prob = model.predict(X_val, verbose=0)
    prec, rec, thresh = precision_recall_curve(y_val, y_pred_prob)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
    best_thresh = thresh[np.argmax(f1)]
    print(f"\nOptimal threshold (from val) : {best_thresh:.4f}")

    y_pred = (y_pred_prob >= best_thresh).astype(int)
    print("\n── Val classification report ─────────────────────────────────────")
    print(classification_report(y_val, y_pred, target_names=["Benign", "Attack"], digits=4))
    cm = confusion_matrix(y_val, y_pred)
    print(f"""Val Confusion Matrix:
      Predicted Benign  Predicted Attack
Actual Benign    {cm[0][0]:<6}           {cm[0][1]:<6}   (False Alarms)
Actual Attack    {cm[1][0]:<6}           {cm[1][1]:<6}   (Missed Attacks)
""")
    print_classification_breakdown("Val", y_val, y_pred, all_val_metadata)
    print_detection_metrics("Val", y_val, y_pred_prob, y_pred, all_val_metadata)

    # ── Held-out test evaluation (final, untouched until now) ─────────────────
    if len(X_test) > 0 and len(np.unique(y_test)) > 1:
        y_test_prob = model.predict(X_test, verbose=0)
        y_test_pred = (y_test_prob >= best_thresh).astype(int)
        print("\n── TEST classification report (held-out runs) ────────────────────")
        print(
            classification_report(y_test, y_test_pred, target_names=["Benign", "Attack"], digits=4)
        )
        cm_t = confusion_matrix(y_test, y_test_pred)
        print(f"""Test Confusion Matrix:
      Predicted Benign  Predicted Attack
Actual Benign    {cm_t[0][0]:<6}           {cm_t[0][1]:<6}   (False Alarms)
Actual Attack    {cm_t[1][0]:<6}           {cm_t[1][1]:<6}   (Missed Attacks)
""")
        print_classification_breakdown("Test", y_test, y_test_pred, all_test_metadata)
        print_detection_metrics("Test", y_test, y_test_prob, y_test_pred, all_test_metadata)
        # Per-run breakdown of where the test drones came from, so a bad
        # test score can be traced back to which run(s) the model misread.
        test_runs_present = sorted({os.path.dirname(os.path.dirname(f)) for f in test_files})
        print("Test runs represented:")
        for r in test_runs_present:
            n_drones = sum(1 for f in test_files if os.path.dirname(os.path.dirname(f)) == r)
            print(f"  {run_labels[r]:<7}  {os.path.basename(r)}  ({n_drones} drone(s))")
    elif len(X_test) == 0:
        print("\n⚠  No test windows produced — test set may have been too small.")
    else:
        print(
            f"\n⚠  Test set has only one class ({Counter(y_test.tolist())}); "
            f"skipping test eval. Consider adjusting --test-frac or run set."
        )

    print(f"\n── Combined system ──")
    print(f"Stage 1 (flat-line rule): see {stage1_path} for live inference")
    print(f"Stage 2 (ML): trained on per-window labels, primary type '{args.primary_type}'")

    model.save(model_name)
    print(f"\n✓ Stage 2 model saved → {model_name}")
    print(f"✓ Stage 1 config saved → {stage1_path}")
