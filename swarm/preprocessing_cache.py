"""Shared preprocessing cache for biLSTM training scripts.

The biLSTM training pipeline does substantial work before windowing:
- Read parquet/CSVs, run row-level feature engineering, deterministic
  train/val/test split (seed=42), fit StandardScaler on training rows.

None of this depends on ``--window-size`` or ``--stride``. Sweeping those
two parameters with this cache lets each child process skip straight to
windowing + model fit.

Cache key inputs (any change busts the cache):
- Set of drone CSV paths under ``--data-dir``, with mtime and size of each.
- ``--primary-type``
- ``--scaler-fit-scope``
- ``--test-frac``, ``--val-frac``
- ``core_cols`` (from ``ATTACK_FEATURE_MAP``)
- Split seed (42, hard-coded)
- Script identity (``1layer`` vs ``2layer``)
- ``CACHE_FORMAT_VERSION`` — bump when preprocessing semantics change.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass

import joblib
import numpy as np


CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheKeyInputs:
    drone_files: tuple[str, ...]
    primary_type: str
    scaler_fit_scope: str
    test_frac: float
    val_frac: float
    core_cols: tuple[str, ...]
    split_seed: int
    script_identity: str


def compute_cache_key(inputs: CacheKeyInputs) -> str:
    """Stable sha256 hash of the inputs that determine cache content.

    File-content fingerprint is ``(path, mtime_ns, size)``; this is
    pragmatic — a file rewritten with the same mtime + size won't bust the
    cache. For training data that is write-once-then-read this is fine.
    """
    h = hashlib.sha256()
    h.update(f"v{CACHE_FORMAT_VERSION}\n".encode())
    h.update(f"identity={inputs.script_identity}\n".encode())
    h.update(f"primary_type={inputs.primary_type}\n".encode())
    h.update(f"scaler_fit_scope={inputs.scaler_fit_scope}\n".encode())
    h.update(f"test_frac={inputs.test_frac}\n".encode())
    h.update(f"val_frac={inputs.val_frac}\n".encode())
    h.update(f"core_cols={','.join(inputs.core_cols)}\n".encode())
    h.update(f"split_seed={inputs.split_seed}\n".encode())
    for path in inputs.drone_files:
        st = os.stat(path)
        h.update(f"{path}|{st.st_mtime_ns}|{st.st_size}\n".encode())
    return h.hexdigest()[:16]


def _row_items_path(cache_dir: str, split: str) -> str:
    return os.path.join(cache_dir, f"{split}_rows.joblib")


def cache_exists(cache_dir: str) -> bool:
    return os.path.exists(os.path.join(cache_dir, "manifest.json"))


def load_cache(
    cache_dir: str,
) -> dict[str, object]:
    """Load the cached preprocessing outputs.

    Returns a dict with keys: ``train_rows``, ``val_rows``, ``test_rows``,
    ``feature_columns``, ``scaler_bundle``, ``manifest``, ``has_stage1``,
    and the on-disk paths to the scaler/manifest/stage1 files in the cache.
    Each row item has a ``scaled_features`` ndarray already.
    """
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(cache_dir, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    scaler_path = os.path.join(cache_dir, "scaler.joblib")
    scaler_bundle = joblib.load(scaler_path)
    train_rows = joblib.load(_row_items_path(cache_dir, "train"))
    val_rows = joblib.load(_row_items_path(cache_dir, "val"))
    test_rows = joblib.load(_row_items_path(cache_dir, "test"))
    stage1_src = os.path.join(cache_dir, "stage1.pkl")
    has_stage1 = os.path.exists(stage1_src)
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "feature_columns": feature_columns,
        "scaler_bundle": scaler_bundle,
        "manifest": manifest,
        "scaler_src": scaler_path,
        "manifest_src": os.path.join(cache_dir, "manifest.json"),
        "stage1_src": stage1_src if has_stage1 else None,
        "has_stage1": has_stage1,
    }


def write_cache(
    cache_dir: str,
    *,
    key: str,
    train_rows: list[dict[str, object]],
    val_rows: list[dict[str, object]],
    test_rows: list[dict[str, object]],
    feature_columns: list[str],
    scaler_src: str,
    manifest_src: str,
    stage1_src: str | None,
) -> None:
    """Write preprocessing artifacts to the cache.

    ``*_rows`` items must already contain a ``scaled_features`` ndarray —
    scaling is part of the cache so child sweep processes skip it.
    The on-disk scaler/manifest/stage1 are *copied* (not moved) so the
    originating run's artifacts stay intact.
    """
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)
    shutil.copy2(scaler_src, os.path.join(cache_dir, "scaler.joblib"))
    shutil.copy2(manifest_src, os.path.join(cache_dir, "manifest.json"))
    if stage1_src is not None:
        shutil.copy2(stage1_src, os.path.join(cache_dir, "stage1.pkl"))

    joblib.dump(_serialize_row_items(train_rows), _row_items_path(cache_dir, "train"))
    joblib.dump(_serialize_row_items(val_rows), _row_items_path(cache_dir, "val"))
    joblib.dump(_serialize_row_items(test_rows), _row_items_path(cache_dir, "test"))

    with open(os.path.join(cache_dir, "_cache_meta.json"), "w") as f:
        json.dump(
            {
                "cache_key": key,
                "format_version": CACHE_FORMAT_VERSION,
                "n_train_files": len(train_rows),
                "n_val_files": len(val_rows),
                "n_test_files": len(test_rows),
            },
            f,
            indent=2,
        )


def _serialize_row_items(items: list[dict[str, object]]) -> list[dict[str, object]]:
    """Strip non-essential fields and downcast arrays for compact storage."""
    out: list[dict[str, object]] = []
    for it in items:
        scaled = np.asarray(it["scaled_features"], dtype=np.float32)
        out.append(
            {
                "scaled_features": scaled,
                "row_labels": np.asarray(it["row_labels"], dtype=np.int64),
                "timestamps": np.asarray(it["timestamps"]),
                "run_dir": it["run_dir"],
                "drone_file": it["drone_file"],
                "run_label": it["run_label"],
                "log_path": it["log_path"],
            }
        )
    return out


def copy_artifacts_from_cache(
    cache_dir: str,
    *,
    scaler_dst: str,
    manifest_dst: str,
    stage1_dst: str | None,
) -> None:
    """On cache hit, mirror cached scaler/manifest/stage1 to the run's --output paths.

    This keeps downstream tooling that reads the per-run scaler bundle, split
    manifest, or stage1 config working unchanged — those files live where they
    would have if preprocessing had run from scratch.
    """
    shutil.copy2(os.path.join(cache_dir, "scaler.joblib"), scaler_dst)
    shutil.copy2(os.path.join(cache_dir, "manifest.json"), manifest_dst)
    cached_stage1 = os.path.join(cache_dir, "stage1.pkl")
    if stage1_dst is not None and os.path.exists(cached_stage1):
        shutil.copy2(cached_stage1, stage1_dst)
