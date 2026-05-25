"""Score a trained biLSTM model against arbitrary held-out runs.

Loads a saved ``*.keras`` model plus its ``*_scaler.joblib`` and
``*_split_manifest.json`` sidecars, then runs the same preprocessing the
trainer used (imported from ``swarm/biLSTM-1Layer-Protocol.py``) on every
``csv/drone_*.csv`` under ``--data-dir`` and prints per-run + aggregate
metrics.

Initial scope: 1Layer biLSTM models. The 2Layer trainer's
``load_preprocessed_rows`` filters attack rows for attack runs and the
pipeline adds a stage-1 sklearn model, so it needs its own glue — deferred.

Usage:
    python -m swarm.dataset.evaluate \\
        --model batch512_1layer \\
        --data-dir training-data/benign/run_legacy-nominal-1_none_legacy

    python -m swarm.dataset.evaluate \\
        --model batch512_1layer/aeroshield_ATTITUDE.keras \\
        --data-dir training-data/benign \\
        --threshold 0.7 --json-out /tmp/eval.json
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


# Recognize the `_w<window>_s<stride>` convention some users encode into
# model filenames (e.g. ``aeroshield_1layer_w160_s10.keras``). Used purely
# as a cross-check against the manifest values — never silently overrides.
_FILENAME_WS_RE = re.compile(r"_w(\d+)_s(\d+)", re.IGNORECASE)

# Heavy imports (numpy, joblib, tensorflow, the trainer module) are deferred
# into ``main`` so ``--help`` is fast and module-import works on hosts that
# only have the stdlib.

_TRAINER_PATH = Path(__file__).resolve().parents[1] / "biLSTM-1Layer-Protocol.py"


def _load_trainer_module() -> Any:
    """Load the 1Layer trainer as a module (its filename has dashes, so a
    plain ``from … import`` doesn't work).
    """
    spec = importlib.util.spec_from_file_location("bilstm_1layer", _TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load trainer module from {_TRAINER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolve_artifacts(model_path: Path) -> tuple[Path, Path, Path]:
    """Given a ``.keras`` file or a dir containing exactly one, return
    ``(keras, scaler, manifest)`` paths. Sidecars derive from the keras stem.
    """
    if model_path.is_dir():
        candidates = sorted(model_path.glob("*.keras"))
        if not candidates:
            raise FileNotFoundError(f"no *.keras in {model_path}")
        if len(candidates) > 1:
            names = ", ".join(p.name for p in candidates)
            raise FileNotFoundError(
                f"multiple *.keras in {model_path}: {names} — "
                f"pass --model {candidates[0]} explicitly"
            )
        keras = candidates[0]
    elif model_path.is_file() and model_path.suffix == ".keras":
        keras = model_path
    else:
        raise FileNotFoundError(
            f"--model must be a .keras file or a directory containing one: {model_path}"
        )
    stem_path = keras.with_suffix("")
    scaler = stem_path.with_name(stem_path.name + "_scaler.joblib")
    manifest = stem_path.with_name(stem_path.name + "_split_manifest.json")
    if not scaler.exists():
        raise FileNotFoundError(f"scaler sidecar not found: {scaler}")
    if not manifest.exists():
        raise FileNotFoundError(f"manifest sidecar not found: {manifest}")
    return keras, scaler, manifest


def _discover_runs(data_dir: Path, run_filter: list[str] | None) -> dict[str, list[Path]]:
    """Mirror the trainer's discovery: glob ``**/csv/drone_*.csv`` and group
    by parent of ``csv/``. Optional ``run_filter`` matches on the run dir's
    basename.
    """
    drone_files = sorted(
        Path(p) for p in glob.glob(
            str(data_dir / "**" / "csv" / "drone_*.csv"), recursive=True,
        )
    )
    if not drone_files:
        raise FileNotFoundError(f"no csv/drone_*.csv files found under {data_dir}")
    runs: dict[str, list[Path]] = {}
    for f in drone_files:
        # .../run_X/csv/drone_NNN.csv  →  .../run_X
        run_dir = str(f.parent.parent)
        if run_filter and Path(run_dir).name not in run_filter:
            continue
        runs.setdefault(run_dir, []).append(f)
    return runs


def _evaluate_run(
    run_dir: str,
    drone_files: list[Path],
    run_label: str,
    *,
    bilstm: Any,
    model: Any,
    scaler: Any,
    feature_columns: list[str] | None,
    primary: str,
    core_cols: list[str],
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    """Window every drone in one run, predict, return the raw arrays.

    Mirrors the trainer's ``_aligned_features``: reindexes each drone's
    feature frame to the column order the scaler was fit on, filling
    missing columns with 0 (matches training-time behavior). Without
    this step, ``scaler.transform`` silently scales the wrong columns
    whenever a drone happens to have fewer/extra engineered features.
    """
    import numpy as np
    Xs: list[Any] = []
    ys: list[int] = []
    meta: list[dict[str, Any]] = []
    skipped = 0
    for f in drone_files:
        rows = bilstm.load_preprocessed_rows(
            str(f), primary, core_cols, run_label=run_label,
        )
        if rows is None:
            skipped += 1
            continue
        frame = rows["features"]
        if feature_columns is not None:
            frame = frame.reindex(columns=feature_columns, fill_value=0).fillna(0)
        scaled = scaler.transform(frame.values)
        wlm = bilstm.make_windows_from_scaled_rows(rows, scaled, window_size, stride)
        if wlm is None:
            skipped += 1
            continue
        w, l, m = wlm
        Xs.extend(w)
        ys.extend(l)
        meta.extend(m)
    if not Xs:
        return {
            "run_dir": run_dir, "run_label": run_label,
            "drones": len(drone_files), "skipped": skipped, "windows": 0,
        }
    X = np.stack(Xs)
    y_true = np.asarray(ys, dtype=np.int64)
    y_prob = model.predict(X, verbose=0).ravel()
    return {
        "run_dir": run_dir, "run_label": run_label,
        "drones": len(drone_files), "skipped": skipped, "windows": int(len(X)),
        "y_true": y_true, "y_prob": y_prob, "meta": meta,
    }


def _summarize_probs(y_prob: Any) -> dict[str, float]:
    """min/mean/p95/p99/max so users see both bulk and tail behavior."""
    import numpy as np
    return {
        "min": float(y_prob.min()),
        "mean": float(y_prob.mean()),
        "p95": float(np.percentile(y_prob, 95)),
        "p99": float(np.percentile(y_prob, 99)),
        "max": float(y_prob.max()),
    }


def _confusion(y_true: Any, y_pred: Any) -> dict[str, int]:
    """Full 2x2 confusion matrix as a dict — TP/FP/FN/TN integers."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _precision_recall_f1(cm: dict[str, int]) -> tuple[float, float, float]:
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def _infer_window_from_model(model: Any) -> int | None:
    """Extract window_size from the Keras model's input shape.

    Bidirectional LSTM input is ``(batch, timesteps, features)``; we want
    ``timesteps`` (index 1). Returns ``None`` if the shape is unparseable
    (e.g. nested model, dict-shaped input).
    """
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, tuple) and len(shape) >= 2 and isinstance(shape[1], int):
        return int(shape[1])
    return None


def _flight_hours(meta_list: list[dict[str, Any]]) -> float:
    """Per-drone (max end-ts − min start-ts) summed, converted ms→hours.

    Each drone's windows are causal in time, so the timespan covered by
    its windows is a sensible proxy for that drone's flight duration.
    Returns 0.0 if no metadata or no positive spans.
    """
    if not meta_list:
        return 0.0
    spans: dict[tuple[str, str], list[float]] = {}
    for m in meta_list:
        key = (m["run_dir"], m["drone_file"])
        rng = spans.setdefault(key, [float("inf"), float("-inf")])
        rng[0] = min(rng[0], float(m["window_start_timestamp"]))
        rng[1] = max(rng[1], float(m["window_end_timestamp"]))
    total_ms = sum(hi - lo for lo, hi in spans.values() if hi > lo)
    return total_ms / 1000.0 / 3600.0


def _per_run_metrics(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """JSON-friendly per-run summary at one threshold."""
    base: dict[str, Any] = {
        "name": Path(result["run_dir"]).name,
        "run_label": result["run_label"],
        "drones": result["drones"],
        "skipped": result["skipped"],
        "windows": result["windows"],
    }
    if result["windows"] == 0:
        return base
    y_true = result["y_true"]
    y_prob = result["y_prob"]
    y_pred = (y_prob >= threshold).astype(int)
    cm = _confusion(y_true, y_pred)
    p, r, f1 = _precision_recall_f1(cm)
    probs = _summarize_probs(y_prob)
    flight_h = _flight_hours(result.get("meta", []))
    base.update({
        "threshold": threshold,
        "predicted_positive_rate": float(y_pred.mean()),
        "flight_hours": flight_h,
        **cm,
        "precision": p, "recall": r, "f1": f1,
        **{f"prob_{k}": v for k, v in probs.items()},
    })
    if result["run_label"] == "benign":
        base["false_positives"] = cm["fp"]
        base["false_positive_rate"] = cm["fp"] / result["windows"]
    return base


def _print_per_run(m: dict[str, Any]) -> None:
    name = m["name"]
    if m["windows"] == 0:
        print(f"  {name:50s}  EMPTY (drones={m['drones']}, skipped={m['skipped']})")
        return
    head = (
        f"  {name:50s}  drones={m['drones']:>3d}  windows={m['windows']:>9,d}  "
        f"pred+={m['predicted_positive_rate']:6.2%}  "
        f"prob(mean/p95/p99/max)="
        f"{m['prob_mean']:.3f}/{m['prob_p95']:.3f}/{m['prob_p99']:.3f}/{m['prob_max']:.3f}"
    )
    if m["run_label"] == "benign":
        print(f"{head}  FP={m['fp']:>7,d}  FPR={m['false_positive_rate']:.3%}")
    else:
        print(f"{head}  TP={m['tp']:,} FP={m['fp']:,} FN={m['fn']:,} TN={m['tn']:,}"
              f"  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")


def _aggregate_at_threshold(
    benign_results: list[dict[str, Any]],
    attack_results: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    """Aggregate metrics at one threshold, computed from raw y_prob arrays
    (not summed per-run, which would round-trip through stored ints).
    """
    import numpy as np
    agg: dict[str, Any] = {"threshold": threshold}
    if benign_results:
        y_prob = np.concatenate([r["y_prob"] for r in benign_results])
        y_true = np.concatenate([r["y_true"] for r in benign_results])
        y_pred = (y_prob >= threshold).astype(int)
        cm = _confusion(y_true, y_pred)
        n_windows = len(y_prob)
        n_drones = sum(r["drones"] for r in benign_results)
        hours = sum(_flight_hours(r.get("meta", [])) for r in benign_results)
        agg["benign"] = {
            "windows": n_windows,
            "drones": n_drones,
            "flight_hours": hours,
            **cm,
            "fpr": cm["fp"] / n_windows if n_windows else 0.0,
            "fp_per_10k_windows": (cm["fp"] / n_windows * 1e4) if n_windows else 0.0,
            "fp_per_drone": cm["fp"] / n_drones if n_drones else 0.0,
            "fp_per_flight_hour": (cm["fp"] / hours) if hours else None,
            **{f"prob_{k}": v for k, v in _summarize_probs(y_prob).items()},
        }
    if attack_results:
        y_prob = np.concatenate([r["y_prob"] for r in attack_results])
        y_true = np.concatenate([r["y_true"] for r in attack_results])
        y_pred = (y_prob >= threshold).astype(int)
        cm = _confusion(y_true, y_pred)
        p, r, f1 = _precision_recall_f1(cm)
        agg["attack"] = {
            "windows": len(y_prob),
            **cm,
            "precision": p, "recall": r, "f1": f1,
            **{f"prob_{k}": v for k, v in _summarize_probs(y_prob).items()},
        }
    return agg


def _print_aggregate_single(agg: dict[str, Any]) -> None:
    t = agg["threshold"]
    print(f"\n=== Aggregate @ threshold={t:.4f} ===")
    if "benign" in agg:
        b = agg["benign"]
        print(f"  benign:  windows={b['windows']:>9,d}  drones={b['drones']:>3d}  "
              f"flight={b['flight_hours']:.2f}h")
        print(f"    TN={b['tn']:>9,d}  FP={b['fp']:>7,d}  "
              f"FPR={b['fpr']:.3%}  "
              f"FP/10k={b['fp_per_10k_windows']:.2f}  "
              f"FP/drone={b['fp_per_drone']:.2f}"
              + (f"  FP/hr={b['fp_per_flight_hour']:.3f}"
                 if b['fp_per_flight_hour'] is not None else ""))
        print(f"    prob mean/p95/p99/max="
              f"{b['prob_mean']:.4f}/{b['prob_p95']:.4f}/{b['prob_p99']:.4f}/{b['prob_max']:.4f}")
    if "attack" in agg:
        a = agg["attack"]
        print(f"  attack:  windows={a['windows']:>9,d}")
        print(f"    TP={a['tp']:>9,d}  FP={a['fp']:>7,d}  "
              f"FN={a['fn']:>7,d}  TN={a['tn']:>7,d}")
        print(f"    P={a['precision']:.4f}  R={a['recall']:.4f}  F1={a['f1']:.4f}")
        print(f"    prob mean/p95/p99/max="
              f"{a['prob_mean']:.4f}/{a['prob_p95']:.4f}/{a['prob_p99']:.4f}/{a['prob_max']:.4f}")


def _print_threshold_sweep(
    benign_results: list[dict[str, Any]],
    attack_results: list[dict[str, Any]],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    """Print compact per-threshold table. Returns the rows for JSON output."""
    rows = [_aggregate_at_threshold(benign_results, attack_results, t) for t in thresholds]
    if benign_results:
        b0 = rows[0]["benign"]
        print(f"\n=== Benign threshold sweep "
              f"({b0['windows']:,} windows, {b0['drones']} drones, "
              f"{b0['flight_hours']:.2f} flight-hours) ===")
        print(f"  {'Threshold':>10s} |{'FP':>10s} {'FPR':>9s} {'FP/10k':>10s} "
              f"{'FP/drone':>10s} {'FP/hr':>9s} | "
              f"{'prob mean':>10s} {'p95':>8s} {'p99':>8s} {'max':>8s}")
        print(f"  {'-'*10}-+{'-'*43}-+{'-'*36}")
        for row in rows:
            b = row["benign"]
            fphr = f"{b['fp_per_flight_hour']:>9.3f}" if b['fp_per_flight_hour'] is not None else f"{'n/a':>9s}"
            print(f"  {row['threshold']:>10.4f} |{b['fp']:>10,d} {b['fpr']:>8.3%} "
                  f"{b['fp_per_10k_windows']:>10.2f} {b['fp_per_drone']:>10.2f} {fphr} | "
                  f"{b['prob_mean']:>10.4f} {b['prob_p95']:>8.4f} "
                  f"{b['prob_p99']:>8.4f} {b['prob_max']:>8.4f}")
    if attack_results:
        a0 = rows[0]["attack"]
        print(f"\n=== Attack threshold sweep ({a0['windows']:,} windows) ===")
        print(f"  {'Threshold':>10s} |{'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s} | "
              f"{'P':>7s} {'R':>7s} {'F1':>7s}")
        print(f"  {'-'*10}-+{'-'*36}-+{'-'*24}")
        for row in rows:
            a = row["attack"]
            print(f"  {row['threshold']:>10.4f} |{a['tp']:>8,d} {a['fp']:>8,d} "
                  f"{a['fn']:>8,d} {a['tn']:>8,d} | "
                  f"{a['precision']:>7.4f} {a['recall']:>7.4f} {a['f1']:>7.4f}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, type=Path,
                    help="Path to *.keras or to a directory containing one (plus its "
                         "*_scaler.joblib and *_split_manifest.json sidecars).")
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="Root dir. Every **/csv/drone_*.csv under it is evaluated; "
                         "each run's class label comes from its metadata.json:attack.")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Single probability threshold for 0/1 decision (default 0.5). "
                         "Backward-compat alias for `--thresholds <one value>`.")
    ap.add_argument("--thresholds", type=float, nargs="+", default=None,
                    help="One or more decision thresholds. Pass multiple values to "
                         "get a per-threshold sweep table with FP, FPR, FP/10k, "
                         "FP/drone, and FP/flight-hour columns. Example: "
                         "`--thresholds 0.05 0.10 0.25 0.50 0.75 0.95`.")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Optional whitelist of run-dir basenames to evaluate.")
    ap.add_argument("--json-out", type=Path, default=None,
                    help="If set, also dump a machine-readable JSON summary here.")
    ap.add_argument("--verbose-detection", action="store_true",
                    help="Also call the trainer's print_detection_metrics per run "
                         "(verbose: per-drone timing + FN streak stats).")
    ap.add_argument("--window-size", type=int, default=None,
                    help="Override the manifest's window_size. Useful when the "
                         "manifest was overwritten by a different training run "
                         "(common when multiple --output variants share a stem).")
    ap.add_argument("--stride", type=int, default=None,
                    help="Override the manifest's stride. See --window-size.")
    args = ap.parse_args(argv)

    try:
        keras_path, scaler_path, manifest_path = _resolve_artifacts(args.model.resolve())
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text())
    manifest_primary = manifest.get("primary_type")
    manifest_window = int(manifest["window_size"]) if "window_size" in manifest else None
    manifest_stride = int(manifest["stride"]) if "stride" in manifest else None

    # Heavy imports now that we know we need them.
    bilstm = _load_trainer_module()
    import joblib
    import tensorflow as tf

    # The trainer saves a bundle dict to *_scaler.joblib that includes the
    # actual StandardScaler PLUS authoritative copies of primary_type,
    # window_size, stride, and feature_columns. The bundle is the source of
    # truth: it's written atomically with the model, so it can't drift the
    # way the manifest can (manifests share a stem with the .keras and get
    # overwritten by later runs with the same --output).
    scaler_blob = joblib.load(scaler_path)
    if isinstance(scaler_blob, dict) and "scaler" in scaler_blob:
        scaler = scaler_blob["scaler"]
        bundle_primary = scaler_blob.get("primary_type")
        bundle_window = scaler_blob.get("window_size")
        bundle_stride = scaler_blob.get("stride")
        feature_columns = scaler_blob.get("feature_columns")
    else:
        # Older format: bare scaler, no metadata bundle.
        scaler = scaler_blob
        bundle_primary = bundle_window = bundle_stride = feature_columns = None

    # Precedence: CLI override → bundle → manifest.
    primary = bundle_primary or manifest_primary
    window_size = (
        args.window_size if args.window_size is not None
        else bundle_window if bundle_window is not None
        else manifest_window
    )
    stride = (
        args.stride if args.stride is not None
        else bundle_stride if bundle_stride is not None
        else manifest_stride
    )
    if primary is None or window_size is None or stride is None:
        print("error: could not determine primary_type/window_size/stride from "
              "bundle, manifest, or CLI overrides", file=sys.stderr)
        return 1

    # Warn if the manifest disagrees with the bundle — common after a stem
    # collision overwrote the manifest. Bundle wins; this is informational.
    if (bundle_window is not None and manifest_window is not None
            and bundle_window != manifest_window):
        print(f"NOTE: bundle says window={bundle_window} but manifest says "
              f"window={manifest_window} — using bundle (the manifest was "
              f"likely overwritten by a different training run).",
              file=sys.stderr)

    # Cross-check the model filename for a `_w<N>_s<N>` hint and warn loudly
    # if the values we're about to use disagree.
    fn_match = _FILENAME_WS_RE.search(keras_path.stem)
    if fn_match:
        fn_w, fn_s = int(fn_match.group(1)), int(fn_match.group(2))
        if (fn_w, fn_s) != (window_size, stride):
            print(
                f"WARNING: model filename suggests window={fn_w} stride={fn_s}, "
                f"but using window={window_size} stride={stride}. "
                f"Pass --window-size {fn_w} --stride {fn_s} if the filename is right.",
                file=sys.stderr,
            )

    print(f"model    : {keras_path}")
    print(f"scaler   : {scaler_path.name}")
    print(f"manifest : {manifest_path.name}")
    src = ("CLI" if (args.window_size or args.stride) else
           "bundle" if (bundle_window or bundle_stride) else "manifest")
    print(f"primary  : {primary}  window={window_size}  stride={stride}  [source: {src}]")
    if feature_columns is not None:
        print(f"features : {len(feature_columns)} columns (from scaler bundle)")

    core_cols = bilstm.get_feature_cols(primary + ".csv")
    if core_cols is None:
        print(f"error: no feature map for primary type {primary!r}", file=sys.stderr)
        return 1

    data_dir = args.data_dir.resolve()
    try:
        runs = _discover_runs(data_dir, args.runs)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not runs:
        avail = sorted({Path(p).parent.parent.name
                        for p in glob.glob(str(data_dir / "**" / "csv" / "drone_*.csv"),
                                           recursive=True)})
        print(
            f"error: --runs {args.runs} matched 0 of {len(avail)} available run(s) "
            f"under {data_dir}.\n       available: {', '.join(avail)}",
            file=sys.stderr,
        )
        return 1
    total_files = sum(len(v) for v in runs.values())
    print(f"\nDiscovered {total_files} drone file(s) across {len(runs)} run(s) "
          f"under {data_dir}\n")

    # Resolve thresholds (CLI: --thresholds list wins; --threshold single is a
    # backward-compat alias; default is [0.5]).
    if args.thresholds is not None:
        thresholds = list(args.thresholds)
    elif args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [0.5]
    primary_threshold = thresholds[0]
    sweep_mode = len(thresholds) > 1

    model = tf.keras.models.load_model(str(keras_path), compile=False)

    # The Keras model itself is the source of truth for window_size: the
    # first hidden dim of input_shape is what every window must be reshaped
    # to. If it disagrees with bundle/manifest/CLI, the model is right —
    # the bundle/manifest got clobbered by a later training run sharing
    # the same --output stem. Without this check, an incorrect window_size
    # would either OOM on reshape or feed nonsense rows in.
    model_window = _infer_window_from_model(model)
    if model_window is not None and model_window != window_size:
        cli_override = args.window_size is not None
        if cli_override:
            print(
                f"WARNING: model expects window={model_window} but CLI passed "
                f"--window-size {window_size} — keeping CLI value (you'll get "
                f"a shape mismatch from Keras if this is wrong).",
                file=sys.stderr,
            )
        else:
            print(
                f"NOTE: model's input shape says window={model_window}; "
                f"overriding window_size={window_size} from {src} "
                f"(bundle/manifest was clobbered by a later training run).",
                file=sys.stderr,
            )
            window_size = model_window
            # If the filename's w<N>_s<N> also matches the model, trust its
            # stride too (more likely correct than the clobbered sidecar).
            if fn_match:
                fn_w, fn_s = int(fn_match.group(1)), int(fn_match.group(2))
                if fn_w == model_window and fn_s != stride:
                    print(
                        f"NOTE: filename agrees with model on window={fn_w}; "
                        f"also trusting filename's stride={fn_s} over "
                        f"{src}'s stride={stride}.",
                        file=sys.stderr,
                    )
                    stride = fn_s
            print(f"           → using window={window_size} stride={stride}",
                  file=sys.stderr)

    # Run every run; keep raw arrays so we can re-score at any threshold.
    raw_results: list[dict[str, Any]] = []
    per_run: list[dict[str, Any]] = []
    for run_dir in sorted(runs):
        run_label = bilstm._run_label_from_metadata(run_dir)
        if run_label is None:
            print(f"  {Path(run_dir).name:50s}  SKIP (no metadata.json)")
            continue
        result = _evaluate_run(
            run_dir, runs[run_dir], run_label,
            bilstm=bilstm, model=model, scaler=scaler,
            feature_columns=feature_columns, primary=primary,
            core_cols=core_cols, window_size=window_size, stride=stride,
        )
        raw_results.append(result)
        m = _per_run_metrics(result, primary_threshold)
        per_run.append(m)
        _print_per_run(m)
        if args.verbose_detection and result["windows"] > 0:
            y_pred = (result["y_prob"] >= primary_threshold).astype(int)
            bilstm.print_detection_metrics(
                Path(run_dir).name,
                result["y_true"], result["y_prob"], y_pred, result["meta"],
            )

    benign_results = [r for r in raw_results if r["run_label"] == "benign" and r["windows"] > 0]
    attack_results = [r for r in raw_results if r["run_label"] == "attack" and r["windows"] > 0]

    primary_agg = _aggregate_at_threshold(benign_results, attack_results, primary_threshold)
    _print_aggregate_single(primary_agg)

    sweep_rows: list[dict[str, Any]] = []
    if sweep_mode:
        sweep_rows = _print_threshold_sweep(benign_results, attack_results, thresholds)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({
            "model": str(keras_path),
            "scaler": str(scaler_path),
            "manifest": str(manifest_path),
            "data_dir": str(data_dir),
            "primary_type": primary,
            "window_size": window_size,
            "stride": stride,
            "thresholds": thresholds,
            "primary_threshold": primary_threshold,
            "per_run": per_run,
            "aggregate_primary": primary_agg,
            "threshold_sweep": sweep_rows,
        }, indent=2, default=str))
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
