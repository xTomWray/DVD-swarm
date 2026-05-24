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
    primary: str,
    core_cols: list[str],
    window_size: int,
    stride: int,
) -> dict[str, Any]:
    """Window every drone in one run, predict, return the raw arrays."""
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
        scaled = scaler.transform(rows["features"].values)
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


def _per_run_metrics(result: dict[str, Any], threshold: float) -> dict[str, Any]:
    """Compute JSON-friendly per-run metrics (no big arrays)."""
    base: dict[str, Any] = {
        "name": Path(result["run_dir"]).name,
        "run_label": result["run_label"],
        "drones": result["drones"],
        "skipped": result["skipped"],
        "windows": result["windows"],
    }
    if result["windows"] == 0:
        return base
    import numpy as np
    y_true = result["y_true"]
    y_prob = result["y_prob"]
    y_pred = (y_prob >= threshold).astype(int)
    base.update({
        "predicted_positive_rate": float(y_pred.mean()),
        "prob_mean": float(y_prob.mean()),
        "prob_p95": float(np.percentile(y_prob, 95)),
        "prob_max": float(y_prob.max()),
    })
    if result["run_label"] == "benign":
        fp = int(y_pred.sum())
        base.update({"false_positives": fp,
                     "false_positive_rate": fp / result["windows"]})
    else:
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        base.update({"tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1})
    return base


def _print_per_run(m: dict[str, Any]) -> None:
    name = m["name"]
    if m["windows"] == 0:
        print(f"  {name:50s}  EMPTY (drones={m['drones']}, skipped={m['skipped']})")
        return
    head = (
        f"  {name:50s}  drones={m['drones']:>3d}  windows={m['windows']:>9,d}  "
        f"pred+={m['predicted_positive_rate']:6.2%}  "
        f"prob(mean/p95/max)={m['prob_mean']:.3f}/{m['prob_p95']:.3f}/{m['prob_max']:.3f}"
    )
    if m["run_label"] == "benign":
        print(f"{head}  FP={m['false_positives']:>7,d}  FPR={m['false_positive_rate']:.3%}")
    else:
        print(f"{head}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")


def _print_aggregate(per_run: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    benign = [m for m in per_run if m["run_label"] == "benign" and m["windows"] > 0]
    attack = [m for m in per_run if m["run_label"] == "attack" and m["windows"] > 0]
    total_windows = sum(m["windows"] for m in per_run)
    print(f"\n=== Aggregate @ threshold={threshold:.2f} ===")
    print(f"  runs:    benign={len(benign)}  attack={len(attack)}  "
          f"total windows={total_windows:,}")
    agg: dict[str, Any] = {
        "threshold": threshold, "total_windows": total_windows,
        "n_benign_runs": len(benign), "n_attack_runs": len(attack),
    }
    if benign:
        b_windows = sum(m["windows"] for m in benign)
        b_fp = sum(m["false_positives"] for m in benign)
        fpr = b_fp / b_windows if b_windows else 0.0
        print(f"  benign:  windows={b_windows:>9,d}  FP={b_fp:>7,d}  FPR={fpr:.3%}")
        agg.update({"benign_windows": b_windows, "false_positives": b_fp,
                    "false_positive_rate": fpr})
    if attack:
        a_tp = sum(m["tp"] for m in attack)
        a_fp = sum(m["fp"] for m in attack)
        a_fn = sum(m["fn"] for m in attack)
        p = a_tp / max(a_tp + a_fp, 1)
        r = a_tp / max(a_tp + a_fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"  attack:  TP={a_tp:,}  FP={a_fp:,}  FN={a_fn:,}  "
              f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
        agg.update({"tp": a_tp, "fp": a_fp, "fn": a_fn,
                    "precision": p, "recall": r, "f1": f1})
    return agg


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
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Probability threshold for 0/1 decision (default 0.5).")
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
    primary = manifest["primary_type"]
    manifest_window = int(manifest["window_size"])
    manifest_stride = int(manifest["stride"])

    # CLI overrides win over the manifest (manifest sidecars get overwritten
    # when users train multiple variants with the same --output stem).
    window_size = args.window_size if args.window_size is not None else manifest_window
    stride = args.stride if args.stride is not None else manifest_stride

    # Cross-check the model filename for a `_w<N>_s<N>` hint and warn loudly
    # if the values we're about to use disagree — most common cause of
    # "results look wrong but I can't tell why".
    fn_match = _FILENAME_WS_RE.search(keras_path.stem)
    if fn_match:
        fn_w, fn_s = int(fn_match.group(1)), int(fn_match.group(2))
        if (fn_w, fn_s) != (window_size, stride):
            print(
                f"WARNING: model filename suggests window={fn_w} stride={fn_s}, "
                f"but using window={window_size} stride={stride} "
                f"(manifest: window={manifest_window} stride={manifest_stride}"
                f"{', overridden by CLI' if args.window_size or args.stride else ''}). "
                f"Pass --window-size {fn_w} --stride {fn_s} if the filename is right.",
                file=sys.stderr,
            )

    print(f"model    : {keras_path}")
    print(f"scaler   : {scaler_path.name}")
    print(f"manifest : {manifest_path.name}")
    print(f"primary  : {primary}  window={window_size}  stride={stride}"
          f"{f'  (manifest had window={manifest_window} stride={manifest_stride})' if (window_size, stride) != (manifest_window, manifest_stride) else ''}")

    # Heavy imports now that we know we need them.
    bilstm = _load_trainer_module()
    import joblib
    import tensorflow as tf

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

    model = tf.keras.models.load_model(str(keras_path), compile=False)
    scaler = joblib.load(scaler_path)

    per_run: list[dict[str, Any]] = []
    for run_dir in sorted(runs):
        run_label = bilstm._run_label_from_metadata(run_dir)
        if run_label is None:
            print(f"  {Path(run_dir).name:50s}  SKIP (no metadata.json)")
            continue
        result = _evaluate_run(
            run_dir, runs[run_dir], run_label,
            bilstm=bilstm, model=model, scaler=scaler, primary=primary,
            core_cols=core_cols, window_size=window_size, stride=stride,
        )
        m = _per_run_metrics(result, args.threshold)
        per_run.append(m)
        _print_per_run(m)
        if args.verbose_detection and result["windows"] > 0:
            y_pred = (result["y_prob"] >= args.threshold).astype(int)
            bilstm.print_detection_metrics(
                Path(run_dir).name,
                result["y_true"], result["y_prob"], y_pred, result["meta"],
            )

    aggregate = _print_aggregate(per_run, args.threshold)

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
            "threshold": args.threshold,
            "per_run": per_run,
            "aggregate": aggregate,
        }, indent=2, default=str))
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
