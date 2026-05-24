"""Hyperparameter-sweep launcher for biLSTM training.

Runs window-size/stride sweeps in parallel across GPUs. The cache module
(``preprocessing_cache.py``) ensures the slow CSV→scaler stages happen exactly
once: this launcher warms the cache on CPU first, then dispatches one training
child per GPU.

Typical usage:

    python swarm/sweep_train.py --config sweep.yaml

Or all-CLI for ad-hoc runs:

    python swarm/sweep_train.py \\
        --data-dir output/ --primary-type ATTITUDE \\
        --cache-dir .cache/biLSTM --output-dir sweep/ \\
        --model both --gpus 0,1,2,3,4 --mixed-precision \\
        -e w=60,s=1 -e w=80,s=2 -e w=120,s=2 -e w=160,s=4 -e w=200,s=4

The launcher exits non-zero if any child fails.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPTS = {
    "1layer": os.path.join(SCRIPT_DIR, "biLSTM-1Layer-Protocol.py"),
    "2layer": os.path.join(SCRIPT_DIR, "biLSTM-2Layer-Protocol.py"),
}


@dataclass(frozen=True)
class Experiment:
    window_size: int
    stride: int

    @property
    def tag(self) -> str:
        return f"w{self.window_size}_s{self.stride}"


@dataclass(frozen=True)
class SweepConfig:
    data_dir: str
    primary_type: str
    cache_dir: str
    output_dir: str
    models: tuple[str, ...]
    gpus: tuple[int, ...]
    experiments: tuple[Experiment, ...]
    scaler_fit_scope: str
    test_frac: float
    val_frac: float
    mixed_precision: bool
    skip_warm: bool
    batch_size: int


def _parse_experiment_spec(spec: str) -> Experiment:
    """Parse ``w=80,s=2`` or ``window_size=80,stride=2``."""
    parts = dict(p.split("=", 1) for p in spec.split(","))
    aliases = {"w": "window_size", "s": "stride"}
    normalized = {aliases.get(k.strip(), k.strip()): v.strip() for k, v in parts.items()}
    if "window_size" not in normalized or "stride" not in normalized:
        raise ValueError(f"experiment spec missing window_size/stride: {spec!r}")
    return Experiment(window_size=int(normalized["window_size"]), stride=int(normalized["stride"]))


def _experiments_from_yaml(raw: list[dict[str, int]]) -> tuple[Experiment, ...]:
    out = []
    for item in raw:
        if "window_size" not in item or "stride" not in item:
            raise ValueError(f"experiment entry missing keys: {item!r}")
        out.append(Experiment(window_size=int(item["window_size"]), stride=int(item["stride"])))
    return tuple(out)


def _resolve_models(value: str) -> tuple[str, ...]:
    value = value.strip().lower()
    if value == "both":
        return ("1layer", "2layer")
    if value in ("1layer", "2layer"):
        return (value,)
    raise ValueError(f"--model must be 1layer | 2layer | both, got {value!r}")


def _load_config(args: argparse.Namespace) -> SweepConfig:
    cfg: dict[str, Any] = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    def pick(name: str, default: Any = None) -> Any:
        cli_val = getattr(args, name, None)
        if cli_val not in (None, [], ()):
            return cli_val
        return cfg.get(name, default)

    data_dir = pick("data_dir")
    primary_type = pick("primary_type")
    cache_dir = pick("cache_dir")
    output_dir = pick("output_dir", "sweep")
    model = pick("model", "2layer")
    gpus_raw = pick("gpus")
    scaler_fit_scope = pick("scaler_fit_scope", "train_all")
    test_frac = float(pick("test_frac", 0.2))
    val_frac = float(pick("val_frac", 0.2))
    mixed_precision = bool(pick("mixed_precision", False))
    skip_warm = bool(args.skip_warm)
    batch_size = int(pick("batch_size", 64))

    if not (data_dir and primary_type and cache_dir):
        raise SystemExit("data_dir, primary_type, and cache_dir are all required.")

    if isinstance(gpus_raw, str):
        gpus = tuple(int(x) for x in gpus_raw.split(",") if x.strip())
    elif isinstance(gpus_raw, (list, tuple)):
        gpus = tuple(int(x) for x in gpus_raw)
    else:
        raise SystemExit("--gpus or yaml `gpus:` required (e.g. 0,1,2,3,4)")

    if not gpus:
        raise SystemExit("at least one GPU id required")

    if args.experiments:
        experiments = tuple(_parse_experiment_spec(s) for s in args.experiments)
    else:
        experiments = _experiments_from_yaml(cfg.get("experiments", []))
    if not experiments:
        raise SystemExit("no experiments specified (use -e or yaml `experiments:`)")

    return SweepConfig(
        data_dir=data_dir,
        primary_type=primary_type,
        cache_dir=cache_dir,
        output_dir=output_dir,
        models=_resolve_models(model),
        gpus=gpus,
        experiments=experiments,
        scaler_fit_scope=scaler_fit_scope,
        test_frac=test_frac,
        val_frac=val_frac,
        mixed_precision=mixed_precision,
        skip_warm=skip_warm,
        batch_size=batch_size,
    )


def _train_script(model: str) -> str:
    path = TRAIN_SCRIPTS.get(model)
    if path is None or not os.path.exists(path):
        raise SystemExit(f"training script not found for model {model!r}: {path}")
    return path


def _warm_cache(cfg: SweepConfig) -> None:
    """Run the training script(s) once with --dry-run on CPU to populate the cache."""
    if cfg.skip_warm:
        print("[sweep] --skip-warm set; assuming caches are already populated.")
        return

    for model in cfg.models:
        # Pick any experiment for the warm-up — window-size/stride don't affect
        # cache contents, but the script still requires the flags.
        first = cfg.experiments[0]
        cmd = [
            sys.executable,
            _train_script(model),
            "--data-dir", cfg.data_dir,
            "--primary-type", cfg.primary_type,
            "--cache-dir", cfg.cache_dir,
            "--window-size", str(first.window_size),
            "--stride", str(first.stride),
            "--scaler-fit-scope", cfg.scaler_fit_scope,
            "--test-frac", str(cfg.test_frac),
            "--val-frac", str(cfg.val_frac),
            "--cpu",
            "--dry-run",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        warm_log = os.path.join(cfg.output_dir, "logs", f"warm_{model}.log")
        os.makedirs(os.path.dirname(warm_log), exist_ok=True)
        print(f"[sweep] warming cache for model={model} → {warm_log}")
        t0 = time.monotonic()
        with open(warm_log, "w") as logf:
            ret = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
        dt = time.monotonic() - t0
        if ret.returncode != 0:
            sys.stderr.write(f"[sweep] cache warm failed for {model}; see {warm_log}\n")
            raise SystemExit(ret.returncode)
        print(f"[sweep] cache warm OK for {model} in {dt:.1f}s")


def _child_cmd(cfg: SweepConfig, model: str, exp: Experiment, output_path: str) -> list[str]:
    cmd = [
        sys.executable,
        _train_script(model),
        "--data-dir", cfg.data_dir,
        "--primary-type", cfg.primary_type,
        "--cache-dir", cfg.cache_dir,
        "--window-size", str(exp.window_size),
        "--stride", str(exp.stride),
        "--scaler-fit-scope", cfg.scaler_fit_scope,
        "--test-frac", str(cfg.test_frac),
        "--val-frac", str(cfg.val_frac),
        "--output", output_path,
    ]
    if cfg.mixed_precision:
        cmd.append("--mixed-precision")
    cmd.extend(["--batch-size", str(cfg.batch_size)])
    return cmd


def _run_in_batches(cfg: SweepConfig) -> list[dict[str, Any]]:
    """Dispatch children round-robin across GPUs, waiting between full batches."""
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(cfg.output_dir, "models"), exist_ok=True)

    plan: list[tuple[str, Experiment]] = [(m, e) for m in cfg.models for e in cfg.experiments]
    print(f"[sweep] {len(plan)} experiments total, {len(cfg.gpus)} GPU(s) — "
          f"{(len(plan) + len(cfg.gpus) - 1) // len(cfg.gpus)} batch(es)")

    results: list[dict[str, Any]] = []
    for batch_start in range(0, len(plan), len(cfg.gpus)):
        batch = plan[batch_start : batch_start + len(cfg.gpus)]
        procs: list[tuple[subprocess.Popen, str, Experiment, str, str, float]] = []
        for slot_idx, (model, exp) in enumerate(batch):
            gpu = cfg.gpus[slot_idx]
            tag = f"{model}_{exp.tag}"
            output_path = os.path.join(cfg.output_dir, "models", f"aeroshield_{tag}.keras")
            log_path = os.path.join(cfg.output_dir, "logs", f"{tag}.log")
            cmd = _child_cmd(cfg, model, exp, output_path)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            logf = open(log_path, "w")
            t0 = time.monotonic()
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
            print(f"[sweep] launched {tag} PID={proc.pid} GPU={gpu} → {log_path}")
            procs.append((proc, model, exp, output_path, log_path, t0))

        for proc, model, exp, output_path, log_path, t0 in procs:
            rc = proc.wait()
            dt = time.monotonic() - t0
            metrics = _parse_log_metrics(log_path)
            print(f"[sweep] finished {model}_{exp.tag} rc={rc} in {dt:.1f}s "
                  f"(val_f1={metrics.get('val_attack_f1', 'n/a')}, "
                  f"test_f1={metrics.get('test_attack_f1', 'n/a')})")
            results.append(
                {
                    "model": model,
                    "window_size": exp.window_size,
                    "stride": exp.stride,
                    "return_code": rc,
                    "wall_seconds": round(dt, 1),
                    "output": output_path,
                    "log": log_path,
                    **metrics,
                }
            )
    return results


_METRIC_LINE = re.compile(
    r"^\s*Attack\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", re.MULTILINE
)
_THRESH_LINE = re.compile(r"Optimal threshold \(from val\)\s*:\s*([\d.]+)")


def _parse_log_metrics(log_path: str) -> dict[str, Any]:
    """Extract Attack-class precision/recall/F1/support from val + test reports.

    classification_report formats one row per class with the class name in the
    leftmost column. The Val report appears before the TEST report, so we pick
    the first match for val and the second for test.
    """
    try:
        text = open(log_path, encoding="utf-8", errors="replace").read()
    except OSError:
        return {}
    matches = _METRIC_LINE.findall(text)
    out: dict[str, Any] = {}
    if len(matches) >= 1:
        p, r, f1, sup = matches[0]
        out.update(val_attack_precision=float(p), val_attack_recall=float(r),
                   val_attack_f1=float(f1), val_attack_support=int(sup))
    if len(matches) >= 2:
        p, r, f1, sup = matches[1]
        out.update(test_attack_precision=float(p), test_attack_recall=float(r),
                   test_attack_f1=float(f1), test_attack_support=int(sup))
    thresh = _THRESH_LINE.search(text)
    if thresh:
        out["best_threshold"] = float(thresh.group(1))
    return out


def _write_summary(cfg: SweepConfig, results: list[dict[str, Any]]) -> str:
    summary_path = os.path.join(cfg.output_dir, "summary.csv")
    if not results:
        return summary_path
    keys = sorted({k for r in results for k in r.keys()})
    # Stable column order: identity first, then metrics.
    front = ["model", "window_size", "stride", "return_code", "wall_seconds",
             "val_attack_f1", "test_attack_f1", "best_threshold",
             "val_attack_precision", "val_attack_recall", "val_attack_support",
             "test_attack_precision", "test_attack_recall", "test_attack_support"]
    ordered = [k for k in front if k in keys] + [k for k in keys if k not in front]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(results, key=lambda x: (x.get("test_attack_f1") or -1), reverse=True):
            writer.writerow(r)
    print(f"\n[sweep] summary → {summary_path}")
    print(f"{'model':<8} {'w':>5} {'s':>4} {'rc':>3} {'val_F1':>8} {'test_F1':>8}  log")
    for r in sorted(results, key=lambda x: (x.get("test_attack_f1") or -1), reverse=True):
        print(f"{r['model']:<8} {r['window_size']:>5} {r['stride']:>4} "
              f"{r['return_code']:>3} {r.get('val_attack_f1', float('nan')):>8} "
              f"{r.get('test_attack_f1', float('nan')):>8}  {r['log']}")
    return summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=str, help="YAML config (CLI flags override).")
    parser.add_argument("--data-dir", type=str, dest="data_dir")
    parser.add_argument("--primary-type", type=str, dest="primary_type")
    parser.add_argument("--cache-dir", type=str, dest="cache_dir")
    parser.add_argument("--output-dir", type=str, dest="output_dir")
    parser.add_argument("--model", type=str, help="1layer | 2layer | both")
    parser.add_argument("--gpus", type=str, help="Comma-separated GPU ids, e.g. 0,1,2,3,4")
    parser.add_argument("--scaler-fit-scope", type=str, dest="scaler_fit_scope")
    parser.add_argument("--test-frac", type=float, dest="test_frac")
    parser.add_argument("--val-frac", type=float, dest="val_frac")
    parser.add_argument("--mixed-precision", action="store_true", dest="mixed_precision")
    parser.add_argument(
        "-e", "--experiment", action="append", dest="experiments", default=[],
        help='"w=80,s=2" or "window_size=80,stride=2". Repeatable.',
    )
    parser.add_argument(
        "--skip-warm", action="store_true",
        help="Skip the CPU-only cache warm step; assume caches already populated.",
    )
    parser.add_argument(
        "--batch-size", type=int, dest="batch_size",
        help="Training batch size passed to each child (default: 64, recommend 512–2048 for A6000).",
    )
    args = parser.parse_args()
    cfg = _load_config(args)

    print(f"[sweep] models={cfg.models} gpus={cfg.gpus} experiments={len(cfg.experiments)}")
    _warm_cache(cfg)
    results = _run_in_batches(cfg)
    _write_summary(cfg, results)
    any_failure = any(r["return_code"] != 0 for r in results)
    return 1 if any_failure else 0


if __name__ == "__main__":
    sys.exit(main())
