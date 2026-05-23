"""Dataset audit — usable row/window counts + sufficiency warnings.

Scans a data directory of DVD-swarm run dirs, applies the same labelling
rules as ``biLSTM-2Layer-Protocol.py`` / ``biLSTM-1Layer-Protocol.py``, and
reports per-class / per-shape / per-run counts plus warnings about
insufficiency. Read-only — never touches ``flight.parquet`` files.

Usage:
    python -m swarm.dataset.audit --data-dir training-data/all/ --primary-type ATTITUDE
    python -m swarm.dataset.audit --data-dir output/ --primary-type ATTITUDE --window-size 120 --stride 4

The audit mirrors trainer semantics exactly:

- Run label comes from ``metadata.json["attack"]`` (``'none'`` → benign, else attack).
- Benign runs contribute every ``mav_packet_type == <primary>`` row; attack runs
  only contribute rows where ``attack_type != 'null'``.
- ``n_windows`` per drone = ``(n_rows - W) // S + 1`` if ``n_rows >= W``, else 0.
- ``preprocess_df`` preserves row count (all rolling/diff ops use ``fillna(0)``),
  so the window count here is bit-exact with what training will see.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb


# ── Helpers shared with the trainer ──────────────────────────────────────────


def _run_label_from_metadata(run_dir: Path) -> str | None:
    """Replicates biLSTM-2Layer-Protocol.py::_run_label_from_metadata.

    Returns 'benign', 'attack', or None when metadata.json is missing/unreadable.
    """
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return "benign" if str(meta.get("attack", "")).lower() == "none" else "attack"


def _shape_from_run_name(run_basename: str) -> str:
    """Extract the shape suffix from a run dir name.

    Run names look like ``run_<UTC>_<attack_slug>_n50_s60_e1200``. The shape
    is everything after the third underscore-separated chunk: ``n50_s60_e1200``.
    Returns ``'unknown'`` if the layout doesn't match.
    """
    parts = run_basename.split("_")
    if len(parts) < 4:
        return "unknown"
    return "_".join(parts[3:])


def _n_windows(n_rows: int, window_size: int, stride: int) -> int:
    """Mirrors ``range(window_size, n_rows + 1, stride)`` in make_windows_from_scaled_rows."""
    if n_rows < window_size:
        return 0
    return (n_rows - window_size) // stride + 1


# ── Audit result ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetAudit:
    primary_type: str
    window_size: int
    stride: int
    drone_files: int
    usable_rows_by_class: dict[str, int]
    usable_windows_by_class: dict[str, int]
    rows_per_run: dict[str, int]
    windows_per_run: dict[str, int]
    runs_by_class: dict[str, list[str]]
    runs_by_shape: dict[str, list[str]]
    shape_class_map: dict[str, set[str]]
    rows_per_shape: dict[str, int]
    windows_per_shape: dict[str, int]
    shape_class_of: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped_runs: list[tuple[str, str]] = field(default_factory=list)

    def _ints(self, by_run: dict[str, int], names: list[str]) -> list[int]:
        return [by_run.get(n, 0) for n in names]

    def min_rows_per_run(self, class_name: str | None = None) -> int:
        runs = self.runs_by_class.get(class_name, list(self.rows_per_run)) if class_name \
            else list(self.rows_per_run)
        vals = self._ints(self.rows_per_run, runs)
        return min(vals) if vals else 0

    def max_rows_per_run(self, class_name: str | None = None) -> int:
        runs = self.runs_by_class.get(class_name, list(self.rows_per_run)) if class_name \
            else list(self.rows_per_run)
        vals = self._ints(self.rows_per_run, runs)
        return max(vals) if vals else 0

    def min_windows_per_run(self, class_name: str | None = None) -> int:
        runs = self.runs_by_class.get(class_name, list(self.windows_per_run)) if class_name \
            else list(self.windows_per_run)
        vals = self._ints(self.windows_per_run, runs)
        return min(vals) if vals else 0

    def max_windows_per_run(self, class_name: str | None = None) -> int:
        runs = self.runs_by_class.get(class_name, list(self.windows_per_run)) if class_name \
            else list(self.windows_per_run)
        vals = self._ints(self.windows_per_run, runs)
        return max(vals) if vals else 0


# ── Core scan ────────────────────────────────────────────────────────────────


def _per_drone_row_counts(
    con: duckdb.DuckDBPyConnection,
    parquet_path: Path,
    primary_type: str,
    is_benign: bool,
) -> list[int]:
    """One DuckDB query → per-drone row counts after primary-type + attack-row filtering."""
    pq = str(parquet_path).replace("'", "''")
    if is_benign:
        sql = f"""
            SELECT drone_id, COUNT(*) AS n_rows
            FROM read_parquet('{pq}')
            WHERE mav_packet_type = ?
            GROUP BY drone_id
            ORDER BY drone_id
        """
    else:
        # Attack runs: drop rows where attack_type is null/'null' to match
        # the trainer's per-row filter ([biLSTM-2Layer-Protocol.py:472-477]).
        sql = f"""
            SELECT drone_id, COUNT(*) AS n_rows
            FROM read_parquet('{pq}')
            WHERE mav_packet_type = ?
              AND LOWER(COALESCE(CAST(attack_type AS VARCHAR), 'null')) <> 'null'
            GROUP BY drone_id
            ORDER BY drone_id
        """
    rows = con.execute(sql, [primary_type]).fetchall()
    return [int(r[1]) for r in rows]


def audit_dataset(
    data_dir: Path,
    primary_type: str,
    window_size: int,
    stride: int,
) -> DatasetAudit:
    """Scan ``data_dir`` for ``run_*`` dirs and produce a structural audit.

    Resolves symlinked run dirs (the sweep workflow uses ``training-data/all/``
    as a curated symlink tree pointing into ``training-data/{attack,benign}/``
    and ``output/``).
    """
    data_dir = data_dir.resolve()
    run_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.name.startswith("run_") and (p.is_dir() or p.is_symlink())
    )

    drone_files = 0
    usable_rows_by_class: dict[str, int] = defaultdict(int)
    usable_windows_by_class: dict[str, int] = defaultdict(int)
    rows_per_run: dict[str, int] = {}
    windows_per_run: dict[str, int] = {}
    runs_by_class: dict[str, list[str]] = defaultdict(list)
    runs_by_shape: dict[str, list[str]] = defaultdict(list)
    shape_class_map: dict[str, set[str]] = defaultdict(set)
    shape_class_of: dict[str, str] = {}
    rows_per_shape: dict[str, int] = defaultdict(int)
    windows_per_shape: dict[str, int] = defaultdict(int)
    skipped_runs: list[tuple[str, str]] = []
    runs_below_window: list[str] = []
    runs_zero_windows: list[str] = []

    con = duckdb.connect()
    try:
        for run in run_dirs:
            name = run.name
            label = _run_label_from_metadata(run)
            if label is None:
                skipped_runs.append((name, "missing or unreadable metadata.json"))
                continue
            pq = run / "flight.parquet"
            if not pq.exists():
                skipped_runs.append((name, "missing flight.parquet — run `make analyze RUN=<run>`"))
                continue

            shape = _shape_from_run_name(name)
            shape_class_map[shape].add(label)
            shape_class_of[shape] = label  # last one wins; only meaningful when class set has size 1
            runs_by_class[label].append(name)
            runs_by_shape[shape].append(name)

            csv_dir = run / "csv"
            if csv_dir.is_dir():
                drone_files += len(list(csv_dir.glob("drone_*.csv")))

            per_drone = _per_drone_row_counts(con, pq, primary_type, is_benign=(label == "benign"))
            run_rows = sum(per_drone)
            run_windows = sum(_n_windows(n, window_size, stride) for n in per_drone)

            rows_per_run[name] = run_rows
            windows_per_run[name] = run_windows
            usable_rows_by_class[label] += run_rows
            usable_windows_by_class[label] += run_windows
            rows_per_shape[shape] += run_rows
            windows_per_shape[shape] += run_windows

            if any(n < window_size for n in per_drone) and run_windows == 0:
                runs_below_window.append(name)
            if run_windows == 0:
                runs_zero_windows.append(name)
    finally:
        con.close()

    warnings = _build_warnings(
        primary_type=primary_type,
        window_size=window_size,
        stride=stride,
        usable_windows_by_class=dict(usable_windows_by_class),
        runs_by_class={k: sorted(v) for k, v in runs_by_class.items()},
        runs_by_shape={k: sorted(v) for k, v in runs_by_shape.items()},
        shape_class_map={k: set(v) for k, v in shape_class_map.items()},
        windows_per_shape=dict(windows_per_shape),
        runs_below_window=sorted(set(runs_below_window)),
        runs_zero_windows=sorted(set(runs_zero_windows)),
        skipped_runs=skipped_runs,
    )

    return DatasetAudit(
        primary_type=primary_type,
        window_size=window_size,
        stride=stride,
        drone_files=drone_files,
        usable_rows_by_class=dict(usable_rows_by_class),
        usable_windows_by_class=dict(usable_windows_by_class),
        rows_per_run=rows_per_run,
        windows_per_run=windows_per_run,
        runs_by_class={k: sorted(v) for k, v in runs_by_class.items()},
        runs_by_shape={k: sorted(v) for k, v in runs_by_shape.items()},
        shape_class_map={k: set(v) for k, v in shape_class_map.items()},
        rows_per_shape=dict(rows_per_shape),
        windows_per_shape=dict(windows_per_shape),
        shape_class_of=shape_class_of,
        warnings=warnings,
        skipped_runs=skipped_runs,
    )


# ── Warnings ─────────────────────────────────────────────────────────────────

_SEVERITY_RANK = {"SEVERE": 0, "WARN": 1, "INFO": 2}


def _w(severity: str, message: str) -> str:
    return f"[{severity}] {message}"


def _build_warnings(
    *,
    primary_type: str,
    window_size: int,
    stride: int,
    usable_windows_by_class: dict[str, int],
    runs_by_class: dict[str, list[str]],
    runs_by_shape: dict[str, list[str]],
    shape_class_map: dict[str, set[str]],
    windows_per_shape: dict[str, int],
    runs_below_window: list[str],
    runs_zero_windows: list[str],
    skipped_runs: list[tuple[str, str]],
) -> list[str]:
    del primary_type, window_size, stride  # not used in messages, but reserved for future
    warnings: list[str] = []

    # Runs with rows < window_size — yields 0 windows. Collapse with zero-window
    # warning when the two sets are identical (the common case).
    if runs_below_window and set(runs_below_window) == set(runs_zero_windows):
        warnings.append(_w("WARN",
            f"{len(runs_below_window)} run(s) have rows < window_size (produce 0 windows): "
            f"{', '.join(runs_below_window[:5])}{' …' if len(runs_below_window) > 5 else ''}"))
    else:
        if runs_below_window:
            warnings.append(_w("WARN",
                f"{len(runs_below_window)} run(s) have rows < window_size: "
                f"{', '.join(runs_below_window[:5])}{' …' if len(runs_below_window) > 5 else ''}"))
        if runs_zero_windows:
            extra = sorted(set(runs_zero_windows) - set(runs_below_window))
            if extra:
                warnings.append(_w("WARN",
                    f"{len(extra)} run(s) produce 0 windows for other reasons: "
                    f"{', '.join(extra[:5])}{' …' if len(extra) > 5 else ''}"))

    # Shape exists in only one class
    for shape, classes in sorted(shape_class_map.items()):
        if len(classes) == 1:
            sole = next(iter(classes))
            warnings.append(_w("WARN", f"shape `{shape}` exists only in {sole} class"))

    # Class-count sufficiency
    n_attack = len(runs_by_class.get("attack", []))
    n_benign = len(runs_by_class.get("benign", []))
    for cls, n in (("attack", n_attack), ("benign", n_benign)):
        if n == 0:
            warnings.append(_w("SEVERE", f"{cls} class has 0 runs — cannot train"))
        elif n < 3:
            warnings.append(_w("SEVERE",
                f"{cls} class has {n} run(s) — not enough for run-level train/val/test (need ≥3)"))
        elif n < 5:
            warnings.append(_w("WARN",
                f"{cls} class has {n} run(s) — fragile train/val/test; recommend ≥5"))

    # Projected split sizes (mirrors trainer defaults test_frac=0.2, val_frac=0.2 if
    # caller doesn't override). Warn when val or test cannot contain both classes.
    test_frac = 0.2
    val_frac = 0.2
    for cls, n in (("attack", n_attack), ("benign", n_benign)):
        if n >= 3:
            n_test = max(1, round(n * test_frac))
            n_trainval = n - n_test
            n_val = max(1, round(n_trainval * val_frac))
            n_train = n_trainval - n_val
            if n_test == 0 or n_val == 0 or n_train == 0:
                warnings.append(_w("WARN",
                    f"{cls} split (test_frac=0.2, val_frac=0.2) projects "
                    f"train={n_train} val={n_val} test={n_test} — at least one is empty"))

    # If either val or test would have only one class, flag it. We approximate by
    # checking whether either class has ≤ ceil(1 / test_frac) runs.
    if 0 < n_attack < 5 or 0 < n_benign < 5:
        warnings.append(_w("WARN",
            "val/test sets may not contain both classes given minority class size; "
            "consider leave-one-run-out or collect more runs"))

    # Row balance — using windows (the actual training samples)
    aw = usable_windows_by_class.get("attack", 0)
    bw = usable_windows_by_class.get("benign", 0)
    if aw and bw:
        bigger, smaller = max(aw, bw), min(aw, bw)
        ratio = bigger / smaller
        if ratio >= 3:
            minority = "benign" if bw < aw else "attack"
            warnings.append(_w("WARN",
                f"window balance attack:benign = {aw:,}:{bw:,} ({ratio:.2f}:1); "
                f"consider more {minority} data"))
        else:
            warnings.append(_w("INFO",
                f"window balance attack:benign = {aw:,}:{bw:,} ({aw/max(bw,1):.2f}:1)"))

    # Run-count comparison (independent of row counts)
    if n_attack and n_benign and (n_attack <= 5 or n_benign <= 5):
        if n_benign < n_attack:
            warnings.append(_w("WARN",
                f"benign runs ({n_benign}) < attack runs ({n_attack}); collect more benign runs"))
        elif n_attack < n_benign:
            warnings.append(_w("WARN",
                f"attack runs ({n_attack}) < benign runs ({n_benign}); collect more attack runs"))

    # Shape dominance
    total_windows = sum(windows_per_shape.values())
    if total_windows > 0:
        for shape, w in sorted(windows_per_shape.items(), key=lambda x: -x[1]):
            frac = w / total_windows
            if frac >= 0.7:
                warnings.append(_w("INFO",
                    f"shape `{shape}` dominates {frac*100:.0f}% of windows; "
                    "consider per-shape metrics or more diverse data"))
                break  # only flag the top shape

    # Skipped runs note
    if skipped_runs:
        missing_pq = [r for r, why in skipped_runs if "flight.parquet" in why]
        missing_meta = [r for r, why in skipped_runs if "metadata.json" in why]
        if missing_pq:
            warnings.append(_w("INFO",
                f"{len(missing_pq)} run(s) skipped (missing flight.parquet; run `make analyze` first)"))
        if missing_meta:
            warnings.append(_w("INFO",
                f"{len(missing_meta)} run(s) skipped (missing/unreadable metadata.json)"))

    warnings.sort(key=lambda s: _SEVERITY_RANK.get(s.split("]")[0].lstrip("["), 99))
    return warnings


# ── Printed report ───────────────────────────────────────────────────────────


def _format_int(n: int) -> str:
    return f"{n:,}"


def print_report(audit: DatasetAudit, data_dir: Path) -> None:
    print(f"DATASET AUDIT — primary_type={audit.primary_type}  "
          f"window_size={audit.window_size}  stride={audit.stride}")
    print(f"data_dir: {data_dir}")
    print()
    print(f"drone_files: {_format_int(audit.drone_files)}")
    print()

    # Per-class summary
    header = f"  {'class':<8} {'runs':>5} {'usable_rows':>14} {'usable_windows':>16} " \
             f"{'min_rows':>10} {'max_rows':>10} {'min_wins':>10} {'max_wins':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for cls in ("attack", "benign"):
        runs = audit.runs_by_class.get(cls, [])
        rows = audit.usable_rows_by_class.get(cls, 0)
        wins = audit.usable_windows_by_class.get(cls, 0)
        print(f"  {cls:<8} {len(runs):>5} {_format_int(rows):>14} "
              f"{_format_int(wins):>16} "
              f"{_format_int(audit.min_rows_per_run(cls)):>10} "
              f"{_format_int(audit.max_rows_per_run(cls)):>10} "
              f"{_format_int(audit.min_windows_per_run(cls)):>10} "
              f"{_format_int(audit.max_windows_per_run(cls)):>10}")

    # Per-shape table
    print("\nper-shape totals")
    shape_header = f"  {'shape':<25} {'class':<8} {'runs':>5} {'rows':>14} {'windows':>14}"
    print(shape_header)
    print("  " + "─" * (len(shape_header) - 2))
    for shape in sorted(audit.runs_by_shape):
        runs = audit.runs_by_shape[shape]
        # If shape spans multiple classes, show "mixed"
        classes = audit.shape_class_map[shape]
        cls_str = next(iter(classes)) if len(classes) == 1 else "mixed"
        rows = audit.rows_per_shape.get(shape, 0)
        wins = audit.windows_per_shape.get(shape, 0)
        print(f"  {shape:<25} {cls_str:<8} {len(runs):>5} "
              f"{_format_int(rows):>14} {_format_int(wins):>14}")

    # Warnings
    print("\nWARNINGS")
    if not audit.warnings:
        print("  (none — dataset looks structurally healthy)")
    else:
        for w in audit.warnings:
            print(f"  {w}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Audit a DVD-swarm dataset: usable rows/windows + sufficiency warnings.",
    )
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing run_* subdirectories (symlinks ok).")
    ap.add_argument("--primary-type", type=str, required=True,
                    help="MAVLink message type to audit (e.g. ATTITUDE).")
    ap.add_argument("--window-size", type=int, default=80,
                    help="Trainer's --window-size (default 80).")
    ap.add_argument("--stride", type=int, default=2,
                    help="Trainer's --stride (default 2).")
    args = ap.parse_args()

    if not args.data_dir.is_dir():
        print(f"ERROR: --data-dir not found: {args.data_dir}", file=sys.stderr)
        return 2

    audit = audit_dataset(
        data_dir=args.data_dir,
        primary_type=args.primary_type,
        window_size=args.window_size,
        stride=args.stride,
    )
    print_report(audit, args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
