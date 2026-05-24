"""Fast DuckDB-based legacy-data importer.

Converts the legacy AeroShield dvdsh capture layout into the current
DVD-swarm format under ``training-data/{attack,benign}/run_*/`` so the
existing trainer/audit/sweep flow can consume it directly.

Source layouts:
    benign:  <src>/Nominal-<R>/Nominal-<R>-<M>/<TYPE>.csv
    attack:  <src>/<attack-kind>/<RUN-NAME>/<RUN-NAME>-<M>/<TYPE>.csv
             attack-kind examples: attitude-spoof, battery-spoof, gps-spoof,
             vfr-hud-spoof, systemStatus-spoof, criticalError-spoof,
             emergencyStatus-spoof, satellite-spoof.

Output layout (per run):
    <dst>/run_legacy-<slug>-<N>/
        csv/drone_<NNN>.csv     <- union of all per-type CSVs for that drone
        flight.parquet           <- ZSTD parquet for the fast loader
        metadata.json            <- {"attack": "...", "source": "...", ...}

The conversion is two DuckDB queries per drone (concat → CSV, then per-run
parquet build). For a 50-drone capture this is typically <10s total — vs.
minutes for the old row-by-row converter at
``swarm/dataset/convert_nominal.py``.

Usage:
    python -m swarm.dataset.transfer --src /path/nominal_data --class benign
    python -m swarm.dataset.transfer --src /path/attack_data  --class attack
    python -m swarm.dataset.transfer --src /path --class attack --runs attitude-spoof
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import duckdb


# Map directory-name attack kinds (as they appear in /attack_data/) to the
# canonical snake_case attack-type value that DVD-swarm writes today.
# Anything not in this map is auto-converted: lowercased, hyphens→underscores.
_ATTACK_KIND_TO_TYPE: dict[str, str] = {
    "attitude-spoof": "attitude_spoof",
    "battery-spoof": "battery_spoof",
    "criticalerror-spoof": "critical_error_spoof",
    "emergencystatus-spoof": "emergency_status_spoof",
    "gps-spoof": "gps_spoof",
    "satellite-spoof": "satellite_spoof",
    "systemstatus-spoof": "system_status_spoof",
    "vfr-hud-spoof": "vfr_hud_spoof",
}

_DRONE_NUMBER_RE = re.compile(r"-(\d+)$")


def _attack_type_for_kind(kind_dir_name: str) -> str:
    key = kind_dir_name.lower()
    if key in _ATTACK_KIND_TO_TYPE:
        return _ATTACK_KIND_TO_TYPE[key]
    # Fallback: lowercase, hyphens → underscores.
    return key.replace("-", "_")


def _run_number(run_dir_name: str) -> int | None:
    """Extract the trailing integer from a run dir name (e.g. 'Nominal-11' → 11)."""
    m = _DRONE_NUMBER_RE.search(run_dir_name)
    return int(m.group(1)) if m else None


def _slug_for_dst(class_name: str, attack_kind: str | None) -> str:
    """Compact slug for the destination dir; no underscores so audit's shape
    parser produces a clean 'legacy' shape suffix.
    """
    if class_name == "benign":
        return "nominal"
    # Attack: compress hyphens out so the slug is one token.
    assert attack_kind is not None
    return attack_kind.replace("-", "").lower()


@dataclass(frozen=True)
class LegacyRun:
    """One source run to convert."""
    src_run_dir: Path
    attack_kind: str | None   # None for benign; e.g. 'attitude-spoof' for attack


def _discover_runs(src: Path, class_name: str, run_filter: list[str] | None) -> list[LegacyRun]:
    """Walk ``src`` and return every legacy run dir matching ``class_name``.

    For benign: ``<src>/Nominal-<R>/``.
    For attack: ``<src>/<kind>/<RUN-NAME>/`` — kind dir at depth 1, run at depth 2.
    ``run_filter`` (optional) restricts to specific subdir names (matched at the
    appropriate level for the class).
    """
    runs: list[LegacyRun] = []
    if class_name == "benign":
        for sub in sorted(p for p in src.iterdir() if p.is_dir()):
            if not sub.name.lower().startswith("nominal"):
                continue
            if run_filter and sub.name not in run_filter:
                continue
            runs.append(LegacyRun(src_run_dir=sub, attack_kind=None))
        return runs

    # attack
    for kind_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        if run_filter and kind_dir.name not in run_filter:
            # When --runs is supplied for attack mode, we treat it as a kind
            # filter (e.g. --runs attitude-spoof gps-spoof) since users
            # typically want to import one or two attack kinds at a time.
            continue
        for run_dir in sorted(p for p in kind_dir.iterdir() if p.is_dir()):
            runs.append(LegacyRun(src_run_dir=run_dir, attack_kind=kind_dir.name))
    return runs


def _build_flight_parquet_from_legacy(
    con: duckdb.DuckDBPyConnection,
    src_run_dir: Path,
    out_pq: Path,
    attack_type_value: str,
    *,
    duckdb_threads: int | None = None,
) -> tuple[int, list[int]]:
    """One DuckDB query: read every per-type CSV across every drone in a run.

    Returns ``(total_rows, sorted_drone_ids)``. The drone_id is extracted
    from the filename path (e.g. ``.../Nominal-11-7/ATTITUDE.csv`` → 7) using
    the trailing-int-before-/<file>.csv pattern; works for both nominal and
    attack tree layouts. ``attack_type`` is appended as a constant column.

    DuckDB scans all ~1500 per-type files (50 drones × ~30 types) in parallel,
    instead of the previous per-drone loop which underfed DuckDB's worker pool.
    """
    out_pq.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(src_run_dir / "*" / "*.csv").replace("'", "''")
    pq_path = str(out_pq).replace("'", "''")
    atk = attack_type_value.replace("'", "''")

    if duckdb_threads is not None:
        con.execute(f"SET threads = {duckdb_threads}")

    # all_varchar=true matches analyze.py (keeps mixed-type checksum columns
    # readable); pandas coerces numerics later in the trainer.
    # ORDER BY drone_id, mav_packet_type tightens row-group statistics so the
    # trainer's predicate-pushdown skips ~N-1/N row groups per drone read.
    # COMPRESSION_LEVEL 1 cuts write time ~2-3× vs default level 3 at ~10% size cost.
    con.execute(f"""
        COPY (
            SELECT
                *,
                CAST(regexp_extract(filename, '-([0-9]+)/[^/]+\\.csv$', 1) AS INTEGER)
                    AS drone_id,
                '{atk}' AS attack_type
            FROM read_csv_auto('{pattern}',
                               union_by_name=true,
                               all_varchar=true,
                               filename=true)
            ORDER BY drone_id, mav_packet_type
        ) TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 1)
    """)

    rows = con.execute(f"""
        SELECT drone_id, COUNT(*) AS n
        FROM read_parquet('{pq_path}')
        WHERE drone_id IS NOT NULL
        GROUP BY drone_id
        ORDER BY drone_id
    """).fetchall()
    drone_ids = [int(r[0]) for r in rows]
    total = sum(int(r[1]) for r in rows)
    return total, drone_ids


def _write_drone_csv_stubs(
    csv_dir: Path,
    drone_ids: list[int],
) -> None:
    """Write tiny header-only ``drone_NNN.csv`` files so the trainer's glob
    discovers each drone. The trainer prefers ``flight.parquet`` (which we
    built) and only falls back to these stubs in the rare parquet-read failure
    case. A minimal header keeps ``pd.read_csv`` from raising EmptyDataError.
    """
    csv_dir.mkdir(parents=True, exist_ok=True)
    header = "mav_packet_type,timestamp,attack_type\n"
    for drone_id in drone_ids:
        (csv_dir / f"drone_{drone_id:03d}.csv").write_text(header)


class IntegrityError(RuntimeError):
    """Raised when the converted parquet doesn't match the legacy source."""


def verify_run(
    con: duckdb.DuckDBPyConnection,
    src_run_dir: Path,
    out_run_dir: Path,
    *,
    expected_attack_type: str,
) -> dict[str, object]:
    """Cross-check the converted parquet against the legacy source.

    Three invariants — fail loudly on any drift:
      1. parquet row count == sum of source per-type CSV rows (minus headers).
      2. distinct drone_id in parquet == count of drone subdirs in source.
      3. every parquet row's attack_type equals ``expected_attack_type``
         (the constant we set during conversion).

    Returns a dict with all measured values for logging. Raises
    ``IntegrityError`` on mismatch.
    """
    pq = out_run_dir / "flight.parquet"
    if not pq.exists():
        raise IntegrityError(f"flight.parquet not found at {pq}")

    src_pattern = str(src_run_dir / "*" / "*.csv").replace("'", "''")
    pq_path = str(pq).replace("'", "''")

    # Source row count: COUNT(*) across all per-type CSVs. DuckDB strips the
    # header row automatically when reading via read_csv_auto.
    src_rows = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{src_pattern}', "
        f"union_by_name=true, all_varchar=true)"
    ).fetchone()[0]
    pq_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{pq_path}')").fetchone()[0]

    src_drones = sum(1 for p in src_run_dir.iterdir()
                     if p.is_dir() and _DRONE_NUMBER_RE.search(p.name))
    pq_drones = con.execute(
        f"SELECT COUNT(DISTINCT drone_id) FROM read_parquet('{pq_path}') "
        f"WHERE drone_id IS NOT NULL"
    ).fetchone()[0]
    null_drone_id = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{pq_path}') WHERE drone_id IS NULL"
    ).fetchone()[0]

    # attack_type distinct values: must be exactly {expected_attack_type}.
    atk_distinct = [
        r[0] for r in con.execute(
            f"SELECT DISTINCT attack_type FROM read_parquet('{pq_path}')"
        ).fetchall()
    ]

    issues: list[str] = []
    if src_rows != pq_rows:
        issues.append(f"row count mismatch: src={src_rows:,} pq={pq_rows:,} diff={src_rows-pq_rows:+,}")
    if src_drones != pq_drones:
        issues.append(f"drone count mismatch: src={src_drones} pq={pq_drones}")
    if null_drone_id > 0:
        issues.append(f"{null_drone_id:,} parquet row(s) have NULL drone_id (regex extraction failed)")
    if atk_distinct != [expected_attack_type]:
        issues.append(f"attack_type values not constant: {atk_distinct!r} (expected [{expected_attack_type!r}])")

    if issues:
        raise IntegrityError(
            f"verify failed for {out_run_dir.name}:\n  - " + "\n  - ".join(issues)
        )

    return {
        "src_rows": int(src_rows),
        "pq_rows": int(pq_rows),
        "src_drones": int(src_drones),
        "pq_drones": int(pq_drones),
        "attack_type": expected_attack_type,
    }


def convert_run(
    legacy: LegacyRun,
    dst_root: Path,
    class_name: str,
    *,
    build_parquet: bool = True,
    force: bool = False,
    duckdb_threads: int | None = None,
) -> Path:
    """Convert one legacy run to the current format. Returns the output run dir.

    Idempotent unless ``force=True``: skips conversion when both
    ``flight.parquet`` and any drone CSVs already exist at the destination.
    """
    run_num = _run_number(legacy.src_run_dir.name) or 0
    slug = _slug_for_dst(class_name, legacy.attack_kind)
    attack_type_value = (
        "null" if class_name == "benign"
        else _attack_type_for_kind(legacy.attack_kind or "")
    )
    metadata_attack = (
        "none" if class_name == "benign"
        else _attack_type_for_kind(legacy.attack_kind or "")
    )

    out_run_dir = dst_root / f"run_legacy-{slug}-{run_num}_{metadata_attack}_legacy"
    if not force and (out_run_dir / "flight.parquet").exists():
        print(f"  skip (already converted): {out_run_dir.name}")
        return out_run_dir
    out_run_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    try:
        if build_parquet:
            total_rows, drone_ids = _build_flight_parquet_from_legacy(
                con,
                src_run_dir=legacy.src_run_dir,
                out_pq=out_run_dir / "flight.parquet",
                attack_type_value=attack_type_value,
                duckdb_threads=duckdb_threads,
            )
            _write_drone_csv_stubs(out_run_dir / "csv", drone_ids)
            drone_count = len(drone_ids)
            print(f"  parquet: {total_rows:,} rows across {drone_count} drone(s)")
            # Integrity check — fail loudly if the parquet doesn't match source.
            stats = verify_run(
                con, legacy.src_run_dir, out_run_dir,
                expected_attack_type=attack_type_value,
            )
            print(f"  ✓ verified: {stats['src_rows']:,} src == {stats['pq_rows']:,} pq, "
                  f"{stats['pq_drones']} drone(s)")
        else:
            # --no-parquet fallback: still discover drones, write CSV stubs only.
            drone_ids = []
            for drone_dir in sorted(p for p in legacy.src_run_dir.iterdir() if p.is_dir()):
                m = _DRONE_NUMBER_RE.search(drone_dir.name)
                if m:
                    drone_ids.append(int(m.group(1)))
            _write_drone_csv_stubs(out_run_dir / "csv", drone_ids)
            drone_count = len(drone_ids)
            total_rows = 0
            print(f"  --no-parquet: wrote {drone_count} CSV stub(s) only")
    finally:
        con.close()

    (out_run_dir / "metadata.json").write_text(json.dumps({
        "attack": metadata_attack,
        "source": str(legacy.src_run_dir.resolve()),
        "drone_count": drone_count,
        "total_rows": total_rows,
        "converter": "swarm.dataset.transfer",
    }, indent=2))
    print(f"  -> {out_run_dir} ({drone_count} drones, {total_rows:,} rows)")
    return out_run_dir


def _convert_run_worker(
    legacy: LegacyRun,
    dst: Path,
    class_name: str,
    build_parquet: bool,
    force: bool,
    duckdb_threads: int,
) -> None:
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    convert_run(
        legacy, dst, class_name,
        build_parquet=build_parquet,
        force=force,
        duckdb_threads=duckdb_threads,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--src", type=Path, required=True,
                    help="Top-level legacy data dir (nominal_data/ or attack_data/).")
    ap.add_argument("--class", dest="class_name", required=True,
                    choices=("attack", "benign"),
                    help="Whether the source tree contains attack or benign runs.")
    ap.add_argument("--dst", type=Path, default=None,
                    help="Destination root. Defaults to "
                         "<repo>/training-data/{attack,benign}/.")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="Restrict to specific subdir names. For --class benign these "
                         "are run dirs (Nominal-1 ...); for --class attack they are "
                         "attack-kind dirs (attitude-spoof ...).")
    ap.add_argument("--no-parquet", action="store_true",
                    help="Skip the flight.parquet build (run `make analyze` later).")
    ap.add_argument("--verify-only", action="store_true",
                    help="Skip conversion; just re-run integrity checks on the "
                         "already-converted runs under --dst (cross-checks "
                         "parquet row/drone counts against the legacy source).")
    ap.add_argument("--force", action="store_true",
                    help="Re-convert even if the destination already has flight.parquet.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel conversion workers (default: 1). Each worker handles "
                         "one run; DuckDB threads per worker = cpu_count // workers.")
    args = ap.parse_args(argv)

    src = args.src.resolve()
    if not src.is_dir():
        print(f"error: --src {src} is not a directory", file=sys.stderr)
        return 1

    # Default dst rooted at <repo>/training-data/<class>/.
    if args.dst is None:
        repo_root = Path(__file__).resolve().parents[2]
        dst = repo_root / "training-data" / args.class_name
    else:
        dst = args.dst.resolve()
    dst.mkdir(parents=True, exist_ok=True)

    runs = _discover_runs(src, args.class_name, args.runs)
    if not runs:
        print(f"error: no legacy runs found under {src} for class={args.class_name}",
              file=sys.stderr)
        return 1

    if args.verify_only:
        print(f"Verifying {len(runs)} converted run(s) under {dst}")
        con = duckdb.connect()
        failures: list[str] = []
        try:
            for legacy in runs:
                run_num = _run_number(legacy.src_run_dir.name) or 0
                slug = _slug_for_dst(args.class_name, legacy.attack_kind)
                attack_type_value = (
                    "null" if args.class_name == "benign"
                    else _attack_type_for_kind(legacy.attack_kind or "")
                )
                metadata_attack = (
                    "none" if args.class_name == "benign"
                    else _attack_type_for_kind(legacy.attack_kind or "")
                )
                out_run_dir = dst / f"run_legacy-{slug}-{run_num}_{metadata_attack}_legacy"
                if not (out_run_dir / "flight.parquet").exists():
                    print(f"  {out_run_dir.name}: NOT CONVERTED (skipping)")
                    continue
                try:
                    stats = verify_run(con, legacy.src_run_dir, out_run_dir,
                                       expected_attack_type=attack_type_value)
                    print(f"  ✓ {out_run_dir.name}: "
                          f"{stats['pq_rows']:,} rows, {stats['pq_drones']} drone(s)")
                except IntegrityError as exc:
                    failures.append(str(exc))
                    print(f"  ✗ {exc}")
        finally:
            con.close()
        if failures:
            print(f"\n{len(failures)} run(s) failed verification.", file=sys.stderr)
            return 2
        print("\nAll verified.")
        return 0

    n_workers = min(args.workers, len(runs))
    threads_each = max(1, (os.cpu_count() or 4) // n_workers)

    if n_workers > 1:
        print(
            f"Transferring {len(runs)} legacy run(s) → {dst}  "
            f"[{n_workers} workers × {threads_each} DuckDB thread(s) each]"
        )
        fut_to_run: dict[Future[None], LegacyRun] = {}
        failures: list[str] = []
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for legacy in runs:
                fut = pool.submit(
                    _convert_run_worker,
                    legacy, dst, args.class_name,
                    not args.no_parquet, args.force, threads_each,
                )
                fut_to_run[fut] = legacy

            for fut in as_completed(fut_to_run):
                legacy = fut_to_run[fut]
                label = f"{legacy.attack_kind}/" if legacy.attack_kind else ""
                try:
                    fut.result()
                except Exception as exc:
                    msg = f"FATAL ({label}{legacy.src_run_dir.name}): {exc}"
                    print(f"\n{msg}", file=sys.stderr)
                    failures.append(msg)

        if failures:
            return 2
        return 0

    # Sequential path (--workers 1, the default).
    print(f"Transferring {len(runs)} legacy run(s) → {dst}")
    for legacy in runs:
        label = f"{legacy.attack_kind}/" if legacy.attack_kind else ""
        print(f"Converting {label}{legacy.src_run_dir.name} ...")
        try:
            convert_run(
                legacy, dst, args.class_name,
                build_parquet=not args.no_parquet,
                force=args.force,
                duckdb_threads=threads_each,
            )
        except IntegrityError as exc:
            print(f"\nFATAL: {exc}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
