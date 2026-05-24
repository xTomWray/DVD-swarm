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
import re
import sys
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


def _convert_drone(
    con: duckdb.DuckDBPyConnection,
    drone_dir: Path,
    out_csv: Path,
    attack_type_value: str,
) -> int:
    """Union all per-type CSVs in ``drone_dir`` into one drone CSV.

    Returns the number of data rows written. Uses DuckDB's
    ``read_csv_auto(..., union_by_name=true)`` so missing columns across types
    become NULLs — same behaviour as the old row-by-row converter but vastly
    faster. ``attack_type`` is appended as a constant column.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(drone_dir / "*.csv").replace("'", "''")
    out_path = str(out_csv).replace("'", "''")
    atk = attack_type_value.replace("'", "''")

    # all_varchar=true matches analyze.py — keeps the mixed-type checksum cols
    # readable without coercion failures, and the trainer's pandas read coerces
    # numerics later.
    con.execute(f"""
        COPY (
            SELECT
                *,
                '{atk}' AS attack_type
            FROM read_csv_auto('{pattern}',
                               union_by_name=true,
                               all_varchar=true)
            ORDER BY TRY_CAST(timestamp AS BIGINT)
        ) TO '{out_path}' (FORMAT CSV, HEADER)
    """)
    n = con.execute(
        f"SELECT COUNT(*) FROM read_csv_auto('{out_path}', all_varchar=true)"
    ).fetchone()[0]
    return int(n)


def _build_run_parquet(con: duckdb.DuckDBPyConnection, run_out_dir: Path) -> None:
    """Build flight.parquet from the per-drone CSVs in ``run_out_dir/csv/``.

    Mirrors swarm/dataset/analyze.py:_ensure_parquet so the resulting parquet
    has the same schema (including drone_id derived from filename) that the
    trainer's fast-loader expects.
    """
    csv_pattern = str(run_out_dir / "csv" / "drone_*.csv").replace("'", "''")
    pq_path = str(run_out_dir / "flight.parquet").replace("'", "''")
    con.execute(f"""
        COPY (
            SELECT
                *,
                CAST(regexp_extract(filename, 'drone_(\\d+)\\.csv', 1) AS INTEGER)
                    AS drone_id
            FROM read_csv_auto('{csv_pattern}',
                               union_by_name=true,
                               all_varchar=true,
                               filename=true)
        ) TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)


def convert_run(
    legacy: LegacyRun,
    dst_root: Path,
    class_name: str,
    *,
    build_parquet: bool = True,
    force: bool = False,
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
    drone_count = 0
    total_rows = 0
    try:
        for drone_dir in sorted(p for p in legacy.src_run_dir.iterdir() if p.is_dir()):
            m = _DRONE_NUMBER_RE.search(drone_dir.name)
            if not m:
                continue
            drone_id = int(m.group(1))
            out_csv = out_run_dir / "csv" / f"drone_{drone_id:03d}.csv"
            n = _convert_drone(con, drone_dir, out_csv, attack_type_value)
            drone_count += 1
            total_rows += n
            print(f"  drone_{drone_id:03d}: {n:>7} rows ({drone_dir.name})")

        if build_parquet and drone_count > 0:
            _build_run_parquet(con, out_run_dir)
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
    ap.add_argument("--force", action="store_true",
                    help="Re-convert even if the destination already has flight.parquet.")
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

    print(f"Transferring {len(runs)} legacy run(s) → {dst}")
    for legacy in runs:
        label = f"{legacy.attack_kind}/" if legacy.attack_kind else ""
        print(f"Converting {label}{legacy.src_run_dir.name} ...")
        convert_run(
            legacy, dst, args.class_name,
            build_parquet=not args.no_parquet,
            force=args.force,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
