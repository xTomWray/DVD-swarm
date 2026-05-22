"""Fast post-run analysis.

Converts per-drone CSVs to ZSTD-compressed Parquet once, then runs canned
reports or ad-hoc SQL against the cached file. Sub-second queries on the
50-drone, ~150-column captures that pandas/CSV reading would spend ~30s on.

Usage:
    python -m swarm.dataset.analyze                  # report on latest run
    python -m swarm.dataset.analyze --run <path>     # specific run
    python -m swarm.dataset.analyze --force          # rebuild parquet
    python -m swarm.dataset.analyze --sql "SELECT ..."   # ad-hoc query
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output"


def _resolve_run(path: Path | None) -> Path:
    if path is not None:
        p = path.resolve()
        if not p.is_dir():
            raise SystemExit(f"Run dir not found: {p}")
        return p
    candidates = sorted(_OUTPUT_ROOT.glob("run_*"), key=lambda x: x.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No output/run_* directories found under {_OUTPUT_ROOT}.")
    return candidates[-1]


def _ensure_parquet(run_dir: Path, force: bool) -> Path:
    pq = run_dir / "flight.parquet"
    if pq.exists() and not force:
        return pq

    csv_dir = run_dir / "csv"
    if not csv_dir.is_dir():
        raise SystemExit(f"No csv/ subdir under {run_dir}.")
    if not any(csv_dir.glob("drone_*.csv")):
        raise SystemExit(f"No drone_*.csv files under {csv_dir}.")

    pattern = str(csv_dir / "drone_*.csv").replace("'", "''")
    pq_path = str(pq).replace("'", "''")

    print(f"Converting {csv_dir}/drone_*.csv → {pq.name}…", file=sys.stderr)
    duckdb.sql(f"""
        COPY (
            SELECT
                *,
                CAST(regexp_extract(filename, 'drone_(\\d+)\\.csv', 1) AS INTEGER)
                    AS drone_id
            FROM read_csv_auto('{pattern}',
                               union_by_name=true,
                               all_varchar=true,
                               filename=true)
        )
        TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    print(f"  → {pq.stat().st_size / 1e6:.1f} MB", file=sys.stderr)
    return pq


def _connect(pq: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    pq_path = str(pq).replace("'", "''")
    con.execute(f"CREATE VIEW flight AS SELECT * FROM '{pq_path}'")
    return con


def _report(con: duckdb.DuckDBPyConnection, run_dir: Path) -> None:
    meta_path = run_dir / "metadata.json"
    meta: dict[str, object] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            pass

    print(f"\n=== {run_dir.name} ===")
    if meta:
        ready = meta.get("ready_drones") or []
        skipped = meta.get("skipped_drones") or []
        targets = meta.get("attack_targets") or []
        print(
            f"  attack={meta.get('attack', '?')}  "
            f"N={meta.get('swarm_size', '?')}  "
            f"window=[{meta.get('start_s', '?')}, {meta.get('end_s', '?')}]s  "
            f"ready={len(ready) if isinstance(ready, list) else '?'}  "
            f"skipped={len(skipped) if isinstance(skipped, list) else '?'}  "
            f"targets={len(targets) if isinstance(targets, list) else '?'}"
        )

    print("\n[Volume]")
    con.sql("""
        SELECT COUNT(*) AS rows, COUNT(DISTINCT drone_id) AS drones
        FROM flight
    """).show()

    print("[Per-drone row counts]")
    con.sql("""
        SELECT MIN(c)  AS min_rows,
               ROUND(AVG(c), 0) AS mean_rows,
               MAX(c)  AS max_rows
        FROM (SELECT drone_id, COUNT(*) AS c FROM flight GROUP BY drone_id)
    """).show()

    print("[Attack labels]")
    con.sql("""
        SELECT attack_type, COUNT(*) AS rows
        FROM flight
        GROUP BY attack_type
        ORDER BY rows DESC
    """).show()

    print("[Message types — top 12]")
    con.sql("""
        SELECT mav_packet_type,
               COUNT(*) AS total,
               CAST(COUNT(*) / NULLIF(COUNT(DISTINCT drone_id), 0) AS INTEGER)
                   AS per_drone
        FROM flight
        WHERE mav_packet_type IS NOT NULL
        GROUP BY mav_packet_type
        ORDER BY total DESC
        LIMIT 12
    """).show()

    print("[VFR_HUD dynamics by attack_type]")
    con.sql("""
        SELECT attack_type,
               COUNT(*)                                AS rows,
               ROUND(MAX(TRY_CAST(groundspeed AS FLOAT)), 2) AS max_spd_ms,
               ROUND(AVG(TRY_CAST(groundspeed AS FLOAT)), 2) AS mean_spd_ms,
               ROUND(MAX(TRY_CAST(alt AS FLOAT)), 1)         AS max_alt_m
        FROM flight
        WHERE mav_packet_type = 'VFR_HUD'
        GROUP BY attack_type
        ORDER BY attack_type
    """).show()

    print("[ATTITUDE dynamics by attack_type]")
    con.sql("""
        SELECT attack_type,
               COUNT(*)                                                  AS rows,
               ROUND(MAX(ABS(TRY_CAST(roll  AS FLOAT))) * 57.296, 1)     AS max_roll_deg,
               ROUND(MAX(ABS(TRY_CAST(pitch AS FLOAT))) * 57.296, 1)     AS max_pitch_deg
        FROM flight
        WHERE mav_packet_type = 'ATTITUDE'
        GROUP BY attack_type
        ORDER BY attack_type
    """).show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast post-run analysis (Parquet + canned reports).")
    ap.add_argument("--run", type=Path, default=None,
                    help="Run dir to analyze (default: latest output/run_*).")
    ap.add_argument("--force", action="store_true",
                    help="Re-build flight.parquet even if it already exists.")
    ap.add_argument("--sql", type=str, default=None,
                    help="Run an ad-hoc SQL query against view `flight` and exit.")
    args = ap.parse_args()

    run_dir = _resolve_run(args.run)
    pq = _ensure_parquet(run_dir, force=args.force)
    con = _connect(pq)

    if args.sql:
        con.sql(args.sql).show()
    else:
        _report(con, run_dir)


if __name__ == "__main__":
    main()
