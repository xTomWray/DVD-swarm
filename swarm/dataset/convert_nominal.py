"""Convert legacy per-message-type nominal CSVs to the per-drone union format.

Input tree (legacy AeroShield/dvdsh format):
    <src>/Nominal-<R>/Nominal-<R>-<M>/<TYPE>.csv
        where R is the run index, M the drone index, and TYPE is one of
        the MAVLink message types (ATTITUDE, HEARTBEAT, ...).

Output tree (current packet_writer.py format):
    <dst>/run_old-nominal-<R>/
        csv/drone_<MMM>.csv     <- per-drone union-schema CSVs
        metadata.json            <- {"source": "<src>/Nominal-<R>", "attack": "none"}
        sim.log                  <- short conversion note

Every row's `attack_type` is set to "null" (these are benign captures).
Missing payload cells (for fields not in this row's message type) are "null".

Usage:
    python -m swarm.dataset.convert_nominal \\
        --src swarm/data/nominal_data \\
        --dst output \\
        [--runs Nominal-1 Nominal-3]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator

from .dvdsh_compat import (
    IP_HEADER_FIELDS,
    MAV_HEADER_FIELDS,
    TCP_HEADER_FIELDS,
    UDP_HEADER_FIELDS,
)
from .mavlink_schema import SCHEMA


def _build_payload_union() -> list[str]:
    """Return the sorted union of payload field names, deduplicated against
    fixed column names.

    Same logic as packet_writer._build_payload_union — payload fields whose
    name collides with a reserved fixed column are excluded.

    Returns:
        Alphabetically sorted list of unique payload field names that do not
        collide with any reserved column name.
    """
    reserved: set[str] = {
        "mav_packet_type", "sim_uuid", "timestamp",
        "frame_timestamp", "attack_type",
        *MAV_HEADER_FIELDS,
        *(f"ip_{f}" for f in IP_HEADER_FIELDS),
        *(f"udp_{f}" for f in UDP_HEADER_FIELDS),
        *(f"tcp_{f}" for f in TCP_HEADER_FIELDS),
    }
    names: set[str] = set()
    for _msgid, (_msgname, fieldnames) in SCHEMA.items():
        names.update(fieldnames)
    return sorted(names - reserved)


def _build_columns() -> list[str]:
    """Return the full ordered column list matching packet_writer._build_columns.

    Returns:
        List of column names in the fixed dvdsh-compatible order.
    """
    cols: list[str] = ["mav_packet_type", "sim_uuid", "timestamp"]
    cols.extend(MAV_HEADER_FIELDS)
    cols.extend(_build_payload_union())
    cols.extend(f"ip_{f}" for f in IP_HEADER_FIELDS)
    cols.extend(f"udp_{f}" for f in UDP_HEADER_FIELDS)
    cols.extend(f"tcp_{f}" for f in TCP_HEADER_FIELDS)
    cols.append("frame_timestamp")
    cols.append("attack_type")
    return cols


def _format_cell(value: object) -> str:
    """Convert *value* to a CSV-safe cell using dvdsh ``clean_arr_csv`` rules.

    Matches packet_writer._format_cell with the additional rule that empty
    strings from legacy CSVs are normalised to "null".

    Args:
        value: Raw Python value from a legacy CSV cell or derived field.

    Returns:
        A string containing no unescaped commas or newlines.
    """
    if value is None or value == "":
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return ";".join(_format_cell(v) for v in value)
    if isinstance(value, bytes):
        return ";".join(str(b) for b in value)
    return str(value).replace(",", ";").replace("\n", " ")


def _iter_legacy_drones(run_dir: Path) -> Iterator[tuple[int, Path]]:
    """Yield (drone_number, drone_dir) pairs sorted by drone_number.

    Drone directories are named ``<run_name>-<M>`` (e.g. ``Nominal-1-7``).

    Args:
        run_dir: Path to a ``Nominal-<R>/`` directory.

    Yields:
        Tuples of (drone_number, drone_directory_path).
    """
    pattern = re.compile(r"-(\d+)$")
    found: list[tuple[int, Path]] = []
    for sub in run_dir.iterdir():
        if not sub.is_dir():
            continue
        m = pattern.search(sub.name)
        if m:
            found.append((int(m.group(1)), sub))
    found.sort()
    for n, p in found:
        yield n, p


def _convert_drone(
    drone_dir: Path,
    sim_uuid: str,
    out_path: Path,
    columns: list[str],
    col_index: dict[str, int],
) -> int:
    """Convert one legacy drone directory to a union-schema CSV.

    Reads every ``<TYPE>.csv`` in the directory, merges rows, sorts by
    ``timestamp`` if present, and writes one unified file with all
    reserved/payload columns.

    Args:
        drone_dir: Directory containing per-message-type CSV files.
        sim_uuid: Run identifier to embed in every row's ``sim_uuid`` cell.
        out_path: Destination path for the output CSV file.
        columns: Ordered list of output column names.
        col_index: Mapping from column name to its position in *columns*.

    Returns:
        Number of data rows written (excluding the header).
    """
    rows: list[tuple[int, list[str]]] = []  # (sort_key, cells)
    for csv_path in sorted(drone_dir.glob("*.csv")):
        msg_type = csv_path.stem  # e.g. "ATTITUDE"
        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for src_row in reader:
                cells = ["null"] * len(columns)
                cells[col_index["mav_packet_type"]] = msg_type
                cells[col_index["sim_uuid"]] = sim_uuid
                cells[col_index["attack_type"]] = "null"

                # Try to extract a timestamp for sorting; legacy files use a
                # column named "timestamp" with millisecond ints (dvdsh
                # convention). Fall back to "Timestamp" casing. Leave as null
                # and sort key 0 if absent.
                ts_str = src_row.get("timestamp", "") or src_row.get("Timestamp", "")
                if ts_str:
                    cells[col_index["timestamp"]] = ts_str
                    try:
                        sort_key = int(float(ts_str))
                    except ValueError:
                        sort_key = 0
                else:
                    sort_key = 0

                # Map every source column that exists in the union schema.
                for src_name, src_val in src_row.items():
                    if src_name is None:
                        continue
                    col = col_index.get(src_name)
                    if col is None:
                        continue
                    if src_val == "" or src_val is None:
                        continue
                    cells[col] = _format_cell(src_val)

                rows.append((sort_key, cells))

    rows.sort(key=lambda x: x[0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        fh.write(",".join(columns) + "\n")
        for _key, cells in rows:
            fh.write(",".join(cells) + "\n")
    return len(rows)


def convert_run(src_run_dir: Path, dst_root: Path) -> None:
    """Convert one legacy run directory (e.g. ``Nominal-1``) to the new format.

    Creates ``<dst_root>/run_old-nominal-<r>/csv/drone_<NNN>.csv`` for every
    drone found, plus ``metadata.json`` and ``sim.log``.

    Args:
        src_run_dir: Path to the ``Nominal-<R>/`` source directory.
        dst_root: Destination root (typically ``output/``).
    """
    run_name = src_run_dir.name                        # "Nominal-1"
    sim_uuid = f"old-{run_name.lower()}"               # "old-nominal-1"
    out_dir = dst_root / f"run_old-{run_name.lower()}"
    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    columns = _build_columns()
    col_index: dict[str, int] = {c: i for i, c in enumerate(columns)}

    total_rows = 0
    drone_count = 0
    for drone_num, drone_dir in _iter_legacy_drones(src_run_dir):
        out_path = csv_dir / f"drone_{drone_num:03d}.csv"
        n = _convert_drone(drone_dir, sim_uuid, out_path, columns, col_index)
        total_rows += n
        drone_count += 1
        print(f"  drone_{drone_num:03d}: {n} rows ({drone_dir.name})")

    metadata = {
        "source": str(src_run_dir.resolve()),
        "attack": "none",
        "drone_count": drone_count,
        "total_rows": total_rows,
        "converter": "swarm.dataset.convert_nominal",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "sim.log").write_text(
        f"Converted from {src_run_dir.resolve()}\n"
        f"{drone_count} drones, {total_rows} rows total\n"
    )
    print(f"  -> {out_dir} ({drone_count} drones, {total_rows} rows)")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="Root containing Nominal-<R>/ directories.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Destination root (typically `output/`).",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        metavar="NAME",
        help="Specific run names to convert (default: all Nominal-*).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the converter.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Exit code: 0 on success, 1 on error.
    """
    args = parse_args(argv)
    if not args.src.is_dir():
        print(f"error: --src {args.src} is not a directory", file=sys.stderr)
        return 1
    args.dst.mkdir(parents=True, exist_ok=True)

    if args.runs:
        run_dirs = [args.src / name for name in args.runs]
    else:
        run_dirs = sorted(
            p for p in args.src.iterdir()
            if p.is_dir() and p.name.startswith("Nominal-")
        )

    if not run_dirs:
        print(f"error: no Nominal-* dirs under {args.src}", file=sys.stderr)
        return 1

    print(f"Converting {len(run_dirs)} run(s) -> {args.dst}")
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"  skip {run_dir.name}: not a directory")
            continue
        print(f"Converting {run_dir.name}...")
        convert_run(run_dir, args.dst)

    return 0


if __name__ == "__main__":
    sys.exit(main())
