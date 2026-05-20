"""CSV writer that streams every MAVLink event to a single ``log.csv``.

One row per MAVLink message, with columns being the **union** of all
ardupilotmega message-type schemas plus MAV/IP/UDP/TCP headers,
``frame_timestamp``, and ``attack_type``. Cells absent for a given row's
message type are written as ``"null"`` (dvdsh ``clean_arr_csv`` convention).

The writer is intentionally agnostic of attack types, message types, and
training shapes — it just records what happened, when, and whether the
``LabelLookup`` flagged it. Sorting, filtering, windowing, and per-window
label derivation all happen downstream in the training pipeline.

Rows are written in **arrival order**. Training-time consumers sort by the
``timestamp`` column (millisecond integer) before windowing.

The column order is built deterministically from
:data:`~swarm.dataset.mavlink_schema.SCHEMA`, so a live MAVLink consumer at
inference time can reuse the identical layout in-memory.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

from .dvdsh_compat import (
    IP_HEADER_FIELDS,
    MAV_HEADER_FIELDS,
    TCP_HEADER_FIELDS,
    UDP_HEADER_FIELDS,
)
from .labels import LabelLookup
from .mavlink_schema import SCHEMA
from .mavlink_schema import lookup as schema_lookup

# MAVLink v2 start-of-frame byte.
_MAV2_MAGIC: int = 253


# ---------------------------------------------------------------------------
# Cell + header helpers
# ---------------------------------------------------------------------------


def _format_cell(value: Any) -> str:
    """Convert *value* to a CSV-safe cell using dvdsh ``clean_arr_csv`` rules.

    Args:
        value: Raw Python value from a MAVLink field or packet header.

    Returns:
        A string containing no unescaped commas or newlines.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return ";".join(_format_cell(v) for v in value)
    if isinstance(value, bytes):
        return ";".join(str(b) for b in value)
    return str(value).replace(",", ";").replace("\n", " ")


def _mav_header_value(mav_msg: Any, name: str) -> Any:
    """Extract one MAV header field by dvdsh-format name.

    Args:
        mav_msg: A pymavlink ``MAVLink_message`` instance.
        name: One of the dvdsh ``MAV_HEADER_FIELDS`` names
            (e.g. ``"payloadLength"``, ``"incompatibilityFlags"``).

    Returns:
        The header field value, or ``None`` when unavailable.
    """
    hdr = getattr(mav_msg, "_header", None)
    match name:
        case "magic":
            return _MAV2_MAGIC
        case "payloadLength":
            if hdr is not None and hasattr(hdr, "mlen"):
                return hdr.mlen
            buf = getattr(mav_msg, "_msgbuf", None)
            return len(buf) if buf is not None else None
        case "incompatibilityFlags":
            return getattr(hdr, "incompat_flags", 0) if hdr is not None else 0
        case "compatibilityFlags":
            return getattr(hdr, "compat_flags", 0) if hdr is not None else 0
        case "seq":
            return getattr(hdr, "seq", None) if hdr is not None else None
        case "sysid":
            return getattr(hdr, "srcSystem", None) if hdr is not None else None
        case "compid":
            return getattr(hdr, "srcComponent", None) if hdr is not None else None
        case "msgid":
            return mav_msg.get_msgId()
        case "checksum":
            get_crc = getattr(mav_msg, "get_crc", None)
            return get_crc() if callable(get_crc) else None
        case "signature":
            return None
        case _:
            return None


def _build_payload_union() -> list[str]:
    """Return the sorted union of payload field names, deduplicated against
    fixed column names.

    Some MAVLink payloads use field names that collide with MAV-header
    columns (e.g. MISSION_ITEM has a ``seq`` field, but ``seq`` is also a MAV
    header column). To keep the CSV header unique, payload fields whose name
    is already reserved by a fixed column are excluded from the union.

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


# ---------------------------------------------------------------------------
# PacketWriter
# ---------------------------------------------------------------------------


class PacketWriter:
    """Streams every MAVLink event to a single ``log.csv`` file.

    Thread-safe: a single :class:`threading.Lock` serialises the per-row
    ``write`` call so that lines from multiple drone-listener threads do not
    interleave.

    Attributes:
        output_path: Destination path for the unified log CSV.
        labels: Label resolver consulted for each row's ``attack_type``.
        sim_uuid: Run identifier inserted as the ``sim_uuid`` column.
        counts: Mapping from MAVLink message-type name to rows written.
    """

    def __init__(
        self,
        output_path: Path,
        labels: LabelLookup,
        sim_uuid: str,
    ) -> None:
        """Open *output_path* and write the union-schema header row.

        Args:
            output_path: Destination CSV path (e.g. ``<run_dir>/log.csv``).
            labels: :class:`~swarm.dataset.labels.LabelLookup` for row annotation.
            sim_uuid: Run identifier string inserted on every row.
        """
        self.output_path = output_path
        self.labels = labels
        self.sim_uuid = sim_uuid
        self.counts: dict[str, int] = {}

        self._payload_fields = _build_payload_union()
        self._columns = self._build_columns()
        self._col_index: dict[str, int] = {c: i for i, c in enumerate(self._columns)}
        self._lock = threading.Lock()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: io.TextIOWrapper = open(output_path, "w", buffering=1 << 20)
        self._fh.write(",".join(self._columns) + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(
        self,
        mav_msg: Any,
        ip_fields: dict[str, Any],
        udp_fields: dict[str, Any] | None,
        tcp_fields: dict[str, Any] | None,
        frame_epoch: float,
        drone_id: int | None,
    ) -> None:
        """Write one MAVLink message as a row in the unified log.

        Args:
            mav_msg: A pymavlink ``MAVLink_message`` instance.
            ip_fields: Dict keyed by unprefixed
                :data:`~swarm.dataset.dvdsh_compat.IP_HEADER_FIELDS` names.
            udp_fields: Dict keyed by ``UDP_HEADER_FIELDS`` names, or ``None``.
            tcp_fields: Dict keyed by ``TCP_HEADER_FIELDS`` names, or ``None``.
            frame_epoch: UNIX timestamp of the captured frame.
            drone_id: MAVLink system-id of the originating drone, or ``None``.
        """
        idx = self._col_index
        cells: list[str] = ["null"] * len(self._columns)

        type_name: str = mav_msg.get_type()
        cells[idx["mav_packet_type"]] = _format_cell(type_name)
        cells[idx["sim_uuid"]] = self.sim_uuid
        cells[idx["timestamp"]] = str(int(frame_epoch * 1000))

        for name in MAV_HEADER_FIELDS:
            cells[idx[name]] = _format_cell(_mav_header_value(mav_msg, name))

        schema = schema_lookup(mav_msg.get_msgId())
        payload_fields: list[str] = schema[1] if schema is not None else []
        for name in payload_fields:
            col = idx.get(name)
            if col is not None:
                cells[col] = _format_cell(getattr(mav_msg, name, None))

        for name in IP_HEADER_FIELDS:
            cells[idx[f"ip_{name}"]] = _format_cell(ip_fields.get(name))

        if udp_fields is not None:
            for name in UDP_HEADER_FIELDS:
                cells[idx[f"udp_{name}"]] = _format_cell(udp_fields.get(name))

        if tcp_fields is not None:
            for name in TCP_HEADER_FIELDS:
                cells[idx[f"tcp_{name}"]] = _format_cell(tcp_fields.get(name))

        cells[idx["frame_timestamp"]] = repr(frame_epoch)
        cells[idx["attack_type"]] = _format_cell(self.labels.lookup(frame_epoch, drone_id))

        line = ",".join(cells) + "\n"
        with self._lock:
            self._fh.write(line)

        self.counts[type_name] = self.counts.get(type_name, 0) + 1

    def close(self) -> None:
        """Flush and close the log file. Safe to call multiple times."""
        if self._fh is not None and not self._fh.closed:
            try:
                self._fh.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_columns(self) -> list[str]:
        """Build the full ordered column list for ``log.csv``.

        Returns:
            List of column names in the fixed dvdsh-compatible order.
        """
        cols: list[str] = ["mav_packet_type", "sim_uuid", "timestamp"]
        cols.extend(MAV_HEADER_FIELDS)
        cols.extend(self._payload_fields)
        cols.extend(f"ip_{f}" for f in IP_HEADER_FIELDS)
        cols.extend(f"udp_{f}" for f in UDP_HEADER_FIELDS)
        cols.extend(f"tcp_{f}" for f in TCP_HEADER_FIELDS)
        cols.append("frame_timestamp")
        cols.append("attack_type")
        return cols
