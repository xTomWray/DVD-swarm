"""CSV writer for MAVLink packet captures compatible with dvdsh DUMP_CSV/JOIN_CSV.

Each unique MAVLink message type is written to its own ``<TYPE>.csv`` file
under *output_dir*.  Column ordering matches dvdsh exactly:

    mav_packet_type, <MAV_HEADER_FIELDS>, <payload_fields>,
    ip_<IP_HEADER_FIELDS>, udp_<UDP_HEADER_FIELDS>|tcp_<TCP_HEADER_FIELDS>,
    frame_timestamp, attack_type

Values are formatted with the dvdsh ``clean_arr_csv`` convention:
- ``None``         → ``"null"``
- ``list``/``tuple``/``bytes`` → ``";"``-joined elements
- ``bool``         → ``"True"``/``"False"``
- Everything else  → ``str(value)``

Commas inside cell values are replaced with ``";"`` (no quoting, dvdsh-compat).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from .dvdsh_compat import (
    IP_HEADER_FIELDS,
    MAV_HEADER_FIELDS,
    TCP_HEADER_FIELDS,
    UDP_HEADER_FIELDS,
)
from .labels import LabelLookup
from .mavlink_schema import lookup as schema_lookup

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# MAVLink v2 start-of-frame byte.
_MAV2_MAGIC: int = 253


def _mav_header_value(mav_msg: Any, name: str) -> Any:
    """Extract one MAV_HEADER_FIELDS value from *mav_msg*.

    Args:
        mav_msg: A pymavlink ``MAVLink_message`` instance.
        name: One of the dvdsh ``MAV_HEADER_FIELDS`` names.

    Returns:
        The header field value, or ``None`` when unavailable.
    """
    hdr = getattr(mav_msg, "_header", None)
    match name:
        case "magic":
            return _MAV2_MAGIC
        case "len":
            if hdr is not None and hasattr(hdr, "mlen"):
                return hdr.mlen
            buf = getattr(mav_msg, "_msgbuf", None)
            return len(buf) if buf is not None else None
        case "incompat_flags":
            return getattr(hdr, "incompat_flags", 0) if hdr is not None else 0
        case "compat_flags":
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


def _format_cell(value: Any) -> str:
    """Convert *value* to a CSV-safe cell string using dvdsh conventions.

    Args:
        value: Raw Python value from a MAVLink field or packet header.

    Returns:
        A string with no unescaped commas or newlines.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return ";".join(_format_cell(v) for v in value)
    if isinstance(value, bytes):
        return ";".join(str(b) for b in value)
    raw = str(value)
    # Replace comma + newline to keep CSV parseable (dvdsh clean_arr_csv).
    return raw.replace(",", ";").replace("\n", " ")


# ---------------------------------------------------------------------------
# Header memoisation
# ---------------------------------------------------------------------------

def _build_header(type_name: str, payload_fields: list[str]) -> str:
    """Build the CSV header line for a given message type.

    Args:
        type_name: MAVLink message type name (e.g. ``"ATTITUDE"``).
        payload_fields: Ordered list of payload field names for this type.

    Returns:
        The full header row string, including trailing newline.
    """
    cols: list[str] = ["mav_packet_type"]
    cols.extend(MAV_HEADER_FIELDS)
    cols.extend(payload_fields)
    cols.extend(f"ip_{f}" for f in IP_HEADER_FIELDS)
    cols.extend(f"udp_{f}" for f in UDP_HEADER_FIELDS)
    cols.extend(f"tcp_{f}" for f in TCP_HEADER_FIELDS)
    cols.append("frame_timestamp")
    cols.append("attack_type")
    return ",".join(cols) + "\n"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class PacketWriter:
    """Writes decoded MAVLink packets to per-type CSV files.

    One ``<TYPE>.csv`` file is created under *output_dir* for each unique
    MAVLink message type seen during the capture.  File handles are cached
    and buffered for performance; call :meth:`close` when capture ends.

    Attributes:
        output_dir: Directory where CSV files are written.
        labels: Label resolver for annotating each row with ``attack_type``.
        counts: Mapping from message-type name to rows written so far.
    """

    def __init__(self, output_dir: Path, labels: LabelLookup) -> None:
        """Initialise the writer and create *output_dir* if necessary.

        Args:
            output_dir: Directory for CSV output files.
            labels: :class:`~swarm.dataset.labels.LabelLookup` for row annotation.
        """
        self.output_dir = output_dir
        self.labels = labels
        # type_name -> (file_handle, header_written)
        self._files: dict[str, tuple[io.TextIOWrapper, bool]] = {}
        self.counts: dict[str, int] = {}
        output_dir.mkdir(parents=True, exist_ok=True)

        # Memoised header strings keyed by type_name.
        self._headers: dict[str, str] = {}

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
        """Write one MAVLink message as a CSV row.

        Creates a new file for the message type if this is the first time it
        is seen, writing the header row first.

        Args:
            mav_msg: A pymavlink ``MAVLink_message`` instance.
            ip_fields: Dict keyed by :data:`~swarm.dataset.dvdsh_compat.IP_HEADER_FIELDS`
                names (without the ``ip_`` prefix).
            udp_fields: Dict keyed by
                :data:`~swarm.dataset.dvdsh_compat.UDP_HEADER_FIELDS` names, or
                ``None`` for TCP packets.
            tcp_fields: Dict keyed by
                :data:`~swarm.dataset.dvdsh_compat.TCP_HEADER_FIELDS` names, or
                ``None`` for UDP packets.
            frame_epoch: UNIX timestamp of the captured frame.
            drone_id: MAVLink system-id of the originating drone, or ``None``.
        """
        type_name: str = mav_msg.get_type()
        msg_id: int = mav_msg.get_msgId()

        schema = schema_lookup(msg_id)
        payload_fields: list[str] = schema[1] if schema is not None else []

        fh, header_written = self._get_file(type_name, payload_fields)

        row = self._build_row(
            mav_msg, type_name, payload_fields,
            ip_fields, udp_fields, tcp_fields,
            frame_epoch, drone_id,
        )
        fh.write(row)

        self.counts[type_name] = self.counts.get(type_name, 0) + 1

    def close(self) -> None:
        """Flush and close all open CSV file handles.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        for fh, _ in self._files.values():
            try:
                fh.close()
            except Exception:
                pass
        self._files.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_file(
        self, type_name: str, payload_fields: list[str]
    ) -> tuple[io.TextIOWrapper, bool]:
        """Return (file_handle, header_written) for *type_name*, creating if new.

        Args:
            type_name: MAVLink message type name.
            payload_fields: Payload field names used to build the header if new.

        Returns:
            A tuple of the open file handle and whether the header was already
            written (always ``True`` after this call returns).
        """
        if type_name in self._files:
            return self._files[type_name]

        path = self.output_dir / f"{type_name}.csv"
        fh: io.TextIOWrapper = open(path, "a", buffering=1 << 20)  # noqa: WPS515

        header = self._headers.get(type_name)
        if header is None:
            header = _build_header(type_name, payload_fields)
            self._headers[type_name] = header

        fh.write(header)
        self._files[type_name] = (fh, True)
        return fh, True

    def _build_row(
        self,
        mav_msg: Any,
        type_name: str,
        payload_fields: list[str],
        ip_fields: dict[str, Any],
        udp_fields: dict[str, Any] | None,
        tcp_fields: dict[str, Any] | None,
        frame_epoch: float,
        drone_id: int | None,
    ) -> str:
        """Assemble one CSV data row.

        Args:
            mav_msg: pymavlink message instance.
            type_name: Message type string (e.g. ``"ATTITUDE"``).
            payload_fields: Ordered payload field names.
            ip_fields: IP header value dict (no prefix).
            udp_fields: UDP header value dict, or ``None``.
            tcp_fields: TCP header value dict, or ``None``.
            frame_epoch: Frame UNIX timestamp.
            drone_id: Originating drone system-id, or ``None``.

        Returns:
            A complete CSV row string, including trailing newline.
        """
        cells: list[str] = []

        # mav_packet_type
        cells.append(_format_cell(type_name))

        # MAV header fields
        for name in MAV_HEADER_FIELDS:
            cells.append(_format_cell(_mav_header_value(mav_msg, name)))

        # Payload fields
        for name in payload_fields:
            cells.append(_format_cell(getattr(mav_msg, name, None)))

        # IP header fields
        for name in IP_HEADER_FIELDS:
            cells.append(_format_cell(ip_fields.get(name)))

        # UDP header fields (null when packet is TCP)
        for name in UDP_HEADER_FIELDS:
            value = udp_fields.get(name) if udp_fields is not None else None
            cells.append(_format_cell(value))

        # TCP header fields (null when packet is UDP)
        for name in TCP_HEADER_FIELDS:
            value = tcp_fields.get(name) if tcp_fields is not None else None
            cells.append(_format_cell(value))

        # frame_timestamp — full float precision
        cells.append(repr(frame_epoch))

        # attack_type label
        cells.append(_format_cell(self.labels.lookup(frame_epoch, drone_id)))

        return ",".join(cells) + "\n"
