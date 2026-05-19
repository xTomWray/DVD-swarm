"""Live packet capture feeding MAVLink messages to a :class:`PacketWriter`.

Uses scapy's ``AsyncSniffer`` to capture UDP/TCP traffic on nominated ports,
parse embedded MAVLink frames with pymavlink, and write each decoded message
via :class:`~swarm.dataset.packet_writer.PacketWriter`.

Drone IDs are inferred from the ``10.13.<N>.<x>`` subnet convention: the
``<N>`` octet of whichever endpoint (src or dst) matches this pattern becomes
the ``drone_id`` passed to the writer.

Usage::

    writer = PacketWriter(output_dir, labels)
    sniffer = start_sniffer("any", [14540, 14550, 5760], writer)
    # ... run experiment ...
    sniffer.stop()
    writer.close()
"""

from __future__ import annotations

import re
from threading import Lock
from typing import TYPE_CHECKING, Any

from pymavlink.mavutil import mavlink

from .dvdsh_compat import IP_HEADER_FIELDS, TCP_HEADER_FIELDS, UDP_HEADER_FIELDS
from .packet_writer import PacketWriter

if TYPE_CHECKING:
    pass  # AsyncSniffer imported at runtime to avoid scapy startup cost

# ---------------------------------------------------------------------------
# Regex for extracting drone id from IP address.
# ---------------------------------------------------------------------------
_DRONE_IP_RE = re.compile(r"^10\.13\.(\d+)\.\d+$")

# ---------------------------------------------------------------------------
# Per-(src_ip, src_port) MAVLink parser cache.
# ---------------------------------------------------------------------------
_parser_lock = Lock()
_parsers: dict[tuple[str, int], Any] = {}


def _get_parser(src_ip: str, src_port: int) -> Any:
    """Return a cached per-flow pymavlink parser, creating one if needed.

    Args:
        src_ip: Source IP address string.
        src_port: Source UDP/TCP port number.

    Returns:
        A ``pymavlink.mavutil.mavlink.MAVLink`` parser instance.
    """
    key = (src_ip, src_port)
    with _parser_lock:
        if key not in _parsers:
            _parsers[key] = mavlink.MAVLink(None)
        return _parsers[key]


# ---------------------------------------------------------------------------
# IP field extraction helpers
# ---------------------------------------------------------------------------

def _ip_fields(ip: Any) -> dict[str, Any]:
    """Build an IP header fields dict from a scapy IP layer.

    Field names match :data:`~swarm.dataset.dvdsh_compat.IP_HEADER_FIELDS`.

    Args:
        ip: Scapy ``IP`` layer object.

    Returns:
        Dict keyed by unprefixed IP_HEADER_FIELDS names.
    """
    payload_hex: str = bytes(ip.payload).hex()
    return {
        "version": ip.version,
        "hdr_len": ip.ihl * 4,
        "tos": ip.tos,
        "len": ip.len,
        "id": ip.id,
        "flags": int(ip.flags),
        "frag_offset": ip.frag,
        "ttl": ip.ttl,
        "proto": ip.proto,
        "checksum": ip.chksum,
        "checksum_status": None,
        "src": ip.src,
        "src_host": ip.src,
        "addr": ip.src,   # dvdsh idiom: src duplicated as 'addr'
        "host": ip.src,
        "dst": ip.dst,
        "dst_host": ip.dst,
        "payload": payload_hex,
    }


def _udp_fields(udp: Any) -> dict[str, Any]:
    """Build a UDP header fields dict from a scapy UDP layer.

    Field names match :data:`~swarm.dataset.dvdsh_compat.UDP_HEADER_FIELDS`.

    Args:
        udp: Scapy ``UDP`` layer object.

    Returns:
        Dict keyed by unprefixed UDP_HEADER_FIELDS names.
    """
    return {
        "srcport": udp.sport,
        "dstport": udp.dport,
        "length": udp.len,
        "checksum": udp.chksum,
        "checksum_status": None,
        "payload": bytes(udp.payload).hex(),
        "text": None,
    }


def _tcp_fields(tcp: Any) -> dict[str, Any]:
    """Build a TCP header fields dict from a scapy TCP layer.

    Field names match :data:`~swarm.dataset.dvdsh_compat.TCP_HEADER_FIELDS`.

    Args:
        tcp: Scapy ``TCP`` layer object.

    Returns:
        Dict keyed by unprefixed TCP_HEADER_FIELDS names.
    """
    return {
        "srcport": tcp.sport,
        "dstport": tcp.dport,
        "seq": tcp.seq,
        "ack": tcp.ack,
        "hdr_len": tcp.dataofs * 4,
        "flags": int(tcp.flags),
        "flags_str": str(tcp.flags),
        "window_size": tcp.window,
        "checksum": tcp.chksum,
        "checksum_status": None,
        "urgent_pointer": tcp.urgptr,
        "options": str(tcp.options),
        "options_nop": None,
        "options_timestamp": None,
        "payload": bytes(tcp.payload).hex(),
        "text": None,
    }


# ---------------------------------------------------------------------------
# Drone-id extraction
# ---------------------------------------------------------------------------

def _extract_drone_id(src_ip: str, dst_ip: str) -> int | None:
    """Extract the drone instance number from IP addresses.

    Checks *src_ip* first, then *dst_ip*, against the ``10.13.<N>.<x>``
    convention.

    Args:
        src_ip: Source IP address string.
        dst_ip: Destination IP address string.

    Returns:
        The integer ``<N>`` component if found, otherwise ``None``.
    """
    for addr in (src_ip, dst_ip):
        m = _DRONE_IP_RE.match(addr)
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def start_sniffer(
    iface: str,
    ports: list[int],
    writer: PacketWriter,
) -> Any:
    """Start an async packet sniffer and return the running ``AsyncSniffer``.

    Captures UDP and TCP traffic on *ports* from *iface*, decodes embedded
    MAVLink frames, and calls ``writer.handle`` for each complete message.
    Non-IP packets (e.g. ARP) are silently skipped.

    The caller is responsible for calling ``sniffer.stop()`` when capture
    should end.

    Args:
        iface: Network interface name to sniff (e.g. ``"any"`` or ``"eth0"``).
        ports: List of UDP/TCP port numbers to capture.
        writer: :class:`~swarm.dataset.packet_writer.PacketWriter` instance that
            receives each decoded MAVLink message.

    Returns:
        A started scapy ``AsyncSniffer`` instance.
    """
    from scapy.all import AsyncSniffer, IP, TCP, UDP  # local import: avoid scapy init at module level

    bpf_parts = [f"udp port {p} or tcp port {p}" for p in ports]
    bpf_filter = " or ".join(bpf_parts)

    def _prn(pkt: Any) -> None:
        """Process one captured packet.

        Args:
            pkt: Scapy packet object.
        """
        # Silently skip non-IP frames.
        if IP not in pkt:
            return

        ip = pkt[IP]
        ip_hdr = _ip_fields(ip)
        frame_epoch: float = float(pkt.time)

        src_ip: str = ip.src
        dst_ip: str = ip.dst
        drone_id = _extract_drone_id(src_ip, dst_ip)

        if UDP in pkt:
            udp = pkt[UDP]
            udp_hdr = _udp_fields(udp)
            tcp_hdr = None
            payload_bytes = bytes(udp.payload)
            src_port: int = udp.sport
        elif TCP in pkt:
            tcp = pkt[TCP]
            udp_hdr = None
            tcp_hdr = _tcp_fields(tcp)
            payload_bytes = bytes(tcp.payload)
            src_port = tcp.sport
        else:
            return

        if not payload_bytes:
            return

        parser = _get_parser(src_ip, src_port)

        # Feed each byte through the per-flow MAVLink parser.
        with _parser_lock:
            for byte in payload_bytes:
                result = parser.parse_char(bytes([byte]))
                if result is None:
                    continue
                # parse_char may return a list or a single message.
                msgs = result if isinstance(result, list) else [result]
                for msg in msgs:
                    if msg is None:
                        continue
                    try:
                        writer.handle(
                            mav_msg=msg,
                            ip_fields=ip_hdr,
                            udp_fields=udp_hdr,
                            tcp_fields=tcp_hdr,
                            frame_epoch=frame_epoch,
                            drone_id=drone_id,
                        )
                    except Exception:
                        # Never let a write error kill the capture loop.
                        pass

    sniffer = AsyncSniffer(iface=iface, filter=bpf_filter, prn=_prn, store=False)
    sniffer.start()
    return sniffer
