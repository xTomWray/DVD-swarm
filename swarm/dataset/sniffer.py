"""Live packet capture feeding MAVLink messages to a :class:`PacketWriter`.

Spawns ``tcpdump -i <iface>`` as a subprocess writing pcap to stdout, then
parses frames with scapy's ``PcapReader``. This is the only reliable way to
capture from a Linux ``any`` pseudo-interface — scapy's own ``get_if_list()``
does not enumerate docker per-network bridges (``br-*``), and binding to a
list of veth endpoints misses traffic that flows entirely through a bridge.

The same physical packet typically appears 2–3 times on ``-i any`` (once per
veth endpoint of a bridge, plus once on the bridge itself), so the reader
fingerprints each frame by ``(src, dst, ip.id, proto, sport, dport, len)``
and skips duplicates within a short window.

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

import collections
import logging
import re
import shutil
import subprocess
import threading
from threading import Lock
from typing import Any

from pymavlink.mavutil import mavlink

from .dvdsh_compat import IP_HEADER_FIELDS, TCP_HEADER_FIELDS, UDP_HEADER_FIELDS
from .packet_writer import PacketWriter

log = logging.getLogger("swarm.dataset.sniffer")

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

def _packet_fingerprint(pkt: Any, IP: Any, UDP: Any, TCP: Any) -> tuple[Any, ...] | None:
    """Compute a dedup key for a captured frame.

    ``-i any`` shows the same wire packet multiple times (once per veth
    endpoint plus once on the docker bridge). The IP id is set by the
    sending host and persists across all observation points, so combining
    it with src/dst/proto and L4 port info uniquely identifies a logical
    packet within a short time window.

    Args:
        pkt: Scapy packet object.
        IP: scapy IP layer class.
        UDP: scapy UDP layer class.
        TCP: scapy TCP layer class.

    Returns:
        A hashable tuple, or ``None`` for non-IP frames.
    """
    if IP not in pkt:
        return None
    ip = pkt[IP]
    if UDP in pkt:
        u = pkt[UDP]
        return (ip.src, ip.dst, ip.id, ip.proto, u.sport, u.dport, u.len)
    if TCP in pkt:
        t = pkt[TCP]
        return (ip.src, ip.dst, ip.id, ip.proto, t.sport, t.dport, t.seq, t.ack)
    return (ip.src, ip.dst, ip.id, ip.proto)


def _process_packet(pkt: Any, writer: PacketWriter, IP: Any, UDP: Any, TCP: Any) -> None:
    """Feed one captured frame's payload through MAVLink parsing into *writer*.

    Args:
        pkt: Scapy packet object.
        writer: :class:`PacketWriter` to receive decoded messages.
        IP: scapy IP layer class.
        UDP: scapy UDP layer class.
        TCP: scapy TCP layer class.
    """
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

    for byte in payload_bytes:
        result = parser.parse_char(bytes([byte]))
        if result is None:
            continue
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
                pass


class _SnifferHandle:
    """Handle returned by :func:`start_sniffer`.

    The capture runs as a tcpdump subprocess writing pcap directly to
    disk. ``stop()`` terminates tcpdump and then replays the saved file
    through the MAVLink parser. Parsing offline avoids the live-pipe
    starvation we hit when reading ``tcpdump -w -`` while the
    orchestrator was busy executing GCS stages.
    """

    def __init__(
        self,
        proc: subprocess.Popen[bytes],
        pcap_path: Any,
        writer: PacketWriter,
    ) -> None:
        self._proc = proc
        self._pcap_path = pcap_path
        self._writer = writer
        self._stopped = False

    def stop(self) -> None:
        """Stop capture, then parse the saved pcap into the writer.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._stopped:
            return
        self._stopped = True

        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=2)
        except Exception as exc:
            log.warning("tcpdump terminate failed: %s", exc)

        if self._proc.stderr is not None:
            err = self._proc.stderr.read().decode(errors="replace").strip()
            if err:
                log.info("tcpdump stderr: %s", err)

        self._replay()

    def _replay(self) -> None:
        """Read the saved pcap file and feed every packet to the writer."""
        from scapy.layers.inet import IP, TCP, UDP
        from scapy.utils import PcapReader

        if not self._pcap_path.exists():
            log.warning("Capture file %s missing; nothing to replay.", self._pcap_path)
            return

        total = 0
        deduped = 0
        seen: collections.OrderedDict[tuple[Any, ...], None] = collections.OrderedDict()
        seen_cap = 65536

        with PcapReader(str(self._pcap_path)) as pcap:
            for pkt in pcap:
                total += 1
                fp = _packet_fingerprint(pkt, IP, UDP, TCP)
                if fp is not None:
                    if fp in seen:
                        deduped += 1
                        continue
                    seen[fp] = None
                    if len(seen) > seen_cap:
                        seen.popitem(last=False)
                _process_packet(pkt, self._writer, IP, UDP, TCP)

        log.info(
            "Replay complete: %d frames in pcap, %d duplicates skipped (%s)",
            total,
            deduped,
            self._pcap_path,
        )


def start_sniffer(
    iface: str,
    ports: list[int],
    writer: PacketWriter,
    pcap_path: Any,
) -> _SnifferHandle:
    """Start tcpdump capturing to *pcap_path*; replay on stop.

    The tcpdump process writes pcap to a file rather than to a pipe, so
    the parent process cannot starve the capture by failing to drain a
    pipe. Packet decoding happens offline in :meth:`_SnifferHandle.stop`
    after the capture window closes — labels are timestamp-based so this
    delay does not affect the final CSV contents. The raw pcap is left in
    place as a debug artifact.

    Args:
        iface: Interface name to sniff. ``"any"`` is preferred on Linux.
        ports: UDP/TCP port numbers to capture.
        writer: :class:`PacketWriter` instance that receives decoded
            MAVLink messages during the replay phase.
        pcap_path: Filesystem path the pcap will be written to.

    Returns:
        A :class:`_SnifferHandle` exposing ``.stop()``.

    Raises:
        RuntimeError: When tcpdump is missing or exits immediately.
    """
    if shutil.which("tcpdump") is None:
        raise RuntimeError("tcpdump not found on PATH — install it (apt-get install tcpdump)")

    bpf_parts = [f"udp port {p} or tcp port {p}" for p in ports]
    bpf_filter = " or ".join(bpf_parts)

    pcap_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "tcpdump",
        "-i", iface,
        "-U",            # packet-buffered writes
        "-n",            # no DNS
        "-s", "0",       # full snaplen
        "-w", str(pcap_path),
        bpf_filter,
    ]
    log.info("Starting capture: %s", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Brief pause to catch an immediate crash (bad iface, perms, BPF syntax).
    import time as _time
    _time.sleep(0.5)
    if proc.poll() is not None:
        err = b""
        if proc.stderr is not None:
            err = proc.stderr.read()
        raise RuntimeError(
            f"tcpdump exited immediately (code {proc.returncode}): "
            f"{err.decode(errors='replace').strip()}"
        )

    return _SnifferHandle(proc, pcap_path, writer)
