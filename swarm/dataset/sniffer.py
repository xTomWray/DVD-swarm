"""Real-time MAVLink listener feeding messages to a :class:`PacketWriter`.

Connects to each drone's mavlink-routerd TCP endpoint (``10.13.<N>.3:5760``)
directly from the host.  All messages are received in real time — including
attack packets injected via UDP 14550, which mavlink-routerd routes to every
connected TCP client.

This replaces the earlier tcpdump → scapy → offline-replay pipeline, which
suffered two fatal flaws on modern Linux:

* Scapy probes every network interface on import.  After ``docker compose up``
  there are dozens of ``br-*`` / ``veth`` interfaces, causing a multi-minute
  freeze at replay time.
* ``tcpdump -i any`` now writes DLT_LINUX_SLL2 (link type 276).  Importing
  only ``scapy.layers.inet`` does not register the SLL2 dissector, so
  ``IP in pkt`` returns False for every frame, yielding near-zero CSV rows.

Usage::

    writer = PacketWriter(output_dir, labels)
    sniffer = start_sniffer(list(range(1, N + 1)), writer)
    # ... run experiment ...
    sniffer.stop()
    writer.close()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from pymavlink import mavutil

from .dvdsh_compat import IP_HEADER_FIELDS, TCP_HEADER_FIELDS, UDP_HEADER_FIELDS
from .packet_writer import PacketWriter

log = logging.getLogger("swarm.dataset.sniffer")

# TCP port that mavlink-routerd exposes on each companion computer.
_MAVLINK_TCP_PORT = 5760


def _ip_fields_for_drone(drone_id: int) -> dict[str, Any]:
    """Build a partial IP header dict for a drone's companion connection.

    Only fields derivable from the connection address are populated; the rest
    are ``None`` (rendered as ``"null"`` in the CSV).

    Args:
        drone_id: Drone instance number ``N`` (subnet octet in ``10.13.N.x``).

    Returns:
        Dict keyed by unprefixed :data:`~swarm.dataset.dvdsh_compat.IP_HEADER_FIELDS` names.
    """
    companion_ip = f"10.13.{drone_id}.3"
    host_ip = f"10.13.{drone_id}.1"  # Docker bridge gateway seen from host
    return {
        "version": 4,
        "hdr_len": None,
        "tos": None,
        "len": None,
        "id": None,
        "flags": None,
        "frag_offset": None,
        "ttl": None,
        "proto": 6,  # TCP
        "checksum": None,
        "checksum_status": None,
        "src": companion_ip,
        "src_host": companion_ip,
        "addr": companion_ip,
        "host": companion_ip,
        "dst": host_ip,
        "dst_host": host_ip,
        "payload": None,
    }


def _null_transport_fields(fields: list[str]) -> dict[str, Any]:
    return {f: None for f in fields}


# Pre-build null UDP/TCP dicts once — shared across all listener threads.
_NULL_UDP: dict[str, Any] = _null_transport_fields(UDP_HEADER_FIELDS)
_NULL_TCP: dict[str, Any] = _null_transport_fields(TCP_HEADER_FIELDS)


class _DroneListener:
    """Listens on one pymavlink TCP connection and feeds messages to writer."""

    def __init__(
        self,
        drone_id: int,
        writer: PacketWriter,
        stop_event: threading.Event,
    ) -> None:
        self._drone_id = drone_id
        self._writer = writer
        self._stop_event = stop_event
        self._conn_str = f"tcp:10.13.{drone_id}.3:{_MAVLINK_TCP_PORT}"
        self._ip_hdr = _ip_fields_for_drone(drone_id)

    def run(self) -> None:
        """Connect and loop until stop_event is set."""
        try:
            conn = mavutil.mavlink_connection(
                self._conn_str,
                dialect="ardupilotmega",
                autoreconnect=True,
            )
        except Exception as exc:
            log.error("Drone %d: failed to open connection %s: %s", self._drone_id, self._conn_str, exc)
            return

        log.info("Listener connected: drone %d (%s)", self._drone_id, self._conn_str)

        while not self._stop_event.is_set():
            msg = conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            if msg.get_type() == "BAD_DATA":
                continue

            frame_epoch = time.time()
            try:
                self._writer.handle(
                    mav_msg=msg,
                    ip_fields=self._ip_hdr,
                    udp_fields=_NULL_UDP,
                    tcp_fields=None,
                    frame_epoch=frame_epoch,
                    drone_id=self._drone_id,
                )
            except Exception:
                log.debug(
                    "writer.handle failed for drone %d msg %s",
                    self._drone_id,
                    msg.get_type(),
                    exc_info=True,
                )

        try:
            conn.close()
        except Exception:
            pass
        log.info("Listener stopped: drone %d", self._drone_id)


class _SnifferHandle:
    """Handle returned by :func:`start_sniffer`."""

    def __init__(
        self,
        threads: list[threading.Thread],
        stop_event: threading.Event,
    ) -> None:
        self._threads = threads
        self._stop_event = stop_event

    def stop(self) -> None:
        """Signal all listener threads to stop and wait for them to finish.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=6)


def start_sniffer(
    drone_ids: list[int],
    writer: PacketWriter,
) -> _SnifferHandle:
    """Start real-time MAVLink listeners for each drone.

    One listener thread per drone connects to ``tcp:10.13.<N>.3:5760``
    (mavlink-routerd's TCP server on the companion computer).  All messages
    flowing through mavlink-routerd — normal telemetry *and* attack packets
    injected via UDP 14550 — arrive on this connection and are written to
    *writer* immediately, so label timestamps match perfectly with no offline
    replay step.

    Args:
        drone_ids: Ordered list of drone instance numbers to monitor.
        writer: :class:`PacketWriter` instance that receives decoded messages.

    Returns:
        A :class:`_SnifferHandle` exposing ``.stop()``.
    """
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    for n in drone_ids:
        listener = _DroneListener(n, writer, stop_event)
        t = threading.Thread(
            target=listener.run,
            daemon=True,
            name=f"sniffer-drone-{n}",
        )
        t.start()
        threads.append(t)
        log.info("Started listener thread for drone %d (tcp:10.13.%d.3:%d)", n, n, _MAVLINK_TCP_PORT)

    return _SnifferHandle(threads, stop_event)
