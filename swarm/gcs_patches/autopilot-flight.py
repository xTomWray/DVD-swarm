"""Patched copy of ground-control-station/stages/autopilot-flight.py.

Bind-mounted over /opt/gcs/stages/autopilot-flight.py at runtime by
swarm/generate_swarm.py so the upstream DVD source stays untouched.

Differences from upstream:

  * MISSION_REQUEST timeout: env var MISSION_REQUEST_TIMEOUT (default 30s)
    instead of the hard-coded 5s that caused random mid-upload failures
    at N>=10 under host CPU caps.
  * MISSION_ACK timeout: env var MISSION_ACK_TIMEOUT (default 30s) instead
    of 10s.
  * Outer retry loop honoring MISSION_UPLOAD_RETRIES (default 3 retries,
    i.e. up to 4 attempts). Each retry clears the FC's partial mission
    before restarting from waypoint 0.

The MAVLink upload protocol itself is byte-identical to upstream so this
file remains a drop-in replacement.
"""

import os
import sys
import time

from pymavlink import mavutil

_instance = os.getenv("SWARM_INSTANCE", "0")
_REQUEST_TIMEOUT = float(os.getenv("MISSION_REQUEST_TIMEOUT", "30"))
_ACK_TIMEOUT = float(os.getenv("MISSION_ACK_TIMEOUT", "30"))
_RETRIES = int(os.getenv("MISSION_UPLOAD_RETRIES", "3"))

# Use TCP 5760 on companion — UDP 14550 is held by mavproxy.py at GCS startup
connection_string = f"tcp:10.13.{_instance}.3:5760"


def read_waypoints(filename):
    waypoints = []
    with open(filename) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                lat, lon, alt = map(float, line.split(","))
                waypoints.append((lat, lon, alt))
    return waypoints


def connect_to_drone(connection_string, timeout=30, retries=5):
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} of {retries} to connect to drone")
            master = mavutil.mavlink_connection(connection_string)
            start_time = time.time()

            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError("Timed out waiting for heartbeat")

                msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
                if msg:
                    print("Connected to drone")
                    return master
                else:
                    print("Waiting for heartbeat...")

        except TimeoutError as e:
            print(str(e))
        except Exception as e:
            print(f"Unexpected error: {str(e)}")

        time.sleep(5)

    raise ConnectionError("Failed to connect to the drone after multiple attempts")


def upload_once(master, waypoints):
    """Run one complete MISSION_COUNT -> MISSION_REQUEST loop -> MISSION_ACK.

    Raises TimeoutError / RuntimeError on any failure so the outer retry
    loop can clear the FC's partial state and try again.
    """
    master.waypoint_clear_all_send()
    master.mav.mission_count_send(
        master.target_system, master.target_component, len(waypoints)
    )

    uploaded = 0
    while uploaded < len(waypoints):
        msg = master.recv_match(
            type=["MISSION_REQUEST"], blocking=True, timeout=_REQUEST_TIMEOUT
        )
        if msg is None:
            raise TimeoutError(
                f"MISSION_REQUEST timeout after {_REQUEST_TIMEOUT:.0f}s "
                f"(uploaded {uploaded}/{len(waypoints)})"
            )
        seq = msg.seq
        if seq >= len(waypoints):
            raise RuntimeError(f"FC requested out-of-range seq {seq}")
        lat, lon, alt = waypoints[seq]
        master.mav.mission_item_int_send(
            master.target_system,
            master.target_component,
            seq,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            0,
            0,
            0,
            0,
            0,
            0,
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
        )
        uploaded = seq + 1

    ack = master.recv_match(
        type=["MISSION_ACK"], blocking=True, timeout=_ACK_TIMEOUT
    )
    if ack is None:
        raise TimeoutError(f"MISSION_ACK missing after {_ACK_TIMEOUT:.0f}s")
    if ack.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
        raise RuntimeError(f"MISSION_ACK type={ack.type} (not ACCEPTED)")


waypoints = read_waypoints("/opt/gcs/missions/waypoints_custom_zigzag_square.txt")
master = connect_to_drone(connection_string)

attempts = _RETRIES + 1
last_err: Exception | None = None
for attempt in range(1, attempts + 1):
    try:
        upload_once(master, waypoints)
        master.set_mode_auto()
        print(f"AUTO mode set, mission started (attempt {attempt}/{attempts})")
        sys.exit(0)
    except (TimeoutError, RuntimeError) as exc:
        last_err = exc
        print(f"Mission upload attempt {attempt}/{attempts} failed: {exc}")
        try:
            master.waypoint_clear_all_send()
        except Exception as clear_exc:
            print(f"  clear-all failed: {clear_exc}")
        time.sleep(1.0)

print(f"Mission upload failed after {attempts} attempts: {last_err}")
sys.exit(1)
