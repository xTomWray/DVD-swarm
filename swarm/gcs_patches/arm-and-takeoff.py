"""Patched copy of ground-control-station/stages/arm-and-takeoff.py.

Bind-mounted over /opt/gcs/stages/arm-and-takeoff.py at runtime by
swarm/generate_swarm.py so the upstream DVD source stays untouched.

Differences from upstream:

  * ``wait_for_ekf_status()`` now honours ``EKF_WAIT_TIMEOUT`` (default 600s).
    The upstream version was unbounded — under CPU contention with N>=20
    drones, EKF convergence could hang indefinitely and the orchestrator
    would only notice when the HTTP request hit its own timeout.
  * Per-sub-step timing prints (``[stage2 instance=N] step=... elapsed=...s``)
    so we can pinpoint which step is slow when N is high. ``flush=True`` is
    required — print() in a container buffers indefinitely without it.

Functional behaviour is otherwise identical: same param injection, same
waypoint clearing, same GUIDED-mode / arm / takeoff sequence.
"""

import os
import time

from pymavlink import mavutil

_instance = os.getenv("SWARM_INSTANCE", "0")
_EKF_WAIT_TIMEOUT = float(os.getenv("EKF_WAIT_TIMEOUT", "600"))


# Per-sub-step timing — see autopilot-flight.py for matching pattern.
_t0 = time.monotonic()


def _step(label: str) -> None:
    global _t0
    elapsed = time.monotonic() - _t0
    print(f"[stage2 instance={_instance}] step={label} elapsed={elapsed:.2f}s", flush=True)
    _t0 = time.monotonic()


def wait_for_mode(master, mode):
    while True:
        msgs = master.recv_match(blocking=True)
        if msgs is not None and msgs.get_type() == "HEARTBEAT" and msgs.custom_mode == mode:
            break
        time.sleep(0.1)


def is_armed(heartbeat):
    return (heartbeat.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0


def wait_for_gps_fix(master, timeout=60):
    start_time = time.time()
    print("Waiting for GPS fix...", flush=True)

    while time.time() - start_time < timeout:
        msg = master.recv_match(type="GPS_RAW_INT", blocking=True, timeout=1)
        if msg is not None and msg.fix_type >= 3:
            print("GPS fix acquired", flush=True)
            return True
        time.sleep(0.5)

    print("Timeout waiting for GPS fix", flush=True)
    # Bypass wait (upstream behaviour preserved)...
    return True


def wait_for_ekf_status(master):
    """Bounded EKF wait — upstream is unbounded.

    Raises TimeoutError after EKF_WAIT_TIMEOUT seconds. The script exits
    non-zero in that case so the orchestrator records ``stage=2 failed``
    for this drone and the rest of the swarm proceeds.
    """
    print(
        f"Waiting for EKF status to be OK (timeout {_EKF_WAIT_TIMEOUT:.0f}s)...",
        flush=True,
    )
    deadline = time.time() + _EKF_WAIT_TIMEOUT
    while time.time() < deadline:
        msg = master.recv_match(type="EKF_STATUS_REPORT", blocking=True, timeout=1.0)
        if msg is not None and msg.flags & mavutil.mavlink.EKF_POS_HORIZ_ABS:
            print("EKF status OK", flush=True)
            return
        time.sleep(0.5)
    raise TimeoutError(f"EKF did not converge in {_EKF_WAIT_TIMEOUT:.0f}s")


# Connect to companion computer's mavlink-routerd TCP port directly.
# mavproxy.py in the GCS container also binds udp:0.0.0.0:14550, so using UDP
# here causes a race — stage scripts get starved. TCP 5760 has no competition.
connection_string = f"tcp:10.13.{_instance}.3:5760"
master = mavutil.mavlink_connection(connection_string)

# Wait for the first heartbeat
master.wait_heartbeat()
print(
    f"Heartbeat from system (system {master.target_system} component {master.target_component})",
    flush=True,
)
_step("connect_+_heartbeat")

# Validated single-drone flight params. Inject via PARAM_SET + wait for the
# PARAM_VALUE ack, otherwise ArduPilot silently drops some sends and the
# eeprom-saved value wins (observed: ~10 m/s cruise instead of the 9.72 m/s
# these params should give).
_flight_params = {
    "WPNAV_SPEED": 972.0,
    "LOIT_SPEED": 972.0,
    "WPNAV_SPEED_UP": 700.0,
    "WPNAV_SPEED_DN": 400.0,
    # Loop the waypoint mission instead of loitering on completion — keeps
    # drones maneuvering for the full capture window so attack/null rows
    # both reflect active flight, not hover.
    "MIS_RESTART": 1.0,
    # SR0_* stream rates (Hz). drone.parm sets these to 4 but the lite-image
    # eeprom overrides them downward (observed ~0.3-0.7 Hz). Inject explicit
    # rates to represent real-world ArduCopter telemetry density.
    # Aggregate ≈ 67 msg/sec/drone; 50 drones ≈ 3.4k msg/sec — well under
    # the per-thread pymavlink ceiling.
    "SR0_EXTRA1":   25.0,   # ATTITUDE, AHRS2, VIBRATION
    "SR0_EXTRA2":   10.0,   # VFR_HUD
    "SR0_EXTRA3":   2.0,    # BATTERY_STATUS, WIND, AHRS3
    "SR0_POSITION": 10.0,   # GLOBAL_POSITION_INT, LOCAL_POSITION_NED, GPS_RAW_INT
    "SR0_RAW_SENS": 10.0,   # RAW_IMU, SCALED_IMU2/3
    "SR0_EXT_STAT": 2.0,    # SYS_STATUS, MEMINFO, MISSION_CURRENT
    "SR0_RC_CHAN":  4.0,    # RC_CHANNELS
    "SR0_RAW_CTRL": 4.0,    # SERVO_OUTPUT_RAW
}


def _set_param_confirmed(name, value, max_attempts=5, ack_timeout=2.0):
    name_bytes = name.encode("utf-8").ljust(16, b"\x00")
    for attempt in range(1, max_attempts + 1):
        master.mav.param_set_send(
            master.target_system,
            master.target_component,
            name_bytes,
            value,
            mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
        )
        deadline = time.time() + ack_timeout
        while time.time() < deadline:
            msg = master.recv_match(type="PARAM_VALUE", blocking=True, timeout=ack_timeout)
            if msg is None:
                break
            ack_id = msg.param_id
            if isinstance(ack_id, bytes):
                ack_id = ack_id.split(b"\x00", 1)[0].decode("utf-8", "replace")
            if ack_id == name and abs(msg.param_value - value) < 0.5:
                print(f"  {name} = {msg.param_value} (confirmed, attempt {attempt})", flush=True)
                return True
        print(f"  {name} = {value} not confirmed on attempt {attempt}, retrying…", flush=True)
    print(f"  {name} = {value} FAILED after {max_attempts} attempts", flush=True)
    return False


print("Injecting flight params…", flush=True)
for _name, _val in _flight_params.items():
    _set_param_confirmed(_name, _val)
_step("param_injection")

master.waypoint_clear_all_send()
print("Clearing waypoints...", flush=True)
_step("waypoint_clear")

# Wait for a good GPS fix before continuing
if not wait_for_gps_fix(master):
    print("Failed to acquire GPS fix...", flush=True)
    exit(1)
_step("gps_fix")

# Change to GUIDED mode
master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mavutil.mavlink.COPTER_MODE_GUIDED,
)
wait_for_mode(master, mavutil.mavlink.COPTER_MODE_GUIDED)
print("GUIDED mode set", flush=True)
_step("guided_mode")

# Wait for a good EKF state before continuing — bounded by EKF_WAIT_TIMEOUT.
wait_for_ekf_status(master)
_step("ekf_wait")

# Arm the drone
master.arducopter_arm()
print("Arming motors", flush=True)

# Wait for the drone to be armed with a timeout
arming_timeout = 10
start_time = time.time()
armed = False
while True:
    if time.time() - start_time > arming_timeout:
        print("Arming timeout reached", flush=True)
        break

    heartbeat = master.recv_match(type="HEARTBEAT", blocking=True)
    if heartbeat is not None and is_armed(heartbeat):
        print("Drone is armed", flush=True)
        armed = True
        break

_step("arm")

if armed:
    # Takeoff command
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        10.0,
    )
    print("Takeoff command sent", flush=True)

    time.sleep(5)
    print("Takeoff complete", flush=True)
    _step("takeoff")
else:
    print("Failed to arm motors within timeout", flush=True)
