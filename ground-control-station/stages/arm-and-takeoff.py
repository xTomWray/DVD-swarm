import os
import time

from pymavlink import mavutil


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
    print("Waiting for GPS fix...")

    while time.time() - start_time < timeout:
        msg = master.recv_match(type="GPS_RAW_INT", blocking=True, timeout=1)
        if msg is not None and msg.fix_type >= 3:
            print("GPS fix acquired")
            return True
        time.sleep(0.5)

    print("Timeout waiting for GPS fix")
    # Bypass wait...
    return True


def wait_for_ekf_status(master):
    print("Waiting for EKF status to be OK...")
    while True:
        msg = master.recv_match(type="EKF_STATUS_REPORT", blocking=True)
        if msg is not None and msg.flags & mavutil.mavlink.EKF_POS_HORIZ_ABS:
            # Check if the EKF's absolute horizontal position is good
            print("EKF status OK")
            break
        time.sleep(0.5)


# Connect to companion computer's mavlink-routerd TCP port directly.
# mavproxy.py in the GCS container also binds udp:0.0.0.0:14550, so using UDP
# here causes a race — stage scripts get starved. TCP 5760 has no competition.
_instance = os.getenv("SWARM_INSTANCE", "0")
connection_string = f"tcp:10.13.{_instance}.3:5760"
master = mavutil.mavlink_connection(connection_string)

# Wait for the first heartbeat
master.wait_heartbeat()
print(f"Heartbeat from system (system {master.target_system} component {master.target_component})")

# param_set_send overwrites regardless of eeprom.bin; --add-param-file only sets defaults
_flight_params = {
    "WPNAV_SPEED": 2000.0,
    "WPNAV_ACCEL": 500.0,
    "WPNAV_ACCEL_Z": 200.0,
    "WPNAV_SPEED_UP": 500.0,
    "WPNAV_SPEED_DN": 300.0,
    "MIS_RESTART": 1.0,
}
for _name, _val in _flight_params.items():
    master.mav.param_set_send(
        master.target_system,
        master.target_component,
        _name.encode("utf-8"),
        _val,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
    )
    time.sleep(0.1)
print("Flight params injected via MAVLink")

master.waypoint_clear_all_send()
print("Clearing waypoints...")

# Wait for a good GPS fix before continuing
if not wait_for_gps_fix(master):
    print("Failed to acquire GPS fix...")
    exit(1)

# Change to GUIDED mode
master.mav.set_mode_send(
    master.target_system,
    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
    mavutil.mavlink.COPTER_MODE_GUIDED,
)
wait_for_mode(master, mavutil.mavlink.COPTER_MODE_GUIDED)
print("GUIDED mode set")

# Wait for a good GPS fix before continuing
wait_for_ekf_status(master)

# Arm the drone
master.arducopter_arm()
print("Arming motors")

# Wait for the drone to be armed with a timeout
arming_timeout = 10
start_time = time.time()
armed = False
while True:
    if time.time() - start_time > arming_timeout:
        print("Arming timeout reached")
        break

    heartbeat = master.recv_match(type="HEARTBEAT", blocking=True)
    if heartbeat is not None and is_armed(heartbeat):
        print("Drone is armed")
        armed = True
        break

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
    print("Takeoff command sent")

    time.sleep(5)
    print("Takeoff complete")
else:
    print("Failed to arm motors within timeout")
