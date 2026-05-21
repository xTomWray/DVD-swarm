"""Host-side MAVLink stage driver for DVD-swarm drones.

Mirrors the shape of the upstream DVD TypeScript ``Stages`` class
(``set`` / ``seq`` / ``get`` / ``is_booted`` / ``reset``) but executes each
stage as a direct ``pymavlink`` session against the per-drone router at
``tcp:10.13.<n>.3:5760``, rather than ``docker exec``ing the GCS stage
scripts (which re-open their own MAVLink connection per stage — wasteful
and racy at N≥10 where ``autopilot-flight.py`` reliably times out on
``MISSION_REQUEST`` mid-upload).

Owning the state machine in-process buys us:
    1. One MAVLink session per drone for the full GUIDED → arm → takeoff →
       AUTO sequence (no reconnect + queue-flush between stages).
    2. Configurable per-request and per-retry timeouts (vs. hard-coded 5 s
       in ``ground-control-station/stages/autopilot-flight.py``).
    3. Awaitable per-stage state for richer orchestrator logging.

No changes to any container image are required — every drone's router is
already listening on ``tcp:10.13.<n>.3:5760`` (the sniffer connects there).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from pymavlink import mavutil

log = logging.getLogger(__name__)

# EKF_STATUS_REPORT.flags bit required before arming. Mirrors the legacy
# arm-and-takeoff.py check (`flags & EKF_POS_HORIZ_ABS`) — that's the
# canonical "good enough to fly" signal the FC emits. Requiring more bits
# (ATTITUDE + velocity + position vert) is too strict: at N=10 cold-boot
# the FC takes a few seconds to assert all of them and 9/10 drones never
# satisfy a 30 s wait, even though their HEARTBEAT and ATTITUDE streams
# are flowing fine.
_EKF_POS_HORIZ_ABS = 16


class Stage(IntEnum):
    """Per-drone stage progression. Mirrors the DVD UI stages 1-5 loosely
    but expresses the *orchestrator*'s view of each drone, not the SITL
    process state held in the simulator-mgmt database."""

    BOOT = 0          # SITL not yet running on the FC container
    HEARTBEAT = 1     # MAVLink connected via router; FC alive
    GUIDED_ARMED = 2  # GUIDED mode set, armed, takeoff acknowledged
    AUTO = 3          # mission uploaded, AUTO mode set
    LANDED = 4        # RTL complete (unused by sim.py today)


@dataclass(frozen=True)
class StageConfig:
    """All knobs for one drone's stage progression.

    Immutable so the orchestrator can hand the same config to multiple
    workers without worrying about cross-drone state leakage.
    """

    instance: int
    waypoint_file: Path
    takeoff_alt_m: float = 10.0
    # Liveness-based waits — every stage keeps waiting as long as the FC
    # is sending ANY MAVLink message. Only bails when the FC has gone
    # silent for `inactivity_timeout` seconds. This matches the legacy
    # arm-and-takeoff.py "while True" behaviour while still bailing on
    # a genuinely dead drone so one bad apple doesn't hang the pool.
    inactivity_timeout: float = 60.0
    # Mission upload retries. Each retry clears the FC's partial mission
    # before restarting from waypoint 0.
    mission_upload_retries: int = 3
    # TCP socket-open + first HEARTBEAT after open. Distinct from the
    # liveness timeout: pre-HEARTBEAT there are no messages to time.
    connect_timeout: float = 60.0


class Stages:
    """One ``Stages`` instance per drone. Holds one pymavlink session."""

    def __init__(self, cfg: StageConfig) -> None:
        self.cfg = cfg
        self.endpoint = f"tcp:10.13.{cfg.instance}.3:5760"
        self._master: mavutil.mavfile | None = None
        self._stage: Stage = Stage.BOOT

    # ---- TS-inspired API ----------------------------------------------------

    def set(self, target: Stage) -> None:
        """Advance to *target* stage, executing every intermediate transition.

        Idempotent if already at or past *target*.
        """
        while self._stage < target:
            self._advance_one()

    def seq(self, start: Stage, end: Stage) -> None:
        """Sequentially advance from *start* through *end*, mirroring the TS
        ``seq(start, end)`` helper."""
        self.set(start)
        self.set(end)

    def get(self) -> Stage:
        """Return the orchestrator's view of the current stage."""
        return self._stage

    def is_booted(self) -> bool:
        """Predicate matching the TS ``is_booted`` (``get() >= 2``)."""
        return self._stage >= Stage.GUIDED_ARMED

    def reset(self) -> None:
        """Close the MAVLink session and reset stage to BOOT. Always safe to
        call multiple times (e.g. from a ``finally`` in the orchestrator)."""
        if self._master is not None:
            try:
                self._master.close()
            except Exception as exc:
                log.debug("Drone %d: error closing master: %s", self.cfg.instance, exc)
            self._master = None
        self._stage = Stage.BOOT

    # ---- Orchestrator convenience ------------------------------------------

    def run_to_auto(self) -> None:
        """End-to-end drive: BOOT → HEARTBEAT → GUIDED_ARMED → AUTO."""
        self.set(Stage.AUTO)

    # ---- Stage implementations ---------------------------------------------

    def _advance_one(self) -> None:
        n = self.cfg.instance
        next_stage = Stage(self._stage + 1)
        log.debug("Drone %d: advancing %s → %s", n, self._stage.name, next_stage.name)

        if next_stage == Stage.HEARTBEAT:
            self._connect_and_heartbeat()
        elif next_stage == Stage.GUIDED_ARMED:
            self._guided_arm_takeoff()
        elif next_stage == Stage.AUTO:
            self._upload_mission_and_auto()
        elif next_stage == Stage.LANDED:
            self._return_to_land()

        self._stage = next_stage

    def _connect_and_heartbeat(self) -> None:
        n = self.cfg.instance
        self._master = mavutil.mavlink_connection(self.endpoint)
        deadline = time.time() + self.cfg.connect_timeout
        while time.time() < deadline:
            msg = self._master.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
            if msg:
                log.info("Drone %d: HEARTBEAT received (%s)", n, self.endpoint)
                return
        raise TimeoutError(f"drone {n}: no HEARTBEAT on {self.endpoint}")

    def _guided_arm_takeoff(self) -> None:
        """Mirror ground-control-station/stages/arm-and-takeoff.py exactly.

        Order matters: GPS fix before mode-set, mode-set confirmed via
        HEARTBEAT before EKF wait. The reversed order (GUIDED-then-GPS)
        appears to delay EKF POS_HORIZ_ABS assertion at N≥10 — only 1 of
        10 drones got there inside 30 s in observed runs.
        """
        assert self._master is not None
        m = self._master
        n = self.cfg.instance

        m.waypoint_clear_all_send()

        # GPS fix — non-fatal on inactivity (legacy bypasses GPS timeout too).
        try:
            self._wait_for("GPS fix", "GPS_RAW_INT", lambda msg: msg.fix_type >= 3)
        except TimeoutError as exc:
            log.warning("Drone %d: %s (continuing — legacy behaviour)", n, exc)

        # GUIDED — use the legacy set_mode_send + wait_for_mode pattern
        # rather than mavutil.set_mode (different MAVLink message + no
        # confirmation), so the FC has actually transitioned before we
        # poll EKF status.
        m.mav.set_mode_send(
            m.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mavutil.mavlink.COPTER_MODE_GUIDED,
        )
        self._wait_for(
            "GUIDED mode",
            "HEARTBEAT",
            lambda msg: msg.custom_mode == mavutil.mavlink.COPTER_MODE_GUIDED,
        )
        log.info("Drone %d: GUIDED mode set", n)

        # EKF — keep waiting as long as the FC is sending messages. Some
        # drones take several minutes to assert POS_HORIZ_ABS at large N;
        # the legacy script had no timeout at all and always worked.
        self._wait_for(
            "EKF POS_HORIZ_ABS",
            "EKF_STATUS_REPORT",
            lambda msg: bool(msg.flags & _EKF_POS_HORIZ_ABS),
        )
        log.info("Drone %d: EKF OK", n)

        m.arducopter_arm()
        m.motors_armed_wait()
        log.info("Drone %d: armed", n)

        m.mav.command_long_send(
            m.target_system,
            m.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0, 0, 0, 0, 0, 0,
            self.cfg.takeoff_alt_m,
        )
        # Don't block on altitude — the legacy arm-and-takeoff.py didn't
        # either, and the mission upload tolerates climbing-in-progress.
        log.info("Drone %d: takeoff command sent (alt %.1fm)", n, self.cfg.takeoff_alt_m)

    def _upload_mission_and_auto(self) -> None:
        assert self._master is not None
        m = self._master
        n = self.cfg.instance
        waypoints = self._read_waypoints()
        log.info("Drone %d: uploading %d waypoint(s)", n, len(waypoints))

        last_err: str | None = None
        attempts = self.cfg.mission_upload_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                self._upload_once(waypoints)
                m.set_mode_auto()
                log.info(
                    "Drone %d: AUTO mode set, mission started (attempt %d/%d)",
                    n, attempt, attempts,
                )
                return
            except (TimeoutError, RuntimeError) as exc:
                last_err = str(exc)
                log.warning(
                    "Drone %d: mission upload attempt %d/%d failed: %s",
                    n, attempt, attempts, exc,
                )
                # FC may have a half-uploaded mission; clear before retry.
                try:
                    m.waypoint_clear_all_send()
                except Exception as clear_exc:
                    log.debug("Drone %d: clear-all failed: %s", n, clear_exc)
                time.sleep(1.0)

        raise RuntimeError(
            f"drone {n}: mission upload failed after {attempts} attempts: {last_err}"
        )

    def _upload_once(self, waypoints: list[tuple[float, float, float]]) -> None:
        """Upload all *waypoints*, using FC-liveness (not wall-clock) as the
        bail condition. Each incoming MAVLink message of any type resets
        the inactivity clock; we only fail if the FC goes silent.
        """
        assert self._master is not None
        m = self._master
        m.waypoint_clear_all_send()
        m.mav.mission_count_send(m.target_system, m.target_component, len(waypoints))

        uploaded = 0
        last_activity = time.time()
        while uploaded < len(waypoints):
            msg = m.recv_match(blocking=True, timeout=1)
            if msg is None:
                if time.time() - last_activity > self.cfg.inactivity_timeout:
                    raise TimeoutError(
                        f"FC silent for {self.cfg.inactivity_timeout:.0f}s "
                        f"(uploaded {uploaded}/{len(waypoints)})"
                    )
                continue
            last_activity = time.time()
            if msg.get_type() not in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                continue
            seq = msg.seq
            if seq >= len(waypoints):
                raise RuntimeError(f"FC requested out-of-range seq {seq}")
            lat, lon, alt = waypoints[seq]
            m.mav.mission_item_int_send(
                m.target_system,
                m.target_component,
                seq,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                0, 0, 0, 0, 0, 0,
                int(lat * 1e7),
                int(lon * 1e7),
                alt,
            )
            uploaded = seq + 1

        # Same liveness logic for the terminal MISSION_ACK.
        last_activity = time.time()
        while True:
            msg = m.recv_match(blocking=True, timeout=1)
            if msg is None:
                if time.time() - last_activity > self.cfg.inactivity_timeout:
                    raise TimeoutError(
                        f"FC silent for {self.cfg.inactivity_timeout:.0f}s "
                        "waiting for MISSION_ACK"
                    )
                continue
            last_activity = time.time()
            if msg.get_type() != "MISSION_ACK":
                continue
            if msg.type != mavutil.mavlink.MAV_MISSION_ACCEPTED:
                raise RuntimeError(f"MISSION_ACK type={msg.type}")
            return

    def _return_to_land(self) -> None:
        assert self._master is not None
        self._master.set_mode("RTL")
        log.info("Drone %d: RTL mode set", self.cfg.instance)

    # ---- Helpers ------------------------------------------------------------

    def _read_waypoints(self) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        with open(self.cfg.waypoint_file) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                lat, lon, alt = (float(x) for x in line.split(","))
                out.append((lat, lon, alt))
        return out

    def _wait_for(self, label: str, msg_type: str, predicate) -> None:
        """Wait until the FC sends a *msg_type* satisfying *predicate*.

        Liveness-based: as long as the FC is sending ANY MAVLink message,
        we keep waiting. Only bails when the FC has been silent for
        ``inactivity_timeout`` seconds. Mirrors the legacy "while True"
        wait loops in ground-control-station/stages/arm-and-takeoff.py
        which always worked.

        Args:
            label: Short human-readable description for log/error text.
            msg_type: MAVLink message type to filter on (e.g. "HEARTBEAT").
            predicate: Callable applied to messages of *msg_type*; returns
                True to terminate the wait.

        Raises:
            TimeoutError: If no MAVLink message of any type arrives for
                ``inactivity_timeout`` seconds — the FC is genuinely dead.
        """
        assert self._master is not None
        last_activity = time.time()
        while True:
            msg = self._master.recv_match(blocking=True, timeout=1)
            if msg is None:
                if time.time() - last_activity > self.cfg.inactivity_timeout:
                    raise TimeoutError(
                        f"drone {self.cfg.instance}: FC silent for "
                        f"{self.cfg.inactivity_timeout:.0f}s while waiting for {label}"
                    )
                continue
            last_activity = time.time()
            if msg.get_type() == msg_type and predicate(msg):
                return
