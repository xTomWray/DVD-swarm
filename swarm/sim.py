"""DVD-swarm simulation orchestrator.

Brings up a Docker-compose swarm, executes GCS flight stages, runs an attack
during a configurable window, captures MAVLink traffic to labelled CSV files,
and tears everything down cleanly.

Usage::

    python -m swarm.sim run --size 5 --start 30 --end 90 --attack attitude_spoof

All phases run in the correct order; dataset output lands in
``output/run_<UTC>_<attack>_n<N>/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import swarm.missions as missions
from swarm.attacks.registry import ATTACK_HANDLERS
from swarm.attacks.targets import companion_endpoint, parse_targets
from swarm.dataset.labels import AttackWindow, LabelLookup
from swarm.dataset.packet_writer import PacketWriter
from swarm.dataset.sniffer import start_sniffer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("swarm.sim")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with all sub-command arguments.
    """
    parser = argparse.ArgumentParser(
        prog="swarm.sim",
        description="DVD-swarm simulation orchestrator.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a simulation experiment.")
    run_p.add_argument("--size", type=int, required=True, help="Swarm size (number of drones).")
    run_p.add_argument(
        "--start",
        type=float,
        required=True,
        help="Seconds after capture start to begin attack.",
    )
    run_p.add_argument(
        "--end",
        type=float,
        required=True,
        help="Seconds after capture start to end attack (also controls mission duration).",
    )
    run_p.add_argument(
        "--attack",
        required=True,
        choices=list(ATTACK_HANDLERS.keys()) + ["none"],
        help="Attack type to inject. Use 'none' for a benign-only capture "
             "(every row labelled 'null' — useful for building a baseline).",
    )
    run_p.add_argument(
        "--targets",
        default="",
        help="Comma-separated drone IDs or ranges to attack (default: all).",
    )
    run_p.add_argument("--mission", default=None, help="Override mission file path (unused; future).")
    run_p.add_argument("--rate-hz", type=float, default=5.0, help="Attack send rate in Hz.")
    run_p.add_argument("--iface", default=None, help="(deprecated — ignored; kept for backwards compat)")
    run_p.add_argument("--output", default=None, help="Override output directory path.")
    run_p.add_argument(
        "--keep-up",
        action="store_true",
        help="Do not tear down Docker compose after run.",
    )
    run_p.add_argument("--seed", type=int, default=None, help="RNG seed for mission generation.")
    run_p.add_argument(
        "--strict",
        action="store_true",
        help="Abort the run if any drone fails any readiness probe. Default is "
             "to drop failed drones and continue with the rest — recommended "
             "at large N where 5-10%% transient failures are common.",
    )
    run_p.add_argument(
        "--gcs-concurrency",
        type=int,
        default=12,
        help="Max parallel Stages workers driving the GUIDED → arm → takeoff → "
             "AUTO sequence over MAVLink. Lower if you see widespread mission "
             "upload retries under host load (default: 12).",
    )
    run_p.add_argument(
        "--mission-request-timeout",
        type=float,
        default=60.0,
        help="Per-MISSION_REQUEST timeout (seconds) during mission upload. "
             "The legacy autopilot-flight.py hard-coded 5s, which dropped "
             "uploads at N≥10 under CPU caps. 60s is safe at N=50 (default: 60).",
    )
    run_p.add_argument(
        "--mission-upload-retries",
        type=int,
        default=3,
        help="Number of times to re-attempt a full mission upload if a "
             "MISSION_REQUEST or MISSION_ACK times out. Each retry clears the "
             "FC's partial mission first (default: 3 retries = up to 4 attempts).",
    )
    run_p.add_argument(
        "--stage1-concurrency",
        type=int,
        default=16,
        help="Max parallel POST /stage1 invocations to simulator-lite containers "
             "(boots arducopter inside each FC container). Was serial before — at "
             "N=50 serial × 60s timeout = ~50 min wallclock, exhausting the "
             "HEARTBEAT budget. Default 16 keeps the whole Stage 1 phase under ~3 "
             "min at N=50 (default: 16).",
    )
    run_p.add_argument(
        "--readiness-probe-timeout",
        type=float,
        default=5.0,
        help="Per-probe TCP/Flask timeout (seconds) during the 6-phase swarm "
             "readiness check. Default 5s handles cold-boot at N=50; raise to "
             "10s+ if probes fail under heavier host load (default: 5).",
    )
    run_p.add_argument(
        "--readiness-total-timeout",
        type=float,
        default=0.0,
        help="Total wall-clock budget (seconds) for each readiness phase. "
             "0 (default) auto-sizes as max(180, 60 × ceil(N/10)) — N=50 gets "
             "300s, N=100 gets 600s. Override only for very slow hosts.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------


def _preflight_checks() -> None:
    """Verify that runtime dependencies are available.

    Raises:
        SystemExit: When a required dependency is missing.
    """
    if shutil.which("docker") is None:
        log.error("docker not found on PATH. Please install Docker.")
        sys.exit(1)

    try:
        import pymavlink.mavutil  # noqa: F401
    except ImportError:
        log.error("pymavlink is not installed. Run: pip install pymavlink")
        sys.exit(1)



# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Determine and return the output directory path.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Resolved :class:`pathlib.Path` for the output directory. Default
        format is ``output/run_<UTC>_<attack>_n<N>_s<START>_e<END>`` so the
        directory name alone identifies the experiment's swarm size and
        attack-window timing.
    """
    if args.output:
        return Path(args.output)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    attack_slug = args.attack.replace("-", "_")
    return Path(
        f"output/run_{ts}_{attack_slug}"
        f"_n{args.size}_s{int(args.start)}_e{int(args.end)}"
    )


# ---------------------------------------------------------------------------
# Mission helpers
# ---------------------------------------------------------------------------


def _backup_original_mission(path: Path) -> None:
    """Copy *path* to ``<path>.bak`` if it exists and no backup exists yet.

    Args:
        path: Path to the mission file to back up.
    """
    bak = path.with_suffix(".txt.bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)
        log.info("Backed up original mission to %s", bak)


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
    """Run *cmd* as a subprocess, logging the invocation.

    Args:
        cmd: Command and arguments list.
        **kwargs: Extra keyword arguments forwarded to :func:`subprocess.run`.

    Returns:
        Completed process result.

    Raises:
        subprocess.CalledProcessError: When the process exits non-zero and
            ``check=True`` (the default).
    """
    log.info("$ %s", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Readiness probe
# ---------------------------------------------------------------------------


def _probe_flask(n: int, timeout: float) -> bool:
    """Return True when the companion Flask app is reachable."""
    import requests

    try:
        resp = requests.get(f"http://localhost:{3000 + n}/socket-health", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _start_mavlink_router(n: int) -> None:
    """POST to companion to start mavlink-routerd with TCP server on port 5760.

    The endpoint requires ``serial_device`` and ``baud_rate`` — without them
    the handler crashes on ``serial_device + ":" + str(baud_rate)`` and silently
    fails to launch the router, so we always send the companion's defaults
    (matching ``companion-computer/interface/config.json``).

    Raises:
        RuntimeError: When the POST does not return HTTP 200.
    """
    import requests

    url = f"http://localhost:{3000 + n}/telemetry/start-telemetry"
    payload = {
        "serial_device": "/dev/ttyUSB0",
        "baud_rate": 57600,
        "enable_tcp_server": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
    except Exception as exc:
        raise RuntimeError(f"drone {n}: POST {url} failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"drone {n}: POST {url} returned {resp.status_code}: {resp.text[:200]}"
        )
    log.info("MAVLink router started for drone %d (%s).", n, resp.json().get("cmd", ""))


def _probe_mavlink_tcp(n: int, timeout: float) -> bool:
    """Return True when TCP 5760 on the companion is accepting connections.

    Runs from inside the GCS container — the same network path used by
    ``arm-and-takeoff.py`` — so a pass here guarantees the GCS stages connect.
    """
    try:
        result = subprocess.run(
            [
                "docker", "exec", f"ground-control-station-lite-{n}",
                "python3", "-c",
                f"import socket; socket.create_connection(('10.13.{n}.3', 5760), {timeout}).close()",
            ],
            capture_output=True,
            timeout=timeout + 2,
        )
        return result.returncode == 0
    except Exception:
        return False


def _probe_simulator_mgmt(n: int, timeout: float) -> bool:
    """Return True when the simulator-lite mgmt Flask is reachable."""
    import requests

    try:
        resp = requests.get(f"http://localhost:{8000 + n}/", timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def _trigger_stage1(n: int) -> None:
    """POST simulator-lite ``/stage1`` to boot SITL inside the FC container.

    Without this, ``mavlink-routerd`` has nothing on the other end of its
    ``/dev/ttyUSB0`` serial pipe — TCP 5760 accepts connections but never
    delivers a HEARTBEAT, so ``arm-and-takeoff.py`` blocks on
    ``wait_heartbeat()`` indefinitely.

    Stage 1 also re-POSTs ``start-telemetry`` in a daemon thread; since our
    router is already running, that thread silently 500s and is harmless.

    Raises:
        RuntimeError: When the POST does not return HTTP 200.
    """
    import requests

    url = f"http://localhost:{8000 + n}/stage1"
    try:
        resp = requests.post(url, timeout=60)
    except Exception as exc:
        raise RuntimeError(f"drone {n}: POST {url} failed: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"drone {n}: POST {url} returned {resp.status_code}: {resp.text[:200]}"
        )
    log.info("Stage 1 (SITL) triggered for drone %d.", n)


def _probe_mavlink_heartbeat(n: int, timeout: float) -> bool:
    """Return True when a HEARTBEAT arrives via TCP 5760 from inside GCS.

    This confirms the full FC → router → GCS chain is live, which TCP-accept
    alone does not. Uses the same library and connection string as
    ``arm-and-takeoff.py`` so a pass here guarantees its ``wait_heartbeat``
    will succeed.
    """
    script = (
        "from pymavlink import mavutil; import sys; "
        f"m = mavutil.mavlink_connection('tcp:10.13.{n}.3:5760'); "
        f"hb = m.wait_heartbeat(timeout={timeout}); "
        "sys.exit(0 if hb is not None else 1)"
    )
    try:
        result = subprocess.run(
            [
                "docker", "exec", f"ground-control-station-lite-{n}",
                "python3", "-c", script,
            ],
            capture_output=True,
            timeout=timeout + 5,
        )
        return result.returncode == 0
    except Exception:
        return False


def _poll_until_ready(
    probe: object,
    drone_ids: "set[int] | list[int]",
    label: str,
    per_timeout: float = 2.0,
    total_timeout: float = 120.0,
    *,
    allow_partial: bool = False,
) -> set[int]:
    """Poll *probe(n, per_timeout)* for the given drones until they all pass or timeout.

    Args:
        probe: Callable ``(n: int, timeout: float) -> bool``.
        drone_ids: Drone instance numbers to probe (a set or list).
        label: Human-readable name used in log messages.
        per_timeout: Timeout passed to each probe call.
        total_timeout: Maximum wall-clock seconds to wait overall.
        allow_partial: When True, return the subset of drones that did pass
            instead of raising ``TimeoutError`` on timeout.

    Returns:
        Set of drone IDs that passed the probe.

    Raises:
        TimeoutError: When not all instances pass within *total_timeout* and
            ``allow_partial`` is False.
    """
    target = set(drone_ids)
    pending = set(target)
    deadline = time.monotonic() + total_timeout
    log.info("Waiting for %s on %d drone(s) (budget %.0fs)…", label, len(target), total_timeout)

    while pending and time.monotonic() < deadline:
        with ThreadPoolExecutor(max_workers=min(len(pending), 32)) as pool:
            futures = {pool.submit(probe, n, per_timeout): n for n in pending}
            for fut in as_completed(futures):
                n = futures[fut]
                if fut.result():
                    pending.discard(n)
                    log.info("Drone %d: %s ✓", n, label)

        if pending:
            time.sleep(per_timeout)

    ready = target - pending
    if pending:
        if allow_partial:
            log.warning(
                "Continuing without %s on %d drone(s): %s",
                label, len(pending), sorted(pending),
            )
        else:
            raise TimeoutError(f"Timed out waiting for {label} on drone(s): {sorted(pending)}")
    else:
        log.info("All %d drones: %s ✓", len(target), label)

    return ready


def _wait_for_swarm_ready(
    size: int,
    per_instance_timeout: float,
    total_timeout: float,
    *,
    allow_partial: bool = False,
    stage1_concurrency: int = 16,
) -> set[int]:
    """Bring the swarm to a state where GCS stages can run.

    Sequential phases:
    1. Wait for companion Flask (``/socket-health`` → 200).
    2. Wait for simulator-lite mgmt Flask (``/`` → 200).
    3. Trigger ``mavlink-routerd`` (POST companion ``/telemetry/start-telemetry``).
    4. Wait for TCP 5760 on the companion to accept connections.
    5. Trigger Stage 1 (POST simulator-lite ``/stage1``) to boot SITL.
    6. Wait for an actual HEARTBEAT to arrive over TCP 5760.

    With ``allow_partial=True``, drones that fail any phase are dropped from
    later phases and from the returned ready set — the run continues with
    whichever drones reached HEARTBEAT successfully. Useful at large N
    where 5–10% transient failures are common.

    Args:
        size: Number of drone instances.
        per_instance_timeout: Per-probe timeout in seconds (TCP / Flask probes).
        total_timeout: Budget for each polled phase.
        allow_partial: Skip failed drones instead of raising.

    Returns:
        Set of drone IDs that completed all six phases.
    """
    all_drones = set(range(1, size + 1))

    ready = _poll_until_ready(
        _probe_flask, all_drones, "companion Flask",
        per_instance_timeout, total_timeout, allow_partial=allow_partial,
    )
    ready = _poll_until_ready(
        _probe_simulator_mgmt, ready, "simulator mgmt Flask",
        per_instance_timeout, total_timeout, allow_partial=allow_partial,
    )

    routed: set[int] = set()
    for n in sorted(ready):
        try:
            _start_mavlink_router(n)
            routed.add(n)
        except Exception as exc:
            if allow_partial:
                log.warning("Drone %d: failed to start mavlink-routerd: %s", n, exc)
            else:
                raise

    ready = _poll_until_ready(
        _probe_mavlink_tcp, routed, "MAVLink TCP 5760",
        per_instance_timeout, total_timeout, allow_partial=allow_partial,
    )

    workers = max(1, min(len(ready), stage1_concurrency))
    log.info("Triggering Stage 1 on %d drone(s) (%d parallel)…", len(ready), workers)
    staged: set[int] = set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_trigger_stage1, n): n for n in sorted(ready)}
        for fut in as_completed(futures):
            n = futures[fut]
            try:
                fut.result()
                staged.add(n)
            except Exception as exc:
                if allow_partial:
                    log.warning("Drone %d: failed to trigger stage1: %s", n, exc)
                else:
                    raise

    # SITL needs ~5–15s to start emitting heartbeats. Use a longer per-probe
    # timeout here so each attempt actually has time to receive one.
    ready = _poll_until_ready(
        _probe_mavlink_heartbeat, staged, "MAVLink HEARTBEAT",
        10.0, total_timeout, allow_partial=allow_partial,
    )

    return ready


# ---------------------------------------------------------------------------
# GCS flight stage execution
# ---------------------------------------------------------------------------


_WAYPOINT_FILE = (
    Path(__file__).resolve().parent.parent
    / "ground-control-station" / "missions" / "waypoints_custom_zigzag_square.txt"
)


def _drive_stages(
    n: int,
    waypoint_file: Path,
    mission_request_timeout: float,
    mission_upload_retries: int,
) -> None:
    """Drive drone *n* from BOOT → AUTO via a host-side MAVLink session.

    Replaces the legacy ``docker exec arm-and-takeoff && docker exec
    autopilot-flight`` chain with a single ``Stages`` instance — one
    pymavlink connection per drone, host-side timeouts, and retryable
    mission upload (the hard-coded 5s MISSION_REQUEST timeout in the
    legacy ``autopilot-flight.py`` was the root cause of N≥10 dropouts).

    Args:
        n: Drone instance number (1-based).
        waypoint_file: Path to the host-side waypoints text file.
        mission_request_timeout: Per-MISSION_REQUEST timeout in seconds.
        mission_upload_retries: Number of upload re-attempts on failure.

    Raises:
        Exception: Propagates any unrecoverable stage failure (caught by
            the orchestrator and logged per-drone).
    """
    from swarm.stages import StageConfig, Stages

    cfg = StageConfig(
        instance=n,
        waypoint_file=waypoint_file,
        mission_request_timeout=mission_request_timeout,
        mission_upload_retries=mission_upload_retries,
    )
    stages = Stages(cfg)
    try:
        stages.run_to_auto()
    finally:
        stages.reset()


# ---------------------------------------------------------------------------
# Sleep helpers
# ---------------------------------------------------------------------------


def _sleep_until(target_epoch: float) -> None:
    """Sleep until *target_epoch* (UNIX timestamp), with a floor of zero.

    Args:
        target_epoch: Target UNIX timestamp to sleep until.
    """
    delay = target_epoch - time.time()
    if delay > 0:
        time.sleep(delay)


# ---------------------------------------------------------------------------
# Main lifecycle
# ---------------------------------------------------------------------------


def main() -> int:
    """Entry point for the ``swarm.sim run`` command.

    Orchestrates the full experiment lifecycle: mission generation, Docker
    compose bring-up, readiness waiting, packet capture, attack injection,
    GCS stage execution, log collection, teardown, and metadata writing.

    Returns:
        Exit code (0 on success).
    """
    args = _parse_args()
    _preflight_checks()

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-run logfile inside the output dir. Every log.* call from any module
    # in this process now writes to <output_dir>/sim.log as well as stdout.
    # Subprocess output (docker compose, etc.) still only hits stdout — for
    # that, look at the make-loop log under logs/.
    run_log_handler = logging.FileHandler(output_dir / "sim.log", mode="w")
    run_log_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    run_log_handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(run_log_handler)

    log.info("Output directory: %s", output_dir)

    # Phase 2: generate mission and deploy to GCS mount path.
    mission_host_path = Path(
        "ground-control-station/missions/waypoints_custom_zigzag_square.txt"
    )
    _backup_original_mission(mission_host_path)
    missions.generate(
        duration_s=int(args.end),
        out_path=mission_host_path,
        seed=args.seed,
        cruise_speed_m_s=3.0,  # conservative: generates ~67% more waypoints than needed
    )
    # Snapshot mission into output dir.
    (output_dir / "mission.txt").write_bytes(mission_host_path.read_bytes())
    log.info("Mission written to %s", mission_host_path)

    # Phase 3-4: generate compose file and bring up swarm.
    # NOTE: --start-index 1 is intentional. The orchestrator iterates drones
    # 1..N for readiness, GCS-exec, attack targets, and log collection. Using
    # --auto-start-index could shift the block on a host with existing DVD
    # containers and silently mis-target every per-drone phase.
    # Per-run raw-data dir so FC logs from one run don't collide with the next
    # (Docker writes inside the container as root, leaving root-owned files
    # on the host bind-mount target).
    raw_dir = (output_dir / "raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    _run([
        "python3", "swarm/generate_swarm.py",
        "--instances", str(args.size),
        "--start-index", "1",
        "--out", "docker-compose.swarm.yml",
        "--raw-dir", str(raw_dir),
    ])
    _run([
        "docker", "compose",
        "--progress", "plain",
        "-p", "dvd-swarm",
        "-f", "docker-compose.swarm.yml",
        "up", "-d",
    ])

    # -----------------------------------------------------------------
    # Variables that must be accessible in the finally block.
    # -----------------------------------------------------------------
    sniffer: object | None = None
    writer: PacketWriter | None = None
    capture_start_epoch: float = 0.0
    target_ids: list[int] = []
    drone_ids: list[int] = list(range(1, args.size + 1))  # default for cleanup paths
    skipped: list[int] = []

    try:
        # Phase 5: wait for all instances to report readiness.
        readiness_total = (
            args.readiness_total_timeout
            if args.readiness_total_timeout > 0
            else max(180.0, 60.0 * math.ceil(args.size / 10))
        )
        ready_drones = _wait_for_swarm_ready(
            args.size,
            per_instance_timeout=args.readiness_probe_timeout,
            total_timeout=readiness_total,
            allow_partial=not args.strict,
            stage1_concurrency=args.stage1_concurrency,
        )
        if not ready_drones:
            raise RuntimeError("No drones reached HEARTBEAT readiness — aborting.")
        skipped = sorted(set(range(1, args.size + 1)) - ready_drones)
        if skipped:
            log.warning(
                "Proceeding with %d/%d drone(s); skipping %s",
                len(ready_drones), args.size, skipped,
            )
        drone_ids = sorted(ready_drones)

        # Phase 6: set up label windows and start sniffer.
        capture_start_epoch = time.time()
        target_ids = (
            parse_targets(args.targets) if args.targets
            else drone_ids
        )
        # Don't attempt to attack drones that never came up.
        target_ids = sorted(set(target_ids) & ready_drones)

        # Benign-only runs use an empty window list — every row gets
        # attack_type='null'. Attack runs install one window covering
        # [start, end] for the targeted drones.
        if args.attack == "none":
            windows: list[AttackWindow] = []
        else:
            windows = [
                AttackWindow(
                    start_epoch=capture_start_epoch + args.start,
                    end_epoch=capture_start_epoch + args.end,
                    target_drones=frozenset(target_ids),
                    attack_type=args.attack,
                )
            ]
        labels = LabelLookup(windows)
        writer = PacketWriter(
            output_dir / "csv",
            labels,
            sim_uuid=output_dir.name,
            drone_ids=drone_ids,
        )
        sniffer = start_sniffer(drone_ids, writer)
        log.info("Capture started (epoch %.3f).", capture_start_epoch)

        # Phase 7: drive each ready drone GUIDED → arm → takeoff → AUTO from
        # the host over MAVLink. One pymavlink session per drone; the per-
        # drone router at tcp:10.13.<n>.3:5760 multiplexes us alongside the
        # sniffer. Concurrency caps the simultaneous sessions — too many
        # active uploads share host I/O and slow MISSION_REQUEST round-trips.
        gcs_workers = max(1, min(len(drone_ids), args.gcs_concurrency))
        log.info(
            "Driving %d drone(s) to AUTO (concurrency=%d, mission_request_timeout=%.0fs, retries=%d)…",
            len(drone_ids), gcs_workers,
            args.mission_request_timeout, args.mission_upload_retries,
        )
        with ThreadPoolExecutor(max_workers=gcs_workers) as pool:
            futures_arm = {
                pool.submit(
                    _drive_stages,
                    n,
                    _WAYPOINT_FILE,
                    args.mission_request_timeout,
                    args.mission_upload_retries,
                ): n
                for n in drone_ids
            }
            for fut in as_completed(futures_arm):
                n = futures_arm[fut]
                try:
                    fut.result()
                except Exception as exc:
                    log.warning("Drone %d Stages error: %s", n, exc)

        if args.attack == "none":
            # Benign-only: no spoofing threads, just capture for the mission
            # duration so every row in every drone's CSV is labelled null.
            log.info("Benign mode (--attack none); capturing for %.1fs with no spoofing.", args.end)
            _sleep_until(capture_start_epoch + args.end)
        else:
            # Phase 8: sleep until attack start.
            log.info("Waiting until attack window opens at +%.1fs…", args.start)
            _sleep_until(capture_start_epoch + args.start)

            # Phase 9-10: launch attack threads; wait until window closes; stop.
            attack_duration = args.end - args.start
            stop_event = threading.Event()
            handler = ATTACK_HANDLERS[args.attack]

            attack_threads: list[threading.Thread] = []
            for did in target_ids:
                ip, port = companion_endpoint(did)
                t = threading.Thread(
                    target=handler,
                    args=(ip, port, attack_duration, args.rate_hz, stop_event),
                    daemon=True,
                    name=f"attack-drone-{did}",
                )
                t.start()
                attack_threads.append(t)
                log.info("Attack thread started for drone %d (%s:%d).", did, ip, port)

            log.info("Waiting until attack window closes at +%.1fs…", args.end)
            _sleep_until(capture_start_epoch + args.end)
            stop_event.set()

            for t in attack_threads:
                t.join(timeout=5.0)
            log.info("Attack complete.")

    finally:
        # Phase 11: stop sniffer and close writer.
        if sniffer is not None:
            try:
                sniffer.stop()  # type: ignore[union-attr]
                log.info("Sniffer stopped.")
            except Exception as exc:
                log.warning("Error stopping sniffer: %s", exc)

        if writer is not None:
            writer.close()
            log.info("Writer closed. Row counts: %s", writer.counts)

        if windows:
            labeled = 0
            csv_dir = output_dir / "csv"
            for csv_path in csv_dir.glob("drone_*.csv"):
                try:
                    with open(csv_path) as fh:
                        header = fh.readline().rstrip("\n").split(",")
                        try:
                            idx = header.index("attack_type")
                        except ValueError:
                            continue
                        for line in fh:
                            cells = line.rstrip("\n").split(",")
                            if idx < len(cells) and cells[idx] != "null":
                                labeled += 1
                except OSError as exc:
                    log.warning("Label check: could not read %s: %s", csv_path, exc)
            log.info(
                "Label coverage: %d row(s) with non-null attack_type across %d attack window(s)",
                labeled, len(windows),
            )
            if labeled == 0:
                log.warning(
                    "Attack run produced ZERO labeled rows — biLSTM will train on all-benign "
                    "data. Check --start/--end window timing and --targets vs ready_drones."
                )

        # Phase 12: collect ArduPilot flight-controller logs.
        # Source is per-run raw_dir (output_dir/raw/instance-N) so we don't
        # have to clean a shared configs/data/raw between runs. We try every
        # drone slot (not just `drone_ids`) so partial-readiness skips still
        # collect whatever logs the failed drones did manage to write.
        for n in range(1, args.size + 1):
            src = raw_dir / f"instance-{n}"
            dst = output_dir / "logs" / f"instance-{n}"
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                log.info("Collected logs for instance %d.", n)

        # Phase 13: teardown Docker compose (unless --keep-up).
        if not args.keep_up:
            try:
                _run([
                    "docker", "compose",
                    "--progress", "plain",
                    "-p", "dvd-swarm",
                    "-f", "docker-compose.swarm.yml",
                    "down", "-v", "--remove-orphans",
                ])
            except Exception as exc:
                log.warning("Compose teardown error: %s", exc)

        # Phase 14: write metadata.json.
        metadata = {
            "run_started_utc": datetime.now(UTC).isoformat(),
            "swarm_size": args.size,
            "ready_drones": drone_ids,
            "skipped_drones": skipped,
            "start_s": args.start,
            "end_s": args.end,
            "attack": args.attack,
            "attack_targets": target_ids,
            "rate_hz": args.rate_hz,
            "capture_start_epoch": capture_start_epoch,
            "row_counts": writer.counts if writer is not None else {},
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log.info("Metadata written to %s/metadata.json", output_dir)

        # Detach + close the per-run file handler so the log file's last
        # buffered lines flush before the process exits.
        logging.getLogger().removeHandler(run_log_handler)
        run_log_handler.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
