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
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import swarm.missions as missions
from swarm.attacks.registry import ATTACK_HANDLERS
from swarm.attacks.targets import gcs_endpoint, parse_targets
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
        choices=list(ATTACK_HANDLERS.keys()),
        help="Attack type to inject.",
    )
    run_p.add_argument(
        "--targets",
        default="",
        help="Comma-separated drone IDs or ranges to attack (default: all).",
    )
    run_p.add_argument("--mission", default=None, help="Override mission file path (unused; future).")
    run_p.add_argument("--rate-hz", type=float, default=5.0, help="Attack send rate in Hz.")
    run_p.add_argument("--iface", default="any", help="Network interface to sniff.")
    run_p.add_argument("--output", default=None, help="Override output directory path.")
    run_p.add_argument(
        "--keep-up",
        action="store_true",
        help="Do not tear down Docker compose after run.",
    )
    run_p.add_argument("--seed", type=int, default=None, help="RNG seed for mission generation.")

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
        import scapy.all  # noqa: F401
    except ImportError:
        log.error("scapy is not installed. Run: pip install scapy")
        sys.exit(1)

    try:
        import pymavlink.mavutil  # noqa: F401
    except ImportError:
        log.error("pymavlink is not installed. Run: pip install pymavlink")
        sys.exit(1)

    if os.getuid() != 0:
        log.warning("Not running as root — scapy may fail to open raw sockets.")


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    """Determine and return the output directory path.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Resolved :class:`pathlib.Path` for the output directory.
    """
    if args.output:
        return Path(args.output)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    attack_slug = args.attack.replace("-", "_")
    return Path(f"output/run_{ts}_{attack_slug}_n{args.size}")


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


def _probe_one(n: int, timeout: float) -> bool:
    """Check whether drone instance *n*'s flight controller is ready.

    Opens a raw TCP connection to the FC's MAVLink port. A successful
    connect means ArduPilot is listening; we don't need to wait for a
    HEARTBEAT here (the GCS stages do that themselves).

    Args:
        n: Drone instance number (1-based).
        timeout: Connection timeout in seconds.

    Returns:
        ``True`` when the TCP port accepts a connection.
    """
    try:
        with socket.create_connection((f"10.13.{n}.2", 5760), timeout=timeout):
            return True
    except OSError:
        return False


def _wait_for_swarm_ready(
    size: int,
    per_instance_timeout: float = 2.0,
    total_timeout: float = 30.0,
) -> None:
    """Block until all *size* drone GCS endpoints report readiness.

    Polls all instances in parallel every ``per_instance_timeout`` seconds
    until every instance is ready or *total_timeout* is exceeded.

    Args:
        size: Number of drone instances to wait for.
        per_instance_timeout: Per-HTTP-request timeout in seconds.
        total_timeout: Maximum total wait time in seconds.

    Raises:
        TimeoutError: When not all instances become ready within *total_timeout*.
    """
    deadline = time.monotonic() + total_timeout
    pending = set(range(1, size + 1))
    log.info("Waiting for %d drone(s) to become ready (budget %.0fs)…", size, total_timeout)

    while pending and time.monotonic() < deadline:
        with ThreadPoolExecutor(max_workers=min(len(pending), 32)) as pool:
            futures = {pool.submit(_probe_one, n, per_instance_timeout): n for n in pending}
            for fut in as_completed(futures):
                n = futures[fut]
                if fut.result():
                    pending.discard(n)
                    log.info("Drone %d ready.", n)

        if pending:
            time.sleep(per_instance_timeout)

    if pending:
        raise TimeoutError(f"Timed out waiting for drone(s): {sorted(pending)}")

    log.info("All %d drones ready.", size)


# ---------------------------------------------------------------------------
# GCS flight stage execution
# ---------------------------------------------------------------------------


def _arm_and_autopilot(n: int, log_dir: Path) -> None:
    """Execute arm-and-takeoff then autopilot-flight GCS stages for drone *n*.

    Args:
        n: Drone instance number (1-based).
        log_dir: Directory where GCS stage stdout/stderr is written.

    Raises:
        subprocess.CalledProcessError: When either stage exits non-zero.
    """
    container = f"ground-control-station-lite-{n}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "gcs-stage.log"

    with open(log_path, "ab") as lf:
        for stage in ("arm-and-takeoff.py", "autopilot-flight.py"):
            result = subprocess.run(
                ["docker", "exec", container, "python3", f"/opt/gcs/stages/{stage}"],
                capture_output=True,
            )
            lf.write(f"=== {stage} exit={result.returncode} ===\n".encode())
            lf.write(result.stdout)
            lf.write(result.stderr)
            if result.returncode != 0:
                log.warning("Drone %d %s exited %d", n, stage, result.returncode)


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
    (output_dir / "csv").mkdir(exist_ok=True)

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
    )
    # Snapshot mission into output dir.
    (output_dir / "mission.txt").write_bytes(mission_host_path.read_bytes())
    log.info("Mission written to %s", mission_host_path)

    # Phase 3-4: generate compose file and bring up swarm.
    # NOTE: --start-index 1 is intentional. The orchestrator iterates drones
    # 1..N for readiness, GCS-exec, attack targets, and log collection. Using
    # --auto-start-index could shift the block on a host with existing DVD
    # containers and silently mis-target every per-drone phase.
    _run([
        "python3", "swarm/generate_swarm.py",
        "--instances", str(args.size),
        "--start-index", "1",
        "--out", "docker-compose.swarm.yml",
    ])
    _run([
        "docker", "compose",
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

    try:
        # Phase 5: wait for all instances to report readiness.
        _wait_for_swarm_ready(
            args.size,
            per_instance_timeout=2.0,
            total_timeout=max(120.0, 30.0 * math.ceil(args.size / 10)),
        )

        # Phase 6: set up label windows and start sniffer.
        capture_start_epoch = time.time()
        target_ids = (
            parse_targets(args.targets) if args.targets
            else list(range(1, args.size + 1))
        )

        windows = [
            AttackWindow(
                start_epoch=capture_start_epoch + args.start,
                end_epoch=capture_start_epoch + args.end,
                target_drones=frozenset(target_ids),
                attack_type=args.attack,
            )
        ]
        labels = LabelLookup(windows)
        writer = PacketWriter(output_dir / "csv", labels)
        sniffer = start_sniffer(args.iface, [14540, 14550, 5760], writer)
        log.info("Capture started (epoch %.3f).", capture_start_epoch)

        # Phase 7: push all drones through GCS flight stages in parallel.
        log.info("Arming and launching %d drone(s)…", args.size)
        with ThreadPoolExecutor(max_workers=min(args.size, 32)) as pool:
            futures_arm = {
                pool.submit(
                    _arm_and_autopilot,
                    n,
                    output_dir / "logs" / f"instance-{n}",
                ): n
                for n in range(1, args.size + 1)
            }
            for fut in as_completed(futures_arm):
                n = futures_arm[fut]
                try:
                    fut.result()
                except Exception as exc:
                    log.warning("Drone %d GCS stage error: %s", n, exc)

        # Phase 8: sleep until attack start.
        log.info("Waiting until attack window opens at +%.1fs…", args.start)
        _sleep_until(capture_start_epoch + args.start)

        # Phase 9-10: launch attack threads; wait until window closes; stop.
        attack_duration = args.end - args.start
        stop_event = threading.Event()
        handler = ATTACK_HANDLERS[args.attack]

        attack_threads: list[threading.Thread] = []
        for did in target_ids:
            ip, port = gcs_endpoint(did)
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

        # Phase 12: collect ArduPilot flight-controller logs.
        for n in range(1, args.size + 1):
            src = Path(f"configs/data/raw/instance-{n}/")
            dst = output_dir / "logs" / f"instance-{n}"
            if src.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                log.info("Collected logs for instance %d.", n)

        # Phase 13: teardown Docker compose (unless --keep-up).
        if not args.keep_up:
            try:
                _run([
                    "docker", "compose",
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
