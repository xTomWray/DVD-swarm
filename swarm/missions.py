"""Random realistic waypoint generator for DVD-swarm missions.

Generates a lawnmower-with-jitter waypoint pattern in plain-CSV format
compatible with ground-control-station autopilot-flight stage.

Plain-CSV format: one ``lat, lon, alt`` per line; ``#`` comments allowed;
blank lines skipped.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path


# Flat-earth constants
_METRES_PER_DEG_LAT: float = 111_320.0


def _metres_per_deg_lon(lat_rad: float) -> float:
    """Return metres per degree of longitude at the given latitude."""
    return 111_320.0 * math.cos(lat_rad)


def generate(
    duration_s: int,
    out_path: Path,
    *,
    seed: int | None = None,
    cruise_speed_m_s: float = 5.0,
    centroid_lat: float = 37.241861,
    centroid_lon: float = -115.796917,
    leg_spacing_m: float = 50.0,
    jitter_m: float = 5.0,
    alt_floor_m: float = 30.0,
    alt_ceiling_m: float = 60.0,
) -> Path:
    """Generate a lawnmower-with-jitter waypoint file for a timed mission.

    Computes the required path length from ``duration_s`` and
    ``cruise_speed_m_s`` (with a 20% margin so the drone doesn't finish
    early), then lays out alternating east/west legs spaced ``leg_spacing_m``
    apart centred on (``centroid_lat``, ``centroid_lon``).

    Each waypoint is perturbed by ±``jitter_m`` (uniform random) in both
    horizontal axes. Altitudes ramp linearly from ``alt_floor_m`` to
    ``alt_ceiling_m`` across the mission, with ±2 m uniform jitter per
    waypoint.

    Coordinates are converted from local metres to lat/lon using a flat-earth
    approximation (no geodesy library required).

    Args:
        duration_s: Intended mission duration in seconds.
        out_path: Destination file path for the generated CSV.
        seed: RNG seed for reproducible output. ``None`` uses ``time.time_ns()``
            and records the chosen seed in the comment header.
        cruise_speed_m_s: Assumed cruise speed used to compute required path
            length.
        centroid_lat: Latitude of the pattern centre in decimal degrees.
        centroid_lon: Longitude of the pattern centre in decimal degrees.
        leg_spacing_m: North-south spacing between parallel legs in metres.
        jitter_m: Maximum horizontal position perturbation in metres (uniform).
        alt_floor_m: Altitude at the first waypoint in metres AGL.
        alt_ceiling_m: Altitude at the last waypoint in metres AGL.

    Returns:
        ``out_path`` after the file has been written.

    Raises:
        ValueError: If ``alt_floor_m`` > ``alt_ceiling_m`` or
            ``leg_spacing_m`` <= 0.
    """
    if alt_floor_m > alt_ceiling_m:
        raise ValueError(f"alt_floor_m ({alt_floor_m}) must be <= alt_ceiling_m ({alt_ceiling_m})")
    if leg_spacing_m <= 0:
        raise ValueError(f"leg_spacing_m must be positive, got {leg_spacing_m}")

    actual_seed: int = seed if seed is not None else time.time_ns()
    rng = random.Random(actual_seed)

    # --- 1. Compute required path length ------------------------------------ #
    length_m: float = duration_s * cruise_speed_m_s * 1.2

    # --- 2. Build lawnmower pattern ----------------------------------------- #
    centroid_lat_rad = math.radians(centroid_lat)
    m_per_deg_lon = _metres_per_deg_lon(centroid_lat_rad)

    # Determine leg length so total path >= length_m.
    # Each "leg" is a single east-or-west traverse plus one north step.
    # Strategy: pick leg_length_m first so that an integer number of legs
    # covers the required distance.  We start with a square aspect ratio and
    # add legs as needed.
    #
    # path ≈ n_legs * leg_length_m  (the N/S hops between legs are short).
    # Pick leg_length_m = leg_spacing_m * 2 (reasonable square-ish cell),
    # then solve for n_legs.

    leg_length_m: float = max(leg_spacing_m * 2, length_m / 50)  # at least 50 legs
    n_legs: int = math.ceil(length_m / leg_length_m)
    if n_legs < 2:
        n_legs = 2

    # Increase leg_length_m until the total estimated path covers length_m,
    # accounting for the short N/S hops.
    while True:
        estimated = n_legs * leg_length_m + (n_legs - 1) * leg_spacing_m
        if estimated >= length_m:
            break
        n_legs += 1

    # --- 3. Lay out waypoint centres ---------------------------------------- #
    # The pattern spans (n_legs - 1) * leg_spacing_m north-south.
    # Centre it on centroid by offsetting south by half that span.
    half_span_m: float = (n_legs - 1) * leg_spacing_m / 2.0

    # Each leg has waypoints at the start and end (plus the turn point).
    # We place one waypoint per leg endpoint (2 points per leg).
    raw_points: list[tuple[float, float]] = []  # (north_m, east_m)

    for i in range(n_legs):
        north_m = i * leg_spacing_m - half_span_m
        if i % 2 == 0:
            # West-to-east leg
            raw_points.append((north_m, -leg_length_m / 2.0))
            raw_points.append((north_m, leg_length_m / 2.0))
        else:
            # East-to-west leg
            raw_points.append((north_m, leg_length_m / 2.0))
            raw_points.append((north_m, -leg_length_m / 2.0))

    n_waypoints: int = len(raw_points)

    # --- 4. Apply jitter + convert to lat/lon + assign altitude ------------- #
    waypoints: list[tuple[float, float, float]] = []

    for idx, (north_m, east_m) in enumerate(raw_points):
        # Horizontal jitter
        jx = rng.uniform(-jitter_m, jitter_m)
        jy = rng.uniform(-jitter_m, jitter_m)
        jitted_north = north_m + jx
        jitted_east = east_m + jy

        # Flat-earth conversion
        lat = centroid_lat + jitted_north / _METRES_PER_DEG_LAT
        lon = centroid_lon + jitted_east / m_per_deg_lon

        # Altitude: linear ramp + ±2 m jitter
        t = idx / max(n_waypoints - 1, 1)
        alt_base = alt_floor_m + t * (alt_ceiling_m - alt_floor_m)
        alt = alt_base + rng.uniform(-2.0, 2.0)
        alt = max(alt_floor_m, min(alt_ceiling_m, alt))  # clamp to valid range

        waypoints.append((lat, lon, alt))

    # --- 5. Write CSV ------------------------------------------------------- #
    utc_now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    header = (
        f"# generated by swarm.missions on {utc_now}, "
        f"duration_s={duration_s}, "
        f"n_waypoints={n_waypoints}, "
        f"seed={actual_seed}"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for lat, lon, alt in waypoints:
            fh.write(f"{lat:.8f}, {lon:.8f}, {alt:.1f}\n")

    return out_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m swarm.missions",
        description="Generate a lawnmower-with-jitter waypoint CSV for a timed drone mission.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        required=True,
        metavar="SECONDS",
        help="Intended mission duration in seconds.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="PATH",
        help="Output file path for the generated CSV.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="INT",
        help="RNG seed for reproducible output (omit for random).",
    )
    parser.add_argument(
        "--cruise-speed",
        type=float,
        default=5.0,
        metavar="M_S",
        help="Assumed cruise speed in m/s (default: 5.0).",
    )
    parser.add_argument(
        "--centroid-lat",
        type=float,
        default=37.241861,
        metavar="DEG",
        help="Latitude of pattern centre in decimal degrees.",
    )
    parser.add_argument(
        "--centroid-lon",
        type=float,
        default=-115.796917,
        metavar="DEG",
        help="Longitude of pattern centre in decimal degrees.",
    )
    parser.add_argument(
        "--leg-spacing",
        type=float,
        default=50.0,
        metavar="M",
        help="N-S spacing between legs in metres (default: 50.0).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=5.0,
        metavar="M",
        help="Max horizontal position jitter in metres (default: 5.0).",
    )
    parser.add_argument(
        "--alt-floor",
        type=float,
        default=30.0,
        metavar="M",
        help="Altitude at first waypoint in metres AGL (default: 30.0).",
    )
    parser.add_argument(
        "--alt-ceiling",
        type=float,
        default=60.0,
        metavar="M",
        help="Altitude at last waypoint in metres AGL (default: 60.0).",
    )
    return parser


def main() -> None:
    """CLI entry-point for ``python -m swarm.missions``."""
    parser = _build_parser()
    args = parser.parse_args()

    result = generate(
        duration_s=args.duration,
        out_path=args.out,
        seed=args.seed,
        cruise_speed_m_s=args.cruise_speed,
        centroid_lat=args.centroid_lat,
        centroid_lon=args.centroid_lon,
        leg_spacing_m=args.leg_spacing,
        jitter_m=args.jitter,
        alt_floor_m=args.alt_floor,
        alt_ceiling_m=args.alt_ceiling,
    )
    print(f"Written {result}")


if __name__ == "__main__":
    main()
