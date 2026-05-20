"""Utilities for resolving drone target endpoints and parsing target specs."""

from __future__ import annotations


def gcs_endpoint(drone_id: int) -> tuple[str, int]:
    """GCS UDP MAVLink endpoint for the given drone instance.

    Args:
        drone_id: Numeric drone identifier (positive integer).

    Returns:
        A (host, port) tuple for the drone's GCS MAVLink UDP endpoint.
    """
    return (f"10.13.{drone_id}.4", 14550)


def companion_endpoint(drone_id: int) -> tuple[str, int]:
    """Companion-computer TCP MAVLink endpoint for the given drone instance.

    Attack traffic sent here flows through mavlink-routerd, which routes it
    to the SITL serial link, the GCS UDP endpoint, and all TCP clients
    (including the dataset sniffer). This ensures spoofed packets appear in
    the captured dataset with their actual field values.

    Args:
        drone_id: Numeric drone identifier (positive integer).

    Returns:
        A (host, port) tuple for the drone's companion TCP MAVLink endpoint.
    """
    return (f"10.13.{drone_id}.3", 5760)


def parse_targets(spec: str) -> list[int]:
    """Parse a comma-separated drone list with optional ranges.

    Accepts inputs like ``'1,3,5-10'`` and returns a sorted, deduplicated
    list of drone IDs.

    Args:
        spec: Comma-separated list of IDs or inclusive ranges, e.g. ``'1,3,5-10'``.

    Returns:
        Sorted, deduplicated list of drone IDs.

    Raises:
        ValueError: If the spec is empty, malformed, contains non-positive IDs,
            or specifies a range where the start exceeds the end.
    """
    if not spec.strip():
        raise ValueError("Target spec must not be empty.")

    ids: set[int] = set()

    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"Empty token in target spec: {spec!r}")

        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid range expression {token!r} in spec {spec!r}"
                )
            start_str, end_str = parts
            if not start_str.strip() or not end_str.strip():
                raise ValueError(
                    f"Invalid range expression {token!r} in spec {spec!r}"
                )
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise ValueError(
                    f"Non-integer value in range {token!r} in spec {spec!r}"
                )
            if start <= 0 or end <= 0:
                raise ValueError(
                    f"Drone IDs must be positive; got range {token!r} in spec {spec!r}"
                )
            if start > end:
                raise ValueError(
                    f"Range start {start} exceeds end {end} in spec {spec!r}"
                )
            ids.update(range(start, end + 1))
        else:
            try:
                drone_id = int(token)
            except ValueError:
                raise ValueError(
                    f"Non-integer drone ID {token!r} in spec {spec!r}"
                )
            if drone_id <= 0:
                raise ValueError(
                    f"Drone ID must be positive; got {drone_id} in spec {spec!r}"
                )
            ids.add(drone_id)

    return sorted(ids)
