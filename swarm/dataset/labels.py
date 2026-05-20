"""Attack-window label definitions for dataset annotation.

Provides dataclasses for describing labelled attack windows in a capture
session, and a ``LabelLookup`` helper for resolving a (timestamp, drone_id)
pair to the corresponding attack type string.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttackWindow:
    """An immutable record of a single labelled attack window.

    Attributes:
        start_epoch: UNIX timestamp (seconds) at which the window opens.
        end_epoch: UNIX timestamp (seconds) at which the window closes.
        target_drones: Set of drone system-ids affected by this attack.
        attack_type: Human-readable attack label (e.g. ``'attitude_spoof'``).
    """

    start_epoch: float
    end_epoch: float
    target_drones: frozenset[int]
    attack_type: str


@dataclass
class LabelLookup:
    """Resolves a (timestamp, drone_id) pair to an attack type string.

    Attributes:
        windows: Ordered list of ``AttackWindow`` records to search.
    """

    windows: list[AttackWindow] = field(default_factory=list)

    def lookup(self, frame_epoch: float, drone_id: int | None) -> str | None:
        """Return the active attack_type for (timestamp, drone_id), or ``None``.

        ``None`` is rendered as ``"null"`` by :func:`packet_writer._format_cell`,
        keeping the ``attack_type`` column visually consistent with every other
        null cell in the CSV.

        Linear scan over ``self.windows``; efficient for the small number of
        windows expected in a typical capture schedule.

        Args:
            frame_epoch: UNIX timestamp of the packet/frame being labelled.
            drone_id: MAVLink system-id of the drone, or ``None`` for
                non-MAVLink frames.

        Returns:
            The ``attack_type`` string of the first matching window, or
            ``None`` when no window applies.
        """
        if drone_id is None:
            return None
        for w in self.windows:
            if w.start_epoch <= frame_epoch <= w.end_epoch and drone_id in w.target_drones:
                return w.attack_type
        return None
