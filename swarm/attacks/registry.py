"""Registry mapping attack names to their handler callables."""

from __future__ import annotations

import threading
from collections.abc import Callable

from . import attitude_spoof

AttackHandler = Callable[[str, int, float, float, threading.Event], None]

ATTACK_HANDLERS: dict[str, AttackHandler] = {
    "attitude-spoof": attitude_spoof.send_loop,
    "attitude_spoof": attitude_spoof.send_loop,  # accept both spellings
}
