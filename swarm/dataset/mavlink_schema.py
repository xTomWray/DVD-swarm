"""MAVLink message schema built from pymavlink's ardupilotmega dialect.

This module is the Python equivalent of dvdsh's ``mavlink_min.json``. The
schema is constructed once at import time from the ``ardupilotmega`` v2.0
dialect, which was confirmed as the canonical dialect during Phase 1
exploration.

Only ``pymavlink`` is required as an external dependency.
"""

from pymavlink.dialects.v20 import ardupilotmega as _mav

SCHEMA: dict[int, tuple[str, list[str]]] = {
    msgid: (cls.msgname, list(cls.fieldnames))
    for msgid, cls in _mav.mavlink_map.items()
}


def lookup(msgid: int) -> tuple[str, list[str]] | None:
    """Return message name and field list for a MAVLink message id.

    Args:
        msgid: Numeric MAVLink message identifier.

    Returns:
        A ``(msg_type_name, fieldnames)`` tuple if the id is known, or
        ``None`` if the id is not present in the ardupilotmega dialect.
    """
    return SCHEMA.get(msgid)
