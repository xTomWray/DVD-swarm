"""dvdsh compatibility constants for CSV header fields.

These constants mirror dvdsh's ``lib/shell/abstract/mavlink.ts`` header-field
lists. They are referenced by ``packet_writer.py`` (Phase 3) to build CSV
headers that are byte-for-byte compatible with dvdsh's ``DUMP_CSV`` and
``JOIN_CSV`` commands.

IMPORTANT: Keep this file in sync with the upstream dvdsh TypeScript source
whenever dvdsh's field ordering or naming changes.
"""

MAV_HEADER_FIELDS: list[str] = [
    "magic",
    "payloadLength",
    "incompatibilityFlags",
    "compatibilityFlags",
    "seq",
    "sysid",
    "compid",
    "msgid",
    "checksum",
    "signature",
]

IP_HEADER_FIELDS: list[str] = [
    "version",
    "hdr_len",
    "tos",
    "len",
    "id",
    "flags",
    "frag_offset",
    "ttl",
    "proto",
    "checksum",
    "checksum_status",
    "src",
    "src_host",
    "addr",
    "host",
    "dst",
    "dst_host",
    "payload",
]

UDP_HEADER_FIELDS: list[str] = [
    "srcport",
    "dstport",
    "length",
    "checksum",
    "checksum_status",
    "payload",
    "text",
]

TCP_HEADER_FIELDS: list[str] = [
    "srcport",
    "dstport",
    "seq",
    "ack",
    "hdr_len",
    "flags",
    "flags_str",
    "window_size",
    "checksum",
    "checksum_status",
    "urgent_pointer",
    "options",
    "options_nop",
    "options_timestamp",
    "payload",
    "text",
]
