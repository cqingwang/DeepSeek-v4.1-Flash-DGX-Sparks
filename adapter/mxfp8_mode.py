"""Shared configuration semantics for the optional MXFP8 overlay."""

DISABLED_MXFP8_BACKENDS = frozenset(("", "cutlass", "off", "0"))


def mxfp8_backend_disabled(value):
    return value.strip().lower() in DISABLED_MXFP8_BACKENDS
