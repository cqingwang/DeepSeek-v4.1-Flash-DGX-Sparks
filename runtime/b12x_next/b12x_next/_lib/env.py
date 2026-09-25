"""Environment-variable conventions for b12x_next."""

from __future__ import annotations

import os

PREFIX = "B12X_NEXT_"


def env_raw(suffix: str) -> str | None:
    """Read a b12x_next knob by suffix, e.g. ``env_raw("MLA_FORCE_SPLIT")``."""
    return os.environ.get(PREFIX + suffix)


def env_flag(suffix: str, *, default: bool = False) -> bool:
    raw = env_raw(suffix)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


__all__ = ["PREFIX", "env_raw", "env_flag"]
