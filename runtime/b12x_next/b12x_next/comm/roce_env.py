"""Environment parsing shared by B12X RoCE adapters."""

from __future__ import annotations

import os


def required_roce_gid_index(*names: str) -> int:
    """Read a configured non-negative GID index; never assume a site default."""

    for name in names:
        raw = os.getenv(name, "").strip()
        if not raw:
            continue
        try:
            index = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative integer, got {raw!r}") from exc
        if index < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {index}")
        return index
    if not names:
        raise ValueError("at least one RoCE GID index environment key must be named")
    raise RuntimeError(f"RoCE GID index must be configured in one of: {', '.join(names)}")
