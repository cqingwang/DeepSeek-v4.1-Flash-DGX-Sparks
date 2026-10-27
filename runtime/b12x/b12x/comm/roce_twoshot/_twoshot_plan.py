"""Two-shot geometry and routing, torch-free (shared by the runtime, the CPU tests and the probe).

The pipelined two-shot all-reduce (flashinfer-ai/flashinfer#6915, Luke Alonso) splits a
``P``-pack message (16-byte packs) into ``world`` shards, shard ``j`` covering packs
``[j P / world, (j + 1) P / world)``, and every shard into ``chunks`` chunks of
``chunk_packs``.  Rank ``j`` reduces shard ``j`` in fixed rank order (FP32, sources
0..world-1, the same order and the same rounding as the one-shot kernel) and every other
rank copies that shard's bits, so the result is bit-identical to the one-shot on every rank.

Routing is a pure function of shape, dtype size and replicated configuration, never of a
pointer, a clock or node state: every tensor-parallel rank takes the same decision for the
same collective (see sglang-scheduler-decisions-must-be-rank-invariant in the ds41 notes).
"""

from __future__ import annotations

PACK_BYTES = 16
# Proxy limits (_roce_proxy.c): ROCE_MAX_CHUNKS chunk flags per (phase, source).
MAX_CHUNKS = 64
# flashinfer#6915 defaults, measured there on DGX Sparks behind one 200 GbE switch: 4 blocks
# and 128 KiB shard chunks reach about 96 % of the wire from 16 MiB up.  The ds41 production
# profile sets B12X_ROCE_PIPE_CHUNK_BYTES=65536 for its 61 KB..983 KB decode payloads.
DEFAULT_PIPE_BLOCKS = 4
DEFAULT_PIPE_CHUNK_BYTES = 128 * 1024
# 0 = never: the two-shot path is opt-in per row threshold (DSV41_ROCE_TWOSHOT_ROWS).
DEFAULT_MIN_ROWS = 0


def shard_start(shard: int, total_packs: int, world: int) -> int:
    """First pack of ``shard`` (the kernel and the proxy split identically)."""
    return (int(shard) * int(total_packs)) // int(world)


def pipe_geometry(
    nbytes: int, world: int, chunk_packs_min: int, max_chunks: int = MAX_CHUNKS
) -> tuple[int, int]:
    """(chunk_packs, chunks) splitting every shard of an ``nbytes`` message."""
    if nbytes <= 0 or nbytes % PACK_BYTES:
        raise ValueError(f"two-shot message must be a positive multiple of 16 bytes, got {nbytes}")
    shard = -(-(nbytes // PACK_BYTES) // int(world))
    chunk_packs = max(int(chunk_packs_min), -(-shard // int(max_chunks)))
    return chunk_packs, -(-shard // chunk_packs)


def rows_of(shape: tuple[int, ...]) -> int:
    """Rows of a collective's tensor: the leading dimension of a 2-D or wider tensor, else 0."""
    return int(shape[0]) if len(shape) >= 2 else 0


def uses_pipe(rows: int, nbytes: int, *, pipe_bytes: int, min_rows: int) -> bool:
    """True when an all-reduce of ``rows`` rows and ``nbytes`` bytes takes the two-shot path.

    ``min_rows <= 0`` is off (every all-reduce stays one-shot, the production behaviour).
    """
    return (
        int(min_rows) > 0
        and int(pipe_bytes) > 0
        and int(rows) >= int(min_rows)
        and 0 < int(nbytes) <= int(pipe_bytes)
        and int(nbytes) % PACK_BYTES == 0
    )


__all__ = [
    "DEFAULT_MIN_ROWS",
    "DEFAULT_PIPE_BLOCKS",
    "DEFAULT_PIPE_CHUNK_BYTES",
    "MAX_CHUNKS",
    "PACK_BYTES",
    "pipe_geometry",
    "rows_of",
    "shard_start",
    "uses_pipe",
]
