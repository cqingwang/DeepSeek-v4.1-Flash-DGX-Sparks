#!/usr/bin/env python3
"""Fail-closed verification for the SGLang #43448 pinned-tree port."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    path = python_root / "sglang/kernels/ops/layernorm/mhc.py"
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    method = next(
        (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prewarm_mhc_pre"),
        None,
    )
    if method is None:
        raise SystemExit("missing prewarm_mhc_pre")
    segment = ast.get_source_segment(source, method) or ""
    markers = (
        "schedule = get_schedule()",
        "max_num_tokens = schedule.chunked_prefill_size",
        "if max_num_tokens is None or max_num_tokens <= 0:",
        "max_num_tokens = schedule.max_prefill_tokens",
        "get_mhc_pre_token_count_representatives(\n        max_num_tokens",
    )
    missing = [marker for marker in markers if marker not in segment]
    if missing:
        raise SystemExit(f"incomplete no-chunk MHC prewarm fallback: {missing}")
    if not (
        segment.index("max_num_tokens = schedule.chunked_prefill_size")
        < segment.index("if max_num_tokens is None or max_num_tokens <= 0:")
        < segment.index("max_num_tokens = schedule.max_prefill_tokens")
        < segment.index("get_mhc_pre_token_count_representatives(")
    ):
        raise SystemExit("MHC prewarm fallback ordering drifted")
    print("SGLang #43448 no-chunk MHC prewarm fallback: verified")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    verify(root)
