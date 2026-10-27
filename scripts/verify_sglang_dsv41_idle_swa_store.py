#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree SGLang #45586/#45699 port."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _method_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name} in {path}, found {len(matches)}")
    return ast.get_source_segment(source, matches[0]) or ""


def verify(python_root: Path) -> None:
    backend = (
        python_root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
    )
    window = python_root / "sglang/srt/mem_cache/dsv41_request_window.py"
    idle = _method_source(window, "activate_idle")
    store = _method_source(backend, "get_swa_out_cache_loc")
    init = _method_source(window, "__init__")

    idle_required = (
        "layout = self._idle_layout",
        "layout.pos.numel() != num_rows",
        "zeros = torch.zeros(",
        "zero_row=self.zero_row",
        "capacity=self.capacity",
        "num_groups=1",
        "self._idle_layout = layout",
        "self.activate(layout)",
        "return layout.write_loc",
    )
    missing = [item for item in idle_required if item not in idle]
    if missing:
        raise SystemExit(f"incomplete idle request-window layout: {missing}")
    if "self._idle_layout = None" not in init:
        raise SystemExit("idle request-window cache is not initialized")

    store_required = (
        "out_cache_loc = forward_batch.out_cache_loc",
        "window = self.token_to_kv_pool.request_window",
        "if forward_batch.forward_mode.is_idle():",
        "return window.activate_idle(out_cache_loc.shape[0])",
        "translate_loc_from_full_to_swa(",
        "metadata = self.forward_metadata",
        "window.activate(layout)",
    )
    missing = [item for item in store_required if item not in store]
    if missing:
        raise SystemExit(f"incomplete idle SWA store routing: {missing}")
    if store.index("metadata = self.forward_metadata") < store.index(
        "if forward_batch.forward_mode.is_idle():"
    ):
        raise SystemExit("idle path still reads potentially stale attention metadata")
    if "and not forward_batch.forward_mode.is_idle()" in store:
        raise SystemExit("late idle guard survived after the early return")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45586/#45699 idle SWA-store routing: verified")


if __name__ == "__main__":
    main()
