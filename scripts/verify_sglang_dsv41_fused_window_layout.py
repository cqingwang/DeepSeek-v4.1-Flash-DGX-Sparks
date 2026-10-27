#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree SGLang #45726 adaptation."""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path


UPSTREAM_KERNEL_SHA256 = (
    "4ca6d7d7d79340bb47b9f960d1cf4828674ede86c46f1858f04fe2b6165eb171"
)


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name}, found {len(matches)}")
    return matches[0]


def _source(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def verify(python_root: Path) -> None:
    kernel_path = (
        python_root / "sglang/kernels/ops/kvcache/dsv41_window_layout.py"
    )
    digest = hashlib.sha256(kernel_path.read_bytes()).hexdigest()
    if digest != UPSTREAM_KERNEL_SHA256:
        raise SystemExit(f"unexpected fused layout kernel digest: {digest}")

    kernel_text = kernel_path.read_text()
    kernel_tree = ast.parse(kernel_text, filename=str(kernel_path))
    builder_source = _source(kernel_text, _function(kernel_tree, "build_window_layout"))
    required_kernel = (
        "_window_layout_queries[(triton.cdiv(padded, 4),)](",
        "_window_layout_history[(triton.cdiv(history_rows, 256),)](",
        "starts = torch.full((groups,), -1",
        'return dict(',
        "history_src=hist_src",
        "size=history_rows + padded",
    )
    missing = [item for item in required_kernel if item not in builder_source]
    if missing:
        raise SystemExit(f"incomplete upstream fused layout kernel: {missing}")

    query_source = _source(kernel_text, _function(kernel_tree, "_window_layout_queries"))
    if "(group < GROUPS)" not in query_source:
        raise SystemExit("query kernel does not bound request-group starts")
    for runtime_name in ("N", "PAD", "HISTORY", "GROUPS"):
        arg = next(
            arg
            for arg in _function(kernel_tree, "_window_layout_queries").args.args
            if arg.arg == runtime_name
        )
        if arg.annotation is not None:
            raise SystemExit(f"{runtime_name} unexpectedly became compile-time")

    window_path = python_root / "sglang/srt/mem_cache/dsv41_request_window.py"
    window_text = window_path.read_text()
    window_tree = ast.parse(window_text, filename=str(window_path))
    layout_source = _source(window_text, _function(window_tree, "window_layout"))
    required_integration = (
        "if req.is_cuda and torch.version.hip is None:",
        "group, first, last = _first_row_offsets(req)",
        "expect(_WINDOW_GROUPS_FIT, group[-1:] < num_groups, msg=\"request runs\")",
        "**build_window_layout(",
        "zero_row=zero_row",
        "padded_rows=None",
    )
    missing = [item for item in required_integration if item not in layout_source]
    if missing:
        raise SystemExit(f"incomplete CUDA integration: {missing}")
    if layout_source.index("if req.is_cuda") > layout_source.index(
        "req = req.to(torch.int64)"
    ):
        raise SystemExit("CUDA fused path runs after the eager fallback")

    init_text = (python_root / "sglang/kernels/ops/kvcache/__init__.py").read_text()
    if '("dsv41_window_layout", "build_window_layout")' not in init_text:
        raise SystemExit("fused layout kernel is absent from kernel inventory")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45726 fused request-window layout: verified")


if __name__ == "__main__":
    main()
