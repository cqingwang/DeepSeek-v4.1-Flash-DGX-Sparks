#!/usr/bin/env python3
"""Fail-closed source verifier for SGLang #45630 commit 741aea0a."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(python_root: Path) -> None:
    target = python_root / "sglang/srt/models/deepseek_v4.py"
    source = target.read_text()
    tree = ast.parse(source, filename=str(target))

    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "autotune_prefill_kernels"
    ]
    if len(methods) != 1:
        raise SystemExit(
            f"expected one autotune_prefill_kernels method, found {len(methods)}"
        )

    values = []
    for node in ast.walk(methods[0]):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target_node = node.targets[0]
        if (
            isinstance(target_node, ast.Attribute)
            and target_node.attr == "mxfp8_prefill_autotune_min_tokens"
            and isinstance(node.value, ast.Constant)
        ):
            values.append(node.value.value)

    if values != [1]:
        raise SystemExit(
            "SGLang #45630 MXFP8 tactic threshold is missing or drifted: "
            f"found {values!r}"
        )

    method_source = ast.get_source_segment(source, methods[0]) or ""
    required = (
        "backend.is_flashinfer_cutedsl()",
        "if method.block_fp8_as_mxfp8:",
        "method.mxfp8_prefill_autotune_min_tokens = 1",
        "method.apply(layer, x)",
    )
    missing = [needle for needle in required if needle not in method_source]
    if missing:
        raise SystemExit(f"incomplete SGLang #45630 port: {missing}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45630 MXFP8 decode tactics: verified")


if __name__ == "__main__":
    main()
