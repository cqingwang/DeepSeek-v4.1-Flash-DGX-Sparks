#!/usr/bin/env python3
"""Fail-closed source verification for the SGLang #40424 adaptation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise SystemExit(f"missing {class_name}.{method_name}")


def verify(python_root: Path) -> None:
    backend_path = python_root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
    engram_path = python_root / "sglang/srt/layers/engram.py"
    backend_source = backend_path.read_text()
    engram_source = engram_path.read_text()
    backend_tree = ast.parse(backend_source, filename=str(backend_path))
    engram_tree = ast.parse(engram_source, filename=str(engram_path))

    low_ratio = _method(backend_tree, "DeepseekV4AttnBackend", "forward_low_ratio_sources")
    low_source = ast.get_source_segment(backend_source, low_ratio) or ""
    low_markers = (
        "hoisted_pos.shape[0] < positions.shape[0]",
        "num_real = hoisted_pos.shape[0]",
        "x, positions = x[:num_real], positions[:num_real]",
        "if q_lora is not None:",
        "q_lora = q_lora[:num_real]",
        "hoisted_pos.shape[0] == positions.shape[0]",
    )
    missing = [marker for marker in low_markers if marker not in low_source]
    if missing:
        raise SystemExit(f"incomplete DP-padding live-row slice: {missing}")
    if not (
        low_source.index("hoisted_pos.shape[0] < positions.shape[0]")
        < low_source.index("hoisted_pos.shape[0] == positions.shape[0]")
        < low_source.index("token_req_indices(")
    ):
        raise SystemExit("live-row slicing/hoisted selection/fallback ordering drifted")

    engram = _method(engram_tree, "EngramHasher", "forward")
    engram_source_segment = ast.get_source_segment(engram_source, engram) or ""
    engram_markers = (
        "mode = forward_batch.forward_mode",
        "if num_tokens == 0 or mode.is_idle():",
        "return torch.zeros(",
        "(num_tokens, self.primes.shape[0], self.offsets.shape[1])",
    )
    missing = [marker for marker in engram_markers if marker not in engram_source_segment]
    if missing:
        raise SystemExit(f"incomplete Engram idle-row handling: {missing}")
    if not (
        engram_source_segment.index("mode = forward_batch.forward_mode")
        < engram_source_segment.index("if num_tokens == 0 or mode.is_idle():")
        < engram_source_segment.index("req_slots = forward_batch.req_pool_indices")
        < engram_source_segment.index("assert mode.is_extend()")
    ):
        raise SystemExit("Engram idle return no longer precedes request-table access")

    print("SGLang #40424 DP-padding/idle-row fix: verified")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    verify(root)
