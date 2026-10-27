#!/usr/bin/env python3
"""Fail-closed source verification for the SGLang #45301 pinned-tree port."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise SystemExit(f"missing class {name}")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"missing method {cls.name}.{name}")


def verify(python_root: Path) -> None:
    backend_path = python_root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
    model_path = python_root / "sglang/srt/models/deepseek_v4.py"
    backend_source = backend_path.read_text()
    model_source = model_path.read_text()
    backend_tree = ast.parse(backend_source, filename=str(backend_path))
    model_tree = ast.parse(model_source, filename=str(model_path))

    metadata = _class(backend_tree, "DSV4Metadata")
    captured = [
        node
        for node in metadata.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "q_pad_captured"
    ]
    if len(captured) != 1 or not isinstance(captured[0].value, ast.Constant):
        raise SystemExit("q_pad_captured must be one dataclass field with a literal default")
    if captured[0].value.value is not False:
        raise SystemExit("q_pad_captured must default to False")

    attention_base = _class(model_tree, "MqaAttentionBase")
    helper = _method(attention_base, "_zero_padded_q")
    helper_source = ast.get_source_segment(model_source, helper) or ""
    required_helper = (
        'buf.shape[0] < x.shape[0]',
        'buf.shape[1:] != (kernel_num_heads, self.head_dim)',
        'assert not getattr(meta, "q_pad_captured", False)',
        'torch.cuda.is_current_stream_capturing()',
        'meta.q_pad_captured = True',
        'return buf[: x.shape[0]]',
    )
    missing = [marker for marker in required_helper if marker not in helper_source]
    if missing:
        raise SystemExit(f"incomplete graph-owned padded-query helper: {missing}")

    attention = _class(model_tree, "MQALayer")
    forward = _method(attention, "forward")
    calls = [
        node
        for node in ast.walk(forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_zero_padded_q"
    ]
    if len(calls) != 1 or len(calls[0].args) != 3:
        raise SystemExit("forward must delegate the V4.1 padded-query allocation exactly once")

    forward_source = ast.get_source_segment(model_source, forward) or ""
    if 'want = (x.shape[0], kernel_num_heads, self.head_dim)' in forward_source:
        raise SystemExit("old exact-shape replacement path is still present")

    print("SGLang #45301 graph-owned padded-query buffer: verified")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    verify(root)
