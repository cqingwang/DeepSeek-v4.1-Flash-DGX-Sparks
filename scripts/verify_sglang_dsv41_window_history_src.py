#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree SGLang #45579 adaptation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name}, found {len(matches)}")
    return matches[0]


def _class(tree: ast.AST, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name}, found {len(matches)}")
    return matches[0]


def _source(text: str, node: ast.AST) -> str:
    return ast.get_source_segment(text, node) or ""


def verify(python_root: Path) -> None:
    window_path = python_root / "sglang/srt/mem_cache/dsv41_request_window.py"
    backend_path = (
        python_root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
    )
    window_text = window_path.read_text()
    backend_text = backend_path.read_text()
    window_tree = ast.parse(window_text, filename=str(window_path))
    backend_tree = ast.parse(backend_text, filename=str(backend_path))

    layout = _class(window_tree, "WindowLayout")
    annotated = {
        node.target.id
        for node in layout.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    if "history_src" not in annotated:
        raise SystemExit("WindowLayout does not carry history_src")
    copy_source = _source(window_text, _function(layout, "copy_"))
    if "self.history_src.copy_(other.history_src)" not in copy_source:
        raise SystemExit("graph replay does not refresh history_src")

    builder = _function(window_tree, "window_layout")
    kwonly = {arg.arg for arg in builder.args.kwonlyargs}
    if "zero_row" not in kwonly:
        raise SystemExit("window_layout does not require zero_row")
    builder_source = _source(window_text, builder)
    formula = (
        "history_valid, history_req * capacity + history_pos % capacity, zero_row"
    )
    if formula not in " ".join(builder_source.split()):
        raise SystemExit("history_src is not computed from the upstream row formula")
    # The later #45726 fused CUDA path returns a WindowLayout before entering
    # this CPU fallback; compare against the fallback's final constructor, not
    # the first (CUDA) constructor in the function.
    if builder_source.index("history_src = torch.where(") > builder_source.rindex(
        "return WindowLayout("
    ):
        raise SystemExit("history_src is computed after layout construction")

    request_window = _class(window_tree, "RequestWindow")
    methods = {
        node.name
        for node in request_window.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "_history_src" in methods:
        raise SystemExit("per-layer _history_src recomputation survived")
    buffer_source = _source(window_text, _function(request_window, "buffer"))
    if "src = layout.history_src" not in buffer_source:
        raise SystemExit("RequestWindow.buffer does not reuse layout.history_src")

    calls = [
        node
        for node in ast.walk(backend_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "window_layout"
    ]
    if len(calls) != 1:
        raise SystemExit(f"expected one backend window_layout call, found {len(calls)}")
    keywords = {kw.arg: kw.value for kw in calls[0].keywords}
    zero = keywords.get("zero_row")
    expected_zero = "self.token_to_kv_pool.request_window.zero_row"
    if zero is None or ast.unparse(zero) != expected_zero:
        raise SystemExit("backend does not pass the active request window's zero row")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45579 request-window history-source reuse: verified")


if __name__ == "__main__":
    main()
