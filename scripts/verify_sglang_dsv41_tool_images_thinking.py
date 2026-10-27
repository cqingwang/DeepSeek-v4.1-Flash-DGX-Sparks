#!/usr/bin/env python3
"""Fail-closed source verifier for the pinned-tree SGLang #44751 port."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def verify(root: Path) -> None:
    encoder = root / "sglang/srt/entrypoints/openai/encoding_dsv41.py"
    if not encoder.is_file():
        raise SystemExit(f"missing dsv41 encoder: {encoder}")
    source = encoder.read_text()
    required = (
        "User messages containing only tool results do not start a new turn.",
        'block.get("type") in ("text", "image", "image_url")',
        'content = msg["content_blocks"]',
        "context = copy.deepcopy(context or [])",
        "messages = copy.deepcopy(messages)",
        "# Validate before merging can discard message-level textual fields.",
        "# Collect images only after tool results are in their final prompt order.",
        "messages = sort_tool_results_by_call_order(context + messages)[len(context) :]",
    )
    for anchor in required:
        if source.count(anchor) != 1:
            raise SystemExit(f"SGLang #44751 encoder anchor drifted: {anchor}")

    tree = ast.parse(source, filename=str(encoder))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    encode = functions.get("encode_messages")
    text_encode = functions.get("_encode_messages_text")
    if encode is None or text_encode is None:
        raise SystemExit("SGLang #44751 encoder functions are missing")

    encode_calls = [
        node.func.id
        for node in ast.walk(encode)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    for name in (
        "_validate_no_image_sp_tokens",
        "merge_tool_messages",
        "sort_tool_results_by_call_order",
        "process_image_messages",
        "_encode_messages_text",
    ):
        if name not in encode_calls:
            raise SystemExit(f"encode_messages no longer calls {name}")
    if any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"merge_tool_messages", "sort_tool_results_by_call_order"}
        for node in ast.walk(text_encode)
    ):
        raise SystemExit("tool preprocessing moved back after image collection")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #44751 tool-image order and current-turn thinking verified")
