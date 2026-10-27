#!/usr/bin/env python3
"""Fail-closed verifier for the pinned-tree SGLang #45804 adaptation."""

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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name} in {path}, found {len(matches)}")
    return ast.get_source_segment(source, matches[0]) or ""


def verify(python_root: Path) -> None:
    base = python_root / "sglang/srt/entrypoints/openai"
    encoding = (base / "chat_encoding.py").read_text()
    chat = _method_source(base / "serving_chat.py", "_get_reasoning_from_request")
    responses = _method_source(
        base / "serving_responses.py", "_is_thinking_enabled_for_request"
    )

    helper = (
        'def spec_prompt_reads_default_thinking_env(spec: Optional[str]) -> bool:',
        'return spec in ("dsv4", "dsv41", "dsv32")',
    )
    missing = [needle for needle in helper if needle not in encoding]
    if missing:
        raise SystemExit(f"missing native-encoder scope helper: {missing}")

    chat_required = (
        "chat_encoding.spec_prompt_reads_default_thinking_env(",
        "self.chat_encoding_spec",
        'if "thinking" in chat_template_kwargs:',
        'return chat_template_kwargs["thinking"] is True',
        "return envs.SGLANG_DEFAULT_THINKING.get()",
    )
    missing = [needle for needle in chat_required if needle not in chat]
    if missing:
        raise SystemExit(f"incomplete Chat parity port: {missing}")
    if 'if toggle == "thinking":' in chat:
        raise SystemExit("over-broad parser-family env fallback survived in Chat")
    chat_scope = chat.index("spec_prompt_reads_default_thinking_env")
    if chat.find("config = self.template_manager.reasoning_config", chat_scope) < 0:
        raise SystemExit("Chat native-encoder parity check runs after template fallback")

    responses_required = (
        "chat_encoding.spec_prompt_reads_default_thinking_env(",
        "self.chat_encoding_spec",
        "return envs.SGLANG_DEFAULT_THINKING.get()",
    )
    missing = [needle for needle in responses_required if needle not in responses]
    if missing:
        raise SystemExit(f"incomplete Responses parity port: {missing}")
    responses_scope = responses.index("spec_prompt_reads_default_thinking_env")
    if responses.find(
        "config = self.template_manager.reasoning_config", responses_scope
    ) < 0:
        raise SystemExit(
            "Responses native-encoder parity check runs after template fallback"
        )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #45804 native DeepSeek thinking parity: verified")


if __name__ == "__main__":
    main()
