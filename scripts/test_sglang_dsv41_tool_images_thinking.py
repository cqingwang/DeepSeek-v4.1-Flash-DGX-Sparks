#!/usr/bin/env python3
"""Behavior tests for the pinned-tree SGLang #44751 encoder port."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


def load_encoder(root: Path):
    path = root / "sglang/srt/entrypoints/openai/encoding_dsv41.py"
    spec = importlib.util.spec_from_file_location("tested_encoding_dsv41", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_order(enc) -> None:
    image = enc.IMAGE_PLACEHOLDER
    history = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "history.png"}}]},
        {
            "role": "assistant",
            "tool_calls": [
                {"id": name, "type": "function", "function": {"name": name, "arguments": "{}"}}
                for name in ("a", "b")
            ],
        },
    ]
    results = [
        {
            "type": "tool_result",
            "tool_use_id": name,
            "content": [
                {"type": "text", "text": f"result {name}"},
                {"type": "image_url", "image_url": {"url": f"{name}.png"}},
            ],
        }
        for name in ("b", "a")
    ]
    for input_format in ("content_blocks", "tool_messages", "tool_content_blocks", "context"):
        if input_format in ("tool_messages", "tool_content_blocks"):
            field = "content_blocks" if input_format == "tool_content_blocks" else "content"
            messages = history + [
                {"role": "tool", "tool_call_id": block["tool_use_id"], field: block["content"]}
                for block in results
            ]
            context = []
        elif input_format == "context":
            messages = [{"role": "user", "content_blocks": results}]
            context = history
        else:
            messages = history + [{"role": "user", "content_blocks": results}]
            context = []
        original = copy.deepcopy((messages, context))
        prompt, media = enc.encode_messages(
            messages,
            thinking_mode="chat",
            context=context,
            return_multi_modal_data=True,
        )
        expected_tail = (
            f"<｜User｜><tool_result>result a\n\n{image}</tool_result>\n\n"
            f"<tool_result>result b\n\n{image}</tool_result><｜Assistant｜></think>"
        )
        assert prompt.endswith(expected_tail), (input_format, prompt[-400:])
        expected_urls = ["a.png", "b.png"] if context else ["history.png", "a.png", "b.png"]
        assert media == {
            "images": [{"type": "image", "url": url} for url in expected_urls]
        }, (input_format, media)
        assert (messages, context) == original, f"{input_format} mutated caller input"


def test_current_turn_thinking(enc) -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
        {"role": "user", "content": "q2"},
        {
            "role": "assistant",
            "reasoning_content": "r2",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "value"},
    ]
    prompt = enc.encode_messages(messages, thinking_mode="thinking", drop_thinking=True)
    assert "r1" not in prompt
    assert "<think>r2</think>" in prompt
    assert prompt.endswith("<tool_result>value</tool_result><｜Assistant｜><think>")
    next_turn = enc.encode_messages(
        messages + [{"role": "user", "content": "q3"}], thinking_mode="thinking"
    )
    assert "r2" not in next_turn
    assert next_turn.endswith("q3<｜Assistant｜><think>")


def test_sanitizer_keeps_inputs_private(enc) -> None:
    messages = [{"role": "user", "content": f"literal {enc.IMAGE_PLACEHOLDER}"}]
    original = copy.deepcopy(messages)
    prompt = enc.encode_messages(messages, thinking_mode="chat")
    assert "literal [image]" in prompt
    assert enc.IMAGE_PLACEHOLDER not in prompt
    assert messages == original


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    encoder = load_encoder(Path(sys.argv[1]))
    test_image_order(encoder)
    test_current_turn_thinking(encoder)
    test_sanitizer_keeps_inputs_private(encoder)
    print("PASS SGLang #44751 tool-image/thinking behavior")


if __name__ == "__main__":
    main()
