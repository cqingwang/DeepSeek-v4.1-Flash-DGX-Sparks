#!/usr/bin/env python3
"""Exercise the staged Chat and Responses methods for SGLang #45804."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


class _EnvValue:
    value = False

    @classmethod
    def get(cls):
        return cls.value


def _extract(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def _chat_serving(spec):
    return SimpleNamespace(
        reasoning_parser="deepseek-v41",
        chat_encoding_spec=spec,
        template_manager=SimpleNamespace(reasoning_config=None),
        _reasoning_detector=SimpleNamespace(reasoning_default="explicit_thinking"),
    )


def _responses_serving(spec):
    return SimpleNamespace(
        reasoning_parser="deepseek-v41",
        chat_encoding_spec=spec,
        template_manager=SimpleNamespace(
            force_reasoning=False, reasoning_config=None
        ),
        _reasoning_detector=SimpleNamespace(reasoning_default="explicit_thinking"),
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    base = Path(sys.argv[1]) / "sglang/srt/entrypoints/openai"
    helper = _extract(
        base / "chat_encoding.py",
        "spec_prompt_reads_default_thinking_env",
        {},
    )
    chat_encoding = SimpleNamespace(
        spec_prompt_reads_default_thinking_env=helper
    )
    common = {
        "chat_encoding": chat_encoding,
        "envs": SimpleNamespace(SGLANG_DEFAULT_THINKING=_EnvValue),
        "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
    }
    chat = _extract(
        base / "serving_chat.py", "_get_reasoning_from_request", dict(common)
    )
    responses = _extract(
        base / "serving_responses.py",
        "_is_thinking_enabled_for_request",
        dict(common),
    )

    assert all(helper(spec) for spec in ("dsv4", "dsv41", "dsv32"))
    assert not any(helper(spec) for spec in (None, "kimi_k3", "inkling"))

    request = lambda kwargs=None: SimpleNamespace(
        chat_template_kwargs=kwargs, reasoning_effort=None, reasoning=None
    )
    for value in (False, True):
        _EnvValue.value = value
        assert chat(_chat_serving("dsv41"), request()) is value
        assert responses(_responses_serving("dsv41"), request()) is value
        assert chat(_chat_serving("dsv41"), request({"thinking": True})) is True
        assert chat(_chat_serving("dsv41"), request({"thinking": False})) is False
        assert (
            responses(
                _responses_serving("dsv41"), request({"thinking": True})
            )
            is True
        )
        assert (
            responses(
                _responses_serving("dsv41"), request({"thinking": False})
            )
            is False
        )

    _EnvValue.value = True
    assert chat(_chat_serving(None), request()) is False
    assert responses(_responses_serving(None), request()) is False
    assert chat(_chat_serving(None), request({"thinking": True})) is True
    assert responses(_responses_serving(None), request({"thinking": True})) is True
    print("SGLang #45804 Chat/Responses thinking semantics: passed")


if __name__ == "__main__":
    main()
