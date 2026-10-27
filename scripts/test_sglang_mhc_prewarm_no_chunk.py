#!/usr/bin/env python3
"""CPU-only extracted behavior test for the SGLang #43448 fallback."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def _probe(path: Path):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    method = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prewarm_mhc_pre"
    )
    start = next(
        i
        for i, stmt in enumerate(method.body)
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Tuple) for t in stmt.targets)
        and "hc_mult" in ast.unparse(stmt.targets[0])
    )
    body = method.body[start : start + 5]
    body.append(
        ast.Return(
            ast.Tuple(
                [ast.Name("max_num_tokens", ast.Load()), ast.Name("buckets", ast.Load())],
                ast.Load(),
            )
        )
    )
    probe = ast.FunctionDef(
        name="probe",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="residual")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )
    module = ast.Module(body=[probe], type_ignores=[])
    ast.fix_missing_locations(module)
    state = SimpleNamespace(schedule=None, calls=[])

    def get_schedule():
        return state.schedule

    def representatives(max_num_tokens, hc_hidden_size):
        state.calls.append((max_num_tokens, hc_hidden_size))
        return (max_num_tokens, hc_hidden_size)

    namespace = {
        "get_schedule": get_schedule,
        "get_mhc_pre_token_count_representatives": representatives,
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["probe"], state


def run(python_root: Path) -> None:
    path = python_root / "sglang/kernels/ops/layernorm/mhc.py"
    probe, state = _probe(path)
    residual = SimpleNamespace(shape=(1, 4, 7168))
    for configured, budget, expected in (
        (2048, 8192, 2048),
        (-1, 8192, 8192),
        (0, 4096, 4096),
        (None, 16384, 16384),
    ):
        state.schedule = SimpleNamespace(
            chunked_prefill_size=configured, max_prefill_tokens=budget
        )
        max_tokens, buckets = probe(residual)
        assert max_tokens == expected
        assert buckets == (expected, 4 * 7168)
    assert state.calls == [(2048, 28672), (8192, 28672), (4096, 28672), (16384, 28672)]
    print("PASS SGLang #43448 no-chunk MHC prewarm behavior")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    run(root)
