#!/usr/bin/env python3
"""CPU behavior checks for the staged SGLang #45450 draft metadata glue."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def extract(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(nodes) != 1:
        raise AssertionError(f"expected one {name}, found {len(nodes)}")
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            nodes[0],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


class FakeTensor:
    _next_ptr = 1000

    def __init__(self, values, ptr=None):
        self.values = list(values)
        self.shape = (len(self.values),)
        if ptr is None:
            ptr = FakeTensor._next_ptr
            FakeTensor._next_ptr += 100
        self._ptr = ptr

    def __getitem__(self, item):
        values = self.values[item]
        if not isinstance(values, list):
            values = [values]
        return FakeTensor(values, self._ptr)

    def data_ptr(self):
        return self._ptr


class FakeCuda:
    available = False
    capturing = False

    @classmethod
    def is_available(cls):
        return cls.available

    @classmethod
    def is_current_stream_capturing(cls):
        return cls.capturing


def backend(**overrides):
    fields = dict(
        is_dspark_draft=True,
        trtllm_attn=False,
        online_c128_mtp=SimpleNamespace(enabled=lambda: False),
        token_to_kv_pool=SimpleNamespace(request_window=None),
        draft_metadata_glued=False,
        _draft_extend_lens={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    root = Path(sys.argv[1]) / "sglang/srt"
    dsv4_path = root / "layers/attention/deepseek_v4_backend.py"
    runner_path = root / "model_executor/runner/decode_cuda_graph_runner.py"

    parallel = SimpleNamespace(attn_cp_size=1)
    glue_guard = extract(
        dsv4_path,
        "glue_draft_metadata",
        {"get_parallel": lambda: parallel},
    )
    audited = backend()
    assert glue_guard(audited) == "" and audited.draft_metadata_glued
    for expected, changes in (
        ("not the DSpark draft runner", {"is_dspark_draft": False}),
        ("trtllm-gen attention", {"trtllm_attn": True}),
        (
            "online c128 MTP",
            {"online_c128_mtp": SimpleNamespace(enabled=lambda: True)},
        ),
        (
            "a request window",
            {"token_to_kv_pool": SimpleNamespace(request_window=object())},
        ),
    ):
        candidate = backend(**changes)
        assert glue_guard(candidate) == expected
        assert not candidate.draft_metadata_glued
    parallel.attn_cp_size = 2
    assert glue_guard(backend()) == "context parallelism"

    fake_torch = SimpleNamespace(cuda=FakeCuda)
    cache_method = extract(
        dsv4_path,
        "_draft_block_extend_lens",
        {"torch": fake_torch},
    )
    owner = backend(draft_metadata_glued=True)
    moves = []

    def move(values):
        moves.append(list(values))
        return FakeTensor(values)

    owner._move_to_device = move
    first = cache_method(owner, [6, 6])
    assert cache_method(owner, [6, 6]) is first
    cache_method(owner, [6])
    assert moves == [[6, 6], [6]]
    FakeCuda.available = FakeCuda.capturing = True
    assert cache_method(owner, [6, 6]) is first
    try:
        cache_method(owner, [6, 6, 6])
    except RuntimeError as exc:
        assert "first seen inside" in str(exc)
    else:
        raise AssertionError("new capture-time constant was accepted")
    FakeCuda.available = FakeCuda.capturing = False

    rebind = extract(runner_path, "_glue_draft_inputs", {})
    slot = FakeTensor(range(16))
    step = FakeTensor(range(6))
    runner = SimpleNamespace(
        _draft_metadata_glue=False,
        buffers=SimpleNamespace(out_cache_loc=slot),
    )
    view = SimpleNamespace(out_cache_loc=step)
    rebind(runner, view)
    assert view.out_cache_loc is step
    runner._draft_metadata_glue = True
    rebind(runner, view)
    assert view.out_cache_loc.data_ptr() == slot.data_ptr()
    assert view.out_cache_loc.shape == step.shape

    print("SGLang #45450 draft metadata-glue behavior: PASS")


if __name__ == "__main__":
    main()
