#!/usr/bin/env python3
"""CPU-only behavior falsifier for the production SGLang #45301 helper."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


class FakeTensor:
    allocations: list["FakeTensor"] = []

    def __init__(self, shape, dtype="bf16", *, storage=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = self if storage is None else storage

    def new_zeros(self, *shape):
        out = FakeTensor(shape, self.dtype)
        self.allocations.append(out)
        return out

    def __getitem__(self, item):
        if not isinstance(item, slice) or item.start is not None or item.step is not None:
            raise AssertionError(f"unexpected slice {item!r}")
        rows = self.shape[0] if item.stop is None else min(item.stop, self.shape[0])
        return FakeTensor((rows, *self.shape[1:]), self.dtype, storage=self.storage)


class FakeCuda:
    capturing = False

    @classmethod
    def is_current_stream_capturing(cls):
        return cls.capturing


class FakeTorch:
    Tensor = FakeTensor
    cuda = FakeCuda


def _load_helper(python_root: Path):
    path = python_root / "sglang/srt/models/deepseek_v4.py"
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MqaAttentionBase":
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == "_zero_padded_q":
                    module = ast.Module(body=[member], type_ignores=[])
                    ast.fix_missing_locations(module)
                    namespace = {"torch": FakeTorch}
                    exec(compile(module, str(path), "exec"), namespace)
                    return namespace[member.name]
    raise AssertionError("production _zero_padded_q helper not found")


def run(python_root: Path) -> None:
    helper = _load_helper(python_root)
    host = SimpleNamespace(head_dim=128)
    meta = SimpleNamespace(q_pad_buffer=None, q_pad_captured=False)
    backend = SimpleNamespace(forward_metadata=meta)
    FakeTensor.allocations.clear()

    first = helper(host, FakeTensor((8, 16, 128)), 64, backend)
    assert first.shape == (8, 64, 128)
    assert len(FakeTensor.allocations) == 1
    allocation = first.storage

    smaller = helper(host, FakeTensor((4, 16, 128)), 64, backend)
    assert smaller.shape == (4, 64, 128)
    assert smaller.storage is allocation
    assert len(FakeTensor.allocations) == 1

    FakeCuda.capturing = True
    captured = helper(host, FakeTensor((2, 16, 128)), 64, backend)
    FakeCuda.capturing = False
    assert captured.storage is allocation
    assert meta.q_pad_captured is True

    for x, heads in ((FakeTensor((9, 16, 128)), 64), (FakeTensor((4, 16, 128)), 128)):
        try:
            helper(host, x, heads, backend)
        except AssertionError as exc:
            assert "grew after a CUDA graph captured it" in str(exc)
        else:
            raise AssertionError("captured storage replacement was not rejected")
    assert len(FakeTensor.allocations) == 1

    print("PASS SGLang #45301 padded-query buffer behavior")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    run(root)
