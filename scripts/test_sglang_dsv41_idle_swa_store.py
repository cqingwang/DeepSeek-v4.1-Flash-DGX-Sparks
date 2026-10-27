#!/usr/bin/env python3
"""Exercise the staged idle SWA-store methods without importing SGLang."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def _extract(path: Path, name: str, namespace: dict):
    tree = ast.parse(path.read_text(), filename=str(path))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
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


class Tensor:
    def __init__(self, rows, label="tensor"):
        self.shape = (rows,)
        self.label = label
        self.dtype = None

    def to(self, dtype):
        self.dtype = dtype
        return self

    def numel(self):
        return self.shape[0]


class DSV4Metadata:
    pass


class Serving:
    def __init__(self, pool, metadata=None, forbid_metadata=False):
        self.token_to_kv_pool = pool
        self._metadata = metadata
        self.forbid_metadata = forbid_metadata

    @property
    def forward_metadata(self):
        if self.forbid_metadata:
            raise AssertionError("idle path read stale metadata")
        return self._metadata


class Window:
    def __init__(self):
        self.idle_calls = []
        self.activated = []

    def activate_idle(self, rows):
        self.idle_calls.append(rows)
        return Tensor(rows, "idle-window")

    def activate(self, layout):
        self.activated.append(layout)


class Pool:
    def __init__(self, window):
        self.request_window = window
        self.translated = []

    def translate_loc_from_full_to_swa(self, value):
        self.translated.append(value)
        return Tensor(value.shape[0], "translated")


def batch(rows, idle):
    return SimpleNamespace(
        out_cache_loc=Tensor(rows, "out"),
        forward_mode=SimpleNamespace(is_idle=lambda: idle),
    )


def test_store_router(method):
    window = Window()
    pool = Pool(window)
    result = method(Serving(pool, forbid_metadata=True), batch(7, True))
    assert result.label == "idle-window" and window.idle_calls == [7]

    pool = Pool(None)
    result = method(Serving(pool, forbid_metadata=True), batch(5, True))
    assert result.label == "translated" and result.dtype == "int32"

    layout = SimpleNamespace(write_loc=Tensor(3, "layout-write"))
    metadata = SimpleNamespace(
        core_attn_metadata=SimpleNamespace(
            request_window_layout=layout, swa_out_cache_loc=None
        )
    )
    window = Window()
    result = method(Serving(Pool(window), metadata), batch(3, False))
    assert result is layout.write_loc and window.activated == [layout]

    tail = Tensor(2, "tail")
    metadata = DSV4Metadata()
    metadata.late_layer_tail = SimpleNamespace(swa_out_cache_loc=tail)
    assert method(Serving(Pool(None), metadata), batch(9, False)) is tail

    cached = Tensor(4, "cached")
    metadata = SimpleNamespace(
        core_attn_metadata=SimpleNamespace(swa_out_cache_loc=cached)
    )
    assert method(Serving(Pool(None), metadata), batch(4, False)) is cached
    result = method(Serving(Pool(None), metadata), batch(6, False))
    assert result.label == "translated"


def test_idle_layout(method):
    calls = []

    def zeros(rows, **kwargs):
        calls.append(("zeros", rows, kwargs))
        return Tensor(rows, "zeros")

    def window_layout(req, pos, **kwargs):
        calls.append(("layout", req.shape[0], kwargs))
        return SimpleNamespace(pos=pos, write_loc=Tensor(req.shape[0], "write"))

    owner = SimpleNamespace(
        _idle_layout=None,
        tags=SimpleNamespace(device="cuda"),
        zero_row=4096,
        capacity=256,
        activated=[],
    )
    owner.activate = owner.activated.append
    method.__globals__["torch"] = SimpleNamespace(zeros=zeros, int64="int64")
    method.__globals__["window_layout"] = window_layout

    first = method(owner, 3)
    first_layout = owner._idle_layout
    second = method(owner, 3)
    third = method(owner, 5)
    third_layout = owner._idle_layout
    assert first.shape == (3,) and second is first and third.shape == (5,)
    layouts = [item for item in calls if item[0] == "layout"]
    assert len(layouts) == 2
    assert layouts[0][2] == {"zero_row": 4096, "capacity": 256, "num_groups": 1}
    assert owner.activated == [first_layout, first_layout, third_layout]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    base = Path(sys.argv[1]) / "sglang/srt"
    store = _extract(
        base / "layers/attention/deepseek_v4_backend.py",
        "get_swa_out_cache_loc",
        {"DSV4Metadata": DSV4Metadata, "torch": SimpleNamespace(int32="int32")},
    )
    idle = _extract(
        base / "mem_cache/dsv41_request_window.py",
        "activate_idle",
        {},
    )
    test_store_router(store)
    test_idle_layout(idle)
    print("SGLang #45586/#45699 idle SWA-store behavior: PASS")


if __name__ == "__main__":
    main()
