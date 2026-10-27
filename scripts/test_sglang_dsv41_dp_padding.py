#!/usr/bin/env python3
"""CPU-only falsifiers extracted from the production SGLang #40424 methods."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace


def _method(path: Path, class_name: str, method_name: str):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for member in node.body:
                if isinstance(member, ast.FunctionDef) and member.name == method_name:
                    return member
    raise AssertionError(f"missing {class_name}.{method_name}")


def _compile_low_ratio(path: Path):
    method = _method(path, "DeepseekV4AttnBackend", "forward_low_ratio_sources")
    start = next(
        i
        for i, stmt in enumerate(method.body)
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "meta" for t in stmt.targets)
    )
    body = method.body[start : start + 5]
    body.append(
        ast.Return(
            ast.Tuple(
                [ast.Name(name, ast.Load()) for name in ("req", "pos", "x", "q_lora", "positions")],
                ast.Load(),
            )
        )
    )
    probe = ast.FunctionDef(
        name="probe",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in ("self", "x", "q_lora", "positions", "forward_batch")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )
    module = ast.Module(body=[probe], type_ignores=[])
    ast.fix_missing_locations(module)

    def fallback(_batch, *, num_tokens):
        return Rows(num_tokens, "fallback")

    namespace = {"token_req_indices": fallback}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["probe"]


def _compile_engram_prefix(path: Path):
    method = _method(path, "EngramHasher", "forward")
    idle_index = next(
        i
        for i, stmt in enumerate(method.body)
        if isinstance(stmt, ast.If) and "mode.is_idle()" in ast.unparse(stmt.test)
    )
    body = method.body[: idle_index + 1] + [ast.Return(ast.Constant(None))]
    probe = ast.FunctionDef(
        name="probe",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=n) for n in ("self", "input_ids", "forward_batch")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )
    module = ast.Module(body=[probe], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": FakeTorch}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["probe"]


class Rows:
    def __init__(self, rows: int, label: str):
        self.shape = (rows,)
        self.label = label

    def __getitem__(self, item):
        assert isinstance(item, slice) and item.start is None and item.step is None
        return Rows(min(self.shape[0], item.stop), self.label)


class FakeTorch:
    int64 = "int64"

    @staticmethod
    def zeros(shape, *, dtype, device):
        return SimpleNamespace(shape=tuple(shape), dtype=dtype, device=device, zero=True)


class Mode:
    def __init__(self, idle: bool):
        self.idle = idle

    def is_idle(self):
        return self.idle


def run(python_root: Path) -> None:
    backend_path = python_root / "sglang/srt/layers/attention/deepseek_v4_backend.py"
    engram_path = python_root / "sglang/srt/layers/engram.py"
    low_ratio = _compile_low_ratio(backend_path)

    meta = SimpleNamespace(low_ratio_req_indices=Rows(3, "req"), low_ratio_pos_i64=Rows(3, "pos"))
    host = SimpleNamespace(forward_metadata=meta)
    req, pos, x, q_lora, positions = low_ratio(
        host, Rows(5, "x"), Rows(5, "q"), Rows(5, "positions"), object()
    )
    assert (req.label, pos.label) == ("req", "pos")
    assert (x.shape[0], q_lora.shape[0], positions.shape[0]) == (3, 3, 3)

    req, pos, x, q_lora, positions = low_ratio(
        host, Rows(5, "x"), None, Rows(5, "positions"), object()
    )
    assert q_lora is None and (req.shape[0], pos.shape[0], x.shape[0], positions.shape[0]) == (3, 3, 3, 3)

    meta.low_ratio_req_indices = meta.low_ratio_pos_i64 = None
    req, pos, x, q_lora, positions = low_ratio(
        host, Rows(4, "x"), Rows(4, "q"), Rows(4, "positions"), object()
    )
    assert req.label == "fallback" and pos.label == "positions"
    assert (x.shape[0], q_lora.shape[0], positions.shape[0]) == (4, 4, 4)

    engram = _compile_engram_prefix(engram_path)
    engram_host = SimpleNamespace(
        history=object(), max_ngram_size=4, primes=Rows(2, "p"), offsets=SimpleNamespace(shape=(2, 3))
    )
    for rows, idle in ((5, True), (0, False)):
        result = engram(
            engram_host,
            SimpleNamespace(shape=(rows,), device="cpu"),
            SimpleNamespace(forward_mode=Mode(idle)),
        )
        assert result.shape == (rows, 2, 3) and result.zero
    assert engram(
        engram_host,
        SimpleNamespace(shape=(5,), device="cpu"),
        SimpleNamespace(forward_mode=Mode(False)),
    ) is None

    print("PASS SGLang #40424 DP-padding/idle-row behavior")


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "/sgl-workspace/sglang/python")
    run(root)
