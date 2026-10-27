#!/usr/bin/env python3
"""Fail-closed source verifier for SGLang #44923 (greedy flags refreshed on filter)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"SamplingBatchInfo.{name} is missing")


def _segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def verify(python_root: Path) -> None:
    target = python_root / "sglang/srt/sampling/sampling_batch_info.py"
    source = target.read_text()
    tree = ast.parse(source, filename=str(target))
    cls = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SamplingBatchInfo"
        ),
        None,
    )
    if cls is None:
        raise SystemExit("SamplingBatchInfo is missing")

    fields = [
        node
        for node in cls.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "greedy_rows"
    ]
    if (
        len(fields) != 1
        or not isinstance(fields[0].value, ast.Constant)
        or fields[0].value.value is not None
    ):
        raise SystemExit("greedy_rows field is missing or not defaulted to None")

    build = _segment(source, _method(cls, "from_schedule_batch"))
    for needle in (
        "greedy_rows = [r.sampling_params.top_k <= 1 for r in reqs]",
        "is_all_greedy=all(greedy_rows)",
        "is_any_greedy=any(greedy_rows)",
        "greedy_rows=greedy_rows",
    ):
        if build.count(needle) != 1:
            raise SystemExit(f"from_schedule_batch does not build per-row flags: {needle}")

    filt = _segment(source, _method(cls, "filter_batch"))
    for needle in (
        "if self.greedy_rows is not None:",
        "self.greedy_rows = [self.greedy_rows[i] for i in keep_indices]",
        "self.is_all_greedy = all(self.greedy_rows)",
        "self.is_any_greedy = any(self.greedy_rows)",
    ):
        if filt.count(needle) != 1:
            raise SystemExit(f"filter_batch does not refresh the greedy flags: {needle}")
    if filt.index("self.greedy_rows = [") > filt.index("self.adjusted_filter_batch("):
        raise SystemExit("greedy flags are refreshed after the subclass filter hook")

    merge = _segment(source, _method(cls, "merge_batch"))
    for needle in (
        "self.greedy_rows + other.greedy_rows",
        "if self.greedy_rows is not None and other.greedy_rows is not None",
        "self.is_all_greedy &= other.is_all_greedy",
        "self.is_any_greedy |= other.is_any_greedy",
    ):
        if merge.count(needle) != 1:
            raise SystemExit(f"merge_batch does not combine per-row flags: {needle}")
    # copy_for_forward() shares the list; an in-place extend would leak rows
    # into the forward copy.
    if "greedy_rows.extend" in source or "greedy_rows +=" in source:
        raise SystemExit("greedy_rows is mutated in place")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    verify(Path(sys.argv[1]))
    print("SGLang #44923 greedy-flag refresh on filter_batch: verified")


if __name__ == "__main__":
    main()
