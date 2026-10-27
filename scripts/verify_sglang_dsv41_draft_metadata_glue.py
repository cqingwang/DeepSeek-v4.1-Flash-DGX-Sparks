#!/usr/bin/env python3
"""Fail-closed checks for the SGLang #45450 draft metadata-glue port."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


def method_source(path: Path, name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise SystemExit(f"expected one {name} in {path}, found {len(matches)}")
    return ast.get_source_segment(source, matches[0]) or ""


def require(source: str, snippets: tuple[str, ...], label: str) -> None:
    missing = [snippet for snippet in snippets if snippet not in source]
    if missing:
        raise SystemExit(f"incomplete {label}: {missing}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} SGLANG_PYTHON_ROOT")
    root = Path(sys.argv[1]) / "sglang/srt"
    base = root / "layers/attention/base_attn_backend.py"
    dsv4 = root / "layers/attention/deepseek_v4_backend.py"
    runner = root / "model_executor/runner/decode_cuda_graph_runner.py"
    glue = root / "model_executor/runner/metadata_glue_graph.py"

    for path in (base, dsv4, runner, glue):
        ast.parse(path.read_text(), filename=str(path))

    require(
        method_source(base, "glue_draft_metadata"),
        ('return "the attention backend declares no host-free draft prep"',),
        "default deny",
    )
    require(
        method_source(dsv4, "glue_draft_metadata"),
        (
            "not self.is_dspark_draft",
            "self.trtllm_attn",
            "self.online_c128_mtp.enabled()",
            "self.token_to_kv_pool.request_window is not None",
            "get_parallel().attn_cp_size != 1",
            "self.draft_metadata_glued = True",
        ),
        "DSV4 host-free guard",
    )
    require(
        method_source(dsv4, "_draft_block_extend_lens"),
        (
            "if not self.draft_metadata_glued",
            "key = tuple(lens)",
            "self._draft_extend_lens.get(key)",
            "torch.cuda.is_current_stream_capturing()",
            "self._draft_extend_lens[key] = self._move_to_device(lens)",
        ),
        "persistent extend-length cache",
    )
    draft_block = method_source(dsv4, "init_forward_metadata_dspark_draft_block")
    if "self._draft_block_extend_lens(lengths.extend_seq_lens_cpu)" not in draft_block:
        raise SystemExit("draft block still allocates a capture-unsafe pinned tensor")

    require(
        method_source(runner, "_glues_draft_metadata"),
        (
            "if not self.model_runner.is_draft_worker",
            'envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static"',
            "self.attn_backend.glue_draft_metadata()",
        ),
        "runner routing guard",
    )
    require(
        method_source(runner, "_glue_draft_inputs"),
        (
            "if self._draft_metadata_glue and fb_view.out_cache_loc is not None",
            "fb_view.out_cache_loc = self.buffers.out_cache_loc[:n]",
        ),
        "pointer-stable replay input",
    )
    runner_source = runner.read_text()
    if runner_source.index("self._glue_draft_inputs(fb_view)") > runner_source.index(
        "self._metadata_glue.run("
    ):
        raise SystemExit("draft input rebinding must happen before glue replay")

    require(
        glue.read_text(),
        (
            "self.replays = 0",
            "self.eager = 0",
            'capture_error_mode="thread_local"',
            "self.replays += 1",
            "self.eager += 1",
        ),
        "glue capture and counters",
    )
    print("SGLang #45450 draft metadata glue: verified")


if __name__ == "__main__":
    main()
