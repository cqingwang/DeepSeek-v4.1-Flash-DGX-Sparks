#!/usr/bin/env python3
"""Packaging and fail-closed checks for the SGLang #43448 adaptation."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VERIFY = _load("verify_mhc_no_chunk", "scripts/verify_sglang_mhc_prewarm_no_chunk.py")
BEHAVIOR = _load("test_mhc_no_chunk", "scripts/test_sglang_mhc_prewarm_no_chunk.py")


SOURCE = '''
def prewarm_mhc_pre(residual):
    from sglang.srt.runtime_context import get_schedule
    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    schedule = get_schedule()
    max_num_tokens = schedule.chunked_prefill_size
    if max_num_tokens is None or max_num_tokens <= 0:
        max_num_tokens = schedule.max_prefill_tokens
    buckets = get_mhc_pre_token_count_representatives(
        max_num_tokens, hc_mult * hidden_size
    )
    return buckets
'''


def _tree(root: Path, source: str = SOURCE) -> Path:
    path = root / "sglang/kernels/ops/layernorm/mhc.py"
    path.parent.mkdir(parents=True)
    path.write_text(source)
    return root


class MHCPrewarmPackagingTests(unittest.TestCase):
    def test_patch_is_pinned_credited_and_packaged(self):
        patch = ROOT / "patches/sglang-43448-mhc-prewarm-no-chunk.patch"
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "8c559f5c7d5a0ecb520d592c5445680119d8565607223c0ba56d5e69852505ec",
        )
        readme = (ROOT / "README.md").read_text()
        self.assertIn("SGLang #43448", readme)
        self.assertIn("maocheng23", readme)
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-43448-mhc-prewarm-no-chunk.patch", fetch)
        self.assertIn(digest, fetch)
        for dockerfile in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            content = (ROOT / dockerfile).read_text()
            self.assertIn("verify_sglang_mhc_prewarm_no_chunk.py", content)
            self.assertIn("test_sglang_mhc_prewarm_no_chunk.py", content)


class MHCPrewarmVerifierTests(unittest.TestCase):
    def test_complete_contract_and_behavior_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree(Path(tmp))
            VERIFY.verify(tree)
            BEHAVIOR.run(tree)

    def test_zero_chunk_guard_drift_fails(self):
        broken = SOURCE.replace(
            "max_num_tokens is None or max_num_tokens <= 0",
            "max_num_tokens is None",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(_tree(Path(tmp), broken))

    def test_budget_fallback_drift_fails(self):
        broken = SOURCE.replace(
            "max_num_tokens = schedule.max_prefill_tokens",
            "max_num_tokens = 1",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(_tree(Path(tmp), broken))


if __name__ == "__main__":
    unittest.main()
