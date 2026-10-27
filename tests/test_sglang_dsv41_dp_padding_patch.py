#!/usr/bin/env python3
"""Packaging and fail-closed checks for the SGLang #40424 adaptation."""

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


VERIFY = _load("verify_dp_pad", "scripts/verify_sglang_dsv41_dp_padding.py")
BEHAVIOR = _load("test_dp_pad", "scripts/test_sglang_dsv41_dp_padding.py")


BACKEND = '''
class DeepseekV4AttnBackend:
    def forward_low_ratio_sources(self, x, q_lora, positions, forward_batch):
        meta = self.forward_metadata
        hoisted_req = getattr(meta, "low_ratio_req_indices", None)
        hoisted_pos = getattr(meta, "low_ratio_pos_i64", None)
        if hoisted_pos is not None and hoisted_pos.shape[0] < positions.shape[0]:
            num_real = hoisted_pos.shape[0]
            x, positions = x[:num_real], positions[:num_real]
            if q_lora is not None:
                q_lora = q_lora[:num_real]
        if (
            hoisted_req is not None
            and hoisted_pos is not None
            and hoisted_pos.shape[0] == positions.shape[0]
        ):
            req, pos = hoisted_req, hoisted_pos
        else:
            req = token_req_indices(forward_batch, num_tokens=positions.shape[0])
            pos = positions
        return req, pos, x, q_lora, positions
'''

ENGRAM = '''
class EngramHasher:
    def forward(self, input_ids, forward_batch):
        assert self.history is not None, "EngramHasher.init_history was not called"
        n = self.max_ngram_size
        num_tokens = input_ids.shape[0]
        mode = forward_batch.forward_mode
        if num_tokens == 0 or mode.is_idle():
            return torch.zeros(
                (num_tokens, self.primes.shape[0], self.offsets.shape[1]),
                dtype=torch.int64,
                device=input_ids.device,
            )
        req_slots = forward_batch.req_pool_indices
        assert mode.is_extend()
        return req_slots
'''


def _tree(root: Path, *, backend: str = BACKEND, engram: str = ENGRAM) -> Path:
    base = root / "sglang/srt/layers"
    backend_path = base / "attention/deepseek_v4_backend.py"
    engram_path = base / "engram.py"
    backend_path.parent.mkdir(parents=True)
    engram_path.parent.mkdir(parents=True, exist_ok=True)
    backend_path.write_text(backend)
    engram_path.write_text(engram)
    return root


class DPPaddingPackagingTests(unittest.TestCase):
    def test_patch_is_pinned_credited_and_packaged(self):
        patch = ROOT / "patches/sglang-40424-dsv41-dp-padding.patch"
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "27cfd0e1511733fc853cd0f55679cab9b536c4422f40f2dac902a73b4d7c46e3",
        )
        readme = (ROOT / "README.md").read_text()
        self.assertIn("SGLang #40424", readme)
        self.assertIn("JiaShuai0209", readme)
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-40424-dsv41-dp-padding.patch", fetch)
        self.assertIn(digest, fetch)
        for dockerfile in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            content = (ROOT / dockerfile).read_text()
            self.assertIn("verify_sglang_dsv41_dp_padding.py", content)
            self.assertIn("test_sglang_dsv41_dp_padding.py", content)


class DPPaddingVerifierTests(unittest.TestCase):
    def test_complete_contract_and_behavior_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree(Path(tmp))
            VERIFY.verify(tree)
            BEHAVIOR.run(tree)

    def test_missing_live_row_slice_fails(self):
        broken = BACKEND.replace(
            "x, positions = x[:num_real], positions[:num_real]",
            "x = x[:num_real]",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(_tree(Path(tmp), backend=broken))

    def test_idle_zero_shape_drift_fails(self):
        broken = ENGRAM.replace(
            "(num_tokens, self.primes.shape[0], self.offsets.shape[1])",
            "(0, self.primes.shape[0], self.offsets.shape[1])",
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(_tree(Path(tmp), engram=broken))


if __name__ == "__main__":
    unittest.main()
