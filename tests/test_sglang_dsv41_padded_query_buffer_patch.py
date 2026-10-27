#!/usr/bin/env python3
"""Packaging and fail-closed tests for the SGLang #45301 adaptation."""

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


VERIFY = _load("verify_q_pad", "scripts/verify_sglang_dsv41_padded_query_buffer.py")
BEHAVIOR = _load("test_q_pad", "scripts/test_sglang_dsv41_padded_query_buffer.py")


BACKEND = '''
from dataclasses import dataclass
from typing import Optional

class Tensor: pass

@dataclass
class DSV4Metadata:
    q_pad_buffer: Optional[Tensor] = None
    q_pad_captured: bool = False
'''

MODEL = '''
class MqaAttentionBase:
    def _zero_padded_q(self, x, kernel_num_heads, attn_backend) -> torch.Tensor:
        meta = getattr(attn_backend, "forward_metadata", None)
        buf = getattr(meta, "q_pad_buffer", None)
        if (
            buf is None
            or buf.shape[0] < x.shape[0]
            or buf.shape[1:] != (kernel_num_heads, self.head_dim)
            or buf.dtype != x.dtype
        ):
            assert not getattr(meta, "q_pad_captured", False), (
                "the padded query buffer grew after a CUDA graph captured it"
            )
            buf = x.new_zeros(x.shape[0], kernel_num_heads, self.head_dim)
            if meta is not None:
                try:
                    meta.q_pad_buffer = buf
                except (AttributeError, TypeError):
                    pass
        if meta is not None and torch.cuda.is_current_stream_capturing():
            try:
                meta.q_pad_captured = True
            except (AttributeError, TypeError):
                pass
        return buf[: x.shape[0]]

class MQALayer(MqaAttentionBase):
    def forward(self, x, kernel_num_heads, attn_backend):
        q_padded = self._zero_padded_q(x, kernel_num_heads, attn_backend)
        return q_padded
'''


def _tree(root: Path, *, backend: str = BACKEND, model: str = MODEL) -> Path:
    base = root / "sglang/srt"
    backend_path = base / "layers/attention/deepseek_v4_backend.py"
    model_path = base / "models/deepseek_v4.py"
    backend_path.parent.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    backend_path.write_text(backend)
    model_path.write_text(model)
    return root


class PaddedQueryPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_credited_and_packaged(self):
        patch = ROOT / "patches/sglang-45301-dsv41-padded-query-buffer.patch"
        digest = hashlib.sha256(patch.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            "6cbfa3fef31507cb4bf8d11d38d6a24004375c2b1649fb6a794298a34bf58b19",
        )
        readme = (ROOT / "README.md").read_text()
        self.assertIn("SGLang #45301", readme)
        self.assertIn("hnyls2002", readme)
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-45301-dsv41-padded-query-buffer.patch", fetch)
        self.assertIn(digest, fetch)
        for dockerfile in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            content = (ROOT / dockerfile).read_text()
            self.assertIn("verify_sglang_dsv41_padded_query_buffer.py", content)
            self.assertIn("test_sglang_dsv41_padded_query_buffer.py", content)


class PaddedQueryVerifierTests(unittest.TestCase):
    def test_complete_contract_and_behavior_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _tree(Path(tmp))
            VERIFY.verify(tree)
            BEHAVIOR.run(tree)

    def test_capture_guard_drift_fails(self):
        broken = MODEL.replace(
            'assert not getattr(meta, "q_pad_captured", False)',
            'assert True',
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(_tree(Path(tmp), model=broken))

    def test_graph_state_must_default_false(self):
        broken = BACKEND.replace("q_pad_captured: bool = False", "q_pad_captured: bool = True")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                VERIFY.verify(_tree(Path(tmp), backend=broken))


if __name__ == "__main__":
    unittest.main()
