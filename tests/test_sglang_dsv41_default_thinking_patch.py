#!/usr/bin/env python3
"""Packaging checks for the SGLang #45762 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45762-dsv41-default-thinking-parser.patch"


class Dsv41DefaultThinkingPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("f2aad50c4163d047d740fd61defea2d122ac5d30", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45762-dsv41-default-thinking-parser.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_fetch_checks_intermediate_semantics_and_images_check_final_scope(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("verify_sglang_dsv41_default_thinking.py", fetch)
        self.assertIn("test_sglang_dsv41_default_thinking.py", fetch)
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_thinking_parity.py", source)
                self.assertIn("test_sglang_dsv41_thinking_parity.py", source)
                self.assertIn(
                    "test_sglang_dsv41_default_thinking_patch.py", source
                )


if __name__ == "__main__":
    unittest.main()
