#!/usr/bin/env python3
"""Packaging checks for the SGLang #45804 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45804-dsv41-thinking-parity.patch"


class Dsv41ThinkingParityPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("af882cfe1e7ff244db2a49bb70fdc72199c68bb9", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45804-dsv41-thinking-parity.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_check_final_chat_and_responses_semantics(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_thinking_parity.py", source)
                self.assertIn("test_sglang_dsv41_thinking_parity.py", source)
                self.assertIn("test_sglang_dsv41_thinking_parity_patch.py", source)


if __name__ == "__main__":
    unittest.main()
