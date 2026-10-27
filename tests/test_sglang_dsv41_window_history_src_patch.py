#!/usr/bin/env python3
"""Packaging checks for the SGLang #45579 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45579-window-history-src.patch"


class Dsv41WindowHistorySourcePackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("ad55c50bc3a45bdf957626eb0a9350ca0f4f5176", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45579-window-history-src.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_verify_cached_row_mapping(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_window_history_src.py", source)
                self.assertIn("test_sglang_dsv41_window_history_src.py", source)
                self.assertIn("test_sglang_dsv41_window_history_src_patch.py", source)


if __name__ == "__main__":
    unittest.main()
