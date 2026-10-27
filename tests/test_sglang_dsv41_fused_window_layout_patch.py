#!/usr/bin/env python3
"""Packaging checks for the SGLang #45726 pinned-tree adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45726-fused-window-layout-integration.patch"


class Dsv41FusedWindowLayoutPackagingTests(unittest.TestCase):
    def test_upstream_kernel_and_integration_are_content_pinned(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("9d5386fd13f7a91e9e84dc0c0ab61c5b1fbaba98", fetch)
        self.assertIn(
            "4ca6d7d7d79340bb47b9f960d1cf4828674ede86c46f1858f04fe2b6165eb171",
            fetch,
        )
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45726-fused-window-layout-integration.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_verify_fused_layout(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_fused_window_layout.py", source)
                self.assertIn("test_sglang_dsv41_fused_window_layout.py", source)
                self.assertIn(
                    "test_sglang_dsv41_fused_window_layout_patch.py", source
                )


if __name__ == "__main__":
    unittest.main()
