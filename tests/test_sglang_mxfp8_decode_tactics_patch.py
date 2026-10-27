#!/usr/bin/env python3
"""Packaging checks for the SGLang #45630 MXFP8 tactic backport."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45630-mxfp8-decode-tactics.patch"


class Mxfp8DecodeTacticsPackagingTests(unittest.TestCase):
    def test_patch_is_narrow_and_exact(self):
        patch = PATCH.read_text()
        self.assertEqual(patch.count("diff --git "), 1)
        self.assertIn("mxfp8_prefill_autotune_min_tokens = 1", patch)
        self.assertIn("mxfp8_prefill_autotune_min_tokens = 4096", patch)
        self.assertNotIn("runtime/sglang-canary", patch)

    def test_fetch_pins_applies_and_verifies_patch(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("741aea0ab6c51ccfbb862203a5a4e4aeaf7e8466", fetch)
        self.assertIn("sglang-45630-mxfp8-decode-tactics.patch", fetch)
        self.assertIn("verify_sglang_mxfp8_decode_tactics.py", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_repeat_source_and_packaging_checks(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_mxfp8_decode_tactics.py", source)
                self.assertIn("test_sglang_mxfp8_decode_tactics_patch.py", source)


if __name__ == "__main__":
    unittest.main()
