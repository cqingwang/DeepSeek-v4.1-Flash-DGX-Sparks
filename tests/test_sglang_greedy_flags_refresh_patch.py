#!/usr/bin/env python3
"""Packaging checks for the SGLang #44923 greedy-flag refresh backport."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-44923-greedy-flags-refresh.patch"


class GreedyFlagsRefreshPackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("6e0355d972cb926f137d2b8bb70caf5c9af5d129", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-44923-greedy-flags-refresh.patch", fetch)
        self.assertIn("verify_sglang_greedy_flags_refresh.py", fetch)
        self.assertIn("test_sglang_greedy_flags_refresh.py", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_patch_touches_only_sampling_batch_info(self):
        headers = [
            line
            for line in PATCH.read_text().splitlines()
            if line.startswith(("--- ", "+++ "))
        ]
        self.assertEqual(
            headers,
            [
                "--- a/python/sglang/srt/sampling/sampling_batch_info.py",
                "+++ b/python/sglang/srt/sampling/sampling_batch_info.py",
            ],
        )

    def test_images_repeat_verifier_semantics_and_packaging(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_greedy_flags_refresh.py", source)
                self.assertIn("test_sglang_greedy_flags_refresh.py", source)
                self.assertIn("test_sglang_greedy_flags_refresh_patch.py", source)


if __name__ == "__main__":
    unittest.main()
