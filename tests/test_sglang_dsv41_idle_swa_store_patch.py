#!/usr/bin/env python3
"""Packaging checks for the SGLang #45586/#45699 adaptation."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45586-45699-idle-swa-store.patch"


class Dsv41IdleSwaStorePackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("99cf403c9e84272b80cf36e7cca953d83be4b1d5", fetch)
        self.assertIn("66366028d2dd259372df5066d7c809b4460b8f05", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45586-45699-idle-swa-store.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_verify_idle_routing(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_idle_swa_store.py", source)
                self.assertIn("test_sglang_dsv41_idle_swa_store.py", source)
                self.assertIn("test_sglang_dsv41_idle_swa_store_patch.py", source)


if __name__ == "__main__":
    unittest.main()
