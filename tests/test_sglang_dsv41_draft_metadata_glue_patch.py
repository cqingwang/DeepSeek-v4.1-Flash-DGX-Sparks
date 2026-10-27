#!/usr/bin/env python3
"""Packaging checks for the SGLang #45450 draft metadata-glue port."""

from __future__ import annotations

import hashlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches/sglang-45450-draft-metadata-glue.patch"


class Dsv41DraftMetadataGluePackagingTests(unittest.TestCase):
    def test_patch_is_content_pinned_and_zero_fuzz(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
        self.assertIn("1f8e4f3e6cdc378d94a8a08630a9763c648e88eb", fetch)
        self.assertIn(digest, fetch)
        self.assertIn("sglang-45450-draft-metadata-glue.patch", fetch)
        self.assertIn("--fuzz=0", fetch)

    def test_images_verify_draft_glue(self):
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn(
                    "verify_sglang_dsv41_draft_metadata_glue.py", source
                )
                self.assertIn("test_sglang_dsv41_draft_metadata_glue.py", source)
                self.assertIn(
                    "test_sglang_dsv41_draft_metadata_glue_patch.py", source
                )


if __name__ == "__main__":
    unittest.main()
