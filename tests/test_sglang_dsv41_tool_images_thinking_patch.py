#!/usr/bin/env python3
"""Packaging and source-contract checks for the SGLang #44751 port."""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_PATH = ROOT / "scripts/verify_sglang_dsv41_tool_images_thinking.py"
SPEC = importlib.util.spec_from_file_location("verify_dsv41_tool_images_thinking", VERIFY_PATH)
VERIFY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VERIFY)


class ToolImagesThinkingPackagingTests(unittest.TestCase):
    def test_patch_is_credited_and_content_pinned(self):
        patch_path = ROOT / "patches/sglang-44751-dsv41-tool-images-thinking.patch"
        digest = hashlib.sha256(patch_path.read_bytes()).hexdigest()
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        readme = (ROOT / "README.md").read_text()
        self.assertEqual(digest, "0851a8f33fd00da18630a5fd4a275b8248e06ca57bb1dd205d7acc1179c36fa2")
        self.assertIn("TOOL_IMAGES_THINKING_PATCH_SHA256", fetch)
        self.assertIn("a038a0dca05cefca1e0edc4ac43e3bfbec2e064c", fetch)
        self.assertIn("SGLang #44751", readme)
        self.assertIn("Justin Tong", readme)

    def test_fetch_and_both_canary_images_verify_behavior(self):
        fetch = (ROOT / "scripts/fetch-sglang-canary.sh").read_text()
        self.assertIn("sglang-44751-dsv41-tool-images-thinking.patch", fetch)
        self.assertIn("verify_sglang_dsv41_tool_images_thinking.py", fetch)
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            with self.subTest(name=name):
                source = (ROOT / name).read_text()
                self.assertIn("verify_sglang_dsv41_tool_images_thinking.py", source)
                self.assertIn("test_sglang_dsv41_tool_images_thinking.py", source)


class ToolImagesThinkingVerifierTests(unittest.TestCase):
    SOURCE = '''
import copy

def find_last_user_index(messages):
    """User messages containing only tool results do not start a new turn."""
    for idx in range(len(messages) - 1, -1, -1):
        content_blocks = messages[idx].get("content_blocks")
        if content_blocks and not any(
            block.get("type") in ("text", "image", "image_url")
            for block in content_blocks
        ):
            continue
        return idx
    return -1

def merge_tool_messages(messages):
    msg = {"content_blocks": []}
    content = msg["content_blocks"]
    return messages

def sort_tool_results_by_call_order(messages):
    return messages

def _validate_no_image_sp_tokens(msg):
    return None

def process_image_messages(messages):
    return messages, []

def _encode_messages_text(messages, **kwargs):
    return ""

def encode_messages(messages, context=None):
    context = copy.deepcopy(context or [])
    messages = copy.deepcopy(messages)
    # Validate before merging can discard message-level textual fields.
    for msg in context + messages:
        _validate_no_image_sp_tokens(msg)
    # Collect images only after tool results are in their final prompt order.
    context = merge_tool_messages(context)
    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    process_image_messages(context)
    process_image_messages(messages)
    return _encode_messages_text(messages)
'''

    def make_tree(self, root: Path, source: str | None = None) -> Path:
        path = root / "sglang/srt/entrypoints/openai/encoding_dsv41.py"
        path.parent.mkdir(parents=True)
        path.write_text(self.SOURCE if source is None else source)
        return root

    def test_complete_tree_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            VERIFY.verify(self.make_tree(Path(tmp)))

    def test_private_copy_drift_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = self.SOURCE.replace(
                "messages = copy.deepcopy(messages)", "messages = list(messages)"
            )
            with self.assertRaises(SystemExit):
                VERIFY.verify(self.make_tree(Path(tmp), broken))

    def test_late_preprocessing_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            broken = self.SOURCE.replace(
                'def _encode_messages_text(messages, **kwargs):\n    return ""',
                'def _encode_messages_text(messages, **kwargs):\n'
                '    return merge_tool_messages(messages)',
            )
            with self.assertRaises(SystemExit):
                VERIFY.verify(self.make_tree(Path(tmp), broken))


if __name__ == "__main__":
    unittest.main()
