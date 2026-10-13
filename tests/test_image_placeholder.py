"""Build-time patch coverage against the unmodified pinned SGLang encoder."""
import copy
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runtime/sglang-canary/python/sglang/srt/entrypoints/openai/encoding_dsv41.py'
PIN_SHA256 = 'e34f8eb2265aed4cb994f489a3d611167bcd1774d598f145595ea4601322afdc'
PATCHER = ROOT / 'runtime/patch_encoding_dsv41.py'


@unittest.skipUnless(SOURCE.is_file(), 'stage pinned SGLang with scripts/fetch-sglang-canary.sh')
class ImagePlaceholderTests(unittest.TestCase):
    def setUp(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), PIN_SHA256)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target = Path(self.tmp.name) / 'encoding.py'
        self.target.write_bytes(SOURCE.read_bytes())
        if not os.environ.get('DSV41_TEST_STOCK_ENCODER'):
            result = self.patch()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        spec = importlib.util.spec_from_file_location('test_encoder', self.target)
        self.encoder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.encoder)

    def patch(self):
        return subprocess.run([sys.executable, str(PATCHER), str(self.target)],
                              capture_output=True, text=True)

    def test_string_and_reasoning_content_sanitize_and_warn(self):
        token = self.encoder.IMAGE_PLACEHOLDER
        original = [{'role': 'user', 'content': 'literal ' + token,
                     'reasoning_content': token}]
        with self.assertLogs('test_encoder', level='WARNING') as logs:
            processed, images = self.encoder.process_image_messages(original)
        self.assertEqual(processed[0]['content'], 'literal [image]')
        self.assertEqual(processed[0]['reasoning_content'], '[image]')
        self.assertEqual(images, [])
        self.assertEqual(original[0]['reasoning_content'], token)
        self.assertEqual(len(logs.records), 2)

    def test_text_and_nested_tool_result_sanitize(self):
        token = self.encoder.IMAGE_PLACEHOLDER
        original = [{'role': 'user', 'content': [
            {'type': 'text', 'text': token},
            {'type': 'tool_result', 'content': [{'type': 'text', 'text': token}]}]}]
        saved = copy.deepcopy(original)
        with self.assertLogs('test_encoder', level='WARNING'):
            processed, images = self.encoder.process_image_messages(original)
        blocks = processed[0]['content_blocks']
        self.assertEqual(blocks[0]['text'], '[image]')
        self.assertEqual(blocks[1]['content'][0]['text'], '[image]')
        self.assertEqual(original, saved)
        self.assertEqual(images, [])

    def test_plain_image_literal_and_real_image_block_are_unchanged(self):
        original = [{'role': 'user', 'content': [
            {'type': 'text', 'text': '[image]'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]}]
        processed, images = self.encoder.process_image_messages(original)
        self.assertEqual(processed[0]['content_blocks'][0]['text'], '[image]')
        self.assertEqual(processed[0]['content_blocks'][1]['text'], self.encoder.IMAGE_PLACEHOLDER)
        self.assertEqual(len(images), 1)

    def test_patch_is_idempotent(self):
        before = self.target.read_bytes()
        result = self.patch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.target.read_bytes(), before)

    def test_drift_in_either_anchor_fails_without_writing(self):
        source = SOURCE.read_text()
        for old in ('Reject user-supplied', 'Text block contains image placeholder'):
            self.target.write_text(source.replace(old, 'drifted anchor'))
            before = self.target.read_bytes()
            self.assertNotEqual(self.patch().returncode, 0)
            self.assertEqual(self.target.read_bytes(), before)

    def test_drift_after_patch_is_refused(self):
        source = self.target.read_text()
        self.target.write_text(source.replace('content.replace(IMAGE_PLACEHOLDER, "[image]")', 'content'))
        self.assertNotEqual(self.patch().returncode, 0)

    def test_every_serving_recipe_runs_patcher(self):
        for name in ('Dockerfile', 'Dockerfile.canary', 'Dockerfile.canary-roce'):
            self.assertIn('RUN python3 /opt/dsv41/runtime/patch_encoding_dsv41.py', (ROOT / name).read_text())

if __name__ == '__main__':
    unittest.main()
