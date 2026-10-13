"""Messages thinking compatibility against the real pinned protocol on CPU."""
import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
import encoding_compat

SOURCE = ROOT / 'runtime/sglang-canary/python/sglang/srt/entrypoints/anthropic/protocol.py'


def load_protocol():
    if SOURCE.is_file():
        spec = importlib.util.spec_from_file_location('staged_anthropic_protocol', SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    from sglang.srt.entrypoints.anthropic import protocol
    return protocol


@unittest.skipUnless(SOURCE.is_file(), 'staged protocol source is not installed')
class AnthropicValidatorPinTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'AnthropicThinkingParam')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == '_validate_thinking_shape')
        namespace = {'model_validator': lambda **kwargs: lambda func: func}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(SOURCE), 'exec'), namespace)
        self.entry = SimpleNamespace(func=namespace['_validate_thinking_shape'])
        self.model = SimpleNamespace(
            __pydantic_decorators__=SimpleNamespace(
                model_validators={'_validate_thinking_shape': self.entry}),
            model_rebuild=Mock())
        self.protocol = SimpleNamespace(
            AnthropicThinkingParam=self.model,
            AnthropicMessagesRequest=SimpleNamespace(model_rebuild=Mock()),
            AnthropicCountTokensRequest=SimpleNamespace(model_rebuild=Mock()))

    def assert_unchanged(self, original):
        self.assertIs(self.entry.func, original)
        for model in vars(self.protocol).values():
            model.model_rebuild.assert_not_called()

    def test_pinned_validator_is_accepted(self):
        self.assertTrue(encoding_compat.install_anthropic_protocol(self.protocol))
        param = SimpleNamespace(type='enabled', budget_tokens=None)
        self.assertIs(self.entry.func(param), param)
        with self.assertRaises(ValueError):
            self.entry.func(SimpleNamespace(type='enabled', budget_tokens=512))
        for model in vars(self.protocol).values():
            model.model_rebuild.assert_called_once_with(force=True)

    def test_same_named_validator_drift_changes_nothing(self):
        def _validate_thinking_shape(self):
            raise ValueError('additional upstream validation')

        self.entry.func = _validate_thinking_shape
        with self.assertLogs(encoding_compat.logger, level='WARNING') as logs:
            self.assertFalse(encoding_compat.install_anthropic_protocol(self.protocol))
        self.assertIn('source drift', '\n'.join(logs.output))
        self.assert_unchanged(_validate_thinking_shape)

    def test_unavailable_validator_source_changes_nothing(self):
        original = self.entry.func
        with patch('inspect.getsource', side_effect=OSError('source unavailable')):
            with self.assertLogs(encoding_compat.logger, level='WARNING'):
                self.assertFalse(encoding_compat.install_anthropic_protocol(self.protocol))
        self.assert_unchanged(original)


class AnthropicThinkingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.protocol = load_protocol()
        except ImportError as exc:
            self.skipTest(f'stage SGLang and install Pydantic v2: {exc}')
        if not os.environ.get('DSV41_TEST_STOCK_PROTOCOL'):
            encoding_compat.install_anthropic_protocol(self.protocol)

    def test_enabled_without_budget(self):
        param = self.protocol.AnthropicThinkingParam(type='enabled')
        self.assertIsNone(param.budget_tokens)

    def test_messages_and_count_requests_without_budget(self):
        body = dict(model='m', messages=[{'role': 'user', 'content': 'hi'}],
                    thinking={'type': 'enabled'})
        self.protocol.AnthropicMessagesRequest(max_tokens=16, **body)
        self.protocol.AnthropicCountTokensRequest(**body)

    def test_explicit_valid_budget_accepted(self):
        self.protocol.AnthropicThinkingParam(type='enabled', budget_tokens=2048)

    def test_existing_validation_rules_preserved(self):
        for fields in (dict(type='enabled', budget_tokens=512),
                       dict(type='disabled', budget_tokens=2048),
                       dict(type='disabled', display='omitted'),
                       dict(type='adaptive', budget_tokens=2048),
                       dict(type='unknown')):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.protocol.AnthropicThinkingParam(**fields)
        self.protocol.AnthropicThinkingParam(type='adaptive')
        self.protocol.AnthropicThinkingParam(type='disabled')

    def test_install_is_idempotent_and_missing_symbol_warns(self):
        self.assertFalse(encoding_compat.install_anthropic_protocol(self.protocol))
        from types import SimpleNamespace
        with self.assertLogs(encoding_compat.logger, level='WARNING'):
            self.assertFalse(encoding_compat.install_anthropic_protocol(SimpleNamespace()))

    def test_sitecustomize_registers_protocol(self):
        source = (ROOT / 'adapter/sitecustomize.py').read_text()
        self.assertEqual(source.count("'sglang.srt.entrypoints.anthropic.protocol'"), 2)
        self.assertIn('install_anthropic_protocol(module)', source)

if __name__ == '__main__':
    unittest.main()
