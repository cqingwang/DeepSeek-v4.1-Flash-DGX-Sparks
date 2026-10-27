#!/usr/bin/env python3
"""The serving alias list must be reduced to one model name for API requests."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'adapter'))
spec = importlib.util.spec_from_file_location('boot', ROOT / 'boot.py')
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


class PrimaryModelAliasTests(unittest.TestCase):
    def test_single_alias_is_preserved(self):
        self.assertEqual(boot.primary_model_name('deepseek-v4.1-flash'),
                         'deepseek-v4.1-flash')

    def test_first_nonempty_alias_is_selected(self):
        self.assertEqual(
            boot.primary_model_name(' , deepseek-v4.1-flash,deepseek-v4-flash-vision-exp'),
            'deepseek-v4.1-flash')

    def test_empty_alias_list_uses_the_compatibility_name(self):
        self.assertEqual(boot.primary_model_name(' , '), 'deepseek-v4.1-flash')


if __name__ == '__main__':
    unittest.main()
