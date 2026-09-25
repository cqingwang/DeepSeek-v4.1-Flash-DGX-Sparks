#!/usr/bin/env python3
"""AST-only check: do not import boot.py (it has runtime side effects)."""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CacheReportingTests(unittest.TestCase):
    def test_unconditional_sglang_flag(self):
        tree = ast.parse((ROOT / "boot.py").read_text())
        base_args = [
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets)
            and isinstance(node.value, ast.List)
        ]
        self.assertEqual(len(base_args), 1)
        flags = [n.value for n in base_args[0].elts if isinstance(n, ast.Constant)]
        self.assertEqual(flags.count("--enable-cache-report"), 1)
        self.assertNotIn("--enable-prompt-tokens-details", flags)
        self.assertNotIn("--enable-prompt-token-details", flags)


if __name__ == "__main__":
    unittest.main()
