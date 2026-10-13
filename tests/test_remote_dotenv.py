"""Local dotenv parsing; no remote commands are invoked."""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('remote_dotenv', ROOT / 'scripts/remote.py')
remote = importlib.util.module_from_spec(spec)
# Dotenv parsing does not use the optional SSH dependency.
with patch.dict(sys.modules, {'pexpect': ModuleType('pexpect')}):
    spec.loader.exec_module(remote)


class RemoteDotenvTests(unittest.TestCase):
    def test_single_quoted_values_are_literal(self):
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / '.env'
            dotenv.write_text("WORKER_PASS='literal$HOME/${HOME}'\n")
            with patch.dict(os.environ, {'HOME': '/test/home'}, clear=True):
                remote.load_dotenv(str(dotenv))
                self.assertEqual(os.environ['WORKER_PASS'], 'literal$HOME/${HOME}')

    def test_double_quoted_and_unquoted_values_expand(self):
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / '.env'
            dotenv.write_text('DOUBLE="$HOME/${HOME}"\nUNQUOTED=$HOME/${HOME}\n')
            with patch.dict(os.environ, {'HOME': '/test/home'}, clear=True):
                remote.load_dotenv(str(dotenv))
                self.assertEqual(os.environ['DOUBLE'], '/test/home//test/home')
                self.assertEqual(os.environ['UNQUOTED'], '/test/home//test/home')

    def test_existing_environment_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            dotenv = Path(tmp) / '.env'
            dotenv.write_text("WORKER_PASS='new$HOME'\n")
            with patch.dict(os.environ, {'WORKER_PASS': 'existing'}, clear=True):
                remote.load_dotenv(str(dotenv))
                self.assertEqual(os.environ['WORKER_PASS'], 'existing')


if __name__ == '__main__':
    unittest.main()
