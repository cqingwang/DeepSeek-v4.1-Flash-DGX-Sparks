"""Persistent Triton cache is opt-in and identical in all four rank commands."""
import tempfile
import unittest
from pathlib import Path
from test_launcher_build import shell


class TritonCacheTests(unittest.TestCase):
    def commands(self, enabled, extra=""):
        with tempfile.TemporaryDirectory() as tmp:
            return shell('''
              mkdir -p "$MODEL_DIR"
              declare -a args=()
              docker_common_args args 10.0.0.1 3
              printf '%s\\n' "${args[@]}"
              for rank in 1 2 3; do worker_env_lines 10.0.0.$((rank + 1)) 3 "$rank"; done
            ''', Path(tmp), {'DSV41_TRITON_CACHE': enabled, 'WORKER_IPS': '10.0.0.2 10.0.0.3 10.0.0.4',
                             'NNODES': '4', 'TP_SIZE': '4', 'EXTRA_CONTAINER_ENV': extra})

    def test_off_is_inert(self):
        self.assertNotIn('TRITON_CACHE_DIR=', self.commands('0'))

    def test_on_reaches_every_rank(self):
        self.assertEqual(self.commands('1').count('TRITON_CACHE_DIR=/state/triton'), 4)

    def test_explicit_cache_override_has_same_precedence_on_every_rank(self):
        import shlex
        text = self.commands('1', 'TRITON_CACHE_DIR=/custom')
        head, *workers = text.split('        -e NODE_RANK=')
        effective = []
        for command in [head, *workers]:
            values = [token.split('=', 1)[1] for token in shlex.split(command.replace('\\\n', ' '))
                      if token.startswith('TRITON_CACHE_DIR=')]
            effective.append(values[-1])
        self.assertEqual(effective, ['/custom'] * 4)

if __name__ == '__main__':
    unittest.main()
