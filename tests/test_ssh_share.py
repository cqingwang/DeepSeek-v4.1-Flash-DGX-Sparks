"""Regression coverage for public-key recovery, key identity, and share errors."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from test_launcher_build import shell


class SshShareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = self.root / 'SSH_IDENTITY'
        subprocess.run(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(self.key)], check=True,
                       capture_output=True)
        self.pub = Path(str(self.key) + '.pub')
        self.remote_home = self.root / 'remote-home'
        (self.remote_home / '.ssh').mkdir(parents=True)
        self.authorized = self.remote_home / '.ssh/authorized_keys'

    def run_keys(self):
        shell('''
          WORKER_HOSTS=(worker1)
          remote_on() { shift; HOME="$REMOTE_TEST_HOME" bash -c "$*"; }
          ensure_ssh_keys
        ''', self.root, {'REMOTE_TEST_HOME': str(self.remote_home)})

    def test_missing_pub_is_derived(self):
        expected = self.pub.read_text().split()[:2]
        self.pub.unlink()
        self.run_keys()
        self.assertEqual(self.authorized.read_text().split()[:2], expected)

    def test_empty_pub_is_derived(self):
        self.pub.write_text('')
        self.run_keys()
        self.assertTrue(self.authorized.read_text().startswith('ssh-ed25519 '))

    def test_changed_comment_does_not_duplicate_key(self):
        key = ' '.join(self.pub.read_text().split()[:2])
        self.authorized.write_text(key + ' old-comment\n')
        self.pub.write_text(key + " new comment with 'quotes'\n")
        self.run_keys()
        self.assertEqual(self.authorized.read_text(), key + ' old-comment\n')

    def test_new_key_and_quoted_comment_are_appended_once(self):
        key = ' '.join(self.pub.read_text().split()[:2])
        self.authorized.write_text('ssh-ed25519 OTHER other-key\n')
        self.pub.write_text(key + " a comment with 'quotes'\n")
        self.run_keys()
        self.run_keys()
        self.assertEqual(self.authorized.read_text().splitlines(),
                         ['ssh-ed25519 OTHER other-key', key + " a comment with 'quotes'"])

    def test_share_names_missing_serving_image(self):
        model = self.root / 'MODEL_DIR'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        out = shell('''
          ensure_ssh_keys() { :; }
          nfs_ensure_server() { :; }
          nfs_publish_model() { :; }
          nfs_ensure_worker_volume() { :; }
          nfs_worker_has_model() { return 1; }
          remote_ok_on() { return 1; }
          if (cmd_share) 2>&1; then echo PASSED; else echo FAILED; fi
        ''', self.root)
        self.assertIn('Serving image dsv41-3x-spark:local is not on', out)
        self.assertIn('build', out)
        self.assertNotIn('Check ACL', out)

if __name__ == '__main__':
    unittest.main()
