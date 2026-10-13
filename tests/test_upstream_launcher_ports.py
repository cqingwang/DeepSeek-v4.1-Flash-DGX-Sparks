"""Offline regressions from Mia launcher PRs #4, #5, #8 and #26."""
import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from test_launcher_build import ROOT, shell


class NfsPortsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # `model_src()` is intentionally fail-fast: packing must never proceed
        # against an empty mount. Keep these launcher-only tests focused on NFS
        # ordering by providing the smallest valid checkpoint sentinel.
        (self.root / 'MODEL_DIR').mkdir()
        (self.root / 'MODEL_DIR' / 'config.json').write_text('{}')

    def test_own_exporter_reuse_keeps_checkpoint_at_root(self):
        out = shell('''
          nfs_live_container() { echo "$NFS_CONTAINER"; }
          nfs_rpc_ready() { return 0; }
          nfs_container_export_mount() { printf 'bind\t%s\n' "$MODEL_DIR"; }
          nfs_write_exports() { :; }
          NFS_REUSE_EXPORT=1
          nfs_ensure_server
          nfs_publish_model
          echo "REUSE=$NFS_REUSE_EXPORT DEVICE=$NFS_DEVICE"
        ''', self.root)
        self.assertIn('REUSE=0 DEVICE=:/', out)

    def test_own_exporter_with_stale_model_bind_is_refused_before_reconfigure(self):
        out = shell('''
          nfs_live_container() { echo "$NFS_CONTAINER"; }
          nfs_rpc_ready() { return 0; }
          nfs_container_export_mount() { printf 'bind\t%s\n' "$OLD_MODEL"; }
          nfs_write_exports() { echo BAD_RECONFIGURE; }
          if (nfs_ensure_server) 2>&1; then echo BAD_REUSE; else echo REFUSED; fi
        ''', self.root, {'OLD_MODEL': str(self.root / 'old-model')})
        self.assertIn('REFUSED', out)
        self.assertNotIn('BAD_RECONFIGURE', out)
        self.assertNotIn('BAD_REUSE', out)

    def test_own_exporter_requires_host_bind_at_export(self):
        out = shell('''
          nfs_live_container() { echo "$NFS_CONTAINER"; }
          nfs_rpc_ready() { return 0; }
          nfs_container_export_mount() { printf 'volume\t%s\n' "$MODEL_DIR"; }
          nfs_write_exports() { echo BAD_RECONFIGURE; }
          if (nfs_ensure_server) 2>&1; then echo BAD_REUSE; else echo REFUSED; fi
        ''', self.root)
        self.assertIn('REFUSED', out)
        self.assertNotIn('BAD_RECONFIGURE', out)
        self.assertNotIn('BAD_REUSE', out)

    def test_own_exporter_compares_canonical_model_paths(self):
        actual = self.root / 'actual-model'
        actual.mkdir()
        alias = self.root / 'model-alias'
        alias.symlink_to(actual, target_is_directory=True)
        out = shell('''
          nfs_live_container() { echo "$NFS_CONTAINER"; }
          nfs_rpc_ready() { return 0; }
          nfs_container_export_mount() { printf 'bind\t%s\n' "$ACTUAL_MODEL"; }
          nfs_write_exports() { :; }
          nfs_ensure_server
          echo "REUSE=$NFS_REUSE_EXPORT"
        ''', self.root, {'MODEL_DIR': str(alias), 'ACTUAL_MODEL': str(actual)})
        self.assertIn('REUSE=0', out)

    def test_hf_cache_exporter_still_publishes_subtree(self):
        out = shell('''
          nfs_live_container() { echo vllm-fn-nfs; }
          nfs_rpc_ready() { return 0; }
          nfs_write_exports() { :; }
          nfs_ensure_server
          echo "REUSE=$NFS_REUSE_EXPORT"
        ''', self.root)
        self.assertIn('REUSE=1', out)

    def test_single_rail_server_override(self):
        out = shell('nfs_server_ip_for worker1 0; nfs_server_ip_for worker3 2', self.root,
                    {'NFS_SERVER_IPS': '10.100.96.1'})
        self.assertEqual(out.splitlines(), ['10.100.96.1'] * 2)

    def test_pack_checks_and_reshares_before_any_pack(self):
        out = shell('''
          nfs_worker_has_model() { echo "PROBE $1"; return 1; }
          cmd_share() { echo SHARE; }
          docker() { echo "PACK_HEAD"; }
          remote_on() { echo PACK_WORKER; }
          cmd_pack
        ''', self.root)
        self.assertIn('SHARE', out)
        self.assertLess(out.index('SHARE'), out.index('PACK_HEAD'))
        self.assertEqual(out.count('PROBE'), 2)

    def test_local_pack_does_not_reshare(self):
        out = shell('''
          nfs_worker_has_model() { return 1; }
          cmd_share() { echo BAD_SHARE; }
          docker() { :; }
          remote_on() { :; }
          cmd_pack
        ''', self.root, {'NFS_SHARE': '0'})
        self.assertNotIn('BAD_SHARE', out)

    def test_gid_auto_is_opt_in_and_explicit_pin_wins(self):
        out = shell('''
          echo "DEFAULT=$(resolved_gid '')"
          NCCL_IB_GID_AUTO=1
          echo "AUTO=$(resolved_gid '')"
          echo "DETECTED=$(resolved_gid 4)"
          NCCL_IB_GID_INDEX=7
          echo "PIN=$(resolved_gid '')"
        ''', self.root)
        self.assertIn('DEFAULT=3\nAUTO=\nDETECTED=4\nPIN=7', out)

    def test_worker_omits_empty_gid(self):
        out = shell('worker_env_lines 10.0.0.2 "" 1', self.root)
        self.assertNotIn('NCCL_IB_GID_INDEX=', out)


class RemoteDefaultsTests(unittest.TestCase):
    def test_env_loaded_before_argument_defaults(self):
        spec = importlib.util.spec_from_file_location('remote_defaults', Path(os.environ.get('DSV41_REMOTE_TEST_SOURCE', ROOT / 'scripts/remote.py')))
        remote = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'pexpect': SimpleNamespace()}):
            spec.loader.exec_module(remote)
        with tempfile.TemporaryDirectory() as tmp:
            envfile = Path(tmp) / 'profile'
            envfile.write_text('WORKER_HOST=custom-host\nWORKER_USER=custom-user\n'
                               'SSH_IDENTITY="$HOME/custom-key"\nREMOTE_TIMEOUT=17\n')
            for option in (['--env-file', str(envfile)], ['--env-file=' + str(envfile)]):
                with patch.dict(os.environ, {'HOME': tmp}, clear=True), \
                     patch.object(sys, 'argv', ['remote.py', *option, 'true']), \
                     patch.object(remote, 'ssh_cmd', return_value=0) as ssh:
                    self.assertEqual(remote.main(), 0)
                    self.assertEqual(ssh.call_args.args, ('custom-host', 'custom-user', 'true', None,
                                                        tmp + '/custom-key', 17))

if __name__ == '__main__':
    unittest.main()
