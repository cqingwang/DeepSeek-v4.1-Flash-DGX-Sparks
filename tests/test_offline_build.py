"""CPU-only build/pull regressions; all Docker and remote commands are stubs."""
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_launcher_build import ROOT, shell


class OfflineBuildTests(unittest.TestCase):
    def run_shell(self, code, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            return shell('ensure_ssh_keys() { :; }\n' + code, Path(tmp), settings)

    def test_skip_pull_uses_tag_without_digest(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*" >> "$LOG_DIR/commands"; }
          remote_on() { shift; if [[ "$1" == --timeout ]]; then shift 2; fi; bash -c "$(declare -f docker)
$*"; }
          cmd_pull
          cat "$LOG_DIR/commands"
        ''', SKIP_PULL='1', BASE_IMAGE='example/base:v1@sha256:123')
        self.assertNotIn('DOCKER pull', out)
        self.assertNotIn('@sha256:', '\n'.join(line for line in out.splitlines() if 'DOCKER' in line))
        self.assertEqual(out.count('DOCKER image inspect example/base:v1'), 3)

    def test_missing_local_base_fails_without_pull(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*"; return 1; }
          if (cmd_pull) 2>&1; then echo PASSED; else echo FAILED; fi
        ''', SKIP_PULL='1')
        self.assertIn('FAILED', out)
        self.assertNotIn('DOCKER pull', out)
        self.assertIn('SKIP_PULL=1', out)

    def test_proxy_precheck_precedes_every_pull(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*" >> "$LOG_DIR/commands"; }
          docker_proxy_precheck() { echo "PROXY_CHECK $1"; }
          remote_on() { shift; if [[ "$1" == --timeout ]]; then shift 2; fi; bash -c "$(declare -f docker docker_proxy_precheck)
$*"; }
          cmd_pull
          cat "$LOG_DIR/commands"
        ''')
        self.assertEqual(out.count('PROXY_CHECK'), 3)
        self.assertEqual(out.count('PROXY_CHECK lmsysorg/sglang:dev-dsv41'), 3)
        self.assertEqual(out.count('DOCKER pull'), 3)
        self.assertLess(out.index('PROXY_CHECK'), out.index('DOCKER pull'))

    def test_dead_proxy_stops_pull(self):
        out = self.run_shell('''
          docker() { echo "DOCKER $*" >> "$LOG_DIR/commands"; }
          docker_proxy_precheck() { echo 'unreachable docker proxy'; return 1; }
          if (cmd_pull) 2>&1; then echo PASSED; else echo FAILED; fi
        ''')
        self.assertIn('FAILED', out)
        self.assertNotIn('DOCKER pull', out)

    def test_builds_pass_the_resolved_base_to_from(self):
        source = (ROOT / 'start.sh').read_text()
        build = source[source.index('cmd_build()'):source.index('cmd_share()')]
        self.assertIn('--build-arg', build)
        self.assertGreaterEqual(build.count('BASE_IMAGE='), 2)
        for name in ('Dockerfile', 'Dockerfile.canary', 'Dockerfile.canary-roce'):
            source = (ROOT / name).read_text()
            self.assertIn('ARG BASE_IMAGE=lmsysorg/sglang:dev-dsv41', source)
            self.assertIn('FROM ${BASE_IMAGE}', source)

    def test_recipes_with_fetch_copies_define_the_fetch_stage(self):
        for name in ('Dockerfile.canary',):
            source = (ROOT / name).read_text()
            self.assertIn('FROM ${BASE_IMAGE} AS fetch', source)
            self.assertIn('RUN bash /src/scripts/fetch_runtime.sh /fetched', source)
            self.assertIn('COPY --from=fetch', source)
        source = (ROOT / 'Dockerfile.canary-roce').read_text()
        self.assertIn('FROM ${BASE_IMAGE} AS fetch', source)
        self.assertIn('RUN bash /src/scripts/fetch_runtime.sh /fetched', source)
        self.assertIn('FROM ${FETCH_STAGE} AS fetch_selected', source)
        self.assertIn('COPY --from=fetch_selected', source)
        self.assertNotIn('runtime/fetched', (ROOT / '.dockerignore').read_text())

    def test_roce_recipe_supports_mac_prefetched_runtime(self):
        source = (ROOT / 'Dockerfile.canary-roce').read_text()
        self.assertIn('FROM ${BASE_IMAGE} AS fetch_local', source)
        self.assertIn('ARG FETCH_STAGE=fetch', source)
        self.assertIn('FROM ${FETCH_STAGE} AS fetch_selected', source)

    def test_ring_shipper_uses_adjacent_hops_and_same_image_id(self):
        source = (ROOT / 'scripts/ship_image_ring.sh').read_text()
        self.assertIn('ring-transfer-route', source)
        self.assertIn('--append-verify', source)
        self.assertIn('-b $source_fabric', source)
        self.assertIn('ssh -n -o BatchMode=yes -o ConnectTimeout=15 -b "$source_fabric"', source)
        self.assertIn("docker image inspect --format '{{.Id}}'", source)
        self.assertNotIn('rsync --delete', source)

    def test_flat_archive_preserves_runtime_config_and_ring_load_shape(self):
        source = (ROOT / 'scripts/make_flat_docker_archive.py').read_text()
        self.assertIn('"docker", "export"', source)
        self.assertIn('"docker", "rm"', source)
        self.assertIn('"config"', source)
        self.assertIn('rootfs', source)
        self.assertIn('manifest.json', source)
        self.assertIn('docker load', (ROOT / 'scripts/ship_image_ring.sh').read_text())


class ProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('proxy', ROOT / 'scripts/docker_proxy_check.py')
        cls.proxy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.proxy)

    def test_no_proxy_is_inert(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({})
            connect.assert_not_called()

    def test_proxy_fields_timeout_and_credentials_redacted(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({'HttpProxy': 'http://secret:password@proxy.example:3128',
                                      'HttpsProxy': 'http://secret:password@proxy.example:3128'})
            connect.assert_called_once_with(('proxy.example', 3128), timeout=2)

    def test_unreachable_proxy_fails_clearly(self):
        with patch.object(self.proxy.socket, 'create_connection', side_effect=OSError('refused')):
            with self.assertRaisesRegex(RuntimeError, 'docker proxy proxy.example:3128 unreachable') as exc:
                self.proxy.check_proxies({'HTTPSProxy': 'http://secret:password@proxy.example:3128'})
            self.assertNotIn('password', str(exc.exception))

    def test_only_https_proxy_applies_to_registry_pull(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({'HttpProxy': 'http://unused.example:3128',
                                      'HttpsProxy': 'http://selected.example:8080'})
            connect.assert_called_once_with(('selected.example', 8080), timeout=2)

    def test_http_only_proxy_does_not_apply(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({'HTTPProxy': 'http://unused.example:3128'})
            connect.assert_not_called()

    def test_no_proxy_bypasses_default_registry(self):
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            self.proxy.check_proxies({'HttpsProxy': 'http://proxy.example:3128',
                                      'NoProxy': ' other.example, docker.io '})
            connect.assert_not_called()

    def test_registry_is_derived_from_image(self):
        for image, host in (
                ('ubuntu:latest', 'docker.io'),
                ('library/ubuntu:latest', 'docker.io'),
                ('lmsysorg/sglang:dev-dsv41', 'docker.io'),
                ('docker.io/library/ubuntu:latest', 'docker.io'),
                ('nvcr.io/nvidia/base@sha256:123', 'nvcr.io'),
                ('registry.example:5000/org/base:v1', 'registry.example'),
                ('localhost/base:v1', 'localhost')):
            with self.subTest(image=image):
                with patch.object(self.proxy.socket, 'create_connection') as connect:
                    self.proxy.check_proxies({'HttpsProxy': 'http://proxy.example:3128',
                                              'NoProxy': host}, image)
                    connect.assert_not_called()

    def test_no_proxy_exact_suffix_and_wildcard(self):
        for bypass in ('registry.example', '.example', '*'):
            with self.subTest(bypass=bypass):
                with patch.object(self.proxy.socket, 'create_connection') as connect:
                    self.proxy.check_proxies({'HttpsProxy': 'http://proxy.example:3128',
                                              'NoProxy': bypass}, 'registry.example/base:v1')
                    connect.assert_not_called()

    def test_no_proxy_nonmatching_hosts_still_check_proxy(self):
        for bypass in ('other.example', 'example', '.registry.example', '.other.example'):
            with self.subTest(bypass=bypass):
                with patch.object(self.proxy.socket, 'create_connection') as connect:
                    self.proxy.check_proxies({'HttpsProxy': 'http://proxy.example:3128',
                                              'NoProxy': bypass}, 'registry.example/base:v1')
                    connect.assert_called_once_with(('proxy.example', 3128), timeout=2)

    def test_cli_uses_image_argument(self):
        import runpy
        import sys
        details = {'HttpsProxy': 'http://proxy.example:3128', 'NoProxy': '.example'}
        with patch.object(self.proxy.socket, 'create_connection') as connect:
            with patch.object(sys, 'stdin', io.StringIO(json.dumps(details))):
                with patch.object(sys, 'argv', ['-c', 'registry.example/base:v1']):
                    runpy.run_path(str(ROOT / 'scripts/docker_proxy_check.py'), run_name='__main__')
            connect.assert_not_called()

if __name__ == '__main__':
    unittest.main()
