"""CPU-only build/pull regressions; all Docker and remote commands are stubs."""
import hashlib
import importlib.util
import io
import json
import os
import stat
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_launcher_build import ROOT, shell


class OfflineBuildTests(unittest.TestCase):
    def run_shell(self, code, **settings):
        with tempfile.TemporaryDirectory() as tmp:
            return shell('ensure_ssh_keys() { :; }\n' + code, Path(tmp), settings)

    def test_b12x_gid_patch_imports_shared_roce_env_from_parent_comm(self):
        patch_path = ROOT / 'scripts/b12x-required-gid.patch'
        patch_source = patch_path.read_text()
        import_line = next(
            line for line in patch_source.splitlines()
            if line.startswith('+from .') and 'required_roce_gid_index' in line
        )
        relative_module = import_line[len('+from '):].split(' import ', 1)[0]
        resolved_module = importlib.util.resolve_name(
            relative_module,
            'b12x.comm.roce',
        )
        self.assertEqual(resolved_module, 'b12x.comm.roce_env')

        fetch_source = (ROOT / 'scripts/fetch_runtime.sh').read_text()
        self.assertIn(
            'cp "$source/roce_env.py" "$OUT/b12x/b12x/comm/roce_env.py"',
            fetch_source,
        )
        manifest_rows = (ROOT / 'runtime/sources.manifest').read_text().splitlines()
        manifest_row = next(
            line.split() for line in manifest_rows
            if line.split()[:2] == ['local', 'scripts/b12x-required-gid.patch']
        )
        self.assertEqual(
            hashlib.sha256(patch_path.read_bytes()).hexdigest(),
            manifest_row[2],
        )

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
        self.assertEqual(out.count('DOCKER image inspect example/base:v1'), 4)

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
        self.assertEqual(out.count('PROXY_CHECK'), 4)
        self.assertEqual(out.count('PROXY_CHECK lmsysorg/sglang:dev-dsv41'), 4)
        self.assertEqual(out.count('DOCKER pull'), 4)
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
            self.assertIn('bash /src/scripts/fetch-sglang-canary.sh', source)
        self.assertIn('COPY --from=fetch /fetched/sglang-canary/python /sgl-workspace/sglang/python', source)
        source = (ROOT / 'Dockerfile.canary-roce').read_text()
        self.assertIn('FROM ${BASE_IMAGE} AS fetch', source)
        self.assertIn(
            'RUN SPARK_RING_DIR=/src/spark-ring bash /src/scripts/fetch_runtime.sh /fetched b12x b12x-next',
            source,
        )
        self.assertIn('FROM ${FETCH_STAGE} AS fetch_selected', source)
        self.assertIn('COPY --from=fetch_selected', source)
        self.assertNotIn('runtime/fetched', (ROOT / '.dockerignore').read_text())

    def test_upstream_runtime_tests_are_copied_from_the_pinned_fetch_stage(self):
        canary = (ROOT / 'Dockerfile.canary').read_text()
        self.assertIn(
            'COPY --from=fetch /fetched/sglang-canary/upstream-tests/',
            canary,
        )
        self.assertNotIn(
            'COPY runtime/sglang-canary/upstream-tests/',
            canary,
        )

        roce = (ROOT / 'Dockerfile.canary-roce').read_text()
        self.assertIn(
            'COPY --from=fetch_selected /fetched/sglang-canary/upstream-tests/',
            roce,
        )
        fetch_local = roce.split('FROM ${BASE_IMAGE} AS fetch_local', 1)[1].split(
            'FROM ${FETCH_STAGE} AS fetch_selected', 1
        )[0]
        self.assertIn(
            'COPY runtime/sglang-canary/upstream-tests /fetched/sglang-canary/upstream-tests',
            fetch_local,
        )
        self.assertNotIn(
            'COPY runtime/sglang-canary/upstream-tests/test_',
            roce,
        )

    def test_canary_profiles_keep_drm_and_tp4_draft_head_coverage(self):
        for name in ('Dockerfile.canary', 'Dockerfile.canary-roce'):
            with self.subTest(dockerfile=name):
                source = (ROOT / name).read_text()
                self.assertIn('-DDSV41_ROW_STORE_DRM adapter/row_store.cpp', source)
                self.assertIn('tests/test_draft_head_fp8_tp4.py', source)

    def test_roce_default_fetch_applies_the_pinned_sglang_backports(self):
        for name in ('Dockerfile.canary', 'Dockerfile.canary-roce'):
            with self.subTest(dockerfile=name):
                source = (ROOT / name).read_text()
                self.assertIn('COPY patches /src/patches', source)
                self.assertIn('COPY scripts /src/scripts', source)
                self.assertIn('COPY runtime/patch_*.py /src/runtime/', source)
                self.assertIn('bash /src/scripts/fetch-sglang-canary.sh', source)
                self.assertIn(
                    'cp -R /src/runtime/sglang-canary /fetched/sglang-canary',
                    source,
                )
                self.assertNotIn(
                    'fetch_runtime.sh /fetched sglang-canary',
                    source,
                )

    def test_roce_local_fetch_requires_a_prepatched_sglang_tree(self):
        source = (ROOT / 'Dockerfile.canary-roce').read_text()
        fetch_local = source.split('FROM ${BASE_IMAGE} AS fetch_local', 1)[1].split(
            'FROM ${FETCH_STAGE} AS fetch_selected', 1
        )[0]
        self.assertIn(
            'COPY runtime/sglang-canary/python /fetched/sglang-canary/python',
            fetch_local,
        )
        self.assertIn(
            'COPY runtime/sglang-canary/REF runtime/sglang-canary/PATCHES '
            '/fetched/sglang-canary/',
            fetch_local,
        )
        self.assertIn('test -s /fetched/sglang-canary/PATCHES', fetch_local)
        self.assertNotIn('COPY runtime/fetched/sglang-canary', fetch_local)

    def test_canary_final_stages_fit_the_pinned_overlay2_layer_budget(self):
        # The pinned arm64 base has 77 filesystem layers; overlay2 supports 128 total.
        for name in ('Dockerfile.canary', 'Dockerfile.canary-roce'):
            with self.subTest(dockerfile=name):
                source = (ROOT / name).read_text()
                final_stage_start = source.rfind('\nFROM ${BASE_IMAGE}\n') + 1
                final_stage = source[final_stage_start:]
                layer_instructions = sum(
                    line.startswith(('RUN ', 'COPY ', 'ADD '))
                    for line in final_stage.splitlines()
                )
                self.assertLessEqual(
                    layer_instructions,
                    51,
                    f'{name} adds {layer_instructions} layers to a 77-layer base',
                )

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
        self.assertIn('ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_USER@$source_management" "$remote_command"', source)
        self.assertIn('ssh -n -o BatchMode=yes "$REMOTE_USER@$target_management" "$REMOTE_VERIFY_AND_LOAD"', source)
        self.assertNotIn('ssh -n -o BatchMode=yes -o ConnectTimeout=15 -b "$source_fabric"', source)
        self.assertIn("docker image inspect --format '{{.Id}}'", source)
        self.assertIn('archive_image_id "$IMAGE_ARCHIVE"', source)
        self.assertIn('archive_image_id "$PARTIAL_ARCHIVE"', source)
        self.assertIn('sha256sum -c -', source)
        self.assertIn('REMOTE_VERIFY_AND_LOAD', source)
        self.assertNotIn('rsync --delete', source)

    def test_ring_shipper_archive_identity_is_bound_to_the_selected_tag_and_config(self):
        source = (ROOT / 'scripts/ship_image_ring.sh').read_text()
        start = source.index('archive_image_id() {')
        end = source.index('\n}\n', start) + len('\n}\n')
        function_source = source[start:end]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'image.tar'
            config = b'{"architecture":"arm64","rootfs":{"type":"layers","diff_ids":[]}}\n'
            config_hash = hashlib.sha256(config).hexdigest()

            bin_dir = root / 'bin'
            bin_dir.mkdir()
            sha256sum = bin_dir / 'sha256sum'
            sha256sum.write_text(
                '#!/usr/bin/env python3\n'
                'import hashlib, sys\n'
                'print(hashlib.sha256(sys.stdin.buffer.read()).hexdigest() + "  -")\n'
            )
            sha256sum.chmod(sha256sum.stat().st_mode | stat.S_IXUSR)
            env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}')
            command = f'IMAGE="example/image:current"\n{function_source}\narchive_image_id "$1"'

            def inspect_archive(config_path, include_config=True):
                manifest = json.dumps([{
                    'Config': config_path,
                    'RepoTags': ['example/image:current'],
                    'Layers': [],
                }]).encode()
                with tarfile.open(archive, 'w') as tar:
                    entries = [('manifest.json', manifest)]
                    if include_config:
                        entries.append((config_path, config))
                    for name, contents in entries:
                        entry = tarfile.TarInfo(name)
                        entry.size = len(contents)
                        tar.addfile(entry, io.BytesIO(contents))
                return subprocess.run(
                    ['bash', '-c', command, '--', str(archive)],
                    env=env, capture_output=True, text=True, check=False,
                )

            for config_path in (f'{config_hash}.json', f'blobs/sha256/{config_hash}'):
                with self.subTest(config_path=config_path):
                    result = inspect_archive(config_path)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), f'sha256:{config_hash}')

            traversal = inspect_archive(f'../{config_hash}.json', include_config=False)
            self.assertNotEqual(traversal.returncode, 0)
            self.assertIn('unsafe Docker image config path', traversal.stderr)

            unsupported = inspect_archive(f'configs/{config_hash}.json', include_config=False)
            self.assertNotEqual(unsupported.returncode, 0)
            self.assertIn('unsafe Docker image config path', unsupported.stderr)

            wrong_tag = command.replace('example/image:current', 'example/image:stale', 1)
            rejected = subprocess.run(
                ['bash', '-c', wrong_tag, '--', str(archive)],
                env=env, capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('RepoTags entry', rejected.stderr)

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
