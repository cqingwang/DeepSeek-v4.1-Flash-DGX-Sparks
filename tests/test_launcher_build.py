#!/usr/bin/env python3
"""Offline tests for the build step (`cmd_build`) of the launcher.

No Docker, SSH or network: the only thing driven here is the exclude list of the
rsync that stages the tree on the workers, because that one runs with `--delete`
against a copy of the repository and a wrong pattern removes files from it.
"""
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HAVE_RSYNC = shutil.which("rsync") is not None


def build_source():
    """The body of cmd_build() from start.sh."""
    source = (ROOT / "start.sh").read_text()
    build = source[source.index("cmd_build()"):]
    return build[:build.index("\n}\n")]


def rsync_excludes():
    return re.findall(r"--exclude '([^']+)'", build_source())


def launcher_source():
    """start.sh from the variable defaults through the helper functions, no dispatch."""
    source = (ROOT / "start.sh").read_text()
    start = source.index('if [[ "${DSV41_LAUNCHER:-}" == tp4 ]]')
    end = source.index('case "$CMD" in')
    # This slice starts after start.sh has parsed .env and materialized the
    # worker arrays. Recreate that launcher boundary explicitly instead of
    # weakening production's fail-fast configuration requirements.
    fixture = r'''
DSV41_LAUNCHER=tp4
TARGET_ENGINE_ID=deepseek_v4.1_flash_switchless
TARGET_CONTAINER_PREFIX=deepseek_v4.1_flash_switchless
NFS_SHARE="${NFS_SHARE:-0}"
WEIGHTS_MODE="${WEIGHTS_MODE:-local}"
NCCL_SWITCHLESS_RING_ONLY=0
HEAD_IP=10.0.0.1
read -r -a WORKER_IPS <<< "${WORKER_IPS:-10.0.0.2 10.0.0.3 10.0.0.4}"
read -r -a WORKER_HOSTS <<< "${WORKER_HOSTS:-worker-b worker-c worker-d}"
'''
    return "set -euo pipefail\n" + fixture + source[start:end]


def shell(code, directory, settings=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"], "ROOT": str(ROOT)}
    for name in ("MODEL_DIR", "COMMON_MODEL", "SSH_IDENTITY", "NCCL_HOST_DIR",
                 "STATE_DIR", "LOG_DIR", "WORKER_DIR", "WORKER_ENGRAM_DIR", "ENGRAM_DIR", "HF_EXPORT_ROOT"):
        env[name] = str(directory / name)
    env.update({
        "WORKER_IPS": "10.0.0.2 10.0.0.3 10.0.0.4",
        "WORKER_HOSTS": "worker-b worker-c worker-d",
    })
    env.update(settings or {})
    result = subprocess.run(["bash", "-c", launcher_source() + "\n" + code],
                            env=env, cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def rsync(source, destination):
    args = ["rsync", "-aH", "--delete"]
    for pattern in rsync_excludes():
        args += ["--exclude", pattern]
    args += [f"{source}/", f"{destination}/"]
    result = subprocess.run(args, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


class BuildTargetTests(unittest.TestCase):
    """`IMAGE` is whatever the profile names, so `build` must compile that recipe."""

    def test_managed_tp4_container_names_follow_the_parent_target_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = shell(r"""
              printf 'HEAD=%s\n' "$HEAD_CTN"
              for rank in 0 1 2 3; do
                printf '%s\n' "$(container_name_for_rank "$rank")"
              done
            """, Path(tmp), {
                "TARGET_CONTAINER_PREFIX": "deepseek_v4.1_flash_switchless",
            })
        self.assertEqual(output.splitlines(), [
            "HEAD=deepseek_v4.1_flash_switchless-r0",
            "deepseek_v4.1_flash_switchless-r0",
            "deepseek_v4.1_flash_switchless-r1",
            "deepseek_v4.1_flash_switchless-r2",
            "deepseek_v4.1_flash_switchless-r3",
        ])

    def test_managed_local_weights_use_readonly_host_mount_without_named_volume(self):
        source = (ROOT / "start.sh").read_text()
        self.assertIn('local) printf \'type=bind,source=%s,target=%s,readonly', source)
        self.assertIn('worker_model_has_weights()', source)
        self.assertIn('WORKER_MODEL_DIR', source)
        self.assertIn('"$src:/models/DeepSeek-V4.1-Flash:ro"', source)

        with tempfile.TemporaryDirectory() as tmp:
            output = shell(r"""
          WEIGHTS_MODE=local
          NFS_SHARE=0
          WORKER_MODEL_DIR=/opt/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
          NFS_VOLUME=dsv41-weights
          nfs_worker_has_model() { echo named-volume-probe; return 1; }
          worker_local_model_has_weights() { printf 'host-path-check %s %s\n' "$1" "$2"; }
          worker_model_has_weights spark-b
          worker_model_mount_spec
        """, Path(tmp))
        self.assertIn("host-path-check spark-b /opt/models/deepseek-ai/DeepSeek-V4.1-Flash-1017", output)
        self.assertNotIn("named-volume-probe", output)
        self.assertIn(
            "type=bind,source=/opt/models/deepseek-ai/DeepSeek-V4.1-Flash-1017,"
            "target=/models/DeepSeek-V4.1-Flash,readonly",
            output,
        )
        self.assertIn('-v "$src:/models/DeepSeek-V4.1-Flash:ro"', source)
        self.assertIn('--mount $(printf \'%q\' "$worker_model_mount")', source)

    def test_local_worker_preflight_checks_config_and_every_shard_without_a_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            capture = Path(tmp) / "remote-check.txt"
            shell(r"""
              remote_on() { printf 'HOST=%s\nTIMEOUT=%s\n%s\n' "$1" "$3" "$4" > "$CAPTURE"; }
              worker_local_model_has_weights spark-b /opt/models/deepseek-ai/DeepSeek-V4.1-Flash-1017
            """, Path(tmp), {"EXPECTED_SHARDS": "48", "CAPTURE": str(capture)})
            output = capture.read_text()
        self.assertIn("HOST=spark-b", output)
        self.assertIn("test -r \"$model_dir/config.json\" && test -s \"$model_dir/config.json\"", output)
        self.assertIn('set -- "$model_dir"/model-*-of-*.safetensors', output)
        self.assertIn('test "$#" -ge 48', output)
        self.assertIn('for shard do test -r "$shard" && test -s "$shard"', output)
        self.assertNotIn("docker volume", output)

    def test_local_checkpoint_preflight_happens_before_build_or_container_replacement(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        self.assertLess(serve.index("worker_local_model_has_weights"), serve.index("cmd_build"))
        self.assertLess(serve.index("worker_local_model_has_weights"), serve.index("docker rm"))

    def test_managed_tp4_preflights_every_rank_before_replacing_containers(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        self.assertIn("preflight_all_nodes", serve)
        self.assertLess(serve.index("preflight_all_nodes"), serve.index('docker rm -f "$HEAD_CTN"'))

    def test_serve_rejects_images_without_the_local_overlay_identity(self):
        source = (ROOT / "start.sh").read_text()
        serve = source[source.index("cmd_serve()"):]
        serve = serve[:serve.index("\n}\n")]
        self.assertIn("overlay_image_present()", source)
        self.assertIn("com.spark.dsv41.overlay", source)
        self.assertLess(serve.index("overlay_image_present"), serve.index('docker rm -f "$HEAD_CTN"'))
        self.assertIn("overlay_image_matches_remote", serve)
        self.assertLess(serve.index("overlay_image_matches_remote"), serve.index('docker rm -f "$HEAD_CTN"'))

        with tempfile.TemporaryDirectory() as tmp:
            output = shell(r"""
              IMAGE=dsv41-test
              docker() {
                [[ "$*" == *"com.spark.dsv41.overlay"* ]] || return 2
                printf '%s\n' "${DOCKER_OVERLAY_LABEL:-}"
              }
              DOCKER_OVERLAY_LABEL=1
              if overlay_image_present; then echo OVERLAY_OK; else echo OVERLAY_MISSING; fi
              DOCKER_OVERLAY_LABEL=
              if overlay_image_present; then echo OVERLAY_OK; else echo OVERLAY_MISSING; fi
            """, Path(tmp))
        self.assertIn("OVERLAY_OK", output)
        self.assertIn("OVERLAY_MISSING", output)

    def test_worker_ring_mounts_use_argument_arrays(self):
        source = (ROOT / "start.sh").read_text()
        self.assertIn("nccl_args=()", source)
        self.assertIn("nccl_mount_args nccl_args", source)
        # 远端 worker 命令使用 Bash 的空数组安全展开；内层 ${...} 还要再转义一次，
        # 因为命令会经过 ssh 的远端 shell -lc 这一层。
        self.assertIn(r'\"\${nccl_args[@]}\"', source)
        self.assertNotIn("NCCL_VOL=", source)
        self.assertNotIn("SHIM_VOL=", source)

    def test_build_compiles_the_configured_dockerfile(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SSH_IDENTITY.pub").write_text("ssh-ed25519 AAAA test\n")
            log = Path(tmp, "docker.log")
            shell(r"""
              BUILD_DOCKERFILE=Dockerfile.canary
              BUILD_ARGS="--build-arg PIP_INDEX=https://mirror.invalid/simple"
              bash() { printf 'SCRIPT %s\n' "$*"; }
              # Bash 3.2 treats expansion of an empty array as an unbound variable
              # under `set -u`. Keep one fake worker and stub the two remote effects;
              # this test is about the selected local Dockerfile/build arguments.
              WORKER_HOSTS=(worker)
              remote_on() { return 0; }
              rsync() { return 0; }
              docker() { printf '%s\n' "$*" >> "$LOG"
                         case "$1 $2" in "image inspect") echo arm64;; esac; }
              cmd_build
            """, Path(tmp), {"LOG": str(log)})
            calls = [line for line in log.read_text().splitlines() if line.startswith("build ")]
            self.assertEqual(len(calls), 1, log.read_text())
            self.assertIn(f"-f {ROOT}/Dockerfile.canary", calls[0])
            self.assertIn("--build-arg PIP_INDEX=https://mirror.invalid/simple", calls[0])

    def test_tp4_builds_once_then_uses_ring_shipper(self):
        build = build_source()
        tp4 = build[build.index('if [[ "${DSV41_LAUNCHER:-}" == tp4 ]]'):build.index("ensure_ssh_keys")]
        self.assertIn('bash "$ROOT/scripts/ship_image_ring.sh"', tp4)
        self.assertIn("return 0", tp4)
        self.assertLess(build.index("docker build"), build.index('bash "$ROOT/scripts/ship_image_ring.sh"'))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ring_dir = root / "ring"
            (ring_dir / "comm/roce_ring").mkdir(parents=True)
            log = root / "docker.log"
            out = shell(r"""
              BUILD_DOCKERFILE=Dockerfile.canary-roce
              BUILD_ARGS="--build-arg PIP_INDEX=https://mirror.invalid/simple"
              B12X_RING_HOST_DIR="$RING_DIR"
              build_base_image() { echo lmsysorg/sglang:dev-dsv41; }
              base_image_ready() { :; }
              docker() {
                printf 'DOCKER %s\n' "$*" >> "$LOG"
                if [[ "$1 $2" == "image inspect" ]]; then echo arm64; fi
                return 0
              }
              bash() { printf 'SCRIPT %s\n' "$*" >> "$LOG"; }
              remote_on() { printf 'REMOTE %s\n' "$*" >> "$LOG"; }
              rsync() { printf 'RSYNC %s\n' "$*" >> "$LOG"; }
              cmd_build
            """, root, {"LOG": str(log), "RING_DIR": str(ring_dir)})
            calls = log.read_text().splitlines()
            self.assertEqual(sum(line.startswith("DOCKER build ") for line in calls), 1, out + str(calls))
            self.assertTrue(any("--build-context spark-ring=" in line for line in calls), calls)
            self.assertEqual(sum(line.startswith("SCRIPT ") for line in calls), 1, calls)
            self.assertFalse(any(line.startswith(("REMOTE ", "RSYNC ")) for line in calls), calls)
            self.assertNotIn("docker build on", out)

    def test_roce_build_passes_the_shared_parent_ring_as_named_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "SSH_IDENTITY.pub").write_text("ssh-ed25519 AAAA test\n")
            ring_dir = Path(tmp, "ring", "b12x")
            (ring_dir / "comm/roce_ring").mkdir(parents=True)
            (ring_dir / "comm/roce_ring/roce_oneshot.py").write_text("# shared source\n")
            log = Path(tmp, "docker.log")
            shell(
                r"""
                  BUILD_DOCKERFILE=Dockerfile.canary-roce
                  WORKER_HOSTS=()
              bash() { printf 'SCRIPT %s\n' "$*"; }
                  docker() { printf '%s\n' "$*" >> "$LOG"
                             case "$1 $2" in "image inspect") echo arm64;; esac; }
                  cmd_build
                """,
                Path(tmp),
                {"LOG": str(log), "B12X_RING_HOST_DIR": str(ring_dir)},
            )
            calls = [line for line in log.read_text().splitlines() if line.startswith("build ")]
            self.assertEqual(len(calls), 1, log.read_text())
            self.assertIn(f"--build-context spark-ring={ring_dir}", calls[0])
            self.assertIn(f"-f {ROOT}/Dockerfile.canary-roce", calls[0])

    def test_a_dockerfile_outside_the_repository_is_rejected(self):
        """The workers only see what the rsync staged, so an absolute path cannot work."""
        build = build_source()
        self.assertIn("BUILD_DOCKERFILE must be inside the repository", build)


@unittest.skipUnless(HAVE_RSYNC, "rsync is not installed")
class RsyncExcludesTests(unittest.TestCase):
    """cmd_build must not exclude the staged SGLang tree from the workers."""

    def test_the_staged_tree_reaches_the_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp, "src"), Path(tmp, "dst")
            staged = "runtime/sglang-canary/python/sglang/srt/models"
            (src / staged).mkdir(parents=True)
            (src / staged / "deepseek_v4.py").write_text("branch")
            (dst / staged).mkdir(parents=True)
            (dst / staged / "deepseek_v4.py").write_text("stale")
            (src / "models").mkdir()
            (src / "models" / "shard.safetensors").write_text("weights")

            rsync(src, dst)

            self.assertEqual((dst / staged / "deepseek_v4.py").read_text(), "branch",
                             "the staged SGLang tree never reached the worker")
            self.assertFalse((dst / "models").exists(),
                             "the checkpoint directory at the repository root was copied")

    def test_every_exclude_is_anchored(self):
        """Only cache and exact env-name excludes are intentionally unanchored."""
        for pattern in rsync_excludes():
            if pattern in (".env", ".env.tp4", "__pycache__", "*.pyc"):
                continue  # exact env filenames and Python-cache subtree are intentional
            self.assertTrue(pattern.startswith("/"), f"unanchored exclude: {pattern}")


class CanaryDockerfileTests(unittest.TestCase):
    """Dockerfile.canary installs two wheels from PyPI; an offline site needs a mirror."""

    def test_tp4_images_keep_the_launcher_overlay_identity(self):
        """启动器必须能在每个节点识别本地构建的 TP4 镜像。"""
        for name in ("Dockerfile.canary", "Dockerfile.canary-roce"):
            text = (ROOT / name).read_text()
            self.assertIn('LABEL com.spark.dsv41.overlay="1"', text, name)

    def test_roce_image_applies_encoding_compatibility_patch(self):
        """RoCE 派生镜像不能依赖旧基础层遗留的文本编码补丁。"""
        text = (ROOT / "Dockerfile.canary-roce").read_text()
        self.assertIn("COPY runtime/patch_encoding_dsv41.py /opt/dsv41/runtime/patch_encoding_dsv41.py", text)
        self.assertIn("RUN python3 /opt/dsv41/runtime/patch_encoding_dsv41.py", text)

    def test_roce_image_declares_pinned_provenance(self):
        """TP4 RoCE 镜像必须把源码和第三方基线写进 OCI 标签。"""
        text = (ROOT / "Dockerfile.canary-roce").read_text()
        for marker in ("com.spark.dsv41.source-revision", "com.spark.dsv41.base-image",
                       "com.spark.dsv41.sglang-commit", "com.spark.dsv41.b12x-commit",
                       "com.spark.dsv41.roce-ring-commit"):
            self.assertIn(marker, text)

    def test_roce_image_embeds_runtime_provenance_files(self):
        """最终镜像必须能离线审计第三方源码基线。"""
        text = (ROOT / "Dockerfile.canary-roce").read_text()
        self.assertIn("COPY runtime/sources.manifest /opt/dsv41/runtime/sources.manifest", text)
        self.assertIn("runtime/b12x/source-provenance.json", text)
        self.assertIn("COPY --from=spark-ring /roce_ring-provenance.json", text)
        self.assertIn("COPY --from=spark-ring /comm/roce_ring", text)
        ring_source_dir = ROOT / "runtime/b12x/b12x/comm/roce_ring"
        vendored_sources = [
            path for path in ring_source_dir.rglob("*")
            if path.is_file() and "__pycache__" not in path.relative_to(ring_source_dir).parts
        ] if ring_source_dir.is_dir() else []
        self.assertFalse(
            vendored_sources,
            "path-aware RoCEnante source must remain owned by the parent spark/ring module",
        )

    def test_the_pip_index_is_overridable(self):
        text = (ROOT / "Dockerfile.canary").read_text()
        self.assertIn("ARG PIP_INDEX=https://pypi.org/simple", text)
        self.assertIn('-i "$PIP_INDEX"', text)


class DockerfilePrefillBypassTests(unittest.TestCase):
    def test_every_serving_image_checks_hook_against_its_final_scheduler(self):
        source_root = ('DSV41_PREFILL_BYPASS_SOURCE_ROOT='
                       '/sgl-workspace/sglang/python/sglang/srt/managers')
        for name in ('Dockerfile', 'Dockerfile.canary', 'Dockerfile.canary-roce'):
            with self.subTest(name=name):
                text = (ROOT / name).read_text()
                command = next(line for line in text.splitlines()
                               if 'test_prefill_bypass.py' in line)
                self.assertIn(source_root, command)
                self.assertIn('PYTHONPATH=/opt/dsv41/adapter', command)
                expected = 'unsupported' if name == 'Dockerfile' else 'supported'
                self.assertIn(f'DSV41_PREFILL_BYPASS_EXPECT={expected}', command)
