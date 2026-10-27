#!/usr/bin/env python3
"""stop.sh must remove only the configured target's exact rank containers."""
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "deepseek_v4.1_flash_switchless"

DOCKER = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_DIR/docker.calls"
case "$*" in
  "ps --format {{.Names}}") cat "$STUB_DIR/names" ;;
  "rm -f "*) ;;
  "exec "*) ;;
  *) exit 64 ;;
esac
"""
STUBS = {
    "docker": DOCKER,
    "curl": "#!/bin/sh\nexit 7\n",
    "pkill": "#!/bin/sh\nexit 0\n",
    "timeout": "#!/bin/sh\nshift\nexec \"$@\"\n",
}
REMOTE = """import os
import sys

host = sys.argv[sys.argv.index("--host") + 1]
with open(os.environ["REMOTE_COMMANDS"], "a") as output:
    output.write(host + "\\n" + sys.argv[-1] + "\\n---\\n")
print("STOPPED_" + host)
"""


def run_stop():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, body in STUBS.items():
            path = root / name
            path.write_text(body)
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        (root / "names").write_text("dsv41-nfs\nother-engine-r0\ndsv41-old\n")
        remote_script = root / "remote.py"
        remote_script.write_text(REMOTE)
        env_file = root / "stop.env"
        env_file.write_text(
            f"DSV41_LAUNCHER=tp4\n"
            f"TARGET_CONTAINER_PREFIX={PREFIX}\n"
            "NNODES=4\n"
            "WORKER_HOSTS=w1,w2,w3\n"
            "WORKER_IPS=192.0.2.11,192.0.2.12,192.0.2.13\n"
            "WORKER_USER=\n"
            "NFS_VOLUME=dsv41-weights\n"
        )
        env = {
            "PATH": f"{root}:{os.environ['PATH']}",
            "HOME": tmp,
            "STUB_DIR": tmp,
            "REMOTE_COMMANDS": str(root / "remote.commands"),
            "ENV_FILE": str(env_file),
            "REMOTE_PY": str(remote_script),
            "SSH_IDENTITY": str(root / "id"),
            "LOG_DIR": tmp,
        }
        result = subprocess.run(
            ["bash", str(ROOT / "stop.sh")],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        docker_calls = (root / "docker.calls").read_text().splitlines()
        remote_calls = (root / "remote.commands").read_text().split("---\n")
    return result.stdout, docker_calls, remote_calls


class StopTargetContainers(unittest.TestCase):
    def test_stop_uses_parent_target_prefix_for_all_four_ranks(self):
        out, docker_calls, remote_calls = run_stop()
        self.assertIn("4× Spark", out)
        self.assertEqual(
            [call for call in docker_calls if call.startswith("rm ")],
            [f"rm -f {PREFIX}-r0"],
        )
        for rank, host in enumerate(("w1", "w2", "w3"), start=1):
            remote_script = next(call for call in remote_calls if call.startswith(host + "\n"))
            self.assertIn(f"docker rm -f {PREFIX}-r{rank}", remote_script)
            self.assertNotIn("docker rm -f $ids", remote_script)
            self.assertNotIn("--filter name=dsv41-", remote_script)
        self.assertIn(f"no {PREFIX}-r* serving containers on head", out)

    def test_stop_never_removes_shared_or_other_engine_containers(self):
        out, docker_calls, remote_calls = run_stop()
        removals = [call for call in docker_calls if call.startswith("rm ")]
        self.assertEqual(removals, [f"rm -f {PREFIX}-r0"])
        self.assertTrue(all("dsv41-nfs" not in call for call in docker_calls if call.startswith("rm ")))
        self.assertTrue(all("other-engine-r0" not in call for call in docker_calls if call.startswith("rm ")))
        self.assertTrue(all("dsv41-old" not in call for call in docker_calls if call.startswith("rm ")))
        self.assertTrue(all("--filter name=dsv41-" not in call for call in remote_calls))
        self.assertIn("shared NFS exporter (dsv41-nfs)", out)


if __name__ == "__main__":
    unittest.main()
