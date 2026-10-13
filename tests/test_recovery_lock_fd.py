"""Recovery-lock inheritance contract for start.sh's detached log follower."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
START = ROOT / "start.sh"


def shell_function(source: str, name: str) -> str:
    marker = f"{name}() {{\n"
    start = source.index(marker)
    depth = 0
    for match in re.finditer(r"[{}]", source[start:]):
        depth += 1 if match.group() == "{" else -1
        if depth == 0:
            return source[start : start + match.end()] + "\n"
    raise AssertionError(f"unterminated shell function: {name}")


class RecoveryLockFdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = START.read_text()
        cls.validate = shell_function(source, "_validate_d2_recovery_lock_fd")
        cls.launch = shell_function(source, "_start_detached_head_log_follower")

    def test_invalid_value_fails_before_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / "mutation"
            script = f"""set -euo pipefail
err() {{ printf '%s\\n' "$*" >&2; }}
{self.validate}
D2_RECOVERY_LOCK_FD=8
_validate_d2_recovery_lock_fd
: >{str(marker)!r}
"""
            run = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            self.assertEqual(run.returncode, 64, run.stdout + run.stderr)
            self.assertFalse(marker.exists())
            self.assertIn("empty or exactly 9", run.stderr)

    def test_empty_value_keeps_existing_behavior(self) -> None:
        script = f"""set -euo pipefail
err() {{ :; }}
{self.validate}
unset D2_RECOVERY_LOCK_FD
_validate_d2_recovery_lock_fd
"""
        subprocess.run(["bash", "-c", script], check=True)

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc/self/fd").is_dir(), "Linux procfs proof")
    def test_detached_follower_drops_fd9_but_launcher_keeps_lock(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bindir = root / "bin"
            bindir.mkdir()
            follower_pid = root / "follower.pid"
            wrapper_ready = root / "wrapper.ready"
            wrapper_release = root / "wrapper.release"
            docker = bindir / "docker"
            docker.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s\\n' \"$$\" >{str(follower_pid)!r}\n"
                "while :; do sleep 1; done\n"
            )
            docker.chmod(0o755)
            setsid = bindir / "setsid"
            setsid.write_text("#!/usr/bin/env bash\nexec \"$@\"\n")
            setsid.chmod(0o755)
            lock = root / "recovery.lock"
            serve_log = root / "serve.log"
            harness = root / "harness.sh"
            harness.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
{self.launch}
HEAD_CTN=test-a-head
SERVE_LOG={str(serve_log)!r}
D2_RECOVERY_LOCK_FD=9
export D2_RECOVERY_LOCK_FD
_start_detached_head_log_follower
: >{str(wrapper_ready)!r}
while [[ ! -e {str(wrapper_release)!r} ]]; do sleep .02; done
""")
            harness.chmod(0o755)

            wrapper_code = """
import fcntl,os,sys
fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600)
os.dup2(fd,9)
if fd!=9: os.close(fd)
fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
os.execve(sys.argv[2],[sys.argv[2]],dict(os.environ,PATH=sys.argv[3]))
"""
            proc = subprocess.Popen(
                [sys.executable, "-c", wrapper_code, str(lock), str(harness), f"{bindir}:/usr/bin:/bin"]
            )
            follower = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not (wrapper_ready.exists() and follower_pid.exists()):
                    time.sleep(0.02)
                self.assertTrue(wrapper_ready.exists())
                follower = int(follower_pid.read_text())
                self.assertTrue(Path(f"/proc/{follower}").exists())
                self.assertFalse(Path(f"/proc/{follower}/fd/9").exists())

                contender = os.open(lock, os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    wrapper_release.touch()
                    self.assertEqual(proc.wait(timeout=5), 0)
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                finally:
                    os.close(contender)
                self.assertTrue(Path(f"/proc/{follower}").exists(), "detached follower must remain alive")
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
                if follower is not None and Path(f"/proc/{follower}").exists():
                    os.kill(follower, signal.SIGTERM)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux flock/process proof")
    def test_wrapper_death_does_not_unlock_while_mutating_child_lives(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            lock = root / "recovery.lock"
            child_pid = root / "child.pid"
            code = """
import fcntl,os,sys,time
fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT,0o600);os.dup2(fd,9)
if fd!=9:os.close(fd)
fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
pid=os.fork()
if pid==0: time.sleep(2);os._exit(0)
open(sys.argv[2],'w').write(str(pid))
time.sleep(30)
"""
            wrapper = subprocess.Popen([sys.executable, "-c", code, str(lock), str(child_pid)])
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and not child_pid.exists():
                    time.sleep(0.02)
                self.assertTrue(child_pid.exists())
                wrapper.kill()
                wrapper.wait(timeout=5)
                contender = os.open(lock, os.O_RDWR)
                try:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    deadline = time.monotonic() + 4
                    while time.monotonic() < deadline:
                        try:
                            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            time.sleep(0.05)
                    else:
                        self.fail("mutating child did not release inherited recovery lock")
                finally:
                    os.close(contender)
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
