#!/usr/bin/env python3
"""stop.sh must remove every dsv41-* container except the shared NFS exporter, on the head
and on every worker.

The worker filter lives inside a double-quoted remote script, so its awk program is only
right after bash expands that string: bare quotes around "dsv41-nfs" there closed the outer
string and turned the test into `$2 != dsv41-nfs` (numeric, always true), which removed the
exporter on workers. This expands each filter the way bash does and runs it on sample input.
"""
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "id-nfs dsv41-nfs\nid-w ds41-worker\nid-x dsv41-old\n"


def awk_lines():
    return [l for l in (ROOT / "stop.sh").read_text().splitlines() if "awk '" in l and "dsv41-nfs" in l]


def run_filter(line, remote):
    prog = line.strip().lstrip("|").strip()
    prog = prog[: prog.rindex("||")].strip() if "||" in prog else prog
    if remote:
        # inside the remote script's double quotes: let bash expand it once
        prog = subprocess.run(["bash", "-c", f's="{prog}"; printf %s "$s"'],
                              capture_output=True, text=True, check=True).stdout
    return subprocess.run(["bash", "-c", prog], input=SAMPLE, capture_output=True, text=True).stdout.split()


class StopKeepsNfs(unittest.TestCase):
    def test_head_and_worker_filters_keep_the_exporter(self):
        lines = awk_lines()
        self.assertEqual(len(lines), 2, lines)
        head, worker = lines
        for line, remote in ((head, False), (worker, True)):
            kept_out = run_filter(line, remote)
            self.assertNotIn("id-nfs", kept_out, line)
            self.assertIn("id-x", kept_out, line)


# docker answers stop.sh's own queries from the names in $STUB_DIR/names and logs every call
DOCKER = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_DIR/docker.calls"
case "$*" in
  "ps --format {{.Names}}") cat "$STUB_DIR/names" ;;
  "ps -a --filter name=dsv41- --format {{.ID}} {{.Names}}")
    grep '^dsv41-' "$STUB_DIR/names" | sed 's/^/id-/; s/^id-\\(.*\\)$/id-\\1 \\1/' ;;
  "rm -f "*) ;;
  *) exit 64 ;;
esac
"""
STUBS = {"docker": DOCKER, "curl": "#!/bin/sh\nexit 7\n", "pkill": "#!/bin/sh\nexit 0\n",
         "timeout": "#!/bin/sh\nshift\nexec \"$@\"\n"}
REMOTE = "import sys\nprint('STOPPED_' + sys.argv[sys.argv.index('--host') + 1])\n"


def run_stop(names):
    """stop.sh on the head with three stub workers and `names` running after its own cleanup."""
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in STUBS.items():
            f = Path(tmp, name)
            f.write_text(body)
            f.chmod(f.stat().st_mode | stat.S_IEXEC)
        Path(tmp, "names").write_text("".join(n + "\n" for n in names))
        Path(tmp, "remote.py").write_text(REMOTE)
        Path(tmp, "stop.env").write_text("")
        env = {"PATH": f"{tmp}:{os.environ['PATH']}", "HOME": tmp, "STUB_DIR": tmp, "LOG_DIR": tmp,
               "ENV_FILE": str(Path(tmp, "stop.env")), "WORKER_HOSTS": "w1,w2,w3",
               "REMOTE_PY": str(Path(tmp, "remote.py")), "SSH_IDENTITY": str(Path(tmp, "id"))}
        out = subprocess.run(["bash", str(ROOT / "stop.sh")], env=env, capture_output=True, text=True,
                             timeout=60, check=True).stdout
        calls = Path(tmp, "docker.calls").read_text().splitlines()
    return out, [c for c in calls if c.startswith("rm ")]


class StopReport(unittest.TestCase):
    def test_the_kept_exporter_is_not_left_over_and_the_banner_counts_nodes(self):
        out, removed = run_stop(["dsv41-nfs"])
        self.assertEqual(removed, ["rm -f dsv41-head"])  # the exporter is not removed
        self.assertIn("4\u00d7 Spark", out)
        self.assertNotIn("still running on head", out)
        self.assertIn("shared NFS exporter (dsv41-nfs)", out)
        for h in ("w1", "w2", "w3"):
            self.assertIn(f"{h}: container gone", out)

    def test_a_reused_exporter_is_named(self):
        for name in ("vllm-fn-nfs", "glm53-nfs"):
            out, _ = run_stop([name])
            self.assertIn(f"shared NFS exporter ({name})", out)
            self.assertNotIn("still running on head", out)

    def test_no_exporter_is_not_reported_as_kept(self):
        out, _ = run_stop([])
        self.assertNotIn("NFS exporter", out)
        self.assertIn("no dsv41-* serving containers on head", out)

    def test_a_real_leftover_still_warns(self):
        out, _ = run_stop(["dsv41-nfs", "dsv41-old"])
        self.assertIn("still running on head: dsv41-old", out)
        self.assertNotIn("dsv41-nfs", out.split("still running on head:")[1].splitlines()[0])

if __name__ == "__main__":
    unittest.main()
