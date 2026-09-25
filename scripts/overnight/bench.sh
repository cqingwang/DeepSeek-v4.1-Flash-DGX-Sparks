#!/usr/bin/env bash
# Run benchmarks/overnight_bench.py from spark2 (a worker, off the head's CPU) and
# join the head's "Decode batch" acceptance lines for each phase window.
# Usage: scripts/overnight/bench.sh LABEL [reps] [max_tokens]   (PHASES=c1,c4)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
LABEL="$1"; REPS="${2:-5}"; MAXT="${3:-512}"
R="${RESULTS_DIR:?set RESULTS_DIR}"; mkdir -p "$R"
REM="ssh -i $HOME/.ssh/id_ed25519_shared -o IdentitiesOnly=yes zurih@spark2"
KEY=$(cat state/api-key 2>/dev/null || true)
scp -q -i ~/.ssh/id_ed25519_shared benchmarks/overnight_bench.py zurih@spark2:/tmp/overnight_bench.py
$REM "rm -f /tmp/ob-$LABEL.jsonl; API_KEY=$KEY PHASES=${PHASES:-c1,c4} C1_WORKLOADS=${C1_WORKLOADS:-prose,code,chat_sampled,prose2} BASE_URL=http://10.0.0.1:8888 python3 /tmp/overnight_bench.py /tmp/ob-$LABEL.jsonl $REPS $MAXT" | tr -d '\r' | tee "$R/bench-$LABEL.txt"
scp -q -i ~/.ssh/id_ed25519_shared "zurih@spark2:/tmp/ob-$LABEL.jsonl" "zurih@spark2:/tmp/ob-$LABEL.summary.json" "$R/"
python3 scripts/overnight/accept.py "$R/ob-$LABEL.summary.json" | tee -a "$R/bench-$LABEL.txt"
