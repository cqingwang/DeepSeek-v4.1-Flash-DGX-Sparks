#!/usr/bin/env bash
# Run benchmarks/overnight_quality.py from spark2. Usage: quality.sh LABEL [needle_k,...]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
LABEL="$1"; NEEDLES="${2:-}"
R="${RESULTS_DIR:?set RESULTS_DIR}"; mkdir -p "$R"
SSH="ssh -i $HOME/.ssh/id_ed25519_shared -o IdentitiesOnly=yes zurih@spark2"
scp -q -i ~/.ssh/id_ed25519_shared benchmarks/overnight_quality.py zurih@spark2:/tmp/overnight_quality.py
$SSH "BASE_URL=http://10.0.0.1:8888 python3 /tmp/overnight_quality.py /tmp/oq-$LABEL.json '$NEEDLES'" | tee "$R/quality-$LABEL.txt"
scp -q -i ~/.ssh/id_ed25519_shared "zurih@spark2:/tmp/oq-$LABEL.json" "$R/"
