#!/usr/bin/env bash
# mkenv.sh NAME KEY=VAL ... -> $RESULTS_DIR/envs/NAME.env: a copy of .env with the given keys replaced/appended.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$ROOT"
NAME="$1"; shift
OUT="${RESULTS_DIR:?}/envs/$NAME.env"; BASE="${BASE_ENV:-.env}"
cp "$BASE" "$OUT"; chmod 600 "$OUT"
for kv in "$@"; do
  k="${kv%%=*}"
  if grep -q "^$k=" "$OUT"; then sed -i "s|^$k=.*|$kv|" "$OUT"; else printf '%s\n' "$kv" >> "$OUT"; fi
done
echo "$OUT"
