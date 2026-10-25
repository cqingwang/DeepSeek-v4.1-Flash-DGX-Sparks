#!/usr/bin/env bash
# Wait for this node's configured fabric links before running the boot-time mesh setup.
set -euo pipefail

CONFIG=${SPARK_CONFIG:-/opt/spark/config.yaml}
PROGRAM=${SPARK_PROGRAM:-/opt/spark/program.py}
NODE=${SPARK_NODE:-$(hostname)}
PORTS=$(python3 "$PROGRAM" ring-local-fabric --config "$CONFIG" --node "$NODE")
[[ -n "$PORTS" ]] || { echo "no configured fabric ports for $NODE" >&2; exit 2; }
mapfile -t NETDEVS < <(awk -F '\t' 'NF { print $1 }' <<< "$PORTS")
[[ ${#NETDEVS[@]} -eq 4 ]] || { echo "$NODE: expected four configured fabric interfaces" >&2; exit 2; }

for attempt in $(seq 1 60); do
  ready=1
  for netdev in "${NETDEVS[@]}"; do
    [[ "$(cat "/sys/class/net/$netdev/operstate" 2>/dev/null || true)" == up ]] || ready=0
  done
  [[ $ready == 1 ]] && exit 0
  sleep 2
done
echo "$NODE: configured fabric interfaces did not become ready: ${NETDEVS[*]}" >&2
exit 1
