#!/usr/bin/env bash
# Apply the driverinit hairpin queue value from common.ring, then wait for every configured port.
# Reinitializing a function drops its RDMA connections; the service runs this only at boot.
set -euo pipefail

CONFIG=${SPARK_CONFIG:-/opt/spark/config.yaml}
PROGRAM=${SPARK_PROGRAM:-/opt/spark/program.py}
NODE=${SPARK_NODE:-$(hostname)}
PORTS=$(python3 "$PROGRAM" ring-local-fabric --config "$CONFIG" --node "$NODE")
[[ -n "$PORTS" ]] || { echo "no configured fabric ports for $NODE" >&2; exit 2; }

while IFS=$'\t' read -r netdev rdma mtu gid_index queue_size; do
  [[ -n "$netdev" && -n "$rdma" && -n "$mtu" && -n "$gid_index" && -n "$queue_size" ]] || {
    echo "invalid ring-local-fabric record for $NODE" >&2
    exit 2
  }
  bdf=$(basename "$(readlink -f "/sys/class/infiniband/$rdma/device")")
  device="pci/$bdf"
  current=$(devlink dev param show "$device" name hairpin_queue_size \
    | sed -n 's/.*cmode driverinit value \([0-9]*\).*/\1/p')
  [[ -n "$current" ]] || { echo "$NODE $netdev: cannot read hairpin_queue_size" >&2; exit 1; }
  if [[ "$current" == "$queue_size" ]]; then
    echo "  $NODE $netdev: hairpin_queue_size already $queue_size"
    continue
  fi

  devlink dev param set "$device" name hairpin_queue_size value "$queue_size" cmode driverinit
  devlink dev reload "$device" action driver_reinit >/dev/null
  for attempt in $(seq 1 30); do
    [[ "$(cat "/sys/class/net/$netdev/operstate" 2>/dev/null || true)" == up ]] \
      && ip -4 addr show "$netdev" | grep -q inet \
      && [[ "$(cat "/sys/class/net/$netdev/mtu" 2>/dev/null || true)" == "$mtu" ]] \
      && grep -q '4: ACTIVE' "/sys/class/infiniband/$rdma/ports/1/state" 2>/dev/null \
      && [[ "$(cat "/sys/class/infiniband/$rdma/ports/1/gids/$gid_index" 2>/dev/null || true)" != 0000:0000:0000:0000:0000:0000:0000:0000 ]] \
      && break
    sleep 2
  done
  [[ $attempt -lt 30 ]] || { echo "  $NODE $netdev: did not return after driver re-init" >&2; exit 2; }
  echo "  $NODE $netdev: hairpin_queue_size $current -> $queue_size, back after $((attempt * 2)) s"
done <<< "$PORTS"
