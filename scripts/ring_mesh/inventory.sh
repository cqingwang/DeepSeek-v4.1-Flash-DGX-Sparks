#!/usr/bin/env bash
# Read configured ConnectX fabric functions for plan.py; this script never changes host state.
set -euo pipefail
gid_index=${1:?usage: inventory.sh GID_INDEX NETDEV:HCA ...}
shift
[[ "$gid_index" =~ ^[0-9]+$ && $# -eq 4 ]] || {
  echo "inventory requires one GID index and four configured NETDEV:HCA pairs" >&2
  exit 2
}

first=1
printf '{"host":"%s","ports":{' "$(hostname)"
for pair in "$@"; do
  netdev=${pair%%:*}
  rdma=${pair#*:}
  [[ "$pair" == *:* && -n "$netdev" && -n "$rdma" ]] || {
    echo "invalid configured port pair: $pair" >&2
    exit 2
  }
  [[ -d "/sys/class/net/$netdev" && -d "/sys/class/infiniband/$rdma" ]] || {
    echo "missing configured interface/HCA $netdev/$rdma" >&2
    exit 1
  }
  bdf=$(basename "$(readlink -f "/sys/class/infiniband/$rdma/device")")
  ip=$(ip -4 -o addr show dev "$netdev" | awk '{print $4}' | head -1)
  param() {
    devlink dev param show "pci/$bdf" name "$1" 2>/dev/null \
      | sed -n 's/.*value \([0-9a-z]*\).*/\1/p' | head -1
  }
  gid=$(cat "/sys/class/infiniband/$rdma/ports/1/gids/$gid_index" 2>/dev/null || true)
  gid_netdev=$(cat "/sys/class/infiniband/$rdma/ports/1/gid_attrs/ndevs/$gid_index" 2>/dev/null || true)
  [[ $first == 1 ]] || printf ','
  first=0
  printf '"%s":{"rdma":"%s","ipv4":"%s","mac":"%s","mtu":%s,"roce_mtu":"%s","state":"%s","gid":"%s","gid_netdev":"%s","tc_offload":"%s","hairpin_num_queues":"%s","hairpin_queue_size":"%s","flow_steering_mode":"%s"}' \
    "$netdev" "$rdma" "$ip" "$(cat "/sys/class/net/$netdev/address")" \
    "$(cat "/sys/class/net/$netdev/mtu")" "$(cat "/sys/class/infiniband/$rdma/ports/1/active_mtu" 2>/dev/null || true)" \
    "$(cat "/sys/class/net/$netdev/operstate")" "$gid" "$gid_netdev" \
    "$(ethtool -k "$netdev" 2>/dev/null | awk '/hw-tc-offload/{print $2}')" \
    "$(param hairpin_num_queues)" "$(param hairpin_queue_size)" "$(param flow_steering_mode)"
done
printf '}}\n'
