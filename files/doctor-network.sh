# Read-only host checks; never replace the HCAs used by serve.
doctor_hca_node() {
  local node="$1" root="$2" hcas="$3" hca state active="" bad=0 d
  for d in "$root"/class/infiniband/*; do
    state=$(cat "$d/ports/1/state" 2>/dev/null || true)
    if [[ "$state" == '4: ACTIVE' ]]; then active+="${active:+,}${d##*/}"; fi
  done
  echo "$node: ACTIVE HCAs: ${active:-none}"
  # NCCL permits hca:port entries and an exact-match '=' prefix. This check
  # deliberately checks port 1, as required by the Spark recipe.
  for hca in ${hcas//,/ }; do
    hca="${hca#=}"; hca="${hca%%:*}"
    state=$(cat "$root/class/infiniband/$hca/ports/1/state" 2>/dev/null || true)
    if [[ "$state" != '4: ACTIVE' ]]; then
      echo "[!] $node: HCA $hca port 1 is ${state:-missing}; expected 4: ACTIVE"
      bad=1
    fi
  done
  return "$bad"
}

doctor_socket_node() {
  local node="$1" root="$2" interfaces="$3" iface bad=0
  for iface in ${interfaces//,/ }; do
    iface="${iface#=}"
    if [[ ! -d "$root/class/net/$iface" ]]; then
      echo "[!] $node: socket interface $iface missing (GLOO_SOCKET_IFNAME/NCCL_SOCKET_IFNAME)"
      bad=1
    fi
  done
  [[ "$bad" != 0 ]] || echo "$node: socket interfaces OK"
  return "$bad"
}

doctor_network_checks() {
  local root="${DOCTOR_SYSFS_ROOT:-/sys}" hcas="$IB_HCA" kv host bad=0 command
  for kv in ${EXTRA_CONTAINER_ENV:-}; do
    case "$kv" in B12X_ROCE_HCA=*) hcas+=",${kv#*=}";; esac
  done
  doctor_hca_node head "$root" "$hcas" || bad=1
  doctor_socket_node head "$root" "$GLOO_SOCKET_IFNAME,$NCCL_SOCKET_IFNAME" || bad=1
  for host in "${WORKER_HOSTS[@]}"; do
    command="$(declare -f doctor_hca_node doctor_socket_node)
bad=0
doctor_hca_node $(printf '%q' "$host") $(printf '%q' "$root") $(printf '%q' "$hcas") || bad=1
doctor_socket_node $(printf '%q' "$host") $(printf '%q' "$root") $(printf '%q' "$GLOO_SOCKET_IFNAME,$NCCL_SOCKET_IFNAME") || bad=1
exit \$bad"
    remote_on "$host" "$command" || { warn "$host: network check failed"; bad=1; }
  done
  [[ "$bad" == 0 || "${DOCTOR_STRICT:-0}" != 1 ]]
}
