# NCCL setup shared by the head and the generated worker Bash script.
# Ring transport/overlay integration derives from PR #3 by @Saolence and is
# extended here with current-main compatibility, all-rank preflight fixtures,
# safer quoting/path mapping, and local-weight validation.
# The underlying NCCL transport patch is from FujitsuPolycom/sparkring.
# Sourced by start.sh; no probing or side effects at import time.

switchless_ring_enabled() {
  [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == "1" ]]
}

nccl_extra_docker_env_has_key() {
  local wanted="$1" pair
  for pair in ${EXTRA_DOCKER_ENV:-}; do
    [[ "${pair%%=*}" == "$wanted" ]] && return 0
  done
  return 1
}

nccl_extra_docker_env_value() {
  local wanted="$1" pair
  for pair in ${EXTRA_DOCKER_ENV:-}; do
    if [[ "${pair%%=*}" == "$wanted" && "$pair" == *=* ]]; then
      printf '%s\n' "${pair#*=}"
      return 0
    fi
  done
  echo "Ring target config is missing $wanted in EXTRA_DOCKER_ENV" >&2
  return 1
}

nccl_validate_managed_tp4_transport_policy() {
  [[ "${DSV41_LAUNCHER:-}" == tp4 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  local key algo minimum maximum
  [[ "${NCCL_SKIP_TREE_CONNECT:-}" == 1 ]] || {
    echo "TP4 ring requires target-owned NCCL_SKIP_TREE_CONNECT=1" >&2
    return 1
  }
  for key in NCCL_ALGO NCCL_CROSS_NIC NCCL_IB_MERGE_NICS NCCL_IB_SUBNET_AWARE_ROUTING \
    NCCL_CUMEM_ENABLE NCCL_DEBUG NCCL_BUFFSIZE NCCL_LL128_BUFFSIZE NCCL_PROTO \
    NCCL_MAX_NCHANNELS NCCL_DEBUG_SUBSYS NCCL_MIN_NCHANNELS NCCL_P2P_LEVEL \
    NCCL_IB_TC NCCL_IB_PRESERVE_PCI_DOMAIN NCCL_IB_ROUTE_DIAGNOSTICS NCCL_IB_QPS_PER_CONNECTION; do
    nccl_extra_docker_env_has_key "$key" || {
      echo "TP4 ring requires target-owned $key in EXTRA_DOCKER_ENV" >&2
      return 1
    }
  done
  algo="$(nccl_extra_docker_env_value NCCL_ALGO)" || return 1
  [[ "${algo^^}" == RING ]] || { echo "TP4 ring requires target-owned NCCL_ALGO=Ring" >&2; return 1; }
  local subnet_aware
  subnet_aware="$(nccl_extra_docker_env_value NCCL_IB_SUBNET_AWARE_ROUTING)" || return 1
  [[ "$subnet_aware" == 1 ]] || { echo "TP4 ring requires target-owned NCCL_IB_SUBNET_AWARE_ROUTING=1" >&2; return 1; }
  minimum="$(nccl_extra_docker_env_value NCCL_MIN_NCHANNELS)" || return 1
  maximum="$(nccl_extra_docker_env_value NCCL_MAX_NCHANNELS)" || return 1
  [[ "$minimum" =~ ^[1-9][0-9]{0,2}$ && "$maximum" =~ ^[1-9][0-9]{0,2}$ ]] \
    && (( minimum <= maximum )) \
    || { echo 'TP4 ring requires target-owned positive NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS' >&2; return 1; }
}

nccl_ring_subnet_prefix() {
  local prefix="${NCCL_IB_SUBNET_PREFIX_LEN:-}"
  if [[ ! "$prefix" =~ ^[1-9][0-9]?$ ]] || (( prefix > 32 )); then
    echo "Ring requires a valid NCCL_IB_SUBNET_PREFIX_LEN from common.ring" >&2
    return 1
  fi
  printf '%s\n' "$prefix"
}

nccl_library() {
  local dir="$1" name
  for name in libnccl.so.2.30.7 libnccl.so.2; do
    if [[ -f "$dir/$name" && -r "$dir/$name" ]]; then
      printf '%s\n' "$dir/$name"
      return 0
    fi
  done
  return 1
}

_nccl_array_append() {
  local array_name="$1" quoted value
  shift
  for value in "$@"; do
    printf -v quoted '%q' "$value"
    eval "$array_name+=($quoted)"
  done
}

nccl_mount_args() {
  local array_name="$1"
  local library
  if library=$(nccl_library "$NCCL_HOST_DIR"); then
    if [[ "$NCCL_OVERLAY_PIP" == 1 ]]; then
      _nccl_array_append "$array_name" -v "$library:$NCCL_PIP_SO:ro"
    else
      _nccl_array_append "$array_name" -v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro" \
        -e "LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR"
    fi
  elif [[ "$NCCL_OVERLAY_PIP" == 1 || "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2
    return 1
  fi
}

dual_pci_domain_args() {
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] || return 0
  local array_name="$1" key value
  local -a settings=(
    NCCL_IB_EXTENDED_IPV4_GIDS=1
    "NCCL_IB_PRESERVE_PCI_DOMAIN=${NCCL_IB_PRESERVE_PCI_DOMAIN:-1}"
    "NCCL_IB_ROUTE_DIAGNOSTICS=${NCCL_IB_ROUTE_DIAGNOSTICS:-1}"
    "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}"
  )
  for value in "${settings[@]}"; do
    key="${value%%=*}"
    nccl_extra_docker_env_has_key "$key" || _nccl_array_append "$array_name" -e "$value"
  done
}

dual_pci_domain_env_string() {
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] || return 0
  local key value first=1
  local -a settings=(
    NCCL_IB_EXTENDED_IPV4_GIDS=1
    "NCCL_IB_PRESERVE_PCI_DOMAIN=${NCCL_IB_PRESERVE_PCI_DOMAIN:-1}"
    "NCCL_IB_ROUTE_DIAGNOSTICS=${NCCL_IB_ROUTE_DIAGNOSTICS:-1}"
    "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}"
  )
  for value in "${settings[@]}"; do
    key="${value%%=*}"
    nccl_extra_docker_env_has_key "$key" && continue
    if [[ "$first" == 1 ]]; then
      printf -- '-e %q' "$value"
      first=0
    else
      printf -- ' -e %q' "$value"
    fi
  done
}

nccl_gid_publication_check() {
  switchless_ring_enabled || return 0
  local -a hcas=()
  IFS=, read -r -a hcas <<<"${IB_HCA#=}"
  (( ${#hcas[@]} > 2 )) || return 0
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] && return 0
  echo "warning: IB_HCA lists ${#hcas[@]} devices (${hcas[*]}) but listener GID publication is capped at 2" >&2
  echo "warning: set NCCL_IB_EXTENDED_IPV4_GIDS=1 with a dual-PCI-domain NCCL build" >&2
}

switchless_ring_args() {
  switchless_ring_enabled || return 0
  local array_name="$1" prefix skip_tree
  nccl_validate_managed_tp4_transport_policy || return 1
  prefix="$(nccl_ring_subnet_prefix)" || return 1
  skip_tree="${NCCL_SKIP_TREE_CONNECT:-1}"
  _nccl_array_append "$array_name" -e NCCL_SWITCHLESS_RING_ONLY=1
  _nccl_array_append "$array_name" -e "NCCL_SKIP_TREE_CONNECT=$skip_tree" \
    -e "NCCL_IB_SUBNET_PREFIX_LEN=$prefix"
  if [[ "${DSV41_LAUNCHER:-}" != tp4 ]]; then
    nccl_extra_docker_env_has_key NCCL_ALGO \
      || _nccl_array_append "$array_name" -e "NCCL_ALGO=${NCCL_ALGO:-Ring}"
    nccl_extra_docker_env_has_key NCCL_MIN_NCHANNELS \
      || _nccl_array_append "$array_name" -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}"
    nccl_extra_docker_env_has_key NCCL_P2P_LEVEL \
      || _nccl_array_append "$array_name" -e "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}"
  fi
  dual_pci_domain_args "$array_name"
}

switchless_ring_env_string() {
  switchless_ring_enabled || return 0
  local prefix value skip_tree
  nccl_validate_managed_tp4_transport_policy || return 1
  local -a ring_values=(NCCL_SWITCHLESS_RING_ONLY=1)
  prefix="$(nccl_ring_subnet_prefix)" || return 1
  skip_tree="${NCCL_SKIP_TREE_CONNECT:-1}"
  ring_values+=("NCCL_SKIP_TREE_CONNECT=$skip_tree" "NCCL_IB_SUBNET_PREFIX_LEN=$prefix")
  if [[ "${DSV41_LAUNCHER:-}" != tp4 ]]; then
    nccl_extra_docker_env_has_key NCCL_ALGO \
      || ring_values+=("NCCL_ALGO=${NCCL_ALGO:-Ring}")
    nccl_extra_docker_env_has_key NCCL_MIN_NCHANNELS \
      || ring_values+=("NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}")
    nccl_extra_docker_env_has_key NCCL_P2P_LEVEL \
      || ring_values+=("NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}")
  fi
  printf -- '-e %q' "${ring_values[0]}"
  for value in "${ring_values[@]:1}"; do
    printf -- ' -e %q' "$value"
  done
  local extra
  extra=$(dual_pci_domain_env_string)
  [[ -z "$extra" ]] || printf ' %s' "$extra"
}

nccl_validate_config() {
  local flag
  for flag in NCCL_SWITCHLESS_RING_ONLY NCCL_OVERLAY_PIP DSV41_SERIAL_WEIGHT_LOAD; do
    case "${!flag:-0}" in 0|1) ;; *) echo "$flag must be 0 or 1" >&2; return 1;; esac
  done
  if [[ -n "${NCCL_WORKER_DIR:-}" && "$NCCL_WORKER_DIR" != /* ]]; then
    echo 'NCCL_WORKER_DIR must be absolute' >&2
    return 1
  fi
  [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  nccl_ring_subnet_prefix >/dev/null || return 1
  nccl_validate_managed_tp4_transport_policy || return 1
  # 环网覆盖 TP=4；EP=1/2 只允许由 start-tp4.sh（导出 DSV41_LAUNCHER=tp4）使用，
  # 普通三节点启动器不能静默接受 TP4 专家并行配置。
  local ep_ok=0 ep_message="EP_SIZE=4"
  if [[ "$EP_SIZE" == 4 ]]; then
    ep_ok=1
  elif [[ "${DSV41_LAUNCHER:-}" == tp4 && "$EP_SIZE" =~ ^[12]$ ]]; then
    ep_ok=1
    ep_message="EP_SIZE 1, 2 or 4"
  fi
  if [[ "$NNODES" != 4 || "$TP_SIZE" != 4 || "$ep_ok" != 1 ||
        "$NCCL_OVERLAY_PIP" != 1 || "$NCCL_NET" != IB || "$NCCL_IB_DISABLE" != 0 ]]; then
    echo "Ring requires NNODES=TP_SIZE=4, $ep_message, NCCL_OVERLAY_PIP=1, NCCL_ALGO=Ring, NCCL_NET=IB, NCCL_IB_DISABLE=0 and subnet-aware routing=1" >&2
    return 1
  fi
  if [[ "${DSV41_LAUNCHER:-}" != tp4 ]]; then
    local minimum="${NCCL_MIN_NCHANNELS:-4}" maximum="${NCCL_MAX_NCHANNELS:-32}"
    if [[ ! "$minimum" =~ ^[1-9][0-9]{0,2}$ || ! "$maximum" =~ ^[1-9][0-9]{0,2}$ ]] ||
       (( minimum > maximum )); then
      echo 'Ring requires positive NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS' >&2
      return 1
    fi
  fi
}

# MANAGEMENT IPs need not appear in the CX7 GID table. Find a common, nonzero
# IPv4 RoCE v2 index on all explicitly selected ports, or validate the override.
# The optional sysfs root is used by host-side fixtures.
ring_gid_index() {
  local sysfs="${1:-/sys/class/infiniband}" entry hca port state base g idx valid gid
  local -a entries=() ports=() candidates=()
  IFS=, read -r -a entries <<<"${IB_HCA#=}"
  for entry in "${entries[@]}"; do
    if [[ ! "$entry" =~ ^[a-zA-Z0-9_]+(:[1-9][0-9]*)?$ ]]; then
      echo 'Ring IB_HCA must list exact HCA names, optionally with :port' >&2
      return 1
    fi
    hca="${entry%%:*}"; port=1
    [[ "$entry" != *:* ]] || port="${entry##*:}"
    base="$sysfs/$hca/ports/$port"
    state=$(cat "$base/state" 2>/dev/null) || state=''
    if [[ "$state" != '4: ACTIVE' ]]; then
      echo "Ring: $hca:$port is not ACTIVE" >&2
      return 1
    fi
    ports+=("$base")
  done
  [[ ${#ports[@]} -gt 0 ]] || { echo 'Ring: IB_HCA is empty' >&2; return 1; }
  if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
    [[ "$NCCL_IB_GID_INDEX" =~ ^[0-9]+$ ]] || return 1
    candidates=("$NCCL_IB_GID_INDEX")
  else
    for g in "${ports[0]}"/gids/*; do candidates+=("${g##*/}"); done
  fi
  for idx in "${candidates[@]}"; do
    valid=1
    for base in "${ports[@]}"; do
      if [[ "$(cat "$base/gid_attrs/types/$idx" 2>/dev/null)" != 'RoCE v2' ]]; then
        valid=0; break
      fi
      gid=$(cat "$base/gids/$idx" 2>/dev/null) || gid=''
      if [[ ! "$gid" =~ ^0000:0000:0000:0000:0000:ffff:[[:xdigit:]]{4}:[[:xdigit:]]{4}$ ||
            "$gid" == *:0000:0000 ]]; then
        valid=0; break
      fi
    done
    if [[ "$valid" == 1 ]]; then printf '%s\n' "$idx"; return 0; fi
  done
  echo "Ring: no common IPv4 RoCE v2 GID on $IB_HCA (override: ${NCCL_IB_GID_INDEX:-auto})" >&2
  return 1
}

nccl_preflight() {
  local library
  switchless_ring_enabled || {
    [[ "${NCCL_OVERLAY_PIP:-0}" == 1 ]] || return 0
  }
  if [[ "${NCCL_OVERLAY_PIP:-0}" == 1 || "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    library=$(nccl_library "$NCCL_HOST_DIR") || {
      echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2; return 1;
    }
    if [[ "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] && ! grep -qa SWITCHLESS_RING_ONLY "$library"; then
      echo "NCCL: $library lacks SWITCHLESS_RING_ONLY; install the patched build" >&2
      return 1
    fi
    if [[ -n "${NCCL_SHA256:-}" ]]; then
      if [[ ! "$NCCL_SHA256" =~ ^[a-f0-9]{64}$ ]]; then
        echo "NCCL: NCCL_SHA256 must be a 64-character lowercase SHA-256" >&2
        return 1
      fi
      local actual_sha256
      actual_sha256="$(sha256sum "$library" | awk '{print $1}')" || {
        echo "NCCL: unable to hash $library" >&2
        return 1
      }
      if [[ "$actual_sha256" != "$NCCL_SHA256" ]]; then
        echo "NCCL: NCCL_SHA256 mismatch for $library: expected $NCCL_SHA256 got $actual_sha256" >&2
        return 1
      fi
    fi
    [[ "$NCCL_PIP_SO" == /* && "$NCCL_PIP_SO" != *:* && "$NCCL_HOST_DIR" == /* && "$NCCL_HOST_DIR" != *:* ]] || {
      echo 'NCCL overlay requires an absolute container path and paths without colons' >&2; return 1;
    }
    docker run --rm --pull=never --network none --entrypoint /bin/sh "$IMAGE" \
      -c 'test -f "$1"' sh "$NCCL_PIP_SO" || {
      echo "NCCL: $IMAGE lacks the pip library at $NCCL_PIP_SO" >&2; return 1;
    }
  fi
  if switchless_ring_enabled; then ring_gid_index || return 1; fi
  return 0
}

# Home-relative head paths follow the worker's HOME, including custom subdirs.
# An explicit NCCL_WORKER_DIR (absolute) overrides that mapping on every worker.
nccl_worker_settings() {
  if [[ -n "${NCCL_WORKER_DIR:-}" ]]; then
    printf 'NCCL_HOST_DIR=%q\n' "$NCCL_WORKER_DIR"
  elif [[ "$NCCL_HOST_DIR" == "$HOME/"* ]]; then
    printf 'NCCL_HOST_DIR="$HOME"/%q\n' "${NCCL_HOST_DIR#"$HOME/"}"
  else
    printf 'NCCL_HOST_DIR=%q\n' "$NCCL_HOST_DIR"
  fi
  local key
  for key in NCCL_OVERLAY_PIP NCCL_PIP_SO NCCL_SHA256 NCCL_CONTAINER_DIR NCCL_SWITCHLESS_RING_ONLY IB_HCA IMAGE NCCL_IB_GID_INDEX; do
    printf '%s=%q\n' "$key" "${!key:-}"
  done
  declare -f switchless_ring_enabled nccl_library nccl_mount_args ring_gid_index \
    nccl_preflight dual_pci_domain_args dual_pci_domain_env_string nccl_gid_publication_check
}
