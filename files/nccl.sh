# Switchless-ring NCCL setup, shared by the head and the generated worker Bash script.
#
# A four-node CX7 ring wired without a switch (a-b-c-d-a, one DAC per adjacency per
# rail) cannot boot this stack on stock settings: NCCL builds a *tree* in addition to
# the ring, the tree wants a direct fabric path between opposite nodes (rank0 <-> rank2)
# which a four-node ring does not have, and RoCE queue pairs do not follow IP routing,
# so the tree never connects and `ncclCommInitRank` dies with an unhandled system error.
#
# Turn this on with NCCL_SWITCHLESS_RING_ONLY=1 (default 0: every other deployment is
# byte-for-byte unchanged). It then:
#   * injects the ring-only NCCL environment into the head AND every worker,
#   * mounts the patched NCCL over the image's pip-installed library, instead of putting
#     it on LD_LIBRARY_PATH (two visible NCCL runtimes make DeepEP's check_nccl_so()
#     abort before NCCL is ever initialised),
#   * validates the ring configuration and every rank's HCA/GID before any container is
#     replaced, and refuses stock/unpatched NCCL by checking for the SWITCHLESS_RING_ONLY
#     marker in the library.
#
# The NCCL transport patch itself is not ours: it is from FujitsuPolycom/sparkring
# (the `switchless-cycle` / `skip-tree-pat` patches). The launcher integration,
# the overlay-mount strategy, the per-rank preflight and this file derive from
# MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks #3 by @Saolence, carried forward in #19
# and rebased here onto this profile.
#
# Sourced by start.sh; no probing or side effects at source time.

# The switch. Everything below is a no-op unless this is 1.
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

nccl_extra_container_env_count() {
  local wanted="$1" pair count=0
  for pair in ${EXTRA_CONTAINER_ENV:-}; do
    [[ "${pair%%=*}" == "$wanted" && "$pair" == *=* ]] || continue
    count=$((count + 1))
  done
  printf '%s\n' "$count"
}

nccl_extra_container_env_value() {
  local wanted="$1" pair
  for pair in ${EXTRA_CONTAINER_ENV:-}; do
    if [[ "${pair%%=*}" == "$wanted" && "$pair" == *=* ]]; then
      printf '%s\n' "${pair#*=}"
      return 0
    fi
  done
  echo "Ring target config is missing $wanted in EXTRA_CONTAINER_ENV" >&2
  return 1
}

# The parent config is the sole source of physical HCA/GID/peer routes. Check
# that its rendered payload is complete and internally consistent without
# embedding any site's device names, addresses, GIDs, or route map here.
validate_tp4_ring_policy() {
  [[ "${DSV41_LAUNCHER:-}" == tp4 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  [[ "${TARGET_ENGINE_ID:-}" == deepseek_v4.1_flash_switchless ]] || {
    echo "managed TP4 ring requires the managed target ID TARGET_ENGINE_ID=deepseek_v4.1_flash_switchless" >&2
    return 1
  }

  local key count map_hcas map_gid next_gid peer_maps payload pair rank
  for rank in 0 1 2 3; do
    key="PEER_HCA_RANK$rank"
    [[ -z "${!key:-}" ]] || {
      echo "managed TP4 NCCL does not consume $key; use the parent-derived B12X peer map" >&2
      return 1
    }
  done
  for payload in "${EXTRA_DOCKER_ENV:-}" "${EXTRA_CONTAINER_ENV:-}"; do
    for pair in $payload; do
      key="${pair%%=*}"
      case "$key" in
        PEER_HCA_RANK[0-3]|NCCL_IB_PEER_HCA|NCCL_IB_ADDR_RANGE)
          echo "managed TP4 NCCL does not consume $key; use the parent-derived B12X peer map" >&2
          return 1
          ;;
      esac
    done
  done

  for key in B12X_ROCE_HCA B12X_ROCE_GID_INDEX B12X_NEXT_ROCE_GID_INDEX B12X_ROCE_PEER_HCA_MAPS; do
    count="$(nccl_extra_container_env_count "$key")"
    [[ "$count" == 1 ]] || {
      echo "managed TP4 ring requires exactly one config.yaml-derived $key in EXTRA_CONTAINER_ENV" >&2
      return 1
    }
  done

  map_hcas="$(nccl_extra_container_env_value B12X_ROCE_HCA)" || return 1
  map_gid="$(nccl_extra_container_env_value B12X_ROCE_GID_INDEX)" || return 1
  next_gid="$(nccl_extra_container_env_value B12X_NEXT_ROCE_GID_INDEX)" || return 1
  peer_maps="$(nccl_extra_container_env_value B12X_ROCE_PEER_HCA_MAPS)" || return 1

  [[ -n "${IB_HCA:-}" && "$map_hcas" == "${IB_HCA#=}" ]] || {
    echo "B12X HCA order must match parent-managed IB_HCA" >&2
    return 1
  }
  [[ "$map_gid" =~ ^[0-9]+$ && "$next_gid" =~ ^[0-9]+$ && "$map_gid" == "$next_gid" ]] || {
    echo "B12X RoCE GID indexes must use the same non-negative config.yaml GID index" >&2
    return 1
  }

  local -a hcas=() rank_maps=() rank_entries=() seen_peers=()
  local entry peer_rank local_hca peer_hca peer
  IFS=, read -r -a hcas <<<"$map_hcas"
  (( ${#hcas[@]} >= 2 )) || {
    echo "B12X HCA map must contain at least two parent-managed devices" >&2
    return 1
  }
  IFS=';' read -r -a rank_maps <<<"$peer_maps"
  [[ ${#rank_maps[@]} -eq 4 ]] || {
    echo "B12X peer map must contain four config.yaml-derived rank maps" >&2
    return 1
  }

  for rank in 0 1 2 3; do
    IFS=, read -r -a rank_entries <<<"${rank_maps[$rank]}"
    [[ ${#rank_entries[@]} -eq 3 ]] || {
      echo "B12X rank $rank map must contain exactly three peer paths" >&2
      return 1
    }
    seen_peers=(0 0 0 0)
    for entry in "${rank_entries[@]}"; do
      [[ "$entry" =~ ^([0-3])=([0-9]+)/([0-9]+)$ ]] || {
        echo "B12X rank $rank map has an invalid peer path: $entry" >&2
        return 1
      }
      peer_rank="${BASH_REMATCH[1]}"
      local_hca="${BASH_REMATCH[2]}"
      peer_hca="${BASH_REMATCH[3]}"
      [[ "$peer_rank" != "$rank" && "${seen_peers[$peer_rank]}" == 0 ]] || {
        echo "B12X rank $rank map has a self or duplicate peer: $peer_rank" >&2
        return 1
      }
      (( 10#$local_hca < ${#hcas[@]} && 10#$peer_hca < ${#hcas[@]} )) || {
        echo "B12X rank $rank map refers to an HCA outside the managed HCA list" >&2
        return 1
      }
      seen_peers[$peer_rank]=1
    done
    for peer in 0 1 2 3; do
      [[ "$peer" == "$rank" || "${seen_peers[$peer]}" == 1 ]] || {
        echo "B12X rank $rank map is missing peer $peer" >&2
        return 1
      }
    done
  done
}

# Managed TP4 values belong to the parent target query; never turn a missing
# managed transport value into a launcher default on the ring path.
nccl_validate_managed_tp4_transport_policy() {
  [[ "${DSV41_LAUNCHER:-}" == tp4 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  local key algo minimum maximum subnet_aware gid_force
  for key in NCCL_NET NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_SHM_DISABLE \
    NCCL_IB_EXTENDED_IPV4_GIDS NCCL_IB_GID_INDEX_FORCE; do
    [[ -n "${!key:-}" ]] || {
      echo "TP4 ring requires parent-managed $key" >&2
      return 1
    }
  done
  gid_force="$NCCL_IB_GID_INDEX_FORCE"
  [[ "$gid_force" == -1 || "$gid_force" =~ ^[0-9]+$ ]] || {
    echo "TP4 ring requires NCCL_IB_GID_INDEX_FORCE=-1 or a non-negative GID index" >&2
    return 1
  }
  if [[ "$gid_force" == -1 ]]; then
    [[ -z "${NCCL_IB_GID_INDEX:-}" ]] || {
      echo "NCCL_IB_GID_INDEX conflicts with parent-managed NCCL_IB_GID_INDEX_FORCE=-1" >&2
      return 1
    }
  elif [[ -n "${NCCL_IB_GID_INDEX:-}" && "$NCCL_IB_GID_INDEX" != "$gid_force" ]]; then
    echo "NCCL_IB_GID_INDEX conflicts with parent-managed NCCL_IB_GID_INDEX_FORCE=$gid_force" >&2
    return 1
  fi
  [[ "$NCCL_NET" == IB && "$NCCL_IB_DISABLE" == 0 &&
     "$NCCL_P2P_DISABLE" == 1 && "$NCCL_SHM_DISABLE" == 1 &&
     "$NCCL_IB_EXTENDED_IPV4_GIDS" == 1 ]] || {
    echo "TP4 ring requires NCCL_NET=IB, NCCL_IB_DISABLE=0, NCCL_P2P_DISABLE=1, NCCL_SHM_DISABLE=1, and NCCL_IB_EXTENDED_IPV4_GIDS=1" >&2
    return 1
  }
  [[ "${NCCL_PIP_SO:-}" == /* && "${NCCL_PIP_SO:-}" != *:* ]] || {
    echo "TP4 ring requires an absolute parent-managed NCCL_PIP_SO path" >&2
    return 1
  }
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
  case "$algo" in
    RING|Ring|ring) ;;
    *) echo "TP4 ring requires target-owned NCCL_ALGO=Ring" >&2; return 1 ;;
  esac
  subnet_aware="$(nccl_extra_docker_env_value NCCL_IB_SUBNET_AWARE_ROUTING)" || return 1
  [[ "$subnet_aware" == 1 ]] || {
    echo "TP4 ring requires target-owned NCCL_IB_SUBNET_AWARE_ROUTING=1" >&2
    return 1
  }
  minimum="$(nccl_extra_docker_env_value NCCL_MIN_NCHANNELS)" || return 1
  maximum="$(nccl_extra_docker_env_value NCCL_MAX_NCHANNELS)" || return 1
  [[ "$minimum" =~ ^[1-9][0-9]{0,2}$ && "$maximum" =~ ^[1-9][0-9]{0,2}$ ]] \
    && (( minimum <= maximum )) \
    || { echo 'TP4 ring requires target-owned positive NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS' >&2; return 1; }
}

nccl_apply_managed_tp4_gid_policy() {
  [[ "${DSV41_LAUNCHER:-}" == tp4 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] || return 0
  local forced="${NCCL_IB_GID_INDEX_FORCE:-}"
  [[ "$forced" == -1 || "$forced" =~ ^[0-9]+$ ]] || {
    echo "managed TP4 ring requires NCCL_IB_GID_INDEX_FORCE=-1 or a non-negative GID index" >&2
    return 1
  }
  if [[ "$forced" == -1 ]]; then
    [[ -z "${NCCL_IB_GID_INDEX:-}" ]] || {
      echo "NCCL_IB_GID_INDEX conflicts with parent-managed NCCL_IB_GID_INDEX_FORCE=-1" >&2
      return 1
    }
    unset NCCL_IB_GID_INDEX
    NCCL_IB_GID_AUTO=1
    return 0
  fi
  [[ -z "${NCCL_IB_GID_INDEX:-}" || "$NCCL_IB_GID_INDEX" == "$forced" ]] || {
    echo "NCCL_IB_GID_INDEX conflicts with parent-managed NCCL_IB_GID_INDEX_FORCE=$forced" >&2
    return 1
  }
  NCCL_IB_GID_INDEX="$forced"
  NCCL_IB_GID_AUTO=0
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

nccl_file_sha256() {
  local file="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$file" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$file" | awk '{print $1}'
  else
    echo "NCCL: neither sha256sum nor shasum is available to verify $file" >&2
    return 1
  fi
}

# Mount the patched NCCL over the image's pip library when the ring is on, else keep
# this profile's existing LD_LIBRARY_PATH mount.
nccl_mount_args() {
  local -n mounts=$1
  local library
  if library=$(nccl_library "$NCCL_HOST_DIR"); then
    if [[ "$NCCL_OVERLAY_PIP" == 1 ]]; then
      mounts+=(-v "$library:$NCCL_PIP_SO:ro")
    else
      mounts+=(-v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro"
               -e "LD_LIBRARY_PATH=$NCCL_CONTAINER_DIR")
    fi
  elif [[ "$NCCL_OVERLAY_PIP" == 1 || "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]]; then
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2
    return 1
  fi
}

# Ring-only environment, appended to a docker-argument array (head side).
# The switchless-cycle patch publishes at most two listener GIDs per rank
# (`gidSlot < 2` in net_ib/connect.cc). Every device past the second in IB_HCA is
# therefore never advertised, so no peer can match its subnet and those ports carry
# zero bytes -- silently, because NCCL logs nothing about it at INFO. The cumulative
# dual-PCI-domain patch raises the bound to four gated on
# NCCL_IB_EXTENDED_IPV4_GIDS, which also matters when a second ConnectX-7 sits in a
# different PCI root domain (0002: vs 0000: on this hardware).
dual_pci_domain_args() {
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] || return 0
  # The managed TP4 launcher injects the GID policy from the target field and
  # the companion PCI settings from extra_docker_env; this helper must not duplicate them.
  [[ "${DSV41_LAUNCHER:-}" == tp4 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] && return 0
  local -n dd_args=$1
  dd_args+=(-e NCCL_IB_EXTENDED_IPV4_GIDS=1
    -e "NCCL_IB_PRESERVE_PCI_DOMAIN=${NCCL_IB_PRESERVE_PCI_DOMAIN:-1}"
    -e "NCCL_IB_ROUTE_DIAGNOSTICS=${NCCL_IB_ROUTE_DIAGNOSTICS:-1}"
    -e "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}")
}

dual_pci_domain_env_string() {
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] || return 0
  [[ "${DSV41_LAUNCHER:-}" == tp4 && "${NCCL_SWITCHLESS_RING_ONLY:-0}" == 1 ]] && return 0
  printf -- '-e NCCL_IB_EXTENDED_IPV4_GIDS=1 -e %q -e %q -e %q' \
    "NCCL_IB_PRESERVE_PCI_DOMAIN=${NCCL_IB_PRESERVE_PCI_DOMAIN:-1}" \
    "NCCL_IB_ROUTE_DIAGNOSTICS=${NCCL_IB_ROUTE_DIAGNOSTICS:-1}" \
    "NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-1}"
}

# Silent otherwise: NCCL accepts any number of devices in IB_HCA and simply stops
# advertising after the second, so half a dual-plane board stays idle with no error.
nccl_gid_publication_check() {
  switchless_ring_enabled || return 0
  local -a hcas=()
  IFS=, read -r -a hcas <<<"${IB_HCA#=}"
  (( ${#hcas[@]} > 2 )) || return 0
  [[ "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 ]] && return 0
  echo "warning: IB_HCA lists ${#hcas[@]} devices but listener GID publication is capped at 2" >&2
  echo "warning: only ${hcas[0]} and ${hcas[1]} can be selected by a peer; the rest stay at zero" >&2
  echo "warning: set NCCL_IB_EXTENDED_IPV4_GIDS=1 with a dual-PCI-domain NCCL build" >&2
  return 0
}

switchless_ring_args() {
  switchless_ring_enabled || return 0
  local subnet_prefix
  local -n ring_args=$1
  subnet_prefix="$(nccl_ring_subnet_prefix)" || return 1
  ring_args+=(-e NCCL_SWITCHLESS_RING_ONLY=1)
  if [[ "${DSV41_LAUNCHER:-}" == tp4 ]]; then
    nccl_extra_docker_env_has_key NCCL_ALGO \
      || ring_args+=(-e "NCCL_ALGO=${NCCL_ALGO:-Ring}")
    ring_args+=(-e "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}"
      -e "NCCL_IB_SUBNET_PREFIX_LEN=$subnet_prefix")
    nccl_extra_docker_env_has_key NCCL_MIN_NCHANNELS \
      || ring_args+=(-e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}")
    nccl_extra_docker_env_has_key NCCL_P2P_LEVEL \
      || ring_args+=(-e "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}")
  else
    ring_args+=(-e "NCCL_ALGO=${NCCL_ALGO:-Ring}"
      -e "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}"
      -e "NCCL_IB_SUBNET_PREFIX_LEN=$subnet_prefix"
      -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}"
      -e "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}")
  fi
  dual_pci_domain_args ring_args
}

# Same environment as one shell-quoted `-e KEY=VALUE` string, for the worker heredoc.
switchless_ring_env_string() {
  switchless_ring_enabled || return 0
  local subnet_prefix
  subnet_prefix="$(nccl_ring_subnet_prefix)" || return 1
  printf -- '-e NCCL_SWITCHLESS_RING_ONLY=1'
  if [[ "${DSV41_LAUNCHER:-}" == tp4 ]]; then
    nccl_extra_docker_env_has_key NCCL_ALGO \
      || printf ' -e %q' "NCCL_ALGO=${NCCL_ALGO:-Ring}"
    printf ' -e %q -e %q' \
      "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}" \
      "NCCL_IB_SUBNET_PREFIX_LEN=$subnet_prefix"
    nccl_extra_docker_env_has_key NCCL_MIN_NCHANNELS \
      || printf ' -e %q' "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}"
    nccl_extra_docker_env_has_key NCCL_P2P_LEVEL \
      || printf ' -e %q' "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}"
  else
    printf ' -e %q -e %q -e %q -e %q -e %q' \
      "NCCL_ALGO=${NCCL_ALGO:-Ring}" \
      "NCCL_SKIP_TREE_CONNECT=${NCCL_SKIP_TREE_CONNECT:-1}" \
      "NCCL_IB_SUBNET_PREFIX_LEN=$subnet_prefix" \
      "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-4}" \
      "NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}"
  fi
  local dd
  dd=$(dual_pci_domain_env_string)
  [[ -z "$dd" ]] || printf ' %s' "$dd"
}

nccl_validate_config() {
  local flag
  for flag in NCCL_SWITCHLESS_RING_ONLY NCCL_OVERLAY_PIP; do
    case "${!flag:-0}" in 0|1) ;; *) echo "$flag must be 0 or 1" >&2; return 1;; esac
  done
  if [[ -n "${NCCL_WORKER_DIR:-}" && "$NCCL_WORKER_DIR" != /* ]]; then
    echo 'NCCL_WORKER_DIR must be absolute' >&2
    return 1
  fi
  switchless_ring_enabled || return 0
  nccl_apply_managed_tp4_gid_policy || return 1
  nccl_ring_subnet_prefix >/dev/null || return 1
  validate_tp4_ring_policy || return 1
  nccl_validate_managed_tp4_transport_policy || return 1
  if [[ "${DSV41_LAUNCHER:-}" == tp4 ]]; then
    [[ "${NCCL_HOST_DIR:-}" == /* ]] || {
      echo 'managed TP4 ring requires an absolute NCCL_HOST_DIR' >&2
      return 1
    }
    [[ "${NCCL_SHA256:-}" =~ ^[[:xdigit:]]{64}$ ]] || {
      echo 'managed TP4 ring requires a 64-digit NCCL_SHA256 from the parent config' >&2
      return 1
    }
  fi
  if [[ "$NNODES" != 4 || "$TP_SIZE" != 4 ||
        ! "$EP_SIZE" =~ ^[1-9][0-9]*$ || "$EP_SIZE" -gt "$TP_SIZE" ||
        "$NCCL_OVERLAY_PIP" != 1 || "${NCCL_ALGO:-Ring}" != Ring ||
        "$NCCL_NET" != IB || "$NCCL_IB_DISABLE" != 0 ||
        "${NCCL_IB_SUBNET_AWARE_ROUTING:-1}" != 1 ]]; then
    echo 'Ring requires NNODES=TP_SIZE=4, 1 <= EP_SIZE <= TP_SIZE (EP_SIZE=TP_SIZE is the validated default), NCCL_OVERLAY_PIP=1, NCCL_ALGO=Ring, NCCL_NET=IB, NCCL_IB_DISABLE=0 and subnet-aware routing=1' >&2
    return 1
  fi
  local minimum="${NCCL_MIN_NCHANNELS:-4}" maximum="${NCCL_MAX_NCHANNELS:-32}"
  if [[ ! "$minimum" =~ ^[1-9][0-9]{0,2}$ || ! "$maximum" =~ ^[1-9][0-9]{0,2}$ ]] ||
     (( minimum > maximum )); then
    echo 'Ring requires positive NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS' >&2
    return 1
  fi
}

# MANAGEMENT IPs need not appear in the CX7 GID table. Find a common, nonzero
# IPv4 RoCE v2 index on all explicitly selected ports, or validate the override.
# With NCCL_IB_GID_AUTO=1 and no explicit index, pin a GID matching the optional
# host IP when present; otherwise validate each port independently and let NCCL
# select a GID per device. The sysfs root and host IP are test-fixture inputs.
ring_gid_index() {
  local sysfs="${1:-/sys/class/infiniband}" ip="${2:-}" entry hca port state base g idx valid gid hex
  local -a entries=() raw=() ports=() candidates=()
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
  if [[ -z "${NCCL_IB_GID_INDEX:-}" && "${NCCL_IB_GID_AUTO:-0}" == 1 ]]; then
    if [[ "$ip" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]]; then
      hex=$(printf '%02x%02x:%02x%02x' ${ip//./ })
      # Match the launcher's address probe: raw IB_HCA entries, port 1 only.
      IFS=, read -r -a raw <<<"$IB_HCA"
      for entry in "${raw[@]}"; do
        [[ -d "$sysfs/$entry" ]] || continue
        base="$sysfs/$entry/ports/1"
        for g in "$base"/gids/*; do
          idx="${g##*/}"
          [[ "$(cat "$base/gid_attrs/types/$idx" 2>/dev/null)" == 'RoCE v2' ]] || continue
          [[ "$(cat "$g" 2>/dev/null)" == *ffff:"$hex" ]] || continue
          NCCL_IB_GID_INDEX="$idx" NCCL_IB_GID_AUTO=0 ring_gid_index "$sysfs"
          return
        done
      done
    fi
    for entry in "${entries[@]}"; do
      IB_HCA="$entry" NCCL_IB_GID_AUTO=0 ring_gid_index "$sysfs" >/dev/null || return 1
    done
    printf 'auto\n'
    return 0
  fi
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

# Head-side ring preflight. Silent (and free) unless the switch is on. The optional
# argument is the host address the launcher detects the GID index from.
nccl_preflight() {
  local library actual_sha
  if ! switchless_ring_enabled && [[ "${NCCL_OVERLAY_PIP:-0}" != 1 ]]; then
    return 0
  fi
  library=$(nccl_library "$NCCL_HOST_DIR") || {
    echo "NCCL: no readable library in $NCCL_HOST_DIR" >&2; return 1;
  }
  if [[ -n "${NCCL_SHA256:-}" ]]; then
    actual_sha="$(nccl_file_sha256 "$library")" || return 1
    [[ "$actual_sha" == "$NCCL_SHA256" ]] || {
      echo "NCCL_SHA256 mismatch: expected=$NCCL_SHA256 actual=$actual_sha ($library)" >&2
      return 1
    }
  fi
  if switchless_ring_enabled && ! grep -qa SWITCHLESS_RING_ONLY "$library"; then
    echo "NCCL: $library lacks SWITCHLESS_RING_ONLY; install the patched build" >&2
    return 1
  fi
  if [[ "$NCCL_OVERLAY_PIP" == 1 ]]; then
    [[ "$NCCL_PIP_SO" == /* && "$NCCL_PIP_SO" != *:* && "$NCCL_HOST_DIR" == /* && "$NCCL_HOST_DIR" != *:* ]] || {
      echo 'NCCL overlay requires an absolute container path and paths without colons' >&2; return 1;
    }
    docker image inspect "$IMAGE" >/dev/null 2>&1 || {
      echo "NCCL: image $IMAGE is not present locally" >&2; return 1;
    }
    docker run --rm --pull=never --network none --entrypoint /bin/sh "$IMAGE" \
      -c 'test -f "$1"' sh "$NCCL_PIP_SO" || {
      echo "NCCL: $IMAGE lacks the pip library at $NCCL_PIP_SO" >&2; return 1;
    }
  fi
  if switchless_ring_enabled; then ring_gid_index '' "${1:-}" || return 1; fi
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
  for key in NCCL_OVERLAY_PIP NCCL_PIP_SO NCCL_SHA256 NCCL_CONTAINER_DIR NCCL_SWITCHLESS_RING_ONLY IB_HCA IMAGE NCCL_IB_GID_INDEX NCCL_IB_GID_AUTO; do
    printf '%s=%q\n' "$key" "${!key:-}"
  done
  # switchless_ring_enabled is the first thing nccl_preflight and
  # nccl_gid_publication_check call; leaving it out makes the remote payload
  # exit 0 without running any check (command not found swallowed by `|| return 0`).
  declare -f switchless_ring_enabled nccl_library nccl_mount_args ring_gid_index \
    nccl_file_sha256 nccl_preflight dual_pci_domain_args dual_pci_domain_env_string \
    nccl_gid_publication_check
}
