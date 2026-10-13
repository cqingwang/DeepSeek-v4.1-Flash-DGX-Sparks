# BASE_IMAGE controls FROM in every serving recipe. Offline loads retain tags,
# not registry RepoDigests, so strip a digest only on the explicit offline path.
build_base_image() {
  local base="$BASE_IMAGE"
  if [[ "${SKIP_PULL:-0}" == 1 ]]; then
    base="${base%@*}"
    [[ "${base##*/}" == *:* ]] || die "SKIP_PULL=1 requires BASE_IMAGE with a local tag"
  fi
  printf '%s\n' "$base"
}

docker_proxy_precheck() {
  # The caller serializes the checker into remote commands before recipe sync.
  local details
  details=$(timeout 5 docker info --format '{{json .}}') \
    || { err "cannot read docker daemon proxy configuration (5s timeout)"; return 1; }
  printf '%s\n' "$details" | python3 -c "$DOCKER_PROXY_CHECK_CODE" "$1"
}

base_image_ready() {
  local base="$1" force="${2:-0}"
  if [[ "${SKIP_PULL:-0}" == 1 ]]; then
    docker image inspect "$base" >/dev/null \
      || { err "SKIP_PULL=1: local base image $base missing; docker load/tag it first"; return 1; }
  elif [[ "$force" == 1 ]] || ! docker image inspect "$base" >/dev/null 2>&1; then
    docker_proxy_precheck "$base" || return 1
    docker pull --platform linux/arm64 "$base" || return 1
  fi
}

base_image_worker_command() {
  local base="$1" force="${2:-0}"
  printf 'set -e\n%s\nSKIP_PULL=%q\nDOCKER_PROXY_CHECK_CODE=%q\nbase_image_ready %q %q\n' \
    "$(declare -f base_image_ready docker_proxy_precheck err)" \
    "${SKIP_PULL:-0}" "$DOCKER_PROXY_CHECK_CODE" "$base" "$force"
}
