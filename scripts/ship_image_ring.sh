#!/usr/bin/env bash
# 在已构建的 rank0 镜像上，沿本地物理环逐跳传输同一个 ARM64 制品。
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_DIR="$(cd "$MODULE_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$MODULE_DIR/.env.tp4}"
REMOTE_ARCHIVE_DIR="${REMOTE_ARCHIVE_DIR:-/var/tmp}"

[[ -f "$ENV_FILE" ]] || { echo "缺少 managed 环境文件: $ENV_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
source "$ENV_FILE"

REMOTE_USER="${WORKER_USER:?WORKER_USER 未设置}"
IMAGE="${IMAGE:?IMAGE 未设置}"
command -v docker >/dev/null 2>&1 || { echo "rank0 缺少 docker" >&2; exit 2; }
command -v rsync >/dev/null 2>&1 || { echo "rank0 缺少 rsync" >&2; exit 2; }
command -v ssh >/dev/null 2>&1 || { echo "rank0 缺少 ssh" >&2; exit 2; }

ROUTE_FILE="$(mktemp -t dsv41-ring-image-route.XXXXXX)"
python3 "$SPARK_DIR/program.py" ring-transfer-route --config "$SPARK_DIR/config.yaml" >"$ROUTE_FILE"

HEAD_NODE="$(awk -F '\t' 'NR == 1 { print $1 }' "$ROUTE_FILE")"
[[ "$(hostname -s)" == "$HEAD_NODE" ]] || {
  echo "必须在 rank0 ($HEAD_NODE) 执行，当前为 $(hostname -s)" >&2
  exit 2
}

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null)" || {
  echo "rank0 未找到镜像: $IMAGE" >&2
  exit 2
}

SAFE_IMAGE="${IMAGE//\//_}"
SAFE_IMAGE="${SAFE_IMAGE//:/_}"
IMAGE_ARCHIVE="$(mktemp --tmpdir=/var/tmp "${SAFE_IMAGE}.XXXXXX.tar")"
PARTIAL_ARCHIVE="$IMAGE_ARCHIVE.partial"
REMOTE_ARCHIVE="${REMOTE_ARCHIVE:-$REMOTE_ARCHIVE_DIR/${SAFE_IMAGE}-$$.tar}"
trap 'rm -f "$ROUTE_FILE" "$IMAGE_ARCHIVE" "$PARTIAL_ARCHIVE" "$IMAGE_ARCHIVE.sha256.partial"' EXIT

archive_image_id() {
  local archive="$1" config_path
  config_path="$(tar -xOf "$archive" manifest.json | python3 -c '
import json
import re
import sys
from pathlib import PurePosixPath

image = sys.argv[1]
try:
    manifest = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError) as exc:
    raise SystemExit(f"invalid Docker archive manifest: {exc}")
matches = [entry for entry in manifest if image in (entry.get("RepoTags") or [])]
if len(matches) != 1:
    raise SystemExit(f"archive must contain exactly one RepoTags entry for {image}")
config = matches[0].get("Config", "")
config_path = PurePosixPath(config)
legacy_config = re.fullmatch(r"[0-9a-f]{64}\.json", config) is not None
oci_config = re.fullmatch(r"blobs/sha256/[0-9a-f]{64}", config) is not None
if (
    not (legacy_config or oci_config)
    or config_path.is_absolute()
    or ".." in config_path.parts
):
    raise SystemExit(f"unsafe Docker image config path: {config!r}")
print(config)
' "$IMAGE")" || return 1
  tar -xOf "$archive" "$config_path" | sha256sum | awk '{print "sha256:" $1}'
}

echo "导出 $IMAGE ($IMAGE_ID) 到临时 archive $IMAGE_ARCHIVE"
if ! docker save "$IMAGE" >"$PARTIAL_ARCHIVE"; then
  rm -f "$PARTIAL_ARCHIVE"
  echo "docker save 失败: $IMAGE" >&2
  exit 1
fi
saved_image_id="$(archive_image_id "$PARTIAL_ARCHIVE")" || {
  rm -f "$PARTIAL_ARCHIVE"
  echo "无法从新 archive 读取 Docker image ID: $PARTIAL_ARCHIVE" >&2
  exit 1
}
[[ "$saved_image_id" == "$IMAGE_ID" ]] || {
  rm -f "$PARTIAL_ARCHIVE"
  echo "新 archive image ID 不匹配: expected=$IMAGE_ID actual=$saved_image_id" >&2
  exit 1
}
mv "$PARTIAL_ARCHIVE" "$IMAGE_ARCHIVE"
IMAGE_SHA256="$(sha256sum "$IMAGE_ARCHIVE" | awk '{print $1}')"
echo "archive_sha256=$IMAGE_SHA256"

[[ "$IMAGE_SHA256" =~ ^[a-f0-9]{64}$ ]] || { echo "archive SHA-256 无效" >&2; exit 1; }
[[ "$REMOTE_ARCHIVE" == /* && "$REMOTE_ARCHIVE" != *[$' \t\r\n']* ]] || {
  echo "REMOTE_ARCHIVE 必须是无空白的绝对路径" >&2
  exit 2
}
REMOTE_ARCHIVE_QUOTED="$(printf '%q' "$REMOTE_ARCHIVE")"
REMOTE_VERIFY_AND_LOAD="printf '%s  %s\\n' '$IMAGE_SHA256' $REMOTE_ARCHIVE_QUOTED | sha256sum -c - && \
  docker load -i $REMOTE_ARCHIVE_QUOTED >/dev/null"

hop=0
while IFS=$'\t' read -r source_node source_management source_iface source_fabric target_node target_management target_fabric; do
  [[ -n "$target_fabric" ]] || { echo "环网路径缺少目标地址: $source_node" >&2; exit 2; }
  echo "第 $((hop + 1)) 跳: $source_node -> $target_node ($source_fabric -> $target_fabric)"
  if [[ "$hop" == 0 ]]; then
    rsync -aP --partial --append-verify \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -b $source_fabric" \
      "$IMAGE_ARCHIVE" "$REMOTE_USER@$target_fabric:$REMOTE_ARCHIVE"
  else
    remote_target="$REMOTE_USER@$target_fabric:$REMOTE_ARCHIVE"
    printf -v remote_command "rsync -aP --partial --append-verify -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -b %s' %q %q" \
      "$source_fabric" "$REMOTE_ARCHIVE" "$remote_target"
    ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_USER@$source_management" "$remote_command"
  fi
  # The archive is loaded only after the receiver independently verifies every
  # byte; management SSH carries commands, never the image payload.
  ssh -n -o BatchMode=yes "$REMOTE_USER@$target_management" "$REMOTE_VERIFY_AND_LOAD"
  target_image_id="$(ssh -n -o BatchMode=yes "$REMOTE_USER@$target_management" "docker image inspect --format '{{.Id}}' '$IMAGE'")"
  [[ "$target_image_id" == "$IMAGE_ID" ]] || {
    echo "$target_node image ID 不一致: expected=$IMAGE_ID actual=$target_image_id" >&2
    exit 1
  }
  echo "$target_node image_id=$target_image_id verified"
  if [[ "$hop" -gt 0 ]]; then
    ssh -n -o BatchMode=yes "$REMOTE_USER@$source_management" "rm -f $REMOTE_ARCHIVE_QUOTED"
  fi
  hop=$((hop + 1))
done <"$ROUTE_FILE"

[[ "$hop" == 3 ]] || { echo "期望 3 个相邻环网 hop，实际 $hop" >&2; exit 1; }
last_management="$(awk -F '\t' 'END { print $6 }' "$ROUTE_FILE")"
ssh -n -o BatchMode=yes "$REMOTE_USER@$last_management" "rm -f '$REMOTE_ARCHIVE'"
echo "镜像已沿物理环传输并在四个 rank 使用同一 image ID: $IMAGE_ID"
