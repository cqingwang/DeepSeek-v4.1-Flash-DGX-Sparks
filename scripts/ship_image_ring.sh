#!/usr/bin/env bash
# 在已构建的 rank0 镜像上，沿本地物理环逐跳传输同一个 ARM64 制品。
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPARK_DIR="$(cd "$MODULE_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$MODULE_DIR/.env.tp4}"
IMAGE_ARCHIVE_DIR="${IMAGE_ARCHIVE_DIR:-/opt/spark/images}"
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
trap 'rm -f "$ROUTE_FILE"' EXIT
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
mkdir -p "$IMAGE_ARCHIVE_DIR"
IMAGE_ARCHIVE="${IMAGE_ARCHIVE:-$IMAGE_ARCHIVE_DIR/${SAFE_IMAGE}.tar}"
PARTIAL_ARCHIVE="$IMAGE_ARCHIVE.partial"
REMOTE_ARCHIVE="${REMOTE_ARCHIVE:-$REMOTE_ARCHIVE_DIR/${SAFE_IMAGE}-$$.tar}"

if [[ -f "$IMAGE_ARCHIVE" && -s "$IMAGE_ARCHIVE" ]]; then
  echo "复用已生成的 Docker archive: $IMAGE_ARCHIVE"
else
  echo "导出 $IMAGE ($IMAGE_ID)"
  docker save "$IMAGE" >"$PARTIAL_ARCHIVE"
  mv "$PARTIAL_ARCHIVE" "$IMAGE_ARCHIVE"
fi
IMAGE_SHA256="$(sha256sum "$IMAGE_ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$IMAGE_SHA256" "$IMAGE_ARCHIVE" >"$IMAGE_ARCHIVE.sha256"
echo "archive_sha256=$IMAGE_SHA256"

hop=0
while IFS=$'\t' read -r source_node source_management source_iface source_fabric target_node target_management target_fabric; do
  [[ -n "$target_fabric" ]] || { echo "环网路径缺少目标地址: $source_node" >&2; exit 2; }
  echo "第 $((hop + 1)) 跳: $source_node -> $target_node ($source_fabric -> $target_fabric)"
  if [[ "$hop" == 0 ]]; then
    rsync -aP --partial --append-verify \
      -e "ssh -o BatchMode=yes -o ConnectTimeout=15 -b $source_fabric" \
      "$IMAGE_ARCHIVE" "$REMOTE_USER@$target_fabric:$REMOTE_ARCHIVE"
    ssh -n -o BatchMode=yes -o ConnectTimeout=15 -b "$source_fabric" \
      "$REMOTE_USER@$target_fabric" "docker load -i '$REMOTE_ARCHIVE' >/dev/null"
  else
    remote_command="rsync -aP --partial --append-verify -e 'ssh -o BatchMode=yes -o ConnectTimeout=15 -b $source_fabric' '$REMOTE_ARCHIVE' '$REMOTE_USER@$target_fabric:$REMOTE_ARCHIVE' && ssh -o BatchMode=yes -o ConnectTimeout=15 -b '$source_fabric' '$REMOTE_USER@$target_fabric' 'docker load -i $REMOTE_ARCHIVE >/dev/null'"
    ssh -n -o BatchMode=yes -o ConnectTimeout=15 "$REMOTE_USER@$source_management" "$remote_command"
    ssh -n -o BatchMode=yes "$REMOTE_USER@$source_management" "rm -f '$REMOTE_ARCHIVE'"
  fi
  target_image_id="$(ssh -n -o BatchMode=yes "$REMOTE_USER@$target_management" "docker image inspect --format '{{.Id}}' '$IMAGE'")"
  [[ "$target_image_id" == "$IMAGE_ID" ]] || {
    echo "$target_node image ID 不一致: expected=$IMAGE_ID actual=$target_image_id" >&2
    exit 1
  }
  echo "$target_node image_id=$target_image_id verified"
  hop=$((hop + 1))
done <"$ROUTE_FILE"

[[ "$hop" == 3 ]] || { echo "期望 3 个相邻环网 hop，实际 $hop" >&2; exit 1; }
last_management="$(awk -F '\t' 'END { print $6 }' "$ROUTE_FILE")"
ssh -n -o BatchMode=yes "$REMOTE_USER@$last_management" "rm -f '$REMOTE_ARCHIVE'"
echo "镜像已沿物理环传输并在四个 rank 使用同一 image ID: $IMAGE_ID"
