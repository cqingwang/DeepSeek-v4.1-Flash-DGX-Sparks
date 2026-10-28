#!/bin/bash
# Managed target head (rank0) health monitor.
# The current deployment owns container names and runtime paths; no home-directory
# or legacy state profile is used here.
set -u
ROOT="${DEPLOY_ROOT:-/opt/spark/deepseek_v4.1_flash_switchless}"
RUNTIME_ASSETS="${RUNTIME_ASSETS:-/opt/runtime_assets/deepseek_v4.1_flash_switchless}"
TARGET_ID="${TARGET_ENGINE_ID:-deepseek_v4.1_flash_switchless}"
NAME="${TARGET_ID}-r0"
URL="${HEALTH_URL:-http://127.0.0.1:8888/health}"
FAILS=0
MAINT_FLAG="$RUNTIME_ASSETS/state/maintenance.flag"
LOG_FILE="$RUNTIME_ASSETS/logs/monitor.log"
mkdir -p "$RUNTIME_ASSETS/logs"

while :; do
  if [ -f "$MAINT_FLAG" ]; then
    sleep 300; continue
  fi
  ok=1
  if docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    h=$(docker inspect -f '{{.State.Health.Status}}' "$NAME" 2>/dev/null || echo starting)
    code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null || echo 000)
    [ "$h" = healthy ] && [ "$code" = 200 ] || ok=0
  else
    ok=0
  fi
  if [ "$ok" = 1 ]; then FAILS=0; sleep 60; continue; fi
  FAILS=$((FAILS+1))
  if [ "$FAILS" -lt 3 ]; then sleep 60; continue; fi
  echo "[dsv41-monitor] $(date) rebuild: container=$(docker ps -q --filter name=$NAME | wc -l) /health=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null)" >>"$LOG_FILE"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  cd "$ROOT" && ENV_FILE="$ROOT/.env.tp4" ./start-tp4.sh serve >>"$LOG_FILE" 2>&1 || true
  FAILS=0
  sleep 60
done
