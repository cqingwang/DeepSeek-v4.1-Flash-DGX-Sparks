#!/usr/bin/env bash
# start-tp4.sh — DeepSeek-V4.1-Flash on 4× DGX Spark (TP4/EP1 managed profile).
#
# Same engine, image and commands as start.sh, with a profile of its own:
#   .env.tp4      settings for this profile (copied from .env.tp4.example on first run)
#   state-tp4/    launch record, smoke result, api key, DSpark tables
#   logs-tp4/     engine log
# The 3-Spark profile (.env, state/, logs/) is untouched, so one checkout can drive
# either fleet. Every TP4-specific runtime choice (context, KV pool, concurrency,
# prefill chunk, memory fraction, no head padding) lives in .env.tp4.example.
# Serving knobs shared with TP3 (DSpark k, max_tokens cap, loop abort, thinking
# alias) are in start.sh + the overlay image; .env.tp4.example defaults match.
#
# Usage: ./start-tp4.sh doctor | build | share | pack | serve | stop | status | logs | smoke
#
# DSV41_LAUNCHER=tp4 is the only switch that arms TP4 behavior inside start.sh
# and the adapter. In the managed Spark workspace, the parent owns the environment,
# rank order and BuildKit ring context; `build` compiles once on rank 0 and invokes
# the adjacent-hop image shipper. ./start.sh leaves the switch unset, preserving
# the standalone three-node path.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ENV_FILE="$ROOT/.env.tp4"
export ENV_EXAMPLE="$ROOT/.env.tp4.example"
export STATE_DIR="${STATE_DIR:-$ROOT/state-tp4}"
export LOG_DIR="${LOG_DIR:-$ROOT/logs-tp4}"
export SERVE_LOG="${SERVE_LOG:-$LOG_DIR/dsv41.log}"
export DSV41_LAUNCHER=tp4
exec "$ROOT/start.sh" "$@"
