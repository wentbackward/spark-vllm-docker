#!/bin/bash
# Bring up the full spark-vllm cluster config. Idempotent: skips any
# endpoint already serving, so it's safe to run at boot AND by hand.
# Single source of truth for the cluster layout (replaces the scattered
# per-model services that drifted stale and broke the 2026-06 power-loss
# recovery).
#
# Layout:
#   spark-01:3042  27B-FP8 TP=2 across both Sparks (head here), MTP=3,
#                  prefix-caching ON, 262K, gmu 0.30/node, text-only
#   spark-01:3040  35B-A3B-FP8, MTP-OFF (KNOWLEDGE §5), 128k, gmu 0.38
#                  (KV trimmed 2026-07-07 to back ~4.3x @131K — matches
#                   max_num_seqs=4. Measured floor: 34.2 GiB weights +
#                   ~6 GiB fixed overhead = ~40 GiB, so 0.38 is near the
#                   minimum that seats 4 full-context slots. 0.32 FAILS
#                   (negative KV). Frees only ~2.4 GiB vs the old 0.40 —
#                   the model can't go much lower without cutting context.)
#   spark-02:3043  4B-Instruct-2507-FP8, small utility, 2048, gmu 0.12
# All on the vLLM 0.23 image (vllm-node-tf5-v0231).
#
# HARDENED 2026-07-06 after the second power-loss boot failed:
#   1. SERIAL launches with wait-for-ready between models. Parallel "-d"
#      dispatch made the vLLM memory profilers race each other at cold
#      boot ("No available memory for the cache blocks" on 35B AND 4B).
#   2. STALE-CONTAINER teardown before each launch. Docker's restart
#      policy revives old containers at boot as empty shells (dead Ray
#      state inside) — the TP=2 relaunch collided with one
#      (ray ActorHandleNotFoundError). Any container whose port is not
#      serving gets docker rm -f before its launch.

set -uo pipefail

VLLM_DIR="$HOME/hacking/spark-vllm-docker"
IMAGE="vllm-node-tf5-v0231"
SPARK01_CX7="192.168.200.13"
SPARK02_CX7="192.168.200.12"
SPARK02_SSH="paul@${SPARK02_CX7}"

cd "$VLLM_DIR" || { echo "FATAL: no repo at $VLLM_DIR"; exit 1; }
source .venv/bin/activate 2>/dev/null || true

log() { echo "[start-cluster $(date +%T)] $*"; }
port_up() { curl -sf -m 3 "http://localhost:$1/v1/models" >/dev/null 2>&1; }
remote_port_up() { ssh -o ConnectTimeout=5 -o BatchMode=yes "$SPARK02_SSH" "curl -sf -m 3 http://localhost:$1/v1/models >/dev/null 2>&1"; }
spark02_ok() { ssh -o ConnectTimeout=5 -o BatchMode=yes "$SPARK02_SSH" 'echo ok' >/dev/null 2>&1; }

# wait_port <local|remote> <port> <tries x10s> — poll until serving
wait_port() {
    local where=$1 port=$2 tries=$3 i
    for i in $(seq 1 "$tries"); do
        if [ "$where" = local ]; then port_up "$port" && return 0
        else remote_port_up "$port" && return 0; fi
        sleep 10
    done
    return 1
}

# --- wait for spark-02 to be ssh-reachable (TP=2 worker + 4B host) ---
log "waiting for spark-02 ssh..."
for i in $(seq 1 60); do
    spark02_ok && { log "spark-02 reachable"; break; }
    [ "$i" -eq 60 ] && log "WARNING: spark-02 unreachable after 5 min — TP=2 + 4B will be skipped"
    sleep 5
done

# --- 1. 27B-FP8 TP=2 (cross-node; head = spark-01:3042) ---
if port_up 3042; then
    log "27B TP=2 already serving on :3042 — skip"
elif spark02_ok; then
    log "tearing down stale TP=2 containers (both nodes)..."
    docker rm -f vllm_tp2 >/dev/null 2>&1
    ssh -o BatchMode=yes "$SPARK02_SSH" 'docker rm -f vllm_tp2 >/dev/null 2>&1'
    log "launching 27B TP=2 across both Sparks..."
    ./run-recipe.py qwen3.6-27b-fp8-mtp-vlm \
        -n "${SPARK01_CX7},${SPARK02_CX7}" --tp 2 --name vllm_tp2 \
        -t "$IMAGE" --gpu-mem 0.30 -d \
        -- --language-model-only --distributed-executor-backend ray \
        || log "WARNING: 27B TP=2 launch failed"
    if wait_port local 3042 60; then log "27B TP=2 up on :3042"
    else log "WARNING: 27B TP=2 not serving after 10 min — continuing"; fi
else
    log "WARNING: spark-02 down — skipping 27B TP=2"
fi

# --- 2. 35B-A3B-FP8 MTP-OFF (spark-01 solo, :3040) ---
if port_up 3040; then
    log "35B already serving on :3040 — skip"
else
    docker rm -f vllm_35b >/dev/null 2>&1
    log "launching 35B-A3B-FP8 MTP-off..."
    ./run-recipe.py qwen3.6-35b-a3b-fp8-nomtp \
        --solo --name vllm_35b -t "$IMAGE" \
        --gpu-mem 0.38 --max-model-len 131072 -d \
        || log "WARNING: 35B launch failed"
    if wait_port local 3040 42; then log "35B up on :3040"
    else log "WARNING: 35B not serving after 7 min — continuing"; fi
fi

# --- 3. 4B utility (spark-02 solo, :3043) ---
if remote_port_up 3043; then
    log "4B already serving on spark-02:3043 — skip"
elif spark02_ok; then
    ssh -o BatchMode=yes "$SPARK02_SSH" 'docker rm -f vllm_4b >/dev/null 2>&1'
    log "launching 4B on spark-02..."
    ssh -o ConnectTimeout=8 "$SPARK02_SSH" \
        "cd ~/hacking/spark-vllm-docker && source .venv/bin/activate 2>/dev/null; ./run-recipe.py qwen3-4b-instruct-2507-fp8 --solo --name vllm_4b -t $IMAGE -d" \
        || log "WARNING: 4B launch failed"
    if wait_port remote 3043 30; then log "4B up on spark-02:3043"
    else log "WARNING: 4B not serving after 5 min"; fi
else
    log "WARNING: spark-02 down — skipping 4B"
fi

log "cluster bring-up done."
