#!/bin/bash
# Phase A perf matrix: same workload across three hikyaku routes that differ
# only in load-balancing strategy. The TTFT/throughput delta isolates the
# stickiness vs round-robin vs single-backend effect.
#
#   bench-sticky                  → code-cluster (sticky_least_loaded), 2 backends
#   bench-rr                      → code-cluster-rr (round_robin), 2 backends
#   Qwen/Qwen3.6-27B-AWQ-INT4     → port-3042 only (single backend)
#
# Pass 1 (quick) parameters chosen so the whole matrix runs in ~10 min.
# Pass 2 (full) is invoked with PASS=2 for the longer matrix.

set -euo pipefail

PASS="${PASS:-1}"
BASE_URL="${BASE_URL:-http://limone:4000/v1}"
TOKENIZER="${TOKENIZER:-$HOME/.cache/huggingface/hub/models--cyankiwi--Qwen3.6-27B-AWQ-INT4/snapshots/c9b937c5466c5c0575fc15edd1f8c516cb1e62fd}"
OUT_DIR="${OUT_DIR:-$(dirname "$0")/llama-benchy-runs/pass${PASS}-$(date +%Y%m%d-%H%M%S)}"

if [[ "$PASS" == "1" ]]; then
    PP="1024"
    TG="128"
    DEPTH="0 8192"
    CONCURRENCY="1 4"
    RUNS="2"
else
    PP="1024"
    TG="256"
    DEPTH="0 2048 8192 16384"
    CONCURRENCY="1 2 4 8"
    RUNS="3"
fi

ROUTES=(
    "bench-sticky"
    "bench-rr"
    "Qwen/Qwen3.6-27B-AWQ-INT4"
)

mkdir -p "$OUT_DIR"
echo "=== Phase A pass $PASS ==="
echo "results dir: $OUT_DIR"
echo "params: pp=$PP tg=$TG depth=[$DEPTH] concurrency=[$CONCURRENCY] runs=$RUNS"
echo

for ROUTE in "${ROUTES[@]}"; do
    SAFE_NAME="$(echo "$ROUTE" | tr '/' '_')"
    OUT_FILE="$OUT_DIR/${SAFE_NAME}.json"
    echo "--- $ROUTE → $OUT_FILE ---"
    llama-benchy \
        --base-url "$BASE_URL" \
        --model "$ROUTE" \
        --served-model-name "$ROUTE" \
        --tokenizer "$TOKENIZER" \
        --pp $PP \
        --tg $TG \
        --depth $DEPTH \
        --concurrency $CONCURRENCY \
        --runs "$RUNS" \
        --enable-prefix-caching \
        --latency-mode generation \
        --skip-coherence \
        --save-result "$OUT_FILE" \
        --format json \
        --save-total-throughput-timeseries \
        2>&1 | grep -E '^(Running test|Run [0-9]|Warmup|llama-benchy|Error|FAIL)' || true
    echo
done

echo "=== matrix done ==="
ls -la "$OUT_DIR"
