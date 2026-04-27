# spark-vllm-docker — Knowledge Base

This file is the durable, repo-bound source of truth for everything learned
operating vLLM on the DGX Spark cluster. It travels with the repo. AI
assistants and humans should both read this first; agents should update it
when they discover something durable.

Scope: vLLM serving (single-node and multi-node), recipes, benchmarking,
and the network/memory topology specific to running this repo on a 2-node
DGX Spark cluster. Out of scope: ComfyUI, TTS, video gen, etc. — those have
their own homes.

---

## 1. Hardware

Two **NVIDIA DGX Spark** nodes, identical:

- aarch64 (ARM64), NVIDIA GB10 SoC, compute capability **SM121a**
- **121.69 GiB unified VRAM** per node — GPU and CPU share this pool
- Memory bandwidth: **~273 GB/s** (this is the dominant ceiling for dense
  decode workloads — see §6)
- Two ConnectX-7 200 GbE NICs per node (only one cable connected; see §2)
- Filesystem: standard Ubuntu, HF cache at `~/.cache/huggingface/hub/`
- **`gpu_memory_utilization` reserves a fraction of TOTAL VRAM**, not of
  remaining VRAM. Two vLLM processes on one node summing to >1.0 will OOM.

### Node naming and IPs

| Node | LAN | CX7 (cabled) | CX7 (cabled, secondary) |
|---|---|---|---|
| spark-01 | 192.168.1.235 | 192.168.200.13 | 192.168.100.11 |
| spark-02 | 192.168.1.247 | 192.168.200.12 | 192.168.100.10 |

The unused CX7 port on each node also gets an IP from NVIDIA's setup
script (`enP2p1s0f1np1` → `192.168.200.{16,17}`) but has no peer.
This dual-port-on-same-subnet config breaks the repo's `autodiscover.sh`
(see §2).

### External services

- **`limone.royal-armadillo.ts.net`** (Jetson Orin Nano) runs the
  **llm-proxy** that routes client traffic to the right vLLM endpoint by
  model name. Proxy is no longer hosted on spark-01 itself.
- Tailnet: both Sparks are joined and routable by hostname (`paul@spark-02`).

---

## 2. Network specifics and gotchas

### Single cable, dual subnets

The two Sparks are connected with **one** CX7 cable on port 1
(`enp1s0f1np1`). NVIDIA's setup tool assigns IPs on **both** ports and on
both `192.168.200.0/24` and `192.168.100.0/24` subnets. As a result:

- `enp1s0f1np1` (cabled) gets BOTH `192.168.200.x/24` AND `192.168.100.x/24`
- `enP2p1s0f1np1` (NOT cabled) ALSO gets `192.168.200.x/24` — stale,
  but appears UP because of internal SoC link state

The repo's `autodiscover.sh` refuses to choose between two interfaces on
the same subnet and exits with:
`Error: Interfaces enP2p1s0f1np1 and enp1s0f1np1 share the same subnet`

**Workaround:** create a manual `.env` in this directory pinning the
intended interface — `detect_interfaces()` returns early when both are set,
bypassing the conflict check:

```dotenv
CLUSTER_NODES=192.168.200.12,192.168.200.13
COPY_HOSTS=192.168.200.12
LOCAL_IP=192.168.200.13
ETH_IF=enp1s0f1np1
IB_IF=rocep1s0f1
```

(`rocep1s0f1` is the RoCE device matching `enp1s0f1np1` — confirm with
`ibdev2netdev`.)

### SSH over the CX7 link

Direct SSH between the Sparks works on either subnet. The CX7 link delivers
~450-500 MB/s for rsync (limited by SSD I/O and CPU encryption, not the
fabric — the fabric is ~25 GB/s).

```bash
ssh paul@192.168.200.12 ...      # spark-02 over CX7
rsync -aH --info=progress2 ...   # ~80 s for a 35 GiB model
```

### HF auth on spark-02 (and any new Spark)

Ubuntu's default `.bashrc` has an early interactive-shell guard:

```bash
case $- in *i*) ;; *) return ;; esac
```

This means `HF_TOKEN` exports placed in `.bashrc` are **not visible** to
non-interactive SSH commands (which is what `run-recipe.py` and `rsync`
trigger from a remote node). Symptom: `hf auth whoami` works interactively
but fails from scripts.

**Fix:** add to `~/.profile` instead (no interactive guard):

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=hf_...
```

---

## 3. vLLM operating notes

### Image

Current image: **`vllm-node-tf5`** (transformers ≥5, vLLM main).
Build: `./build-and-copy.sh --tf5 [--copy-to <host>]`. Use
`--no-build --copy-to ...` to ship an already-built image to a peer.

### Containers

- **Default container name** from `launch-cluster.sh` is `vllm_node`.
  Override per-instance with `--name <something>` to avoid collisions when
  running multiple endpoints on the same host.
- Container persists between launches via `sleep infinity`; models exec into
  it. Stopping a container reclaims the GPU memory; just `pkill`-ing the
  vLLM Python process inside the container does NOT — the EngineCore
  child processes survive and hold VRAM. Restart the container or kill from
  the host to reclaim.

### Mods and chat templates

`mods/fix-qwen3.5-chat-template` copies `unsloth.jinja` into the container.
This is **lost on every container restart** and must be re-applied. The
recipes' `mods:` block plus `run-recipe.py` handle this automatically.

For Qwen3.5/3.6 models, the recipe must include:

```
--chat-template unsloth.jinja
```

The `unsloth.jinja` file lives at:

```
/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/chat_templates/unsloth.jinja
```

inside the container after the mod is applied.

### Multi-node serving

For tensor parallelism across nodes:

```bash
./run-recipe.py <recipe> -n 192.168.200.13,192.168.200.12 --tp 2 -d
```

Required additions to the recipe (or recipe override) for multi-node:

- **`--distributed-executor-backend ray`** — vLLM's default
  multiprocessing executor assumes all GPUs are on one node.
  Without this, you'll see:
  `World size (2) is larger than the number of available GPUs (1) in this node`

`launch-cluster.sh` automatically starts Ray head/worker, applies mods to
both nodes, and runs the launch script.

### `HF_HUB_OFFLINE` in container

Containers don't inherit the host's HF_TOKEN by default. If the recipe
uses a model that's already cached locally, set `HF_HUB_OFFLINE: 1` in
the recipe's `env:` block to prevent vLLM from trying to fetch model
metadata over HTTP and failing with 401 (especially relevant for gated
repos like z-lab/Qwen3.6-27B-DFlash).

### Solo mode

`./run-recipe.py <recipe> --solo` skips Ray and node detection. Solo recipes
must NOT include `--distributed-executor-backend ray`.

### Stopping a cluster

```bash
./launch-cluster.sh -t vllm-node-tf5 --name <name> -n <ips> stop
```

This stops both the head and worker containers cleanly.

---

## 4. Models

### Active models

| Model | Quant | Weights | Multimodal | Notes |
|---|---|---|---|---|
| Qwen3.6-27B-FP8 | FP8 | ~28 GiB | yes (text-only via `--language-model-only`) | Dense, coding-focused, 256K context |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | AWQ-INT4 | ~14 GiB | text-only | Halves bandwidth → ~2× decode (see §5) |
| Qwen3.6-35B-A3B-FP8 | FP8 | ~34 GiB | text | MoE, 3B active params, ~3× decode of dense 27B at FP8 |

Served-model-name is independent of the actual model directory.
By convention all 27B variants serve as `Qwen/Qwen3.6-27B-...` so clients
don't need to know which quant. To add multiple aliases:

```
--served-model-name Qwen/Qwen3.6-27B-AWQ-INT4 Qwen/Qwen3.6-27B
```

### Quant quality notes

- **cyankiwi/Qwen3.6-27B-AWQ-INT4** is good — preserves MTP heads, agentic
  output quality matches FP8 within a few percentage points on observed
  workloads.
- **QuantTrio/Qwen3.5-35B-A3B-AWQ** (different model, different author) is
  bad — output quality unusable. Avoid.
- **NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 / NVFP4**: poor agentic quality on
  this stack as of 2026-04. Recipe retained but model not in active rotation.

### Speculative-decode draft models

| Method | Draft | Size | Notes |
|---|---|---|---|
| `dflash` | z-lab/Qwen3.6-27B-DFlash | 3.3 GiB | Gated HF repo — needs license accept |
| `qwen3_next_mtp` | (built-in heads) | 0 | Native to Qwen3.5/3.6, included in cyankiwi quant |

DFlash repo gating: requires `huggingface.co/z-lab/Qwen3.6-27B-DFlash`
acceptance. Once cached, set `HF_HUB_OFFLINE=1` in the recipe so vLLM
doesn't try to re-fetch the index over HTTP at startup (it would 401).

---

## 5. Speculative decoding

### Why it matters more on Spark than elsewhere

Public reports of "15-30% speedup from spec decoding" come from
**compute-bound** GPUs where memory bandwidth isn't the limit. On Spark
the memory bandwidth (273 GB/s) divided by model size sets a hard ceiling
for non-spec decode:

| Model size | Theoretical ceiling | Measured (~85% of ceiling) |
|---|---|---|
| 28 GiB (27B FP8) | 9.7 t/s | 8.22 t/s |
| 14 GiB (27B INT4) | 19.5 t/s | ~17 t/s (projected) |
| 34 GiB (35B FP8) | 8.0 t/s | ~6.5 t/s |

Spec decode amortizes weight reads across multiple verified tokens, so on
Spark it gives **2-3× actual decode throughput**, not 15-30%.

### Benchmark results — 27B, decode-only (tg=256, pp=512)

All measured with `llama-benchy --pp 512 --tg 256 --runs 3` on clean
single-Spark deployments at full memory budget (`--gpu-mem 0.7`):

| config | quant | t/s decode | peak | t/s prefill | TTFT (ms) |
|---|---|---|---|---|---|
| DFlash + AWQ-INT4 | INT4 | **35.99 ± 1.3** | 57.7 | 492 ± 282 | 2325 ± 2236 |
| DFlash + FP8 | FP8 | 23.86 ± 2.4 | 52.3 | 986 ± 117 | 529 ± 58 |
| MTP + AWQ-INT4 | INT4 | 21.30 ± 0.7 | 26.7 | 571 ± 225 | 1143 ± 618 |
| MTP + FP8 | FP8 | 15.97 ± 0.9 | 20.0 | 710 ± 287 | 940 ± 531 |
| No-spec + FP8 | FP8 | 8.22 ± 0.02 | 9.0 | 1320 ± 39 | 391 ± 12 |

### Workload-to-config mapping

- **Long-form decoding (agentic coding, openclaw, long generations)**:
  DFlash + AWQ-INT4. Roughly 4× the decode of no-spec FP8.
- **TTFT-sensitive interactive chat**: No-spec + FP8. Lowest, most
  consistent first-token latency.
- **Long-prompt prefill (large-context analysis)**: No-spec + FP8.
  flashinfer wins prefill; AWQ-INT4 actually loses on prefill.

### Speculative-decode gotchas

- **DFlash requires `--attention-backend flash_attn`**. The draft head uses
  non-causal attention; flashinfer rejects this with
  `non-causal attention not supported`.
- **flash_attn rejects `--kv-cache-dtype fp8`** with
  `kv_cache_dtype not supported`. So DFlash configs run with bf16 KV.
  This doubles KV memory but is fine at `--gpu-mem 0.7`.
- **MTP is fine with flashinfer + fp8 KV** — it uses the model's built-in
  multi-token prediction heads, which produce normal causal attention.
  cyankiwi's AWQ quant does preserve these heads (verified empirically).
- **`min_p` in `--override-generation-config`** is rejected by vLLM's
  spec-decode path even at the no-op value `0.0`. Drop it from the override.
- **DFlash variance is high** (decode std ±2.4 t/s) because acceptance is
  bimodal — some forward passes accept all 15 drafts, others only a couple.
  MTP variance is much lower because `num_speculative_tokens=2` caps the
  range.
- **MTP `num_speculative_tokens` is bounded by training**. Pushing past 2
  may hurt acceptance more than it helps throughput; needs validation per
  model.

---

## 6. Memory and resource patterns

### `gpu_memory_utilization` semantics

Reserves that fraction of **total** VRAM (121 GiB on Spark), not of
remaining-after-other-processes. So two vLLM instances on one node must
sum to ≤1.0 minus OS overhead (~5-10 GiB).

Working budgets per model at full context:

| Model | Util | Reservation | Headroom for KV |
|---|---|---|---|
| 27B-FP8 | 0.7 | ~85 GiB | ~57 GiB |
| 27B-AWQ-INT4 | 0.7 | ~85 GiB | ~71 GiB |
| 35B-A3B-FP8 | 0.40 | ~48 GiB | ~14 GiB |

### Co-location patterns and swap

Lessons from running multiple instances on one Spark:

- **27B (cluster head, 0.42 util) + 35B (0.40 util) + ComfyUI + proxy stack**
  on one node → 113/121 GiB used, 6.2 GiB swap, observable slowdown.
- **Two vLLM instances on one Spark**, even with one idle, can push into swap
  because both processes hold their full reservations in unified memory.
  Idle 35B + active 27B test showed 8 t/s instead of expected 16 t/s
  (~50% slowdown from swap pressure, not GPU contention).
- **Distribution across both Sparks** keeps each node ~50-90 GiB used,
  no swap, predictable performance.

### Recommended balance

For two-node deployment with one model dominant on each:

- **spark-01**: one large model at high util (e.g. 27B at 0.7)
- **spark-02**: another large model at high util OR free for tests/sweeps

For multi-node TP=2 of a single model:

- Model spread across both nodes, each at ~0.7 util
- Nothing else on either node; all auxiliary services move to limone or
  other hosts

---

## 7. Benchmarking tools

Both installed via `uv tool install` (the latter bundles the former):

- **`tool-eval-bench`** — 69 deterministic tool-calling scenarios across 15
  categories (selection, parameter precision, refusal, prompt injection,
  multi-step, etc.). Pass/partial/fail scoring.
  Install: `uv tool install 'tool-eval-bench[perf] @ git+https://github.com/SeraphimSerapis/tool-eval-bench.git'`
- **`llama-benchy`** — `llama-bench`-style pp/tg measurement against any
  OpenAI-compatible endpoint. Critically:
  - Handles MTP/spec-decode chunks correctly (vLLM's own bench-sweep does not)
  - Avoids prefix-cache-induced TTFT skew (uses Project Gutenberg corpus,
    realistic prompt lengths)
  - Auto-detects model from `/v1/models` if not specified

### Standard benchmarking commands

```bash
# llama-benchy: decode-focused
llama-benchy --base-url http://<host>:<port>/v1 \
  --model <served-model-name> --tokenizer <hf-model-id> \
  --pp 512 --tg 256 --depth 0 --runs 3 --skip-coherence \
  --format md --save-result /tmp/benchy.md

# tool-eval-bench: smoke
tool-eval-bench --base-url http://<host>:<port> \
  --scenarios TC-01 TC-02 TC-03 TC-04 TC-05 --no-think \
  --output-dir /tmp/teb-smoke

# tool-eval-bench: quick quality (15 scenarios)
tool-eval-bench --base-url http://<host>:<port> --short --seed 42

# tool-eval-bench: full 69
tool-eval-bench --base-url http://<host>:<port> --seed 42
```

If the endpoint expects `/v1/...` paths, append `/v1` to `--base-url`
for `llama-benchy` (it errored on `/models` without it).

For a clean A/B (e.g. spec-decode vs not), each instance must have a
**clean GPU** — no co-located vLLM processes, no swap pressure.

---

## 8. Recipes — what each one is for

| Recipe | Model | Spec | Backend | Notes |
|---|---|---|---|---|
| `qwen3.6-27b-fp8.yaml` | FP8 | none | flashinfer + fp8 KV | Baseline 27B |
| `qwen3.6-27b-fp8-mtp.yaml` | FP8 | qwen3_next_mtp | flashinfer + fp8 KV | FP8 + native MTP |
| `qwen3.6-27b-fp8-dflash.yaml` | FP8 | dflash | flash_attn + bf16 KV | FP8 + DFlash draft |
| `qwen3.6-27b-awq-int4.yaml` | INT4 (cyankiwi) | none | flashinfer + fp8 KV | Baseline AWQ |
| `qwen3.6-27b-awq-int4-mtp.yaml` | INT4 | qwen3_next_mtp | flashinfer + fp8 KV | AWQ + native MTP |
| `qwen3.6-27b-awq-int4-dflash.yaml` | INT4 | dflash | flash_attn + bf16 KV | AWQ + DFlash; recommended for agentic coding |
| `qwen3.6-35b-a3b-fp8.yaml` | FP8 | none | flashinfer + fp8 KV | 35B MoE |
| `nemotron-3-nano-fp8.yaml` | FP8 | none | — | Not in active use, poor quality |

All recipes default to `--max-num-seqs 4`. Increasing past this risks
"max_num_seqs (N) exceeds available Mamba cache blocks" at low utils.

### Common defaults to remember

- `gpu_memory_utilization: 0.7` for solo on a clean Spark
- `gpu_memory_utilization: 0.42` for a 27B alongside a 35B at 0.40 (cluster)
- `max_model_len: 262144` for 27B (Qwen 3.6 max), `131072` for 35B
- Always include `--language-model-only` for 27B if not using vision
- Always include `--default-chat-template-kwargs '{"preserve_thinking": true}'`
  for agentic coding workloads; `false` for orchestration agents that don't
  need historical reasoning chains

---

## 9. Open questions and current state pointers

These change session-to-session; check current reality before relying on
them:

- **35B status**: typically running on whichever Spark has spare capacity;
  client routing via limone proxy. If down, openclaw fails over to HF.
- **DFlash + 35B**: not yet attempted. The MTP heads on 35B-A3B work, but
  DFlash would need a draft model trained against Qwen3.6-35B specifically
  (none exists publicly as of 2026-04).
- **TP=2 vs DFlash trade-off**: empirically, single-node DFlash on AWQ-INT4
  beats 2-node TP for memory-bandwidth-bound dense models. TP=2 wins when
  KV cache is the bottleneck (very long contexts) or for compute-heavy
  prefill on large prompts.
- **`resolver/TEST-MATRIX.md`**: aspirational sweep plan, partly stale (was
  written before the agentic harness rewrite). Re-read before running.

---

## 10. Useful one-liners

```bash
# Memory state on both nodes
echo "=== spark-01 ==="; free -h | head -3
echo "=== spark-02 ==="; ssh paul@192.168.200.12 "free -h | head -3"

# Show which model is in each container
for c in $(docker ps --format '{{.Names}}'); do
  echo "--- $c ---"
  docker exec "$c" sh -c "ps -ef | grep 'vllm serve' | grep -v grep" | head -1
done

# Spec-decode acceptance metrics (run after some traffic)
docker logs <container> 2>&1 | grep -i 'spec.*accept\|acceptance rate' | tail -5

# Stop a multi-node cluster cleanly
./launch-cluster.sh -t vllm-node-tf5 --name <name> -n <ips> stop

# Sync a model to spark-02 over CX7
rsync -aH --info=progress2 \
  ~/.cache/huggingface/hub/models--<repo>--<name>/ \
  paul@192.168.200.12:~/.cache/huggingface/hub/models--<repo>--<name>/

# Quick endpoint sanity check
curl -s -m 5 http://<host>:<port>/v1/models | python3 -m json.tool
```

---

## How to maintain this file

- Add findings here when something durable is learned (a workaround, a
  benchmark result, a constraint, a quant quality observation).
- Don't add transient state (today's container names, current memory free,
  what's running right now). Those go in commit messages or chat.
- Don't duplicate what's in the recipes — point to them.
- When a fact changes (e.g., a new vLLM version fixes a bug noted here),
  update or strike through the old fact rather than just appending.
- Keep section structure stable so future-you knows where to look.
