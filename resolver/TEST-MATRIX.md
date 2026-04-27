# DGX Spark Model Performance Test Matrix

## Context

With the agentic test harness now capable of producing reproducible, metric-rich results — and a second DGX Spark arriving soon — we can systematically characterize how model capability, deployment topology, and concurrency interact on this hardware.

This document defines the test matrix, precise definitions for each axis, methodology, and execution order.

## Models under test

- **Qwen3.6-35B-A3B-FP8** (primary)
- **Nemotron-3-Nano-30B-A3B-FP8** (secondary — for comparison, MoE-on-single-Spark baseline)

Both are MoE with ~3B active params; roughly comparable compute profile. Qwen is the daily driver; Nemotron is the contrast case.

## Test matrix

**4 deployment modes × 2 MTP states × 4 concurrency levels = 32 cells per model.**

### Axis 1 — Deployment topology

| Code | Topology | Notes |
|---|---|---|
| S1 | Single machine, serial | Baseline. All other results normalized to this. |
| S1C | Single machine, concurrent | Tests per-node scheduler/batching. `gpu_memory_utilization` and `max_num_seqs` matter here. |
| S2 | Two machines, serial | Measures inter-node overhead with only one request in flight. Expect ~20-50% slower than S1 due to 200GbE round-trips per decode step. |
| S2C | Two machines, concurrent | The actual production scenario for multi-Spark clusters. Reveals if aggregate throughput scales. |

### Axis 2 — Speculative decoding (MTP)

| Code | State |
|---|---|
| M− | MTP off |
| M+ | MTP on (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` for Qwen3.5 or `"qwen3_next_mtp"` for Qwen3.6) |

**Hypothesis to test:** MTP wins on low-concurrency single-stream work, loses (or breaks even) at high concurrency because continuous batching already saturates compute. Nemotron: check if MTP is even supported.

### Axis 3 — Concurrency level

| Code | Definition | Client-side arrangement |
|---|---|---|
| C1 | 1 request at a time | Serial scenarios, one connection |
| C2 | 2 in-flight | See "Concurrency definitions" below |
| C3 | 3 in-flight | " |
| C4 | 4 in-flight | " |

Since the harness supports both patterns, **run both**:

- **C_n(a)** — N **separate scenarios** running in parallel (N client connections, each driving a full scenario). Tests queue fairness and isolation.
- **C_n(b)** — **1 scenario** issuing N concurrent tool/sub-agent calls. Tests intra-scenario parallelism within a single logical session.

## Concurrency definitions (precise)

- **"Serial"** = one HTTP connection to the proxy, requests issued sequentially (`request N+1` starts after `request N`'s final token).
- **"C_n(a)"** = N independent client processes, each driving an independent scenario to completion. Start staggered by ~200ms to avoid pathological cold-start collisions. Report per-client metrics and aggregate.
- **"C_n(b)"** = 1 scenario that internally issues N parallel requests (e.g., via the harness's sub-agent spawning, or explicit `Promise.all()`-style tool dispatch). Report scenario-level metrics and per-request breakdown.

## Methodology

### Warm-up

Each cell gets **one throw-away scenario run** before measurement begins. This pays the torch.compile + CUDA graph capture cost (~1-2 min first request after startup) so it doesn't pollute measured latencies.

### Seeds per cell

Each scenario runs **N=5 seeds** per cell (configurable), reporting:
- Pass rate + 95% CI
- p50, p95, max for latency metrics

### Baseline run — S1 M− C1

Run this cell **first**. Every other cell's metrics are reported relative to this baseline. Absolute numbers are useful too, but normalized comparisons (`1.15× slower`, `2.3× more throughput`) are easier to reason about across 32 cells.

### Backend consistency

- All vLLM servers use the same image (`vllm-node-tf5`)
- Same recipe per model (only `gpu_memory_utilization` and speculative config differ between cells)
- `kv_cache_dtype: fp8`, `enforce_eager: true` for stability across all cells
- Multi-machine: use **expert parallel (EP)**, NOT tensor parallel (TP). For A3B MoE, EP only sends expert-routing messages over the network (~100× less traffic than TP all-reduce). With 200GbE interconnect, TP would be catastrophic.

## Per-cell metrics

Every scenario run captures:

**Correctness:**
- Pass rate per criterion
- Failure mode histogram (wrong-tool, hallucinated-tool, forbidden-tool, timeout, bad-JSON, loop)

**Latency:**
- TTFT (time to first token) per request
- TPOT (time per output token) — steady-state decode speed
- Per-request wall time
- Per-scenario wall time (multi-turn completion)

**Throughput:**
- Aggregate tokens/sec across all concurrent requests
- Effective tokens/sec per request (aggregate ÷ concurrency)

**Context:**
- Input tokens, output tokens, cached tokens (vLLM prefix cache)
- Peak context size
- Context growth rate per turn

**Resource:**
- Peak GPU memory per node
- Peak network bandwidth between nodes (multi-machine only)
- CPU time on proxy

## Execution order (recommended)

Run in this order so high-value findings come early even if the full sweep isn't finished:

| Phase | Cells | Wall time estimate | What you learn |
|---|---|---|---|
| **Phase A — baseline** | S1 M− C1 | 1 × suite | Absolute performance ceiling, correctness baseline |
| **Phase B — MTP sanity** | S1 M+ C1 | 1 × suite | Does MTP help on this model at all? If no, skip M+ in later phases |
| **Phase C — concurrency scaling** | S1 M− C{2,3,4}(a) | 3 × suite | Single-machine throughput curve, decide where scaling stops |
| **Phase D — intra-scenario parallel** | S1 M− C{2,3,4}(b) | 3 × suite | Compare to (a) — reveals batching vs sub-agent dispatch behavior |
| **Phase E — multi-machine serial** | S2 M− C1 | 1 × suite | Cost of inter-node overhead per request |
| **Phase F — multi-machine concurrent** | S2 M− C{2,3,4}(a+b) | 6 × suite | The actual multi-Spark scenario |
| **Phase G — MTP × topology matrix** | remaining M+ cells | rest | Fills out the matrix if MTP showed value in Phase B |

Each phase feeds the next — Phase A/B results inform which cells in later phases are worth running. If MTP is a no-op for this model, don't spend hours filling out the 16 M+ cells.

## Deliverables

- **`reports/matrix/{model}/{cell}.json`** — raw scorecard per cell (spec §7 format)
- **`reports/matrix/{model}/summary.csv`** — one row per cell with key metrics, suitable for pivot tables
- **`reports/matrix/{model}/curves/`** — generated plots:
  - pass_rate vs concurrency (per topology, per MTP state)
  - p50_latency vs concurrency (per topology, per MTP state)
  - aggregate_throughput vs concurrency (per topology, per MTP state)
  - speedup_vs_baseline heatmap (all 32 cells)

## Open questions to record per run

- What's the crossover point where MTP stops helping?
- Does multi-machine EP scale linearly up to C4, or plateau sooner?
- Does Nemotron-Nano behave the same as Qwen3.6 under concurrency, or does its hybrid Mamba architecture scale differently?
- Does the Spark's 273 GB/s memory bandwidth become the bottleneck before network does, even on multi-machine?
- Are there correctness regressions under high concurrency? (context-editing/compaction races, tool-call interleaving bugs)

## Notes on multi-machine setup

Two Sparks connected via dual 200 GbE (ConnectX-7) gives ~50 GB/s inter-node. That's ~18× slower than single-node memory bandwidth (273 GB/s) but fine for A3B MoE if configured correctly.

**Use expert parallel**, not tensor parallel:

```
vllm serve <model> \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --distributed-executor-backend ray
```

(Exact flags depend on upstream state at test time — verify against current recipes.)

Expect **~20-40% loss** on per-request latency due to routing hops, but near-linear scaling on aggregate throughput under concurrency because each token's expert hops are small (< 1 KB each).

## Risks and caveats

- **Torch.compile cache invalidation across topology changes** — changing TP/PP/DP triggers full recompile. Budget extra time on first run of each new topology.
- **Prefix cache across topologies** — single-machine cache doesn't transfer to multi-machine. Cold-start cost is real per configuration.
- **Network noise on shared LAN** — other traffic on the 200GbE link (backups, large file transfers) will distort S2 timings. Isolate if possible.
- **Thermal throttling under sustained load** — Spark is designed for short bursts. A 4-hour sweep at concurrency 4 may throttle. Log GPU clock rate via `nv-monitor` and flag if it drops.
- **Statistical noise on small scenario counts** — 5 seeds per cell is minimum. If a finding is marginal, bump to 10 seeds before publishing.
