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
  **hikyaku** that routes client traffic to the right vLLM endpoint by
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

Currently deployed (as of 2026-05-06): vLLM `0.19.2rc1.dev4+gb5f6c5f83`
(image built 2026-04-18), torch 2.11.0+cu130, transformers 5.5.4.

**Known-bad: vLLM 0.20.** NVIDIA forum reports of severe quality
regression on Qwen3.6-27B BF16 and FP8 (eval scores dropping 93→44 /
100). Affected users downgraded to 0.19.2 — which is exactly the
version we're currently on. When it's time to rebuild, skip 0.20 and
go to 0.21+ (or whatever is then known-good). Re-test eval quality
on a representative coding task before promoting any new build.
**The 0.20 regression was prefix caching** (eugr root-caused it) — the
same bug class as the §below. Treat "Qwen quality regression on a new
vLLM" as prefix-caching-until-proven-otherwise.

### Prefix caching corrupts Qwen MoE output under concurrency (RECURRING)

**`--enable-prefix-caching` causes cross-request data corruption on
Qwen MoE models under concurrent load with shared prefixes.** Confirmed
on **0.23.1rc1.dev226** with `Qwen/Qwen3.6-35B-A3B-FP8` + MTP; the same
class of bug was the 0.20 quality regression above. This is a
**recurring** vLLM-prefix-cache × Qwen-MoE failure — assume any new
vLLM build is suspect until tested.

**Confirmed by a clean single-variable A/B** (only `enable_prefix_caching`
toggled, everything else identical — same image, gmu, MTP, DPI, N):
- **OFF:** N=1, N=4, N=8 produce *byte-identical* output, matching the
  single-thread 200-DPI ground truth. Deterministic and safe.
- **ON:** florid corruption at N=8 — and it's **state leakage, not
  benign nondeterminism**:
  - cross-*field*: a company name lands in the `currency` field, a date
    in the `currency` field, the literal schema word `date` as a value;
  - cross-*receipt*: request A's values appear in request B's output
    (one receipt got a different receipt's amount/vendor);
  - token insertion mid-value (`33.33.62`, `202026-01-19`, `48.0.00`).
  Mechanism: concurrent requests sharing prefix-cache KV blocks read
  each other's cached state.

**A fixed sampling seed makes NO difference** — corruption persists with
prefix-on even with a pinned seed. This rules out the sampling/RNG layer:
the bug is in KV/prefix-cache *state management* (cross-request block
contamination), which no seed can touch. (At temp 0 the seed is moot
anyway — greedy doesn't sample — so the point is doubly clean.) Don't
waste time chasing determinism via seeds/temperature; the only fix is
turning prefix caching off.

**Trigger profile:** maximal when many concurrent requests share a long
prefix (same system prompt + JSON schema) — exactly the shape of a
batch extraction/agent workload. Single-threaded is always clean.
Diverse-prefix concurrency is *less likely* to trip it but is NOT proven
safe (and at scale, diverse requests still share batches — the failure
just becomes invisible without a strict output gate).

**Fix:** `--no-enable-prefix-caching`. Costs ~30% on single-request
prefill (the shared prefix is reprocessed every request) but is fully
recovered under concurrency (N=8 throughput ≈ N=1-with-caching). For
this stack the correctness win dwarfs the prefill cost — **run perf
recipes prefix-caching-off until a vLLM build is proven clean.**

**Two durable lessons:**
1. **Validate `single == multi` as an acceptance gate** before shipping
   any concurrency. Field-diff N=1 vs N=8 output; non-identical (beyond
   benign value-divergence) = do not ship.
2. **You cannot validate your way out of a contaminating inference
   layer.** Per-field checks catch *malformed* values loudly but pass
   *valid-but-wrong* contamination silently (a leaked vendor/amount is
   still a valid vendor/amount). The defenses are: don't run the
   contaminating config, and add cross-request reconciliation/dedup.

### Tool-call parser: use `qwen3_coder`, not `qwen3_xml`

`qwen3_xml` looks like it improves things on Qwen3.6 — but Qwen
themselves recommend the latest `qwen3_coder` parser. We previously
tried `qwen3_xml` and it appeared to help, but the apparent
improvement was masking the real problem: KV cache running out of
space. The XML format produces shorter / differently-shaped tool
calls that fit where the real-format calls didn't. Once context /
KV sizing was fixed, `qwen3_coder` was correct.

Stick with `--tool-call-parser qwen3_coder` for all Qwen3.6 recipes
and treat `qwen3_xml`-helps-quality as a smell that points at a
context/KV problem upstream, not a parser problem.

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

### Trap: `HF_HUB_OFFLINE=1` + `--trust-remote-code` + custom-code models

A specific failure mode discovered the hard way:

1. `HF_HUB_OFFLINE=1` makes vLLM rewrite the HF model_id to a local
   snapshot path (`/root/.cache/huggingface/hub/.../snapshots/<sha>/`).
2. transformers' dynamic-modules loader caches custom Python code at a
   path *derived from how the model was loaded*:
   - by HF model_id → `transformers_modules/<owner>/<repo>/<sha>/`
   - by local path → `transformers_modules/_<sha>/` (leading underscore)
3. If you pre-populate the cache via `AutoConfig.from_pretrained(<hf_id>)`
   to seed the modules, but vLLM then loads via the rewritten local
   path, vLLM hits the `_<sha>/` directory which is **missing the
   relative-import files** (transformers' resolver doesn't fully
   populate it for local-path loads).
4. Symptom: `FileNotFoundError: ... transformers_modules/_<sha>/<file>.py`
   mid-startup, after the model SAFE-tensors load.

**Fix options:**
- Drop `HF_HUB_OFFLINE` for that model so vLLM keeps the model_id
  through the load chain (preferred).
- After pre-population, copy the populated `<owner>/<repo>/<sha>/`
  directory contents to `_<sha>/`.

This only bites with `--trust-remote-code` models. Stock-architecture
models load fine offline.

### Solo mode

`./run-recipe.py <recipe> --solo` skips Ray and node detection. Solo recipes
must NOT include `--distributed-executor-backend ray`.

### Stopping a cluster

```bash
./launch-cluster.sh -t vllm-node-tf5 --name <name> -n <ips> stop
```

This stops both the head and worker containers cleanly.

### Load balancing across replicas (DP-style)

When the same model name is served by multiple replicas (e.g. one MTP
endpoint on each Spark for capacity), the **hikyaku** layer distributes
requests with affinity-aware routing. Design owned by the hikyaku repo;
a reference copy lives at [`docs/LOAD-BALANCING.md`](./docs/LOAD-BALANCING.md).

Why it matters for serving here: vLLM's prefix cache is per-replica, so a
multi-turn coding session bouncing between backends pays full prefill
every turn. Proper sticky routing preserves cache hits and turns each
session's TTFT from seconds back to milliseconds. See `docs/LOAD-BALANCING.md`
§ "How the cache actually works" for the underlying mechanism, and the
phasing plan if you're implementing or extending it.

---

## 4. Models

### Active models

| Model | Quant | Weights | Multimodal | Notes |
|---|---|---|---|---|
| Qwen3.6-27B-FP8 | FP8 | ~28 GiB | yes (text-only via `--language-model-only`) | Dense, coding-focused, 256K context |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | AWQ-INT4 | ~14 GiB | text-only | Halves bandwidth → ~2× decode (see §5). **Use `max_model_len: 196608` (75% of 262K) and MTP `num_speculative_tokens: 2`**. Higher MTP causes output looping on real agentic workloads. |
| Qwen3.6-35B-A3B-FP8 | FP8 | ~34 GiB | text | MoE, 3B active params, ~3× decode of dense 27B at FP8 |
| Qwen2.5-VL-3B-Instruct | FP16 | ~6 GiB | yes (vision) | Small fast VLM. First-class vLLM support, no `--trust-remote-code`. |

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

### Model-suitability notes (independent of quant)

- **PaddleOCR-VL-1.5 on Spark: don't bother (as of 2026-05-19).**
  Architecture was merged into vLLM mainline on 2026-05-18, but no
  community success on Blackwell SM121a + aarch64 — DGX Spark forum
  posters who tried it reverted to Qwen VL models for PDF-parsing
  workloads, reported performance was "sufficient" for bulk
  transcription. Baidu's pre-baked container
  (`ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server`)
  is x86_64-only and almost certainly not compiled for sm_121a. For
  agentic-OCR (understanding-grounded — "what does this receipt
  mean?" rather than glyph transcription) use the existing
  `Qwen/Qwen3-VL-30B-A3B-Instruct` on spark-01:3041. For bulk
  transcription throughput, fall back to `Qwen2.5-VL-3B` or
  `Qwen3-VL-8B` on spark-02 (both first-class in vLLM, no patching).
  Revisit only if a specific OCR-throughput bottleneck materialises
  AND the upstream PR has matured AND someone has reported Blackwell
  + aarch64 success.

- **Qwen3.6-35B-A3B is not a coding model.** Tested 2026-05-16 across
  both Intel/Qwen3.6-35B-A3B-int4-mixed-AutoRound and the official
  Qwen/Qwen3.6-35B-A3B-FP8, with Qwen-recommended sampling enforced
  by hikyaku (temp 0.6, top_p 0.95, top_k 20, presence_penalty 0.1,
  repetition_penalty 1.05, MTP=2). On stressful coding workloads the
  35B "gets excited then drops into a tight loop"; the 27B FP8
  resolves the same prompts immediately. Same recipe shape, same
  sampling, same MTP setting — only the model differs. Rules out
  quant, sampling, and recipe wiring as causes; the 3B active params
  per token can't carry per-token decisions under coding pressure
  even when the wrapper is 35B. The 35B-A3B is still fine for
  simple/general agentic tasks (it ran openclaw productively for
  weeks), just don't put it in the coder slot. For coding use the
  27B FP8.

### VLMs that DON'T work in vLLM (don't reattempt without fresh evidence)

- **`apple/FastVLM-*`**: architecture `llava_qwen2` not in vLLM's
  registry; transformers' AutoConfig rejects it before vLLM gets a
  chance. Apple's research models typically need their reference repo
  for inference.
- **`vikhyatk/moondream2`** and **`moondream/moondream-2b-2025-04-14*`**:
  architecture `HfMoondream` not in vLLM's registry. Even with
  `trust_remote_code` and dynamic-modules cache populated, vLLM
  rejects at the architecture-match step. The 4-bit variant
  additionally pins to an old `torchao` API
  (`int4_weight_only`) that upstream torchao has renamed.

For small/fast VLM use, **`Qwen2.5-VL-3B-Instruct`** is the validated
working choice on this stack. Other vLLM-supported small VLMs:
SmolVLMForConditionalGeneration, PaliGemmaForConditionalGeneration,
Phi3VForCausalLM, InternVLForConditionalGeneration.

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
  **Caveat for coding workloads:** observed quality issues on coding
  tasks with this combination on our deployment — analysis suggested
  the DFlash draft head was trained more on prose than code (Sherlock
  Holmes novels figured prominently). Online reports vary; treat
  empirically per workload before committing. The current default for
  coding is MTP + AWQ-INT4 (~21 t/s) for that reason — the speed
  trade-off is intentional.
  **Practical-vs-benchmark gap (empirical, 2026-05-07):** on real
  interactive agentic work (planning + coding via pi coder), MTP felt
  *vastly* superior to DFlash despite DFlash's ~70% benchmark decode
  advantage — both on TTFT *and* sustained generation. The TTFT side
  is well-understood (2325 ± **2236** ms std dev ≈ mean, every turn
  pays the tax). The sustained-generation side is acceptance-rate-
  driven:

  Each spec-decode step generates `1 + accepted_tokens` at a cost
  of `target_forward + draft_overhead`:

  | method | step cost | accepted | tokens/step | tokens / unit cost |
  |---|---|---|---|---|
  | MTP (`num_spec=2`) | 1.0× | ~1.85 | 2.85 | **2.85** |
  | DFlash (`num_spec=15`) high-accept | 1.5× | ~7   | 8 | 5.3 |
  | DFlash low-accept                    | 1.5× | ~3   | 4 | 2.7 |
  | DFlash very-low-accept               | 1.5× | ~2   | 3 | 2.0 |

  **DFlash beats MTP only above ~25-30% acceptance rate.** Below that,
  the draft-head overhead and rejection rollback cost outweigh the
  speculative wins. DFlash's bimodal acceptance (huge ±2.4 t/s
  variance vs MTP's ±0.7) is the visible signature: workloads where
  the draft head was well-aligned with the content distribution land
  in the high-accept regime (60-80%); workloads with code, JSON,
  tool calls, structured output land in the low-accept regime
  (15-25%). Synthetic benchmarks (`llama-benchy --pp 512 --tg 256`)
  produce predictable token sequences that hit the draft head's
  training distribution; real coding/agentic content rarely does.

  Treat the speed table as upper bounds for batched decode on
  benchmark-shaped workloads, not as predictions of felt latency or
  sustained throughput in agentic workflows. To validate DFlash for
  any new workload, look at the `Avg Draft acceptance rate` in
  `[metrics.py:101] SpecDecoding metrics` log lines — if it sits
  below 30% on real traffic, MTP is the right choice.

  **Real-workload MTP acceptance, observed (2026-05-07):**
  Compaction-style task on both Sparks under MTP+INT4:
  - cyankiwi/Qwen3.6-27B-AWQ-INT4 + MTP, coding-heavy, smaller context: **84%**
  - Intel/Qwen3.6-27B-int4-AutoRound + MTP, design + code-review, larger: **78%**
  Both well into MTP-favouring territory; the 6-point delta is
  workload-shape (content type + context size), not a quant-quality
  signal. AutoRound's MTP heads work, acceptance is in the same band
  as cyankiwi's.

  **Quant quality verdict (2026-05-12):** despite AutoRound's general
  reputation as a higher-accuracy 4-bit method, on real coding work
  **cyankiwi's AWQ-INT4 made noticeably better decisions in stressful
  situations than Intel's AutoRound-INT4** — and this held even when
  the AutoRound endpoint was tuned harder (sampling params pushed via
  hikyaku). Quant-method reputation didn't translate to this model /
  this workload.

  **FP8 (the 8-bit step up) — strongly positive (2026-05-12):**
  `Qwen/Qwen3.6-27B-FP8 + MTP=2` produced an unambiguously better
  coding experience than either 4-bit quant: no large circular
  thinking blocks, every `edit` tool use landed first try (vs the
  4-bit endpoints where roughly every other edit failed and needed a
  different method), changes were more accurate up front, far less
  rework. MTP acceptance on FP8 also climbs with session depth —
  hit >90%, sustained ~83% as more context accumulates (established
  pattern → draft head predictions land more often → decode speeds
  up the longer you work). FP8's ~25-30% lower raw decode t/s is more
  than repaid by the drop in wasted tokens.

  ### Methodological note: tokens/s ≠ productivity

  Raw decode throughput measures the engine, not the system. When
  model quality differs, the two diverge sharply across at least
  three axes a t/s number is blind to:

  1. **Token efficiency.** A lower-quality model spends its budget on
     circular reasoning, failed tool calls (and the retry-with-a-
     different-method that follows), and rework on inaccurate changes
     — *higher* t/s, *lower* useful-work-per-minute. FP8 at ~15 t/s
     completes real coding tasks faster than the 4-bit endpoints at
     ~21 t/s because it generates the right tokens, not a lot of
     tokens.
  2. **Tool-call success rate.** On the 4-bit endpoints roughly every
     other `edit` tool use failed and needed a different method; on
     FP8, edits landed first try. Each failed tool call is a wasted
     round-trip plus the recovery turn.
  3. **Context-budget efficiency (compactions per unit of work).**
     Compaction is expensive — a large summarisation generation,
     fidelity loss (summary < full history), and a slow re-orientation
     turn afterwards. A worse model bloats the context window faster
     with the junk from (1) and (2), forcing more compactions.
     Observed: zero compactions in a day on FP8 vs frequent ones on
     INT4. (This is also why INT4 kept hitting the ~180K-input
     compaction wall — it filled context fast enough to keep slamming
     into it.)
  4. **Stability at deep context (the quality cliff).** INT4 quants
     visibly degrade — screwy output, hallucination, contradicting
     themselves — well *before* hitting the model's nominal context
     limit; we saw it on both cyankiwi AWQ-INT4 and Intel AutoRound-INT4
     at ~70-80% of the 262K window. FP8 stayed coherent at the same
     depth. Underlying mechanism is the same as the "stressful-situation
     decisions" observation: INT4 has less precision margin, and deep
     context is just another form of pressure (more accumulated
     rounding errors propagating through more attention passes).
     **Note:** the standard long-context benchmarks (NIAH, RULER, etc.)
     measure *recall accuracy* at depth — they don't catch this
     subjective degradation, which is the user-visible failure mode.
     Yet another way benchmarks miss the productivity-relevant thing.

  Implication for the PRO-003 perf baseline (hikyaku-pro): the
  headline number should not be tokens/s alone. Where feasible,
  measure on a fixed real task: time-to-complete, token *budget*
  consumed to reach a correct result, tool-call success rate, and
  number of compactions triggered. Tokens/s is a necessary input,
  not the deliverable.
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
- **MTP `num_speculative_tokens` is bounded by training. For Qwen3.6-27B,
  2 is the ceiling — 3 is empirically net-negative.** Tested FP8 + MTP=3
  on real coding workload (2026-05-12):

  | spec position | acceptance rate (real coding) |
  |---|---|
  | position 0 (1st draft) | ~0.6-0.9 |
  | position 1 (2nd draft) | ~0.3-0.7 |
  | position 2 (3rd draft) | **~0.15-0.30** |

  The 3rd position sits below the ~25-30% break-even where speculation
  pays for itself — it costs a full MTP-layer forward pass per step but
  yields a token only ~1 in 4-5. Net effects vs MTP=2 on the same
  workloads: avg draft acceptance dropped from 70-80% to ~35-70%
  (mean ~50%); mean acceptance length barely moved (2.0-3.1 out of
  max 4, vs 2.4-2.6 out of max 3); generation throughput *fell* from
  14-19 t/s to 9-15 t/s because step cost rose ~50% while accepted-
  token yield was flat. vLLM warns about this at startup:
  *"Enabling num_speculative_tokens > 1 will run multiple times of
  forward on same MTP layer, which may result in lower acceptance
  rate"* — believe it. The Qwen MTP heads are trained for 2-token
  lookahead; position 2 is essentially uncalibrated. Also: MTP=3
  previously caused thinking-loop behaviour on agentic workloads
  (the original reason prod sits at 2). Keep `num_speculative_tokens: 2`.

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
| `qwen2.5-vl-3b.yaml` | FP16 | none | flashinfer | Vision-language, 3B. Stock vLLM support, no `--trust-remote-code` |
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
- **TP=2 across two Sparks — MEASURED, and it beats single-node (2026-06-22).**
  Earlier guidance here said "single-node DFlash on AWQ-INT4 beats 2-node TP" —
  that was inferred from the DFlash/INT4 setup and is **wrong for FP8+MTP**.
  Clean A/B on Qwen3.6-27B-FP8 + MTP (same llama-benchy, pp512/tg256), vLLM
  0.23.1 image:

  | config | decode t/s | prefill t/s |
  |---|---|---|
  | single-node (1 Spark) | 15.5 | 690 |
  | **TP=2 (2 Sparks, Ray, RoCE)** | **20.3 (+31%)** | **1476 (+114%)** |

  So TP=2 is **net-positive** here: each node reads half the weights →
  faster bandwidth-bound decode (1.31× of the theoretical 2×; the CX7
  all-reduce eats the rest), and prefill gets ~2.1× (the all-reduce
  amortizes over the big batch). KV pool also balloons (split model →
  13.8× concurrency @ 262K at gmu 0.7; ~2.3× at gmu 0.30).

  **But TP=2 vs DP is a latency-vs-throughput choice, not "which is faster":**
  - **TP=2** = one *faster* pipeline (20.3 t/s single-request). Best **latency**.
  - **DP** (replica per Spark + sticky routing) = two pipelines, ~2× **aggregate**
    throughput (≈31 t/s), zero inter-node comms. Best **throughput**.
  Crossover is concurrency: TP=2 wins at low concurrency (1-2 users), DP
  wins as concurrent load climbs. For a small team weigh single-request
  latency vs total capacity. Reach for TP only when both nodes are free to
  dedicate; multi-node launch is `-n <ip1>,<ip2> --tp 2 -- --distributed-executor-backend ray`.
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
