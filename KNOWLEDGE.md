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
- Two ConnectX-7 200 GbE ports per node (only one cable connected). Each
  port is wired at 200 Gb/s but host-attached at **PCIe Gen5 x4 → ~100 Gb/s
  usable per rail**; see §2 for the bandwidth measurements and the
  two-rails-on-one-cable detail.
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

- **`limone`** (Jetson Orin Nano, on the tailnet) runs the
  **hikyaku** that routes client traffic to the right vLLM endpoint by
  model name. Proxy is no longer hosted on spark-01 itself.
- Tailnet: both Sparks are joined and routable by hostname (`paul@spark-02`).
- **Voice (TTS+STT)**: single container `clone-voice` on **spark-01:3030**
  (tailnet `100.118.152.61:3030`), OpenAI-compatible `/v1/audio/speech` +
  `/v1/audio/transcriptions`. TTS = **Chatterbox** (Resemble AI, MIT), STT =
  Whisper. Source/recipe: `~/hacking/clone-voice-service` (github
  wentbackward/clone-voice-service), run via `./run.sh`. Replaced F5-TTS
  2026-07-19 (F5 read long text crammed — no learned duration predictor;
  Chatterbox reads naturally, service sentence-chunks long text). Old F5 image
  kept as `clone-voice:f5-legacy`.

### Running non-vLLM ML packages on GB10 (sm_121a) — torch build recipe

Reusable pattern learned deploying Chatterbox + Whisper (applies to any
PyTorch package on the Spark). The GPU is **Blackwell sm_121a (cap 12,1)** and
needs **CUDA 13**; only the **cu130 aarch64** torch wheels have working
kernels for it.

- **Install torch FIRST from cu130, then the package `--no-deps`** if the
  package pins an older torch. Chatterbox pins `torch==2.6` (cu126, *no*
  sm_121a) — installing it normally silently downgrades torch to a
  CPU-fallback/incompatible build. Recipe:
  ```
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
  pip install -r requirements.txt        # deps EXCLUDING torch
  pip install --no-deps chatterbox-tts   # so it can't drag torch back
  ```
  Verify with a real CUDA op (not just `is_available()`):
  `torch.cuda.get_device_capability()` → `(12, 1)` and a `cuda` matmul.
- **CosyVoice 2 does NOT build here** (evaluated, rejected 2026-07-19): pins
  `torch==2.3.1`+cu121 (no sm_121a) *and* `tensorrt-cu12` / `onnxruntime-gpu`
  / `deepspeed`, all hostile to aarch64+CUDA-13. Would need deep dep surgery.
- Docker base for such services: **`nvidia/cuda:13.2.0-runtime-ubuntu24.04`**
  (arm64) + `build-essential cmake` (some deps like `praat-parselmouth`
  compile from source — a ~10 min LTO C++ build on ARM). Ubuntu 24.04 pip is
  PEP-668 externally-managed → `pip --break-system-packages`.
- `torchaudio.save` routes through `torchcodec` in ≥2.11 (not bundled) — write
  audio with `soundfile` instead (also handles ogg/mp3/flac, no ffmpeg).

---

## 2. Network specifics and gotchas

### Single cable, dual subnets

The two Sparks are connected with **one** CX7 cable on port 1
(`enp1s0f1np1`). NVIDIA's setup tool assigns IPs on **both** ports and on
both `192.168.200.0/24` and `192.168.100.0/24` subnets. As a result:

- `enp1s0f1np1` (cabled) gets BOTH `192.168.200.x/24` AND `192.168.100.x/24`
- `enP2p1s0f1np1` ALSO gets `192.168.200.x/24` and appears UP — **this is a
  REAL second PCIe rail, not a phantom** (a previous note here called it
  stale/uncabled; that was wrong — see the bandwidth subsection below). It is
  almost certainly the second x4 rail of the *same* physical cable, which is
  exactly why both ports land on `192.168.200.x` and trip autodiscover.

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

### THE CX7 CABLE MUST BE CONNECTED **BEFORE BOOT** (2026-08-01)

**If a Spark boots with no carrier on the CX7 and the cable is plugged in
afterwards, its RDMA TRANSMIT path is permanently degraded to ~13 Gb/s
until reboot.** Receive is unaffected. Nothing anywhere reports an error.

Measured on spark-01 (booted with spark-02 down, cable plugged in later):

| condition | spark-01 → spark-02 | spark-02 → spark-01 |
|---|---|---|
| cable hot-plugged after boot | **13.3 Gb/s** | 97.98 |
| after reboot with link live, MTU 1024 | 92.51 | — |
| after reboot with link live + jumbo | **97.98** | **97.98** |

**The one diagnostic that finds this fast: TEST BOTH DIRECTIONS.** The
asymmetry (full rate inbound, 7x slow outbound) is the signature and points
straight at the sending host. A single-direction test looks like a fabric
fault and sends you hunting the wrong things.

Everything below was checked and was NOT the cause — do not re-chase them:
cabling, switch config, bridge hardware offload, MTU/jumbo, FEC, PCIe link
width/speed, MPS/MRRS, NIC QoS/ETS/rate-limits, IOMMU, and packet loss
(every NIC and switch error counter read zero throughout). Also ruled out:
vLLM memory occupancy — 92.51 Gb/s with 88 GiB resident vs 90.52 with an
empty box, i.e. **models resident do not slow RDMA**, which is good news
for TP=2.

Mechanism (inferred): the mlx5 driver initialises against a dead port and
never fully recovers the transmit path on later link-up. DMA writes
(receive) are posted and stay fast; DMA reads (transmit) are non-posted and
latency-sensitive, which is why only one direction degrades.

**Operational consequence:** power-on order matters. Switch and cabling
must be live before the Sparks boot. If a Spark did come up first, a reboot
is the fix — and note `start-cluster.sh`'s loopback fallback (§11) exists
precisely because a Spark *can* boot with a dead fabric, so that path and
this trap go together.

### Switched fabric (CRS504-4XQ-IN) — costs essentially nothing (2026-08-01)

Replacing the direct DAC with a MikroTik CRS504-4XQ-IN gives **97.98 Gb/s
symmetric**, versus the 99.9 Gb/s direct-cable baseline. Wire rate drops
200 → 100 Gb/s (the CRS504 is a 100G switch) but that was never the limit —
PCIe Gen5 x4 was. Config that matters:

- Switch: `/interface/ethernet/set [find name~"qsfp"] l2mtu=9216`. A bridge
  clamps to the **lowest** member `l2mtu`, so missing a member silently
  defeats jumbo. Persisted; survives reboot.
- Hosts: `mtu: 9000` in `/etc/netplan/40-cx7.yaml` on BOTH netdevs of the
  cabled port (they are the same physical port — keep them in sync).
  Yields RDMA `active_mtu` 4096. Worth ~6% (92.51 → 97.98), no more.
- Management: switch is `192.168.88.1`. Without a directly-connected
  address in that subnet the route goes out the LAN default gateway and
  reaches the **CRS310** instead (both switches ship on 192.168.88.1).
  Netplan now pins `192.168.88.100` (spark-01) / `.101` (spark-02) on
  `enp1s0f1np1`, so both nodes can reach the CRS504 over the fibre.

**Switch-management trap:** with the RJ45 unplugged, the switch's ONLY
management path is the QSFP ports. Changing `l2mtu` on all of them at once
cut the CPU path and froze the session (data plane kept forwarding —
hardware offload does not need the CPU). Recovery: MNDP (`sudo mactelnet -l`)
still showed the switch, proving the CPU was alive, and the **second PCIe
domain's address on a different switch port still reached it**. Always run
RouterOS **safe mode (`Ctrl+X`)** before touching port config — it
auto-reverts on session drop, and did. Note `mactelnet` 0.4.4 (Ubuntu) cannot
authenticate to RouterOS ≥ 6.43 — it reports "incorrect username or
password" regardless of the real password; use SSH/WebFig by IP instead.

### CX7 bandwidth: PCIe Gen5 x4 per rail (~100 Gb/s), 200 nominal

Measured 2026-06-28 with RDMA `ib_write_bw` over RoCEv2 (after a "why is it
slow / why doesn't nv-monitor show traffic" investigation — the slowness was
*not* the fabric, see §5/§6):

- Each CX7 port negotiates **200 Gb/s on the wire** (ethtool / sysfs
  `speed=200000`), but its host attach is **PCIe Gen5 x4**
  (`current_link_width == max_link_width == 4` on all four functions — i.e.
  at capability, *not* degraded). Gen5 x4 ≈ 128 Gb/s raw → **~100 Gb/s
  effective RDMA per rail**: single-QP `ib_write_bw` = 99.9 Gb/s, and **8 QPs
  gave no more** (94.5), so ~100 is the real per-rail ceiling, not a
  tuning/CPU artefact.
- **RDMA bypasses the kernel network stack**, so RoCE traffic is INVISIBLE to
  `/proc/net/dev` and anything built on it (nv-monitor). During TP=2 decode
  the kernel counters read ~0 while the RoCE hardware counters showed
  ~49 MB/s. **Never diagnose fabric health from nv-monitor / kernel byte
  counters** — read `/sys/class/infiniband/<dev>/ports/1/{counters,hw_counters}`
  or run `ib_write_bw` instead.
- **Decode is latency-bound, not bandwidth-bound.** TP=2 token all-reduce
  moves KB-scale tensors; measured ~49 MB/s = <0.5% of one rail. So the
  fabric is never the inference bottleneck — GPU memory bandwidth
  (273 GB/s, §6) is. More network would not speed up decode at all.
- **CORRECTION (2026-08-01): there are NOT two rails — it is ONE physical
  port seen twice.** `enp1s0f1np1` (PCIe domain 0000) and `enP2p1s0f1np1`
  (domain 0002) report **identical `phys_switch_id` and identical
  `phys_port_name=p1`** — same physical QSFP port, two PCIe functions with
  different MACs. That is why both show `carrier=1` with only ONE cable
  plugged in. So the NVIDIA playbook's four addresses map to two physical
  ports (p0, p1), not four, and only p1 is cabled. This invalidates the
  "aggregate both rails for 200 Gb/s" idea below as written: you cannot
  aggregate a port with itself. ~100 Gb/s IS the ceiling for one cable.
- **Reaching the quoted 200 Gb/s / 25 GB/s "unit-wide lane limit" needs both
  x4 rails aggregated.** UNCONFIRMED here: a simultaneous 2-rail
  `ib_write_bw` summed to only ~107 Gb/s, but that test is contaminated —
  both rails sit on the same `192.168.200.0/24`, so kernel routing collapses
  both streams onto one rail (the same ambiguity that breaks autodiscover).
  A clean 2-rail test requires the rails on **separate subnets**. Only worth
  doing for BULK transfer (model loads, rsync over CX7) — it does nothing for
  latency-bound decode. Health/error counters on the active rail were all
  zero (no `port_xmit_wait`, no retransmits, no seq errors).

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

**Update 2026-06-28:** the live cluster now runs the **0.23 image
`vllm-node-tf5-v0231`** (vLLM `0.23.1rc1.dev226`) on every endpoint — see
§11 for the layout. The 0.19.2 `vllm-node-tf5` image is retained as a
fallback. Note: the FlashQLA mod / `mods/flashqla` recipe block does **not**
apply on 0.23 (its patch target `gdn_linear_attn.py` was relocated and it
aborts launch) — omit it from v0231 recipes; the dense 27B doesn't need it.

**Known-bad: vLLM 0.20.** NVIDIA forum reports of severe quality
regression on Qwen3.6-27B BF16 and FP8 (eval scores dropping 93→44 /
100). Affected users downgraded to 0.19.2 — which is exactly the
version we're currently on. When it's time to rebuild, skip 0.20 and
go to 0.21+ (or whatever is then known-good). Re-test eval quality
on a representative coding task before promoting any new build.
**The 0.20 regression was prefix caching** (eugr root-caused it) — the
same bug class as the §below. Treat "Qwen quality regression on a new
vLLM" as prefix-caching-until-proven-otherwise.

### `enable_thinking: true` SILENTLY BREAKS structured output (2026-08-05)

**If thinking is enabled, `response_format: {"type":"json_schema"}` is not
enforced. No error, no warning — you just get free-form prose back.** And
it is **INTERMITTENT**, which is worse than a clean failure: the same
request can return schema-exact JSON on one call and markdown on the next.

Measured on the receipts-extraction shape, schema using deliberately
unnatural field names (`zzq_vendor`, `xk9_amount`, `w3_ccy`) so that
enforcement is unambiguous — the model would never invent those itself:

| config | schema-exact | completion tokens |
|---|---|---|
| thinking **OFF** (27B, direct) | **3/3** | ~42 |
| thinking **OFF** (35B, direct) | **3/3** | 45 |
| thinking **OFF** (35B via hikyaku) | **5/5** | ~46 |
| thinking ON + budget 4096 (27B, direct) | **0/4** | 574-938 |
| thinking ON (35B, direct) | **0/3** | 567-825 |
| thinking ON (27B via hikyaku) | 2/3 | 620-689 |
| thinking ON + budget (35B via hikyaku) | 2/3 | 654-1161 |

**The reasoning parser is NOT the cause — ruled out 2026-08-05.** Obvious
suspect, since `StructuredOutputsConfig` names it and needs it to know where
reasoning ends. But `--reasoning-parser qwen3` works correctly and
consistently: over 7 runs the `reasoning` field was populated (1680-3278
chars) in **both** the grammar-applied and grammar-not-applied cases. The
parser separates thinking cleanly every time; the failure is in **grammar
resumption after the reasoning block**, matching `enable_in_reasoning=False`.
Changing the parser will not help.

- **Field-name gotcha:** this vLLM version returns thinking in
  **`reasoning`**, NOT `reasoning_content`. Checking the wrong field makes
  it look like the parser is broken when it is fine.
- **Three distinct failure modes**, not one — do not validate naively:
  1. content is prose, no JSON at all;
  2. content is valid JSON *followed by trailing junk* (passes a "starts
     with `{`" check, fails strict parsing — the dangerous one);
  3. structurally valid but semantically corrupted — with
     `additionalProperties:false` and no field for a value it wants to
     report, the model **smuggles it into another string**, e.g.
     `"w3_ccy": "USD,y7_date_issued_2026-03-11"`. Grammar guarantees
     structure, never sanity: include every field the model may want to
     emit, or tell it explicitly to discard extras.

**Removing `thinking_token_budget` does NOT fix it — tested 2026-08-05.**
The budget makes it worse, but thinking alone is sufficient to break
enforcement. Same schema, 27B:

| config | schema-exact |
|---|---|
| thinking OFF | 3/3, 5/5, 3/3 — always |
| thinking ON, **no budget** (direct) | 3/5 |
| thinking ON, no budget (via `gresh-coder`) | 1/5 |
| thinking ON, **budget 4096** | 0/5 |

Unbudgeted thinking gives a **coin-flip**, which is the most dangerous
outcome — it is the version most likely to pass a spot-check and fail
later. (n=5 each; treat as "thinking-on ~20-60%, thinking-off 100%" rather
than reading the individual ratios, and note the direct-vs-proxy gap at the
same settings is noise, not a hikyaku effect.)

**Scope — it is the vLLM build, not the model or the proxy:**

- Reproduces on the **dense 27B and the MoE 35B** alike.
- Reproduces **direct to vLLM and through hikyaku** identically, so a proxy
  is not stripping the field. (This was the initial suspicion and it was
  WRONG — hikyaku passes `response_format` through faithfully.)
- Mechanism is visible in the server's own config:
  `StructuredOutputsConfig(..., reasoning_parser='qwen3',
  enable_in_reasoning=False)` — the grammar is not applied while reasoning
  is active, and on this build it does not reliably resume afterwards.
- **`response_format: {"type":"json_object"}` still works with thinking
  on** — the looser "must be valid JSON" mode is unaffected. Only strict
  schema enforcement breaks. Note json_object does NOT give you *your*
  field names or types (it returned `company` / `"42.50 USD"` as a string
  where the schema gave `vendor` / `42.50` as a number).

**Cost, independent of correctness:** thinking-off answered in ~46 tokens;
thinking-on burned 570-1160 for the same extraction. Roughly 15-25x.

**Operational consequence: this is an either/or per route.** Thinking helps
tool-selection and reduces agentic looping; it silently breaks structured
output. There is no setting that gives both on this build, so split them by
route — a thinking route for agentic/tool work, a non-thinking route for
extraction. **Never send `response_format` to a thinking-enabled route.**

**Verified-good extraction config** (hikyaku `marvin-fast`, 5/5): backend
35B, `temperature 0.6, top_p 0.95, top_k 20, presence_penalty 0.1,
repetition_penalty 1.05, max_tokens 16384, enable_thinking: false`. Note
temperature did NOT need lowering — grammar-constrained decoding already
restricts the token space, so 0.6 is fine.

**How to test it properly:** use a schema with field names the model would
never choose on its own, and **run it at least 3-5 times**. A single
passing run proves nothing on a thinking route — "I tested it and it
worked" is exactly how this reaches production and then fails.

**Route audit 2026-08-05** — these have thinking ON and would fail if a
client ever sends a strict schema: `marvin`, `cortana`, `gresh-coder`,
`orchestrator`. Safe (thinking off): `marvin-fast`, `gresh-instruct`,
`gresh-general`, `worker`, `vlm`.

**DOES NOT EXTEND TO TOOL CALLS — do not over-generalise this.** Tested
2026-08-05 on the 27B, thinking ON (budget 4096) vs OFF:

| | tool-call args valid | tool chosen from 10 |
|---|---|---|
| thinking OFF | 6/6 | `remote_files` 8/8 |
| thinking ON | 6/6 | `remote_files` 8/8 |

Tool calling is unaffected in both **formatting and selection** — selection
was perfectly deterministic in both conditions. The reason is that they use
a **different mechanism**: tool calls are extracted from the output text by
`--tool-call-parser qwen3_coder`, whereas `response_format` uses guided
decoding with a grammar. Thinking breaks the grammar path only.

Consequence: thinking is cheap to keep enabled on agentic/tool routes; it
is only strict schemas that must avoid it.

### Prefix caching corrupts Qwen MoE output under concurrency (RECURRING)

**`--enable-prefix-caching` causes cross-request data corruption on
Qwen MoE models under concurrent load with shared prefixes.** Confirmed
on **0.23.1rc1.dev226** with `Qwen/Qwen3.6-35B-A3B-FP8` + MTP; the same
class of bug was the 0.20 quality regression above. This is a
**recurring** vLLM-prefix-cache × Qwen-MoE failure — assume any new
vLLM build is suspect until tested.

**STATUS 2026-08-03: STILL OPEN. An attempt to re-test it was INVALID —
learn from the mistake.** The 35B was run with `--enable-prefix-caching`
against a 51-receipt OCR harness with reconciliation vs known-good,
swept at n=4/8/12/16. Result looked perfect (51 OK / 0 WARN / 0 FAIL,
identical categories at every level) — but the prefix cache logged
**785,216 queries and ZERO hits (0.0%)**. The cache never served a
block, so cross-request contamination had no opportunity to occur. A
clean result under 0% hit rate is not evidence of anything.

- **Why 0%:** every receipt is a unique image, and vLLM matches prefix
  blocks from the **start of the sequence**. With the image early in the
  prompt, nothing ever matches across requests.
- **A VALID test needs a long SHARED prefix with the varying content
  LAST** — e.g. a big fixed system prompt then a short differing query —
  so the cache genuinely hits, then reconcile the outputs.
- **Do not be fooled by a high cumulative hit rate on another endpoint.**
  The dense 27B shows ~69-76%, but that is dominated by long-lived
  coding conversations; its OCR portion was almost certainly ~0% too.
  Always check the hit rate **for the workload under test**, not the
  process lifetime figure.
- **Dense models look unaffected** — the 27B has run prefix caching +
  MTP continuously under real concurrent coding load with no reported
  corruption. The documented bug is MoE-specific. Prefix caching and MTP
  are **not** mutually exclusive.
- **Cost when it does not hit:** pure overhead. On the OCR sweep the KV
  pool fell 66,464 → 53,600 tokens and every concurrency level slowed
  (n=4 81→114s, n=8 68→73s, n=12 64→66s, n=16 62→72s).

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
| Qwen3.6-35B-A3B-FP8 | FP8 | ~34 GiB | **yes (image + video)** | MoE, 3B active params, ~3× decode of dense 27B at FP8. Qwen3-VL-class — image AND video (no audio). See §video below. |
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

- **27B-FP8 does far superior work to 35B-A3B — it is just slower
  (2026-07-29).** Observed across sustained real use (openclaw coding
  *and* agentic editing: documents, writing scripts). This is the
  dense-vs-MoE trade in practice: the 35B-A3B activates ~3B params so
  it is fast, but the dense 27B's output quality is not close. Read it
  together with the §5 "tokens/s ≠ productivity" note — the 27B's
  lower t/s is bought back in fewer failed tool calls, less rework and
  less circular reasoning. **Default to the 27B for anything
  substantive; reach for the 35B only when latency matters more than
  quality, or for vision/video (the 27B is served text-only with
  `--language-model-only`, and DFlash's drafter is text-only anyway).**
  Caveat on "slower": this was observed with **thinking ON** (the
  hikyaku routes were not clamped `enable_thinking: false` until later
  on 2026-07-29). Thinking was inflating the token count per turn by a
  large factor, so the 27B's real latency penalty is smaller than it
  appeared. The quality verdict is unaffected.

- **Qwen3-VL-8B needs more input resolution + prompt work than the
  Qwen3.6 generation (2026-07-29).** Running the 8B as the spark-01
  vision slot (in place of a Qwen3.6-class model, which no longer fits
  beside the 262K coder) worked, but only after **increasing the DPI of
  the images passed in** and **tweaking the prompts**. Head-to-head the
  newer Qwen3.6 model was clearly superior on the same task. So the 8B
  is a capacity-driven compromise, not a quality-neutral swap: budget
  for higher-resolution inputs and per-task prompt tuning, and prefer a
  Qwen3.6-class VLM (or `Qwen3-VL-30B-A3B`) whenever memory allows —
  e.g. once spark-02 is free to host it.

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

  **UPDATE 2026-07-20 — some of that "just loops" was CONTEXT POLLUTION,
  not raw ceiling.** The 35B-A3B recipe was still running
  `preserve_thinking: true` (never got the 27B's fix), so old `<think>`
  blocks accumulated in context every turn; combined with a long uncleared
  openclaw session and near-greedy sampling (temp 0.4 / top_p 1.0 in
  hikyaku), it degraded into message-level repetition loops (emitting an
  acknowledgement over and over, never acting). Fixes: recipe now
  `preserve_thinking: false`; hikyaku sampling to Qwen thinking values
  (temp 0.6 / top_p 0.95 / top_k 20) + a `thinking_token_budget` (~1024) in
  `extra_body`. Verified the raw model emits correct `tool_calls` on a clean
  single-turn request (it read the docs and even inferred an undocumented
  config key). After the fixes **+ a `/new` (clean context)**, it correctly
  drove an agentic file-edit in openclaw. So: the model IS capable of
  simple technical agentic work; **Qwen degrades sharply on polluted/long
  context — context hygiene is a first-class requirement** (auto-summarize
  then `/new`; abort loops early). Caveats: still over-confirms (mild
  repetition), and remains LESS robust than the dense 27B under context
  stress — for hard/long agentic coding the 27B FP8 is still the pick.

### Qwen3.6-35B-A3B-FP8 multimodal: image + video, LIVE-VERIFIED (2026-07-20)

The 35B-A3B is a **Qwen3-VL-class** model — its config has both
`image_token_id` (248056) and `video_token_id` (248057), a `vision_config`,
and `Qwen3VLProcessor` + `video_preprocessor_config.json`. On spark-01:3040
we run it **without** `--language-model-only`, so the vision tower loads.
Verified live on the running FP8 endpoint:

- **Image**: sent a 1×1 PNG → answered "Pink" correctly.
- **Video**: sent a 2-frame mp4 (green→yellow, base64 `video_url`) → answered
  "green and yellow" correctly. So it genuinely decodes/understands *frames*,
  not just accepts the upload.
- **Audio: NO** — no audio encoder / no `audio_config` (that's what the
  clone-voice Whisper STT on :3030 is for).

**How to call it** (OpenAI chat/completions):
- image: content part `{"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}`
- video: content part `{"type":"video_url","video_url":{"url":"data:video/mp4;base64,…"}}`
  (or an `http(s)://` URL).

**Two gotchas that cost time — set these or you get empty output:**
1. **Disable/cap thinking for VLM turns.** First attempts returned
   `content:null, finish_reason:"length"` because reasoning ate the whole
   token budget. Pass `extra_body={"chat_template_kwargs":{"enable_thinking":
   false}}` (and/or a small `thinking_token_budget`), or give generous
   `max_tokens`. With thinking off/capped it answered cleanly.
2. **Frame/context budget.** Video expands to many image tokens fast — long
   clips eat the 131K window + KV. Fine for short clips; sample frames for
   anything long. (The endpoint had no `--limit-mm-per-prompt` set, so vLLM
   defaults apply.)

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

### DFlash on 35B-A3B + vLLM 0.23: BLOCKED on unmerged upstream (2026-06-23)

The latest `z-lab/Qwen3.6-35B-A3B-DFlash` draft (snapshot `f181eece`,
~1.7 GB, pulled 2026-06) **requires drafter sliding-window-attention
support from vLLM PR #40898, which is not merged.** Findings:

- **Latest draft crashes** on the v0231 build (0.23.1rc1.dev226) with
  `AssertionError` in EngineCore init (no drafter-SWA support).
- **Old draft `42d3b34d` (905 MB) loads** but runs **degraded** — SWA
  layers in the drafter run as full attention → poor acceptance,
  especially long-context. Usable for a *stability*-only test, not a
  representative *speed* test.
- **Can't patch it in:** both `--apply-vllm-pr 40898` and its temp
  cherry-pick `--apply-vllm-pr 44807` **fail the build's `git merge`
  against current `main`** — #40898 conflicts in 4 files
  (qwen3_dflash.py, scheduler.py, dflash.py, gpu_model_runner.py);
  #44807 conflicts in gpu_model_runner.py. Both are stale drafts
  (needs-rebase). Hand-resolving = custom patch on core engine
  internals that breaks every vLLM bump — not worth it.
- **Verdict:** park DFlash on the 35B until #40898 (or successor)
  rebases/merges into a build we run, then `--apply-vllm-pr` works
  trivially. Meanwhile **MTP-off is the stable 35B answer** (see §5).
- **Tracking:** umbrella feature ticket
  [vllm#46105](https://github.com/vllm-project/vllm/issues/46105)
  (subscribed 2026-06-23). Revisit DFlash when it lands — rebuild
  with `--apply-vllm-pr` against a build that includes it, re-pull the
  latest `z-lab/Qwen3.6-35B-A3B-DFlash` draft, and run the agentic
  *stability* test (does the separate-draft path avoid the
  native-MTP-on-MoE loops?).

(Mechanics note: the build applies PRs via `git merge pr-NNN`, so a PR
must merge cleanly into the `VLLM_REF` base — a conflicted PR fails fast
at build step ~#14, within ~1 min. Launch long remote builds via
`tmux new-session -d` — nohup-over-ssh kept dying.)

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
  **REAL-WORKLOAD DFlash acceptance, measured 2026-07-29** on
  `Qwen/Qwen3.6-27B-FP8` + `z-lab/Qwen3.6-27B-DFlash` (May-2026 draft
  head, vLLM 0.26.1 + PR #47914, `num_speculative_tokens: 15`,
  spark-01 solo, 262K). Both readings are **passive `/metrics` counter
  deltas over the user's own traffic** — no synthetic load added.
  **IMPORTANT: both were taken with THINKING ON** — the hikyaku routes
  were only clamped to `enable_thinking: false` later on 2026-07-29,
  after these numbers were captured. Treat them as *thinking-on*
  baselines; thinking output is prose-shaped, which is this drafter's
  weak regime (see the old prose/code note above), so thinking-off
  acceptance should be **higher**. Not yet re-measured:

  | workload (THINKING ON) | acceptance | accepted/step | notes |
  |---|---|---|---|
  | **coding** (openclaw) | **33.3%** | 5.00 | just above the MTP break-even |
  | **agentic editing** (documents, writing scripts) | **16.9%** | 2.53 | **below** break-even → MTP likely better |

  Agentic-editing profile for context: 122 requests, **35,752 prompt
  tok/req → 578 output tok/req (62:1)**, mean TTFT 8.4 s, mean e2e
  38.0 s, prefix-cache hit rate 77.7%, implied decode **~19.5 t/s**.
  Note that is only *parity* with the old two-node TP=2 + MTP (20.3
  t/s) — DFlash on one node matched two nodes, it did not beat them.
  Per-position decay was steep: 74.5% accepted at pos 0, 1.5% by pos
  13, i.e. drafting 15 to land ~2.5.

  **Workload shape moves acceptance by ~2x, so record which workload a
  number came from.** At 16.9% the economics favour MTP: DFlash 3.53
  tok/step ÷ 1.5x step cost = **2.35** tok per unit cost, vs MTP
  num_spec=2's **2.85** — and §5 records 27B-FP8 + MTP sustaining
  ~83% (peaking >90%) as session depth grows, the opposite of
  DFlash's behaviour here.

  ### VERDICT: MTP BEATS DFLASH — A/B RESOLVED (2026-08-02)

  The A/B was run. **MTP wins decisively on real traffic and DFlash has
  been retired from the coder.**

  | | DFlash (thinking off) | **MTP** |
  |---|---|---|
  | acceptance, real agentic/coding | 18.9% | **85.5%** (281 reqs) |
  | acceptance, receipts OCR | — | **96.8-97.1%** |
  | decode | 21.7 t/s | **28.2 t/s** |
  | mean TTFT | 5.83s | **5.24s** |

  Thinking-off did NOT rescue DFlash (16.9% → 18.9%), leaving it below
  the ~25-30% break-even where its ~1.5x drafter overhead stops paying.
  MTP's built-in heads are effectively free by comparison.

  **MTP also unblocks vision.** DFlash's drafter is text-only and forces
  `--language-model-only`; MTP does not. One endpoint can therefore serve
  coding, agentic work AND vision/video — which is what allowed the
  separate VL-8B to be retired entirely (§4).

  **And it allows fp8 KV to be dropped.** DFlash needs non-causal
  attention, so it required `flash_attn`, which rejects fp8 KV — a
  constraint that disappears with MTP. Combined with TP=2 the 27B went
  from 1.07x to 2.46x concurrency at the full 262K window.

  Historical note: earlier synthetic benchmarks made DFlash look strong
  (up to 36 t/s). Those used `llama-benchy`-style predictable token
  sequences that suit a draft head; they do not predict real workloads.
  Trust `/metrics` deltas over real traffic.

  **DO NOT benchmark this with synthetic prompts — they overstate
  DFlash badly.** Toy code/JSON/prose snippets on an idle box gave
  36-67% acceptance and 26-38 t/s, i.e. 2-4x the real-traffic figures
  above, and inverted the content ranking. Short prompts miss the
  long-context degradation that dominates real sessions. Use passive
  `/metrics` deltas over real traffic instead. (An earlier revision of
  this section reported those synthetic numbers as if they overturned
  the prose-bias claim above — they did not.)

  **Corollary — `enable_thinking: false` is worth a LOT**, but for
  token count, not rate: thinking does not lower t/s (every token
  costs the same), it inflates the number of tokens generated, which
  is what dominates felt latency. Measured on 27B-FP8, prompt "What is
  2+2? Answer with just the number.":

  | request | completion tokens |
  |---|---|
  | `chat_template_kwargs: {enable_thinking: false}` | **2** |
  | no kwargs (model default = thinking ON) | **146** |
  | **top-level** `enable_thinking: false` — SILENTLY IGNORED | **170** |

  **The trap: `enable_thinking` must go inside `chat_template_kwargs`.**
  Passed as a top-level request field it is accepted and ignored, so
  thinking stays on and nothing warns you. Verified that hikyaku's
  per-route `clamp: enable_thinking: false` DOES translate correctly
  (matches the 2-token result). Also note `thinking_token_budget` is
  **rejected outright by the V2 model runner** the v0728 image uses, so
  routes must not send it.

  **This invalidated a round of measurements (2026-07-29):** the two
  acceptance figures above, and the "27B takes a lot longer" note in
  §4, were all captured while thinking was unknowingly ON. Before
  trusting any DFlash-vs-MTP or latency comparison, confirm thinking
  state first — check `usage.completion_tokens` on a trivial prompt
  (expect ~2, not ~150).
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
  (15-25%). **(Still broadly right on real traffic with the May-2026
  draft head: measured 16.9% on agentic editing and 33.3% on coding —
  see the real-workload table above. Synthetic snippets suggest much
  higher numbers; don't trust them.)** Synthetic benchmarks (`llama-benchy --pp 512 --tg 256`)
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
- **MTP `num_speculative_tokens=3` is WORKLOAD-DEPENDENT, not a flat
  ceiling** (revised 2026-06-29 — see the UPDATE at the end of this bullet).
  The 2026-05-12 test below found it net-negative on *that* workload; a
  2026-06-29 re-test found it net-POSITIVE on the current TP=2 coder. Both
  results stand — **per-position acceptance is the deciding variable.**
  Original FP8 + MTP=3 test (2026-05-12, real coding workload):

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
  (the original reason prod sat at 2).

  **UPDATE 2026-06-29 — MTP=3 re-tested on the TP=2 FP8 coder with prefix
  caching ON: NET-POSITIVE here; the 2026-05 result was workload-specific.**
  Measured per-position acceptance on predictable structured code output:
  **0.86-0.97 / 0.61-0.89 / 0.49-0.83** (position 0/1/2) — the 3rd draft
  runs **49-83%**, far above the ~25-30% break-even (vs 0.15-0.30 in May).
  Mean acceptance length 3.0-3.7 (of max 4) vs MTP=2's 2.6 (of max 3);
  throughput **23.5 t/s vs MTP=2's 22.8** (flat-to-better, NOT the ~30%
  drop seen in May). The deciding variable is acceptance: on
  high-predictability coding (deep context + prefix cache) the 3rd draft
  pays; on lower-acceptance content it reverts to the May picture, so this
  is not a blanket "always use 3." **Caveat:** the OTHER MTP=3 risk —
  thinking-loops on real *agentic* runs — is behavioural and won't show in
  synthetic throughput tests; watch for it in live use (now partly
  mitigated by `preserve_thinking=false` on this endpoint, set the same
  day). Prod `qwen3.6-27b-fp8-mtp-vlm` (the TP=2 coder) runs
  `num_speculative_tokens: 3` as of 2026-06-29.

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
| 35B-A3B-FP8 | 0.38 | ~46 GiB | ~5.7 GiB (see floor note below) |

### 35B-A3B-FP8 memory floor — measure, don't trust old KV snapshots (2026-07-07)

Trimming the 35B's gmu to free memory for a co-located ComfyUI/Flux.2
node ran into a hard floor. Measured on a **clean** spark-01 (117 GiB
free, nothing else on GPU), vLLM 0.23 image, max_model_len 131072,
max_num_seqs 4, fp8 KV:

| gmu | budget | Available KV | Concurrency @131K | Result |
|---|---|---|---|---|
| 0.32 | 38.9 GiB | **−1.4 GiB** | — | **FAILS** (no cache blocks) |
| 0.38 | 46.2 GiB | 5.72 GiB (566K tok) | **4.32×** | serves; ~min for 4 slots |
| 0.40 | 48.7 GiB | ~8.3 GiB (est.) | ~5× | — |

- **Hard floor ≈ 40 GiB**: 34.23 GiB weights + ~6 GiB fixed overhead
  (activation + non_torch + ~0.56 GiB cudagraph). This is independent of
  gmu, so the model **cannot run below ~gmu 0.34** — below that KV goes
  negative and the engine aborts with "No available memory for the cache
  blocks". Practical minimum for 4 full-context slots is **gmu 0.38**.
- **Only ~2.4 GiB is reclaimable** by dropping 0.40→0.38. To free more you
  must cut `max_model_len` (KV per full seq ≈ 1.35 GiB at 131K); e.g. 65K
  at 4× needs ~2.7 GiB KV vs 5.7 at 131K, worth ~3 GiB more — but clips
  long document-processing turns.
- **The `non_torch` floor drifts between boots.** A 2026-07-06 snapshot of
  this exact recipe at gmu 0.40 reported **15.97 GiB KV / 12.11×** — NOT
  reproducible on 2026-07-07 (floor was ~8 GiB higher). The 12× figure was
  stale and led to a wrong "free ~10 GiB" estimate. **Always read the live
  `Available KV cache memory` / `Maximum concurrency` lines from the target
  launch's log — never size off a previous boot's numbers.**
- **Buffer-cache flush is NOT the fix here.** `drop_caches` (the UMA quirk,
  §quirk) was ruled out — flushing 67→0.8 GiB buff/cache did not change the
  negative-KV result. The floor is real process memory, not cache.
- To read the numbers: `docker logs <ctr> 2>&1 | grep -iE 'Available KV
  cache memory|Maximum concurrency'`.

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

### KV CACHE DTYPE: use `auto` / 16-bit on GB10, NEVER fp8 (2026-08-02/03)

**`--kv-cache-dtype fp8` on these Qwen FP8 checkpoints runs with
UNCALIBRATED attention scales and is a likely cause of long-context
looping.** The checkpoints ship no attention scaling factors, so vLLM
falls back to `q_scale`/`prob_scale` of 1.0 and logs *"Using uncalibrated
q_scale ... This may cause accuracy issues"*.

- **Evidence it mattered:** after switching the 27B coder to 16-bit KV it
  handled complex agentic edits at **70.4% of a 262K context** without
  looping — the failure mode that had plagued deep sessions. The 27B
  DFlash config, which was forced to bf16 KV (flash_attn rejects fp8),
  never showed the problem either. Both 27B and 35B recipes moved to
  `auto` on 2026-08-02.
- **The cost is far smaller than it looks: ~27%, not 50%.** Qwen3.6 uses
  a HYBRID attention layout, so most layers hold little KV. Measured on
  the 27B: KV pool 686,400 → 483,200 tokens, concurrency 9.59x → 7.05x.
  Cheap insurance for correct attention numerics.
- Zero `uncalibrated` warnings in the log is the check that it took.

### Sizing rules that actually hold (2026-08-03)

- **`gpu_memory_utilization` does NOT bound peak memory.** It reserves
  weights + KV; **activation spikes come on top**, and image tokens +
  long prompts + MoE expert routing make those spikes large. Two crashes
  in one day came from sizing to idle headroom and then driving real
  load through both models at once.
- **vLLM profiles memory AFTER loading weights**, so the ~35 GiB of
  safetensors it just read is sitting in page cache when it sizes KV.
  Dropping caches beforehand helps only marginally (measured +10%,
  65,392 → 71,824 tokens) because the load refills the cache itself.
- **vLLM/CUDA sees `MemFree`, not `MemAvailable`.** Page cache therefore
  *looks* like used memory to the startup check — e.g. 26G "available"
  but only 19.15G visible to CUDA, causing a hard startup failure. Never
  size a launch from `free -h`'s available column.
- **KV sizing is NOT reproducible run-to-run.** Identical gmu gave
  65,392 / 71,824 / 75,040 / 66,464 tokens on successive 35B launches
  (~±9%). Do not tune off a single measurement, and re-read the actual
  `Maximum concurrency` line after every launch.
- **`max_num_batched_tokens` is NOT a free knob** — it sizes activation
  buffers, so raising it takes memory from KV. On the 35B, 16384 → 32768
  halved the KV pool (65,392 → 35,376 tokens, 1.85x → 1.01x) and did
  **not** improve prefill (3.61s → 3.77s). Reverted.
- **`max_num_seqs` IS nearly free** — a scheduler limit, not a memory
  reservation. Raising it 4 → 8 → 16 cost no measurable KV.

### Swap: turned OFF on both Sparks (2026-08-03)

`vm.swappiness=1` plus **swap disabled entirely** (`swapoff -a`, fstab
line commented, `systemctl mask swap.img.swap`). Rationale: a
recoverable OOM-kill beats an unrecoverable hard hang.

- **Swap did not prevent the crashes** — spark-01 died at 99% memory
  having used only "the tiniest bit" of swap. Swappiness sets the
  anon-vs-pagecache reclaim ratio; it does not create memory.
- **Swapping vLLM is catastrophic anyway:** 1.3 GiB of engine memory
  paged out **halved 27B decode, 28.2 → 12.4 t/s**, with no error
  anywhere. `VmSwap` in `/proc/<pid>/status` is the tell.
- **TRAP — `vm.swappiness` set via `tee -a`:** the documented setup step
  appended to `/etc/sysctl.conf`, so re-running it left THREE conflicting
  entries (`=1` at line 7, `=10` at 67 and 68). Last-wins, so every boot
  came up at 10 despite the intended 1. Edit in place; verify with
  `grep -c '^vm.swappiness' /etc/sysctl.conf` returning 1.

### NVRM unified-memory OOM hard-hangs the node — not thermal (2026-07-12)

spark-01 "died" during heavy ComfyUI/Flux.2 rendering co-located with the
35B. **It was memory exhaustion, not thermal** — an important distinction
because the two have opposite fixes:

- Kernel log before the hang: repeated
  `NVRM: ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal @
  mem_desc.c:1359`, and **zero thermal-trip / critical-temp messages**. GPU
  was ~70 °C with no throttle on the next boot.
- **The Linux OOM-killer does NOT fire** on this class of failure — the NVRM
  (GPU driver) allocator fails on the unified pool and the **whole node
  hard-hangs and must be power-cycled** (it did not auto-reboot; ~30 min
  down until manual power-cycle). So there's no graceful process kill to
  save you — overcommitting the 122 GiB unified pool takes the box down.
- **Watch `free -h`, not temps**, when stacking Comfy/Flux.2 on vLLM. Keep
  total committed under ~110 GiB. Two concurrent Flux.2 pipelines on top of
  the 35B is the danger zone.

**STILL PRESENT after the 2026-08-02 firmware upgrade.** Both Sparks were
taken to EC `0x03000508` and UEFI/SoC `0x02009b0b` (from `0x03000302` /
`0x0200980f`), plus kernel `6.17.0-1029-nvidia` and driver `580.173.02`,
with a full cold power-cycle. **It did not fix the behaviour** — spark-01
still died at 99% memory utilisation running 27B TP=2 + 35B + clone-voice
under a receipts workload, and again needed a power-cycle. Do not assume
newer firmware removes the need to size conservatively.

- **Recovery is now remote:** smart plugs on both Sparks, the CRS504 and
  limone. A hard hang costs a power-cycle, not a trip to the machine.
- **POWER-ON ORDER MATTERS:** bring the **CRS504 up first and let it
  finish training its ports (~60-90s) BEFORE the Sparks**. A Spark that
  boots with a dead CX7 keeps a degraded RDMA transmit path (~13 Gb/s vs
  ~98) with no error reported anywhere — see §2. `start-cluster.sh` now
  logs a loud warning block if it sees no carrier at boot.
- **Two models of this size do not co-exist on one node under load.**
  27B TP=2 (~44 GiB) + 35B (~44 GiB) + clone-voice (9 GiB) + system left
  ~11 GiB at idle, which real traffic consumed. The working layout is one
  large model per node (see §11).

**ComfyUI memory guards on the unified box (start-comfyui.sh):**

- `--reserve-vram N` is **relative to CUDA-free memory, not an absolute
  cap.** ComfyUI reads real free memory via `torch.cuda.mem_get_info`, which
  on the Spark correctly excludes vLLM's allocation — but it only tells Comfy
  "use up to `free − N`". It does **not** bound Comfy's total, and it does
  not react to a co-tenant (vLLM KV cache) growing *after* Comfy sized
  itself.
- **The trap:** killing 27B to "free memory for Comfy" *raises* Comfy's
  ceiling — free jumps, so Comfy grabs more — same collision. The apparent
  safety of a low `--reserve-vram` earlier was an accident of the other
  models having already squeezed the pool.
- **`--vram-headroom N`** (ComfyUI's DynamicVRAM, `--enable-dynamic-vram` is
  default-on) is the correct guard: keeps N GiB **continuously free even
  counting other apps**. Confirm it took: the log prints
  `comfy-aimdo integrated Linux GPU RAM headroom: <N*1000> MB`.
- **Deployed 2026-07-12:** spark-01 `--reserve-vram 16 --vram-headroom 8`
  (caps Comfy ~44–52 GiB alongside the 35B); spark-02 `--reserve-vram 10
  --vram-headroom 8`. Per-run overrides: `COMFY_RESERVE_VRAM`,
  `COMFY_VRAM_HEADROOM`.
- Reboot note: the boot chain relaunches 27B TP=2; Comfy is NOT in boot. If
  you reboot mid-render, re-kill 27B before resuming heavy Comfy.

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

### Receipts OCR harness — the most useful benchmark we have (2026-08-03)

A 51-receipt PDF→structured-extraction batch with **reconciliation against
known-good results**, driven at configurable client concurrency. Unlike
synthetic benchmarks it measures **quality and speed together** on a real
workload, and it has already settled questions that pure timing could not.
Prefer it for any config change that could affect output correctness.

**Headline result — the MoE 35B beats the dense 27B on BOTH axes:**

| model | best time (51 receipts) | per receipt | notes |
|---|---|---|---|
| **35B-A3B (MoE, 3B active)** | **62s** @ n=16 | 1.2s | also more accurate |
| 27B (dense) | 242s @ n=8 | 4.7s | ~4x slower, slot-limited at n=16 |

~4x is architectural: the dense 27B activates all 27B params per token
while the 35B-A3B activates ~3B. **This holds even though the 35B needs
more input resolution** (see DPI below), i.e. it wins while processing
~2.25x more image tokens.

**Accuracy:** the 35B correctly classified two domain-renewal receipts as
`ai_software_cloud` where the 27B said `other` — adjudicated as better by
reconciliation. Both scored 51 OK / 0 WARN / 0 FAIL.

**Minimum DPI for accurate extraction** (an accuracy threshold, not a
preference — below it, results degrade):

| model | min DPI |
|---|---|
| 27B (dense) | **100-125** — most token-efficient |
| 35B-A3B | 150 |
| Qwen3-VL-8B (retired) | 150, some retries at 175 |

Image tokens scale with DPI² so 100 → 150 is ~2.25x the tokens: the 27B
fits far more receipts per context window, which matters for large batch
reconciliation even though it is slower per receipt.

**Concurrency sweep (35B, `max_num_seqs=16`):** n=4 81s | n=8 68s |
n=12 64s | n=16 62s. Throughput scales to ~n=16 with diminishing returns
(6% then 3%). **Watch for the server-side cap** — an earlier sweep
plateaued at 68s purely because `max_num_seqs` was 8; the flat spot was
configuration, not hardware. Confirm `mean queue time ≈ 0` in
`/metrics` before concluding you have hit a physical limit.

**MTP acceptance on OCR is exceptional: 96.8-97.1%** (35B, num_spec=2).
Receipt output is highly predictable, so the draft heads land nearly every
token — a large part of why the 35B is so fast here. Compare ~19% for
DFlash on agentic coding (§5).

**Steady-state ceiling:** ~62s / 51 receipts ≈ **2,540 prompt tok/s** on
one Spark. More Sparks would give more parallel batches; only faster
silicon shrinks a single batch.

### Standard tools

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

## 11. Boot and reboot-recovery (persistence)

The cluster comes back after a reboot or power loss via **one idempotent
script**, not a pile of per-model systemd units. The old per-model design
drifted stale and mis-recovered the 2026-06 power loss (it relaunched a
retired AWQ-INT4 35B on the wrong port). Consolidated 2026-06-28.

**Single source of truth: `~/admin/start-cluster.sh`** (host-local on
spark-01, not in this repo). It defines the whole layout and is safe to run
by hand or at boot:

**CURRENT RUNNING LAYOUT (2026-08-03)** — one large model per node, after
two same-day crashes caused by co-locating both on spark-01:

| host:port | model | role |
|---|---|---|
| spark-01:3042 (head) | 27B-FP8 + MTP=3, **TP=2 across both Sparks**, 262K, gmu 0.36, 16-bit KV, vision, prefix-caching ON | coding + agentic |
| spark-02:3040 | 35B-A3B-FP8 + MTP=2, 131K, gmu 0.38, 16-bit KV, vision, `max_num_seqs` 16 | fast general / VLM / compaction |
| spark-01:3030 | clone-voice | own restart policy, not in the script |
| ~~spark-02:3043~~ | ~~4B utility~~ | **RETIRED** — the 35B replaced it, far more capable |

**`start-cluster.sh` does NOT yet boot this layout** — it still launches
the 4B and knows nothing about the 35B. Update it before relying on a
reboot to restore service.

**Firmware/OS baseline after the 2026-08-02 upgrade** (both nodes
identical): kernel `6.17.0-1029-nvidia`, driver `580.173.02`, EC
`0x03000508`, UEFI/SoC `0x02009b0b`, CX7 firmware `28.45.4028`
(NOT changed by the upgrade), Ubuntu 24.04.4, docker 29.2.1. Verify with
`fwupdmgr get-devices`; **EC updates require a full cold power-cycle**,
not a warm reboot. Netplan (MTU 9000 + `192.168.88.100/.101` switch
management) and `vm.swappiness=1` all survive reboots correctly.

**Layout REVISED 2026-07-29** (superseded by the above):

| host:port | model | how |
|---|---|---|
| spark-01:3042 | 27B-FP8 + **DFlash**, **solo** (not TP=2), 262K, text-only | recipe `qwen3.6-27b-fp8-dflash` `--solo`, gmu 0.61, image `vllm-node-tf5-v0728` |
| spark-01:3041 | Qwen3-VL-8B-Instruct-FP8 vision slot, 16K | recipe `qwen3-vl-8b-fp8` `--solo`, gmu 0.14 |
| spark-02:3043 | 4B-Instruct-2507-FP8 utility, 2048 | recipe `qwen3-4b-instruct-2507-fp8` `--solo` (launched over ssh) |

Why it changed: solo DFlash now **beats the old two-node TP=2** on decode
(~26.5 t/s on code vs 20.3) — see §5. TP=2 still wins prefill (1476 vs
~1050 t/s), so revisit if prefill latency becomes the complaint. The 35B
is **no longer started at boot**: spark-01 cannot seat it alongside the
262K coder + VL-8B + clone-voice (needs ~40 GiB, ~15 GiB spare). Run it on
spark-02 by hand if wanted.

**Sizing/image/port now live in the RECIPES, not in `start-cluster.sh`** —
the script passes no `--gpu-mem/--max-model-len/-t` for the spark-01
models, so the two cannot drift apart again (that drift is exactly what
broke the 2026-06 recovery). `clone-voice` (:3030) is NOT in the script:
it's a plain container with `--restart unless-stopped`.

**Previous layout (pre-2026-07-29, for reference):** :3042 27B-FP8 TP=2
across both Sparks (`qwen3.6-27b-fp8-mtp-vlm`, ray, gmu 0.30/node);
:3040 35B-A3B-FP8 MTP-OFF 128k (gmu 0.40). All on the 0.23 image
`vllm-node-tf5-v0231`, **prefix-caching OFF** (§3).
The script is **idempotent** — each step skips if its port already serves —
so re-runs are harmless. It waits for spark-02 to be ssh-reachable before
the TP=2 and 4B steps (both depend on spark-02).

**Boot chain (spark-01):** `spark-services.service` (enabled) →
`start-services.sh` → `start-vllm.sh` (now just `exec start-cluster.sh`) +
`start-monitor.sh`. (ComfyUI was removed from boot 2026-06-29 — as-needed
only; run `~/admin/start-comfyui.sh` manually and re-check its
`--reserve-vram` against the current vLLM layout first.)

**Retired 2026-06-28** (disabled — do NOT re-enable, they fight
start-cluster.sh): `spark-vllm-35b`, `spark-vllm-qwen-vl`,
`spark-vllm-smolvlm-500m` (spark-01) and `spark-vllm` (spark-02). Their
orphan `~/admin/start-vllm-*.sh` scripts remain on disk but are unwired.
**spark-02 needs nothing enabled** — its endpoints (TP=2 worker + 4B) are
launched from spark-01 over ssh.

**To change the layout, edit in one place** — the relevant recipe (in this
repo) and/or the matching block in `start-cluster.sh`. Do not reintroduce
per-model systemd units.

**Launching by hand: `run-recipe.py` DIES WITH ITS SHELL SESSION.** It
stays in the foreground streaming logs and traps SIGTERM to stop its
container ("Stopping cluster... Cluster stopped"). So launching it as a
plain background job of an interactive/agent session means **the model is
torn down when that session exits** — this silently killed both spark-01
endpoints on 2026-07-29. Use one of:
- `./run-recipe.py <recipe> --solo --name <n> -d`  (daemon mode; what
  `start-cluster.sh` uses)
- `setsid nohup ./run-recipe.py ... > log 2>&1 < /dev/null &`
Verify detachment with `ps -o ppid=,sid= <pid>` — want **ppid 1** and
sid == pid. `clone-voice` is immune (plain `--restart unless-stopped`).

**Fabric fallback at boot (added 2026-07-29).** `.env` pins
`LOCAL_IP`/`ETH_IF` to the CX7 point-to-point link. If spark-02 is down
that interface has no carrier and **even `--solo` launches hang in Gloo**
("Unable to find address for: enp1s0f1np1") — `.env` feeds `VLLM_HOST_IP`,
so a recipe-level env override does NOT save you. `start-cluster.sh` now
swaps `.env` to `LOCAL_IP=127.0.0.1 / ETH_IF=lo` when spark-02 never
appears, and restores it via an EXIT trap. Same manual workaround applies
when launching by hand on a lone Spark.

**Hardened 2026-07-06** after a real dual-node power-loss boot exposed two
launch bugs (the systemd→start-cluster chain itself fired correctly):
1. **Launches must be SERIAL, not parallel.** Dispatching all three models
   at once (`-d` back-to-back) makes their vLLM memory profilers race at
   cold boot — both the 35B and 4B died with "No available memory for the
   cache blocks" while the others were mid-load. The script now launches
   TP=2 → wait-until-serving → 35B → wait → 4B (~5-10 min total,
   deterministic). Don't "optimize" it back to parallel.
2. **Stale containers must be torn down before launch.** Docker's restart
   policy revives the old containers at boot as empty shells with dead Ray
   state; the TP=2 relaunch collided with one
   (`ray ActorHandleNotFoundError: ... previous session`). The script now
   `docker rm -f`s any container whose port isn't serving before launching
   into it.
The hardened script recovered the full cluster from exactly that broken
state in 5.5 min (validated 2026-07-06). Remaining unproven: a cold boot
that runs the hardened script *from systemd* end-to-end — expected fine
(chain + script each proven separately); confirm on the next reboot via
`journalctl -u spark-services` `[start-cluster ...]` lines.

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
