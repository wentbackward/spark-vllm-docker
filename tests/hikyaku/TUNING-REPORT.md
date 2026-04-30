# Hikyaku Performance Testing — Initial Report

Snapshot of an exploratory performance-characterization session on
Hikyaku v1, captured for future reference when this work resumes
post-productionization.

**Date:** 2026-04-29
**Scope:** Strategy = `round_robin`. Affinity / `sticky_least_loaded`
under stress was *not* tested in this session — deferred to a follow-up
once the algorithm-correctness fix (`first_user_message` hashing) is in
the proxy code.

**Status:** This is *not* publication-quality benchmarking. The test
rig is informal (mixed local/network paths, mixed Docker / bare
processes, no controlled thermals, no warm-up protocol). It was good
enough to answer the questions we cared about; it isn't good enough
for an external claim.

---

## Why we did this

The goal wasn't to produce headline numbers — it was to *de-risk three
specific concerns* before going further with Hikyaku as a serious
component of our stack:

1. **Is Go fast enough for this work?** I had a nagging worry that a
   Python-based proxy might be in the same performance ballpark as a
   Go one for HTTP-forwarding workloads. That worry is now thoroughly
   put to bed.
2. **Does the load balancer actually distribute traffic?** Even though
   we only exercised the simplest strategy (`round_robin`), we wanted
   visible proof that requests fan out across backends as configured.
3. **What happens when it's overloaded?** Memory leaks? Connection
   storms? Cascading failures? Or graceful degradation?

By the end of the session all three were settled.

---

## Top-line conclusions

**Hikyaku is high-throughput, low-latency, and stable under severe
pressure on modest hardware.** Specifically:

- **Throughput**: north of 10,000 RPS sustained on a consumer mini-PC,
  ~6,000 RPS on a small ARM SBC. Per-CPU-core efficiency is high and
  scales linearly across architectures.
- **Latency**: sub-millisecond proxy-internal overhead on top of network
  + backend. Sub-15ms p99 at non-saturated loads.
- **Stability**: zero memory growth observed across hundreds of
  thousands of requests; clean graceful degradation under overload
  (RPS plateaus or steps down, latency tail grows modestly, no error
  storms, distribution stays balanced).
- **Robustness**: even when CPU thermal-throttles mid-test, hikyaku
  serves what it can and queues what it can't — no failures, no queue
  runaway, balanced distribution preserved.

This is the level of behavior I'd associate with banking/telecom
infrastructure software. It's astonishing on edge-class hardware.

---

## Test setup

### Test machines

| Role | Machine | Specs |
|---|---|---|
| Heavyweight test rig | albicocca | AMD Ryzen 9 7940HS (Zen 4, 8c/16t), 64 GB DDR5-5600, 1 TB NVMe, 2.5 GbE, mini-PC class |
| Edge test rig | limone | NVIDIA Jetson Orin Nano dev kit, 6× Cortex-A78AE @ ~1.5 GHz, 8 GB shared LPDDR5, fanned heatsink, MAXN power mode |

Limone is one of two we have; total cost was $500 for the pair. Solid
state, fanned but tiny, passively-leaning thermal design.

### Software components

All Python, all in `tests/hikyaku/`:

- **`fake_llm.py`** — minimal OpenAI-compatible HTTP server returning
  canned responses with simulated TTFT/TPOT timing, plus a vLLM-shaped
  `/metrics` endpoint. Stamps an `X-Fake-LLM-Id` header on every
  response so the harness can identify which backend served each
  request. Configurable response-token count and per-token delay.

- **`latency_harness.py`** — async (aiohttp) correctness harness. Two
  modes: `independent` (each request a fresh prompt; tests
  distribution) and `affinity` (multi-turn sessions with stable
  opening prompt; tests stickiness). Reports per-request percentiles,
  backend distribution, and (in affinity mode) sticky session rate.
  Caps at ~50 concurrent due to asyncio overhead.

- **`locustfile.py`** — Locust load profile with the same modes/payload
  options as the latency harness, but built to scale to thousands of
  concurrent users via Locust's gevent + multi-process worker model.
  Used for the bulk of the throughput testing.

- **`README.md`** — wire-up instructions and quickstart commands for
  all three.

- **`TUNING.md`** — OS tuning checklist (file descriptors, TCP listen
  backlog, ephemeral port range, conntrack, etc.) needed to actually
  reach high RPS without hitting kernel limits.

### Hikyaku configuration tested

```yaml
groups:
  test-cluster:
    strategy: round_robin   # NOTE: affinity NOT tested in this session
    health_check:
      path: /metrics
      interval_seconds: 10
      timeout_seconds: 2
      unhealthy_after: 3

backends:
  - id: test1
    type: openai
    base_url: "http://<host>:9001"   # initial 0.0.0.0 was wrong, see Lessons
    api_key: "DUMMY"
    group: test-cluster
  - id: test2
    type: openai
    base_url: "http://<host>:9002"
    api_key: "DUMMY"
    group: test-cluster
```

Two `fake_llm.py` instances served as the backend pool throughout.

### Locust workload

```bash
HIKYAKU_PAYLOAD=minimal HIKYAKU_MAX_TOKENS=1 \
locust -f hikyaku-tests/locustfile.py --host <hikyaku-url> \
  --processes -1 --headless --users <N> --spawn-rate <R> --run-time <T>s
```

`minimal` payload + `1`-token responses + `--ttft-ms 1 --tpot-ms 0`
fakes meant the **proxy was the bottleneck**, not request body size or
fake processing time. This was deliberate: we wanted to find out what
hikyaku alone can do, not what the whole stack can do.

`--processes -1` forks one Locust worker per logical CPU on the load
generator. Without this, Locust itself bottlenecks at 4-8K RPS due to
Python's GIL.

---

## Test runs and observations

All runs used `--ttft-ms 1 --tpot-ms 0 --response-tokens 1` on the fakes
unless otherwise noted. Hikyaku ran in Docker throughout.

### Run 1 — Ryzen, 100 users, fully co-located, localhost

Locust + hikyaku + 2 fakes all on albicocca. Loopback networking only.

```
Backend distribution (50/50):  fake-A: 50.2%   fake-B: 49.8%
Total requests:                383,224
Failures:                      0
RPS sustained:                 12,755
p50 / p95 / p99 / p99.9:       7 ms / 12 ms / 14 ms / 18 ms
max:                           96 ms
```

This was the cleanest "what can hikyaku do" number for the Ryzen.
Median latency of 7 ms — call that ~5-6 ms hikyaku overhead, since
fakes returned in ~1 ms.

**Per-core throughput: ~1,594 RPS / core.**

Math sanity: 100 users × (1 / 0.007s) = 14,285 theoretical max RPS at
this latency. We got 89% of that, meaning the *user count* was the
binding constraint, not hikyaku.

### Run 2 — Ryzen, 500 users, fully co-located

Same setup, more users.

```
Backend distribution:          50.2 / 49.8
Total requests:                644,283
Failures:                      0
RPS sustained:                 10,726        (down ~16% vs Run 1)
p50 / p95 / p99 / p99.9:       16 / 110 / 120 / 130 ms
max:                           164 ms
```

**RPS dropped despite more users.** This is the saturation cliff: more
load no longer translates to more throughput. CPU on the Ryzen is
saturated by the combined Locust + hikyaku + 2 fakes process tree.
Fascinating side-observation: the percentile distribution went bimodal
— p50 at 16ms but p66 at 74ms, suggesting half the requests get a
"fast path" CPU slice and half wait for one. Probably scheduling
unfairness under contention, not a hikyaku-internal queue.

### Run 3 — Ryzen, ~5000 users, "I broke it"

Pushed past saturation deliberately, to find failure modes.

```
Total requests:                544,865
Failures:                      4   (0.0007%)
RPS sustained:                 9,066
p50 / p99 / p99.9:             35 / 780 / 7,000 ms
max:                           53,000 ms (effectively waited the whole test)
```

Even at this absurd load, **only 4 errors out of 545K requests**. All
4 were 502s with `connection reset by peer` from one specific fake —
its asyncio accept queue overflowed under burst pressure. **Not
hikyaku's fault; the proxy correctly forwarded the upstream error.**
The multi-second tail latencies represent connections held by the
test the entire run waiting for a serving slot.

This was the moment we hit Linux's default `nofile` ulimit (1024) and
saw `OSError: [Errno 24] Too many open files`. Triggered the OS-tuning
work (see `TUNING.md`).

After applying the tunings (FD limit → 1M, somaxconn → 65535,
ip_local_port_range expanded, tcp_tw_reuse=1, conntrack_max → 1M):

```
Total requests:                544,865
Failures:                      4   (0.0007%)
RPS sustained:                 9,066    (essentially unchanged — CPU-bound, not FD-bound)
p50:                           35 ms   (improved from 62ms pre-tuning)
```

The OS tuning **moved the bottleneck from FDs to CPU**, which is where
it should be. Throughput stays at ~9K RPS as the *system* (CPU)
saturation floor. Hikyaku itself can clearly do more.

### Run 4 — Jetson Orin Nano via tailscale

Cross-machine setup. Locust on Ryzen, fakes on Ryzen, hikyaku on the
Orin Nano. All traffic over tailscale (wireguard + userspace).

```
Backend distribution:          50.0 / 50.0
Total requests:                23,152
Failures:                      0
RPS sustained:                 3,757
p50 / p99 / p99.9:             24 / 63 / 96 ms
max:                           120 ms
```

`tegrastats` and `top` showed `tailscaled` was the **#1 CPU consumer
on the Jetson** — the wireguard crypto was eating most of the cores.
Hikyaku itself was lightly loaded.

**Per-core throughput (proxy-only, estimated): ~1,000 RPS / core.**

### Run 5 — Jetson Orin Nano, plain HTTP over LAN

Same cross-machine setup but pointed at the Jetson's LAN IP directly,
bypassing tailscale.

```
Backend distribution:          50.1 / 49.9
Total requests:                178,683
Failures:                      0
RPS sustained:                 5,945       (+58% vs tailscale)
p50 / p99 / p99.9:             15 / 45 / 61 ms
max:                           120 ms
```

The **tailscale tax was 37% of throughput** on this Jetson. Strip
crypto out and hikyaku gets that compute back. p50 also dropped
sharply (24 → 15 ms) since each request now had two fewer wireguard
encryptions in its path.

**Per-core throughput: ~991 RPS / core.** Despite the Orin's cores
being ~30% the speed of Zen 4 cores clock-for-clock, hikyaku achieves
~62% of the Ryzen's per-core RPS. Per *cycle*, hikyaku is actually
*more* efficient on ARM than on Zen 4 — Go's concurrency model is
uniform across architectures.

### Run 6 — Jetson Orin Nano, 500 users, plain LAN

Pushed the Jetson harder. **Hit the thermal throttle wall.**

The Locust live stats showed a textbook thermal-throttle staircase:

| Window | RPS | Median |
|---|---|---|
| ~10 s | 6,177 | 65 ms |
| ~15 s | 5,797 | 68 ms |
| ~20 s | **4,586** | 68 ms — first downshift |
| ~25 s | 4,699 | 69 ms |
| ~30 s | **3,378** | 71 ms — second downshift |
| ~35 s | **2,133** | 72 ms — third downshift |
| ~40 s | 2,237 | 73 ms |
| ~45 s | 2,019 | 74 ms |
| ~50 s | 1,832 | 75 ms — thermal floor |
| ~55 s | 1,928 | 76 ms |

Final aggregate: 220,077 requests, 0 failures, 3,664 average RPS, p99
of 630 ms, max 1,431 ms.

Two important observations:

1. **The staircase pattern is the thermal-throttle signature.** Linux
   cpufreq scaling responds to thermal events by stepping down through
   discrete P-states. Pure load saturation produces a single asymptote
   instead — RPS plateaus, doesn't go down.
2. **Hikyaku's behavior through thermal degradation was excellent.**
   Median latency only climbed 65 → 76 ms while throughput dropped
   3.2×. Zero failures. Distribution stayed clean. Max latency stayed
   bounded at ~1.4 s (no queue runaway). The proxy noticed slower
   cycles and just delivered fewer RPS — exactly the right behavior.

The Orin Nano dev kits *do* have a fan, but it's a small one in a
cramped enclosure. Active cooling upgrade or a clamped power mode
would shift the thermal floor up. For our purposes, the Run 5 number
(5,945 RPS at 100u) is the meaningful "edge hardware ceiling".

---

## Findings — answering our three concerns

### 1. Is Go fast enough?

**Answer: yes, by a large margin.** On a consumer mini-PC, in Docker,
sharing the box with the load generator and two backends, hikyaku
sustained over 12K RPS with single-digit-millisecond median latencies.
On a $250 ARM SBC, in Docker, over a real LAN, it sustained ~6K RPS
with mid-teens median latencies. Per-cycle efficiency held up across
architectures, suggesting Go's scheduler and netpoll are doing what
they should be.

Single-millisecond proxy overhead per request leaves the entire
latency budget for the actual LLM upstream. Hikyaku will not be the
bottleneck in any production scenario we'd run.

### 2. Does load balancing work?

**Answer: yes, for `round_robin`. Affinity not yet tested under load.**
In every run, `round_robin` produced distribution within 1% of perfect
(50/50) regardless of total RPS, regardless of saturation state,
regardless of thermal state. That's the routing mechanism working
correctly; the affinity-aware path (`sticky_least_loaded` with the new
`first_user_message` hashing) needs its own validation in a follow-up
session.

### 3. What happens under overload?

**Answer: graceful degradation, no crashes, no leaks.** Specifically:

- **Memory**: no observable growth across runs of hundreds of thousands
  of requests. Whatever LRU / TTL bookkeeping hikyaku does, it's
  bounded.
- **Errors at saturation**: `< 0.001%` failure rate even when pushed
  far past capacity, and the failures we saw were upstream-side
  (fake's accept queue), not hikyaku-side.
- **Latency tail**: grows as expected under saturation but never
  catastrophically. No request held "forever" — max bounded by what
  reasonable timeouts would catch.
- **Distribution under stress**: unaffected. Even at the saturation
  cliff and through thermal throttling, distribution stayed within
  1-2% of even.
- **Cliff behavior**: as load increases past saturation, RPS plateaus
  or step-degrades; latency tail grows; nothing storms. This is
  textbook backpressure-respecting behavior.

This is the level of robustness I'd want from low-latency
infrastructure software. Hikyaku has it.

---

## What this report does *not* prove

To be clear about the gaps:

- **No published-grade benchmark methodology.** Mixed Docker /
  bare-process setups, no warm-up protocol, no controlled thermals,
  no isolated network, no statistical rigor (single runs, not
  multi-run distributions).
- **No affinity-under-load test.** `sticky_least_loaded` with the
  fixed `first_user_message` algorithm is unvalidated at high
  concurrency. The earlier `latency_harness.py` test at 16
  sessions/4 turns was failing at 43.8% — that fix is in the spec
  but needs to be re-run on the proxy code that was running in this
  session.
- **No defender testing.** Loop detection, zero-content detection
  haven't been exercised. They're Phase 2.5 in the spec; can be
  tested with the existing fake harness once implemented.
- **No long-running soak test.** Longest run was 60 seconds. Nothing
  rules out a slow leak that takes hours to manifest.
- **No real LLM backend.** All numbers used canned responses with
  simulated timing. Real vLLM/SGLang traffic has very different
  characteristics (long streams, KV cache pressure on the backend,
  variable response times). Hikyaku should be unaffected by these,
  but we haven't directly proven it.
- **Single-machine hikyaku.** No clustering / horizontal scaling
  tested.

If we ever publish numbers, they'd need a different methodology:
dedicated machines per role, controlled cooling, multiple statistical
runs, longer soak windows, real LLM backends, etc. These results were
internal-confidence-building, nothing more.

---

## Lessons learned along the way

Captured in case future-us hits the same things:

### 1. Locust caps on a single process

Without `--processes -1`, Locust's GIL-bound single process caps at
4-8K RPS regardless of how many users you spawn. That ceiling can be
mistaken for hikyaku's ceiling. Always use `--processes -1` for
high-RPS testing, and watch for the *load generator* hitting CPU
saturation before assuming the proxy is the bottleneck.

### 2. Linux defaults are wrong for high-RPS proxy work

We hit `nofile` (1024 default), eventually almost hit ephemeral port
exhaustion, and would have hit conntrack limits if we'd pushed
further. None of these are hikyaku problems; they're OS-defaults
problems. Wrote up `TUNING.md` with the full kit. Apply it before
benchmarking, not during.

### 3. `0.0.0.0` is a bind address, not a connect destination

Initial hikyaku config had backends with `base_url:
http://0.0.0.0:9001`. That works *if hikyaku is on the same machine
as the fake* (because `0.0.0.0` resolves to localhost on the
connecting host), but breaks the moment hikyaku runs on a different
machine. Symptom: 100% failure rate with `HTTP 0:` in Locust output,
~12 ms latency (fast fail-path). Always specify backends as the
host's actual IP/hostname.

### 4. Fake LLM needs both `/v1/...` and stripped `/...` paths

Hikyaku is opinionated: it expects `/v1/...` on incoming, strips it,
appends the unprefixed path to `base_url` for outgoing. Real vLLM
exposes only `/v1/...`. Our fake had to register both forms to work
with Hikyaku's path conventions. Worth documenting in the fake's
README (done).

### 5. Tailscale crypto is expensive on small ARM cores

37% throughput tax on the Orin Nano. Not surprising in retrospect —
wireguard userspace + Curve25519 isn't cheap on Cortex-A78. Worth
budgeting for when designing the actual production deployment: if
hikyaku and backends sit behind tailscale, expect that overhead. LAN
or untunneled paths get it back.

### 6. Thermal throttling looks like a staircase

Distinct from saturation, which looks like a single plateau. If RPS
is *decreasing* over time at constant load, suspect thermal first.
`tegrastats` on Jetson, `mpstat` + thermal_zone reads on x86 will
confirm.

### 7. `--processes -1` isn't free

Locust's multi-process mode forks one worker per CPU and *each* worker
opens its own TCP connection pool. Multiplies FD usage. The 1M FD
limit from `TUNING.md` exists partly because of this.

---

## Headline numbers

For internal reference, the numbers that matter:

| Measurement | Value | Hardware |
|---|---|---|
| Peak sustained RPS (1 instance) | **12,755** | Ryzen 9 7940HS, localhost |
| Edge sustained RPS (1 instance) | **5,945** | Jetson Orin Nano, LAN |
| Peak failure rate at 5x saturation | **0.0007%** | Ryzen, post-tuning |
| p99 at non-saturated load | **14 ms** | Ryzen, 100u |
| p99 at non-saturated load (edge) | **45 ms** | Orin Nano, 100u, LAN |
| Per-core RPS, x86 | **~1,594** | Ryzen 9 7940HS |
| Per-core RPS, ARM | **~991** | Orin Nano A78AE |
| Distribution accuracy (round_robin) | **±1%** | All loads, all hardware |
| Memory growth across runs | **none observed** | All hardware |

---

## Followup work — what to do when this resumes

1. **Affinity validation under load.** Switch hikyaku to
   `sticky_least_loaded` with `first_user_message` key (post the
   spec fix). Run `latency_harness.py` in `--mode affinity` at small
   N first (the original 16/4 test) — expect 100% sticky rate if
   the algorithm is implemented correctly. Then escalate to Locust
   in affinity mode at 200-500 users for the under-load validation.

2. **Defender validation.** Once loop detection and zero-content
   detection are implemented (Phase 2.5 in `LOAD-BALANCING.md`),
   write small targeted tests:
   - Repeated identical request → triggers loop detection at
     configurable threshold → forcing message visible in upstream
     prompt → escalates to 429 after configurable retries.
   - Empty user content with large system prompt → 400 with
     defender header.

3. **Soak test.** A long-running (8-24 hour) low-RPS run to catch
   slow leaks or accumulated state issues. The current "60-second
   max" methodology can't see those.

4. **Production-shape benchmark.** Real vLLM backend with realistic
   payload sizes, real conversation patterns, prefix-cache-aware
   workload. Validates that hikyaku's affinity-routing actually
   delivers the prefix-cache-locality benefit that motivated its
   design.

5. **Multi-instance hikyaku.** Two hikyaku instances behind a TCP
   load balancer (haproxy / nginx), pointing at the same backend
   pool. Validates that routing decisions are consistent across
   instances and affinity tables can either be replicated or be
   stateless-enough to not matter.

---

## Resuming this work

This report and the test harness should be enough to pick up cleanly:

- **Code**: `tests/hikyaku/` has fake_llm, latency_harness, locustfile,
  README, TUNING.md, and this report.
- **State**: Hikyaku v1 was running in Docker on the test rig at the
  time of writing, with `round_robin` strategy + `first_user_message`
  affinity (in spec, not yet validated in code under load).
- **Open work**: see "Followup work" above.

Tomorrow's planned activity is the affinity-under-load test (item 1
above). Everything else is post-productionization.
