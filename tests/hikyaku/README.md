# hikyaku — latency test harness

Two scripts for measuring proxy overhead and routing behavior with
**zero LLM-side variance**. The fake LLM returns canned responses with
configurable timing; the harness drives traffic and reports
percentile latencies + backend distribution.

```
fake_llm.py       — OpenAI-compatible fake backend, returns canned
                    responses with simulated TTFT/TPOT timing.
latency_harness.py — async client that hits hikyaku and aggregates
                    latency + backend-distribution stats.
```

## Dependencies

```bash
pip install aiohttp
```

Both scripts work with Python 3.10+.

## Setup

### 1. Start two fake LLMs as the backend pool

```bash
chmod +x fake_llm.py latency_harness.py

./fake_llm.py --id fake-A --port 9001 --ttft-ms 20 --tpot-ms 5 --response-tokens 50 &
./fake_llm.py --id fake-B --port 9002 --ttft-ms 20 --tpot-ms 5 --response-tokens 50 &
```

Each fake:
- Responds to `GET /v1/models`, `POST /v1/chat/completions`, `GET /metrics`
- Stamps `X-Fake-LLM-Id: <id>` on every response (the harness reads
  this to identify which backend served each request).
- Simulates inference time as `ttft_ms + tokens × tpot_ms`. Defaults
  produce ~270 ms total per 50-token response — long enough to expose
  proxy overhead, short enough to keep test runs fast.
- Exposes a vLLM-shaped `/metrics` endpoint so hikyaku's capability
  probe finds Prometheus gauges as expected.

### 2. Configure hikyaku with a test route

```yaml
routes:
  test-route:
    real_model: fake/canned-1B
    backends:
      - url: http://localhost:9001
      - url: http://localhost:9002
    strategy: sticky_least_loaded   # or round_robin for distribution baseline
    affinity:
      key: canonical_prefix
      prefix_bytes: 1024
      ttl_seconds: 300
      max_entries: 1000
    overload:
      max_concurrency: 32
      kv_cache_pct: 0.85
    health_check:
      path: /v1/models
      interval_seconds: 10
    metrics_scrape:
      enabled: auto
      interval_seconds: 5
```

For a baseline run with no smart routing, swap to:

```yaml
    strategy: round_robin
```

### 3. Run the harness

#### Baseline — proxy overhead at concurrency 1

```bash
./latency_harness.py --base-url http://hikyaku:4000 --model test-route \
  --mode independent --payload small --requests 200 --concurrency 1
```

Subtract the fake's intrinsic latency (`ttft + tokens × tpot` =
~270 ms with defaults) from p50 to isolate proxy overhead.

#### Distribution — under load

```bash
./latency_harness.py --base-url http://hikyaku:4000 --model test-route \
  --mode independent --payload small --requests 400 --concurrency 8
```

Should split traffic ~50/50 between fake-A and fake-B. Skew suggests
load-balancer scoring bug.

#### Affinity — sticky-with-hashing

```bash
./latency_harness.py --base-url http://hikyaku:4000 --model test-route \
  --mode affinity --payload small --requests 16 --turns 4 --concurrency 4
```

Reports affinity hit rate. Should be 100% — every session pins to one
backend across all 4 turns. Anything below 95% is a bug.

#### Large-context overhead — proxy parsing/hashing cost on big bodies

```bash
./latency_harness.py --base-url http://hikyaku:4000 --model test-route \
  --mode independent --payload large --requests 200 --concurrency 4
```

`large` payload is a ~50 KB system prompt. The proxy has to parse JSON
and hash a portion for the affinity key. The p50 increase vs `small`
should be a few ms at most.

## What the output looks like

```
Hitting http://hikyaku:4000 with model=test-route, sys-prompt-bytes=33

=== AFFINITY — payload=small, requests=16, turns=4, concurrency=4 ===
Total HTTP requests: 64
  p50:    272.1 ms
  p95:    314.7 ms
  p99:    342.0 ms
  min:    258.4 ms
  max:    361.8 ms
  mean:   281.5 ms  (stdev 18.2)

Backend distribution (64 requests):
  fake-A                       32   50.0%  #########################
  fake-B                       32   50.0%  #########################

Affinity hit rate: 100.0%  (PASS)
```

## Reading the numbers — what's expected

Assuming defaults (`ttft=20 ms`, `tpot=5 ms`, `response-tokens=50`):

- **Fake intrinsic latency** ≈ 20 + 50 × 5 = **270 ms** for full response
- **Proxy overhead at concurrency=1, small payload, no affinity**: p50
  should be **~272-280 ms** (proxy adds 2-10 ms of header forwarding).
- **Proxy overhead with affinity hashing on**: small payload should
  add ~1-2 ms more (xxhash on a few KB is fast). Large payload
  (50 KB) should add ~5-10 ms more (parsing + hashing the bigger body).
- **Distribution under independent traffic**: 50/50 split within
  ±10%, regardless of payload size.
- **Affinity sticky rate**: 100% of sessions, full stop. If it's not
  100%, the affinity-key hash isn't stable across turns of the same
  session — likely a bug in canonical-prefix derivation.

## Locust load testing — stress mode (`locustfile.py`)

`latency_harness.py` is good for correctness validation but caps out at
low concurrency. To find hikyaku's RPS ceiling and tail-latency
behavior under real load, use Locust.

### What Locust is

Locust is a Python-based load-testing tool. You write one Python file
describing one user's behavior; Locust spins up many such users in
parallel against your service. Live stats (RPS, percentile latencies,
error rate) appear in a browser at `http://localhost:8089`.

It scales to thousands of concurrent users on a single machine — far
beyond what asyncio harnesses can drive — and is the industry
standard for HTTP service stress testing.

### Install

```bash
pip install locust
```

(Pulls in gevent + flask. ~30 MB total.)

### Run interactively (web UI, recommended for first runs)

```bash
locust -f locustfile.py --host http://limone.royal-armadillo.ts.net:4000
```

Then open `http://localhost:8089`. Set:

- **Number of users**: how many concurrent simulated users
- **Spawn rate**: users started per second (ramp-up)

Click **Start** and watch live stats. Stop the test from the UI; backend
distribution + affinity hit rate (if affinity mode) print to the
terminal.

### Run headless (no UI, for scripted runs)

```bash
locust -f locustfile.py --host http://limone.royal-armadillo.ts.net:4000 \
  --headless --users 1000 --spawn-rate 100 --run-time 60s
```

Stops automatically after `--run-time`. Final stats print to stdout.

### Modes (set via env var)

```bash
# Independent mode (default) — fresh prompt per request, tests RPS + distribution
HIKYAKU_MODE=independent locust -f locustfile.py --host http://...

# Affinity mode — multi-turn sessions, tests stickiness under load
HIKYAKU_MODE=affinity HIKYAKU_TURNS=4 locust -f locustfile.py --host http://...
```

### Payload sizes

```bash
# Minimal — ~20 byte prompts; finds raw RPS ceiling
HIKYAKU_PAYLOAD=minimal locust -f locustfile.py --host http://...

# Small (default) — ~80 byte prompts; realistic light request
HIKYAKU_PAYLOAD=small locust -f locustfile.py --host http://...

# Large — 50 KB system prompts; tests proxy parsing + affinity hashing
# under sustained load
HIKYAKU_PAYLOAD=large locust -f locustfile.py --host http://...
```

### Recommended scenarios

#### Find RPS ceiling

Tune the fakes for near-zero latency first so the proxy is the
bottleneck, not the fakes:

```bash
./fake_llm.py --id test1 --port 9001 --ttft-ms 1 --tpot-ms 0 --response-tokens 1 &
./fake_llm.py --id test2 --port 9002 --ttft-ms 1 --tpot-ms 0 --response-tokens 1 &

HIKYAKU_PAYLOAD=minimal HIKYAKU_MAX_TOKENS=1 \
locust -f locustfile.py --host http://hikyaku:4000 \
  --headless --users 2000 --spawn-rate 200 --run-time 60s
```

Watch CPU, p95, p99. The number of users where p95 spikes is your
saturation point.

#### Affinity under load

Verify sticky-with-hashing holds when many sessions run concurrently:

```bash
HIKYAKU_MODE=affinity HIKYAKU_TURNS=10 \
locust -f locustfile.py --host http://hikyaku:4000 \
  --headless --users 200 --spawn-rate 20 --run-time 120s
```

Each "user" runs sessions of 10 turns. After the run, look for:

- `Affinity hit rate: X/Y = NNN.N% (PASS)` ≥ 95%
- Per-route distribution evenness (each backend within ±10%)

#### Realistic mixed load

Add some think time between requests for a more realistic profile —
edit `wait_time = between(0, 0)` in `locustfile.py` to e.g.
`between(0.5, 2.0)`.

### What to look for in the output

Locust shows live, per-second:

- **RPS** — requests per second sustained
- **Failures/s** — anything non-200 from hikyaku
- **Median / p95 / p99** — latency percentiles
- **Total** — cumulative count

After the run, the terminal prints:

```
============================================================
 hikyaku load test — final tallies
============================================================

Backend distribution (60123 requests):
  test1                     30055   50.0%  #########################
  test2                     30068   50.0%  #########################

Affinity hit rate: 1024/1024 = 100.0%  (PASS)
```

### How this complements `latency_harness.py`

| | `latency_harness.py` | Locust |
|---|---|---|
| Purpose | Correctness | Stress / capacity |
| Max concurrency | ~50 (asyncio) | Thousands (gevent) |
| Output | Single summary table | Live dashboard + final tallies |
| Affinity check | ✓ | ✓ |
| RPS ceiling | weak | strong |
| Setup time | none | `pip install locust` |
| Best for | "does it route correctly?" | "how fast before it breaks?" |

Use both. `latency_harness.py` for the correctness gates;
`locustfile.py` for the capacity/RPS story.

## Stress-testing tips

- **Crank concurrency** to find the proxy's CPU ceiling on whatever
  host runs hikyaku. Each fake handles many concurrent requests fine
  (it's just sleeping); the bottleneck will be the proxy's request
  handling.
- **Bump `--response-tokens` and `--tpot-ms`** to simulate longer
  generations, which keeps backends busy longer and exercises the
  proxy's `least_loaded` accounting under sustained load.
- **Add latency on one backend** (e.g. `--ttft-ms 200` on fake-B
  while fake-A stays at 20) to verify the proxy's load scoring
  prefers the faster backend when affinity isn't a factor.

## Cleanup

```bash
pkill -f "fake_llm.py"
```

Or be specific:

```bash
lsof -ti :9001 :9002 | xargs -r kill
```
