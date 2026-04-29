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
