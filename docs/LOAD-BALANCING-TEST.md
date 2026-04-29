# hikyaku load-balancer — Benchmarking & Validation

Companion to `LOAD-BALANCING.md`. Five tests, ordered cheap → thorough.
Each test answers a specific question about the proxy's behavior.

## Setup

Both endpoints symmetric, AWQ-INT4 + MTP, `max_model_len=196608`,
`gpu_memory_utilization=0.50`:

```
spark-01:3041   (direct backend A)
spark-02:3042   (direct backend B)
limone:4000     (proxy in front of A and B, route gresh-coder → both)
```

The benchmarking machine should be **off-Spark** — running this from
spark-01 itself biases the network path. Run from limone, or from the
client laptop. Identical paths to both backends matter for clean numbers.

`gresh-coder` route on limone should be configured with `strategy:
sticky_least_loaded` and both backends listed (per LOAD-BALANCING.md
Configuration example).

---

## Test 1 — Proxy overhead

**Question:** Does the proxy add measurable per-request latency on the
hot path?

**Method:** Run llama-benchy at concurrency=1 against (a) one backend
directly and (b) the proxy with one backend in the pool. Compare TTFT.

```bash
# Direct
llama-benchy --base-url http://spark-01:3041/v1 \
  --model Qwen/Qwen3.6-27B-AWQ-INT4 \
  --tokenizer cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --pp 512 --tg 256 --depth 0 --runs 5 \
  --skip-coherence --format md \
  --save-result /tmp/bench-direct.md

# Through proxy (only spark-01 in the pool, or spark-02 idle)
llama-benchy --base-url http://limone:4000/v1 \
  --model gresh-coder \
  --tokenizer cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --pp 512 --tg 256 --depth 0 --runs 5 \
  --skip-coherence --format md \
  --save-result /tmp/bench-via-proxy.md
```

**Pass criteria:**
- TTFT delta (proxy − direct): **< 5 ms** at p50, **< 15 ms** at p95
- Decode t/s within ±2% of direct
- pp t/s within ±5% of direct

**If it fails:** the proxy is doing too much per-request work. Common
causes: full-body buffering when streaming should pass through;
synchronous metric scrape on the request path; lock contention in the
affinity table.

---

## Test 2 — Distribution scaling

**Question:** Does aggregate throughput scale with backend count under
independent (no-affinity-overlap) traffic?

**Method:** llama-benchy generates fresh prompts per request (no
multi-turn), so each request looks like a "new session" to the proxy.
The proxy should distribute roughly evenly via `least_loaded`. Compare
proxy-with-2-backends to single-backend at the same concurrency.

```bash
# Single backend baseline at concurrency 4
llama-benchy --base-url http://spark-01:3041/v1 \
  --model Qwen/Qwen3.6-27B-AWQ-INT4 \
  --tokenizer cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --pp 512 --tg 256 --depth 0 --runs 5 \
  --concurrency 1 2 4 8 \
  --skip-coherence --format md \
  --save-result /tmp/bench-single-c.md

# Through proxy with both backends, same concurrency sweep
llama-benchy --base-url http://limone:4000/v1 \
  --model gresh-coder \
  --tokenizer cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  --pp 512 --tg 256 --depth 0 --runs 5 \
  --concurrency 1 2 4 8 \
  --skip-coherence --format md \
  --save-result /tmp/bench-proxy2-c.md
```

**Pass criteria (per concurrency level):**
- Aggregate decode t/s through proxy ≥ **1.7×** single backend at the
  same concurrency. (Theoretical max is 2.0×; 1.7 leaves room for proxy
  overhead and slight imbalance.)
- Per-request t/s through proxy ≈ single backend at *half* the
  concurrency. (Each backend sees C/2 requests; per-request rate
  matches that lower load.)
- Backend dispatch counts within ±10% of even (read from
  `hikyaku_backend_dispatch_total{backend="..."}` if Phase 3
  metrics are exposed, or from each backend's `vllm:num_requests_running`
  history during the run).

**If it fails:**
- Aggregate < 1.5× single → proxy is serializing or one backend is
  starved (check `least_loaded` scoring).
- Per-request t/s ≈ single-backend full-concurrency rate → proxy is
  routing all traffic to one backend (check affinity bug:
  llama-benchy's prompts may share enough leading bytes to collide on
  one canonical-prefix hash).

---

## Test 3 — Affinity preservation (custom harness)

**Question:** Does each multi-turn session pin to one backend across
all its turns?

**Why custom:** llama-benchy treats every request as independent.
Affinity logic is invisible to it. We need a small driver that
maintains per-session conversation state across N turns, records which
backend handled each turn, and asserts stickiness.

### Prerequisite: backend identity in response headers

Proxy must set `X-hikyaku-backend: <backend-id>` on every response
(see `LOAD-BALANCING.md` Open Questions — this is the trivial-but-
useful one). Without this header, the test can't directly observe
which backend served each turn; it would have to infer from per-backend
metric deltas, which is much noisier.

### Harness

`tests/affinity_harness.py` — drop-in once written:

```python
#!/usr/bin/env python3
"""
Affinity harness: spawn N concurrent sessions of T turns each, assert
each session pins to one backend.
"""
import asyncio, hashlib, time, argparse, sys
import aiohttp

SYS_PROMPT = "You are a concise coding assistant."
TURN_PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Now make it iterative instead of recursive.",
    "Add type hints.",
    "Add a docstring with an example.",
    "Now write a unit test for it using pytest.",
]

async def run_session(session_id: int, base_url: str, model: str, n_turns: int):
    """One session: send N turns sequentially, record backend per turn."""
    messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": f"Session {session_id}: {TURN_PROMPTS[0]}"},
    ]
    backends = []
    timings = []
    async with aiohttp.ClientSession() as cli:
        for turn in range(n_turns):
            t0 = time.monotonic()
            async with cli.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 200,
                    "temperature": 0.6,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                assert resp.status == 200, f"session {session_id} turn {turn}: HTTP {resp.status}"
                backend = resp.headers.get("x-hikyaku-backend", "UNKNOWN")
                body = await resp.json()
            backends.append(backend)
            timings.append(time.monotonic() - t0)
            content = body["choices"][0]["message"].get("content") or ""
            messages.append({"role": "assistant", "content": content})
            if turn + 1 < n_turns:
                messages.append({"role": "user", "content": TURN_PROMPTS[(turn + 1) % len(TURN_PROMPTS)]})
    return session_id, backends, timings

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://limone:4000")
    ap.add_argument("--model", default="gresh-coder")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=4)
    args = ap.parse_args()

    tasks = [
        run_session(i, args.base_url, args.model, args.turns)
        for i in range(args.sessions)
    ]
    results = await asyncio.gather(*tasks)

    print(f"\n{'session':>8} {'turns':>5} {'unique-backends':>16} {'backends':<60} {'avg-ttft (s)':>13}")
    print("-" * 110)
    pass_count = 0
    backend_hits = {}
    for sid, backends, timings in results:
        unique = set(backends)
        sticky = len(unique) == 1
        if sticky:
            pass_count += 1
            backend_hits[backends[0]] = backend_hits.get(backends[0], 0) + 1
        print(f"{sid:>8} {len(backends):>5} {len(unique):>16} {','.join(backends):<60} {sum(timings)/len(timings):>13.2f}")

    affinity_rate = pass_count / len(results)
    print()
    print(f"Affinity hit rate: {pass_count}/{len(results)} = {affinity_rate:.1%}")
    print(f"Backend distribution: {backend_hits}")

    sys.exit(0 if affinity_rate >= 0.95 else 1)

if __name__ == "__main__":
    asyncio.run(main())
```

### Run

```bash
python3 tests/affinity_harness.py --sessions 8 --turns 4
```

**Pass criteria:**
- **Affinity hit rate ≥ 95%** — i.e., at least 7 out of 8 sessions pin
  to a single backend across all 4 turns.
- **Backend distribution** within ±25% of even — with 8 sessions, that
  means each backend sees 3-5. Tighter is better but small N is noisy.
- **TTFT consistent** within a session — first-turn TTFT may be slow
  (cache cold); turns 2-4 should drop noticeably as prefix cache warms
  on the pinned backend. This is the "prefix-cache locality preserved"
  signal; if turn-2+ TTFT is flat against turn-1, the session isn't
  benefiting from sticky routing.

**If it fails:**
- Sessions split across backends → affinity key collision (verify the
  canonical-prefix hash is stable across turns; debug with key
  computation logs).
- All sessions land on one backend → least-loaded scoring is tilted
  (check `InFlightLocal` accounting; check whether scrape is enabled
  but stale, biasing toward "no in-flight" backend).

---

## Test 4 — Failover under live traffic

**Question:** Does killing a backend mid-test cause errors visible to
clients, or does the proxy degrade gracefully?

**Method:** Run the affinity harness with longer turns (10 each), and
in a separate shell `docker stop vllm_mtp` on one backend at the 5-turn
mark.

```bash
# Terminal 1 — start harness
python3 tests/affinity_harness.py --sessions 8 --turns 10

# Terminal 2 — wait ~30s, then kill one backend
sleep 30 && ssh paul@spark-02 "docker stop vllm_mtp"
```

**Pass criteria:**
- Zero 5xx responses to the harness during the kill.
- Sessions pinned to the killed backend re-pin to the survivor on their
  next turn (visible in the harness output as the backend ID changing
  exactly once per affected session).
- No session aborts; all 8 sessions complete all 10 turns.
- Health check marks the backend unhealthy within `unhealthy_after ×
  interval_seconds` (default 30s).

**If it fails:**
- 5xx in the harness output → proxy isn't catching backend-side
  connection errors and re-routing.
- Sessions stuck pinned to the dead backend → affinity logic isn't
  consulting the healthy bool before honoring a pin.
- Health check oscillation → tune `unhealthy_after`.

---

## Test 5 — Defender: loop detection

**Question:** Does loop detection fire on identical-body retry storms,
and does the forcing message break the loop?

**Method:** Send the same chat completion request 5 times in tight
succession (mimics the CLI retry loop), with affinity key derived from
a known canonical prefix.

```bash
PAYLOAD='{"model":"gresh-coder","messages":[
  {"role":"user","content":"Run: git commit -m foo"}
],"max_tokens":50,"temperature":0.6}'

for i in 1 2 3 4 5 6; do
  echo "=== request $i ==="
  curl -s -m 30 http://limone:4000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -i -d "$PAYLOAD" | head -20
  echo ""
done
```

**Pass criteria:**
- Requests 1-2 (below threshold): 200 OK, no `X-hikyaku-defender`
  header.
- Request 3 (at threshold): 200 OK with `X-hikyaku-defender:
  loop_detection_inject` and a forcing system message visibly
  prepended in the model's input (visible in backend's vllm logs as
  an extra system message in the rendered prompt — or in the response
  if the model addresses the injected instruction).
- Requests 4-5 (still identical despite injection): 200 OK with the
  injection still firing.
- Request 6 (after `escalate_after=2` further attempts): 429 with body
  explaining the loop and the headers `X-hikyaku-defender:
  loop_detection_refused`.

**If it fails:**
- No detection at request 3 → counter map isn't keyed on
  `(affinity_key, body_hash)`; check the hash function and
  affinity-key derivation.
- Detection fires too early → window or threshold misconfigured.
- Forcing message not visible in backend prompt → injection happens
  but is being filtered or overwritten somewhere.

---

## Optional — Test 6: zero-content detection

**Question:** Does the zero-content defender refuse content-light
requests when configured to?

**Method:** Send a request with a large system prompt + tools array
but a trivially-short last user message.

```bash
PAYLOAD='{"model":"gresh-coder","messages":[
  {"role":"system","content":"'$(python3 -c "print('x'*8000)")'"},
  {"role":"user","content":"."}
],"max_tokens":50}'

curl -s -m 30 -i http://limone:4000/v1/chat/completions \
  -H "Content-Type: application/json" -d "$PAYLOAD" | head -20
```

**Pass criteria:**
- HTTP 400 with explanatory body
- `X-hikyaku-defender: zero_content_blocked` header
- Backend logs show no inference happened (request was short-circuited
  at the proxy)

---

## Test execution order

Run cheapest-first; stop at the first failure unless you want to
characterize multiple issues.

1. **Test 1** (~2 min) — proxy overhead
2. **Test 2** (~10 min) — distribution scaling
3. **Test 3** (~5 min) — affinity preservation
4. **Test 5** (~2 min) — loop detection
5. **Test 6** (~1 min) — zero-content (if defender is wired)
6. **Test 4** (~5 min) — failover (run last; explicitly disruptive)

Wall time for the full suite: ~25 min, dominated by Test 2's
concurrency sweep.

## Report shape

Each test produces either pass/fail or a small table of metrics. The
suite report at the end is:

| Test | Pass | Notes |
|---|---|---|
| 1 — overhead | ✓ | TTFT +2.1 ms, decode within 0.5% |
| 2 — distribution | ✓ | 1.83× aggregate at C=4, 53/47% backend split |
| 3 — affinity | ✓ | 8/8 sessions sticky, 4/4 turn-2+ TTFT ≤ 200ms |
| 4 — failover | ✓ | 0 5xx, 4 sessions re-pinned within 12s |
| 5 — loop detect | ✓ | injection at req 3, 429 at req 6 |
| 6 — zero content | ✓ | 400 returned, backend not contacted |

Once all green, the load balancer is ready for live `gresh-coder`
traffic. Until then, run with the proxy in front of a single backend
(strategy: single) so the existing routing path is unchanged.
