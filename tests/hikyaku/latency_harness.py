#!/usr/bin/env python3
"""Latency harness for hikyaku.

Measures proxy overhead and routing behavior using fake_llm backends —
removes all LLM-side variance so timing differences come purely from
the proxy.

Modes:
  independent    each request has a fresh user prompt; tests distribution
  affinity       N sessions of T turns each, consistent first user message
                 per session; tests stickiness

Payload sizes:
  small          minimal system prompt; baseline overhead
  large          ~50 KB system prompt; tests body parsing/hashing cost

Reports per-request p50/p95/p99 wall time, backend distribution, and
(in affinity mode) sticky session rate.

Dependencies: aiohttp (pip install aiohttp).
"""
import argparse
import asyncio
import random
import statistics
import string
import time
from collections import Counter

import aiohttp


SMALL_SYS = "You are a concise coding assistant."

# ~50 KB system prompt. Realistic-ish: long enough that the proxy's body
# parsing and canonical-prefix hashing have measurable cost, but not so
# long that the fake LLM's request-handling becomes the bottleneck.
def gen_large_system_prompt(target_kb: int = 50) -> str:
    n = target_kb * 1024
    seed = ("You are a senior code reviewer. Review the following code "
            "for correctness, performance, and maintainability. ") * 30
    if len(seed) >= n:
        return seed[:n]
    filler = "".join(random.choices(string.ascii_letters + " " * 5, k=n - len(seed)))
    return seed + filler


SMALL_TURNS = [
    "Reverse this string: hello world",
    "Capital of France?",
    "List 5 sorting algorithms.",
    "Explain TCP in one sentence.",
    "What does 'idempotent' mean?",
    "Difference between TCP and UDP?",
]


async def run_request(cli, base_url, model, messages, max_tokens=50):
    t0 = time.monotonic()
    async with cli.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=aiohttp.ClientTimeout(total=120),
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        await resp.read()
        elapsed = time.monotonic() - t0
        backend = (
            resp.headers.get("x-hikyaku-backend")
            or resp.headers.get("x-llm-proxy-backend")  # legacy header name
            or resp.headers.get("x-fake-llm-id")
            or "UNKNOWN"
        )
    return elapsed, backend


async def independent_mode(args, sys_prompt):
    """N independent requests with random user prompts."""
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    async def one(i):
        async with sem:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"req-{i}: {random.choice(SMALL_TURNS)}"},
            ]
            return await run_request(cli, args.base_url, args.model, messages)

    async with aiohttp.ClientSession(connector=connector) as cli:
        results = await asyncio.gather(*(one(i) for i in range(args.requests)))

    elapsed = [e for e, _ in results]
    backends = Counter(b for _, b in results)
    return elapsed, backends, None


async def affinity_mode(args, sys_prompt):
    """N sessions × T turns. Same opening user message per session."""
    sem = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    async def session(sid):
        async with aiohttp.ClientSession(connector=connector) as cli:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Session {sid}: {SMALL_TURNS[sid % len(SMALL_TURNS)]}"},
            ]
            session_elapsed = []
            session_backends = []
            for t in range(args.turns):
                async with sem:
                    elapsed, backend = await run_request(
                        cli, args.base_url, args.model, messages
                    )
                session_elapsed.append(elapsed)
                session_backends.append(backend)
                messages.append({"role": "assistant", "content": "ack"})
                if t + 1 < args.turns:
                    messages.append(
                        {"role": "user", "content": f"turn {t+1}: continue"}
                    )
            return session_elapsed, session_backends

    all_results = await asyncio.gather(*(session(i) for i in range(args.requests)))

    elapsed_all = []
    backends_all = Counter()
    sticky_count = 0
    for session_elapsed, session_backends in all_results:
        elapsed_all.extend(session_elapsed)
        backends_all.update(session_backends)
        if len(set(session_backends)) == 1:
            sticky_count += 1

    affinity_rate = sticky_count / max(len(all_results), 1)
    return elapsed_all, backends_all, affinity_rate


def report(elapsed, backends, affinity_rate, args):
    elapsed_ms = sorted(e * 1000 for e in elapsed)
    n = len(elapsed_ms)
    p50 = elapsed_ms[n // 2]
    p95 = elapsed_ms[min(int(n * 0.95), n - 1)]
    p99 = elapsed_ms[min(int(n * 0.99), n - 1)]
    mean = statistics.mean(elapsed_ms)
    stdev = statistics.pstdev(elapsed_ms) if n > 1 else 0.0

    print()
    print(f"=== {args.mode.upper()} — payload={args.payload}, "
          f"requests={args.requests}, "
          f"{'turns=' + str(args.turns) + ', ' if args.mode == 'affinity' else ''}"
          f"concurrency={args.concurrency} ===")
    print(f"Total HTTP requests: {n}")
    print(f"  p50:  {p50:7.1f} ms")
    print(f"  p95:  {p95:7.1f} ms")
    print(f"  p99:  {p99:7.1f} ms")
    print(f"  min:  {min(elapsed_ms):7.1f} ms")
    print(f"  max:  {max(elapsed_ms):7.1f} ms")
    print(f"  mean: {mean:7.1f} ms  (stdev {stdev:.1f})")

    total = sum(backends.values())
    print(f"\nBackend distribution ({total} requests):")
    if total:
        for backend, count in sorted(backends.items()):
            pct = 100 * count / total
            bar = "#" * int(pct / 2)
            print(f"  {backend:24s} {count:>5}  {pct:5.1f}%  {bar}")

    if affinity_rate is not None:
        print(f"\nAffinity hit rate: {affinity_rate:.1%}  "
              f"({'PASS' if affinity_rate >= 0.95 else 'FAIL — expected ≥95%'})")


async def amain():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True,
                    help="proxy URL, e.g. http://hikyaku:4000")
    ap.add_argument("--model", required=True,
                    help="route name configured in hikyaku, e.g. test-route")
    ap.add_argument("--mode", choices=["independent", "affinity"], default="independent")
    ap.add_argument("--payload", choices=["small", "large"], default="small")
    ap.add_argument("--requests", type=int, default=100,
                    help="num requests (independent) or num sessions (affinity)")
    ap.add_argument("--turns", type=int, default=4,
                    help="turns per session (affinity mode only)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    sys_prompt = gen_large_system_prompt(50) if args.payload == "large" else SMALL_SYS

    print(f"Hitting {args.base_url} with model={args.model}, "
          f"sys-prompt-bytes={len(sys_prompt)}")

    if args.mode == "independent":
        elapsed, backends, affinity_rate = await independent_mode(args, sys_prompt)
    else:
        elapsed, backends, affinity_rate = await affinity_mode(args, sys_prompt)

    report(elapsed, backends, affinity_rate, args)


if __name__ == "__main__":
    asyncio.run(amain())
