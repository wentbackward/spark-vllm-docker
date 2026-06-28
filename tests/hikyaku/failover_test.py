#!/usr/bin/env python3
"""Failover correctness test for hikyaku.

Drives N sessions × T turns through hikyaku concurrently. After --kill-after
seconds, runs --kill-cmd (a shell command that takes one of the fake_llm
backends down). Reports:

  - Per-session migration pattern: clean A→B (good), bounce A↔B (bad), errors.
  - Pre-kill / post-kill backend distribution.
  - Error counts (timeouts, 5xx) — the failover window itself may produce
    a few; runaway errors after the window indicates the proxy didn't notice
    the dead backend.

Pass criteria (typical):
  - error_rate ≤ ~1% (tight failover window)
  - sessions_pinned_clean + sessions_migrated_clean ≥ 95%
  - sessions_bouncing == 0   (no flapping after migration completes)
  - 0 dispatches to the killed backend after kill_time + ~5s grace

Dependencies: aiohttp.
"""
import argparse
import asyncio
import shlex
import subprocess
import time
from collections import Counter, defaultdict

import aiohttp


SMALL_TURNS = [
    "Reverse this string: hello world",
    "Capital of France?",
    "List 5 sorting algorithms.",
    "Explain TCP in one sentence.",
    "What does 'idempotent' mean?",
    "Difference between TCP and UDP?",
]


async def one_turn(cli, base_url, model, messages, timeout):
    t0 = time.monotonic()
    try:
        async with cli.post(
            f"{base_url}/v1/chat/completions",
            json={"model": model, "messages": messages,
                  "max_tokens": 50, "temperature": 0.0, "stream": False},
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            if resp.status != 200:
                return time.monotonic() - t0, f"ERR_{resp.status}", t0
            await resp.read()
            backend = (resp.headers.get("x-fake-llm-id")
                       or resp.headers.get("x-hikyaku-backend") or "UNKNOWN")
            return time.monotonic() - t0, backend, t0
    except asyncio.TimeoutError:
        return time.monotonic() - t0, "ERR_TIMEOUT", t0
    except aiohttp.ClientError as e:
        return time.monotonic() - t0, f"ERR_CLIENT_{type(e).__name__}", t0


async def run_session(sid, base_url, model, turns, sem, kill_event, kill_time_holder):
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": f"Session {sid}: {SMALL_TURNS[sid % len(SMALL_TURNS)]}"},
    ]
    history = []  # list of (t_start, backend)
    async with aiohttp.ClientSession() as cli:
        for t in range(turns):
            async with sem:
                _, backend, t_start = await one_turn(cli, base_url, model, messages, timeout=30)
            history.append((t_start, backend))
            messages.append({"role": "assistant", "content": "ack"})
            if t + 1 < turns:
                messages.append({"role": "user", "content": f"turn {t+1}: continue"})
    return sid, history


async def kill_after(seconds, kill_cmd, kill_time_holder):
    await asyncio.sleep(seconds)
    print(f"\n>>> [{seconds:.1f}s] executing kill: {kill_cmd}")
    kill_time_holder["t"] = time.monotonic()
    proc = await asyncio.create_subprocess_shell(
        kill_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    print(f">>> kill exit {proc.returncode}")
    if out:
        print(f">>> stdout: {out.decode().strip()[:200]}")
    if err:
        print(f">>> stderr: {err.decode().strip()[:200]}")


def classify_session(history, kill_time, killed_backend):
    """Classify a session's behaviour. Returns one of:
    - 'pinned_clean'    : all turns on one healthy backend
    - 'pinned_to_dead'  : all turns on the killed backend (test ended before migration needed)
    - 'migrated_clean'  : turns on killed backend before kill, then exclusively a different backend after
    - 'bouncing'        : turns flip-flop between healthy backends (BAD)
    - 'errors'          : any ERR_* outcomes
    """
    backends = [b for _, b in history]
    if any(b.startswith("ERR_") for b in backends):
        return "errors"
    distinct = set(backends)
    if len(distinct) == 1:
        only = next(iter(distinct))
        return "pinned_to_dead" if only == killed_backend else "pinned_clean"
    if kill_time is None:
        return "bouncing"  # multi-backend without kill = bug
    pre = [b for t, b in history if t < kill_time]
    post = [b for t, b in history if t >= kill_time]
    pre_distinct = set(pre)
    post_distinct = set(post)
    if len(post_distinct) <= 1 and (not post or killed_backend not in post_distinct):
        return "migrated_clean"
    return "bouncing"


def report(results, kill_time, killed_backend):
    classifications = Counter()
    pre_dist = Counter()
    post_dist = Counter()
    err_count = 0
    total_reqs = 0
    post_kill_to_dead = 0
    grace_post = (kill_time or 0) + 5.0  # 5s grace window after kill

    for sid, history in results:
        c = classify_session(history, kill_time, killed_backend)
        classifications[c] += 1
        for t, b in history:
            total_reqs += 1
            if b.startswith("ERR_"):
                err_count += 1
                continue
            if kill_time is None or t < kill_time:
                pre_dist[b] += 1
            else:
                post_dist[b] += 1
                if b == killed_backend and t > grace_post:
                    post_kill_to_dead += 1

    print()
    print("=" * 60)
    print("FAILOVER REPORT")
    print("=" * 60)
    print(f"Total sessions:    {sum(classifications.values())}")
    print(f"Total HTTP reqs:   {total_reqs}")
    print(f"Errors:            {err_count}  ({100*err_count/max(total_reqs,1):.1f}%)")
    print()
    print("Session classification:")
    for k in ("pinned_clean", "pinned_to_dead", "migrated_clean", "bouncing", "errors"):
        v = classifications.get(k, 0)
        marker = ""
        if k == "bouncing" and v > 0:
            marker = "  <-- BAD: should be 0"
        if k == "migrated_clean" and v > 0:
            marker = "  <-- expected behaviour for sessions on the killed backend"
        print(f"  {k:<20s} {v:>4d}{marker}")
    print()
    print(f"Pre-kill distribution:  {dict(pre_dist)}")
    print(f"Post-kill distribution: {dict(post_dist)}")
    print(f"Post-kill (after {kill_time+5:.1f}s grace) routes to killed backend: {post_kill_to_dead}")
    if post_kill_to_dead > 0:
        print("  ^^ BAD: proxy still routing to dead backend past grace window")

    sessions = sum(classifications.values())
    clean = classifications.get("pinned_clean", 0) + classifications.get("migrated_clean", 0)
    pct = 100 * clean / max(sessions, 1)
    err_rate = 100 * err_count / max(total_reqs, 1)
    bouncing = classifications.get("bouncing", 0)
    verdict = "PASS" if pct >= 95 and bouncing == 0 and err_rate <= 2 and post_kill_to_dead == 0 else "FAIL"
    print()
    print(f"Clean (pinned or migrated): {pct:.1f}%")
    print(f"VERDICT: {verdict}")


async def amain():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sessions", type=int, default=20)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--kill-after", type=float, default=5.0,
                    help="seconds after start to run --kill-cmd")
    ap.add_argument("--kill-cmd", required=True,
                    help="shell command that kills a backend, e.g. 'pkill -f \"fake-A\"'")
    ap.add_argument("--killed-backend", required=True,
                    help="X-Fake-LLM-Id (or hikyaku backend id) that gets killed; for classification")
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    kill_event = asyncio.Event()
    kill_time_holder = {"t": None}

    print(f"Driving {args.sessions} sessions × {args.turns} turns @ c={args.concurrency} "
          f"against {args.base_url} model={args.model}")
    print(f"Killing {args.killed_backend} at t+{args.kill_after}s via: {args.kill_cmd}")

    t_start = time.monotonic()
    sessions_task = asyncio.gather(*(
        run_session(sid, args.base_url, args.model, args.turns, sem, kill_event, kill_time_holder)
        for sid in range(args.sessions)
    ))
    kill_task = asyncio.create_task(kill_after(args.kill_after, args.kill_cmd, kill_time_holder))

    results = await sessions_task
    await kill_task
    elapsed = time.monotonic() - t_start
    kill_time = kill_time_holder["t"] - t_start if kill_time_holder["t"] else None
    # Convert each turn's t_start to relative time (s since test start) for classification
    rel_results = [(sid, [(t - t_start, b) for t, b in hist]) for sid, hist in results]

    print(f"\nTest finished in {elapsed:.1f}s, kill_time={kill_time}")
    report(rel_results, kill_time, args.killed_backend)


if __name__ == "__main__":
    import traceback
    try:
        asyncio.run(amain())
    except Exception:
        print("\n!!! UNHANDLED EXCEPTION !!!")
        traceback.print_exc()
        raise
