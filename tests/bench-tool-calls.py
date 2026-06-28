#!/usr/bin/env python3
"""Tool-call benchmark for hikyaku routes.

Drives a fixed set of prompts at a route, each one should elicit a
specific tool call. Measures:
  - clean_pct  — response has well-formed structured tool_calls (no XML leak)
  - correct_pct — tool name + required-arg presence match expectation
  - p50 / p95 latency (wall clock)
  - mean generation tokens

Usage:
  python3 bench-tool-calls.py --base-url http://limone:4000 --model worker
  python3 bench-tool-calls.py --base-url http://limone:4000 --model orchestrator --concurrency 1

Lower concurrency for thinker-mode routes (single forward pass takes longer).
"""
import argparse
import asyncio
import json
import statistics
import time
from collections import Counter

import aiohttp


TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "get_stock_price",
        "description": "Get current stock price for a ticker symbol",
        "parameters": {"type": "object",
                       "properties": {"symbol": {"type": "string"}},
                       "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "search_files",
        "description": "Search for files matching a query",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"},
                                      "path": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "send_email",
        "description": "Send an email",
        "parameters": {"type": "object",
                       "properties": {"to": {"type": "string"},
                                      "subject": {"type": "string"},
                                      "body": {"type": "string"}},
                       "required": ["to"]}}},
    {"type": "function", "function": {
        "name": "list_directory",
        "description": "List files in a directory",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "execute_shell",
        "description": "Execute a shell command and return stdout",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}},
]


# (prompt, expected_tool_name, expected_args_subset_or_None)
# expected_args=None means "any args ok, just the tool name must match".
PROMPTS = [
    ("What's the weather in Tokyo right now?",                     "get_weather",      {"city": "Tokyo"}),
    ("Get the weather forecast for Paris.",                        "get_weather",      {"city": "Paris"}),
    ("Look up the current stock price for ticker AAPL.",           "get_stock_price",  {"symbol": "AAPL"}),
    ("What's NVIDIA stock at right now? Symbol NVDA.",             "get_stock_price",  {"symbol": "NVDA"}),
    ("Find all .py files containing 'def main' under /home/user.", "search_files",     None),
    ("Search for files with 'TODO' in /tmp.",                      "search_files",     None),
    ("Read the contents of /etc/hostname for me.",                 "read_file",        {"path": "/etc/hostname"}),
    ("Show me what's in /var/log/syslog.",                         "read_file",        {"path": "/var/log/syslog"}),
    ("Send an email to alice@example.com with subject 'Quick check'.", "send_email",   None),
    ("Email bob@company.org saying the deployment finished.",      "send_email",       None),
    ("List the files in /home directory.",                         "list_directory",   {"path": "/home"}),
    ("Show me what's in /tmp.",                                    "list_directory",   {"path": "/tmp"}),
    ("Run 'ls -la' in the current directory.",                     "execute_shell",    None),
    ("Execute the command 'df -h'.",                               "execute_shell",    None),
    ("What's the temperature in Sydney today?",                    "get_weather",      {"city": "Sydney"}),
]


async def one_request(cli, base_url, model, prompt, expected_tool, expected_args, timeout):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": TOOLS,
        "max_tokens": 1024,
    }
    t0 = time.monotonic()
    try:
        async with cli.post(f"{base_url}/v1/chat/completions",
                            json=body,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            elapsed = time.monotonic() - t0
            if resp.status != 200:
                txt = (await resp.text())[:200]
                return {"prompt": prompt, "expected_tool": expected_tool,
                        "elapsed": elapsed, "clean": False, "correct": False,
                        "error": f"HTTP {resp.status}: {txt}"}
            data = await resp.json()
            elapsed = time.monotonic() - t0
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        return {"prompt": prompt, "expected_tool": expected_tool,
                "elapsed": time.monotonic() - t0, "clean": False, "correct": False,
                "error": f"{type(e).__name__}: {e}"}

    msg = data["choices"][0]["message"]
    finish = data["choices"][0].get("finish_reason")
    gen_tokens = data.get("usage", {}).get("completion_tokens", 0)
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []

    # XML leak detection — content shouldn't contain raw <tool_call> or <function=>
    has_xml_leak = "<tool_call>" in content or "<function=" in content

    # clean = at least one structured tool_call AND no XML leak in content
    clean = len(tool_calls) > 0 and not has_xml_leak

    # correct = clean AND tool name matches AND (if expected_args) those keys are present
    correct = False
    actual_tool = None
    actual_args = {}
    if clean:
        first = tool_calls[0]
        actual_tool = first.get("function", {}).get("name")
        try:
            actual_args = json.loads(first.get("function", {}).get("arguments", "{}"))
        except json.JSONDecodeError:
            actual_args = {}
        if actual_tool == expected_tool:
            if expected_args is None:
                correct = True
            else:
                # case-insensitive value compare for required keys
                correct = all(
                    str(actual_args.get(k, "")).strip().lower() == str(v).strip().lower()
                    for k, v in expected_args.items()
                )

    return {
        "prompt": prompt[:60],
        "expected_tool": expected_tool,
        "actual_tool": actual_tool,
        "expected_args": expected_args,
        "actual_args": actual_args,
        "clean": clean,
        "correct": correct,
        "xml_leak": has_xml_leak,
        "elapsed": elapsed,
        "gen_tokens": gen_tokens,
        "finish": finish,
    }


async def amain():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="hikyaku virtual model name")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=1, help="repeat the prompt set N times")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--save", help="JSON results file")
    ap.add_argument("--show-failures", action="store_true")
    args = ap.parse_args()

    sem = asyncio.Semaphore(args.concurrency)
    work = []
    for r in range(args.repeats):
        for prompt, tool, expected_args in PROMPTS:
            work.append((prompt, tool, expected_args))

    print(f"benchmark: {args.base_url} model={args.model} concurrency={args.concurrency} "
          f"prompts={len(work)} ({len(PROMPTS)}×{args.repeats})")
    t_start = time.monotonic()

    async with aiohttp.ClientSession() as cli:
        async def run_one(p, t, ea):
            async with sem:
                return await one_request(cli, args.base_url, args.model, p, t, ea, args.timeout)
        results = await asyncio.gather(*(run_one(*w) for w in work))

    t_total = time.monotonic() - t_start

    n = len(results)
    clean = sum(1 for r in results if r["clean"])
    correct = sum(1 for r in results if r["correct"])
    leaks = sum(1 for r in results if r.get("xml_leak"))
    errs = sum(1 for r in results if "error" in r)
    elapsed = sorted(r["elapsed"] for r in results)
    gen = [r.get("gen_tokens", 0) for r in results]

    print()
    print(f"=== {args.model} ===")
    print(f"  total reqs:     {n}")
    print(f"  clean tool_calls: {clean}/{n}  ({100*clean/n:.1f}%)")
    print(f"  correct (tool+args): {correct}/{n}  ({100*correct/n:.1f}%)")
    print(f"  xml leaks:      {leaks}")
    print(f"  errors:         {errs}")
    print(f"  p50 latency:    {elapsed[n//2]*1000:.0f} ms")
    print(f"  p95 latency:    {elapsed[min(int(n*0.95), n-1)]*1000:.0f} ms")
    print(f"  mean latency:   {statistics.mean(elapsed)*1000:.0f} ms")
    print(f"  total time:     {t_total:.1f}s")
    print(f"  mean gen tokens: {statistics.mean(gen):.0f}")

    if args.show_failures:
        for r in results:
            if not r["correct"]:
                print(f"  FAIL: {r['prompt']!r}")
                print(f"        expected={r['expected_tool']}({r.get('expected_args')})  "
                      f"got={r.get('actual_tool')}({r.get('actual_args')})")
                if r.get("error"):
                    print(f"        error={r['error']}")
                if r.get("xml_leak"):
                    print(f"        XML LEAK in content")

    if args.save:
        summary = {
            "model": args.model, "base_url": args.base_url,
            "concurrency": args.concurrency, "n": n,
            "clean": clean, "correct": correct, "leaks": leaks, "errors": errs,
            "p50_ms": elapsed[n//2]*1000,
            "p95_ms": elapsed[min(int(n*0.95), n-1)]*1000,
            "mean_ms": statistics.mean(elapsed)*1000,
            "total_s": t_total,
            "mean_gen_tokens": statistics.mean(gen),
            "results": results,
        }
        with open(args.save, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  saved: {args.save}")


if __name__ == "__main__":
    asyncio.run(amain())
