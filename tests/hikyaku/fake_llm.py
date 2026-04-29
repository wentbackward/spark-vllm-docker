#!/usr/bin/env python3
"""Fake OpenAI-compatible LLM for hikyaku latency testing.

Implements the minimum surface hikyaku needs to treat this as a real backend:
  GET  /v1/models               — model list
  POST /v1/chat/completions     — canned response (streaming or full)
  GET  /metrics                 — Prometheus format (vllm:* gauges)

Each response carries:
  X-Fake-LLM-Id: <id>           — backend identifier (so the harness can
                                  see which fake handled each request)

Inference time is simulated as: ttft_ms + (response_tokens × tpot_ms).
Defaults give ~270 ms total for 50 tokens, which is realistic enough to
expose proxy overhead while keeping the test run fast.

Run two instances on different ports to give the proxy a backend pool.

Dependencies: aiohttp (pip install aiohttp).
"""
import argparse
import asyncio
import json
import time
from aiohttp import web


CANNED_WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
    "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
]


def make_canned(n_tokens: int) -> str:
    return " ".join(CANNED_WORDS[i % len(CANNED_WORDS)] for i in range(n_tokens))


class FakeLLMServer:
    def __init__(self, server_id, model_name, response_tokens, ttft_ms, tpot_ms):
        self.server_id = server_id
        self.model_name = model_name
        self.response_tokens = response_tokens
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.running_reqs = 0
        self.completed_reqs = 0

    def _id_header(self):
        return {"X-Fake-LLM-Id": self.server_id}

    async def handle_models(self, request):
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {"id": self.model_name, "object": "model", "owned_by": "fake"}
                ],
            },
            headers=self._id_header(),
        )

    async def handle_metrics(self, request):
        # vLLM-shaped Prometheus metrics — just enough for hikyaku's scraper
        # to recognize the backend as a vLLM and parse load gauges.
        text = (
            "# HELP vllm:num_requests_running\n"
            "# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running {self.running_reqs}\n"
            "# HELP vllm:num_requests_waiting\n"
            "# TYPE vllm:num_requests_waiting gauge\n"
            "vllm:num_requests_waiting 0\n"
            "# HELP vllm:gpu_cache_usage_perc\n"
            "# TYPE vllm:gpu_cache_usage_perc gauge\n"
            "vllm:gpu_cache_usage_perc 0.10\n"
        )
        return web.Response(
            text=text, content_type="text/plain", headers=self._id_header()
        )

    async def handle_chat(self, request):
        payload = await request.json()
        stream = bool(payload.get("stream", False))

        n_tokens = self.response_tokens
        max_tok = payload.get("max_tokens")
        if max_tok is not None:
            n_tokens = min(n_tokens, int(max_tok))

        self.running_reqs += 1
        try:
            if stream:
                return await self._stream_response(request, n_tokens)
            return await self._full_response(n_tokens)
        finally:
            self.running_reqs -= 1
            self.completed_reqs += 1

    async def _full_response(self, n_tokens):
        await asyncio.sleep(self.ttft_ms / 1000.0)
        await asyncio.sleep(n_tokens * self.tpot_ms / 1000.0)
        completion_id = f"chatcmpl-fake-{int(time.time() * 1000)}"
        return web.json_response(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": make_canned(n_tokens)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": n_tokens,
                    "total_tokens": 100 + n_tokens,
                },
            },
            headers=self._id_header(),
        )

    async def _stream_response(self, request, n_tokens):
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                **self._id_header(),
            },
        )
        await resp.prepare(request)
        await asyncio.sleep(self.ttft_ms / 1000.0)
        words = make_canned(n_tokens).split()
        completion_id = f"chatcmpl-fake-{int(time.time() * 1000)}"
        for i, word in enumerate(words):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {"index": 0, "delta": {"content": word + " "}, "finish_reason": None}
                ],
            }
            await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if self.tpot_ms > 0 and i + 1 < len(words):
                await asyncio.sleep(self.tpot_ms / 1000.0)
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        await resp.write(f"data: {json.dumps(final)}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--id", required=True, help="server identifier (returned in X-Fake-LLM-Id)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--model", default="fake/canned-1B")
    ap.add_argument("--response-tokens", type=int, default=50,
                    help="canned response length in tokens (default 50)")
    ap.add_argument("--ttft-ms", type=int, default=20,
                    help="simulated time-to-first-token in ms (default 20)")
    ap.add_argument("--tpot-ms", type=float, default=5.0,
                    help="simulated time-per-output-token in ms (default 5)")
    args = ap.parse_args()

    srv = FakeLLMServer(args.id, args.model, args.response_tokens,
                        args.ttft_ms, args.tpot_ms)
    app = web.Application()
    # Register both /v1/... and stripped /... paths — proxies vary on whether
    # they pass the version prefix through to the backend.
    app.router.add_get("/v1/models", srv.handle_models)
    app.router.add_get("/models", srv.handle_models)
    app.router.add_get("/metrics", srv.handle_metrics)
    app.router.add_post("/v1/chat/completions", srv.handle_chat)
    app.router.add_post("/chat/completions", srv.handle_chat)

    print(f"fake_llm id={args.id} listening on {args.host}:{args.port}")
    print(f"  model={args.model}  ttft={args.ttft_ms}ms  tpot={args.tpot_ms}ms  "
          f"resp_tokens={args.response_tokens}")
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
