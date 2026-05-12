#!/usr/bin/env python3
"""Ask a SmolVLM2 endpoint about an image (or video).

Usage:
    vlm-ask.py <file> <prompt> [options]

Examples:
    vlm-ask.py receipt.jpg "OCR this — list line items and totals"
    vlm-ask.py diagram.png "Explain what this shows" --max-tokens 800
    vlm-ask.py clip.mp4 "Summarise what happens" --model 500m

Endpoints (on spark-01, served OpenAI-compatibly by vLLM):
    --model 2.2b  -> :3041  HuggingFaceTB/SmolVLM2-2.2B-Instruct       (default)
    --model 500m  -> :3042  HuggingFaceTB/SmolVLM2-500M-Video-Instruct (video too)

Stdlib only — no pip deps. Pass --host if `spark-01` doesn't resolve from
where you're running this.
"""
import argparse
import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request

# Windows `cmd.exe` legacy codepages choke on non-ASCII model output;
# force UTF-8 stdout where the runtime supports it (Python 3.7+). Harmless
# everywhere else.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ENDPOINTS = {
    "2.2b": (3041, "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    "500m": (3042, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.splitlines()[1:]),
    )
    ap.add_argument("file", help="image or video file to send")
    ap.add_argument("prompt", help="the question / instruction")
    ap.add_argument("--model", choices=list(ENDPOINTS), default="2.2b",
                    help="which endpoint (default: 2.2b)")
    ap.add_argument("--host", default="spark-01",
                    help="hostname/IP of the vLLM box (default: spark-01)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--timeout", type=float, default=120,
                    help="HTTP timeout in seconds (default: 120)")
    ap.add_argument("--raw", action="store_true",
                    help="print the full JSON response, not just the text")
    args = ap.parse_args()

    port, model = ENDPOINTS[args.model]
    url = f"http://{args.host}:{port}/v1/chat/completions"

    mime, _ = mimetypes.guess_type(args.file)
    if mime is None:
        mime = "application/octet-stream"
    try:
        with open(args.file, "rb") as fh:
            data = fh.read()
    except OSError as e:
        sys.exit(f"can't read {args.file}: {e}")
    data_uri = f"data:{mime};base64,{base64.b64encode(data).decode()}"

    if mime.startswith("video/"):
        media_part = {"type": "video_url", "video_url": {"url": data_uri}}
    else:
        # images, and anything else we couldn't identify — treat as an image
        media_part = {"type": "image_url", "image_url": {"url": data_uri}}

    body = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": args.prompt},
            media_part,
        ]}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            out = json.load(resp)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {url}:\n{e.read().decode(errors='replace')[:800]}")
    except urllib.error.URLError as e:
        sys.exit(f"can't reach {url}: {e.reason}")

    if args.raw:
        print(json.dumps(out, indent=2))
    else:
        print(out["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
