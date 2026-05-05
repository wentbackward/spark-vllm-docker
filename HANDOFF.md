# Session Handoff

Brief for a fresh session resuming this work. Pair with `KNOWLEDGE.md`
(durable repo facts) and `tests/hikyaku/TUNING-REPORT.md` (perf testing
journal). This file is short on purpose — read those for depth.

## Two parallel projects in flight

1. **DGX Spark vLLM serving** — this repo. Recipes, multi-node cluster,
   model lifecycle.
2. **Hikyaku** — separate Go reverse-proxy/router on `limone` (Jetson
   Orin Nano). The proxy was renamed from `llm-proxy` to `hikyaku`.
   Design specs live here at `docs/LOAD-BALANCING.md` (a reference
   copy; the canonical home is the hikyaku repo elsewhere).
   - **UAT certification work happens in `~/hacking/hikyaku-pro`.**
     For the current state of that effort, read
     [`hikyaku-pro/HANDOFF.md`](../hikyaku-pro/HANDOFF.md) — that's
     where the per-scenario verdicts, KNOWN-ISSUES, and Pro-tier
     requirements drafts live.

## Current operational state (as of this handoff)

| host | port | container | model | persistent |
|---|---|---|---|---|
| spark-01 | 3041 | vllm_mtp | Qwen3.6-27B-AWQ-INT4 + native MTP | ✓ via `~/admin/start-vllm.sh` |
| spark-02 | 3042 | vllm_mtp | Qwen3.6-27B-AWQ-INT4 + native MTP | ✓ via systemd `spark-vllm.service` |
| spark-02 | 3043 | vllm_qwen_vl | Qwen2.5-VL-3B-Instruct (vision) | **✗ NOT persistent** — reboot kills it |
| limone | 4000 | (Go process) | hikyaku proxy | (proxy-side) |

35B-A3B is **not currently running anywhere** — openclaw was redirected
to HuggingFace inference earlier; revisit if you want it local again.

## Major decisions made (durable)

- **27B inference config**: AWQ-INT4 (cyankiwi quant) + native MTP at
  `num_speculative_tokens=2`. MTP=3 caused output looping on real
  agentic workloads; reverted. Don't bump back up.
- **`max_model_len: 196608`** (75% of full 262K window). Forces the
  CLI's auto-compaction to trigger before vLLM's hard limit, breaking
  history-driven loops. Lower than this loses headroom; higher
  reintroduces the loop conditions.
- **Hikyaku affinity-key algorithm: `first_user_message`** (hash
  first user msg's content, skip system + assistant turns). The
  earlier `canonical_prefix` algorithm was empirically broken — see
  `docs/LOAD-BALANCING.md` § Affinity for why and the validation data.
- **Hikyaku health = metrics-scrape** when `/metrics` is available;
  fall back to `/models` poll otherwise. Saves a redundant HTTP
  round-trip per backend per interval.
- **VLM choice**: Qwen2.5-VL-3B-Instruct. FastVLM and Moondream were
  tried first; both blocked by architecture-not-in-vLLM-registry and
  custom-code dependency issues. **Don't reattempt either** without
  fresh evidence vLLM has added support.

## Performance testing — completed (significant work)

Yesterday's perf characterization closed all three of the original
concerns about hikyaku:

1. ✅ **Go fast enough?** — yes, ~12,755 RPS sustained on a Ryzen 9
   7940HS minipc, ~5,945 RPS on a Jetson Orin Nano (LAN), per-core
   efficiency ~22× more than published Python proxy alternatives.
2. ✅ **Load balancing works?** — yes for both `round_robin` (perfect
   ±1% distribution) and `sticky_least_loaded` (100% affinity hit
   rate across 12,666 sessions in three independent tests).
3. ✅ **Graceful under pressure?** — yes, <0.001% failure rate at 5×
   saturation, no memory growth, no queue runaway, distribution holds
   through thermal throttling.

Full journal in **`tests/hikyaku/TUNING-REPORT.md`** with run-by-run
numbers, the staircase thermal-throttle data on the Orin Nano, and an
addendum with the affinity-validation results.

Test harness in **`tests/hikyaku/`**:
- `fake_llm.py` — OpenAI-compatible canned-response backend
- `latency_harness.py` — async correctness harness (asyncio caps ~50)
- `locustfile.py` — Locust load profile (scales to thousands of users)
- `README.md` — wire-up + commands
- `TUNING.md` — OS pre-flight (ulimit, sysctl, Docker, systemd)

OS tuning lessons captured in `tests/hikyaku/TUNING.md` — apply before
any future benchmarking on a new test rig.

## What's next (priority-ordered)

### Immediate — small unblocks
1. **Persist `vllm_qwen_vl` across reboots.** Mirror the existing
   spark-02 systemd pattern: add `~/admin/start-vlm.sh` and
   `services/spark-vlm.service`, install + enable. ~5 min job.
2. **Wire VLM into hikyaku** as `gresh-vision` route → `spark-02:3043`,
   `strategy: single`. ~3 min on the proxy config + SIGHUP.

### Hikyaku — Phase 2.5 work (defenders) — DONE
Loop detection, zero-content detection, and drop-empty all landed in
`v0.4.0-dev.21`–`v0.4.0-dev.23` and are verified by
`hikyaku-pro/scenarios/{06,07,08}-defender-*`. Suite is 9/9 PASS on
dev.23 (2nd rebuild). See `hikyaku-pro/HANDOFF.md` for the next
hikyaku-side priorities (10-soak, 11/12-perf scenarios, PRO-003
SLA work).

### Bigger
3. **Production-shape benchmark.** Real vLLM upstream, real
   conversation patterns, prefix-cache-aware workload. Validates that
   hikyaku's affinity routing actually delivers the cache-locality
   benefit it was designed for. *(Partially superseded by the
   PRO-003 work in hikyaku-pro — but production-shape against real
   vLLM is still its own thing.)*
4. **Multi-instance hikyaku.** Two hikyaku instances behind a TCP
   load balancer. Validates routing consistency across replicas.

## Lessons / gotchas worth knowing

- **Don't bounce active vLLM endpoints.** Check `Running:` count AND
  re-read which endpoint the user is on before stopping anything.
  Lesson recorded as a feedback memory note.
- **`--rm` Docker containers don't survive reboots.** That's why the
  systemd unit pattern matters for anything you want persistent.
- **`HF_HUB_OFFLINE=1` + `trust_remote_code` is a trap.** When vLLM
  rewrites the model_id to a local snapshot path, transformers caches
  the dynamic modules under `_<sha>/` (with leading underscore), but
  pre-population by HF model-id puts them under `<owner>/<repo>/<sha>/`
  (no underscore). Files end up in the wrong cache directory, vLLM
  can't find them, "FileNotFoundError" mid-startup. Workaround: copy
  files between the two cache paths after pre-population, or just
  drop `HF_HUB_OFFLINE` for that model.
- **Locust at high RPS needs `--processes -1`.** Single-process Locust
  caps at 4-8K RPS due to Python's GIL — that ceiling can be mistaken
  for hikyaku's ceiling.
- **Architectures not in vLLM's registry can't be loaded** even with
  `--trust-remote-code`. Check `docker logs <container> | grep "not
  supported"` early to avoid chasing dependency dead-ends. Moondream
  (`HfMoondream`) and FastVLM (`llava_qwen2`) are both currently
  outside vLLM's registry.

## Files that matter (in this repo)

```
HANDOFF.md                       — this file
KNOWLEDGE.md                     — durable knowledge base, READ FIRST
CLAUDE.md                        — thin pointer to KNOWLEDGE.md

docs/LOAD-BALANCING.md           — hikyaku design spec (canonical home is
                                   the hikyaku repo; this is a reference)
docs/LOAD-BALANCING-TEST.md      — test plan for hikyaku validation
docs/NETWORKING.md               — network topology (older)

tests/hikyaku/                   — full perf-testing toolkit
  fake_llm.py
  latency_harness.py
  locustfile.py
  README.md
  TUNING.md                      — OS pre-flight checklist
  TUNING-REPORT.md               — perf testing journal + addendum

recipes/qwen3.6-27b-awq-int4-mtp.yaml   — main coding model
recipes/qwen2.5-vl-3b.yaml              — VLM (NEW this session)
recipes/qwen3.6-35b-a3b-fp8.yaml        — 35B (currently unused)
```

`limone` proxy config is in the hikyaku repo (separate, on the Jetson),
not this repo.

## Resuming

1. Read `KNOWLEDGE.md` for the durable picture.
2. Read this file for current state.
3. Read `tests/hikyaku/TUNING-REPORT.md` if relevant to the task.
4. Pick from "What's next" or take a new direction.
