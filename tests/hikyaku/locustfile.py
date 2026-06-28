"""
Locust load profile for hikyaku.

Locust is a Python load-testing tool that runs many simulated "users" in
parallel against an HTTP service. Each user defined here repeatedly hits
hikyaku at /v1/chat/completions; Locust aggregates throughput, latency
percentiles, and error rate, with live charts in a browser UI.

Quickstart:

    pip install locust

    # web UI at http://localhost:8089 — set users + spawn rate, then start
    locust -f locustfile.py --host http://limone:4000

    # headless (command-line, no UI) — auto-stop after duration
    locust -f locustfile.py --host http://limone:4000 \\
        --headless --users 1000 --spawn-rate 100 --run-time 60s

Modes (set via env var before invoking locust):

    HIKYAKU_MODE=independent  (default) — each user sends fresh prompts
                                          tests distribution + raw RPS

    HIKYAKU_MODE=affinity              — each user runs multi-turn
                                          sessions; tests stickiness

Payload sizes:

    HIKYAKU_PAYLOAD=minimal            — ~20 byte prompts; max RPS test
    HIKYAKU_PAYLOAD=small  (default)   — ~80 byte prompts; realistic light
    HIKYAKU_PAYLOAD=large              — ~50 KB system prompts; tests
                                          proxy parsing + hashing under load

Other tunables:

    HIKYAKU_MODEL=test-route           — route name in hikyaku
    HIKYAKU_MAX_TOKENS=50              — tokens per response
    HIKYAKU_TURNS=4                    — turns per session (affinity mode)

Environment example:

    HIKYAKU_MODE=affinity HIKYAKU_PAYLOAD=large \\
    locust -f locustfile.py --host http://hikyaku:4000

Backend distribution and affinity hit rate are printed at the end of
each run.

Dependencies: locust (pip install locust). Locust pulls gevent + flask.
"""
import os
import random
import string
from collections import Counter
from threading import Lock

from locust import HttpUser, between, events, task


# ---------------------------------------------------------------------------
# Configuration (env-driven so locust CLI stays simple)
# ---------------------------------------------------------------------------

MODE = os.getenv("HIKYAKU_MODE", "independent")
PAYLOAD = os.getenv("HIKYAKU_PAYLOAD", "small")
MODEL = os.getenv("HIKYAKU_MODEL", "test-route")
MAX_TOKENS = int(os.getenv("HIKYAKU_MAX_TOKENS", "50"))
TURNS_PER_SESSION = int(os.getenv("HIKYAKU_TURNS", "4"))


SMALL_TURNS = [
    "Reverse this string: hello world",
    "Capital of France?",
    "Difference between TCP and UDP?",
    "Explain idempotent in one sentence.",
    "What's a hash table?",
    "Define monad in 10 words or fewer.",
    "How does TLS handshake work briefly?",
]


def _small_sys():
    return "You are a concise assistant."


def _large_sys():
    seed = "You are a senior code reviewer reviewing the following module. " * 25
    target_bytes = 50_000
    if len(seed) >= target_bytes:
        return seed[:target_bytes]
    filler = "".join(random.choices(string.ascii_letters + " " * 5,
                                    k=target_bytes - len(seed)))
    return seed + filler


def _minimal_sys():
    return "ok"


# Generated once at import — same large prompt across all users (realistic;
# clients usually share a system prompt across sessions of the same role).
_LARGE_SYS = _large_sys()


def _sys_prompt():
    if PAYLOAD == "minimal":
        return _minimal_sys()
    if PAYLOAD == "large":
        return _LARGE_SYS
    return _small_sys()


# ---------------------------------------------------------------------------
# Aggregate trackers (thread-safe; locust runs greenlets via gevent)
# ---------------------------------------------------------------------------

_lock = Lock()
_backend_hits = Counter()
_session_backends = {}   # session_id (str) → set of backend ids seen


def _record_backend(session_id: str | None, backend: str):
    with _lock:
        _backend_hits[backend] += 1
        if session_id is not None:
            s = _session_backends.setdefault(session_id, set())
            s.add(backend)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Print backend distribution + affinity hit rate after the run."""
    print("\n" + "=" * 60)
    print(" hikyaku load test — final tallies")
    print("=" * 60)
    total = sum(_backend_hits.values())
    print(f"\nBackend distribution ({total} requests):")
    if total:
        for backend, count in sorted(_backend_hits.items()):
            pct = 100 * count / total
            bar = "#" * int(pct / 2)
            print(f"  {backend:24s} {count:>7}  {pct:5.1f}%  {bar}")
    if MODE == "affinity":
        with _lock:
            n_sessions = len(_session_backends)
            sticky = sum(1 for s in _session_backends.values() if len(s) == 1)
        if n_sessions:
            rate = sticky / n_sessions
            verdict = "PASS" if rate >= 0.95 else "FAIL — expected ≥95%"
            print(f"\nAffinity hit rate: {sticky}/{n_sessions} = {rate:.1%}  ({verdict})")


# ---------------------------------------------------------------------------
# User definition
# ---------------------------------------------------------------------------

class HikyakuUser(HttpUser):
    """
    One simulated user. Locust spawns many of these and runs each
    independently, calling tasks in a loop with `wait_time` between.

    `wait_time = between(0, 0)` means zero think time → maximum RPS per user.
    For more realistic traffic, use e.g. `between(1, 3)` (1-3 sec idle
    between requests).
    """
    wait_time = between(0, 0)

    def on_start(self):
        """Called once when this user starts. Per-user state setup."""
        self._new_session()

    def _new_session(self):
        """Initialize a fresh affinity-mode session."""
        # Use a high-entropy session id so opening prompts are unique
        self.session_id = f"S{random.randrange(2**62):x}"
        self.opening = f"{self.session_id}: {random.choice(SMALL_TURNS)}"
        self.turn = 0
        self.messages = [
            {"role": "system", "content": _sys_prompt()},
            {"role": "user", "content": self.opening},
        ]

    @task
    def chat_completion(self):
        if MODE == "independent":
            self._independent()
        else:
            self._affinity()

    # -- mode implementations ---------------------------------------------

    def _independent(self):
        """Fresh user prompt every call. Each request is a brand-new
        affinity key (no stickiness expected)."""
        messages = [
            {"role": "system", "content": _sys_prompt()},
            {"role": "user", "content": f"req-{random.randrange(2**31)}: "
                                        f"{random.choice(SMALL_TURNS)}"},
        ]
        self._post(messages, session_id=None)

    def _affinity(self):
        """Continue the current session. After TURNS_PER_SESSION turns,
        rotate to a new session (so a long-running user simulates many
        sessions over time)."""
        self._post(self.messages, session_id=self.session_id)
        # Append assistant + next user turn
        self.messages.append({"role": "assistant", "content": "ack"})
        self.turn += 1
        if self.turn >= TURNS_PER_SESSION:
            self._new_session()
        else:
            self.messages.append(
                {"role": "user", "content": f"turn-{self.turn}: continue"}
            )

    # -- HTTP ------------------------------------------------------------

    def _post(self, messages, session_id):
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
            catch_response=True,
            name="/v1/chat/completions",   # group all requests under one name
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            backend = (
                resp.headers.get("x-hikyaku-backend")
                or resp.headers.get("x-fake-llm-id")
                or "UNKNOWN"
            )
            _record_backend(session_id, backend)
