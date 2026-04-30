# OS Tuning for Hikyaku Load Testing

Defaults on most Linux distros are sized for desktop / server-application
workloads, not for high-concurrency reverse proxies pushing 10K+ RPS on
a single machine. Without tuning, you'll hit file-descriptor limits, TCP
listen-backlog drops, ephemeral-port exhaustion, or connection-tracking
table fills before you find hikyaku's actual ceiling.

This file is the canonical pre-flight checklist for running hikyaku
benchmarks. Apply once per test machine; verify before each test session.

The tunables here apply to **the load-generator host, the proxy host,
and any backend hosts** — wherever a process opens a lot of TCP sockets.

---

## TL;DR — minimum viable tuning

For a one-off benchmark session in the current shell:

```bash
ulimit -n 1048576
sudo sysctl -w \
  net.core.somaxconn=65535 \
  net.ipv4.tcp_max_syn_backlog=65535 \
  net.ipv4.ip_local_port_range="10000 65535" \
  net.ipv4.tcp_tw_reuse=1
```

For Docker-hosted hikyaku, add `--ulimit nofile=1048576:1048576` to the
`docker run` command (or set it in compose / the systemd unit).

That's the 80% case. Read on for the why and the persistent versions.

---

## 1. File descriptor limits (`nofile`)

**The most common limit you'll hit.** Default soft `nofile` is **1024**
on many distros. At 10K RPS with multi-process load generators (e.g.
Locust `--processes -1` on a 16-thread CPU spawns ~16 workers), each
worker plus the proxy plus the backends hold hundreds-to-thousands of
concurrent TCP connections. 1024 falls over instantly.

Symptom when this fails:

```
OSError: [Errno 24] Too many open files
```

or proxy-side:

```
HTTP 502: read tcp ...: connection reset by peer
```

(the latter happens when the *backend* exhausts FDs, not the proxy)

### One-shot (current shell only)

```bash
ulimit -n 1048576    # set soft limit
ulimit -n            # verify
ulimit -nH           # check hard limit; must be ≥ soft
```

If the hard limit is below your target, raise it as root:

```bash
sudo prlimit --pid $$ --nofile=1048576:1048576
```

### Persistent (per-user)

Edit `/etc/security/limits.conf` (or drop a file in
`/etc/security/limits.d/`):

```conf
# /etc/security/limits.d/99-hikyaku.conf
*       soft    nofile  1048576
*       hard    nofile  1048576
root    soft    nofile  1048576
root    hard    nofile  1048576
```

Log out and back in for it to take effect. Verify with `ulimit -n`.

### Persistent (systemd service)

If hikyaku or the fakes run under systemd, add to the unit file
(`/etc/systemd/system/hikyaku.service` or similar):

```ini
[Service]
LimitNOFILE=1048576
```

Then `sudo systemctl daemon-reload && sudo systemctl restart hikyaku`.

### Docker

Containers inherit ulimits from the Docker daemon's defaults, *not*
from the host's `limits.conf`. You must set them explicitly:

**At run time (one-off):**

```bash
docker run --ulimit nofile=1048576:1048576 ... <image>
```

**On an existing running container:**

```bash
docker update --ulimit nofile=1048576:1048576 <container>
docker restart <container>
```

**In docker-compose:**

```yaml
services:
  hikyaku:
    ulimits:
      nofile:
        soft: 1048576
        hard: 1048576
```

**Daemon default for all containers:**

In `/etc/docker/daemon.json`:

```json
{
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Soft": 1048576,
      "Hard": 1048576
    }
  }
}
```

Restart the daemon: `sudo systemctl restart docker`.

**Verify inside the container:**

```bash
docker exec <container> sh -c 'ulimit -n'
```

---

## 2. TCP listen backlog (`somaxconn`)

When new TCP connections arrive faster than `accept()` calls drain the
backlog, the kernel queues them. Default backlog is **4096** on recent
kernels but historically **128**. Under burst load with a slow
accept-loop (e.g. asyncio under contention), the queue overflows and
new connections get RST'd.

Symptom: sporadic `connection reset by peer` from clients, with no
corresponding application-side log entry on the server.

```bash
# one-shot
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# persistent
echo 'net.core.somaxconn = 65535' | sudo tee -a /etc/sysctl.d/99-hikyaku.conf
echo 'net.ipv4.tcp_max_syn_backlog = 65535' | sudo tee -a /etc/sysctl.d/99-hikyaku.conf
sudo sysctl --system
```

The application also has to ask for the larger backlog when it calls
`listen()`. Most production-grade servers do; aiohttp's default is
128, override with the `--backlog` argument or in code. Hikyaku in Go
follows whatever its configured server says — check its config.

---

## 3. Ephemeral port range (`ip_local_port_range`)

Every outbound TCP connection consumes one ephemeral port from the
local port range. Default is `32768-60999` on most distros = ~28K
ports. At 10K RPS with short-lived connections, ports recycle fast
enough through TIME_WAIT (default 60-120s after close) that you can
run out.

Symptom: `connect: cannot assign requested address` from the proxy or
load generator.

```bash
# one-shot — expand to ~55K ports
sudo sysctl -w net.ipv4.ip_local_port_range="10000 65535"

# persistent
echo 'net.ipv4.ip_local_port_range = 10000 65535' | sudo tee -a /etc/sysctl.d/99-hikyaku.conf
sudo sysctl --system
```

---

## 4. TIME_WAIT socket reuse (`tcp_tw_reuse`)

When a TCP connection closes, the socket sits in TIME_WAIT for ~60s by
default to handle stray retransmissions. At high RPS this fills the
ephemeral port range with TIME_WAIT entries even after the
`ip_local_port_range` expansion above. `tcp_tw_reuse` lets the kernel
reuse a TIME_WAIT socket for a *new* outgoing connection if the
timestamp is newer than the old socket — safe and standard since
kernel 4.12.

```bash
# one-shot
sudo sysctl -w net.ipv4.tcp_tw_reuse=1

# persistent
echo 'net.ipv4.tcp_tw_reuse = 1' | sudo tee -a /etc/sysctl.d/99-hikyaku.conf
sudo sysctl --system
```

**Do NOT enable `tcp_tw_recycle`.** It was removed in kernel 4.12 for
good reason (broke NAT clients). Only `tcp_tw_reuse` is safe.

---

## 5. Connection tracking (`nf_conntrack_max`)

If your host has iptables/nftables connection tracking enabled
(common on systems running Docker with bridge networking), the
conntrack table has a max entry count. Default is typically 65,536 or
262,144. At 10K RPS with short connections, this fills fast.

Symptom: `nf_conntrack: table full, dropping packet` in
`dmesg` / kernel log; sporadic SYN drops.

```bash
# one-shot — bump to 1M entries
sudo sysctl -w net.netfilter.nf_conntrack_max=1048576

# persistent
echo 'net.netfilter.nf_conntrack_max = 1048576' | sudo tee -a /etc/sysctl.d/99-hikyaku.conf
sudo sysctl --system
```

Check current usage:

```bash
sudo sysctl net.netfilter.nf_conntrack_count
sudo sysctl net.netfilter.nf_conntrack_max
```

If your test machine doesn't run iptables/nftables/Docker, you can
skip this — conntrack isn't loaded and the file won't exist.

---

## 6. Socket buffers (usually fine at defaults)

Modern Linux defaults are usually sufficient. Bump only if you observe
TCP buffer-fill stalls (rare for short JSON payloads, common for large
streaming responses).

```bash
# only if needed — increase max receive/send buffers
sudo sysctl -w net.core.rmem_max=16777216
sudo sysctl -w net.core.wmem_max=16777216
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 16777216"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 16777216"
```

For typical hikyaku benchmarks (small JSON requests, short responses),
defaults are fine. Tune if you're benchmarking large-context streaming.

---

## 7. Optional: bypass ipv6 (if not needed)

If your test setup is IPv4-only, disabling IPv6 on the test interface
can shave a few µs of name-resolution and listen-socket setup. Usually
not worth the operational cost. Skip unless profiling indicates IPv6
overhead.

---

## Setup script (apply everything at once)

```bash
#!/bin/bash
# hikyaku-tune.sh — apply all benchmark-friendly tunings at once.
# Idempotent. Requires sudo.

set -euo pipefail

# 1. File descriptors (current shell)
ulimit -n 1048576

# 2. Persistent FD limit (root)
sudo tee /etc/security/limits.d/99-hikyaku.conf > /dev/null <<EOF
*       soft    nofile  1048576
*       hard    nofile  1048576
root    soft    nofile  1048576
root    hard    nofile  1048576
EOF

# 3. Network stack
sudo tee /etc/sysctl.d/99-hikyaku.conf > /dev/null <<EOF
# hikyaku benchmark tuning
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 65535
net.ipv4.ip_local_port_range = 10000 65535
net.ipv4.tcp_tw_reuse = 1
EOF

# 4. Conntrack — only if module is loaded
if [[ -f /proc/sys/net/netfilter/nf_conntrack_max ]]; then
    echo 'net.netfilter.nf_conntrack_max = 1048576' \
        | sudo tee -a /etc/sysctl.d/99-hikyaku.conf > /dev/null
fi

sudo sysctl --system

echo
echo "=== applied ==="
ulimit -n
sudo sysctl net.core.somaxconn
sudo sysctl net.ipv4.tcp_max_syn_backlog
sudo sysctl net.ipv4.ip_local_port_range
sudo sysctl net.ipv4.tcp_tw_reuse
[[ -f /proc/sys/net/netfilter/nf_conntrack_max ]] && \
    sudo sysctl net.netfilter.nf_conntrack_max

echo
echo "Logout/login (or new shell) for the persistent FD limit to take effect."
```

Save as `tune.sh`, run with `bash tune.sh`. Re-running is safe.

---

## Verification before a benchmark run

Quick checklist to confirm everything's in place:

```bash
# 1. FD limit on the launching shell
ulimit -n           # expect 1048576 (or at least >> 1024)

# 2. Network sysctls
sysctl net.core.somaxconn                # expect 65535
sysctl net.ipv4.ip_local_port_range      # expect 10000 65535
sysctl net.ipv4.tcp_tw_reuse             # expect 1

# 3. Inside the hikyaku container (if Dockerized)
docker exec <hikyaku-container> sh -c 'ulimit -n'   # expect 1048576

# 4. During the run (in another terminal):
ss -s                       # current socket counts
sudo sysctl net.netfilter.nf_conntrack_count   # if applicable
sudo dmesg -T | tail        # any kernel warnings (conntrack full, etc.)
```

If any of these are at default values, the benchmark numbers are
artificially constrained.

---

## What "good" looks like during a run

While the benchmark is running, in another terminal:

```bash
# socket state — should mostly be ESTAB and TIME-WAIT
ss -s
# Total: 12345 (kernel 0)
# TCP:   8000 (estab 5000, closed 3000, ...)

# per-process FD usage — confirm nowhere near limit
for pid in $(pgrep -f hikyaku) $(pgrep -f fake_llm) $(pgrep -f locust); do
    n=$(ls -l /proc/$pid/fd 2>/dev/null | wc -l)
    cmd=$(ps -p $pid -o comm=)
    echo "PID $pid ($cmd): $n FDs"
done

# kernel ring buffer — should be quiet during a clean run
sudo dmesg -T | grep -iE "conntrack|drop|reset" | tail
```

If hikyaku's FD count is climbing without bound during a run, there's
either a connection leak (bug) or backend connections are pooling
faster than they're being reused (config issue).

---

## Per-distro notes

- **Ubuntu 22.04 / Debian 12**: defaults applied above work cleanly.
- **RHEL / CentOS Stream / Rocky**: same sysctl names; SELinux may
  interfere with `prlimit` on some configurations — disable it for
  test rigs (`setenforce 0`) if you see permission denied errors.
- **Alpine** (common in containers): some sysctls aren't writable
  inside an unprivileged container; set them on the host instead, or
  run with `--privileged` in dev.
- **WSL2**: limits are inherited from the WSL VM, not the Windows
  host. Apply the persistent file inside WSL.

---

## Why this should be upstreamed to hikyaku

Anyone benchmarking hikyaku will hit these limits. Without this doc:
- They'll see weird numbers and conclude hikyaku is the bottleneck
- They'll file misleading performance issues
- Their published numbers will under-represent hikyaku's actual capacity

With this doc shipped alongside the benchmark scripts, the test
methodology is reproducible and the numbers are credible.

Suggested home in the hikyaku repo:

```
hikyaku/
├── docs/
│   ├── BENCHMARKING.md          (the test plan)
│   └── TUNING.md                (this file)
└── ...
```

Or under `tests/` alongside the load-test scripts. Either is fine.
