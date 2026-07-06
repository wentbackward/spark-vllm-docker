# Boot scripts (reference copies)

`start-cluster.sh` — the idempotent whole-cluster bring-up script.
**The live copy runs from `~/admin/start-cluster.sh` on spark-01** (invoked
at boot via `spark-services.service` → `start-services.sh` → `start-vllm.sh`);
this repo copy exists so the hardening travels with the repo. If you change
one, sync the other. See `KNOWLEDGE.md` §11 for the full boot/recovery chain
and the 2026-07-06 hardening rationale (serial launches, stale-container
teardown).
