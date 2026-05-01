# Project guidance for AI assistants

**Read [`KNOWLEDGE.md`](./KNOWLEDGE.md) first** for the durable picture
and **[`HANDOFF.md`](./HANDOFF.md) second** for what's currently in
flight and what's next. KNOWLEDGE.md is the durable, repo-bound source
of truth (hardware, network, model choices, conventions, gotchas).
HANDOFF.md is a session-bridging brief that goes stale faster — verify
its claims against current state before acting.

When you learn something durable while working in this repo, **update
KNOWLEDGE.md** rather than relying on session memory. Knowledge that lives
only in chat or in agent memory is lost when the repo moves or the session
ends.

## Repo orientation (non-redundant pointers)

- `recipes/` — vLLM launch configs. `KNOWLEDGE.md` §8 indexes them by purpose.
- `run-recipe.py` / `launch-cluster.sh` — solo and multi-node launchers.
- `autodiscover.sh` — network-topology detection. Has a known limitation on
  the current Spark setup; see `KNOWLEDGE.md` §2 for the workaround.
- `mods/` — patches applied to the container at launch (e.g. chat templates).
  Lost on container restart; recipes re-apply them automatically.
- `resolver/` — test harness scaffolding (partly aspirational, see §9).
- `.env` — manual cluster config; do not delete (autodiscover would clobber
  it with values that fail on this hardware).

## Working preferences

- Don't add CLAUDE.md or AGENTS.md content that duplicates KNOWLEDGE.md.
  Point to it instead.
- Don't write transient state (today's containers, current memory) into
  durable files. Those belong in commit messages or chat.
- Recipes are the operational ground truth — edit them rather than passing
  long flag overrides at the command line, unless the override is genuinely
  one-shot.
