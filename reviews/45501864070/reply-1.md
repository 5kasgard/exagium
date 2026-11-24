<!-- nvidia nvidia/nemotron-3-ultra-550b-a55b status=200 latency=58.49s -->

VERDICT per layer:
1. Event spine: CHANGE — mirroring every event to Git bloats history, hits GitHub size limits, and conflates archive with hot log. Use Cloudflare D1/KV for the spine; snapshot daily to repo.
2. Threads/channels + A2A/MCP: CHANGE — A2A and MCP are unstable specs; building on them now couples you to churn. Start with typed request/response over the spine; adopt protocols only when interop demands it.
3. Key broker: CHANGE — centralizing 27 keys creates a single compromise target and routing complexity (per-provider limits, model caps). Keep keys in GitHub Actions secrets per workflow; add a budget sidecar that logs and enforces limits without holding secrets.
4. Executors: KEEP — GitHub Actions is the right sandbox today, but admit its flakiness; plan self-hosted runners for the guard-critical path.
5. Views: CUT — visual flow editor and self-modifying flows via PR are scope creep. Ship a read-only console and CLI chat first; defer editor until flows are stable and human-reviewed.

MISSING:
1. Persistent agent memory/state across runs (not tied to queue items).
2. Fine-grained tool sandboxing (network, FS, syscalls) — Actions is too coarse.
3. Decision-level observability: trace, replay, debug a single agent step.
4. Agent identity/reputation — keys ≠ trust; need scoring for autonomous delegation.
5. Disaster recovery: spine corruption, broker compromise, key rotation drills.

FAILURE MODES:
1. Cloudflare free tier exhaustion — 100k req/day and 30 s CPU/request evaporate under event volume + broker hops + agent loops. Warning: Durable Object CPU-time errors > 1%.
2. GitHub Actions flakiness cascades — missed dispatches, lost events, queue stalls. Warning: "keeper" job restart rate doubles.
3. Key broker becomes latency bottleneck and blast radius — all 27 keys behind one service. Warning: broker p99 latency > 5 s.

SECURITY: Largest risk — broker centralizes all 27 API keys; compromise = total exfiltration. Cheapest control: move keys to HashiCorp Vault (free tier) or AWS Secrets Manager; broker mints short-lived tokens only, never sees raw keys.

ORDER: Wrong. 1. Replay (keep). 2. Broker + one executor end-to-end (proves routing, guards, secrets). 3. Spine under real load (measures free-tier burn). 4. Two-agent thread (needs broker + spine). 5. Views last.

CHEAPEST FALSIFYING TEST: Run 100 real queue items through spine→broker→Actions executor on Cloudflare free tier. Measure total requests, CPU ms, p99 latency, failure rate. If >50% free quota consumed or p99 > 30 s, the architecture is wrong.
