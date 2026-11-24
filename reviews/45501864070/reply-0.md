<!-- groq openai/gpt-oss-120b status=200 latency=3.24s -->

**VERDICT per layer (1‑5)**  
1. **Event spine – CHANGE** – committing every event to Git creates massive churn, hits rate limits and makes rebases a nightmare; a lightweight DB (e.g., SQLite on GitHub Pages or a cheap external KV store) is far more sane.  
2. **Threads & channels – CUT** – building a custom A2A protocol on top of file commits is dead weight; GitHub issues/PR comments already give durable threads and are far more reliable.  
3. **Key broker – CHANGE** – keep the broker off‑GitHub (e.g., Cloudflare Durable Objects) and enforce per‑call tokenization; otherwise you re‑introduce the exact secret‑leakage you’re trying to avoid.  
4. **Executors – CHANGE** – GitHub Actions are flaky (hourly keeper already mis‑fires) and have 6 h runtime caps; a self‑hosted runner or cheap external compute pool is required for true “always‑on”.  
5. **Views – CUT** – a console + file‑based visual flow editor will drown the repo in PR noise; you need a dedicated UI backed by the event store, not a PR‑centric editor.

---

**MISSING (most important first)**  
1. Real‑time messaging (WebSocket/queue) for true agent‑to‑agent dialogue.  
2. Transactional consistency across the four branches (atomic commits).  
3. Observability/metrics dashboard (latency, budget use, queue health).  
4. Idempotent replay & failure‑recovery semantics.  
5. Fine‑grained ACLs for each model/key to prevent accidental cross‑use.

---

**FAILURE MODES**  
1. **Duplicate claim race** – multiple “claimed” events for the same queue item appear. *Early sign*: spike in “duplicate claim” entries in the log.  
2. **Secret leakage via PR merges** – CI guard logs show a secret in a PR diff. *Early sign*: any “secret‑exposed” warning from the guard.  
3. **Durable Objects quota exhaustion** – broker returns 429/limit errors. *Early sign*: sudden drop in successful broker calls per minute.

---

**SECURITY**  
*Largest new risk*: the spine logs every broker request, potentially recording API keys or token identifiers in plaintext.  
*Cheapest control*: hash/encrypt any key‑related fields before committing; store raw keys only in sealed Cloudflare secrets accessible to the broker runtime.

---

**ORDER**  
The proposed test order is sub‑optimal. Revised sequence:  
1. **Key broker prototype** (validate secret handling & quota).  
2. **Minimal spine on free tier** (measure request cost).  
3. **Two‑agent thread interaction** (prove A2A works).  
4. **Replay of today’s queue** (full‑system sanity check).  

---

**CHEAPEST FALSIFYING TEST**  
Create a single dummy event file in the spine, push it, and let the existing GitHub Action worker process it. If the worker fails to locate or parse the event because of Git rate‑limit, file‑path mismatch, or missing branch, the whole append‑only‑via‑Git design is invalid. This costs nothing but immediately disproves the core premise.
