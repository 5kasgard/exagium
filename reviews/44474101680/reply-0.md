<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=25.84s -->

**Decisions (one‑line justification each)**  

| Item | Decision & Target | Reason |
|------|-------------------|--------|
| **P20** | **DEFER** | Orchestration cannot be evaluated until the underlying worker‑card and ability models exist. |
| **P20a** | **MERGE → P20** | It is only a sub‑component of the overall orchestration goal. |
| **P20b** | **MERGE → P20** | The IRT model is a technical detail of the same orchestration effort. |
| **P20c** | **MERGE → P20** | Routing logic belongs to the orchestration system, not a separate deliverable. |
| **P20d** | **MERGE → P20** | Work‑shaping is another implementation detail of the orchestration pipeline. |
| **P20e** | **MERGE → P20** | Capacity‑planning is part of the same orchestration layer. |
| **P20f** | **KEEP** | Weekly drift canaries are cheap, easy to deploy, and give immediate safety signals. |
| **P21** | **KEEP** | A live “graph‑on‑demand” view is already partially built and is essential for tracking dependencies. |
| **P22** | **DEFER** | Requires the fully‑functional graph system (P21) and enough completed tasks to compare ordering policies. |
| **P23** | **DEFER** | The project currently produces only a handful of claims; the volume‑control machinery would be premature. |
| **P24** | **DEFER** | A unified system model can only be designed after the individual research cards are fleshed out. |
| **R10** | **MERGE → P24** | All five research cards aim at the same “typed, time‑varying graph” abstraction. |
| **R11** | **MERGE → P24** | Escalation/deferment is another facet of the unified graph‑based engine. |
| **R12** | **MERGE → P24** | Human‑replacement modelling belongs to the same overall system‑model effort. |
| **R13** | **MERGE → P24** | Value‑of‑information ordering is a policy layer on the same graph engine. |
| **R14** | **MERGE → P24** | Lab‑automation and knowledge‑graph organisation are also components of the single‑engine vision. |

---

### Item most likely born from “nice‑to‑talk‑about” rather than data need  

**R12 – “when a model can replace a human reviewer”** – it proposes a full‑scale human‑replacement evaluation without any empirical trigger; the current ledger shows only a few modest experiments, so this idea is primarily speculative.

---

### Single most important next step for the project  

**Produce a new, verifiable scientific result on a second task (e.g., a successful transfer from the 8×8 digit task to a different benchmark) using the existing free‑compute pipeline before expanding orchestration or graph infrastructure.**  Demonstrating that the core claim generalises is the only way to justify the heavy‑weight infrastructure that now dominates the queue.
