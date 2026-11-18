<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=63.68s -->

## 1.  Claims vs. the **EVIDENCE** section  

| # | README line / snippet (≈) | What the evidence **does NOT** support / over‑states | Severity | Recommended fix |
|---|---------------------------|------------------------------------------------------|----------|-----------------|
| 1 | “**The first architecture … is about 2–3.4× cheaper per image than the best small MLP at scale**” | The scale‑run shows **lgn_blockread ≈ 31 ns/img vs. mlp16 ≈ 104 ns/img** → **≈ 3.3×** cheaper (upper bound OK).  The *lower* bound of **2×** is not met: the plain‑lgn_d cost is **≈ 1.9×** cheaper (54.8 ns vs. 103.5 ns).  The claim therefore over‑states the minimum speed‑up. | **Major** – misleads readers about the guaranteed advantage. | Change to “up to 3.3× cheaper per image (≈ 1.9× cheaper for the un‑blocked version)”. |
| 2 | “**The measuring and recording machinery works and has been reviewed.**” | Evidence only shows that the *ledger* and *probe* have been anchored and that a few external reviewers looked at individual pieces (e.g. gpt‑oss‑120b review of `scale.py`).  No independent review of the *entire* measuring‑and‑recording pipeline is documented. | **Major** – a blanket “reviewed” statement is unsupported. | Replace with “The core components (ledger, probe, counter) have been independently inspected; full‑system review is pending.” |
| 3 | “**Machine probe, then deterministic counter, done and reviewed**” (Progress table, row *by Nov 5*) | No evidence in the supplied list confirms that the *counter* has been reviewed, only that CI passed (21/21).  The “reviewed” part is unsubstantiated. | **Major** | Delete “reviewed” or cite a concrete review (e.g. a pull‑request comment or external audit). |
| 4 | “**[`probe/`](probe/) machine probe – done, **7 independent reviews**” (Status table) | The evidence mentions “reviewed” but never quantifies *seven* independent reviews.  The number cannot be verified. | **Major** | Either provide a citation showing the seven reviews or change to “reviewed by multiple parties (≥ 2).” |
| 5 | “**[`router/`](router/) … benches 2–5 complete**” (Status table) | Evidence only lists **bench‑2** (Nemotron 3 Ultra, 3/3) and **bench‑5** (NIM pool, 11/15).  Bench‑3 and bench‑4 are never mentioned, so saying “2–5 complete” is inaccurate. | **Major** | List only the benches that have evidence (2 & 5) or add the missing bench results before claiming completion. |
| 6 | “**High‑volume bursts – Cerebras, xAI (paid credit) – only behind a spend cap, not yet built**” (Who does what) | Evidence says the keys *hold paid credit* and *no spend cap exists yet*.  Claim that a spend‑cap *is* in place contradicts the evidence. | **Minor** | Re‑word to “…available via paid credit; a spend‑cap has not yet been implemented.” |
| 7 | “**Short checks – Groq gpt‑oss‑120b (free, **8,000 tokens/min**) – answered the probe exactly**” | Evidence gives the quota as *1,000 requests and 8,000 tokens **per window***, not per minute.  The rate‑per‑minute phrasing is unsupported. | **Minor** | Change to “8,000 tokens per quota window (≈ 8,000 tokens per period).” |
| 8 | “**One CI job writes the ledger**” (How it works, step 5) | No evidence shows a *single* CI job performing the write; only that “One CI job writes it” is asserted.  The ledger entries could be produced by multiple jobs. | **Minor** | Either cite the specific CI workflow file or replace with “A CI workflow writes the ledger.” |
| 9 | “**Every result is weighed against real hardware**” (What it is) | Evidence shows two meters (instruction count & native timer) and a few hardware‑level measurements, but not that *every* result (including all future agents) has been weighed yet. | **Minor** | Qualify: “Results that have been processed are weighed against real hardware.” |
|10| “**Deterministic in any VM**” (How it works, step 3) | No evidence proves that the instruction‑counter yields identical results across *all* VMs; only that it is deterministic on the CI runners used. | **Minor** | Qualify: “Deterministic on the supported CI VMs.” |
|11| “**Claude (scarce; used least)**” (Who does what) | No evidence is provided for any work performed by Claude, nor for its scarcity. | **Minor** | Either add a reference (e.g. a commit log) or drop the row. |
|12| “**`scale.py` adds the cost per image at large run sizes, because rankings flip with N**” (How it works) | Evidence confirms the flip at N = 956, but the statement implies that *all* future rankings will flip, which is not demonstrated. | **Minor** | Re‑phrase to “…which can cause rankings to change for some workloads, as observed at N = 956.” |

### Summary of claim‑verification problems
- **7 major issues** (over‑stated performance, un‑substantiated reviews, incomplete bench reporting, contradictory spend‑cap statement).  
- **5 minor issues** (rate wording, CI job description, blanket “real‑hardware” claim, deterministic‑VM claim, Claude entry).  

---

## 2.  Can a newcomer grasp the project in **≈ 2 minutes**?

| Problem | README line / area | Why it blocks rapid understanding | Severity | Fix |
|---------|-------------------|-----------------------------------|----------|-----|
| **A. No concise “elevator pitch”** | Opening paragraph & “What it is” | The first two paragraphs mix etymology, a vague “lab where agents build” description, and a list of abstract mechanisms.  A newcomer cannot tell *what* the software actually does (e.g. “run a classifier on MNIST”) without scrolling further. | **Critical** | Add a 2‑sentence summary: “exagium is a reproducible benchmarking platform that lets anyone submit a tiny neural‑network implementation (train + infer) and automatically measures its accuracy, instruction count and wall‑clock cost on real hardware.” |
| **B. Heavy reliance on internal branch names and files** | Throughout (e.g. `pilot/lgn-20261115`, `router-runs` branch) | Readers unfamiliar with the repo cannot locate the referenced material; the names give no context. | **Major** | Replace branch names with plain‑language descriptions or provide direct links (e.g. “see the *pilot‑lgn‑Nov15* branch for the logic‑gate network”). |
| **C. Unexplained jargon** | Terms: “meter”, “ledger”, “jail”, “pre‑registered plan”, “large‑N”, “block‑read”, “middle‑segment cheat”. | No definitions are given before they are used; a newcomer must hunt elsewhere to understand the core concepts. | **Major** | Add a short “Glossary” box early on, or inline parenthetical definitions. |
| **D. Over‑dense tables** | Progress, Best result, Who does what, Status of each part | Tables contain many abbreviations and cross‑references; without prior knowledge they look like data dumps rather than a narrative. | **Major** | Convert each table into a brief bullet list with one‑sentence explanations per row. |
| **E. Missing “How to run it yourself” summary** | Only a link to `VERIFY.md` appears at the very end. | A newcomer wants a quick “clone → run → see results” guide; having to open another file defeats the 2‑minute goal. | **Major** | Insert a 3‑step “Quick‑start” section before the detailed “How it works”. |
| **F. Inconsistent terminology** | “agents”, “free models”, “candidate”, “bench” used interchangeably. | Confuses whether the user is writing code, calling an LLM, or both. | **Minor** | Standardise terminology (e.g. “candidate = submitted model implementation”). |

**Overall:** A newcomer would need > 5 minutes and a deep dive to understand the purpose, workflow, and current results. The README fails the 2‑minute comprehension test.

---

## 3.  Confusing, missing or contradictory elements  

| # | Location | Issue | Severity | Fix |
|---|----------|-------|----------|-----|
| 1 | “**Two meters price every candidate**” (How it works) | It is unclear whether *price* means “measure cost” or “assign a monetary value”.  Later sections treat it as a performance metric. | **Major** | Replace “price” with “measure”. |
| 2 | “**The test set opens only for a plan registered in the ledger beforehand**” | No description of *how* a plan is registered or what “opens” means (does the test data become available?). | **Major** | Add a short paragraph describing the plan‑registration workflow. |
| 3 | “**`loop/scale.py` adds the cost per image at large run sizes, because rankings flip with N**” | The phrase “add the cost” is ambiguous – does the script *record* the cost, *compute* it, or *modify* the ledger? | **Minor** | Clarify: “`scale.py` records per‑image cost for large batch sizes.” |
| 4 | “**Free compute: GitHub Actions runners across 6–8 runner types**” vs. evidence “loop on 6 runner types, counter on 8”. | The numbers are swapped (loop runs on 6, counter on 8) but the sentence suggests *all* jobs use both 6–8 types. | **Minor** | Re‑write: “We use GitHub‑Actions runners of 6 distinct types for `loop` and 8 types for `counter`.” |
| 5 | “**`router/` … runs pre‑registered agent benches (the `router‑runs` branch holds the raw data).**” | No explanation of what an “agent bench” entails; is it a benchmark of an LLM, of a model, or of the router itself? | **Major** | Add a concise definition of “agent bench”. |
| 6 | “**`audit/2026-11-17/` on the pilot branch**” – the path suggests the audit lives on a *pilot* branch, not on `main`.  Yet the status table says “14 fixed on `main`”. | Potential confusion about where the authoritative audit resides. | **Minor** | Clarify that the audit was performed on the pilot branch and the fixes were merged to `main`. |
| 7 | “**Spend caps at the call site, before any paid pool takes real work**” (Next) | No concrete description of *where* the caps will be enforced; readers cannot verify the plan. | **Minor** | Provide a placeholder link to the future implementation or a brief design note. |
| 8 | “**`router/` … required pools probed daily**” | No evidence that *daily* probing occurs; the evidence only mentions that the pools have been probed (bench‑2, bench‑5). | **Minor** | Either supply evidence (CI logs) or change “daily” to “periodically”. |
| 9 | “**`probe/` – done, 7 independent reviews**” vs. evidence lacking the count. | Already noted in claim‑verification, but also creates a direct contradiction that can’t be resolved. | **Major** | Supply the missing citations or lower the claim. |
|10| “**`counter/` deterministic instruction and cache counts, cross‑host CI**” – no evidence that the counts are *cache* counts; only “instruction” counts are mentioned in evidence. | Potential over‑statement of what the counter measures. | **Minor** | Adjust wording to “instruction‑level counts (including simulated cache misses)”. |

---

## 4.  Grading & Publication Recommendation

| Category | Issues | Overall grade |
|----------|--------|---------------|
| **Accuracy of factual claims** | 7 major, 5 minor mismatches with the supplied evidence. | **Critical** |
| **Readability / newcomer onboarding** | No clear 2‑minute overview, heavy jargon, missing workflow description. | **Critical** |
| **Internal consistency** | Contradictory numbers (spend‑cap, bench completeness, review counts). | **Major** |
| **Overall** | The README contains several factual overstretches, unsubstantiated review claims, and a structure that prevents rapid comprehension. | **Rewrite required** |

### Recommendation
**✘ Publish after fixes** – the document must be rewritten to (1) align every quantitative claim with the evidence, (2) remove or qualify unsupported review statements, (3) add a concise high‑level summary and a quick‑start guide, (4) standardise terminology and resolve contradictions. Only after these changes should the README be considered for publication.
