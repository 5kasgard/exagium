<!-- cloudflare @cf/openai/gpt-oss-120b status=200 latency=13.31s -->

**P25 – Gnomon as the Forge’s knowledge store** – **DEFER** – need a pilot that actually loads the existing Forge artefacts into Gnomon and demonstrates ≥ 8/10 correct answers with zero false positives before committing any glue‑code effort.  

**P26 – Parts‑audit of Gnomon, Polymechania, Mycel and Axiom** – **KEEP** – a systematic “what to carry, what to retest, what to drop” is essential to prevent the project from drowning in machinery that never contributes empirical findings.  

**P27 – Continuous‑operation generator** – **DEFER** – the hypothesis that generated work will reliably cut idle time by ≥ 80 % and be accepted at parity with human‑proposed items is untested; defer until a controlled 7‑day trial shows the required metrics.  

---

### Which item most likely exists because the idea was *pleasing* rather than demanded by data?  

**P27** – the “always‑on self‑generating work” stream reads more like an attractive narrative about autonomous research than a response to any observed shortage of queued work.  

---

### Single biggest risk of running the system 24/7 unattended  

**Uncontrolled resource consumption** – a feedback loop in the generator or a malfunctioning guard could cause the system to launch ever‑more runs, exhausting the free‑compute budget (or incurring hidden costs) and silently corrupting the ledger.  

### Cheapest guard against that risk  

**Enforce a hard external budget cap** (e.g., a CI‑runner or cloud‑provider quota that automatically aborts any job once the daily free‑compute allowance is reached) together with a watchdog that logs the stop event. This costs essentially nothing beyond the existing quota mechanism and instantly halts runaway execution.
