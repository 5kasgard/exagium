You are a skeptical outside reviewer of a one-person research project, "exagium": agents build and test small neural networks and other AI against real hardware, with pre-registered claims and an append-only ledger, on free compute only.

Facts: after about two weeks, results hold for one task (8x8 digits); the only transfer test to other tasks was negative; a general task contract held on 4 tasks. Most recent work is infrastructure: a free-model queue worker, queue tooling, review machinery. An internal review just named the top gap as "too few findings for the infrastructure built". In the same session, the 16 items below were added to a queue of about 75 open items. The project's next result-producing steps are a mixed task suite (P3), one jail fix (Q56b), and a free proposer that writes candidate models around the clock (P5).

### P20: Worker-aware orchestration: the system knows what each worker (model x pool x harness x tools x budget) can do and what each task demands, and puts work where it will succeed most cheaply
blocked_by ['P8r']
no point giving models tasks past their context and reasoning ability, and don't be limited by what the  Claude's design (hypothesis): (1) a worker is model x pool x harness x tools x budget (same weights on Groq and Cloudflare may differ; OpenHands+Nemotron is not Nemotron); (2) worker cards, declared (context, output, rate, cost, modality, tools, reasoning; each with a source) and measured (success, empty-answer and refusal rates, latency, cost by task type and size; effective context; reasoning budget needed; injection susceptibility from canaries; run-to-run variance; calibration), with intervals, priors from public benchmarks (research worker), updated by our data; (3) task demand estimated before assignment (size, reasoning depth, tools, checkability, risk, splittability), with item response theory fitting task difficulty and worker ability together, the same method P6 uses for AI parts, so the Forge studies its own workers with its own engine; (4) routing to the cheapest worker predicted to succeed, cascades cheap -> strong -> Claude -> owner on a failed check, a little exploration; (5) shaping work to fit: split, retrieve instead of stuffing context, handoffs between steps, reasoning budgets from measured need; (6) capacity planning across pools and hours from remaining daily and per-minute allowances (the OmniRoute idea, data-run); (7) drift probes per worker weekly; (

### P20a: Worker cards: declared limits with sources, measured outcomes from the logs, intervals; pre-call size check
blocked_by ['P8r', 'R9']
Cards per (pool, model[, harness]) from provider docs (sourced) and from ledger/calls.jsonl, Q22 router logs and graded release-test outcomes; a pre-call check refuses or splits prompts past the measured safe size. Then the offline replay test in P20.

### P20b: Task demand and item response theory over worker tasks (difficulty and worker ability fitted together)
blocked_by ['P20a']
Fit a 2-parameter IRT model on graded worker outcomes (release tests, reviews, benches); test whether it predicts held-out outcomes better than per-worker averages. Shares its method with P6.

### P20c: Routing by cards with cascades (cheap -> strong -> Claude -> owner) and a little exploration
blocked_by ['P20b', 'P7b', 'P7c', 'P7d']
Replace the worker's fixed drafter/checker choice with a router call; measure cost per accepted item against fixed assignment live, both arms interleaved. Blocked by P7b/P7c/P7d because all touch worker/research_job.py; if P7b is still waiting on the owner's YouTube key when this item is otherwise ready, move P7b behind this item instead (blocker reversed, recorded here).

### P20d: Shape work to fit the worker: split oversized tasks, retrieval instead of stuffed context, step handoffs, reasoning budgets from need
blocked_by ['P20a', 'P7b', 'P7c', 'P7d']
Only when P20a's data shows failures rising with size or depth for the workers in use; then measure acceptance before and after on the same items. Blocked by P7b/P7c/P7d because all touch worker/research_job.py; if P7b is still waiting on the owner's YouTube key when this item is otherwise ready, move P7b behind this item instead (blocker reversed, recorded here).

### P20e: Capacity planning across pools and hours (remaining daily and per-minute allowances)
blocked_by ['P20a']
Only when the worker's queue regularly waits on one pool's cap while others have room (digest measures); then schedule by headroom.

### P20f: Drift probes: small weekly canaries per worker detect silent model changes
blocked_by ['P20a']
Weekly fixed probes per worker (a doc canary, an injection canary, a long-context item); flag a change beyond the interval of its history.

### P21: Graphs on demand: any question about the system becomes a graph, drawn from the record (typed nodes and links; views are queries, not fixed tools)
blocked_by []
dependency- or prerequisite-based work, a web where nodes, connections and flows are easy to see and manipulate. v0 built (audit/2026-11-17/web/build.py: reads queue.json only; 75 live items form one piece of 93 nodes, one of 12, and 24 small ones incl. 14 items with no link at all: check each for a missing link). Next: (1) link kinds in queue.json: gate (go/no-go; a no closes everything downstream), better-after, alongside blocked_by; render.py enforces them; today 3 'only if' gates live only in prose (D21, P20d, P20e) plus B10's P14->P15; (2) a cycle is resolved, not just refused: split off the smallest working version of one side; (3) each prerequisite states the smallest version needed and a cost limit, past which it goes to the owner; (4) a worker that hits a missing piece files it and waits on it; (5) a changed upstream result flags its dependents for recheck; (6) everything waiting on the owner is gathered into one request; (7) editing the web proposes a record change with confirmation, never edits the picture alone; (8) after P11, the web covers parts, tasks, runs, claims, sources, workers and pools. TEST before relying on it (honest-views rule 8): time to answer 'what does P11 unlock', 'what waits on the owner', 'what breaks if Q56b slips' in the web against the table, a separate agent timing both. Outside review 2026-11-19 (gpt-oss-120b, router-runs reviews/4447037763

### P22: Does ordering by information per unit of cost change the queue's order against 'how many items it unlocks'?
blocked_by ['P21']
Needs P21's gate links. Rule fixed before running (outside review finding 6: 'top 5 differ' alone is unsound): compute both orders over ready items; if Kendall tau >= 0.8 keep the current rule. If tau < 0.8, run the two orders in alternate weeks for 4 weeks and adopt the new key only if decisions per worker-hour (items closed or gated out) are higher with it by more than the week-to-week spread.

### P23: Claims at volume: a test-set budget per data set, false-discovery control, and a forecast on every plan
blocked_by []
Whole-system review 2026-11-19: pre-registering each plan works for a dozen claims, not for thousands of proposed candidates. Add to the runner's test gate: (1) each test split has a use count and budget; past it, the split refreshes or claims go to a fresh holdout (reusable-holdout method); (2) claims from one search are corrected for many comparisons (false-discovery control) before a verdict; (3) every plan states the claimant's probability of a pass before running, kept in the ledger, scored by P11 for calibration per source. Gates P5's first test-set claim, not P5 itself. Outside review 2026-11-19 (gpt-oss-120b, router-runs reviews/44470377631; llama-4-scout timed out): finding 3 accepted in part: the design must say how error is controlled across tasks and across iterations of one search, not only within one search; the method and its numbers are chosen in this item from the literature (research card first), not fixed here.

### P24: One model of the system: do P6, P11, P20, P21, P22 and P23 share one engine (a typed, time-varying graph plus fitted abilities plus allocation)?
blocked_by ['R10', 'R11', 'R12', 'R13', 'R14']
the whole thing is a static and dynamic heterogeneous system and much of the queue overlaps. Measured 2026-11-19 (TF-IDF over 49 live non-canary items): word overlap between substantive items is low (highest P20c-P20d 0.60; the 0.5-0.76 scores among R3-R8 come from shared boilerplate), so the overlap is conceptual, not in wording, and a word measure cannot see it. Hypothesis: compartments (P6), the store (P11), worker orchestration (P20), graphs on demand (P21), ordering (P22) and claims at volume (P23) are one engine seen from different sides: typed nodes and links over time; abilities fitted from outcomes (item response theory for parts, tasks and workers alike); allocation by value of information. Test after R10-R14: write the shared model in docs/system-model.md, map every live item to it, and count items whose design shrinks or merges (target: a separate agent confirms at least 3 merges or removed duplicate designs); if none, keep them separate.

### R10: Research card: typed, time-varying graphs of many kinds of thing (heterogeneous, dynamic) and how to model, store and query them with provenance
blocked_by ['P8r']
Worker job (research) after release test 2, owner 2026-11-19: many queue items overlap conceptually (graphs and links, agent hierarchies, ordering, evidence); research before building. Feeds P21, P11, P6 and P24.

### R11: Research card: escalation between models and to humans: cascades, routing and learning to defer, with measured accuracy and cost
blocked_by ['P8r']
Worker job (research) after release test 2, owner 2026-11-19: many queue items overlap conceptually (graphs and links, agent hierarchies, ordering, evidence); research before building. Feeds P20c, P20, P8 and P24.

### R12: Research card: when a model can replace a human reviewer: LLM judges, scalable oversight and weak-to-strong, with measured agreement and failure modes
blocked_by ['P8r']
Worker job (research) after release test 2, owner 2026-11-19: many queue items overlap conceptually (graphs and links, agent hierarchies, ordering, evidence); research before building. Feeds P20c, P8, P10 and P24.

### R13: Research card: ordering work with prerequisites and go/no-go gates under uncertainty: value of information and experiment prioritisation
blocked_by ['P8r']
Worker job (research) after release test 2, owner 2026-11-19: many queue items overlap conceptually (graphs and links, agent hierarchies, ordering, evidence); research before building. Feeds P22, P21, P5 and P24.

### R14: Research card: how automated labs and AI-scientist systems organise experiments, knowledge and agents
blocked_by ['P8r']
Worker job (research) after release test 2, owner 2026-11-19: many queue items overlap conceptually (graphs and links, agent hierarchies, ordering, evidence); research before building. Feeds P5, P11, P20, P9 and P24.

For each of the 16 items say exactly one of: KEEP (worth it now), DEFER (only after named results exist; say which), MERGE (into which item), CUT. Give one line of reason each. Then: which of these items most likely exist because an idea was pleasing to discuss rather than because data called for it? Then the single most important thing this project should do next. No praise.