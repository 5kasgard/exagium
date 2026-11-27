# Queue, parked jobs and job-type assignment: research brief

Author: MiMo-V2.6 Pro (xiaomi/mimo-v2.6-pro) in LobeHub (Forge Researcher)
Date: 2026-11-27

Scope: Three interlocking mechanisms in the Forge queue (audit/2026-11-17/queue.json, orchestrated by worker/queue_ops.py, worker/transitions.py, worker/classify.py, worker/split.py, worker/generate.py, worker/backlog.py):
(1) how work becomes visible and claimable ("the queue"), (2) how items that cannot proceed now are stored and later revived ("parked jobs"), (3) how items are given a work type so a worker knows what job to run ("job type assigning"). This brief reports what the repo records say, what external sources say, and where they agree or conflict. Recommendations are inference from both, not measured results.

---

## 1. What the repo records actually show (measured / recorded)

### 1.1 Queue capacity today

From Exagium.queue_summary (2026-11-27T~05:00Z snapshot):
- parked: 97 items
- done: 212
- open: 81
- waiting on the owner: 3
- escalated: 3
- "Can run now": 60

The queue_summary tool itself carries the note: "Items a free worker can claim need a job type (targets/research/code); see queue item P38."

### 1.2 Job types in code

worker/queue_ops.py (queue P8 docstring and code):
> ready: status "open", every blocker done, the item carries a job the worker can do, and no live claim touches any of its "touches". Job types, by what the item carries: "targets" (exact files and line ranges a script can check) -> a doc job (worker/doc_job.py); "research" (a question and search queries) -> a research job that writes a literature card (worker/research_job.py, queue P7)

def job_type(i):
    return "doc" if i.get("targets") else "research" if i.get("research") else "code" if i.get("code") else None

So job_type() is derived structurally from which field the item carries (targets, research, code), not from a declared label. An item with none of these three fields is never ready(), no matter how simple or valuable it is.

generate.py (queue P27) already contains a partial fix for literature items only: attach_jobs() narrows touches to the item's own research/cards/<id>.md and writes research: {question, queries} fields onto the item itself -- so a whole class of items that used to be invisible to ready() now has a job type automatically.

### 1.3 Work types (richer, newer, not yet wired to job_type)

worker/classify.py (owner 2026-11-21: "three job types cannot hold the work") maintains an open list of 13 work types: code_change, doc_change, literature, experiment, decision, build, data, owner_action, sandbox_or_workflow, review, already_done, held, analysis. Each item gets work_types {type: p} (a probability distribution, kept only for p >= 0.2) and work_types_at; nothing else about the item changes. Items are re-judged only when their title or next step changed (work_types_key). An item with no type at p >= 0.5 is a misfit, listed in output "so the list can grow."

This work_types map is currently consumed by worker/split.py (splittable() reads it) to decide what to break into worker-sized sub-items, not by job_type()/ready(). So there are effectively two job typologies running side-by-side.

### 1.4 Parked: what "parked" actually means here

worker/transitions.py (Q59/Q60):
- escalated/resplit -> parked, review/resplit -> parked -- "a code item that failed its retry is parked for the splitter to cut smaller, not sent to the owner."
- proposed/reject (generator/splitter gate rejects a generated item) -> parked.
- Owner's decide_done/decide_open/park/decide_blocked/decide_scheduled/decide_owner can move any item to any of those states.

So "parked" is overloaded: at least four semantically distinct things:
1. resplit pending (item is real and should come back after splitters/smaller pieces exist)
2. gate-rejected (a proposed item was judged not worth doing -> semantically "closed/declined")
3. owner-parked (deliberate hold)
4. dead-end parked (research found nothing that survived narrowing)

worker/backlog.py already partially encodes this distinction in its lane logic ("resplit: parent <p.id> is <status>" vs "resplit: parent finished" vs "parked with no recorded reason" -> ORPHAN). This is the clearest existing signal that "parked" needs sub-typing -- the code already reaches inside parked to recover intent from the last run's result string.

backlog.py also already defines STALE (age-days limit per lane, explicitly labeled "guesses... until a week of lane ages is measured") and ORPHAN -- both exit code 1.

### 1.5 The measured failures these mechanisms were designed to fix (from audit/2026-11-17/QUEUE.md)

- Nov 24: "Why work piles up, measured: since Nov 20 the queue took in 70 items and finished 32 (22 of them research cards). None of the 77 open items could be picked up by a free worker: none carried a job type (targets, research or code), so ready() never offered them, and of 56 sub-items the splitter cut from them, 4 were done and 46 parked."
- Nov 24, item Q60: "Explicit item state machine: passes end, checks go to review, failed checks escalate one tier at a time."
- Nov 25 (handoff): "Worker idle. 0 of 75 open items are claimable, because none has a job type. The splitter last ran on new items on 21 Nov, and 52 of its 72 pieces are parked."
- P38's own next field (measured 2026-11-25T08:40Z, chat-claude): ready() was 0 of 76 open. "The worker is not the bottleneck; its inputs are." Direct probe (Q73, Q77): an item given a research job by hand was claimed and landed in ~3 minutes. Recommended smallest-change ordering: (a) queue learns of merges/closures by itself, (b) splitter sub-items carry a known-answer test written before the code, (c) widen auto-merge beyond experiments/.
- R67 (research card already in repo): "In job-queue schedulers, does priority aging or re-evaluating parked tasks' readiness predicates restore dispatch when no tasks are ready but a backlog exists?" Its single surviving measured claim: modified multilevel feedback queue with aging reduces waiting time for short jobs across priorities. Open questions: whether re-evaluating parked tasks' readiness predicates specifically restores dispatch, and which dispatch-rate metrics isolate that effect.

### 1.6 The backlog.py lane taxonomy (an existing draft answer to "which lanes")

Lanes, each naming the job that consumes it: worker (ready now), splitter, gate, review, in progress, scheduled, blocked, claude, owner-only, parked. This is already a job-type assignment schema expressed as dispatch destinations. So the repo effectively has three parallel typologies: job_type() (structural, gates claims), work_types (semantic distribution, feeds splitter), backlog.lanes (dispatch destination). They are not reconciled anywhere -- this is the single clearest design gap surfaced by the code itself.

---

## 2. What external sources say

### 2.1 Dispatch and job classification in production schedulers

Temporal - Task Routing (docs.temporal.io/task-routing):
> Task Routing is when a Task Queue is paired with one or more Workers... Multiple priorities: If your use case involves more than one priority, you can create one Task Queue per priority, with a Worker pool per priority. Alternatively, you can use Task Queue Priority, which lets you assign priority levels to Tasks within a single Task Queue.

This directly parallels the Forge's situation: either grow the number of queues/lanes (backlog.py approach), or keep one queue but add a typing annotation consumed at claim time (job_type()/work_types approach). Temporal's doc explicitly notes "Workers with different capabilities" (GPU vs non-GPU) get their own queue -- a worker-capability-to-job-type match, exactly what worker/queue_ops.py already does structurally via job_type().

HTCondor (htcondor.org/htcondor/overview) uses ClassAds -- declared request ads matched against machine offer ads:
> HTCondor provides powerful resource management by match-making resource owners with resource consumers... a user specifies a resource request ad when submitting a job... HTCondor acts as a broker by matching and ranking resource offer ads with resource request ads.

This is the "declare requirements declaratively, match at claim time" pattern -- matches what ready() does with touches + job_type(), but with a richer, bidirectional declaration (the Forge's router/workers/ "worker cards" idea (P20a) is the same concept).

GitHub's agentic "gh-aw" work queue spec (github.github.io/gh-aw/specs/work-queue-specification/):
> ready tasks are assigned in the order the queue accepted them... The queue looks past tasks already assigned, waiting on prerequisites, or waiting for a retry, and can assign unrelated ready work when capacity permits.
> Fairness: a large backlog does not buy all the turns... Platform does not get fifty times the turns just because its backlog is fifty times larger.

This confirms: (1) separate "parked/waiting-on-prereq/waiting-for-retry" as distinct reasons to look past an item -- what backlog.py's "why" field does, and (2) an explicit fairness rule to prevent one class of work (esp. machine-generated items) from starving human-priority work. Note "waiting for a retry" is called out as distinct from "waiting on prerequisites" -- the Forge uses one parked label for both-ish cases.

### 2.2 Backoff, deferral, and expired/parked work

Temporal Retry Policies (docs.temporal.io/docs/temporal-explained/timeouts-and-retries):
> Initial Interval multiplied by the Backoff Coefficient raised to the power of the number of retries. The Maximum Interval [caps it]... Use the Maximum Interval attribute to prevent the coefficient from increasing the retry interval too much.

Temporal (error-handling patterns, "Resumable Activity"):
> Parks the Workflow after retries are exhausted and waits for a human to correct the data or approve continuing, then resumes from where it left off.

This is explicit, designed "parked" semantics: not a passive "nothing to do" state but a deliberate wait-with-a-known-trigger-to-resume state. This is exactly the distinction backlog.py is reaching for when it flags orphaned "parked with no recorded reason."

Real-world agent-system queue design (negiadventures.github.io/blog/backpressure-concurrency-ai-agents) enumerates the exact failure modes this repo has already hit:
> * Stale work: queued jobs outlive the context or approval they depended on.
> * Jobs carry TTLs so stale work expires instead of executing late.
> * Priority queues help urgent work, but only if low-priority work can actually be dropped.

"Stale work: queued jobs outlive the context or approval they depended on" is a precise description of the Forge's Nov-24/Nov-25 situation.

### 2.3 Job-type classification for multi-worker agent systems

Content-based routing pattern (github.com/roanbrasil/agents-integration-patterns/.../content-based-router.md):
> A router agent examines the content, metadata, or intent of each incoming task and forwards it to the appropriate specialist. The router may use: LLM-based intent classification, rule-based matching, or embedding similarity.
> F1 Latency - classification adds a router step before the specialist call; usually worth it when specialist quality gain is large. F10 Adaptability - new specializations can be added by registering new routes without modifying existing agents.

This is a direct match for worker/classify.py: it is an intent classifier with an open type list designed to grow ("a type is added when items fit none of them"), which is exactly the F10 Adaptability benefit the pattern names.

Multi-agent dynamic task graph (DynTaskMAS, ICAPS 2025) (ojs.aaai.org/ICAPS/article/download/36130/38284/40203):
> function DECOMPOSETASK(t): if ISATOMICTASK(t) then return t... for each subtask s of t: S <- S u DECOMPOSETASK(s) [recursive]
> function CALCPRIORITY(v, G): if Succ(v) = 0 then return C(v) else return C(v) / max_{u in Succ(v)}(W(v,u)+CALCPRIORITY(u,G))

This is the closest academic analog to the Forge's splitter (worker/split.py): recursive decomposition into atomic, worker-sized sub-tasks with weights derived from downstream criticality, not just item order. Note the priority formula is recursive over the successor chain -- a task's priority depends on what (and how much) it unlocks. This is exactly the question queue item P22 is testing ("Does ordering by information per unit of cost change the queue's order against 'how many items it unlocks'?").

WorkflowOps (arXiv 2610.07860):
> our orchestration layers remain largely memoryless: each new task is decomposed, assigned, and executed from scratch with no benefit from prior successful executions... Given a natural language task T, an LLM decomposes it into K subtasks (3 <= K <= 7)

Note: MAX_SUB = 3 in the Forge's split.py sits at the low end of the published 3-7 range.

Queue design for AI workloads specifically (agentruntime.io/blog/queue-design-for-ai-workloads):
> The solution is queue segmentation: separate queues (and worker pools) for different workflow types, sized independently... For AI workloads, rate limits add a second back-pressure source that is invisible to the queue: the LLM API... This requires a token budget estimator and a throttling layer ahead of dispatch...

The Forge already has exactly this second back-pressure source (budget/refill caps per pool). The "token budget estimator and throttling layer ahead of dispatch" is un-built; it's a gap worth naming explicitly.

Priority queue mechanics & aging (sysdesignprep.com/concepts/priority-queues):
> Aging lets waiting jobs climb, capped below the top class... How do you prevent everyone from marking jobs as high priority? Assign priority by job type in the platform rather than letting callers choose.

This second point is a direct, published restatement of the same concern the Forge's own record raises ("this system's own gates must not be steered by model text"). Aging ("jobs climb in priority the longer they wait, capped below the top class") is exactly what R67's open question asks about for parked items.

Little's law framing for backlogs (service-management-metrics.github.io/en-us/topics/backlog-and-refinement-metrics):
> Measure age as well as size... AI tools can generate and split backlog items at volume, which inflates backlog size without adding value. Measure ready work depth and discard rate... Automation that closes stale items should itself be measured for wrongly closed items.

Directly relevant to worker/generate.py -- it caps MAX_PER_DAY = 10 for exactly this reason (its comment cites "research items land 18 of 25 (72%); splitter code items 6 of 51").

Stale item triage practice (GitLab Handbook, Linear docs, ONES guides):

GitLab Handbook (handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage):
> For issues that haven't been updated in the last 3 months the "Awaiting Feedback" label should be added... After 14 days... the issue should be closed... If they respond at any point in the future, the issue can be considered for reopening.

Linear docs (linear.app/docs/triage) add a distinct snooze action: "Snoozing will hide the issue from the triage queue to return at a time of your choosing, or when there's new activity on that issue: whichever comes first."

Both give exact precedent for the Forge's overloaded parked: the distinction between "snoozed (auto-returns at an event or date)" and "closed (needs a fresh positive action to reopen)" is precisely what the current single label does not capture.

ONES guidance (ones.com/blog/best-practices-for-setting-review-triggers-and-cadence-for-stale-product-backlog-items) cautions explicitly against age-only automated closure:
> Inactivity [is] a review trigger, not a verdict... Do not use "deferred" as a permanent parking label. A deferral needs a reason and a trigger for reconsideration.

This is a strong external endorsement of the design instinct already visible in worker/backlog.py and a caution against the naive fix of "auto-unpark anything older than N days."

---

## 3. Synthesis: how these pieces fit together

| Forge mechanism | What's already built | Closest published pattern | Known gap |
|---|---|---|---|
| Queue (claimability) | ready() in queue_ops.py (status + blockers done + job_type + no touch overlap); CAS-via-git; leases with expiry | Temporal Task Queues / HTCondor ClassAds matchmaking | job_type() only recognizes 3 structural fields (targets/research/code); anything else is invisible -- the Nov-24 measured cause of "0 of 77 claimable" |
| Parked jobs | transitions.py maps escalated/resplit, review/resplit, proposed/reject, and owner park all to one parked state; generate.py regates old proposed items after 7 days; split.py marks split_tried and retries after 7 days | Temporal's "Resumable Activity" + Linear's "Snooze" + GitLab's close/reopen; multilevel feedback queue aging (R67) | parked is overloaded (resplit-pending vs gate-rejected vs owner-hold vs dead-end); backlog.py already tries to disambiguate by parsing last-run's result string -- recovery logic that deserves to be structural |
| Job-type assignment | job_type() (3 structural values); work_types (13-type probability distribution from classify.py, open-ended with "misfit" reporting); backlog.py lanes (10 dispatch destinations) | Intent classification (content-based router pattern) with an extensible type registry | Three parallel typologies exist (structural job_type, semantic work_types, dispatch backlog.lanes) and are not reconciled -- an item can have rich work_types but still job_type() == None (invisible to workers) |

Two strongest external confirmations of the Forge's current design instincts:
1. The known-answer-control pattern used throughout classify.py, generate.py, split.py is the same discipline external sources call for explicitly ("Automation that closes stale items should itself be measured for wrongly closed items").
2. backlog.py's lanes + age + orphan/stale detection is a textbook implementation of "measure age as well as size... flag items that cross an age threshold for a human decision, not for automatic deletion."

Two strongest external warnings worth carrying forward:
1. Don't make parked a permanent state with no revisit trigger -- "Do not use 'deferred' as a permanent parking label" (ONES). Today, parked from proposed/reject has no automatic trigger to return (it is effectively closed), while parked from resplit does revisit via splitter's split_tried retry window.
2. Priority/aging must be capped, and assigned by the system, not by whichever caller inflates the claim ("Assign priority by job type in the platform rather than letting callers choose") -- directly applicable given items are sometimes generated machine-to-machine (generate.py, split.py) and their value field is written by a model at creation time.

---

## 4. Concrete recommendations (inference, not measured -- to be validated by the queue's own known-answer discipline)

1. Unify job-type assignment around one canonical field, sourced from work_types when present (falling back to structural inference). Concretely: add a derived mapping in queue_ops.job_type() that consults max(work_types, key=...) when targets/research/code are all absent, restricted to types with an executor (doc_change -> doc job, literature -> research job, code_change -> code job, experiment -> experiment dispatch). This directly fixes the measured "0 of 77 claimable" failure using a typology (work_types) that is already computed for many items.

2. Split parked into at least three explicit sub-states in transitions.py: parked-resplit (returns via splitter's split_tried retry, already automatic), parked-rejected (a deliberate close; requires owner action or new evidence to reopen -- GitLab's close/reopen model), parked-held (owner-specified wait, only the owner can move it). Record the sub-state structurally (a field), not by parsing the last run's free-text result (what backlog.py already does as a workaround).

3. Add aging/priority boost for stale-but-valuable parked items, capped below fresh high-value work: when backlog.py flags an item stale, optionally re-prioritize it (an age-based "climb"), matching the multilevel-feedback-queue-with-aging result already cited in R67 (its one surviving measured claim), but guard against auto-revival of gate-rejected items (recommendation 2 must land first).

4. Reconcile job_type(), work_types, and backlog.lanes into a single documented schema -- today three code paths independently re-derive "what kind of work is this" (claim gate, splitter eligibility, dispatch lane) and can disagree (an item can be splittable() by work_types but job_type() == None, so it goes to the splitter but never to the worker directly).

5. Separate the LLM budget back-pressure from item-level re-checking: a pooled "token/cost budget estimator" (P20a's worker cards are the building block) should gate dispatch before a claim, not only fail after quota exhaustion mid-run.

---

## 5. What I could not verify / conflicting or unresolved

- research/cards/R67.md is explicitly not judged by Claude ("Sources found by API scripts; claims read by @cf/openai/gpt-oss-120b... Not yet judged by Claude") -- its single surviving measured claim (aging helps short jobs) has not been through the tier the Forge normally requires. I report it as recorded precedent, not verified truth.
- backlog.py STALE/ORPHAN LIMIT_DAYS values are explicitly labeled in-code "guesses... until a week of lane ages is measured" -- no measured baseline yet exists. Any aging threshold adopted from external sources (e.g. GitLab's 3-month) would be arbitrary for this queue's cadence; recommend measuring before setting.
- Two searches I attempted (Web Browsing native search) failed with "all configured providers returned errors" -- I used Keenable Search and Parallel Search instead to produce every external citation in section 2; those two engines cross-corroborate on Temporal/HTCondor/queue-design material but I have not independently verified every quoted page in full (only excerpts). Full-page verification of github.github.io/gh-aw/specs/work-queue-specification/ and agentruntime.io/blog/queue-design-for-ai-workloads was not done -- flagged as secondary/industry sources rather than peer-reviewed primary research.
- The mapping table in section 3 (three typologies not reconciled) is my inference from reading the code -- I did not run the code to demonstrate an actual item that is splittable() yet job_type() == None simultaneously.
- Better academic grounding for "parked jobs specifically" (as opposed to generic retries/aging) was thinner than for classification/routing: the strongest primary-source anchor found (R67, in-repo) asks exactly this question and its own notes say the question is still open empirically. So this specific sub-question does not yet have a published answer I could find even after two independent search engines -- worth noting as a genuine open research question the Forge is well-positioned to answer with its own backlog.py traces once a week of lane-age data exists.
- Papers caution that routing cycles across workers can destabilize queues even at low load ("Design principle: avoid routing cycles across servers"). The Forge's blocked_by graph is a DAG by construction, but worker-to-pool assignment across time could form cycles -- I have not checked for such cycles in the live queue; flagging as an untested risk, not a claim.

---

## References (all URLs seen via Keenable Search / Parallel Search, 2026-11-27)

- https://docs.temporal.io/task-routing
- https://docs.temporal.io/task-queue
- https://docs.temporal.io/docs/temporal-explained/timeouts-and-retries
- https://docs.temporal.io/design-patterns/error-handling-patterns
- https://htcondor.org/htcondor/overview
- https://developer.hashicorp.com/nomad/docs/concepts/scheduling/batch-job-queue
- https://durable-workflow.com/docs/polyglot/task-matching-dispatch
- https://cs.brown.edu/people/jrasley/papers/eurosys16-yaq-final.pdf
- https://growasengineers.com/system-design/design-a-scheduling-job-queue/guide
- https://github.github.io/gh-aw/specs/work-queue-specification/
- https://sysdesignprep.com/concepts/priority-queues
- https://agentruntime.io/blog/queue-design-for-ai-workloads
- https://negiadventures.github.io/blog/backpressure-concurrency-ai-agents
- https://docs.celeryq.dev/en/stable/userguide/routing.html
- https://github.com/roanbrasil/agents-integration-patterns/blob/main/patterns/routing/content-based-router.md
- https://builder.aws.com/content/3AtvWf5Z0DfZLINb0E4iVFHWzeG/building-multi-agent-applications-orchestration-patterns
- https://ojs.aaai.org/index.php/ICAPS/article/download/36130/38284/40203 (DynTaskMAS)
- https://aclanthology.org/2026.findings-acl.1888.pdf (QUARTZ)
- https://arxiv.org/html/2504.07347v3 (Throughput-Optimal Scheduling for LLM Inference and AI Agents)
- https://handbook.gitlab.com/handbook/product-development/how-we-work/issue-triage
- https://linear.app/docs/triage
- https://ones.com/blog/best-practices-for-setting-review-triggers-and-cadence-for-stale-product-backlog-items
- https://service-management-metrics.github.io/en-us/topics/backlog-and-refinement-metrics
- https://szaher.github.io/agentic/lesson/10/05-task-decomposition
- https://open-multi-agent.com/reference/task-scheduling

In-repo references (read directly, cited by path):
- audit/fabric/README.md, audit/notes/2026-11-25/HANDOFF-chat-2026-11-25.md, audit/2026-11-17/QUEUE.md
- worker/queue_ops.py, worker/transitions.py, worker/classify.py, worker/split.py, worker/generate.py, worker/backlog.py
- research/cards/R67.md (via GitHub tool, path research/cards/R67.md in 5kasgard/exagium)
- Queue items: P38 (backlog does not reach free workers), Q60 (state machine), K58 (routing by pool room), P20a/P20e (worker cards, spread work), P22 (ordering test), P24 (unified system model)

---

## Suggested next step for the Forge's own record

If this brief is useful, it should itself be filed as a queue item (a doc/analysis kind, one item, no code risk) with touches = a single notes file path, so the free worker or a Claude shift can consume it like any other work -- consistent with the repo's own "every record goes through the queue" rule. It is filed here as issue #254 alongside this PR, both cross-referenced, for the queue's normal intake flow (decide kind/type, queue.json entry) to process and split into worker-sized pieces.
