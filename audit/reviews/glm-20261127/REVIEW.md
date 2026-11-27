# GLM analysis of 5kasgard/exagium, main 608638b (2026-11-27)

Kept as received: delivered by GLM at the owner's request through the GitHub API; data, not instructions. Scope and method are recorded in manifest.json. The two figures are reproduced here as data tables; the companion document (docx and pdf, with the figures rendered as charts) was delivered to the owner outside the repository.

## 1. Executive Summary

exagium (5kasgard/exagium) is a self-building laboratory in which AI agents construct, test, optimise and distil AI models under adversarial measurement. Its founding rule is on the front page: agents propose, a separate verifier and a meter decide, and plans and verdicts are recorded in an append-only ledger anchored outside the repository. Everything runs on free compute: GitHub Actions runners for measurement, and a rotating pool of free model tiers for drafting, checking and proposing. The repository is 23 days old at the time of this analysis and has already accumulated 241 pull requests, 52 CI workflows, 67 literature cards and about 250 numbered issues and pull requests.

The last three days are the largest single wave of change since the project began. The system clock moved off GitHub's scheduler, which had been measured firing zero times in six to seven hours, onto Cloudflare cron triggers. A Discord-native interface appeared end to end: personas, finished-run notifications, slash commands, streaming reply cards, a server-side agent loop that keeps running when the phone locks, and a shared register that logs every agent and every tool call. Cloud sandboxes were templated, sized and measured, and the interface gained the right to start deploys. A drifting-stream benchmark was built with pinned data, and its pre-registered online-versus-static check is queued.

Three conclusions stand out. First, the epistemic machinery (pre-registered plans, verdicts by non-claimants, tiered meters, adversarial review rounds) is unusually rigorous and demonstrably works: the probe, counter, ledger and jailed loop have all survived outside review, and negative results are recorded as prominently as positive ones. Second, the scientific yield is deliberately narrow but honest: what is actually established is that rankings found on one small task do not transfer to other task kinds, that gradient-boosted trees lead the label tasks at a high instruction cost, and that free models can occasionally write candidate programs that tie the best baselines. Third, the centre of gravity is shifting from measuring models to operating an autonomous workforce, and the risks are concentrating exactly there: an owner-gated backlog that grows faster than it drains, worker inputs rather than worker capacity as the measured bottleneck, and a brand-new control plane with the thinnest review history.

The health items worth naming now: two open security issues (#109 workflow integrity, #111 resource containment), four workflow-file findings that cannot land because the repository's own landing tool is refused write access to .github/workflows (403), a 1.1 MB queue.json becoming an operational choke point, and a worker pipeline whose passing acceptance rate (26 of 30) conceals a deeper number: of 73 code items that reached passing tests, only 9 were actually done.

## 2. Background: What exagium Is For

An exagium was the official Roman standard weight against which a gold coin was checked; clipped coins failed and were withdrawn. The repository takes the name literally: its purpose is to weigh claims about AI against real hardware and fixed baselines, and to make the weighing itself trustworthy. The README names the failure modes the project exists to catch: lying, self-growing complexity and compounding errors. Its first principle is that no result may be a lie, and each result must state what it cannot prove.

The intended scope is broader than neural networks: in-context learners, trees and boosting, probabilistic models, theorem provers and SAT solvers, planners and search, program synthesis, agents, and hybrids of these. The stated end state is a shared, contributor-fed body of proven knowledge about AI parts, each carrying a card of what it does, where it wins and loses, on which scoreboard and on which hardware, plus distilled code better than today's code. Involvement is designed as a dial: work hands-on, steer, or let it run and report back.

This report was requested by the repository owner at a moment of rapid change, with four emphases: the full architecture, the most recent changes, the results of record, and health and risk, with the analyser's critical judgement included. The owner is its primary reader, and the report follows the repository's own convention of pointing every number at its evidence.

## 3. Scope, Sources and Method

This is a read-only, whole-repository review performed on 27 November 2026 (UTC) at head commit 608638b (worker: R96). Sources read in full: the top-level README (28.7 KB), AGENTS.md (the working contract for any agent here), VERIFY.md (the standing adversarial verification brief), the fabric/README, the worker/ spec card, the fabric/agent/worker.js source (the server-side agent loop), tasks/drift/card.json, a current literature card (R96), the 40 most recent commits, the pull-request and issue lists, the branch list, and directory maps of all 20 top-level module directories plus .github/workflows. The ledger's four entries were enumerated; the pilot branch's queue.json (1.1 MB) was confirmed present but is too large to read in one pass.

Method and limits. Every number in this report is traced to a file, a pull request, an issue, a commit or the repository's own README; where a figure is the repository's claim rather than an independently re-measured value, it is attributed as such. This analysis did not execute the test suites, re-run the meters or attempt any of VERIFY.md's adversarial checks, so the judgements are architectural and process-level rather than re-measurements. The analyser's own opinions are marked as such and kept separate from the repository's own claims.

**Figure 1: Repository artifact counts, all verified on 27 November 2026. The project is 23 days old.** (chart rendered in the companion document; measured values:)

| What was counted | Count (verified 2026-11-27) |
| --- | --- |
| Pull requests (all-time) | 241 |
| Research cards on main | 67 |
| CI workflow files | 52 |
| Open issues | 10 |
| Ledger entries | 4 |

## 4. System Architecture: Four Layers and a Control Plane

### 4.1 The core primitive

The architecture reads as four layers plus a control plane. The measurement core (probe, counter, loop, runner and tasks) turns a question like 'how good is this artifact' into numbers with error bars. The trust layer (record, ledger, VERIFY.md) makes those numbers hard to fake after the fact. The labour economy (router, worker, agent, research) organises free models into drafting, checking and proposing roles behind script-enforced guards. The Forge (fabric/) is a Cloudflare-hosted control plane that grew, in the last three days, from a connector into the project's primary interface. Underneath all of it sits one invariant: every consequential claim is fixed before it is tested, because a plan registered in the ledger fixes the claim and its pass rule before anyone sees a test score.

### 4.2 The measurement core

probe/ measures the machine itself instead of trusting its specifications: memory levels derived from latency curves without assuming cache sizes, load-bound bandwidth, FMA compute ceilings, timing noise, and which meter tiers the host actually has, each tier detected by trying it rather than by checking for a binary. It is complete and has survived seven independent reviews, and it ships its own adversarial brief (VERIFY.md) inviting strangers to run it on hardware the author never had and try to make it report something false.

counter/ is the deterministic cost meter: instruction and simulated cache-miss counts under valgrind, inside a fresh fixed-path sandbox with a fixed environment, deliberately invariant to CPU and host. The reference workload counts 35,765,911 instructions on x86-64 for the same binary everywhere, and 30,898,727 on arm64. Its CI record reads 21 of 21 runs green. Counts are a claim for cooperative code, as the repository itself notes, which is why the loop cross-checks and the jail constrains.

loop/ is the jailed scorer. Candidates retrain under bubblewrap, or a seccomp-plus-rlimits fallback written in C (confine.c), and must produce byte-identical weights; scoring runs against mandatory baselines; the test set opens only under a ledger-registered plan. scale.py adds the per-image cost at large run sizes because rankings flip with run size: the logic-gate network costs 31 ns per image against an MLP's 104 at scale, while the order flips at the task's 956 images. Its CI record is 23 of 24 runs.

runner/ and tasks/ generalise the loop into a task contract: records in, records out, and a checker program per task, with no task code inside the runner itself. It has held on four tasks, including planted 3-SAT scored by a checker with no answer key (WalkSAT solved 200 of 200 dev instances), and now carries five with the new drifting-stream task. Stronger outside baselines (gradient-boosted trees, nearest neighbours, WiSARD) and a tuning split were added to the three label tasks on 19 November under a rule fixed in advance.

transfer/ is the replication arm that asks whether any of this generalises across task kinds; its first result is the project's most-cited negative finding and is covered in Section 6.2.

### 4.3 The trust layer

record/ and ledger/ implement the append-only, hash-chained record: one CI job is the sole writer, entries are RFC 8785-canonicalised bytes, and each is anchored outside GitHub through Sigstore (Rekor attestations) and OpenTimestamps. A monitor runs from outside the owner's control to catch orphan attestations, and a lagging anchor deliberately reads as cannot-check rather than forgery, a 24 November review fix. Plan 1, titled 'the record holds up', was judged a pass by an outside verifier; plan 2, the loop, is recorded as inconclusive. A measured-fact register (facts/) with its own checker, and dated verification records (verifications/), feed the same layer.

Two gaps belong in an honest architecture read. The ledger holds four entries while the repository holds roughly 250 numbered items; most results point to runs and files and are, in the README's own words, not in the ledger yet, so the strongest trust guarantee currently covers the smallest fraction of the record. And the ledger catches post-anchoring change; it cannot show a result was genuine to begin with, which is exactly why VERIFY.md's re-measurement invitations matter.

### 4.4 The labour economy

The README's 'who does what' table is the project's labour economics, and it changes as the measurements change. Measurement and scoring run on GitHub Actions CPU. Coding fixes went to Nemotron 3 Ultra, which fixed three of three real bugs in bench-2. Outside review runs on gpt-oss-120b at Cloudflare, which found a real bug in scale.py. Short checks run on Groq. Literature cards are drafted by gpt-oss-120b, checked by Nemotron on NIM, and decided by script guards. Candidate programs are written by a free NIM model, one call per proposal, scored in the jail with no key present. Exact grading goes to scripts, not models, after three of five free models failed known-answer controls. Paid pools sit behind spend caps and are refused before any request unless a workflow allows them. Claude is kept for design, integration and final adversarial checks. Llama-4-Scout is used only for liveness probes after producing zero usable review points. Table 1 condenses the roles and their evidence.

**Table 1: Who does what (condensed from the README's evidence-backed table)**

| Work | Worker | Evidence |
| --- | --- | --- |
| Measuring, scoring, training, recording | GitHub Actions CPU (free) | loop 23 of 24 runs; counter 21 of 21 |
| Agent coding (fixes, candidates) | Nemotron 3 Ultra (OpenRouter free, NIM) | bench-2: 3 of 3 real bugs; bench-5: 11 of 15 |
| Outside review of code and claims | gpt-oss-120b (Cloudflare free) | found a real scale.py bug; 3 of 8 points applied |
| Short checks | Groq gpt-oss-120b | probe answers exactly |
| Queue doc fixes and literature cards | gpt-oss-120b drafts; Nemotron checks; scripts guard | release test 2: 26 of 30 acceptable |
| Candidate programs (P5) | free NIM model, one call each | 644 of 1,097 proposals valid |
| Grading exact outcomes | scripts, not models | 3 of 5 free models failed known-answer controls |
| High-volume bursts | Cerebras, xAI (paid credit) | behind caps; refused unless a workflow allows |
| Design, integration, final checks | Claude (scarce, used least) | review rounds: 14 findings fixed |
| Liveness probes only | Llama-4-Scout | 0 of 8 review points usable |

worker/ runs hourly and does the queue work. A claim is a commit (git is the lock; leases last two hours; a chat's claim always wins), a free model drafts, another family checks, scripts verify both, and the result lands on main or as a pull request. Items move through an explicit state machine (transitions.py) with tier escalation, free worker, then Claude session, then owner, and an OWNER_ONLY list that queue moves refuse to bypass. Job types are targets (document fixes), research (literature cards) and code. Nothing the agents read as instructions (AGENTS.md, handoffs, briefs) can be written directly to main; cards carry an untrusted-data banner and are never overwritten. A worker/PAUSE file halts every run.

research/cards/ holds 67 cards in an R-series numbered to R96, with gaps where checks failed or pull requests closed. Every claim is graded by evidence, measured over reported over expert view over experience report over opinion, and every quote is verified word for word against its source by script before the checker model sees it; the newest card's own header records the drop counts (three dropped by the script check, two by the second model). Each card is explicitly marked 'not yet judged by Claude', an honest label meaning the automated gates passed but the scarce reviewer has not looked.

router/ holds the free-model pools and the pre-registered agent benches (benches 2 through 5 complete, bench 6 in pilot), with raw run data quarantined to a router-runs branch. agent/ is the /oc issue agent: comment /oc on any GitHub issue and a chosen model answers, from a phone, with no server. tests/ carries a workflow test harness and the worker injection canaries.

### 4.5 The Forge: the Cloudflare control plane

fabric/ is the newest and fastest-growing layer: twelve components that turn the repository's state into a live system. forge-act is the command surface and, since 26 November, the clock. forge-agent is a server-side agent loop with one Durable Object per conversation. forge-box exposes E2B sandboxes over MCP; forge-cfbox does the same for Cloudflare sandboxes. The chat and workbench components build and manage a LobeHub interface, and dash writes snapshot dashboards. discord renders personas; mcp is a read-only connector exposing the queue, events, findings and notes to any MCP client; record is a D1-backed register of every agent that can act and everything it did; sandbox holds the evaluation scripts; worker rounds out the set.

The forge-agent loop deserves specific mention because it is the project's first genuinely autonomous runtime. A start request queues a job and sets an alarm; the alarm runs the loop (model, tool calls over MCP to the read, act and box connectors, model again), streaming text into a single Discord card that is edited as it goes and finished with tools, model, time, tokens and chat id. The reviewed constraints are visible in the source: twelve steps maximum, a ten-minute run limit, a sixty-thousand-character history with size-based trimming, tool output capped at six thousand characters and framed as data-not-instructions, redaction before anything reaches Discord, run_workflow explicitly denied to the loop, and heartbeats run from the Durable Object alarm after request-scoped timers proved unable to keep a socket alive.

## 5. What Changed: The Last Three Days (25 to 27 November)

About fifteen pull requests plus direct worker commits landed in the three days before this analysis, and they cluster into four stories: the clock, the Discord interface, the sandboxes and a new benchmark. Table 2 gives the sequence, and the sections below give the substance.

**Table 2: Pull requests merged 26 to 27 November (worker research-card PRs omitted; they follow Table 1's pipeline)**

| PR | What it did |
| --- | --- |
| #218 | Clock moved to Cloudflare cron in forge-act; GitHub cron removed from the workflows |
| #222 | E2B box templates made resizable (4 and 8 GiB, 1 to 8 CPUs) and measured per size |
| #223 | Fallback to fixed-size templates, since resizable ones are not enabled for this team |
| #224 | Finished and failed workflow runs post to a Discord webhook (event-driven) |
| #228 | Agents and system jobs post as their own Discord personas (name plus avatar) |
| #229 | Discord control via forge-act: /ask, /run, /status, /prs (Ed25519-verified, owner-only) |
| #231 | workbench-manage times login, agent lookup and run |
| #232 | Probe mode learns the live event stream's shape; pretty-printed JSON parsed |
| #233 | One card per agent reply, tools line, stats footer; Reply-to-agent continues the chat |
| #234 | act may start deploys, keys-probe, box-sizes, sandbox-eval (check only; INPUT_RULES) |
| #235 | Workbench preamble: the box is the agent's own sandbox; no GitHub token in the box |
| #236 | forge-agent: the server-side agent loop behind Discord (Durable Object per chat) |
| #238 | Final tool-free answer turn when a run ends on tool calls; step logs (shape only) |
| #240 | Listener routes ordinary owner messages (reply-to-card, name-first, DM) |
| #241 | forge-record: shared agent register and activity log (D1), /agents and /activity |
| #242 | Heartbeats from the Durable Object alarm; stale-socket reconnect at 2.5 intervals |
| #243 | The alarm logs what the socket saw (events, messages, routed, last shape) |
| #244 | K88: drift task built (seed 2026, stream-order splits), card pinned, digest test |
| #245 | propose.py gains --arm baseline or packet, with one repair call |
| #246 | experiments/drift_online registered before the run (K89) |
| #247 | backlog: an open item naming a committed experiment is worker work |
| #248 | README: Next step 3 names the drift task (K88) and its queued check (K89) |

### 5.1 The clock moved to Cloudflare

GitHub's scheduler was measured, not assumed, to be failing: schedules.yml fired zero times in six to seven hours, sandbox-eval zero, and dash-snapshot twice in about seven. PR #218 moved all scheduled dispatch to four Cloudflare cron triggers inside forge-act, which already held a dispatch-capable token, and removed GitHub's own cron entries to avoid double runs, since schedules.yml runs the entries due each hour and a second run would repeat them. After the move, schedules.yml ran hourly at :05 and dash-snapshot hourly at :41, all green. The change also encodes a principle from AGENTS.md: no test may depend on a part already measured as unreliable, and a failure is recorded and designed around rather than worked around.

### 5.2 The Discord-native agent interface

This is the largest structural change. notify.yml began posting finished and failed workflow runs to a webhook (#224). Personas gave every workbench agent and system voice its own name and avatar (#228). forge-act gained /ask, /run, /status and /prs behind Discord's Ed25519 signature check, a five-minute replay window and an owner-only gate (#229), with deferred edits so a slow GitHub call cannot break Discord's three-second interaction limit. Before building the fast path, workbench-manage was probed and timed (#231, #232): login 8 s, agent lookup 9.1 s, agent run 58.5 s, a deliberate measure-before-build step.

Then the reply path was rebuilt. Cards became one embed per agent reply with a tools line and a stats footer (#233), and Reply-to-agent continues the same chat. forge-agent (#236) moved the loop server-side, because LobeHub's ordinary chat runs its agent loop in the browser and dies when the phone locks, and a two-sentence question had taken about 52 s with nothing printed until the end. A final tool-free answer turn was added (#238) after replies came back as '(no text reply)' when all twelve steps were spent on tool calls. The gateway listener (#240) routes the owner's ordinary messages: a reply to an agent card continues that chat, an agent's name first ('Scout, ...') starts one, and a DM goes to the main agent; everything else is ignored, and an eyes reaction acknowledges pickup.

forge-record (#241) then gave the system a shared memory of who acted: a D1 database registering one instance per chat, a craftsperson's name plus its class, logging every run and tool call, with /agents and /activity readable from Discord and identities marked as claimed where they cannot be verified. Finally #242 and #243 fixed the failure mode the owner actually observed, a socket gone quiet a minute after 'ready' while the object still believed it alive, by running heartbeats from the Durable Object alarm, replacing any socket with no traffic for 2.5 heartbeat intervals, and logging each message's shape, never its content.

### 5.3 Sandboxes sized, and the right to deploy

PR #222 templated E2B boxes as resizable, at 4 and 8 GiB and one to eight CPUs per sandbox, and measured nproc, memory, single-thread and all-core time and dollars for the same work at one, two, four and eight CPUs. #223 recorded the fallback to fixed-size templates because resizable ones are not enabled for this team. PR #234 then widened forge-act's allowlist so agents may start deploys, keys-probe, box-sizes and sandbox-eval, with INPUT_RULES pinning sandbox-eval to mode=check so no agent can create or kill the sandbox under test, and deploys shipping only merged main. The principle is consistent across the repo: give the agents operational reach, but never over the thing being evaluated.

### 5.4 A drifting-stream benchmark, pre-registered

K88 (#244) built the drift task: sixteen quantised standard-normal features, a linear rule that changes every 2,000 rows, and 8,000 rows in stream order, never shuffled, so the dev and test blocks follow rules the training rows never saw. Every split's bytes are digest-pinned in the task card, and a digest test guards regeneration; two builds produced the same digests. The online-versus-static check (K89, experiments/drift_online.py) was registered before the run and queued (#246), and #247 fixed the backlog classifier so an open item naming a committed experiment routes as worker work instead of stalling as 'no job type fits'. The drift task exists to give stateful and online families a task where they can show an advantage; it is the first task in the suite designed for them.

### 5.5 Research cadence continues

The worker kept landing literature cards through the nights: R75 through R96 appear in the commit log for 26 and 27 November alone, each with the standard pipeline signature, drafted by gpt-oss-120b, checked by Nemotron 3 Ultra, quotes verified by script, and each still marked 'not yet judged by Claude'. That label is the honest state of a card that has passed the automated gates but not the scarce reviewer.

## 6. Results of Record: What Has Actually Been Established

### 6.1 What holds

The machinery works and has been reviewed: the probe survived seven independent reviews; counter is 21 of 21 on CI; the loop is 23 of 24; the ledger's plan 1 was judged a pass by an outside verifier. The general task contract held on four tasks with no task code in the runner, including a SAT checker with no answer key. Cross-host training was made deterministic by pinning one OpenBLAS kernel type (Q55), and the jail was hardened to run every candidate under rlimits and seccomp, including CI's fallback mode (Q10). The fresh SWE-rebench line is evidence rather than anecdote: from the issue text alone, the free model named the file to change for 72 percent of 40 SWE-bench Verified tasks but only 35 percent of 40 fresh ones, a gap of 37.5 points with a 95 percent interval of 17 to 58 under a rule of 15. The fresh set's graders were sealed and validated on 32 of 32 known-answer controls before any model attempt, after a rework that cuts the network after a warm-up.

### 6.2 What does not hold: the negative results

The project's headline findings are deliberately negative. Rankings do not transfer across task kinds: the same five families on digits, next-character text and UCI Adult produced three accuracy flips over two family pairs and four instruction-count flips, with the rule fixed before running. The logic-gate network's cost lead is digits-only: 0.47 times mlp16's instructions per item on digits, 2.1 times on text and 1.3 times on Adult, at or near last on accuracy on both. The README's best-result section states plainly that it does not carry over.

Stronger baselines weakened the founding hypothesis. Gradient-boosted trees beat linear and mlp16 on text (0.392 versus 0.338, plus 10.4 standard errors) and on Adult (0.866 versus 0.859, plus 2.4), though not on digits, at three to eighteen times the instructions; WiSARD lookup tables were neither cheaper nor as accurate anywhere. The hypothesis that structures matching the hardware are cheapest (Q54) is weakened for every family of that kind tried so far. On the best task, mlp16 reaches 0.980 dev accuracy at one times the instructions and 104 ns per image at scale, while lgn_d reaches 0.973 at 0.45 times the instructions and 55 ns (31 with block reads).

The proposer's best results also corrected themselves under held-out data, a winner's curse caught by design: the dev best tied gradient-boosted trees on a held-out twenty percent of train (plus 0.0017, plus or minus 0.0046), while the top five proposals lost 0.005 more than the baselines did; three reached gbdt's accuracy within one standard error at a third or less of its instructions. The two-layer MLP's extra depth helped digits only, losing 0.006 on Adult and 0.050 on text, below linear. The P28 feasibility pilot was negative at the specified budget: two of eight attempts ended by themselves against a rule of at least six, and two of eight resolved, so the set is too hard as piloted, and the next direction (budget, model or task set) is explicitly the owner's call.

### 6.3 The proposer leaderboard

The one-call proposer has produced 1,097 proposals, 644 valid (a rate of 0.59, Wilson interval 0.56 to 0.62), 194 beating label-linear, and 11 on the accuracy-cost front of their task, at 1.7 free calls per valid proposal; the first twenty were kept (11 valid) under a rule fixed in the plan file. The leaderboard's ranking metric was itself settled by data (audit F054): whole-run instructions per item, startup included, because the marginal count can be gamed by moving work into the empty run. On the 165 proposals carrying both counts, the front is the same under either metric, with Spearman 0.9996 or more per task.

## 7. Health Assessment: Risks, Blockers and Weak Points

### 7.1 Open issues

Ten issues are open, and they cluster into three groups: queue throughput, sandbox direction, and security follow-ups. Table 3 triages them with the analyser's read.

**Table 3: Open issues at analysis time (27 November 2026)**

| Issue | Subject | Analyser's read |
| --- | --- | --- |
| #251 | Reduce parked jobs in the queue (forge-act) | Newest signal of the pile-up measured by #150 |
| #239 | graph_view: queue integration and visualisation | Backlog tooling; not load-bearing |
| #193 | Sandbox Stage A: 4 worker-claimable research items | Feeder for #191; already worker-shaped |
| #191 | Persistent sandboxes: E2B vs the Forge jail | A direction call, owner-level, feeding #111 |
| #150 | Worker idle: 0 of 75 open items claimable | The measured bottleneck: inputs, not workers |
| #113 / #1 | Weekly indicator snapshots (24 and 18 Nov) | Routine recurring audits |
| #111 | Harden loop and runner fallback (bug, security) | Owner-gated; pairs with #191 |
| #109 | Workflow integrity and reservation retry gaps (security) | Partially blocked on the workflow-file 403 |
| #103 | Test compositions vs strongest constituent | Queued science item, unclaimed |

### 7.2 Owner-gated work

The most structural blocker is not technical. The repository's own change-landing tool is forbidden to write .github/workflows (a 403), so four review findings on workflow files (F006, F033, F011, F009) have been waiting since 23 November for the owner's hand. The P28 direction call has been waiting since 25 November. The AGENTS.md rule that nothing goes to the owner that data can settle is well designed, but the residue that must reach the owner (credentials, repository settings, guard merges, holds) is arriving faster than it is being drained, and issues #150 and #251 both measure the pile.

### 7.3 Complexity and the self-knowledge promise

The project's stated failure modes are lying, self-growing complexity and compounding errors, and the third is the live one: 20 top-level modules, 52 workflows, twelve Forge components, twenty-plus branches, and a queue state of 1.1 MB. The weekly indicators exist precisely for this; the baseline measured 4.3 percent duplicated Python function lines and 31 workflow lines that swallow failures. But the newest surface, the entire fabric/ control plane, postdates that baseline and has not yet had an indicator pass or a dedicated adversarial review round of the kind the measuring machinery received in its first two weeks.

### 7.4 The worker funnel

The worker's own measurements are the best health data in the repository. Acceptance passed at 26 of 30 in release test 2 while running unattended failed and was fixed. The deeper number is P38's: of 73 small code items, 49 reached a pull request whose tests passed, but only 9 were done; 28 were rejected after passing because the model's own tests passed without doing the item, and 9 waited on a person. Q85 located the research funnel's loss at the last check rather than at search: the free claim decider, which has no known-answer controls, dropped verified claims in 9 of 18 empty runs. The system's own diagnosis, that the workers are not the bottleneck but their inputs are, matches the numbers, and both fixes (a known-answer test fixed before the code, and a labelled replay to set the claim decider's bar) are queued rather than speculative.

**Figure 2: The worker code-item funnel, the repository's own P38 measurement (25 November). Passing tests and being done are different things.** (chart rendered in the companion document; measured values:)

| Funnel stage | Items (queue P38 measurement, 2026-11-25) |
| --- | --- |
| Small code items in window | 73 |
| Reached a PR whose tests passed | 49 |
| Rejected after passing tests | 28 |
| Waited on a person | 9 |
| Done | 9 |

### 7.5 Incident pattern in the newest layer

Analyser's observation: in three days the Discord path produced three observed failure modes, runs ending on tool calls with no text, the browser-side loop dying when the phone locks, and a gateway socket going quietly dead while the object believed it alive, each fixed within a day, in public, with the root cause in the commit message. That honesty is a strength. The incident rate itself says the layer is still settling, and it is the layer with the least adversarial review coverage.

## 8. Critical Assessment (the Analyser's Judgement)

Five strengths are load-bearing:

1. Epistemic discipline is institutionalised, not aspirational: pre-registration, verdicts by non-claimants, tiered meters, evidence pointers on every number, and a public face that leads with what it cannot prove.
2. Adversarial review is measured and tuned: reviews of ideas changed decisions (8 of 11 points accepted at least in part), while code reviews mostly found what tests had already caught (3 of 15 held), so scarce review is aimed where it pays.
3. The labour economy is data-driven end to end: the who-does-what table carries evidence columns, paid pools sit behind caps, and grading was moved to scripts after models failed known-answer controls.
4. The security posture is unusually mature for a solo project: jails with seccomp and rlimits, fail-closed probes, Ed25519 with a replay window, redaction, tool output framed as data-not-instructions, spend caps and owner-only routing.
5. Negative results are first-class: the transfer study, the winner's-curve correction on proposals and the P28 pilot failure are all recorded with rules fixed in advance.

Six weaknesses, equally plainly:

1. Bus factor is one, and the owner gate is accumulating: workflow-file fixes blocked by tool permissions, the P28 direction call, and two security follow-ups all wait on the same person.
2. The control-plane sprint has outrun the review machinery that earned the rest of the repository its credibility; fabric/ has findings folded in piecemeal but no dedicated adversarial round.
3. The worker's oracle measures the wrong thing for done: 28 of 73 items passed tests without doing the work, so the queued known-answer-first fix is the correct remedy, not more review.
4. queue.json at 1.1 MB is state, claim-lock and merge surface in a single artifact; both #150 and #251 point at it.
5. The ledger covers 4 of roughly 250 items, so the trust promise is ahead of its use; results of record live mostly as runs and files.
6. The interface grew four overlapping entry paths (webhook personas, slash commands, gateway listener, workbench UI) whose maintenance cost is not yet measured.

Net judgement: the project is doing the hardest thing right, building the trust machinery before the scale, and its measurement layer is genuinely sound. The risk is not that the machinery lies; it is that the pace of interface-building consumes the review capacity that keeps the machinery honest. The remedy already exists in the repository's own playbook: extend the review rounds and the weekly indicators to fabric/ before it grows another story.

## 9. Conclusions and Recommendations

exagium at 23 days is a working adversarial measurement lab with an autonomous workforce and a new control plane, whose results of record are mostly negative and mostly honest. The immediate list, ordered by how much measured work each item unblocks:

1. Land the four workflow-file findings by hand (F006, F033, F011, F009). Owner-only; closes the largest part of #109's workflow integrity gap and the oldest waiting items in the repo.
2. Make the P28 direction call (budget, model or task set), or defer it explicitly and record why. Owner; the bench line is parked until then, and a recorded deferral at least lets the queue item park honestly.
3. Move the queue state out of the single 1.1 MB JSON (a store-backed or lane-sharded layout). Owner and worker; both #150 and #251 point at this artifact.
4. Ship the two queued input-quality fixes first (known-answer tests fixed before code; labelled replay for the claim decider). Worker; they attack the measured 28-of-73 and 9-of-18 loss points directly.
5. Run a dedicated adversarial review round over fabric/ (agent, act, gateway, record). Claude session; the measuring machinery had four such rounds before being trusted, and the control plane now carries real authority.
6. Register the next verdicts (K89 drift-online, the next proposer checkpoint) as ledger entries. Worker and record; starts closing the gap between results of record and the anchored record.
7. Extend the weekly indicators to the Forge (workflow count, branch count, queue size, fabric code volume). Worker; makes self-growing complexity visible exactly where it is currently growing fastest.

None of this requires new invention; every remedy is the repository's own pattern applied to its newest parts. The three weeks of recorded evidence say the pattern works.

## Appendix A: Module Map

Twenty top-level module directories plus .github; the four top-level documents (README, AGENTS, VERIFY and the gitignore) are covered in the body. The state branch pilot/lgn-20261115 carries the queue, findings and notes; the code lives on main.

**Table 4: Top-level modules and their roles**

| Module | Role |
| --- | --- |
| agent/ | The /oc issue agent: model aliases, paid switch, output scrubbing |
| audit/ | Periodic whole-system indicators, adversarial review rounds, fabric rules, findings register |
| counter/ | Deterministic instruction and cache-miss meter (valgrind sandbox, cross-host CI) |
| experiments/ | Pre-registered checks: drift_online, packet_pilot, retrieval_bm25, claim_roots, queue_drift |
| fabric/ | The Forge: twelve Cloudflare components (act, agent, box, cfbox, chat, dash, discord, mcp, record, sandbox, workbench, worker) |
| facts/ | Measured-fact register with a checker (check.py, register.json) |
| ledger/ | Anchored append-only entries (four so far) |
| loop/ | Jailed retraining and scoring; per-image meter scale.py; confine.c fallback jail |
| probe/ | Machine self-measurement; seven independent reviews; VERIFY.md target |
| record/ | Ledger writer and verifier, anchors, monitor |
| research/cards/ | Sixty-seven literature cards, R-series to R96, pipeline-made |
| router/ | Free-model pools and agent benches 2 to 6 |
| runs/ | Raw meter records (counter runs, probe profiles) |
| runner/ + tasks/ | The general task contract; five tasks (optdigits, text, adult, sat, drift) |
| store/gnomon-trial/ | A trial knowledge store: questions, scripted answers |
| tests/ | Workflow test harness and the worker injection canaries |
| transfer/ | Cross-task ranking replication (the negative result) |
| verifications/ | Dated verification records (two runs so far) |
| worker/ | The hourly free queue worker: jobs, transitions, tiers, budget, digest |

## Appendix B: Key Numbers

**Table 5: Key numbers, verified 27 November 2026**

| Measure | Value |
| --- | --- |
| Age (created 4 November 2026) | 23 days |
| Pull requests (all-time) | 241, latest #250 on 27 November |
| Open issues | 10 of roughly 250 numbered items |
| CI workflow files | 52 |
| Research cards on main | 67 (R-series to R96 plus two Q cards) |
| Ledger entries | 4 (plan 1 pass; plan 2 inconclusive) |
| Queue state file | 1.1 MB (audit/2026-11-17/queue.json, pilot branch) |
| Top-level modules / Forge components | 20 / 12 |
| Proposer totals | 1,097 proposals; 644 valid; 11 on the front; 1.7 calls per valid |
| Loop / counter CI record | 23 of 24 / 21 of 21 runs |
| Worker test suite | test_worker.py at about 110 KB (169 tests green in the latest run note) |
| forge-agent loop limits | 12 steps; 10-minute cap; 60,000-char history; 6,000-char tool output |
| Drift task | 8,000 rows; rule changes every 2,000; five pinned digests |
| Transfer headline | LGN at 0.47x / 2.1x / 1.3x mlp16 instructions (digits / text / adult) |
