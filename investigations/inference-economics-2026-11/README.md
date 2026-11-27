# Inference economics investigation (evidence-led)

## Purpose
Investigate inference/API expenditure and computational inefficiency in Exagium. Treat this as an engineering investigation, not a request to indiscriminately minimise model quality or switch every task to the cheapest model.

## Source provenance
- `anthropic_console_usage.csv`: Anthropic Console usage export. Token counts by date/model/context window, including uncached input, 5-minute and 1-hour cache writes, cache reads, and output.
- `openrouter_usage_nonzero.csv`: OpenRouter usage export, retaining rows with non-zero reported `total_usage`. The supplied OpenRouter export also contained many zero-cost model/session rows; those were excluded from this cost-focused extract. Do not interpret this file as a complete record of all model activity.
- The exports are independent provider ledgers. Do not join rows by date/model and assume they represent the same request/session. OpenRouter `total_usage` currency/unit is not specified in the supplied extract; establish this from the source export/account before labelling it USD or AUD.
- Date range represented: 2026-11-20 through 2026-11-27 UTC.
- Preserve raw source values. Do not infer missing session IDs, request IDs, prices, or causal links.

## Initial observations (not yet root-cause findings)
1. OpenRouter reports a large concentration on 2026-11-25 for Claude Opus 5.5 (6.57274 usage units), compared with the other non-zero rows. Confirm the unit and whether this is cost, then trace requests if identifiers are available.
2. Anthropic Console shows very large cache activity on 2026-11-25:
   - Haiku: 403,113 cache-write tokens and 1,000,795 cache-read tokens in the >200k context-window row.
   - Opus: 906,476 cache-write tokens and 2,463,691 cache-read tokens in the <=200k row.
   - Sonnet: 119,319 cache-write tokens and 404,699 cache-read tokens.
   - These are token counts, not costs. Cache reads can be cheaper than uncached input but are not free. Verify model-specific pricing and cache TTL pricing from the actual provider tariff applicable on the usage date.
3. Haiku rows across 2026-11-20 to 2026-11-24 show substantial uncached input and output. Determine whether this reflects repeated full-history submission, distinct tasks, repository reads, or another usage pattern.
4. The OpenRouter export contains many zero-cost rows for models such as Nemotron, North Mini Code, ERNIE, Flint, Particle, Decider, etc. Those may represent free routes, no billable inference, zero-rounded usage, or failed/no-op requests. Do not assume which without tracing.
5. These two datasets cannot currently establish the user's reported ~$20 credit reduction. Reconcile billing periods, provider accounts, currencies, credits, taxes, and any omitted usage before making a total-spend claim.

## Investigation plan — inspect actual implementation
Search the current repository and all active branches for:
- Provider adapters and API clients: Anthropic, OpenRouter, model/provider registries, request construction, retry/fallback paths.
- Model routing and task selection: routing policies, capability metadata, model aliases, default models, escalation rules, fallback chains.
- Context assembly: conversation/history inclusion, repository/file ingestion, prompt templates, system/developer prompt duplication, tool-result retention, truncation, summarisation, context-window selection.
- Caching: Anthropic prompt-cache boundaries, cache-control placement, stable-prefix construction, cache invalidation, TTL selection, cache hit/miss observability.
- Agent orchestration: parent/child task creation, delegation, recursive fan-out, tool loops, repeated planning, retries, verification loops, queue re-enqueueing, duplicate jobs.
- Persistence and memory: whether completed work/context is reconstructed from scratch; whether durable task artefacts or summaries are reused.
- Usage telemetry and accounting: per-request input/output/cache tokens, provider-reported usage, model, route, task/job/agent IDs, retries, latency, status, cost, currency, rate-card version.
- Budget and scheduler controls: per-task and per-run budgets, concurrency caps, maximum recursion/depth, retry limits, cancellation, circuit breakers, provider spend ceilings.
- Tests and workflows: existing fixtures, cost tests, routing tests, usage accounting tests, CI and scheduled jobs.

## Questions to answer with evidence
A. Can each provider request be attributed to a task, agent, parent task, execution, and user-visible outcome?
B. Are token usage and cost recorded for every attempt, including failed requests, retries and cache operations?
C. Is context repeatedly resent where stable-prefix prompt caching or durable intermediate artefacts should be used?
D. Are cache boundaries stable and correctly placed? Are dynamic fields placed before static reusable content and therefore destroying cache hits?
E. Is model routing based on measured task performance and verified outcomes, or static assumptions/model labels?
F. Can the orchestrator recursively create more work than it can process or afford?
G. Are duplicate jobs, retries, tool loops, and verification loops bounded and observable?
H. Can work be paused or cancelled when budgets, marginal value, or safety limits are exceeded?
I. Are cost estimates based on dated provider pricing and actual cache/input/output token classes?
J. Can Exagium compare API inference with deterministic code, local CPU/GPU, batch inference, embeddings, retrieval, or other suitable compute?

## Required deliverables
1. Repository map of the relevant execution, routing, context, cache, queue, telemetry and budget code, with exact paths and symbols.
2. Trace at least one real or reproducible request path end-to-end; identify missing correlation IDs and telemetry.
3. Reconcile provider exports without falsely joining unrelated records; calculate totals only where units and billing scope are verified.
4. Ranked findings with evidence, severity, confidence, impact and reproduction steps. Separate confirmed defects from hypotheses.
5. Tests or small probes for suspected repeated-context, cache-boundary, retry/fan-out and cost-attribution issues.
6. A staged implementation proposal. Do not make broad changes or silently alter model quality/routing defaults before evidence and tests exist.

## Candidate subsystem design (proposal to evaluate, not pre-approved architecture)
- A context compiler that constructs task-scoped context from explicit dependencies and durable artefacts.
- A provider-neutral usage event schema with provider, model, request/attempt ID, task/parent/agent IDs, input/output/cache token classes, latency, status, estimated/actual cost, currency and pricing-version provenance.
- A dated rate-card registry that distinguishes uncached input, cache writes by TTL, cache reads, output, batch discounts and other provider-specific billing dimensions.
- A model/compute router that optimises verified outcome quality, latency and total cost under task-specific constraints.
- A budget controller with per-task/run/provider limits, recursion/fan-out/concurrency/retry bounds, cancellation and circuit breakers.
- Evaluation feedback that measures cost per verified useful outcome, context amplification and rework ratio.

## Safety and scope constraints
- Do not send secrets, API keys, private prompts, or sensitive repository contents into new providers during investigation.
- Do not execute live paid requests as probes without explicit budget limits and approval.
- Do not treat a cheaper model as equivalent unless task-specific evaluation demonstrates acceptable quality.
- Do not infer that cache reads are waste; compare them against uncached input and the intended cache reuse.
- Do not merge the two provider datasets into one ledger without request-level correlation and verified units.
