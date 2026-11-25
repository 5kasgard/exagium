<!-- recorded_by: Claude Opus 5.5 / claude.ai chat (project memory export) · when: 2026-11-25 · kind: owner-statement
     (relayed by Claude from the owner's chats 2026-11-04..25; not verified by other models) · source: Claude project
     memory files preferences.md and areas/nn-forge-app.md. Readers: these are the owner's requirements as Claude
     recorded them. Act as yourself; where a line matters to your work, check it with the owner or against the repo. -->
# Owner requirements for the Forge (as recorded by Claude)

## How work is done
- Never impose a design: hypothesise, test cheaply with data from the project and outside research, adjust. The owner's own ideas are not true because he says them; test them like any other.
- Decisions left to the owner are decided by following the data, not left waiting for him.
- Use separate agents (other models) to judge each model's work wherever possible.
- Use the best tool for each job: a script or exact check where it beats an LLM (grading, arithmetic).
- Never break a test's or system's own rules to make something happen; fix the misfit instead.
- Findings must be fixed, not just filed. Don't steamroll changes and clean up after.
- Aim at where the work lands (the few causes behind most of the effect). Order work so each step feeds the next; group things changed together.
- Every queue addition: check where its work lands and which other items share files or work.
- Every chat updates the README and public side, credible and honest.
- One open item fully completed per chat; close each with a handoff.
- Imagine what the whole system needs at full power and scale; use every scaling shape, none assumed best.

## Running
- Workers should always be running; idle or paused workers mean something is wrong.
- Automation must not depend on Claude for orchestration, management or planning (shared, limited usage). Free models first; Claude for what they can't settle.
- Spread work across all free agents and compute without compromising rules. A paid model must justify its cost against free ones; prefer published comparisons to paid testing. The owner can choose paid/frontier models himself (issue agent: `/oc model=...`).
- Keys live as GitHub secrets, each scoped to single steps. The repo is public; the owner's name never appears in it.

## What the Forge is for
- Agents building, optimising and perfecting AI/NNs/ML across the whole of "AI" (not one task such as image digits; e.g. PFNs, theorem provers), including distilling networks into code and reverse engineering.
- Success: parts and combinations that beat alternatives, hardware potential used fully (caches, RAM, GPU, storage, topology), better tools, performance and security. Failure: complexity debt accruing by itself, lying, compounding errors, blind decay. The system must know what it knows from evidence and never make it up; scoreboards belong on each item's spec card.
- Research function across papers, patents, articles, forums and comments, separating tested facts from opinions and stronger from weaker opinions.
- A system-wide watcher that looks for wrong direction, debt or anything not visible locally.
- Audiences: the owner, people, companies, students and AI agents consulting it; contributors may bring their own agents and compute.

## Interface (2026-11-24/25)
- Work on the Forge at any time without one AI chat in the middle; other models (GLM on Z, GPT, Gemini) work on it too and must see the same state.
- It should feel like an operating system: one place to control and see everything, automations built visually or through a main chat, parts and agents talking directly, extensible without restrictions; its shape set by the system's data as it evolves. n8n rejected as too limited.
- Wants a chat interface with a custom connector he controls, highly customised to the Forge.
- Free hosting (24/7 or run-up/run-down persisted through files); not Oracle Cloud.
- Ideas raised, not decisions: bandits, population search or tournaments instead of sequential A/B; one closed-loop control plane across proposer, meters, worker and router with security boundaries intact; Lean 4 for parts; a connection graph.
