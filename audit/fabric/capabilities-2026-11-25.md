<!-- recorded_by: Claude Opus 5.5 / claude.ai chat · when: 2026-11-25 · kind: measured (models.dev registry, fetched 2026-11-25; raw: providers-models-2026-11-25.json) plus model-output (the brief and the reading of the numbers) -->
# What the Forge's keys reach (P40)

## Model providers (measured from models.dev; prices as listed there)

Totals: 987 model listings across 17 providers with keys in this repo. 4 keyed providers (LLM7, Inferbase, Nara,
CrowLLM) are not in the registry and need their own `/models` call.

| provider (secret) | models | listed free | tool calling | reasoning | file/image input | open weights | max context | beyond text |
|---|---|---|---|---|---|---|---|---|
| openrouter | 375 | 14 | 315 | 271 | 255 | 164 | 2,000,000 | audio, image, pdf, video in; audio, image, video out |
| opencode | 146 | 54 | 146 | 141 | 101 | 58 | 1,050,000 | audio, image, pdf, video in |
| huggingface | 120 | 0* | 108 | 85 | 49 | 120 | 1,048,576 | image, video in |
| deepinfra | 67 | 0 | 65 | 48 | 41 | 59 | 1,048,576 | audio, image, pdf, video in |
| nvidia | 52 | 52 | 25 | 19 | 35 | 48 | 1,048,576 | audio, image, video in and out |
| google (Gemini) | 36 | 2 | 25 | 28 | 32 | 2 | 1,048,576 | audio, image, pdf, video in; audio, image out |
| fireworks-ai | 35 | 0 | 35 | 35 | 26 | 30 | 1,048,576 | image in |
| cloudflare-workers-ai | 30 | 0* | 20 | 18 | 11 | 30 | 1,048,576 | image in |
| mistral | 26 | 0 | 23 | 12 | 21 | 23 | 1,048,576 | audio, image, pdf in; audio out |
| cohere | 19 | 5 | 14 | 4 | 3 | 19 | 256,000 | image in |
| anthropic | 18 | 0 | 18 | 18 | 18 | 0 | 1,000,000 | image, pdf in |
| scaleway | 17 | 0 | 13 | 10 | 9 | 17 | 262,144 | audio, image, video in |
| vultr | 15 | 0 | 14 | 15 | 10 | 15 | 1,048,576 | audio, image, video in |
| groq | 14 | 1* | 7 | 5 | 4 | 12 | 131,072 | audio, image in; audio out |
| xai | 11 | 0 | 6 | 6 | 11 | 0 | 1,000,000 | audio, image, pdf, video in; image, video out |
| sambanova | 4 | 0 | 4 | 4 | 2 | 4 | 1,048,576 | image, video in |
| cerebras | 2 | 0 | 2 | 2 | 1 | 2 | 131,072 | image in |

\* Listed at a price, but the provider has a free daily allowance (Cloudflare Workers AI neurons, Groq's free tier,
Hugging Face credits). models.dev records list prices, not free tiers; the router's own ledger and caps
(`router/engines.py`) are the measured truth for what is free in practice.

**Reading (model-output, to be checked):** the keys reach frontier text models; image, audio and video
understanding; image, audio and video generation (OpenRouter, NVIDIA, Google, xAI); million-token contexts; and
more than 100 free or free-tier models. The Forge's chat surfaces use about 6 of these. That gap is the "not much
right now".

## Not yet researched (brief for the next pass; each needs its docs read, not a model's memory)
- **Cloudflare beyond Workers:** Workers AI (models, embeddings, image, speech), Vectorize, D1, KV, R2, Durable
  Objects, Queues, Workflows, AI Gateway (one endpoint, caching, logging and rate limits across providers), Browser
  Rendering, Containers (paid), Access, Pages. Free-plan limits for each.
- **Hugging Face beyond inference:** Inference Providers, Spaces (ZeroGPU), datasets, the Hub API, jobs.
- **Compute:** E2B sandboxes, Vast (GPU rental), Vultr (VMs, GPUs), Scaleway: what each gives for training and
  distillation (the Forge's purpose), and at what price.
- **Data and state:** Upstash (Redis, QStash, Vector).
- **Research APIs:** OpenAlex, CORE, Semantic Scholar, OpenML, BenchLM, Artificial Analysis, YouTube Data.
- **Chat and workspace UIs** with MCP, any provider, memory, files and agents: LobeHub (Vercel + Postgres + S3/R2),
  Open WebUI, LibreChat, Cloudflare agents-starter. Measure them on running free on these accounts, using the Forge
  connector, and covering the owner's feature list (Kimi/ChatGPT/Claude/Z level).

Output for each: what it offers; free-tier limits; what in the Forge it would serve; and the cheapest test.

## Agent abilities: what a workbench must let agents do (owner 2026-11-25: replace other chats, work indefinitely, wired into the system)

<!-- kind: model-output (Claude's mapping of needs to assets; each row is a hypothesis with a cheapest test) -->

| Ability | Already in the Forge | Gap | Candidate (to test) |
|---|---|---|---|
| Chat with any model, any modality | keys to 987 listings; `/oc`; chat.exagium.xyz (thin) | a full UI: memory, files, artifacts, voice, branching | LobeHub on Vercel + Neon Postgres + R2 (free), or Open WebUI / LibreChat on a paid box |
| Read the Forge's state | connector.exagium.xyz/mcp (8 read tools, live) | none for reading | - |
| Act on the Forge | workers, tier and `queue_ops move` (in Actions only) | write tools for chats: propose a queue move or edit as a PR, dispatch a workflow, open an issue thread; all signed and gated | Forge MCP write tools that open PRs and dispatch workflows; nothing merges without the owner or the existing guards |
| Run code safely | the Actions jail (runner/, loop/) | interactive sandboxes from chat | E2B (key present; it has an MCP server), or Actions dispatch for long jobs |
| Browse and fetch the web | research_job (search APIs) | page fetch and render for agents | Cloudflare Browser Rendering (free-tier limits to check); Firecrawl (owner's connector, not keyed here) |
| Research the literature | OpenAlex, CORE, Semantic Scholar, OpenML keys; research cards | exposed to chats as tools | the same APIs behind Forge MCP tools |
| Remember across sessions and models | the repo fabric, event spine, handoffs | retrieval over everything | embeddings (Workers AI) + Vectorize or Upstash Vector, served through the connector |
| Files in and out | the repo | uploads, large artifacts | R2 (free 10 GB) |
| Long-running and scheduled agents | worker, tier, keeper, generator in Actions | started and watched from chat | dispatch tools + recent_events; later Cloudflare Workflows or Durable Objects |
| Agents talking to each other | issue threads; event log (P37b) | live threads | P36 plan: hot event store on a free tier (measured on 100 items), threads per item |
| Route and cap spend across providers | router/pick.py, ledger, engines.py PAID caps | one place for chat traffic too | Cloudflare AI Gateway (caching, logs, rate limits, one endpoint) |
| Train, distil, experiment (the Forge's purpose) | runner, loop, experiments in Actions (CPU) | GPUs | HF ZeroGPU Spaces (free quota), Vast or Vultr GPUs (paid, per hour) |

**Proposed shape (hypothesis):** one workbench UI (LobeHub first; measure it on this list) with the Forge's
own MCP servers as its hands: forge-state (read, live), forge-act (write via PRs and dispatch, gated), plus E2B for
code and research tools. Keys stay in one store (P33; the AI Gateway can front the model calls).

**Order:**
1. LobeHub deployed, with the read connector.
2. forge-act write tools, PR-only.
3. E2B code tool.
4. Retrieval memory.
5. AI Gateway.
6. GPUs when an experiment needs them.

Each step must pass a cheapest test before the next.
