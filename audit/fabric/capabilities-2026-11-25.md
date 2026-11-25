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
