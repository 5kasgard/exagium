# Using the Forge from any chat (queue P36)

The Forge's state is in this repo: the queue, events, findings and notes on branch `pilot/lgn-20261115`, and the code on
`main`. The rules for reading and writing it are in `audit/fabric/README.md` on that branch. Three ways in, all on
your own keys, none tied to one AI vendor:

1. **The `/oc` issue agent (live).** Comment `/oc [model=<alias|provider/model>] <question>` on any GitHub issue.
   It works from a phone with no server. See `agent/models.json` for aliases and the paid switch.
2. **The custom connector: `fabric/mcp/forge_mcp.py`.** A read-only MCP server with tools `queue_summary`,
   `queue_item`, `queue_search`, `recent_events`, `findings`, `list_notes`, `read_note` and `refresh`. Any MCP
   client can use it. Tested in-process on 2026-11-25: every tool answered, and a `../../etc/passwd` read was
   refused.
3. **A chat UI you own**, with any model and this connector. Both candidates below are untested here: they are
   hypotheses to try, measured on whether they answer the same queue questions correctly (compare with
   `queue_summary`) and what they cost.

## Run the connector

```bash
git clone https://github.com/5kasgard/exagium && git clone -b pilot/lgn-20261115 https://github.com/5kasgard/exagium exagium-state
cd exagium && python3 -m venv .venv && .venv/bin/pip install -r fabric/mcp/requirements.txt
FORGE_STATE=../exagium-state .venv/bin/python fabric/mcp/forge_mcp.py streamable-http   # http://localhost:8765/mcp
```

## Option A: LibreChat (agents, MCP, any OpenAI-compatible provider; about 2 GB RAM for the full stack)

In `librechat.yaml`, add the connector and your providers. Keys go in LibreChat's `.env`, never in this repo:

```yaml
mcpServers:
  forge:
    type: streamable-http
    url: http://host.docker.internal:8765/mcp
endpoints:
  custom:
    - name: OpenRouter
      apiKey: "${OPENROUTER_API_KEY}"
      baseURL: https://openrouter.ai/api/v1
      models: { default: ["openai/gpt-oss-120b:free"], fetch: true }
    - name: Groq
      apiKey: "${GROQ_API_KEY}"
      baseURL: https://api.groq.com/openai/v1
      models: { default: ["openai/gpt-oss-120b"], fetch: true }
```

Anthropic and Google have built-in LibreChat endpoints (`ANTHROPIC_API_KEY` and `GOOGLE_KEY` in `.env`).

## Option B: Open WebUI (lighter; one container)

```bash
docker run -d -p 3000:8080 -v open-webui:/app/backend/data --name open-webui ghcr.io/open-webui/open-webui:main
```

Add providers as OpenAI-compatible connections (OpenRouter, Groq, NVIDIA and others) in Admin Settings, then add the
connector, `http://host.docker.internal:8765/mcp`, as a tool server. If your version only accepts OpenAPI tool
servers, wrap the connector with `mcpo`. Check the current Open WebUI docs before relying on either menu path.

## Where to run it

- Your own computer: free and always yours.
- A GitHub Codespace: start and stop it on demand, with personal free hours each month and state kept in the repo.
  Codespaces secrets are separate from the Actions secrets.

Both follow the owner's run-up/run-down preference.

## Next, if the data supports it

- Write tools (propose a queue move as a PR, dispatch a workflow) behind the same rules as the workers: signed events,
  no secrets in the connector, owner approval for anything that merges.
- The P33 key broker, so the chat UI and the workers draw on one key store.
