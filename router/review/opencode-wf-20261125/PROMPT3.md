You are an independent security reviewer. An approved workflow (owner-only /oc issue agent; read-only model job with no
shell; a script-only post job) is being changed so that model keys come from repository secrets instead of one vendor
key. Attack the CHANGE. Be specific; under 350 words.

What changed:
- The answer job now has one step per provider (anthropic, google, openrouter, groq, nvidia, cerebras, xai, mistral,
  deepinfra, fireworks-ai, huggingface, plus opencode free/paid). Each step has `if: ... provider == '<p>'` and holds
  exactly one secret in its env under the variable opencode reads (e.g. ANTHROPIC_API_KEY: secrets.CLAUDE_API_KEY).
  Only the step matching the requested provider runs. Each runs `bash agent/run.sh`, which runs
  `opencode run --agent answer -m "$MODEL" -f prompt.md "..."` or `opencode models "$PROVIDER"`.
- agent/oc_pick.py (no model) sets provider from the requested id "provider/model..." and refuses unknown providers;
  model ids match ^[a-z0-9][a-z0-9-]*/[A-Za-z0-9][A-Za-z0-9._:/-]{0,120}$. Paid status comes from a committed
  models.json (anthropic, google, cerebras, xai, mistral, deepinfra, fireworks, huggingface paid; openrouter paid unless
  the id ends ':free'; groq, nvidia free); paid is refused unless allow_paid is true.
- A unit test fails if any step holds two secrets, a provider's step holds the wrong secret, or the post job holds one.
- Unchanged: read-only agent as system config (bash/edit/web/external_directory denied; measured), owner-only trigger,
  only owner text reaches the model, one run at a time repo-wide, pinned opencode via npm ci.

Answer:
NEW RISKS: what this change adds, concretely, or "none found" with reasoning.
MISCLASSIFIED: any way a paid call passes as free, or a key reaches the wrong provider or the model.
VERDICT: SHIP AS IS, SHIP WITH CHANGES (list), or DO NOT SHIP.
