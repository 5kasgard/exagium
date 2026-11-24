<!-- nvidia nvidia/nemotron-3-ultra-550b-a55b status=200 latency=84.64s -->

**NEW RISKS**
1. **OpenRouter `:free` suffix bypass**: The model ID regex `^[a-z0-9-]*/[A-Za-z0-9._:/-]+$` permits colons in the model segment. `oc_pick.py` treats any ID ending `:free` as free. If OpenRouter’s API accepts `provider/model:free` as a valid alias for a paid model (or strips the suffix), a paid model executes under free logic. The committed `models.json` cannot enumerate dynamic OpenRouter model aliases.
2. **Secret exposure via crash/debug logs**: Each step injects a raw secret into `env` (e.g., `ANTHROPIC_API_KEY`). `bash agent/run.sh` executes `opencode`. If `opencode` crashes, hits a panic, or honors a debug flag (e.g., `OPENCODE_DEBUG=1` set via compromised `prompt.md` or env), it may dump headers/env to stdout/GHA logs. The "read-only" denial applies to *tools*, not the process environment or stderr.
3. **Provider routing drift**: The step `if:` condition uses the workflow input `provider`. `oc_pick.py` derives provider from the `model` arg (`provider/model`). A user supplying `provider=anthropic` + `model=openrouter/foo` runs the Anthropic step (injecting `CLAUDE_API_KEY`) but `oc_pick.py` validates `openrouter` (paid, allowed). `opencode` receives OpenRouter model + Anthropic key → auth failure (safe), *but* logic divergence exists. If `opencode` ever supports a "universal" fallback key, cross-billing occurs.

**MISCLASSIFIED**
*   **OpenRouter `:free` classification** (see Risk 1). The regex and suffix check are insufficient proof of "free" status against a dynamic catalog.
*   **Paid status via `models.json` supply chain**: `models.json` is committed. A malicious commit (or compromised maintainer account) flipping `groq` to `paid: true` blocks free usage (DoS), but flipping `anthropic` to `paid: false` enables unbilled paid usage. Owner-only trigger mitigates push, but not a compromised `models.json` in a dependency if ever externalized.

**VERDICT: SHIP WITH CHANGES**
1.  **Block colons in model IDs**: Change regex to `^[a-z0-9-]*/[A-Za-z0-9._/-]+$` (remove `:`). Handle OpenRouter free tiers via explicit allowlist in `models.json` (e.g., `provider: "openrouter:free"`).
2.  **Harden `run.sh`**: Add `set +x; export -n OPENCODE_DEBUG` (or equivalent) at top. Ensure opencode config disables telemetry/debug logging by default.
3.  **Align routing**: Move provider derivation to a pre-job step outputting `provider`/`model`; make step `if:` consume that output. Eliminates user-controlled drift between step selection and validation.
