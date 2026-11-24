<!-- groq openai/gpt-oss-120b status=200 latency=3.45s -->

**NEW RISKS**

1. **Secret‑spill via step‑level environment** – Each provider step now injects a single API key as an environment variable. The key is visible to **any process** executed in that step, including `agent/run.sh`. If the script or any downstream tool (e.g., `opencode`) logs its environment, the key can end up in the job log or an artifact and be harvested by a malicious PR author who can influence the *provider* input.  
2. **Conditional step abuse** – The `if: provider == '<p>'` guard is a simple string equality. A crafted provider value that contains a newline or other control characters could cause **multiple steps to evaluate true** (e.g., `provider="anthropic\ngoogle"`), leading to **multiple secrets being present in the same runner** and a single step potentially echoing the wrong key.  
3. **Paid‑vs‑free mis‑classification** – Paid‑status is derived from a static `models.json`. An attacker can request a paid model that is **listed as free** (e.g., a paid Anthropic model not yet reflected in the JSON) and bypass the `allow_paid` guard. OpenRouter’s rule “paid unless the id ends with `:free`” can be abused by appending `:free` to a paid model ID, causing a paid call to be treated as free.  
4. **Supply‑chain expansion** – Adding a step for every provider increases the attack surface of the workflow file itself. A malicious contributor with write‑access to the repo could insert a **dummy step** that re‑uses an existing secret name and prints it, or modify an existing step to copy the secret to an artifact, bypassing the unit‑test that only checks for *exactly one* secret per step.

**MISCLASSIFIED**

* Paid calls can be accepted as free when the static `models.json` is out‑of‑date or when a provider‑specific naming trick (`<model>:free`) is used.  
* A secret may be delivered to the wrong provider if the `if` expression evaluates true for more than one step (see newline injection) or if the workflow is edited to swap secret names.

**VERDICT**

**SHIP WITH CHANGES** – the change introduces concrete exposure vectors. Mitigations:

1. **Mask all provider keys**: use `secrets: inherit` at the job level but reference them only via `${{ secrets.<NAME> }}` inside the step; never expose them as plain environment variables.  
2. **Sanitize `provider` input**: enforce a strict regex (`^[a-z0-9]+$`) before the conditional matrix; reject any value containing whitespace or control characters.  
3. **Enforce single‑secret environment**: add a pre‑step that clears any previously set secret variables (`unset ANTHROPIC_API_KEY GOOGLE_API_KEY …`).  
4. **Dynamic paid‑status check**: query the provider’s pricing API or maintain a signed, version‑controlled list; reject any model not explicitly marked free in the signed list.  
5. **Audit logging**: ensure `agent/run.sh` does not `set -x` or otherwise echo the environment; add a step that verifies no secret appears in the log (`grep -q "$SECRET"`).  

Once these safeguards are in place, the workflow can be merged safely.
