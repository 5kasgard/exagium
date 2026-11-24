<!-- groq openai/gpt-oss-120b status=200 latency=3.59s -->

**EXPLOITABLE**  
*None found.*  
All triggers require the comment author to be the repository **owner** (`author_association == 'OWNER' && user.login == repository_owner`). A non‑owner cannot start the `answer` job, cannot affect the `post` job, and cannot supply any input that reaches the model except via files already present in the checked‑out repository. The model’s agent configuration (`answer`) explicitly denies any shell, edit, web‑fetch, web‑search, external‑directory or write capabilities, leaving it with read‑only access to the repository files. Because the model never receives the `OPENCODE_API_KEY` (it is only set as an environment variable in the paid‑model step) and never runs code, there is no vector for a malicious commenter to make the workflow perform actions beyond reading the repo and posting a single comment.

**KEY**  
The `OPENCODE_API_KEY` is injected only as an environment variable for the paid‑model step:

```yaml
env:
  OPENCODE_API_KEY: ${{ secrets.OPENCODE_API_KEY }}
```

The key is **not** written to any file, not echoed to stdout/stderr, and not passed to later steps. The model never receives it (the agent configuration has `read` only, no `env` access). Consequently the key cannot leak to the model, to the answer artifact, or to the `post` job.

**SPEND**  
Run‑time limits (`timeout 900`) cap each inference call to 15 minutes, and the workflow aborts if the process exceeds this limit. The only way to spend money is to request a paid model that is listed in `models.json` with `allow_paid: true`. No additional throttling (per‑user, per‑run, or monetary ceiling) is enforced by the workflow itself; spend is limited only by the provider‑side quotas the owner set in the OpenCode console. If the owner enables many paid models or sets high per‑model quotas, a malicious owner could voluntarily drive up costs, but that is an owner‑level risk, not a workflow vulnerability.

**RESIDUAL**  
*Risk:* The owner could be socially engineered into posting many `/oc …` comments (or a malicious bot acting as the owner could be compromised), causing repeated paid‑model invocations and unexpected spend.  
*Cheapest control:* Add a **per‑run rate‑limit** (e.g., a step that stores the timestamp of the last successful `answer` run in a repository‑wide secret or artifact and aborts if a new request occurs within N minutes). This limits rapid repeat invocations without altering the core security model.

**VERDICT**  
**SHIP AS IS** – the workflow, as written, does not allow a non‑owner to elevate privileges, leak the API key, or cause unintended write actions. The only practical risk is owner‑initiated spend, which can be mitigated with a simple rate‑limit if desired.
